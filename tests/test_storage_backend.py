from __future__ import annotations

from collections import namedtuple

from vindr_mammo.storage import (
    estimate_export_space,
    format_bytes,
    get_disk_space,
    nearest_existing_ancestor,
)


def test_disk_space_uses_nearest_existing_ancestor_without_creating_output(tmp_path) -> None:
    existing = tmp_path / "mounted-output"
    existing.mkdir()
    requested = existing / "new-dataset" / "nested"
    calls = []
    Usage = namedtuple("Usage", "total used free")

    def fake_disk_usage(path):
        calls.append(path)
        return Usage(total=1_000_000, used=250_000, free=750_000)

    assert nearest_existing_ancestor(requested) == existing
    space = get_disk_space(requested, disk_usage_func=fake_disk_usage)

    assert calls == [existing]
    assert space.requested_path == requested
    assert space.probe_path == existing
    assert space.total_bytes == 1_000_000
    assert space.used_bytes == 250_000
    assert space.free_bytes == 750_000
    assert space.free_fraction == 0.75
    assert not requested.exists()


def test_format_bytes_uses_binary_units() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(3 * 1024**3, precision=2) == "3.00 GiB"
    assert format_bytes(-1536) == "-1.5 KiB"


def test_conservative_deterministic_estimate_includes_padded_edge_grid_and_pairs() -> None:
    config = {
        "export": {"save_square_crops": True, "save_baseline_uncropped": False},
        "square_crops": {
            "crop_size": 1024,
            "stride": 512,
            "edge_policy": "regular_stride_pad",
            "train_crop_mode": "deterministic",
            "val_crop_mode": "deterministic",
            "test_crop_mode": "deterministic",
        },
        "paired_whole_images": {
            "enabled": True,
            "storage_mode": "single_file_per_source",
            "target_width": 1024,
            "target_height": 1024,
        },
        "preserved_16bit": {"save": False},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 3.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }
    records = [
        {
            "image_id": "one",
            "export_split": "train",
            "width": 1500,
            "height": 1000,
            "num_masses": 1,
        }
    ]

    estimate = estimate_export_space(config, records)

    # Normal stride origins are x=[0,512], y=[0]. The last x window extends
    # beyond the source and is padded; no extra origin is emitted after that
    # crop already covers the far edge.
    assert estimate.source_image_count == 1
    assert estimate.crop_image_count == 2
    assert estimate.paired_whole_image_count == 1
    assert estimate.model_image_count == 3
    assert estimate.model_pixel_count == 3 * 1024 * 1024
    assert estimate.raw_estimated_bytes == 3 * 1024 * 1024 * 3
    assert estimate.conservative_bytes == estimate.raw_estimated_bytes


def test_whole_only_estimate_counts_every_resized_resolution() -> None:
    config = {
        "export": {"save_square_crops": False, "save_baseline_uncropped": False},
        "paired_whole_images": {
            "enabled": True,
            "save_original": True,
            "save_resized": True,
            "resized_variants": [
                {"name": "16x16", "width": 16, "height": 16},
                {"name": "8x8", "width": 8, "height": 8},
            ],
        },
        "preserved_16bit": {"save": False},
        "float32_export": {"enabled": False},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 1.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }

    estimate = estimate_export_space(
        config,
        [{"width": 16, "height": 32, "export_split": "train"}],
    )

    assert estimate.crop_image_count == 0
    assert estimate.paired_whole_image_count == 3
    assert estimate.model_image_count == 3
    assert estimate.model_pixel_count == 16 * 32 + 16 * 16 + 8 * 8
    assert estimate.conservative_bytes == estimate.model_pixel_count


def test_random_estimate_uses_annotations_and_one_to_one_target() -> None:
    config = {
        "export": {"save_square_crops": True, "save_baseline_uncropped": False},
        "square_crops": {
            "crop_size": 256,
            "stride": 128,
            "train_crop_mode": "random",
            "random_crops_per_annotation": 1,
            "train_positive_fraction": 0.5,
            "random_crops_per_negative_image": 1,
        },
        "preserved_16bit": {"save": False},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 1.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }
    records = [
        {
            "export_split": "train",
            "width": 1000,
            "height": 1000,
            "num_masses": 2,
        }
    ]

    estimate = estimate_export_space(config, records)

    assert estimate.crop_image_count == 4  # two positive + two negative
    assert estimate.model_pixel_count == 4 * 256 * 256
    assert estimate.conservative_bytes == 4 * 256 * 256


def test_aggregate_baseline_estimate_uses_target_canvas_and_preserved_copy() -> None:
    config = {
        "export": {"save_square_crops": False, "save_baseline_uncropped": True},
        "baseline_uncropped": {
            "resize_mode": "fit_pad",
            "target_width": 1024,
            "target_height": 1024,
        },
        "preserved_16bit": {"save": True},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 1.0,
            "preserved_16bit_bytes_per_pixel": 2.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }

    estimate = estimate_export_space(
        config,
        {"images": 2, "max_width": 3000, "max_height": 4000},
    )

    assert estimate.crop_image_count == 0
    assert estimate.baseline_image_count == 2
    assert estimate.model_pixel_count == 2 * 1024 * 1024
    assert estimate.breakdown_bytes["rgb_images"] == 2 * 1024 * 1024
    assert estimate.breakdown_bytes["preserved_16bit_images"] == 4 * 1024 * 1024
    assert estimate.conservative_bytes == 6 * 1024 * 1024
    assert any("aggregate" in assumption.casefold() for assumption in estimate.assumptions)


def test_native_paired_estimate_uses_fixed_common_canvas() -> None:
    config = {
        "export": {"save_square_crops": True, "save_baseline_uncropped": False},
        "square_crops": {
            "crop_size": 16,
            "stride": 16,
            "train_crop_mode": "random",
            "random_crops_per_annotation": 1,
            "random_crops_per_negative_image": 1,
        },
        "paired_whole_images": {
            "enabled": True,
            "save_native_resolution": True,
            "target_width": 16,
            "target_height": 16,
            "canvas_mode": "fixed",
            "canvas_width": 32,
            "canvas_height": 48,
        },
        "preserved_16bit": {"save": False},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 1.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }
    records = [{"export_split": "train", "width": 10, "height": 20, "num_masses": 1}]

    estimate = estimate_export_space(config, records)

    expected_crop = estimate.crop_image_count * 16 * 16
    expected_resized_whole = 16 * 16
    expected_native_whole = 32 * 48
    assert estimate.model_pixel_count == (
        expected_crop + expected_resized_whole + expected_native_whole
    )
    assert any("fixed 32 x 48" in item for item in estimate.assumptions)


def test_estimate_applies_explicit_positive_cohort_and_vendor_filters() -> None:
    config = {
        "source_cohort": {"positive_images_only": True},
        "vendor_filter": {"enabled": True, "include_vendors": ["Allowed"]},
        "export": {"save_square_crops": False, "save_baseline_uncropped": True},
        "baseline_uncropped": {
            "resize_mode": "fit_pad",
            "target_width": 10,
            "target_height": 10,
        },
        "preserved_16bit": {"save": False},
        "storage_estimate": {
            "rgb_bytes_per_pixel": 1.0,
            "metadata_bytes_per_sample": 0,
            "metadata_bytes_per_source": 0,
            "fixed_metadata_bytes": 0,
            "safety_factor": 1.0,
        },
    }
    records = [
        {"width": 100, "height": 100, "has_mass": True, "vendor": "Allowed"},
        {"width": 100, "height": 100, "has_mass": False, "vendor": "Allowed"},
        {"width": 100, "height": 100, "has_mass": True, "vendor": "Excluded"},
    ]

    estimate = estimate_export_space(config, records)

    assert estimate.source_image_count == 1
    assert estimate.baseline_image_count == 1
    assert estimate.conservative_bytes == 100
