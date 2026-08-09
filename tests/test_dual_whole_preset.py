from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import torch
from PIL import Image

from vindr_mammo.dash_app import (
    _build_export_cfg_from_params,
    _config_control_outputs,
    _config_control_values,
    _preview_read_max_side,
    _sample_view,
)
from vindr_mammo.crops import boxes_in_window
from vindr_mammo.export import (
    _compact_source_cadence,
    _crop_label_streaming_record_schedule,
    _crop_location_row,
    _empty_coco,
    _expected_completion_files,
    _make_crop_filename,
    _online_positive_ratio_selection_enabled,
    _paired_whole_geometry_metadata,
    _pad_rgb_to_canvas,
    _save_paired_whole_image_for_crop,
    _whole_image_filename_from_crop_filename,
    _write_dataset_readme,
    _write_reproducibility_bundle,
    _write_shared_export_files,
    _write_whole_image_annotation_indexes,
    _windows_for_export_split,
    export_whole_image_variants_only,
)
from vindr_mammo.presets import (
    DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    apply_study_preset,
)
from vindr_mammo.preprocessing import apply_geometry_preprocessing
from vindr_mammo.storage import estimate_export_space
from vindr_mammo.visualize import create_annotation_geometry_report


def _preset() -> dict:
    return apply_study_preset(
        {"paths": {"data_root": "/data/vindr", "output_root": "/exports/old"}},
        DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    )


def test_default_research_preset_matches_requested_pixel_and_geometry_contract() -> None:
    config = _preset()

    assert config["paths"]["output_root"] == "/exports/preprocessed-vindr-default-research-dataset-v2"
    assert config["study_preset_provenance"]["preset_version"] == 2
    assert config["image"]["normalize"] == "percentile"
    assert config["image"]["percentile_range"] == [0.5, 99.5]
    assert config["preprocess"]["crop_breast"] is True
    assert config["preprocess"]["mask_outside_breast"] is True
    assert config["preprocess"]["mirror_right_to_left"] is True
    assert config["preprocess"]["min_box_visibility_after_crop"] == 0.05
    assert config["crop_annotation_policy"]["allow_partial_annotations"] is True
    assert config["crop_annotation_policy"]["min_box_visibility"] == 0.05

    crop = config["square_crops"]
    assert (crop["crop_size"], crop["stride"]) == (1024, 128)
    assert crop["size_divisor"] == 16
    assert crop["edge_policy"] == "regular_stride_pad"
    for split in ("train", "val", "test"):
        assert crop[f"{split}_crop_mode"] == "deterministic"
    assert crop["train_deterministic_selection_mode"] == "crop_label_ratio"
    assert crop["train_deterministic_target_positive_ratio"] == 0.5
    assert crop["train_online_positive_ratio_selection_for_deterministic"] is True
    assert crop["train_balance_execution"] == "streaming_one_pass"
    assert crop["train_keep_all_positive_windows"] is True
    assert crop["val_deterministic_selection_mode"] == "all"
    assert crop["test_deterministic_selection_mode"] == "all"
    assert crop["preserve_positive_windows_below_min_breast_fraction"] is True
    for split in ("val", "test"):
        assert crop[f"{split}_require_min_breast_fraction_for_all_crops"] is True
        assert crop[f"{split}_min_breast_fraction_for_all_crops"] == 0.05

    channels = config["image_export"]["custom_channel_pipeline"]
    assert channels["R"] == channels["G"] == channels["B"]
    assert channels["R"]["steps"] == [{
        "op": "clahe",
        "apply_before_crop": True,
        "params": {"clip_limit": 2.0, "tile_grid_size": 8},
    }]
    assert config["paired_whole_images"]["enabled"] is True
    assert config["paired_whole_images"]["save_original"] is True
    assert config["paired_whole_images"]["save_resized"] is True
    assert config["paired_whole_images"]["save_high_resolution"] is False
    assert config["paired_whole_images"]["target_width"] == 1024
    assert config["paired_whole_images"]["target_height"] == 1024
    assert [
        (item["name"], item["width"], item["height"])
        for item in config["paired_whole_images"]["resized_variants"]
    ] == [
        ("1024x1024", 1024, 1024),
        ("640x640", 640, 640),
    ]
    assert config["dataset_layout"]["kind"] == "images_annotations_v1"
    assert config["lazy_crop_grids"] == [
        {"window_size": 1024, "stride": 128},
        {"window_size": 1024, "stride": 256},
        {"window_size": 1024, "stride": 512},
        {"window_size": 640, "stride": 160},
    ]
    assert config["paired_whole_images"]["resized_canvas_mode"] == "per_image_square"
    assert config["paired_whole_images"]["high_resolution_canvas_mode"] == "per_image_square"
    assert "high_resolution_canvas_width" not in config["paired_whole_images"]
    assert "high_resolution_canvas_height" not in config["paired_whole_images"]
    assert config["paired_whole_images"]["size_divisor"] == 16
    assert config["export"]["save_square_crops"] is False
    assert config["export"]["require_empty_output_root"] is True
    assert config["float32_export"]["variants"]["crops"] is False
    assert config["float32_export"]["variants"]["resized_whole"] is True
    assert config["dataset_review"]["enabled"] is False
    assert config["dataset_review"]["source_assets_per_split"] == 100
    assert config["annotation_geometry_report"] == {
        "enabled": False,
        "histogram_bins": 40,
        "output_subdir": "annotation_geometry",
        "fit_definition": "geometry_only_ignore_annotation_and_crop_locations",
    }
    assert config["runtime"]["simple_profiler_enabled"] is True
    assert config["whole_image_export_contract"]["expected_source_images"] == {
        "train": 13600,
        "val": 2400,
        "test": 4000,
    }
    assert config["whole_image_export_contract"]["expected_positive_sources"] == {
        "train": 743,
        "val": 151,
        "test": 219,
    }
    assert config["reproducibility_bundle"] == {
        "enabled": True,
        "output_subdir": "reproducibility",
        "schema_version": 1,
        "write_metadata_sha256": True,
        "include_software_source_snapshot": True,
        "include_source_dicom_sha256": False,
        "include_exported_image_sha256": False,
    }


def test_default_research_dinov3_inputs_are_divisible_by_patch_size() -> None:
    config = _preset()
    crop = config["square_crops"]
    paired = config["paired_whole_images"]
    assert paired["save_high_resolution"] is False
    assert crop["crop_size"] % 16 == 0
    assert crop["stride"] % 16 == 0
    assert paired["target_width"] % 16 == 0
    assert paired["target_height"] % 16 == 0


def test_default_research_preset_synchronizes_whole_variant_controls() -> None:
    config = _preset()
    values = _config_control_values(config)
    outputs = _config_control_outputs()
    assert len(values) == len(outputs)
    controls = {
        output.component_id: value
        for output, value in zip(outputs, values, strict=True)
    }
    assert controls["paired-whole-enabled"] == ["on"]
    assert controls["paired-whole-original"] == ["on"]
    assert controls["paired-whole-resized"] == ["on"]
    assert controls["paired-whole-high-resolution"] == []
    assert controls["paired-whole-common-canvas"] == []
    assert controls["paired-whole-size"] == 1024
    assert controls["paired-whole-canvas-width"] is None
    assert controls["paired-whole-canvas-height"] is None
    assert controls["export-review-enabled"] == []
    assert controls["annotation-report-enabled"] == []
    assert controls["annotation-report-bins"] == 40
    assert controls["reproducibility-enabled"] == ["on"]
    assert controls["reproducibility-checksums"] == ["on"]
    assert controls["view-geometry"] == "whole"
    assert controls["export-balance-mode"] == "crop_label_ratio"
    assert controls["export-target-positive-ratio"] == 0.5
    assert controls["require-foreground"] == []
    assert controls["min-foreground-fraction"] == 0.10
    assert controls["val-require-foreground"] == ["on"]
    assert controls["val-min-foreground-fraction"] == 0.05
    assert controls["test-require-foreground"] == ["on"]
    assert controls["test-min-foreground-fraction"] == 0.05
    assert controls["positivity-threshold"] == 0.05
    assert controls["min-box-visibility"] == 0.05


def test_simple_crop_pipeline_labels_mass_visible_at_five_percent() -> None:
    config = _preset()
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]], dtype=torch.float32)
    clipped, keep = boxes_in_window(
        boxes,
        (0, 0, 5, 100),
        config["crop_annotation_policy"],
    )
    assert keep.tolist() == [True]
    assert clipped.tolist() == [[0.0, 0.0, 5.0, 100.0]]


def test_research_breast_crop_expands_to_preserve_source_mass_boxes() -> None:
    image = torch.zeros((1, 100, 100), dtype=torch.float32)
    image[:, 20:80, 20:80] = 1.0
    image[:, 5:12, 4:10] = 0.5
    mass_box = torch.tensor([[4.0, 5.0, 10.0, 12.0]], dtype=torch.float32)
    result = apply_geometry_preprocessing(
        image,
        mass_boxes=mass_box,
        options={
            **_preset()["preprocess"],
            "crop_padding": 0,
            "crop_padding_fraction": 0.0,
            "minimum_padding_px": 0,
            "maximum_padding_px": 0,
            "mirror_right_to_left": False,
        },
    )

    assert result.mass_box_keep.tolist() == [True]
    assert result.mass_boxes.shape == (1, 4)
    assert result.info["breast_crop_expanded_to_preserve_mass_boxes"] is True
    assert result.info["breast_mask_expanded_to_preserve_mass_box_pixels"] is True
    crop_x0, crop_y0, _crop_x1, _crop_y1 = result.info["crop_box_xyxy"]
    assert crop_x0 <= 4
    assert crop_y0 <= 5
    x0, y0, x1, y1 = [int(value) for value in result.mass_boxes[0].tolist()]
    assert bool((result.image[:, y0:y1, x0:x1] > 0).all())


@pytest.mark.parametrize("split", ["val", "test"])
def test_research_eval_foreground_filter_is_loose_and_never_drops_positive_window(
    split: str,
) -> None:
    crop_cfg = {
        **_preset()["square_crops"],
        "crop_size": 10,
        "stride": 10,
    }
    image = torch.zeros((1, 10, 20), dtype=torch.float32)
    retained_breast_mask = np.zeros((10, 20), dtype=bool)
    retained_breast_mask[0, 0] = True  # 1% occupancy in the positive window.
    windows = _windows_for_export_split(
        split_name=split,
        image_width=20,
        image_height=10,
        image_tensor=image,
        mass_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        crop_options={
            "crop_size": 10,
            "allow_partial_annotations": True,
            "min_box_visibility": 0.05,
            "reject_partial_windows": False,
            "negative_max_box_visibility": 0.0,
        },
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(11),
        foreground_mask=retained_breast_mask,
    )

    # The 1%-tissue positive window is protected. The completely blank negative
    # window is rejected by the otherwise-active 5% evaluation filter.
    assert [window for window, _info in windows] == [(0, 0, 10, 10)]
    info = windows[0][1]
    assert info["is_positive_window"] == 1
    assert info["foreground_fraction"] == 0.01
    assert info["min_breast_fraction_for_all_crops"] == 0.05
    assert info["preserve_positive_windows_below_min_breast_fraction"] == 1


def test_annotation_geometry_report_ignores_location_and_reports_crop_fit(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        {"split": "train", "source_image_id": "a", "bbox_width_px": 100, "bbox_height_px": 200},
        {"split": "train", "source_image_id": "b", "bbox_width_px": 1024, "bbox_height_px": 1024},
        {"split": "val", "source_image_id": "c", "bbox_width_px": 1200, "bbox_height_px": 50},
        {"split": "test", "source_image_id": "d", "bbox_width_px": 50, "bbox_height_px": 1100},
        {"split": "test", "source_image_id": "e", "bbox_width_px": 1500, "bbox_height_px": 1300},
    ])
    result = create_annotation_geometry_report(
        rows,
        output_dir=tmp_path / "visualizations" / "annotation_geometry",
        crop_width=1024,
        crop_height=1024,
        histogram_bins=10,
    )

    assert result.summary["overall"]["total_mass_annotations"] == 5
    assert result.summary["overall"]["can_fit_fully_by_size"] == 2
    assert result.summary["overall"]["cannot_fit_fully_by_size"] == 3
    detail = pd.read_csv(result.output_dir / "mass_box_geometry.csv")
    assert detail["can_fit_fully_by_size"].tolist() == [True, True, False, False, False]
    assert detail["cannot_fit_reason"].tolist() == [
        "fits", "fits", "too_wide", "too_tall", "too_wide_and_too_tall"
    ]
    for filename in [
        "mass_box_fit_summary.csv",
        "mass_box_fit_summary.json",
        "mass_box_size_histograms.png",
        "mass_box_width_height_crop_fit.png",
        "mass_box_crop_fit_counts.png",
        "README.md",
        "index.html",
    ]:
        assert (result.output_dir / filename).exists()


def test_live_preview_contains_crop_and_both_whole_mammogram_views() -> None:
    full = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
    crop = full[:2, :2].copy()
    result = {
        "image": full,
        "crop_image": crop,
        "crop_mass_boxes": np.zeros((0, 4), dtype=np.float32),
        "mass_boxes": np.zeros((0, 4), dtype=np.float32),
        "selected_crop": {"window": (0, 0, 2, 2)},
        "target_summary": {},
        "record_index": 0,
        "title": "preview",
    }
    pipeline = {
        channel: {"source": "current_crop", "steps": []}
        for channel in ["R", "G", "B"]
    }
    component = _sample_view(
        result,
        pipeline,
        {
            "show_annotations": [],
            "visible_channels": ["R", "G", "B"],
            "show_channel_panels": [],
            "view_geometry": "crop",
            "paired_whole_enabled": ["on"],
            "paired_whole_original": ["on"],
            "paired_whole_resized": ["on"],
            "paired_whole_high_resolution": ["on"],
            "paired_whole_common_canvas": ["on"],
            "paired_whole_size": 8,
            "paired_whole_canvas_width": 16,
            "paired_whole_canvas_height": 16,
        },
        compact=True,
    )

    def component_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(component_text(item) for item in value)
        return component_text(getattr(value, "children", None))

    text = component_text(component)
    assert "Selected crop — 2 × 2" in text
    assert "Exact source-resolution window: (0, 0, 2, 2)" in text
    assert "Whole mammogram — original-size processed" in text
    assert "Whole mammogram — resized 8" in text
    assert "Whole mammogram — high-resolution padded" in text


def test_crop_preview_never_resizes_the_source_before_window_extraction() -> None:
    assert _preview_read_max_side({"view_geometry": "crop", "preview_max_side": 1024}) == 0
    assert _preview_read_max_side({"view_geometry": "whole", "preview_max_side": 1024}) == 1024


def test_dash_export_preserves_balanced_negative_breasts_and_debug_defaults() -> None:
    config = _preset()
    exported = _build_export_cfg_from_params(
        config,
        pd.DataFrame(),
        {
            "view_geometry": "crop",
            "pipeline_mode": "yaml",
            "pipeline": config["image_export"]["custom_channel_pipeline"],
            "export_parent": "/exports",
            "export_name": "default-research",
            "crop_size": 1024,
            "crop_stride": 512,
            "crop_edge_policy": "regular_stride_pad",
            "require_foreground": [],
            "min_foreground_fraction": 0.10,
            "val_require_foreground": ["on"],
            "val_min_foreground_fraction": 0.05,
            "test_require_foreground": ["on"],
            "test_min_foreground_fraction": 0.05,
            "fg_threshold_mode": "auto",
            "train_crop_mode": "deterministic",
            "val_crop_mode": "deterministic",
            "test_crop_mode": "deterministic",
            "export_balance_mode": "crop_label_ratio",
            "export_target_positive_ratio": 0.50,
            "paired_whole_enabled": ["on"],
            "paired_whole_high_resolution": ["on"],
            "paired_whole_common_canvas": ["on"],
            "paired_whole_size": 1024,
            "paired_whole_canvas_width": 3584,
            "paired_whole_canvas_height": 3584,
            "paired_whole_hardlink": ["on"],
            "annotation_report_enabled": ["on"],
            "annotation_report_bins": 40,
            "export_review_enabled": ["on"],
            "export_review_samples": 100,
            "export_review_max_side": 1200,
            "export_review_seed": 123,
            "export_review_crop_gifs": ["on"],
            "export_review_save_masks": ["on"],
            "export_review_mask_gifs": ["on"],
            "split_strategy": "random_study_fraction",
            "split_val_fraction": 0.15,
            "split_seed": 123,
            "split_stratify_birads": ["on"],
        },
    )

    square = exported["square_crops"]
    assert square["train_deterministic_selection_mode"] == "crop_label_ratio"
    assert square["train_online_positive_ratio_selection_for_deterministic"] is True
    assert square["train_require_min_breast_fraction_for_all_crops"] is False
    assert square["train_negative_require_foreground"] is True
    assert square["train_negative_min_foreground_fraction"] == 0.10
    assert square["val_deterministic_selection_mode"] == "all"
    assert square["test_deterministic_selection_mode"] == "all"
    assert square["val_require_min_breast_fraction_for_all_crops"] is True
    assert square["val_min_breast_fraction_for_all_crops"] == 0.05
    assert square["test_require_min_breast_fraction_for_all_crops"] is True
    assert square["test_min_breast_fraction_for_all_crops"] == 0.05
    assert square["preserve_positive_windows_below_min_breast_fraction"] is True
    assert exported["paired_whole_images"]["save_high_resolution"] is True
    assert exported["paired_whole_images"]["save_original"] is True
    assert exported["paired_whole_images"]["save_resized"] is True
    assert exported["paired_whole_images"]["resized_canvas_mode"] == "per_image_square"
    assert exported["paired_whole_images"]["high_resolution_canvas_mode"] == "fixed"
    assert exported["paired_whole_images"]["high_resolution_canvas_width"] == 3584
    assert exported["paired_whole_images"]["high_resolution_canvas_height"] == 3584
    assert exported["paired_whole_images"]["size_divisor"] == 16
    assert exported["paired_whole_images"]["storage_mode"] == "single_file_per_source"
    assert exported["dataset_review"]["enabled"] is True
    assert exported["dataset_review"]["save_masks"] is True
    assert exported["annotation_geometry_report"]["enabled"] is True
    assert exported["annotation_geometry_report"]["histogram_bins"] == 40
    assert exported["reproducibility_bundle"]["enabled"] is True
    assert exported["reproducibility_bundle"]["write_metadata_sha256"] is True
    assert square["train_balance_execution"] == "streaming_one_pass"


def test_streaming_execution_policy_overrides_stale_false_online_flag() -> None:
    crop = {
        "train_deterministic_selection_mode": "crop_label_ratio",
        "train_online_positive_ratio_selection_for_deterministic": False,
        "train_balance_execution": "streaming_one_pass",
    }
    assert _online_positive_ratio_selection_enabled(crop, "train", "deterministic") is True


def test_crop_label_streaming_schedule_starts_positive_and_bounds_negative_sources() -> None:
    records = [
        {"image_id": "positive-1", "study_id": "p1", "laterality": "L"},
        {"image_id": "paired-clean-view", "study_id": "p1", "laterality": "L"},
        {"image_id": "positive-2", "study_id": "p2", "laterality": "R"},
        *[
            {"image_id": f"negative-{index}", "study_id": f"n{index}", "laterality": "L"}
            for index in range(20)
        ],
    ]
    schedule, reserve, info = _crop_label_streaming_record_schedule(
        records=records,
        source_image_has_mass_lookup={"positive-1": 1, "positive-2": 1},
        source_breast_has_mass_lookup={("p1", "L"), ("p2", "R")},
        target_positive_ratio=0.5,
        rng=np.random.default_rng(123),
        shuffle=True,
    )

    scheduled_ids = [record["image_id"] for record in schedule]
    assert scheduled_ids[0].startswith("positive-")
    assert {"positive-1", "positive-2"}.issubset(scheduled_ids)
    assert "paired-clean-view" not in scheduled_ids
    assert info["positive_source_images"] == 2
    assert info["positive_source_cadence"] == 1
    assert info["negative_source_cadence"] == 1
    assert info["scheduled_negative_source_images"] == 2
    assert len(schedule) == 4
    assert scheduled_ids[0].startswith("positive-")
    assert scheduled_ids[1].startswith("negative-")
    assert scheduled_ids[2].startswith("positive-")
    assert scheduled_ids[3].startswith("negative-")
    assert info["unscheduled_negative_source_images"] == 18
    assert len(reserve) == 18


def test_crop_label_streaming_schedule_uses_four_positive_one_negative_for_eighty_twenty() -> None:
    records = [
        *[
            {"image_id": f"positive-{index}", "study_id": f"p{index}", "laterality": "L"}
            for index in range(8)
        ],
        *[
            {"image_id": f"negative-{index}", "study_id": f"n{index}", "laterality": "R"}
            for index in range(10)
        ],
    ]
    schedule, reserve, info = _crop_label_streaming_record_schedule(
        records=records,
        source_image_has_mass_lookup={f"positive-{index}": 1 for index in range(8)},
        source_breast_has_mass_lookup={(f"p{index}", "L") for index in range(8)},
        target_positive_ratio=0.8,
        rng=np.random.default_rng(123),
        shuffle=False,
    )

    scheduled_ids = [record["image_id"] for record in schedule]
    assert info["positive_source_cadence"] == 4
    assert info["negative_source_cadence"] == 1
    assert info["source_cadence_block_size"] == 5
    assert scheduled_ids[:5] == [
        "positive-0", "positive-1", "positive-2", "positive-3", "negative-0"
    ]
    assert scheduled_ids[5:] == [
        "positive-4", "positive-5", "positive-6", "positive-7", "negative-1"
    ]
    assert len(reserve) == 8


def test_compact_source_cadence_is_computed_from_arbitrary_ratios() -> None:
    assert _compact_source_cadence(0.50) == (1, 1)
    assert _compact_source_cadence(0.80) == (4, 1)
    assert _compact_source_cadence(0.70) == (3, 1)
    assert _compact_source_cadence(0.30) == (1, 3)
    assert _compact_source_cadence(1.00) == (1, 0)


def test_preset_completion_contract_includes_whole_validation_and_reproducibility_bundle(tmp_path: Path) -> None:
    expected = _expected_completion_files(tmp_path, _preset())
    assert tmp_path / "metadata" / "whole_image_manifest.csv" in expected
    assert tmp_path / "annotations" / "whole_image_annotations.csv" in expected
    assert tmp_path / "metadata" / "whole_image_validation.json" in expected
    assert tmp_path / "annotations" / "resized" / "640x640" / "coco" / "instances_test.json" in expected
    assert tmp_path / "visualizations" / "annotation_geometry" / "index.html" not in expected
    assert tmp_path / "reproducibility" / "source_images.csv" in expected
    assert tmp_path / "reproducibility" / "source_processing.csv" in expected
    assert tmp_path / "reproducibility" / "crops.csv" in expected
    assert tmp_path / "reproducibility" / "checksums.sha256" in expected
    assert tmp_path / "reproducibility" / "software_source_snapshot.zip" in expected


def test_reproducibility_bundle_records_exact_membership_crop_order_and_annotations(tmp_path: Path) -> None:
    data_root = tmp_path / "source"
    data_root.mkdir()
    for name in ["metadata.csv", "breast-level_annotations.csv", "finding_annotations.csv"]:
        (data_root / name).write_text("image_id\nimage-1\n", encoding="utf-8")
    dicom_path = data_root / "images" / "study-1" / "image-1.dicom"
    dicom_path.parent.mkdir(parents=True)
    dicom_path.write_bytes(b"test-dicom")

    output_root = tmp_path / "export"
    metadata_row = {
        "dataset": "square_crops",
        "split": "train",
        "file_name": "crop.png",
        "source_image_id": "image-1",
        "source_study_id": "study-1",
        "source_dicom_path": str(dicom_path),
        "training_image": "images/train/crop.png",
        "paired_whole_image": "whole_images/train/crop.png",
        "paired_whole_high_resolution_image": "whole_images_high_resolution/train/crop.png",
        "paired_whole_key": "image-1",
        "crop_info": {
            "window_xyxy": [512, 1024, 1536, 2048],
            "crop_mode": "deterministic",
            "is_positive_window": 1,
            "balance_execution": "streaming_one_pass_no_global_planning",
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 36,
            "pad_bottom": 48,
        },
        "preprocess_info": {
            "processed_shape": [2000, 1500],
            "original_shape": [2200, 1700],
            "crop_box_xyxy": [100, 200, 1600, 2200],
            "mirrored": False,
        },
        "encoding": {
            "paired_whole_pad_left": 0,
            "paired_whole_pad_top": 0,
            "paired_whole_canvas_width": 2000,
            "paired_whole_canvas_height": 2000,
            "paired_whole_scale_x": 0.512,
            "paired_whole_scale_y": 0.512,
            "paired_whole_high_resolution_pad_left": 0,
            "paired_whole_high_resolution_pad_top": 0,
            "paired_whole_high_resolution_canvas_width": 3584,
            "paired_whole_high_resolution_canvas_height": 3584,
        },
        "metadata_csv_rows": [],
        "dicom_meta": {},
        "export_boxes_xyxy": [[10.0, 20.0, 110.0, 220.0]],
    }
    coco = {split: _empty_coco() for split in ["train", "val", "test"]}
    coco["train"]["images"].append({
        "id": 1,
        "file_name": "crop.png",
        "source_image_id": "image-1",
        "source_study_id": "study-1",
    })
    coco["train"]["annotations"].append({
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "source_annotation_id": "mass-7",
        "source_bbox_xyxy": [522.0, 1044.0, 622.0, 1244.0],
        "source_bbox_original_xyxy": [622.0, 1244.0, 722.0, 1444.0],
    })
    stats_row = {
        "dataset": "square_crops",
        "split": "train",
        "file_name": "crop.png",
        "source_image_id": "image-1",
        "num_mass_boxes": 1,
        "has_mass": 1,
        "mean_mass_area_percentage": 1.0,
        "max_mass_area_percentage": 1.0,
        "balance_execution": "streaming_one_pass_no_global_planning",
    }
    _write_shared_export_files(
        output_root / "square_crops",
        coco,
        [stats_row],
        [metadata_row],
        dataset_kind="square_crops",
    )

    record = {
        "study_id": "study-1",
        "image_id": "image-1",
        "split": "training",
        "laterality": "L",
        "view_position": "CC",
        "dicom_path": str(dicom_path),
    }
    dataset = SimpleNamespace(
        findings_by_image_id={"image-1": [{"finding_categories": "Mass"}]},
        _filter_mass_findings=lambda findings: findings,
    )
    config = _preset()
    summary, created = _write_reproducibility_bundle(
        output_root=output_root,
        data_root=data_root,
        dataset=dataset,
        split_records={"train": [record], "val": [], "test": []},
        config=config,
    )

    bundle = output_root / "reproducibility"
    assert summary["source_image_count"] == 1
    assert summary["saved_crop_count"] == 1
    assert summary["saved_annotation_count"] == 1
    assert bundle / "checksums.sha256" in created
    source = pd.read_csv(bundle / "source_images.csv")
    assert source.loc[0, "source_dicom_relative_path"] == "images/study-1/image-1.dicom"
    crops = pd.read_csv(bundle / "crops.csv")
    assert crops.loc[0, "crop_export_order"] == 0
    assert crops.loc[0, "crop_x0"] == 512
    assert crops.loc[0, "original_crop_x0"] == 612
    assert crops.loc[0, "has_mass"] == 1
    annotations = pd.read_csv(bundle / "crop_annotations.csv")
    assert annotations.loc[0, "source_annotation_id"] == "mass-7"
    assert annotations.loc[0, "crop_bbox_width"] == 100.0
    checksums = (bundle / "checksums.sha256").read_text(encoding="utf-8")
    assert "source_images.csv" in checksums
    assert "resolved_config.yaml" in checksums
    assert (bundle / "software_source_files.csv").is_file()
    assert (bundle / "software_source_snapshot.zip").is_file()


def test_paired_whole_export_writes_resized_and_high_resolution_square_images(tmp_path: Path) -> None:
    source = torch.tensor([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]], dtype=torch.float32)
    paired_cfg = {
        "enabled": True,
        "save_high_resolution": True,
        "target_width": 4,
        "target_height": 4,
        "resized_canvas_mode": "per_image_square",
        "high_resolution_canvas_mode": "per_image_square",
        "pad_value": 0.0,
        "pad_anchor": "left_top",
        "storage_mode": "single_file_per_source",
    }
    config = {
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
    }

    cache: dict[tuple[str, ...], Path] = {}
    info = _save_paired_whole_image_for_crop(
        source_image=source,
        crop_root=tmp_path,
        split_name="train",
        filename="study-1__source-1__crop__train_0000_x0_y0_w4_h4.png",
        source_image_id="source-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache=cache,
    )
    repeated = _save_paired_whole_image_for_crop(
        source_image=source,
        crop_root=tmp_path,
        split_name="train",
        filename="study-1__source-1__crop__train_0001_x2_y0_w4_h4.png",
        source_image_id="source-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache=cache,
    )

    resized = Image.open(tmp_path / info["paired_whole_image_path"])
    high_resolution = Image.open(
        tmp_path / info["paired_whole_high_resolution_image_path"]
    )
    assert resized.size == (4, 4)
    assert high_resolution.size == (3, 3)
    assert info["paired_whole_high_resolution_padded_without_resize"] is True
    assert info["paired_whole_source_width"] == 3
    assert info["paired_whole_source_height"] == 2
    assert info["paired_whole_key"] == "study-1__source-1"
    assert Path(info["paired_whole_image_path"]).name == "study-1__source-1.png"
    assert repeated["paired_whole_image_path"] == info["paired_whole_image_path"]
    assert repeated["paired_whole_high_resolution_image_path"] == info["paired_whole_high_resolution_image_path"]
    assert repeated["paired_whole_write_status"] == "reused"
    assert repeated["paired_whole_high_resolution_write_status"] == "reused"
    assert len(list((tmp_path / "whole_images" / "train").glob("*.png"))) == 1
    assert len(list((tmp_path / "whole_images_high_resolution" / "train").glob("*.png"))) == 1


def test_grouped_layout_writes_multiple_resolutions_below_images_and_annotations(
    tmp_path: Path,
) -> None:
    config = {
        "dataset_layout": {"kind": "images_annotations_v1"},
        "export": {"save_empty_label_files": True},
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
        "float32_export": {
            "enabled": True,
            "variants": {"resized_whole": True, "original_whole": False},
        },
    }
    paired_cfg = {
        "enabled": True,
        "save_original": True,
        "save_resized": True,
        "save_high_resolution": False,
        "resized_variants": [
            {"name": "16x16", "width": 16, "height": 16},
            {"name": "8x8", "width": 8, "height": 8},
        ],
        "resized_canvas_mode": "per_image_square",
        "pad_value": 0.0,
        "pad_anchor": "left_top",
    }
    info = _save_paired_whole_image_for_crop(
        source_image=torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) / 23,
        source_boxes=torch.tensor([[1.0, 1.0, 4.0, 3.0]]),
        source_annotation_ids=["mass-1"],
        crop_root=tmp_path,
        split_name="train",
        filename="study__image__crop__train.png",
        source_image_id="image",
        source_study_id="study",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache={},
    )
    created = _write_whole_image_annotation_indexes(
        tmp_path,
        [{
            "split": "train",
            "source_image_id": "image",
            "source_study_id": "study",
            "encoding": info,
        }],
        config=config,
    )

    assert (tmp_path / "images" / "original" / "train" / "study__image.png").is_file()
    for resolution in ["16x16", "8x8"]:
        assert (tmp_path / "images" / "resized" / resolution / "train" / "study__image.png").is_file()
        assert (tmp_path / "images" / "float32" / "resized" / resolution / "train" / "study__image.pt").is_file()
        assert (tmp_path / "annotations" / "resized" / resolution / "yolo" / "train" / "study__image.txt").is_file()
        assert (tmp_path / "annotations" / "resized" / resolution / "json" / "train" / "study__image.json").is_file()
        assert (tmp_path / "annotations" / "resized" / resolution / "coco" / "instances_train.json").is_file()
    assert (tmp_path / "annotations" / "whole_image_annotations.csv") in created
    manifest = pd.read_csv(tmp_path / "metadata" / "whole_image_manifest.csv")
    assert set(manifest["variant"]) == {
        "original", "resized_16x16", "resized_8x8"
    }
    assert not (tmp_path / "square_crops").exists()
    assert not list(tmp_path.glob("whole_labels_*"))


def test_resized_whole_does_not_inherit_high_resolution_common_canvas(
    tmp_path: Path,
) -> None:
    source = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 1.0]]],
        dtype=torch.float32,
    )
    paired_cfg = {
        "enabled": True,
        "save_high_resolution": True,
        "target_width": 4,
        "target_height": 4,
        "resized_canvas_mode": "per_image_square",
        "high_resolution_canvas_mode": "fixed",
        "high_resolution_canvas_width": 8,
        "high_resolution_canvas_height": 8,
        "pad_value": 0.0,
        "pad_anchor": "left_top",
    }
    config = {
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
    }
    info = _save_paired_whole_image_for_crop(
        source_image=source,
        crop_root=tmp_path,
        split_name="train",
        filename="study-1__source-1__crop__train_0000_x0_y0_w4_h4.png",
        source_image_id="source-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache={},
    )

    resized = np.asarray(Image.open(tmp_path / info["paired_whole_image_path"]))
    high_resolution = np.asarray(
        Image.open(tmp_path / info["paired_whole_high_resolution_image_path"])
    )
    assert resized.shape == (4, 4, 3)
    assert high_resolution.shape == (8, 8, 3)
    # The 2x4 source first gets a 4x4 per-image square, so both source rows
    # remain at full width in the compact output. Resizing the 8x8 common
    # canvas would incorrectly shrink the source into only the top-left 2x1.
    assert np.all(resized[1, :, :] > 0)
    assert not np.any(resized[2:, :, :])
    assert np.all(high_resolution[:2, :4, :] > 0)
    assert not np.any(high_resolution[2:, :, :])
    assert not np.any(high_resolution[:, 4:, :])


def test_every_whole_variant_gets_matched_annotations(tmp_path: Path) -> None:
    source = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 1.0]]],
        dtype=torch.float32,
    )
    paired_cfg = {
        "enabled": True,
        "save_original": True,
        "save_resized": True,
        "save_high_resolution": True,
        "target_width": 8,
        "target_height": 8,
        "resized_canvas_mode": "per_image_square",
        "high_resolution_canvas_mode": "fixed",
        "high_resolution_canvas_width": 16,
        "high_resolution_canvas_height": 16,
        "size_divisor": 1,
        "pad_value": 0.0,
        "pad_anchor": "left_top",
    }
    config = {
        "export": {"save_empty_label_files": True},
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
    }
    info = _save_paired_whole_image_for_crop(
        source_image=source,
        source_boxes=torch.tensor([[1.0, 0.0, 3.0, 2.0]]),
        source_annotation_ids=["mass-1"],
        source_annotation_rows=[7],
        crop_root=tmp_path,
        split_name="train",
        filename="study-1__source-1__crop__train_0000_x0_y0_w4_h4.png",
        source_image_id="source-1",
        source_study_id="study-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache={},
    )

    assert Image.open(tmp_path / info["paired_whole_original_image_path"]).size == (4, 2)
    assert Image.open(tmp_path / info["paired_whole_image_path"]).size == (8, 8)
    assert Image.open(tmp_path / info["paired_whole_high_resolution_image_path"]).size == (16, 16)
    assert info["paired_whole_original_annotations"][0]["bbox_xyxy"] == [1.0, 0.0, 3.0, 2.0]
    assert info["paired_whole_annotations"][0]["bbox_xyxy"] == [2.0, 0.0, 6.0, 4.0]
    assert info["paired_whole_high_resolution_annotations"][0]["bbox_xyxy"] == [1.0, 0.0, 3.0, 2.0]

    for key in [
        "paired_whole_original_label_path",
        "paired_whole_label_path",
        "paired_whole_high_resolution_label_path",
        "paired_whole_original_annotation_path",
        "paired_whole_annotation_path",
        "paired_whole_high_resolution_annotation_path",
    ]:
        assert (tmp_path / info[key]).is_file()
    original_yolo = (tmp_path / info["paired_whole_original_label_path"]).read_text()
    resized_yolo = (tmp_path / info["paired_whole_label_path"]).read_text()
    high_yolo = (tmp_path / info["paired_whole_high_resolution_label_path"]).read_text()
    assert original_yolo.strip() == "0 0.50000000 0.50000000 0.50000000 1.00000000"
    assert resized_yolo.strip() == "0 0.50000000 0.25000000 0.50000000 0.50000000"
    assert high_yolo.strip() == "0 0.12500000 0.06250000 0.12500000 0.12500000"

    metadata_rows = [{
        "split": "train",
        "source_image_id": "source-1",
        "source_study_id": "study-1",
        "encoding": info,
    }]
    created = _write_whole_image_annotation_indexes(tmp_path, metadata_rows)
    assert tmp_path / "metadata" / "whole_image_manifest.csv" in created
    for variant in ["original", "resized", "high_resolution"]:
        coco_path = (
            tmp_path
            / "mmdetection"
            / f"whole_{variant}"
            / "annotations"
            / "instances_train.json"
        )
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["source_annotation_id"] == "mass-1"


def test_default_research_writes_original_and_resized_annotations_without_high_resolution(
    tmp_path: Path,
) -> None:
    paired_cfg = dict(_preset()["paired_whole_images"])
    config = {
        "export": {"save_empty_label_files": True},
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
    }
    info = _save_paired_whole_image_for_crop(
        source_image=torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 1.0]]],
            dtype=torch.float32,
        ),
        source_boxes=torch.tensor([[1.0, 0.0, 3.0, 2.0]]),
        source_annotation_ids=["mass-1"],
        source_annotation_rows=[7],
        crop_root=tmp_path,
        split_name="train",
        filename="study-1__source-1__crop__train_0000_x0_y0_w1024_h1024.png",
        source_image_id="source-1",
        source_study_id="study-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache={},
    )

    assert Image.open(tmp_path / info["paired_whole_original_image_path"]).size == (4, 2)
    assert Image.open(tmp_path / info["paired_whole_image_path"]).size == (1024, 1024)
    assert info["paired_whole_original_annotations"][0]["bbox_xyxy"] == [
        1.0, 0.0, 3.0, 2.0
    ]
    assert info["paired_whole_annotations"][0]["bbox_xyxy"] == [
        256.0, 0.0, 768.0, 512.0
    ]
    assert "paired_whole_high_resolution_image_path" not in info
    assert not (tmp_path / "whole_images_high_resolution").exists()
    assert not (tmp_path / "whole_labels_high_resolution").exists()
    assert not (tmp_path / "whole_annotations_high_resolution").exists()

    created = _write_whole_image_annotation_indexes(
        tmp_path,
        [{
            "split": "train",
            "source_image_id": "source-1",
            "source_study_id": "study-1",
            "encoding": info,
        }],
    )
    assert tmp_path / "metadata" / "whole_image_annotations.csv" in created
    for variant in ["original", "resized"]:
        coco_path = (
            tmp_path
            / "mmdetection"
            / f"whole_{variant}"
            / "annotations"
            / "instances_train.json"
        )
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
    assert not (tmp_path / "mmdetection" / "whole_high_resolution").exists()


def test_whole_variants_can_export_when_crop_checkbox_is_off(tmp_path: Path) -> None:
    class FakeDataset:
        def _read_preprocessed_record_no_square(self, record):
            image = torch.tensor(
                [[[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 1.0]]],
                dtype=torch.float32,
            )
            target = {
                "mass": {
                    "boxes": torch.tensor([[1.0, 0.0, 3.0, 2.0]]),
                    "findings": [{
                        "source_annotation_id": "mass-1",
                        "source_annotation_row": 2,
                    }],
                },
                "_foreground_mask": np.ones((2, 4), dtype=bool),
                "preprocessing": {
                    "processed_shape": [2, 4],
                    "original_shape": [2, 4],
                    "mirrored": False,
                },
            }
            return image, target

    config = {
        "export": {
            "save_square_crops": False,
            "save_empty_label_files": True,
        },
        "paired_whole_images": {
            "enabled": True,
            "save_original": True,
            "save_resized": True,
            "save_high_resolution": True,
            "target_width": 8,
            "target_height": 8,
            "resized_canvas_mode": "per_image_square",
            "high_resolution_canvas_mode": "fixed",
            "high_resolution_canvas_width": 16,
            "high_resolution_canvas_height": 16,
            "pad_value": 0.0,
            "pad_anchor": "left_top",
        },
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {"source": "current_crop", "steps": []}
                for channel in ["R", "G", "B"]
            },
        },
        "histogram_equalization": {"enabled": False},
        "float32_export": {
            "enabled": True,
            "variants": {
                "crops": False,
                "original_whole": False,
                "resized_whole": True,
                "high_resolution_whole": False,
            },
        },
        "whole_image_export_contract": {
            "enabled": True,
            "strict": True,
            "expected_variants": ["original", "resized", "high_resolution"],
            "float32_required_variants": ["resized"],
            "expected_source_images": {"train": 1, "val": 0, "test": 0},
            "expected_positive_sources": {"train": 1, "val": 0, "test": 0},
            "expected_mass_annotations": {"train": 1, "val": 0, "test": 0},
        },
        "dataset_review": {"enabled": False},
        "runtime": {"show_progress": False},
    }
    summary, created = export_whole_image_variants_only(
        FakeDataset(),
        {
            "train": [{"image_id": "source-1", "study_id": "study-1"}],
            "val": [],
            "test": [],
        },
        config,
        tmp_path,
    )

    assert summary["num_source_images"] == 1
    assert summary["validation"]["status"] == "passed"
    assert summary["validation"]["source_images_without_model_inputs_added"] == 0
    assert not (tmp_path / "square_crops" / "images").exists()
    for directory in [
        "whole_images_original",
        "whole_images",
        "whole_images_high_resolution",
    ]:
        assert len(list((tmp_path / "square_crops" / directory / "train").glob("*.png"))) == 1
    assert (
        tmp_path
        / "square_crops"
        / "metadata"
        / "whole_image_annotations.csv"
    ).is_file()
    manifest = pd.read_csv(
        tmp_path / "square_crops" / "metadata" / "whole_image_manifest.csv"
    )
    resized_row = manifest[manifest["variant"] == "resized"].iloc[0]
    assert resized_row.float32_dtype == "float32"
    assert resized_row.float32_layout == "CHW"
    assert json.loads(resized_row.float32_shape) == [3, 8, 8]
    assert 0.0 <= resized_row.float32_min <= resized_row.float32_max <= 1.0
    tensor = torch.load(
        tmp_path / "square_crops" / resized_row.float32_path,
        map_location="cpu",
    )
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 8, 8)
    assert tensor.is_contiguous()
    assert all(path.exists() for path in created)


def test_fixed_whole_canvas_pads_only_bottom_and_right_to_a_common_shape() -> None:
    cfg = {
        "target_width": 4,
        "target_height": 4,
        "canvas_mode": "fixed",
        "canvas_width": 8,
        "canvas_height": 6,
        "size_divisor": 2,
        "pad_value": 0.0,
        "pad_anchor": "left_top",
    }
    small = np.full((2, 3, 3), 255, dtype=np.uint8)
    large = np.full((5, 7, 3), 127, dtype=np.uint8)

    small_canvas, small_meta = _pad_rgb_to_canvas(small, cfg)
    large_canvas, large_meta = _pad_rgb_to_canvas(large, cfg)

    assert small_canvas.shape == large_canvas.shape == (6, 8, 3)
    assert np.all(small_canvas[:2, :3] == 255)
    assert not np.any(small_canvas[2:, :])
    assert not np.any(small_canvas[:, 3:])
    assert small_meta["paired_whole_pad_left"] == 0
    assert small_meta["paired_whole_pad_top"] == 0
    assert small_meta["paired_whole_pad_right"] == 5
    assert small_meta["paired_whole_pad_bottom"] == 4
    assert large_meta["paired_whole_pad_right"] == 1
    assert large_meta["paired_whole_pad_bottom"] == 1
    assert small_meta["paired_whole_common_canvas"] is True
    assert small_meta["paired_whole_size_divisor"] == 2


def test_fixed_whole_canvas_rejects_shape_drift_and_non_divisible_dimensions() -> None:
    with pytest.raises(ValueError, match="smaller than a preprocessed mammogram"):
        _paired_whole_geometry_metadata(
            9,
            7,
            {
                "canvas_mode": "fixed",
                "canvas_width": 8,
                "canvas_height": 8,
                "size_divisor": 2,
            },
        )

    with pytest.raises(ValueError, match="divisible by size_divisor=16"):
        _paired_whole_geometry_metadata(
            5,
            5,
            {
                "canvas_mode": "fixed",
                "canvas_width": 30,
                "canvas_height": 32,
                "target_width": 16,
                "target_height": 16,
                "size_divisor": 16,
            },
        )


def test_crop_name_directly_resolves_shared_whole_name() -> None:
    crop_name = _make_crop_filename(
        {"study_id": "study-a", "image_id": "image-b"},
        "train",
        7,
        (512, 1024, 1536, 2048),
    )

    assert crop_name == (
        "study-a__image-b__crop__train_0007_x512_y1024_w1024_h1024.png"
    )
    assert _whole_image_filename_from_crop_filename(crop_name) == (
        "study-a__image-b.png"
    )


def test_crop_locations_csv_and_generated_readme_explain_coordinate_contract(tmp_path: Path) -> None:
    config = _preset()
    metadata_row = {
        "dataset": "square_crops",
        "split": "train",
        "file_name": "crop.png",
        "source_image_id": "image-1",
        "source_study_id": "study-1",
        "training_image": "images/train/crop.png",
        "paired_whole_image": "whole_images/train/crop.png",
        "paired_whole_high_resolution_image": "whole_images_high_resolution/train/crop.png",
        "paired_whole_key": "image-1",
        "crop_info": {
            "window_xyxy": [512, 1024, 1536, 2048],
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 36,
            "pad_bottom": 48,
        },
        "preprocess_info": {
            "processed_shape": [2000, 1500],
            "original_shape": [2000, 1500],
            "mirrored": False,
        },
        "encoding": {
            "paired_whole_pad_left": 0,
            "paired_whole_pad_top": 0,
            "paired_whole_canvas_width": 2000,
            "paired_whole_canvas_height": 2000,
            "paired_whole_scale_x": 0.512,
            "paired_whole_scale_y": 0.512,
            "paired_whole_high_resolution_pad_left": 0,
            "paired_whole_high_resolution_pad_top": 0,
            "paired_whole_high_resolution_canvas_width": 3584,
            "paired_whole_high_resolution_canvas_height": 3584,
        },
        "metadata_csv_rows": [],
        "dicom_meta": {},
        "export_boxes_xyxy": [],
    }
    location = _crop_location_row(metadata_row)
    assert location is not None
    assert location["crop_x0"] == 512
    assert location["paired_whole_key"] == "image-1"
    assert location["source_intersection_x1"] == 1500
    assert location["whole_resized_crop_x0"] == 512 * 0.512

    files = _write_shared_export_files(
        tmp_path / "square_crops",
        {split: _empty_coco() for split in ["train", "val", "test"]},
        [],
        [metadata_row],
        dataset_kind="square_crops",
    )
    locations_path = tmp_path / "square_crops" / "metadata" / "crop_locations.csv"
    assert locations_path in files
    locations = pd.read_csv(locations_path)
    assert locations.loc[0, "crop_x0"] == 512
    assert locations.loc[0, "crop_pad_right"] == 36

    readme_path = _write_dataset_readme(
        output_root=tmp_path,
        config=config,
        summary={"square_crops": {"splits": {"train": {"num_images": 1}}}},
    )
    readme = readme_path.read_text(encoding="utf-8")
    assert "crop_locations.csv" in readme
    assert "images/original" in readme
    assert "images/resized/1024x1024" in readme
    assert "images/resized/640x640" in readme
    assert "images/high_resolution" not in readme
    assert "single_file_per_source" not in readme  # documented as behavior, not config jargon
    assert "written exactly once" in readme
    assert "__crop__" in readme
    assert "0.5" in readme and "99.5" in readme


def test_storage_estimate_includes_high_resolution_padded_whole_companions() -> None:
    config = {
        "export": {"save_square_crops": True, "save_baseline_uncropped": False},
        "square_crops": {
            "crop_size": 1024,
            "stride": 512,
            "edge_policy": "regular_stride_pad",
            "train_crop_mode": "deterministic",
        },
        "paired_whole_images": {
            "enabled": True,
            "save_high_resolution": True,
            "target_width": 1024,
            "target_height": 1024,
            "high_resolution_canvas_mode": "per_image_square",
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
    estimate = estimate_export_space(
        config,
        [{"image_id": "one", "export_split": "train", "width": 1500, "height": 1000}],
    )
    assert estimate.crop_image_count == 2
    assert estimate.paired_whole_image_count == 2
    assert estimate.model_image_count == 4
    assert estimate.model_pixel_count == 3 * 1024 * 1024 + 1500 * 1500
