from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vindr_mammo.mask_debug import (
    compare_mask_methods,
    create_mask_padding_debug_bundle,
    mask_quality_metrics,
)


def test_debug_bundle_uses_full_crop_denominator_and_writes_padding(tmp_path) -> None:
    image = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 2:8] = True
    windows = [
        (0, 0, 6, 6),
        (4, 0, 10, 6),  # 2 columns, or 12/36 pixels, are padding.
        (2, 0, 8, 6),
    ]

    result = create_mask_padding_debug_bundle(
        image=image,
        unmasked_image=image,
        mask=mask,
        windows=windows,
        output_dir=tmp_path,
        crop_size=6,
        min_breast_fraction=16 / 36,
        comparison="strictly_greater_than",
        comparison_masks={"same": mask, "empty": np.zeros_like(mask)},
        max_crop_previews=3,
    )

    with (tmp_path / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert int(rows[1]["valid_source_pixels"]) == 24
    assert int(rows[1]["padding_pixels"]) == 12
    assert float(rows[1]["breast_fraction"]) == 16 / 36
    assert rows[1]["kept"] == "False"  # Equality fails the strict rule.
    assert rows[2]["kept"] == "True"

    saved_mask = np.load(tmp_path / "retained_breast_mask.npy", allow_pickle=False)
    np.testing.assert_array_equal(saved_mask, mask)
    padding = np.asarray(
        Image.open(tmp_path / "crop_previews" / "00_window_0001_padding.png")
    )
    assert int((padding > 0).sum()) == 12

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["window_count"] == 3
    assert summary["padded_window_count"] == 1
    assert summary["kept_window_count"] == 1
    assert summary["mask_method_comparison"][0]["iou_with_retained_mask"] == 1.0
    assert result.summary == summary


def test_mask_quality_metrics_flags_implausible_masks() -> None:
    empty = mask_quality_metrics(np.zeros((10, 10), dtype=bool))
    assert empty["quality_flags"] == ["empty_mask"]

    almost_full = np.ones((10, 10), dtype=bool)
    almost_full[0, 0] = False
    metrics = mask_quality_metrics(almost_full)
    assert "mask_covers_more_than_95_percent" in metrics["quality_flags"]
    assert metrics["bbox_xyxy"] == [0, 0, 10, 10]


def test_mask_method_comparison_returns_boolean_masks() -> None:
    yy, xx = np.mgrid[:96, :80]
    image = np.zeros((96, 80), dtype=np.float32)
    image[((xx - 15) / 42) ** 2 + ((yy - 48) / 38) ** 2 <= 1] = 1.0

    masks = compare_mask_methods(
        image,
        {
            "breast_mask_open_kernel": 1,
            "breast_mask_close_kernel": 1,
            "breast_mask_fill_holes": True,
            "breast_mask_keep_largest_component": True,
        },
    )

    assert set(masks) == {
        "largest_connected_tissue",
        "otsu_largest_connected_component",
        "percentile_threshold_largest_component",
    }
    assert all(mask.shape == image.shape for mask in masks.values())
    assert all(mask.dtype == np.bool_ for mask in masks.values())
    assert all(mask.any() for mask in masks.values())


def test_gui_preview_uses_retained_mask_and_marks_padding(monkeypatch) -> None:
    from vindr_mammo import gui_app

    image = np.ones((4, 6), dtype=np.float32)
    retained = np.zeros_like(image, dtype=bool)
    retained[:, :2] = True
    loaded = {
        "image": image,
        "foreground_mask": retained,
        "mass_boxes": np.zeros((0, 4), dtype=np.float32),
        "all_boxes": np.zeros((0, 4), dtype=np.float32),
        "record": {},
        "target_summary": {
            "image_id": "synthetic",
            "study_id": "study",
            "laterality": "L",
            "view_position": "CC",
            "num_masses": 0,
            "metadata": {},
        },
    }
    monkeypatch.setattr(gui_app, "_read_preprocessed_cached", lambda *_args: loaded)
    dataset = SimpleNamespace(
        data_root="/tmp",
        normalize="none",
        percentile_range=(0.5, 99.5),
        use_voi_lut=True,
        preprocess_options={},
    )
    controls = {
        "mode": "deterministic",
        "crop_size": 4,
        "stride": 4,
        "edge_policy": "regular_stride_pad",
        "positivity_threshold": 0.30,
        "only_mass_crops": False,
        "require_foreground": True,
        "min_foreground_fraction": -1.0,
        "foreground_threshold": None,
        "foreground_mask_preview": True,
        "crop_options": {
            "enabled": True,
            "mode": "deterministic",
            "crop_size": 4,
            "stride": 4,
            "edge_policy": "regular_stride_pad",
            "allow_partial_annotations": True,
            "min_box_visibility": 0.30,
            "reject_partial_windows": False,
            "pad_if_needed": True,
            "pad_value": 0.0,
        },
    }

    result = gui_app._prepare_sample(dataset, 0, controls, crop_index=1)

    assert [crop["foreground_fraction"] for crop in result["crops"]] == [0.5, 0.0]
    assert result["selected_crop"]["window"] == (4, 0, 8, 4)
    assert not result["foreground_mask_crop"].any()
    assert int(result["padding_mask_crop"].sum()) == 8
    assert result["crop_padding_info"] == {
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 2,
        "pad_bottom": 0,
    }
