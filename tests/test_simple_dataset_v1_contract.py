from __future__ import annotations

import numpy as np
import pytest
import torch

import vindr_mammo.export as export_module
import vindr_mammo.preprocessing as preprocessing_module
from vindr_mammo.export import (
    _make_rgb_image,
    _online_positive_ratio_selection_enabled,
    _save_export_images,
    _windows_for_export_split,
)
from vindr_mammo.pipeline_scope import crop_array_to_window, step_applies_before_crop
from vindr_mammo.preprocessing import _postprocess_breast_mask
from vindr_mammo.presets import SIMPLE_PRESET_KEY, apply_study_preset


def _simple_config() -> dict:
    return apply_study_preset(
        {"paths": {"data_root": "/data/vindr", "output_root": "/exports/old"}},
        SIMPLE_PRESET_KEY,
    )


@pytest.mark.skipif(preprocessing_module.cv2 is None, reason="OpenCV mask postprocessing unavailable")
def test_hole_fill_with_foreground_at_origin_preserves_exterior_background() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[:9, :9] = True  # breast foreground includes (0, 0)
    mask[3:5, 3:5] = False  # enclosed internal hole

    filled = _postprocess_breast_mask(
        mask,
        {
            "breast_mask_open_kernel": 0,
            "breast_mask_close_kernel": 0,
            "breast_mask_keep_largest_component": False,
            "breast_mask_fill_holes": True,
        },
    )

    assert filled[0, 0]
    assert filled[3:5, 3:5].all()
    assert not filled[9:, :].any()
    assert not filled[:, 9:].any()


@pytest.mark.skipif(preprocessing_module.cv2 is None, reason="OpenCV mask postprocessing unavailable")
def test_black_training_negative_cannot_pass_80_percent_with_retained_mask() -> None:
    raw_mask = np.zeros((10, 10), dtype=bool)
    raw_mask[:7, :] = True  # foreground touches (0, 0), but covers only 70%
    raw_mask[2:4, 3:5] = False  # a real enclosed hole should be filled
    retained_mask = _postprocess_breast_mask(
        raw_mask,
        {
            "breast_mask_open_kernel": 0,
            "breast_mask_close_kernel": 0,
            "breast_mask_keep_largest_component": False,
            "breast_mask_fill_holes": True,
        },
    )
    assert retained_mask.mean() == pytest.approx(0.70)

    crop_cfg = {
        **_simple_config()["square_crops"],
        "crop_size": 10,
        "stride": 10,
        "edge_policy": "regular_stride_pad",
    }
    crop_options = {
        "crop_size": 10,
        "allow_partial_annotations": True,
        "min_box_visibility": 0.30,
        "reject_partial_windows": False,
        "negative_max_box_visibility": 0.0,
    }
    windows = _windows_for_export_split(
        split_name="train",
        image_width=10,
        image_height=10,
        image_tensor=torch.zeros((1, 10, 10), dtype=torch.float32),
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_options,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(19),
        foreground_mask=retained_mask,
    )

    assert windows == []


def test_simple_v1_declares_train_only_80_percent_online_negative_rule() -> None:
    crop_cfg = _simple_config()["square_crops"]

    assert crop_cfg.get("require_min_breast_fraction_for_all_crops", False) is False
    assert crop_cfg["train_deterministic_min_foreground_fraction"] == 0.80
    assert crop_cfg["train_negative_min_foreground_fraction"] == 0.80
    assert crop_cfg["train_negative_reject_blank_output"] is True
    assert crop_cfg["train_negative_min_output_signal_fraction"] == 0.01
    assert crop_cfg["train_online_positive_ratio_selection_for_deterministic"] is True
    assert _online_positive_ratio_selection_enabled(
        crop_cfg, "train", "deterministic"
    ) is True
    for split in ("val", "test"):
        assert crop_cfg[f"{split}_deterministic_selection_mode"] == "all"
        assert crop_cfg[f"{split}_deterministic_require_foreground"] is False
        assert crop_cfg[f"{split}_negative_require_foreground"] is False
        assert crop_cfg[f"{split}_negative_reject_blank_output"] is False
        assert _online_positive_ratio_selection_enabled(
            crop_cfg, split, "deterministic"
        ) is False


def test_train_annotated_crop_bypasses_80_percent_rule_but_negative_does_not() -> None:
    crop_cfg = {
        "crop_size": 10,
        "stride": 10,
        "edge_policy": "regular_stride_pad",
        "train_crop_mode": "deterministic",
        "train_deterministic_selection_mode": "positive_ratio",
        "train_deterministic_require_foreground": True,
        "train_deterministic_min_foreground_fraction": 0.80,
    }
    crop_options = {
        "crop_size": 10,
        "allow_partial_annotations": True,
        "min_box_visibility": 0.30,
        "reject_partial_windows": False,
        "negative_max_box_visibility": 0.0,
    }
    image = torch.ones((1, 10, 10), dtype=torch.float32)
    positive_box = torch.tensor([[2.0, 2.0, 4.0, 4.0]], dtype=torch.float32)

    mask_10 = np.zeros((10, 10), dtype=bool)
    mask_10.reshape(-1)[:10] = True
    mask_79 = np.zeros((10, 10), dtype=bool)
    mask_79.reshape(-1)[:79] = True
    mask_80 = np.zeros((10, 10), dtype=bool)
    mask_80.reshape(-1)[:80] = True

    annotated = _windows_for_export_split(
        split_name="train",
        image_width=10,
        image_height=10,
        image_tensor=image,
        mass_boxes=positive_box,
        crop_options=crop_options,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(7),
        foreground_mask=mask_10,
    )
    unannotated_below = _windows_for_export_split(
        split_name="train",
        image_width=10,
        image_height=10,
        image_tensor=image,
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_options,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(7),
        foreground_mask=mask_79,
    )
    unannotated_at_threshold = _windows_for_export_split(
        split_name="train",
        image_width=10,
        image_height=10,
        image_tensor=image,
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_options,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(7),
        foreground_mask=mask_80,
    )

    assert len(annotated) == 1
    assert annotated[0][1]["is_positive_window"] == 1
    assert annotated[0][1]["foreground_fraction"] == 0.10
    assert unannotated_below == []
    assert len(unannotated_at_threshold) == 1
    assert unannotated_at_threshold[0][1]["is_positive_window"] == 0
    assert unannotated_at_threshold[0][1]["foreground_fraction"] == 0.80


def test_simple_v1_validation_and_test_keep_complete_grid_without_mask_filter() -> None:
    crop_cfg = _simple_config()["square_crops"]
    image = torch.zeros((1, 10, 20), dtype=torch.float32)
    empty_mask = np.zeros((10, 20), dtype=bool)
    crop_options = {
        "crop_size": 10,
        "allow_partial_annotations": True,
        "min_box_visibility": 0.30,
        "reject_partial_windows": False,
        "negative_max_box_visibility": 0.0,
    }
    # Use a small synthetic grid while retaining the preset's split policies.
    crop_cfg = {**crop_cfg, "crop_size": 10, "stride": 10}
    for split in ("val", "test"):
        windows = _windows_for_export_split(
            split_name=split,
            image_width=20,
            image_height=10,
            image_tensor=image,
            mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
            crop_options=crop_options,
            crop_cfg=crop_cfg,
            rng=np.random.default_rng(11),
            foreground_mask=empty_mask,
        )
        assert [window for window, _info in windows] == [
            (0, 0, 10, 10),
            (10, 0, 20, 10),
        ]


def test_simple_v1_histogram_equalization_uses_whole_image_before_crop() -> None:
    config = _simple_config()
    pipeline = config["image_export"]["custom_channel_pipeline"]
    for channel in "RGB":
        assert pipeline[channel]["steps"][0]["op"] == "hist_equalize"
        assert pipeline[channel]["steps"][1]["op"] == "percentile_normalize"
        assert all(
            step_applies_before_crop(step)
            for step in pipeline[channel]["steps"]
        )

    window = (30, 16, 62, 48)
    full_a = np.zeros((64, 96), dtype=np.float32)
    full_a[4:60, 4:92] = 20.0
    full_a[16:48, 30:62] = np.linspace(
        100.0, 300.0, 32 * 32, dtype=np.float32
    ).reshape(32, 32)
    full_b = full_a.copy()
    outside_values = np.linspace(
        50.0, 1000.0, 12 * 88, dtype=np.float32
    ).reshape(12, 88)
    full_b[4:16, 4:92] = outside_values
    full_b[48:60, 4:92] = outside_values

    crop_a = crop_array_to_window(full_a, window)
    crop_b = crop_array_to_window(full_b, window)
    assert np.array_equal(crop_a, crop_b)

    whole_scoped_a, meta_a = _make_rgb_image(
        crop_a,
        config,
        full_source_arrays={"current_crop": full_a},
        crop_window=window,
    )
    whole_scoped_b, meta_b = _make_rgb_image(
        crop_b,
        config,
        full_source_arrays={"current_crop": full_b},
        crop_window=window,
    )
    crop_local_a, _ = _make_rgb_image(crop_a, config)
    crop_local_b, _ = _make_rgb_image(crop_b, config)

    # Identical crop pixels produce different results only when statistics are
    # gathered from the whole breast before the window is extracted.
    assert not np.array_equal(whole_scoped_a, whole_scoped_b)
    assert np.array_equal(crop_local_a, crop_local_b)
    for metadata in (meta_a, meta_b):
        for channel in "RGB":
            scope = metadata[f"custom_{channel}_scope"]
            assert scope["whole_image_steps_applied"] == 1
            assert scope["whole_image_steps_fell_back_to_crop"] == 0
            assert scope["scope_execution_order"] == (
                "whole_image_steps_then_crop_then_crop_steps"
            )


def test_whole_scoped_pipeline_uses_authoritative_retained_mask(monkeypatch) -> None:
    config = _simple_config()
    full = np.linspace(1.0, 4000.0, 80 * 48, dtype=np.float32).reshape(80, 48)
    retained_mask = np.ones(full.shape, dtype=bool)
    window = (0, 16, 48, 64)
    crop = crop_array_to_window(full, window)

    # Reproduce the failure mode: a second foreground estimate on a tightly
    # cropped breast keeps only one saturated pixel and annihilates hist-eq.
    def broken_reestimate(arr, threshold=None):
        del threshold
        mask = np.zeros(np.asarray(arr).shape, dtype=bool)
        mask[0, 0] = True
        return mask

    monkeypatch.setattr(export_module, "_foreground_mask", broken_reestimate)
    broken_rgb, _ = _make_rgb_image(
        crop,
        config,
        full_source_arrays={"current_crop": full},
        crop_window=window,
    )
    fixed_rgb, metadata = _make_rgb_image(
        crop,
        config,
        full_source_arrays={"current_crop": full},
        full_source_masks={"current_crop": retained_mask},
        crop_window=window,
    )

    assert not np.any(broken_rgb)
    assert np.any(fixed_rgb)
    for channel in "RGB":
        assert metadata[f"custom_{channel}_scope"][
            "whole_image_supplied_stat_mask_used"
        ] == 1


def test_whole_scoped_pipeline_rejects_mismatched_retained_mask() -> None:
    config = _simple_config()
    full = np.arange(80 * 48, dtype=np.float32).reshape(80, 48)
    window = (0, 16, 48, 64)
    crop = crop_array_to_window(full, window)

    with pytest.raises(ValueError, match="statistics mask shape"):
        _make_rgb_image(
            crop,
            config,
            full_source_arrays={"current_crop": full},
            full_source_masks={"current_crop": np.ones((79, 48), dtype=bool)},
            crop_window=window,
        )


def test_blank_output_guard_does_not_write_training_image(tmp_path) -> None:
    config = _simple_config()
    config["preserved_16bit"]["save"] = False
    rel_path = "images/train/blank.png"

    result = _save_export_images(
        torch.zeros((1, 32, 32), dtype=torch.float32),
        tmp_path,
        rel_path,
        config,
        reject_blank_output=True,
        min_output_signal_fraction=0.01,
    )

    assert result["output_rejected_blank"] is True
    assert result["output_signal_fraction"] == 0.0
    assert not (tmp_path / rel_path).exists()
