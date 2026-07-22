from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vindr_mammo.dash_app import (
    _build_export_cfg_from_params,
    _config_control_outputs,
    _config_control_values,
)
from vindr_mammo.export import (
    _make_rgb_image,
    _pad_then_resize_rgb,
    _save_paired_whole_image_for_crop,
    _select_positive_ratio_candidates,
)
from vindr_mammo.gui_app import apply_channel_pipeline
from vindr_mammo.pipeline_scope import crop_array_to_window, step_applies_before_crop
from vindr_mammo.preprocessing import apply_geometry_preprocessing
from vindr_mammo.presets import (
    PAPER_69_PRESET_KEY,
    SIMPLE_PRESET_KEY,
    apply_study_preset,
)


def _base() -> dict:
    return {"paths": {"data_root": "/data/vindr", "output_root": "/exports/old"}}


def test_simple_preset_is_hermetic_and_matches_requested_pipeline() -> None:
    config = apply_study_preset(
        {
            **_base(),
            "square_crops": {"crop_size": 77, "stride": 3, "edge_policy": "edge_align"},
            "paired_whole_images": {"enabled": False, "target_width": 12},
            "image_export": {"rgb_scheme": "bitpack16"},
        },
        SIMPLE_PRESET_KEY,
    )

    assert config["paths"] == {
        "data_root": "/data/vindr",
        "output_root": "/exports/preprocessed-vindr-simple-preset-v1",
    }
    assert config["preprocess"]["crop_breast"] is True
    assert config["preprocess"]["mask_outside_breast"] is True
    assert config["preprocess"]["mirror_right_to_left"] is True
    crop = config["square_crops"]
    assert crop["crop_size"] == 1024
    assert crop["stride"] == 512
    assert crop["edge_policy"] == "regular_stride_pad"
    for split in ("train", "val", "test"):
        assert crop[f"{split}_crop_mode"] == "deterministic"
    assert crop["train_deterministic_selection_mode"] == "positive_ratio"
    assert crop["train_deterministic_target_positive_ratio"] == 0.5
    assert crop["train_require_clean_negative_windows"] is True
    assert crop["train_online_positive_ratio_selection_for_deterministic"] is True
    for split in ("val", "test"):
        assert crop[f"{split}_deterministic_selection_mode"] == "all"
        assert crop[f"{split}_require_clean_negative_windows"] is False
        assert crop[f"{split}_deterministic_require_foreground"] is False
        assert crop[f"{split}_negative_require_foreground"] is False
        assert crop[f"{split}_online_positive_ratio_selection_for_deterministic"] is False

    pipeline = config["image_export"]["custom_channel_pipeline"]
    expected = {"R": [0.0, 100.0], "G": [50.0, 100.0], "B": [75.0, 100.0]}
    for channel, percentiles in expected.items():
        steps = pipeline[channel]["steps"]
        assert steps[0]["op"] == "hist_equalize"
        assert step_applies_before_crop(steps[0]) is True
        assert steps[1]["op"] == "percentile_normalize"
        assert step_applies_before_crop(steps[1]) is True
        assert steps[1]["params"]["percentiles"] == percentiles

    paired = config["paired_whole_images"]
    assert paired["enabled"] is True
    assert (paired["target_width"], paired["target_height"]) == (1024, 1024)
    assert paired["canvas_mode"] == "per_image_square"
    assert paired["pad_value"] == 0.0


def test_simple_preset_synchronizes_new_dash_controls() -> None:
    config = apply_study_preset(_base(), SIMPLE_PRESET_KEY)
    values = _config_control_values(config)
    outputs = _config_control_outputs()
    assert len(values) == len(outputs)
    controls = {
        output.component_id: value
        for output, value in zip(outputs, values, strict=True)
    }
    assert controls["view-geometry"] == "crop"
    assert controls["crop-size"] == 1024
    assert controls["crop-stride"] == 512
    assert controls["crop-edge-policy"] == "regular_stride_pad"
    assert controls["require-foreground"] == ["on"]
    assert controls["min-foreground-fraction"] == 0.80
    assert controls["export-balance-mode"] == "positive_ratio"
    assert controls["export-target-positive-ratio"] == 0.5
    assert controls["paired-whole-enabled"] == ["on"]
    assert controls["paired-whole-size"] == 1024
    assert controls["split-strategy"] == "random_study_fraction"
    assert controls["split-val-fraction"] == 0.15


def test_simple_dash_export_preserves_online_train_and_complete_eval_grids() -> None:
    config = apply_study_preset(_base(), SIMPLE_PRESET_KEY)
    exported = _build_export_cfg_from_params(
        config,
        pd.DataFrame(),
        {
            "view_geometry": "crop",
            "pipeline_mode": "yaml",
            "pipeline": config["image_export"]["custom_channel_pipeline"],
            "export_parent": "/exports",
            "export_name": "simple-online",
            "crop_size": 1024,
            "crop_stride": 512,
            "crop_edge_policy": "regular_stride_pad",
            "require_foreground": ["on"],
            "min_foreground_fraction": 0.80,
            "fg_threshold_mode": "auto",
            "train_crop_mode": "deterministic",
            "val_crop_mode": "deterministic",
            "test_crop_mode": "deterministic",
            "export_balance_mode": "positive_ratio",
            "export_target_positive_ratio": 0.50,
            "split_strategy": "random_study_fraction",
            "split_val_fraction": 0.15,
            "split_seed": 123,
            "split_stratify_birads": ["on"],
        },
    )

    crop = exported["square_crops"]
    assert crop["train_deterministic_selection_mode"] == "positive_ratio"
    assert crop["train_online_positive_ratio_selection_for_deterministic"] is True
    for split in ("val", "test"):
        assert crop[f"{split}_deterministic_selection_mode"] == "all"
        assert crop[f"{split}_online_positive_ratio_selection_for_deterministic"] is False
        assert crop[f"{split}_deterministic_require_foreground"] is False
        assert crop[f"{split}_negative_require_foreground"] is False


def test_paper69_preset_preserves_full_resolution_and_adds_practical_validation() -> None:
    config = apply_study_preset(_base(), PAPER_69_PRESET_KEY)

    assert config["paths"]["output_root"] == "/exports/preprocessed-vindr-paper69-em-detr-v3"
    assert config["study_preset_provenance"]["preset_version"] == 3
    assert config["replication_contract"]["name"] == "paper69_vindr_practical_validation_mass_v3"
    assert config["image"]["normalize"] == "none"
    assert config["image"]["use_voi_lut"] is False
    pp = config["preprocess"]
    assert pp["trim_border_px"] == 5
    assert pp["intensity_scale_before_geometry"] == "minmax_uint8"
    assert pp["crop_breast"] is True
    assert pp["crop_padding"] == 0
    assert pp["breast_mask_method"] == "mammo_clip_contiguous_variance"
    assert pp["mirror_right_to_left"] is False
    assert pp["mask_outside_breast"] is False
    assert config["export"]["save_square_crops"] is False
    assert config["export"]["save_baseline_uncropped"] is True
    assert config["baseline_uncropped"]["resize_mode"] == "none"
    assert config["splits"] == {
        "strategy": "random_study_fraction",
        "val_fraction_from_training": 0.15,
        "validation_study_count": None,
        "validation_image_count": None,
        "seed": 123,
        "stratify_by_birads": True,
    }
    assert config["replication_contract"]["expected_source_images"] == {
        "train": 13600,
        "val": 2400,
        "test": 4000,
    }
    assert config["replication_contract"]["preserve_official_test"] is True
    assert config["training_augmentation"]["multiscale_widths"] == list(range(480, 801, 32))


def test_paper69_gui_can_restore_strict_official_membership_without_validation() -> None:
    config = apply_study_preset(_base(), PAPER_69_PRESET_KEY)
    exported = _build_export_cfg_from_params(
        config,
        pd.DataFrame(),
        {
            "view_geometry": "whole",
            "pipeline_mode": "yaml",
            "export_parent": "/exports",
            "export_name": "paper69-strict",
            "split_strategy": "official_only",
            "split_val_fraction": 0.15,
            "split_validation_study_count": 0,
            "split_validation_image_count": 0,
            "split_seed": 123,
            "split_stratify_birads": ["on"],
        },
    )

    assert exported["splits"]["strategy"] == "official_only"
    assert exported["splits"]["validation_study_count"] == 0
    assert exported["replication_contract"]["enabled"] is True
    assert exported["replication_contract"]["expected_source_images"] == {
        "train": 16000,
        "val": 0,
        "test": 4000,
    }


def test_paper69_mammoclip_background_crop_updates_boxes() -> None:
    image = torch.zeros((1, 100, 120), dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, 70).reshape(-1, 1)
    x = torch.linspace(0.0, 1.0, 80).reshape(1, -1)
    image[0, 15:85, 20:100] = 100.0 + 100.0 * y + 20.0 * x
    box = torch.tensor([[30.0, 30.0, 50.0, 50.0]])

    result = apply_geometry_preprocessing(
        image,
        boxes=box,
        mass_boxes=box,
        options={
            "trim_border_px": 5,
            "intensity_scale_before_geometry": "minmax_uint8",
            "crop_breast": True,
            "crop_padding": 0,
            "breast_mask_method": "mammo_clip_contiguous_variance",
            "min_box_visibility_after_crop": 0.0,
        },
    )

    assert tuple(result.image.shape) == (1, 70, 80)
    assert result.info["trim_box_xyxy"] == (5, 5, 115, 95)
    assert result.info["crop_box_xyxy"] == (20, 15, 100, 85)
    assert torch.equal(result.boxes, torch.tensor([[10.0, 15.0, 30.0, 35.0]]))
    assert result.box_keep.tolist() == [True]


def test_whole_scoped_operation_is_crop_of_full_transform_and_preview_matches_export() -> None:
    full = np.zeros((32, 40), dtype=np.float32)
    full[3:29, 4:36] = np.arange(26 * 32, dtype=np.float32).reshape(26, 32) + 1.0
    window = (28, 15, 44, 31)
    crop = crop_array_to_window(full, window, pad_value=0.0)
    pipeline = {
        channel: {
            "source": "current_crop",
            "steps": [
                {"op": "hist_equalize", "apply_before_crop": True, "params": {}},
                {"op": "percentile_normalize", "params": {"percentiles": [0.0, 100.0]}},
            ],
        }
        for channel in "RGB"
    }
    config = {
        "image_export": {"rgb_scheme": "custom_channel_pipeline", "custom_channel_pipeline": pipeline},
        "histogram_equalization": {"enabled": False},
    }

    exported, meta = _make_rgb_image(
        crop,
        config,
        full_source_arrays={"current_crop": full},
        crop_window=window,
    )
    preview, preview_meta = apply_channel_pipeline(
        crop,
        pipeline,
        source_full_images={"current_crop": full},
        crop_window=window,
    )
    crop_local, _ = _make_rgb_image(crop, config)

    assert np.array_equal(exported, preview)
    assert not np.array_equal(exported, crop_local)
    assert meta["custom_R_scope"]["whole_image_steps_applied"] == 1
    assert preview_meta["channels"]["R"]["whole_image_steps_applied"] == 1
    assert np.all(exported[:, -4:, :] == 0)  # out-of-image right padding stays background


def test_paired_whole_image_uses_crop_basename_and_pad_then_resize(tmp_path: Path) -> None:
    source = torch.zeros((1, 12, 20), dtype=torch.float32)
    source[:, 2:10, 2:18] = torch.linspace(0.1, 1.0, 8 * 16).reshape(1, 8, 16)
    pipeline = {
        channel: {
            "source": "current_crop",
            "steps": [{"op": "percentile_normalize", "params": {"percentiles": [0.0, 100.0]}}],
        }
        for channel in "RGB"
    }
    config = {
        "image_export": {"rgb_scheme": "custom_channel_pipeline", "custom_channel_pipeline": pipeline},
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
    }
    paired_cfg = {
        "enabled": True,
        "target_width": 32,
        "target_height": 32,
        "canvas_mode": "per_image_square",
        "pad_value": 0.0,
        "pad_anchor": "left_top",
        "storage_mode": "hardlink",
    }
    cache: dict[tuple[str, str], Path] = {}
    first = _save_paired_whole_image_for_crop(
        source_image=source,
        crop_root=tmp_path,
        split_name="train",
        filename="crop-a.png",
        source_image_id="source-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache=cache,
    )
    second = _save_paired_whole_image_for_crop(
        source_image=source,
        crop_root=tmp_path,
        split_name="train",
        filename="crop-b.png",
        source_image_id="source-1",
        config=config,
        paired_cfg=paired_cfg,
        source_path_cache=cache,
    )

    first_path = tmp_path / first["paired_whole_image_path"]
    second_path = tmp_path / second["paired_whole_image_path"]
    assert first_path.name == "crop-a.png"
    assert second_path.name == "crop-b.png"
    assert Image.open(first_path).size == (32, 32)
    assert Image.open(second_path).size == (32, 32)
    assert second["paired_whole_storage"] in {"hardlink", "copy_fallback"}
    geometry_keys = {
        "paired_whole_canvas_mode",
        "paired_whole_canvas_width",
        "paired_whole_canvas_height",
        "paired_whole_pad_left",
        "paired_whole_pad_top",
        "paired_whole_pad_value",
        "paired_whole_pad_anchor",
        "paired_whole_source_width",
        "paired_whole_source_height",
        "paired_whole_scale_x",
        "paired_whole_scale_y",
    }
    assert geometry_keys <= first.keys()
    assert {key: second[key] for key in geometry_keys} == {
        key: first[key] for key in geometry_keys
    }

    tiny = np.full((2, 4, 3), 255, dtype=np.uint8)
    resized, info = _pad_then_resize_rgb(tiny, {"target_width": 4, "target_height": 4, "pad_value": 0.0})
    assert info["paired_whole_canvas_width"] == 4
    assert info["paired_whole_canvas_height"] == 4
    assert np.all(resized[:2] == 255)
    assert np.all(resized[2:] == 0)


def test_simple_positive_ratio_selection_keeps_all_positives_and_one_negative_each() -> None:
    candidates = []
    for index in range(4):
        record = {"image_id": f"positive-source-{index}"}
        window = (index * 10, 0, index * 10 + 8, 8)
        candidates.append((record, window, {"is_positive_window": 1}))
    for index in range(12):
        record = {"image_id": f"negative-source-{index}"}
        window = (index * 10, 0, index * 10 + 8, 8)
        candidates.append((record, window, {"is_positive_window": 0, "is_clean_negative_window": 1}))

    cfg = {"train_deterministic_target_positive_ratio": 0.5}
    first = _select_positive_ratio_candidates(
        candidates, cfg, "train", np.random.default_rng(123)
    )
    second = _select_positive_ratio_candidates(
        candidates, cfg, "train", np.random.default_rng(123)
    )

    first_keys = [(item[0]["image_id"], item[1]) for item in first]
    assert first_keys == [(item[0]["image_id"], item[1]) for item in second]
    selected_positives = [item for item in first if item[2]["is_positive_window"] == 1]
    selected_negatives = [item for item in first if item[2]["is_positive_window"] == 0]
    assert len(selected_positives) == 4
    assert len(selected_negatives) == 4
    assert {
        item[0]["image_id"] for item in selected_positives
    } == {f"positive-source-{index}" for index in range(4)}
    assert all(item[2]["is_clean_negative_window"] == 1 for item in selected_negatives)


def test_paper69_rgb_is_exact_replicated_uint8() -> None:
    arr = np.asarray([[0.0, 1.0], [127.9, 255.0]], dtype=np.float32)
    rgb, meta = _make_rgb_image(
        arr,
        {
            "image_export": {"rgb_scheme": "paper69_mammoclip_uint8"},
            "histogram_equalization": {"enabled": False},
        },
    )
    assert rgb[..., 0].tolist() == [[0, 1], [127, 255]]
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])
    assert meta["paper69_mammoclip_uint8_replicated"] is True
