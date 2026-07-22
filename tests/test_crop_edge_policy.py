from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vindr_mammo.crops import (
    crop_image_and_boxes_to_window,
    make_crop_options,
    sliding_square_windows,
)
from vindr_mammo.export import _windows_for_export_split
from vindr_mammo.gui_app import _build_gui_export_config, _current_preprocessing_yaml_payload


def test_edge_align_remains_the_default_and_preserves_legacy_origins() -> None:
    assert make_crop_options(None)["edge_policy"] == "edge_align"

    windows = sliding_square_windows(width=2500, height=1800, crop_size=1024, stride=512)
    xs = sorted({window[0] for window in windows})
    ys = sorted({window[1] for window in windows})

    assert xs == [0, 512, 1024, 1476]
    assert ys == [0, 512, 776]
    assert all(window[2] <= 2500 and window[3] <= 1800 for window in windows)


def test_regular_stride_policy_keeps_grid_origins_and_extends_past_edges() -> None:
    windows = sliding_square_windows(
        width=2500,
        height=1800,
        crop_size=1024,
        stride=512,
        edge_policy="regular_stride_pad",
    )
    xs = sorted({window[0] for window in windows})
    ys = sorted({window[1] for window in windows})

    assert xs == [0, 512, 1024, 1536]
    assert ys == [0, 512, 1024]
    assert np.diff(xs).tolist() == [512, 512, 512]
    assert np.diff(ys).tolist() == [512, 512]
    assert windows[-1] == (1536, 1024, 2560, 2048)


def test_regular_stride_edge_window_is_zero_padded_and_keeps_boxes_aligned() -> None:
    image = torch.ones((1, 1800, 2500), dtype=torch.float32)
    box = torch.tensor([[2300.0, 1600.0, 2450.0, 1750.0]], dtype=torch.float32)
    options = {
        "enabled": True,
        "mode": "deterministic",
        "crop_size": 1024,
        "stride": 512,
        "edge_policy": "regular_stride_pad",
        "pad_if_needed": True,
        "pad_value": 0.0,
        "allow_partial_annotations": False,
    }

    result = crop_image_and_boxes_to_window(
        image,
        boxes=box,
        mass_boxes=box,
        window_xyxy=(1536, 1024, 2560, 2048),
        options=options,
    )

    assert tuple(result.image.shape) == (1, 1024, 1024)
    assert result.info["pad_right"] == 60
    assert result.info["pad_bottom"] == 248
    assert result.info["edge_policy"] == "regular_stride_pad"
    assert torch.all(result.image[:, :776, :964] == 1.0)
    assert torch.all(result.image[:, :, 964:] == 0.0)
    assert torch.all(result.image[:, 776:, :] == 0.0)
    assert result.mass_boxes.tolist() == [[764.0, 576.0, 914.0, 726.0]]


def test_export_window_planner_threads_regular_stride_policy() -> None:
    windows = _windows_for_export_split(
        split_name="train",
        image_width=2500,
        image_height=1800,
        image_tensor=torch.ones((1, 1800, 2500), dtype=torch.float32),
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options={
            "crop_size": 1024,
            "stride": 512,
            "edge_policy": "regular_stride_pad",
            "allow_partial_annotations": False,
        },
        crop_cfg={
            "crop_size": 1024,
            "stride": 512,
            "edge_policy": "regular_stride_pad",
            "train_crop_mode": "deterministic",
            "train_deterministic_selection_mode": "all",
            "train_deterministic_require_foreground": False,
            "train_require_clean_negative_windows": True,
        },
        rng=np.random.default_rng(123),
    )

    planned = [window for window, _info in windows]
    assert planned[-1] == (1536, 1024, 2560, 2048)
    assert len(planned) == 12


def test_legacy_gui_payload_and_export_config_preserve_edge_policy() -> None:
    crop_controls = {
        "crop_size": 1024,
        "stride": 512,
        "edge_policy": "regular_stride_pad",
        "mode": "deterministic",
        "crop_options": {
            "edge_policy": "regular_stride_pad",
            "allow_partial_annotations": False,
            "min_box_visibility": 0.3,
            "reject_partial_windows": True,
            "negative_max_box_visibility": 0.0,
        },
        "contralateral_source_alignment": {},
    }
    payload = _current_preprocessing_yaml_payload(
        config_path=Path("config/export_config.yaml"),
        cfg={},
        crop_controls=crop_controls,
        display_controls={},
        pipeline={},
    )
    assert payload["crop_preview_settings"]["edge_policy"] == "regular_stride_pad"
    assert payload["export_config_patch"]["square_crops"]["edge_policy"] == "regular_stride_pad"

    export_cfg = _build_gui_export_config(
        cfg={},
        output_root=Path("/tmp/edge-policy-export"),
        clean_output=False,
        selected_vendors=[],
        deterministic_selection={},
        split_crop_modes={split: "deterministic" for split in ["train", "val", "test"]},
        save_square=True,
        save_baseline=False,
        crop_controls=crop_controls,
        pipeline={},
    )
    assert export_cfg["square_crops"]["edge_policy"] == "regular_stride_pad"
