from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vindr_mammo.dash_app import (
    VISUAL_PIPELINE_PARAM_KEYS,
    _build_export_cfg_from_params,
    _config_control_outputs,
    _config_control_values,
    _visual_pipeline_builder,
    _visual_pipeline_from_params,
)
from vindr_mammo.export import (
    _expand_training_to_patient_breast_views,
    _select_positive_ratio_candidates,
    _select_source_breast_ratio_candidates,
    _windows_for_export_split,
)
from vindr_mammo.gui_app import _build_gui_export_config
from vindr_mammo.preprocessing import apply_geometry_preprocessing
from vindr_mammo.presets import (
    PAPER_22_IMPROVED_PRESET_KEY,
    PAPER_22_PRESET_KEY,
    apply_study_preset,
)


def _base() -> dict:
    return {
        "paths": {
            "data_root": "/mnt/t9/vindr-data/vindr",
            "output_root": "/mnt/t9/vindr-data/base",
        }
    }


def _improved() -> dict:
    return apply_study_preset(_base(), PAPER_22_IMPROVED_PRESET_KEY)


def test_old_and_improved_paper22_are_separate_versioned_presets() -> None:
    old = apply_study_preset(_base(), PAPER_22_PRESET_KEY)
    improved = _improved()

    assert old["paths"]["output_root"].endswith("preprocessed-vindr-paper22-v2")
    assert improved["paths"]["output_root"].endswith(
        "preprocessed-vindr-paper22-improved-v8"
    )
    assert old["square_crops"]["train_deterministic_selection_mode"] == "negative_fraction"
    assert improved["square_crops"]["train_deterministic_selection_mode"] == "crop_label_ratio"
    assert improved["square_crops"]["train_online_positive_ratio_selection_for_deterministic"] is True
    assert improved["study_preset_provenance"]["preset_version"] == 8
    assert improved["crop_annotation_policy"]["allow_partial_annotations"] is True
    assert improved["crop_annotation_policy"]["min_box_visibility"] == 0.05
    assert improved["replication_contract"]["expected_min_box_visibility"] == 0.05
    assert improved["dataset_review"]["enabled"] is True
    assert improved["dataset_review"]["save_original_previews"] is True
    assert improved["dataset_review"]["save_source_previews"] is True
    assert improved["dataset_review"]["samples_per_split"] == 200
    assert improved["paired_whole_images"]["enabled"] is True
    assert improved["source_cohort"]["train_expand_to_all_patient_breast_views"] is True
    assert old["preprocess"]["mirror_right_to_left"] is False
    assert improved["preprocess"]["mirror_right_to_left"] is True
    expected_clahe = [{
        "op": "clahe",
        "apply_before_crop": True,
        "params": {"clip_limit": 2.0, "tile_grid_size": 8},
    }]
    for channel in ["R", "G", "B"]:
        assert improved["image_export"]["custom_channel_pipeline"][channel] == {
            "source": "current_crop",
            "steps": expected_clahe,
        }
    for split in ["train", "val", "test"]:
        assert improved["square_crops"][
            f"{split}_require_min_breast_fraction_for_all_crops"
        ] is True
        assert improved["square_crops"][
            f"{split}_min_breast_fraction_for_all_crops"
        ] == 0.10
        assert improved["square_crops"][
            f"{split}_breast_fraction_comparison_for_all_crops"
        ] == "strictly_greater_than"


def test_improved_preset_canonical_orientation_mirrors_pixels_boxes_and_mask() -> None:
    image = torch.zeros((1, 100, 120), dtype=torch.float32)
    for y in range(10, 90):
        image[:, y, 30 + abs(y - 50):110] = 1.0
    box = torch.tensor([[80.0, 30.0, 100.0, 70.0]], dtype=torch.float32)

    result = apply_geometry_preprocessing(
        image,
        boxes=box,
        mass_boxes=box,
        options={**_improved()["preprocess"], "crop_threshold": 0.5},
    )

    assert result.info["mirrored"] is True
    image_xs = torch.where(result.image.squeeze(0) > 0)[1].to(torch.float32)
    assert float(image_xs.mean()) < 60.0
    assert torch.equal(
        result.mass_boxes,
        torch.tensor([[20.0, 30.0, 40.0, 70.0]], dtype=torch.float32),
    )
    assert result.foreground_mask is not None
    assert float(np.where(result.foreground_mask)[1].mean()) < 60.0


def test_dash_visual_builder_hydrates_from_improved_preset_pipeline() -> None:
    root = _visual_pipeline_builder(
        _improved()["image_export"]["custom_channel_pipeline"]
    )

    def _walk(component):
        if isinstance(component, (list, tuple)):
            for item in component:
                yield from _walk(item)
            return
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)

    components = {
        str(getattr(component, "id", None)): component
        for component in _walk(root)
        if getattr(component, "id", None) is not None
    }
    for channel in ["R", "G", "B"]:
        prefix = str({"type": "stage-op", "stage": channel, "idx": 0})
        before = str({"type": "stage-before-crop", "stage": channel, "idx": 0})
        clip = str({"type": "stage-clip", "stage": channel, "idx": 0})
        tile = str({"type": "stage-tile", "stage": channel, "idx": 0})
        assert components[prefix].value == "clahe"
        assert components[before].value == ["on"]
        assert components[clip].value == 2.0
        assert components[tile].value == 8


def test_hydrated_visual_builder_round_trips_to_advanced_yaml_pipeline() -> None:
    params = {key: [] for key in VISUAL_PIPELINE_PARAM_KEYS}
    params.update({
        "stage_sources": ["current_crop", "current_crop", "current_crop"],
        "stage_ops": (
            ["none"] * 4
            + ["clahe", "none", "none", "none"] * 3
            + ["none"] * 4
        ),
        "stage_before_crop": (
            [[]] * 4
            + [["on"], [], [], []] * 3
            + [[]] * 4
        ),
        "stage_clips": [2.0, 2.0, 2.0],
        "stage_tiles": [8, 8, 8],
    })

    assert _visual_pipeline_from_params(params) == _improved()["image_export"][
        "custom_channel_pipeline"
    ]


class _FakeDataset:
    def __init__(self) -> None:
        self.image_records = [
            {"study_id": "train", "image_id": "tl-cc", "laterality": "L", "view_position": "CC", "split": "training"},
            {"study_id": "train", "image_id": "tl-mlo", "laterality": "L", "view_position": "MLO", "split": "training"},
            {"study_id": "train", "image_id": "tr-cc", "laterality": "R", "view_position": "CC", "split": "training"},
            {"study_id": "train", "image_id": "tr-mlo", "laterality": "R", "view_position": "MLO", "split": "training"},
            {"study_id": "val", "image_id": "val-positive", "laterality": "L", "view_position": "CC", "split": "training"},
            {"study_id": "val", "image_id": "val-extra", "laterality": "R", "view_position": "CC", "split": "training"},
            {"study_id": "test", "image_id": "test-positive", "laterality": "L", "view_position": "CC", "split": "test"},
            {"study_id": "test", "image_id": "test-extra", "laterality": "R", "view_position": "CC", "split": "test"},
        ]
        self.findings_by_image_id = {
            "tl-cc": [{"finding_categories": "['Mass']"}],
            "val-positive": [{"finding_categories": "['Mass']"}],
            "test-positive": [{"finding_categories": "['Mass']"}],
        }

    @staticmethod
    def _filter_mass_findings(findings):
        return [finding for finding in findings if "Mass" in str(finding.get("finding_categories", ""))]


def test_training_expansion_is_patient_safe_and_breast_laterality_aware() -> None:
    dataset = _FakeDataset()
    original = {
        "train": [dataset.image_records[0]],
        "val": [dataset.image_records[4]],
        "test": [dataset.image_records[6]],
    }
    expanded, summary = _expand_training_to_patient_breast_views(
        dataset, original, _improved()
    )

    assert {record["image_id"] for record in expanded["train"]} == {
        "tl-cc", "tl-mlo", "tr-cc", "tr-mlo"
    }
    assert [record["image_id"] for record in expanded["val"]] == ["val-positive"]
    assert [record["image_id"] for record in expanded["test"]] == ["test-positive"]
    statuses = {record["image_id"]: record["_source_breast_has_mass"] for record in expanded["train"]}
    assert statuses == {"tl-cc": 1, "tl-mlo": 1, "tr-cc": 0, "tr-mlo": 0}
    image_statuses = {
        record["image_id"]: record["_source_image_has_mass"]
        for record in expanded["train"]
    }
    assert image_statuses == {"tl-cc": 1, "tl-mlo": 0, "tr-cc": 0, "tr-mlo": 0}
    assert summary["validation_membership_changed"] is False
    assert summary["test_membership_changed"] is False


def test_source_breast_selector_is_exact_half_and_keeps_every_lesion_window() -> None:
    candidates = []
    for index in range(8):
        candidates.append((
            {"image_id": f"mass-{index}"},
            (index, 0, index + 1, 1),
            {"source_breast_has_mass": 1, "is_positive_window": int(index < 3)},
        ))
    for index in range(5):
        candidates.append((
            {"image_id": f"negative-{index}"},
            (index, 1, index + 1, 2),
            {"source_breast_has_mass": 0, "is_positive_window": 0},
        ))

    selected = _select_source_breast_ratio_candidates(
        candidates,
        {"train_deterministic_target_source_breast_mass_ratio": 0.50},
        "train",
        np.random.default_rng(123),
    )

    assert len(selected) == 10
    assert sum(int(candidate[2]["source_breast_has_mass"]) for candidate in selected) == 5
    assert {f"mass-{index}" for index in range(3)} <= {
        candidate[0]["image_id"] for candidate in selected
    }
    assert {candidate[2]["source_breast_achieved_mass_ratio"] for candidate in selected} == {0.5}


def test_crop_label_ratio_keeps_all_mass_crops_and_uses_only_negative_breasts() -> None:
    candidates = [
        ({"image_id": "mass-a"}, (0, 0, 10, 10), {
            "is_positive_window": 1, "source_image_has_mass": 1,
            "source_breast_has_mass": 1,
        }),
        ({"image_id": "mass-b"}, (10, 0, 20, 10), {
            "is_positive_window": 1, "source_image_has_mass": 1,
            "source_breast_has_mass": 1,
        }),
        ({"image_id": "mass-a"}, (20, 0, 30, 10), {
            "is_positive_window": 0, "source_image_has_mass": 1,
            "source_breast_has_mass": 1,
        }),
        ({"image_id": "paired-view"}, (0, 0, 10, 10), {
            "is_positive_window": 0, "source_image_has_mass": 0,
            "source_breast_has_mass": 1,
        }),
        ({"image_id": "negative-a"}, (0, 0, 10, 10), {
            "is_positive_window": 0, "source_image_has_mass": 0,
            "source_breast_has_mass": 0,
        }),
        ({"image_id": "negative-b"}, (0, 0, 10, 10), {
            "is_positive_window": 0, "source_image_has_mass": 0,
            "source_breast_has_mass": 0,
        }),
        ({"image_id": "negative-c"}, (0, 0, 10, 10), {
            "is_positive_window": 0, "source_image_has_mass": 0,
            "source_breast_has_mass": 0,
        }),
    ]

    selected = _select_positive_ratio_candidates(
        candidates,
        {"train_deterministic_target_positive_ratio": 0.50},
        "train",
        np.random.default_rng(4),
        negative_images_only=True,
        negative_breasts_only=True,
        selection_label="crop_label_ratio_negative_breasts_only",
    )

    positives = [candidate for candidate in selected if candidate[2]["is_positive_window"] == 1]
    negatives = [candidate for candidate in selected if candidate[2]["is_positive_window"] == 0]
    assert {candidate[0]["image_id"] for candidate in positives} == {"mass-a", "mass-b"}
    assert len(positives) == len(negatives) == 2
    assert all(candidate[2]["source_image_has_mass"] == 0 for candidate in negatives)
    assert all(candidate[2]["source_breast_has_mass"] == 0 for candidate in negatives)
    assert "paired-view" not in {
        candidate[0]["image_id"] for candidate in negatives
    }
    assert all(
        candidate[2]["negative_crop_source_policy"] == "mass_negative_breasts_only"
        for candidate in selected
    )


def test_improved_preset_labels_a_mass_fragment_at_five_percent_visibility() -> None:
    crop_cfg = {
        **_improved()["square_crops"],
        "crop_size": 100,
        "stride": 100,
        "edge_policy": "regular_stride_pad",
    }
    image = torch.ones((1, 100, 200), dtype=torch.float32)
    mask = np.ones((100, 200), dtype=bool)

    at_threshold = _windows_for_export_split(
        split_name="train",
        image_width=200,
        image_height=100,
        image_tensor=image,
        mass_boxes=torch.tensor([[99.0, 10.0, 119.0, 30.0]]),
        crop_options=_improved()["crop_annotation_policy"],
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(3),
        foreground_mask=mask,
    )
    below_threshold = _windows_for_export_split(
        split_name="train",
        image_width=200,
        image_height=100,
        image_tensor=image,
        mass_boxes=torch.tensor([[99.2, 10.0, 119.2, 30.0]]),
        crop_options=_improved()["crop_annotation_policy"],
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(3),
        foreground_mask=mask,
    )

    at_by_x = {window[0]: info for window, info in at_threshold}
    below_by_x = {window[0]: info for window, info in below_threshold}
    assert at_by_x[0]["is_positive_window"] == 1
    assert 0 not in below_by_x


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_split_requires_strictly_more_than_ten_percent_breast_mask(
    split: str,
) -> None:
    crop_cfg = {
        **_improved()["square_crops"],
        "crop_size": 10,
        "stride": 10,
        "edge_policy": "regular_stride_pad",
    }
    crop_policy = _improved()["crop_annotation_policy"]
    image = torch.ones((1, 10, 10), dtype=torch.float32)
    exactly_ten = np.zeros((10, 10), dtype=bool)
    exactly_ten.reshape(-1)[:10] = True
    eleven = np.zeros((10, 10), dtype=bool)
    eleven.reshape(-1)[:11] = True

    at_threshold = _windows_for_export_split(
        split_name=split,
        image_width=10,
        image_height=10,
        image_tensor=image,
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(1),
        foreground_mask=exactly_ten,
    )
    above_threshold = _windows_for_export_split(
        split_name=split,
        image_width=10,
        image_height=10,
        image_tensor=image,
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(1),
        foreground_mask=eleven,
    )

    assert at_threshold == []
    assert len(above_threshold) == 1
    assert above_threshold[0][1]["foreground_fraction"] == pytest.approx(0.11)


def test_improved_validation_and_test_reject_background_only_grid_windows() -> None:
    crop_cfg = {
        **_improved()["square_crops"],
        "crop_size": 10,
        "stride": 10,
        "edge_policy": "regular_stride_pad",
    }
    image = torch.zeros((1, 10, 20), dtype=torch.float32)
    empty_mask = np.zeros((10, 20), dtype=bool)
    for split in ["val", "test"]:
        windows = _windows_for_export_split(
            split_name=split,
            image_width=20,
            image_height=10,
            image_tensor=image,
            mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
            crop_options=_improved()["crop_annotation_policy"],
            crop_cfg=crop_cfg,
            rng=np.random.default_rng(2),
            foreground_mask=empty_mask,
        )
        assert windows == []


def test_improved_preset_and_source_breast_mode_round_trip_through_dash_gui() -> None:
    config = _improved()
    controls = {
        output.component_id: value
        for output, value in zip(
            _config_control_outputs(), _config_control_values(config), strict=True
        )
    }
    assert controls["export-balance-mode"] == "crop_label_ratio"
    assert controls["positivity-threshold"] == 0.05
    assert controls["min-box-visibility"] == 0.05
    assert controls["paired-whole-enabled"] == ["on"]
    assert controls["export-review-enabled"] == ["on"]
    assert controls["export-review-samples"] == 200
    assert controls["require-foreground"] == ["on"]
    assert controls["min-foreground-fraction"] == 0.10
    assert controls["val-require-foreground"] == ["on"]
    assert controls["val-min-foreground-fraction"] == 0.10
    assert controls["test-require-foreground"] == ["on"]
    assert controls["test-min-foreground-fraction"] == 0.10

    exported = _build_export_cfg_from_params(
        config,
        pd.DataFrame(),
        {
            "view_geometry": "crop",
            "pipeline_mode": "yaml",
            "pipeline": config["image_export"]["custom_channel_pipeline"],
            "export_parent": "/mnt/t9/vindr-data",
            "export_name": "preprocessed-vindr-paper22-improved-v8",
            "crop_size": 640,
            "crop_stride": 512,
            "crop_edge_policy": "edge_align",
            "allow_partial": ["on"],
            "min_box_visibility": 0.05,
            "positivity_threshold": 0.05,
            "require_foreground": ["on"],
            "min_foreground_fraction": 0.10,
            "val_require_foreground": ["on"],
            "val_min_foreground_fraction": 0.10,
            "test_require_foreground": ["on"],
            "test_min_foreground_fraction": 0.10,
            "fg_threshold_mode": "auto",
            "train_crop_mode": "deterministic",
            "val_crop_mode": "deterministic",
            "test_crop_mode": "deterministic",
            "export_balance_mode": "crop_label_ratio",
            "export_target_positive_ratio": 0.50,
            "split_strategy": "exact_study_count",
            "split_validation_study_count": 71,
            "split_validation_image_count": 136,
            "split_seed": 123,
            "split_stratify_birads": ["on"],
        },
    )
    square = exported["square_crops"]
    assert exported["crop_annotation_policy"]["min_box_visibility"] == 0.05
    assert exported["replication_contract"]["expected_min_box_visibility"] == 0.05
    assert square["train_deterministic_selection_mode"] == "crop_label_ratio"
    assert square["train_online_positive_ratio_selection_for_deterministic"] is True
    assert square["train_min_breast_fraction_for_all_crops"] == 0.10
    assert square["train_breast_fraction_comparison_for_all_crops"] == "strictly_greater_than"
    assert square["val_deterministic_selection_mode"] == "all"
    assert square["test_deterministic_selection_mode"] == "all"
    assert {square[f"{split}_crop_mode"] for split in ["train", "val", "test"]} == {"deterministic"}
    assert square["val_deterministic_require_foreground"] is False
    assert square["test_deterministic_require_foreground"] is False
    for split in ["train", "val", "test"]:
        assert square[f"{split}_require_min_breast_fraction_for_all_crops"] is True
        assert square[f"{split}_min_breast_fraction_for_all_crops"] == 0.10
        assert square[f"{split}_breast_fraction_comparison_for_all_crops"] == (
            "strictly_greater_than"
        )
        assert square[f"{split}_require_retained_breast_mask_for_all_crops"] is True
    assert exported["source_cohort"]["train_expand_to_all_patient_breast_views"] is True
    assert exported["splits"]["val_fraction_from_training"] == 0.0
    assert exported["replication_contract"]["enabled"] is True
    assert exported["replication_contract"]["expected_train_selection_mode"] == "crop_label_ratio"
    assert exported["replication_contract"]["expected_train_crop_positive_fraction"] == 0.50
    assert exported["replication_contract"]["require_training_negative_crops_from_mass_negative_images"] is True
    assert exported["replication_contract"]["require_training_negative_crops_from_mass_negative_breasts"] is True


def test_dash_gui_can_override_each_split_breast_filter_independently() -> None:
    config = _improved()
    exported = _build_export_cfg_from_params(
        config,
        pd.DataFrame(),
        {
            "view_geometry": "crop",
            "pipeline_mode": "yaml",
            "pipeline": config["image_export"]["custom_channel_pipeline"],
            "export_parent": "/mnt/t9/vindr-data",
            "export_name": "independent-mask-controls",
            "crop_size": 640,
            "crop_stride": 512,
            "crop_edge_policy": "edge_align",
            "require_foreground": ["on"],
            "min_foreground_fraction": 0.10,
            "val_require_foreground": [],
            "val_min_foreground_fraction": 0.20,
            "test_require_foreground": ["on"],
            "test_min_foreground_fraction": 0.30,
            "fg_threshold_mode": "auto",
            "train_crop_mode": "deterministic",
            "val_crop_mode": "deterministic",
            "test_crop_mode": "deterministic",
            "export_balance_mode": "source_breast_ratio",
            "export_target_positive_ratio": 0.50,
            "split_strategy": "exact_study_count",
            "split_validation_study_count": 71,
            "split_validation_image_count": 136,
            "split_seed": 123,
            "split_stratify_birads": ["on"],
        },
    )
    square = exported["square_crops"]
    assert square["train_require_min_breast_fraction_for_all_crops"] is True
    assert square["train_min_breast_fraction_for_all_crops"] == 0.10
    assert square["val_require_min_breast_fraction_for_all_crops"] is False
    assert square["val_min_breast_fraction_for_all_crops"] == 0.20
    assert square["test_require_min_breast_fraction_for_all_crops"] is True
    assert square["test_min_breast_fraction_for_all_crops"] == 0.30
    assert exported["replication_contract"][
        "min_breast_fraction_strictly_greater_than_by_split"
    ] == {"train": 0.10, "test": 0.30}
    assert exported["replication_contract"]["expected_train_selection_mode"] == "source_breast_ratio"
    assert exported["replication_contract"]["expected_train_mass_breast_crop_fraction"] == 0.50
    assert "expected_train_crop_positive_fraction" not in exported["replication_contract"]
    assert (
        "require_training_negative_crops_from_mass_negative_images"
        not in exported["replication_contract"]
    )
    assert (
        "require_training_negative_crops_from_mass_negative_breasts"
        not in exported["replication_contract"]
    )


def test_streamlit_gui_can_override_each_split_breast_filter_independently() -> None:
    config = _improved()
    selection = {
        "train": {"mode": "source_breast_ratio", "target_positive_ratio": 0.50},
        "val": {"mode": "all", "target_positive_ratio": 0.50},
        "test": {"mode": "all", "target_positive_ratio": 0.50},
    }
    exported = _build_gui_export_config(
        cfg=config,
        output_root=Path("/mnt/t9/vindr-data/streamlit-independent-mask-controls"),
        clean_output=True,
        selected_vendors=[],
        deterministic_selection=selection,
        split_crop_modes={split: "deterministic" for split in ("train", "val", "test")},
        save_square=True,
        save_baseline=False,
        crop_controls={
            "crop_size": 640,
            "stride": 512,
            "edge_policy": "edge_align",
            "require_foreground": True,
            "min_foreground_fraction": 0.10,
            "foreground_threshold": None,
            "split_breast_filters": {
                "train": {"enabled": True, "minimum": 0.10},
                "val": {"enabled": False, "minimum": 0.20},
                "test": {"enabled": True, "minimum": 0.30},
            },
            "crop_options": {},
        },
        pipeline=config["image_export"]["custom_channel_pipeline"],
    )
    square = exported["square_crops"]
    assert square["train_require_min_breast_fraction_for_all_crops"] is True
    assert square["train_min_breast_fraction_for_all_crops"] == 0.10
    assert square["val_require_min_breast_fraction_for_all_crops"] is False
    assert square["val_min_breast_fraction_for_all_crops"] == 0.20
    assert square["test_require_min_breast_fraction_for_all_crops"] is True
    assert square["test_min_breast_fraction_for_all_crops"] == 0.30
