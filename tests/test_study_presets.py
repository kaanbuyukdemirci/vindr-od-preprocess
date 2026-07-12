from __future__ import annotations

from collections import Counter

import numpy as np

from vindr_mammo.export import (
    _deterministic_selection_mode,
    _select_negative_fraction_candidates,
    make_train_val_test_split,
)
from vindr_mammo.dash_app import _config_control_outputs, _config_control_values
from vindr_mammo.presets import PAPER_22_PRESET_KEY, apply_study_preset


def test_paper_22_preset_preserves_data_parent_and_sets_reported_patch_recipe() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/old-name"}},
        PAPER_22_PRESET_KEY,
    )

    assert config["paths"] == {
        "data_root": "/data",
        "output_root": "/exports/preprocessed-vindr-paper22-v2",
    }
    assert config["visualizations"]["output_dir"] == "/exports/preprocessed-vindr-paper22-v2/visualizations"
    assert config["gui"]["filter_split"] == "all"
    assert config["gui"]["filter_positive"] == "positive only"
    assert config["gui"]["filter_vendor_mode"] == "all vendors"
    assert config["image"]["use_voi_lut"] is True
    assert config["preprocess"]["invert_to_black_background"] is True
    assert config["preprocess"]["crop_breast"] is False
    assert config["preprocess"]["mask_outside_breast"] is True
    assert config["square_crops"]["crop_size"] == 640
    assert config["square_crops"]["stride"] == 512
    assert config["square_crops"]["train_deterministic_selection_mode"] == "negative_fraction"
    assert config["square_crops"]["train_deterministic_negative_keep_fraction"] == 0.20
    assert config["square_crops"]["negative_min_foreground_fraction"] == 0.05
    assert config["square_crops"]["val_deterministic_selection_mode"] == "all"
    assert config["square_crops"]["test_deterministic_selection_mode"] == "all"
    assert config["source_cohort"] == {"finding_category": "Mass", "positive_images_only": True}
    assert config["splits"]["validation_study_count"] == 71
    assert config["splits"]["validation_image_count"] == 136
    assert config["replication_contract"]["expected_source_annotations"]["test"] == 237


def test_negative_fraction_keeps_all_positives_and_twenty_percent_of_negatives() -> None:
    crop_config = {
        "train_deterministic_selection_mode": "negative_fraction",
        "train_deterministic_negative_keep_fraction": 0.20,
    }
    candidates = [
        ({"image_id": f"positive-{i}"}, (i, 0, i + 1, 1), {"is_positive_window": 1})
        for i in range(3)
    ]
    candidates.extend(
        ({"image_id": f"negative-{i}"}, (i, 1, i + 1, 2), {"is_positive_window": 0})
        for i in range(10)
    )

    selected = _select_negative_fraction_candidates(
        candidates,
        crop_config,
        "train",
        np.random.default_rng(123),
    )

    assert _deterministic_selection_mode(crop_config, "train") == "negative_fraction"
    assert sum(item[2]["is_positive_window"] for item in selected) == 3
    assert len(selected) == 5
    assert {item[2]["negative_selected_count"] for item in selected} == {2}


def test_paper_22_preset_clears_conflicting_inherited_export_settings() -> None:
    config = apply_study_preset(
        {
            "paths": {"data_root": "/data", "output_root": "/exports/hostile"},
            "splits": {"seed": 999, "validation_study_count": 1, "validation_image_count": 1},
            "vendor_filter": {"enabled": True, "include_vendors": ["Hidden vendor"]},
            "square_crops": {
                "train_deterministic_require_foreground": False,
                "train_deterministic_min_foreground_fraction": 0.91,
                "train_negative_require_foreground": False,
                "train_deterministic_max_windows_per_image": 1,
            },
        },
        PAPER_22_PRESET_KEY,
    )

    assert config["paths"]["data_root"] == "/data"
    assert config["vendor_filter"] == {"enabled": False, "include_vendors": []}
    assert config["splits"]["seed"] == 123
    assert config["splits"]["validation_study_count"] == 71
    assert config["splits"]["validation_image_count"] == 136
    assert config["square_crops"]["train_deterministic_require_foreground"] is True
    assert config["square_crops"]["train_deterministic_min_foreground_fraction"] == 0.05
    assert config["square_crops"]["train_negative_require_foreground"] is True
    assert config["square_crops"]["train_deterministic_max_windows_per_image"] is None


def test_paper_22_preset_synchronizes_every_visible_cross_section_control() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/old-name"}},
        PAPER_22_PRESET_KEY,
    )
    values = _config_control_values(config)
    controls = {
        output.component_id: value
        for output, value in zip(_config_control_outputs(), values, strict=True)
    }

    assert controls["filter-split"] == "all"
    assert controls["filter-positive"] == "positive only"
    assert controls["filter-vendor-mode"] == "all vendors"
    assert controls["view-geometry"] == "crop"
    assert controls["pp-invert"] == ["on"]
    assert controls["pp-crop-breast"] == []
    assert controls["pp-mask-outside"] == ["on"]
    assert controls["pp-mirror"] == []
    assert controls["crop-size"] == 640
    assert controls["final-crop-resize"] == 640
    assert controls["crop-stride"] == 512
    assert controls["train-crop-mode"] == "deterministic"
    assert controls["val-crop-mode"] == "deterministic"
    assert controls["test-crop-mode"] == "deterministic"
    assert controls["export-balance-mode"] == "negative_fraction"
    assert controls["export-negative-keep-fraction"] == 0.20
    assert controls["export-parent"] == "/exports"
    assert controls["export-name"] == "preprocessed-vindr-paper22-v2"
    assert controls["saved-root"] == "/exports/preprocessed-vindr-paper22-v2"


def test_birads_stratification_is_study_level_and_preserves_official_test() -> None:
    records = []
    for birads in ["1", "2", "3", "4"]:
        for patient in range(10):
            study_id = f"{birads}-{patient}"
            for view in range(4):
                records.append({
                    "study_id": study_id,
                    "image_id": f"{study_id}-{view}",
                    "split": "training",
                    "breast_birads": birads,
                })
    for view in range(4):
        records.append({
            "study_id": "official-test",
            "image_id": f"test-{view}",
            "split": "test",
            "breast_birads": "4",
        })

    splits, _table = make_train_val_test_split(
        records,
        val_fraction=0.20,
        seed=123,
        stratify_by_birads=True,
    )

    train_ids = {record["study_id"] for record in splits["train"]}
    val_ids = {record["study_id"] for record in splits["val"]}
    assert not train_ids & val_ids
    assert {record["study_id"] for record in splits["test"]} == {"official-test"}
    val_strata = Counter(
        next(record["breast_birads"] for record in splits["val"] if record["study_id"] == study_id)
        for study_id in val_ids
    )
    assert val_strata == {"1": 2, "2": 2, "3": 2, "4": 2}
