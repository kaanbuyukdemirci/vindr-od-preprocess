from __future__ import annotations

import numpy as np
import torch

from vindr_mammo.dataset import VindrMammoDataset
from vindr_mammo.export import (
    _append_coco_annotations,
    _fixed_preprocessed_boxes_to_original,
    _make_rgb_image,
    _validate_square_crop_contract,
    _windows_for_export_split,
    make_train_val_test_split,
)
from vindr_mammo.preprocessing import make_preprocess_options
from vindr_mammo.presets import (
    PAPER_22_IMPROVED_PRESET_KEY,
    PAPER_22_PRESET_KEY,
    apply_study_preset,
)


def _paper_crop_config() -> tuple[dict, dict]:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/base"}},
        PAPER_22_PRESET_KEY,
    )
    return config["square_crops"], config["crop_annotation_policy"]


def test_mask_only_preprocessing_changes_pixels_without_changing_geometry() -> None:
    dataset = object.__new__(VindrMammoDataset)
    dataset.preprocess_options = make_preprocess_options({
        "crop_breast": False,
        "mask_outside_breast": True,
        "mirror_right_to_left": False,
        "breast_mask_open_kernel": 1,
        "breast_mask_close_kernel": 1,
        "breast_mask_fill_holes": False,
        "breast_mask_keep_largest_component": True,
        "retain_breast_mask_for_export": True,
    })
    image = torch.zeros((1, 128, 128), dtype=torch.float32)
    image[:, 20:108, 20:108] = 1.0
    image[:, 2:5, 2:5] = 1.0  # isolated acquisition artifact
    box = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    target = {
        "boxes": box.clone(),
        "mass": {
            "boxes": box.clone(),
            "labels": torch.ones((1,), dtype=torch.int64),
            "finding_birads": ["BI-RADS 4"],
            "finding_birads_ids": [4],
            "findings": [{"source_annotation_id": 7}],
        },
    }

    processed, processed_target = dataset._apply_preprocessing(image, target)

    assert tuple(processed.shape) == (1, 128, 128)
    assert torch.equal(processed_target["boxes"], box)
    assert torch.equal(processed_target["mass"]["boxes"], box)
    assert float(processed[0, 3, 3]) == 0.0
    assert float(processed[0, 50, 50]) == 1.0
    assert processed_target["preprocessing"]["crop_box_xyxy"] is None
    assert processed_target["_foreground_mask"].shape == (128, 128)


def test_positive_window_bypasses_zero_foreground_mask() -> None:
    crop_cfg, crop_policy = _paper_crop_config()
    image = torch.zeros((1, 640, 640), dtype=torch.float32)
    boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    diagnostics: dict = {}

    windows = _windows_for_export_split(
        split_name="train",
        image_width=640,
        image_height=640,
        image_tensor=image,
        mass_boxes=boxes,
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(123),
        foreground_mask=np.zeros((640, 640), dtype=bool),
        diagnostics=diagnostics,
    )

    assert [window for window, _info in windows] == [(0, 0, 640, 640)]
    assert windows[0][1]["is_positive_window"] == 1
    assert windows[0][1]["foreground_fraction"] == 0.0
    assert diagnostics["positive_candidate_windows"] == 1


def test_named_paper22_foreground_regression_windows_are_retained() -> None:
    crop_cfg, crop_policy = _paper_crop_config()
    box = torch.tensor([[2089.51, 1072.55, 2533.87, 1550.61]])
    windows = _windows_for_export_split(
        split_name="train",
        image_width=2800,
        image_height=3518,
        image_tensor=torch.zeros((1, 1, 1), dtype=torch.float32),
        mass_boxes=box,
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(123),
        foreground_mask=np.zeros((3518, 2800), dtype=bool),
    )
    keys = {window for window, info in windows if info["is_positive_window"] == 1}

    assert (2048, 1024, 2688, 1664) in keys
    assert (2160, 1024, 2800, 1664) in keys


def test_training_excludes_ambiguous_partial_lesion_from_negative_pool() -> None:
    crop_cfg, crop_policy = _paper_crop_config()
    image = torch.ones((1, 640, 1280), dtype=torch.float32)
    boxes = torch.tensor([[600.0, 100.0, 740.0, 200.0]])
    mask = np.ones((640, 1280), dtype=bool)

    train_windows = _windows_for_export_split(
        split_name="train",
        image_width=1280,
        image_height=640,
        image_tensor=image,
        mass_boxes=boxes,
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(123),
        foreground_mask=mask,
    )
    val_windows = _windows_for_export_split(
        split_name="val",
        image_width=1280,
        image_height=640,
        image_tensor=image,
        mass_boxes=boxes,
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(123),
        foreground_mask=mask,
    )

    assert [window[0] for window, _info in train_windows] == [512, 640]
    assert [window[0] for window, _info in val_windows] == [0, 512, 640]
    assert train_windows[0][1]["is_positive_window"] == 1


def test_planner_slices_supplied_full_source_mask() -> None:
    crop_cfg, crop_policy = _paper_crop_config()
    windows = _windows_for_export_split(
        split_name="val",
        image_width=640,
        image_height=640,
        image_tensor=torch.ones((1, 640, 640), dtype=torch.float32),
        mass_boxes=torch.zeros((0, 4), dtype=torch.float32),
        crop_options=crop_policy,
        crop_cfg=crop_cfg,
        rng=np.random.default_rng(123),
        foreground_mask=np.zeros((640, 640), dtype=bool),
    )
    assert windows == []


def test_count_matched_split_is_exact_study_level_and_reproducible() -> None:
    records = []
    for index, size in enumerate([1, 1, 2, 2, 2, 3, 3, 4]):
        for view in range(size):
            records.append({
                "study_id": f"study-{index}",
                "image_id": f"study-{index}-image-{view}",
                "split": "training",
                "breast_birads": f"BI-RADS {3 + index % 3}",
            })
    records.append({
        "study_id": "official-test",
        "image_id": "official-test-image",
        "split": "test",
        "breast_birads": "BI-RADS 4",
    })

    first, _ = make_train_val_test_split(
        records,
        val_fraction=0.15,
        seed=123,
        stratify_by_birads=True,
        validation_study_count=3,
        validation_image_count=6,
    )
    second, _ = make_train_val_test_split(
        records,
        val_fraction=0.99,
        seed=123,
        stratify_by_birads=True,
        validation_study_count=3,
        validation_image_count=6,
    )

    assert len(first["val"]) == 6
    assert len({record["study_id"] for record in first["val"]}) == 3
    assert {record["image_id"] for record in first["val"]} == {
        record["image_id"] for record in second["val"]
    }
    assert {record["study_id"] for record in first["test"]} == {"official-test"}


def test_coco_annotations_retain_stable_source_identity() -> None:
    coco = {"images": [{"id": 5}], "annotations": []}
    count = _append_coco_annotations(
        coco,
        image_id=5,
        start_ann_id=10,
        boxes=torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        source_annotation_ids=[42],
        source_annotation_rows=[44],
        source_boxes=torch.tensor([[101.0, 202.0, 111.0, 222.0]]),
        source_original_boxes=torch.tensor([[301.0, 402.0, 311.0, 422.0]]),
    )

    assert count == 1
    assert coco["annotations"][0]["source_annotation_id"] == 42
    assert coco["annotations"][0]["source_annotation_row"] == 44
    assert coco["annotations"][0]["source_bbox_xyxy"] == [101.0, 202.0, 111.0, 222.0]
    assert coco["annotations"][0]["source_bbox_coordinate_space"] == "fixed_preprocessed"
    assert coco["annotations"][0]["source_bbox_original_xyxy"] == [301.0, 402.0, 311.0, 422.0]
    assert coco["annotations"][0]["source_bbox_original_coordinate_space"] == "original_dicom"


def test_mirrored_fixed_boxes_are_restored_to_original_dicom_coordinates() -> None:
    restored = _fixed_preprocessed_boxes_to_original(
        torch.tensor([[20.0, 30.0, 40.0, 70.0]]),
        {
            "mirrored": True,
            "processed_shape": (100, 120),
            "crop_box_xyxy": (10, 15, 130, 115),
        },
    )

    assert torch.equal(restored, torch.tensor([[90.0, 45.0, 110.0, 85.0]]))


def test_source_identity_contract_catches_duplicate_instances_hiding_missing_gt() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/base"}},
        PAPER_22_PRESET_KEY,
    )
    config["replication_contract"]["strict"] = False
    config["replication_contract"]["min_inference_grid_fraction"] = 0.0
    source_debug = {
        ("train", "image-a"): {
            "n_source_mass_boxes": 2,
            "_included_annotation_indices": {100},
            "saved_positive_crops": 10,
            "saved_negative_crops": 0,
            "saved_crops": 10,
            "candidate_windows": 10,
            "positive_candidate_windows": 10,
            "complete_grid_windows": 10,
        },
    }
    report = _validate_square_crop_contract(
        split_records={"train": [], "val": [], "test": []},
        coco_by_split={split: {"images": [], "annotations": []} for split in ["train", "val", "test"]},
        source_debug=source_debug,
        candidate_positive_window_keys=set(),
        saved_positive_window_keys=set(),
        crop_cfg=config["square_crops"],
        config=config,
    )

    assert report["status"] == "fail"
    assert any("represented 1/2 source annotations" in error for error in report["errors"])


def test_custom_contract_checks_negative_sources_against_independent_provenance() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/base"}},
        PAPER_22_IMPROVED_PRESET_KEY,
    )
    config["replication_contract"]["strict"] = False
    source_debug = {
        ("train", "mass-in-this-view"): {
            "source_image_has_mass": 1,
            "source_breast_has_mass": 1,
            "n_source_mass_boxes": 1,
            "_included_annotation_indices": set(),
        },
        ("train", "mass-in-paired-view"): {
            "source_image_has_mass": 0,
            "source_breast_has_mass": 1,
            "n_source_mass_boxes": 0,
            "_included_annotation_indices": set(),
        },
    }
    train_images = [
        {
            "id": index,
            "file_name": f"negative-{index}.png",
            "width": 640,
            "height": 640,
            "crop_window_xyxy": [0, 0, 640, 640],
            "breast_fraction": 0.50,
            "source_image_id": source_image_id,
            # Simulate the exact v7 failure: self-reported metadata says clean.
            "source_image_has_mass": 0,
            "source_breast_has_mass": 0,
        }
        for index, source_image_id in enumerate(
            ["mass-in-this-view", "mass-in-paired-view"],
            start=1,
        )
    ]
    report = _validate_square_crop_contract(
        split_records={
            "train": [
                {"image_id": "mass-in-this-view"},
                {"image_id": "mass-in-paired-view"},
            ],
            "val": [],
            "test": [],
        },
        coco_by_split={
            "train": {"images": train_images, "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []},
        },
        source_debug=source_debug,
        candidate_positive_window_keys=set(),
        saved_positive_window_keys=set(),
        crop_cfg=config["square_crops"],
        config=config,
    )

    source_policy = report["metrics"]["train_negative_crop_source_policy"]
    assert source_policy["required"] == "mass_negative_breasts_only"
    assert source_policy["invalid_source_image_crops"] == 1
    assert source_policy["invalid_source_breast_crops"] == 2
    assert source_policy["invalid_negative_crops"] == 2
    assert source_policy["source_image_metadata_mismatches"] == 1
    assert source_policy["source_breast_metadata_mismatches"] == 2
    assert any(
        "2 empty crops came from breasts with Mass in the source or paired view"
        in error
        for error in report["errors"]
    )


def test_paper_png_recipe_is_exact_replicated_uint8_grayscale() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/base"}},
        PAPER_22_PRESET_KEY,
    )
    rgb, _meta = _make_rgb_image(
        np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
        config,
    )

    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])
    assert rgb[..., 0].tolist() == [[0, 64], [128, 255]]
