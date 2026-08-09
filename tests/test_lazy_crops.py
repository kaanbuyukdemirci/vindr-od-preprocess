from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from vindr_mammo.dash_app import _lazy_crop_config_from_controls, _lazy_crop_controls_dash
from vindr_mammo.lazy_crops import (
    default_lazy_crop_config,
    estimate_lazy_crop_rows,
    extract_complete_lazy_crop_family,
    extract_lazy_crop_manifests,
    scan_lazy_crop_source,
)
from vindr_mammo.presets import DEFAULT_RESEARCH_DATASET_PRESET_KEY, apply_study_preset


def _write_source_export(tmp_path: Path) -> Path:
    root = tmp_path / "preprocessed-vindr-default-research-dataset-v1"
    metadata = root / "square_crops" / "metadata"
    metadata.mkdir(parents=True)

    sources = [
        ("train", "study-pos", "image-pos", 6, 4, "study-pos:L", 1),
        ("train", "study-neg", "image-neg", 6, 4, "study-neg:L", 0),
        ("val", "study-val", "image-val", 5, 4, "study-val:L", 1),
        ("test", "study-test", "image-test", 3, 3, "study-test:L", 0),
    ]
    manifest_rows = []
    sample_rows = []
    for split, study_id, image_id, width, height, breast_key, breast_has_mass in sources:
        stem = f"{study_id}__{image_id}"
        manifest_rows.extend(
            [
                {
                    "variant": "original",
                    "split": split,
                    "source_image_id": image_id,
                    "source_study_id": study_id,
                    "image_path": f"whole_images_original/{split}/{stem}.png",
                    "label_path": "",
                    "annotation_path": "",
                    "width": width,
                    "height": height,
                    "num_annotations": breast_has_mass,
                },
                {
                    "variant": "resized",
                    "split": split,
                    "source_image_id": image_id,
                    "source_study_id": study_id,
                    "image_path": f"whole_images/{split}/{stem}.png",
                    "label_path": "",
                    "annotation_path": "",
                    "width": 4,
                    "height": 4,
                    "num_annotations": breast_has_mass,
                },
            ]
        )
        sample_rows.append(
            {
                "source_image_id": image_id,
                "source_breast_key": breast_key,
                "source_breast_has_mass": breast_has_mass,
                "source_preprocessing_mirrored": False,
                "source_coordinate_space": "fixed_preprocessed",
                "paired_whole_original_image": f"whole_images_original/{split}/{stem}.png",
                "paired_whole_original_float32_image": "",
                "paired_whole_image": f"whole_images/{split}/{stem}.png",
                "paired_whole_float32_image": f"float32/whole_images/{split}/{stem}.pt",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(metadata / "whole_image_manifest.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(metadata / "samples_metadata_flat.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "original",
                "split": "train",
                "source_image_id": "image-pos",
                "source_study_id": "study-pos",
                "annotation_index": 0,
                "source_annotation_id": "mass-train",
                "source_bbox_xyxy": "[1, 1, 3, 3]",
                "bbox_xyxy": "[1, 1, 3, 3]",
            },
            {
                "variant": "original",
                "split": "val",
                "source_image_id": "image-val",
                "source_study_id": "study-val",
                "annotation_index": 0,
                "source_annotation_id": "mass-val",
                "source_bbox_xyxy": "[4, 1, 5, 3]",
                "bbox_xyxy": "[4, 1, 5, 3]",
            },
        ]
    ).to_csv(metadata / "whole_image_annotations.csv", index=False)
    return root


def test_metadata_only_lazy_crop_export_writes_stable_split_contract(tmp_path: Path) -> None:
    root = _write_source_export(tmp_path)
    cfg = default_lazy_crop_config(
        root,
        window_size=4,
        stride=2,
        min_box_visibility=0.50,
        train_positive_fraction=0.50,
        train_min_source_extent_fraction=0.0,
        eval_min_source_extent_fraction=0.80,
        seed=7,
    )
    result = extract_lazy_crop_manifests(cfg)

    output = Path(result["output_root"])
    train = pd.read_csv(output / "lazy_crop_manifest_train.csv")
    train_annotations = pd.read_csv(output / "lazy_crop_annotations_train.csv")
    val = pd.read_csv(output / "lazy_crop_manifest_val.csv")
    val_annotations = pd.read_csv(output / "lazy_crop_annotations_val.csv")
    test = pd.read_csv(output / "lazy_crop_manifest_test.csv")

    assert result["decoded_source_images"] == 0
    assert len(train) == 4
    assert int(train["is_mass_positive"].sum()) == 2
    assert int((train["is_mass_positive"] == 0).sum()) == 2
    assert set(train.loc[train["is_mass_positive"] == 0, "source_image_id"]) == {"image-neg"}
    assert set(train["window_size"]) == {4}
    assert set(train["stride"]) == {2}
    assert set(train["edge_policy"]) == {"regular_stride_pad"}
    assert len(train_annotations) == 2
    assert set(train_annotations["visible_fraction"].round(2)) == {0.5, 1.0}
    assert set(train_annotations["crop_bbox_x0"]) == {0.0, 1.0}

    assert len(val) == 2
    padded_positive = val[val["is_mass_positive"] == 1].iloc[0]
    assert padded_positive.source_extent_fraction == 0.75
    assert padded_positive.positive_bypassed_source_extent_filter == 1
    assert padded_positive.pad_right == 1
    assert len(val_annotations) == 1
    # The 3x3 source occupies 9/16 of the padded 4x4 window and is correctly
    # removed by the strict 0.80 evaluation threshold.
    assert test.empty

    manifest = json.loads((output / "lazy_crop_manifest.json").read_text())
    assert manifest["decoded_source_images"] == 0
    assert manifest["selection"]["train_positive_fraction_target"] == 0.5
    resolved = yaml.safe_load((output / "lazy_crop_config_resolved.yaml").read_text())
    assert resolved["runtime"]["decode_source_images"] is False
    readme = (output / "README.md").read_text()
    assert "no crop PNGs" in readme
    assert "source_extent_fraction" in readme
    assert "context_resized_float32_path" in readme
    assert "lazy_crop_manifest_train.csv" in readme
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.pt"))


def test_lazy_crop_scan_and_estimate_never_require_source_files(tmp_path: Path) -> None:
    root = _write_source_export(tmp_path)
    scan = scan_lazy_crop_source(root)
    assert scan == {
        "dataset_root": str(root),
        "square_crops_root": str(root / "square_crops"),
        "source_images": 4,
        "source_images_by_split": {"train": 2, "val": 1, "test": 1},
        "source_mass_annotations": 2,
        "same_geometry_float32_sources": 0,
        "png_sources": 4,
        "decoded_images": 0,
    }
    estimate = estimate_lazy_crop_rows(
        default_lazy_crop_config(root, window_size=4, stride=2)
    )
    assert estimate["complete_grid_rows_by_split"] == {
        "train": 4,
        "val": 2,
        "test": 1,
    }
    assert estimate["decoded_images"] == 0


def test_complete_lazy_crop_family_writes_unfiltered_grids_at_multiple_strides(
    tmp_path: Path,
) -> None:
    root = _write_source_export(tmp_path)
    results = extract_complete_lazy_crop_family(
        root,
        window_size=4,
        strides=[2, 4],
        min_box_visibility=0.5,
    )

    assert set(results) == {2, 4}
    for stride, result in results.items():
        output = Path(result["output_root"])
        resolved = yaml.safe_load(
            (output / "lazy_crop_config_resolved.yaml").read_text()
        )
        assert resolved["geometry"] == {
            "window_size": 4,
            "stride": stride,
            "edge_policy": "regular_stride_pad",
        }
        assert resolved["filters"]["source_extent_filter_enabled"] is False
        assert resolved["sampling"]["train_selection_policy"] == "all_eligible_windows"
        train = pd.read_csv(output / "lazy_crop_manifest_train.csv")
        assert set(train["source_image_id"]) == {"image-pos", "image-neg"}
        assert set(train["source_extent_filter_enabled"]) == {0}


def test_grouped_dataset_supports_multiple_window_sizes_and_resolution_contexts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "grouped-research"
    (root / "images" / "original" / "train").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    manifest_rows = []
    for variant, image_path, width, height in [
        ("original", "images/original/train/source.png", 8, 6),
        ("resized_8x8", "images/resized/8x8/train/source.png", 8, 8),
        ("resized_4x4", "images/resized/4x4/train/source.png", 4, 4),
    ]:
        manifest_rows.append({
            "variant": variant,
            "split": "train",
            "source_image_id": "source",
            "source_study_id": "study",
            "image_path": image_path,
            "float32_path": image_path.replace("images/", "images/float32/").replace(".png", ".pt"),
            "width": width,
            "height": height,
            "source_breast_key": "study:L",
            "source_breast_has_mass": 1,
        })
    pd.DataFrame(manifest_rows).to_csv(
        root / "metadata" / "whole_image_manifest.csv", index=False
    )
    pd.DataFrame([{
        "variant": "original",
        "split": "train",
        "source_image_id": "source",
        "source_study_id": "study",
        "annotation_index": 0,
        "source_annotation_id": "mass",
        "source_bbox_xyxy": "[1, 1, 3, 3]",
        "bbox_xyxy": "[1, 1, 3, 3]",
    }]).to_csv(root / "annotations" / "whole_image_annotations.csv", index=False)

    results = extract_complete_lazy_crop_family(
        root,
        grids=[
            {"window_size": 8, "stride": 2},
            {"window_size": 4, "stride": 1},
        ],
        min_box_visibility=0.5,
    )

    assert set(results) == {"window_8_stride_2", "window_4_stride_1"}
    for window, stride in [(8, 2), (4, 1)]:
        output = root / "annotations" / "windows" / f"window_{window}_stride_{stride}"
        assert output.is_dir()
        frame = pd.read_csv(output / "lazy_crop_manifest_train.csv")
        assert set(frame["window_size"]) == {window}
        assert set(frame["stride"]) == {stride}
        assert set(frame["context_resized_png_path"]) == {
            f"images/resized/{window}x{window}/train/source.png"
        }
    assert not (root / "lazy_crop_manifests").exists()


def test_research_preset_and_lazy_crop_gui_default_to_1024_stride128(tmp_path: Path) -> None:
    preset = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": str(tmp_path / "old")}},
        DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    )
    assert preset["square_crops"]["crop_size"] == 1024
    assert preset["square_crops"]["stride"] == 128

    controls = _lazy_crop_controls_dash(preset)
    by_id: dict[str, object] = {}

    def visit(component: object) -> None:
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str):
            by_id[component_id] = component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)
        elif children is not None and not isinstance(children, (str, int, float)):
            visit(children)

    visit(controls)
    assert getattr(by_id["lazy-crop-window-size"], "value") == 1024
    assert getattr(by_id["lazy-crop-stride"], "value") == 128
    assert getattr(by_id["lazy-crop-grids"], "value") == (
        "1024:128, 1024:256, 1024:512, 640:160"
    )


def test_lazy_crop_gui_config_preserves_filter_and_balance_controls(tmp_path: Path) -> None:
    root = _write_source_export(tmp_path)
    output = tmp_path / "custom-manifests"
    cfg = _lazy_crop_config_from_controls(
        dataset_root=str(root),
        output_root=str(output),
        window_size=1024,
        stride=128,
        min_box_visibility=0.05,
        train_min_extent=0.10,
        eval_min_extent=0.05,
        preserve_positives=["on"],
        positive_fraction=0.50,
        clean_negative_breasts=["on"],
        seed=123,
        overwrite=["on"],
    )
    assert cfg["paths"]["output_root"] == str(output)
    assert cfg["geometry"] == {
        "window_size": 1024,
        "stride": 128,
        "edge_policy": "regular_stride_pad",
    }
    assert cfg["annotations"]["min_box_visibility"] == 0.05
    assert cfg["filters"]["preserve_positive_windows_below_threshold"] is True
    assert cfg["sampling"]["train_require_mass_negative_breasts"] is True
    assert cfg["sampling"]["train_positive_fraction"] == 0.5
    assert cfg["runtime"] == {"overwrite": True, "decode_source_images": False}
