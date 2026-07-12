from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


PAPER_22_PRESET_KEY = "bulatovic_yolov8_patched_inference_vindr"


STUDY_PRESETS: dict[str, dict[str, Any]] = {
    PAPER_22_PRESET_KEY: {
        "label": "Bulatović et al. — YOLOv8 patched inference (VinDr-Mammo)",
        "description": (
            "Reproduces the VinDr-Mammo data pipeline reported in “Refining YOLOv8 for Full Field "
            "Digital Mammograms”: MONOCHROME2-style inversion, DICOM VOI windowing, background "
            "removal, 8-bit replicated grayscale PNG, and 640 px patches with 20% overlap. Training "
            "keeps every annotated patch and a seeded 20% sample of negative patch candidates. The "
            "official test cohort is preserved and the undisclosed train/validation IDs are reproduced "
            "as a deterministic count-matched split. The dataset folder is preprocessed-vindr-paper22-v2."
        ),
        "output_folder_name": "preprocessed-vindr-paper22-v2",
        # These sections affect dataset identity or pixels and must not inherit
        # stale GUI/YAML overrides. Paths and runtime preferences remain intact.
        "replace_sections": [
            "gui",
            "image",
            "preprocess",
            "image_export",
            "histogram_equalization",
            "preserved_16bit",
            "metadata",
            "source_cohort",
            "splits",
            "vendor_filter",
            "export",
            "square_crops",
            "baseline_uncropped",
            "crop_annotation_policy",
            "replication_contract",
            "study_preset_provenance",
        ],
        "config_patch": {
            "study_preset_provenance": {
                "preset_key": PAPER_22_PRESET_KEY,
                "preset_version": 2,
                "paper_title": "Refining YOLOv8 for Full Field Digital Mammograms: Improving Small Object Detection through Resolution-Preserving Patched Inference",
                "doi": "10.1109/RCAR65431.2025.11139462",
                "replication_scope": "VinDr-Mammo Mass detector dataset preprocessing",
                "split_identity": "deterministic_count_matched; author train/validation IDs and seed are not published",
                "assumptions": {
                    "split_seed": 123,
                    "partial_box_min_visibility": 0.30,
                    "foreground_min_fraction": 0.05,
                    "negative_sampling_scope": "global training split",
                    "negative_sampling_rounding": "round(candidate_count * 0.20)",
                    "edge_policy": "regular stride-512 starts plus final edge-aligned start",
                    "rgb_encoding": "single grayscale channel replicated identically into RGB",
                    "intensity_scaling": "per-image minmax after VOI/polarity processing",
                    "dicom_transform_order": "modality LUT, VOI LUT/windowing, MONOCHROME1 inversion, minmax",
                },
            },
            "gui": {
                "filter_split": "all",
                "filter_positive": "positive only",
                "filter_vendor_mode": "all vendors",
            },
            "image": {
                "normalize": "minmax",
                "percentile_range": [0.5, 99.5],
                "use_voi_lut": True,
                "strict_voi_lut": True,
            },
            "preprocess": {
                "invert_to_black_background": True,
                # The paper removes artifacts/background but does not report a
                # breast-bounding-box crop or canonical left/right mirroring.
                "crop_breast": False,
                "mask_outside_breast": True,
                "mirror_right_to_left": False,
                "crop_threshold": None,
                "breast_mask_method": "largest_connected_tissue",
                "breast_mask_close_kernel": 21,
                "breast_mask_open_kernel": 7,
                "breast_mask_fill_holes": True,
                "breast_mask_keep_largest_component": True,
                "min_component_area_fraction": 0.001,
                "retain_breast_mask_for_export": True,
            },
            "image_export": {
                "rgb_scheme": "custom_channel_pipeline",
                "custom_channel_pipeline": {
                    channel: {"source": "current_crop", "steps": []}
                    for channel in ["R", "G", "B"]
                },
            },
            "histogram_equalization": {"enabled": False},
            "preserved_16bit": {"save": False},
            "metadata": {
                "save_full_source_csvs": True,
                "include_dicom_meta": True,
            },
            "source_cohort": {
                "finding_category": "Mass",
                "positive_images_only": True,
            },
            "splits": {
                # The official VinDr test set remains untouched. The remaining
                # mass-positive studies are split at study/patient level within
                # BI-RADS strata. Exact author IDs/seed were not published.
                "validation_study_count": 71,
                "validation_image_count": 136,
                "seed": 123,
                "stratify_by_birads": True,
            },
            "vendor_filter": {
                "enabled": False,
                "include_vendors": [],
            },
            "export": {
                "clean_output_root": True,
                "save_square_crops": True,
                "save_baseline_uncropped": False,
                "save_empty_label_files": True,
            },
            "square_crops": {
                "crop_size": 640,
                # 640 * (1 - 0.20 overlap) = 512.
                "stride": 512,
                "train_crop_mode": "deterministic",
                "val_crop_mode": "deterministic",
                "test_crop_mode": "deterministic",
                "deterministic_selection_mode": "negative_fraction",
                "deterministic_negative_keep_fraction": 0.20,
                "train_deterministic_selection_mode": "negative_fraction",
                "train_deterministic_negative_keep_fraction": 0.20,
                "val_deterministic_selection_mode": "all",
                "test_deterministic_selection_mode": "all",
                "train_deterministic_include_empty": True,
                "val_deterministic_include_empty": True,
                "test_deterministic_include_empty": True,
                "train_deterministic_max_windows_per_image": None,
                "val_deterministic_max_windows_per_image": None,
                "test_deterministic_max_windows_per_image": None,
                "deterministic_require_foreground": True,
                "deterministic_min_foreground_fraction": 0.05,
                "deterministic_foreground_threshold": None,
                "train_deterministic_require_foreground": True,
                "val_deterministic_require_foreground": True,
                "test_deterministic_require_foreground": True,
                "train_deterministic_min_foreground_fraction": 0.05,
                "val_deterministic_min_foreground_fraction": 0.05,
                "test_deterministic_min_foreground_fraction": 0.05,
                # Ambiguous partial lesions must never enter the sampled training
                # negatives. Validation/test are inference grids, not training
                # negatives, so every non-background tile remains available.
                "train_require_clean_negative_windows": True,
                "val_require_clean_negative_windows": False,
                "test_require_clean_negative_windows": False,
                "negative_require_foreground": True,
                "negative_min_foreground_fraction": 0.05,
                "train_negative_require_foreground": True,
                "val_negative_require_foreground": True,
                "test_negative_require_foreground": True,
                "train_negative_min_foreground_fraction": 0.05,
                "val_negative_min_foreground_fraction": 0.05,
                "test_negative_min_foreground_fraction": 0.05,
                "require_foreground_for_empty_crops": True,
                "min_foreground_fraction": 0.05,
                "pad_if_needed": True,
                "pad_value": 0.0,
                "deduplicate_windows_per_source": True,
                "seed": 123,
            },
            "crop_annotation_policy": {
                "allow_partial_annotations": True,
                "min_box_visibility": 0.30,
                "reject_partial_windows": False,
                "negative_max_box_visibility": 0.0,
            },
            "replication_contract": {
                "enabled": True,
                "strict": True,
                "name": "paper22_vindr_mass_v2",
                "preserve_official_test": True,
                "require_positive_source_images": True,
                "require_all_source_annotations_represented": True,
                "require_source_annotation_ids": True,
                "min_inference_grid_fraction": 0.30,
                "expected_source_images": {"train": 758, "val": 136, "test": 219},
                "expected_source_studies": {"train": 398, "val": 71, "test": 115},
                "expected_source_annotations": {"test": 237},
            },
        },
    },
}


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Return a copied recursive merge without mutating either input."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_study_preset(config: dict[str, Any], preset_key: str) -> dict[str, Any]:
    """Apply a study preset while preserving the data path and output parent."""
    try:
        preset = STUDY_PRESETS[str(preset_key)]
    except KeyError as exc:
        raise ValueError(f"Unknown study preset: {preset_key!r}") from exc
    out = copy.deepcopy(config)
    for section in preset.get("replace_sections", []):
        out.pop(str(section), None)
    out = deep_merge(out, preset["config_patch"])
    folder_name = str(preset.get("output_folder_name", "")).strip()
    if folder_name:
        current_root = Path(str(out.get("paths", {}).get("output_root", folder_name)))
        output_root = current_root.parent / folder_name
        out.setdefault("paths", {})["output_root"] = str(output_root)
        out.setdefault("visualizations", {})["output_dir"] = str(output_root / "visualizations")
    return out
