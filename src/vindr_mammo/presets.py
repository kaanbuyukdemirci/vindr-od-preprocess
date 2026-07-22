from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


PAPER_22_PRESET_KEY = "bulatovic_yolov8_patched_inference_vindr"
PAPER_22_IMPROVED_PRESET_KEY = "paper22_patient_breast_balanced_v4"
PAPER_69_PRESET_KEY = "bhat_exemplar_med_detr_vindr"
SIMPLE_PRESET_KEY = "simple-preset"


STUDY_PRESETS: dict[str, dict[str, Any]] = {
    PAPER_22_PRESET_KEY: {
        "label": "Paper 22 — closest available reproduction (v2; not exact)",
        "description": (
            "Paper 22 reproduction preset for the VinDr-Mammo data pipeline reported in “Refining YOLOv8 for Full Field "
            "Digital Mammograms”: MONOCHROME2-style inversion, DICOM VOI windowing, background "
            "removal, 8-bit replicated grayscale PNG, and 640 px patches with 20% overlap. Training "
            "keeps every annotated patch and a seeded 20% sample of negative patch candidates. The "
            "official test cohort is preserved and the undisclosed train/validation IDs are reproduced "
            "as a deterministic count-matched split. This is the closest available reproduction, "
            "not an exact copy, because author IDs and several implementation details are unpublished. "
            "The dataset folder is preprocessed-vindr-paper22-v2."
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
                "strategy": "exact_study_count",
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
                "edge_policy": "edge_align",
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
    PAPER_69_PRESET_KEY: {
        "label": "Paper 69 — closest available reproduction (v3; not exact)",
        "description": (
            "Paper 69 reproduction preset for the preprocessing disclosed for Exemplar Med-DETR: preserve the official "
            "VinDr test cohort, keep mammograms at full resolution, and crop only excess "
            "background outside the breast. The paper does not publish its crop code; this preset "
            "uses the cited MammoCLIP public 5-pixel trim, MONOCHROME1 correction, per-image uint8 "
            "scaling, threshold-40 longest-run crop, while deliberately omitting MammoCLIP's resize. "
            "For usable early stopping, 15% of official training studies are held out with seed 123; "
            "the GUI can switch back to strict official train/test membership with no validation. "
            "This is the closest available reproduction, not an exact copy, because the paper does "
            "not publish its crop code or validation identities."
        ),
        "output_folder_name": "preprocessed-vindr-paper69-em-detr-v3",
        "replace_sections": [
            "gui", "image", "preprocess", "image_export", "histogram_equalization",
            "preserved_16bit", "metadata", "source_cohort", "splits", "vendor_filter",
            "export", "square_crops", "baseline_uncropped", "paired_whole_images",
            "crop_annotation_policy", "replication_contract", "study_preset_provenance",
            "training_augmentation",
        ],
        "config_patch": {
            "study_preset_provenance": {
                "preset_key": PAPER_69_PRESET_KEY,
                "preset_version": 3,
                "paper_number": 69,
                "paper_title": "Exemplar Med-DETR: Toward Generalized and Robust Lesion Detection in Mammogram Images and Beyond",
                "doi": "10.1007/978-3-032-04978-0_20",
                "replication_scope": "Published VinDr-Mammo offline preprocessing for the Mass detector",
                "disclosure_limit": (
                    "The paper, supplement, and author feedback do not publish the exact breast-crop "
                    "threshold/margin code, DICOM intensity pipeline, validation IDs, random seeds, "
            "or Stage-II/III background-box coordinates. This is the closest available reproduction, "
            "not an exact copy."
                ),
                "evidence": {
                    "offline": "full resolution; crop excess background outside breast; no downscaling",
                    "online_train_resize": "MMDetection aspect-preserving multiscale widths 480..800 step 32 with 1333 cap",
                    "input": "whole mammogram, not sliding-window lesion crops",
                },
                "assumptions": {
                    "crop_algorithm": "closest public cited implementation: MammoCLIP ExtractBreast",
                    "trim_border_px": 5,
                    "crop_detection_threshold_uint8": 40,
                    "crop_padding_px": 0,
                    "rgb_encoding": "uint8 grayscale replicated into RGB",
                    "validation_split": (
                        "training-oriented assumption: seeded 15% study-level BI-RADS-stratified holdout "
                        "from official training; the paper does not disclose validation IDs or policy"
                    ),
                    "class_scope": "Mass only because this exporter is a one-class mass dataset tool",
                },
            },
            "gui": {
                "filter_split": "all",
                "filter_positive": "all images",
                "filter_vendor_mode": "all vendors",
            },
            "image": {
                "normalize": "none",
                "percentile_range": [0.5, 99.5],
                "use_voi_lut": False,
                "strict_voi_lut": False,
            },
            "preprocess": {
                "invert_to_black_background": True,
                "trim_border_px": 5,
                "intensity_scale_before_geometry": "minmax_uint8",
                "crop_breast": True,
                "mask_outside_breast": False,
                "mirror_right_to_left": False,
                "crop_padding": 0,
                "crop_threshold": None,
                "breast_mask_method": "mammo_clip_contiguous_variance",
                "breast_mask_close_kernel": 0,
                "breast_mask_open_kernel": 0,
                "breast_mask_fill_holes": False,
                "breast_mask_keep_largest_component": False,
                "min_component_area_fraction": 0.0,
                "min_box_visibility_after_crop": 0.0,
                "retain_breast_mask_for_export": False,
            },
            "image_export": {
                "rgb_scheme": "paper69_mammoclip_uint8",
            },
            "histogram_equalization": {"enabled": False, "apply_to": "none"},
            "preserved_16bit": {"save": False},
            "metadata": {"save_full_source_csvs": True, "include_dicom_meta": True},
            "source_cohort": {"finding_category": "Mass", "positive_images_only": False},
            "splits": {
                "strategy": "random_study_fraction",
                "val_fraction_from_training": 0.15,
                "validation_study_count": None,
                "validation_image_count": None,
                "seed": 123,
                "stratify_by_birads": True,
            },
            "vendor_filter": {"enabled": False, "include_vendors": []},
            "export": {
                "clean_output_root": True,
                "save_square_crops": False,
                "save_baseline_uncropped": True,
                "save_empty_label_files": True,
            },
            "baseline_uncropped": {
                "resize_mode": "none",
                "target_width": 0,
                "target_height": 0,
                "pad_value": 0.0,
                "pad_anchor": "left_top",
            },
            "paired_whole_images": {"enabled": False},
            "square_crops": {
                "crop_size": 1024,
                "stride": 512,
                "edge_policy": "edge_align",
                "train_crop_mode": "deterministic",
                "val_crop_mode": "deterministic",
                "test_crop_mode": "deterministic",
                "seed": 123,
            },
            "crop_annotation_policy": {
                "allow_partial_annotations": True,
                "min_box_visibility": 0.0,
                "reject_partial_windows": False,
                "negative_max_box_visibility": 0.0,
            },
            "training_augmentation": {
                "offline": False,
                "framework": "MMDetection",
                "aspect_ratio_preserved": True,
                "multiscale_widths": list(range(480, 801, 32)),
                "long_edge_cap": 1333,
            },
            "replication_contract": {
                "enabled": True,
                "strict": True,
                "name": "paper69_vindr_practical_validation_mass_v3",
                "preserve_official_test": True,
                "require_positive_source_images": False,
                "expected_source_images": {"train": 13600, "val": 2400, "test": 4000},
                "expected_source_studies": {"train": 3400, "val": 600, "test": 1000},
                "expected_source_annotations": {"test": 237},
            },
        },
    },
    SIMPLE_PRESET_KEY: {
        "label": "Custom — balanced 1024 crops + paired whole breast (v1)",
        "description": (
            "Common mammography cleanup, whole-breast histogram equalization and R/G/B percentile "
            "normalization at 0–100, 50–100, and 75–100 before cropping, exact-stride 1024 crops "
            "with 512 stride and zero-padded edges, every training positive plus online-sampled "
            "clean training negatives with at least 80% breast-mask occupancy toward a 1:1 ratio, "
            "complete validation/test crop grids, and a "
            "pad-first 1024 whole-image companion sharing every crop basename."
        ),
        "output_folder_name": "preprocessed-vindr-simple-preset-v1",
        "replace_sections": [
            "gui", "image", "preprocess", "image_export", "histogram_equalization",
            "preserved_16bit", "metadata", "source_cohort", "splits", "vendor_filter",
            "export", "square_crops", "baseline_uncropped", "paired_whole_images",
            "crop_annotation_policy", "replication_contract", "study_preset_provenance",
        ],
        "config_patch": {
            "study_preset_provenance": {
                "preset_key": SIMPLE_PRESET_KEY,
                "preset_version": 1,
                "replication_scope": "User-defined simple mass-detection extraction pipeline",
                "assumptions": {
                    "whole_image_definition": "after fixed inversion/breast crop/mask/mirroring, before square crop",
                    "photometric_scope": "histogram equalization and channel percentile normalization run on the whole fixed-preprocessed breast before square cropping",
                    "crop_breast_occupancy": "training negatives contain at least 80% breast pixels measured from the retained preprocessing mask; training positives bypass the breast-fraction rule and validation/test do not apply it; zero padding counts as non-breast",
                    "positive_definition": "at least 30% annotation visibility; positive crops bypass the breast-fraction threshold",
                    "negative_definition": "training only: zero visibility of every Mass box and at least 80% breast-mask occupancy",
                    "balance_scope": "training only: streaming approximate 1:1 positive/negative ratio; every positive is retained and eligible negatives are admitted online from shuffled sources/windows; validation/test keep every grid crop",
                    "paired_whole_canvas": "pad each breast to a square at left/top, then resize to 1024x1024",
                },
            },
            "gui": {
                "filter_split": "all",
                "filter_positive": "all images",
                "filter_vendor_mode": "all vendors",
            },
            "image": {
                "normalize": "none",
                "percentile_range": [0.5, 99.5],
                "use_voi_lut": True,
                "strict_voi_lut": False,
            },
            "preprocess": {
                "invert_to_black_background": True,
                "trim_border_px": 0,
                "intensity_scale_before_geometry": "none",
                "crop_breast": True,
                "mask_outside_breast": True,
                "mirror_right_to_left": True,
                "crop_padding": None,
                "crop_padding_fraction": 0.03,
                "minimum_padding_px": 32,
                "maximum_padding_px": 128,
                "crop_threshold": None,
                "breast_mask_method": "largest_connected_tissue",
                "breast_mask_close_kernel": 21,
                "breast_mask_open_kernel": 7,
                "breast_mask_fill_holes": True,
                "breast_mask_keep_largest_component": True,
                "min_component_area_fraction": 0.001,
                "min_box_visibility_after_crop": 0.30,
                "retain_breast_mask_for_export": True,
            },
            "image_export": {
                "rgb_scheme": "custom_channel_pipeline",
                "custom_channel_pipeline": {
                    "R": {
                        "source": "current_crop",
                        "steps": [
                            {"op": "hist_equalize", "apply_before_crop": True, "params": {}},
                            {"op": "percentile_normalize", "apply_before_crop": True, "params": {"percentiles": [0.0, 100.0]}},
                        ],
                    },
                    "G": {
                        "source": "current_crop",
                        "steps": [
                            {"op": "hist_equalize", "apply_before_crop": True, "params": {}},
                            {"op": "percentile_normalize", "apply_before_crop": True, "params": {"percentiles": [50.0, 100.0]}},
                        ],
                    },
                    "B": {
                        "source": "current_crop",
                        "steps": [
                            {"op": "hist_equalize", "apply_before_crop": True, "params": {}},
                            {"op": "percentile_normalize", "apply_before_crop": True, "params": {"percentiles": [75.0, 100.0]}},
                        ],
                    },
                },
            },
            "histogram_equalization": {"enabled": False, "apply_to": "none"},
            "preserved_16bit": {"save": False},
            "metadata": {"save_full_source_csvs": True, "include_dicom_meta": True},
            "source_cohort": {"finding_category": "Mass", "positive_images_only": False},
            "splits": {
                "strategy": "random_study_fraction",
                "val_fraction_from_training": 0.15,
                "seed": 123,
                "stratify_by_birads": True,
            },
            "vendor_filter": {"enabled": False, "include_vendors": []},
            "export": {
                "clean_output_root": True,
                "save_square_crops": True,
                "save_baseline_uncropped": False,
                "save_empty_label_files": True,
            },
            "square_crops": {
                "crop_size": 1024,
                "stride": 512,
                "edge_policy": "regular_stride_pad",
                "pad_if_needed": True,
                "pad_value": 0.0,
                "train_crop_mode": "deterministic",
                "val_crop_mode": "deterministic",
                "test_crop_mode": "deterministic",
                "deterministic_selection_mode": "positive_ratio",
                "deterministic_target_positive_ratio": 0.50,
                "train_deterministic_selection_mode": "positive_ratio",
                "val_deterministic_selection_mode": "all",
                "test_deterministic_selection_mode": "all",
                "train_deterministic_target_positive_ratio": 0.50,
                "val_deterministic_target_positive_ratio": 0.50,
                "test_deterministic_target_positive_ratio": 0.50,
                "train_deterministic_include_empty": True,
                "val_deterministic_include_empty": True,
                "test_deterministic_include_empty": True,
                "train_require_clean_negative_windows": True,
                "val_require_clean_negative_windows": False,
                "test_require_clean_negative_windows": False,
                "online_positive_ratio_selection_for_deterministic": False,
                "train_online_positive_ratio_selection_for_deterministic": True,
                "val_online_positive_ratio_selection_for_deterministic": False,
                "test_online_positive_ratio_selection_for_deterministic": False,
                "online_balance_shuffle_source_records": True,
                "train_online_balance_shuffle_source_records": True,
                "online_balance_shuffle_windows": True,
                "train_online_balance_shuffle_windows": True,
                "deterministic_require_foreground": True,
                "deterministic_min_foreground_fraction": 0.80,
                "train_deterministic_require_foreground": True,
                "val_deterministic_require_foreground": False,
                "test_deterministic_require_foreground": False,
                "train_deterministic_min_foreground_fraction": 0.80,
                "val_deterministic_min_foreground_fraction": 0.0,
                "test_deterministic_min_foreground_fraction": 0.0,
                "negative_require_foreground": True,
                "negative_min_foreground_fraction": 0.80,
                "train_negative_require_foreground": True,
                "val_negative_require_foreground": False,
                "test_negative_require_foreground": False,
                "train_negative_min_foreground_fraction": 0.80,
                "val_negative_min_foreground_fraction": 0.0,
                "test_negative_min_foreground_fraction": 0.0,
                # Final fail-safe after RGB encoding. The retained mask is the
                # primary 80% contract; this prevents a broken photometric
                # operation from silently writing a black training negative.
                "negative_reject_blank_output": False,
                "train_negative_reject_blank_output": True,
                "val_negative_reject_blank_output": False,
                "test_negative_reject_blank_output": False,
                "negative_min_output_signal_fraction": 0.01,
                "train_negative_min_output_signal_fraction": 0.01,
                "require_foreground_for_empty_crops": True,
                "min_foreground_fraction": 0.80,
                "deduplicate_windows_per_source": True,
                "whole_stage_cache_items": 12,
                "seed": 123,
            },
            "paired_whole_images": {
                "enabled": True,
                "target_width": 1024,
                "target_height": 1024,
                "canvas_mode": "per_image_square",
                "pad_value": 0.0,
                "pad_anchor": "left_top",
                "storage_mode": "hardlink",
            },
            "baseline_uncropped": {
                "resize_mode": "none",
                "target_width": 1024,
                "target_height": 1024,
                "pad_value": 0.0,
                "pad_anchor": "left_top",
            },
            "crop_annotation_policy": {
                "allow_partial_annotations": True,
                "min_box_visibility": 0.30,
                "reject_partial_windows": False,
                "negative_max_box_visibility": 0.0,
            },
            "replication_contract": {"enabled": False},
        },
    },
}


# Keep the audited paper-like v2 recipe intact and expose the requested training
# subset as a separate, explicitly custom version.  Starting from the old preset
# also guarantees that the DICOM, breast-mask, RGB, patch-size, overlap, and box
# projection behavior cannot drift between the two Paper 22 choices.
_paper22_improved = copy.deepcopy(STUDY_PRESETS[PAPER_22_PRESET_KEY])
_paper22_improved.update({
    "label": "Custom Paper 22 — improved breast-balanced foreground crops (v4)",
    "description": (
        "Custom Paper 22 variant. It preserves the old patient-level validation IDs and the "
        "official mass-positive test cohort. It filters training, validation, and test crops "
        "with the retained breast mask. "
        "Training expands only the already selected training patients to all views, labels each "
        "breast by study/patient plus laterality, requires every retained training crop to contain "
        "strictly more than 10% of the fixed breast mask, and samples an exact 50/50 crop mixture "
        "from negative breasts and breasts with at least one Mass. The dataset folder is "
        "preprocessed-vindr-paper22-improved-v4."
    ),
    "output_folder_name": "preprocessed-vindr-paper22-improved-v4",
})
_paper22_patch = _paper22_improved["config_patch"]
_paper22_patch["study_preset_provenance"] = {
    **_paper22_patch["study_preset_provenance"],
    "preset_key": PAPER_22_IMPROVED_PRESET_KEY,
    "preset_version": 4,
    "replication_scope": "Custom Paper 22-inspired VinDr-Mammo Mass detector preprocessing",
    "split_identity": "paper22_v2_patient_split; training expanded by study/patient and breast laterality",
    "assumptions": {
        **_paper22_patch["study_preset_provenance"]["assumptions"],
        "training_source_unit": "breast=(study_id,laterality); include all views from selected training patients",
        "training_crop_source_balance": "exact 50% mass-positive breasts and 50% negative breasts",
        "breast_coverage_rule": "retained fixed-preprocessing breast mask fraction > 0.10",
        "validation_and_test_grid": "640px edge-aligned candidates at stride 512; retain only breast_fraction > 0.10",
    },
}
_paper22_patch["source_cohort"] = {
    "finding_category": "Mass",
    "positive_images_only": True,
    "train_expand_to_all_patient_breast_views": True,
    "train_breast_status_unit": "study_laterality",
}
_paper22_patch["square_crops"].update({
    "deterministic_selection_mode": "source_breast_ratio",
    "train_deterministic_selection_mode": "source_breast_ratio",
    "train_deterministic_target_source_breast_mass_ratio": 0.50,
    "train_deterministic_target_positive_ratio": 0.50,
    "train_require_min_breast_fraction_for_all_crops": True,
    "train_min_breast_fraction_for_all_crops": 0.10,
    "train_breast_fraction_comparison_for_all_crops": "strictly_greater_than",
    "train_require_retained_breast_mask_for_all_crops": True,
    "train_deterministic_require_foreground": False,
    "train_negative_require_foreground": False,
    "val_deterministic_selection_mode": "all",
    "test_deterministic_selection_mode": "all",
    "val_require_min_breast_fraction_for_all_crops": True,
    "val_min_breast_fraction_for_all_crops": 0.10,
    "val_breast_fraction_comparison_for_all_crops": "strictly_greater_than",
    "val_require_retained_breast_mask_for_all_crops": True,
    "test_require_min_breast_fraction_for_all_crops": True,
    "test_min_breast_fraction_for_all_crops": 0.10,
    "test_breast_fraction_comparison_for_all_crops": "strictly_greater_than",
    "test_require_retained_breast_mask_for_all_crops": True,
    "val_deterministic_require_foreground": False,
    "test_deterministic_require_foreground": False,
    "val_negative_require_foreground": False,
    "test_negative_require_foreground": False,
    "val_require_clean_negative_windows": False,
    "test_require_clean_negative_windows": False,
})
_paper22_patch["replication_contract"] = {
    "enabled": True,
    "strict": True,
    "name": "paper22_patient_breast_balanced_v4",
    "preserve_official_test": True,
    "require_positive_source_images": False,
    "require_positive_source_images_by_split": {"train": False, "val": True, "test": True},
    "require_all_source_annotations_represented": True,
    "require_source_annotation_ids": True,
    "expected_train_selection_mode": "source_breast_ratio",
    "expected_eval_selection_mode": "all",
    "expected_train_mass_breast_crop_fraction": 0.50,
    "min_breast_fraction_strictly_greater_than_by_split": {
        "train": 0.10,
        "val": 0.10,
        "test": 0.10,
    },
    "expected_source_images": {"train": 1592, "val": 136, "test": 219},
    "expected_source_studies": {"train": 398, "val": 71, "test": 115},
    "expected_source_annotations": {"test": 237},
    "expected_train_breasts": {"mass": 417, "negative": 379},
    "expected_train_source_views_by_breast_status": {"mass": 834, "negative": 758},
}

# Preserve a predictable adjacent order in the GUI: old Paper 22, improved
# Paper 22, Paper 69, then the general-purpose simple preset.
STUDY_PRESETS = {
    PAPER_22_PRESET_KEY: STUDY_PRESETS[PAPER_22_PRESET_KEY],
    PAPER_22_IMPROVED_PRESET_KEY: _paper22_improved,
    **{
        key: value
        for key, value in STUDY_PRESETS.items()
        if key != PAPER_22_PRESET_KEY
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
