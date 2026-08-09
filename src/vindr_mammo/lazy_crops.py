from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .crops import sliding_square_windows
from .dataset_layout import (
    is_grouped_content_root,
    resolve_existing_dataset_content_root,
    window_grid_configs,
)


LAZY_CROP_SCHEMA_VERSION = 1
LAZY_CROP_DIRECTORY = "lazy_crop_manifests"
SPLITS = ("train", "val", "test")

CROP_COLUMNS = [
    "schema_version",
    "crop_id",
    "split",
    "manifest_row_index",
    "source_image_id",
    "source_study_id",
    "source_breast_key",
    "source_image_has_mass",
    "source_breast_has_mass",
    "source_png_path",
    "source_float32_path",
    "context_resized_png_path",
    "context_resized_float32_path",
    "source_width",
    "source_height",
    "source_preprocessing_mirrored",
    "source_coordinate_space",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "crop_width",
    "crop_height",
    "source_intersection_x0",
    "source_intersection_y0",
    "source_intersection_x1",
    "source_intersection_y1",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "window_size",
    "stride",
    "edge_policy",
    "source_extent_fraction",
    "source_extent_filter_enabled",
    "min_source_extent_fraction",
    "source_extent_comparison",
    "positive_bypassed_source_extent_filter",
    "num_mass_annotations",
    "is_mass_positive",
    "is_clean_negative",
    "max_source_box_visibility",
    "min_box_visibility",
    "selection_policy",
    "negative_source_policy",
    "sampling_seed",
]

ANNOTATION_COLUMNS = [
    "schema_version",
    "crop_id",
    "split",
    "source_image_id",
    "source_study_id",
    "source_annotation_id",
    "source_annotation_index",
    "source_bbox_x0",
    "source_bbox_y0",
    "source_bbox_x1",
    "source_bbox_y1",
    "source_bbox_width",
    "source_bbox_height",
    "visible_source_x0",
    "visible_source_y0",
    "visible_source_x1",
    "visible_source_y1",
    "crop_bbox_x0",
    "crop_bbox_y0",
    "crop_bbox_x1",
    "crop_bbox_y1",
    "crop_bbox_width",
    "crop_bbox_height",
    "visible_fraction",
    "category_id",
    "category_name",
]


def resolve_lazy_crop_dataset_root(path: str | Path) -> tuple[Path, Path]:
    """Return ``(export_root, content_root)`` for grouped or legacy datasets."""

    return resolve_existing_dataset_content_root(path)


def lazy_crop_output_folder(
    dataset_root: str | Path,
    *,
    window_size: int,
    stride: int,
) -> Path:
    export_root, content_root = resolve_lazy_crop_dataset_root(dataset_root)
    if is_grouped_content_root(content_root):
        return (
            export_root
            / "annotations"
            / "windows"
            / f"window_{int(window_size)}_stride_{int(stride)}"
        )
    return export_root / LAZY_CROP_DIRECTORY / f"window{int(window_size)}_stride{int(stride)}"


def default_lazy_crop_config(
    dataset_root: str | Path,
    *,
    window_size: int = 1024,
    stride: int = 128,
    min_box_visibility: float = 0.05,
    train_positive_fraction: float = 0.50,
    train_min_source_extent_fraction: float = 0.10,
    eval_min_source_extent_fraction: float = 0.05,
    seed: int = 123,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the metadata-only defaults used by the research-dataset GUI."""

    output_root = lazy_crop_output_folder(
        dataset_root,
        window_size=window_size,
        stride=stride,
    )
    return {
        "paths": {
            "dataset_root": str(Path(dataset_root).expanduser().resolve(strict=False)),
            "output_root": str(output_root),
        },
        "geometry": {
            "window_size": int(window_size),
            "stride": int(stride),
            "edge_policy": "regular_stride_pad",
        },
        "annotations": {
            "allow_partial_annotations": True,
            "min_box_visibility": float(min_box_visibility),
            "negative_max_box_visibility": 0.0,
        },
        "filters": {
            # The source images were already breast-cropped and background-masked.
            # Without decoding pixels or a full-resolution saved mask, source extent
            # is the only exact metadata-only occupancy signal. It is deliberately
            # named as such throughout the output instead of being presented as a
            # pixel-derived breast-mask fraction.
            "source_extent_filter_enabled": True,
            "min_source_extent_fraction_by_split": {
                "train": float(train_min_source_extent_fraction),
                "val": float(eval_min_source_extent_fraction),
                "test": float(eval_min_source_extent_fraction),
            },
            "source_extent_comparison": "strictly_greater_than",
            "preserve_positive_windows_below_threshold": True,
        },
        "sampling": {
            "train_selection_policy": "keep_all_positive_balance_clean_negative_breasts",
            "train_positive_fraction": float(train_positive_fraction),
            "train_require_clean_negative_windows": True,
            "train_require_mass_negative_breasts": True,
            "val_selection_policy": "all_eligible_windows",
            "test_selection_policy": "all_eligible_windows",
            "seed": int(seed),
        },
        "runtime": {
            "overwrite": bool(overwrite),
            "decode_source_images": False,
        },
    }


def scan_lazy_crop_source(dataset_root: str | Path) -> dict[str, Any]:
    """Inspect only CSV metadata and report available lazy-crop sources."""

    export_root, crop_root = resolve_lazy_crop_dataset_root(dataset_root)
    sources, annotations = _load_sources_and_annotations(crop_root)
    split_counts = {
        split: int(sum(source["split"] == split for source in sources))
        for split in SPLITS
    }
    annotation_count = int(sum(len(value) for value in annotations.values()))
    original_float_count = int(sum(bool(source["source_float32_path"]) for source in sources))
    return {
        "dataset_root": str(export_root),
        "square_crops_root": str(crop_root),
        "source_images": len(sources),
        "source_images_by_split": split_counts,
        "source_mass_annotations": annotation_count,
        "same_geometry_float32_sources": original_float_count,
        "png_sources": int(sum(bool(source["source_png_path"]) for source in sources)),
        "decoded_images": 0,
    }


def estimate_lazy_crop_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate complete-grid rows from source dimensions without classifying pixels."""

    normalized = _normalize_config(config)
    _export_root, crop_root = resolve_lazy_crop_dataset_root(
        normalized["paths"]["dataset_root"]
    )
    geometry = normalized["geometry"]
    sources, annotations = _load_sources_and_annotations(
        crop_root, context_window_size=int(geometry["window_size"])
    )
    by_split = {split: 0 for split in SPLITS}
    for source in sources:
        by_split[source["split"]] += len(
            sliding_square_windows(
                int(source["source_width"]),
                int(source["source_height"]),
                int(geometry["window_size"]),
                int(geometry["stride"]),
                str(geometry["edge_policy"]),
            )
        )
    return {
        "source_images": len(sources),
        "source_mass_annotations": int(sum(len(value) for value in annotations.values())),
        "complete_grid_rows_by_split": by_split,
        "complete_grid_rows": int(sum(by_split.values())),
        "decoded_images": 0,
        "output_root": normalized["paths"]["output_root"],
    }


def extract_lazy_crop_manifests(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write crop/annotation CSVs from whole-image metadata without loading pixels."""

    cfg = _normalize_config(config)
    export_root, crop_root = resolve_lazy_crop_dataset_root(cfg["paths"]["dataset_root"])
    output_root = Path(cfg["paths"]["output_root"])
    sources, annotations_by_source = _load_sources_and_annotations(
        crop_root,
        context_window_size=int(cfg["geometry"]["window_size"]),
    )
    sources_by_split = {
        split: [source for source in sources if source["split"] == split]
        for split in SPLITS
    }
    expected = _expected_output_files(output_root)
    if not bool(cfg["runtime"]["overwrite"]):
        existing = [str(path) for path in expected if path.exists()]
        if existing:
            raise FileExistsError(
                "Lazy crop output already exists. Enable overwrite to replace only the "
                "known manifest files: " + ", ".join(existing)
            )
    output_root.mkdir(parents=True, exist_ok=True)

    train_selection_policy = str(cfg["sampling"]["train_selection_policy"])
    train_keeps_all_windows = train_selection_policy == "all_eligible_windows"
    train_counts = {"positive": 0, "negative_candidates": 0, "eligible": 0}
    train_sources = sources_by_split["train"]
    total_source_passes = len(sources) + len(train_sources)
    processed_source_passes = 0
    for source in train_sources:
        for candidate in _iter_source_candidates(source, annotations_by_source, cfg):
            if not candidate["eligible"]:
                continue
            train_counts["eligible"] += 1
            if candidate["is_mass_positive"]:
                train_counts["positive"] += 1
            elif train_keeps_all_windows or _eligible_train_negative(candidate, cfg):
                train_counts["negative_candidates"] += 1
        processed_source_passes += 1
        _emit_progress(
            progress_callback,
            processed_source_passes,
            total_source_passes,
            stage="count_train_candidates",
            source=source,
        )

    positive_fraction = float(cfg["sampling"]["train_positive_fraction"])
    desired_negatives = _desired_negative_count(
        train_counts["positive"], positive_fraction
    )
    selected_negative_count = (
        train_counts["negative_candidates"]
        if train_keeps_all_windows
        else min(desired_negatives, train_counts["negative_candidates"])
    )
    selected_negative_indices = (
        set(range(selected_negative_count))
        if train_keeps_all_windows
        else set(
            random.Random(int(cfg["sampling"]["seed"])).sample(
                range(train_counts["negative_candidates"]),
                selected_negative_count,
            )
        )
    )

    split_summaries: dict[str, dict[str, Any]] = {}
    temporary_files: list[Path] = []
    try:
        for split in SPLITS:
            crop_path = output_root / f"lazy_crop_manifest_{split}.csv"
            annotation_path = output_root / f"lazy_crop_annotations_{split}.csv"
            crop_tmp = crop_path.with_suffix(crop_path.suffix + ".tmp")
            annotation_tmp = annotation_path.with_suffix(annotation_path.suffix + ".tmp")
            temporary_files.extend([crop_tmp, annotation_tmp])
            summary = {
                "source_images": len(sources_by_split[split]),
                "complete_grid_windows": 0,
                "eligible_windows": 0,
                "saved_crops": 0,
                "positive_crops": 0,
                "negative_crops": 0,
                "saved_annotations": 0,
                "skipped_source_extent": 0,
                "skipped_by_train_balance": 0,
            }
            negative_candidate_index = 0
            with crop_tmp.open("w", newline="", encoding="utf-8") as crop_file, annotation_tmp.open(
                "w", newline="", encoding="utf-8"
            ) as annotation_file:
                crop_writer = csv.DictWriter(crop_file, fieldnames=CROP_COLUMNS)
                annotation_writer = csv.DictWriter(
                    annotation_file, fieldnames=ANNOTATION_COLUMNS
                )
                crop_writer.writeheader()
                annotation_writer.writeheader()
                for source in sources_by_split[split]:
                    for candidate in _iter_source_candidates(
                        source, annotations_by_source, cfg
                    ):
                        summary["complete_grid_windows"] += 1
                        if not candidate["eligible"]:
                            summary["skipped_source_extent"] += 1
                            continue
                        summary["eligible_windows"] += 1
                        keep = True
                        if split == "train" and not candidate["is_mass_positive"]:
                            if train_keeps_all_windows:
                                keep = True
                            elif not _eligible_train_negative(candidate, cfg):
                                keep = False
                            else:
                                keep = negative_candidate_index in selected_negative_indices
                                negative_candidate_index += 1
                        if not keep:
                            summary["skipped_by_train_balance"] += 1
                            continue
                        crop_row = _crop_csv_row(
                            candidate,
                            cfg,
                            manifest_row_index=summary["saved_crops"],
                        )
                        crop_writer.writerow(crop_row)
                        summary["saved_crops"] += 1
                        if candidate["is_mass_positive"]:
                            summary["positive_crops"] += 1
                        else:
                            summary["negative_crops"] += 1
                        for annotation in candidate["kept_annotations"]:
                            annotation_writer.writerow(
                                _annotation_csv_row(candidate, annotation)
                            )
                            summary["saved_annotations"] += 1
                    processed_source_passes += 1
                    _emit_progress(
                        progress_callback,
                        processed_source_passes,
                        total_source_passes,
                        stage=f"write_{split}",
                        source=source,
                    )
            os.replace(crop_tmp, crop_path)
            os.replace(annotation_tmp, annotation_path)
            split_summaries[split] = summary

        resolved_path = output_root / "lazy_crop_config_resolved.yaml"
        resolved_tmp = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        temporary_files.append(resolved_tmp)
        resolved_tmp.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(resolved_tmp, resolved_path)

        manifest = {
            "schema_version": LAZY_CROP_SCHEMA_VERSION,
            "kind": "lazy_crop_manifests",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_root": str(export_root),
            "square_crops_root": str(crop_root),
            "content_root": str(crop_root),
            "dataset_layout": (
                "images_annotations_v1"
                if is_grouped_content_root(crop_root)
                else "legacy_square_crops"
            ),
            "output_root": str(output_root),
            "decoded_source_images": 0,
            "source_contract": {
                "coordinate_space": "fixed_preprocessed_original_whole",
                "primary_source": "source_png_path",
                "same_geometry_float32_when_available": "source_float32_path",
                "resized_context_is_not_a_drop_in_crop_source": True,
            },
            "selection": {
                "window_size": cfg["geometry"]["window_size"],
                "stride": cfg["geometry"]["stride"],
                "edge_policy": cfg["geometry"]["edge_policy"],
                "min_box_visibility": cfg["annotations"]["min_box_visibility"],
                "train_positive_fraction_target": positive_fraction,
                "train_selection_policy": train_selection_policy,
                "train_positive_candidates": train_counts["positive"],
                "train_negative_candidates": train_counts["negative_candidates"],
                "train_desired_negatives": desired_negatives,
                "train_selected_negatives": selected_negative_count,
                "seed": cfg["sampling"]["seed"],
            },
            "splits": split_summaries,
            "files": {
                split: {
                    "crops": f"lazy_crop_manifest_{split}.csv",
                    "annotations": f"lazy_crop_annotations_{split}.csv",
                }
                for split in SPLITS
            },
        }
        manifest_path = output_root / "lazy_crop_manifest.json"
        manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_files.append(manifest_tmp)
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(manifest_tmp, manifest_path)

        readme_path = output_root / "README.md"
        readme_tmp = readme_path.with_suffix(readme_path.suffix + ".tmp")
        temporary_files.append(readme_tmp)
        readme_tmp.write_text(_lazy_crop_readme(manifest, cfg), encoding="utf-8")
        os.replace(readme_tmp, readme_path)
    finally:
        for path in temporary_files:
            path.unlink(missing_ok=True)

    _emit_progress(
        progress_callback,
        total_source_passes,
        total_source_passes,
        stage="completed",
        source=None,
    )
    return {
        "output_root": str(output_root),
        "decoded_source_images": 0,
        "saved_crops": int(
            sum(value["saved_crops"] for value in split_summaries.values())
        ),
        "saved_annotations": int(
            sum(value["saved_annotations"] for value in split_summaries.values())
        ),
        "splits": split_summaries,
        "manifest": str(output_root / "lazy_crop_manifest.json"),
        "readme": str(output_root / "README.md"),
    }


def extract_complete_lazy_crop_family(
    dataset_root: str | Path,
    *,
    window_size: int = 1024,
    strides: Sequence[int] = (128, 256, 512),
    grids: Sequence[Mapping[str, Any]] | None = None,
    min_box_visibility: float = 0.05,
    overwrite: bool = False,
) -> dict[Any, dict[str, Any]]:
    """Write complete, unbalanced lazy grids for several window/stride pairs.

    This is the whole-image research export companion. It deliberately disables
    the geometry-only source-extent threshold and retains every grid window in
    every split, so foreground heuristics cannot change model-input coverage.
    """

    use_legacy_keys = grids is None
    if grids is None:
        normalized_strides = [int(stride) for stride in strides]
        grids = [
            {"window_size": int(window_size), "stride": stride}
            for stride in normalized_strides
        ]
    normalized_grids = window_grid_configs({"lazy_crop_grids": list(grids)})
    results: dict[Any, dict[str, Any]] = {}
    for grid in normalized_grids:
        current_window = int(grid["window_size"])
        current_stride = int(grid["stride"])
        config = default_lazy_crop_config(
            dataset_root,
            window_size=current_window,
            stride=current_stride,
            min_box_visibility=float(min_box_visibility),
            overwrite=bool(overwrite),
        )
        config["filters"]["source_extent_filter_enabled"] = False
        config["sampling"].update({
            "train_selection_policy": "all_eligible_windows",
            "train_require_clean_negative_windows": False,
            "train_require_mass_negative_breasts": False,
        })
        key: Any = (
            current_stride
            if use_legacy_keys
            else f"window_{current_window}_stride_{current_stride}"
        )
        results[key] = extract_lazy_crop_manifests(config)
    return results


def _normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(dict(config)))
    paths = dict(cfg.get("paths", {}) or {})
    dataset_root = str(paths.get("dataset_root", "")).strip()
    if not dataset_root:
        raise ValueError("paths.dataset_root is required")
    geometry = dict(cfg.get("geometry", {}) or {})
    window_size = int(geometry.get("window_size", 1024))
    stride = int(geometry.get("stride", 128))
    if window_size <= 0 or stride <= 0:
        raise ValueError("geometry.window_size and geometry.stride must be positive")
    edge_policy = str(geometry.get("edge_policy", "regular_stride_pad"))
    if edge_policy != "regular_stride_pad":
        raise ValueError(
            "Lazy crop manifests currently require edge_policy=regular_stride_pad"
        )
    paths["dataset_root"] = str(
        Path(dataset_root).expanduser().resolve(strict=False)
    )
    output_value = str(paths.get("output_root", "")).strip()
    if not output_value:
        output_value = str(
            lazy_crop_output_folder(
                paths["dataset_root"], window_size=window_size, stride=stride
            )
        )
    paths["output_root"] = str(
        Path(output_value).expanduser().resolve(strict=False)
    )
    annotations = dict(cfg.get("annotations", {}) or {})
    min_visibility = float(annotations.get("min_box_visibility", 0.05))
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("annotations.min_box_visibility must be between 0 and 1")
    filters = dict(cfg.get("filters", {}) or {})
    thresholds = dict(filters.get("min_source_extent_fraction_by_split", {}) or {})
    thresholds = {
        split: float(thresholds.get(split, 0.10 if split == "train" else 0.05))
        for split in SPLITS
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("source-extent thresholds must be between 0 and 1")
    comparison = str(filters.get("source_extent_comparison", "strictly_greater_than"))
    if comparison not in {"strictly_greater_than", "greater_than_or_equal"}:
        raise ValueError("unsupported source_extent_comparison")
    sampling = dict(cfg.get("sampling", {}) or {})
    train_selection_policy = str(
        sampling.get(
            "train_selection_policy",
            "keep_all_positive_balance_clean_negative_breasts",
        )
    )
    if train_selection_policy not in {
        "keep_all_positive_balance_clean_negative_breasts",
        "all_eligible_windows",
    }:
        raise ValueError("unsupported sampling.train_selection_policy")
    positive_fraction = float(sampling.get("train_positive_fraction", 0.50))
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("sampling.train_positive_fraction must be in (0, 1]")
    runtime = dict(cfg.get("runtime", {}) or {})
    if bool(runtime.get("decode_source_images", False)):
        raise ValueError("lazy crop generation never decodes source images")
    return {
        "schema_version": LAZY_CROP_SCHEMA_VERSION,
        "paths": paths,
        "geometry": {
            "window_size": window_size,
            "stride": stride,
            "edge_policy": edge_policy,
        },
        "annotations": {
            "allow_partial_annotations": bool(
                annotations.get("allow_partial_annotations", True)
            ),
            "min_box_visibility": min_visibility,
            "negative_max_box_visibility": float(
                annotations.get("negative_max_box_visibility", 0.0)
            ),
        },
        "filters": {
            "source_extent_filter_enabled": bool(
                filters.get("source_extent_filter_enabled", True)
            ),
            "min_source_extent_fraction_by_split": thresholds,
            "source_extent_comparison": comparison,
            "preserve_positive_windows_below_threshold": bool(
                filters.get("preserve_positive_windows_below_threshold", True)
            ),
        },
        "sampling": {
            "train_selection_policy": str(
                train_selection_policy
            ),
            "train_positive_fraction": positive_fraction,
            "train_require_clean_negative_windows": bool(
                sampling.get("train_require_clean_negative_windows", True)
            ),
            "train_require_mass_negative_breasts": bool(
                sampling.get("train_require_mass_negative_breasts", True)
            ),
            "val_selection_policy": "all_eligible_windows",
            "test_selection_policy": "all_eligible_windows",
            "seed": int(sampling.get("seed", 123)),
        },
        "runtime": {
            "overwrite": bool(runtime.get("overwrite", False)),
            "decode_source_images": False,
        },
    }


def _load_sources_and_annotations(
    crop_root: Path,
    *,
    context_window_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metadata_root = crop_root / "metadata"
    manifest = pd.read_csv(metadata_root / "whole_image_manifest.csv", low_memory=False)
    required = {"variant", "split", "source_image_id", "source_study_id", "image_path", "width", "height"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "whole_image_manifest.csv is missing columns: " + ", ".join(missing)
        )
    originals = manifest[manifest["variant"].astype(str).str.casefold() == "original"].copy()
    if originals.empty:
        raise ValueError("whole_image_manifest.csv contains no original whole-image rows")
    manifest_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_row in manifest.to_dict("records"):
        manifest_image_id = _clean_identifier(manifest_row.get("source_image_id"))
        manifest_variant = str(manifest_row.get("variant", "")).casefold().strip()
        if manifest_image_id and manifest_variant:
            manifest_lookup[(manifest_image_id, manifest_variant)] = manifest_row

    sample_lookup: dict[str, dict[str, Any]] = {}
    samples_path = metadata_root / "samples_metadata_flat.csv"
    if samples_path.is_file():
        wanted = {
            "source_image_id",
            "source_breast_key",
            "source_breast_has_mass",
            "source_preprocessing_mirrored",
            "source_coordinate_space",
            "paired_whole_original_image",
            "paired_whole_original_float32_image",
            "paired_whole_image",
            "paired_whole_float32_image",
        }
        samples = pd.read_csv(
            samples_path,
            usecols=lambda column: column in wanted,
            low_memory=False,
        )
        for row in samples.to_dict("records"):
            image_id = _clean_identifier(row.get("source_image_id"))
            if image_id and image_id not in sample_lookup:
                sample_lookup[image_id] = row

    annotations_path = crop_root / "annotations" / "whole_image_annotations.csv"
    if not annotations_path.is_file():
        annotations_path = metadata_root / "whole_image_annotations.csv"
    annotations_frame = pd.read_csv(annotations_path, low_memory=False)
    if "variant" in annotations_frame:
        annotations_frame = annotations_frame[
            annotations_frame["variant"].astype(str).str.casefold() == "original"
        ]
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fallback_index, row in enumerate(annotations_frame.to_dict("records")):
        image_id = _clean_identifier(row.get("source_image_id"))
        bbox = _parse_bbox(row.get("source_bbox_xyxy", row.get("bbox_xyxy")))
        if not image_id or bbox is None:
            continue
        annotations[image_id].append(
            {
                "source_annotation_id": _clean_identifier(
                    row.get("source_annotation_id")
                )
                or str(fallback_index),
                "source_annotation_index": _clean_int(
                    row.get("annotation_index"), fallback_index
                ),
                "bbox": bbox,
            }
        )

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in originals.to_dict("records"):
        image_id = _clean_identifier(row.get("source_image_id"))
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        sample = sample_lookup.get(image_id, {})
        requested_resized_variant = (
            f"resized_{int(context_window_size)}x{int(context_window_size)}"
            if context_window_size is not None
            else ""
        )
        resized_row = manifest_lookup.get((image_id, requested_resized_variant), {})
        if not resized_row:
            resized_row = manifest_lookup.get((image_id, "resized"), {})
        if not resized_row:
            resized_candidates = [
                value
                for (candidate_image_id, candidate_variant), value in manifest_lookup.items()
                if candidate_image_id == image_id and candidate_variant.startswith("resized_")
            ]
            resized_row = max(
                resized_candidates,
                key=lambda value: int(value.get("width", 0) or 0)
                * int(value.get("height", 0) or 0),
                default={},
            )
        source_has_mass = bool(annotations.get(image_id))
        breast_value = sample.get(
            "source_breast_has_mass", row.get("source_breast_has_mass")
        )
        source_breast_has_mass = (
            _clean_bool(breast_value)
            if not _is_missing(breast_value)
            else source_has_mass
        )
        original_png = _first_text(
            sample.get("paired_whole_original_image"), row.get("image_path")
        )
        sources.append(
            {
                "split": str(row.get("split", "")).casefold().strip(),
                "source_image_id": image_id,
                "source_study_id": _clean_identifier(row.get("source_study_id")),
                "source_breast_key": _first_text(
                    sample.get("source_breast_key"), row.get("source_breast_key"), ""
                ),
                "source_image_has_mass": source_has_mass,
                "source_breast_has_mass": bool(source_breast_has_mass),
                "source_png_path": original_png,
                "source_float32_path": _first_text(
                    sample.get("paired_whole_original_float32_image"),
                    row.get("float32_path"),
                    "",
                ),
                "context_resized_png_path": _first_text(
                    resized_row.get("image_path"),
                    sample.get("paired_whole_image"),
                    "",
                ),
                "context_resized_float32_path": _first_text(
                    resized_row.get("float32_path"),
                    sample.get("paired_whole_float32_image"),
                    "",
                ),
                "source_width": _clean_int(row.get("width"), 0),
                "source_height": _clean_int(row.get("height"), 0),
                "source_preprocessing_mirrored": _clean_bool(
                    sample.get(
                        "source_preprocessing_mirrored",
                        row.get("source_preprocessing_mirrored", False),
                    )
                ),
                "source_coordinate_space": _first_text(
                    sample.get("source_coordinate_space"),
                    row.get("source_coordinate_space"),
                    "fixed_preprocessed",
                ),
            }
        )
    invalid_splits = sorted({source["split"] for source in sources} - set(SPLITS))
    if invalid_splits:
        raise ValueError("unsupported splits in whole manifest: " + ", ".join(invalid_splits))
    if any(source["source_width"] <= 0 or source["source_height"] <= 0 for source in sources):
        raise ValueError("whole-image manifest contains non-positive dimensions")
    return sources, dict(annotations)


def _iter_source_candidates(
    source: dict[str, Any],
    annotations_by_source: Mapping[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    geometry = cfg["geometry"]
    annotations = annotations_by_source.get(source["source_image_id"], [])
    windows = sliding_square_windows(
        source["source_width"],
        source["source_height"],
        geometry["window_size"],
        geometry["stride"],
        geometry["edge_policy"],
    )
    for grid_index, window in enumerate(windows):
        kept_annotations: list[dict[str, Any]] = []
        max_visibility = 0.0
        for annotation in annotations:
            clipped = _clip_annotation(annotation, window)
            max_visibility = max(max_visibility, clipped["visible_fraction"])
            if clipped["visible_fraction"] >= cfg["annotations"]["min_box_visibility"]:
                kept_annotations.append(clipped)
        x0, y0, x1, y1 = window
        width, height = source["source_width"], source["source_height"]
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(width, x1), min(height, y1)
        valid_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        window_area = int(geometry["window_size"]) ** 2
        source_extent_fraction = float(valid_area / max(1, window_area))
        split = source["split"]
        threshold = cfg["filters"]["min_source_extent_fraction_by_split"][split]
        filter_enabled = cfg["filters"]["source_extent_filter_enabled"]
        comparison = cfg["filters"]["source_extent_comparison"]
        threshold_pass = (
            source_extent_fraction > threshold
            if comparison == "strictly_greater_than"
            else source_extent_fraction >= threshold
        )
        is_positive = bool(kept_annotations)
        bypass = bool(
            is_positive
            and not threshold_pass
            and cfg["filters"]["preserve_positive_windows_below_threshold"]
        )
        eligible = bool(not filter_enabled or threshold_pass or bypass)
        crop_id = (
            f"{source['source_study_id']}__{source['source_image_id']}__lazycrop__"
            f"{split}_{grid_index:04d}_x{x0}_y{y0}_w{geometry['window_size']}_h{geometry['window_size']}"
        )
        yield {
            "source": source,
            "window": window,
            "crop_id": crop_id,
            "grid_index": grid_index,
            "source_intersection": (ix0, iy0, ix1, iy1),
            "pad": (
                max(0, -x0),
                max(0, -y0),
                max(0, x1 - width),
                max(0, y1 - height),
            ),
            "source_extent_fraction": source_extent_fraction,
            "min_source_extent_fraction": threshold,
            "threshold_pass": threshold_pass,
            "positive_bypass": bypass,
            "eligible": eligible,
            "kept_annotations": kept_annotations,
            "is_mass_positive": is_positive,
            "is_clean_negative": max_visibility
            <= cfg["annotations"]["negative_max_box_visibility"],
            "max_source_box_visibility": max_visibility,
        }


def _eligible_train_negative(candidate: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if candidate["is_mass_positive"]:
        return False
    if (
        cfg["sampling"]["train_require_clean_negative_windows"]
        and not candidate["is_clean_negative"]
    ):
        return False
    if (
        cfg["sampling"]["train_require_mass_negative_breasts"]
        and candidate["source"]["source_breast_has_mass"]
    ):
        return False
    return True


def _crop_csv_row(
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    *,
    manifest_row_index: int,
) -> dict[str, Any]:
    source = candidate["source"]
    x0, y0, x1, y1 = candidate["window"]
    ix0, iy0, ix1, iy1 = candidate["source_intersection"]
    pad_left, pad_top, pad_right, pad_bottom = candidate["pad"]
    split = source["split"]
    if split == "train":
        selection_policy = cfg["sampling"]["train_selection_policy"]
    else:
        selection_policy = cfg["sampling"][f"{split}_selection_policy"]
    negative_policy = (
        "mass_negative_breasts_only"
        if split == "train"
        and cfg["sampling"]["train_require_mass_negative_breasts"]
        else "any_source"
    )
    return {
        "schema_version": LAZY_CROP_SCHEMA_VERSION,
        "crop_id": candidate["crop_id"],
        "split": split,
        "manifest_row_index": int(manifest_row_index),
        "source_image_id": source["source_image_id"],
        "source_study_id": source["source_study_id"],
        "source_breast_key": source["source_breast_key"],
        "source_image_has_mass": int(source["source_image_has_mass"]),
        "source_breast_has_mass": int(source["source_breast_has_mass"]),
        "source_png_path": source["source_png_path"],
        "source_float32_path": source["source_float32_path"],
        "context_resized_png_path": source["context_resized_png_path"],
        "context_resized_float32_path": source[
            "context_resized_float32_path"
        ],
        "source_width": source["source_width"],
        "source_height": source["source_height"],
        "source_preprocessing_mirrored": int(
            source["source_preprocessing_mirrored"]
        ),
        "source_coordinate_space": source["source_coordinate_space"],
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_width": x1 - x0,
        "crop_height": y1 - y0,
        "source_intersection_x0": ix0,
        "source_intersection_y0": iy0,
        "source_intersection_x1": ix1,
        "source_intersection_y1": iy1,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "window_size": cfg["geometry"]["window_size"],
        "stride": cfg["geometry"]["stride"],
        "edge_policy": cfg["geometry"]["edge_policy"],
        "source_extent_fraction": candidate["source_extent_fraction"],
        "source_extent_filter_enabled": int(
            cfg["filters"]["source_extent_filter_enabled"]
        ),
        "min_source_extent_fraction": candidate["min_source_extent_fraction"],
        "source_extent_comparison": cfg["filters"]["source_extent_comparison"],
        "positive_bypassed_source_extent_filter": int(
            candidate["positive_bypass"]
        ),
        "num_mass_annotations": len(candidate["kept_annotations"]),
        "is_mass_positive": int(candidate["is_mass_positive"]),
        "is_clean_negative": int(candidate["is_clean_negative"]),
        "max_source_box_visibility": candidate["max_source_box_visibility"],
        "min_box_visibility": cfg["annotations"]["min_box_visibility"],
        "selection_policy": selection_policy,
        "negative_source_policy": negative_policy,
        "sampling_seed": cfg["sampling"]["seed"],
    }


def _annotation_csv_row(
    candidate: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any]:
    source = candidate["source"]
    source_box = annotation["bbox"]
    visible = annotation["visible_source_bbox"]
    crop_box = annotation["crop_bbox"]
    return {
        "schema_version": LAZY_CROP_SCHEMA_VERSION,
        "crop_id": candidate["crop_id"],
        "split": source["split"],
        "source_image_id": source["source_image_id"],
        "source_study_id": source["source_study_id"],
        "source_annotation_id": annotation["source_annotation_id"],
        "source_annotation_index": annotation["source_annotation_index"],
        "source_bbox_x0": source_box[0],
        "source_bbox_y0": source_box[1],
        "source_bbox_x1": source_box[2],
        "source_bbox_y1": source_box[3],
        "source_bbox_width": source_box[2] - source_box[0],
        "source_bbox_height": source_box[3] - source_box[1],
        "visible_source_x0": visible[0],
        "visible_source_y0": visible[1],
        "visible_source_x1": visible[2],
        "visible_source_y1": visible[3],
        "crop_bbox_x0": crop_box[0],
        "crop_bbox_y0": crop_box[1],
        "crop_bbox_x1": crop_box[2],
        "crop_bbox_y1": crop_box[3],
        "crop_bbox_width": crop_box[2] - crop_box[0],
        "crop_bbox_height": crop_box[3] - crop_box[1],
        "visible_fraction": annotation["visible_fraction"],
        "category_id": 1,
        "category_name": "Mass",
    }


def _clip_annotation(
    annotation: dict[str, Any], window: Sequence[int]
) -> dict[str, Any]:
    bx0, by0, bx1, by1 = annotation["bbox"]
    x0, y0, x1, y1 = [float(value) for value in window]
    ix0, iy0 = max(bx0, x0), max(by0, y0)
    ix1, iy1 = min(bx1, x1), min(by1, y1)
    visible_width, visible_height = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    source_area = max(1e-12, (bx1 - bx0) * (by1 - by0))
    visible_fraction = float(visible_width * visible_height / source_area)
    return {
        **annotation,
        "visible_source_bbox": (ix0, iy0, ix1, iy1),
        "crop_bbox": (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0),
        "visible_fraction": visible_fraction,
    }


def _desired_negative_count(positive_count: int, positive_fraction: float) -> int:
    if positive_count <= 0 or positive_fraction >= 1.0:
        return 0
    return max(
        0,
        int(round(positive_count * (1.0 - positive_fraction) / positive_fraction)),
    )


def _expected_output_files(output_root: Path) -> list[Path]:
    paths = [
        output_root / "lazy_crop_manifest.json",
        output_root / "lazy_crop_config_resolved.yaml",
        output_root / "README.md",
    ]
    for split in SPLITS:
        paths.extend(
            [
                output_root / f"lazy_crop_manifest_{split}.csv",
                output_root / f"lazy_crop_annotations_{split}.csv",
            ]
        )
    return paths


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    processed: int,
    total: int,
    *,
    stage: str,
    source: Mapping[str, Any] | None,
) -> None:
    if callback is None:
        return
    if processed != total and processed % 25 != 0:
        return
    callback(
        {
            "stage": stage,
            "processed": int(processed),
            "total": int(total),
            "source_image_id": source.get("source_image_id") if source else None,
            "decoded_images": 0,
        }
    )


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if _is_missing(value):
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        box = tuple(float(item) for item in parsed)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_text(*values: Any) -> str:
    for value in values:
        if not _is_missing(value) and str(value).strip():
            return str(value).strip()
    return ""


def _clean_identifier(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _clean_int(value: Any, default: int) -> int:
    if _is_missing(value):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clean_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    if _is_missing(value):
        return False
    return bool(value)


def _lazy_crop_readme(manifest: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    window_size = int(cfg["geometry"]["window_size"])
    stride = int(cfg["geometry"]["stride"])
    train = manifest["splits"]["train"]
    train_policy = str(cfg["sampling"]["train_selection_policy"])
    if train_policy == "all_eligible_windows":
        train_selection_text = (
            "Training, validation, and test retain every metadata-eligible grid "
            "window; no class balancing or foreground/source-extent sampling is "
            "applied when the corresponding filter is disabled."
        )
    else:
        train_selection_text = (
            f"All eligible positive windows were kept. Empty train windows were "
            f"sampled with seed `{int(cfg['sampling']['seed'])}` from breasts marked "
            f"Mass-negative, toward a Mass-positive fraction of "
            f"`{float(cfg['sampling']['train_positive_fraction']):.6g}`. Validation "
            "and test retain every metadata-eligible inference-grid window."
        )
    return f"""# Lazy crop manifests: {window_size}px window, {stride}px stride

This directory describes virtual mammography crops. It contains **no crop PNGs,
no crop float32 tensors, and no copied whole images**. Generation decoded zero
source images; it used the existing whole-image dimensions and Mass annotations.

## Files

- `lazy_crop_manifest_train.csv`, `lazy_crop_manifest_val.csv`, and
  `lazy_crop_manifest_test.csv`: one row per virtual crop.
- `lazy_crop_annotations_<split>.csv`: one row per retained crop-local Mass box,
  joined to the crop table by `crop_id`.
- `lazy_crop_manifest.json`: counts, source contract, and selection summary.
- `lazy_crop_config_resolved.yaml`: the exact settings that produced these rows.

The train manifest contains {int(train['positive_crops']):,} Mass-positive and
{int(train['negative_crops']):,} empty crops. {train_selection_text}

## Coordinate and padding contract

`source_png_path` is relative to the dataset content root recorded in the
manifest (`content_root`; legacy exports call the same path `square_crops_root`) and is
the primary source for `crop_x0:crop_y1`. Its coordinate space is the
fixed-preprocessed, original-size whole mammogram. If `source_float32_path` is
non-empty, it has the same geometry and may be used instead. In this export it
is valid for that field to be empty; do not substitute
`context_resized_float32_path`, because the resized context has different
geometry.

The requested crop is `[crop_x0, crop_y0, crop_x1, crop_y1)` with size
{window_size}×{window_size}. Read the in-bounds
`[source_intersection_x0, source_intersection_y0,
source_intersection_x1, source_intersection_y1)` region and place it into a
zero-filled output using `pad_left`, `pad_top`, `pad_right`, and `pad_bottom`.
The regular stride grid may extend past the right or bottom source edge.

Crop annotations are already clipped and translated. Use
`crop_bbox_x0:crop_bbox_y1` as crop-local XYXY labels. A source Mass is retained
when at least `{float(cfg['annotations']['min_box_visibility']):.6g}` of its
original fixed-preprocessed box area is visible.

## Metadata-only breast/source filter

The original whole mammograms were already breast-cropped and background-masked.
This extractor deliberately does not read their pixels and a full-resolution
breast mask was not saved for every source. Therefore
`source_extent_fraction` is the exact in-bounds source area divided by crop
area; it is a geometry-only proxy/upper bound, **not** a pixel-derived breast-mask
fraction. The threshold and comparison are recorded on every row. Positive
windows can bypass it, matching the preset's rule that an eligible Mass must
not disappear because of background filtering.

## Minimal PyTorch loading pattern

```python
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

dataset_root = Path({str(manifest.get('content_root', manifest['square_crops_root']))!r})
crops = pd.read_csv("lazy_crop_manifest_train.csv")
boxes = pd.read_csv("lazy_crop_annotations_train.csv")

row = crops.iloc[0]
source = pil_to_tensor(Image.open(dataset_root / row.source_png_path)).float() / 255
crop = torch.zeros((source.shape[0], int(row.window_size), int(row.window_size)))
region = source[
    :, int(row.source_intersection_y0):int(row.source_intersection_y1),
    int(row.source_intersection_x0):int(row.source_intersection_x1)
]
crop[
    :, int(row.pad_top):int(row.pad_top) + region.shape[-2],
    int(row.pad_left):int(row.pad_left) + region.shape[-1]
] = region
target_xyxy = boxes.loc[boxes.crop_id == row.crop_id,
    ["crop_bbox_x0", "crop_bbox_y0", "crop_bbox_x1", "crop_bbox_y1"]]
```

Open each source once per worker and cache it while consuming nearby manifest
rows. This preserves the storage benefit without repeatedly decoding the same
whole mammogram for every overlapping crop.
"""


__all__ = [
    "ANNOTATION_COLUMNS",
    "CROP_COLUMNS",
    "LAZY_CROP_DIRECTORY",
    "LAZY_CROP_SCHEMA_VERSION",
    "default_lazy_crop_config",
    "estimate_lazy_crop_rows",
    "extract_complete_lazy_crop_family",
    "extract_lazy_crop_manifests",
    "lazy_crop_output_folder",
    "resolve_lazy_crop_dataset_root",
    "scan_lazy_crop_source",
]
