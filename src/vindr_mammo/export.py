from __future__ import annotations

import json
import hashlib
import math
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import OrderedDict
from fractions import Fraction
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy import ndimage as scipy_ndimage
    from scipy.signal import wiener as scipy_wiener
except Exception:  # pragma: no cover
    scipy_ndimage = None
    scipy_wiener = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .crops import (
    _foreground_fraction_from_mask,
    crop_image_and_boxes_to_window,
    exported_boxes_satisfy_bbox_safe_margin,
    sample_bbox_safe_breast_biased_square_window,
    sample_breast_biased_clean_square_window,
    sample_box_centered_square_window,
    sample_random_square_window,
    sliding_square_windows,
    validate_bbox_safe_window,
    window_is_clean,
    window_has_positive_mass,
)
from .dataset import VindrMammoDataset
from .dataset_layout import (
    dataset_content_root,
    resized_variant_configs,
    uses_grouped_dataset_layout,
)
from .pipeline_scope import apply_scoped_steps
from .preprocessing import _breast_chest_wall_side, _robust_tissue_threshold, align_contralateral_image_to_reference, estimate_contralateral_alignment_info

CLASS_NAMES = ["mass"]
COCO_CATEGORIES = [{"id": 1, "name": "mass", "supercategory": "lesion"}]

FLOAT32_EXPORT_VARIANTS = (
    "crops",
    "resized_whole",
    "original_whole",
    "high_resolution_whole",
    "baseline_whole",
)


def float32_export_variant_selected(config: dict[str, Any], variant: str) -> bool:
    """Return the configured per-family selection, ignoring the global switch.

    Configurations written before per-variant selection existed contain no
    ``variants`` mapping and therefore select every image family.
    """

    float_cfg = dict(config.get("float32_export", {}) or {})
    variants = float_cfg.get("variants")
    if not isinstance(variants, dict):
        return True
    return bool(variants.get(str(variant), False))


def float32_export_variant_enabled(config: dict[str, Any], variant: str) -> bool:
    """Return whether one image family receives a float32 companion."""

    float_cfg = dict(config.get("float32_export", {}) or {})
    return bool(float_cfg.get("enabled", False)) and float32_export_variant_selected(
        config, variant
    )


@dataclass
class ExportResult:
    output_root: Path
    created_files: list[Path]
    summary: dict[str, Any]


class SimpleTimerProfiler:
    """Tiny start/stop timer used during export.

    It stores only aggregated totals and counts. The overhead is one
    perf_counter call at start and one at stop for each measured block.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._open: dict[str, float] = {}
        self._stats: dict[str, dict[str, float]] = {}

    def start(self, name: str) -> None:
        if self.enabled:
            self._open[str(name)] = time.perf_counter()

    def stop(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        key = str(name)
        started = self._open.pop(key, None)
        if started is None:
            return 0.0
        elapsed = max(0.0, time.perf_counter() - started)
        item = self._stats.setdefault(key, {"total_seconds": 0.0, "count": 0.0, "max_seconds": 0.0})
        item["total_seconds"] += float(elapsed)
        item["count"] += 1.0
        item["max_seconds"] = max(float(item.get("max_seconds", 0.0)), float(elapsed))
        return elapsed

    def record(self, name: str, elapsed: float) -> None:
        if not self.enabled:
            return
        key = str(name)
        elapsed = max(0.0, float(elapsed))
        item = self._stats.setdefault(key, {"total_seconds": 0.0, "count": 0.0, "max_seconds": 0.0})
        item["total_seconds"] += elapsed
        item["count"] += 1.0
        item["max_seconds"] = max(float(item.get("max_seconds", 0.0)), elapsed)

    def snapshot(self) -> dict[str, Any]:
        total = sum(float(v.get("total_seconds", 0.0)) for v in self._stats.values())
        items = []
        for name, item in sorted(self._stats.items(), key=lambda kv: float(kv[1].get("total_seconds", 0.0)), reverse=True):
            count = int(item.get("count", 0.0))
            total_seconds = float(item.get("total_seconds", 0.0))
            items.append({
                "name": name,
                "total_seconds": total_seconds,
                "count": count,
                "avg_seconds": float(total_seconds / count) if count else 0.0,
                "max_seconds": float(item.get("max_seconds", 0.0)),
                "percent_of_profiled_time": float(100.0 * total_seconds / total) if total > 0 else 0.0,
            })
        return {"enabled": self.enabled, "total_profiled_seconds": float(total), "items": items}


def _profiled_call(profiler: SimpleTimerProfiler, name: str, func):
    profiler.start(name)
    try:
        return func()
    finally:
        profiler.stop(name)


def load_export_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration used by ``main.py``.

    The project uses YAML instead of many constants in ``main.py`` so changing
    output folders, crop size, random crop count, RGB encoding, metadata export,
    and export flags does not require editing Python code.
    """
    if yaml is None:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def export_from_config(config: dict[str, Any], progress_callback: Callable[[dict[str, Any]], None] | None = None) -> ExportResult:
    """Build preprocessed VinDr-Mammo exports from a configuration dictionary.

    The exporter reads every DICOM only once for each required source image and
    writes multiple dataset views from the same processed sample:

    * Ultralytics YOLO labels: ``labels/<split>/*.txt``.
    * MMDetection/COCO labels: ``mmdetection/annotations/instances_<split>.json``.
    * Training images: configurable 8-bit RGB PNGs in ``images/<split>``.
    * Optional preserved images: 16-bit grayscale PNGs in ``preserved_16bit/<split>``.
    * Full metadata: original CSV copies plus JSONL/CSV metadata per exported sample.

    At the very end, the exporter writes ``manifest.json`` and ``EXPORT_DONE.txt``.
    If those files exist, the export reached the final completion step.
    """
    run_started_at = _utc_now_iso()
    total_start = time.perf_counter()
    stage_timings: list[dict[str, Any]] = []

    def timed_stage(name: str, func):
        if progress_callback is not None:
            progress_callback({"event": "stage_start", "stage": name})
        stage_started_at = _utc_now_iso()
        stage_start = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            elapsed = time.perf_counter() - stage_start
            if progress_callback is not None:
                progress_callback({"event": "stage_failed", "stage": name, "error": repr(exc), "elapsed_seconds": float(elapsed)})
            stage_timings.append(
                {
                    "name": name,
                    "started_at": stage_started_at,
                    "finished_at": _utc_now_iso(),
                    "duration_seconds": float(elapsed),
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            raise
        elapsed = time.perf_counter() - stage_start
        if progress_callback is not None:
            progress_callback({"event": "stage_finish", "stage": name, "elapsed_seconds": float(elapsed)})
        stage_timings.append(
            {
                "name": name,
                "started_at": stage_started_at,
                "finished_at": _utc_now_iso(),
                "duration_seconds": float(elapsed),
                "status": "ok",
            }
        )
        return result

    paths = config.get("paths", {})
    data_root = Path(paths.get("data_root", r"G:/vindr"))
    output_root = Path(paths.get("output_root", r"G:/preprocessed-vindr"))

    export_cfg = config.get("export", {})
    if bool(export_cfg.get("require_empty_output_root", False)) and output_root.exists():
        try:
            output_has_files = next(output_root.iterdir(), None) is not None
        except OSError:
            output_has_files = True
        if output_has_files:
            raise FileExistsError(
                "Immutable export target already contains data: "
                f"{output_root}. Choose a new versioned output directory."
            )
    if bool(export_cfg.get("clean_output_root", False)) and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_cfg = config.get("image", {})

    def build_dataset() -> VindrMammoDataset:
        return VindrMammoDataset(
            data_root=data_root,
            index_level="image",
            split=None,
            read_image=True,
            output_size=None,
            normalize=image_cfg.get("normalize", "none"),
            percentile_range=tuple(image_cfg.get("percentile_range", [0.5, 99.5])),
            use_voi_lut=bool(image_cfg.get("use_voi_lut", False)),
            strict_voi_lut=bool(image_cfg.get("strict_voi_lut", False)),
            return_dicom_meta=bool(config.get("metadata", {}).get("include_dicom_meta", True)),
            validate_paths=bool(config.get("dataset", {}).get("validate_paths", False)),
            preprocess_options=config.get("preprocess", {}),
            crop_options={"enabled": False},
            show_progress=bool(config.get("runtime", {}).get("show_progress", True)),
        )

    dataset = timed_stage("initialize_dataset", build_dataset)

    def make_splits():
        cohort_records_, cohort_summary_ = _records_for_source_cohort(dataset, config)
        split_kwargs = normalize_split_strategy_kwargs(config.get("splits", {}))
        split_records_, split_table_ = make_train_val_test_split(
            cohort_records_,
            **split_kwargs,
        )
        split_records_, expansion_summary_ = _expand_training_to_patient_breast_views(
            dataset,
            split_records_,
            config,
        )
        if expansion_summary_.get("enabled", False):
            cohort_summary_["train_patient_breast_expansion"] = expansion_summary_
            split_table_ = _split_assignment_table(split_records_)
        split_records_, split_table_, vendor_summary_ = _apply_vendor_filter_to_splits(dataset, split_records_, split_table_, config)
        contract_report_ = _validate_source_split_contract(dataset, split_records_, config)
        split_path = output_root / "split_assignments.csv"
        split_table_.to_csv(split_path, index=False)
        return split_records_, split_table_, split_path, vendor_summary_, cohort_summary_, contract_report_

    (
        split_records,
        split_table,
        split_path,
        vendor_filter_summary,
        source_cohort_summary,
        source_contract_report,
    ) = timed_stage("make_train_val_test_split", make_splits)

    created_files: list[Path] = [split_path]
    created_files.extend(timed_stage("write_source_metadata_and_config", lambda: _write_global_metadata_files(output_root, dataset, config)))

    crop_cfg_for_summary = config.get("square_crops", {})
    summary: dict[str, Any] = {
        "num_source_images": len(dataset.image_records),
        "num_selected_source_images": sum(len(records) for records in split_records.values()),
        "source_cohort": source_cohort_summary,
        "source_contract": source_contract_report,
        "splits": {k: len(v) for k, v in split_records.items()},
        "rgb_scheme": config.get("image_export", {}).get("rgb_scheme", "multi_window"),
        "histogram_equalization_enabled": bool(config.get("histogram_equalization", {}).get("enabled", True)),
        "square_crop_modes": {
            "train": crop_cfg_for_summary.get("train_crop_mode", "random"),
            "val": crop_cfg_for_summary.get("val_crop_mode", "deterministic"),
            "test": crop_cfg_for_summary.get("test_crop_mode", "deterministic"),
        },
        "deterministic_include_empty": {
            "train": crop_cfg_for_summary.get("train_deterministic_include_empty", crop_cfg_for_summary.get("deterministic_include_empty", True)),
            "val": crop_cfg_for_summary.get("val_deterministic_include_empty", crop_cfg_for_summary.get("deterministic_include_empty", True)),
            "test": crop_cfg_for_summary.get("test_deterministic_include_empty", crop_cfg_for_summary.get("deterministic_include_empty", True)),
        },
        "deterministic_selection_mode": {
            split: _deterministic_selection_mode(crop_cfg_for_summary, split)
            for split in ["train", "val", "test"]
        },
        "deterministic_target_positive_ratio": {
            split: _deterministic_target_positive_ratio(crop_cfg_for_summary, split)
            for split in ["train", "val", "test"]
        },
        "deterministic_target_source_breast_mass_ratio": {
            split: _deterministic_target_source_breast_mass_ratio(
                crop_cfg_for_summary, split
            )
            for split in ["train", "val", "test"]
        },
        "deterministic_negative_keep_fraction": {
            split: _deterministic_negative_keep_fraction(crop_cfg_for_summary, split)
            for split in ["train", "val", "test"]
        },
        "crop_balance_execution": {
            split: _split_crop_cfg(
                crop_cfg_for_summary,
                split,
                "balance_execution",
                "configured_default",
            )
            for split in ["train", "val", "test"]
        },
        "vendor_filter": vendor_filter_summary,
    }

    save_square_crops = bool(export_cfg.get("save_square_crops", True))
    paired_cfg_for_export = dict(config.get("paired_whole_images", {}) or {})
    paired_variants_enabled = bool(paired_cfg_for_export.get("enabled", False)) and any([
        _paired_original_enabled(paired_cfg_for_export),
        _paired_resized_enabled(paired_cfg_for_export),
        _paired_high_resolution_enabled(paired_cfg_for_export),
    ])
    if save_square_crops:
        crop_summary, crop_files = timed_stage(
            "export_square_crops",
            lambda: export_square_crop_datasets(dataset, split_records, config, output_root, progress_callback=progress_callback),
        )
        summary["square_crops"] = crop_summary
        created_files.extend(crop_files)
    elif paired_variants_enabled:
        whole_summary, whole_files = timed_stage(
            "export_whole_image_variants",
            lambda: export_whole_image_variants_only(
                dataset,
                split_records,
                config,
                output_root,
                progress_callback=progress_callback,
            ),
        )
        summary["whole_image_variants"] = whole_summary
        created_files.extend(whole_files)

    if bool(export_cfg.get("save_baseline_uncropped", True)):
        baseline_summary, baseline_files = timed_stage(
            "export_baseline_uncropped",
            lambda: export_baseline_dataset(dataset, split_records, config, output_root),
        )
        summary["baseline_uncropped"] = baseline_summary
        created_files.extend(baseline_files)

    reproducibility_cfg = dict(config.get("reproducibility_bundle", {}) or {})
    if bool(reproducibility_cfg.get("enabled", False)):
        reproducibility_summary, reproducibility_files = timed_stage(
            "write_reproducibility_bundle",
            lambda: _write_reproducibility_bundle(
                output_root=output_root,
                data_root=data_root,
                dataset=dataset,
                split_records=split_records,
                config=config,
            ),
        )
        summary["reproducibility_bundle"] = reproducibility_summary
        created_files.extend(reproducibility_files)

    summary_path = output_root / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)
    created_files.append(summary_path)

    manifest, manifest_files = timed_stage(
        "write_completion_manifest",
        lambda: _write_completion_files(
            output_root=output_root,
            data_root=data_root,
            config=config,
            summary=summary,
            created_files=created_files,
            stage_timings=stage_timings,
            started_at=run_started_at,
            total_duration_seconds=float(time.perf_counter() - total_start),
        ),
    )
    created_files.extend(manifest_files)

    # Keep the returned in-memory summary acyclic. The full manifest already
    # contains a snapshot of ``summary`` and is written to disk, so placing the
    # whole manifest back into summary would create:
    # summary -> manifest -> summary. That breaks Streamlit/JSON display.
    summary["manifest"] = {
        "status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "total_duration_seconds": manifest.get("total_duration_seconds"),
        "total_duration_minutes": manifest.get("total_duration_minutes"),
        "manifest_path": _path_as_posix(output_root / "manifest.json"),
        "done_path": _path_as_posix(output_root / "EXPORT_DONE.txt"),
    }
    return ExportResult(output_root=output_root, created_files=created_files, summary=_json_safe(summary))


def _apply_vendor_filter_to_splits(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    split_table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, dict[str, Any]]:
    """Optionally restrict export to selected vendors/devices.

    The GUI writes this under ``vendor_filter``. If disabled or empty, the
    exporter preserves the original train/val/test split records.
    """
    vendor_cfg = dict(config.get("vendor_filter", {}) or {})
    enabled = bool(vendor_cfg.get("enabled", False))
    include_vendors = [str(v).strip() for v in vendor_cfg.get("include_vendors", []) if str(v).strip()]
    vendor_map = _vendor_map_for_records(dataset, dataset.image_records)
    before_counts = {split: len(records) for split, records in split_records.items()}
    if not enabled or not include_vendors:
        return split_records, split_table, {
            "enabled": False,
            "include_vendors": include_vendors,
            "before_counts": before_counts,
            "after_counts": before_counts,
        }

    include_set = set(include_vendors)
    out: dict[str, list[dict[str, Any]]] = {}
    for split, records in split_records.items():
        out[split] = [r for r in records if vendor_map.get(str(r.get("image_id", "")), "Unknown") in include_set]

    keep_image_ids = {str(r.get("image_id", "")) for records in out.values() for r in records}
    filtered_table = split_table[split_table["image_id"].astype(str).isin(keep_image_ids)].copy()
    after_counts = {split: len(records) for split, records in out.items()}
    return out, filtered_table, {
        "enabled": True,
        "include_vendors": include_vendors,
        "before_counts": before_counts,
        "after_counts": after_counts,
    }


def _vendor_map_for_records(dataset: VindrMammoDataset, records: list[dict[str, Any]]) -> dict[str, str]:
    """Build image_id -> vendor/model label using metadata.csv and DICOM metadata fallbacks."""
    image_ids = {str(r.get("image_id", "")) for r in records}
    out = {image_id: "Unknown" for image_id in image_ids}

    def update(image_id: Any, row: dict[str, Any]) -> None:
        iid = _clean_image_id_for_vendor(image_id)
        if iid is None or iid not in out:
            return
        out[iid] = _vendor_from_metadata_row(row)

    for image_id, rows in getattr(dataset, "metadata_by_image_id", {}).items():
        if rows:
            update(image_id, rows[0])

    metadata_df = getattr(dataset, "metadata_df", pd.DataFrame())
    if isinstance(metadata_df, pd.DataFrame) and not metadata_df.empty:
        for row in metadata_df.to_dict(orient="records"):
            image_id = _first_existing_for_vendor(row, _METADATA_IMAGE_ID_KEYS_EXPORT)
            if image_id is not None:
                update(image_id, row)

    return out


def _vendor_from_metadata_row(row: dict[str, Any]) -> str:
    manufacturer = _first_existing_for_vendor(row, _VENDOR_MANUFACTURER_KEYS_EXPORT)
    model = _first_existing_for_vendor(row, _VENDOR_MODEL_KEYS_EXPORT)
    parts = [_clean_scalar_for_vendor(x) for x in [manufacturer, model]]
    parts = [x for x in parts if x]
    return " / ".join(parts) if parts else "Unknown"


def _first_existing_for_vendor(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _clean_scalar_for_vendor(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _clean_image_id_for_vendor(value: Any) -> str | None:
    text = _clean_scalar_for_vendor(value)
    if text is None:
        return None
    text = text.replace("\\", "/")
    if "/" in text:
        text = Path(text).stem
    if text.endswith(".dicom") or text.endswith(".dcm"):
        text = Path(text).stem
    return text or None


_VENDOR_MANUFACTURER_KEYS_EXPORT = [
    "Manufacturer", "manufacturer", "ManufacturerName", "manufacturer_name",
    "DeviceManufacturer", "device_manufacturer", "vendor", "Vendor",
]
_VENDOR_MODEL_KEYS_EXPORT = [
    "ManufacturerModelName", "Manufacturer's Model Name", "manufacturer_model_name",
    "ModelName", "model_name", "model", "Model", "DeviceModel", "device_model",
]
_METADATA_IMAGE_ID_KEYS_EXPORT = [
    "image_id", "ImageID", "imageId", "SOPInstanceUID", "sop_instance_uid",
    "SOP Instance UID", "filename", "file_name", "FileName", "dicom_path", "path",
]


def normalize_split_strategy_kwargs(split_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the configured source split strategy for the split builder.

    Configurations written before ``splits.strategy`` existed are inferred from
    their count overrides first, then from their validation fraction. This keeps
    old presets and saved manifests reproducible while making the three current
    strategies mutually exclusive.
    """
    cfg = dict(split_cfg or {})
    strategy_value = cfg.get("strategy")
    if strategy_value is None or not str(strategy_value).strip():
        has_count_override = any(
            key in cfg and cfg.get(key) is not None
            for key in ("validation_study_count", "validation_image_count")
        )
        if has_count_override:
            strategy = "exact_study_count"
        elif float(cfg.get("val_fraction_from_training", 0.15)) > 0:
            strategy = "random_study_fraction"
        else:
            strategy = "official_only"
    else:
        strategy = str(strategy_value).strip().casefold()

    allowed = {"official_only", "random_study_fraction", "exact_study_count"}
    if strategy not in allowed:
        raise ValueError(
            f"Unknown splits.strategy={strategy!r}; expected one of {sorted(allowed)}."
        )

    kwargs: dict[str, Any] = {
        "seed": int(cfg.get("seed", 123)),
        "stratify_by_birads": bool(cfg.get("stratify_by_birads", False)),
    }
    if strategy == "official_only":
        kwargs.update(
            val_fraction=0.0,
            validation_study_count=0,
            validation_image_count=0,
        )
    elif strategy == "random_study_fraction":
        kwargs.update(
            val_fraction=float(cfg.get("val_fraction_from_training", 0.15)),
            validation_study_count=None,
            validation_image_count=None,
        )
    else:
        validation_study_count = cfg.get("validation_study_count")
        if validation_study_count is None:
            raise ValueError(
                "splits.strategy='exact_study_count' requires validation_study_count."
            )
        validation_image_count = cfg.get("validation_image_count")
        kwargs.update(
            val_fraction=0.0,
            validation_study_count=int(validation_study_count),
            validation_image_count=(
                None if validation_image_count is None else int(validation_image_count)
            ),
        )
    return kwargs


def make_train_val_test_split(
    image_records: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
    stratify_by_birads: bool = False,
    validation_study_count: int | None = None,
    validation_image_count: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Use VinDr's official test split and make a study-level val split from training.

    VinDr-Mammo already contains official ``training`` and ``test`` values. This
    function keeps official ``test`` untouched, then splits official
    ``training`` studies into train and val. When ``stratify_by_birads`` is
    enabled, validation studies are sampled proportionally within study-level
    BI-RADS strata. Splitting by ``study_id`` prevents views from the same exam
    leaking across train and validation.
    """
    rng = np.random.default_rng(seed)
    training_records = []
    test_records = []

    for record in image_records:
        split = str(record.get("split", "training")).casefold().strip()
        if split == "test":
            test_records.append(record)
        else:
            training_records.append(record)

    study_ids = sorted({str(r.get("study_id")) for r in training_records})
    if validation_study_count is None:
        n_val = int(round(float(val_fraction) * len(study_ids)))
        n_val = max(1, n_val) if len(study_ids) > 1 and val_fraction > 0 else 0
    else:
        n_val = int(validation_study_count)
        if not 0 <= n_val <= len(study_ids):
            raise ValueError(
                f"validation_study_count={n_val} is outside [0, {len(study_ids)}]."
            )

    target_val_images = None if validation_image_count is None else int(validation_image_count)
    if target_val_images is not None and target_val_images < 0:
        raise ValueError("validation_image_count must be non-negative.")

    if n_val > 0 and target_val_images is not None:
        val_ids = _count_matched_study_sample(
            training_records,
            n_val=n_val,
            n_val_images=target_val_images,
            rng=rng,
            stratify_by_birads=stratify_by_birads,
        )
    elif stratify_by_birads and n_val > 0:
        val_ids = _stratified_study_sample(training_records, n_val=n_val, rng=rng)
    else:
        shuffled = np.array(study_ids, dtype=object)
        rng.shuffle(shuffled)
        val_ids = set(str(x) for x in shuffled[:n_val])

    out = {"train": [], "val": [], "test": []}
    for record in training_records:
        export_split = "val" if str(record.get("study_id")) in val_ids else "train"
        out[export_split].append(record)
    out["test"].extend(test_records)

    if target_val_images is not None and len(out["val"]) != target_val_images:
        raise RuntimeError(
            f"Count-matched split selected {len(out['val'])} validation images; "
            f"expected {target_val_images}."
        )

    return out, _split_assignment_table(out)


def _split_assignment_table(
    split_records: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Build the persisted image assignment table from final split records."""
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        for record in split_records.get(split_name, []):
            rows.append({
                "export_split": split_name,
                "official_split": record.get("split"),
                "study_id": str(record.get("study_id")),
                "image_id": str(record.get("image_id")),
                "laterality": record.get("laterality"),
                "view_position": record.get("view_position"),
                "source_breast_key": record.get("_source_breast_key", ""),
                "source_breast_has_mass": record.get("_source_breast_has_mass", ""),
            })
    return pd.DataFrame(rows)


def _breast_key(record: dict[str, Any]) -> tuple[str, str]:
    """Return VinDr's patient/exam plus laterality breast identity."""
    study_id = str(record.get("study_id", "")).strip()
    laterality = str(record.get("laterality", "")).strip().upper()[:1]
    if not study_id or laterality not in {"L", "R"}:
        raise ValueError(
            "Breast-aware selection requires non-empty study_id and L/R laterality; "
            f"got study_id={study_id!r}, laterality={record.get('laterality')!r}."
        )
    return study_id, laterality


def _mass_positive_breast_keys(dataset: VindrMammoDataset) -> set[tuple[str, str]]:
    positive: set[tuple[str, str]] = set()
    for record in dataset.image_records:
        image_id = str(record.get("image_id", ""))
        if dataset._filter_mass_findings(dataset.findings_by_image_id.get(image_id, [])):
            positive.add(_breast_key(record))
    return positive


def _expand_training_to_patient_breast_views(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Expand only selected training patients and label views by breast status.

    The validation and test lists are deliberately copied without changing their
    image membership.  Training is expanded to every official-training view of
    the already selected training studies.  A breast is ``(study_id,
    laterality)`` and is mass-positive when any of its views has a valid Mass.
    """
    cohort_cfg = dict(config.get("source_cohort", {}) or {})
    enabled = bool(cohort_cfg.get("train_expand_to_all_patient_breast_views", False))
    if not enabled:
        return split_records, {"enabled": False}
    unit = str(cohort_cfg.get("train_breast_status_unit", "study_laterality") or "").strip().casefold()
    if unit not in {"study_laterality", "study+laterality", "patient_laterality"}:
        raise ValueError(
            "source_cohort.train_breast_status_unit must be 'study_laterality'."
        )

    train_studies = {
        str(record.get("study_id", ""))
        for record in split_records.get("train", [])
    }
    val_studies = {
        str(record.get("study_id", ""))
        for record in split_records.get("val", [])
    }
    test_studies = {
        str(record.get("study_id", ""))
        for record in split_records.get("test", [])
    }
    if train_studies & val_studies or train_studies & test_studies or val_studies & test_studies:
        raise ValueError("Cannot expand training breasts because study IDs overlap across splits.")

    positive_breasts = _mass_positive_breast_keys(dataset)

    def annotated(record: dict[str, Any]) -> dict[str, Any]:
        key = _breast_key(record)
        image_id = str(record.get("image_id", ""))
        image_has_mass = bool(
            dataset._filter_mass_findings(
                dataset.findings_by_image_id.get(image_id, [])
            )
        )
        return {
            **record,
            "_source_image_has_mass": int(image_has_mass),
            "_source_breast_key": f"{key[0]}:{key[1]}",
            "_source_breast_has_mass": int(key in positive_breasts),
        }

    expanded_train = [
        annotated(record)
        for record in dataset.image_records
        if str(record.get("split", "training")).casefold().strip() != "test"
        and str(record.get("study_id", "")) in train_studies
    ]
    out = {
        "train": expanded_train,
        "val": [annotated(record) for record in split_records.get("val", [])],
        "test": [annotated(record) for record in split_records.get("test", [])],
    }
    train_breast_keys = {_breast_key(record) for record in expanded_train}
    mass_breasts = train_breast_keys & positive_breasts
    negative_breasts = train_breast_keys - positive_breasts
    mass_views = sum(int(record["_source_breast_has_mass"]) for record in expanded_train)
    negative_views = len(expanded_train) - mass_views
    return out, {
        "enabled": True,
        "unit": "study_id+laterality",
        "train_studies": len(train_studies),
        "selected_positive_images_before_expansion": len(split_records.get("train", [])),
        "expanded_train_views": len(expanded_train),
        "mass_positive_breasts": len(mass_breasts),
        "negative_breasts": len(negative_breasts),
        "mass_positive_breast_views": mass_views,
        "negative_breast_views": negative_views,
        "validation_membership_changed": False,
        "test_membership_changed": False,
    }


def _stratified_study_sample(
    training_records: list[dict[str, Any]], *, n_val: int, rng: np.random.Generator
) -> set[str]:
    """Sample validation studies within their study-level BI-RADS strata."""
    records_by_study: dict[str, list[dict[str, Any]]] = {}
    for record in training_records:
        records_by_study.setdefault(str(record.get("study_id")), []).append(record)

    strata: dict[str, list[str]] = {}
    for study_id, records in records_by_study.items():
        labels = sorted({
            str(record.get("breast_birads", "unknown")).strip().casefold() or "unknown"
            for record in records
        })
        strata.setdefault("|".join(labels), []).append(study_id)

    total = max(1, len(records_by_study))
    allocation: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label, ids in strata.items():
        exact = float(n_val) * len(ids) / total
        base = min(len(ids), int(math.floor(exact)))
        allocation[label] = base
        remainders.append((exact - base, label))

    remaining = max(0, int(n_val) - sum(allocation.values()))
    for _remainder, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocation[label] < len(strata[label]):
            allocation[label] += 1
            remaining -= 1

    selected: set[str] = set()
    for label in sorted(strata):
        ids = np.asarray(sorted(strata[label]), dtype=object)
        rng.shuffle(ids)
        selected.update(str(value) for value in ids[: allocation[label]])
    return selected


def _records_for_source_cohort(
    dataset: VindrMammoDataset,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply an explicit finding-positive cohort before any split is made."""
    cohort_cfg = dict(config.get("source_cohort", {}) or {})
    positive_only = bool(cohort_cfg.get("positive_images_only", False))
    category = str(cohort_cfg.get("finding_category", "Mass") or "Mass").strip()
    all_records = list(dataset.image_records)
    if not positive_only:
        return all_records, {
            "finding_category": category,
            "positive_images_only": False,
            "input_images": len(all_records),
            "selected_images": len(all_records),
        }
    if category.casefold() != "mass":
        raise ValueError(
            "source_cohort.positive_images_only currently supports finding_category='Mass' only."
        )

    positive_ids: set[str] = set()
    valid_annotations = 0
    for image_id, findings in dataset.findings_by_image_id.items():
        mass_findings = dataset._filter_mass_findings(findings)
        if mass_findings:
            positive_ids.add(str(image_id))
            valid_annotations += len(mass_findings)
    selected = [r for r in all_records if str(r.get("image_id", "")) in positive_ids]
    by_official_split: dict[str, int] = {}
    for record in selected:
        split = str(record.get("split", ""))
        by_official_split[split] = by_official_split.get(split, 0) + 1
    return selected, {
        "finding_category": category,
        "positive_images_only": True,
        "input_images": len(all_records),
        "selected_images": len(selected),
        "selected_studies": len({str(r.get("study_id", "")) for r in selected}),
        "valid_source_annotations": int(valid_annotations),
        "selected_images_by_official_split": by_official_split,
    }


def _validate_source_split_contract(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fail early when an enabled replication source/split contract is violated."""
    contract = dict(config.get("replication_contract", {}) or {})
    if not bool(contract.get("enabled", False)):
        return {"enabled": False, "status": "not_requested"}

    expected_images = dict(contract.get("expected_source_images", {}) or {})
    expected_studies = dict(contract.get("expected_source_studies", {}) or {})
    expected_annotations = dict(contract.get("expected_source_annotations", {}) or {})
    require_positive = bool(contract.get("require_positive_source_images", False))
    require_positive_by_split = dict(
        contract.get("require_positive_source_images_by_split", {}) or {}
    )
    expected_train_breasts = dict(contract.get("expected_train_breasts", {}) or {})
    expected_train_views = dict(
        contract.get("expected_train_source_views_by_breast_status", {}) or {}
    )
    errors: list[str] = []
    observed: dict[str, Any] = {}
    images_seen: dict[str, set[str]] = {}
    studies_seen: dict[str, set[str]] = {}
    positive_breast_keys = _mass_positive_breast_keys(dataset)

    for split_name in ["train", "val", "test"]:
        records = list(split_records.get(split_name, []))
        image_ids = {str(record.get("image_id", "")) for record in records}
        study_ids = {str(record.get("study_id", "")) for record in records}
        images_seen[split_name] = image_ids
        studies_seen[split_name] = study_ids
        annotation_count = 0
        nonpositive: list[str] = []
        mass_breast_views = 0
        negative_breast_views = 0
        mass_breasts: set[tuple[str, str]] = set()
        negative_breasts: set[tuple[str, str]] = set()
        records_by_study: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            records_by_study.setdefault(str(record.get("study_id", "")), []).append(record)
        stratum_counts: dict[str, int] = {}
        for study_records in records_by_study.values():
            stratum = _study_birads_stratum(study_records)
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        for image_id in image_ids:
            findings = dataset.findings_by_image_id.get(image_id, [])
            count = len(dataset._filter_mass_findings(findings))
            annotation_count += count
            if count == 0:
                nonpositive.append(image_id)
        for record in records:
            key = _breast_key(record)
            if key in positive_breast_keys:
                mass_breast_views += 1
                mass_breasts.add(key)
            else:
                negative_breast_views += 1
                negative_breasts.add(key)
        observed[split_name] = {
            "source_images": len(image_ids),
            "source_studies": len(study_ids),
            "source_annotations": int(annotation_count),
            "nonpositive_source_images": len(nonpositive),
            "mass_positive_breasts": len(mass_breasts),
            "negative_breasts": len(negative_breasts),
            "mass_positive_breast_views": mass_breast_views,
            "negative_breast_views": negative_breast_views,
            "study_birads_strata": dict(sorted(stratum_counts.items())),
        }
        if split_name in expected_images and len(image_ids) != int(expected_images[split_name]):
            errors.append(
                f"{split_name}: expected {int(expected_images[split_name])} source images, got {len(image_ids)}"
            )
        if split_name in expected_studies and len(study_ids) != int(expected_studies[split_name]):
            errors.append(
                f"{split_name}: expected {int(expected_studies[split_name])} source studies, got {len(study_ids)}"
            )
        if split_name in expected_annotations and annotation_count != int(expected_annotations[split_name]):
            errors.append(
                f"{split_name}: expected {int(expected_annotations[split_name])} source annotations, got {annotation_count}"
            )
        split_requires_positive = bool(
            require_positive_by_split.get(split_name, require_positive)
        )
        if split_requires_positive and nonpositive:
            errors.append(
                f"{split_name}: {len(nonpositive)} source images have no valid Mass annotation"
            )
        if split_name == "train":
            expected_mass_breasts = expected_train_breasts.get("mass")
            expected_negative_breasts = expected_train_breasts.get("negative")
            expected_mass_views = expected_train_views.get("mass")
            expected_negative_views = expected_train_views.get("negative")
            if expected_mass_breasts is not None and len(mass_breasts) != int(expected_mass_breasts):
                errors.append(
                    f"train: expected {int(expected_mass_breasts)} mass-positive breasts, got {len(mass_breasts)}"
                )
            if expected_negative_breasts is not None and len(negative_breasts) != int(expected_negative_breasts):
                errors.append(
                    f"train: expected {int(expected_negative_breasts)} negative breasts, got {len(negative_breasts)}"
                )
            if expected_mass_views is not None and mass_breast_views != int(expected_mass_views):
                errors.append(
                    f"train: expected {int(expected_mass_views)} mass-breast views, got {mass_breast_views}"
                )
            if expected_negative_views is not None and negative_breast_views != int(expected_negative_views):
                errors.append(
                    f"train: expected {int(expected_negative_views)} negative-breast views, got {negative_breast_views}"
                )

    split_names = ["train", "val", "test"]
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            image_overlap = images_seen[left] & images_seen[right]
            study_overlap = studies_seen[left] & studies_seen[right]
            if image_overlap:
                errors.append(f"{left}/{right}: {len(image_overlap)} source image IDs overlap")
            if study_overlap:
                errors.append(f"{left}/{right}: {len(study_overlap)} source study IDs overlap")

    if bool(contract.get("preserve_official_test", False)):
        test_requires_positive = bool(require_positive_by_split.get("test", require_positive))
        official_test_ids = {
            str(record.get("image_id", ""))
            for record in dataset.image_records
            if str(record.get("split", "")).casefold().strip() == "test"
            and (
                not test_requires_positive
                or dataset._filter_mass_findings(
                    dataset.findings_by_image_id.get(str(record.get("image_id", "")), [])
                )
            )
        }
        observed_test_ids = images_seen.get("test", set())
        wrong_test = [
            record for record in split_records.get("test", [])
            if str(record.get("split", "")).casefold().strip() != "test"
        ]
        displaced_test = [
            record
            for split_name in ["train", "val"]
            for record in split_records.get(split_name, [])
            if str(record.get("split", "")).casefold().strip() == "test"
        ]
        if wrong_test or displaced_test or observed_test_ids != official_test_ids:
            errors.append("official VinDr test membership was not preserved")

    report = {
        "enabled": True,
        "name": contract.get("name", "replication_contract"),
        "status": "pass" if not errors else "fail",
        "observed": observed,
        "errors": errors,
    }
    if errors and bool(contract.get("strict", True)):
        raise ValueError("Replication source/split contract failed: " + "; ".join(errors))
    return report


def _study_birads_stratum(records: list[dict[str, Any]]) -> str:
    labels = sorted({
        str(record.get("breast_birads", "unknown")).strip().casefold() or "unknown"
        for record in records
    })
    return "|".join(labels)


def _stratum_allocations(strata: dict[str, list[str]], n_val: int) -> dict[str, int]:
    total = max(1, sum(len(ids) for ids in strata.values()))
    allocation: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label, ids in strata.items():
        exact = float(n_val) * len(ids) / total
        base = min(len(ids), int(math.floor(exact)))
        allocation[label] = base
        remainders.append((exact - base, label))
    remaining = max(0, int(n_val) - sum(allocation.values()))
    for _remainder, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocation[label] < len(strata[label]):
            allocation[label] += 1
            remaining -= 1
    return allocation


def _count_matched_study_sample(
    training_records: list[dict[str, Any]],
    *,
    n_val: int,
    n_val_images: int,
    rng: np.random.Generator,
    stratify_by_birads: bool,
) -> set[str]:
    """Select exact validation study/image counts with seeded tie-breaking.

    Paper 22 does not publish its split identities or seed. The result is thus a
    deterministic count-matched split, kept as close as possible to proportional
    study-level BI-RADS strata, rather than a claim of author-identical IDs.
    """
    records_by_study: dict[str, list[dict[str, Any]]] = {}
    for record in training_records:
        records_by_study.setdefault(str(record.get("study_id")), []).append(record)
    if n_val == 0:
        if n_val_images != 0:
            raise ValueError("Cannot select validation images when validation_study_count is zero.")
        return set()

    study_sizes = [len(records) for records in records_by_study.values()]
    min_images = sum(sorted(study_sizes)[:n_val])
    max_images = sum(sorted(study_sizes, reverse=True)[:n_val])
    if not min_images <= n_val_images <= max_images:
        raise ValueError(
            f"No {n_val}-study split can contain {n_val_images} images; feasible range is "
            f"[{min_images}, {max_images}]."
        )

    strata: dict[str, list[str]] = {}
    for study_id, records in records_by_study.items():
        label = _study_birads_stratum(records) if stratify_by_birads else "all"
        strata.setdefault(label, []).append(study_id)
    allocation = _stratum_allocations(strata, n_val)

    all_ids = sorted(records_by_study)
    bit_for = {study_id: 1 << index for index, study_id in enumerate(all_ids)}
    priority = {study_id: float(rng.random()) for study_id in all_ids}

    options_by_stratum: dict[str, dict[tuple[int, int], tuple[float, int]]] = {}
    for label in sorted(strata):
        states: dict[tuple[int, int], tuple[float, int]] = {(0, 0): (0.0, 0)}
        for study_id in sorted(strata[label]):
            image_count = len(records_by_study[study_id])
            additions: dict[tuple[int, int], tuple[float, int]] = {}
            for (count, images), (score, mask) in list(states.items()):
                new_key = (count + 1, images + image_count)
                if new_key[0] > n_val or new_key[1] > n_val_images:
                    continue
                candidate = (score + priority[study_id], mask | bit_for[study_id])
                current = states.get(new_key) or additions.get(new_key)
                if current is None or candidate[0] < current[0]:
                    additions[new_key] = candidate
            for key, candidate in additions.items():
                current = states.get(key)
                if current is None or candidate[0] < current[0]:
                    states[key] = candidate
        options_by_stratum[label] = states

    chosen_mask: int | None = None
    max_radius = max(n_val, max(allocation.values(), default=0))
    for radius in range(max_radius + 1):
        combined: dict[tuple[int, int], tuple[float, int]] = {(0, 0): (0.0, 0)}
        for label in sorted(strata):
            target = allocation[label]
            options = [
                (key, value)
                for key, value in options_by_stratum[label].items()
                if abs(key[0] - target) <= radius
            ]
            next_combined: dict[tuple[int, int], tuple[float, int]] = {}
            for (total_count, total_images), (total_score, total_mask) in combined.items():
                for (count, images), (score, mask) in options:
                    new_key = (total_count + count, total_images + images)
                    if new_key[0] > n_val or new_key[1] > n_val_images:
                        continue
                    penalty = 1000.0 * float((count - target) ** 2)
                    candidate = (total_score + score + penalty, total_mask | mask)
                    current = next_combined.get(new_key)
                    if current is None or candidate[0] < current[0]:
                        next_combined[new_key] = candidate
            combined = next_combined
            if not combined:
                break
        match = combined.get((n_val, n_val_images))
        if match is not None:
            chosen_mask = match[1]
            break

    if chosen_mask is None:
        raise RuntimeError(
            f"Could not construct an exact count-matched split of {n_val} studies / "
            f"{n_val_images} images."
        )
    return {study_id for study_id in all_ids if chosen_mask & bit_for[study_id]}


def export_square_crop_datasets(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    output_root: Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Export n x n crop datasets.

    The crop mode for each split is controlled by ``square_crops`` config keys:

    * ``train_crop_mode``: ``"random"``, ``"deterministic"``, or ``"bbox_safe_random"``.
    * ``val_crop_mode``: usually ``"deterministic"``, but can also use ``"bbox_safe_random"``.
    * ``test_crop_mode``: usually ``"deterministic"``, but can also use ``"bbox_safe_random"``.

    Earlier versions always used random, mass-centered training crops and
    deterministic validation/test crops. That can create a strong distribution
    mismatch, so deterministic training crops are now supported directly.
    """
    if uses_grouped_dataset_layout(config):
        raise ValueError(
            "The images/annotations dataset layout represents windows as metadata-only "
            "annotation manifests. Disable export.save_square_crops and configure "
            "lazy_crop_grids instead of materializing duplicate crop pixels."
        )
    crop_root = dataset_content_root(output_root, config)
    crop_root.mkdir(parents=True, exist_ok=True)
    crop_cfg = dict(config.get("square_crops", {}))
    crop_size = int(crop_cfg.get("crop_size", 1024))
    stride = int(crop_cfg.get("stride", 512))
    size_divisor = max(1, int(crop_cfg.get("size_divisor", 1) or 1))
    if crop_size % size_divisor != 0 or stride % size_divisor != 0:
        raise ValueError(
            "square_crops crop_size and stride must both be divisible by "
            f"size_divisor={size_divisor}; got crop_size={crop_size}, "
            f"stride={stride}."
        )
    common_crop_options = dict(config.get("crop_annotation_policy", {}))
    common_crop_options.update(
        {
            "enabled": True,
            "crop_size": crop_size,
            "stride": stride,
            "edge_policy": str(crop_cfg.get("edge_policy", "edge_align")),
            "pad_if_needed": bool(crop_cfg.get("pad_if_needed", True)),
            "pad_value": float(crop_cfg.get("pad_value", 0.0)),
            "allow_partial_annotations": bool(common_crop_options.get("allow_partial_annotations", False)),
            "min_box_visibility": float(common_crop_options.get("min_box_visibility", 0.30)),
            "reject_partial_windows": bool(common_crop_options.get("reject_partial_windows", True)),
            "negative_max_box_visibility": float(common_crop_options.get("negative_max_box_visibility", 0.0)),
            "max_random_tries": int(crop_cfg.get("max_random_tries", 80)),
            "center_shift_fraction": float(crop_cfg.get("center_shift_fraction", 0.25)),
        }
    )

    rng = np.random.default_rng(int(crop_cfg.get("seed", 123)))
    save_empty_labels = bool(config.get("export", {}).get("save_empty_label_files", True))
    runtime_cfg = dict(config.get("runtime", {}) or {})
    show_progress = bool(runtime_cfg.get("show_progress", True))
    profiler = SimpleTimerProfiler(enabled=bool(runtime_cfg.get("simple_profiler_enabled", True)))
    profiler_emit_every = max(1, int(runtime_cfg.get("simple_profiler_emit_every", 10)))
    profiler_event_counter = 0

    coco_by_split = {split: _empty_coco() for split in ["train", "val", "test"]}
    stats_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    online_balance_summaries: dict[str, dict[str, Any]] = {}
    image_id_counter = 1
    ann_id_counter = 1
    # Keep a tiny cache only. A full VinDr export can contain thousands of large
    # mammograms, so caching every tensor would consume too much RAM. The small
    # cache still avoids immediate rereads when paired views are processed near
    # each other.
    preprocessed_cache: OrderedDict[str, tuple[torch.Tensor, dict[str, Any]]] = OrderedDict()
    max_preprocessed_cache_items = int(crop_cfg.get("contralateral_preprocessed_cache_items", 16))
    aligned_contralateral_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    max_aligned_contralateral_cache_items = int(crop_cfg.get("contralateral_alignment_cache_items", 2))
    whole_stage_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
    max_whole_stage_cache_items = int(crop_cfg.get("whole_stage_cache_items", 12))
    paired_whole_cfg = dict(config.get("paired_whole_images", {}) or {})
    paired_whole_source_paths: dict[tuple[str, ...], Path] = {}
    review_cfg = dict(config.get("dataset_review", {}) or {})
    review_enabled = bool(review_cfg.get("enabled", False))
    review_source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    review_source_asset_counts = {split: 0 for split in ["train", "val", "test"]}
    review_source_asset_limit_raw = review_cfg.get(
        "source_assets_per_split",
        review_cfg.get("samples_per_split", 100),
    )
    review_source_asset_limit = (
        None
        if review_source_asset_limit_raw is None
        else max(0, int(review_source_asset_limit_raw))
    )
    annotation_report_cfg = dict(config.get("annotation_geometry_report", {}) or {})
    annotation_report_enabled = bool(annotation_report_cfg.get("enabled", False))
    annotation_geometry_rows: list[dict[str, Any]] = []
    annotation_geometry_seen_sources: set[tuple[str, str]] = set()
    source_image_has_mass_lookup = {
        str(record.get("image_id", "")): int(
            bool(
                dataset._filter_mass_findings(
                    dataset.findings_by_image_id.get(
                        str(record.get("image_id", "")),
                        [],
                    )
                )
            )
        )
        for record in dataset.image_records
    }
    source_breast_has_mass_lookup = _mass_positive_breast_keys(dataset)
    contralateral_lookup = _build_contralateral_record_lookup(dataset.image_records)
    needs_contralateral = _config_uses_contralateral_source(config)
    total_records_for_progress = sum(len(split_records.get(split, [])) for split in ["train", "val", "test"])
    processed_records_for_progress = 0

    source_index_lookup: dict[tuple[str, str], int] = {}
    source_debug: dict[tuple[str, str], dict[str, Any]] = {}
    saved_window_keys: set[tuple[str, str, int, int, int, int]] = set()
    candidate_positive_window_keys: set[tuple[str, str, int, int, int, int]] = set()
    saved_positive_window_keys: set[tuple[str, str, int, int, int, int]] = set()
    for _split_name, _records in split_records.items():
        for _idx, _record in enumerate(_records):
            _image_id = str(_record.get("image_id", ""))
            _breast_has_mass = int(
                _breast_key(_record) in source_breast_has_mass_lookup
            )
            source_index_lookup[(_split_name, _image_id)] = int(_idx)
            source_debug[(_split_name, _image_id)] = {
                "split": _split_name,
                "source_index": int(_idx),
                "source_image_id": _image_id,
                "source_study_id": str(_record.get("study_id", "")),
                "laterality": _record.get("laterality", ""),
                "view_position": _record.get("view_position", ""),
                "official_split": _record.get("split", ""),
                "source_breast_key": _record.get("_source_breast_key", ""),
                "source_image_has_mass": int(
                    source_image_has_mass_lookup.get(_image_id, 0)
                ),
                "source_breast_has_mass": _breast_has_mass,
                "processed_source_image": 0,
                "n_source_mass_boxes": 0,
                "has_source_mass": 0,
                "candidate_windows": 0,
                "complete_grid_windows": 0,
                "positive_candidate_windows": 0,
                "ambiguous_partial_windows": 0,
                "attempted_save_windows": 0,
                "saved_crops": 0,
                "saved_positive_crops": 0,
                "saved_negative_crops": 0,
                "exported_mass_box_instances": 0,
                "skipped_windows": 0,
                "skip_foreground_too_low": 0,
                "skip_bbox_safe_failed": 0,
                "skip_duplicate_window": 0,
                "skip_empty_disallowed": 0,
                "skip_blank_output": 0,
                "_included_annotation_indices": set(),
            }

    def source_row_for(split_name_: str, record_: dict[str, Any]) -> dict[str, Any]:
        image_id_ = str(record_.get("image_id", ""))
        key_ = (split_name_, image_id_)
        if key_ not in source_debug:
            breast_has_mass_ = int(
                _breast_key(record_) in source_breast_has_mass_lookup
            )
            source_debug[key_] = {
                "split": split_name_,
                "source_index": int(source_index_lookup.get(key_, -1)),
                "source_image_id": image_id_,
                "source_study_id": str(record_.get("study_id", "")),
                "laterality": record_.get("laterality", ""),
                "view_position": record_.get("view_position", ""),
                "official_split": record_.get("split", ""),
                "source_breast_key": record_.get("_source_breast_key", ""),
                "source_image_has_mass": int(
                    source_image_has_mass_lookup.get(image_id_, 0)
                ),
                "source_breast_has_mass": breast_has_mass_,
                "processed_source_image": 0,
                "n_source_mass_boxes": 0,
                "has_source_mass": 0,
                "candidate_windows": 0,
                "complete_grid_windows": 0,
                "positive_candidate_windows": 0,
                "ambiguous_partial_windows": 0,
                "attempted_save_windows": 0,
                "saved_crops": 0,
                "saved_positive_crops": 0,
                "saved_negative_crops": 0,
                "exported_mass_box_instances": 0,
                "skipped_windows": 0,
                "skip_foreground_too_low": 0,
                "skip_bbox_safe_failed": 0,
                "skip_duplicate_window": 0,
                "skip_empty_disallowed": 0,
                "skip_blank_output": 0,
                "_included_annotation_indices": set(),
            }
        return source_debug[key_]

    def note_source_target(
        split_name_: str,
        record_: dict[str, Any],
        target_: dict[str, Any],
        image_: torch.Tensor | None = None,
        *,
        save_review_assets: bool = True,
    ) -> dict[str, Any]:
        row_ = source_row_for(split_name_, record_)
        boxes_ = target_.get("mass", {}).get("boxes", torch.zeros((0, 4)))
        try:
            n_mass_ = int(boxes_.detach().cpu().reshape(-1, 4).shape[0])
        except Exception:
            n_mass_ = 0
        row_["processed_source_image"] = 1
        row_["n_source_mass_boxes"] = max(int(row_.get("n_source_mass_boxes", 0)), n_mass_)
        row_["has_source_mass"] = int(row_["n_source_mass_boxes"] > 0)
        # Keep source_image_has_mass independent: it was computed directly
        # from the source finding table before preprocessing. This separate
        # target-derived flag is useful for detecting provenance drift.
        row_["preprocessed_target_has_mass"] = int(n_mass_ > 0)
        review_key_ = (split_name_, str(record_.get("image_id", "")))
        if annotation_report_enabled and review_key_ not in annotation_geometry_seen_sources:
            annotation_geometry_seen_sources.add(review_key_)
            try:
                mass_array = boxes_.detach().cpu().to(torch.float32).reshape(-1, 4).numpy()
            except Exception:
                mass_array = np.zeros((0, 4), dtype=np.float32)
            source_height = int(image_.shape[-2]) if image_ is not None else 0
            source_width = int(image_.shape[-1]) if image_ is not None else 0
            for annotation_index, box in enumerate(mass_array):
                x0, y0, x1, y1 = [float(value) for value in box]
                width = max(0.0, x1 - x0)
                height = max(0.0, y1 - y0)
                annotation_geometry_rows.append({
                    "split": split_name_,
                    "source_image_id": str(record_.get("image_id", "")),
                    "source_study_id": str(record_.get("study_id", "")),
                    "laterality": record_.get("laterality", ""),
                    "view_position": record_.get("view_position", ""),
                    "source_annotation_index": int(annotation_index),
                    "coordinate_space": "fixed_preprocessed_source",
                    "source_width_px": source_width,
                    "source_height_px": source_height,
                    "bbox_x0": x0,
                    "bbox_y0": y0,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_width_px": width,
                    "bbox_height_px": height,
                    "bbox_area_px2": width * height,
                })
        review_limit_available_ = (
            review_source_asset_limit is None
            or review_source_asset_counts[split_name_] < review_source_asset_limit
        )
        if (
            review_enabled
            and save_review_assets
            and review_limit_available_
            and image_ is not None
            and review_key_ not in review_source_rows
        ):
            review_source_rows[review_key_] = _save_review_source_assets(
                crop_root=crop_root,
                split_name=split_name_,
                record=record_,
                image=image_,
                target=target_,
                review_cfg=review_cfg,
            )
            review_source_asset_counts[split_name_] += 1
        return row_

    def mark_source_skip(split_name_: str, record_: dict[str, Any], reason: str) -> None:
        row_ = source_row_for(split_name_, record_)
        row_["skipped_windows"] = int(row_.get("skipped_windows", 0)) + 1
        key_name = f"skip_{reason}"
        if key_name in row_:
            row_[key_name] = int(row_.get(key_name, 0)) + 1

    def note_candidate_windows(
        split_name_: str,
        record_: dict[str, Any],
        windows_: list[tuple[tuple[int, int, int, int], dict[str, Any]]],
    ) -> None:
        row_ = source_row_for(split_name_, record_)
        row_["candidate_windows"] = int(row_.get("candidate_windows", 0)) + len(windows_)
        image_id_ = str(record_.get("image_id", ""))
        for window_, info_ in windows_:
            if int(info_.get("is_positive_window", 0)) != 1:
                continue
            candidate_positive_window_keys.add((
                split_name_, image_id_,
                int(window_[0]), int(window_[1]), int(window_[2]), int(window_[3]),
            ))

    def profiler_snapshot_event() -> dict[str, Any]:
        return {"simple_profiler": profiler.snapshot()} if profiler.enabled else {}

    def maybe_emit_profiler_progress(base_event: dict[str, Any], *, force: bool = False) -> None:
        nonlocal profiler_event_counter
        if progress_callback is None:
            return
        profiler_event_counter += 1
        if force or profiler_event_counter % profiler_emit_every == 0:
            progress_callback({**base_event, **profiler_snapshot_event()})
        else:
            progress_callback(base_event)

    def get_preprocessed(record_: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        key = str(record_.get("image_id", ""))
        if key in preprocessed_cache:
            preprocessed_cache.move_to_end(key)
            profiler.record("preprocess cache hit", 0.0)
            return preprocessed_cache[key]
        value = _profiled_call(profiler, "read and fixed-preprocess source image", lambda: dataset._read_preprocessed_record_no_square(record_))
        preprocessed_cache[key] = value
        preprocessed_cache.move_to_end(key)
        while len(preprocessed_cache) > max(1, max_preprocessed_cache_items):
            preprocessed_cache.popitem(last=False)
        return value

    def with_source_breast_status(
        record_: dict[str, Any], extra_info_: dict[str, Any]
    ) -> dict[str, Any]:
        image_id_ = str(record_.get("image_id", ""))
        breast_key_ = _breast_key(record_)
        breast_has_mass_ = int(breast_key_ in source_breast_has_mass_lookup)
        return {
            **dict(extra_info_),
            "source_image_has_mass": int(
                source_image_has_mass_lookup.get(image_id_, 0)
            ),
            "source_breast_key": f"{breast_key_[0]}:{breast_key_[1]}",
            "source_breast_has_mass": breast_has_mass_,
        }

    def save_window(
        *,
        record: dict[str, Any],
        split_name: str,
        crop_number: int,
        window: tuple[int, int, int, int],
        extra_info: dict[str, Any],
    ) -> bool:
        nonlocal image_id_counter, ann_id_counter
        extra_info = with_source_breast_status(record, extra_info)
        image, target = get_preprocessed(record)
        source_row = note_source_target(
            split_name, record, target, image, save_review_assets=False
        )
        source_row["attempted_save_windows"] = int(source_row.get("attempted_save_windows", 0)) + 1
        crop_result = _profiled_call(
            profiler,
            "crop current image and boxes",
            lambda: crop_image_and_boxes_to_window(
                image,
                boxes=target["boxes"],
                mass_boxes=target["mass"]["boxes"],
                window_xyxy=window,
                options=common_crop_options,
            ),
        )
        boxes = crop_result.mass_boxes

        if str(extra_info.get("crop_mode", "")).startswith("bbox_safe_random"):
            skip_unsafe = bool(crop_cfg.get("bbox_safe_skip_unsafe_fallbacks", True))
            margin_fraction = float(crop_cfg.get("bbox_safe_boundary_margin_fraction", 0.02))

            if skip_unsafe and not bool(extra_info.get("accepted", True)):
                mark_source_skip(split_name, record, "bbox_safe_failed")
                return False

            profiler.start("bbox-safe final validation")
            source_ok, _source_safe_info = validate_bbox_safe_window(
                window,
                target["mass"]["boxes"],
                crop_size=crop_size,
                margin_fraction=margin_fraction,
                target_box=None,
            )
            exported_ok, _export_safe_info = exported_boxes_satisfy_bbox_safe_margin(
                boxes,
                crop_size=crop_size,
                margin_fraction=margin_fraction,
            )
            profiler.stop("bbox-safe final validation")
            if skip_unsafe and (not source_ok or not exported_ok):
                mark_source_skip(split_name, record, "bbox_safe_failed")
                return False
            extra_info = {**extra_info, **_source_safe_info, **_export_safe_info}

        if str(extra_info.get("deterministic_selection_mode", "")).casefold() == "mass_only" and boxes.shape[0] == 0:
            mark_source_skip(split_name, record, "empty_disallowed")
            return False
        if int(extra_info.get("deterministic_include_empty", 1)) == 0 and boxes.shape[0] == 0:
            mark_source_skip(split_name, record, "empty_disallowed")
            return False

        deduplicate = bool(crop_cfg.get("deduplicate_windows_per_source", True))
        if deduplicate:
            window_key = (split_name, str(record.get("image_id", "")), int(window[0]), int(window[1]), int(window[2]), int(window[3]))
            if window_key in saved_window_keys:
                mark_source_skip(split_name, record, "duplicate_window")
                return False

        planned_positive = int(extra_info.get("is_positive_window", 0)) == 1
        if planned_positive and boxes.shape[0] == 0:
            message = (
                f"Planned positive window {window} for image {record.get('image_id')} "
                "produced no exported Mass box."
            )
            if bool(config.get("replication_contract", {}).get("strict", False)):
                raise RuntimeError(message)
            mark_source_skip(split_name, record, "empty_disallowed")
            return False

        # Optional hard breast-occupancy contract for every saved crop. It is
        # strict for positive and negative crops unless the selected preset
        # explicitly protects eligible positive windows. Recheck immediately
        # before writing so random modes and future planners obey the same rule.
        all_crop_breast_filter_enabled = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "require_min_breast_fraction_for_all_crops",
            False,
        ))
        if all_crop_breast_filter_enabled:
            preserve_positive_below_min_breast_fraction = bool(_split_crop_cfg(
                crop_cfg,
                split_name,
                "preserve_positive_windows_below_min_breast_fraction",
                False,
            ))
            min_all_crop_breast_fraction = float(_split_crop_cfg(
                crop_cfg,
                split_name,
                "min_breast_fraction_for_all_crops",
                0.30,
            ))
            breast_fraction_comparison = str(_split_crop_cfg(
                crop_cfg,
                split_name,
                "breast_fraction_comparison_for_all_crops",
                "greater_than_or_equal",
            ))
            require_retained_mask = bool(_split_crop_cfg(
                crop_cfg,
                split_name,
                "require_retained_breast_mask_for_all_crops",
                False,
            ))
            full_mask = target.get("_foreground_mask")
            if full_mask is None:
                if require_retained_mask:
                    raise RuntimeError(
                        "The all-crop breast-occupancy policy requires the retained "
                        "fixed-preprocessing breast mask, but the preprocessed target "
                        f"for image {record.get('image_id')} has no _foreground_mask. "
                        "Set preprocess.retain_breast_mask_for_export=true."
                    )
                fg_threshold = _split_crop_cfg(
                    crop_cfg,
                    split_name,
                    "foreground_threshold",
                    crop_cfg.get("deterministic_foreground_threshold", None),
                )
                full_mask = _foreground_mask(
                    _tensor_to_float2d(image), threshold=fg_threshold
                )
                mask_source = "derived_from_preprocessed_image"
            else:
                mask_source = "retained_preprocessing_mask"
            full_mask = np.asarray(full_mask, dtype=bool)
            expected_mask_shape = (int(image.shape[-2]), int(image.shape[-1]))
            if tuple(full_mask.shape[:2]) != expected_mask_shape:
                raise ValueError(
                    "Breast foreground mask shape does not match the fixed-preprocessed "
                    f"image for {record.get('image_id')}: mask={tuple(full_mask.shape)}, "
                    f"image={expected_mask_shape}."
                )
            all_crop_breast_fraction = float(_foreground_fraction_from_mask(
                full_mask,
                window,
                crop_size,
            ))
            extra_info = {
                **extra_info,
                "foreground_filter_enabled": 1,
                "all_crop_breast_fraction_filter_enabled": 1,
                "preserve_positive_windows_below_min_breast_fraction": int(
                    preserve_positive_below_min_breast_fraction
                ),
                "require_retained_breast_mask_for_all_crops": int(require_retained_mask),
                "breast_fraction_mask_source": mask_source,
                "min_foreground_fraction": min_all_crop_breast_fraction,
                "min_breast_fraction_for_all_crops": min_all_crop_breast_fraction,
                "breast_fraction_comparison_for_all_crops": breast_fraction_comparison,
                "foreground_fraction": all_crop_breast_fraction,
            }
            breast_fraction_passes = _breast_fraction_passes(
                all_crop_breast_fraction,
                min_all_crop_breast_fraction,
                breast_fraction_comparison,
            )
            positive_window_is_protected = (
                preserve_positive_below_min_breast_fraction
                and (planned_positive or boxes.shape[0] > 0)
            )
            if not breast_fraction_passes and not positive_window_is_protected:
                mark_source_skip(split_name, record, "foreground_too_low")
                return False

        # Optional hard foreground filter for empty crops. This prevents low-level
        # background noise from creating apparently valid empty/background crops.
        # Reuse the fraction planned from the complete source mask; never estimate
        # foreground independently from the cropped patch.
        if boxes.shape[0] == 0 and not planned_positive and bool(_split_crop_cfg(crop_cfg, split_name, "negative_require_foreground", crop_cfg.get("require_foreground_for_empty_crops", True))):
            min_empty_fg = float(_split_crop_cfg(crop_cfg, split_name, "negative_min_foreground_fraction", crop_cfg.get("min_foreground_fraction", 0.35)))
            fg_threshold = _split_crop_cfg(crop_cfg, split_name, "foreground_threshold", crop_cfg.get("deterministic_foreground_threshold", None))
            fg_fraction = extra_info.get("foreground_fraction")
            if fg_fraction is None:
                full_mask = target.get("_foreground_mask")
                if full_mask is None:
                    full_mask = _foreground_mask(_tensor_to_float2d(image), threshold=fg_threshold)
                fg_fraction = _foreground_fraction_from_mask(
                    np.asarray(full_mask, dtype=bool), window, crop_size
                )
            fg_fraction = float(fg_fraction)
            extra_info = {**extra_info, "negative_foreground_filter_enabled": 1, "negative_min_foreground_fraction": float(min_empty_fg), "negative_foreground_fraction": float(fg_fraction)}
            if fg_fraction < min_empty_fg:
                mark_source_skip(split_name, record, "foreground_too_low")
                return False

        filename = _make_crop_filename(record, split_name, crop_number, window)
        rel_img_path = Path("images") / split_name / filename
        source_arrays = None
        full_source_arrays: dict[str, np.ndarray] = {
            "current_crop": _tensor_to_float2d(image),
        }
        retained_foreground_mask = target.get("_foreground_mask")
        full_source_masks: dict[str, np.ndarray] = {}
        if (
            retained_foreground_mask is None
            and bool(config.get("preprocess", {}).get("retain_breast_mask_for_export", False))
        ):
            raise RuntimeError(
                "Fixed preprocessing was configured to retain its breast mask, "
                f"but image {record.get('image_id')} has no _foreground_mask."
            )
        if retained_foreground_mask is not None:
            retained_foreground_mask = np.asarray(retained_foreground_mask, dtype=bool)
            expected_shape = (int(image.shape[-2]), int(image.shape[-1]))
            if tuple(retained_foreground_mask.shape) != expected_shape:
                raise ValueError(
                    "Retained breast mask shape does not match the fixed-preprocessed "
                    f"image for {record.get('image_id')}: "
                    f"mask={tuple(retained_foreground_mask.shape)}, image={expected_shape}."
                )
            full_source_masks["current_crop"] = retained_foreground_mask
        source_windows: dict[str, tuple[int, int, int, int]] = {
            "current_crop": window,
        }
        if needs_contralateral:
            source_arrays, alignment_info = _profiled_call(
                profiler,
                "contralateral source crop",
                lambda: _contralateral_source_arrays_for_window(
                    record=record,
                    reference_image=image,
                    window=window,
                    crop_options=common_crop_options,
                    contralateral_lookup=contralateral_lookup,
                    get_preprocessed=get_preprocessed,
                    config=config,
                    alignment_cache=aligned_contralateral_cache,
                    max_alignment_cache_items=max_aligned_contralateral_cache_items,
                    profiler=profiler,
                ),
            )
            if alignment_info:
                extra_info = {**extra_info, **alignment_info}
            if source_arrays and source_arrays.get("contralateral_same_view_full") is not None:
                full_source_arrays["contralateral_same_view_crop"] = source_arrays.pop(
                    "contralateral_same_view_full"
                )
                adjusted = alignment_info.get("contralateral_fast_adjusted_window_xyxy") if alignment_info else None
                if adjusted is not None:
                    source_windows["contralateral_same_view_crop"] = tuple(int(v) for v in adjusted)
        save_info = _profiled_call(
            profiler,
            "save RGB and preserved PNG",
            lambda: _save_export_images(
                crop_result.image,
                crop_root,
                rel_img_path,
                config,
                source_arrays=source_arrays,
                full_source_arrays=full_source_arrays,
                full_source_masks=full_source_masks,
                source_windows=source_windows,
                crop_window=window,
                crop_pad_value=float(common_crop_options.get("pad_value", 0.0)),
                whole_stage_cache=whole_stage_cache,
                cache_namespace=f"{split_name}:{record.get('image_id', '')}",
                float32_variant="crops",
                reject_blank_output=(
                    boxes.shape[0] == 0
                    and not planned_positive
                    and bool(_split_crop_cfg(
                        crop_cfg,
                        split_name,
                        "negative_reject_blank_output",
                        False,
                    ))
                ),
                min_output_signal_fraction=float(_split_crop_cfg(
                    crop_cfg,
                    split_name,
                    "negative_min_output_signal_fraction",
                    0.01,
                )),
            ),
        )
        if bool(save_info.get("output_rejected_blank", False)):
            mark_source_skip(split_name, record, "blank_output")
            return False
        if bool(paired_whole_cfg.get("enabled", False)):
            paired_info = _profiled_call(
                profiler,
                "save paired whole image",
                lambda: _save_paired_whole_image_for_crop(
                    source_image=image,
                    crop_root=crop_root,
                    split_name=split_name,
                    filename=filename,
                    source_image_id=str(record.get("image_id", "")),
                    source_study_id=str(record.get("study_id", "")),
                    source_boxes=target.get("mass", {}).get("boxes"),
                    source_annotation_ids=[
                        finding.get("source_annotation_id")
                        for finding in list(
                            target.get("mass", {}).get("findings", []) or []
                        )
                    ],
                    source_annotation_rows=[
                        finding.get("source_annotation_row")
                        for finding in list(
                            target.get("mass", {}).get("findings", []) or []
                        )
                    ],
                    source_foreground_mask=retained_foreground_mask,
                    config=config,
                    paired_cfg=paired_whole_cfg,
                    source_path_cache=paired_whole_source_paths,
                    whole_stage_cache=whole_stage_cache,
                    cache_namespace=f"{split_name}:{record.get('image_id', '')}",
                ),
            )
            save_info.update(paired_info)
        while len(whole_stage_cache) > max(1, max_whole_stage_cache_items):
            whole_stage_cache.popitem(last=False)

        labels_path = crop_root / "labels" / split_name / f"{Path(filename).stem}.txt"
        profiler.start("write labels and metadata rows")
        _write_yolo_label_file(labels_path, boxes, width=crop_size, height=crop_size, save_empty=save_empty_labels)

        crop_info = {
            "window_xyxy": window,
            "source_index": int(source_row.get("source_index", -1)),
            "source_saved_crops_so_far": int(source_row.get("saved_crops", 0)),
            **extra_info,
            **crop_result.info,
        }
        image_meta = _coco_image_record(
            image_id_counter,
            filename,
            crop_size,
            crop_size,
            record,
            save_info,
            split_name,
            dataset_name="square_crops",
            crop_info=crop_info,
            preprocess_info=target.get("preprocessing", {}),
        )
        coco = coco_by_split[split_name]
        coco["images"].append(image_meta)
        keep_flags = crop_result.mass_box_keep.tolist()
        source_findings = list(target.get("mass", {}).get("findings", []) or [])
        kept_findings = [
            finding for finding, keep in zip(source_findings, keep_flags) if keep
        ]
        source_boxes = target["mass"]["boxes"][crop_result.mass_box_keep]
        source_original_boxes = _fixed_preprocessed_boxes_to_original(
            source_boxes,
            target.get("preprocessing", {}),
        )
        source_annotation_ids = [finding.get("source_annotation_id") for finding in kept_findings]
        source_annotation_rows = [finding.get("source_annotation_row") for finding in kept_findings]
        ann_rows = _append_coco_annotations(
            coco,
            image_id_counter,
            ann_id_counter,
            boxes,
            source_annotation_ids=source_annotation_ids,
            source_annotation_rows=source_annotation_rows,
            source_boxes=source_boxes,
            source_original_boxes=source_original_boxes,
        )
        ann_id_counter += ann_rows
        stats_rows.append(
            _sample_stats_row(
                dataset_name="square_crops",
                split=split_name,
                filename=filename,
                image_width=crop_size,
                image_height=crop_size,
                boxes=boxes,
                record=record,
                crop_info=crop_info,
                save_info=save_info,
            )
        )
        metadata_rows.append(
            _sample_metadata_record(
                dataset_name="square_crops",
                split=split_name,
                filename=filename,
                target=target,
                record=record,
                boxes=boxes,
                save_info=save_info,
                crop_info=crop_info,
            )
        )
        source_row["saved_crops"] = int(source_row.get("saved_crops", 0)) + 1
        if boxes.shape[0] > 0:
            source_row["saved_positive_crops"] = int(source_row.get("saved_positive_crops", 0)) + 1
        else:
            source_row["saved_negative_crops"] = int(source_row.get("saved_negative_crops", 0)) + 1
        source_row["exported_mass_box_instances"] = int(source_row.get("exported_mass_box_instances", 0)) + int(boxes.shape[0])
        for source_annotation_id in source_annotation_ids:
            if source_annotation_id is not None:
                source_row.setdefault("_included_annotation_indices", set()).add(source_annotation_id)
        window_key = (
            split_name,
            str(record.get("image_id", "")),
            int(window[0]), int(window[1]), int(window[2]), int(window[3]),
        )
        if boxes.shape[0] > 0:
            saved_positive_window_keys.add(window_key)
        if deduplicate:
            saved_window_keys.add(window_key)
        image_id_counter += 1
        profiler.stop("write labels and metadata rows")
        # A source earns review/debug assets only after at least one crop was
        # successfully written. This prevents rejected candidates from using
        # the bounded debug quota or triggering a second DICOM decode.
        _profiled_call(
            profiler,
            "save bounded debug source assets",
            lambda: note_source_target(
                split_name, record, target, image, save_review_assets=True
            ),
        )
        review_key = (split_name, str(record.get("image_id", "")))
        if review_key in review_source_rows:
            review_source_rows[review_key].update({
                key: value
                for key, value in save_info.items()
                if key.startswith("paired_whole_")
            })
        return True

    for split_name in ["train", "val", "test"]:
        records = split_records.get(split_name, [])
        split_mode = str(crop_cfg.get(f"{split_name}_crop_mode", "random" if split_name == "train" else "deterministic")).casefold().strip()
        selection_mode = _deterministic_selection_mode(crop_cfg, split_name) if split_mode == "deterministic" else ""

        if _online_positive_ratio_selection_enabled(crop_cfg, split_name, split_mode):
            # Single-pass approximate balancing. This avoids the slow planning pass used by
            # exact global selection and starts writing crops immediately. Empty candidates
            # may come from any source image, including no-mass images.
            target_ratio = _target_positive_fraction(crop_cfg, split_name, bbox_safe=(split_mode == "bbox_safe_random"))
            shuffle_online_records = bool(crop_cfg.get(
                f"{split_name}_online_balance_shuffle_source_records",
                crop_cfg.get("online_balance_shuffle_source_records", True),
            ))

            positive_saved = 0
            negative_saved = 0
            attempted_crops = 0
            saved_crops = 0
            selection_mode = _deterministic_selection_mode(crop_cfg, split_name)
            negative_images_only = selection_mode == "crop_label_ratio"
            negative_breasts_only = selection_mode == "crop_label_ratio"
            reserve_negative_records: list[dict[str, Any]] = []
            if selection_mode == "crop_label_ratio":
                (
                    online_records,
                    reserve_negative_records,
                    online_schedule,
                ) = _crop_label_streaming_record_schedule(
                    records=records,
                    source_image_has_mass_lookup=source_image_has_mass_lookup,
                    source_breast_has_mass_lookup=source_breast_has_mass_lookup,
                    target_positive_ratio=target_ratio,
                    rng=rng,
                    shuffle=shuffle_online_records,
                )
                total_records_for_progress -= max(0, len(records) - len(online_records))
            else:
                online_records = list(records)
                if shuffle_online_records:
                    order = rng.permutation(len(online_records)) if online_records else []
                    online_records = [online_records[int(i)] for i in order]
                online_schedule = {
                    "positive_source_images": int(sum(
                        int(source_image_has_mass_lookup.get(str(record.get("image_id", "")), 0))
                        for record in online_records
                    )),
                    "scheduled_negative_source_images": int(len(online_records)),
                    "skipped_ineligible_source_images": 0,
                }

            online_schedule["negative_top_up_source_images_processed"] = 0

            def online_record_stream():
                nonlocal total_records_for_progress
                for scheduled_record in online_records:
                    yield scheduled_record, False
                # The compact cadence is the fast primary pass. If window
                # rejection or unequal per-source crop yields leave a deficit,
                # consume seeded reserve negative breasts until the requested
                # crop ratio is reached or no eligible source remains.
                for reserve_record in reserve_negative_records:
                    if not _online_should_save_negative(
                        positive_saved, negative_saved, target_ratio
                    ):
                        break
                    online_schedule["negative_top_up_source_images_processed"] += 1
                    total_records_for_progress += 1
                    yield reserve_record, True

            iterator = _progress(
                online_record_stream(),
                show_progress,
                f"Export square crops {split_name} streaming balance",
                unit="img",
                total=len(online_records),
            )
            for record, is_negative_top_up_source in iterator:
                image, target = get_preprocessed(record)
                source_row_for(split_name, record)["processed_source_image"] = 1
                # Debug assets are deliberately deferred until save_window accepts
                # at least one crop from this source. Otherwise a sparse-positive
                # split spends hours decoding the DICOM twice and writing debug
                # PNGs for sources that contribute no data.
                note_source_target(
                    split_name,
                    record,
                    target,
                    image,
                    save_review_assets=False,
                )
                height, width = int(image.shape[-2]), int(image.shape[-1])
                windows = _profiled_call(
                    profiler,
                    "generate current-image crop windows",
                    lambda: _windows_for_export_split(
                        split_name=split_name,
                        image_width=width,
                        image_height=height,
                        image_tensor=image,
                        mass_boxes=target["mass"]["boxes"],
                        crop_options=common_crop_options,
                        crop_cfg=crop_cfg,
                        rng=rng,
                        foreground_mask=target.get("_foreground_mask"),
                        diagnostics=source_row_for(split_name, record),
                    ),
                )
                note_candidate_windows(split_name, record, windows)
                if bool(_split_crop_cfg(
                    crop_cfg,
                    split_name,
                    "online_balance_shuffle_windows",
                    True,
                )) and len(windows) > 1:
                    window_order = rng.permutation(len(windows))
                    windows = [windows[int(i)] for i in window_order]
                for window, extra_info in windows:
                    extra_info = with_source_breast_status(record, extra_info)
                    is_positive_window = int(extra_info.get("is_positive_window", 0)) == 1
                    if (
                        not is_positive_window
                        and (
                            (
                                negative_images_only
                                and int(extra_info.get("source_image_has_mass", 0)) == 1
                            )
                            or (
                                negative_breasts_only
                                and int(extra_info.get("source_breast_has_mass", 0)) == 1
                            )
                        )
                    ):
                        # Crop-label balancing for Custom Paper 22 draws empty
                        # crops only from breasts with no Mass annotation in
                        # either source view.
                        # Positive windows are never subject to this filter.
                        continue
                    if not is_positive_window and not _online_should_save_negative(positive_saved, negative_saved, target_ratio):
                        continue

                    attempted_crops += 1
                    proposed_positive = positive_saved + (1 if is_positive_window else 0)
                    proposed_negative = negative_saved + (0 if is_positive_window else 1)
                    enriched_info = _online_balance_extra_info(
                        extra_info,
                        split_name=split_name,
                        target_ratio=target_ratio,
                        positive_count=proposed_positive,
                        negative_count=proposed_negative,
                        split_mode=split_mode,
                    )
                    enriched_info["deterministic_selection_mode"] = selection_mode
                    enriched_info["source_schedule_positive_cadence"] = int(
                        online_schedule.get("positive_source_cadence", 0)
                    )
                    enriched_info["source_schedule_negative_cadence"] = int(
                        online_schedule.get("negative_source_cadence", 0)
                    )
                    enriched_info["negative_top_up_source"] = int(
                        is_negative_top_up_source
                    )
                    enriched_info["negative_crop_source_policy"] = (
                        "mass_negative_breasts_only"
                        if negative_breasts_only
                        else "mass_negative_images_only"
                        if negative_images_only
                        else "any_source_image"
                    )
                    saved_ok = save_window(
                        record=record,
                        split_name=split_name,
                        crop_number=attempted_crops - 1,
                        window=window,
                        extra_info=enriched_info,
                    )
                    if saved_ok:
                        saved_crops += 1
                        if is_positive_window:
                            positive_saved += 1
                        else:
                            negative_saved += 1

                processed_records_for_progress += 1
                maybe_emit_profiler_progress({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": f"{split_name} online saving",
                    "processed": int(processed_records_for_progress),
                    "total": int(total_records_for_progress),
                    "unit": "source images",
                    "saved_crops": int(saved_crops),
                    "saved_positive_crops": int(positive_saved),
                    "saved_negative_crops": int(negative_saved),
                    "running_positive_ratio": float(positive_saved / max(1, positive_saved + negative_saved)),
                    "target_positive_ratio": float(target_ratio),
                    "scheduled_source_images": int(
                        len(online_records)
                        + online_schedule.get("negative_top_up_source_images_processed", 0)
                    ),
                    **online_schedule,
                })
            target_met = not _online_should_save_negative(
                positive_saved, negative_saved, target_ratio
            )
            desired_negative_crops = (
                0
                if target_ratio >= 1.0
                else int(round(
                    float(positive_saved)
                    * (1.0 - float(target_ratio))
                    / float(target_ratio)
                ))
            )
            online_schedule["negative_top_up_target_met"] = int(target_met)
            online_schedule["negative_top_up_reserve_exhausted"] = int(
                not target_met
                and online_schedule["negative_top_up_source_images_processed"]
                >= len(reserve_negative_records)
            )
            online_balance_summaries[split_name] = {
                **online_schedule,
                "target_positive_ratio": float(target_ratio),
                "saved_positive_crops": int(positive_saved),
                "saved_negative_crops": int(negative_saved),
                "desired_negative_crops": int(desired_negative_crops),
                "negative_crop_deficit_after_top_up": int(max(
                    0, desired_negative_crops - negative_saved
                )),
                "achieved_positive_ratio": float(
                    positive_saved / max(1, positive_saved + negative_saved)
                ),
            }
            continue

        if split_mode in {"random", "bbox_safe_random"} and _global_positive_ratio_selection_enabled(crop_cfg, split_name, split_mode):
            candidates: list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]] = []
            iterator = _progress(records, show_progress, f"Plan square crops {split_name}", unit="img")
            for record in iterator:
                image, target = get_preprocessed(record)
                source_row_for(split_name, record)["processed_source_image"] = 1
                note_source_target(
                    split_name, record, target, image, save_review_assets=False
                )
                height, width = int(image.shape[-2]), int(image.shape[-1])
                windows = _profiled_call(
                    profiler,
                    "plan crop windows",
                    lambda: _windows_for_export_split(
                        split_name=split_name,
                        image_width=width,
                        image_height=height,
                        image_tensor=image,
                        mass_boxes=target["mass"]["boxes"],
                        crop_options=common_crop_options,
                        crop_cfg=crop_cfg,
                        rng=rng,
                        foreground_mask=target.get("_foreground_mask"),
                        diagnostics=source_row_for(split_name, record),
                    ),
                )
                note_candidate_windows(split_name, record, windows)
                candidates.extend(
                    (record, window, with_source_breast_status(record, extra_info))
                    for window, extra_info in windows
                )
                processed_records_for_progress += 1
                maybe_emit_profiler_progress({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": f"{split_name} planning",
                    "processed": int(processed_records_for_progress),
                    "total": int(total_records_for_progress),
                    "unit": "source images",
                })

            target_ratio = _target_positive_fraction(crop_cfg, split_name, bbox_safe=(split_mode == "bbox_safe_random"))
            selected = _select_positive_ratio_candidates(
                candidates,
                crop_cfg,
                split_name,
                rng,
                target_ratio=target_ratio,
                selection_label=f"global_{split_mode}_positive_ratio",
            )
            save_iter = _progress(selected, show_progress, f"Save square crops {split_name}", unit="crop")
            total_selected = len(selected)
            saved_count = 0
            for crop_number, (record, window, extra_info) in enumerate(save_iter):
                if save_window(record=record, split_name=split_name, crop_number=crop_number, window=window, extra_info=extra_info):
                    saved_count += 1
                maybe_emit_profiler_progress({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": f"{split_name} saving",
                    "processed": int(crop_number + 1),
                    "total": int(total_selected),
                    "unit": "crops",
                })
            continue

        if split_mode == "deterministic" and selection_mode in {
            "positive_ratio", "crop_label_ratio", "negative_fraction", "source_breast_ratio"
        }:
            candidates: list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]] = []
            iterator = _progress(records, show_progress, f"Plan square crops {split_name}", unit="img")
            for record in iterator:
                image, target = get_preprocessed(record)
                source_row_for(split_name, record)["processed_source_image"] = 1
                note_source_target(
                    split_name, record, target, image, save_review_assets=False
                )
                height, width = int(image.shape[-2]), int(image.shape[-1])
                windows = _profiled_call(
                    profiler,
                    "plan crop windows",
                    lambda: _windows_for_export_split(
                        split_name=split_name,
                        image_width=width,
                        image_height=height,
                        image_tensor=image,
                        mass_boxes=target["mass"]["boxes"],
                        crop_options=common_crop_options,
                        crop_cfg=crop_cfg,
                        rng=rng,
                        foreground_mask=target.get("_foreground_mask"),
                        diagnostics=source_row_for(split_name, record),
                    ),
                )
                note_candidate_windows(split_name, record, windows)
                candidates.extend(
                    (record, window, with_source_breast_status(record, extra_info))
                    for window, extra_info in windows
                )
                processed_records_for_progress += 1
                maybe_emit_profiler_progress({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": f"{split_name} planning",
                    "processed": int(processed_records_for_progress),
                    "total": int(total_records_for_progress),
                    "unit": "source images",
                })

            if selection_mode == "negative_fraction":
                selected = _select_negative_fraction_candidates(candidates, crop_cfg, split_name, rng)
            elif selection_mode == "source_breast_ratio":
                selected = _select_source_breast_ratio_candidates(
                    candidates, crop_cfg, split_name, rng
                )
            else:
                selected = _select_positive_ratio_candidates(
                    candidates,
                    crop_cfg,
                    split_name,
                    rng,
                    negative_images_only=(selection_mode == "crop_label_ratio"),
                    negative_breasts_only=(selection_mode == "crop_label_ratio"),
                    selection_label=(
                        "crop_label_ratio_negative_breasts_only"
                        if selection_mode == "crop_label_ratio"
                        else "positive_ratio"
                    ),
                )
            save_iter = _progress(selected, show_progress, f"Save square crops {split_name}", unit="crop")
            total_selected = len(selected)
            saved_count = 0
            for crop_number, (record, window, extra_info) in enumerate(save_iter):
                if save_window(record=record, split_name=split_name, crop_number=crop_number, window=window, extra_info=extra_info):
                    saved_count += 1
                maybe_emit_profiler_progress({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": f"{split_name} saving",
                    "processed": int(crop_number + 1),
                    "total": int(total_selected),
                    "unit": "crops",
                })
            continue

        iterator = _progress(records, show_progress, f"Export square crops {split_name}", unit="img")
        for record in iterator:
            image, target = get_preprocessed(record)
            source_row_for(split_name, record)["processed_source_image"] = 1
            note_source_target(
                split_name, record, target, image, save_review_assets=False
            )
            height, width = int(image.shape[-2]), int(image.shape[-1])
            windows = _profiled_call(
                profiler,
                "generate current-image crop windows",
                lambda: _windows_for_export_split(
                    split_name=split_name,
                    image_width=width,
                    image_height=height,
                    image_tensor=image,
                    mass_boxes=target["mass"]["boxes"],
                    crop_options=common_crop_options,
                    crop_cfg=crop_cfg,
                    rng=rng,
                    foreground_mask=target.get("_foreground_mask"),
                    diagnostics=source_row_for(split_name, record),
                ),
            )
            note_candidate_windows(split_name, record, windows)
            for crop_number, (window, extra_info) in enumerate(windows):
                save_window(record=record, split_name=split_name, crop_number=crop_number, window=window, extra_info=extra_info)
            processed_records_for_progress += 1
            maybe_emit_profiler_progress({
                "event": "image_progress",
                "stage": "export_square_crops",
                "split": split_name,
                "processed": int(processed_records_for_progress),
                "total": int(total_records_for_progress),
                "unit": "source images",
            })

    created = _profiled_call(
        profiler,
        "write COCO CSV and summary files",
        lambda: _write_shared_export_files(crop_root, coco_by_split, stats_rows, metadata_rows, dataset_kind="square_crops"),
    )
    created.extend(_write_square_crop_debug_logs(crop_root, stats_rows, source_debug))
    annotation_report_summary: dict[str, Any] = {"enabled": False}
    if annotation_report_enabled:
        from .visualize import create_annotation_geometry_report

        report_subdir = str(
            annotation_report_cfg.get("output_subdir", "annotation_geometry")
            or "annotation_geometry"
        ).strip()
        report_dir = output_root / "visualizations" / report_subdir
        annotation_result = create_annotation_geometry_report(
            annotation_geometry_rows,
            output_dir=report_dir,
            crop_width=crop_size,
            crop_height=crop_size,
            histogram_bins=int(annotation_report_cfg.get("histogram_bins", 40) or 40),
        )
        created.extend(annotation_result.created_files)
        annotation_report_summary = {
            "enabled": True,
            "output_dir": _path_as_posix(annotation_result.output_dir),
            **annotation_result.summary,
        }
    review_summary: dict[str, Any] = {"enabled": False}
    if review_enabled:
        review_summary, review_files = _write_dataset_review_bundle(
            crop_root=crop_root,
            stats_rows=stats_rows,
            coco_by_split=coco_by_split,
            source_rows=list(review_source_rows.values()),
            review_cfg=review_cfg,
        )
        created.extend(review_files)
    summary = _summary_from_stats(stats_rows)
    summary["online_balance"] = online_balance_summaries
    summary["dataset_review"] = review_summary
    summary["annotation_geometry_report"] = annotation_report_summary
    summary["debug_logs"] = {
        "folder": _path_as_posix(crop_root / "debug_logs"),
        "crop_log_csv": _path_as_posix(crop_root / "debug_logs" / "crop_log.csv"),
        "source_image_log_csv": _path_as_posix(crop_root / "debug_logs" / "source_image_log.csv"),
        "crops_per_source_histogram_csv": _path_as_posix(crop_root / "debug_logs" / "crops_per_source_histogram.csv"),
        "split_mass_coverage_csv": _path_as_posix(crop_root / "debug_logs" / "split_mass_coverage.csv"),
    }
    summary["replication_contract"] = _validate_square_crop_contract(
        split_records=split_records,
        coco_by_split=coco_by_split,
        source_debug=source_debug,
        candidate_positive_window_keys=candidate_positive_window_keys,
        saved_positive_window_keys=saved_positive_window_keys,
        crop_cfg=crop_cfg,
        config=config,
    )
    summary["simple_profiler"] = profiler.snapshot()
    if progress_callback is not None:
        progress_callback({"event": "profiler_update", "stage": "export_square_crops", **profiler_snapshot_event()})
    return summary, created


def export_whole_image_variants_only(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    output_root: Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Export selected whole variants when crop image export is unchecked."""
    crop_root = dataset_content_root(output_root, config)
    crop_root.mkdir(parents=True, exist_ok=True)
    paired_cfg = dict(config.get("paired_whole_images", {}) or {})
    review_cfg = dict(config.get("dataset_review", {}) or {})
    review_enabled = bool(review_cfg.get("enabled", False))
    review_limit_raw = review_cfg.get(
        "source_assets_per_split", review_cfg.get("samples_per_split", 100)
    )
    review_limit = None if review_limit_raw is None else max(0, int(review_limit_raw))
    show_progress = bool((config.get("runtime", {}) or {}).get("show_progress", True))
    source_path_cache: dict[tuple[str, ...], Path] = {}
    whole_stage_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
    metadata_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    split_counts = {split: 0 for split in ["train", "val", "test"]}
    total = sum(len(split_records.get(split, [])) for split in ["train", "val", "test"])
    processed = 0
    for split_name in ["train", "val", "test"]:
        records = split_records.get(split_name, [])
        iterator = _progress(
            records,
            show_progress,
            f"Export whole variants {split_name}",
            unit="img",
        )
        for record in iterator:
            image, target = dataset._read_preprocessed_record_no_square(record)
            source_image_id = str(record.get("image_id", ""))
            source_study_id = str(record.get("study_id", ""))
            findings = list(target.get("mass", {}).get("findings", []) or [])
            synthetic_crop_name = (
                f"{_safe_name(source_study_id)}__{_safe_name(source_image_id)}"
                "__crop__whole_only.png"
            )
            info = _save_paired_whole_image_for_crop(
                source_image=image,
                crop_root=crop_root,
                split_name=split_name,
                filename=synthetic_crop_name,
                source_image_id=source_image_id,
                source_study_id=source_study_id,
                source_boxes=target.get("mass", {}).get("boxes"),
                source_annotation_ids=[
                    finding.get("source_annotation_id") for finding in findings
                ],
                source_annotation_rows=[
                    finding.get("source_annotation_row") for finding in findings
                ],
                source_foreground_mask=target.get("_foreground_mask"),
                config=config,
                paired_cfg=paired_cfg,
                source_path_cache=source_path_cache,
                whole_stage_cache=whole_stage_cache,
                cache_namespace=f"{split_name}:{source_image_id}",
            )
            try:
                breast_key = _breast_key(record)
                source_breast_key = f"{breast_key[0]}:{breast_key[1]}"
            except ValueError:
                source_breast_key = str(source_study_id)
            preprocessing = dict(target.get("preprocessing", {}) or {})
            metadata_rows.append({
                "dataset": "whole_image_variants",
                "split": split_name,
                "file_name": info.get("paired_whole_filename", ""),
                "source_image_id": source_image_id,
                "source_study_id": source_study_id,
                "source_breast_key": source_breast_key,
                "source_breast_has_mass": bool(
                    record.get("_source_breast_has_mass", bool(findings))
                ),
                "source_preprocessing_mirrored": bool(
                    preprocessing.get("mirrored", False)
                ),
                "source_coordinate_space": "fixed_preprocessed",
                "paired_whole_original_image": info.get(
                    "paired_whole_original_image_path", ""
                ),
                "paired_whole_original_float32_image": info.get(
                    "paired_whole_original_float32_image_path", ""
                ),
                "paired_whole_image": info.get("paired_whole_image_path", ""),
                "paired_whole_float32_image": info.get(
                    "paired_whole_float32_image_path", ""
                ),
                "encoding": info,
            })
            if review_enabled and (
                review_limit is None or split_counts[split_name] < review_limit
            ):
                review_row = _save_review_source_assets(
                    crop_root=crop_root,
                    split_name=split_name,
                    record=record,
                    image=image,
                    target=target,
                    review_cfg=review_cfg,
                )
                review_row.update(info)
                review_rows.append(review_row)
            split_counts[split_name] += 1
            processed += 1
            if progress_callback is not None:
                progress_callback({
                    "event": "image_progress",
                    "stage": "export_whole_image_variants",
                    "split": split_name,
                    "processed": processed,
                    "total": total,
                    "unit": "source images",
                })
            while len(whole_stage_cache) > 12:
                whole_stage_cache.popitem(last=False)

    created = _write_whole_image_annotation_indexes(
        crop_root, metadata_rows, config=config
    )
    metadata_dir = crop_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = metadata_dir / "whole_image_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    created.append(jsonl_path)
    flat_path = metadata_dir / "samples_metadata_flat.csv"
    pd.DataFrame([_flatten_metadata_row(row) for row in metadata_rows]).to_csv(
        flat_path, index=False
    )
    created.append(flat_path)
    validation_summary, validation_path = _validate_whole_image_export_contract(
        crop_root=crop_root,
        split_records=split_records,
        config=config,
    )
    created.append(validation_path)
    review_summary: dict[str, Any] = {"enabled": False}
    if review_enabled:
        review_summary, review_files = _write_dataset_review_bundle(
            crop_root=crop_root,
            stats_rows=[],
            coco_by_split={split: _empty_coco() for split in ["train", "val", "test"]},
            source_rows=review_rows,
            review_cfg=review_cfg,
        )
        created.extend(review_files)
    return {
        "num_source_images": len(metadata_rows),
        "splits": split_counts,
        "save_original": _paired_original_enabled(paired_cfg),
        "save_resized": _paired_resized_enabled(paired_cfg),
        "resized_variants": [
            {
                "name": item["name"],
                "width": item["width"],
                "height": item["height"],
                "save_float32": item["save_float32"],
            }
            for item in resized_variant_configs(paired_cfg)
        ],
        "save_high_resolution": _paired_high_resolution_enabled(paired_cfg),
        "validation": validation_summary,
        "dataset_review": review_summary,
    }, list(dict.fromkeys(created))


def _validate_whole_image_export_contract(
    *,
    crop_root: Path,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Validate complete per-source whole inputs before declaring export success."""

    contract = dict(config.get("whole_image_export_contract", {}) or {})
    enabled = bool(contract.get("enabled", False))
    strict = bool(contract.get("strict", True))
    metadata_dir = crop_root / "metadata"
    report_path = metadata_dir / "whole_image_validation.json"
    if not enabled:
        report = {"enabled": False, "status": "not_requested", "errors": []}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report, report_path

    manifest_path = metadata_dir / "whole_image_manifest.csv"
    annotations_path = (
        crop_root / "annotations" / "whole_image_annotations.csv"
        if uses_grouped_dataset_layout(config or {})
        else metadata_dir / "whole_image_annotations.csv"
    )
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    annotations = pd.read_csv(annotations_path, low_memory=False)
    expected_variants = [
        str(value) for value in contract.get("expected_variants", ["original", "resized"])
    ]
    float_required = {
        str(value) for value in contract.get("float32_required_variants", ["resized"])
    }
    expected_images = dict(contract.get("expected_source_images", {}) or {})
    expected_positive = dict(contract.get("expected_positive_sources", {}) or {})
    expected_annotations = dict(contract.get("expected_mass_annotations", {}) or {})
    errors: list[str] = []
    observed: dict[str, Any] = {}

    duplicate_mask = manifest.duplicated(
        subset=["variant", "split", "source_image_id"], keep=False
    )
    if bool(duplicate_mask.any()):
        errors.append(
            f"whole-image manifest has {int(duplicate_mask.sum())} duplicate variant/split/source rows"
        )
    source_splits = manifest.groupby("source_image_id")["split"].nunique()
    multiple_splits = int((source_splits > 1).sum())
    if multiple_splits:
        errors.append(f"{multiple_splits} source image IDs appear in multiple splits")

    assigned_by_split = {
        split: {str(record.get("image_id", "")) for record in split_records.get(split, [])}
        for split in ["train", "val", "test"]
    }

    def parse_shape(value: Any) -> list[int]:
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        try:
            parsed = json.loads(str(value))
            return [int(item) for item in parsed]
        except Exception:
            return []

    def parse_box(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        try:
            parsed = json.loads(str(value))
            return [float(item) for item in parsed]
        except Exception:
            return []

    for variant in expected_variants:
        observed[variant] = {}
        variant_manifest = manifest[manifest["variant"].astype(str) == variant]
        variant_annotations = annotations[
            annotations["variant"].astype(str) == variant
        ]
        for split in ["train", "val", "test"]:
            rows = variant_manifest[
                variant_manifest["split"].astype(str) == split
            ].copy()
            ann_rows = variant_annotations[
                variant_annotations["split"].astype(str) == split
            ].copy()
            row_ids = set(rows["source_image_id"].astype(str))
            assigned_ids = assigned_by_split[split]
            positive_count = int((pd.to_numeric(rows["num_annotations"]) > 0).sum())
            annotation_count = int(pd.to_numeric(rows["num_annotations"]).sum())
            expected_count = int(expected_images.get(split, len(assigned_ids)))
            observed[variant][split] = {
                "images": int(len(rows)),
                "positive_sources": positive_count,
                "mass_annotations": annotation_count,
                "coco_images": 0,
                "coco_annotations": 0,
                "float32_tensors": 0,
            }
            if len(rows) != expected_count:
                errors.append(
                    f"{variant}/{split}: expected {expected_count} manifest rows, got {len(rows)}"
                )
            if row_ids != assigned_ids:
                errors.append(
                    f"{variant}/{split}: manifest membership differs from split assignment "
                    f"(missing={len(assigned_ids - row_ids)}, extra={len(row_ids - assigned_ids)})"
                )
            if split in expected_positive and positive_count != int(expected_positive[split]):
                errors.append(
                    f"{variant}/{split}: expected {int(expected_positive[split])} positive sources, got {positive_count}"
                )
            if split in expected_annotations and annotation_count != int(expected_annotations[split]):
                errors.append(
                    f"{variant}/{split}: expected {int(expected_annotations[split])} Mass boxes, got {annotation_count}"
                )
            if len(ann_rows) != annotation_count:
                errors.append(
                    f"{variant}/{split}: annotation CSV has {len(ann_rows)} rows, manifest reports {annotation_count}"
                )

            for row in rows.to_dict("records"):
                for column, label in [
                    ("image_path", "image"),
                    ("label_path", "label"),
                    ("annotation_path", "per-image annotation"),
                ]:
                    relative = str(row.get(column, "") or "")
                    if not relative or not (crop_root / relative).is_file():
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: missing {label}"
                        )
                if int(row.get("num_annotations", 0) or 0) == 0:
                    label_path = crop_root / str(row.get("label_path", ""))
                    if label_path.is_file() and label_path.read_text(encoding="utf-8").strip():
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: negative label is not empty"
                        )

                float_path = str(row.get("float32_path", "") or "")
                if variant in float_required:
                    if not float_path or not (crop_root / float_path).is_file():
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: missing float32 tensor"
                        )
                    else:
                        observed[variant][split]["float32_tensors"] += 1
                    expected_shape = [3, int(row["height"]), int(row["width"])]
                    if str(row.get("float32_dtype", "")) != "float32":
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: float32 dtype metadata is invalid"
                        )
                    if str(row.get("float32_layout", "")) != "CHW":
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: float32 layout metadata is invalid"
                        )
                    if parse_shape(row.get("float32_shape")) != expected_shape:
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: float32 shape metadata is invalid"
                        )
                    try:
                        tensor_min = float(row.get("float32_min"))
                        tensor_max = float(row.get("float32_max"))
                    except (TypeError, ValueError):
                        tensor_min, tensor_max = math.nan, math.nan
                    if not (
                        math.isfinite(tensor_min)
                        and math.isfinite(tensor_max)
                        and 0.0 <= tensor_min <= tensor_max <= 1.0
                    ):
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: float32 min/max metadata is invalid"
                        )
                    image_stem = Path(str(row.get("image_path", ""))).stem
                    float_stem = Path(float_path).stem
                    if image_stem != float_stem:
                        errors.append(
                            f"{variant}/{split}/{row.get('source_image_id')}: PNG/tensor stems differ"
                        )

            for ann in ann_rows.to_dict("records"):
                source_box = parse_box(ann.get("source_bbox_xyxy"))
                box = parse_box(ann.get("bbox_xyxy"))
                if len(source_box) != 4 or len(box) != 4:
                    errors.append(f"{variant}/{split}: malformed annotation bbox")
                    continue
                width = float(ann.get("width", 0) or 0)
                height = float(ann.get("height", 0) or 0)
                if not all(math.isfinite(value) for value in box):
                    errors.append(f"{variant}/{split}: non-finite annotation bbox")
                    continue
                if not (0.0 <= box[0] < box[2] <= width and 0.0 <= box[1] < box[3] <= height):
                    errors.append(f"{variant}/{split}: out-of-bounds annotation bbox")
                pad_left = float(ann.get("pad_left", 0.0) or 0.0)
                pad_top = float(ann.get("pad_top", 0.0) or 0.0)
                scale_x = float(ann.get("scale_x", 1.0) or 1.0)
                scale_y = float(ann.get("scale_y", 1.0) or 1.0)
                expected_box = [
                    (source_box[0] + pad_left) * scale_x,
                    (source_box[1] + pad_top) * scale_y,
                    (source_box[2] + pad_left) * scale_x,
                    (source_box[3] + pad_top) * scale_y,
                ]
                if any(abs(left - right) > 1e-3 for left, right in zip(box, expected_box)):
                    errors.append(f"{variant}/{split}: annotation transform mismatch")

            coco_path = crop_root / _whole_coco_relative_path(
                config, variant, split
            )
            if not coco_path.is_file():
                errors.append(f"{variant}/{split}: missing aggregate COCO file")
            else:
                coco = json.loads(coco_path.read_text(encoding="utf-8"))
                coco_images = len(coco.get("images", []) or [])
                coco_annotations = len(coco.get("annotations", []) or [])
                observed[variant][split]["coco_images"] = int(coco_images)
                observed[variant][split]["coco_annotations"] = int(coco_annotations)
                if coco_images != len(rows) or coco_annotations != annotation_count:
                    errors.append(
                        f"{variant}/{split}: COCO counts {coco_images}/{coco_annotations} "
                        f"do not match manifest {len(rows)}/{annotation_count}"
                    )

    original_total = sum(
        observed.get("original", {}).get(split, {}).get("mass_annotations", 0)
        for split in ["train", "val", "test"]
    )
    resized_totals = {
        variant: sum(
            observed.get(variant, {}).get(split, {}).get("mass_annotations", 0)
            for split in ["train", "val", "test"]
        )
        for variant in expected_variants
        if variant == "resized" or variant.startswith("resized_")
    }
    for variant, resized_total in resized_totals.items():
        if original_total != resized_total:
            errors.append(
                f"{variant} model-input annotation count differs from completed source ground truth"
            )
    report = {
        "enabled": True,
        "status": "passed" if not errors else "failed",
        "source_images_without_model_inputs_added": 0 if not errors else None,
        "model_input_annotation_count": {
            variant: int(value) for variant, value in resized_totals.items()
        },
        "completed_source_ground_truth_annotation_count": int(original_total),
        "observed": observed,
        "errors": errors,
    }
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2), encoding="utf-8"
    )
    if errors and strict:
        preview = "; ".join(errors[:12])
        raise RuntimeError(
            f"Whole-image export contract failed with {len(errors)} error(s): {preview}"
        )
    return report, report_path


def _split_crop_cfg(crop_cfg: dict[str, Any], split_name: str, key: str, default: Any) -> Any:
    """Read split-specific square_crops option with global fallback.

    Example: for split_name="train" and key="deterministic_require_foreground",
    first checks train_deterministic_require_foreground. If that key is missing
    or explicitly null in YAML, it falls back to deterministic_require_foreground.
    """
    split_key = f"{split_name}_{key}"
    if split_key in crop_cfg and crop_cfg.get(split_key) is not None:
        return crop_cfg.get(split_key)
    if key in crop_cfg and crop_cfg.get(key) is not None:
        return crop_cfg.get(key)
    return default


def _breast_fraction_passes(
    fraction: float,
    threshold: float,
    comparison: str = "greater_than_or_equal",
) -> bool:
    mode = str(comparison or "greater_than_or_equal").strip().casefold()
    if mode in {"strictly_greater_than", "greater_than", ">", "strict"}:
        return float(fraction) > float(threshold)
    if mode in {
        "greater_than_or_equal", "greater_than_or_equal_to", ">=", "inclusive"
    }:
        return float(fraction) >= float(threshold)
    raise ValueError(
        "breast_fraction_comparison_for_all_crops must be "
        "'strictly_greater_than' or 'greater_than_or_equal'."
    )


def _deterministic_selection_mode(crop_cfg: dict[str, Any], split_name: str) -> str:
    mode = str(_split_crop_cfg(crop_cfg, split_name, "deterministic_selection_mode", "") or "").strip().casefold()
    aliases = {
        "all": "all",
        "all_windows": "all",
        "mass_only": "mass_only",
        "positive_only": "mass_only",
        "mass only": "mass_only",
        "positive_ratio": "positive_ratio",
        "all_mass_plus_sampled_non_mass": "positive_ratio",
        "all mass + sampled non-mass": "positive_ratio",
        "crop_label_ratio": "crop_label_ratio",
        "crop mass/empty ratio": "crop_label_ratio",
        "mass crops + negative-image empty crops": "crop_label_ratio",
        "positive_ratio_negative_images_only": "crop_label_ratio",
        "negative_fraction": "negative_fraction",
        "negative fraction": "negative_fraction",
        "all_positive_plus_negative_fraction": "negative_fraction",
        "all positive + fraction of negatives": "negative_fraction",
        "source_breast_ratio": "source_breast_ratio",
        "source breast ratio": "source_breast_ratio",
        "breast_status_ratio": "source_breast_ratio",
        "mass/negative breasts": "source_breast_ratio",
        "finding_images_all_windows": "finding_images_all_windows",
        "finding_images_only_all_windows": "finding_images_all_windows",
        "findings_images_all_windows": "finding_images_all_windows",
        "finding images, all windows": "finding_images_all_windows",
        "finding images only, all windows": "finding_images_all_windows",
        "images with findings, all windows": "finding_images_all_windows",
        "positive images, all windows": "finding_images_all_windows",
    }
    if mode in aliases:
        return aliases[mode]
    include_empty = bool(
        crop_cfg.get(
            f"{split_name}_deterministic_include_empty",
            crop_cfg.get("deterministic_include_empty", True),
        )
    )
    return "all" if include_empty else "mass_only"


def _deterministic_target_positive_ratio(crop_cfg: dict[str, Any], split_name: str) -> float:
    value = _split_crop_cfg(
        crop_cfg,
        split_name,
        "deterministic_target_positive_ratio",
        crop_cfg.get("deterministic_target_positive_ratio", crop_cfg.get("positive_fraction", 0.50)),
    )
    try:
        ratio = float(value)
    except Exception:
        ratio = 0.50
    return min(max(ratio, 0.01), 1.0)


def _deterministic_target_source_breast_mass_ratio(
    crop_cfg: dict[str, Any], split_name: str
) -> float:
    value = _split_crop_cfg(
        crop_cfg,
        split_name,
        "deterministic_target_source_breast_mass_ratio",
        _deterministic_target_positive_ratio(crop_cfg, split_name),
    )
    try:
        ratio = float(value)
    except Exception:
        ratio = 0.50
    return min(max(ratio, 0.01), 1.0)


def _deterministic_negative_keep_fraction(crop_cfg: dict[str, Any], split_name: str) -> float:
    """Fraction of negative deterministic candidates retained for a split."""
    value = _split_crop_cfg(
        crop_cfg,
        split_name,
        "deterministic_negative_keep_fraction",
        crop_cfg.get("deterministic_negative_keep_fraction", 0.20),
    )
    try:
        fraction = float(value)
    except Exception:
        fraction = 0.20
    return min(max(fraction, 0.0), 1.0)


def _target_positive_fraction(crop_cfg: dict[str, Any], split_name: str, *, bbox_safe: bool = False) -> float:
    """Target mass-positive crop ratio for random, bbox-safe, and legacy configs.

    0.50 means approximately one empty crop for each mass-positive crop.
    Split-specific values written by the GUI take precedence over global fallbacks.
    """
    keys = []
    if bbox_safe:
        keys.append(f"{split_name}_bbox_safe_positive_fraction")
    keys.extend([
        f"{split_name}_positive_fraction",
        f"{split_name}_deterministic_target_positive_ratio",
    ])
    if bbox_safe:
        keys.append("bbox_safe_positive_fraction")
    keys.extend(["positive_fraction", "deterministic_target_positive_ratio"])
    for key in keys:
        if key in crop_cfg and crop_cfg.get(key) is not None:
            try:
                ratio = float(crop_cfg.get(key))
                return min(max(ratio, 0.01), 1.0)
            except Exception:
                continue
    return 0.50


def _global_positive_ratio_selection_enabled(crop_cfg: dict[str, Any], split_name: str, split_mode: str) -> bool:
    split_key = f"{split_name}_global_positive_ratio_selection_for_random"
    if split_key in crop_cfg and crop_cfg.get(split_key) is not None:
        return bool(crop_cfg.get(split_key))
    if "global_positive_ratio_selection_for_random" in crop_cfg:
        return bool(crop_cfg.get("global_positive_ratio_selection_for_random"))
    # Legacy behavior: this flag used to suppress negatives from no-mass images.
    # It now means random/bbox-safe crops are globally selected to the target ratio.
    return bool(crop_cfg.get("balance_train_positive_fraction_globally", True)) and split_mode in {"random", "bbox_safe_random"}


def _crop_label_streaming_record_schedule(
    *,
    records: list[dict[str, Any]],
    source_image_has_mass_lookup: dict[str, int],
    source_breast_has_mass_lookup: set[tuple[str, str]],
    target_positive_ratio: float,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Schedule all positive images plus a bounded negative-breast subset.

    Crop-label balancing does not need to decode every Mass-negative mammogram.
    Source annotations already identify every image capable of producing a
    positive window and every breast eligible to supply negatives. Positives
    are therefore mandatory and are interleaved with a seeded negative subset
    using a compact integer cadence derived from the target. The minority
    interval is ``ceil(1 / minority_fraction)``; one minority source is placed
    in each interval. This makes crop files appear immediately and avoids a
    long debug-only prefix or a scan over thousands of unused negative sources.
    """
    positive_records: list[dict[str, Any]] = []
    eligible_negative_records: list[dict[str, Any]] = []
    ineligible_records = 0
    for record in records:
        image_id = str(record.get("image_id", ""))
        if int(source_image_has_mass_lookup.get(image_id, 0)) == 1:
            positive_records.append(record)
            continue
        try:
            breast_has_mass = _breast_key(record) in source_breast_has_mass_lookup
        except Exception:
            breast_has_mass = True
        if not breast_has_mass:
            eligible_negative_records.append(record)
        else:
            # The paired view of a Mass-positive breast cannot provide an
            # eligible empty crop under crop_label_ratio.
            ineligible_records += 1

    if shuffle:
        if positive_records:
            order = rng.permutation(len(positive_records))
            positive_records = [positive_records[int(index)] for index in order]
        if eligible_negative_records:
            order = rng.permutation(len(eligible_negative_records))
            eligible_negative_records = [
                eligible_negative_records[int(index)] for index in order
            ]

    try:
        ratio = min(max(float(target_positive_ratio), 0.01), 1.0)
    except Exception:
        ratio = 0.50
    positive_cadence, negative_cadence = _compact_source_cadence(ratio)
    negative_budget = min(
        len(eligible_negative_records),
        (
            int(math.ceil(
                len(positive_records)
                * float(negative_cadence)
                / float(max(1, positive_cadence))
            ))
            if negative_cadence > 0
            else 0
        ),
    )
    selected_negatives = eligible_negative_records[:negative_budget]
    reserve_negatives = eligible_negative_records[negative_budget:]

    scheduled: list[dict[str, Any]] = []
    positive_cursor = 0
    negative_cursor = 0
    while positive_cursor < len(positive_records):
        # Every block begins with its positive quota, guaranteeing that the
        # first processed source is positive and making the cadence explicit.
        for _ in range(positive_cadence):
            if positive_cursor >= len(positive_records):
                break
            scheduled.append(positive_records[positive_cursor])
            positive_cursor += 1
        for _ in range(negative_cadence):
            if negative_cursor >= negative_budget:
                break
            scheduled.append(selected_negatives[negative_cursor])
            negative_cursor += 1
    while negative_cursor < negative_budget:
        scheduled.append(selected_negatives[negative_cursor])
        negative_cursor += 1

    return scheduled, reserve_negatives, {
        "positive_source_images": int(len(positive_records)),
        "positive_source_cadence": int(positive_cadence),
        "negative_source_cadence": int(negative_cadence),
        "source_cadence_block_size": int(positive_cadence + negative_cadence),
        "eligible_negative_source_images": int(len(eligible_negative_records)),
        "scheduled_negative_source_images": int(negative_budget),
        "skipped_ineligible_source_images": int(ineligible_records),
        "unscheduled_negative_source_images": int(
            len(reserve_negatives)
        ),
    }


def _compact_source_cadence(target_positive_ratio: float) -> tuple[int, int]:
    """Derive a short positive/negative source block from a target ratio.

    The less common class occurs once every ``ceil(1 / minority_fraction)``
    sources. Integer arithmetic over a bounded Fraction avoids floating-point
    errors such as ``ceil(1 / 0.2)`` accidentally becoming six.

    Examples:
      - 0.50 -> (1 positive, 1 negative)
      - 0.80 -> (4 positive, 1 negative)
      - 0.70 -> (3 positive, 1 negative), the compact rounded cadence
      - 0.30 -> (1 positive, 3 negative), its symmetric counterpart
    """
    try:
        ratio = min(max(float(target_positive_ratio), 0.01), 1.0)
    except Exception:
        ratio = 0.50
    fraction = Fraction(str(ratio)).limit_denominator(10_000)
    positive_units = int(fraction.numerator)
    total_units = int(fraction.denominator)
    negative_units = max(0, total_units - positive_units)
    if negative_units == 0:
        return 1, 0

    def ceil_div(numerator: int, denominator: int) -> int:
        return (int(numerator) + int(denominator) - 1) // int(denominator)

    if positive_units >= negative_units:
        block_size = max(2, ceil_div(total_units, negative_units))
        return block_size - 1, 1
    block_size = max(2, ceil_div(total_units, positive_units))
    return 1, block_size - 1


def _online_positive_ratio_selection_enabled(crop_cfg: dict[str, Any], split_name: str, split_mode: str) -> bool:
    """Use a single-pass approximate target ratio for supported crop modes.

    Unlike global positive-ratio selection, this does not collect all candidates first.
    It writes positive crops immediately and accepts empty candidates only when the
    running saved counts need more empty crops. This is intentionally approximate,
    but it avoids the long planning stage and lets PNGs appear during export.
    """
    split_mode = str(split_mode or "").casefold().strip()
    execution = str(_split_crop_cfg(
        crop_cfg,
        split_name,
        "balance_execution",
        "",
    ) or "").casefold().strip().replace("-", "_")
    force_streaming = execution in {
        "streaming",
        "streaming_one_pass",
        "online",
        "online_one_pass",
    }
    if split_mode == "deterministic":
        if _deterministic_selection_mode(crop_cfg, split_name) not in {
            "positive_ratio",
            "crop_label_ratio",
        }:
            return False
        if force_streaming:
            return True
        split_key = f"{split_name}_online_positive_ratio_selection_for_deterministic"
        if split_key in crop_cfg and crop_cfg.get(split_key) is not None:
            return bool(crop_cfg.get(split_key))
        if "online_positive_ratio_selection_for_deterministic" in crop_cfg:
            return bool(crop_cfg.get("online_positive_ratio_selection_for_deterministic"))
        return False
    if split_mode not in {"random", "bbox_safe_random"}:
        return False
    if force_streaming:
        return True
    split_key = f"{split_name}_online_positive_ratio_selection_for_random"
    if split_key in crop_cfg and crop_cfg.get(split_key) is not None:
        return bool(crop_cfg.get(split_key))
    if "online_positive_ratio_selection_for_random" in crop_cfg:
        return bool(crop_cfg.get("online_positive_ratio_selection_for_random"))
    return False


def _online_should_save_negative(positive_count: int, negative_count: int, target_ratio: float) -> bool:
    """Return True when the running split ratio needs another empty crop."""
    try:
        target_ratio = min(max(float(target_ratio), 0.01), 1.0)
    except Exception:
        target_ratio = 0.50
    if target_ratio >= 1.0:
        return False
    if positive_count <= 0:
        return False
    desired_negatives = int(round(float(positive_count) * (1.0 - target_ratio) / target_ratio))
    return int(negative_count) < max(0, desired_negatives)


def _online_balance_extra_info(
    extra_info: dict[str, Any],
    *,
    split_name: str,
    target_ratio: float,
    positive_count: int,
    negative_count: int,
    split_mode: str = "random",
) -> dict[str, Any]:
    total = int(positive_count) + int(negative_count)
    achieved = float(positive_count / total) if total > 0 else 0.0
    return {
        **dict(extra_info),
        "online_positive_ratio_selection": (
            f"online_{split_name}_{str(split_mode or 'random').casefold()}_positive_ratio"
        ),
        "balance_execution": "streaming_one_pass_no_global_planning",
        "positive_window_policy": "keep_all_eligible_positive_windows",
        "negative_window_policy": "admit_when_running_counts_are_below_target",
        "online_target_positive_ratio": float(target_ratio),
        "online_running_positive_windows": int(positive_count),
        "online_running_negative_windows": int(negative_count),
        "online_running_achieved_positive_ratio": float(achieved),
        # Also fill the older summary/debug names so samples.csv stays easy to compare.
        "global_positive_ratio_selection": "online_streaming_approximate",
        "global_target_positive_ratio": float(target_ratio),
        "global_achieved_positive_ratio": float(achieved),
        "global_selected_positive_windows": int(positive_count),
        "global_selected_negative_windows": int(negative_count),
        "deterministic_target_positive_ratio": float(target_ratio),
        "deterministic_achieved_positive_ratio": float(achieved),
        "deterministic_selected_positive_windows": int(positive_count),
        "deterministic_selected_negative_windows": int(negative_count),
    }


def _select_positive_ratio_candidates(
    candidates: list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]],
    crop_cfg: dict[str, Any],
    split_name: str,
    rng: np.random.Generator,
    *,
    target_ratio: float | None = None,
    selection_label: str = "positive_ratio",
    negative_images_only: bool = False,
    negative_breasts_only: bool = False,
) -> list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]]:
    """Keep all positive crops and globally sample negative candidates.

    This lets random/bbox-safe exports create one crop per annotation while still
    drawing the empty half of a 50/50 dataset from any clean candidate, including
    source images that contain no mass at all.
    """
    if target_ratio is None:
        target_ratio = _deterministic_target_positive_ratio(crop_cfg, split_name)
    try:
        target_ratio = min(max(float(target_ratio), 0.01), 1.0)
    except Exception:
        target_ratio = 0.50

    positives = [c for c in candidates if int(c[2].get("is_positive_window", 0)) == 1]
    negatives = [
        candidate
        for candidate in candidates
        if int(candidate[2].get("is_positive_window", 0)) == 0
        and (
            not negative_images_only
            or int(candidate[2].get("source_image_has_mass", 0)) == 0
        )
        and (
            not negative_breasts_only
            or int(candidate[2].get("source_breast_has_mass", 0)) == 0
        )
    ]
    if not positives:
        return []
    if target_ratio >= 1.0 or not negatives:
        selected_negatives = []
    else:
        wanted_negatives = int(round(len(positives) * (1.0 - target_ratio) / target_ratio))
        wanted_negatives = min(max(0, wanted_negatives), len(negatives))
        if wanted_negatives > 0:
            indices = rng.choice(len(negatives), size=wanted_negatives, replace=False)
            selected_negatives = [negatives[int(i)] for i in indices]
        else:
            selected_negatives = []
    selected = positives + selected_negatives

    # Keep deterministic/reproducible ordering by source image and window position.
    selected.sort(key=lambda c: (str(c[0].get("image_id", "")), int(c[1][1]), int(c[1][0]), int(c[1][3]), int(c[1][2])))
    selected_count = len(selected)
    positive_count = len(positives)
    negative_count = len(selected_negatives)
    achieved_ratio = float(positive_count / selected_count) if selected_count else 0.0
    out = []
    for record, window, extra in selected:
        e = dict(extra)
        e["global_positive_ratio_selection"] = selection_label
        e["global_target_positive_ratio"] = float(target_ratio)
        e["global_achieved_positive_ratio"] = float(achieved_ratio)
        e["global_selected_positive_windows"] = int(positive_count)
        e["global_selected_negative_windows"] = int(negative_count)
        e["negative_crop_source_policy"] = (
            "mass_negative_breasts_only"
            if negative_breasts_only
            else "mass_negative_images_only"
            if negative_images_only
            else "any_source_image"
        )
        # Backward-compatible field names used by deterministic exports and old summaries.
        e["deterministic_target_positive_ratio"] = float(target_ratio)
        e["deterministic_achieved_positive_ratio"] = float(achieved_ratio)
        e["deterministic_selected_positive_windows"] = int(positive_count)
        e["deterministic_selected_negative_windows"] = int(negative_count)
        out.append((record, window, e))
    return out


def _select_negative_fraction_candidates(
    candidates: list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]],
    crop_cfg: dict[str, Any],
    split_name: str,
    rng: np.random.Generator,
) -> list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]]:
    """Keep every positive candidate and a seeded fraction of negatives.

    This is deliberately different from target-positive-ratio balancing. A
    20% negative keep fraction means 20% of all eligible negative patch
    candidates, matching papers that report negative *retention* rather than a
    final positive/negative class ratio.
    """
    positives = [c for c in candidates if int(c[2].get("is_positive_window", 0)) == 1]
    negatives = [c for c in candidates if int(c[2].get("is_positive_window", 0)) == 0]
    keep_fraction = _deterministic_negative_keep_fraction(crop_cfg, split_name)
    wanted = min(len(negatives), max(0, int(round(len(negatives) * keep_fraction))))
    if wanted >= len(negatives):
        selected_negatives = negatives
    elif wanted > 0:
        indices = rng.choice(len(negatives), size=wanted, replace=False)
        selected_negatives = [negatives[int(i)] for i in indices]
    else:
        selected_negatives = []

    selected = positives + selected_negatives
    selected.sort(key=lambda c: (str(c[0].get("image_id", "")), int(c[1][1]), int(c[1][0]), int(c[1][3]), int(c[1][2])))
    achieved = float(len(selected_negatives) / len(negatives)) if negatives else 0.0
    out = []
    for record, window, extra in selected:
        out.append((record, window, {
            **dict(extra),
            "negative_fraction_selection": f"global_{split_name}_negative_fraction",
            "negative_candidate_count": int(len(negatives)),
            "negative_keep_fraction": float(keep_fraction),
            "negative_selected_count": int(len(selected_negatives)),
            "negative_achieved_keep_fraction": float(achieved),
        }))
    return out


def _select_source_breast_ratio_candidates(
    candidates: list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]],
    crop_cfg: dict[str, Any],
    split_name: str,
    rng: np.random.Generator,
) -> list[tuple[dict[str, Any], tuple[int, int, int, int], dict[str, Any]]]:
    """Balance crops by the status of their source breast, not tile labels.

    A mass-breast crop can itself be empty; its group is determined by whether
    any view from the same ``(study_id, laterality)`` breast has a valid Mass.
    All lesion-containing candidate windows are mandatory. Remaining windows
    are sampled without replacement to reach the largest feasible requested
    mixture. For the improved Paper 22 preset this is an exact 50/50 split.
    """
    target = _deterministic_target_source_breast_mass_ratio(
        crop_cfg, split_name
    )
    mass_pool = [
        candidate
        for candidate in candidates
        if int(candidate[2].get("source_breast_has_mass", 0)) == 1
    ]
    negative_pool = [
        candidate
        for candidate in candidates
        if int(candidate[2].get("source_breast_has_mass", 0)) == 0
    ]
    mandatory_mass = [
        candidate
        for candidate in mass_pool
        if int(candidate[2].get("is_positive_window", 0)) == 1
    ]
    if not mass_pool or (target < 1.0 and not negative_pool):
        raise RuntimeError(
            f"{split_name}: source-breast balancing needs both mass-positive and "
            f"negative breast candidates; got {len(mass_pool)} and {len(negative_pool)}."
        )

    if target >= 1.0:
        mass_count = len(mass_pool)
        negative_count = 0
    else:
        # Largest feasible integer mixture.  For 0.50 this is exactly the
        # smaller pool size from each group.
        max_total = min(
            int(math.floor(len(mass_pool) / target)),
            int(math.floor(len(negative_pool) / (1.0 - target))),
        )
        mass_count = min(len(mass_pool), int(round(max_total * target)))
        negative_count = min(len(negative_pool), max_total - mass_count)
        # Correct rounding toward the requested ratio without exceeding pools.
        if mass_count + negative_count > 0:
            candidates_nearby: list[tuple[float, int, int, int]] = []
            for dm in [-1, 0, 1]:
                for dn in [-1, 0, 1]:
                    m = mass_count + dm
                    n = negative_count + dn
                    if 0 <= m <= len(mass_pool) and 0 <= n <= len(negative_pool) and m + n > 0:
                        candidates_nearby.append((abs(m / (m + n) - target), -(m + n), m, n))
            if candidates_nearby:
                _error, _negative_total, mass_count, negative_count = min(candidates_nearby)

    if len(mandatory_mass) > mass_count:
        raise RuntimeError(
            f"{split_name}: exact source-breast ratio would keep only {mass_count} "
            f"mass-breast crops, fewer than the {len(mandatory_mass)} lesion-containing "
            "windows. Add more negative-breast sources or raise the target mass-breast ratio."
        )

    mandatory_ids = {id(candidate) for candidate in mandatory_mass}
    optional_mass = [candidate for candidate in mass_pool if id(candidate) not in mandatory_ids]
    optional_mass_count = mass_count - len(mandatory_mass)

    def sample(pool, count):
        if count >= len(pool):
            return list(pool)
        if count <= 0:
            return []
        indices = rng.choice(len(pool), size=count, replace=False)
        return [pool[int(index)] for index in indices]

    selected_mass = mandatory_mass + sample(optional_mass, optional_mass_count)
    selected_negative = sample(negative_pool, negative_count)
    selected = selected_mass + selected_negative
    selected.sort(
        key=lambda candidate: (
            str(candidate[0].get("image_id", "")),
            int(candidate[1][1]),
            int(candidate[1][0]),
            int(candidate[1][3]),
            int(candidate[1][2]),
        )
    )
    achieved = float(len(selected_mass) / len(selected)) if selected else 0.0
    out = []
    for record, window, extra in selected:
        out.append((record, window, {
            **dict(extra),
            "source_breast_ratio_selection": f"global_{split_name}_source_breast_ratio",
            "source_breast_mass_candidate_count": len(mass_pool),
            "source_breast_negative_candidate_count": len(negative_pool),
            "source_breast_mandatory_positive_window_count": len(mandatory_mass),
            "source_breast_selected_mass_count": len(selected_mass),
            "source_breast_selected_negative_count": len(selected_negative),
            "source_breast_target_mass_ratio": float(target),
            "source_breast_achieved_mass_ratio": float(achieved),
        }))
    return out


def _windows_for_export_split(
    *,
    split_name: str,
    image_width: int,
    image_height: int,
    image_tensor: torch.Tensor,
    mass_boxes: torch.Tensor,
    crop_options: dict[str, Any],
    crop_cfg: dict[str, Any],
    rng: np.random.Generator,
    foreground_mask: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> list[tuple[tuple[int, int, int, int], dict[str, Any]]]:
    """Return crop windows for one image according to train/val/test policy."""
    split_mode = str(crop_cfg.get(f"{split_name}_crop_mode", "random" if split_name == "train" else "deterministic")).casefold().strip()
    if split_mode not in {"random", "deterministic", "bbox_safe_random"}:
        raise ValueError(
            f"square_crops.{split_name}_crop_mode must be 'random', 'deterministic', or 'bbox_safe_random', got {split_mode!r}."
        )

    if split_mode == "deterministic":
        windows = sliding_square_windows(
            width=image_width,
            height=image_height,
            crop_size=int(crop_cfg.get("crop_size", 1024)),
            stride=int(crop_cfg.get("stride", 512)),
            edge_policy=str(crop_cfg.get("edge_policy", "edge_align")),
        )

        selection_mode = _deterministic_selection_mode(crop_cfg, split_name)
        target_positive_ratio = _deterministic_target_positive_ratio(crop_cfg, split_name)
        include_empty = selection_mode not in {"mass_only"}
        source_image_has_finding = bool(mass_boxes.detach().cpu().reshape(-1, 4).shape[0] > 0)

        # Compute positivity before any selection so the positive-ratio mode can
        # keep all mass windows and then globally sample non-mass windows.
        # The finding-images mode keeps all windows from images that contain at
        # least one mass/finding anywhere, but skips source images with no mass.
        positive_by_window = {
            w: bool(window_has_positive_mass(w, mass_boxes, crop_options))
            for w in windows
        }
        clean_negative_by_window = {
            w: bool(window_is_clean(w, mass_boxes, crop_options))
            for w in windows
            if not positive_by_window.get(w, False)
        }
        require_clean_negatives = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "require_clean_negative_windows",
            selection_mode == "negative_fraction",
        ))
        complete_grid_count = len(windows)
        positive_grid_count = sum(int(value) for value in positive_by_window.values())
        if selection_mode == "mass_only":
            windows = [w for w in windows if positive_by_window.get(w, False)]
        elif selection_mode == "finding_images_all_windows" and not source_image_has_finding:
            windows = []
        if require_clean_negatives:
            windows = [
                w for w in windows
                if positive_by_window.get(w, False) or clean_negative_by_window.get(w, False)
            ]

        all_crop_breast_filter_enabled = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "require_min_breast_fraction_for_all_crops",
            False,
        ))
        preserve_positive_below_min_breast_fraction = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "preserve_positive_windows_below_min_breast_fraction",
            False,
        ))
        require_retained_breast_mask = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "require_retained_breast_mask_for_all_crops",
            False,
        ))
        foreground_filter_enabled = all_crop_breast_filter_enabled or bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "deterministic_require_foreground",
            False,
        ))
        if all_crop_breast_filter_enabled:
            min_foreground_fraction = float(_split_crop_cfg(
                crop_cfg,
                split_name,
                "min_breast_fraction_for_all_crops",
                0.30,
            ))
            breast_fraction_comparison = str(_split_crop_cfg(
                crop_cfg,
                split_name,
                "breast_fraction_comparison_for_all_crops",
                "greater_than_or_equal",
            ))
        else:
            min_foreground_fraction = float(_split_crop_cfg(
                crop_cfg,
                split_name,
                "deterministic_min_foreground_fraction",
                0.05,
            ))
            breast_fraction_comparison = "greater_than_or_equal"
        foreground_threshold = crop_cfg.get("deterministic_foreground_threshold", None)
        foreground_fractions: dict[tuple[int, int, int, int], float] = {}
        foreground_mask_source = ""
        if foreground_filter_enabled:
            if foreground_mask is None:
                if all_crop_breast_filter_enabled and require_retained_breast_mask:
                    raise RuntimeError(
                        "The all-crop breast-occupancy policy requires the retained "
                        "fixed-preprocessing breast mask, but no mask was supplied. "
                        "Set preprocess.retain_breast_mask_for_export=true."
                    )
                image_np = image_tensor.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
                foreground_mask = _foreground_mask(image_np, threshold=foreground_threshold)
                foreground_mask_source = "derived_from_preprocessed_image"
            else:
                foreground_mask = np.asarray(foreground_mask, dtype=bool)
                foreground_mask_source = "retained_preprocessing_mask"
            if tuple(foreground_mask.shape[:2]) != (int(image_height), int(image_width)):
                raise ValueError(
                    "Breast foreground mask shape does not match the fixed-preprocessed "
                    f"image: mask={tuple(foreground_mask.shape)}, "
                    f"image={(int(image_height), int(image_width))}."
                )
            kept_windows = []
            for w in windows:
                frac = _foreground_fraction_from_mask(
                    foreground_mask,
                    w,
                    int(crop_cfg.get("crop_size", 1024)),
                )
                foreground_fractions[w] = frac
                # Legacy deterministic filters keep positive windows
                # unconditionally. An explicit all-crop policy remains strict by
                # default, but presets may opt into the same positive safeguard
                # while still filtering low-tissue negative/background windows.
                if (
                    (
                        positive_by_window.get(w, False)
                        and (
                            not all_crop_breast_filter_enabled
                            or preserve_positive_below_min_breast_fraction
                        )
                    )
                    or _breast_fraction_passes(
                        frac,
                        min_foreground_fraction,
                        breast_fraction_comparison,
                    )
                ):
                    kept_windows.append(w)
            windows = kept_windows

        max_windows = crop_cfg.get(f"{split_name}_deterministic_max_windows_per_image", crop_cfg.get("deterministic_max_windows_per_image"))
        if max_windows is not None:
            windows = windows[: int(max_windows)]
        if diagnostics is not None:
            diagnostics.update({
                "complete_grid_windows": int(complete_grid_count),
                "positive_candidate_windows": int(positive_grid_count),
                "ambiguous_partial_windows": int(sum(
                    1
                    for w, clean in clean_negative_by_window.items()
                    if not clean
                )),
                "returned_candidate_windows": int(len(windows)),
            })
        return [
            (
                w,
                {
                    "crop_mode": "deterministic",
                    "split_crop_mode": split_mode,
                    "deterministic_include_empty": int(include_empty),
                    "deterministic_selection_mode": selection_mode,
                    "deterministic_target_positive_ratio": float(target_positive_ratio),
                    "source_image_has_finding": int(source_image_has_finding),
                    "is_positive_window": int(bool(positive_by_window.get(w, False))),
                    "is_clean_negative_window": int(bool(clean_negative_by_window.get(w, False))),
                    "require_clean_negative_windows": int(require_clean_negatives),
                    "foreground_filter_enabled": int(foreground_filter_enabled),
                    "all_crop_breast_fraction_filter_enabled": int(all_crop_breast_filter_enabled),
                    "preserve_positive_windows_below_min_breast_fraction": int(
                        preserve_positive_below_min_breast_fraction
                    ),
                    "require_retained_breast_mask_for_all_crops": int(require_retained_breast_mask),
                    "breast_fraction_mask_source": foreground_mask_source,
                    "min_foreground_fraction": float(min_foreground_fraction),
                    "min_breast_fraction_for_all_crops": (
                        float(min_foreground_fraction) if all_crop_breast_filter_enabled else None
                    ),
                    "breast_fraction_comparison_for_all_crops": breast_fraction_comparison,
                    "foreground_fraction": foreground_fractions.get(w, None),
                },
            )
            for w in windows
        ]


    if split_mode == "bbox_safe_random":
        boxes = mass_boxes.detach().cpu().to(torch.float32).reshape(-1, 4)
        windows: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
        crops_per_ann = int(crop_cfg.get("bbox_safe_crops_per_annotation", crop_cfg.get("random_crops_per_annotation", 5)))
        positive_fraction = _target_positive_fraction(crop_cfg, split_name, bbox_safe=True)
        global_balance = _global_positive_ratio_selection_enabled(crop_cfg, split_name, split_mode)

        safe_options = dict(crop_options)
        for key in [
            "bbox_safe_boundary_margin_fraction",
            "bbox_safe_random_shift_fraction",
            "bbox_safe_candidate_count",
            "bbox_safe_top_k",
            "bbox_safe_breast_bias_strength",
            "bbox_safe_left_bias_strength",
            "bbox_safe_projection_bias_strength",
            "bbox_safe_foreground_threshold",
            "bbox_safe_skip_unsafe_fallbacks",
        ]:
            if key in crop_cfg:
                safe_options[key] = crop_cfg.get(key)

        for ann_index, box in enumerate(boxes):
            for _ in range(max(0, crops_per_ann)):
                window, info = sample_bbox_safe_breast_biased_square_window(
                    image_width=image_width,
                    image_height=image_height,
                    image_tensor=image_tensor,
                    box_xyxy=box,
                    all_mass_boxes=boxes,
                    options=safe_options,
                    rng=rng,
                )
                if bool(crop_cfg.get("bbox_safe_skip_unsafe_fallbacks", True)) and not bool(info.get("accepted", True)):
                    continue
                windows.append((window, {"crop_mode": "bbox_safe_random", "annotation_index": int(ann_index), "target_positive_fraction": float(positive_fraction), "is_positive_window": 1, **info}))

        num_positive = len(windows)
        if global_balance:
            # Candidate negatives are generated from both finding and no-finding images.
            # The final 50/50 ratio is enforced globally by _select_positive_ratio_candidates.
            num_negative = int(crop_cfg.get(
                "bbox_safe_random_crops_per_negative_image_when_balancing",
                crop_cfg.get("global_negative_candidate_crops_per_image_when_balancing", crop_cfg.get("random_crops_per_negative_image_when_balancing", 1)),
            ))
        elif num_positive > 0 and positive_fraction > 0:
            num_negative = int(round(num_positive * max(0.0, 1.0 - positive_fraction) / positive_fraction))
        else:
            num_negative = int(crop_cfg.get("bbox_safe_random_crops_per_negative_image", crop_cfg.get("random_crops_per_negative_image", 1)))

        if boxes.shape[0] == 0 and not global_balance:
            if bool(crop_cfg.get("balance_train_positive_fraction_globally", True)):
                num_negative = int(crop_cfg.get("random_crops_per_negative_image_when_balancing", 0))
            else:
                num_negative = int(crop_cfg.get("bbox_safe_random_crops_per_negative_image", crop_cfg.get("random_crops_per_negative_image", 1)))

        clean_options = dict(safe_options)
        clean_options["positive_fraction"] = 0.0
        for _ in range(max(0, num_negative)):
            window, info = sample_breast_biased_clean_square_window(
                image_width=image_width,
                image_height=image_height,
                image_tensor=image_tensor,
                mass_boxes=boxes,
                options=clean_options,
                rng=rng,
            )
            if bool(crop_cfg.get("bbox_safe_skip_unsafe_fallbacks", True)) and boxes.shape[0] > 0 and not bool(info.get("accepted", True)):
                continue
            windows.append((window, {"crop_mode": "bbox_safe_random_clean", "target_positive_fraction": float(positive_fraction), "is_positive_window": 0, **info}))
        return windows

    # Random mode. Positive crops are centered around annotations.
    boxes = mass_boxes.detach().cpu().to(torch.float32).reshape(-1, 4)
    windows: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    crops_per_ann = int(crop_cfg.get("random_crops_per_annotation", 5))
    positive_fraction = _target_positive_fraction(crop_cfg, split_name, bbox_safe=False)
    global_balance = _global_positive_ratio_selection_enabled(crop_cfg, split_name, split_mode)

    for ann_index, box in enumerate(boxes):
        for _ in range(max(0, crops_per_ann)):
            window, info = sample_box_centered_square_window(
                image_width=image_width,
                image_height=image_height,
                box_xyxy=box,
                all_mass_boxes=boxes,
                options=crop_options,
                rng=rng,
            )
            windows.append((window, {"crop_mode": "random", "annotation_index": int(ann_index), "target_positive_fraction": float(positive_fraction), "is_positive_window": 1, **info}))

    num_positive = len(windows)
    if global_balance:
        # Generate negative candidates from all images, including no-mass images.
        # The final target ratio is enforced globally after all candidates are planned.
        num_negative = int(crop_cfg.get(
            "global_negative_candidate_crops_per_image_when_balancing",
            crop_cfg.get("random_crops_per_negative_image_when_balancing", 1),
        ))
    elif num_positive > 0 and positive_fraction > 0:
        num_negative = int(round(num_positive * max(0.0, 1.0 - positive_fraction) / positive_fraction))
    else:
        num_negative = int(crop_cfg.get("random_crops_per_negative_image", 1))

    if boxes.shape[0] == 0 and not global_balance:
        # Legacy behavior: optionally suppress no-mass-image negatives when using
        # old per-image balancing.
        if bool(crop_cfg.get("balance_train_positive_fraction_globally", True)):
            num_negative = int(crop_cfg.get("random_crops_per_negative_image_when_balancing", 0))
        else:
            num_negative = int(crop_cfg.get("random_crops_per_negative_image", 1))

    clean_options = dict(crop_options)
    clean_options["positive_fraction"] = 0.0
    for _ in range(max(0, num_negative)):
        window, info = sample_random_square_window(
            image_width=image_width,
            image_height=image_height,
            mass_boxes=boxes,
            options=clean_options,
            rng=rng,
        )
        windows.append((window, {"crop_mode": "random_clean", "target_positive_fraction": float(positive_fraction), "is_positive_window": 0, **info}))
    return windows


def _opposite_laterality(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        return "R"
    if text.startswith("R"):
        return "L"
    return None


def _build_contralateral_record_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        study = str(record.get("study_id", ""))
        view = str(record.get("view_position", "")).upper().strip()
        lat = str(record.get("laterality", "")).upper().strip()[:1]
        if study and view and lat in {"L", "R"}:
            by_key[(study, view, lat)] = record

    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        study = str(record.get("study_id", ""))
        view = str(record.get("view_position", "")).upper().strip()
        opposite = _opposite_laterality(record.get("laterality"))
        if image_id and opposite:
            paired = by_key.get((study, view, opposite))
            if paired is not None:
                lookup[image_id] = paired
    return lookup


def _contralateral_source_arrays_for_window(
    *,
    record: dict[str, Any],
    reference_image: torch.Tensor,
    window: tuple[int, int, int, int],
    crop_options: dict[str, Any],
    contralateral_lookup: dict[str, dict[str, Any]],
    get_preprocessed,
    config: dict[str, Any],
    alignment_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] | None = None,
    max_alignment_cache_items: int = 2,
    profiler: SimpleTimerProfiler | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paired_record = contralateral_lookup.get(str(record.get("image_id", "")))
    if paired_record is None:
        return {}, {"contralateral_alignment_requested": True, "contralateral_alignment_found_pair": False}
    paired_image, _paired_target = get_preprocessed(paired_record)
    align_options = dict((config.get("image_export", {}) or {}).get("contralateral_source_alignment", {}) or {})
    ref_id = str(record.get("image_id", ""))
    pair_id = str(paired_record.get("image_id", ""))
    cache_key = (
        ref_id, pair_id, str(align_options.get("method", "")), str(align_options.get("fallback_method", "")),
        float(align_options.get("max_shift_fraction", 0.0) or 0.0),
        float(align_options.get("tip_tolerance_fraction", 0.0) or 0.0),
        int(align_options.get("smooth_rows", 0) or 0),
        int(align_options.get("projection_smooth_rows", 0) or 0),
        int(align_options.get("boundary_smooth_rows", 0) or 0),
    )
    if alignment_cache is not None and cache_key in alignment_cache:
        cached_info = alignment_cache[cache_key]
        alignment_cache.move_to_end(cache_key)
        alignment_info = {**cached_info, "contralateral_alignment_cache_hit": True}
        if profiler is not None:
            profiler.record("contralateral alignment cache hit", 0.0)
    else:
        if profiler is not None:
            profiler.start("contralateral alignment estimate")
        alignment_info = estimate_contralateral_alignment_info(reference_image, paired_image, options=align_options)
        if profiler is not None:
            profiler.stop("contralateral alignment estimate")
        alignment_info = {**alignment_info, "contralateral_alignment_cache_hit": False}
        if alignment_cache is not None:
            alignment_cache[cache_key] = dict(alignment_info)
            alignment_cache.move_to_end(cache_key)
            while len(alignment_cache) > max(1, int(max_alignment_cache_items)):
                alignment_cache.popitem(last=False)
    alignment_info = {
        "contralateral_alignment_requested": True,
        "contralateral_alignment_found_pair": True,
        "contralateral_image_id": pair_id,
        **alignment_info,
    }
    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    shift_y = int(round(float(alignment_info.get("contralateral_alignment_shift_y", 0) or 0)))
    x0, y0, x1, y1 = [int(v) for v in window]
    # Cropping the shifted image at y is equivalent to cropping the original
    # image at y-shift_y. This avoids copying the full contralateral mammogram.
    adjusted_window = (x0, y0 - shift_y, x1, y1 - shift_y)
    alignment_info["contralateral_fast_adjusted_window_xyxy"] = adjusted_window
    if profiler is not None:
        profiler.start("crop aligned contralateral image")
    paired_crop = crop_image_and_boxes_to_window(paired_image, boxes=empty_boxes, mass_boxes=empty_boxes, window_xyxy=adjusted_window, options=crop_options)
    if profiler is not None:
        profiler.stop("crop aligned contralateral image")
    return {
        "contralateral_same_view_crop": _tensor_to_float2d(paired_crop.image),
        "contralateral_same_view_full": _tensor_to_float2d(paired_image),
    }, alignment_info


def export_baseline_dataset(
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Export the baseline set after preprocessing but before n x n square crops."""
    base_root = output_root / "baseline_uncropped"
    base_root.mkdir(parents=True, exist_ok=True)
    save_empty_labels = bool(config.get("export", {}).get("save_empty_label_files", True))
    show_progress = bool(config.get("runtime", {}).get("show_progress", True))
    crop_cfg = dict(config.get("square_crops", {}) or {})
    baseline_cfg = dict(config.get("baseline_uncropped", {}) or {})

    coco_by_split = {split: _empty_coco() for split in ["train", "val", "test"]}
    stats_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    image_id_counter = 1
    ann_id_counter = 1
    # Keep a tiny cache only. A full VinDr export can contain thousands of large
    # mammograms, so caching every tensor would consume too much RAM. The small
    # cache still avoids immediate rereads when paired views are processed near
    # each other.
    preprocessed_cache: OrderedDict[str, tuple[torch.Tensor, dict[str, Any]]] = OrderedDict()
    max_preprocessed_cache_items = int(crop_cfg.get("contralateral_preprocessed_cache_items", 8))
    contralateral_lookup = _build_contralateral_record_lookup(dataset.image_records)
    needs_contralateral = _config_uses_contralateral_source(config)
    total_records_for_progress = sum(len(split_records.get(split, [])) for split in ["train", "val", "test"])
    processed_records_for_progress = 0

    source_index_lookup: dict[tuple[str, str], int] = {}
    source_debug: dict[tuple[str, str], dict[str, Any]] = {}
    saved_window_keys: set[tuple[str, str, int, int, int, int]] = set()
    for _split_name, _records in split_records.items():
        for _idx, _record in enumerate(_records):
            _image_id = str(_record.get("image_id", ""))
            source_index_lookup[(_split_name, _image_id)] = int(_idx)
            source_debug[(_split_name, _image_id)] = {
                "split": _split_name,
                "source_index": int(_idx),
                "source_image_id": _image_id,
                "source_study_id": str(_record.get("study_id", "")),
                "laterality": _record.get("laterality", ""),
                "view_position": _record.get("view_position", ""),
                "official_split": _record.get("split", ""),
                "processed_source_image": 0,
                "n_source_mass_boxes": 0,
                "has_source_mass": 0,
                "candidate_windows": 0,
                "attempted_save_windows": 0,
                "saved_crops": 0,
                "saved_positive_crops": 0,
                "saved_negative_crops": 0,
                "exported_mass_box_instances": 0,
                "skipped_windows": 0,
                "skip_foreground_too_low": 0,
                "skip_bbox_safe_failed": 0,
                "skip_duplicate_window": 0,
                "skip_empty_disallowed": 0,
                "_included_annotation_indices": set(),
            }

    def source_row_for(split_name_: str, record_: dict[str, Any]) -> dict[str, Any]:
        image_id_ = str(record_.get("image_id", ""))
        key_ = (split_name_, image_id_)
        if key_ not in source_debug:
            source_debug[key_] = {
                "split": split_name_,
                "source_index": int(source_index_lookup.get(key_, -1)),
                "source_image_id": image_id_,
                "source_study_id": str(record_.get("study_id", "")),
                "laterality": record_.get("laterality", ""),
                "view_position": record_.get("view_position", ""),
                "official_split": record_.get("split", ""),
                "processed_source_image": 0,
                "n_source_mass_boxes": 0,
                "has_source_mass": 0,
                "candidate_windows": 0,
                "attempted_save_windows": 0,
                "saved_crops": 0,
                "saved_positive_crops": 0,
                "saved_negative_crops": 0,
                "exported_mass_box_instances": 0,
                "skipped_windows": 0,
                "skip_foreground_too_low": 0,
                "skip_bbox_safe_failed": 0,
                "skip_duplicate_window": 0,
                "skip_empty_disallowed": 0,
                "_included_annotation_indices": set(),
            }
        return source_debug[key_]

    def note_source_target(split_name_: str, record_: dict[str, Any], target_: dict[str, Any]) -> dict[str, Any]:
        row_ = source_row_for(split_name_, record_)
        boxes_ = target_.get("mass", {}).get("boxes", torch.zeros((0, 4)))
        try:
            n_mass_ = int(boxes_.detach().cpu().reshape(-1, 4).shape[0])
        except Exception:
            n_mass_ = 0
        row_["processed_source_image"] = 1
        row_["n_source_mass_boxes"] = max(int(row_.get("n_source_mass_boxes", 0)), n_mass_)
        row_["has_source_mass"] = int(row_["n_source_mass_boxes"] > 0)
        return row_

    def mark_source_skip(split_name_: str, record_: dict[str, Any], reason: str) -> None:
        row_ = source_row_for(split_name_, record_)
        row_["skipped_windows"] = int(row_.get("skipped_windows", 0)) + 1
        key_name = f"skip_{reason}"
        if key_name in row_:
            row_[key_name] = int(row_.get(key_name, 0)) + 1

    def get_preprocessed(record_: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        key = str(record_.get("image_id", ""))
        if key in preprocessed_cache:
            preprocessed_cache.move_to_end(key)
            return preprocessed_cache[key]
        value = dataset._read_preprocessed_record_no_square(record_)
        preprocessed_cache[key] = value
        preprocessed_cache.move_to_end(key)
        while len(preprocessed_cache) > max(1, max_preprocessed_cache_items):
            preprocessed_cache.popitem(last=False)
        return value

    for split_name in ["train", "val", "test"]:
        records = split_records.get(split_name, [])
        iterator = _progress(records, show_progress, f"Export baseline {split_name}", unit="img")
        for record in iterator:
            image, target = dataset._read_preprocessed_record_no_square(record)
            image, target, baseline_resize_info = _resize_baseline_image_and_target(image, target, baseline_cfg)
            height, width = int(image.shape[-2]), int(image.shape[-1])
            filename = _make_baseline_filename(record)
            rel_img_path = Path("images") / split_name / filename
            save_info = _save_export_images(
                image,
                base_root,
                rel_img_path,
                config,
                float32_variant="baseline_whole",
            )
            save_info.update(baseline_resize_info)

            boxes = target["mass"]["boxes"]
            labels_path = base_root / "labels" / split_name / f"{Path(filename).stem}.txt"
            _write_yolo_label_file(labels_path, boxes, width=width, height=height, save_empty=save_empty_labels)

            image_meta = _coco_image_record(
                image_id_counter,
                filename,
                width,
                height,
                record,
                save_info,
                split_name,
                dataset_name="baseline_uncropped",
                crop_info=None,
                preprocess_info=target.get("preprocessing", {}),
            )
            coco = coco_by_split[split_name]
            coco["images"].append(image_meta)
            ann_rows = _append_coco_annotations(coco, image_id_counter, ann_id_counter, boxes)
            ann_id_counter += ann_rows
            stats_rows.append(
                _sample_stats_row(
                    dataset_name="baseline_uncropped",
                    split=split_name,
                    filename=filename,
                    image_width=width,
                    image_height=height,
                    boxes=boxes,
                    record=record,
                    crop_info=None,
                    save_info=save_info,
                )
            )
            metadata_rows.append(
                _sample_metadata_record(
                    dataset_name="baseline_uncropped",
                    split=split_name,
                    filename=filename,
                    target=target,
                    record=record,
                    boxes=boxes,
                    save_info=save_info,
                    crop_info=None,
                )
            )
            image_id_counter += 1

    created = _write_shared_export_files(base_root, coco_by_split, stats_rows, metadata_rows, dataset_kind="baseline_uncropped")
    return _summary_from_stats(stats_rows), created


def _resize_baseline_image_and_target(
    image: torch.Tensor,
    target: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    mode = str(cfg.get("resize_mode", "none") or "none").strip().casefold()
    if mode in {"", "none", "native", "original"}:
        return image, target, {"baseline_resize_mode": "none"}

    target_w = max(1, int(cfg.get("target_width", cfg.get("size", 640)) or 640))
    target_h = max(1, int(cfg.get("target_height", cfg.get("size", 640)) or 640))
    pad_value = float(cfg.get("pad_value", 0.0))
    pad_anchor = str(cfg.get("pad_anchor", "left_top") or "left_top").strip().casefold()
    src_h, src_w = int(image.shape[-2]), int(image.shape[-1])
    boxes = target.get("mass", {}).get("boxes", torch.zeros((0, 4), dtype=torch.float32))
    boxes = boxes.detach().clone().float().reshape(-1, 4)

    def _resize_to(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(tensor.unsqueeze(0).float(), size=(int(h), int(w)), mode="bilinear", align_corners=False).squeeze(0)

    if mode == "stretch":
        out = _resize_to(image, target_h, target_w)
        if boxes.numel():
            boxes[:, [0, 2]] *= float(target_w) / max(float(src_w), 1.0)
            boxes[:, [1, 3]] *= float(target_h) / max(float(src_h), 1.0)
        meta = {"baseline_resize_mode": "stretch", "baseline_resize_target_width": target_w, "baseline_resize_target_height": target_h}
        return out, _target_with_boxes(target, boxes, target_w, target_h), meta

    scale = max(float(target_w) / max(float(src_w), 1.0), float(target_h) / max(float(src_h), 1.0)) if mode == "fill_crop" else min(float(target_w) / max(float(src_w), 1.0), float(target_h) / max(float(src_h), 1.0))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = _resize_to(image, resized_h, resized_w)
    if boxes.numel():
        boxes *= float(scale)

    if mode == "fill_crop":
        left = max(0, (resized_w - target_w) // 2)
        top = max(0, (resized_h - target_h) // 2)
        out = resized[..., top:top + target_h, left:left + target_w]
        if out.shape[-2] != target_h or out.shape[-1] != target_w:
            padded = torch.full((*out.shape[:-2], target_h, target_w), pad_value, dtype=out.dtype, device=out.device)
            padded[..., : out.shape[-2], : out.shape[-1]] = out
            out = padded
        if boxes.numel():
            boxes[:, [0, 2]] -= float(left)
            boxes[:, [1, 3]] -= float(top)
    else:
        mode = "fit_pad"
        extra_x = max(0, target_w - resized_w)
        extra_y = max(0, target_h - resized_h)
        if pad_anchor in {"center", "centre"}:
            left = extra_x // 2
            top = extra_y // 2
        elif pad_anchor in {"left_center", "left-centre", "left_centered"}:
            left = 0
            top = extra_y // 2
        elif pad_anchor in {"right_center", "right-centre", "right_centered"}:
            left = extra_x
            top = extra_y // 2
        else:
            left = 0
            top = 0
        out = torch.full((*resized.shape[:-2], target_h, target_w), pad_value, dtype=resized.dtype, device=resized.device)
        out[..., top:top + resized_h, left:left + resized_w] = resized
        if boxes.numel():
            boxes[:, [0, 2]] += float(left)
            boxes[:, [1, 3]] += float(top)

    meta = {
        "baseline_resize_mode": mode,
        "baseline_resize_target_width": target_w,
        "baseline_resize_target_height": target_h,
        "baseline_resize_scale": float(scale),
        "baseline_resize_pad_left": int(left),
        "baseline_resize_pad_top": int(top),
        "baseline_resize_pad_value": pad_value,
        "baseline_resize_pad_anchor": pad_anchor,
    }
    return out, _target_with_boxes(target, boxes, target_w, target_h), meta


def _target_with_boxes(target: dict[str, Any], boxes: torch.Tensor, width: int, height: int) -> dict[str, Any]:
    out = dict(target)
    mass = dict(out.get("mass", {}) or {})
    mass["boxes"] = _clip_boxes_xyxy(boxes, width=width, height=height)
    out["mass"] = mass
    return out


def _clip_boxes_xyxy(boxes: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.reshape(-1, 4).clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(width))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(height))
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep]


def _paired_original_enabled(cfg: dict[str, Any]) -> bool:
    """Whether to save the fixed-preprocessed whole at its unpadded size."""
    return bool(cfg.get("save_original", False))


def _paired_resized_enabled(cfg: dict[str, Any]) -> bool:
    """Whether to save the compact square-padded and resized whole."""
    if "save_resized" in cfg:
        return bool(cfg.get("save_resized"))
    # Backward compatibility: paired whole export historically always wrote it.
    return bool(cfg.get("enabled", False))


def _transform_whole_annotation_records(
    source_boxes: torch.Tensor | None,
    *,
    pad_left: float,
    pad_top: float,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
    source_annotation_ids: list[Any] | None = None,
    source_annotation_rows: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Transform fixed-preprocessed Mass boxes into one whole-image variant."""
    ids = list(source_annotation_ids or [])
    rows = list(source_annotation_rows or [])
    records: list[dict[str, Any]] = []
    for index, source_box in enumerate(_boxes_to_list(source_boxes)):
        sx0, sy0, sx1, sy1 = [float(value) for value in source_box]
        x0 = min(max((sx0 + pad_left) * scale_x, 0.0), float(width))
        y0 = min(max((sy0 + pad_top) * scale_y, 0.0), float(height))
        x1 = min(max((sx1 + pad_left) * scale_x, 0.0), float(width))
        y1 = min(max((sy1 + pad_top) * scale_y, 0.0), float(height))
        if x1 <= x0 or y1 <= y0:
            continue
        annotation = {
            "category_id": 1,
            "category_name": "mass",
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            "source_bbox_xyxy": [sx0, sy0, sx1, sy1],
            "source_bbox_coordinate_space": "fixed_preprocessed",
            "transform": {
                "pad_left": float(pad_left),
                "pad_top": float(pad_top),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
            },
        }
        if index < len(ids) and ids[index] is not None:
            annotation["source_annotation_id"] = ids[index]
        if index < len(rows) and rows[index] is not None:
            annotation["source_annotation_row"] = rows[index]
        records.append(annotation)
    return records


def _write_whole_variant_annotation_files(
    *,
    crop_root: Path,
    config: dict[str, Any],
    split_name: str,
    whole_filename: str,
    source_image_id: str,
    source_study_id: str,
    variant: str,
    resolution: str | None = None,
    image_rel_path: Path,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
    save_empty_labels: bool,
) -> dict[str, Any]:
    """Write matched per-image YOLO and rich JSON annotations for a whole."""
    stem = Path(whole_filename).stem
    variant_id = _whole_variant_id(variant, resolution)
    if uses_grouped_dataset_layout(config):
        annotation_base = Path("annotations") / variant
        if resolution:
            annotation_base /= resolution
        label_rel_path = annotation_base / "yolo" / split_name / f"{stem}.txt"
        annotation_rel_path = annotation_base / "json" / split_name / f"{stem}.json"
    else:
        legacy_suffix = variant if not resolution else variant_id
        label_rel_path = Path(f"whole_labels_{legacy_suffix}") / split_name / f"{stem}.txt"
        annotation_rel_path = (
            Path(f"whole_annotations_{legacy_suffix}") / split_name / f"{stem}.json"
        )
    box_values = [annotation["bbox_xyxy"] for annotation in annotations]
    boxes = (
        torch.tensor(box_values, dtype=torch.float32).reshape(-1, 4)
        if box_values
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    label_path = crop_root / label_rel_path
    if not label_path.exists():
        _write_yolo_label_file(
            label_path,
            boxes,
            width=width,
            height=height,
            save_empty=save_empty_labels,
        )
    annotation_path = crop_root / annotation_rel_path
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if not annotation_path.exists():
        annotation_path.write_text(
            json.dumps(
                _json_safe({
                    "schema_version": 1,
                    "variant": variant_id,
                    "variant_kind": variant,
                    "resolution": resolution or "",
                    "coordinate_space": f"whole_{variant_id}",
                    "image_path": _path_as_posix(image_rel_path),
                    "label_path": _path_as_posix(label_rel_path),
                    "source_image_id": str(source_image_id),
                    "source_study_id": str(source_study_id),
                    "width": int(width),
                    "height": int(height),
                    "annotations": annotations,
                }),
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "label_path": _path_as_posix(label_rel_path),
        "annotation_path": _path_as_posix(annotation_rel_path),
        "annotations": annotations,
        "num_annotations": len(annotations),
    }


def _whole_variant_id(variant: str, resolution: str | None = None) -> str:
    return f"{variant}_{resolution}" if resolution else str(variant)


def _whole_image_relative_path(
    config: dict[str, Any],
    *,
    variant: str,
    split_name: str,
    filename: str,
    resolution: str | None = None,
) -> Path:
    if uses_grouped_dataset_layout(config):
        path = Path("images") / variant
        if resolution:
            path /= resolution
        return path / split_name / filename
    if variant == "original":
        directory = "whole_images_original"
    elif variant == "resized":
        directory = "whole_images" if not resolution else f"whole_images_{resolution}"
    elif variant == "high_resolution":
        directory = "whole_images_high_resolution"
    else:
        directory = f"whole_images_{variant}"
    return Path(directory) / split_name / filename


def _whole_float32_relative_path(
    config: dict[str, Any], image_rel_path: Path
) -> Path:
    if uses_grouped_dataset_layout(config) and image_rel_path.parts[:1] == ("images",):
        return Path("images") / "float32" / Path(*image_rel_path.parts[1:]).with_suffix(".pt")
    return Path("float32") / image_rel_path.with_suffix(".pt")


def _whole_coco_relative_path(
    config: dict[str, Any], variant_id: str, split_name: str
) -> Path:
    if uses_grouped_dataset_layout(config):
        if variant_id.startswith("resized_"):
            base = Path("annotations") / "resized" / variant_id.removeprefix("resized_")
        else:
            base = Path("annotations") / variant_id
        return base / "coco" / f"instances_{split_name}.json"
    return (
        Path("mmdetection")
        / f"whole_{variant_id}"
        / "annotations"
        / f"instances_{split_name}.json"
    )


# -----------------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------------


def _save_paired_whole_image_for_crop(
    *,
    source_image: torch.Tensor,
    crop_root: Path,
    split_name: str,
    filename: str,
    source_image_id: str,
    source_study_id: str = "",
    config: dict[str, Any],
    paired_cfg: dict[str, Any],
    source_path_cache: dict[tuple[str, ...], Path],
    source_boxes: torch.Tensor | None = None,
    source_annotation_ids: list[Any] | None = None,
    source_annotation_rows: list[Any] | None = None,
    source_foreground_mask: np.ndarray | None = None,
    whole_stage_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
    cache_namespace: str = "export",
) -> dict[str, Any]:
    """Write enabled original, resized, and high-resolution whole variants.

    Crop filenames contain a ``__crop__`` suffix.  Removing that suffix yields
    the canonical whole-image basename, so every crop from one source points to
    the same file.  ``source_path_cache`` prevents repeated encoding and writes;
    no hard links or copied aliases are created.

    The three output geometries are deliberately independent. The original
    variant keeps the fixed-preprocessed source dimensions without padding.
    The resized whole uses the legacy per-image square letterbox before
    resizing. The high-resolution whole can optionally use one fixed dataset
    canvas. Every enabled variant receives annotations in its own coordinates.
    """
    # Geometry belongs to every crop/whole pair, not only the first crop that
    # happens to encode the shared whole image.  Derive it independently of the
    # source-file cache so every JSONL row contains the same coordinate-mapping
    # metadata for a given source image.
    source_h, source_w = int(source_image.shape[-2]), int(source_image.shape[-1])
    whole_rgb: np.ndarray | None = None
    whole_float: np.ndarray | None = None
    save_original_float32 = float32_export_variant_enabled(
        config, "original_whole"
    )
    save_resized_float32 = float32_export_variant_enabled(
        config, "resized_whole"
    )
    save_high_resolution_float32 = float32_export_variant_enabled(
        config, "high_resolution_whole"
    )

    def encoded_whole_float() -> np.ndarray:
        nonlocal whole_float
        if whole_float is not None:
            return whole_float
        full_arr = _tensor_to_float2d(source_image)
        src_h, src_w = full_arr.shape
        whole_float, _encoding_meta = _make_rgb_image(
            full_arr,
            config,
            source_arrays={"current_crop": full_arr},
            full_source_arrays={"current_crop": full_arr},
            full_source_masks=(
                {"current_crop": np.asarray(source_foreground_mask, dtype=bool)}
                if source_foreground_mask is not None
                else None
            ),
            crop_window=(0, 0, int(src_w), int(src_h)),
            crop_pad_value=float(paired_cfg.get("pad_value", 0.0)),
            whole_stage_cache=whole_stage_cache,
            cache_namespace=f"{cache_namespace}:float32",
            return_float=True,
        )
        return whole_float

    def encoded_whole_rgb() -> np.ndarray:
        nonlocal whole_rgb
        if whole_rgb is not None:
            return whole_rgb
        full_arr = _tensor_to_float2d(source_image)
        src_h, src_w = full_arr.shape
        whole_rgb, _encoding_meta = _make_rgb_image(
            full_arr,
            config,
            source_arrays={"current_crop": full_arr},
            full_source_arrays={"current_crop": full_arr},
            full_source_masks=(
                {"current_crop": np.asarray(source_foreground_mask, dtype=bool)}
                if source_foreground_mask is not None
                else None
            ),
            crop_window=(0, 0, int(src_w), int(src_h)),
            crop_pad_value=float(paired_cfg.get("pad_value", 0.0)),
            whole_stage_cache=whole_stage_cache,
            cache_namespace=cache_namespace,
            return_float=False,
        )
        return whole_rgb

    def save_variant(
        *,
        rel_path: Path,
        variant: str,
        make_pixels: Callable[[], np.ndarray],
    ) -> str:
        out_path = crop_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source_key = (
            str(split_name),
            str(source_image_id),
            str(variant),
        )
        cached_path = source_path_cache.get(source_key)
        if cached_path is not None:
            if cached_path != out_path:
                raise RuntimeError(
                    "Paired whole-image cache resolved one source mammogram to "
                    f"two paths: {cached_path} and {out_path}."
                )
            if out_path.exists():
                return "reused"
        Image.fromarray(make_pixels(), mode="RGB").save(out_path)
        source_path_cache[source_key] = out_path
        return "written"

    def save_float_variant(
        *,
        rel_path: Path,
        variant: str,
        make_pixels: Callable[[], np.ndarray],
    ) -> tuple[str, str, dict[str, Any]]:
        float_rel_path = _whole_float32_relative_path(config, rel_path)
        out_path = crop_root / float_rel_path
        source_key = (str(split_name), str(source_image_id), f"{variant}_float32")
        pixels = make_pixels()
        tensor_meta = _float32_rgb_tensor_metadata(pixels)
        cached_path = source_path_cache.get(source_key)
        if cached_path is not None:
            if cached_path != out_path:
                raise RuntimeError(
                    "Float32 whole-image cache resolved one source mammogram to "
                    f"two paths: {cached_path} and {out_path}."
                )
            if out_path.exists():
                return _path_as_posix(float_rel_path), "reused", tensor_meta
        _save_float32_rgb_tensor(pixels, out_path)
        source_path_cache[source_key] = out_path
        return _path_as_posix(float_rel_path), "written", tensor_meta

    whole_filename = _whole_image_filename_from_crop_filename(
        filename,
        source_image_id=source_image_id,
    )
    whole_key = Path(whole_filename).stem
    out = {
        "paired_whole_key": whole_key,
        "paired_whole_filename": whole_filename,
        "paired_whole_source_image_id": str(source_image_id),
        "paired_whole_source_study_id": str(source_study_id),
    }
    save_empty_labels = bool(config.get("export", {}).get("save_empty_label_files", True))

    if _paired_original_enabled(paired_cfg):
        original_rel_path = _whole_image_relative_path(
            config,
            variant="original",
            split_name=split_name,
            filename=whole_filename,
        )
        original_write_status = save_variant(
            rel_path=original_rel_path,
            variant="original",
            make_pixels=encoded_whole_rgb,
        )
        original_float_path = ""
        original_float_status = "disabled"
        original_float_meta: dict[str, Any] = {}
        if save_original_float32:
            (
                original_float_path,
                original_float_status,
                original_float_meta,
            ) = save_float_variant(
                rel_path=original_rel_path,
                variant="original",
                make_pixels=encoded_whole_float,
            )
        original_annotations = _transform_whole_annotation_records(
            source_boxes,
            pad_left=0.0,
            pad_top=0.0,
            scale_x=1.0,
            scale_y=1.0,
            width=source_w,
            height=source_h,
            source_annotation_ids=source_annotation_ids,
            source_annotation_rows=source_annotation_rows,
        )
        original_annotation_info = _write_whole_variant_annotation_files(
            crop_root=crop_root,
            config=config,
            split_name=split_name,
            whole_filename=whole_filename,
            source_image_id=source_image_id,
            source_study_id=source_study_id,
            variant="original",
            image_rel_path=original_rel_path,
            width=source_w,
            height=source_h,
            annotations=original_annotations,
            save_empty_labels=save_empty_labels,
        )
        out.update({
            "paired_whole_original_image_path": _path_as_posix(original_rel_path),
            "paired_whole_original_width": source_w,
            "paired_whole_original_height": source_h,
            "paired_whole_original_storage": "single_file_per_source",
            "paired_whole_original_write_status": original_write_status,
            "paired_whole_original_float32_image_path": original_float_path,
            "paired_whole_original_float32_write_status": original_float_status,
            "paired_whole_original_float32_dtype": original_float_meta.get("dtype", ""),
            "paired_whole_original_float32_layout": original_float_meta.get("layout", ""),
            "paired_whole_original_float32_shape": original_float_meta.get("shape", []),
            "paired_whole_original_float32_min": original_float_meta.get("min", ""),
            "paired_whole_original_float32_max": original_float_meta.get("max", ""),
            "paired_whole_original_float32_finite": original_float_meta.get("finite", ""),
            "paired_whole_original_float32_contiguous": original_float_meta.get("contiguous", ""),
            "paired_whole_original_unpadded": True,
            "paired_whole_original_canvas_mode": "none",
            "paired_whole_original_canvas_width": source_w,
            "paired_whole_original_canvas_height": source_h,
            "paired_whole_original_pad_left": 0,
            "paired_whole_original_pad_top": 0,
            "paired_whole_original_pad_right": 0,
            "paired_whole_original_pad_bottom": 0,
            "paired_whole_original_scale_x": 1.0,
            "paired_whole_original_scale_y": 1.0,
            "paired_whole_original_scale_factor": 1.0,
            "paired_whole_original_label_path": original_annotation_info["label_path"],
            "paired_whole_original_annotation_path": original_annotation_info["annotation_path"],
            "paired_whole_original_annotations": original_annotations,
            "paired_whole_original_num_annotations": len(original_annotations),
        })

    if _paired_resized_enabled(paired_cfg):
        resized_outputs: dict[str, dict[str, Any]] = {}
        explicit_resized_variants = "resized_variants" in paired_cfg
        for variant_index, resized_variant in enumerate(
            resized_variant_configs(paired_cfg)
        ):
            resolution = str(resized_variant["name"])
            variant_id = _whole_variant_id(
                "resized", resolution if explicit_resized_variants else None
            )
            resized_cfg = _paired_resized_geometry_config({
                **paired_cfg,
                **resized_variant,
                "target_width": int(resized_variant["width"]),
                "target_height": int(resized_variant["height"]),
            })
            resize_meta = _paired_whole_geometry_metadata(
                source_h, source_w, resized_cfg
            )
            resized_rel_path = _whole_image_relative_path(
                config,
                variant="resized",
                resolution=resolution if explicit_resized_variants else None,
                split_name=split_name,
                filename=whole_filename,
            )
            resized_write_status = save_variant(
                rel_path=resized_rel_path,
                variant=variant_id,
                make_pixels=lambda current_cfg=resized_cfg: _pad_then_resize_rgb(
                    encoded_whole_rgb(), current_cfg
                )[0],
            )
            resized_float_path = ""
            resized_float_status = "disabled"
            resized_float_meta: dict[str, Any] = {}
            if save_resized_float32 and bool(resized_variant.get("save_float32", True)):
                (
                    resized_float_path,
                    resized_float_status,
                    resized_float_meta,
                ) = save_float_variant(
                    rel_path=resized_rel_path,
                    variant=variant_id,
                    make_pixels=lambda current_cfg=resized_cfg: _pad_then_resize_float_rgb(
                        encoded_whole_float(), current_cfg
                    )[0],
                )
            target_w = int(resized_variant["width"])
            target_h = int(resized_variant["height"])
            resized_annotations = _transform_whole_annotation_records(
                source_boxes,
                pad_left=float(resize_meta["paired_whole_pad_left"]),
                pad_top=float(resize_meta["paired_whole_pad_top"]),
                scale_x=float(resize_meta["paired_whole_scale_x"]),
                scale_y=float(resize_meta["paired_whole_scale_y"]),
                width=target_w,
                height=target_h,
                source_annotation_ids=source_annotation_ids,
                source_annotation_rows=source_annotation_rows,
            )
            resized_annotation_info = _write_whole_variant_annotation_files(
                crop_root=crop_root,
                config=config,
                split_name=split_name,
                whole_filename=whole_filename,
                source_image_id=source_image_id,
                source_study_id=source_study_id,
                variant="resized",
                resolution=resolution if explicit_resized_variants else None,
                image_rel_path=resized_rel_path,
                width=target_w,
                height=target_h,
                annotations=resized_annotations,
                save_empty_labels=save_empty_labels,
            )
            variant_output = {
                "variant": variant_id,
                "variant_kind": "resized",
                "resolution": resolution,
                "image_path": _path_as_posix(resized_rel_path),
                "width": target_w,
                "height": target_h,
                "storage": "single_file_per_source",
                "write_status": resized_write_status,
                "float32_image_path": resized_float_path,
                "float32_write_status": resized_float_status,
                "float32_dtype": resized_float_meta.get("dtype", ""),
                "float32_layout": resized_float_meta.get("layout", ""),
                "float32_shape": resized_float_meta.get("shape", []),
                "float32_min": resized_float_meta.get("min", ""),
                "float32_max": resized_float_meta.get("max", ""),
                "float32_finite": resized_float_meta.get("finite", ""),
                "float32_contiguous": resized_float_meta.get("contiguous", ""),
                "label_path": resized_annotation_info["label_path"],
                "annotation_path": resized_annotation_info["annotation_path"],
                "annotations": resized_annotations,
                "num_annotations": len(resized_annotations),
                "pad_then_resize": True,
                "geometry": resize_meta,
            }
            resized_outputs[resolution] = variant_output

            # Preserve the established single-resized metadata fields for the
            # first configured resolution. New consumers should use the mapping
            # above, while old consumers continue to see a deterministic primary.
            if variant_index == 0:
                out.update({
                    "paired_whole_image_path": variant_output["image_path"],
                    "paired_whole_width": target_w,
                    "paired_whole_height": target_h,
                    "paired_whole_storage": variant_output["storage"],
                    "paired_whole_write_status": resized_write_status,
                    "paired_whole_float32_image_path": resized_float_path,
                    "paired_whole_float32_write_status": resized_float_status,
                    "paired_whole_float32_dtype": variant_output["float32_dtype"],
                    "paired_whole_float32_layout": variant_output["float32_layout"],
                    "paired_whole_float32_shape": variant_output["float32_shape"],
                    "paired_whole_float32_min": variant_output["float32_min"],
                    "paired_whole_float32_max": variant_output["float32_max"],
                    "paired_whole_float32_finite": variant_output["float32_finite"],
                    "paired_whole_float32_contiguous": variant_output["float32_contiguous"],
                    "paired_whole_pad_then_resize": True,
                    "paired_whole_label_path": variant_output["label_path"],
                    "paired_whole_annotation_path": variant_output["annotation_path"],
                    "paired_whole_annotations": resized_annotations,
                    "paired_whole_num_annotations": len(resized_annotations),
                    **resize_meta,
                })
        if explicit_resized_variants:
            out["paired_whole_resized_variants"] = resized_outputs

    if _paired_high_resolution_enabled(paired_cfg):
        high_cfg = _paired_high_resolution_geometry_config(paired_cfg)
        high_meta_raw = _paired_whole_geometry_metadata(source_h, source_w, high_cfg)
        high_meta = {
            key.replace("paired_whole_", "paired_whole_high_resolution_", 1): value
            for key, value in high_meta_raw.items()
        }
        high_meta.update({
            "paired_whole_high_resolution_scale_x": 1.0,
            "paired_whole_high_resolution_scale_y": 1.0,
            "paired_whole_high_resolution_scale_factor": 1.0,
        })
        high_rel_path = _whole_image_relative_path(
            config,
            variant="high_resolution",
            split_name=split_name,
            filename=whole_filename,
        )
        high_write_status = save_variant(
            rel_path=high_rel_path,
            variant="high_resolution",
            make_pixels=lambda: _pad_rgb_to_canvas(encoded_whole_rgb(), high_cfg)[0],
        )
        high_float_path = ""
        high_float_status = "disabled"
        high_float_meta: dict[str, Any] = {}
        if save_high_resolution_float32:
            (
                high_float_path,
                high_float_status,
                high_float_meta,
            ) = save_float_variant(
                rel_path=high_rel_path,
                variant="high_resolution",
                make_pixels=lambda: _pad_float_rgb_to_canvas(
                    encoded_whole_float(), high_cfg
                )[0],
            )
        high_w = int(high_meta_raw["paired_whole_canvas_width"])
        high_h = int(high_meta_raw["paired_whole_canvas_height"])
        high_annotations = _transform_whole_annotation_records(
            source_boxes,
            pad_left=float(high_meta_raw["paired_whole_pad_left"]),
            pad_top=float(high_meta_raw["paired_whole_pad_top"]),
            scale_x=1.0,
            scale_y=1.0,
            width=high_w,
            height=high_h,
            source_annotation_ids=source_annotation_ids,
            source_annotation_rows=source_annotation_rows,
        )
        high_annotation_info = _write_whole_variant_annotation_files(
            crop_root=crop_root,
            config=config,
            split_name=split_name,
            whole_filename=whole_filename,
            source_image_id=source_image_id,
            source_study_id=source_study_id,
            variant="high_resolution",
            image_rel_path=high_rel_path,
            width=high_w,
            height=high_h,
            annotations=high_annotations,
            save_empty_labels=save_empty_labels,
        )
        out.update({
            "paired_whole_high_resolution_image_path": _path_as_posix(high_rel_path),
            "paired_whole_high_resolution_width": high_w,
            "paired_whole_high_resolution_height": high_h,
            "paired_whole_high_resolution_storage": "single_file_per_source",
            "paired_whole_high_resolution_write_status": high_write_status,
            "paired_whole_high_resolution_float32_image_path": high_float_path,
            "paired_whole_high_resolution_float32_write_status": high_float_status,
            "paired_whole_high_resolution_float32_dtype": high_float_meta.get("dtype", ""),
            "paired_whole_high_resolution_float32_layout": high_float_meta.get("layout", ""),
            "paired_whole_high_resolution_float32_shape": high_float_meta.get("shape", []),
            "paired_whole_high_resolution_float32_min": high_float_meta.get("min", ""),
            "paired_whole_high_resolution_float32_max": high_float_meta.get("max", ""),
            "paired_whole_high_resolution_float32_finite": high_float_meta.get("finite", ""),
            "paired_whole_high_resolution_float32_contiguous": high_float_meta.get("contiguous", ""),
            "paired_whole_high_resolution_padded_without_resize": True,
            "paired_whole_high_resolution_label_path": high_annotation_info["label_path"],
            "paired_whole_high_resolution_annotation_path": high_annotation_info["annotation_path"],
            "paired_whole_high_resolution_annotations": high_annotations,
            "paired_whole_high_resolution_num_annotations": len(high_annotations),
            **high_meta,
            # Backward-compatible metadata aliases. New exports use the
            # high-resolution terminology and directory above.
            "paired_whole_native_image_path": _path_as_posix(high_rel_path),
            "paired_whole_native_width": high_w,
            "paired_whole_native_height": high_h,
            "paired_whole_native_storage": "single_file_per_source",
            "paired_whole_native_write_status": high_write_status,
            "paired_whole_native_padded_without_resize": True,
            "paired_whole_native_scale_x": 1.0,
            "paired_whole_native_scale_y": 1.0,
        })
    return out


def _paired_high_resolution_enabled(cfg: dict[str, Any]) -> bool:
    """Return the canonical high-resolution flag with legacy fallback."""
    if "save_high_resolution" in cfg:
        return bool(cfg.get("save_high_resolution"))
    return bool(cfg.get("save_native_resolution", False))


def _paired_resized_geometry_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Geometry for the compact whole: per-image square, then resize."""
    out = dict(cfg)
    out["canvas_mode"] = str(
        cfg.get("resized_canvas_mode", "per_image_square") or "per_image_square"
    )
    if "resized_canvas_width" in cfg:
        out["canvas_width"] = cfg.get("resized_canvas_width")
    if "resized_canvas_height" in cfg:
        out["canvas_height"] = cfg.get("resized_canvas_height")
    return out


def _paired_high_resolution_geometry_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Geometry for the unresized high-resolution companion."""
    out = dict(cfg)
    out["canvas_mode"] = str(
        cfg.get("high_resolution_canvas_mode", cfg.get("canvas_mode", "per_image_square"))
        or "per_image_square"
    )
    out["canvas_width"] = cfg.get(
        "high_resolution_canvas_width", cfg.get("canvas_width")
    )
    out["canvas_height"] = cfg.get(
        "high_resolution_canvas_height", cfg.get("canvas_height")
    )
    return out


def _paired_whole_geometry_metadata(
    source_height: int,
    source_width: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the deterministic pad geometry used before whole-image resize."""
    src_h = max(1, int(source_height))
    src_w = max(1, int(source_width))
    canvas_mode = str(cfg.get("canvas_mode", "per_image_square") or "per_image_square").casefold().strip()
    if canvas_mode in {"fixed", "fixed_canvas", "dataset_fixed"}:
        canvas_mode = "fixed"
        canvas_w = int(cfg.get("canvas_width", 0) or 0)
        canvas_h = int(cfg.get("canvas_height", 0) or 0)
        if canvas_w <= 0 or canvas_h <= 0:
            raise ValueError(
                "paired_whole_images fixed canvas mode requires positive "
                "canvas_width and canvas_height."
            )
        size_divisor = max(1, int(cfg.get("size_divisor", 1) or 1))
        if canvas_w % size_divisor != 0 or canvas_h % size_divisor != 0:
            raise ValueError(
                "paired_whole_images fixed canvas dimensions must both be "
                f"divisible by size_divisor={size_divisor}; got "
                f"{canvas_w}x{canvas_h}."
            )
        if src_w > canvas_w or src_h > canvas_h:
            raise ValueError(
                "Fixed paired-whole canvas is smaller than a preprocessed "
                f"mammogram: source={src_w}x{src_h}, "
                f"canvas={canvas_w}x{canvas_h}. Increase the canvas instead "
                "of silently changing the dataset shape."
            )
    else:
        canvas_mode = "per_image_square"
        side = max(src_h, src_w)
        canvas_w = canvas_h = side

    anchor = str(cfg.get("pad_anchor", "left_top") or "left_top").casefold().strip()
    extra_x = max(0, canvas_w - src_w)
    extra_y = max(0, canvas_h - src_h)
    if anchor in {"center", "centre"}:
        anchor = "center"
        left, top = extra_x // 2, extra_y // 2
    elif anchor in {"left_center", "left-centre", "left_centered"}:
        anchor = "left_center"
        left, top = 0, extra_y // 2
    elif anchor in {"right_center", "right-centre", "right_centered"}:
        anchor = "right_center"
        left, top = extra_x, extra_y // 2
    else:
        anchor = "left_top"
        left, top = 0, 0

    target_w = max(1, int(cfg.get("target_width", cfg.get("size", 1024)) or 1024))
    target_h = max(1, int(cfg.get("target_height", cfg.get("size", 1024)) or 1024))
    size_divisor = max(1, int(cfg.get("size_divisor", 1) or 1))
    if target_w % size_divisor != 0 or target_h % size_divisor != 0:
        raise ValueError(
            "paired_whole_images target dimensions must both be divisible by "
            f"size_divisor={size_divisor}; got {target_w}x{target_h}."
        )
    scale_x = float(target_w / canvas_w)
    scale_y = float(target_h / canvas_h)
    return {
        "paired_whole_canvas_mode": canvas_mode,
        "paired_whole_canvas_width": int(canvas_w),
        "paired_whole_canvas_height": int(canvas_h),
        "paired_whole_pad_left": int(left),
        "paired_whole_pad_top": int(top),
        "paired_whole_pad_right": int(max(0, canvas_w - src_w - left)),
        "paired_whole_pad_bottom": int(max(0, canvas_h - src_h - top)),
        "paired_whole_pad_value": float(cfg.get("pad_value", 0.0)),
        "paired_whole_pad_anchor": anchor,
        "paired_whole_size_divisor": int(size_divisor),
        "paired_whole_common_canvas": bool(canvas_mode == "fixed"),
        "paired_whole_source_width": int(src_w),
        "paired_whole_source_height": int(src_h),
        "paired_whole_scale_x": scale_x,
        "paired_whole_scale_y": scale_y,
        "paired_whole_scale_factor": scale_x if math.isclose(scale_x, scale_y) else None,
    }


def _pad_then_resize_rgb(rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Pad a whole RGB image to a common canvas first, then resize it."""
    canvas, geometry = _pad_rgb_to_canvas(rgb, cfg)
    canvas_h, canvas_w = int(canvas.shape[0]), int(canvas.shape[1])

    target_w = max(1, int(cfg.get("target_width", cfg.get("size", 1024)) or 1024))
    target_h = max(1, int(cfg.get("target_height", cfg.get("size", 1024)) or 1024))
    if canvas_w == target_w and canvas_h == target_h:
        out = canvas
    elif cv2 is not None:
        interpolation = cv2.INTER_AREA if target_w <= canvas_w and target_h <= canvas_h else cv2.INTER_LINEAR
        out = cv2.resize(canvas, (target_w, target_h), interpolation=interpolation).astype(np.uint8, copy=False)
    else:  # pragma: no cover - OpenCV is a declared project dependency
        resampling = getattr(Image, "Resampling", Image)
        out = np.asarray(Image.fromarray(canvas, mode="RGB").resize((target_w, target_h), resample=resampling.BILINEAR))
    return out, geometry


def _pad_then_resize_float_rgb(
    rgb: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Float32 equivalent of :func:`_pad_then_resize_rgb` without PNG quantization."""
    canvas, geometry = _pad_float_rgb_to_canvas(rgb, cfg)
    canvas_h, canvas_w = int(canvas.shape[0]), int(canvas.shape[1])
    target_w = max(1, int(cfg.get("target_width", cfg.get("size", 1024)) or 1024))
    target_h = max(1, int(cfg.get("target_height", cfg.get("size", 1024)) or 1024))
    if canvas_w == target_w and canvas_h == target_h:
        return canvas, geometry
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if target_w <= canvas_w and target_h <= canvas_h else cv2.INTER_LINEAR
        out = cv2.resize(canvas, (target_w, target_h), interpolation=interpolation)
    else:  # pragma: no cover - OpenCV is a declared project dependency
        tensor = torch.from_numpy(canvas).permute(2, 0, 1)[None]
        out = (
            torch.nn.functional.interpolate(
                tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )[0]
            .permute(1, 2, 0)
            .numpy()
        )
    return np.asarray(out, dtype=np.float32).clip(0.0, 1.0), geometry


def _pad_rgb_to_canvas(rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Pad an RGB whole image to its configured canvas without resizing it."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {arr.shape}.")
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    geometry = _paired_whole_geometry_metadata(src_h, src_w, cfg)
    canvas_w = int(geometry["paired_whole_canvas_width"])
    canvas_h = int(geometry["paired_whole_canvas_height"])
    left = int(geometry["paired_whole_pad_left"])
    top = int(geometry["paired_whole_pad_top"])
    pad_value = float(geometry["paired_whole_pad_value"])
    pad_u8 = int(round(np.clip(pad_value, 0.0, 1.0) * 255.0))
    canvas = np.full((canvas_h, canvas_w, 3), pad_u8, dtype=np.uint8)
    canvas[top:top + src_h, left:left + src_w] = arr

    return canvas, geometry


def _pad_float_rgb_to_canvas(
    rgb: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pad a normalized float32 RGB image without reducing precision."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {arr.shape}.")
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    geometry = _paired_whole_geometry_metadata(src_h, src_w, cfg)
    canvas_w = int(geometry["paired_whole_canvas_width"])
    canvas_h = int(geometry["paired_whole_canvas_height"])
    left = int(geometry["paired_whole_pad_left"])
    top = int(geometry["paired_whole_pad_top"])
    pad_value = float(geometry["paired_whole_pad_value"])
    canvas = np.full((canvas_h, canvas_w, 3), pad_value, dtype=np.float32)
    canvas[top : top + src_h, left : left + src_w] = arr
    return canvas.clip(0.0, 1.0), geometry


def _save_export_images(
    image: torch.Tensor,
    root: Path,
    rel_img_path: Path,
    config: dict[str, Any],
    *,
    source_arrays: dict[str, np.ndarray] | None = None,
    full_source_arrays: dict[str, np.ndarray] | None = None,
    full_source_masks: dict[str, np.ndarray] | None = None,
    source_windows: dict[str, tuple[int, int, int, int]] | None = None,
    crop_window: tuple[int, int, int, int] | None = None,
    crop_pad_value: float = 0.0,
    whole_stage_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
    cache_namespace: str = "export",
    reject_blank_output: bool = False,
    min_output_signal_fraction: float = 0.0,
    float32_variant: str | None = None,
) -> dict[str, Any]:
    """Save PNG plus optional non-quantized [0, 1] float32 and preserved uint16 data."""
    img_cfg = config.get("image_export", {})
    preserved_cfg = config.get("preserved_16bit", {})
    if float32_variant is None:
        float32_variant = (
            "baseline_whole" if root.name == "baseline_uncropped" else "crops"
        )
    save_float32 = float32_export_variant_enabled(config, float32_variant)

    train_path = root / rel_img_path
    train_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _tensor_to_float2d(image)
    rgb, rgb_meta = _make_rgb_image(
        arr,
        config,
        source_arrays=source_arrays,
        full_source_arrays=full_source_arrays,
        full_source_masks=full_source_masks,
        source_windows=source_windows,
        crop_window=crop_window,
        crop_pad_value=crop_pad_value,
        whole_stage_cache=whole_stage_cache,
        cache_namespace=cache_namespace,
        return_float=False,
    )
    rgb = np.asarray(rgb, dtype=np.uint8)
    if save_float32:
        rgb_float, _float_meta = _make_rgb_image(
            arr,
            config,
            source_arrays=source_arrays,
            full_source_arrays=full_source_arrays,
            full_source_masks=full_source_masks,
            source_windows=source_windows,
            crop_window=crop_window,
            crop_pad_value=crop_pad_value,
            whole_stage_cache=whole_stage_cache,
            cache_namespace=f"{cache_namespace}:float32",
            return_float=True,
        )
        rgb_float = np.asarray(rgb_float, dtype=np.float32)
    else:
        rgb_float = None
    output_signal_fraction = float(np.any(rgb != 0, axis=-1).mean())
    if bool(reject_blank_output) and output_signal_fraction < max(
        float(min_output_signal_fraction),
        np.finfo(np.float32).eps,
    ):
        # Do not leave a stale image behind when an output root is intentionally
        # reused. Labels/metadata are written only after this function returns.
        train_path.unlink(missing_ok=True)
        if save_float32:
            (root / "float32" / rel_img_path.with_suffix(".pt")).unlink(missing_ok=True)
        return {
            "image_path": _path_as_posix(train_path.relative_to(root)),
            "rgb_scheme": str(img_cfg.get("rgb_scheme", "multi_window")),
            "histogram_equalization_enabled": bool(config.get("histogram_equalization", {}).get("enabled", True)),
            "output_signal_fraction": output_signal_fraction,
            "output_rejected_blank": True,
            **rgb_meta,
        }
    Image.fromarray(rgb, mode="RGB").save(train_path)

    out: dict[str, Any] = {
        "image_path": _path_as_posix(train_path.relative_to(root)),
        "rgb_scheme": str(img_cfg.get("rgb_scheme", "multi_window")),
        "histogram_equalization_enabled": bool(config.get("histogram_equalization", {}).get("enabled", True)),
        "output_signal_fraction": output_signal_fraction,
        "output_rejected_blank": False,
        **rgb_meta,
    }

    if rgb_float is not None:
        float_rel = Path("float32") / rel_img_path.with_suffix(".pt")
        float_path = root / float_rel
        _save_float32_rgb_tensor(rgb_float, float_path)
        out.update(
            {
                "float32_image_path": _path_as_posix(float_rel),
                "float32_image_layout": "CHW",
                "float32_image_dtype": "torch.float32",
                "float32_image_range": [0.0, 1.0],
            }
        )

    if bool(preserved_cfg.get("save", True)):
        preserved_rel = Path("preserved_16bit") / rel_img_path.parent.name / rel_img_path.name
        preserved_path = root / preserved_rel
        preserved_path.parent.mkdir(parents=True, exist_ok=True)
        img16, pmeta = _make_uint16_preserved(arr, config)
        Image.fromarray(img16).save(preserved_path)
        out.update(
            {
                "preserved_16bit_path": _path_as_posix(preserved_path.relative_to(root)),
                **pmeta,
            }
        )
    return out


def _tensor_to_float2d(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().squeeze().numpy().astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel image after squeeze, got shape {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _make_rgb_image(
    arr: np.ndarray,
    config: dict[str, Any],
    *,
    source_arrays: dict[str, np.ndarray] | None = None,
    full_source_arrays: dict[str, np.ndarray] | None = None,
    full_source_masks: dict[str, np.ndarray] | None = None,
    source_windows: dict[str, tuple[int, int, int, int]] | None = None,
    crop_window: tuple[int, int, int, int] | None = None,
    crop_pad_value: float = 0.0,
    whole_stage_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
    cache_namespace: str = "export",
    return_float: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an RGB image according to ``image_export.rgb_scheme``.

    Recommended default: ``multi_window``. It creates three visually meaningful
    mammography contrast windows instead of duplicating one grayscale window.
    With ``return_float=True``, the returned HWC array is float32 in [0, 1]
    before the final PNG encoding quantization.
    """
    img_cfg = config.get("image_export", {})
    eq_cfg = config.get("histogram_equalization", {})
    scheme = str(img_cfg.get("rgb_scheme", "multi_window")).casefold().strip()
    mask = _foreground_mask(arr)
    meta: dict[str, Any] = {"window_mask_pixels": int(mask.sum())}

    if scheme == "grayscale_rgb":
        lo, hi = _safe_percentile(arr, img_cfg.get("single_window", [0.5, 99.5]), mask)
        ch = (
            _to_float_window(arr, lo, hi)
            if return_float
            else _to_uint8_window(arr, lo, hi)
        )
        channels = [ch, ch.copy(), ch.copy()]
        meta["rgb_windows"] = [[lo, hi]] * 3

    elif scheme in {"paper69_mammoclip_uint8", "mammoclip_uint8_replicated"}:
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 255.0:
            if return_float:
                ch = (np.clip(arr, 0.0, 255.0) / 255.0).astype(np.float32)
            else:
                ch = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            ch = (
                _to_float_window(arr, lo, hi)
                if return_float
                else _to_uint8_window(arr, lo, hi)
            )
        else:
            ch = np.zeros(
                arr.shape, dtype=np.float32 if return_float else np.uint8
            )
        channels = [ch, ch.copy(), ch.copy()]
        meta["paper69_mammoclip_uint8_replicated"] = True

    elif scheme == "equalized_rgb":
        lo, hi = _safe_percentile(arr, img_cfg.get("single_window", [0.5, 99.5]), mask)
        if return_float:
            ch = _apply_custom_channel_operation_float_preserving(
                _to_float_window(arr, lo, hi), "hist_equalize", {}, mask
            )
        else:
            ch = _equalize_uint8(_to_uint8_window(arr, lo, hi), mask=mask)
        channels = [ch, ch.copy(), ch.copy()]
        meta["rgb_windows"] = [[lo, hi]] * 3
        meta["forced_equalized_rgb"] = True

    elif scheme in {"intensity_equalized_gradient", "ieg", "normal_equalized_gradient"}:
        channels, ieg_meta = _make_intensity_equalized_gradient_rgb(
            arr, img_cfg, mask, return_float=return_float
        )
        meta.update(ieg_meta)

    elif scheme in {"raw_clahe_detail", "raw_replicated", "raw_clahe_masked_raw", "raw_clahe_tophat"}:
        channels, literature_meta = _make_literature_recipe_rgb(
            arr, img_cfg, mask, scheme, return_float=return_float
        )
        meta.update(literature_meta)

    elif scheme in {"custom_channel_pipeline", "gui_channel_pipeline"}:
        channels, custom_meta = _make_custom_channel_pipeline_rgb(
            arr,
            img_cfg,
            mask,
            source_arrays=source_arrays,
            full_source_arrays=full_source_arrays,
            full_source_masks=full_source_masks,
            source_windows=source_windows,
            crop_window=crop_window,
            crop_pad_value=crop_pad_value,
            whole_stage_cache=whole_stage_cache,
            cache_namespace=cache_namespace,
            return_float=return_float,
        )
        meta.update(custom_meta)

    elif scheme == "bitpack16":
        # Not recommended for CNN training. Kept only as an explicit experimental option.
        img16, pmeta = _make_uint16_preserved(arr, config)
        high = ((img16 >> 8) & 255).astype(np.uint8)
        low = (img16 & 255).astype(np.uint8)
        preview = _to_uint8_window(arr, pmeta["preserved_16bit_lo"], pmeta["preserved_16bit_hi"])
        channels = [high, low, preview]
        meta.update(pmeta)
        meta["bitpack_warning"] = "Experimental only: RGB channels encode uint16 bytes, not visual channels."

    elif scheme == "multi_window":
        windows = img_cfg.get("multi_window_percentiles", [[0.5, 99.5], [1.0, 99.0], [2.0, 98.0]])
        if len(windows) != 3:
            raise ValueError("image_export.multi_window_percentiles must contain exactly three [lo, hi] pairs.")
        channels = []
        resolved = []
        for win in windows:
            lo, hi = _safe_percentile(arr, win, mask)
            channels.append(
                _to_float_window(arr, lo, hi)
                if return_float
                else _to_uint8_window(arr, lo, hi)
            )
            resolved.append([lo, hi])
        meta["rgb_windows"] = resolved

    else:
        raise ValueError(
            "Unknown image_export.rgb_scheme. Expected one of: "
            "multi_window, grayscale_rgb, equalized_rgb, "
            "paper69_mammoclip_uint8, "
            "intensity_equalized_gradient, raw_clahe_detail, raw_replicated, "
            "raw_clahe_masked_raw, raw_clahe_tophat, custom_channel_pipeline, bitpack16."
        )

    if bool(eq_cfg.get("enabled", True)) and scheme not in {"equalized_rgb", "bitpack16", "intensity_equalized_gradient", "ieg", "normal_equalized_gradient", "raw_clahe_detail", "raw_replicated", "raw_clahe_masked_raw", "raw_clahe_tophat", "custom_channel_pipeline", "gui_channel_pipeline"}:
        apply_to = str(eq_cfg.get("apply_to", "all_channels")).casefold().strip()
        if apply_to == "all_channels":
            channels = [
                (
                    _apply_custom_channel_operation_float_preserving(
                        ch, "hist_equalize", {}, mask
                    )
                    if return_float
                    else _equalize_uint8(ch, mask=mask)
                )
                for ch in channels
            ]
        elif apply_to == "third_channel":
            channels[2] = (
                _apply_custom_channel_operation_float_preserving(
                    channels[2], "hist_equalize", {}, mask
                )
                if return_float
                else _equalize_uint8(channels[2], mask=mask)
            )
        elif apply_to in {"none", "false", "off"}:
            pass
        else:
            raise ValueError("histogram_equalization.apply_to must be all_channels, third_channel, or none.")
        meta["histogram_equalization_apply_to"] = apply_to

    if return_float:
        if scheme != "bitpack16":
            rgb = np.stack(channels, axis=-1).astype(np.float32, copy=False)
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        else:
            rgb = np.stack(channels, axis=-1).astype(np.float32, copy=False) / 255.0
    else:
        rgb = np.stack(channels, axis=-1).astype(np.uint8, copy=False)
    return rgb, meta


def _make_literature_recipe_rgb(
    arr: np.ndarray,
    img_cfg: dict[str, Any],
    mask: np.ndarray,
    scheme: str,
    *,
    return_float: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cfg = dict(img_cfg.get("literature_recipes", {}) or {})
    raw_window = cfg.get("raw_percentiles", img_cfg.get("single_window", [0.5, 99.5]))
    raw = _apply_custom_channel_operation(arr, "percentile_normalize", {"percentiles": raw_window}, mask)
    if mask is not None and mask.shape == raw.shape:
        raw = np.where(mask, raw, 0.0).astype(np.float32)
    apply_operation = (
        _apply_custom_channel_operation_float_preserving
        if return_float
        else _apply_custom_channel_operation
    )
    clahe = apply_operation(
        raw,
        "clahe",
        {
            "clip_limit": float(cfg.get("clahe_clip_limit", 2.0)),
            "tile_grid_size": int(cfg.get("clahe_tile_grid_size", 8)),
        },
        mask,
    )
    if mask is not None and mask.shape == clahe.shape:
        clahe = np.where(mask, clahe, 0.0).astype(np.float32)
    meta: dict[str, Any] = {
        "rgb_scheme": scheme,
        "literature_recipe": scheme,
        "literature_raw_percentiles": list(raw_window),
        "literature_clahe_clip_limit": float(cfg.get("clahe_clip_limit", 2.0)),
        "literature_clahe_tile_grid_size": int(cfg.get("clahe_tile_grid_size", 8)),
    }
    if scheme == "raw_replicated":
        channel = (
            _float_to_unit_custom(raw)
            if return_float
            else _float_to_uint8_custom(raw)
        )
        return [channel] * 3, {**meta, "rgb_channel_0": "raw", "rgb_channel_1": "raw", "rgb_channel_2": "raw"}
    if scheme == "raw_clahe_masked_raw":
        masked = raw.copy()
        if mask is not None and mask.shape == raw.shape:
            masked = np.where(mask, masked, 0.0).astype(np.float32)
        encode = _float_to_unit_custom if return_float else _float_to_uint8_custom
        return [encode(raw), encode(clahe), encode(masked)], {
            **meta,
            "rgb_channel_0": "raw",
            "rgb_channel_1": "clahe",
            "rgb_channel_2": "masked_raw",
        }
    if scheme == "raw_clahe_tophat":
        tophat = apply_operation(
            clahe if str(cfg.get("tophat_apply_to", "clahe")).casefold() == "clahe" else raw,
            "white_tophat",
            {
                "kernel_size": int(cfg.get("tophat_kernel_size", 9)),
                "kernel_shape": str(cfg.get("tophat_kernel_shape", "ellipse")),
                "percentiles": cfg.get("detail_rescale_percentiles", [1.0, 99.0]),
            },
            mask,
        )
        if mask is not None and mask.shape == tophat.shape:
            tophat = np.where(mask, tophat, 0.0).astype(np.float32)
        encode = _float_to_unit_custom if return_float else _float_to_uint8_custom
        return [encode(raw), encode(clahe), encode(tophat)], {
            **meta,
            "rgb_channel_0": "raw",
            "rgb_channel_1": "clahe",
            "rgb_channel_2": "white_tophat",
        }

    detail = apply_operation(
        clahe,
        "local_detail",
        {
            "sigma": float(cfg.get("detail_blur_sigma", 1.0)),
            "percentiles": cfg.get("detail_rescale_percentiles", [1.0, 99.0]),
        },
        mask,
    )
    if mask is not None and mask.shape == detail.shape:
        detail = np.where(mask, detail, 0.0).astype(np.float32)
    encode = _float_to_unit_custom if return_float else _float_to_uint8_custom
    return [encode(raw), encode(clahe), encode(detail)], {
        **meta,
        "rgb_channel_0": "raw",
        "rgb_channel_1": "clahe",
        "rgb_channel_2": "local_detail_residual",
    }


def _custom_channel_source(pipeline: dict[str, Any], channel: str) -> str:
    value = pipeline.get(channel, {})
    if isinstance(value, dict):
        return str(value.get("source", "current_crop"))
    return "current_crop"


def _custom_channel_steps(pipeline: dict[str, Any], channel: str) -> list[dict[str, Any]]:
    value = pipeline.get(channel, [])
    if isinstance(value, dict):
        return list(value.get("steps", []) or [])
    return list(value or [])


def _config_uses_contralateral_source(config: dict[str, Any]) -> bool:
    img_cfg = config.get("image_export", {}) or {}
    scheme = str(img_cfg.get("rgb_scheme", "multi_window")).casefold().strip()
    if scheme not in {"custom_channel_pipeline", "gui_channel_pipeline"}:
        return False
    pipeline = img_cfg.get("custom_channel_pipeline", {}) or {}
    return any(_custom_channel_source(pipeline, ch) == "contralateral_same_view_crop" for ch in ["R", "G", "B"])


def _make_custom_channel_pipeline_rgb(
    arr: np.ndarray,
    img_cfg: dict[str, Any],
    mask: np.ndarray,
    *,
    source_arrays: dict[str, np.ndarray] | None = None,
    full_source_arrays: dict[str, np.ndarray] | None = None,
    full_source_masks: dict[str, np.ndarray] | None = None,
    source_windows: dict[str, tuple[int, int, int, int]] | None = None,
    crop_window: tuple[int, int, int, int] | None = None,
    crop_pad_value: float = 0.0,
    whole_stage_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
    cache_namespace: str = "export",
    return_float: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Create RGB channels using a GUI-exported per-channel operation pipeline.

    The preprocessing inspector exports this structure under
    image_export.custom_channel_pipeline. Each channel starts from the same
    fixed-preprocessed grayscale crop and applies its own ordered operation list.
    """
    pipeline = img_cfg.get("custom_channel_pipeline", {}) or {}
    channels: list[np.ndarray] = []
    meta: dict[str, Any] = {
        "rgb_scheme": "custom_channel_pipeline",
        "custom_channel_pipeline": pipeline,
        "rgb_channel_0": "custom_R_pipeline",
        "rgb_channel_1": "custom_G_pipeline",
        "rgb_channel_2": "custom_B_pipeline",
    }
    source_arrays = dict(source_arrays or {})
    source_arrays.setdefault("current_crop", arr)
    full_source_arrays = dict(full_source_arrays or {})
    full_source_masks = dict(full_source_masks or {})
    source_windows = dict(source_windows or {})
    source_masks: dict[str, np.ndarray] = {}
    for key, value in source_arrays.items():
        if value is None:
            continue
        try:
            source_masks[key] = _foreground_mask(np.asarray(value, dtype=np.float32))
        except Exception:
            source_masks[key] = mask

    for channel in ["R", "G", "B"]:
        source_name = _custom_channel_source(pipeline, channel)
        if source_name not in source_arrays or source_arrays.get(source_name) is None:
            work = np.asarray(arr, dtype=np.float32).copy()
            source_used = "current_crop"
            source_fallback = True
        else:
            work = np.asarray(source_arrays[source_name], dtype=np.float32).copy()
            source_used = source_name
            source_fallback = False
        # Use a source-specific foreground mask for percentile/statistics operations.
        # This is important for contralateral channels and for full-image crops with
        # large black background.
        stat_mask = source_masks.get(source_used, mask)
        if stat_mask is not None and stat_mask.shape != work.shape:
            stat_mask = _foreground_mask(work)

        work, applied, scope_meta = apply_scoped_steps(
            work,
            _custom_channel_steps(pipeline, channel),
            apply_operation=lambda work_arr, op, params, work_mask: (
                _apply_custom_channel_operation_float_preserving(
                    work_arr, op, params, work_mask
                )
                if return_float
                else _apply_custom_channel_operation(
                    work_arr, op, params, work_mask
                )
            ),
            make_stat_mask=_foreground_mask,
            operation_preserves_background=_custom_operation_should_preserve_background,
            full_source=full_source_arrays.get(source_used),
            full_stat_mask=full_source_masks.get(source_used),
            window_xyxy=source_windows.get(source_used, crop_window),
            pad_value=float(crop_pad_value),
            whole_stage_cache=whole_stage_cache,
            cache_namespace=str(cache_namespace),
            source_name=source_used,
        )
        channels.append(
            _float_to_unit_custom(work)
            if return_float
            else _float_to_uint8_custom(work)
        )
        meta[f"custom_{channel}_source_requested"] = source_name
        meta[f"custom_{channel}_source_used"] = source_used
        meta[f"custom_{channel}_source_fallback"] = int(source_fallback)
        meta[f"custom_{channel}_steps"] = applied
        meta[f"custom_{channel}_scope"] = scope_meta
    return channels, meta


def _apply_custom_channel_operation(
    arr: np.ndarray,
    op: str,
    params: dict[str, Any],
    mask: np.ndarray,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if op == "percentile_normalize":
        lo, hi = _safe_percentile(arr, params.get("percentiles", [1.0, 99.0]), mask)
        return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    if op == "percentile_clip_only":
        lo, hi = _safe_percentile(arr, params.get("percentiles", [1.0, 99.0]), mask)
        return np.clip(arr, lo, hi).astype(np.float32)
    if op == "zscore_clip":
        pixels = arr[mask] if mask is not None and mask.any() else arr[np.isfinite(arr)]
        if pixels.size == 0:
            pixels = arr[np.isfinite(arr)]
        mean = float(np.mean(pixels)) if pixels.size else 0.0
        std = float(np.std(pixels)) if pixels.size else 1.0
        limit = max(float(params.get("z_limit", 3.0)), 1e-6)
        z = np.clip((arr - mean) / max(std, 1e-12), -limit, limit)
        return ((z + limit) / (2.0 * limit)).astype(np.float32)
    if op == "standardize_to_target":
        return _standardize_to_target_custom(arr, params, mask)
    if op == "aggressive_upper_percentile_normalize":
        lo, hi = _safe_percentile(arr, params.get("percentiles", [70.0, 100.0]), mask)
        return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    if op == "hist_equalize":
        return _equalize_uint8(_float_to_uint8_custom(arr), mask=mask, params=params).astype(np.float32) / 255.0
    if op == "clahe":
        img = _float_to_uint8_custom(arr)
        if cv2 is None:
            return _equalize_uint8(img, mask=mask, params=params).astype(np.float32) / 255.0
        clip_limit = float(params.get("clip_limit", 2.0))
        tile = int(params.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        return clahe.apply(img).astype(np.float32) / 255.0
    if op in {"mask_outside_breast", "artifact_cleanup"}:
        if mask is None or mask.shape != arr.shape:
            mask = _foreground_mask(arr)
        return np.where(mask, arr, float(params.get("outside_value", 0.0))).astype(np.float32)
    if op == "gaussian_blur":
        k = _odd_int_custom(params.get("ksize", 5))
        sigma = float(params.get("sigma", 1.0))
        if cv2 is not None:
            return cv2.GaussianBlur(arr.astype(np.float32), (k, k), sigmaX=sigma).astype(np.float32)
        if scipy_ndimage is not None:
            return scipy_ndimage.gaussian_filter(arr.astype(np.float32), sigma=max(sigma, 0.0)).astype(np.float32)
        return arr
    if op == "median_blur":
        k = _odd_int_custom(params.get("ksize", 3))
        if cv2 is None and scipy_ndimage is not None:
            return scipy_ndimage.median_filter(arr.astype(np.float32), size=k).astype(np.float32)
        if cv2 is None:
            return arr
        return cv2.medianBlur(_float_to_uint8_custom(arr), k).astype(np.float32) / 255.0
    if op == "bilateral_filter":
        if cv2 is None:
            sigma = float(params.get("sigma_space", 5.0))
            if scipy_ndimage is not None:
                return scipy_ndimage.gaussian_filter(arr.astype(np.float32), sigma=max(sigma / 3.0, 0.0)).astype(np.float32)
            return arr
        diameter = int(params.get("diameter", 5))
        sigma_color = float(params.get("sigma_color", 0.05))
        sigma_space = float(params.get("sigma_space", 5.0))
        return cv2.bilateralFilter(arr.astype(np.float32), diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
    if op == "wiener_filter":
        if scipy_wiener is None:
            return arr
        k = _odd_int_custom(params.get("ksize", 7))
        noise = params.get("noise", None)
        try:
            out = scipy_wiener(arr.astype(np.float32), mysize=(k, k), noise=None if noise in {None, ""} else float(noise))
        except Exception:
            out = scipy_wiener(arr.astype(np.float32), mysize=(k, k))
        return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if op == "local_detail":
        sigma = float(params.get("sigma", 1.0))
        if cv2 is not None:
            smooth = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma)
        elif scipy_ndimage is not None:
            smooth = scipy_ndimage.gaussian_filter(arr.astype(np.float32), sigma=max(sigma, 0.0))
        else:
            smooth = arr
        detail = arr.astype(np.float32) - smooth.astype(np.float32)
        lo, hi = _safe_percentile(detail, params.get("percentiles", [1.0, 99.0]), mask)
        return ((np.clip(detail, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    if op == "sharpen":
        if cv2 is None:
            return _apply_custom_channel_operation(arr, "unsharp_mask", {"amount": params.get("amount", 0.2), "sigma": 1.0}, mask)
        amount = float(params.get("amount", 1.0))
        kernel = np.array([[0, -1, 0], [-1, 4 + amount, -1], [0, -1, 0]], dtype=np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(arr.astype(np.float32), -1, kernel).astype(np.float32)
    if op == "unsharp_mask":
        amount = float(params.get("amount", 1.5))
        sigma = float(params.get("sigma", 2.0))
        if cv2 is not None:
            blurred = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma)
        elif scipy_ndimage is not None:
            blurred = scipy_ndimage.gaussian_filter(arr.astype(np.float32), sigma=max(sigma, 0.0))
        else:
            blurred = arr
        return (arr + amount * (arr - blurred)).astype(np.float32)
    if op == "sobel_gradient":
        ksize = _odd_int_custom(params.get("ksize", 3))
        if cv2 is not None:
            gx = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)
            mag = cv2.magnitude(gx, gy).astype(np.float32)
        else:
            gy, gx = np.gradient(arr.astype(np.float32))
            mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        lo, hi = _safe_percentile(mag, params.get("percentiles", [1.0, 99.0]), mask)
        return ((np.clip(mag, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    if op == "laplacian":
        if cv2 is not None:
            lap = np.abs(cv2.Laplacian(arr.astype(np.float32), cv2.CV_32F, ksize=_odd_int_custom(params.get("ksize", 3))))
        else:
            gy, gx = np.gradient(arr.astype(np.float32))
            gyy, _ = np.gradient(gy)
            _, gxx = np.gradient(gx)
            lap = np.abs(gxx + gyy).astype(np.float32)
        lo, hi = _safe_percentile(lap, params.get("percentiles", [1.0, 99.0]), mask)
        return ((np.clip(lap, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    if op in {"white_tophat", "tophat"}:
        return _morphology_contrast_custom(arr, params, mask, mode="white_tophat")
    if op == "blackhat":
        return _morphology_contrast_custom(arr, params, mask, mode="blackhat")
    if op == "morphological_open":
        return _morphology_basic_custom(arr, params, op_name="open")
    if op == "morphological_close":
        return _morphology_basic_custom(arr, params, op_name="close")
    if op == "pectoral_suppression":
        return _pectoral_suppression_custom(arr, params)
    if op == "gamma":
        gamma = max(float(params.get("gamma", 1.0)), 1e-6)
        return np.power(np.clip(_normalize_minmax_custom(arr, mask), 0.0, 1.0), gamma).astype(np.float32)
    if op == "log":
        gain = float(params.get("gain", 5.0))
        x = np.clip(_normalize_minmax_custom(arr, mask), 0.0, 1.0)
        return (np.log1p(gain * x) / np.log1p(gain)).astype(np.float32)
    if op == "invert":
        return 1.0 - _normalize_minmax_custom(arr, mask)
    return arr


def _apply_custom_channel_operation_float_preserving(
    arr: np.ndarray,
    op: str,
    params: dict[str, Any],
    mask: np.ndarray,
) -> np.ndarray:
    """Run the image operation without converting pixels to an integer type."""

    value = np.nan_to_num(
        np.asarray(arr, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if op == "clahe":
        return _equalize_adaptive_float32(
            _float_to_unit_custom(value),
            mask=mask,
            clip_limit=float(params.get("clip_limit", 2.0)),
            tile_grid_size=int(params.get("tile_grid_size", 8)),
        )
    if op == "hist_equalize":
        unit = _float_to_unit_custom(value)
        valid_mask = (
            np.asarray(mask, dtype=bool)
            if mask is not None and np.asarray(mask).shape == unit.shape
            else np.ones(unit.shape, dtype=bool)
        )
        levels, mapped = _exact_float_cdf_mapping(
            unit[valid_mask],
            clip_limit=None,
        )
        if levels.size <= 1:
            return unit
        equalized = np.interp(unit, levels, mapped).astype(np.float32)
        return np.where(valid_mask, equalized, unit).astype(np.float32)
    if op == "median_blur" and cv2 is not None:
        return cv2.medianBlur(
            value,
            _odd_int_custom(params.get("ksize", 3)),
        ).astype(np.float32)
    return _apply_custom_channel_operation(value, op, params, mask)


def _exact_float_cdf_mapping(
    values: np.ndarray,
    *,
    clip_limit: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a monotonic CDF from exact float values, without histogram bins."""

    pixels = np.asarray(values, dtype=np.float32).reshape(-1)
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
    levels, counts = np.unique(pixels, return_counts=True)
    if levels.size <= 1:
        return levels.astype(np.float32, copy=False), levels.astype(
            np.float32, copy=False
        )

    weights = counts.astype(np.float64)
    if clip_limit is not None:
        # Express the CLAHE limit relative to the average count per occupied
        # float level. Redistribute clipped mass while keeping every mapping
        # increment positive and therefore preserving intensity ordering.
        average_count = float(pixels.size) / float(levels.size)
        threshold = max(1.0, float(clip_limit) * average_count)
        clipped = np.minimum(weights, threshold)
        clipped += float(np.sum(weights - clipped)) / float(levels.size)
        weights = clipped

    cdf = np.cumsum(weights, dtype=np.float64)
    denominator = float(cdf[-1] - cdf[0])
    if denominator <= np.finfo(np.float64).eps:
        return levels.astype(np.float32, copy=False), levels.astype(
            np.float32, copy=False
        )
    mapped = np.clip((cdf - cdf[0]) / denominator, 0.0, 1.0)
    return (
        levels.astype(np.float32, copy=False),
        mapped.astype(np.float32, copy=False),
    )


def _equalize_adaptive_float32(
    image: np.ndarray,
    *,
    mask: np.ndarray | None,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    """Float-native adaptive histogram equalization with bilinear tile blending.

    Unlike OpenCV's CLAHE interfaces, this implementation never encodes the
    image as uint8 or uint16. Each tile's mapping is computed from the exact
    float32 intensity values present in that tile.
    """

    unit = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if unit.ndim != 2 or unit.size == 0:
        return unit.astype(np.float32, copy=False)
    height, width = unit.shape
    grid_y = max(1, min(int(tile_grid_size), height))
    grid_x = max(1, min(int(tile_grid_size), width))
    y_edges = np.linspace(0, height, grid_y + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_x + 1, dtype=np.int64)
    valid_mask = (
        np.asarray(mask, dtype=bool)
        if mask is not None and np.asarray(mask).shape == unit.shape
        else np.ones(unit.shape, dtype=bool)
    )

    mappings: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for tile_y in range(grid_y):
        mapping_row: list[tuple[np.ndarray, np.ndarray]] = []
        y0, y1 = int(y_edges[tile_y]), int(y_edges[tile_y + 1])
        for tile_x in range(grid_x):
            x0, x1 = int(x_edges[tile_x]), int(x_edges[tile_x + 1])
            tile = unit[y0:y1, x0:x1]
            tile_mask = valid_mask[y0:y1, x0:x1]
            pixels = tile[tile_mask]
            if pixels.size == 0:
                pixels = tile.reshape(-1)
            mapping_row.append(
                _exact_float_cdf_mapping(
                    pixels,
                    clip_limit=max(float(clip_limit), 0.0),
                )
            )
        mappings.append(mapping_row)

    y_centers = (y_edges[:-1] + y_edges[1:] - 1).astype(np.float64) / 2.0
    x_centers = (x_edges[:-1] + x_edges[1:] - 1).astype(np.float64) / 2.0

    def interpolation_axis(
        length: int, centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.arange(length, dtype=np.float64)
        upper = np.searchsorted(centers, positions, side="right")
        lower = np.clip(upper - 1, 0, centers.size - 1)
        upper = np.clip(upper, 0, centers.size - 1)
        denominator = centers[upper] - centers[lower]
        weight = np.zeros(length, dtype=np.float32)
        changing = denominator > 0.0
        weight[changing] = (
            (positions[changing] - centers[lower[changing]])
            / denominator[changing]
        ).astype(np.float32)
        return lower, upper, np.clip(weight, 0.0, 1.0)

    y_lower, y_upper, y_weight = interpolation_axis(height, y_centers)
    x_lower, x_upper, x_weight = interpolation_axis(width, x_centers)
    output = np.empty_like(unit, dtype=np.float32)

    y_pairs = np.stack([y_lower, y_upper], axis=1)
    x_pairs = np.stack([x_lower, x_upper], axis=1)
    for y0_idx, y1_idx in np.unique(y_pairs, axis=0):
        rows = np.flatnonzero(
            (y_lower == y0_idx) & (y_upper == y1_idx)
        )
        if rows.size == 0:
            continue
        row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
        wy = y_weight[row_slice, None]
        for x0_idx, x1_idx in np.unique(x_pairs, axis=0):
            columns = np.flatnonzero(
                (x_lower == x0_idx) & (x_upper == x1_idx)
            )
            if columns.size == 0:
                continue
            column_slice = slice(int(columns[0]), int(columns[-1]) + 1)
            wx = x_weight[None, column_slice]
            block = unit[row_slice, column_slice]

            def apply_mapping(tile_y: int, tile_x: int) -> np.ndarray:
                levels, mapped = mappings[tile_y][tile_x]
                if levels.size <= 1:
                    return block
                return np.interp(block, levels, mapped).astype(np.float32)

            mapped_00 = apply_mapping(int(y0_idx), int(x0_idx))
            mapped_01 = apply_mapping(int(y0_idx), int(x1_idx))
            mapped_10 = apply_mapping(int(y1_idx), int(x0_idx))
            mapped_11 = apply_mapping(int(y1_idx), int(x1_idx))
            top = mapped_00 + wx * (mapped_01 - mapped_00)
            bottom = mapped_10 + wx * (mapped_11 - mapped_10)
            output[row_slice, column_slice] = top + wy * (bottom - top)

    return np.where(valid_mask, output, unit).astype(np.float32)


def _custom_operation_should_preserve_background(op: str) -> bool:
    return str(op or "").casefold().strip() in {
        "hist_equalize",
        "clahe",
        "percentile_normalize",
        "aggressive_upper_percentile_normalize",
        "standardize_to_target",
        "zscore_clip",
        "gamma",
        "log",
        "invert",
    }


def _morphology_kernel_custom(params: dict[str, Any]) -> np.ndarray | None:
    k = _odd_int_custom(params.get("kernel_size", params.get("ksize", 9)))
    shape = str(params.get("kernel_shape", "ellipse")).casefold().strip()
    if cv2 is not None:
        cv_shape = cv2.MORPH_ELLIPSE if shape == "ellipse" else cv2.MORPH_RECT
        return cv2.getStructuringElement(cv_shape, (k, k))
    return np.ones((k, k), dtype=bool)


def _morphology_basic_custom(arr: np.ndarray, params: dict[str, Any], *, op_name: str) -> np.ndarray:
    kernel = _morphology_kernel_custom(params)
    if kernel is None:
        return arr
    if cv2 is not None:
        code = cv2.MORPH_OPEN if op_name == "open" else cv2.MORPH_CLOSE
        return cv2.morphologyEx(arr.astype(np.float32), code, kernel).astype(np.float32)
    if scipy_ndimage is None:
        return arr
    fn = scipy_ndimage.grey_opening if op_name == "open" else scipy_ndimage.grey_closing
    return fn(arr.astype(np.float32), footprint=kernel).astype(np.float32)


def _morphology_contrast_custom(
    arr: np.ndarray,
    params: dict[str, Any],
    mask: np.ndarray | None,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "white_tophat":
        opened = _morphology_basic_custom(arr, params, op_name="open")
        out = arr.astype(np.float32) - opened.astype(np.float32)
    else:
        closed = _morphology_basic_custom(arr, params, op_name="close")
        out = closed.astype(np.float32) - arr.astype(np.float32)
    lo, hi = _safe_percentile(out, params.get("percentiles", [1.0, 99.0]), mask)
    return ((np.clip(out, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)


def _pectoral_suppression_custom(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Conservative optional MLO pectoral suppression.

    This masks a triangular upper-corner region only when explicitly requested.
    It is intentionally simple and off by default because detection near the
    chest wall can be clinically important.
    """
    out = arr.astype(np.float32).copy()
    side = str(params.get("side", "left")).casefold().strip()
    height, width = out.shape
    width_fraction = float(params.get("width_fraction", 0.33))
    height_fraction = float(params.get("height_fraction", 0.45))
    fill_value = float(params.get("fill_value", 0.0))
    tri_w = max(1, min(width, int(round(width * width_fraction))))
    tri_h = max(1, min(height, int(round(height * height_fraction))))
    for y in range(tri_h):
        x_extent = int(round(tri_w * (1.0 - y / max(tri_h - 1, 1))))
        if side == "right":
            out[y, max(0, width - x_extent):width] = fill_value
        else:
            out[y, 0:min(width, x_extent)] = fill_value
    return out


def _standardize_to_target_custom(
    arr: np.ndarray,
    params: dict[str, Any],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Dynamic affine standardization for exported custom-channel pipelines.

    y = a*x + b, where a and b are computed from the current crop/channel so
    the result has approximately the requested target mean and standard
    deviation. Statistics are estimated from foreground/mask pixels when
    available, then optionally trimmed by percentile range.
    """
    x = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pixels = x[mask] if mask is not None and mask.any() else x[np.isfinite(x)]
    if pixels.size == 0:
        pixels = x[np.isfinite(x)]
    if pixels.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    stat_percentiles = params.get("stat_percentiles", [1.0, 99.0])
    try:
        lo, hi = _safe_percentile(pixels, stat_percentiles, None)
        stat_pixels = pixels[(pixels >= lo) & (pixels <= hi)]
    except Exception:
        stat_pixels = pixels
    if stat_pixels.size < 2:
        stat_pixels = pixels

    current_mean = float(np.mean(stat_pixels))
    current_std = float(np.std(stat_pixels))
    target_mean = float(params.get("target_mean", 0.5))
    target_std = max(float(params.get("target_std", 0.2)), 1e-8)
    a = target_std / max(current_std, 1e-8)
    b = target_mean - a * current_mean
    y = (a * x + b).astype(np.float32)
    if bool(params.get("clip_output", True)):
        y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y

def _normalize_minmax_custom(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    pixels = arr[mask] if mask is not None and mask.any() else arr[np.isfinite(arr)]
    if pixels.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(np.min(pixels)), float(np.max(pixels))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _float_to_uint8_custom(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(arr, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    lo, hi = _safe_percentile(arr, [1.0, 99.0], None)
    return _to_uint8_window(arr, lo, hi)


def _float_to_unit_custom(arr: np.ndarray) -> np.ndarray:
    """Normalize one processed channel without reducing it to 256 levels."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    if float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    lo, hi = _safe_percentile(arr, [1.0, 99.0], None)
    return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)


def _float_rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Apply only the final 8-bit PNG quantization to a [0, 1] array."""
    value = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def _float32_rgb_tensor_metadata(rgb: np.ndarray) -> dict[str, Any]:
    """Describe the exact CHW tensor produced from one float RGB branch."""
    value = np.nan_to_num(np.asarray(rgb, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    tensor = torch.from_numpy(np.clip(value, 0.0, 1.0)).permute(2, 0, 1).contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "layout": "CHW",
        "shape": [int(value) for value in tensor.shape],
        "min": float(tensor.min().item()) if tensor.numel() else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() else 0.0,
        "finite": bool(torch.isfinite(tensor).all().item()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _save_float32_rgb_tensor(rgb: np.ndarray, path: Path) -> dict[str, Any]:
    """Save one HWC float RGB image as a contiguous PyTorch CHW tensor."""
    value = np.nan_to_num(np.asarray(rgb, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    tensor = torch.from_numpy(np.clip(value, 0.0, 1.0)).permute(2, 0, 1).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "layout": "CHW",
        "shape": [int(value) for value in tensor.shape],
        "min": float(tensor.min().item()) if tensor.numel() else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() else 0.0,
        "finite": bool(torch.isfinite(tensor).all().item()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _odd_int_custom(value: Any) -> int:
    k = int(value)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _make_intensity_equalized_gradient_rgb(
    arr: np.ndarray,
    img_cfg: dict[str, Any],
    mask: np.ndarray,
    *,
    return_float: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Create RGB channels as [normal intensity, equalized intensity, gradient].

    This is useful when normal multi-window RGB still looks nearly grayscale.
    It does not mathematically preserve all 16-bit DICOM information in the
    8-bit training PNG. The preserved 16-bit PNG is still the depth-preserving
    copy. Instead, this scheme uses the three RGB channels to provide complementary
    views that are meaningful to a CNN:

    * R: a normal robust intensity window.
    * G: the same window after histogram equalization.
    * B: edge/texture information from Sobel gradient magnitude.
    """
    cfg = img_cfg.get("intensity_equalized_gradient", {}) or {}
    intensity_window = cfg.get("intensity_window", img_cfg.get("single_window", [1.0, 99.0]))
    gradient_window = cfg.get("gradient_window", [1.0, 99.0])
    gradient_source = str(cfg.get("gradient_source", "normal")).casefold().strip()
    gradient_ksize = int(cfg.get("gradient_ksize", 3))

    lo, hi = _safe_percentile(arr, intensity_window, mask)
    if return_float:
        normal = _to_float_window(arr, lo, hi)
        equalized = _apply_custom_channel_operation_float_preserving(
            normal, "hist_equalize", {}, mask
        )
    else:
        normal = _to_uint8_window(arr, lo, hi)
        equalized = _equalize_uint8(normal, mask=mask)

    if gradient_source in {"equalized", "histogram_equalized", "eq"}:
        grad_input = equalized
    else:
        grad_input = normal
    if return_float:
        gradient = _apply_custom_channel_operation_float_preserving(
            grad_input,
            "sobel_gradient",
            {"percentiles": gradient_window, "ksize": gradient_ksize},
            mask,
        )
        gradient_method = (
            "opencv_sobel_float32" if cv2 is not None else "numpy_gradient_float32"
        )
        grad_meta = {
            "gradient_method": gradient_method,
            "gradient_resolved_window": list(map(float, gradient_window)),
        }
    else:
        gradient, grad_meta = _sobel_gradient_uint8(
            grad_input,
            mask=mask,
            gradient_window=gradient_window,
            ksize=gradient_ksize,
        )

    meta: dict[str, Any] = {
        "rgb_scheme": "intensity_equalized_gradient",
        "rgb_channel_0": "robust_intensity_window",
        "rgb_channel_1": "histogram_equalized_intensity_window",
        "rgb_channel_2": "sobel_gradient_magnitude",
        "rgb_windows": [[float(lo), float(hi)], [float(lo), float(hi)], grad_meta["gradient_resolved_window"]],
        "ieg_intensity_percentiles": list(map(float, intensity_window)),
        "ieg_gradient_percentiles": list(map(float, gradient_window)),
        "ieg_gradient_source": gradient_source,
        "ieg_gradient_ksize": gradient_ksize,
        **grad_meta,
    }
    return [normal, equalized, gradient], meta


def _sobel_gradient_uint8(
    img: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    gradient_window: Iterable[float] = (1.0, 99.0),
    ksize: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an 8-bit Sobel gradient-magnitude image.

    The input is an 8-bit intensity view. The gradient magnitude is calculated,
    then robustly percentile-windowed over foreground pixels so a few very strong
    edges do not make all weaker mass/tissue edges disappear.
    """
    img_u8 = img.astype(np.uint8, copy=False)
    if cv2 is not None:
        # ksize must be odd and positive for Sobel. ksize=3 is a good compact default.
        if ksize < 1:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(gx, gy).astype(np.float32, copy=False)
        method = "opencv_sobel"
    else:  # lightweight fallback if OpenCV is unavailable
        gy, gx = np.gradient(img_u8.astype(np.float32))
        mag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
        method = "numpy_gradient_fallback"

    gmask = mask if mask is not None and mask.any() else np.isfinite(mag)
    glo, ghi = _safe_percentile(mag, gradient_window, gmask)
    grad = _to_uint8_window(mag, glo, ghi)
    return grad, {
        "gradient_method": method,
        "gradient_resolved_window": [float(glo), float(ghi)],
    }


def _make_uint16_preserved(arr: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config.get("preserved_16bit", {})
    mask = _foreground_mask(arr) if bool(cfg.get("use_foreground_mask", True)) else np.isfinite(arr)
    lo, hi = _safe_percentile(arr, cfg.get("percentile_range", [0.1, 99.9]), mask)
    scaled = np.clip(arr, lo, hi)
    scaled = (scaled - lo) / max(hi - lo, 1e-12)
    img16 = np.round(np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return img16, {"preserved_16bit_lo": float(lo), "preserved_16bit_hi": float(hi)}


def _foreground_fraction_in_window(
    image: np.ndarray,
    window_xyxy: tuple[int, int, int, int],
    *,
    crop_size: int,
    threshold: float | None,
    pad_value: float = 0.0,
) -> float:
    """Return a window fraction from one mask computed on the full image.

    ``pad_value`` is retained for API compatibility; out-of-image mask pixels
    are always background. Crucially, threshold estimation never sees an
    isolated tissue-filled patch.
    """
    del pad_value
    mask = _foreground_mask(np.asarray(image, dtype=np.float32), threshold=threshold)
    return _foreground_fraction_from_mask(mask, window_xyxy, int(crop_size))


def _foreground_mask(arr: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Robust breast foreground mask for statistics and foreground filtering.

    Tiny nonzero background noise used to pass the old low threshold. The new
    default threshold is a small fraction of the image range, and connected
    component cleanup removes isolated noisy speckles.
    """
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=bool)
    vals = arr[finite]
    if threshold is None:
        try:
            threshold = _robust_tissue_threshold(arr)
        except Exception:
            lo, hi = np.percentile(vals, [1.0, 99.5])
            threshold = max(float(lo + 0.02 * (hi - lo)), float(lo) + 1e-6)
    mask = finite & (arr > float(threshold))
    mask = _cleanup_foreground_mask(mask, min_area_fraction=0.001, keep_largest=True)
    if not mask.any() and float(np.nanmax(arr)) > float(np.nanmin(arr)):
        # Very small previews can make the border-based threshold sample most
        # of the image. Fall back to all pixels above the finite minimum rather
        # than handing an all-false mask to operations that would zero output.
        fallback = finite & (arr > float(np.nanmin(arr)))
        mask = _cleanup_foreground_mask(fallback, min_area_fraction=0.001, keep_largest=True)
    return mask


def _cleanup_foreground_mask(mask: np.ndarray, *, min_area_fraction: float = 0.001, keep_largest: bool = True) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    if cv2 is None:
        return mask
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    if areas.size == 0:
        return np.zeros_like(mask, dtype=bool)
    min_area = max(1, int(round(float(mask.size) * float(min_area_fraction))))
    largest_label = int(np.argmax(areas)) + 1
    keep_labels = {largest_label} if keep_largest else set()
    for label_idx, area in enumerate(areas, start=1):
        if int(area) >= min_area:
            keep_labels.add(int(label_idx))
    return np.isin(labels, list(keep_labels)).astype(bool, copy=False)


def _safe_percentile(arr: np.ndarray, percentiles: Iterable[float], mask: np.ndarray | None = None) -> tuple[float, float]:
    p = list(percentiles)
    if len(p) != 2:
        raise ValueError("Percentile window must have two values, e.g. [0.5, 99.5].")
    pixels = arr[mask] if mask is not None and mask.any() else arr[np.isfinite(arr)]
    if pixels.size == 0:
        return 0.0, 1.0
    p0, p1 = float(p[0]), float(p[1])
    # Allow user-facing fractional notation: [0.7, 1.0] means [70, 100] percentiles.
    if 0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0:
        p0, p1 = 100.0 * p0, 100.0 * p1
    lo, hi = np.percentile(pixels, [p0, p1])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(pixels)), float(np.nanmax(pixels))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_uint8_window(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / max(float(hi - lo), 1e-12)
    return np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_float_window(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Window pixels directly into float32 [0, 1] without integer encoding."""

    clipped = np.clip(np.asarray(arr, dtype=np.float32), lo, hi)
    scaled = (clipped - float(lo)) / max(float(hi - lo), 1e-12)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def _equalize_uint8(img: np.ndarray, mask: np.ndarray | None = None, params: dict[str, Any] | None = None) -> np.ndarray:
    """Histogram equalization for an 8-bit single-channel image.

    When a foreground mask is available, the equalization LUT is estimated from
    a trimmed tissue region. This keeps MLO views with a large chest-wall/pectoral
    area from getting a noticeably different global contrast curve than CC views.
    The transform is still applied to the full breast.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    stat_mask = _hist_equalization_stat_mask_uint8(img, mask, params)
    if stat_mask is not None and stat_mask.any():
        values = img[stat_mask]
    else:
        values = img.reshape(-1)
    hist = np.bincount(values, minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    valid = cdf > 0
    if not valid.any():
        return img.copy()
    cdf_min = cdf[valid][0]
    denom = max(float(cdf[-1] - cdf_min), 1.0)
    lut = np.round((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    out = lut[img]
    if mask is not None and mask.shape == img.shape:
        out = np.where(mask, out, 0).astype(np.uint8)
    return out


def _hist_equalization_stat_mask_uint8(
    img: np.ndarray,
    mask: np.ndarray | None,
    params: dict[str, Any] | None = None,
) -> np.ndarray | None:
    params = params or {}
    if mask is None or mask.shape != img.shape or not mask.any():
        return mask
    stat_mask = np.asarray(mask, dtype=bool).copy()

    try:
        exclude_fraction = float(params.get("exclude_chest_wall_fraction", 0.0) or 0.0)
    except Exception:
        exclude_fraction = 0.0
    exclude_fraction = min(max(exclude_fraction, 0.0), 0.45)
    if exclude_fraction > 0.0:
        ys, xs = np.where(stat_mask)
        if xs.size:
            x0, x1 = int(xs.min()), int(xs.max())
            width = max(1, x1 - x0 + 1)
            band = max(1, int(round(width * exclude_fraction)))
            try:
                side = _breast_chest_wall_side(stat_mask) or "left"
            except Exception:
                side = "left"
            if side == "right":
                stat_mask[:, max(x0, x1 - band + 1):x1 + 1] = False
            else:
                stat_mask[:, x0:min(x1 + 1, x0 + band)] = False
            if not stat_mask.any():
                stat_mask = np.asarray(mask, dtype=bool).copy()

    percentiles = params.get("stat_percentiles", [1.0, 99.5])
    try:
        lo, hi = _safe_percentile(img, percentiles, stat_mask)
        trimmed = stat_mask & np.isfinite(img) & (img >= lo) & (img <= hi)
        if trimmed.any():
            stat_mask = trimmed
    except Exception:
        pass
    return stat_mask


# -----------------------------------------------------------------------------
# Annotation and metadata writers
# -----------------------------------------------------------------------------


def _review_safe_token(value: Any) -> str:
    text = str(value or "").strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    token = token.strip("_")
    if token:
        return token[:180]
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _review_image_to_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _tensor_to_float2d(image) if isinstance(image, torch.Tensor) else np.asarray(image, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum >= -1e-6 and maximum <= 1.0 + 1e-6:
        scaled = np.clip(arr, 0.0, 1.0)
    else:
        lo, hi = np.percentile(finite, [0.5, 99.5]).astype(float)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = minimum, maximum
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _review_raw_dicom_to_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Linearly display-scale decoded stored pixels without clinical preprocessing."""
    arr = _tensor_to_float2d(image) if isinstance(image, torch.Tensor) else np.asarray(image)
    arr = np.asarray(np.squeeze(arr), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D raw DICOM pixels, got shape {arr.shape}.")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if maximum <= minimum:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _read_review_raw_dicom_pixels(record: dict[str, Any]) -> np.ndarray:
    """Decode stored DICOM pixels without LUT, polarity, masking, CLAHE, or mirroring."""
    if pydicom is None:  # pragma: no cover
        raise ImportError("pydicom is required to save raw-original review previews.")
    dicom_path = Path(str(record.get("dicom_path", "") or ""))
    if not dicom_path.is_file():
        raise FileNotFoundError(
            f"Cannot save raw-original review preview; DICOM not found: {dicom_path}"
        )
    dataset = pydicom.dcmread(str(dicom_path))
    pixels = np.asarray(np.squeeze(dataset.pixel_array))
    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2D raw mammogram, got shape {pixels.shape} for {dicom_path}."
        )
    return pixels


def _review_resize(image: Image.Image, max_side: int, *, is_mask: bool = False) -> Image.Image:
    max_side = max(64, int(max_side))
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height, 1)))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if size == image.size:
        return image.copy()
    resampling = getattr(Image, "Resampling", Image)
    method = resampling.NEAREST if is_mask else resampling.LANCZOS
    return image.resize(size, method)


def _review_draw_boxes(
    image: Image.Image,
    boxes: Iterable[Iterable[float]],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    color: tuple[int, int, int] = (255, 48, 48),
    width: int = 3,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        values = list(box)
        if len(values) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in values]
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle(
            [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
            outline=color,
            width=max(1, int(width)),
        )
    return out


def _save_review_source_assets(
    *,
    crop_root: Path,
    split_name: str,
    record: dict[str, Any],
    image: torch.Tensor,
    target: dict[str, Any],
    review_cfg: dict[str, Any],
    original_image: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Save raw/fixed source previews plus the exact retained breast mask."""
    source_id = str(record.get("image_id", ""))
    token = _review_safe_token(source_id)
    max_side = max(128, int(review_cfg.get("source_preview_max_side", 1200) or 1200))
    save_source = bool(review_cfg.get("save_source_previews", True))
    save_original = bool(review_cfg.get("save_original_previews", True))
    save_masks = bool(review_cfg.get("save_masks", True))

    gray = _review_image_to_uint8(image)
    source_height, source_width = int(gray.shape[0]), int(gray.shape[1])
    base = _review_resize(Image.fromarray(gray, mode="L").convert("RGB"), max_side)
    preview_width, preview_height = base.size
    scale_x = float(preview_width) / float(max(1, source_width))
    scale_y = float(preview_height) / float(max(1, source_height))
    boxes = _boxes_to_list((target.get("mass", {}) or {}).get("boxes"))
    annotated = _review_draw_boxes(base, boxes, scale_x=scale_x, scale_y=scale_y)

    review_root = crop_root / "review"
    original_path = review_root / "original_images" / split_name / f"{token}.png"
    source_path = review_root / "source_images" / split_name / f"{token}.png"
    mask_path = review_root / "masks" / split_name / f"{token}.png"
    overlay_path = review_root / "mask_overlays" / split_name / f"{token}.png"
    original_width = 0
    original_height = 0
    original_preview_width = 0
    original_preview_height = 0
    if save_original:
        raw_original = (
            _read_review_raw_dicom_pixels(record)
            if original_image is None
            else np.asarray(original_image)
        )
        raw_gray = _review_raw_dicom_to_uint8(raw_original)
        original_height, original_width = int(raw_gray.shape[0]), int(raw_gray.shape[1])
        original_preview = _review_resize(
            Image.fromarray(raw_gray, mode="L").convert("RGB"),
            max_side,
        )
        original_preview_width, original_preview_height = original_preview.size
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_preview.save(original_path, format="PNG", optimize=True)

    if save_source:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(source_path, format="PNG", optimize=True)

    mask = target.get("_foreground_mask")
    mask_source = "retained_preprocessing_mask"
    if mask is None:
        mask = _foreground_mask(_tensor_to_float2d(image), threshold=None)
        mask_source = "derived_from_fixed_preprocessed_image"
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != gray.shape:
        raise ValueError(
            f"Review mask shape {mask_arr.shape} does not match source image {gray.shape} "
            f"for {source_id}."
        )
    if save_masks:
        resized_mask = _review_resize(
            Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L"),
            max_side,
            is_mask=True,
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        resized_mask.save(mask_path, format="PNG", optimize=True)
        mask_bool = np.asarray(resized_mask, dtype=np.uint8) > 0
        overlay_arr = np.asarray(annotated, dtype=np.float32).copy()
        alpha = min(max(float(review_cfg.get("mask_overlay_alpha", 0.40)), 0.0), 1.0)
        red = np.zeros_like(overlay_arr)
        red[..., 0] = 255.0
        overlay_arr[mask_bool] = (1.0 - alpha) * overlay_arr[mask_bool] + alpha * red[mask_bool]
        overlay = Image.fromarray(np.rint(overlay_arr).astype(np.uint8), mode="RGB")
        overlay = _review_draw_boxes(overlay, boxes, scale_x=scale_x, scale_y=scale_y)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path, format="PNG", optimize=True)

    preprocess_info = dict(target.get("preprocessing", {}) or {})
    return {
        "split": split_name,
        "source_image_id": source_id,
        "source_study_id": str(record.get("study_id", "")),
        "source_width": source_width,
        "source_height": source_height,
        "original_width": original_width,
        "original_height": original_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "original_preview_width": original_preview_width,
        "original_preview_height": original_preview_height,
        "original_preview_path": (
            original_path.relative_to(crop_root).as_posix() if save_original else ""
        ),
        "original_preview_processing": (
            "stored_pixel_decode_then_full_range_linear_uint8_display_scale_then_resize"
            if save_original
            else ""
        ),
        "source_preview_path": source_path.relative_to(crop_root).as_posix() if save_source else "",
        "mask_path": mask_path.relative_to(crop_root).as_posix() if save_masks else "",
        "mask_overlay_path": overlay_path.relative_to(crop_root).as_posix() if save_masks else "",
        "mask_source": mask_source,
        "mass_boxes_xyxy": boxes,
        "mirrored": bool(preprocess_info.get("mirrored", False)),
        "original_shape": preprocess_info.get("original_shape"),
        "processed_shape": preprocess_info.get("processed_shape"),
        "crop_box_xyxy": preprocess_info.get("crop_box_xyxy"),
        "coordinate_space": "fixed_preprocessed",
    }


def _review_parse_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4:
        try:
            return tuple(float(v) for v in value)  # type: ignore[return-value]
        except Exception:
            return None
    text = str(value or "").strip().strip("[]()")
    if not text:
        return None
    try:
        values = [float(part.strip()) for part in text.split(",")]
    except Exception:
        return None
    return tuple(values) if len(values) == 4 else None  # type: ignore[return-value]


def _review_letterbox(image: Image.Image, size: int) -> Image.Image:
    size = max(128, int(size))
    rgb = image.convert("RGB")
    scale = min(float(size) / max(1, rgb.width), float(size) / max(1, rgb.height))
    target = (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale))))
    resampling = getattr(Image, "Resampling", Image)
    resized = rgb.resize(target, resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (10, 10, 10))
    canvas.paste(resized, ((size - target[0]) // 2, (size - target[1]) // 2))
    return canvas


def _review_pair_frame(
    left: Image.Image,
    right: Image.Image,
    *,
    left_title: str,
    right_title: str,
    footer: str,
    panel_size: int,
) -> Image.Image:
    panel_size = max(256, int(panel_size))
    header = 38
    footer_height = 30
    gutter = 8
    canvas = Image.new(
        "RGB",
        (panel_size * 2 + gutter, panel_size + header + footer_height),
        (20, 20, 20),
    )
    canvas.paste(_review_letterbox(left, panel_size), (0, header))
    canvas.paste(_review_letterbox(right, panel_size), (panel_size + gutter, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), left_title, fill=(255, 255, 255))
    draw.text((panel_size + gutter + 10, 12), right_title, fill=(255, 255, 255))
    draw.text((10, panel_size + header + 8), footer[:220], fill=(220, 220, 220))
    return canvas


def _review_triplet_frame(
    left: Image.Image,
    middle: Image.Image,
    right: Image.Image,
    *,
    left_title: str,
    middle_title: str,
    right_title: str,
    footer: str,
    panel_size: int,
) -> Image.Image:
    panel_size = max(256, int(panel_size))
    header = 38
    footer_height = 30
    gutter = 8
    canvas = Image.new(
        "RGB",
        (panel_size * 3 + gutter * 2, panel_size + header + footer_height),
        (20, 20, 20),
    )
    canvas.paste(_review_letterbox(left, panel_size), (0, header))
    canvas.paste(
        _review_letterbox(middle, panel_size),
        (panel_size + gutter, header),
    )
    canvas.paste(
        _review_letterbox(right, panel_size),
        (panel_size * 2 + gutter * 2, header),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), left_title, fill=(255, 255, 255))
    draw.text((panel_size + gutter + 10, 12), middle_title, fill=(255, 255, 255))
    draw.text(
        (panel_size * 2 + gutter * 2 + 10, 12),
        right_title,
        fill=(255, 255, 255),
    )
    draw.text((10, panel_size + header + 8), footer[:330], fill=(220, 220, 220))
    return canvas


def _write_review_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> bool:
    if not frame_paths:
        return False
    frames: list[Image.Image] = []
    adaptive = getattr(Image, "ADAPTIVE", 1)
    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                frames.append(frame.convert("RGB").convert("P", palette=adaptive, colors=128))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=max(50, int(duration_ms)),
            loop=0,
            optimize=False,
            disposal=2,
        )
        return True
    finally:
        for frame in frames:
            frame.close()


def _write_whole_variant_review_assets(
    *,
    crop_root: Path,
    source_rows: list[dict[str, Any]],
    review_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Audit whole-image pixels and transformed boxes with overlays and plots."""
    review_root = crop_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    geometry_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    gifs: dict[str, str] = {}
    variant_specs = {
        "original": {
            "path": "paired_whole_original_image_path",
            "annotations": "paired_whole_original_annotations",
            "width": "paired_whole_original_width",
            "height": "paired_whole_original_height",
            "pad_left": "paired_whole_original_pad_left",
            "pad_top": "paired_whole_original_pad_top",
            "pad_right": "paired_whole_original_pad_right",
            "pad_bottom": "paired_whole_original_pad_bottom",
            "scale_x": "paired_whole_original_scale_x",
            "scale_y": "paired_whole_original_scale_y",
        },
        "resized": {
            "path": "paired_whole_image_path",
            "annotations": "paired_whole_annotations",
            "width": "paired_whole_width",
            "height": "paired_whole_height",
            "pad_left": "paired_whole_pad_left",
            "pad_top": "paired_whole_pad_top",
            "pad_right": "paired_whole_pad_right",
            "pad_bottom": "paired_whole_pad_bottom",
            "scale_x": "paired_whole_scale_x",
            "scale_y": "paired_whole_scale_y",
        },
        "high_resolution": {
            "path": "paired_whole_high_resolution_image_path",
            "annotations": "paired_whole_high_resolution_annotations",
            "width": "paired_whole_high_resolution_width",
            "height": "paired_whole_high_resolution_height",
            "pad_left": "paired_whole_high_resolution_pad_left",
            "pad_top": "paired_whole_high_resolution_pad_top",
            "pad_right": "paired_whole_high_resolution_pad_right",
            "pad_bottom": "paired_whole_high_resolution_pad_bottom",
            "scale_x": "paired_whole_high_resolution_scale_x",
            "scale_y": "paired_whole_high_resolution_scale_y",
        },
    }
    for source in source_rows:
        for variant, spec in variant_specs.items():
            image_path_text = str(source.get(spec["path"], "") or "")
            if not image_path_text:
                continue
            width = int(source.get(spec["width"], 0) or 0)
            height = int(source.get(spec["height"], 0) or 0)
            annotations = list(source.get(spec["annotations"], []) or [])
            geometry_rows.append({
                "split": source.get("split", ""),
                "source_image_id": source.get("source_image_id", ""),
                "source_study_id": source.get("source_study_id", ""),
                "variant": variant,
                "image_path": image_path_text,
                "width": width,
                "height": height,
                "pad_left": int(source.get(spec["pad_left"], 0) or 0),
                "pad_top": int(source.get(spec["pad_top"], 0) or 0),
                "pad_right": int(source.get(spec["pad_right"], 0) or 0),
                "pad_bottom": int(source.get(spec["pad_bottom"], 0) or 0),
                "scale_x": float(source.get(spec["scale_x"], 1.0) or 1.0),
                "scale_y": float(source.get(spec["scale_y"], 1.0) or 1.0),
                "num_annotations": len(annotations),
            })
            for index, annotation in enumerate(annotations):
                bbox = [float(value) for value in annotation.get("bbox_xyxy", [])]
                source_bbox = [
                    float(value) for value in annotation.get("source_bbox_xyxy", [])
                ]
                transform = dict(annotation.get("transform", {}) or {})
                expected: list[float] = []
                if len(source_bbox) == 4:
                    expected = [
                        (source_bbox[0] + float(transform.get("pad_left", 0)))
                        * float(transform.get("scale_x", 1)),
                        (source_bbox[1] + float(transform.get("pad_top", 0)))
                        * float(transform.get("scale_y", 1)),
                        (source_bbox[2] + float(transform.get("pad_left", 0)))
                        * float(transform.get("scale_x", 1)),
                        (source_bbox[3] + float(transform.get("pad_top", 0)))
                        * float(transform.get("scale_y", 1)),
                    ]
                max_error = (
                    max(abs(actual - wanted) for actual, wanted in zip(bbox, expected))
                    if len(bbox) == 4 and len(expected) == 4
                    else float("nan")
                )
                box_rows.append({
                    "split": source.get("split", ""),
                    "source_image_id": source.get("source_image_id", ""),
                    "variant": variant,
                    "annotation_index": index,
                    "source_annotation_id": annotation.get("source_annotation_id", ""),
                    "source_bbox_xyxy": json.dumps(source_bbox),
                    "bbox_xyxy": json.dumps(bbox),
                    "within_output_bounds": int(
                        len(bbox) == 4
                        and 0 <= bbox[0] < bbox[2] <= width
                        and 0 <= bbox[1] < bbox[3] <= height
                    ),
                    "max_transform_error_px": max_error,
                })

    geometry_path = review_root / "whole_variant_geometry.csv"
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)
    created.append(geometry_path)
    box_path = review_root / "whole_variant_box_audit.csv"
    pd.DataFrame(box_rows).to_csv(box_path, index=False)
    created.append(box_path)

    max_side = max(256, int(review_cfg.get("source_preview_max_side", 1200) or 1200))
    panel_size = max(256, int(review_cfg.get("gif_panel_size", 640) or 640))
    samples_per_split = max(1, int(review_cfg.get("samples_per_split", 100) or 100))
    duration_ms = max(50, int(review_cfg.get("gif_frame_duration_ms", 700) or 700))
    rng = np.random.default_rng(int(review_cfg.get("seed", 123) or 123) + 1701)
    for split_name in ["train", "val", "test"]:
        candidates = [
            source for source in source_rows
            if str(source.get("split", "")) == split_name
            and any(str(source.get(spec["path"], "") or "") for spec in variant_specs.values())
        ]
        count = min(samples_per_split, len(candidates))
        selected = (
            [candidates[int(i)] for i in rng.choice(len(candidates), size=count, replace=False)]
            if count
            else []
        )
        frame_paths: list[Path] = []
        for frame_index, source in enumerate(selected):
            previews: dict[str, Image.Image] = {}
            overlay_paths: dict[str, str] = {}
            for variant, spec in variant_specs.items():
                path_text = str(source.get(spec["path"], "") or "")
                path = crop_root / path_text if path_text else None
                if path is None or not path.is_file():
                    continue
                annotations = list(source.get(spec["annotations"], []) or [])
                boxes = [annotation.get("bbox_xyxy", []) for annotation in annotations]
                with Image.open(path) as image_file:
                    original = image_file.convert("RGB")
                    preview = _review_resize(original, max_side)
                    preview = _review_draw_boxes(
                        preview,
                        boxes,
                        scale_x=float(preview.width) / float(max(1, original.width)),
                        scale_y=float(preview.height) / float(max(1, original.height)),
                        width=max(2, preview.width // 400),
                    )
                overlay_path = (
                    review_root
                    / "whole_variant_overlays"
                    / variant
                    / split_name
                    / f"{_review_safe_token(source.get('source_image_id'))}.png"
                )
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                preview.save(overlay_path, format="PNG", optimize=True)
                created.append(overlay_path)
                previews[variant] = preview
                overlay_paths[variant] = overlay_path.relative_to(crop_root).as_posix()
            if not previews:
                continue
            placeholder = Image.new("RGB", (panel_size, panel_size), (20, 20, 20))
            footer = (
                f"{split_name} | image={source.get('source_image_id', '')} | "
                f"Mass boxes={len(source.get('mass_boxes_xyxy', []) or [])}"
            )
            titles = {
                "original": "Original-size processed + matched boxes",
                "resized": "Square-padded/resized + matched boxes",
                "high_resolution": "Fixed-canvas high resolution + matched boxes",
            }
            available = [
                variant for variant in ["original", "resized", "high_resolution"]
                if variant in previews
            ]
            if len(available) >= 3:
                frame = _review_triplet_frame(
                    previews["original"],
                    previews["resized"],
                    previews["high_resolution"],
                    left_title=titles["original"],
                    middle_title=titles["resized"],
                    right_title=titles["high_resolution"],
                    footer=footer,
                    panel_size=panel_size,
                )
            else:
                left_variant = available[0]
                right_variant = available[1] if len(available) > 1 else None
                frame = _review_pair_frame(
                    previews[left_variant],
                    previews[right_variant] if right_variant else placeholder,
                    left_title=titles[left_variant],
                    right_title=titles[right_variant] if right_variant else "No second variant enabled",
                    footer=footer,
                    panel_size=panel_size,
                )
            frame_path = (
                review_root
                / "whole_variant_frames"
                / split_name
                / f"{frame_index:04d}_{_review_safe_token(source.get('source_image_id'))}.png"
            )
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(frame_path, format="PNG", optimize=True)
            frame.close()
            for preview in previews.values():
                preview.close()
            placeholder.close()
            frame_paths.append(frame_path)
            created.append(frame_path)
            sampled_rows.append({
                "split": split_name,
                "source_image_id": str(source.get("source_image_id", "")),
                "frame_path": frame_path.relative_to(crop_root).as_posix(),
                **{f"{variant}_overlay_path": path for variant, path in overlay_paths.items()},
            })
        if bool(review_cfg.get("create_whole_variant_gifs", True)) and frame_paths:
            gif_path = review_root / "gifs" / f"{split_name}_whole_variants.gif"
            if _write_review_gif(frame_paths, gif_path, duration_ms):
                created.append(gif_path)
                gifs[split_name] = gif_path.relative_to(crop_root).as_posix()

    sampled_path = review_root / "whole_variant_samples.csv"
    pd.DataFrame(sampled_rows).to_csv(sampled_path, index=False)
    created.append(sampled_path)

    if geometry_rows:
        try:
            import matplotlib.pyplot as plt

            plot_root = review_root / "whole_variant_visualizations"
            plot_root.mkdir(parents=True, exist_ok=True)
            geometry_df = pd.DataFrame(geometry_rows)

            fig, ax = plt.subplots(figsize=(9, 7))
            for variant, group in geometry_df.groupby("variant"):
                ax.scatter(group["width"], group["height"], alpha=0.55, label=variant)
            ax.set_xlabel("Output width (px)")
            ax.set_ylabel("Output height (px)")
            ax.set_title("Whole-image output dimensions by variant")
            ax.legend()
            ax.grid(alpha=0.2)
            dimensions_plot = plot_root / "01_output_dimensions.png"
            fig.tight_layout()
            fig.savefig(dimensions_plot, dpi=160)
            plt.close(fig)
            created.append(dimensions_plot)

            fig, ax = plt.subplots(figsize=(10, 6))
            for variant, group in geometry_df.groupby("variant"):
                padding = group[["pad_left", "pad_top", "pad_right", "pad_bottom"]].sum(axis=1)
                ax.hist(padding, bins=30, alpha=0.45, label=variant)
            ax.set_xlabel("Total padding across four sides (px)")
            ax.set_ylabel("Images")
            ax.set_title("Padding distribution by whole-image variant")
            ax.legend()
            padding_plot = plot_root / "02_padding_distribution.png"
            fig.tight_layout()
            fig.savefig(padding_plot, dpi=160)
            plt.close(fig)
            created.append(padding_plot)

            parity = geometry_df.groupby("variant")["num_annotations"].sum()
            fig, ax = plt.subplots(figsize=(8, 5))
            parity.plot(kind="bar", ax=ax, color=["#5B8FF9", "#61DDAA", "#F6BD16"][:len(parity)])
            ax.set_ylabel("Transformed Mass boxes")
            ax.set_title("Annotation-count parity across whole-image variants")
            ax.tick_params(axis="x", rotation=0)
            parity_plot = plot_root / "03_annotation_count_parity.png"
            fig.tight_layout()
            fig.savefig(parity_plot, dpi=160)
            plt.close(fig)
            created.append(parity_plot)

            fig, ax = plt.subplots(figsize=(10, 6))
            for variant, group in geometry_df.groupby("variant"):
                ax.hist(group["scale_x"], bins=30, alpha=0.45, label=variant)
            ax.set_xlabel("Annotation/image scale factor (x)")
            ax.set_ylabel("Images")
            ax.set_title("Whole-image scale factors by variant")
            ax.legend()
            scale_plot = plot_root / "04_scale_factor_distribution.png"
            fig.tight_layout()
            fig.savefig(scale_plot, dpi=160)
            plt.close(fig)
            created.append(scale_plot)
        except Exception:
            # The CSV and per-image overlays remain the authoritative audit if
            # matplotlib is unavailable in a lightweight export environment.
            pass

    invalid_boxes = sum(
        int(row.get("within_output_bounds", 0)) == 0 for row in box_rows
    )
    max_transform_error = max(
        [float(row["max_transform_error_px"]) for row in box_rows if np.isfinite(row["max_transform_error_px"])]
        or [0.0]
    )
    return {
        "geometry_csv": geometry_path.relative_to(crop_root).as_posix(),
        "box_audit_csv": box_path.relative_to(crop_root).as_posix(),
        "sampled_frames": len(sampled_rows),
        "gifs": gifs,
        "invalid_boxes": invalid_boxes,
        "max_transform_error_px": max_transform_error,
    }, list(dict.fromkeys(created))


def _write_dataset_review_bundle(
    *,
    crop_root: Path,
    stats_rows: list[dict[str, Any]],
    coco_by_split: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    review_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Build persistent side-by-side audit frames and optional GIFs."""
    review_root = crop_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    source_lookup = {
        (str(row.get("split", "")), str(row.get("source_image_id", ""))): row
        for row in source_rows
    }
    for row in source_rows:
        for key in [
            "original_preview_path",
            "source_preview_path",
            "mask_path",
            "mask_overlay_path",
        ]:
            path_text = str(row.get(key, "") or "")
            path = crop_root / path_text if path_text else None
            if path is not None and path.exists():
                created.append(path)

    annotations_by_file: dict[tuple[str, str], list[list[float]]] = {}
    for split_name, coco in coco_by_split.items():
        image_by_id = {int(image["id"]): image for image in coco.get("images", [])}
        for annotation in coco.get("annotations", []):
            image_meta = image_by_id.get(int(annotation.get("image_id", -1)))
            bbox = list(annotation.get("bbox", []) or [])
            if image_meta is None or len(bbox) != 4:
                continue
            x, y, width, height = [float(v) for v in bbox]
            annotations_by_file.setdefault(
                (split_name, str(image_meta.get("file_name", ""))), []
            ).append([x, y, x + width, y + height])

    seed = int(review_cfg.get("seed", 123) or 123)
    rng = np.random.default_rng(seed)
    samples_per_split = max(0, int(review_cfg.get("samples_per_split", 100) or 0))
    panel_size = max(256, int(review_cfg.get("gif_panel_size", 640) or 640))
    create_crop_gifs = bool(review_cfg.get("create_crop_gifs", True))
    create_mask_gifs = bool(review_cfg.get("create_mask_gifs", True))
    duration_ms = max(50, int(review_cfg.get("gif_frame_duration_ms", 700) or 700))
    sampled_crops: list[dict[str, Any]] = []
    sampled_masks: list[dict[str, Any]] = []
    gif_paths: dict[str, dict[str, str]] = {}
    frame_folders: dict[str, dict[str, str]] = {}

    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in stats_rows if str(row.get("split", "")) == split_name]
        sample_count = min(samples_per_split, len(split_rows))
        selected_rows = (
            [split_rows[int(i)] for i in rng.choice(len(split_rows), size=sample_count, replace=False)]
            if sample_count
            else []
        )
        crop_frame_paths: list[Path] = []
        for frame_index, row in enumerate(selected_rows):
            source_key = (split_name, str(row.get("source_image_id", "")))
            source = source_lookup.get(source_key)
            original_path_text = str((source or {}).get("original_preview_path", "") or "")
            source_path_text = str((source or {}).get("source_preview_path", "") or "")
            crop_path = crop_root / "images" / split_name / str(row.get("file_name", ""))
            original_path = crop_root / original_path_text if original_path_text else None
            source_path = crop_root / source_path_text if source_path_text else None
            if (
                source is None
                or original_path is None
                or not original_path.exists()
                or source_path is None
                or not source_path.exists()
                or not crop_path.exists()
            ):
                continue
            with (
                Image.open(original_path) as original_image_file,
                Image.open(source_path) as source_image_file,
                Image.open(crop_path) as crop_image_file,
            ):
                original_image = original_image_file.convert("RGB")
                source_image = source_image_file.convert("RGB")
                window = _review_parse_xyxy(row.get("crop_window_xyxy"))
                if window is not None:
                    sx = float(source_image.width) / float(max(1, int(source.get("source_width", 1))))
                    sy = float(source_image.height) / float(max(1, int(source.get("source_height", 1))))
                    source_image = _review_draw_boxes(
                        source_image,
                        [window],
                        scale_x=sx,
                        scale_y=sy,
                        color=(0, 220, 255),
                        width=4,
                    )
                crop_boxes = annotations_by_file.get(
                    (split_name, str(row.get("file_name", ""))), []
                )
                crop_image = _review_draw_boxes(crop_image_file.convert("RGB"), crop_boxes)
                frame = _review_triplet_frame(
                    original_image,
                    source_image,
                    crop_image,
                    left_title="Original DICOM pixels (display-scaled only)",
                    middle_title="Full fixed-preprocessed mammogram (red mass; cyan crop)",
                    right_title="Exported crop (red mass)",
                    footer=(
                        f"{split_name} | image={row.get('source_image_id', '')} | "
                        f"crop={row.get('file_name', '')} | mirrored={int(bool(source.get('mirrored', False)))}"
                    ),
                    panel_size=panel_size,
                )
            frame_path = review_root / "crop_frames" / split_name / f"{frame_index:04d}_{Path(str(row.get('file_name', 'crop'))).stem}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(frame_path, format="PNG", optimize=True)
            frame.close()
            crop_frame_paths.append(frame_path)
            created.append(frame_path)
            sampled_crops.append({
                "split": split_name,
                "file_name": str(row.get("file_name", "")),
                "source_image_id": str(row.get("source_image_id", "")),
                "crop_frame_path": frame_path.relative_to(crop_root).as_posix(),
            })

        split_gifs: dict[str, str] = {}
        split_frame_folders: dict[str, str] = {}
        if crop_frame_paths:
            split_frame_folders["crop_review"] = (
                crop_frame_paths[0].parent.relative_to(crop_root).as_posix()
            )
        if create_crop_gifs and crop_frame_paths:
            crop_gif_path = review_root / "gifs" / f"{split_name}_crop_review.gif"
            if _write_review_gif(crop_frame_paths, crop_gif_path, duration_ms):
                created.append(crop_gif_path)
                split_gifs["crop_review"] = crop_gif_path.relative_to(crop_root).as_posix()

        mask_candidates = [
            row for row in source_rows
            if str(row.get("split", "")) == split_name
            and str(row.get("source_preview_path", ""))
            and str(row.get("original_preview_path", ""))
            and str(row.get("mask_overlay_path", ""))
        ]
        mask_count = min(samples_per_split, len(mask_candidates))
        selected_masks = (
            [mask_candidates[int(i)] for i in rng.choice(len(mask_candidates), size=mask_count, replace=False)]
            if mask_count
            else []
        )
        mask_frame_paths: list[Path] = []
        for frame_index, source in enumerate(selected_masks):
            original_path = crop_root / str(source["original_preview_path"])
            source_path = crop_root / str(source["source_preview_path"])
            overlay_path = crop_root / str(source["mask_overlay_path"])
            if (
                not original_path.exists()
                or not source_path.exists()
                or not overlay_path.exists()
            ):
                continue
            with (
                Image.open(original_path) as original_file,
                Image.open(source_path) as source_file,
                Image.open(overlay_path) as overlay_file,
            ):
                frame = _review_triplet_frame(
                    original_file.convert("RGB"),
                    source_file.convert("RGB"),
                    overlay_file.convert("RGB"),
                    left_title="Original DICOM pixels (display-scaled only)",
                    middle_title="Full fixed-preprocessed mammogram (red mass)",
                    right_title="Retained breast mask in red",
                    footer=(
                        f"{split_name} | image={source.get('source_image_id', '')} | "
                        f"mask={source.get('mask_source', '')} | mirrored={int(bool(source.get('mirrored', False)))}"
                    ),
                    panel_size=panel_size,
                )
            frame_path = review_root / "mask_frames" / split_name / f"{frame_index:04d}_{_review_safe_token(source.get('source_image_id'))}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(frame_path, format="PNG", optimize=True)
            frame.close()
            mask_frame_paths.append(frame_path)
            created.append(frame_path)
            sampled_masks.append({
                "split": split_name,
                "source_image_id": str(source.get("source_image_id", "")),
                "mask_frame_path": frame_path.relative_to(crop_root).as_posix(),
            })
        if mask_frame_paths:
            split_frame_folders["mask_review"] = (
                mask_frame_paths[0].parent.relative_to(crop_root).as_posix()
            )
        if create_mask_gifs and mask_frame_paths:
            mask_gif_path = review_root / "gifs" / f"{split_name}_mask_review.gif"
            if _write_review_gif(mask_frame_paths, mask_gif_path, duration_ms):
                created.append(mask_gif_path)
                split_gifs["mask_review"] = mask_gif_path.relative_to(crop_root).as_posix()
        gif_paths[split_name] = split_gifs
        frame_folders[split_name] = split_frame_folders

    whole_variant_review, whole_variant_files = _write_whole_variant_review_assets(
        crop_root=crop_root,
        source_rows=source_rows,
        review_cfg=review_cfg,
    )
    created.extend(whole_variant_files)

    sources_csv = review_root / "sources.csv"
    flat_sources = []
    for row in source_rows:
        flat = dict(row)
        flat["mass_boxes_xyxy"] = json.dumps(_json_safe(flat.get("mass_boxes_xyxy", [])))
        flat_sources.append(flat)
    pd.DataFrame(flat_sources).to_csv(sources_csv, index=False)
    created.append(sources_csv)

    index_payload = {
        "version": 2,
        "coordinate_space": "fixed_preprocessed",
        "config": {
            **_json_safe(review_cfg),
            "samples_per_split": samples_per_split,
            "seed": seed,
        },
        "sources": source_rows,
        "sampled_crops": sampled_crops,
        "sampled_masks": sampled_masks,
        "whole_variant_review": whole_variant_review,
        "gifs": gif_paths,
        "frame_folders": frame_folders,
    }
    index_path = review_root / "index.json"
    index_path.write_text(json.dumps(_json_safe(index_payload), indent=2), encoding="utf-8")
    created.append(index_path)

    summary = {
        "enabled": True,
        "root": _path_as_posix(review_root),
        "original_previews": sum(
            bool(row.get("original_preview_path")) for row in source_rows
        ),
        "source_previews": sum(bool(row.get("source_preview_path")) for row in source_rows),
        "saved_masks": sum(bool(row.get("mask_path")) for row in source_rows),
        "sampled_crop_frames": len(sampled_crops),
        "whole_variant_review": whole_variant_review,
        "samples_per_split": samples_per_split,
        "gifs": gif_paths,
        "frame_folders": frame_folders,
        "index_json": _path_as_posix(index_path),
    }
    # Preserve insertion order while removing paths repeated through source rows.
    created = list(dict.fromkeys(created))
    return summary, created


def _write_square_crop_debug_logs(
    crop_root: Path,
    stats_rows: list[dict[str, Any]],
    source_debug: dict[tuple[str, str], dict[str, Any]],
) -> list[Path]:
    """Write human-readable debug logs for crop provenance and coverage."""
    created: list[Path] = []
    debug_dir = crop_root / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)

    crop_log_cols = [
        "split",
        "source_index",
        "source_image_id",
        "source_study_id",
        "file_name",
        "paired_whole_key",
        "paired_whole_original_image_path",
        "paired_whole_image_path",
        "paired_whole_high_resolution_image_path",
        "paired_whole_native_image_path",
        "has_mass",
        "num_mass_boxes",
        "is_positive_window",
        "crop_mode",
        "crop_window_xyxy",
        "deterministic_selection_mode",
        "source_image_has_mass",
        "negative_crop_source_policy",
        "breast_fraction_mask_source",
        "foreground_fraction",
        "min_breast_fraction_for_all_crops",
        "negative_foreground_fraction",
        "bbox_safe_foreground_fraction",
        "bbox_safe_margin_ok",
        "contralateral_image_id",
        "contralateral_alignment_method",
        "contralateral_alignment_shift_y",
    ]
    crop_df = pd.DataFrame(stats_rows)
    for col in crop_log_cols:
        if col not in crop_df.columns:
            crop_df[col] = ""
    crop_log_path = debug_dir / "crop_log.csv"
    crop_df[crop_log_cols].to_csv(crop_log_path, index=False)
    created.append(crop_log_path)

    source_rows: list[dict[str, Any]] = []
    for row in source_debug.values():
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        included = row.get("_included_annotation_indices", set())
        try:
            included_count = len(included)
            included_list = ",".join(str(int(v)) for v in sorted(included))
        except Exception:
            included_count = 0
            included_list = ""
        out["included_target_annotation_count"] = int(included_count)
        out["included_target_annotation_indices"] = included_list
        out["source_image_has_no_saved_crops"] = int(int(out.get("saved_crops", 0)) == 0)
        source_rows.append(out)
    source_df = pd.DataFrame(source_rows)
    if not source_df.empty:
        source_df = source_df.sort_values(["split", "source_index", "source_image_id"])
    source_log_path = debug_dir / "source_image_log.csv"
    source_df.to_csv(source_log_path, index=False)
    created.append(source_log_path)

    hist_path = debug_dir / "crops_per_source_histogram.csv"
    if source_df.empty:
        pd.DataFrame(columns=["split", "n_crops", "n_source_images"]).to_csv(hist_path, index=False)
    else:
        hist_df = (
            source_df.groupby(["split", "saved_crops"], dropna=False)
            .size()
            .reset_index(name="n_source_images")
            .rename(columns={"saved_crops": "n_crops"})
            .sort_values(["split", "n_crops"])
        )
        hist_df.to_csv(hist_path, index=False)
    created.append(hist_path)

    coverage_rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        if source_df.empty:
            sdf = pd.DataFrame()
        else:
            sdf = source_df[source_df["split"] == split_name].copy()
        if crop_df.empty:
            cdf = pd.DataFrame()
        else:
            cdf = crop_df[crop_df["split"] == split_name].copy()
        total_source_masses = int(pd.to_numeric(sdf.get("n_source_mass_boxes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        included_target = int(pd.to_numeric(sdf.get("included_target_annotation_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        exported_mass_instances = int(pd.to_numeric(sdf.get("exported_mass_box_instances", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        saved_crops = int(pd.to_numeric(sdf.get("saved_crops", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        positive_crops = int(pd.to_numeric(sdf.get("saved_positive_crops", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        negative_crops = int(pd.to_numeric(sdf.get("saved_negative_crops", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0
        coverage_rows.append({
            "split": split_name,
            "source_images": int(len(sdf)),
            "source_images_with_mass": int(pd.to_numeric(sdf.get("has_source_mass", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0,
            "source_images_with_no_saved_crops": int(pd.to_numeric(sdf.get("source_image_has_no_saved_crops", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sdf.empty else 0,
            "source_mass_annotations": int(total_source_masses),
            "included_target_mass_annotations": int(included_target),
            "included_target_mass_annotation_fraction": float(included_target / total_source_masses) if total_source_masses > 0 else 0.0,
            "exported_mass_box_instances": int(exported_mass_instances),
            "saved_crops": int(saved_crops),
            "positive_crops": int(positive_crops),
            "negative_crops": int(negative_crops),
            "achieved_positive_crop_ratio": float(positive_crops / max(1, saved_crops)),
            "saved_crop_rows_in_crop_log": int(len(cdf)),
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = debug_dir / "split_mass_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    created.append(coverage_path)

    summary = {
        "debug_folder": _path_as_posix(debug_dir),
        "source_images_total": int(len(source_df)),
        "saved_crops_total": int(len(crop_df)),
        "source_images_with_no_saved_crops_total": int(source_df["source_image_has_no_saved_crops"].sum()) if not source_df.empty and "source_image_has_no_saved_crops" in source_df else 0,
        "coverage_by_split": coverage_rows,
    }
    summary_json_path = debug_dir / "debug_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    created.append(summary_json_path)

    summary_txt_path = crop_root.parent / "square_crop_debug_summary.txt"
    lines = [
        "Square crop debug summary",
        "=========================",
        f"Debug folder: {_path_as_posix(debug_dir)}",
        f"Saved crops total: {summary['saved_crops_total']}",
        f"Source images with no saved crops: {summary['source_images_with_no_saved_crops_total']}",
        "",
        "Per split:",
    ]
    for row in coverage_rows:
        lines.append(
            f"- {row['split']}: saved={row['saved_crops']}, positive={row['positive_crops']}, "
            f"negative={row['negative_crops']}, source_masses={row['source_mass_annotations']}, "
            f"included_target_masses={row['included_target_mass_annotations']}, "
            f"no_data_images={row['source_images_with_no_saved_crops']}"
        )
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    created.append(summary_txt_path)
    return created


def _validate_square_crop_contract(
    *,
    split_records: dict[str, list[dict[str, Any]]],
    coco_by_split: dict[str, dict[str, Any]],
    source_debug: dict[tuple[str, str], dict[str, Any]],
    candidate_positive_window_keys: set[tuple[str, str, int, int, int, int]],
    saved_positive_window_keys: set[tuple[str, str, int, int, int, int]],
    crop_cfg: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate materialized patches before a completion marker can be written."""
    contract = dict(config.get("replication_contract", {}) or {})
    if not bool(contract.get("enabled", False)):
        return {"enabled": False, "status": "not_requested"}

    errors: list[str] = []
    metrics: dict[str, Any] = {}
    crop_policy = dict(config.get("crop_annotation_policy", {}) or {})
    actual_min_box_visibility = float(crop_policy.get("min_box_visibility", 0.30))
    allow_partial_annotations = bool(
        crop_policy.get("allow_partial_annotations", False)
    )
    metrics["crop_annotation_visibility"] = {
        "allow_partial_annotations": allow_partial_annotations,
        "minimum_visible_box_fraction": actual_min_box_visibility,
        "comparison": "greater_than_or_equal",
    }
    expected_min_box_visibility = contract.get("expected_min_box_visibility")
    if expected_min_box_visibility is not None:
        expected_visibility = float(expected_min_box_visibility)
        if not allow_partial_annotations:
            errors.append(
                "crop annotation policy disallows partial boxes; the contract requires "
                f"labels at >= {expected_visibility:.3f} visibility"
            )
        if not math.isclose(
            actual_min_box_visibility,
            expected_visibility,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(
                "crop annotation minimum visibility is "
                f"{actual_min_box_visibility:.6f}; expected {expected_visibility:.6f}"
            )
    missing_positive = candidate_positive_window_keys - saved_positive_window_keys
    unexpected_positive = saved_positive_window_keys - candidate_positive_window_keys
    if missing_positive:
        errors.append(f"{len(missing_positive)} candidate-positive windows were not written")
    if unexpected_positive:
        errors.append(f"{len(unexpected_positive)} written positive windows were not candidates")
    metrics["candidate_positive_windows"] = len(candidate_positive_window_keys)
    metrics["saved_positive_windows"] = len(saved_positive_window_keys)

    source_metrics: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        rows = [row for (split, _image_id), row in source_debug.items() if split == split_name]
        expected_source_rows = len(split_records.get(split_name, []))
        if len(rows) != expected_source_rows:
            errors.append(
                f"{split_name}: debug provenance has {len(rows)}/{expected_source_rows} source images"
            )
        source_mass_count = sum(int(row.get("n_source_mass_boxes", 0)) for row in rows)
        represented_ids = sum(len(row.get("_included_annotation_indices", set())) for row in rows)
        positive_sources_without_positive_patch = sum(
            1
            for row in rows
            if int(row.get("n_source_mass_boxes", 0)) > 0
            and int(row.get("saved_positive_crops", 0)) == 0
        )
        complete_grid = sum(int(row.get("complete_grid_windows", 0)) for row in rows)
        saved_crops = sum(int(row.get("saved_crops", 0)) for row in rows)
        saved_negatives = sum(int(row.get("saved_negative_crops", 0)) for row in rows)
        candidate_windows = sum(int(row.get("candidate_windows", 0)) for row in rows)
        positive_candidates = sum(int(row.get("positive_candidate_windows", 0)) for row in rows)
        eligible_negatives = max(0, candidate_windows - positive_candidates)
        grid_fraction = float(saved_crops / complete_grid) if complete_grid else 0.0
        source_metrics[split_name] = {
            "source_images": len(rows),
            "source_mass_annotations": source_mass_count,
            "represented_source_annotation_ids": represented_ids,
            "positive_sources_without_positive_patch": positive_sources_without_positive_patch,
            "complete_grid_windows": complete_grid,
            "saved_crops": saved_crops,
            "saved_grid_fraction": grid_fraction,
            "eligible_negative_candidates": eligible_negatives,
            "saved_negative_crops": saved_negatives,
        }
        require_annotations_by_split = dict(
            contract.get("require_all_source_annotations_represented_by_split", {}) or {}
        )
        require_annotations = bool(
            require_annotations_by_split.get(
                split_name,
                contract.get("require_all_source_annotations_represented", False),
            )
        )
        if require_annotations and represented_ids != source_mass_count:
            errors.append(
                f"{split_name}: represented {represented_ids}/{source_mass_count} source annotations"
            )
        if positive_sources_without_positive_patch:
            errors.append(
                f"{split_name}: {positive_sources_without_positive_patch} positive source images have no positive patch"
            )

    metrics["splits"] = source_metrics
    train_mode = _deterministic_selection_mode(crop_cfg, "train")
    expected_train_mode = str(
        contract.get("expected_train_selection_mode", "negative_fraction")
    )
    if train_mode != expected_train_mode:
        errors.append(
            f"train: deterministic selection mode is {train_mode!r}, expected {expected_train_mode!r}"
        )
    expected_eval_mode = str(contract.get("expected_eval_selection_mode", "all"))
    for split_name in ["val", "test"]:
        mode = _deterministic_selection_mode(crop_cfg, split_name)
        if mode != expected_eval_mode:
            errors.append(
                f"{split_name}: deterministic selection mode is {mode!r}, expected {expected_eval_mode!r}"
            )

    if expected_train_mode == "negative_fraction":
        train_candidates = int(source_metrics["train"]["eligible_negative_candidates"])
        keep_fraction = _deterministic_negative_keep_fraction(crop_cfg, "train")
        expected_train_negatives = int(round(train_candidates * keep_fraction))
        actual_train_negatives = int(source_metrics["train"]["saved_negative_crops"])
        metrics["train_negative_selection"] = {
            "candidate_count": train_candidates,
            "keep_fraction": keep_fraction,
            "expected_saved_count": expected_train_negatives,
            "actual_saved_count": actual_train_negatives,
        }
        if actual_train_negatives != expected_train_negatives:
            errors.append(
                f"train: saved {actual_train_negatives}/{train_candidates} eligible negatives; "
                f"expected rounded {keep_fraction:.3f} retention ({expected_train_negatives})"
            )

    train_images = list(coco_by_split.get("train", {}).get("images", []))
    train_positive_image_ids = {
        int(annotation.get("image_id"))
        for annotation in coco_by_split.get("train", {}).get("annotations", [])
    }
    train_positive_crops = sum(
        int(int(image.get("id", -1)) in train_positive_image_ids)
        for image in train_images
    )
    train_negative_crops = len(train_images) - train_positive_crops
    train_crop_positive_fraction = (
        float(train_positive_crops / len(train_images)) if train_images else 0.0
    )
    metrics["train_crop_label_balance"] = {
        "positive_crops": train_positive_crops,
        "negative_crops": train_negative_crops,
        "positive_fraction": train_crop_positive_fraction,
    }
    expected_crop_positive_fraction = contract.get(
        "expected_train_crop_positive_fraction"
    )
    crop_ratio_tolerance = float(
        contract.get("train_crop_positive_fraction_tolerance", 0.0) or 0.0
    )
    if expected_crop_positive_fraction is not None and not math.isclose(
        train_crop_positive_fraction,
        float(expected_crop_positive_fraction),
        rel_tol=0.0,
        abs_tol=crop_ratio_tolerance,
    ):
        errors.append(
            "train: crop-label positive fraction is "
            f"{train_crop_positive_fraction:.6f}; expected "
            f"{float(expected_crop_positive_fraction):.6f} ± {crop_ratio_tolerance:.6f}"
        )

    require_negative_images = bool(
        contract.get(
            "require_training_negative_crops_from_mass_negative_images",
            False,
        )
    )
    require_negative_breasts = bool(
        contract.get(
            "require_training_negative_crops_from_mass_negative_breasts",
            False,
        )
    )
    if require_negative_images or require_negative_breasts:
        negative_train_images = [
            image
            for image in train_images
            if int(image.get("id", -1)) not in train_positive_image_ids
        ]
        missing_source_rows: list[dict[str, Any]] = []
        invalid_image_sources: list[dict[str, Any]] = []
        invalid_breast_sources: list[dict[str, Any]] = []
        image_metadata_mismatches: list[dict[str, Any]] = []
        breast_metadata_mismatches: list[dict[str, Any]] = []
        for image in negative_train_images:
            source_image_id = str(image.get("source_image_id", ""))
            source_row = source_debug.get(("train", source_image_id))
            if source_row is None:
                missing_source_rows.append(image)
                continue
            # These values originate independently from the source annotations.
            # Never validate the policy solely against COCO metadata written by
            # the same selector: that allowed a bad default to certify itself.
            actual_image_has_mass = int(
                source_row.get(
                    "source_image_has_mass",
                    int(source_row.get("n_source_mass_boxes", 0)) > 0,
                )
                or 0
            )
            actual_breast_has_mass = int(
                source_row.get("source_breast_has_mass", 0) or 0
            )
            exported_image_has_mass = int(
                image.get("source_image_has_mass", 0) or 0
            )
            exported_breast_has_mass = int(
                image.get("source_breast_has_mass", 0) or 0
            )
            if actual_image_has_mass:
                invalid_image_sources.append(image)
            if actual_breast_has_mass:
                invalid_breast_sources.append(image)
            if exported_image_has_mass != actual_image_has_mass:
                image_metadata_mismatches.append(image)
            if exported_breast_has_mass != actual_breast_has_mass:
                breast_metadata_mismatches.append(image)

        invalid_required_sources = (
            invalid_breast_sources
            if require_negative_breasts
            else invalid_image_sources
        )
        metrics["train_negative_crop_source_policy"] = {
            "required": (
                "mass_negative_breasts_only"
                if require_negative_breasts
                else "mass_negative_images_only"
            ),
            # Retain the original summary key while exposing both scopes.
            "invalid_negative_crops": len(invalid_required_sources),
            "invalid_source_image_crops": len(invalid_image_sources),
            "invalid_source_breast_crops": len(invalid_breast_sources),
            "missing_source_provenance_crops": len(missing_source_rows),
            "source_image_metadata_mismatches": len(image_metadata_mismatches),
            "source_breast_metadata_mismatches": len(breast_metadata_mismatches),
        }
        if require_negative_images and invalid_image_sources:
            errors.append(
                f"train: {len(invalid_image_sources)} empty crops came from source images containing Mass"
            )
        if require_negative_breasts and invalid_breast_sources:
            errors.append(
                "train: "
                f"{len(invalid_breast_sources)} empty crops came from breasts with Mass "
                "in the source or paired view"
            )
        if missing_source_rows:
            errors.append(
                "train: "
                f"{len(missing_source_rows)} empty crops could not be matched to "
                "independent source provenance"
            )
        if image_metadata_mismatches or breast_metadata_mismatches:
            errors.append(
                "train: exported Mass-source metadata disagrees with independent "
                f"source provenance for {len(image_metadata_mismatches)} image-status "
                f"and {len(breast_metadata_mismatches)} breast-status crop records"
            )

    mass_breast_crops = sum(
        int(image.get("source_breast_has_mass", 0) or 0) for image in train_images
    )
    negative_breast_crops = len(train_images) - mass_breast_crops
    achieved_mass_breast_fraction = (
        float(mass_breast_crops / len(train_images)) if train_images else 0.0
    )
    metrics["train_source_breast_crop_balance"] = {
        "mass_breast_crops": mass_breast_crops,
        "negative_breast_crops": negative_breast_crops,
        "mass_breast_fraction": achieved_mass_breast_fraction,
    }
    expected_breast_fraction = contract.get(
        "expected_train_mass_breast_crop_fraction"
    )
    if expected_breast_fraction is not None and not math.isclose(
        achieved_mass_breast_fraction,
        float(expected_breast_fraction),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            "train: source-breast crop balance is "
            f"{achieved_mass_breast_fraction:.6f}; expected {float(expected_breast_fraction):.6f}"
        )

    strict_min_by_split = dict(
        contract.get("min_breast_fraction_strictly_greater_than_by_split", {}) or {}
    )
    # Backward compatibility for manifests/configs written before per-split
    # breast-mask thresholds were supported.
    legacy_train_min = contract.get("train_min_breast_fraction_strictly_greater_than")
    if legacy_train_min is not None:
        strict_min_by_split.setdefault("train", legacy_train_min)
    for split_name, strict_min_breast_fraction in strict_min_by_split.items():
        if split_name not in {"train", "val", "test"}:
            errors.append(
                "min_breast_fraction_strictly_greater_than_by_split contains "
                f"unknown split {split_name!r}"
            )
            continue
        split_images = list(coco_by_split.get(split_name, {}).get("images", []))
        missing_fraction = [
            image for image in split_images if image.get("breast_fraction") is None
        ]
        violating_fraction = [
            image
            for image in split_images
            if image.get("breast_fraction") is not None
            and float(image.get("breast_fraction")) <= float(strict_min_breast_fraction)
        ]
        metrics[f"{split_name}_breast_fraction"] = {
            "required_strictly_greater_than": float(strict_min_breast_fraction),
            "minimum_saved": min(
                (
                    float(image.get("breast_fraction"))
                    for image in split_images
                    if image.get("breast_fraction") is not None
                ),
                default=None,
            ),
            "missing_count": len(missing_fraction),
            "violating_count": len(violating_fraction),
        }
        if missing_fraction or violating_fraction:
            errors.append(
                f"{split_name}: breast-mask coverage contract failed for "
                f"{len(missing_fraction)} missing and {len(violating_fraction)} <= "
                f"{float(strict_min_breast_fraction):.3f} crops"
            )

    min_inference_grid_fraction = float(contract.get("min_inference_grid_fraction", 0.0) or 0.0)
    for split_name in ["val", "test"]:
        fraction = float(source_metrics[split_name]["saved_grid_fraction"])
        if fraction < min_inference_grid_fraction:
            errors.append(
                f"{split_name}: saved {fraction:.3%} of the complete grid; "
                f"minimum is {min_inference_grid_fraction:.3%}"
            )
        complete_grid_by_split = dict(
            contract.get("require_complete_inference_grid_by_split", {}) or {}
        )
        require_complete_grid = bool(
            complete_grid_by_split.get(
                split_name,
                contract.get("require_complete_inference_grid", False),
            )
        )
        if require_complete_grid:
            saved = int(source_metrics[split_name]["saved_crops"])
            complete = int(source_metrics[split_name]["complete_grid_windows"])
            if saved != complete:
                errors.append(
                    f"{split_name}: saved {saved}/{complete} windows; complete inference grid required"
                )

    crop_size = int(crop_cfg.get("crop_size", 0))
    for split_name, coco in coco_by_split.items():
        image_ids = {image.get("id") for image in coco.get("images", [])}
        for image in coco.get("images", []):
            if int(image.get("width", -1)) != crop_size or int(image.get("height", -1)) != crop_size:
                errors.append(f"{split_name}: patch {image.get('file_name')} is not {crop_size}x{crop_size}")
                break
            window = image.get("crop_window_xyxy")
            if not isinstance(window, (list, tuple)) or len(window) != 4 or int(window[2]) - int(window[0]) != crop_size or int(window[3]) - int(window[1]) != crop_size:
                errors.append(f"{split_name}: patch {image.get('file_name')} has an invalid crop window")
                break
        for annotation in coco.get("annotations", []):
            bbox = annotation.get("bbox", [])
            valid = (
                annotation.get("image_id") in image_ids
                and isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and float(bbox[2]) > 0
                and float(bbox[3]) > 0
                and float(annotation.get("area", 0)) > 0
            )
            if not valid:
                errors.append(f"{split_name}: invalid or dangling COCO annotation {annotation.get('id')}")
                break
            if bool(contract.get("require_source_annotation_ids", False)) and annotation.get("source_annotation_id") is None:
                errors.append(f"{split_name}: COCO annotation {annotation.get('id')} lacks source_annotation_id")
                break

    report = {
        "enabled": True,
        "name": contract.get("name", "replication_contract"),
        "status": "pass" if not errors else "fail",
        "metrics": metrics,
        "errors": errors,
    }
    if errors and bool(contract.get("strict", True)):
        raise RuntimeError("Replication patch contract failed: " + "; ".join(errors))
    return report


def _write_shared_export_files(
    root: Path,
    coco_by_split: dict[str, dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    *,
    dataset_kind: str,
) -> list[Path]:
    created: list[Path] = []
    ann_dir = root / "mmdetection" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split_name, coco in coco_by_split.items():
        ann_path = ann_dir / f"instances_{split_name}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(coco), f)
        created.append(ann_path)

    # Write portable Ultralytics YAML files.
    #
    # 1) root/vindr_mass.yaml is the recommended file to pass to YOLO. It uses
    #    paths relative to the dataset root and contains no Windows/Linux absolute
    #    path.
    # 2) root/ultralytics/vindr_mass.yaml is kept for backward compatibility, but
    #    it must use ../images/... because it lives one folder below the dataset
    #    root. Do not write ``path: .`` here: Ultralytics may resolve it against
    #    the current working directory, which breaks portability.
    root_yolo_yaml = root / "vindr_mass.yaml"
    _write_ultralytics_yaml(root_yolo_yaml, train="images/train", val="images/val", test="images/test")
    created.append(root_yolo_yaml)

    legacy_yolo_yaml = root / "ultralytics" / "vindr_mass.yaml"
    legacy_yolo_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_ultralytics_yaml(legacy_yolo_yaml, train="../images/train", val="../images/val", test="../images/test")
    created.append(legacy_yolo_yaml)

    mmdet_readme = root / "mmdetection" / "README_mmdetection_paths.txt"
    _write_mmdetection_note(mmdet_readme, root)
    created.append(mmdet_readme)

    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(stats_rows)
    stats_path = stats_dir / "samples.csv"
    stats_df.to_csv(stats_path, index=False)
    created.append(stats_path)
    summary_path = stats_dir / "summary.csv"
    _summary_dataframe(stats_rows, dataset_kind).to_csv(summary_path, index=False)
    created.append(summary_path)

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_jsonl = metadata_dir / "samples_metadata.jsonl"
    with open(metadata_jsonl, "w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    created.append(metadata_jsonl)

    metadata_csv = metadata_dir / "samples_metadata_flat.csv"
    pd.DataFrame([_flatten_metadata_row(r) for r in metadata_rows]).to_csv(metadata_csv, index=False)
    created.append(metadata_csv)
    if dataset_kind == "square_crops":
        crop_location_rows = [
            location
            for row in metadata_rows
            if (location := _crop_location_row(row)) is not None
        ]
        crop_locations_csv = metadata_dir / "crop_locations.csv"
        pd.DataFrame(crop_location_rows).to_csv(crop_locations_csv, index=False)
        created.append(crop_locations_csv)
        created.extend(_write_whole_image_annotation_indexes(root, metadata_rows))
    return created


def _write_whole_image_annotation_indexes(
    crop_root: Path,
    metadata_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Write aggregate manifests and COCO files for every whole-image variant."""
    created: list[Path] = []
    variant_specs = {
        "original": {
            "image": "paired_whole_original_image_path",
            "label": "paired_whole_original_label_path",
            "annotation": "paired_whole_original_annotation_path",
            "annotations": "paired_whole_original_annotations",
            "width": "paired_whole_original_width",
            "height": "paired_whole_original_height",
            "float32_path": "paired_whole_original_float32_image_path",
            "float32_dtype": "paired_whole_original_float32_dtype",
            "float32_layout": "paired_whole_original_float32_layout",
            "float32_shape": "paired_whole_original_float32_shape",
            "float32_min": "paired_whole_original_float32_min",
            "float32_max": "paired_whole_original_float32_max",
            "float32_finite": "paired_whole_original_float32_finite",
            "float32_contiguous": "paired_whole_original_float32_contiguous",
            "pad_left": "paired_whole_original_pad_left",
            "pad_top": "paired_whole_original_pad_top",
            "pad_right": "paired_whole_original_pad_right",
            "pad_bottom": "paired_whole_original_pad_bottom",
            "scale_x": "paired_whole_original_scale_x",
            "scale_y": "paired_whole_original_scale_y",
        },
        "resized": {
            "image": "paired_whole_image_path",
            "label": "paired_whole_label_path",
            "annotation": "paired_whole_annotation_path",
            "annotations": "paired_whole_annotations",
            "width": "paired_whole_width",
            "height": "paired_whole_height",
            "float32_path": "paired_whole_float32_image_path",
            "float32_dtype": "paired_whole_float32_dtype",
            "float32_layout": "paired_whole_float32_layout",
            "float32_shape": "paired_whole_float32_shape",
            "float32_min": "paired_whole_float32_min",
            "float32_max": "paired_whole_float32_max",
            "float32_finite": "paired_whole_float32_finite",
            "float32_contiguous": "paired_whole_float32_contiguous",
            "pad_left": "paired_whole_pad_left",
            "pad_top": "paired_whole_pad_top",
            "pad_right": "paired_whole_pad_right",
            "pad_bottom": "paired_whole_pad_bottom",
            "scale_x": "paired_whole_scale_x",
            "scale_y": "paired_whole_scale_y",
        },
        "high_resolution": {
            "image": "paired_whole_high_resolution_image_path",
            "label": "paired_whole_high_resolution_label_path",
            "annotation": "paired_whole_high_resolution_annotation_path",
            "annotations": "paired_whole_high_resolution_annotations",
            "width": "paired_whole_high_resolution_width",
            "height": "paired_whole_high_resolution_height",
            "float32_path": "paired_whole_high_resolution_float32_image_path",
            "float32_dtype": "paired_whole_high_resolution_float32_dtype",
            "float32_layout": "paired_whole_high_resolution_float32_layout",
            "float32_shape": "paired_whole_high_resolution_float32_shape",
            "float32_min": "paired_whole_high_resolution_float32_min",
            "float32_max": "paired_whole_high_resolution_float32_max",
            "float32_finite": "paired_whole_high_resolution_float32_finite",
            "float32_contiguous": "paired_whole_high_resolution_float32_contiguous",
            "pad_left": "paired_whole_high_resolution_pad_left",
            "pad_top": "paired_whole_high_resolution_pad_top",
            "pad_right": "paired_whole_high_resolution_pad_right",
            "pad_bottom": "paired_whole_high_resolution_pad_bottom",
            "scale_x": "paired_whole_high_resolution_scale_x",
            "scale_y": "paired_whole_high_resolution_scale_y",
        },
    }
    manifest_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    unique_assets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metadata_rows:
        encoding = dict(row.get("encoding", {}) or {})
        split_name = str(row.get("split", ""))
        source_image_id = str(row.get("source_image_id", ""))
        active_variant_specs = dict(variant_specs)
        resized_outputs = dict(
            encoding.get("paired_whole_resized_variants", {}) or {}
        )
        emit_dynamic_resized = bool(
            resized_outputs and config is not None
        )
        if emit_dynamic_resized:
            active_variant_specs.pop("resized", None)
        for variant, spec in active_variant_specs.items():
            image_path = str(encoding.get(spec["image"], "") or "")
            if not image_path:
                continue
            key = (variant, split_name, source_image_id)
            unique_assets.setdefault(key, {
                "variant": variant,
                "split": split_name,
                "source_image_id": source_image_id,
                "source_study_id": str(row.get("source_study_id", "")),
                "image_path": image_path,
                "label_path": str(encoding.get(spec["label"], "") or ""),
                "annotation_path": str(encoding.get(spec["annotation"], "") or ""),
                "width": int(encoding.get(spec["width"], 0) or 0),
                "height": int(encoding.get(spec["height"], 0) or 0),
                "float32_path": str(encoding.get(spec["float32_path"], "") or ""),
                "float32_dtype": str(encoding.get(spec["float32_dtype"], "") or ""),
                "float32_layout": str(encoding.get(spec["float32_layout"], "") or ""),
                "float32_shape": json.dumps(
                    encoding.get(spec["float32_shape"], []) or []
                ),
                "float32_min": encoding.get(spec["float32_min"], ""),
                "float32_max": encoding.get(spec["float32_max"], ""),
                "float32_finite": encoding.get(spec["float32_finite"], ""),
                "float32_contiguous": encoding.get(
                    spec["float32_contiguous"], ""
                ),
                "pad_left": float(encoding.get(spec["pad_left"], 0.0) or 0.0),
                "pad_top": float(encoding.get(spec["pad_top"], 0.0) or 0.0),
                "pad_right": float(encoding.get(spec["pad_right"], 0.0) or 0.0),
                "pad_bottom": float(encoding.get(spec["pad_bottom"], 0.0) or 0.0),
                "scale_x": float(encoding.get(spec["scale_x"], 1.0) or 1.0),
                "scale_y": float(encoding.get(spec["scale_y"], 1.0) or 1.0),
                "source_breast_key": str(row.get("source_breast_key", "") or ""),
                "source_breast_has_mass": int(
                    bool(row.get("source_breast_has_mass", False))
                ),
                "source_preprocessing_mirrored": int(
                    bool(row.get("source_preprocessing_mirrored", False))
                ),
                "source_coordinate_space": str(
                    row.get("source_coordinate_space", "fixed_preprocessed")
                    or "fixed_preprocessed"
                ),
                "annotations": list(encoding.get(spec["annotations"], []) or []),
            })
        for resolution, raw_variant in (
            resized_outputs.items() if emit_dynamic_resized else []
        ):
            resized = dict(raw_variant or {})
            variant_id = str(
                resized.get("variant")
                or _whole_variant_id("resized", str(resolution))
            )
            image_path = str(resized.get("image_path", "") or "")
            if not image_path:
                continue
            key = (variant_id, split_name, source_image_id)
            geometry = dict(resized.get("geometry", {}) or {})
            unique_assets.setdefault(key, {
                "variant": variant_id,
                "variant_kind": "resized",
                "resolution": str(resolution),
                "split": split_name,
                "source_image_id": source_image_id,
                "source_study_id": str(row.get("source_study_id", "")),
                "image_path": image_path,
                "label_path": str(resized.get("label_path", "") or ""),
                "annotation_path": str(resized.get("annotation_path", "") or ""),
                "width": int(resized.get("width", 0) or 0),
                "height": int(resized.get("height", 0) or 0),
                "float32_path": str(resized.get("float32_image_path", "") or ""),
                "float32_dtype": str(resized.get("float32_dtype", "") or ""),
                "float32_layout": str(resized.get("float32_layout", "") or ""),
                "float32_shape": json.dumps(resized.get("float32_shape", []) or []),
                "float32_min": resized.get("float32_min", ""),
                "float32_max": resized.get("float32_max", ""),
                "float32_finite": resized.get("float32_finite", ""),
                "float32_contiguous": resized.get("float32_contiguous", ""),
                "pad_left": float(geometry.get("paired_whole_pad_left", 0.0) or 0.0),
                "pad_top": float(geometry.get("paired_whole_pad_top", 0.0) or 0.0),
                "pad_right": float(geometry.get("paired_whole_pad_right", 0.0) or 0.0),
                "pad_bottom": float(geometry.get("paired_whole_pad_bottom", 0.0) or 0.0),
                "scale_x": float(geometry.get("paired_whole_scale_x", 1.0) or 1.0),
                "scale_y": float(geometry.get("paired_whole_scale_y", 1.0) or 1.0),
                "source_breast_key": str(row.get("source_breast_key", "") or ""),
                "source_breast_has_mass": int(bool(row.get("source_breast_has_mass", False))),
                "source_preprocessing_mirrored": int(bool(row.get("source_preprocessing_mirrored", False))),
                "source_coordinate_space": str(row.get("source_coordinate_space", "fixed_preprocessed") or "fixed_preprocessed"),
                "annotations": list(resized.get("annotations", []) or []),
            })

    coco_by_variant_split: dict[tuple[str, str], dict[str, Any]] = {}
    for image_id, asset in enumerate(unique_assets.values(), start=1):
        annotations = list(asset.pop("annotations", []) or [])
        manifest_rows.append({**asset, "num_annotations": len(annotations)})
        coco = coco_by_variant_split.setdefault(
            (str(asset["variant"]), str(asset["split"])),
            _empty_coco(),
        )
        coco["info"] = {
            "description": (
                "VinDr-Mammo whole-image Mass annotations — "
                f"{asset['variant']} coordinates"
            )
        }
        coco["images"].append({
            "id": int(image_id),
            "file_name": str(asset["image_path"]),
            "width": int(asset["width"]),
            "height": int(asset["height"]),
            "source_image_id": str(asset["source_image_id"]),
            "source_study_id": str(asset["source_study_id"]),
            "export_split": str(asset["split"]),
            "variant": str(asset["variant"]),
            "label_path": str(asset["label_path"]),
            "annotation_path": str(asset["annotation_path"]),
        })
        for annotation in annotations:
            bbox_xywh = [float(value) for value in annotation.get("bbox_xywh", [])]
            if len(bbox_xywh) != 4:
                continue
            ann_id = len(coco["annotations"]) + 1
            ann = {
                "id": ann_id,
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": bbox_xywh,
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "iscrowd": 0,
                "segmentation": [],
                "source_bbox_xyxy": annotation.get("source_bbox_xyxy"),
                "source_bbox_coordinate_space": "fixed_preprocessed",
            }
            if annotation.get("source_annotation_id") is not None:
                ann["source_annotation_id"] = annotation.get("source_annotation_id")
            coco["annotations"].append(ann)
            annotation_rows.append({
                **{key: value for key, value in asset.items() if key != "annotation_path"},
                "annotation_index": int(ann_id - 1),
                "source_annotation_id": annotation.get("source_annotation_id", ""),
                "source_bbox_xyxy": json.dumps(annotation.get("source_bbox_xyxy", [])),
                "bbox_xyxy": json.dumps(annotation.get("bbox_xyxy", [])),
                "bbox_xywh": json.dumps(bbox_xywh),
            })

    metadata_dir = crop_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "whole_image_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    created.append(manifest_path)
    annotations_path = (
        crop_root / "annotations" / "whole_image_annotations.csv"
        if uses_grouped_dataset_layout(config)
        else metadata_dir / "whole_image_annotations.csv"
    )
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annotation_rows).to_csv(annotations_path, index=False)
    created.append(annotations_path)
    enabled_variants = {
        str(asset["variant"])
        for asset in manifest_rows
    }
    for variant in sorted(enabled_variants):
        for split_name in ["train", "val", "test"]:
            coco = coco_by_variant_split.get((variant, split_name), _empty_coco())
            path = crop_root / _whole_coco_relative_path(
                config or {}, variant, split_name
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_json_safe(coco)), encoding="utf-8")
            created.append(path)
    return created


def _write_global_metadata_files(output_root: Path, dataset: VindrMammoDataset, config: dict[str, Any]) -> list[Path]:
    """Copy full source CSV metadata and save the export configuration."""
    if not bool(config.get("metadata", {}).get("save_full_source_csvs", True)):
        return []
    created: list[Path] = []
    meta_root = output_root / "metadata" / "source_csv"
    meta_root.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("breast-level_annotations.csv", dataset.breast_df),
        ("finding_annotations.csv", dataset.finding_df),
        ("metadata.csv", dataset.metadata_df),
    ]:
        path = meta_root / name
        df.to_csv(path, index=False)
        created.append(path)
    # Save the resolved config in both places. Earlier project versions saved it
    # under ``metadata/export_config_resolved.yaml``; keeping a duplicate under
    # ``metadata/source_csv/`` makes the completion checks less surprising.
    config_paths = [
        output_root / "metadata" / "export_config_resolved.yaml",
        output_root / "metadata" / "source_csv" / "export_config_resolved.yaml",
    ]
    config_text = None
    if yaml is not None:
        config_text = yaml.safe_dump(_json_safe(config), sort_keys=False)
    else:
        config_text = json.dumps(_json_safe(config), indent=2)
    for config_path in config_paths:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
        created.append(config_path)
    return created


def _path_relative_to_root(path_value: Any, root: Path) -> str:
    """Return a portable path relative to root when possible."""
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _write_reproducibility_bundle(
    *,
    output_root: Path,
    data_root: Path,
    dataset: VindrMammoDataset,
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Write a compact, self-contained audit/replay description of an export.

    The normal export metadata remains authoritative. This bundle gathers the
    exact source membership, saved crop order/windows, exported annotations,
    resolved configuration, and software/source-table provenance in one place.
    Whole-image/DICOM checksums are opt-in because calculating them adds a very
    large second I/O pass on a full mammography dataset.
    """
    options = dict(config.get("reproducibility_bundle", {}) or {})
    subdir = str(options.get("output_subdir", "reproducibility") or "reproducibility").strip()
    bundle_root = output_root / subdir
    bundle_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    include_source_hashes = bool(options.get("include_source_dicom_sha256", False))
    include_image_hashes = bool(options.get("include_exported_image_sha256", False))

    source_rows: list[dict[str, Any]] = []
    source_order = 0
    for split_name in ["train", "val", "test"]:
        for split_order, record in enumerate(split_records.get(split_name, [])):
            dicom_path = Path(str(record.get("dicom_path", "") or ""))
            image_id = str(record.get("image_id", ""))
            try:
                mass_findings = dataset._filter_mass_findings(
                    dataset.findings_by_image_id.get(image_id, [])
                )
            except Exception:
                mass_findings = []
            source_row = {
                "source_membership_order": int(source_order),
                "split_membership_order": int(split_order),
                "export_split": split_name,
                "official_split": record.get("split"),
                "study_id": str(record.get("study_id", "")),
                "image_id": image_id,
                "laterality": record.get("laterality"),
                "view_position": record.get("view_position"),
                "source_breast_key": record.get("_source_breast_key", ""),
                "source_breast_has_mass": record.get("_source_breast_has_mass", ""),
                "source_image_has_mass": int(bool(mass_findings)),
                "source_mass_annotation_count": int(len(mass_findings)),
                "source_dicom_relative_path": _path_relative_to_root(dicom_path, data_root),
                "source_dicom_path_recorded": dicom_path.as_posix(),
                "source_record_json": json.dumps(
                    _json_safe(record), ensure_ascii=False, sort_keys=True
                ),
            }
            if include_source_hashes:
                source_row["source_dicom_sha256"] = (
                    _sha256_file(dicom_path) if dicom_path.is_file() else ""
                )
            source_rows.append(source_row)
            source_order += 1
    source_path = bundle_root / "source_images.csv"
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    created.append(source_path)

    crop_root = dataset_content_root(output_root, config)
    legacy_crop_root = output_root / "square_crops"
    if not (crop_root / "metadata" / "crop_locations.csv").is_file() and legacy_crop_root.is_dir():
        crop_root = legacy_crop_root
    crop_locations_path = crop_root / "metadata" / "crop_locations.csv"
    crop_stats_path = crop_root / "stats" / "samples.csv"
    crop_records_source = crop_root / "metadata" / "samples_metadata.jsonl"
    crop_manifest_path = bundle_root / "crops.csv"
    crop_rows_count = 0
    if crop_locations_path.is_file():
        crop_df = pd.read_csv(crop_locations_path)
        crop_df.insert(0, "crop_export_order", np.arange(len(crop_df), dtype=np.int64))
        if "split" in crop_df.columns:
            crop_df.insert(
                1,
                "split_crop_export_order",
                crop_df.groupby("split", sort=False).cumcount(),
            )
        if crop_stats_path.is_file():
            stats_df = pd.read_csv(crop_stats_path)
            join_keys = [key for key in ["split", "file_name"] if key in crop_df.columns and key in stats_df.columns]
            if len(join_keys) == 2:
                extra_columns = [
                    column for column in stats_df.columns
                    if column not in crop_df.columns or column in join_keys
                ]
                crop_df = crop_df.merge(
                    stats_df[extra_columns],
                    on=join_keys,
                    how="left",
                    sort=False,
                    validate="one_to_one",
                )
        if "file_name" in crop_df.columns and "split" in crop_df.columns:
            crop_df["output_label_file"] = [
                (Path("square_crops") / "labels" / str(split) / f"{Path(str(name)).stem}.txt").as_posix()
                for split, name in zip(crop_df["split"], crop_df["file_name"], strict=False)
            ]
        for source_column, output_column in {
            "training_image": "output_crop_image_file",
            "paired_whole_original_image": "output_original_whole_image_file",
            "paired_whole_image": "output_resized_whole_image_file",
            "paired_whole_high_resolution_image": "output_high_resolution_whole_image_file",
            "paired_whole_native_image": "output_native_whole_image_file",
        }.items():
            if source_column in crop_df.columns:
                crop_df[output_column] = [
                    (Path("square_crops") / str(value)).as_posix()
                    if str(value or "").strip() and str(value).casefold() != "nan"
                    else ""
                    for value in crop_df[source_column]
                ]
        if include_image_hashes:
            path_columns = {
                "training_image": "training_image_sha256",
                "paired_whole_original_image": "paired_whole_original_image_sha256",
                "paired_whole_image": "paired_whole_image_sha256",
                "paired_whole_high_resolution_image": "paired_whole_high_resolution_image_sha256",
                "paired_whole_native_image": "paired_whole_native_image_sha256",
            }
            for source_column, hash_column in path_columns.items():
                if source_column not in crop_df.columns:
                    continue
                crop_df[hash_column] = [
                    _sha256_file(crop_root / str(value))
                    if str(value or "") and (crop_root / str(value)).is_file()
                    else ""
                    for value in crop_df[source_column]
                ]
        crop_df.to_csv(crop_manifest_path, index=False)
        crop_rows_count = int(len(crop_df))
    else:
        pd.DataFrame(columns=[
            "crop_export_order", "split_crop_export_order", "split", "file_name",
            "source_image_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
        ]).to_csv(crop_manifest_path, index=False)
    created.append(crop_manifest_path)

    crop_records_path = bundle_root / "crop_records.jsonl"
    if crop_records_source.is_file():
        shutil.copyfile(crop_records_source, crop_records_path)
    else:
        crop_records_path.write_text("", encoding="utf-8")
    created.append(crop_records_path)

    source_processing_source = crop_root / "debug_logs" / "source_image_log.csv"
    source_processing_path = bundle_root / "source_processing.csv"
    if source_processing_source.is_file():
        shutil.copyfile(source_processing_source, source_processing_path)
    else:
        pd.DataFrame(columns=[
            "split", "source_index", "source_image_id", "processed_source_image",
            "saved_crops", "saved_positive_crops", "saved_negative_crops",
        ]).to_csv(source_processing_path, index=False)
    created.append(source_processing_path)

    annotation_rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        coco_path = crop_root / "mmdetection" / "annotations" / f"instances_{split_name}.json"
        if not coco_path.is_file():
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        image_by_id = {
            int(image.get("id")): image for image in list(coco.get("images", []) or [])
        }
        for split_annotation_order, annotation in enumerate(list(coco.get("annotations", []) or [])):
            image = image_by_id.get(int(annotation.get("image_id", -1)), {})
            bbox = list(annotation.get("bbox", []) or [])
            source_bbox = list(annotation.get("source_bbox_xyxy", []) or [])
            original_bbox = list(annotation.get("source_bbox_original_xyxy", []) or [])
            annotation_rows.append({
                "split": split_name,
                "split_annotation_export_order": int(split_annotation_order),
                "crop_file_name": image.get("file_name", ""),
                "source_image_id": image.get("source_image_id", ""),
                "source_study_id": image.get("source_study_id", ""),
                "coco_image_id": annotation.get("image_id"),
                "coco_annotation_id": annotation.get("id"),
                "category_id": annotation.get("category_id"),
                "source_annotation_id": annotation.get("source_annotation_id", ""),
                "source_annotation_row": annotation.get("source_annotation_row", ""),
                "crop_bbox_x": bbox[0] if len(bbox) >= 4 else "",
                "crop_bbox_y": bbox[1] if len(bbox) >= 4 else "",
                "crop_bbox_width": bbox[2] if len(bbox) >= 4 else "",
                "crop_bbox_height": bbox[3] if len(bbox) >= 4 else "",
                "source_bbox_fixed_xyxy_json": json.dumps(source_bbox),
                "source_bbox_original_dicom_xyxy_json": json.dumps(original_bbox),
                "annotation_json": json.dumps(
                    _json_safe(annotation), ensure_ascii=False, sort_keys=True
                ),
            })
    annotations_path = bundle_root / "crop_annotations.csv"
    pd.DataFrame(annotation_rows, columns=(list(annotation_rows[0]) if annotation_rows else [
        "split", "split_annotation_export_order", "crop_file_name", "source_image_id",
        "source_study_id", "coco_image_id", "coco_annotation_id", "category_id",
        "source_annotation_id", "source_annotation_row", "crop_bbox_x", "crop_bbox_y",
        "crop_bbox_width", "crop_bbox_height", "source_bbox_fixed_xyxy_json",
        "source_bbox_original_dicom_xyxy_json", "annotation_json",
    ])).to_csv(annotations_path, index=False)
    created.append(annotations_path)

    config_path = bundle_root / "resolved_config.yaml"
    config_text = (
        yaml.safe_dump(_json_safe(config), sort_keys=False)
        if yaml is not None
        else json.dumps(_json_safe(config), indent=2)
    )
    config_path.write_text(config_text, encoding="utf-8")
    created.append(config_path)

    software_path = bundle_root / "software_environment.json"
    software_path.write_text(
        json.dumps(_json_safe(_software_provenance()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    created.append(software_path)
    source_provenance_path = bundle_root / "source_metadata_provenance.json"
    source_provenance_path.write_text(
        json.dumps(_json_safe(_source_file_provenance(data_root)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    created.append(source_provenance_path)

    if bool(options.get("include_software_source_snapshot", True)):
        repository_root = Path(__file__).resolve().parents[2]
        software_sources = sorted(
            path
            for path in (repository_root / "src" / "vindr_mammo").rglob("*.py")
            if path.is_file()
        )
        software_sources.extend(
            path
            for path in [repository_root / "main.py", repository_root / "pyproject.toml"]
            if path.is_file()
        )
        software_source_rows = [
            {
                "repository_relative_path": path.relative_to(repository_root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in software_sources
        ]
        software_sources_path = bundle_root / "software_source_files.csv"
        pd.DataFrame(software_source_rows).to_csv(software_sources_path, index=False)
        created.append(software_sources_path)
        software_snapshot_path = bundle_root / "software_source_snapshot.zip"
        with zipfile.ZipFile(
            software_snapshot_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source_path in software_sources:
                archive.write(
                    source_path,
                    arcname=source_path.relative_to(repository_root).as_posix(),
                )
        created.append(software_snapshot_path)

    readme_path = bundle_root / "README.md"
    readme_path.write_text(
        """# Reproducibility bundle

This directory describes the exact source membership and saved crop dataset for this run.

- `source_images.csv` lists every included source mammogram, its split and membership order, source DICOM path, breast identity, and source record.
- `source_processing.csv` distinguishes split membership from sources actually processed or contributing saved positive/negative crops.
- `crops.csv` lists saved crops in export order, exact fixed-preprocessed crop coordinates and edge padding, original-DICOM/whole-image coordinate transforms, root-relative output paths, labels, and selection diagnostics.
- `crop_records.jsonl` is the lossless per-crop metadata record, including preprocessing and encoding details.
- `crop_annotations.csv` lists every exported annotation in crop, fixed-preprocessed source, and original-DICOM coordinate spaces.
- `resolved_config.yaml` is the complete effective configuration, including all seeds and pixel/crop settings.
- `software_environment.json` records the code revision and package/runtime versions.
- `software_source_files.csv` and `software_source_snapshot.zip` preserve the exact exporter source used for this run, including uncommitted package changes.
- `source_metadata_provenance.json` contains SHA-256 hashes of the three source CSV tables.
- `bundle_manifest.json` records schema, counts, policies, and artifact names.
- `checksums.sha256` verifies these compact bundle artifacts when metadata checksums are enabled.

Replay the source split and saved crop membership from `source_images.csv` and `crops.csv`; do not resample crops to reproduce this exact dataset. Crop coordinates are `(x0, y0, x1, y1)` in the fixed-preprocessed image and can extend beyond its bounds only where the recorded edge padding was applied. The resolved config defines how the original DICOM becomes that fixed-preprocessed image. The root `manifest.json` and `export_summary.json` provide the completed run summary.

Source-DICOM and exported-PNG hashing are optional and disabled by default because either requires another very large I/O pass. Their paths and identities are still recorded.
""",
        encoding="utf-8",
    )
    created.append(readme_path)

    manifest_path = bundle_root / "bundle_manifest.json"
    manifest_payload = {
        "schema_name": "vindr_mammo_exact_export_reproducibility",
        "schema_version": int(options.get("schema_version", 1) or 1),
        "source_image_count": int(len(source_rows)),
        "saved_crop_count": int(crop_rows_count),
        "saved_annotation_count": int(len(annotation_rows)),
        "source_dicom_sha256_included": include_source_hashes,
        "exported_image_sha256_included": include_image_hashes,
        "software_source_snapshot_included": bool(
            options.get("include_software_source_snapshot", True)
        ),
        "crop_coordinate_space": "fixed_preprocessed_xyxy",
        "crop_replay_policy": "use_recorded_windows_in_crop_export_order_do_not_resample",
        "training_balance_execution": config.get("square_crops", {}).get(
            "train_balance_execution", "configured_by_square_crops"
        ),
        "training_positive_policy": "keep_all_eligible_positive_windows",
        "training_negative_policy": "stream_toward_target_ratio_approximately",
        "artifacts": [
            *[path.name for path in created],
            "bundle_manifest.json",
            *(["checksums.sha256"] if bool(options.get("write_metadata_sha256", True)) else []),
        ],
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    created.append(manifest_path)

    checksums_path: Path | None = None
    if bool(options.get("write_metadata_sha256", True)):
        checksums_path = bundle_root / "checksums.sha256"
        checksum_lines = [
            f"{_sha256_file(path)}  {path.name}"
            for path in created
            if path.is_file()
        ]
        checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        created.append(checksums_path)

    summary = {
        "enabled": True,
        "schema_version": int(options.get("schema_version", 1) or 1),
        "output_dir": _path_as_posix(bundle_root),
        "source_image_count": int(len(source_rows)),
        "saved_crop_count": int(crop_rows_count),
        "saved_annotation_count": int(len(annotation_rows)),
        "metadata_sha256_written": checksums_path is not None,
        "source_dicom_sha256_included": include_source_hashes,
        "exported_image_sha256_included": include_image_hashes,
        "software_source_snapshot_included": bool(
            options.get("include_software_source_snapshot", True)
        ),
    }
    return summary, created


def _write_ultralytics_yaml(path: Path, *, train: str, val: str, test: str) -> None:
    """Write a portable Ultralytics detection YAML.

    The YAML intentionally does not include a ``path`` field. In recent
    Ultralytics versions, a relative ``path: .`` can be resolved against the
    process working directory instead of the YAML location. Omitting ``path``
    and writing train/val/test relative to the YAML file's parent directory
    avoids embedding OS-specific roots such as ``G:/`` or ``/mnt/t9``.
    """
    content = {
        "train": train,
        "val": val,
        "test": test,
        "names": {0: "mass"},
    }
    header = (
        "# VinDr-Mammo mass detection dataset for Ultralytics YOLO.\n"
        "# Portable YAML: no absolute Windows/Linux path and no `path: .`.\n"
        "# The train/val/test paths are relative to this YAML file.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"train: {train}\nval: {val}\ntest: {test}\nnames:\n  0: mass\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(content, f, sort_keys=False)


def _write_mmdetection_note(path: Path, root: Path) -> None:
    text = f"""MMDetection COCO-format paths for this export
================================================

Use these values in an MMDetection config:

data_root = r'{str(root)}'
metainfo = dict(classes=('mass',))

train_dataloader.dataset.ann_file = 'mmdetection/annotations/instances_train.json'
train_dataloader.dataset.data_prefix = dict(img='images/train/')
val_dataloader.dataset.ann_file = 'mmdetection/annotations/instances_val.json'
val_dataloader.dataset.data_prefix = dict(img='images/val/')
test_dataloader.dataset.ann_file = 'mmdetection/annotations/instances_test.json'
test_dataloader.dataset.data_prefix = dict(img='images/test/')

The annotation JSON files use standard COCO detection fields: images,
annotations, and categories. Category id 1 is mass. Extra per-image fields point
to preserved 16-bit images and metadata JSONL rows, but MMDetection can ignore them.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _empty_coco() -> dict[str, Any]:
    return {
        "info": {"description": "VinDr-Mammo mass detection export"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": list(COCO_CATEGORIES),
    }


def _coco_image_record(
    image_id: int,
    filename: str,
    width: int,
    height: int,
    record: dict[str, Any],
    save_info: dict[str, Any],
    split_name: str,
    *,
    dataset_name: str,
    crop_info: dict[str, Any] | None,
    preprocess_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preprocess_info = dict(preprocess_info or {})
    original_shape = list(preprocess_info.get("original_shape") or [])
    processed_shape = list(preprocess_info.get("processed_shape") or [])
    out = {
        "id": int(image_id),
        "file_name": filename,
        "width": int(width),
        "height": int(height),
        "source_image_id": str(record.get("image_id")),
        "source_study_id": str(record.get("study_id")),
        "export_split": split_name,
        "dataset": dataset_name,
        "preserved_16bit_path": save_info.get("preserved_16bit_path"),
        "paired_whole_key": save_info.get("paired_whole_key"),
        "paired_whole_original_image_path": save_info.get(
            "paired_whole_original_image_path"
        ),
        "paired_whole_image_path": save_info.get("paired_whole_image_path"),
        "paired_whole_high_resolution_image_path": save_info.get(
            "paired_whole_high_resolution_image_path"
        ),
        "paired_whole_native_image_path": save_info.get("paired_whole_native_image_path"),
        "paired_whole_canvas_mode": save_info.get("paired_whole_canvas_mode"),
        "paired_whole_canvas_width": save_info.get("paired_whole_canvas_width"),
        "paired_whole_canvas_height": save_info.get("paired_whole_canvas_height"),
        "paired_whole_pad_left": save_info.get("paired_whole_pad_left"),
        "paired_whole_pad_top": save_info.get("paired_whole_pad_top"),
        "paired_whole_pad_right": save_info.get("paired_whole_pad_right"),
        "paired_whole_pad_bottom": save_info.get("paired_whole_pad_bottom"),
        "paired_whole_scale_factor": save_info.get("paired_whole_scale_factor"),
        "paired_whole_high_resolution_canvas_mode": save_info.get(
            "paired_whole_high_resolution_canvas_mode"
        ),
        "paired_whole_high_resolution_canvas_width": save_info.get(
            "paired_whole_high_resolution_canvas_width"
        ),
        "paired_whole_high_resolution_canvas_height": save_info.get(
            "paired_whole_high_resolution_canvas_height"
        ),
        "paired_whole_high_resolution_pad_left": save_info.get(
            "paired_whole_high_resolution_pad_left"
        ),
        "paired_whole_high_resolution_pad_top": save_info.get(
            "paired_whole_high_resolution_pad_top"
        ),
        "paired_whole_high_resolution_pad_right": save_info.get(
            "paired_whole_high_resolution_pad_right"
        ),
        "paired_whole_high_resolution_pad_bottom": save_info.get(
            "paired_whole_high_resolution_pad_bottom"
        ),
        "rgb_scheme": save_info.get("rgb_scheme"),
        # source_bbox_xyxy annotations use this fixed-preprocessed coordinate
        # space.  A downstream evaluator must undo mirroring/cropping before
        # comparing them with the original finding CSV coordinates.
        "source_coordinate_space": "fixed_preprocessed",
        "source_preprocessing_mirrored": bool(preprocess_info.get("mirrored", False)),
        "source_preprocessing_crop_box_xyxy": preprocess_info.get("crop_box_xyxy"),
        "source_original_height": int(original_shape[0]) if len(original_shape) >= 2 else None,
        "source_original_width": int(original_shape[1]) if len(original_shape) >= 2 else None,
        "source_processed_height": int(processed_shape[0]) if len(processed_shape) >= 2 else None,
        "source_processed_width": int(processed_shape[1]) if len(processed_shape) >= 2 else None,
    }
    if crop_info:
        out["crop_window_xyxy"] = crop_info.get("window_xyxy")
        out["source_breast_key"] = crop_info.get("source_breast_key")
        out["source_breast_has_mass"] = crop_info.get("source_breast_has_mass")
        out["source_image_has_mass"] = crop_info.get("source_image_has_mass")
        out["negative_crop_source_policy"] = crop_info.get(
            "negative_crop_source_policy"
        )
        out["breast_fraction"] = crop_info.get("foreground_fraction")
        out["min_breast_fraction"] = crop_info.get(
            "min_breast_fraction_for_all_crops",
            crop_info.get("min_foreground_fraction"),
        )
        out["breast_fraction_mask_source"] = crop_info.get(
            "breast_fraction_mask_source"
        )
    return out


def _append_coco_annotations(
    coco: dict[str, Any],
    image_id: int,
    start_ann_id: int,
    boxes: torch.Tensor,
    *,
    source_annotation_ids: list[Any] | None = None,
    source_annotation_rows: list[Any] | None = None,
    source_boxes: torch.Tensor | None = None,
    source_original_boxes: torch.Tensor | None = None,
) -> int:
    count = 0
    source_annotation_ids = list(source_annotation_ids or [])
    source_annotation_rows = list(source_annotation_rows or [])
    source_box_list = _boxes_to_list(source_boxes)
    source_original_box_list = _boxes_to_list(source_original_boxes)
    for box_index, box in enumerate(_boxes_to_list(boxes)):
        x0, y0, x1, y1 = box
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        if w <= 0 or h <= 0:
            continue
        annotation = {
                "id": int(start_ann_id + count),
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            }
        if box_index < len(source_annotation_ids) and source_annotation_ids[box_index] is not None:
            annotation["source_annotation_id"] = source_annotation_ids[box_index]
        if box_index < len(source_annotation_rows) and source_annotation_rows[box_index] is not None:
            annotation["source_annotation_row"] = source_annotation_rows[box_index]
        if box_index < len(source_box_list):
            annotation["source_bbox_xyxy"] = [float(value) for value in source_box_list[box_index]]
            annotation["source_bbox_coordinate_space"] = "fixed_preprocessed"
        if box_index < len(source_original_box_list):
            annotation["source_bbox_original_xyxy"] = [
                float(value) for value in source_original_box_list[box_index]
            ]
            annotation["source_bbox_original_coordinate_space"] = "original_dicom"
        coco["annotations"].append(annotation)
        count += 1
    return count


def _fixed_preprocessed_boxes_to_original(
    boxes: torch.Tensor | None,
    preprocess_info: dict[str, Any] | None,
) -> torch.Tensor | None:
    """Undo fixed geometry preprocessing for downstream source-image audits."""
    if boxes is None:
        return None
    restored = boxes.detach().clone().to(torch.float32).reshape(-1, 4)
    if restored.numel() == 0:
        return restored
    info = dict(preprocess_info or {})
    processed_shape = list(info.get("processed_shape") or [])
    if bool(info.get("mirrored", False)) and len(processed_shape) >= 2:
        width = float(processed_shape[1])
        old_x0 = restored[:, 0].clone()
        old_x1 = restored[:, 2].clone()
        restored[:, 0] = width - old_x1
        restored[:, 2] = width - old_x0

    crop_box = list(info.get("crop_box_xyxy") or [])
    trim_box = list(info.get("trim_box_xyxy") or [])
    if len(crop_box) >= 4:
        offset_x, offset_y = float(crop_box[0]), float(crop_box[1])
    elif len(trim_box) >= 4:
        offset_x, offset_y = float(trim_box[0]), float(trim_box[1])
    else:
        offset_x, offset_y = 0.0, 0.0
    restored[:, [0, 2]] += offset_x
    restored[:, [1, 3]] += offset_y
    return restored


def _write_yolo_label_file(path: Path, boxes: torch.Tensor, *, width: int, height: int, save_empty: bool) -> None:
    rows = []
    for x0, y0, x1, y1 in _boxes_to_list(boxes):
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        if bw <= 0 or bh <= 0:
            continue
        cx = x0 + bw / 2.0
        cy = y0 + bh / 2.0
        rows.append(f"0 {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f}")
    if rows or save_empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _sample_stats_row(
    *,
    dataset_name: str,
    split: str,
    filename: str,
    image_width: int,
    image_height: int,
    boxes: torch.Tensor,
    record: dict[str, Any],
    crop_info: dict[str, Any] | None,
    save_info: dict[str, Any],
) -> dict[str, Any]:
    boxes_list = _boxes_to_list(boxes)
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes_list]
    image_area = max(float(image_width * image_height), 1.0)
    area_pcts = [100.0 * a / image_area for a in areas]
    row = {
        "dataset": dataset_name,
        "split": split,
        "file_name": filename,
        "source_image_id": str(record.get("image_id")),
        "source_study_id": str(record.get("study_id")),
        "official_split": record.get("split"),
        "laterality": record.get("laterality"),
        "view_position": record.get("view_position"),
        "width": int(image_width),
        "height": int(image_height),
        "num_mass_boxes": int(len(boxes_list)),
        "has_mass": int(len(boxes_list) > 0),
        "mean_mass_area_percentage": float(np.mean(area_pcts)) if area_pcts else 0.0,
        "max_mass_area_percentage": float(np.max(area_pcts)) if area_pcts else 0.0,
        "rgb_scheme": save_info.get("rgb_scheme"),
        "histogram_equalization_enabled": save_info.get("histogram_equalization_enabled"),
        "float32_image_path": save_info.get("float32_image_path", ""),
        "preserved_16bit_path": save_info.get("preserved_16bit_path", ""),
        "preserved_16bit_lo": save_info.get("preserved_16bit_lo", ""),
        "preserved_16bit_hi": save_info.get("preserved_16bit_hi", ""),
        "paired_whole_image_path": save_info.get("paired_whole_image_path", ""),
        "paired_whole_float32_image_path": save_info.get(
            "paired_whole_float32_image_path", ""
        ),
        "paired_whole_key": save_info.get("paired_whole_key", ""),
        "paired_whole_original_image_path": save_info.get("paired_whole_original_image_path", ""),
        "paired_whole_original_float32_image_path": save_info.get(
            "paired_whole_original_float32_image_path", ""
        ),
        "paired_whole_original_width": save_info.get("paired_whole_original_width", ""),
        "paired_whole_original_height": save_info.get("paired_whole_original_height", ""),
        "paired_whole_original_label_path": save_info.get("paired_whole_original_label_path", ""),
        "paired_whole_original_annotation_path": save_info.get("paired_whole_original_annotation_path", ""),
        "paired_whole_width": save_info.get("paired_whole_width", ""),
        "paired_whole_height": save_info.get("paired_whole_height", ""),
        "paired_whole_storage": save_info.get("paired_whole_storage", ""),
        "paired_whole_label_path": save_info.get("paired_whole_label_path", ""),
        "paired_whole_annotation_path": save_info.get("paired_whole_annotation_path", ""),
        "paired_whole_high_resolution_image_path": save_info.get("paired_whole_high_resolution_image_path", ""),
        "paired_whole_high_resolution_float32_image_path": save_info.get(
            "paired_whole_high_resolution_float32_image_path", ""
        ),
        "paired_whole_high_resolution_width": save_info.get("paired_whole_high_resolution_width", ""),
        "paired_whole_high_resolution_height": save_info.get("paired_whole_high_resolution_height", ""),
        "paired_whole_high_resolution_storage": save_info.get("paired_whole_high_resolution_storage", ""),
        "paired_whole_high_resolution_label_path": save_info.get("paired_whole_high_resolution_label_path", ""),
        "paired_whole_high_resolution_annotation_path": save_info.get("paired_whole_high_resolution_annotation_path", ""),
        "paired_whole_native_image_path": save_info.get("paired_whole_native_image_path", ""),
        "paired_whole_native_width": save_info.get("paired_whole_native_width", ""),
        "paired_whole_native_height": save_info.get("paired_whole_native_height", ""),
        "paired_whole_native_storage": save_info.get("paired_whole_native_storage", ""),
        "paired_whole_canvas_mode": save_info.get("paired_whole_canvas_mode", ""),
        "paired_whole_common_canvas": save_info.get("paired_whole_common_canvas", ""),
        "paired_whole_canvas_width": save_info.get("paired_whole_canvas_width", ""),
        "paired_whole_canvas_height": save_info.get("paired_whole_canvas_height", ""),
        "paired_whole_source_width": save_info.get("paired_whole_source_width", ""),
        "paired_whole_source_height": save_info.get("paired_whole_source_height", ""),
        "paired_whole_pad_left": save_info.get("paired_whole_pad_left", ""),
        "paired_whole_pad_top": save_info.get("paired_whole_pad_top", ""),
        "paired_whole_pad_right": save_info.get("paired_whole_pad_right", ""),
        "paired_whole_pad_bottom": save_info.get("paired_whole_pad_bottom", ""),
        "paired_whole_scale_factor": save_info.get("paired_whole_scale_factor", ""),
        "paired_whole_scale_x": save_info.get("paired_whole_scale_x", ""),
        "paired_whole_scale_y": save_info.get("paired_whole_scale_y", ""),
        "paired_whole_size_divisor": save_info.get("paired_whole_size_divisor", ""),
        "paired_whole_high_resolution_canvas_mode": save_info.get("paired_whole_high_resolution_canvas_mode", ""),
        "paired_whole_high_resolution_common_canvas": save_info.get("paired_whole_high_resolution_common_canvas", ""),
        "paired_whole_high_resolution_canvas_width": save_info.get("paired_whole_high_resolution_canvas_width", ""),
        "paired_whole_high_resolution_canvas_height": save_info.get("paired_whole_high_resolution_canvas_height", ""),
        "paired_whole_high_resolution_source_width": save_info.get("paired_whole_high_resolution_source_width", ""),
        "paired_whole_high_resolution_source_height": save_info.get("paired_whole_high_resolution_source_height", ""),
        "paired_whole_high_resolution_pad_left": save_info.get("paired_whole_high_resolution_pad_left", ""),
        "paired_whole_high_resolution_pad_top": save_info.get("paired_whole_high_resolution_pad_top", ""),
        "paired_whole_high_resolution_pad_right": save_info.get("paired_whole_high_resolution_pad_right", ""),
        "paired_whole_high_resolution_pad_bottom": save_info.get("paired_whole_high_resolution_pad_bottom", ""),
        "paired_whole_high_resolution_size_divisor": save_info.get("paired_whole_high_resolution_size_divisor", ""),
    }
    if crop_info:
        window = crop_info.get("window_xyxy")
        row.update(
            {
                "crop_window_xyxy": "" if window is None else str(tuple(int(v) for v in window)),
                "crop_window_x0": "" if window is None else int(window[0]),
                "crop_window_y0": "" if window is None else int(window[1]),
                "crop_window_x1": "" if window is None else int(window[2]),
                "crop_window_y1": "" if window is None else int(window[3]),
                "crop_pad_left": crop_info.get("pad_left", 0),
                "crop_pad_top": crop_info.get("pad_top", 0),
                "crop_pad_right": crop_info.get("pad_right", 0),
                "crop_pad_bottom": crop_info.get("pad_bottom", 0),
                "source_index": crop_info.get("source_index", ""),
                "crop_mode": crop_info.get("crop_mode"),
                "deterministic_selection_mode": crop_info.get("deterministic_selection_mode", ""),
                "negative_fraction_selection": crop_info.get("negative_fraction_selection", ""),
                "negative_candidate_count": crop_info.get("negative_candidate_count", ""),
                "negative_keep_fraction": crop_info.get("negative_keep_fraction", ""),
                "negative_selected_count": crop_info.get("negative_selected_count", ""),
                "negative_achieved_keep_fraction": crop_info.get("negative_achieved_keep_fraction", ""),
                "source_image_has_finding": crop_info.get("source_image_has_finding", ""),
                "source_image_has_mass": crop_info.get("source_image_has_mass", ""),
                "negative_crop_source_policy": crop_info.get("negative_crop_source_policy", ""),
                "source_breast_key": crop_info.get("source_breast_key", ""),
                "source_breast_has_mass": crop_info.get("source_breast_has_mass", ""),
                "source_breast_ratio_selection": crop_info.get("source_breast_ratio_selection", ""),
                "source_breast_mass_candidate_count": crop_info.get("source_breast_mass_candidate_count", ""),
                "source_breast_negative_candidate_count": crop_info.get("source_breast_negative_candidate_count", ""),
                "source_breast_mandatory_positive_window_count": crop_info.get("source_breast_mandatory_positive_window_count", ""),
                "source_breast_selected_mass_count": crop_info.get("source_breast_selected_mass_count", ""),
                "source_breast_selected_negative_count": crop_info.get("source_breast_selected_negative_count", ""),
                "source_breast_target_mass_ratio": crop_info.get("source_breast_target_mass_ratio", ""),
                "source_breast_achieved_mass_ratio": crop_info.get("source_breast_achieved_mass_ratio", ""),
                "is_positive_window": crop_info.get("is_positive_window", ""),
                "requested_positive": crop_info.get("requested_positive"),
                "accepted": crop_info.get("accepted"),
                "foreground_filter_enabled": crop_info.get("foreground_filter_enabled", ""),
                "all_crop_breast_fraction_filter_enabled": crop_info.get("all_crop_breast_fraction_filter_enabled", ""),
                "breast_fraction_mask_source": crop_info.get("breast_fraction_mask_source", ""),
                "foreground_fraction": crop_info.get("foreground_fraction", ""),
                "min_foreground_fraction": crop_info.get("min_foreground_fraction", ""),
                "min_breast_fraction_for_all_crops": crop_info.get("min_breast_fraction_for_all_crops", ""),
                "breast_fraction_comparison_for_all_crops": crop_info.get("breast_fraction_comparison_for_all_crops", ""),
                "negative_foreground_filter_enabled": crop_info.get("negative_foreground_filter_enabled", ""),
                "negative_foreground_fraction": crop_info.get("negative_foreground_fraction", ""),
                "negative_min_foreground_fraction": crop_info.get("negative_min_foreground_fraction", ""),
                "bbox_safe_boundary_margin_fraction": crop_info.get("bbox_safe_boundary_margin_fraction", ""),
                "bbox_safe_boundary_margin_px": crop_info.get("bbox_safe_boundary_margin_px", ""),
                "bbox_safe_visible_boxes": crop_info.get("bbox_safe_visible_boxes", ""),
                "bbox_safe_boxes_inside_margin": crop_info.get("bbox_safe_boxes_inside_margin", ""),
                "bbox_safe_foreground_fraction": crop_info.get("bbox_safe_foreground_fraction", ""),
                "bbox_safe_score": crop_info.get("bbox_safe_score", ""),
                "bbox_safe_margin_ok": crop_info.get("bbox_safe_margin_ok", ""),
                "bbox_safe_export_margin_ok": crop_info.get("bbox_safe_export_margin_ok", ""),
                "bbox_safe_exported_boxes": crop_info.get("bbox_safe_exported_boxes", ""),
                "bbox_safe_exported_boxes_inside_margin": crop_info.get("bbox_safe_exported_boxes_inside_margin", ""),
                "bbox_safe_failure_reason": crop_info.get("bbox_safe_failure_reason", ""),
            }
        )
    return row


def _sample_metadata_record(
    *,
    dataset_name: str,
    split: str,
    filename: str,
    target: dict[str, Any],
    record: dict[str, Any],
    boxes: torch.Tensor,
    save_info: dict[str, Any],
    crop_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep all relevant per-sample metadata, including full CSV rows."""
    return {
        "dataset": dataset_name,
        "split": split,
        "file_name": filename,
        "source_image_id": str(record.get("image_id")),
        "source_study_id": str(record.get("study_id")),
        "source_dicom_path": str(record.get("dicom_path", "")),
        "training_image": save_info.get("image_path"),
        "float32_training_image": save_info.get("float32_image_path"),
        "preserved_16bit_image": save_info.get("preserved_16bit_path"),
        "paired_whole_image": save_info.get("paired_whole_image_path"),
        "paired_whole_float32_image": save_info.get(
            "paired_whole_float32_image_path"
        ),
        "paired_whole_original_image": save_info.get(
            "paired_whole_original_image_path"
        ),
        "paired_whole_original_float32_image": save_info.get(
            "paired_whole_original_float32_image_path"
        ),
        "paired_whole_high_resolution_image": save_info.get(
            "paired_whole_high_resolution_image_path"
        ),
        "paired_whole_high_resolution_float32_image": save_info.get(
            "paired_whole_high_resolution_float32_image_path"
        ),
        "paired_whole_native_image": save_info.get("paired_whole_native_image_path"),
        "paired_whole_key": save_info.get("paired_whole_key"),
        "encoding": save_info,
        "export_boxes_xyxy": _boxes_to_list(boxes),
        "crop_info": crop_info or {},
        "breast_annotation_row": target.get("breast_annotation", record),
        "metadata_csv_rows": target.get("metadata", []),
        "dicom_meta": target.get("dicom_meta", {}),
        "all_finding_rows": target.get("findings", []),
        "mass_finding_rows": target.get("mass", {}).get("findings", []),
        "breast_birads": target.get("breast_birads"),
        "breast_density": target.get("breast_density"),
        "preprocess_info": target.get("preprocessing", {}),
        "square_crop": target.get("square_crop", {}),
    }


def _flatten_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    enc = row.get("encoding", {}) or {}
    crop = row.get("crop_info", {}) or {}
    meta_rows = row.get("metadata_csv_rows", []) or []
    dicom_meta = row.get("dicom_meta", {}) or {}
    preprocess = row.get("preprocess_info", {}) or {}
    original_shape = list(preprocess.get("original_shape") or [])
    processed_shape = list(preprocess.get("processed_shape") or [])
    first_meta = meta_rows[0] if meta_rows else {}
    return {
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "file_name": row.get("file_name"),
        "source_image_id": row.get("source_image_id"),
        "source_study_id": row.get("source_study_id"),
        "source_dicom_path": row.get("source_dicom_path"),
        "training_image": row.get("training_image"),
        "float32_training_image": row.get("float32_training_image"),
        "preserved_16bit_image": row.get("preserved_16bit_image"),
        "paired_whole_image": row.get("paired_whole_image"),
        "paired_whole_float32_image": row.get("paired_whole_float32_image"),
        "paired_whole_original_image": row.get("paired_whole_original_image"),
        "paired_whole_original_float32_image": row.get(
            "paired_whole_original_float32_image"
        ),
        "paired_whole_high_resolution_image": row.get(
            "paired_whole_high_resolution_image"
        ),
        "paired_whole_high_resolution_float32_image": row.get(
            "paired_whole_high_resolution_float32_image"
        ),
        "paired_whole_native_image": row.get("paired_whole_native_image"),
        "paired_whole_key": row.get("paired_whole_key"),
        "paired_whole_original_label": enc.get("paired_whole_original_label_path", ""),
        "paired_whole_original_annotation": enc.get("paired_whole_original_annotation_path", ""),
        "paired_whole_label": enc.get("paired_whole_label_path", ""),
        "paired_whole_annotation": enc.get("paired_whole_annotation_path", ""),
        "paired_whole_high_resolution_label": enc.get("paired_whole_high_resolution_label_path", ""),
        "paired_whole_high_resolution_annotation": enc.get("paired_whole_high_resolution_annotation_path", ""),
        "source_coordinate_space": "fixed_preprocessed",
        "source_preprocessing_mirrored": bool(preprocess.get("mirrored", False)),
        "source_preprocessing_crop_box_xyxy": preprocess.get("crop_box_xyxy", ""),
        "source_original_height": original_shape[0] if len(original_shape) >= 2 else "",
        "source_original_width": original_shape[1] if len(original_shape) >= 2 else "",
        "source_processed_height": processed_shape[0] if len(processed_shape) >= 2 else "",
        "source_processed_width": processed_shape[1] if len(processed_shape) >= 2 else "",
        "breast_birads": row.get("breast_birads"),
        "breast_density": row.get("breast_density"),
        "num_export_boxes": len(row.get("export_boxes_xyxy", []) or []),
        "crop_window_xyxy": crop.get("window_xyxy", ""),
        "crop_mode": crop.get("crop_mode", ""),
        "source_breast_key": crop.get("source_breast_key", ""),
        "source_breast_has_mass": crop.get("source_breast_has_mass", ""),
        "source_breast_target_mass_ratio": crop.get("source_breast_target_mass_ratio", ""),
        "source_breast_achieved_mass_ratio": crop.get("source_breast_achieved_mass_ratio", ""),
        "foreground_filter_enabled": crop.get("foreground_filter_enabled", ""),
        "all_crop_breast_fraction_filter_enabled": crop.get("all_crop_breast_fraction_filter_enabled", ""),
        "breast_fraction_mask_source": crop.get("breast_fraction_mask_source", ""),
        "foreground_fraction": crop.get("foreground_fraction", ""),
        "min_foreground_fraction": crop.get("min_foreground_fraction", ""),
        "min_breast_fraction_for_all_crops": crop.get("min_breast_fraction_for_all_crops", ""),
        "breast_fraction_comparison_for_all_crops": crop.get("breast_fraction_comparison_for_all_crops", ""),
        "bbox_safe_boundary_margin_fraction": crop.get("bbox_safe_boundary_margin_fraction", ""),
        "bbox_safe_foreground_fraction": crop.get("bbox_safe_foreground_fraction", ""),
        "bbox_safe_score": crop.get("bbox_safe_score", ""),
        "manufacturer": first_meta.get("Manufacturer", first_meta.get("manufacturer", dicom_meta.get("Manufacturer", ""))),
        "manufacturer_model_name": first_meta.get("ManufacturerModelName", first_meta.get("manufacturer_model_name", dicom_meta.get("ManufacturerModelName", ""))),
        "photometric_interpretation": dicom_meta.get("PhotometricInterpretation", ""),
        "rgb_scheme": enc.get("rgb_scheme", ""),
        "histogram_equalization_enabled": enc.get("histogram_equalization_enabled", ""),
        "preserved_16bit_lo": enc.get("preserved_16bit_lo", ""),
        "preserved_16bit_hi": enc.get("preserved_16bit_hi", ""),
        "paired_whole_width": enc.get("paired_whole_width", ""),
        "paired_whole_height": enc.get("paired_whole_height", ""),
        "paired_whole_storage": enc.get("paired_whole_storage", ""),
        "paired_whole_original_width": enc.get("paired_whole_original_width", ""),
        "paired_whole_original_height": enc.get("paired_whole_original_height", ""),
        "paired_whole_high_resolution_width": enc.get("paired_whole_high_resolution_width", ""),
        "paired_whole_high_resolution_height": enc.get("paired_whole_high_resolution_height", ""),
        "paired_whole_high_resolution_storage": enc.get("paired_whole_high_resolution_storage", ""),
        "paired_whole_native_width": enc.get("paired_whole_native_width", ""),
        "paired_whole_native_height": enc.get("paired_whole_native_height", ""),
        "paired_whole_native_storage": enc.get("paired_whole_native_storage", ""),
        "paired_whole_canvas_mode": enc.get("paired_whole_canvas_mode", ""),
        "paired_whole_common_canvas": enc.get("paired_whole_common_canvas", ""),
        "paired_whole_canvas_width": enc.get("paired_whole_canvas_width", ""),
        "paired_whole_canvas_height": enc.get("paired_whole_canvas_height", ""),
        "paired_whole_source_width": enc.get("paired_whole_source_width", ""),
        "paired_whole_source_height": enc.get("paired_whole_source_height", ""),
        "paired_whole_pad_left": enc.get("paired_whole_pad_left", ""),
        "paired_whole_pad_top": enc.get("paired_whole_pad_top", ""),
        "paired_whole_pad_right": enc.get("paired_whole_pad_right", ""),
        "paired_whole_pad_bottom": enc.get("paired_whole_pad_bottom", ""),
        "paired_whole_scale_factor": enc.get("paired_whole_scale_factor", ""),
        "paired_whole_scale_x": enc.get("paired_whole_scale_x", ""),
        "paired_whole_scale_y": enc.get("paired_whole_scale_y", ""),
        "paired_whole_size_divisor": enc.get("paired_whole_size_divisor", ""),
        "paired_whole_high_resolution_canvas_mode": enc.get("paired_whole_high_resolution_canvas_mode", ""),
        "paired_whole_high_resolution_common_canvas": enc.get("paired_whole_high_resolution_common_canvas", ""),
        "paired_whole_high_resolution_canvas_width": enc.get("paired_whole_high_resolution_canvas_width", ""),
        "paired_whole_high_resolution_canvas_height": enc.get("paired_whole_high_resolution_canvas_height", ""),
        "paired_whole_high_resolution_source_width": enc.get("paired_whole_high_resolution_source_width", ""),
        "paired_whole_high_resolution_source_height": enc.get("paired_whole_high_resolution_source_height", ""),
        "paired_whole_high_resolution_pad_left": enc.get("paired_whole_high_resolution_pad_left", ""),
        "paired_whole_high_resolution_pad_top": enc.get("paired_whole_high_resolution_pad_top", ""),
        "paired_whole_high_resolution_pad_right": enc.get("paired_whole_high_resolution_pad_right", ""),
        "paired_whole_high_resolution_pad_bottom": enc.get("paired_whole_high_resolution_pad_bottom", ""),
        "paired_whole_high_resolution_scale_factor": enc.get("paired_whole_high_resolution_scale_factor", ""),
        "paired_whole_high_resolution_size_divisor": enc.get("paired_whole_high_resolution_size_divisor", ""),
    }


def _crop_location_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one crop's exact source and whole-image coordinate transforms."""
    crop = dict(row.get("crop_info", {}) or {})
    window = list(crop.get("window_xyxy") or [])
    if len(window) < 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in window[:4]]
    preprocess = dict(row.get("preprocess_info", {}) or {})
    encoding = dict(row.get("encoding", {}) or {})
    processed_shape = list(preprocess.get("processed_shape") or crop.get("source_image_shape") or [])
    original_shape = list(preprocess.get("original_shape") or [])
    processed_h = int(processed_shape[0]) if len(processed_shape) >= 2 else None
    processed_w = int(processed_shape[1]) if len(processed_shape) >= 2 else None

    source_x0, source_x1 = x0, x1
    mirrored = bool(preprocess.get("mirrored", False))
    if mirrored and processed_w is not None:
        source_x0, source_x1 = float(processed_w) - x1, float(processed_w) - x0
    crop_box = list(preprocess.get("crop_box_xyxy") or preprocess.get("trim_box_xyxy") or [])
    offset_x = float(crop_box[0]) if len(crop_box) >= 4 else 0.0
    offset_y = float(crop_box[1]) if len(crop_box) >= 4 else 0.0
    original_x0, original_x1 = source_x0 + offset_x, source_x1 + offset_x
    original_y0, original_y1 = y0 + offset_y, y1 + offset_y

    pad_left = float(encoding.get("paired_whole_pad_left", 0) or 0)
    pad_top = float(encoding.get("paired_whole_pad_top", 0) or 0)
    scale_x = float(encoding.get("paired_whole_scale_x", 1.0) or 1.0)
    scale_y = float(encoding.get("paired_whole_scale_y", 1.0) or 1.0)
    resized_x0, resized_y0 = x0 + pad_left, y0 + pad_top
    resized_x1, resized_y1 = x1 + pad_left, y1 + pad_top
    high_pad_left = float(
        encoding.get("paired_whole_high_resolution_pad_left", pad_left) or 0
    )
    high_pad_top = float(
        encoding.get("paired_whole_high_resolution_pad_top", pad_top) or 0
    )
    high_x0, high_y0 = x0 + high_pad_left, y0 + high_pad_top
    high_x1, high_y1 = x1 + high_pad_left, y1 + high_pad_top

    return {
        "split": row.get("split"),
        "file_name": row.get("file_name"),
        "training_image": row.get("training_image"),
        "float32_training_image": row.get("float32_training_image"),
        "paired_whole_original_image": row.get("paired_whole_original_image"),
        "paired_whole_original_float32_image": row.get(
            "paired_whole_original_float32_image"
        ),
        "paired_whole_image": row.get("paired_whole_image"),
        "paired_whole_float32_image": row.get("paired_whole_float32_image"),
        "paired_whole_high_resolution_image": row.get(
            "paired_whole_high_resolution_image"
        ),
        "paired_whole_high_resolution_float32_image": row.get(
            "paired_whole_high_resolution_float32_image"
        ),
        "paired_whole_native_image": row.get("paired_whole_native_image"),
        "paired_whole_key": row.get("paired_whole_key"),
        "source_image_id": row.get("source_image_id"),
        "source_study_id": row.get("source_study_id"),
        "source_coordinate_space": "fixed_preprocessed",
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_width": int(x1 - x0),
        "crop_height": int(y1 - y0),
        "crop_pad_left": int(crop.get("pad_left", 0) or 0),
        "crop_pad_top": int(crop.get("pad_top", 0) or 0),
        "crop_pad_right": int(crop.get("pad_right", 0) or 0),
        "crop_pad_bottom": int(crop.get("pad_bottom", 0) or 0),
        "source_processed_width": processed_w,
        "source_processed_height": processed_h,
        "source_intersection_x0": max(0, int(x0)),
        "source_intersection_y0": max(0, int(y0)),
        "source_intersection_x1": min(processed_w, int(x1)) if processed_w is not None else int(x1),
        "source_intersection_y1": min(processed_h, int(y1)) if processed_h is not None else int(y1),
        "source_preprocessing_mirrored": mirrored,
        "source_original_width": int(original_shape[1]) if len(original_shape) >= 2 else None,
        "source_original_height": int(original_shape[0]) if len(original_shape) >= 2 else None,
        "original_crop_x0": original_x0,
        "original_crop_y0": original_y0,
        "original_crop_x1": original_x1,
        "original_crop_y1": original_y1,
        "whole_original_crop_x0": x0,
        "whole_original_crop_y0": y0,
        "whole_original_crop_x1": x1,
        "whole_original_crop_y1": y1,
        "whole_high_resolution_crop_x0": high_x0,
        "whole_high_resolution_crop_y0": high_y0,
        "whole_high_resolution_crop_x1": high_x1,
        "whole_high_resolution_crop_y1": high_y1,
        # Backward-compatible coordinate aliases for older loaders.
        "whole_native_crop_x0": high_x0,
        "whole_native_crop_y0": high_y0,
        "whole_native_crop_x1": high_x1,
        "whole_native_crop_y1": high_y1,
        "whole_resized_crop_x0": resized_x0 * scale_x,
        "whole_resized_crop_y0": resized_y0 * scale_y,
        "whole_resized_crop_x1": resized_x1 * scale_x,
        "whole_resized_crop_y1": resized_y1 * scale_y,
        "whole_pad_left": int(pad_left),
        "whole_pad_top": int(pad_top),
        "whole_pad_right": int(encoding.get("paired_whole_pad_right", 0) or 0),
        "whole_pad_bottom": int(encoding.get("paired_whole_pad_bottom", 0) or 0),
        "whole_canvas_mode": encoding.get("paired_whole_canvas_mode"),
        "whole_common_canvas": encoding.get("paired_whole_common_canvas"),
        "whole_canvas_width": encoding.get("paired_whole_canvas_width"),
        "whole_canvas_height": encoding.get("paired_whole_canvas_height"),
        "whole_source_width": encoding.get("paired_whole_source_width"),
        "whole_source_height": encoding.get("paired_whole_source_height"),
        "whole_size_divisor": encoding.get("paired_whole_size_divisor"),
        "whole_resized_scale_factor": encoding.get("paired_whole_scale_factor"),
        "whole_resized_scale_x": scale_x,
        "whole_resized_scale_y": scale_y,
        "whole_high_resolution_pad_left": int(high_pad_left),
        "whole_high_resolution_pad_top": int(high_pad_top),
        "whole_high_resolution_pad_right": int(
            encoding.get("paired_whole_high_resolution_pad_right", 0) or 0
        ),
        "whole_high_resolution_pad_bottom": int(
            encoding.get("paired_whole_high_resolution_pad_bottom", 0) or 0
        ),
        "whole_high_resolution_canvas_mode": encoding.get(
            "paired_whole_high_resolution_canvas_mode"
        ),
        "whole_high_resolution_common_canvas": encoding.get(
            "paired_whole_high_resolution_common_canvas"
        ),
        "whole_high_resolution_canvas_width": encoding.get(
            "paired_whole_high_resolution_canvas_width"
        ),
        "whole_high_resolution_canvas_height": encoding.get(
            "paired_whole_high_resolution_canvas_height"
        ),
        "whole_high_resolution_source_width": encoding.get(
            "paired_whole_high_resolution_source_width"
        ),
        "whole_high_resolution_source_height": encoding.get(
            "paired_whole_high_resolution_source_height"
        ),
        "whole_high_resolution_size_divisor": encoding.get(
            "paired_whole_high_resolution_size_divisor"
        ),
    }


# -----------------------------------------------------------------------------
# Summaries and small helpers
# -----------------------------------------------------------------------------



def _write_completion_files(
    *,
    output_root: Path,
    data_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    created_files: list[Path],
    stage_timings: list[dict[str, Any]],
    started_at: str,
    total_duration_seconds: float,
) -> tuple[dict[str, Any], list[Path]]:
    """Write final completion markers after all export work succeeds.

    ``manifest.json`` is structured and machine-readable. ``EXPORT_DONE.txt`` is
    intentionally simple so you can quickly check completion from File Explorer,
    PowerShell, or a text editor. These files are written only at the end of a
    successful export run.
    """
    finished_at = _utc_now_iso()
    readme_path = _write_dataset_readme(
        output_root=output_root,
        config=config,
        summary=summary,
    )
    created_files_with_readme = [*created_files, readme_path]
    file_counts = _collect_export_file_counts(output_root, config=config)
    expected_files = _expected_completion_files(output_root, config)
    expected_status = [
        {"path": _path_as_posix(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        for path in expected_files
    ]

    manifest = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "total_duration_seconds": float(total_duration_seconds),
        "total_duration_minutes": float(total_duration_seconds) / 60.0,
        "data_root": _path_as_posix(data_root),
        "output_root": _path_as_posix(output_root),
        "summary": summary,
        "stage_timings": stage_timings,
        "file_counts": file_counts,
        "expected_files": expected_status,
        "created_files_count": len(created_files_with_readme),
        "created_files_tail": [_path_as_posix(p) for p in created_files_with_readme[-50:]],
        "config_snapshot": config,
        "source_file_provenance": _source_file_provenance(data_root),
        "software_provenance": _software_provenance(),
    }

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)

    done_path = output_root / "EXPORT_DONE.txt"
    lines = [
        "VinDr-Mammo export completed successfully.",
        f"Started at: {started_at}",
        f"Finished at: {finished_at}",
        f"Total duration seconds: {total_duration_seconds:.3f}",
        f"Total duration minutes: {total_duration_seconds / 60.0:.3f}",
        f"Output root: {_path_as_posix(output_root)}",
        "",
        "Stage timings:",
    ]
    for t in stage_timings:
        lines.append(f"  - {t.get('name')}: {float(t.get('duration_seconds', 0.0)):.3f} s [{t.get('status')}]")
    lines.extend(["", "Key output counts:"])
    for key, value in file_counts.items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("See manifest.json for full details.")
    done_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return manifest, [readme_path, manifest_path, done_path]


def _write_dataset_readme(
    *,
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write a self-contained guide to the resolved dataset export."""
    provenance = dict(config.get("study_preset_provenance", {}) or {})
    crop_cfg = dict(config.get("square_crops", {}) or {})
    crop_policy = dict(config.get("crop_annotation_policy", {}) or {})
    paired_cfg = dict(config.get("paired_whole_images", {}) or {})
    image_cfg = dict(config.get("image", {}) or {})
    preprocess_cfg = dict(config.get("preprocess", {}) or {})
    pipeline = dict((config.get("image_export", {}) or {}).get("custom_channel_pipeline", {}) or {})
    review_cfg = dict(config.get("dataset_review", {}) or {})
    annotation_report_cfg = dict(config.get("annotation_geometry_report", {}) or {})
    reproducibility_cfg = dict(config.get("reproducibility_bundle", {}) or {})
    export_cfg = dict(config.get("export", {}) or {})
    float32_cfg = dict(config.get("float32_export", {}) or {})
    float32_selected_variants = [
        variant
        for variant in FLOAT32_EXPORT_VARIANTS
        if float32_export_variant_selected(config, variant)
    ]
    crop_size = int(crop_cfg.get("crop_size", 1024) or 1024)
    stride = int(crop_cfg.get("stride", 512) or 512)
    label_visibility = float(crop_policy.get("min_box_visibility", 0.30))
    source_visibility = float(preprocess_cfg.get("min_box_visibility_after_crop", 0.30))
    percentile = list(image_cfg.get("percentile_range", [0.5, 99.5]) or [0.5, 99.5])
    channel_steps = {
        channel: [str(step.get("op", "none")) for step in dict(pipeline.get(channel, {}) or {}).get("steps", [])]
        for channel in ["R", "G", "B"]
    }
    channels_identical = bool(channel_steps["R"] == channel_steps["G"] == channel_steps["B"])
    preset_key = str(provenance.get("preset_key", "custom configuration"))
    preset_version = provenance.get("preset_version")
    preset_text = f"{preset_key} (version {preset_version})" if preset_version is not None else preset_key
    square_summary = dict(summary.get("square_crops", {}) or {})
    split_summary = dict(square_summary.get("splits", {}) or {})
    split_lines = [
        f"- `{split}`: {int(dict(values or {}).get('num_images', 0)):,} crops"
        for split, values in split_summary.items()
    ] or ["- Counts are available in `export_summary.json` and `stats/summary.csv`."]
    grouped_layout = uses_grouped_dataset_layout(config)

    whole_target_w = int(paired_cfg.get("target_width", paired_cfg.get("size", 1024)) or 1024)
    whole_target_h = int(paired_cfg.get("target_height", paired_cfg.get("size", 1024)) or 1024)
    resized_geometry_cfg = _paired_resized_geometry_config(paired_cfg)
    resized_canvas_mode = str(
        resized_geometry_cfg.get("canvas_mode", "per_image_square")
        or "per_image_square"
    ).casefold().strip()
    if resized_canvas_mode in {"fixed", "fixed_canvas", "dataset_fixed"}:
        resized_geometry_text = (
            f"a shared {int(resized_geometry_cfg.get('canvas_width', 0) or 0)} x "
            f"{int(resized_geometry_cfg.get('canvas_height', 0) or 0)} canvas"
        )
    else:
        resized_geometry_text = "its own top-left-anchored square canvas"
    whole_lines: list[str] = []
    if _paired_original_enabled(paired_cfg):
        whole_lines.append(
            "- `images/original/<split>/<source key>.png`: one fixed-preprocessed whole image at its original H x W, with no padding or resize."
            if grouped_layout else
            "- `square_crops/whole_images_original/<split>/<source key>.png`: one fixed-preprocessed whole image at its original H x W, with no padding or resize."
        )
    if _paired_resized_enabled(paired_cfg):
        for resized_variant in resized_variant_configs(paired_cfg):
            resolution = str(resized_variant["name"])
            path_text = (
                f"images/resized/{resolution}/<split>/<source key>.png"
                if grouped_layout else
                f"square_crops/whole_images_{resolution}/<split>/<source key>.png"
            )
            whole_lines.append(
                f"- `{path_text}`: one compact, aspect-preserving whole image per source mammogram, padded to {resized_geometry_text} and then resized to {int(resized_variant['width'])} x {int(resized_variant['height'])}."
            )
    if _paired_high_resolution_enabled(paired_cfg):
        high_geometry_cfg = _paired_high_resolution_geometry_config(paired_cfg)
        high_canvas_mode = str(
            high_geometry_cfg.get("canvas_mode", "per_image_square")
            or "per_image_square"
        ).casefold().strip()
        if high_canvas_mode in {"fixed", "fixed_canvas", "dataset_fixed"}:
            high_geometry_text = (
                f"the shared {int(high_geometry_cfg.get('canvas_width', 0) or 0)} x "
                f"{int(high_geometry_cfg.get('canvas_height', 0) or 0)} "
                "top-left-anchored canvas (padding only on the right and bottom)"
            )
        else:
            high_geometry_text = "its own top-left-anchored square canvas"
        whole_lines.append(
            f"- `{'images/high_resolution' if grouped_layout else 'square_crops/whole_images_high_resolution'}/<split>/<source key>.png`: one high-resolution whole image per source mammogram padded independently to {high_geometry_text}, with no resize."
        )
    if not bool(paired_cfg.get("enabled", False)):
        whole_lines = ["- Paired whole-image export was disabled for this run."]

    float_example = (
        "images/float32/resized/1024x1024/train/<source key>.pt"
        if grouped_layout else
        "square_crops/float32/whole_images/train/<source key>.pt"
    )
    readme = f"""# VinDr-Mammo exported dataset

This dataset was generated successfully with preset/configuration `{preset_text}`. The exact resolved configuration is stored in `metadata/export_config_resolved.yaml`; `manifest.json` records completion, provenance, timings, and file counts.

## Pixel processing

- DICOM normalization: `{image_cfg.get('normalize', 'none')}` with percentile range `{percentile}`.
- Breast geometry: crop=`{str(bool(preprocess_cfg.get('crop_breast', False))).lower()}`, mask outside=`{str(bool(preprocess_cfg.get('mask_outside_breast', False))).lower()}`, canonical right-to-left mirroring=`{str(bool(preprocess_cfg.get('mirror_right_to_left', False))).lower()}`.
- RGB channel operations: R={channel_steps['R']}, G={channel_steps['G']}, B={channel_steps['B']}.
- Identical output channel recipes: `{str(channels_identical).lower()}`. Identical recipes produce replicated grayscale RGB when their sources are also identical.
- CLAHE and other steps marked `apply_before_crop` are evaluated on the fixed-preprocessed whole image before a crop is extracted.

## Crop data

- Window size: `{crop_size} x {crop_size}` pixels.
- Stride: `{stride}` pixels.
- Required geometry divisor: `{int(crop_cfg.get('size_divisor', 1) or 1)}`. The exporter rejects incompatible crop/stride settings before writing samples.
- Edge policy: `{crop_cfg.get('edge_policy', 'edge_align')}`. Edge windows can extend outside the source and are filled with the configured crop padding value.
- Saved-label inclusion rule (`crop_annotation_policy.min_box_visibility`): when partial annotations are enabled, a Mass annotation is clipped and written to YOLO/COCO if at least `{label_visibility:.1%}` of its original area is visible inside the square crop. Comparison is greater than or equal (`>=`).
- Initial breast-crop safeguard (`preprocess.min_box_visibility_after_crop`): retains a source annotation if at least `{source_visibility:.1%}` remains after the earlier breast bounding-box crop. This is a separate preprocessing stage, not the square-crop label threshold.
- The GUI's **PREVIEW ONLY: crop counts positive** threshold only controls interactive preview filtering. It does not control saved YOLO/COCO annotation inclusion.
- Training selection: `{crop_cfg.get('train_deterministic_selection_mode', crop_cfg.get('deterministic_selection_mode', 'all'))}` with target positive fraction `{crop_cfg.get('train_deterministic_target_positive_ratio', crop_cfg.get('deterministic_target_positive_ratio', 'n/a'))}`. In `crop_label_ratio` mode every eligible positive crop is retained and empty crops are streamed only from breasts with no Mass in either view toward the target ratio. Source order uses a compact cadence computed as `ceil(1 / minority_fraction)`; for example, 50/50 becomes one positive then one negative and 80/20 becomes four positive then one negative. After the cadence pass, seeded reserve negative breasts are processed only while the saved negative crop count remains below target.
- Validation selection: `{crop_cfg.get('val_deterministic_selection_mode', 'all')}`; test selection: `{crop_cfg.get('test_deterministic_selection_mode', 'all')}`.
- Crop images: `square_crops/images/<split>/`.
- Float32 crop tensors: `square_crops/float32/images/<split>/<same stem>.pt` when selected.
- YOLO labels: `square_crops/labels/<split>/`; class `0` is Mass.
- COCO annotations: `square_crops/mmdetection/annotations/instances_<split>.json`; category id `1` is Mass.

## Whole-image companions

{chr(10).join(whole_lines)}

Whole images are padded before any optional resize, so the mammogram aspect ratio is preserved. The high-resolution output means no resize after fixed preprocessing and canvas padding; it is not the raw DICOM pixel array. DINOv3 feature extraction is not performed by this export.

## Non-quantized float32 image tensors

- Enabled: `{str(bool(float32_cfg.get('enabled', False))).lower()}`.
- Selected image types: `{', '.join(float32_selected_variants) or 'none'}`.
- Each tensor is saved with `torch.save` as contiguous `torch.float32` in CHW layout and normalized to the closed interval `[0, 1]`.
- Tensor paths mirror PNG paths under the nearest `float32/` directory and use the same stem, for example `{float_example}`.
- The float32 branch does not convert image pixels through uint8 or uint16. Preprocessing operations such as percentile clipping or CLAHE remain intentional image transforms, but they run directly on floating-point values; only the separate PNG branch is quantized to 0–255.
- The Feature Extraction window prefers these tensors automatically and warns before falling back to PNG.

Each source mammogram is written exactly once in each enabled whole-image resolution. All crops from that mammogram share the same `paired_whole_image` path and, when enabled, the same `paired_whole_high_resolution_image` path. No hard links or copied aliases are created.

Every enabled whole-image variant has matched YOLO, JSON, and COCO annotations. New grouped-layout exports keep all of them below `annotations/<variant>/`, while legacy exports retain the historical `whole_labels_*`, `whole_annotations_*`, and `mmdetection/` paths. `metadata/whole_image_manifest.csv` and `annotations/whole_image_annotations.csv` provide the grouped-layout cross-format audit indexes.

The naming contract is `<source key>__crop__<crop details>.png` for a crop and `<source key>.png` for both whole-image directories. Therefore a loader can derive the whole filename by removing the final `__crop__...` portion from the crop stem, but the authoritative and recommended mapping is the explicit path in `metadata/samples_metadata_flat.csv`, `metadata/crop_locations.csv`, or the COCO image record. The `paired_whole_key` column/field is the shared source key.

## Exact crop locations

Use `square_crops/metadata/crop_locations.csv` as the direct crop-to-source mapping. `crop_x0`, `crop_y0`, `crop_x1`, and `crop_y1` are the requested window in the fixed-preprocessed source coordinate space. `crop_pad_*` records pixels filled outside the source at image edges. The file also includes:

- the valid source intersection;
- the corresponding original-DICOM window after undoing fixed crop/mirror geometry;
- coordinates on the high-resolution padded whole canvas;
- floating-point coordinates on the resized whole canvas;
- padding offsets and resize scales.

The same `crop_window_xyxy` is also present in `stats/samples.csv`, `metadata/samples_metadata.jsonl`, `metadata/samples_metadata_flat.csv`, `debug_logs/crop_log.csv`, and each COCO image record.

For a point `(x, y)` in fixed-preprocessed source coordinates, its resized whole-image position is:

```text
whole_x = (x + whole_pad_left) * whole_resized_scale_x
whole_y = (y + whole_pad_top)  * whole_resized_scale_y
```

## Split counts

{chr(10).join(split_lines)}

## Annotation geometry and crop-fit report

- Enabled: `{str(bool(annotation_report_cfg.get('enabled', False))).lower()}`.
- Output: `visualizations/{str(annotation_report_cfg.get('output_subdir', 'annotation_geometry') or 'annotation_geometry')}/`.
- This report contains one row per fixed-preprocessed source Mass annotation, size histograms, width-versus-height plots, and counts of annotations that can or cannot fit fully inside the configured `{crop_size} x {crop_size}` crop by dimensions alone.
- Fit means `annotation_width <= crop_width` and `annotation_height <= crop_height`. Annotation location and actual crop-window locations are intentionally ignored.

## Debug and audit files

- Debug review bundle enabled: `{str(bool(review_cfg.get('enabled', False))).lower()}`.
- Debug source-asset limit per split: `{review_cfg.get('source_assets_per_split', review_cfg.get('samples_per_split', 100))}`. Assets are created only after a source contributes a successfully saved crop.
- `square_crops/debug_logs/` contains crop/source coverage logs.
- `square_crops/review/` contains sampled previews, masks, overlays, and GIFs when debug review is enabled.
- `split_assignments.csv` records source image membership.
- `export_summary.json` and `manifest.json` provide machine-readable run summaries.

## Exact reproducibility bundle

- Enabled: `{str(bool(reproducibility_cfg.get('enabled', False))).lower()}`.
- Output: `{str(reproducibility_cfg.get('output_subdir', 'reproducibility') or 'reproducibility')}/`.
- `source_images.csv` records exact source membership; `crops.csv` records saved crop order, coordinates, padding, transforms, paths, labels, and selection diagnostics; `crop_annotations.csv` records exported boxes in crop, fixed-preprocessed source, and original-DICOM coordinate spaces.
- `crop_records.jsonl`, `resolved_config.yaml`, the exporter source snapshot, software/source provenance, and metadata SHA-256 checksums provide the lossless replay and audit details. Reuse the recorded windows rather than resampling to reproduce this exact crop dataset.

## Training entry points

- Ultralytics/YOLO: `square_crops/vindr_mass.yaml`.
- MMDetection: use the paths in `square_crops/mmdetection/README_mmdetection_paths.txt`.
- Save square crops was `{str(bool(export_cfg.get('save_square_crops', True))).lower()}` for this run.
"""
    path = output_root / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_provenance(data_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ["metadata.csv", "breast-level_annotations.csv", "finding_annotations.csv"]:
        path = data_root / name
        out.append({
            "name": name,
            "path": _path_as_posix(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256_file(path) if path.is_file() else None,
        })
    return out


def _software_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip() or None

    status = git_output("status", "--porcelain")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "pillow": getattr(Image, "__version__", None),
    }


def _expected_completion_files(output_root: Path, config: dict[str, Any]) -> list[Path]:
    """Return files that should exist after the configured export finishes."""
    expected: list[Path] = [
        output_root / "README.md",
        output_root / "export_summary.json",
        output_root / "split_assignments.csv",
        output_root / "metadata" / "export_config_resolved.yaml",
        output_root / "metadata" / "source_csv" / "export_config_resolved.yaml",
    ]
    export_cfg = config.get("export", {})
    paired_cfg = dict(config.get("paired_whole_images", {}) or {})
    content_root = dataset_content_root(output_root, config)
    if bool(paired_cfg.get("enabled", False)) and any([
        _paired_original_enabled(paired_cfg),
        _paired_resized_enabled(paired_cfg),
        _paired_high_resolution_enabled(paired_cfg),
    ]):
        expected.extend(
            [
                content_root / "metadata" / "whole_image_manifest.csv",
                (
                    content_root / "annotations" / "whole_image_annotations.csv"
                    if uses_grouped_dataset_layout(config)
                    else content_root / "metadata" / "whole_image_annotations.csv"
                ),
                content_root / "metadata" / "samples_metadata_flat.csv",
            ]
        )
        if not bool(export_cfg.get("save_square_crops", True)):
            expected.extend(
                [
                    content_root / "metadata" / "whole_image_samples.jsonl",
                    content_root / "metadata" / "whole_image_validation.json",
                ]
            )
        variant_ids: list[str] = []
        if _paired_original_enabled(paired_cfg):
            variant_ids.append("original")
        if _paired_resized_enabled(paired_cfg):
            variants = resized_variant_configs(paired_cfg)
            variant_ids.extend(
                _whole_variant_id("resized", item["name"])
                if "resized_variants" in paired_cfg
                else "resized"
                for item in variants
            )
        if _paired_high_resolution_enabled(paired_cfg):
            variant_ids.append("high_resolution")
        for variant in variant_ids:
            expected.extend(
                content_root / _whole_coco_relative_path(config, variant, split)
                for split in ["train", "val", "test"]
            )
    if bool(export_cfg.get("save_square_crops", True)):
        expected.extend(
            [
                output_root / "square_crops" / "stats" / "summary.csv",
                output_root / "square_crops" / "stats" / "samples.csv",
                output_root / "square_crops" / "metadata" / "crop_locations.csv",
                output_root / "square_crops" / "vindr_mass.yaml",
                output_root / "square_crops" / "ultralytics" / "vindr_mass.yaml",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_train.json",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_val.json",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_test.json",
            ]
        )
        annotation_cfg = dict(config.get("annotation_geometry_report", {}) or {})
        if bool(annotation_cfg.get("enabled", False)):
            annotation_dir = output_root / "visualizations" / str(
                annotation_cfg.get("output_subdir", "annotation_geometry")
                or "annotation_geometry"
            )
            expected.extend(
                [
                    annotation_dir / "mass_box_geometry.csv",
                    annotation_dir / "mass_box_fit_summary.csv",
                    annotation_dir / "mass_box_fit_summary.json",
                    annotation_dir / "README.md",
                    annotation_dir / "index.html",
                ]
            )
    if bool(export_cfg.get("save_baseline_uncropped", True)):
        expected.extend(
            [
                output_root / "baseline_uncropped" / "stats" / "summary.csv",
                output_root / "baseline_uncropped" / "stats" / "samples.csv",
                output_root / "baseline_uncropped" / "vindr_mass.yaml",
                output_root / "baseline_uncropped" / "ultralytics" / "vindr_mass.yaml",
                output_root / "baseline_uncropped" / "mmdetection" / "annotations" / "instances_train.json",
                output_root / "baseline_uncropped" / "mmdetection" / "annotations" / "instances_val.json",
                output_root / "baseline_uncropped" / "mmdetection" / "annotations" / "instances_test.json",
            ]
        )
    reproducibility_cfg = dict(config.get("reproducibility_bundle", {}) or {})
    if bool(reproducibility_cfg.get("enabled", False)):
        reproducibility_dir = output_root / str(
            reproducibility_cfg.get("output_subdir", "reproducibility")
            or "reproducibility"
        )
        expected.extend(
            [
                reproducibility_dir / "README.md",
                reproducibility_dir / "source_images.csv",
                reproducibility_dir / "source_processing.csv",
                reproducibility_dir / "crops.csv",
                reproducibility_dir / "crop_records.jsonl",
                reproducibility_dir / "crop_annotations.csv",
                reproducibility_dir / "resolved_config.yaml",
                reproducibility_dir / "software_environment.json",
                reproducibility_dir / "source_metadata_provenance.json",
                reproducibility_dir / "bundle_manifest.json",
            ]
        )
        if bool(reproducibility_cfg.get("include_software_source_snapshot", True)):
            expected.extend(
                [
                    reproducibility_dir / "software_source_files.csv",
                    reproducibility_dir / "software_source_snapshot.zip",
                ]
            )
        if bool(reproducibility_cfg.get("write_metadata_sha256", True)):
            expected.append(reproducibility_dir / "checksums.sha256")
    return expected


def _collect_export_file_counts(
    output_root: Path, *, config: dict[str, Any] | None = None
) -> dict[str, int]:
    """Collect lightweight output counts for quick sanity checks."""
    counts: dict[str, int] = {}
    if uses_grouped_dataset_layout(config or {}):
        paired_cfg = dict((config or {}).get("paired_whole_images", {}) or {})
        for split_name in ["train", "val", "test"]:
            counts[f"images.{split_name}.original_png"] = _count_files(
                output_root / "images" / "original" / split_name, "*.png"
            )
            counts[f"annotations.{split_name}.original_yolo"] = _count_files(
                output_root / "annotations" / "original" / "yolo" / split_name,
                "*.txt",
            )
            for variant in resized_variant_configs(paired_cfg):
                resolution = str(variant["name"])
                counts[f"images.{split_name}.resized_{resolution}_png"] = _count_files(
                    output_root / "images" / "resized" / resolution / split_name,
                    "*.png",
                )
                counts[f"images.{split_name}.resized_{resolution}_float32_pt"] = _count_files(
                    output_root / "images" / "float32" / "resized" / resolution / split_name,
                    "*.pt",
                )
                counts[f"annotations.{split_name}.resized_{resolution}_yolo"] = _count_files(
                    output_root / "annotations" / "resized" / resolution / "yolo" / split_name,
                    "*.txt",
                )
        return counts
    for dataset_name in ["square_crops", "baseline_uncropped"]:
        for split_name in ["train", "val", "test"]:
            img_dir = output_root / dataset_name / "images" / split_name
            float32_root = output_root / dataset_name / "float32"
            float32_img_dir = float32_root / "images" / split_name
            label_dir = output_root / dataset_name / "labels" / split_name
            preserved_dir = output_root / dataset_name / "preserved_16bit" / split_name
            paired_whole_dir = output_root / dataset_name / "whole_images" / split_name
            paired_whole_original_dir = output_root / dataset_name / "whole_images_original" / split_name
            paired_whole_high_resolution_dir = (
                output_root / dataset_name / "whole_images_high_resolution" / split_name
            )
            whole_original_labels_dir = output_root / dataset_name / "whole_labels_original" / split_name
            whole_resized_labels_dir = output_root / dataset_name / "whole_labels_resized" / split_name
            whole_high_labels_dir = output_root / dataset_name / "whole_labels_high_resolution" / split_name
            paired_whole_native_dir = output_root / dataset_name / "whole_images_native" / split_name
            counts[f"{dataset_name}.{split_name}.images_png"] = _count_files(img_dir, "*.png")
            counts[f"{dataset_name}.{split_name}.images_float32_pt"] = _count_files(
                float32_img_dir, "*.pt"
            )
            counts[f"{dataset_name}.{split_name}.labels_txt"] = _count_files(label_dir, "*.txt")
            counts[f"{dataset_name}.{split_name}.preserved_16bit_png"] = _count_files(preserved_dir, "*.png")
            counts[f"{dataset_name}.{split_name}.paired_whole_png"] = _count_files(paired_whole_dir, "*.png")
            counts[f"{dataset_name}.{split_name}.paired_whole_float32_pt"] = _count_files(
                float32_root / "whole_images" / split_name, "*.pt"
            )
            counts[f"{dataset_name}.{split_name}.paired_whole_original_png"] = _count_files(paired_whole_original_dir, "*.png")
            counts[
                f"{dataset_name}.{split_name}.paired_whole_original_float32_pt"
            ] = _count_files(
                float32_root / "whole_images_original" / split_name, "*.pt"
            )
            counts[f"{dataset_name}.{split_name}.paired_whole_original_labels"] = _count_files(whole_original_labels_dir, "*.txt")
            counts[f"{dataset_name}.{split_name}.paired_whole_resized_labels"] = _count_files(whole_resized_labels_dir, "*.txt")
            counts[f"{dataset_name}.{split_name}.paired_whole_high_resolution_labels"] = _count_files(whole_high_labels_dir, "*.txt")
            counts[f"{dataset_name}.{split_name}.paired_whole_high_resolution_png"] = _count_files(
                paired_whole_high_resolution_dir, "*.png"
            )
            counts[
                f"{dataset_name}.{split_name}.paired_whole_high_resolution_float32_pt"
            ] = _count_files(
                float32_root / "whole_images_high_resolution" / split_name,
                "*.pt",
            )
            counts[f"{dataset_name}.{split_name}.paired_whole_native_png"] = _count_files(paired_whole_native_dir, "*.png")
    annotation_dir = output_root / "visualizations" / "annotation_geometry"
    counts["annotation_geometry.csv"] = _count_files(annotation_dir, "*.csv")
    counts["annotation_geometry.png"] = _count_files(annotation_dir, "*.png")
    return counts


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _summary_from_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"num_images": 0}
    out: dict[str, Any] = {"num_images": int(len(df)), "num_positive_images": int(df["has_mass"].sum())}
    out["splits"] = {}
    for split, group in df.groupby("split"):
        out["splits"][str(split)] = {
            "num_images": int(len(group)),
            "num_positive_images": int(group["has_mass"].sum()),
            "num_mass_boxes": int(group["num_mass_boxes"].sum()),
        }
    return out


def _summary_dataframe(rows: list[dict[str, Any]], dataset_kind: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary_rows = []
    if df.empty:
        return pd.DataFrame(columns=["dataset", "split", "num_images", "num_positive_images", "num_mass_boxes"])
    for split, group in df.groupby("split"):
        summary_rows.append(
            {
                "dataset": dataset_kind,
                "split": split,
                "num_images": int(len(group)),
                "num_positive_images": int(group["has_mass"].sum()),
                "positive_image_percent": 100.0 * float(group["has_mass"].mean()),
                "num_mass_boxes": int(group["num_mass_boxes"].sum()),
                "mean_boxes_per_image": float(group["num_mass_boxes"].mean()),
                "mean_mass_area_percentage": float(group["mean_mass_area_percentage"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def _make_baseline_filename(record: dict[str, Any]) -> str:
    return f"{_safe_name(record.get('study_id'))}_{_safe_name(record.get('image_id'))}.png"


def _make_crop_filename(record: dict[str, Any], split: str, crop_number: int, window: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = [int(v) for v in window]
    return (
        f"{_safe_name(record.get('study_id'))}__{_safe_name(record.get('image_id'))}"
        f"__crop__{split}_{crop_number:04d}_x{x0}_y{y0}_w{x1 - x0}_h{y1 - y0}.png"
    )


def _whole_image_filename_from_crop_filename(
    crop_filename: str,
    *,
    source_image_id: str = "",
) -> str:
    """Return the canonical one-per-source whole filename for a crop.

    New crop names contain a final ``__crop__`` separator.  The fallback keeps
    this helper usable with caller-supplied/legacy crop names in private APIs.
    """
    crop_stem = Path(str(crop_filename)).stem
    source_stem, separator, _crop_details = crop_stem.rpartition("__crop__")
    if separator and source_stem:
        return f"{source_stem}.png"
    fallback = _safe_name(source_image_id) or crop_stem or "source"
    return f"{fallback}.png"


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)


def _boxes_to_list(boxes: torch.Tensor | None) -> list[list[float]]:
    if boxes is None or boxes.numel() == 0:
        return []
    return boxes.detach().cpu().to(torch.float32).reshape(-1, 4).tolist()


def _path_as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _progress(
    iterable: Iterable[Any],
    enabled: bool,
    desc: str,
    unit: str = "it",
    total: int | None = None,
) -> Iterable[Any]:
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, unit=unit, total=total)
    return iterable


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert tensors, numpy values, paths, NaNs, and cycles to JSON-safe values."""
    if _seen is None:
        _seen = set()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return _path_as_posix(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_reference>"
        _seen.add(obj_id)
        try:
            return {str(k): _json_safe(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_reference>"
        _seen.add(obj_id)
        try:
            return [_json_safe(v, _seen) for v in value]
        finally:
            _seen.discard(obj_id)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)
