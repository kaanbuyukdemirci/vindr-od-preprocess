from __future__ import annotations

import json
import math
import shutil
from collections import OrderedDict
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .crops import (
    crop_image_and_boxes_to_window,
    sample_box_centered_square_window,
    sample_random_square_window,
    sliding_square_windows,
    window_has_positive_mass,
)
from .dataset import VindrMammoDataset

CLASS_NAMES = ["mass"]
COCO_CATEGORIES = [{"id": 1, "name": "mass", "supercategory": "lesion"}]


@dataclass
class ExportResult:
    output_root: Path
    created_files: list[Path]
    summary: dict[str, Any]


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
            return_dicom_meta=bool(config.get("metadata", {}).get("include_dicom_meta", True)),
            validate_paths=bool(config.get("dataset", {}).get("validate_paths", False)),
            preprocess_options=config.get("preprocess", {}),
            crop_options={"enabled": False},
            show_progress=bool(config.get("runtime", {}).get("show_progress", True)),
        )

    dataset = timed_stage("initialize_dataset", build_dataset)

    def make_splits():
        split_records_, split_table_ = make_train_val_test_split(
            dataset.image_records,
            val_fraction=float(config.get("splits", {}).get("val_fraction_from_training", 0.15)),
            seed=int(config.get("splits", {}).get("seed", 123)),
        )
        split_records_, split_table_, vendor_summary_ = _apply_vendor_filter_to_splits(dataset, split_records_, split_table_, config)
        split_path = output_root / "split_assignments.csv"
        split_table_.to_csv(split_path, index=False)
        return split_records_, split_table_, split_path, vendor_summary_

    split_records, split_table, split_path, vendor_filter_summary = timed_stage("make_train_val_test_split", make_splits)

    created_files: list[Path] = [split_path]
    created_files.extend(timed_stage("write_source_metadata_and_config", lambda: _write_global_metadata_files(output_root, dataset, config)))

    crop_cfg_for_summary = config.get("square_crops", {})
    summary: dict[str, Any] = {
        "num_source_images": len(dataset.image_records),
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
        "vendor_filter": vendor_filter_summary,
    }

    if bool(export_cfg.get("save_square_crops", True)):
        crop_summary, crop_files = timed_stage(
            "export_square_crops",
            lambda: export_square_crop_datasets(dataset, split_records, config, output_root, progress_callback=progress_callback),
        )
        summary["square_crops"] = crop_summary
        created_files.extend(crop_files)

    if bool(export_cfg.get("save_baseline_uncropped", True)):
        baseline_summary, baseline_files = timed_stage(
            "export_baseline_uncropped",
            lambda: export_baseline_dataset(dataset, split_records, config, output_root),
        )
        summary["baseline_uncropped"] = baseline_summary
        created_files.extend(baseline_files)

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
    summary["manifest"] = manifest
    return ExportResult(output_root=output_root, created_files=created_files, summary=summary)


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


def make_train_val_test_split(
    image_records: list[dict[str, Any]], *, val_fraction: float, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Use VinDr's official test split and make a study-level val split from training.

    VinDr-Mammo already contains official ``training`` and ``test`` values. This
    function keeps official ``test`` untouched, then randomly splits official
    ``training`` studies into train and val. Splitting by ``study_id`` prevents
    views from the same exam leaking across train and validation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    training_records = []
    test_records = []

    for record in image_records:
        split = str(record.get("split", "training")).casefold().strip()
        if split == "test":
            test_records.append(record)
        else:
            training_records.append(record)

    study_ids = sorted({str(r.get("study_id")) for r in training_records})
    shuffled = np.array(study_ids, dtype=object)
    rng.shuffle(shuffled)
    n_val = int(round(float(val_fraction) * len(shuffled)))
    n_val = max(1, n_val) if len(shuffled) > 1 and val_fraction > 0 else 0
    val_ids = set(str(x) for x in shuffled[:n_val])

    out = {"train": [], "val": [], "test": []}
    for record in training_records:
        export_split = "val" if str(record.get("study_id")) in val_ids else "train"
        out[export_split].append(record)
    out["test"].extend(test_records)

    for split_name, records in out.items():
        for r in records:
            rows.append(
                {
                    "export_split": split_name,
                    "official_split": r.get("split"),
                    "study_id": str(r.get("study_id")),
                    "image_id": str(r.get("image_id")),
                    "laterality": r.get("laterality"),
                    "view_position": r.get("view_position"),
                }
            )
    return out, pd.DataFrame(rows)


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

    * ``train_crop_mode``: ``"random"`` or ``"deterministic"``.
    * ``val_crop_mode``: usually ``"deterministic"``.
    * ``test_crop_mode``: usually ``"deterministic"``.

    Earlier versions always used random, mass-centered training crops and
    deterministic validation/test crops. That can create a strong distribution
    mismatch, so deterministic training crops are now supported directly.
    """
    crop_root = output_root / "square_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    crop_cfg = dict(config.get("square_crops", {}))
    crop_size = int(crop_cfg.get("crop_size", 1024))
    stride = int(crop_cfg.get("stride", 512))
    common_crop_options = dict(config.get("crop_annotation_policy", {}))
    common_crop_options.update(
        {
            "enabled": True,
            "crop_size": crop_size,
            "stride": stride,
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
    show_progress = bool(config.get("runtime", {}).get("show_progress", True))

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
        iterator = _progress(records, show_progress, f"Export square crops {split_name}", unit="img")
        for record in iterator:
            image, target = get_preprocessed(record)
            height, width = int(image.shape[-2]), int(image.shape[-1])
            windows = _windows_for_export_split(
                split_name=split_name,
                image_width=width,
                image_height=height,
                image_tensor=image,
                mass_boxes=target["mass"]["boxes"],
                crop_options=common_crop_options,
                crop_cfg=crop_cfg,
                rng=rng,
            )
            for crop_number, (window, extra_info) in enumerate(windows):
                crop_result = crop_image_and_boxes_to_window(
                    image,
                    boxes=target["boxes"],
                    mass_boxes=target["mass"]["boxes"],
                    window_xyxy=window,
                    options=common_crop_options,
                )
                boxes = crop_result.mass_boxes
                # Safety guard for split-specific positive-only deterministic exports.
                # Window selection should already have removed empty windows, but this
                # protects against any future annotation-policy change.
                if int(extra_info.get("deterministic_include_empty", 1)) == 0 and boxes.shape[0] == 0:
                    continue

                filename = _make_crop_filename(record, split_name, crop_number, window)
                rel_img_path = Path("images") / split_name / filename
                source_arrays = None
                if needs_contralateral:
                    source_arrays = _contralateral_source_arrays_for_window(
                        record=record,
                        window=window,
                        crop_options=common_crop_options,
                        contralateral_lookup=contralateral_lookup,
                        get_preprocessed=get_preprocessed,
                    )
                save_info = _save_export_images(crop_result.image, crop_root, rel_img_path, config, source_arrays=source_arrays)

                labels_path = crop_root / "labels" / split_name / f"{Path(filename).stem}.txt"
                _write_yolo_label_file(labels_path, boxes, width=crop_size, height=crop_size, save_empty=save_empty_labels)

                image_meta = _coco_image_record(
                    image_id_counter,
                    filename,
                    crop_size,
                    crop_size,
                    record,
                    save_info,
                    split_name,
                    dataset_name="square_crops",
                    crop_info={"window_xyxy": window, **extra_info, **crop_result.info},
                )
                coco = coco_by_split[split_name]
                coco["images"].append(image_meta)
                ann_rows = _append_coco_annotations(coco, image_id_counter, ann_id_counter, boxes)
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
                        crop_info={"window_xyxy": window, **extra_info, **crop_result.info},
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
                        crop_info={"window_xyxy": window, **extra_info, **crop_result.info},
                    )
                )
                image_id_counter += 1
            processed_records_for_progress += 1
            if progress_callback is not None:
                progress_callback({
                    "event": "image_progress",
                    "stage": "export_square_crops",
                    "split": split_name,
                    "processed": int(processed_records_for_progress),
                    "total": int(total_records_for_progress),
                })

    created = _write_shared_export_files(crop_root, coco_by_split, stats_rows, metadata_rows, dataset_kind="square_crops")
    return _summary_from_stats(stats_rows), created


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
) -> list[tuple[tuple[int, int, int, int], dict[str, Any]]]:
    """Return crop windows for one image according to train/val/test policy."""
    split_mode = str(crop_cfg.get(f"{split_name}_crop_mode", "random" if split_name == "train" else "deterministic")).casefold().strip()
    if split_mode not in {"random", "deterministic"}:
        raise ValueError(
            f"square_crops.{split_name}_crop_mode must be 'random' or 'deterministic', got {split_mode!r}."
        )

    if split_mode == "deterministic":
        windows = sliding_square_windows(
            width=image_width,
            height=image_height,
            crop_size=int(crop_cfg.get("crop_size", 1024)),
            stride=int(crop_cfg.get("stride", 512)),
        )

        # Split-specific empty-window control. This is useful for experiments such as
        # v3, where training should be deterministic but positive-only, while
        # validation/test should remain full sliding-window evaluations.
        include_empty = bool(
            crop_cfg.get(
                f"{split_name}_deterministic_include_empty",
                crop_cfg.get("deterministic_include_empty", True),
            )
        )
        if not include_empty:
            windows = [w for w in windows if window_has_positive_mass(w, mass_boxes, crop_options)]

        foreground_filter_enabled = bool(_split_crop_cfg(
            crop_cfg,
            split_name,
            "deterministic_require_foreground",
            False,
        ))
        min_foreground_fraction = float(_split_crop_cfg(
            crop_cfg,
            split_name,
            "deterministic_min_foreground_fraction",
            0.05,
        ))
        foreground_threshold = crop_cfg.get("deterministic_foreground_threshold", None)
        foreground_fractions: dict[tuple[int, int, int, int], float] = {}
        if foreground_filter_enabled:
            image_np = image_tensor.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
            kept_windows = []
            for w in windows:
                frac = _foreground_fraction_in_window(
                    image_np,
                    w,
                    crop_size=int(crop_cfg.get("crop_size", 1024)),
                    threshold=foreground_threshold,
                    pad_value=float(crop_cfg.get("pad_value", 0.0)),
                )
                foreground_fractions[w] = frac
                if frac >= min_foreground_fraction:
                    kept_windows.append(w)
            windows = kept_windows

        max_windows = crop_cfg.get(f"{split_name}_deterministic_max_windows_per_image", crop_cfg.get("deterministic_max_windows_per_image"))
        if max_windows is not None:
            windows = windows[: int(max_windows)]
        return [
            (
                w,
                {
                    "crop_mode": "deterministic",
                    "split_crop_mode": split_mode,
                    "deterministic_include_empty": int(include_empty),
                    "foreground_filter_enabled": int(foreground_filter_enabled),
                    "min_foreground_fraction": float(min_foreground_fraction),
                    "foreground_fraction": foreground_fractions.get(w, None),
                },
            )
            for w in windows
        ]

    # Random mode. Positive crops are centered around annotations.
    boxes = mass_boxes.detach().cpu().to(torch.float32).reshape(-1, 4)
    windows: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    crops_per_ann = int(crop_cfg.get("random_crops_per_annotation", 5))
    positive_fraction = float(crop_cfg.get("positive_fraction", 0.80))

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
            windows.append((window, {"crop_mode": "random", "annotation_index": int(ann_index), **info}))

    num_positive = len(windows)
    if num_positive > 0 and positive_fraction > 0:
        num_negative = int(round(num_positive * max(0.0, 1.0 - positive_fraction) / positive_fraction))
    else:
        num_negative = int(crop_cfg.get("random_crops_per_negative_image", 1))

    if boxes.shape[0] == 0:
        # Important: if the user requests an overall train positive fraction such
        # as 0.80, do not automatically add one negative crop from every normal
        # image. Otherwise the global positive percentage can become much lower
        # than requested because VinDr-Mammo has many images without mass boxes.
        # Set balance_train_positive_fraction_globally=False to restore the old
        # behavior and sample negatives from every no-mass image.
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
        windows.append((window, {"crop_mode": "random_clean", **info}))
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
    window: tuple[int, int, int, int],
    crop_options: dict[str, Any],
    contralateral_lookup: dict[str, dict[str, Any]],
    get_preprocessed,
) -> dict[str, np.ndarray]:
    paired_record = contralateral_lookup.get(str(record.get("image_id", "")))
    if paired_record is None:
        return {}
    paired_image, _paired_target = get_preprocessed(paired_record)
    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    paired_crop = crop_image_and_boxes_to_window(
        paired_image,
        boxes=empty_boxes,
        mass_boxes=empty_boxes,
        window_xyxy=window,
        options=crop_options,
    )
    return {"contralateral_same_view_crop": _tensor_to_float2d(paired_crop.image)}


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
            height, width = int(image.shape[-2]), int(image.shape[-1])
            filename = _make_baseline_filename(record)
            rel_img_path = Path("images") / split_name / filename
            save_info = _save_export_images(image, base_root, rel_img_path, config)

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


# -----------------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------------


def _save_export_images(
    image: torch.Tensor,
    root: Path,
    rel_img_path: Path,
    config: dict[str, Any],
    *,
    source_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Save the model-training RGB image and, optionally, a preserved 16-bit PNG."""
    img_cfg = config.get("image_export", {})
    preserved_cfg = config.get("preserved_16bit", {})

    train_path = root / rel_img_path
    train_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _tensor_to_float2d(image)
    rgb, rgb_meta = _make_rgb_image(arr, config, source_arrays=source_arrays)
    Image.fromarray(rgb, mode="RGB").save(train_path)

    out: dict[str, Any] = {
        "image_path": _path_as_posix(train_path.relative_to(root)),
        "rgb_scheme": str(img_cfg.get("rgb_scheme", "multi_window")),
        "histogram_equalization_enabled": bool(config.get("histogram_equalization", {}).get("enabled", True)),
        **rgb_meta,
    }

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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an 8-bit RGB image according to ``image_export.rgb_scheme``.

    Recommended default: ``multi_window``. It creates three visually meaningful
    mammography contrast windows instead of duplicating one grayscale window.
    """
    img_cfg = config.get("image_export", {})
    eq_cfg = config.get("histogram_equalization", {})
    scheme = str(img_cfg.get("rgb_scheme", "multi_window")).casefold().strip()
    mask = _foreground_mask(arr)
    meta: dict[str, Any] = {"window_mask_pixels": int(mask.sum())}

    if scheme == "grayscale_rgb":
        lo, hi = _safe_percentile(arr, img_cfg.get("single_window", [0.5, 99.5]), mask)
        ch = _to_uint8_window(arr, lo, hi)
        channels = [ch, ch.copy(), ch.copy()]
        meta["rgb_windows"] = [[lo, hi]] * 3

    elif scheme == "equalized_rgb":
        lo, hi = _safe_percentile(arr, img_cfg.get("single_window", [0.5, 99.5]), mask)
        ch = _equalize_uint8(_to_uint8_window(arr, lo, hi), mask=mask)
        channels = [ch, ch.copy(), ch.copy()]
        meta["rgb_windows"] = [[lo, hi]] * 3
        meta["forced_equalized_rgb"] = True

    elif scheme in {"intensity_equalized_gradient", "ieg", "normal_equalized_gradient"}:
        channels, ieg_meta = _make_intensity_equalized_gradient_rgb(arr, img_cfg, mask)
        meta.update(ieg_meta)

    elif scheme in {"custom_channel_pipeline", "gui_channel_pipeline"}:
        channels, custom_meta = _make_custom_channel_pipeline_rgb(arr, img_cfg, mask, source_arrays=source_arrays)
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
            channels.append(_to_uint8_window(arr, lo, hi))
            resolved.append([lo, hi])
        meta["rgb_windows"] = resolved

    else:
        raise ValueError(
            "Unknown image_export.rgb_scheme. Expected one of: "
            "multi_window, grayscale_rgb, equalized_rgb, "
            "intensity_equalized_gradient, custom_channel_pipeline, bitpack16."
        )

    if bool(eq_cfg.get("enabled", True)) and scheme not in {"equalized_rgb", "bitpack16", "intensity_equalized_gradient", "ieg", "normal_equalized_gradient", "custom_channel_pipeline", "gui_channel_pipeline"}:
        apply_to = str(eq_cfg.get("apply_to", "all_channels")).casefold().strip()
        if apply_to == "all_channels":
            channels = [_equalize_uint8(ch, mask=mask) for ch in channels]
        elif apply_to == "third_channel":
            channels[2] = _equalize_uint8(channels[2], mask=mask)
        elif apply_to in {"none", "false", "off"}:
            pass
        else:
            raise ValueError("histogram_equalization.apply_to must be all_channels, third_channel, or none.")
        meta["histogram_equalization_apply_to"] = apply_to

    rgb = np.stack(channels, axis=-1).astype(np.uint8, copy=False)
    return rgb, meta



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
        applied: list[dict[str, Any]] = []
        for step in _custom_channel_steps(pipeline, channel):
            if not isinstance(step, dict):
                continue
            op = str(step.get("op", "none")).casefold().strip()
            params = step.get("params", {}) or {}
            if op in {"none", "", "null"}:
                continue
            work = _apply_custom_channel_operation(work, op, params, mask)
            applied.append({"op": op, "params": params})
        channels.append(_float_to_uint8_custom(work))
        meta[f"custom_{channel}_source_requested"] = source_name
        meta[f"custom_{channel}_source_used"] = source_used
        meta[f"custom_{channel}_source_fallback"] = int(source_fallback)
        meta[f"custom_{channel}_steps"] = applied
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
        return _equalize_uint8(_float_to_uint8_custom(arr), mask=mask).astype(np.float32) / 255.0
    if op == "clahe":
        img = _float_to_uint8_custom(arr)
        if cv2 is None:
            return _equalize_uint8(img, mask=mask).astype(np.float32) / 255.0
        clip_limit = float(params.get("clip_limit", 2.0))
        tile = int(params.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        return clahe.apply(img).astype(np.float32) / 255.0
    if op == "gaussian_blur":
        if cv2 is None:
            return arr
        k = _odd_int_custom(params.get("ksize", 5))
        sigma = float(params.get("sigma", 1.0))
        return cv2.GaussianBlur(arr.astype(np.float32), (k, k), sigmaX=sigma).astype(np.float32)
    if op == "median_blur":
        if cv2 is None:
            return arr
        k = _odd_int_custom(params.get("ksize", 3))
        return cv2.medianBlur(_float_to_uint8_custom(arr), k).astype(np.float32) / 255.0
    if op == "sharpen":
        if cv2 is None:
            return arr
        amount = float(params.get("amount", 1.0))
        kernel = np.array([[0, -1, 0], [-1, 4 + amount, -1], [0, -1, 0]], dtype=np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(arr.astype(np.float32), -1, kernel).astype(np.float32)
    if op == "unsharp_mask":
        if cv2 is None:
            return arr
        amount = float(params.get("amount", 1.5))
        sigma = float(params.get("sigma", 2.0))
        blurred = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma)
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
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    lo, hi = _safe_percentile(arr, [1.0, 99.0], None)
    return _to_uint8_window(arr, lo, hi)


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
    normal = _to_uint8_window(arr, lo, hi)
    equalized = _equalize_uint8(normal, mask=mask)

    if gradient_source in {"equalized", "histogram_equalized", "eq"}:
        grad_input = equalized
    else:
        grad_input = normal
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
    """Return fraction of an n x n crop that appears to be breast foreground.

    This is intentionally simple and fast: it builds the same padded square crop
    that the exporter will save, makes a foreground mask, and returns mask.mean().
    Use it to reject pure-background deterministic windows when breast cropping is
    disabled.
    """
    x0, y0, x1, y1 = [int(v) for v in window_xyxy]
    h, w = image.shape
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    crop = np.full((int(crop_size), int(crop_size)), float(pad_value), dtype=np.float32)
    patch = image[src_y0:src_y1, src_x0:src_x1]
    if patch.size:
        dst_x0 = max(0, -x0)
        dst_y0 = max(0, -y0)
        crop[dst_y0:dst_y0 + patch.shape[0], dst_x0:dst_x0 + patch.shape[1]] = patch
    mask = _foreground_mask(crop, threshold=threshold)
    return float(mask.mean()) if mask.size else 0.0


def _foreground_mask(arr: np.ndarray, threshold: float | None = None) -> np.ndarray:
    finite = np.isfinite(arr)
    if not finite.any():
        return np.ones_like(arr, dtype=bool)
    vals = arr[finite]
    if threshold is None:
        lo, hi = np.percentile(vals, [1.0, 99.0])
        threshold = max(float(lo + 0.03 * (hi - lo)), float(lo) + 1e-6)
    mask = finite & (arr > float(threshold))
    if mask.sum() < max(10, int(0.001 * arr.size)):
        mask = finite
    return mask


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


def _equalize_uint8(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Simple histogram equalization for an 8-bit single-channel image."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if mask is not None and mask.any():
        values = img[mask]
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
    return lut[img]


# -----------------------------------------------------------------------------
# Annotation and metadata writers
# -----------------------------------------------------------------------------


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
) -> dict[str, Any]:
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
        "rgb_scheme": save_info.get("rgb_scheme"),
    }
    if crop_info:
        out["crop_window_xyxy"] = crop_info.get("window_xyxy")
    return out


def _append_coco_annotations(coco: dict[str, Any], image_id: int, start_ann_id: int, boxes: torch.Tensor) -> int:
    count = 0
    for box in _boxes_to_list(boxes):
        x0, y0, x1, y1 = box
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        if w <= 0 or h <= 0:
            continue
        coco["annotations"].append(
            {
                "id": int(start_ann_id + count),
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        count += 1
    return count


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
        "preserved_16bit_path": save_info.get("preserved_16bit_path", ""),
        "preserved_16bit_lo": save_info.get("preserved_16bit_lo", ""),
        "preserved_16bit_hi": save_info.get("preserved_16bit_hi", ""),
    }
    if crop_info:
        window = crop_info.get("window_xyxy")
        row.update(
            {
                "crop_window_xyxy": "" if window is None else str(tuple(int(v) for v in window)),
                "crop_mode": crop_info.get("crop_mode"),
                "requested_positive": crop_info.get("requested_positive"),
                "accepted": crop_info.get("accepted"),
                "foreground_filter_enabled": crop_info.get("foreground_filter_enabled", ""),
                "foreground_fraction": crop_info.get("foreground_fraction", ""),
                "min_foreground_fraction": crop_info.get("min_foreground_fraction", ""),
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
        "preserved_16bit_image": save_info.get("preserved_16bit_path"),
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
    first_meta = meta_rows[0] if meta_rows else {}
    return {
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "file_name": row.get("file_name"),
        "source_image_id": row.get("source_image_id"),
        "source_study_id": row.get("source_study_id"),
        "source_dicom_path": row.get("source_dicom_path"),
        "training_image": row.get("training_image"),
        "preserved_16bit_image": row.get("preserved_16bit_image"),
        "breast_birads": row.get("breast_birads"),
        "breast_density": row.get("breast_density"),
        "num_export_boxes": len(row.get("export_boxes_xyxy", []) or []),
        "crop_window_xyxy": crop.get("window_xyxy", ""),
        "crop_mode": crop.get("crop_mode", ""),
        "foreground_filter_enabled": crop.get("foreground_filter_enabled", ""),
        "foreground_fraction": crop.get("foreground_fraction", ""),
        "min_foreground_fraction": crop.get("min_foreground_fraction", ""),
        "manufacturer": first_meta.get("Manufacturer", first_meta.get("manufacturer", dicom_meta.get("Manufacturer", ""))),
        "manufacturer_model_name": first_meta.get("ManufacturerModelName", first_meta.get("manufacturer_model_name", dicom_meta.get("ManufacturerModelName", ""))),
        "photometric_interpretation": dicom_meta.get("PhotometricInterpretation", ""),
        "rgb_scheme": enc.get("rgb_scheme", ""),
        "histogram_equalization_enabled": enc.get("histogram_equalization_enabled", ""),
        "preserved_16bit_lo": enc.get("preserved_16bit_lo", ""),
        "preserved_16bit_hi": enc.get("preserved_16bit_hi", ""),
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
    file_counts = _collect_export_file_counts(output_root)
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
        "created_files_count": len(created_files),
        "created_files_tail": [_path_as_posix(p) for p in created_files[-50:]],
        "config_snapshot": config,
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

    return manifest, [manifest_path, done_path]


def _expected_completion_files(output_root: Path, config: dict[str, Any]) -> list[Path]:
    """Return files that should exist after the configured export finishes."""
    expected: list[Path] = [
        output_root / "export_summary.json",
        output_root / "split_assignments.csv",
        output_root / "metadata" / "export_config_resolved.yaml",
        output_root / "metadata" / "source_csv" / "export_config_resolved.yaml",
    ]
    export_cfg = config.get("export", {})
    if bool(export_cfg.get("save_square_crops", True)):
        expected.extend(
            [
                output_root / "square_crops" / "stats" / "summary.csv",
                output_root / "square_crops" / "stats" / "samples.csv",
                output_root / "square_crops" / "vindr_mass.yaml",
                output_root / "square_crops" / "ultralytics" / "vindr_mass.yaml",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_train.json",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_val.json",
                output_root / "square_crops" / "mmdetection" / "annotations" / "instances_test.json",
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
    return expected


def _collect_export_file_counts(output_root: Path) -> dict[str, int]:
    """Collect lightweight output counts for quick sanity checks."""
    counts: dict[str, int] = {}
    for dataset_name in ["square_crops", "baseline_uncropped"]:
        for split_name in ["train", "val", "test"]:
            img_dir = output_root / dataset_name / "images" / split_name
            label_dir = output_root / dataset_name / "labels" / split_name
            preserved_dir = output_root / dataset_name / "preserved_16bit" / split_name
            counts[f"{dataset_name}.{split_name}.images_png"] = _count_files(img_dir, "*.png")
            counts[f"{dataset_name}.{split_name}.labels_txt"] = _count_files(label_dir, "*.txt")
            counts[f"{dataset_name}.{split_name}.preserved_16bit_png"] = _count_files(preserved_dir, "*.png")
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
        f"{_safe_name(record.get('study_id'))}_{_safe_name(record.get('image_id'))}"
        f"_{split}_crop{crop_number:04d}_x{x0}_y{y0}_w{x1 - x0}_h{y1 - y0}.png"
    )


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)


def _boxes_to_list(boxes: torch.Tensor | None) -> list[list[float]]:
    if boxes is None or boxes.numel() == 0:
        return []
    return boxes.detach().cpu().to(torch.float32).reshape(-1, 4).tolist()


def _path_as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _progress(iterable: Iterable[Any], enabled: bool, desc: str, unit: str = "it") -> Iterable[Any]:
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, unit=unit)
    return iterable


def _json_safe(value: Any) -> Any:
    """Convert tensors, numpy values, paths, and NaNs to JSON-safe values."""
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
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray, torch.Tensor)) else False:
        return None
    return value
