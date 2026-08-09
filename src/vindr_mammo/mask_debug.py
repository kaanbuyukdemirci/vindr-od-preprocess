from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .crops import sliding_square_windows
from .dataset import VindrMammoDataset
from .export import load_export_config
from .pipeline_scope import crop_array_to_window
from .preprocessing import breast_mask
from .presets import (
    PAPER_22_IMPROVED_PRESET_KEY,
    PAPER_22_PRESET_KEY,
    PAPER_69_PRESET_KEY,
    SIMPLE_PRESET_KEY,
    STUDY_PRESETS,
    apply_study_preset,
)


MASK_COMPARISON_METHODS = (
    "largest_connected_tissue",
    "otsu_largest_connected_component",
    "percentile_threshold_largest_component",
)

PRESET_ALIASES = {
    "paper22": PAPER_22_PRESET_KEY,
    "custom-paper22": PAPER_22_IMPROVED_PRESET_KEY,
    "paper22-improved": PAPER_22_IMPROVED_PRESET_KEY,
    "paper69": PAPER_69_PRESET_KEY,
    "custom": SIMPLE_PRESET_KEY,
    "simple": SIMPLE_PRESET_KEY,
}


@dataclass(frozen=True)
class MaskDebugResult:
    output_dir: Path
    created_files: tuple[Path, ...]
    summary: dict[str, Any]


def mask_quality_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Return compact quality-control measurements for a binary breast mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {binary.shape}.")
    height, width = binary.shape
    pixels = int(binary.sum())
    if pixels:
        ys, xs = np.where(binary)
        bbox: list[int] | None = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    else:
        bbox = None

    component_count = 0
    largest_component_pixels = 0
    if pixels and cv2 is not None:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        component_count = max(0, int(count) - 1)
        if component_count:
            largest_component_pixels = int(stats[1:, cv2.CC_STAT_AREA].max())
    elif pixels:
        component_count = 1
        largest_component_pixels = pixels

    mask_fraction = float(binary.mean()) if binary.size else 0.0
    quality_flags: list[str] = []
    if pixels == 0:
        quality_flags.append("empty_mask")
    elif mask_fraction < 0.05:
        quality_flags.append("mask_covers_less_than_5_percent")
    elif mask_fraction > 0.95:
        quality_flags.append("mask_covers_more_than_95_percent")
    if component_count > 1:
        quality_flags.append("mask_has_multiple_components")

    return {
        "shape": [height, width],
        "mask_pixels": pixels,
        "mask_fraction": mask_fraction,
        "bbox_xyxy": bbox,
        "component_count": component_count,
        "largest_component_pixels": largest_component_pixels,
        "largest_component_fraction_of_mask": (
            float(largest_component_pixels / pixels) if pixels else 0.0
        ),
        "touches_top": bool(binary[0, :].any()) if height else False,
        "touches_bottom": bool(binary[-1, :].any()) if height else False,
        "touches_left": bool(binary[:, 0].any()) if width else False,
        "touches_right": bool(binary[:, -1].any()) if width else False,
        "quality_flags": quality_flags,
    }


def compare_mask_methods(
    image: np.ndarray,
    base_options: dict[str, Any],
    *,
    methods: Iterable[str] = MASK_COMPARISON_METHODS,
) -> dict[str, np.ndarray]:
    """Compute alternative masks without changing the active export settings."""
    image_t = torch.as_tensor(
        np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    ).unsqueeze(0)
    masks: dict[str, np.ndarray] = {}
    for method in methods:
        options = {**dict(base_options or {}), "breast_mask_method": str(method)}
        masks[str(method)] = np.asarray(
            breast_mask(image_t, options=options), dtype=bool
        )
    return masks


def create_mask_padding_debug_bundle(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    windows: Iterable[tuple[int, int, int, int]],
    output_dir: str | Path,
    crop_size: int,
    min_breast_fraction: float,
    comparison: str = "strictly_greater_than",
    pad_value: float = 0.0,
    unmasked_image: np.ndarray | None = None,
    comparison_masks: dict[str, np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None,
    max_crop_previews: int = 12,
) -> MaskDebugResult:
    """Write an auditable mask, window, and padding diagnostic bundle.

    Breast fractions use the full ``crop_size x crop_size`` denominator, so
    out-of-image padding always counts as non-breast exactly as it does during
    export. The supplied ``mask`` should be the retained preprocessing mask.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Expected 2-D image and mask; got image={image.shape}, mask={mask.shape}."
        )
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image={image.shape}, mask={mask.shape}."
        )
    n = int(crop_size)
    if n <= 0:
        raise ValueError("crop_size must be positive.")
    source_for_overlay = (
        image
        if unmasked_image is None
        else np.asarray(unmasked_image, dtype=np.float32)
    )
    if source_for_overlay.shape != image.shape:
        raise ValueError(
            "unmasked_image must use the same coordinate system as image and mask."
        )

    created: list[Path] = []

    def _write_png(name: str, value: np.ndarray) -> Path:
        path = output_dir / name
        arr = np.asarray(value)
        if arr.ndim == 2:
            if arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255
            elif arr.dtype != np.uint8:
                arr = _display_uint8(arr)
        if arr.ndim == 3 and arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(arr)).save(path)
        created.append(path)
        return path

    _write_png("01_fixed_preprocessed_image.png", image)
    if unmasked_image is not None:
        _write_png("00_before_breast_mask.png", source_for_overlay)
    _write_png("02_retained_breast_mask.png", mask)
    _write_png("03_mask_overlay.png", _mask_overlay(source_for_overlay, mask))
    _write_png("04_masked_image.png", np.where(mask, source_for_overlay, 0.0))
    mask_npy = output_dir / "retained_breast_mask.npy"
    np.save(mask_npy, mask.astype(bool, copy=False), allow_pickle=False)
    created.append(mask_npy)

    valid_source = np.ones(mask.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    normalized_windows = [tuple(int(v) for v in window) for window in windows]
    for window_index, window in enumerate(normalized_windows):
        mask_crop = crop_array_to_window(
            mask,
            window,
            output_height=n,
            output_width=n,
            pad_value=False,
        ).astype(bool, copy=False)
        valid_crop = crop_array_to_window(
            valid_source,
            window,
            output_height=n,
            output_width=n,
            pad_value=False,
        ).astype(bool, copy=False)
        pad_map = ~valid_crop
        breast_fraction = float(mask_crop.mean()) if mask_crop.size else 0.0
        keep = _fraction_passes(
            breast_fraction, float(min_breast_fraction), comparison
        )
        rows.append({
            "window_index": int(window_index),
            "x0": window[0],
            "y0": window[1],
            "x1": window[2],
            "y1": window[3],
            "breast_pixels": int(mask_crop.sum()),
            "breast_fraction": breast_fraction,
            "valid_source_pixels": int(valid_crop.sum()),
            "padding_pixels": int(pad_map.sum()),
            "padding_fraction": float(pad_map.mean()) if pad_map.size else 0.0,
            "minimum_breast_fraction": float(min_breast_fraction),
            "comparison": str(comparison),
            "kept": bool(keep),
        })

    windows_csv = output_dir / "windows.csv"
    with windows_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "window_index", "x0", "y0", "x1", "y1", "breast_pixels",
            "breast_fraction", "valid_source_pixels", "padding_pixels",
            "padding_fraction", "minimum_breast_fraction", "comparison", "kept",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    created.append(windows_csv)

    _write_png(
        "05_crop_grid_overlay.png",
        _grid_overlay(source_for_overlay, normalized_windows, rows),
    )

    crop_dir = output_dir / "crop_previews"
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview_indices = _representative_window_indices(
        rows,
        threshold=float(min_breast_fraction),
        limit=max(0, int(max_crop_previews)),
    )
    for preview_number, row_index in enumerate(preview_indices):
        row = rows[row_index]
        window = normalized_windows[row_index]
        image_crop = crop_array_to_window(
            source_for_overlay,
            window,
            output_height=n,
            output_width=n,
            pad_value=float(pad_value),
        ).astype(np.float32, copy=False)
        mask_crop = crop_array_to_window(
            mask,
            window,
            output_height=n,
            output_width=n,
            pad_value=False,
        ).astype(bool, copy=False)
        valid_crop = crop_array_to_window(
            valid_source,
            window,
            output_height=n,
            output_width=n,
            pad_value=False,
        ).astype(bool, copy=False)
        padding_map = ~valid_crop
        stem = f"{preview_number:02d}_window_{int(row['window_index']):04d}"
        for suffix, value in {
            "image": image_crop,
            "mask": mask_crop,
            "padding": padding_map,
            "mask_plus_padding": _mask_plus_padding(mask_crop, padding_map),
            "overlay": _crop_overlay(image_crop, mask_crop, padding_map),
        }.items():
            path = crop_dir / f"{stem}_{suffix}.png"
            arr = np.asarray(value)
            if arr.ndim == 2:
                arr = arr.astype(np.uint8) * 255 if arr.dtype == bool else _display_uint8(arr)
            Image.fromarray(np.ascontiguousarray(arr)).save(path)
            created.append(path)

    active_metrics = mask_quality_metrics(mask)
    method_rows: list[dict[str, Any]] = []
    method_dir = output_dir / "mask_method_comparison"
    for method, candidate in dict(comparison_masks or {}).items():
        candidate = np.asarray(candidate, dtype=bool)
        if candidate.shape != mask.shape:
            raise ValueError(
                f"Comparison mask {method!r} has shape {candidate.shape}, expected {mask.shape}."
            )
        method_dir.mkdir(parents=True, exist_ok=True)
        safe_method = _safe_name(method)
        for suffix, value in {
            "mask": candidate,
            "overlay": _mask_overlay(source_for_overlay, candidate),
        }.items():
            path = method_dir / f"{safe_method}_{suffix}.png"
            arr = np.asarray(value)
            if arr.ndim == 2:
                arr = arr.astype(np.uint8) * 255
            Image.fromarray(np.ascontiguousarray(arr)).save(path)
            created.append(path)
        intersection = int(np.logical_and(mask, candidate).sum())
        union = int(np.logical_or(mask, candidate).sum())
        method_rows.append({
            "method": str(method),
            **mask_quality_metrics(candidate),
            "iou_with_retained_mask": float(intersection / union) if union else 1.0,
        })

    if method_rows:
        methods_json = method_dir / "metrics.json"
        methods_json.write_text(
            json.dumps(method_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        created.append(methods_json)

    summary = {
        "mask": active_metrics,
        "crop_size": n,
        "window_count": len(rows),
        "kept_window_count": sum(bool(row["kept"]) for row in rows),
        "rejected_window_count": sum(not bool(row["kept"]) for row in rows),
        "padded_window_count": sum(int(row["padding_pixels"]) > 0 for row in rows),
        "minimum_breast_fraction": float(min_breast_fraction),
        "comparison": str(comparison),
        "representative_window_indices": [
            int(rows[index]["window_index"]) for index in preview_indices
        ],
        "mask_method_comparison": method_rows,
        "metadata": metadata or {},
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    created.append(summary_json)
    return MaskDebugResult(output_dir, tuple(created), summary)


def debug_masks_from_config(
    config_path: str | Path,
    *,
    preset_key: str = "custom-paper22",
    output_dir: str | Path = "mask_debug",
    image_indices: Iterable[int] | None = None,
    image_ids: Iterable[str] | None = None,
    max_images: int = 8,
    max_crop_previews: int = 12,
    compare_methods: bool = True,
    split: str = "train",
) -> MaskDebugResult:
    """Run mask/padding diagnostics on real DICOMs selected from a preset."""
    config = load_export_config(Path(config_path))
    resolved_preset = PRESET_ALIASES.get(str(preset_key), str(preset_key))
    config = apply_study_preset(config, resolved_preset)
    dataset = _dataset_from_config(config)
    unmasked_config = {
        **config,
        "preprocess": {
            **dict(config.get("preprocess", {}) or {}),
            "mask_outside_breast": False,
            "retain_breast_mask_for_export": False,
        },
    }
    unmasked_dataset = _dataset_from_config(unmasked_config)
    selected_indices = _select_record_indices(
        dataset,
        image_indices=image_indices,
        image_ids=image_ids,
        max_images=max_images,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_created: list[Path] = []
    image_summaries: list[dict[str, Any]] = []
    crop_cfg = dict(config.get("square_crops", {}) or {})
    crop_size = int(crop_cfg.get("crop_size", 640))
    stride = int(crop_cfg.get("stride", 512))
    edge_policy = str(crop_cfg.get("edge_policy", "edge_align"))
    split = str(split).casefold().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    threshold = float(
        crop_cfg.get(
            f"{split}_min_breast_fraction_for_all_crops",
            crop_cfg.get("deterministic_min_foreground_fraction", 0.05),
        )
    )
    comparison = str(
        crop_cfg.get(
            f"{split}_breast_fraction_comparison_for_all_crops",
            "greater_than_or_equal",
        )
    )
    pad_value = float(crop_cfg.get("pad_value", 0.0))

    for record_index in selected_indices:
        record = dataset.image_records[record_index]
        image_t, target = dataset._read_preprocessed_record_no_square(record)
        unmasked_t, unmasked_target = unmasked_dataset._read_preprocessed_record_no_square(
            unmasked_dataset.image_records[record_index]
        )
        image = image_t.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
        unmasked = unmasked_t.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
        actual_mirrored = bool((target.get("preprocessing", {}) or {}).get("mirrored", False))
        unmasked_mirrored = bool(
            (unmasked_target.get("preprocessing", {}) or {}).get("mirrored", False)
        )
        if actual_mirrored != unmasked_mirrored:
            unmasked = np.ascontiguousarray(np.fliplr(unmasked))
        retained = target.get("_foreground_mask")
        if retained is None:
            raise RuntimeError(
                f"Image {record.get('image_id')} has no retained preprocessing mask. "
                "Set preprocess.retain_breast_mask_for_export=true."
            )
        retained = np.asarray(retained, dtype=bool)
        windows = sliding_square_windows(
            image.shape[1], image.shape[0], crop_size, stride, edge_policy=edge_policy
        )
        alternatives = (
            compare_mask_methods(unmasked, config.get("preprocess", {}) or {})
            if compare_methods
            else {}
        )
        sample_dir = root / f"{record_index:05d}_{_safe_name(record.get('image_id', 'image'))}"
        result = create_mask_padding_debug_bundle(
            image=image,
            unmasked_image=unmasked,
            mask=retained,
            windows=windows,
            output_dir=sample_dir,
            crop_size=crop_size,
            min_breast_fraction=threshold,
            comparison=comparison,
            pad_value=pad_value,
            comparison_masks=alternatives,
            metadata={
                "record_index": record_index,
                "image_id": record.get("image_id"),
                "study_id": record.get("study_id"),
                "laterality": record.get("laterality"),
                "view_position": record.get("view_position"),
                "source_split": record.get("split"),
                "preprocessing": target.get("preprocessing", {}),
                "preset_key": resolved_preset,
                "split_policy_inspected": split,
                "crop_size": crop_size,
                "stride": stride,
                "edge_policy": edge_policy,
            },
            max_crop_previews=max_crop_previews,
        )
        all_created.extend(result.created_files)
        image_summaries.append({
            "output_dir": str(sample_dir),
            **result.summary,
        })

    overall = {
        "preset_key": resolved_preset,
        "split_policy_inspected": split,
        "config_path": str(config_path),
        "output_dir": str(root),
        "image_count": len(image_summaries),
        "images": image_summaries,
    }
    index_json = root / "index.json"
    index_json.write_text(
        json.dumps(overall, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    all_created.append(index_json)
    return MaskDebugResult(root, tuple(all_created), overall)


def _dataset_from_config(config: dict[str, Any]) -> VindrMammoDataset:
    paths = dict(config.get("paths", {}) or {})
    image_cfg = dict(config.get("image", {}) or {})
    return VindrMammoDataset(
        data_root=paths.get("data_root"),
        index_level="image",
        split=None,
        read_image=True,
        output_size=None,
        normalize=str(image_cfg.get("normalize", "none")),
        percentile_range=tuple(image_cfg.get("percentile_range", [0.5, 99.5])),
        use_voi_lut=bool(image_cfg.get("use_voi_lut", False)),
        strict_voi_lut=bool(image_cfg.get("strict_voi_lut", False)),
        return_dicom_meta=True,
        validate_paths=bool(config.get("dataset", {}).get("validate_paths", False)),
        preprocess_options=dict(config.get("preprocess", {}) or {}),
        crop_options={"enabled": False},
        show_progress=False,
    )


def _select_record_indices(
    dataset: VindrMammoDataset,
    *,
    image_indices: Iterable[int] | None,
    image_ids: Iterable[str] | None,
    max_images: int,
) -> list[int]:
    count = len(dataset.image_records)
    requested_ids = {str(value) for value in (image_ids or []) if str(value)}
    if requested_ids:
        selected = [
            index
            for index, record in enumerate(dataset.image_records)
            if str(record.get("image_id")) in requested_ids
        ]
        missing = requested_ids - {
            str(dataset.image_records[index].get("image_id")) for index in selected
        }
        if missing:
            raise ValueError(f"Unknown image_id values: {sorted(missing)}")
        return selected
    if image_indices is not None:
        selected = [int(index) for index in image_indices]
        invalid = [index for index in selected if index < 0 or index >= count]
        if invalid:
            raise IndexError(f"Image indices outside [0, {count}): {invalid}")
        return selected
    sample_count = min(max(0, int(max_images)), count)
    if sample_count == 0:
        return []
    return sorted({
        int(round(value))
        for value in np.linspace(0, count - 1, num=sample_count)
    })


def _fraction_passes(value: float, minimum: float, comparison: str) -> bool:
    mode = str(comparison or "greater_than_or_equal").casefold().strip()
    if mode in {"strictly_greater_than", "greater_than", "gt", ">"}:
        return float(value) > float(minimum)
    return float(value) >= float(minimum)


def _display_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.round(np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary
    if cv2 is not None:
        eroded = cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8))
        return binary & ~eroded.astype(bool)
    interior = binary.copy()
    interior[1:-1, 1:-1] &= (
        binary[:-2, 1:-1]
        & binary[2:, 1:-1]
        & binary[1:-1, :-2]
        & binary[1:-1, 2:]
    )
    return binary & ~interior


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _display_uint8(image)
    rgb = np.stack([gray, gray, gray], axis=-1)
    outside = ~np.asarray(mask, dtype=bool)
    rgb[outside] = np.round(rgb[outside].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[_mask_boundary(mask)] = np.array([0, 255, 80], dtype=np.uint8)
    return rgb


def _crop_overlay(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    padding_map: np.ndarray,
) -> np.ndarray:
    rgb = _mask_overlay(image_crop, mask_crop)
    rgb[np.asarray(padding_map, dtype=bool)] = np.array([255, 0, 255], dtype=np.uint8)
    return rgb


def _mask_plus_padding(mask_crop: np.ndarray, padding_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*np.asarray(mask_crop).shape, 3), dtype=np.uint8)
    rgb[np.asarray(mask_crop, dtype=bool)] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[np.asarray(padding_map, dtype=bool)] = np.array([255, 0, 255], dtype=np.uint8)
    return rgb


def _grid_overlay(
    image: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    rows: list[dict[str, Any]],
) -> np.ndarray:
    rgb = np.stack([_display_uint8(image)] * 3, axis=-1)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape
    line_width = max(1, int(round(max(height, width) / 1200)))
    for window, row in zip(windows, rows, strict=True):
        x0, y0, x1, y1 = window
        clipped = (
            max(0, min(width - 1, x0)),
            max(0, min(height - 1, y0)),
            max(0, min(width - 1, x1 - 1)),
            max(0, min(height - 1, y1 - 1)),
        )
        padded = int(row["padding_pixels"]) > 0
        color = (255, 0, 255) if padded else ((0, 255, 80) if row["kept"] else (255, 64, 64))
        draw.rectangle(clipped, outline=color, width=line_width)
    return np.asarray(canvas)


def _representative_window_indices(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    limit: int,
) -> list[int]:
    if not rows or limit <= 0:
        return []
    candidates: list[int] = []
    candidates.extend(
        index for index, row in enumerate(rows) if int(row["padding_pixels"]) > 0
    )
    candidates.extend(
        sorted(
            range(len(rows)),
            key=lambda index: abs(float(rows[index]["breast_fraction"]) - threshold),
        )[:4]
    )
    candidates.extend(
        sorted(range(len(rows)), key=lambda index: float(rows[index]["breast_fraction"]))[:2]
    )
    candidates.extend(
        sorted(
            range(len(rows)),
            key=lambda index: float(rows[index]["breast_fraction"]),
            reverse=True,
        )[:2]
    )
    if len(candidates) < limit:
        candidates.extend(
            int(round(value))
            for value in np.linspace(0, len(rows) - 1, num=min(limit, len(rows)))
        )
    selected: list[int] = []
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) >= limit:
            break
    return selected


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "image")).strip("-.")
    return text[:120] or "image"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the exact retained breast mask, crop padding maps, crop-level "
            "breast fractions, and alternative-mask comparisons for VinDr DICOMs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.cwd() / "config" / "export_config.yaml",
        help="Export config path (default: ./config/export_config.yaml).",
    )
    parser.add_argument(
        "--preset",
        choices=[*PRESET_ALIASES, *STUDY_PRESETS],
        default="custom-paper22",
        help="Preset whose fixed preprocessing and crop rules should be audited.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split-specific breast-fraction rule to display (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "mask_debug",
        help="Diagnostic output directory (default: ./mask_debug).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Exact dataset record indices. By default, records are sampled evenly.",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=None,
        help="Exact source image_id; repeat this option to inspect several images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Number of evenly spaced records when no IDs/indices are supplied.",
    )
    parser.add_argument(
        "--max-crop-previews",
        type=int,
        default=12,
        help="Maximum representative crop bundles written per source image.",
    )
    parser.add_argument(
        "--no-compare-methods",
        action="store_true",
        help="Skip Otsu and percentile comparison masks.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for the mask-and-padding audit utility."""
    args = _build_parser().parse_args(argv)
    result = debug_masks_from_config(
        args.config,
        preset_key=args.preset,
        output_dir=args.output_dir,
        image_indices=args.indices,
        image_ids=args.image_id,
        max_images=args.max_images,
        max_crop_previews=args.max_crop_previews,
        compare_methods=not args.no_compare_methods,
        split=args.split,
    )
    print(f"Mask debug bundle: {result.output_dir}")
    print(f"Images inspected: {result.summary['image_count']}")
    for item in result.summary["images"]:
        metadata = dict(item.get("metadata", {}) or {})
        print(
            "  "
            f"{metadata.get('image_id', 'unknown')}: "
            f"mask={float(item['mask']['mask_fraction']):.2%}, "
            f"windows kept/rejected={item['kept_window_count']}/"
            f"{item['rejected_window_count']}, "
            f"padded={item['padded_window_count']}, "
            f"flags={item['mask']['quality_flags']}"
        )
    print(f"Machine-readable index: {result.output_dir / 'index.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
