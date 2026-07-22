from __future__ import annotations

import math
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple


Pathish = str | os.PathLike[str]


class _DiskUsageTuple(NamedTuple):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class DiskSpace:
    """Disk usage for the filesystem that will contain a requested output path."""

    requested_path: Path
    probe_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    device_id: int | None = None

    @property
    def free_fraction(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return float(self.free_bytes) / float(self.total_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_path": str(self.requested_path),
            "probe_path": str(self.probe_path),
            "device_id": self.device_id,
            "total_bytes": int(self.total_bytes),
            "used_bytes": int(self.used_bytes),
            "free_bytes": int(self.free_bytes),
            "free_fraction": float(self.free_fraction),
            "total": format_bytes(self.total_bytes),
            "used": format_bytes(self.used_bytes),
            "free": format_bytes(self.free_bytes),
        }


@dataclass(frozen=True)
class ExportSpaceEstimate:
    """A deliberately conservative estimate for one resolved export config.

    PNG compression varies strongly with image content. The estimate therefore
    uses configurable, near-uncompressed bytes-per-pixel assumptions and applies
    a final safety factor. It is intended for admission warnings and queue
    planning, not as a promise of the final byte-exact directory size.
    """

    source_image_count: int
    crop_image_count: int
    baseline_image_count: int
    paired_whole_image_count: int
    model_image_count: int
    model_pixel_count: int
    raw_estimated_bytes: int
    conservative_bytes: int
    safety_factor: float
    breakdown_bytes: dict[str, int]
    assumptions: tuple[str, ...]

    @property
    def estimated_bytes(self) -> int:
        """Alias used by queue/UI code."""

        return int(self.conservative_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_image_count": int(self.source_image_count),
            "crop_image_count": int(self.crop_image_count),
            "baseline_image_count": int(self.baseline_image_count),
            "paired_whole_image_count": int(self.paired_whole_image_count),
            "model_image_count": int(self.model_image_count),
            "model_pixel_count": int(self.model_pixel_count),
            "raw_estimated_bytes": int(self.raw_estimated_bytes),
            "conservative_bytes": int(self.conservative_bytes),
            "estimated_bytes": int(self.conservative_bytes),
            "raw_estimated": format_bytes(self.raw_estimated_bytes),
            "conservative": format_bytes(self.conservative_bytes),
            "safety_factor": float(self.safety_factor),
            "breakdown_bytes": dict(self.breakdown_bytes),
            "breakdown": {
                key: format_bytes(value) for key, value in self.breakdown_bytes.items()
            },
            "assumptions": list(self.assumptions),
        }


def nearest_existing_ancestor(path: Pathish, *, cwd: Pathish | None = None) -> Path:
    """Return ``path`` or its nearest existing parent.

    Export targets commonly do not exist yet. ``shutil.disk_usage`` needs an
    existing path, so disk checks must walk upward without creating anything.
    Relative paths are made absolute against ``cwd`` (or the process cwd).
    """

    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = Path(cwd or Path.cwd()).expanduser() / requested
    requested = requested.resolve(strict=False)

    candidate = requested
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(
                f"Could not find an existing ancestor for output path {requested}"
            )
        candidate = parent
    return candidate


def get_disk_space(
    path: Pathish,
    *,
    cwd: Pathish | None = None,
    disk_usage_func: Callable[[Pathish], Any] | None = None,
) -> DiskSpace:
    """Return filesystem capacity for a possibly non-existent output path."""

    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = (Path(cwd or Path.cwd()).expanduser() / requested).resolve(strict=False)
    else:
        requested = requested.resolve(strict=False)
    probe = nearest_existing_ancestor(requested)
    usage_func = disk_usage_func or shutil.disk_usage
    raw = usage_func(probe)
    usage = _DiskUsageTuple(int(raw.total), int(raw.used), int(raw.free))
    try:
        device_id: int | None = int(probe.stat().st_dev)
    except OSError:
        device_id = None
    return DiskSpace(
        requested_path=requested,
        probe_path=probe,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        device_id=device_id,
    )


def format_bytes(num_bytes: int | float, *, precision: int = 1) -> str:
    """Format a byte count with binary units (KiB, MiB, GiB, ...)."""

    value = float(num_bytes)
    if not math.isfinite(value):
        raise ValueError("num_bytes must be finite")
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{sign}{int(round(value))} B"
    return f"{sign}{value:.{max(0, int(precision))}f} {unit}"


def estimate_export_space(
    config: Mapping[str, Any],
    record_summary: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Any,
) -> ExportSpaceEstimate:
    """Estimate a conservative output size from config and record metadata.

    ``record_summary`` may be a sequence of image records, a mapping containing
    ``records``, or an aggregate mapping. Aggregate mappings understand common
    keys including ``images``/``image_count``, ``positive_images``,
    ``mass_annotations``, ``max_width`` and ``max_height``. A pandas DataFrame is
    accepted through its ``to_dict(orient="records")`` method without importing
    pandas here.

    When individual records are available, their raw dimensions are used. This
    intentionally overestimates exports that crop to the breast foreground.
    Deterministic window counts follow ``square_crops.edge_policy`` exactly.
    Foreground rejection and positive/negative downsampling are ignored for the
    upper bound.
    """

    cfg = dict(config or {})
    storage_cfg = dict(cfg.get("storage_estimate", {}) or {})
    records, aggregate = _normalize_record_summary(record_summary)
    records = _filter_records_for_config(records, cfg)
    assumptions: list[str] = [
        "PNG size is bounded with a near-uncompressed RGB bytes-per-pixel assumption.",
        "Foreground filters and positive/negative downsampling are ignored for the upper bound.",
    ]

    default_width = _positive_int(
        storage_cfg.get("default_image_width", aggregate.get("max_width", aggregate.get("width"))),
        4096,
    )
    default_height = _positive_int(
        storage_cfg.get("default_image_height", aggregate.get("max_height", aggregate.get("height"))),
        4096,
    )

    if records:
        source_image_count = len(records)
        dimensions: list[tuple[int, int]] = []
        missing_dimensions = 0
        for record in records:
            width = _record_dimension(record, "width", default_width)
            height = _record_dimension(record, "height", default_height)
            if not _has_record_dimension(record, "width") or not _has_record_dimension(record, "height"):
                missing_dimensions += 1
            dimensions.append((width, height))
        if missing_dimensions:
            assumptions.append(
                f"{missing_dimensions} source image(s) lacked dimensions and used "
                f"the {default_width}x{default_height} fallback."
            )
    else:
        source_image_count = _aggregate_image_count(aggregate)
        dimensions = [(default_width, default_height)] * source_image_count
        if source_image_count:
            assumptions.append(
                "Only aggregate records were available; all source images use the configured "
                f"{default_width}x{default_height} conservative dimensions."
            )

    export_cfg = dict(cfg.get("export", {}) or {})
    crop_cfg = dict(cfg.get("square_crops", {}) or {})
    save_square = bool(export_cfg.get("save_square_crops", True))
    save_baseline = bool(export_cfg.get("save_baseline_uncropped", False))

    crop_size = _positive_int(crop_cfg.get("crop_size"), 1024)
    stride = _positive_int(crop_cfg.get("stride"), 512)
    crop_output_width, crop_output_height = _output_dimensions(
        crop_cfg,
        default_width=crop_size,
        default_height=crop_size,
        keys=("final_size", "output_size", "resize_size", "final_crop_resize"),
    )

    crop_image_count = 0
    if save_square and source_image_count:
        if records:
            for record, (width, height) in zip(records, dimensions, strict=False):
                crop_image_count += _conservative_record_crop_count(
                    record,
                    width=width,
                    height=height,
                    crop_cfg=crop_cfg,
                    crop_size=crop_size,
                    stride=stride,
                )
        else:
            crop_image_count = _conservative_aggregate_crop_count(
                aggregate,
                source_image_count=source_image_count,
                width=default_width,
                height=default_height,
                crop_cfg=crop_cfg,
                crop_size=crop_size,
                stride=stride,
            )

    baseline_image_count = source_image_count if save_baseline else 0
    baseline_cfg = dict(cfg.get("baseline_uncropped", {}) or {})
    baseline_pixels = 0
    if baseline_image_count:
        resize_mode = str(baseline_cfg.get("resize_mode", "none") or "none").casefold()
        if resize_mode == "none":
            baseline_pixels = sum(width * height for width, height in dimensions)
        else:
            baseline_width, baseline_height = _output_dimensions(
                baseline_cfg,
                default_width=1024,
                default_height=1024,
                keys=("size", "output_size"),
            )
            baseline_pixels = baseline_image_count * baseline_width * baseline_height

    paired_cfg = _paired_whole_config(cfg)
    paired_enabled = bool(paired_cfg.get("enabled", False))
    paired_per_crop = bool(paired_cfg.get("one_per_crop", paired_cfg.get("per_crop", True)))
    paired_whole_image_count = (
        crop_image_count if paired_enabled and paired_per_crop else source_image_count if paired_enabled else 0
    )
    if paired_enabled and str(paired_cfg.get("storage_mode", "hardlink")).casefold() in {
        "hardlink",
        "link",
        "deduplicate",
    }:
        assumptions.append(
            "Paired whole-image companions are costed as full copies so the estimate remains "
            "safe if hard-link creation falls back to copying."
        )
    paired_width, paired_height = _output_dimensions(
        paired_cfg,
        default_width=1024,
        default_height=1024,
        keys=("size", "output_size"),
    )

    crop_pixels = crop_image_count * crop_output_width * crop_output_height
    paired_pixels = paired_whole_image_count * paired_width * paired_height
    model_pixel_count = int(crop_pixels + baseline_pixels + paired_pixels)
    preserved_pixel_count = int(crop_pixels + baseline_pixels)
    model_image_count = int(crop_image_count + baseline_image_count + paired_whole_image_count)

    rgb_bytes_per_pixel = _positive_float(storage_cfg.get("rgb_bytes_per_pixel"), 3.25)
    preserved_bytes_per_pixel = _positive_float(
        storage_cfg.get("preserved_16bit_bytes_per_pixel"), 2.10
    )
    metadata_bytes_per_sample = _nonnegative_int(
        storage_cfg.get("metadata_bytes_per_sample"), 16 * 1024
    )
    metadata_bytes_per_source = _nonnegative_int(
        storage_cfg.get("metadata_bytes_per_source"), 4 * 1024
    )
    fixed_metadata_bytes = _nonnegative_int(
        storage_cfg.get("fixed_metadata_bytes"), 32 * 1024 * 1024
    )
    safety_factor = _positive_float(storage_cfg.get("safety_factor"), 1.15)

    rgb_bytes = int(math.ceil(model_pixel_count * rgb_bytes_per_pixel))
    save_preserved = bool((cfg.get("preserved_16bit", {}) or {}).get("save", True))
    # Paired whole-image companions are RGB-only. Preserved uint16 files are
    # written for normal crop/baseline model images.
    preserved_bytes = (
        int(math.ceil(preserved_pixel_count * preserved_bytes_per_pixel))
        if save_preserved
        else 0
    )
    sample_metadata_bytes = model_image_count * metadata_bytes_per_sample
    source_metadata_bytes = source_image_count * metadata_bytes_per_source
    breakdown = {
        "rgb_images": int(rgb_bytes),
        "preserved_16bit_images": int(preserved_bytes),
        "sample_labels_and_metadata": int(sample_metadata_bytes),
        "source_metadata": int(source_metadata_bytes),
        "fixed_metadata_and_manifests": int(fixed_metadata_bytes),
    }
    raw_estimated_bytes = int(sum(breakdown.values()))
    conservative_bytes = int(math.ceil(raw_estimated_bytes * safety_factor))
    assumptions.append(f"A final {safety_factor:.3g}x safety factor is applied.")

    return ExportSpaceEstimate(
        source_image_count=int(source_image_count),
        crop_image_count=int(crop_image_count),
        baseline_image_count=int(baseline_image_count),
        paired_whole_image_count=int(paired_whole_image_count),
        model_image_count=int(model_image_count),
        model_pixel_count=int(model_pixel_count),
        raw_estimated_bytes=raw_estimated_bytes,
        conservative_bytes=conservative_bytes,
        safety_factor=float(safety_factor),
        breakdown_bytes=breakdown,
        assumptions=tuple(assumptions),
    )


def _normalize_record_summary(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate: dict[str, Any] = {}
    records_value: Any = None
    if isinstance(value, Mapping):
        aggregate = dict(value)
        records_value = aggregate.get("records")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records_value = value
    elif hasattr(value, "to_dict"):
        try:
            records_value = value.to_dict(orient="records")
        except TypeError:
            records_value = value.to_dict("records")

    records: list[dict[str, Any]] = []
    if isinstance(records_value, Iterable) and not isinstance(
        records_value, (str, bytes, bytearray, Mapping)
    ):
        records = [dict(item) for item in records_value if isinstance(item, Mapping)]
    return records, aggregate


def _filter_records_for_config(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply only explicit filters whose required summary fields are present."""

    selected = list(records)
    source_cohort = config.get("source_cohort", {}) or {}
    if isinstance(source_cohort, Mapping) and bool(source_cohort.get("positive_images_only", False)):
        # The improved Paper 22 cohort starts from positive images to assign
        # patient-safe splits, then expands training to all views from those
        # training patients. The GUI's enriched records already carry the final
        # export_split assignment, so retain those exact members for estimation.
        if bool(source_cohort.get("train_expand_to_all_patient_breast_views", False)) and any(
            str(record.get("export_split", "")).casefold() in {"train", "val", "test"}
            for record in selected
        ):
            selected = [
                record
                for record in selected
                if str(record.get("export_split", "")).casefold() in {"train", "val", "test"}
            ]
        elif any("has_mass" in record or "num_masses" in record for record in selected):
            selected = [record for record in selected if _record_mass_count(record) > 0]

    vendor_filter = config.get("vendor_filter", {}) or {}
    if isinstance(vendor_filter, Mapping) and bool(vendor_filter.get("enabled", False)):
        allowed = {
            str(value).strip().casefold()
            for value in (vendor_filter.get("include_vendors", []) or [])
            if str(value).strip()
        }
        if allowed and any("vendor" in record for record in selected):
            selected = [
                record
                for record in selected
                if str(record.get("vendor", "")).strip().casefold() in allowed
            ]
    return selected


def _aggregate_image_count(summary: Mapping[str, Any]) -> int:
    for key in (
        "image_count",
        "images",
        "num_images",
        "num_selected_source_images",
        "num_source_images",
    ):
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0, int(value))
    split_counts = summary.get("splits")
    if isinstance(split_counts, Mapping):
        return sum(_nonnegative_int(value, 0) for value in split_counts.values())
    return 0


def _aggregate_mass_count(summary: Mapping[str, Any]) -> int:
    for key in (
        "mass_annotations",
        "num_mass_annotations",
        "annotation_count",
        "positive_crop_seed_count",
        "positive_images",
    ):
        if key in summary:
            return _nonnegative_int(summary.get(key), 0)
    return 0


def _record_dimension(record: Mapping[str, Any], axis: str, default: int) -> int:
    keys = (axis, f"image_{axis}")
    for key in keys:
        value = record.get(key)
        try:
            parsed = int(float(value))
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            return parsed
    return int(default)


def _has_record_dimension(record: Mapping[str, Any], axis: str) -> bool:
    for key in (axis, f"image_{axis}"):
        try:
            if int(float(record.get(key))) > 0:
                return True
        except (TypeError, ValueError, OverflowError):
            continue
    return False


def _record_mass_count(record: Mapping[str, Any]) -> int:
    for key in ("num_masses", "mass_count", "mass_annotations", "num_mass_annotations"):
        if key in record:
            return _nonnegative_int(record.get(key), 0)
    for key in ("mass_boxes", "boxes"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value)
    return 1 if bool(record.get("has_mass", False)) else 0


def _conservative_record_crop_count(
    record: Mapping[str, Any],
    *,
    width: int,
    height: int,
    crop_cfg: Mapping[str, Any],
    crop_size: int,
    stride: int,
) -> int:
    split_value = str(record.get("export_split", record.get("split", "")) or "").casefold()
    if split_value in {"train", "val", "test"}:
        splits = (split_value,)
    elif split_value == "training":
        splits = ("train", "val")
    else:
        splits = ("train", "val", "test")
    mass_count = _record_mass_count(record)
    return max(
        _conservative_crop_count_for_mode(
            split,
            width=width,
            height=height,
            mass_count=mass_count,
            crop_cfg=crop_cfg,
            crop_size=crop_size,
            stride=stride,
        )
        for split in splits
    )


def _conservative_aggregate_crop_count(
    summary: Mapping[str, Any],
    *,
    source_image_count: int,
    width: int,
    height: int,
    crop_cfg: Mapping[str, Any],
    crop_size: int,
    stride: int,
) -> int:
    mass_count = _aggregate_mass_count(summary)
    grid_per_image = _conservative_sliding_grid_count(
        width,
        height,
        crop_size,
        stride,
        edge_policy=str(crop_cfg.get("edge_policy", "edge_align")),
    )
    deterministic_total = source_image_count * grid_per_image

    random_per_annotation = _positive_int(crop_cfg.get("random_crops_per_annotation"), 5)
    bbox_per_annotation = _positive_int(
        crop_cfg.get("bbox_safe_crops_per_annotation"), random_per_annotation
    )
    positive_candidates = mass_count * max(random_per_annotation, bbox_per_annotation)
    negative_candidates_per_image = _max_configured_negative_candidates(crop_cfg)
    random_total = positive_candidates + source_image_count * negative_candidates_per_image
    target_ratio = min(
        _positive_float(crop_cfg.get("positive_fraction"), 0.5),
        1.0,
    )
    if positive_candidates and target_ratio > 0:
        random_total = max(random_total, int(math.ceil(positive_candidates / target_ratio)))

    active_totals: list[int] = []
    for split in ("train", "val", "test"):
        mode = str(crop_cfg.get(f"{split}_crop_mode", "deterministic") or "deterministic").casefold()
        active_totals.append(deterministic_total if mode == "deterministic" else random_total)
    return max(active_totals, default=0)


def _conservative_crop_count_for_mode(
    split: str,
    *,
    width: int,
    height: int,
    mass_count: int,
    crop_cfg: Mapping[str, Any],
    crop_size: int,
    stride: int,
) -> int:
    default_mode = "random" if split == "train" else "deterministic"
    mode = str(crop_cfg.get(f"{split}_crop_mode", default_mode) or default_mode).casefold()
    if mode == "deterministic":
        count = _conservative_sliding_grid_count(
            width,
            height,
            crop_size,
            stride,
            edge_policy=str(crop_cfg.get("edge_policy", "edge_align")),
        )
        max_windows = crop_cfg.get(
            f"{split}_deterministic_max_windows_per_image",
            crop_cfg.get("deterministic_max_windows_per_image"),
        )
        if max_windows is not None:
            count = min(count, _nonnegative_int(max_windows, count))
        return count

    per_annotation = _positive_int(
        crop_cfg.get("bbox_safe_crops_per_annotation")
        if mode == "bbox_safe_random"
        else crop_cfg.get("random_crops_per_annotation"),
        _positive_int(crop_cfg.get("random_crops_per_annotation"), 5),
    )
    positives = mass_count * per_annotation
    ratio = _split_positive_ratio(crop_cfg, split, bbox_safe=mode == "bbox_safe_random")
    balanced_negatives = (
        int(math.ceil(positives * (1.0 - ratio) / ratio))
        if positives > 0 and 0.0 < ratio < 1.0
        else 0
    )
    candidate_negatives = _max_configured_negative_candidates(crop_cfg, bbox_safe=mode == "bbox_safe_random")
    return positives + max(balanced_negatives, candidate_negatives)


def _conservative_sliding_grid_count(
    width: int,
    height: int,
    crop_size: int,
    stride: int,
    *,
    edge_policy: str,
) -> int:
    def edge_aligned_axis(length: int) -> int:
        if length <= crop_size:
            return 1
        return int(math.ceil((length - crop_size) / stride)) + 1

    def padded_axis(length: int) -> int:
        if length <= crop_size:
            return 1
        # ``regular_stride_pad`` advances from zero only until a crop covers
        # the far edge. The last origin stays on the stride grid and its crop
        # may extend beyond the image; it does not keep emitting origins all
        # the way to ``length - 1``.
        return int(math.ceil((length - crop_size) / stride)) + 1

    policy = str(edge_policy or "edge_align").casefold().strip()
    if policy in {"regular_stride_pad", "pad"}:
        return padded_axis(width) * padded_axis(height)
    return edge_aligned_axis(width) * edge_aligned_axis(height)


def _split_positive_ratio(
    crop_cfg: Mapping[str, Any], split: str, *, bbox_safe: bool
) -> float:
    keys = []
    if bbox_safe:
        keys.append(f"{split}_bbox_safe_positive_fraction")
    keys.extend(
        [
            f"{split}_positive_fraction",
            f"{split}_deterministic_target_positive_ratio",
            "positive_fraction",
            "deterministic_target_positive_ratio",
        ]
    )
    for key in keys:
        if key in crop_cfg and crop_cfg.get(key) is not None:
            try:
                value = float(crop_cfg.get(key))
            except (TypeError, ValueError):
                continue
            return min(max(value, 0.001), 1.0)
    return 0.5


def _max_configured_negative_candidates(
    crop_cfg: Mapping[str, Any], *, bbox_safe: bool = False
) -> int:
    keys = [
        "global_negative_candidate_crops_per_image_when_balancing",
        "random_crops_per_negative_image_when_balancing",
        "random_crops_per_negative_image",
        "random_crops_per_image",
    ]
    if bbox_safe:
        keys = [
            "bbox_safe_random_crops_per_negative_image_when_balancing",
            "bbox_safe_random_crops_per_negative_image",
            *keys,
        ]
    values = [_nonnegative_int(crop_cfg.get(key), 0) for key in keys if key in crop_cfg]
    return max(values, default=1)


def _paired_whole_config(config: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "paired_whole_images",
        "corresponding_whole_images",
        "whole_image_companions",
    ):
        value = config.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    square = config.get("square_crops", {}) or {}
    enabled = bool(
        square.get("save_corresponding_whole_images", False)
        if isinstance(square, Mapping)
        else False
    )
    return {"enabled": enabled, "one_per_crop": True, "size": 1024}


def _output_dimensions(
    config: Mapping[str, Any],
    *,
    default_width: int,
    default_height: int,
    keys: Sequence[str],
) -> tuple[int, int]:
    width = config.get("target_width")
    height = config.get("target_height")
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) >= 2:
                height = value[0]
                width = value[1]
                break
        else:
            width = value
            height = value
            break
    return _positive_int(width, default_width), _positive_int(height, default_height)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        parsed = int(default)
    return parsed if parsed > 0 else int(default)


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        parsed = int(default)
    return max(0, parsed)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = float(default)
    if not math.isfinite(parsed) or parsed <= 0:
        return float(default)
    return parsed


__all__ = [
    "DiskSpace",
    "ExportSpaceEstimate",
    "estimate_export_space",
    "format_bytes",
    "get_disk_space",
    "nearest_existing_ancestor",
]
