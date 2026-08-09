from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GROUPED_DATASET_LAYOUT = "images_annotations_v1"


def uses_grouped_dataset_layout(config: Mapping[str, Any] | None) -> bool:
    """Return whether an export uses the scalable images/annotations layout."""

    layout = dict((config or {}).get("dataset_layout", {}) or {})
    return str(layout.get("kind", "") or "").strip().casefold() == GROUPED_DATASET_LAYOUT


def dataset_content_root(output_root: str | Path, config: Mapping[str, Any] | None) -> Path:
    """Return the root below which image paths and whole-image metadata live.

    New exports use the export root directly. Legacy exports retain the historical
    ``square_crops`` container so old configurations and completed datasets remain
    readable.
    """

    root = Path(output_root)
    return root if uses_grouped_dataset_layout(config) else root / "square_crops"


def resolve_existing_dataset_content_root(path: str | Path) -> tuple[Path, Path]:
    """Resolve ``(export_root, content_root)`` for new and legacy datasets."""

    selected = Path(path).expanduser().resolve(strict=False)
    if _has_whole_metadata(selected):
        export_root = selected.parent if selected.name == "square_crops" else selected
        return export_root, selected
    legacy = selected / "square_crops"
    if _has_whole_metadata(legacy):
        return selected, legacy
    required = [
        selected / "metadata" / "whole_image_manifest.csv",
        selected / "annotations" / "whole_image_annotations.csv",
        selected / "metadata" / "whole_image_annotations.csv",
        legacy / "metadata" / "whole_image_manifest.csv",
        legacy / "metadata" / "whole_image_annotations.csv",
    ]
    raise FileNotFoundError(
        "Dataset requires whole-image metadata in <root>/metadata (new layout) "
        "or <root>/square_crops/metadata (legacy layout); checked: "
        + ", ".join(str(item) for item in required)
    )


def is_grouped_content_root(content_root: str | Path) -> bool:
    root = Path(content_root)
    return root.name != "square_crops" and (root / "images").exists()


def resized_variant_configs(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize one or many resized-whole definitions.

    ``paired_whole_images.resized_variants`` accepts either a list of mappings or
    a mapping keyed by a display/storage name. Legacy ``target_width`` and
    ``target_height`` values are converted into a single variant.
    """

    cfg = dict(config or {})
    raw = cfg.get("resized_variants")
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            item = dict(value or {}) if isinstance(value, Mapping) else {"size": value}
            item.setdefault("name", str(name))
            entries.append(item)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for value in raw:
            if isinstance(value, Mapping):
                entries.append(dict(value))
            else:
                entries.append({"size": value})

    if not entries:
        enabled = bool(
            cfg.get("save_resized")
            if "save_resized" in cfg
            else cfg.get("enabled", False)
        )
        if not enabled:
            return []
        entries = [{
            "width": cfg.get("target_width", cfg.get("size", 1024)),
            "height": cfg.get("target_height", cfg.get("size", 1024)),
            "name": cfg.get("resized_variant_name"),
        }]

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not bool(entry.get("enabled", True)):
            continue
        size = entry.get("size")
        width = int(entry.get("width", entry.get("target_width", size or 0)) or 0)
        height = int(entry.get("height", entry.get("target_height", size or 0)) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Every resized whole-image variant needs positive width and height")
        raw_name = str(entry.get("name") or f"{width}x{height}").strip()
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
        if not name:
            name = f"{width}x{height}"
        if name in seen:
            raise ValueError(f"Duplicate resized whole-image variant name: {name!r}")
        seen.add(name)
        normalized.append({
            **entry,
            "name": name,
            "width": width,
            "height": height,
            "target_width": width,
            "target_height": height,
            "save_float32": bool(entry.get("save_float32", True)),
        })
    return normalized


def resized_sizes_text(config: Mapping[str, Any] | None) -> str:
    values = resized_variant_configs(config)
    return ", ".join(
        str(item["width"])
        if int(item["width"]) == int(item["height"])
        else f"{item['width']}x{item['height']}"
        for item in values
    )


def parse_resized_sizes(value: Any) -> list[dict[str, Any]]:
    """Parse GUI text such as ``1024, 640`` or ``1024x768, 640x640``."""

    tokens = [token.strip() for token in re.split(r"[,;\n]+", str(value or ""))]
    variants: list[dict[str, Any]] = []
    for token in tokens:
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*[xX]\s*(\d+))?", token)
        if match is None:
            raise ValueError(
                f"Invalid resized size {token!r}; use comma-separated values such as 1024, 640"
            )
        width = int(match.group(1))
        height = int(match.group(2) or width)
        variants.append({
            "name": f"{width}x{height}",
            "width": width,
            "height": height,
            "save_float32": True,
        })
    if not variants:
        raise ValueError("At least one resized whole-image size is required")
    # Reuse canonical validation and duplicate detection.
    return resized_variant_configs({"resized_variants": variants})


def window_grid_configs(config: Mapping[str, Any] | None) -> list[dict[str, int]]:
    raw = (config or {}).get("lazy_crop_grids", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("lazy_crop_grids must be a list of window_size/stride mappings")
    out: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("Each lazy crop grid must be a mapping")
        window = int(value.get("window_size", value.get("size", 0)) or 0)
        stride = int(value.get("stride", 0) or 0)
        if window <= 0 or stride <= 0:
            raise ValueError("Every lazy crop grid needs positive window_size and stride")
        key = (window, stride)
        if key in seen:
            raise ValueError(f"Duplicate lazy crop grid: window={window}, stride={stride}")
        seen.add(key)
        out.append({"window_size": window, "stride": stride})
    return out


def parse_window_grids(value: Any) -> list[dict[str, int]]:
    """Parse GUI text such as ``1024:128, 1024:256, 640:160``."""

    tokens = [token.strip() for token in re.split(r"[,;\n]+", str(value or ""))]
    grids: list[dict[str, int]] = []
    for token in tokens:
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[:/]\s*(\d+)", token)
        if match is None:
            raise ValueError(
                f"Invalid window grid {token!r}; use window:stride pairs such as 1024:128, 640:160"
            )
        grids.append({"window_size": int(match.group(1)), "stride": int(match.group(2))})
    if not grids:
        raise ValueError("At least one window:stride pair is required")
    return window_grid_configs({"lazy_crop_grids": grids})


def window_grids_text(config: Mapping[str, Any] | None) -> str:
    return ", ".join(
        f"{item['window_size']}:{item['stride']}"
        for item in window_grid_configs(config)
    )


def _has_whole_metadata(root: Path) -> bool:
    metadata = root / "metadata"
    return (
        (metadata / "whole_image_manifest.csv").is_file()
        and (
            (root / "annotations" / "whole_image_annotations.csv").is_file()
            or (metadata / "whole_image_annotations.csv").is_file()
        )
    )


__all__ = [
    "GROUPED_DATASET_LAYOUT",
    "dataset_content_root",
    "is_grouped_content_root",
    "parse_resized_sizes",
    "parse_window_grids",
    "resized_sizes_text",
    "resized_variant_configs",
    "resolve_existing_dataset_content_root",
    "uses_grouped_dataset_layout",
    "window_grid_configs",
    "window_grids_text",
]
