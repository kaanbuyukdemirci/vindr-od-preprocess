from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from tqdm.auto import tqdm

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pragma: no cover - compatibility with older pydicom releases.
    from pydicom.pixel_data_handlers.util import apply_modality_lut


DEFAULT_TARGET_HEIGHTS = (512, 640, 800, 1024, 1280, 1600)
MANIFEST_FIELDS = (
    "variant",
    "split",
    "study_id",
    "image_id",
    "source_dicom_path",
    "float32_path",
    "label_path",
    "source_height",
    "source_width",
    "canvas_height",
    "canvas_width",
    "resized_content_height",
    "resized_content_width",
    "output_height",
    "output_width",
    "source_pad_bottom",
    "source_pad_right",
    "output_pad_bottom",
    "output_pad_right",
    "scale_x",
    "scale_y",
    "num_mass_annotations",
    "float32_dtype",
    "float32_layout",
    "float32_shape",
    "float32_written",
    "label_written",
)


@dataclass(frozen=True)
class MassBox:
    annotation_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class SourceRecord:
    study_id: str
    image_id: str
    split: str
    dicom_path: str
    height: int
    width: int
    laterality: str
    view_position: str
    mass_boxes: tuple[MassBox, ...]


@dataclass(frozen=True)
class OutputVariant:
    name: str
    output_height: int
    output_width: int
    resized_content_height: int
    resized_content_width: int
    scale_x: float
    scale_y: float
    is_padded_original: bool = False


@dataclass(frozen=True)
class GeometryPlan:
    maximum_source_height: int
    maximum_source_width: int
    canvas_height: int
    canvas_width: int
    size_divisor: int
    variants: tuple[OutputVariant, ...]


@dataclass(frozen=True)
class WorkerConfig:
    output_root: str
    canvas_height: int
    canvas_width: int
    variants: tuple[OutputVariant, ...]
    overwrite: bool


@dataclass(frozen=True)
class ExportResult:
    rows: tuple[dict[str, Any], ...]
    error: str | None = None


_WORKER_CONFIG: WorkerConfig | None = None


def round_up_to_multiple(value: int, divisor: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return ((int(value) + int(divisor) - 1) // int(divisor)) * int(divisor)


def build_geometry_plan(
    maximum_source_height: int,
    maximum_source_width: int,
    target_heights: Sequence[int] = DEFAULT_TARGET_HEIGHTS,
    *,
    size_divisor: int = 32,
) -> GeometryPlan:
    """Build one common bottom/right-padded canvas and its resized variants.

    The common canvas is the dataset-wide maximum source height and width rounded
    upward independently to ``size_divisor``. Each compact variant is produced by
    resizing that complete canvas to the requested height with the same aspect
    ratio. Its integer content width is rounded to the nearest pixel, then padded
    on the right to ``size_divisor``. This final pad avoids distorting the image.
    """
    maximum_source_height = int(maximum_source_height)
    maximum_source_width = int(maximum_source_width)
    canvas_height = round_up_to_multiple(maximum_source_height, size_divisor)
    canvas_width = round_up_to_multiple(maximum_source_width, size_divisor)
    if canvas_height <= canvas_width:
        raise ValueError(
            "Expected a portrait dataset-wide canvas with height > width, got "
            f"{canvas_height}x{canvas_width} (HxW)."
        )

    unique_heights = tuple(dict.fromkeys(int(value) for value in target_heights))
    if not unique_heights:
        raise ValueError("At least one target height is required.")
    if any(value <= 0 for value in unique_heights):
        raise ValueError(f"Target heights must be positive: {unique_heights}")
    if any(value % size_divisor for value in unique_heights):
        raise ValueError(
            f"Every target height must be divisible by {size_divisor}: {unique_heights}"
        )
    if any(value >= canvas_height for value in unique_heights):
        raise ValueError(
            "Target heights must be smaller than the padded original height "
            f"({canvas_height}): {unique_heights}"
        )

    variants: list[OutputVariant] = [
        OutputVariant(
            name=f"original_h{canvas_height}_w{canvas_width}",
            output_height=canvas_height,
            output_width=canvas_width,
            resized_content_height=canvas_height,
            resized_content_width=canvas_width,
            scale_x=1.0,
            scale_y=1.0,
            is_padded_original=True,
        )
    ]
    for target_height in unique_heights:
        exact_width = canvas_width * target_height / canvas_height
        resized_width = max(1, int(math.floor(exact_width + 0.5)))
        output_width = round_up_to_multiple(resized_width, size_divisor)
        variants.append(
            OutputVariant(
                name=f"h{target_height}_w{output_width}",
                output_height=target_height,
                output_width=output_width,
                resized_content_height=target_height,
                resized_content_width=resized_width,
                scale_x=resized_width / canvas_width,
                scale_y=target_height / canvas_height,
            )
        )

    return GeometryPlan(
        maximum_source_height=maximum_source_height,
        maximum_source_width=maximum_source_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        size_divisor=int(size_divisor),
        variants=tuple(variants),
    )


def pad_bottom_right(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    *,
    value: int = 0,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height > output_height or width > output_width:
        raise ValueError(
            f"Image {height}x{width} does not fit output {output_height}x{output_width} (HxW)."
        )
    output = np.full((output_height, output_width), value, dtype=image.dtype)
    output[:height, :width] = image
    return output


def _parse_finding_categories(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item).strip() for item in parsed)
    return (text,)


def _load_split_map(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Split-assignment CSV not found: {path}")
    frame = pd.read_csv(path, dtype={"study_id": str, "image_id": str})
    required = {"study_id", "image_id", "export_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split-assignment CSV is missing columns {missing}: {path}")
    if frame.duplicated(["study_id", "image_id"]).any():
        raise ValueError(f"Duplicate study/image keys in split-assignment CSV: {path}")
    result = {
        (str(row.study_id), str(row.image_id)): str(row.export_split).strip()
        for row in frame.itertuples(index=False)
    }
    invalid = sorted(set(result.values()) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"Unsupported export splits in {path}: {invalid}")
    return result


def load_source_records(
    data_root: str | Path,
    *,
    split_assignments: str | Path | None = None,
) -> list[SourceRecord]:
    data_root = Path(data_root).resolve()
    breast_path = data_root / "breast-level_annotations.csv"
    finding_path = data_root / "finding_annotations.csv"
    for required_path in (breast_path, finding_path, data_root / "images"):
        if not required_path.exists():
            raise FileNotFoundError(f"Required VinDr source path not found: {required_path}")

    breast = pd.read_csv(
        breast_path,
        dtype={"study_id": str, "image_id": str, "series_id": str},
    )
    required_breast = {
        "study_id",
        "image_id",
        "height",
        "width",
        "split",
        "laterality",
        "view_position",
    }
    missing = sorted(required_breast - set(breast.columns))
    if missing:
        raise ValueError(f"breast-level_annotations.csv is missing columns: {missing}")
    if breast.duplicated(["study_id", "image_id"]).any():
        raise ValueError("breast-level_annotations.csv contains duplicate study/image keys.")

    findings = pd.read_csv(
        finding_path,
        dtype={"study_id": str, "image_id": str, "series_id": str},
    ).reset_index(names="source_annotation_id")
    required_findings = {
        "study_id",
        "image_id",
        "finding_categories",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    missing = sorted(required_findings - set(findings.columns))
    if missing:
        raise ValueError(f"finding_annotations.csv is missing columns: {missing}")

    mass_boxes_by_image: dict[tuple[str, str], list[MassBox]] = {}
    for row in findings.itertuples(index=False):
        if "Mass" not in _parse_finding_categories(row.finding_categories):
            continue
        coordinates = (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Non-finite Mass box for {row.study_id}/{row.image_id}: {coordinates}"
            )
        xmin, ymin, xmax, ymax = coordinates
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"Invalid Mass box for {row.study_id}/{row.image_id}: {coordinates}"
            )
        key = (str(row.study_id), str(row.image_id))
        mass_boxes_by_image.setdefault(key, []).append(
            MassBox(
                annotation_id=int(row.source_annotation_id),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    split_path = Path(split_assignments).resolve() if split_assignments else None
    split_map = _load_split_map(split_path)
    records: list[SourceRecord] = []
    for row in breast.itertuples(index=False):
        study_id = str(row.study_id)
        image_id = str(row.image_id)
        key = (study_id, image_id)
        official_split = str(row.split).strip().casefold()
        fallback_split = {"training": "train", "test": "test"}.get(official_split)
        if fallback_split is None:
            raise ValueError(
                f"Unsupported official split {row.split!r} for {study_id}/{image_id}."
            )
        split = split_map.get(key, fallback_split)
        if split_map and key not in split_map:
            raise ValueError(f"Missing split assignment for {study_id}/{image_id} in {split_path}")
        records.append(
            SourceRecord(
                study_id=study_id,
                image_id=image_id,
                split=split,
                dicom_path=str(data_root / "images" / study_id / f"{image_id}.dicom"),
                height=int(row.height),
                width=int(row.width),
                laterality=str(row.laterality),
                view_position=str(row.view_position),
                mass_boxes=tuple(mass_boxes_by_image.get(key, ())),
            )
        )
    return records


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_geometry_cache(
    path: Path,
    records: Sequence[SourceRecord],
) -> list[SourceRecord] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"study_id": str, "image_id": str})
    required = {"study_id", "image_id", "height", "width", "dicom_path"}
    if required - set(frame.columns) or len(frame) != len(records):
        return None
    cached = {
        (str(row.study_id), str(row.image_id)): (int(row.height), int(row.width), str(row.dicom_path))
        for row in frame.itertuples(index=False)
    }
    updated: list[SourceRecord] = []
    for record in records:
        value = cached.get((record.study_id, record.image_id))
        if value is None or Path(value[2]) != Path(record.dicom_path):
            return None
        updated.append(replace(record, height=value[0], width=value[1]))
    return updated


def scan_dicom_geometry(
    records: Sequence[SourceRecord],
    *,
    cache_path: str | Path | None = None,
    rescan: bool = False,
    show_progress: bool = True,
) -> list[SourceRecord]:
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and not rescan:
        cached = _load_geometry_cache(cache, records)
        if cached is not None:
            print(f"Using cached DICOM geometry: {cache}")
            return cached

    iterator: Iterable[SourceRecord] = records
    if show_progress:
        iterator = tqdm(
            records,
            desc="Scan DICOM headers",
            unit="image",
            dynamic_ncols=True,
        )
    updated: list[SourceRecord] = []
    rows: list[dict[str, Any]] = []
    for record in iterator:
        path = Path(record.dicom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source DICOM not found: {path}")
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=["Rows", "Columns"],
        )
        height = int(dataset.Rows)
        width = int(dataset.Columns)
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid DICOM shape {height}x{width}: {path}")
        updated_record = replace(record, height=height, width=width)
        updated.append(updated_record)
        rows.append(
            {
                "study_id": record.study_id,
                "image_id": record.image_id,
                "dicom_path": record.dicom_path,
                "height": height,
                "width": width,
            }
        )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(cache)
    return updated


def _normalize_dicom_to_uint16(path: str | Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    values = dataset.pixel_array
    try:
        values = apply_modality_lut(values, dataset)
    except Exception:
        # The official VinDr files use identity rescale values. If a particular
        # file has malformed optional LUT metadata, retaining decoded pixels is
        # safer than dropping the image.
        pass
    values = np.asarray(values, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Expected one 2D frame, got {values.shape}: {path}")
    finite = np.isfinite(values)
    if not bool(finite.all()):
        if not bool(finite.any()):
            raise ValueError(f"DICOM contains no finite pixels: {path}")
        values = np.where(finite, values, float(np.nanmin(values[finite])))
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        values = float(values.max()) + float(values.min()) - values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint16)
    normalized = (values - low) / (high - low)
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _atomic_write_float32(path: Path, image: np.ndarray) -> None:
    """Save one grayscale image as contiguous CHW [1,H,W] float32 in [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pt")
    values = image.astype(np.float32)
    values *= np.float32(1.0 / 65535.0)
    tensor = torch.from_numpy(values).unsqueeze(0).contiguous()
    torch.save(tensor, temporary)
    temporary.replace(path)


def _stem(record: SourceRecord) -> str:
    return f"{record.study_id}__{record.image_id}"


def _variant_paths(
    output_root: Path,
    variant: OutputVariant,
    record: SourceRecord,
) -> tuple[Path, Path]:
    filename = _stem(record)
    float32_path = (
        output_root / "images" / "float32" / variant.name / record.split / f"{filename}.pt"
    )
    label_path = output_root / "labels" / variant.name / record.split / f"{filename}.txt"
    return float32_path, label_path


def _transform_box(
    box: MassBox,
    variant: OutputVariant,
    *,
    source_height: int,
    source_width: int,
) -> tuple[float, float, float, float]:
    xmin = min(max(float(box.xmin), 0.0), float(source_width)) * variant.scale_x
    xmax = min(max(float(box.xmax), 0.0), float(source_width)) * variant.scale_x
    ymin = min(max(float(box.ymin), 0.0), float(source_height)) * variant.scale_y
    ymax = min(max(float(box.ymax), 0.0), float(source_height)) * variant.scale_y
    xmin = min(max(xmin, 0.0), float(variant.output_width))
    xmax = min(max(xmax, 0.0), float(variant.output_width))
    ymin = min(max(ymin, 0.0), float(variant.output_height))
    ymax = min(max(ymax, 0.0), float(variant.output_height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Mass box became empty after clipping: {box}")
    return xmin, ymin, xmax, ymax


def _yolo_label_text(record: SourceRecord, variant: OutputVariant) -> str:
    lines: list[str] = []
    for box in record.mass_boxes:
        xmin, ymin, xmax, ymax = _transform_box(
            box,
            variant,
            source_height=record.height,
            source_width=record.width,
        )
        center_x = ((xmin + xmax) * 0.5) / variant.output_width
        center_y = ((ymin + ymax) * 0.5) / variant.output_height
        width = (xmax - xmin) / variant.output_width
        height = (ymax - ymin) / variant.output_height
        lines.append(f"0 {center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _render_variant(canvas: np.ndarray, variant: OutputVariant) -> np.ndarray:
    if variant.is_padded_original:
        return canvas
    resized = cv2.resize(
        canvas,
        (variant.resized_content_width, variant.resized_content_height),
        interpolation=cv2.INTER_AREA,
    )
    return pad_bottom_right(resized, variant.output_height, variant.output_width)


def _init_worker(config: WorkerConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    cv2.setNumThreads(1)


def _export_one(record: SourceRecord) -> ExportResult:
    config = _WORKER_CONFIG
    if config is None:  # pragma: no cover - catches incorrect direct internal use.
        raise RuntimeError("Export worker was not initialized.")
    try:
        output_root = Path(config.output_root)
        work: list[tuple[OutputVariant, Path, Path, bool, bool]] = []
        for variant in config.variants:
            float32_path, label_path = _variant_paths(output_root, variant, record)
            write_float32 = bool(config.overwrite or not float32_path.exists())
            # Ultralytics treats a missing label file as a background image. On
            # this exFAT volume, an empty file consumes one 128 KiB allocation
            # unit, so omitting empty labels saves about 20 GiB across 8 variants.
            write_label = bool(
                record.mass_boxes and (config.overwrite or not label_path.exists())
            )
            work.append(
                (
                    variant,
                    float32_path,
                    label_path,
                    write_float32,
                    write_label,
                )
            )

        canvas: np.ndarray | None = None
        if any(write_float32 for _, _, _, write_float32, _ in work):
            image = _normalize_dicom_to_uint16(record.dicom_path)
            decoded_shape = (int(image.shape[0]), int(image.shape[1]))
            expected_shape = (record.height, record.width)
            if decoded_shape != expected_shape:
                raise ValueError(
                    f"Decoded shape {decoded_shape} does not match header {expected_shape}: "
                    f"{record.dicom_path}"
                )
            canvas = pad_bottom_right(image, config.canvas_height, config.canvas_width)

        rows: list[dict[str, Any]] = []
        for (
            variant,
            float32_path,
            label_path,
            write_float32,
            write_label,
        ) in work:
            if write_float32:
                assert canvas is not None
                rendered = _render_variant(canvas, variant)
                _atomic_write_float32(float32_path, rendered)
            if write_label:
                _atomic_write_text(label_path, _yolo_label_text(record, variant))
            rows.append(
                {
                    "variant": variant.name,
                    "split": record.split,
                    "study_id": record.study_id,
                    "image_id": record.image_id,
                    "source_dicom_path": record.dicom_path,
                    "float32_path": float32_path.relative_to(output_root).as_posix(),
                    "label_path": (
                        label_path.relative_to(output_root).as_posix()
                        if record.mass_boxes
                        else ""
                    ),
                    "source_height": record.height,
                    "source_width": record.width,
                    "canvas_height": config.canvas_height,
                    "canvas_width": config.canvas_width,
                    "resized_content_height": variant.resized_content_height,
                    "resized_content_width": variant.resized_content_width,
                    "output_height": variant.output_height,
                    "output_width": variant.output_width,
                    "source_pad_bottom": config.canvas_height - record.height,
                    "source_pad_right": config.canvas_width - record.width,
                    "output_pad_bottom": variant.output_height - variant.resized_content_height,
                    "output_pad_right": variant.output_width - variant.resized_content_width,
                    "scale_x": variant.scale_x,
                    "scale_y": variant.scale_y,
                    "num_mass_annotations": len(record.mass_boxes),
                    "float32_dtype": "float32",
                    "float32_layout": "CHW",
                    "float32_shape": f"[1, {variant.output_height}, {variant.output_width}]",
                    "float32_written": int(write_float32),
                    "label_written": int(write_label),
                }
            )
        return ExportResult(rows=tuple(rows))
    except Exception:
        return ExportResult(
            rows=(),
            error=(
                f"source={record.study_id}/{record.image_id}\n"
                f"path={record.dicom_path}\n{traceback.format_exc()}"
            ),
        )


def _iter_export_results(
    records: Sequence[SourceRecord],
    config: WorkerConfig,
    *,
    workers: int,
) -> Iterator[ExportResult]:
    if workers <= 1:
        _init_worker(config)
        for record in records:
            yield _export_one(record)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:
        yield from executor.map(_export_one, records, chunksize=1)


def _write_coco_files(
    output_root: Path,
    records: Sequence[SourceRecord],
    variants: Sequence[OutputVariant],
) -> None:
    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for split in ("train", "val", "test"):
            split_records = [record for record in records if record.split == split]
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            annotation_number = 1
            for image_number, record in enumerate(split_records, start=1):
                float32_path, _ = _variant_paths(output_root, variant, record)
                images.append(
                    {
                        "id": image_number,
                        "file_name": float32_path.relative_to(output_root).as_posix(),
                        "float32_path": float32_path.relative_to(output_root).as_posix(),
                        "storage_format": "float32_only",
                        "height": variant.output_height,
                        "width": variant.output_width,
                        "study_id": record.study_id,
                        "image_id": record.image_id,
                    }
                )
                for box in record.mass_boxes:
                    xmin, ymin, xmax, ymax = _transform_box(
                        box,
                        variant,
                        source_height=record.height,
                        source_width=record.width,
                    )
                    width = xmax - xmin
                    height = ymax - ymin
                    annotations.append(
                        {
                            "id": annotation_number,
                            "image_id": image_number,
                            "category_id": 1,
                            "bbox": [xmin, ymin, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "source_annotation_id": box.annotation_id,
                        }
                    )
                    annotation_number += 1
            _atomic_write_json(
                annotation_root / variant.name / f"instances_{split}.json",
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "Mass"}],
                },
            )
def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"No existing ancestor for output path: {path}")
        candidate = parent
    return candidate


def _float32_output_bytes(image_count: int, variants: Sequence[OutputVariant]) -> int:
    pixels_per_image = sum(
        variant.output_height * variant.output_width for variant in variants
    )
    return int(image_count) * int(pixels_per_image) * np.dtype(np.float32).itemsize


def _float32_filesystem_estimate(
    image_count: int,
    positive_image_count: int,
    variants: Sequence[OutputVariant],
    *,
    allocation_unit: int,
) -> int:
    """Estimate allocated bytes for .pt files, positive labels, and metadata."""
    # torch.save adds a small zip-container header around each tensor. Four KiB
    # is conservative for the observed files and is rounded with the payload to
    # the destination filesystem's allocation unit.
    tensor_bytes_per_image = sum(
        round_up_to_multiple(
            variant.output_height * variant.output_width * np.dtype(np.float32).itemsize
            + 4096,
            allocation_unit,
        )
        for variant in variants
    )
    label_bytes = (
        int(positive_image_count) * len(variants) * int(allocation_unit)
    )
    metadata_allowance = 1024**3
    return int(image_count) * tensor_bytes_per_image + label_bytes + metadata_allowance


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_readme(output_root: Path, plan: GeometryPlan, image_count: int) -> None:
    variant_lines = "\n".join(
        f"- `{variant.name}`: {variant.output_height} x {variant.output_width} HxW; "
        f"resized content width {variant.resized_content_width}; "
        f"final right pad {variant.output_width - variant.resized_content_width}px"
        for variant in plan.variants
    )
    text = f"""# VinDr global-padded multi-resolution dataset

This export contains {image_count} original VinDr mammograms. DICOM pixels are read
without a VOI/display LUT, MONOCHROME1 images are inverted to black-background
orientation, and every image is independently min-max normalized to uint16.
No breast crop, masking, mirroring, or geometric augmentation is applied.

The largest source is {plan.maximum_source_height} x {plan.maximum_source_width} HxW.
Every source is anchored at top-left and padded only on the bottom and right to the
common {plan.canvas_height} x {plan.canvas_width} canvas. Compact variants resize that
complete canvas without changing its aspect ratio, then add only the minimum right
padding needed for divisibility by {plan.size_divisor}.

## Variants

{variant_lines}

Every variant is stored only as a contiguous single-channel CHW `[1,H,W]` PyTorch
float32 tensor in `[0,1]`. No PNG and no unpadded image copy are saved. Labels are
Ultralytics YOLO text files with class 0 = Mass. Mass-negative images omit the empty
label file, which Ultralytics interprets as background. Consolidated COCO JSON is
under `annotations/<variant>/`, and `metadata/manifest.csv` records every geometry
transform and storage path.

The exporter is resumable: rerunning without `--overwrite` skips existing final
float32 and label paths. Files are written atomically, so interrupted temporary files
are not mistaken for completed outputs.
"""
    _atomic_write_text(output_root / "README.md", text)


def export_dataset(
    records: Sequence[SourceRecord],
    plan: GeometryPlan,
    *,
    output_root: str | Path,
    workers: int,
    overwrite: bool,
    show_progress: bool,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    status_path = metadata_root / "status.json"
    status = {
        "status": "running",
        "started_at": _now(),
        "pid": os.getpid(),
        "image_count": len(records),
        "workers": workers,
        "output_root": str(output_root),
        "geometry": {
            **asdict(plan),
            "variants": [asdict(variant) for variant in plan.variants],
        },
    }
    _atomic_write_json(status_path, status)

    config = WorkerConfig(
        output_root=str(output_root),
        canvas_height=plan.canvas_height,
        canvas_width=plan.canvas_width,
        variants=plan.variants,
        overwrite=overwrite,
    )
    manifest_path = metadata_root / "manifest.csv"
    partial_manifest = metadata_root / "manifest.partial.csv"
    errors_path = metadata_root / "export_errors.jsonl"
    errors: list[str] = []
    rows_written = 0
    float32_written = 0
    labels_written = 0

    try:
        with partial_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            iterator = _iter_export_results(records, config, workers=workers)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(records),
                    desc="Export mammograms",
                    unit="image",
                    dynamic_ncols=True,
                )
            for result in iterator:
                if result.error:
                    errors.append(result.error)
                    continue
                writer.writerows(result.rows)
                rows_written += len(result.rows)
                float32_written += sum(
                    int(row["float32_written"]) for row in result.rows
                )
                labels_written += sum(int(row["label_written"]) for row in result.rows)
        partial_manifest.replace(manifest_path)

        if errors:
            _atomic_write_text(
                errors_path,
                "".join(json.dumps({"error": value}) + "\n" for value in errors),
            )
        elif errors_path.exists():
            errors_path.unlink()

        _write_coco_files(output_root, records, plan.variants)
        _write_readme(output_root, plan, len(records))
        status.update(
            {
                "status": "completed" if not errors else "completed_with_errors",
                "finished_at": _now(),
                "error_count": len(errors),
                "manifest_rows": rows_written,
                "float32_tensors_written_this_run": float32_written,
                "labels_written_this_run": labels_written,
            }
        )
        _atomic_write_json(status_path, status)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error": repr(exc),
            }
        )
        _atomic_write_json(status_path, status)
        raise

    if errors:
        raise RuntimeError(
            f"Export completed with {len(errors)} failed source images. "
            f"See {errors_path}; rerun the same command after correcting the failures."
        )
    return status


def _copy_source_metadata(data_root: Path, split_assignments: Path | None, output_root: Path) -> None:
    destination = output_root / "metadata" / "source_csv"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (
        data_root / "breast-level_annotations.csv",
        data_root / "finding_annotations.csv",
        data_root / "metadata.csv",
    ):
        shutil.copy2(source, destination / source.name)
    if split_assignments is not None:
        shutil.copy2(split_assignments, destination / "split_assignments.csv")


def _build_parser() -> argparse.ArgumentParser:
    def optional_path(value: str) -> Path | None:
        if value.strip().casefold() in {"none", "official"}:
            return None
        return Path(value)

    parser = argparse.ArgumentParser(
        description=(
            "Export native VinDr mammograms on one dataset-wide stride-32 canvas "
            "plus aspect-preserving multi-resolution variants."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/t9/vindr-data/vindr"),
        help="Official VinDr root containing images/ and the source CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/t9/vindr-data/preprocessed-vindr-global-pad-multires-v1"),
    )
    parser.add_argument(
        "--split-assignments",
        type=optional_path,
        default=Path("/mnt/t9/vindr-data/vindr/global_pad_split_assignments.csv"),
        help=(
            "Existing train/val/test assignment CSV. Pass 'official' to use only "
            "the source dataset's official training/test split."
        ),
    )
    parser.add_argument(
        "--target-heights",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_HEIGHTS),
        help="Compact output heights. Every value must be divisible by --size-divisor.",
    )
    parser.add_argument("--size-divisor", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) // 2)),
        help="Parallel DICOM decode/export workers. Use 1 for sequential debugging.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Ignore cached DICOM header geometry and scan every source again.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan/print geometry and capacity requirements without exporting images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only the first N images after scanning the full dataset; useful for a smoke test.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--allow-low-space",
        action="store_true",
        help=(
            "Proceed when free space is below the estimated T9 allocation for "
            "float32 tensors, positive labels, and metadata."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    split_assignments = args.split_assignments
    if split_assignments is not None:
        split_assignments = split_assignments.resolve()
    records = load_source_records(
        data_root,
        split_assignments=split_assignments,
    )
    geometry_cache = output_root / "metadata" / "source_geometry.csv"
    records = scan_dicom_geometry(
        records,
        cache_path=geometry_cache,
        rescan=args.rescan,
        show_progress=not args.no_progress,
    )
    maximum_height = max(record.height for record in records)
    maximum_width = max(record.width for record in records)
    plan = build_geometry_plan(
        maximum_height,
        maximum_width,
        args.target_heights,
        size_divisor=args.size_divisor,
    )

    print("\nGlobal geometry (H x W)")
    print(f"  Maximum source: {maximum_height} x {maximum_width}")
    print(f"  Padded original: {plan.canvas_height} x {plan.canvas_width}")
    for variant in plan.variants[1:]:
        print(
            f"  {variant.name}: {variant.output_height} x {variant.output_width} "
            f"(resized content {variant.resized_content_height} x "
            f"{variant.resized_content_width}, right pad "
            f"{variant.output_width - variant.resized_content_width})"
        )

    export_records = records[: args.limit] if args.limit is not None else records
    float32_bytes = _float32_output_bytes(len(export_records), plan.variants)
    probe_path = _nearest_existing_ancestor(output_root)
    allocation_unit = int(os.statvfs(probe_path).f_frsize)
    positive_image_count = sum(bool(record.mass_boxes) for record in export_records)
    estimated_allocated_bytes = _float32_filesystem_estimate(
        len(export_records),
        positive_image_count,
        plan.variants,
        allocation_unit=allocation_unit,
    )
    free_bytes = shutil.disk_usage(probe_path).free
    print(f"\nImages selected: {len(export_records):,} / {len(records):,}")
    print(f"Dense single-channel float32 payload: {_format_bytes(float32_bytes)}")
    print(
        "Estimated destination allocation: "
        f"{_format_bytes(estimated_allocated_bytes)} "
        f"({allocation_unit:,}-byte allocation unit)"
    )
    print(f"Currently free at {probe_path}: {_format_bytes(free_bytes)}")

    resolved = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "split_assignments": str(split_assignments) if split_assignments else None,
        "target_heights": list(args.target_heights),
        "size_divisor": args.size_divisor,
        "workers": args.workers,
        "overwrite": args.overwrite,
        "source_image_count": len(records),
        "selected_image_count": len(export_records),
        "float32_channels": 1,
        "float32_layout": "CHW",
        "float32_output_bytes": float32_bytes,
        "destination_allocation_unit_bytes": allocation_unit,
        "estimated_allocated_output_bytes": estimated_allocated_bytes,
        "geometry": {
            **asdict(plan),
            "variants": [asdict(variant) for variant in plan.variants],
        },
    }
    _atomic_write_json(output_root / "metadata" / "resolved_config.json", resolved)

    if args.scan_only:
        print("\nScan-only mode complete; no mammogram pixels were exported.")
        return
    if free_bytes < estimated_allocated_bytes and not args.allow_low_space:
        raise SystemExit(
            "Refusing to start: free space is below the estimated allocation for "
            "float32 tensors, positive labels, and metadata. "
            "Free more space or pass --allow-low-space if prior outputs/resume make this safe."
        )

    _copy_source_metadata(data_root, split_assignments, output_root)
    status = export_dataset(
        export_records,
        plan,
        output_root=output_root,
        workers=args.workers,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print("\nExport complete")
    print(f"  Output: {output_root}")
    print(f"  Manifest rows: {status['manifest_rows']:,}")
    print(
        "  Float32 tensors written this run: "
        f"{status['float32_tensors_written_this_run']:,}"
    )
    print(f"  Labels written this run: {status['labels_written_this_run']:,}")


if __name__ == "__main__":
    main()
