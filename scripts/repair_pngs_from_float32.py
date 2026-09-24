#!/usr/bin/env python3
"""Regenerate exported RGB PNGs from their authoritative float32 tensors.

The repair is resumable: an already-correct PNG is skipped. Changed PNGs are
written to a temporary file in the destination directory and atomically moved
into place only after encoding succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path(
    "/mnt/t9/vindr-data/preprocessed-vindr-default-research-dataset-v2"
)


@dataclass(frozen=True)
class RepairResult:
    tensor_path: str
    png_path: str
    status: str
    pixels: int = 0
    differing_values: int = 0
    absolute_difference_sum: int = 0
    maximum_absolute_difference: int = 0
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate resized RGB PNGs by applying the one final uint8 "
            "quantization to matching CHW float32 tensors."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Export root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["1024x1024", "640x640"],
        help="Resized variants to repair.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to repair.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent tensor readers/PNG encoders. Default: 4.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        choices=range(10),
        default=6,
        metavar="0..9",
        help="PNG compression level. Default: 6 (Pillow default).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tensors; useful for testing.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace differing PNGs. Without this flag, only check them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first per-file error.",
    )
    return parser.parse_args()


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(value).__name__}.")
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW [3,H,W], got {tuple(tensor.shape)}.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("Tensor contains NaN or infinite values.")
    return tensor.contiguous()


def _final_png_pixels(tensor: torch.Tensor) -> np.ndarray:
    return (
        torch.round(tensor.clamp(0.0, 1.0) * 255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def _read_png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _atomic_save_png(pixels: np.ndarray, path: Path, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.repair-",
            suffix=".png",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        Image.fromarray(pixels, mode="RGB").save(
            temporary_path,
            format="PNG",
            compress_level=int(compress_level),
        )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _repair_one(
    tensor_path: Path,
    png_path: Path,
    *,
    write: bool,
    compress_level: int,
) -> RepairResult:
    try:
        expected = _final_png_pixels(_load_tensor(tensor_path))
        existing = _read_png(png_path)
        if existing is None:
            if write:
                _atomic_save_png(expected, png_path, compress_level)
            return RepairResult(
                tensor_path=str(tensor_path),
                png_path=str(png_path),
                status="created" if write else "missing",
                pixels=int(expected.size),
                differing_values=int(expected.size),
                maximum_absolute_difference=255,
            )
        if existing.shape != expected.shape:
            if write:
                _atomic_save_png(expected, png_path, compress_level)
            return RepairResult(
                tensor_path=str(tensor_path),
                png_path=str(png_path),
                status="replaced_shape_mismatch" if write else "shape_mismatch",
                pixels=int(expected.size),
                differing_values=int(expected.size),
                maximum_absolute_difference=255,
            )

        difference = np.abs(
            existing.astype(np.int16) - expected.astype(np.int16)
        )
        differing_values = int(np.count_nonzero(difference))
        if differing_values == 0:
            return RepairResult(
                tensor_path=str(tensor_path),
                png_path=str(png_path),
                status="already_correct",
                pixels=int(expected.size),
            )
        if write:
            _atomic_save_png(expected, png_path, compress_level)
        return RepairResult(
            tensor_path=str(tensor_path),
            png_path=str(png_path),
            status="repaired" if write else "would_repair",
            pixels=int(expected.size),
            differing_values=differing_values,
            absolute_difference_sum=int(difference.sum(dtype=np.int64)),
            maximum_absolute_difference=int(difference.max()),
        )
    except Exception as exc:
        return RepairResult(
            tensor_path=str(tensor_path),
            png_path=str(png_path),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _discover_pairs(
    dataset_root: Path,
    resolutions: list[str],
    splits: list[str],
) -> tuple[list[tuple[Path, Path]], list[str]]:
    float_root = dataset_root / "images" / "float32" / "resized"
    png_root = dataset_root / "images" / "resized"
    pairs: list[tuple[Path, Path]] = []
    problems: list[str] = []
    for resolution in resolutions:
        for split in splits:
            tensor_dir = float_root / resolution / split
            png_dir = png_root / resolution / split
            if not tensor_dir.is_dir():
                problems.append(f"Missing tensor directory: {tensor_dir}")
                continue
            tensor_paths = sorted(tensor_dir.glob("*.pt"))
            if not tensor_paths:
                problems.append(f"No tensors found: {tensor_dir}")
                continue
            tensor_stems = {path.stem for path in tensor_paths}
            png_stems = {path.stem for path in png_dir.glob("*.png")} if png_dir.is_dir() else set()
            orphan_pngs = sorted(png_stems - tensor_stems)
            if orphan_pngs:
                problems.append(
                    f"{png_dir}: {len(orphan_pngs)} PNG(s) have no matching tensor"
                )
            pairs.extend(
                (tensor_path, png_dir / f"{tensor_path.stem}.png")
                for tensor_path in tensor_paths
            )
    return pairs, problems


def _write_report(dataset_root: Path, report: dict[str, Any]) -> Path:
    repair_root = dataset_root / "repairs"
    repair_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = repair_root / f"png_from_float32_{timestamp}.json"
    temporary_path = repair_root / f".{report_path.name}.tmp"
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, report_path)
    return report_path


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    pairs, discovery_problems = _discover_pairs(
        dataset_root,
        list(args.resolutions),
        list(args.splits),
    )
    if args.limit is not None:
        pairs = pairs[: max(0, int(args.limit))]
    if not pairs:
        raise RuntimeError("No tensor/PNG pairs were discovered.")
    if discovery_problems:
        print("Preflight warnings:")
        for problem in discovery_problems:
            print(f"  - {problem}")

    print(f"Dataset: {dataset_root}")
    print(f"Pairs: {len(pairs):,}")
    print(f"Mode: {'WRITE (atomic in-place replacement)' if args.write else 'CHECK ONLY'}")
    print(f"Workers: {args.workers}")

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    counts: dict[str, int] = {}
    pixels = 0
    differing_values = 0
    absolute_difference_sum = 0
    maximum_absolute_difference = 0
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(
                _repair_one,
                tensor_path,
                png_path,
                write=bool(args.write),
                compress_level=int(args.compress_level),
            ): (tensor_path, png_path)
            for tensor_path, png_path in pairs
        }
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            unit="image",
            dynamic_ncols=True,
            desc="Repairing PNGs" if args.write else "Checking PNGs",
        )
        for future in progress:
            result = future.result()
            counts[result.status] = counts.get(result.status, 0) + 1
            pixels += int(result.pixels)
            differing_values += int(result.differing_values)
            absolute_difference_sum += int(result.absolute_difference_sum)
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                int(result.maximum_absolute_difference),
            )
            if result.error:
                errors.append({
                    "tensor_path": result.tensor_path,
                    "png_path": result.png_path,
                    "error": result.error,
                })
            progress.set_postfix(
                repaired=counts.get("repaired", 0),
                correct=counts.get("already_correct", 0),
                errors=counts.get("error", 0),
                refresh=False,
            )
            if args.fail_fast and result.error:
                for pending in futures:
                    pending.cancel()
                break

    finished_at = datetime.now(timezone.utc)
    duration_seconds = time.monotonic() - started_monotonic
    report: dict[str, Any] = {
        "schema_version": 1,
        "operation": "final_uint8_png_encoding_from_authoritative_float32_tensor",
        "dataset_root": str(dataset_root),
        "write_enabled": bool(args.write),
        "resolutions": list(args.resolutions),
        "splits": list(args.splits),
        "workers": int(args.workers),
        "png_compress_level": int(args.compress_level),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "pairs_requested": len(pairs),
        "counts": dict(sorted(counts.items())),
        "comparison": {
            "channel_values_compared": pixels,
            "differing_channel_values": differing_values,
            "differing_fraction": (
                float(differing_values) / float(pixels) if pixels else 0.0
            ),
            "mean_absolute_difference_0_255": (
                float(absolute_difference_sum) / float(pixels) if pixels else 0.0
            ),
            "maximum_absolute_difference_0_255": maximum_absolute_difference,
        },
        "discovery_warnings": discovery_problems,
        "errors": errors[:100],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", "unknown"),
        },
    }
    report_path = _write_report(dataset_root, report)

    print("\nCompleted")
    print(json.dumps(report["counts"], indent=2))
    print(f"Duration: {duration_seconds / 60.0:.2f} minutes")
    print(f"Report: {report_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
