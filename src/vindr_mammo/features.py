from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


ProgressCallback = Callable[[dict[str, Any]], None]


DEFAULT_DINO_V3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_DINO_V3_COMPUTE_DTYPE = "float32"
DEFAULT_DINO_V3_INPUT_SIZE = 1024
DEFAULT_RESEARCH_DATASET_FOLDER = "preprocessed-vindr-default-research-dataset-v2"
DINO_V3_LVD_MEAN = (0.485, 0.456, 0.406)
DINO_V3_LVD_STD = (0.229, 0.224, 0.225)


DINO_V3_MODELS: dict[str, dict[str, Any]] = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-S/16 — LVD-1689M (21M)",
        "hidden_size": 384,
        "layers": 12,
        "patch_size": 16,
        "register_tokens": 4,
    },
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-S+/16 — LVD-1689M (29M)",
        "hidden_size": 384,
        "layers": 12,
        "patch_size": 16,
        "register_tokens": 4,
    },
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-B/16 — LVD-1689M (86M)",
        "hidden_size": 768,
        "layers": 12,
        "patch_size": 16,
        "register_tokens": 4,
    },
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-L/16 — LVD-1689M (300M)",
        "hidden_size": 1024,
        "layers": 24,
        "patch_size": 16,
        "register_tokens": 4,
    },
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-H+/16 — LVD-1689M (840M)",
        "hidden_size": 1280,
        "layers": 32,
        "patch_size": 16,
        "register_tokens": 4,
    },
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
        "label": "DINOv3 ViT-7B/16 — LVD-1689M (6.7B)",
        "hidden_size": 4096,
        "layers": 40,
        "patch_size": 16,
        "register_tokens": 4,
    },
}


VARIANT_SPECS: dict[str, dict[str, str]] = {
    "crops": {
        "label": "1024 crops",
        "dataset_subdir": "square_crops",
        "image_subdir": "images",
    },
    "resized_whole": {
        "label": "Original image resized down to 1024",
        "dataset_subdir": "square_crops",
        "image_subdir": "whole_images",
    },
    "original_whole": {
        "label": "Original-size processed image",
        "dataset_subdir": "square_crops",
        "image_subdir": "whole_images_original",
    },
    "high_resolution_whole": {
        "label": "High-resolution padded whole image",
        "dataset_subdir": "square_crops",
        "image_subdir": "whole_images_high_resolution",
    },
    "baseline_whole": {
        "label": "Baseline whole image",
        "dataset_subdir": "baseline_uncropped",
        "image_subdir": "images",
    },
}


@dataclass(frozen=True)
class DatasetImage:
    variant: str
    split: str
    relative_path: str
    png_path: str
    float32_path: str
    has_png: bool
    has_float32: bool


def resolve_dataset_root(path: str | Path) -> Path:
    """Resolve either an export root or its ``square_crops`` child."""

    root = Path(path).expanduser().resolve(strict=False)
    return root.parent if root.name == "square_crops" else root


def default_feature_dataset_root(config: Mapping[str, Any]) -> Path:
    """Return the Default Research Dataset beside the configured export root."""

    paths = dict(config.get("paths", {}) or {})
    configured = Path(
        str(
            paths.get(
                "output_root",
                f"/mnt/t9/vindr-data/{DEFAULT_RESEARCH_DATASET_FOLDER}",
            )
        )
    ).expanduser()
    if configured.name == DEFAULT_RESEARCH_DATASET_FOLDER:
        return configured.resolve(strict=False)
    return (configured.parent / DEFAULT_RESEARCH_DATASET_FOLDER).resolve(strict=False)


def scan_dataset_image_variants(path: str | Path) -> dict[str, Any]:
    """Return the image types and float32 companions present in an export."""

    dataset_root = resolve_dataset_root(path)
    variants: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = dict(VARIANT_SPECS)
    grouped_images = dataset_root / "images"
    if grouped_images.is_dir():
        specs = {
            "original_whole": {
                "label": "Original-size processed image",
                "image_root": grouped_images / "original",
                "float_root": grouped_images / "float32" / "original",
            }
        }
        resized_root = grouped_images / "resized"
        if resized_root.is_dir():
            for resolution_dir in sorted(
                (item for item in resized_root.iterdir() if item.is_dir()),
                key=lambda item: item.name,
            ):
                key = f"resized_whole_{resolution_dir.name}"
                specs[key] = {
                    "label": f"Whole image resized to {resolution_dir.name}",
                    "image_root": resolution_dir,
                    "float_root": grouped_images / "float32" / "resized" / resolution_dir.name,
                }
    for key, spec in specs.items():
        if "image_root" in spec:
            image_root = Path(spec["image_root"])
            float_root = Path(spec["float_root"])
        else:
            container = dataset_root / spec["dataset_subdir"]
            image_root = container / spec["image_subdir"]
            float_root = container / "float32" / spec["image_subdir"]
        png_paths = _image_paths(image_root)
        float_paths = sorted(float_root.rglob("*.pt")) if float_root.exists() else []
        relative_png = {
            path.relative_to(image_root).with_suffix("").as_posix(): path for path in png_paths
        }
        relative_float = {
            path.relative_to(float_root).with_suffix("").as_posix(): path for path in float_paths
        }
        keys = sorted(set(relative_png) | set(relative_float))
        if not keys:
            continue
        split_counts: dict[str, int] = defaultdict(int)
        rows: list[DatasetImage] = []
        for relative_key in keys:
            png = relative_png.get(relative_key)
            float_path = relative_float.get(relative_key)
            relative = Path(relative_key)
            split = relative.parts[0] if len(relative.parts) > 1 else "all"
            split_counts[split] += 1
            if png is None:
                png = image_root / relative.with_suffix(".png")
            if float_path is None:
                float_path = float_root / relative.with_suffix(".pt")
            rows.append(
                DatasetImage(
                    variant=key,
                    split=str(split),
                    relative_path=relative.with_suffix(".png").as_posix(),
                    png_path=str(png),
                    float32_path=str(float_path),
                    has_png=png.exists(),
                    has_float32=float_path.exists(),
                )
            )
        variants[key] = {
            "key": key,
            "label": spec["label"],
            "count": len(rows),
            "png_count": sum(row.has_png for row in rows),
            "float32_count": sum(row.has_float32 for row in rows),
            "missing_float32_count": sum(not row.has_float32 for row in rows),
            "splits": dict(sorted(split_counts.items())),
            "image_root": str(image_root),
            "float32_root": str(float_root),
            "images": [asdict(row) for row in rows],
        }
    return {
        "ok": bool(variants),
        "dataset_root": str(dataset_root),
        "square_crops_root": str(dataset_root / "square_crops"),
        "content_root": str(dataset_root if grouped_images.is_dir() else dataset_root / "square_crops"),
        "layout": "images_annotations_v1" if grouped_images.is_dir() else "legacy_square_crops",
        "is_default_research_dataset": _is_default_research_dataset(dataset_root),
        "variants": variants,
        "total_images": sum(int(item["count"]) for item in variants.values()),
    }


def default_selected_variants(scan: Mapping[str, Any]) -> list[str]:
    """Select every available type, except research-preset original wholes."""

    available = list(dict(scan.get("variants", {}) or {}))
    if bool(scan.get("is_default_research_dataset", False)):
        return [key for key in available if key != "original_whole"]
    return available


def estimate_dataset_channel_stats(
    path: str | Path,
    *,
    variants: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
    max_images: int = 32,
    prefer_float32_sources: bool = True,
    seed: int = 123,
) -> dict[str, Any]:
    """Estimate pixel-weighted RGB moments from a deterministic image sample."""

    scan = scan_dataset_image_variants(path)
    selected_variants = list(variants or default_selected_variants(scan))
    selected_splits = {
        str(value) for value in (splits or []) if str(value) and str(value) != "all"
    }
    images: list[DatasetImage] = []
    for variant in selected_variants:
        variant_info = dict((scan.get("variants", {}) or {}).get(str(variant), {}) or {})
        for raw in variant_info.get("images", []) or []:
            image = DatasetImage(**raw)
            if selected_splits and image.split not in selected_splits:
                continue
            images.append(image)
    if not images:
        raise FileNotFoundError("No images match the selected dataset types and splits.")

    population_images = len(images)
    requested = max(1, int(max_images or 1))
    if len(images) > requested:
        images = random.Random(int(seed)).sample(images, requested)
    images.sort(key=lambda item: (item.variant, item.split, item.relative_path))

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_square_sum = torch.zeros(3, dtype=torch.float64)
    pixels_per_channel = 0
    minimum = math.inf
    maximum = -math.inf
    maximum_channel_delta = 0.0
    png_fallback_count = 0
    failures: list[str] = []
    sampled_variants: dict[str, int] = defaultdict(int)
    sampled_splits: dict[str, int] = defaultdict(int)
    processed = 0

    for image in images:
        try:
            tensor, source_kind, _warning = _load_dataset_image(
                image,
                prefer_float=bool(prefer_float32_sources),
            )
            if source_kind == "png":
                png_fallback_count += 1
            height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
            channel_sum += tensor.sum(dim=(1, 2), dtype=torch.float64)
            channel_square_sum += tensor.square().sum(dim=(1, 2), dtype=torch.float64)
            pixels_per_channel += height * width
            minimum = min(minimum, float(tensor.min()))
            maximum = max(maximum, float(tensor.max()))
            maximum_channel_delta = max(
                maximum_channel_delta,
                float((tensor[0] - tensor[1]).abs().max()),
                float((tensor[0] - tensor[2]).abs().max()),
                float((tensor[1] - tensor[2]).abs().max()),
            )
            sampled_variants[image.variant] += 1
            sampled_splits[image.split] += 1
            processed += 1
        except Exception as exc:
            failures.append(f"{image.relative_path}: {exc}")

    if not processed or not pixels_per_channel:
        detail = f" First failure: {failures[0]}" if failures else ""
        raise RuntimeError(f"Could not read any images for statistics.{detail}")

    mean = channel_sum / pixels_per_channel
    variance = (channel_square_sum / pixels_per_channel - mean.square()).clamp_min(0)
    std = variance.sqrt()
    grayscale_replicated = maximum_channel_delta <= 1e-7
    if grayscale_replicated:
        scalar_mean = float(mean.mean())
        scalar_std = float(std.mean())
        recommended_mean = [scalar_mean] * 3
        recommended_std = [scalar_std] * 3
    else:
        recommended_mean = [float(value) for value in mean]
        recommended_std = [float(value) for value in std]
    if any(value <= 1e-12 for value in recommended_std):
        raise ValueError("Estimated standard deviation is zero; the sample has no usable contrast.")

    return {
        "dataset_root": str(scan.get("dataset_root", resolve_dataset_root(path))),
        "population_images": population_images,
        "requested_sample_images": requested,
        "sampled_images": processed,
        "failed_images": len(failures),
        "png_fallback_count": png_fallback_count,
        "pixels_per_channel": pixels_per_channel,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "recommended_mean": recommended_mean,
        "recommended_std": recommended_std,
        "minimum": float(minimum),
        "maximum": float(maximum),
        "grayscale_replicated": grayscale_replicated,
        "maximum_channel_delta": maximum_channel_delta,
        "sampled_variants": dict(sorted(sampled_variants.items())),
        "sampled_splits": dict(sorted(sampled_splits.items())),
        "failures": failures,
        "seed": int(seed),
        "method": "pixel_weighted_population_moments_from_deterministic_sample",
    }


def feature_shape_summary(
    model_id: str,
    *,
    input_width: int,
    input_height: int,
    resize_mode: str,
    outputs: Iterable[str],
    batch_size: int = 1,
) -> dict[str, Any]:
    """Describe model input and saved tensor shapes without loading weights."""

    info = dict(DINO_V3_MODELS.get(str(model_id), DINO_V3_MODELS[next(iter(DINO_V3_MODELS))]))
    patch = int(info["patch_size"])
    hidden = int(info["hidden_size"])
    batch = max(1, int(batch_size))
    selected = set(outputs or [])
    if str(resize_mode) == "none":
        return {
            **info,
            "input": "original H x W (each side is cropped down to a multiple of 16)",
            "patch_grid": "variable H/16 x W/16",
            "saved_shapes": "variable by source image",
        }
    width = max(patch, int(input_width))
    height = max(patch, int(input_height))
    patch_w = width // patch
    patch_h = height // patch
    shapes: dict[str, list[int]] = {}
    if "cls_token" in selected:
        shapes["cls_token"] = [hidden]
    if "patch_tokens" in selected:
        shapes["patch_tokens"] = [hidden, patch_h, patch_w]
    if "mean_patch_token" in selected:
        shapes["mean_patch_token"] = [hidden]
    if "register_tokens" in selected:
        shapes["register_tokens"] = [int(info["register_tokens"]), hidden]
    return {
        **info,
        "input": f"batch {batch} x 3 x {height} x {width}",
        "patch_grid": f"{patch_h} x {patch_w} ({patch_h * patch_w:,} patches)",
        "token_sequence": 1 + int(info["register_tokens"]) + patch_h * patch_w,
        "saved_shapes": shapes,
    }


def feature_output_folder(config: Mapping[str, Any]) -> Path:
    paths = dict(config.get("paths", {}) or {})
    if paths.get("output_root"):
        return Path(str(paths["output_root"])).expanduser().resolve(strict=False)
    dataset_root = resolve_dataset_root(str(paths.get("dataset_root", ".")))
    model_cfg = dict(config.get("model", {}) or {})
    input_cfg = dict(config.get("input", {}) or {})
    extraction_cfg = dict(config.get("extraction", {}) or {})
    model_name = str(model_cfg.get("model_id", "dinov3")).split("/")[-1]
    resize_mode = str(input_cfg.get("resize_mode", "exact"))
    if resize_mode == "none":
        size_name = "native"
    else:
        size_name = f"{int(input_cfg.get('height', 1024))}x{int(input_cfg.get('width', 1024))}"
    output_names = "-".join(
        sorted(str(v) for v in extraction_cfg.get("outputs", ["patch_tokens", "cls_token"]))
    )
    layer = int(extraction_cfg.get("layer", -1))
    identity = {
        "network": config.get("network", "dinov3"),
        "model_id": model_cfg.get("model_id"),
        "model_path": model_cfg.get("model_path"),
        "compute_dtype": model_cfg.get(
            "compute_dtype", DEFAULT_DINO_V3_COMPUTE_DTYPE
        ),
        "input": input_cfg,
        "layer": layer,
        "outputs": sorted(extraction_cfg.get("outputs", [])),
        "save_dtype": extraction_cfg.get("save_dtype", "float32"),
    }
    digest = hashlib.sha256(
        json.dumps(_json_safe(identity), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    folder = _slug(
        f"{model_name}__{resize_mode}-{size_name}__layer-{layer}__{output_names}__cfg-{digest}"
    )
    return dataset_root / "features" / folder


def extract_features_from_config(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    model_loader: Callable[[Mapping[str, Any], torch.device], Any] | None = None,
) -> dict[str, Any]:
    """Extract frozen DINOv3 features from an already exported dataset."""

    callback = progress_callback or (lambda _event: None)
    cfg = dict(config or {})
    paths = dict(cfg.get("paths", {}) or {})
    dataset_root = resolve_dataset_root(str(paths.get("dataset_root", "")))
    scan = scan_dataset_image_variants(dataset_root)
    selected_variants = [str(value) for value in cfg.get("variants", [])]
    selected_splits = {str(value) for value in cfg.get("splits", []) if str(value) != "all"}
    if not selected_variants:
        raise ValueError("Select at least one existing dataset image type.")

    images: list[DatasetImage] = []
    for variant in selected_variants:
        variant_info = dict((scan.get("variants", {}) or {}).get(variant, {}) or {})
        for raw in variant_info.get("images", []) or []:
            image = DatasetImage(**raw)
            if selected_splits and image.split not in selected_splits:
                continue
            images.append(image)
    if not images:
        raise FileNotFoundError("No images match the selected dataset types and splits.")

    output_root = feature_output_folder(cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    model_cfg = dict(cfg.get("model", {}) or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    extraction_cfg = dict(cfg.get("extraction", {}) or {})
    device = _resolve_device(str(model_cfg.get("device", "auto")))
    callback({"event": "stage_start", "stage": "load_dinov3", "total": len(images)})
    loader = model_loader or _load_dinov3_model
    model = loader(model_cfg, device)
    model.eval()
    callback({"event": "stage_finish", "stage": "load_dinov3", "total": len(images)})

    batch_size = max(1, int(extraction_cfg.get("batch_size", 1) or 1))
    overwrite = bool(extraction_cfg.get("overwrite", False))
    prefer_float = bool(extraction_cfg.get("prefer_float32_sources", True))
    output_dtype = _torch_dtype(str(extraction_cfg.get("save_dtype", "float32")))
    outputs_to_save = set(extraction_cfg.get("outputs", ["patch_tokens", "cls_token"]) or [])
    if not outputs_to_save:
        raise ValueError("Select at least one feature tensor to save.")
    layer = int(extraction_cfg.get("layer", -1))
    layer_count = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    if layer_count and not (-(layer_count + 1) <= layer <= layer_count):
        raise ValueError(
            f"Requested layer {layer} is outside this model's hidden-state range "
            f"[-{layer_count + 1}, {layer_count}]."
        )

    manifest_path = output_root / "features_manifest.jsonl"
    manifest_index = _read_feature_manifest(manifest_path)
    warnings: list[str] = []
    fallback_png_count = 0
    completed = 0
    skipped = 0
    failed = 0
    newly_saved = 0
    callback(
        {"event": "stage_start", "stage": "extract_features", "processed": 0, "total": len(images)}
    )

    for chunk in _chunks(images, batch_size):
        prepared: list[tuple[DatasetImage, torch.Tensor, str, str | None]] = []
        for image in chunk:
            destination = _feature_path(output_root, image)
            if destination.exists() and not overwrite:
                skipped += 1
                completed += 1
                callback(
                    {
                        "event": "image_progress",
                        "stage": "extract_features",
                        "processed": completed,
                        "total": len(images),
                        "skipped": skipped,
                    }
                )
                continue
            try:
                tensor, source_kind, warning = _load_dataset_image(image, prefer_float=prefer_float)
                if source_kind == "png":
                    fallback_png_count += 1
                if warning:
                    warnings.append(warning)
                tensor = _prepare_input(tensor, input_cfg)
                prepared.append((image, tensor, source_kind, warning))
            except Exception as exc:
                failed += 1
                completed += 1
                warnings.append(f"{image.relative_path}: {exc}")
                callback(
                    {
                        "event": "image_progress",
                        "stage": "extract_features",
                        "processed": completed,
                        "total": len(images),
                        "failed": failed,
                    }
                )
        if not prepared:
            continue
        # Native-size extraction may produce mixed shapes. Keep batching for
        # equal shapes and fall back to smaller groups without resizing pixels.
        shape_groups: dict[
            tuple[int, ...], list[tuple[DatasetImage, torch.Tensor, str, str | None]]
        ] = defaultdict(list)
        for item in prepared:
            shape_groups[tuple(item[1].shape)].append(item)
        for group in shape_groups.values():
            batch = torch.stack([item[1] for item in group], dim=0).to(device)
            try:
                feature_batches = _run_dinov3(model, batch, cfg, outputs_to_save)
                for index, (image, input_tensor, source_kind, warning) in enumerate(group):
                    destination = _feature_path(output_root, image)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    tensors = {
                        key: value[index].detach().cpu().to(output_dtype).contiguous()
                        for key, value in feature_batches.items()
                    }
                    source_path = image.float32_path if source_kind == "float32" else image.png_path
                    payload = {
                        "schema_version": 1,
                        "features": tensors,
                        "source": {
                            "dataset_root": str(dataset_root),
                            "variant": image.variant,
                            "split": image.split,
                            "relative_image_path": image.relative_path,
                            "png_path": image.png_path,
                            "float32_path": image.float32_path,
                            "loaded_path": source_path,
                            "loaded_format": source_kind,
                        },
                        "model": {
                            "network": "dinov3",
                            "model_id": str(model_cfg.get("model_id", "")),
                            "compute_dtype": str(
                                model_cfg.get(
                                    "compute_dtype", DEFAULT_DINO_V3_COMPUTE_DTYPE
                                )
                            ),
                            "layer": int(extraction_cfg.get("layer", -1)),
                        },
                        "input": {
                            "shape_chw": list(input_tensor.shape),
                            **input_cfg,
                        },
                    }
                    torch.save(payload, destination)
                    row = {
                        "feature_path": destination.relative_to(output_root).as_posix(),
                        "source_variant": image.variant,
                        "source_split": image.split,
                        "source_relative_image": image.relative_path,
                        "source_png_path": image.png_path,
                        "source_float32_path": image.float32_path,
                        "loaded_source_path": source_path,
                        "loaded_source_format": source_kind,
                        "input_shape_chw": list(input_tensor.shape),
                        "feature_shapes": {
                            key: list(value[index].shape) for key, value in feature_batches.items()
                        },
                        "warning": warning,
                    }
                    manifest_index[str(row["feature_path"])] = row
                    newly_saved += 1
                    completed += 1
                    callback(
                        {
                            "event": "image_progress",
                            "stage": "extract_features",
                            "processed": completed,
                            "total": len(images),
                            "fallback_png_count": fallback_png_count,
                            "failed": failed,
                        }
                    )
            except Exception as exc:
                for image, _tensor, _source_kind, _warning in group:
                    failed += 1
                    completed += 1
                    warnings.append(f"{image.relative_path}: DINOv3 inference failed: {exc}")
                    callback(
                        {
                            "event": "image_progress",
                            "stage": "extract_features",
                            "processed": completed,
                            "total": len(images),
                            "failed": failed,
                        }
                    )

    manifest_rows = [manifest_index[key] for key in sorted(manifest_index)]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    resolved_config_path = output_root / "extraction_config_resolved.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(_json_safe(cfg), sort_keys=False), encoding="utf-8"
    )
    summary = {
        "status": "completed" if failed == 0 else "completed_with_errors",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "requested_images": len(images),
        "saved_features": newly_saved,
        "indexed_features": len(manifest_rows),
        "skipped_existing": skipped,
        "failed": failed,
        "png_fallback_count": fallback_png_count,
        "warnings": warnings,
        "config": _json_safe(cfg),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_feature_readme(output_root, summary)
    (output_root / "FEATURES_DONE.txt").write_text(
        f"Feature extraction finished.\nSaved this run: {newly_saved}\nIndexed total: {len(manifest_rows)}\nFailed: {failed}\nPNG fallbacks: {fallback_png_count}\n",
        encoding="utf-8",
    )
    callback(
        {
            "event": "stage_finish",
            "stage": "extract_features",
            "processed": completed,
            "total": len(images),
            "failed": failed,
            "fallback_png_count": fallback_png_count,
        }
    )
    return summary


def _load_dinov3_model(model_cfg: Mapping[str, Any], device: torch.device) -> Any:
    _ensure_transformers_pytree_compatibility()
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - depends on optional runtime stack
        raise RuntimeError(
            "DINOv3 extraction requires transformers>=4.56. Install the project dependencies, then restart the app."
        ) from exc
    model_id = str(model_cfg.get("model_path") or model_cfg.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("A DINOv3 Hugging Face model id or local model directory is required.")
    try:
        model = AutoModel.from_pretrained(
            model_id,
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
    except Exception as exc:
        if _is_gated_huggingface_error(exc):
            raise RuntimeError(_gated_huggingface_error_message(model_id)) from exc
        raise
    compute_dtype = _torch_dtype(
        str(model_cfg.get("compute_dtype", DEFAULT_DINO_V3_COMPUTE_DTYPE))
    )
    return model.to(device=device, dtype=compute_dtype)


def _is_gated_huggingface_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    markers = (
        "gated repo",
        "gatedrepoerror",
        "cannot access gated",
        "access to model",
        "access to this resource is restricted",
        "you must have access to it and be authenticated",
    )
    return any(marker in text for marker in markers)


def _gated_huggingface_error_message(model_id: str) -> str:
    model_url = f"https://huggingface.co/{model_id}"
    return (
        f"DINOv3 checkpoint access is gated for {model_id}. "
        f"First accept Meta's model license at {model_url}. Then authenticate the same "
        "environment that launches this app with `hf auth login` (or set `HF_TOKEN` before "
        "launching it), verify with `hf auth whoami`, and restart the app. A token alone is not "
        "enough until the Hugging Face account has been granted access to the model."
    )


def _ensure_transformers_pytree_compatibility() -> None:
    """Expose PyTorch 2.2's public pytree name when running the app on 2.1.

    The project's MMDetection environment currently pins PyTorch 2.1, whose
    equivalent registration function is still private. Transformers 4.56 uses
    the public spelling while otherwise supporting this runtime.
    """

    pytree = getattr(torch.utils, "_pytree", None)
    if pytree is None or hasattr(pytree, "register_pytree_node"):
        return
    private_register = getattr(pytree, "_register_pytree_node", None)
    if private_register is None:
        return

    def register_pytree_node(
        typ: Any,
        flatten_fn: Callable[..., Any],
        unflatten_fn: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        supported = {
            key: value
            for key, value in kwargs.items()
            if key in {"to_dumpable_context", "from_dumpable_context"}
        }
        return private_register(typ, flatten_fn, unflatten_fn, **supported)

    pytree.register_pytree_node = register_pytree_node


def _run_dinov3(
    model: Any,
    batch: torch.Tensor,
    config: Mapping[str, Any],
    outputs_to_save: set[str],
) -> dict[str, torch.Tensor]:
    model_cfg = dict(config.get("model", {}) or {})
    extraction_cfg = dict(config.get("extraction", {}) or {})
    input_cfg = dict(config.get("input", {}) or {})
    mean = torch.tensor(
        _triplet(input_cfg.get("mean"), DINO_V3_LVD_MEAN), device=batch.device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        _triplet(input_cfg.get("std"), DINO_V3_LVD_STD), device=batch.device
    ).view(1, 3, 1, 1)
    normalized = (batch.to(torch.float32) - mean) / std
    layer = int(extraction_cfg.get("layer", -1))
    compute_dtype_name = str(
        model_cfg.get("compute_dtype", DEFAULT_DINO_V3_COMPUTE_DTYPE)
    )
    compute_dtype = _torch_dtype(compute_dtype_name)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if batch.device.type == "cuda" and compute_dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        result = model(
            pixel_values=normalized,
            output_hidden_states=layer != -1,
            return_dict=True,
        )
    tokens = result.last_hidden_state if layer == -1 else result.hidden_states[layer]
    patch_size = int(getattr(model.config, "patch_size", 16))
    register_count = int(getattr(model.config, "num_register_tokens", 4))
    patch_h = int(normalized.shape[-2]) // patch_size
    patch_w = int(normalized.shape[-1]) // patch_size
    patch_count = patch_h * patch_w
    patch_start = 1 + register_count
    if int(tokens.shape[1]) < patch_start + patch_count:
        raise RuntimeError(
            f"Model returned {tokens.shape[1]} tokens; expected at least {patch_start + patch_count}."
        )
    patch_tokens = tokens[:, patch_start : patch_start + patch_count]
    out: dict[str, torch.Tensor] = {}
    if "cls_token" in outputs_to_save:
        out["cls_token"] = tokens[:, 0]
    if "register_tokens" in outputs_to_save:
        out["register_tokens"] = tokens[:, 1:patch_start]
    if "mean_patch_token" in outputs_to_save:
        out["mean_patch_token"] = patch_tokens.mean(dim=1)
    if "patch_tokens" in outputs_to_save:
        out["patch_tokens"] = patch_tokens.reshape(
            tokens.shape[0], patch_h, patch_w, tokens.shape[-1]
        ).permute(0, 3, 1, 2)
    return out


def _load_dataset_image(
    image: DatasetImage, *, prefer_float: bool
) -> tuple[torch.Tensor, str, str | None]:
    if prefer_float and image.has_float32:
        value = _safe_torch_load(Path(image.float32_path))
        if isinstance(value, Mapping):
            value = value.get("image", value.get("tensor"))
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Float32 source is not a tensor: {image.float32_path}")
        tensor = value.detach().cpu().to(torch.float32)
        return _coerce_chw01(tensor), "float32", None
    if not image.has_png:
        raise FileNotFoundError(
            f"Neither a float32 tensor nor PNG exists for {image.relative_path}."
        )
    warning = None
    if prefer_float:
        warning = (
            f"Float32 source missing for {image.relative_path}; read the PNG fallback, "
            "which has already been quantized to 8-bit."
        )
    with Image.open(image.png_path) as pil:
        array = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous(), "png", warning


def _prepare_input(tensor: torch.Tensor, input_cfg: Mapping[str, Any]) -> torch.Tensor:
    tensor = _coerce_chw01(tensor)
    patch_size = 16
    mode = str(input_cfg.get("resize_mode", "exact") or "exact")
    if mode == "none":
        height = max(patch_size, (int(tensor.shape[-2]) // patch_size) * patch_size)
        width = max(patch_size, (int(tensor.shape[-1]) // patch_size) * patch_size)
        return tensor[:, :height, :width].contiguous()
    if mode not in {"exact", "fit_pad"}:
        raise ValueError("resize_mode must be exact, fit_pad, or none.")
    target_h = max(patch_size, int(input_cfg.get("height", 1024) or 1024))
    target_w = max(patch_size, int(input_cfg.get("width", 1024) or 1024))
    if target_h % patch_size or target_w % patch_size:
        raise ValueError("DINOv3 input width and height must be divisible by patch size 16.")
    if tuple(tensor.shape[-2:]) == (target_h, target_w):
        # Preserve an already correctly sized float32 source exactly instead
        # of routing it through a redundant interpolation operation.
        return tensor.contiguous()
    if mode == "exact":
        return F.interpolate(
            tensor[None],
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].clamp(0, 1)
    source_h, source_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = min(target_h / max(1, source_h), target_w / max(1, source_w))
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    resized = F.interpolate(
        tensor[None],
        size=(resized_h, resized_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0].clamp(0, 1)
    pad_value = float(input_cfg.get("pad_value", 0.0) or 0.0)
    canvas = torch.full((3, target_h, target_w), pad_value, dtype=torch.float32)
    canvas[:, :resized_h, :resized_w] = resized
    return canvas


def _coerce_chw01(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().cpu().to(torch.float32).squeeze()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4} and tensor.shape[0] not in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3 or tensor.shape[0] not in {1, 3, 4}:
        raise ValueError(f"Expected CHW/HWC image tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 4:
        tensor = tensor[:3]
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
        raise ValueError("Float32 source values must be normalized to the closed interval [0, 1].")
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).contiguous()


def _feature_path(output_root: Path, image: DatasetImage) -> Path:
    return output_root / image.variant / Path(image.relative_path).with_suffix(".pt")


def _image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in extensions
    )


def _read_feature_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            feature_path = str(row.get("feature_path", ""))
            if feature_path:
                rows[feature_path] = row
    except Exception:
        # A damaged/interrupted manifest must not prevent rebuilding it from a
        # subsequent overwrite run.
        return {}
    return rows


def _is_default_research_dataset(dataset_root: Path) -> bool:
    manifest = dataset_root / "manifest.json"
    try:
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            key = ((payload.get("config_snapshot") or {}).get("study_preset_provenance") or {}).get(
                "preset_key"
            )
            if str(key) == "simple_crop_pipeline_v1":
                return True
    except Exception:
        pass
    resolved = dataset_root / "metadata" / "export_config_resolved.yaml"
    try:
        if resolved.exists():
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            key = (payload.get("study_preset_provenance") or {}).get("preset_key")
            return str(key) == "simple_crop_pipeline_v1"
    except Exception:
        pass
    return "default-research-dataset" in dataset_root.name.casefold()


def _write_feature_readme(output_root: Path, summary: Mapping[str, Any]) -> Path:
    cfg = dict(summary.get("config", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    extraction_cfg = dict(cfg.get("extraction", {}) or {})
    text = f"""# DINOv3 feature dataset

Features were extracted from `{summary.get("dataset_root")}` with `{model_cfg.get("model_id")}`. The backbone is frozen and pretrained; no fine-tuning occurs here.

## Layout and source mapping

Each feature keeps the source image's split and stem:

```text
<this folder>/<variant>/<split>/<source image stem>.pt
```

`features_manifest.jsonl` is the authoritative index. Every row records the feature path, original PNG path, float32 path, the source format actually read, input shape, and saved feature shapes. Each `.pt` payload repeats that mapping under `payload["source"]`.

Float32 `.pt` images normalized to `[0, 1]` are preferred. When one is absent, extraction emits a warning and reads the corresponding 8-bit PNG.

## Reading one feature

```python
from pathlib import Path
import torch

path = next(Path(r"{output_root}").glob("*/*/*.pt"))
try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:  # PyTorch < 2.6
    payload = torch.load(path, map_location="cpu")

patch_map = payload["features"].get("patch_tokens")  # [channels, patch_h, patch_w]
cls_token = payload["features"].get("cls_token")     # [channels]
source_png = payload["source"]["png_path"]
source_float32 = payload["source"]["float32_path"]
```

Saved tensors: `{sorted(extraction_cfg.get("outputs", []))}`. Saved dtype: `{extraction_cfg.get("save_dtype", "float32")}`. Exact parameters are in `extraction_config_resolved.yaml`; counts and fallback warnings are in `summary.json`.
"""
    path = output_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "auto").casefold()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but PyTorch cannot access a CUDA device.")
    return device


def _torch_dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(value).casefold(), torch.float32)


def _triplet(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    try:
        parsed = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return default
    return parsed if len(parsed) == 3 and all(math.isfinite(part) for part in parsed) else default


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _chunks(values: list[DatasetImage], size: int) -> Iterable[list[DatasetImage]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-_")[:180]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
