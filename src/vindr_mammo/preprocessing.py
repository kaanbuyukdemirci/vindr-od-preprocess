from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


DEFAULT_PREPROCESS_OPTIONS: dict[str, Any] = {
    "invert_to_black_background": False,  # handled in dicom_io through PhotometricInterpretation
    "crop_breast": False,
    "mirror_right_to_left": False,
    "crop_padding": 10,
    "crop_threshold": None,
    "min_component_area_fraction": 0.001,
}


@dataclass(frozen=True)
class PreprocessGeometryResult:
    image: torch.Tensor
    boxes: torch.Tensor
    mass_boxes: torch.Tensor
    box_keep: torch.Tensor
    mass_box_keep: torch.Tensor
    info: dict[str, Any]


def make_preprocess_options(options: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_PREPROCESS_OPTIONS)
    if options:
        out.update(options)
    return out


def apply_geometry_preprocessing(
    image: torch.Tensor,
    *,
    boxes: torch.Tensor | None = None,
    mass_boxes: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
) -> PreprocessGeometryResult:
    """Apply crop and horizontal mirror transforms and update boxes.

    Photometric inversion is intentionally not handled here, because it should be
    applied before normalization. See ``read_dicom_image(..., invert_monochrome1=True)``.
    """
    opts = make_preprocess_options(options)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected image tensor shaped [1, H, W], got {tuple(image.shape)}")

    boxes = _empty_boxes() if boxes is None else boxes.clone()
    mass_boxes = _empty_boxes() if mass_boxes is None else mass_boxes.clone()
    box_keep = torch.ones((boxes.shape[0],), dtype=torch.bool)
    mass_box_keep = torch.ones((mass_boxes.shape[0],), dtype=torch.bool)

    info: dict[str, Any] = {
        "crop_breast": bool(opts["crop_breast"]),
        "mirror_right_to_left": bool(opts["mirror_right_to_left"]),
        "crop_box_xyxy": None,
        "mirrored": False,
        "original_shape": (int(image.shape[-2]), int(image.shape[-1])),
    }

    if opts["crop_breast"]:
        crop_box = breast_crop_box(
            image,
            threshold=opts.get("crop_threshold"),
            padding=int(opts.get("crop_padding", 10)),
            min_component_area_fraction=float(opts.get("min_component_area_fraction", 0.001)),
        )
        x0, y0, x1, y1 = crop_box
        image = image[:, y0:y1, x0:x1].contiguous()
        boxes, box_keep = _crop_boxes(boxes, crop_box)
        mass_boxes, mass_box_keep = _crop_boxes(mass_boxes, crop_box)
        info["crop_box_xyxy"] = crop_box

    if opts["mirror_right_to_left"]:
        should_mirror = breast_is_on_right(image, threshold=opts.get("crop_threshold"))
        info["mirrored"] = bool(should_mirror)
        if should_mirror:
            width = int(image.shape[-1])
            image = torch.flip(image, dims=[-1]).contiguous()
            boxes = _mirror_boxes(boxes, width)
            mass_boxes = _mirror_boxes(mass_boxes, width)

    info["processed_shape"] = (int(image.shape[-2]), int(image.shape[-1]))
    return PreprocessGeometryResult(image, boxes, mass_boxes, box_keep, mass_box_keep, info)


def breast_crop_box(
    image: torch.Tensor,
    *,
    threshold: float | None = None,
    padding: int = 10,
    min_component_area_fraction: float = 0.001,
) -> tuple[int, int, int, int]:
    arr = _image_to_numpy(image)
    height, width = arr.shape
    mask = _breast_mask(arr, threshold=threshold)
    if not mask.any():
        return (0, 0, width, height)

    if cv2 is not None:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas) + 1)
            if areas[largest - 1] >= min_component_area_fraction * height * width:
                mask = labels == largest

    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, width, height)
    return (x0, y0, x1, y1)


def breast_is_on_right(image: torch.Tensor, *, threshold: float | None = None) -> bool:
    arr = _image_to_numpy(image)
    mask = _breast_mask(arr, threshold=threshold)
    if not mask.any():
        return False
    xs = np.where(mask)[1]
    return float(xs.mean()) > (arr.shape[1] / 2.0)


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _breast_mask(arr: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    if threshold is None:
        p_low, p_high = np.percentile(arr, [1.0, 99.0])
        threshold = max(float(p_low + 0.03 * (p_high - p_low)), float(p_low) + 1e-6)
    return arr > float(threshold)


def _empty_boxes() -> torch.Tensor:
    return torch.zeros((0, 4), dtype=torch.float32)


def _crop_boxes(boxes: torch.Tensor, crop_box: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes.clone(), torch.zeros((0,), dtype=torch.bool)
    x0, y0, x1, y1 = crop_box
    out = boxes.clone()
    out[:, [0, 2]] -= float(x0)
    out[:, [1, 3]] -= float(y0)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(x1 - x0))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(y1 - y0))
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[keep].contiguous(), keep.cpu()


def _mirror_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()
    out = boxes.clone()
    old_xmin = out[:, 0].clone()
    old_xmax = out[:, 2].clone()
    out[:, 0] = float(width) - old_xmax
    out[:, 2] = float(width) - old_xmin
    return out
