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

# -----------------------------------------------------------------------------
# Contralateral source alignment helpers
# -----------------------------------------------------------------------------

DEFAULT_CONTRALATERAL_ALIGNMENT_OPTIONS: dict[str, Any] = {
    "enabled": True,
    "method": "nipple_y",
    "threshold": None,
    "tip_side": "auto",
    "tip_tolerance_fraction": 0.006,
    "tip_tolerance_px": None,
    "smooth_rows": 31,
    "max_shift_fraction": 0.20,
    "pad_value": 0.0,
}


def make_contralateral_alignment_options(options: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_CONTRALATERAL_ALIGNMENT_OPTIONS)
    if options:
        out.update(options)
    return out


def align_contralateral_image_to_reference(
    reference_image: torch.Tensor | np.ndarray,
    moving_image: torch.Tensor | np.ndarray,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Vertically align an opposite-breast source image to a reference image.

    The intended use is the custom RGB source ``contralateral_same_view_crop``:
    both images are first fixed-preprocessed, the opposite breast is shifted in
    the full-image coordinate system, and only then is the same square-crop
    window extracted. Positive ``shift_y`` means the moving/opposite image was
    shifted downward.
    """
    opts = make_contralateral_alignment_options(options)
    moving_tensor = _as_1hw_tensor(moving_image)
    info: dict[str, Any] = {
        "contralateral_alignment_enabled": bool(opts.get("enabled", True)),
        "contralateral_alignment_method": str(opts.get("method", "nipple_y")),
        "contralateral_alignment_shift_y": 0,
        "contralateral_alignment_applied": False,
        "contralateral_alignment_failure_reason": None,
    }

    if not bool(opts.get("enabled", True)):
        info["contralateral_alignment_failure_reason"] = "alignment_disabled"
        return moving_tensor, info

    method = str(opts.get("method", "nipple_y") or "nipple_y").strip().casefold()
    if method in {"none", "disabled", "off"}:
        info["contralateral_alignment_failure_reason"] = "alignment_method_none"
        return moving_tensor, info

    if method in {"projection_y", "projection", "intensity_projection", "projection_intensity"}:
        # Placeholder requested by the user. Keep it explicit so configs can be
        # switched to this method later without silently doing the wrong thing.
        info["contralateral_alignment_failure_reason"] = "projection_intensity_alignment_placeholder_not_implemented"
        return moving_tensor, info

    if method not in {"nipple_y", "nipple", "foreground_tip", "foreground_tip_y"}:
        info["contralateral_alignment_failure_reason"] = f"unknown_alignment_method:{method}"
        return moving_tensor, info

    reference_arr = _as_float2d(reference_image)
    moving_arr = _as_float2d(moving_tensor)
    ref_tip = detect_breast_nipple_tip_from_foreground(reference_arr, options=opts)
    mov_tip = detect_breast_nipple_tip_from_foreground(moving_arr, options=opts)
    info["reference_nipple_tip"] = ref_tip
    info["moving_nipple_tip"] = mov_tip

    if ref_tip.get("found") is not True or mov_tip.get("found") is not True:
        info["contralateral_alignment_failure_reason"] = "nipple_tip_not_found"
        return moving_tensor, info

    shift_y = int(round(float(ref_tip["tip_y"]) - float(mov_tip["tip_y"])))
    max_shift = int(round(float(opts.get("max_shift_fraction", 0.20)) * float(max(moving_arr.shape[0], 1))))
    if max_shift >= 0:
        shift_y = int(np.clip(shift_y, -max_shift, max_shift))
    shifted = _shift_tensor_y(moving_tensor, shift_y, pad_value=float(opts.get("pad_value", 0.0)))
    info.update({
        "contralateral_alignment_shift_y": int(shift_y),
        "contralateral_alignment_applied": bool(shift_y != 0),
        "contralateral_alignment_max_shift_px": int(max_shift),
    })
    return shifted, info


def detect_breast_nipple_tip_from_foreground(
    image: torch.Tensor | np.ndarray,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate nipple-tip position from the breast foreground mask.

    After right-to-left mirroring, the breast foreground is normally on the left
    and the nipple is near the rightmost foreground boundary. With ``tip_side``
    set to ``auto``, the function chooses the right boundary when the breast
    centroid is left of image center, and the left boundary otherwise.
    """
    opts = make_contralateral_alignment_options(options)
    arr = _as_float2d(image)
    height, width = arr.shape
    mask = _breast_mask(arr, threshold=opts.get("threshold"))
    mask = _largest_component_mask(mask)
    if not mask.any():
        return {"found": False, "failure_reason": "empty_foreground_mask", "image_shape": (int(height), int(width))}

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"found": False, "failure_reason": "empty_foreground_coordinates", "image_shape": (int(height), int(width))}

    requested_side = str(opts.get("tip_side", "auto") or "auto").strip().casefold()
    if requested_side in {"auto", ""}:
        side = "right" if float(xs.mean()) < (float(width) / 2.0) else "left"
    elif requested_side in {"right", "r", "right_after_mirror"}:
        side = "right"
    elif requested_side in {"left", "l"}:
        side = "left"
    else:
        side = "right"

    rows = np.unique(ys).astype(np.int32)
    if rows.size == 0:
        return {"found": False, "failure_reason": "no_foreground_rows", "image_shape": (int(height), int(width))}

    profile = np.full((height,), np.nan, dtype=np.float32)
    if side == "right":
        for y in rows:
            row_xs = xs[ys == y]
            if row_xs.size:
                profile[int(y)] = float(row_xs.max())
    else:
        for y in rows:
            row_xs = xs[ys == y]
            if row_xs.size:
                profile[int(y)] = float(row_xs.min())

    smooth_rows = int(opts.get("smooth_rows", 31) or 31)
    if smooth_rows > 1 and np.isfinite(profile).sum() >= 3:
        smooth_rows = max(3, smooth_rows | 1)
        half = smooth_rows // 2
        valid = np.isfinite(profile).astype(np.float32)
        filled = np.nan_to_num(profile, nan=0.0).astype(np.float32)
        kernel = np.ones((smooth_rows,), dtype=np.float32)
        numerator = np.convolve(filled, kernel, mode="same")
        denominator = np.convolve(valid, kernel, mode="same")
        smoothed = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-6), np.nan)
        # Do not let the convolution invent values outside observed rows.
        smoothed[: max(0, int(rows.min()) - half)] = np.nan
        smoothed[min(height, int(rows.max()) + half + 1) :] = np.nan
    else:
        smoothed = profile

    valid_idx = np.where(np.isfinite(smoothed))[0]
    if valid_idx.size == 0:
        return {"found": False, "failure_reason": "empty_boundary_profile", "image_shape": (int(height), int(width))}

    tol_px_opt = opts.get("tip_tolerance_px", None)
    if tol_px_opt is None:
        tol_px = max(2.0, float(width) * float(opts.get("tip_tolerance_fraction", 0.006)))
    else:
        tol_px = max(0.0, float(tol_px_opt))

    valid_values = smoothed[valid_idx]
    if side == "right":
        peak_x = float(np.nanmax(valid_values))
        candidate_rows = valid_idx[valid_values >= peak_x - tol_px]
    else:
        peak_x = float(np.nanmin(valid_values))
        candidate_rows = valid_idx[valid_values <= peak_x + tol_px]

    if candidate_rows.size == 0:
        if side == "right":
            best_row = int(valid_idx[int(np.nanargmax(valid_values))])
        else:
            best_row = int(valid_idx[int(np.nanargmin(valid_values))])
        candidate_rows = np.array([best_row], dtype=np.int32)

    tip_y = float(np.median(candidate_rows))
    return {
        "found": True,
        "tip_x": float(peak_x),
        "tip_y": float(tip_y),
        "tip_side": side,
        "candidate_rows": int(candidate_rows.size),
        "tip_tolerance_px": float(tol_px),
        "foreground_pixels": int(mask.sum()),
        "image_shape": (int(height), int(width)),
    }


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or cv2 is None:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    largest = int(np.argmax(areas) + 1)
    return labels == largest


def _as_float2d(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image or [1,H,W] image, got shape {tuple(arr.shape)}")
    return np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _as_1hw_tensor(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Expected moving image shaped [1,H,W] or [H,W], got {tuple(tensor.shape)}")
        return tensor.contiguous()
    arr = _as_float2d(image)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _shift_tensor_y(image: torch.Tensor, shift_y: int, *, pad_value: float = 0.0) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected image shaped [1,H,W], got {tuple(image.shape)}")
    shift_y = int(shift_y)
    if shift_y == 0:
        return image.clone().contiguous()
    _, height, _ = image.shape
    out = torch.full_like(image, float(pad_value))
    if abs(shift_y) >= int(height):
        return out.contiguous()
    if shift_y > 0:
        out[:, shift_y:, :] = image[:, : height - shift_y, :]
    else:
        sy = -shift_y
        out[:, : height - sy, :] = image[:, sy:, :]
    return out.contiguous()

