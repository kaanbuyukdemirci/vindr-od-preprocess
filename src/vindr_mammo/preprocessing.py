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

# -----------------------------------------------------------------------------
# Contralateral source alignment helpers
# -----------------------------------------------------------------------------

DEFAULT_CONTRALATERAL_ALIGNMENT_OPTIONS: dict[str, Any] = {
    "enabled": True,
    # Best default for the asymmetry channel. It first tries robust 1D profile
    # matching, then falls back to the interpretable nipple-y estimate.
    "method": "hybrid_profile_y",
    "fallback_method": "nipple_y",
    "threshold": None,
    "tip_side": "auto",
    "tip_tolerance_fraction": 0.006,
    "tip_tolerance_px": None,
    "smooth_rows": 31,
    "projection_smooth_rows": 51,
    "boundary_smooth_rows": 31,
    "max_shift_fraction": 0.20,
    "min_profile_overlap_fraction": 0.60,
    "min_profile_score": 0.05,
    "profile_score_margin": 0.03,
    "max_profile_nipple_disagreement_fraction": 0.05,
    "max_profile_nipple_disagreement_px": None,
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

    Available methods:

    ``hybrid_profile_y``
        Default. Computes row-projection, boundary-profile, nipple-y, and
        centroid-y candidate shifts. It chooses the strongest profile match and
        falls back to nipple-y/centroid-y if the profile score is weak.
    ``row_projection_y``
        Aligns the full foreground distribution along image rows.
    ``boundary_profile_y``
        Aligns the breast outer boundary profile along rows.
    ``mask_centroid_y``
        Aligns the vertical foreground centroid.
    ``nipple_y``
        Aligns the estimated nipple/tip row from the foreground boundary.
    ``projection_y`` / ``intensity_projection_y``
        Aligns the row-sum intensity profile. This is implemented, but it is
        less robust than foreground-mask profiles when vendors/windowing differ.
    """
    opts = make_contralateral_alignment_options(options)
    moving_tensor = _as_1hw_tensor(moving_image)
    info: dict[str, Any] = {
        "contralateral_alignment_enabled": bool(opts.get("enabled", True)),
        "contralateral_alignment_method": str(opts.get("method", "hybrid_profile_y")),
        "contralateral_alignment_selected_method": None,
        "contralateral_alignment_shift_y": 0,
        "contralateral_alignment_applied": False,
        "contralateral_alignment_failure_reason": None,
    }

    if not bool(opts.get("enabled", True)):
        info["contralateral_alignment_failure_reason"] = "alignment_disabled"
        return moving_tensor, info

    method = _normalize_alignment_method(opts.get("method", "hybrid_profile_y"))
    if method in {"none", "disabled", "off"}:
        info["contralateral_alignment_failure_reason"] = "alignment_method_none"
        return moving_tensor, info

    reference_arr = _as_float2d(reference_image)
    moving_arr = _as_float2d(moving_tensor)
    max_shift = int(round(float(opts.get("max_shift_fraction", 0.20)) * float(max(moving_arr.shape[0], 1))))
    max_shift = max(0, max_shift)

    if method in {"hybrid", "hybrid_profile", "hybrid_profile_y", "profile_hybrid", "profile_hybrid_y"}:
        selected = estimate_hybrid_profile_alignment_shift(reference_arr, moving_arr, options=opts)
    else:
        selected = estimate_alignment_shift_y(reference_arr, moving_arr, method=method, options=opts)

    info.update({
        "contralateral_alignment_candidates": selected.get("candidates", {}),
        "contralateral_alignment_selected_method": selected.get("selected_method"),
        "contralateral_alignment_selection_reason": selected.get("selection_reason"),
        "contralateral_alignment_warning": selected.get("warning"),
        "contralateral_alignment_max_shift_px": int(max_shift),
    })

    # Keep backward-compatible debug fields for the previous nipple-y GUI/output.
    if "nipple_y" in selected.get("candidates", {}):
        nipple = selected["candidates"]["nipple_y"]
        info["reference_nipple_tip"] = nipple.get("reference_nipple_tip")
        info["moving_nipple_tip"] = nipple.get("moving_nipple_tip")

    if selected.get("found") is not True:
        info["contralateral_alignment_failure_reason"] = selected.get("failure_reason", "alignment_shift_not_found")
        return moving_tensor, info

    shift_y = int(round(float(selected.get("shift_y", 0))))
    if max_shift >= 0:
        shift_y = int(np.clip(shift_y, -max_shift, max_shift))
    shifted = _shift_tensor_y(moving_tensor, shift_y, pad_value=float(opts.get("pad_value", 0.0)))
    info.update({
        "contralateral_alignment_shift_y": int(shift_y),
        "contralateral_alignment_applied": bool(shift_y != 0),
    })
    return shifted, info


def _normalize_alignment_method(method: Any) -> str:
    m = str(method or "hybrid_profile_y").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "nipple": "nipple_y",
        "foreground_tip": "nipple_y",
        "foreground_tip_y": "nipple_y",
        "centroid": "mask_centroid_y",
        "foreground_centroid": "mask_centroid_y",
        "foreground_centroid_y": "mask_centroid_y",
        "row_projection": "row_projection_y",
        "foreground_projection": "row_projection_y",
        "foreground_projection_y": "row_projection_y",
        "mask_projection": "row_projection_y",
        "mask_projection_y": "row_projection_y",
        "boundary_profile": "boundary_profile_y",
        "profile_boundary_y": "boundary_profile_y",
        "projection": "intensity_projection_y",
        "projection_y": "intensity_projection_y",
        "intensity_projection": "intensity_projection_y",
        "projection_intensity": "intensity_projection_y",
    }
    return aliases.get(m, m)


def estimate_hybrid_profile_alignment_shift(
    reference_image: torch.Tensor | np.ndarray,
    moving_image: torch.Tensor | np.ndarray,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate vertical shift with profile matching plus landmark fallbacks."""
    opts = make_contralateral_alignment_options(options)
    reference_arr = _as_float2d(reference_image)
    moving_arr = _as_float2d(moving_image)
    height = max(int(reference_arr.shape[0]), int(moving_arr.shape[0]), 1)

    candidates: dict[str, dict[str, Any]] = {}
    for method in ("row_projection_y", "boundary_profile_y", "nipple_y", "mask_centroid_y"):
        cand = estimate_alignment_shift_y(reference_arr, moving_arr, method=method, options=opts)
        candidates[method] = cand

    min_profile_score = float(opts.get("min_profile_score", 0.05))
    score_margin = float(opts.get("profile_score_margin", 0.03))
    row = candidates.get("row_projection_y", {})
    boundary = candidates.get("boundary_profile_y", {})
    profile_candidates = [c for c in (row, boundary) if c.get("found") is True]

    selected: dict[str, Any] | None = None
    reason = ""
    if profile_candidates:
        # Prefer row-projection when both profile scores are close, because it uses
        # the whole breast foreground distribution rather than only a boundary.
        row_score = float(row.get("score", -np.inf)) if row.get("found") is True else -np.inf
        boundary_score = float(boundary.get("score", -np.inf)) if boundary.get("found") is True else -np.inf
        if row.get("found") is True and row_score >= boundary_score - score_margin:
            selected = row
            reason = "row_projection_profile_selected"
        elif boundary.get("found") is True:
            selected = boundary
            reason = "boundary_profile_score_higher"
        else:
            selected = profile_candidates[0]
            reason = "first_valid_profile_selected"

        if float(selected.get("score", -np.inf)) < min_profile_score:
            selected = None
            reason = "profile_score_below_minimum"

    if selected is None:
        fallback_method = _normalize_alignment_method(opts.get("fallback_method", "nipple_y"))
        fallback = candidates.get(fallback_method)
        if fallback is None:
            fallback = estimate_alignment_shift_y(reference_arr, moving_arr, method=fallback_method, options=opts)
            candidates[fallback_method] = fallback
        if fallback.get("found") is True:
            selected = fallback
            reason = f"fallback_{fallback_method}"
        elif candidates.get("nipple_y", {}).get("found") is True:
            selected = candidates["nipple_y"]
            reason = "fallback_nipple_y"
        elif candidates.get("mask_centroid_y", {}).get("found") is True:
            selected = candidates["mask_centroid_y"]
            reason = "fallback_mask_centroid_y"

    if selected is None or selected.get("found") is not True:
        return {
            "found": False,
            "failure_reason": "no_valid_alignment_candidate",
            "selected_method": None,
            "selection_reason": reason or "all_candidates_failed",
            "candidates": candidates,
        }

    warning = None
    nipple = candidates.get("nipple_y", {})
    if nipple.get("found") is True and selected.get("method") in {"row_projection_y", "boundary_profile_y", "intensity_projection_y"}:
        disagreement = abs(float(selected.get("shift_y", 0.0)) - float(nipple.get("shift_y", 0.0)))
        max_disagreement_px_opt = opts.get("max_profile_nipple_disagreement_px", None)
        if max_disagreement_px_opt is None:
            max_disagreement_px = float(opts.get("max_profile_nipple_disagreement_fraction", 0.05)) * float(height)
        else:
            max_disagreement_px = float(max_disagreement_px_opt)
        if disagreement > max_disagreement_px:
            warning = (
                "profile_nipple_disagreement:" 
                f"profile_shift={selected.get('shift_y')}," 
                f"nipple_shift={nipple.get('shift_y')}," 
                f"difference={disagreement:.1f}px"
            )

    return {
        "found": True,
        "method": "hybrid_profile_y",
        "selected_method": selected.get("method"),
        "selection_reason": reason,
        "shift_y": int(round(float(selected.get("shift_y", 0)))) ,
        "score": selected.get("score"),
        "warning": warning,
        "candidates": candidates,
    }


def estimate_alignment_shift_y(
    reference_image: torch.Tensor | np.ndarray,
    moving_image: torch.Tensor | np.ndarray,
    *,
    method: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate a vertical shift for one specific alignment method."""
    opts = make_contralateral_alignment_options(options)
    method = _normalize_alignment_method(method)
    reference_arr = _as_float2d(reference_image)
    moving_arr = _as_float2d(moving_image)
    height = max(int(reference_arr.shape[0]), int(moving_arr.shape[0]), 1)
    max_shift = int(round(float(opts.get("max_shift_fraction", 0.20)) * float(height)))
    max_shift = max(0, max_shift)

    if method == "nipple_y":
        ref_tip = detect_breast_nipple_tip_from_foreground(reference_arr, options=opts)
        mov_tip = detect_breast_nipple_tip_from_foreground(moving_arr, options=opts)
        if ref_tip.get("found") is not True or mov_tip.get("found") is not True:
            return {
                "found": False,
                "method": "nipple_y",
                "failure_reason": "nipple_tip_not_found",
                "reference_nipple_tip": ref_tip,
                "moving_nipple_tip": mov_tip,
            }
        shift_y = int(round(float(ref_tip["tip_y"]) - float(mov_tip["tip_y"])))
        shift_y = int(np.clip(shift_y, -max_shift, max_shift))
        return {
            "found": True,
            "method": "nipple_y",
            "shift_y": int(shift_y),
            "score": None,
            "reference_nipple_tip": ref_tip,
            "moving_nipple_tip": mov_tip,
        }

    if method == "mask_centroid_y":
        ref = foreground_profile_y(reference_arr, options=opts, profile_kind="row_projection")
        mov = foreground_profile_y(moving_arr, options=opts, profile_kind="row_projection")
        if ref.get("found") is not True or mov.get("found") is not True:
            return {"found": False, "method": "mask_centroid_y", "failure_reason": "empty_foreground_profile"}
        ref_profile = np.asarray(ref["profile"], dtype=np.float32)
        mov_profile = np.asarray(mov["profile"], dtype=np.float32)
        ref_centroid = _weighted_profile_centroid(ref_profile)
        mov_centroid = _weighted_profile_centroid(mov_profile)
        if ref_centroid is None or mov_centroid is None:
            return {"found": False, "method": "mask_centroid_y", "failure_reason": "empty_foreground_centroid"}
        shift_y = int(round(float(ref_centroid) - float(mov_centroid)))
        shift_y = int(np.clip(shift_y, -max_shift, max_shift))
        return {
            "found": True,
            "method": "mask_centroid_y",
            "shift_y": int(shift_y),
            "score": None,
            "reference_centroid_y": float(ref_centroid),
            "moving_centroid_y": float(mov_centroid),
        }

    if method in {"row_projection_y", "boundary_profile_y", "intensity_projection_y"}:
        if method == "row_projection_y":
            ref = foreground_profile_y(reference_arr, options=opts, profile_kind="row_projection")
            mov = foreground_profile_y(moving_arr, options=opts, profile_kind="row_projection")
        elif method == "boundary_profile_y":
            ref = foreground_profile_y(reference_arr, options=opts, profile_kind="boundary")
            mov = foreground_profile_y(moving_arr, options=opts, profile_kind="boundary")
        else:
            ref = intensity_profile_y(reference_arr, options=opts)
            mov = intensity_profile_y(moving_arr, options=opts)

        if ref.get("found") is not True or mov.get("found") is not True:
            return {"found": False, "method": method, "failure_reason": "profile_not_found", "reference_profile": ref, "moving_profile": mov}

        match = find_best_vertical_profile_shift(
            np.asarray(ref["profile"], dtype=np.float32),
            np.asarray(mov["profile"], dtype=np.float32),
            reference_valid=np.asarray(ref.get("valid", np.isfinite(ref["profile"])), dtype=bool),
            moving_valid=np.asarray(mov.get("valid", np.isfinite(mov["profile"])), dtype=bool),
            max_shift=max_shift,
            min_overlap_fraction=float(opts.get("min_profile_overlap_fraction", 0.60)),
        )
        if match.get("found") is not True:
            return {"found": False, "method": method, "failure_reason": "profile_match_failed", **match}
        return {
            "found": True,
            "method": method,
            "shift_y": int(match["shift_y"]),
            "score": float(match["score"]),
            "overlap_rows": int(match["overlap_rows"]),
            "profile_kind": ref.get("profile_kind"),
        }

    return {"found": False, "method": method, "failure_reason": f"unknown_alignment_method:{method}"}


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
    profile = foreground_profile_y(image, options=opts, profile_kind="boundary")
    if profile.get("found") is not True:
        return {
            "found": False,
            "failure_reason": profile.get("failure_reason", "boundary_profile_not_found"),
            "image_shape": tuple(int(x) for x in _as_float2d(image).shape),
        }

    smoothed = np.asarray(profile["profile"], dtype=np.float32)
    valid_idx = np.where(np.asarray(profile.get("valid", np.isfinite(smoothed)), dtype=bool) & np.isfinite(smoothed))[0]
    if valid_idx.size == 0:
        return {"found": False, "failure_reason": "empty_boundary_profile", "image_shape": profile.get("image_shape")}

    height, width = tuple(profile.get("image_shape", (len(smoothed), 1)))
    side = str(profile.get("tip_side", "right"))
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
        "foreground_pixels": int(profile.get("foreground_pixels", 0)),
        "image_shape": (int(height), int(width)),
    }


def foreground_profile_y(
    image: torch.Tensor | np.ndarray,
    *,
    options: dict[str, Any] | None = None,
    profile_kind: str = "row_projection",
) -> dict[str, Any]:
    """Build a 1D vertical profile from the foreground breast mask."""
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

    kind = str(profile_kind or "row_projection").strip().casefold()
    if kind in {"row_projection", "projection", "mask_projection"}:
        profile = mask.sum(axis=1).astype(np.float32)
        valid = profile > 0
        smooth_rows = int(opts.get("projection_smooth_rows", opts.get("smooth_rows", 51)) or 51)
        profile = _smooth_profile_1d(profile, valid=valid, smooth_rows=smooth_rows, keep_invalid_zero=True)
        return {
            "found": True,
            "profile_kind": "row_projection",
            "profile": profile,
            "valid": valid,
            "foreground_pixels": int(mask.sum()),
            "image_shape": (int(height), int(width)),
        }

    if kind in {"boundary", "boundary_profile", "outer_boundary"}:
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

        smooth_rows = int(opts.get("boundary_smooth_rows", opts.get("smooth_rows", 31)) or 31)
        smoothed = _smooth_profile_1d(profile, valid=np.isfinite(profile), smooth_rows=smooth_rows, keep_invalid_zero=False)
        valid = np.isfinite(smoothed)
        return {
            "found": bool(valid.any()),
            "failure_reason": None if bool(valid.any()) else "empty_boundary_profile",
            "profile_kind": "boundary_profile",
            "profile": smoothed,
            "valid": valid,
            "tip_side": side,
            "foreground_pixels": int(mask.sum()),
            "image_shape": (int(height), int(width)),
        }

    return {"found": False, "failure_reason": f"unknown_profile_kind:{profile_kind}", "image_shape": (int(height), int(width))}


def intensity_profile_y(
    image: torch.Tensor | np.ndarray,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a 1D vertical intensity projection profile.

    This method is kept as the more expensive/intensity-based alternative the
    user originally described. In practice it is most useful when both paired
    images have similar intensity preprocessing.
    """
    opts = make_contralateral_alignment_options(options)
    arr = _as_float2d(image)
    height, width = arr.shape
    mask = _breast_mask(arr, threshold=opts.get("threshold"))
    mask = _largest_component_mask(mask)
    if not mask.any():
        return {"found": False, "failure_reason": "empty_foreground_mask", "image_shape": (int(height), int(width))}
    masked = np.where(mask, arr, 0.0).astype(np.float32)
    profile = masked.sum(axis=1).astype(np.float32)
    valid = mask.sum(axis=1) > 0
    smooth_rows = int(opts.get("projection_smooth_rows", opts.get("smooth_rows", 51)) or 51)
    profile = _smooth_profile_1d(profile, valid=valid, smooth_rows=smooth_rows, keep_invalid_zero=True)
    return {
        "found": bool(valid.any()),
        "failure_reason": None if bool(valid.any()) else "empty_intensity_profile",
        "profile_kind": "intensity_projection",
        "profile": profile,
        "valid": valid,
        "foreground_pixels": int(mask.sum()),
        "image_shape": (int(height), int(width)),
    }


def find_best_vertical_profile_shift(
    reference_profile: np.ndarray,
    moving_profile: np.ndarray,
    *,
    reference_valid: np.ndarray | None = None,
    moving_valid: np.ndarray | None = None,
    max_shift: int = 0,
    min_overlap_fraction: float = 0.60,
) -> dict[str, Any]:
    """Find integer y-shift that maximizes normalized 1D profile correlation.

    Positive shift means the moving profile is shifted downward.
    """
    ref = np.asarray(reference_profile, dtype=np.float32).reshape(-1)
    mov = np.asarray(moving_profile, dtype=np.float32).reshape(-1)
    n = int(max(len(ref), len(mov)))
    if n <= 0:
        return {"found": False, "failure_reason": "empty_profiles"}
    if len(ref) != n:
        ref = _pad_1d_to_length(ref, n, value=np.nan)
    if len(mov) != n:
        mov = _pad_1d_to_length(mov, n, value=np.nan)

    if reference_valid is None:
        ref_valid = np.isfinite(ref)
    else:
        ref_valid = _pad_1d_to_length(np.asarray(reference_valid, dtype=bool).reshape(-1), n, value=False).astype(bool)
    if moving_valid is None:
        mov_valid = np.isfinite(mov)
    else:
        mov_valid = _pad_1d_to_length(np.asarray(moving_valid, dtype=bool).reshape(-1), n, value=False).astype(bool)

    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    mov = np.nan_to_num(mov, nan=0.0, posinf=0.0, neginf=0.0)

    max_shift = int(max(0, min(int(max_shift), max(n - 1, 0))))
    min_overlap = int(max(5, round(float(min_overlap_fraction) * float(max(1, min(int(ref_valid.sum()), int(mov_valid.sum()), n))))))

    best_shift: int | None = None
    best_score = -np.inf
    best_overlap = 0
    for shift in range(-max_shift, max_shift + 1):
        ref_seg, mov_seg, valid = _shifted_profile_overlap(ref, mov, ref_valid, mov_valid, shift)
        overlap = int(valid.sum())
        if overlap < min_overlap:
            continue
        score = _normalized_cross_correlation(ref_seg[valid], mov_seg[valid])
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_shift = int(shift)
            best_overlap = int(overlap)

    if best_shift is None:
        return {"found": False, "failure_reason": "no_shift_with_enough_overlap", "min_overlap_rows": int(min_overlap)}
    return {
        "found": True,
        "shift_y": int(best_shift),
        "score": float(best_score),
        "overlap_rows": int(best_overlap),
        "min_overlap_rows": int(min_overlap),
    }


def _shifted_profile_overlap(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_valid: np.ndarray,
    mov_valid: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(len(ref), len(mov)))
    if shift >= 0:
        # moving is shifted down, so moving row 0 corresponds to reference row shift
        ref_seg = ref[shift:n]
        mov_seg = mov[: n - shift]
        valid = ref_valid[shift:n] & mov_valid[: n - shift]
    else:
        sy = -int(shift)
        ref_seg = ref[: n - sy]
        mov_seg = mov[sy:n]
        valid = ref_valid[: n - sy] & mov_valid[sy:n]
    return ref_seg, mov_seg, valid


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size < 3 or b.size < 3:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _weighted_profile_centroid(profile: np.ndarray) -> float | None:
    p = np.nan_to_num(np.asarray(profile, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= 1e-8:
        return None
    y = np.arange(p.size, dtype=np.float32)
    return float(np.dot(y, p) / total)


def _smooth_profile_1d(
    profile: np.ndarray,
    *,
    valid: np.ndarray | None,
    smooth_rows: int,
    keep_invalid_zero: bool,
) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float32).reshape(-1)
    if valid is None:
        valid_arr = np.isfinite(values)
    else:
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_arr.size != values.size:
            valid_arr = _pad_1d_to_length(valid_arr, values.size, value=False).astype(bool)
    smooth_rows = int(smooth_rows or 1)
    if smooth_rows <= 1 or int(valid_arr.sum()) < 3:
        return np.nan_to_num(values, nan=0.0) if keep_invalid_zero else np.where(valid_arr, values, np.nan)
    smooth_rows = max(3, smooth_rows | 1)
    finite_values = np.where(valid_arr & np.isfinite(values), values, 0.0).astype(np.float32)
    weights = (valid_arr & np.isfinite(values)).astype(np.float32)
    kernel = np.ones((smooth_rows,), dtype=np.float32)
    numerator = np.convolve(finite_values, kernel, mode="same")
    denominator = np.convolve(weights, kernel, mode="same")
    out = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-6), np.nan).astype(np.float32)
    if keep_invalid_zero:
        return np.nan_to_num(out, nan=0.0)
    observed = np.where(valid_arr)[0]
    if observed.size:
        half = smooth_rows // 2
        out[: max(0, int(observed.min()) - half)] = np.nan
        out[min(values.size, int(observed.max()) + half + 1) :] = np.nan
    return out


def _pad_1d_to_length(arr: np.ndarray, length: int, *, value: Any) -> np.ndarray:
    arr = np.asarray(arr)
    length = int(length)
    if arr.size >= length:
        return arr[:length]
    out = np.full((length,), value, dtype=arr.dtype)
    out[: arr.size] = arr
    return out


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
