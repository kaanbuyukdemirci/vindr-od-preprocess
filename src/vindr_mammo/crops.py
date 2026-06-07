from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


DEFAULT_CROP_OPTIONS: dict[str, Any] = {
    "enabled": False,
    "mode": "random",                 # "random" or "deterministic"
    "crop_size": 1024,                 # square crop size n x n
    "stride": 768,                     # deterministic sliding-window stride
    "random_crops_per_image": 1,       # only used by index_level="crop" in random mode
    "positive_fraction": 0.80,         # probability of sampling a mass-positive crop
    "center_on_mass": True,            # for positive random crops
    "center_shift_fraction": 0.25,     # random shift around mass center, relative to crop size
    "allow_partial_annotations": False,
    "min_box_visibility": 0.30,        # used if partial annotations are allowed
    "reject_partial_windows": True,    # if partial annotations are not allowed, reject windows cutting a mass
    "negative_max_box_visibility": 0.0, # clean crops should not include visible mass by default
    "pad_if_needed": True,             # pad small crops/images to n x n
    "pad_value": 0.0,
    "max_random_tries": 80,
    # BBox-safe, breast-biased random crop mode. The annotation boxes that are
    # visible in the crop must be fully inside the crop-safe inner region, not
    # near the crop boundary. A candidate pool is sampled randomly and then
    # biased toward windows containing more breast foreground.
    "bbox_safe_boundary_margin_fraction": 0.02,
    "bbox_safe_random_shift_fraction": 0.35,
    "bbox_safe_candidate_count": 120,
    "bbox_safe_top_k": 8,
    "bbox_safe_breast_bias_strength": 1.0,
    "bbox_safe_left_bias_strength": 0.25,
    "bbox_safe_projection_bias_strength": 0.25,
    "bbox_safe_foreground_threshold": None,
    "bbox_safe_foreground_threshold_fraction": 0.02,
    "bbox_safe_foreground_keep_largest_component": True,
    "bbox_safe_foreground_min_component_area_fraction": 0.001,
    "bbox_safe_min_foreground_fraction": 0.35,
    # If True, bbox_safe_random may report an unsafe fallback, but exporters and
    # previews should skip it instead of writing a crop that violates the margin.
    "bbox_safe_skip_unsafe_fallbacks": True,
    "deterministic_include_empty": True,
    "deterministic_max_windows_per_image": None,
    "seed": None,
}


@dataclass(frozen=True)
class SquareCropResult:
    image: torch.Tensor
    boxes: torch.Tensor
    mass_boxes: torch.Tensor
    box_keep: torch.Tensor
    mass_box_keep: torch.Tensor
    info: dict[str, Any]


def make_crop_options(options: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_CROP_OPTIONS)
    if options:
        out.update(options)
    out["mode"] = str(out.get("mode", "random")).casefold().strip()
    if out["mode"] not in {"random", "deterministic", "bbox_safe_random"}:
        raise ValueError("crop_options['mode'] must be 'random', 'deterministic', or 'bbox_safe_random'.")
    out["crop_size"] = int(out["crop_size"])
    out["stride"] = int(out["stride"])
    if out["crop_size"] <= 0:
        raise ValueError("crop_options['crop_size'] must be positive.")
    if out["stride"] <= 0:
        raise ValueError("crop_options['stride'] must be positive.")
    return out


def sliding_square_windows(width: int, height: int, crop_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Return n x n sliding windows that cover an image, including edge-aligned windows."""
    n = int(crop_size)
    stride = int(stride)
    width = int(width)
    height = int(height)

    if width <= n:
        xs = [0]
    else:
        xs = list(range(0, width - n + 1, stride))
        if xs[-1] != width - n:
            xs.append(width - n)

    if height <= n:
        ys = [0]
    else:
        ys = list(range(0, height - n + 1, stride))
        if ys[-1] != height - n:
            ys.append(height - n)

    return [(int(x), int(y), int(x + n), int(y + n)) for y in ys for x in xs]


def sample_random_square_window(
    *,
    image_width: int,
    image_height: int,
    mass_boxes: torch.Tensor,
    options: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Sample a random n x n crop window with optional mass-positive balancing."""
    opts = make_crop_options(options)
    n = int(opts["crop_size"])
    positive_fraction = float(opts.get("positive_fraction", 0.8))
    boxes = _as_boxes(mass_boxes)
    has_mass = boxes.shape[0] > 0
    want_positive = bool(has_mass and rng.random() < positive_fraction)
    max_tries = int(opts.get("max_random_tries", 80))

    last_window = _random_any_window(image_width, image_height, n, rng)
    for _ in range(max_tries):
        if want_positive:
            window = _random_positive_window(
                image_width=image_width,
                image_height=image_height,
                crop_size=n,
                mass_boxes=boxes,
                center_on_mass=bool(opts.get("center_on_mass", True)),
                center_shift_fraction=float(opts.get("center_shift_fraction", 0.25)),
                rng=rng,
            )
            ok = window_has_positive_mass(window, boxes, opts)
        else:
            window = _random_any_window(image_width, image_height, n, rng)
            ok = window_is_clean(window, boxes, opts)
        last_window = window
        if ok:
            return window, {"requested_positive": want_positive, "accepted": True}

    # Fallback: return the last tried crop. This avoids infinite loops in dense cases.
    return last_window, {"requested_positive": want_positive, "accepted": False, "fallback_after_tries": max_tries}


def crop_image_and_boxes_to_window(
    image: torch.Tensor,
    *,
    boxes: torch.Tensor,
    mass_boxes: torch.Tensor,
    window_xyxy: tuple[int, int, int, int],
    options: dict[str, Any],
) -> SquareCropResult:
    """Crop/pad image to a square window and shift/clip boxes into crop coordinates."""
    opts = make_crop_options(options)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected image tensor shaped [1, H, W], got {tuple(image.shape)}")

    n = int(opts["crop_size"])
    x0, y0, x1, y1 = [int(v) for v in window_xyxy]
    height, width = int(image.shape[-2]), int(image.shape[-1])

    # The requested window can extend beyond the image if image is smaller than n.
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(width, x1)
    src_y1 = min(height, y1)

    cropped = image[:, src_y0:src_y1, src_x0:src_x1].contiguous()

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    if any(v > 0 for v in [pad_left, pad_right, pad_top, pad_bottom]):
        if not bool(opts.get("pad_if_needed", True)):
            raise ValueError("Crop window extends outside image and pad_if_needed=False.")
        cropped = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), value=float(opts.get("pad_value", 0.0)))

    # A safety pad/crop in case edge cases produce one-pixel mismatch.
    cropped = cropped[:, :n, :n]
    if cropped.shape[-2] < n or cropped.shape[-1] < n:
        cropped = F.pad(cropped, (0, n - cropped.shape[-1], 0, n - cropped.shape[-2]), value=float(opts.get("pad_value", 0.0)))

    shifted_boxes, box_keep = boxes_in_window(boxes, window_xyxy, opts)
    shifted_mass_boxes, mass_box_keep = boxes_in_window(mass_boxes, window_xyxy, opts)

    info = {
        "square_crop_enabled": True,
        "crop_mode": opts["mode"],
        "crop_size": n,
        "window_xyxy": (x0, y0, x1, y1),
        "source_image_shape": (height, width),
        "crop_shape": (int(cropped.shape[-2]), int(cropped.shape[-1])),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "num_mass_boxes_after_crop": int(shifted_mass_boxes.shape[0]),
        "is_positive_crop": int(shifted_mass_boxes.shape[0]) > 0,
    }
    return SquareCropResult(cropped, shifted_boxes, shifted_mass_boxes, box_keep, mass_box_keep, info)


def boxes_in_window(
    boxes: torch.Tensor,
    window_xyxy: tuple[int, int, int, int],
    options: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip boxes to window and return kept boxes plus keep mask over original boxes."""
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    x0, y0, x1, y1 = [float(v) for v in window_xyxy]
    opts = make_crop_options(options)
    allow_partial = bool(opts.get("allow_partial_annotations", False))
    min_visibility = float(opts.get("min_box_visibility", 0.30))

    inter_x0 = torch.clamp(boxes[:, 0], min=x0, max=x1)
    inter_y0 = torch.clamp(boxes[:, 1], min=y0, max=y1)
    inter_x1 = torch.clamp(boxes[:, 2], min=x0, max=x1)
    inter_y1 = torch.clamp(boxes[:, 3], min=y0, max=y1)
    inter_w = (inter_x1 - inter_x0).clamp(min=0)
    inter_h = (inter_y1 - inter_y0).clamp(min=0)
    inter_area = inter_w * inter_h

    box_area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).clamp(min=1e-12)
    visibility = inter_area / box_area
    full_inside = (boxes[:, 0] >= x0) & (boxes[:, 1] >= y0) & (boxes[:, 2] <= x1) & (boxes[:, 3] <= y1)
    if allow_partial:
        keep = visibility >= min_visibility
    else:
        keep = full_inside

    shifted = torch.stack([inter_x0 - x0, inter_y0 - y0, inter_x1 - x0, inter_y1 - y0], dim=1)
    shifted = shifted[keep].contiguous().to(dtype=torch.float32)
    return shifted, keep.cpu()


def window_has_positive_mass(window_xyxy: tuple[int, int, int, int], boxes: torch.Tensor, options: dict[str, Any]) -> bool:
    kept, keep = boxes_in_window(boxes, window_xyxy, options)
    if kept.shape[0] == 0:
        return False
    opts = make_crop_options(options)
    if bool(opts.get("allow_partial_annotations", False)):
        return True
    if bool(opts.get("reject_partial_windows", True)):
        return not _window_cuts_any_box(window_xyxy, boxes)
    return bool(keep.any())


def window_is_clean(window_xyxy: tuple[int, int, int, int], boxes: torch.Tensor, options: dict[str, Any]) -> bool:
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return True
    max_vis = float(make_crop_options(options).get("negative_max_box_visibility", 0.0))
    vis = box_visibility_in_window(boxes, window_xyxy)
    return bool(float(vis.max().item()) <= max_vis)


def box_visibility_in_window(boxes: torch.Tensor, window_xyxy: tuple[int, int, int, int]) -> torch.Tensor:
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    x0, y0, x1, y1 = [float(v) for v in window_xyxy]
    inter_x0 = torch.clamp(boxes[:, 0], min=x0, max=x1)
    inter_y0 = torch.clamp(boxes[:, 1], min=y0, max=y1)
    inter_x1 = torch.clamp(boxes[:, 2], min=x0, max=x1)
    inter_y1 = torch.clamp(boxes[:, 3], min=y0, max=y1)
    inter = (inter_x1 - inter_x0).clamp(min=0) * (inter_y1 - inter_y0).clamp(min=0)
    area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).clamp(min=1e-12)
    return inter / area


def crop_size_fit_table(mass_df: Any, current_crop_size: int | None = None) -> Any:
    """Return crop-size recommendations for individual boxes and all masses per image."""
    import pandas as pd

    if mass_df is None or len(mass_df) == 0:
        columns = ["basis", "num_items", "n_for_90_percent", "n_for_95_percent", "n_for_99_percent", "n_for_100_percent", "current_n", "current_n_fits_percent"]
        return pd.DataFrame(columns=columns)

    df = mass_df.copy()
    df["bbox_width"] = pd.to_numeric(df["bbox_width"], errors="coerce")
    df["bbox_height"] = pd.to_numeric(df["bbox_height"], errors="coerce")
    df = df.dropna(subset=["bbox_width", "bbox_height"])
    if df.empty:
        return pd.DataFrame()

    per_box = np.maximum(df["bbox_width"].to_numpy(float), df["bbox_height"].to_numpy(float))
    rows = [_fit_row("single_mass_box", per_box, current_crop_size)]

    if "image_id" in df.columns:
        grouped = []
        for _, group in df.groupby("image_id"):
            xmin = pd.to_numeric(group["xmin"], errors="coerce").min()
            ymin = pd.to_numeric(group["ymin"], errors="coerce").min()
            xmax = pd.to_numeric(group["xmax"], errors="coerce").max()
            ymax = pd.to_numeric(group["ymax"], errors="coerce").max()
            if pd.notna(xmin) and pd.notna(ymin) and pd.notna(xmax) and pd.notna(ymax):
                grouped.append(max(float(xmax - xmin), float(ymax - ymin)))
        if grouped:
            rows.append(_fit_row("all_mass_boxes_in_same_image", np.asarray(grouped, dtype=float), current_crop_size))
    return pd.DataFrame(rows)


def _fit_row(name: str, values: np.ndarray, current_crop_size: int | None) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"basis": name, "num_items": 0}
    current_fit = None
    if current_crop_size is not None:
        current_fit = 100.0 * float(np.mean(values <= float(current_crop_size)))
    return {
        "basis": name,
        "num_items": int(values.size),
        "n_for_90_percent": float(np.quantile(values, 0.90)),
        "n_for_95_percent": float(np.quantile(values, 0.95)),
        "n_for_99_percent": float(np.quantile(values, 0.99)),
        "n_for_100_percent": float(np.max(values)),
        "current_n": None if current_crop_size is None else int(current_crop_size),
        "current_n_fits_percent": current_fit,
    }


def _random_any_window(width: int, height: int, crop_size: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    n = int(crop_size)
    max_x = max(0, int(width) - n)
    max_y = max(0, int(height) - n)
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    return (x0, y0, x0 + n, y0 + n)

def _safe_origin_interval_for_box(
    box: torch.Tensor,
    crop_size: int,
    margin_px: float,
    max_origin: int,
    *,
    axis: int,
) -> tuple[int, int] | None:
    """Return crop-origin interval that keeps a box inside the safe inner area."""
    n = float(crop_size)
    b0 = float(box[0 if axis == 0 else 1])
    b1 = float(box[2 if axis == 0 else 3])
    lo = int(math.ceil(b1 - (n - float(margin_px))))
    hi = int(math.floor(b0 - float(margin_px)))
    lo = max(0, lo)
    hi = min(int(max_origin), hi)
    if lo > hi:
        return None
    return lo, hi


def _coarse_positions_between(min_start: int, max_start: int, step: int) -> list[int]:
    min_start = int(min_start)
    max_start = int(max_start)
    if max_start <= min_start:
        return [min_start]
    vals = list(range(min_start, max_start + 1, max(1, int(step))))
    if vals[-1] != max_start:
        vals.append(max_start)
    return vals


def _bbox_safe_failed_window(
    *,
    image_width: int,
    image_height: int,
    crop_size: int,
    box: torch.Tensor,
    rng: np.random.Generator,
    candidate_count: int,
    margin_fraction: float,
    margin_px: float,
    reason: str,
    last_window: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Return an unsafe fallback window, explicitly marked so callers can skip it."""
    if last_window is None:
        last_window = _random_positive_window(
            image_width=image_width,
            image_height=image_height,
            crop_size=crop_size,
            mass_boxes=box.reshape(1, 4),
            center_on_mass=True,
            center_shift_fraction=0.0,
            rng=rng,
        )
    return last_window, {
        "requested_positive": True,
        "accepted": False,
        "centered_on_annotation": True,
        "bbox_safe_failed": True,
        "bbox_safe_failure_reason": str(reason),
        "fallback_after_tries": int(candidate_count),
        "bbox_safe_boundary_margin_fraction": float(margin_fraction),
        "bbox_safe_boundary_margin_px": float(margin_px),
    }



def _random_positive_window(
    *,
    image_width: int,
    image_height: int,
    crop_size: int,
    mass_boxes: torch.Tensor,
    center_on_mass: bool,
    center_shift_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    n = int(crop_size)
    boxes = _as_boxes(mass_boxes)
    if boxes.numel() == 0 or not center_on_mass:
        return _random_any_window(image_width, image_height, n, rng)

    box = boxes[int(rng.integers(0, boxes.shape[0]))]
    cx = float((box[0] + box[2]) / 2.0)
    cy = float((box[1] + box[3]) / 2.0)
    shift = float(center_shift_fraction) * float(n)
    cx += float(rng.uniform(-shift, shift))
    cy += float(rng.uniform(-shift, shift))

    x0 = int(round(cx - n / 2.0))
    y0 = int(round(cy - n / 2.0))
    x0 = min(max(0, x0), max(0, int(image_width) - n))
    y0 = min(max(0, y0), max(0, int(image_height) - n))
    return (x0, y0, x0 + n, y0 + n)




def sample_bbox_safe_breast_biased_square_window(
    *,
    image_width: int,
    image_height: int,
    image_tensor: torch.Tensor,
    box_xyxy: torch.Tensor | list[float] | tuple[float, float, float, float],
    all_mass_boxes: torch.Tensor,
    options: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Sample a random crop where visible boxes are away from crop boundaries.

    This is a bbox-aware strategy for object-detection export. It samples a
    random candidate pool around a selected annotation, rejects candidates where
    any visible annotation is clipped or lands inside the boundary margin, and
    then randomly chooses among the best breast-foreground candidates. The result
    is random, but constrained so masses are not placed near the crop edge.
    """
    opts = make_crop_options(options)
    n = int(opts["crop_size"])
    boxes = _as_boxes(all_mass_boxes)
    box = torch.as_tensor(box_xyxy, dtype=torch.float32).reshape(4)
    max_tries = int(opts.get("max_random_tries", 80))
    candidate_count = max(1, int(opts.get("bbox_safe_candidate_count", max_tries)))
    candidate_count = max(candidate_count, max_tries)
    top_k = max(1, int(opts.get("bbox_safe_top_k", 8)))
    margin_fraction = float(opts.get("bbox_safe_boundary_margin_fraction", 0.02))
    margin_px = max(0.0, min(0.49, margin_fraction)) * float(n)
    shift = float(opts.get("bbox_safe_random_shift_fraction", opts.get("center_shift_fraction", 0.25))) * float(n)

    image_np = _image_tensor_to_2d_numpy(image_tensor)
    foreground_mask = _foreground_mask_for_crop_sampling(image_np, opts.get("bbox_safe_foreground_threshold", None), opts)
    projection = foreground_mask.sum(axis=0).astype(np.float32) if foreground_mask.size else np.zeros((max(1, int(image_width)),), dtype=np.float32)
    peak_x = int(np.argmax(projection)) if projection.size and float(projection.max()) > 0 else int(image_width) // 2

    cx0 = float((box[0] + box[2]) / 2.0)
    cy0 = float((box[1] + box[3]) / 2.0)
    max_x = max(0, int(image_width) - n)
    max_y = max(0, int(image_height) - n)
    candidates: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
    last_window = None

    feasible_x = _safe_origin_interval_for_box(box, n, margin_px, max_x, axis=0)
    feasible_y = _safe_origin_interval_for_box(box, n, margin_px, max_y, axis=1)
    if feasible_x is None or feasible_y is None:
        return _bbox_safe_failed_window(
            image_width=image_width,
            image_height=image_height,
            crop_size=n,
            box=box,
            rng=rng,
            candidate_count=candidate_count,
            margin_fraction=margin_fraction,
            margin_px=margin_px,
            reason="target_box_cannot_fit_inside_safe_margin",
        )

    x_lo, x_hi = feasible_x
    y_lo, y_hi = feasible_y

    for _ in range(candidate_count):
        # Sample only from origins that make the target annotation margin-safe.
        # This avoids clamping a crop to the image edge and accidentally placing
        # the target box in the forbidden boundary band.
        if rng.random() < 0.75:
            cx = cx0 + float(rng.uniform(-shift, shift))
            cy = cy0 + float(rng.uniform(-shift, shift))
            x0 = int(round(cx - n / 2.0))
            y0 = int(round(cy - n / 2.0))
            x0 = min(max(x_lo, x0), x_hi)
            y0 = min(max(y_lo, y0), y_hi)
        else:
            x0 = int(rng.integers(x_lo, x_hi + 1)) if x_hi > x_lo else int(x_lo)
            y0 = int(rng.integers(y_lo, y_hi + 1)) if y_hi > y_lo else int(y_lo)
        window = (x0, y0, x0 + n, y0 + n)
        last_window = window
        ok, safety_info = _window_satisfies_bbox_safe_margin(window, boxes, box, margin_px)
        if not ok:
            continue
        fg = _foreground_fraction_from_mask(foreground_mask, window, n)
        left_score = 1.0 - (float(x0) / float(max_x)) if max_x > 0 else 1.0
        projection_score = _projection_peak_score(peak_x, window, n)
        score = (
            float(opts.get("bbox_safe_breast_bias_strength", 1.0)) * fg
            + float(opts.get("bbox_safe_left_bias_strength", 0.25)) * left_score
            + float(opts.get("bbox_safe_projection_bias_strength", 0.25)) * projection_score
        )
        info = {
            "requested_positive": True,
            "accepted": True,
            "centered_on_annotation": True,
            "bbox_safe_boundary_margin_fraction": float(margin_fraction),
            "bbox_safe_boundary_margin_px": float(margin_px),
            "bbox_safe_foreground_fraction": float(fg),
            "bbox_safe_left_score": float(left_score),
            "bbox_safe_projection_score": float(projection_score),
            "bbox_safe_score": float(score),
            **safety_info,
        }
        candidates.append((score, window, info))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[: min(top_k, len(candidates))]
        # Randomly choose among the best candidates so this remains stochastic,
        # while still preferring high-breast-foreground crops.
        idx = int(rng.integers(0, len(top)))
        return top[idx][1], top[idx][2]

    # Fallback: choose the best valid window on a coarse deterministic grid. This
    # keeps the export robust if random sampling misses a narrow feasible region.
    # The grid is still restricted to target-safe origins.
    grid_candidates: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
    for x0 in _coarse_positions_between(x_lo, x_hi, max(8, n // 4)):
        for y0 in _coarse_positions_between(y_lo, y_hi, max(8, n // 4)):
            window = (int(x0), int(y0), int(x0 + n), int(y0 + n))
            ok, safety_info = _window_satisfies_bbox_safe_margin(window, boxes, box, margin_px)
            if not ok:
                continue
            fg = _foreground_fraction_from_mask(foreground_mask, window, n)
            left_score = 1.0 - (float(x0) / float(max_x)) if max_x > 0 else 1.0
            projection_score = _projection_peak_score(peak_x, window, n)
            score = fg + 0.25 * left_score + 0.25 * projection_score
            grid_candidates.append((score, window, {"requested_positive": True, "accepted": True, "bbox_safe_grid_fallback": True, "bbox_safe_foreground_fraction": float(fg), **safety_info}))
    if grid_candidates:
        grid_candidates.sort(key=lambda item: item[0], reverse=True)
        return grid_candidates[0][1], grid_candidates[0][2]

    return _bbox_safe_failed_window(
        image_width=image_width,
        image_height=image_height,
        crop_size=n,
        box=box,
        rng=rng,
        candidate_count=candidate_count,
        margin_fraction=margin_fraction,
        margin_px=margin_px,
        reason="no_candidate_satisfied_all_visible_boxes",
        last_window=last_window,
    )


def sample_breast_biased_clean_square_window(
    *,
    image_width: int,
    image_height: int,
    image_tensor: torch.Tensor,
    mass_boxes: torch.Tensor,
    options: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Sample a clean negative crop biased toward breast foreground."""
    opts = make_crop_options(options)
    n = int(opts["crop_size"])
    boxes = _as_boxes(mass_boxes)
    max_tries = max(1, int(opts.get("max_random_tries", 80)))
    candidate_count = max(max_tries, int(opts.get("bbox_safe_candidate_count", max_tries)))
    image_np = _image_tensor_to_2d_numpy(image_tensor)
    foreground_mask = _foreground_mask_for_crop_sampling(image_np, opts.get("bbox_safe_foreground_threshold", None), opts)
    projection = foreground_mask.sum(axis=0).astype(np.float32) if foreground_mask.size else np.zeros((max(1, int(image_width)),), dtype=np.float32)
    peak_x = int(np.argmax(projection)) if projection.size and float(projection.max()) > 0 else int(image_width) // 2
    max_x = max(0, int(image_width) - n)
    best: tuple[float, tuple[int, int, int, int], dict[str, Any]] | None = None
    for _ in range(candidate_count):
        window = _random_any_window(image_width, image_height, n, rng)
        if not window_is_clean(window, boxes, opts):
            continue
        x0 = window[0]
        fg = _foreground_fraction_from_mask(foreground_mask, window, n)
        min_fg = float(opts.get("bbox_safe_min_foreground_fraction", opts.get("min_foreground_fraction", 0.0)) or 0.0)
        if min_fg > 0.0 and fg < min_fg:
            continue
        left_score = 1.0 - (float(x0) / float(max_x)) if max_x > 0 else 1.0
        projection_score = _projection_peak_score(peak_x, window, n)
        score = fg + 0.25 * left_score + 0.25 * projection_score
        info = {
            "requested_positive": False,
            "accepted": True,
            "bbox_safe_foreground_fraction": float(fg),
            "bbox_safe_left_score": float(left_score),
            "bbox_safe_projection_score": float(projection_score),
            "bbox_safe_score": float(score),
        }
        if best is None or score > best[0]:
            best = (score, window, info)
    if best is not None:
        return best[1], best[2]
    window = _random_any_window(image_width, image_height, n, rng)
    return window, {"requested_positive": False, "accepted": False, "bbox_safe_clean_fallback": True, "fallback_after_tries": int(candidate_count)}


def validate_bbox_safe_window(
    window_xyxy: tuple[int, int, int, int],
    boxes: torch.Tensor,
    *,
    crop_size: int,
    margin_fraction: float,
    target_box: torch.Tensor | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate the hard bbox-safe rule for a proposed export crop.

    All visible boxes must be completely inside the crop and fully outside the
    forbidden boundary band. If target_box is provided, that target box must also
    be visible and safe.
    """
    margin_px = max(0.0, min(0.49, float(margin_fraction))) * float(crop_size)
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return True, {
            "bbox_safe_boundary_margin_fraction": float(margin_fraction),
            "bbox_safe_boundary_margin_px": float(margin_px),
            "bbox_safe_visible_boxes": 0,
            "bbox_safe_boxes_inside_margin": 0,
            "bbox_safe_margin_ok": 1,
        }
    if target_box is not None:
        target = torch.as_tensor(target_box, dtype=torch.float32).reshape(4)
        ok, info = _window_satisfies_bbox_safe_margin(window_xyxy, boxes, target, margin_px)
        info.update({
            "bbox_safe_boundary_margin_fraction": float(margin_fraction),
            "bbox_safe_boundary_margin_px": float(margin_px),
        })
        return ok, info

    x0, y0, x1, y1 = [float(v) for v in window_xyxy]
    vis = box_visibility_in_window(boxes, window_xyxy)
    visible = vis > 0.0
    if not bool(visible.any().item()):
        return True, {
            "bbox_safe_boundary_margin_fraction": float(margin_fraction),
            "bbox_safe_boundary_margin_px": float(margin_px),
            "bbox_safe_visible_boxes": 0,
            "bbox_safe_boxes_inside_margin": 0,
            "bbox_safe_margin_ok": 1,
        }
    visible_boxes = boxes[visible]
    sx0 = visible_boxes[:, 0] - x0
    sy0 = visible_boxes[:, 1] - y0
    sx1 = visible_boxes[:, 2] - x0
    sy1 = visible_boxes[:, 3] - y0
    crop_w = x1 - x0
    crop_h = y1 - y0
    full_inside = (
        (visible_boxes[:, 0] >= x0)
        & (visible_boxes[:, 1] >= y0)
        & (visible_boxes[:, 2] <= x1)
        & (visible_boxes[:, 3] <= y1)
    )
    inside_margin = (
        (sx0 >= margin_px)
        & (sy0 >= margin_px)
        & (sx1 <= crop_w - margin_px)
        & (sy1 <= crop_h - margin_px)
    )
    safe = full_inside & inside_margin
    ok = bool(safe.all().item())
    return ok, {
        "bbox_safe_boundary_margin_fraction": float(margin_fraction),
        "bbox_safe_boundary_margin_px": float(margin_px),
        "bbox_safe_visible_boxes": int(visible.sum().item()),
        "bbox_safe_boxes_inside_margin": int(safe.sum().item()),
        "bbox_safe_margin_ok": int(ok),
        "bbox_safe_min_visibility_visible_boxes": float(vis[visible].min().item()) if bool(visible.any().item()) else 0.0,
    }


def exported_boxes_satisfy_bbox_safe_margin(
    boxes: torch.Tensor,
    *,
    crop_size: int,
    margin_fraction: float,
) -> tuple[bool, dict[str, Any]]:
    """Validate already shifted crop-coordinate boxes before writing labels."""
    boxes = _as_boxes(boxes)
    margin_px = max(0.0, min(0.49, float(margin_fraction))) * float(crop_size)
    if boxes.numel() == 0:
        return True, {
            "bbox_safe_exported_boxes": 0,
            "bbox_safe_exported_boxes_inside_margin": 0,
            "bbox_safe_export_margin_ok": 1,
        }
    safe = (
        (boxes[:, 0] >= margin_px)
        & (boxes[:, 1] >= margin_px)
        & (boxes[:, 2] <= float(crop_size) - margin_px)
        & (boxes[:, 3] <= float(crop_size) - margin_px)
    )
    ok = bool(safe.all().item())
    return ok, {
        "bbox_safe_exported_boxes": int(boxes.shape[0]),
        "bbox_safe_exported_boxes_inside_margin": int(safe.sum().item()),
        "bbox_safe_export_margin_ok": int(ok),
    }


def _window_satisfies_bbox_safe_margin(
    window_xyxy: tuple[int, int, int, int],
    boxes: torch.Tensor,
    target_box: torch.Tensor,
    margin_px: float,
) -> tuple[bool, dict[str, Any]]:
    """Return whether every visible annotation is safely inside the crop."""
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return False, {"bbox_safe_visible_boxes": 0, "bbox_safe_boxes_inside_margin": 0}
    x0, y0, x1, y1 = [float(v) for v in window_xyxy]
    vis = box_visibility_in_window(boxes, window_xyxy)
    visible = vis > 0.0
    if not bool(visible.any().item()):
        return False, {"bbox_safe_visible_boxes": 0, "bbox_safe_boxes_inside_margin": 0}

    # The target box must be visible and safe. This prevents a crop from being
    # accepted only because a different mass happens to be safely inside.
    target = target_box.detach().cpu().to(dtype=torch.float32).reshape(1, 4)
    target_vis = box_visibility_in_window(target, window_xyxy)
    if target_vis.numel() == 0 or float(target_vis.max().item()) <= 0.0:
        return False, {"bbox_safe_visible_boxes": int(visible.sum().item()), "bbox_safe_target_visible": 0}

    visible_boxes = boxes[visible]
    sx0 = visible_boxes[:, 0] - x0
    sy0 = visible_boxes[:, 1] - y0
    sx1 = visible_boxes[:, 2] - x0
    sy1 = visible_boxes[:, 3] - y0
    crop_w = x1 - x0
    crop_h = y1 - y0
    full_inside = (visible_boxes[:, 0] >= x0) & (visible_boxes[:, 1] >= y0) & (visible_boxes[:, 2] <= x1) & (visible_boxes[:, 3] <= y1)
    inside_margin = (sx0 >= margin_px) & (sy0 >= margin_px) & (sx1 <= crop_w - margin_px) & (sy1 <= crop_h - margin_px)
    safe = full_inside & inside_margin
    ok = bool(safe.all().item())
    return ok, {
        "bbox_safe_visible_boxes": int(visible.sum().item()),
        "bbox_safe_boxes_inside_margin": int(safe.sum().item()),
        "bbox_safe_margin_ok": int(ok),
        "bbox_safe_min_visibility_visible_boxes": float(vis[visible].min().item()) if bool(visible.any().item()) else 0.0,
    }


def _image_tensor_to_2d_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    arr = image_tensor.detach().cpu().squeeze().numpy().astype(np.float32, copy=False)
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(arr.shape[-2], arr.shape[-1]).astype(np.float32, copy=False)
    return arr


def _foreground_mask_for_crop_sampling(image_np: np.ndarray, threshold: Any, options: dict[str, Any] | None = None) -> np.ndarray:
    """Robust breast foreground mask used for crop scoring/filtering.

    The old version used a very low threshold, so tiny nonzero background noise
    could turn most of the crop into foreground. This version uses a threshold
    based on the image intensity range and then optionally keeps only the largest
    connected component, which is normally the breast.
    """
    opts = options or {}
    arr = np.asarray(image_np, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    if threshold is None:
        lo, hi = np.percentile(finite, [1.0, 99.5])
        frac = float(opts.get("bbox_safe_foreground_threshold_fraction", opts.get("foreground_threshold_fraction", 0.02)) or 0.02)
        thr = max(float(lo + frac * (hi - lo)), float(lo) + 1e-6)
    else:
        thr = float(threshold)
    mask = finite_mask & (arr > thr)
    if bool(opts.get("bbox_safe_foreground_keep_largest_component", opts.get("foreground_keep_largest_component", True))):
        min_area_fraction = float(opts.get("bbox_safe_foreground_min_component_area_fraction", opts.get("foreground_min_component_area_fraction", 0.001)) or 0.001)
        mask = _keep_large_foreground_components(mask, min_area_fraction=min_area_fraction)
    return mask


def _keep_large_foreground_components(mask: np.ndarray, *, min_area_fraction: float = 0.001) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    if cv2 is None:
        return mask
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    if areas.size == 0:
        return np.zeros_like(mask, dtype=bool)
    min_area = max(1, int(round(float(mask.size) * float(min_area_fraction))))
    largest_idx = int(np.argmax(areas)) + 1
    # Keep the largest breast component and any other reasonably large connected
    # foreground component. Tiny speckles from detector/background noise are removed.
    keep_labels = {largest_idx}
    for idx, area in enumerate(areas, start=1):
        if int(area) >= min_area:
            keep_labels.add(int(idx))
    out = np.isin(labels, list(keep_labels))
    return out.astype(bool, copy=False)


def _foreground_fraction_from_mask(mask: np.ndarray, window_xyxy: tuple[int, int, int, int], crop_size: int) -> float:
    x0, y0, x1, y1 = [int(v) for v in window_xyxy]
    h, w = mask.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return 0.0
    count = float(mask[src_y0:src_y1, src_x0:src_x1].sum())
    return count / float(max(1, int(crop_size) * int(crop_size)))


def _projection_peak_score(peak_x: int, window_xyxy: tuple[int, int, int, int], crop_size: int) -> float:
    x0, _y0, x1, _y1 = [int(v) for v in window_xyxy]
    if x0 <= int(peak_x) <= x1:
        return 1.0
    center = 0.5 * (float(x0) + float(x1))
    distance = abs(float(peak_x) - center)
    return max(0.0, 1.0 - distance / float(max(1, crop_size)))


def _coarse_positions(max_start: int, step: int) -> list[int]:
    max_start = int(max_start)
    if max_start <= 0:
        return [0]
    vals = list(range(0, max_start + 1, max(1, int(step))))
    if vals[-1] != max_start:
        vals.append(max_start)
    return vals

def _window_cuts_any_box(window_xyxy: tuple[int, int, int, int], boxes: torch.Tensor) -> bool:
    boxes = _as_boxes(boxes)
    if boxes.numel() == 0:
        return False
    vis = box_visibility_in_window(boxes, window_xyxy)
    return bool(((vis > 0.0) & (vis < 1.0)).any().item())


def _as_boxes(boxes: torch.Tensor | None) -> torch.Tensor:
    if boxes is None or boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return boxes.detach().cpu().to(dtype=torch.float32).reshape(-1, 4)


def sample_box_centered_square_window(
    *,
    image_width: int,
    image_height: int,
    box_xyxy: torch.Tensor | list[float] | tuple[float, float, float, float],
    all_mass_boxes: torch.Tensor,
    options: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Sample an n x n crop centered near one specific annotation box.

    This is mainly used by the export pipeline when you want, for example,
    five random positive crops per mass annotation. The sampled crop is shifted
    around the mass center by ``center_shift_fraction * crop_size`` and is
    accepted only if it satisfies the same partial-annotation rules as the rest
    of the crop code.
    """
    opts = make_crop_options(options)
    n = int(opts["crop_size"])
    max_tries = int(opts.get("max_random_tries", 80))
    shift = float(opts.get("center_shift_fraction", 0.25)) * float(n)
    box = torch.as_tensor(box_xyxy, dtype=torch.float32).reshape(4)
    boxes = _as_boxes(all_mass_boxes)

    cx0 = float((box[0] + box[2]) / 2.0)
    cy0 = float((box[1] + box[3]) / 2.0)
    last_window = _random_positive_window(
        image_width=image_width,
        image_height=image_height,
        crop_size=n,
        mass_boxes=box.reshape(1, 4),
        center_on_mass=True,
        center_shift_fraction=float(opts.get("center_shift_fraction", 0.25)),
        rng=rng,
    )
    for _ in range(max_tries):
        cx = cx0 + float(rng.uniform(-shift, shift))
        cy = cy0 + float(rng.uniform(-shift, shift))
        x0 = int(round(cx - n / 2.0))
        y0 = int(round(cy - n / 2.0))
        x0 = min(max(0, x0), max(0, int(image_width) - n))
        y0 = min(max(0, y0), max(0, int(image_height) - n))
        window = (x0, y0, x0 + n, y0 + n)
        last_window = window
        if window_has_positive_mass(window, boxes, opts):
            return window, {"requested_positive": True, "accepted": True, "centered_on_annotation": True}

    return last_window, {
        "requested_positive": True,
        "accepted": False,
        "centered_on_annotation": True,
        "fallback_after_tries": max_tries,
    }
