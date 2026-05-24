from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
    if out["mode"] not in {"random", "deterministic"}:
        raise ValueError("crop_options['mode'] must be 'random' or 'deterministic'.")
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
