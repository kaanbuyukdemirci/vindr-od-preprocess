from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np


WHOLE_IMAGE_SCOPES = {
    "whole",
    "whole_image",
    "whole-image",
    "whole_image_before_crop",
    "before_crop",
    "pre_crop",
    "pre-crop",
}


def step_applies_before_crop(step: dict[str, Any] | None) -> bool:
    """Return whether a custom channel step runs before square cropping.

    Existing YAML files have no scope field and therefore remain crop-local.
    Both the concise boolean written by the GUI and a readable ``scope`` value
    are accepted so hand-authored configurations remain easy to understand.
    """
    if not isinstance(step, dict):
        return False
    if "apply_before_crop" in step:
        value = step.get("apply_before_crop")
        if isinstance(value, str):
            return value.strip().casefold() in {"1", "true", "yes", "on", "enabled"}
        return bool(value)
    scope = str(step.get("scope", "crop") or "crop").strip().casefold()
    return scope in WHOLE_IMAGE_SCOPES


def split_scoped_steps(steps: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        (before if step_applies_before_crop(step) else after).append(step)
    return before, after


def crop_array_to_window(
    array: np.ndarray,
    window_xyxy: tuple[int, int, int, int],
    *,
    output_height: int | None = None,
    output_width: int | None = None,
    pad_value: float | bool = 0.0,
) -> np.ndarray:
    """Crop a 2-D array to an arbitrary window and pad out-of-image pixels.

    This intentionally mirrors ``crop_image_and_boxes_to_window`` without
    requiring a Torch round-trip. It is used after whole-image photometric
    preprocessing so the exact normal stride window can extend past an edge.
    """
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}.")
    x0, y0, x1, y1 = (int(v) for v in window_xyxy)
    wanted_h = int(output_height if output_height is not None else y1 - y0)
    wanted_w = int(output_width if output_width is not None else x1 - x0)
    if wanted_h <= 0 or wanted_w <= 0:
        raise ValueError("Crop output dimensions must be positive.")

    src_h, src_w = arr.shape
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(src_w, x1), min(src_h, y1)
    out = np.full((wanted_h, wanted_w), pad_value, dtype=arr.dtype)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = max(0, -x0)
    dst_y0 = max(0, -y0)
    copy_w = min(src_x1 - src_x0, wanted_w - dst_x0)
    copy_h = min(src_y1 - src_y0, wanted_h - dst_y0)
    if copy_w > 0 and copy_h > 0:
        out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = arr[
            src_y0:src_y0 + copy_h,
            src_x0:src_x0 + copy_w,
        ]
    return out


def scoped_steps_cache_key(
    namespace: str,
    source_name: str,
    steps: list[dict[str, Any]],
) -> str:
    payload = json.dumps(steps, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}:{source_name}:{digest}"


def apply_scoped_steps(
    crop_source: np.ndarray,
    steps: list[dict[str, Any]] | None,
    *,
    apply_operation: Callable[[np.ndarray, str, dict[str, Any], np.ndarray | None], np.ndarray],
    make_stat_mask: Callable[[np.ndarray], np.ndarray],
    operation_preserves_background: Callable[[str], bool],
    full_source: np.ndarray | None = None,
    full_stat_mask: np.ndarray | None = None,
    window_xyxy: tuple[int, int, int, int] | None = None,
    pad_value: float = 0.0,
    whole_stage_cache: MutableMapping[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
    cache_namespace: str = "",
    source_name: str = "current_crop",
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Run whole-image-scoped steps, crop, then crop-scoped steps.

    The two phases are explicit: all checked steps retain their relative order
    and run on the fixed-preprocessed whole breast; the selected window is then
    extracted (with padding); finally all unchecked steps retain their relative
    order and run on that crop. If a caller cannot provide a whole source/window,
    checked steps fall back to the crop and the metadata records that fallback.
    """
    crop_arr = np.nan_to_num(np.asarray(crop_source, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    before, after = split_scoped_steps(steps)
    used_whole = bool(before and full_source is not None and window_xyxy is not None)
    whole_fallback = bool(before and not used_whole)
    applied: list[dict[str, Any]] = []

    if used_whole:
        supplied_full_mask: np.ndarray | None = None
        if full_stat_mask is not None:
            supplied_full_mask = np.asarray(full_stat_mask, dtype=bool)
            expected_shape = tuple(np.asarray(full_source).shape)
            if tuple(supplied_full_mask.shape) != expected_shape:
                raise ValueError(
                    "Whole-image statistics mask shape does not match its source: "
                    f"mask={tuple(supplied_full_mask.shape)}, source={expected_shape}."
                )
        cache_key = scoped_steps_cache_key(cache_namespace, source_name, before)
        cached = whole_stage_cache.get(cache_key) if whole_stage_cache is not None else None
        if cached is not None:
            full_work = np.asarray(cached[0], dtype=np.float32)
            full_mask = None if cached[1] is None else np.asarray(cached[1], dtype=bool)
            cache_hit = True
        else:
            full_work = np.nan_to_num(np.asarray(full_source, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
            # Fixed preprocessing already computed the authoritative breast
            # mask. Reuse it when supplied instead of re-estimating foreground
            # from a tightly breast-cropped image whose borders may be tissue.
            full_mask = supplied_full_mask if supplied_full_mask is not None else make_stat_mask(full_work)
            for step in before:
                op = str(step.get("op", "none") or "none").casefold().strip()
                params = dict(step.get("params", {}) or {})
                if op in {"", "none", "null"}:
                    continue
                full_work = apply_operation(full_work, op, params, full_mask)
                if full_mask is not None and full_mask.shape != full_work.shape:
                    full_mask = make_stat_mask(full_work)
                elif operation_preserves_background(op) and full_mask is not None and full_mask.any():
                    full_work = np.where(full_mask, full_work, float(params.get("outside_value", 0.0))).astype(np.float32)
            if whole_stage_cache is not None:
                whole_stage_cache[cache_key] = (
                    np.asarray(full_work, dtype=np.float32),
                    None if full_mask is None else np.asarray(full_mask, dtype=bool),
                )
            cache_hit = False

        work = crop_array_to_window(
            full_work,
            window_xyxy,
            output_height=int(crop_arr.shape[0]),
            output_width=int(crop_arr.shape[1]),
            pad_value=float(pad_value),
        ).astype(np.float32, copy=False)
        stat_mask = None
        if full_mask is not None:
            stat_mask = crop_array_to_window(
                np.asarray(full_mask, dtype=bool),
                window_xyxy,
                output_height=int(crop_arr.shape[0]),
                output_width=int(crop_arr.shape[1]),
                pad_value=False,
            ).astype(bool, copy=False)
        for step in before:
            op = str(step.get("op", "none") or "none").casefold().strip()
            if op not in {"", "none", "null"}:
                applied.append({"op": op, "params": dict(step.get("params", {}) or {}), "apply_before_crop": True})
    else:
        work = crop_arr.copy()
        stat_mask = make_stat_mask(work)
        cache_hit = False
        # A robust fallback keeps hand-authored YAML usable in whole-image view
        # modes where the crop and the whole source are already identical.
        for step in before:
            op = str(step.get("op", "none") or "none").casefold().strip()
            params = dict(step.get("params", {}) or {})
            if op in {"", "none", "null"}:
                continue
            work = apply_operation(work, op, params, stat_mask)
            if stat_mask is not None and stat_mask.shape != work.shape:
                stat_mask = make_stat_mask(work)
            elif operation_preserves_background(op) and stat_mask is not None and stat_mask.any():
                work = np.where(stat_mask, work, float(params.get("outside_value", 0.0))).astype(np.float32)
            applied.append({"op": op, "params": params, "apply_before_crop": True})

    for step in after:
        op = str(step.get("op", "none") or "none").casefold().strip()
        params = dict(step.get("params", {}) or {})
        if op in {"", "none", "null"}:
            continue
        work = apply_operation(work, op, params, stat_mask)
        if stat_mask is not None and stat_mask.shape != work.shape:
            stat_mask = make_stat_mask(work)
        elif operation_preserves_background(op) and stat_mask is not None and stat_mask.any():
            work = np.where(stat_mask, work, float(params.get("outside_value", 0.0))).astype(np.float32)
        applied.append({"op": op, "params": params, "apply_before_crop": False})

    return work.astype(np.float32, copy=False), applied, {
        "whole_image_steps_requested": int(bool(before)),
        "whole_image_steps_applied": int(used_whole),
        "whole_image_supplied_stat_mask_used": int(
            used_whole and full_stat_mask is not None
        ),
        "whole_image_steps_fell_back_to_crop": int(whole_fallback),
        "whole_stage_cache_hit": int(cache_hit),
        "scope_execution_order": "whole_image_steps_then_crop_then_crop_steps",
    }
