from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Streamlit is required for the preprocessing inspector. Install with: pip install streamlit"
    ) from exc

from .crops import (
    box_visibility_in_window,
    crop_image_and_boxes_to_window,
    sliding_square_windows,
)
from .dataset import VindrMammoDataset
from .export import load_export_config, make_train_val_test_split


# -----------------------------------------------------------------------------
# Public Streamlit entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the interactive VinDr-Mammo preprocessing inspector."""
    st.set_page_config(page_title="VinDr-Mammo preprocessing inspector", layout="wide")
    st.title("VinDr-Mammo preprocessing inspector")
    st.caption(
        "Inspect raw/preprocessed mammography crops, mass boxes, vendor differences, "
        "and candidate RGB preprocessing pipelines before exporting another dataset."
    )

    config_path = _get_config_path_from_query_or_cli()
    cfg = _load_config_ui(config_path)

    dataset = _load_dataset_from_config(cfg)
    split_records, split_df = _load_split_records(dataset, cfg)
    enriched = _build_enriched_record_table(dataset, split_df)

    mode = st.sidebar.radio("Mode", ["Single image", "Vendor / image comparison"], index=0)
    crop_controls = _crop_controls(cfg)
    show_annotations = st.sidebar.checkbox("Show mass annotations", value=True)
    display_window = st.sidebar.slider(
        "Grayscale display window percentiles", 0.0, 100.0, (1.0, 99.0), 0.5,
        help="Only affects visualization in the GUI, not the underlying DICOM values.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("RGB preprocessing pipeline")
    st.sidebar.caption("Build the output RGB crop channel by channel.")
    pipeline = _pipeline_controls()

    if mode == "Single image":
        _render_single_mode(dataset, enriched, crop_controls, pipeline, show_annotations, display_window)
    else:
        _render_comparison_mode(dataset, enriched, crop_controls, pipeline, show_annotations, display_window)


# -----------------------------------------------------------------------------
# Config and dataset loading
# -----------------------------------------------------------------------------


def _get_config_path_from_query_or_cli() -> Path:
    # Supports both:
    #   streamlit run inspect_preprocessing_app.py -- --config config/export_config.yaml
    # and query parameter ?config=...
    config_from_query = None
    try:
        config_from_query = st.query_params.get("config")
    except Exception:
        config_from_query = None

    if config_from_query:
        return Path(str(config_from_query)).expanduser()

    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            return Path(args[idx + 1]).expanduser()

    return Path.cwd() / "config" / "export_config.yaml"


@st.cache_data(show_spinner=False)
def _load_config_cached(path: str) -> dict[str, Any]:
    return load_export_config(path)


def _load_config_ui(config_path: Path) -> dict[str, Any]:
    st.sidebar.subheader("Configuration")
    cfg_path_text = st.sidebar.text_input("Config YAML", value=str(config_path))
    path = Path(cfg_path_text).expanduser()
    if not path.exists():
        st.error(f"Config file does not exist: {path}")
        st.stop()
    cfg = _load_config_cached(str(path.resolve()))

    data_root_default = str(cfg.get("paths", {}).get("data_root", ""))
    data_root_text = st.sidebar.text_input("VinDr data root", value=data_root_default)
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["data_root"] = data_root_text
    return cfg


@st.cache_resource(show_spinner="Loading VinDr-Mammo metadata...")
def _load_dataset_cached(config_json: str) -> VindrMammoDataset:
    cfg = json.loads(config_json)
    paths = cfg.get("paths", {})
    image_cfg = cfg.get("image", {})
    return VindrMammoDataset(
        data_root=paths.get("data_root"),
        index_level="image",
        split=None,
        read_image=True,
        output_size=None,
        normalize=image_cfg.get("normalize", "none"),
        percentile_range=tuple(image_cfg.get("percentile_range", [0.5, 99.5])),
        use_voi_lut=bool(image_cfg.get("use_voi_lut", False)),
        return_dicom_meta=bool(cfg.get("metadata", {}).get("include_dicom_meta", True)),
        validate_paths=bool(cfg.get("dataset", {}).get("validate_paths", False)),
        preprocess_options=cfg.get("preprocess", {}),
        crop_options={"enabled": False},
        show_progress=False,
    )


def _load_dataset_from_config(cfg: dict[str, Any]) -> VindrMammoDataset:
    # A stable string is used as the cache key. If paths/preprocess settings change,
    # Streamlit reloads the dataset object.
    relevant = {
        "paths": cfg.get("paths", {}),
        "image": cfg.get("image", {}),
        "preprocess": cfg.get("preprocess", {}),
        "metadata": cfg.get("metadata", {}),
        "dataset": cfg.get("dataset", {}),
    }
    return _load_dataset_cached(json.dumps(relevant, sort_keys=True))


@st.cache_data(show_spinner=False)
def _split_df_cached(records_json: str, val_fraction: float, seed: int) -> pd.DataFrame:
    records = json.loads(records_json)
    _, split_df = make_train_val_test_split(records, val_fraction=val_fraction, seed=seed)
    return split_df


def _load_split_records(dataset: VindrMammoDataset, cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    split_cfg = cfg.get("splits", {})
    split_records, split_df = make_train_val_test_split(
        dataset.image_records,
        val_fraction=float(split_cfg.get("val_fraction_from_training", 0.15)),
        seed=int(split_cfg.get("seed", 123)),
    )
    return split_records, split_df


@st.cache_data(show_spinner="Building image filter table...")
def _build_enriched_record_table_cached(records_json: str, metadata_json: str, findings_json: str, split_df_json: str) -> pd.DataFrame:
    records = pd.DataFrame(json.loads(records_json))
    metadata_rows = json.loads(metadata_json)
    findings = json.loads(findings_json)
    # pandas 2.1+ may treat a raw JSON string as a file path.
    # Wrap the JSON literal in StringIO so it is parsed as JSON content.
    split_df = pd.read_json(io.StringIO(split_df_json), orient="split")

    records["image_id"] = records["image_id"].astype(str)
    records["study_id"] = records["study_id"].astype(str)
    split_small = split_df[["image_id", "export_split"]].copy()
    split_small["image_id"] = split_small["image_id"].astype(str)
    records = records.merge(split_small, on="image_id", how="left")

    vendor_map = {}
    meta_preview_map = {}
    for image_id, rows in metadata_rows.items():
        row = rows[0] if rows else {}
        manufacturer = _first_existing(row, ["Manufacturer", "manufacturer", "manufacturers"])
        model = _first_existing(row, ["ManufacturerModelName", "manufacturer_model_name", "model_name", "model"])
        vendor = " / ".join([str(x) for x in [manufacturer, model] if x not in [None, "", "nan"]])
        vendor_map[str(image_id)] = vendor if vendor else "Unknown"
        meta_preview_map[str(image_id)] = row

    records["vendor"] = records["image_id"].map(vendor_map).fillna("Unknown")
    records["has_mass"] = records["image_id"].map(lambda x: bool(findings.get(str(x), [])))
    records["record_index"] = np.arange(len(records), dtype=int)
    records["display_name"] = records.apply(
        lambda r: (
            f"{int(r['record_index']):05d} | {r.get('export_split','?')} | "
            f"{r.get('image_id','')} | {r.get('laterality','')}-{r.get('view_position','')} | "
            f"mass={int(bool(r.get('has_mass', False)))} | {r.get('vendor','Unknown')}"
        ),
        axis=1,
    )
    return records


def _build_enriched_record_table(dataset: VindrMammoDataset, split_df: pd.DataFrame) -> pd.DataFrame:
    metadata_json = json.dumps(dataset.metadata_by_image_id, default=_json_default, sort_keys=True)
    findings = {}
    for image_id, rows in dataset.findings_by_image_id.items():
        mass_rows = [r for r in rows if dataset._is_mass_finding(r)]
        findings[str(image_id)] = mass_rows
    findings_json = json.dumps(findings, default=_json_default, sort_keys=True)
    return _build_enriched_record_table_cached(
        json.dumps(dataset.image_records, default=_json_default, sort_keys=True),
        metadata_json,
        findings_json,
        split_df.to_json(orient="split"),
    )


# -----------------------------------------------------------------------------
# UI controls
# -----------------------------------------------------------------------------


def _crop_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    st.sidebar.divider()
    st.sidebar.subheader("Crop controls")
    crop_cfg = cfg.get("square_crops", {})
    policy = cfg.get("crop_annotation_policy", {})
    crop_size = st.sidebar.number_input("Crop size n", min_value=128, max_value=4096, step=128, value=int(crop_cfg.get("crop_size", 1024)))
    stride = st.sidebar.number_input("Stride", min_value=64, max_value=4096, step=64, value=int(crop_cfg.get("stride", 512)))
    only_mass_crops = st.sidebar.checkbox("Show only crops with visible mass", value=True)
    positivity_threshold = st.sidebar.slider(
        "Positive crop threshold, visible mass fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="A crop is considered positive if at least one mass box has this fraction visible inside the crop.",
    )
    allow_partial = st.sidebar.checkbox("Display partial boxes after clipping", value=bool(policy.get("allow_partial_annotations", False)))
    min_box_visibility = st.sidebar.slider("Minimum box visibility to draw/keep", 0.0, 1.0, float(policy.get("min_box_visibility", 0.30)), 0.05)
    return {
        "crop_size": int(crop_size),
        "stride": int(stride),
        "only_mass_crops": bool(only_mass_crops),
        "positivity_threshold": float(positivity_threshold),
        "crop_options": {
            "enabled": True,
            "mode": "deterministic",
            "crop_size": int(crop_size),
            "stride": int(stride),
            "allow_partial_annotations": bool(allow_partial),
            "min_box_visibility": float(min_box_visibility),
            "reject_partial_windows": not bool(allow_partial),
            "negative_max_box_visibility": 0.0,
            "pad_if_needed": True,
            "pad_value": 0.0,
        },
    }


OP_NAMES = [
    "none",
    "percentile_normalize",
    "percentile_clip_only",
    "zscore_clip",
    "hist_equalize",
    "clahe",
    "gaussian_blur",
    "median_blur",
    "sharpen",
    "unsharp_mask",
    "sobel_gradient",
    "laplacian",
    "gamma",
    "log",
    "invert",
]


def _pipeline_controls() -> dict[str, Any]:
    default_channel_steps = {
        "R": ["percentile_normalize"],
        "G": ["percentile_normalize", "hist_equalize"],
        "B": ["percentile_normalize", "sobel_gradient"],
    }
    pipeline: dict[str, Any] = {}
    for channel in ["R", "G", "B"]:
        with st.sidebar.expander(f"{channel} channel", expanded=(channel == "R")):
            n_steps = st.number_input(f"Number of steps ({channel})", min_value=0, max_value=8, value=len(default_channel_steps[channel]), key=f"{channel}_n_steps")
            steps = []
            for i in range(int(n_steps)):
                default_op = default_channel_steps[channel][i] if i < len(default_channel_steps[channel]) else "none"
                op = st.selectbox(
                    f"Step {i + 1}",
                    OP_NAMES,
                    index=OP_NAMES.index(default_op),
                    key=f"{channel}_op_{i}",
                )
                params = _op_parameter_controls(channel, i, op)
                steps.append({"op": op, "params": params})
            pipeline[channel] = steps
    return pipeline


def _op_parameter_controls(channel: str, step: int, op: str) -> dict[str, Any]:
    prefix = f"{channel}_{step}_{op}"
    if op in {"percentile_normalize", "percentile_clip_only"}:
        lo, hi = st.slider("Percentile window", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"{prefix}_win")
        return {"percentiles": [float(lo), float(hi)]}
    if op == "zscore_clip":
        limit = st.slider("Z-score clip", 0.5, 10.0, 3.0, 0.5, key=f"{prefix}_z")
        return {"z_limit": float(limit)}
    if op == "clahe":
        clip = st.slider("CLAHE clip limit", 0.5, 8.0, 2.0, 0.5, key=f"{prefix}_clip")
        tile = st.select_slider("CLAHE tile size", options=[4, 8, 16, 32], value=8, key=f"{prefix}_tile")
        return {"clip_limit": float(clip), "tile_grid_size": int(tile)}
    if op == "gaussian_blur":
        k = st.select_slider("Gaussian kernel", options=[3, 5, 7, 9, 11, 15, 21], value=5, key=f"{prefix}_k")
        sigma = st.slider("Gaussian sigma", 0.0, 10.0, 1.0, 0.25, key=f"{prefix}_sigma")
        return {"ksize": int(k), "sigma": float(sigma)}
    if op == "median_blur":
        k = st.select_slider("Median kernel", options=[3, 5, 7, 9, 11], value=3, key=f"{prefix}_k")
        return {"ksize": int(k)}
    if op == "sharpen":
        amount = st.slider("Sharpen strength", 0.0, 5.0, 1.0, 0.25, key=f"{prefix}_amt")
        return {"amount": float(amount)}
    if op == "unsharp_mask":
        amount = st.slider("Unsharp amount", 0.0, 5.0, 1.5, 0.25, key=f"{prefix}_amt")
        sigma = st.slider("Unsharp blur sigma", 0.25, 10.0, 2.0, 0.25, key=f"{prefix}_sigma")
        return {"amount": float(amount), "sigma": float(sigma)}
    if op == "sobel_gradient":
        k = st.select_slider("Sobel kernel", options=[1, 3, 5, 7], value=3, key=f"{prefix}_k")
        lo, hi = st.slider("Gradient window", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"{prefix}_win")
        return {"ksize": int(k), "percentiles": [float(lo), float(hi)]}
    if op == "laplacian":
        k = st.select_slider("Laplacian kernel", options=[1, 3, 5, 7], value=3, key=f"{prefix}_k")
        lo, hi = st.slider("Laplacian window", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"{prefix}_win")
        return {"ksize": int(k), "percentiles": [float(lo), float(hi)]}
    if op == "gamma":
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0, 0.1, key=f"{prefix}_gamma")
        return {"gamma": float(gamma)}
    if op == "log":
        gain = st.slider("Log gain", 0.1, 20.0, 5.0, 0.5, key=f"{prefix}_gain")
        return {"gain": float(gain)}
    return {}


# -----------------------------------------------------------------------------
# Single and comparison rendering
# -----------------------------------------------------------------------------


def _render_single_mode(
    dataset: VindrMammoDataset,
    records_df: pd.DataFrame,
    crop_controls: dict[str, Any],
    pipeline: dict[str, Any],
    show_annotations: bool,
    display_window: tuple[float, float],
) -> None:
    filtered = _record_filter_controls(records_df, prefix="single")
    st.subheader("Image selection")
    st.write(f"Filtered images: **{len(filtered)}**")
    if filtered.empty:
        st.warning("No images match the current filters.")
        return

    selected_pos = st.number_input("Image index within filtered list", min_value=0, max_value=max(0, len(filtered) - 1), value=0, step=1)
    selected_row = filtered.iloc[int(selected_pos)]
    result = _prepare_sample(dataset, int(selected_row["record_index"]), crop_controls, crop_index=None)
    crops = result["crops"]
    st.write(f"Crops available after crop filter: **{len(crops)}**")
    if not crops:
        st.warning("This image has no crops under the current crop filter/positivity threshold.")
        return

    crop_idx = st.number_input("Crop index", min_value=0, max_value=max(0, len(crops) - 1), value=0, step=1)
    result = _prepare_sample(dataset, int(selected_row["record_index"]), crop_controls, crop_index=int(crop_idx))
    _show_sample(result, pipeline, show_annotations=show_annotations, display_window=display_window)



def _render_comparison_mode(
    dataset: VindrMammoDataset,
    records_df: pd.DataFrame,
    crop_controls: dict[str, Any],
    pipeline: dict[str, Any],
    show_annotations: bool,
    display_window: tuple[float, float],
) -> None:
    st.subheader("Vendor / image comparison")
    st.caption("All comparison slots use the same crop controls and RGB preprocessing pipeline from the sidebar.")
    n_slots = st.slider("Number of comparison slots", 2, 6, 2)
    results = []
    slot_cols = st.columns(n_slots)
    for slot_idx, col in enumerate(slot_cols):
        with col:
            st.markdown(f"**Slot {slot_idx + 1}**")
            filtered = _record_filter_controls(records_df, prefix=f"cmp_{slot_idx}", compact=True)
            st.caption(f"{len(filtered)} matching images")
            if filtered.empty:
                results.append(None)
                continue
            img_idx = st.number_input("Image idx", 0, max(0, len(filtered) - 1), 0, 1, key=f"cmp_{slot_idx}_imgidx")
            row = filtered.iloc[int(img_idx)]
            tmp = _prepare_sample(dataset, int(row["record_index"]), crop_controls, crop_index=None)
            crop_count = len(tmp["crops"])
            st.caption(f"{crop_count} crops")
            if crop_count == 0:
                results.append(None)
                continue
            cidx = st.number_input("Crop idx", 0, max(0, crop_count - 1), 0, 1, key=f"cmp_{slot_idx}_cropidx")
            results.append(_prepare_sample(dataset, int(row["record_index"]), crop_controls, crop_index=int(cidx)))

    for i, result in enumerate(results):
        if result is None:
            continue
        st.divider()
        st.markdown(f"### Slot {i + 1}: {result['title']}")
        _show_sample(result, pipeline, show_annotations=show_annotations, display_window=display_window, compact=True)



def _record_filter_controls(records_df: pd.DataFrame, *, prefix: str, compact: bool = False) -> pd.DataFrame:
    container = st if not compact else st.container()
    with container:
        split_options = ["all", "train", "val", "test"]
        split_choice = st.selectbox("Split", split_options, index=0, key=f"{prefix}_split")
        positive_choice = st.radio("Images", ["positive only", "all images"], index=0, horizontal=True, key=f"{prefix}_positive")
        vendors = sorted([v for v in records_df["vendor"].dropna().unique().tolist() if str(v).strip()])
        vendor_mode = st.radio("Vendor filter", ["all vendors", "selected vendors"], index=0, horizontal=True, key=f"{prefix}_vendor_mode")
        selected_vendors: list[str] = []
        if vendor_mode == "selected vendors":
            selected_vendors = st.multiselect("Vendors", vendors, default=vendors[:1] if vendors else [], key=f"{prefix}_vendors")

    df = records_df.copy()
    if split_choice != "all":
        df = df[df["export_split"] == split_choice]
    if positive_choice == "positive only":
        df = df[df["has_mass"] == True]  # noqa: E712
    if vendor_mode == "selected vendors" and selected_vendors:
        df = df[df["vendor"].isin(selected_vendors)]
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Sample preparation
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner="Reading and preprocessing DICOM...", max_entries=32, hash_funcs={VindrMammoDataset: lambda _: "dataset"})
def _read_preprocessed_cached(dataset: VindrMammoDataset, dataset_cache_key: str, record_index: int) -> dict[str, Any]:
    _ = dataset_cache_key  # part of the Streamlit cache key
    record = dataset.image_records[int(record_index)]
    image, target = dataset._read_preprocessed_record_no_square(record)
    image_np = image.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
    mass_boxes = target["mass"]["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
    all_boxes = target["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
    meta_records = target.get("metadata", []) or []
    meta_first = meta_records[0] if meta_records else {}
    return {
        "image": image_np,
        "mass_boxes": mass_boxes,
        "all_boxes": all_boxes,
        "record": record,
        "target_summary": {
            "image_id": target.get("image_id"),
            "study_id": target.get("study_id"),
            "split": target.get("split"),
            "laterality": target.get("laterality"),
            "view_position": target.get("view_position"),
            "num_masses": int(target.get("num_masses", 0)),
            "breast_birads": target.get("breast_birads"),
            "breast_density": target.get("breast_density"),
            "preprocessing": target.get("preprocessing", {}),
            "dicom_meta": target.get("dicom_meta", {}),
            "metadata": meta_first,
        },
    }


def _dataset_cache_key(dataset: VindrMammoDataset) -> str:
    return json.dumps({
        "data_root": str(dataset.data_root),
        "normalize": dataset.normalize,
        "percentile_range": list(dataset.percentile_range),
        "use_voi_lut": dataset.use_voi_lut,
        "preprocess_options": dataset.preprocess_options,
    }, sort_keys=True, default=_json_default)


def _prepare_sample(dataset: VindrMammoDataset, record_index: int, crop_controls: dict[str, Any], crop_index: int | None) -> dict[str, Any]:
    loaded = _read_preprocessed_cached(dataset, _dataset_cache_key(dataset), int(record_index))
    image = loaded["image"]
    boxes = torch.as_tensor(loaded["all_boxes"], dtype=torch.float32)
    mass_boxes = torch.as_tensor(loaded["mass_boxes"], dtype=torch.float32)
    height, width = image.shape
    windows = sliding_square_windows(width, height, crop_controls["crop_size"], crop_controls["stride"])

    crops = []
    for w in windows:
        max_vis = 0.0
        if mass_boxes.numel() > 0:
            vis = box_visibility_in_window(mass_boxes, w)
            max_vis = float(vis.max().item()) if vis.numel() > 0 else 0.0
        is_positive = max_vis >= float(crop_controls["positivity_threshold"])
        if crop_controls["only_mass_crops"] and not is_positive:
            continue
        crops.append({"window": w, "max_visibility": max_vis, "positive_by_slider": is_positive})

    selected = None
    crop_image = None
    crop_boxes = np.zeros((0, 4), dtype=np.float32)
    crop_mass_boxes = np.zeros((0, 4), dtype=np.float32)
    if crops:
        selected = crops[int(crop_index or 0) % len(crops)]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        crop_result = crop_image_and_boxes_to_window(
            image_tensor,
            boxes=boxes,
            mass_boxes=mass_boxes,
            window_xyxy=selected["window"],
            options=crop_controls["crop_options"],
        )
        crop_image = crop_result.image.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
        crop_boxes = crop_result.boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        crop_mass_boxes = crop_result.mass_boxes.detach().cpu().numpy().astype(np.float32, copy=False)

    summary = loaded["target_summary"]
    title = (
        f"image_id={summary.get('image_id')} | study={summary.get('study_id')} | "
        f"{summary.get('laterality')}-{summary.get('view_position')} | masses={summary.get('num_masses')} | "
        f"vendor={_vendor_from_summary(summary)}"
    )
    return {
        **loaded,
        "title": title,
        "crops": crops,
        "selected_crop": selected,
        "crop_image": crop_image,
        "crop_boxes": crop_boxes,
        "crop_mass_boxes": crop_mass_boxes,
        "record_index": int(record_index),
    }


# -----------------------------------------------------------------------------
# Display and statistics
# -----------------------------------------------------------------------------


def _show_sample(
    result: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    show_annotations: bool,
    display_window: tuple[float, float],
    compact: bool = False,
) -> None:
    full = result["image"]
    crop = result["crop_image"]
    if crop is None:
        st.warning("No crop selected.")
        return
    full_boxes = result["mass_boxes"] if show_annotations else None
    crop_boxes = result["crop_mass_boxes"] if show_annotations else None
    selected = result["selected_crop"] or {}
    window = selected.get("window")

    processed_rgb, processing_meta = apply_channel_pipeline(crop, pipeline)
    crop_gray = _to_uint8_percentile(crop, display_window)
    full_gray = _to_uint8_percentile(full, display_window)

    # Draw crop window on full image and boxes on all image views.
    full_draw = _draw_boxes(_gray_to_rgb(full_gray), full_boxes, color=(255, 80, 80))
    if window is not None:
        full_draw = _draw_rect(full_draw, window, color=(80, 255, 80), thickness=max(2, full.shape[1] // 1000))
    crop_draw = _draw_boxes(_gray_to_rgb(crop_gray), crop_boxes, color=(255, 80, 80))
    proc_draw = _draw_boxes(processed_rgb.copy(), crop_boxes, color=(255, 80, 80))

    st.write(result["title"])
    if window is not None:
        st.caption(f"Selected crop window xyxy={tuple(int(v) for v in window)} | max mass visibility={selected.get('max_visibility', 0.0):.3f}")

    cols = st.columns(3)
    cols[0].image(full_draw, caption="Original black/white image with selected crop window", use_container_width=True)
    cols[1].image(crop_draw, caption="Original crop", use_container_width=True)
    cols[2].image(proc_draw, caption="Preprocessed RGB crop", use_container_width=True)

    with st.expander("Metadata and statistics", expanded=not compact):
        stat_df = _stats_table(full, crop, processed_rgb)
        st.dataframe(stat_df, use_container_width=True)
        st.json(_compact_metadata(result["target_summary"], processing_meta))
        fig = _histogram_figure(full, crop, processed_rgb)
        st.pyplot(fig, clear_figure=True)



def _stats_table(full: np.ndarray, crop: np.ndarray, processed_rgb: np.ndarray) -> pd.DataFrame:
    rows = []
    rows.append(_stats_row("full_grayscale", full))
    rows.append(_stats_row("crop_grayscale", crop))
    for i, name in enumerate(["R", "G", "B"]):
        rows.append(_stats_row(f"processed_{name}", processed_rgb[..., i]))
    return pd.DataFrame(rows)


def _stats_row(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.array([0.0])
    return {
        "image": name,
        "shape": "x".join(map(str, arr.shape)),
        "dtype": str(arr.dtype),
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _histogram_figure(full: np.ndarray, crop: np.ndarray, processed_rgb: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(_sample_pixels(full), bins=80, alpha=0.35, label="full grayscale")
    ax.hist(_sample_pixels(crop), bins=80, alpha=0.35, label="crop grayscale")
    for i, name in enumerate(["R", "G", "B"]):
        ax.hist(_sample_pixels(processed_rgb[..., i]), bins=80, alpha=0.25, label=f"processed {name}")
    ax.set_title("Pixel intensity distributions")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Channel preprocessing operations
# -----------------------------------------------------------------------------


def apply_channel_pipeline(crop_float: np.ndarray, pipeline: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    channels = []
    meta = {"channels": {}}
    for channel in ["R", "G", "B"]:
        arr = np.asarray(crop_float, dtype=np.float32).copy()
        applied = []
        for step in pipeline.get(channel, []):
            op = step.get("op", "none")
            params = step.get("params", {}) or {}
            if op == "none":
                continue
            arr = _apply_operation(arr, op, params)
            applied.append({"op": op, "params": params})
        ch = _float_to_uint8(arr)
        channels.append(ch)
        meta["channels"][channel] = applied
    return np.stack(channels, axis=-1).astype(np.uint8, copy=False), meta


def _apply_operation(arr: np.ndarray, op: str, params: dict[str, Any]) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if op == "percentile_normalize":
        return _normalize_percentile(arr, params.get("percentiles", [1.0, 99.0]))
    if op == "percentile_clip_only":
        lo, hi = _safe_percentile(arr, params.get("percentiles", [1.0, 99.0]))
        return np.clip(arr, lo, hi).astype(np.float32)
    if op == "zscore_clip":
        m = float(np.mean(arr))
        s = float(np.std(arr)) or 1.0
        z = (arr - m) / max(s, 1e-12)
        limit = float(params.get("z_limit", 3.0))
        z = np.clip(z, -limit, limit)
        return ((z + limit) / max(2 * limit, 1e-12)).astype(np.float32)
    if op == "hist_equalize":
        return _equalize(_float_to_uint8(arr)).astype(np.float32) / 255.0
    if op == "clahe":
        return _clahe(_float_to_uint8(arr), params).astype(np.float32) / 255.0
    if op == "gaussian_blur":
        if cv2 is None:
            return arr
        k = _odd_int(params.get("ksize", 5))
        sigma = float(params.get("sigma", 1.0))
        return cv2.GaussianBlur(arr.astype(np.float32), (k, k), sigmaX=sigma).astype(np.float32)
    if op == "median_blur":
        if cv2 is None:
            return arr
        k = _odd_int(params.get("ksize", 3))
        return cv2.medianBlur(_float_to_uint8(arr), k).astype(np.float32) / 255.0
    if op == "sharpen":
        if cv2 is None:
            return arr
        amount = float(params.get("amount", 1.0))
        kernel = np.array([[0, -1, 0], [-1, 4 + amount, -1], [0, -1, 0]], dtype=np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(arr.astype(np.float32), -1, kernel).astype(np.float32)
    if op == "unsharp_mask":
        if cv2 is None:
            return arr
        amount = float(params.get("amount", 1.5))
        sigma = float(params.get("sigma", 2.0))
        blurred = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma)
        return (arr + amount * (arr - blurred)).astype(np.float32)
    if op == "sobel_gradient":
        return _sobel(arr, params)
    if op == "laplacian":
        return _laplacian(arr, params)
    if op == "gamma":
        gamma = max(float(params.get("gamma", 1.0)), 1e-6)
        return np.power(np.clip(_normalize_minmax(arr), 0.0, 1.0), gamma).astype(np.float32)
    if op == "log":
        gain = float(params.get("gain", 5.0))
        x = np.clip(_normalize_minmax(arr), 0.0, 1.0)
        return (np.log1p(gain * x) / np.log1p(gain)).astype(np.float32)
    if op == "invert":
        return 1.0 - _normalize_minmax(arr)
    return arr


def _sobel(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if cv2 is None:
        gy, gx = np.gradient(arr.astype(np.float32))
        mag = np.sqrt(gx * gx + gy * gy)
    else:
        u8 = _float_to_uint8(arr)
        k = _odd_int(params.get("ksize", 3))
        gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=k)
        mag = cv2.magnitude(gx, gy)
    return _normalize_percentile(mag.astype(np.float32), params.get("percentiles", [1.0, 99.0]))


def _laplacian(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if cv2 is None:
        gy, gx = np.gradient(arr.astype(np.float32))
        gyy, _ = np.gradient(gy)
        _, gxx = np.gradient(gx)
        lap = np.abs(gxx + gyy)
    else:
        u8 = _float_to_uint8(arr)
        k = _odd_int(params.get("ksize", 3))
        lap = np.abs(cv2.Laplacian(u8, cv2.CV_32F, ksize=k))
    return _normalize_percentile(lap.astype(np.float32), params.get("percentiles", [1.0, 99.0]))


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _to_uint8_percentile(arr: np.ndarray, percentiles: tuple[float, float] | list[float]) -> np.ndarray:
    lo, hi = _safe_percentile(arr, percentiles)
    return _to_uint8(arr, lo, hi)


def _to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (np.clip(arr, lo, hi) - lo) / max(float(hi - lo), 1e-12)
    return np.round(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 1.0:
        return np.round(np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return _to_uint8_percentile(arr, [1.0, 99.0])


def _safe_percentile(arr: np.ndarray, percentiles: list[float] | tuple[float, float]) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [float(percentiles[0]), float(percentiles[1])])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize_percentile(arr: np.ndarray, percentiles: list[float] | tuple[float, float]) -> np.ndarray:
    lo, hi = _safe_percentile(arr, percentiles)
    return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _equalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8, copy=False)
    if cv2 is not None:
        return cv2.equalizeHist(img)
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    valid = cdf > 0
    if not valid.any():
        return img.copy()
    lut = np.round((cdf - cdf[valid][0]) / max(cdf[-1] - cdf[valid][0], 1.0) * 255).clip(0, 255).astype(np.uint8)
    return lut[img]


def _clahe(img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    img = img.astype(np.uint8, copy=False)
    if cv2 is None:
        return _equalize(img)
    clip_limit = float(params.get("clip_limit", 2.0))
    tile = int(params.get("tile_grid_size", 8))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(img)


def _odd_int(value: Any) -> int:
    k = int(value)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _draw_rect(img: np.ndarray, box: tuple[int, int, int, int], *, color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if cv2 is not None:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    else:
        out[max(0, y0):max(0, y0 + thickness), max(0, x0):x1] = color
        out[max(0, y1 - thickness):y1, max(0, x0):x1] = color
        out[max(0, y0):y1, max(0, x0):max(0, x0 + thickness)] = color
        out[max(0, y0):y1, max(0, x1 - thickness):x1] = color
    return out


def _draw_boxes(img: np.ndarray, boxes: np.ndarray | None, *, color=(255, 0, 0)) -> np.ndarray:
    out = img.copy()
    if boxes is None:
        return out
    for box in np.asarray(boxes).reshape(-1, 4):
        out = _draw_rect(out, tuple(int(round(v)) for v in box), color=color, thickness=2)
    return out


def _gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return np.repeat(img[..., None], 3, axis=-1).astype(np.uint8, copy=False)


def _sample_pixels(arr: np.ndarray, max_pixels: int = 200_000) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size > max_pixels:
        rng = np.random.default_rng(123)
        flat = rng.choice(flat, size=max_pixels, replace=False)
    return flat


def _compact_metadata(target_summary: dict[str, Any], processing_meta: dict[str, Any]) -> dict[str, Any]:
    metadata = target_summary.get("metadata", {}) or {}
    dicom_meta = target_summary.get("dicom_meta", {}) or {}
    return {
        "image": {k: target_summary.get(k) for k in ["image_id", "study_id", "split", "laterality", "view_position", "num_masses", "breast_birads", "breast_density"]},
        "vendor": _vendor_from_summary(target_summary),
        "dicom_meta": {k: dicom_meta.get(k) for k in ["Manufacturer", "ManufacturerModelName", "PhotometricInterpretation", "Rows", "Columns", "PixelSpacing", "ViewPosition", "ImageLaterality"] if k in dicom_meta},
        "metadata_csv": {k: metadata.get(k) for k in list(metadata)[:20]},
        "fixed_geometry_preprocessing": target_summary.get("preprocessing", {}),
        "rgb_pipeline": processing_meta,
    }


def _vendor_from_summary(summary: dict[str, Any]) -> str:
    meta = summary.get("metadata", {}) or {}
    dicom = summary.get("dicom_meta", {}) or {}
    manufacturer = _first_existing(meta, ["Manufacturer", "manufacturer"])
    if manufacturer in [None, "", "nan"]:
        manufacturer = _first_existing(dicom, ["Manufacturer"])
    model = _first_existing(meta, ["ManufacturerModelName", "manufacturer_model_name", "model_name", "model"])
    if model in [None, "", "nan"]:
        model = _first_existing(dicom, ["ManufacturerModelName"])
    vendor = " / ".join([str(x) for x in [manufacturer, model] if x not in [None, "", "nan"]])
    return vendor if vendor else "Unknown"


def _first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in [None, "", "nan"]:
            return row[key]
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


if __name__ == "__main__":
    main()
