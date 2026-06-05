from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

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
    sample_random_square_window,
    sliding_square_windows,
)
from .dataset import VindrMammoDataset
from .export import export_from_config, load_export_config, make_train_val_test_split
from .visualize import create_visualizations_from_export


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
    cfg = _apply_loaded_manifest_settings_if_any(cfg)
    cfg = _global_preprocess_controls(cfg)

    dataset = _load_dataset_from_config(cfg)
    split_records, split_df = _load_split_records(dataset, cfg)
    enriched = _build_enriched_record_table(dataset, split_df)

    mode = st.sidebar.radio(
        "Mode",
        [
            "Single image",
            "Vendor / image comparison",
            "Dataset visualizations",
            "Manifest comparison / load settings",
        ],
        index=0,
    )
    crop_controls = _crop_controls(cfg)
    show_annotations = st.sidebar.checkbox("Show mass annotations", value=True)
    display_window = st.sidebar.slider(
        "Grayscale display window percentiles", 0.0, 100.0, (1.0, 99.0), 0.5,
        help="Only affects visualization in the GUI, not the underlying DICOM values.",
    )
    display_controls = _display_controls()

    st.sidebar.divider()
    st.sidebar.subheader("RGB preprocessing pipeline")
    st.sidebar.caption("Build the output RGB crop channel by channel.")
    pipeline = _pipeline_controls(cfg)
    _loaded_pipeline_debug_panel(cfg, pipeline)
    _export_current_preprocessing_yaml_panel(
        config_path=config_path,
        cfg=cfg,
        crop_controls=crop_controls,
        display_controls=display_controls,
        pipeline=pipeline,
    )
    _export_dataset_from_gui_panel(
        cfg=cfg,
        records_df=enriched,
        crop_controls=crop_controls,
        pipeline=pipeline,
    )

    if mode == "Single image":
        _render_single_mode(dataset, enriched, crop_controls, pipeline, show_annotations, display_window, display_controls)
    elif mode == "Vendor / image comparison":
        _render_comparison_mode(dataset, enriched, crop_controls, pipeline, show_annotations, display_window, display_controls)
    elif mode == "Dataset visualizations":
        _render_dataset_visualization_mode(cfg)
    else:
        _render_manifest_comparison_mode(cfg)



# -----------------------------------------------------------------------------
# Dataset visualization mode
# -----------------------------------------------------------------------------


def _render_dataset_visualization_mode(cfg: dict[str, Any]) -> None:
    """Create and view exported-dataset visualizations from an arbitrary path."""
    st.subheader("Dataset visualizations from path")
    st.caption(
        "Enter an already exported dataset path, then calculate the same fast visualizations "
        "that visualize_export.py creates. This reads CSV/COCO JSON files only. It does not "
        "read DICOMs and it does not regenerate crops."
    )

    default_root = Path(str(cfg.get("paths", {}).get("output_root", "/mnt/t9/preprocessed-vindr-v3")))
    path_text = st.text_input(
        "Exported dataset path",
        value=str(default_root),
        help=(
            "Usually this is the export root, for example /mnt/t9/preprocessed-vindr-v3. "
            "You may also point directly to square_crops or baseline_uncropped."
        ),
        key="viz_mode_dataset_path",
    )
    input_path = Path(path_text).expanduser()

    inferred = _infer_visualization_paths(input_path)
    output_root = inferred["output_root"]
    include_square = inferred["include_square_crops"]
    include_baseline = inferred["include_baseline_uncropped"]

    info_cols = st.columns(3)
    info_cols[0].metric("Resolved export root", str(output_root))
    info_cols[1].metric("Use square_crops", "yes" if include_square else "no")
    info_cols[2].metric("Use baseline", "yes" if include_baseline else "no")

    with st.expander("Visualization options", expanded=True):
        output_dir_default = output_root / "visualizations"
        output_dir_text = st.text_input(
            "Visualization output folder",
            value=str(output_dir_default),
            key="viz_mode_output_dir",
        )
        option_cols = st.columns(4)
        include_square = option_cols[0].checkbox("Include square_crops", value=bool(include_square), key="viz_mode_include_square")
        include_baseline = option_cols[1].checkbox("Include baseline_uncropped", value=bool(include_baseline), key="viz_mode_include_baseline")
        write_html = option_cols[2].checkbox("Write HTML report", value=True, key="viz_mode_write_html")
        use_row_limit = option_cols[3].checkbox("Limit samples.csv rows", value=False, key="viz_mode_use_row_limit")
        max_rows = None
        if use_row_limit:
            max_rows = st.number_input("Max rows per samples.csv", min_value=100, value=5000, step=1000, key="viz_mode_max_rows")

    if not output_root.exists():
        st.error(f"Path does not exist: {output_root}")
        return

    run_cols = st.columns([1.2, 2.0])
    run_now = run_cols[0].button("Calculate / refresh visualizations", type="primary", use_container_width=True)
    run_cols[1].caption(
        "This creates CSV summaries and PNG plots under the visualization output folder, "
        "then displays the main COCO-size statistics below."
    )

    result = None
    if run_now:
        progress = st.progress(0.0, text="Starting visualization calculation")
        try:
            progress.progress(0.15, text="Reading exported CSV and COCO JSON files")
            with st.spinner("Calculating visualizations from exported dataset files..."):
                result = create_visualizations_from_export(
                    output_root=output_root,
                    output_dir=Path(output_dir_text).expanduser(),
                    include_square_crops=bool(include_square),
                    include_baseline=bool(include_baseline),
                    write_html_report=bool(write_html),
                    max_rows_per_samples_csv=max_rows,
                )
            progress.progress(1.0, text="Visualization calculation complete")
            st.success(f"Created {len(result.created_files)} files in {result.output_dir}")
        except Exception as exc:
            progress.progress(1.0, text="Visualization calculation failed")
            st.exception(exc)
            return

    output_dir = Path(output_dir_text).expanduser()
    _show_visualization_outputs(output_dir, result=result)


def _infer_visualization_paths(path: Path) -> dict[str, Any]:
    """Accept either an export root or a direct dataset subfolder."""
    name = path.name
    if name == "square_crops":
        return {"output_root": path.parent, "include_square_crops": True, "include_baseline_uncropped": False}
    if name == "baseline_uncropped":
        return {"output_root": path.parent, "include_square_crops": False, "include_baseline_uncropped": True}
    return {
        "output_root": path,
        "include_square_crops": (path / "square_crops").exists(),
        "include_baseline_uncropped": (path / "baseline_uncropped").exists(),
    }


def _show_visualization_outputs(output_dir: Path, *, result: Any | None = None) -> None:
    """Show generated tables and figures directly in Streamlit."""
    st.divider()
    st.markdown("### Visualization results")
    if result is not None:
        safe_summary = _streamlit_json_safe(getattr(result, "summary", {}) or {})
        with st.expander("Visualization run summary", expanded=False):
            st.json(safe_summary)

    if not output_dir.exists():
        st.info(f"No visualization output folder found yet: {output_dir}")
        return

    st.write(f"Output folder: `{output_dir}`")
    html_path = output_dir / "index.html"
    if html_path.exists():
        st.write(f"HTML report: `{html_path}`")

    table_specs = [
        ("coco_box_size_stats.csv", "COCO small / medium / large box size statistics"),
        ("coco_box_annotations.csv", "Per-box COCO annotation table"),
        ("combined_summary.csv", "Combined image-level summary"),
        ("sanity_report.json", "Sanity report"),
    ]
    for filename, title in table_specs:
        path = output_dir / filename
        if not path.exists():
            continue
        with st.expander(title, expanded=(filename == "coco_box_size_stats.csv")):
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button(
                        f"Download {filename}",
                        data=path.read_bytes(),
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True,
                        key=f"download_{filename}",
                    )
                else:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    st.json(obj)
            except Exception as exc:
                st.warning(f"Could not display {path}: {exc}")

    st.markdown("### COCO-size plots")
    coco_plot_files = [
        "20_coco_box_size_counts.png",
        "21_coco_box_size_percentages.png",
        "22_coco_sqrt_box_area_hist.png",
        "23_coco_box_width_height_scatter.png",
    ]
    available_plots = [output_dir / name for name in coco_plot_files if (output_dir / name).exists()]
    if available_plots:
        cols = st.columns(2)
        for i, path in enumerate(available_plots):
            with cols[i % 2]:
                st.image(str(path), caption=path.name, use_container_width=True)
    else:
        st.info("No COCO-size plot PNGs found yet. Click Calculate / refresh visualizations.")

    with st.expander("All generated PNG plots", expanded=False):
        pngs = sorted(output_dir.glob("*.png"))
        if not pngs:
            st.write("No PNG plots found.")
        else:
            cols = st.columns(2)
            for i, path in enumerate(pngs):
                with cols[i % 2]:
                    st.image(str(path), caption=path.name, use_container_width=True)



# -----------------------------------------------------------------------------
# Manifest comparison and settings loading
# -----------------------------------------------------------------------------


def _apply_loaded_manifest_settings_if_any(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply a loaded manifest/config snapshot to the active GUI config.

    The loaded config must be treated as the source of truth. Widget state in
    Streamlit can survive reruns, so this function also exposes an explicit
    refresh button that rebuilds all config-backed widgets from the loaded
    snapshot.
    """
    raw_loaded = st.session_state.get("loaded_manifest_config_snapshot")
    if not isinstance(raw_loaded, dict):
        return cfg

    keep_paths = bool(st.session_state.get("loaded_manifest_keep_current_paths", False))
    effective_loaded = st.session_state.get("loaded_manifest_effective_config_snapshot")
    if not isinstance(effective_loaded, dict):
        effective_loaded = copy.deepcopy(raw_loaded)
        if keep_paths:
            effective_loaded["paths"] = copy.deepcopy(cfg.get("paths", {}) or {})
        st.session_state["loaded_manifest_effective_config_snapshot"] = copy.deepcopy(effective_loaded)

    # Merge into the base config so missing newly-added keys still have defaults,
    # but loaded values always win for export-relevant settings.
    merged = _deep_merge_dict(copy.deepcopy(cfg), copy.deepcopy(effective_loaded))

    with st.sidebar.expander("Loaded manifest settings", expanded=False):
        source = st.session_state.get("loaded_manifest_source", "manifest")
        st.success(f"Using settings loaded from: {source}")
        if keep_paths:
            st.caption("Current config paths were kept. All other preprocessing, crop, vendor, annotation, and RGB settings were loaded.")
        else:
            st.caption("Full config snapshot was loaded, including paths.")
        st.caption(
            "If a control still looks stale, click Refresh loaded settings. This clears the relevant Streamlit widget state "
            "and rebuilds all controls from the loaded config."
        )
        refresh_col, clear_col = st.columns(2)
        with refresh_col:
            if st.button("Refresh loaded settings", key="refresh_loaded_manifest_settings", use_container_width=True):
                _refresh_loaded_config_widget_state(effective_loaded)
                st.rerun()
        with clear_col:
            if st.button("Clear loaded settings", key="clear_loaded_manifest_settings", use_container_width=True):
                for key in [
                    "loaded_manifest_config_snapshot",
                    "loaded_manifest_effective_config_snapshot",
                    "loaded_manifest_source",
                    "loaded_manifest_keep_current_paths",
                    "loaded_manifest_widget_token",
                    "loaded_manifest_refresh_counter",
                ]:
                    st.session_state.pop(key, None)
                _clear_relevant_gui_widget_state()
                st.rerun()
    return merged


def _refresh_loaded_config_widget_state(effective_config: dict[str, Any] | None = None) -> None:
    """Force config-backed widgets to rebuild from the loaded config.

    This fixes the common Streamlit issue where a widget keeps an old value
    because it has the same key after a rerun. We intentionally change the
    token even if the loaded manifest is the same file.
    """
    if effective_config is None or not isinstance(effective_config, dict):
        effective_config = st.session_state.get("loaded_manifest_effective_config_snapshot")
    if not isinstance(effective_config, dict):
        effective_config = st.session_state.get("loaded_manifest_config_snapshot")
    if not isinstance(effective_config, dict):
        return
    counter = int(st.session_state.get("loaded_manifest_refresh_counter", 0) or 0) + 1
    st.session_state["loaded_manifest_refresh_counter"] = counter
    st.session_state["loaded_manifest_widget_token"] = f"{_config_widget_token(effective_config)}_{counter}"
    _clear_relevant_gui_widget_state()


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge_dict(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _config_widget_token(cfg: dict[str, Any]) -> str:
    """Stable short token used to force Streamlit widgets to rebuild after loading settings."""
    try:
        text = json.dumps(_make_yaml_safe(cfg), sort_keys=True, default=str)
    except Exception:
        text = repr(cfg)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _active_widget_suffix() -> str:
    """Return a suffix for widget keys. Changes when manifest/config settings are loaded."""
    token = st.session_state.get("loaded_manifest_widget_token")
    if token:
        return f"__loaded_{token}"
    return "__base"


def _is_loaded_config_active() -> bool:
    """True when the GUI is being driven by a loaded manifest/config snapshot."""
    return isinstance(st.session_state.get("loaded_manifest_config_snapshot"), dict)


def _widget_key(base: str) -> str:
    """Widget key that changes whenever a manifest/config is loaded.

    Streamlit widget state persists across reruns. If keys are reused after
    loading a manifest, old widget values can override the loaded config.
    Every config-backed widget should use this helper.
    """
    return f"{base}{_active_widget_suffix()}"


def _clear_relevant_gui_widget_state() -> None:
    prefixes = [
        "R_", "G_", "B_",
        "gui_export_", "fixed_preprocess_", "gui_display_",
        "crop_", "crop_size", "crop_stride", "crop_proposal",
        "refresh_preview_",
    ]
    # Do not clear manifest_compare_paths or upload widgets here, because users
    # often refresh while staying on the manifest page. The loaded config token
    # already forces config-backed widgets to be recreated.
    exact_keys = {
        "Visible RGB channels", "Show individual processed channels",
    }
    for key in list(st.session_state.keys()):
        key_s = str(key)
        if key in exact_keys or any(key_s.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def _render_manifest_comparison_mode(cfg: dict[str, Any]) -> None:
    st.subheader("Manifest comparison and settings loading")
    st.caption(
        "Enter exported dataset folders or manifest paths. The app will look for manifest.json first, then export_summary.json. "
        "It explains the first manifest, then explains each next manifest as a change from the previous one."
    )

    default_paths = "\n".join([
        str(cfg.get("paths", {}).get("output_root", "/mnt/t9/preprocessed-vindr-v3")),
    ])
    path_text = st.text_area(
        "Dataset directories or manifest/export_summary paths, one per line",
        value=st.session_state.get("manifest_compare_paths", default_paths),
        height=140,
        key="manifest_compare_paths",
        help="Examples: /mnt/t9/preprocessed-vindr-v5 or /mnt/t9/preprocessed-vindr-v5/manifest.json",
    )
    uploaded = st.file_uploader(
        "Optional: upload manifest/export_summary JSON files",
        type=["json", "txt"],
        accept_multiple_files=True,
        key="manifest_compare_uploads",
    )

    if st.button("Read and compare manifests", type="primary", key="manifest_compare_run"):
        st.session_state["manifest_compare_results"] = _load_manifests_from_inputs(path_text, uploaded)

    results = st.session_state.get("manifest_compare_results")
    if not results:
        st.info("Add at least one dataset directory or manifest path, then click Read and compare manifests.")
        return

    valid = [r for r in results if r.get("ok")]
    invalid = [r for r in results if not r.get("ok")]
    if invalid:
        with st.expander("Paths/files that could not be read", expanded=True):
            for item in invalid:
                st.error(f"{item.get('source')}: {item.get('error')}")
    if not valid:
        return

    summaries = [_manifest_summary_row(item["manifest"], item["source"]) for item in valid]
    overview_df = pd.DataFrame(summaries)
    st.markdown("### Overview")
    st.dataframe(overview_df, hide_index=True, use_container_width=True)

    st.markdown("### Step-by-step explanation")
    for idx, item in enumerate(valid):
        manifest = item["manifest"]
        title = _manifest_short_name(manifest, item["source"])
        with st.expander(f"{idx + 1}. {title}", expanded=True):
            if idx == 0:
                st.markdown(_explain_manifest_baseline(manifest))
            else:
                previous = valid[idx - 1]["manifest"]
                st.markdown(_explain_manifest_change(previous, manifest))
            _render_manifest_key_settings(manifest)

    if len(valid) >= 2:
        st.markdown("### Pairwise differences")
        diff_rows = []
        for idx in range(1, len(valid)):
            diff_rows.extend(_manifest_diff_rows(valid[idx - 1]["manifest"], valid[idx]["manifest"], idx))
        diff_df = pd.DataFrame(diff_rows)
        st.dataframe(diff_df, hide_index=True, use_container_width=True)

        csv_data = diff_df.to_csv(index=False)
        st.download_button(
            "Download pairwise difference CSV",
            data=csv_data,
            file_name="manifest_pairwise_differences.csv",
            mime="text/csv",
            key="manifest_diff_download",
        )

    st.markdown("### Load settings from a manifest")
    options = [f"{i + 1}. {_manifest_short_name(item['manifest'], item['source'])}" for i, item in enumerate(valid)]
    selected_label = st.selectbox("Manifest to load", options, key="manifest_load_select")
    selected_index = options.index(selected_label)
    keep_paths = st.checkbox(
        "Keep current data/output paths instead of manifest paths",
        value=False,
        key="manifest_load_keep_paths",
        help=(
            "For strict replay, leave this unchecked so paths are loaded too. "
            "Check it only when you want the same settings but a different data/output location."
        ),
    )
    load_cols = st.columns(2)
    with load_cols[0]:
        if st.button("Load settings into GUI session", type="primary", key="manifest_load_settings"):
            selected = valid[selected_index]
            snapshot = selected["manifest"].get("config_snapshot") or {}
            if not isinstance(snapshot, dict) or not snapshot:
                st.error("This manifest does not contain a config_snapshot block to load.")
            else:
                effective_snapshot = copy.deepcopy(snapshot)
                if keep_paths:
                    effective_snapshot["paths"] = copy.deepcopy(cfg.get("paths", {}) or {})
                st.session_state["loaded_manifest_config_snapshot"] = copy.deepcopy(snapshot)
                st.session_state["loaded_manifest_effective_config_snapshot"] = effective_snapshot
                st.session_state["loaded_manifest_source"] = _manifest_short_name(selected["manifest"], selected["source"])
                st.session_state["loaded_manifest_keep_current_paths"] = bool(keep_paths)
                # Force a fresh set of widget keys every time a manifest/config is loaded,
                # even if the same manifest is loaded again.
                _refresh_loaded_config_widget_state(effective_snapshot)
                st.success("Settings loaded. The app will rerun now.")
                st.rerun()
    with load_cols[1]:
        snapshot = valid[selected_index]["manifest"].get("config_snapshot") or {}
        if isinstance(snapshot, dict) and snapshot:
            yaml_text = yaml.safe_dump(_make_yaml_safe(snapshot), sort_keys=False, allow_unicode=True, width=120)
            st.download_button(
                "Download selected config snapshot YAML",
                data=yaml_text,
                file_name="loaded_manifest_config_snapshot.yaml",
                mime="application/x-yaml",
                key="manifest_snapshot_download",
            )


def _load_manifests_from_inputs(path_text: str, uploaded_files: list[Any] | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw_line in str(path_text or "").splitlines():
        raw = raw_line.strip()
        if not raw or raw.startswith("#"):
            continue
        source_path = Path(raw).expanduser()
        try:
            manifest_path = _resolve_manifest_path(source_path)
            data = _load_manifest_or_config_file(manifest_path)
            results.append({"ok": True, "source": str(manifest_path), "manifest": data})
        except Exception as exc:
            results.append({"ok": False, "source": raw, "error": str(exc)})

    for upload in uploaded_files or []:
        try:
            content = upload.getvalue().decode("utf-8")
            data = _parse_manifest_or_config_text(content, getattr(upload, "name", "uploaded manifest"))
            results.append({"ok": True, "source": getattr(upload, "name", "uploaded manifest"), "manifest": data})
        except Exception as exc:
            results.append({"ok": False, "source": getattr(upload, "name", "uploaded manifest"), "error": str(exc)})
    return results


def _load_manifest_or_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return _parse_manifest_or_config_text(text, str(path))


def _parse_manifest_or_config_text(text: str, source: str) -> dict[str, Any]:
    """Parse a manifest/export_summary JSON or a plain export_config YAML file.

    Plain YAML config files are wrapped into a pseudo-manifest so the same
    Load settings button can apply their settings.
    """
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest/config is not a dictionary: {source}")
    if "config_snapshot" in data or "summary" in data or "stage_timings" in data:
        return data
    # Treat a plain export_config.yaml as settings to load.
    return {
        "status": "config_yaml",
        "output_root": ((data.get("paths") or {}).get("output_root") if isinstance(data.get("paths"), dict) else ""),
        "summary": {
            "rgb_scheme": ((data.get("image_export") or {}).get("rgb_scheme") if isinstance(data.get("image_export"), dict) else ""),
            "square_crop_modes": {
                split: ((data.get("square_crops") or {}).get(f"{split}_crop_mode") if isinstance(data.get("square_crops"), dict) else "")
                for split in ["train", "val", "test"]
            },
        },
        "config_snapshot": data,
    }


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    candidates = [
        path / "manifest.json",
        path / "export_summary.json",
        path / "EXPORT_MANIFEST.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No manifest.json or export_summary.json found under: {path}")


def _manifest_short_name(manifest: dict[str, Any], source: str) -> str:
    root = str(manifest.get("output_root") or manifest.get("summary", {}).get("output_root") or source)
    return Path(root).name or str(source)


def _manifest_summary_row(manifest: dict[str, Any], source: str) -> dict[str, Any]:
    summary = manifest.get("summary", {}) or {}
    square = summary.get("square_crops", {}) or {}
    splits = square.get("splits", {}) or {}
    vendor = summary.get("vendor_filter", {}) or {}
    selection = summary.get("deterministic_selection_mode", {}) or {}
    target_pos = summary.get("deterministic_target_positive_ratio", {}) or {}
    return {
        "dataset": _manifest_short_name(manifest, source),
        "output_root": manifest.get("output_root", ""),
        "duration_min": round(float(manifest.get("total_duration_minutes", 0.0) or 0.0), 1),
        "vendor_filter": bool(vendor.get("enabled", False)),
        "vendors": ", ".join(vendor.get("include_vendors", []) or []),
        "source_train/val/test": _split_count_text(summary.get("splits", {}) or {}),
        "crop_modes": _split_count_text(summary.get("square_crop_modes", {}) or {}),
        "selection_modes": _split_count_text(selection),
        "target_pos": _split_count_text(target_pos),
        "square_images": int(square.get("num_images", 0) or 0),
        "positive_images": int(square.get("num_positive_images", 0) or 0),
        "train_images": int((splits.get("train", {}) or {}).get("num_images", 0) or 0),
        "train_pos": int((splits.get("train", {}) or {}).get("num_positive_images", 0) or 0),
        "val_images": int((splits.get("val", {}) or {}).get("num_images", 0) or 0),
        "test_images": int((splits.get("test", {}) or {}).get("num_images", 0) or 0),
        "R": _channel_compact(manifest, "R"),
        "G": _channel_compact(manifest, "G"),
        "B": _channel_compact(manifest, "B"),
    }


def _split_count_text(value: dict[str, Any]) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    return ", ".join(f"{k}={v}" for k, v in value.items())


def _channel_compact(manifest: dict[str, Any], channel: str) -> str:
    ch = (((manifest.get("config_snapshot") or {}).get("image_export") or {}).get("custom_channel_pipeline") or {}).get(channel, {}) or {}
    source = ch.get("source", "current_crop")
    ops = []
    for step in ch.get("steps", []) or []:
        op = str(step.get("op", "none"))
        params = step.get("params", {}) or {}
        if op == "percentile_normalize" and "percentiles" in params:
            op = f"percentile{params['percentiles']}"
        ops.append(op)
    return f"{source}: " + " -> ".join(ops)


def _explain_manifest_baseline(manifest: dict[str, Any]) -> str:
    row = _manifest_summary_row(manifest, str(manifest.get("output_root", "")))
    vendor_text = row["vendors"] if row["vendor_filter"] else "all vendors"
    return (
        f"This dataset is **{row['dataset']}**. It used **{vendor_text}**, "
        f"created **{row['square_images']}** square crops, and **{row['positive_images']}** of them are mass-positive. "
        f"Train/val/test exported crop counts are **{row['train_images']} / {row['val_images']} / {row['test_images']}**. "
        f"Crop modes are: **{row['crop_modes']}**."
    )


def _explain_manifest_change(prev: dict[str, Any], cur: dict[str, Any]) -> str:
    changes = []
    prev_row = _manifest_summary_row(prev, str(prev.get("output_root", "")))
    cur_row = _manifest_summary_row(cur, str(cur.get("output_root", "")))

    def add_change(label: str, old: Any, new: Any) -> None:
        if old != new:
            changes.append(f"- **{label}** changed from `{old}` to `{new}`.")

    add_change("vendor filter", prev_row["vendors"] if prev_row["vendor_filter"] else "all vendors", cur_row["vendors"] if cur_row["vendor_filter"] else "all vendors")
    add_change("source split counts", prev_row["source_train/val/test"], cur_row["source_train/val/test"])
    add_change("crop modes", prev_row["crop_modes"], cur_row["crop_modes"])
    add_change("selection modes", prev_row["selection_modes"], cur_row["selection_modes"])
    add_change("target positive ratios", prev_row["target_pos"], cur_row["target_pos"])
    add_change("R channel", prev_row["R"], cur_row["R"])
    add_change("G channel", prev_row["G"], cur_row["G"])
    add_change("B channel", prev_row["B"], cur_row["B"])

    old_n = int(prev_row["square_images"])
    new_n = int(cur_row["square_images"])
    if old_n != new_n:
        delta = new_n - old_n
        changes.append(f"- **Total square crops** changed from `{old_n}` to `{new_n}` (`{delta:+d}`).")
    old_pos = int(prev_row["positive_images"])
    new_pos = int(cur_row["positive_images"])
    if old_pos != new_pos:
        delta = new_pos - old_pos
        changes.append(f"- **Positive square crops** changed from `{old_pos}` to `{new_pos}` (`{delta:+d}`).")
    if not changes:
        return "No important manifest-level settings changed, except output path/timestamps."
    return "\n".join(changes)


def _render_manifest_key_settings(manifest: dict[str, Any]) -> None:
    config = manifest.get("config_snapshot", {}) or {}
    summary = manifest.get("summary", {}) or {}
    cols = st.columns(3)
    cols[0].metric("Output", Path(str(manifest.get("output_root", ""))).name)
    cols[1].metric("Total crops", int((summary.get("square_crops", {}) or {}).get("num_images", 0) or 0))
    cols[2].metric("Positive crops", int((summary.get("square_crops", {}) or {}).get("num_positive_images", 0) or 0))
    with st.expander("Resolved config snapshot", expanded=False):
        st.json(_make_yaml_safe(config))


def _manifest_diff_rows(prev: dict[str, Any], cur: dict[str, Any], pair_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    watched = [
        ("output_root", ["output_root"]),
        ("vendor_filter", ["summary", "vendor_filter"]),
        ("source_splits", ["summary", "splits"]),
        ("square_crop_modes", ["summary", "square_crop_modes"]),
        ("deterministic_include_empty", ["summary", "deterministic_include_empty"]),
        ("deterministic_selection_mode", ["summary", "deterministic_selection_mode"]),
        ("deterministic_target_positive_ratio", ["summary", "deterministic_target_positive_ratio"]),
        ("square_crops", ["summary", "square_crops"]),
        ("preprocess", ["config_snapshot", "preprocess"]),
        ("R_pipeline", ["config_snapshot", "image_export", "custom_channel_pipeline", "R"]),
        ("G_pipeline", ["config_snapshot", "image_export", "custom_channel_pipeline", "G"]),
        ("B_pipeline", ["config_snapshot", "image_export", "custom_channel_pipeline", "B"]),
        ("square_crops_config", ["config_snapshot", "square_crops"]),
        ("crop_annotation_policy", ["config_snapshot", "crop_annotation_policy"]),
    ]
    pair = f"{_manifest_short_name(prev, '')} -> {_manifest_short_name(cur, '')}"
    for label, path in watched:
        old = _nested_get(prev, path, None)
        new = _nested_get(cur, path, None)
        if old != new:
            rows.append({
                "pair": pair,
                "section": label,
                "old": _compact_json(old),
                "new": _compact_json(new),
            })
    return rows


def _nested_get(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _compact_json(value: Any, max_len: int = 500) -> str:
    text = json.dumps(_make_yaml_safe(value), sort_keys=True, ensure_ascii=False, default=_json_default)
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text

# -----------------------------------------------------------------------------
# Current GUI configuration export
# -----------------------------------------------------------------------------


def _export_current_preprocessing_yaml_panel(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    crop_controls: dict[str, Any],
    display_controls: dict[str, Any],
    pipeline: dict[str, Any],
) -> None:
    """Expose a one-click YAML export of the current GUI preprocessing setup."""
    payload = _current_preprocessing_yaml_payload(
        config_path=config_path,
        cfg=cfg,
        crop_controls=crop_controls,
        display_controls=display_controls,
        pipeline=pipeline,
    )
    yaml_text = yaml.safe_dump(
        _make_yaml_safe(payload),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )

    st.sidebar.divider()
    with st.sidebar.expander("Export current preprocessing YAML", expanded=False):
        st.caption(
            "Downloads the current fixed preprocessing, crop controls, channel visibility, "
            "and per-channel RGB pipeline. This is intended as a reproducible experiment record "
            "and as a patch you can copy into export_config.yaml."
        )
        st.download_button(
            label="Download current preprocessing YAML",
            data=yaml_text,
            file_name="vindr_current_preprocessing_gui.yaml",
            mime="application/x-yaml",
            use_container_width=True,
        )
        if st.checkbox("Show YAML preview", value=False, key="show_current_preprocessing_yaml_preview"):
            st.text_area(
                "YAML preview",
                value=yaml_text,
                height=360,
                key="current_preprocessing_yaml_preview_text",
            )


def _current_preprocessing_yaml_payload(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    crop_controls: dict[str, Any],
    display_controls: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    crop_options = dict(crop_controls.get("crop_options", {}) or {})
    return {
        "format": "vindr_mammo_preprocessing_gui_export_v1",
        "source_config": str(config_path),
        "note": (
            "This file was exported from the preprocessing inspector GUI. "
            "It records the current interactive preprocessing/crop/RGB settings."
        ),
        "fixed_preprocessing_before_crops": dict(cfg.get("preprocess", {}) or {}),
        "crop_preview_settings": {
            "mode": crop_controls.get("mode"),
            "crop_size": crop_controls.get("crop_size"),
            "stride": crop_controls.get("stride"),
            "only_mass_crops": crop_controls.get("only_mass_crops"),
            "positive_crop_visible_mass_fraction": crop_controls.get("positivity_threshold"),
            "require_foreground": crop_controls.get("require_foreground"),
            "min_foreground_fraction": crop_controls.get("min_foreground_fraction"),
            "foreground_threshold": crop_controls.get("foreground_threshold"),
            "random_preview_count": crop_controls.get("random_preview_count"),
            "random_seed": crop_controls.get("random_seed"),
            "random_positive_fraction": crop_options.get("positive_fraction"),
            "center_shift_fraction": crop_options.get("center_shift_fraction"),
            "allow_partial_annotations": crop_options.get("allow_partial_annotations"),
            "min_box_visibility": crop_options.get("min_box_visibility"),
        },
        "display_debug_settings": {
            "visible_rgb_channels": list(display_controls.get("visible_channels", ["R", "G", "B"]) or []),
            "show_individual_processed_channels": bool(display_controls.get("show_channel_panels", False)),
        },
        "rgb_channel_pipeline": {
            "description": (
                "Each channel has a source crop plus an ordered operation list. "
                "source=current_crop uses the selected crop. "
                "source=contralateral_same_view_crop uses the same xyxy window from the opposite breast with the same view."
            ),
            "R": _pipeline_channel_payload(pipeline, "R"),
            "G": _pipeline_channel_payload(pipeline, "G"),
            "B": _pipeline_channel_payload(pipeline, "B"),
        },
        "export_config_patch": {
            "preprocess": dict(cfg.get("preprocess", {}) or {}),
            "crop_annotation_policy": {
                "allow_partial_annotations": crop_options.get("allow_partial_annotations"),
                "min_box_visibility": crop_options.get("min_box_visibility"),
                "reject_partial_windows": crop_options.get("reject_partial_windows"),
                "negative_max_box_visibility": crop_options.get("negative_max_box_visibility"),
            },
            "square_crops": {
                "crop_size": crop_controls.get("crop_size"),
                "stride": crop_controls.get("stride"),
                "deterministic_require_foreground": crop_controls.get("require_foreground"),
                "deterministic_min_foreground_fraction": crop_controls.get("min_foreground_fraction"),
                "deterministic_foreground_threshold": crop_controls.get("foreground_threshold"),
                "positive_fraction": crop_options.get("positive_fraction"),
                "center_shift_fraction": crop_options.get("center_shift_fraction"),
            },
            "image_export": {
                "rgb_scheme": "custom_channel_pipeline",
                "custom_channel_pipeline": {
                    "R": _pipeline_channel_payload(pipeline, "R"),
                    "G": _pipeline_channel_payload(pipeline, "G"),
                    "B": _pipeline_channel_payload(pipeline, "B"),
                },
            },
        },
    }


def _make_yaml_safe(value: Any) -> Any:
    """Convert NumPy/Pandas/Path objects to plain YAML-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): _make_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_yaml_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


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
def _build_enriched_record_table_cached(
    records_json: str,
    metadata_json: str,
    metadata_table_json: str,
    findings_json: str,
    split_df_json: str,
) -> pd.DataFrame:
    records = pd.DataFrame(json.loads(records_json))
    metadata_rows = json.loads(metadata_json)
    metadata_table_rows = json.loads(metadata_table_json)
    findings = json.loads(findings_json)
    # pandas 2.1+ may treat a raw JSON string as a file path.
    # Wrap the JSON literal in StringIO so it is parsed as JSON content.
    split_df = pd.read_json(io.StringIO(split_df_json), orient="split")

    records["image_id"] = records["image_id"].astype(str)
    records["study_id"] = records["study_id"].astype(str)
    split_small = split_df[["image_id", "export_split"]].copy()
    split_small["image_id"] = split_small["image_id"].astype(str)
    records = records.merge(split_small, on="image_id", how="left")

    vendor_map, meta_preview_map = _build_vendor_maps(metadata_rows, metadata_table_rows, records)
    records["vendor"] = records["image_id"].astype(str).map(vendor_map).fillna("Unknown")
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
    metadata_table_json = json.dumps(dataset.metadata_df.to_dict(orient="records"), default=_json_default, sort_keys=True)
    findings = {}
    for image_id, rows in dataset.findings_by_image_id.items():
        mass_rows = [r for r in rows if dataset._is_mass_finding(r)]
        findings[str(image_id)] = mass_rows
    findings_json = json.dumps(findings, default=_json_default, sort_keys=True)
    return _build_enriched_record_table_cached(
        json.dumps(dataset.image_records, default=_json_default, sort_keys=True),
        metadata_json,
        metadata_table_json,
        findings_json,
        split_df.to_json(orient="split"),
    )


# -----------------------------------------------------------------------------
# UI controls
# -----------------------------------------------------------------------------


def _global_preprocess_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expose fixed geometry/display preprocessing in the GUI sidebar.

    These options are applied before crop selection and before the per-channel RGB
    experiment pipeline. They match the preprocessing used by the export code:
    MONOCHROME1 correction, optional breast-region crop, and right-to-left mirroring.
    """
    cfg = dict(cfg)
    cfg["preprocess"] = dict(cfg.get("preprocess", {}) or {})
    pp = cfg["preprocess"]

    st.sidebar.divider()
    st.sidebar.subheader("Fixed preprocessing before crops")
    st.sidebar.caption(
        "These steps are applied before square crop selection and before the RGB channel pipeline. "
        "They are not experimental per-channel filters."
    )

    pp["invert_to_black_background"] = st.sidebar.checkbox(
        "Invert MONOCHROME1 to black background",
        value=bool(pp.get("invert_to_black_background", True)),
        key=_widget_key("fixed_preprocess_invert_to_black_background"),
        help=(
            "If enabled, only DICOM images tagged MONOCHROME1 are inverted. "
            "MONOCHROME2 images are left unchanged."
        ),
    )
    pp["crop_breast"] = st.sidebar.checkbox(
        "Crop to breast foreground",
        value=bool(pp.get("crop_breast", False)),
        key=_widget_key("fixed_preprocess_crop_breast"),
        help=(
            "Find the breast foreground and remove as much pure background as possible. "
            "Default is off so deterministic full-image crop experiments are not silently altered."
        ),
    )
    pp["mirror_right_to_left"] = st.sidebar.checkbox(
        "Mirror right-entering breasts to left-entering",
        value=bool(pp.get("mirror_right_to_left", True)),
        key=_widget_key("fixed_preprocess_mirror_right_to_left"),
        help="If the breast foreground is mostly on the right side, flip horizontally and update boxes.",
    )
    pp["crop_padding"] = st.sidebar.number_input(
        "Breast crop padding, pixels",
        min_value=0,
        max_value=512,
        step=5,
        value=int(pp.get("crop_padding", 20)),
        key=_widget_key("fixed_preprocess_crop_padding"),
        help="Padding added around the detected breast foreground crop.",
    )

    threshold_mode = st.sidebar.radio(
        "Breast crop threshold",
        ["auto", "manual"],
        index=0 if pp.get("crop_threshold", None) is None else 1,
        key=_widget_key("fixed_preprocess_crop_threshold_mode"),
        horizontal=True,
        help="Auto usually works best. Manual is useful for debugging foreground segmentation.",
    )
    if threshold_mode == "manual":
        pp["crop_threshold"] = st.sidebar.number_input(
            "Manual threshold value",
            value=float(pp.get("crop_threshold", 0.0) or 0.0),
            step=0.01,
            key=_widget_key("fixed_preprocess_crop_threshold_value"),
            format="%.6f",
        )
    else:
        pp["crop_threshold"] = None

    pp["min_component_area_fraction"] = st.sidebar.slider(
        "Minimum breast component area fraction",
        min_value=0.0,
        max_value=0.05,
        value=float(pp.get("min_component_area_fraction", 0.001)),
        key=_widget_key("fixed_preprocess_min_component_area_fraction"),
        step=0.0005,
        format="%.4f",
        help="Small connected components below this relative image area are ignored while finding the breast crop.",
    )

    with st.sidebar.expander("What happened to the image?", expanded=False):
        st.write(
            "For each loaded image, the metadata panel reports whether the image was actually "
            "inverted, what breast crop box was used, and whether mirroring was applied."
        )

    return cfg


def _display_controls() -> dict[str, Any]:
    st.sidebar.divider()
    st.sidebar.subheader("Display/debug controls")
    visible_channels = st.sidebar.multiselect(
        "Visible RGB channels",
        options=["R", "G", "B"],
        default=["R", "G", "B"],
        key=_widget_key("gui_display_visible_channels"),
        help=(
            "Controls only the GUI display of the processed RGB crop. Hidden channels "
            "are set to zero, which makes it easier to debug individual channel pipelines."
        ),
    )
    show_channel_panels = st.sidebar.checkbox(
        "Show individual processed channels",
        value=True,
        key=_widget_key("gui_display_show_channel_panels"),
        help="Show R, G, and B as separate grayscale panels below the main images.",
    )
    return {
        "visible_channels": list(visible_channels),
        "show_channel_panels": bool(show_channel_panels),
    }


def _render_loaded_crop_config_debug(crop_cfg: dict[str, Any], policy: dict[str, Any]) -> None:
    """Show the crop/export settings currently loaded from the active config."""
    if not _is_loaded_config_active():
        return
    crop_preview = {
        "crop_size": crop_cfg.get("crop_size"),
        "stride": crop_cfg.get("stride"),
        "crop_modes": {split: crop_cfg.get(f"{split}_crop_mode") for split in ["train", "val", "test"]},
        "deterministic_selection_mode": {split: _selection_mode_from_config(crop_cfg, split) for split in ["train", "val", "test"]},
        "deterministic_include_empty": {
            split: bool(crop_cfg.get(f"{split}_deterministic_include_empty", crop_cfg.get("deterministic_include_empty", True)))
            for split in ["train", "val", "test"]
        },
        "target_positive_ratio": {
            split: crop_cfg.get(f"{split}_deterministic_target_positive_ratio", crop_cfg.get("deterministic_target_positive_ratio", crop_cfg.get("positive_fraction")))
            for split in ["train", "val", "test"]
        },
        "foreground_filter": {
            "deterministic_require_foreground": crop_cfg.get("deterministic_require_foreground"),
            "deterministic_min_foreground_fraction": crop_cfg.get("deterministic_min_foreground_fraction"),
            "deterministic_foreground_threshold": crop_cfg.get("deterministic_foreground_threshold"),
            "train_deterministic_require_foreground": crop_cfg.get("train_deterministic_require_foreground"),
            "val_deterministic_require_foreground": crop_cfg.get("val_deterministic_require_foreground"),
            "test_deterministic_require_foreground": crop_cfg.get("test_deterministic_require_foreground"),
        },
        "random_crop_options": {
            "positive_fraction": crop_cfg.get("positive_fraction"),
            "random_crops_per_annotation": crop_cfg.get("random_crops_per_annotation"),
            "center_shift_fraction": crop_cfg.get("center_shift_fraction"),
            "seed": crop_cfg.get("seed"),
        },
        "crop_annotation_policy": copy.deepcopy(policy),
    }
    with st.sidebar.expander("Loaded crop settings check", expanded=False):
        st.caption("This is the crop configuration currently read from the active config before export.")
        st.code(yaml.safe_dump(_make_yaml_safe(crop_preview), sort_keys=False, allow_unicode=True, width=120), language="yaml")
        if st.button("Refresh crop controls from loaded config", key="refresh_preview_crop_controls", use_container_width=True):
            _refresh_loaded_config_widget_state()
            st.rerun()


def _crop_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    st.sidebar.divider()
    st.sidebar.subheader("Crop controls")
    crop_cfg = cfg.get("square_crops", {})
    policy = cfg.get("crop_annotation_policy", {})
    _render_loaded_crop_config_debug(crop_cfg, policy)

    crop_size = st.sidebar.number_input(
        "Crop size n",
        min_value=128,
        max_value=4096,
        step=128,
        value=int(crop_cfg.get("crop_size", 1024)),
        key=_widget_key("crop_size_n"),
    )
    stride = st.sidebar.number_input(
        "Deterministic stride",
        min_value=64,
        max_value=4096,
        step=64,
        value=int(crop_cfg.get("stride", 512)),
        key=_widget_key("crop_stride"),
    )

    default_crop_mode = "stochastic random" if str(crop_cfg.get("train_crop_mode", "deterministic")) == "random" else "deterministic sliding"
    crop_mode = st.sidebar.radio(
        "Crop proposal mode",
        ["deterministic sliding", "stochastic random"],
        index=["deterministic sliding", "stochastic random"].index(default_crop_mode),
        key=_widget_key("crop_proposal_mode"),
        help=(
            "Deterministic uses the normal sliding-window grid. Stochastic samples random "
            "windows and can be biased toward masses. This is for GUI inspection only unless "
            "you also use the corresponding export config options."
        ),
    )

    default_only_mass = _selection_mode_from_config(crop_cfg, "train") == "mass_only"
    only_mass_crops = st.sidebar.checkbox("Show only crops with visible mass", value=default_only_mass, key=_widget_key("crop_only_mass_crops"))
    positivity_threshold = st.sidebar.slider(
        "Positive crop threshold, visible mass fraction",
        min_value=0.0,
        max_value=1.0,
        value=float(policy.get("min_box_visibility", 0.30)),
        step=0.05,
        key=_widget_key("crop_positivity_threshold"),
        help="A crop is considered positive if at least one mass box has this fraction visible inside the crop.",
    )
    # This is a GUI display/debug option, so default it to enabled even when the
    # export annotation policy is strict. Users can still turn it off manually.
    partial_key = _widget_key("gui_display_partial_boxes_after_clipping")
    if partial_key not in st.session_state:
        st.session_state[partial_key] = bool(policy.get("allow_partial_annotations", True)) if _is_loaded_config_active() else True
    allow_partial = st.sidebar.checkbox(
        "Display partial boxes after clipping",
        key=partial_key,
        help=(
            "GUI-only debugging default. When enabled, boxes that intersect the crop boundary "
            "are clipped and drawn if they satisfy the visibility threshold. This does not "
            "change config/export_config.yaml unless you explicitly export and apply the GUI YAML."
        ),
    )
    min_box_visibility = st.sidebar.slider(
        "Minimum box visibility to draw/keep",
        0.0,
        1.0,
        float(policy.get("min_box_visibility", 0.30)),
        0.05,
        key=_widget_key("crop_min_box_visibility"),
    )

    random_preview_count = int(crop_cfg.get("random_crops_per_annotation", 20) or 20)
    random_positive_fraction = float(crop_cfg.get("positive_fraction", 0.80))
    random_seed = int(crop_cfg.get("seed", 123))
    center_shift_fraction = float(crop_cfg.get("center_shift_fraction", 0.25))
    if crop_mode == "stochastic random":
        with st.sidebar.expander("Stochastic crop options", expanded=True):
            random_preview_count = st.number_input(
                "Random crops to preview",
                min_value=1,
                max_value=500,
                value=int(random_preview_count),
                step=1,
                key=_widget_key("crop_random_preview_count"),
            )
            random_positive_fraction = st.slider(
                "Random positive fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(crop_cfg.get("positive_fraction", 0.80)),
                step=0.05,
                key=_widget_key("crop_random_positive_fraction"),
                help="Probability of requesting a mass-positive random crop when the image has masses.",
            )
            center_shift_fraction = st.slider(
                "Mass-center random shift fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(crop_cfg.get("center_shift_fraction", 0.25)),
                step=0.05,
                key=_widget_key("crop_center_shift_fraction"),
            )
            random_seed = st.number_input("Random preview seed", min_value=0, max_value=999999, value=random_seed, step=1, key=_widget_key("crop_random_seed"))

    with st.sidebar.expander("Foreground-ratio crop filter", expanded=False):
        require_foreground = st.checkbox(
            "Require crop to contain breast foreground",
            value=bool(crop_cfg.get("deterministic_require_foreground", False)),
            key=_widget_key("crop_require_foreground"),
            help=(
                "Reject crop windows whose foreground/breast-pixel fraction is below the threshold. "
                "This is useful if you turn off the fixed breast crop and want square crops to remove pure background windows."
            ),
        )
        min_foreground_fraction = st.slider(
            "Minimum foreground fraction in crop",
            min_value=0.0,
            max_value=1.0,
            value=float(crop_cfg.get("deterministic_min_foreground_fraction", 0.05)),
            step=0.01,
            key=_widget_key("crop_min_foreground_fraction"),
        )
        fg_threshold_mode = st.radio(
            "Foreground threshold for square crops",
            ["auto", "manual"],
            index=0 if crop_cfg.get("deterministic_foreground_threshold", None) is None else 1,
            horizontal=True,
            key=_widget_key("crop_foreground_threshold_mode"),
        )
        if fg_threshold_mode == "manual":
            foreground_threshold = st.number_input(
                "Manual foreground threshold",
                value=float(crop_cfg.get("deterministic_foreground_threshold", 0.0) or 0.0),
                step=0.01,
                format="%.6f",
                key=_widget_key("crop_foreground_threshold_value"),
            )
        else:
            foreground_threshold = None
        foreground_mask_preview = st.checkbox(
            "Show foreground mask preview for selected crop",
            value=False,
            key=_widget_key("crop_foreground_mask_preview"),
            help="Shows which crop pixels counted as breast foreground.",
        )

    crop_options = {
        "enabled": True,
        "mode": "random" if crop_mode == "stochastic random" else "deterministic",
        "crop_size": int(crop_size),
        "stride": int(stride),
        "allow_partial_annotations": bool(allow_partial),
        "min_box_visibility": float(min_box_visibility),
        "reject_partial_windows": bool(policy.get("reject_partial_windows", not bool(allow_partial))) if _is_loaded_config_active() else not bool(allow_partial),
        "negative_max_box_visibility": float(policy.get("negative_max_box_visibility", 0.0)),
        "pad_if_needed": True,
        "pad_value": 0.0,
        "positive_fraction": float(random_positive_fraction),
        "center_shift_fraction": float(center_shift_fraction),
        "max_random_tries": int(crop_cfg.get("max_random_tries", 80)),
    }
    return {
        "crop_size": int(crop_size),
        "stride": int(stride),
        "mode": "random" if crop_mode == "stochastic random" else "deterministic",
        "random_preview_count": int(random_preview_count),
        "random_seed": int(random_seed),
        "only_mass_crops": bool(only_mass_crops),
        "positivity_threshold": float(positivity_threshold),
        "require_foreground": bool(require_foreground),
        "min_foreground_fraction": float(min_foreground_fraction),
        "foreground_threshold": foreground_threshold,
        "foreground_mask_preview": bool(foreground_mask_preview),
        "crop_options": crop_options,
    }


OP_NAMES = [
    "none",
    "percentile_normalize",
    "percentile_clip_only",
    "zscore_clip",
    "standardize_to_target",
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


def _custom_pipeline_from_image_export_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the GUI channel pipeline implied by image_export settings.

    Older manifests/configs may use high-level rgb_scheme values such as
    intensity_equalized_gradient instead of custom_channel_pipeline. The GUI is
    channel-pipeline based, so convert those schemes to equivalent editable
    R/G/B steps for inspection and for GUI-driven exports.
    """
    image_export = (cfg or {}).get("image_export", {}) or {}
    explicit = image_export.get("custom_channel_pipeline") or {}
    scheme = str(image_export.get("rgb_scheme", "custom_channel_pipeline") or "custom_channel_pipeline")
    if scheme == "custom_channel_pipeline" and isinstance(explicit, dict) and explicit:
        return copy.deepcopy(explicit)

    single_window = image_export.get("single_window", [1.0, 99.0])
    if not isinstance(single_window, (list, tuple)) or len(single_window) < 2:
        single_window = [1.0, 99.0]

    if scheme == "multi_window":
        windows = image_export.get("multi_window_percentiles", [[0.5, 99.5], [1.0, 99.0], [2.0, 98.0]])
        while len(windows) < 3:
            windows.append(single_window)
        return {
            "R": {"source": "current_crop", "steps": [{"op": "percentile_normalize", "params": {"percentiles": list(windows[0])}}]},
            "G": {"source": "current_crop", "steps": [{"op": "percentile_normalize", "params": {"percentiles": list(windows[1])}}]},
            "B": {"source": "current_crop", "steps": [{"op": "percentile_normalize", "params": {"percentiles": list(windows[2])}}]},
        }

    if scheme == "equalized_rgb":
        step = [
            {"op": "percentile_normalize", "params": {"percentiles": list(single_window)}},
            {"op": "hist_equalize", "params": {}},
        ]
        return {ch: {"source": "current_crop", "steps": copy.deepcopy(step)} for ch in ["R", "G", "B"]}

    if scheme == "intensity_equalized_gradient":
        ieg = image_export.get("intensity_equalized_gradient", {}) or {}
        intensity_window = ieg.get("intensity_window", single_window)
        gradient_window = ieg.get("gradient_window", single_window)
        gradient_ksize = int(ieg.get("gradient_ksize", 3) or 3)
        return {
            "R": {
                "source": "current_crop",
                "steps": [{"op": "percentile_normalize", "params": {"percentiles": list(intensity_window)}}],
            },
            "G": {
                "source": "current_crop",
                "steps": [
                    {"op": "percentile_normalize", "params": {"percentiles": list(intensity_window)}},
                    {"op": "hist_equalize", "params": {}},
                ],
            },
            "B": {
                "source": "current_crop",
                "steps": [{"op": "sobel_gradient", "params": {"ksize": gradient_ksize, "percentiles": list(gradient_window)}}],
            },
        }

    # grayscale_rgb, bitpack16 fallback, and unknown schemes become three equal grayscale channels.
    step = [{"op": "percentile_normalize", "params": {"percentiles": list(single_window)}}]
    return {ch: {"source": "current_crop", "steps": copy.deepcopy(step)} for ch in ["R", "G", "B"]}


def _loaded_pipeline_debug_panel(cfg: dict[str, Any], pipeline: dict[str, Any]) -> None:
    if not st.session_state.get("loaded_manifest_config_snapshot"):
        return
    with st.sidebar.expander("Loaded RGB pipeline check", expanded=False):
        snapshot_pipeline = _custom_pipeline_from_image_export_config(cfg)
        st.caption("This is the RGB pipeline currently being used by the GUI after loading settings.")
        st.code(yaml.safe_dump(_make_yaml_safe(snapshot_pipeline), sort_keys=False, allow_unicode=True, width=100), language="yaml")


def _pipeline_controls(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_pipeline = _custom_pipeline_from_image_export_config(cfg)
    widget_suffix = _active_widget_suffix()
    default_channel_sources = {
        "R": str(cfg_pipeline.get("R", {}).get("source", "current_crop")),
        "G": str(cfg_pipeline.get("G", {}).get("source", "current_crop")),
        "B": str(cfg_pipeline.get("B", {}).get("source", "contralateral_same_view_crop")),
    }
    default_channel_steps = {
        "R": [str(s.get("op", "none")) for s in cfg_pipeline.get("R", {}).get("steps", [])] or ["percentile_normalize", "hist_equalize", "standardize_to_target"],
        "G": [str(s.get("op", "none")) for s in cfg_pipeline.get("G", {}).get("steps", [])] or ["percentile_normalize", "hist_equalize", "percentile_normalize", "hist_equalize", "standardize_to_target"],
        "B": [str(s.get("op", "none")) for s in cfg_pipeline.get("B", {}).get("steps", [])] or ["percentile_normalize", "hist_equalize", "standardize_to_target"],
    }
    source_options = {
        "current_crop": "current crop",
        "contralateral_same_view_crop": "opposite breast, same view, same xyxy crop",
    }
    pipeline: dict[str, Any] = {}
    for channel in ["R", "G", "B"]:
        with st.sidebar.expander(f"{channel} channel", expanded=(channel == "R")):
            default_source = default_channel_sources[channel]
            if default_source not in source_options:
                default_source = "current_crop"
            source_label = st.selectbox(
                f"Source ({channel})",
                options=list(source_options.values()),
                index=list(source_options.keys()).index(default_source),
                key=f"{channel}_source{widget_suffix}",
                help=(
                    "Use the selected crop, or use the same crop coordinates from the opposite breast "
                    "with the same view position in the same study. The opposite-breast source is computed "
                    "after fixed preprocessing, so MONOCHROME1 correction and optional mirroring are applied first."
                ),
            )
            source = {v: k for k, v in source_options.items()}[source_label]
            default_steps_raw = list(cfg_pipeline.get(channel, {}).get("steps", []) or [])
            n_steps = st.number_input(
                f"Number of steps ({channel})",
                min_value=0,
                max_value=10,
                value=len(default_channel_steps[channel]),
                key=f"{channel}_n_steps{widget_suffix}",
            )
            steps = []
            for i in range(int(n_steps)):
                default_op = default_channel_steps[channel][i] if i < len(default_channel_steps[channel]) else "none"
                if default_op not in OP_NAMES:
                    default_op = "none"
                op = st.selectbox(
                    f"Step {i + 1}",
                    OP_NAMES,
                    index=OP_NAMES.index(default_op),
                    key=f"{channel}_op_{i}{widget_suffix}",
                )
                default_params = {}
                if i < len(default_steps_raw) and isinstance(default_steps_raw[i], dict) and str(default_steps_raw[i].get("op", "none")) == op:
                    default_params = dict(default_steps_raw[i].get("params", {}) or {})
                params = _op_parameter_controls(channel, i, op, default_params, widget_suffix=widget_suffix)
                steps.append({"op": op, "params": params})
            pipeline[channel] = {"source": source, "steps": steps}
    return pipeline


def _pipeline_channel_payload(pipeline: dict[str, Any], channel: str) -> dict[str, Any]:
    value = pipeline.get(channel, {})
    if isinstance(value, dict):
        return {
            "source": str(value.get("source", "current_crop")),
            "steps": list(value.get("steps", []) or []),
        }
    return {"source": "current_crop", "steps": list(value or [])}


def _channel_source(pipeline: dict[str, Any], channel: str) -> str:
    value = pipeline.get(channel, {})
    if isinstance(value, dict):
        return str(value.get("source", "current_crop"))
    return "current_crop"


def _channel_steps(pipeline: dict[str, Any], channel: str) -> list[dict[str, Any]]:
    value = pipeline.get(channel, [])
    if isinstance(value, dict):
        return list(value.get("steps", []) or [])
    return list(value or [])


def _pipeline_uses_contralateral(pipeline: dict[str, Any]) -> bool:
    return any(_channel_source(pipeline, ch) == "contralateral_same_view_crop" for ch in ["R", "G", "B"])

def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return min(max(number, minimum), maximum)


def _pair_from_params(params: dict[str, Any], key: str, default: tuple[float, float], minimum: float = 0.0, maximum: float = 100.0) -> tuple[float, float]:
    value = params.get(key, default) if isinstance(params, dict) else default
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        value = default
    lo = _clamp_float(value[0], minimum, maximum, default[0])
    hi = _clamp_float(value[1], minimum, maximum, default[1])
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _op_parameter_controls(
    channel: str,
    step: int,
    op: str,
    default_params: dict[str, Any] | None = None,
    widget_suffix: str = "",
) -> dict[str, Any]:
    params = dict(default_params or {})
    prefix = f"{channel}_{step}_{op}{widget_suffix}"
    if op in {"percentile_normalize", "percentile_clip_only"}:
        fallback_window = (70.0, 100.0) if channel == "G" and step == 2 and op == "percentile_normalize" else (1.0, 99.0)
        default_window = _pair_from_params(params, "percentiles", fallback_window)
        help_text = (
            "This is a normal percentile stretch. For bright-region channels, use a high window such as [50, 100] "
            "or [70, 100]. When loaded from a manifest, this value comes from that manifest."
            if default_window[0] >= 40.0 else None
        )
        lo, hi = st.slider("Percentile window", 0.0, 100.0, default_window, 0.5, key=f"{prefix}_win", help=help_text)
        return {"percentiles": [float(lo), float(hi)]}
    if op == "zscore_clip":
        default_limit = _clamp_float(params.get("z_limit", 3.0), 0.5, 10.0, 3.0)
        limit = st.slider("Z-score clip", 0.5, 10.0, default_limit, 0.5, key=f"{prefix}_z")
        return {"z_limit": float(limit)}
    if op == "standardize_to_target":
        target_mean = st.slider(
            "Target mean",
            min_value=0.0,
            max_value=1.0,
            value=_clamp_float(params.get("target_mean", 0.50), 0.0, 1.0, 0.50),
            step=0.01,
            key=f"{prefix}_target_mean",
            help="After dynamic ax+b standardization, this is the desired channel mean on the 0 to 1 scale.",
        )
        target_std = st.slider(
            "Target std",
            min_value=0.001,
            max_value=0.50,
            value=_clamp_float(params.get("target_std", 0.20), 0.001, 0.50, 0.20),
            step=0.005,
            key=f"{prefix}_target_std",
            help="After dynamic ax+b standardization, this is the desired channel standard deviation on the 0 to 1 scale.",
        )
        default_stat_win = _pair_from_params(params, "stat_percentiles", (1.0, 99.0))
        stat_lo, stat_hi = st.slider(
            "Statistic percentile range",
            min_value=0.0,
            max_value=100.0,
            value=default_stat_win,
            step=0.5,
            key=f"{prefix}_stat_win",
            help="Mean and std are estimated from pixels inside this percentile range to reduce outlier influence.",
        )
        clip_output = st.checkbox(
            "Clip standardized output to [0, 1]",
            value=bool(params.get("clip_output", True)),
            key=f"{prefix}_clip_output",
            help="Recommended for export, otherwise later 8-bit conversion may apply another percentile window.",
        )
        return {
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "stat_percentiles": [float(stat_lo), float(stat_hi)],
            "clip_output": bool(clip_output),
        }
    if op == "clahe":
        clip = st.slider("CLAHE clip limit", 0.5, 8.0, _clamp_float(params.get("clip_limit", 2.0), 0.5, 8.0, 2.0), 0.5, key=f"{prefix}_clip")
        tile_options = [4, 8, 16, 32]
        tile_default = int(params.get("tile_grid_size", 8) or 8)
        if tile_default not in tile_options:
            tile_default = 8
        tile = st.select_slider("CLAHE tile size", options=tile_options, value=tile_default, key=f"{prefix}_tile")
        return {"clip_limit": float(clip), "tile_grid_size": int(tile)}
    if op == "gaussian_blur":
        k_options = [3, 5, 7, 9, 11, 15, 21]
        k_default = int(params.get("ksize", 5) or 5)
        if k_default not in k_options:
            k_default = 5
        k = st.select_slider("Gaussian kernel", options=k_options, value=k_default, key=f"{prefix}_k")
        sigma = st.slider("Gaussian sigma", 0.0, 10.0, _clamp_float(params.get("sigma", 1.0), 0.0, 10.0, 1.0), 0.25, key=f"{prefix}_sigma")
        return {"ksize": int(k), "sigma": float(sigma)}
    if op == "median_blur":
        k_options = [3, 5, 7, 9, 11]
        k_default = int(params.get("ksize", 3) or 3)
        if k_default not in k_options:
            k_default = 3
        k = st.select_slider("Median kernel", options=k_options, value=k_default, key=f"{prefix}_k")
        return {"ksize": int(k)}
    if op == "sharpen":
        amount = st.slider("Sharpen strength", 0.0, 5.0, _clamp_float(params.get("amount", 1.0), 0.0, 5.0, 1.0), 0.25, key=f"{prefix}_amt")
        return {"amount": float(amount)}
    if op == "unsharp_mask":
        amount = st.slider("Unsharp amount", 0.0, 5.0, _clamp_float(params.get("amount", 1.5), 0.0, 5.0, 1.5), 0.25, key=f"{prefix}_amt")
        sigma = st.slider("Unsharp blur sigma", 0.25, 10.0, _clamp_float(params.get("sigma", 2.0), 0.25, 10.0, 2.0), 0.25, key=f"{prefix}_sigma")
        return {"amount": float(amount), "sigma": float(sigma)}
    if op == "sobel_gradient":
        k_options = [1, 3, 5, 7]
        k_default = int(params.get("ksize", 3) or 3)
        if k_default not in k_options:
            k_default = 3
        k = st.select_slider("Sobel kernel", options=k_options, value=k_default, key=f"{prefix}_k")
        lo, hi = st.slider("Gradient window", 0.0, 100.0, _pair_from_params(params, "percentiles", (1.0, 99.0)), 0.5, key=f"{prefix}_win")
        return {"ksize": int(k), "percentiles": [float(lo), float(hi)]}
    if op == "laplacian":
        k_options = [1, 3, 5, 7]
        k_default = int(params.get("ksize", 3) or 3)
        if k_default not in k_options:
            k_default = 3
        k = st.select_slider("Laplacian kernel", options=k_options, value=k_default, key=f"{prefix}_k")
        lo, hi = st.slider("Laplacian window", 0.0, 100.0, _pair_from_params(params, "percentiles", (1.0, 99.0)), 0.5, key=f"{prefix}_win")
        return {"ksize": int(k), "percentiles": [float(lo), float(hi)]}
    if op == "gamma":
        gamma = st.slider("Gamma", 0.1, 5.0, _clamp_float(params.get("gamma", 1.0), 0.1, 5.0, 1.0), 0.1, key=f"{prefix}_gamma")
        return {"gamma": float(gamma)}
    if op == "log":
        gain = st.slider("Log gain", 0.1, 20.0, _clamp_float(params.get("gain", 5.0), 0.1, 20.0, 5.0), 0.5, key=f"{prefix}_gain")
        return {"gain": float(gain)}
    return {}



# -----------------------------------------------------------------------------
# GUI-driven dataset export
# -----------------------------------------------------------------------------



def _selection_mode_from_config(crop_cfg: dict[str, Any], split: str) -> str:
    mode = str(crop_cfg.get(f"{split}_deterministic_selection_mode", "") or "").strip().casefold()
    aliases = {
        "mass only": "mass_only",
        "mass_only": "mass_only",
        "positive_only": "mass_only",
        "all": "all",
        "all windows": "all",
        "positive_ratio": "positive_ratio",
        "all mass + sampled non-mass": "positive_ratio",
        "all_mass_plus_sampled_non_mass": "positive_ratio",
        "finding_images_all_windows": "finding_images_all_windows",
        "finding_images_only_all_windows": "finding_images_all_windows",
        "findings_images_all_windows": "finding_images_all_windows",
        "finding images only, all crops": "finding_images_all_windows",
        "finding images only, all windows": "finding_images_all_windows",
        "images with findings, all crops": "finding_images_all_windows",
        "positive images, all crops": "finding_images_all_windows",
    }
    if mode in aliases:
        return aliases[mode]
    include_empty = bool(crop_cfg.get(f"{split}_deterministic_include_empty", crop_cfg.get("deterministic_include_empty", True)))
    return "all" if include_empty else "mass_only"


def _deterministic_selection_controls(crop_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    options = [
        "mass only",
        "all",
        "all mass + sampled non-mass",
        "finding images only, all crops",
    ]
    mode_to_label = {
        "mass_only": "mass only",
        "all": "all",
        "positive_ratio": "all mass + sampled non-mass",
        "finding_images_all_windows": "finding images only, all crops",
    }
    payload: dict[str, dict[str, Any]] = {}
    cols = st.columns(3)
    for split, col in zip(["train", "val", "test"], cols):
        with col:
            current_mode = _selection_mode_from_config(crop_cfg, split)
            label = mode_to_label.get(current_mode, "all")
            selected_label = st.radio(
                f"{split.title()} windows",
                options,
                index=options.index(label),
                key=_widget_key(f"gui_export_{split}_deterministic_selection_mode"),
                help=(
                    "mass only exports only crops with visible mass. all exports every deterministic window. "
                    "all mass + sampled non-mass keeps all positive windows and samples enough negative windows "
                    "to approach the target positive crop ratio for that split. "
                    "finding images only, all crops skips source images with no findings, but keeps every crop "
                    "from source images that contain at least one mass/finding."
                ),
            )
            target_ratio = float(crop_cfg.get(f"{split}_deterministic_target_positive_ratio", crop_cfg.get("deterministic_target_positive_ratio", crop_cfg.get("positive_fraction", 0.80))))
            if selected_label == "all mass + sampled non-mass":
                target_ratio = st.slider(
                    f"{split.title()} target positive ratio",
                    min_value=0.01,
                    max_value=1.0,
                    value=min(max(target_ratio, 0.01), 1.0),
                    step=0.01,
                    key=_widget_key(f"gui_export_{split}_target_positive_ratio"),
                )
            payload[split] = {
                "mode": {
                    "mass only": "mass_only",
                    "all": "all",
                    "all mass + sampled non-mass": "positive_ratio",
                    "finding images only, all crops": "finding_images_all_windows",
                }[selected_label],
                "target_positive_ratio": float(target_ratio),
            }
    return payload


def _apply_deterministic_selection_to_config(square: dict[str, Any], payload: dict[str, dict[str, Any]]) -> None:
    for split in ["train", "val", "test"]:
        split_payload = payload.get(split, {})
        mode = str(split_payload.get("mode", "all")).strip().casefold()
        if mode not in {"mass_only", "all", "positive_ratio", "finding_images_all_windows"}:
            mode = "all"
        ratio = float(split_payload.get("target_positive_ratio", square.get("positive_fraction", 0.80)))
        ratio = min(max(ratio, 0.01), 1.0)
        square[f"{split}_deterministic_selection_mode"] = mode
        square[f"{split}_deterministic_target_positive_ratio"] = ratio
        # Backward-compatible field used by older code and summaries.
        # finding_images_all_windows keeps empty crops, but only from source images with findings.
        square[f"{split}_deterministic_include_empty"] = mode != "mass_only"


def _export_dataset_from_gui_panel(
    *,
    cfg: dict[str, Any],
    records_df: pd.DataFrame,
    crop_controls: dict[str, Any],
    pipeline: dict[str, Any],
) -> None:
    """Run a dataset export directly from the GUI using current controls."""
    st.sidebar.divider()
    with st.sidebar.expander("Export dataset from GUI", expanded=False):
        loaded_active = _is_loaded_config_active()
        strict_replay_loaded = False
        if loaded_active:
            st.info(
                "A manifest/config is loaded. Defaults in this panel now come from the loaded config. "
                "Enable strict replay to export directly from the loaded config snapshot, ignoring any stale or edited GUI widgets."
            )
            if st.button("Refresh all export controls from loaded config", key="gui_export_refresh_loaded_config", use_container_width=True):
                _refresh_loaded_config_widget_state()
                st.rerun()
            strict_replay_loaded = st.checkbox(
                "Strict replay loaded config, ignore GUI control edits",
                value=True,
                key=_widget_key("gui_export_strict_replay_loaded_config"),
                help=(
                    "Recommended when you want a loaded manifest/config to reproduce the same export settings. "
                    "The output path and clean-output checkbox below are still applied so you can avoid overwriting old data."
                ),
            )
        st.caption(
            "Create a dataset using the current fixed preprocessing, crop controls, vendor filter, "
            "and RGB channel pipeline. This runs the same exporter as main.py, but with a GUI-built config."
        )
        export_cfg_defaults = dict(cfg.get("export", {}) or {})
        current_output = Path(str(cfg.get("paths", {}).get("output_root", "/mnt/t9/preprocessed-vindr-v3")))
        output_parent = st.text_input(
            "Export parent folder",
            value=str(current_output.parent),
            help="The dataset folder will be created inside this parent folder.",
            key=_widget_key("gui_export_parent_folder"),
        )
        dataset_name = st.text_input(
            "Dataset folder name",
            value=current_output.name if current_output.name else "preprocessed-vindr-gui",
            help="Final output path is parent/name. Example: /mnt/t9/preprocessed-vindr-v4.",
            key=_widget_key("gui_export_dataset_name"),
        )
        output_root = Path(output_parent) / dataset_name
        st.code(str(output_root), language="text")
        clean_output = st.checkbox(
            "Delete output folder before export",
            value=bool(export_cfg_defaults.get("clean_output_root", False)),
            key=_widget_key("gui_export_clean_output"),
            help="Enable only when you are sure the target folder can be removed.",
        )
        save_square = st.checkbox(
            "Export square_crops dataset",
            value=bool(export_cfg_defaults.get("save_square_crops", True)),
            key=_widget_key("gui_export_square_crops"),
        )
        save_baseline = st.checkbox(
            "Also export baseline_uncropped dataset",
            value=bool(export_cfg_defaults.get("save_baseline_uncropped", False)),
            key=_widget_key("gui_export_baseline"),
        )

        vendors = _available_vendors(records_df)
        vendor_cfg = dict(cfg.get("vendor_filter", {}) or {})
        vendor_options = ["all vendors", "selected vendors only"]
        default_vendor_label = "selected vendors only" if bool(vendor_cfg.get("enabled", False)) else "all vendors"
        vendor_mode = st.radio(
            "Vendor/device export filter",
            vendor_options,
            index=vendor_options.index(default_vendor_label),
            horizontal=True,
            key=_widget_key("gui_export_vendor_mode"),
        )
        selected_vendors: list[str] = []
        if vendor_mode == "selected vendors only":
            configured_vendors = [str(v) for v in (vendor_cfg.get("include_vendors") or [])]
            default_vendors = [v for v in configured_vendors if v in vendors]
            if not default_vendors:
                default_vendors = _default_comparison_vendors(records_df, min(5, len(vendors)))
            selected_vendors = st.multiselect(
                "Vendors/devices to include",
                options=vendors,
                default=[v for v in default_vendors if v in vendors] or (vendors[:1] if vendors else []),
                key=_widget_key("gui_export_selected_vendors"),
                help="Only source images from these detected devices will be included in train/val/test before crop export.",
            )
            if not selected_vendors:
                st.warning("Select at least one vendor, or switch to all vendors.")

        st.markdown("**Split-specific deterministic crop selection**")
        st.caption(
            "For each split, deterministic sliding windows can be exported as all windows, mass windows only, "
            "all mass windows plus sampled non-mass windows, or all windows from images that contain findings."
        )
        crop_cfg = dict(cfg.get("square_crops", {}) or {})
        selection_payload = _deterministic_selection_controls(crop_cfg)

        crop_export_modes = ["deterministic sliding for all splits", "train stochastic, val/test deterministic"]
        if str(crop_cfg.get("train_crop_mode", "deterministic")) == "random" and str(crop_cfg.get("val_crop_mode", "deterministic")) == "deterministic" and str(crop_cfg.get("test_crop_mode", "deterministic")) == "deterministic":
            default_export_mode = "train stochastic, val/test deterministic"
        else:
            default_export_mode = "deterministic sliding for all splits"
        export_mode = st.radio(
            "Crop export mode",
            crop_export_modes,
            index=crop_export_modes.index(default_export_mode),
            key=_widget_key("gui_export_crop_mode"),
            help="The split-specific deterministic selection controls above apply when a split is deterministic.",
        )
        confirm = st.checkbox("I checked the output path and want to start export", value=False, key=_widget_key("gui_export_confirm"))

        with st.expander("Effective loaded/export settings preview", expanded=False):
            if strict_replay_loaded:
                preview_cfg = _strict_replay_export_config(output_root=output_root, clean_output=clean_output)
            else:
                preview_cfg = _build_gui_export_config(
                    cfg=cfg,
                    output_root=output_root,
                    clean_output=clean_output,
                    selected_vendors=selected_vendors if vendor_mode == "selected vendors only" else [],
                    deterministic_selection=selection_payload,
                    export_mode=export_mode,
                    save_square=save_square,
                    save_baseline=save_baseline,
                    crop_controls=crop_controls,
                    pipeline=pipeline,
                )
            st.code(yaml.safe_dump(_make_yaml_safe(preview_cfg), sort_keys=False, allow_unicode=True, width=120), language="yaml")

        if st.button("Start exporting dataset", type="primary", use_container_width=True, disabled=not confirm):
            if vendor_mode == "selected vendors only" and not selected_vendors:
                st.error("No vendors selected. Select at least one vendor before exporting.")
                return
            if not save_square and not save_baseline:
                st.error("Nothing selected to export. Enable square_crops and/or baseline_uncropped.")
                return
            if strict_replay_loaded:
                export_cfg = _strict_replay_export_config(output_root=output_root, clean_output=clean_output)
            else:
                export_cfg = _build_gui_export_config(
                    cfg=cfg,
                    output_root=output_root,
                    clean_output=clean_output,
                    selected_vendors=selected_vendors if vendor_mode == "selected vendors only" else [],
                    deterministic_selection=selection_payload,
                    export_mode=export_mode,
                    save_square=save_square,
                    save_baseline=save_baseline,
                    crop_controls=crop_controls,
                    pipeline=pipeline,
                )
            _run_export_with_streamlit_progress(export_cfg)



def _strict_replay_export_config(*, output_root: Path, clean_output: bool) -> dict[str, Any]:
    """Return the loaded config snapshot with only safe output controls applied.

    This is intended for exact manifest/config replay. It bypasses GUI-derived
    crop, vendor, preprocessing, and RGB pipeline widgets, because those widgets
    can be stale or manually edited.
    """
    loaded = st.session_state.get("loaded_manifest_effective_config_snapshot")
    if not isinstance(loaded, dict):
        loaded = st.session_state.get("loaded_manifest_config_snapshot")
    if not isinstance(loaded, dict):
        raise RuntimeError("No loaded manifest/config snapshot is available for strict replay.")
    out = copy.deepcopy(loaded)
    out.setdefault("paths", {})["output_root"] = str(output_root)
    out.setdefault("export", {})["clean_output_root"] = bool(clean_output)
    return out


def _build_gui_export_config(
    *,
    cfg: dict[str, Any],
    output_root: Path,
    clean_output: bool,
    selected_vendors: list[str],
    deterministic_selection: dict[str, dict[str, Any]],
    export_mode: str,
    save_square: bool,
    save_baseline: bool,
    crop_controls: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("paths", {})["output_root"] = str(output_root)
    out.setdefault("export", {})["clean_output_root"] = bool(clean_output)
    out.setdefault("export", {})["save_square_crops"] = bool(save_square)
    out.setdefault("export", {})["save_baseline_uncropped"] = bool(save_baseline)
    out.setdefault("export", {})["save_empty_label_files"] = bool(out.get("export", {}).get("save_empty_label_files", True))

    out["vendor_filter"] = {
        "enabled": bool(selected_vendors),
        "include_vendors": list(selected_vendors),
    }

    crop_options = dict(crop_controls.get("crop_options", {}) or {})
    square = out.setdefault("square_crops", {})
    square["crop_size"] = int(crop_controls.get("crop_size", square.get("crop_size", 1024)))
    square["stride"] = int(crop_controls.get("stride", square.get("stride", 512)))
    if export_mode == "train stochastic, val/test deterministic":
        square["train_crop_mode"] = "random"
        square["val_crop_mode"] = "deterministic"
        square["test_crop_mode"] = "deterministic"
    else:
        square["train_crop_mode"] = "deterministic"
        square["val_crop_mode"] = "deterministic"
        square["test_crop_mode"] = "deterministic"
    _apply_deterministic_selection_to_config(square, deterministic_selection)
    square["deterministic_include_empty"] = bool(square.get("deterministic_include_empty", True))
    square["deterministic_require_foreground"] = bool(crop_controls.get("require_foreground", False))
    square["deterministic_min_foreground_fraction"] = float(crop_controls.get("min_foreground_fraction", 0.05))
    square["deterministic_foreground_threshold"] = crop_controls.get("foreground_threshold", None)
    square["positive_fraction"] = float(crop_options.get("positive_fraction", square.get("positive_fraction", 0.80)))
    square["center_shift_fraction"] = float(crop_options.get("center_shift_fraction", square.get("center_shift_fraction", 0.25)))

    out["crop_annotation_policy"] = {
        "allow_partial_annotations": bool(crop_options.get("allow_partial_annotations", False)),
        "min_box_visibility": float(crop_options.get("min_box_visibility", 0.30)),
        "reject_partial_windows": bool(crop_options.get("reject_partial_windows", True)),
        "negative_max_box_visibility": float(crop_options.get("negative_max_box_visibility", 0.0)),
    }
    out["image_export"] = dict(out.get("image_export", {}) or {})
    out["image_export"]["rgb_scheme"] = "custom_channel_pipeline"
    out["image_export"]["custom_channel_pipeline"] = {
        "R": _pipeline_channel_payload(pipeline, "R"),
        "G": _pipeline_channel_payload(pipeline, "G"),
        "B": _pipeline_channel_payload(pipeline, "B"),
    }
    return out


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _run_export_with_streamlit_progress(export_cfg: dict[str, Any]) -> None:
    stages = [
        "initialize_dataset",
        "make_train_val_test_split",
        "write_source_metadata_and_config",
        "export_square_crops",
        "export_baseline_uncropped",
        "write_completion_manifest",
    ]
    active_stages = list(stages)
    if not bool(export_cfg.get("export", {}).get("save_baseline_uncropped", False)):
        active_stages.remove("export_baseline_uncropped")

    progress_bar = st.progress(0.0, text="Export not started yet")
    status_box = st.empty()
    time_box = st.empty()
    log_box = st.empty()
    log_lines: list[str] = []
    export_started_at = time.monotonic()
    current_stage_started_at = export_started_at

    def _time_text(*, stage_fraction: float | None, event_name: str) -> str:
        """Return elapsed and remaining time text.

        Earlier versions estimated remaining time from the coarse overall export
        fraction. That made ETA look very close to elapsed, because the long
        square-crop stage starts after several short stages and therefore the
        overall fraction is already large. Here the ETA is based on the current
        stage progress, especially the image/crop progress inside
        ``export_square_crops``.
        """
        now = time.monotonic()
        total_elapsed = max(0.0, now - export_started_at)
        stage_elapsed = max(0.0, now - current_stage_started_at)
        if event_name == "stage_finish":
            return f"Elapsed: {_format_duration(total_elapsed)} | Estimated remaining: 0s"
        if stage_fraction is not None and stage_fraction > 0.001:
            frac = float(min(max(stage_fraction, 0.001), 0.999))
            stage_remaining = max(0.0, stage_elapsed * (1.0 - frac) / frac)
            return (
                f"Elapsed: {_format_duration(total_elapsed)} | "
                f"Stage elapsed: {_format_duration(stage_elapsed)} | "
                f"Estimated remaining: {_format_duration(stage_remaining)}"
            )
        return (
            f"Elapsed: {_format_duration(total_elapsed)} | "
            f"Stage elapsed: {_format_duration(stage_elapsed)} | "
            "Estimated remaining: calculating"
        )

    def update_progress(event: dict[str, Any]) -> None:
        nonlocal current_stage_started_at
        stage = str(event.get("stage", ""))
        try:
            stage_idx = active_stages.index(stage)
        except ValueError:
            stage_idx = 0
        event_name = str(event.get("event", ""))
        frac_inside_stage: float | None = None
        if event_name == "stage_start":
            current_stage_started_at = time.monotonic()
            frac_inside_stage = 0.0
        elif event_name == "image_progress" and int(event.get("total", 0) or 0) > 0:
            frac_inside_stage = min(1.0, max(0.0, float(event.get("processed", 0)) / float(event.get("total", 1))))
        elif event_name == "stage_finish":
            frac_inside_stage = 1.0
        elif event_name == "stage_failed":
            frac_inside_stage = None
        else:
            frac_inside_stage = 0.0

        # Keep the progress bar monotonic at a coarse stage level, but calculate
        # ETA from stage_fraction above. The ETA should answer: how much of the
        # currently running long operation remains?
        progress_fraction_for_bar = 0.02 if frac_inside_stage is None else float(frac_inside_stage)
        overall = (stage_idx + progress_fraction_for_bar) / max(len(active_stages), 1)
        text = stage.replace("_", " ")
        if event_name == "image_progress":
            unit = str(event.get("unit", "source images"))
            text += f" | {event.get('split', '?')} | {event.get('processed', 0)}/{event.get('total', 0)} {unit}"
        elif event_name == "stage_start":
            text += " | started"
        elif event_name == "stage_finish":
            text += " | complete"
        elif event_name == "stage_failed":
            text += " | failed"
        overall_clamped = float(min(max(overall, 0.0), 1.0))
        timer_text = _time_text(stage_fraction=frac_inside_stage, event_name=event_name)
        progress_bar.progress(overall_clamped, text=f"{text} | {timer_text}")
        time_box.info(timer_text)
        if event_name in {"stage_start", "stage_finish", "stage_failed"}:
            log_lines.append(text)
            log_box.code("\n".join(log_lines[-12:]), language="text")

    try:
        status_box.info("Export is running. Keep this browser tab open until completion.")
        result = export_from_config(export_cfg, progress_callback=update_progress)
        progress_bar.progress(1.0, text="Export complete")
        status_box.success(f"Export complete: {result.output_root}")
        _render_export_result_summary(result)
    except Exception as exc:
        status_box.error(f"Export failed: {exc}")
        raise


def _render_export_result_summary(result: Any) -> None:
    """Render export completion info without passing circular objects to st.json."""
    summary = _streamlit_json_safe(getattr(result, "summary", {}) or {})
    output_root = Path(getattr(result, "output_root", ""))

    st.subheader("Export result")
    st.write(f"Output root: `{output_root}`")

    summary_table_rows = []
    for key in [
        "num_source_images",
        "rgb_scheme",
        "histogram_equalization_enabled",
    ]:
        if key in summary:
            summary_table_rows.append({"field": key, "value": summary.get(key)})

    if isinstance(summary.get("splits"), dict):
        for split, count in summary["splits"].items():
            summary_table_rows.append({"field": f"split_{split}_source_images", "value": count})

    if isinstance(summary.get("manifest"), dict):
        manifest = summary["manifest"]
        for key in ["status", "finished_at", "total_duration_minutes", "manifest_path", "done_path"]:
            if key in manifest:
                summary_table_rows.append({"field": f"manifest_{key}", "value": manifest.get(key)})

    if summary_table_rows:
        st.dataframe(pd.DataFrame(summary_table_rows), use_container_width=True, hide_index=True)

    with st.expander("Full export summary", expanded=False):
        st.json(summary)

    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    st.download_button(
        "Download export summary JSON",
        data=summary_json,
        file_name="export_summary_gui.json",
        mime="application/json",
        use_container_width=True,
    )


def _streamlit_json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Make arbitrary nested values safe for st.json and json.dumps."""
    if _seen is None:
        _seen = set()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_reference>"
        _seen.add(obj_id)
        try:
            return {str(k): _streamlit_json_safe(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular_reference>"
        _seen.add(obj_id)
        try:
            return [_streamlit_json_safe(v, _seen) for v in value]
        finally:
            _seen.discard(obj_id)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)

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
    display_controls: dict[str, Any],
) -> None:
    filtered = _record_filter_controls(records_df, prefix="single")
    st.subheader("Image selection")
    if filtered.empty:
        st.write(f"Filtered images: **0**")
        st.warning("No images match the current filters.")
        return

    img_info_col, img_idx_col = st.columns([2.0, 1.0], vertical_alignment="bottom")
    img_info_col.write(f"Filtered images: **{len(filtered)}**")
    selected_pos = img_idx_col.number_input(
        "Image index",
        min_value=0,
        max_value=max(0, len(filtered) - 1),
        value=0,
        step=1,
        key="single_image_index_inline",
    )
    selected_row = filtered.iloc[int(selected_pos)]
    result = _prepare_sample(dataset, int(selected_row["record_index"]), crop_controls, crop_index=None, need_contralateral=False)
    crops = result["crops"]
    if not crops:
        st.write(f"Crops available after crop filter: **0**")
        st.warning("This image has no crops under the current crop filter/positivity threshold.")
        return

    crop_info_col, crop_idx_col = st.columns([2.0, 1.0], vertical_alignment="bottom")
    crop_info_col.write(f"Crops available after crop filter: **{len(crops)}**")
    crop_idx = crop_idx_col.number_input(
        "Crop index",
        min_value=0,
        max_value=max(0, len(crops) - 1),
        value=0,
        step=1,
        key="single_crop_index_inline",
    )
    result = _prepare_sample(dataset, int(selected_row["record_index"]), crop_controls, crop_index=int(crop_idx), need_contralateral=_pipeline_uses_contralateral(pipeline))
    _show_sample(result, pipeline, show_annotations=show_annotations, display_window=display_window, display_controls=display_controls)



def _render_comparison_mode(
    dataset: VindrMammoDataset,
    records_df: pd.DataFrame,
    crop_controls: dict[str, Any],
    pipeline: dict[str, Any],
    show_annotations: bool,
    display_window: tuple[float, float],
    display_controls: dict[str, Any],
) -> None:
    st.subheader("Vendor / image comparison")
    st.caption(
        "All comparison slots use the same crop controls and RGB preprocessing pipeline from the sidebar. "
        "By default, the first five slots are assigned to different detected vendors/devices when possible."
    )
    n_slots = st.slider("Number of comparison slots", 2, 10, 5)
    default_vendors = _default_comparison_vendors(records_df, n_slots)
    if default_vendors:
        st.caption("Default slot vendors: " + ", ".join(default_vendors))

    results = []
    slot_cols = st.columns(n_slots)
    for slot_idx, col in enumerate(slot_cols):
        with col:
            default_vendor = default_vendors[slot_idx] if slot_idx < len(default_vendors) else None
            filtered = _record_filter_controls(
                records_df,
                prefix=f"cmp_{slot_idx}",
                compact=True,
                default_vendor_mode="selected vendors" if default_vendor else "all vendors",
                default_selected_vendors=[default_vendor] if default_vendor else None,
                default_positive_choice="positive only",
                default_split="all",
            )
            header_col, img_idx_col = st.columns([1.15, 0.85], vertical_alignment="bottom")
            header_col.markdown(f"**Slot {slot_idx + 1}**")
            if default_vendor:
                header_col.caption(f"{len(filtered)} matching images | default: {default_vendor}")
            else:
                header_col.caption(f"{len(filtered)} matching images")
            if filtered.empty:
                img_idx_col.caption("No image index")
                results.append(None)
                continue
            img_idx = img_idx_col.number_input(
                "Image idx",
                0,
                max(0, len(filtered) - 1),
                0,
                1,
                key=f"cmp_{slot_idx}_imgidx",
            )
            row = filtered.iloc[int(img_idx)]
            tmp = _prepare_sample(dataset, int(row["record_index"]), crop_controls, crop_index=None, need_contralateral=False)
            crop_count = len(tmp["crops"])
            crop_label_col, crop_idx_col = st.columns([1.15, 0.85], vertical_alignment="bottom")
            crop_label_col.caption(f"{crop_count} crops")
            if crop_count == 0:
                crop_idx_col.caption("No crop index")
                results.append(None)
                continue
            cidx = crop_idx_col.number_input(
                "Crop idx",
                0,
                max(0, crop_count - 1),
                0,
                1,
                key=f"cmp_{slot_idx}_cropidx",
            )
            results.append(_prepare_sample(dataset, int(row["record_index"]), crop_controls, crop_index=int(cidx), need_contralateral=_pipeline_uses_contralateral(pipeline)))

    valid_results = [r for r in results if r is not None]
    if len(valid_results) >= 2:
        st.divider()
        _show_comparison_statistics(valid_results, pipeline)

    for i, result in enumerate(results):
        if result is None:
            continue
        st.divider()
        st.markdown(f"### Slot {i + 1}: {result['title']}")
        _show_sample(result, pipeline, show_annotations=show_annotations, display_window=display_window, display_controls=display_controls, compact=True)



# -----------------------------------------------------------------------------
# Compare selected images by statistics, not pixel alignment
# -----------------------------------------------------------------------------


def _show_comparison_statistics(results: list[dict[str, Any]], pipeline: dict[str, Any]) -> None:
    """Compare selected comparison slots using final-output image statistics.

    The selected images are not spatially registered and may be different patients,
    views, and vendors. Therefore this section compares summary covariates and
    compact 1-D intensity summaries rather than pixel-wise similarity. The main
    distance score now focuses on the final processed RGB output, because that is
    what the model will see after the channel pipeline and standardization. Raw
    grayscale crop statistics are still shown for diagnosis, but they are not used
    in the main final-output distance.
    """
    st.markdown("### Statistics comparison across selected slots")
    st.caption(
        "This compares the final processed RGB outputs by summary statistics and compact "
        "intensity-distribution distances. It does not compare pixels spatially. "
        "Raw crop statistics are shown separately because they describe the source crop before the RGB preprocessing pipeline."
    )

    feature_rows = []
    histograms: dict[str, dict[str, np.ndarray]] = {}
    for i, result in enumerate(results):
        slot_name = f"slot_{i + 1}"
        processed_rgb, _meta = apply_channel_pipeline(
            result["crop_image"],
            pipeline,
            source_crops=_source_crops_from_result(result),
        )
        row, hists = _comparison_features_for_sample(slot_name, result, processed_rgb)
        feature_rows.append(row)
        histograms[slot_name] = hists

    features_df = pd.DataFrame(feature_rows)

    identity_cols = [
        "slot", "vendor", "split", "image_id", "laterality", "view",
        "num_masses", "foreground_fraction",
    ]
    output_cols = [
        "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
        "R_iqr", "G_iqr", "B_iqr", "R_entropy", "G_entropy", "B_entropy",
    ]
    raw_cols = ["crop_mean", "crop_std", "crop_iqr", "crop_entropy", "crop_p1", "crop_p99"]

    st.markdown("**Final processed RGB statistics, model input**")
    show_cols = [c for c in identity_cols + output_cols if c in features_df.columns]
    st.dataframe(features_df[show_cols], use_container_width=True, hide_index=True)

    with st.expander("Raw source crop statistics, before channel preprocessing", expanded=False):
        raw_show_cols = [c for c in identity_cols + raw_cols if c in features_df.columns]
        st.dataframe(features_df[raw_show_cols], use_container_width=True, hide_index=True)
        st.caption(
            "These crop_* values are intentionally different from R/G/B values. They come from the grayscale crop before "
            "percentile normalization, histogram equalization, contralateral substitution, and standardization."
        )

    metric_df = _pairwise_statistical_similarity(features_df, histograms)
    if metric_df.empty:
        return

    st.markdown("**Pairwise final-output statistical similarity**")
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    best = metric_df.sort_values("combined_final_rgb_distance", ascending=True).iloc[0]
    worst = metric_df.sort_values("combined_final_rgb_distance", ascending=False).iloc[0]
    st.info(
        f"Most similar pair by final RGB distance: {best['pair']} "
        f"(combined={best['combined_final_rgb_distance']:.3f}). "
        f"Most different pair: {worst['pair']} "
        f"(combined={worst['combined_final_rgb_distance']:.3f})."
    )
    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            "- **final_rgb_mean_abs_std_diff**: average absolute standardized difference over processed R/G/B summary features. "
            "Lower is more similar. If your standardization is working, this should be small for R/G/B mean and std features.\n"
            "- **final_rgb_max_abs_std_diff**: largest single standardized feature difference among processed R/G/B features. This catches one channel/statistic being off.\n"
            "- **final_rgb_js_distance**: Jensen-Shannon distance between compact normalized final RGB intensity summaries. Lower is more similar.\n"
            "- **final_rgb_wasserstein_distance**: 1-D Earth-mover-style distance between compact normalized final RGB summaries on [0, 1]. Lower is more similar.\n"
            "- **combined_final_rgb_distance**: the main sorting score. It uses final processed RGB outputs only, not raw crop_* features.\n"
            "- **raw_crop_mean_abs_std_diff**: diagnostic distance for the grayscale crop before preprocessing. It can stay high even when final RGB statistics are matched."
        )


def _comparison_features_for_sample(slot_name: str, result: dict[str, Any], processed_rgb: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    crop = np.asarray(result["crop_image"], dtype=np.float32)
    summary = result.get("target_summary", {}) or {}
    selected = result.get("selected_crop", {}) or {}

    row: dict[str, Any] = {
        "slot": slot_name,
        "vendor": _vendor_from_summary(summary),
        "split": summary.get("split"),
        "image_id": summary.get("image_id"),
        "laterality": summary.get("laterality"),
        "view": summary.get("view_position"),
        "num_masses": int(summary.get("num_masses", 0) or 0),
        "foreground_fraction": selected.get("foreground_fraction", np.nan),
    }

    # Raw/source crop statistics are intentionally computed before the custom
    # RGB channel pipeline. They help diagnose scanner/crop differences, but
    # they are not the final model input if custom preprocessing is enabled.
    crop_norm = _normalize_percentile(crop, [1.0, 99.0])
    _add_feature_prefix(row, "crop", crop_norm)

    # Final processed RGB features represent what the detector sees.
    for channel_idx, channel_name in enumerate(["R", "G", "B"]):
        ch = processed_rgb[..., channel_idx].astype(np.float32) / 255.0
        _add_feature_prefix(row, channel_name, ch)

    hists = {"crop": _probability_histogram(crop_norm)}
    for channel_idx, channel_name in enumerate(["R", "G", "B"]):
        hists[channel_name] = _probability_histogram(processed_rgb[..., channel_idx].astype(np.float32) / 255.0)
    return row, hists


def _add_feature_prefix(row: dict[str, Any], prefix: str, arr: np.ndarray) -> None:
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    p1, p5, p25, p50, p75, p95, p99 = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite))
    row[f"{prefix}_p1"] = float(p1)
    row[f"{prefix}_p5"] = float(p5)
    row[f"{prefix}_p25"] = float(p25)
    row[f"{prefix}_p50"] = float(p50)
    row[f"{prefix}_p75"] = float(p75)
    row[f"{prefix}_p95"] = float(p95)
    row[f"{prefix}_p99"] = float(p99)
    row[f"{prefix}_iqr"] = float(p75 - p25)
    row[f"{prefix}_entropy"] = float(_normalized_entropy(finite))


def _pairwise_statistical_similarity(features_df: pd.DataFrame, histograms: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    final_feature_cols = [
        c for c in features_df.columns
        if any(c.startswith(prefix + "_") for prefix in ["R", "G", "B"])
    ]
    raw_crop_cols = [c for c in features_df.columns if c.startswith("crop_")]
    if len(features_df) < 2 or not final_feature_cols:
        return pd.DataFrame()

    final_matrix = features_df[final_feature_cols].astype(float).to_numpy()
    final_scale = _stable_feature_scale(final_matrix)

    crop_matrix = features_df[raw_crop_cols].astype(float).to_numpy() if raw_crop_cols else None
    crop_scale = _stable_feature_scale(crop_matrix) if crop_matrix is not None else None

    rows = []
    slots = features_df["slot"].tolist()
    for i in range(len(features_df)):
        for j in range(i + 1, len(features_df)):
            final_diff = np.abs((final_matrix[i] - final_matrix[j]) / final_scale)
            final_diff = final_diff[np.isfinite(final_diff)]
            final_mean_abs = float(np.mean(final_diff)) if final_diff.size else 0.0
            final_max_abs = float(np.max(final_diff)) if final_diff.size else 0.0

            final_js_vals = []
            final_w_vals = []
            for name in ["R", "G", "B"]:
                p = histograms[slots[i]][name]
                q = histograms[slots[j]][name]
                final_js_vals.append(_jensen_shannon_distance(p, q))
                final_w_vals.append(_wasserstein_distance_from_histograms(p, q))
            final_js_mean = float(np.mean(final_js_vals))
            final_w_mean = float(np.mean(final_w_vals))
            combined_final = float(np.mean([final_mean_abs, final_max_abs / 2.0, final_js_mean, final_w_mean]))

            raw_mean_abs = np.nan
            raw_js = np.nan
            raw_w = np.nan
            if crop_matrix is not None and crop_scale is not None:
                crop_diff = np.abs((crop_matrix[i] - crop_matrix[j]) / crop_scale)
                crop_diff = crop_diff[np.isfinite(crop_diff)]
                raw_mean_abs = float(np.mean(crop_diff)) if crop_diff.size else 0.0
                raw_js = _jensen_shannon_distance(histograms[slots[i]]["crop"], histograms[slots[j]]["crop"])
                raw_w = _wasserstein_distance_from_histograms(histograms[slots[i]]["crop"], histograms[slots[j]]["crop"])

            rows.append({
                "pair": f"{slots[i]} vs {slots[j]}",
                "vendor_pair": f"{features_df.iloc[i].get('vendor', 'Unknown')} vs {features_df.iloc[j].get('vendor', 'Unknown')}",
                "final_rgb_mean_abs_std_diff": final_mean_abs,
                "final_rgb_max_abs_std_diff": final_max_abs,
                "final_rgb_js_distance": final_js_mean,
                "final_rgb_wasserstein_distance": final_w_mean,
                "combined_final_rgb_distance": combined_final,
                "raw_crop_mean_abs_std_diff": raw_mean_abs,
                "raw_crop_js_distance": raw_js,
                "raw_crop_wasserstein_distance": raw_w,
            })
    return pd.DataFrame(rows).sort_values("combined_final_rgb_distance", ascending=True).reset_index(drop=True)


def _stable_feature_scale(matrix: np.ndarray) -> np.ndarray:
    scale = np.nanstd(matrix, axis=0)
    alt_scale = np.nanmean(np.abs(matrix - np.nanmean(matrix, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, alt_scale)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return scale


def _probability_histogram(arr: np.ndarray, bins: int = 64) -> np.ndarray:
    vals = _sample_pixels(np.asarray(arr, dtype=np.float32))
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)
    vals = np.clip(_normalize_minmax(vals), 0.0, 1.0)
    hist, _ = np.histogram(vals, bins=int(bins), range=(0.0, 1.0))
    hist = hist.astype(np.float64) + 1e-12
    return hist / hist.sum()


def _jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log2(p / np.maximum(m, 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2(q / np.maximum(m, 1e-12)), 0.0))
    return float(np.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))


def _wasserstein_distance_from_histograms(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    # Approximate 1-D Wasserstein distance on a fixed [0, 1] support by the area
    # between cumulative distribution functions.
    bin_width = 1.0 / max(len(p), 1)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * bin_width)


def _normalized_entropy(values: np.ndarray) -> float:
    hist = _probability_histogram(values, bins=64)
    entropy = -float(np.sum(hist * np.log2(np.maximum(hist, 1e-12))))
    return entropy / math.log2(max(len(hist), 2))


def _record_filter_controls(
    records_df: pd.DataFrame,
    *,
    prefix: str,
    compact: bool = False,
    default_vendor_mode: str = "all vendors",
    default_selected_vendors: list[str] | None = None,
    default_positive_choice: str = "positive only",
    default_split: str = "all",
) -> pd.DataFrame:
    # Always create an actual Streamlit container object. The `streamlit` module
    # itself is not a context manager, so `with st:` raises
    # TypeError: 'module' object does not support the context manager protocol.
    ui_container = st.container(border=bool(compact))
    with ui_container:
        split_options = ["all", "train", "val", "test"]
        split_index = split_options.index(default_split) if default_split in split_options else 0
        split_choice = st.selectbox("Split", split_options, index=split_index, key=f"{prefix}_split")
        image_options = ["positive only", "all images"]
        positive_index = image_options.index(default_positive_choice) if default_positive_choice in image_options else 0
        positive_choice = st.radio("Images", image_options, index=positive_index, horizontal=True, key=f"{prefix}_positive")
        vendors = _available_vendors(records_df)
        vendor_modes = ["all vendors", "selected vendors"]
        vendor_mode_index = vendor_modes.index(default_vendor_mode) if default_vendor_mode in vendor_modes else 0
        vendor_mode = st.radio("Vendor filter", vendor_modes, index=vendor_mode_index, horizontal=True, key=f"{prefix}_vendor_mode")
        selected_vendors: list[str] = []
        st.caption(f"Vendor options available: {len(vendors)}")
        if vendor_mode == "selected vendors":
            vendor_key = f"{prefix}_vendors"
            requested_defaults = [v for v in (default_selected_vendors or []) if v in vendors]
            if not requested_defaults and vendors:
                requested_defaults = vendors[:1]
            if vendor_key in st.session_state:
                # Drop stale selections when another filter or a rerun changes the option list.
                st.session_state[vendor_key] = [v for v in st.session_state[vendor_key] if v in vendors]
            if vendors:
                selected_vendors = st.multiselect(
                    "Vendors",
                    options=vendors,
                    default=requested_defaults,
                    key=vendor_key,
                    help="Vendor values come from metadata.csv Manufacturer and Manufacturer's Model Name when available.",
                )
                with st.expander("Vendor counts", expanded=False):
                    counts = records_df["vendor"].fillna("Unknown").replace("", "Unknown").value_counts().reset_index()
                    counts.columns = ["vendor", "num_images"]
                    st.dataframe(counts, hide_index=True, use_container_width=True)
            else:
                st.warning(
                    "No vendor values were found in metadata.csv. The dataset will still work, "
                    "but vendor filtering cannot be used until Manufacturer/Model columns are available."
                )

    df = records_df.copy()
    if split_choice != "all":
        df = df[df["export_split"] == split_choice]
    if positive_choice == "positive only":
        df = df[df["has_mass"] == True]  # noqa: E712
    if vendor_mode == "selected vendors" and selected_vendors:
        df = df[df["vendor"].isin(selected_vendors)]
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Contralateral pairing
# -----------------------------------------------------------------------------


def _opposite_laterality(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        return "R"
    if text.startswith("R"):
        return "L"
    return None


def _find_contralateral_record_index(dataset: VindrMammoDataset, record: dict[str, Any]) -> int | None:
    opposite = _opposite_laterality(record.get("laterality"))
    if opposite is None:
        return None
    study_id = str(record.get("study_id", ""))
    view = str(record.get("view_position", "")).upper().strip()
    current_image_id = str(record.get("image_id", ""))
    for idx, candidate in enumerate(dataset.image_records):
        if str(candidate.get("image_id", "")) == current_image_id:
            continue
        if str(candidate.get("study_id", "")) != study_id:
            continue
        if str(candidate.get("view_position", "")).upper().strip() != view:
            continue
        if str(candidate.get("laterality", "")).upper().strip().startswith(opposite):
            return int(idx)
    return None


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


def _prepare_sample(
    dataset: VindrMammoDataset,
    record_index: int,
    crop_controls: dict[str, Any],
    crop_index: int | None,
    *,
    need_contralateral: bool = False,
) -> dict[str, Any]:
    loaded = _read_preprocessed_cached(dataset, _dataset_cache_key(dataset), int(record_index))
    image = loaded["image"]
    boxes = torch.as_tensor(loaded["all_boxes"], dtype=torch.float32)
    mass_boxes = torch.as_tensor(loaded["mass_boxes"], dtype=torch.float32)
    height, width = image.shape
    if crop_controls.get("mode") == "random":
        rng = np.random.default_rng(int(crop_controls.get("random_seed", 123)) + int(record_index))
        windows = []
        random_options = dict(crop_controls["crop_options"])
        random_options["mode"] = "random"
        for _ in range(int(crop_controls.get("random_preview_count", 20))):
            w, _info = sample_random_square_window(
                image_width=width,
                image_height=height,
                mass_boxes=mass_boxes,
                options=random_options,
                rng=rng,
            )
            windows.append(w)
    else:
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
        foreground_fraction = None
        if bool(crop_controls.get("require_foreground", False)):
            foreground_fraction = _foreground_fraction_in_window(
                image,
                w,
                crop_size=int(crop_controls["crop_size"]),
                threshold=crop_controls.get("foreground_threshold"),
                pad_value=float(crop_controls["crop_options"].get("pad_value", 0.0)),
            )
            if foreground_fraction < float(crop_controls.get("min_foreground_fraction", 0.05)):
                continue
        crops.append({
            "window": w,
            "max_visibility": max_vis,
            "positive_by_slider": is_positive,
            "foreground_fraction": foreground_fraction,
        })

    selected = None
    crop_image = None
    crop_boxes = np.zeros((0, 4), dtype=np.float32)
    crop_mass_boxes = np.zeros((0, 4), dtype=np.float32)
    foreground_mask_crop = None
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
        if bool(crop_controls.get("foreground_mask_preview", False)):
            foreground_mask_crop = _foreground_mask_for_crop(
                crop_image,
                threshold=crop_controls.get("foreground_threshold"),
            )

    contralateral_crop_image = None
    contralateral_info: dict[str, Any] = {"requested": bool(need_contralateral), "found": False}
    if need_contralateral and selected is not None and crop_image is not None:
        contralateral_index = _find_contralateral_record_index(dataset, loaded["record"])
        contralateral_info["record_index"] = contralateral_index
        if contralateral_index is not None:
            paired = _read_preprocessed_cached(dataset, _dataset_cache_key(dataset), int(contralateral_index))
            paired_image = paired["image"]
            image_tensor = torch.from_numpy(np.ascontiguousarray(paired_image)).unsqueeze(0)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            paired_crop_result = crop_image_and_boxes_to_window(
                image_tensor,
                boxes=empty_boxes,
                mass_boxes=empty_boxes,
                window_xyxy=selected["window"],
                options=crop_controls["crop_options"],
            )
            contralateral_crop_image = paired_crop_result.image.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
            paired_summary = paired.get("target_summary", {}) or {}
            contralateral_info.update({
                "found": True,
                "image_id": paired_summary.get("image_id"),
                "laterality": paired_summary.get("laterality"),
                "view_position": paired_summary.get("view_position"),
            })
        else:
            # Keep the display/export robust. The metadata makes the fallback explicit.
            contralateral_crop_image = crop_image.copy()
            contralateral_info["fallback"] = "current_crop"

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
        "foreground_mask_crop": foreground_mask_crop,
        "show_foreground_mask_preview": bool(crop_controls.get("foreground_mask_preview", False)),
        "contralateral_crop_image": contralateral_crop_image,
        "contralateral_info": contralateral_info,
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
    display_controls: dict[str, Any] | None = None,
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

    processed_rgb, processing_meta = apply_channel_pipeline(crop, pipeline, source_crops=_source_crops_from_result(result))
    display_controls = display_controls or {"visible_channels": ["R", "G", "B"], "show_channel_panels": False}
    visible_channels = display_controls.get("visible_channels", ["R", "G", "B"]) or []
    processed_rgb_display = _mask_rgb_channels(processed_rgb, visible_channels)
    crop_gray = _to_uint8_percentile(crop, display_window)
    full_gray = _to_uint8_percentile(full, display_window)

    # Draw crop window on full image and boxes on all image views.
    full_draw = _draw_boxes(_gray_to_rgb(full_gray), full_boxes, color=(255, 80, 80))
    if window is not None:
        full_draw = _draw_rect(full_draw, window, color=(80, 255, 80), thickness=max(2, full.shape[1] // 1000))
    crop_draw = _draw_boxes(_gray_to_rgb(crop_gray), crop_boxes, color=(255, 80, 80))
    proc_draw = _draw_boxes(processed_rgb_display.copy(), crop_boxes, color=(255, 80, 80))

    st.write(result["title"])
    if window is not None:
        st.caption(f"Selected crop window xyxy={tuple(int(v) for v in window)} | max mass visibility={selected.get('max_visibility', 0.0):.3f}")

    cols = st.columns(3)
    cols[0].image(full_draw, caption="Fixed-preprocessed grayscale image with selected crop window", use_container_width=True)
    cols[1].image(crop_draw, caption="Crop from fixed-preprocessed image", use_container_width=True)
    cols[2].image(proc_draw, caption=f"Preprocessed RGB crop, visible channels={''.join(visible_channels) or 'none'}", use_container_width=True)

    if display_controls.get("show_channel_panels", False):
        ch_cols = st.columns(3)
        for i, name in enumerate(["R", "G", "B"]):
            channel_u8 = processed_rgb[..., i]
            channel_draw = _gray_to_rgb(channel_u8)
            if show_annotations:
                channel_draw = _draw_boxes(channel_draw, crop_boxes, color=(255, 80, 80))
            ch_cols[i].image(
                channel_draw,
                caption=f"Processed {name} channel" + (" with boxes" if show_annotations else ""),
                use_container_width=True,
                clamp=True,
            )

    if (result.get("foreground_mask_crop") is not None) and result.get("show_foreground_mask_preview", False):
        fg_cols = st.columns(2)
        fg_cols[0].image(result["foreground_mask_crop"].astype(np.uint8) * 255, caption="Foreground mask used for crop filtering", use_container_width=True, clamp=True)
        fg_cols[1].write({"foreground_fraction": result.get("selected_crop", {}).get("foreground_fraction")})

    with st.expander("Metadata and statistics", expanded=not compact):
        stat_df = _stats_table(full, crop, processed_rgb)
        st.dataframe(stat_df, use_container_width=True)
        st.json(_compact_metadata(result["target_summary"], processing_meta))
        st.caption(
            "The old pixel-intensity distribution plot was removed. Use compare mode for "
            "numeric distribution distances and standardized statistic differences."
        )



def _mask_rgb_channels(rgb: np.ndarray, visible_channels: list[str]) -> np.ndarray:
    out = np.zeros_like(rgb)
    for idx, name in enumerate(["R", "G", "B"]):
        if name in visible_channels:
            out[..., idx] = rgb[..., idx]
    return out


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
    p1, p5, p25, p50, p75, p95, p99 = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "image": name,
        "shape": "x".join(map(str, arr.shape)),
        "dtype": str(arr.dtype),
        "min": float(np.min(finite)),
        "p1": float(p1),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "iqr": float(p75 - p25),
        "entropy": float(_normalized_entropy(finite)),
    }


def _histogram_figure(full: np.ndarray, crop: np.ndarray, processed_rgb: np.ndarray):
    """Plot comparable pixel intensity distributions.

    Every plotted series is independently min-max normalized to the x-axis range
    [0, 1]. Histogram heights are also normalized to relative frequency so the
    full image, crop, and RGB channels can be compared even though they contain
    different numbers of pixels.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0.0, 1.0, 81)

    _plot_normalized_hist(ax, full, bins=bins, alpha=0.35, label="full grayscale")
    _plot_normalized_hist(ax, crop, bins=bins, alpha=0.35, label="crop grayscale")
    for i, name in enumerate(["R", "G", "B"]):
        _plot_normalized_hist(ax, processed_rgb[..., i], bins=bins, alpha=0.25, label=f"processed {name}")

    ax.set_title("Pixel intensity distributions, each series normalized to [0, 1]")
    ax.set_xlabel("Normalized intensity, per image/channel")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_normalized_hist(ax, arr: np.ndarray, *, bins: np.ndarray, alpha: float, label: str) -> None:
    values = _sample_pixels(arr).astype(np.float32, copy=False)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    values = _normalize_minmax(values)
    weights = np.ones(values.shape, dtype=np.float32) / max(int(values.size), 1)
    ax.hist(values, bins=bins, weights=weights, alpha=alpha, label=label)


# -----------------------------------------------------------------------------
# Channel preprocessing operations
# -----------------------------------------------------------------------------


def _source_crops_from_result(result: dict[str, Any]) -> dict[str, np.ndarray]:
    crop = result.get("crop_image")
    contralateral = result.get("contralateral_crop_image")
    out: dict[str, np.ndarray] = {"current_crop": crop}
    if contralateral is not None:
        out["contralateral_same_view_crop"] = contralateral
    return out


def apply_channel_pipeline(
    crop_float: np.ndarray,
    pipeline: dict[str, Any],
    *,
    source_crops: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_crops = dict(source_crops or {})
    source_crops.setdefault("current_crop", crop_float)
    channels = []
    meta = {"channels": {}}
    for channel in ["R", "G", "B"]:
        source_name = _channel_source(pipeline, channel)
        if source_name not in source_crops or source_crops.get(source_name) is None:
            # Robust fallback: if the paired opposite breast is missing, keep the GUI/export usable.
            source_name_used = "current_crop"
            arr = np.asarray(source_crops["current_crop"], dtype=np.float32).copy()
            source_fallback = True
        else:
            source_name_used = source_name
            arr = np.asarray(source_crops[source_name], dtype=np.float32).copy()
            source_fallback = False

        # Compute statistics over the visible breast/foreground pixels by default.
        # This matters when the fixed breast crop is disabled: otherwise black
        # background can dominate low percentile windows, for example [0, 10],
        # and make the operation appear unchanged.
        stat_mask = _operation_stat_mask(arr)

        applied = []
        for step in _channel_steps(pipeline, channel):
            op = step.get("op", "none")
            params = step.get("params", {}) or {}
            if op == "none":
                continue
            arr = _apply_operation(arr, op, params, stat_mask=stat_mask)
            # Keep the mask tied to the same physical foreground region after
            # point-wise transforms; recompute only if the shape changes.
            if stat_mask.shape != arr.shape:
                stat_mask = _operation_stat_mask(arr)
            applied.append({"op": op, "params": params})
        ch = _float_to_uint8(arr)
        channels.append(ch)
        meta["channels"][channel] = {
            "source_requested": source_name,
            "source_used": source_name_used,
            "source_fallback": source_fallback,
            "steps": applied,
        }
    return np.stack(channels, axis=-1).astype(np.uint8, copy=False), meta

def _apply_operation(arr: np.ndarray, op: str, params: dict[str, Any], stat_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if stat_mask is not None and stat_mask.shape != arr.shape:
        stat_mask = None
    if op == "percentile_normalize":
        return _normalize_percentile(arr, params.get("percentiles", [1.0, 99.0]), stat_mask)
    if op == "percentile_clip_only":
        lo, hi = _safe_percentile(arr, params.get("percentiles", [1.0, 99.0]), stat_mask)
        return np.clip(arr, lo, hi).astype(np.float32)
    if op == "zscore_clip":
        pixels = arr[stat_mask] if stat_mask is not None and stat_mask.any() else arr[np.isfinite(arr)]
        if pixels.size == 0:
            pixels = arr[np.isfinite(arr)]
        m = float(np.mean(pixels)) if pixels.size else 0.0
        s = float(np.std(pixels)) or 1.0
        z = (arr - m) / max(s, 1e-12)
        limit = float(params.get("z_limit", 3.0))
        z = np.clip(z, -limit, limit)
        return ((z + limit) / max(2 * limit, 1e-12)).astype(np.float32)
    if op == "standardize_to_target":
        return _standardize_to_target(arr, params, stat_mask)
    if op == "aggressive_upper_percentile_normalize":
        return _normalize_percentile(arr, params.get("percentiles", [70.0, 100.0]), stat_mask)
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
        return _sobel(arr, params, stat_mask)
    if op == "laplacian":
        return _laplacian(arr, params, stat_mask)
    if op == "gamma":
        gamma = max(float(params.get("gamma", 1.0)), 1e-6)
        return np.power(np.clip(_normalize_minmax(arr, stat_mask), 0.0, 1.0), gamma).astype(np.float32)
    if op == "log":
        gain = float(params.get("gain", 5.0))
        x = np.clip(_normalize_minmax(arr, stat_mask), 0.0, 1.0)
        return (np.log1p(gain * x) / np.log1p(gain)).astype(np.float32)
    if op == "invert":
        return 1.0 - _normalize_minmax(arr, stat_mask)
    return arr


def _standardize_to_target(arr: np.ndarray, params: dict[str, Any], stat_mask: np.ndarray | None = None) -> np.ndarray:
    """Dynamic affine standardization: y = a*x + b.

    a and b are chosen from the current image/channel statistics so the output
    has approximately the requested mean and standard deviation. By default,
    the output is clipped to [0, 1] so the following uint8 conversion preserves
    this chosen scale instead of applying another percentile remapping.
    """
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    pixels = x[stat_mask] if stat_mask is not None and stat_mask.any() else finite
    if pixels.size == 0:
        pixels = finite

    stat_percentiles = params.get("stat_percentiles", [1.0, 99.0])
    try:
        lo, hi = _safe_percentile(pixels, stat_percentiles)
        stat_pixels = pixels[(pixels >= lo) & (pixels <= hi)]
    except Exception:
        stat_pixels = pixels
    if stat_pixels.size < 2:
        stat_pixels = pixels

    current_mean = float(np.mean(stat_pixels))
    current_std = float(np.std(stat_pixels))
    target_mean = float(params.get("target_mean", 0.5))
    target_std = max(float(params.get("target_std", 0.2)), 1e-8)
    a = target_std / max(current_std, 1e-8)
    b = target_mean - a * current_mean
    y = (a * x + b).astype(np.float32)

    if bool(params.get("clip_output", True)):
        y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y

def _sobel(arr: np.ndarray, params: dict[str, Any], stat_mask: np.ndarray | None = None) -> np.ndarray:
    if cv2 is None:
        gy, gx = np.gradient(arr.astype(np.float32))
        mag = np.sqrt(gx * gx + gy * gy)
    else:
        u8 = _float_to_uint8(arr)
        k = _odd_int(params.get("ksize", 3))
        gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=k)
        mag = cv2.magnitude(gx, gy)
    return _normalize_percentile(mag.astype(np.float32), params.get("percentiles", [1.0, 99.0]), stat_mask)


def _laplacian(arr: np.ndarray, params: dict[str, Any], stat_mask: np.ndarray | None = None) -> np.ndarray:
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



def _foreground_fraction_in_window(
    image: np.ndarray,
    window_xyxy: tuple[int, int, int, int],
    *,
    crop_size: int,
    threshold: float | None,
    pad_value: float = 0.0,
) -> float:
    x0, y0, x1, y1 = [int(v) for v in window_xyxy]
    h, w = image.shape
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    crop = np.full((int(crop_size), int(crop_size)), float(pad_value), dtype=np.float32)
    patch = image[src_y0:src_y1, src_x0:src_x1]
    if patch.size:
        dst_x0 = max(0, -x0)
        dst_y0 = max(0, -y0)
        crop[dst_y0:dst_y0 + patch.shape[0], dst_x0:dst_x0 + patch.shape[1]] = patch
    mask = _foreground_mask_for_crop(crop, threshold=threshold)
    return float(mask.mean()) if mask.size else 0.0


def _foreground_mask_for_crop(arr: np.ndarray, *, threshold: float | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=bool)
    if threshold is None:
        vals = arr[finite]
        lo, hi = np.percentile(vals, [1.0, 99.0])
        threshold = max(float(lo + 0.03 * (hi - lo)), float(lo) + 1e-6)
    mask = finite & (arr > float(threshold))
    if mask.sum() < max(10, int(0.001 * arr.size)):
        mask = finite
    return mask


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


def _operation_stat_mask(arr: np.ndarray) -> np.ndarray:
    """Return the default pixel mask for contrast/statistics operations.

    Percentile operations should normally ignore the black mammogram background,
    especially when global breast cropping is disabled. Otherwise low percentile
    ranges such as [0, 10] often collapse to [0, 0] because the crop contains a
    large black background area.
    """
    try:
        return _foreground_mask_for_crop(np.asarray(arr, dtype=np.float32), threshold=None)
    except Exception:
        return np.isfinite(arr)


def _safe_percentile(arr: np.ndarray, percentiles: list[float] | tuple[float, float], mask: np.ndarray | None = None) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    if mask is not None and mask.shape == arr.shape and mask.any():
        finite = arr[mask & np.isfinite(arr)]
    else:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    p0, p1 = float(percentiles[0]), float(percentiles[1])
    # Allow user-facing fractional notation: [0.7, 1.0] means [70, 100] percentiles.
    if 0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0:
        p0, p1 = 100.0 * p0, 100.0 * p1
    lo, hi = np.percentile(finite, [p0, p1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize_percentile(arr: np.ndarray, percentiles: list[float] | tuple[float, float], mask: np.ndarray | None = None) -> np.ndarray:
    lo, hi = _safe_percentile(arr, percentiles, mask)
    return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-12)).astype(np.float32)


def _normalize_minmax(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if mask is not None and mask.shape == arr.shape and mask.any():
        finite = arr[mask & np.isfinite(arr)]
    else:
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



_VENDOR_MANUFACTURER_KEYS = [
    "Manufacturer",
    "manufacturer",
    "manufacturers",
    "ManufacturerName",
    "manufacturer_name",
    "Manufacturer's Name",
    "Manufacturer’s Name",
    "0008,0070",
    "(0008,0070)",
]

_VENDOR_MODEL_KEYS = [
    "ManufacturerModelName",
    "Manufacturer's Model Name",
    "Manufacturer’s Model Name",
    "manufacturer_model_name",
    "manufacturer_model",
    "model_name",
    "model",
    "ModelName",
    "0008,1090",
    "(0008,1090)",
]

_METADATA_IMAGE_ID_KEYS = [
    "image_id",
    "ImageID",
    "imageId",
    "SOPInstanceUID",
    "sop_instance_uid",
    "SOP Instance UID",
    "filename",
    "file_name",
    "FileName",
    "dicom_path",
    "path",
]


def _available_vendors(records_df: pd.DataFrame) -> list[str]:
    if "vendor" not in records_df.columns:
        return []
    vendors: list[str] = []
    for value in records_df["vendor"].tolist():
        cleaned = _clean_scalar(value)
        if cleaned is None:
            continue
        vendors.append(cleaned)
    vendors = sorted(set(vendors))
    # If vendor information exists only as missing metadata, still expose Unknown so
    # the selector is not empty and the user can tell the GUI is functioning.
    if not vendors and len(records_df) > 0:
        vendors = ["Unknown"]
    return vendors


def _default_comparison_vendors(records_df: pd.DataFrame, n_slots: int) -> list[str]:
    """Choose default compare-slot vendors, preferring positive images and diversity.

    The comparison tab is mainly used for cross-device debugging. By default we
    therefore assign each visible slot to a different vendor/device when possible.
    Vendors with mass-positive records are preferred because the default image
    filter is also positive-only.
    """
    if "vendor" not in records_df.columns or records_df.empty:
        return []
    df = records_df.copy()
    df["vendor"] = df["vendor"].fillna("Unknown").replace("", "Unknown")
    if "has_mass" in df.columns:
        positive_df = df[df["has_mass"] == True]  # noqa: E712
        if not positive_df.empty:
            df = positive_df
    counts = df["vendor"].value_counts()
    vendors = [str(v) for v in counts.index.tolist()]
    if len(vendors) > 1:
        known = [v for v in vendors if v != "Unknown"]
        unknown = [v for v in vendors if v == "Unknown"]
        vendors = known + unknown
    return vendors[: max(0, int(n_slots))]


def _build_vendor_maps(
    metadata_by_image_id: dict[str, list[dict[str, Any]]],
    metadata_table_rows: list[dict[str, Any]],
    records: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Build image_id -> vendor maps robustly across metadata.csv variants.

    Some VinDr-Mammo copies expose columns as `Manufacturer` and
    `Manufacturer's Model Name`, while others use normalized names such as
    `manufacturer_model_name`. This helper accepts both and also falls back to
    common DICOM/SOP UID style image-id columns.
    """
    image_ids = {str(x) for x in records["image_id"].astype(str).tolist()} if "image_id" in records else set()
    vendor_map: dict[str, str] = {}
    meta_preview_map: dict[str, dict[str, Any]] = {}

    def update(image_id: Any, row: dict[str, Any]) -> None:
        iid = _clean_image_id(image_id)
        if iid is None or (image_ids and iid not in image_ids):
            return
        vendor = _vendor_from_row(row)
        vendor_map[iid] = vendor
        meta_preview_map[iid] = row

    # First use the Dataset's direct grouping when metadata.csv has image_id.
    for image_id, rows in (metadata_by_image_id or {}).items():
        row = rows[0] if rows else {}
        update(image_id, row)

    # Then scan the full metadata table. This catches alternative image-id column
    # names and also improves vendor extraction when metadata_by_image_id was empty.
    for row in metadata_table_rows or []:
        image_id = _metadata_row_image_id(row)
        if image_id is not None:
            update(image_id, row)

    # Last resort: if a metadata row has study_id + view/laterality but no image_id,
    # match it to a unique record with the same tuple.
    tuple_to_image: dict[tuple[str, str, str], str] = {}
    duplicate_keys: set[tuple[str, str, str]] = set()
    for _, record in records.iterrows():
        key = (
            _clean_scalar(record.get("study_id")) or "",
            _clean_scalar(record.get("laterality")) or "",
            _clean_scalar(record.get("view_position")) or "",
        )
        if not all(key):
            continue
        if key in tuple_to_image:
            duplicate_keys.add(key)
        else:
            tuple_to_image[key] = str(record.get("image_id"))
    for key in duplicate_keys:
        tuple_to_image.pop(key, None)

    for row in metadata_table_rows or []:
        if _metadata_row_image_id(row) is not None:
            continue
        key = (
            _clean_scalar(_first_existing(row, ["study_id", "StudyID", "StudyInstanceUID"])) or "",
            _clean_scalar(_first_existing(row, ["laterality", "Laterality", "ImageLaterality", "Image Laterality"])) or "",
            _clean_scalar(_first_existing(row, ["view_position", "ViewPosition", "View Position"])) or "",
        )
        image_id = tuple_to_image.get(key)
        if image_id is not None:
            update(image_id, row)

    # Fill missing records with Unknown, otherwise the multiselect can become empty.
    for image_id in image_ids:
        vendor_map.setdefault(image_id, "Unknown")
    return vendor_map, meta_preview_map


def _metadata_row_image_id(row: dict[str, Any]) -> str | None:
    value = _first_existing(row, _METADATA_IMAGE_ID_KEYS)
    return _clean_image_id(value)


def _clean_image_id(value: Any) -> str | None:
    text = _clean_scalar(value)
    if text is None:
        return None
    # If a path is supplied, use the file stem because VinDr image_id is the DICOM filename stem.
    text = text.replace("\\", "/")
    if "/" in text:
        text = Path(text).stem
    if text.endswith(".dicom") or text.endswith(".dcm"):
        text = Path(text).stem
    return text or None


def _vendor_from_row(row: dict[str, Any]) -> str:
    manufacturer = _first_existing(row, _VENDOR_MANUFACTURER_KEYS)
    model = _first_existing(row, _VENDOR_MODEL_KEYS)
    parts = [_clean_scalar(x) for x in [manufacturer, model]]
    parts = [x for x in parts if x]
    return " / ".join(parts) if parts else "Unknown"


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
    vendor = _vendor_from_row(meta)
    if vendor == "Unknown":
        vendor = _vendor_from_row(dicom)
    return vendor


def _first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    if not isinstance(row, dict):
        return None
    # First try exact column names.
    for key in keys:
        if key in row:
            cleaned = _clean_scalar(row[key])
            if cleaned is not None:
                return cleaned
    # Then try normalized names so variants like "Manufacturer's Model Name" and
    # "manufacturer_model_name" can match each other.
    normalized_row = {_normalize_key(k): v for k, v in row.items()}
    for key in keys:
        value = normalized_row.get(_normalize_key(key))
        cleaned = _clean_scalar(value)
        if cleaned is not None:
            return cleaned
    return None


def _normalize_key(value: Any) -> str:
    text = str(value).strip().lower()
    for ch in ["\'", "’", "`", "\"", "(", ")", ",", ":", ";", "-", "/"]:
        text = text.replace(ch, " ")
    return "_".join(text.split())


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return text


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
