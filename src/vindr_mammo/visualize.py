from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Use a non-interactive backend so this works from VSCode, terminals, and servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


SPLIT_ORDER = ["train", "val", "test"]
DATASET_ORDER = ["square_crops", "baseline_uncropped"]

# COCO evaluation area ranges for bbox AP. In pycocotools these are
# [0, 32^2], [32^2, 96^2], [96^2, 1e5^2]. For dataset summaries we assign
# each box to exactly one non-overlapping bin using sqrt(area):
# small < 32 px, medium 32 <= sqrt(area) < 96 px, large >= 96 px.
COCO_SMALL_SIDE = 32.0
COCO_LARGE_SIDE = 96.0
COCO_SMALL_AREA = COCO_SMALL_SIDE ** 2
COCO_LARGE_AREA = COCO_LARGE_SIDE ** 2


@dataclass
class VisualizationResult:
    output_dir: Path
    created_files: list[Path]
    summary: dict[str, Any]


def create_annotation_geometry_report(
    annotations: pd.DataFrame | list[dict[str, Any]],
    *,
    output_dir: str | Path,
    crop_width: int,
    crop_height: int,
    histogram_bins: int = 40,
) -> VisualizationResult:
    """Write source-annotation size data and crop-fit visualizations.

    Fit is deliberately geometry-only: a box can fit when its width is no
    greater than ``crop_width`` and its height is no greater than
    ``crop_height``. Existing annotation position and actual sliding-window
    origins are ignored, as they should be for this diagnostic.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(annotations).copy()
    required = ["bbox_width_px", "bbox_height_px"]
    for column in required:
        if column not in data.columns:
            data[column] = pd.Series(dtype=float)
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        data["bbox_width_px"].notna()
        & data["bbox_height_px"].notna()
        & (data["bbox_width_px"] > 0)
        & (data["bbox_height_px"] > 0)
    ].copy()
    crop_width = max(1, int(crop_width))
    crop_height = max(1, int(crop_height))
    histogram_bins = max(5, int(histogram_bins))
    data["bbox_area_px2"] = data["bbox_width_px"] * data["bbox_height_px"]
    data["bbox_max_side_px"] = data[["bbox_width_px", "bbox_height_px"]].max(axis=1)
    data["crop_width_px"] = crop_width
    data["crop_height_px"] = crop_height
    data["can_fit_fully_by_size"] = (
        (data["bbox_width_px"] <= crop_width)
        & (data["bbox_height_px"] <= crop_height)
    )
    data["cannot_fit_reason"] = np.select(
        [
            (data["bbox_width_px"] > crop_width) & (data["bbox_height_px"] > crop_height),
            data["bbox_width_px"] > crop_width,
            data["bbox_height_px"] > crop_height,
        ],
        ["too_wide_and_too_tall", "too_wide", "too_tall"],
        default="fits",
    )

    created: list[Path] = []
    detail_path = output_dir / "mass_box_geometry.csv"
    data.to_csv(detail_path, index=False)
    created.append(detail_path)

    summary_rows = [_annotation_fit_summary_row(data, "all", crop_width, crop_height)]
    if "split" in data.columns:
        for split in SPLIT_ORDER:
            summary_rows.append(
                _annotation_fit_summary_row(
                    data[data["split"].astype(str) == split],
                    split,
                    crop_width,
                    crop_height,
                )
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "mass_box_fit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    created.append(summary_csv)

    overall = summary_rows[0]
    summary_payload = {
        "definition": (
            "geometry_only; annotation location and generated crop locations are ignored; "
            "fits iff bbox_width_px <= crop_width_px and bbox_height_px <= crop_height_px"
        ),
        "coordinate_space": "fixed_preprocessed_source",
        "crop_width_px": crop_width,
        "crop_height_px": crop_height,
        "histogram_bins": histogram_bins,
        "overall": overall,
        "by_split": summary_rows[1:],
    }
    summary_json = output_dir / "mass_box_fit_summary.json"
    summary_json.write_text(
        json.dumps(_json_safe(summary_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    created.append(summary_json)

    if not data.empty:
        created.extend(
            _plot_annotation_geometry_figures(
                data,
                output_dir,
                crop_width=crop_width,
                crop_height=crop_height,
                histogram_bins=histogram_bins,
            )
        )

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _annotation_geometry_readme(summary_payload),
        encoding="utf-8",
    )
    created.append(readme_path)
    html_path = _write_annotation_geometry_html(output_dir, created, summary_payload)
    created.append(html_path)
    return VisualizationResult(
        output_dir=output_dir,
        created_files=created,
        summary=summary_payload,
    )


def _annotation_fit_summary_row(
    data: pd.DataFrame,
    scope: str,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    total = int(len(data))
    fit = int(data.get("can_fit_fully_by_size", pd.Series(dtype=bool)).astype(bool).sum())
    reasons = data.get("cannot_fit_reason", pd.Series(dtype=str)).value_counts().to_dict()
    return {
        "scope": scope,
        "crop_width_px": int(crop_width),
        "crop_height_px": int(crop_height),
        "total_mass_annotations": total,
        "can_fit_fully_by_size": fit,
        "cannot_fit_fully_by_size": total - fit,
        "can_fit_percent": (100.0 * fit / total) if total else 0.0,
        "cannot_fit_percent": (100.0 * (total - fit) / total) if total else 0.0,
        "too_wide": int(reasons.get("too_wide", 0)),
        "too_tall": int(reasons.get("too_tall", 0)),
        "too_wide_and_too_tall": int(reasons.get("too_wide_and_too_tall", 0)),
        "median_width_px": float(data["bbox_width_px"].median()) if total else None,
        "median_height_px": float(data["bbox_height_px"].median()) if total else None,
        "p95_max_side_px": float(data["bbox_max_side_px"].quantile(0.95)) if total else None,
        "maximum_width_px": float(data["bbox_width_px"].max()) if total else None,
        "maximum_height_px": float(data["bbox_height_px"].max()) if total else None,
    }


def _plot_annotation_geometry_figures(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    crop_width: int,
    crop_height: int,
    histogram_bins: int,
) -> list[Path]:
    created: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plots = [
        ("bbox_width_px", "Mass box widths", "width (pixels)", crop_width),
        ("bbox_height_px", "Mass box heights", "height (pixels)", crop_height),
        ("bbox_max_side_px", "Mass box maximum side", "max(width, height) (pixels)", max(crop_width, crop_height)),
        ("bbox_area_px2", "Mass box areas", "area (pixels²)", crop_width * crop_height),
    ]
    for ax, (column, title, xlabel, threshold) in zip(axes.ravel(), plots, strict=True):
        ax.hist(data[column], bins=histogram_bins, color="#3974b9", alpha=0.82)
        ax.axvline(float(threshold), color="#c83f49", linestyle="--", linewidth=2, label="crop bound")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mass annotations")
        ax.legend(loc="best")
    p = output_dir / "mass_box_size_histograms.png"
    _savefig(fig, p)
    created.append(p)

    fig, ax = plt.subplots(figsize=(9, 7))
    fit_mask = data["can_fit_fully_by_size"].astype(bool)
    ax.scatter(
        data.loc[fit_mask, "bbox_width_px"],
        data.loc[fit_mask, "bbox_height_px"],
        s=18,
        alpha=0.45,
        label="can fit by size",
        color="#2d8a57",
    )
    ax.scatter(
        data.loc[~fit_mask, "bbox_width_px"],
        data.loc[~fit_mask, "bbox_height_px"],
        s=24,
        alpha=0.65,
        label="cannot fit by size",
        color="#c83f49",
    )
    ax.axvline(crop_width, color="#333333", linestyle="--", linewidth=1.5)
    ax.axhline(crop_height, color="#333333", linestyle="--", linewidth=1.5)
    ax.set_title(f"Mass box width vs height for a {crop_width} × {crop_height} crop")
    ax.set_xlabel("box width (pixels)")
    ax.set_ylabel("box height (pixels)")
    ax.legend(loc="best")
    p = output_dir / "mass_box_width_height_crop_fit.png"
    _savefig(fig, p)
    created.append(p)

    scopes = [("all", data)]
    if "split" in data.columns:
        scopes.extend((split, data[data["split"].astype(str) == split]) for split in SPLIT_ORDER)
    labels = [scope for scope, _group in scopes]
    fit_counts = [int(group["can_fit_fully_by_size"].astype(bool).sum()) for _scope, group in scopes]
    no_fit_counts = [int(len(group) - fit) for fit, (_scope, group) in zip(fit_counts, scopes, strict=True)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, fit_counts, label="can fit fully by size", color="#2d8a57")
    ax.bar(x, no_fit_counts, bottom=fit_counts, label="cannot fit fully by size", color="#c83f49")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mass annotations")
    ax.set_title("Geometric crop-fit counts (annotation location ignored)")
    ax.legend(loc="best")
    p = output_dir / "mass_box_crop_fit_counts.png"
    _savefig(fig, p)
    created.append(p)
    return created


def _annotation_geometry_readme(summary: dict[str, Any]) -> str:
    overall = dict(summary.get("overall", {}) or {})
    return f"""# Annotation geometry report

This report analyzes source Mass bounding-box dimensions in the fixed-preprocessed coordinate space.

The test is deliberately independent of crop placement. For the configured `{summary.get('crop_width_px')} x {summary.get('crop_height_px')}` crop, an annotation can fit fully by size when both its width and height are within those crop bounds. It does **not** claim that a particular generated sliding window contains the annotation.

- Total Mass annotations: {int(overall.get('total_mass_annotations', 0))}
- Can fit fully by size: {int(overall.get('can_fit_fully_by_size', 0))} ({float(overall.get('can_fit_percent', 0.0)):.2f}%)
- Cannot fit fully by size: {int(overall.get('cannot_fit_fully_by_size', 0))} ({float(overall.get('cannot_fit_percent', 0.0)):.2f}%)

Files:

- `mass_box_geometry.csv`: one row per source Mass annotation with dimensions, area, fit flag, and failure reason.
- `mass_box_fit_summary.csv` and `.json`: overall and split-level counts and size statistics.
- `mass_box_size_histograms.png`: width, height, maximum-side, and area histograms.
- `mass_box_width_height_crop_fit.png`: width/height scatter with crop bounds.
- `mass_box_crop_fit_counts.png`: fit versus cannot-fit counts.
"""


def _write_annotation_geometry_html(
    output_dir: Path,
    created_files: list[Path],
    summary: dict[str, Any],
) -> Path:
    images = [path for path in created_files if path.suffix.lower() == ".png"]
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Annotation Geometry Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1200px;} img{max-width:100%;border:1px solid #ddd;margin-bottom:28px;} pre{background:#f4f4f4;padding:12px;overflow:auto;}</style>",
        "</head><body><h1>Annotation geometry and crop-fit report</h1>",
        "<p>Fit is based only on annotation width and height; annotation and crop locations are ignored.</p>",
        f"<pre>{_html_escape(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))}</pre>",
    ]
    for path in images:
        html.append(f"<h2>{_html_escape(path.stem.replace('_', ' '))}</h2>")
        html.append(f"<img src='{_html_escape(path.name)}' alt='{_html_escape(path.name)}'>")
    html.append("</body></html>")
    path = output_dir / "index.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def visualize_export_from_config(config: dict[str, Any]) -> VisualizationResult:
    """Create plots from already-exported CSV/JSON files.

    This function does not read DICOM files and does not regenerate the dataset.
    It is intended for quick post-export analysis of the files under
    ``output_root``. The main inputs are:

    - ``<output_root>/<dataset>/stats/summary.csv``
    - ``<output_root>/<dataset>/stats/samples.csv``
    - ``<output_root>/<dataset>/metadata/samples_metadata_flat.csv`` if present
    - ``<output_root>/manifest.json`` if present
    """
    paths_cfg = config.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "G:/preprocessed-vindr"))
    viz_cfg = config.get("visualizations", {})
    output_dir = Path(viz_cfg.get("output_dir", output_root / "visualizations"))
    if not output_dir.is_absolute():
        output_dir = output_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result = create_visualizations_from_export(
        output_root=output_root,
        output_dir=output_dir,
        include_square_crops=bool(viz_cfg.get("include_square_crops", True)),
        include_baseline=bool(viz_cfg.get("include_baseline_uncropped", True)),
        write_html_report=bool(viz_cfg.get("write_html_report", True)),
        max_rows_per_samples_csv=viz_cfg.get("max_rows_per_samples_csv"),
    )
    return result


def create_visualizations_from_export(
    *,
    output_root: str | Path,
    output_dir: str | Path | None = None,
    include_square_crops: bool = True,
    include_baseline: bool = True,
    write_html_report: bool = True,
    max_rows_per_samples_csv: int | None = None,
) -> VisualizationResult:
    """Create dataset visualizations using only existing export artifacts.

    Parameters
    ----------
    output_root:
        Root folder of the finished export, e.g. ``G:/preprocessed-vindr``.
    output_dir:
        Folder where PNG plots and CSV summaries are saved. If ``None``, plots
        go to ``<output_root>/visualizations``.
    include_square_crops:
        Include ``square_crops/stats`` when present.
    include_baseline:
        Include ``baseline_uncropped/stats`` when present.
    write_html_report:
        If true, also create ``index.html`` linking all plots.
    max_rows_per_samples_csv:
        Optional row limit for very large samples.csv files. ``None`` reads all
        rows. You normally want all rows because these CSVs are much smaller than
        the original DICOM dataset.

    Returns
    -------
    VisualizationResult
        Created files and a compact summary. No DICOMs are read.
    """
    output_root = Path(output_root)
    if output_dir is None:
        output_dir = output_root / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names: list[str] = []
    if include_square_crops:
        dataset_names.append("square_crops")
    if include_baseline:
        dataset_names.append("baseline_uncropped")

    summaries: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []
    metadata_flat: list[pd.DataFrame] = []
    coco_boxes: list[pd.DataFrame] = []
    warnings: list[str] = []

    for dataset_name in dataset_names:
        summary_path = output_root / dataset_name / "stats" / "summary.csv"
        samples_path = output_root / dataset_name / "stats" / "samples.csv"
        metadata_path = output_root / dataset_name / "metadata" / "samples_metadata_flat.csv"

        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["dataset"] = df.get("dataset", dataset_name)
            summaries.append(df)
        else:
            warnings.append(f"Missing summary file: {summary_path}")

        if samples_path.exists():
            kwargs = {}
            if max_rows_per_samples_csv is not None:
                kwargs["nrows"] = int(max_rows_per_samples_csv)
            df = pd.read_csv(samples_path, **kwargs)
            df["dataset"] = df.get("dataset", dataset_name)
            samples.append(df)
        else:
            warnings.append(f"Missing samples file: {samples_path}")

        if metadata_path.exists():
            kwargs = {}
            if max_rows_per_samples_csv is not None:
                kwargs["nrows"] = int(max_rows_per_samples_csv)
            df = pd.read_csv(metadata_path, **kwargs)
            df["dataset"] = df.get("dataset", dataset_name)
            metadata_flat.append(df)

        coco_dir = output_root / dataset_name / "mmdetection" / "annotations"
        if coco_dir.exists():
            for split in SPLIT_ORDER:
                coco_path = coco_dir / f"instances_{split}.json"
                if coco_path.exists():
                    coco_boxes.append(_read_coco_box_dataframe(coco_path, dataset_name=dataset_name, split=split))

    created: list[Path] = []
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    samples_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    meta_df = pd.concat(metadata_flat, ignore_index=True) if metadata_flat else pd.DataFrame()
    coco_box_df = pd.concat(coco_boxes, ignore_index=True) if coco_boxes else pd.DataFrame()

    # Save combined copies so the user can inspect everything in one place.
    if not summary_df.empty:
        p = output_dir / "combined_summary.csv"
        summary_df.to_csv(p, index=False)
        created.append(p)
    if not samples_df.empty:
        p = output_dir / "combined_samples.csv"
        samples_df.to_csv(p, index=False)
        created.append(p)
    if not coco_box_df.empty:
        p = output_dir / "coco_box_annotations.csv"
        coco_box_df.to_csv(p, index=False)
        created.append(p)
        stats = _coco_box_size_summary(coco_box_df)
        p = output_dir / "coco_box_size_stats.csv"
        stats.to_csv(p, index=False)
        created.append(p)

    if not summary_df.empty:
        created.extend(_plot_summary_figures(summary_df, output_dir))
    if not samples_df.empty:
        created.extend(_plot_sample_figures(samples_df, output_dir))
    if not coco_box_df.empty:
        created.extend(_plot_coco_box_size_figures(coco_box_df, output_dir))
    if not meta_df.empty:
        created.extend(_plot_metadata_figures(meta_df, output_dir))

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        created.extend(_plot_manifest_figures(manifest_path, output_dir))

    sanity = _sanity_report(output_root, dataset_names, summary_df)
    if not coco_box_df.empty:
        sanity["coco_box_size_stats"] = _coco_box_size_summary(coco_box_df).to_dict(orient="records")
    sanity_path = output_dir / "sanity_report.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe({"warnings": warnings, **sanity}), f, indent=2, ensure_ascii=False)
    created.append(sanity_path)

    if write_html_report:
        html_path = _write_html_report(output_dir, created, warnings, sanity)
        created.append(html_path)

    return VisualizationResult(output_dir=output_dir, created_files=created, summary={"warnings": warnings, **sanity})


def _plot_summary_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    created: list[Path] = []
    df = _ordered(df)

    created.append(_bar_plot(
        df,
        x="split_label",
        y="num_images",
        title="Number of exported images by dataset and split",
        ylabel="images",
        output_path=output_dir / "01_num_images_by_split.png",
    ))

    if "positive_image_percent" in df.columns:
        created.append(_bar_plot(
            df,
            x="split_label",
            y="positive_image_percent",
            title="Mass-positive image percentage by dataset and split",
            ylabel="positive images (%)",
            output_path=output_dir / "02_positive_image_percent_by_split.png",
        ))

    if "num_mass_boxes" in df.columns:
        created.append(_bar_plot(
            df,
            x="split_label",
            y="num_mass_boxes",
            title="Number of mass boxes by dataset and split",
            ylabel="mass boxes",
            output_path=output_dir / "03_num_mass_boxes_by_split.png",
        ))

    if "mean_boxes_per_image" in df.columns:
        created.append(_bar_plot(
            df,
            x="split_label",
            y="mean_boxes_per_image",
            title="Mean mass boxes per exported image",
            ylabel="boxes per image",
            output_path=output_dir / "04_mean_boxes_per_image.png",
        ))

    if "mean_mass_area_percentage" in df.columns:
        created.append(_bar_plot(
            df,
            x="split_label",
            y="mean_mass_area_percentage",
            title="Mean mass area percentage per exported sample",
            ylabel="mean area (%)",
            output_path=output_dir / "05_mean_mass_area_percentage.png",
        ))

    return created


def _plot_sample_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    created: list[Path] = []
    df = _coerce_numeric(df)
    df = _ordered(df)

    if {"width", "height"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 6))
        for key, group in df.groupby("dataset", dropna=False):
            ax.scatter(group["width"], group["height"], alpha=0.35, s=8, label=str(key))
        ax.set_title("Exported image sizes")
        ax.set_xlabel("width (pixels)")
        ax.set_ylabel("height (pixels)")
        ax.legend(loc="best")
        p = output_dir / "06_image_size_scatter.png"
        _savefig(fig, p)
        created.append(p)

    if "num_mass_boxes" in df.columns:
        # Cap the range so a few outliers do not make the plot unreadable.
        capped = df.copy()
        capped["num_mass_boxes_capped"] = capped["num_mass_boxes"].clip(upper=5)
        p = _grouped_count_bar(
            capped,
            group_col="num_mass_boxes_capped",
            title="Mass boxes per exported image, values above 5 are clipped to 5",
            xlabel="number of mass boxes in image",
            output_path=output_dir / "07_boxes_per_image_distribution.png",
        )
        created.append(p)

    if "mean_mass_area_percentage" in df.columns:
        positive = df[df.get("has_mass", 0).astype(float) > 0].copy() if "has_mass" in df.columns else df.copy()
        if not positive.empty:
            p = _hist_by_dataset(
                positive,
                value_col="mean_mass_area_percentage",
                title="Mass area percentage distribution, positive samples only",
                xlabel="mean mass area in sample (%)",
                output_path=output_dir / "08_mass_area_percentage_hist.png",
                bins=40,
            )
            created.append(p)

    if "max_mass_area_percentage" in df.columns:
        positive = df[df.get("has_mass", 0).astype(float) > 0].copy() if "has_mass" in df.columns else df.copy()
        if not positive.empty:
            p = _hist_by_dataset(
                positive,
                value_col="max_mass_area_percentage",
                title="Largest mass area percentage distribution, positive samples only",
                xlabel="largest mass area in sample (%)",
                output_path=output_dir / "09_max_mass_area_percentage_hist.png",
                bins=40,
            )
            created.append(p)

    if "crop_mode" in df.columns and df["crop_mode"].notna().any():
        p = _grouped_count_bar(
            df[df["crop_mode"].notna()].copy(),
            group_col="crop_mode",
            title="Square-crop sampling mode distribution",
            xlabel="crop mode",
            output_path=output_dir / "10_crop_mode_distribution.png",
        )
        created.append(p)

    for col, title, filename in [
        ("view_position", "View position distribution", "11_view_position_distribution.png"),
        ("laterality", "Laterality distribution", "12_laterality_distribution.png"),
        ("rgb_scheme", "RGB scheme distribution", "13_rgb_scheme_distribution.png"),
        ("histogram_equalization_enabled", "Histogram equalization setting distribution", "14_histogram_equalization_distribution.png"),
    ]:
        if col in df.columns and df[col].notna().any():
            p = _grouped_count_bar(df.copy(), group_col=col, title=title, xlabel=col, output_path=output_dir / filename)
            created.append(p)

    return created



def _read_coco_box_dataframe(coco_path: Path, *, dataset_name: str, split: str) -> pd.DataFrame:
    """Read per-box statistics from an exported COCO/MMDetection JSON file."""
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img.get("id"): img for img in coco.get("images", [])}
    rows: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox", [np.nan, np.nan, np.nan, np.nan])
        if len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        try:
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
        except Exception:
            continue
        area = ann.get("area", w * h)
        try:
            area = float(area)
        except Exception:
            area = float(w * h)
        if not np.isfinite(area) or area < 0:
            area = float(w * h)
        sqrt_area = float(np.sqrt(area)) if area >= 0 else np.nan
        if sqrt_area < COCO_SMALL_SIDE:
            coco_size = "small"
        elif sqrt_area < COCO_LARGE_SIDE:
            coco_size = "medium"
        else:
            coco_size = "large"
        img = images.get(ann.get("image_id"), {})
        rows.append({
            "dataset": dataset_name,
            "split": split,
            "coco_json": str(coco_path),
            "image_id": ann.get("image_id"),
            "file_name": img.get("file_name"),
            "annotation_id": ann.get("id"),
            "category_id": ann.get("category_id"),
            "bbox_x": x if np.isfinite(x) else np.nan,
            "bbox_y": y if np.isfinite(y) else np.nan,
            "bbox_width_px": w,
            "bbox_height_px": h,
            "bbox_area_px2": area,
            "bbox_sqrt_area_px": sqrt_area,
            "bbox_aspect_ratio_w_over_h": (w / h) if np.isfinite(h) and h > 0 else np.nan,
            "coco_size": coco_size,
            "is_coco_small": bool(coco_size == "small"),
            "is_coco_medium": bool(coco_size == "medium"),
            "is_coco_large": bool(coco_size == "large"),
        })
    return pd.DataFrame(rows)


def _coco_box_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize COCO small/medium/large box counts by dataset and split."""
    if df.empty:
        return pd.DataFrame()
    data = df.copy()
    data["coco_size"] = pd.Categorical(data["coco_size"], categories=["small", "medium", "large"], ordered=True)
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(["dataset", "split"], observed=True, dropna=False):
        dataset, split = keys
        total = int(len(group))
        counts = group["coco_size"].value_counts().to_dict()
        row = {
            "dataset": str(dataset),
            "split": str(split),
            "num_boxes": total,
            "small_boxes": int(counts.get("small", 0)),
            "medium_boxes": int(counts.get("medium", 0)),
            "large_boxes": int(counts.get("large", 0)),
            "small_percent": 100.0 * int(counts.get("small", 0)) / total if total else 0.0,
            "medium_percent": 100.0 * int(counts.get("medium", 0)) / total if total else 0.0,
            "large_percent": 100.0 * int(counts.get("large", 0)) / total if total else 0.0,
            "median_width_px": float(group["bbox_width_px"].median()) if total else np.nan,
            "median_height_px": float(group["bbox_height_px"].median()) if total else np.nan,
            "median_sqrt_area_px": float(group["bbox_sqrt_area_px"].median()) if total else np.nan,
            "p10_sqrt_area_px": float(group["bbox_sqrt_area_px"].quantile(0.10)) if total else np.nan,
            "p90_sqrt_area_px": float(group["bbox_sqrt_area_px"].quantile(0.90)) if total else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _ordered(out)
    return out


def _plot_coco_box_size_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    created: list[Path] = []
    data = df.copy()
    if data.empty:
        return created
    data = _ordered(data)
    data["coco_size"] = pd.Categorical(data["coco_size"], categories=["small", "medium", "large"], ordered=True)

    stats = _coco_box_size_summary(data)
    if not stats.empty:
        label_col = "split_label" if "split_label" in stats.columns else "split"
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(stats))
        bottom = np.zeros(len(stats))
        for size, col in [("small", "small_boxes"), ("medium", "medium_boxes"), ("large", "large_boxes")]:
            vals = pd.to_numeric(stats[col], errors="coerce").fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=size)
            bottom += vals
        ax.set_title("COCO box size categories by dataset and split")
        ax.set_xlabel("")
        ax.set_ylabel("number of boxes")
        ax.set_xticks(x)
        ax.set_xticklabels(stats[label_col].astype(str), rotation=30, ha="right")
        ax.legend(loc="best")
        p = output_dir / "20_coco_box_size_counts.png"
        _savefig(fig, p)
        created.append(p)

        fig, ax = plt.subplots(figsize=(11, 5))
        bottom = np.zeros(len(stats))
        for size, col in [("small", "small_percent"), ("medium", "medium_percent"), ("large", "large_percent")]:
            vals = pd.to_numeric(stats[col], errors="coerce").fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=size)
            bottom += vals
        ax.set_title("COCO box size percentages by dataset and split")
        ax.set_xlabel("")
        ax.set_ylabel("boxes (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(stats[label_col].astype(str), rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.legend(loc="best")
        p = output_dir / "21_coco_box_size_percentages.png"
        _savefig(fig, p)
        created.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in data.groupby(["dataset", "split"], observed=True, dropna=False):
        values = pd.to_numeric(group["bbox_sqrt_area_px"], errors="coerce").dropna()
        if len(values):
            ax.hist(values, bins=45, alpha=0.35, label="/".join(map(str, label)))
    ax.axvline(COCO_SMALL_SIDE, linestyle="--", linewidth=1)
    ax.axvline(COCO_LARGE_SIDE, linestyle="--", linewidth=1)
    ax.text(COCO_SMALL_SIDE, ax.get_ylim()[1] * 0.95, "32 px", rotation=90, va="top", ha="right")
    ax.text(COCO_LARGE_SIDE, ax.get_ylim()[1] * 0.95, "96 px", rotation=90, va="top", ha="right")
    ax.set_title("COCO size distribution by sqrt(box area)")
    ax.set_xlabel("sqrt(box area) in pixels")
    ax.set_ylabel("boxes")
    ax.legend(loc="best", fontsize=8)
    p = output_dir / "22_coco_sqrt_box_area_hist.png"
    _savefig(fig, p)
    created.append(p)

    fig, ax = plt.subplots(figsize=(8, 7))
    for size, group in data.groupby("coco_size", observed=True, dropna=False):
        ax.scatter(group["bbox_width_px"], group["bbox_height_px"], s=10, alpha=0.35, label=str(size))
    ax.set_title("Box width vs height colored by COCO area bin")
    ax.set_xlabel("box width (pixels)")
    ax.set_ylabel("box height (pixels)")
    ax.legend(loc="best")
    p = output_dir / "23_coco_box_width_height_scatter.png"
    _savefig(fig, p)
    created.append(p)

    return created

def _plot_metadata_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    created: list[Path] = []
    # The flattened metadata column names can vary depending on source metadata.
    # Try several common VinDr/DICOM-like names.
    candidates = [
        ("metadata_csv_rows.Manufacturer", "Manufacturer distribution", "15_manufacturer_distribution.png"),
        ("metadata_csv_rows.ManufacturerModelName", "Manufacturer model distribution", "16_model_distribution.png"),
        ("breast_annotation_row.breast_birads", "Breast BI-RADS distribution", "17_breast_birads_distribution.png"),
        ("breast_annotation_row.breast_density", "Breast density distribution", "18_breast_density_distribution.png"),
    ]
    for col, title, filename in candidates:
        if col in df.columns and df[col].notna().any():
            p = _grouped_count_bar(df.copy(), group_col=col, title=title, xlabel=col, output_path=output_dir / filename, top_n=20)
            created.append(p)
    return created


def _plot_manifest_figures(manifest_path: Path, output_dir: Path) -> list[Path]:
    created: list[Path] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    timings = manifest.get("stage_timings", [])
    if timings:
        df = pd.DataFrame(timings)
        if "duration_seconds" in df.columns and "name" in df.columns:
            df = df.copy()
            df["duration_minutes"] = pd.to_numeric(df["duration_seconds"], errors="coerce") / 60.0
            fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
            ax.barh(df["name"].astype(str), df["duration_minutes"])
            ax.set_title("Export stage durations")
            ax.set_xlabel("duration (minutes)")
            ax.set_ylabel("stage")
            p = output_dir / "19_export_stage_durations.png"
            _savefig(fig, p)
            created.append(p)
    return created


def _sanity_report(output_root: Path, dataset_names: list[str], summary_df: pd.DataFrame) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for dataset_name in dataset_names:
        for split in SPLIT_ORDER:
            img_dir = output_root / dataset_name / "images" / split
            label_dir = output_root / dataset_name / "labels" / split
            preserved_dir = output_root / dataset_name / "preserved_16bit" / split
            img_count = _count_files(img_dir, "*.png")
            label_count = _count_files(label_dir, "*.txt")
            preserved_count = _count_files(preserved_dir, "*.png")
            counts[f"{dataset_name}.{split}.images"] = img_count
            counts[f"{dataset_name}.{split}.labels"] = label_count
            counts[f"{dataset_name}.{split}.preserved_16bit"] = preserved_count
            counts[f"{dataset_name}.{split}.image_label_match"] = bool(img_count == label_count)
    summary_rows = []
    if not summary_df.empty:
        for row in summary_df.to_dict(orient="records"):
            summary_rows.append({k: _json_scalar(v) for k, v in row.items()})
    return {"file_counts": counts, "summary_rows": summary_rows}


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "split" in df.columns:
        df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    if "dataset" in df.columns:
        df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    if {"dataset", "split"}.issubset(df.columns):
        df["split_label"] = df["dataset"].astype(str) + "\n" + df["split"].astype(str)
        return df.sort_values(["dataset", "split"]).reset_index(drop=True)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "width",
        "height",
        "num_mass_boxes",
        "has_mass",
        "mean_mass_area_percentage",
        "max_mass_area_percentage",
        "mean_boxes_per_image",
        "positive_image_percent",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bar_plot(df: pd.DataFrame, *, x: str, y: str, title: str, ylabel: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x].astype(str), pd.to_numeric(df[y], errors="coerce"))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    _annotate_bars(ax)
    _savefig(fig, output_path)
    return output_path


def _grouped_count_bar(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    top_n: int | None = None,
) -> Path:
    data = df.copy()
    data[group_col] = data[group_col].fillna("missing").astype(str)
    if top_n is not None:
        top_values = data[group_col].value_counts().head(top_n).index
        data.loc[~data[group_col].isin(top_values), group_col] = "other"
    if {"dataset", "split"}.issubset(data.columns):
        counts = data.groupby(["dataset", "split", group_col], observed=True).size().reset_index(name="count")
        counts["label"] = counts["dataset"].astype(str) + "/" + counts["split"].astype(str) + "/" + counts[group_col].astype(str)
    elif "split" in data.columns:
        counts = data.groupby(["split", group_col], observed=True).size().reset_index(name="count")
        counts["label"] = counts["split"].astype(str) + "/" + counts[group_col].astype(str)
    else:
        counts = data.groupby([group_col], observed=True).size().reset_index(name="count")
        counts["label"] = counts[group_col].astype(str)
    counts = counts.sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(counts)), 5))
    ax.bar(counts["label"], counts["count"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=75)
    _savefig(fig, output_path)
    return output_path


def _hist_by_dataset(df: pd.DataFrame, *, value_col: str, title: str, xlabel: str, output_path: Path, bins: int = 40) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for dataset_name, group in df.groupby("dataset", observed=True, dropna=False):
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if len(values):
            ax.hist(values, bins=bins, alpha=0.45, label=str(dataset_name))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.legend(loc="best")
    _savefig(fig, output_path)
    return output_path


def _annotate_bars(ax: Any) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.1f}" if abs(height) < 100 else f"{height:,.0f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _savefig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_html_report(output_dir: Path, created_files: list[Path], warnings: list[str], sanity: dict[str, Any]) -> Path:
    images = [p for p in created_files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    rel = lambda p: p.relative_to(output_dir).as_posix() if p.is_relative_to(output_dir) else p.as_posix()
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>VinDr-Mammo Export Visualizations</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;margin:10px 0 30px 0;} code{background:#f2f2f2;padding:2px 4px;} .warn{color:#9a4a00;}</style>",
        "</head><body>",
        "<h1>VinDr-Mammo Export Visualizations</h1>",
        "<p>These plots are generated from exported CSV/JSON files only. No DICOM files are read.</p>",
    ]
    if warnings:
        html.append("<h2>Warnings</h2><ul>")
        for w in warnings:
            html.append(f"<li class='warn'>{_html_escape(w)}</li>")
        html.append("</ul>")
    html.append("<h2>Sanity summary</h2><pre>")
    html.append(_html_escape(json.dumps(_json_safe(sanity), indent=2, ensure_ascii=False)))
    html.append("</pre>")
    html.append("<h2>Plots</h2>")
    for p in images:
        html.append(f"<h3>{_html_escape(p.stem.replace('_', ' '))}</h3>")
        html.append(f"<img src='{_html_escape(rel(p))}' alt='{_html_escape(p.name)}'>")
    html.extend(["</body></html>"])
    path = output_dir / "index.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def _html_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;").replace('"', "&quot;")


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return _json_scalar(obj)
