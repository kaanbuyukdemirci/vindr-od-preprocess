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


@dataclass
class VisualizationResult:
    output_dir: Path
    created_files: list[Path]
    summary: dict[str, Any]


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

    created: list[Path] = []
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    samples_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    meta_df = pd.concat(metadata_flat, ignore_index=True) if metadata_flat else pd.DataFrame()

    # Save combined copies so the user can inspect everything in one place.
    if not summary_df.empty:
        p = output_dir / "combined_summary.csv"
        summary_df.to_csv(p, index=False)
        created.append(p)
    if not samples_df.empty:
        p = output_dir / "combined_samples.csv"
        samples_df.to_csv(p, index=False)
        created.append(p)

    if not summary_df.empty:
        created.extend(_plot_summary_figures(summary_df, output_dir))
    if not samples_df.empty:
        created.extend(_plot_sample_figures(samples_df, output_dir))
    if not meta_df.empty:
        created.extend(_plot_metadata_figures(meta_df, output_dir))

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        created.extend(_plot_manifest_figures(manifest_path, output_dir))

    sanity = _sanity_report(output_root, dataset_names, summary_df)
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
