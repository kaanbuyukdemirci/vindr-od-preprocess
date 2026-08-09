from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vindr_mammo.export import export_from_config, load_export_config
from vindr_mammo.dataset_layout import parse_window_grids, window_grid_configs
from vindr_mammo.lazy_crops import extract_complete_lazy_crop_family
from vindr_mammo.presets import (
    DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    apply_study_preset,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete default-research whole images, then metadata-only "
            "window grids at multiple window sizes and strides."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/export_config.yaml"))
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="WINDOW:STRIDE",
        help="Repeat for each metadata-only window grid; defaults to the preset list.",
    )
    # Backward-compatible one-window family flags.
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--strides", type=int, nargs="+")
    args = parser.parse_args()

    config = apply_study_preset(
        load_export_config(args.config), DEFAULT_RESEARCH_DATASET_PRESET_KEY
    )
    output_root = Path(config["paths"]["output_root"])
    if args.grid:
        grids = parse_window_grids(",".join(args.grid))
    elif args.window_size is not None or args.strides is not None:
        grids = window_grid_configs({
            "lazy_crop_grids": [
                {
                    "window_size": int(args.window_size or 1024),
                    "stride": int(stride),
                }
                for stride in (args.strides or [128, 256, 512])
            ]
        })
    else:
        grids = window_grid_configs(config)
    status_path = output_root.parent / "default-research-v2-extraction-status.json"
    status: dict[str, Any] = {
        "status": "running",
        "pid": int(os.getpid()),
        "started_at": _now(),
        "output_root": str(output_root),
        "grids": grids,
    }
    _write_status(status_path, status)
    try:
        result = export_from_config(config)
        status["whole_image_export_finished_at"] = _now()
        status["whole_image_manifest"] = str(
            result.output_root / "metadata" / "whole_image_manifest.csv"
        )
        _write_status(status_path, status)
        lazy = extract_complete_lazy_crop_family(
            result.output_root,
            grids=grids,
            min_box_visibility=0.05,
            overwrite=False,
        )
        status.update({
            "status": "completed",
            "finished_at": _now(),
            "lazy_crop_outputs": {
                str(grid): value["output_root"] for grid, value in lazy.items()
            },
        })
        _write_status(status_path, status)
    except Exception as exc:
        status.update({
            "status": "failed",
            "failed_at": _now(),
            "error": repr(exc),
        })
        _write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()
