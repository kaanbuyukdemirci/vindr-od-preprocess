from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from .export import export_from_config, load_export_config
from .presets import (
    DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    DUAL_WHOLE_PRESET_KEY,
    PAPER_22_IMPROVED_PRESET_KEY,
    PAPER_22_PRESET_KEY,
    PAPER_69_PRESET_KEY,
    SIMPLE_PRESET_KEY,
    STUDY_PRESETS,
    apply_study_preset,
)
from .visualize import visualize_export_from_config


def _default_config_path() -> Path:
    """Return the default config path used by the project scripts.

    The command line tools default to ``config/export_config.yaml`` relative to
    the current working directory. This keeps the installed package independent
    from machine-specific paths while still letting you run the project from
    VSCode without arguments.
    """
    return Path.cwd() / "config" / "export_config.yaml"


def _build_parser(description: str, *, include_preset: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to export_config.yaml. Default: ./config/export_config.yaml",
    )
    if include_preset:
        parser.add_argument(
            "--preset",
            choices=[
                "paper22",
                "custom-paper22",
                "paper22-improved",
                "paper69",
                "custom",
                "simple",
                "default-research",
                "simple-crop",
                "dual-whole",
                *STUDY_PRESETS.keys(),
            ],
            default=None,
            help=(
                "Apply a study preset after loading YAML. Clear aliases are paper22, "
                "custom-paper22, paper69, custom, and default-research; paper22-improved, simple, simple-crop, and dual-whole remain "
                "supported for backward compatibility."
            ),
        )
    return parser


def run_export_from_config_path(config_path: str | Path, *, preset_key: str | None = None):
    """Run the full export pipeline from a YAML config path."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_export_config(config_path)
    if preset_key:
        aliases = {
            "paper22": PAPER_22_PRESET_KEY,
            "custom-paper22": PAPER_22_IMPROVED_PRESET_KEY,
            "paper22-improved": PAPER_22_IMPROVED_PRESET_KEY,
            "paper69": PAPER_69_PRESET_KEY,
            "custom": SIMPLE_PRESET_KEY,
            "simple": SIMPLE_PRESET_KEY,
            "default-research": DEFAULT_RESEARCH_DATASET_PRESET_KEY,
            "simple-crop": DUAL_WHOLE_PRESET_KEY,
            "dual-whole": DUAL_WHOLE_PRESET_KEY,
        }
        resolved_key = aliases.get(str(preset_key), str(preset_key))
        cfg = apply_study_preset(cfg, resolved_key)
    return export_from_config(cfg)


def run_visualization_from_config_path(config_path: str | Path, *, preset_key: str | None = None):
    """Create fast visualizations from an already exported dataset."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_export_config(config_path)
    if preset_key:
        aliases = {
            "paper22": PAPER_22_PRESET_KEY,
            "custom-paper22": PAPER_22_IMPROVED_PRESET_KEY,
            "paper22-improved": PAPER_22_IMPROVED_PRESET_KEY,
            "paper69": PAPER_69_PRESET_KEY,
            "custom": SIMPLE_PRESET_KEY,
            "simple": SIMPLE_PRESET_KEY,
            "default-research": DEFAULT_RESEARCH_DATASET_PRESET_KEY,
            "simple-crop": DUAL_WHOLE_PRESET_KEY,
            "dual-whole": DUAL_WHOLE_PRESET_KEY,
        }
        resolved_key = aliases.get(str(preset_key), str(preset_key))
        cfg = apply_study_preset(cfg, resolved_key)
    return visualize_export_from_config(cfg)


def export_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-export``."""
    parser = _build_parser("Export VinDr-Mammo mass-detection datasets.")
    args = parser.parse_args(argv)
    result = run_export_from_config_path(args.config, preset_key=args.preset)
    print("\nExport finished")
    print("Output root:", result.output_root)
    print("Created key metadata/config files:")
    for path in result.created_files:
        if path.suffix.lower() in {".yaml", ".json", ".csv", ".txt"}:
            print("  ", path)
    print("\nSummary:")
    pprint(result.summary)


def visualize_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-visualize``."""
    parser = _build_parser("Create visualizations from an exported VinDr-Mammo dataset.")
    args = parser.parse_args(argv)
    result = run_visualization_from_config_path(args.config, preset_key=args.preset)
    print("\nVisualization finished")
    print("Output directory:", result.output_dir)
    print("Created files:")
    for path in result.created_files:
        print("  ", path)
    print("\nSummary:")
    pprint(result.summary)


def gui_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-gui``.

    This starts the Dash preprocessing studio. Pass an optional config:

    ``vindr-mammo-gui --config config/export_config.yaml``
    """
    from .dash_app import main as dash_main

    dash_main(argv)


def streamlit_gui_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-streamlit-gui``.

    This keeps the legacy Streamlit inspector available during the Dash migration.
    """
    import sys
    from streamlit.web import cli as stcli

    parser = _build_parser("Open the legacy Streamlit preprocessing inspector GUI.", include_preset=False)
    args = parser.parse_args(argv)
    app_path = Path(__file__).resolve().parent / "gui_app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--", "--config", str(args.config)]
    raise SystemExit(stcli.main())
