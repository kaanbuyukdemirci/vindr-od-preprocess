from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from .export import export_from_config, load_export_config
from .visualize import visualize_export_from_config


def _default_config_path() -> Path:
    """Return the default config path used by the project scripts.

    The command line tools default to ``config/export_config.yaml`` relative to
    the current working directory. This keeps the installed package independent
    from machine-specific paths while still letting you run the project from
    VSCode without arguments.
    """
    return Path.cwd() / "config" / "export_config.yaml"


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to export_config.yaml. Default: ./config/export_config.yaml",
    )
    return parser


def run_export_from_config_path(config_path: str | Path):
    """Run the full export pipeline from a YAML config path."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_export_config(config_path)
    return export_from_config(cfg)


def run_visualization_from_config_path(config_path: str | Path):
    """Create fast visualizations from an already exported dataset."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_export_config(config_path)
    return visualize_export_from_config(cfg)


def export_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-export``."""
    parser = _build_parser("Export VinDr-Mammo mass-detection datasets.")
    args = parser.parse_args(argv)
    result = run_export_from_config_path(args.config)
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
    result = run_visualization_from_config_path(args.config)
    print("\nVisualization finished")
    print("Output directory:", result.output_dir)
    print("Created files:")
    for path in result.created_files:
        print("  ", path)
    print("\nSummary:")
    pprint(result.summary)


def gui_main(argv: list[str] | None = None) -> None:
    """Console entry point: ``vindr-mammo-gui``.

    This starts the Streamlit preprocessing inspector. Pass an optional config:

    ``vindr-mammo-gui --config config/export_config.yaml``
    """
    import sys
    from streamlit.web import cli as stcli

    parser = _build_parser("Open the interactive preprocessing inspector GUI.")
    args = parser.parse_args(argv)
    app_path = Path(__file__).resolve().parent / "gui_app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--", "--config", str(args.config)]
    raise SystemExit(stcli.main())
