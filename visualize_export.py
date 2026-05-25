from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Allows running this file directly with the VSCode Run button without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vindr_mammo import load_export_config, visualize_export_from_config  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "export_config.yaml"


def main() -> None:
    """Create visualization PNGs from an already exported dataset.

    This script does not read the original DICOMs and does not regenerate images.
    It reads the CSV/JSON files already saved under paths.output_root in the YAML.
    """
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return

    cfg = load_export_config(CONFIG_PATH)
    result = visualize_export_from_config(cfg)

    print("\nVisualization finished")
    print("Output directory:", result.output_dir)
    print("Created files:")
    for path in result.created_files:
        print("  ", path)
    print("\nSummary:")
    pprint(result.summary)


if __name__ == "__main__":
    main()
