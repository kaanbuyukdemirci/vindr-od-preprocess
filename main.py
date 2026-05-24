from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Allows running this file directly with the VSCode Run button without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vindr_mammo import export_from_config, load_export_config  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "export_config.yaml"


def main() -> None:
    """Run the export described in config/export_config.yaml.

    You should normally only edit the YAML file. Keep this Python file simple so
    it can be started from the VSCode Run button without command-line arguments.
    """
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return

    cfg = load_export_config(CONFIG_PATH)
    result = export_from_config(cfg)

    print("\nExport finished")
    print("Output root:", result.output_root)
    print("Created key metadata/config files:")
    for path in result.created_files:
        if path.suffix.lower() in {".yaml", ".json", ".csv", ".txt"}:
            print("  ", path)
    print("\nSummary:")
    pprint(result.summary)


if __name__ == "__main__":
    main()
