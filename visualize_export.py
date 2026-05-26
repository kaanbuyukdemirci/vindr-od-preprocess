from __future__ import annotations

import sys
from pathlib import Path

# Allows running this file directly with the VSCode Run button before installing.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vindr_mammo.cli import visualize_main  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "export_config.yaml"


if __name__ == "__main__":
    # Equivalent to: vindr-mammo-visualize --config config/export_config.yaml
    visualize_main(["--config", str(CONFIG_PATH)])
