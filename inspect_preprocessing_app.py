from __future__ import annotations

import sys
from pathlib import Path

# Allows running this file directly before installing.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vindr_mammo.dash_app import main  # noqa: E402

if __name__ == "__main__":
    main(["--config", str(PROJECT_ROOT / "config" / "export_config.yaml")])
