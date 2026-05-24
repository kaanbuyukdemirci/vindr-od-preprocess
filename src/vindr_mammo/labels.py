from __future__ import annotations

import ast
import re
from typing import Any

FINDING_CATEGORY_TO_ID: dict[str, int] = {
    "Mass": 1,
    "Calcification": 2,
    "Asymmetry": 3,
    "Focal Asymmetry": 4,
    "Global Asymmetry": 5,
    "Architectural Distortion": 6,
    "Suspicious Lymph Node": 7,
    "Skin Thickening": 8,
    "Skin Retraction": 9,
    "Nipple Retraction": 10,
}


def birads_to_int(value: Any) -> int | None:
    """Convert values like 'BI-RADS 4' or 4 to an integer."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and value != value:  # NaN
            return None
    except Exception:
        pass

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def density_to_int(value: Any) -> int | None:
    """Convert density categories A/B/C/D to 0/1/2/3 when possible."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    # Handles 'A', 'DENSITY A', 'BI-RADS A', etc.
    for letter, idx in {"A": 0, "B": 1, "C": 2, "D": 3}.items():
        if re.search(rf"\b{letter}\b", text) or text == letter:
            return idx
    return None


def parse_finding_categories(value: Any) -> list[str]:
    """Parse the finding_categories column into a list of strings."""
    if value is None:
        return []
    try:
        if isinstance(value, float) and value != value:  # NaN
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return [str(v) for v in value]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    # The official CSV stores examples like ["Mass", "Skin Retraction"].
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass

    # Fallback for comma-separated strings.
    return [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]


def finding_categories_to_ids(categories: list[str]) -> list[int]:
    return [FINDING_CATEGORY_TO_ID[c] for c in categories if c in FINDING_CATEGORY_TO_ID]
