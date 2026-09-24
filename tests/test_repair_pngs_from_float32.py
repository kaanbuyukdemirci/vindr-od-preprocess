from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_pngs_from_float32.py"
)
SPEC = importlib.util.spec_from_file_location("repair_pngs_from_float32", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


def test_repair_is_atomic_exact_and_resumable(tmp_path: Path) -> None:
    tensor_path = (
        tmp_path
        / "images"
        / "float32"
        / "resized"
        / "640x640"
        / "train"
        / "sample.pt"
    )
    png_path = (
        tmp_path
        / "images"
        / "resized"
        / "640x640"
        / "train"
        / "sample.png"
    )
    tensor_path.parent.mkdir(parents=True)
    png_path.parent.mkdir(parents=True)
    tensor = torch.linspace(0.0, 1.0, 3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    torch.save(tensor, tensor_path)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(png_path)

    first = repair._repair_one(
        tensor_path,
        png_path,
        write=True,
        compress_level=1,
    )
    second = repair._repair_one(
        tensor_path,
        png_path,
        write=True,
        compress_level=1,
    )

    expected = repair._final_png_pixels(tensor)
    with Image.open(png_path) as image:
        actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert first.status == "repaired"
    assert first.differing_values > 0
    assert second.status == "already_correct"
    assert np.array_equal(actual, expected)
    assert list(png_path.parent.glob(".*.repair-*.png")) == []


def test_discovery_maps_matching_resolution_and_split(tmp_path: Path) -> None:
    tensor_path = (
        tmp_path
        / "images"
        / "float32"
        / "resized"
        / "1024x1024"
        / "test"
        / "sample.pt"
    )
    png_path = (
        tmp_path
        / "images"
        / "resized"
        / "1024x1024"
        / "test"
        / "sample.png"
    )
    tensor_path.parent.mkdir(parents=True)
    png_path.parent.mkdir(parents=True)
    torch.save(torch.zeros((3, 4, 4), dtype=torch.float32), tensor_path)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(png_path)

    pairs, problems = repair._discover_pairs(
        tmp_path,
        ["1024x1024"],
        ["test"],
    )

    assert pairs == [(tensor_path, png_path)]
    assert problems == []
