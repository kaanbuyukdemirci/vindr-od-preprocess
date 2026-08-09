from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from vindr_mammo.export import (
    _save_review_source_assets,
    _write_dataset_review_bundle,
    _write_whole_variant_review_assets,
)
from vindr_mammo.gui_app import (
    _load_saved_dataset_viewer_index,
    _load_saved_viewer_image,
    _prepare_saved_source_display_image,
)


def test_review_bundle_saves_every_source_mask_frames_gifs_and_viewer_index(tmp_path: Path) -> None:
    crop_root = tmp_path / "square_crops"
    crop_path = crop_root / "images" / "train" / "sample.png"
    crop_path.parent.mkdir(parents=True)
    crop_pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    crop_pixels[6:20, 8:24] = 180
    Image.fromarray(crop_pixels).save(crop_path)
    whole_path = crop_root / "whole_images" / "train" / "sample.png"
    original_whole_path = crop_root / "whole_images_original" / "train" / "sample.png"
    native_whole_path = crop_root / "whole_images_native" / "train" / "sample.png"
    whole_path.parent.mkdir(parents=True)
    original_whole_path.parent.mkdir(parents=True)
    native_whole_path.parent.mkdir(parents=True)
    Image.fromarray(crop_pixels).save(whole_path)
    Image.fromarray(crop_pixels).save(original_whole_path)
    Image.fromarray(crop_pixels).save(native_whole_path)

    source_image = torch.zeros((1, 60, 80), dtype=torch.float32)
    source_image[:, 5:55, 8:72] = 0.75
    original_image = np.arange(60 * 80, dtype=np.uint16).reshape(60, 80)
    source_mask = np.zeros((60, 80), dtype=bool)
    source_mask[5:55, 8:72] = True
    source_box = torch.tensor([[20.0, 15.0, 36.0, 35.0]], dtype=torch.float32)
    target = {
        "mass": {"boxes": source_box},
        "_foreground_mask": source_mask,
        "preprocessing": {
            "mirrored": True,
            "original_shape": (60, 80),
            "processed_shape": (60, 80),
            "crop_box_xyxy": None,
        },
    }
    review_cfg = {
        "enabled": True,
        "save_original_previews": True,
        "save_source_previews": True,
        "save_masks": True,
        "source_preview_max_side": 256,
        "samples_per_split": 1,
        "seed": 7,
        "create_crop_gifs": True,
        "create_mask_gifs": True,
        "gif_panel_size": 256,
        "gif_frame_duration_ms": 100,
    }
    source_row = _save_review_source_assets(
        crop_root=crop_root,
        split_name="train",
        record={"image_id": "source-a", "study_id": "study-a"},
        image=source_image,
        target=target,
        review_cfg=review_cfg,
        original_image=original_image,
    )
    stats_rows = [{
        "split": "train",
        "source_index": 0,
        "source_image_id": "source-a",
        "source_study_id": "study-a",
        "file_name": "sample.png",
        "has_mass": True,
        "num_mass_boxes": 1,
        "is_positive_window": 1,
        "crop_mode": "deterministic",
        "crop_window_xyxy": (10, 10, 42, 42),
        "paired_whole_original_image_path": "whole_images_original/train/sample.png",
    }]
    coco = {
        "images": [{"id": 1, "file_name": "sample.png"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [8, 6, 16, 14]}],
    }
    summary, created = _write_dataset_review_bundle(
        crop_root=crop_root,
        stats_rows=stats_rows,
        coco_by_split={"train": coco, "val": {"images": [], "annotations": []}, "test": {"images": [], "annotations": []}},
        source_rows=[source_row],
        review_cfg=review_cfg,
    )

    assert summary["original_previews"] == 1
    assert summary["source_previews"] == 1
    assert summary["saved_masks"] == 1
    assert summary["sampled_crop_frames"] == 1
    assert (crop_root / "review" / "gifs" / "train_crop_review.gif").exists()
    assert (crop_root / "review" / "gifs" / "train_mask_review.gif").exists()
    crop_frame_path = next((crop_root / "review" / "crop_frames" / "train").glob("*.png"))
    mask_frame_path = next((crop_root / "review" / "mask_frames" / "train").glob("*.png"))
    with Image.open(crop_frame_path) as crop_frame:
        assert crop_frame.size == (256 * 3 + 8 * 2, 256 + 38 + 30)
    with Image.open(mask_frame_path) as mask_frame:
        assert mask_frame.size == (256 * 3 + 8 * 2, 256 + 38 + 30)
    assert all(path.exists() for path in created)
    index = json.loads((crop_root / "review" / "index.json").read_text(encoding="utf-8"))
    assert index["version"] == 2
    assert index["sources"][0]["coordinate_space"] == "fixed_preprocessed"
    assert index["sources"][0]["mirrored"] is True
    assert index["sources"][0]["original_preview_path"].startswith(
        "review/original_images/train/"
    )
    assert index["frame_folders"]["train"] == {
        "crop_review": "review/crop_frames/train",
        "mask_review": "review/mask_frames/train",
    }
    assert index["sampled_crops"][0]["crop_frame_path"].startswith(
        "review/crop_frames/train/"
    )
    assert index["sampled_masks"][0]["mask_frame_path"].startswith(
        "review/mask_frames/train/"
    )

    debug_dir = crop_root / "debug_logs"
    debug_dir.mkdir(parents=True)
    pd.DataFrame(stats_rows).to_csv(debug_dir / "crop_log.csv", index=False)
    loaded = _load_saved_dataset_viewer_index(str(crop_root), refresh_token=1)
    assert loaded["ok"] is True
    viewer_row = pd.Series(loaded["rows"][0])
    assert viewer_row["source_preview_exists"] is True
    assert viewer_row["mask_overlay_exists"] is True
    assert viewer_row["paired_whole_exists"] is True
    assert viewer_row["paired_whole_original_exists"] is True
    assert viewer_row["paired_whole_native_exists"] is True
    assert Path(str(viewer_row["paired_whole_path"])) == whole_path
    assert Path(str(viewer_row["paired_whole_original_path"])) == original_whole_path
    assert Path(str(viewer_row["paired_whole_native_path"])) == native_whole_path
    full_image = _load_saved_viewer_image(Path(str(viewer_row["source_preview_path"])))
    assert full_image is not None
    with_window = _prepare_saved_source_display_image(full_image, viewer_row)
    assert np.any(np.all(with_window == np.array([0, 220, 255], dtype=np.uint8), axis=-1))


def test_saved_viewer_derives_one_per_source_whole_paths_from_new_crop_name(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "square_crops"
    crop_name = "study-a__image-b__crop__train_0002_x512_y0_w1024_h1024.png"
    whole_name = "study-a__image-b.png"
    crop_path = crop_root / "images" / "train" / crop_name
    whole_path = crop_root / "whole_images" / "train" / whole_name
    native_path = crop_root / "whole_images_native" / "train" / whole_name
    for path in [crop_path, whole_path, native_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)

    debug_dir = crop_root / "debug_logs"
    debug_dir.mkdir(parents=True)
    pd.DataFrame([{
        "split": "train",
        "source_image_id": "image-b",
        "file_name": crop_name,
        "has_mass": 0,
        "is_positive_window": 0,
    }]).to_csv(debug_dir / "crop_log.csv", index=False)

    loaded = _load_saved_dataset_viewer_index(str(crop_root), refresh_token=2)
    assert loaded["ok"] is True
    row = loaded["rows"][0]
    assert Path(row["paired_whole_path"]) == whole_path
    assert Path(row["paired_whole_native_path"]) == native_path
    assert row["paired_whole_exists"] is True
    assert row["paired_whole_native_exists"] is True


@pytest.mark.parametrize(
    ("include_high_resolution", "expected_frame_width"),
    [(False, 256 * 2 + 8), (True, 256 * 3 + 16)],
)
def test_whole_variant_review_writes_overlays_audits_gif_and_plots(
    tmp_path: Path,
    include_high_resolution: bool,
    expected_frame_width: int,
) -> None:
    crop_root = tmp_path / "square_crops"
    variants = {
        "original": ("whole_images_original/train/source.png", 20, 10, [2, 1, 8, 7], 1.0),
        "resized": ("whole_images/train/source.png", 16, 16, [1.6, 0.8, 6.4, 5.6], 0.8),
        "high_resolution": ("whole_images_high_resolution/train/source.png", 32, 32, [2, 1, 8, 7], 1.0),
    }
    if not include_high_resolution:
        variants.pop("high_resolution")
    row: dict[str, object] = {
        "split": "train",
        "source_image_id": "source",
        "source_study_id": "study",
        "mass_boxes_xyxy": [[2, 1, 8, 7]],
    }
    prefixes = {
        "original": "paired_whole_original",
        "resized": "paired_whole",
        "high_resolution": "paired_whole_high_resolution",
    }
    for variant, (path_text, width, height, bbox, scale) in variants.items():
        path = crop_root / path_text
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (width, height), (90, 90, 90)).save(path)
        prefix = prefixes[variant]
        row[f"{prefix}_image_path"] = path_text
        row[f"{prefix}_width"] = width
        row[f"{prefix}_height"] = height
        row[f"{prefix}_pad_left"] = 0
        row[f"{prefix}_pad_top"] = 0
        row[f"{prefix}_pad_right"] = max(0, width - 20)
        row[f"{prefix}_pad_bottom"] = max(0, height - 10)
        row[f"{prefix}_scale_x"] = scale
        row[f"{prefix}_scale_y"] = scale
        row[f"{prefix}_annotations"] = [{
            "source_annotation_id": "mass-1",
            "source_bbox_xyxy": [2, 1, 8, 7],
            "bbox_xyxy": bbox,
            "transform": {
                "pad_left": 0,
                "pad_top": 0,
                "scale_x": scale,
                "scale_y": scale,
            },
        }]

    summary, created = _write_whole_variant_review_assets(
        crop_root=crop_root,
        source_rows=[row],
        review_cfg={
            "samples_per_split": 1,
            "source_preview_max_side": 256,
            "gif_panel_size": 256,
            "gif_frame_duration_ms": 100,
            "create_whole_variant_gifs": True,
            "seed": 4,
        },
    )

    assert summary["invalid_boxes"] == 0
    assert summary["max_transform_error_px"] < 1e-9
    assert summary["sampled_frames"] == 1
    assert (crop_root / "review" / "whole_variant_geometry.csv").is_file()
    assert (crop_root / "review" / "whole_variant_box_audit.csv").is_file()
    assert (crop_root / "review" / "gifs" / "train_whole_variants.gif").is_file()
    frame_path = next(
        (crop_root / "review" / "whole_variant_frames" / "train").glob("*.png")
    )
    with Image.open(frame_path) as frame:
        assert frame.size == (expected_frame_width, 256 + 38 + 30)
    for variant in variants:
        assert next(
            (crop_root / "review" / "whole_variant_overlays" / variant / "train").glob("*.png")
        ).is_file()
    for filename in [
        "01_output_dimensions.png",
        "02_padding_distribution.png",
        "03_annotation_count_parity.png",
        "04_scale_factor_distribution.png",
    ]:
        assert (
            crop_root / "review" / "whole_variant_visualizations" / filename
        ).is_file()
    assert all(path.exists() for path in created)
