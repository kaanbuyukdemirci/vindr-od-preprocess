from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import vindr_mammo.export as export_module
import vindr_mammo.features as feature_module
from vindr_mammo.export import (
    _apply_custom_channel_operation_float_preserving,
    _float_rgb_to_uint8,
    _make_rgb_image,
    _save_export_images,
    _save_paired_whole_image_for_crop,
)
from vindr_mammo.features import (
    DEFAULT_DINO_V3_COMPUTE_DTYPE,
    DEFAULT_DINO_V3_INPUT_SIZE,
    DEFAULT_DINO_V3_MODEL_ID,
    DEFAULT_RESEARCH_DATASET_FOLDER,
    DINO_V3_LVD_MEAN,
    DINO_V3_LVD_STD,
    _load_dinov3_model,
    _prepare_input,
    default_feature_dataset_root,
    default_selected_variants,
    estimate_dataset_channel_stats,
    extract_features_from_config,
    feature_output_folder,
    native_resized_variant_input_sizes,
    scan_dataset_image_variants,
)
from vindr_mammo.presets import (
    DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    DEFAULT_RESEARCH_HE_RGB_PRESET_KEY,
    STUDY_PRESETS,
    apply_study_preset,
)
from vindr_mammo.storage import estimate_export_space


def _pipeline_config(*, save_float32: bool = True) -> dict:
    return {
        "image_export": {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {
                    "source": "current_crop",
                    "steps": [
                        {
                            "op": "percentile_normalize",
                            "params": {"percentiles": [0.0, 100.0]},
                        }
                    ],
                }
                for channel in "RGB"
            },
        },
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"save": False},
        "float32_export": {"enabled": save_float32},
        "export": {"save_empty_label_files": True},
    }


def test_float32_export_is_lossless_relative_to_final_png_quantization(tmp_path) -> None:
    image = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 32, 32)
    info = _save_export_images(
        image,
        tmp_path,
        tmp_path.joinpath("images/train/example.png").relative_to(tmp_path),
        _pipeline_config(),
    )

    tensor_path = tmp_path / info["float32_image_path"]
    saved = torch.load(tensor_path, map_location="cpu")
    png = np.asarray(Image.open(tmp_path / info["image_path"]))

    assert saved.dtype == torch.float32
    assert saved.shape == (3, 32, 32)
    assert float(saved.min()) == 0.0
    assert float(saved.max()) == 1.0
    assert torch.unique(saved).numel() > 256
    assert np.unique(png[..., 0]).size <= 256


def test_float32_export_uses_native_float_clahe_path(tmp_path) -> None:
    config = _pipeline_config()
    for channel in "RGB":
        config["image_export"]["custom_channel_pipeline"][channel]["steps"] = [
            {"op": "clahe", "params": {"clip_limit": 2.0, "tile_grid_size": 8}}
        ]
    generator = torch.Generator().manual_seed(17)
    image = torch.rand((1, 256, 256), generator=generator, dtype=torch.float32)

    info = _save_export_images(
        image,
        tmp_path,
        tmp_path.joinpath("images/train/clahe.png").relative_to(tmp_path),
        config,
    )
    saved = torch.load(tmp_path / info["float32_image_path"], map_location="cpu")

    # It is neither an 8-bit image nor a uint16 image merely stored as float.
    assert torch.unique(saved[0]).numel() > 256
    uint16_round_trip = torch.round(saved[0] * 65535.0) / 65535.0
    assert torch.any(saved[0] != uint16_round_trip)


def test_float32_clahe_does_not_call_integer_opencv_interface(monkeypatch) -> None:
    if export_module.cv2 is not None:
        monkeypatch.setattr(
            export_module.cv2,
            "createCLAHE",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("integer CLAHE interface must not be called")
            ),
        )
    generator = np.random.default_rng(9)
    image = generator.random((64, 64), dtype=np.float32)
    result = _apply_custom_channel_operation_float_preserving(
        image,
        "clahe",
        {"clip_limit": 2.0, "tile_grid_size": 8},
        np.ones_like(image, dtype=bool),
    )

    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


@pytest.mark.parametrize("preset_key", list(STUDY_PRESETS))
def test_every_study_preset_quantizes_only_the_completed_float32_result(
    preset_key,
    monkeypatch,
) -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/custom"}},
        preset_key,
    )
    if export_module.cv2 is not None:
        monkeypatch.setattr(
            export_module.cv2,
            "createCLAHE",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("integer OpenCV CLAHE must not be called")
            ),
        )
        monkeypatch.setattr(
            export_module.cv2,
            "equalizeHist",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("integer OpenCV histogram equalization must not be called")
            ),
        )
    generator = np.random.default_rng(31)
    image = generator.random((32, 32), dtype=np.float32)

    png, png_meta = _make_rgb_image(image, config)
    processed, float_meta = _make_rgb_image(image, config, return_float=True)

    assert processed.dtype == np.float32
    assert png.dtype == np.uint8
    assert np.array_equal(png, _float_rgb_to_uint8(processed))
    assert png_meta["pipeline_processing_dtype"] == "float32"
    assert png_meta["pipeline_final_encoding"] == "uint8_png"
    assert float_meta["pipeline_processing_dtype"] == "float32"
    assert float_meta["pipeline_final_encoding"] == "float32"


@pytest.mark.parametrize(
    "image_export",
    [
        {"rgb_scheme": "grayscale_rgb", "single_window": [0.0, 100.0]},
        {"rgb_scheme": "equalized_rgb", "single_window": [0.0, 100.0]},
        {"rgb_scheme": "paper69_mammoclip_uint8"},
        {"rgb_scheme": "intensity_equalized_gradient"},
        {"rgb_scheme": "raw_clahe_detail"},
        {"rgb_scheme": "raw_replicated"},
        {"rgb_scheme": "raw_clahe_masked_raw"},
        {"rgb_scheme": "raw_clahe_tophat"},
        {
            "rgb_scheme": "custom_channel_pipeline",
            "custom_channel_pipeline": {
                channel: {
                    "source": "current_crop",
                    "steps": [
                        {"op": "percentile_normalize", "params": {"percentiles": [0.0, 100.0]}},
                        {"op": "hist_equalize", "params": {}},
                    ],
                }
                for channel in "RGB"
            },
        },
        {"rgb_scheme": "bitpack16"},
        {
            "rgb_scheme": "multi_window",
            "multi_window_percentiles": [[0.0, 100.0], [1.0, 99.0], [2.0, 98.0]],
        },
    ],
    ids=lambda config: config["rgb_scheme"],
)
def test_every_rgb_scheme_quantizes_only_after_its_float32_recipe(image_export) -> None:
    generator = np.random.default_rng(43)
    image = generator.random((32, 32), dtype=np.float32)
    config = {
        "image_export": image_export,
        "histogram_equalization": {"enabled": False},
        "preserved_16bit": {"percentile_range": [0.0, 100.0]},
    }

    png, png_meta = _make_rgb_image(image, config)
    processed, float_meta = _make_rgb_image(image, config, return_float=True)

    assert processed.dtype == np.float32
    assert png.dtype == np.uint8
    assert np.array_equal(png, _float_rgb_to_uint8(processed))
    assert png_meta["pipeline_processing_dtype"] == "float32"
    assert float_meta["pipeline_processing_dtype"] == "float32"


@pytest.mark.parametrize(
    ("operation", "params"),
    [
        ("percentile_normalize", {"percentiles": [0.0, 100.0]}),
        ("percentile_clip_only", {"percentiles": [1.0, 99.0]}),
        ("zscore_clip", {"z_limit": 3.0}),
        ("standardize_to_target", {"target_mean": 0.5, "target_std": 0.2}),
        ("aggressive_upper_percentile_normalize", {"percentiles": [70.0, 100.0]}),
        ("hist_equalize", {}),
        ("clahe", {"clip_limit": 2.0, "tile_grid_size": 4}),
        ("mask_outside_breast", {}),
        ("artifact_cleanup", {}),
        ("gaussian_blur", {"ksize": 3, "sigma": 1.0}),
        ("median_blur", {"ksize": 3}),
        ("bilateral_filter", {"diameter": 3, "sigma_color": 0.05, "sigma_space": 3.0}),
        ("wiener_filter", {"ksize": 3}),
        ("local_detail", {"sigma": 1.0}),
        ("sharpen", {"amount": 0.5}),
        ("unsharp_mask", {"amount": 0.5, "sigma": 1.0}),
        ("sobel_gradient", {"ksize": 3}),
        ("laplacian", {"ksize": 3}),
        ("white_tophat", {"kernel_size": 3}),
        ("blackhat", {"kernel_size": 3}),
        ("morphological_open", {"kernel_size": 3}),
        ("morphological_close", {"kernel_size": 3}),
        ("pectoral_suppression", {}),
        ("gamma", {"gamma": 1.2}),
        ("log", {"gain": 5.0}),
        ("invert", {}),
    ],
)
def test_every_custom_operation_accepts_and_returns_float32_without_quantizing(
    operation,
    params,
    monkeypatch,
) -> None:
    generator = np.random.default_rng(47)
    image = generator.random((32, 32), dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    monkeypatch.setattr(
        export_module,
        "_float_to_uint8_custom",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intermediate uint8 conversion is forbidden")
        ),
    )
    if export_module.cv2 is not None:
        monkeypatch.setattr(
            export_module.cv2,
            "createCLAHE",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("integer OpenCV CLAHE must not be called")
            ),
        )

    result = _apply_custom_channel_operation_float_preserving(
        image,
        operation,
        params,
        mask,
    )

    assert result.dtype == np.float32
    assert result.shape == image.shape
    assert np.isfinite(result).all()


def test_default_research_clahe_is_float32_before_png_encoding(
    tmp_path,
    monkeypatch,
) -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/custom"}},
        DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    )
    if export_module.cv2 is not None:
        monkeypatch.setattr(
            export_module.cv2,
            "createCLAHE",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("research v2 must not use integer OpenCV CLAHE")
            ),
        )
    generator = torch.Generator().manual_seed(23)
    image = torch.rand((1, 64, 64), generator=generator, dtype=torch.float32)

    info = _save_export_images(
        image,
        tmp_path,
        tmp_path.joinpath("images/train/research-v2.png").relative_to(tmp_path),
        config,
        float32_variant="resized_whole",
    )

    saved = torch.load(tmp_path / info["float32_image_path"], map_location="cpu")
    png = np.asarray(Image.open(tmp_path / info["image_path"]))
    quantized_float = (
        torch.round(saved.permute(1, 2, 0).clip(0.0, 1.0) * 255.0)
        .to(torch.uint8)
        .numpy()
    )

    assert saved.dtype == torch.float32
    assert torch.unique(saved[0]).numel() > 256
    assert np.array_equal(png, quantized_float)
    assert info["custom_channel_pipeline_processing_dtype"] == "float32"


def test_he_rgb_png_is_quantized_only_after_float32_histogram_equalization(
    tmp_path,
) -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/custom"}},
        DEFAULT_RESEARCH_HE_RGB_PRESET_KEY,
    )
    generator = torch.Generator().manual_seed(29)
    image = torch.rand((1, 64, 64), generator=generator, dtype=torch.float32)

    info = _save_export_images(
        image,
        tmp_path,
        tmp_path.joinpath("images/train/he-rgb.png").relative_to(tmp_path),
        config,
        float32_variant="resized_whole",
    )

    saved = torch.load(tmp_path / info["float32_image_path"], map_location="cpu")
    png = np.asarray(Image.open(tmp_path / info["image_path"]))
    quantized_float = (
        torch.round(saved.permute(1, 2, 0).clip(0.0, 1.0) * 255.0)
        .to(torch.uint8)
        .numpy()
    )

    assert saved.dtype == torch.float32
    assert torch.unique(saved[0]).numel() > 256
    assert np.array_equal(png, quantized_float)
    assert info["custom_channel_pipeline_processing_dtype"] == "float32"
    assert info["custom_channel_pipeline_final_encoding"] == "uint8_png"


def test_paired_whole_float32_paths_mirror_png_variants(tmp_path) -> None:
    image = torch.linspace(0.0, 1.0, 24, dtype=torch.float32).reshape(1, 4, 6)
    paired = {
        "enabled": True,
        "save_original": True,
        "save_resized": True,
        "save_high_resolution": False,
        "target_width": 16,
        "target_height": 16,
        "resized_canvas_mode": "per_image_square",
        "pad_value": 0.0,
        "pad_anchor": "left_top",
    }
    info = _save_paired_whole_image_for_crop(
        source_image=image,
        crop_root=tmp_path,
        split_name="train",
        filename="study__image__crop__x0_0_y0_0.png",
        source_image_id="image",
        config=_pipeline_config(),
        paired_cfg=paired,
        source_path_cache={},
    )

    original = torch.load(
        tmp_path / info["paired_whole_original_float32_image_path"],
        map_location="cpu",
    )
    resized = torch.load(tmp_path / info["paired_whole_float32_image_path"], map_location="cpu")
    assert original.shape == (3, 4, 6)
    assert resized.shape == (3, 16, 16)
    assert (tmp_path / info["paired_whole_original_image_path"]).exists()
    assert (tmp_path / info["paired_whole_image_path"]).exists()


def test_float32_pipeline_quantizes_paired_whole_pngs_only_after_resize(tmp_path) -> None:
    config = _pipeline_config()
    config["image_export"]["custom_channel_pipeline_dtype"] = "float32"
    for channel in "RGB":
        config["image_export"]["custom_channel_pipeline"][channel]["steps"] = [
            {
                "op": "percentile_normalize",
                "params": {"percentiles": [0.0, 100.0]},
            },
            {"op": "hist_equalize", "params": {}},
        ]
    generator = torch.Generator().manual_seed(41)
    image = torch.rand((1, 24, 40), generator=generator, dtype=torch.float32)
    info = _save_paired_whole_image_for_crop(
        source_image=image,
        crop_root=tmp_path,
        split_name="train",
        filename="study__image__crop__x0_0_y0_0.png",
        source_image_id="image",
        config=config,
        paired_cfg={
            "enabled": True,
            "save_original": True,
            "save_resized": True,
            "save_high_resolution": False,
            "target_width": 32,
            "target_height": 32,
            "resized_canvas_mode": "per_image_square",
            "pad_value": 0.0,
            "pad_anchor": "left_top",
        },
        source_path_cache={},
    )

    for png_key, float_key in [
        (
            "paired_whole_original_image_path",
            "paired_whole_original_float32_image_path",
        ),
        ("paired_whole_image_path", "paired_whole_float32_image_path"),
    ]:
        png = np.asarray(Image.open(tmp_path / info[png_key]))
        saved = torch.load(tmp_path / info[float_key], map_location="cpu")
        quantized_float = (
            torch.round(saved.permute(1, 2, 0).clip(0.0, 1.0) * 255.0)
            .to(torch.uint8)
            .numpy()
        )
        assert np.array_equal(png, quantized_float)


def test_float32_variant_selection_skips_original_and_keeps_resized(tmp_path) -> None:
    image = torch.linspace(0.0, 1.0, 24, dtype=torch.float32).reshape(1, 4, 6)
    config = _pipeline_config()
    config["float32_export"]["variants"] = {
        "crops": True,
        "resized_whole": True,
        "original_whole": False,
        "high_resolution_whole": False,
        "baseline_whole": False,
    }
    info = _save_paired_whole_image_for_crop(
        source_image=image,
        crop_root=tmp_path,
        split_name="train",
        filename="study__image__crop__x0_0_y0_0.png",
        source_image_id="image",
        config=config,
        paired_cfg={
            "enabled": True,
            "save_original": True,
            "save_resized": True,
            "save_high_resolution": False,
            "target_width": 16,
            "target_height": 16,
            "resized_canvas_mode": "per_image_square",
            "pad_value": 0.0,
            "pad_anchor": "left_top",
        },
        source_path_cache={},
    )

    assert info["paired_whole_original_float32_image_path"] == ""
    assert info["paired_whole_original_float32_write_status"] == "disabled"
    assert (tmp_path / info["paired_whole_float32_image_path"]).exists()


def test_research_dataset_scan_selects_every_type_except_original_whole(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "config_snapshot": {
                    "study_preset_provenance": {"preset_key": "simple_crop_pipeline_v1"}
                }
            }
        ),
        encoding="utf-8",
    )
    for directory, stem in [
        ("images", "crop"),
        ("whole_images", "whole"),
        ("whole_images_original", "original"),
    ]:
        path = tmp_path / "square_crops" / directory / "train" / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)

    scan = scan_dataset_image_variants(tmp_path)

    assert scan["is_default_research_dataset"] is True
    assert set(scan["variants"]) == {"crops", "resized_whole", "original_whole"}
    assert default_selected_variants(scan) == ["crops", "resized_whole"]


def test_grouped_dataset_scan_exposes_each_resized_resolution(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "config_snapshot": {
                "study_preset_provenance": {"preset_key": "simple_crop_pipeline_v1"}
            }
        }),
        encoding="utf-8",
    )
    for relative in [
        "images/original/train/original.png",
        "images/resized/1024x1024/train/large.png",
        "images/resized/640x640/train/small.png",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    float_path = (
        tmp_path
        / "images"
        / "float32"
        / "resized"
        / "640x640"
        / "train"
        / "small.pt"
    )
    float_path.parent.mkdir(parents=True)
    torch.save(torch.zeros((3, 8, 8), dtype=torch.float32), float_path)

    scan = scan_dataset_image_variants(tmp_path)

    assert scan["layout"] == "images_annotations_v1"
    assert set(scan["variants"]) == {
        "original_whole",
        "resized_whole_1024x1024",
        "resized_whole_640x640",
    }
    assert default_selected_variants(scan) == [
        "resized_whole_640x640",
        "resized_whole_1024x1024",
    ]
    assert scan["variants"]["resized_whole_640x640"]["float32_count"] == 1


def test_he_rgb_variant_is_detected_as_default_research_dataset(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "config_snapshot": {
                "study_preset_provenance": {
                    "preset_key": DEFAULT_RESEARCH_HE_RGB_PRESET_KEY
                }
            }
        }),
        encoding="utf-8",
    )

    scan = scan_dataset_image_variants(tmp_path)

    assert scan["is_default_research_dataset"] is True


def test_research_preset_enables_lossless_float32_image_export() -> None:
    config = apply_study_preset(
        {"paths": {"data_root": "/data", "output_root": "/exports/custom"}},
        DEFAULT_RESEARCH_DATASET_PRESET_KEY,
    )

    assert config["float32_export"] == {
        "enabled": True,
        "format": "pytorch_tensor",
        "dtype": "float32",
        "layout": "CHW",
        "value_range": [0.0, 1.0],
        "mirror_png_paths": True,
        "variants": {
            "crops": False,
            "resized_whole": True,
            "original_whole": False,
            "high_resolution_whole": False,
            "baseline_whole": False,
        },
    }


def test_space_estimate_respects_float32_image_type_selection() -> None:
    estimate = estimate_export_space(
        {
            "export": {"save_square_crops": True, "save_baseline_uncropped": False},
            "square_crops": {"crop_size": 16, "stride": 16},
            "paired_whole_images": {
                "enabled": True,
                "save_original": True,
                "save_resized": True,
                "save_high_resolution": False,
                "target_width": 16,
                "target_height": 16,
            },
            "preserved_16bit": {"save": False},
            "float32_export": {
                "enabled": True,
                "variants": {
                    "crops": True,
                    "resized_whole": True,
                    "original_whole": False,
                    "high_resolution_whole": False,
                    "baseline_whole": False,
                },
            },
            "storage_estimate": {
                "rgb_bytes_per_pixel": 1.0,
                "metadata_bytes_per_sample": 0,
                "metadata_bytes_per_source": 0,
                "fixed_metadata_bytes": 0,
                "safety_factor": 1.0,
            },
        },
        [{"width": 16, "height": 32, "export_split": "train"}],
    )

    # One estimated 16x16 crop plus one resized whole; the original is off.
    assert estimate.breakdown_bytes["float32_pytorch_images"] == 2 * 16 * 16 * 12


def test_space_estimate_includes_uncompressed_float32_companions() -> None:
    estimate = estimate_export_space(
        {
            "export": {"save_square_crops": True, "save_baseline_uncropped": False},
            "square_crops": {"crop_size": 16, "stride": 16},
            "paired_whole_images": {"enabled": False},
            "preserved_16bit": {"save": False},
            "float32_export": {"enabled": True},
            "storage_estimate": {
                "rgb_bytes_per_pixel": 1.0,
                "metadata_bytes_per_sample": 0,
                "metadata_bytes_per_source": 0,
                "fixed_metadata_bytes": 0,
                "safety_factor": 1.0,
            },
        },
        [{"width": 16, "height": 16, "export_split": "train"}],
    )

    assert estimate.breakdown_bytes["float32_pytorch_images"] == (
        estimate.model_pixel_count * 3 * 4
    )


class _FakeDino:
    def __init__(self) -> None:
        self.config = SimpleNamespace(patch_size=16, num_register_tokens=4)

    def eval(self):
        return self

    def __call__(
        self,
        *,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        del output_hidden_states, return_dict
        batch = int(pixel_values.shape[0])
        patches = int(pixel_values.shape[-2] // 16) * int(pixel_values.shape[-1] // 16)
        tokens = torch.arange(
            batch * (1 + 4 + patches) * 8,
            device=pixel_values.device,
            dtype=torch.float32,
        ).reshape(batch, 1 + 4 + patches, 8)
        return SimpleNamespace(last_hidden_state=tokens, hidden_states=(tokens,))


def test_feature_extractor_prefers_float_and_warns_on_png_fallback(tmp_path) -> None:
    image_root = tmp_path / "square_crops" / "images" / "train"
    float_root = tmp_path / "square_crops" / "float32" / "images" / "train"
    image_root.mkdir(parents=True)
    float_root.mkdir(parents=True)
    for stem in ["lossless", "fallback"]:
        Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_root / f"{stem}.png")
    torch.save(torch.linspace(0, 1, 3 * 16 * 16).reshape(3, 16, 16), float_root / "lossless.pt")

    config = {
        "paths": {"dataset_root": str(tmp_path)},
        "variants": ["crops"],
        "splits": ["all"],
        "model": {
            "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "device": "cpu",
            "compute_dtype": "float32",
        },
        "input": {
            "resize_mode": "exact",
            "width": 16,
            "height": 16,
            "mean": "0.485,0.456,0.406",
            "std": "0.229,0.224,0.225",
        },
        "extraction": {
            "layer": -1,
            "outputs": ["patch_tokens", "cls_token"],
            "batch_size": 2,
            "save_dtype": "float32",
            "prefer_float32_sources": True,
            "overwrite": False,
        },
    }
    summary = extract_features_from_config(
        config,
        model_loader=lambda _cfg, _device: _FakeDino(),
    )

    assert summary["saved_features"] == 2
    assert summary["png_fallback_count"] == 1
    output_root = tmp_path / "features"
    extraction_root = next(path for path in output_root.iterdir() if path.is_dir())
    lossless = torch.load(extraction_root / "crops" / "train" / "lossless.pt")
    fallback = torch.load(extraction_root / "crops" / "train" / "fallback.pt")
    assert lossless["source"]["loaded_format"] == "float32"
    assert fallback["source"]["loaded_format"] == "png"
    assert lossless["features"]["patch_tokens"].shape == (8, 1, 1)
    assert lossless["features"]["cls_token"].shape == (8,)
    assert (extraction_root / "README.md").exists()
    assert (extraction_root / "features_manifest.jsonl").exists()


def test_feature_extractor_processes_640_and_1024_variants_at_native_sizes_in_one_run(
    tmp_path,
) -> None:
    variants = {
        "resized_whole_640x640": 16,
        "resized_whole_1024x1024": 32,
    }
    for variant, size in variants.items():
        resolution = variant.removeprefix("resized_whole_")
        image_path = tmp_path / "images" / "resized" / resolution / "train" / "sample.png"
        float_path = (
            tmp_path
            / "images"
            / "float32"
            / "resized"
            / resolution
            / "train"
            / "sample.pt"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        float_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(image_path)
        torch.save(torch.rand((3, size, size), dtype=torch.float32), float_path)

    config = {
        "paths": {"dataset_root": str(tmp_path)},
        "variants": list(variants),
        "splits": ["all"],
        "model": {
            "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "device": "cpu",
            "compute_dtype": "float32",
        },
        "input": {
            "resize_mode": "exact",
            "width": 32,
            "height": 32,
            "variant_input_sizes": {
                "resized_whole_640x640": {"width": 16, "height": 16},
                "resized_whole_1024x1024": {"width": 32, "height": 32},
            },
            "mean": "0.485,0.456,0.406",
            "std": "0.229,0.224,0.225",
        },
        "extraction": {
            "layer": -1,
            "outputs": ["patch_tokens", "cls_token"],
            "batch_size": 2,
            "save_dtype": "float32",
            "prefer_float32_sources": True,
            "overwrite": False,
        },
    }

    summary = extract_features_from_config(
        config,
        model_loader=lambda _cfg, _device: _FakeDino(),
    )
    output_root = feature_output_folder(config)
    small = torch.load(output_root / "resized_whole_640x640" / "train" / "sample.pt")
    large = torch.load(output_root / "resized_whole_1024x1024" / "train" / "sample.pt")

    assert summary["requested_images"] == 2
    assert summary["saved_features"] == 2
    assert summary["png_fallback_count"] == 0
    assert "multires-16x16-32x32" in output_root.name
    assert small["input"]["shape_chw"] == [3, 16, 16]
    assert large["input"]["shape_chw"] == [3, 32, 32]
    assert small["features"]["patch_tokens"].shape == (8, 1, 1)
    assert large["features"]["patch_tokens"].shape == (8, 2, 2)


def test_native_resized_variant_sizes_are_inferred_for_both_research_branches() -> None:
    assert native_resized_variant_input_sizes(
        ["resized_whole_640x640", "resized_whole_1024x1024", "original_whole"]
    ) == {
        "resized_whole_640x640": {"width": 640, "height": 640},
        "resized_whole_1024x1024": {"width": 1024, "height": 1024},
    }


def test_high_accuracy_feature_defaults_target_research_dataset(tmp_path) -> None:
    root = default_feature_dataset_root(
        {"paths": {"output_root": str(tmp_path / "preprocessed-vindr-v19")}}
    )

    assert root == (tmp_path / DEFAULT_RESEARCH_DATASET_FOLDER).resolve()
    assert DEFAULT_DINO_V3_MODEL_ID == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert DEFAULT_DINO_V3_COMPUTE_DTYPE == "float32"
    assert DEFAULT_DINO_V3_INPUT_SIZE == 1024
    assert DINO_V3_LVD_MEAN == (0.485, 0.456, 0.406)
    assert DINO_V3_LVD_STD == (0.229, 0.224, 0.225)


def test_already_sized_float32_input_bypasses_interpolation(monkeypatch) -> None:
    tensor = torch.rand((3, 32, 32), dtype=torch.float32)
    monkeypatch.setattr(
        feature_module.F,
        "interpolate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-size input must not be interpolated")
        ),
    )

    prepared = _prepare_input(
        tensor,
        {"resize_mode": "exact", "width": 32, "height": 32},
    )

    assert torch.equal(prepared, tensor)


def test_channel_stat_estimator_repeats_grayscale_moments(tmp_path) -> None:
    root = tmp_path / "square_crops" / "float32" / "images" / "train"
    root.mkdir(parents=True)
    torch.save(torch.zeros((3, 2, 2), dtype=torch.float32), root / "black.pt")
    torch.save(torch.ones((3, 2, 2), dtype=torch.float32), root / "white.pt")

    stats = estimate_dataset_channel_stats(
        tmp_path,
        variants=["crops"],
        splits=["train"],
        max_images=2,
    )

    assert stats["sampled_images"] == 2
    assert stats["pixels_per_channel"] == 8
    assert stats["grayscale_replicated"] is True
    assert stats["mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert stats["std"] == pytest.approx([0.5, 0.5, 0.5])
    assert stats["recommended_mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert stats["recommended_std"] == pytest.approx([0.5, 0.5, 0.5])


def test_channel_stat_estimator_keeps_distinct_rgb_moments(tmp_path) -> None:
    root = tmp_path / "square_crops" / "float32" / "images" / "train"
    root.mkdir(parents=True)
    first = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.5], [0.0, 0.5]],
            [[0.25, 0.75], [0.25, 0.75]],
        ],
        dtype=torch.float32,
    )
    second = 1.0 - first
    torch.save(first, root / "first.pt")
    torch.save(second, root / "second.pt")

    stats = estimate_dataset_channel_stats(
        tmp_path,
        variants=["crops"],
        splits=["train"],
        max_images=2,
    )

    assert stats["grayscale_replicated"] is False
    assert stats["recommended_mean"] == pytest.approx(stats["mean"])
    assert stats["recommended_std"] == pytest.approx(stats["std"])


def test_gated_dinov3_error_explains_license_and_login(monkeypatch) -> None:
    transformers_module = ModuleType("transformers")

    class _GatedAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise OSError(
                "Cannot access gated repo. You must have access to it and be authenticated."
            )

    transformers_module.AutoModel = _GatedAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    with pytest.raises(RuntimeError) as error:
        _load_dinov3_model(
            {
                "model_id": DEFAULT_DINO_V3_MODEL_ID,
                "compute_dtype": "float32",
            },
            torch.device("cpu"),
        )

    message = str(error.value)
    assert f"https://huggingface.co/{DEFAULT_DINO_V3_MODEL_ID}" in message
    assert "hf auth login" in message
    assert "hf auth whoami" in message
    assert "restart the app" in message
