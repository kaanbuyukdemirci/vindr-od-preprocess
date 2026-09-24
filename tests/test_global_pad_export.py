from __future__ import annotations

from pathlib import Path

import numpy as np

from vindr_mammo.global_pad_export import (
    MassBox,
    SourceRecord,
    _float32_filesystem_estimate,
    _float32_output_bytes,
    _render_variant,
    _transform_box,
    _yolo_label_text,
    build_geometry_plan,
    pad_bottom_right,
    round_up_to_multiple,
)


def test_vindr_global_geometry_and_default_resolutions() -> None:
    plan = build_geometry_plan(
        maximum_source_height=3580,
        maximum_source_width=2812,
        target_heights=(512, 640, 800, 1024, 1280, 1600),
    )

    assert (plan.canvas_height, plan.canvas_width) == (3584, 2816)
    assert [
        (variant.output_height, variant.output_width, variant.resized_content_width)
        for variant in plan.variants
    ] == [
        (3584, 2816, 2816),
        (512, 416, 402),
        (640, 512, 503),
        (800, 640, 629),
        (1024, 832, 805),
        (1280, 1024, 1006),
        (1600, 1280, 1257),
    ]
    assert all(variant.output_height % 32 == 0 for variant in plan.variants)
    assert all(variant.output_width % 32 == 0 for variant in plan.variants)


def test_padding_is_top_left_anchored_and_only_added_right_and_bottom() -> None:
    image = np.arange(15, dtype=np.uint16).reshape(5, 3)

    padded = pad_bottom_right(image, output_height=8, output_width=6)

    assert padded.dtype == np.uint16
    assert np.array_equal(padded[:5, :3], image)
    assert np.count_nonzero(padded[:5, 3:]) == 0
    assert np.count_nonzero(padded[5:, :]) == 0


def test_compact_variant_resizes_canvas_then_adds_stride_padding() -> None:
    plan = build_geometry_plan(64, 32, target_heights=(32,))
    canvas = np.full((64, 32), 65535, dtype=np.uint16)

    compact = _render_variant(canvas, plan.variants[1])

    assert compact.shape == (32, 32)
    assert np.all(compact[:, :16] == 65535)
    assert np.all(compact[:, 16:] == 0)


def test_mass_boxes_scale_and_yolo_normalize_against_padded_output() -> None:
    plan = build_geometry_plan(64, 32, target_heights=(32,))
    variant = plan.variants[1]
    box = MassBox(annotation_id=7, xmin=8.0, ymin=16.0, xmax=24.0, ymax=48.0)
    record = SourceRecord(
        study_id="study",
        image_id="image",
        split="train",
        dicom_path=str(Path("source.dicom")),
        height=64,
        width=32,
        laterality="L",
        view_position="CC",
        mass_boxes=(box,),
    )

    transformed = _transform_box(box, variant, source_height=64, source_width=32)
    label = _yolo_label_text(record, variant).strip().split()

    assert transformed == (4.0, 8.0, 12.0, 24.0)
    assert label[0] == "0"
    assert [float(value) for value in label[1:]] == [0.25, 0.5, 0.25, 0.5]


def test_round_up_to_multiple() -> None:
    assert round_up_to_multiple(3580, 32) == 3584
    assert round_up_to_multiple(2812, 32) == 2816
    assert round_up_to_multiple(512, 32) == 512


def test_storage_estimate_has_float32_for_all_variants_and_filesystem_overhead() -> None:
    plan = build_geometry_plan(64, 32, target_heights=(32,))
    payload = 2 * (64 * 32 + 32 * 32) * 4

    assert _float32_output_bytes(2, plan.variants) == payload
    assert _float32_filesystem_estimate(
        2,
        1,
        plan.variants,
        allocation_unit=4096,
    ) > payload
