from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except Exception as exc:  # pragma: no cover
    pydicom = None
    apply_modality_lut = None
    apply_voi_lut = None
    _PYDICOM_IMPORT_ERROR = exc
else:
    _PYDICOM_IMPORT_ERROR = None

NormalizeMode = Literal["none", "minmax", "percentile", "zscore"]


@dataclass(frozen=True)
class DicomImage:
    """Container returned by read_dicom_image."""

    image: torch.Tensor
    dicom_meta: dict[str, Any]
    original_shape: tuple[int, int]
    output_shape: tuple[int, int]


def _as_float_array(
    ds: Any,
    use_voi_lut: bool,
    invert_monochrome1: bool,
    *,
    strict_voi_lut: bool = False,
    trace: dict[str, Any] | None = None,
) -> np.ndarray:
    """Read pixels from a pydicom dataset and convert them to float32."""
    arr = ds.pixel_array

    if apply_modality_lut is not None:
        try:
            arr = apply_modality_lut(arr, ds)
            if trace is not None:
                trace["ModalityLUTApplied"] = True
        except Exception:
            if trace is not None:
                trace["ModalityLUTApplied"] = False
            pass

    if use_voi_lut and apply_voi_lut is None and strict_voi_lut:
        raise RuntimeError("VOI LUT/windowing was required, but pydicom.apply_voi_lut is unavailable.")
    if use_voi_lut and apply_voi_lut is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Applying a VOI LUT on a float input array may give incorrect results",
                    category=UserWarning,
                )
                arr = apply_voi_lut(arr, ds)
            if trace is not None:
                trace["VOILUTApplied"] = True
        except Exception as exc:
            if trace is not None:
                trace["VOILUTApplied"] = False
                trace["VOILUTError"] = repr(exc)
            if strict_voi_lut:
                raise RuntimeError("Required DICOM VOI LUT/windowing failed.") from exc

    arr = np.asarray(arr, dtype=np.float32)

    # Optional MONOCHROME1 -> MONOCHROME2-style conversion.
    # MONOCHROME1 means the minimum stored value is intended to display as white.
    # After this inversion, low values are background/black and high values are tissue/white.
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if invert_monochrome1 and photometric == "MONOCHROME1":
        arr = arr.max() + arr.min() - arr

    if trace is not None:
        trace.setdefault("ModalityLUTApplied", False)
        trace.setdefault("VOILUTApplied", False)
        trace["VOILUTRequested"] = bool(use_voi_lut)
        trace["Monochrome1Inverted"] = bool(invert_monochrome1 and photometric == "MONOCHROME1")
        trace["IntensityTransformOrder"] = "modality_lut_then_voi_lut_then_monochrome1_inversion_then_normalization"

    return arr


def _normalize(arr: np.ndarray, mode: NormalizeMode, percentile_range: tuple[float, float]) -> np.ndarray:
    if mode == "none":
        return arr.astype(np.float32, copy=False)

    if mode == "minmax":
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    if mode == "percentile":
        lo, hi = np.nanpercentile(arr, percentile_range)
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    if mode == "zscore":
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        if std <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mean) / std).astype(np.float32)

    raise ValueError(f"Unknown normalize mode: {mode}")


def _resize_tensor_chw(image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}")
    image_4d = image.unsqueeze(0)
    image_4d = F.interpolate(image_4d, size=output_size, mode="bilinear", align_corners=False)
    return image_4d.squeeze(0)


def read_dicom_image(
    path: str | Path,
    *,
    normalize: NormalizeMode = "minmax",
    percentile_range: tuple[float, float] = (0.5, 99.5),
    use_voi_lut: bool = False,
    output_size: tuple[int, int] | None = None,
    add_channel_dim: bool = True,
    return_dicom_meta: bool = False,
    invert_monochrome1: bool = False,
    strict_voi_lut: bool = False,
) -> DicomImage:
    """Read a mammography DICOM file as a torch tensor.

    Parameters
    ----------
    path:
        Path to a DICOM file.
    normalize:
        ``"minmax"`` returns values in [0, 1]. ``"percentile"`` clips before scaling.
        ``"none"`` keeps the DICOM numeric scale after modality LUT.
    use_voi_lut:
        If True, applies DICOM windowing or VOI LUT when present. For raw training data,
        leaving this False is often safer.
    output_size:
        Optional ``(height, width)`` resize. If used through the Dataset, bounding boxes
        are scaled to this size.
    add_channel_dim:
        If True, returns ``[1, H, W]``. Otherwise returns ``[H, W]``.
    return_dicom_meta:
        If True, returns a small set of useful DICOM tags in ``dicom_meta``.
    invert_monochrome1:
        If True and the DICOM tag ``PhotometricInterpretation`` is ``MONOCHROME1``,
        invert pixel intensities so the returned image follows the MONOCHROME2-style
        convention: black background and bright tissue.
    strict_voi_lut:
        If True, raise instead of silently falling back when requested VOI
        LUT/windowing cannot be applied.
    """
    if pydicom is None:  # pragma: no cover
        raise ImportError(
            "pydicom could not be imported. Install requirements.txt first. "
            f"Original error: {_PYDICOM_IMPORT_ERROR}"
        )

    path = Path(path)
    ds = pydicom.dcmread(str(path))
    transform_trace: dict[str, Any] = {}
    arr = _as_float_array(
        ds,
        use_voi_lut=use_voi_lut,
        invert_monochrome1=invert_monochrome1,
        strict_voi_lut=strict_voi_lut,
        trace=transform_trace,
    )

    if arr.ndim != 2:
        # VinDr-Mammo should be grayscale. This keeps the code explicit if a handler
        # returns a different shape.
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D mammogram image, got shape {arr.shape} for {path}")

    original_shape = (int(arr.shape[0]), int(arr.shape[1]))
    arr = _normalize(arr, normalize, percentile_range)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if add_channel_dim:
        tensor = tensor.unsqueeze(0)

    if output_size is not None:
        if not add_channel_dim:
            tensor = tensor.unsqueeze(0)
            tensor = _resize_tensor_chw(tensor, output_size).squeeze(0)
        else:
            tensor = _resize_tensor_chw(tensor, output_size)

    output_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))

    dicom_meta: dict[str, Any] = {
        "InvertedMonochrome1": bool(invert_monochrome1 and str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"),
        **transform_trace,
    } if return_dicom_meta else {}
    if return_dicom_meta:
        for tag in [
            "PatientAge",
            "Manufacturer",
            "ManufacturerModelName",
            "PhotometricInterpretation",
            "Rows",
            "Columns",
            "PixelSpacing",
            "ImagerPixelSpacing",
            "ViewPosition",
            "ImageLaterality",
            "RescaleSlope",
            "RescaleIntercept",
        ]:
            if hasattr(ds, tag):
                value = getattr(ds, tag)
                try:
                    value = value.value
                except Exception:
                    pass
                dicom_meta[tag] = str(value) if not isinstance(value, (int, float, list, tuple)) else value

    return DicomImage(
        image=tensor,
        dicom_meta=dicom_meta,
        original_shape=original_shape,
        output_shape=output_shape,
    )
