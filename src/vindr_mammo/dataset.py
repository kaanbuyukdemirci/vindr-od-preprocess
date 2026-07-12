from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional but listed in requirements.
    tqdm = None

from .dicom_io import NormalizeMode, read_dicom_image
from .labels import birads_to_int, density_to_int, finding_categories_to_ids, parse_finding_categories
from .preprocessing import apply_geometry_preprocessing, breast_crop_box, breast_is_on_right, make_preprocess_options
from .crops import (
    crop_image_and_boxes_to_window,
    crop_size_fit_table,
    make_crop_options,
    sample_random_square_window,
    sliding_square_windows,
    window_has_positive_mass,
)

IndexLevel = Literal["image", "exam", "crop"]

# Mass bounding-box area bins, expressed as percentage of the full image area.
# Example: 0.05 means the box occupies 0.05% of the full mammogram image.
SIZE_TINY = 0.05
SIZE_VERY_SMALL = 0.10
SIZE_SMALL = 0.50
SIZE_MEDIUM = 1.00
SIZE_LARGE = 1.00


class VindrMammoDataset(Dataset):
    """PyTorch Dataset for VinDr-Mammo.

    The project directory expected by this class is the official VinDr-Mammo
    directory containing ``metadata.csv``, ``breast-level_annotations.csv``,
    ``finding_annotations.csv``, and ``images/<study_id>/<image_id>.dicom``.

    Three indexing styles are supported:

    * ``index_level="image"``: one item is one DICOM image. This is useful for
      image-level classification, detection, or quick debugging.
    * ``index_level="exam"``: one item is one study/exam. The four standard
      views are grouped into one returned sample, which is useful for multi-view
      mammography models.
    * ``index_level="crop"``: one item is one final square object-detection crop.
      Use this with ``crop_options={"enabled": True, ...}``.

    Images are returned at their original DICOM size by default. Resizing only
    happens when ``output_size`` is explicitly set. The provided
    ``vindr_mammo_collate`` keeps images in Python lists, so the DataLoader does
    not stack, pad, or resize variable-size mammograms.
    """

    # Expose the mass area thresholds on the class, so you can access them as
    # VindrMammoDataset.SIZE_TINY or dataset.SIZE_TINY.
    SIZE_TINY = SIZE_TINY
    SIZE_VERY_SMALL = SIZE_VERY_SMALL
    SIZE_SMALL = SIZE_SMALL
    SIZE_MEDIUM = SIZE_MEDIUM
    SIZE_LARGE = SIZE_LARGE

    def __init__(
        self,
        data_root: str | Path,
        *,
        index_level: IndexLevel = "image",
        split: str | Iterable[str] | None = None,
        read_image: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        joint_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        normalize: NormalizeMode = "minmax",
        percentile_range: tuple[float, float] = (0.5, 99.5),
        use_voi_lut: bool = False,
        strict_voi_lut: bool = False,
        output_size: tuple[int, int] | None = None,
        return_dicom_meta: bool = False,
        validate_paths: bool = False,
        preprocess_options: dict[str, Any] | None = None,
        crop_options: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> None:
        """Create a VinDr-Mammo dataset object.

        Parameters
        ----------
        data_root:
            Path to the downloaded VinDr-Mammo ``1.0.0`` folder. This folder
            must contain ``metadata.csv``, ``breast-level_annotations.csv``,
            ``finding_annotations.csv``, and the ``images`` folder.
        index_level:
            ``"image"`` returns one sample per DICOM image. ``"exam"``
            returns one sample per study and groups the study's views together.
            ``"crop"`` returns one final square object-detection crop per item.
        split:
            Uses the official ``split`` column from the annotation CSVs. Use
            ``None`` for all images, ``"training"`` for the training split,
            ``"test"`` for the test split, or an iterable such as
            ``["training", "test"]``.
        read_image:
            If ``True``, DICOM pixels are read in ``__getitem__``. If ``False``,
            samples still return annotations, paths, boxes, and metadata, but
            ``image`` is ``None``. This is useful for quick CSV inspection.
        transform:
            Optional image-only transform applied after reading and optional
            resizing. It receives a tensor shaped ``[1, H, W]``. If your
            transform changes geometry, you are responsible for updating boxes,
            or use ``joint_transform`` instead.
        target_transform:
            Optional target-only transform applied to the target dictionary.
        joint_transform:
            Optional transform applied to the full returned sample. Use this
            when image and boxes must be transformed together.
        normalize:
            Pixel normalization mode used by ``read_dicom_image``. Options are
            ``"none"``, ``"minmax"``, ``"percentile"``, and ``"zscore"``.
        percentile_range:
            Lower and upper percentiles used only when
            ``normalize="percentile"``. Example: ``(0.5, 99.5)``.
        use_voi_lut:
            If ``True``, applies DICOM VOI LUT or windowing when present. For
            raw model training, the default ``False`` is usually safer because
            it avoids display-style windowing.
        strict_voi_lut:
            If ``True``, fail when requested VOI LUT/windowing cannot be applied.
        output_size:
            Optional resize as ``(height, width)``. Keep ``None`` to preserve
            the original DICOM size. When set, returned ``boxes`` are scaled to
            the resized image, while ``boxes_original`` keeps CSV coordinates.
        return_dicom_meta:
            If ``True``, returns a small dictionary of useful DICOM tags in
            ``target["dicom_meta"]``. Pixel data is still read in the same way.
        validate_paths:
            If ``True``, checks every resolved DICOM path during initialization
            and raises an error for missing files. Keep ``False`` when you want
            faster startup or metadata-only inspection.
        preprocess_options:
            Optional dictionary controlling compact preprocessing steps. Supported
            keys are ``invert_to_black_background``, ``crop_breast``,
            ``mirror_right_to_left``, ``crop_padding``, ``crop_threshold``, and
            ``min_component_area_fraction``. Geometry-changing steps update
            returned boxes automatically.
        crop_options:
            Optional dictionary for the final square crop stage used for object
            detection training. Set ``enabled=True`` and choose ``mode="random"``
            or ``mode="deterministic"``. Important keys are ``crop_size``,
            ``stride``, ``positive_fraction``, ``center_shift_fraction``,
            ``allow_partial_annotations``, and ``min_box_visibility``. Square
            crops are applied after inversion, breast crop, and mirroring.
        show_progress:
            If ``True``, long-running loops use ``tqdm`` progress bars in the
            VSCode terminal. This is useful for preprocessed statistics,
            deterministic crop indexing, crop-stat sampling, visual-test search,
            and GIF saving. Set ``False`` for silent library use.
        """
        self.data_root = Path(data_root)
        self.images_root = self.data_root / "images"
        self.index_level = index_level
        self.split = split
        self.read_image = read_image
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.normalize = normalize
        self.percentile_range = percentile_range
        self.use_voi_lut = use_voi_lut
        self.strict_voi_lut = bool(strict_voi_lut)
        self.output_size = output_size
        self.return_dicom_meta = return_dicom_meta
        self.validate_paths = validate_paths
        self.show_progress = bool(show_progress)
        self.preprocess_options = make_preprocess_options(preprocess_options)
        self.crop_options = make_crop_options(crop_options)
        seed = self.crop_options.get("seed")
        self._crop_rng = np.random.default_rng(seed)
        self._processed_stats_cache: dict[int | None, tuple[pd.DataFrame, pd.DataFrame]] = {}

        self._validate_root()
        self.breast_df = self._read_csv("breast-level_annotations.csv")
        self.finding_df = self._read_csv("finding_annotations.csv")
        self.metadata_df = self._read_csv("metadata.csv")

        self._standardize_columns()
        self._filter_split()
        self._attach_paths()
        self._build_indexes()
        self._build_crop_index_if_needed()

    def _progress(
        self,
        iterable: Iterable[Any],
        *,
        desc: str,
        total: int | None = None,
        unit: str = "it",
        leave: bool = True,
    ) -> Iterable[Any]:
        """Return ``iterable`` wrapped in tqdm when progress display is enabled."""
        if not self.show_progress or tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)

    def _validate_root(self) -> None:
        required = [
            self.data_root / "breast-level_annotations.csv",
            self.data_root / "finding_annotations.csv",
            self.data_root / "metadata.csv",
            self.images_root,
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            missing_text = "\n".join(str(p) for p in missing)
            raise FileNotFoundError(
                "VinDr-Mammo root is missing required files or folders.\n"
                f"Root received: {self.data_root}\n"
                f"Missing:\n{missing_text}\n\n"
                "Expected structure:\n"
                "  data_root/metadata.csv\n"
                "  data_root/breast-level_annotations.csv\n"
                "  data_root/finding_annotations.csv\n"
                "  data_root/images/<study_id>/<image_id>.dicom"
            )

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_root / filename
        return pd.read_csv(path)

    def _standardize_columns(self) -> None:
        for df in [self.breast_df, self.finding_df, self.metadata_df]:
            df.columns = [str(c).strip() for c in df.columns]

        if "image_id" not in self.breast_df.columns:
            raise ValueError(
                "breast-level_annotations.csv is expected to contain image_id. "
                "The PhysioNet data description and paper list image_id as an image-level field."
            )

        # Make identifier columns string typed to avoid accidental numeric parsing.
        for df in [self.breast_df, self.finding_df, self.metadata_df]:
            for col in ["study_id", "series_id", "image_id"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        # Preserve a stable identity for every source annotation before any
        # filtering/reset_index operation. Patch exports can then prove
        # one-to-one source-GT coverage even when overlapping patches duplicate
        # the same lesion.
        if "source_annotation_id" not in self.finding_df.columns:
            self.finding_df["source_annotation_id"] = self.finding_df.index.astype(int)
        if "source_annotation_row" not in self.finding_df.columns:
            # CSV row number including the header line (useful for audits).
            self.finding_df["source_annotation_row"] = self.finding_df.index.astype(int) + 2

    def _filter_split(self) -> None:
        """Filter the annotation tables using the official split column."""
        if self.split is None:
            return
        if "split" not in self.breast_df.columns:
            raise ValueError("split filtering was requested, but breast-level_annotations.csv has no split column.")

        if isinstance(self.split, str):
            wanted = {self.split}
        else:
            wanted = set(self.split)

        self.breast_df = self.breast_df[self.breast_df["split"].isin(wanted)].reset_index(drop=True)
        if "split" in self.finding_df.columns:
            self.finding_df = self.finding_df[self.finding_df["split"].isin(wanted)].reset_index(drop=True)

    def _attach_paths(self) -> None:
        """Add a resolved DICOM path to each breast-level annotation row."""
        self.breast_df = self.breast_df.copy()
        self.breast_df["dicom_path"] = self.breast_df.apply(self._resolve_image_path_from_row, axis=1)
        if self.validate_paths:
            paths = self.breast_df["dicom_path"].tolist()
            missing = [
                p
                for p in self._progress(paths, desc="Validating DICOM paths", total=len(paths), unit="file")
                if not Path(p).exists()
            ]
            if missing:
                preview = "\n".join(str(p) for p in missing[:10])
                raise FileNotFoundError(f"Some DICOM files were not found. First missing files:\n{preview}")

    def _resolve_image_path_from_row(self, row: pd.Series) -> str:
        study_id = str(row["study_id"])
        image_id = str(row["image_id"])
        study_dir = self.images_root / study_id

        candidates = [
            study_dir / f"{image_id}.dicom",
            study_dir / f"{image_id}.dcm",
            study_dir / image_id,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Keep the official path even if the data is not present yet. This lets the
        # Dataset be constructed for metadata inspection when validate_paths=False.
        return str(candidates[0])

    def _build_indexes(self) -> None:
        """Create image-level, study-level, finding, and metadata lookup tables."""
        self.image_records = self.breast_df.to_dict(orient="records")

        # Group finding rows by image_id. If an image has no finding rows, it returns an empty list.
        self.findings_by_image_id: dict[str, list[dict[str, Any]]] = {}
        if "image_id" in self.finding_df.columns:
            for image_id, group in self.finding_df.groupby("image_id", dropna=False):
                self.findings_by_image_id[str(image_id)] = [self._clean_finding_record(r) for r in group.to_dict(orient="records")]

        # Metadata can be image-level or study-level depending on how the CSV is read.
        self.metadata_by_image_id: dict[str, list[dict[str, Any]]] = {}
        self.metadata_by_study_id: dict[str, list[dict[str, Any]]] = {}
        if "image_id" in self.metadata_df.columns:
            for image_id, group in self.metadata_df.groupby("image_id", dropna=False):
                self.metadata_by_image_id[str(image_id)] = group.to_dict(orient="records")
        if "study_id" in self.metadata_df.columns:
            for study_id, group in self.metadata_df.groupby("study_id", dropna=False):
                self.metadata_by_study_id[str(study_id)] = group.to_dict(orient="records")

        self.study_ids = self.breast_df["study_id"].drop_duplicates().tolist()
        self.records_by_study_id: dict[str, list[dict[str, Any]]] = {}
        for study_id, group in self.breast_df.groupby("study_id", sort=False):
            records = group.to_dict(orient="records")
            self.records_by_study_id[str(study_id)] = sorted(records, key=self._view_sort_key)

    @staticmethod
    def _view_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        laterality_order = {"L": 0, "R": 1}
        view_order = {"CC": 0, "MLO": 1}
        laterality = str(record.get("laterality", ""))
        view = str(record.get("view_position", ""))
        return (laterality_order.get(laterality, 99), view_order.get(view, 99), str(record.get("image_id", "")))

    def __len__(self) -> int:
        if self.index_level == "image":
            return len(self.image_records)
        if self.index_level == "exam":
            return len(self.study_ids)
        if self.index_level == "crop":
            return len(self.crop_records)
        raise ValueError(f"Unknown index_level: {self.index_level}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.index_level == "image":
            return self._get_image_item(index)
        if self.index_level == "exam":
            return self._get_exam_item(index)
        if self.index_level == "crop":
            return self._get_crop_item(index)
        raise ValueError(f"Unknown index_level: {self.index_level}")


    def _build_crop_index_if_needed(self) -> None:
        """Build lightweight crop records when ``index_level='crop'`` is requested."""
        self.crop_records: list[dict[str, Any]] = []
        if self.index_level != "crop":
            return
        if not bool(self.crop_options.get("enabled", False)):
            raise ValueError("index_level='crop' requires crop_options={'enabled': True, ...}.")

        mode = str(self.crop_options.get("mode", "random")).casefold()
        if mode == "random":
            reps = int(self.crop_options.get("random_crops_per_image", 1))
            iterable = self._progress(
                enumerate(self.image_records),
                desc="Building random crop index",
                total=len(self.image_records),
                unit="img",
            )
            for record_index, _ in iterable:
                for crop_number in range(max(1, reps)):
                    self.crop_records.append({"record_index": record_index, "crop_number": crop_number, "window_xyxy": None})
            return

        # Deterministic mode precomputes sliding windows after the normal geometry
        # preprocessing, because breast cropping changes the image size.
        iterable = self._progress(
            enumerate(self.image_records),
            desc="Building deterministic crop index",
            total=len(self.image_records),
            unit="img",
        )
        for record_index, record in iterable:
            image, target = self._read_preprocessed_record_no_square(record)
            height, width = int(image.shape[-2]), int(image.shape[-1])
            windows = sliding_square_windows(
                width=width,
                height=height,
                crop_size=int(self.crop_options["crop_size"]),
                stride=int(self.crop_options["stride"]),
            )
            kept = 0
            max_windows = self.crop_options.get("deterministic_max_windows_per_image")
            for window in windows:
                if not bool(self.crop_options.get("deterministic_include_empty", True)):
                    if not window_has_positive_mass(window, target["mass"]["boxes"], self.crop_options):
                        continue
                self.crop_records.append({"record_index": record_index, "crop_number": kept, "window_xyxy": window})
                kept += 1
                if max_windows is not None and kept >= int(max_windows):
                    break

    def _get_image_item(self, index: int) -> dict[str, Any]:
        """Return one image plus its target dictionary."""
        record = self.image_records[index]
        image, dicom_meta, output_shape, original_shape = self._maybe_read_image(record)
        target = self._make_target(record, output_shape=output_shape, original_shape=original_shape, dicom_meta=dicom_meta)
        image, target = self._apply_preprocessing(image, target)
        image, target = self._apply_square_crop_if_enabled(image, target)

        if self.transform is not None and image is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {
            "index": index,
            "index_level": "image",
            "image": image,
            "target": target,
        }
        if self.joint_transform is not None:
            sample = self.joint_transform(sample)
        return sample

    def _get_exam_item(self, index: int) -> dict[str, Any]:
        """Return one study with all available views and targets."""
        study_id = str(self.study_ids[index])
        records = self.records_by_study_id[study_id]

        images: list[torch.Tensor | None] = []
        targets: list[dict[str, Any]] = []
        for record in records:
            image, dicom_meta, output_shape, original_shape = self._maybe_read_image(record)
            target = self._make_target(record, output_shape=output_shape, original_shape=original_shape, dicom_meta=dicom_meta)
            image, target = self._apply_preprocessing(image, target)
            image, target = self._apply_square_crop_if_enabled(image, target)

            if self.transform is not None and image is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            images.append(image)
            targets.append(target)

        sample = {
            "index": index,
            "index_level": "exam",
            "study_id": study_id,
            "images": images,
            "targets": targets,
            "num_images": len(images),
        }
        if self.joint_transform is not None:
            sample = self.joint_transform(sample)
        return sample


    def _get_crop_item(self, index: int) -> dict[str, Any]:
        """Return one square crop sample for object detection training."""
        crop_record = self.crop_records[int(index)]
        record = self.image_records[int(crop_record["record_index"])]
        image, target = self._read_preprocessed_record_no_square(record)

        if crop_record.get("window_xyxy") is None:
            height, width = int(image.shape[-2]), int(image.shape[-1])
            window, random_info = sample_random_square_window(
                image_width=width,
                image_height=height,
                mass_boxes=target["mass"]["boxes"],
                options=self.crop_options,
                rng=self._crop_rng,
            )
        else:
            window = tuple(crop_record["window_xyxy"])
            random_info = {"requested_positive": None, "accepted": True}

        image, target = self._apply_square_window(image, target, window_xyxy=window, extra_info=random_info)

        if self.transform is not None and image is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {
            "index": index,
            "index_level": "crop",
            "source_image_index": int(crop_record["record_index"]),
            "crop_number": int(crop_record["crop_number"]),
            "image": image,
            "target": target,
        }
        if self.joint_transform is not None:
            sample = self.joint_transform(sample)
        return sample

    def _read_preprocessed_record_no_square(self, record: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read one image and apply inversion/breast-crop/mirror, but not square crops."""
        result = read_dicom_image(
            record["dicom_path"],
            normalize=self.normalize,
            percentile_range=self.percentile_range,
            use_voi_lut=self.use_voi_lut,
            strict_voi_lut=self.strict_voi_lut,
            output_size=self.output_size,
            add_channel_dim=True,
            return_dicom_meta=self.return_dicom_meta,
            invert_monochrome1=bool(self.preprocess_options.get("invert_to_black_background", False)),
        )
        target = self._make_target(record, output_shape=result.output_shape, original_shape=result.original_shape, dicom_meta=result.dicom_meta)
        image, target = self._apply_preprocessing(result.image, target)
        return image, target

    def _maybe_read_image(
        self, record: dict[str, Any]
    ) -> tuple[torch.Tensor | None, dict[str, Any], tuple[int, int] | None, tuple[int, int] | None]:
        if not self.read_image:
            return None, {}, None, None

        result = read_dicom_image(
            record["dicom_path"],
            normalize=self.normalize,
            percentile_range=self.percentile_range,
            use_voi_lut=self.use_voi_lut,
            strict_voi_lut=self.strict_voi_lut,
            output_size=self.output_size,
            add_channel_dim=True,
            return_dicom_meta=self.return_dicom_meta,
            invert_monochrome1=bool(self.preprocess_options.get("invert_to_black_background", False)),
        )
        return result.image, result.dicom_meta, result.output_shape, result.original_shape

    def _apply_preprocessing(
        self, image: torch.Tensor | None, target: dict[str, Any]
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Apply optional crop/mirror preprocessing and keep boxes aligned."""
        if image is None:
            target["preprocessing"] = dict(self.preprocess_options)
            return image, target

        if not (
            self.preprocess_options.get("crop_breast")
            or self.preprocess_options.get("mask_outside_breast")
            or self.preprocess_options.get("mirror_right_to_left")
        ):
            target["preprocessing"] = {
                **self.preprocess_options,
                "processed_shape": tuple(image.shape[-2:]),
                "mirrored": False,
                "crop_box_xyxy": None,
            }
            return image, target

        result = apply_geometry_preprocessing(
            image,
            boxes=target["boxes"],
            mass_boxes=target["mass"]["boxes"],
            options=self.preprocess_options,
        )
        image = result.image
        target["boxes"] = result.boxes
        target["output_shape"] = result.info["processed_shape"]
        target["height"], target["width"] = result.info["processed_shape"]
        target["preprocessing"] = {**self.preprocess_options, **result.info}
        if bool(self.preprocess_options.get("retain_breast_mask_for_export", False)) and result.foreground_mask is not None:
            target["_foreground_mask"] = result.foreground_mask

        mass = target["mass"]
        keep = result.mass_box_keep
        mass["boxes"] = result.mass_boxes
        if len(keep) == len(mass["labels"]):
            mass["labels"] = mass["labels"][keep]
            for key in ["finding_birads", "finding_birads_ids", "findings"]:
                mass[key] = [v for v, ok in zip(mass[key], keep.tolist()) if ok]
        mass["has_mass"] = int(mass["boxes"].shape[0]) > 0
        mass["num_masses"] = int(mass["boxes"].shape[0])
        mass["area_fractions"] = self._box_area_fractions(mass["boxes"], image_shape=result.info["processed_shape"])
        mass["area_percentages"] = mass["area_fractions"] * 100.0
        mass["size_bins"] = [self._mass_area_size_bin(float(x)) for x in mass["area_percentages"].tolist()]
        mass["shape_groups"] = [
            self._bbox_shape_group(float((box[2] - box[0]) / max(box[3] - box[1], 1e-12)))
            for box in mass["boxes"]
        ]
        target["has_mass"] = mass["has_mass"]
        target["num_masses"] = mass["num_masses"]
        return image, target


    def _apply_square_crop_if_enabled(
        self, image: torch.Tensor | None, target: dict[str, Any]
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Apply the final n x n crop stage in image/exam modes when enabled."""
        if image is None or not bool(self.crop_options.get("enabled", False)):
            target["square_crop"] = {"enabled": bool(self.crop_options.get("enabled", False))}
            return image, target

        mode = str(self.crop_options.get("mode", "random")).casefold()
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if mode == "deterministic":
            # In image/exam mode we cannot return all deterministic windows from one
            # index, so use the first deterministic window. Use index_level='crop'
            # when you want every sliding window as a separate training sample.
            windows = sliding_square_windows(width, height, int(self.crop_options["crop_size"]), int(self.crop_options["stride"]))
            window = windows[0]
            info = {"requested_positive": None, "accepted": True, "note": "first_deterministic_window_in_image_mode"}
        else:
            window, info = sample_random_square_window(
                image_width=width,
                image_height=height,
                mass_boxes=target["mass"]["boxes"],
                options=self.crop_options,
                rng=self._crop_rng,
            )
        return self._apply_square_window(image, target, window_xyxy=window, extra_info=info)

    def _apply_square_window(
        self,
        image: torch.Tensor,
        target: dict[str, Any],
        *,
        window_xyxy: tuple[int, int, int, int],
        extra_info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Crop to one square window and update all target boxes and mass fields."""
        result = crop_image_and_boxes_to_window(
            image,
            boxes=target["boxes"],
            mass_boxes=target["mass"]["boxes"],
            window_xyxy=window_xyxy,
            options=self.crop_options,
        )
        target = dict(target)
        target["boxes"] = result.boxes
        target["height"], target["width"] = result.info["crop_shape"]
        target["output_shape"] = result.info["crop_shape"]
        target["square_crop"] = {**self.crop_options, **result.info, **(extra_info or {})}

        mass = dict(target["mass"])
        keep = result.mass_box_keep
        mass["boxes"] = result.mass_boxes
        if len(keep) == len(mass["labels"]):
            mass["labels"] = mass["labels"][keep]
            for key in ["finding_birads", "finding_birads_ids", "findings"]:
                mass[key] = [v for v, ok in zip(mass[key], keep.tolist()) if ok]
        mass["has_mass"] = int(mass["boxes"].shape[0]) > 0
        mass["num_masses"] = int(mass["boxes"].shape[0])
        mass["area_fractions"] = self._box_area_fractions(mass["boxes"], image_shape=result.info["crop_shape"])
        mass["area_percentages"] = mass["area_fractions"] * 100.0
        mass["size_bins"] = [self._mass_area_size_bin(float(x)) for x in mass["area_percentages"].tolist()]
        mass["shape_groups"] = [
            self._bbox_shape_group(float((box[2] - box[0]) / max(box[3] - box[1], 1e-12)))
            for box in mass["boxes"]
        ]
        target["mass"] = mass
        target["has_mass"] = mass["has_mass"]
        target["num_masses"] = mass["num_masses"]
        return result.image, target

    def _make_target(
        self,
        record: dict[str, Any],
        *,
        output_shape: tuple[int, int] | None,
        original_shape: tuple[int, int] | None,
        dicom_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge breast-level labels, findings, boxes, metadata, and DICOM info."""
        image_id = str(record["image_id"])
        study_id = str(record["study_id"])
        findings = self.findings_by_image_id.get(image_id, [])
        boxes_original = self._boxes_from_findings(findings)
        boxes = self._scale_boxes(boxes_original, original_shape=original_shape, output_shape=output_shape)

        mass_findings = self._filter_mass_findings(findings)
        mass_boxes_original = self._boxes_from_findings(mass_findings)
        mass_boxes = self._scale_boxes(mass_boxes_original, original_shape=original_shape, output_shape=output_shape)
        mass_area_fractions_original = self._box_area_fractions(
            mass_boxes_original,
            image_shape=self._shape_from_record_or_dicom(record, original_shape),
        )
        mass_area_fractions = self._box_area_fractions(
            mass_boxes,
            image_shape=self._shape_from_record_or_dicom(record, output_shape),
        )

        breast_info = self._clean_record(record)
        metadata_records = self._metadata_for_record(record)

        finding_categories = [f.get("finding_categories_list", []) for f in findings]
        finding_category_ids = [f.get("finding_category_ids", []) for f in findings]

        target = {
            "image_id": image_id,
            "study_id": study_id,
            "series_id": str(record.get("series_id", "")),
            "dicom_path": str(record.get("dicom_path", "")),
            "laterality": record.get("laterality"),
            "view_position": record.get("view_position"),
            "split": record.get("split"),
            "height": self._safe_int(record.get("height")),
            "width": self._safe_int(record.get("width")),
            "breast_birads": record.get("breast_birads"),
            "breast_birads_id": birads_to_int(record.get("breast_birads")),
            "breast_density": record.get("breast_density"),
            "breast_density_id": density_to_int(record.get("breast_density")),
            "boxes": boxes,
            "boxes_original": boxes_original,
            # Mass-specific detection target. This is usually what you want for
            # mass detection models: boxes + class labels + mass rows only.
            "mass": {
                "has_mass": len(mass_findings) > 0,
                "num_masses": len(mass_findings),
                "boxes": mass_boxes,
                "boxes_original": mass_boxes_original,
                "labels": torch.ones((len(mass_findings),), dtype=torch.int64),
                "area_fractions": mass_area_fractions,
                "area_percentages": mass_area_fractions * 100.0,
                "area_fractions_original": mass_area_fractions_original,
                "area_percentages_original": mass_area_fractions_original * 100.0,
                "size_bins": [self._mass_area_size_bin(float(x)) for x in (mass_area_fractions * 100.0).tolist()],
                "shape_groups": [
                    self._bbox_shape_group(float((box[2] - box[0]) / max(box[3] - box[1], 1e-12)))
                    for box in mass_boxes
                ],
                "finding_birads": [f.get("finding_birads") for f in mass_findings],
                "finding_birads_ids": [f.get("finding_birads_id") for f in mass_findings],
                "findings": mass_findings,
            },
            "has_mass": len(mass_findings) > 0,
            "num_masses": len(mass_findings),
            "findings": findings,
            "finding_categories": finding_categories,
            "finding_category_ids": finding_category_ids,
            "num_findings": len(findings),
            "breast_annotation": breast_info,
            "metadata": metadata_records,
            "dicom_meta": dicom_meta,
            "original_shape": original_shape,
            "output_shape": output_shape,
        }
        return target

    @staticmethod
    def _is_mass_finding(finding: dict[str, Any]) -> bool:
        """Return True when a finding row contains the official ``Mass`` category."""
        categories = finding.get("finding_categories_list")
        if categories is None:
            categories = parse_finding_categories(finding.get("finding_categories"))
        return any(str(category).strip().casefold() == "mass" for category in categories)

    @classmethod
    def _filter_mass_findings(cls, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep valid bounding-box rows whose category list contains ``Mass``."""
        return [
            finding
            for finding in findings
            if cls._is_mass_finding(finding) and cls._finding_has_valid_box(finding)
        ]

    @staticmethod
    def _finding_has_valid_box(finding: dict[str, Any]) -> bool:
        try:
            xmin, ymin, xmax, ymax = [
                float(finding.get(key)) for key in ("xmin", "ymin", "xmax", "ymax")
            ]
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite([xmin, ymin, xmax, ymax]).all() and xmax > xmin and ymax > ymin)

    @staticmethod
    def _shape_from_record_or_dicom(record: dict[str, Any], dicom_shape: tuple[int, int] | None) -> tuple[int, int] | None:
        """Prefer explicit CSV height/width, then fall back to the read DICOM shape."""
        height = VindrMammoDataset._safe_int(record.get("height"))
        width = VindrMammoDataset._safe_int(record.get("width"))
        if height is not None and width is not None and height > 0 and width > 0:
            return height, width
        return dicom_shape

    @staticmethod
    def _box_area_fractions(boxes: torch.Tensor, image_shape: tuple[int, int] | None) -> torch.Tensor:
        """Compute box area divided by image area for each box."""
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        if image_shape is None:
            return torch.full((boxes.shape[0],), float("nan"), dtype=torch.float32)
        height, width = image_shape
        image_area = float(height) * float(width)
        if image_area <= 0:
            return torch.full((boxes.shape[0],), float("nan"), dtype=torch.float32)
        box_widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        box_heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return (box_widths * box_heights) / image_area

    def _metadata_for_record(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        image_id = str(record.get("image_id", ""))
        study_id = str(record.get("study_id", ""))
        if image_id in self.metadata_by_image_id:
            return [self._clean_record(r) for r in self.metadata_by_image_id[image_id]]
        if study_id in self.metadata_by_study_id:
            return [self._clean_record(r) for r in self.metadata_by_study_id[study_id]]
        return []

    def _clean_finding_record(self, record: dict[str, Any]) -> dict[str, Any]:
        clean = self._clean_record(record)
        categories = parse_finding_categories(clean.get("finding_categories"))
        clean["finding_categories_list"] = categories
        clean["finding_category_ids"] = finding_categories_to_ids(categories)
        clean["finding_birads_id"] = birads_to_int(clean.get("finding_birads"))
        return clean

    @staticmethod
    def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                out[key] = None
            elif isinstance(value, np.generic):
                out[key] = value.item()
            else:
                out[key] = value
        return out

    @staticmethod
    def _boxes_from_findings(findings: list[dict[str, Any]]) -> torch.Tensor:
        boxes: list[list[float]] = []
        for finding in findings:
            coords = [finding.get("xmin"), finding.get("ymin"), finding.get("xmax"), finding.get("ymax")]
            if any(v is None for v in coords):
                continue
            try:
                box = [float(v) for v in coords]
            except Exception:
                continue
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _scale_boxes(
        boxes: torch.Tensor,
        *,
        original_shape: tuple[int, int] | None,
        output_shape: tuple[int, int] | None,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.clone()
        if original_shape is None or output_shape is None:
            return boxes.clone()
        orig_h, orig_w = original_shape
        out_h, out_w = output_shape
        if orig_h == out_h and orig_w == out_w:
            return boxes.clone()

        scaled = boxes.clone()
        scaled[:, [0, 2]] *= float(out_w) / float(orig_w)
        scaled[:, [1, 3]] *= float(out_h) / float(orig_h)
        return scaled

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            if isinstance(value, float) and np.isnan(value):
                return None
            return int(value)
        except Exception:
            return None

    def mass_annotations_dataframe(self) -> pd.DataFrame:
        """Return one row per mass annotation with useful derived geometry.

        The official VinDr-Mammo finding file gives bounding boxes and finding
        categories, but it does not provide detailed mass morphology such as
        round, oval, irregular, circumscribed, or spiculated. The ``shape_group``
        in this table is therefore a bounding-box geometry descriptor based on
        aspect ratio:

        * ``wide``: width / height > 1.33
        * ``tall``: width / height < 0.75
        * ``square_like``: otherwise

        The ``area_fraction`` and ``area_percentage`` columns are the bounding
        box area divided by the whole image area. They are useful for estimating
        how small the mass target is relative to the full mammogram.
        """
        rows: list[dict[str, Any]] = []
        for raw_record in self.finding_df.to_dict(orient="records"):
            finding = self._clean_finding_record(raw_record)
            if not self._is_mass_finding(finding):
                continue

            coords = [finding.get("xmin"), finding.get("ymin"), finding.get("xmax"), finding.get("ymax")]
            try:
                xmin, ymin, xmax, ymax = [float(v) for v in coords]
            except Exception:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue

            image_height = self._safe_int(finding.get("height"))
            image_width = self._safe_int(finding.get("width"))
            box_width = xmax - xmin
            box_height = ymax - ymin
            box_area = box_width * box_height
            image_area = None
            area_fraction = None
            area_percentage = None
            if image_height is not None and image_width is not None and image_height > 0 and image_width > 0:
                image_area = float(image_height) * float(image_width)
                area_fraction = box_area / image_area
                area_percentage = 100.0 * area_fraction

            aspect_ratio = box_width / box_height if box_height > 0 else None
            shape_group = self._bbox_shape_group(aspect_ratio)
            size_bin = self._mass_area_size_bin(area_percentage)

            rows.append(
                {
                    "image_id": str(finding.get("image_id", "")),
                    "study_id": str(finding.get("study_id", "")),
                    "series_id": str(finding.get("series_id", "")),
                    "split": finding.get("split"),
                    "laterality": finding.get("laterality"),
                    "view_position": finding.get("view_position"),
                    "breast_birads": finding.get("breast_birads"),
                    "breast_birads_id": birads_to_int(finding.get("breast_birads")),
                    "breast_density": finding.get("breast_density"),
                    "breast_density_id": density_to_int(finding.get("breast_density")),
                    "finding_birads": finding.get("finding_birads"),
                    "finding_birads_id": birads_to_int(finding.get("finding_birads")),
                    "finding_categories": finding.get("finding_categories_list", []),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "bbox_width": box_width,
                    "bbox_height": box_height,
                    "bbox_area_px": box_area,
                    "image_height": image_height,
                    "image_width": image_width,
                    "image_area_px": image_area,
                    "area_fraction": area_fraction,
                    "area_percentage": area_percentage,
                    "aspect_ratio": aspect_ratio,
                    "shape_group": shape_group,
                    "size_bin": size_bin,
                }
            )
        return pd.DataFrame(rows)


    def _processed_annotation_tables(self, *, max_images: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute preprocessed mass-level and image-level tables in one DICOM pass."""
        cache_key = None if max_images is None else int(max_images)
        if cache_key in self._processed_stats_cache:
            mass_df, image_df = self._processed_stats_cache[cache_key]
            return mass_df.copy(), image_df.copy()

        mass_rows: list[dict[str, Any]] = []
        image_rows: list[dict[str, Any]] = []
        records = self.image_records if max_images is None else self.image_records[: int(max_images)]
        for record in self._progress(
            records,
            desc="Computing preprocessed statistics",
            total=len(records),
            unit="img",
        ):
            image, target = self._read_preprocessed_record_no_square(record)
            mass = target["mass"]
            boxes = mass["boxes"].detach().cpu().numpy()
            image_height, image_width = int(image.shape[-2]), int(image.shape[-1])
            image_area = float(image_height) * float(image_width)

            image_rows.append(
                {
                    "image_id": target["image_id"],
                    "study_id": target["study_id"],
                    "split": target.get("split"),
                    "laterality": target.get("laterality"),
                    "view_position": target.get("view_position"),
                    "breast_birads": target.get("breast_birads"),
                    "breast_density": target.get("breast_density"),
                    "height": image_height,
                    "width": image_width,
                    "has_mass": bool(mass["has_mass"]),
                    "num_masses": int(mass["num_masses"]),
                    "crop_box_xyxy": target.get("preprocessing", {}).get("crop_box_xyxy"),
                    "mirrored": target.get("preprocessing", {}).get("mirrored"),
                }
            )

            for box_index, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = [float(v) for v in box]
                if xmax <= xmin or ymax <= ymin:
                    continue
                box_width = xmax - xmin
                box_height = ymax - ymin
                box_area = box_width * box_height
                area_fraction = box_area / image_area if image_area > 0 else None
                area_percentage = None if area_fraction is None else 100.0 * area_fraction
                aspect_ratio = box_width / box_height if box_height > 0 else None
                finding = mass["findings"][box_index] if box_index < len(mass["findings"]) else {}
                mass_rows.append(
                    {
                        "image_id": target["image_id"],
                        "study_id": target["study_id"],
                        "series_id": target.get("series_id"),
                        "split": target.get("split"),
                        "laterality": target.get("laterality"),
                        "view_position": target.get("view_position"),
                        "breast_birads": target.get("breast_birads"),
                        "breast_birads_id": target.get("breast_birads_id"),
                        "breast_density": target.get("breast_density"),
                        "breast_density_id": target.get("breast_density_id"),
                        "finding_birads": mass["finding_birads"][box_index] if box_index < len(mass["finding_birads"]) else finding.get("finding_birads"),
                        "finding_birads_id": mass["finding_birads_ids"][box_index] if box_index < len(mass["finding_birads_ids"]) else birads_to_int(finding.get("finding_birads")),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_width": box_width,
                        "bbox_height": box_height,
                        "bbox_area_px": box_area,
                        "image_height": image_height,
                        "image_width": image_width,
                        "image_area_px": image_area,
                        "area_fraction": area_fraction,
                        "area_percentage": area_percentage,
                        "aspect_ratio": aspect_ratio,
                        "shape_group": self._bbox_shape_group(aspect_ratio),
                        "size_bin": self._mass_area_size_bin(area_percentage),
                        "crop_box_xyxy": target.get("preprocessing", {}).get("crop_box_xyxy"),
                        "mirrored": target.get("preprocessing", {}).get("mirrored"),
                    }
                )

        mass_df = pd.DataFrame(mass_rows)
        image_df = pd.DataFrame(image_rows)
        self._processed_stats_cache[cache_key] = (mass_df, image_df)
        return mass_df.copy(), image_df.copy()

    def processed_mass_annotations_dataframe(self, *, max_images: int | None = None) -> pd.DataFrame:
        """Return one row per mass after inversion, breast crop, and mirroring.

        This reads DICOM pixels because the breast crop and mirror decision depend
        on image content. Results are cached inside the dataset instance, so
        repeated statistics/plot calls do not reread all images.
        """
        return self._processed_annotation_tables(max_images=max_images)[0]

    def processed_image_mass_dataframe(self, *, max_images: int | None = None) -> pd.DataFrame:
        """Return one row per image after geometry preprocessing.

        This shares the same cached DICOM pass used by
        ``processed_mass_annotations_dataframe``.
        """
        return self._processed_annotation_tables(max_images=max_images)[1]

    def image_mass_dataframe(self) -> pd.DataFrame:
        """Return one row per indexed image with mass counts and breast labels."""
        mass_df = self.mass_annotations_dataframe()
        mass_counts = mass_df.groupby("image_id").size().rename("num_masses") if not mass_df.empty else pd.Series(dtype=int)
        rows = []
        for record in self.image_records:
            image_id = str(record.get("image_id", ""))
            num_masses = int(mass_counts.get(image_id, 0)) if len(mass_counts) else 0
            rows.append(
                {
                    "image_id": image_id,
                    "study_id": str(record.get("study_id", "")),
                    "split": record.get("split"),
                    "laterality": record.get("laterality"),
                    "view_position": record.get("view_position"),
                    "breast_birads": record.get("breast_birads"),
                    "breast_density": record.get("breast_density"),
                    "height": self._safe_int(record.get("height")),
                    "width": self._safe_int(record.get("width")),
                    "has_mass": num_masses > 0,
                    "num_masses": num_masses,
                }
            )
        return pd.DataFrame(rows)

    def statistics(self, *, stage: str = "raw", max_images: int | None = None) -> dict[str, Any]:
        """Compute dataset statistics.

        ``stage="raw"`` uses CSV coordinates and does not read DICOM pixels.
        ``stage="preprocessed"`` reads DICOMs and recomputes mass statistics
        after inversion, breast cropping, and mirroring. Square crop statistics
        are handled by the crop-specific methods.
        """
        stage = stage.casefold().strip()
        if stage == "raw":
            mass_df = self.mass_annotations_dataframe()
            image_mass_df = self.image_mass_dataframe()
        elif stage in {"processed", "preprocessed", "after"}:
            stage = "preprocessed"
            mass_df = self.processed_mass_annotations_dataframe(max_images=max_images)
            image_mass_df = self.processed_image_mass_dataframe(max_images=max_images)
        else:
            raise ValueError("stage must be raw or preprocessed")
        manufacturer_col = self._find_column(
            self.metadata_df,
            ["Manufacturer", "manufacturer", "dicom_manufacturer", "00080070"],
        )
        model_col = self._find_column(
            self.metadata_df,
            ["Manufacturer's Model Name", "ManufacturerModelName", "manufacturer_model_name", "model_name", "00081090"],
        )

        stats: dict[str, Any] = self.summary()
        stats["stage"] = stage
        stats["vendor_counts"] = self._value_counts_dict(self.metadata_df[manufacturer_col]) if manufacturer_col else {}
        stats["model_counts"] = self._value_counts_dict(self.metadata_df[model_col]) if model_col else {}
        stats["metadata_columns"] = list(self.metadata_df.columns)

        stats["mass"] = {
            "num_mass_annotations": int(len(mass_df)),
            "num_images_with_mass": int(image_mass_df["has_mass"].sum()) if not image_mass_df.empty else 0,
            "percent_images_with_mass": float(100.0 * image_mass_df["has_mass"].mean()) if not image_mass_df.empty else 0.0,
            "mass_count_per_image": self._describe_numeric(image_mass_df["num_masses"]) if not image_mass_df.empty else {},
            "finding_birads_counts": self._value_counts_dict(mass_df["finding_birads"]) if "finding_birads" in mass_df else {},
            "breast_birads_counts_for_mass_images": self._value_counts_dict(mass_df["breast_birads"]) if "breast_birads" in mass_df else {},
            "breast_density_counts_for_mass_images": self._value_counts_dict(mass_df["breast_density"]) if "breast_density" in mass_df else {},
            "view_counts": self._value_counts_dict(mass_df["view_position"]) if "view_position" in mass_df else {},
            "laterality_counts": self._value_counts_dict(mass_df["laterality"]) if "laterality" in mass_df else {},
            "shape_group_counts": self._value_counts_dict(mass_df["shape_group"]) if "shape_group" in mass_df else {},
            "size_bin_counts": self._value_counts_dict(mass_df["size_bin"]) if "size_bin" in mass_df else {},
            "area_percentage": self._describe_numeric(mass_df["area_percentage"]) if "area_percentage" in mass_df else {},
            "aspect_ratio": self._describe_numeric(mass_df["aspect_ratio"]) if "aspect_ratio" in mass_df else {},
            "bbox_width": self._describe_numeric(mass_df["bbox_width"]) if "bbox_width" in mass_df else {},
            "bbox_height": self._describe_numeric(mass_df["bbox_height"]) if "bbox_height" in mass_df else {},
        }
        return stats

    def print_statistics(self, *, stage: str = "raw", max_images: int | None = None) -> None:
        """Print a readable dataset and mass-statistics summary."""
        stats = self.statistics(stage=stage, max_images=max_images)
        print(f"\nVinDr-Mammo statistics ({stats.get('stage', 'raw')})")
        print("=" * 72)
        print(f"data_root: {stats['data_root']}")
        print(f"index_level: {stats['index_level']}")
        print(f"num_images: {stats['num_images']}")
        print(f"num_studies: {stats['num_studies']}")
        print(f"num_finding_rows: {stats['num_finding_rows']}")
        print(f"split_counts: {stats['split_counts']}")
        print(f"breast_birads_counts: {stats['breast_birads_counts']}")
        print(f"breast_density_counts: {stats['breast_density_counts']}")
        print(f"vendor_counts: {stats['vendor_counts']}")
        print(f"model_counts: {stats['model_counts']}")

        mass = stats["mass"]
        print("\nMass statistics")
        print("-" * 72)
        print(f"num_mass_annotations: {mass['num_mass_annotations']}")
        print(f"num_images_with_mass: {mass['num_images_with_mass']}")
        print(f"percent_images_with_mass: {mass['percent_images_with_mass']:.2f}%")
        print(f"finding_birads_counts: {mass['finding_birads_counts']}")
        print(f"shape_group_counts: {mass['shape_group_counts']}")
        print(f"size_bin_counts: {mass['size_bin_counts']}")
        print(f"area_percentage: {mass['area_percentage']}")
        print(f"aspect_ratio: {mass['aspect_ratio']}")

    def save_statistics_tables(self, output_dir: str | Path = "outputs") -> list[Path]:
        """Save statistics tables as CSV files and return their paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        mass_df = self.mass_annotations_dataframe()
        image_mass_df = self.image_mass_dataframe()
        tables = {
            "mass_annotations.csv": mass_df,
            "image_mass_summary.csv": image_mass_df,
            "breast_birads_counts.csv": self._counts_dataframe(self.breast_df, "breast_birads"),
            "breast_density_counts.csv": self._counts_dataframe(self.breast_df, "breast_density"),
            "split_counts.csv": self._counts_dataframe(self.breast_df, "split"),
        }

        manufacturer_col = self._find_column(
            self.metadata_df,
            ["Manufacturer", "manufacturer", "dicom_manufacturer", "00080070"],
        )
        model_col = self._find_column(
            self.metadata_df,
            ["Manufacturer's Model Name", "ManufacturerModelName", "manufacturer_model_name", "model_name", "00081090"],
        )
        if manufacturer_col:
            tables["manufacturer_counts.csv"] = self._counts_dataframe(self.metadata_df, manufacturer_col)
        if model_col:
            tables["manufacturer_model_counts.csv"] = self._counts_dataframe(self.metadata_df, model_col)
        if not mass_df.empty:
            tables["mass_shape_group_counts.csv"] = self._counts_dataframe(mass_df, "shape_group")
            tables["mass_size_bin_counts.csv"] = self._counts_dataframe(mass_df, "size_bin")
            tables["mass_finding_birads_counts.csv"] = self._counts_dataframe(mass_df, "finding_birads")
            tables["mass_area_percentage_summary.csv"] = pd.DataFrame([self._describe_numeric(mass_df["area_percentage"])])
            tables["mass_aspect_ratio_summary.csv"] = pd.DataFrame([self._describe_numeric(mass_df["aspect_ratio"])])

        for filename, table in tables.items():
            path = output_dir / filename
            table.to_csv(path, index=False)
            saved.append(path)

        stats_path = output_dir / "statistics_summary.json"
        stats_path.write_text(self._json_dumps(self.statistics()), encoding="utf-8")
        saved.append(stats_path)
        return saved

    def save_statistics_plots(self, output_dir: str | Path = "outputs", *, dpi: int = 150, top_k: int = 20) -> list[Path]:
        """Save dataset and mass-statistics plots into ``output_dir``.

        The method only uses CSV metadata and annotation files, not DICOM pixels.
        It creates plots for split, breast BI-RADS, density, vendor/model, mass
        BI-RADS, mass bounding-box shape groups, mass area percentage, aspect
        ratio, and mass count per image.
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        def save_bar_from_series(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
            nonlocal saved
            counts = series.fillna("Unknown").astype(str).value_counts().head(top_k)
            if counts.empty:
                return
            fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(counts)), 4.5))
            counts.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)

        if "split" in self.breast_df:
            save_bar_from_series(self.breast_df["split"], "Official split distribution", "Split", "Images", "split_distribution.png")
        if "breast_birads" in self.breast_df:
            save_bar_from_series(
                self.breast_df["breast_birads"],
                "Breast BI-RADS distribution",
                "BI-RADS",
                "Images",
                "breast_birads_distribution.png",
            )
        if "breast_density" in self.breast_df:
            save_bar_from_series(
                self.breast_df["breast_density"],
                "Breast density distribution",
                "Density",
                "Images",
                "breast_density_distribution.png",
            )

        manufacturer_col = self._find_column(
            self.metadata_df,
            ["Manufacturer", "manufacturer", "dicom_manufacturer", "00080070"],
        )
        model_col = self._find_column(
            self.metadata_df,
            ["Manufacturer's Model Name", "ManufacturerModelName", "manufacturer_model_name", "model_name", "00081090"],
        )
        if manufacturer_col:
            save_bar_from_series(self.metadata_df[manufacturer_col], "Scanner manufacturer distribution", "Manufacturer", "Rows", "manufacturer_distribution.png")
        if model_col:
            save_bar_from_series(self.metadata_df[model_col], "Scanner model distribution", "Model", "Rows", "manufacturer_model_distribution.png")

        image_mass_df = self.image_mass_dataframe()
        if not image_mass_df.empty:
            save_bar_from_series(
                image_mass_df["num_masses"],
                "Mass count per image",
                "Number of mass boxes in image",
                "Images",
                "mass_count_per_image_distribution.png",
            )

        mass_df = self.mass_annotations_dataframe()
        if not mass_df.empty:
            save_bar_from_series(mass_df["finding_birads"], "Mass finding BI-RADS distribution", "Finding BI-RADS", "Mass boxes", "mass_finding_birads_distribution.png")
            save_bar_from_series(mass_df["shape_group"], "Mass bounding-box shape distribution", "BBox shape group", "Mass boxes", "mass_bbox_shape_distribution.png")
            save_bar_from_series(mass_df["size_bin"], "Mass area-size bin distribution", "Area bin", "Mass boxes", "mass_area_size_bin_distribution.png")
            save_bar_from_series(mass_df["view_position"], "Mass view-position distribution", "View", "Mass boxes", "mass_view_distribution.png")
            save_bar_from_series(mass_df["breast_density"], "Breast density for mass boxes", "Density", "Mass boxes", "mass_breast_density_distribution.png")

            self._save_histogram(
                mass_df["area_percentage"],
                title="Mass box area as percentage of full image",
                xlabel="Area percentage (%)",
                ylabel="Mass boxes",
                filename=output_dir / "mass_area_percentage_histogram.png",
                dpi=dpi,
            )
            saved.append(output_dir / "mass_area_percentage_histogram.png")

            self._save_histogram(
                mass_df["aspect_ratio"],
                title="Mass bounding-box aspect ratio",
                xlabel="Width / height",
                ylabel="Mass boxes",
                filename=output_dir / "mass_aspect_ratio_histogram.png",
                dpi=dpi,
            )
            saved.append(output_dir / "mass_aspect_ratio_histogram.png")

            self._save_scatter(
                x=mass_df["bbox_width"],
                y=mass_df["bbox_height"],
                title="Mass bounding-box width vs height",
                xlabel="Width (pixels)",
                ylabel="Height (pixels)",
                filename=output_dir / "mass_bbox_width_vs_height.png",
                dpi=dpi,
            )
            saved.append(output_dir / "mass_bbox_width_vs_height.png")

        return saved


    def save_processed_statistics_tables(self, output_dir: str | Path = "outputs", *, max_images: int | None = None) -> list[Path]:
        """Save statistics tables after geometry preprocessing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        mass_df = self.processed_mass_annotations_dataframe(max_images=max_images)
        image_mass_df = self.processed_image_mass_dataframe(max_images=max_images)
        tables = {
            "mass_annotations.csv": mass_df,
            "image_mass_summary.csv": image_mass_df,
            "breast_birads_counts.csv": self._counts_dataframe(image_mass_df, "breast_birads"),
            "breast_density_counts.csv": self._counts_dataframe(image_mass_df, "breast_density"),
            "split_counts.csv": self._counts_dataframe(image_mass_df, "split"),
        }
        if not mass_df.empty:
            tables["mass_shape_group_counts.csv"] = self._counts_dataframe(mass_df, "shape_group")
            tables["mass_size_bin_counts.csv"] = self._counts_dataframe(mass_df, "size_bin")
            tables["mass_finding_birads_counts.csv"] = self._counts_dataframe(mass_df, "finding_birads")
            tables["mass_area_percentage_summary.csv"] = pd.DataFrame([self._describe_numeric(mass_df["area_percentage"])])
            tables["mass_aspect_ratio_summary.csv"] = pd.DataFrame([self._describe_numeric(mass_df["aspect_ratio"])])

        for filename, table in tables.items():
            path = output_dir / filename
            table.to_csv(path, index=False)
            saved.append(path)
        stats_path = output_dir / "statistics_summary.json"
        stats_path.write_text(self._json_dumps(self.statistics(stage="preprocessed", max_images=max_images)), encoding="utf-8")
        saved.append(stats_path)
        return saved

    def save_processed_statistics_plots(
        self, output_dir: str | Path = "outputs", *, dpi: int = 150, top_k: int = 20, max_images: int | None = None
    ) -> list[Path]:
        """Save plots after geometry preprocessing."""
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        def save_bar_from_series(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
            nonlocal saved
            if series is None or len(series) == 0:
                return
            counts = series.fillna("Unknown").astype(str).value_counts().head(top_k)
            if counts.empty:
                return
            fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(counts)), 4.5))
            counts.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)

        image_mass_df = self.processed_image_mass_dataframe(max_images=max_images)
        mass_df = self.processed_mass_annotations_dataframe(max_images=max_images)
        if not image_mass_df.empty:
            save_bar_from_series(image_mass_df["split"], "Official split distribution after preprocessing", "Split", "Images", "split_distribution.png")
            save_bar_from_series(image_mass_df["breast_birads"], "Breast BI-RADS after preprocessing", "BI-RADS", "Images", "breast_birads_distribution.png")
            save_bar_from_series(image_mass_df["breast_density"], "Breast density after preprocessing", "Density", "Images", "breast_density_distribution.png")
            save_bar_from_series(image_mass_df["num_masses"], "Mass count per image after preprocessing", "Number of mass boxes", "Images", "mass_count_per_image_distribution.png")
            save_bar_from_series(image_mass_df["mirrored"], "Images mirrored by preprocessing", "Mirrored", "Images", "mirrored_distribution.png")
        if not mass_df.empty:
            save_bar_from_series(mass_df["finding_birads"], "Mass finding BI-RADS after preprocessing", "Finding BI-RADS", "Mass boxes", "mass_finding_birads_distribution.png")
            save_bar_from_series(mass_df["shape_group"], "Mass box shape after preprocessing", "BBox shape group", "Mass boxes", "mass_bbox_shape_distribution.png")
            save_bar_from_series(mass_df["size_bin"], "Mass area-size bin after preprocessing", "Area bin", "Mass boxes", "mass_area_size_bin_distribution.png")
            self._save_histogram(
                mass_df["area_percentage"],
                title="Mass box area percentage after preprocessing",
                xlabel="Area percentage (%)",
                ylabel="Mass boxes",
                filename=output_dir / "mass_area_percentage_histogram.png",
                dpi=dpi,
            )
            saved.append(output_dir / "mass_area_percentage_histogram.png")
            self._save_scatter(
                x=mass_df["bbox_width"],
                y=mass_df["bbox_height"],
                title="Mass box width vs height after preprocessing",
                xlabel="Width (pixels)",
                ylabel="Height (pixels)",
                filename=output_dir / "mass_bbox_width_vs_height.png",
                dpi=dpi,
            )
            saved.append(output_dir / "mass_bbox_width_vs_height.png")
        return saved

    def crop_size_fit_statistics(self, *, crop_size: int | None = None, stage: str = "preprocessed") -> pd.DataFrame:
        """Estimate what square crop size n is needed to contain mass boxes.

        Two bases are reported:
        1. ``single_mass_box``: each mass annotation independently.
        2. ``all_mass_boxes_in_same_image``: the square size needed to cover all
           masses from the same image at once.
        """
        if crop_size is None:
            crop_size = int(self.crop_options.get("crop_size", 1024))
        stage = stage.casefold().strip()
        if stage in {"processed", "preprocessed", "after"}:
            mass_df = self.processed_mass_annotations_dataframe()
        elif stage == "raw":
            mass_df = self.mass_annotations_dataframe()
        else:
            raise ValueError("stage must be raw or preprocessed")
        table = crop_size_fit_table(mass_df, current_crop_size=crop_size)
        table.insert(0, "stage", "preprocessed" if stage in {"processed", "preprocessed", "after"} else "raw")
        return table

    def save_crop_size_report(
        self, output_dir: str | Path = "outputs", *, crop_size: int | None = None, stage: str = "preprocessed", dpi: int = 150
    ) -> dict[str, Path]:
        """Save crop-size recommendation table and a mass-size histogram."""
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if crop_size is None:
            crop_size = int(self.crop_options.get("crop_size", 1024))
        stage = stage.casefold().strip()
        mass_df = self.processed_mass_annotations_dataframe() if stage in {"processed", "preprocessed", "after"} else self.mass_annotations_dataframe()
        table = crop_size_fit_table(mass_df, current_crop_size=crop_size)
        table_path = output_dir / f"crop_size_fit_statistics_{stage}.csv"
        table.to_csv(table_path, index=False)

        plot_path = output_dir / f"crop_size_fit_histogram_{stage}.png"
        if not mass_df.empty:
            values = np.maximum(pd.to_numeric(mass_df["bbox_width"], errors="coerce"), pd.to_numeric(mass_df["bbox_height"], errors="coerce"))
            values = pd.Series(values).dropna()
            if not values.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(values, bins=40)
                ax.axvline(float(crop_size), linestyle="--", label=f"current n={crop_size}")
                ax.set_title(f"Minimum n to fit each mass box ({stage})")
                ax.set_xlabel("Minimum square crop size n (pixels)")
                ax.set_ylabel("Mass boxes")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_path, dpi=dpi)
                plt.close(fig)
        return {"table": table_path, "plot": plot_path}

    def crop_dataset_statistics(self, *, max_crops: int | None = 1000) -> pd.DataFrame:
        """Sample or enumerate crop samples and return crop-level statistics."""
        if not bool(self.crop_options.get("enabled", False)):
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        if self.index_level == "crop":
            iterable = range(len(self.crop_records))
            if max_crops is not None:
                iterable = range(min(len(self.crop_records), int(max_crops)))
            total = len(iterable) if hasattr(iterable, "__len__") else None
            for idx in self._progress(iterable, desc="Computing crop stats", total=total, unit="crop"):
                sample = self._get_crop_item(idx)
                target = sample["target"]
                rows.append(self._crop_stats_row(idx, target))
        else:
            records = self.image_records if max_crops is None else self.image_records[: int(max_crops)]
            iterable = self._progress(
                enumerate(records),
                desc="Sampling crop stats",
                total=len(records),
                unit="img",
            )
            for idx, record in iterable:
                image, target = self._read_preprocessed_record_no_square(record)
                window, info = sample_random_square_window(
                    image_width=int(image.shape[-1]),
                    image_height=int(image.shape[-2]),
                    mass_boxes=target["mass"]["boxes"],
                    options=self.crop_options,
                    rng=self._crop_rng,
                )
                _, target = self._apply_square_window(image, target, window_xyxy=window, extra_info=info)
                rows.append(self._crop_stats_row(idx, target))
        return pd.DataFrame(rows)

    def save_crop_dataset_statistics_report(self, output_dir: str | Path = "outputs", *, max_crops: int | None = 1000) -> Path | None:
        """Save sampled crop-level statistics as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.crop_dataset_statistics(max_crops=max_crops)
        if df.empty:
            return None
        path = output_dir / "crop_dataset_statistics.csv"
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def _crop_stats_row(index: int, target: dict[str, Any]) -> dict[str, Any]:
        crop = target.get("square_crop", {})
        return {
            "crop_index": index,
            "image_id": target.get("image_id"),
            "study_id": target.get("study_id"),
            "window_xyxy": crop.get("window_xyxy"),
            "crop_size": crop.get("crop_size"),
            "is_positive_crop": crop.get("is_positive_crop"),
            "num_masses": target.get("mass", {}).get("num_masses"),
            "mass_area_percentages": target.get("mass", {}).get("area_percentages", torch.zeros(0)).tolist(),
            "requested_positive": crop.get("requested_positive"),
            "accepted": crop.get("accepted"),
        }

    def show_square_crop_test(
        self,
        *,
        mode: str | None = None,
        index: int | None = None,
        output_dir: str | Path = "outputs",
        filename: str | None = None,
        show: bool = True,
        save: bool = True,
    ) -> Path | None:
        """Save/show a test of the final n x n object-detection crop stage."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        record = self._visual_test_record(index, step="crop_generation")
        image, target = self._read_preprocessed_record_no_square(record)
        opts = dict(self.crop_options)
        opts["enabled"] = True
        if mode is not None:
            opts["mode"] = mode
        opts = make_crop_options(opts)
        old_options = self.crop_options
        try:
            self.crop_options = opts
            if opts["mode"] == "deterministic":
                windows = sliding_square_windows(int(image.shape[-1]), int(image.shape[-2]), int(opts["crop_size"]), int(opts["stride"]))
                window = next((w for w in windows if window_has_positive_mass(w, target["mass"]["boxes"], opts)), windows[0])
                info = {"requested_positive": None, "accepted": True}
            else:
                window, info = sample_random_square_window(
                    image_width=int(image.shape[-1]),
                    image_height=int(image.shape[-2]),
                    mass_boxes=target["mass"]["boxes"],
                    options=opts,
                    rng=self._crop_rng,
                )
            crop_image, crop_target = self._apply_square_window(image, target, window_xyxy=window, extra_info=info)
        finally:
            self.crop_options = old_options

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        full_np = image.squeeze(0).detach().cpu().numpy()
        crop_np = crop_image.squeeze(0).detach().cpu().numpy()
        axes[0].imshow(full_np, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Preprocessed full image + selected n x n window")
        x0, y0, x1, y1 = [float(v) for v in window]
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2.0))
        for box in target["mass"]["boxes"].detach().cpu().numpy():
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            axes[0].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=1.5))

        axes[1].imshow(crop_np, cmap="gray")
        axes[1].axis("off")
        axes[1].set_title(f"Final crop | masses={crop_target['mass']['num_masses']}")
        for box in crop_target["mass"]["boxes"].detach().cpu().numpy():
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            axes[1].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=2.0))
        fig.tight_layout()

        saved_path: Path | None = None
        if save:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = filename or f"square_crop_test_{opts['mode']}.png"
            saved_path = output_dir / filename
            fig.savefig(saved_path, dpi=140, bbox_inches="tight")
            print(f"Saved square crop test to: {saved_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return saved_path

    def show_mass_animation(
        self,
        *,
        output_dir: str | Path = "outputs",
        only_with_mass: bool = True,
        max_images: int | None = None,
        start_index: int = 0,
        interval_ms: int = 700,
        figsize: tuple[float, float] = (8.0, 10.0),
        cmap: str = "gray",
        box_color: str = "red",
        text_color: str = "yellow",
        linewidth: float = 2.0,
        show_labels: bool = True,
        show: bool = True,
        save_gif: bool = False,
        gif_name: str = "mass_annotations_animation.gif",
        dpi: int = 120,
        animation_stage: str = "auto",
    ) -> Any:
        """Open an animation that shows mammograms with mass boxes overlaid.

        By default, this method follows the current dataset mode. If
        ``index_level='crop'``, it animates the final n x n crop samples, after
        inversion, breast cropping, mirroring, and square-crop box adjustment.
        If ``index_level='image'`` or ``'exam'``, it animates image records after
        the enabled preprocessing steps. Frames are read on demand, so it does
        not load all full-resolution mammograms into memory at once.

        Parameters
        ----------
        output_dir:
            Directory used when ``save_gif=True``. The GIF is saved there.
        only_with_mass:
            If ``True``, animate only images with mass annotations. If ``False``,
            animate all indexed images and draw boxes only where masses exist.
        max_images:
            Optional maximum number of frames. This is useful for quick checks,
            because animating every mass-positive image can take a long time.
        start_index:
            Start frame offset after filtering. Useful if you want to inspect a
            later part of the mass-positive list.
        interval_ms:
            Delay between frames in milliseconds.
        figsize:
            Matplotlib figure size.
        cmap:
            Colormap used to display grayscale mammograms.
        box_color:
            Rectangle color for mass boxes.
        text_color:
            Text color for labels.
        linewidth:
            Rectangle line width.
        show_labels:
            If ``True``, write finding BI-RADS and area percentage near each box.
        show:
            If ``True``, call ``plt.show()`` and open a Matplotlib window.
        save_gif:
            If ``True``, save the animation as a GIF under ``output_dir``.
        gif_name:
            GIF filename used when ``save_gif=True``.
        dpi:
            DPI used when saving the GIF.
        animation_stage:
            ``"auto"`` follows the dataset mode. In crop mode it shows final
            n x n crop samples. ``"preprocessed"`` forces full preprocessed
            images before the square-crop stage. ``"crop"`` forces crop
            animation and requires square cropping to be enabled.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object. The object is also stored in
            ``self._last_mass_animation`` to prevent garbage collection while
            the window is open.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle

        frame_sources, resolved_stage = self._mass_animation_frame_sources(
            only_with_mass=only_with_mass,
            max_frames=max_images,
            start_index=start_index,
            animation_stage=animation_stage,
        )
        if not frame_sources:
            mode = "mass-positive" if only_with_mass else "indexed"
            raise ValueError(f"No {mode} frames were found for animation stage={resolved_stage!r}.")

        fig, ax = plt.subplots(figsize=figsize)

        def update(frame_index: int) -> list[Any]:
            source = frame_sources[frame_index]
            image, target = self._read_mass_animation_source(source)
            image_np = image.squeeze(0).detach().cpu().numpy() if image is not None else np.zeros((1, 1), dtype=np.float32)

            ax.clear()
            ax.imshow(image_np, cmap=cmap)
            ax.axis("off")

            mass = target["mass"]
            boxes = mass["boxes"].detach().cpu().numpy()
            area_percentages = mass["area_percentages"].detach().cpu().numpy()
            finding_birads = mass["finding_birads"]
            size_bins = mass.get("size_bins", [])

            for box_index, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = [float(v) for v in box]
                width = xmax - xmin
                height = ymax - ymin
                rect = Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    fill=False,
                    edgecolor=box_color,
                    linewidth=linewidth,
                )
                ax.add_patch(rect)

                if show_labels:
                    birads = finding_birads[box_index] if box_index < len(finding_birads) else "Mass"
                    area = area_percentages[box_index] if box_index < len(area_percentages) else float("nan")
                    size_bin = size_bins[box_index] if box_index < len(size_bins) else ""
                    label = f"{birads} | {area:.3f}% | {size_bin}"
                    ax.text(
                        xmin,
                        max(0.0, ymin - 8.0),
                        label,
                        color=text_color,
                        fontsize=8,
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                    )

            height, width = int(image_np.shape[0]), int(image_np.shape[1])
            title = (
                f"{frame_index + 1}/{len(frame_sources)} | "
                f"image_id={target['image_id']} | study_id={target['study_id']} | "
                f"{target.get('laterality')} {target.get('view_position')} | "
                f"shape={height}x{width} | masses={mass['num_masses']}"
            )
            square_crop = target.get("square_crop", {})
            if square_crop.get("square_crop_enabled"):
                title += f" | square_crop={square_crop.get('window_xyxy')}"
            title += f" | stage={resolved_stage}"
            ax.set_title(title)
            return []

        anim = FuncAnimation(fig, update, frames=len(frame_sources), interval=interval_ms, repeat=True, blit=False)
        self._last_mass_animation = anim

        if save_gif:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if resolved_stage == "crop" and gif_name == "mass_annotations_animation.gif":
                gif_name = "mass_annotations_square_crops.gif"
            gif_path = output_dir / gif_name
            if self.show_progress and tqdm is not None:
                progress_bar = tqdm(total=len(frame_sources), desc="Saving mass GIF", unit="frame")

                def _gif_progress_callback(current_frame: int, total_frames: int) -> None:
                    # Matplotlib reports a zero-based frame counter. Update by the delta
                    # so tqdm remains accurate even if frames are skipped or retried.
                    desired = min(int(current_frame) + 1, int(total_frames))
                    progress_bar.update(max(0, desired - progress_bar.n))

                try:
                    anim.save(gif_path, writer="pillow", dpi=dpi, progress_callback=_gif_progress_callback)
                finally:
                    progress_bar.close()
            else:
                anim.save(gif_path, writer="pillow", dpi=dpi)
            print(f"Saved mass annotation animation to: {gif_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim

    # Short alias for interactive use.
    animate_mass_annotations = show_mass_animation

    def show_preprocessing_test(
        self,
        *,
        step: str = "all",
        index: int | None = None,
        output_dir: str | Path = "outputs",
        filename: str | None = None,
        show: bool = True,
        save: bool = True,
    ) -> Path | None:
        """Show a before/after test for one preprocessing step.

        ``step`` can be ``"inversion"``, ``"crop"``, ``"mirror"``, or ``"all"``.
        The method saves a side-by-side PNG to ``output_dir`` when ``save=True``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        step = step.casefold().strip()
        if step not in {"inversion", "crop", "mirror", "all"}:
            raise ValueError("step must be one of: inversion, crop, mirror, all")

        record = self._visual_test_record(index, step=step)
        raw = self._read_record_variant(record, invert=False, geometry_options={"crop_breast": False, "mirror_right_to_left": False})

        if step == "inversion":
            before = raw
            after = self._read_record_variant(record, invert=True, geometry_options={"crop_breast": False, "mirror_right_to_left": False})
        elif step == "crop":
            invert = bool(self.preprocess_options.get("invert_to_black_background", True))
            before = self._read_record_variant(record, invert=invert, geometry_options={"crop_breast": False, "mirror_right_to_left": False})
            after = self._read_record_variant(record, invert=invert, geometry_options={"crop_breast": True, "mirror_right_to_left": False})
        elif step == "mirror":
            invert = bool(self.preprocess_options.get("invert_to_black_background", True))
            before_geom = {"crop_breast": bool(self.preprocess_options.get("crop_breast", False)), "mirror_right_to_left": False}
            after_geom = {"crop_breast": bool(self.preprocess_options.get("crop_breast", False)), "mirror_right_to_left": True}
            before = self._read_record_variant(record, invert=invert, geometry_options=before_geom)
            after = self._read_record_variant(record, invert=invert, geometry_options=after_geom)
        else:
            before = raw
            after = self._read_record_variant(
                record,
                invert=bool(self.preprocess_options.get("invert_to_black_background", False)),
                geometry_options={
                    "crop_breast": bool(self.preprocess_options.get("crop_breast", False)),
                    "mirror_right_to_left": bool(self.preprocess_options.get("mirror_right_to_left", False)),
                },
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        for ax, item, title in zip(axes, [before, after], ["Before", "After"]):
            image_np = item["image"].squeeze(0).detach().cpu().numpy()
            target = item["target"]
            ax.imshow(image_np, cmap="gray")
            ax.axis("off")
            ax.set_title(f"{title}: {step}\n{target['image_id']} | {target.get('laterality')} {target.get('view_position')}")
            for box in target["mass"]["boxes"].detach().cpu().numpy():
                xmin, ymin, xmax, ymax = [float(v) for v in box]
                ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=2.0))
        fig.tight_layout()

        saved_path: Path | None = None
        if save:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = filename or f"preprocessing_test_{step}.png"
            saved_path = output_dir / filename
            fig.savefig(saved_path, dpi=140, bbox_inches="tight")
            print(f"Saved preprocessing test to: {saved_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return saved_path

    def show_inversion_test(self, **kwargs: Any) -> Path | None:
        return self.show_preprocessing_test(step="inversion", **kwargs)

    def show_crop_test(self, **kwargs: Any) -> Path | None:
        return self.show_preprocessing_test(step="crop", **kwargs)

    def show_mirror_test(self, **kwargs: Any) -> Path | None:
        return self.show_preprocessing_test(step="mirror", **kwargs)

    def _visual_test_record(self, index: int | None, *, step: str = "all", max_search: int = 250) -> dict[str, Any]:
        """Pick a record where the requested visual test actually changes the image."""
        if index is not None:
            return self.image_records[int(index)]

        # Prefer mass-positive images because they also show whether boxes remain aligned.
        candidates = self._mass_animation_records(only_with_mass=True) + list(self.image_records)
        seen: set[str] = set()
        unique_candidates: list[dict[str, Any]] = []
        for record in candidates:
            image_id = str(record.get("image_id", ""))
            if image_id not in seen:
                seen.add(image_id)
                unique_candidates.append(record)

        search_records = unique_candidates[:max_search]
        for record in self._progress(
            search_records,
            desc=f"Finding changed example ({step})",
            total=len(search_records),
            unit="img",
        ):
            try:
                if self._step_changes_record(record, step=step):
                    return record
            except Exception:
                continue
        print(f"Warning: could not find a clear changed example for preprocessing step={step!r}; using the first available image.")
        return unique_candidates[0] if unique_candidates else self.image_records[0]

    def _step_changes_record(self, record: dict[str, Any], *, step: str) -> bool:
        step = step.casefold().strip()
        if step == "inversion":
            return self._record_would_invert(record)
        if step == "crop":
            item = self._read_record_variant(
                record,
                invert=bool(self.preprocess_options.get("invert_to_black_background", True)),
                geometry_options={"crop_breast": True, "mirror_right_to_left": False},
            )
            crop_box = item["target"].get("preprocessing", {}).get("crop_box_xyxy")
            shape0 = item["target"].get("preprocessing", {}).get("original_shape")
            if crop_box is None or shape0 is None:
                return False
            h, w = shape0
            return tuple(crop_box) != (0, 0, int(w), int(h))
        if step == "mirror":
            invert = bool(self.preprocess_options.get("invert_to_black_background", True))
            before_geom = {"crop_breast": bool(self.preprocess_options.get("crop_breast", False)), "mirror_right_to_left": False}
            after_geom = {"crop_breast": bool(self.preprocess_options.get("crop_breast", False)), "mirror_right_to_left": True}
            before = self._read_record_variant(record, invert=invert, geometry_options=before_geom)
            after = self._read_record_variant(record, invert=invert, geometry_options=after_geom)
            if after["target"].get("preprocessing", {}).get("mirrored"):
                return True
            # fallback: compare arrays if needed
            return not torch.equal(before["image"], after["image"])
        if step in {"all", "crop_generation"}:
            return self._step_changes_record(record, step="inversion") or self._step_changes_record(record, step="crop") or self._step_changes_record(record, step="mirror")
        return False

    @staticmethod
    def _record_would_invert(record: dict[str, Any]) -> bool:
        try:
            import pydicom

            ds = pydicom.dcmread(str(record["dicom_path"]), stop_before_pixels=True, force=True)
            return str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
        except Exception:
            return False

    def _read_record_variant(self, record: dict[str, Any], *, invert: bool, geometry_options: dict[str, Any]) -> dict[str, Any]:
        result = read_dicom_image(
            record["dicom_path"],
            normalize=self.normalize,
            percentile_range=self.percentile_range,
            use_voi_lut=self.use_voi_lut,
            strict_voi_lut=self.strict_voi_lut,
            output_size=self.output_size,
            add_channel_dim=True,
            return_dicom_meta=False,
            invert_monochrome1=invert,
        )
        target = self._make_target(record, output_shape=result.output_shape, original_shape=result.original_shape, dicom_meta={})
        old_options = self.preprocess_options
        try:
            self.preprocess_options = make_preprocess_options({**old_options, **geometry_options})
            image, target = self._apply_preprocessing(result.image, target)
        finally:
            self.preprocess_options = old_options
        return {"image": image, "target": target}

    def _mass_animation_frame_sources(
        self,
        *,
        only_with_mass: bool,
        max_frames: int | None,
        start_index: int,
        animation_stage: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Build frame sources for mass animation.

        The returned frame sources are intentionally lightweight. For crop mode,
        each source points to a crop dataset index, so the animation uses exactly
        the same crop image and crop-coordinate target that ``dataset[index]``
        returns during object-detection training.
        """
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")

        stage = str(animation_stage or "auto").casefold().strip()
        if stage == "auto":
            stage = "crop" if self.index_level == "crop" else "preprocessed"
        if stage not in {"preprocessed", "crop"}:
            raise ValueError("animation_stage must be 'auto', 'preprocessed', or 'crop'.")
        if stage == "crop" and not bool(self.crop_options.get("enabled", False)):
            raise ValueError("animation_stage='crop' requires crop_options={'enabled': True, ...}.")

        if stage == "preprocessed":
            records = self._mass_animation_records(only_with_mass=only_with_mass)
            records = records[start_index:]
            if max_frames is not None:
                records = records[: int(max_frames)]
            return [{"kind": "preprocessed", "record": r} for r in records], stage

        # Crop animation: materialize the final n x n crop samples that will
        # appear in the GIF/window. This is intentional. In random crop mode,
        # calling __getitem__ twice can sample two different crop windows, so
        # storing the selected sample here guarantees that the displayed frame
        # is the exact square-crop image and crop-coordinate target.
        crop_indices = range(start_index, len(self.crop_records))
        frame_sources: list[dict[str, Any]] = []
        scan_total = max(0, len(self.crop_records) - start_index)
        iterable = self._progress(crop_indices, desc="Selecting square-crop animation frames", total=scan_total, unit="crop")
        for crop_index in iterable:
            sample = self._get_crop_item(int(crop_index))
            target = sample["target"]
            if only_with_mass and not bool(target.get("mass", {}).get("has_mass", False)):
                continue
            frame_sources.append(
                {
                    "kind": "ready",
                    "index": int(crop_index),
                    "image": sample["image"],
                    "target": target,
                }
            )
            if max_frames is not None and len(frame_sources) >= int(max_frames):
                break
        return frame_sources, stage

    def _read_mass_animation_source(self, source: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read one animation frame source as final image plus final target."""
        kind = source.get("kind")
        if kind == "ready":
            return source["image"], source["target"]
        if kind == "crop":
            sample = self._get_crop_item(int(source["index"]))
            return sample["image"], sample["target"]
        if kind == "preprocessed":
            return self._read_record_for_mass_visualization(source["record"])
        raise ValueError(f"Unknown animation frame source kind: {kind!r}")

    def _mass_animation_records(self, *, only_with_mass: bool) -> list[dict[str, Any]]:
        """Return image records used by the mass annotation animation."""
        if not only_with_mass:
            return list(self.image_records)
        mass_df = self.mass_annotations_dataframe()
        if mass_df.empty or "image_id" not in mass_df.columns:
            return []
        mass_image_ids = set(mass_df["image_id"].astype(str).tolist())
        return [record for record in self.image_records if str(record.get("image_id", "")) in mass_image_ids]

    def _read_record_for_mass_visualization(self, record: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read a DICOM image and build a correctly scaled target for visualization."""
        result = read_dicom_image(
            record["dicom_path"],
            normalize=self.normalize,
            percentile_range=self.percentile_range,
            use_voi_lut=self.use_voi_lut,
            strict_voi_lut=self.strict_voi_lut,
            output_size=self.output_size,
            add_channel_dim=True,
            return_dicom_meta=False,
            invert_monochrome1=bool(self.preprocess_options.get("invert_to_black_background", False)),
        )
        target = self._make_target(
            record,
            output_shape=result.output_shape,
            original_shape=result.original_shape,
            dicom_meta={},
        )
        image, target = self._apply_preprocessing(result.image, target)
        return image, target

    def save_statistics_report(
        self,
        output_dir: str | Path = "outputs",
        *,
        include_preprocessed: bool = True,
        include_crop_reports: bool = True,
        max_processed_images: int | None = None,
        max_crop_stats: int | None = 1000,
    ) -> dict[str, list[Path]]:
        """Save statistics before and after preprocessing.

        The raw report is saved to ``output_dir/raw``. The preprocessed report
        is saved to ``output_dir/preprocessed`` and recomputes box areas after
        breast cropping and mirroring. Crop-size and crop-sampling reports are
        saved under ``output_dir/crops``.
        """
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"
        pre_dir = output_dir / "preprocessed"
        crop_dir = output_dir / "crops"

        table_paths = self.save_statistics_tables(raw_dir)
        plot_paths = self.save_statistics_plots(raw_dir)

        if include_preprocessed:
            table_paths.extend(self.save_processed_statistics_tables(pre_dir, max_images=max_processed_images))
            plot_paths.extend(self.save_processed_statistics_plots(pre_dir, max_images=max_processed_images))

        if include_crop_reports:
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_report = self.save_crop_size_report(crop_dir, stage="preprocessed")
            table_paths.append(crop_report["table"])
            if crop_report.get("plot") is not None:
                plot_paths.append(crop_report["plot"])
            crop_stats_path = self.save_crop_dataset_statistics_report(crop_dir, max_crops=max_crop_stats)
            if crop_stats_path is not None:
                table_paths.append(crop_stats_path)

        return {"tables": table_paths, "plots": plot_paths}

    # Alias with a shorter name for interactive use.
    plot_statistics = save_statistics_plots

    @staticmethod
    def _bbox_shape_group(aspect_ratio: float | None) -> str | None:
        if aspect_ratio is None or not np.isfinite(aspect_ratio):
            return None
        if aspect_ratio > 1.33:
            return "wide"
        if aspect_ratio < 0.75:
            return "tall"
        return "square_like"

    @staticmethod
    def _mass_area_size_bin(area_percentage: float | None) -> str | None:
        if area_percentage is None or not np.isfinite(area_percentage):
            return None
        if area_percentage < SIZE_TINY:
            return f"tiny (<{SIZE_TINY:.2f}%)"
        if area_percentage < SIZE_VERY_SMALL:
            return f"very_small ({SIZE_TINY:.2f}-{SIZE_VERY_SMALL:.2f}%)"
        if area_percentage < SIZE_SMALL:
            return f"small ({SIZE_VERY_SMALL:.2f}-{SIZE_SMALL:.2f}%)"
        if area_percentage < SIZE_MEDIUM:
            return f"medium ({SIZE_SMALL:.2f}-{SIZE_MEDIUM:.2f}%)"
        return f"large (>={SIZE_LARGE:.2f}%)"

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return "".join(ch for ch in str(name).casefold() if ch.isalnum())

    @classmethod
    def _find_column(cls, df: pd.DataFrame, candidates: list[str]) -> str | None:
        normalized_to_original = {cls._normalize_column_name(c): c for c in df.columns}
        for candidate in candidates:
            normalized = cls._normalize_column_name(candidate)
            if normalized in normalized_to_original:
                return normalized_to_original[normalized]
        return None

    @staticmethod
    def _value_counts_dict(series: pd.Series) -> dict[str, int]:
        counts = series.fillna("Unknown").astype(str).value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}

    @staticmethod
    def _counts_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            return pd.DataFrame(columns=[column, "count"])
        counts = df[column].fillna("Unknown").astype(str).value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]
        return counts

    @staticmethod
    def _describe_numeric(series: pd.Series) -> dict[str, float | int | None]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return {"count": 0}
        return {
            "count": int(values.shape[0]),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }

    @staticmethod
    def _json_dumps(value: Any) -> str:
        def default(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return str(obj)
        import json

        return json.dumps(value, indent=2, ensure_ascii=False, default=default)

    @staticmethod
    def _save_histogram(series: pd.Series, *, title: str, xlabel: str, ylabel: str, filename: Path, dpi: int) -> None:
        import matplotlib.pyplot as plt

        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=40)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def _save_scatter(*, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: Path, dpi: int) -> None:
        import matplotlib.pyplot as plt

        x_values = pd.to_numeric(x, errors="coerce")
        y_values = pd.to_numeric(y, errors="coerce")
        mask = x_values.notna() & y_values.notna()
        if not mask.any():
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x_values[mask], y_values[mask], s=10, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without loading DICOM pixel data."""
        split_counts = self.breast_df["split"].value_counts(dropna=False).to_dict() if "split" in self.breast_df else {}
        birads_counts = (
            self.breast_df["breast_birads"].value_counts(dropna=False).to_dict()
            if "breast_birads" in self.breast_df
            else {}
        )
        density_counts = (
            self.breast_df["breast_density"].value_counts(dropna=False).to_dict()
            if "breast_density" in self.breast_df
            else {}
        )
        return {
            "data_root": str(self.data_root),
            "index_level": self.index_level,
            "num_indexed_items": len(self),
            "num_images": len(self.image_records),
            "num_studies": len(self.study_ids),
            "num_finding_rows": int(len(self.finding_df)),
            "split_counts": split_counts,
            "breast_birads_counts": birads_counts,
            "breast_density_counts": density_counts,
        }


def vindr_mammo_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that keeps variable-sized mammograms and targets as lists.

    Full-resolution mammograms can have large and variable shapes, and detection targets
    have a variable number of boxes. This collate function avoids accidental stacking.
    """
    if not batch:
        return {}

    index_level = batch[0].get("index_level")
    if index_level == "image":
        return {
            "images": [item["image"] for item in batch],
            "targets": [item["target"] for item in batch],
            "indices": [item["index"] for item in batch],
        }

    if index_level == "exam":
        return {
            "exams": batch,
            "study_ids": [item["study_id"] for item in batch],
            "indices": [item["index"] for item in batch],
        }

    if index_level == "crop":
        return {
            "images": [item["image"] for item in batch],
            "targets": [item["target"] for item in batch],
            "indices": [item["index"] for item in batch],
            "source_image_indices": [item["source_image_index"] for item in batch],
            "crop_numbers": [item["crop_number"] for item in batch],
        }

    return {"items": batch}
