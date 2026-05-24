from __future__ import annotations

import os
import sys
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

# Allows running this file directly with the VSCode Run button without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vindr_mammo import VindrMammoDataset, vindr_mammo_collate  # noqa: E402

# Change this path, or set VINDR_MAMMO_ROOT in .vscode/launch.json or your system environment.
# It should be the folder that contains metadata.csv, breast-level_annotations.csv,
# finding_annotations.csv, and the images folder.
DATA_ROOT = r"G:/vindr"

# Choose "image" for one item per image, "exam" for one item per study,
# or "crop" for one object-detection crop per item.
USE_SQUARE_CROP_DATASET = True
INDEX_LEVEL = "crop" if USE_SQUARE_CROP_DATASET else "image"

# Use None for all data, or "training" / "test" for the official VinDr-Mammo split.
# None keeps all 20,000 images in image mode, or all 5,000 exams in exam mode.
SPLIT = None

# Keep original mammogram size by default.
# Do not resize here unless you explicitly want to create fixed-size tensors.
OUTPUT_SIZE = None  # Optional example: (1024, 768) for (height, width)

# Optional preprocessing. Geometry-changing steps update the mass boxes.
PREPROCESS_OPTIONS = {
    "invert_to_black_background": True,  # MONOCHROME1 -> MONOCHROME2-style display.
    "crop_breast": True,                # Remove pure background as tightly as possible.
    "mirror_right_to_left": True,        # If breast enters from right, flip so it enters from left.
    "crop_padding": 20,
    "crop_threshold": None,             # None = automatic threshold.
}

# Final optional n x n square crop stage for object detection training.
# This is applied after inversion, breast crop, and mirroring.
CROP_OPTIONS = {
    "enabled": USE_SQUARE_CROP_DATASET,
    "mode": "random",                  # "random" or "deterministic"
    "crop_size": 1024,                  # n for n x n crops
    "stride": 768,                      # deterministic sliding-window stride
    "random_crops_per_image": 1,
    "positive_fraction": 0.80,          # example: 80% mass-positive, 20% clean
    "center_on_mass": True,
    "center_shift_fraction": 0.25,
    "allow_partial_annotations": False, # False means keep only fully included boxes
    "min_box_visibility": 0.30,
    "reject_partial_windows": True,
    "negative_max_box_visibility": 0.0,
    "pad_if_needed": True,
    "pad_value": 0.0,
    "max_random_tries": 80,
    "deterministic_include_empty": True,
    "deterministic_max_windows_per_image": None,
    "seed": 123,
}

# Set this False if you only want to inspect CSV annotations without reading DICOM pixels.
READ_IMAGE = True

# Shows tqdm progress bars for long operations in the VSCode terminal.
SHOW_PROGRESS = True

# Dataset/statistics output directory. Statistics tables and plots are saved here.
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Raw statistics use CSVs. Preprocessed statistics read DICOMs because crop/mirror changes geometry.
SAVE_STATISTICS = False
MAX_PROCESSED_STAT_IMAGES = None  # Optional quick test: 200
SAVE_CROP_SIZE_REPORT = False
SAVE_SQUARE_CROP_TEST = True

# Opens a Matplotlib animation window showing mass-positive images with boxes.
OPEN_MASS_ANIMATION = True
SAVE_MASS_ANIMATION_GIF = True
MASS_ANIMATION_MAX_IMAGES = 10  # Keep the GIF quick while debugging square crops.
MASS_ANIMATION_INTERVAL_MS = 700
MASS_ANIMATION_STAGE = "crop" if USE_SQUARE_CROP_DATASET else "preprocessed"

# Saves quick before/after PNGs for inversion, crop, and mirror preprocessing.
SAVE_PREPROCESSING_TESTS = True
SHOW_PREPROCESSING_TESTS = False


def resolve_data_root() -> Path:
    env_root = os.environ.get("VINDR_MAMMO_ROOT")
    if env_root:
        return Path(env_root)
    return Path(DATA_ROOT)


def print_image_sample(sample: dict) -> None:
    image = sample["image"]
    target = sample["target"]

    print("\nFirst image-indexed sample")
    print("image shape:", None if image is None else tuple(image.shape))
    print("image dtype:", None if image is None else image.dtype)
    print("image_id:", target["image_id"])
    print("study_id:", target["study_id"])
    print("view:", target["laterality"], target["view_position"])
    print("split:", target["split"])
    print("breast_birads:", target["breast_birads"], "->", target["breast_birads_id"])
    print("breast_density:", target["breast_density"], "->", target["breast_density_id"])
    print("num_findings:", target["num_findings"])
    print("boxes shape:", tuple(target["boxes"].shape))
    print("has_mass:", target["mass"]["has_mass"])
    print("num_masses:", target["mass"]["num_masses"])
    print("mass boxes shape:", tuple(target["mass"]["boxes"].shape))
    print("mass area percentages:", target["mass"]["area_percentages"].tolist())
    print("mass size bins:", target["mass"].get("size_bins", []))
    print("dicom_path:", target["dicom_path"])
    if target["findings"]:
        print("first finding:")
        pprint(target["findings"][0])
    if target["metadata"]:
        print("first metadata record:")
        pprint(target["metadata"][0])


def print_exam_sample(sample: dict) -> None:
    print("\nFirst exam-indexed sample")
    print("study_id:", sample["study_id"])
    print("num_images:", sample["num_images"])
    for i, (image, target) in enumerate(zip(sample["images"], sample["targets"])):
        print(
            f"  image {i}: shape={None if image is None else tuple(image.shape)}, "
            f"view={target['laterality']} {target['view_position']}, "
            f"birads={target['breast_birads']}, findings={target['num_findings']}, "
            f"masses={target['mass']['num_masses']}"
        )


def main() -> None:
    data_root = resolve_data_root()
    if not data_root.exists() or str(data_root).startswith("CHANGE_ME"):
        print("Please set DATA_ROOT in main.py or VINDR_MAMMO_ROOT in .vscode/launch.json.")
        print("Expected folder: .../vindr-mammo/1.0.0")
        print("It must contain metadata.csv, breast-level_annotations.csv, finding_annotations.csv, and images/.")
        return

    dataset = VindrMammoDataset(
        data_root=data_root,
        index_level=INDEX_LEVEL,
        split=SPLIT,
        read_image=READ_IMAGE,
        output_size=OUTPUT_SIZE,
        normalize="minmax",
        use_voi_lut=False,
        return_dicom_meta=True,
        validate_paths=False,
        preprocess_options=PREPROCESS_OPTIONS,
        crop_options=CROP_OPTIONS,
        show_progress=SHOW_PROGRESS,
    )

    print("Dataset summary")
    pprint(dataset.summary())

    if SAVE_STATISTICS:
        dataset.print_statistics(stage="raw")
        dataset.print_statistics(stage="preprocessed", max_images=MAX_PROCESSED_STAT_IMAGES)
        saved = dataset.save_statistics_report(
            OUTPUT_DIR,
            include_preprocessed=True,
            include_crop_reports=SAVE_CROP_SIZE_REPORT,
            max_processed_images=MAX_PROCESSED_STAT_IMAGES,
        )
        print(f"\nSaved {len(saved['tables'])} statistics tables and {len(saved['plots'])} plots under: {OUTPUT_DIR}")

    if SAVE_PREPROCESSING_TESTS or SHOW_PREPROCESSING_TESTS:
        dataset.show_inversion_test(output_dir=OUTPUT_DIR, show=SHOW_PREPROCESSING_TESTS, save=SAVE_PREPROCESSING_TESTS)
        dataset.show_crop_test(output_dir=OUTPUT_DIR, show=SHOW_PREPROCESSING_TESTS, save=SAVE_PREPROCESSING_TESTS)
        dataset.show_mirror_test(output_dir=OUTPUT_DIR, show=SHOW_PREPROCESSING_TESTS, save=SAVE_PREPROCESSING_TESTS)

    if SAVE_SQUARE_CROP_TEST:
        dataset.show_square_crop_test(output_dir=OUTPUT_DIR, show=False, save=True)

    if OPEN_MASS_ANIMATION or SAVE_MASS_ANIMATION_GIF:
        dataset.show_mass_animation(
            output_dir=OUTPUT_DIR,
            only_with_mass=True,
            max_images=MASS_ANIMATION_MAX_IMAGES,
            interval_ms=MASS_ANIMATION_INTERVAL_MS,
            show=OPEN_MASS_ANIMATION,
            save_gif=SAVE_MASS_ANIMATION_GIF,
            animation_stage=MASS_ANIMATION_STAGE,
        )

    sample = dataset[0]
    if INDEX_LEVEL in {"image", "crop"}:
        print_image_sample(sample)
        if INDEX_LEVEL == "crop":
            print("crop window:", sample["target"].get("square_crop", {}).get("window_xyxy"))
    else:
        print_exam_sample(sample)

    # Example DataLoader. Batch size is small because mammograms are large.
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Keep 0 for easier debugging in VSCode on Windows.
        collate_fn=vindr_mammo_collate,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))
    print("\nDataLoader batch keys:", list(batch.keys()))
    if INDEX_LEVEL in {"image", "crop"}:
        print("batch image shapes:", [None if x is None else tuple(x.shape) for x in batch["images"]])
        print("batch target image_ids:", [t["image_id"] for t in batch["targets"]])
        if INDEX_LEVEL == "crop":
            print("batch crop windows:", [t.get("square_crop", {}).get("window_xyxy") for t in batch["targets"]])
    else:
        print("batch study_ids:", batch["study_ids"])


if __name__ == "__main__":
    main()
