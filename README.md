# VinDr-Mammo PyTorch Dataset Reader

This project gives you a simple PyTorch `Dataset` for the VinDr-Mammo dataset. It is meant to be opened in VSCode and run directly from `main.py`, without needing console arguments.

The official dataset layout contains three CSV files and an `images` folder. The images are DICOM files stored as `images/<study_id>/<image_id>.dicom`.

## What this project supports

- Image-level indexing: one item per DICOM image, usually 20,000 items.
- Exam-level indexing: one item per study, usually 5,000 items, with the available views grouped together.
- Official split filtering through the dataset's own `split` column.
- Original DICOM image size by default. No resizing unless you explicitly set `output_size`.
- Annotation merging from:
  - `breast-level_annotations.csv`
  - `finding_annotations.csv`
  - `metadata.csv`
- Bounding box extraction as `[xmin, ymin, xmax, ymax]` tensors.
- Mass-specific target output in `target["mass"]`, including mass boxes, class labels, mass findings, area percentages, area-size bins, and bounding-box shape groups.
- Dataset statistics and plotting utilities for mass boxes, BI-RADS, breast density, scanner vendor/model, split distributions, and mass annotation animation.
- All statistics plots and CSV tables can be saved to an `outputs/` directory.
- Long operations show `tqdm` progress bars in the VSCode terminal by default.
- A safe DataLoader `collate_fn` that keeps images in lists, so variable image sizes are preserved.

## Expected dataset folder

After downloading from PhysioNet, set the root to the folder that looks like this:

```text
vindr-mammo/1.0.0/
  metadata.csv
  breast-level_annotations.csv
  finding_annotations.csv
  images/
    <study_id>/
      <image_id>.dicom
      <image_id>.dicom
      <image_id>.dicom
      <image_id>.dicom
```

## Installation

Create and activate your environment, then install the requirements:

```bash
pip install -r requirements.txt
```

The DICOM stack uses `pydicom` plus `pylibjpeg` plugins. If your DICOM files are compressed and pydicom cannot decode them, install an additional handler such as `python-gdcm`, if available for your platform.

## Run in VSCode

Open this project folder in VSCode.

Option A: edit `main.py`:

```python
DATA_ROOT = r"G:/vindr"
```

Option B: edit `.vscode/launch.json`:

```json
"VINDR_MAMMO_ROOT": "G:/vindr"
```

Then press the Run button on `main.py`.

## Basic usage

```python
from vindr_mammo import VindrMammoDataset, vindr_mammo_collate
from torch.utils.data import DataLoader

# One item per image, usually 20,000 items.
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="image",
    split=None,
    output_size=None,  # keep original image size
)

sample = dataset[0]
image = sample["image"]
target = sample["target"]
print(image.shape)
print(target["breast_birads"])
print(target["boxes"])
print(target["mass"]["boxes"])
print(target["mass"]["area_percentages"])
print(target["mass"]["size_bins"])

loader = DataLoader(dataset, batch_size=2, collate_fn=vindr_mammo_collate)
batch = next(iter(loader))
```

```python
# One item per exam, usually 5,000 items, with views grouped together.
exam_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="exam",
    split="training",
    output_size=None,  # keep original image size
)

exam = exam_dataset[0]
print(exam["study_id"])
print(len(exam["images"]))
for image, target in zip(exam["images"], exam["targets"]):
    print(image.shape, target["laterality"], target["view_position"])
```

## Dataset initialization parameters

The main class is:

```python
VindrMammoDataset(
    data_root,
    *,
    index_level="image",
    split=None,
    read_image=True,
    transform=None,
    target_transform=None,
    joint_transform=None,
    normalize="minmax",
    percentile_range=(0.5, 99.5),
    use_voi_lut=False,
    output_size=None,
    return_dicom_meta=False,
    validate_paths=False,
    preprocess_options=None,
    crop_options=None,
)
```

### `data_root`

Path to the downloaded VinDr-Mammo `1.0.0` folder. This is the folder that directly contains the three CSV files and the `images` folder.

Correct:

```python
data_root = r"G:/vindr"
```

Wrong:

```python
data_root = r"G:/vindr/images"
```

### `index_level`

Controls what one dataset index returns.

Use image mode when you want one DICOM image per item:

```python
dataset = VindrMammoDataset(DATA_ROOT, index_level="image")
sample = dataset[0]
image = sample["image"]
target = sample["target"]
```

Use exam mode when you want one study per item:

```python
dataset = VindrMammoDataset(DATA_ROOT, index_level="exam")
sample = dataset[0]
images = sample["images"]
targets = sample["targets"]
```

### `split`

Uses the official `split` column from the dataset annotations.

```python
# all data
dataset = VindrMammoDataset(DATA_ROOT, split=None)

# official training split
train_dataset = VindrMammoDataset(DATA_ROOT, split="training")

# official test split
test_dataset = VindrMammoDataset(DATA_ROOT, split="test")
```

You can also pass an iterable:

```python
dataset = VindrMammoDataset(DATA_ROOT, split=["training", "test"])
```

### `read_image`

If `True`, `__getitem__` reads the DICOM pixel data. If `False`, the dataset only returns annotations, paths, boxes, and metadata. This is useful when you first want to check the CSV information without waiting for large DICOM files to load.

```python
dataset = VindrMammoDataset(DATA_ROOT, read_image=False)
sample = dataset[0]
print(sample["image"])  # None
print(sample["target"]["dicom_path"])
```

### `transform`

Image-only transform. It receives a tensor shaped `[1, H, W]`.

Use this only for operations that do not change geometry, or when you do not care about box coordinates. If the transform resizes, crops, or flips the image, boxes will not be automatically updated here.

```python
def my_transform(image):
    return image.float()

dataset = VindrMammoDataset(DATA_ROOT, transform=my_transform)
```

### `target_transform`

Target-only transform. It receives the target dictionary and returns a modified target dictionary.

```python
def my_target_transform(target):
    target["is_positive"] = target["breast_birads_id"] is not None and target["breast_birads_id"] >= 3
    return target

dataset = VindrMammoDataset(DATA_ROOT, target_transform=my_target_transform)
```

### `joint_transform`

Full-sample transform. Use this when the image and target must be changed together, for example image cropping plus box coordinate updates.

```python
def my_joint_transform(sample):
    image = sample["image"]
    target = sample["target"]
    # Apply image and box transforms together here.
    sample["image"] = image
    sample["target"] = target
    return sample

dataset = VindrMammoDataset(DATA_ROOT, joint_transform=my_joint_transform)
```

### `normalize`

Controls pixel normalization in `read_dicom_image`.

Allowed values:

| Value | Meaning |
|---|---|
| `"none"` | Keep the numeric pixel scale after DICOM modality LUT. |
| `"minmax"` | Scale each image to `[0, 1]` using its own minimum and maximum. This is the default. |
| `"percentile"` | Clip each image to `percentile_range`, then scale to `[0, 1]`. |
| `"zscore"` | Convert each image to zero mean and unit standard deviation. |

Example:

```python
dataset = VindrMammoDataset(DATA_ROOT, normalize="percentile", percentile_range=(0.5, 99.5))
```

### `percentile_range`

Only used when `normalize="percentile"`. The tuple is `(lower_percentile, upper_percentile)`.

```python
dataset = VindrMammoDataset(
    DATA_ROOT,
    normalize="percentile",
    percentile_range=(1.0, 99.0),
)
```

### `use_voi_lut`

If `True`, applies DICOM VOI LUT or windowing when present.

Default is `False`. For training, leaving it `False` is usually safer because it avoids display-style windowing. For visualization, setting it to `True` can sometimes produce images closer to how they are displayed in a DICOM viewer.

```python
dataset = VindrMammoDataset(DATA_ROOT, use_voi_lut=True)
```

### `output_size`

Optional resize as `(height, width)`.

Default is `None`, which means original DICOM size is preserved.

```python
# Original size, recommended for your current stage.
dataset = VindrMammoDataset(DATA_ROOT, output_size=None)

# Explicit resize, only if you want fixed-size images.
dataset = VindrMammoDataset(DATA_ROOT, output_size=(1024, 768))
```

When `output_size` is not `None`:

- `target["boxes"]` is scaled to the resized image.
- `target["boxes_original"]` keeps the original CSV coordinates.

When `output_size=None`:

- `target["boxes"]` and `target["boxes_original"]` are the same coordinates.
- The returned image keeps its original DICOM height and width.

### `return_dicom_meta`

If `True`, returns a small dictionary of useful DICOM tags in `target["dicom_meta"]`.

```python
dataset = VindrMammoDataset(DATA_ROOT, return_dicom_meta=True)
sample = dataset[0]
print(sample["target"]["dicom_meta"])
```

### `validate_paths`

If `True`, checks all resolved DICOM paths during initialization. This is useful for debugging path problems, but slower at startup.

```python
dataset = VindrMammoDataset(DATA_ROOT, validate_paths=True)
```

Leave it `False` if you want faster initialization.


## Mass target

Because you are mostly interested in masses, every returned target now includes a mass-specific sub-dictionary:

```python
target = sample["target"]
mass_target = target["mass"]

print(mass_target["has_mass"])
print(mass_target["num_masses"])
print(mass_target["boxes"])               # [M, 4], only mass boxes
print(mass_target["labels"])              # [M], all ones for the mass class
print(mass_target["area_percentages"])    # mass box area / image area * 100
print(mass_target["size_bins"])           # tiny, very_small, small, medium, large
print(mass_target["shape_groups"])        # wide, tall, or square_like
print(mass_target["findings"])            # original finding rows for masses only
```

The full `target["boxes"]` still contains all annotated findings. The new `target["mass"]["boxes"]` contains only findings whose `finding_categories` list contains `"Mass"`.

For an image without a mass, the returned tensors are empty:

```python
target["mass"]["boxes"].shape   # torch.Size([0, 4])
target["mass"]["labels"].shape  # torch.Size([0])
```

The project does not invent mass morphology labels. VinDr-Mammo gives mass bounding boxes and finding BI-RADS, but not detailed morphology descriptors such as oval, round, irregular, circumscribed, or spiculated. The statistics utilities therefore define mass shape from bounding-box geometry:

| Shape group | Definition |
|---|---|
| `wide` | box width / box height > 1.33 |
| `tall` | box width / box height < 0.75 |
| `square_like` | otherwise |

The mass area-size bins use these constants, where values are percentages of the full image area:

```python
SIZE_TINY = 0.05
SIZE_VERY_SMALL = 0.10
SIZE_SMALL = 0.50
SIZE_MEDIUM = 1.00
SIZE_LARGE = 1.00
```

This means, for example, `SIZE_TINY = 0.05` means a bounding box area smaller than `0.05%` of the full mammogram image. The `SIZE_LARGE` value is the lower threshold for the large bin.

## Mass annotation animation

The dataset class can open a Matplotlib animation that shows each mass-positive sample with mass boxes drawn on top. By default, `animation_stage="auto"` follows the dataset mode: if `index_level="crop"`, the animation shows the final n x n object-detection crops after inversion, breast crop, mirroring, and square-crop coordinate adjustment. If `index_level="image"`, it shows the preprocessed image frames.

```python
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="image",
    split=None,
    output_size=None,
)

dataset.show_mass_animation(
    output_dir="outputs",
    only_with_mass=True,
    max_images=100,          # use None for every mass-positive image
    interval_ms=700,
    show=True,               # opens a Matplotlib window
    save_gif=False,          # set True to save outputs/mass_annotations_animation.gif
    animation_stage="auto",  # "auto", "preprocessed", or "crop"
)
```

A short alias is also available:

```python
dataset.animate_mass_annotations(max_images=100)
```

The animation reads DICOM images frame by frame instead of loading all full-resolution mammograms into memory. If you set `save_gif=True`, the GIF is saved under `outputs/`. For a quick check, start with `max_images=50` or `max_images=100`. In crop mode, `only_with_mass=True` means crop-positive frames, so empty sliding-window crops are skipped.

In `main.py`, the animation is controlled by these flags:

```python
OPEN_MASS_ANIMATION = False
SAVE_MASS_ANIMATION_GIF = False
MASS_ANIMATION_MAX_IMAGES = None
MASS_ANIMATION_INTERVAL_MS = 700
MASS_ANIMATION_STAGE = "auto"
```

Set `OPEN_MASS_ANIMATION = True` when you want the window to open from the VSCode Run button.

## Statistics and plots

You can compute and save CSV tables plus plots without loading DICOM pixel data:

```python
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="image",
    split=None,
    read_image=False,
)

dataset.print_statistics()
dataset.save_statistics_report("outputs")
```

This saves outputs such as:

```text
outputs/
  statistics_summary.json
  mass_annotations.csv
  image_mass_summary.csv
  breast_birads_counts.csv
  breast_density_counts.csv
  manufacturer_counts.csv
  manufacturer_model_counts.csv
  mass_shape_group_counts.csv
  mass_area_percentage_summary.csv
  split_distribution.png
  breast_birads_distribution.png
  breast_density_distribution.png
  manufacturer_distribution.png
  manufacturer_model_distribution.png
  mass_finding_birads_distribution.png
  mass_bbox_shape_distribution.png
  mass_area_percentage_histogram.png
  mass_aspect_ratio_histogram.png
  mass_bbox_width_vs_height.png
```

Useful methods:

| Method | What it does |
|---|---|
| `dataset.mass_annotations_dataframe()` | Returns one row per mass annotation with box size, aspect ratio, area percentage, BI-RADS, density, view, and split. |
| `dataset.image_mass_dataframe()` | Returns one row per image with `has_mass` and `num_masses`. |
| `dataset.statistics()` | Returns a nested Python dictionary of dataset, vendor, BI-RADS, density, and mass statistics. |
| `dataset.print_statistics()` | Prints the statistics in a readable format. |
| `dataset.save_statistics_tables("outputs")` | Saves CSV and JSON summary tables. |
| `dataset.save_statistics_plots("outputs")` | Saves plots as PNG files. |
| `dataset.save_statistics_report("outputs")` | Saves both tables and plots. |
| `dataset.show_mass_animation(...)` | Opens a Matplotlib animation with mass boxes overlaid on mammograms. |

### Progress bars for long operations

The project now uses `tqdm` for operations that can take a long time, especially when DICOM pixels must be read. You should see progress bars in the VSCode terminal for:

- preprocessed statistics, because every image must be loaded to recompute breast crop, mirror decisions, image size, and mass box percentages,
- deterministic square-crop index construction,
- crop-level statistics,
- preprocessing before/after test search,
- path validation when `validate_paths=True`,
- mass animation GIF saving.

You can turn this off when using the class as a quiet library:

```python
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    show_progress=False,
)
```

In `main.py`, this is controlled by:

```python
SHOW_PROGRESS = True
```

The processed statistics are also cached inside the dataset instance. So if you call `print_statistics(stage="preprocessed")` and then `save_statistics_report(...)`, the project should not reread all DICOMs for every repeated processed-statistics table/plot.

## Returned image-level sample

`dataset[0]` with `index_level="image"` returns:

```python
{
    "index": 0,
    "index_level": "image",
    "image": torch.Tensor,      # [1, H, W], or None if read_image=False
    "target": {
        "image_id": str,
        "study_id": str,
        "series_id": str,
        "dicom_path": str,
        "laterality": "L" or "R",
        "view_position": "CC" or "MLO",
        "split": "training" or "test",
        "height": int,
        "width": int,
        "breast_birads": str,
        "breast_birads_id": int,
        "breast_density": str,
        "breast_density_id": int,
        "boxes": torch.Tensor,          # [N, 4], all finding boxes
        "boxes_original": torch.Tensor, # [N, 4], original CSV coordinates
        "mass": {
            "has_mass": bool,
            "num_masses": int,
            "boxes": torch.Tensor,                  # [M, 4], mass boxes only
            "boxes_original": torch.Tensor,         # [M, 4], original CSV coordinates
            "labels": torch.Tensor,                 # [M], class id 1 for mass
            "area_fractions": torch.Tensor,         # [M], mass box area / image area
            "area_percentages": torch.Tensor,       # [M], area_fractions * 100
            "area_fractions_original": torch.Tensor,
            "area_percentages_original": torch.Tensor,
            "size_bins": list[str],
            "shape_groups": list[str],
            "finding_birads": list,
            "finding_birads_ids": list,
            "findings": list[dict],                 # mass finding rows only
        },
        "has_mass": bool,
        "num_masses": int,
        "findings": list[dict],
        "finding_categories": list[list[str]],
        "finding_category_ids": list[list[int]],
        "num_findings": int,
        "breast_annotation": dict,
        "metadata": list[dict],
        "dicom_meta": dict,
        "original_shape": tuple[int, int],
        "output_shape": tuple[int, int],
    },
}
```

## Returned exam-level sample

`dataset[0]` with `index_level="exam"` returns:

```python
{
    "index": 0,
    "index_level": "exam",
    "study_id": str,
    "images": list[torch.Tensor],
    "targets": list[dict],
    "num_images": int,
}
```

The image and target lists are ordered by laterality and view when possible:

1. left CC
2. left MLO
3. right CC
4. right MLO

If a study has missing or unusual view metadata, the code still returns all available rows and sorts unknown values last.

## DataLoader behavior

Use the provided collate function:

```python
from torch.utils.data import DataLoader
from vindr_mammo import vindr_mammo_collate

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=vindr_mammo_collate,
)
```

The collate function does not resize, pad, or stack images.

For image mode, a batch looks like:

```python
{
    "images": [image_0, image_1, ...],
    "targets": [target_0, target_1, ...],
    "indices": [0, 1, ...],
}
```

For exam mode, a batch looks like:

```python
{
    "exams": [exam_0, exam_1, ...],
    "study_ids": [study_id_0, study_id_1, ...],
    "indices": [0, 1, ...],
}
```

## Important notes

- Bounding boxes are in `[xmin, ymin, xmax, ymax]` format.
- `target["boxes"]` contains all finding boxes, while `target["mass"]["boxes"]` contains mass boxes only.
- `target["mass"]["labels"]` is a one-class detection label tensor where every mass box has class id `1`.
- By default, `output_size=None`, so images are not resized.
- The default collate function keeps images as lists, so variable-size mammograms can pass through a DataLoader safely.
- `read_image=False` is useful for checking annotations before loading pixels.
- `use_voi_lut=False` by default avoids applying display windowing during training data loading.
- `main.py` is intentionally simple so that you can run and debug it with the VSCode Run button.

## Optional preprocessing

The dataset now accepts a compact `preprocess_options` dictionary. These steps are optional, but the default `main.py` enables them because they are useful for mass-focused detection experiments.

```python
PREPROCESS_OPTIONS = {
    "invert_to_black_background": True,
    "crop_breast": True,
    "mirror_right_to_left": True,
    "crop_padding": 20,
    "crop_threshold": None,
}

Dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    preprocess_options=PREPROCESS_OPTIONS,
)
```

### `invert_to_black_background`

When this is `True`, DICOM images with `PhotometricInterpretation == "MONOCHROME1"` are inverted during reading, before normalization. This makes the returned tensor follow the convention you usually want for visualization and model input: black background and bright breast tissue. Images that are already `MONOCHROME2` are left unchanged.

Test method:

```python
dataset.show_inversion_test(output_dir="outputs", show=True, save=True)
```

### `crop_breast`

When this is `True`, the image is cropped to the largest foreground component, ignoring `L-CC`, `R-MLO`, and similar metadata. In other words, the crop is image-content based, not label based. The method tries to remove pure black background while keeping the breast. Mass boxes are shifted and clipped automatically.

Useful options:

```python
"crop_padding": 20,       # extra pixels around the detected breast crop
"crop_threshold": None,  # None means automatic threshold
```

Test method:

```python
dataset.show_crop_test(output_dir="outputs", show=True, save=True)
```

### `mirror_right_to_left`

When this is `True`, the code estimates whether the foreground breast region is mostly on the right side of the image. If so, it flips the image horizontally, so all breasts enter the image from the left side. This is also image-content based and does not depend on `L-CC`, `R-CC`, `L-MLO`, or `R-MLO`. Mass boxes are mirrored automatically.

Test method:

```python
dataset.show_mirror_test(output_dir="outputs", show=True, save=True)
```

### All preprocessing steps together

```python
dataset.show_preprocessing_test(step="all", output_dir="outputs", show=True, save=True)
```

After preprocessing, the returned training boxes are in `target["mass"]["boxes"]`. The original CSV boxes are still preserved in `target["mass"]["boxes_original"]`.

`main.py` now also opens and saves the mass animation by default:

```python
OPEN_MASS_ANIMATION = True
SAVE_MASS_ANIMATION_GIF = True
MASS_ANIMATION_MAX_IMAGES = 100
MASS_ANIMATION_INTERVAL_MS = 700
```

## Statistics before and after preprocessing

`dataset.save_statistics_report("outputs")` now saves the statistics in separate stages:

```text
outputs/
  raw/
    mass_annotations.csv
    image_mass_summary.csv
    ...
  preprocessed/
    mass_annotations.csv
    image_mass_summary.csv
    ...
  crops/
    crop_size_fit_statistics_preprocessed.csv
    crop_size_fit_histogram_preprocessed.png
    crop_dataset_statistics.csv   # only when crop sampling is enabled
```

The `raw/` folder uses the original CSV coordinates. It does not need to load DICOM pixels. The `preprocessed/` folder reads the DICOMs and recomputes image sizes, mass box areas, area percentages, and shape groups after inversion, breast cropping, and mirroring. This matters because breast cropping removes background and therefore increases the mass area percentage relative to the remaining image.

Useful calls:

```python
dataset.print_statistics(stage="raw")
dataset.print_statistics(stage="preprocessed")
dataset.save_statistics_report("outputs")
```

For a quick test on a small subset:

```python
dataset.print_statistics(stage="preprocessed", max_images=200)
dataset.save_statistics_report("outputs", max_processed_images=200)
```

## Changed-image preprocessing tests

The before/after test methods now automatically search for images where the selected preprocessing step actually changes the image. For example, `show_inversion_test()` tries to find a `MONOCHROME1` image; `show_crop_test()` tries to find an image where the breast crop is not the full image; and `show_mirror_test()` tries to find an image where the breast is actually mirrored.

```python
dataset.show_inversion_test(output_dir="outputs", show=True, save=True)
dataset.show_crop_test(output_dir="outputs", show=True, save=True)
dataset.show_mirror_test(output_dir="outputs", show=True, save=True)
dataset.show_preprocessing_test(step="all", output_dir="outputs", show=True, save=True)
```

You can still force a specific index:

```python
dataset.show_crop_test(index=123, output_dir="outputs")
```

## Final n x n crop stage for object detection

The dataset now supports a final square crop stage after inversion, breast crop, and mirroring. This is intended for high-resolution object detection training, where full mammograms are too large and mass-positive crops are rare.

In `main.py`, the crop options are:

```python
USE_SQUARE_CROP_DATASET = False
INDEX_LEVEL = "crop" if USE_SQUARE_CROP_DATASET else "image"

CROP_OPTIONS = {
    "enabled": USE_SQUARE_CROP_DATASET,
    "mode": "random",                  # "random" or "deterministic"
    "crop_size": 1024,                  # n for n x n crops
    "stride": 768,                      # deterministic sliding-window stride
    "random_crops_per_image": 1,
    "positive_fraction": 0.80,          # 80% mass-positive, 20% clean when possible
    "center_on_mass": True,
    "center_shift_fraction": 0.25,
    "allow_partial_annotations": False,
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
```

### Random crops

Random mode samples one or more n x n crops per image. If an image has mass annotations, `positive_fraction` controls how often the crop is sampled around a mass. With `positive_fraction=0.80`, the sampler tries to create about 80% mass-positive crops and 20% clean crops. If a clean crop cannot be found after `max_random_tries`, it falls back to the last sampled window.

```python
crop_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="crop",
    preprocess_options=PREPROCESS_OPTIONS,
    crop_options={**CROP_OPTIONS, "enabled": True, "mode": "random"},
)

sample = crop_dataset[0]
image = sample["image"]                 # [1, n, n]
target = sample["target"]
print(target["mass"]["boxes"])         # crop-coordinate boxes
print(target["square_crop"]["window_xyxy"])
```

### Deterministic sliding-window crops

Deterministic mode slides an n x n window over each preprocessed image using a fixed stride. It creates one dataset item per crop when `index_level="crop"`.

```python
crop_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="crop",
    preprocess_options=PREPROCESS_OPTIONS,
    crop_options={
        **CROP_OPTIONS,
        "enabled": True,
        "mode": "deterministic",
        "crop_size": 1024,
        "stride": 768,
        "deterministic_include_empty": True,
    },
)
```

If you only want crops with at least one mass box, use:

```python
"deterministic_include_empty": False
```

### Partial annotation behavior

If `allow_partial_annotations=False`, a mass box is kept only when it is fully inside the crop. This is the safest option when you do not want truncated boxes in your detection labels.

If `allow_partial_annotations=True`, boxes are clipped to the crop window and kept when at least `min_box_visibility` of the original box remains visible.

```python
"allow_partial_annotations": True,
"min_box_visibility": 0.30,
```

### Crop tests and crop-size recommendations

To visualize one selected crop and the final crop-coordinate boxes:

```python
dataset.show_square_crop_test(output_dir="outputs", mode="random", show=True, save=True)
dataset.show_square_crop_test(output_dir="outputs", mode="deterministic", show=True, save=True)
```

To estimate what square crop size `n` is needed to contain the mass annotations:

```python
fit_table = dataset.crop_size_fit_statistics(crop_size=1024, stage="preprocessed")
print(fit_table)

dataset.save_crop_size_report("outputs/crops", crop_size=1024, stage="preprocessed")
```

The table reports:

- `n_for_90_percent`: crop size needed so 90% of mass boxes fit.
- `n_for_95_percent`: crop size needed so 95% fit.
- `n_for_99_percent`: crop size needed so 99% fit.
- `n_for_100_percent`: crop size needed so all fit.
- `current_n_fits_percent`: what percentage of annotations fit your current `crop_size`.

It reports this for two cases:

1. `single_mass_box`, each mass box independently.
2. `all_mass_boxes_in_same_image`, one square crop that must contain all mass boxes from the same image.
