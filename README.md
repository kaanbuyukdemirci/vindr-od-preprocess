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

This project is now installable as a normal Python package. For development, use editable mode from the project root:

```bash
pip install -e .
```

This reads `pyproject.toml`, installs the package from `src/vindr_mammo`, and creates the console commands:

```bash
vindr-mammo-export --config config/export_config.yaml
vindr-mammo-visualize --config config/export_config.yaml
vindr-mammo-gui --config config/export_config.yaml
```

You can still install from the requirements file if you only want the dependencies:

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

## Library and command-line usage

After installing with `pip install -e .`, you can import the package from any Python script:

```python
from vindr_mammo import VindrMammoDataset, export_from_config, load_export_config

cfg = load_export_config("config/export_config.yaml")
result = export_from_config(cfg)
```

You can also run the two console commands created by `pyproject.toml`:

```bash
vindr-mammo-export --config config/export_config.yaml
vindr-mammo-visualize --config config/export_config.yaml
```

`main.py` and `visualize_export.py` are kept as simple VSCode-friendly wrappers around those same command-line entry points.

## Dash preprocessing GUI

The interactive preprocessing inspector now runs as a Dash app:

```bash
vindr-mammo-gui --config config/export_config.yaml
```

It organizes the workflow into Preview, Preprocess, Crops, Pipeline, Export, Saved Data, Manifests, and Guide tabs. Each important parameter has an in-app `?` explanation with practical examples. The legacy Streamlit inspector is still available for comparison:

```bash
vindr-mammo-streamlit-gui --config config/export_config.yaml
```

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

The dataset now accepts a compact `preprocess_options` dictionary. These steps are optional. The current default dataset export keeps global breast cropping disabled so full-image deterministic crop experiments are not silently altered. You can still enable it in the config or GUI when needed.

```python
PREPROCESS_OPTIONS = {
    "invert_to_black_background": True,
    "crop_breast": False,
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

## Exporting Ultralytics, MMDetection, and baseline datasets

The newest version adds a YAML-driven export pipeline. Edit:

```text
config/export_config.yaml
```

Then run:

```text
main.py
```

from the VSCode Run button.

By default, the exporter reads VinDr-Mammo from:

```yaml
paths:
  data_root: "G:/vindr"
```

and saves all processed datasets under:

```yaml
paths:
  output_root: "G:/preprocessed-vindr"
```

The exporter creates:

```text
G:/preprocessed-vindr/
  square_crops/          # n x n object-detection crops
  baseline_uncropped/    # preprocessing only, no final n x n crop
```

For `square_crops`, the policy is:

- train: random crops
- val: deterministic sliding crops
- test: deterministic sliding crops

The default crop settings are:

```yaml
square_crops:
  crop_size: 1024
  stride: 512
  random_crops_per_annotation: 1
  positive_fraction: 0.80
```

Each dataset variant contains both annotation formats:

```text
vindr_mass.yaml                         # recommended portable Ultralytics YAML
ultralytics/vindr_mass.yaml             # compatibility copy, also portable
mmdetection/annotations/instances_train.json
mmdetection/annotations/instances_val.json
mmdetection/annotations/instances_test.json
```

The Ultralytics YAML files are now written without an absolute `path:` field. This avoids machine-specific paths such as `G:/...` or `/mnt/t9/...`. The recommended file is directly inside the dataset root, for example:

```text
G:/preprocessed-vindr/square_crops/vindr_mass.yaml
/mnt/t9/preprocessed-vindr/square_crops/vindr_mass.yaml
```

Its content is:

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: mass
```

The compatibility copy in `ultralytics/vindr_mass.yaml` uses `../images/train`, `../images/val`, and `../images/test` because it lives one folder below the dataset root. Do not use `path: .` in that file.

Images are saved once and shared by both formats to avoid wasting disk space. For a simple explanation of every YAML field and the saved folder structure, read:

```text
docs/EXPORT_FORMATS_AND_YAML.md
```

## Export update: metadata, histogram equalization, RGB images, and 16-bit preserved PNGs

The export pipeline now saves model-ready RGB PNGs for Ultralytics/MMDetection, plus optional preserved 16-bit grayscale PNGs for inspection and data integrity.

Recommended default:

```yaml
image:
  normalize: "none"

image_export:
  rgb_scheme: "intensity_equalized_gradient"
  intensity_equalized_gradient:
    intensity_window: [1.0, 99.0]
    gradient_source: "normal"
    gradient_window: [1.0, 99.0]
    gradient_ksize: 3

histogram_equalization:
  enabled: true
  apply_to: "third_channel"

preserved_16bit:
  save: true
  percentile_range: [0.1, 99.9]
```

The default RGB export uses three complementary channels: normal intensity, histogram-equalized intensity, and Sobel gradient magnitude. This gives the detector contrast and edge information while the separate `preserved_16bit/` output remains the true high-bit-depth preservation path.

The exported folder contains:

```text
G:/preprocessed-vindr/
  square_crops/
    images/train, val, test                 # 8-bit RGB PNGs for YOLO/MMDetection
    preserved_16bit/train, val, test        # optional 16-bit grayscale PNGs
    labels/train, val, test                 # Ultralytics labels
    vindr_mass.yaml                         # recommended portable YOLO YAML
    ultralytics/vindr_mass.yaml             # compatibility portable YOLO YAML
    mmdetection/annotations/*.json          # COCO-style annotations
    metadata/samples_metadata.jsonl         # full per-sample metadata
    metadata/samples_metadata_flat.csv      # quick metadata table

  baseline_uncropped/
    images/train, val, test
    preserved_16bit/train, val, test
    labels/train, val, test
    vindr_mass.yaml
    ultralytics/vindr_mass.yaml
    mmdetection/annotations/*.json
    metadata/samples_metadata.jsonl
    metadata/samples_metadata_flat.csv

  metadata/source_csv/                      # full copied source CSVs
    breast-level_annotations.csv
    finding_annotations.csv
    metadata.csv
```

For training, use the RGB PNGs in `images/<split>`. The 16-bit files are preserved copies for debugging and analysis, not the default training input.


## Important note about `positive_fraction`

`positive_fraction: 0.80` is meant for the **training square-crop export**, not for the whole VinDr-Mammo dataset, not for the uncropped baseline, and not for validation/test deterministic sliding crops.

With `balance_train_positive_fraction_globally: true`, the exporter tries to keep about 80% of training square crops mass-positive by creating positive random crops around mass annotations and adding only the needed number of clean crops. It does not automatically add one negative crop from every image without a mass, because that can make the final positive percentage much lower than 80%.

Check the actual achieved percentage after export here:

```text
G:/preprocessed-vindr/square_crops/stats/summary.csv
```

The column to check is `positive_image_percent` for the `train` split.


## How to confirm a long export finished

After a successful export, check these final marker files:

```text
G:/preprocessed-vindr/EXPORT_DONE.txt
G:/preprocessed-vindr/manifest.json
```

`EXPORT_DONE.txt` is a short human-readable report. `manifest.json` contains the same completion status plus per-stage timings, output counts, expected-file checks, and a copy of the configuration used for the run. These files are written only at the end of the export.

## Fast visualizations from an existing export

If the export already finished, do **not** rerun `main.py` just to make plots. Run:

```bash
python visualize_export.py
```

or open `visualize_export.py` in VSCode and press the Run button.

This reads only the already exported files under `paths.output_root`, mainly:

```text
square_crops/stats/summary.csv
square_crops/stats/samples.csv
baseline_uncropped/stats/summary.csv
baseline_uncropped/stats/samples.csv
manifest.json, if present
```

It does **not** read DICOMs and does **not** regenerate crops, so it should be much faster than the full export.

The plots are saved to:

```text
G:/preprocessed-vindr/visualizations/
```

The most useful output is:

```text
G:/preprocessed-vindr/visualizations/index.html
```

Open that file in a browser to see the generated plots. The folder also contains PNG files and combined CSV copies.

Useful plots include:

- number of exported images by split,
- mass-positive image percentage by split,
- number of mass boxes by split,
- mass area percentage histograms,
- image-size scatter plots,
- crop-mode distribution,
- view/laterality distribution,
- RGB scheme and histogram-equalization summaries,
- export stage duration plot when `manifest.json` exists.

The visualization settings are controlled from `config/export_config.yaml`:

```yaml
visualizations:
  output_dir: "G:/preprocessed-vindr/visualizations"
  include_square_crops: true
  include_baseline_uncropped: true
  write_html_report: true
  max_rows_per_samples_csv: null
```

Keep `max_rows_per_samples_csv: null` for exact plots. Set it to a number such as `5000` only if you want a very quick approximate preview.


## v3 dataset default: deterministic positive-only training crops

The default `config/export_config.yaml` now creates a new dataset under:

```text
/mnt/t9/preprocessed-vindr-v3
```

This v3 dataset is designed for the experiment where the training split is generated deterministically, but empty train crops are excluded. In other words, the exporter first slides a `1024 x 1024` window with stride `512`, then keeps a train crop only if the final crop contains at least one mass bounding box according to `crop_annotation_policy`. Validation and test still use deterministic sliding windows with empty crops included, so they remain realistic evaluation sets.

Relevant config section:

```yaml
square_crops:
  crop_size: 1024
  stride: 512

  train_crop_mode: "deterministic"
  val_crop_mode: "deterministic"
  test_crop_mode: "deterministic"

  deterministic_include_empty: true
  train_deterministic_include_empty: false
  val_deterministic_include_empty: true
  test_deterministic_include_empty: true

crop_annotation_policy:
  allow_partial_annotations: false
  reject_partial_windows: true
```

With `allow_partial_annotations: false`, a crop only counts as positive if the complete mass box is inside the crop. This avoids training on crops that cut through a lesion. If you later want to include partially visible masses, set `allow_partial_annotations: true` and choose `min_box_visibility`.

The portable Ultralytics YAML is written at:

```text
/mnt/t9/preprocessed-vindr-v3/square_crops/vindr_mass.yaml
```

Use it with:

```bash
yolo detect train model=yolo11n.pt data=/mnt/t9/preprocessed-vindr-v3/square_crops/vindr_mass.yaml imgsz=1024
```


## Interactive preprocessing inspector GUI

This version includes a local Dash GUI for inspecting mammograms, square crops, mass boxes, vendors, and experimental RGB preprocessing pipelines before exporting a new dataset. Run it with:

```bash
pip install -e .
vindr-mammo-gui --config config/export_config.yaml
```

or, from the repository root:

```bash
python inspect_preprocessing_app.py
```

### Cross-section study presets

The Dash inspector places **Study preset** above all tabs because these presets intentionally update DICOM loading, fixed preprocessing, crop geometry, sampling, RGB export, and save settings together. The **Bulatović et al. — YOLOv8 patched inference (VinDr-Mammo)** preset applies the data recipe reported in *Refining YOLOv8 for Full Field Digital Mammograms*:

- MONOCHROME1 correction to bright tissue on dark background;
- DICOM VOI LUT/windowing for VinDr-Mammo and 8-bit replicated-grayscale PNG export;
- artifact/background masking without an unreported breast bounding-box crop or left/right mirroring;
- deterministic 640 × 640 patches with 512 px stride (20% overlap);
- all positive training patches plus a seeded 20% sample of eligible negative candidates;
- all eligible validation/test patches; and
- a valid-Mass-positive source cohort only;
- official VinDr test preservation with a deterministic, BI-RADS-stratified,
  study-level train/validation split matching the published 398/758,
  71/136, and 115/219 study/image counts; and
- a versioned output folder named `preprocessed-vindr-paper22-v2` under the
  currently configured output parent.

The preset replaces inherited dataset-affecting settings rather than retaining
stale vendor, split, foreground, or crop overrides. It also blocks completion
unless the published cohort counts, source-study isolation, 237 official-test
mass boxes, 100% source-annotation representation, exact rounded 20% training
negative retention, and complete validation/test inference policy pass. Source
CSV hashes, the Git revision, the resolved settings, and the disclosed versus
assumed choices are recorded in the manifest.

The paper does not publish its train/validation IDs or seed, foreground-mask
algorithm, partial-box rule, or the granularity/rounding of negative sampling.
The preset therefore labels its deterministic seed, 5% foreground rule, 30%
box-visibility rule, global sampling, and edge-aligned final grid starts as
replication assumptions. It does not claim that the count-matched split is
author-identical. Model architecture, training augmentation, source-coordinate
inference, and Maximum Box Fusion remain in the model repository.

Run the same hermetic preset without the GUI with:

```bash
vindr-mammo-export --config config/export_config.yaml --preset paper22

# Or without installing the console script:
python main.py --config config/export_config.yaml --preset paper22
```

The GUI can filter by train/val/test, positive images only or all images, vendor/device, crop positivity threshold, and crop index. It displays the full grayscale image, the selected grayscale crop, and the processed RGB crop together with mass boxes, statistics, metadata, optional per-channel panels with mass boxes, and compare-mode statistical similarity metrics.

For large DICOMs, change multiple parameters quickly, then click **Render / refresh** once to run the expensive DICOM read, breast crop, crop selection, RGB preprocessing, and rendering step.

Detailed docs:

- GUI workflow: `docs/GUI_PREPROCESSING_INSPECTOR.md`
- Every GUI parameter: `docs/GUI_PARAMETER_REFERENCE.md`

## v26 GUI and crop-filter update

The preprocessing inspector GUI now includes:

- visible-channel controls for R/G/B debugging,
- optional individual R/G/B channel panels,
- deterministic sliding crop preview,
- stochastic random crop preview,
- a foreground-ratio crop filter that can reject square crops with too little breast foreground.

The same foreground-ratio filter is available in the exporter under `square_crops`:

```yaml
preprocess:
  crop_breast: false

square_crops:
  deterministic_require_foreground: true
  deterministic_min_foreground_fraction: 0.05
  deterministic_foreground_threshold: null
```

Use the GUI first to tune `deterministic_min_foreground_fraction`, then export a dataset with the same settings.

## v27 GUI comparison update

The preprocessing inspector now overlays mass annotations on the individual processed R/G/B channel panels when annotation display is enabled. Compare mode also includes a statistics-comparison section before the image panels. It compares selected samples using summary feature differences, Jensen-Shannon distance on compact intensity summaries, and an approximate 1-D Wasserstein distance. The pixel-intensity distribution plot was removed from the metadata panel.

### GUI preprocessing YAML export

The preprocessing inspector GUI now includes an **Export current preprocessing YAML** panel. It downloads the current fixed preprocessing settings, crop preview settings, channel visibility settings, and R/G/B preprocessing pipelines. The exported YAML includes an `export_config_patch` block. The exporter supports `image_export.rgb_scheme: custom_channel_pipeline` for using GUI-designed R/G/B pipelines during dataset export.


## Version 0.30 contralateral-channel preprocessing

The default `config/export_config.yaml` now uses `image_export.rgb_scheme: custom_channel_pipeline` with a contralateral same-view source for the B channel. This lets one RGB channel contain the same crop coordinates from the opposite breast in the same study and view, while R and G use the current crop with normal and bright-region preprocessing.

Run the GUI to inspect or edit it:

```bash
vindr-mammo-gui --config config/export_config.yaml
```

Then export the dataset:

```bash
vindr-mammo-export --config config/export_config.yaml
```

### GUI-driven export controls

The preprocessing GUI now includes an **Export dataset from GUI** panel. It supports selected-vendor export, split-specific mass-window-only export for train/val/test, output folder/name controls, and a progress bar while the dataset is being written.

Use it when you want the exported dataset to exactly match the current GUI preprocessing settings without manually editing `config/export_config.yaml`.

## Visualizing cropped dataset and COCO box-size bins

After an export finishes, regenerate the visualization report with:

```bash
vindr-mammo-visualize --config config/export_config.yaml
```

or run `visualize_export.py` from VSCode.

Open:

```text
<output_root>/visualizations/index.html
```

The report now includes COCO-style box-size statistics from the exported COCO/MMDetection JSON files. It writes `coco_box_size_stats.csv`, `coco_box_annotations.csv`, and plots for small, medium, and large object bins.


### GUI visualization of an exported dataset

The Dash GUI includes a **Dataset visualizations** mode for viewing visualization files under an exported dataset root such as `/mnt/t9/preprocessed-vindr-v3/visualizations`.

### v41 GUI export additions

The GUI export panel now reports elapsed time and estimated remaining time during dataset export. It also supports split-specific deterministic crop selection modes: `mass_only`, `all`, `positive_ratio`, and `finding_images_all_windows`. In `positive_ratio` mode, all mass-positive deterministic windows are kept and non-mass windows are sampled to approach the requested positive crop ratio for train, validation, or test independently. In `finding_images_all_windows` mode, source images with no mass/finding are skipped, but every deterministic crop from source images with at least one mass/finding is kept, including crop windows that do not themselves contain the mass.


### v43 ETA fix

The GUI export progress panel now estimates remaining time from the current active stage progress instead of the coarse overall export fraction. This is especially important because square-crop export dominates runtime while earlier setup stages are short.

## v44 notes: manifest loading and defaults

- Manifest/config loading now applies the full RGB preprocessing pipeline parameters, not only the operation names.
  For example, loaded percentile windows such as `[50, 100]` or `[75, 100]` are now reflected in the GUI sliders.
- High-level `image_export.rgb_scheme` values are converted into editable GUI channel pipelines when loaded.
  For example, `intensity_equalized_gradient` appears as R intensity, G equalized intensity, and B Sobel-gradient channel steps.
- The default `config/export_config.yaml` was reset to the user-provided configuration with `rgb_scheme: intensity_equalized_gradient`, breast crop enabled, train random crops, and val/test deterministic crops.


## v48 note, manifest loading and crop-control refresh

Manifest/config loading treats the active YAML as the source of truth for export-related settings. Load a previous resolved config with the Config YAML field to rebuild the Dash controls from that snapshot.

A new `Loaded crop settings check` sidebar expander shows the crop-size, stride, split crop modes, deterministic selection modes, target positive ratios, foreground filtering options, random-crop options, and crop annotation policy currently active from the loaded config.

### New crop mode: bbox-safe breast-biased random crops

A new crop proposal mode is available for train, validation, and test:

```yaml
square_crops:
  train_crop_mode: bbox_safe_random
  val_crop_mode: bbox_safe_random
  test_crop_mode: bbox_safe_random
```

This mode samples random crops around annotations, but rejects crops where any
visible mass annotation is clipped or lands too close to the crop boundary. It
then prefers candidates that contain more breast foreground, with optional
left/chest-wall and x-projection peak bias. The setting
`bbox_safe_boundary_margin_fraction` controls the forbidden boundary zone.


### v51 GUI clarity and failed-crop preview fix

Changes added after v50:

- The GUI no longer hides an image when all crop candidates are rejected by the current crop filter. It shows the selected image and a failed crop preview with a clear reason, such as `preview_filter_requires_visible_mass`, `foreground_fraction_below_threshold`, or a bbox-safe failure reason.
- The sidebar crop mode and positive probability controls are now explicitly marked `PREVIEW ONLY`. They affect browsing/inspection, not the exported train/val/test crop generator.
- The `Export dataset from GUI` panel is now the single place to set train/val/test crop modes and the mass-vs-empty export balance.
- Default export balance is now 0.50, meaning approximately 50% mass-positive crops and 50% empty crops. The exact achieved ratio is still written to `summary.csv`.
- Random, bbox-safe random, and deterministic crop modes now all read split-specific export target ratios from the GUI/YAML.

Example for a 50/50 export:

```yaml
square_crops:
  train_positive_fraction: 0.50
  val_positive_fraction: 0.50
  test_positive_fraction: 0.50
  train_deterministic_selection_mode: positive_ratio
  val_deterministic_selection_mode: positive_ratio
  test_deterministic_selection_mode: positive_ratio
```

## v50 bbox-safe hard-boundary fix

`bbox_safe_random` now treats the boundary rule as a hard export constraint. Set:

```yaml
square_crops:
  bbox_safe_skip_unsafe_fallbacks: true
```

With this enabled, if no crop can keep the visible annotations fully inside the safe inner region, the exporter skips that crop instead of writing a fallback. The exporter also performs a final validation on the actual crop-coordinate boxes before saving labels. This prevents annotations from touching or entering the forbidden boundary band.

## v52 notes: contralateral source nipple-y alignment

This version adds vertical alignment for the custom RGB source `contralateral_same_view_crop`, shown in the GUI as `opposite breast, same view, same xyxy crop`.

When enabled, the exporter and GUI estimate the nipple y location from the breast foreground boundary in the current image and the opposite-breast image. The opposite full preprocessed image is shifted up or down so the two estimated nipple y locations match. After that shift, the same `xyxy` crop window is extracted from the opposite breast.

The default alignment config is:

```yaml
image_export:
  contralateral_source_alignment:
    enabled: true
    method: nipple_y
    threshold: null
    tip_side: auto
    tip_tolerance_fraction: 0.006
    tip_tolerance_px: null
    smooth_rows: 31
    max_shift_fraction: 0.20
    pad_value: 0.0
```

`method: projection_y` is included as an explicit placeholder. It currently leaves the opposite image unchanged and records `projection_intensity_alignment_placeholder_not_implemented` in the debug metadata.

## v53 notes: profile-based contralateral source alignment

This version upgrades `source: contralateral_same_view_crop` alignment from a single nipple-y rule to multiple implemented vertical alignment methods.

The new default is:

```yaml
image_export:
  contralateral_source_alignment:
    enabled: true
    method: hybrid_profile_y
    fallback_method: nipple_y
    projection_smooth_rows: 51
    boundary_smooth_rows: 31
    max_shift_fraction: 0.20
    min_profile_overlap_fraction: 0.60
    min_profile_score: 0.05
    profile_score_margin: 0.03
    max_profile_nipple_disagreement_fraction: 0.05
```

Available methods:

- `hybrid_profile_y`: default. Computes row-projection, boundary-profile, nipple-y, and centroid-y candidate shifts. It usually chooses `row_projection_y`, because that uses the full vertical foreground distribution. If the profile match is weak, it falls back to `nipple_y` or `mask_centroid_y`.
- `row_projection_y`: builds a 1D profile where each row value is the number of breast-foreground pixels in that row. It then searches for the vertical shift with the best normalized correlation.
- `boundary_profile_y`: builds a 1D profile from the outer breast boundary row by row, then aligns that shape profile.
- `nipple_y`: previous v52 method. It estimates the nipple/tip row from the foreground boundary and aligns those rows.
- `mask_centroid_y`: aligns the vertical centroid of the breast foreground mask.
- `intensity_projection_y`: row-sum intensity profile matching. This is now implemented, but it is not the default because intensity differences across vendors/windowing can make it less stable than foreground-mask profiles.

The exporter and GUI write debug metadata such as:

```text
contralateral_alignment_method
contralateral_alignment_selected_method
contralateral_alignment_selection_reason
contralateral_alignment_shift_y
contralateral_alignment_candidates
contralateral_alignment_warning
```

For example, with `method: hybrid_profile_y`, the debug fields may show that `row_projection_y` was selected, the shift was `-58` pixels, and nipple-y estimated `-65` pixels. If the profile and nipple estimates disagree too much, the profile shift is still used but a warning is recorded.

## v54 speed defaults and simple profiler

The default contralateral alignment method is now `nipple_y`, with `mask_centroid_y` as fallback. This is much faster than `hybrid_profile_y` for full dataset export. The profile methods are still available for inspection or smaller debug exports.

The exporter also includes a lightweight start/stop profiler. It records coarse timing buckets such as preprocessing, crop planning, contralateral source crop generation, image saving, and metadata writing. In the GUI export panel, enable **Show simple timing breakdown during export** to see the live table. The table is updated every `runtime.simple_profiler_emit_every` progress updates to avoid slowing down the export.


## v56 random/bbox-safe global balance note

For random and bbox-safe random exports, the default is now one positive crop per annotation and global selection of negative crops to match the requested target positive ratio. With `positive_fraction: 0.50`, the exporter keeps all positive crop candidates and randomly selects enough clean crops from the global clean-candidate pool, including images with no mass, to make the saved crop set approximately 50% mass-positive and 50% empty.

```yaml
square_crops:
  random_crops_per_annotation: 1
  bbox_safe_crops_per_annotation: 1
  positive_fraction: 0.50
  global_positive_ratio_selection_for_random: true
  global_negative_candidate_crops_per_image_when_balancing: 1
  random_crops_per_negative_image_when_balancing: 1
  bbox_safe_random_crops_per_negative_image_when_balancing: 1
```

The contralateral source path was also made faster: the exporter now estimates and caches a vertical shift, then crops the opposite image from an adjusted window instead of shifting the full mammogram tensor before every crop.


## v56 global random balance and faster contralateral crop

Defaults changed for random and bbox-safe random exports:

```yaml
square_crops:
  random_crops_per_annotation: 1
  bbox_safe_crops_per_annotation: 1
  positive_fraction: 0.50
  global_positive_ratio_selection_for_random: true
  global_negative_candidate_crops_per_image_when_balancing: 1
  random_crops_per_negative_image_when_balancing: 1
  bbox_safe_random_crops_per_negative_image_when_balancing: 1
  bbox_safe_boundary_margin_fraction: 0.02
```

This means the exporter keeps one mass-centered crop candidate per annotation, then globally samples clean crops to reach the requested target mass-positive crop ratio. Clean crops can come from no-mass images as well as clean windows from finding images.

The contralateral source path is faster now. Instead of shifting the whole opposite mammogram tensor and then cropping it, the exporter estimates and caches a vertical shift, then takes the same crop from an adjusted y-window. This preserves the same aligned crop behavior while avoiding a large full-image copy for every pair.

## Saved dataset viewer

The Dash GUI includes a **Saved dataset viewer** mode for checking exported `square_crops` without reloading the original DICOM files. It reads the exported PNG crops, YOLO labels, `stats/samples.csv`, `debug_logs/crop_log.csv`, and `metadata/export_config_resolved.yaml`.

Features:

- Manual scan with Previous, Next, and crop-number slider.
- Automatic playback with a user-selected period in seconds.
- Annotation boxes drawn from saved YOLO label files.
- Source image index, image id, split, positivity, crop window, and file name shown on screen and in the side metadata panel.

Run the GUI as usual:

```bash
vindr-mammo-gui --config config/export_config.yaml
```

Then choose `Saved dataset viewer` from the Mode selector.
