# Dataset API notes

This file documents the main behavior of `VindrMammoDataset` in a compact form.

## Constructor

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
    show_progress=True,
)
```

## Parameter summary

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `data_root` | `str` or `Path` | required | Path to the VinDr-Mammo folder. In your current setup this is `r"G:/vindr"`. |
| `index_level` | `"image"`, `"exam"`, or `"crop"` | `"image"` | Selects one-image, one-study, or one-square-crop samples. |
| `split` | `str`, iterable, or `None` | `None` | Uses the official `split` column. Use `"training"`, `"test"`, or `None`. |
| `read_image` | `bool` | `True` | Reads DICOM pixels in `__getitem__`. If `False`, returns annotations only. |
| `transform` | callable or `None` | `None` | Image-only transform. Receives `[1, H, W]`. |
| `target_transform` | callable or `None` | `None` | Target-only transform. Receives the target dictionary. |
| `joint_transform` | callable or `None` | `None` | Full-sample transform for image and target together. |
| `normalize` | `str` | `"minmax"` | One of `"none"`, `"minmax"`, `"percentile"`, `"zscore"`. |
| `percentile_range` | tuple | `(0.5, 99.5)` | Used only for `normalize="percentile"`. |
| `use_voi_lut` | `bool` | `False` | Applies DICOM VOI LUT or windowing if `True`. |
| `output_size` | tuple or `None` | `None` | Optional `(height, width)` resize. `None` preserves original size. |
| `return_dicom_meta` | `bool` | `False` | Adds selected DICOM tags to `target["dicom_meta"]`. |
| `validate_paths` | `bool` | `False` | Checks all DICOM paths at initialization. |
| `preprocess_options` | dict or `None` | `None` | Optional inversion, breast crop, and mirror settings. |
| `crop_options` | dict or `None` | `None` | Optional final n x n object-detection crop settings. |
| `show_progress` | `bool` | `True` | Shows `tqdm` progress bars for slow loops such as processed statistics, crop index creation, crop stats, and GIF saving. |

## Mass-specific target

Every image target contains both all findings and mass-only findings:

```python
sample = dataset[0]
target = sample["target"]

all_boxes = target["boxes"]
mass_boxes = target["mass"]["boxes"]
mass_labels = target["mass"]["labels"]
mass_area_percentages = target["mass"]["area_percentages"]
mass_size_bins = target["mass"]["size_bins"]
```

`target["mass"]` contains:

| Key | Meaning |
|---|---|
| `has_mass` | `True` if the image has at least one mass annotation. |
| `num_masses` | Number of mass boxes in the image. |
| `boxes` | `[M, 4]` tensor of mass boxes. Scaled if `output_size` is used. |
| `boxes_original` | `[M, 4]` tensor of original CSV mass boxes. |
| `labels` | `[M]` tensor. All values are `1`, meaning the one-class mass label. |
| `area_fractions` | `[M]` tensor, mass box area divided by full image area. |
| `area_percentages` | `[M]` tensor, `area_fractions * 100`. |
| `area_fractions_original` | `[M]` tensor computed from original CSV coordinates. |
| `area_percentages_original` | `[M]` tensor computed from original CSV coordinates. |
| `size_bins` | List of mass area-size bins: `tiny`, `very_small`, `small`, `medium`, `large`. |
| `shape_groups` | List of bounding-box shape groups: `wide`, `tall`, or `square_like`. |
| `finding_birads` | Finding-level BI-RADS strings for mass findings. |
| `finding_birads_ids` | Parsed integer BI-RADS values. |
| `findings` | Original finding rows, filtered to mass rows only. |

## Typical choices

For your current stage, use original sizes and no batching resize:

```python
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="image",
    split=None,
    output_size=None,
)
```

For the official training split:

```python
train_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="image",
    split="training",
    output_size=None,
)
```

For metadata/statistics inspection without DICOM loading:

```python
dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    read_image=False,
)
```

## Mass area-size constants

The area-size bins use percentage of full image area:

```python
SIZE_TINY = 0.05
SIZE_VERY_SMALL = 0.10
SIZE_SMALL = 0.50
SIZE_MEDIUM = 1.00
SIZE_LARGE = 1.00
```

`SIZE_LARGE` is the lower threshold for the large bin. You can access these from the module or from the class.

```python
from vindr_mammo import SIZE_TINY, SIZE_LARGE
print(SIZE_TINY, SIZE_LARGE)
print(VindrMammoDataset.SIZE_TINY)
```

## Statistics and visualization utilities

These methods are available directly on the dataset class:

| Method | Output |
|---|---|
| `mass_annotations_dataframe()` | One row per mass annotation with box width, height, area percentage, aspect ratio, shape group, view, split, BI-RADS, and density. |
| `image_mass_dataframe()` | One row per image with `has_mass` and `num_masses`. |
| `statistics()` | Nested dictionary containing dataset, vendor/model, BI-RADS, density, and mass statistics. |
| `print_statistics()` | Prints a readable summary. |
| `save_statistics_tables("outputs")` | Saves CSV tables and `statistics_summary.json`. |
| `save_statistics_plots("outputs")` | Saves PNG plots. |
| `save_statistics_report("outputs")` | Saves both tables and plots. |
| `show_mass_animation(...)` | Opens a Matplotlib animation of mass-positive samples with boxes overlaid. With `animation_stage="auto"`, crop datasets show final n x n crop frames. |
| `animate_mass_annotations(...)` | Alias for `show_mass_animation(...)`. |

Example:

```python
dataset = VindrMammoDataset(r"G:/vindr", read_image=False)
dataset.print_statistics()
dataset.save_statistics_report("outputs")
```

Animation example:

```python
dataset = VindrMammoDataset(r"G:/vindr", output_size=None)
dataset.show_mass_animation(
    output_dir="outputs",
    only_with_mass=True,
    max_images=100,
    interval_ms=700,
    show=True,
    save_gif=False,
    animation_stage="auto",
)
```

`animation_stage="auto"` follows the current dataset mode. For `index_level="crop"`, the GIF/window shows the final crop image and crop-coordinate boxes. Use `animation_stage="preprocessed"` only when you intentionally want the full pre-square-crop view.

The mass shape distribution is based on bounding-box aspect ratio because the official annotations do not provide detailed mass morphology descriptors.

| Shape group | Definition |
|---|---|
| `wide` | box width / box height > 1.33 |
| `tall` | box width / box height < 0.75 |
| `square_like` | otherwise |

## Output size and bounding boxes

When `output_size=None`, the image tensor keeps its original DICOM size and the box coordinates stay in the original coordinate system.

When `output_size=(height, width)`, the image is resized and `target["boxes"]` and `target["mass"]["boxes"]` are scaled to match the resized image. The original CSV coordinates remain available in `target["boxes_original"]` and `target["mass"]["boxes_original"]`.

## Collate function

`vindr_mammo_collate` intentionally keeps images as lists.

This means a batch can contain images with different heights and widths:

```python
batch = next(iter(loader))
print([image.shape for image in batch["images"]])
```

No padding, stacking, or resizing is done inside the collate function.

## Preprocessing options

`VindrMammoDataset(..., preprocess_options={...})` supports these keys:

| Key | Type | Meaning |
|---|---|---|
| `invert_to_black_background` | `bool` | If `True`, invert `MONOCHROME1` DICOMs before normalization so background is black and tissue is bright. |
| `crop_breast` | `bool` | If `True`, crop to the largest foreground breast component. Boxes are shifted and clipped. |
| `mirror_right_to_left` | `bool` | If `True`, flip images whose breast foreground is mostly on the right side. Boxes are mirrored. |
| `crop_padding` | `int` | Extra padding around the detected breast crop. |
| `crop_threshold` | `float` or `None` | Foreground threshold. Use `None` for automatic thresholding. |
| `min_component_area_fraction` | `float` | Minimum largest-component area fraction used when cleaning the foreground mask. |

Quick visual tests:

```python
dataset.show_inversion_test(output_dir="outputs")
dataset.show_crop_test(output_dir="outputs")
dataset.show_mirror_test(output_dir="outputs")
dataset.show_preprocessing_test(step="all", output_dir="outputs")
```

The methods save side-by-side before/after PNGs and can optionally open a Matplotlib window with `show=True`.

## New constructor arguments

The constructor also accepts:

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `preprocess_options` | dict or `None` | `None` | Optional inversion, breast crop, and mirror settings. Geometry-changing steps update boxes. |
| `crop_options` | dict or `None` | `None` | Optional final n x n square crop stage for object detection. Applied after normal preprocessing. |

`index_level` can now also be `"crop"`. In crop mode, one dataset item is one final square object-detection crop.

## Square crop options

`crop_options` supports:

| Key | Meaning |
|---|---|
| `enabled` | Enables the final square crop stage. Required for `index_level="crop"`. |
| `mode` | `"random"` or `"deterministic"`. |
| `crop_size` | The crop size `n`, giving `[1, n, n]` image tensors. |
| `stride` | Sliding-window stride for deterministic mode. |
| `random_crops_per_image` | Number of random crop samples per source image in crop mode. |
| `positive_fraction` | Fraction of random crops that should contain at least one mass when possible. |
| `center_on_mass` | Positive random crops are sampled around mass centers. |
| `center_shift_fraction` | Random shift around the selected mass center, relative to crop size. |
| `allow_partial_annotations` | If `False`, only fully included mass boxes are kept. If `True`, clipped boxes can be kept. |
| `min_box_visibility` | Minimum visible fraction required when partial boxes are allowed. |
| `reject_partial_windows` | In non-partial mode, reject windows that cut through a mass box. |
| `negative_max_box_visibility` | Maximum allowed visible mass fraction in clean negative crops. |
| `pad_if_needed` | Pad images/crops when the selected crop extends outside the source image. |
| `seed` | Random seed for reproducible random crop sampling. |

Example random crop dataset:

```python
crop_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="crop",
    preprocess_options=PREPROCESS_OPTIONS,
    crop_options={
        "enabled": True,
        "mode": "random",
        "crop_size": 1024,
        "positive_fraction": 0.80,
        "allow_partial_annotations": False,
    },
)
```

Example deterministic sliding-window crop dataset:

```python
crop_dataset = VindrMammoDataset(
    data_root=r"G:/vindr",
    index_level="crop",
    preprocess_options=PREPROCESS_OPTIONS,
    crop_options={
        "enabled": True,
        "mode": "deterministic",
        "crop_size": 1024,
        "stride": 768,
        "deterministic_include_empty": True,
    },
)
```

Returned crop samples have:

```python
sample["image"].shape                 # [1, n, n]
sample["target"]["mass"]["boxes"]     # crop-coordinate boxes
sample["target"]["square_crop"]        # crop metadata, including window_xyxy
```

## Two-stage statistics and crop-size reports

The statistics report now writes separate folders:

```text
outputs/raw/
outputs/preprocessed/
outputs/crops/
```

Important methods:

| Method | Meaning |
|---|---|
| `processed_mass_annotations_dataframe()` | One row per mass after inversion, breast crop, and mirroring. |
| `processed_image_mass_dataframe()` | One row per image after preprocessing. |
| `statistics(stage="raw")` | Raw CSV statistics. |
| `statistics(stage="preprocessed")` | Statistics after preprocessing. Reads DICOM pixels. |
| `crop_size_fit_statistics(crop_size=1024, stage="preprocessed")` | Reports required n so 90%, 95%, 99%, and 100% of mass boxes fit. |
| `save_crop_size_report("outputs/crops")` | Saves crop-size recommendations and a histogram. |
| `show_square_crop_test(...)` | Saves/shows a selected full-image window and the resulting crop. |

Example:

```python
dataset.print_statistics(stage="raw")
dataset.print_statistics(stage="preprocessed")
dataset.save_statistics_report("outputs")
dataset.show_square_crop_test(output_dir="outputs", mode="random")
```

## Progress bars

Long operations use `tqdm` when `show_progress=True`. The main slow operations are:

- `dataset.print_statistics(stage="preprocessed")`
- `dataset.save_statistics_report(...)`
- deterministic `index_level="crop"` initialization
- `dataset.crop_dataset_statistics(...)`
- `dataset.show_inversion_test(...)`, `show_crop_test(...)`, and `show_mirror_test(...)` when they search for changed examples
- `dataset.show_mass_animation(save_gif=True, ...)`

Set `show_progress=False` to disable progress bars. The project also caches processed-statistics dataframes inside the dataset object, so repeated calls reuse the first DICOM pass when possible.

## Export API

The export pipeline is in `src/vindr_mammo/export.py`.

```python
from vindr_mammo import load_export_config, export_from_config

cfg = load_export_config("config/export_config.yaml")
result = export_from_config(cfg)
print(result.output_root)
```

Usually you do not need to call this manually. `main.py` does exactly this so you can run it from the VSCode Run button.

The exporter creates two dataset variants:

1. `square_crops`: final `n x n` crops for object detection. Train uses random crops. Val and test use deterministic sliding crops.
2. `baseline_uncropped`: preprocessing only, no final `n x n` crop.

Each variant contains:

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test` for Ultralytics YOLO
- `ultralytics/vindr_mass.yaml`
- `mmdetection/annotations/instances_train.json`
- `mmdetection/annotations/instances_val.json`
- `mmdetection/annotations/instances_test.json`
- `stats/samples.csv`
- `stats/summary.csv`

Important YAML fields are explained in `docs/EXPORT_FORMATS_AND_YAML.md`.
