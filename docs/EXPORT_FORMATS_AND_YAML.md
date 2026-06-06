# Export formats and YAML configuration

This project can export preprocessed VinDr-Mammo mass detection data for two common object detection training stacks:

1. **Ultralytics YOLO format**
2. **MMDetection using COCO-format JSON annotations**

The exporter saves the processed image files only once per dataset variant, then writes both annotation formats from the same samples. This avoids repeating the expensive DICOM reading and preprocessing.

## Output folder

The default output root is:

```yaml
paths:
  output_root: "G:/preprocessed-vindr"
```

The exported structure is:

```text
G:/preprocessed-vindr/
  split_assignments.csv
  export_summary.json

  square_crops/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    vindr_mass.yaml                  # recommended portable Ultralytics YAML
    ultralytics/
      vindr_mass.yaml                # compatibility copy, also portable
    mmdetection/
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
      README_mmdetection_paths.txt
    stats/
      samples.csv
      summary.csv

  baseline_uncropped/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    vindr_mass.yaml                  # recommended portable Ultralytics YAML
    ultralytics/
      vindr_mass.yaml                # compatibility copy, also portable
    mmdetection/
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
      README_mmdetection_paths.txt
    stats/
      samples.csv
      summary.csv
```

`baseline_uncropped` means **no final n x n square crop**. It still uses the normal preprocessing steps: inversion, breast crop, and optional left/right mirroring.

## Split behavior

VinDr-Mammo already has an official `split` column with `training` and `test`. The exporter keeps official `test` as test. It then splits official `training` exams into `train` and `val` by `study_id`, so views from the same exam do not leak across train and validation.

```yaml
splits:
  val_fraction_from_training: 0.15
  seed: 123
```

## Square crop behavior

For the square-crop export:

- `train` uses **random crops**.
- `val` uses **deterministic sliding crops**.
- `test` uses **deterministic sliding crops**.

```yaml
square_crops:
  crop_size: 1024
  stride: 512
  random_crops_per_annotation: 1
  random_crops_per_negative_image: 1
  positive_fraction: 0.80
```

The most important parameters are:

| YAML key | Meaning |
|---|---|
| `crop_size` | The square crop size `n`, so each crop is `n x n`. |
| `stride` | Sliding-window stride for validation and test. Lower means more overlap and more crops. |
| `random_crops_per_annotation` | Number of mass-centered random crops to create per mass box in the training split. |
| `random_crops_per_negative_image` | Number of clean random crops for images with no mass annotations. |
| `positive_fraction` | Approximate target positive/negative balance for training crops. For example, `0.80` means about 80 percent positive crops. |
| `center_shift_fraction` | Random shift around the mass center, as a fraction of crop size. With `0.25` and `1024`, the center can shift up to about 256 pixels. |
| `deterministic_include_empty` | Global default for deterministic splits. If true, deterministic crops include clean windows. |
| `<split>_deterministic_include_empty` | Split-specific override, for example `train_deterministic_include_empty: false` keeps only positive deterministic train crops while val/test can still include empty windows. |

## Partial annotation policy

```yaml
crop_annotation_policy:
  allow_partial_annotations: false
  min_box_visibility: 0.30
  reject_partial_windows: true
  negative_max_box_visibility: 0.0
```

If `allow_partial_annotations: false`, a mass box is kept only if the whole box is inside the square crop.

If `allow_partial_annotations: true`, a mass box can be clipped to the crop boundary and kept if at least `min_box_visibility` of the original box remains.

## Ultralytics output

Each image has one `.txt` label file with rows like:

```text
class x_center y_center width height
```

All coordinates are normalized to `[0, 1]`. The only class is:

```yaml
0: mass
```

Train using the generated root-level YAML, for example:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="G:/preprocessed-vindr/square_crops/vindr_mass.yaml", imgsz=1024)
```

For the baseline dataset, use:

```python
model.train(data="G:/preprocessed-vindr/baseline_uncropped/vindr_mass.yaml", imgsz=1024)
```

On Linux, the same YAML works after moving the dataset because it contains only relative paths. Example:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/mnt/t9/preprocessed-vindr/square_crops/vindr_mass.yaml \
  imgsz=1024
```

The root-level `vindr_mass.yaml` is written as:

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: mass
```

The compatibility copy under `ultralytics/vindr_mass.yaml` is written as:

```yaml
train: ../images/train
val: ../images/val
test: ../images/test
names:
  0: mass
```

Both files intentionally omit `path:`. Do not change them to `path: .`, because some Ultralytics versions may resolve that relative to the current working directory instead of the YAML directory.

## MMDetection output

The MMDetection export uses COCO-style JSON files:

```text
mmdetection/annotations/instances_train.json
mmdetection/annotations/instances_val.json
mmdetection/annotations/instances_test.json
```

Category id `1` is `mass`. The helper file `mmdetection/README_mmdetection_paths.txt` contains the exact path snippets to paste into an MMDetection config.

## Statistics

The exporter creates statistics while saving the images, instead of doing another slow pass later:

```text
stats/samples.csv
stats/summary.csv
```

`samples.csv` has one row per exported image or crop. It includes source image id, source study id, split, crop window, image size, number of mass boxes, and mass area percentages.

`summary.csv` gives split-level counts.

## Version 12: metadata, RGB encoding, histogram equalization, and 16-bit preservation

The exporter now writes both model-ready RGB PNGs and optional preserved 16-bit grayscale PNGs.

### Default RGB scheme: `intensity_equalized_gradient`

Mammograms are originally high-bit-depth grayscale images. Most YOLO/MMDetection pipelines expect normal 8-bit, 3-channel images. No 8-bit RGB scheme can truly keep the full DICOM pixel depth, so the exporter still saves a separate preserved 16-bit PNG. The RGB image is only the model input representation.

The current default is:

```yaml
image_export:
  rgb_scheme: "intensity_equalized_gradient"
  intensity_equalized_gradient:
    intensity_window: [1.0, 99.0]
    gradient_source: "normal"
    gradient_window: [1.0, 99.0]
    gradient_ksize: 3
```

This creates:

```text
R = normal robust intensity window
G = histogram-equalized version of the same intensity window
B = Sobel gradient magnitude, robustly rescaled to 8-bit
```

This is useful when `multi_window` still looks visually close to grayscale. The channels are more complementary: one stable intensity channel, one contrast-enhanced channel, and one edge/texture channel.

### Available RGB schemes

```yaml
image_export:
  rgb_scheme: "intensity_equalized_gradient"   # current default
```

Supported values:

| Value | Meaning | Recommended? |
|---|---|---|
| `intensity_equalized_gradient` | R: normal intensity, G: equalized intensity, B: Sobel gradient magnitude. | Current default |
| `multi_window` | Three different percentile windows become R, G, and B. | Good alternative |
| `grayscale_rgb` | One window is repeated into all three channels. | OK baseline |
| `equalized_rgb` | One window is histogram-equalized and repeated. | OK for visual/debug tests |
| `bitpack16` | High 8 bits and low 8 bits of a preserved uint16 image are packed into RGB channels. | Not recommended |

`bitpack16` is included only for experiments. Normal color augmentations, HSV augmentation, and pretrained RGB backbones do not understand that the channels are byte-coded intensity. The preserved 16-bit PNG is the safer way to keep high-bit-depth information.

### Histogram equalization

Simple histogram equalization is enabled by default:

```yaml
histogram_equalization:
  enabled: true
  apply_to: "third_channel"
```

For `intensity_equalized_gradient`, the second channel is already equalized by the scheme, so global post-equalization is skipped.

Options for `apply_to`:

```yaml
"all_channels"   # equalize R, G, and B after windowing
"third_channel"  # equalize only the third channel
"none"           # keep the windowed channels without equalization
```

### Preserved 16-bit images

The model-training images are saved as 8-bit RGB PNGs under:

```text
images/train
images/val
images/test
```

A separate preserved 16-bit grayscale copy can be saved under:

```text
preserved_16bit/train
preserved_16bit/val
preserved_16bit/test
```

Configure it with:

```yaml
preserved_16bit:
  save: true
  percentile_range: [0.1, 99.9]
  use_foreground_mask: true
```

The 16-bit image is not used directly by YOLO/MMDetection. It is saved so that you can inspect the processed data without losing as much intensity detail as the 8-bit training image.

### Full metadata export

The exporter now saves metadata in three ways:

```text
G:/preprocessed-vindr/metadata/source_csv/
  breast-level_annotations.csv
  finding_annotations.csv
  metadata.csv

G:/preprocessed-vindr/square_crops/metadata/
  samples_metadata.jsonl
  samples_metadata_flat.csv

G:/preprocessed-vindr/baseline_uncropped/metadata/
  samples_metadata.jsonl
  samples_metadata_flat.csv
```

`samples_metadata.jsonl` is the complete per-sample metadata. It includes:

- source `study_id`, `image_id`, and DICOM path
- exported RGB image path
- optional preserved 16-bit image path
- breast-level annotation row
- metadata CSV rows
- DICOM metadata tags, if enabled
- all finding rows
- mass-only finding rows
- exported mass boxes
- crop window and preprocessing/export information

`samples_metadata_flat.csv` is a lighter table for quick filtering in pandas or Excel.

### DICOM numeric scale during export

The default is now:

```yaml
image:
  normalize: "none"
```

This keeps the DICOM numeric values after modality LUT/rescale and MONOCHROME1 correction until the export step. The exporter then computes percentile windows per processed image or crop. This is usually better than normalizing too early if you also care about 16-bit preserved outputs.


## Important note about `positive_fraction`

`positive_fraction: 0.80` is meant for the **training square-crop export**, not for the whole VinDr-Mammo dataset, not for the uncropped baseline, and not for validation/test deterministic sliding crops.

With `balance_train_positive_fraction_globally: true`, the exporter tries to keep about 80% of training square crops mass-positive by creating positive random crops around mass annotations and adding only the needed number of clean crops. It does not automatically add one negative crop from every image without a mass, because that can make the final positive percentage much lower than 80%.

Check the actual achieved percentage after export here:

```text
G:/preprocessed-vindr/square_crops/stats/summary.csv
```

The column to check is `positive_image_percent` for the `train` split.


## Completion markers and timing manifest

At the end of a successful export, the project writes two final files directly under the output root:

```text
G:/preprocessed-vindr/EXPORT_DONE.txt
G:/preprocessed-vindr/manifest.json
```

These files are written only after the square-crop export, baseline export, labels, COCO JSON files, metadata, and summary files have been created. If the run crashes or is interrupted before the end, these files should be missing or stale.

`EXPORT_DONE.txt` is a quick human-readable completion marker. It contains the start time, finish time, total duration, and per-stage durations.

`manifest.json` is the machine-readable version. It contains:

- `status`, usually `completed`
- `started_at` and `finished_at`
- `total_duration_seconds` and `total_duration_minutes`
- `stage_timings`, with one timing entry for each major export stage
- `file_counts`, including image, label, and preserved-16-bit PNG counts per dataset and split
- `expected_files`, with existence and file size checks for important outputs
- `summary`, the same high-level export summary returned by the Python function
- `config_snapshot`, the resolved configuration used for the run

A quick PowerShell check is:

```powershell
Get-Content "G:\preprocessed-vindr\EXPORT_DONE.txt"
Get-Content "G:\preprocessed-vindr\manifest.json" -TotalCount 80
```

The resolved YAML configuration is saved in both of these locations:

```text
G:/preprocessed-vindr/metadata/export_config_resolved.yaml
G:/preprocessed-vindr/metadata/source_csv/export_config_resolved.yaml
```

The duplicate under `source_csv` is intentional. It makes it easy to keep the copied source CSVs and the exact export configuration in the same folder.

## Visualizing an already exported dataset

After the export is complete, use `visualize_export.py` to make plots without reading DICOM files again:

```bash
python visualize_export.py
```

This script uses the CSV/JSON files already saved under `paths.output_root`. It is meant to be fast because it reads files like:

```text
square_crops/stats/summary.csv
square_crops/stats/samples.csv
baseline_uncropped/stats/summary.csv
baseline_uncropped/stats/samples.csv
manifest.json
```

It saves figures and an HTML report to:

```text
G:/preprocessed-vindr/visualizations/
```

Configure it in YAML:

```yaml
visualizations:
  output_dir: "G:/preprocessed-vindr/visualizations"
  include_square_crops: true
  include_baseline_uncropped: true
  write_html_report: true
  max_rows_per_samples_csv: null
```

`max_rows_per_samples_csv: null` means use all rows. This is recommended for final plots. If the CSVs are very large and you only want a quick preview, use a number such as `5000`.

The script creates plots for image counts, positive-image percentage, mass-box counts, mass-area distributions, image sizes, crop modes, view/laterality, RGB scheme, histogram equalization, and export stage timings when `manifest.json` exists.

## Deterministic training crop export

The exporter now supports per-split crop modes under `square_crops`:

```yaml
square_crops:
  crop_size: 1024
  stride: 512
  train_crop_mode: "deterministic"
  val_crop_mode: "deterministic"
  test_crop_mode: "deterministic"
```

Use this when you want the training distribution to match validation and test. Earlier exports used random, mass-centered crops for training and deterministic sliding windows for validation/test. That can make the model overfit to centered positive crops and fail on sliding-window validation.

Supported values are:

- `deterministic`: use sliding square windows with the configured `stride`.
- `random`: use random/mass-centered crops. For train only, this also enables options like `random_crops_per_annotation`, `positive_fraction`, and `balance_train_positive_fraction_globally`.

When `train_crop_mode: "deterministic"`, the random-crop balancing options remain in the YAML for convenience, but they are ignored.

The recommended portable Ultralytics YAML is written at:

```text
<output_root>/square_crops/vindr_mass.yaml
```

It contains no absolute path and no `path: .` field:

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: mass
```

Pass this file directly to Ultralytics. The older compatibility YAML under `square_crops/ultralytics/vindr_mass.yaml` uses `../images/...` paths because it lives one folder below the dataset root.


### v3 deterministic positive-only train dataset

The default configuration now targets `/mnt/t9/preprocessed-vindr-v3`. The purpose is to create a deterministic training set but exclude empty training windows:

```yaml
paths:
  output_root: "/mnt/t9/preprocessed-vindr-v3"

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
```

This means:

* Train: sliding-window crops only, but crops with zero mass boxes are not exported.
* Val/test: full sliding-window crops, including empty crops, for realistic evaluation.
* The exact train positive percentage should be close to 100 percent when `allow_partial_annotations: false`, because empty train crops are filtered before export.

The positive/empty decision uses the final crop annotation policy. With the current default:

```yaml
crop_annotation_policy:
  allow_partial_annotations: false
  reject_partial_windows: true
```

the exporter keeps a deterministic train window only if at least one complete mass box is inside the crop and the crop does not cut through another mass box.

### Deterministic foreground-ratio crop filter

The exporter can now filter deterministic sliding-window crops by the amount of breast foreground inside the crop. This is different from `preprocess.crop_breast`:

- `preprocess.crop_breast: true` crops the whole mammogram to the detected breast box before square-crop generation. By default this is now `false`, so the global breast crop is skipped unless you enable it.
- `deterministic_require_foreground: true` keeps the full preprocessed image, creates normal sliding windows, and rejects individual windows if too little of that crop is breast foreground.

This is useful for experiments where you want to disable the global breast crop and let the square-crop stage decide which windows contain enough breast tissue:

```yaml
preprocess:
  invert_to_black_background: true
  crop_breast: false
  mirror_right_to_left: true

square_crops:
  train_crop_mode: "deterministic"
  val_crop_mode: "deterministic"
  test_crop_mode: "deterministic"

  deterministic_require_foreground: true
  deterministic_min_foreground_fraction: 0.05
  deterministic_foreground_threshold: null
```

You can override the foreground filter per split by setting:

```yaml
square_crops:
  train_deterministic_require_foreground: true
  val_deterministic_require_foreground: true
  test_deterministic_require_foreground: true

  train_deterministic_min_foreground_fraction: 0.05
  val_deterministic_min_foreground_fraction: 0.05
  test_deterministic_min_foreground_fraction: 0.05
```

The exporter stores `foreground_filter_enabled`, `foreground_fraction`, and `min_foreground_fraction` in `samples.csv` and per-sample metadata so you can audit which crops passed the filter.

## COCO size statistics in visualization reports

The visualization command now reads the exported COCO/MMDetection annotation files under:

```text
<output_root>/square_crops/mmdetection/annotations/instances_train.json
<output_root>/square_crops/mmdetection/annotations/instances_val.json
<output_root>/square_crops/mmdetection/annotations/instances_test.json
```

It computes per-box COCO-style object size bins using bbox area in pixels:

- small: `sqrt(area) < 32`, approximately area `< 32 x 32`
- medium: `32 <= sqrt(area) < 96`, approximately `32 x 32` to `< 96 x 96`
- large: `sqrt(area) >= 96`, approximately area `>= 96 x 96`

The report writes:

```text
visualizations/coco_box_annotations.csv
visualizations/coco_box_size_stats.csv
visualizations/20_coco_box_size_counts.png
visualizations/21_coco_box_size_percentages.png
visualizations/22_coco_sqrt_box_area_hist.png
visualizations/23_coco_box_width_height_scatter.png
```

These are meant to help interpret AP_small, AP_medium, and AP_large behavior in COCO-style evaluation.


## v55 random/bbox-safe global balance note

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
