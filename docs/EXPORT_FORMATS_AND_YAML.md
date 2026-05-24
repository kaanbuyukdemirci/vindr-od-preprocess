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
    ultralytics/
      vindr_mass.yaml
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
    ultralytics/
      vindr_mass.yaml
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
  random_crops_per_annotation: 5
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
| `deterministic_include_empty` | If true, val/test deterministic crops include clean windows. This is usually better for realistic evaluation. |

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

Train using the generated YAML, for example:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="G:/preprocessed-vindr/square_crops/ultralytics/vindr_mass.yaml", imgsz=1024)
```

For the baseline dataset, use:

```python
model.train(data="G:/preprocessed-vindr/baseline_uncropped/ultralytics/vindr_mass.yaml", imgsz=1024)
```

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

