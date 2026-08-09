# Paper 22 dataset handoff

Copy this document into the model project and treat it as the dataset contract.
The GUI/CLI exposes two separate presets. Do not mix their splits, crops,
labels, manifests, or reported results.

## Selectable presets

| Preset | Key | Output root | Status |
|---|---|---|---|
| Paper 22 — closest available reproduction (v2; not exact) | `bulatovic_yolov8_patched_inference_vindr` | `/mnt/t9/vindr-data/preprocessed-vindr-paper22-v2` | Existing paper-like dataset |
| Custom Paper 22 — CLAHE, canonical orientation, crop-balanced (v8) | `paper22_crop_label_balanced_v8` | `/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v8` | Corrected preset; regenerate before training |

The CLI aliases are `paper22` and `custom-paper22`. The custom preset is a
controlled experiment inspired by Paper 22, not an exact reproduction.

## Important v7 provenance warning

Do not describe the existing
`/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v7` export as using
Mass-negative-image-only or Mass-negative-breast-only sampling.

An independent audit of its 2,014 empty training crops found:

| Audit result | Empty crops |
|---|---:|
| Source image itself contains a Mass elsewhere | 1,770 |
| Current or paired view of the same breast contains a Mass | 1,798 |
| Breast contains no Mass in either view | 216 |

The v7 crop-local labels are internally usable as ordinary hard negatives, but
the manifest's stricter provenance claim is false. The cause was an exporter
default that marked expanded source records as Mass-negative before selection;
the old contract then trusted the same incorrect exported flag.

V8 fixes both layers:

1. `source_image_has_mass` is computed directly from source findings.
2. `source_breast_has_mass` is computed across both views sharing
   `(study_id, laterality)`.
3. An empty crop is eligible only when both flags are zero.
4. The strict completion contract checks COCO metadata against independently
   computed source provenance and fails on either a policy violation or a
   metadata mismatch.

V8 has a new preset key and output directory so it cannot silently overwrite
or redefine the completed v7 artifact.

## V8 source split and pixels

VinDr does not expose a separate reusable patient identifier in the supplied
tables. This pipeline therefore uses `study_id` as the patient/exam grouping
unit; no study may cross train, validation, or test.

The v2 Mass-positive split is selected first. V8 expands only the selected
training studies to all official-training views. Validation membership and the
official-test Mass-positive source membership remain the same as v2.

| Split | Studies/patient-exams | Source images | Source Mass boxes |
|---|---:|---:|---:|
| Train | 398 | 1,592 | 841 |
| Validation | 71 | 136 | 148 |
| Test | 115 | 219 | 237 |

Training contains 796 breasts:

| Source-breast status | Breasts | Views |
|---|---:|---:|
| Mass in at least one view | 417 | 834 |
| No Mass in either view | 379 | 758 |

After DICOM/VOI and polarity processing, the exporter masks the breast and
mirrors right-facing images, boxes, and masks together so the chest wall is on
the left. CLAHE (`clip_limit=2.0`, `tile_grid_size=8`) is then applied once to
the full fixed-preprocessed mammogram before tiling. The enhanced grayscale
signal is replicated into R/G/B.

## Crop and label policy

- Crop size: `640 x 640`.
- Stride: 512 pixels, corresponding to 20% overlap.
- Final grid start: edge-aligned.
- Every split requires retained breast-mask fraction strictly greater than
  10%; exactly 10% is rejected.
- A clipped Mass is labeled when at least 5% of its original box area remains;
  exactly 5% is included.
- Every eligible Mass-containing training crop is mandatory.
- Empty training crops may come only from breasts with no Mass annotation in
  either view.
- The running training target is approximately 50% positive crops and 50%
  empty crops. It is a one-pass seeded selection, so a small deviation such as
  52/48 is acceptable.
- A sub-threshold lesion fragment is not eligible as a clean negative.
- Validation and test are not class-balanced; they keep every grid window that
  passes the `>10%` breast-mask rule.

Because v8 has not yet been generated, its crop counts must be read from its
new manifest after export. Do not copy v7's 4,028/1,347/2,124 crop counts into
a v8 experiment report.

## Paper comparison

The original paper reports 640-pixel patches, 20% overlap, background removal,
and retaining 20% of negative-patch candidates. V8 deliberately differs by
using CLAHE, canonical left-facing orientation, a `>=5%` partial-box rule,
strict `>10%` breast-mask crop filtering, all-view training expansion, and an
approximately 50/50 crop-label target whose negatives come only from breasts
with no Mass in either view.

V8 test uses the same 219 official VinDr source mammograms and 237 source Mass
annotations as the paper-like v2 test, but it is not the same model-facing
patch set. Mirroring, CLAHE, breast-mask filtering, and the 5% box rule change
the pixels, retained windows, and crop-local annotation instances.

## Files to load after regeneration

```text
<root> = /mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v8/square_crops
```

COCO/MMDetection:

```text
train images: <root>/images/train
train COCO:   <root>/mmdetection/annotations/instances_train.json
val images:   <root>/images/val
val COCO:     <root>/mmdetection/annotations/instances_val.json
test images:  <root>/images/test
test COCO:    <root>/mmdetection/annotations/instances_test.json
```

Ultralytics uses `<root>/vindr_mass.yaml`. Do not re-tile the PNGs, resplit
source images, or regenerate crop labels from the source CSV.

Each crop references a 1024x1024 whole-mammogram context image under
`<root>/whole_images/<split>/`. The file is stored once per source mammogram,
so multiple crop rows intentionally share it. Resolve the path from
`paired_whole_image`/`paired_whole_image_path` in the exported metadata or COCO
image record. Crop-only models may ignore it.

Do not repeat DICOM transforms, polarity correction, min-max scaling, breast
masking, mirroring, CLAHE, or per-image normalization in the model project.
Normal uint8-to-float conversion and division by 255 are appropriate.

## Source-coordinate evaluation

Training labels are crop-local. Source-level evaluation must reconstruct
predictions in mammogram coordinates and group them by `source_image_id`.
COCO crop records include `crop_window_xyxy`,
`source_preprocessing_mirrored`, source dimensions, and coordinate-space
metadata. Annotations include `source_annotation_id`,
`source_bbox_xyxy` in fixed-preprocessed coordinates, and
`source_bbox_original_xyxy` in original-DICOM coordinates.

For source-level evaluation:

1. Add the crop origin to crop-local predictions.
2. Undo horizontal mirroring when mapping predictions to original coordinates.
3. Clamp to the source dimensions.
4. Group by `source_image_id`.
5. Apply the chosen cross-window fusion within each source image.
6. Evaluate against original image-level ground truth.
7. Deduplicate ground truth by `source_annotation_id`.

## Required post-export checks

After regenerating v8, run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v8')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['status'] == 'completed'
assert manifest['summary']['source_contract']['status'] == 'pass'
contract = manifest['summary']['square_crops']['replication_contract']
assert contract['status'] == 'pass', contract

for split in ('train', 'val', 'test'):
    metric = contract['metrics'][f'{split}_breast_fraction']
    assert metric['missing_count'] == 0
    assert metric['violating_count'] == 0
    assert metric['minimum_saved'] > 0.10

balance = contract['metrics']['train_crop_label_balance']
assert abs(balance['positive_fraction'] - 0.50) <= 0.05, balance

source = contract['metrics']['train_negative_crop_source_policy']
assert source['required'] == 'mass_negative_breasts_only', source
assert source['invalid_negative_crops'] == 0, source
assert source['invalid_source_image_crops'] == 0, source
assert source['invalid_source_breast_crops'] == 0, source
assert source['missing_source_provenance_crops'] == 0, source
assert source['source_image_metadata_mismatches'] == 0, source
assert source['source_breast_metadata_mismatches'] == 0, source

visibility = contract['metrics']['crop_annotation_visibility']
assert visibility['allow_partial_annotations'] is True
assert visibility['minimum_visible_box_fraction'] == 0.05
print('Custom Paper 22 v8: PASS')
PY
```

Only after this passes should v8 be described as train-ready under the strict
Mass-negative-breast sampling contract.

## Visual review bundle

V8 enables the debug review bundle by default with 200 random crop samples per
split. Crop and mask GIFs show raw original, fixed-preprocessed full
mammogram, and crop/red-mask panels. Every GIF frame is also stored separately
under `square_crops/review/crop_frames/<split>/` or
`square_crops/review/mask_frames/<split>/`.

## Generate v8

From `/home/kaan/Desktop/vindr-od-preprocess`:

```bash
python main.py --config config/export_config.yaml --preset custom-paper22
```

This writes the corrected v8 directory without overwriting v2 or v7.
