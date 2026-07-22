# Paper 22 dataset handoff

Copy this document into the model project and treat it as the Paper 22 data
contract. The GUI/CLI exposes two presets: the legacy paper-like v2 preset and
the current custom improved v4 preset. Never combine splits, labels, manifests,
or results from different versions.

## Selectable presets

| Preset | Key | Output root | Status |
|---|---|---|---|
| Paper 22 — closest available reproduction (v2; not exact) | `bulatovic_yolov8_patched_inference_vindr` | `/mnt/t9/vindr-data/preprocessed-vindr-paper22-v2` | Materialized and accepted |
| Custom Paper 22 — improved breast-balanced foreground crops (v4) | `paper22_patient_breast_balanced_v4` | `/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v4` | Configuration ready; run required |

The clear CLI aliases are `paper22` and `custom-paper22` respectively.
`paper22-improved` remains accepted for backward compatibility.

An accepted v3 export remains at
`/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v3`, but it is archived:
its validation/test sets contain complete unfiltered grids. Do not use v3 when
the experiment requires the current all-split breast-foreground rule.

## Improved v4 source split

VinDr's supplied tables do not expose a separate reusable patient identifier.
`study_id` is therefore the patient/exam grouping unit: all CC/MLO and
left/right views from one study stay in one split. No study may cross splits.

The v2 mass-positive split is selected first. V4 expands only the 398 selected
training studies to all of their official-training views. Validation source
membership and official-test source membership remain identical to v2/v3.

| Split | Studies/patient-exams | Source images | Source Mass boxes |
|---|---:|---:|---:|
| Train | 398 | 1,592 | 841 |
| Validation | 71 | 136 | 148 |
| Test | 115 | 219 | 237 |

Training contains 796 breasts, defined as `(study_id, laterality)`:

| Source-breast status | Breasts | Views |
|---|---:|---:|
| Mass in at least one view | 417 | 834 |
| No Mass in either view | 379 | 758 |

## Breast foreground and crop selection

The full-mammogram breast mask is computed once before tiling and retained for
crop decisions. Outside-breast pixels are masked in the exported image, and
every train, validation, and test crop must satisfy:

```text
breast_fraction > 0.10
```

The comparison is strict: exactly 10% is rejected. The denominator is the full
640x640 crop, so zero padding counts as non-breast. Background-only windows and
standalone laterality/view markers such as LCC therefore do not enter any
split. A retained window may still touch the image boundary or contain an empty
label when more than 10% of it is genuine breast tissue.

In the GUI, open **Foreground-ratio crop filter** to adjust train, validation,
and test independently. Custom Paper 22 v4 loads all three switches as enabled
at `0.10`, but each switch and threshold can be changed before export. The
resolved config and manifest record the effective per-split choices.

All candidate windows use 640x640 crops, stride 512, 20% overlap, and an
edge-aligned final start. Validation/test keep every grid candidate passing the
breast-mask rule; they are not class-balanced or negatively sampled. Because
they are foreground-filtered, they are not complete geometric grids.

Training is additionally sampled without replacement into an exact 50/50 crop
mixture by source-breast status:

- 50% of training crops come from mass-positive breasts.
- 50% come from negative breasts.
- The group is determined by the source breast, not by whether that individual
  crop contains a visible Mass.
- Every eligible lesion-containing training window is mandatory before the
  remaining candidates are sampled.
- A sub-threshold lesion fragment cannot be admitted as a clean negative.

The previous v3 run produced 8,455 crops from each source-breast group. V4 keeps
the same deterministic training rule, but its final manifest remains the
authority after regeneration.

## Training labels versus evaluation labels

Training consumes crop-local labels. Each source Mass is intersected with the
crop window, clipped to crop bounds, and translated into 640x640 crop
coordinates. COCO stores crop-local `xywh`; YOLO stores normalized
`class x_center y_center width height` with class ID `0`.

Source-level evaluation must not treat crops as independent mammograms. Each
COCO crop records:

```text
source_image_id
source_study_id
crop_window_xyxy = [x0, y0, x1, y1]
```

Each crop annotation also records `source_annotation_id`,
`source_annotation_row`, and `source_bbox_xyxy`. The original image-level Mass
labels are preserved in `metadata/source_csv/finding_annotations.csv`.

For Maximum Box Fusion evaluation:

1. Add crop `(x0, y0)` to each predicted crop box.
2. Clamp it to the original mammogram dimensions.
3. Group predictions by `source_image_id`.
4. Apply Maximum Box Fusion within each source image.
5. Evaluate fused predictions against the original image-level boxes, not the
   clipped crop labels.
6. Deduplicate ground truth by `source_annotation_id`.

Foreground filtering does not change source coordinates or source ground truth.
The model-side evaluator should assess submitted-window coverage against the
eligible foreground grid rather than requiring 100% of the geometric grid.

## Files to load

After generating v4, set:

```text
<root> = /mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v4/square_crops
```

COCO/MMDetection paths:

```text
train images: <root>/images/train
train COCO:   <root>/mmdetection/annotations/instances_train.json
val images:   <root>/images/val
val COCO:     <root>/mmdetection/annotations/instances_val.json
test images:  <root>/images/test
test COCO:    <root>/mmdetection/annotations/instances_test.json
```

For Ultralytics, load `<root>/vindr_mass.yaml`. Do not regenerate labels from
the source CSV, retile the PNGs, or resplit source images.

Normal uint8-to-float conversion and division by 255 are appropriate. Do not
repeat the DICOM transform, VOI processing, polarity correction, min-max
scaling, breast masking, or per-image min-max normalization.

Example model configuration:

```yaml
data_root: /mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v4/square_crops
output_root: /mnt/t9/vindr-model/models-vindr-paper22-improved-v4
patching:
  enabled: false
  already_applied: true
```

## Required post-export checks

The current model-project `scripts/audit_paper22.py` defaults to v2
cardinalities. Use it unchanged only for v2. It will reject v4 unless its
expectations are made version-aware.

After generating v4, require its own manifest gates before training:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/t9/vindr-data/preprocessed-vindr-paper22-improved-v4')
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
print('Paper 22 improved v4: PASS')
PY
```

Also verify that every source annotation is represented, every crop has a
matching label and COCO record, and no study overlaps another split. Do not
publish v4 crop counts until the materialized manifest passes.

## Generate the current improved preset

From `/home/kaan/Desktop/vindr-od-preprocess`:

```bash
python main.py --config config/export_config.yaml --preset custom-paper22
```

This command writes v4 under `/mnt/t9/vindr-data`. It does not overwrite the
legacy v2 or archived v3 directories. V4 is a custom controlled experiment,
not an author-exact Paper 22 subset.
