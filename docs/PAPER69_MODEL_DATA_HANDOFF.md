# Paper 69 model-project data handoff

Copy this document into the model project and treat it as the dataset loading
contract for Bhat et al., *Exemplar Med-DETR: Toward Generalized and Robust
Lesion Detection in Mammogram Images and Beyond* (MICCAI 2025, DOI
`10.1007/978-3-032-04978-0_20`). This preprocessing repository exports the
one-class VinDr-Mammo **Mass** dataset only.

The GUI name is **Paper 69 — closest available reproduction (v3; not exact)**
and its CLI alias is `paper69`. It is not labeled an exact copy because the
paper does not publish its complete crop code, validation IDs, or all
Stage-II/III details.

## Dataset decision

The canonical corrected preset is version 3 and writes:

```text
/mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v3/baseline_uncropped
```

As of the 2026-07-21 audit, that six-hour full export has not yet been
materialized. Generate it from the preprocessing repository with:

```bash
cd /home/kaan/Desktop/vindr-od-preprocess
/home/kaan/anaconda3/envs/data-mmdet/bin/python main.py \
  --config config/export_config.yaml --preset paper69
```

Do not point a clean v3 replication at either of these older names:

```text
/mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v2/baseline_uncropped
/mnt/t9/vindr-data/preprocessed-vindr-paper69/baseline_uncropped
```

The two older trees are currently content-equivalent and their image/COCO/YOLO
packaging passes a complete consistency audit. However, their stored manifest
shows `crop_padding: 32` and box visibility `0.30`, while the audited closest
public MammoCLIP surrogate uses zero extra crop margin and zero crop-stage
visibility loss. No source Mass box was lost in v2, but it must not be presented
as byte-equivalent to the corrected v3 contract.

The model project's committed Paper 69 experiment currently names the ambiguous
unversioned root. After v3 finishes and passes the checks below, change it to:

```yaml
data_root: /mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v3/baseline_uncropped
```

## Exact files to load after v3 export

```text
root:        /mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v3/baseline_uncropped
train imgs:  images/train
train ann:   mmdetection/annotations/instances_train.json
val imgs:    images/val
val ann:     mmdetection/annotations/instances_val.json
test imgs:   images/test
test ann:    mmdetection/annotations/instances_test.json
```

Each COCO `file_name` is a basename, resolved against
`<root>/images/<split>`. COCO category ID `1` means `mass`; convert it to model
class index `0` when required. Empty images are intentional normal/background
mammograms and must not be discarded globally.

## Expected cardinality

| Split | Studies | Images | Positive images | Mass boxes |
|---|---:|---:|---:|---:|
| Train | 3,400 | 13,600 | 743 | 829 |
| Validation | 600 | 2,400 | 151 | 160 |
| Test | 1,000 | 4,000 | 219 | 237 |

The 4,000-image official VinDr test cohort is unchanged. The paper publishes a
16,000/4,000 train/test membership but does not disclose a validation policy.
This export's 13,600/2,400 split is a seeded, BI-RADS-stratified, study-level
15% holdout from official training for checkpoint selection. It is an explicit
replication assumption. Never tune on the 4,000 test images and never resplit
individual views.

## Image and label semantics

- Inputs are whole breast-cropped mammograms, not square lesion tiles.
- PNG height and width vary per image. Read them from COCO or the actual PNG;
  never assume a fixed offline canvas.
- Images are lossless uint8 RGB with exact replicated grayscale channels.
- Offline preprocessing corrects MONOCHROME1 polarity, removes five source
  border pixels, performs per-image min-max conversion to uint8, and applies the
  MammoCLIP-style threshold-40 longest-contiguous breast extent.
- Version 3 adds no crop padding, resize, letterbox, histogram equalization,
  tissue masking, or left/right mirroring.
- COCO boxes are `xywh` in the saved breast-crop coordinate system. They are
  already translated and clipped; do not subtract the breast crop again.
- Normal tensor conversion/division by 255 is expected. Do not re-run DICOM
  windowing or a second per-image min-max operation on the PNG.

The paper's reported resize belongs in the online model pipeline. Preserve
aspect ratio and use the MMDetection scales:

```yaml
train_scales:
  - [480, 1333]
  - [512, 1333]
  - [544, 1333]
  - [576, 1333]
  - [608, 1333]
  - [640, 1333]
  - [672, 1333]
  - [704, 1333]
  - [736, 1333]
  - [768, 1333]
  - [800, 1333]
keep_ratio: true
validation_test_scale: [800, 1333]
```

Do not offline-resize the materialized PNGs to these dimensions.

## Exemplar Med-DETR stage boundary

This dataset export provides images and lesion ground truth. It does not create
the model-dependent background exemplars described by the paper:

- Stage I consumes the retained Mass ground truth.
- Stage II adds eight randomly sampled background regions from normal training
  images.
- Stage III adds the Stage-II model's eight highest false-positive regions.

Generate Stage-II and Stage-III regions inside the model workflow using training
data only. Do not store them as if they were source VinDr annotations, and do
not mine them from validation or test.

## Required acceptance checks

Before training, require all of the following:

1. `manifest.json` at the v3 dataset parent has `status: completed`.
2. Its config snapshot has preset version `3`, `crop_padding: 0`,
   `min_box_visibility_after_crop: 0.0`, and baseline resize mode `none`.
3. Counts are exactly 13,600/2,400/4,000 images and 829/160/237 Mass boxes.
4. Train/validation/test study ID sets are disjoint and the official test
   cohort is preserved.
5. Every COCO image exists, matches its recorded dimensions, and has exactly
   one adjacent YOLO label file if the YOLO representation is used.
6. Every box has positive area and lies inside its image.

Then run the model-project preflight:

```bash
cd /home/kaan/Desktop/vindr-od-many
PYTHONPATH=src python scripts/paper69_workflow.py preflight \
  --experiment experiments/paper69_em_detr_mass_only.yaml \
  --source-coco /mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v3/baseline_uncropped/mmdetection/annotations/instances_train.json
```

Abort if the v3 path does not exist, the manifest is incomplete, or preflight
fails.

## Fidelity and score interpretation

The paper and author feedback specify full-resolution background cropping and
online multiscale resizing, but do not publish the exact crop code, DICOM
intensity procedure, validation IDs, Stage-II random regions, or Stage-III
checkpoint/false positives. Version 3 uses the closest cited public MammoCLIP
crop as an explicit surrogate; byte-identical author data is not possible from
the publication alone.

The audited v2 packaging retained all 1,226 source Mass boxes, so a validation
mAP of exactly zero with hundreds of detections per image is not explained by a
missing-label or category-ID failure in that export. After moving to v3 for a
clean data contract, investigate initialization, frozen components, learning
rates, staged exemplar construction, prediction category mapping, and the
evaluation adapter if scores remain near zero.

Primary references:

- https://papers.miccai.org/miccai-2025/0310-Paper2054.html
- https://papers.miccai.org/miccai-2025/paper/2054_paper.pdf
- https://github.com/batmanlab/Mammo-CLIP
