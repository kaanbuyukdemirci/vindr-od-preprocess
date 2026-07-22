# Paper 69 preset audit

This document records the evidence boundary and the implementation of the preset
`bhat_exemplar_med_detr_vindr` (CLI alias: `paper69`). It covers offline
VinDr-Mammo data preparation for the repository's one-class **Mass** export. It
does not claim to reproduce the Exemplar Med-DETR model or its training loop.

## What the authors disclose

The paper uses the official VinDr-Mammo cohort for separate mass and
calcification detection experiments: 16,000 training images and 4,000 test
images. It says that the data was created according to MammoCLIP and describes
offline preprocessing only as cropping background from full-resolution images.
The authors' public feedback is more explicit: the crop removes excess
background outside the breast, there is no offline downscaling, and the model
receives whole mammograms rather than exported lesion tiles.

The resize reported by the authors belongs to online training augmentation, not
offline dataset creation. It uses MMDetection aspect-ratio-preserving multiscale
resizing with the short dimension sampled from 480 through 800 in increments of
32 and a 1333-pixel long-dimension bound.

The paper does **not** disclose a validation split, validation IDs, epoch count,
early-stopping rule, patience, or checkpoint-selection metric. Its only dataset
membership statement is 16,000 training and 4,000 test images. Therefore an
early-stopping validation policy cannot be reproduced from the publication.

The paper's iterative training data also changes by stage:

- Stage I uses all available lesion annotations.
- Stage II trains one lesion class against background. For mammography, eight
  random background boxes are sampled from normal images and generated once
  offline.
- Stage III uses the Stage-II model's eight highest false-positive regions as
  background and generates those annotations once offline.

These Stage-II/III boxes are model-training artifacts. They are not produced by
this repository's Paper 69 image-export preset.

## What the preset implements

The preset is hermetic for pixel- and dataset-defining configuration sections,
so unrelated GUI or YAML settings cannot silently change its output. Paths and
runtime settings remain user-configurable.

Dataset membership and annotations in preset version 3:

- Preserve all 4,000 official test images as the final held-out test cohort.
- For practical early stopping, reserve a seeded, BI-RADS-stratified 15% of
  official training studies as validation. VinDr studies contain four views, so
  this produces 13,600 training, 2,400 validation, and 4,000 test images across
  3,400/600/1,000 studies.
- Keep every view from a study together. The validation split uses seed 123.
- Record this validation policy as a training-oriented assumption because it is
  not disclosed by Paper 69.
- Export **Mass** boxes only. The paper also reports a separate calcification
  detector, but this application currently exports a one-class mass dataset.
- Validate the expected selected train/validation counts, official test
  membership, and expected 237 test Mass annotations before accepting export.

The Dash **Save Data → Dataset train / validation / test assignment** section
can instead select **Original VinDr train/test only**. That restores the strict
16,000/0/4,000 membership, but it provides no validation metric for early
stopping. The official test cohort is never silently used as validation.

Offline image transformation, in order:

1. Read the DICOM without VOI LUT/windowing and correct `MONOCHROME1` polarity
   so breast tissue is bright on a dark background.
2. Remove five pixels from every image edge and translate/clip annotations into
   that coordinate system.
3. Linearly map the finite per-image minimum and maximum to `[0, 255]`, then
   convert to unsigned 8-bit values.
4. Detect the breast extent with the closest public MammoCLIP-style crop:
   values at or below 40 are background; within the central 80% of image height,
   select the longest contiguous run of nonconstant columns; within the central
   80% of that width, select the longest contiguous run of nonconstant rows.
5. Crop to those row and column extents with zero additional margin and
   translate/clip boxes again.
6. Save the native-size cropped breast as an 8-bit RGB PNG by copying the same
   grayscale values into R, G, and B.

There is no offline resize, padding, histogram equalization, tissue masking,
left/right mirroring, sliding-window crop export, or paired whole-image export
in this preset. Model-time augmentation parameters are preserved as provenance
metadata but are not applied to the saved images.

## Fidelity limit

This is a faithful implementation of the *disclosed* protocol, but exact
byte-for-byte reproduction of the authors' private dataset is not possible. The
paper, supplement, and author feedback do not publish:

- the actual excess-background crop implementation, threshold, or margin;
- the exact DICOM rescale/LUT/window/polarity and quantization order;
- any internal validation membership or random seed, if one was used;
- the Stage-II random box coordinates; or
- the trained Stage-II checkpoint and resulting Stage-III false-positive boxes.

The official paper page lists no code repository. Consequently, the five-pixel
trim, threshold of 40, longest-contiguous-run breast crop, and uint8 encoding are
explicit surrogate choices derived from the closest public MammoCLIP workflow,
not details claimed by Paper 69 itself. The exported manifest embeds these
choices under `study_preset_provenance` so downstream work can audit them.

## Version-2 materialization finding and version-3 correction

The materialized trees at
`/mnt/t9/vindr-data/preprocessed-vindr-paper69-em-detr-v2` and
`/mnt/t9/vindr-data/preprocessed-vindr-paper69` pass their own file and annotation
consistency checks: 20,000 images and labels, 1,226 in-bounds Mass boxes, exact
COCO/YOLO/metadata agreement, and disjoint study splits. However, their stored
config snapshot predates the zero-margin preset correction. It records a fixed
32-pixel breast-crop margin and 0.30 crop-stage box visibility, rather than the
documented zero margin and 0.0 crop-stage visibility.

No source Mass annotation was lost, but reusing the v2 name for the corrected
pixel contract would make two different datasets indistinguishable. The
corrected preset is therefore version 3 and writes
`preprocessed-vindr-paper69-em-detr-v3`. Do not relabel or overwrite v2. A model
run claiming the version-3 contract must wait for a completed v3 manifest and
pass the checks in
[Paper 69 model-project data handoff](PAPER69_MODEL_DATA_HANDOFF.md).

The version-3 pipeline was additionally exercised on eight real VinDr DICOMs
covering positive and normal images across train, validation, and official test.
Every sample produced a native-size zero-margin crop, retained valid in-bounds
Mass boxes, stayed in integer `[0, 255]` intensity space, and encoded exact
replicated uint8 RGB channels.

## Relationship to `simple-preset`

`simple-preset` is the requested custom paired crop/whole-image pipeline; it is
not part of Exemplar Med-DETR. In particular, it adds breast masking and
left/right canonicalization, whole-breast histogram equalization and channel-wise
percentile normalization before cropping, an 80% retained-mask minimum for online
sampled training negatives (positives bypass it), complete validation/test crop
grids, 1024-pixel crops at stride 512 with regular-stride zero padding, and one
padded-then-resized whole image companion per crop. See
[Paired crop data contract](PAIRED_CROP_DATA_CONTRACT.md)
for the downstream loading contract.

## Primary sources

- [MICCAI open-access paper, reviews, author feedback, supplement, and code status](https://papers.miccai.org/miccai-2025/0310-Paper2054.html)
- [Open-access Paper 69 PDF](https://papers.miccai.org/miccai-2025/paper/2054_paper.pdf)
- [MammoCLIP official implementation](https://github.com/batmanlab/Mammo-CLIP)
- [MammoCLIP paper](https://papers.miccai.org/miccai-2024/paper/0926_paper.pdf)
- [Official VinDr-Mammo dataset page](https://vindr.ai/datasets/mammo)
- [VinDr-Mammo v1.0.0 on PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/)
