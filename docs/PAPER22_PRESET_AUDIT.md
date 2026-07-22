# Paper 22 VinDr-Mammo preset audit

> This audit applies to the original paper-like preset alias `paper22` and its
> `preprocessed-vindr-paper22-v2` output. A second GUI/CLI preset,
> `Custom Paper 22 — improved breast-balanced foreground crops (v4)` (CLI alias
> `custom-paper22`) is a deliberately custom subset: patient-safe breast
> expansion, strict `>10%` breast-mask coverage for train/validation/test crops,
> and exact 50/50 training-crop balance by `(study_id, laterality)` breast
> status. Validation/test retain every grid candidate that passes the same mask
> rule. See
> [`PAPER22_MODEL_DATA_HANDOFF.md`](PAPER22_MODEL_DATA_HANDOFF.md) before choosing
> a dataset; the two versions must not be mixed or described as equivalent.

Paper: Bulatović et al., *Refining YOLOv8 for Full Field Digital
Mammograms: Improving Small Object Detection through Resolution-Preserving
Patched Inference* (RCAR 2025), DOI
`10.1109/RCAR65431.2025.11139462`.

## Implemented export contract

- One-class `Mass` detector source cohort containing only images with at least
  one valid Mass box.
- Official VinDr test membership is preserved.
- Official-training mass-positive studies are split deterministically at
  study/patient level within compound BI-RADS strata.
- Counts must be exactly 398 studies / 758 images for train, 71 / 136 for
  validation, and 115 / 219 for test. Test must contain 237 source boxes.
- DICOM processing is modality LUT, requested VOI LUT/windowing,
  MONOCHROME1-to-MONOCHROME2-style polarity correction when needed, min-max
  scaling, full-mammogram background masking, and lossless 8-bit PNG output.
- Original geometry is preserved: no breast bounding-box crop and no canonical
  laterality mirror.
- PNG channels are exact grayscale replicas (`R == G == B`).
- Patch size is 640×640. Nominal stride is 512 (20% overlap); the final start on
  each axis is edge-aligned when a regular stride does not land on the border.
- A patch is positive when at least 30% of a source Mass box is visible.
  Positive patches bypass foreground rejection unconditionally.
- Training keeps all positive patches plus a seeded, globally sampled, rounded
  20% of clean foreground-negative candidates. A patch intersecting a lesion
  below the 30% rule is excluded from the training-negative pool.
- Validation/test keep all non-background inference patches and never apply the
  training negative sampler.
- Every COCO image records source image/study IDs and source-space crop
  coordinates. Every COCO annotation records a stable source annotation ID,
  source CSV row, and source-space box.

## Paper facts versus replication assumptions

The paper reports the class-specific study/image counts, patient-level split,
official VinDr test use, BI-RADS stratification, VOI processing, background
removal, 0–255 PNG output, 640 patches, 20% overlap, and 20% negative-patch
retention for training.

The paper does **not** publish the train/validation identities or seed, its
background-removal algorithm, partial-box rule, negative-sampling granularity
or rounding, exact edge policy, PNG channel representation, or author code.
This implementation therefore records the following assumptions in every
resolved config/manifest:

- seed 123 and a deterministic **count-matched**, not author-identical, split;
- 5% minimum breast-mask area for non-positive patches;
- 30% source-box visibility;
- global split-level negative sampling with `round(N * 0.20)`;
- edge-aligned final grid starts;
- replicated grayscale RGB;
- per-image min-max scaling after VOI/polarity processing; and
- the conventional DICOM order of modality LUT, VOI, then display-polarity
  inversion. Representative MONOCHROME1 and MONOCHROME2 VinDr files were
  checked to end with dark borders and bright tissue.

## Strict completion gates

The exporter refuses to write a completion manifest when any of these fail:

1. Published source study/image counts and 237 official-test boxes.
2. Exact official positive-test membership and no study/image leakage.
3. Every selected source image has a valid Mass box.
4. Candidate-positive and written-positive window key sets are identical.
5. Every source annotation ID is represented by at least one written patch.
6. Exact rounded 20% training-negative retention.
7. Validation/test use `all`, not the training sampler.
8. Validation/test retain at least 30% of the complete grid.
9. Every patch/window is 640×640 and every COCO box/reference is valid.

The manifest also records the resolved preset, source CSV SHA-256 hashes, Git
revision/dirty state, runtime versions, split counts, and contract reports.

## Verification on the local VinDr source

The full planning audit read all 1,113 valid Mass-positive images and all 1,226
Mass boxes:

| Split | Sources | Source boxes represented | Complete grid | Eligible grid | Eligible fraction |
|---|---:|---:|---:|---:|---:|
| Train | 758 | 841 / 841 | 29,777 | 14,649 | 49.20% |
| Validation | 136 | 148 / 148 | 5,457 | 2,948 | 54.02% |
| Test | 219 | 237 / 237 | 8,486 | 4,520 | 53.26% |

No positive source lacked a positive window. Training has 12,867 eligible
clean negatives, so the materialized export will retain 2,573 of them.

A three-source real-DICOM end-to-end smoke export also passed both internal
strict gates and the model repository's strict Paper 22 dataset audit at 100%
source-annotation coverage. All 65 generated PNGs were 640×640 replicated RGB,
and every generated COCO annotation was source-traceable.

## Materialized v2 acceptance audit (2026-07-21)

The completed export at `/mnt/t9/vindr-data/preprocessed-vindr-paper22-v2` passed the
internal strict contract and the model repository's independent strict audit
with zero errors/warnings and 100% source-annotation coverage in every split.
A separate file-level audit checked all 11,823 PNGs and found:

- exact COCO/image/YOLO/metadata filename sets and 11,823 empty-or-populated
  label files;
- 4,355/2,948/4,520 train/validation/test tiles and 1,862/319/514 tile boxes;
- 640×640 uint8 RGB rasters with identical R/G/B channels for every tile;
- valid in-bounds COCO boxes, matching YOLO boxes, matching metadata boxes, and
  matching independently reconstructed source-to-tile transforms;
- disjoint study and source-image IDs across all splits; and
- no missing or extra metadata row.

The materialized v2 dataset is accepted for model use. The model project's
committed Paper 22 experiment is pinned to
`/mnt/t9/vindr-data/preprocessed-vindr-paper22-v2/square_crops`. See
[Paper 22 model-project data handoff](PAPER22_MODEL_DATA_HANDOFF.md).

## Usage

From the repository root:

```bash
vindr-mammo-export --config config/export_config.yaml --preset paper22

# Or without installing the console script:
python main.py --config config/export_config.yaml --preset paper22
```

The preset writes to `preprocessed-vindr-paper22-v2` under the output parent so
the known-invalid `preprocessed-vindr-paper22` export is not overwritten.

After the full export, run the model-project gate:

```bash
python /home/kaan/Desktop/vindr-od-many/scripts/audit_paper22.py \
  /mnt/t9/vindr-data/preprocessed-vindr-paper22-v2/square_crops \
  --strict --min-coverage 1.0 \
  --json-output /tmp/paper22-v2-audit.json
```

An author-identical or bit-for-bit replica remains impossible without the
authors' split manifest and missing preprocessing/sampling/MBF details. The
preset makes every such choice explicit and reproducible instead of presenting
it as a disclosed paper fact.
