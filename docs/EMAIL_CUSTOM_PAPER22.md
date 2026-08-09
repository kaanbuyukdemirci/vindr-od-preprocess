# Email draft — Custom Paper 22 v8 experiment

**Subject:** Follow-up on two VinDr-Mammo patched-inference experiments

Dear Prof. Wang,

I am trying to reproduce the VinDr-Mammo mass-detection results from your
study, *“Refining YOLOv8 for Full Field Digital Mammograms: Improving Small
Object Detection through Resolution-Preserving Patched Inference,”* as part of
a study I hope to submit to SPIE Medical Imaging. I recently completed two
audited experiments with the same YOLOv8s-P2-CBAM model, training recipe, and
source-coordinate evaluator: a count-matched paper-like reconstruction and a
controlled custom preprocessing experiment. Both remain substantially below
the paper's reported source-mammogram mAP50 of 0.671.

The custom experiment deliberately differs from the closest reconstruction in
the following ways:

1. After the DICOM modality transform, VOI LUT/windowing, polarity correction,
   and per-image min-max scaling, I estimate a breast mask on the complete
   mammogram. The mask uses a threshold relative to the border intensity and a
   robust noise estimate, a 7-pixel morphological opening, a 21-pixel closing,
   largest-component selection, and hole filling.
2. I mirror right-facing mammograms, their Mass boxes, and their breast masks
   so the chest wall is consistently on the left.
3. I apply CLAHE once to the complete fixed-preprocessed mammogram, before
   tiling, with `clip_limit=2.0` and an 8 x 8 tile grid. I replicate the
   resulting grayscale image into three identical channels.
4. I extract 640 x 640 windows at stride 512 (20% overlap), with an edge-aligned
   final start. I retain a window only when strictly more than 10% of its full
   640 x 640 area is covered by the breast mask; out-of-image padding counts as
   non-breast.
5. I retain a clipped Mass annotation when at least 5% of its original area is
   visible. The 5% boundary is inclusive, and a smaller lesion fragment is not
   treated as a clean negative.
6. I expand the selected 398 training studies to all 1,592 available views. I
   retain every eligible Mass-containing training patch and sample empty
   patches only from breasts with no Mass annotation in either the current or
   paired view, producing an exact 50/50 positive/empty training-patch split.
   Validation and test are not balanced; they retain every window passing the
   custom breast-mask rule.

The source cohorts are:

| Split | Paper-like source cohort | Custom source cohort |
|---|---:|---:|
| Train | 398 studies / 758 images / 841 Mass boxes | 398 studies / 1,592 images / 841 Mass boxes |
| Validation | 71 studies / 136 images / 148 Mass boxes | 71 studies / 136 images / 148 Mass boxes |
| Test | 115 studies / 219 images / 237 Mass boxes | 115 studies / 219 images / 237 Mass boxes |

The custom training cohort contains 417 Mass-positive breasts (834 views) and
379 breasts with no Mass in either view (758 views). The completed v8 patch
dataset contains:

| Split | 640 x 640 patches | Positive patches | Empty patches | Crop-local Mass boxes |
|---|---:|---:|---:|---:|
| Train | 4,028 | 2,014 | 2,014 | 2,112 |
| Validation | 1,347 | 328 | 1,019 | 337 |
| Test | 2,124 | 532 | 1,592 | 549 |
| **Total** | **7,499** | **2,874** | **4,625** | **2,998** |

The strict export and model-side audits passed. All 841/148/237 source Mass
annotations are represented in train/validation/test, no study crosses splits,
every retained patch has breast-mask coverage strictly above 10%, and none of
the 2,014 empty training patches comes from a Mass-positive image or from a
breast with a Mass in its paired view.

For both experiments, I used a COCO-pretrained YOLOv8s with P2/P3/P4/P5
detection heads and CBAM after the added P2 neck block. I trained with
Ultralytics 8.4.91, image size 640, batch size 32, SGD, initial learning rate
0.0003, momentum 0.937, weight decay 0.0005, seed 42, automatic mixed
precision, a 400-epoch limit, and patience 50. Augmentations were horizontal
and vertical flips at probability 0.5 each, HSV (`h=0.015`, `s=0.7`, `v=0.4`),
mosaic at probability 1.0, and random erasing at probability 0.4. Translation,
scale, shear, mixup, and copy-paste were disabled.

The latest results are:

| Experiment | Best training-time patch-validation mAP50 | Source-test mAP50 | Source-test mAP50:95 |
|---|---:|---:|---:|
| Paper-like reconstruction | 0.451 (epoch 66) | 0.173 | 0.070 |
| Custom preprocessing experiment | 0.411 (epoch 37) | 0.131 | 0.051 |
| Paper's reported VinDr Mass result | — | 0.671 | 0.366 |

The custom run stopped after 87 epochs; the paper-like run stopped after 116.
The independent-patch validation columns are included only as diagnostics and
are not directly comparable with the paper's full-mammogram metric. The two
source-test rows use the same 219 mammograms and 237 source Mass annotations,
but the custom run evaluates a different set of model-facing patches because
mirroring, CLAHE, foreground filtering, and the 5% partial-box rule alter the
pixels and retained windows.

For source-level evaluation, I restore patch predictions to mammogram
coordinates and apply an approximate Maximum Box Fusion with detection
threshold 0.001, patch NMS IoU 0.70, fusion IoU 0.50, minimum overlap-cluster
size two, and confidence-weighted coordinates and scores. The custom evaluator
verified complete coverage of all 2,124 windows eligible under its `>10%`
foreground contract. I cannot claim author-equivalent fusion because the
paper's numerical `Tdet`/`Tc` values and correlation-adjustment procedure are
not reported.

Could you please clarify whether your implementation used:

- a particular breast-mask/background-removal algorithm;
- orientation standardization or histogram equalization/CLAHE;
- a specific partial-box and edge-window rule;
- global, per-image, or per-patient negative-patch sampling;
- published or reproducible train/validation identities;
- particular Ultralytics, optimizer, schedule, and augmentation settings; and
- numerical Maximum Box Fusion thresholds or correlation-adjustment code?

The custom changes did not improve source-level accuracy, so any guidance on
which discrepancy to investigate first would be especially valuable.

Best regards,  
Kaan Buyukdemirci  
M.Sc. Student, Bilkent University  
Advisor: Prof. Emine Ulku Saritas
