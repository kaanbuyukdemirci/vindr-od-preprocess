# Email draft — Paper 22 closest reproduction v2

**Subject:** Implementation questions about your VinDr-Mammo patched-inference results

Dear Prof. Wang,

I am trying to reproduce the VinDr-Mammo mass-detection results from your
study, *“Refining YOLOv8 for Full Field Digital Mammograms: Improving Small
Object Detection through Resolution-Preserving Patched Inference,”* as part of
a study I hope to submit to SPIE Medical Imaging. My latest audited
reproduction remains substantially below the paper's reported source-mammogram
mAP50 of 0.671, and I would be very grateful for clarification of a few
implementation details that are not specified in the paper.

My closest protocol-aligned reconstruction uses the following data pipeline:

1. I apply the DICOM modality transform and VOI LUT/windowing, correct
   MONOCHROME1 polarity when necessary, and perform per-image min-max scaling.
2. I estimate the breast region on the complete mammogram by thresholding
   relative to the border intensity and a robust noise estimate, followed by a
   7-pixel morphological opening, 21-pixel closing, largest-component
   selection, and hole filling. Pixels outside the retained component are set
   to zero.
3. I preserve the full-mammogram geometry and laterality: I do not crop to the
   breast bounding box, mirror right breasts, or apply CLAHE. The normalized
   grayscale image is replicated into three identical channels and saved as an
   8-bit lossless PNG.
4. I extract 640 x 640 windows at stride 512 (20% overlap), including an
   edge-aligned final window. A clipped Mass annotation is retained when at
   least 30% of its original area is visible. A smaller lesion fragment is not
   eligible as a clean negative.
5. I retain all positive training windows and a seeded, global 20% sample of
   eligible clean negative windows. Validation and test retain all eligible
   non-background windows.

The source cohort contains only mammograms with at least one valid Mass
annotation. I preserve the official VinDr test cohort. Because the paper does
not provide the train/validation identities or split seed, I use a
deterministic, study-level, BI-RADS-stratified split that matches the reported
counts:

| Split | Studies/exams | Source images | Source Mass boxes |
|---|---:|---:|---:|
| Train | 398 | 758 | 841 |
| Validation | 71 | 136 | 148 |
| Test | 115 | 219 | 237 |

The audited patch dataset contains:

| Split | 640 x 640 patches | Positive patches | Empty patches | Crop-local Mass boxes |
|---|---:|---:|---:|---:|
| Train | 4,355 | 1,782 | 2,573 | 1,862 |
| Validation | 2,948 | 312 | 2,636 | 319 |
| Test | 4,520 | 499 | 4,021 | 514 |
| **Total** | **11,823** | **2,593** | **9,230** | **2,695** |

For training, 12,867 clean negative candidates are eligible and 2,573 are
retained, which is the rounded 20% global sample. All 1,226 source Mass
annotations are represented by at least one patch, and no study crosses
splits.

I train a COCO-pretrained YOLOv8s with P2/P3/P4/P5 detection heads and CBAM
after the added P2 neck block. I use Ultralytics 8.4.91, image size 640, batch
size 32, SGD, initial learning rate 0.0003, momentum 0.937, weight decay
0.0005, seed 42, automatic mixed precision, a 400-epoch limit, and patience 50.
The enabled augmentations are horizontal and vertical flips with probability
0.5 each, HSV (`h=0.015`, `s=0.7`, `v=0.4`), mosaic with probability 1.0, and
random erasing with probability 0.4. Translation, scale, shear, mixup, and
copy-paste are disabled.

During training, independent-patch validation mAP50 peaked at 0.451 at epoch
66, and training stopped after 116 epochs. This patch-level result is not
directly comparable with the paper's full-mammogram result. Using the saved
best checkpoint, I restored predictions to source coordinates, grouped them by
mammogram, and applied my approximation of Maximum Box Fusion. On the 219-image
official test cohort, I obtained:

| Evaluation | mAP50 | mAP50:95 |
|---|---:|---:|
| My reconstructed source-mammogram test result | 0.173 | 0.070 |
| Paper's reported VinDr Mass result | 0.671 | 0.366 |

My inference settings are a detection threshold of 0.001, patch NMS IoU of
0.70, fusion IoU of 0.50, a minimum overlap-cluster size of two, and
confidence-weighted box coordinates and scores. I regard this as approximate
rather than author-equivalent because the numerical `Tdet`/`Tc` thresholds and
the correlation-adjustment procedure are not reported.

Could you please clarify, if possible:

- the exact train/validation identities or split seed;
- the breast-mask/background-removal method and final edge-window policy;
- the rule for including a partially visible Mass box;
- whether the 20% negative sampling was global, per image, or per patient, and
  how it was rounded;
- the Ultralytics version, optimizer/schedule, and numerical augmentation
  settings; and
- the numerical Maximum Box Fusion thresholds and correlation-adjustment
  procedure?

Even an indication of which of these differences is most likely to explain the
gap would be extremely helpful.

Best regards,  
Kaan Buyukdemirci  
M.Sc. Student, Bilkent University  
Advisor: Prof. Emine Ulku Saritas
