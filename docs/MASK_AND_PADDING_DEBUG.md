# Breast-mask and crop-padding audit

The diagnostic command writes the **exact retained full-image mask used by the
exporter**, the crop grid, crop-level mask fractions, and out-of-image padding
maps. It does not modify the dataset or preset.

From the repository root, inspect eight evenly spaced source images with the
Custom Paper 22 settings:

```bash
python debug_preprocessing_masks.py \
  --config config/export_config.yaml \
  --preset custom-paper22 \
  --output-dir mask_debug/custom-paper22
```

After `pip install -e .`, the equivalent command is:

```bash
vindr-mammo-debug-masks \
  --config config/export_config.yaml \
  --preset custom-paper22 \
  --output-dir mask_debug/custom-paper22
```

Inspect exact records or source image IDs with:

```bash
python debug_preprocessing_masks.py \
  --preset custom-paper22 \
  --indices 0 1000 5000 \
  --output-dir mask_debug/selected-records

python debug_preprocessing_masks.py \
  --preset custom-paper22 \
  --image-id SOURCE_IMAGE_ID \
  --output-dir mask_debug/selected-image
```

Use `--split val` or `--split test` to display that split's threshold and
comparison rule. Custom Paper 22 currently uses the same strict `> 0.10` rule
for all three splits. `--max-images` and `--max-crop-previews` control output
volume.

## Output for each mammogram

- `00_before_breast_mask.png`: fixed-preprocessed mammogram before outside
  pixels are zeroed.
- `01_fixed_preprocessed_image.png`: image after the preset's fixed breast
  masking and orientation handling.
- `02_retained_breast_mask.png` and `retained_breast_mask.npy`: viewable and
  exact lossless copies of the mask used by export.
- `03_mask_overlay.png`: breast interior at normal brightness, outside region
  dimmed, and the mask boundary in green.
- `04_masked_image.png`: source signal with non-breast pixels removed.
- `05_crop_grid_overlay.png`: kept windows in green, rejected windows in red,
  and windows containing out-of-image padding in magenta.
- `windows.csv`: coordinates, breast pixels, full-crop breast fraction,
  valid-source pixels, padding pixels, rule, and keep/reject decision for every
  candidate window.
- `crop_previews/`: representative crops near the threshold, at the mask
  extremes, and with padding. White means breast; magenta means padding.
- `mask_method_comparison/`: the current robust largest-component method next
  to Otsu and percentile-threshold alternatives, including IoU and mask-quality
  measurements.
- `summary.json`: per-image QC flags and aggregate window counts.

The breast-fraction denominator is always the complete crop area (for this
preset, `640 x 640`). Padding is therefore false in the breast mask and counts
as non-breast exactly as it does in export. With the current `edge_align`
policy, ordinary VinDr images larger than 640 pixels normally have zero crop
padding; a blank padding map is expected. Magenta appears when the source is
smaller than the crop or a padding edge policy is selected.

## What to review

Review several CC and MLO images from both lateralities and several vendors.
The green boundary should follow the skin line and chest wall without retaining
standalone text/markers. Pay special attention to:

- an empty mask or one covering less than 5% or more than 95% of the image;
- clipped breast tissue at the nipple, inferior fold, or pectoral region;
- labels joined to the breast component;
- holes inside dense tissue; and
- crops immediately above and below the 10% threshold.

`summary.json` flags implausible mask coverage automatically. The comparison
masks make a replacement method easy to evaluate, but they do not silently
change production output. If the retained method fails consistently, compare
the same named image IDs across methods before changing the preset and
regenerating its versioned dataset.
