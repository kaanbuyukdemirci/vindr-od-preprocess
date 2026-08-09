# Lazy crop manifest contract

The Dash GUI's **Lazy Crop Manifests** tab creates a dataset index for extracting
overlapping crops at training time. It is intended for a high-overlap grid that
would be unnecessarily large if every crop were stored as both PNG and float32.

Default Research Dataset v1 supplies these defaults:

```yaml
geometry:
  window_size: 1024
  stride: 128
  edge_policy: regular_stride_pad
annotations:
  min_box_visibility: 0.05
sampling:
  train_positive_fraction: 0.50
  train_require_clean_negative_windows: true
  train_require_mass_negative_breasts: true
  seed: 123
filters:
  min_source_extent_fraction_by_split:
    train: 0.10
    val: 0.05
    test: 0.05
  preserve_positive_windows_below_threshold: true
```

## Inputs and speed

The extractor accepts either the export root or its `square_crops` child. It
reads:

```text
square_crops/metadata/whole_image_manifest.csv
square_crops/metadata/whole_image_annotations.csv
square_crops/metadata/samples_metadata_flat.csv  # optional pairing/status fields
```

It uses saved width/height, source-coordinate Mass boxes, split membership,
source/breast Mass status, and existing relative paths. It does not open source
images, calculate image statistics, regenerate breast masks, or write image
files. Both the result and progress events report `decoded_images: 0`.

## Output names

The automatic directory is:

```text
<export-root>/lazy_crop_manifests/window<window>_stride<stride>/
```

It contains:

```text
README.md
lazy_crop_config_resolved.yaml
lazy_crop_manifest.json
lazy_crop_manifest_train.csv
lazy_crop_manifest_val.csv
lazy_crop_manifest_test.csv
lazy_crop_annotations_train.csv
lazy_crop_annotations_val.csv
lazy_crop_annotations_test.csv
```

The generated README embeds the exact run values, counts, coordinate contract,
and a minimal PyTorch loader. Existing known manifest files are replaced only
when **overwrite** is selected; unrelated files are never cleaned from the
folder.

## Crop table

Each `lazy_crop_manifest_<split>.csv` row is one virtual crop. Important field
groups are:

| Group | Fields | Meaning |
|---|---|---|
| Identity | `crop_id`, `split`, `manifest_row_index` | Stable join/order keys; `crop_id` also joins annotations. |
| Source | `source_image_id`, `source_study_id`, `source_breast_key`, `source_width`, `source_height` | Fixed-preprocessed original-whole identity and shape. |
| Paths | `source_png_path`, `source_float32_path` | Same-coordinate-space sources. Use float32 only when the field is non-empty. |
| Context-only paths | `context_resized_png_path`, `context_resized_float32_path` | Compact whole-image context with different geometry; never substitute it directly for the crop source. |
| Window | `crop_x0`, `crop_y0`, `crop_x1`, `crop_y1`, `window_size`, `stride`, `edge_policy` | Requested half-open source-coordinate window. |
| In-bounds copy | `source_intersection_*`, `pad_left`, `pad_top`, `pad_right`, `pad_bottom` | Region to copy and zero padding needed to create the fixed-size crop. |
| Labels | `num_mass_annotations`, `is_mass_positive`, `max_source_box_visibility`, `min_box_visibility` | Crop-local label classification and visibility rule. |
| Source status | `source_image_has_mass`, `source_breast_has_mass`, `is_clean_negative` | Inputs to the clean-negative policy. |
| Filtering | `source_extent_fraction`, `min_source_extent_fraction`, `source_extent_comparison`, `positive_bypassed_source_extent_filter` | Exact metadata-only geometry filter and positive safeguard. |
| Sampling | `selection_policy`, `negative_source_policy`, `sampling_seed` | Reproducible selection provenance. |

Paths are relative to the existing `square_crops` root. `source_png_path` is the
primary source. The present Default Research Dataset v1 export has resized-whole
float32 tensors but no original-size same-geometry float32 tensors, so
`source_float32_path` is empty while `context_resized_float32_path` is populated.
The resized tensor is useful as global context, not as a direct source-coordinate
crop replacement.

## Annotation table

Each `lazy_crop_annotations_<split>.csv` row is one retained Mass label. It
contains:

- `crop_id`, split, source image/study, and source annotation ID;
- original fixed-preprocessed `source_bbox_*` XYXY and dimensions;
- clipped `visible_source_*` XYXY;
- translated `crop_bbox_*` XYXY and dimensions;
- intersection/original-area `visible_fraction`; and
- category ID 1, category name `Mass`.

A label is retained when `visible_fraction >= min_box_visibility`. The crop
table's `num_mass_annotations` equals the number of joined annotation rows.

## Grid, filtering, and sampling

`regular_stride_pad` starts at zero and advances only by the configured stride.
If the final regular window does not cover the far source edge, one more regular
origin is used while it remains inside the source. Any right/bottom overflow is
zero-padded. Origins are never shifted backward merely to align with an edge.

The generator classifies labels before applying the metadata filter. Because
the original wholes were already breast-cropped and background-masked,
`source_extent_fraction = in_bounds_source_area / window_area` is a useful fast
edge/background proxy. It is not a pixel-derived breast fraction: it does not
know how many in-bounds pixels are masked background. The per-row method and
threshold make this limitation explicit. Positive bypass is enabled by default.

For train, all eligible positive windows are mandatory. The generator counts
eligible clean negatives, computes the requested negative count from the target
positive fraction, and samples exact candidate indices without replacement
using the seed. With the default policy, an empty window is eligible only when
its complete source breast is marked Mass-negative and it has zero intersection
with any source Mass box. Validation and test write every eligible grid window
and are not balanced.

## Training-time extraction

For each row:

1. Resolve `source_png_path` below `square_crops`, or use
   `source_float32_path` if that field is non-empty.
2. Allocate a zero-filled `window_size x window_size` output.
3. Read the half-open `source_intersection_*` region.
4. Paste it at `(pad_left, pad_top)`.
5. Join annotation rows on `crop_id` and use `crop_bbox_*` directly.

Cache a decoded whole image while reading adjacent rows from the same
`source_image_id`; otherwise dense overlap would trade disk capacity for repeated
decode cost. Keep train/validation/test membership from the manifest to preserve
source-level split isolation.
