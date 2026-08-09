# Paired Crop + Whole-Image Data Contract

This contract applies when `paired_whole_images.enabled: true`, including the
`Default Research Dataset (v1)` export. The sample unit is **one crop and one whole-breast
context image**. Detection labels belong to the crop.

## Files and pairing key

The exporter writes every crop separately but writes each source mammogram only
once per enabled whole-image resolution:

```text
square_crops/
  images/<split>/<source-key>__crop__<details>.png # detector crop
  whole_images_original/<split>/<source-key>.png   # original-size processed whole
  whole_images/<split>/<source-key>.png            # one resized whole per source
  whole_images_high_resolution/<split>/<source-key>.png # optional common-canvas high-resolution whole
  whole_labels_original/<split>/<source-key>.txt
  whole_labels_resized/<split>/<source-key>.txt
  whole_labels_high_resolution/<split>/<source-key>.txt
  whole_annotations_original/<split>/<source-key>.json
  whole_annotations_resized/<split>/<source-key>.json
  whole_annotations_high_resolution/<split>/<source-key>.json
  labels/<split>/<source-key>__crop__<details>.txt # YOLO label for the crop
  metadata/crop_locations.csv            # exact crop/source/whole transforms
  metadata/samples_metadata.jsonl
  metadata/samples_metadata_flat.csv
  mmdetection/annotations/instances_<split>.json
```

The shared `source-key` is available as `paired_whole_key`. Crop rows from the
same source intentionally repeat the same whole-image paths; the files
themselves are not repeated, copied, or hard-linked under crop-specific names.

Do not parse study or image IDs when metadata is available. The recommended
loader index is `metadata/samples_metadata_flat.csv`:

- `training_image`: crop path, relative to `square_crops/`
- `paired_whole_original_image`: unpadded original-size processed whole path
- `paired_whole_image`: whole-image path, relative to `square_crops/`
- `paired_whole_high_resolution_image`: optional padded high-resolution whole path
- `paired_whole_key`: source-level key shared by the crop and all enabled whole files
- `split`, `file_name`, `source_image_id`, and `source_study_id`
- `crop_window_xyxy`: crop position in the fixed-preprocessed source image
- `num_export_boxes`: number of Mass boxes retained in the crop

`samples_metadata.jsonl` contains the same pairing plus full box coordinates,
preprocessing details, and pad/resize metadata. COCO image records also contain
`paired_whole_key`, `paired_whole_image_path` and, when enabled,
`paired_whole_high_resolution_image_path`.

Every image variant receives annotations in its own pixel coordinate space:

- crops: `labels/` and `mmdetection/annotations/`;
- original wholes: `whole_labels_original/`, `whole_annotations_original/`, and `mmdetection/whole_original/annotations/`;
- resized wholes: `whole_labels_resized/`, `whole_annotations_resized/`, and `mmdetection/whole_resized/annotations/`;
- high-resolution wholes: `whole_labels_high_resolution/`, `whole_annotations_high_resolution/`, and `mmdetection/whole_high_resolution/annotations/`.

`metadata/whole_image_manifest.csv` indexes every whole asset and
`metadata/whole_image_annotations.csv` records every transformed Mass box.

`metadata/crop_locations.csv` is the simplest authoritative location index. It
stores numeric `crop_x0`, `crop_y0`, `crop_x1`, and `crop_y1` columns in the
fixed-preprocessed source space, edge padding, the valid source intersection,
the original-DICOM mapping, and coordinates in both high-resolution and resized padded
whole-image canvases.

### Filename-only fallback

New crop names follow this contract:

```text
<source-key>__crop__<split>_<crop-number>_x<X>_y<Y>_w<W>_h<H>.png
```

Remove the final `__crop__...` part from the crop stem and add `.png` to obtain
the whole filename. Use that filename under any enabled whole-image directory.
Splitting on the final occurrence is important
(`rsplit("__crop__", 1)` in Python). Metadata paths remain authoritative.

## Shapes and pixels in Default Research Dataset v1

Crop PNGs and resized-whole PNGs are 8-bit RGB and decode to
`1024 x 1024 x 3`. Original-size processed whole PNGs retain their variable
fixed-preprocessed `H x W x 3`. The optional padded high-resolution branch is
disabled by default; if enabled manually, its shape is determined by its
configured canvas mode and dimensions. A PyTorch loader normally converts
these images to channels-first tensors and divides by 255.

Before square-window extraction, the fixed preprocessing corrects polarity,
crops and masks breast tissue, removes outside-breast artifacts/labels, and
mirrors orientation to the canonical side. The entire RGB recipe is also run on
that complete fixed-preprocessed breast before the square window is extracted.
The source DICOM is normalized per image with the `0.5–99.5` percentile range,
then whole-image CLAHE (`clip_limit=2.0`, `tile_grid_size=8`) is applied. The
same processed grayscale signal is replicated into R, G and B.

Consequently, an in-bounds crop pixel is the exact corresponding pixel from the
fully channel-processed whole breast before the companion image is padded and
resized. Channel statistics are not recalculated separately for each crop.

For every enabled paired whole image, the same recipe is evaluated on the
complete fixed-preprocessed image. Their geometry then branches. The compact companion
is top-left anchored in its own square and resized from that square to
`1024 x 1024`. If enabled, the high-resolution companion is independently
padded according to its own canvas configuration and saved without resizing.
Zero padding is added only on the right and bottom. The compact companion
therefore never inherits padding from that optional branch. A paired whole
image is context rather than a second crop, but it still receives a matched
annotation asset for auditing and optional whole-image training. The
original-size branch receives neither padding nor resizing.

The resized whole, crop size (`1024`) and stride (`512`) are all divisible by
16. The exporter validates enabled output geometry but does not run DINOv3 or
save feature tensors.

The crop grid uses `1024 x 1024` windows with stride 128. With
`edge_policy: regular_stride_pad`, all origins remain on the normal stride grid.
The final right/bottom window may extend beyond the image and its missing pixels
are filled with zero. Therefore `crop_window_xyxy` can exceed the native source
width or height.

For training, a clean negative candidate must contain at least 10% breast pixels.
A training crop containing an exported Mass annotation bypasses this threshold.
The fraction is measured from the boolean breast mask retained by fixed
preprocessing and divided by the full `1024 x 1024` crop area, so out-of-image
zero padding counts as non-breast. Validation and test do not apply this filter:
they retain every window in the complete stride grid. The measured value and
threshold, when the training filter is active, are available as
`foreground_fraction` and `min_foreground_fraction` in the flat metadata/statistics.

The crop keeps source-resolution preprocessed pixels, while the compact whole
view is scaled. Its per-image square size can vary, so use that row's recorded
padding and scale. If a model needs the crop location in compact whole-view
coordinates, read the full `samples_metadata.jsonl` row. For a source point
`(x, y)`, use:

```text
whole_x = (x + paired_whole_pad_left) * paired_whole_width  / paired_whole_canvas_width
whole_y = (y + paired_whole_pad_top)  * paired_whole_height / paired_whole_canvas_height
```

For the unscaled common canvas, use the canonical
`paired_whole_high_resolution_pad_*` and
`paired_whole_high_resolution_canvas_*` fields. `crop_locations.csv` also
provides `whole_high_resolution_crop_*` directly. Legacy `native`-named path
and coordinate aliases remain in metadata for older loaders, but new code
should use the high-resolution names.

For a source box `(x0, y0, x1, y1)`, every whole-variant annotation is produced
with the same transform as its pixels:

```text
x' = (x + pad_left) * scale_x
y' = (y + pad_top)  * scale_y
```

The original-size and top-left-anchored high-resolution variants have
`pad_left=pad_top=0` and `scale=1`, so their box coordinates remain unchanged.
The resized branch uses its per-image square size to calculate the scale; it
never uses the optional high-resolution branch's canvas scale.

Clamp an out-of-bounds crop window to the source width/height before drawing it
on the whole view. The unclamped part is precisely the zero-padded part of the
crop.

## Detection targets and balance

YOLO label rows are:

```text
0 center_x center_y width height
```

All four coordinates are normalized by the `1024 x 1024` crop dimensions and
class `0` means Mass. A negative crop has an existing, empty `.txt` file. COCO
annotations are the equivalent crop-coordinate boxes in
`mmdetection/annotations/instances_<split>.json`.

Default Research Dataset v1 keeps every eligible training positive crop regardless of
breast-mask fraction. It streams shuffled training sources/windows and saves an
eligible negative only while the running counts need another negative toward a
1:1 ratio. A clean negative has zero visibility of every Mass box and at least
10% breast occupancy; ambiguous partial-lesion windows are not training
negatives. Online balancing avoids a full candidate-planning pass, so the final
training ratio is approximate and depends on candidate order. It never discards
a positive to force the ratio and never fabricates negative duplicates.

Validation and test are inference grids rather than balanced training samples.
A loose 5% retained-breast-mask filter removes nearly blank negative windows.
Every eligible Mass-positive window is protected from that filter, even below
5% occupancy, so foreground filtering cannot remove a lesion-containing crop.

## Loader rules

1. Use a metadata row as the sample index.
2. Read `square_crops/<training_image>` as the crop input.
3. Read `square_crops/<paired_whole_image>` as its context input. This path is
   shared by every crop from the same source.
4. Read the YOLO label with the same split and basename, or use the matching
   COCO image ID and annotations.
5. Apply spatial detection augmentation to the crop and its boxes together.
   Do not apply the crop's box transform to the whole image. If desired, apply
   non-spatial intensity augmentation independently or consistently according
   to the model design.
6. Keep the exported split. Multiple crops from one source mammogram share the
   same whole-image content; re-splitting rows can leak a mammogram across
   train and validation.

There are no crop-specific whole-image aliases. Treat every metadata row as its
own `(crop, whole, target)` training sample, but expect many rows to reference
the same source-level whole path. A loader may cache decoded whole tensors by
`paired_whole_key` or `paired_whole_image`.

A useful model-side return structure is:

```python
{
    "crop": crop_tensor,                 # float32 [3, 1024, 1024]
    "whole": whole_tensor,               # float32 [3, 1024, 1024]
    "boxes": crop_boxes_xyxy,            # crop coordinates
    "labels": mass_class_ids,
    "source_image_id": source_image_id,
    "crop_window_xyxy": crop_window,
}
```
