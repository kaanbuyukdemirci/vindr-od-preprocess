# Paired Crop + Whole-Image Data Contract

This contract applies when `paired_whole_images.enabled: true`, including the
`simple-preset` export. The sample unit is **one crop and one whole-breast
context image**. Detection labels belong to the crop.

## Files and pairing key

For every exported crop, the exporter writes paths with the same split and
basename:

```text
square_crops/
  images/<split>/<basename>.png          # detector crop
  whole_images/<split>/<basename>.png    # paired whole-breast context
  labels/<split>/<basename>.txt          # YOLO label for the crop
  metadata/samples_metadata.jsonl
  metadata/samples_metadata_flat.csv
  mmdetection/annotations/instances_<split>.json
```

The identical basename is the pairing key. Do not reconstruct a pair by
parsing study or image IDs from the name when a metadata manifest is available.
The recommended loader index is `metadata/samples_metadata_flat.csv`:

- `training_image`: crop path, relative to `square_crops/`
- `paired_whole_image`: whole-image path, relative to `square_crops/`
- `split`, `file_name`, `source_image_id`, and `source_study_id`
- `crop_window_xyxy`: crop position in the fixed-preprocessed native image
- `num_export_boxes`: number of Mass boxes retained in the crop

`samples_metadata.jsonl` contains the same pairing plus full box coordinates,
preprocessing details, and pad/resize metadata. COCO image records also contain
`paired_whole_image_path`.

## Shapes and pixels in `simple-preset`

Both PNGs are 8-bit RGB and both decode to `1024 x 1024 x 3`. A PyTorch loader
normally converts each to float `[3, 1024, 1024]` and divides by 255.

Before square-window extraction, the fixed preprocessing corrects polarity,
crops and masks breast tissue, removes outside-breast artifacts/labels, and
mirrors orientation to the canonical side. The entire RGB recipe is also run on
that complete fixed-preprocessed breast before the square window is extracted:

- R: whole-image histogram equalization, then whole-image percentile `0-100`
- G: whole-image histogram equalization, then whole-image percentile `50-100`
- B: whole-image histogram equalization, then whole-image percentile `75-100`

Consequently, an in-bounds crop pixel is the exact corresponding pixel from the
fully channel-processed whole breast before the companion image is padded and
resized. Channel statistics are not recalculated separately for each crop.

For the paired whole image, the same recipe is evaluated on the complete
fixed-preprocessed image. It is then padded with zero to a square canvas at the
left/top anchor and resized to `1024 x 1024`. Padding happens before resizing,
so aspect ratio is preserved. The paired whole image is context; it is not a
second crop and it does not have a separate detection label.

The crop grid uses `1024 x 1024` windows with stride 512. With
`edge_policy: regular_stride_pad`, all origins remain on the normal stride grid.
The final right/bottom window may extend beyond the image and its missing pixels
are filled with zero. Therefore `crop_window_xyxy` can exceed the native source
width or height.

For training, a clean negative candidate must contain at least 80% breast pixels.
A training crop containing an exported Mass annotation bypasses this threshold.
The fraction is measured from the boolean breast mask retained by fixed
preprocessing and divided by the full `1024 x 1024` crop area, so out-of-image
zero padding counts as non-breast. Validation and test do not apply this filter:
they retain every window in the complete stride grid. The measured value and
threshold, when the training filter is active, are available as
`foreground_fraction` and `min_foreground_fraction` in the flat metadata/statistics.

The two inputs have related anatomy but are not pixel-aligned tensors: the crop
keeps native preprocessed pixels, while the whole view is scaled. If a model
needs the crop location in whole-view coordinates, read the full
`samples_metadata.jsonl` row. For a source point `(x, y)`, use:

```text
whole_x = (x + paired_whole_pad_left) * paired_whole_width  / paired_whole_canvas_width
whole_y = (y + paired_whole_pad_top)  * paired_whole_height / paired_whole_canvas_height
```

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

The simple preset keeps every eligible training positive crop regardless of
breast-mask fraction. It streams shuffled training sources/windows and saves an
eligible negative only while the running counts need another negative toward a
1:1 ratio. A clean negative has zero visibility of every Mass box and at least
80% breast occupancy; ambiguous partial-lesion windows are not training
negatives. Online balancing avoids a full candidate-planning pass, so the final
training ratio is approximate and depends on candidate order. It never discards
a positive to force the ratio and never fabricates negative duplicates.

Validation and test are inference grids rather than balanced training samples:
every generated `1024 x 1024` stride-grid crop is saved, including empty,
low-breast-occupancy, and partial-lesion windows.

## Loader rules

1. Use a metadata row as the sample index.
2. Read `square_crops/<training_image>` as the crop input.
3. Read `square_crops/<paired_whole_image>` as its context input.
4. Read the YOLO label with the same split and basename, or use the matching
   COCO image ID and annotations.
5. Apply spatial detection augmentation to the crop and its boxes together.
   Do not apply the crop's box transform to the whole image. If desired, apply
   non-spatial intensity augmentation independently or consistently according
   to the model design.
6. Keep the exported split. Multiple crops from one source mammogram share the
   same whole-image content; re-splitting rows can leak a mammogram across
   train and validation.

By default, repeated whole-image companions from one source mammogram are hard
links: they have separate filenames but share file data on disk. This is only a
storage optimization and is transparent to a loader. Treat every metadata row
as its own `(crop, whole, target)` training sample.

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
