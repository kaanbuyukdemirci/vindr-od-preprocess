# GUI parameter reference

This page documents the controls in the VinDr-Mammo preprocessing inspector. The GUI is a visual inspection and export-planning tool; augmentation is intentionally out of scope.

## Launch

From the repository root:

```bash
pip install -e .
vindr-mammo-gui --config config/export_config.yaml
```

Alternative direct Streamlit command:

```bash
streamlit run inspect_preprocessing_app.py -- --config config/export_config.yaml
```

## Performance Controls

| GUI label | Config key | Meaning |
|---|---|---|
| Manual preview refresh | GUI only | When enabled, changing parameters does not immediately reload and preprocess DICOM pixels. Click **Render / refresh preview** after a group of edits. This is recommended for large mammograms. |
| Render / refresh preview | GUI only | Runs the expensive image read, fixed preprocessing, crop selection, RGB channel processing, and rendering for the current controls. |

## Configuration

| GUI label | Config key | Meaning |
|---|---|---|
| Config YAML | CLI/query setting | YAML file used to initialize paths, preprocessing, crop, export, and visualization defaults. |
| VinDr data root | `paths.data_root` | Folder containing `metadata.csv`, `breast-level_annotations.csv`, `finding_annotations.csv`, and `images/<study_id>/<image_id>.dicom`. |

## Mode

| GUI label | Meaning |
|---|---|
| Single image | Inspect one source mammogram and one selected crop. |
| Vendor / image comparison | Compare several images/crops side by side using the same preprocessing pipeline. |
| Dataset visualizations | Read an already exported dataset and generate/display summary plots and tables. |
| Saved dataset viewer | Browse saved `square_crops` PNGs and YOLO labels without reading original DICOMs. |
| Manifest comparison / load settings | Compare export manifests and optionally load a previous config snapshot into the GUI. |

## Fixed Preprocessing Before Crops

These controls run before square-crop selection and before the RGB channel pipeline.

| GUI label | Config key | Meaning |
|---|---|---|
| Invert MONOCHROME1 to black background | `preprocess.invert_to_black_background` | Inverts DICOMs tagged `MONOCHROME1` so breast tissue has a consistent bright-on-dark convention. |
| Crop to breast foreground | `preprocess.crop_breast` | Crops to the breast-mask bounding rectangle and updates boxes. Recommended before sliding-crop export. |
| Mask outside breast foreground | `preprocess.mask_outside_breast` | Sets non-breast pixels to zero to suppress labels, borders, and background artifacts. |
| Mirror right-entering breasts to left-entering | `preprocess.mirror_right_to_left` | Horizontally flips images whose breast foreground is mostly on the right; boxes are mirrored. |
| Breast crop padding | `preprocess.crop_padding`, `preprocess.crop_padding_fraction` | Chooses fixed pixel padding or fractional padding around the breast crop. |
| Minimum crop padding, pixels | `preprocess.minimum_padding_px` | Lower bound for fractional crop padding. |
| Maximum crop padding, pixels | `preprocess.maximum_padding_px` | Upper bound for fractional crop padding. |
| Breast crop threshold | `preprocess.crop_threshold` | Auto uses Otsu/percentile thresholding. Manual supplies the foreground threshold directly. |
| Minimum breast component area fraction | `preprocess.min_component_area_fraction` | Ignores connected components smaller than this fraction of the image. |
| Mask method | `preprocess.breast_mask_method` | `otsu_largest_connected_component` uses Otsu plus connected components; percentile mode uses the robust percentile threshold. |
| Open kernel | `preprocess.breast_mask_open_kernel` | Morphological opening kernel for removing small foreground specks. |
| Close kernel | `preprocess.breast_mask_close_kernel` | Morphological closing kernel for filling gaps in the breast mask. |
| Fill breast-mask holes | `preprocess.breast_mask_fill_holes` | Fills holes inside the detected breast mask. |
| Keep largest connected component | `preprocess.breast_mask_keep_largest_component` | Keeps the dominant breast component and removes smaller foreground components. |
| Minimum box visibility after breast crop | `preprocess.min_box_visibility_after_crop` | Drops boxes if too little of the original box remains visible after the breast crop. |

## Display And Debug

| GUI label | Config key | Meaning |
|---|---|---|
| Show mass annotations | GUI only | Draws mass boxes on image panels. |
| Grayscale display window percentiles | GUI only | Changes only display contrast, not exported pixel values. |
| Visible RGB channels | GUI only | Selects which processed channels are visible in the composite preview. Exported data still has all channels. |
| Show individual processed channels | GUI only | Shows R, G, and B as separate grayscale panels for debugging. |

## Crop Preview And Shared Crop Geometry

| GUI label | Config key | Meaning |
|---|---|---|
| Crop size n | `square_crops.crop_size` | Square crop size in pixels. |
| Deterministic stride | `square_crops.stride` | Sliding-window stride in pixels. |
| PREVIEW ONLY, crop proposal mode | GUI preview, export uses split controls | Selects deterministic sliding, stochastic random, or bbox-safe breast-biased random crop proposals for browsing. |
| PREVIEW ONLY, show only crops with visible mass | GUI only | Filters preview candidates to positive crops. Export balance is controlled in the export panel. |
| PREVIEW ONLY, positive crop threshold | GUI only | Visibility fraction required for the GUI to call a crop positive. |
| Display partial boxes after clipping | GUI only | Draws partially visible boxes after clipping to crop boundaries. |
| Minimum box visibility to draw/keep | `crop_annotation_policy.min_box_visibility` when exported from GUI | Minimum visible fraction for partial boxes. |
| Random crops to preview | GUI only | Number of stochastic preview candidates. |
| PREVIEW ONLY, random positive request probability | GUI only | Probability that the preview sampler requests a positive random crop. |
| Mass-center random shift fraction | `square_crops.center_shift_fraction` | How far random positive crops can shift around a target box center. |
| Random preview seed | `square_crops.seed` | Random seed for preview sampling and export defaults. |
| Annotation boundary exclusion fraction | `square_crops.bbox_safe_boundary_margin_fraction` | In bbox-safe mode, visible boxes must stay away from crop edges by this fraction. |
| BBox-safe random shift fraction | `square_crops.bbox_safe_random_shift_fraction` | Candidate crop-center jitter in bbox-safe mode. |
| Candidate windows per crop | `square_crops.bbox_safe_candidate_count` | Number of candidate windows sampled for each bbox-safe crop. |
| Randomly choose among top K candidates | `square_crops.bbox_safe_top_k` | Adds randomness after ranking valid bbox-safe candidates. |
| Breast foreground bias strength | `square_crops.bbox_safe_breast_bias_strength` | Prefers candidate crops with more breast foreground. |
| Left/chest-wall alignment bias strength | `square_crops.bbox_safe_left_bias_strength` | Gently prefers left-aligned crops after orientation normalization. |
| X-projection peak bias strength | `square_crops.bbox_safe_projection_bias_strength` | Prefers crops containing strong horizontal foreground support. |
| Require crop to contain breast foreground | `square_crops.deterministic_require_foreground` | Rejects windows with too little foreground. |
| Minimum foreground fraction in crop | `square_crops.deterministic_min_foreground_fraction` | Minimum breast-mask fraction for foreground-filtered crops. |
| Foreground threshold for square crops | `square_crops.deterministic_foreground_threshold` | Auto or manual threshold for crop foreground filtering. |
| Show foreground mask preview for selected crop | GUI only | Displays the foreground pixels used by the crop filter. |

## Opposite-Breast Source Alignment

Used only when an RGB channel source is `contralateral_same_view_crop`.

| GUI label | Config key | Meaning |
|---|---|---|
| Enable opposite-breast vertical alignment | `image_export.contralateral_source_alignment.enabled` | Vertically shifts the opposite breast before extracting the same crop window. |
| Alignment method | `image_export.contralateral_source_alignment.method` | Chooses nipple-y, row projection, hybrid profile, boundary profile, centroid, intensity projection, or none. |
| Fallback method for hybrid | `image_export.contralateral_source_alignment.fallback_method` | Used when the selected alignment method cannot estimate a reliable shift. |
| Maximum vertical shift fraction | `image_export.contralateral_source_alignment.max_shift_fraction` | Maximum allowed vertical shift as a fraction of image height. |
| Minimum profile overlap fraction | `image_export.contralateral_source_alignment.min_profile_overlap_fraction` | Minimum shared support for profile-based alignment. |

## RGB Preprocessing Pipeline

| GUI label | Config key | Meaning |
|---|---|---|
| 3-channel recipe | `image_export.rgb_scheme` or `image_export.custom_channel_pipeline` | Loads presets such as raw+CLAHE+detail, raw replicated, masked raw, TopHat, or a denoise ablation. |
| Source | `image_export.custom_channel_pipeline.<R/G/B>.source` | Uses the current crop or the aligned opposite-breast crop. |
| Number of steps | `image_export.custom_channel_pipeline.<R/G/B>.steps` | Number of ordered preprocessing operations in that channel. |
| Step | `image_export.custom_channel_pipeline.<R/G/B>.steps[].op` | Operation name. Parameters below depend on the selected operation. |

### Channel Operations

| Operation | Main parameters | Meaning |
|---|---|---|
| `percentile_normalize` | `percentiles` | Clips to a percentile window and scales to `[0, 1]`. |
| `percentile_clip_only` | `percentiles` | Clips outliers without rescaling at that step. |
| `zscore_clip` | `z_limit` | Z-score normalizes, clips, then maps to `[0, 1]`. |
| `aggressive_upper_percentile_normalize` | `percentiles` | Legacy alias for high-percentile normalization. |
| `standardize_to_target` | `target_mean`, `target_std`, `stat_percentiles`, `clip_output` | Dynamically affine-standardizes the channel. |
| `mask_outside_breast`, `artifact_cleanup` | `outside_value` | Sets non-breast pixels to a constant. |
| `hist_equalize` | none | Global histogram equalization. |
| `clahe` | `clip_limit`, `tile_grid_size` | Local contrast enhancement. |
| `gaussian_blur` | `ksize`, `sigma` | Gaussian smoothing. |
| `median_blur` | `ksize` | Median denoising. |
| `bilateral_filter` | `diameter`, `sigma_color`, `sigma_space` | Edge-preserving denoising. |
| `wiener_filter` | `ksize`, `noise` | Adaptive denoising using local statistics. |
| `local_detail` | `sigma`, `percentiles` | CLAHE/detail-style residual channel: image minus a smooth version. |
| `sharpen` | `amount` | Simple sharpening. |
| `unsharp_mask` | `amount`, `sigma` | Adds high-frequency residual back to the image. |
| `sobel_gradient` | `ksize`, `percentiles` | Gradient-magnitude edge channel. |
| `laplacian` | `ksize`, `percentiles` | Second-derivative edge response. |
| `white_tophat` | `kernel_shape`, `kernel_size`, `percentiles` | Highlights small bright structures such as calcification-like details. |
| `blackhat` | `kernel_shape`, `kernel_size`, `percentiles` | Highlights small dark structures. |
| `morphological_open` | `kernel_shape`, `kernel_size` | Removes small bright structures. |
| `morphological_close` | `kernel_shape`, `kernel_size` | Fills small dark gaps. |
| `pectoral_suppression` | `side`, `width_fraction`, `height_fraction`, `fill_value` | Optional conservative triangular suppression; off unless explicitly selected. |
| `gamma` | `gamma` | Nonlinear brightness/contrast remapping. |
| `log` | `gain` | Log contrast remapping. |
| `invert` | none | Inverts the normalized channel. |

## Export Current Preprocessing YAML

| GUI label | Meaning |
|---|---|
| Download current preprocessing YAML | Saves fixed preprocessing, crop preview settings, display settings, RGB channel pipeline, and an `export_config_patch`. |
| Show YAML preview | Displays the YAML in the sidebar before downloading. |

## Export Dataset From GUI

| GUI label | Config key | Meaning |
|---|---|---|
| Strict replay loaded config, ignore GUI control edits | GUI only | Replays a loaded manifest/config exactly except for output path and clean-output choices. |
| Export parent folder | `paths.output_root` parent | Parent directory for the new export. |
| Dataset folder name | `paths.output_root` name | Folder name for the new export. |
| Delete output folder before export | `export.clean_output_root` | Removes the target export folder before writing. Use carefully. |
| Sliding crop export (square_crops) | `export.save_square_crops` | Writes overlapping fixed-size crop images and labels. |
| Whole-image export (baseline_uncropped) | `export.save_baseline_uncropped` | Writes one preprocessed image per mammogram without final square cropping. |
| Vendor mode / selected vendors | `vendor_filter` | Exports all vendors or only selected vendors. |
| Train/val/test crop mode | `square_crops.<split>_crop_mode` | Per-split deterministic, random, or bbox-safe random export mode. |
| Train/val/test mass/empty selection | `square_crops.<split>_deterministic_selection_mode` | Exports mass only, all windows, finding-image windows, or all mass plus sampled non-mass. |
| Train/val/test target mass-positive crop ratio | `square_crops.<split>_positive_fraction` and related fields | Target positive fraction for deterministic sampling or random crop requests. |
| BBox-safe export parameters | `square_crops.bbox_safe_*` | Reuses the crop-preview bbox-safe controls for export. |
| Show simple timing breakdown during export | `runtime.simple_profiler_enabled` | Enables coarse timing buckets during export. |
| Profiler GUI update frequency | `runtime.simple_profiler_emit_every` | Reduces UI overhead by updating timing every N progress events. |

