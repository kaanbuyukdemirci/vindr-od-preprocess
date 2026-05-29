# VinDr-Mammo preprocessing inspector GUI

This project includes a local Streamlit GUI for interactively checking how different image preprocessing choices affect mass visibility before you export another YOLO/MMDetection dataset.

## Why this GUI exists

Mass detection performance was sensitive to the training crop distribution and to image preprocessing. Instead of exporting a full dataset every time, use this GUI to inspect:

- source images by split: `train`, `val`, `test`, or all,
- positive-only images or all images,
- vendor/device subsets,
- deterministic sliding-window crops,
- positive-crop definitions based on visible mass fraction,
- mass boxes before and after crop clipping,
- custom RGB preprocessing pipelines,
- intensity statistics and compare-mode statistical similarity metrics.

The GUI reads the original VinDr-Mammo DICOMs and the CSV annotations. It does not train a model and does not export a dataset.

## Run from the repository

From the project root:

```bash
pip install -e .
streamlit run inspect_preprocessing_app.py -- --config config/export_config.yaml
```

or use the package command:

```bash
vindr-mammo-gui --config config/export_config.yaml
```

A browser window should open automatically. If it does not, copy the local Streamlit URL from the terminal.

## What the GUI shows

For each selected image and crop, the GUI shows three panels:

1. **Original black/white image**: the full mammogram after fixed geometric preprocessing from the YAML, including MONOCHROME1 inversion, breast crop, and right-to-left mirroring if enabled. The selected square crop window is drawn on top.
2. **Original crop**: the selected `n x n` crop shown as grayscale.
3. **Preprocessed RGB crop**: the result of the interactive channel-wise preprocessing pipeline.

Mass annotations can be shown or hidden with the sidebar checkbox.

## Filters

The image selector supports:

- split: `all`, `train`, `val`, `test`,
- image positivity: positive images only or all images,
- vendor filtering: all vendors or selected vendors.

The split logic matches the export code: official VinDr `test` remains test, and official VinDr `training` is split into train/val by study ID according to `splits.val_fraction_from_training` and `splits.seed`.

## Fixed preprocessing controls

The sidebar has a **Fixed preprocessing before crops** section. These are the preprocessing steps used before square-crop selection and before the experimental RGB channel pipeline:

- `Invert MONOCHROME1 to black background`: only images tagged `MONOCHROME1` are inverted; `MONOCHROME2` images are left unchanged.
- `Crop to breast foreground`: crops away pure background and updates boxes.
- `Mirror right-entering breasts to left-entering`: if the detected breast foreground is mostly on the right side, the image is flipped horizontally and boxes are mirrored.
- `Breast crop padding`, `Breast crop threshold`, and `Minimum breast component area fraction`: control the breast foreground crop.

The metadata panel reports what was actually applied for the currently loaded image, including `InvertedMonochrome1`, `crop_box_xyxy`, and `mirrored`.

## Crop controls

The crop controls include:

- crop size `n`, default from `square_crops.crop_size`, usually `1024`,
- stride, default from `square_crops.stride`, usually `512`,
- whether to show all crops or only crops with visible mass,
- a slider defining positive crop visibility threshold,
- whether to display partially clipped boxes, enabled by default for debugging,
- minimum box visibility for keeping/drawing partial boxes.

The positive-crop slider is independent of the final export settings. It is meant for visual exploration.

## RGB preprocessing pipeline

Each channel, `R`, `G`, and `B`, has its own pipeline. For each channel, choose the number of steps and then select ordered operations.

Available operations:

- `percentile_normalize`: clip to percentile range and scale to `[0, 1]`,
- `percentile_clip_only`: clip outliers without scaling at that step,
- `zscore_clip`: standardize, clip by z-score, then scale,
- `standardize_to_target`: dynamically compute `a` and `b` in `a*x + b` so the channel reaches a target mean and standard deviation,
- `hist_equalize`: simple histogram equalization,
- `clahe`: contrast-limited adaptive histogram equalization,
- `gaussian_blur`: smoothing,
- `median_blur`: median denoising,
- `sharpen`: simple sharpening kernel,
- `unsharp_mask`: blur-subtract sharpening,
- `sobel_gradient`: edge/texture gradient magnitude,
- `laplacian`: second-derivative edge response,
- `gamma`: gamma correction,
- `log`: logarithmic contrast,
- `invert`: invert the normalized channel.

The default interactive pipeline is close to the current export representation:

```text
R = percentile-normalized intensity
G = percentile-normalized intensity + histogram equalization
B = percentile-normalized intensity + Sobel gradient
```

Use **Export current preprocessing YAML** in the sidebar to download the current fixed preprocessing settings, crop controls, visible-channel settings, and per-channel RGB pipeline. The downloaded file includes an `export_config_patch` section that can be copied into `export_config.yaml`. The exporter also supports `image_export.rgb_scheme: custom_channel_pipeline`, so GUI-exported R/G/B pipelines can be used for dataset generation.


## Export current preprocessing YAML

The sidebar includes **Export current preprocessing YAML**. The download contains:

- `fixed_preprocessing_before_crops`: MONOCHROME1 inversion, breast crop, mirroring, crop padding, and threshold settings,
- `crop_preview_settings`: deterministic/stochastic crop settings, positivity threshold, and foreground-ratio filter settings,
- `display_debug_settings`: currently visible RGB channels and channel-panel display state,
- `rgb_channel_pipeline`: the exact ordered operations selected for R, G, and B,
- `export_config_patch`: a compact block intended for copy-paste into `export_config.yaml`.

If you paste the custom RGB pipeline into the main export config, use:

```yaml
image_export:
  rgb_scheme: custom_channel_pipeline
  custom_channel_pipeline:
    R:
      - op: percentile_normalize
        params:
          percentiles: [1.0, 99.0]
    G:
      - op: percentile_normalize
        params:
          percentiles: [1.0, 99.0]
      - op: hist_equalize
        params: {}
    B:
      - op: percentile_normalize
        params:
          percentiles: [1.0, 99.0]
      - op: sobel_gradient
        params:
          ksize: 3
          percentiles: [1.0, 99.0]
```

## Metadata and statistics

The statistics expander shows:

- image shape and dtype,
- min, max, mean, standard deviation,
- 1st, 50th, and 99th percentiles,
- DICOM and metadata fields when available,
- fixed preprocessing info,
- selected RGB pipeline JSON,
- summary statistics for the full grayscale image, grayscale crop, and R/G/B channels.

## Comparison mode

Use **Vendor / image comparison** mode to compare multiple images side by side. Each slot can select a different split, positivity filter, vendor, image index, and crop index. All slots share the same crop controls and RGB pipeline, making vendor-level visual differences easier to inspect.

## Notes

- The GUI is intended for qualitative inspection, not final training.
- Very large DICOMs can still take a few seconds to load. Recently viewed images are cached by Streamlit.
- If you change the YAML preprocessing settings, reload the app to ensure the cached dataset uses the new settings.

## Vendor selector notes

The GUI vendor selector is populated from `metadata.csv`. It now checks multiple common column names, including `Manufacturer`, `manufacturer`, `ManufacturerModelName`, `Manufacturer's Model Name`, and normalized variants such as `manufacturer_model_name`. It also falls back to common image identifier columns such as `image_id`, `SOPInstanceUID`, and filename/path stems.

If no manufacturer/model metadata can be matched to the current image records, the selector will show `Unknown` and the Vendor counts expander will make that visible. This means the GUI is working, but the metadata file does not expose usable vendor columns or image identifiers for matching.

## Channel visibility controls

The sidebar includes **Visible RGB channels**. This only changes what the GUI displays. It does not change the underlying channel preprocessing pipeline.

Examples:

- select only `R` to inspect the normal intensity channel,
- select only `G` to inspect the equalized/contrast channel,
- select only `B` to inspect the edge or gradient channel,
- select all channels to see the composite RGB crop.

By default, all channels are visible and **Show individual processed channels** is enabled, so R, G, and B appear as separate grayscale images under the main three-panel view. This is only a GUI display choice and does not change exported data.

## Deterministic, stochastic, and foreground-ratio crop preview

The GUI crop controls now support two crop proposal modes:

- **deterministic sliding**: normal sliding-window crops using `crop_size` and `stride`,
- **stochastic random**: random crops, optionally biased toward masses through the positive-fraction setting.

The **Foreground-ratio crop filter** can be enabled in the sidebar. It computes a simple foreground/breast mask inside each candidate crop and rejects the crop if the foreground fraction is below the selected threshold. This is useful when you turn off `preprocess.crop_breast` and want square crops to remove pure detector background windows instead.

For example:

```yaml
preprocess:
  crop_breast: false

square_crops:
  deterministic_require_foreground: true
  deterministic_min_foreground_fraction: 0.05
  deterministic_foreground_threshold: null
```

`deterministic_foreground_threshold: null` uses the automatic threshold. Manual thresholding is available in the GUI for debugging.

## Compare-mode statistical similarity

The old pixel-intensity histogram plot was removed because it was not very useful for deciding whether different images or vendors can be trained in one network. Compare mode now reports numeric similarity between selected slots. The main pairwise table now focuses on the final processed R/G/B outputs, because those are what the model receives after percentile normalization, histogram equalization, contralateral-channel substitution, and standardization.

The raw `crop_*` statistics are still shown in a separate expander, but they describe the grayscale source crop before the R/G/B preprocessing pipeline. They can remain different even when the final R/G/B means and standard deviations are matched.

Lower final-output distance values mean the selected examples are more similar statistically as model inputs. These metrics are not a clinical quality score and do not compare pixels spatially. They are meant as a practical data-consistency check across vendors and preprocessing choices.

## v27 additions

- Individual processed R/G/B channel panels now show mass annotation boxes when `Show mass annotations` is enabled.
- Compare mode now includes a `Statistics comparison across selected slots` section.
- The pixel-intensity histogram plot was removed from the metadata panel.

## v28 additions

- Added a sidebar download button to export the current GUI preprocessing/crop/channel-pipeline settings as YAML.
- Added exporter support for `image_export.rgb_scheme: custom_channel_pipeline`, so GUI-exported RGB pipelines can be used when generating new datasets.

## v29 additions

- Single-image and comparison-slot index controls are now placed beside their corresponding image/crop status text instead of appearing as a disconnected control below the selector.
- Added `standardize_to_target` as a per-channel preprocessing step. It estimates the current channel mean and standard deviation, optionally using only a percentile-trimmed pixel range, then applies `y = a*x + b` with `a = target_std / current_std` and `b = target_mean - a * current_mean`.
- The same `standardize_to_target` operation is supported by the exporter when using `image_export.rgb_scheme: custom_channel_pipeline`.


## Contralateral same-view channel source

Version 0.30 adds a channel source called `contralateral_same_view_crop`. This is not a normal filter. It changes where a channel starts from:

- `current_crop`: the selected crop from the current image,
- `contralateral_same_view_crop`: the same `[xmin, ymin, xmax, ymax]` crop window from the opposite breast in the same study and the same view position.

The default GUI/export pipeline is now:

```yaml
image_export:
  rgb_scheme: custom_channel_pipeline
  custom_channel_pipeline:
    R:
      source: current_crop
      steps:
        - op: percentile_normalize
          params: {percentiles: [1.0, 99.0]}
        - op: hist_equalize
          params: {}
        - op: standardize_to_target
          params: {target_mean: 0.5, target_std: 0.2, stat_percentiles: [1.0, 99.0], clip_output: true}
    G:
      source: current_crop
      steps:
        - op: percentile_normalize
          params: {percentiles: [1.0, 99.0]}
        - op: hist_equalize
          params: {}
        - op: percentile_normalize
          params: {percentiles: [70.0, 100.0]}
        - op: hist_equalize
          params: {}
        - op: standardize_to_target
          params: {target_mean: 0.5, target_std: 0.2, stat_percentiles: [1.0, 99.0], clip_output: true}
    B:
      source: contralateral_same_view_crop
      steps:
        - op: percentile_normalize
          params: {percentiles: [1.0, 99.0]}
        - op: hist_equalize
          params: {}
        - op: standardize_to_target
          params: {target_mean: 0.5, target_std: 0.2, stat_percentiles: [1.0, 99.0], clip_output: true}
```

The pairing key is `study_id + view_position + opposite laterality`. If the opposite image is missing, the code falls back to the current crop and records the fallback in metadata so the GUI/export does not crash.

The old `aggressive_upper_percentile_normalize` operation is no longer used in the default pipeline because it is equivalent to a normal `percentile_normalize` step with `percentiles: [70.0, 100.0]`. It is still supported as a legacy alias so older GUI-exported YAML files keep working.

## v32 additions

- Default GUI display now starts with all R/G/B channels visible, individual channel panels enabled, and partial box display enabled.
- Compare-mode statistics now separates raw source-crop statistics from final processed R/G/B statistics. The main pairwise similarity metrics and combined distance focus on the final R/G/B model input, while raw crop distances are retained only as diagnostic columns.
