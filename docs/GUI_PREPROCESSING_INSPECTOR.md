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
- whether to display partially clipped boxes,
- minimum box visibility for keeping/drawing partial boxes.

The positive-crop slider is independent of the final export settings. It is meant for visual exploration.

## RGB preprocessing pipeline

Each channel, `R`, `G`, and `B`, has its own pipeline. For each channel, choose the number of steps and then select ordered operations.

Available operations:

- `percentile_normalize`: clip to percentile range and scale to `[0, 1]`,
- `percentile_clip_only`: clip outliers without scaling at that step,
- `zscore_clip`: standardize, clip by z-score, then scale,
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

This GUI does not automatically modify `export_config.yaml`. Once you find settings you like, copy the corresponding logic into the export code/config or ask ChatGPT to update the exporter.

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

## Histogram normalization

The metadata/statistics expander shows pixel intensity histograms for the full grayscale image, selected crop, and processed RGB channels. For easier comparison, the GUI normalizes each histogram series independently:

- the x-axis is min-max normalized to `[0, 1]` per image or channel;
- the y-axis is relative frequency, not raw pixel count.

This means a full mammogram and a 1024 x 1024 crop can be visually compared without the full image dominating only because it has more pixels.

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

The optional **Show individual processed channels** checkbox displays R, G, and B as separate grayscale images under the main three-panel view.

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

The old pixel-intensity histogram plot was removed because it was not very useful for deciding whether different images or vendors can be trained in one network. Compare mode now reports numeric similarity between selected slots. It compares summary features such as mean, standard deviation, percentiles, IQR, and entropy for the grayscale crop and processed R/G/B channels. It also reports Jensen-Shannon distance and an approximate 1-D Wasserstein distance on compact normalized intensity summaries.

Lower values mean the selected images are more similar statistically. These metrics are not a clinical quality score and do not compare pixels spatially. They are meant as a practical data-consistency check across vendors and preprocessing choices.

## v27 additions

- Individual processed R/G/B channel panels now show mass annotation boxes when `Show mass annotations` is enabled.
- Compare mode now includes a `Statistics comparison across selected slots` section.
- The pixel-intensity histogram plot was removed from the metadata panel.
