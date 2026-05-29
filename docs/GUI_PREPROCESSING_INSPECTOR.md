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
- intensity statistics and histograms.

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
- histograms for the full grayscale image, grayscale crop, and R/G/B channels.

## Comparison mode

Use **Vendor / image comparison** mode to compare multiple images side by side. Each slot can select a different split, positivity filter, vendor, image index, and crop index. All slots share the same crop controls and RGB pipeline, making vendor-level visual differences easier to inspect.

## Notes

- The GUI is intended for qualitative inspection, not final training.
- Very large DICOMs can still take a few seconds to load. Recently viewed images are cached by Streamlit.
- If you change the YAML preprocessing settings, reload the app to ensure the cached dataset uses the new settings.
