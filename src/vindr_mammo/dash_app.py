from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate
from PIL import Image

from .dataset import VindrMammoDataset
from .export import load_export_config, make_train_val_test_split, normalize_split_strategy_kwargs
from .export_queue import (
    DuplicateOutputRootError,
    ExportQueueManager,
    InvalidJobStateError,
    QueueJobNotFoundError,
)
from .presets import (
    PAPER_22_IMPROVED_PRESET_KEY,
    PAPER_69_PRESET_KEY,
    SIMPLE_PRESET_KEY,
    STUDY_PRESETS,
    apply_study_preset,
)
from .storage import estimate_export_space, format_bytes, get_disk_space

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
for _streamlit_logger_name in [
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.caching.cache_resource_api",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
]:
    logging.getLogger(_streamlit_logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(_streamlit_logger_name).disabled = True

from .gui_app import (
    LITERATURE_PIPELINE_PRESETS,
    OP_NAMES,
    _available_vendors,
    _build_enriched_record_table,
    _build_gui_export_config,
    _channel_steps,
    _compact_metadata,
    _current_preprocessing_yaml_payload,
    _default_comparison_vendors,
    _draw_boxes,
    _draw_rect,
    _find_contralateral_record_index,
    _histogram_figure,
    _load_config_cached,
    _load_saved_dataset_viewer_index,
    _load_saved_viewer_image,
    _load_split_records,
    _load_yolo_boxes_for_saved_image,
    _make_yaml_safe,
    _mask_rgb_channels,
    _pipeline_channel_payload,
    _pipeline_uses_contralateral,
    _prepare_saved_viewer_display_image,
    _prepare_sample,
    _saved_viewer_caption,
    _source_crops_from_result,
    _source_full_images_from_result,
    _stats_table,
    _streamlit_json_safe,
    _to_uint8_percentile,
    _vendor_from_summary,
    apply_channel_pipeline,
)


APP_TITLE = "VinDr-Mammo preprocessing studio"
SPLITS = ["all", "train", "val", "test"]
CHANNELS = ["R", "G", "B"]
PIPELINE_STAGES = ["common_start", "R", "G", "B", "common_end"]
PIPELINE_STAGE_LABELS = {
    "common_start": "Common start",
    "R": "R channel",
    "G": "G channel",
    "B": "B channel",
    "common_end": "Common end",
}
PIPELINE_STEP_COUNT = 4


PARAM_HELP: dict[str, dict[str, str]] = {
    "preview_max_side": {
        "title": "Preview Max Side",
        "body": "Limits the image size used for interactive preview. This is for speed only; export still uses the export YAML settings.",
        "example": "Use 1536 for fast tuning, 2048 for a sharper preview, or full only when you need exact native-resolution behavior.",
    },
    "view_geometry": {
        "title": "View Geometry",
        "body": "Choose whether the preview shows a selected square crop or the whole preprocessed image resized for inspection.",
        "example": "Use whole image when designing global preprocessing. Use square crop when tuning detector crop/export behavior.",
    },
    "preview_contralateral": {
        "title": "Preview Opposite Breast Channels",
        "body": "Controls only interactive preview. Turning it off makes contralateral channel sources use the current image instead, avoiding a second DICOM read and alignment.",
        "example": "Keep off while tuning basic contrast. Turn on only when checking the actual opposite-breast channel behavior.",
    },
    "common_channel_steps": {
        "title": "Common Channel Steps",
        "body": "Operations listed here are prepended to R, G, and B before each channel's own steps. This is the fastest way to apply the same normalization, masking, denoising, or contrast operation everywhere.",
        "example": "Example YAML: - op: percentile_normalize\\n  params: {percentiles: [0.5, 99.5]}",
    },
    "config_path": {
        "title": "Config YAML",
        "body": "Initializes paths, split settings, preprocessing defaults, crop defaults, export settings, and visualization paths.",
        "example": "Example: config/export_config.yaml loads the repository default export plan.",
    },
    "data_root": {
        "title": "VinDr data root",
        "body": "Folder that contains the VinDr metadata CSV files and the DICOM image tree. The app reads original images only in preview and comparison modes.",
        "example": "Expected files include metadata.csv, breast-level_annotations.csv, finding_annotations.csv, and images/<study_id>/<image_id>.dicom.",
    },
    "invert_to_black_background": {
        "title": "Invert MONOCHROME1",
        "body": "Some DICOMs store bright tissue as low values. This option inverts only MONOCHROME1 files so tissue is bright on a dark background.",
        "example": "Leave on for mixed DICOM collections. MONOCHROME2 images are not changed.",
    },
    "crop_breast": {
        "title": "Crop To Breast Foreground",
        "body": "Finds the breast mask, crops away pure background, and shifts boxes into the new coordinate system.",
        "example": "Recommended for square crops because a 1024 crop covers more useful breast tissue after the background is removed.",
    },
    "mask_outside_breast": {
        "title": "Mask Outside Breast",
        "body": "Sets pixels outside the detected breast mask to zero. This suppresses labels, borders, markers, and scanner artifacts.",
        "example": "Use with crop_breast when exported crops contain too much burned-in annotation or edge noise.",
    },
    "mirror_right_to_left": {
        "title": "Mirror Right-Entering Breasts",
        "body": "If the breast foreground is mostly on the right side, the image is flipped horizontally and boxes are mirrored.",
        "example": "After enabling this, the chest wall is usually on the left, making crop bias and model input orientation more consistent.",
    },
    "crop_padding_fraction": {
        "title": "Breast Crop Padding Fraction",
        "body": "Adds padding around the detected breast crop as a fraction of the detected breast extent. Minimum and maximum pixel bounds still apply.",
        "example": "0.03 on a 3000 px breast extent requests about 90 px of padding, then clamps to the min/max padding values.",
    },
    "crop_padding": {
        "title": "Fixed Breast Crop Padding",
        "body": "Adds the same number of pixels on every side of the detected breast crop. Use this when you want exact padding rather than scale-relative padding.",
        "example": "32 keeps a small border around the breast; 0 crops tightly to the mask bounding box.",
    },
    "minimum_padding_px": {
        "title": "Minimum Padding",
        "body": "Lower bound used only in fractional-padding mode. It prevents tiny breasts or views from being cropped too tightly.",
        "example": "With fraction 0.01 and minimum 32, the crop still gets at least 32 px padding.",
    },
    "maximum_padding_px": {
        "title": "Maximum Padding",
        "body": "Upper bound used only in fractional-padding mode. It prevents very large images from keeping too much background.",
        "example": "With fraction 0.10 on a 4000 px image, requested padding is 400 px, but maximum 128 clamps it to 128 px.",
    },
    "crop_threshold": {
        "title": "Breast Crop Threshold",
        "body": "Controls foreground detection. Auto uses Otsu or robust percentile logic; manual uses your numeric threshold directly.",
        "example": "Use auto first. Try manual only when a specific DICOM has a bad breast mask.",
    },
    "min_component_area_fraction": {
        "title": "Minimum Component Area",
        "body": "Ignores connected foreground blobs smaller than this fraction of the full image area.",
        "example": "0.001 ignores blobs smaller than 0.1% of the image, which removes many labels or tiny artifacts.",
    },
    "breast_mask_method": {
        "title": "Mask Method",
        "body": "Chooses how the breast foreground mask is found. The default largest_connected_tissue estimates biased background from image borders, thresholds above that noise floor, then keeps the largest connected tissue component.",
        "example": "Use largest_connected_tissue for most mammograms. Otsu remains available only as a fallback/debug option.",
    },
    "breast_mask_open_kernel": {
        "title": "Open Kernel",
        "body": "Morphological opening removes small bright specks from the mask. Larger kernels remove larger specks but can erode thin tissue.",
        "example": "7 is conservative; 0 disables opening; 15 is more aggressive.",
    },
    "breast_mask_close_kernel": {
        "title": "Close Kernel",
        "body": "Morphological closing fills small gaps in the mask. Larger kernels make the mask smoother but can bridge nearby artifacts.",
        "example": "21 is a useful default for mammograms with small internal gaps.",
    },
    "breast_mask_fill_holes": {
        "title": "Fill Mask Holes",
        "body": "Fills enclosed holes inside the detected breast mask so dense internal dark regions remain part of the foreground.",
        "example": "Keep on when mask_outside_breast creates dark holes inside the breast.",
    },
    "breast_mask_keep_largest_component": {
        "title": "Keep Largest Component",
        "body": "Keeps the dominant connected foreground object and removes smaller separated components.",
        "example": "Useful when text labels or markers are brighter than background and detected as foreground.",
    },
    "min_box_visibility_after_crop": {
        "title": "Minimum Box Visibility After Breast Crop",
        "body": "Drops an annotation box if the breast crop leaves less than this fraction of its original area visible.",
        "example": "0.30 means at least 30% of the box must remain after cropping.",
    },
    "crop_size": {
        "title": "Crop Size",
        "body": "Side length of each square crop in pixels. The exported model image size usually starts here before any training-time resizing.",
        "example": "1024 creates 1024 x 1024 crop images. Larger crops contain more context but cost more disk and training memory.",
    },
    "stride": {
        "title": "Deterministic Stride",
        "body": "Distance between neighboring sliding-window crop origins. Smaller stride creates more overlap and more crops.",
        "example": "crop_size=1024 and stride=512 gives 50% overlap.",
    },
    "preview_mode": {
        "title": "Preview Crop Mode",
        "body": "Controls crop proposals in the preview only. Export crop mode is set separately per train/val/test split.",
        "example": "Use deterministic to scan windows, random to sample around masses, bbox-safe to reject masses near crop boundaries.",
    },
    "only_mass_crops": {
        "title": "Show Only Mass Crops",
        "body": "Preview filter that hides crops without enough visible mass. It does not change export class balance.",
        "example": "Turn on to quickly inspect positives; turn off to debug empty/background crops.",
    },
    "positivity_threshold": {
        "title": "Positive Crop Threshold",
        "body": "A preview crop is positive if at least one mass box has this visible fraction inside the crop.",
        "example": "0.30 means a crop is positive when 30% or more of a mass box remains visible.",
    },
    "allow_partial_annotations": {
        "title": "Display Partial Boxes",
        "body": "When enabled, boxes crossing the crop boundary are clipped and shown if they meet visibility requirements.",
        "example": "Useful for debugging; strict exports often disable partial annotations.",
    },
    "min_box_visibility": {
        "title": "Minimum Box Visibility",
        "body": "Minimum visible fraction required to draw or keep a clipped crop annotation.",
        "example": "0.50 keeps boxes only when at least half the original box is visible.",
    },
    "random_preview_count": {
        "title": "Random Crops To Preview",
        "body": "Number of candidate random windows generated for a selected image in preview.",
        "example": "20 is quick; 100 gives more variety but takes longer.",
    },
    "positive_fraction": {
        "title": "Random Positive Request Probability",
        "body": "Probability that the random preview sampler asks for a crop around a mass instead of a clean crop.",
        "example": "0.50 asks for about half positive candidates when positive crops are available.",
    },
    "center_shift_fraction": {
        "title": "Mass-Center Shift Fraction",
        "body": "Controls how far a random positive crop center may move away from the target mass center, relative to crop size.",
        "example": "0.25 on a 1024 crop allows about 256 px of jitter.",
    },
    "random_seed": {
        "title": "Random Seed",
        "body": "Makes stochastic preview crop selection repeatable for the same record and parameters.",
        "example": "Change from 123 to 124 to explore a different set of random candidate windows.",
    },
    "bbox_safe_boundary_margin_fraction": {
        "title": "BBox-Safe Boundary Margin",
        "body": "In bbox-safe mode, the outer fraction of the crop is forbidden for visible annotations. This prevents training boxes from hugging crop edges.",
        "example": "0.02 on a 1024 crop creates a 20 px forbidden band on each side.",
    },
    "bbox_safe_random_shift_fraction": {
        "title": "BBox-Safe Random Shift",
        "body": "How much jitter bbox-safe positive candidates may use around the target box.",
        "example": "0.25 gives variety while still sampling near the mass.",
    },
    "bbox_safe_candidate_count": {
        "title": "BBox-Safe Candidate Count",
        "body": "Number of candidate windows to try before choosing a bbox-safe crop. More candidates improve quality but cost time.",
        "example": "120 is a balanced default; 300 can rescue difficult edge cases.",
    },
    "bbox_safe_top_k": {
        "title": "BBox-Safe Top K",
        "body": "Randomly chooses among the top-ranked valid candidates instead of always taking the best one.",
        "example": "8 keeps variety; 1 makes the sampler deterministic after candidate scoring.",
    },
    "bbox_safe_breast_bias_strength": {
        "title": "Breast Foreground Bias",
        "body": "Scoring weight that prefers candidate crops containing more breast foreground after hard bbox rules pass.",
        "example": "0 ignores this preference; 1 is a useful default; 3 strongly avoids background-heavy crops.",
    },
    "bbox_safe_left_bias_strength": {
        "title": "Left/Chest-Wall Bias",
        "body": "After orientation normalization, gently prefers crops closer to the left chest-wall side.",
        "example": "0 disables this; 0.25 is mild; higher values keep more chest-wall context.",
    },
    "bbox_safe_projection_bias_strength": {
        "title": "X-Projection Peak Bias",
        "body": "Prefers crops that include strong horizontal breast support in the foreground mask.",
        "example": "Useful when valid crops otherwise contain a mass plus mostly background.",
    },
    "require_foreground": {
        "title": "Require Breast Foreground",
        "body": "Rejects crop windows with too little breast-mask foreground.",
        "example": "Use this to stop deterministic sliding export from saving nearly empty corner crops.",
    },
    "min_foreground_fraction": {
        "title": "Minimum Foreground Fraction",
        "body": "Minimum fraction of crop pixels that must be breast foreground when require foreground is enabled.",
        "example": "0.05 means at least 5% of the crop must be breast pixels.",
    },
    "foreground_threshold": {
        "title": "Square-Crop Foreground Threshold",
        "body": "Threshold used for the crop foreground filter. Auto is usually safer; manual is useful for debugging.",
        "example": "If empty-looking crops pass the filter, try a higher manual threshold.",
    },
    "show_foreground_mask": {
        "title": "Show Foreground Mask",
        "body": "Displays the foreground mask used by the crop filter for the selected crop.",
        "example": "Use this when a crop is unexpectedly accepted or rejected by foreground fraction.",
    },
    "display_window": {
        "title": "Display Window Percentiles",
        "body": "Changes contrast of the grayscale preview only. It does not alter exported pixel values or channel pipeline inputs.",
        "example": "1 to 99 shows robust contrast; 0.5 to 99.5 keeps more extreme bright/dark details.",
    },
    "visible_channels": {
        "title": "Visible RGB Channels",
        "body": "Temporarily hides channels in the RGB preview by setting them to zero. Export still keeps all three channels.",
        "example": "Show only B to inspect the detail channel without red/green contributions.",
    },
    "show_channel_panels": {
        "title": "Show Channel Panels",
        "body": "Shows R, G, and B as separate grayscale images below the composite output.",
        "example": "Helpful when one channel is blank, inverted, or over-contrasted.",
    },
    "alignment_enabled": {
        "title": "Opposite-Breast Alignment",
        "body": "Vertically aligns the opposite breast before extracting the same crop window for contralateral channel sources.",
        "example": "If the paired breast sits 50 px lower, alignment shifts it before taking the crop.",
    },
    "alignment_method": {
        "title": "Alignment Method",
        "body": "Chooses how the vertical shift is estimated: nipple tip, row projection, hybrid profile, boundary profile, centroid, intensity projection, or none.",
        "example": "nipple_y is fast; row_projection_y is more global; hybrid_profile_y is useful for debugging.",
    },
    "max_shift_fraction": {
        "title": "Maximum Vertical Shift",
        "body": "Maximum allowed alignment shift as a fraction of image height.",
        "example": "0.10 on a 3000 px image allows at most 300 px vertical shift.",
    },
    "min_profile_overlap_fraction": {
        "title": "Minimum Profile Overlap",
        "body": "For profile-based alignment, candidate shifts need enough overlapping breast-profile support to be considered.",
        "example": "0.60 ignores shifts where less than 60% of the profiles overlap.",
    },
    "preset": {
        "title": "3-Channel Recipe",
        "body": "Loads a complete R/G/B preprocessing recipe. Custom keeps the pipeline from the active YAML.",
        "example": "raw + CLAHE + local detail puts normalized raw in R, local contrast in G, and a detail residual in B.",
    },
    "study_preset": {
        "title": "Study preset",
        "body": "Applies one paper's full data recipe across DICOM loading, fixed preprocessing, patch geometry, sampling, channel export, and save options.",
        "example": "Apply this before fine-tuning individual tabs. Local input/output paths are preserved.",
    },
    "negative_keep_fraction": {
        "title": "Negative patch keep fraction",
        "body": "Keeps this fraction of all eligible negative sliding-window candidates while retaining every positive patch.",
        "example": "0.20 means a seeded 20% sample of negative candidates; it does not mean negatives are 20% of the final dataset.",
    },
    "channel_source": {
        "title": "Channel Source",
        "body": "Each channel can use the current crop or the same window from the aligned opposite breast in the same study/view.",
        "example": "Use contralateral_same_view_crop in B to provide opposite-breast context.",
    },
    "channel_steps": {
        "title": "Channel Steps",
        "body": "Ordered operations applied to one channel. The output of step 1 feeds step 2, and so on.",
        "example": "percentile_normalize -> clahe -> local_detail first normalizes, then enhances local contrast, then extracts fine detail.",
    },
    "export_path": {
        "title": "Export Output Path",
        "body": "Parent folder plus dataset folder name becomes paths.output_root in the generated export config.",
        "example": "/mnt/t9/vindr-data plus preprocessed-vindr-v4 writes to /mnt/t9/vindr-data/preprocessed-vindr-v4.",
    },
    "clean_output": {
        "title": "Delete Output Folder",
        "body": "Removes the output folder before export starts. Use only when you are sure the target folder can be overwritten.",
        "example": "Keep off when experimenting with a path that may contain previous results.",
    },
    "save_square_crops": {
        "title": "Sliding Crop Export",
        "body": "Writes square crop images and YOLO labels for detector training.",
        "example": "Enable this for Ultralytics/MMDetection crop datasets.",
    },
    "save_baseline_uncropped": {
        "title": "Whole-Image Export",
        "body": "Writes one preprocessed full mammogram per source image without final square crop extraction.",
        "example": "Use for baseline visualization or non-crop experiments.",
    },
}


OP_HELP: dict[str, str] = {
    "none": "No operation. The channel passes through unchanged.",
    "percentile_normalize": "Clip to a low/high percentile window and scale to 0..1. Example: 0.5,99.5 removes extreme outliers.",
    "percentile_clip_only": "Clip outliers to the percentile window without rescaling at this step.",
    "zscore_clip": "Standardize by mean/std, clip to +/- z_limit, then map to 0..1.",
    "aggressive_upper_percentile_normalize": "Legacy high-percentile normalization alias.",
    "standardize_to_target": "Affine-standardize a channel to a target mean and standard deviation.",
    "mask_outside_breast": "Set pixels outside the detected breast mask to outside_value.",
    "artifact_cleanup": "Alias-style cleanup that masks non-breast regions to outside_value.",
    "hist_equalize": "Global histogram equalization. Useful for experiments, but can amplify noise.",
    "clahe": "Local contrast enhancement controlled by clip_limit and tile_grid_size.",
    "gaussian_blur": "Gaussian smoothing. Larger kernel/sigma removes more high-frequency noise.",
    "median_blur": "Median denoising. Good for isolated speckle.",
    "bilateral_filter": "Edge-preserving denoising. Slower, with intensity and spatial sigmas.",
    "wiener_filter": "Adaptive local-statistics denoising.",
    "local_detail": "Subtracts a smoothed image and normalizes the residual to highlight fine structures.",
    "sharpen": "Simple sharpening controlled by amount.",
    "unsharp_mask": "Adds high-frequency residual back to the image.",
    "sobel_gradient": "Gradient magnitude edge channel.",
    "laplacian": "Second-derivative edge/detail response.",
    "white_tophat": "Highlights small bright structures using morphology.",
    "blackhat": "Highlights small dark structures using morphology.",
    "morphological_open": "Removes small bright structures.",
    "morphological_close": "Fills small dark gaps.",
    "pectoral_suppression": "Masks a conservative triangular pectoral region.",
    "gamma": "Nonlinear brightness remapping. Gamma < 1 brightens; gamma > 1 darkens.",
    "log": "Log contrast remapping controlled by gain.",
    "invert": "Invert normalized intensities.",
}

OP_DETAILS: dict[str, dict[str, str]] = {
    "none": {
        "what": "Does nothing. The image continues to the next step unchanged.",
        "when": "Use this as an empty placeholder row, or to temporarily disable a step without deleting the rest of the pipeline.",
        "settings": "No settings.",
        "example": "Set a row back to None when you want to compare the pipeline with and without that operation.",
    },
    "percentile_normalize": {
        "what": "Finds two intensity percentiles, clips everything below/above them, then rescales the result to 0..1.",
        "when": "Good first or last step for mammograms because a few extreme pixels can otherwise dominate contrast.",
        "settings": "Low percentile controls how much dark tail is clipped. High percentile controls how much bright tail is clipped. Smaller/larger windows increase contrast but can remove real detail.",
        "example": "0.5 and 99.5 usually keeps almost all breast detail while ignoring extreme labels, edges, or detector artifacts.",
    },
    "percentile_clip_only": {
        "what": "Clips intensities to a percentile window but does not rescale the result afterward.",
        "when": "Use when you want to suppress outliers before another operation that will do its own normalization.",
        "settings": "Low and high percentiles define the clipping window.",
        "example": "Clip 1 to 99 before a custom standardization step.",
    },
    "zscore_clip": {
        "what": "Converts intensities to z-scores using mean and standard deviation, clips extreme z-scores, then maps them to 0..1.",
        "when": "Useful when images have different absolute intensity scales and percentile windows are not stable enough.",
        "settings": "Z limit controls how many standard deviations are kept. Smaller values increase contrast but discard more extremes.",
        "example": "Z limit 3.0 keeps values within roughly three standard deviations.",
    },
    "aggressive_upper_percentile_normalize": {
        "what": "A legacy normalization that focuses on the bright upper intensity range.",
        "when": "Useful as an ablation when dense/bright tissue contrast matters more than dark background detail.",
        "settings": "Percentiles define the upper-focused window. Defaults are usually much higher than ordinary normalization.",
        "example": "70 to 100 strongly emphasizes bright tissue and suppresses lower intensities.",
    },
    "standardize_to_target": {
        "what": "Linearly shifts and scales the channel so foreground pixels approach a target mean and standard deviation.",
        "when": "Useful when you want channel statistics to be similar across images after other preprocessing.",
        "settings": "Target mean and target standard deviation define the desired output statistics. Stat percentiles can ignore extremes before estimating mean/std.",
        "example": "Target mean 0.5 and std 0.2 makes most images land in a comparable brightness range.",
    },
    "hist_equalize": {
        "what": "Global histogram equalization redistributes intensities so the image uses the available brightness range more evenly.",
        "when": "Useful as an experiment when an image looks flat, but it is not a great final default because it can make CC/MLO or vendor differences stronger.",
        "settings": "Stat low/high percentiles ignore extreme tails while learning the mapping.",
        "example": "Try it after percentile normalize. If the result looks harsh or inconsistent across views, prefer CLAHE.",
    },
    "clahe": {
        "what": "Contrast Limited Adaptive Histogram Equalization enhances contrast locally in small tiles instead of across the whole image.",
        "when": "Useful for improving local tissue texture while limiting extreme noise amplification.",
        "settings": "Clip limit caps how aggressively contrast can grow. Tile grid size controls local region size; smaller tiles are more local, larger tiles are smoother.",
        "example": "Clip 2.0 and tile 8 is a conservative start. Raise clip toward 4.0 for stronger texture, lower it if noise gets harsh.",
    },
    "mask_outside_breast": {
        "what": "Detects the breast foreground and replaces pixels outside it with a constant value.",
        "when": "Useful to remove black borders, labels, scanner background, and non-breast regions before contrast operations.",
        "settings": "Outside value is the number written outside the mask. 0 keeps the background black after normalization.",
        "example": "Use outside value 0 before CLAHE so local contrast does not spend effort enhancing the background.",
    },
    "artifact_cleanup": {
        "what": "Uses the same foreground-mask idea as mask outside breast to suppress non-breast artifacts.",
        "when": "Use when labels, borders, markers, or background artifacts are leaking into later contrast operations.",
        "settings": "Outside value controls the replacement value outside the detected foreground.",
        "example": "Use value 0 to make artifacts disappear into the background.",
    },
    "gaussian_blur": {
        "what": "Smooths the image with a Gaussian kernel, reducing high-frequency noise.",
        "when": "Useful before edge/detail operations or when equalization makes fine noise too visible.",
        "settings": "Kernel size controls neighborhood width. Sigma controls blur strength. Larger values smooth more and can soften small lesions.",
        "example": "Kernel 5, sigma 1.0 is mild. Kernel 11, sigma 2.0 is much stronger.",
    },
    "median_blur": {
        "what": "Replaces each pixel with the median of nearby pixels.",
        "when": "Good for isolated speckles or salt-and-pepper noise while preserving edges better than ordinary averaging.",
        "settings": "Kernel size controls neighborhood size. Larger values remove larger speckles but can flatten tiny structures.",
        "example": "Kernel 3 or 5 is usually enough for small isolated bright/dark dots.",
    },
    "bilateral_filter": {
        "what": "Smooths similar nearby pixels while trying to preserve edges.",
        "when": "Useful when you want denoising without washing out tissue boundaries. It is slower than Gaussian blur.",
        "settings": "Diameter is spatial neighborhood size. Sigma color controls how different intensities can mix. Sigma space controls spatial smoothing distance.",
        "example": "Diameter 5, sigma color 1, sigma space 1 is mild. Increase sigmas for stronger smoothing.",
    },
    "wiener_filter": {
        "what": "Adaptive denoising based on local image statistics.",
        "when": "Useful as a denoising ablation, especially for grainy images, but it can soften subtle findings.",
        "settings": "Kernel size controls the local neighborhood. Noise can be left at 0/blank for automatic behavior.",
        "example": "Kernel 7 is a reasonable starting point.",
    },
    "local_detail": {
        "what": "Subtracts a blurred version of the image and normalizes the residual, emphasizing local texture and fine detail.",
        "when": "Useful as a separate channel when you want the model to see small local changes rather than raw brightness.",
        "settings": "Detail sigma controls the scale treated as background. Percentiles control how the residual is clipped/rescaled.",
        "example": "Sigma 12 highlights smaller structures; sigma 30 focuses on broader local contrast differences.",
    },
    "unsharp_mask": {
        "what": "Sharpens by adding high-frequency detail back onto the image.",
        "when": "Useful after smoothing or when lesion borders look too soft.",
        "settings": "Amount controls sharpening strength. Blur sigma controls the detail scale being boosted.",
        "example": "Amount 1.0, sigma 1.0 is moderate. High amount can create bright halos around edges.",
    },
    "sharpen": {
        "what": "Applies a simple sharpening filter that boosts local contrast around edges.",
        "when": "Useful as a lightweight sharpening ablation. It is less controlled than unsharp masking.",
        "settings": "Amount controls sharpening strength.",
        "example": "Amount 0.5 is mild; amount 1.5 can create edge artifacts.",
    },
    "sobel_gradient": {
        "what": "Computes edge strength using first-derivative gradients.",
        "when": "Useful as an edge-focused channel for boundaries, tissue transitions, and mass margins.",
        "settings": "Kernel size controls edge scale. Percentiles clip/rescale the gradient response.",
        "example": "Kernel 3 detects fine edges; kernel 7 is smoother and broader.",
    },
    "laplacian": {
        "what": "Computes a second-derivative response that emphasizes rapid intensity changes.",
        "when": "Useful for fine texture and edge/detail channels, but can be noise-sensitive.",
        "settings": "Kernel size controls derivative scale. Percentiles clip/rescale the response.",
        "example": "Use after mild smoothing if the result looks too noisy.",
    },
    "white_tophat": {
        "what": "Morphological filter that highlights bright structures smaller than the kernel.",
        "when": "Useful for small bright findings or calcification-like bright details.",
        "settings": "Kernel shape and size define what counts as background. Larger kernels highlight larger bright structures.",
        "example": "Kernel 9 highlights small bright spots; kernel 31 highlights broader bright regions.",
    },
    "blackhat": {
        "what": "Morphological filter that highlights dark structures smaller than the kernel.",
        "when": "Useful when dark gaps, lines, or local dark structures matter.",
        "settings": "Kernel shape and size define the structure scale.",
        "example": "Try ellipse kernel 15 to emphasize dark local depressions.",
    },
    "morphological_open": {
        "what": "Erodes then dilates, removing small bright structures and smoothing bright regions.",
        "when": "Useful for suppressing small bright specks or labels.",
        "settings": "Kernel shape and size control what is considered small enough to remove.",
        "example": "Kernel 5 removes tiny bright dots; kernel 15 is much more aggressive.",
    },
    "morphological_close": {
        "what": "Dilates then erodes, filling small dark holes and gaps.",
        "when": "Useful for making breast/tissue regions more continuous.",
        "settings": "Kernel shape and size control the size of gaps that are filled.",
        "example": "Kernel 9 can fill small dark interruptions without changing broad structure too much.",
    },
    "pectoral_suppression": {
        "what": "Masks a triangular upper-corner region to suppress pectoral muscle.",
        "when": "Only use as an experiment for MLO-like views. It can remove clinically relevant chest-wall tissue.",
        "settings": "Side chooses left or right upper corner. Width/height fractions define triangle size. Fill value replaces the triangle.",
        "example": "Left side, width 0.33, height 0.45, fill 0 is conservative but still risky for detection.",
    },
    "gamma": {
        "what": "Nonlinear brightness remapping. Values below 1 brighten dark/mid tones; values above 1 darken them.",
        "when": "Useful for tuning brightness after normalization without changing rank order.",
        "settings": "Gamma is the exponent. Keep changes modest unless you intentionally want a strong brightness shift.",
        "example": "Gamma 0.8 brightens; gamma 1.2 darkens.",
    },
    "log": {
        "what": "Logarithmic intensity remapping that compresses bright values and expands darker values.",
        "when": "Useful when bright tissue dominates and darker tissue needs more visibility.",
        "settings": "Gain controls strength before the log transform.",
        "example": "Gain 1.0 is a safe start; raise it if dark detail still looks compressed.",
    },
    "invert": {
        "what": "Flips intensity: dark becomes bright and bright becomes dark.",
        "when": "Useful for experiments where a downstream model or visualization benefits from inverted contrast.",
        "settings": "No settings.",
        "example": "Use after normalization if you want tissue/background brightness reversed.",
    },
}


PIPELINE_BUILDER_OPS = [
    "none",
    "percentile_normalize",
    "percentile_clip_only",
    "zscore_clip",
    "aggressive_upper_percentile_normalize",
    "standardize_to_target",
    "hist_equalize",
    "clahe",
    "mask_outside_breast",
    "artifact_cleanup",
    "gaussian_blur",
    "median_blur",
    "bilateral_filter",
    "wiener_filter",
    "local_detail",
    "sharpen",
    "unsharp_mask",
    "sobel_gradient",
    "laplacian",
    "white_tophat",
    "blackhat",
    "morphological_open",
    "morphological_close",
    "pectoral_suppression",
    "gamma",
    "log",
    "invert",
]

PIPELINE_OP_LABELS = {
    "none": "None",
    "percentile_normalize": "Percentile normalize",
    "percentile_clip_only": "Percentile clip only",
    "zscore_clip": "Z-score clip",
    "aggressive_upper_percentile_normalize": "Aggressive upper normalize",
    "standardize_to_target": "Standardize to target",
    "hist_equalize": "Histogram equalization",
    "clahe": "CLAHE local contrast",
    "mask_outside_breast": "Mask outside breast",
    "artifact_cleanup": "Artifact cleanup",
    "gaussian_blur": "Gaussian blur",
    "median_blur": "Median blur",
    "bilateral_filter": "Bilateral filter",
    "wiener_filter": "Wiener filter",
    "local_detail": "Local detail",
    "sharpen": "Sharpen",
    "unsharp_mask": "Unsharp mask",
    "sobel_gradient": "Sobel gradient",
    "laplacian": "Laplacian",
    "white_tophat": "White top-hat",
    "blackhat": "Black-hat",
    "morphological_open": "Morphological open",
    "morphological_close": "Morphological close",
    "pectoral_suppression": "Pectoral suppression",
    "gamma": "Gamma",
    "log": "Log remap",
    "invert": "Invert",
}

# A loaded YAML or study preset is the initial source of truth. The visual
# builder starts empty and takes over only after the user adds a step.
DEFAULT_VISUAL_OPS: dict[str, dict[int, str]] = {}

DATASET_OBJECT_CACHE: dict[str, VindrMammoDataset] = {}


EXPORT_QUEUE = ExportQueueManager()
DATASET_METADATA_CACHE: dict[str, dict[str, Any]] = {}
PREVIEW_SAMPLE_CACHE: dict[str, dict[str, Any]] = {}


def _default_config_path() -> Path:
    return Path.cwd() / "config" / "export_config.yaml"


def create_app(config_path: str | Path | None = None) -> Dash:
    config_path = Path(config_path or _default_config_path())
    cfg = _load_config_for_dash(config_path)
    assets_folder = Path(__file__).resolve().parents[2] / "assets"
    app = Dash(
        __name__,
        title=APP_TITLE,
        suppress_callback_exceptions=True,
        assets_folder=str(assets_folder),
    )
    app.layout = _layout(config_path, cfg)
    _register_callbacks(app)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open the Dash VinDr-Mammo preprocessing studio.")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)


def _load_config_for_dash(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return _load_config_cached(str(path))
    except Exception:
        pass
    if path.exists():
        return load_export_config(path)
    return {}


def _config_file_options(selected_path: str | Path | None = None) -> list[dict[str, str]]:
    root = Path.cwd()
    candidates: set[Path] = set()
    for pattern in [
        "config/**/*.yaml",
        "config/**/*.yml",
        "**/metadata/export_config_resolved.yaml",
        "**/export_config_resolved.yaml",
    ]:
        for path in root.glob(pattern):
            if path.is_file():
                candidates.add(path.resolve())
    if selected_path:
        path = Path(selected_path).expanduser()
        if path.exists():
            candidates.add(path.resolve())
    options = []
    for path in sorted(candidates, key=lambda p: str(p)):
        try:
            label = str(path.relative_to(root.resolve()))
        except ValueError:
            label = str(path)
        options.append({"label": label, "value": str(path)})
    return options


def _layout(config_path: Path, cfg: dict[str, Any]) -> html.Div:
    return html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="config-store", data=_jsonable(cfg)),
            dcc.Store(id="dataset-store", data={}),
            dcc.Store(id="pipeline-store", data=_initial_pipeline(cfg)),
            dcc.Store(id="pipeline-mode", data="yaml"),
            dcc.Store(id="selected-help-store", data="config_path"),
            dcc.Store(id="space-estimate-store", data={}),
            dcc.Store(id="active-export-job-store", data=None),
            dcc.Interval(id="job-poll", interval=1200, disabled=True),
            # Keep queue/disk monitoring live in a separately opened browser
            # window even though that client did not click a queue action.
            dcc.Interval(id="queue-poll", interval=1200, disabled=False),
            dcc.Location(id="page-location", refresh=False),
            _style_block(),
            html.A(id="download-link", download="current_preprocessing.yaml", href="", style={"display": "none"}),
            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        [
                            html.H1(APP_TITLE),
                            html.P("Fast Dash interface for previewing, comparing, exporting, and explaining VinDr-Mammo preprocessing settings."),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            _field(
                                "Choose YAML",
                                dcc.Dropdown(
                                    id="config-picker",
                                    options=_config_file_options(config_path),
                                    value=str(config_path) if Path(config_path).exists() else None,
                                    clearable=True,
                                    placeholder="Select a YAML config...",
                                ),
                                "config_path",
                            ),
                            _field("Config YAML", dcc.Input(id="config-path", value=str(config_path), type="text", debounce=True), "config_path"),
                            html.Div(
                                [
                                    html.Button("Rescan", id="rescan-configs", n_clicks=0),
                                    html.Button("Load", id="load-config", n_clicks=0, className="primary"),
                                ],
                                className="config-actions",
                            ),
                        ],
                        className="config-load",
                    ),
                ],
            ),
            dcc.Loading(
                html.Div(id="load-status", className="status-line"),
                type="dot",
                color="#2563eb",
            ),
            html.Main(
                className="workspace",
                children=[
                    html.Section(
                        className="control-pane",
                        children=[
                            _study_preset_controls(),
                            dcc.Tabs(
                                id="control-tabs",
                                value="preview",
                                children=[
                                    dcc.Tab(label="Source", value="preview", children=html.Div(_preview_controls_dash(cfg), className="controls-body")),
                                    dcc.Tab(
                                        label="Preprocessing",
                                        value="preprocess",
                                        children=html.Div(
                                            [
                                                _preprocess_controls(cfg),
                                                _pipeline_controls_dash(cfg, _initial_pipeline(cfg)),
                                            ],
                                            className="controls-body",
                                        ),
                                    ),
                                    dcc.Tab(label="Final View", value="crops", children=html.Div(_crop_controls_dash(cfg), className="controls-body")),
                                    dcc.Tab(label="Save Data", value="export", children=html.Div(_export_controls_dash(cfg), className="controls-body")),
                                    dcc.Tab(label="Storage & Queue", value="queue", children=html.Div(_queue_controls_dash(cfg), className="controls-body")),
                                    dcc.Tab(label="Saved Viewer", value="saved", children=html.Div(_saved_controls_dash(cfg), className="controls-body")),
                                    dcc.Tab(label="Manifest", value="manifests", children=html.Div(_manifest_controls_dash(), className="controls-body")),
                                    dcc.Tab(label="Guide", value="guide", children=html.Div(_guide_controls_dash(), className="controls-body")),
                                ],
                            ),
                            html.Div(
                                className="control-context",
                                children=[
                                    html.Details(
                                        children=[
                                            html.Summary("Selected parameter help"),
                                            html.Div(id="help-body"),
                                        ],
                                    ),
                                    html.Details(
                                        children=[
                                            html.Summary("Loaded dataset"),
                                            dcc.Loading(
                                                html.Div(id="dataset-summary", className="summary-box"),
                                                type="dot",
                                                color="#2563eb",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="viewer-pane",
                        children=[
                            html.Div(
                                className="viewer-toolbar",
                                children=[
                                    dcc.RadioItems(
                                        id="mode",
                                        options=[
                                            {"label": "Single image", "value": "single"},
                                            {"label": "Comparison", "value": "comparison"},
                                            {"label": "Dataset visualizations", "value": "visualizations"},
                                            {"label": "Saved dataset viewer", "value": "saved"},
                                            {"label": "Manifest tools", "value": "manifest"},
                                            {"label": "Storage & queue", "value": "queue"},
                                        ],
                                        value="single",
                                        inline=True,
                                    ),
                                    html.Button("Render / refresh", id="render", n_clicks=0, className="primary"),
                                ],
                            ),
                            dcc.Loading(
                                html.Div(id="render-status", className="render-status note"),
                                type="dot",
                                color="#2563eb",
                            ),
                            dcc.Loading(
                                html.Div(id="viewer-body", className="viewer-body"),
                                type="circle",
                                color="#2563eb",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _study_preset_controls() -> html.Div:
    options = [{"label": "Custom / loaded configuration", "value": "custom"}] + [
        {"label": str(preset["label"]), "value": key}
        for key, preset in STUDY_PRESETS.items()
    ]
    return html.Div(
        className="study-preset-panel",
        children=[
            _field(
                "Study preset",
                dcc.Dropdown(id="study-preset", options=options, value="custom", clearable=False),
                "study_preset",
            ),
            html.Div(
                "Study presets are placed first because they intentionally update settings across multiple tabs.",
                className="note",
            ),
            html.Div(id="study-preset-status", className="status-line"),
        ],
    )


def _style_block() -> Any:
    return html.Div()


def _field(label: str, child: Any, help_key: str | None = None, note: str | None = None) -> html.Div:
    effective_help_key = help_key or f"field:{label}"
    tooltip = _help_tooltip(effective_help_key)
    uid = _help_uid(label, child)
    label_children: list[Any] = [
        html.Span(label),
        html.Button(
            "?",
            id={"type": "help", "key": effective_help_key, "uid": uid},
            n_clicks=0,
            className="help-button",
            title=tooltip,
            **{"aria-label": tooltip},
        ),
    ]
    return html.Div(
        className="field",
        children=[
            html.Label(label_children),
            child,
            html.Small(note) if note else None,
        ],
    )


def _help_tooltip(help_key: str) -> str:
    info = _help_info(help_key)
    parts = [info["title"], info["body"]]
    if info.get("example"):
        parts.append(info["example"])
    return "\n\n".join(parts)


def _help_info(help_key: str) -> dict[str, str]:
    key = str(help_key)
    if key in PARAM_HELP:
        return PARAM_HELP[key]
    if key.startswith("field:"):
        label = key.split(":", 1)[1].strip() or "Parameter"
        return {
            "title": label,
            "body": "This control changes the current GUI state. Use the surrounding section title and the YAML preview/export output to see where it lands in the configuration.",
            "example": "Tip: change one parameter, click Render / refresh, and compare the image panels before changing the next one.",
        }
    return PARAM_HELP["config_path"]


def _help_uid(label: str, child: Any) -> str:
    child_id = getattr(child, "id", None)
    if isinstance(child_id, str):
        raw = f"{label}-{child_id}"
    elif isinstance(child_id, dict):
        raw = f"{label}-{json.dumps(child_id, sort_keys=True)}"
    else:
        raw = label
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()[:80]


def _number(id_: str, value: Any, *, min_: Any = None, max_: Any = None, step: Any = None) -> dcc.Input:
    return dcc.Input(id=id_, value=value, type="number", min=min_, max=max_, step=step, debounce=True)


def _check(id_: str, label: str, value: bool, help_key: str | None = None) -> html.Div:
    return _field(
        label,
        dcc.Checklist(id=id_, options=[{"label": "Enabled", "value": "on"}], value=["on"] if value else []),
        help_key,
    )


def _register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("selected-help-store", "data"),
        Input({"type": "help", "key": ALL, "uid": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_help(_clicks: list[int]) -> str:
        triggered = callback_context.triggered_id
        if isinstance(triggered, dict):
            return str(triggered.get("key", "config_path"))
        raise PreventUpdate

    @app.callback(Output("help-body", "children"), Input("selected-help-store", "data"))
    def _render_help(key: str) -> Any:
        info = _help_info(str(key))
        return html.Div(
            [
                html.H3(info["title"]),
                html.P(info["body"]),
                html.Div(info.get("example", ""), className="note") if info.get("example") else None,
            ]
        )

    @app.callback(Output("method-guide-body", "children"), Input("method-guide-op", "value"))
    def _render_method_guide(op: str) -> Any:
        op = str(op or "none")
        info = OP_DETAILS.get(op, {"what": OP_HELP.get(op, ""), "when": "", "settings": "", "example": ""})
        return [
            html.H3(PIPELINE_OP_LABELS.get(op, op)),
            html.P(info.get("what", "")),
            html.P([html.Strong("When to use: "), info.get("when", "")]),
            html.P([html.Strong("Settings: "), info.get("settings", "")]),
            html.Div(info.get("example", ""), className="note") if info.get("example") else None,
        ]

    @app.callback(
        Output("config-picker", "options"),
        Input("rescan-configs", "n_clicks"),
        State("config-path", "value"),
        prevent_initial_call=False,
    )
    def _rescan_config_options(_n: int, current_path: str) -> list[dict[str, str]]:
        return _config_file_options(current_path)

    @app.callback(
        Output("config-path", "value"),
        Input("config-picker", "value"),
        prevent_initial_call=True,
    )
    def _choose_config(path: str | None) -> str:
        if not path:
            raise PreventUpdate
        return str(path)

    @app.callback(
        Output("config-store", "data"),
        Output("pipeline-store", "data"),
        Output("pipeline-yaml", "value"),
        Output("pipeline-mode", "data"),
        Output("load-status", "children"),
        Output("study-preset", "value"),
        Input("load-config", "n_clicks"),
        State("config-path", "value"),
        prevent_initial_call=True,
    )
    def _load_config(_n: int, path_text: str) -> tuple[dict[str, Any], dict[str, Any], str, str, Any, str]:
        started = time.perf_counter()
        path = Path(str(path_text or "")).expanduser()
        if not path.exists():
            return no_update, no_update, no_update, no_update, html.Div(f"Config not found: {path}", className="error note"), no_update
        try:
            cfg = _load_config_for_dash(path)
            pipeline = _initial_pipeline(cfg)
            pipeline_yaml = yaml.safe_dump(_make_yaml_safe(pipeline), sort_keys=False, width=100)
            elapsed = time.perf_counter() - started
            return (
                _jsonable(cfg),
                pipeline,
                pipeline_yaml,
                "yaml",
                html.Div(f"Loaded {path} in {elapsed:.2f}s. Controls synced from YAML.", className="note"),
                "custom",
            )
        except Exception as exc:
            return no_update, no_update, no_update, no_update, html.Div(f"Could not load config: {exc}", className="error note"), no_update

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("pipeline-yaml", "value", allow_duplicate=True),
        Output("pipeline-mode", "data", allow_duplicate=True),
        Output("study-preset-status", "children"),
        Input("study-preset", "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_cross_section_preset(preset_key: str, cfg: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        if not preset_key or preset_key == "custom":
            raise PreventUpdate
        try:
            effective = apply_study_preset(cfg or {}, str(preset_key))
            pipeline = _initial_pipeline(effective)
            pipeline_yaml = yaml.safe_dump(_make_yaml_safe(pipeline), sort_keys=False, width=100)
            preset = STUDY_PRESETS[str(preset_key)]
            return (
                _jsonable(effective),
                pipeline,
                pipeline_yaml,
                "yaml",
                html.Div([html.Strong("Applied. "), str(preset["description"])], className="note"),
            )
        except Exception as exc:
            return no_update, no_update, no_update, no_update, html.Div(f"Could not apply preset: {exc}", className="error note")

    @app.callback(
        *_config_control_outputs(),
        Input("config-store", "data"),
        prevent_initial_call=False,
    )
    def _sync_controls_from_config(cfg: dict[str, Any]) -> tuple[Any, ...]:
        return _config_control_values(cfg or {})

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("pipeline-yaml", "value", allow_duplicate=True),
        Output("pipeline-mode", "data", allow_duplicate=True),
        Input("pipeline-preset", "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def _preset_pipeline(preset: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        if not preset or preset == "custom":
            pipeline = _initial_pipeline(cfg or {})
        else:
            pipeline = copy.deepcopy(LITERATURE_PIPELINE_PRESETS[str(preset)]["pipeline"])
        return pipeline, yaml.safe_dump(_make_yaml_safe(pipeline), sort_keys=False, width=100), "yaml"

    @app.callback(
        Output("pipeline-mode", "data", allow_duplicate=True),
        Input({"type": "stage-op", "stage": ALL, "idx": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _activate_visual_pipeline(_ops: list[Any]) -> str:
        return "visual"

    @app.callback(
        Output("pipeline-mode", "data", allow_duplicate=True),
        Input("pipeline-yaml", "value"),
        prevent_initial_call=True,
    )
    def _activate_yaml_pipeline(_yaml_text: str) -> str:
        return "yaml"

    @app.callback(
        Output({"type": "stage-row", "stage": ALL, "idx": ALL}, "style"),
        Input({"type": "stage-op", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-row", "stage": ALL, "idx": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _progressive_pipeline_rows(ops: list[Any], row_ids: list[dict[str, Any]]) -> list[dict[str, str]]:
        op_by_stage_idx = _stage_idx_map(row_ids, ops)
        styles = []
        for row_id in row_ids:
            stage = str(row_id.get("stage"))
            idx = int(row_id.get("idx", 0))
            visible = idx == 0
            if idx > 0:
                current_active = str(op_by_stage_idx.get((stage, idx), "none") or "none") != "none"
                previous_active = str(op_by_stage_idx.get((stage, idx - 1), "none") or "none") != "none"
                visible = current_active or previous_active
            styles.append({} if visible else {"display": "none"})
        return styles

    @app.callback(
        Output({"type": "stage-settings", "stage": ALL, "idx": ALL}, "children"),
        Input({"type": "stage-op", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-op", "stage": ALL, "idx": ALL}, "id"),
        State({"type": "stage-settings", "stage": ALL, "idx": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _operation_setting_children(
        ops: list[Any],
        op_ids: list[dict[str, Any]],
        setting_ids: list[dict[str, Any]],
    ) -> list[Any]:
        op_by_stage_idx = _stage_idx_map(op_ids, ops)
        children = []
        for setting_id in setting_ids:
            stage = str(setting_id.get("stage"))
            idx = int(setting_id.get("idx", 0))
            op = str(op_by_stage_idx.get((stage, idx), "none") or "none")
            children.append(_settings_controls_for_op(op, stage, idx))
        return children

    @app.callback(
        Output("whole-options-panel", "style"),
        Output("crop-options-panel", "style"),
        Output("stochastic-crop-options-panel", "style"),
        Output("foreground-crop-options-panel", "style"),
        Input("view-geometry", "value"),
        prevent_initial_call=False,
    )
    def _final_view_mode_panels(view_geometry: str) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
        if str(view_geometry) == "whole":
            return {}, {"display": "none"}, {"display": "none"}, {"display": "none"}
        return {"display": "none"}, {}, {}, {}

    @app.callback(Output("export-mode-summary", "children"), Input("view-geometry", "value"), prevent_initial_call=False)
    def _export_mode_summary(view_geometry: str) -> str:
        if str(view_geometry) == "whole":
            return "Export will save whole-image baseline_uncropped data because Final View is set to whole image."
        return "Export will save square_crops data because Final View is set to square crop."

    @app.callback(
        Output("split-val-fraction", "disabled"),
        Output("split-validation-study-count", "disabled"),
        Output("split-validation-image-count", "disabled"),
        Output("split-stratify-birads", "options"),
        Input("split-strategy", "value"),
        prevent_initial_call=False,
    )
    def _split_control_mode(strategy: str) -> tuple[bool, bool, bool, list[dict[str, Any]]]:
        strategy = str(strategy or "random_study_fraction")
        official_only = strategy == "official_only"
        random_fraction = strategy == "random_study_fraction"
        return (
            not random_fraction,
            strategy != "exact_study_count",
            strategy != "exact_study_count",
            [{
                "label": "Enabled",
                "value": "on",
                "disabled": official_only,
            }],
        )

    @app.callback(
        Output("split-assignment-summary", "children"),
        Input("dataset-store", "data"),
        Input("config-store", "data"),
        Input("split-strategy", "value"),
        Input("split-val-fraction", "value"),
        Input("split-validation-study-count", "value"),
        Input("split-validation-image-count", "value"),
        Input("split-seed", "value"),
        Input("split-stratify-birads", "value"),
        prevent_initial_call=False,
    )
    def _preview_split_assignment(
        dataset_store: dict[str, Any],
        cfg: dict[str, Any],
        strategy: str,
        val_fraction: float,
        validation_study_count: int,
        validation_image_count: int,
        seed: int,
        stratify: list[str],
    ) -> Any:
        params = {
            "split_strategy": strategy,
            "split_val_fraction": val_fraction,
            "split_validation_study_count": validation_study_count,
            "split_validation_image_count": validation_image_count,
            "split_seed": seed,
            "split_stratify_birads": stratify,
        }
        split_cfg = _split_config_from_params(params)
        records = list((dataset_store or {}).get("records", []) or [])
        if not records:
            if split_cfg["strategy"] == "official_only":
                return html.Div("Original VinDr membership selected: validation will be empty.", className="warning note")
            return html.Div("Load dataset metadata with Render / refresh to see projected split counts.", className="note")
        try:
            cohort_cfg = dict((cfg or {}).get("source_cohort", {}) or {})
            split_input_records = records
            if bool(cohort_cfg.get("positive_images_only", False)):
                split_input_records = [
                    record for record in records if bool(record.get("has_mass", False))
                ]
            split_records, _ = make_train_val_test_split(
                split_input_records,
                **normalize_split_strategy_kwargs(split_cfg),
            )
            if bool(cohort_cfg.get("train_expand_to_all_patient_breast_views", False)):
                train_studies = {
                    str(record.get("study_id", ""))
                    for record in split_records.get("train", [])
                }
                split_records["train"] = [
                    record
                    for record in records
                    if str(record.get("split", "training")).casefold().strip() != "test"
                    and str(record.get("study_id", "")) in train_studies
                ]
        except Exception as exc:
            return html.Div(f"Split settings are invalid: {exc}", className="error note")
        metrics = []
        for split_name in ["train", "val", "test"]:
            rows = split_records.get(split_name, [])
            studies = {str(row.get("study_id", "")) for row in rows}
            metrics.append(_metric(split_name.capitalize(), f"{len(rows):,} images / {len(studies):,} studies"))
        note = (
            "Counts include the configured source cohort and patient-level training expansion, before vendor filters. "
            "The official test membership is unchanged."
        )
        return html.Div([html.Div(metrics, className="metric-row"), html.Div(note, className="note")])

    @app.callback(
        Output("dataset-store", "data"),
        Output("dataset-summary", "children"),
        Output("render-status", "children"),
        Input("render", "n_clicks"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def _load_dataset(_n: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
        started = time.perf_counter()
        cfg = cfg or {}
        try:
            cache_key = _metadata_cache_key(cfg)
            if cache_key in DATASET_METADATA_CACHE:
                cached = copy.deepcopy(DATASET_METADATA_CACHE[cache_key])
                elapsed = time.perf_counter() - started
                summary = cached.get("summary", {})
                status = html.Div(
                    f"Dataset metadata restored from cache in {elapsed:.1f}s. Preparing image preview...",
                    className="note",
                )
                return cached, _summary_children(summary), status

            t_dataset = time.perf_counter()
            dataset = _dataset_from_cfg(cfg, read_image=False)
            dataset_seconds = time.perf_counter() - t_dataset
            t_split = time.perf_counter()
            _split_records, split_df = _load_split_records(dataset, cfg)
            split_seconds = time.perf_counter() - t_split
            t_enrich = time.perf_counter()
            records = _build_enriched_record_table(dataset, split_df)
            enrich_seconds = time.perf_counter() - t_enrich
            elapsed = time.perf_counter() - started
            summary = {
                "images": int(len(records)),
                "positive_images": int(records["has_mass"].sum()) if "has_mass" in records else 0,
                "vendors": int(records["vendor"].nunique()) if "vendor" in records else 0,
                "data_root": str(cfg.get("paths", {}).get("data_root", "")),
            }
            status = html.Div(
                [
                    html.Div(f"Dataset metadata loaded in {elapsed:.1f}s. Preparing image preview..."),
                    html.Div(
                        f"Breakdown: CSV/index {dataset_seconds:.1f}s, split table {split_seconds:.1f}s, enriched filters {enrich_seconds:.1f}s.",
                        className="status-detail",
                    ),
                ],
                className="note",
            )
            payload = {"ok": True, "records": records.to_dict("records"), "summary": summary}
            DATASET_METADATA_CACHE[cache_key] = copy.deepcopy(payload)
            return payload, _summary_children(summary), status
        except Exception as exc:
            payload = {"ok": False, "error": str(exc), "records": [], "summary": {}}
            return payload, html.Div(f"Dataset not loaded: {exc}", className="error note"), html.Div(f"Dataset not loaded: {exc}", className="error note")

    @app.callback(
        Output("filter-vendors", "options"),
        Output("filter-vendors", "value"),
        Output("export-vendors", "options"),
        Output("export-vendors", "value"),
        Input("dataset-store", "data"),
        State("config-store", "data"),
    )
    def _populate_vendor_controls(dataset_store: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[str], list[dict[str, str]], list[str]]:
        records = pd.DataFrame((dataset_store or {}).get("records", []))
        vendors = _available_vendors(records) if not records.empty else []
        options = [{"label": v, "value": v} for v in vendors]
        configured = [str(v) for v in ((cfg or {}).get("vendor_filter", {}) or {}).get("include_vendors", [])]
        selected = [v for v in configured if v in vendors]
        if not selected:
            selected = vendors[: min(5, len(vendors))]
        return options, selected, options, selected

    @app.callback(
        Output("viewer-body", "children"),
        Input("render", "n_clicks"),
        Input("mode", "value"),
        Input("dataset-store", "data"),
        State("config-store", "data"),
        *_all_preview_states(),
        prevent_initial_call=False,
    )
    def _render_viewer(_n: int, mode: str, dataset_store: dict[str, Any], cfg: dict[str, Any], *state: Any) -> Any:
        started = time.perf_counter()
        cfg = cfg or {}
        params = _state_to_params(state)
        if mode == "saved":
            return _render_saved_dataset(params, cfg)
        if mode == "visualizations":
            return _render_visualizations(cfg)
        if mode == "manifest":
            return _render_manifest_tools(params, cfg)
        if mode == "queue":
            return html.Div([
                html.H2("Storage and extraction queue"),
                html.P("Use the Storage & Queue control tab to estimate, add, start, remove, or retry pipelines."),
                html.Div(
                    id="queue-window-table",
                    children=_queue_table_children(EXPORT_QUEUE.snapshot()),
                ),
            ])
        if not dataset_store or not dataset_store.get("ok"):
            return html.Div("Load a valid config and click Render / refresh to load dataset metadata.", className="note")
        try:
            deps_ok, deps_message = _dicom_dependencies_available()
            if not deps_ok:
                return html.Div(
                    [
                        html.Div(deps_message, className="error note"),
                        html.Pre("conda install -n data-mmdet pydicom\n# or, inside the active env:\npython -m pip install pydicom"),
                    ]
                )
            dataset = _dataset_from_cfg(
                _cfg_with_preprocess(cfg, params),
                read_image=True,
                preview_max_side=int(params.get("preview_max_side") or 0),
            )
            records = pd.DataFrame(dataset_store.get("records", []))
            if mode == "comparison":
                content = _render_comparison(dataset, records, params)
            else:
                content = _render_single(dataset, records, params)
            elapsed = time.perf_counter() - started
            return html.Div([html.Div(f"Preview rendered in {elapsed:.1f}s.", className="note"), content])
        except Exception as exc:
            return html.Div(f"Render failed: {exc}", className="error note")

    @app.callback(
        Output("export-preview", "children"),
        Input("export-preview-button", "n_clicks"),
        State("config-store", "data"),
        State("dataset-store", "data"),
        *_all_preview_states(),
        prevent_initial_call=True,
    )
    def _preview_export_config(_n: int, cfg: dict[str, Any], dataset_store: dict[str, Any], *state: Any) -> Any:
        params = _state_to_params(state)
        records = pd.DataFrame((dataset_store or {}).get("records", []))
        export_cfg = _build_export_cfg_from_params(cfg or {}, records, params)
        text = yaml.safe_dump(_make_yaml_safe(export_cfg), sort_keys=False, allow_unicode=True, width=120)
        return html.Pre(text)

    @app.callback(
        Output("export-status", "children"),
        Output("job-poll", "disabled"),
        Output("active-export-job-store", "data"),
        Input("export-start-button", "n_clicks"),
        State("config-store", "data"),
        State("dataset-store", "data"),
        *_all_preview_states(),
        prevent_initial_call=True,
    )
    def _start_export(
        _n: int,
        cfg: dict[str, Any],
        dataset_store: dict[str, Any],
        *state: Any,
    ) -> tuple[Any, bool, Any]:
        params = _state_to_params(state)
        if not params.get("export_confirm"):
            return (
                html.Div(
                    "Check the confirmation box before starting export.",
                    className="warning note",
                ),
                True,
                no_update,
            )
        records = pd.DataFrame((dataset_store or {}).get("records", []))
        try:
            export_cfg = _build_export_cfg_from_params(cfg or {}, records, params)
        except Exception as exc:
            return (
                html.Div(f"Could not build export config: {exc}", className="error note"),
                True,
                no_update,
            )
        try:
            estimate = estimate_export_space(export_cfg, (dataset_store or {}).get("records", []))
            job_id = EXPORT_QUEUE.enqueue(
                export_cfg,
                name=str(params.get("export_name") or "export"),
                estimated_bytes=estimate.estimated_bytes,
                metadata={"estimate": estimate.as_dict(), "started_from": "Save Data"},
            )
            EXPORT_QUEUE.start()
        except Exception as exc:
            return (
                html.Div(f"Could not start queued export: {exc}", className="error note"),
                True,
                no_update,
            )
        return (
            html.Div(f"Export queued and started. Job id: {job_id}", className="note"),
            False,
            job_id,
        )

    @app.callback(
        Output("export-status", "children", allow_duplicate=True),
        Output("job-poll", "disabled", allow_duplicate=True),
        Input("job-poll", "n_intervals"),
        State("export-status", "children"),
        State("active-export-job-store", "data"),
        prevent_initial_call=True,
    )
    def _poll_export(
        _ticks: int,
        current: Any,
        active_job_id: str | None,
    ) -> tuple[Any, bool]:
        if not active_job_id:
            return current, True
        try:
            job = EXPORT_QUEUE.get_job(str(active_job_id))
        except QueueJobNotFoundError:
            return (
                html.Div(
                    f"Export queue item {active_job_id} is no longer retained.",
                    className="warning note",
                ),
                True,
            )
        job_id = job.get("job_id")
        status = str(job.get("status"))
        if status in {"queued", "running"}:
            detail = _queue_progress_text(job)
            return html.Div(f"Export {status}. Job id: {job_id}. {detail}", className="note"), False
        if status == "completed":
            return html.Div([html.Div(f"Export complete. Job id: {job_id}", className="note"), html.Pre(json.dumps(job.get("result"), indent=2))]), True
        if status == "failed":
            return html.Div(f"Export failed: {job.get('error')}", className="error note"), True
        return current, True

    @app.callback(
        Output("mode", "value"),
        Output("control-tabs", "value"),
        Input("page-location", "search"),
        prevent_initial_call=False,
    )
    def _open_queue_window(search: str | None) -> tuple[Any, Any]:
        if "queue=1" in str(search or ""):
            return "queue", "queue"
        return no_update, no_update

    @app.callback(
        Output("disk-space-status", "children"),
        Input("export-parent", "value"),
        Input("export-name", "value"),
        Input("queue-poll", "n_intervals"),
        prevent_initial_call=False,
    )
    def _refresh_disk_space(parent: str, name: str, _tick: int) -> Any:
        output_root = Path(str(parent or ".")) / str(name or "")
        try:
            disk = get_disk_space(output_root)
            snapshot = EXPORT_QUEUE.snapshot()
            reserved = sum(
                int(job.get("estimated_bytes") or 0)
                for job in snapshot.get("jobs", [])
                if job.get("status") in {"queued", "running"}
                and _same_disk_for_queue_path(job.get("output_root"), disk.device_id)
            )
            after = int(disk.free_bytes) - int(reserved)
            cls = "warning note" if after < 0 else "note"
            return html.Div(
                className=cls,
                children=[
                    html.Div([html.Strong("Selected filesystem: "), str(disk.probe_path)]),
                    html.Div(f"Free {format_bytes(disk.free_bytes)} of {format_bytes(disk.total_bytes)} ({disk.free_fraction:.1%})."),
                    html.Div(f"Queued/running conservative reservations: {format_bytes(reserved)}. Predicted free after queue: {format_bytes(after)}."),
                ],
            )
        except Exception as exc:
            return html.Div(f"Could not read disk capacity for {output_root}: {exc}", className="error note")

    @app.callback(
        Output("space-estimate-store", "data"),
        Output("space-estimate-status", "children"),
        Input("estimate-space-button", "n_clicks"),
        State("config-store", "data"),
        State("dataset-store", "data"),
        *_all_preview_states(),
        prevent_initial_call=True,
    )
    def _estimate_current_pipeline(
        _n: int,
        cfg: dict[str, Any],
        dataset_store: dict[str, Any],
        *state: Any,
    ) -> tuple[dict[str, Any], Any]:
        params = _state_to_params(state)
        records = pd.DataFrame((dataset_store or {}).get("records", []))
        try:
            effective = _build_export_cfg_from_params(cfg or {}, records, params)
            estimate = estimate_export_space(effective, records)
            payload = estimate.as_dict()
            payload["output_root"] = str(effective.get("paths", {}).get("output_root", ""))
            return payload, _estimate_summary_children(payload)
        except Exception as exc:
            return {}, html.Div(f"Space estimate failed: {exc}", className="error note")

    @app.callback(
        Output("queue-action-status", "children"),
        Output("queue-poll", "disabled"),
        Input("enqueue-pipeline-button", "n_clicks"),
        Input("queue-start-button", "n_clicks"),
        Input("queue-remove-button", "n_clicks"),
        Input("queue-retry-button", "n_clicks"),
        State("space-estimate-store", "data"),
        State("queue-selected-job", "value"),
        State("queue-job-name", "value"),
        State("config-store", "data"),
        State("dataset-store", "data"),
        *_all_preview_states(),
        prevent_initial_call=True,
    )
    def _queue_action(
        _enqueue: int,
        _start: int,
        _remove: int,
        _retry: int,
        estimate_payload: dict[str, Any],
        selected_job: str | None,
        job_name: str,
        cfg: dict[str, Any],
        dataset_store: dict[str, Any],
        *state: Any,
    ) -> tuple[Any, bool]:
        trigger = callback_context.triggered_id
        try:
            if trigger == "enqueue-pipeline-button":
                params = _state_to_params(state)
                records = pd.DataFrame((dataset_store or {}).get("records", []))
                effective = _build_export_cfg_from_params(cfg or {}, records, params)
                estimate = estimate_export_space(effective, records)
                job_id = EXPORT_QUEUE.enqueue(
                    effective,
                    name=str(job_name or params.get("export_name") or "pipeline"),
                    estimated_bytes=estimate.estimated_bytes,
                    metadata={"estimate": estimate.as_dict(), "estimate_preview": estimate_payload or {}},
                )
                return html.Div(f"Added pipeline {job_id} to the queue. Its config is now frozen.", className="note"), False
            if trigger == "queue-start-button":
                EXPORT_QUEUE.start()
                return html.Div("Queue worker started. Jobs run one at a time in FIFO order.", className="note"), False
            if not selected_job:
                return html.Div("Select a queue item first.", className="warning note"), False
            if trigger == "queue-remove-button":
                EXPORT_QUEUE.remove(str(selected_job))
                return html.Div(f"Removed queue item {selected_job}.", className="note"), False
            if trigger == "queue-retry-button":
                EXPORT_QUEUE.retry(str(selected_job))
                EXPORT_QUEUE.start()
                return html.Div(f"Requeued failed item {selected_job}.", className="note"), False
        except Exception as exc:
            return html.Div(str(exc), className="error note"), False
        raise PreventUpdate

    @app.callback(
        Output("queue-table", "children"),
        Output("queue-selected-job", "options"),
        Output("queue-selected-job", "value"),
        Output("queue-window-table", "children"),
        Input("queue-poll", "n_intervals"),
        State("queue-selected-job", "value"),
        prevent_initial_call=False,
    )
    def _poll_queue_view(
        _tick: int,
        selected: str | None,
    ) -> tuple[Any, list[dict[str, str]], str | None, Any]:
        snapshot = EXPORT_QUEUE.snapshot()
        jobs = snapshot.get("jobs", [])
        options = [
            {"label": f"{job.get('name')} — {job.get('status')} — {str(job.get('job_id'))[:8]}", "value": str(job.get("job_id"))}
            for job in jobs
        ]
        ids = {item["value"] for item in options}
        value = selected if selected in ids else (options[-1]["value"] if options else None)
        table = _queue_table_children(snapshot)
        return table, options, value, table


def _preview_controls_dash(cfg: dict[str, Any]) -> Any:
    return html.Div(
        [
            html.Details(
                open=True,
                children=[
                    html.Summary("Image selection"),
                    _field("Split", dcc.Dropdown(id="filter-split", options=SPLITS, value="all", clearable=False), None),
                    _field("Images", dcc.RadioItems(id="filter-positive", options=["positive only", "all images"], value="positive only", inline=True), None),
                    _field("Vendor filter", dcc.RadioItems(id="filter-vendor-mode", options=["all vendors", "selected vendors"], value="all vendors", inline=True), None),
                    _field("Selected vendors", dcc.Dropdown(id="filter-vendors", options=[], value=[], multi=True), None, "Vendor choices populate after dataset metadata loads."),
                    _field("Image index", _number("image-index", 0, min_=0, step=1), None),
                    _field("Crop index", _number("crop-index", 0, min_=0, step=1), None),
                    _field("Comparison slots", _number("comparison-slots", 5, min_=2, max_=10, step=1), None),
                ],
            ),
            html.Details(
                open=True,
                children=[
                    html.Summary("Display"),
                    _check("preview-contralateral", "Preview opposite-breast channel sources", False, "preview_contralateral"),
                    _check("show-annotations", "Show mass annotations", True),
                    _field("Display low percentile", _number("display-low", 1.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Display high percentile", _number("display-high", 99.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Visible RGB channels", dcc.Checklist(id="visible-channels", options=CHANNELS, value=CHANNELS, inline=True), "visible_channels"),
                    _check("show-channel-panels", "Show individual processed channels", True, "show_channel_panels"),
                ],
            ),
        ]
    )


def _preprocess_controls(cfg: dict[str, Any]) -> Any:
    pp = cfg.get("preprocess", {}) or {}
    padding_mode = "fractional" if pp.get("crop_padding", None) is None else "fixed"
    threshold_mode = "auto" if pp.get("crop_threshold", None) is None else "manual"
    return html.Div(
        [
            html.Details(
                open=True,
                children=[
                    html.Summary("Shared fixed preprocessing"),
                    _check("pp-invert", "Invert when needed so tissue is bright on dark background", bool(pp.get("invert_to_black_background", True)), "invert_to_black_background"),
                    _check("pp-crop-breast", "Breast foreground crop step", bool(pp.get("crop_breast", True)), "crop_breast"),
                    _check("pp-mask-outside", "Mask outside detected foreground step", bool(pp.get("mask_outside_breast", True)), "mask_outside_breast"),
                    _check("pp-mirror", "Mirror images so breast enters from the left", bool(pp.get("mirror_right_to_left", True)), "mirror_right_to_left"),
                    _field("Breast crop padding mode", dcc.RadioItems(id="pp-padding-mode", options=["fractional", "fixed"], value=padding_mode, inline=True), "crop_padding"),
                    html.Div(className="grid-2", children=[
                        _field("Padding fraction", _number("pp-padding-fraction", float(pp.get("crop_padding_fraction", 0.03)), min_=0, max_=0.15, step=0.005), "crop_padding_fraction"),
                        _field("Fixed padding px", _number("pp-padding-fixed", int(pp.get("crop_padding", 32) or 32), min_=0, max_=512, step=5), "crop_padding"),
                        _field("Minimum padding px", _number("pp-min-padding", int(pp.get("minimum_padding_px", 32)), min_=0, max_=512, step=8), "minimum_padding_px"),
                        _field("Maximum padding px", _number("pp-max-padding", int(pp.get("maximum_padding_px", 128)), min_=0, max_=1024, step=8), "maximum_padding_px"),
                    ]),
                    _field("Breast crop threshold mode", dcc.RadioItems(id="pp-threshold-mode", options=["auto", "manual"], value=threshold_mode, inline=True), "crop_threshold"),
                    _field("Manual threshold value", _number("pp-threshold", float(pp.get("crop_threshold", 0.0) or 0.0), step=0.01), "crop_threshold"),
                    _field("Minimum breast component area fraction", _number("pp-min-component", float(pp.get("min_component_area_fraction", 0.001)), min_=0, max_=0.05, step=0.0005), "min_component_area_fraction"),
                ],
            ),
            html.Details(
                open=True,
                children=[
                    html.Summary("Mask operation parameters"),
                    _field("Mask method", dcc.Dropdown(id="pp-mask-method", options=["largest_connected_tissue", "percentile_threshold_largest_component", "otsu_largest_connected_component", "mammo_clip_contiguous_variance"], value=str(pp.get("breast_mask_method", "largest_connected_tissue")), clearable=False), "breast_mask_method"),
                    _field("Open kernel", dcc.Dropdown(id="pp-open-kernel", options=[0, 3, 5, 7, 9, 11, 15], value=int(pp.get("breast_mask_open_kernel", 7) or 7), clearable=False), "breast_mask_open_kernel"),
                    _field("Close kernel", dcc.Dropdown(id="pp-close-kernel", options=[0, 7, 11, 15, 21, 31, 41], value=int(pp.get("breast_mask_close_kernel", 21) or 21), clearable=False), "breast_mask_close_kernel"),
                    _check("pp-fill-holes", "Fill breast-mask holes", bool(pp.get("breast_mask_fill_holes", True)), "breast_mask_fill_holes"),
                    _check("pp-largest-component", "Keep largest connected component", bool(pp.get("breast_mask_keep_largest_component", True)), "breast_mask_keep_largest_component"),
                    _field("Minimum box visibility after breast crop", _number("pp-min-box-after-crop", float(pp.get("min_box_visibility_after_crop", 0.30)), min_=0, max_=1, step=0.05), "min_box_visibility_after_crop"),
                ],
            ),
        ]
    )


def _crop_controls_dash(cfg: dict[str, Any]) -> Any:
    crop = cfg.get("square_crops", {}) or {}
    policy = cfg.get("crop_annotation_policy", {}) or {}
    return html.Div(
        [
            html.Details(
                open=True,
                children=[
                    html.Summary("Working view geometry"),
                    _field("Final view type", dcc.RadioItems(id="view-geometry", options=[{"label": "square crop", "value": "crop"}, {"label": "whole image", "value": "whole"}], value="whole", inline=True), "view_geometry"),
                ],
            ),
            html.Div(
                id="whole-options-panel",
                children=[
                    html.Details(
                        open=True,
                        children=[
                            html.Summary("Whole-image options"),
                            _field("Preview image resize", dcc.Dropdown(id="preview-max-side", options=[{"label": "1024 px default", "value": 1024}, {"label": "640 px fastest", "value": 640}, {"label": "1536 px sharper", "value": 1536}, {"label": "2048 px sharpest", "value": 2048}, {"label": "Full resolution / no resize", "value": 0}], value=1024, clearable=False), "preview_max_side"),
                            html.Div("1024 px is the default whole-image working resolution. Use 640 when you need faster previews.", className="note"),
                            _field("Whole-image export resize mode", dcc.Dropdown(id="whole-resize-mode", options=[{"label": "No resize", "value": "none"}, {"label": "Resize to custom exact size (stretch)", "value": "stretch"}, {"label": "Resize to fit custom canvas + pad", "value": "fit_pad"}, {"label": "Resize to fill custom canvas + center crop", "value": "fill_crop"}], value="fit_pad", clearable=False), "preview_max_side"),
                            html.Div(className="grid-2", children=[
                                _field("Custom target width px", _number("whole-resize-width", 1024, min_=1, step=1), "preview_max_side"),
                                _field("Custom target height px", _number("whole-resize-height", 1024, min_=1, step=1), "preview_max_side"),
                                _field("Padding value", _number("whole-pad-value", 0.0, min_=0, max_=1, step=0.05), "preview_max_side"),
                                _field("Padding anchor", dcc.Dropdown(id="whole-pad-anchor", options=[{"label": "left/top", "value": "left_top"}, {"label": "left/center", "value": "left_center"}, {"label": "center", "value": "center"}, {"label": "right/center", "value": "right_center"}], value="left_top", clearable=False), "preview_max_side"),
                            ]),
                            html.Div("Type any positive width and height. The default fit+pad mode preserves anatomy shape and pads to the right/bottom so mirrored breasts stay left-aligned.", className="note"),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="crop-options-panel",
                children=[
                    html.Details(
                        open=True,
                        children=[
                            html.Summary("Crop geometry"),
                            html.Div(className="grid-2", children=[
                                _field("Crop size n", _number("crop-size", int(crop.get("crop_size", 1024)), min_=128, max_=4096, step=128), "crop_size"),
                                _field("Final crop resize", _number("final-crop-resize", int(crop.get("crop_size", 1024)), min_=128, max_=4096, step=128), "crop_size"),
                                _field("Deterministic stride", _number("crop-stride", int(crop.get("stride", 512)), min_=64, max_=4096, step=64), "stride"),
                            ]),
                            _field(
                                "Sliding-window edge behavior",
                                dcc.RadioItems(
                                    id="crop-edge-policy",
                                    options=[
                                        {"label": "Move final window back to align with image edge (legacy)", "value": "edge_align"},
                                        {"label": "Keep the exact stride and zero-pad pixels outside the image", "value": "regular_stride_pad"},
                                    ],
                                    value="regular_stride_pad" if str(crop.get("edge_policy", "edge_align")) in {"regular_stride_pad", "pad"} else "edge_align",
                                ),
                                "stride",
                            ),
                            _field("Preview crop proposal mode", dcc.RadioItems(id="preview-mode", options=[{"label": "deterministic sliding", "value": "deterministic"}, {"label": "stochastic random", "value": "random"}, {"label": "bbox-safe breast-biased random", "value": "bbox_safe_random"}], value=str(crop.get("train_crop_mode", "deterministic")), inline=False), "preview_mode"),
                            _check("only-mass-crops", "Preview only: show only crops with visible mass", False, "only_mass_crops"),
                            _field("Preview positive crop threshold", _number("positivity-threshold", float(policy.get("min_box_visibility", 0.30)), min_=0, max_=1, step=0.05), "positivity_threshold"),
                            _check("allow-partial", "Display partial boxes after clipping", bool(policy.get("allow_partial_annotations", True)), "allow_partial_annotations"),
                            _field("Minimum box visibility to draw/keep", _number("min-box-visibility", float(policy.get("min_box_visibility", 0.30)), min_=0, max_=1, step=0.05), "min_box_visibility"),
                        ],
                    ),
                ],
            ),
            html.Details(
                id="stochastic-crop-options-panel",
                open=True,
                children=[
                    html.Summary("Stochastic and bbox-safe crop options"),
                    html.Div(className="grid-2", children=[
                        _field("Random crops to preview", _number("random-preview-count", int(crop.get("random_crops_per_annotation", 20) or 20), min_=1, max_=500, step=1), "random_preview_count"),
                        _field("Random preview seed", _number("random-seed", int(crop.get("seed", 123)), min_=0, max_=999999, step=1), "random_seed"),
                        _field("Random positive request probability", _number("random-positive-fraction", float(crop.get("positive_fraction", 0.50)), min_=0, max_=1, step=0.05), "positive_fraction"),
                        _field("Mass-center random shift fraction", _number("center-shift-fraction", float(crop.get("center_shift_fraction", 0.25)), min_=0, max_=1, step=0.05), "center_shift_fraction"),
                        _field("Annotation boundary exclusion fraction", _number("bbox-boundary-margin", float(crop.get("bbox_safe_boundary_margin_fraction", 0.02)), min_=0, max_=0.45, step=0.01), "bbox_safe_boundary_margin_fraction"),
                        _field("BBox-safe random shift fraction", _number("bbox-random-shift", float(crop.get("bbox_safe_random_shift_fraction", crop.get("center_shift_fraction", 0.25))), min_=0, max_=1, step=0.05), "bbox_safe_random_shift_fraction"),
                        _field("Candidate windows per crop", _number("bbox-candidate-count", int(crop.get("bbox_safe_candidate_count", 120)), min_=1, max_=1000, step=10), "bbox_safe_candidate_count"),
                        _field("Randomly choose among top K candidates", _number("bbox-top-k", int(crop.get("bbox_safe_top_k", 8)), min_=1, max_=100, step=1), "bbox_safe_top_k"),
                        _field("Breast foreground bias strength", _number("bbox-breast-bias", float(crop.get("bbox_safe_breast_bias_strength", 1.0)), min_=0, max_=5, step=0.1), "bbox_safe_breast_bias_strength"),
                        _field("Left/chest-wall alignment bias strength", _number("bbox-left-bias", float(crop.get("bbox_safe_left_bias_strength", 0.25)), min_=0, max_=5, step=0.1), "bbox_safe_left_bias_strength"),
                        _field("X-projection peak bias strength", _number("bbox-projection-bias", float(crop.get("bbox_safe_projection_bias_strength", 0.25)), min_=0, max_=5, step=0.1), "bbox_safe_projection_bias_strength"),
                    ]),
                ],
            ),
            html.Details(
                id="foreground-crop-options-panel",
                children=[
                    html.Summary("Foreground-ratio crop filter"),
                    html.Div(
                        className="grid-3",
                        children=[
                            html.Div([
                                html.H4("Train"),
                                _check(
                                    "require-foreground",
                                    "Filter by breast coverage",
                                    bool(crop.get(
                                        "train_require_min_breast_fraction_for_all_crops",
                                        crop.get("deterministic_require_foreground", False),
                                    )),
                                    "require_foreground",
                                ),
                                _field(
                                    "Minimum breast fraction",
                                    _number(
                                        "min-foreground-fraction",
                                        float(crop.get(
                                            "train_min_breast_fraction_for_all_crops",
                                            crop.get("deterministic_min_foreground_fraction", 0.05),
                                        )),
                                        min_=0,
                                        max_=1,
                                        step=0.01,
                                    ),
                                    "min_foreground_fraction",
                                ),
                            ]),
                            html.Div([
                                html.H4("Validation"),
                                _check(
                                    "val-require-foreground",
                                    "Filter by breast coverage",
                                    bool(crop.get(
                                        "val_require_min_breast_fraction_for_all_crops",
                                        crop.get("deterministic_require_foreground", False),
                                    )),
                                    "require_foreground",
                                ),
                                _field(
                                    "Minimum breast fraction",
                                    _number(
                                        "val-min-foreground-fraction",
                                        float(crop.get(
                                            "val_min_breast_fraction_for_all_crops",
                                            crop.get("deterministic_min_foreground_fraction", 0.05),
                                        )),
                                        min_=0,
                                        max_=1,
                                        step=0.01,
                                    ),
                                    "min_foreground_fraction",
                                ),
                            ]),
                            html.Div([
                                html.H4("Test"),
                                _check(
                                    "test-require-foreground",
                                    "Filter by breast coverage",
                                    bool(crop.get(
                                        "test_require_min_breast_fraction_for_all_crops",
                                        crop.get("deterministic_require_foreground", False),
                                    )),
                                    "require_foreground",
                                ),
                                _field(
                                    "Minimum breast fraction",
                                    _number(
                                        "test-min-foreground-fraction",
                                        float(crop.get(
                                            "test_min_breast_fraction_for_all_crops",
                                            crop.get("deterministic_min_foreground_fraction", 0.05),
                                        )),
                                        min_=0,
                                        max_=1,
                                        step=0.01,
                                    ),
                                    "min_foreground_fraction",
                                ),
                            ]),
                        ],
                    ),
                    html.Div(
                        "Each split is independent. A threshold of 0.10 with strict comparison keeps only crops with more than 10% retained breast-mask coverage.",
                        className="note",
                    ),
                    _field("Foreground threshold mode", dcc.RadioItems(id="fg-threshold-mode", options=["auto", "manual"], value="auto" if crop.get("deterministic_foreground_threshold", None) is None else "manual", inline=True), "foreground_threshold"),
                    _field("Manual foreground threshold", _number("fg-threshold", float(crop.get("deterministic_foreground_threshold", 0.0) or 0.0), step=0.01), "foreground_threshold"),
                    _check("show-foreground-mask", "Show foreground mask preview for selected crop", False, "show_foreground_mask"),
                ],
            ),
            html.Details(
                children=[
                    html.Summary("Opposite-breast source alignment"),
                    _alignment_controls(cfg),
                ],
            ),
        ]
    )


def _alignment_controls(cfg: dict[str, Any]) -> Any:
    align = ((cfg.get("image_export", {}) or {}).get("contralateral_source_alignment", {}) or {})
    return html.Div(
        [
            _check("alignment-enabled", "Enable opposite-breast vertical alignment", bool(align.get("enabled", True)), "alignment_enabled"),
            _field("Alignment method", dcc.Dropdown(id="alignment-method", options=["nipple_y", "row_projection_y", "hybrid_profile_y", "boundary_profile_y", "mask_centroid_y", "intensity_projection_y", "none"], value=str(align.get("method", "nipple_y")), clearable=False), "alignment_method"),
            _field("Fallback method for hybrid", dcc.Dropdown(id="alignment-fallback", options=["nipple_y", "mask_centroid_y", "row_projection_y", "none"], value=str(align.get("fallback_method", "mask_centroid_y")), clearable=False), "alignment_method"),
            html.Div(className="grid-2", children=[
                _field("Maximum vertical shift fraction", _number("alignment-max-shift", float(align.get("max_shift_fraction", 0.10)), min_=0, max_=0.5, step=0.01), "max_shift_fraction"),
                _field("Minimum profile overlap fraction", _number("alignment-min-overlap", float(align.get("min_profile_overlap_fraction", 0.60)), min_=0.1, max_=0.95, step=0.05), "min_profile_overlap_fraction"),
                _field("Minimum profile match score", _number("alignment-min-score", float(align.get("min_profile_score", 0.05)), min_=-1, max_=1, step=0.01), "alignment_method"),
                _field("Row-vs-boundary score margin", _number("alignment-score-margin", float(align.get("profile_score_margin", 0.03)), min_=0, max_=0.25, step=0.01), "alignment_method"),
                _field("Row-distribution smoothing rows", _number("alignment-projection-smooth", int(align.get("projection_smooth_rows", 31) or 31), min_=1, max_=301, step=2), "alignment_method"),
                _field("Boundary profile smoothing rows", _number("alignment-boundary-smooth", int(align.get("boundary_smooth_rows", align.get("smooth_rows", 21)) or 21), min_=1, max_=201, step=2), "alignment_method"),
            ]),
        ]
    )


def _pipeline_controls_dash(cfg: dict[str, Any], pipeline: dict[str, Any]) -> Any:
    preset_options = [{"label": "Custom / loaded pipeline", "value": "custom"}] + [
        {"label": str(v["label"]), "value": k} for k, v in LITERATURE_PIPELINE_PRESETS.items()
    ]
    return html.Div(
        [
            html.Details(open=True, children=[
                html.Summary("Preset or custom"),
                _field("3-channel recipe", dcc.Dropdown(id="pipeline-preset", options=preset_options, value="custom", clearable=False), "preset"),
                html.Div("Choose a preset as a starting point, or build a custom pipeline below. The visual builder is used when any step is enabled.", className="note"),
            ]),
            html.Details(open=True, children=[
                html.Summary("Visual preprocessing builder"),
                html.Div(
                    "Pipeline order is: common start -> channel-specific R/G/B -> common end. Rows expand as you add steps.",
                    className="note",
                ),
                _visual_pipeline_builder(),
            ]),
            html.Details(open=True, children=[
                html.Summary("Preprocessing method guide"),
                _operation_guide(),
            ]),
            html.Details(open=True, children=[
                html.Summary("Current/advanced YAML"),
                _pipeline_summary(pipeline or _initial_pipeline(cfg)),
                _field(
                    "Common steps applied to every channel",
                    dcc.Textarea(
                        id="common-steps-yaml",
                        value="[]\n",
                    ),
                    "common_channel_steps",
                    "YAML list of operations prepended to R, G, and B for preview and export YAML generation.",
                ),
                _field(
                    "R/G/B custom_channel_pipeline",
                    dcc.Textarea(
                        id="pipeline-yaml",
                        value=yaml.safe_dump(_make_yaml_safe(pipeline or _initial_pipeline(cfg)), sort_keys=False, width=100),
                    ),
                    "channel_steps",
                    "Edit sources, operation order, and params here. Render/export uses this YAML when it parses successfully.",
                ),
            ]),
            html.Details(children=[
                html.Summary("Operation reference"),
                html.Div([html.P([html.Strong(op), ": ", text]) for op, text in OP_HELP.items()]),
            ]),
        ]
    )


def _operation_guide() -> html.Div:
    return html.Div(
        className="method-guide",
        children=[
            _field(
                "Method to explain",
                dcc.Dropdown(
                    id="method-guide-op",
                    options=[{"label": PIPELINE_OP_LABELS.get(op, op), "value": op} for op in PIPELINE_BUILDER_OPS],
                    value="percentile_normalize",
                    clearable=False,
                ),
                None,
            ),
            html.Div(id="method-guide-body", className="method-card"),
        ],
    )


def _visual_pipeline_builder() -> html.Div:
    return html.Div(
        className="pipeline-builder",
        children=[
            html.Div(
                className="pipeline-stage",
                children=[
                    html.H3(PIPELINE_STAGE_LABELS[stage]),
                    _field(
                        "Source",
                        dcc.Dropdown(
                            id={"type": "stage-source", "stage": stage},
                            options=[
                                {"label": "current image/view", "value": "current_crop"},
                                {"label": "opposite breast same view", "value": "contralateral_same_view_crop"},
                            ],
                            value="current_crop",
                            clearable=False,
                            disabled=stage not in CHANNELS,
                        ),
                        "channel_source",
                    ) if stage in CHANNELS else None,
                    *[_pipeline_step_row(stage, idx) for idx in range(PIPELINE_STEP_COUNT)],
                ],
            )
            for stage in PIPELINE_STAGES
        ],
    )


def _pipeline_step_row(stage: str, idx: int) -> html.Div:
    default_op = DEFAULT_VISUAL_OPS.get(stage, {}).get(idx, "none")
    style = None if _initial_row_visible(stage, idx) else {"display": "none"}
    return html.Div(
        className="pipeline-step-row",
        id={"type": "stage-row", "stage": stage, "idx": idx},
        style=style,
        children=[
            _field(
                f"Step {idx + 1}",
                dcc.Dropdown(
                    id={"type": "stage-op", "stage": stage, "idx": idx},
                    options=[{"label": PIPELINE_OP_LABELS.get(op, op), "value": op} for op in PIPELINE_BUILDER_OPS],
                    value=default_op,
                    clearable=False,
                ),
                "channel_steps",
            ),
            _field(
                "Processing extent",
                dcc.Checklist(
                    id={"type": "stage-before-crop", "stage": stage, "idx": idx},
                    options=[{
                        "label": "Apply to whole fixed-preprocessed image before square cropping",
                        "value": "on",
                    }],
                    value=[],
                ),
                "channel_steps",
                "Unchecked keeps this method crop-local. Checked computes it on the whole breast, then extracts the crop.",
            ),
            _operation_settings(stage, idx),
        ],
    )


def _operation_settings(stage: str, idx: int) -> html.Div:
    default_op = DEFAULT_VISUAL_OPS.get(stage, {}).get(idx, "none")
    return html.Div(
        id={"type": "stage-settings", "stage": stage, "idx": idx},
        className="operation-settings",
        children=_settings_controls_for_op(default_op, stage, idx),
    )


def _settings_controls_for_op(op: str, stage: str, idx: int) -> list[Any]:
    group = _settings_group_for_op(op)
    if group == "percentiles":
        return [
            html.Div(
                className="param-group grid-3 compact-grid",
                children=[
                    _field("Low percentile", _number({"type": "stage-lo", "stage": stage, "idx": idx}, _default_stage_lo(stage, idx), min_=0, max_=100, step=0.5), "display_window"),
                    _field("High percentile", _number({"type": "stage-hi", "stage": stage, "idx": idx}, _default_stage_hi(stage, idx), min_=0, max_=100, step=0.5), "display_window"),
                ],
            )
        ]
    if group == "clahe":
        return [
            html.Div(
                className="param-group grid-3 compact-grid",
                children=[
                    _field("Clip limit", _number({"type": "stage-clip", "stage": stage, "idx": idx}, 4.0, min_=0.1, max_=20, step=0.1), "channel_steps"),
                    _field("Tile grid size", _number({"type": "stage-tile", "stage": stage, "idx": idx}, 8, min_=2, max_=64, step=1), "channel_steps"),
                ],
            )
        ]
    if group == "hist_equalize":
        return [
            html.Div(
                className="param-group grid-3 compact-grid",
                children=[
                    _field("Stat low %", _number({"type": "stage-histeq-lo", "stage": stage, "idx": idx}, 1.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Stat high %", _number({"type": "stage-histeq-hi", "stage": stage, "idx": idx}, 99.5, min_=0, max_=100, step=0.5), "display_window"),
                ],
            )
        ]
    if group == "mask":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Outside value", _number({"type": "stage-outside", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.05), "channel_steps")])]
    if group == "blur":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Kernel size", _number({"type": "stage-kernel", "stage": stage, "idx": idx}, 5, min_=1, max_=101, step=2), "breast_mask_open_kernel"),
            _field("Sigma", _number({"type": "stage-sigma", "stage": stage, "idx": idx}, 1.0, min_=0, max_=25, step=0.25), "channel_steps"),
        ])]
    if group == "median":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Kernel size", _number({"type": "stage-median-kernel", "stage": stage, "idx": idx}, 5, min_=1, max_=101, step=2), "breast_mask_open_kernel")])]
    if group == "bilateral":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Diameter", _number({"type": "stage-bilateral-diameter", "stage": stage, "idx": idx}, 5, min_=1, max_=51, step=2), "channel_steps"),
            _field("Sigma color", _number({"type": "stage-sigma-color", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=50, step=0.1), "channel_steps"),
            _field("Sigma space", _number({"type": "stage-sigma-space", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=50, step=0.1), "channel_steps"),
        ])]
    if group == "local_detail":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Low percentile", _number({"type": "stage-local-lo", "stage": stage, "idx": idx}, 1.0, min_=0, max_=100, step=0.5), "display_window"),
            _field("High percentile", _number({"type": "stage-local-hi", "stage": stage, "idx": idx}, 99.0, min_=0, max_=100, step=0.5), "display_window"),
            _field("Detail sigma", _number({"type": "stage-detail-sigma", "stage": stage, "idx": idx}, 12.0, min_=0.1, max_=128, step=0.5), "channel_steps"),
        ])]
    if group == "unsharp":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Amount", _number({"type": "stage-amount", "stage": stage, "idx": idx}, 1.0, min_=0, max_=20, step=0.1), "channel_steps"),
            _field("Blur sigma", _number({"type": "stage-unsharp-sigma", "stage": stage, "idx": idx}, 1.0, min_=0.1, max_=25, step=0.25), "channel_steps"),
        ])]
    if group == "edge":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Low percentile", _number({"type": "stage-edge-lo", "stage": stage, "idx": idx}, 0.5, min_=0, max_=100, step=0.5), "display_window"),
            _field("High percentile", _number({"type": "stage-edge-hi", "stage": stage, "idx": idx}, 99.5, min_=0, max_=100, step=0.5), "display_window"),
            _field("Kernel size", _number({"type": "stage-edge-kernel", "stage": stage, "idx": idx}, 3, min_=1, max_=31, step=2), "channel_steps"),
        ])]
    if group == "morphology":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Kernel shape", dcc.Dropdown(id={"type": "stage-kernel-shape", "stage": stage, "idx": idx}, options=["ellipse", "rect", "cross"], value="ellipse", clearable=False), "channel_steps"),
            _field("Kernel size", _number({"type": "stage-morph-kernel", "stage": stage, "idx": idx}, 9, min_=1, max_=151, step=2), "breast_mask_open_kernel"),
        ])]
    if group == "gamma":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Gamma", _number({"type": "stage-gamma", "stage": stage, "idx": idx}, 1.0, min_=0.05, max_=5, step=0.05), "channel_steps")])]
    if group == "log":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Gain", _number({"type": "stage-gain", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=20, step=0.1), "channel_steps")])]
    if group == "zscore":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Z limit", _number({"type": "stage-z-limit", "stage": stage, "idx": idx}, 3.0, min_=0.1, max_=12, step=0.1), "channel_steps")])]
    if group == "standardize":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Target mean", _number({"type": "stage-target-mean", "stage": stage, "idx": idx}, 0.5, min_=0, max_=1, step=0.01), "channel_steps"),
            _field("Target std", _number({"type": "stage-target-std", "stage": stage, "idx": idx}, 0.2, min_=0.001, max_=1, step=0.01), "channel_steps"),
            _field("Stat low %", _number({"type": "stage-stat-lo", "stage": stage, "idx": idx}, 1.0, min_=0, max_=100, step=0.5), "display_window"),
            _field("Stat high %", _number({"type": "stage-stat-hi", "stage": stage, "idx": idx}, 99.0, min_=0, max_=100, step=0.5), "display_window"),
        ])]
    if group == "wiener":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Kernel size", _number({"type": "stage-wiener-kernel", "stage": stage, "idx": idx}, 7, min_=1, max_=101, step=2), "channel_steps"),
            _field("Noise estimate", _number({"type": "stage-wiener-noise", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.001), "channel_steps"),
        ])]
    if group == "sharpen":
        return [html.Div(className="param-group grid-3 compact-grid", children=[_field("Amount", _number({"type": "stage-sharpen-amount", "stage": stage, "idx": idx}, 1.0, min_=0, max_=20, step=0.1), "channel_steps")])]
    if group == "pectoral":
        return [html.Div(className="param-group grid-3 compact-grid", children=[
            _field("Side", dcc.Dropdown(id={"type": "stage-pectoral-side", "stage": stage, "idx": idx}, options=["left", "right"], value="left", clearable=False), "channel_steps"),
            _field("Width fraction", _number({"type": "stage-pectoral-width", "stage": stage, "idx": idx}, 0.33, min_=0, max_=1, step=0.01), "channel_steps"),
            _field("Height fraction", _number({"type": "stage-pectoral-height", "stage": stage, "idx": idx}, 0.45, min_=0, max_=1, step=0.01), "channel_steps"),
            _field("Fill value", _number({"type": "stage-pectoral-fill", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.05), "channel_steps"),
        ])]
    return [html.Div("This operation has no settings.", className="note compact-note")]

def _operation_settings_old(stage: str, idx: int) -> html.Div:
    return html.Div(
        className="operation-settings",
        children=[
            _param_group(
                stage,
                idx,
                "percentiles",
                [
                    _field("Low percentile", _number({"type": "stage-lo", "stage": stage, "idx": idx}, _default_stage_lo(stage, idx), min_=0, max_=100, step=0.5), "display_window"),
                    _field("High percentile", _number({"type": "stage-hi", "stage": stage, "idx": idx}, _default_stage_hi(stage, idx), min_=0, max_=100, step=0.5), "display_window"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "zscore",
                [
                    _field("Z limit", _number({"type": "stage-z-limit", "stage": stage, "idx": idx}, 3.0, min_=0.1, max_=12, step=0.1), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "standardize",
                [
                    _field("Target mean", _number({"type": "stage-target-mean", "stage": stage, "idx": idx}, 0.5, min_=0, max_=1, step=0.01), "channel_steps"),
                    _field("Target std", _number({"type": "stage-target-std", "stage": stage, "idx": idx}, 0.2, min_=0.001, max_=1, step=0.01), "channel_steps"),
                    _field("Stat low %", _number({"type": "stage-stat-lo", "stage": stage, "idx": idx}, 1.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Stat high %", _number({"type": "stage-stat-hi", "stage": stage, "idx": idx}, 99.0, min_=0, max_=100, step=0.5), "display_window"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "clahe",
                [
                    _field("Clip limit", _number({"type": "stage-clip", "stage": stage, "idx": idx}, 2.0, min_=0.1, max_=20, step=0.1), "channel_steps"),
                    _field("Tile grid size", _number({"type": "stage-tile", "stage": stage, "idx": idx}, 8, min_=2, max_=64, step=1), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "mask",
                [
                    _field("Outside value", _number({"type": "stage-outside", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.05), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "blur",
                [
                    _field("Kernel size", _number({"type": "stage-kernel", "stage": stage, "idx": idx}, 5, min_=1, max_=101, step=2), "breast_mask_open_kernel"),
                    _field("Sigma", _number({"type": "stage-sigma", "stage": stage, "idx": idx}, 1.0, min_=0, max_=25, step=0.25), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "median",
                [
                    _field("Kernel size", _number({"type": "stage-median-kernel", "stage": stage, "idx": idx}, 5, min_=1, max_=101, step=2), "breast_mask_open_kernel"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "bilateral",
                [
                    _field("Diameter", _number({"type": "stage-bilateral-diameter", "stage": stage, "idx": idx}, 5, min_=1, max_=51, step=2), "channel_steps"),
                    _field("Sigma color", _number({"type": "stage-sigma-color", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=50, step=0.1), "channel_steps"),
                    _field("Sigma space", _number({"type": "stage-sigma-space", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=50, step=0.1), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "wiener",
                [
                    _field("Kernel size", _number({"type": "stage-wiener-kernel", "stage": stage, "idx": idx}, 7, min_=1, max_=101, step=2), "channel_steps"),
                    _field("Noise estimate", _number({"type": "stage-wiener-noise", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.001), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "local_detail",
                [
                    _field("Low percentile", _number({"type": "stage-local-lo", "stage": stage, "idx": idx}, 1.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("High percentile", _number({"type": "stage-local-hi", "stage": stage, "idx": idx}, 99.0, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Detail sigma", _number({"type": "stage-detail-sigma", "stage": stage, "idx": idx}, 12.0, min_=0.1, max_=128, step=0.5), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "unsharp",
                [
                    _field("Amount", _number({"type": "stage-amount", "stage": stage, "idx": idx}, 1.0, min_=0, max_=20, step=0.1), "channel_steps"),
                    _field("Blur sigma", _number({"type": "stage-unsharp-sigma", "stage": stage, "idx": idx}, 1.0, min_=0.1, max_=25, step=0.25), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "sharpen",
                [
                    _field("Amount", _number({"type": "stage-sharpen-amount", "stage": stage, "idx": idx}, 1.0, min_=0, max_=20, step=0.1), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "edge",
                [
                    _field("Low percentile", _number({"type": "stage-edge-lo", "stage": stage, "idx": idx}, 0.5, min_=0, max_=100, step=0.5), "display_window"),
                    _field("High percentile", _number({"type": "stage-edge-hi", "stage": stage, "idx": idx}, 99.5, min_=0, max_=100, step=0.5), "display_window"),
                    _field("Kernel size", _number({"type": "stage-edge-kernel", "stage": stage, "idx": idx}, 3, min_=1, max_=31, step=2), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "morphology",
                [
                    _field("Kernel shape", dcc.Dropdown(id={"type": "stage-kernel-shape", "stage": stage, "idx": idx}, options=["ellipse", "rect", "cross"], value="ellipse", clearable=False), "channel_steps"),
                    _field("Kernel size", _number({"type": "stage-morph-kernel", "stage": stage, "idx": idx}, 9, min_=1, max_=151, step=2), "breast_mask_open_kernel"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "gamma",
                [
                    _field("Gamma", _number({"type": "stage-gamma", "stage": stage, "idx": idx}, 1.0, min_=0.05, max_=5, step=0.05), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "pectoral",
                [
                    _field("Side", dcc.Dropdown(id={"type": "stage-pectoral-side", "stage": stage, "idx": idx}, options=["left", "right"], value="left", clearable=False), "channel_steps"),
                    _field("Width fraction", _number({"type": "stage-pectoral-width", "stage": stage, "idx": idx}, 0.33, min_=0, max_=1, step=0.01), "channel_steps"),
                    _field("Height fraction", _number({"type": "stage-pectoral-height", "stage": stage, "idx": idx}, 0.45, min_=0, max_=1, step=0.01), "channel_steps"),
                    _field("Fill value", _number({"type": "stage-pectoral-fill", "stage": stage, "idx": idx}, 0.0, min_=0, max_=1, step=0.05), "channel_steps"),
                ],
            ),
            _param_group(
                stage,
                idx,
                "log",
                [
                    _field("Gain", _number({"type": "stage-gain", "stage": stage, "idx": idx}, 1.0, min_=0.001, max_=20, step=0.1), "channel_steps"),
                ],
            ),
            _param_group(stage, idx, "empty", [html.Div("This operation has no settings.", className="note compact-note")]),
        ],
    )


def _initial_row_visible(stage: str, idx: int) -> bool:
    if idx == 0:
        return True
    return idx in DEFAULT_VISUAL_OPS.get(stage, {}) or (idx - 1) in DEFAULT_VISUAL_OPS.get(stage, {})


def _default_stage_lo(stage: str, idx: int) -> float:
    if stage == "common_start" and idx == 0:
        return 0.0
    if stage == "B" and idx == 0:
        return 50.0
    return 0.5


def _default_stage_hi(stage: str, idx: int) -> float:
    if stage in {"common_start", "B"} and idx == 0:
        return 100.0
    return 99.5


def _param_group(stage: str, idx: int, name: str, children: list[Any]) -> html.Div:
    return html.Div(
        id={"type": "stage-param-group", "stage": stage, "idx": idx, "group": name},
        className="param-group grid-3 compact-grid",
        children=children,
    )


def _pipeline_summary(pipeline: dict[str, Any]) -> Any:
    children = []
    for ch in CHANNELS:
        payload = _pipeline_channel_payload(pipeline, ch)
        steps = payload.get("steps", [])
        children.append(
            html.Div(
                className="image-card",
                children=[
                    html.H3(f"{ch} channel"),
                    html.P(f"Source: {payload.get('source', 'current_crop')}"),
                    html.Pre(yaml.safe_dump(_make_yaml_safe(steps), sort_keys=False, width=100)),
                ],
            )
        )
    return html.Div(className="grid-3", children=children)


def _split_strategy_from_config(split_cfg: dict[str, Any] | None) -> str:
    split_cfg = dict(split_cfg or {})
    explicit = str(split_cfg.get("strategy", "") or "").casefold().strip()
    aliases = {
        "official_only": "official_only",
        "original": "official_only",
        "original_official": "official_only",
        "random_study_fraction": "random_study_fraction",
        "study_fraction": "random_study_fraction",
        "exact_study_count": "exact_study_count",
        "count_matched": "exact_study_count",
    }
    if explicit in aliases:
        return aliases[explicit]
    study_count = split_cfg.get("validation_study_count")
    if study_count is not None:
        return "official_only" if int(study_count) == 0 else "exact_study_count"
    return (
        "random_study_fraction"
        if float(split_cfg.get("val_fraction_from_training", 0.0) or 0.0) > 0.0
        else "official_only"
    )


def _split_config_from_params(params: dict[str, Any]) -> dict[str, Any]:
    strategy = str(params.get("split_strategy") or "random_study_fraction")
    if strategy not in {"official_only", "random_study_fraction", "exact_study_count"}:
        strategy = "random_study_fraction"
    try:
        fraction = min(max(float(params.get("split_val_fraction") or 0.0), 0.0), 0.50)
    except Exception:
        fraction = 0.15
    try:
        study_count = max(0, int(params.get("split_validation_study_count") or 0))
    except Exception:
        study_count = 0
    try:
        image_count_raw = int(params.get("split_validation_image_count") or 0)
    except Exception:
        image_count_raw = 0
    out = {
        "strategy": strategy,
        "val_fraction_from_training": fraction,
        "validation_study_count": None,
        "validation_image_count": None,
        "seed": int(params.get("split_seed") or 123),
        "stratify_by_birads": _is_on(params.get("split_stratify_birads")),
    }
    if strategy == "official_only":
        out.update({
            "val_fraction_from_training": 0.0,
            "validation_study_count": 0,
            "validation_image_count": 0,
            "stratify_by_birads": False,
        })
    elif strategy == "exact_study_count":
        out["validation_study_count"] = study_count
        out["validation_image_count"] = image_count_raw if image_count_raw > 0 else None
    return out


def _split_signature(split_cfg: dict[str, Any] | None) -> tuple[Any, ...]:
    split_cfg = dict(split_cfg or {})
    return (
        _split_strategy_from_config(split_cfg),
        float(split_cfg.get("val_fraction_from_training", 0.0) or 0.0),
        split_cfg.get("validation_study_count"),
        split_cfg.get("validation_image_count"),
        int(split_cfg.get("seed", 123) or 123),
        bool(split_cfg.get("stratify_by_birads", False)),
    )


def _export_controls_dash(cfg: dict[str, Any]) -> Any:
    export_cfg = cfg.get("export", {}) or {}
    current_output = Path(str(cfg.get("paths", {}).get("output_root", "/mnt/t9/vindr-data/preprocessed-vindr-gui")))
    crop_cfg = cfg.get("square_crops", {}) or {}
    paired_cfg = cfg.get("paired_whole_images", {}) or {}
    split_cfg = cfg.get("splits", {}) or {}
    split_strategy = _split_strategy_from_config(split_cfg)
    return html.Div(
        [
            html.Details(open=True, children=[
                html.Summary("Output"),
                _field("Export parent folder", dcc.Input(id="export-parent", value=str(current_output.parent), type="text", debounce=True), "export_path"),
                _field("Dataset folder name", dcc.Input(id="export-name", value=current_output.name or "preprocessed-vindr-gui", type="text", debounce=True), "export_path"),
                _check("clean-output", "Delete output folder before export", bool(export_cfg.get("clean_output_root", False)), "clean_output"),
                html.Div(
                    [
                        _check("save-square-crops", "Sliding crop export (square_crops)", bool(export_cfg.get("save_square_crops", True)), "save_square_crops"),
                        _check("save-baseline", "Whole-image export (baseline_uncropped)", bool(export_cfg.get("save_baseline_uncropped", False)), "save_baseline_uncropped"),
                    ],
                    style={"display": "none"},
                ),
                html.Div(id="export-mode-summary", className="note"),
            ]),
            html.Details(open=bool(paired_cfg.get("enabled", False)), children=[
                html.Summary("Paired whole image for every crop"),
                _check(
                    "paired-whole-enabled",
                    "Save a corresponding pad-first whole-breast image with every crop basename",
                    bool(paired_cfg.get("enabled", False)),
                ),
                _field(
                    "Paired whole target size",
                    _number("paired-whole-size", int(paired_cfg.get("target_width", paired_cfg.get("size", 1024)) or 1024), min_=128, max_=4096, step=128),
                    "crop_size",
                ),
                _check(
                    "paired-whole-hardlink",
                    "Deduplicate repeated whole images with hard links when supported",
                    str(paired_cfg.get("storage_mode", "hardlink")).casefold() in {"hardlink", "link", "deduplicate"},
                ),
                html.Div("The breast is padded to a square first and only then resized. Every companion is written under square_crops/whole_images/<split>/ using the crop's filename.", className="note"),
            ]),
            html.Details(open=True, children=[
                html.Summary("Dataset train / validation / test assignment"),
                _field(
                    "Split policy",
                    dcc.Dropdown(
                        id="split-strategy",
                        options=[
                            {
                                "label": "Random validation studies from official training (recommended)",
                                "value": "random_study_fraction",
                            },
                            {
                                "label": "Original VinDr train/test only (no validation)",
                                "value": "official_only",
                            },
                            {
                                "label": "Exact validation study count",
                                "value": "exact_study_count",
                            },
                        ],
                        value=split_strategy,
                        clearable=False,
                    ),
                    None,
                ),
                html.Div(className="grid-3", children=[
                    _field(
                        "Validation fraction of official training studies",
                        _number(
                            "split-val-fraction",
                            float(split_cfg.get("val_fraction_from_training", 0.15) or 0.0),
                            min_=0.0,
                            max_=0.50,
                            step=0.01,
                        ),
                        None,
                    ),
                    _field(
                        "Validation study count",
                        _number(
                            "split-validation-study-count",
                            int(split_cfg.get("validation_study_count", 0) or 0),
                            min_=0,
                            step=1,
                        ),
                        None,
                    ),
                    _field(
                        "Exact validation image count (0 = unconstrained)",
                        _number(
                            "split-validation-image-count",
                            int(split_cfg.get("validation_image_count", 0) or 0),
                            min_=0,
                            step=1,
                        ),
                        None,
                    ),
                    _field(
                        "Split seed",
                        _number("split-seed", int(split_cfg.get("seed", 123) or 123), min_=0, step=1),
                        None,
                    ),
                    _check(
                        "split-stratify-birads",
                        "Stratify validation studies by BI-RADS",
                        bool(split_cfg.get("stratify_by_birads", True)),
                    ),
                ]),
                html.Div(
                    "VinDr provides official training and test cohorts, but no official validation cohort. "
                    "The recommended mode samples complete studies from official training, so all views stay together, "
                    "and leaves the official test set untouched. Original-only mode is the strict published membership "
                    "and cannot support validation-based early stopping.",
                    className="warning note",
                ),
                html.Div(id="split-assignment-summary", className="summary-box"),
            ]),
            html.Details(open=True, children=[
                html.Summary("Split crop modes"),
                html.Div(className="grid-3", children=[
                    _field("Train crop mode", dcc.Dropdown(id="train-crop-mode", options=["deterministic", "random", "bbox_safe_random"], value=str(crop_cfg.get("train_crop_mode", "deterministic")), clearable=False), "preview_mode"),
                    _field("Val crop mode", dcc.Dropdown(id="val-crop-mode", options=["deterministic", "random", "bbox_safe_random"], value=str(crop_cfg.get("val_crop_mode", "deterministic")), clearable=False), "preview_mode"),
                    _field("Test crop mode", dcc.Dropdown(id="test-crop-mode", options=["deterministic", "random", "bbox_safe_random"], value=str(crop_cfg.get("test_crop_mode", "deterministic")), clearable=False), "preview_mode"),
                ]),
            ]),
            html.Details(open=True, children=[
                html.Summary("Mass/empty export balance"),
                _field(
                    "Balance mode",
                    dcc.Dropdown(
                        id="export-balance-mode",
                        options=[
                            {"label": "All mass + sampled empty", "value": "positive_ratio"},
                            {"label": "All positive + fraction of negative candidates (training)", "value": "negative_fraction"},
                            {"label": "50/50 crops by source breast status (training)", "value": "source_breast_ratio"},
                            {"label": "Mass only", "value": "mass_only"},
                            {"label": "All windows/candidates", "value": "all"},
                        ],
                        value=str(crop_cfg.get("train_deterministic_selection_mode", crop_cfg.get("deterministic_selection_mode", "positive_ratio")) or "positive_ratio"),
                        clearable=False,
                    ),
                    "positive_fraction",
                ),
                _field("Target mass ratio", _number("export-target-positive-ratio", float(crop_cfg.get("deterministic_target_positive_ratio", crop_cfg.get("positive_fraction", 0.50))), min_=0.01, max_=1.0, step=0.01), "positive_fraction"),
                _field(
                    "Training negative patch keep fraction",
                    _number(
                        "export-negative-keep-fraction",
                        float(crop_cfg.get("train_deterministic_negative_keep_fraction", crop_cfg.get("deterministic_negative_keep_fraction", 0.20))),
                        min_=0.0,
                        max_=1.0,
                        step=0.01,
                    ),
                    "negative_keep_fraction",
                ),
                html.Div(
                    "Target-ratio mode keeps all positive crops and samples empty crops toward the requested final ratio. "
                    "Negative-fraction mode instead keeps all positive training patches plus the requested fraction of "
                    "eligible negative candidates. Source-breast mode defines a breast as patient/study + laterality, "
                    "expands only selected training patients to all views, applies the visible breast-mask threshold "
                    "strictly to train, validation, and test crops, excluding background-only marker regions.",
                    className="note",
                ),
                html.Div(
                    "The Source tab's Images = all images setting controls which source images are available for inspection. "
                    "It is separate from patch-level export balance.",
                    className="note",
                ),
            ]),
            html.Details(children=[
                html.Summary("Vendor filter"),
                _field("Vendor export filter", dcc.RadioItems(id="export-vendor-mode", options=["all vendors", "selected vendors only"], value="selected vendors only" if bool((cfg.get("vendor_filter", {}) or {}).get("enabled", False)) else "all vendors", inline=True), None),
                _field("Vendors/devices to include", dcc.Dropdown(id="export-vendors", options=[], value=[], multi=True), None),
            ]),
            html.Details(children=[
                html.Summary("Run export"),
                _check("export-confirm", "I checked the output path and want to start export", False),
                html.Button("Preview effective export YAML", id="export-preview-button"),
                html.Button("Start export", id="export-start-button", className="primary"),
                html.Div(id="export-status"),
                html.Div(id="export-preview"),
            ]),
        ]
    )


def _queue_controls_dash(cfg: dict[str, Any]) -> Any:
    output_root = Path(str(cfg.get("paths", {}).get("output_root", ".")))
    return html.Div(
        [
            html.Details(open=True, children=[
                html.Summary("Selected disk and estimate"),
                html.Div(
                    "Capacity follows the Export parent/folder selected in Save Data. "
                    "The estimate is conservative because breast-crop dimensions and PNG compression vary.",
                    className="note",
                ),
                html.Div(id="disk-space-status", className="summary-box"),
                _field(
                    "Queue item name",
                    dcc.Input(id="queue-job-name", value=output_root.name or "extraction-pipeline", type="text", debounce=True),
                    None,
                ),
                html.Div(className="config-actions", children=[
                    html.Button("Estimate current pipeline", id="estimate-space-button", n_clicks=0),
                    html.Button("Add current pipeline to queue", id="enqueue-pipeline-button", n_clicks=0, className="primary"),
                ]),
                html.Div(id="space-estimate-status"),
            ]),
            html.Details(open=True, children=[
                html.Summary("Extraction queue"),
                html.Div(
                    "Queued configurations are frozen when added and run sequentially. A failed item does not stop the next one.",
                    className="note",
                ),
                html.Div(className="config-actions", children=[
                    html.Button("Start / continue queue", id="queue-start-button", n_clicks=0, className="primary"),
                    html.A("Open queue in another window", href="/?queue=1", target="_blank", className="button-link"),
                ]),
                _field("Selected queue item", dcc.Dropdown(id="queue-selected-job", options=[], value=None, clearable=True), None),
                html.Div(className="config-actions", children=[
                    html.Button("Remove selected", id="queue-remove-button", n_clicks=0),
                    html.Button("Retry selected failure", id="queue-retry-button", n_clicks=0),
                ]),
                html.Div(id="queue-action-status"),
                html.Div(id="queue-table"),
            ]),
        ]
    )


def _saved_controls_dash(cfg: dict[str, Any]) -> Any:
    default_root = str(cfg.get("paths", {}).get("output_root", "/mnt/t9/vindr-data/preprocessed-vindr-gui"))
    return html.Div(
        [
            _field("Exported dataset root or square_crops folder", dcc.Input(id="saved-root", value=default_root, type="text", debounce=True), None),
            _field("Split", dcc.Dropdown(id="saved-split", options=["all", "train", "val", "test"], value="all", clearable=False), None),
            _field("Crop type", dcc.Dropdown(id="saved-positive", options=["all", "positive only", "empty only"], value="all", clearable=False), None),
            _field("Image id/index contains", dcc.Input(id="saved-search", value="", type="text", debounce=True), None),
            _check("saved-existing-only", "Only existing image files", True),
            _check("saved-show-boxes", "Draw annotations", True),
            _field("Saved crop index", _number("saved-index", 0, min_=0, step=1), None),
        ]
    )


def _manifest_controls_dash() -> Any:
    return html.Div(
        [
            _field("Manifest/config paths", dcc.Textarea(id="manifest-paths", value="", placeholder="One manifest.json or export_config.yaml path per line."), None),
            html.Div("Manifest comparison is rendered in the main Manifest tools mode.", className="note"),
        ]
    )


def _guide_controls_dash() -> Any:
    rows = []
    for key, info in PARAM_HELP.items():
        rows.append(html.Details(children=[html.Summary(info["title"]), html.P(info["body"]), html.Div(info.get("example", ""), className="note") if info.get("example") else None]))
    return html.Div(rows)


def _all_preview_states() -> list[State]:
    return [
        State("filter-split", "value"), State("filter-positive", "value"), State("filter-vendor-mode", "value"), State("filter-vendors", "value"),
        State("image-index", "value"), State("crop-index", "value"), State("comparison-slots", "value"),
        State("view-geometry", "value"), State("preview-max-side", "value"), State("preview-contralateral", "value"),
        State("show-annotations", "value"), State("display-low", "value"), State("display-high", "value"), State("visible-channels", "value"), State("show-channel-panels", "value"),
        State("pp-invert", "value"), State("pp-crop-breast", "value"), State("pp-mask-outside", "value"), State("pp-mirror", "value"), State("pp-padding-mode", "value"),
        State("pp-padding-fraction", "value"), State("pp-padding-fixed", "value"), State("pp-min-padding", "value"), State("pp-max-padding", "value"), State("pp-threshold-mode", "value"),
        State("pp-threshold", "value"), State("pp-min-component", "value"), State("pp-mask-method", "value"), State("pp-open-kernel", "value"), State("pp-close-kernel", "value"),
        State("pp-fill-holes", "value"), State("pp-largest-component", "value"), State("pp-min-box-after-crop", "value"),
        State("crop-size", "value"), State("crop-stride", "value"), State("crop-edge-policy", "value"), State("preview-mode", "value"), State("only-mass-crops", "value"), State("positivity-threshold", "value"),
        State("allow-partial", "value"), State("min-box-visibility", "value"), State("random-preview-count", "value"), State("random-seed", "value"), State("random-positive-fraction", "value"),
        State("center-shift-fraction", "value"), State("bbox-boundary-margin", "value"), State("bbox-random-shift", "value"), State("bbox-candidate-count", "value"), State("bbox-top-k", "value"),
        State("bbox-breast-bias", "value"), State("bbox-left-bias", "value"), State("bbox-projection-bias", "value"), State("require-foreground", "value"), State("min-foreground-fraction", "value"),
        State("val-require-foreground", "value"), State("val-min-foreground-fraction", "value"), State("test-require-foreground", "value"), State("test-min-foreground-fraction", "value"),
        State("fg-threshold-mode", "value"), State("fg-threshold", "value"), State("show-foreground-mask", "value"),
        State("alignment-enabled", "value"), State("alignment-method", "value"), State("alignment-fallback", "value"), State("alignment-max-shift", "value"), State("alignment-min-overlap", "value"),
        State("alignment-min-score", "value"), State("alignment-score-margin", "value"), State("alignment-projection-smooth", "value"), State("alignment-boundary-smooth", "value"),
        State("pipeline-store", "data"), State("pipeline-yaml", "value"), State("pipeline-mode", "data"), State("common-steps-yaml", "value"),
        State({"type": "stage-source", "stage": ALL}, "value"),
        State({"type": "stage-op", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-before-crop", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-lo", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-hi", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-kernel", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-sigma", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-amount", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-clip", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-tile", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-histeq-exclude", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-histeq-lo", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-histeq-hi", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-outside", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-median-kernel", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-bilateral-diameter", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-sigma-color", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-sigma-space", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-local-lo", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-local-hi", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-detail-sigma", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-unsharp-sigma", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-edge-lo", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-edge-hi", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-edge-kernel", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-kernel-shape", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-morph-kernel", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-gamma", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-gain", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-z-limit", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-target-mean", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-target-std", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-stat-lo", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-stat-hi", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-wiener-kernel", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-wiener-noise", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-sharpen-amount", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-pectoral-side", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-pectoral-width", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-pectoral-height", "stage": ALL, "idx": ALL}, "value"),
        State({"type": "stage-pectoral-fill", "stage": ALL, "idx": ALL}, "value"),
        State("whole-resize-mode", "value"), State("whole-resize-width", "value"), State("whole-resize-height", "value"), State("whole-pad-value", "value"), State("whole-pad-anchor", "value"),
        State("export-parent", "value"), State("export-name", "value"), State("clean-output", "value"), State("save-square-crops", "value"), State("save-baseline", "value"),
        State("paired-whole-enabled", "value"), State("paired-whole-size", "value"), State("paired-whole-hardlink", "value"),
        State("split-strategy", "value"), State("split-val-fraction", "value"), State("split-validation-study-count", "value"), State("split-validation-image-count", "value"), State("split-seed", "value"), State("split-stratify-birads", "value"),
        State("train-crop-mode", "value"), State("val-crop-mode", "value"), State("test-crop-mode", "value"), State("export-balance-mode", "value"), State("export-target-positive-ratio", "value"), State("export-negative-keep-fraction", "value"), State("export-vendor-mode", "value"), State("export-vendors", "value"), State("export-confirm", "value"),
        State("saved-root", "value"), State("saved-split", "value"), State("saved-positive", "value"), State("saved-search", "value"), State("saved-existing-only", "value"), State("saved-show-boxes", "value"), State("saved-index", "value"),
        State("manifest-paths", "value"),
    ]


def _config_control_outputs() -> list[Output]:
    ids = [
        "filter-split", "filter-positive", "filter-vendor-mode", "view-geometry",
        "pp-invert", "pp-crop-breast", "pp-mask-outside", "pp-mirror", "pp-padding-mode",
        "pp-padding-fraction", "pp-padding-fixed", "pp-min-padding", "pp-max-padding",
        "pp-threshold-mode", "pp-threshold", "pp-min-component", "pp-mask-method",
        "pp-open-kernel", "pp-close-kernel", "pp-fill-holes", "pp-largest-component",
        "pp-min-box-after-crop", "crop-size", "final-crop-resize", "crop-stride", "crop-edge-policy", "preview-mode",
        "only-mass-crops", "positivity-threshold", "allow-partial", "min-box-visibility",
        "random-preview-count", "random-seed", "random-positive-fraction",
        "center-shift-fraction", "bbox-boundary-margin", "bbox-random-shift",
        "bbox-candidate-count", "bbox-top-k", "bbox-breast-bias", "bbox-left-bias",
        "bbox-projection-bias", "require-foreground", "min-foreground-fraction",
        "val-require-foreground", "val-min-foreground-fraction",
        "test-require-foreground", "test-min-foreground-fraction",
        "fg-threshold-mode", "fg-threshold", "show-foreground-mask",
        "alignment-enabled", "alignment-method", "alignment-fallback",
        "alignment-max-shift", "alignment-min-overlap", "alignment-min-score",
        "alignment-score-margin", "alignment-projection-smooth",
        "alignment-boundary-smooth", "whole-resize-mode", "whole-resize-width",
        "whole-resize-height", "whole-pad-value", "whole-pad-anchor",
        "export-parent", "export-name", "clean-output",
        "save-square-crops", "save-baseline", "paired-whole-enabled", "paired-whole-size", "paired-whole-hardlink",
        "split-strategy", "split-val-fraction", "split-validation-study-count", "split-validation-image-count", "split-seed", "split-stratify-birads", "train-crop-mode", "val-crop-mode",
        "test-crop-mode", "export-balance-mode", "export-target-positive-ratio", "export-negative-keep-fraction",
        "export-vendor-mode", "saved-root",
    ]
    return [Output(component_id, "value") for component_id in ids]


def _on_value(value: bool) -> list[str]:
    return ["on"] if bool(value) else []


def _config_control_values(cfg: dict[str, Any]) -> tuple[Any, ...]:
    gui_cfg = cfg.get("gui", {}) or {}
    pp = cfg.get("preprocess", {}) or {}
    crop = cfg.get("square_crops", {}) or {}
    policy = cfg.get("crop_annotation_policy", {}) or {}
    export_cfg = cfg.get("export", {}) or {}
    baseline_cfg = cfg.get("baseline_uncropped", {}) or {}
    paired_cfg = cfg.get("paired_whole_images", {}) or {}
    split_cfg = cfg.get("splits", {}) or {}
    align = ((cfg.get("image_export", {}) or {}).get("contralateral_source_alignment", {}) or {})
    current_output = Path(str(cfg.get("paths", {}).get("output_root", "/mnt/t9/vindr-data/preprocessed-vindr-gui")))
    vendor_cfg = cfg.get("vendor_filter", {}) or {}
    padding_mode = "fractional" if pp.get("crop_padding", None) is None else "fixed"
    threshold_mode = "auto" if pp.get("crop_threshold", None) is None else "manual"
    fg_threshold_mode = "auto" if crop.get("deterministic_foreground_threshold", None) is None else "manual"
    preview_mode = str(crop.get("train_crop_mode", "deterministic") or "deterministic")
    if preview_mode not in {"deterministic", "random", "bbox_safe_random"}:
        preview_mode = "deterministic"
    return (
        str(gui_cfg.get("filter_split", "all")),
        str(gui_cfg.get("filter_positive", "positive only")),
        str(gui_cfg.get("filter_vendor_mode", "all vendors")),
        "crop" if bool(export_cfg.get("save_square_crops", True)) else "whole",
        _on_value(pp.get("invert_to_black_background", True)),
        _on_value(pp.get("crop_breast", True)),
        _on_value(pp.get("mask_outside_breast", True)),
        _on_value(pp.get("mirror_right_to_left", True)),
        padding_mode,
        float(pp.get("crop_padding_fraction", 0.03)),
        int(pp.get("crop_padding", 32) or 32),
        int(pp.get("minimum_padding_px", 32)),
        int(pp.get("maximum_padding_px", 128)),
        threshold_mode,
        float(pp.get("crop_threshold", 0.0) or 0.0),
        float(pp.get("min_component_area_fraction", 0.001)),
        str(pp.get("breast_mask_method", "largest_connected_tissue")),
        int(pp.get("breast_mask_open_kernel", 7) or 7),
        int(pp.get("breast_mask_close_kernel", 21) or 21),
        _on_value(pp.get("breast_mask_fill_holes", True)),
        _on_value(pp.get("breast_mask_keep_largest_component", True)),
        float(pp.get("min_box_visibility_after_crop", 0.30)),
        int(crop.get("crop_size", 1024)),
        int(crop.get("crop_size", 1024)),
        int(crop.get("stride", 512)),
        "regular_stride_pad" if str(crop.get("edge_policy", "edge_align")) in {"regular_stride_pad", "pad"} else "edge_align",
        preview_mode,
        [],
        float(policy.get("min_box_visibility", 0.30)),
        _on_value(policy.get("allow_partial_annotations", True)),
        float(policy.get("min_box_visibility", 0.30)),
        int(crop.get("random_crops_per_annotation", 20) or 20),
        int(crop.get("seed", 123)),
        float(crop.get("preview_positive_fraction", crop.get("positive_fraction", 0.50))),
        float(crop.get("center_shift_fraction", 0.25)),
        float(crop.get("bbox_safe_boundary_margin_fraction", 0.02)),
        float(crop.get("bbox_safe_random_shift_fraction", crop.get("center_shift_fraction", 0.25))),
        int(crop.get("bbox_safe_candidate_count", 120)),
        int(crop.get("bbox_safe_top_k", 8)),
        float(crop.get("bbox_safe_breast_bias_strength", 1.0)),
        float(crop.get("bbox_safe_left_bias_strength", 0.25)),
        float(crop.get("bbox_safe_projection_bias_strength", 0.25)),
        _on_value(crop.get(
            "train_require_min_breast_fraction_for_all_crops",
            crop.get("deterministic_require_foreground", False),
        )),
        float(crop.get(
            "train_min_breast_fraction_for_all_crops",
            crop.get("deterministic_min_foreground_fraction", 0.05),
        )),
        _on_value(crop.get(
            "val_require_min_breast_fraction_for_all_crops",
            crop.get("deterministic_require_foreground", False),
        )),
        float(crop.get(
            "val_min_breast_fraction_for_all_crops",
            crop.get("deterministic_min_foreground_fraction", 0.05),
        )),
        _on_value(crop.get(
            "test_require_min_breast_fraction_for_all_crops",
            crop.get("deterministic_require_foreground", False),
        )),
        float(crop.get(
            "test_min_breast_fraction_for_all_crops",
            crop.get("deterministic_min_foreground_fraction", 0.05),
        )),
        fg_threshold_mode,
        float(crop.get("deterministic_foreground_threshold", 0.0) or 0.0),
        [],
        _on_value(align.get("enabled", True)),
        str(align.get("method", "nipple_y")),
        str(align.get("fallback_method", "mask_centroid_y")),
        float(align.get("max_shift_fraction", 0.10)),
        float(align.get("min_profile_overlap_fraction", 0.60)),
        float(align.get("min_profile_score", 0.05)),
        float(align.get("profile_score_margin", 0.03)),
        int(align.get("projection_smooth_rows", 31) or 31),
        int(align.get("boundary_smooth_rows", align.get("smooth_rows", 21)) or 21),
        str(baseline_cfg.get("resize_mode", "fit_pad")),
        int(baseline_cfg.get("target_width", 1024) or 1024),
        int(baseline_cfg.get("target_height", 1024) or 1024),
        float(baseline_cfg.get("pad_value", 0.0)),
        str(baseline_cfg.get("pad_anchor", "left_top") or "left_top"),
        str(current_output.parent),
        current_output.name or "preprocessed-vindr-gui",
        _on_value(export_cfg.get("clean_output_root", False)),
        _on_value(export_cfg.get("save_square_crops", True)),
        _on_value(export_cfg.get("save_baseline_uncropped", False)),
        _on_value(paired_cfg.get("enabled", False)),
        int(paired_cfg.get("target_width", paired_cfg.get("size", 1024)) or 1024),
        _on_value(str(paired_cfg.get("storage_mode", "hardlink")).casefold() in {"hardlink", "link", "deduplicate"}),
        _split_strategy_from_config(split_cfg),
        float(split_cfg.get("val_fraction_from_training", 0.15) or 0.0),
        int(split_cfg.get("validation_study_count", 0) or 0),
        int(split_cfg.get("validation_image_count", 0) or 0),
        int(split_cfg.get("seed", 123) or 123),
        _on_value(split_cfg.get("stratify_by_birads", False)),
        str(crop.get("train_crop_mode", "deterministic")),
        str(crop.get("val_crop_mode", "deterministic")),
        str(crop.get("test_crop_mode", "deterministic")),
        str(crop.get("train_deterministic_selection_mode", crop.get("deterministic_selection_mode", "positive_ratio")) or "positive_ratio"),
        float(crop.get("train_deterministic_target_positive_ratio", crop.get("deterministic_target_positive_ratio", crop.get("positive_fraction", 0.50)))),
        float(crop.get("train_deterministic_negative_keep_fraction", crop.get("deterministic_negative_keep_fraction", 0.20))),
        "selected vendors only" if bool(vendor_cfg.get("enabled", False)) else "all vendors",
        str(cfg.get("paths", {}).get("output_root", "/mnt/t9/vindr-data/preprocessed-vindr-gui")),
    )


def _state_to_params(values: tuple[Any, ...]) -> dict[str, Any]:
    keys = [
        "filter_split", "filter_positive", "filter_vendor_mode", "filter_vendors", "image_index", "crop_index", "comparison_slots",
        "view_geometry", "preview_max_side", "preview_contralateral",
        "show_annotations", "display_low", "display_high", "visible_channels", "show_channel_panels",
        "pp_invert", "pp_crop_breast", "pp_mask_outside", "pp_mirror", "pp_padding_mode", "pp_padding_fraction", "pp_padding_fixed", "pp_min_padding", "pp_max_padding",
        "pp_threshold_mode", "pp_threshold", "pp_min_component", "pp_mask_method", "pp_open_kernel", "pp_close_kernel", "pp_fill_holes", "pp_largest_component", "pp_min_box_after_crop",
        "crop_size", "crop_stride", "crop_edge_policy", "preview_mode", "only_mass_crops", "positivity_threshold", "allow_partial", "min_box_visibility", "random_preview_count", "random_seed",
        "random_positive_fraction", "center_shift_fraction", "bbox_boundary_margin", "bbox_random_shift", "bbox_candidate_count", "bbox_top_k", "bbox_breast_bias", "bbox_left_bias",
        "bbox_projection_bias", "require_foreground", "min_foreground_fraction",
        "val_require_foreground", "val_min_foreground_fraction",
        "test_require_foreground", "test_min_foreground_fraction",
        "fg_threshold_mode", "fg_threshold", "show_foreground_mask",
        "alignment_enabled", "alignment_method", "alignment_fallback", "alignment_max_shift", "alignment_min_overlap", "alignment_min_score", "alignment_score_margin",
        "alignment_projection_smooth", "alignment_boundary_smooth", "pipeline", "pipeline_yaml", "pipeline_mode", "common_steps_yaml",
        "stage_sources", "stage_ops", "stage_before_crop", "stage_los", "stage_his", "stage_kernels", "stage_sigmas", "stage_amounts", "stage_clips",
        "stage_tiles", "stage_histeq_excludes", "stage_histeq_los", "stage_histeq_his", "stage_outside_values", "stage_median_kernels", "stage_bilateral_diameters", "stage_sigma_colors", "stage_sigma_spaces",
        "stage_local_los", "stage_local_his", "stage_detail_sigmas", "stage_unsharp_sigmas", "stage_edge_los", "stage_edge_his",
        "stage_edge_kernels", "stage_kernel_shapes", "stage_morph_kernels", "stage_gammas", "stage_gains",
        "stage_z_limits", "stage_target_means", "stage_target_stds", "stage_stat_los", "stage_stat_his",
        "stage_wiener_kernels", "stage_wiener_noises", "stage_sharpen_amounts", "stage_pectoral_sides",
        "stage_pectoral_widths", "stage_pectoral_heights", "stage_pectoral_fills",
        "whole_resize_mode", "whole_resize_width", "whole_resize_height", "whole_pad_value", "whole_pad_anchor",
        "export_parent", "export_name", "clean_output", "save_square_crops", "save_baseline", "paired_whole_enabled", "paired_whole_size", "paired_whole_hardlink",
        "split_strategy", "split_val_fraction", "split_validation_study_count", "split_validation_image_count", "split_seed", "split_stratify_birads",
        "train_crop_mode", "val_crop_mode", "test_crop_mode", "export_balance_mode", "export_target_positive_ratio", "export_negative_keep_fraction", "export_vendor_mode", "export_vendors", "export_confirm",
        "saved_root", "saved_split", "saved_positive", "saved_search", "saved_existing_only", "saved_show_boxes", "saved_index", "manifest_paths",
    ]
    return dict(zip(keys, values, strict=False))


def _stage_idx_map(ids: list[dict[str, Any]], values: list[Any]) -> dict[tuple[str, int], Any]:
    out: dict[tuple[str, int], Any] = {}
    for item_id, value in zip(ids or [], values or [], strict=False):
        try:
            out[(str(item_id.get("stage")), int(item_id.get("idx", 0)))] = value
        except Exception:
            continue
    return out


def _settings_group_for_op(op: str) -> str:
    op = str(op or "none")
    if op in {"percentile_normalize", "percentile_clip_only", "aggressive_upper_percentile_normalize"}:
        return "percentiles"
    if op == "zscore_clip":
        return "zscore"
    if op == "standardize_to_target":
        return "standardize"
    if op == "clahe":
        return "clahe"
    if op == "hist_equalize":
        return "hist_equalize"
    if op in {"mask_outside_breast", "artifact_cleanup"}:
        return "mask"
    if op == "gaussian_blur":
        return "blur"
    if op == "median_blur":
        return "median"
    if op == "bilateral_filter":
        return "bilateral"
    if op == "wiener_filter":
        return "wiener"
    if op == "local_detail":
        return "local_detail"
    if op == "unsharp_mask":
        return "unsharp"
    if op == "sharpen":
        return "sharpen"
    if op in {"sobel_gradient", "laplacian"}:
        return "edge"
    if op in {"white_tophat", "blackhat", "morphological_open", "morphological_close"}:
        return "morphology"
    if op == "pectoral_suppression":
        return "pectoral"
    if op == "gamma":
        return "gamma"
    if op == "log":
        return "log"
    return "empty"


def _metadata_cache_key(cfg: dict[str, Any]) -> str:
    relevant = {
        "paths": cfg.get("paths", {}),
        "splits": cfg.get("splits", {}),
        "vendor_filter": cfg.get("vendor_filter", {}),
        "metadata": cfg.get("metadata", {}),
        "dataset": cfg.get("dataset", {}),
    }
    return json.dumps(_jsonable(relevant), sort_keys=True)


def _dicom_dependencies_available() -> tuple[bool, str]:
    try:
        import pydicom  # noqa: F401
    except Exception as exc:
        return False, (
            "pydicom is not installed in the Python environment running this GUI. "
            "Metadata can load without it, but image preview needs it to read DICOM files. "
            f"Original error: {exc}"
        )
    return True, ""


def _is_on(value: Any) -> bool:
    return isinstance(value, list) and "on" in value


def _pipeline_from_params(params: dict[str, Any]) -> dict[str, Any]:
    if str(params.get("pipeline_mode") or "yaml") == "visual":
        visual = _visual_pipeline_from_params(params)
        if visual is not None:
            return visual
    pipeline: dict[str, Any] = {}
    text = params.get("pipeline_yaml")
    if isinstance(text, str) and text.strip():
        try:
            parsed = yaml.safe_load(text) or {}
            if isinstance(parsed, dict):
                pipeline = copy.deepcopy(parsed)
        except Exception:
            pipeline = {}
    if not pipeline:
        fallback = params.get("pipeline")
        pipeline = copy.deepcopy(fallback if isinstance(fallback, dict) else {})
    common_steps = _common_steps_from_params(params)
    if common_steps:
        for channel in CHANNELS:
            payload = _pipeline_channel_payload(pipeline, channel)
            payload["steps"] = copy.deepcopy(common_steps) + list(payload.get("steps", []) or [])
            pipeline[channel] = payload
    return pipeline


def _visual_pipeline_from_params(params: dict[str, Any]) -> dict[str, Any] | None:
    ops = params.get("stage_ops")
    if not isinstance(ops, list) or not any(str(op or "none") != "none" for op in ops):
        return None
    los = params.get("stage_los") or []
    his = params.get("stage_his") or []
    kernels = params.get("stage_kernels") or []
    sigmas = params.get("stage_sigmas") or []
    amounts = params.get("stage_amounts") or []
    clips = params.get("stage_clips") or []
    tiles = params.get("stage_tiles") or []
    histeq_excludes = params.get("stage_histeq_excludes") or []
    histeq_los = params.get("stage_histeq_los") or []
    histeq_his = params.get("stage_histeq_his") or []
    outside_values = params.get("stage_outside_values") or []
    median_kernels = params.get("stage_median_kernels") or []
    bilateral_diameters = params.get("stage_bilateral_diameters") or []
    sigma_colors = params.get("stage_sigma_colors") or []
    sigma_spaces = params.get("stage_sigma_spaces") or []
    local_los = params.get("stage_local_los") or []
    local_his = params.get("stage_local_his") or []
    detail_sigmas = params.get("stage_detail_sigmas") or []
    unsharp_sigmas = params.get("stage_unsharp_sigmas") or []
    edge_los = params.get("stage_edge_los") or []
    edge_his = params.get("stage_edge_his") or []
    edge_kernels = params.get("stage_edge_kernels") or []
    kernel_shapes = params.get("stage_kernel_shapes") or []
    morph_kernels = params.get("stage_morph_kernels") or []
    gammas = params.get("stage_gammas") or []
    gains = params.get("stage_gains") or []
    z_limits = params.get("stage_z_limits") or []
    target_means = params.get("stage_target_means") or []
    target_stds = params.get("stage_target_stds") or []
    stat_los = params.get("stage_stat_los") or []
    stat_his = params.get("stage_stat_his") or []
    wiener_kernels = params.get("stage_wiener_kernels") or []
    wiener_noises = params.get("stage_wiener_noises") or []
    sharpen_amounts = params.get("stage_sharpen_amounts") or []
    pectoral_sides = params.get("stage_pectoral_sides") or []
    pectoral_widths = params.get("stage_pectoral_widths") or []
    pectoral_heights = params.get("stage_pectoral_heights") or []
    pectoral_fills = params.get("stage_pectoral_fills") or []
    sources = params.get("stage_sources") or ["current_crop", "current_crop", "current_crop"]
    before_crop_values = params.get("stage_before_crop") or []
    cursors: dict[str, int] = {}

    def _take_float(name: str, values: list[Any], default: float) -> float:
        idx = cursors.get(name, 0)
        cursors[name] = idx + 1
        return _list_get_float(values, idx, default)

    def _take_int(name: str, values: list[Any], default: int) -> int:
        idx = cursors.get(name, 0)
        cursors[name] = idx + 1
        return _list_get_int(values, idx, default)

    def _take_str(name: str, values: list[Any], default: str) -> str:
        idx = cursors.get(name, 0)
        cursors[name] = idx + 1
        return _list_get_str(values, idx, default)

    stage_steps = {stage: [] for stage in PIPELINE_STAGES}
    for flat_idx, op_raw in enumerate(ops):
        stage = PIPELINE_STAGES[min(flat_idx // PIPELINE_STEP_COUNT, len(PIPELINE_STAGES) - 1)]
        idx_in_stage = int(flat_idx % PIPELINE_STEP_COUNT)
        op = str(op_raw or "none")
        if op == "none":
            continue
        group = _settings_group_for_op(op)
        lo = _default_stage_lo(stage, idx_in_stage)
        hi = _default_stage_hi(stage, idx_in_stage)
        kernel = 5
        sigma = 1.0
        amount = 1.0
        clip = 4.0
        tile = 8
        histeq_exclude = 0.0
        histeq_lo = 1.0
        histeq_hi = 99.5
        outside_value = 0.0
        median_kernel = 5
        bilateral_diameter = 5
        sigma_color = 1.0
        sigma_space = 1.0
        local_lo = 1.0
        local_hi = 99.0
        detail_sigma = 12.0
        unsharp_sigma = 1.0
        edge_lo = 0.5
        edge_hi = 99.5
        edge_kernel = 3
        kernel_shape = "ellipse"
        morph_kernel = 9
        gamma = 1.0
        gain = 1.0
        z_limit = 3.0
        target_mean = 0.5
        target_std = 0.2
        stat_lo = 1.0
        stat_hi = 99.0
        wiener_kernel = 7
        wiener_noise = 0.0
        sharpen_amount = 1.0
        pectoral_side = "left"
        pectoral_width = 0.33
        pectoral_height = 0.45
        pectoral_fill = 0.0
        if group == "percentiles":
            lo = _take_float("lo", los, lo)
            hi = _take_float("hi", his, hi)
        elif group == "clahe":
            clip = _take_float("clip", clips, clip)
            tile = _take_int("tile", tiles, tile)
        elif group == "hist_equalize":
            histeq_exclude = _take_float("histeq_exclude", histeq_excludes, histeq_exclude)
            histeq_lo = _take_float("histeq_lo", histeq_los, histeq_lo)
            histeq_hi = _take_float("histeq_hi", histeq_his, histeq_hi)
        elif group == "mask":
            outside_value = _take_float("outside", outside_values, outside_value)
        elif group == "blur":
            kernel = _take_int("kernel", kernels, kernel)
            sigma = _take_float("sigma", sigmas, sigma)
        elif group == "median":
            median_kernel = _take_int("median_kernel", median_kernels, median_kernel)
        elif group == "bilateral":
            bilateral_diameter = _take_int("bilateral_diameter", bilateral_diameters, bilateral_diameter)
            sigma_color = _take_float("sigma_color", sigma_colors, sigma_color)
            sigma_space = _take_float("sigma_space", sigma_spaces, sigma_space)
        elif group == "local_detail":
            local_lo = _take_float("local_lo", local_los, local_lo)
            local_hi = _take_float("local_hi", local_his, local_hi)
            detail_sigma = _take_float("detail_sigma", detail_sigmas, detail_sigma)
        elif group == "unsharp":
            amount = _take_float("amount", amounts, amount)
            unsharp_sigma = _take_float("unsharp_sigma", unsharp_sigmas, unsharp_sigma)
        elif group == "edge":
            edge_lo = _take_float("edge_lo", edge_los, edge_lo)
            edge_hi = _take_float("edge_hi", edge_his, edge_hi)
            edge_kernel = _take_int("edge_kernel", edge_kernels, edge_kernel)
        elif group == "morphology":
            kernel_shape = _take_str("kernel_shape", kernel_shapes, kernel_shape)
            morph_kernel = _take_int("morph_kernel", morph_kernels, morph_kernel)
        elif group == "gamma":
            gamma = _take_float("gamma", gammas, gamma)
        elif group == "log":
            gain = _take_float("gain", gains, gain)
        elif group == "zscore":
            z_limit = _take_float("z_limit", z_limits, z_limit)
        elif group == "standardize":
            target_mean = _take_float("target_mean", target_means, target_mean)
            target_std = _take_float("target_std", target_stds, target_std)
            stat_lo = _take_float("stat_lo", stat_los, stat_lo)
            stat_hi = _take_float("stat_hi", stat_his, stat_hi)
        elif group == "wiener":
            wiener_kernel = _take_int("wiener_kernel", wiener_kernels, wiener_kernel)
            wiener_noise = _take_float("wiener_noise", wiener_noises, wiener_noise)
        elif group == "sharpen":
            sharpen_amount = _take_float("sharpen_amount", sharpen_amounts, sharpen_amount)
        elif group == "pectoral":
            pectoral_side = _take_str("pectoral_side", pectoral_sides, pectoral_side)
            pectoral_width = _take_float("pectoral_width", pectoral_widths, pectoral_width)
            pectoral_height = _take_float("pectoral_height", pectoral_heights, pectoral_height)
            pectoral_fill = _take_float("pectoral_fill", pectoral_fills, pectoral_fill)
        params_for_op = _params_for_visual_op(
            op,
            lo=lo, hi=hi, kernel=kernel, sigma=sigma, amount=amount, clip=clip, tile=tile,
            outside_value=outside_value, median_kernel=median_kernel, bilateral_diameter=bilateral_diameter,
            sigma_color=sigma_color, sigma_space=sigma_space, local_lo=local_lo, local_hi=local_hi,
            detail_sigma=detail_sigma, unsharp_sigma=unsharp_sigma, edge_lo=edge_lo, edge_hi=edge_hi,
            edge_kernel=edge_kernel, kernel_shape=kernel_shape, morph_kernel=morph_kernel, gamma=gamma,
            gain=gain, z_limit=z_limit, target_mean=target_mean, target_std=target_std, stat_lo=stat_lo,
            stat_hi=stat_hi, wiener_kernel=wiener_kernel, wiener_noise=wiener_noise,
            sharpen_amount=sharpen_amount, pectoral_side=pectoral_side, pectoral_width=pectoral_width,
            pectoral_height=pectoral_height, pectoral_fill=pectoral_fill,
            histeq_exclude=histeq_exclude, histeq_lo=histeq_lo, histeq_hi=histeq_hi,
        )
        stage_steps[stage].append({
            "op": op,
            "params": params_for_op,
            "apply_before_crop": _is_on(before_crop_values[flat_idx]) if flat_idx < len(before_crop_values) else False,
        })

    source_by_channel = {
        "R": str(sources[0] if len(sources) > 0 else "current_crop"),
        "G": str(sources[1] if len(sources) > 1 else "current_crop"),
        "B": str(sources[2] if len(sources) > 2 else "current_crop"),
    }
    pipeline = {}
    for channel in CHANNELS:
        pipeline[channel] = {
            "source": source_by_channel[channel],
            "steps": copy.deepcopy(stage_steps["common_start"]) + copy.deepcopy(stage_steps[channel]) + copy.deepcopy(stage_steps["common_end"]),
        }
    return pipeline


def _params_for_visual_op(
    op: str,
    *,
    lo: float,
    hi: float,
    kernel: int,
    sigma: float,
    amount: float,
    clip: float,
    tile: int,
    outside_value: float,
    median_kernel: int,
    bilateral_diameter: int,
    sigma_color: float,
    sigma_space: float,
    local_lo: float,
    local_hi: float,
    detail_sigma: float,
    unsharp_sigma: float,
    edge_lo: float,
    edge_hi: float,
    edge_kernel: int,
    kernel_shape: str,
    morph_kernel: int,
    gamma: float,
    gain: float,
    z_limit: float,
    target_mean: float,
    target_std: float,
    stat_lo: float,
    stat_hi: float,
    wiener_kernel: int,
    wiener_noise: float,
    sharpen_amount: float,
    pectoral_side: str,
    pectoral_width: float,
    pectoral_height: float,
    pectoral_fill: float,
    histeq_exclude: float,
    histeq_lo: float,
    histeq_hi: float,
) -> dict[str, Any]:
    if hi < lo:
        lo, hi = hi, lo
    if op in {"percentile_normalize", "percentile_clip_only"}:
        return {"percentiles": [float(lo), float(hi)]}
    if op == "aggressive_upper_percentile_normalize":
        return {"percentiles": [float(lo), float(hi)]}
    if op == "zscore_clip":
        return {"z_limit": max(float(z_limit), 1e-6)}
    if op == "standardize_to_target":
        if stat_hi < stat_lo:
            stat_lo, stat_hi = stat_hi, stat_lo
        return {
            "target_mean": float(target_mean),
            "target_std": max(float(target_std), 1e-8),
            "stat_percentiles": [float(stat_lo), float(stat_hi)],
            "clip_output": True,
        }
    if op == "clahe":
        return {"clip_limit": float(clip), "tile_grid_size": int(max(2, tile))}
    if op == "hist_equalize":
        if histeq_hi < histeq_lo:
            histeq_lo, histeq_hi = histeq_hi, histeq_lo
        return {
            "exclude_chest_wall_fraction": min(max(float(histeq_exclude), 0.0), 0.45),
            "stat_percentiles": [float(histeq_lo), float(histeq_hi)],
        }
    if op in {"mask_outside_breast", "artifact_cleanup"}:
        return {"outside_value": float(outside_value)}
    if op == "gaussian_blur":
        return {"ksize": int(_odd_positive(kernel)), "sigma": float(sigma)}
    if op == "median_blur":
        return {"ksize": int(_odd_positive(median_kernel))}
    if op == "bilateral_filter":
        return {
            "diameter": int(_odd_positive(bilateral_diameter)),
            "sigma_color": max(float(sigma_color), 0.001),
            "sigma_space": max(float(sigma_space), 0.001),
        }
    if op == "wiener_filter":
        return {"ksize": int(_odd_positive(wiener_kernel)), "noise": max(float(wiener_noise), 0.0)}
    if op == "local_detail":
        if local_hi < local_lo:
            local_lo, local_hi = local_hi, local_lo
        return {"percentiles": [float(local_lo), float(local_hi)], "sigma": float(detail_sigma)}
    if op == "unsharp_mask":
        return {"amount": float(amount), "sigma": float(unsharp_sigma)}
    if op == "sharpen":
        return {"amount": float(sharpen_amount)}
    if op in {"sobel_gradient", "laplacian"}:
        if edge_hi < edge_lo:
            edge_lo, edge_hi = edge_hi, edge_lo
        return {"percentiles": [float(edge_lo), float(edge_hi)], "ksize": int(_odd_positive(edge_kernel))}
    if op in {"white_tophat", "blackhat", "morphological_open", "morphological_close"}:
        return {"kernel_shape": _kernel_shape_value(kernel_shape), "kernel_size": int(_odd_positive(morph_kernel))}
    if op == "pectoral_suppression":
        return {
            "side": "right" if str(pectoral_side).lower() == "right" else "left",
            "width_fraction": min(max(float(pectoral_width), 0.0), 1.0),
            "height_fraction": min(max(float(pectoral_height), 0.0), 1.0),
            "fill_value": float(pectoral_fill),
        }
    if op == "gamma":
        return {"gamma": max(float(gamma), 0.001)}
    if op == "log":
        return {"gain": max(float(gain), 0.001)}
    return {}


def _list_get_float(values: list[Any], idx: int, default: float) -> float:
    try:
        return float(values[idx])
    except Exception:
        return float(default)


def _list_get_int(values: list[Any], idx: int, default: int) -> int:
    try:
        return int(values[idx])
    except Exception:
        return int(default)


def _list_get_str(values: list[Any], idx: int, default: str) -> str:
    try:
        value = str(values[idx])
    except Exception:
        value = default
    return value or default


def _kernel_shape_value(value: str) -> str:
    value = str(value or "ellipse").lower()
    return value if value in {"ellipse", "rect", "cross"} else "ellipse"


def _odd_positive(value: int) -> int:
    out = max(1, int(value))
    return out if out % 2 == 1 else out + 1


def _common_steps_from_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    text = params.get("common_steps_yaml")
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        parsed = yaml.safe_load(text) or []
    except Exception:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("steps", [])
    if not isinstance(parsed, list):
        return []
    return [step for step in parsed if isinstance(step, dict) and str(step.get("op", "none")) != "none"]


def _preview_pipeline_from_params(params: dict[str, Any]) -> dict[str, Any]:
    pipeline = copy.deepcopy(_pipeline_from_params(params))
    if not _is_on(params.get("preview_contralateral")):
        for channel in CHANNELS:
            payload = pipeline.get(channel)
            if isinstance(payload, dict) and payload.get("source") == "contralateral_same_view_crop":
                payload["source"] = "current_crop"
    return pipeline


def _cfg_with_preprocess(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    pp = out.setdefault("preprocess", {})
    pp["invert_to_black_background"] = _is_on(params.get("pp_invert"))
    pp["crop_breast"] = _is_on(params.get("pp_crop_breast"))
    pp["mask_outside_breast"] = _is_on(params.get("pp_mask_outside"))
    pp["mirror_right_to_left"] = _is_on(params.get("pp_mirror"))
    if params.get("pp_padding_mode") == "fixed":
        pp["crop_padding"] = int(params.get("pp_padding_fixed") or 0)
    else:
        pp["crop_padding"] = None
        pp["crop_padding_fraction"] = float(params.get("pp_padding_fraction") or 0.03)
        pp["minimum_padding_px"] = int(params.get("pp_min_padding") or 32)
        pp["maximum_padding_px"] = int(params.get("pp_max_padding") or 128)
    pp["crop_threshold"] = None if params.get("pp_threshold_mode") == "auto" else float(params.get("pp_threshold") or 0.0)
    pp["min_component_area_fraction"] = float(params.get("pp_min_component") or 0.001)
    pp["breast_mask_method"] = str(params.get("pp_mask_method") or "largest_connected_tissue")
    pp["breast_mask_open_kernel"] = int(params.get("pp_open_kernel") or 0)
    pp["breast_mask_close_kernel"] = int(params.get("pp_close_kernel") or 0)
    pp["breast_mask_fill_holes"] = _is_on(params.get("pp_fill_holes"))
    pp["breast_mask_keep_largest_component"] = _is_on(params.get("pp_largest_component"))
    pp["min_box_visibility_after_crop"] = float(params.get("pp_min_box_after_crop") or 0.30)
    return out


def _crop_controls_from_params(params: dict[str, Any]) -> dict[str, Any]:
    align = {
        "enabled": _is_on(params.get("alignment_enabled")),
        "method": str(params.get("alignment_method") or "nipple_y"),
        "fallback_method": str(params.get("alignment_fallback") or "mask_centroid_y"),
        "max_shift_fraction": float(params.get("alignment_max_shift") or 0.10),
        "min_profile_overlap_fraction": float(params.get("alignment_min_overlap") or 0.60),
        "min_profile_score": float(params.get("alignment_min_score") or 0.05),
        "profile_score_margin": float(params.get("alignment_score_margin") or 0.03),
        "projection_smooth_rows": int(params.get("alignment_projection_smooth") or 31),
        "boundary_smooth_rows": int(params.get("alignment_boundary_smooth") or 21),
        "smooth_rows": int(params.get("alignment_boundary_smooth") or 21),
        "pad_value": 0.0,
    }
    crop_options = {
        "enabled": True,
        "mode": str(params.get("preview_mode") or "deterministic"),
        "crop_size": int(params.get("crop_size") or 1024),
        "stride": int(params.get("crop_stride") or 512),
        "edge_policy": str(params.get("crop_edge_policy") or "edge_align"),
        "allow_partial_annotations": _is_on(params.get("allow_partial")),
        "min_box_visibility": float(params.get("min_box_visibility") or 0.30),
        "reject_partial_windows": not _is_on(params.get("allow_partial")),
        "negative_max_box_visibility": 0.0,
        "pad_if_needed": True,
        "pad_value": 0.0,
        "positive_fraction": float(params.get("random_positive_fraction") or 0.50),
        "center_shift_fraction": float(params.get("center_shift_fraction") or 0.25),
        "max_random_tries": 80,
        "bbox_safe_boundary_margin_fraction": float(params.get("bbox_boundary_margin") or 0.02),
        "bbox_safe_random_shift_fraction": float(params.get("bbox_random_shift") or 0.25),
        "bbox_safe_candidate_count": int(params.get("bbox_candidate_count") or 120),
        "bbox_safe_top_k": int(params.get("bbox_top_k") or 8),
        "bbox_safe_breast_bias_strength": float(params.get("bbox_breast_bias") or 1.0),
        "bbox_safe_left_bias_strength": float(params.get("bbox_left_bias") or 0.25),
        "bbox_safe_projection_bias_strength": float(params.get("bbox_projection_bias") or 0.25),
    }
    return {
        "crop_size": int(params.get("crop_size") or 1024),
        "stride": int(params.get("crop_stride") or 512),
        "edge_policy": str(params.get("crop_edge_policy") or "edge_align"),
        "mode": str(params.get("preview_mode") or "deterministic"),
        "random_preview_count": int(params.get("random_preview_count") or 20),
        "random_seed": int(params.get("random_seed") or 123),
        "only_mass_crops": _is_on(params.get("only_mass_crops")),
        "positivity_threshold": float(params.get("positivity_threshold") or 0.30),
        "require_foreground": _is_on(params.get("require_foreground")),
        "min_foreground_fraction": float(params.get("min_foreground_fraction") or 0.05),
        "split_breast_filters": {
            "train": {
                "enabled": _is_on(params.get("require_foreground")),
                "minimum": float(params.get("min_foreground_fraction") or 0.05),
            },
            "val": {
                "enabled": _is_on(params.get("val_require_foreground")),
                "minimum": float(params.get("val_min_foreground_fraction") or 0.05),
            },
            "test": {
                "enabled": _is_on(params.get("test_require_foreground")),
                "minimum": float(params.get("test_min_foreground_fraction") or 0.05),
            },
        },
        "foreground_threshold": None if params.get("fg_threshold_mode") == "auto" else float(params.get("fg_threshold") or 0.0),
        "foreground_mask_preview": _is_on(params.get("show_foreground_mask")),
        "bbox_safe_boundary_margin_fraction": crop_options["bbox_safe_boundary_margin_fraction"],
        "bbox_safe_random_shift_fraction": crop_options["bbox_safe_random_shift_fraction"],
        "bbox_safe_candidate_count": crop_options["bbox_safe_candidate_count"],
        "bbox_safe_top_k": crop_options["bbox_safe_top_k"],
        "bbox_safe_breast_bias_strength": crop_options["bbox_safe_breast_bias_strength"],
        "bbox_safe_left_bias_strength": crop_options["bbox_safe_left_bias_strength"],
        "bbox_safe_projection_bias_strength": crop_options["bbox_safe_projection_bias_strength"],
        "crop_options": crop_options,
        "contralateral_source_alignment": align,
    }


def _dataset_from_cfg(
    cfg: dict[str, Any],
    *,
    read_image: bool = True,
    preview_max_side: int | None = None,
) -> VindrMammoDataset:
    paths = cfg.get("paths", {}) or {}
    pp = cfg.get("preprocess", {}) or {}
    image_cfg = cfg.get("image", {}) or {}
    output_size = None
    if read_image and preview_max_side:
        max_side = int(preview_max_side)
        if max_side > 0:
            output_size = (max_side, max_side)
    dataset_kwargs = {
        "data_root": paths.get("data_root"),
        "index_level": "image",
        "split": None,
        "read_image": bool(read_image),
        "output_size": output_size,
        "normalize": image_cfg.get("normalize", "none"),
        "percentile_range": tuple(image_cfg.get("percentile_range", [0.5, 99.5])),
        "use_voi_lut": bool(image_cfg.get("use_voi_lut", False)),
        "strict_voi_lut": bool(image_cfg.get("strict_voi_lut", False)),
        "return_dicom_meta": bool(read_image) and bool(cfg.get("metadata", {}).get("include_dicom_meta", True)),
        "validate_paths": bool(cfg.get("dataset", {}).get("validate_paths", False)),
        "preprocess_options": pp,
        "crop_options": {"enabled": False},
        "show_progress": False,
    }
    cache_key = json.dumps(_jsonable(dataset_kwargs), sort_keys=True)
    cached = DATASET_OBJECT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dataset = VindrMammoDataset(**dataset_kwargs)
    if len(DATASET_OBJECT_CACHE) > 8:
        DATASET_OBJECT_CACHE.pop(next(iter(DATASET_OBJECT_CACHE)))
    DATASET_OBJECT_CACHE[cache_key] = dataset
    return dataset


def _filter_records(records: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if records.empty:
        return records
    df = records.copy()
    try:
        split_cfg = _split_config_from_params(params)
        _split_records, split_table = make_train_val_test_split(
            df.to_dict("records"),
            **normalize_split_strategy_kwargs(split_cfg),
        )
        assignment = dict(
            zip(
                split_table["image_id"].astype(str),
                split_table["export_split"].astype(str),
                strict=False,
            )
        )
        df["export_split"] = df["image_id"].astype(str).map(assignment).fillna(df.get("export_split", "train"))
    except Exception:
        # Invalid exact-count settings are displayed in the split summary. Keep
        # the last loaded assignment so image preview remains usable meanwhile.
        pass
    split = str(params.get("filter_split") or "all")
    if split != "all" and "export_split" in df:
        df = df[df["export_split"] == split]
    if str(params.get("filter_positive") or "positive only") == "positive only" and "has_mass" in df:
        df = df[df["has_mass"] == True]  # noqa: E712
    vendors = params.get("filter_vendors") or []
    if str(params.get("filter_vendor_mode") or "all vendors") == "selected vendors" and vendors and "vendor" in df:
        df = df[df["vendor"].isin(vendors)]
    return df.reset_index(drop=True)


def _prepare_whole_image_sample(dataset: VindrMammoDataset, record_index: int) -> dict[str, Any]:
    record = dataset.image_records[int(record_index)]
    image_t, target = dataset._read_preprocessed_record_no_square(record)
    image = image_t.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
    mass_boxes = target["mass"]["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
    all_boxes = target["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
    height, width = image.shape
    meta_records = target.get("metadata", []) or []
    meta_first = meta_records[0] if meta_records else {}
    summary = {
        "image_id": target.get("image_id"),
        "study_id": target.get("study_id"),
        "split": target.get("split"),
        "laterality": target.get("laterality"),
        "view_position": target.get("view_position"),
        "num_masses": int(target.get("num_masses", 0)),
        "breast_birads": target.get("breast_birads"),
        "breast_density": target.get("breast_density"),
        "preprocessing": target.get("preprocessing", {}),
        "dicom_meta": target.get("dicom_meta", {}),
        "metadata": meta_first,
    }
    title = (
        f"image_id={summary.get('image_id')} | study={summary.get('study_id')} | "
        f"{summary.get('laterality')}-{summary.get('view_position')} | masses={summary.get('num_masses')} | "
        f"vendor={_vendor_from_summary(summary)}"
    )
    return {
        "image": image,
        "mass_boxes": mass_boxes,
        "all_boxes": all_boxes,
        "record": record,
        "target_summary": summary,
        "title": title,
        "crops": [{"window": (0, 0, width, height), "max_visibility": 1.0, "positive_by_slider": bool(len(mass_boxes))}],
        "failed_crops": [],
        "selected_crop": {"window": (0, 0, width, height), "max_visibility": 1.0, "positive_by_slider": bool(len(mass_boxes))},
        "showing_failed_crop": False,
        "crop_image": image,
        "crop_boxes": all_boxes,
        "crop_mass_boxes": mass_boxes,
        "foreground_mask_crop": None,
        "show_foreground_mask_preview": False,
        "contralateral_crop_image": None,
        "contralateral_info": {"requested": False, "found": False, "mode": "whole_image_preview"},
        "record_index": int(record_index),
    }


def _cached_preview_sample(
    dataset: VindrMammoDataset,
    record_index: int,
    params: dict[str, Any],
    crop_controls: dict[str, Any],
    *,
    crop_index: int,
    need_contralateral: bool,
) -> dict[str, Any]:
    key_payload = {
        "data_root": str(dataset.data_root),
        "record_index": int(record_index),
        "output_size": dataset.output_size,
        "preprocess_options": dataset.preprocess_options,
        "view_geometry": params.get("view_geometry"),
        "crop_index": int(crop_index),
        "crop_controls": _preview_cache_safe_crop_controls(crop_controls),
        "need_contralateral": bool(need_contralateral),
    }
    key = json.dumps(_jsonable(key_payload), sort_keys=True)
    cached = PREVIEW_SAMPLE_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    if str(params.get("view_geometry") or "crop") == "whole":
        result = _prepare_whole_image_sample(dataset, int(record_index))
    else:
        result = _prepare_sample(
            dataset,
            int(record_index),
            crop_controls,
            crop_index=int(crop_index),
            need_contralateral=bool(need_contralateral),
        )
    if len(PREVIEW_SAMPLE_CACHE) > 12:
        PREVIEW_SAMPLE_CACHE.pop(next(iter(PREVIEW_SAMPLE_CACHE)))
    PREVIEW_SAMPLE_CACHE[key] = copy.deepcopy(result)
    return result


def _preview_cache_safe_crop_controls(crop_controls: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "crop_size", "stride", "mode", "random_preview_count", "random_seed",
        "only_mass_crops", "positivity_threshold", "require_foreground",
        "min_foreground_fraction", "foreground_threshold", "bbox_safe_boundary_margin_fraction",
        "bbox_safe_random_shift_fraction", "bbox_safe_candidate_count", "bbox_safe_top_k",
        "bbox_safe_breast_bias_strength", "bbox_safe_left_bias_strength",
        "bbox_safe_projection_bias_strength", "crop_options", "contralateral_source_alignment",
    ]
    return {key: crop_controls.get(key) for key in keys}


def _render_single(dataset: VindrMammoDataset, records: pd.DataFrame, params: dict[str, Any]) -> Any:
    filtered = _filter_records(records, params)
    if filtered.empty:
        return html.Div("No images match the current filters.", className="warning note")
    row = filtered.iloc[int(params.get("image_index") or 0) % len(filtered)]
    crop_controls = _crop_controls_from_params(params)
    pipeline = _preview_pipeline_from_params(params)
    record_index = int(row["record_index"])
    if str(params.get("view_geometry") or "crop") == "whole":
        result = _cached_preview_sample(dataset, record_index, params, crop_controls, crop_index=0, need_contralateral=False)
    else:
        result = _cached_preview_sample(dataset, record_index, params, crop_controls, crop_index=int(params.get("crop_index") or 0), need_contralateral=_pipeline_uses_contralateral(pipeline))
    return _sample_view(result, pipeline, params, title=f"Single image: {int(params.get('image_index') or 0) % len(filtered) + 1} of {len(filtered)}")


def _render_comparison(dataset: VindrMammoDataset, records: pd.DataFrame, params: dict[str, Any]) -> Any:
    n_slots = max(2, min(10, int(params.get("comparison_slots") or 5)))
    crop_controls = _crop_controls_from_params(params)
    pipeline = _preview_pipeline_from_params(params)
    vendors = _default_comparison_vendors(records, n_slots)
    cards = []
    for i in range(n_slots):
        slot_params = dict(params)
        if vendors:
            slot_params["filter_vendor_mode"] = "selected vendors"
            slot_params["filter_vendors"] = [vendors[i % len(vendors)]]
        filtered = _filter_records(records, slot_params)
        if filtered.empty:
            cards.append(html.Div(f"Slot {i + 1}: no matching images", className="warning note"))
            continue
        row = filtered.iloc[int(params.get("image_index") or 0) % len(filtered)]
        record_index = int(row["record_index"])
        if str(params.get("view_geometry") or "crop") == "whole":
            result = _cached_preview_sample(dataset, record_index, params, crop_controls, crop_index=0, need_contralateral=False)
        else:
            result = _cached_preview_sample(dataset, record_index, params, crop_controls, crop_index=int(params.get("crop_index") or 0), need_contralateral=_pipeline_uses_contralateral(pipeline))
        cards.append(html.Div([html.Div(f"Slot {i + 1}", className="note"), _sample_view(result, pipeline, params, compact=True)]))
    return html.Div([html.Div(f"Comparison uses {n_slots} slots. Default vendor rotation: {', '.join(vendors) if vendors else 'none'}", className="note"), html.Div(className="comparison-grid", children=cards)])


def _sample_view(result: dict[str, Any], pipeline: dict[str, Any], params: dict[str, Any], *, title: str = "", compact: bool = False) -> Any:
    full = result["image"]
    crop = result.get("crop_image")
    if crop is None:
        full_gray = _to_uint8_percentile(full, _display_window(params))
        return html.Div([html.Div(result.get("title", title), className="note"), _image_card("Full image", _draw_boxes(_gray_to_rgb(full_gray), result.get("mass_boxes")), "No crop could be prepared.")])
    show_annotations = _is_on(params.get("show_annotations"))
    crop_boxes = result["crop_mass_boxes"] if show_annotations else None
    full_boxes = result["mass_boxes"] if show_annotations else None
    selected = result.get("selected_crop") or {}
    window = selected.get("window")
    processed_rgb, processing_meta = apply_channel_pipeline(
        crop,
        pipeline,
        source_crops=_source_crops_from_result(result),
        source_full_images=_source_full_images_from_result(result),
        crop_window=window,
        cache_namespace=f"preview:{result.get('record_index', 0)}",
    )
    visible = params.get("visible_channels") or CHANNELS
    processed_display = _mask_rgb_channels(processed_rgb, visible)
    full_gray = _to_uint8_percentile(full, _display_window(params))
    crop_gray = _to_uint8_percentile(crop, _display_window(params))
    full_draw = _draw_boxes(_gray_to_rgb(full_gray), full_boxes)
    if window is not None:
        full_draw = _draw_rect(full_draw, window, color=(80, 255, 80), thickness=max(2, full.shape[1] // 1000))
    crop_draw = _draw_boxes(_gray_to_rgb(crop_gray), crop_boxes)
    proc_draw = _draw_boxes(processed_display.copy(), crop_boxes)
    whole_mode = str(params.get("view_geometry") or "crop") == "whole"
    pieces = [
        html.Div(result.get("title", title), className="note"),
        html.Div(className="image-grid", children=[
            _image_card("Source after shared preprocessing", full_draw, "Whole working image." if whole_mode else "Green box is the selected crop window."),
            _image_card("Model working view", crop_draw, "Whole image resized preview." if whole_mode else f"Window: {tuple(int(v) for v in window) if window is not None else 'n/a'}"),
            _image_card("Processed RGB output", proc_draw, f"Visible channels: {''.join(visible) or 'none'}"),
        ]),
    ]
    if _is_on(params.get("show_channel_panels")):
        pieces.append(html.Div(className="image-grid", children=[
            _image_card(f"{ch} channel", _draw_boxes(_gray_to_rgb(processed_rgb[..., idx]), crop_boxes), OP_HELP.get(_channel_steps(pipeline, ch)[-1]["op"], "") if _channel_steps(pipeline, ch) else "")
            for idx, ch in enumerate(CHANNELS)
        ]))
    if result.get("foreground_mask_crop") is not None and result.get("show_foreground_mask_preview"):
        pieces.append(_image_card("Foreground mask", result["foreground_mask_crop"].astype(np.uint8) * 255, f"Foreground fraction: {selected.get('foreground_fraction')}"))
    if not compact:
        stats = _stats_table(full, crop, processed_rgb)
        meta = _compact_metadata(result["target_summary"], processing_meta)
        if result.get("contralateral_info"):
            meta["contralateral_source"] = result.get("contralateral_info")
        pieces.append(html.Details(open=True, children=[html.Summary("Metadata and statistics"), _table(stats), html.Pre(json.dumps(_jsonable(meta), indent=2))]))
    return html.Div(pieces)


def _render_saved_dataset(params: dict[str, Any], cfg: dict[str, Any]) -> Any:
    root = str(params.get("saved_root") or cfg.get("paths", {}).get("output_root", ""))
    loaded = _load_saved_dataset_viewer_index(root, 0)
    if not loaded.get("ok"):
        return html.Div(str(loaded.get("error", "Could not load saved dataset.")), className="error note")
    rows = pd.DataFrame(loaded.get("rows", []))
    if rows.empty:
        return html.Div("Saved dataset table is empty.", className="warning note")
    view = rows.copy()
    split = str(params.get("saved_split") or "all")
    if split != "all":
        view = view[view["split"].astype(str) == split]
    pos = str(params.get("saved_positive") or "all")
    if pos == "positive only":
        view = view[view["positive"]]
    elif pos == "empty only":
        view = view[~view["positive"]]
    search = str(params.get("saved_search") or "").strip().casefold()
    if search:
        hay = (view.get("source_image_id", "").astype(str) + " " + view.get("source_index", "").astype(str) + " " + view.get("file_name", "").astype(str)).str.casefold()
        view = view[hay.str.contains(search, regex=False, na=False)]
    if _is_on(params.get("saved_existing_only")):
        view = view[view["image_exists"].astype(bool)]
    view = view.reset_index(drop=True)
    metrics = html.Div(className="metric-row", children=[
        _metric("Crops", len(rows)),
        _metric("Positive", int(rows["positive"].sum())),
        _metric("Empty", int((~rows["positive"]).sum())),
        _metric("Images found", int(rows["image_exists"].sum())),
        _metric("Filtered", len(view)),
    ])
    if view.empty:
        return html.Div([metrics, html.Div("No saved crops match current filters.", className="warning note")])
    idx = int(params.get("saved_index") or 0) % len(view)
    row = view.iloc[idx]
    image_path = Path(str(row.get("image_path", "")))
    label_path = Path(str(row.get("label_path", "")))
    image = _load_saved_viewer_image(image_path)
    if image is None:
        return html.Div([metrics, html.Div(f"Could not read image: {image_path}", className="error note")])
    boxes = _load_yolo_boxes_for_saved_image(label_path, width=int(image.shape[1]), height=int(image.shape[0]))
    display = _prepare_saved_viewer_display_image(image, boxes if _is_on(params.get("saved_show_boxes")) else np.zeros((0, 4)), row, idx, len(view))
    metadata_fields = ["split", "viewer_row", "source_index", "source_image_id", "file_name", "has_mass", "is_positive_window", "num_mass_boxes", "crop_mode", "crop_window_xyxy", "image_path", "label_path"]
    shown = {k: _streamlit_json_safe(row.get(k)) for k in metadata_fields if k in row.index}
    return html.Div([metrics, html.Div(className="image-grid", children=[_image_card("Saved crop", display, _saved_viewer_caption(row, idx, len(view))), html.Div(className="image-card", children=[html.H3("Metadata"), html.Pre(json.dumps(_jsonable(shown), indent=2))])])])


def _render_visualizations(cfg: dict[str, Any]) -> Any:
    paths = cfg.get("paths", {}) or {}
    output = Path(str(paths.get("output_root", "")))
    vis_dir = output / "visualizations"
    if not vis_dir.exists():
        return html.Div(f"No visualization directory found at {vis_dir}. Run the visualization CLI/export first.", className="warning note")
    files = sorted([p for p in vis_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv", ".json"}])[:80]
    cards = []
    for p in files:
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            cards.append(_image_card(p.name, np.asarray(Image.open(p).convert("RGB")), str(p)))
        else:
            preview = p.read_text(encoding="utf-8", errors="replace")[:4000]
            cards.append(html.Details(children=[html.Summary(str(p.relative_to(vis_dir))), html.Pre(preview)]))
    return html.Div(cards or [html.Div(f"No displayable visualization files found in {vis_dir}", className="warning note")])


def _render_manifest_tools(params: dict[str, Any], cfg: dict[str, Any]) -> Any:
    path_text = str(params.get("manifest_paths") or "")
    paths = [Path(line.strip()).expanduser() for line in path_text.splitlines() if line.strip()]
    if not paths:
        return html.Div("Enter manifest/config paths in the Manifests tab, one per line.", className="note")
    rows = []
    details = []
    for path in paths:
        if not path.exists():
            rows.append({"source": str(path), "status": "missing"})
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
            rows.append({"source": str(path), "status": "ok", "top_level_keys": ", ".join(sorted(map(str, data.keys()))) if isinstance(data, dict) else type(data).__name__})
            details.append(html.Details(children=[html.Summary(path.name), html.Pre(yaml.safe_dump(_make_yaml_safe(data), sort_keys=False, width=120)[:12000])]))
        except Exception as exc:
            rows.append({"source": str(path), "status": f"error: {exc}"})
    return html.Div([_table(pd.DataFrame(rows)), html.Div(details)])


def _build_export_cfg_from_params(cfg: dict[str, Any], records: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    crop_controls = _crop_controls_from_params(params)
    preset_key = str(((cfg.get("study_preset_provenance", {}) or {}).get("preset_key", "")))
    selected_vendors = params.get("export_vendors") or []
    split_modes = {
        "train": str(params.get("train_crop_mode") or "deterministic"),
        "val": str(params.get("val_crop_mode") or "deterministic"),
        "test": str(params.get("test_crop_mode") or "deterministic"),
    }
    balance_mode = str(params.get("export_balance_mode") or "positive_ratio")
    if balance_mode not in {
        "positive_ratio", "negative_fraction", "source_breast_ratio", "mass_only", "all"
    }:
        balance_mode = "positive_ratio"
    if balance_mode == "source_breast_ratio":
        # This policy is defined on a complete deterministic candidate grid.
        split_modes = {split: "deterministic" for split in ["train", "val", "test"]}
    try:
        target_ratio = min(max(float(params.get("export_target_positive_ratio") or 0.50), 0.01), 1.0)
    except Exception:
        target_ratio = 0.50
    try:
        negative_keep_fraction = min(max(float(params.get("export_negative_keep_fraction") or 0.20), 0.0), 1.0)
    except Exception:
        negative_keep_fraction = 0.20
    deterministic_selection = {
        split: {
            "mode": (
                "all"
                if preset_key == SIMPLE_PRESET_KEY and split in {"val", "test"}
                else (
                    (balance_mode if split == "train" else "all")
                    if balance_mode in {"negative_fraction", "source_breast_ratio"}
                    else balance_mode
                )
            ),
            "target_positive_ratio": target_ratio,
            "negative_keep_fraction": negative_keep_fraction,
        }
        for split in ["train", "val", "test"]
    }
    out = _build_gui_export_config(
        cfg=_cfg_with_preprocess(cfg, params),
        output_root=Path(str(params.get("export_parent") or ".")) / str(params.get("export_name") or "preprocessed-vindr-gui"),
        clean_output=_is_on(params.get("clean_output")),
        selected_vendors=selected_vendors if params.get("export_vendor_mode") == "selected vendors only" else [],
        deterministic_selection=deterministic_selection,
        split_crop_modes=split_modes,
        save_square=str(params.get("view_geometry") or "crop") != "whole",
        save_baseline=str(params.get("view_geometry") or "crop") == "whole",
        crop_controls=crop_controls,
        pipeline=_pipeline_from_params(params),
        simple_profiler_enabled=True,
        simple_profiler_emit_every=10,
    )
    configured_scheme = str(((cfg.get("image_export", {}) or {}).get("rgb_scheme", ""))).casefold().strip()
    selected_splits = _split_config_from_params(params)
    original_splits = dict(cfg.get("splits", {}) or {})
    split_changed = _split_signature(selected_splits) != _split_signature(original_splits)
    out["splits"] = selected_splits
    if split_changed:
        provenance = out.setdefault("study_preset_provenance", {})
        assumptions = provenance.setdefault("assumptions", {})
        assumptions["split_override"] = {
            "reason": "user-selected GUI training/validation policy",
            "paper_disclosed": False,
            **copy.deepcopy(selected_splits),
        }
        contract = out.setdefault("replication_contract", {})
        if preset_key == PAPER_69_PRESET_KEY and selected_splits["strategy"] == "official_only":
            contract.update({
                "enabled": True,
                "strict": True,
                "name": "paper69_vindr_official_full_image_mass_v1",
                "preserve_official_test": True,
                "require_positive_source_images": False,
                "expected_source_images": {"train": 16000, "val": 0, "test": 4000},
                "expected_source_studies": {"train": 4000, "val": 0, "test": 1000},
                "expected_source_annotations": {"test": 237},
            })
        elif bool(contract.get("enabled", False)):
            contract["enabled"] = False
            contract["disabled_reason"] = (
                "GUI split assignment differs from the preset's count-validated replication contract."
            )
    if (
        preset_key == PAPER_69_PRESET_KEY
        and configured_scheme in {"paper69_mammoclip_uint8", "mammoclip_uint8_replicated"}
        and str(params.get("pipeline_mode") or "yaml") != "visual"
    ):
        # Paper 69's exact replicated-uint8 encoding is not a custom operation
        # pipeline. Preserve it unless the user explicitly activates the visual
        # pipeline builder.
        out["image_export"] = copy.deepcopy(cfg.get("image_export", {}) or {})
    baseline = out.setdefault("baseline_uncropped", {})
    baseline["resize_mode"] = str(params.get("whole_resize_mode") or "none")
    baseline["target_width"] = int(params.get("whole_resize_width") or 1024)
    baseline["target_height"] = int(params.get("whole_resize_height") or 1024)
    baseline["pad_value"] = float(params.get("whole_pad_value") or 0.0)
    baseline["pad_anchor"] = str(params.get("whole_pad_anchor") or "left_top")
    paired = out.setdefault("paired_whole_images", {})
    paired_size = max(1, int(params.get("paired_whole_size") or 1024))
    paired["enabled"] = _is_on(params.get("paired_whole_enabled"))
    paired["target_width"] = paired_size
    paired["target_height"] = paired_size
    paired["canvas_mode"] = "per_image_square"
    paired["pad_value"] = 0.0
    paired["pad_anchor"] = "left_top"
    paired["storage_mode"] = "hardlink" if _is_on(params.get("paired_whole_hardlink")) else "copy"
    square = out.setdefault("square_crops", {})
    square["positive_fraction"] = float(target_ratio)
    square["deterministic_target_positive_ratio"] = float(target_ratio)
    square["online_positive_ratio_selection_for_random"] = True
    square["online_balance_shuffle_source_records"] = True
    square["global_positive_ratio_selection_for_random"] = False
    for split in ["train", "val", "test"]:
        square[f"{split}_positive_fraction"] = float(target_ratio)
        square[f"{split}_deterministic_target_positive_ratio"] = float(target_ratio)
        square[f"{split}_deterministic_selection_mode"] = (
            "all"
            if preset_key == SIMPLE_PRESET_KEY and split in {"val", "test"}
            else (
                (balance_mode if split == "train" else "all")
                if balance_mode in {"negative_fraction", "source_breast_ratio"}
                else balance_mode
            )
        )
        square[f"{split}_deterministic_negative_keep_fraction"] = float(negative_keep_fraction)
        square[f"{split}_online_positive_ratio_selection_for_random"] = True
    split_filter_params = {
        "train": ("require_foreground", "min_foreground_fraction"),
        "val": ("val_require_foreground", "val_min_foreground_fraction"),
        "test": ("test_require_foreground", "test_min_foreground_fraction"),
    }
    split_breast_filters: dict[str, tuple[bool, float]] = {}
    for split, (enabled_key, minimum_key) in split_filter_params.items():
        enabled_raw = params.get(enabled_key)
        enabled = (
            bool(square.get(f"{split}_require_min_breast_fraction_for_all_crops", False))
            if enabled_raw is None
            else _is_on(enabled_raw)
        )
        minimum_raw = params.get(minimum_key)
        minimum = float(
            square.get(f"{split}_min_breast_fraction_for_all_crops", 0.05)
            if minimum_raw is None
            else minimum_raw
        )
        split_breast_filters[split] = (enabled, minimum)
        square[f"{split}_require_min_breast_fraction_for_all_crops"] = enabled
        square[f"{split}_min_breast_fraction_for_all_crops"] = minimum
        square[f"{split}_breast_fraction_comparison_for_all_crops"] = (
            "strictly_greater_than"
        )
        square[f"{split}_require_retained_breast_mask_for_all_crops"] = enabled
    if balance_mode == "source_breast_ratio":
        out["source_cohort"] = {
            "finding_category": "Mass",
            "positive_images_only": True,
            "train_expand_to_all_patient_breast_views": True,
            "train_breast_status_unit": "study_laterality",
        }
        square["train_deterministic_target_source_breast_mass_ratio"] = float(target_ratio)
        square["train_deterministic_require_foreground"] = False
        square["train_negative_require_foreground"] = False
        for split in ["val", "test"]:
            square[f"{split}_deterministic_require_foreground"] = False
            square[f"{split}_negative_require_foreground"] = False
            square[f"{split}_require_clean_negative_windows"] = False
        contract = out.setdefault("replication_contract", {})
        if bool(contract.get("enabled", False)):
            contract["min_breast_fraction_strictly_greater_than_by_split"] = {
                split: minimum
                for split, (enabled, minimum) in split_breast_filters.items()
                if enabled
            }
        if preset_key != PAPER_22_IMPROVED_PRESET_KEY:
            out.setdefault("replication_contract", {})["enabled"] = False
            out["replication_contract"]["disabled_reason"] = (
                "Source-breast GUI mode was enabled outside the audited improved Paper 22 preset."
            )
    if preset_key == SIMPLE_PRESET_KEY:
        square["train_online_positive_ratio_selection_for_deterministic"] = True
        square["val_online_positive_ratio_selection_for_deterministic"] = False
        square["test_online_positive_ratio_selection_for_deterministic"] = False
    return out


def _display_window(params: dict[str, Any]) -> tuple[float, float]:
    lo = float(params.get("display_low") or 1.0)
    hi = float(params.get("display_high") or 99.0)
    if hi <= lo:
        hi = min(100.0, lo + 1.0)
    return lo, hi


def _image_card(title: str, image: np.ndarray, caption: str = "") -> html.Div:
    src = _image_src(image)
    return html.Div(
        className="image-card",
        children=[
            html.H3(title),
            html.Div(className="image-scroll", children=html.Img(src=src)),
            html.P(caption),
        ],
    )


def _image_src(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")


def _gray_to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        return arr
    return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)


def _table(df: pd.DataFrame) -> Any:
    if df.empty:
        return html.Div("No rows.", className="note")
    return html.Table(
        [html.Thead(html.Tr([html.Th(str(c)) for c in df.columns]))]
        + [html.Tr([html.Td(str(v)) for v in row]) for row in df.astype(str).to_numpy().tolist()],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"},
    )


def _metric(label: str, value: Any) -> html.Div:
    return html.Div(className="metric", children=[html.Strong(f"{value:,}" if isinstance(value, int) else str(value)), html.Span(label)])


def _summary_children(summary: dict[str, Any]) -> Any:
    return html.Div([html.Div(f"{k}: {v}") for k, v in summary.items()])


def _same_disk_for_queue_path(path: Any, device_id: int | None) -> bool:
    if not path:
        return False
    try:
        other = get_disk_space(str(path))
    except Exception:
        return False
    if device_id is None or other.device_id is None:
        return str(other.probe_path.anchor) == str(Path(str(path)).anchor)
    return int(other.device_id) == int(device_id)


def _estimate_summary_children(payload: dict[str, Any]) -> Any:
    if not payload:
        return html.Div("No estimate available.", className="note")
    assumptions = payload.get("assumptions", []) or []
    breakdown = payload.get("breakdown", {}) or {}
    return html.Div(
        className="summary-box",
        children=[
            html.H3(f"Conservative estimate: {payload.get('conservative', format_bytes(payload.get('estimated_bytes', 0)))}"),
            html.Div(
                f"Sources: {payload.get('source_image_count', 0):,} | crops: {payload.get('crop_image_count', 0):,} | "
                f"paired whole images: {payload.get('paired_whole_image_count', 0):,} | baseline: {payload.get('baseline_image_count', 0):,}"
            ),
            html.Ul([html.Li(f"{key.replace('_', ' ')}: {value}") for key, value in breakdown.items()]),
            html.Details(children=[html.Summary("Estimate assumptions"), html.Ul([html.Li(str(item)) for item in assumptions])]),
        ],
    )


def _queue_progress_text(job: dict[str, Any]) -> str:
    progress = job.get("progress", {}) or {}
    stage = str(progress.get("stage") or "")
    processed = progress.get("processed")
    total = progress.get("total")
    fraction = job.get("progress_fraction")
    parts = []
    if stage:
        parts.append(stage)
    if processed is not None and total is not None:
        parts.append(f"{processed}/{total}")
    if fraction is not None:
        parts.append(f"{float(fraction):.1%}")
    return " · ".join(parts) or "Waiting for progress details"


def _queue_table_children(snapshot: dict[str, Any]) -> Any:
    jobs = list(snapshot.get("jobs", []) or [])
    if not jobs:
        return html.Div("The extraction queue is empty.", className="note")
    rows = []
    for job in jobs:
        status = str(job.get("status", ""))
        error = str(job.get("error") or "")
        rows.append(html.Tr([
            html.Td(str(job.get("queue_position") or "—")),
            html.Td(str(job.get("name") or "")),
            html.Td(status),
            html.Td(format_bytes(int(job.get("estimated_bytes") or 0))),
            html.Td(str(job.get("output_root") or "")),
            html.Td(error if status == "failed" else _queue_progress_text(job)),
        ]))
    reserved = sum(
        int(job.get("estimated_bytes") or 0)
        for job in jobs
        if job.get("status") in {"queued", "running"}
    )
    return html.Div([
        html.Div(
            f"{len(jobs)} retained item(s); queued/running conservative total {format_bytes(reserved)}. "
            f"Worker {'running' if snapshot.get('started') else 'not started'}.",
            className="note",
        ),
        html.Table(
            className="queue-table",
            children=[
                html.Thead(html.Tr([html.Th("#"), html.Th("Pipeline"), html.Th("Status"), html.Th("Estimate"), html.Th("Output"), html.Th("Progress / error")])),
                html.Tbody(rows),
            ],
        ),
    ])


def _initial_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    image_export = cfg.get("image_export", {}) or {}
    pipeline = image_export.get("custom_channel_pipeline")
    if isinstance(pipeline, dict) and pipeline:
        return copy.deepcopy(pipeline)
    return copy.deepcopy(LITERATURE_PIPELINE_PRESETS["raw_clahe_detail"]["pipeline"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    main()
