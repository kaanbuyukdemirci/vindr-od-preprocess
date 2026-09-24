# Global-padded multi-resolution VinDr export

Run `scripts/export_global_padded_multires.py` to build one source-faithful,
multi-resolution dataset directly from the official VinDr DICOM mammograms.

The command scans the DICOM `Rows` and `Columns` headers before decoding pixels.
For the current 20,000-image source, the largest native shape is 3580 x 2812 HxW,
so both dimensions are rounded upward to the common 3584 x 2816 canvas. Every
image is anchored at the top-left; zeros are added only on the right and bottom.

Compact images are made by resizing the complete common canvas at one uniform
aspect ratio, followed by the minimum extra right padding required to make both
output dimensions divisible by 32:

| Variant | Output HxW | Resized content HxW | Extra right pad |
|---|---:|---:|---:|
| Padded original | 3584 x 2816 | 3584 x 2816 | 0 |
| 512 | 512 x 416 | 512 x 402 | 14 |
| 640 | 640 x 512 | 640 x 503 | 9 |
| 800 | 800 x 640 | 800 x 629 | 11 |
| 1024 | 1024 x 832 | 1024 x 805 | 27 |
| 1280 | 1280 x 1024 | 1280 x 1006 | 18 |
| 1600 | 1600 x 1280 | 1600 x 1257 | 23 |

The 2048 tier is intentionally excluded from the default run. Its optional geometry
is 2048 x 1632 HxW with 2048 x 1609 resized content and 23 pixels of final right
padding.

Every variant, including the padded original, is saved only as a contiguous
single-channel CHW `torch.float32` tensor in `[0,1]`. No PNG and no unpadded image
copy is saved. The exporter applies the DICOM modality LUT, does not apply a
display/VOI LUT, inverts MONOCHROME1 sources to a consistent black-background
convention, and min-max normalizes each source. It does not crop, mask, mirror, or
augment the mammograms.

Mass boxes come from the official finding CSV. Mass-positive images have YOLO label
files; Mass-negative images omit the empty file because Ultralytics already treats
a missing label as background. This saves about 20 GB on the T9's 128 KiB allocation
units. Each variant also has consolidated COCO JSON. The preserved
`/mnt/t9/vindr-data/vindr/global_pad_split_assignments.csv` map is used so
train/val/test membership stays consistent with prior experiments.

## Start the complete run

From the repository root:

```bash
PYTHONPATH=src /home/kaan/anaconda3/envs/data/bin/python \
  scripts/export_global_padded_multires.py \
  --workers 8
```

The default output is:

```text
/mnt/t9/vindr-data/preprocessed-vindr-global-pad-multires-v1
```

The terminal displays progress, rate, elapsed time, and ETA for both the DICOM
header scan and pixel export. The default float32 payload without 2048 is about
1.228 TB, with an estimated T9 allocation around 1.245 TB after tensor-container,
positive-label, and metadata overhead. The 2048 tier adds about 267.4 GB of payload
or roughly 270 GB of T9 allocation when added later.

The run is resumable. If it stops, rerun the exact same command. Completed atomic
float32 and label paths are skipped. Use `--overwrite` only when every output should
be regenerated.

To inspect geometry and available capacity without exporting mammogram pixels:

```bash
PYTHONPATH=src /home/kaan/anaconda3/envs/data/bin/python \
  scripts/export_global_padded_multires.py --scan-only
```

For a small end-to-end trial that retains the full-dataset canvas geometry:

```bash
PYTHONPATH=src /home/kaan/anaconda3/envs/data/bin/python \
  scripts/export_global_padded_multires.py \
  --limit 8 \
  --output-root /mnt/t9/vindr-data/preprocessed-vindr-global-pad-smoke
```

To add 2048 later, rerun against the production output with all heights included:

```bash
PYTHONPATH=src /home/kaan/anaconda3/envs/data/bin/python \
  scripts/export_global_padded_multires.py \
  --workers 8 \
  --target-heights 512 640 800 1024 1280 1600 2048 \
  --allow-low-space
```

Existing atomic float32 tensors and labels are skipped. Before using
`--allow-low-space`, confirm that at least 270 GB is free for the new 2048 branch.

## Dash GUI

The Dash preprocessing studio now includes a **Global Multi-Res** tab. It uses the
same defaults as the command above and provides:

- source, split-assignment, and output path controls;
- 512, 640, 800, 1024, 1280, and 1600 selected by default, with 2048 optional;
- dense float32 payload and output-shape previews from the scanned geometry;
- a copyable equivalent terminal command;
- start/resume controls with overwrite, rescan, and low-space options; and
- read-only live process, manifest progress, elapsed-time, ETA, free-space, and log
  monitoring.

Start the GUI in an environment containing both the project imaging dependencies
and Dash. On this workstation:

```bash
PYTHONPATH=src /home/kaan/anaconda3/envs/data-mmdet/bin/python \
  -m vindr_mammo.dash_app \
  --config config/export_config.yaml
```

Opening or closing the GUI does not affect an exporter started in a terminal. For
an active exporter using the selected output root, the tab detects its PID from
`metadata/status.json`, disables the start button, and monitors it without sending
signals. There is deliberately no Stop button. GUI-started runs are detached and
write their terminal output to `metadata/gui_export.log`, so they also continue if
the browser or GUI server is closed.
