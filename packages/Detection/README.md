# Detection

Flower detection package built on Ultralytics YOLO. This package is used to train a detector, export it to ONNX, and run a YOLO + SAM2 segmentation workflow.

## Overview

Main scripts:

- `train.py`: train a YOLO detector on the flower dataset
- `toOnnx.py`: export trained weights to ONNX
- `yolo_sam2.py`: run YOLO detection followed by SAM2 mask extraction and visualization

The current workflow is designed for:

- flower detection with YOLO
- instance-level mask extraction with SAM2
- per-image result folders containing overlay and masked outputs

## Environment

Required packages are defined in `pyproject.toml` and `requirements.txt`.

Core dependencies:

- Python 3.12+
- PyTorch
- ultralytics
- numpy
- opencv-python

## Project Structure

- `datasets/`: training and test data for detection
- `train.py`: YOLO training entry point
- `yolo_sam2.py`: detection + segmentation inference pipeline
- `toOnnx.py`: ONNX export script
- `trainging_results/`: model training outputs

## Dataset Format

The training script expects a YOLO-style dataset configuration file:

- `datasets/flower.yaml`

The dataset should contain image and label files in the structure required by Ultralytics YOLO.

## Training

Run training from the `Detection` directory:

```bash
uv run train.py
```

Training behavior:

- uses `yolo11n.pt` as the base model
- trains at higher image resolution to improve small-object detection
- enables augmentation such as mosaic, flips, HSV jitter, and shear
- saves results under the configured project name

Key training settings in `train.py`:

- `epochs = 200`
- `imgsz = 960`
- `batch = 8`
- `device = '2,3'`
- `patience = 100`

## Inference Workflow

Run the combined YOLO + SAM2 pipeline from the `Detection` directory:

```bash
uv run yolo_sam2.py
```

This script:

- loads YOLO weights
- detects flower instances
- filters boxes near image borders
- runs SAM2 using each remaining box
- writes visual outputs into a per-image result folder

## ONNX Export

Export a trained detector to ONNX from the `Detection` directory:

```bash
uv run toOnnx.py
```

This is useful for deployment or downstream integration with non-PyTorch runtimes.

## Outputs

Typical generated artifacts include:

- YOLO training runs and weights
- ONNX export files
- per-image segmentation folders from the YOLO + SAM2 pipeline

## Configuration Notes

Important parameters in `train.py`:

- dataset path: `datasets/flower.yaml`
- model: `yolo11n.pt`
- image size: `960`
- augmentation policy: mosaic, flips, HSV jitter, shear

Important parameters in `yolo_sam2.py`:

- YOLO checkpoint path must point to a valid `best.pt`
- `TEST_DIR` must contain readable `.jpg`, `.png`, or similar image files
- SAM2 device selection must match the available CUDA devices

## Troubleshooting

If training does not start:

- confirm the YOLO dataset YAML path
- confirm the dataset folders exist
- confirm the selected CUDA devices are available

If SAM2 inference fails:

- confirm YOLO weights are readable
- confirm SAM2 weights can be downloaded or loaded
- confirm the input images are readable by OpenCV

## License

No license file is currently included in this repository.
