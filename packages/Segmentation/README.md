# Segmentation

Orchid organ segmentation package based on a DINOv3 ViT-L backbone. This project provides training, inference, and evaluation pipelines for multi-class semantic segmentation.

## Overview

The current pipeline targets an 8-class segmentation task:

- Background
- Column
- Dorsal Sepal
- Labellum
- Lateral Sepal
- Petal
- Petal_L
- Petal_R

Main scripts:

- `fine-tuning.py`: train the segmentation model
- `inference.py`: run prediction and generate visualizations
- `evaluation.py`: compute metrics and confusion matrices

## Environment

Required packages are defined in `pyproject.toml` and `requirements.txt`.

Core dependencies:

- Python 3.12+
- PyTorch
- torchvision
- transformers
- numpy
- pillow
- opencv-python
- matplotlib
- seaborn
- pandas
- tqdm

## Project Structure

- `Datasets/`: training and validation data
- `Inference/`: test inputs and inference outputs
- `Evaluation/`: evaluation outputs
- `training_result/`: checkpoints and CSV logs
- `models/`: model assets and weights
- `fine-tuning.py`: training pipeline
- `inference.py`: inference and visualization pipeline
- `evaluation.py`: metric computation pipeline

## Data Format

Expected dataset layout:

- `Datasets/<dataset_name>/train/images`
- `Datasets/<dataset_name>/train/masks`
- `Datasets/<dataset_name>/val/images`
- `Datasets/<dataset_name>/val/masks`

Rules:

- images: `.jpg` or `.png`
- masks: `.png`
- mask pixel values must match the class indices used in training

## Training

Run training from the `Segmentation` directory:

```bash
uv run fine-tuning.py
```

Training behavior:

- DINOv3 ViT-L backbone with a lightweight segmentation head
- synchronized image/mask transforms
- color jitter and light Gaussian blur for image augmentation
- validation-based checkpoint saving
- early stopping on validation mIoU

Training outputs:

- `params.csv`: training configuration
- `log.csv`: per-epoch metrics
- `best_finetune_LandR.pth`: best checkpoint by validation mIoU

## Inference

Run inference from the `Segmentation` directory:

```bash
uv run inference.py
```

Inference outputs:

- predicted class mask
- colorized segmentation map
- overlay visualization
- saved 3-panel figure for each image

Input folder:

- `Inference/test_datasets`

Output folder:

- `Inference/7classes_vitl_layer-1_LandR_v3_argumented`

## Evaluation

Run evaluation from the `Segmentation` directory:

```bash
uv run evaluation.py
```

Evaluation outputs:

- global pixel accuracy
- mean IoU
- mean accuracy
- per-class IoU
- per-class accuracy
- confusion matrix

## Configuration Notes

Important settings in `fine-tuning.py`:

- `num_classes = 8`
- `patch_size = 16`
- `img_size = 2400`
- `extract_layers = [-1]`
- `early_stopping_patience = 15`
- `early_stopping_min_delta = 5e-4`

Important settings in `inference.py`:

- class colors must match the training label mapping
- checkpoint path must point to the best saved model
- test images must be readable `.jpg` or `.png` files

## Augmentation Policy

Current training augmentations:

- random horizontal flip
- random rotation
- color jitter
- light Gaussian blur

Not used:

- random scaling
- random crop
- translation

These geometric transforms are excluded because they can move target organs out of frame or clip boundaries.

## Metrics

The training loop monitors:

- `CrossEntropyLoss` for optimization
- validation mIoU for checkpoint selection
- early stopping to prevent unnecessary epochs

## Troubleshooting

If the predicted colors look wrong:

- verify that the class index mapping matches the checkpoint
- verify the color table in `inference.py`

If training masks look inconsistent:

- confirm that image and mask transforms are synchronized
- confirm that mask values are valid class indices

If checkpoint loading fails:

- confirm the checkpoint path
- confirm that the model configuration matches the saved training run

## License

No license file is currently included in this repository.
