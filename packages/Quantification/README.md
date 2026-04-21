# Quantification

Analysis package for post-processing and quantification of orchid-related research data. This package currently contains notebook-based workflows for color analysis, flower separation, and geometric alignment experiments.

## Overview

This package is organized around exploratory analysis notebooks rather than a single training script.

Current notebooks:

- `ColorQuantify.ipynb`: color-based quantification workflow
- `floweral_seperate.ipynb`: flower separation or segmentation-related analysis
- `main-quantification.ipynb`: main quantification workflow
- `procruste.ipynb`: Procrustes-style shape alignment or comparison analysis

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
- pandas
- tqdm

## Project Structure

- `Datasets/`: input data used by notebooks
- `ColorQuantify.ipynb`: color quantification notebook
- `floweral_seperate.ipynb`: flower separation notebook
- `main-quantification.ipynb`: main analysis notebook
- `procruste.ipynb`: shape alignment notebook

## Typical Workflow

1. Prepare the target dataset under `Datasets/`.
2. Open the notebook that matches the analysis task.
3. Run preprocessing and visualization cells.
4. Export measurements, charts, or tables as needed.

## Analysis Tasks

Common tasks covered by this package include:

- color quantification
- flower separation and mask-driven analysis
- geometric alignment or comparison
- plotting and tabular summary generation

## Usage Notes

- This package is notebook-first.
- No single CLI entry point is defined at the moment.
- Keep notebook inputs and output paths consistent across runs.

## Data Notes

The notebooks expect data to be organized in or under `Datasets/`.

If you add new notebooks or scripts, keep their input and output conventions documented in this README.

## License

No license file is currently included in this repository.
