# Noise Signal Classification

This project demonstrates how to build a simple signal classifier in PyTorch that can
separate pure noise from structured waveforms. It also includes a lightweight model
registry for keeping track of multiple trained checkpoints.

## Project layout

```
noise_classifier/
├── datasets.py      # Synthetic dataset generation utilities
├── models.py        # 1D CNN architecture for classification
├── registry.py      # JSON-backed model registry and CLI helper
└── train.py         # Training loop and command line interface
```

## Installation

Create a virtual environment and install the dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install torch
```

The code uses only the PyTorch core package so there are no additional dependencies.

## Training a model

Run the training script with default hyperparameters:

```
python -m noise_classifier.train --epochs 5 --num-samples 4096
```

Command line arguments let you control the dataset size, signal length, learning rate,
and more. The script prints progress for each epoch, saves the best model checkpoint, and
stores metadata alongside the weights.

Artifacts are saved inside the directory passed to `--output-dir` (defaults to `./models`).
Each run generates:

- `<name>.pt`: serialized model weights (`state_dict`).
- `<name>.json`: metadata containing training configuration and metrics.
- `registry.json`: persistent registry tracking every trained model.

## Managing trained models

The registry can be inspected or pruned with the CLI provided in `registry.py`:

```
python -m noise_classifier.registry models --list
python -m noise_classifier.registry models --remove noise_classifier_20240101_120000
```

## Next steps

- Replace the synthetic dataset with your real signal data by creating a custom
  `torch.utils.data.Dataset` implementation.
- Extend the registry with richer metadata, tagging, or automated evaluation scripts.
- Integrate the code into a larger application or API for classifying live audio streams.
