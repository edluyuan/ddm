# DDDM 2D Toy Implementation

This repository contains a minimal PyTorch implementation of **Distributional
Denoising Diffusion Models** adapted for simple two dimensional datasets.  The
code is refactored into a small Python package and provides utilities for
training, sampling and evaluation on the Gaussian mixture benchmark from the
paper.

## Usage

```
python run_example.py --epochs 1000 --steps 20 --device cpu
```

This trains a model on the 2D Gaussian mixture, draws samples and reports the
RBF MMD$^2$ metric.

## Structure

- `dddm/`: package containing model, training and sampling utilities.
- `run_example.py`: script that trains the model and evaluates it.

## Requirements

- PyTorch
- Matplotlib (for sample visualization)

Install with:

```
pip install torch matplotlib
```
