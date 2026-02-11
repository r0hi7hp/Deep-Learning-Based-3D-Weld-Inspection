# PointNet++ with Data Augmentation

This directory contains the PointNet++ implementation configured for training on augmented datasets.

## Overview

This version of the model is intended to be trained on a dataset that includes augmented samples (e.g., rotated views) to improve model robustness and generalization.

## Usage

### Training

To train the model on the augmented dataset:

```bash
python train_pointnetpp.py --batch-size 4 --epochs 100
```

The script will automatically look for an `Augmented Data` folder within the dataset directory.

**Arguments:**
- `--batch-size`: Batch size (default: 4).
- `--epochs`: Number of epochs (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
