# Hybrid Model with Data Augmentation

This directory contains the Hybrid Model implementation situated in the augmented data folder.

## Overview

This version of the model is intended to be used with augmented datasets. While the code is similar to the standard version, it serves as the designated location for experimentation with data augmentation techniques.

## Usage

### Training

To train the model:

```bash
python train_hybrid.py --batch-size 4 --epochs 100
```

**Arguments:**
- `--batch-size`: Batch size (default: 4).
- `--epochs`: Number of epochs (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
