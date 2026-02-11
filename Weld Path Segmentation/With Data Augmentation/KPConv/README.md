# KPConv with Data Augmentation

This directory contains the KPConv implementation for weld path segmentation, situated in the augmented data folder.

## Overview

This version of the model is intended to be used with data augmentation strategies. The training script `train_kpconv.py` is configured to enable augmentation during training.

## Usage

### Training

To train the model:

```bash
python train_kpconv.py
```

**Note:** Ensure the `dataset_path` in `train_kpconv.py` points to your dataset. The script is set up to likely use augmentation features provided by the dataset loader.
