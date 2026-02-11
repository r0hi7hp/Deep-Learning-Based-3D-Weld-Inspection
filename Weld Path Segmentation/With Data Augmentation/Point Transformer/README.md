# Point Transformer with Data Augmentation

This directory contains the Point Transformer implementation configured for robust training using on-the-fly data augmentation.

## Key Features

- **Data Augmentation**: The `dataset.py` in this directory includes built-in augmentation techniques to improve model generalization:
  - Random Z-axis Rotation
  - Random Scaling (0.8x - 1.25x)
  - Random Translation
  - Gaussian Noise Jittering
  - Random Point Dropout

## Usage

### Training

To train the model with augmentation enabled:

```bash
python train_pt.py --batch-size 4 --epochs 100
```

The training process uses the specialized `WeldDataset` from `dataset.py` which applies random transformations to the training data.

**Arguments:**
- `--batch-size`: Batch size (default: 4).
- `--epochs`: Number of epochs (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
