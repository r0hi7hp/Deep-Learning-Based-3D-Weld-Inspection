# Point Transformer for Weld Path Segmentation

This directory contains the implementation of Point Transformer for semantic segmentation of weld paths in 3D point clouds. This model uses self-attention mechanisms to capture complex dependencies and local structures.

## Requirements

- Python 3.8+
- PyTorch
- NumPy

## File Structure

- `model.py`: Contains the `PointTransformerSeg` architecture.
- `train_pt.py`: Script to train the model.
- `predict_pt.py`: Script to generate predictions.
- `dataset.py`: Dataset loader.
- `config.py`: Configuration file (contains model constants).

## Usage

### Training

To train the model, run:

```bash
python train_pt.py --batch-size 4 --epochs 100 --learning-rate 0.001
```

**Arguments:**
- `--batch-size`: Batch size (default: 4).
- `--epochs`: Number of epochs (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
- `--patience`: Early stopping patience (default: 15).

### Inference

To run inference on new data:

```bash
python predict_pt.py
```
