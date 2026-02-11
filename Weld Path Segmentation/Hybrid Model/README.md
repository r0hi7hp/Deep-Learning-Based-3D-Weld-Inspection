# Hybrid Model for Weld Path Segmentation

This directory contains the implementation of a Hybrid Model that combines PointNet++ and Point Transformer features for semantic segmentation of weld paths.

## Requirements

- Python 3.8+
- PyTorch
- NumPy

## File Structure

- `hybrid_model.py`: Contains the `HybridPointNet` architecture.
- `train_hybrid.py`: Script to train the model.
- `predict_hybrid.py`: Script to generate predictions.
- `weld_dataset.py`: Dataset loader.

## Usage

### Training

To train the model, run:

```bash
python train_hybrid.py --batch-size 4 --epochs 100 --learning-rate 0.001
```

**Arguments:**
- `--batch-size`: Batch size (default: 4).
- `--epochs`: Number of epochs (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
- `--weld-weight`: Class weight for the weld class (default: 20.0).
- `--patience`: Early stopping patience (default: 15).

### Inference

To run inference on new data:

```bash
python predict_hybrid.py
```
