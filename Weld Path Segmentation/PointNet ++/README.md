# PointNet++ for Weld Path Segmentation

This directory contains the implementation of PointNet++ for semantic segmentation of weld paths in 3D point clouds. PointNet++ improves upon PointNet by capturing local structures at varying scales.

## Requirements

- Python 3.8+
- PyTorch
- NumPy

## File Structure

- `pointnet2.py`: Contains the PointNet++ architecture definition (`PointNet2SemSeg`).
- `train_pointnetpp.py`: Script to train the PointNet++ model.
- `predict_pointnetpp.py`: Script to generate predictions using a trained model.
- `weld_dataset.py`: Dataset loader.

## Usage

### Training

To train the model, run:

```bash
python train_pointnetpp.py --batch-size 4 --epochs 100 --learning-rate 0.001
```

**Arguments:**
- `--batch-size`: Batch size for training (default: 4).
- `--epochs`: Number of epochs to train (default: 100).
- `--learning-rate`: Learning rate (default: 0.001).
- `--weld-weight`: Class weight for the weld class (default: 20.0).
- `--patience`: Early stopping patience (default: 15).

### Inference

To run inference on new data:

```bash
python predict_pointnetpp.py
```
