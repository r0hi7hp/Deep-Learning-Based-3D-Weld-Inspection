# PointNet for Weld Path Segmentation

This directory contains the implementation of PointNet for semantic segmentation of weld paths in 3D point clouds.

## Requirements

- Python 3.8+
- PyTorch
- NumPy

## File Structure

- `pointnet.py`: Contains the PointNet architecture definition (`PointNetSemSeg`, `PointNetEncoder`, `TNet`).
- `train_pointnet.py`: Script to train the PointNet model.
- `predict_pointnet.py`: Script to generate predictions using a trained model.
- `weld_dataset.py`: Dataset loader for parsing `.npz` point cloud files.

## Usage

### Training

To train the model, run:

```bash
python train_pointnet.py --batch-size 4 --epochs 100 --learning-rate 0.001
```

**Arguments:**
- `--batch-size`: Batch size for training (default: 4).
- `--epochs`: Number of epochs to train (default: 100).
- `--learning-rate`: Learning rate for Adam optimizer (default: 0.001).
- `--weld-weight`: Class weight for the weld class (default: 20.0).
- `--feature-reg`: Regularization strength for feature transform (default: 0.001).

### Inference

To run inference on new data:

```bash
python predict_pointnet.py
```

Check `predict_pointnet.py` for input/output path configurations.
