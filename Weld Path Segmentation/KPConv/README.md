# KPConv for Weld Path Segmentation

This directory contains the implementation of Kernel Point Convolution (KPConv) for semantic segmentation of weld paths.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

## File Structure

- `kpconv_model.py`: Contains the `KPConvSegmentation` architecture.
- `kpconv_ops.py`: wrapper for KPConv operations.
- `train_kpconv.py`: Script to train the model.
- `predict_kpconv.py`: Script to generate predictions.
- `weld_dataset.py`: Dataset loader.

## Usage

### Training

To train the model, run:

```bash
python train_kpconv.py
```

The training script uses a dictionary `config` within the `main` function to control hyperparameters. You may need to edit this file to adjust paths or parameters like:
- `dataset_path`: Path to the dataset.
- `batch_size`: Training batch size.
- `epochs`: Number of training epochs.

### Inference

To run inference on new data:

```bash
python predict_kpconv.py
```
