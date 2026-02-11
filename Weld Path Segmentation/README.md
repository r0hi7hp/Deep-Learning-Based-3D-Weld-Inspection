# Automated Surface Weld Defect Detection System

This project contains deep learning models for detecting defects in surface welds. It includes various 3D point cloud architectures and tools for data augmentation to improve model performance.

## Project Structure

The project is organized into different folders for each model capability:

*   **Models**:
    *   `PointNet`, `PointNet ++`: Point-based Deep Learning architectures.
    *   `Point Transformer`: Transformer-based architecture for point clouds.
    *   `KPConv`: Kernel Point Convolution model.
    *   `Hybrid Model`: A hybrid architecture combining different approaches.
*   **With Data Augmentation**:
    *   Contains specific versions of the models configured to train with augmented data.
    *   Includes scripts to generate augmented datasets (e.g., rotating data to handle different orientations).

## Prerequisites

*   Windows OS (Project environment)
*   Anaconda or Miniconda
*   Python 3.10
*   CUDA-enabled GPU (Recommended)

## Installation

1.  **Environment Setup**:
    Open your terminal (Anaconda Prompt recommended) and navigate to this folder. Run the following command to create the required environment:
    ```bash
    conda env create -f environment.yml
    conda activate weld_detection
    ```

2.  **Data Setup**:
    Ensure you have a folder named `Dataset` in the root directory (the same place as this README).
    *   Inside `Dataset`, you should have your model folders (e.g., `model_1`, `model_2`, ...).
    *   Each model folder should contain the `.npz` data file (e.g., `label_1.npz`).

## How to Use

### 1. Training Standard Models

To train a model (e.g., PointNet++):
1.  Navigate to the model folder: `cd "PointNet ++"`
2.  Run the training script:
    ```bash
    python train_pointnetpp.py
    ```

### 2. Data Augmentation

Data augmentation helps the models learn better by creating variations of your training data (e.g., rotating the 3D scans).

**Step 1: Generate Augmented Data**
1.  Navigate to the augmentation folder:
    ```bash
    cd "With Data Augmentation"
    ```
2.  Run the generation script:
    ```bash
    python generate_augmented_data.py
    ```
    *   This will create 180-degree rotated copies of your training data.
    *   The new files are saved in `Dataset/Augmented Data`.

**Step 2: Train with Augmented Data**
1.  Go to the specific model folder inside `With Data Augmentation` (e.g., `PointNet++`):
    ```bash
    cd PointNet++
    ```
2.  Run the training script:
    ```bash
    python train_pointnetpp.py
    ```
    *   *Note*: This script is pre-configured to automatically include the data from `Dataset/Augmented Data` during training.

### 3. Prediction

To run predictions using a trained model:
1.  Navigate to the model folder.
2.  Run the prediction script:
    ```bash
    python predict_pointnetpp.py
    ```

## Notes
*   Always ensure your `Dataset` folder is structure correctly before running scripts.
*   The `With Data Augmentation` models are designed to be robust to rotation changes in the weld scans.
