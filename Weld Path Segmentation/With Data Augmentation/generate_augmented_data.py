"""
Offline Data Augmentation Script for Weld Detection Dataset

This script generates augmented .npz files from the original training dataset.
- Uses 180° Z-axis rotation as the augmentation technique (deterministic, no noise)
- Creates 70 augmented files from 70 training files = 140 total training samples
- Saves to Dataset/Augmented Data/ folder
- Validation set (30 files) remains unchanged

Usage:
    python generate_augmented_data.py
"""

import numpy as np
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rotate_z_180(points):
    """
    Apply 180° rotation around Z-axis.
    This is a deterministic, clean geometric transformation with no noise.
    
    Rotation matrix for 180°:
    cos(π) = -1, sin(π) = 0
    R = [[-1,  0, 0],
         [ 0, -1, 0],
         [ 0,  0, 1]]
    """
    R = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return (points @ R.T).astype(points.dtype)


def generate_augmented_dataset():
    """
    Generate augmented .npz files for the training dataset.
    Uses the same 70/30 split as the training scripts (random seed 42).
    """
    # Paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    ROOT_DIR = SCRIPT_DIR.parent / 'Dataset'
    OUTPUT_DIR = ROOT_DIR / 'Augmented Data'
    
    logging.info("=" * 60)
    logging.info("Offline Data Augmentation for Weld Detection")
    logging.info("=" * 60)
    logging.info(f"Dataset directory: {ROOT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all model files
    all_files = []
    for model_dir in sorted(ROOT_DIR.glob('model_*')):
        if not model_dir.is_dir():
            continue
        npz_files = list(model_dir.glob('label_*.npz'))
        if len(npz_files) == 1:
            all_files.append(npz_files[0])
    
    logging.info(f"Found {len(all_files)} total models")
    
    if len(all_files) < 100:
        logging.warning(f"Expected 100 models, found {len(all_files)}")
    
    # Use the same split as training (random seed 42)
    random.seed(42)
    shuffled_files = all_files.copy()
    random.shuffle(shuffled_files)
    
    train_files = shuffled_files[:70]
    val_files = shuffled_files[70:]
    
    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Validation files: {len(val_files)} (not augmented)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Generate augmented files
    augmented_count = 0
    for original_file in train_files:
        # Get the original model folder name (e.g., model_1)
        model_folder_name = original_file.parent.name
        original_file_name = original_file.stem  # e.g., label_1
        
        # Create augmented folder and file names
        aug_folder_name = f"{model_folder_name}_aug"
        aug_file_name = f"{original_file_name}_aug.npz"
        
        # Create output paths
        aug_folder_path = OUTPUT_DIR / aug_folder_name
        aug_file_path = aug_folder_path / aug_file_name
        
        # Load original data
        data = np.load(original_file)
        points = data['points']
        labels = data['labels']
        
        # Apply 180° Z-rotation
        augmented_points = rotate_z_180(points)
        
        # Save augmented data (labels remain the same)
        aug_folder_path.mkdir(exist_ok=True, parents=True)
        np.savez(aug_file_path, points=augmented_points, labels=labels)
        
        augmented_count += 1
        if augmented_count % 10 == 0:
            logging.info(f"Generated {augmented_count}/{len(train_files)} augmented files...")
    
    logging.info("=" * 60)
    logging.info(f"AUGMENTATION COMPLETE")
    logging.info(f"Generated {augmented_count} augmented files")
    logging.info(f"Saved to: {OUTPUT_DIR}")
    logging.info("=" * 60)
    
    # Verify one sample
    sample_aug_folder = OUTPUT_DIR / f"{train_files[0].parent.name}_aug"
    sample_aug_file = list(sample_aug_folder.glob('*.npz'))[0]
    sample_data = np.load(sample_aug_file)
    
    logging.info("\nSample verification:")
    logging.info(f"  File: {sample_aug_file.name}")
    logging.info(f"  Points shape: {sample_data['points'].shape}")
    logging.info(f"  Labels shape: {sample_data['labels'].shape}")
    logging.info(f"  Points dtype: {sample_data['points'].dtype}")
    logging.info(f"  Labels dtype: {sample_data['labels'].dtype}")
    
    return augmented_count


if __name__ == '__main__':
    generate_augmented_dataset()
