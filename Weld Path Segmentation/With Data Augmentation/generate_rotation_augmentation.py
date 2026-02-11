"""
Rotation-Only Data Augmentation Script for Weld Detection Dataset

This script generates CLEAN augmented .npz files using ONLY rotation transforms.
- Uses 90°, 180°, 270° Z-axis rotations (deterministic, no noise)
- Creates 3 augmented variants per original file
- 100 original × 3 = 300 clean augmented files
- NO jitter, NO scaling, NO translation, NO point dropout

This is a conservative augmentation approach that:
1. Preserves weld geometry precisely
2. Teaches rotation invariance without adding noise
3. Provides clean training data for better generalization

Usage:
    python generate_rotation_augmentation.py
    
After running, update your training script to use the new 'Augmented Data Clean' folder.
"""

import numpy as np
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rotate_z(points, angle_degrees):
    """
    Apply rotation around Z-axis by specified angle.
    
    Args:
        points: numpy array of shape (N, 3)
        angle_degrees: rotation angle in degrees (90, 180, or 270)
    
    Returns:
        Rotated points with same shape and dtype
    """
    angle_rad = np.radians(angle_degrees)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return (points @ R.T).astype(points.dtype)


def generate_rotation_augmented_dataset():
    """
    Generate rotation-only augmented .npz files for ALL original files.
    
    Creates 3 variants per file: 90°, 180°, 270° Z-rotations
    Total: 100 original × 3 = 300 augmented files
    """
    # Paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    ROOT_DIR = SCRIPT_DIR.parent / 'Dataset'
    OUTPUT_DIR = ROOT_DIR / 'Augmented Data Clean'
    
    logging.info("=" * 60)
    logging.info("Rotation-Only Data Augmentation for Weld Detection")
    logging.info("=" * 60)
    logging.info(f"Dataset directory: {ROOT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all original model files
    original_files = []
    for model_dir in sorted(ROOT_DIR.glob('model_*')):
        if not model_dir.is_dir():
            continue
        npz_files = list(model_dir.glob('label_*.npz'))
        if len(npz_files) == 1:
            original_files.append(npz_files[0])
    
    logging.info(f"Found {len(original_files)} original models")
    
    if len(original_files) < 100:
        logging.warning(f"Expected 100 models, found {len(original_files)}")
    
    # Clear existing output directory to start fresh
    if OUTPUT_DIR.exists():
        logging.info(f"Clearing existing augmented data in {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Rotation angles to apply
    rotation_angles = [90, 180, 270]  # 0° is the original, so we skip it
    
    # Generate augmented files for ALL original files
    augmented_count = 0
    
    for original_file in original_files:
        # Get the original model number
        model_name = original_file.parent.name  # e.g., "model_1"
        model_num = model_name.split('_')[1]    # e.g., "1"
        
        # Load original data
        data = np.load(original_file)
        points = data['points']
        labels = data['labels']
        
        # Generate 3 rotated variants
        for angle in rotation_angles:
            # Create output filename: model_1_rot90.npz, model_1_rot180.npz, model_1_rot270.npz
            aug_filename = f"{model_name}_rot{angle}.npz"
            aug_file_path = OUTPUT_DIR / aug_filename
            
            # Apply rotation
            rotated_points = rotate_z(points, angle)
            
            # Save augmented data (labels remain the same - rotation doesn't change them)
            np.savez(aug_file_path, points=rotated_points, labels=labels)
            
            augmented_count += 1
        
        if (original_files.index(original_file) + 1) % 20 == 0:
            files_done = original_files.index(original_file) + 1
            logging.info(f"Processed {files_done}/{len(original_files)} original files "
                        f"({augmented_count} augmented files generated)")
    
    logging.info("=" * 60)
    logging.info("AUGMENTATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Original files processed: {len(original_files)}")
    logging.info(f"Augmented files generated: {augmented_count}")
    logging.info(f"Rotation angles used: {rotation_angles}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Verify a sample
    sample_files = list(OUTPUT_DIR.glob('*.npz'))[:3]
    if sample_files:
        logging.info("\nSample verification:")
        for sample_file in sample_files:
            sample_data = np.load(sample_file)
            logging.info(f"  {sample_file.name}: points={sample_data['points'].shape}, "
                        f"labels={sample_data['labels'].shape}")
    
    # Print summary for training
    logging.info("\n" + "=" * 60)
    logging.info("NEXT STEPS FOR TRAINING")
    logging.info("=" * 60)
    logging.info("Update your training script to use:")
    logging.info(f"  - Training: 70 original + 300 augmented = 370 files")
    logging.info(f"  - Validation: 30 original files ONLY (no augmented)")
    logging.info(f"  - Augmented data path: {OUTPUT_DIR}")
    
    return augmented_count


if __name__ == '__main__':
    generate_rotation_augmented_dataset()
