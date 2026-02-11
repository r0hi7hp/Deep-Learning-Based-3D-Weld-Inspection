"""
Dataset handler for Point Transformer model
Wraps the main weld_dataset.py with Point Transformer specific configurations
"""
import sys
from pathlib import Path

# Import all functions and classes from local weld_dataset
from weld_dataset import (
    WeldDataset,
    pc_normalize,
    random_point_dropout,
    random_scale_point_cloud,
    shift_point_cloud,
    jitter_point_cloud,
    rotate_point_cloud,
    random_rotation_perturbation,
    random_mirror,
    adaptive_normalize
)

# Re-export for convenience
__all__ = [
    'WeldDataset',
    'pc_normalize',
    'random_point_dropout',
    'random_scale_point_cloud',
    'shift_point_cloud',
    'jitter_point_cloud',
    'rotate_point_cloud',
    'random_rotation_perturbation',
    'random_mirror',
    'adaptive_normalize'
]


if __name__ == '__main__':
    # Test the dataset
    from config import Config
    
    print("Testing WeldDataset...")
    dataset = WeldDataset(
        root_dir=Config.DATASET_ROOT,
        num_points=Config.NUM_POINTS,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        points, labels = dataset[0]
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {labels.unique()}")
        print(f"Weld points: {(labels == 1).sum().item()}")
        print(f"Background points: {(labels == 0).sum().item()}")
