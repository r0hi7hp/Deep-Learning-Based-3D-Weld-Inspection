"""
Dataset handler for Point Transformer model (WITH DATA AUGMENTATION)
Standalone dataset with built-in augmentation for improved generalization
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging

# ============================================================================
# Data Augmentation Functions
# ============================================================================

def random_z_rotation(points):
    """Random rotation around Z-axis"""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return points @ R.T

def random_scaling(points, scale_low=0.8, scale_high=1.25):
    """Random uniform scaling"""
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale

def random_translation(points, shift_range=0.1):
    """Random translation"""
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3)).astype(np.float32)
    return points + shifts

def jitter_points(points, sigma=0.01, clip=0.05):
    """Add Gaussian noise"""
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip).astype(np.float32)
    return points + jitter

def random_point_dropout(points, labels, dropout_ratio=0.1):
    """Randomly drop points"""
    n = len(points)
    keep_mask = np.random.rand(n) > dropout_ratio
    if keep_mask.sum() < 10:  # Ensure minimum points
        keep_mask[:10] = True
    return points[keep_mask], labels[keep_mask]

def augment_point_cloud(points, labels):
    """
    Main augmentation function with probabilistic transforms.
    Applied only during training to improve generalization.
    """
    if np.random.rand() < 0.9:
        points = random_z_rotation(points)

    if np.random.rand() < 0.8:
        points = random_scaling(points)

    if np.random.rand() < 0.8:
        points = random_translation(points)

    if np.random.rand() < 0.7:
        points = jitter_points(points)

    if np.random.rand() < 0.5:
        points, labels = random_point_dropout(points, labels)

    return points, labels

# ============================================================================
# Normalization Functions
# ============================================================================

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def adaptive_normalize(pc):
    """Improved normalization with outlier handling"""
    distances = np.sqrt(np.sum(pc ** 2, axis=1))
    percentile_95 = np.percentile(distances, 95)
    
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    
    scale = max(percentile_95, 1e-6)
    pc = pc / scale
    
    return pc

# ============================================================================
# Dataset Class
# ============================================================================

class WeldDataset(Dataset):
    def __init__(self, root_dir=None, file_list=None, num_points=2048, split='train', 
                 balance_classes=False, target_weld_ratio=0.4, augment=True):
        """
        Args:
            root_dir: Root directory containing model folders
            file_list: List of file paths to use
            num_points: Number of points to sample
            split: 'train', 'val', or 'all'
            balance_classes: If True, balance weld/background points via oversampling
            target_weld_ratio: Target ratio of weld points
            augment: If True, apply data augmentation (only for training)
        """
        self.num_points = num_points
        self.split = split
        self.file_list = []
        self.balance_classes = balance_classes
        self.target_weld_ratio = target_weld_ratio
        # Only augment during training
        self.augment = augment and split == 'train'
        
        if file_list is not None:
            self.file_list = file_list
        elif root_dir is not None:
            self.root_dir = Path(root_dir)
            for model_dir in sorted(self.root_dir.glob('model_*')):
                if not model_dir.is_dir():
                    continue
                npz_files = list(model_dir.glob('label_*.npz'))
                if len(npz_files) == 1:
                    self.file_list.append(npz_files[0])
                else:
                    logging.warning(f"Skipping {model_dir}: Found {len(npz_files)} .npz files")
            logging.info(f"Found {len(self.file_list)} samples in {root_dir}")
        else:
            raise ValueError("Either root_dir or file_list must be provided")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        points = data['points'].astype(np.float32)
        labels = data['labels'].astype(np.int64)

        # Apply adaptive normalization
        points = adaptive_normalize(points)

        # Apply data augmentation if enabled (training only)
        if self.augment:
            points, labels = augment_point_cloud(points, labels)

        # Deterministic split of points
        total_points = len(points)
        indices = np.arange(total_points)
        
        rs = np.random.RandomState(42) 
        rs.shuffle(indices)
        
        split_idx = int(total_points * 0.8)
        
        if self.split == 'train':
            indices = indices[:split_idx]
        elif self.split == 'val':
            indices = indices[split_idx:]
        
        # Get points and labels for the split
        split_points = points[indices]
        split_labels = labels[indices]
        
        # Balanced sampling if enabled
        if self.balance_classes:
            weld_mask = split_labels == 1
            bg_mask = split_labels == 0
            
            weld_indices = np.where(weld_mask)[0]
            bg_indices = np.where(bg_mask)[0]
            
            num_weld = len(weld_indices)
            num_bg = len(bg_indices)
            
            target_weld_points = int(self.num_points * self.target_weld_ratio)
            target_bg_points = self.num_points - target_weld_points
            
            if num_weld > 0:
                if num_weld < target_weld_points:
                    weld_choice = np.random.choice(num_weld, target_weld_points, replace=True)
                else:
                    weld_choice = np.random.choice(num_weld, target_weld_points, replace=False)
                selected_weld_indices = weld_indices[weld_choice]
            else:
                selected_weld_indices = np.array([], dtype=np.int64)
                target_weld_points = 0
                target_bg_points = self.num_points
            
            if num_bg > 0:
                if num_bg < target_bg_points:
                    bg_choice = np.random.choice(num_bg, target_bg_points, replace=True)
                else:
                    bg_choice = np.random.choice(num_bg, target_bg_points, replace=False)
                selected_bg_indices = bg_indices[bg_choice]
            else:
                selected_bg_indices = np.array([], dtype=np.int64)
            
            if len(selected_weld_indices) > 0 and len(selected_bg_indices) > 0:
                selected_indices = np.concatenate([selected_weld_indices, selected_bg_indices])
            elif len(selected_weld_indices) > 0:
                selected_indices = selected_weld_indices
            else:
                selected_indices = selected_bg_indices
                
            np.random.shuffle(selected_indices)
            
            points = split_points[selected_indices, :]
            labels = split_labels[selected_indices]
        else:
            if len(indices) < self.num_points:
                choice = np.random.choice(len(indices), self.num_points, replace=True)
            else:
                choice = np.random.choice(len(indices), self.num_points, replace=False)
                 
            selected_indices = indices[choice]
            points = points[selected_indices, :]
            labels = labels[selected_indices]

        return torch.from_numpy(points).float(), torch.from_numpy(labels)


# Re-export for convenience
__all__ = [
    'WeldDataset',
    'pc_normalize',
    'adaptive_normalize',
    'augment_point_cloud',
    'random_z_rotation',
    'random_scaling',
    'random_translation',
    'jitter_points',
    'random_point_dropout'
]


if __name__ == '__main__':
    from config import Config
    
    print("Testing WeldDataset with Data Augmentation...")
    dataset = WeldDataset(
        root_dir=Config.DATASET_ROOT,
        num_points=Config.NUM_POINTS,
        split='train',
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Augmentation enabled: {dataset.augment}")
    
    if len(dataset) > 0:
        points, labels = dataset[0]
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {labels.unique()}")
        print(f"Weld points: {(labels == 1).sum().item()}")
        print(f"Background points: {(labels == 0).sum().item()}")
