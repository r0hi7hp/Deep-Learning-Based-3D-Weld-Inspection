"""
Weld Dataset for Point Cloud Segmentation - Production Ready
Optimized data loading with augmentation utilities.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m if m > 1e-6 else pc


def adaptive_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize with outlier handling using 95th percentile."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    
    distances = np.sqrt(np.sum(pc ** 2, axis=1))
    scale = np.percentile(distances, 95)
    
    return pc / max(scale, 1e-6)


def random_rotation_perturbation(points: np.ndarray, angle_sigma: float = 0.06, 
                                  angle_clip: float = 0.18) -> np.ndarray:
    """Apply small random rotations for augmentation."""
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(points.reshape((-1, 3)), R)


class WeldDataset(Dataset):
    """Dataset for weld point cloud segmentation."""
    
    def __init__(self, root_dir: Optional[str] = None, 
                 file_list: Optional[List[Path]] = None,
                 num_points: int = 2048, 
                 split: str = 'train',
                 balance_classes: bool = False, 
                 target_weld_ratio: float = 0.4):
        """
        Args:
            root_dir: Root directory containing model folders
            file_list: List of file paths to use
            num_points: Number of points to sample
            split: 'train', 'val', or 'all'
            balance_classes: If True, balance weld/background points
            target_weld_ratio: Target ratio of weld points
        """
        self.num_points = num_points
        self.split = split
        self.balance_classes = balance_classes
        self.target_weld_ratio = target_weld_ratio
        self.file_list: List[Path] = []
        
        if file_list is not None:
            self.file_list = list(file_list)
        elif root_dir is not None:
            root_path = Path(root_dir)
            for model_dir in sorted(root_path.glob('model_*')):
                if model_dir.is_dir():
                    npz_files = list(model_dir.glob('label_*.npz'))
                    if len(npz_files) == 1:
                        self.file_list.append(npz_files[0])
        else:
            raise ValueError("Either root_dir or file_list must be provided")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.file_list[idx])
        points = data['points'].astype(np.float32)
        labels = data['labels'].astype(np.int64)

        points = adaptive_normalize(points)

        # Deterministic split
        total_points = len(points)
        indices = np.arange(total_points)
        
        rs = np.random.RandomState(42)
        rs.shuffle(indices)
        
        split_idx = int(total_points * 0.8)
        
        if self.split == 'train':
            indices = indices[:split_idx]
        elif self.split == 'val':
            indices = indices[split_idx:]
        
        split_points = points[indices]
        split_labels = labels[indices]
        
        if self.balance_classes:
            points, labels = self._balanced_sample(split_points, split_labels)
        else:
            points, labels = self._random_sample(split_points, split_labels, indices)

        return torch.from_numpy(points).float(), torch.from_numpy(labels)
    
    def _balanced_sample(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample with class balancing."""
        weld_indices = np.where(labels == 1)[0]
        bg_indices = np.where(labels == 0)[0]
        
        target_weld = int(self.num_points * self.target_weld_ratio)
        target_bg = self.num_points - target_weld
        
        if len(weld_indices) > 0:
            replace = len(weld_indices) < target_weld
            weld_choice = np.random.choice(len(weld_indices), target_weld, replace=replace)
            selected_weld = weld_indices[weld_choice]
        else:
            selected_weld = np.array([], dtype=np.int64)
            target_bg = self.num_points
        
        if len(bg_indices) > 0:
            replace = len(bg_indices) < target_bg
            bg_choice = np.random.choice(len(bg_indices), target_bg, replace=replace)
            selected_bg = bg_indices[bg_choice]
        else:
            selected_bg = np.array([], dtype=np.int64)
        
        selected = np.concatenate([selected_weld, selected_bg]) if len(selected_weld) > 0 else selected_bg
        np.random.shuffle(selected)
        
        return points[selected], labels[selected]
    
    def _random_sample(self, points: np.ndarray, labels: np.ndarray, 
                       indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random sampling to num_points."""
        n = len(indices)
        replace = n < self.num_points
        choice = np.random.choice(n, self.num_points, replace=replace)
        
        return points[choice], labels[choice]
