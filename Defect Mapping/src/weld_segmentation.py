"""
Step 1: Weld Region Segmentation Module

Integrates with existing PointNet++ and KPConv models from the Weld Path project
to perform weld region segmentation on point clouds.
"""
import sys
import torch
import numpy as np
import open3d as o3d
import logging
from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial import KDTree

from .config import PipelineConfig, ModelConfig

logger = logging.getLogger(__name__)


class WeldSegmenter:
    """
    Weld region segmentation using DL models (PointNet++, KPConv, etc.)
    
    This class wraps the existing prediction functionality from your Weld Path project.
    """
    
    def __init__(self, config: PipelineConfig, model_type: Optional[str] = None):
        """
        Initialize the weld segmenter
        
        Args:
            config: Pipeline configuration
            model_type: Override model type (default: from config)
        """
        self.config = config
        self.model_type = model_type or config.model.model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Add model source path to Python path
        source_path = config.get_source_path(self.model_type)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
            
        logger.info(f"WeldSegmenter initialized with {self.model_type} on {self.device}")
    
    def load_model(self) -> None:
        """Load the DL model from checkpoint"""
        model_path = self.config.get_model_path(self.model_type)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        logger.info(f"Loading {self.model_type} model from {model_path}")
        
        if self.model_type == "pointnet++":
            self._load_pointnet()
        elif self.model_type == "kpconv":
            self._load_kpconv()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _load_pointnet(self) -> None:
        """Load PointNet++ model"""
        from pointnet2 import PointNet2SemSeg
        
        model_path = self.config.get_model_path("pointnet++")
        self.model = PointNet2SemSeg(num_classes=self.config.model.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
    
    def _load_kpconv(self) -> None:
        """Load KPConv model"""
        from kpconv_model import KPConvSegmentation
        
        model_path = self.config.get_model_path("kpconv")
        self.model = KPConvSegmentation(
            in_channels=3,
            num_classes=self.config.model.num_classes,
            num_layers=4,
            init_features=64
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
    
    def load_point_cloud(self, path: str) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Load point cloud from PLY or STL file
        
        Args:
            path: Path to the point cloud file
            
        Returns:
            Tuple of (points array [N, 3], Open3D PointCloud)
        """
        path = Path(path)
        logger.info(f"Loading point cloud from {path}")
        
        if path.suffix.lower() == '.ply':
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
        elif path.suffix.lower() == '.stl':
            mesh = o3d.io.read_triangle_mesh(str(path))
            if not mesh.has_vertices():
                raise ValueError(f"Failed to load mesh from {path}")
            # Sample points from mesh
            pcd = mesh.sample_points_uniformly(number_of_points=100000)
            points = np.asarray(pcd.points)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded {len(points)} points")
        return points, pcd
    
    def normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize points (center and scale)
        
        Args:
            points: Input points [N, 3]
            
        Returns:
            Tuple of (normalized points, centroid, scale factor)
        """
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        scale = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        if scale == 0:
            scale = 1.0
        points_norm = points_centered / scale
        return points_norm, centroid, scale
    
    def segment(self, points: np.ndarray) -> np.ndarray:
        """
        Segment weld region from point cloud
        
        Args:
            points: Input points [N, 3]
            
        Returns:
            Binary labels [N] where 1 = weld, 0 = non-weld
        """
        if self.model is None:
            self.load_model()
        
        num_points = self.config.model.num_points
        
        # Normalize
        points_norm, _, _ = self.normalize_points(points)
        
        # Sample for model input
        if len(points_norm) > num_points:
            choice = np.random.choice(len(points_norm), num_points, replace=False)
        else:
            choice = np.random.choice(len(points_norm), num_points, replace=True)
        points_input = points_norm[choice]
        
        # Prepare tensor
        points_tensor = torch.from_numpy(points_input).float().to(self.device)
        points_tensor = points_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, 3, N]
        
        # Inference
        with torch.no_grad():
            outputs = self.model(points_tensor)  # [1, C, N]
            pred = outputs.max(dim=1)[1].cpu().numpy()[0]  # [N]
        
        # Interpolate back to original points using KDTree
        logger.info("Interpolating labels to all points...")
        tree = KDTree(points_input)
        _, indices = tree.query(points_norm, k=1)
        labels = pred[indices]
        
        weld_count = np.sum(labels == 1)
        logger.info(f"Segmentation complete: {weld_count}/{len(labels)} points classified as weld")
        
        return labels
    
    def get_weld_mask(self, path: str) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
        """
        Complete weld segmentation pipeline
        
        Args:
            path: Path to point cloud file
            
        Returns:
            Tuple of (points [N, 3], weld_mask [N], point_cloud)
        """
        points, pcd = self.load_point_cloud(path)
        labels = self.segment(points)
        
        # Create boolean mask (True for weld points)
        weld_mask = labels == 1
        
        return points, weld_mask, pcd
    
    def save_segmentation(
        self, 
        points: np.ndarray, 
        labels: np.ndarray, 
        output_path: str,
        pcd: Optional[o3d.geometry.PointCloud] = None
    ) -> None:
        """
        Save segmentation result
        
        Args:
            points: Point coordinates [N, 3]
            labels: Segmentation labels [N]
            output_path: Output file path
            pcd: Optional existing point cloud to colorize
        """
        output_path = Path(output_path)
        
        # Save NPZ for programmatic use
        npz_path = output_path.with_suffix('.npz')
        np.savez(str(npz_path), points=points, labels=labels)
        logger.info(f"Saved segmentation data to {npz_path}")
        
        # Save colored PLY
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.zeros_like(points)
        colors[labels == 0] = [0.68, 0.85, 0.9]  # Light blue for non-weld
        colors[labels == 1] = [1.0, 0.0, 0.0]    # Red for weld
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_path = output_path.with_suffix('.ply')
        o3d.io.write_point_cloud(str(ply_path), pcd)
        logger.info(f"Saved colored point cloud to {ply_path}")


def load_existing_segmentation(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load existing segmentation from NPZ file
    
    This is useful if segmentation has already been performed
    
    Args:
        npz_path: Path to NPZ file with 'points' and 'labels' arrays
        
    Returns:
        Tuple of (points [N, 3], labels [N])
    """
    data = np.load(npz_path)
    return data['points'], data['labels']
