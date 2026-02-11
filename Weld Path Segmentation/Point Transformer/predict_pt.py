"""
Point Transformer Weld Detection Predictor - Production Ready
Deployment-ready class-based implementation with memory optimization.
"""
import gc
import torch
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
from scipy.spatial import KDTree

from model import PointTransformerSeg


class PointTransformerPredictor:
    """Production-ready Point Transformer predictor for weld detection."""
    
    def __init__(self, model_path: str, num_classes: int = 2, num_points: int = 2048, num_heads: int = 8):
        """
        Initialize the predictor with model weights.
        
        Args:
            model_path: Path to the trained model checkpoint
            num_classes: Number of segmentation classes
            num_points: Number of points for model input
            num_heads: Number of attention heads
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> PointTransformerSeg:
        """Load and prepare the model for inference."""
        model = PointTransformerSeg(num_classes=self.num_classes, num_heads=self.num_heads).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model
    
    def preprocess(self, points: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Preprocess point cloud for model input.
        
        Args:
            points: Original point cloud [N, 3]
            
        Returns:
            points_tensor: Model input tensor [1, 3, num_points]
            points_norm: Normalized points for interpolation
            metadata: Preprocessing metadata for postprocessing
        """
        # Normalize
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        scale = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        points_norm = points_centered / scale
        
        # Sample points
        n_pts = len(points_norm)
        if n_pts > self.num_points:
            choice = np.random.choice(n_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(n_pts, self.num_points, replace=True)
        points_input = points_norm[choice, :]
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points_input).float().to(self.device)
        points_tensor = points_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, 3, N]
        
        metadata = {
            'centroid': centroid,
            'scale': scale,
            'points_input': points_input
        }
        
        return points_tensor, points_norm, metadata
    
    def predict(self, points_tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on preprocessed points.
        
        Args:
            points_tensor: Preprocessed input tensor [1, 3, N]
            
        Returns:
            predictions: Per-point predictions [N]
        """
        with torch.no_grad():
            outputs = self.model(points_tensor)
            predictions = outputs.max(dim=1)[1].cpu().numpy()[0]
        return predictions
    
    def postprocess(self, predictions: np.ndarray, points_norm: np.ndarray, 
                    metadata: Dict) -> np.ndarray:
        """
        Interpolate predictions back to original point cloud.
        
        Args:
            predictions: Model predictions [num_points]
            points_norm: Normalized original points [N, 3]
            metadata: Preprocessing metadata
            
        Returns:
            labels: Labels for all original points [N]
        """
        tree = KDTree(metadata['points_input'])
        _, indices = tree.query(points_norm, k=1)
        labels = predictions[indices]
        return labels
    
    def save_results(self, points: np.ndarray, labels: np.ndarray, 
                     output_path: str, save_npz: bool = True) -> Dict[str, str]:
        """
        Save prediction results to files.
        
        Args:
            points: Original points [N, 3]
            labels: Predicted labels [N]
            output_path: Output PLY file path
            save_npz: Whether to save NPZ file
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_path)
        saved_files = {}
        
        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.zeros_like(points)
        colors[labels == 0] = [0.68, 0.85, 0.9]  # Light Blue (Background)
        colors[labels == 1] = [1.0, 0.0, 0.0]    # Red (Weld)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PLY
        o3d.io.write_point_cloud(str(output_path), pcd)
        saved_files['ply'] = str(output_path)
        
        # Save NPZ
        if save_npz:
            npz_path = output_path.with_suffix('.npz')
            np.savez(str(npz_path), points=points, labels=labels)
            saved_files['npz'] = str(npz_path)
        
        # Cleanup
        del pcd, colors
        
        return saved_files
    
    def __call__(self, ply_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Run full prediction pipeline.
        
        Args:
            ply_path: Path to input PLY file
            output_dir: Optional output directory
            
        Returns:
            Dictionary with predictions and file paths
        """
        ply_path = Path(ply_path)
        
        # Setup output paths
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(exist_ok=True, parents=True)
            output_path = out_dir / f"{ply_path.stem}_pred.ply"
        else:
            output_path = ply_path.with_name(f"{ply_path.stem}_pred.ply")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points_orig = np.asarray(pcd.points)
        del pcd
        
        # Run pipeline
        points_tensor, points_norm, metadata = self.preprocess(points_orig)
        predictions = self.predict(points_tensor)
        labels = self.postprocess(predictions, points_norm, metadata)
        
        # Cleanup intermediate tensors
        del points_tensor, points_norm, metadata, predictions
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Save results
        saved_files = self.save_results(points_orig, labels, output_path)
        
        result = {
            'labels': labels,
            'points': points_orig,
            'weld_count': int(np.sum(labels == 1)),
            'total_count': len(labels),
            'files': saved_files
        }
        
        # Final cleanup
        gc.collect()
        
        return result
    
    def cleanup(self):
        """Release model and GPU memory."""
        del self.model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


# Backward-compatible function wrapper
def predict(ply_path: str, model_path: str, num_points: int = 2048) -> Dict:
    """
    Backward-compatible prediction function.
    
    Args:
        ply_path: Path to input PLY file
        model_path: Path to model checkpoint
        num_points: Number of points for model input
        
    Returns:
        Prediction results dictionary
    """
    predictor = PointTransformerPredictor(model_path, num_points=num_points)
    try:
        result = predictor(ply_path)
        return result
    finally:
        predictor.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Transformer Weld Detection')
    parser.add_argument('ply_path', type=str, help='Path to input PLY file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', 
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    result = predict(args.ply_path, args.model)
    print(f"Detected {result['weld_count']} weld points out of {result['total_count']} total")
