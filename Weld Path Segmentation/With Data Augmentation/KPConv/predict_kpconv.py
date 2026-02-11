"""
KPConv Weld Detection Predictor - Production Ready (Data Augmentation Version)
Deployment-ready class-based implementation with memory optimization.
Supports inference on PLY/NPZ files.
"""
import gc
import sys
import torch
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent))

from kpconv_model import KPConvSegmentation


class KPConvPredictor:
    """Production-ready KPConv predictor for weld detection."""
    
    def __init__(self, model_path: str, num_points: int = 2048):
        """
        Initialize the predictor with model weights.
        
        Args:
            model_path: Path to the trained model checkpoint
            num_points: Number of points per inference window
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        self.model, self.config = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> Tuple[KPConvSegmentation, Dict]:
        """Load and prepare the model for inference."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        model = KPConvSegmentation(
            num_classes=config.get('num_classes', 2),
            in_channels=config.get('in_channels', 3),
            encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dims=config.get('decoder_dims', [256, 128, 64, 64]),
            num_points=config.get('num_points_hierarchy', [512, 256, 128, 64]),
            radii=config.get('radii', [0.1, 0.2, 0.4, 0.8]),
            kernel_size=config.get('kernel_size', 15)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config
    
    def _adaptive_normalize(self, points: np.ndarray) -> np.ndarray:
        """Adaptive normalization matching training."""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        distances = np.linalg.norm(points, axis=1)
        scale = np.percentile(distances, 95)
        
        if scale > 1e-6:
            points = points / scale
        
        return points
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from PLY or NPZ file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.ply':
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            del pcd
        elif file_path.suffix.lower() == '.npz':
            data = np.load(file_path)
            points = data['points']
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return points.astype(np.float32)
    
    def predict(self, points: np.ndarray, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for entire point cloud using sliding window approach.
        
        Args:
            points: [N, 3] point coordinates
            batch_size: Batch size for inference
            
        Returns:
            labels: [N] predicted labels
            confidence: [N] prediction confidence
        """
        self.model.eval()
        
        N = len(points)
        all_votes = np.zeros((N, 2), dtype=np.float32)
        vote_counts = np.zeros(N, dtype=np.float32)
        
        # Normalize points
        points_normalized = self._adaptive_normalize(points.copy())
        
        # Create overlapping windows
        num_windows = max(1, (N + self.num_points // 2) // self.num_points)
        
        with torch.no_grad():
            for window_idx in range(num_windows):
                # Sample points for this window
                if N <= self.num_points:
                    indices = np.arange(N)
                    if N < self.num_points:
                        pad_indices = np.random.choice(N, self.num_points - N, replace=True)
                        indices = np.concatenate([indices, pad_indices])
                else:
                    center_idx = (window_idx * N) // num_windows
                    distances = np.abs(np.arange(N) - center_idx)
                    probs = 1.0 / (distances + 1)
                    probs = probs / probs.sum()
                    indices = np.random.choice(N, self.num_points, replace=False, p=probs)
                
                # Get window points
                window_points = points_normalized[indices]
                
                # Convert to tensor
                window_tensor = torch.from_numpy(window_points).unsqueeze(0).to(self.device)
                window_tensor = window_tensor.transpose(1, 2)  # [1, 3, N]
                
                # Predict
                logits = self.model(window_tensor)
                probs = torch.exp(logits).squeeze(0).cpu().numpy()
                
                # Accumulate votes
                valid_indices = indices[:min(len(indices), N)]
                all_votes[valid_indices] += probs[:, :len(valid_indices)].T
                vote_counts[valid_indices] += 1
                
                # Cleanup
                del window_tensor, logits, probs
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Average votes
        vote_counts = np.maximum(vote_counts, 1)
        avg_probs = all_votes / vote_counts[:, np.newaxis]
        
        # Get predictions
        labels = np.argmax(avg_probs, axis=1)
        confidence = np.max(avg_probs, axis=1)
        
        return labels, confidence
    
    def save_results(self, points: np.ndarray, labels: np.ndarray, 
                     confidence: np.ndarray, output_path: str) -> Dict[str, str]:
        """
        Save prediction results to files.
        
        Args:
            points: Original points [N, 3]
            labels: Predicted labels [N]
            confidence: Prediction confidence [N]
            output_path: Output PLY file path
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_path)
        saved_files = {}
        
        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.zeros((len(points), 3))
        colors[labels == 0] = [0.2, 0.4, 0.8]  # Blue (Background)
        
        weld_mask = labels == 1
        if weld_mask.any():
            weld_confidence = confidence[weld_mask]
            colors[weld_mask, 0] = 0.5 + 0.5 * weld_confidence
            colors[weld_mask, 1] = 0.1 * (1 - weld_confidence)
            colors[weld_mask, 2] = 0.1 * (1 - weld_confidence)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PLY
        o3d.io.write_point_cloud(str(output_path), pcd)
        saved_files['ply'] = str(output_path)
        
        # Save NPZ
        npz_path = output_path.with_suffix('.npz')
        np.savez(str(npz_path), points=points, labels=labels, confidence=confidence)
        saved_files['npz'] = str(npz_path)
        
        # Cleanup
        del pcd, colors
        
        return saved_files
    
    def __call__(self, file_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Run full prediction pipeline.
        
        Args:
            file_path: Path to input PLY/NPZ file
            output_dir: Optional output directory
            
        Returns:
            Dictionary with predictions and file paths
        """
        file_path = Path(file_path)
        
        # Setup output paths
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(exist_ok=True, parents=True)
            output_path = out_dir / f"{file_path.stem}_kpconv_prediction.ply"
        else:
            output_path = file_path.with_name(f"{file_path.stem}_kpconv_prediction.ply")
        
        # Load point cloud
        points = self._load_point_cloud(str(file_path))
        
        # Run prediction
        labels, confidence = self.predict(points)
        
        # Save results
        saved_files = self.save_results(points, labels, confidence, output_path)
        
        result = {
            'labels': labels,
            'confidence': confidence,
            'points': points,
            'weld_count': int(np.sum(labels == 1)),
            'total_count': len(labels),
            'mean_confidence': float(np.mean(confidence)),
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
def predict(file_path: str, model_path: str, num_points: int = 2048,
            output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible prediction function.
    
    Args:
        file_path: Path to input PLY/NPZ file
        model_path: Path to model checkpoint
        num_points: Number of points per inference window
        output_dir: Output directory for predictions
        
    Returns:
        Tuple of (labels, confidence)
    """
    predictor = KPConvPredictor(model_path, num_points=num_points)
    try:
        result = predictor(file_path, output_dir)
        return result['labels'], result['confidence']
    finally:
        predictor.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KPConv Weld Detection')
    parser.add_argument('file_path', type=str, help='Path to input PLY/NPZ file')
    parser.add_argument('--model', type=str,
                       default=str(Path(__file__).parent / 'checkpoints' / 'best_model.pth'),
                       help='Path to model checkpoint')
    parser.add_argument('--num-points', type=int, default=2048,
                       help='Number of points per inference window')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    labels, confidence = predict(args.file_path, args.model, args.num_points, args.output_dir)
    weld_count = int(np.sum(labels == 1))
    print(f"Detected {weld_count} weld points out of {len(labels)} total")
    print(f"Mean confidence: {np.mean(confidence):.3f}")
