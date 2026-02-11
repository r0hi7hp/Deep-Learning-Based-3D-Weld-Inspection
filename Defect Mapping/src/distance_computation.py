"""
Step 4: Distance Computation Module

Computes nearest-neighbor distances between aligned test and reference point clouds.
Points with high distances indicate geometric deviations (potential defects).
"""
import numpy as np
import open3d as o3d
import logging
from typing import Tuple, Optional
from scipy.spatial import KDTree

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DistanceComputer:
    """
    Computes point-to-point distances between aligned point clouds
    
    For each point in the test cloud, finds the nearest neighbor in the
    reference cloud and computes the Euclidean distance.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the distance computer
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
    
    def compute_nn_distances(
        self,
        test_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        Compute nearest-neighbor distances from test to reference
        
        d(p_i) = ||p_i - NN(p_i)||
        
        Args:
            test_pcd: Test (defective) point cloud
            ref_pcd: Reference (defect-free) point cloud
            
        Returns:
            Array of distances [N] for each point in test cloud
        """
        test_points = np.asarray(test_pcd.points)
        ref_points = np.asarray(ref_pcd.points)
        
        logger.info(f"Computing NN distances: test={len(test_points)}, ref={len(ref_points)} points")
        
        # Build KDTree for reference cloud
        tree = KDTree(ref_points)
        
        # Query nearest neighbors
        distances, indices = tree.query(test_points, k=1)
        
        logger.info(f"Distance stats: min={distances.min():.4f}, max={distances.max():.4f}, "
                   f"mean={distances.mean():.4f}, std={distances.std():.4f}")
        
        return distances
    
    def compute_bidirectional_distances(
        self,
        test_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances in both directions
        
        This helps identify both:
        - Excess material (test points far from reference)
        - Missing material (reference points far from test)
        
        Args:
            test_pcd: Test point cloud
            ref_pcd: Reference point cloud
            
        Returns:
            Tuple of (test_to_ref distances, ref_to_test distances)
        """
        test_points = np.asarray(test_pcd.points)
        ref_points = np.asarray(ref_pcd.points)
        
        # Build KDTrees
        ref_tree = KDTree(ref_points)
        test_tree = KDTree(test_points)
        
        # Test -> Reference (excess material detection)
        test_to_ref, _ = ref_tree.query(test_points, k=1)
        
        # Reference -> Test (missing material detection)
        ref_to_test, _ = test_tree.query(ref_points, k=1)
        
        logger.info(f"Test->Ref distances: mean={test_to_ref.mean():.4f}")
        logger.info(f"Ref->Test distances: mean={ref_to_test.mean():.4f}")
        
        return test_to_ref, ref_to_test
    
    def compute_signed_distances(
        self,
        test_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        Compute signed distances using surface normals
        
        Positive = point is outside reference surface (excess material)
        Negative = point is inside reference surface (missing material)
        
        Args:
            test_pcd: Test point cloud
            ref_pcd: Reference point cloud (must have normals)
            
        Returns:
            Array of signed distances [N]
        """
        test_points = np.asarray(test_pcd.points)
        ref_points = np.asarray(ref_pcd.points)
        
        if not ref_pcd.has_normals():
            logger.warning("Reference cloud has no normals, estimating...")
            ref_pcd.estimate_normals()
        
        ref_normals = np.asarray(ref_pcd.normals)
        
        # Build KDTree
        tree = KDTree(ref_points)
        distances, indices = tree.query(test_points, k=1)
        
        # Compute signed distance using dot product with normal
        delta = test_points - ref_points[indices]
        normals = ref_normals[indices]
        
        # Sign is determined by whether point is along normal direction
        signs = np.sign(np.sum(delta * normals, axis=1))
        signed_distances = distances * signs
        
        positive_count = np.sum(signed_distances > 0)
        negative_count = np.sum(signed_distances < 0)
        logger.info(f"Signed distances: {positive_count} positive (excess), {negative_count} negative (deficit)")
        
        return signed_distances
    
    def build_distance_field(
        self,
        test_points: np.ndarray,
        ref_pcd: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        Build distance field from test points to reference cloud
        
        This is useful when working with raw numpy arrays instead of
        Open3D point clouds.
        
        Args:
            test_points: Test points array [N, 3]
            ref_pcd: Reference point cloud
            
        Returns:
            Distance array [N]
        """
        ref_points = np.asarray(ref_pcd.points)
        tree = KDTree(ref_points)
        distances, _ = tree.query(test_points, k=1)
        return distances
    
    def get_distance_statistics(self, distances: np.ndarray) -> dict:
        """
        Compute statistics for distance array
        
        Args:
            distances: Array of distances
            
        Returns:
            Dictionary with statistics
        """
        return {
            'min': float(distances.min()),
            'max': float(distances.max()),
            'mean': float(distances.mean()),
            'std': float(distances.std()),
            'median': float(np.median(distances)),
            'percentile_90': float(np.percentile(distances, 90)),
            'percentile_95': float(np.percentile(distances, 95)),
            'percentile_99': float(np.percentile(distances, 99))
        }
    
    def visualize_distance_field(
        self,
        pcd: o3d.geometry.PointCloud,
        distances: np.ndarray,
        max_distance: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """
        Create a colored point cloud based on distances
        
        Uses a colormap where:
        - Blue = low distance (good match)
        - Yellow = medium distance
        - Red = high distance (potential defect)
        
        Args:
            pcd: Point cloud to colorize
            distances: Distance values
            max_distance: Maximum distance for colormap scaling
            
        Returns:
            Colored point cloud
        """
        import copy
        colored_pcd = copy.deepcopy(pcd)
        
        # Normalize distances
        if max_distance is None:
            max_distance = np.percentile(distances, 99)
        
        normalized = np.clip(distances / max_distance, 0, 1)
        
        # Create colormap (blue -> yellow -> red)
        colors = np.zeros((len(distances), 3))
        
        # Blue to yellow (0 to 0.5)
        mask_low = normalized < 0.5
        t_low = normalized[mask_low] * 2
        colors[mask_low, 0] = t_low          # R: 0 -> 1
        colors[mask_low, 1] = t_low          # G: 0 -> 1
        colors[mask_low, 2] = 1 - t_low      # B: 1 -> 0
        
        # Yellow to red (0.5 to 1)
        mask_high = ~mask_low
        t_high = (normalized[mask_high] - 0.5) * 2
        colors[mask_high, 0] = 1             # R: 1
        colors[mask_high, 1] = 1 - t_high    # G: 1 -> 0
        colors[mask_high, 2] = 0             # B: 0
        
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return colored_pcd
