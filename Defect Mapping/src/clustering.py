"""
Step 7: DBSCAN Clustering Module

Clusters high-deviation weld points into distinct defect regions.
"""
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DefectClusterer:
    """
    Clusters high-deviation points into defect regions using DBSCAN
    
    DBSCAN is used because:
    - It doesn't require specifying number of clusters
    - It can find arbitrarily shaped clusters
    - It identifies noise points (isolated deviations)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the clusterer
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.clustering_config = config.clustering
        
    def compute_adaptive_eps(
        self, 
        points: np.ndarray,
        percentile: Optional[float] = None
    ) -> float:
        """
        Compute adaptive eps based on point cloud density
        
        Uses k-th nearest neighbor distance at given percentile
        
        Args:
            points: Point coordinates [N, 3]
            percentile: Percentile of NN distances to use
            
        Returns:
            Computed eps value
        """
        if len(points) < 2:
            return 1.0
            
        percentile = percentile or self.clustering_config.eps_percentile
        
        # Find k-th nearest neighbor distances
        k = min(self.clustering_config.min_samples, len(points) - 1)
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)  # +1 because first is self
        
        # Use the k-th neighbor distance
        kth_distances = distances[:, -1]
        
        # Sort and take percentile
        sorted_distances = np.sort(kth_distances)
        eps = sorted_distances[int(len(sorted_distances) * (100 - percentile) / 100)]
        
        logger.info(f"Computed adaptive eps={eps:.4f} from {len(points)} points")
        
        return eps
    
    def cluster_deviations(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply DBSCAN clustering to deviation points
        
        Args:
            points: Point coordinates [N, 3]
            distances: Deviation distances [N] (used for weighted clustering)
            eps: DBSCAN radius parameter
            min_samples: Minimum points per cluster
            
        Returns:
            Cluster labels [N] where -1 = noise
        """
        if len(points) == 0:
            return np.array([], dtype=int)
            
        # Use adaptive eps if not provided
        if eps is None:
            eps = self.clustering_config.eps
            if eps is None:
                eps = self.compute_adaptive_eps(points)
        
        min_samples = min_samples or self.clustering_config.min_samples
        
        logger.info(f"Running DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
        
        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(points)
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def get_cluster_info(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        distances: np.ndarray
    ) -> List[Dict]:
        """
        Extract information about each cluster
        
        Args:
            points: Point coordinates [N, 3]
            labels: Cluster labels [N]
            distances: Deviation distances [N]
            
        Returns:
            List of dictionaries with cluster information
        """
        cluster_info = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
                
            mask = labels == label
            cluster_points = points[mask]
            cluster_distances = distances[mask]
            
            # Compute cluster properties
            center_of_mass = np.mean(cluster_points, axis=0)
            bbox_min = np.min(cluster_points, axis=0)
            bbox_max = np.max(cluster_points, axis=0)
            
            info = {
                'cluster_id': int(label),
                'num_points': int(np.sum(mask)),
                'center_of_mass': center_of_mass.tolist(),
                'bounding_box': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist(),
                    'size': (bbox_max - bbox_min).tolist()
                },
                'max_deviation': float(np.max(cluster_distances)),
                'mean_deviation': float(np.mean(cluster_distances)),
                'rmse': float(np.sqrt(np.mean(cluster_distances ** 2))),
                'point_indices': np.where(mask)[0].tolist()
            }
            
            cluster_info.append(info)
        
        # Sort by max deviation (most severe first)
        cluster_info.sort(key=lambda x: x['max_deviation'], reverse=True)
        
        return cluster_info
    
    def merge_nearby_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        merge_distance: float
    ) -> np.ndarray:
        """
        Merge clusters that are close together
        
        This can help consolidate fragmented defect regions
        
        Args:
            points: Point coordinates [N, 3]
            labels: Cluster labels [N]
            merge_distance: Distance threshold for merging
            
        Returns:
            Updated labels after merging
        """
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        if len(unique_labels) < 2:
            return labels
        
        # Compute cluster centers
        centers = {}
        for label in unique_labels:
            mask = labels == label
            centers[label] = np.mean(points[mask], axis=0)
        
        # Find clusters to merge
        merge_map = {l: l for l in unique_labels}
        label_list = list(unique_labels)
        
        for i, label1 in enumerate(label_list):
            for label2 in label_list[i+1:]:
                dist = np.linalg.norm(centers[label1] - centers[label2])
                if dist < merge_distance:
                    # Merge label2 into label1
                    merge_map[label2] = merge_map[label1]
        
        # Apply merge map
        new_labels = labels.copy()
        for old_label, new_label in merge_map.items():
            new_labels[labels == old_label] = new_label
        
        # Renumber clusters consecutively
        unique_new = set(new_labels)
        unique_new.discard(-1)
        label_remap = {old: new for new, old in enumerate(sorted(unique_new))}
        
        result_labels = new_labels.copy()
        for old, new in label_remap.items():
            result_labels[new_labels == old] = new
        
        n_original = len(unique_labels)
        n_merged = len(unique_new)
        if n_original != n_merged:
            logger.info(f"Merged {n_original} clusters into {n_merged}")
        
        return result_labels
