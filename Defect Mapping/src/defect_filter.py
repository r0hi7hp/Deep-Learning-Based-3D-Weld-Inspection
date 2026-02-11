"""
Steps 5-6: Weld-Constrained Defect Filtering Module

Filters distance field to focus only on weld region deviations:
- Step 5: Apply weld mask from DL segmentation
- Step 6: Select top percentile of deviations for clustering
"""
import numpy as np
import logging
from typing import Tuple, Optional

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DefectFilter:
    """
    Filters deviations to focus on weld region only
    
    This is a key modification from traditional approaches:
    Instead of analyzing all high-deviation points, we constrain
    analysis to the weld seam identified by the DL model.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the filter
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.top_percentile = config.deviation.top_percentile
    
    def apply_weld_mask(
        self,
        distances: np.ndarray,
        weld_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter distances to weld region only
        
        D_weld = {d(p_i) | p_i ∈ M}
        
        Args:
            distances: Full distance array [N]
            weld_mask: Boolean mask [N] where True = weld point
            
        Returns:
            Tuple of (weld_indices, weld_distances)
        """
        weld_indices = np.where(weld_mask)[0]
        weld_distances = distances[weld_indices]
        
        total_points = len(distances)
        weld_points = len(weld_indices)
        
        logger.info(f"Applied weld mask: {weld_points}/{total_points} points "
                   f"({100*weld_points/total_points:.1f}%) in weld region")
        
        if weld_points == 0:
            logger.warning("No weld points found! Check segmentation results.")
            
        return weld_indices, weld_distances
    
    def select_top_percentile(
        self,
        distances: np.ndarray,
        percentile: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Select points with top percentile deviations
        
        Args:
            distances: Distance array
            percentile: Top percentile to select (e.g., 10 = top 10%)
            
        Returns:
            Tuple of (boolean mask for selected points, threshold value)
        """
        percentile = percentile or self.top_percentile
        
        # Threshold is (100 - percentile) since we want top deviations
        threshold = np.percentile(distances, 100 - percentile)
        
        selected_mask = distances >= threshold
        selected_count = np.sum(selected_mask)
        
        logger.info(f"Selected top {percentile}%: {selected_count} points above threshold={threshold:.4f}")
        
        return selected_mask, threshold
    
    def filter_weld_deviations(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        weld_mask: np.ndarray,
        top_percentile: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete filtering pipeline: weld mask + top percentile
        
        This is the key step that differentiates from the original paper.
        
        Args:
            points: Full point cloud [N, 3]
            distances: Full distance array [N]
            weld_mask: Boolean weld mask [N]
            top_percentile: Top percentile to select
            
        Returns:
            Tuple of (selected_points, selected_distances, selected_indices)
        """
        top_percentile = top_percentile or self.top_percentile
        
        # Step 5: Apply weld mask
        weld_indices, weld_distances = self.apply_weld_mask(distances, weld_mask)
        weld_points = points[weld_indices]
        
        if len(weld_distances) == 0:
            return np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=int)
        
        # Step 6: Select top deviations within weld region
        selected_mask, threshold = self.select_top_percentile(weld_distances, top_percentile)
        
        # Get final selection
        selected_points = weld_points[selected_mask]
        selected_distances = weld_distances[selected_mask]
        selected_indices = weld_indices[selected_mask]
        
        logger.info(f"Filtered to {len(selected_points)} high-deviation weld points")
        
        return selected_points, selected_distances, selected_indices
    
    def get_filter_statistics(
        self,
        distances: np.ndarray,
        weld_mask: np.ndarray,
        filtered_distances: np.ndarray
    ) -> dict:
        """
        Compute statistics about the filtering process
        
        Args:
            distances: Original distances
            weld_mask: Weld region mask
            filtered_distances: Final filtered distances
            
        Returns:
            Dictionary with statistics
        """
        weld_distances = distances[weld_mask]
        non_weld_distances = distances[~weld_mask]
        
        return {
            'total_points': len(distances),
            'weld_points': int(np.sum(weld_mask)),
            'non_weld_points': int(np.sum(~weld_mask)),
            'filtered_points': len(filtered_distances),
            'weld_distance_mean': float(weld_distances.mean()) if len(weld_distances) > 0 else 0,
            'weld_distance_max': float(weld_distances.max()) if len(weld_distances) > 0 else 0,
            'non_weld_distance_mean': float(non_weld_distances.mean()) if len(non_weld_distances) > 0 else 0,
            'filtered_distance_mean': float(filtered_distances.mean()) if len(filtered_distances) > 0 else 0,
            'filtered_distance_min': float(filtered_distances.min()) if len(filtered_distances) > 0 else 0,
        }
