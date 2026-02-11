"""Weld point detection and labeling operations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    import open3d as o3d
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WeldDetector:
    """Detects and labels weld regions in point clouds."""

    @staticmethod
    def compute_weld_points(
        aligned_mesh: o3d.geometry.TriangleMesh,
        reference_mesh: o3d.geometry.TriangleMesh,
        sample_count: int,
        distance_threshold: float,
        dbscan_eps: float,
        dbscan_min_samples: int,
    ) -> NDArray[np.floating]:
        """Compute weld candidate points by comparing aligned and reference meshes.
        
        Args:
            aligned_mesh: Aligned (welded) mesh.
            reference_mesh: Reference (weld-free) mesh.
            sample_count: Number of points to sample from each mesh.
            distance_threshold: Distance threshold for weld candidates.
            dbscan_eps: DBSCAN epsilon parameter.
            dbscan_min_samples: DBSCAN minimum samples parameter.
            
        Returns:
            Array of weld point coordinates.
        """
        # Sample points from both meshes
        aligned_pcd = aligned_mesh.sample_points_poisson_disk(sample_count)
        aligned_pts = np.asarray(aligned_pcd.points, dtype=np.float32)
        del aligned_pcd

        ref_pcd = reference_mesh.sample_points_poisson_disk(sample_count)
        ref_pts = np.asarray(ref_pcd.points, dtype=np.float32)
        del ref_pcd

        if len(aligned_pts) == 0 or len(ref_pts) == 0:
            logger.warning("No points sampled from meshes")
            return np.empty((0, 3), dtype=np.float32)

        # Find points in aligned mesh far from reference
        tree = KDTree(ref_pts)
        distances, _ = tree.query(aligned_pts, k=1)
        del ref_pts, tree

        weld_candidates = aligned_pts[distances > distance_threshold]
        del aligned_pts, distances

        logger.info("Weld candidates before clustering: %d", len(weld_candidates))

        if len(weld_candidates) == 0:
            return weld_candidates

        # Cluster to remove noise
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
            weld_candidates
        )
        filtered = weld_candidates[clustering.labels_ != -1].copy()
        del weld_candidates, clustering

        logger.info("Weld points after filtering: %d", len(filtered))
        return filtered

    @staticmethod
    def label_points(
        points: NDArray[np.floating],
        weld_points: NDArray[np.floating],
        weld_radius: float,
    ) -> NDArray[np.uint8]:
        """Label points based on proximity to weld points.
        
        Args:
            points: Full point cloud array.
            weld_points: Weld region point coordinates.
            weld_radius: Maximum distance to be labeled as weld.
            
        Returns:
            Binary label array (1 = weld, 0 = non-weld).
        """
        if len(weld_points) == 0:
            return np.zeros(len(points), dtype=np.uint8)

        tree = KDTree(weld_points)
        distances, _ = tree.query(points, k=1)
        labels = (distances < weld_radius).astype(np.uint8)
        del tree, distances
        return labels
