"""Mesh loading, alignment, and sampling operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MeshOperations:
    """Handles mesh I/O, alignment, and point sampling operations."""

    @staticmethod
    def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
        """Load a triangle mesh from PLY or STL file.
        
        Args:
            path: Path to the mesh file.
            
        Returns:
            Loaded triangle mesh.
            
        Raises:
            ValueError: If the mesh is empty.
        """
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise ValueError(f"Mesh is empty: {path}")
        return mesh

    @staticmethod
    def sample_points(
        mesh: o3d.geometry.TriangleMesh,
        sample_count: int,
    ) -> NDArray[np.floating]:
        """Sample points from mesh using Poisson disk sampling.
        
        Args:
            mesh: Input triangle mesh.
            sample_count: Number of points to sample.
            
        Returns:
            Array of sampled 3D points with shape (N, 3).
        """
        pcd = mesh.sample_points_poisson_disk(sample_count)
        points = np.asarray(pcd.points, dtype=np.float32)
        del pcd  # Explicit cleanup
        return points

    @staticmethod
    def align_meshes(
        source_path: Path,
        target_path: Path,
        samples: int,
        max_distance: float,
    ) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """Align source mesh to target mesh using ICP.
        
        Args:
            source_path: Path to source (welded) mesh.
            target_path: Path to target (reference) mesh.
            samples: Number of samples for ICP alignment.
            max_distance: Maximum correspondence distance.
            
        Returns:
            Tuple of (aligned_source_mesh, target_mesh).
        """
        logger.info("Loading meshes: %s vs %s", source_path.name, target_path.name)

        source_mesh = MeshOperations.load_mesh(source_path)
        target_mesh = MeshOperations.load_mesh(target_path)

        # Sample point clouds for registration
        source_pcd = source_mesh.sample_points_poisson_disk(samples)
        target_pcd = target_mesh.sample_points_poisson_disk(samples)

        source_pcd.estimate_normals()
        target_pcd.estimate_normals()

        logger.info("Running ICP alignment")
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=max_distance,
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        # Cleanup temporary point clouds
        del source_pcd, target_pcd

        source_mesh.transform(icp_result.transformation)
        return source_mesh, target_mesh
