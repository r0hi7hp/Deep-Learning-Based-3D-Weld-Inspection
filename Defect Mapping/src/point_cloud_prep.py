"""
Step 2: Point Cloud Preparation and Sampling Module

Prepares reference and test point clouds for comparison:
- Loads mesh/point cloud files
- Performs uniform sampling
- Ensures comparable number of points
"""
import numpy as np
import open3d as o3d
import logging
from pathlib import Path
from typing import Tuple, Optional
import trimesh

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class PointCloudPreparator:
    """
    Prepares point clouds for defect analysis
    
    Handles:
    - Loading from various formats (PLY, STL, OBJ)
    - Mesh to point cloud conversion
    - Uniform sampling for consistent density
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the preparator
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.target_points = config.sampling.target_points
        self.voxel_size = config.sampling.voxel_size
    
    def load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        """
        Load mesh from file
        
        Args:
            path: Path to mesh file (STL, OBJ, PLY)
            
        Returns:
            Open3D TriangleMesh
        """
        path = Path(path)
        logger.info(f"Loading mesh from {path}")
        
        if path.suffix.lower() == '.ply':
            # Try loading as mesh first
            mesh = o3d.io.read_triangle_mesh(str(path))
            if not mesh.has_triangles():
                # It's a point cloud, not a mesh
                logger.info("PLY file is point cloud, not mesh")
                return None
            return mesh
        elif path.suffix.lower() in ['.stl', '.obj']:
            mesh = o3d.io.read_triangle_mesh(str(path))
            if not mesh.has_triangles():
                raise ValueError(f"Failed to load mesh triangles from {path}")
            return mesh
        else:
            raise ValueError(f"Unsupported mesh format: {path.suffix}")
    
    def load_point_cloud(self, path: str) -> o3d.geometry.PointCloud:
        """
        Load point cloud from file
        
        Args:
            path: Path to point cloud or mesh file
            
        Returns:
            Open3D PointCloud
        """
        path = Path(path)
        logger.info(f"Loading point cloud from {path}")
        
        if path.suffix.lower() == '.ply':
            pcd = o3d.io.read_point_cloud(str(path))
            if len(pcd.points) == 0:
                # Try as mesh and sample
                mesh = o3d.io.read_triangle_mesh(str(path))
                if mesh.has_triangles():
                    pcd = mesh.sample_points_uniformly(self.target_points)
        elif path.suffix.lower() in ['.stl', '.obj']:
            mesh = o3d.io.read_triangle_mesh(str(path))
            # Use Poisson disk sampling for more uniform distribution
            pcd = mesh.sample_points_poisson_disk(self.target_points)
        elif path.suffix.lower() == '.npz':
            # Load from numpy archive
            data = np.load(str(path))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data['points'])
            if 'normals' in data:
                pcd.normals = o3d.utility.Vector3dVector(data['normals'])
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        logger.info(f"Loaded {len(pcd.points)} points")
        return pcd
    
    def mesh_to_point_cloud(
        self, 
        mesh: o3d.geometry.TriangleMesh, 
        n_points: Optional[int] = None,
        method: str = "poisson"
    ) -> o3d.geometry.PointCloud:
        """
        Convert mesh to point cloud via sampling
        
        Args:
            mesh: Input triangle mesh
            n_points: Number of points to sample (default: from config)
            method: Sampling method ("uniform" or "poisson")
            
        Returns:
            Sampled point cloud
        """
        n_points = n_points or self.target_points
        
        if method == "poisson":
            pcd = mesh.sample_points_poisson_disk(n_points)
        else:
            pcd = mesh.sample_points_uniformly(n_points)
        
        logger.info(f"Sampled {len(pcd.points)} points from mesh using {method} sampling")
        return pcd
    
    def uniform_downsample(
        self, 
        pcd: o3d.geometry.PointCloud, 
        target_points: Optional[int] = None
    ) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud to target number of points
        
        Args:
            pcd: Input point cloud
            target_points: Target number of points
            
        Returns:
            Downsampled point cloud
        """
        target_points = target_points or self.target_points
        current_points = len(pcd.points)
        
        if current_points <= target_points:
            logger.info(f"Point cloud has {current_points} points, no downsampling needed")
            return pcd
        
        # Use voxel downsampling
        voxel_size = self.voxel_size
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        # If still too many points, use random sampling
        while len(downsampled.points) > target_points and voxel_size < 10:
            voxel_size *= 1.5
            downsampled = pcd.voxel_down_sample(voxel_size)
        
        if len(downsampled.points) > target_points:
            # Random subset
            indices = np.random.choice(len(downsampled.points), target_points, replace=False)
            downsampled = downsampled.select_by_index(indices)
        
        logger.info(f"Downsampled from {current_points} to {len(downsampled.points)} points")
        return downsampled
    
    def estimate_normals(
        self, 
        pcd: o3d.geometry.PointCloud, 
        radius: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """
        Estimate normals for point cloud
        
        Args:
            pcd: Input point cloud
            radius: Search radius for normal estimation
            
        Returns:
            Point cloud with normals
        """
        if pcd.has_normals():
            logger.info("Point cloud already has normals")
            return pcd
        
        radius = radius or self.config.alignment.fpfh_radius_normal
        
        logger.info(f"Estimating normals with radius {radius}")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        return pcd
    
    def prepare_pair(
        self, 
        reference_path: str, 
        test_path: str,
        compute_normals: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Prepare a pair of point clouds for comparison
        
        Both point clouds are loaded, sampled to comparable sizes,
        and optionally have normals computed.
        
        Args:
            reference_path: Path to reference (defect-free) model
            test_path: Path to test (potentially defective) model
            compute_normals: Whether to compute normals
            
        Returns:
            Tuple of (reference_pcd, test_pcd)
        """
        logger.info("Preparing point cloud pair for comparison")
        
        # Load both point clouds
        ref_pcd = self.load_point_cloud(reference_path)
        test_pcd = self.load_point_cloud(test_path)
        
        # Downsample to comparable sizes
        ref_pcd = self.uniform_downsample(ref_pcd)
        test_pcd = self.uniform_downsample(test_pcd)
        
        # Compute normals if needed
        if compute_normals:
            ref_pcd = self.estimate_normals(ref_pcd)
            test_pcd = self.estimate_normals(test_pcd)
        
        logger.info(f"Prepared pair: reference={len(ref_pcd.points)}, test={len(test_pcd.points)} points")
        
        return ref_pcd, test_pcd
    
    def save_point_cloud(
        self, 
        pcd: o3d.geometry.PointCloud, 
        output_path: str,
        include_normals: bool = True
    ) -> None:
        """
        Save point cloud to file
        
        Args:
            pcd: Point cloud to save
            output_path: Output file path
            include_normals: Whether to save normals
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.npz':
            data = {'points': np.asarray(pcd.points)}
            if include_normals and pcd.has_normals():
                data['normals'] = np.asarray(pcd.normals)
            if pcd.has_colors():
                data['colors'] = np.asarray(pcd.colors)
            np.savez(str(output_path), **data)
        else:
            o3d.io.write_point_cloud(str(output_path), pcd)
        
        logger.info(f"Saved point cloud to {output_path}")
