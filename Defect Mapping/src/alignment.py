"""
Step 3: Point Cloud Alignment Module

Implements coarse and fine alignment:
- FPFH (Fast Point Feature Histograms) for coarse alignment via Fast Global Registration
- ICP (Iterative Closest Point) for fine alignment
"""
import numpy as np
import open3d as o3d
import logging
from typing import Tuple, Optional
import copy

from .config import PipelineConfig, AlignmentConfig

logger = logging.getLogger(__name__)


class PointCloudAligner:
    """
    Aligns test point cloud to reference point cloud
    
    Uses a two-step approach:
    1. FPFH-based Fast Global Registration for coarse alignment
    2. ICP refinement for fine alignment
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the aligner
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.alignment_config = config.alignment
        
    def compute_fpfh_features(
        self, 
        pcd: o3d.geometry.PointCloud, 
        voxel_size: Optional[float] = None
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Compute FPFH features for a point cloud
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling
            
        Returns:
            Tuple of (downsampled point cloud, FPFH features)
        """
        voxel_size = voxel_size or self.alignment_config.fpfh_voxel_size
        
        logger.info(f"Computing FPFH features with voxel_size={voxel_size}")
        
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals if not present
        if not pcd_down.has_normals():
            radius_normal = voxel_size * 2
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )
        
        # Compute FPFH features
        radius_feature = voxel_size * 4
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        logger.info(f"Computed FPFH features: {fpfh.data.shape[1]} points, {fpfh.data.shape[0]} dimensions")
        
        return pcd_down, fpfh
    
    def fast_global_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
        voxel_size: Optional[float] = None
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Perform Fast Global Registration using FPFH features
        
        Args:
            source: Source point cloud (test)
            target: Target point cloud (reference)
            source_fpfh: FPFH features of source
            target_fpfh: FPFH features of target
            voxel_size: Voxel size used for feature computation
            
        Returns:
            Registration result with transformation matrix
        """
        voxel_size = voxel_size or self.alignment_config.fpfh_voxel_size
        distance_threshold = voxel_size * 1.5
        
        logger.info(f"Running Fast Global Registration (distance_threshold={distance_threshold})")
        
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            )
        )
        
        logger.info(f"FGR result: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}")
        
        return result
    
    def icp_refinement(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray,
        threshold: Optional[float] = None,
        max_iterations: Optional[int] = None
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Refine alignment using ICP
        
        Args:
            source: Source point cloud (test)
            target: Target point cloud (reference)
            initial_transform: Initial transformation from coarse alignment
            threshold: Distance threshold for correspondences
            max_iterations: Maximum ICP iterations
            
        Returns:
            Refined registration result
        """
        threshold = threshold or self.alignment_config.icp_threshold
        max_iterations = max_iterations or self.alignment_config.icp_max_iterations
        
        logger.info(f"Running ICP refinement (threshold={threshold}, max_iter={max_iterations})")
        
        # Point-to-plane ICP for better convergence
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.alignment_config.icp_relative_fitness,
                relative_rmse=self.alignment_config.icp_relative_rmse,
                max_iteration=max_iterations
            )
        )
        
        logger.info(f"ICP result: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}")
        
        return result
    
    def align(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        voxel_size: Optional[float] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, dict]:
        """
        Complete alignment pipeline: FPFH + FGR + ICP
        
        Args:
            source_pcd: Source point cloud (test/defective)
            target_pcd: Target point cloud (reference/defect-free)
            voxel_size: Voxel size for feature computation
            
        Returns:
            Tuple of (aligned source point cloud, transformation matrix, metrics dict)
        """
        voxel_size = voxel_size or self.alignment_config.fpfh_voxel_size
        
        logger.info("Starting alignment pipeline")
        
        # Ensure normals exist
        if not source_pcd.has_normals():
            source_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
            )
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
            )
        
        # Step 1: Compute FPFH features
        source_down, source_fpfh = self.compute_fpfh_features(source_pcd, voxel_size)
        target_down, target_fpfh = self.compute_fpfh_features(target_pcd, voxel_size)
        
        # Step 2: Coarse alignment with Fast Global Registration
        coarse_result = self.fast_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        
        # Step 3: Fine alignment with ICP
        fine_result = self.icp_refinement(
            source_pcd, target_pcd, coarse_result.transformation
        )
        
        # Apply transformation
        aligned_source = copy.deepcopy(source_pcd)
        aligned_source.transform(fine_result.transformation)
        
        # Collect metrics
        metrics = {
            'coarse_fitness': coarse_result.fitness,
            'coarse_rmse': coarse_result.inlier_rmse,
            'fine_fitness': fine_result.fitness,
            'fine_rmse': fine_result.inlier_rmse,
            'transformation': fine_result.transformation.tolist()
        }
        
        logger.info(f"Alignment complete. Final fitness={fine_result.fitness:.4f}, RMSE={fine_result.inlier_rmse:.4f}")
        
        return aligned_source, fine_result.transformation, metrics
    
    def evaluate_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        threshold: float = 1.0
    ) -> dict:
        """
        Evaluate alignment quality
        
        Args:
            source: Aligned source point cloud
            target: Target point cloud
            threshold: Distance threshold for correspondence
            
        Returns:
            Dictionary with alignment metrics
        """
        result = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold
        )
        
        return {
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set_size': len(result.correspondence_set)
        }
