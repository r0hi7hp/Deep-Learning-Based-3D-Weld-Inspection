"""
Main Pipeline Orchestration Module

Chains all 8 steps into a complete weld defect mapping pipeline.
"""
import sys
import numpy as np
import open3d as o3d
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from .config import PipelineConfig, create_config
from .weld_segmentation import WeldSegmenter
from .point_cloud_prep import PointCloudPreparator
from .alignment import PointCloudAligner
from .distance_computation import DistanceComputer
from .defect_filter import DefectFilter
from .clustering import DefectClusterer
from .defect_analysis import DefectAnalyzer, WeldDefect, DefectReport
from .visualization import DefectVisualizer

logger = logging.getLogger(__name__)


class WeldDefectPipeline:
    """
    Main pipeline for weld defect mapping
    
    Orchestrates all 8 steps:
    1. Weld region segmentation (DL)
    2. Point cloud preparation
    3. Alignment (FPFH + ICP)
    4. Distance computation
    5. Weld-constrained filtering
    6. Deviation selection
    7. DBSCAN clustering
    8. Defect analysis
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline
        
        Args:
            config: Pipeline configuration (or use defaults)
        """
        self.config = config or PipelineConfig()
        
        # Initialize modules
        self.segmenter = WeldSegmenter(self.config)
        self.preparator = PointCloudPreparator(self.config)
        self.aligner = PointCloudAligner(self.config)
        self.distance_computer = DistanceComputer(self.config)
        self.defect_filter = DefectFilter(self.config)
        self.clusterer = DefectClusterer(self.config)
        self.analyzer = DefectAnalyzer(self.config)
        self.visualizer = DefectVisualizer(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Pipeline initialized with model_type={self.config.model.model_type}")
    
    def run(
        self,
        reference_path: str,
        test_path: str,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        skip_segmentation: bool = False,
        existing_segmentation: Optional[str] = None
    ) -> DefectReport:
        """
        Run the complete pipeline
        
        Args:
            reference_path: Path to reference (defect-free) model
            test_path: Path to test (defective) model
            output_dir: Output directory for results
            model_type: Override model type from config
            skip_segmentation: Skip DL segmentation (for testing)
            existing_segmentation: Path to existing NPZ with segmentation
            
        Returns:
            DefectReport with all results
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type:
            self.config.model.model_type = model_type
            self.segmenter = WeldSegmenter(self.config, model_type)
        
        logger.info("="*60)
        logger.info("WELD DEFECT MAPPING PIPELINE")
        logger.info("="*60)
        logger.info(f"Reference: {reference_path}")
        logger.info(f"Test: {test_path}")
        logger.info(f"Model: {self.config.model.model_type}")
        logger.info("="*60)
        
        # Step 1: Weld Segmentation
        logger.info("\n[STEP 1/8] Weld Region Segmentation")
        if existing_segmentation:
            logger.info(f"Loading existing segmentation from {existing_segmentation}")
            data = np.load(existing_segmentation)
            test_points = data['points']
            weld_labels = data['labels']
            weld_mask = weld_labels == 1
            test_pcd = o3d.geometry.PointCloud()
            test_pcd.points = o3d.utility.Vector3dVector(test_points)
        elif skip_segmentation:
            logger.info("Skipping segmentation, using all points as weld region")
            test_pcd = self.preparator.load_point_cloud(test_path)
            test_points = np.asarray(test_pcd.points)
            weld_mask = np.ones(len(test_points), dtype=bool)
        else:
            test_points, weld_mask, test_pcd = self.segmenter.get_weld_mask(test_path)
        
        # Step 2: Point Cloud Preparation
        logger.info("\n[STEP 2/8] Point Cloud Preparation")
        ref_pcd = self.preparator.load_point_cloud(reference_path)
        ref_pcd = self.preparator.uniform_downsample(ref_pcd)
        ref_pcd = self.preparator.estimate_normals(ref_pcd)
        
        # Ensure test_pcd has normals
        if not test_pcd.has_normals():
            test_pcd = self.preparator.estimate_normals(test_pcd)
        
        # Step 3: Alignment
        logger.info("\n[STEP 3/8] Point Cloud Alignment (FPFH + ICP)")
        aligned_test_pcd, transformation, alignment_metrics = self.aligner.align(
            test_pcd, ref_pcd
        )
        
        # Update test points with aligned positions
        aligned_test_points = np.asarray(aligned_test_pcd.points)
        
        # Step 4: Distance Computation
        logger.info("\n[STEP 4/8] Distance Computation")
        distances = self.distance_computer.compute_nn_distances(aligned_test_pcd, ref_pcd)
        
        # Save intermediate distance field if configured
        if self.config.save_intermediate:
            dist_pcd = self.distance_computer.visualize_distance_field(
                aligned_test_pcd, distances
            )
            o3d.io.write_point_cloud(
                str(output_dir / "distance_heatmap.ply"), dist_pcd
            )
        
        # Steps 5-6: Weld-Constrained Filtering
        logger.info("\n[STEP 5-6/8] Weld-Constrained Defect Filtering")
        selected_points, selected_distances, selected_indices = \
            self.defect_filter.filter_weld_deviations(
                aligned_test_points, distances, weld_mask
            )
        
        filter_statistics = self.defect_filter.get_filter_statistics(
            distances, weld_mask, selected_distances
        )
        
        # Step 7: DBSCAN Clustering
        logger.info("\n[STEP 7/8] DBSCAN Clustering")
        if len(selected_points) > 0:
            cluster_labels = self.clusterer.cluster_deviations(
                selected_points, selected_distances
            )
            cluster_info = self.clusterer.get_cluster_info(
                selected_points, cluster_labels, selected_distances
            )
        else:
            cluster_labels = np.array([], dtype=int)
            cluster_info = []
        
        # Step 8: Defect Analysis
        logger.info("\n[STEP 8/8] Defect Localization & Severity Estimation")
        defects = self.analyzer.localize_defects(
            selected_points, cluster_labels, selected_distances, cluster_info
        )
        
        # Create full label array for visualization
        full_labels = np.full(len(aligned_test_points), -1, dtype=int)
        for i, idx in enumerate(selected_indices):
            full_labels[idx] = cluster_labels[i] if i < len(cluster_labels) else -1
        
        # Generate report
        report = self.analyzer.generate_report(
            defects=defects,
            reference_path=reference_path,
            test_path=test_path,
            model_type=self.config.model.model_type,
            total_points=len(aligned_test_points),
            weld_points=int(np.sum(weld_mask)),
            alignment_metrics=alignment_metrics,
            filter_statistics=filter_statistics
        )
        
        # Save report
        report.save_json(str(output_dir / "defect_report.json"))
        report.print_summary()
        
        # Visualizations
        logger.info("\nCreating visualizations...")
        self.visualizer.visualize_all(
            aligned_test_points,
            weld_mask,
            distances,
            full_labels,
            defects,
            str(output_dir)
        )
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        return report
    
    def run_with_all_models(
        self,
        reference_path: str,
        test_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, DefectReport]:
        """
        Run pipeline with all available models
        
        Args:
            reference_path: Path to reference model
            test_path: Path to test model
            output_dir: Base output directory
            
        Returns:
            Dictionary of model_type -> DefectReport
        """
        output_dir = Path(output_dir or self.config.output_dir)
        
        reports = {}
        for model_type in self.config.model.model_paths.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running with model: {model_type}")
            logger.info(f"{'='*60}")
            
            model_output_dir = output_dir / model_type
            
            try:
                report = self.run(
                    reference_path=reference_path,
                    test_path=test_path,
                    output_dir=str(model_output_dir),
                    model_type=model_type
                )
                reports[model_type] = report
            except Exception as e:
                logger.error(f"Failed to run with {model_type}: {e}")
                reports[model_type] = None
        
        # Summary comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        for model_type, report in reports.items():
            if report:
                logger.info(f"{model_type}: {report.defect_count} defects, "
                           f"Quality: {report.summary['quality_assessment']}")
            else:
                logger.info(f"{model_type}: Failed")
        
        return reports


def run_pipeline(
    reference_path: str,
    test_path: str,
    output_dir: str = "results",
    model_type: str = "pointnet++",
    top_percentile: float = 10.0,
    run_all_models: bool = False
) -> DefectReport:
    """
    Convenience function to run the pipeline
    
    Args:
        reference_path: Path to reference (defect-free) model
        test_path: Path to test (defective) model
        output_dir: Output directory
        model_type: Model to use ("pointnet++", "kpconv")
        top_percentile: Top percentile of deviations to analyze
        run_all_models: If True, run with all available models
        
    Returns:
        DefectReport (or dict of reports if run_all_models=True)
    """
    config = create_config(
        model_type=model_type,
        output_dir=output_dir,
        top_percentile=top_percentile
    )
    
    pipeline = WeldDefectPipeline(config)
    
    if run_all_models:
        return pipeline.run_with_all_models(reference_path, test_path, output_dir)
    else:
        return pipeline.run(reference_path, test_path, output_dir, model_type)
