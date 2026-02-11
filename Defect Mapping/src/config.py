"""
Configuration for Weld Defect Mapping Pipeline

Contains all configurable parameters including:
- Model paths and types
- Processing parameters (sampling, alignment)
- DBSCAN clustering parameters
- ISO 5817 severity thresholds (Moderate quality level)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for DL segmentation models"""
    model_type: str = "pointnet++"  # Options: "pointnet++", "kpconv", "hybrid", "point_transformer"
    
    # Model checkpoint paths (relative to Weld Path folder)
    model_paths: Dict[str, str] = field(default_factory=lambda: {
        "pointnet++": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\PointNet ++\checkpoints\Training 1\best_model.pth",
        "kpconv": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\KPConv\checkpoints\best_model.pth",
    })
    
    # Source code paths for model imports
    source_paths: Dict[str, str] = field(default_factory=lambda: {
        "pointnet++": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\PointNet ++",
        "kpconv": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\KPConv",
    })
    
    num_points: int = 2048  # Points for DL model inference
    num_classes: int = 2    # Binary: weld (1) vs non-weld (0)


@dataclass
class SamplingConfig:
    """Configuration for point cloud sampling"""
    target_points: int = 50000  # Target number of points for processing
    voxel_size: float = 0.5     # Voxel size for downsampling (mm)
    

@dataclass
class AlignmentConfig:
    """Configuration for point cloud alignment (FPFH + ICP)"""
    # FPFH parameters
    fpfh_voxel_size: float = 2.0        # Voxel size for FPFH feature extraction (mm)
    fpfh_radius_normal: float = 4.0     # Radius for normal estimation (2x voxel_size)
    fpfh_radius_feature: float = 8.0    # Radius for FPFH feature computation (4x voxel_size)
    
    # ICP parameters
    icp_max_iterations: int = 100
    icp_threshold: float = 1.0          # Distance threshold for ICP (mm)
    icp_relative_fitness: float = 1e-6
    icp_relative_rmse: float = 1e-6


@dataclass
class ClusteringConfig:
    """Configuration for DBSCAN clustering"""
    eps: Optional[float] = None  # If None, computed adaptively from point density
    min_samples: int = 10        # Minimum points to form a cluster
    eps_percentile: float = 5.0  # Percentile of NN distances for adaptive eps


@dataclass
class DeviationConfig:
    """Configuration for deviation analysis"""
    top_percentile: float = 10.0  # Select top X% of deviations for clustering


@dataclass
class ISO5817ThresholdsModerate:
    """
    ISO 5817 Moderate Quality (Level C) Thresholds
    
    These are typical thresholds for moderate quality welds.
    Values are in millimeters unless otherwise noted.
    
    Reference: ISO 5817:2014 - Welding - Fusion-welded joints in steel, nickel, titanium and their alloys
    """
    # Excess weld metal (excessive convexity)
    excess_weld_metal_max: float = 5.0  # mm (for plate thickness > 3mm)
    
    # Undercut
    undercut_max_depth: float = 0.5     # mm (short, < 25mm)
    undercut_continuous_max: float = 0.3  # mm (continuous)
    
    # Lack of fusion (root gap)
    lack_of_fusion_max: float = 2.0     # mm
    
    # Porosity
    porosity_max_diameter: float = 2.5  # mm (single pore)
    porosity_max_area_percent: float = 2.0  # % of weld area
    
    # Cracks - not permitted in any quality level
    crack_tolerance: float = 0.0
    
    # General geometric deviation thresholds (mm)
    minor_threshold: float = 0.5        # < 0.5mm: minor
    moderate_threshold: float = 1.5     # 0.5-1.5mm: moderate  
    severe_threshold: float = 3.0       # > 1.5mm: severe (reject for moderate quality)


@dataclass 
class SeverityConfig:
    """Severity classification based on ISO 5817 Moderate"""
    thresholds: ISO5817ThresholdsModerate = field(default_factory=ISO5817ThresholdsModerate)
    
    def classify_severity(self, max_deviation: float, rmse: float) -> str:
        """
        Classify defect severity based on maximum deviation
        
        Args:
            max_deviation: Maximum point-to-point deviation in mm
            rmse: Root mean square error of cluster deviations
            
        Returns:
            Severity level: "minor", "moderate", "severe", or "critical"
        """
        if max_deviation < self.thresholds.minor_threshold:
            return "minor"
        elif max_deviation < self.thresholds.moderate_threshold:
            return "moderate"
        elif max_deviation < self.thresholds.severe_threshold:
            return "severe"
        else:
            return "critical"


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    colormap_base: str = "lightblue"      # Color for base material
    colormap_weld: str = "blue"           # Color for weld region
    colormap_defect: str = "red"          # Color for defects
    point_size: int = 2
    defect_point_size: int = 6
    save_html: bool = True
    save_png: bool = True


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline"""
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    deviation: DeviationConfig = field(default_factory=DeviationConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Output settings
    output_dir: str = "results"
    save_intermediate: bool = True  # Save intermediate results for debugging
    verbose: bool = True
    
    def get_model_path(self, model_type: Optional[str] = None) -> str:
        """Get the checkpoint path for specified model type"""
        mt = model_type or self.model.model_type
        if mt not in self.model.model_paths:
            raise ValueError(f"Unknown model type: {mt}. Available: {list(self.model.model_paths.keys())}")
        return self.model.model_paths[mt]
    
    def get_source_path(self, model_type: Optional[str] = None) -> str:
        """Get the source code path for specified model type"""
        mt = model_type or self.model.model_type
        if mt not in self.model.source_paths:
            raise ValueError(f"Unknown model type: {mt}. Available: {list(self.model.source_paths.keys())}")
        return self.model.source_paths[mt]


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()


def create_config(
    model_type: str = "pointnet++",
    output_dir: str = "results",
    top_percentile: float = 10.0,
    **kwargs
) -> PipelineConfig:
    """
    Create a custom pipeline configuration
    
    Args:
        model_type: DL model type ("pointnet++", "kpconv")
        output_dir: Output directory for results
        top_percentile: Top percentile of deviations to cluster
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured PipelineConfig instance
    """
    config = get_default_config()
    config.model.model_type = model_type
    config.output_dir = output_dir
    config.deviation.top_percentile = top_percentile
    return config
