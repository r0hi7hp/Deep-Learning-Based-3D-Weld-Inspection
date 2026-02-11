"""
Visualization Module

Provides 3D visualization for:
- Weld mask overlay
- Distance heatmaps
- Defect clusters
- Interactive HTML reports
"""
import numpy as np
import open3d as o3d
import logging
from pathlib import Path
from typing import List, Optional, Dict
import copy

from .config import PipelineConfig
from .defect_analysis import WeldDefect

logger = logging.getLogger(__name__)


class DefectVisualizer:
    """
    Creates visualizations for defect analysis results
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the visualizer
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.vis_config = config.visualization
    
    def create_weld_mask_visualization(
        self,
        pcd: o3d.geometry.PointCloud,
        weld_mask: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Color point cloud by weld mask
        
        Args:
            pcd: Point cloud
            weld_mask: Boolean mask [N]
            
        Returns:
            Colored point cloud
        """
        colored = copy.deepcopy(pcd)
        points = np.asarray(pcd.points)
        
        colors = np.zeros((len(points), 3))
        colors[~weld_mask] = [0.68, 0.85, 0.9]  # Light blue for base
        colors[weld_mask] = [0.0, 0.0, 1.0]     # Blue for weld
        
        colored.colors = o3d.utility.Vector3dVector(colors)
        return colored
    
    def create_distance_heatmap(
        self,
        pcd: o3d.geometry.PointCloud,
        distances: np.ndarray,
        max_distance: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """
        Create heatmap visualization of distances
        
        Args:
            pcd: Point cloud
            distances: Distance values [N]
            max_distance: Max value for colormap scaling
            
        Returns:
            Colored point cloud
        """
        colored = copy.deepcopy(pcd)
        
        if max_distance is None:
            max_distance = np.percentile(distances, 99)
        
        normalized = np.clip(distances / max_distance, 0, 1)
        
        # Blue (low) -> Yellow -> Red (high)
        colors = np.zeros((len(distances), 3))
        
        mask_low = normalized < 0.5
        t_low = normalized[mask_low] * 2
        colors[mask_low, 0] = t_low
        colors[mask_low, 1] = t_low
        colors[mask_low, 2] = 1 - t_low
        
        mask_high = ~mask_low
        t_high = (normalized[mask_high] - 0.5) * 2
        colors[mask_high, 0] = 1
        colors[mask_high, 1] = 1 - t_high
        colors[mask_high, 2] = 0
        
        colored.colors = o3d.utility.Vector3dVector(colors)
        return colored
    
    def create_cluster_visualization(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
        defects: List[WeldDefect]
    ) -> o3d.geometry.PointCloud:
        """
        Color clusters by severity
        
        Args:
            pcd: Point cloud
            labels: Cluster labels [N]
            defects: List of defect objects
            
        Returns:
            Colored point cloud
        """
        colored = copy.deepcopy(pcd)
        points = np.asarray(pcd.points)
        
        # Default color (light gray for non-defect)
        colors = np.ones((len(points), 3)) * 0.8
        
        # Color by severity
        severity_colors = {
            'minor': [0.0, 1.0, 0.0],      # Green
            'moderate': [1.0, 1.0, 0.0],   # Yellow
            'severe': [1.0, 0.5, 0.0],     # Orange
            'critical': [1.0, 0.0, 0.0]    # Red
        }
        
        for defect in defects:
            mask = labels == defect.cluster_id
            colors[mask] = severity_colors.get(defect.severity, [1.0, 0.0, 1.0])
        
        colored.colors = o3d.utility.Vector3dVector(colors)
        return colored
    
    def save_plotly_visualization(
        self,
        points: np.ndarray,
        weld_mask: np.ndarray,
        distances: np.ndarray,
        labels: np.ndarray,
        defects: List[WeldDefect],
        output_path: str
    ) -> None:
        """
        Create interactive Plotly visualization
        
        Args:
            points: Point coordinates [N, 3]
            weld_mask: Weld region mask [N]
            distances: Distance values [N]
            labels: Cluster labels [N]
            defects: List of defects
            output_path: Output HTML file path
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not installed, skipping interactive visualization")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Downsample for performance
        max_points = 50000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            weld_mask = weld_mask[indices]
            distances = distances[indices]
            labels = labels[indices]
        
        # Create traces
        traces = []
        
        # Non-weld points (background)
        non_weld_mask = ~weld_mask
        if np.sum(non_weld_mask) > 0:
            traces.append(go.Scatter3d(
                x=points[non_weld_mask, 0],
                y=points[non_weld_mask, 1],
                z=points[non_weld_mask, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.3),
                name='Base Material'
            ))
        
        # Weld points (non-defect)
        weld_non_defect = weld_mask & (labels == -1)
        if np.sum(weld_non_defect) > 0:
            traces.append(go.Scatter3d(
                x=points[weld_non_defect, 0],
                y=points[weld_non_defect, 1],
                z=points[weld_non_defect, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.5),
                name='Weld Region'
            ))
        
        # Defect clusters by severity
        severity_colors = {
            'minor': 'green',
            'moderate': 'yellow',
            'severe': 'orange',
            'critical': 'red'
        }
        
        for defect in defects:
            mask = labels == defect.cluster_id
            if np.sum(mask) > 0:
                traces.append(go.Scatter3d(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    z=points[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=severity_colors.get(defect.severity, 'magenta'),
                        opacity=1.0
                    ),
                    name=f'Defect {defect.cluster_id} ({defect.severity})',
                    hovertemplate=(
                        f'<b>Defect {defect.cluster_id}</b><br>'
                        f'Severity: {defect.severity}<br>'
                        f'Max Dev: {defect.max_deviation:.3f} mm<br>'
                        f'RMSE: {defect.rmse:.3f} mm'
                        '<extra></extra>'
                    )
                ))
        
        # Create figure
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            title='Weld Defect Mapping Results',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive visualization to {output_path}")
    
    def save_matplotlib_visualization(
        self,
        points: np.ndarray,
        weld_mask: np.ndarray,
        labels: np.ndarray,
        defects: List[WeldDefect],
        output_path: str
    ) -> None:
        """
        Create static matplotlib visualization
        
        Args:
            points: Point coordinates [N, 3]
            weld_mask: Weld region mask [N]
            labels: Cluster labels [N]
            defects: List of defects
            output_path: Output image path
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("Matplotlib not installed, skipping static visualization")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Downsample for plotting
        max_points = 20000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            weld_mask = weld_mask[indices]
            labels = labels[indices]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot base material
        base_mask = ~weld_mask
        if np.sum(base_mask) > 0:
            ax.scatter(
                points[base_mask, 0],
                points[base_mask, 1],
                points[base_mask, 2],
                c='lightgray', s=1, alpha=0.2, label='Base'
            )
        
        # Plot weld region
        weld_ok = weld_mask & (labels == -1)
        if np.sum(weld_ok) > 0:
            ax.scatter(
                points[weld_ok, 0],
                points[weld_ok, 1],
                points[weld_ok, 2],
                c='blue', s=2, alpha=0.4, label='Weld'
            )
        
        # Plot defects
        severity_colors = {
            'minor': 'green',
            'moderate': 'yellow',
            'severe': 'orange',
            'critical': 'red'
        }
        
        for defect in defects:
            mask = labels == defect.cluster_id
            if np.sum(mask) > 0:
                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    points[mask, 2],
                    c=severity_colors.get(defect.severity, 'magenta'),
                    s=10, alpha=0.9,
                    label=f'Defect {defect.cluster_id}'
                )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Weld Defect Mapping Results')
        ax.legend(loc='upper left', fontsize=8)
        
        # Auto-scale
        center = points.mean(axis=0)
        max_range = (points.max(axis=0) - points.min(axis=0)).max() * 0.6
        for axis, coord in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            axis(coord - max_range, coord + max_range)
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        
        logger.info(f"Saved static visualization to {output_path}")
    
    def visualize_all(
        self,
        points: np.ndarray,
        weld_mask: np.ndarray,
        distances: np.ndarray,
        labels: np.ndarray,
        defects: List[WeldDefect],
        output_dir: str,
        base_name: str = "defect_analysis"
    ) -> Dict[str, str]:
        """
        Create all visualizations
        
        Args:
            points: Point coordinates
            weld_mask: Weld mask
            distances: Distance values
            labels: Cluster labels
            defects: List of defects
            output_dir: Output directory
            base_name: Base filename
            
        Returns:
            Dictionary of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        # Interactive HTML
        if self.vis_config.save_html:
            html_path = output_dir / f"{base_name}_interactive.html"
            self.save_plotly_visualization(
                points, weld_mask, distances, labels, defects, str(html_path)
            )
            created_files['html'] = str(html_path)
        
        # Static PNG
        if self.vis_config.save_png:
            png_path = output_dir / f"{base_name}_visualization.png"
            self.save_matplotlib_visualization(
                points, weld_mask, labels, defects, str(png_path)
            )
            created_files['png'] = str(png_path)
        
        return created_files
