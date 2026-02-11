"""
Quick pipeline test script

Tests the pipeline modules individually to verify they work correctly.
"""
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import create_config
from src.point_cloud_prep import PointCloudPreparator
from src.distance_computation import DistanceComputer
from src.defect_filter import DefectFilter
from src.clustering import DefectClusterer
from src.defect_analysis import DefectAnalyzer
from src.visualization import DefectVisualizer


def test_pipeline():
    """Test the pipeline with sample data"""
    print("="*60)
    print("WELD DEFECT MAPPING - QUICK TEST")
    print("="*60)
    
    # Paths
    scan_path = "Models/model_1/scan.ply"
    nominal_path = "Models/model_1/actual.ply"
    output_dir = Path("results/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading point clouds...")
    scan_pcd = o3d.io.read_point_cloud(scan_path)
    nominal_pcd = o3d.io.read_point_cloud(nominal_path)
    
    scan_points = np.asarray(scan_pcd.points)
    nominal_points = np.asarray(nominal_pcd.points)
    
    print(f"    Scan: {len(scan_points)} points")
    print(f"    Nominal weld path: {len(nominal_points)} points")
    
    # Create config
    config = create_config(output_dir=str(output_dir))
    
    # Downsample scan for faster processing
    print("\n[2] Downsampling scan...")
    prep = PointCloudPreparator(config)
    scan_pcd_down = prep.uniform_downsample(scan_pcd, target_points=50000)
    scan_points_down = np.asarray(scan_pcd_down.points)
    print(f"    Downsampled to: {len(scan_points_down)} points")
    
    # Simulate weld mask (for testing, mark points close to nominal path as weld)
    print("\n[3] Creating simulated weld mask...")
    from scipy.spatial import KDTree
    nominal_tree = KDTree(nominal_points)
    distances_to_nominal, _ = nominal_tree.query(scan_points_down, k=1)
    
    # Points within 2mm of nominal path are considered weld
    weld_threshold = 2.0
    weld_mask = distances_to_nominal < weld_threshold
    print(f"    Weld points: {np.sum(weld_mask)} ({100*np.sum(weld_mask)/len(weld_mask):.1f}%)")
    
    # Compute distances for all points
    print("\n[4] Computing distances from weld points to nominal path...")
    # Get only weld points
    weld_points = scan_points_down[weld_mask]
    
    # Distance computation
    dist_computer = DistanceComputer(config)
    weld_distances_to_nominal, _ = nominal_tree.query(weld_points, k=1)
    
    print(f"    Distance stats: min={weld_distances_to_nominal.min():.3f}, "
          f"max={weld_distances_to_nominal.max():.3f}, mean={weld_distances_to_nominal.mean():.3f}")
    
    # Filter top deviations
    print("\n[5] Filtering top deviations...")
    defect_filter = DefectFilter(config)
    top_mask, threshold = defect_filter.select_top_percentile(weld_distances_to_nominal, percentile=20.0)
    
    selected_weld_points = weld_points[top_mask]
    selected_distances = weld_distances_to_nominal[top_mask]
    print(f"    Selected {len(selected_weld_points)} high-deviation points (threshold={threshold:.3f})")
    
    # Cluster
    print("\n[6] Clustering defects...")
    clusterer = DefectClusterer(config)
    if len(selected_weld_points) > 0:
        labels = clusterer.cluster_deviations(selected_weld_points, selected_distances)
        cluster_info = clusterer.get_cluster_info(selected_weld_points, labels, selected_distances)
        print(f"    Found {len(cluster_info)} defect clusters")
    else:
        labels = np.array([])
        cluster_info = []
        print("    No high-deviation points found")
    
    # Analyze
    print("\n[7] Analyzing defects...")
    analyzer = DefectAnalyzer(config)
    defects = analyzer.localize_defects(selected_weld_points, labels, selected_distances, cluster_info)
    
    for d in defects[:5]:  # Show first 5
        print(f"    Defect {d.cluster_id}: severity={d.severity}, max_dev={d.max_deviation:.3f}mm")
    
    # Visualize
    print("\n[8] Creating visualization...")
    
    # Create full labels array for visualization (map back to downsampled scan)
    full_labels = np.full(len(scan_points_down), -2, dtype=int)  # -2 = non-weld
    full_labels[weld_mask] = -1  # -1 = weld but not defect
    
    # Map cluster labels to weld points
    weld_indices = np.where(weld_mask)[0]
    selected_weld_indices = weld_indices[top_mask]
    for i, idx in enumerate(selected_weld_indices):
        if i < len(labels):
            full_labels[idx] = labels[i]
    
    # Create distances array (0 for non-weld, actual distance for weld)
    full_distances = np.zeros(len(scan_points_down))
    full_distances[weld_mask] = weld_distances_to_nominal
    
    visualizer = DefectVisualizer(config)
    visualizer.visualize_all(
        scan_points_down,
        weld_mask,
        full_distances,
        full_labels,
        defects,
        str(output_dir)
    )
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
