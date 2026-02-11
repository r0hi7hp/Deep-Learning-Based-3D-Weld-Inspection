"""
Weld Defect Mapping Pipeline - Interactive Main Script

This script:
1. Preprocesses CAD models (aligns and samples to same point count)
2. Prompts for DL model path to segment weld area
3. Runs defect mapping on the segmented weld region
"""
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from src.distance_computation import DistanceComputer

# Add project root and weld path modules to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for preprocessing and defect detection"""
    # ICP alignment
    icp_samples: int = 15000
    icp_max_distance: float = 2.0
    
    # Point sampling
    sample_count: int = 50000
    
    # Weld detection (from DL model)
    num_points_dl: int = 2048
    
    # Defect detection
    top_percentile: float = 10.0
    dbscan_eps: float = 2.0
    dbscan_min_samples: int = 10
    
    # ISO 5817 Moderate thresholds (mm)
    minor_threshold: float = 0.5
    moderate_threshold: float = 1.5
    severe_threshold: float = 3.0


def load_mesh_or_pointcloud(path: str) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    """Load mesh or point cloud from file"""
    path = Path(path)
    logger.info(f"Loading: {path}")
    
    mesh = None
    if path.suffix.lower() in ['.stl', '.obj']:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.is_empty():
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_poisson_disk(50000)
            logger.info(f"  Loaded mesh, sampled to {len(pcd.points)} points")
            return pcd, mesh
    
    # Try as point cloud
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        logger.info(f"  Loaded point cloud with {len(pcd.points)} points")
        return pcd, None
    
    # Try reading as mesh then converting
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty():
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(50000)
        logger.info(f"  Loaded as mesh, sampled to {len(pcd.points)} points")
        return pcd, mesh
    
    raise ValueError(f"Could not load file: {path}")


def align_point_clouds(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    config: ProcessingConfig
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Align source to target using ICP"""
    logger.info("Aligning point clouds...")
    
    # Downsample for ICP
    source_down = source_pcd.voxel_down_sample(voxel_size=1.0)
    target_down = target_pcd.voxel_down_sample(voxel_size=1.0)
    
    # Estimate normals
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    
    # Run ICP
    logger.info("  Running ICP alignment...")
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=config.icp_max_distance,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    logger.info(f"  ICP fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")
    
    # Apply transformation
    aligned = source_pcd.transform(icp_result.transformation)
    
    return aligned, icp_result.transformation


def sample_to_target_count(pcd: o3d.geometry.PointCloud, target_count: int) -> o3d.geometry.PointCloud:
    """Sample point cloud to target number of points"""
    current_count = len(pcd.points)
    
    if current_count == target_count:
        return pcd
    
    if current_count > target_count:
        # Downsample using voxel
        voxel_size = 0.5
        result = pcd.voxel_down_sample(voxel_size)
        
        # Iteratively adjust voxel size
        while len(result.points) > target_count and voxel_size < 20:
            voxel_size *= 1.2
            result = pcd.voxel_down_sample(voxel_size)
        
        # Random sample if still too many
        if len(result.points) > target_count:
            indices = np.random.choice(len(result.points), target_count, replace=False)
            result = result.select_by_index(indices)
    else:
        # Upsample (keep all points)
        result = pcd
    
    logger.info(f"  Sampled from {current_count} to {len(result.points)} points")
    return result


# Default model paths
DEFAULT_MODELS = {
    "pointnet++": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\PointNet ++\checkpoints\Training 1\best_model.pth",
    "kpconv": r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\KPConv\checkpoints\best_model.pth",
}


def get_dl_model_path(model_arg: Optional[str] = None) -> Tuple[str, str]:
    """Get DL model path - from argument or interactively"""
    
    # If model specified via CLI argument
    if model_arg:
        model_type = model_arg.lower()
        if model_type in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[model_type]
            if Path(model_path).exists():
                logger.info(f"Using {model_type} model: {model_path}")
                return model_type, model_path
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        else:
            # Assume it's a custom path
            if Path(model_arg).exists():
                # Try to detect model type from path
                if 'pointnet' in model_arg.lower():
                    model_type = 'pointnet++'
                elif 'kpconv' in model_arg.lower():
                    model_type = 'kpconv'
                else:
                    model_type = 'pointnet++'
                logger.info(f"Using custom model: {model_arg} (type: {model_type})")
                return model_type, model_arg
            raise FileNotFoundError(f"Model not found: {model_arg}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("SELECT DEEP LEARNING MODEL FOR WELD SEGMENTATION")
    print("="*60)
    
    print("\nAvailable models:")
    for i, (name, path) in enumerate(DEFAULT_MODELS.items(), 1):
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  [{i}] {name}: {status}")
    print("  [3] Enter custom path")
    
    try:
        choice = input("\nSelect model (1/2/3): ").strip()
    except EOFError:
        logger.warning("No stdin available, using PointNet++ as default")
        choice = "1"
    
    if choice == "1":
        model_type = "pointnet++"
        model_path = DEFAULT_MODELS["pointnet++"]
    elif choice == "2":
        model_type = "kpconv"
        model_path = DEFAULT_MODELS["kpconv"]
    elif choice == "3":
        model_path = input("Enter model path: ").strip()
        model_type = input("Enter model type (pointnet++/kpconv): ").strip().lower()
    else:
        logger.warning("Invalid choice, using PointNet++ as default")
        model_type = "pointnet++"
        model_path = DEFAULT_MODELS["pointnet++"]
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nSelected: {model_type} at {model_path}")
    return model_type, model_path


def segment_weld_region(
    pcd: o3d.geometry.PointCloud,
    model_type: str,
    model_path: str,
    config: ProcessingConfig
) -> np.ndarray:
    """Segment weld region using DL model"""
    import torch
    
    points = np.asarray(pcd.points)
    logger.info(f"Segmenting weld region with {model_type}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Using device: {device}")
    
    # Normalize points
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    if scale == 0:
        scale = 1.0
    points_norm = points_centered / scale
    
    # Sample for model input
    num_points = config.num_points_dl
    if len(points_norm) > num_points:
        choice = np.random.choice(len(points_norm), num_points, replace=False)
    else:
        choice = np.random.choice(len(points_norm), num_points, replace=True)
    points_input = points_norm[choice]
    
    # Load model based on type
    if model_type == "pointnet++":
        # Add PointNet++ path
        pointnet_path = r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\PointNet ++"
        if pointnet_path not in sys.path:
            sys.path.insert(0, pointnet_path)
        
        from pointnet2 import PointNet2SemSeg
        model = PointNet2SemSeg(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
    elif model_type == "kpconv":
        # Add KPConv path
        kpconv_path = r"C:\Users\rohit\Desktop\Weld Inspection\Weld Path\KPConv"
        if kpconv_path not in sys.path:
            sys.path.insert(0, kpconv_path)
        
        from kpconv_model import KPConvSegmentation
        model = KPConvSegmentation(in_channels=3, num_classes=2, num_layers=4, init_features=64).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Inference
    points_tensor = torch.from_numpy(points_input).float().to(device)
    points_tensor = points_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, 3, N]
    
    with torch.no_grad():
        outputs = model(points_tensor)
        pred = outputs.max(dim=1)[1].cpu().numpy()[0]
    
    # Interpolate to all points
    logger.info("  Interpolating labels to all points...")
    tree = KDTree(points_input)
    _, indices = tree.query(points_norm, k=1)
    labels = pred[indices]
    
    weld_count = np.sum(labels == 1)
    logger.info(f"  Weld points: {weld_count}/{len(labels)} ({100*weld_count/len(labels):.1f}%)")
    
    return labels


def detect_defects(
    points: np.ndarray,
    weld_mask: np.ndarray,
    ref_pcd: o3d.geometry.PointCloud,
    config: ProcessingConfig
) -> dict:
    """Detect defects in weld region by comparing to reference"""
    logger.info("Detecting defects...")
    
    # Get weld points
    weld_mask_indices = np.where(weld_mask)[0]
    weld_points = points[weld_mask]
    
    if len(weld_points) == 0:
        logger.warning("No weld points found!")
        return {"defects": [], "weld_points": 0}
    
    # Compute signed distances to reference (for defect type classification)
    logger.info("  Computing signed distances to reference...")
    
    # We need to creat temporary point clouds for the DistanceComputer
    weld_pcd = o3d.geometry.PointCloud()
    weld_pcd.points = o3d.utility.Vector3dVector(weld_points)
    
    dist_computer = DistanceComputer(config)
    signed_distances = dist_computer.compute_signed_distances(weld_pcd, ref_pcd)
    abs_distances = np.abs(signed_distances)
    
    logger.info(f"  Distance stats: min={signed_distances.min():.3f}, max={signed_distances.max():.3f}, mean={np.mean(abs_distances):.3f}")
    
    # Select top deviations (based on absolute distance)
    threshold = np.percentile(abs_distances, 100 - config.top_percentile)
    top_mask = abs_distances >= threshold
    
    selected_points = weld_points[top_mask]
    selected_abs_distances = abs_distances[top_mask]
    selected_signed_distances = signed_distances[top_mask]
    
    logger.info(f"  Selected {len(selected_points)} high-deviation points (threshold={threshold:.3f})")
    
    if len(selected_points) < config.dbscan_min_samples:
        logger.info("  Not enough high-deviation points for clustering")
        return {"defects": [], "weld_points": len(weld_points)}
    
    # DBSCAN clustering
    logger.info("  Clustering defects with DBSCAN...")
    clustering = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
    labels = clustering.fit_predict(selected_points)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"  Found {n_clusters} defect clusters")
    
    # Analyze each cluster
    defects = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_points = selected_points[mask]
        cluster_dists = selected_abs_distances[mask]
        cluster_signed = selected_signed_distances[mask]
        
        max_dev = float(np.max(cluster_dists))
        rmse = float(np.sqrt(np.mean(cluster_dists ** 2)))
        mean_signed = float(np.mean(cluster_signed))
        std_signed = float(np.std(cluster_signed))
        
        # Compute bounding box and geometry metrics
        bb_min = cluster_points.min(axis=0)
        bb_max = cluster_points.max(axis=0)
        bb_size = bb_max - bb_min
        length = np.linalg.norm(bb_size)  # diagonal length
        height = bb_size[2]  # Z is typically height/outward
        width = np.sqrt(bb_size[0]**2 + bb_size[1]**2)  # XY extent
        num_pts = len(cluster_points)
        
        # Compute additional metrics for classification
        aspect_ratio = length / max(height, 0.1) if height > 0.1 else length
        point_density = num_pts / max(length, 0.1)
        signed_variance = std_signed / max(abs(mean_signed), 0.1) if abs(mean_signed) > 0.1 else std_signed
        
        # Comprehensive ISO 5817 classification with multiple defect types
        # Uses hierarchical decision tree based on geometry and deviation characteristics
        
        if mean_signed < -config.moderate_threshold:
            # Strong negative deviation = material missing/undercut
            if length > config.dbscan_eps * 8:
                defect_type = "Continuous undercut"
                iso_code = "5011"
            else:
                defect_type = "Undercut"
                iso_code = "501"
        
        elif mean_signed > config.moderate_threshold * 1.5:
            # Strong positive deviation = excess material
            if height > width * 0.5:
                defect_type = "Excess weld metal"
                iso_code = "502"
            elif length > config.dbscan_eps * 10:
                defect_type = "Overlap"
                iso_code = "506"
            else:
                defect_type = "Weld reinforcement excess"
                iso_code = "503"
        
        elif num_pts < 100:
            # Very small cluster = porosity or gas pore
            if max_dev > config.moderate_threshold:
                defect_type = "Surface pore"
                iso_code = "2017"
            else:
                defect_type = "Gas pore"
                iso_code = "2011"
        
        elif num_pts < 300 and signed_variance > 1.0:
            # Small cluster with high variance = porosity cluster
            defect_type = "Clustered porosity"
            iso_code = "2013"
        
        elif aspect_ratio > 5 and length > config.dbscan_eps * 5:
            # Long, thin defect
            if mean_signed < 0:
                defect_type = "Linear misalignment"
                iso_code = "507"
            else:
                defect_type = "Irregular bead"
                iso_code = "514"
        
        elif height > config.severe_threshold:
            # Very tall protrusion
            defect_type = "Excessive convexity"
            iso_code = "504"
        
        elif point_density < 10:
            # Sparse points = spatter or arc strike
            defect_type = "Spatter"
            iso_code = "602"
        
        elif abs(mean_signed) < config.minor_threshold and max_dev > config.moderate_threshold:
            # Low mean but high max = localized defect
            defect_type = "Incompletely filled groove"
            iso_code = "511"
        
        else:
            # Default: irregular profile
            defect_type = "Irregular profile"
            iso_code = "505"
            
        # Classify severity
        if max_dev < config.minor_threshold:
            severity = "minor"
        elif max_dev < config.moderate_threshold:
            severity = "moderate"
        elif max_dev < config.severe_threshold:
            severity = "severe"
        else:
            severity = "critical"
        
        defect = {
            "cluster_id": cluster_id,
            "center": np.mean(cluster_points, axis=0).tolist(),
            "points": cluster_points.tolist(),
            "num_points": int(np.sum(mask)),
            "max_deviation": max_dev,
            "mean_signed_deviation": mean_signed,
            "rmse": rmse,
            "severity": severity,
            "type": defect_type,
            "iso_code": iso_code,
            "bbox_min": bb_min.tolist(),
            "bbox_max": bb_max.tolist()
        }
        defects.append(defect)
        
        logger.info(f"    Defect {cluster_id}: {defect_type} (ISO 5817 {iso_code}), severity={severity}, max_dev={max_dev:.3f}mm")
    
    return {
        "defects": defects,
        "weld_points": len(weld_points),
        "high_deviation_points": len(selected_points),
        "threshold": float(threshold),
        "cluster_labels": labels.tolist(),
        "selected_points": selected_points.tolist()
    }


def save_results(
    output_dir: Path,
    scan_points: np.ndarray,
    weld_mask: np.ndarray,
    defect_results: dict
):
    """Save results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ with segmentation
    np.savez(
        output_dir / "segmentation.npz",
        points=scan_points,
        labels=weld_mask.astype(np.uint8)
    )
    
    # Save defect report
    import json
    with open(output_dir / "defect_report.json", "w") as f:
        json.dump(defect_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    # Create visualization
    try:
        import plotly.graph_objects as go
        import matplotlib.cm as cm
        
        # Collect all defect point indices to exclude from base
        defect_point_set = set()
        for defect in defect_results.get('defects', []):
            for pt in defect.get('points', []):
                defect_point_set.add(tuple(pt))
        
        # Create base material trace (ALL points that are NOT in defect clusters)
        # Show everything in blue like the reference image
        base_points = []
        for i, pt in enumerate(scan_points):
            if tuple(pt) not in defect_point_set:
                base_points.append(pt)
        base_points = np.array(base_points) if base_points else np.array([]).reshape(0, 3)
        
        traces = []
        
        # Base material - SOLID BLUE (entire model except defects)
        if len(base_points) > 0:
            # Downsample base for performance
            max_base = 40000
            if len(base_points) > max_base:
                idx = np.random.choice(len(base_points), max_base, replace=False)
                vis_base = base_points[idx]
            else:
                vis_base = base_points
            
            traces.append(go.Scatter3d(
                x=vis_base[:, 0],
                y=vis_base[:, 1],
                z=vis_base[:, 2],
                mode='markers',
                marker=dict(size=2, color='steelblue', opacity=0.6),
                name='Base Material',
                hoverinfo='skip'  # Don't show hover for base
            ))
        
        # Use tab20 colormap for distinct defect colors
        n_defects = len(defect_results.get('defects', []))
        if n_defects > 0:
            cmap = cm.get_cmap('tab20', max(n_defects, 20))
        
        # Render each defect cluster with its OWN unique color
        for i, defect in enumerate(defect_results.get('defects', [])):
            defect_type = defect.get('type', 'Unknown')
            iso_code = defect.get('iso_code', '')
            cluster_points = np.array(defect.get('points', []))
            
            if len(cluster_points) == 0:
                continue
            
            # Get unique color from tab20 colormap
            rgba = cmap(i % 20)
            color = f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'
            
            # Label with defect type and ISO code
            label = f"{defect_type} (ISO 5817 {iso_code})" if iso_code else defect_type
            
            # Add all points in the cluster with this color
            traces.append(go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.95),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Max Dev: {defect['max_deviation']:.2f}mm<br>"
                    f"Points: {defect['num_points']}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Weld Defect Mapping Results',
            scene=dict(
                aspectmode='data',
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)")
        )
        fig.write_html(str(output_dir / "visualization.html"))
        logger.info(f"Saved interactive visualization")
        
    except ImportError:
        logger.warning("Plotly not installed, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description='Weld Defect Mapping Pipeline')
    parser.add_argument('--scan', '-s', required=True, help='Path to scan/defective model (PLY/STL)')
    parser.add_argument('--reference', '-r', required=True, help='Path to reference/CAD model (PLY/STL)')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--model', '-m', default=None, 
                        help='DL model: pointnet++ or kpconv, or path to custom model')
    parser.add_argument('--sample-count', type=int, default=50000, help='Target point count for both models')
    parser.add_argument('--skip-alignment', action='store_true', help='Skip ICP alignment')
    
    args = parser.parse_args()
    
    config = ProcessingConfig(sample_count=args.sample_count)
    output_dir = Path(args.output)
    
    print("\n" + "="*60)
    print("WELD DEFECT MAPPING PIPELINE")
    print("="*60)
    
    # Step 1: Load models
    print("\n[STEP 1/5] Loading CAD Models")
    scan_pcd, _ = load_mesh_or_pointcloud(args.scan)
    ref_pcd, _ = load_mesh_or_pointcloud(args.reference)
    
    # Step 2: Preprocess - sample to same point count
    print("\n[STEP 2/5] Preprocessing - Uniform Sampling")
    scan_pcd = sample_to_target_count(scan_pcd, config.sample_count)
    ref_pcd = sample_to_target_count(ref_pcd, config.sample_count)
    
    # Step 3: Align
    if not args.skip_alignment:
        print("\n[STEP 3/5] Aligning Point Clouds")
        scan_pcd, transform = align_point_clouds(scan_pcd, ref_pcd, config)
    else:
        print("\n[STEP 3/5] Skipping alignment (--skip-alignment)")
    
    scan_points = np.asarray(scan_pcd.points)
    ref_points = np.asarray(ref_pcd.points)
    
    # Step 4: Get DL model and segment weld
    print("\n[STEP 4/5] Weld Segmentation")
    model_type, model_path = get_dl_model_path(args.model)
    weld_labels = segment_weld_region(scan_pcd, model_type, model_path, config)
    weld_mask = weld_labels == 1
    
    # Step 5: Detect defects
    print("\n[STEP 5/5] Defect Detection")
    # Step 5: Detect defects
    print("\n[STEP 5/5] Defect Detection")
    # Need reference PCD for signed distance computation
    defect_results = detect_defects(scan_points, weld_mask, ref_pcd, config)
    
    # Save results
    print("\n" + "-"*60)
    save_results(output_dir, scan_points, weld_mask, defect_results)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total points: {len(scan_points)}")
    print(f"Weld region points: {defect_results['weld_points']}")
    print(f"Defects found: {len(defect_results['defects'])}")
    
    if defect_results['defects']:
        print("\nDefects by severity:")
        severity_count = {}
        for d in defect_results['defects']:
            severity_count[d['severity']] = severity_count.get(d['severity'], 0) + 1
        for sev, count in severity_count.items():
            print(f"  {sev.capitalize()}: {count}")
    
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("Done!\n")


if __name__ == "__main__":
    main()
