"""
Weld Path Detection Web UI - Flask Backend
Provides endpoints for CAD model and DL model uploads with weld path prediction
Supports multiple model architectures: Hybrid, Point Transformer, KPConv
"""
import os
import sys
import json
import tempfile
import logging
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import open3d as o3d

# Add parent directory and model directories to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Hybrid Model"))
sys.path.insert(0, str(BASE_DIR / "Point Transformer"))
sys.path.insert(0, str(BASE_DIR / "KPConv"))
sys.path.insert(0, str(BASE_DIR / "PointNet ++"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_CAD_EXTENSIONS = {'ply', 'stl'}
ALLOWED_MODEL_EXTENSIONS = {'pth'}

# File size limits
MAX_CAD_SIZE_MB = 100
MAX_MODEL_SIZE_MB = 500

# Global variables to store uploaded files
current_cad_path = None
current_model_path = None
current_model = None
current_architecture = None


def allowed_cad_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CAD_EXTENSIONS


def allowed_model_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS


def validate_file_size(file, file_type):
    """Validate uploaded file size"""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Seek back to start
    
    if file_type == 'cad':
        if size > MAX_CAD_SIZE_MB * 1024 * 1024:
            raise ValueError(f"CAD file too large (max {MAX_CAD_SIZE_MB}MB)")
        if size == 0:
            raise ValueError("CAD file is empty")
    elif file_type == 'model':
        if size > MAX_MODEL_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Model file too large (max {MAX_MODEL_SIZE_MB}MB)")
        if size == 0:
            raise ValueError("Model file is empty")
    
    return size


def load_mesh(file_path):
    """Load point cloud from PLY or mesh from STL file with robust error handling"""
    logger.info(f"Loading from: {file_path}")
    
    try:
        file_ext = str(file_path).lower().split('.')[-1]
        
        if file_ext == 'ply':
            # Try to load as point cloud first
            pcd = o3d.io.read_point_cloud(str(file_path))
            if pcd.is_empty() or len(np.asarray(pcd.points)) == 0:
                # Fallback: try loading as mesh and sample
                logger.info("PLY has no points, trying as mesh...")
                mesh = o3d.io.read_triangle_mesh(str(file_path))
                if mesh.is_empty():
                    raise ValueError("Failed to load PLY file as point cloud or mesh")
                # Compute mesh properties for better sampling
                mesh.compute_vertex_normals()
                pcd = mesh.sample_points_poisson_disk(20000)
            points = np.asarray(pcd.points, dtype=np.float32)
        
        elif file_ext == 'stl':
            # STL - load as mesh and sample points
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            if mesh.is_empty():
                raise ValueError("Failed to load STL mesh or mesh is empty")
            
            # Compute normals for better sampling
            mesh.compute_vertex_normals()
            
            # Sample points from mesh
            num_points = 20000
            pcd = mesh.sample_points_poisson_disk(num_points)
            if pcd.is_empty() or len(np.asarray(pcd.points)) == 0:
                # Fallback to uniform sampling
                pcd = mesh.sample_points_uniformly(num_points)
            
            points = np.asarray(pcd.points, dtype=np.float32)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if len(points) == 0:
            raise ValueError("No points could be extracted from the file")
        
        logger.info(f"Loaded {len(points)} points")
        return points
        
    except Exception as e:
        logger.error(f"Error loading mesh: {e}")
        raise ValueError(f"Failed to load CAD file: {str(e)}")


def simple_normalize(pc):
    """
    Simple normalization matching training scripts.
    Centers to centroid and scales by max distance.
    """
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc_centered ** 2, axis=1)))
    if max_dist > 1e-6:
        pc_centered = pc_centered / max_dist
    return pc_centered, centroid, max_dist


def needs_orientation_fix(points):
    """
    Check if point cloud orientation differs significantly from training data.
    Training data has Y as the longest axis. If a different axis is longest,
    we need to apply PCA alignment.
    
    Returns:
        bool: True if orientation fix is needed, False otherwise
    """
    ranges = np.array([
        points[:, 0].max() - points[:, 0].min(),  # X range
        points[:, 1].max() - points[:, 1].min(),  # Y range
        points[:, 2].max() - points[:, 2].min()   # Z range
    ])
    
    longest_axis = np.argmax(ranges)
    axis_names = ['X', 'Y', 'Z']
    
    # Training data has Y (index 1) as the longest axis
    # If a different axis is longest, we need alignment
    needs_fix = longest_axis != 1
    
    if needs_fix:
        logger.info(f"Orientation check: Longest axis is {axis_names[longest_axis]} (ranges: X={ranges[0]:.1f}, Y={ranges[1]:.1f}, Z={ranges[2]:.1f}). Training expects Y. Will apply PCA alignment.")
    else:
        logger.info(f"Orientation check: Longest axis is Y (matching training). No alignment needed.")
    
    return needs_fix


def pca_align(points):
    """
    PCA-based orientation alignment to handle CAD models in different planes.
    Aligns point cloud principal axes to match training data orientation:
    - Longest axis -> Y (vertical, matching training data's [0, 50] range)
    - Second longest -> X
    - Shortest -> Z
    
    Args:
        points: [N, 3] numpy array
    Returns:
        aligned_points: [N, 3] numpy array with aligned orientation
        rotation_matrix: [3, 3] rotation matrix used for alignment
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute covariance matrix and PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalues (ascending), then reverse to get descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Training data has Y as the longest axis (0 to 50), X and Z symmetric
    # We need to reorder: longest->Y, second->X, shortest->Z
    # Current order after sorting: [longest, second, shortest] = columns [0, 1, 2]
    # Target order: X=second(1), Y=longest(0), Z=shortest(2)
    reorder = [1, 0, 2]  # Map: new_X=old_second, new_Y=old_longest, new_Z=old_shortest
    rotation_matrix = eigenvectors[:, reorder]
    
    # Ensure right-handed coordinate system (det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    
    # Apply rotation
    aligned = centered @ rotation_matrix
    
    # Check if Y should be flipped (training data has Y in [0, 50], positive)
    # If aligned Y is mostly negative, flip it
    if np.mean(aligned[:, 1]) < 0:
        aligned[:, 1] *= -1
        rotation_matrix[:, 1] *= -1
    
    logger.info(f"PCA alignment: eigenvalues={eigenvalues}, applied reordering to match training orientation")
    
    return aligned, rotation_matrix


def adaptive_normalize_kpconv(points):
    """
    Adaptive normalization for KPConv matching its training script.
    Centers to mean and scales using 95th percentile to handle outliers.
    """
    # Center to mean
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale using 95th percentile to handle outliers
    distances = np.linalg.norm(points, axis=1)
    scale = np.percentile(distances, 95)
    
    if scale > 1e-6:
        points = points / scale
    
    return points


def predict_point_cloud_kpconv(model, points, device, num_sample_points=2048, use_pca_align=False):
    """
    KPConv-specific prediction using sliding window approach with vote aggregation.
    Matches the predict_kpconv.py standalone script's approach.
    
    Args:
        model: trained KPConv model
        points: [N, 3] point coordinates
        device: torch device
        num_sample_points: points per window
        use_pca_align: 'auto' (default) - auto-detect based on orientation
                       True - always apply PCA alignment
                       False - never apply PCA alignment
    Returns:
        labels: [N] predicted labels
        confidence: [N] prediction confidence
    """
    model.eval()
    
    N = len(points)
    all_votes = np.zeros((N, 2), dtype=np.float32)
    vote_counts = np.zeros(N, dtype=np.float32)
    
    # Determine if we need PCA alignment
    if use_pca_align == 'auto':
        apply_alignment = needs_orientation_fix(points)
    else:
        apply_alignment = use_pca_align
    
    # Apply PCA alignment if needed
    if apply_alignment:
        logger.info("Applying PCA alignment for KPConv orientation normalization...")
        points_aligned, _ = pca_align(points.copy())
    else:
        points_aligned = points.copy()
    
    # Normalize points using adaptive normalization
    points_normalized = adaptive_normalize_kpconv(points_aligned)
    
    # Create overlapping windows
    num_windows = max(1, (N + num_sample_points // 2) // num_sample_points)
    
    logger.info(f"KPConv prediction: {N} points, {num_windows} windows")
    
    with torch.no_grad():
        for window_idx in range(num_windows):
            # Sample points for this window
            if N <= num_sample_points:
                indices = np.arange(N)
                if N < num_sample_points:
                    # Pad with repeated points
                    pad_indices = np.random.choice(N, num_sample_points - N, replace=True)
                    indices = np.concatenate([indices, pad_indices])
            else:
                # Random sampling with overlap - prioritize nearby points
                center_idx = (window_idx * N) // num_windows
                distances = np.abs(np.arange(N) - center_idx)
                probs = 1.0 / (distances + 1)
                probs = probs / probs.sum()
                indices = np.random.choice(N, num_sample_points, replace=False, p=probs)
            
            # Get window points
            window_points = points_normalized[indices]
            
            # Convert to tensor: [1, 3, N]
            window_tensor = torch.from_numpy(window_points).float().unsqueeze(0).to(device)
            window_tensor = window_tensor.transpose(1, 2)
            
            # Predict
            logits = model(window_tensor)  # [1, 2, N]
            probs = torch.exp(logits).squeeze(0).cpu().numpy()  # [2, N]
            
            # Accumulate votes
            valid_indices = indices[:min(len(indices), N)]
            all_votes[valid_indices] += probs[:, :len(valid_indices)].T
            vote_counts[valid_indices] += 1
    
    # Average votes
    vote_counts = np.maximum(vote_counts, 1)
    avg_probs = all_votes / vote_counts[:, np.newaxis]
    
    # Get predictions
    labels = np.argmax(avg_probs, axis=1)
    confidence = np.max(avg_probs, axis=1)
    
    return labels, confidence


def detect_model_architecture(state_dict_keys):
    """
    Detect model architecture from checkpoint keys.
    Returns: (architecture_name, module_path, class_name, kwargs)
    """
    keys_list = list(state_dict_keys)
    keys_str = ' '.join(keys_list)
    
    logger.info(f"Detecting architecture from {len(keys_list)} keys...")
    logger.info(f"Sample keys: {keys_list[:10]}")
    
    # Check for specific patterns in keys
    has_enc_dec = any('enc1.' in k or 'dec1.' in k or 'enc2.' in k or 'dec2.' in k for k in keys_list)
    has_down_up_numbered = any('down1.' in k or 'up1.' in k or 'down2.' in k or 'up2.' in k for k in keys_list)
    has_edge_aware = any('edge_aware' in k for k in keys_list)
    has_sa = any('sa1.' in k or 'sa2.' in k or 'sa3.' in k or 'sa4.' in k for k in keys_list)
    has_fp = any('fp1.' in k or 'fp2.' in k or 'fp3.' in k or 'fp4.' in k for k in keys_list)
    has_attn = any('attn1' in k or 'attn2' in k or 'attn3' in k or 'attn4' in k for k in keys_list)
    has_aux_head = any('aux_head' in k for k in keys_list)
    has_input_embed = any('input_embed' in k for k in keys_list)
    
    # Point Transformer detection: enc/dec blocks + down/up transitions + edge_aware + aux_head
    # Key patterns: enc1, enc2, enc3, enc4, dec1, dec2, dec3, dec4, down1, down2, down3, up1, up2, up3, up4
    if has_enc_dec and has_down_up_numbered and has_edge_aware and has_aux_head:
        logger.info("Detected: Point Transformer architecture")
        return ('PointTransformer', 'model', 'PointTransformerSeg', {'num_classes': 2, 'num_heads': 8})
    
    # KPConv detection: has input_embed + sa/fp layers BUT no attn layers
    # KPConv uses SetAbstraction (sa1-4) and FeaturePropagation (fp1-4) like PointNet++
    # but has input_embed layer and does NOT have self-attention layers
    if has_input_embed and has_sa and has_fp and not has_attn:
        logger.info("Detected: KPConv architecture")
        return ('KPConv', 'kpconv_model', 'KPConvSegmentation', {'num_classes': 2})
    
    # PointNet++ detection: sa + fp + attn layers (without edge_aware)
    # Key patterns: sa1, sa2, sa3, sa4, fp1, fp2, fp3, fp4, attn1, attn2... (NO edge_aware)
    if has_sa and has_fp and has_attn and not has_edge_aware:
        logger.info("Detected: PointNet++ architecture")
        return ('PointNet++', 'pointnet2', 'PointNet2SemSeg', {'num_classes': 2})
    
    # ImprovedHybridPointNet (pn2_sa1 pattern - 4-level hierarchy)
    if any('pn2_sa1' in k for k in keys_list):
        logger.info("Detected: ImprovedHybridPointNet architecture")
        return ('ImprovedHybridPointNet', 'improved_hybrid_model', 'ImprovedHybridPointNet', 
                {'num_classes': 2})
    
    # EnhancedHybridPointNet (mssa or multi_scale pattern)
    if any('mssa' in k or 'multi_scale' in k.lower() for k in keys_list):
        logger.info("Detected: EnhancedHybridPointNet architecture (v2/v3)")
        return ('EnhancedHybridPointNet', 'hybrid_model_v2', 'EnhancedHybridPointNet', 
                {'num_classes': 2, 'num_heads': 8})
    
    # Standard HybridPointNet: sa + fp + edge_aware (pt_block, edge_module patterns)
    if has_sa and has_fp and has_edge_aware:
        logger.info("Detected: HybridPointNet architecture")
        return ('HybridPointNet', 'hybrid_model', 'HybridPointNet', 
                {'num_classes': 2, 'num_heads': 4})
    
    # Default fallback - try PointNet++
    logger.warning("Could not detect architecture, defaulting to PointNet++")
    return ('PointNet++', 'pointnet2', 'PointNet2SemSeg', {'num_classes': 2})


def load_model_checkpoint(model_path, device):
    """Load model from checkpoint with architecture auto-detection"""
    global current_architecture
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        
        # Extract state dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("Found 'state_dict' in checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info("Found 'model' in checkpoint")
            else:
                # Check if it's a raw state dict (keys are layer names)
                first_key = list(checkpoint.keys())[0] if checkpoint else ""
                if '.' in first_key or first_key.startswith('sa') or first_key.startswith('fp'):
                    state_dict = checkpoint
                    logger.info("Checkpoint appears to be a raw state dict")
                else:
                    # Unknown format, try to use as-is
                    state_dict = checkpoint
                    logger.warning("Unknown checkpoint format, attempting to use as state dict")
        else:
            state_dict = checkpoint
            logger.info("Checkpoint is a direct state dict")
        
        # Detect architecture
        arch_name, module_name, class_name, kwargs = detect_model_architecture(state_dict.keys())
        current_architecture = arch_name
        
        # Dynamic import based on detected architecture
        model = None
        
        if arch_name == 'PointTransformer':
            try:
                from model import PointTransformerSeg
                model = PointTransformerSeg(**kwargs).to(device)
            except ImportError as e:
                logger.error(f"Failed to import Point Transformer: {e}")
                raise ImportError(f"Point Transformer model not found. Ensure 'Point Transformer/model.py' exists.")
        
        elif arch_name == 'KPConv':
            try:
                from kpconv_model import KPConvSegmentation
                model = KPConvSegmentation(**kwargs).to(device)
            except ImportError as e:
                logger.error(f"Failed to import KPConv: {e}")
                raise ImportError(f"KPConv model not found. Ensure 'KPConv/kpconv_model.py' exists.")
        
        elif arch_name == 'PointNet++':
            try:
                from pointnet2 import PointNet2SemSeg
                model = PointNet2SemSeg(**kwargs).to(device)
            except ImportError as e:
                logger.error(f"Failed to import PointNet++: {e}")
                raise ImportError(f"PointNet++ model not found. Ensure 'PointNet ++/pointnet2.py' exists.")
        
        elif arch_name == 'ImprovedHybridPointNet':
            try:
                from improved_hybrid_model import ImprovedHybridPointNet
                model = ImprovedHybridPointNet(**kwargs).to(device)
            except ImportError as e:
                logger.error(f"Failed to import ImprovedHybridPointNet: {e}")
                raise ImportError(f"ImprovedHybridPointNet not found.")
        
        elif arch_name == 'EnhancedHybridPointNet':
            try:
                from hybrid_model_v2 import EnhancedHybridPointNet
                model = EnhancedHybridPointNet(**kwargs).to(device)
            except ImportError:
                try:
                    from hybrid_model_v3 import EnhancedHybridPointNet
                    model = EnhancedHybridPointNet(**kwargs).to(device)
                except ImportError as e:
                    logger.error(f"Failed to import EnhancedHybridPointNet: {e}")
                    raise ImportError(f"EnhancedHybridPointNet not found in v2 or v3.")
        
        else:  # Default: HybridPointNet
            try:
                from hybrid_model import HybridPointNet
                model = HybridPointNet(**kwargs).to(device)
            except ImportError as e:
                logger.error(f"Failed to import HybridPointNet: {e}")
                raise ImportError(f"HybridPointNet not found.")
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Try to handle mismatched keys
            logger.warning(f"State dict mismatch: {e}")
            logger.info("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        logger.info(f"Model loaded successfully: {arch_name}")
        
        return model, arch_name
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to load model: {str(e)}")


def predict_point_cloud(model, points, device, num_sample_points=2048, use_pca_align=False):
    """
    Predict labels for entire point cloud.
    Uses KDTree interpolation to match standalone prediction scripts.
    This approach:
    1. Applies PCA alignment to handle different orientations (auto/True/False)
    2. Normalizes points using simple centroid + max distance method
    3. Samples points for model input (up to num_sample_points)
    4. Uses KDTree to interpolate predictions back to all original points
    
    Args:
        use_pca_align: 'auto' (default) - auto-detect based on orientation
                       True - always apply PCA alignment
                       False - never apply PCA alignment
    """
    from scipy.spatial import KDTree
    
    model.eval()
    num_points = len(points)
    
    # Determine if we need PCA alignment
    if use_pca_align == 'auto':
        apply_alignment = needs_orientation_fix(points)
    else:
        apply_alignment = use_pca_align
    
    # Apply PCA alignment to handle models in different planes
    if apply_alignment:
        logger.info("Applying PCA alignment for orientation normalization...")
        points_aligned, _ = pca_align(points.copy())
    else:
        points_aligned = points.copy()
    
    # Normalize points - simple normalization matching training
    points_normalized, centroid, max_dist = simple_normalize(points_aligned)
    
    # Sample points for model input
    if num_points > num_sample_points:
        choice = np.random.choice(num_points, num_sample_points, replace=False)
    else:
        choice = np.random.choice(num_points, num_sample_points, replace=True)
    
    points_input = points_normalized[choice]
    
    # Prepare tensor: [1, 3, N]
    points_tensor = torch.from_numpy(points_input).float().unsqueeze(0).to(device)
    points_tensor = points_tensor.transpose(1, 2)
    
    with torch.no_grad():
        outputs = model(points_tensor)  # [1, num_classes, N]
        
        # Handle log_softmax output (PointNet++) vs raw logits
        # Check if output is log probabilities (negative values typical)
        if outputs.min() < -1.0:  # Likely log_softmax
            probs = torch.exp(outputs)
        else:  # Raw logits
            probs = torch.softmax(outputs, dim=1)
        
        pred = outputs.max(dim=1)[1].cpu().numpy()[0]  # [N]
        confidence_sampled = probs.max(dim=1)[0].cpu().numpy()[0]  # [N]
    
    # Use KDTree to interpolate predictions back to all original points
    logger.info("Interpolating labels to original point cloud using KDTree...")
    tree = KDTree(points_input)
    _, indices = tree.query(points_normalized, k=1)
    
    # Map predictions to all points
    labels = pred[indices]
    confidence = confidence_sampled[indices]
    
    return labels, confidence


def create_plotly_data(points, labels, confidence, architecture=None):
    """Create Plotly-compatible JSON data for 3D visualization"""
    # Subsample for performance if too many points
    if len(points) > 50000:
        indices = np.random.choice(len(points), 50000, replace=False)
        points = points[indices]
        labels = labels[indices]
        confidence = confidence[indices]
    
    background_mask = labels == 0
    weld_mask = labels == 1
    
    traces = []
    
    if background_mask.any():
        bg_points = points[background_mask]
        traces.append({
            'type': 'scatter3d',
            'x': bg_points[:, 0].tolist(),
            'y': bg_points[:, 1].tolist(),
            'z': bg_points[:, 2].tolist(),
            'mode': 'markers',
            'name': 'Background',
            'marker': {
                'size': 2,
                'color': 'rgba(100, 149, 237, 0.4)',
                'opacity': 0.4
            },
            'hovertemplate': 'Background<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        })
    
    if weld_mask.any():
        weld_points = points[weld_mask]
        weld_conf = confidence[weld_mask]
        traces.append({
            'type': 'scatter3d',
            'x': weld_points[:, 0].tolist(),
            'y': weld_points[:, 1].tolist(),
            'z': weld_points[:, 2].tolist(),
            'mode': 'markers',
            'name': 'Weld Path',
            'marker': {
                'size': 4,
                'color': weld_conf.tolist(),
                'colorscale': [[0, 'orange'], [1, 'red']],
                'opacity': 0.9,
                'colorbar': {
                    'title': 'Confidence',
                    'thickness': 15
                }
            },
            'hovertemplate': 'Weld<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Confidence: %{marker.color:.2f}<extra></extra>'
        })
    
    layout = {
        'title': {
            'text': 'Weld Path Detection - 3D Visualization',
            'font': {'size': 20, 'color': '#fff'}
        },
        'scene': {
            'xaxis': {'title': 'X', 'gridcolor': '#444', 'color': '#fff'},
            'yaxis': {'title': 'Y', 'gridcolor': '#444', 'color': '#fff'},
            'zaxis': {'title': 'Z', 'gridcolor': '#444', 'color': '#fff'},
            'aspectmode': 'data',
            'bgcolor': '#1a1a2e'
        },
        'paper_bgcolor': '#16213e',
        'plot_bgcolor': '#1a1a2e',
        'showlegend': True,
        'legend': {
            'font': {'color': '#fff'},
            'bgcolor': 'rgba(0,0,0,0.3)'
        },
        'margin': {'l': 0, 'r': 0, 't': 50, 'b': 0}
    }
    
    # Calculate statistics
    stats = {
        'total_points': int(len(points)),
        'weld_points': int(weld_mask.sum()),
        'background_points': int(background_mask.sum()),
        'weld_percentage': float(weld_mask.sum() / len(points) * 100),
        'avg_confidence': float(confidence.mean()),
        'architecture': architecture or 'Unknown'
    }
    
    return {'data': traces, 'layout': layout, 'stats': stats}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload/cad', methods=['POST'])
def upload_cad():
    global current_cad_path
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_cad_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .ply or .stl files'}), 400
    
    try:
        # Validate file size
        file_size = validate_file_size(file, 'cad')
        
        # Save file
        filename = file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        current_cad_path = filepath
        
        # Verify file can be loaded
        try:
            points = load_mesh(filepath)
            point_count = len(points)
        except Exception as e:
            # Remove invalid file
            filepath.unlink(missing_ok=True)
            current_cad_path = None
            return jsonify({'error': f'Invalid CAD file: {str(e)}'}), 400
        
        logger.info(f"CAD file uploaded: {filename} ({point_count} points)")
        return jsonify({
            'success': True,
            'filename': filename,
            'point_count': point_count,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'message': f'Successfully uploaded {filename} ({point_count:,} points)'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error uploading CAD file: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/upload/model', methods=['POST'])
def upload_model():
    global current_model_path, current_model, current_architecture
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_model_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .pth files'}), 400
    
    try:
        # Validate file size
        file_size = validate_file_size(file, 'model')
        
        # Save file
        filename = file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        current_model_path = filepath
        current_model = None  # Reset loaded model
        current_architecture = None
        
        # Try to detect architecture (but don't fully load yet)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            arch_name, _, _, _ = detect_model_architecture(state_dict.keys())
        except Exception as e:
            logger.warning(f"Could not detect architecture during upload: {e}")
            arch_name = "Unknown"
        
        logger.info(f"Model file uploaded: {filename} (Architecture: {arch_name})")
        return jsonify({
            'success': True,
            'filename': filename,
            'architecture': arch_name,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'message': f'Successfully uploaded {filename} (Detected: {arch_name})'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error uploading model file: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/process', methods=['POST'])
def process():
    global current_model, current_architecture
    
    if current_cad_path is None:
        return jsonify({'error': 'Please upload a CAD model first'}), 400
    
    if current_model_path is None:
        return jsonify({'error': 'Please upload a DL model first'}), 400
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load mesh and get points
        points = load_mesh(current_cad_path)
        
        # Load model if not already loaded
        if current_model is None:
            current_model, current_architecture = load_model_checkpoint(current_model_path, device)
        
        # Run prediction - use architecture-specific prediction function
        logger.info(f"Running inference with {current_architecture} model...")
        if current_architecture == 'KPConv':
            labels, confidence = predict_point_cloud_kpconv(current_model, points, device)
        else:
            labels, confidence = predict_point_cloud(current_model, points, device)
        
        # Create Plotly data
        plotly_data = create_plotly_data(points, labels, confidence, current_architecture)
        
        logger.info(f"Processing complete. Found {plotly_data['stats']['weld_points']} weld points")
        
        return jsonify({
            'success': True,
            'plotly': plotly_data
        })
        
    except Exception as e:
        logger.error(f"Error processing: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        'cad_uploaded': current_cad_path is not None,
        'cad_filename': current_cad_path.name if current_cad_path else None,
        'model_uploaded': current_model_path is not None,
        'model_filename': current_model_path.name if current_model_path else None,
        'model_loaded': current_model is not None,
        'architecture': current_architecture
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Reset all uploaded files and loaded models"""
    global current_cad_path, current_model_path, current_model, current_architecture
    
    current_cad_path = None
    current_model_path = None
    current_model = None
    current_architecture = None
    
    return jsonify({'success': True, 'message': 'Reset complete'})


if __name__ == '__main__':
    logger.info("Starting Weld Path Detection Web UI...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Supported architectures: PointNet++, HybridPointNet, ImprovedHybridPointNet, EnhancedHybridPointNet, PointTransformer, KPConv")
    app.run(debug=True, host='0.0.0.0', port=5000)
