"""
KPConv Segmentation Model (Optimized)
Fast encoder-decoder architecture for weld detection
Optimized with torch.cdist and reduced complexity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

# Try to import optimized libraries for faster operations
_USE_TORCH_CLUSTER = False
try:
    from torch_cluster import fps as torch_cluster_fps
    from torch_cluster import knn as torch_cluster_knn
    _USE_TORCH_CLUSTER = True
    print("[KPConv] Using torch-cluster for accelerated operations")
except ImportError:
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared distance using torch.cdist (faster on GPU)"""
    return torch.cdist(src, dst, p=2.0).pow(2)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find k-nearest neighbors"""
    sq_dist = square_distance(x, x)
    _, indices = torch.topk(sq_dist, k=k, dim=-1, largest=False)
    return indices


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling (optimized)"""
    device = xyz.device
    B, N, C = xyz.shape
    
    if _USE_TORCH_CLUSTER:
        # Use torch-cluster's CUDA-optimized FPS
        batch = torch.arange(B, device=device).repeat_interleave(N)
        xyz_flat = xyz.reshape(-1, C)
        ratio = npoint / N
        idx = torch_cluster_fps(xyz_flat, batch, ratio=ratio)
        centroids = (idx % N).reshape(B, -1)[:, :npoint]
        return centroids
    
    # Optimized version with torch.minimum (vectorized)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz - centroid).pow(2), dim=-1)  # [B, N]
        distance = torch.minimum(distance, dist)  # Vectorized min
        farthest = distance.argmax(dim=-1)
    
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points using indices"""
    device = points.device
    B = points.shape[0]
    
    if len(idx.shape) == 2:
        # [B, S] -> [B, S, C]
        batch_idx = torch.arange(B, device=device).view(-1, 1).expand_as(idx)
        return points[batch_idx, idx]
    else:
        # [B, S, K] -> [B, S, K, C]
        B, S, K = idx.shape
        batch_idx = torch.arange(B, device=device).view(-1, 1, 1).expand(B, S, K)
        return points[batch_idx, idx]


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        B, C, N = probs.shape
        targets_one_hot = F.one_hot(targets, C).permute(0, 2, 1).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        return (self.alpha * focal_weight * ce_loss).mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        B, C, N = probs.shape
        targets_one_hot = F.one_hot(targets, C).permute(0, 2, 1).float()
        intersection = (probs * targets_one_hot).sum(dim=(0, 2))
        cardinality = (probs + targets_one_hot).sum(dim=(0, 2))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal_loss(inputs, targets) + \
               self.dice_weight * self.dice_loss(inputs, targets)


# ============================================================================
# KPConv Core Layers
# ============================================================================

class SharedMLP(nn.Module):
    """Shared MLP applied to each point"""
    
    def __init__(self, channels: List[int], bn: bool = True):
        super(SharedMLP, self).__init__()
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, N]"""
        return self.mlp(x)


class KPConvSimple(nn.Module):
    """Simplified KPConv layer using local aggregation"""
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super(KPConvSimple, self).__init__()
        self.k = k
        
        # Edge feature MLP
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, 1)
        )
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, 3, N]
            features: [B, C, N]
        Returns:
            [B, C', N]
        """
        B, C, N = features.shape
        
        # Find k-nearest neighbors
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        idx = knn(xyz_t, self.k)  # [B, N, k]
        
        # Get neighbor features and positions
        features_t = features.transpose(1, 2)  # [B, N, C]
        neighbor_features = index_points(features_t, idx)  # [B, N, k, C]
        neighbor_xyz = index_points(xyz_t, idx)  # [B, N, k, 3]
        
        # Compute relative positions
        center_xyz = xyz_t.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, 3]
        relative_pos = neighbor_xyz - center_xyz  # [B, N, k, 3]
        
        # Expand center features
        center_features = features_t.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C]
        
        # Concatenate edge features
        edge_features = torch.cat([center_features, neighbor_features, relative_pos], dim=-1)
        edge_features = edge_features.permute(0, 3, 1, 2)  # [B, 2C+3, N, k]
        
        # Apply edge MLP
        edge_features = self.edge_mlp(edge_features)  # [B, C', N, k]
        
        # Attention weights
        attn_weights = self.attention(edge_features)  # [B, 1, N, k]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Weighted aggregation
        output = (edge_features * attn_weights).sum(dim=-1)  # [B, C', N]
        
        return output


class SetAbstraction(nn.Module):
    """Set Abstraction layer with downsampling"""
    
    def __init__(self, npoint: int, in_channels: int, out_channels: int, k: int = 16):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.k = k
        
        self.conv = KPConvSimple(in_channels, out_channels, k)
        self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, 3, N]
            features: [B, C, N]
        Returns:
            new_xyz: [B, 3, npoint]
            new_features: [B, C', npoint]
        """
        B, _, N = xyz.shape
        
        # FPS to select points
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        fps_idx = farthest_point_sample(xyz_t, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz_t, fps_idx).transpose(1, 2)  # [B, 3, npoint]
        
        # Apply convolution first
        conv_features = self.conv(xyz, features)  # [B, C', N]
        
        # Select features at FPS points
        conv_features_t = conv_features.transpose(1, 2)  # [B, N, C']
        new_features = index_points(conv_features_t, fps_idx).transpose(1, 2)  # [B, C', npoint]
        
        new_features = self.norm(new_features)
        
        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    """Feature Propagation layer with upsampling"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(FeaturePropagation, self).__init__()
        
        total_in = in_channels + skip_channels
        self.mlp = SharedMLP([total_in, out_channels, out_channels])
    
    def forward(self,
                xyz1: torch.Tensor,
                xyz2: torch.Tensor,
                features1: torch.Tensor,
                features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz1: [B, 3, N1] - target positions (more points)
            xyz2: [B, 3, N2] - source positions (fewer points)
            features1: [B, C1, N1] - skip features at target
            features2: [B, C2, N2] - features to propagate
        Returns:
            new_features: [B, C', N1]
        """
        B, _, N1 = xyz1.shape
        N2 = xyz2.shape[2]
        
        xyz1_t = xyz1.transpose(1, 2)  # [B, N1, 3]
        xyz2_t = xyz2.transpose(1, 2)  # [B, N2, 3]
        
        # Find 3 nearest neighbors from xyz2 for each point in xyz1
        sq_dist = square_distance(xyz1_t, xyz2_t)  # [B, N1, N2]
        dist, idx = torch.topk(sq_dist, k=min(3, N2), dim=-1, largest=False)  # [B, N1, 3]
        dist = torch.clamp(dist, min=1e-10)
        
        # Inverse distance weighting
        weight = 1.0 / dist
        weight = weight / weight.sum(dim=-1, keepdim=True)  # [B, N1, 3]
        
        # Interpolate features
        features2_t = features2.transpose(1, 2)  # [B, N2, C2]
        neighbor_features = index_points(features2_t, idx)  # [B, N1, 3, C2]
        interpolated = (neighbor_features * weight.unsqueeze(-1)).sum(dim=2)  # [B, N1, C2]
        interpolated = interpolated.transpose(1, 2)  # [B, C2, N1]
        
        # Concatenate with skip features
        if features1 is not None:
            concat_features = torch.cat([interpolated, features1], dim=1)
        else:
            concat_features = interpolated
        
        # Apply MLP
        new_features = self.mlp(concat_features)
        
        return new_features


# ============================================================================
# Complete Model
# ============================================================================

class KPConvSegmentation(nn.Module):
    """
    Complete KPConv Segmentation Network (Optimized)
    Fast encoder-decoder with reduced complexity for faster training
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 3,
                 init_features: int = 32,  # Reduced from 64
                 k: int = 12):  # Reduced from 16
        super(KPConvSegmentation, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial embedding (lighter)
        self.input_embed = SharedMLP([in_channels, init_features])
        
        # Encoder (reduced points and channels for speed)
        self.sa1 = SetAbstraction(256, init_features, 64, k)   # 2048->256 pts
        self.sa2 = SetAbstraction(128, 64, 128, k)             # 256->128 pts
        self.sa3 = SetAbstraction(64, 128, 256, k)             # 128->64 pts
        self.sa4 = SetAbstraction(32, 256, 512, k)             # 64->32 pts
        
        # Decoder (matching reduced channels)
        self.fp4 = FeaturePropagation(512, 256, 256)
        self.fp3 = FeaturePropagation(256, 128, 128)
        self.fp2 = FeaturePropagation(128, 64, 64)
        self.fp1 = FeaturePropagation(64, init_features, 64)
        
        # Classification head (lighter)
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, num_classes, 1)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, 3, N] or [B, N, 3] input coordinates
        Returns:
            logits: [B, num_classes, N] log probabilities
        """
        # Handle input format
        if xyz.shape[1] != 3:
            xyz = xyz.transpose(1, 2)  # [B, 3, N]
        
        B, _, N = xyz.shape
        
        # Initial embedding
        l0_features = self.input_embed(xyz)  # [B, 32, N]
        l0_xyz = xyz
        
        # Encoder (optimized point counts)
        l1_xyz, l1_features = self.sa1(l0_xyz, l0_features)  # [B, 3, 256], [B, 64, 256]
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)  # [B, 3, 128], [B, 128, 128]
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)  # [B, 3, 64], [B, 256, 64]
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)  # [B, 3, 32], [B, 512, 32]
        
        # Decoder
        l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)  # [B, 256, 64]
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)  # [B, 128, 128]
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)  # [B, 64, 256]
        l0_features = self.fp1(l0_xyz, l1_xyz, l0_features, l1_features)  # [B, 64, N]
        
        # Classification
        logits = self.classifier(l0_features)  # [B, num_classes, N]
        
        return F.log_softmax(logits, dim=1)
    
    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                 class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate NLLLoss"""
        if class_weights is not None:
            criterion = nn.NLLLoss(weight=class_weights)
        else:
            criterion = nn.NLLLoss()
        return criterion(logits, labels)


def create_kpconv_model(config: dict = None) -> KPConvSegmentation:
    """Create model from config"""
    if config is None:
        config = {}
    
    return KPConvSegmentation(
        num_classes=config.get('num_classes', 2),
        in_channels=config.get('in_channels', 3),
        init_features=config.get('init_features', 64),
        k=config.get('k_neighbors', 16)
    )



