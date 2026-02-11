"""
Enhanced Point Transformer for Point Cloud Segmentation
Based on "Point Transformer" (Zhao et al., 2021)

Advanced features:
- Vector self-attention with position encoding
- Multi-head attention mechanism (8 heads)
- 4-level encoder-decoder hierarchy
- Edge-aware feature refinement
- Multi-scale supervision with auxiliary heads
- Channel attention and residual connections
- Optimized for weld detection in 3D point clouds
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

def knn(x, k):
    """
    Find k-nearest neighbors
    Args:
        x: point coordinates [B, N, 3]
        k: number of neighbors
    Returns:
        idx: indices of k-nearest neighbors [B, N, k]
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def index_points(points, idx):
    """
    Index points using indices
    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]
    Returns:
        indexed points
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    Args:
        xyz: [B, N, 3]
        npoint: number of points to sample
    Returns:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points
    Args:
        src: [B, N, C]
        dst: [B, M, C]
    Returns:
        dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(-2)
    return dist


# ============================================================================
# Enhanced Point Transformer Block
# ============================================================================

class PointTransformerBlock(nn.Module):
    """Enhanced Point Transformer Layer with vector self-attention and 8 attention heads"""
    
    def __init__(self, in_channels, out_channels, num_heads=8, k=32):
        super(PointTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.head_dim = out_channels // num_heads
        
        # Linear transformations for Q, K, V
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_k = nn.Linear(in_channels, out_channels)
        self.fc_v = nn.Linear(in_channels, out_channels)
        
        # Position encoding with deeper network
        self.fc_delta = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
        
        # Attention gamma for learnable attention computation
        self.fc_gamma = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
        
        # Output projection
        self.fc_out = nn.Linear(out_channels, out_channels)
        
        # Channel attention module
        self.channel_attn = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.norm3 = nn.LayerNorm(out_channels)
        
        # Skip connection projection if dimensions don't match
        self.skip_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, N, 3] point coordinates
            features: [B, N, C] point features
        Returns:
            new_features: [B, N, C'] transformed features
        """
        B, N, C = features.shape
        
        # Normalize input
        features_norm = self.norm1(features)
        
        # Find k-nearest neighbors
        idx = knn(xyz, self.k)  # [B, N, k]
        
        # Get neighbor features and positions
        neighbor_xyz = index_points(xyz, idx)
        neighbor_features = index_points(features_norm, idx)
        
        # Position encoding
        pos_diff = xyz.unsqueeze(2) - neighbor_xyz  # [B, N, k, 3]
        pos_encoding = self.fc_delta(pos_diff)  # [B, N, k, C']
        
        # Query, Key, Value
        q = self.fc_q(features_norm).unsqueeze(2)  # [B, N, 1, C']
        k = self.fc_k(neighbor_features)  # [B, N, k, C']
        v = self.fc_v(neighbor_features)  # [B, N, k, C']
        
        # Multi-head attention
        q = q.view(B, N, 1, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        pos_encoding = pos_encoding.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention scores: γ(φ(x_i) - ψ(x_j) + δ(p_i - p_j))
        attn_input = q - k + pos_encoding
        attn_input = attn_input.permute(0, 2, 3, 1, 4).reshape(B, N, self.k, -1)
        attn_scores = self.fc_gamma(attn_input)
        attn_scores = F.softmax(attn_scores / np.sqrt(self.head_dim), dim=2)
        
        # Weighted sum: Σ softmax(...) ⊙ (α(x_j) + δ(p_i - p_j))
        attn_scores = attn_scores.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        weighted_features = attn_scores * (v + pos_encoding)
        weighted_features = weighted_features.sum(dim=3)  # [B, num_heads, N, head_dim]
        weighted_features = weighted_features.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        # Output projection
        output = self.fc_out(weighted_features)
        output = self.norm2(output)
        
        # Channel attention
        channel_weights = self.channel_attn(output.mean(dim=1))  # Global pooling [B, C]
        output = output * channel_weights.unsqueeze(1)
        output = self.norm3(output)
        
        # Residual connection
        if self.skip_proj is not None:
            features = self.skip_proj(features)
        
        return output + features


# ============================================================================
# Transition Layers (Downsampling and Upsampling)
# ============================================================================

class TransitionDown(nn.Module):
    """Downsampling transition with FPS and feature aggregation"""
    
    def __init__(self, in_channels, out_channels, ratio=0.25, k=32):
        super(TransitionDown, self).__init__()
        self.ratio = ratio
        self.k = k
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C]
        Returns:
            new_xyz: [B, N', 3]
            new_features: [B, N', C']
        """
        B, N, _ = xyz.shape
        npoint = max(int(N * self.ratio), 1)
        
        # Farthest point sampling
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        
        # K-NN grouping
        idx = knn(xyz, min(self.k, N))
        neighbor_xyz = index_points(xyz, idx)
        neighbor_features = index_points(features, idx)
        
        # Sample the grouped features at FPS points
        fps_neighbor_idx = knn(new_xyz, min(self.k, N))
        fps_neighbor_xyz = index_points(xyz, fps_neighbor_idx)
        fps_neighbor_features = index_points(features, fps_neighbor_idx)
        
        # Relative positions
        pos_diff = fps_neighbor_xyz - new_xyz.unsqueeze(2)
        
        # Concatenate features and positions
        combined = torch.cat([fps_neighbor_features, pos_diff], dim=-1)
        
        # MLP and max pooling
        combined = self.mlp(combined)
        new_features = torch.max(combined, dim=2)[0]
        
        return new_xyz, new_features


class TransitionUp(nn.Module):
    """Upsampling transition with interpolation and skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(TransitionUp, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, xyz1, xyz2, features1, features2):
        """
        Args:
            xyz1: [B, N1, 3] - upsampled positions
            xyz2: [B, N2, 3] - current positions
            features1: [B, N1, C1] - skip connection features
            features2: [B, N2, C2] - current features
        Returns:
            new_features: [B, N1, C']
        """
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        
        if N2 == 1:
            # Global features, just repeat
            interpolated = features2.repeat(1, N1, 1)
        else:
            # 3-NN interpolation
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated = torch.sum(index_points(features2, idx) * weight.unsqueeze(-1), dim=2)
        
        # Concatenate with skip connection
        if features1 is not None:
            new_features = torch.cat([features1, interpolated], dim=-1)
        else:
            new_features = interpolated
        
        # MLP
        new_features = self.mlp(new_features)
        
        return new_features


# ============================================================================
# Edge-Aware Module
# ============================================================================

class EdgeAwareModule(nn.Module):
    """Detect and enhance features near weld boundaries"""
    
    def __init__(self, channels, k=16):
        super(EdgeAwareModule, self).__init__()
        self.k = k
        
        # Geometric feature extraction
        self.geo_mlp = nn.Sequential(
            nn.Linear(10, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )
        
        # Edge probability predictor
        self.edge_pred = nn.Sequential(
            nn.Linear(channels + 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Edge-conditioned feature refinement
        self.edge_refine = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True)
        )
    
    def compute_geometric_features(self, xyz):
        """Compute local geometric features (curvature indicators, normal variations)"""
        B, N, _ = xyz.shape
        
        # Find k nearest neighbors
        idx = knn(xyz, min(self.k, N))
        neighbors = index_points(xyz, idx)  # [B, N, k, 3]
        
        # Local statistics
        centroid = neighbors.mean(dim=2)  # [B, N, 3]
        centered = neighbors - centroid.unsqueeze(2)  # [B, N, k, 3]
        
        # Covariance-based features
        cov = torch.matmul(centered.transpose(2, 3), centered) / self.k
        
        # Curvature indicators
        trace = cov[:, :, 0, 0] + cov[:, :, 1, 1] + cov[:, :, 2, 2]
        det = (cov[:, :, 0, 0] * cov[:, :, 1, 1] * cov[:, :, 2, 2]).clamp(min=1e-8)
        
        # Normal variation
        std_x = centered[:, :, :, 0].std(dim=2)
        std_y = centered[:, :, :, 1].std(dim=2)
        std_z = centered[:, :, :, 2].std(dim=2)
        
        # Distance variation
        dist_to_neighbors = torch.norm(centered, dim=3)
        mean_dist = dist_to_neighbors.mean(dim=2)
        std_dist = dist_to_neighbors.std(dim=2)
        
        # Concatenate geometric features
        geo_feat = torch.stack([
            trace, det, std_x, std_y, std_z, mean_dist, std_dist,
            std_x / (trace + 1e-6), std_y / (trace + 1e-6), std_z / (trace + 1e-6)
        ], dim=-1)  # [B, N, 10]
        
        return geo_feat
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C]
        Returns:
            refined_features: [B, N, C]
            edge_prob: [B, N, 1]
        """
        # Compute geometric features
        geo_feat = self.compute_geometric_features(xyz)  # [B, N, 10]
        geo_feat = self.geo_mlp(geo_feat)  # [B, N, 16]
        
        # Predict edge probability
        combined = torch.cat([features, geo_feat], dim=-1)
        edge_prob = self.edge_pred(combined)  # [B, N, 1]
        
        # Refine features based on edge probability
        refined = self.edge_refine(features)
        refined = features + refined * edge_prob
        
        return refined, edge_prob


# ============================================================================
# Auxiliary Head
# ============================================================================

class AuxiliaryHead(nn.Module):
    """Auxiliary classifier for intermediate supervision"""
    
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LayerNorm(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_channels // 2, num_classes)
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, N, C]
        Returns:
            logits: [B, N, num_classes]
        """
        return self.classifier(features)


# ============================================================================
# Main Point Transformer Model
# ============================================================================

class PointTransformerSeg(nn.Module):
    """
    Enhanced Point Transformer for Semantic Segmentation
    
    Features:
    - 4-level encoder-decoder hierarchy
    - Vector self-attention with 8 attention heads
    - Edge-aware feature refinement
    - Multi-scale auxiliary supervision
    - Channel attention and residual connections
    """
    
    def __init__(self, num_classes=2, num_heads=8):
        super(PointTransformerSeg, self).__init__()
        
        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        
        # ========== Encoder ==========
        # Level 1: N → N (1024 if downsampled)
        self.enc1 = PointTransformerBlock(128, 128, num_heads, k=32)
        self.down1 = TransitionDown(128, 256, ratio=0.5, k=32)
        
        # Level 2: 512 → 512 → 256
        self.enc2 = PointTransformerBlock(256, 256, num_heads, k=32)
        self.down2 = TransitionDown(256, 512, ratio=0.5, k=32)
        
        # Level 3: 256 → 256 → 64
        self.enc3 = PointTransformerBlock(512, 512, num_heads, k=16)
        self.down3 = TransitionDown(512, 1024, ratio=0.25, k=16)
        
        # Level 4: 64 → 64 (bottleneck)
        self.enc4 = PointTransformerBlock(1024, 1024, num_heads, k=16)
        
        # ========== Decoder ==========
        self.up4 = TransitionUp(1024, 512, 512)
        self.dec4 = PointTransformerBlock(512, 512, num_heads, k=16)
        
        self.up3 = TransitionUp(512, 256, 256)
        self.dec3 = PointTransformerBlock(256, 256, num_heads, k=32)
        
        self.up2 = TransitionUp(256, 128, 128)
        self.dec2 = PointTransformerBlock(128, 128, num_heads, k=32)
        
        self.up1 = TransitionUp(128, 128, 128)
        self.dec1 = PointTransformerBlock(128, 128, num_heads, k=32)
        
        # ========== Edge-Aware Modules ==========
        self.edge_aware3 = EdgeAwareModule(256, k=16)
        self.edge_aware2 = EdgeAwareModule(128, k=16)
        self.edge_aware1 = EdgeAwareModule(128, k=16)
        
        # ========== Auxiliary Heads ==========
        self.aux_head3 = AuxiliaryHead(256, num_classes)
        self.aux_head2 = AuxiliaryHead(128, num_classes)
        
        # ========== Final Classifier ==========
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, xyz):
        """
        Args:
            xyz: [B, 3, N] point coordinates
        Returns:
            If training: (main_logits, aux_logits_list, edge_probs_list)
            If inference: log_probs [B, num_classes, N]
        """
        B, _, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        
        # Input embedding
        l0_xyz = xyz
        l0_feat = self.embed(xyz)  # [B, N, 128]
        
        # ========== Encoder ==========
        # Level 1
        l1_feat = self.enc1(l0_xyz, l0_feat)  # [B, N, 128]
        l1_xyz, l1_feat_down = self.down1(l0_xyz, l1_feat)  # [B, N/2, 256]
        
        # Level 2
        l2_feat = self.enc2(l1_xyz, l1_feat_down)  # [B, N/2, 256]
        l2_xyz, l2_feat_down = self.down2(l1_xyz, l2_feat)  # [B, N/4, 512]
        
        # Level 3
        l3_feat = self.enc3(l2_xyz, l2_feat_down)  # [B, N/4, 512]
        l3_xyz, l3_feat_down = self.down3(l2_xyz, l3_feat)  # [B, N/16, 1024]
        
        # Level 4 (bottleneck)
        l4_feat = self.enc4(l3_xyz, l3_feat_down)  # [B, N/16, 1024]
        
        # ========== Decoder ==========
        # Upsample from level 4 to 3
        d3_feat = self.up4(l2_xyz, l3_xyz, l3_feat, l4_feat)  # [B, N/4, 512]
        d3_feat = self.dec4(l2_xyz, d3_feat)  # [B, N/4, 512]
        
        # Upsample from level 3 to 2
        d2_feat = self.up3(l1_xyz, l2_xyz, l2_feat, d3_feat)  # [B, N/2, 256]
        d2_feat = self.dec3(l1_xyz, d2_feat)  # [B, N/2, 256]
        d2_feat, edge_prob3 = self.edge_aware3(l1_xyz, d2_feat)
        
        # Upsample from level 2 to 1
        d1_feat = self.up2(l0_xyz, l1_xyz, l1_feat, d2_feat)  # [B, N, 128]
        d1_feat = self.dec2(l0_xyz, d1_feat)  # [B, N, 128]
        d1_feat, edge_prob2 = self.edge_aware2(l0_xyz, d1_feat)
        
        # Final upsampling
        d0_feat = self.up1(l0_xyz, l0_xyz, l0_feat, d1_feat)  # [B, N, 128]
        d0_feat = self.dec1(l0_xyz, d0_feat)  # [B, N, 128]
        d0_feat, edge_prob1 = self.edge_aware1(l0_xyz, d0_feat)
        
        # ========== Auxiliary Predictions ==========
        aux_logits3 = self.aux_head3(d2_feat)  # [B, N/2, num_classes]
        aux_logits2 = self.aux_head2(d1_feat)  # [B, N, num_classes]
        
        # ========== Final Classification ==========
        main_logits = self.classifier(d0_feat)  # [B, N, num_classes]
        
        # Transpose to [B, num_classes, N] for consistency with loss functions
        main_logits = main_logits.permute(0, 2, 1)
        aux_logits2 = aux_logits2.permute(0, 2, 1)
        aux_logits3 = aux_logits3.permute(0, 2, 1)
        edge_prob1 = edge_prob1.permute(0, 2, 1)
        edge_prob2 = edge_prob2.permute(0, 2, 1)
        edge_prob3 = edge_prob3.permute(0, 2, 1)
        
        # Return different outputs for training vs inference
        if self.training:
            return main_logits, [aux_logits2, aux_logits3], [edge_prob1, edge_prob2, edge_prob3]
        else:
            log_probs = F.log_softmax(main_logits, dim=1)
            return log_probs


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, N] logits
            targets: [B, N] labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, N] logits
            targets: [B, N] labels
        """
        num_classes = inputs.size(1)
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()
        
        # Compute Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss with auxiliary supervision"""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0,
                 aux_weight=0.3, boundary_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, main_logits, targets, aux_logits_list=None, edge_probs_list=None):
        """
        Args:
            main_logits: [B, C, N]
            targets: [B, N]
            aux_logits_list: list of auxiliary predictions
            edge_probs_list: list of edge probabilities
        """
        # Main loss
        focal = self.focal_loss(main_logits, targets)
        dice = self.dice_loss(main_logits, targets)
        main_loss = self.focal_weight * focal + self.dice_weight * dice
        
        total_loss = main_loss
        
        # Auxiliary loss
        if aux_logits_list is not None:
            aux_loss = 0
            aux_weights = [0.5, 0.3]  # Decreasing weights for deeper layers
            
            for aux_logits, weight in zip(aux_logits_list, aux_weights):
                B, _, N_aux = aux_logits.shape
                if N_aux != targets.shape[1]:
                    # Downsample targets
                    ratio = targets.shape[1] // N_aux
                    targets_down = targets[:, ::ratio][:, :N_aux]
                else:
                    targets_down = targets
                
                aux_focal = self.focal_loss(aux_logits, targets_down)
                aux_loss += weight * aux_focal
            
            total_loss = total_loss + self.aux_weight * aux_loss
        
        # Boundary loss - regularization on edge probabilities
        if edge_probs_list is not None and self.boundary_weight > 0:
            boundary_loss = 0
            for edge_prob in edge_probs_list:
                # Encourage moderate edge detection
                boundary_loss += (edge_prob.mean() - 0.1).abs()
            
            total_loss = total_loss + self.boundary_weight * boundary_loss
        
        return total_loss



