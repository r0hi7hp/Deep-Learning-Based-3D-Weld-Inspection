"""
Enhanced Hybrid PointNet++ and Point Transformer Architecture V2
Combines the strengths of:
- PointNet++: Multi-scale hierarchical feature extraction
- Point Transformer: Attention-based contextual features
- Cross-Attention Fusion: Advanced bidirectional feature fusion
- Edge-Aware Modules: Better boundary detection
- Channel Attention: Feature recalibration

This improved architecture is designed for robust weld detection in 3D point clouds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

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
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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
        npoint: number of samples
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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Query ball point grouping
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: [B, N, 3]
        new_xyz: [B, S, 3]
    Returns:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn(x, k):
    """
    Find k-nearest neighbors
    Args:
        x: [B, N, 3]
        k: number of neighbors
    Returns:
        idx: [B, N, k]
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


# ============================================================================
# Channel Attention Module (SE Block)
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention for [B, C, N] format"""
    
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, N] - channel-first format
        Returns:
            attended: [B, C, N]
        """
        # Global average pooling over spatial dimension N
        avg = x.mean(dim=2)  # [B, C]
        attn = self.fc(avg).unsqueeze(2)  # [B, C, 1]
        return x * attn



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
            nn.Linear(10, channels // 2),  # 3 (normal) + 3 (curvature direction) + 3 (relative pos) + 1 (dist)
            nn.LayerNorm(channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels // 2)
        )
        
        # Feature difference encoder
        self.diff_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels // 2)
        )
        
        # Edge probability predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement (input is features + edge_feat*edge_prob = C + C = 2C)
        self.refine = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def compute_geometric_features(self, xyz):
        """Compute local geometric features"""
        B, N, _ = xyz.shape
        
        # Get k-nearest neighbors
        idx = knn(xyz, self.k)  # [B, N, k]
        neighbors = index_points(xyz, idx)  # [B, N, k, 3]
        
        # Relative positions
        rel_pos = neighbors - xyz.unsqueeze(2)  # [B, N, k, 3]
        
        # Compute covariance matrix for each point
        cov = torch.matmul(rel_pos.transpose(-2, -1), rel_pos) / self.k  # [B, N, 3, 3]
        
        # Eigendecomposition for normals and curvature
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            normal = eigenvectors[..., 0]  # Smallest eigenvalue direction
            curvature_dir = eigenvectors[..., 2]  # Largest eigenvalue direction
        except:
            normal = torch.zeros(B, N, 3, device=xyz.device)
            normal[..., 2] = 1.0
            curvature_dir = torch.zeros(B, N, 3, device=xyz.device)
            curvature_dir[..., 0] = 1.0
        
        # Mean relative position and distance
        mean_rel = rel_pos.mean(dim=2)  # [B, N, 3]
        mean_dist = torch.norm(rel_pos, dim=-1).mean(dim=-1, keepdim=True)  # [B, N, 1]
        
        # Concatenate geometric features
        geo_feat = torch.cat([normal, curvature_dir, mean_rel, mean_dist], dim=-1)  # [B, N, 10]
        
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
        B, N, C = features.shape
        
        # Compute geometric features
        geo_feat = self.compute_geometric_features(xyz)
        geo_encoded = self.geo_mlp(geo_feat)  # [B, N, C//2]
        
        # Compute feature differences with neighbors
        idx = knn(xyz, self.k)
        neighbor_feats = index_points(features, idx)  # [B, N, k, C]
        feat_diff = (features.unsqueeze(2) - neighbor_feats).abs().mean(dim=2)  # [B, N, C]
        diff_encoded = self.diff_mlp(feat_diff)  # [B, N, C//2]
        
        # Combined edge features
        edge_feat = torch.cat([geo_encoded, diff_encoded], dim=-1)  # [B, N, C]
        
        # Predict edge probability
        edge_prob = self.edge_predictor(edge_feat)  # [B, N, 1]
        
        # Refine features - emphasize edge regions
        enhanced = torch.cat([features, edge_feat * edge_prob], dim=-1)
        refined = self.refine(enhanced)
        
        # Residual connection
        refined = refined + features
        
        return refined, edge_prob


# ============================================================================
# PointNet++ Components
# ============================================================================

class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer with attention enhancement"""
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.GroupNorm(8, out_channel))
            last_channel = out_channel
        
        # Lightweight attention for feature enhancement
        self.attention = nn.Sequential(
            nn.Conv2d(last_channel, last_channel // 4, 1),
            nn.GroupNorm(4, last_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_channel // 4, last_channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, C, N]
            points: [B, D, N]
        Returns:
            new_xyz: [B, C, S]
            new_points: [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
        
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            if points is not None:
                new_points = torch.cat([xyz, points], dim=-1)
            else:
                new_points = xyz
            new_points = new_points.permute(0, 2, 1).unsqueeze(-1)  # [B, C+D, 1, 1]
        else:
            # Farthest point sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            # Ball query grouping
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm
            
            new_points = new_points.permute(0, 3, 1, 2)  # [B, C+D, npoint, nsample]
        
        # MLP processing
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Apply attention
        attn_weights = self.attention(new_points)
        new_points = new_points * attn_weights
        
        # Max pooling
        new_points = torch.max(new_points, -1)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation Layer with residual connection"""
    
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.GroupNorm(8, out_channel))
            last_channel = out_channel
        
        # Channel attention for refinement
        self.channel_attn = ChannelAttention(last_channel)
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: [B, C, N]
            xyz2: [B, C, S]
            points1: [B, D, N]
            points2: [B, D, S]
        Returns:
            new_points: [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(index_points(points2.permute(0, 2, 1), idx) * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.permute(0, 2, 1)
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Apply channel attention
        new_points = self.channel_attn(new_points)
        
        return new_points


# ============================================================================
# Enhanced Point Transformer Block
# ============================================================================

class PointTransformerBlock(nn.Module):
    """Enhanced Point Transformer Layer with vector self-attention and 8 heads"""
    
    def __init__(self, in_channels, out_channels, num_heads=8, k=16):
        super(PointTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.head_dim = out_channels // num_heads
        
        # Linear transformations
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_k = nn.Linear(in_channels, out_channels)
        self.fc_v = nn.Linear(in_channels, out_channels)
        
        # Position encoding with more capacity
        self.fc_delta = nn.Sequential(
            nn.Linear(3, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, out_channels)
        )
        
        # Attention gamma (relation function)
        self.fc_gamma = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
        
        # Output projection with gating
        self.fc_out = nn.Linear(out_channels, out_channels)
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
        
        # Normalization and residual
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # FFN for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels),
            nn.Dropout(0.1)
        )
        self.norm3 = nn.LayerNorm(out_channels)
        
        # Skip connection projection if dimensions change
        self.skip_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C]
        Returns:
            new_features: [B, N, C']
        """
        B, N, C = features.shape
        
        # Pre-norm
        features_norm = self.norm1(features)
        
        # Find k-nearest neighbors
        idx = knn(xyz, self.k)  # [B, N, k]
        
        # Get neighbor features and positions
        neighbor_xyz = index_points(xyz, idx)  # [B, N, k, 3]
        neighbor_features = index_points(features_norm, idx)  # [B, N, k, C]
        
        # Position encoding
        pos_diff = xyz.unsqueeze(2) - neighbor_xyz  # [B, N, k, 3]
        pos_encoding = self.fc_delta(pos_diff)  # [B, N, k, out_channels]
        
        # Query, Key, Value
        q = self.fc_q(features_norm).unsqueeze(2)  # [B, N, 1, out_channels]
        k = self.fc_k(neighbor_features)  # [B, N, k, out_channels]
        v = self.fc_v(neighbor_features)  # [B, N, k, out_channels]
        
        # Multi-head attention
        q = q.view(B, N, 1, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        pos_encoding = pos_encoding.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention: gamma(q - k + delta)
        attn_input = q - k + pos_encoding
        attn_input = attn_input.permute(0, 2, 3, 1, 4).reshape(B, N, self.k, -1)
        attn_scores = self.fc_gamma(attn_input)
        attn_scores = F.softmax(attn_scores / np.sqrt(self.head_dim), dim=2)
        
        # Weighted sum: attention * (v + delta)
        attn_scores = attn_scores.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        weighted_features = attn_scores * (v + pos_encoding)
        weighted_features = weighted_features.sum(dim=3)
        weighted_features = weighted_features.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        # Output projection with gating
        output = self.fc_out(weighted_features)
        
        # Residual connection
        if self.skip_proj is not None:
            features = self.skip_proj(features)
        
        output = self.norm2(output + features)
        
        # FFN with residual
        output = self.norm3(output + self.ffn(output))
        
        return output


# ============================================================================
# Cross-Attention Fusion (Improved)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Improved cross-attention for bidirectional feature fusion"""
    
    def __init__(self, pn2_channels, pt_channels, out_channels, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        # Project both inputs to same dimension
        self.pn2_proj = nn.Linear(pn2_channels, out_channels)
        self.pt_proj = nn.Linear(pt_channels, out_channels)
        
        # Cross-attention: PN2 queries PT
        self.q_pn2 = nn.Linear(out_channels, out_channels)
        self.k_pt = nn.Linear(out_channels, out_channels)
        self.v_pt = nn.Linear(out_channels, out_channels)
        
        # Cross-attention: PT queries PN2
        self.q_pt = nn.Linear(out_channels, out_channels)
        self.k_pn2 = nn.Linear(out_channels, out_channels)
        self.v_pn2 = nn.Linear(out_channels, out_channels)
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.Sigmoid()
        )
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, pn2_features, pt_features):
        """
        Args:
            pn2_features: [B, N, C1] features from PointNet++
            pt_features: [B, N, C2] features from Point Transformer
        Returns:
            fused_features: [B, N, out_channels]
        """
        B, N, _ = pn2_features.shape
        
        # Project to same dimension
        pn2_feat = self.pn2_proj(pn2_features)
        pt_feat = self.pt_proj(pt_features)
        
        # PN2 attends to PT
        q1 = self.q_pn2(pn2_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_pt(pt_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_pt(pt_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.matmul(attn1, v1).transpose(1, 2).reshape(B, N, -1)
        
        # PT attends to PN2
        q2 = self.q_pt(pt_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_pn2(pn2_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_pn2(pn2_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.matmul(attn2, v2).transpose(1, 2).reshape(B, N, -1)
        
        # Gated fusion
        gate_input = torch.cat([out1, out2, pn2_feat + pt_feat], dim=-1)
        gate = self.gate(gate_input)
        
        # Fuse with learned gating
        fused = torch.cat([out1 * gate, out2 * (1 - gate)], dim=-1)
        fused = self.fusion(fused)
        
        # Residual and norm
        fused = self.norm(fused + pn2_feat + pt_feat)
        
        return fused


# ============================================================================
# Improved Hybrid Model
# ============================================================================

class HybridPointNet(nn.Module):
    """
    Enhanced Hybrid PointNet++ and Point Transformer architecture V2
    
    Improvements:
    - 8-head attention in Point Transformer (up from 4)
    - Cross-attention fusion with gating mechanism
    - Edge-aware modules for boundary detection
    - Channel attention in decoder
    - FFN after transformer blocks
    - Better skip connections
    """
    
    def __init__(self, num_classes=2, num_heads=8):
        super(HybridPointNet, self).__init__()
        
        # ========== PointNet++ Branch ==========
        self.pn2_sa1 = PointNetSetAbstraction(512, 0.2, 32, 3 + 0, [64, 64, 128], False)
        self.pn2_sa2 = PointNetSetAbstraction(128, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.pn2_sa3 = PointNetSetAbstraction(32, 0.8, 32, 256 + 3, [256, 512, 1024], False)
        
        # ========== Point Transformer Branch ==========
        self.pt_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        
        # Enhanced transformer encoders with 8 heads
        self.pt_enc1 = PointTransformerBlock(128, 128, num_heads, k=32)
        self.pt_enc2 = PointTransformerBlock(128, 256, num_heads, k=32)
        self.pt_enc3 = PointTransformerBlock(256, 512, num_heads, k=16)
        
        # ========== Cross-Attention Fusion ==========
        self.fusion1 = CrossAttentionFusion(128, 128, 128, num_heads)
        self.fusion2 = CrossAttentionFusion(256, 256, 256, num_heads)
        self.fusion3 = CrossAttentionFusion(1024, 512, 512, num_heads)
        
        # ========== Edge-Aware Modules ==========
        self.edge_aware2 = EdgeAwareModule(128, k=16)
        self.edge_aware1 = EdgeAwareModule(128, k=16)
        
        # ========== Decoder ==========
        self.pn2_fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.pn2_fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.pn2_fp1 = PointNetFeaturePropagation(128 + 1, [128, 128, 128])  # +1 for edge prob
        
        # ========== Final Classification Head ==========
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, num_classes, 1)
        )
    
    def forward(self, xyz):
        """
        Args:
            xyz: [B, 3, N] point coordinates
        Returns:
            logits: [B, num_classes, N]
        """
        B, _, N = xyz.shape
        
        # ========== PointNet++ Branch Forward ==========
        pn2_l1_xyz, pn2_l1_points = self.pn2_sa1(xyz, None)  # [B, 3, 512], [B, 128, 512]
        pn2_l2_xyz, pn2_l2_points = self.pn2_sa2(pn2_l1_xyz, pn2_l1_points)  # [B, 3, 128], [B, 256, 128]
        pn2_l3_xyz, pn2_l3_points = self.pn2_sa3(pn2_l2_xyz, pn2_l2_points)  # [B, 3, 32], [B, 1024, 32]
        
        # ========== Point Transformer Branch Forward ==========
        xyz_pt = xyz.permute(0, 2, 1)  # [B, N, 3]
        pt_feat = self.pt_embed(xyz_pt)  # [B, N, 128]
        
        # Level 1: Process and downsample to 512 points
        pt_l1_feat = self.pt_enc1(xyz_pt, pt_feat)  # [B, N, 128]
        fps_idx1 = farthest_point_sample(xyz_pt, 512)
        pt_l1_xyz = index_points(xyz_pt, fps_idx1)  # [B, 512, 3]
        pt_l1_feat = index_points(pt_l1_feat, fps_idx1)  # [B, 512, 128]
        
        # Level 2: Process and downsample to 128 points
        pt_l2_feat = self.pt_enc2(pt_l1_xyz, pt_l1_feat)  # [B, 512, 256]
        fps_idx2 = farthest_point_sample(pt_l1_xyz, 128)
        pt_l2_xyz = index_points(pt_l1_xyz, fps_idx2)
        pt_l2_feat = index_points(pt_l2_feat, fps_idx2)  # [B, 128, 256]
        
        # Level 3: Process and downsample to 32 points
        pt_l3_feat = self.pt_enc3(pt_l2_xyz, pt_l2_feat)  # [B, 128, 512]
        fps_idx3 = farthest_point_sample(pt_l2_xyz, 32)
        pt_l3_xyz = index_points(pt_l2_xyz, fps_idx3)
        pt_l3_feat = index_points(pt_l3_feat, fps_idx3)  # [B, 32, 512]
        
        # ========== Cross-Attention Fusion ==========
        # Level 3: 32 points
        pn2_l3_feat = pn2_l3_points.permute(0, 2, 1)  # [B, 32, 1024]
        fused_l3 = self.fusion3(pn2_l3_feat, pt_l3_feat)  # [B, 32, 512]
        fused_l3 = fused_l3.permute(0, 2, 1)  # [B, 512, 32]
        
        # Level 2: 128 points
        pn2_l2_feat = pn2_l2_points.permute(0, 2, 1)  # [B, 128, 256]
        fused_l2 = self.fusion2(pn2_l2_feat, pt_l2_feat)  # [B, 128, 256]
        fused_l2 = fused_l2.permute(0, 2, 1)  # [B, 256, 128]
        
        # Level 1: 512 points
        pn2_l1_feat = pn2_l1_points.permute(0, 2, 1)  # [B, 512, 128]
        fused_l1 = self.fusion1(pn2_l1_feat, pt_l1_feat)  # [B, 512, 128]
        fused_l1 = fused_l1.permute(0, 2, 1)  # [B, 128, 512]
        
        # ========== Decoder with Edge-Awareness ==========
        l2_points = self.pn2_fp3(pn2_l2_xyz, pn2_l3_xyz, fused_l2, fused_l3)  # [B, 256, 128]
        l1_points = self.pn2_fp2(pn2_l1_xyz, pn2_l2_xyz, fused_l1, l2_points)  # [B, 128, 512]
        
        # Apply edge-aware refinement
        l1_xyz = pn2_l1_xyz.permute(0, 2, 1)  # [B, 512, 3]
        l1_feat = l1_points.permute(0, 2, 1)  # [B, 512, 128]
        l1_feat_refined, edge_prob1 = self.edge_aware2(l1_xyz, l1_feat)
        l1_points = l1_feat_refined.permute(0, 2, 1)  # [B, 128, 512]
        
        # Final propagation with edge probability
        edge_prob_up = F.interpolate(edge_prob1.permute(0, 2, 1), size=N, mode='nearest')  # [B, 1, N]
        l0_points = self.pn2_fp1(xyz, pn2_l1_xyz, edge_prob_up, l1_points)  # [B, 128, N]
        
        # Apply edge-aware at full resolution
        l0_xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        l0_feat = l0_points.permute(0, 2, 1)  # [B, N, 128]
        l0_feat_refined, _ = self.edge_aware1(l0_xyz, l0_feat)
        l0_points = l0_feat_refined.permute(0, 2, 1)  # [B, 128, N]
        
        # ========== Classification ==========
        logits = self.classifier(l0_points)  # [B, num_classes, N]
        
        # Apply log_softmax for NLLLoss
        log_probs = F.log_softmax(logits, dim=1)
        
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
        num_classes = inputs.size(1)
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss with class weights"""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal = focal.mean()
        
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice



