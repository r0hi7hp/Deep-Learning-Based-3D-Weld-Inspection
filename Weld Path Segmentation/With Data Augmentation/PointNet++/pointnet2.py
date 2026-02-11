"""
PointNet++ Model for Semantic Segmentation - Production Ready
Architecture for weld detection on point clouds with attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared Euclidean distance between point sets."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points based on given indices."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query for grouping points."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, 
                     points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample and group points for set abstraction."""
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group all points as a single set."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class SelfAttention(nn.Module):
    """Lightweight self-attention for local feature enhancement."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.size()
        query = self.query(x).view(B, -1, N).permute(0, 2, 1)
        key = self.key(x).view(B, -1, N)
        value = self.value(x).view(B, -1, N)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        return self.gamma * out + x


class PointNetSetAbstraction(nn.Module):
    """Set abstraction layer for hierarchical point processing."""
    
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, 
                 mlp: list, group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation layer for upsampling."""
    
    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, 
                points1: Optional[torch.Tensor], points2: torch.Tensor) -> torch.Tensor:
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2SemSeg(nn.Module):
    """PointNet++ for semantic segmentation with attention."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Encoder
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 64, 3, [64, 64, 128], False)
        self.attn1 = SelfAttention(128)
        self.sa2 = PointNetSetAbstraction(512, 0.2, 64, 128 + 3, [128, 128, 256], False)
        self.attn2 = SelfAttention(256)
        self.sa3 = PointNetSetAbstraction(128, 0.4, 64, 256 + 3, [256, 256, 512], False)
        self.attn3 = SelfAttention(512)
        self.sa4 = PointNetSetAbstraction(32, 0.8, 64, 512 + 3, [512, 512, 1024], False)
        self.attn4 = SelfAttention(1024)
        
        # Decoder
        self.fp4 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp4_attn = SelfAttention(512)
        self.fp3 = PointNetFeaturePropagation(768, [512, 512])
        self.fp3_attn = SelfAttention(512)
        self.fp2 = PointNetFeaturePropagation(640, [512, 256])
        self.fp2_attn = SelfAttention(256)
        self.fp1 = PointNetFeaturePropagation(256, [256, 128, 128])
        self.fp1_attn = SelfAttention(128)
        
        # Classifier
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.gn1 = nn.GroupNorm(16, 256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.gn2 = nn.GroupNorm(8, 128)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.drop3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        l0_xyz = xyz
        l0_points = None

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.attn1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.attn2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.attn3(l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.attn4(l4_points)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.fp4_attn(l3_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.fp3_attn(l2_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.fp2_attn(l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        l0_points = self.fp1_attn(l0_points)

        # Classifier
        x = self.drop1(F.relu(self.gn1(self.conv1(l0_points))))
        x = self.drop2(F.relu(self.gn2(self.conv2(x))))
        x = self.drop3(F.relu(self.gn3(self.conv3(x))))
        x = self.conv4(x)
        return F.log_softmax(x, dim=1)