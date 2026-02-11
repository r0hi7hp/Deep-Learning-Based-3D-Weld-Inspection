"""
KPConv Operations - Kernel Point Convolution Operations
Implements core KPConv layer with rigid and deformable variants
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# Utility Functions
# ============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between each pair of points
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


def gather_nd(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather features using indices
    Args:
        features: [B, N, C]
        idx: [B, S] or [B, S, K]
    Returns:
        gathered features: [B, S, C] or [B, S, K, C]
    """
    batch_size = features.shape[0]
    
    if len(idx.shape) == 2:
        # [B, S] -> [B, S, C]
        batch_indices = torch.arange(batch_size, device=idx.device).view(-1, 1).expand_as(idx)
        return features[batch_indices, idx]
    else:
        # [B, S, K] -> [B, S, K, C]
        B, S, K = idx.shape
        batch_indices = torch.arange(batch_size, device=idx.device).view(-1, 1, 1).expand(B, S, K)
        return features[batch_indices, idx]


def radius_neighbors(query_points: torch.Tensor, 
                    support_points: torch.Tensor,
                    radius: float,
                    max_neighbors: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find neighbors within a radius
    Args:
        query_points: [B, N, 3]
        support_points: [B, M, 3]
        radius: search radius
        max_neighbors: maximum number of neighbors
    Returns:
        neighbor_idx: [B, N, max_neighbors]
        neighbor_mask: [B, N, max_neighbors] boolean mask
    """
    B, N, _ = query_points.shape
    M = support_points.shape[1]
    
    # Compute squared distances
    sq_dists = square_distance(query_points, support_points)  # [B, N, M]
    
    # Find points within radius
    radius_sq = radius ** 2
    in_radius = sq_dists < radius_sq  # [B, N, M]
    
    # Get indices and mask
    # Sort by distance and take top max_neighbors
    sq_dists_masked = torch.where(in_radius, sq_dists, torch.full_like(sq_dists, float('inf')))
    distances, indices = torch.topk(sq_dists_masked, k=min(max_neighbors, M), dim=-1, largest=False)
    
    # Create mask for valid neighbors
    neighbor_mask = distances < float('inf')
    
    # Replace invalid indices with 0 (will be masked anyway)
    neighbor_idx = torch.where(neighbor_mask, indices, torch.zeros_like(indices))
    
    return neighbor_idx, neighbor_mask


def knn_search(query_points: torch.Tensor, 
               support_points: torch.Tensor,
               k: int) -> torch.Tensor:
    """
    K-nearest neighbor search
    Args:
        query_points: [B, N, 3]
        support_points: [B, M, 3]
        k: number of neighbors
    Returns:
        neighbor_idx: [B, N, k]
    """
    sq_dists = square_distance(query_points, support_points)  # [B, N, M]
    _, indices = torch.topk(sq_dists, k=k, dim=-1, largest=False)
    return indices


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
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
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Random starting point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def grid_subsampling(points: torch.Tensor, 
                     grid_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grid subsampling - select points as barycenters of grid cells
    Args:
        points: [B, N, 3]
        grid_size: size of grid cells
    Returns:
        subsampled_points: [B, M, 3]
        indices: [B, M] indices of selected points
    """
    B, N, _ = points.shape
    device = points.device
    
    # Use FPS as approximate grid subsampling
    # Estimate number of points based on grid size
    point_range = points.max(dim=1)[0] - points.min(dim=1)[0]  # [B, 3]
    avg_range = point_range.mean(dim=1)  # [B]
    npoint = max(1, int((avg_range[0].item() / grid_size) ** 2))
    npoint = min(npoint, N // 2, 512)  # Cap at reasonable size
    
    indices = farthest_point_sample(points, npoint)  # [B, npoint]
    subsampled = gather_nd(points, indices)  # [B, npoint, 3]
    
    return subsampled, indices


# ============================================================================
# Kernel Point Generation
# ============================================================================

def generate_kernel_points(num_kpoints: int = 15, 
                          radius: float = 1.0,
                          dimension: int = 3) -> torch.Tensor:
    """
    Generate kernel points distributed on a sphere using optimization
    Args:
        num_kpoints: number of kernel points
        radius: radius of the kernel
        dimension: spatial dimension (3 for 3D)
    Returns:
        kernel_points: [num_kpoints, dimension]
    """
    # Initialize points on sphere surface
    if num_kpoints == 1:
        return torch.zeros(1, dimension)
    
    # Use fibonacci sphere for initial distribution
    kernel_points = torch.zeros(num_kpoints, dimension)
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for i in range(num_kpoints):
        y = 1 - (i / float(num_kpoints - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)
        
        theta = golden_angle * i
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        kernel_points[i] = torch.tensor([x, y, z]) * radius
    
    return kernel_points


# ============================================================================
# KPConv Layer
# ============================================================================

class KPConv(nn.Module):
    """
    Kernel Point Convolution layer
    Implements convolution using learnable kernel points
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 15,
                 radius: float = 0.1,
                 sigma: float = 0.5,
                 deformable: bool = False):
        """
        Args:
            in_channels: input feature dimension
            out_channels: output feature dimension
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of kernel points (as fraction of radius)
            deformable: whether to use deformable kernels
        """
        super(KPConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma * radius
        self.deformable = deformable
        
        # Generate initial kernel points
        kernel_points = generate_kernel_points(kernel_size, radius * 0.8)
        self.register_buffer('kernel_points', kernel_points)  # [K, 3]
        
        # Kernel weights: [K, in_channels, out_channels]
        self.weights = nn.Parameter(
            torch.zeros(kernel_size, in_channels, out_channels)
        )
        nn.init.kaiming_uniform_(self.weights.view(-1, out_channels), a=np.sqrt(5))
        
        # Deformable offsets
        if deformable:
            self.offset_mlp = nn.Sequential(
                nn.Linear(in_channels, kernel_size * 3),
            )
            nn.init.zeros_(self.offset_mlp[0].weight)
            nn.init.zeros_(self.offset_mlp[0].bias)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self,
                query_points: torch.Tensor,
                support_points: torch.Tensor,
                support_features: torch.Tensor,
                neighbor_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            query_points: [B, N, 3] query point positions
            support_points: [B, M, 3] support point positions
            support_features: [B, M, C_in] support point features
            neighbor_idx: [B, N, K'] indices of neighbors
        Returns:
            output_features: [B, N, C_out]
        """
        B, N, _ = query_points.shape
        M = support_points.shape[1]
        K_neighbors = neighbor_idx.shape[2]
        
        # Get neighbor positions and features
        neighbor_points = gather_nd(support_points, neighbor_idx)  # [B, N, K', 3]
        neighbor_features = gather_nd(support_features, neighbor_idx)  # [B, N, K', C_in]
        
        # Compute relative positions
        relative_pos = neighbor_points - query_points.unsqueeze(2)  # [B, N, K', 3]
        
        # Get kernel points (possibly with deformable offsets)
        if self.deformable:
            # Compute offsets based on center features
            center_features = support_features.mean(dim=1, keepdim=True)  # [B, 1, C_in]
            offsets = self.offset_mlp(center_features)  # [B, 1, K*3]
            offsets = offsets.view(B, 1, self.kernel_size, 3)  # [B, 1, K, 3]
            kernel_points = self.kernel_points.unsqueeze(0).unsqueeze(0) + offsets  # [B, 1, K, 3]
            kernel_points = kernel_points.expand(B, N, -1, -1)  # [B, N, K, 3]
        else:
            kernel_points = self.kernel_points.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 3]
            kernel_points = kernel_points.expand(B, N, -1, -1)  # [B, N, K, 3]
        
        # Compute kernel influences using Gaussian
        # relative_pos: [B, N, K', 3]
        # kernel_points: [B, N, K, 3]
        
        # Expand for broadcasting
        relative_pos_exp = relative_pos.unsqueeze(3)  # [B, N, K', 1, 3]
        kernel_points_exp = kernel_points.unsqueeze(2)  # [B, N, 1, K, 3]
        
        # Squared distance between neighbors and kernel points
        sq_dist = torch.sum((relative_pos_exp - kernel_points_exp) ** 2, dim=-1)  # [B, N, K', K]
        
        # Gaussian influence
        influences = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # [B, N, K', K]
        
        # Normalize influences per neighbor
        influences = influences / (influences.sum(dim=-1, keepdim=True) + 1e-6)  # [B, N, K', K]
        
        # Aggregate features weighted by influences
        # neighbor_features: [B, N, K', C_in]
        # influences: [B, N, K', K]
        # weights: [K, C_in, C_out]
        
        # Reshape for efficient computation
        weighted_features = torch.einsum('bnkc,bnkj,jco->bno',
                                         neighbor_features, influences, self.weights)
        
        # Add bias
        output = weighted_features + self.bias
        
        return output


# ============================================================================
# KPConv Residual Block
# ============================================================================

class KPConvBlock(nn.Module):
    """
    KPConv Residual Block with bottleneck design
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 radius: float,
                 kernel_size: int = 15,
                 deformable: bool = False,
                 strided: bool = False):
        """
        Args:
            in_channels: input feature dimension
            out_channels: output feature dimension
            radius: convolution radius
            kernel_size: number of kernel points
            deformable: use deformable convolution
            strided: whether this is a strided (downsampling) block
        """
        super(KPConvBlock, self).__init__()
        
        self.strided = strided
        bottleneck_dim = out_channels // 4
        
        # Pre-processing
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Linear(in_channels, bottleneck_dim)
        
        # KPConv
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        self.relu2 = nn.LeakyReLU(0.1)
        self.kpconv = KPConv(
            bottleneck_dim, bottleneck_dim,
            kernel_size=kernel_size,
            radius=radius,
            deformable=deformable
        )
        
        # Post-processing
        self.bn3 = nn.BatchNorm1d(bottleneck_dim)
        self.relu3 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Linear(bottleneck_dim, out_channels)
        
        # Shortcut
        if in_channels != out_channels or strided:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self,
                query_points: torch.Tensor,
                support_points: torch.Tensor,
                features: torch.Tensor,
                neighbor_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            query_points: [B, N, 3]
            support_points: [B, M, 3]
            features: [B, M, C_in]
            neighbor_idx: [B, N, K]
        Returns:
            output: [B, N, C_out]
        """
        B, M, C = features.shape
        N = query_points.shape[1]
        
        # Pre-processing
        x = self.bn1(features.transpose(1, 2)).transpose(1, 2)
        x = self.relu1(x)
        x = self.conv1(x)  # [B, M, bottleneck_dim]
        
        # KPConv
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu2(x)
        x = self.kpconv(query_points, support_points, x, neighbor_idx)  # [B, N, bottleneck_dim]
        
        # Post-processing
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu3(x)
        x = self.conv3(x)  # [B, N, out_channels]
        
        # Shortcut
        if self.strided:
            # Need to downsample features for shortcut
            shortcut_features = gather_nd(features, neighbor_idx[:, :, 0])  # [B, N, C_in]
            shortcut = self.shortcut(shortcut_features)
        else:
            shortcut = self.shortcut(features)
        
        return x + shortcut


class SimpleKPConvBlock(nn.Module):
    """
    Simplified KPConv Block without residual connection (for first layer)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 radius: float,
                 kernel_size: int = 15):
        super(SimpleKPConvBlock, self).__init__()
        
        self.kpconv = KPConv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            radius=radius,
            deformable=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self,
                query_points: torch.Tensor,
                support_points: torch.Tensor,
                features: torch.Tensor,
                neighbor_idx: torch.Tensor) -> torch.Tensor:
        x = self.kpconv(query_points, support_points, features, neighbor_idx)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        return x


# ============================================================================
# Upsampling Layer
# ============================================================================

class NearestUpsampleBlock(nn.Module):
    """
    Upsample features using nearest neighbor interpolation
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(NearestUpsampleBlock, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self,
                query_points: torch.Tensor,
                support_points: torch.Tensor,
                support_features: torch.Tensor,
                skip_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Upsample features and optionally concatenate skip features
        Args:
            query_points: [B, N, 3] target points
            support_points: [B, M, 3] source points
            support_features: [B, M, C] source features
            skip_features: [B, N, C'] optional skip connection features
        Returns:
            output: [B, N, C_out]
        """
        B, N, _ = query_points.shape
        M = support_points.shape[1]
        
        # Find nearest neighbors
        sq_dist = square_distance(query_points, support_points)  # [B, N, M]
        _, nearest_idx = torch.min(sq_dist, dim=-1)  # [B, N]
        
        # Gather features
        upsampled = gather_nd(support_features, nearest_idx)  # [B, N, C]
        
        # Concatenate skip features if provided
        if skip_features is not None:
            upsampled = torch.cat([upsampled, skip_features], dim=-1)
        
        # MLP
        B, N, C = upsampled.shape
        output = self.mlp[0](upsampled)
        output = self.mlp[1](output.transpose(1, 2)).transpose(1, 2)
        output = self.mlp[2](output)
        
        return output


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    print("Testing KPConv Operations...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    B, N, M = 2, 256, 512
    C_in, C_out = 64, 128
    
    # Create dummy data
    query_points = torch.randn(B, N, 3, device=device)
    support_points = torch.randn(B, M, 3, device=device)
    support_features = torch.randn(B, M, C_in, device=device)
    
    # Find neighbors
    neighbor_idx, _ = radius_neighbors(query_points, support_points, radius=0.5, max_neighbors=32)
    print(f"Neighbor indices shape: {neighbor_idx.shape}")
    
    # Test KPConv layer
    kpconv = KPConv(C_in, C_out, kernel_size=15, radius=0.5).to(device)
    output = kpconv(query_points, support_points, support_features, neighbor_idx)
    print(f"KPConv output shape: {output.shape}")
    
    # Test KPConv block
    block = KPConvBlock(C_in, C_out, radius=0.5, strided=True).to(device)
    output = block(query_points, support_points, support_features, neighbor_idx)
    print(f"KPConvBlock output shape: {output.shape}")
    
    # Test upsampling
    upsample = NearestUpsampleBlock(C_out, 64).to(device)
    upsampled = upsample(support_points, query_points, output[:, :N])
    print(f"Upsampled features shape: {upsampled.shape}")
    
    print("\nAll KPConv operations tests passed!")
