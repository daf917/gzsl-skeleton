"""
Motion Attribute Computation Module

This module computes part-level motion attributes from skeleton sequences
as described in the paper:
"Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives"

Each body part is characterized by an 8-dimensional motion attribute vector:
1. Compactness Mean (d_mu)
2. Compactness Variance (d_sigma^2)
3. Spatial Extent (e)
4. PCA Eigenvalue Ratio (rho)
5. Principal Angle (theta)
6. Velocity Magnitude (k)
7. Acceleration Magnitude (a_c)
8. Relative Deformation (E_rel)
"""

import torch
from typing import Tuple, Dict, List


# Body part partitioning based on human topology priors
# Each part contains indices of joints from the standard skeleton
PART_JOINTS = {
    0: [2, 3, 20],              # Head/neck in NTU RGB+D 25-joint format
    1: [0, 1, 2, 12, 16, 20],   # Torso and hips
    2: [4, 5, 6, 7, 21, 22],    # Left arm and hand tips
    3: [8, 9, 10, 11, 23, 24],  # Right arm and hand tips
    4: [12, 13, 14, 15],        # Left leg
    5: [16, 17, 18, 19],        # Right leg
}


def compute_part_centroid(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the centroid of a body part.
    
    Args:
        points: Tensor of shape (N, 2) or (batch, N, 2) containing 2D joint coordinates
        
    Returns:
        Centroid tensor of shape (2,) or (batch, 2)
    """
    return torch.mean(points, dim=-2)


def compute_compactness(points: torch.Tensor, centroid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute compactness mean and variance based on distances to centroid.
    
    Args:
        points: Tensor of shape (N, 2) or (batch, N, 2)
        centroid: Tensor of shape (2,) or (batch, 2)
        
    Returns:
        Tuple of (mean, variance) of distances
    """
    distances = torch.norm(points - centroid.unsqueeze(-2), dim=-1)  # (N,) or (batch, N)
    
    mean = torch.mean(distances, dim=-1)  # () or (batch,)
    variance = torch.var(distances, dim=-1, unbiased=False)  # () or (batch,)
    
    return mean, variance


def compute_spatial_extent(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the diagonal length of the axis-aligned bounding box.
    
    Args:
        points: Tensor of shape (N, 2) or (batch, N, 2)
        
    Returns:
        Spatial extent scalar
    """
    max_coords = torch.max(points, dim=-2)[0]  # (2,) or (batch, 2)
    min_coords = torch.min(points, dim=-2)[0]  # (2,) or (batch, 2)
    
    extent = torch.norm(max_coords - min_coords, dim=-1)  # () or (batch,)
    
    return extent


def compute_pca_features(points: torch.Tensor, centroid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PCA eigenvalue ratio and principal angle.
    
    Args:
        points: Tensor of shape (N, 2) or (batch, N, 2)
        centroid: Tensor of shape (2,) or (batch, 2)
        
    Returns:
        Tuple of (eigenvalue_ratio, principal_angle)
    """
    centered = points - centroid.unsqueeze(-2)
    covariance = centered.transpose(-2, -1) @ centered / centered.shape[-2]

    # torch.linalg.eigh returns eigenvalues in ascending order with matching columns.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    eigenvalue_ratio = eigenvalues[..., -1] / (eigenvalues[..., -2] + 1e-8)

    # Projection onto the fixed x-axis. Its magnitude discards eigenvector sign.
    principal_projection = eigenvectors[..., 0, -1].abs().clamp_max(1.0)
    principal_angle = torch.acos(principal_projection)

    return eigenvalue_ratio, principal_angle


def compute_velocity_acceleration(
    centroids: torch.Tensor, temporal_window: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return trailing-window means of per-frame velocity and acceleration magnitudes."""
    if temporal_window < 1:
        raise ValueError("The temporal window must be at least one frame.")
    previous = torch.cat((centroids[..., :1, :], centroids[..., :-1, :]), dim=-2)
    before_previous = torch.cat((previous[..., :1, :], previous[..., :-1, :]), dim=-2)
    velocity = (centroids - previous).norm(dim=-1)
    acceleration = (centroids - 2 * previous + before_previous).norm(dim=-1)

    magnitudes = torch.stack((velocity, acceleration), dim=-2)
    prefix = torch.cat((torch.zeros_like(magnitudes[..., :1]), magnitudes.cumsum(dim=-1)), dim=-1)
    end = torch.arange(1, centroids.size(-2) + 1, device=centroids.device)
    start = (end - temporal_window).clamp_min(0)
    averages = (prefix[..., 1:] - prefix.index_select(-1, start)) / (end - start)
    return averages.unbind(dim=-2)


def compute_relative_deformation(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Return per-frame mean joint deformation relative to the part centroid."""
    relative = points - centroids.unsqueeze(-2)
    previous = torch.cat((relative[..., :1, :, :], relative[..., :-1, :, :]), dim=-3)
    return (relative - previous).norm(dim=-1).mean(dim=-1)


class MotionAttributeExtractor:
    """
    Extracts motion attributes from skeleton sequences for each body part.
    """
    
    def __init__(self, num_parts: int = 6, temporal_window: int = 10, 
                 epsilon: float = 1e-8):
        """
        Args:
            num_parts: Number of body parts (default: 6)
            temporal_window: Window size for temporal smoothing
            epsilon: Small constant for numerical stability
        """
        self.num_parts = num_parts
        self.temporal_window = temporal_window
        self.eps = epsilon
        
        # Define body part joint indices
        self.part_joints = PART_JOINTS
        
    def compute_attributes(self, skeleton: torch.Tensor, 
                          part_idx: int) -> torch.Tensor:
        """
        Compute motion attributes for a single body part.
        
        Args:
            skeleton: Skeletons of shape (T, J, 2) or (B, T, J, 2), optionally with a third channel.
            part_idx: Index of the body part (0-5)
            
        Returns:
            Motion attributes of shape (T, 8) or (B, T, 8).
        """
        joint_indices = self.part_joints[part_idx]
        if not joint_indices:
            raise ValueError(f"No joints assigned to part {part_idx}.")
        part_points = skeleton[..., joint_indices, :2]
        centroids = compute_part_centroid(part_points)
        d_mu, d_sigma = compute_compactness(part_points, centroids)
        extent = compute_spatial_extent(part_points)
        ratio, angle = compute_pca_features(part_points, centroids)
        velocity, acceleration = compute_velocity_acceleration(centroids, self.temporal_window)
        deformation = compute_relative_deformation(part_points, centroids)
        return torch.stack((d_mu, d_sigma, extent, ratio, angle, velocity, acceleration, deformation), dim=-1)

    def compute_all_parts(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Compute motion attributes for all body parts.
        
        Args:
            skeleton: Skeletons of shape (T, J, 2) or (B, T, J, 2), optionally with a third channel.
            
        Returns:
            Motion attributes of shape (T, P, 8) or (B, T, P, 8).
        """
        all_attributes = []
        
        for part_idx in range(self.num_parts):
            part_attr = self.compute_attributes(skeleton, part_idx)
            all_attributes.append(part_attr)
        
        attributes = torch.stack(all_attributes, dim=-2)
        
        return attributes
    
    def normalize_attributes(self, attributes: torch.Tensor, 
                           mean: torch.Tensor = None, 
                           std: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply z-score normalization to attributes.
        
        Args:
            attributes: Attribute tensor of shape (T, P, 8) or (B, T, P, 8)
            mean: Pre-computed mean (if None, compute from input)
            std: Pre-computed std (if None, compute from input)
            
        Returns:
            Tuple of (normalized_attributes, mean, std)
        """
        attributes_flat = attributes.reshape(-1, 8)
        if mean is None:
            mean = attributes_flat.mean(dim=0)
        if std is None:
            std = attributes_flat.std(dim=0, unbiased=False)
        return (attributes - mean) / (std + self.eps), mean, std


def create_part_joint_mapping(num_joints: int) -> Dict[int, List[int]]:
    """
    Create body part to joint mapping for different skeleton formats.
    
    Args:
        num_joints: Number of joints in the skeleton
        
    Returns:
        Dictionary mapping part index to list of joint indices
    """
    if num_joints == 25:
        # NTU RGB+D format
        return PART_JOINTS
    elif num_joints == 17:
        # COCO format
        return {
            0: [0, 1, 2, 3, 4],    # Head
            1: [5, 6, 11, 12],     # Torso
            2: [5, 7, 9],          # Left arm
            3: [6, 8, 10],         # Right arm
            4: [11, 13, 15],       # Left leg
            5: [12, 14, 16],       # Right leg
        }
    else:
        # Generic mapping - divide joints evenly
        joints_per_part = num_joints // 6
        mapping = {}
        for i in range(6):
            start = i * joints_per_part
            end = start + joints_per_part if i < 5 else num_joints
            mapping[i] = list(range(start, end))
        return mapping


if __name__ == "__main__":
    # Test the motion attribute extractor
    extractor = MotionAttributeExtractor(num_parts=6)
    
    # Create an example skeleton sequence (T=30 frames, J=25 joints)
    skeleton = torch.randn(30, 25, 3)  # 3D coordinates
    
    # Compute attributes
    attributes = extractor.compute_all_parts(skeleton)
    
    print(f"Skeleton shape: {skeleton.shape}")
    print(f"Attributes shape: {attributes.shape}")  # (30, 6, 8)
    
    # Normalize
    normalized, mean, std = extractor.normalize_attributes(attributes)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
