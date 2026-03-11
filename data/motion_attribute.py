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
import numpy as np
from typing import Tuple, Dict, List


# Body part partitioning based on human topology priors
# Each part contains indices of joints from the standard skeleton
PART_JOINTS = {
    0: [0, 1, 2, 3, 4],       # Head: nose, neck, head_left, head_right, head_top
    1: [5, 6, 7, 8, 9, 10],   # Torso: spine1, spine2, spine3, hip_left, hip_right
    2: [11, 12, 13, 14, 15], # Left Arm: shoulder_left, elbow_left, wrist_left, hand_left
    3: [16, 17, 18, 19, 20], # Right Arm: shoulder_right, elbow_right, wrist_right, hand_right
    4: [21, 22, 23, 24],     # Left Leg: hip_left, knee_left, ankle_left, foot_left
    5: [25, 26, 27, 28],     # Right Leg: hip_right, knee_right, ankle_right, foot_right
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
    variance = torch.var(distances, dim=-1, unbiased=True)  # () or (batch,)
    
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
    # Center the points
    centered = points - centroid.unsqueeze(-2)  # (N, 2) or (batch, N, 2)
    
    # Compute covariance matrix
    if centered.dim() == 2:
        # Single sample: (N, 2) -> (2, 2)
        cov = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)
    else:
        # Batch: (batch, N, 2) -> (batch, 2, 2)
        centered_t = centered.transpose(-2, -1)
        cov = torch.matmul(centered_t, centered) / (centered.shape[-2] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # (2,) or (batch, 2), (2, 2) or (batch, 2, 2)
    
    # Sort eigenvalues in descending order
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    # Eigenvalue ratio (with epsilon for stability)
    eps = 1e-8
    eigenvalue_ratio = eigenvalues[..., 0] / (eigenvalues[..., 1] + eps)
    
    # Principal angle with reference direction (1, 0) i.e., x-axis
    principal_vector = eigenvectors[..., :, 0]  # (2,) or (batch, 2)
    reference = torch.tensor([1.0, 0.0], device=points.device, dtype=points.dtype)
    if principal_vector.dim() > 1:
        reference = reference.unsqueeze(0).expand_as(principal_vector)
    
    principal_angle = torch.acos(torch.clamp(
        torch.sum(principal_vector * reference, dim=-1), 
        min=-1.0, max=1.0
    ))
    
    return eigenvalue_ratio, principal_angle


def compute_velocity_acceleration(centroids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute velocity and acceleration magnitudes from centroid trajectory.
    
    Args:
        centroids: Tensor of shape (T, 2) or (batch, T, 2)
        
    Returns:
        Tuple of (velocity_magnitude, acceleration_magnitude)
    """
    if centroids.dim() == 2:
        centroids = centroids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, T = centroids.shape[:2]
    
    # Velocity: v(t) = c(t) - c(t-1)
    velocity = centroids[:, 1:] - centroids[:, :-1]  # (batch, T-1, 2)
    
    # Acceleration: a(t) = c(t) - 2c(t-1) + c(t-2)
    accel = (centroids[:, 2:] - 2 * centroids[:, 1:-1] + centroids[:, :-2])  # (batch, T-2, 2)
    
    # Magnitudes
    velocity_mag = torch.norm(velocity, dim=-1)  # (batch, T-1)
    accel_mag = torch.norm(accel, dim=-1)  # (batch, T-2)
    
    # Average magnitudes
    velocity_mag = torch.mean(velocity_mag, dim=-1)  # (batch,)
    accel_mag = torch.mean(accel_mag, dim=-1)  # (batch,)
    
    if squeeze_output:
        velocity_mag = velocity_mag.squeeze(0)
        accel_mag = accel_mag.squeeze(0)
    
    return velocity_mag, accel_mag


def compute_relative_deformation(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Compute relative deformation magnitude (frame-to-frame internal deformation).
    
    Args:
        points: Tensor of shape (T, N, 2)
        centroids: Tensor of shape (T, 2)
        
    Returns:
        Relative deformation magnitude
    """
    T, N = points.shape[:2]
    
    if T < 2:
        return torch.tensor(0.0, device=points.device)
    
    # Relative positions
    rel_pos = points - centroids.unsqueeze(1)  # (T, N, 2)
    
    # Frame-to-frame difference
    rel_diff = rel_pos[1:] - rel_pos[:-1]  # (T-1, N, 2)
    
    # Magnitude
    diff_mag = torch.norm(rel_diff, dim=-1)  # (T-1, N)
    
    # Average
    deformation = torch.mean(diff_mag)  # scalar
    
    return deformation


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
            skeleton: Skeleton sequence of shape (T, J, 2) or (T, J, 3)
            part_idx: Index of the body part (0-5)
            
        Returns:
            Motion attribute tensor of shape (T, 8)
        """
        # Take only x, y coordinates if 3D
        if skeleton.shape[-1] == 3:
            skeleton = skeleton[..., :2]
        
        T, J = skeleton.shape[:2]
        
        # Get joint indices for this part
        joint_indices = self.part_joints[part_idx]
        part_points = skeleton[:, joint_indices, :]  # (T, N_p, 2)
        
        # Initialize attribute tensor
        attributes = torch.zeros(T, 8, device=skeleton.device, dtype=skeleton.dtype)
        
        for t in range(T):
            points = part_points[t]  # (N_p, 2)
            N_p = points.shape[0]
            
            # 1. Centroid
            centroid = compute_part_centroid(points)  # (2,)
            
            # 2. Compactness mean and variance
            d_mu, d_sigma = compute_compactness(points, centroid)
            
            # 3. Spatial extent
            e = compute_spatial_extent(points)
            
            # 4. PCA features
            rho, theta = compute_pca_features(points, centroid)
            
            # Store spatial features (first 5 dimensions)
            attributes[t, 0] = d_mu
            attributes[t, 1] = d_sigma
            attributes[t, 2] = e
            attributes[t, 3] = rho
            attributes[t, 4] = theta
        
        # Temporal features (require trajectory)
        if T >= 2:
            # Get centroids over time
            centroids_list = []
            for t in range(T):
                points = part_points[t]
                centroid = compute_part_centroid(points)
                centroids_list.append(centroid)
            centroids = torch.stack(centroids_list, dim=0)  # (T, 2)
            
            # Velocity and acceleration
            velocity_mag, accel_mag = compute_velocity_acceleration(centroids)
            attributes[:, 5] = velocity_mag
            attributes[:, 6] = accel_mag
            
            # Relative deformation
            deformation = compute_relative_deformation(part_points, centroids)
            attributes[:, 7] = deformation
        
        return attributes
    
    def compute_all_parts(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Compute motion attributes for all body parts.
        
        Args:
            skeleton: Skeleton sequence of shape (T, J, 2) or (T, J, 3)
            
        Returns:
            Motion attribute tensor of shape (T, P, 8)
        """
        all_attributes = []
        
        for part_idx in range(self.num_parts):
            part_attr = self.compute_attributes(skeleton, part_idx)
            all_attributes.append(part_attr)
        
        attributes = torch.stack(all_attributes, dim=1)  # (T, P, 8)
        
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
        if attributes.dim() == 3:
            # (T, P, 8) -> reshape for normalization
            original_shape = attributes.shape
            attributes_flat = attributes.view(-1, 8)  # (T*P, 8)
        else:
            original_shape = attributes.shape
            attributes_flat = attributes.view(-1, 8)
        
        if mean is None:
            mean = torch.mean(attributes_flat, dim=0)
        if std is None:
            std = torch.std(attributes_flat, dim=0)
        
        # Normalize
        normalized = (attributes_flat - mean) / (std + self.eps)
        
        # Reshape back
        normalized = normalized.view(original_shape)
        
        return normalized, mean, std


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
            0: [0, 1, 2, 3, 4],       # Head
            1: [5, 6, 7, 8, 9, 10],   # Torso
            2: [5, 6, 7],             # Left Arm
            3: [11, 12, 13],          # Right Arm
            4: [5, 14, 15, 16],       # Left Leg
            5: [5, 11, 12, 13],       # Right Leg (corrected)
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
    
    # Create dummy skeleton sequence (T=30 frames, J=25 joints)
    skeleton = torch.randn(30, 25, 3)  # 3D coordinates
    
    # Compute attributes
    attributes = extractor.compute_all_parts(skeleton)
    
    print(f"Skeleton shape: {skeleton.shape}")
    print(f"Attributes shape: {attributes.shape}")  # (30, 6, 8)
    
    # Normalize
    normalized, mean, std = extractor.normalize_attributes(attributes)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
