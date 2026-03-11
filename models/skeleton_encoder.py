"""
Skeleton Encoder Module
Implements the skeleton branch with Shift-GCN and Local Skeleton Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ShiftGCN(nn.Module):
    """
    Shift-GCN: Spatial-Temporal Graph Convolutional Network with shift operations
    Reference: https://github.com/liu-zhy/Shift-GCN
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Shift parameters
        self.shift = nn.Parameter(
            torch.zeros(1, in_channels, 1, kernel_size)
        )
        
        # 1x1 convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (N, C, T, V)
            
        Returns:
            Output tensor (N, C', T, V')
        """
        # Apply shift
        # x: (N, C, T, V, K)
        x_shift = self._shift(x, self.shift)
        
        # 1x1 conv
        x_conv = self.conv(x_shift)
        x_conv = self.bn(x_conv)
        
        return x_conv
    
    def _shift(self, x, shift):
        """Apply shift operation"""
        b, c, t, v = x.size()
        k = self.kernel_size
        
        # Reshape for shift
        x = x.view(b, c, t, v, 1)
        
        # Apply shift
        shift = shift.view(1, c, 1, 1, k)
        x = x * shift
        
        x = x.view(b, c, t, v * k)
        
        return x


class LocalSkeletonEncoder(nn.Module):
    """
    Local Skeleton Encoder for part-level skeletal features
    Encodes motion attributes from each body part
    """
    
    def __init__(self,
                 input_dim: int = 8,  # Motion attribute dimension
                 hidden_dim: int = 256,
                 num_parts: int = 6,
                 num_frames: int = 64,
                 dropout: float = 0.5):
        """
        Args:
            input_dim: Input dimension (motion attribute dimension)
            hidden_dim: Hidden dimension
            num_parts: Number of body parts
            num_frames: Number of frames
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_parts = num_parts
        self.hidden_dim = hidden_dim
        
        # Part-specific temporal convolutions
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            for _ in range(num_parts)
        ])
        
        # Part conditioning via attention
        self.part_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, motion_attributes: torch.Tensor, 
                global_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            motion_attributes: Motion attribute tensor (B, T, P, 8)
            global_feature: Optional global skeletal feature for guidance (B, D)
            
        Returns:
            Part-level skeletal features (B, P, D)
        """
        B, T, P, D_attr = motion_attributes.shape
        
        # Process each part
        part_features = []
        
        for p in range(P):
            # Extract part-specific motion (B, T, 8)
            part_attr = motion_attributes[:, :, p, :]
            
            # Transpose for conv1d: (B, 8, T)
            part_attr = part_attr.transpose(1, 2)
            
            # Apply temporal conv
            part_feat = self.part_convs[p](part_attr)  # (B, D, 1)
            part_feat = part_feat.squeeze(-1)  # (B, D)
            
            part_features.append(part_feat)
        
        # Stack: (B, P, D)
        part_features = torch.stack(part_features, dim=1)
        
        # Apply attention for refinement
        if global_feature is not None:
            # Use global feature as query
            query = global_feature.unsqueeze(1)  # (B, 1, D)
            
            refined_features, _ = self.part_attention(
                query=query,
                key=part_features,
                value=part_features
            )
            
            # Add residual
            part_features = part_features + refined_features
        
        # Output projection
        part_features = self.output_proj(part_features)
        part_features = F.normalize(part_features, p=2, dim=-1)
        
        return part_features


class SkeletonEncoder(nn.Module):
    """
    Complete Skeleton Encoder combining Shift-GCN and Local Skeleton Encoder
    """
    
    def __init__(self,
                 num_joints: int = 25,
                 num_classes: int = 60,
                 feature_dim: int = 256,
                 num_parts: int = 6,
                 dropout: float = 0.5,
                 pretrained_path: Optional[str] = None):
        """
        Args:
            num_joints: Number of skeleton joints
            num_classes: Number of action classes
            feature_dim: Feature dimension
            num_parts: Number of body parts
            dropout: Dropout rate
            pretrained_path: Path to pretrained Shift-GCN weights
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        
        # Shift-GCN for global features (frozen)
        self.shift_gcn = self._build_shift_gcn()
        
        # Local Skeleton Encoder
        self.local_encoder = LocalSkeletonEncoder(
            input_dim=8,  # Motion attribute dimension
            hidden_dim=feature_dim,
            num_parts=num_parts,
            dropout=dropout
        )
        
        # Global pooling for Shift-GCN output
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global feature projection
        self.global_proj = nn.Linear(2048, feature_dim)  # Shift-GCN output dim is 2048
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
    
    def _build_shift_gcn(self):
        """Build Shift-GCN architecture"""
        # This is a simplified version - use actual Shift-GCN in practice
        layers = []
        
        # Spatial temporal graph convolution layers
        in_channels_list = [3, 64, 128, 256]
        out_channels_list = [64, 128, 256, 2048]
        
        for i, (in_ch, out_ch) in enumerate(zip(in_channels_list, out_channels_list)):
            layers.append(
                ShiftGCN(in_ch, out_ch, kernel_size=3)
            )
            if i < len(in_channels_list) - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _load_pretrained(self, path: str):
        """Load pretrained Shift-GCN weights"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.shift_gcn.load_state_dict(state_dict)
            print(f"Loaded pretrained Shift-GCN from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, skeleton: torch.Tensor, 
                motion_attributes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            skeleton: Raw skeleton sequence (B, T, J, 3)
            motion_attributes: Motion attribute tensor (B, T, P, 8)
            
        Returns:
            Tuple of (global_features, part_features)
                global_features: (B, D)
                part_features: (B, P, D)
        """
        # Get global features from Shift-GCN
        # Transpose: (B, T, J, 3) -> (B, 3, T, J)
        x = skeleton.permute(0, 3, 1, 2)  # (B, 3, T, J)
        
        # Apply Shift-GCN
        with torch.no_grad():  # Frozen
            global_features = self.shift_gcn(x)
        
        # Global pooling
        global_features = self.global_pool(global_features)  # (B, 2048, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (B, 2048)
        
        # Project to feature dimension
        global_features = self.global_proj(global_features)  # (B, D)
        global_features = F.normalize(global_features, p=2, dim=-1)
        
        # Get part-level features from Local Skeleton Encoder
        part_features = self.local_encoder(motion_attributes, global_features)
        
        return global_features, part_features
    
    def freeze_shift_gcn(self):
        """Freeze Shift-GCN parameters"""
        for param in self.shift_gcn.parameters():
            param.requires_grad = False
        self.shift_gcn.eval()
    
    def unfreeze_shift_gcn(self):
        """Unfreeze Shift-GCN parameters"""
        for param in self.shift_gcn.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test skeleton encoder
    encoder = SkeletonEncoder(
        num_joints=25,
        num_classes=60,
        feature_dim=256,
        num_parts=6
    )
    
    # Dummy input
    B, T, J, P = 4, 64, 25, 6
    skeleton = torch.randn(B, T, J, 3)
    motion_attr = torch.randn(B, T, P, 8)
    
    global_feat, part_feat = encoder(skeleton, motion_attr)
    
    print(f"Skeleton shape: {skeleton.shape}")
    print(f"Motion attributes shape: {motion_attr.shape}")
    print(f"Global features shape: {global_feat.shape}")
    print(f"Part features shape: {part_feat.shape}")
