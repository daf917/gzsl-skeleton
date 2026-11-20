"""
Aggregation Module
Composes part-level features into a global representation using attention
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveAggregation(nn.Module):
    """
    Aggregates part-level (primitive) features into a compositional global representation
    using attention-weighted aggregation
    """
    
    def __init__(self, feature_dim: int = 512, num_parts: int = 6):
        """
        Args:
            feature_dim: Feature dimension
            num_parts: Number of body parts (primitives)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        
        # Attention score computation
        self.W = nn.Linear(feature_dim, feature_dim, bias=False)
        self.w = nn.Linear(feature_dim, 1, bias=False)
        
    def forward(self, part_features: torch.Tensor) -> tuple:
        """
        Aggregate part-level features into global representation
        
        Args:
            part_features: Part-level skeletal features (B, P, D)
            
        Returns:
            Tuple of (aggregated_global_features, attention_weights)
                aggregated_global_features: (B, D)
                attention_weights: (B, P)
        """
        B, P, D = part_features.shape
        
        # Compute attention scores for each part
        # gamma_i,p = w^T * sigma(W * h_i,p)
        hidden = self.W(part_features)  # (B, P, D)
        hidden = F.gelu(hidden)  # Apply nonlinearity
        scores = self.w(hidden).squeeze(-1)  # (B, P)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, P)
        
        # Aggregate: sum over parts with attention weights
        aggregated = torch.bmm(attention_weights.unsqueeze(1), part_features)  # (B, 1, D)
        aggregated = aggregated.squeeze(1)  # (B, D)
        
        # L2 normalize
        aggregated = F.normalize(aggregated, p=2, dim=-1)
        
        return aggregated, attention_weights


class PrimitiveDecorrelationRegularizer(nn.Module):
    """Penalize squared off-diagonal batch correlations between primitive features."""

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_features: Part-level features (B, P, D).

        Returns:
            Sum of squared off-diagonal correlations as in the paper.
        """
        num_parts = part_features.size(1)
        centered = part_features - part_features.mean(dim=0, keepdim=True)
        eps = max(1e-8, torch.finfo(centered.dtype).tiny)
        normalized = F.normalize(centered, p=2, dim=-1, eps=eps)
        correlations = (normalized @ normalized.transpose(-2, -1)).mean(dim=0)
        diagonal = torch.eye(num_parts, dtype=torch.bool, device=part_features.device)
        return correlations.masked_fill(diagonal, 0.0).square().sum()


class PrimitiveIndependenceRegularizer(PrimitiveDecorrelationRegularizer):
    """Legacy API returning the mean off-diagonal squared correlation."""

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        num_parts = part_features.size(1)
        return super().forward(part_features) / max(num_parts * (num_parts - 1), 1)


class GlobalConsistencyLoss(nn.Module):
    """
    Enforces consistency between compositional global features and Shift-GCN global features
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                compositional_global: torch.Tensor, 
                shiftgcn_global: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss
        
        Args:
            compositional_global: Aggregated global features (B, D)
            shiftgcn_global: Shift-GCN global features (B, D)
            
        Returns:
            Consistency loss scalar
        """
        # Both inputs are L2-normalized before evaluating the dot product.
        similarity = (compositional_global * shiftgcn_global).sum(dim=-1)
        
        loss = 1.0 - similarity.mean()
        
        return loss


class AlignmentLoss(nn.Module):
    """Compare skeletal features with one text anchor per seen class."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def primitive_level_alignment(
        self,
        skeleton_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compare each sample and body part against that part for every seen class."""
        scores = torch.einsum("bpd,cpd->bpc", skeleton_features, text_features)
        scores = scores / self.temperature
        targets = labels[:, None].expand(-1, skeleton_features.size(1))
        return F.cross_entropy(scores.flatten(0, 1), targets.reshape(-1))

    def global_level_alignment(
        self,
        skeleton_global: torch.Tensor,
        text_global: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compare each Shift-GCN feature against the global seen-class text anchors."""
        scores = skeleton_global @ text_global.T / self.temperature
        return F.cross_entropy(scores, labels)

    def forward(
        self,
        skeleton_global: torch.Tensor,
        skeleton_part: torch.Tensor,
        text_global: torch.Tensor,
        text_part: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return global and primitive losses, with labels indexing text-anchor rows."""
        global_loss = self.global_level_alignment(skeleton_global, text_global, labels)
        primitive_loss = self.primitive_level_alignment(skeleton_part, text_part, labels)
        return global_loss, primitive_loss


if __name__ == "__main__":
    # Test aggregation module
    aggregation = PrimitiveAggregation(feature_dim=512, num_parts=6)
    
    # Example input
    B, P, D = 4, 6, 256
    part_features = torch.randn(B, P, D)
    
    global_feat, attention = aggregation(part_features)
    
    print(f"Part features shape: {part_features.shape}")
    print(f"Global features shape: {global_feat.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention sum (should be 1): {attention.sum(dim=-1)}")
    
    decorrelation = PrimitiveDecorrelationRegularizer()
    decorr_loss = decorrelation(part_features)
    print(f"\nDecorrelation loss: {decorr_loss.item():.4f}")
    
    # Test alignment loss
    alignment = AlignmentLoss(temperature=0.07)
    
    skeleton_global = torch.randn(B, D)
    skeleton_part = torch.randn(B, P, D)
    text_global = torch.randn(B, D)
    text_part = torch.randn(B, P, D)
    labels = torch.tensor([0, 1, 2, 3])
    
    g_loss, p_loss = alignment(skeleton_global, skeleton_part, text_global, text_part, labels)
    print(f"\nGlobal alignment loss: {g_loss.item():.4f}")
    print(f"Primitive alignment loss: {p_loss.item():.4f}")
