"""
Aggregation Module
Composes part-level features into a global representation using attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveAggregation(nn.Module):
    """
    Aggregates part-level (primitive) features into a compositional global representation
    using attention-weighted aggregation
    """
    
    def __init__(self, feature_dim: int = 256, num_parts: int = 6):
        """
        Args:
            feature_dim: Feature dimension
            num_parts: Number of body parts (primitives)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        
        # Attention score computation
        self.W = nn.Linear(feature_dim, feature_dim)
        self.w = nn.Linear(feature_dim, 1)
        
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


class PrimitiveIndependenceRegularizer(nn.Module):
    """
    Enforces independence between different primitive (part) features
    Penalizes off-diagonal correlations between primitive features
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        """
        Compute independence regularization loss
        
        Args:
            part_features: Part-level features (B, P, D)
            
        Returns:
            Independence loss scalar
        """
        B, P, D = part_features.shape
        
        # Mean-center each primitive feature across batch
        mean = part_features.mean(dim=0, keepdim=True)  # (1, P, D)
        centered = part_features - mean  # (B, P, D)
        
        # Normalize
        norms = centered.norm(dim=-1, keepdim=True)  # (B, P, 1)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-8)
        normalized = centered / norms  # (B, P, D)
        
        # Compute pairwise similarities (off-diagonal correlations)
        # Similarity matrix: (P, P)
        similarity = torch.mean(
            torch.bmm(normalized, normalized.transpose(-2, -1)),
            dim=0
        )  # (P, P)
        
        # Penalize off-diagonal elements
        # Create mask to exclude diagonal
        mask = 1.0 - torch.eye(P, device=part_features.device)
        
        loss = (similarity ** 2) * mask
        loss = loss.sum() / (P * (P - 1))  # Average over off-diagonal
        
        return loss


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
        # Cosine similarity: 1 - cosine_similarity
        similarity = F.cosine_similarity(compositional_global, shiftgcn_global, dim=-1)
        
        loss = 1.0 - similarity.mean()
        
        return loss


class AlignmentLoss(nn.Module):
    """
    Cross-modal alignment losses (InfoNCE-style contrastive loss)
    for both global and primitive-level alignment
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.temperature = temperature
    
    def primitive_level_alignment(
        self, 
        skeleton_features: torch.Tensor, 
        text_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Primitive-level alignment loss
        
        Args:
            skeleton_features: Part-level skeleton features (B, P, D)
            text_features: Part-level text features (B, P, D)
            labels: Class labels (B,)
            
        Returns:
            Alignment loss
        """
        B, P, D = skeleton_features.shape
        
        # Reshape for contrastive loss
        skeleton_flat = skeleton_features.view(B * P, D)  # (B*P, D)
        text_flat = text_features.view(B * P, D)  # (B*P, D)
        
        # Labels for each primitive (repeated P times)
        labels_expanded = labels.repeat(P)  # (B*P,)
        
        return self._contrastive_loss(skeleton_flat, text_flat, labels_expanded, B * P)
    
    def global_level_alignment(
        self,
        skeleton_global: torch.Tensor,
        text_global: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Global-level alignment loss
        
        Args:
            skeleton_global: Global skeleton features (B, D)
            text_global: Global text features (B, D)
            labels: Class labels (B,)
            
        Returns:
            Alignment loss
        """
        return self._contrastive_loss(skeleton_global, text_global, labels, labels.shape[0])
    
    def _contrastive_loss(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor, 
        labels: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss
        
        Args:
            features1: First modality features (N, D)
            features2: Second modality features (N, D)
            labels: Ground truth labels (N,)
            batch_size: Batch size
            
        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        similarity = torch.matmul(features1, features2.T) / self.temperature  # (N, N)
        
        # Create positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()  # (N, N)
        
        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Mean log probability for positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
        
        # Loss
        loss = -mean_log_prob
        
        return loss.mean()
    
    def forward(
        self,
        skeleton_global: torch.Tensor,
        skeleton_part: torch.Tensor,
        text_global: torch.Tensor,
        text_part: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Compute both global and primitive alignment losses
        
        Returns:
            Tuple of (global_loss, primitive_loss)
        """
        global_loss = self.global_level_alignment(skeleton_global, text_global, labels)
        primitive_loss = self.primitive_level_alignment(skeleton_part, text_part, labels)
        
        return global_loss, primitive_loss


if __name__ == "__main__":
    # Test aggregation module
    aggregation = PrimitiveAggregation(feature_dim=256, num_parts=6)
    
    # Dummy input
    B, P, D = 4, 6, 256
    part_features = torch.randn(B, P, D)
    
    global_feat, attention = aggregation(part_features)
    
    print(f"Part features shape: {part_features.shape}")
    print(f"Global features shape: {global_feat.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention sum (should be 1): {attention.sum(dim=-1)}")
    
    # Test independence regularizer
    independence = PrimitiveIndependenceRegularizer()
    ind_loss = independence(part_features)
    print(f"\nIndependence loss: {ind_loss.item():.4f}")
    
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
