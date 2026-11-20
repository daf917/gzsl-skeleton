"""Frozen global Shift-GCN features and part-conditioned motion encoding."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .shift_gcn import ShiftGCN


class LocalSkeletonEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512, num_parts=6, num_frames=64, dropout=0.5):
        super().__init__()
        self.part_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_parts)
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.part_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, motion_attributes, global_feature):
        local = torch.stack([
            encoder(motion_attributes[:, :, part]).mean(dim=1)
            for part, encoder in enumerate(self.part_encoders)
        ], dim=1)
        global_token = self.global_norm(global_feature).unsqueeze(1)
        guidance, _ = self.part_attention(local, global_token, global_token, need_weights=False)
        return F.normalize(self.output_proj(self.output_norm(local + guidance)), dim=-1)


class SkeletonEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 25,
        num_classes: int = 60,
        feature_dim: int = 512,
        num_parts: int = 6,
        dropout: float = 0.5,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.shift_gcn = ShiftGCN(num_joints)
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            self.shift_gcn.load_state_dict(checkpoint['backbone_state_dict'])
        self.freeze_shift_gcn()
        self.global_proj = nn.Sequential(
            nn.Linear(ShiftGCN.output_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim),
        )
        self.local_encoder = LocalSkeletonEncoder(
            hidden_dim=feature_dim, num_parts=num_parts, dropout=dropout,
        )

    def forward(
        self, skeleton: torch.Tensor, motion_attributes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global_feature = F.normalize(self.global_proj(self.shift_gcn(skeleton)), dim=-1)
        return global_feature, self.local_encoder(motion_attributes, global_feature)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.shift_gcn.parameters()):
            self.shift_gcn.eval()
        return self

    def freeze_shift_gcn(self):
        self.shift_gcn.requires_grad_(False).eval()

    def unfreeze_shift_gcn(self):
        self.shift_gcn.requires_grad_(True).train(self.training)
