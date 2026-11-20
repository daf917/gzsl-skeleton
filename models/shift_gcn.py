"""Shift-GCN feature backbone for two-dimensional joint sequences."""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialShift(nn.Module):
    """Mix joints by shifting each channel before and after a linear transform."""

    def __init__(self, num_joints, in_channels, out_channels):
        super().__init__()
        joints = torch.arange(num_joints).unsqueeze(-1)
        incoming = torch.arange(in_channels).unsqueeze(0)
        outgoing = torch.arange(out_channels).unsqueeze(0)
        self.register_buffer(
            'input_indices', (((joints + incoming) % num_joints) * in_channels + incoming).flatten(),
            persistent=False,
        )
        self.register_buffer(
            'output_indices', (((joints - outgoing) % num_joints) * out_channels + outgoing).flatten(),
            persistent=False,
        )
        self.mask = nn.Parameter(torch.zeros(num_joints, in_channels))
        self.mix = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(num_joints * out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        )
        self.num_joints = num_joints

    def forward(self, features):
        batch, _, frames, _ = features.shape
        tokens = features.permute(0, 2, 3, 1).flatten(0, 1).flatten(1)
        tokens = tokens.index_select(1, self.input_indices).reshape(batch * frames, self.num_joints, -1)
        tokens = self.mix(tokens * (1 + self.mask.tanh())).flatten(1)
        tokens = self.norm(tokens.index_select(1, self.output_indices))
        shifted = tokens.reshape(batch, frames, self.num_joints, -1).permute(0, 3, 1, 2)
        return F.relu(shifted + self.skip(features))


class LearnableShift(nn.Module):
    """Sample channel-specific offsets on a time–joint grid with bilinear interpolation."""

    def __init__(self, channels, stride=1):
        super().__init__()
        self.time_offset = nn.Parameter(torch.empty(channels).uniform_(-1, 1))
        self.joint_offset = nn.Parameter(torch.zeros(channels))
        self.stride = stride

    def forward(self, features):
        _, _, frames, joints = features.shape
        time = torch.arange(0, frames, self.stride, device=features.device, dtype=features.dtype)
        joint = torch.arange(joints, device=features.device, dtype=features.dtype)
        time = time.unsqueeze(0) + self.time_offset.unsqueeze(-1)
        if self.stride > 1:
            time = time + (self.stride - 1) / 2
        joint = joint.unsqueeze(0) + self.joint_offset.unsqueeze(-1)
        y = (2 * (time + 0.5) / frames - 1).unsqueeze(-1).expand(-1, -1, joints)
        x = (2 * (joint + 0.5) / joints - 1).unsqueeze(1).expand_as(y)
        grid = torch.stack((x, y), dim=-1)
        shifted = F.grid_sample(features.transpose(0, 1), grid, align_corners=False)
        return shifted.transpose(0, 1)


class ShiftBlock(nn.Module):
    def __init__(self, joints, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.spatial = SpatialShift(joints, in_channels, out_channels)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            LearnableShift(out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(),
            LearnableShift(out_channels, stride),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features):
        output = self.temporal(self.spatial(features))
        if self.residual is not None:
            output = output + self.residual(features)
        return F.relu(output)


class ShiftGCN(nn.Module):
    """Spatial and temporal shift blocks with the 64/128/256 Shift-GCN stage widths."""

    output_dim = 256

    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.input_norm = nn.BatchNorm1d(2 * num_joints)
        widths = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
        self.blocks = nn.Sequential(*(
            ShiftBlock(num_joints, channels_in, channels_out, 2 if i in (4, 7) else 1, i != 0)
            for i, (channels_in, channels_out) in enumerate(zip((2, *widths[:-1]), widths))
        ))

    def forward(self, skeleton, *, return_temporal_tokens=False):
        """Return the pooled feature and optionally the joint-pooled temporal tokens."""
        if skeleton.ndim != 4 or skeleton.shape[-2:] != (self.num_joints, 2):
            raise ValueError(f'Expected skeletons shaped (batch, time, {self.num_joints}, 2).')
        batch, frames, joints, channels = skeleton.shape
        features = self.input_norm(skeleton.reshape(batch, frames, -1).transpose(1, 2))
        features = features.reshape(batch, joints, channels, frames).permute(0, 2, 3, 1)
        feature_map = self.blocks(features)
        pooled = feature_map.mean(dim=(2, 3))
        if return_temporal_tokens:
            temporal_tokens = feature_map.mean(dim=3).transpose(1, 2)
            return pooled, temporal_tokens
        return pooled
