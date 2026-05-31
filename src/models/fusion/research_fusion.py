from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, channels: int, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.rgb_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.rgb_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.rgb_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.th_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.th_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.th_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.rgb_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.th_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(max(q.size(-1), 1))
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor):
        b, c, h, w = rgb_feat.shape
        n = h * w

        rgb_q = self.rgb_q(rgb_feat).reshape(b, c, n).transpose(1, 2)
        th_k = self.th_k(thermal_feat).reshape(b, c, n).transpose(1, 2)
        th_v = self.th_v(thermal_feat).reshape(b, c, n).transpose(1, 2)

        th_q = self.th_q(thermal_feat).reshape(b, c, n).transpose(1, 2)
        rgb_k = self.rgb_k(rgb_feat).reshape(b, c, n).transpose(1, 2)
        rgb_v = self.rgb_v(rgb_feat).reshape(b, c, n).transpose(1, 2)

        rgb_ctx = self._attention(rgb_q, th_k, th_v).transpose(1, 2).reshape(b, c, h, w)
        th_ctx = self._attention(th_q, rgb_k, rgb_v).transpose(1, 2).reshape(b, c, h, w)

        rgb_out = rgb_feat + self.rgb_proj(rgb_ctx)
        th_out = thermal_feat + self.th_proj(th_ctx)
        return rgb_out, th_out


class MMTMBlock(nn.Module):
    """
    MMTM-style channel recalibration for RGB-T feature interaction.
    """

    def __init__(self, channels: int, ratio: int = 4, attn_dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(channels // ratio, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
        )
        self.rgb_excitation = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())
        self.thermal_excitation = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())
        self.cross_attn = CrossModalAttentionBlock(channels, attn_dropout=attn_dropout)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor):
        b, c, _, _ = rgb_feat.shape
        rgb_vec = self.squeeze(rgb_feat).view(b, c)
        thermal_vec = self.squeeze(thermal_feat).view(b, c)
        joint = self.fc(torch.cat([rgb_vec, thermal_vec], dim=1))

        rgb_scale = self.rgb_excitation(joint).view(b, c, 1, 1)
        thermal_scale = self.thermal_excitation(joint).view(b, c, 1, 1)

        rgb_out = rgb_feat * rgb_scale
        thermal_out = thermal_feat * thermal_scale

        rgb_ctx, thermal_ctx = self.cross_attn(rgb_out, thermal_out)
        fused = self.fuse_conv(torch.cat([rgb_ctx, thermal_ctx], dim=1))
        return fused, rgb_out, thermal_out
