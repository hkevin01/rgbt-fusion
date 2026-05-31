from __future__ import annotations

import torch
import torch.nn as nn


class MMTMBlock(nn.Module):
    """
    MMTM-style channel recalibration for RGB-T feature interaction.
    """

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        hidden = max(channels // ratio, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
        )
        self.rgb_excitation = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())
        self.thermal_excitation = nn.Sequential(nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor):
        b, c, _, _ = rgb_feat.shape
        rgb_vec = self.squeeze(rgb_feat).view(b, c)
        thermal_vec = self.squeeze(thermal_feat).view(b, c)
        joint = self.fc(torch.cat([rgb_vec, thermal_vec], dim=1))

        rgb_scale = self.rgb_excitation(joint).view(b, c, 1, 1)
        thermal_scale = self.thermal_excitation(joint).view(b, c, 1, 1)

        rgb_out = rgb_feat * rgb_scale
        thermal_out = thermal_feat * thermal_scale
        fused = rgb_out + thermal_out
        return fused, rgb_out, thermal_out
