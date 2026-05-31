from __future__ import annotations

import torch
import torch.nn as nn


class MidLevelFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([rgb_feat, thermal_feat], dim=1)
        alpha = self.gate(concat)
        return alpha * rgb_feat + (1.0 - alpha) * thermal_feat
