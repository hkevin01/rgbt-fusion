from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    def __init__(self, in_channels_rgb: int = 3, in_channels_thermal: int = 1) -> None:
        super().__init__()
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels_rgb + in_channels_thermal, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, thermal], dim=1)
        return self.input_adapter(x)
