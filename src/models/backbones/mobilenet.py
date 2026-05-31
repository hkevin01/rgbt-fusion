from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)

        if in_channels != 3:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
            model.features[0][0] = new_conv

        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = 576

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        low = self.features[:4](x)
        feat = self.features[4:](low)
        pooled = self.pool(feat).flatten(1)
        return {"feat": feat, "low": low, "pooled": pooled}
