from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        if in_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if in_channels == 1:
                    model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    model.conv1.weight[:, :3].copy_(old_conv.weight)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        feat = self.layer4(x)
        pooled = self.pool(feat).flatten(1)
        return {"feat": feat, "low": low, "pooled": pooled}
