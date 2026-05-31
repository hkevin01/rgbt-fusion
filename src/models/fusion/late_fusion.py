from __future__ import annotations

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.rgb_head = nn.Linear(in_dim, num_classes)
        self.thermal_head = nn.Linear(in_dim, num_classes)
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, rgb_vec: torch.Tensor, thermal_vec: torch.Tensor) -> torch.Tensor:
        rgb_logits = self.rgb_head(rgb_vec)
        thermal_logits = self.thermal_head(thermal_vec)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * rgb_logits + w[1] * thermal_logits
