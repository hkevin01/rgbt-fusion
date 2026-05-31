from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
    return arr


def save_fusion_visualization(
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    fused_feature: Optional[torch.Tensor],
    pred: Optional[torch.Tensor],
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rgb_np = _to_numpy_img(rgb)
    thermal_np = _to_numpy_img(thermal)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.title("RGB")
    plt.imshow(rgb_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Thermal")
    if thermal_np.ndim == 3 and thermal_np.shape[-1] == 1:
        thermal_np = thermal_np[..., 0]
    plt.imshow(thermal_np, cmap="inferno")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Fused")
    if fused_feature is not None:
        fmap = fused_feature.detach().cpu().mean(dim=0)
        fmap = (fmap - fmap.min()) / max(fmap.max() - fmap.min(), 1e-8)
        plt.imshow(fmap.numpy(), cmap="viridis")
    else:
        plt.text(0.2, 0.5, "N/A")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Prediction")
    if pred is not None:
        if pred.ndim == 2:
            plt.imshow(pred.detach().cpu().numpy(), cmap="tab20")
        else:
            plt.text(0.2, 0.5, str(int(pred.item())))
    else:
        plt.text(0.2, 0.5, "N/A")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
