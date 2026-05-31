from __future__ import annotations

from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_scheduler(optimizer: Optimizer, cfg: Dict[str, Any], epochs: int):
    name = cfg.get("name", "cosine").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg.get("min_lr", 1e-6)))
    if name == "poly":
        power = float(cfg.get("power", 0.9))
        min_lr = float(cfg.get("min_lr", 1e-6))
        base_lr = optimizer.param_groups[0]["lr"]
        min_factor = min_lr / max(base_lr, 1e-12)

        def poly(epoch: int) -> float:
            return max((1 - epoch / max(epochs, 1)) ** power, min_factor)

        return LambdaLR(optimizer, lr_lambda=poly)
    raise ValueError(f"Unknown scheduler: {name}")
