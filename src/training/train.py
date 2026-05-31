from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from src.utils.metrics import ClassificationMetrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    logger,
):
    model.train()
    task_name = cfg["task"]["name"]
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    grad_clip_norm = cfg["training"].get("grad_clip_norm", None)

    running_loss = 0.0
    cls_metrics = ClassificationMetrics() if task_name == "classification" else None

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        rgb = batch["rgb"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(rgb, thermal)
            loss = criterion(logits, target)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.item())

        if cls_metrics is not None:
            cls_metrics.update(logits.detach(), target)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    result = {
        "loss": running_loss / max(len(loader), 1),
    }

    if cls_metrics is not None:
        result["accuracy"] = cls_metrics.compute()

    logger.info(f"Train epoch {epoch}: {result}")
    return result
