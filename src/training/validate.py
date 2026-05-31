from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from tqdm import tqdm

from src.utils.metrics import ClassificationMetrics, MeanIoU
from src.utils.visualization import save_fusion_visualization


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    logger,
):
    model.eval()
    task_name = cfg["task"]["name"]
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    cls_metrics = ClassificationMetrics() if task_name == "classification" else None
    miou_metrics = None
    if task_name == "segmentation":
        miou_metrics = MeanIoU(
            num_classes=int(cfg["task"]["num_classes"]),
            ignore_index=int(cfg["task"].get("ignore_index", 255)),
        )

    running_loss = 0.0

    progress = tqdm(loader, desc=f"val epoch {epoch}", leave=False)
    first_batch = None
    for batch in progress:
        rgb = batch["rgb"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits, feats = model(rgb, thermal, return_features=True)
            loss = criterion(logits, target)

        running_loss += float(loss.item())

        if cls_metrics is not None:
            cls_metrics.update(logits, target)
        if miou_metrics is not None:
            miou_metrics.update(logits, target)

        if first_batch is None:
            first_batch = (rgb[0], thermal[0], feats.get("fused_feature"), logits[0])

    result = {
        "loss": running_loss / max(len(loader), 1),
    }

    if cls_metrics is not None:
        result["accuracy"] = cls_metrics.compute()
    if miou_metrics is not None:
        result["miou"] = miou_metrics.compute()

    if first_batch is not None:
        rgb0, thermal0, fused0, pred0 = first_batch
        if task_name == "classification":
            pred0 = pred0.argmax(dim=0)
            fused_map = fused0 if fused0 is not None else None
        else:
            pred0 = pred0.argmax(dim=0)
            fused_map = fused0 if fused0 is not None else None

        save_fusion_visualization(
            rgb=rgb0,
            thermal=thermal0,
            fused_feature=fused_map,
            pred=pred0,
            output_path=f"{cfg['experiment']['output_dir']}/viz/epoch_{epoch:03d}.png",
        )

    logger.info(f"Val epoch {epoch}: {result}")
    return result
