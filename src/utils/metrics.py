from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationMetrics:
    correct: int = 0
    total: int = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        self.correct += int((pred == target).sum().item())
        self.total += int(target.numel())

    def compute(self) -> float:
        return float(self.correct / max(self.total, 1))


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if pred.numel() == 0:
            return
        idx = target * self.num_classes + pred
        bins = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.confmat += bins.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> float:
        inter = torch.diag(self.confmat).float()
        union = self.confmat.sum(dim=1).float() + self.confmat.sum(dim=0).float() - inter
        iou = inter / torch.clamp(union, min=1.0)
        return float(iou.mean().item())
