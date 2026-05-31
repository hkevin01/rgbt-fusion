from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.fusion import EarlyFusion, LateFusion, MMTMBlock, MidLevelFusion


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.head(feat)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class FusionModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        task_cfg = cfg["task"]
        model_cfg = cfg["model"]

        self.task_name = task_cfg["name"]
        self.num_classes = int(task_cfg["num_classes"])
        self.strategy = model_cfg["fusion_strategy"].lower()
        self.backbone_name = model_cfg["backbone"]
        self.pretrained = bool(model_cfg.get("pretrained", True))

        self.rgb_encoder = build_backbone(self.backbone_name, in_channels=3, pretrained=self.pretrained)
        self.thermal_encoder = build_backbone(self.backbone_name, in_channels=1, pretrained=self.pretrained)
        channels = self.rgb_encoder.out_channels

        if self.strategy == "early":
            self.early_fusion = EarlyFusion(3, 1)
            self.shared_encoder = build_backbone(self.backbone_name, in_channels=3, pretrained=self.pretrained)
            channels = self.shared_encoder.out_channels
        elif self.strategy == "mid":
            self.mid_fusion = MidLevelFusion(channels)
        elif self.strategy == "late":
            self.late_fusion = LateFusion(channels, self.num_classes)
        elif self.strategy == "research":
            self.research_fusion = MMTMBlock(channels)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

        self.classifier = nn.Linear(channels, self.num_classes)
        self.seg_head = SegmentationDecoder(channels, self.num_classes)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, return_features: bool = False):
        features: Dict[str, torch.Tensor] = {}

        if self.strategy == "early":
            x = self.early_fusion(rgb, thermal)
            out = self.shared_encoder(x)
            feat = out["feat"]
            pooled = out["pooled"]
            features["fused_feature"] = feat

        else:
            rgb_out = self.rgb_encoder(rgb)
            thermal_out = self.thermal_encoder(thermal)

            rgb_feat, thermal_feat = rgb_out["feat"], thermal_out["feat"]
            rgb_vec, thermal_vec = rgb_out["pooled"], thermal_out["pooled"]

            if self.strategy == "mid":
                feat = self.mid_fusion(rgb_feat, thermal_feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                features["rgb_feature"] = rgb_feat
                features["thermal_feature"] = thermal_feat
                features["fused_feature"] = feat
            elif self.strategy == "late":
                if self.task_name != "classification":
                    feat = 0.5 * (rgb_feat + thermal_feat)
                    pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                else:
                    logits = self.late_fusion(rgb_vec, thermal_vec)
                    if return_features:
                        features["rgb_feature"] = rgb_feat
                        features["thermal_feature"] = thermal_feat
                        return logits, features
                    return logits
            else:
                feat, rgb_cal, thermal_cal = self.research_fusion(rgb_feat, thermal_feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                features["rgb_feature"] = rgb_cal
                features["thermal_feature"] = thermal_cal
                features["fused_feature"] = feat

        if self.task_name == "classification":
            logits = self.classifier(pooled)
        elif self.task_name == "segmentation":
            logits = self.seg_head(feat, output_size=(rgb.shape[-2], rgb.shape[-1]))
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        if return_features:
            return logits, features
        return logits
