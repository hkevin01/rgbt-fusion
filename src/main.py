from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.fusion_dataset import FusionDataset
from src.models.fusion_model import FusionModel
from src.training.scheduler import build_scheduler
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.utils.config import load_config, parse_args
from src.utils.logger import build_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_criterion(cfg):
    task_name = cfg["task"]["name"]
    if task_name == "classification":
        return nn.CrossEntropyLoss()
    if task_name == "segmentation":
        ignore_index = int(cfg["task"].get("ignore_index", 255))
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    raise ValueError(f"Unsupported task: {task_name}")


def maybe_wrap_multi_gpu(model, cfg):
    mgpu = cfg.get("multi_gpu", {})
    if not mgpu.get("enabled", False):
        return model

    if mgpu.get("distributed", False):
        # Basic project support: use torchrun externally and avoid DataParallel.
        return model

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def save_checkpoint(state, out_dir: str, is_best: bool = False) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    latest_path = Path(out_dir) / "checkpoint_latest.pt"
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, Path(out_dir) / "checkpoint_best.pt")


def load_checkpoint(model, optimizer, scaler, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_metric", -1.0))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    logger = build_logger(out_dir)
    tb_writer = SummaryWriter(log_dir=f"{out_dir}/tb") if cfg.get("logging", {}).get("tensorboard", False) else None

    set_seed(int(cfg["experiment"].get("seed", 42)))

    device_name = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    task_name = cfg["task"]["name"]
    image_size = tuple(cfg["dataset"].get("image_size", [256, 256]))

    train_cfg = dict(cfg["dataset"]["train"])
    val_cfg = dict(cfg["dataset"]["val"])
    train_cfg["image_size"] = image_size
    val_cfg["image_size"] = image_size

    train_ds = FusionDataset(train_cfg, task_name=task_name, train=True)
    val_ds = FusionDataset(val_cfg, task_name=task_name, train=False)

    num_workers = int(cfg["training"].get("num_workers", 4))
    pin_memory = bool(cfg["training"].get("pin_memory", True)) and device.type == "cuda"
    batch_size = int(cfg["training"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = FusionModel(cfg).to(device)
    model = maybe_wrap_multi_gpu(model, cfg)

    criterion = build_criterion(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["training"]["epochs"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), epochs)

    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    start_epoch = 1
    best_metric = -1.0

    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path:
        start_epoch, best_metric = load_checkpoint(model, optimizer, scaler, resume_path, device)
        logger.info(f"Resumed training from {resume_path} at epoch {start_epoch}")

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"Running on device: {device} | AMP: {amp_enabled}")

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
            logger=logger,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            epoch=epoch,
            logger=logger,
        )

        scheduler.step()

        primary_metric = val_stats.get("accuracy", val_stats.get("miou", -1.0))
        is_best = primary_metric > best_metric
        best_metric = max(best_metric, primary_metric)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "best_metric": best_metric,
            "config": cfg,
        }
        save_checkpoint(state, out_dir, is_best=is_best)

        if tb_writer is not None:
            for k, v in train_stats.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_stats.items():
                tb_writer.add_scalar(f"val/{k}", v, epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    if tb_writer is not None:
        tb_writer.close()

    logger.info(f"Training complete. Best metric={best_metric:.4f}")


if __name__ == "__main__":
    main()
