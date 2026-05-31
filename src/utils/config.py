from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("Invalid config format")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-T Fusion Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    return parser.parse_args()
