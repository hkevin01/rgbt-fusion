from __future__ import annotations

import logging
from pathlib import Path


def build_logger(output_dir: str, name: str = "rgbt_fusion") -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(output_dir) / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
