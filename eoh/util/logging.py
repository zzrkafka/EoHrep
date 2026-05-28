"""'eoh' named logger, mirroring upstream utils/logger.py.

File + stdout handler, `[%(asctime)s] %(message)s` format, INFO/DEBUG toggle.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(log_path: str | Path, debug: bool = False) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eoh")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
