"""日志工具：控制台 + 文件双输出，统一格式。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .env import PROJECT_ROOT

_CONFIGURED = False


def setup_logger(name: str = "criminal_intent", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED and not log_file:
        return logger
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _CONFIGURED = True
    return logger


def get_logger(name: str = "criminal_intent") -> logging.Logger:
    return setup_logger(name)


def default_log_dir() -> Path:
    d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
