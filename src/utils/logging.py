"""日志工具：控制台 + 文件双输出，统一格式。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .env import PROJECT_ROOT

_LOGGERS: dict[str, logging.Logger] = {}
_ROOT_CONFIGURED = False


def setup_logger(name: str = "criminal_intent", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """配置并返回指定名称的 logger。多次调用同名 logger 不会重复添加 handler。"""
    global _ROOT_CONFIGURED
    logger = logging.getLogger(name)
    if name in _LOGGERS and not log_file:
        return logger
    logger.setLevel(level)
    if name not in _LOGGERS:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    _LOGGERS[name] = logger
    _ROOT_CONFIGURED = True
    return logger


def get_logger(name: str = "criminal_intent") -> logging.Logger:
    """获取 logger，若未配置过则自动添加 StreamHandler。"""
    if name not in _LOGGERS:
        setup_logger(name)
    return _LOGGERS[name]


def default_log_dir() -> Path:
    d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
