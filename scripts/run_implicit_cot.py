"""Phase 3 Stepwise Internalization 训练入口。"""
from __future__ import annotations

import argparse

from src.training.implicit_cot import train_implicit_cot
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 3: 隐式 CoT 内化 (Stepwise Internalization)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "implicit_cot.log")
    train_implicit_cot(cfg)


if __name__ == "__main__":
    main()
