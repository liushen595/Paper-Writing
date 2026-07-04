"""Phase 2 DPO 训练入口。"""
from __future__ import annotations

import argparse

from src.training.dpo import train_dpo
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 2: DPO 偏好对齐")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "dpo.log")
    train_dpo(cfg)


if __name__ == "__main__":
    main()
