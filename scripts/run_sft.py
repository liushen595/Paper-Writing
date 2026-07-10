"""Phase 1 SFT 训练入口。"""
from __future__ import annotations

import argparse

from src.training.sft import train_sft
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 1: SFT (QLoRA + ToXCL 分类头)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="限制训练+验证样本数（smoke test 用）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "sft.log")
    train_sft(cfg, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
