"""硬负样本组装入口。"""
from __future__ import annotations

import argparse

from src.data.hard_negatives import merge_hard_negatives
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="组装硬负样本（非犯罪背景 + 增强安全言论）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--augmented", default=None, help="可选的 LLM 增强硬负样本 jsonl 路径")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "hard_negatives.log")
    merge_hard_negatives(cfg.data, augmented_path=args.augmented)


if __name__ == "__main__":
    main()
