"""Phase 2 DPO 训练入口。"""
from __future__ import annotations

import argparse

from src.training.dpo import train_dpo
from src.utils.config import load_config
from src.utils.env import setup_hf_mirror
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 2: DPO 偏好对齐")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--use-hf-mirror", action="store_true", default=None,
                    help="使用 HuggingFace 镜像站 hf-mirror.com 加速下载（覆盖配置文件）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.use_hf_mirror is not None:
        cfg.use_hf_mirror = args.use_hf_mirror
    setup_hf_mirror(cfg.use_hf_mirror)
    log = setup_logger(log_file=default_log_dir() / "dpo.log")
    train_dpo(cfg)


if __name__ == "__main__":
    main()
