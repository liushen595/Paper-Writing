"""Phase 2 DPO 偏好对生成入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preference import run_preference_generation
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 2: 用 LLM-as-Judge 生成 DPO 偏好对")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--judge", default="glm", help="裁判 provider")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--sft-ckpt", default=None,
        help="SFT checkpoint 目录；提供则用真实 SFT 模型采样候选，否则回退到占位生成器",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "preference.log")

    sft_ckpt = args.sft_ckpt or cfg.sft.output_dir
    if not Path(sft_ckpt).exists():
        log.warning(f"SFT checkpoint 不存在: {sft_ckpt}; 将使用占位候选生成器（仅用于流程验证）")
        sft_ckpt = None

    run_preference_generation(
        cfg.data, cfg.dpo, judge_provider=args.judge,
        judge_model=args.judge_model, limit=args.limit,
        sft_cfg=cfg.sft if sft_ckpt else None,
        sft_ckpt=sft_ckpt,
    )


if __name__ == "__main__":
    main()
