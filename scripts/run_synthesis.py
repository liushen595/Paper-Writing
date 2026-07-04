"""Phase 0 造数入口：用 Teacher LLM 把 DOJ 犯罪叙事改写为隐式意图言论 + Explicit CoT。

用法:
    python -m scripts.run_synthesis --provider glm --model glm-4-flash --limit 200

注意：运行前请先复制 .env.example 为 .env 并填入 API_KEY 与 BASE_URL。
本脚本只造数，不做训练。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.synthesis import run_synthesis
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 0: 用 Teacher LLM 合成隐式意图训练数据")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--provider", default=None, help="glm/gemini/openai; 留空取 .env 中首个可用")
    ap.add_argument("--model", default=None, help="覆盖默认 teacher 模型名")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条 DOJ 记录（调试用）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有 train/test.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "synthesis.log")
    log.info("=== Phase 0 数据合成 ===")
    run_synthesis(
        data_cfg=cfg.data, provider_name=args.provider, model=args.model,
        limit=args.limit, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
