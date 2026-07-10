"""LLM-as-Judge 质量评估入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.llm_judge import run_judge_eval
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge 质量评估")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictions", required=True, help="predictions_<baseline>.json 路径")
    ap.add_argument("--judge", default="aliyun")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="限制评估样本数（smoke test 用）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "judge_eval.log")
    out = Path(cfg.eval.output_dir) / f"judge_eval_{Path(args.predictions).stem}.json"
    run_judge_eval(args.predictions, judge_provider=args.judge, judge_model=args.judge_model, out_path=out, limit=args.limit)


if __name__ == "__main__":
    main()
