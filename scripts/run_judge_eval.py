## Copyright [2026] [Yijun Liu, Soochow University]
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

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
