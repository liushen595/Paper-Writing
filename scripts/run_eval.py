"""评估主入口：在盲测集上跑所有 baseline，输出指标矩阵。"""
from __future__ import annotations

import argparse

from src.eval.run_eval import run_eval
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="盲测集评估")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baselines", nargs="*", default=None, help="覆盖配置中的 baselines 列表")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "eval.log")
    run_eval(cfg, baseline_names=args.baselines)


if __name__ == "__main__":
    main()
