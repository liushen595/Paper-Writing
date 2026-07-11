"""评估主入口：在盲测集上跑指定 baseline，计算指标矩阵，输出报告。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.run_eval import run_eval
from src.utils.config import load_config
from src.utils.env import setup_hf_mirror
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="盲测集评估")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baselines", nargs="*", default=None, help="覆盖配置中的 baselines 列表")
    ap.add_argument("--limit", type=int, default=None, help="限制盲测样本数（smoke test 用）")
    ap.add_argument("--pre-generated", nargs="*", default=None,
                    help="使用预生成预测，跳过 GPU 推理。格式: name=path（如 sft-no-dpo=outputs/eval/predictions_sft-no-dpo.json）。可指定多个。")
    ap.add_argument("--use-hf-mirror", action="store_true", default=None,
                    help="使用 HuggingFace 镜像站 hf-mirror.com 加速下载（覆盖配置文件）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.use_hf_mirror is not None:
        cfg.use_hf_mirror = args.use_hf_mirror
    setup_hf_mirror(cfg.use_hf_mirror)
    log = setup_logger(log_file=default_log_dir() / "eval.log")

    pre_gen = {}
    if args.pre_generated:
        for item in args.pre_generated:
            name, path = item.split("=", 1)
            pre_gen[name] = path
        log.info(f"预生成预测: {pre_gen}")

    run_eval(cfg, baseline_names=args.baselines, limit=args.limit, pre_generated=pre_gen)


if __name__ == "__main__":
    main()
