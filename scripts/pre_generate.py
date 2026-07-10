"""本地多线程 API 预生成工具。

将耗时的串行 API 调用改为多线程并行，在本地机器（无需 GPU）上运行。
利用 qwen-plus 高限流（30k RPM / 5M TPM），10 线程可 10x 加速。

两个子任务（都是 judge 风格的 API 调用，不涉及模型推理）:
  judge       — 读取 SFT 候选 JSONL，多线程 judge API 生成 DPO 偏好对
  judge_eval  — 读取 predictions JSON，多线程 quality judge 生成评估结果

注意：qwen-zeroshot 不在这里，因为它必须用 Qwen3-8B 模型做 GPU 推理（不是 API 调用）。
用 qwen-plus API 替代是学术造假（不同模型、不同规模）。qwen-zeroshot 由 eval 阶段的
predict_batch 在 GPU 上批量完成。

用法:
  # 1. DPO 偏好对生成（读取服务器生成的 candidates.jsonl）
  python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10

  # 2. Judge 质量评估（读取 eval 输出的 predictions）
  python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_sft-no-dpo.json --max-workers 10

  # 通用参数
  --limit N            限制样本数
  --max-workers N      并行线程数（默认 10）
  --provider aliyun    LLM provider（默认 aliyun）
  --model qwen-plus    模型名（默认从 .env 读取）
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preference import judge_candidates_only
from src.eval.llm_judge import run_judge_eval
from src.utils.config import load_config


def _task_judge(args, cfg) -> None:
    """多线程 judge 生成 DPO 偏好对。"""
    judge_candidates_only(
        cfg.data, cfg.dpo,
        candidates_path=args.input,
        judge_provider=args.provider,
        judge_model=args.model,
        max_workers=args.max_workers,
        out_path=args.output,
    )


def _task_judge_eval(args, cfg) -> None:
    """多线程 judge 质量评估。"""
    out_path = args.output
    if out_path is None:
        stem = Path(args.input).stem
        out_path = Path(cfg.eval.output_dir) / f"judge_eval_{stem}.json"
    run_judge_eval(
        predictions_path=args.input,
        judge_provider=args.provider,
        judge_model=args.model,
        out_path=out_path,
        limit=args.limit,
        max_workers=args.max_workers,
    )


def main():
    ap = argparse.ArgumentParser(description="本地多线程 API 预生成工具")
    ap.add_argument("task", choices=["judge", "judge_eval"],
                    help="judge: DPO偏好对 | judge_eval: 质量评估")
    ap.add_argument("--input", required=True, help="输入文件路径")
    ap.add_argument("--output", default=None, help="输出路径（默认自动推断）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="限制样本数")
    ap.add_argument("--max-workers", type=int, default=10, help="并行线程数")
    ap.add_argument("--provider", default="aliyun", help="LLM provider（默认 aliyun）")
    ap.add_argument("--model", default=None, help="模型名（默认从 .env 读取）")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.task == "judge":
        _task_judge(args, cfg)
    elif args.task == "judge_eval":
        _task_judge_eval(args, cfg)


if __name__ == "__main__":
    main()
