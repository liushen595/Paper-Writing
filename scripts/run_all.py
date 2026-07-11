"""犯罪意图识别框架 — 流水线驱动器。

替代旧版 run_all.sh，提供更灵活的阶段控制。

流水线分为三类阶段：
  - GPU 阶段（服务器 3090）：haystack, sft, gen_candidates, dpo, blind, eval
  - API 阶段（本地多线程，无需 GPU）：judge, judge_eval
  - 混合阶段：eval（可配合 --pre-generated 跳过部分 GPU baseline）

用法:
  # 服务器：跑 GPU 阶段
  python -m scripts.run_all --from sft --to eval           # sft → eval（服务器 GPU）
  python -m scripts.run_all --only sft                     # 只跑 sft
  python -m scripts.run_all --only gen_candidates --limit 3000  # 生成候选

  # 使用 HF 镜像站加速下载
  python -m scripts.run_all --only haystack --use-hf-mirror

  # 本地：跑 API 阶段（多线程，无需 GPU）
  python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10
  python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_sft-no-dpo.json --max-workers 10

阶段定义（按依赖顺序）:
  haystack        下载 WildChat-nontoxic 草垛（需 HF token + 网络）
  sft             Phase 1 监督微调（QLoRA + ToXCL 分类头）
  gen_candidates  Phase 2A 用 SFT 模型生成 DPO 候选（GPU，无 API）
  judge           Phase 2B 多线程 judge API 生成偏好对（本地 API）
  dpo             Phase 2 DPO 训练（依赖 sft checkpoint + 偏好对）
  blind           盲测集组装（本地，<1min）
  eval            盲测集评估（GPU baseline）
  judge_eval      LLM-as-judge 质量评估（本地 API 多线程）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.utils.env import PROJECT_ROOT, HF_MIRROR_ENDPOINT

# GPU 阶段（run_all 驱动 subprocess）
GPU_STAGES: list[str] = ["haystack", "sft", "gen_candidates", "dpo", "blind", "eval"]

API_STAGES: list[str] = ["judge", "judge_eval"]

# 全部阶段（文档展示用）
ALL_STAGES = GPU_STAGES + API_STAGES

# 每个阶段的命令模板
STAGE_COMMANDS: dict[str, list[str]] = {
    "haystack":       ["python", "-m", "scripts.prepare_haystack", "--n", "5000"],
    "sft":            ["python", "-m", "scripts.run_sft", "{limit}"],
    "gen_candidates": ["python", "-m", "scripts.generate_candidates", "{limit}"],
    "dpo":            ["python", "-m", "scripts.run_dpo"],
    "blind":          ["python", "-m", "scripts.run_blind_set"],
    "eval":           ["python", "-m", "scripts.run_eval", "{limit}"],
}

# 支持 --limit 的阶段
LIMIT_STAGES: set[str] = {"sft", "gen_candidates", "eval"}


def _fill_command(
    stage: str,
    limit: int | None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> list[str]:
    """填充阶段命令模板，去掉空占位符。"""
    template = STAGE_COMMANDS[stage]
    cmd: list[str] = []
    for part in template:
        if part == "{limit}":
            if limit and stage in LIMIT_STAGES:
                cmd.extend(["--limit", str(limit)])
            continue
        cmd.append(part)
    if stage in {"sft", "gen_candidates", "dpo", "eval"}:
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if stage in {"sft", "dpo"} and gradient_accumulation_steps is not None:
            cmd.extend(["--gradient-accumulation-steps", str(gradient_accumulation_steps)])
    return cmd


def _run(cmd: list[str], stage: str, use_hf_mirror: bool = False) -> None:
    """执行单条命令，失败即退出。"""
    env = os.environ.copy()
    if use_hf_mirror:
        env.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    print(f"\n{'='*60}")
    print(f"[{stage}] {' '.join(cmd)}")
    if use_hf_mirror:
        print(f"[{stage}] HF_ENDPOINT={env['HF_ENDPOINT']}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        print(f"[{stage}] 失败，返回码 {result.returncode}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="犯罪意图识别框架流水线驱动器（GPU 阶段）")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help=f"从指定阶段开始跑到结尾（GPU 阶段: {', '.join(GPU_STAGES)})")
    ap.add_argument("--to", dest="to_stage", default=None,
                    help=f"跑到指定阶段为止（含该阶段，GPU 阶段: {', '.join(GPU_STAGES)})")
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"只跑指定阶段（GPU 阶段: {', '.join(GPU_STAGES)})")
    ap.add_argument("--limit", type=int, default=None,
                    help="限制样本数（传给 sft/gen_candidates/eval）")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="覆盖 sft/gen_candidates/dpo/eval 的 batch size")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=None,
                    help="覆盖 sft/dpo 的梯度累积步数")
    ap.add_argument("--use-hf-mirror", action="store_true", default=False,
                    help="使用 HuggingFace 镜像站 hf-mirror.com 加速下载")
    args = ap.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        ap.error("--batch-size 必须大于等于 1")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps < 1:
        ap.error("--gradient-accumulation-steps 必须大于等于 1")

    # 确定要跑的阶段列表
    if args.only:
        chosen = args.only
    elif args.from_stage or args.to_stage:
        start = GPU_STAGES.index(args.from_stage) if args.from_stage else 0
        end = GPU_STAGES.index(args.to_stage) + 1 if args.to_stage else len(GPU_STAGES)
        chosen = GPU_STAGES[start:end]
    else:
        chosen = GPU_STAGES

    # 校验阶段名
    for s in chosen:
        if s not in GPU_STAGES:
            print(f"未知或非 GPU 阶段: {s}; GPU 阶段: {', '.join(GPU_STAGES)}")
            print(f"API 阶段（用 pre_generate.py）: {', '.join(API_STAGES)}")
            sys.exit(1)

    print(f"将执行 GPU 阶段: {' -> '.join(chosen)}")
    if args.limit:
        print(f"limit={args.limit}（传给支持的阶段）")
    if args.use_hf_mirror:
        print(f"使用 HF 镜像站: {HF_MIRROR_ENDPOINT}")
    if "judge" in chosen or "judge_eval" in chosen:
        print("\n注意: judge/judge_eval 是 API 阶段，请用:")
        print(f"  python -m scripts.pre_generate judge --input ... --max-workers 10")
        print(f"  python -m scripts.pre_generate judge_eval --input ... --max-workers 10")

    for stage in chosen:
        cmd = _fill_command(
            stage,
            limit=args.limit,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        _run(cmd, stage, use_hf_mirror=args.use_hf_mirror)

    print("\n所有指定 GPU 阶段执行完毕。")
    if "gen_candidates" in chosen:
        print("\n下一步（本地 API）: python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10")
    if "eval" in chosen:
        print("\n下一步（本地 API）: python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_<name>.json --max-workers 10")


if __name__ == "__main__":
    main()
