"""犯罪意图识别框架 — 流水线驱动器。

替代旧版 run_all.sh，提供更灵活的阶段控制：

用法:
  python -m scripts.run_all                    # 从头跑完整流水线
  python -m scripts.run_all --from sft         # 从 sft 阶段开始跑到结尾
  python -m scripts.run_all --only sft         # 只跑 sft
  python -m scripts.run_all --only pref dpo    # 只跑 pref + dpo
  python -m scripts.run_all --from sft --to eval  # sft 到 eval（含两端）
  python -m scripts.run_all --limit 200        # 限制样本数（传给支持的阶段）
  python -m scripts.run_all --judge-model glm-4-flash  # 覆盖 judge 模型

阶段定义（按依赖顺序）:
  haystack  下载 WildChat-nontoxic 草垛（需 HF token + 网络）
  sft       Phase 1 监督微调（QLoRA + ToXCL 分类头）
  pref      Phase 2 偏好对生成（依赖 sft checkpoint + LLM judge API）
  dpo       Phase 2 DPO 训练（依赖 sft checkpoint + 偏好对）
  blind     盲测集组装（本地，<1min）
  eval      盲测集评估（依赖盲测集 + 4 个 baseline checkpoint）
  judge     LLM-as-judge 质量评估（依赖 eval 输出的 predictions）

已弃用阶段（数据已入仓，本机已跑完，不再纳入流水线）:
  crawl / filter / synth / hardneg / implicit（Phase 3 改为 future work）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.env import PROJECT_ROOT

# 阶段按依赖顺序排列
STAGES: list[str] = ["haystack", "sft", "pref", "dpo", "blind", "eval", "judge"]

# 每个阶段的命令模板（{limit} 仅在阶段支持 --limit 时替换）
STAGE_COMMANDS: dict[str, list[str]] = {
    "haystack": ["python", "-m", "scripts.prepare_haystack", "--n", "5000"],
    "sft":     ["python", "-m", "scripts.run_sft", "{limit}"],
    "pref":    ["python", "-m", "scripts.run_preference", "--judge", "glm", "--judge-model", "{judge_model}", "{limit}"],
    "dpo":     ["python", "-m", "scripts.run_dpo"],
    "blind":   ["python", "-m", "scripts.run_blind_set"],
    "eval":    ["python", "-m", "scripts.run_eval", "{limit}"],
    "judge":   [],  # 特殊处理：遍历 predictions_*.json，见 run_judge
}

# 支持 --limit 的阶段
LIMIT_STAGES: set[str] = {"sft", "pref", "eval"}


def _fill_command(stage: str, limit: int | None, judge_model: str) -> list[str]:
    """填充阶段命令模板，去掉空占位符。"""
    template = STAGE_COMMANDS[stage]
    if stage == "judge":
        return []  # 不走这里
    cmd: list[str] = []
    for part in template:
        if part == "{limit}":
            if limit and stage in LIMIT_STAGES:
                cmd.extend(["--limit", str(limit)])
            continue
        if part == "{judge_model}":
            cmd.append(judge_model)
            continue
        cmd.append(part)
    return cmd


def _run_judge(limit: int | None, judge_model: str) -> None:
    """对 outputs/eval 下每个 predictions_<name>.json 跑 LLM-as-judge。"""
    out_dir = PROJECT_ROOT / "outputs" / "eval"
    if not out_dir.exists():
        print(f"[judge] {out_dir} 不存在，请先运行 eval")
        sys.exit(1)
    pred_files = sorted(out_dir.glob("predictions_*.json"))
    if not pred_files:
        print(f"[judge] {out_dir} 下无 predictions_*.json，请先运行 eval")
        sys.exit(1)
    for pred in pred_files:
        cmd = ["python", "-m", "scripts.run_judge_eval",
               "--predictions", str(pred), "--judge", "aliyun", "--judge-model", judge_model]
        if limit:
            cmd.extend(["--limit", str(limit)])
        _run(cmd, f"judge({pred.name})")


def _run(cmd: list[str], stage: str) -> None:
    """执行单条命令，失败即退出。"""
    print(f"\n{'='*60}")
    print(f"[{stage}] {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[{stage}] 失败，返回码 {result.returncode}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="犯罪意图识别框架流水线驱动器")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help=f"从指定阶段开始跑到结尾（可选: {', '.join(STAGES)})")
    ap.add_argument("--to", dest="to_stage", default=None,
                    help=f"跑到指定阶段为止（含该阶段，可选: {', '.join(STAGES)})")
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"只跑指定阶段（可选: {', '.join(STAGES)})")
    ap.add_argument("--limit", type=int, default=None,
                    help="限制样本数，传给支持的阶段（sft/pref/eval/judge），用于 smoke test")
    ap.add_argument("--judge-model", default="qwen-plus",
                    help="LLM judge 模型名（默认 glm-4-flash）")
    args = ap.parse_args()

    # 确定要跑的阶段列表
    if args.only:
        chosen = args.only
    elif args.from_stage or args.to_stage:
        start = STAGES.index(args.from_stage) if args.from_stage else 0
        end = STAGES.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
        chosen = STAGES[start:end]
    else:
        chosen = STAGES

    # 校验阶段名
    for s in chosen:
        if s not in STAGES:
            print(f"未知阶段: {s}; 可选: {', '.join(STAGES)}")
            sys.exit(1)

    print(f"将执行阶段: {' -> '.join(chosen)}")
    if args.limit:
        print(f"limit={args.limit}（传给支持的阶段）")

    for stage in chosen:
        if stage == "judge":
            _run_judge(limit=args.limit, judge_model=args.judge_model)
        else:
            cmd = _fill_command(stage, limit=args.limit, judge_model=args.judge_model)
            _run(cmd, stage)

    print("\n所有指定阶段执行完毕。")


if __name__ == "__main__":
    main()
