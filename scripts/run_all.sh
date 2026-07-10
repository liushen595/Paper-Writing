#!/usr/bin/env bash
# 犯罪意图识别框架 — 流水线驱动器（bash shim，实际逻辑在 run_all.py）
#
# GPU 阶段（服务器 3090）:
#   bash scripts/run_all.sh                                    # 全部 GPU 阶段
#   bash scripts/run_all.sh --from sft --to eval              # sft 到 eval
#   bash scripts/run_all.sh --only gen_candidates --limit 3000
#
# API 阶段（本地多线程，无需 GPU）:
#   python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10
#   python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_<name>.json --max-workers 10
# qwen-zeroshot 在 eval 阶段 GPU 上批量完成
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "ML"; then
  PYBIN="conda run -n ML python"
else
  PYBIN="python"
fi
exec $PYBIN -m scripts.run_all "$@"
