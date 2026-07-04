#!/usr/bin/env bash
# 犯罪意图识别框架 — 全流程脚本（按顺序执行各阶段）
# 用法: bash scripts/run_all.sh [stage]
#   stage: synth | hardneg | pref | sft | dpo | implicit | blind | eval | all
# 注意：训练脚本需要 conda activate ML 环境，且需先配置 .env
set -e

STAGE="${1:-all}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

run_synth()    { python -m scripts.run_synthesis --provider glm --model glm-4-flash; }
run_hardneg()  { python -m scripts.run_hard_negatives; }
run_pref()     { python -m scripts.run_preference --judge glm --judge-model glm-4-flash; }
run_sft()      { python -m scripts.run_sft; }
run_dpo()      { python -m scripts.run_dpo; }
run_implicit() { python -m scripts.run_implicit_cot; }
run_blind()    { python -m scripts.run_blind_set; }
run_eval()     { python -m scripts.run_eval; }

case "$STAGE" in
  synth)    run_synth ;;
  hardneg)  run_hardneg ;;
  pref)     run_pref ;;
  sft)      run_sft ;;
  dpo)      run_dpo ;;
  implicit) run_implicit ;;
  blind)    run_blind ;;
  eval)     run_eval ;;
  all)
    run_synth && run_hardneg && run_pref && run_sft && run_dpo && run_implicit && run_blind && run_eval
    ;;
  *)
    echo "未知 stage: $STAGE"; exit 1 ;;
esac
