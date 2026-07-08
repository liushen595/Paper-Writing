#!/usr/bin/env bash
# 犯罪意图识别框架 — 全流程脚本（按顺序执行各阶段）
# 用法: bash scripts/run_all.sh [stage]
#   stage: crawl | filter | haystack | synth | hardneg | pref | sft | dpo | implicit | blind | eval | judge | all
# 注意：训练脚本需要 conda activate ML 环境，且需先配置 .env
set -e

STAGE="${1:-all}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

run_crawl()   { python crawler/run.py; }
run_filter()  { python crawler/filter_criminal.py; }
run_haystack() { python -m scripts.prepare_haystack --n 5000; }
run_synth()    { python -m scripts.run_synthesis --provider glm --model glm-4-flash --append; }
run_hardneg()  { python -m scripts.run_hard_negatives; }
run_pref()     { python -m scripts.run_preference --judge glm --judge-model glm-4-flash; }
run_sft()      { python -m scripts.run_sft; }
run_dpo()      { python -m scripts.run_dpo; }
run_implicit() { python -m scripts.run_implicit_cot; }
run_blind()    { python -m scripts.run_blind_set; }
run_eval()     { python -m scripts.run_eval; }
run_judge()    {
  # 对 outputs/eval 下每个 predictions_<name>.json 跑 LLM-as-judge 质量评估
  local out_dir="outputs/eval"
  if [ ! -d "$out_dir" ]; then
    echo "[judge] outputs/eval 不存在，请先运行 eval"; exit 1
  fi
  for pred in "$out_dir"/predictions_*.json; do
    [ -f "$pred" ] || continue
    python -m scripts.run_judge_eval --predictions "$pred" --judge glm --judge-model glm-4-flash
  done
}

case "$STAGE" in
  crawl)    run_crawl ;;
  filter)   run_filter ;;
  haystack) run_haystack ;;
  synth)    run_synth ;;
  hardneg)  run_hardneg ;;
  pref)     run_pref ;;
  sft)      run_sft ;;
  dpo)      run_dpo ;;
  implicit) run_implicit ;;
  blind)    run_blind ;;
  eval)     run_eval ;;
  judge)    run_judge ;;
  all)
    run_crawl && run_filter && run_haystack && run_synth && run_hardneg && run_pref && run_sft && run_dpo && run_implicit && run_blind && run_eval && run_judge
    ;;
  *)
    echo "未知 stage: $STAGE"; exit 1 ;;
esac
