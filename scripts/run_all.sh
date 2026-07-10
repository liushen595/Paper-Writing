#!/usr/bin/env bash
# 犯罪意图识别框架 — 流水线驱动器（bash shim，实际逻辑在 run_all.py）
# 用法:
#   bash scripts/run_all.sh                              # 从头跑完整流水线
#   bash scripts/run_all.sh --from sft                  # 从 sft 阶段开始
#   bash scripts/run_all.sh --only sft                  # 只跑 sft
#   bash scripts/run_all.sh --only pref dpo             # 只跑 pref + dpo
#   bash scripts/run_all.sh --from sft --to eval        # sft 到 eval（含两端）
#   bash scripts/run_all.sh --limit 200 --from sft      # 限制样本数（smoke test）
#   bash scripts/run_all.sh --judge-model glm-4-flash   # 覆盖 judge 模型
#
# 阶段: haystack | sft | pref | dpo | blind | eval | judge
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 优先用 conda 环境里的 python，回退到系统 python
if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "ML"; then
  PYBIN="conda run -n ML python"
else
  PYBIN="python"
fi
exec $PYBIN -m scripts.run_all "$@"
