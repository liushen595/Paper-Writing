"""从已有预测中诊断生成标签，不运行模型。"""
from __future__ import annotations

import argparse
import json

from src.eval.generation_diagnostics import diagnose_files


def main() -> None:
    parser = argparse.ArgumentParser(description="严格解析已有 CoT 的最终标签并生成诊断报告")
    parser.add_argument("inputs", nargs="+", help="predictions_*.json 文件")
    parser.add_argument("--output", default="outputs/eval/generation_diagnostics.json")
    args = parser.parse_args()
    report = diagnose_files(args.inputs, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()