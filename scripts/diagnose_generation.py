## Copyright [2026] [Yijun Liu, Soochow University]
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

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