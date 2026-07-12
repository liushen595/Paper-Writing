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

"""硬负样本组装入口。"""
from __future__ import annotations

import argparse

from src.data.hard_negatives import merge_hard_negatives
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="组装硬负样本（非犯罪背景 + 增强安全言论）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--augmented", default=None, help="可选的 LLM 增强硬负样本 jsonl 路径")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "hard_negatives.log")
    merge_hard_negatives(cfg.data, augmented_path=args.augmented)


if __name__ == "__main__":
    main()
