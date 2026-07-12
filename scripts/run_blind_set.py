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

"""盲测集组装入口。

草垛来源（优先级）:
  1. cfg.data.haystack_path（如 data/haystack/wildchat_nontoxic.jsonl）— 主路径
  2. --extra-haystack CLI 参数 — 向后兼容额外补充
  3. data/synthesized/hard_negatives.jsonl — 总是自动包含
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.blind_set import assemble_blind_set
from src.utils.config import load_config
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="组装盲测集 (haystack + needles)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--extra-haystack", default=None, help="额外 haystack jsonl 路径")
    ap.add_argument("--needles-per-label", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "blind_set.log")
    assemble_blind_set(
        cfg.data,
        extra_haystack_path=Path(args.extra_haystack) if args.extra_haystack else None,
        needles_per_label=args.needles_per_label,
    )


if __name__ == "__main__":
    main()
