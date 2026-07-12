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

"""Phase A（服务器 GPU）：用 SFT 模型批量生成 DPO 候选回复。

不调用任何 API，纯 GPU 推理。输出 candidates.jsonl，供本地多线程 judge 使用。

用法:
  python -m scripts.generate_candidates --limit 3000
  python -m scripts.generate_candidates --limit 3000 --out data/preference/candidates.jsonl
"""
from __future__ import annotations

import argparse

from src.data.preference import generate_candidates_only
from src.utils.config import load_config
from src.utils.env import setup_hf_mirror
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase A: SFT 候选生成（GPU，无 API）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=3000, help="限制样本数")
    ap.add_argument("--out", default=None, help="输出路径（默认 data/preference/candidates.jsonl）")
    ap.add_argument("--batch-size", type=int, default=1, help="GPU 批量生成大小")
    ap.add_argument("--use-hf-mirror", action="store_true", default=None,
                    help="使用 HuggingFace 镜像站 hf-mirror.com 加速下载（覆盖配置文件）")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size 必须大于等于 1")
    cfg = load_config(args.config)
    if args.use_hf_mirror is not None:
        cfg.use_hf_mirror = args.use_hf_mirror
    setup_hf_mirror(cfg.use_hf_mirror)
    log = setup_logger(log_file=default_log_dir() / "generate_candidates.log")
    generate_candidates_only(
        cfg.data, cfg.sft, cfg.sft.output_dir,
        limit=args.limit, out_path=args.out, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
