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

"""Phase 3 Stepwise Internalization 训练入口。"""
from __future__ import annotations

import argparse

from src.training.implicit_cot import train_implicit_cot
from src.utils.config import load_config
from src.utils.env import setup_hf_mirror
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 3: 隐式 CoT 内化 (Stepwise Internalization)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 implicit_cot.per_device_batch_size")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=None,
                    help="覆盖 implicit_cot.gradient_accumulation_steps")
    ap.add_argument("--use-hf-mirror", action="store_true", default=None,
                    help="使用 HuggingFace 镜像站 hf-mirror.com 加速下载（覆盖配置文件）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            ap.error("--batch-size 必须大于等于 1")
        cfg.implicit_cot.per_device_batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        if args.gradient_accumulation_steps < 1:
            ap.error("--gradient-accumulation-steps 必须大于等于 1")
        cfg.implicit_cot.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.use_hf_mirror is not None:
        cfg.use_hf_mirror = args.use_hf_mirror
    setup_hf_mirror(cfg.use_hf_mirror)
    log = setup_logger(log_file=default_log_dir() / "implicit_cot.log")
    train_implicit_cot(cfg)


if __name__ == "__main__":
    main()
