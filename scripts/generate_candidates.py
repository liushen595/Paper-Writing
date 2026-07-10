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
from src.utils.logging import setup_logger, default_log_dir


def main():
    ap = argparse.ArgumentParser(description="Phase A: SFT 候选生成（GPU，无 API）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="限制样本数")
    ap.add_argument("--out", default=None, help="输出路径（默认 data/preference/candidates.jsonl）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = setup_logger(log_file=default_log_dir() / "generate_candidates.log")
    generate_candidates_only(
        cfg.data, cfg.sft, cfg.sft.output_dir,
        limit=args.limit, out_path=args.out,
    )


if __name__ == "__main__":
    main()
