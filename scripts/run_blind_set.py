"""盲测集组装入口。"""
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
