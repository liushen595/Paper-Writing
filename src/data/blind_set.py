"""盲测数据集组装：草垛(haystack) + 针(needles) -> test_blind.csv。

- Haystack: 合成 hard_negatives + (可选)下载的泛语料。
- Needles:  synthesis 的 test.jsonl 中的 implicit_threat（未参与训练）。
- 通过随机种子混合，记录 source 与 ground-truth。
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .dataset import load_jsonl

log = get_logger("blind_set")


def assemble_blind_set(
    data_cfg: DataConfig,
    extra_haystack_path: Optional[Path] = None,
    needles_per_label: Optional[int] = None,
) -> Path:
    set_seed(data_cfg.seed)
    blind_dir = (PROJECT_ROOT / data_cfg.blind_dir).resolve()
    blind_dir.mkdir(parents=True, exist_ok=True)
    out_csv = blind_dir / "test_blind.csv"

    needles = load_jsonl((PROJECT_ROOT / data_cfg.synthesized_dir / "test.jsonl").resolve())
    needle_records = []
    for n in needles:
        text = n.get("implicit_threat") or n.get("text", "")
        if not text:
            continue
        needle_records.append({
            "text": text,
            "label": n.get("label", "Threat"),
            "source": "needle",
            "category": n.get("category", ""),
            "ground_truth_cot": n.get("thought_process", ""),
        })

    haystack: list[dict] = []
    hard = load_jsonl((PROJECT_ROOT / data_cfg.synthesized_dir / "hard_negatives.jsonl").resolve())
    for h in hard:
        haystack.append({"text": h.get("text", ""), "label": h.get("label", "Safe"), "source": "haystack_hard", "category": h.get("category", ""), "ground_truth_cot": h.get("thought_process", "")})
    if extra_haystack_path and Path(extra_haystack_path).exists():
        for r in load_jsonl(extra_haystack_path):
            haystack.append({"text": r.get("text", ""), "label": "Safe", "source": "haystack_extra", "category": "", "ground_truth_cot": ""})

    if needles_per_label is not None:
        threat_needles = [r for r in needle_records if r["label"] == "Threat"]
        random.shuffle(threat_needles)
        needle_records = threat_needles[:needles_per_label]

    all_records = needle_records + haystack
    random.shuffle(all_records)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "category", "ground_truth_cot"])
        writer.writeheader()
        writer.writerows(all_records)
    log.info(f"盲测集组装完成: needles={len(needle_records)}, haystack={len(haystack)}, total={len(all_records)} -> {out_csv}")
    return out_csv
