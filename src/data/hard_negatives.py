"""硬负样本构造与扩充。

数据来源：
1. synthesis.py 已为每个隐式正样本配对一条 hard_negative（语境安全）。
2. 可选调用 Teacher LLM 对边界模糊的安全言论做"近犯罪"重写，构造更难的硬负样本。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from .llm_client import ChatMessage, BaseClient, build_client, safe_json_extract

log = get_logger("hard_negatives")


SYSTEM_PROMPT = """You are a safe-utterance generation expert. Given a crime theme, generate a **topically similar but obviously context-safe** utterance, used to train a model to reduce false positive rate (FPR).
Safe contexts include: gaming discussion, movie/TV plot, academic research, fiction writing, hypothetical scenario, news reporting.
Must NOT contain real criminal intent. Output ONLY JSON, no explanation. All content in English."""

USER_TEMPLATE = """Crime theme: {category}
Related case summary: {summary}

Output JSON (all content in English):
{{"text": "<safe utterance>", "context": "<safe context type>", "label": "Safe"}}"""


def llm_augment(client: BaseClient, category: str, summary: str, n: int = 1) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        msgs = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", USER_TEMPLATE.format(category=category, summary=summary[:300])),
        ]
        try:
            raw = client.chat(msgs, temperature=0.9, max_tokens=256)
            obj = safe_json_extract(raw)
            if obj.get("label") == "Safe" and "text" in obj:
                out.append({
                    "text": obj["text"],
                    "thought_process": f"[推理] 语境为{obj.get('context','安全')} -> 无犯罪意图 -> Safe。",
                    "label": "Safe",
                    "probability": 0.0,
                    "category": category,
                })
        except Exception as e:  # noqa: BLE001
            log.warning(f"硬负样本 LLM 增强失败: {e}")
    return out


def merge_hard_negatives(data_cfg: DataConfig, augmented_path: Optional[Path] = None) -> Path:
    """合并所有硬负样本到一个文件（hard_negatives 来自 synthesis，可选 LLM 增强），供 dataset.py 使用。"""
    out_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_negatives.jsonl"
    samples: list[dict] = []
    if augmented_path and Path(augmented_path).exists():
        with open(augmented_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"硬负样本合并完成: {len(samples)} 条 -> {out_path}")
    return out_path
