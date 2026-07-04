"""硬负样本构造与扩充。

数据来源：
1. synthesis.py 已为每个隐式正样本配对一条 hard_negative（语境安全）。
2. 本模块从 crawler/output/doj_non_criminal.jsonl 抽取安全背景语料。
3. 可选调用 Teacher LLM 对边界模糊的安全言论做"近犯罪"重写，构造更难的硬负样本。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from .doj_loader import DOJRecord, load_doj_records
from .llm_client import ChatMessage, BaseClient, build_client, safe_json_extract

log = get_logger("hard_negatives")


SYSTEM_PROMPT = """你是安全言论生成专家。给定一个犯罪主题，生成一条**话题相关、词汇相近但语境明显安全**的言论，用于训练模型降低误报率(FPR)。
安全语境包括：游戏讨论、影视剧情、学术研究、小说创作、假设场景、新闻报道引用。
不得包含真实犯罪意图。只输出 JSON，不要解释。"""

USER_TEMPLATE = """犯罪主题: {category}
相关案情摘要: {summary}

输出 JSON:
{{"text": "<安全言论>", "context": "<安全语境类型>", "label": "Safe"}}"""


def from_non_criminal(data_cfg: DataConfig, limit: Optional[int] = None) -> list[dict]:
    """从 doj_non_criminal.jsonl 提取非犯罪背景作为安全样本。"""
    path = (PROJECT_ROOT / data_cfg.raw_non_criminal).resolve()
    records = load_doj_records(path, limit=limit)
    out: list[dict] = []
    for r in records:
        text = r.summary or r.title
        if not text:
            continue
        out.append({
            "text": text,
            "thought_process": "[推理] 来自司法部非犯罪新闻稿 -> 民事/行政/政策事项 -> Safe。",
            "label": "Safe",
            "probability": 0.0,
            "category": "NonCriminal",
            "source_url": r.url,
        })
    log.info(f"从非犯罪新闻稿生成 {len(out)} 条安全样本")
    return out


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
    """合并所有硬负样本到一个文件，供 dataset.py 使用。"""
    out_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_negatives.jsonl"
    samples = from_non_criminal(data_cfg)
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
