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


def from_synth_internal(data_cfg: DataConfig) -> list[dict]:
    """从已合成的 train/test.jsonl 中抽取 hard_negative 字段，展开为独立的 Safe 样本。

    synthesis.py 的输出 schema 把 implicit_threat 与 hard_negative 放在同一记录里，
    label 只标 Threat；下游 dataset.py 需要独立的 Safe 样本，否则分类头学不到 Safe 类。
    本函数把每条记录的 hard_negative 字段拆成一条 {text, thought_process, label:Safe, ...}。
    """
    synth_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    samples: list[dict] = []
    for split in ("train", "test"):
        path = synth_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text = d.get("hard_negative") or d.get("text_safe", "")
                if not text:
                    continue
                prob = d.get("probability")
                if isinstance(prob, (int, float)):
                    safe_prob = max(0.0, min(0.3, 1.0 - float(prob)))
                else:
                    safe_prob = 0.05
                samples.append({
                    "text": text,
                    "thought_process": d.get("hard_negative_thought_process")
                        or "[Reasoning] topically similar but safe context -> no criminal intent -> Safe.",
                    "label": "Safe",
                    "probability": safe_prob,
                    "category": d.get("category", "NonCriminal"),
                    "source_url": d.get("source_url", ""),
                    "split_origin": split,
                })
    log.info(f"从 synth 内部展开 hard_negative: {len(samples)} 条 Safe 样本")
    return samples


def merge_hard_negatives(data_cfg: DataConfig, augmented_path: Optional[Path] = None) -> Path:
    """合并所有硬负样本到一个文件，供 dataset.py 使用。

    顺序：
      1. 无条件从 train/test.jsonl 抽取 hard_negative 字段（核心来源，避免硬负样本数据丢失）。
      2. 可选合并 LLM 增强的额外硬负样本（augmented_path，例如边界模糊的安全言论重写）。
    """
    out_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_negatives.jsonl"
    samples: list[dict] = from_synth_internal(data_cfg)
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
