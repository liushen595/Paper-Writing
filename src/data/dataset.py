"""数据集与数据整理：合成正样本 + 硬负样本 -> 训练/验证集；prompt 模板；tokenization 协作。"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed

log = get_logger("dataset")


# --- Prompt 模板（Llama-3-Instruct 格式） ---
SYSTEM_PROMPT_SFT = (
    "You are a criminal-intent recognition assistant. Read the user's message and decide whether it "
    "expresses an implicit criminal intent. First reason step by step inside <thought>...</thought>, "
    "then output a category prefix [Category: X], then output the final label as either 'Threat' or 'Safe'."
)

INSTRUCTION_TEMPLATE = "Message: {text}\n\nAnalyze the intent."


@dataclass
class TrainExample:
    text: str
    thought_process: str
    label: str
    category: str
    probability: float

    def render_prompt(self) -> str:
        return INSTRUCTION_TEMPLATE.format(text=self.text)

    def render_target(self) -> str:
        return f"<thought>{self.thought_process}</thought>\n[Category: {self.category}]\n{self.label}"


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _from_synth(d: dict) -> TrainExample:
    return TrainExample(
        text=d.get("implicit_threat") or d.get("text", ""),
        thought_process=d.get("thought_process", ""),
        label=d.get("label", "Threat"),
        category=d.get("category", "Other"),
        probability=float(d.get("probability", 1.0 if d.get("label") == "Threat" else 0.0)),
    )


def _from_hard(d: dict) -> TrainExample:
    return TrainExample(
        text=d.get("text", ""),
        thought_process=d.get("thought_process", "[Reasoning] Safe context -> Safe."),
        label=d.get("label", "Safe"),
        category=d.get("category", "NonCriminal"),
        probability=float(d.get("probability", 0.0)),
    )


def build_train_examples(data_cfg: DataConfig, split: str = "train") -> list[TrainExample]:
    """合成正样本(train/test) + 硬负样本 -> TrainExample 列表。"""
    synth_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    synth = load_jsonl(synth_dir / f"{split}.jsonl")
    hard = load_jsonl(synth_dir / "hard_negatives.jsonl")
    examples = [_from_synth(d) for d in synth] + [_from_hard(d) for d in hard]
    set_seed(data_cfg.seed + (0 if split == "train" else 1))
    random.shuffle(examples)
    log.info(f"构建 {split} 集样本: synth={len(synth)}, hard={len(hard)}, total={len(examples)}")
    return examples


def label_to_id(label: str) -> int:
    return 1 if label.lower().startswith("threat") else 0


def id_to_label(i: int) -> str:
    return "Threat" if i == 1 else "Safe"


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
