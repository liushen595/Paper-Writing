"""Phase 1 SFT 数据整理：把 TrainExample 转成 tokenized 的 input_ids / labels_clm / labels_cls。

Qwen3 ChatML template (apply_chat_template, enable_thinking=False 显式关闭):
  <|system|>SYSTEM_PROMPT_SFT<|end|>
  <|user|>INSTRUCTION<|end|>
  <|assistant|><thought>thought</thought>\n[Category: X]\nLabel<|end|>

CLM loss 仅对 assistant 段计算（prompt 段 label=-100），cls label 取 0/1。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from ..data.dataset import TrainExample, SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE, label_to_id
from ..models.student import load_tokenizer
from ..utils.logging import get_logger

log = get_logger("sft_dataset")


@dataclass
class CollatorOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels_clm: torch.Tensor
    labels_cls: torch.Tensor


def format_chat(example: TrainExample, tokenizer) -> tuple[str, str]:
    """返回 (full_text, assistant_text) 用于 tokenization。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SFT},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(text=example.text)},
        {"role": "assistant", "content": example.render_target()},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_SFT},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(text=example.text)},
    ]
    prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return full, prompt


class SFTDataset(Dataset):
    def __init__(self, examples: list[TrainExample], tokenizer, max_seq_len: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._cache: list[dict] = []
        self._build()

    def _build(self):
        for ex in tqdm(self.examples, desc="Building SFT dataset", unit="sample"):
            full, prompt = format_chat(ex, self.tokenizer)
            full_ids = self.tokenizer(full, truncation=True, max_length=self.max_seq_len, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_seq_len, add_special_tokens=False)["input_ids"]
            labels_clm = list(full_ids)
            # prompt 段置 -100
            for i in range(min(len(prompt_ids), len(labels_clm))):
                labels_clm[i] = -100
            # padding 在 collator 中做
            self._cache.append({
                "input_ids": full_ids,
                "labels_clm": labels_clm,
                "labels_cls": label_to_id(ex.label),
            })

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


class SFTCollator:
    def __init__(self, tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> CollatorOutput:
        max_len = min(self.max_seq_len, max(len(b["input_ids"]) for b in batch))
        input_ids, attn, labels_clm, labels_cls = [], [], [], []
        for b in batch:
            ids = b["input_ids"][:max_len]
            lab = b["labels_clm"][:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)
            labels_clm.append(lab + [-100] * pad_len)
            labels_cls.append(b["labels_cls"])
        return CollatorOutput(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attn, dtype=torch.long),
            labels_clm=torch.tensor(labels_clm, dtype=torch.long),
            labels_cls=torch.tensor(labels_cls, dtype=torch.long),
        )
