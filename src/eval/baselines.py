"""评估基线：toxic-bert / sft-no-dpo / dpo-only。

每个 baseline 暴露 predict(text) -> (label, prob, cot?) 的统一接口。
支持批量推理（predict_batch）加速评估。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..utils.logging import get_logger
from .metrics import label_to_int, threshold_predictions

log = get_logger("baselines")


@dataclass
class Prediction:
    label: str
    prob: float
    cot: Optional[str] = None
    latency_ms: float = 0.0
    tokens: int = 0


class Baseline:
    name: str = "base"

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]


class ToxicBertBaseline(Baseline):
    """unitary/toxic-bert 判别式基线。"""

    name = "toxic-bert"

    def __init__(self, model_name: str = "unitary/toxic-bert"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text: str) -> Prediction:
        import time, torch
        t0 = time.perf_counter()
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        ms = (time.perf_counter() - t0) * 1000
        return Prediction(label="Threat" if prob > 0.5 else "Safe", prob=prob, latency_ms=ms)


class StudentBaseline(Baseline):
    """通用：加载某个 StudentModel checkpoint 做推理（用于 sft-no-dpo / dpo-only 对比）。"""

    def __init__(self, name: str, ckpt_dir: str | Path, sft_cfg, conditional_decoding: bool = True):
        from ..models.student import StudentModel, load_tokenizer
        import torch
        self.name = name
        self.ckpt_dir = Path(ckpt_dir)
        self.sft_cfg = sft_cfg
        self.tokenizer = load_tokenizer(sft_cfg.base_model)
        self.model = StudentModel.load(sft_cfg, ckpt_dir)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # Qwen3 的 generation_config.json 默认含 temperature/top_p/top_k，
        # do_sample=False 时会触发 "not valid and may be ignored" 警告。清除之。
        gen_cfg = getattr(self.model.base, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.temperature = None
            gen_cfg.top_p = None
            gen_cfg.top_k = None
        self.conditional_decoding = conditional_decoding
        from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE
        self.system = SYSTEM_PROMPT_SFT
        self.instr_tpl = INSTRUCTION_TEMPLATE

    def predict(self, text: str) -> Prediction:
        import time, torch
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.instr_tpl.format(text=text)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        n_prompt = inputs["input_ids"].size(1)
        with torch.no_grad():
            cls_logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["cls_logits"]
            cls_prob = float(torch.softmax(cls_logits, dim=-1)[0, 1].item())
            if self.conditional_decoding and cls_prob <= 0.5:
                ms = (time.perf_counter() - t0) * 1000
                return Prediction(label="Safe", prob=cls_prob, latency_ms=ms, tokens=0)
            out = self.model.base.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)
        ms = (time.perf_counter() - t0) * 1000
        label = "Threat" if cls_prob > 0.5 else "Safe"
        return Prediction(label=label, prob=cls_prob, cot=gen, latency_ms=ms, tokens=out.size(1) - n_prompt)


def load_blind_set(csv_path: str | Path) -> list[dict]:
    csv_path = Path(csv_path)
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


class FileBaseline(Baseline):
    """从预生成的 predictions JSON 加载预测结果，不走 GPU。

    用于基线对比中需要 GPU 推理但耗时过长的场景：
    先在服务器上跑批量生成（或用其他方式），保存为 predictions_<name>.json，
    eval 时直接加载，跳过 GPU 推理。
    """

    def __init__(self, name: str, predictions_path: str | Path):
        import json
        self.name = name
        self.predictions_path = Path(predictions_path)
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            self._all_preds = json.load(f)
        # 按 text 建索引，O(1) 查找
        self._by_text: dict[str, dict] = {}
        for p in self._all_preds:
            self._by_text[p.get("text", "")] = p
        log.info(f"FileBaseline({name}): 加载 {len(self._all_preds)} 条预生成预测 from {self.predictions_path}")

    def predict(self, text: str) -> Prediction:
        p = self._by_text.get(text, {})
        return Prediction(
            label=p.get("model_label", "Safe"),
            prob=float(p.get("model_prob", 0.0)),
            cot=p.get("model_cot", ""),
            latency_ms=float(p.get("latency_ms", 0.0)),
            tokens=int(p.get("tokens", 0)),
        )
