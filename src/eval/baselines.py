"""评估基线：toxic-bert / qwen-zeroshot / explicit-cot / sft-no-dpo / dpo-only / implicit-cot。

每个 baseline 暴露 predict(text) -> (label, prob, cot?) 的统一接口。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


class QwenZeroShotBaseline(Baseline):
    """未微调 Qwen3-8B 零样本提示（默认 thinking 模式）。"""

    name = "qwen-zeroshot"

    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_tpl = (
            "Read the message and decide if it expresses implicit criminal intent. "
            "Output JSON {{\"label\": \"Threat\"|\"Safe\", \"prob\": 0.0-1.0, \"reason\": \"...\"}}.\n"
            "Message: {text}\nJSON:"
        )

    def predict(self, text: str) -> Prediction:
        import time, torch
        from ..data.llm_client import safe_json_extract
        t0 = time.perf_counter()
        prompt = self.prompt_tpl.format(text=text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = self.tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        ms = (time.perf_counter() - t0) * 1000
        try:
            obj = safe_json_extract(gen)
            label = obj.get("label", "Safe")
            prob = float(obj.get("prob", 0.0))
        except Exception:
            label = "Safe"
            prob = 0.0
        return Prediction(label=label, prob=prob, cot=gen, latency_ms=ms, tokens=out.size(1))


class StudentBaseline(Baseline):
    """通用：加载某个 StudentModel checkpoint 做推理（用于 explicit-cot / sft-no-dpo / implicit-cot 对比）。"""

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
        self.conditional_decoding = conditional_decoding
        from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE
        self.system = SYSTEM_PROMPT_SFT
        self.instr_tpl = INSTRUCTION_TEMPLATE

    def predict(self, text: str) -> Prediction:
        import time, torch
        from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.instr_tpl.format(text=text)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            cls_logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["cls_logits"]
            cls_prob = float(torch.softmax(cls_logits, dim=-1)[0, 1].item())
            if self.conditional_decoding and cls_prob <= 0.5:
                ms = (time.perf_counter() - t0) * 1000
                return Prediction(label="Safe", prob=cls_prob, latency_ms=ms, tokens=inputs["input_ids"].size(1))
            out = self.model.base.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        ms = (time.perf_counter() - t0) * 1000
        label = "Threat" if cls_prob > 0.5 else "Safe"
        return Prediction(label=label, prob=cls_prob, cot=gen, latency_ms=ms, tokens=out.size(1))


def load_blind_set(csv_path: str | Path) -> list[dict]:
    csv_path = Path(csv_path)
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows
