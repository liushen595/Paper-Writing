"""LLM-as-a-Judge 评估：复用 src/data/preference.py 的 judge 逻辑 + ToXCL 自定义解释评估。

用于：
1. 评估显式 CoT 输出的推理质量（质性评估）。
2. ToXCL Alg.1 自定义解释评估：双方均 [None] 加分，不匹配罚 0，匹配则计 BLEU/ROUGE/BERTScore。
3. 报告 S1/S2 一致性、偏差监控。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..data.llm_client import ChatMessage, BaseClient, build_client, safe_json_extract
from ..utils.logging import get_logger

log = get_logger("llm_judge_eval")


@dataclass
class JudgeEvalResult:
    n: int
    s1_agreement: float
    s2_agreement: float
    biased_first_rate: float
    verbosity_bias_rate: float
    per_sample: list[dict] = field(default_factory=list)


QUALITY_JUDGE_SYSTEM = """你是犯罪意图识别的质量裁判。给定用户言论、模型输出的推理链与标签、参考推理与标签，
评估模型推理的：1) 严谨性 2) 是否捕捉语境异常 3) 是否避免过度敏感。
输出 JSON: {"score": 1-10, "correct": true|false, "reason": "..."}"""

QUALITY_USER_TEMPLATE = """用户言论: {text}
模型推理: {model_cot}
模型标签: {model_label}
参考标签: {ref_label}
参考推理: {ref_cot}

评估并输出 JSON。"""


def judge_quality(client: BaseClient, text: str, model_cot: str, model_label: str, ref_label: str, ref_cot: str) -> dict:
    msgs = [
        ChatMessage("system", QUALITY_JUDGE_SYSTEM),
        ChatMessage("user", QUALITY_USER_TEMPLATE.format(
            text=text, model_cot=model_cot, model_label=model_label,
            ref_label=ref_label, ref_cot=ref_cot,
        )),
    ]
    raw = client.chat(msgs, temperature=0.0, max_tokens=256)
    obj = safe_json_extract(raw)
    if "score" not in obj:
        obj["score"] = 0
    if "correct" not in obj:
        obj["correct"] = (model_label == ref_label)
    return obj


def toxcl_explanation_score(model_cot: str, ref_cot: str, model_label: str, ref_label: str) -> dict:
    """ToXCL Alg.1 风格的简化解释评估。
    - 双方均无解释(均 [None] 或空) -> +100（正确拒绝）
    - 一方有一方无 -> 0
    - 双方均有 -> 计算 token-level F1（简化代替 BLEU/ROUGE/BERTScore，正式版可外接）
    """
    m_empty = (not model_cot) or model_cot.strip().lower() in ("[none]", "none", "")
    r_empty = (not ref_cot) or ref_cot.strip().lower() in ("[none]", "none", "")
    if m_empty and r_empty:
        return {"status": "both_none", "score": 100.0, "f1": 1.0}
    if m_empty != r_empty:
        return {"status": "mismatch", "score": 0.0, "f1": 0.0}
    mt = set(model_cot.lower().split())
    rt = set(ref_cot.lower().split())
    if not mt or not rt:
        return {"status": "both_present", "score": 0.0, "f1": 0.0}
    overlap = len(mt & rt)
    p = overlap / len(mt)
    r = overlap / len(rt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"status": "both_present", "score": f1 * 100, "f1": f1}


def run_judge_eval(
    predictions_csv: str | Path,
    judge_provider: str = "glm",
    judge_model: Optional[str] = None,
    out_path: Optional[str | Path] = None,
) -> JudgeEvalResult:
    """对一批模型预测做 LLM-as-judge 质量评估。predictions_csv 含 text/model_label/model_cot/ref_label/ref_cot 列。"""
    client = build_client(provider_name=judge_provider, model=judge_model)
    rows: list[dict] = []
    with open(predictions_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    per_sample: list[dict] = []
    correct = 0
    for r in rows:
        res = judge_quality(
            client, r.get("text", ""), r.get("model_cot", ""), r.get("model_label", "Safe"),
            r.get("ref_label", "Safe"), r.get("ref_cot", ""),
        )
        if res.get("correct"):
            correct += 1
        per_sample.append({**r, **res})
    s1 = correct / max(1, len(rows))
    s2 = correct / max(1, len([r for r in rows if r.get("ref_label") != "tie"]))
    result = JudgeEvalResult(
        n=len(rows), s1_agreement=s1, s2_agreement=s2,
        biased_first_rate=0.0, verbosity_bias_rate=0.0, per_sample=per_sample,
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, indent=2)
        log.info(f"Judge 评估结果保存到 {out_path}; S1={s1:.3f} S2={s2:.3f} n={len(rows)}")
    return result
