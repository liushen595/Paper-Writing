"""Phase 2 DPO 偏好对自动生成（LLM-as-a-Judge，Zheng et al. 2023）。

策略：
1. 对每个训练 prompt，让待对齐的 SFT 模型生成两个候选回复（不同温度/采样）。
2. 用 Teacher LLM 作为裁判打分；对推理密集型样本采用 reference-guided。
3. 位置交换一致性过滤：A/B 顺序交换调用两次，仅一致才采纳，否则丢弃。
4. 三分类偏好方案（Wen et al. 2023）：chosen 应为"更隐式/更严谨推理"，rejected 为"更草率/更表面"。
5. 规则检测器预过滤（奖励整形）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.config import DPOConfig, DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .llm_client import ChatMessage, BaseClient, build_client, safe_json_extract

log = get_logger("preference")


JUDGE_SYSTEM = """You are a judge for criminal intent detection. Given a user message and two candidate responses (A/B), determine which one more accurately identifies implicit criminal intent.
Evaluation dimensions: (1) Is the reasoning rigorous? (2) Does it avoid over-sensitivity? (3) Does it capture contextual anomalies?
For reasoning-intensive samples, refer to the provided ground-truth label and reference reasoning.
Output ONLY JSON: {"winner": "A"|"B"|"tie", "reason": "<brief reason>"}"""

JUDGE_USER_TEMPLATE = """User message: {prompt}
Reference label: {ref_label}
Reference reasoning: {ref_cot}

Candidate A:
{answer_a}

Candidate B:
{answer_b}

Which candidate is more accurate? Output JSON."""


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected, "reason": self.reason}


def _judge_once(
    client: BaseClient, prompt: str, a: str, b: str, ref_label: str, ref_cot: str, reference_guided: bool
) -> dict:
    user = JUDGE_USER_TEMPLATE.format(
        prompt=prompt,
        ref_label=ref_label if reference_guided else "N/A",
        ref_cot=ref_cot if reference_guided else "N/A",
        answer_a=a,
        answer_b=b,
    )
    raw = client.chat([ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", user)], temperature=0.0, max_tokens=256)
    obj = safe_json_extract(raw)
    if obj.get("winner") not in ("A", "B", "tie"):
        obj["winner"] = "tie"
    return obj


def judge_with_swap(
    client: BaseClient,
    prompt: str,
    a: str,
    b: str,
    ref_label: str,
    ref_cot: str,
    reference_guided: bool = True,
) -> Optional[tuple[str, str]]:
    """位置交换一致性过滤。返回 (winner_text, loser_text) 或 None（不一致）。"""
    r1 = _judge_once(client, prompt, a, b, ref_label, ref_cot, reference_guided)
    r2 = _judge_once(client, prompt, b, a, ref_label, ref_cot, reference_guided)
    w1 = r1["winner"]
    w2 = r2["winner"]
    if w1 == "tie" or w2 == "tie":
        return None
    if w1 == "A" and w2 == "B":  # 两次都判第一个胜 -> 一致
        return a, b
    if w1 == "B" and w2 == "A":  # 两次都判第二个胜 -> 一致
        return b, a
    return None


def rule_filter(prompt: str, rule_keywords: list[str]) -> bool:
    """规则检测器预过滤：命中强犯罪关键词的 prompt 跳过偏好对生成（直接信 ground-truth）。"""
    p = prompt.lower()
    return any(kw in p for kw in rule_keywords)


def build_preference_pairs(
    samples: list[dict],
    candidate_generator,  # callable(prompt, n) -> list[str]
    judge_client: BaseClient,
    data_cfg: DataConfig,
    dpo_cfg: DPOConfig,
    rule_keywords: Optional[list[str]] = None,
    swap_positions: bool = True,
    reference_guided: bool = True,
) -> list[PreferencePair]:
    set_seed(data_cfg.seed)
    rule_keywords = rule_keywords or []
    pairs: list[PreferencePair] = []
    for i, s in enumerate(samples):
        prompt = s.get("implicit_threat") or s.get("text", "")
        ref_label = s.get("label", "Threat")
        ref_cot = s.get("thought_process", "")
        if rule_filter(prompt, rule_keywords):
            continue
        cands = candidate_generator(prompt, n=2)
        if len(cands) < 2:
            continue
        a, b = cands[0], cands[1]
        if swap_positions:
            res = judge_with_swap(judge_client, prompt, a, b, ref_label, ref_cot, reference_guided)
        else:
            r = _judge_once(judge_client, prompt, a, b, ref_label, ref_cot, reference_guided)
            res = (a, b) if r["winner"] == "A" else ((b, a) if r["winner"] == "B" else None)
        if res is None:
            continue
        chosen, rejected = res
        pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
        if (i + 1) % 100 == 0:
            log.info(f"偏好对生成进度 {i+1}/{len(samples)}, 已采纳 {len(pairs)}")
    return pairs


def save_preference_pairs(pairs: list[PreferencePair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"保存 {len(pairs)} 条偏好对 -> {path}")


def load_preference_pairs(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_sft_candidate_generator(sft_cfg, ckpt_dir: str | Path, temperatures: tuple[float, ...] = (0.3, 1.0)) -> "callable":
    """构造基于 SFT checkpoint 的候选生成器。

    对同一 prompt 用不同温度采样 n 个候选：低温=保守（倾向 chosen），高温=多样（更可能产生被 reject 的草率回答）。
    生成内容为 SFT 学到的 "<thought_process> -> <label>" 形式，便于 DPO judge 比较推理严谨度。
    """
    import torch
    from ..models.student import StudentModel, load_tokenizer
    from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE

    log.info(f"加载 SFT 候选生成器: ckpt={ckpt_dir}")
    tokenizer = load_tokenizer(sft_cfg.base_model)
    model = StudentModel.load(sft_cfg, ckpt_dir)
    model.eval()
    device = next(model.parameters()).device
    system = SYSTEM_PROMPT_SFT
    instr_tpl = INSTRUCTION_TEMPLATE

    def _gen(prompt: str, n: int = 2) -> list[str]:
        temps = list(temperatures)
        while len(temps) < n:
            temps.append(temps[-1] + 0.2)
        cands: list[str] = []
        for i in range(n):
            t = temps[i % len(temps)]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instr_tpl.format(text=prompt)},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=sft_cfg.max_seq_len).to(device)
            with torch.no_grad():
                out = model.base.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=128, do_sample=True, temperature=t, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            cands.append(gen)
        return cands

    return _gen


def run_preference_generation(
    data_cfg: DataConfig,
    dpo_cfg: DPOConfig,
    judge_provider: str = "glm",
    judge_model: Optional[str] = None,
    candidate_generator=None,
    sft_cfg=None,
    sft_ckpt: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> None:
    from .synthesis import load_synthesized
    synth_path = (PROJECT_ROOT / data_cfg.synthesized_dir / "train.jsonl").resolve()
    if not synth_path.exists():
        log.error(f"合成训练数据不存在: {synth_path}; 请先运行 synthesis")
        return
    samples = load_synthesized(synth_path)
    if limit:
        samples = samples[:limit]
    judge = build_client(provider_name=judge_provider, model=judge_model)

    # 候选生成器：优先显式传入 -> SFT 采样 -> 占位
    if candidate_generator is None and sft_cfg is not None and sft_ckpt is not None:
        candidate_generator = make_sft_candidate_generator(sft_cfg, sft_ckpt)
    if candidate_generator is None:
        log.warning("未提供 SFT 候选生成器，回退到 _dummy_candidate_gen；正式 DPO 训练不要用此结果")
        candidate_generator = _dummy_candidate_gen

    out_path = (PROJECT_ROOT / data_cfg.preference_dir / "dpo_pairs.jsonl").resolve()
    pairs = build_preference_pairs(
        samples, candidate_generator, judge, data_cfg, dpo_cfg,
        swap_positions=True, reference_guided=True,
    )
    save_preference_pairs(pairs, out_path)


def _dummy_candidate_gen(prompt: str, n: int = 2) -> list[str]:
    """占位候选生成器；正式运行时由 make_sft_candidate_generator 提供 SFT 模型采样。"""
    log.warning("使用占位候选生成器，正式训练请传入 SFT checkpoint")
    return [f"[Reasoning] {prompt[:50]}... -> Threat.", f"[Reasoning] {prompt[:50]}... -> Safe."][:n]
