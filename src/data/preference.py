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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from tqdm import tqdm

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
    try:
        obj = safe_json_extract(raw)
    except (ValueError, json.JSONDecodeError):
        obj = {}
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
    candidate_generator,  # callable(prompts: list[str], n: int) -> list[list[str]]
    judge_client: BaseClient,
    data_cfg: DataConfig,
    dpo_cfg: DPOConfig,
    rule_keywords: Optional[list[str]] = None,
    swap_positions: bool = True,
    reference_guided: bool = True,
) -> list[PreferencePair]:
    set_seed(data_cfg.seed)
    rule_keywords = rule_keywords or []

    # 预过滤 rule-based 样本，收集所有有效 prompt
    valid_prompts: list[str] = []
    valid_refs: list[tuple[str, str]] = []  # (ref_label, ref_cot)
    for s in samples:
        prompt = s.get("implicit_threat") or s.get("text", "")
        if rule_filter(prompt, rule_keywords):
            continue
        valid_prompts.append(prompt)
        valid_refs.append((s.get("label", "Threat"), s.get("thought_process", "")))

    # 批量生成所有候选（一次 GPU 调用处理多个 prompt）
    log.info(f"批量生成 {len(valid_prompts)} 个 prompt 的候选回复...")
    all_cands = candidate_generator(valid_prompts, n=2)

    # 逐个 judge
    pairs: list[PreferencePair] = []
    zipped = list(zip(valid_prompts, all_cands, valid_refs))
    for prompt, cands, (ref_label, ref_cot) in tqdm(zipped, desc="Preference pairs", unit="sample"):
        if len(cands) < 2:
            continue
        a, b = cands[0], cands[1]
        try:
            if swap_positions:
                res = judge_with_swap(judge_client, prompt, a, b, ref_label, ref_cot, reference_guided)
            else:
                r = _judge_once(judge_client, prompt, a, b, ref_label, ref_cot, reference_guided)
                res = (a, b) if r["winner"] == "A" else ((b, a) if r["winner"] == "B" else None)
        except Exception as e:
            log.warning(f"Judge 调用异常，跳过样本: {e}")
            continue
        if res is None:
            continue
        chosen, rejected = res
        pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
    log.info(f"偏好对生成完成: 共处理 {len(valid_prompts)} 条，采纳 {len(pairs)} 条")
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


def _batch_generate(
    generate_model,
    tokenizer,
    prompt_texts: list[str],
    temperature: float,
    max_new_tokens: int = 128,
    max_seq_len: int = 1024,
    top_p: float = 0.95,
    batch_size: int = 16,
    desc: str = "Generate",
) -> list[str]:
    """批量 GPU 推理：一次 generate() 处理多个 prompt，显著提升 GPU 利用率。

    prompt_texts 必须是已经 apply_chat_template 的纯文本字符串。
    生成时使用左填充（left-padding），解码时按 attention_mask 切片取生成部分。
    """
    import torch

    device = next(generate_model.parameters()).device

    results: list[str] = []
    total = len(prompt_texts)
    num_batches = (total + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc=desc, unit="batch")
    for start in range(0, total, batch_size):
        batch_texts = prompt_texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)
        input_lens = inputs["attention_mask"].sum(dim=1)  # (B,) 每个样本的实际 token 数

        with torch.no_grad():
            outputs = generate_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(len(batch_texts)):
            gen = tokenizer.decode(
                outputs[j][input_lens[j]:], skip_special_tokens=True,
            )
            results.append(gen)
        pbar.update(1)
    pbar.close()

    return results


def make_sft_candidate_generator(sft_cfg, ckpt_dir: str | Path, temperatures: tuple[float, ...] = (0.3, 1.0)) -> Callable[[list[str], int], list[list[str]]]:
    """构造基于 SFT checkpoint 的批量候选生成器。

    对全部 prompt 先预计算 chat_template 文本，再按不同温度整批调用 _batch_generate，
    避免逐样本串行 GPU 推理。返回的 callable 签名为 (prompts: list[str], n: int) -> list[list[str]]。
    """
    import torch
    from ..models.student import StudentModel, load_tokenizer
    from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE

    log.info(f"加载 SFT 候选生成器: ckpt={ckpt_dir}")
    tokenizer = load_tokenizer(sft_cfg.base_model)
    model = StudentModel.load(sft_cfg, ckpt_dir)
    model.eval()
    system = SYSTEM_PROMPT_SFT
    instr_tpl = INSTRUCTION_TEMPLATE

    def _gen_batch(prompts: list[str], n: int = 2) -> list[list[str]]:
        # 预计算所有 prompt 的 chat_template 文本
        prompt_texts: list[str] = []
        for p in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instr_tpl.format(text=p)},
            ]
            prompt_texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        # 扩展温度列表到 n 个
        temps = list(temperatures)
        while len(temps) < n:
            temps.append(temps[-1] + 0.2)

        # 每个温度整批生成一轮
        all_gen: list[list[str]] = [[] for _ in prompts]
        for t in temps[:n]:
            gen_texts = _batch_generate(
                model.base, tokenizer, prompt_texts, temperature=t,
                max_new_tokens=128, max_seq_len=sft_cfg.max_seq_len,
            )
            for i, g in enumerate(gen_texts):
                all_gen[i].append(g)
        return all_gen

    return _gen_batch


def run_preference_generation(
    data_cfg: DataConfig,
    dpo_cfg: DPOConfig,
    judge_provider: str = "aliyun",
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


def _dummy_candidate_gen(prompts: list[str], n: int = 2) -> list[list[str]]:
    """占位批量候选生成器；正式运行时由 make_sft_candidate_generator 提供 SFT 模型采样。"""
    log.warning("使用占位候选生成器，正式训练请传入 SFT checkpoint")
    return [
        [f"[Reasoning] {p[:50]}... -> Threat.", f"[Reasoning] {p[:50]}... -> Safe."][:n]
        for p in prompts
    ]


def generate_candidates_only(
    data_cfg: DataConfig,
    sft_cfg,
    sft_ckpt: str | Path,
    limit: Optional[int] = None,
    out_path: Optional[str | Path] = None,
    batch_size: int = 16,
) -> Path:
    """Phase A（GPU）：用 SFT 模型**批量**生成候选回复，保存到 candidates.jsonl。

    不调用任何 API，纯 GPU 批量推理。输出每行：
    {"prompt": ..., "candidate_a": ..., "candidate_b": ...,
     "ref_label": ..., "ref_cot": ...}

    batch_size 控制每轮 GPU 推理的样本数（RTX 3090 24GB 推荐 16）。
    """
    from .synthesis import load_synthesized
    from ..models.student import StudentModel, load_tokenizer
    from ..data.dataset import SYSTEM_PROMPT_SFT, INSTRUCTION_TEMPLATE

    synth_path = (PROJECT_ROOT / data_cfg.synthesized_dir / "train.jsonl").resolve()
    if not synth_path.exists():
        raise RuntimeError(f"合成训练数据不存在: {synth_path}")
    samples = load_synthesized(synth_path)
    if limit:
        samples = samples[:limit]

    if out_path is None:
        out_path = (PROJECT_ROOT / data_cfg.preference_dir / "candidates.jsonl").resolve()
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：跳过已有行数
    start_idx = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(samples):
            log.info(f"全部 {len(samples)} 条已生成完毕，跳过")
            return out_path
        samples = samples[start_idx:]
        log.info(f"断点续跑: 跳过前 {start_idx} 条，剩余 {len(samples)} 条")

    log.info(f"加载 SFT 模型: {sft_ckpt}")
    tokenizer = load_tokenizer(sft_cfg.base_model)
    model = StudentModel.load(sft_cfg, sft_ckpt)
    model.eval()

    temperatures = (0.3, 1.0)
    system = SYSTEM_PROMPT_SFT
    instr_tpl = INSTRUCTION_TEMPLATE

    # 预计算所有 prompt 文本
    prompts: list[str] = []
    ref_labels: list[str] = []
    ref_cots: list[str] = []
    prompt_texts: list[str] = []
    for s in tqdm(samples, desc="Template prompts", unit="sample"):
        p = s.get("implicit_threat") or s.get("text", "")
        prompts.append(p)
        ref_labels.append(s.get("label", "Threat"))
        ref_cots.append(s.get("thought_process", ""))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instr_tpl.format(text=p)},
        ]
        prompt_texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    log.info(f"预计算 {len(prompt_texts)} 条 prompt 文本完成，开始批量生成 (batch_size={batch_size})")

    # 两轮批量生成：先低温后高温
    cands_low = _batch_generate(
        model.base, tokenizer, prompt_texts, temperature=temperatures[0],
        max_new_tokens=128, max_seq_len=sft_cfg.max_seq_len, batch_size=batch_size,
        desc="Temp 0.3",
    )
    cands_high = _batch_generate(
        model.base, tokenizer, prompt_texts, temperature=temperatures[1],
        max_new_tokens=128, max_seq_len=sft_cfg.max_seq_len, batch_size=batch_size,
        desc="Temp 1.0",
    )

    # 追加写入 JSONL
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(len(samples)):
            record = {
                "prompt": prompts[i],
                "candidate_a": cands_low[i],
                "candidate_b": cands_high[i],
                "ref_label": ref_labels[i],
                "ref_cot": ref_cots[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"候选生成完成: {start_idx + len(samples)} 条 -> {out_path}")
    log.info(f"{'='*60}")
    log.info(f"[下一步] 把 {out_path} 拷贝到本地，然后运行:")
    log.info(f"  scp server:{out_path} ./data/preference/")
    log.info(f"  python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10")
    log.info(f"  # judge 输出: data/preference/dpo_pairs.jsonl")
    log.info(f"  scp ./data/preference/dpo_pairs.jsonl server:{PROJECT_ROOT / data_cfg.preference_dir / 'dpo_pairs.jsonl'}")
    log.info(f"  # 拷贝回服务器后运行: python -m scripts.run_all --only dpo")
    log.info(f"{'='*60}")
    return out_path


def judge_candidates_only(
    data_cfg: DataConfig,
    dpo_cfg: DPOConfig,
    candidates_path: str | Path,
    judge_provider: str = "aliyun",
    judge_model: Optional[str] = None,
    max_workers: int = 10,
    rpm: float = 30000,
    out_path: Optional[str | Path] = None,
) -> Path:
    """Phase B（API，多线程）：读取候选 JSONL，多线程调 judge API 生成偏好对。

    不依赖 GPU，可在本地机器跑。judge_provider 默认 aliyun（qwen-plus）。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates_path = Path(candidates_path)
    if not candidates_path.exists():
        raise RuntimeError(f"候选文件不存在: {candidates_path}")

    candidates: list[dict] = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    log.info(f"加载 {len(candidates)} 条候选 from {candidates_path}")
    judge = build_client(provider_name=judge_provider, model=judge_model)
    log.info(f"Judge provider={judge.provider.name}, model={judge.model}, max_workers={max_workers}")

    if out_path is None:
        out_path = (PROJECT_ROOT / data_cfg.preference_dir / "dpo_pairs.jsonl").resolve()
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rate_limiter = _RateLimiter(rpm)
    write_lock = threading.Lock()
    pairs: list[PreferencePair] = []

    def _judge_one(c: dict) -> Optional[PreferencePair]:
        rate_limiter.wait()
        res = judge_with_swap(
            judge, c["prompt"], c["candidate_a"], c["candidate_b"],
            c.get("ref_label", "Threat"), c.get("ref_cot", ""),
            reference_guided=True,
        )
        if res is None:
            return None
        chosen, rejected = res
        return PreferencePair(prompt=c["prompt"], chosen=chosen, rejected=rejected, reason="")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, c): c for c in candidates}
        done = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judge", unit="pair"):
            done += 1
            try:
                pair = future.result()
            except Exception as e:
                log.warning(f"Judge 线程异常: {e}")
                continue
            if pair is not None:
                with write_lock:
                    pairs.append(pair)

    save_preference_pairs(pairs, out_path)
    log.info(f"偏好对生成完成: {len(pairs)} 条 -> {out_path}")
    log.info(f"{'='*60}")
    log.info(f"[下一步] 把 {out_path} 拷贝回服务器，然后运行:")
    log.info(f"  scp {out_path} server:{PROJECT_ROOT / data_cfg.preference_dir / 'dpo_pairs.jsonl'}")
    log.info(f"  python -m scripts.run_all --only dpo")
    log.info(f"{'='*60}")
    return out_path


class _RateLimiter:
    """Token-bucket 速率限制器，多线程共享。"""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / max(rpm, 1.0)
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        import time
        with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()
