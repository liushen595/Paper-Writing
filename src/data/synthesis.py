"""Phase 0 造数：用 Teacher LLM 把 DOJ 犯罪叙事改写为隐式意图言论 + Explicit CoT。

流程（参照 Wen et al. 2023 的语言学特征提示 + ToXCL 的犯罪类别生成）:
  1. 从 DOJ 记录抽取案情要素（crime_types / title / summary / body）。
  2. Teacher LLM 生成:
     a) implicit_threat: 不含敏感词的隐式意图言论（应用 euphemism/circumlocution/sarcasm/metaphor/rhetorical-question）。
     b) hard_negative: 语义相近但语境安全的对照言论。
     c) thought_process: 显式 CoT 推理链。
     d) label / probability / category。
  3. 写入 data/synthesized/train.jsonl 与 test.jsonl（按 seed 切分，test 不参与训练）。
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .doj_loader import DOJRecord, extract_case_elements, load_doj_records
from .llm_client import ChatMessage, BaseClient, build_client, safe_json_extract

log = get_logger("synthesis")


SYSTEM_PROMPT = """You are a criminal-intent data synthesis expert. Your task is to rewrite a given U.S. Department of Justice (DOJ) press release into an "implicit criminal intent utterance" with its accompanying reasoning chain, used to train a small language model to recognize implicit threats.

You MUST strictly follow these rules:
1. implicit_threat must be a single sentence that **an ordinary person might post online**. It must **NOT contain any sensitive or criminal keywords** (e.g., kill, bomb, poison, gun, rob, drug names, etc.). It MUST express criminal intent through one of: euphemism, circumlocution, sarcasm, metaphor, or rhetorical question.
2. hard_negative must be a sentence that is **topically similar** to implicit_threat but has an **obviously safe context** (e.g., gaming, movie, academic, fiction, hypothetical scenario). Used to reduce false positive rate.
3. thought_process is an explicit chain-of-thought reasoning in the format: "[Reasoning] A -> B -> C -> Conclusion", step by step inferring intent from context.
4. label is either "Threat" or "Safe"; probability is 0.0-1.0; category is chosen from the given list or self-defined.

**Output ONLY a JSON object. No extra text, no markdown code blocks. All content must be in English.**"""


USER_TEMPLATE = """DOJ Case Elements:
- Title: {title}
- Summary: {summary}
- Body: {body}
- Crime Types: {crime_types}

Output the following JSON (no markdown code blocks, all content in English):
{{
  "implicit_threat": "<implicit intent utterance without sensitive keywords>",
  "hard_negative": "<safe-context utterance>",
  "thought_process": "[Reasoning] ... -> ... -> This constitutes a high-risk implicit intent.",
  "label": "Threat",
  "probability": 0.95,
  "category": "<e.g. Cyber/Narcotics/Aviation/Fraud/Violence/...>"
}}"""


def _build_messages(record: DOJRecord) -> list[ChatMessage]:
    elem = extract_case_elements(record)
    body_excerpt = record.body[:500] if record.body else record.summary
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", USER_TEMPLATE.format(
            title=elem["title"],
            summary=elem["summary"],
            body=body_excerpt,
            crime_types=", ".join(elem["crime_types"]),
        )),
    ]


def _parse_synthesis(raw: str, record: DOJRecord) -> Optional[dict]:
    try:
        obj = safe_json_extract(raw)
    except ValueError as e:
        log.warning(f"解析失败: {e}; url={record.url}")
        return None
    required = ("implicit_threat", "hard_negative", "thought_process", "label", "probability", "category")
    if not all(k in obj for k in required):
        log.warning(f"字段缺失: {set(required) - set(obj.keys())}; url={record.url}")
        return None
    if obj["label"] not in ("Threat", "Safe"):
        obj["label"] = "Threat" if obj["label"].lower().startswith("threat") else "Safe"
    try:
        obj["probability"] = float(obj["probability"])
    except (TypeError, ValueError):
        obj["probability"] = 1.0 if obj["label"] == "Threat" else 0.0
    obj["source_url"] = record.url
    obj["source_date"] = record.date
    return obj


def synthesize_one(client: BaseClient, record: DOJRecord, temperature: float = 0.7, max_retries: int = 8) -> Optional[dict]:
    """合成单条数据。数据宝贵，最多重试 max_retries 次，每次失败后调整策略。"""
    msgs = _build_messages(record)
    for attempt in range(max_retries):
        log.info(f"开始调用 Teacher (尝试 {attempt+1}/{max_retries}), url={record.url}")
        start_time = time.time()
        try:
            raw = client.chat(msgs, temperature=temperature, max_tokens=2048)
            elapsed = time.time() - start_time
            log.info(f"Teacher 调用成功，耗时 {elapsed:.1f}s，响应长度 {len(raw)} 字符")
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - start_time
            log.error(f"Teacher 调用失败({attempt+1}/{max_retries}), 耗时 {elapsed:.1f}s: {e}; url={record.url}")
            time.sleep(2 ** min(attempt, 4))
            continue
        result = _parse_synthesis(raw, record)
        if result is not None:
            return result
        # 解析失败，记录原始响应用于调试
        log.warning(f"解析失败(尝试 {attempt+1}/{max_retries}), url={record.url}")
        if attempt < max_retries - 1:
            log.debug(f"原始响应前200字: {raw[:200]}")
    log.error(f"合成彻底失败，已重试 {max_retries} 次; url={record.url}")
    return None


def run_synthesis(
    data_cfg: DataConfig,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    set_seed(data_cfg.seed)
    out_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    if train_path.exists() and not overwrite:
        log.warning(f"{train_path} 已存在且 overwrite=False，跳过；删除该文件或设置 overwrite=True 重跑")
        return

    # 首次运行时清空输出文件
    train_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    criminal_path = (PROJECT_ROOT / data_cfg.raw_criminal).resolve()
    records = load_doj_records(criminal_path, limit=limit)
    log.info(f"待合成 DOJ 记录数: {len(records)}")

    client = build_client(provider_name=provider_name, model=model)
    log.info(f"使用 Teacher provider={client.provider.name}, model={client.model}")

    total_ok = 0
    for i, rec in enumerate(records):
        log.info(f"处理进度 [{i+1}/{len(records)}]")
        synth = synthesize_one(client, rec)
        if synth is not None:
            total_ok += 1
            _save_single(synth, train_path, test_path, data_cfg.train_ratio, data_cfg.seed)
            log.info(f"成功合成并保存，当前累计成功 {total_ok} 条")
        else:
            log.warning(f"合成失败，url={rec.url}")
    log.info(f"合成完成: 共 {total_ok} 条 -> {train_path}(train) + {test_path}(test)")


def _save_single(item: dict, train_path: Path, test_path: Path, train_ratio: float, seed: int) -> None:
    """单条数据保存：根据 url 的确定性 hash 决定写入 train 还是 test。"""
    url = item.get("source_url", "")
    h = int(hashlib.md5(f"{url}:{seed}".encode()).hexdigest(), 16) % 10000
    threshold = int(train_ratio * 10000)
    target_path = train_path if h < threshold else test_path
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _flush(results: list[dict], train_path: Path, test_path: Path, train_ratio: float, seed: int, partial: bool) -> None:
    """将 results 按 train_ratio 划分后追加写入文件。partial=True 时不清空文件。"""
    if not results:
        return
    rng = random.Random(seed)
    indexed = list(enumerate(results))
    rng.shuffle(indexed)
    n_train = int(len(indexed) * train_ratio)
    train_idx = {i for i, _ in indexed[:n_train]}
    # 始终用追加模式；首次写入前由 run_synthesis 清空文件
    with open(train_path, "a", encoding="utf-8") as f:
        for i, r in indexed:
            if i in train_idx:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path, "a", encoding="utf-8") as f:
        for i, r in indexed:
            if i not in train_idx:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results.clear()


def load_synthesized(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
