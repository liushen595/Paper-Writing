"""Phase 0 造数：用 Teacher LLM 处理 DOJ 新闻稿，生成训练数据。

流程：
  1. 从全量 DOJ 记录（不再预过滤犯罪/非犯罪）抽取 title / summary / body。
  2. Teacher LLM 首先判断该新闻稿是否描述刑事案件。
     - 若是，按 Wen et al. 2023 的语言学特征提示改写为隐式意图言论 + 硬负样本。
     - 若是明显的非刑事案件（民事、政策、行政等），生成一条 Safe 噪声样本
       （probability≈0，无 hard_negative），用于扩展负例多样性。
  3. 写入 data/synthesized/train.jsonl 与 test.jsonl（按 seed 切分，test 不参与训练）。
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .doj_loader import DOJRecord, extract_case_elements, load_doj_records
from .llm_client import (
    BaseClient,
    BurstLimitError,
    ChatMessage,
    ContentSafetyError,
    RateLimitError,
    build_client,
    safe_json_extract,
)

log = get_logger("synthesis")

_write_lock = threading.Lock()


class RateLimiter:
    """Token-bucket 速率限制器，多线程共享，确保总 RPM 不超标。"""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / max(rpm, 1.0)
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()


SYSTEM_PROMPT = """You are a criminal-intent data synthesis expert. Your task is to process U.S. Department of Justice (DOJ) press releases and produce training data for a small language model to recognize implicit criminal intent.

For each input, FIRST decide whether the press release describes a criminal case (e.g., conviction, sentencing, indictment, arrest, guilty plea, criminal charges).

Branch A — Criminal case:
1. implicit_threat must be a single sentence that an ordinary person might post online. It must NOT contain sensitive or criminal keywords (e.g., kill, bomb, poison, gun, rob, drug names). It MUST express criminal intent through one of: euphemism, circumlocution, sarcasm, metaphor, or rhetorical question.
2. hard_negative must be a sentence topically similar to implicit_threat but with an obviously safe context (e.g., gaming, movie, academic, fiction, hypothetical scenario). Used to reduce false positive rate.
3. thought_process is an explicit chain-of-thought in the format: "[Reasoning] A -> B -> C -> Conclusion", inferring intent step by step.
4. label is "Threat"; probability is a calibrated confidence 0.0-1.0 (vary by case clarity, do NOT hardcode a constant); category is the crime type.

Branch B — Non-criminal case (civil lawsuit, policy announcement, administrative appointment, report, public comment request, etc.):
1. implicit_threat should be a brief neutral summary or obviously safe utterance based on the press release content (this field is reused as the text of a Safe training sample).
2. hard_negative must be an empty string "". Do NOT generate a hard negative for non-criminal cases.
3. thought_process should state that the press release concerns a non-criminal matter and therefore has no criminal intent.
4. label is "Safe"; probability is 0.0 or extremely low (0.0-0.05); category is "NonCriminal".

Output ONLY a JSON object. No extra text, no markdown code blocks. All content must be in English."""


USER_TEMPLATE = """DOJ Press Release:
- Title: {title}
- Summary: {summary}
- Body: {body}
- Crime Types (heuristic only): {crime_types}

Step 1: Decide whether this press release describes a criminal case.
Step 2: Output exactly the following JSON according to the criminal/non-criminal branch (no markdown code blocks, all content in English):
{{
  "implicit_threat": "<Branch A: implicit intent utterance without sensitive keywords; Branch B: brief neutral safe summary>",
  "hard_negative": "<Branch A: safe-context utterance; Branch B: empty string>",
  "thought_process": "[Reasoning] ... -> ... -> Conclusion",
  "label": "<Threat or Safe>",
  "probability": "<calibrated 0.0-1.0; use 0.0 or near 0.0 for Safe>",
  "category": "<crime type or NonCriminal>"
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


def synthesize_one(
    client: BaseClient,
    record: DOJRecord,
    temperature: float = 0.7,
    max_retries: int = 8,
    rate_limiter: RateLimiter | None = None,
) -> Optional[dict]:
    """合成单条数据。数据宝贵，最多重试 max_retries 次。

    感知 Aliyun 限流/安全错误，分别采用不同的重试策略：
    - ContentSafetyError → 不重试，直接跳过
    - RateLimitError    → sleep 60s 后重试
    - BurstLimitError   → sleep 10s * 2^attempt 后重试
    """
    msgs = _build_messages(record)
    for attempt in range(max_retries):
        if rate_limiter is not None:
            rate_limiter.wait()
        log.info(f"开始调用 Teacher (尝试 {attempt+1}/{max_retries}), url={record.url}")
        start_time = time.time()
        try:
            raw = client.chat(msgs, temperature=temperature, max_tokens=2048)
            elapsed = time.time() - start_time
            log.info(f"Teacher 调用成功，耗时 {elapsed:.1f}s，响应长度 {len(raw)} 字符")
        except ContentSafetyError as e:
            elapsed = time.time() - start_time
            log.warning(f"内容安全拒绝，跳过: {e}; url={record.url}")
            return None
        except RateLimitError as e:
            elapsed = time.time() - start_time
            log.warning(f"RPM/TPM 限流({attempt+1}/{max_retries}), 60s 后重试: {e}; url={record.url}")
            time.sleep(60)
            continue
        except BurstLimitError as e:
            elapsed = time.time() - start_time
            wait = 10 * (2 ** attempt)
            log.warning(f"RPS 突发限流({attempt+1}/{max_retries}), {wait}s 后重试: {e}; url={record.url}")
            time.sleep(min(wait, 120))
            continue
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - start_time
            log.error(f"Teacher 调用失败({attempt+1}/{max_retries}), 耗时 {elapsed:.1f}s: {e}; url={record.url}")
            time.sleep(2 ** min(attempt, 4))
            continue
        result = _parse_synthesis(raw, record)
        if result is not None:
            return result
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
    start: int = 0,
    overwrite: bool = False,
    append: bool = False,
    max_workers: int = 1,
    rpm: float = 120,
) -> None:
    set_seed(data_cfg.seed)
    out_dir = (PROJECT_ROOT / data_cfg.synthesized_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    if append:
        log.info("追加模式：新数据将追加到已有 train/test.jsonl 末尾")
    elif train_path.exists() and not overwrite:
        log.warning(f"{train_path} 已存在且 overwrite=False，跳过；删除该文件或设置 overwrite=True 重跑")
        return
    else:
        train_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)

    raw_doj_path = (PROJECT_ROOT / data_cfg.raw_doj).resolve()
    records = load_doj_records(raw_doj_path, limit=limit)
    if start > 0:
        records = records[start:]
        log.info(f"跳过前 {start} 条，剩余待合成: {len(records)}")
    log.info(f"待合成 DOJ 记录数: {len(records)}")

    client = build_client(provider_name=provider_name, model=model)
    log.info(f"使用 Teacher provider={client.provider.name}, model={client.model}")

    if max_workers <= 1:
        _run_sequential(records, client, train_path, test_path, data_cfg)
    else:
        _run_parallel(records, client, train_path, test_path, data_cfg, max_workers, rpm)

    # 自动提取硬负样本到 hard_negatives.jsonl
    log.info("自动提取硬负样本...")
    from .hard_negatives import merge_hard_negatives
    merge_hard_negatives(data_cfg)


def _save_single(item: dict, train_path: Path, test_path: Path, train_ratio: float, seed: int) -> None:
    """单条数据保存：根据 url 的确定性 hash 决定写入 train 还是 test。线程安全。"""
    url = item.get("source_url", "")
    h = int(hashlib.md5(f"{url}:{seed}".encode()).hexdigest(), 16) % 10000
    threshold = int(train_ratio * 10000)
    target_path = train_path if h < threshold else test_path
    with _write_lock:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _run_sequential(
    records: list,
    client: BaseClient,
    train_path: Path,
    test_path: Path,
    data_cfg: DataConfig,
) -> None:
    total_ok = 0
    for i, rec in enumerate(tqdm(records, desc="Synthesis", unit="rec")):
        synth = synthesize_one(client, rec)
        if synth is not None:
            total_ok += 1
            _save_single(synth, train_path, test_path, data_cfg.train_ratio, data_cfg.seed)
        else:
            log.warning(f"合成失败，url={rec.url}")
    log.info(f"合成完成: 共 {total_ok} 条 -> {train_path}(train) + {test_path}(test)")


def _run_parallel(
    records: list,
    client: BaseClient,
    train_path: Path,
    test_path: Path,
    data_cfg: DataConfig,
    max_workers: int,
    rpm: float,
) -> None:
    rate_limiter = RateLimiter(rpm)
    ok_counter = 0
    fail_counter = 0
    counter_lock = threading.Lock()

    def _proc(rec) -> Optional[dict]:
        return synthesize_one(client, rec, rate_limiter=rate_limiter)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_proc, rec): rec for rec in records}
        done = 0
        total = len(futures)
        for future in tqdm(as_completed(futures), total=total, desc="Synthesis", unit="rec"):
            done += 1
            rec = futures[future]
            try:
                result = future.result()
            except Exception as e:
                log.error(f"线程异常: {e}; url={rec.url}")
                with counter_lock:
                    fail_counter += 1
                continue
            if result is not None:
                _save_single(result, train_path, test_path, data_cfg.train_ratio, data_cfg.seed)
                with counter_lock:
                    ok_counter += 1
            else:
                with counter_lock:
                    fail_counter += 1
    log.info(f"合成完成: 共 {ok_counter} 条 -> {train_path}(train) + {test_path}(test)")


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
