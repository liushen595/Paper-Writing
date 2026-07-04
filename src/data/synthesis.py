"""Phase 0 造数：用 Teacher LLM 把 DOJ 犯罪叙事改写为隐式意图言论 + Explicit CoT。

流程（参照 Wen et al. 2023 的语言学特征提示 + ToXCL 的犯罪类别生成）:
  1. 从 DOJ 记录抽取案情要素（crime_types / title / summary）。
  2. Teacher LLM 生成:
     a) implicit_threat: 不含敏感词的隐式意图言论（应用 euphemism/circumlocution/sarcasm/metaphor/rhetorical-question）。
     b) hard_negative: 语义相近但语境安全的对照言论。
     c) thought_process: 显式 CoT 推理链。
     d) label / probability / category。
  3. 写入 data/synthesized/train.jsonl 与 test.jsonl（按 seed 切分，test 不参与训练）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..utils.config import DataConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .doj_loader import DOJRecord, extract_case_elements, load_doj_records
from .llm_client import ChatMessage, BaseClient, build_client, safe_json_extract

log = get_logger("synthesis")


SYSTEM_PROMPT = """你是一名犯罪意图数据合成专家。你的任务是把给定的美国司法部(DOJ)犯罪新闻稿，改写成"隐式犯罪意图言论"及其配套推理，用于训练一个能识别隐式威胁的小语言模型。

你必须严格遵守：
1. implicit_threat 必须是一句**普通人可能在网上发布的话**，**不得包含任何敏感词或犯罪关键词**（如：杀、炸、毒、枪、劫、毒品名等），必须通过委婉语(euphemism)、迂回(circumlocution)、反讽(sarcasm)、隐喻(metaphor)、反问(rhetorical question) 之一来表达犯罪意图。
2. hard_negative 是与 implicit_threat 话题相近、词汇相近但**语境明显安全**的话（如游戏/影视/学术/小说/假设场景），用来降低误报。
3. thought_process 是显式推理链，用 "[推理] A -> B -> C -> 结论" 的格式，逐步从语境推断意图。
4. label 取 "Threat" 或 "Safe"；probability 取 0.0-1.0；category 从给定列表选取或自拟。

只输出一个 JSON 对象，不要任何额外解释。"""


USER_TEMPLATE = """DOJ 案情要素:
- 标题: {title}
- 摘要: {summary}
- 犯罪类型: {crime_types}

请输出如下 JSON（不要 markdown 代码块）:
{{
  "implicit_threat": "<不含敏感词的隐式意图言论>",
  "hard_negative": "<语境安全的对照言论>",
  "thought_process": "[推理] ... -> ... -> 属于高危隐式意图。",
  "label": "Threat",
  "probability": 0.95,
  "category": "<如 Cyber/Narcotics/Aviation/Fraud/Violence/...>"
}}"""


def _build_messages(record: DOJRecord) -> list[ChatMessage]:
    elem = extract_case_elements(record)
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", USER_TEMPLATE.format(
            title=elem["title"],
            summary=elem["summary"],
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


def synthesize_one(client: BaseClient, record: DOJRecord, temperature: float = 0.8) -> Optional[dict]:
    msgs = _build_messages(record)
    try:
        raw = client.chat(msgs, temperature=temperature, max_tokens=512)
    except Exception as e:  # noqa: BLE001
        log.error(f"Teacher 调用失败: {e}; url={record.url}")
        return None
    return _parse_synthesis(raw, record)


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

    criminal_path = (PROJECT_ROOT / data_cfg.raw_criminal).resolve()
    records = load_doj_records(criminal_path, limit=limit)
    log.info(f"待合成 DOJ 记录数: {len(records)}")

    client = build_client(provider_name=provider_name, model=model)
    log.info(f"使用 Teacher provider={client.provider.name}, model={client.model}")

    results: list[dict] = []
    for i, rec in enumerate(records):
        synth = synthesize_one(client, rec)
        if synth is not None:
            results.append(synth)
        if (i + 1) % 50 == 0:
            log.info(f"进度 {i+1}/{len(records)}, 已成功 {len(results)}")
            _flush(results, train_path, test_path, data_cfg.train_ratio, data_cfg.seed, partial=True)
    _flush(results, train_path, test_path, data_cfg.train_ratio, data_cfg.seed, partial=False)
    log.info(f"合成完成: 共 {len(results)} 条 -> {train_path}(train) + {test_path}(test)")


def _flush(results: list[dict], train_path: Path, test_path: Path, train_ratio: float, seed: int, partial: bool) -> None:
    mode = "a" if partial else "w"
    if not partial:
        train_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)
    if not results:
        return
    rng = random.Random(seed)
    indexed = list(enumerate(results))
    rng.shuffle(indexed)
    n_train = int(len(indexed) * train_ratio)
    train_idx = {i for i, _ in indexed[:n_train]}
    with open(train_path, mode, encoding="utf-8") as f:
        for i, r in indexed:
            if i in train_idx:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path, mode, encoding="utf-8") as f:
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
