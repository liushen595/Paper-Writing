"""下载并裁剪 WildChat-nontoxic 的英文子集作为盲测集草垛 (haystack)。

输出: data/haystack/wildchat_nontoxic.jsonl
  每行: {"text": "<first user message>", "source": "wildchat-nontoxic",
         "turn": 1, "language": "English"}

策略:
  1. load_dataset("allenai/WildChat-nontoxic") (HuggingFace gated, 需 HF_TOKEN/huggingface-cli login)
  2. 过滤 language == "English" 且 redacted == False
  3. 取 conversation 中第一条 role=="user" 的 content 作 text（最像单条网络评论）
  4. seed=42 随机采样 n 条（默认 5000）
  5. 写到 data/haystack/wildchat_nontoxic.jsonl

注意: HF datasets 默认使用 ~/.cache/huggingface 缓存；本脚本接受该默认行为，
     最终被 pipeline 消费的是裁剪后落到 data/haystack/ 的 JSONL，缓存可事后清理。

用法:
  python -m scripts.prepare_haystack --n 5000
  python -m scripts.prepare_haystack --n 5000 --out data/haystack/wildchat_nontoxic.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.utils.env import PROJECT_ROOT
from src.utils.logging import get_logger, setup_logger, default_log_dir
from src.utils.seed import set_seed

log = get_logger("prepare_haystack")


def _extract_first_user_text(conversation: list[dict]) -> str:
    """从多轮对话列表里取第一条 user 消息的 content。"""
    if not conversation:
        return ""
    for turn in conversation:
        if turn.get("role") == "user":
            content = turn.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    # 退化：取第一轮任意 role 的内容
    return (conversation[0].get("content") or "").strip() if conversation else ""


def main():
    ap = argparse.ArgumentParser(description="准备 WildChat-nontoxic 英文草垛")
    ap.add_argument("--dataset-id", default="allenai/WildChat-nontoxic",
                    help="HuggingFace dataset id（默认 allenai/WildChat-nontoxic）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5000, help="采样条数")
    ap.add_argument("--out", default="data/haystack/wildchat_nontoxic.jsonl")
    ap.add_argument("--language", default="English")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-text-len", type=int, default=20, help="最短字符数过滤")
    ap.add_argument("--max-text-len", type=int, default=2000, help="最长字符数过滤")
    args = ap.parse_args()

    setup_logger(log_file=default_log_dir() / "prepare_haystack.log")
    set_seed(args.seed)

    # 延迟 import，避免脚本启动慢
    from datasets import load_dataset

    log.info(f"开始加载 {args.dataset_id} (split={args.split})...")
    log.info("若失败，请先 `huggingface-cli login` 并在 HF 页面申请该数据集访问权限。")
    ds = load_dataset(args.dataset_id, split=args.split)
    log.info(f"加载完成，原始条数: {len(ds)}")

    # 过滤：English + 非 redacted + 文本长度合理
    candidates: list[str] = []
    skipped_redacted = 0
    skipped_lang = 0
    skipped_len = 0
    skipped_empty = 0
    for row in tqdm(ds, desc="Filtering WildChat", unit="row"):
        if row.get("redacted", False):
            skipped_redacted += 1
            continue
        lang = row.get("language")
        if lang != args.language:
            skipped_lang += 1
            continue
        text = _extract_first_user_text(row.get("conversation", []))
        if not text:
            skipped_empty += 1
            continue
        if len(text) < args.min_text_len or len(text) > args.max_text_len:
            skipped_len += 1
            continue
        candidates.append(text)
    log.info(f"过滤统计: 候选={len(candidates)} 跳过(redacted={skipped_redacted}, "
             f"lang={skipped_lang}, len={skipped_len}, empty={skipped_empty})")

    if not candidates:
        log.error("无可候选文本，终止。请检查 dataset schema 或 language 过滤条件。")
        return

    # 随机采样
    rng = random.Random(args.seed)
    if args.n < len(candidates):
        sampled = rng.sample(candidates, args.n)
    else:
        log.warning(f"候选仅 {len(candidates)} 条，不足 {args.n}，全部采用。")
        sampled = candidates

    # 写出
    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in sampled:
            f.write(json.dumps({
                "text": text,
                "source": "wildchat-nontoxic",
                "turn": 1,
                "language": args.language,
            }, ensure_ascii=False) + "\n")
    log.info(f"写入完成: {len(sampled)} 条 -> {out_path}")


if __name__ == "__main__":
    main()
