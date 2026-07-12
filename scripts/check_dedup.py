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

"""查验并清理 synthesis 输出中的重复数据。

用法:
    python -m scripts.check_dedup                     # 仅查验，报告重复
    python -m scripts.check_dedup --clean             # 去重并覆写 train/test，重建 hard_negatives
    python -m scripts.check_dedup --clean --dry-run   # 预览去重效果，不写入

去重逻辑:
  - train.jsonl / test.jsonl: 按 source_url 去重，保留第一次出现的记录。
  - hard_negatives.jsonl: 不单独去重，而是从去重后的 train/test 重新提取。
  - 不删除 train/test 之间的划归冲突（hash 划分已保证无交集，验证通过）。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.data.hard_negatives import merge_hard_negatives
from src.utils.config import load_config


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_jsonl(path: Path, records: list[dict]) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.move(str(path), str(bak))
    try:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception:
        if bak.exists():
            shutil.move(str(bak), str(path))
        raise
    else:
        bak.unlink(missing_ok=True)


def _dedup_by_url(records: list[dict]) -> tuple[list[dict], int]:
    """按 source_url 去重，保留首次出现。返回 (去重后列表, 移除数)。"""
    seen: set[str] = set()
    out: list[dict] = []
    removed = 0
    for r in records:
        url = r.get("source_url", "")
        if url in seen:
            removed += 1
        else:
            seen.add(url)
            out.append(r)
    return out, removed


def check(data_dir: str) -> dict:
    """查验重复并打印报告。返回统计 dict。"""
    base = Path(data_dir)
    stats: dict[str, dict] = {}

    for name in ("train", "test", "hard_negatives"):
        path = base / f"{name}.jsonl"
        records = _load_jsonl(path)
        url_counts = Counter(r.get("source_url", "") for r in records)
        unique = len(url_counts)
        total = len(records)
        dups = total - unique
        dup_urls = sorted(((u, c) for u, c in url_counts.items() if c > 1), key=lambda x: -x[1])
        stats[name] = {"total": total, "unique": unique, "dups": dups, "dup_urls": dup_urls}

        print(f"\n=== {name}.jsonl ===")
        print(f"  总记录数: {total}")
        print(f"  唯一 source_url: {unique}")
        print(f"  重复记录数: {dups}")
        if dup_urls:
            print(f"  重复的 source_url 数: {len(dup_urls)}")
            top_n = min(10, len(dup_urls))
            print(f"  Top {top_n} 重复 URL:")
            for url, count in dup_urls[:top_n]:
                print(f"    [{count}次] {url[:100]}")

    # train/test 交叉检查
    train_urls = set(r.get("source_url", "") for r in _load_jsonl(base / "train.jsonl"))
    test_urls = set(r.get("source_url", "") for r in _load_jsonl(base / "test.jsonl"))
    overlap = train_urls & test_urls
    if overlap:
        print(f"\n!!! 警告: train 和 test 有 {len(overlap)} 个重叠 source_url")
    else:
        print(f"\ntrain 与 test 无 source_url 交集 — hash 划分正确。")

    total_unique = len(train_urls | test_urls)
    print(f"\n=== 总计 ===")
    print(f"  train+test 唯一 source_url: {total_unique}")
    print(f"  train 重复: {stats['train']['dups']} | test 重复: {stats['test']['dups']} | hn 重复: {stats['hard_negatives']['dups']}")

    return stats


def clean(data_dir: str, dry_run: bool = False) -> None:
    """去重 train/test，重建 hard_negatives。"""
    base = Path(data_dir)
    total_removed = 0

    for name in ("train", "test"):
        path = base / f"{name}.jsonl"
        records = _load_jsonl(path)
        deduped, removed = _dedup_by_url(records)
        total_removed += removed
        print(f"{name}.jsonl: {len(records)} -> {len(deduped)} 条 ({'移除' if dry_run else '已移除'} {removed} 条重复)")

        if not dry_run:
            _save_jsonl(path, deduped)

    if not dry_run:
        print("从去重后的 train/test 重建 hard_negatives.jsonl...")
        hn_before = len(_load_jsonl(base / "hard_negatives.jsonl"))
        cfg = load_config("configs/default.yaml")
        merge_hard_negatives(cfg.data)
        hn_after = len(_load_jsonl(base / "hard_negatives.jsonl"))
        print(f"hard_negatives.jsonl: {hn_before} -> {hn_after} 条")
        print(f"\n清理完成，共移除 {total_removed} 条重复记录。")
    else:
        print(f"\n[dry-run] 将移除 {total_removed} 条重复记录（未实际写入）。")


def main() -> None:
    ap = argparse.ArgumentParser(description="查验/清理 synthesis 输出中的重复数据")
    ap.add_argument("--data-dir", default="data/synthesized", help="synthesis 输出目录")
    ap.add_argument("--clean", action="store_true", help="去重并覆写 train/test，重建 hard_negatives")
    ap.add_argument("--dry-run", action="store_true", help="与 --clean 配合，仅预览不写入")
    ap.add_argument("--yes", "-y", action="store_true", help="跳过确认提示，直接执行")
    args = ap.parse_args()

    if not Path(args.data_dir).exists():
        print(f"目录 {args.data_dir} 不存在")
        return

    check(args.data_dir)

    if args.clean:
        if args.dry_run:
            print("\n--- dry-run 模式 ---")
        elif not args.yes:
            yn = input(f"\n确认去重 [y/N]? ")
            if yn.strip().lower() != "y":
                print("取消。")
                return
        clean(args.data_dir, dry_run=args.dry_run)

        if not args.dry_run:
            print("\n去重后最终情况:")
            check(args.data_dir)


if __name__ == "__main__":
    main()
