"""一次性脚本：把 train/test.jsonl 里被硬编码的 probability 字段重赋为合理分布。

背景：
  Phase 0 造数时 SYSTEM_PROMPT/USER_TEMPLATE 把 "probability": 0.95 写死，
  导致 1005 条 Threat 全为 0.95、1231 条 Safe（展开自 hard_negative）反推 0.05。
  probability 在下游 training/eval 代码中并不被消费（SFT 用 label 做 L_cls、
  评估用分类头 sigmoid 概率），所以重赋只是为了数据集自身的分布合理性与
  论文叙事一致性。

策略（seed=42）：
  - Threat 侧：Beta(2,2) 截到 [0.70, 0.99]，反映 Teacher 对正样本的高置信但有变化。
  - Safe 侧： Beta(2,2) 截到 [0.01, 0.30]，反映 Teacher 对安全样本的低威胁置信。
  - 备份原文件为 *.bak；幂等（重复运行先恢复备份再写）。
  - 同步刷新 hard_negatives.jsonl 中的 probability 字段（依赖 train/test 的抽样结果）。

用法:
  python -m scripts.patch_probability
  python -m scripts.patch_probability --dry-run   # 仅打印统计不写盘
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.utils.config import load_config
from src.utils.env import PROJECT_ROOT
from src.utils.logging import get_logger
from src.utils.seed import set_seed

import numpy as np

log = get_logger("patch_probability")


def _sample_beta(low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """从 Beta(2,2) 截断到 [low, high] 采样 n 个值。Beta(2,2) 均值 0.5，钟形分布。"""
    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        batch = rng.beta(2.0, 2.0, size=(n - filled) * 2)
        batch = batch[(batch >= low) & (batch <= high)]
        take = min(len(batch), n - filled)
        out[filled:filled + take] = batch[:take]
        filled += take
    return out


def patch_file(path: Path, rng: np.random.Generator, dry_run: bool) -> tuple[int, int]:
    """就地 patch 单个 jsonl 文件的 probability 字段。返回 (n_threat, n_safe)。"""
    if not path.exists():
        log.warning(f"文件不存在: {path}")
        return (0, 0)
    # 幂等：若有 .bak 则先恢复，避免叠加 patch
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        shutil.copy2(bak, path)
        log.info(f"检测到备份，先恢复原始内容: {path}")
    else:
        shutil.copy2(path, bak)
        log.info(f"已备份: {path} -> {bak}")

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return (0, 0)

    # 同一记录里 Threat 用主概率，hard_negative 反推 Safe 概率
    threat_idx = [i for i, r in enumerate(records) if r.get("label", "Threat") == "Threat"]
    safe_idx = [i for i, r in enumerate(records) if r.get("label", "Threat") == "Safe"]
    # 注意：当前 train/test.jsonl 的 label 字段都是 "Threat"（synthesis 输出），
    # 真正的 Safe 信息在 hard_negative 字段里。本脚本只 patch 主 probability 字段。
    n_threat = len(threat_idx)
    if n_threat > 0:
        probs = _sample_beta(0.70, 0.99, n_threat, rng)
        for i, p in zip(threat_idx, probs):
            records[i]["probability"] = round(float(p), 3)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return (n_threat, len(safe_idx))


def patch_hard_negatives(path: Path, rng: np.random.Generator, dry_run: bool) -> int:
    """刷新 hard_negatives.jsonl 的 probability 字段（label=Safe 全部）。"""
    if not path.exists():
        log.warning(f"hard_negatives.jsonl 不存在: {path}（请先运行 run_hard_negatives）")
        return 0
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    if n == 0:
        return 0
    probs = _sample_beta(0.01, 0.30, n, rng)
    for r, p in zip(records, probs):
        r["probability"] = round(float(p), 3)
    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n


def _stats(path: Path) -> None:
    if not path.exists():
        return
    probs: list[float] = []
    labels: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            labels.append(d.get("label", "?"))
            probs.append(float(d.get("probability", 0.0)))
    if not probs:
        return
    arr = np.array(probs)
    log.info(f"[{path.name}] n={len(probs)} label={dict(Counter(labels))} "
             f"prob min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f} std={arr.std():.3f}")


def main():
    ap = argparse.ArgumentParser(description="重赋 train/test/hard_negatives 的 probability 字段")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    synth_dir = (PROJECT_ROOT / cfg.data.synthesized_dir).resolve()
    log.info(f"=== patch_probability (dry_run={args.dry_run}) ===")
    n1 = patch_file(synth_dir / "train.jsonl", rng, args.dry_run)
    n2 = patch_file(synth_dir / "test.jsonl", rng, args.dry_run)
    n3 = patch_hard_negatives(synth_dir / "hard_negatives.jsonl", rng, args.dry_run)
    log.info(f"patch 完成: train(t={n1[0]},s={n1[1]}) test(t={n2[0]},s={n2[1]}) hard(s={n3})")
    log.info("--- 事后统计 ---")
    _stats(synth_dir / "train.jsonl")
    _stats(synth_dir / "test.jsonl")
    _stats(synth_dir / "hard_negatives.jsonl")


if __name__ == "__main__":
    main()
