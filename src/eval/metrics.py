"""量化指标矩阵：FPR / TPR / F1 / Precision / Accuracy + 混淆矩阵 + 推理延迟。

所有测试基于硬阈值（判定概率 > threshold 即视为 Threat）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.logging import get_logger

log = get_logger("metrics")


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    tpr: float  # recall
    fpr: float
    precision: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def confusion_matrix(self) -> np.ndarray:
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)


def compute_binary_metrics(preds: list[int], labels: list[int]) -> BinaryMetrics:
    assert len(preds) == len(labels)
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
    acc = (tp + tn) / max(1, len(labels))
    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr, precision=precision, f1=f1, accuracy=acc)


@dataclass
class LatencyMetrics:
    n: int
    total_ms: float
    mean_ms: float
    tokens_per_sec: float
    p95_ms: float


def compute_latency(records: list[dict]) -> LatencyMetrics:
    """records: [{"ms": float, "tokens": int}, ...]"""
    if not records:
        return LatencyMetrics(0, 0.0, 0.0, 0.0, 0.0)
    ms = np.array([r["ms"] for r in records])
    toks = np.array([r.get("tokens", 0) for r in records])
    total = float(ms.sum())
    mean = float(ms.mean())
    tps = float(toks.sum() / (total / 1000.0)) if total > 0 else 0.0
    p95 = float(np.percentile(ms, 95))
    return LatencyMetrics(n=len(records), total_ms=total, mean_ms=mean, tokens_per_sec=tps, p95_ms=p95)


def threshold_predictions(probs: list[float], threshold: float = 0.5) -> list[int]:
    return [1 if p > threshold else 0 for p in probs]


def label_to_int(label: str) -> int:
    return 1 if label.lower().startswith("threat") else 0


def format_metrics_table(rows: list[dict]) -> str:
    """格式化成 markdown 表。"""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return f"{header}\n{sep}\n{body}"
