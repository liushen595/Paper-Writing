"""评估可视化：把 run_eval 产出的指标矩阵渲染成论文图/表。

输出（写入 cfg.eval.output_dir）：
- confusion_matrix_<name>.png  每个 baseline 一张混淆矩阵（图 1 系列）
- metrics_table.csv            Table 1 机器可读版（TPR/FPR/F1/Precision/Accuracy）
- tpr_fpr_bars.png             跨 baseline TPR/FPR 柱状对比（核心结果图）
- latency_table.md             Table 2 显式 vs 隐式延迟对比（含 tokens/sec、p95）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # 无显示设备也能保存
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import get_logger
from .metrics import compute_binary_metrics, compute_latency, label_to_int

log = get_logger("visualize")


def _confusion_counts(predictions: list[dict]) -> tuple[int, int, int, int]:
    """从 predictions 推 TP/FP/FN/TN。"""
    tp = fp = fn = tn = 0
    for p in predictions:
        pred = 1 if str(p.get("model_label", "")).lower().startswith("threat") else 0
        gt = label_to_int(p.get("ref_label", "Safe"))
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def plot_confusion_matrix(predictions: list[dict], name: str, out_path: Path) -> None:
    """单 baseline 混淆矩阵 PNG。"""
    tp, fp, fn, tn = _confusion_counts(predictions)
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.sum(), 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Safe", "Pred Threat"])
    ax.set_yticklabels(["True Safe", "True Threat"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"混淆矩阵 -> {out_path}")


def plot_tpr_fpr_bars(reports: list[dict], out_path: Path) -> None:
    """跨 baseline TPR/FPR 柱状对比图。"""
    if not reports:
        return
    names = [r["baseline"] for r in reports]
    tpr = [r.get("tpr", 0.0) for r in reports]
    fpr = [r.get("fpr", 0.0) for r in reports]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 4.5))
    ax.bar(x - width / 2, tpr, width, label="TPR (Recall)", color="#2E86C1")
    ax.bar(x + width / 2, fpr, width, label="FPR", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("TPR / FPR across baselines")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, (t, f) in enumerate(zip(tpr, fpr)):
        ax.text(i - width / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"TPR/FPR 柱状图 -> {out_path}")


def write_metrics_csv(reports: list[dict], out_path: Path) -> None:
    """Table 1 机器可读版。"""
    if not reports:
        return
    cols = ["baseline", "tp", "fp", "fn", "tn", "tpr", "fpr", "precision", "f1", "accuracy", "mean_ms", "tokens_per_sec", "p95_ms"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in reports:
            row = {c: r.get(c, "") for c in cols}
            w.writerow(row)
    log.info(f"指标 CSV -> {out_path}")


def write_latency_table(reports: list[dict], out_path: Path) -> None:
    """Table 2 显式 vs 隐式延迟对比 markdown 表。

    按 baseline 行展开 mean_ms / p95_ms / tokens_per_sec，重点关注
    explicit-cot vs implicit-cot 的延迟比（论文核心工程价值论据）。
    """
    if not reports:
        return
    cols = ["baseline", "mean_ms", "p95_ms", "tokens_per_sec"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for r in reports:
        body_lines.append("| " + " | ".join([
            r.get("baseline", ""),
            f"{r.get('mean_ms', 0):.1f}",
            f"{r.get('p95_ms', 0):.1f}",
            f"{r.get('tokens_per_sec', 0):.1f}",
        ]) + " |")
    table = f"{header}\n{sep}\n" + "\n".join(body_lines)

    # 显式 vs 隐式延迟比
    explicit = next((r for r in reports if r.get("baseline") == "explicit-cot"), None)
    implicit = next((r for r in reports if r.get("baseline") == "implicit-cot"), None)
    note = ""
    if explicit and implicit and implicit.get("mean_ms", 0) > 0:
        speedup = explicit["mean_ms"] / implicit["mean_ms"]
        note = f"\n\n**Explicit vs Implicit 加速比**: {speedup:.2f}× （explicit-cot {explicit['mean_ms']:.1f}ms → implicit-cot {implicit['mean_ms']:.1f}ms）"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Table 2: 推理延迟对比\n\n")
        f.write(table)
        f.write(note)
        f.write("\n")
    log.info(f"延迟表 -> {out_path}")


def visualize_all(out_dir: str | Path, reports: list[dict]) -> None:
    """主入口：根据 reports 渲染全部图表。

    reports 同 run_eval 的结构：[{"baseline", tp, fp, fn, tn, tpr, fpr, precision, f1, accuracy, mean_ms, tokens_per_sec, p95_ms}]
    同时读取同目录下的 predictions_<name>.json 画混淆矩阵。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩阵：每个 baseline 一张
    for r in reports:
        name = r["baseline"]
        pred_path = out_dir / f"predictions_{name}.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
            plot_confusion_matrix(preds, name, out_dir / f"confusion_matrix_{name}.png")
        else:
            log.warning(f"未找到 {pred_path}; 跳过 {name} 混淆矩阵")

    # 跨 baseline 对比图
    plot_tpr_fpr_bars(reports, out_dir / "tpr_fpr_bars.png")

    # Table 1 / Table 2
    write_metrics_csv(reports, out_dir / "metrics_table.csv")
    write_latency_table(reports, out_dir / "latency_table.md")
