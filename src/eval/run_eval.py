"""评估主入口：在盲测集上跑指定 baseline，计算指标矩阵，输出报告。"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..utils.config import EvalConfig, ExperimentConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from .baselines import Baseline, load_blind_set
from .metrics import compute_binary_metrics, compute_latency, format_metrics_table, label_to_int, threshold_predictions

log = get_logger("run_eval")


@dataclass
class EvalReport:
    baseline_name: str
    binary: dict
    latency: dict
    predictions: list[dict] = field(default_factory=list)


def run_one_baseline(baseline: Baseline, blind_csv: str | Path, threshold: float = 0.5, limit: Optional[int] = None) -> EvalReport:
    rows = load_blind_set(blind_csv)
    if limit:
        rows = rows[:limit]
    texts = [r.get("text", "") for r in rows]
    preds, labels, latencies = [], [], []
    predictions: list[dict] = []

    # 批量推理（QwenZeroShotBaseline 等支持 predict_batch，大幅加速）
    has_batch = type(baseline).predict_batch is not Baseline.predict_batch
    if has_batch and len(texts) > 1:
        log.info(f"[{baseline.name}] 批量推理 {len(texts)} 条样本...")
        batch_preds = baseline.predict_batch(texts)
        for r, pred in zip(rows, batch_preds):
            gt_label = r.get("label", "Safe")
            preds.append(1 if pred.label == "Threat" else 0)
            labels.append(label_to_int(gt_label))
            latencies.append({"ms": pred.latency_ms, "tokens": pred.tokens})
            predictions.append({
                "text": r.get("text", ""), "ref_label": gt_label, "model_label": pred.label,
                "model_prob": pred.prob, "model_cot": pred.cot or "",
                "ref_cot": r.get("ground_truth_cot", ""), "source": r.get("source", ""),
                "latency_ms": pred.latency_ms, "tokens": pred.tokens,
            })
    else:
        for r in tqdm(rows, desc=f"Eval {baseline.name}", unit="sample"):
            text = r.get("text", "")
            gt_label = r.get("label", "Safe")
            pred = baseline.predict(text)
            preds.append(1 if pred.label == "Threat" else 0)
            labels.append(label_to_int(gt_label))
            latencies.append({"ms": pred.latency_ms, "tokens": pred.tokens})
            predictions.append({
                "text": text, "ref_label": gt_label, "model_label": pred.label,
                "model_prob": pred.prob, "model_cot": pred.cot or "",
                "ref_cot": r.get("ground_truth_cot", ""), "source": r.get("source", ""),
                "latency_ms": pred.latency_ms, "tokens": pred.tokens,
            })
    binary = compute_binary_metrics(preds, labels).as_dict()
    latency = compute_latency(latencies).__dict__
    log.info(f"[{baseline.name}] TPR={binary['tpr']:.3f} FPR={binary['fpr']:.3f} F1={binary['f1']:.3f} "
             f"latency_mean={latency['mean_ms']:.1f}ms tps={latency['tokens_per_sec']:.1f}")
    return EvalReport(baseline_name=baseline.name, binary=binary, latency=latency, predictions=predictions)


def run_eval(cfg: ExperimentConfig, baseline_names: Optional[list[str]] = None, limit: Optional[int] = None,
             pre_generated: Optional[dict[str, str]] = None) -> Path:
    eval_cfg = cfg.eval
    names = baseline_names or eval_cfg.baselines
    blind_csv = (PROJECT_ROOT / eval_cfg.blind_csv).resolve()
    if not blind_csv.exists():
        raise RuntimeError(f"盲测集不存在: {blind_csv}; 请先运行 src/data/blind_set.py")
    out_dir = (PROJECT_ROOT / eval_cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_generated = pre_generated or {}
    reports: list[dict] = []
    for name in tqdm(names, desc="Baselines", unit="baseline"):
        if name in pre_generated:
            # 使用预生成预测，跳过 GPU 推理
            pred_path = pre_generated[name]
            log.info(f"[{name}] 使用预生成预测: {pred_path}")
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            if limit:
                predictions = predictions[:limit]
            preds = [1 if p.get("model_label", "Safe") == "Threat" else 0 for p in predictions]
            labels = [label_to_int(p.get("ref_label", "Safe")) for p in predictions]
            latencies = [{"ms": p.get("latency_ms", 0.0), "tokens": p.get("tokens", 0)} for p in predictions]
            binary = compute_binary_metrics(preds, labels).as_dict()
            latency = compute_latency(latencies).__dict__
        else:
            baseline = _build_baseline(name, cfg)
            rep = run_one_baseline(baseline, blind_csv, threshold=eval_cfg.threshold, limit=limit)
            predictions = rep.predictions
            binary = rep.binary
            latency = rep.latency
        with open(out_dir / f"predictions_{name}.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        reports.append({
            "baseline": name,
            **binary,
            "mean_ms": latency["mean_ms"],
            "tokens_per_sec": latency["tokens_per_sec"],
            "p95_ms": latency["p95_ms"],
        })
    table = format_metrics_table(reports)
    table_path = out_dir / "metrics_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# 盲测集量化指标矩阵\n\n")
        f.write(table)
        f.write("\n")
    log.info(f"评估完成, 指标表 -> {table_path}")

    # 渲染论文图表（混淆矩阵 PNG / Table 1 CSV / TPR-FPR 柱状图 / Table 2 延迟表）
    try:
        from .visualize import visualize_all
        visualize_all(out_dir, reports)
    except Exception as e:
        log.warning(f"可视化渲染失败（不影响指标表）: {e}")

    return table_path


def _build_baseline(name: str, cfg: ExperimentConfig) -> Baseline:
    from .baselines import QwenZeroShotBaseline, StudentBaseline, ToxicBertBaseline
    if name == "toxic-bert":
        return ToxicBertBaseline()
    if name == "qwen-zeroshot":
        return QwenZeroShotBaseline(model_name=cfg.sft.base_model)
    if name in ("explicit-cot", "sft-no-dpo"):
        return StudentBaseline("sft-no-dpo", cfg.sft.output_dir, cfg.sft, conditional_decoding=False)
    if name == "dpo-only":
        return StudentBaseline("dpo-only", cfg.dpo.output_dir, cfg.sft, conditional_decoding=False)
    if name == "implicit-cot":
        raise NotImplementedError("implicit-cot baseline 已退役（Phase 3 隐式内化改为 future work）")
    if name == "roberta-large":
        raise NotImplementedError("roberta-large 基线已移除（按决策不再训练 RoBERTa Teacher）")
    raise ValueError(f"未知 baseline: {name}")
