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


def _write_json_atomic(path: Path, data) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _prediction_record(row: dict, pred) -> dict:
    gt_label = row.get("label", "Safe")
    return {
        "text": row.get("text", ""), "ref_label": gt_label, "model_label": pred.label,
        "model_prob": pred.prob, "model_cot": pred.cot or "",
        "ref_cot": row.get("ground_truth_cot", ""), "source": row.get("source", ""),
        "latency_ms": pred.latency_ms, "tokens": pred.tokens,
    }


def _report_from_predictions(name: str, predictions: list[dict]) -> EvalReport:
    preds = [1 if p.get("model_label", "Safe") == "Threat" else 0 for p in predictions]
    labels = [label_to_int(p.get("ref_label", "Safe")) for p in predictions]
    latencies = [{"ms": p.get("latency_ms", 0.0), "tokens": p.get("tokens", 0)} for p in predictions]
    binary = compute_binary_metrics(preds, labels).as_dict()
    latency = compute_latency(latencies).__dict__
    log.info(f"[{name}] TPR={binary['tpr']:.3f} FPR={binary['fpr']:.3f} F1={binary['f1']:.3f} "
             f"latency_mean={latency['mean_ms']:.1f}ms tps={latency['tokens_per_sec']:.1f}")
    return EvalReport(baseline_name=name, binary=binary, latency=latency, predictions=predictions)


def run_one_baseline(
    baseline: Baseline,
    blind_csv: str | Path,
    threshold: float = 0.5,
    limit: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
) -> EvalReport:
    rows = load_blind_set(blind_csv)
    if limit:
        rows = rows[:limit]
    predictions: list[dict] = []
    checkpoint_path = checkpoint_path or Path()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        if len(predictions) > len(rows):
            predictions = predictions[:len(rows)]
            _write_json_atomic(checkpoint_path, predictions)
        if len(predictions) >= len(rows):
            log.info(f"[{baseline.name}] checkpoint 已完成: {checkpoint_path} ({len(predictions)} 条)，跳过推理")
            return _report_from_predictions(baseline.name, predictions)
        log.info(f"[{baseline.name}] 从 checkpoint 恢复: 已完成 {len(predictions)}/{len(rows)} 条")

    start = len(predictions)
    rows_left = rows[start:]
    texts_left = [r.get("text", "") for r in rows_left]

    # 批量推理（支持 predict_batch 的 baseline 可大幅加速）
    has_batch = type(baseline).predict_batch is not Baseline.predict_batch
    if has_batch and len(texts_left) > 1:
        batch_size = max(1, int(getattr(baseline, "batch_size", len(texts_left))))
        log.info(f"[{baseline.name}] 批量推理 {len(texts_left)} 条样本，batch_size={batch_size}...")
        for offset in tqdm(range(0, len(rows_left), batch_size), desc=f"Eval {baseline.name}", unit="batch"):
            batch_rows = rows_left[offset:offset + batch_size]
            batch_texts = [r.get("text", "") for r in batch_rows]
            batch_preds = baseline.predict_batch(batch_texts)
            predictions.extend(_prediction_record(r, pred) for r, pred in zip(batch_rows, batch_preds))
            if checkpoint_path:
                _write_json_atomic(checkpoint_path, predictions)
    else:
        for r in tqdm(rows_left, desc=f"Eval {baseline.name}", unit="sample"):
            text = r.get("text", "")
            pred = baseline.predict(text)
            predictions.append(_prediction_record(r, pred))
            if checkpoint_path:
                _write_json_atomic(checkpoint_path, predictions)
    return _report_from_predictions(baseline.name, predictions)


def _collect_completed_reports(out_dir: Path, names: list[str], expected_count: int) -> list[dict]:
    reports: list[dict] = []
    for name in names:
        pred_path = out_dir / f"predictions_{name}.json"
        if not pred_path.exists():
            continue
        with open(pred_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        predictions = predictions[:expected_count]
        if len(predictions) < expected_count:
            log.info(f"[{name}] checkpoint 未完成: {len(predictions)}/{expected_count} 条，不纳入当前汇总表")
            continue
        rep = _report_from_predictions(name, predictions)
        reports.append({
            "baseline": name,
            **rep.binary,
            "mean_ms": rep.latency["mean_ms"],
            "tokens_per_sec": rep.latency["tokens_per_sec"],
            "p95_ms": rep.latency["p95_ms"],
        })
    return reports


def _write_eval_artifacts(out_dir: Path, reports: list[dict]) -> Path:
    table = format_metrics_table(reports)
    table_path = out_dir / "metrics_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# 盲测集量化指标矩阵\n\n")
        f.write(table)
        f.write("\n")
    log.info(f"指标表已刷新 -> {table_path}")

    try:
        from .visualize import visualize_all
        visualize_all(out_dir, reports)
    except Exception as e:
        log.warning(f"可视化渲染失败（不影响指标表）: {e}")
    return table_path


def run_eval(cfg: ExperimentConfig, baseline_names: Optional[list[str]] = None, limit: Optional[int] = None,
             pre_generated: Optional[dict[str, str]] = None, batch_size: int = 8) -> Path:
    eval_cfg = cfg.eval
    names = baseline_names or eval_cfg.baselines
    blind_csv = (PROJECT_ROOT / eval_cfg.blind_csv).resolve()
    if not blind_csv.exists():
        raise RuntimeError(f"盲测集不存在: {blind_csv}; 请先运行 src/data/blind_set.py")
    out_dir = (PROJECT_ROOT / eval_cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_generated = pre_generated or {}
    rows_for_count = load_blind_set(blind_csv)
    if limit:
        rows_for_count = rows_for_count[:limit]
    expected_count = len(rows_for_count)
    reports: list[dict] = _collect_completed_reports(out_dir, names, expected_count)
    if reports:
        _write_eval_artifacts(out_dir, reports)
    for name in tqdm(names, desc="Baselines", unit="baseline"):
        pred_out = out_dir / f"predictions_{name}.json"
        if pred_out.exists() and name not in pre_generated:
            rows = load_blind_set(blind_csv)
            if limit:
                rows = rows[:limit]
            with open(pred_out, "r", encoding="utf-8") as f:
                existing_predictions = json.load(f)
            if len(existing_predictions) >= len(rows):
                log.info(f"[{name}] 已有完整预测文件，跳过: {pred_out}")
                continue
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
            _write_json_atomic(pred_out, predictions)
        else:
            baseline = _build_baseline(name, cfg, batch_size=batch_size)
            rep = run_one_baseline(
                baseline, blind_csv, threshold=eval_cfg.threshold,
                limit=limit, checkpoint_path=pred_out,
            )
            predictions = rep.predictions
            binary = rep.binary
            latency = rep.latency
        report = {
            "baseline": name,
            **binary,
            "mean_ms": latency["mean_ms"],
            "tokens_per_sec": latency["tokens_per_sec"],
            "p95_ms": latency["p95_ms"],
        }
        reports = [r for r in reports if r["baseline"] != name]
        reports.append(report)
        table_path = _write_eval_artifacts(out_dir, reports)

    table_path = _write_eval_artifacts(out_dir, reports)
    log.info(f"评估完成, 指标表 -> {table_path}")
    return table_path


def _build_baseline(name: str, cfg: ExperimentConfig, batch_size: int = 8) -> Baseline:
    from .baselines import StudentBaseline, ToxicBertBaseline
    if name == "toxic-bert":
        return ToxicBertBaseline(batch_size=batch_size)
    if name in ("explicit-cot", "sft-no-dpo"):
        return StudentBaseline("sft-no-dpo", cfg.sft.output_dir, cfg.sft, conditional_decoding=False, batch_size=batch_size)
    if name == "dpo-only":
        return StudentBaseline("dpo-only", cfg.dpo.output_dir, cfg.sft, conditional_decoding=False, batch_size=batch_size)
    if name == "implicit-cot":
        raise NotImplementedError("implicit-cot baseline 已退役（Phase 3 隐式内化改为 future work）")
    if name == "roberta-large":
        raise NotImplementedError("roberta-large 基线已移除（按决策不再训练 RoBERTa Teacher）")
    raise ValueError(f"未知 baseline: {name}")
