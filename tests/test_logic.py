"""单元测试：纯逻辑模块（不依赖 GPU / 模型权重 / 网络）。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_metrics_binary():
    from src.eval.metrics import compute_binary_metrics
    m = compute_binary_metrics([1, 0, 1, 0], [1, 0, 0, 0])
    # pred=1,label=1: idx0 -> TP=1; pred=1,label=0: idx2 -> FP=1; FN=0; TN: idx1,idx3 -> 2
    assert m.tp == 1 and m.fp == 1 and m.fn == 0 and m.tn == 2
    assert m.tpr == 1.0 and abs(m.fpr - 1/3) < 1e-6
    assert m.confusion_matrix().shape == (2, 2)


def test_latency():
    from src.eval.metrics import compute_latency
    res = compute_latency([{"ms": 10, "tokens": 5}, {"ms": 30, "tokens": 15}])
    assert res.n == 2 and res.total_ms == 40 and abs(res.mean_ms - 20) < 1e-6
    assert res.tokens_per_sec == 20 / 0.04  # 500


def test_removal_schedule():
    from src.training.implicit_cot import removal_schedule
    assert removal_schedule(0, 100, 8) == 0
    assert removal_schedule(50, 100, 8) == 4
    assert removal_schedule(100, 100, 8) == 8
    assert removal_schedule(150, 100, 8) == 8  # clamp


def test_removal_smoothing_offset():
    from src.training.implicit_cot import removal_smoothing_offset
    import random
    random.seed(0)
    for _ in range(100):
        o = removal_smoothing_offset(4.0)
        assert o >= 0
    assert removal_smoothing_offset(float("inf")) == 0


def test_apply_removal_left():
    full = list(range(20))
    labels = list(range(20))
    span = (5, 10)  # thought 段 [5,10)
    new_ids, new_labels = __import__("src.training.implicit_cot", fromlist=["apply_removal"]).apply_removal(
        full, labels, span, 3, left=True
    )
    # 左移除 3：保留 [8,10) + [10,20)，前 [0,5) 不变
    assert new_ids[:5] == [0, 1, 2, 3, 4]
    assert new_ids[5:7] == [8, 9]
    assert new_ids[7:17] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert len(new_ids) == 17


def test_label_conversion():
    from src.data.dataset import label_to_id, id_to_label
    assert label_to_id("Threat") == 1
    assert label_to_id("Safe") == 0
    assert id_to_label(1) == "Threat"
    assert id_to_label(0) == "Safe"


def test_toxcl_explanation_score():
    from src.eval.llm_judge import toxcl_explanation_score
    r1 = toxcl_explanation_score("", "", "Safe", "Safe")
    assert r1["status"] == "both_none" and r1["score"] == 100.0
    r2 = toxcl_explanation_score("some reasoning", "", "Threat", "Safe")
    assert r2["status"] == "mismatch" and r2["score"] == 0.0
    r3 = toxcl_explanation_score("a b c", "a b c", "Threat", "Threat")
    assert r3["status"] == "both_present" and r3["f1"] == 1.0


def test_judge_human_agreement():
    from src.models.judge import judge_human_agreement
    # human non-tie = indices [0,1,2] -> judge matches 2/3
    res = judge_human_agreement(["A", "B", "tie"], ["A", "B", "A"], s2=True)
    assert abs(res["s2"] - 2/3) < 1e-6
    # S1: tie 也算一致 -> 2/3 (idx0 match, idx1 match, idx2 judge=tie counts as consistent)
    # 实际 S1 统计: (pred=="A" & label=="A") || (pred=="tie" || label=="tie")
    # idx0: A==A -> yes; idx1: B==B -> yes; idx2: tie -> yes; -> 3/3=1.0
    assert abs(res["s1"] - 1.0) < 1e-6


def test_safe_json_extract():
    from src.data.llm_client import safe_json_extract
    obj = safe_json_extract('废话 {"a": 1, "b": "x"} 尾巴')
    assert obj == {"a": 1, "b": "x"}
    with pytest.raises(ValueError):
        safe_json_extract("no json here")


def test_safe_json_extract_repair():
    from src.data.llm_client import safe_json_extract
    # 截断修复：嵌套 JSON 外层缺闭合括号
    trunc = '{"label": "Threat", "details": {"a": 1}, "cat": "Cyber", "prob": 0.9'
    r = safe_json_extract(trunc)
    assert r["label"] == "Threat"
    assert r["details"]["a"] == 1
    assert r["cat"] == "Cyber"


def test_doj_loader(tmp_path: Path):
    from src.data.doj_loader import load_doj_records, extract_case_elements
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"url": "u", "title": "Fentanyl Trafficking", "date": "x", "tags": [], "summary": "drug", "body": ""}) + "\n", encoding="utf-8")
    recs = load_doj_records(p)
    assert len(recs) == 1 and recs[0].title == "Fentanyl Trafficking"
    elem = extract_case_elements(recs[0])
    assert "Narcotics" in elem["crime_types"]


def test_env_loading_no_crash():
    from src.utils.env import load_env_config
    cfg = load_env_config()  # 无 .env 也不应崩溃
    assert cfg.wandb_project == "criminal-intent"


def test_config_roundtrip(tmp_path: Path):
    from src.utils.config import default_config, save_config, load_config
    cfg = default_config()
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    cfg2 = load_config(p)
    assert cfg2.sft.base_model == cfg.sft.base_model
    assert cfg2.dpo.beta == cfg.dpo.beta
    assert cfg2.implicit_cot.delta_per_epoch == cfg.implicit_cot.delta_per_epoch
