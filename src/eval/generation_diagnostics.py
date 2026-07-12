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

"""对已生成的 CoT 进行严格、只读的标签诊断。"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import compute_binary_metrics, label_to_int

FINAL_LABEL_RE = re.compile(r"(?:^|\n)\s*(Threat|Safe)\s*[.!]?\s*$", re.IGNORECASE)


def parse_final_label(text: str) -> str | None:
    """仅接受输出末尾独立出现的唯一最终标签。"""
    matches = FINAL_LABEL_RE.findall(text or "")
    labels = {match.lower() for match in matches}
    if len(labels) != 1:
        return None
    return "Threat" if labels.pop() == "threat" else "Safe"


def diagnose_predictions(predictions: list[dict]) -> dict:
    parsed: list[tuple[dict, str]] = []
    invalid = 0
    disagreements = 0
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for prediction in predictions:
        generated_label = parse_final_label(str(prediction.get("model_cot", "")))
        if generated_label is None:
            invalid += 1
            continue
        parsed.append((prediction, generated_label))
        if generated_label != prediction.get("model_label", "Safe"):
            disagreements += 1
        by_source[str(prediction.get("source", "unknown"))].append(
            (label_to_int(generated_label), label_to_int(prediction.get("ref_label", "Safe")))
        )

    preds = [label_to_int(label) for _, label in parsed]
    labels = [label_to_int(prediction.get("ref_label", "Safe")) for prediction, _ in parsed]
    metrics = compute_binary_metrics(preds, labels).as_dict()
    source_metrics = {
        source: {"n": len(items), **compute_binary_metrics(
            [pred for pred, _ in items], [label for _, label in items]
        ).as_dict()}
        for source, items in sorted(by_source.items())
    }
    return {
        "n_total": len(predictions),
        "n_parsed": len(parsed),
        "n_invalid": invalid,
        "coverage": len(parsed) / len(predictions) if predictions else 0.0,
        "disagreement_rate_on_parsed": disagreements / len(parsed) if parsed else 0.0,
        "generated_label_counts": dict(Counter(label for _, label in parsed)),
        "metrics_on_parsed": metrics,
        "metrics_by_source": source_metrics,
    }


def diagnose_files(paths: list[str | Path], output_path: str | Path) -> dict:
    report: dict[str, dict] = {}
    for path_value in paths:
        path = Path(path_value)
        with open(path, "r", encoding="utf-8") as file:
            report[path.stem.removeprefix("predictions_")] = diagnose_predictions(json.load(file))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report