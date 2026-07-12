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

"""为实验和代码归档生成 SHA-256 provenance manifest。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.env import PROJECT_ROOT


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with open(path, "rb") as file:
            return sum(1 for line in file if line.strip())
    if path.suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as file:
            return sum(1 for _ in csv.DictReader(file))
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file)
        return len(value) if isinstance(value, (list, dict)) else None
    return None


def artifact_entry(path: Path, role: str, distributed: bool = True) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"工件不存在: {resolved}")
    try:
        relative_path = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = resolved
    return {
        "path": relative_path.as_posix(),
        "role": role,
        "distribution": "included" if distributed else "not_distributed",
        "sha256": hash_file(resolved),
        "bytes": resolved.stat().st_size,
        "records": count_records(resolved),
    }


def git_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def write_manifest(output_path: Path, artifacts: list[dict]) -> dict:
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "external_resources": [
            {
                "role": "student_base",
                "repo_id": "Qwen/Qwen3-8B",
                "resolved_revision": None,
                "revision_status": "not_recorded_for_existing_run",
            },
            {
                "role": "safe_stress_dataset",
                "repo_id": "allenai/WildChat-nontoxic",
                "resolved_revision": None,
                "revision_status": "not_recorded_for_existing_run",
            },
            {"role": "teacher", "provider": "unknown", "model": "not_recorded"},
            {"role": "preference_judge", "provider": "unknown", "model": "not_recorded"},
        ],
        "artifacts": artifacts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temp_path.replace(output_path)
    return manifest


def parse_artifact(value: str) -> tuple[Path, str, bool]:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("格式必须为 path:role[:included|not_distributed]")
    distributed = len(parts) == 2 or parts[2] == "included"
    if len(parts) == 3 and parts[2] not in {"included", "not_distributed"}:
        raise argparse.ArgumentTypeError("distribution 必须是 included 或 not_distributed")
    return Path(parts[0]), parts[1], distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验工件 SHA-256 provenance manifest")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/provenance.json")
    parser.add_argument("--artifact", action="append", type=parse_artifact, default=[])
    args = parser.parse_args()
    requested = args.artifact or [
        (PROJECT_ROOT / "configs/default.yaml", "experiment_config", True),
        (PROJECT_ROOT / "requirements.txt", "dependency_lock", True),
        (PROJECT_ROOT / "data/raw/doj_raw.jsonl", "raw_doj_input", True),
        (PROJECT_ROOT / "data/synthesized/train.jsonl", "sft_train", True),
        (PROJECT_ROOT / "data/synthesized/test.jsonl", "held_out_synthetic", True),
        (PROJECT_ROOT / "data/synthesized/hard_negatives.jsonl", "hard_negatives", True),
    ]
    entries = [artifact_entry(path if path.is_absolute() else PROJECT_ROOT / path, role, distributed)
               for path, role, distributed in requested]
    manifest = write_manifest(args.output.resolve(), entries)
    print(json.dumps({"output": str(args.output), "artifacts": len(manifest["artifacts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()