"""将爬虫输出校验并导入为规范 DOJ 原始数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.utils.env import PROJECT_ROOT


def inspect_jsonl(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    records = 0
    byte_count = 0
    with open(path, "rb") as file:
        for line_number, line in enumerate(file, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"无效 JSONL: {path}:{line_number}: {error}") from error
            records += 1
    return {"sha256": digest.hexdigest(), "bytes": byte_count, "records": records}


def import_doj_raw(source: Path, destination: Path, overwrite: bool = False) -> dict[str, int | str]:
    if not source.is_file():
        raise FileNotFoundError(f"DOJ 源文件不存在: {source}")
    source_info = inspect_jsonl(source)
    if destination.exists():
        destination_info = inspect_jsonl(destination)
        if destination_info == source_info:
            return source_info
        if not overwrite:
            raise FileExistsError(f"目标已存在且内容不同: {destination}; 使用 --overwrite 显式覆盖")

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        if inspect_jsonl(temp_path) != source_info:
            raise RuntimeError("复制后的 DOJ 数据校验失败")
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return source_info


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 crawler 输出并导入 data/raw 规范路径")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "crawler/output/doj_raw.jsonl")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "data/raw/doj_raw.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖内容不同的目标文件")
    args = parser.parse_args()
    info = import_doj_raw(args.source.resolve(), args.destination.resolve(), args.overwrite)
    print(json.dumps({"source": str(args.source), "destination": str(args.destination), **info}, ensure_ascii=False))


if __name__ == "__main__":
    main()