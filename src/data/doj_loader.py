"""DOJ 原始新闻稿加载与案情要素抽取。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..utils.logging import get_logger

log = get_logger("doj_loader")


@dataclass
class DOJRecord:
    url: str
    title: str
    date: str
    tags: list[str]
    summary: str
    body: str

    @classmethod
    def from_dict(cls, d: dict) -> "DOJRecord":
        return cls(
            url=d.get("url", ""),
            title=d.get("title", ""),
            date=d.get("date", ""),
            tags=list(d.get("tags", [])),
            summary=d.get("summary", ""),
            body=d.get("body", ""),
        )

    @property
    def combined_text(self) -> str:
        return f"{self.title}\n{self.summary}\n{self.body}"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_doj_records(path: str | Path, limit: int | None = None) -> list[DOJRecord]:
    records: list[DOJRecord] = []
    for i, d in enumerate(iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        records.append(DOJRecord.from_dict(d))
    log.info(f"从 {path} 加载 {len(records)} 条 DOJ 记录")
    return records


def extract_case_elements(record: DOJRecord) -> dict:
    """从 title + summary 中提取犯罪类型种子（粗抽取，供 Teacher LLM 改写）。"""
    text = (record.title + " " + record.summary).lower()
    crime_types: list[str] = []
    type_keywords = {
        "Cyber": ["cyber", "computer intrusion", "ransomware", "hacking", "scattered spider"],
        "Narcotics": ["fentanyl", "cocaine", "heroin", "methamphetamine", "drug trafficking", "narcotics"],
        "Fraud": ["fraud", "scheme", "defraud", "wire fraud", "mail fraud", "securities"],
        "Violence": ["murder", "assault", "robbery", "carjacking", "kidnapping", "arson"],
        "Firearms": ["firearm", "gun trafficking", "armed"],
        "Trafficking": ["sex trafficking", "child exploitation", "csam", "forced labor", "human smuggling"],
        "NationalSecurity": ["terroris", "espionage", "material support", "fto"],
        "Financial": ["money laundering", "bribery", "kickback", "embezzlement", "extortion"],
    }
    for ctype, kws in type_keywords.items():
        if any(kw in text for kw in kws):
            crime_types.append(ctype)
    return {
        "url": record.url,
        "title": record.title,
        "summary": record.summary,
        "date": record.date,
        "crime_types": crime_types or ["Other"],
    }
