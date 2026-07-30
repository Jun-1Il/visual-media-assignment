from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def normalize_prediction(raw: str, task: str) -> str:
    text = raw.upper().strip()
    if task == "code":
        if "NONE" in text:
            return ""
        tokens = re.findall(r"(?<![A-Z0-9])([A-Z0-9]{4})(?![A-Z0-9])", text)
        stopwords = {"CODE", "THIS", "THAT", "READ", "ONLY", "WITH", "FROM"}
        candidates = [token for token in tokens if token not in stopwords]
        if candidates:
            return candidates[-1]
        compact = re.sub(r"[^A-Z0-9]", "", text)
        return compact[:4]
    matches = re.findall(r"(?<!\d)\d{1,2}(?!\d)", text)
    # Models sometimes emit a short equation despite the "answer only"
    # instruction (for example, "4 + 5 = 9"). The final integer is the answer.
    return matches[-1] if matches else ""


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def summarize(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["variant"]),
                str(row["domain"]),
                str(row["condition"]),
            )
        ].append(int(row["correct"]))

    summary = []
    for (variant, domain, condition), values in sorted(groups.items()):
        correct = sum(values)
        count = len(values)
        summary.append(
            {
                "variant": variant,
                "domain": domain,
                "condition": condition,
                "correct": correct,
                "count": count,
                "accuracy": correct / count if count else 0.0,
            }
        )
    return summary


def write_summary_json(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    by_variant: dict[str, list[int]] = defaultdict(list)
    for row in materialized:
        by_variant[str(row["variant"])].append(int(row["correct"]))
    payload = {
        "overall": {
            variant: {
                "correct": sum(values),
                "count": len(values),
                "accuracy": sum(values) / len(values) if values else 0.0,
            }
            for variant, values in sorted(by_variant.items())
        },
        "by_condition": summarize(materialized),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
