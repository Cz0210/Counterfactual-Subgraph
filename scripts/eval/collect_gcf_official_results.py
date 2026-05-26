#!/usr/bin/env python3
"""Collect coverage/cost metrics from official GCFExplainer summary logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
K_RE = re.compile(r"\b(?:top|k|size|summary\s*size)\s*[-_:=# ]*\s*(\d+)\b", re.IGNORECASE)
TOP_VALUE_RE = re.compile(r"\bTop\s+(\d+)\s*:\s*(" + FLOAT_RE + r")\b", re.IGNORECASE)


@dataclass
class MetricRecord:
    k: int
    values: dict[str, float] = field(default_factory=dict)
    lines: dict[str, str] = field(default_factory=dict)


def _extract_named_metric(line: str, name: str) -> float | None:
    pattern = re.compile(
        r"\b" + re.escape(name) + r"\b\s*(?:[:=]|\bis\b)?\s*(" + FLOAT_RE + r")",
        re.IGNORECASE,
    )
    match = pattern.search(line)
    if not match:
        return None
    return float(match.group(1))


def _extract_k(line: str) -> int | None:
    match = K_RE.search(line)
    if match:
        return int(match.group(1))
    return None


def parse_summary_log_text(text: str) -> tuple[list[dict[str, object]], str]:
    """Parse common official summary formats without assuming a fixed layout.

    The upstream AIDS summary currently prints unlabeled ``Top k: value`` lines
    twice: once after a coverage section header and once after ``Calculating
    cost...``. This parser also accepts more explicit lines containing
    coverage/cost names, k/top/size markers, and either ``=`` or ``:`` values.
    """

    records: dict[int, MetricRecord] = {}
    current_section: str | None = None
    errors: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        has_coverage = "coverage" in lowered
        has_cost = "cost" in lowered

        if has_coverage and not has_cost:
            current_section = "coverage"
        elif has_cost and not has_coverage:
            current_section = "cost"

        k_value = _extract_k(line)
        if k_value is not None:
            record = records.setdefault(k_value, MetricRecord(k=k_value))
            for metric_name in ("coverage", "cost"):
                metric_value = _extract_named_metric(line, metric_name)
                if metric_value is not None:
                    record.values[metric_name] = metric_value
                    record.lines[metric_name] = line

        top_match = TOP_VALUE_RE.search(line)
        if top_match:
            top_k = int(top_match.group(1))
            top_value = float(top_match.group(2))
            if current_section in {"coverage", "cost"}:
                record = records.setdefault(top_k, MetricRecord(k=top_k))
                record.values[current_section] = top_value
                record.lines[current_section] = line
            else:
                errors.append(f"Could not assign unlabeled Top line to coverage/cost section: {line}")

    rows: list[dict[str, object]] = []
    for k_value in sorted(records):
        record = records[k_value]
        if "coverage" not in record.values and "cost" not in record.values:
            continue
        rows.append(
            {
                "k": record.k,
                "coverage": record.values.get("coverage"),
                "cost": record.values.get("cost"),
                "coverage_line": record.lines.get("coverage", ""),
                "cost_line": record.lines.get("cost", ""),
            }
        )

    if not rows:
        message = "No coverage/cost metrics could be parsed from summary log."
        if errors:
            message += " " + " ".join(errors[:3])
        return rows, message
    if errors:
        return rows, "Parsed metrics with warnings: " + " ".join(errors[:3])
    return rows, ""


def collect_summary(summary_log: Path) -> dict[str, object]:
    raw_path = str(summary_log)
    if not summary_log.exists():
        return {
            "raw_summary_log_path": raw_path,
            "parse_ok": False,
            "error_message": f"summary log not found: {summary_log}",
            "rows": [],
        }

    try:
        text = summary_log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "raw_summary_log_path": raw_path,
            "parse_ok": False,
            "error_message": f"failed to read summary log: {exc}",
            "rows": [],
        }

    rows, parse_message = parse_summary_log_text(text)
    return {
        "raw_summary_log_path": raw_path,
        "parse_ok": bool(rows),
        "error_message": "" if rows else parse_message,
        "parse_warning": parse_message if rows and parse_message else "",
        "rows": rows,
    }


def write_json(payload: dict[str, object], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_rows(payload: dict[str, object]) -> Iterable[dict[str, object]]:
    rows = payload.get("rows", [])
    if isinstance(rows, list) and rows:
        for row in rows:
            if isinstance(row, dict):
                yield {
                    "k": row.get("k", ""),
                    "coverage": row.get("coverage", ""),
                    "cost": row.get("cost", ""),
                    "coverage_line": row.get("coverage_line", ""),
                    "cost_line": row.get("cost_line", ""),
                    "raw_summary_log_path": payload.get("raw_summary_log_path", ""),
                    "parse_ok": payload.get("parse_ok", False),
                    "error_message": payload.get("error_message", ""),
                    "parse_warning": payload.get("parse_warning", ""),
                }
        return

    yield {
        "k": "",
        "coverage": "",
        "cost": "",
        "coverage_line": "",
        "cost_line": "",
        "raw_summary_log_path": payload.get("raw_summary_log_path", ""),
        "parse_ok": payload.get("parse_ok", False),
        "error_message": payload.get("error_message", ""),
        "parse_warning": payload.get("parse_warning", ""),
    }


def write_csv(payload: dict[str, object], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k",
        "coverage",
        "cost",
        "coverage_line",
        "cost_line",
        "raw_summary_log_path",
        "parse_ok",
        "error_message",
        "parse_warning",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _csv_rows(payload):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect official GCFExplainer summary metrics.")
    parser.add_argument("--summary-log", required=True, type=Path, help="Path to summary.log from summary.py.")
    parser.add_argument("--out-json", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = collect_summary(args.summary_log)
    write_json(payload, args.out_json)
    write_csv(payload, args.out_csv)

    if payload.get("parse_ok"):
        print(f"[GCF_COLLECT] parse_ok=true rows={len(payload.get('rows', []))}")
    else:
        print(f"[GCF_COLLECT][WARN] parse_ok=false error={payload.get('error_message')}", file=sys.stderr)
    print(f"[GCF_COLLECT] wrote_json={args.out_json}")
    print(f"[GCF_COLLECT] wrote_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
