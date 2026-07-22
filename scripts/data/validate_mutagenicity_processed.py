#!/usr/bin/env python3
"""Validate the complete processed Mutagenicity benchmark and split contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity import processed_validation, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--processed-dir", default="data/processed/Mutagenicity/v1")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--report-txt", default=None)
    return parser


def _report(summary: dict[str, Any]) -> str:
    lines = ["Mutagenicity processed validation", ""]
    lines.extend(f"{key}={value}" for key, value in sorted(summary.items()) if key != "errors")
    lines.append(f"errors={summary['errors']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed_dir = Path(args.processed_dir)
    summary, errors = processed_validation(processed_dir)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else processed_dir / "validation_summary.json"
    )
    report_path = (
        Path(args.report_txt) if args.report_txt else processed_dir / "validation_report.txt"
    )
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if errors:
        print("[MUTAGENICITY_PROCESSED_VALIDATE_FAILED]", flush=True)
        return 1
    print("[MUTAGENICITY_PROCESSED_VALIDATE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
