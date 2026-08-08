#!/usr/bin/env python3
"""Audit four method artifacts against one frozen cross-scaffold protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bbbp_paper_artifacts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    FIGURE3_FIELDS,
    FIGURE4_FIELDS,
    TABLE2_FIELDS,
)


METHODS = ("ours", "globalgce", "gcfexplainer", "comrecgc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split-audit", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), [dict(row) for row in reader]


def audit(root: Path, split_audit: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    leakage = json.loads(split_audit.read_text(encoding="utf-8"))
    if leakage.get("passed") is not True or leakage.get("scaffold_overlap_count", 0) != 0:
        raise ValueError("Cross-scaffold evaluation requires a passing zero-overlap split audit.")
    results: list[dict[str, Any]] = []
    for method in methods:
        method_root = root / method
        paths = (
            method_root / "figure3_coverage_vs_k.csv",
            method_root / "figure4_coverage_vs_threshold.csv",
            method_root / f"table2_{method}_k10.csv",
            method_root / "summary.json",
            method_root / "protocol_manifest.json",
            method_root / "final_artifact_audit.json",
        )
        missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Cross-scaffold {method} artifacts missing: {missing}")
        fields3, rows3 = _csv(paths[0])
        fields4, rows4 = _csv(paths[1])
        fields2, rows2 = _csv(paths[2])
        if fields3 != FIGURE3_FIELDS or fields4 != FIGURE4_FIELDS or fields2 != TABLE2_FIELDS:
            raise ValueError(f"Cross-scaffold artifact schema changed for {method}.")
        if [int(row["k"]) for row in rows3] != list(range(1, 21)):
            raise ValueError(f"Cross-scaffold Figure 3 K grid changed for {method}.")
        coverages = [float(row["coverage"]) for row in rows3]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coverages):
            raise ValueError(f"Cross-scaffold coverage invalid for {method}.")
        protocol = json.loads(paths[4].read_text(encoding="utf-8"))
        if protocol.get("threshold_source") != "calibration" or protocol.get("test_usage") != "final_evaluation_only":
            raise ValueError(f"Cross-scaffold protocol leakage for {method}.")
        summary = json.loads(paths[3].read_text(encoding="utf-8"))
        results.append({"method": method, "k10": rows2[0], "summary": summary})
    return {
        "schema_version": "cross_scaffold_common4_audit_v1",
        "passed": True,
        "dataset": "BBBP",
        "methods": list(methods),
        "cf_mode": CF_MODE,
        "distance_line": DISTANCE_LINE,
        "scaffold_overlap_count": 0,
        "test_usage": "final_evaluation_only",
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not methods or sorted(set(methods) - set(METHODS)):
        raise ValueError(f"Cross-scaffold methods must be drawn from {METHODS}.")
    result = audit(
        Path(args.root).expanduser().resolve(),
        Path(args.split_audit).expanduser().resolve(),
        methods,
    )
    if args.validate_only or args.dry_run:
        print(json.dumps({**result, "status":"VALIDATED_NOT_RUN", "formal_output_written":False}, sort_keys=True))
        return 0
    output = Path(args.output_json).expanduser().resolve() if args.output_json else Path(args.root).expanduser().resolve() / "cross_scaffold_summary.json"
    if output.exists():
        raise FileExistsError(f"Cross-scaffold audit output exists: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[CROSS_SCAFFOLD_AUDIT_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
