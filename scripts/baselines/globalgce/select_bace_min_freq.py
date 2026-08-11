#!/usr/bin/env python3
"""Freeze BACE GlobalGCE min_freq from calibration-only metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_min_freq import (  # noqa: E402
    bace_min_freq_grid,
    read_calibration_metrics,
    select_bace_min_freq,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-train-parent-count", type=int, required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_grid = bace_min_freq_grid(args.source_train_parent_count)
    metrics = Path(args.metrics_csv).expanduser().resolve()
    teacher = Path(args.teacher_path).expanduser().resolve()
    thresholds = Path(args.thresholds_json).expanduser().resolve()
    for path in (metrics, teacher, thresholds):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    rows = read_calibration_metrics(metrics)
    observed = tuple(sorted(int(row["min_freq"]) for row in rows))
    if observed != expected_grid:
        raise ValueError(
            f"Calibration grid mismatch: actual={observed}, expected={expected_grid}."
        )
    selected = select_bace_min_freq(rows)
    payload: dict[str, object] = {
        "schema_version": "bace_globalgce_min_freq_calibration_v1",
        "dataset": "BACE",
        "selected_min_freq": int(selected["min_freq"]),
        "selected_pool_path": str(selected["pool_path"]),
        "candidate_grid": list(expected_grid),
        "selection_split": "calibration",
        "selection_rule": [
            "max prefix_auc_k1_k10",
            "max multi_threshold_prefix_auc",
            "lower cost",
            "lower coverage_redundancy",
            "fewer rules",
            "lower min_freq",
        ],
        "test_loaded": False,
        "teacher_sha256": _sha(teacher),
        "threshold_sha256": _sha(thresholds),
        "metrics_sha256": _sha(metrics),
        "git_commit": str(args.git_commit),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.validate_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("[BACE_GLOBALGCE_MIN_FREQ_VALIDATE_OK]")
        return 0
    output = Path(args.output_dir).expanduser().resolve()
    if (output / "globalgce_bace_min_freq_manifest.json").exists():
        raise FileExistsError(output / "globalgce_bace_min_freq_manifest.json")
    _write(output / "globalgce_bace_min_freq_manifest.json", payload)
    _write(
        output / "min_freq_selection_audit.json",
        {**payload, "passed": True, "candidate_rows": rows},
    )
    with (output / "min_freq_candidates.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("min_freq", "ratio"))
        writer.writeheader()
        for value in expected_grid:
            writer.writerow(
                {
                    "min_freq": value,
                    "ratio": value / int(args.source_train_parent_count),
                }
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("[BACE_GLOBALGCE_MIN_FREQ_SELECTION_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
