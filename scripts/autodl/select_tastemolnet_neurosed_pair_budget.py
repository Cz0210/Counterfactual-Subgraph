#!/usr/bin/env python3
"""Validate real 100/500/1000 GEDLIB reports and select 5k/10k/20k."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    combine_disjoint_benchmark_reports,
    select_fixed_pair_budget,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"{path} is not one JSON object")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-100", type=Path, required=True)
    parser.add_argument("--benchmark-500", type=Path, required=True)
    parser.add_argument("--benchmark-1000", type=Path, required=True)
    parser.add_argument("--selected-workers", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--disk-reservation-pass", action="store_true")
    parser.add_argument("--cpu-contention-gate-pass", action="store_true")
    parser.add_argument("--maximum-label-hours", type=float, default=24.0)
    parser.add_argument("--timeout-rate-maximum", type=float, default=0.05)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("pair-budget output directory is not fresh")
    reports = {
        100: _load(args.benchmark_100),
        500: _load(args.benchmark_500),
        1000: _load(args.benchmark_1000),
    }
    summary = combine_disjoint_benchmark_reports(reports)
    if summary["status"] != "PASS":
        _atomic_json(args.output_dir / "gedlib_benchmark_summary.json", summary)
        print("BLOCKED_GEDLIB_THROUGHPUT", file=sys.stderr)
        return 78
    plan = select_fixed_pair_budget(
        reports[1000],
        selected_workers=args.selected_workers,
        disk_reservation_pass=args.disk_reservation_pass,
        cpu_contention_gate_pass=args.cpu_contention_gate_pass,
        maximum_label_hours=args.maximum_label_hours,
        timeout_rate_maximum=args.timeout_rate_maximum,
        safety_factor=args.safety_factor,
    )
    _atomic_json(args.output_dir / "gedlib_benchmark_summary.json", summary)
    _atomic_json(args.output_dir / "neurosed_pair_budget_plan.json", plan)
    if plan["status"] != "PASS":
        print("BLOCKED_GEDLIB_THROUGHPUT", file=sys.stderr)
        for budget in (5000, 10000, 20000):
            hours = plan["projections"][str(budget)]["projected_label_hours"]
            print(f"projected_{budget}_label_hours={hours:.6f}", file=sys.stderr)
        return 78
    print("[TASTE_NEUROSED_PAIR_BUDGET_SELECTED]")
    print(
        "selected_neurosed_train_pair_budget="
        f"{plan['selected_neurosed_train_pair_budget']}"
    )
    print(
        "selected_neurosed_validation_pair_budget="
        f"{plan['selected_neurosed_validation_pair_budget']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
