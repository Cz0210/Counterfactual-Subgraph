#!/usr/bin/env python3
"""Machine-select GEDLIB workers, then select the fixed NeuroSED pair budget."""

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
    LEGAL_GEDLIB_WORKER_COUNTS,
    build_gedlib_worker_selection_manifest,
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


def _physical_core_count() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        sockets: set[tuple[str, str]] = set()
        physical = "0"
        core = ""
        for line in cpuinfo.read_text(encoding="utf-8").splitlines() + [""]:
            if not line.strip():
                if core:
                    sockets.add((physical, core))
                physical, core = "0", ""
            elif line.startswith("physical id"):
                physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
        if sockets:
            return len(sockets)
    return int(os.cpu_count() or 1)


def _load_worker_reports(values: list[str]) -> dict[int, dict]:
    reports: dict[int, dict] = {}
    for value in values:
        worker_text, separator, path_text = value.partition("=")
        if not separator or not worker_text.isdigit() or not path_text:
            raise RuntimeError("--worker-benchmark must be WORKERS=/absolute/report.json")
        workers = int(worker_text)
        if workers not in LEGAL_GEDLIB_WORKER_COUNTS or workers in reports:
            raise RuntimeError("worker benchmark count is invalid or duplicated")
        path = Path(path_text)
        if not path.is_absolute() or path.is_symlink():
            raise RuntimeError("worker benchmark report must be absolute and non-symlink")
        reports[workers] = _load(path)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-100", type=Path, required=True)
    parser.add_argument("--benchmark-500", type=Path, required=True)
    parser.add_argument("--benchmark-1000", type=Path, required=True)
    parser.add_argument(
        "--worker-benchmark",
        action="append",
        required=True,
        metavar="WORKERS=/ABSOLUTE/REPORT.JSON",
        help="repeat for every physical-core-eligible worker count",
    )
    parser.add_argument("--disk-reservation-pass", action="store_true")
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
    worker_reports = _load_worker_reports(args.worker_benchmark)
    worker_selection = build_gedlib_worker_selection_manifest(
        worker_reports,
        physical_core_count=_physical_core_count(),
    )
    planning_pair_ids = {
        pair_id
        for report in reports.values()
        for pair_id in report["pair_ids"]
    }
    worker_pair_ids = {
        pair_id
        for report in worker_selection["reports"].values()
        for pair_id in report["pair_ids"]
    }
    if planning_pair_ids.intersection(worker_pair_ids):
        raise RuntimeError(
            "worker-selection trials reused a 100/500/1000 planning pair"
        )
    _atomic_json(args.output_dir / "gedlib_benchmark_summary.json", summary)
    _atomic_json(
        args.output_dir / "gedlib_worker_selection.json", worker_selection
    )
    if summary["status"] != "PASS":
        print("BLOCKED_GEDLIB_THROUGHPUT", file=sys.stderr)
        return 78
    if worker_selection["status"] != "PASS":
        print(worker_selection["status"], file=sys.stderr)
        return 78
    plan = select_fixed_pair_budget(
        reports[1000],
        worker_selection_manifest=worker_selection,
        disk_reservation_pass=args.disk_reservation_pass,
        maximum_label_hours=args.maximum_label_hours,
        timeout_rate_maximum=args.timeout_rate_maximum,
        safety_factor=args.safety_factor,
    )
    _atomic_json(args.output_dir / "neurosed_pair_budget_plan.json", plan)
    if plan["status"] != "PASS":
        print("BLOCKED_GEDLIB_THROUGHPUT", file=sys.stderr)
        for budget in (5000, 10000, 20000):
            hours = plan["projections"][str(budget)]["projected_label_hours"]
            print(f"projected_{budget}_label_hours={hours:.6f}", file=sys.stderr)
        return 78
    print("[TASTE_NEUROSED_PAIR_BUDGET_SELECTED]")
    print(f"selected_gedlib_workers={plan['selected_gedlib_workers']}")
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
