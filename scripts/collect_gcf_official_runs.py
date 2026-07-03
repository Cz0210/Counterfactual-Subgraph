#!/usr/bin/env python3
"""Collect official GCFExplainer native eval runs and select the best alpha."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--runs-root", default="outputs/hpc/gcfexplainer_official")
    parser.add_argument("--summary-glob", default="**/native_ccrcov_summary.csv")
    parser.add_argument("--out-dir", default="outputs/hpc/gcfexplainer_official/final")
    parser.add_argument("--select-k", type=int, default=10)
    parser.add_argument("--select-theta", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob(args.summary_glob)):
        for row in _read_csv(path):
            run_dir = str(path.parent)
            alpha = row.get("alpha", "")
            if not alpha:
                for part in path.parts:
                    if part.startswith("alpha_"):
                        alpha = part.split("_", 1)[1]
                        break
            rows.append(
                {
                    "alpha": alpha,
                    "k": row.get("k", ""),
                    "theta": row.get("theta", ""),
                    "coverage": row.get("coverage", ""),
                    "median_cost": row.get("median_cost", ""),
                    "avg_cost": row.get("avg_cost", ""),
                    "run_dir": run_dir,
                    "selected_graphs_path": row.get("selected_graphs_path", ""),
                    "summary_path": str(path),
                }
            )
    target_rows = [
        row
        for row in rows
        if _int(row.get("k")) == int(args.select_k)
        and abs(_float(row.get("theta")) - float(args.select_theta)) < 1e-12
    ]
    best = max(target_rows, key=lambda row: _float(row.get("coverage")), default=None)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "gcf_official_all_runs.csv"
    _write_csv(
        all_csv,
        rows,
        ["alpha", "k", "theta", "coverage", "median_cost", "avg_cost", "run_dir", "selected_graphs_path", "summary_path"],
    )
    summary = {
        "selection_rule": f"max coverage at K={args.select_k}, theta={args.select_theta}",
        "num_rows": len(rows),
        "num_target_rows": len(target_rows),
        "best": best,
        "all_runs_csv": str(all_csv),
    }
    best_json = out_dir / "gcf_official_best_summary.json"
    best_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[GCF_OFFICIAL_COLLECT_DONE]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

