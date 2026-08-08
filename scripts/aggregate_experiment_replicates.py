#!/usr/bin/env python3
"""Aggregate multi-seed parent metrics with parent-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bootstrap_metrics import DEFAULT_METRICS, parent_level_bootstrap  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--parent-metrics-csv", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Replicate parent metrics are empty: {path}")
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def aggregate(paths: list[Path], seeds: list[int], *, samples: int, bootstrap_seed: int) -> dict[str, Any]:
    if len(paths) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Replicate CSVs and unique seed values must have equal length.")
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError("BBBP multi-seed protocol is frozen to seeds 0,1,2.")
    replicate_results: list[dict[str, Any]] = []
    for path, seed in zip(paths, seeds, strict=True):
        rows = _read(path)
        replicate_results.append(
            {
                "seed": seed,
                "path": str(path),
                "num_parents": len(rows),
                "bootstrap": parent_level_bootstrap(
                    rows,
                    metrics=DEFAULT_METRICS,
                    num_samples=samples,
                    seed=bootstrap_seed + seed,
                ),
            }
        )
    return {
        "schema_version": "multi_seed_parent_bootstrap_v1",
        "status": "COMPLETE",
        "seeds": seeds,
        "resampling_unit": "parent_id",
        "pair_row_bootstrap": False,
        "bootstrap_samples": samples,
        "replicates": replicate_results,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [Path(value).expanduser().resolve() for value in args.parent_metrics_csv]
    result = aggregate(paths, args.seed, samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    if args.validate_only or args.dry_run:
        print(json.dumps({**result, "status":"VALIDATED_NOT_RUN", "formal_output_written":False}, sort_keys=True))
        return 0
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Replicate aggregate output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "bootstrap_summary.json", result)
    _atomic_json(output / "replicate_summary.json", result)
    print("[REPLICATE_BOOTSTRAP_AGGREGATE_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
