#!/usr/bin/env python3
"""Inspect embedding availability in a candidate_pool.jsonl file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.subgraph_similarity import parse_embedding  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for Slurm wrapper parity. This checker uses explicit CLI values.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept only for Slurm wrapper parity.",
    )
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--embedding-field", default="final_fragment_embedding")
    parser.add_argument("--max-rows", type=int, default=20)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at line {line_number}: {path}")
            rows.append(payload)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    pool_path = args.candidate_pool.expanduser().resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"candidate pool not found: {pool_path}")

    rows = _read_jsonl(pool_path)
    dimensions: Counter[int] = Counter()
    missing_count = 0
    present_count = 0
    invalid_count = 0
    nan_or_inf_count = 0
    examples: list[dict[str, Any]] = []
    invalid_examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        value = row.get(args.embedding_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_count += 1
            continue
        try:
            embedding = parse_embedding(value)
        except ValueError as exc:
            invalid_count += 1
            if "NaN" in str(exc) or "Inf" in str(exc):
                nan_or_inf_count += 1
            if len(invalid_examples) < max(1, int(args.max_rows)):
                invalid_examples.append(
                    {
                        "row_index": index,
                        "final_fragment": row.get("final_fragment"),
                        "reason": str(exc),
                    }
                )
            continue
        present_count += 1
        dimensions[int(embedding.size)] += 1
        if len(examples) < max(1, int(args.max_rows)):
            examples.append(
                {
                    "row_index": index,
                    "final_fragment": row.get("final_fragment"),
                    "embedding_dim": int(embedding.size),
                }
            )

    summary = {
        "candidate_pool": str(pool_path),
        "embedding_field": str(args.embedding_field),
        "total_rows": len(rows),
        "rows_with_embedding": present_count,
        "rows_missing_embedding": missing_count,
        "rows_invalid_embedding": invalid_count,
        "has_nan_or_inf": nan_or_inf_count > 0,
        "nan_or_inf_row_count": nan_or_inf_count,
        "embedding_dimension_distribution": {
            str(dimension): count for dimension, count in sorted(dimensions.items())
        },
        "examples": examples,
        "invalid_examples": invalid_examples,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if invalid_count > 0 or nan_or_inf_count > 0:
        return 2
    if missing_count > 0:
        return 1
    if not rows:
        return 1
    if not dimensions:
        return 1
    if any(not math.isfinite(float(key)) for key in dimensions):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
