#!/usr/bin/env python3
"""Merge preregistered BACE candidate regimes without test leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.molclr_node_embeddings import canonicalize_smiles


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object in {path}")
                result.append(payload)
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parent_id(row: dict[str, Any]) -> str:
    for field in ("molecule_id", "parent_id", "parent_index", "source_parent_id"):
        if str(row.get(field) or "").strip():
            return str(row[field]).strip()
    raise ValueError("Candidate row lacks a stable parent ID")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--base-pool", required=True)
    parser.add_argument("--regime-pool", action="append", default=[])
    parser.add_argument("--train-parent-ids", required=True)
    parser.add_argument("--test-parent-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = [Path(args.base_pool).expanduser().resolve()] + [
        Path(value).expanduser().resolve() for value in args.regime_pool
    ]
    train_ids = {
        line.strip()
        for line in Path(args.train_parent_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_ids = {
        line.strip()
        for line in Path(args.test_parent_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if train_ids & test_ids:
        raise ValueError("Train and test candidate-source ID sets overlap")
    chosen: dict[tuple[str, str], tuple[tuple[Any, ...], dict[str, Any]]] = {}
    input_count = 0
    for source_index, source in enumerate(sources):
        for row_index, row in enumerate(_rows(source)):
            input_count += 1
            parent_id = _parent_id(row)
            if parent_id not in train_ids or parent_id in test_ids:
                raise ValueError(f"Candidate source is outside frozen train cohort: {parent_id}")
            fragment = canonicalize_smiles(str(row.get("final_fragment") or ""))
            if not fragment:
                continue
            payload = dict(row)
            payload["final_fragment"] = fragment
            payload["candidate_lineage_source"] = str(source)
            payload["candidate_lineage_source_index"] = source_index
            payload["candidate_lineage_row_index"] = row_index
            key = (parent_id, fragment)
            score = (
                int(_bool(row.get("cf_flip"))),
                _float(row.get("cf_drop"), float("-inf")),
                -_float(row.get("atom_ratio"), float("inf")),
                -source_index,
                -row_index,
            )
            if key not in chosen or score > chosen[key][0]:
                chosen[key] = (score, payload)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Merge output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = [chosen[key][1] for key in sorted(chosen)]
    pool = output / "candidate_pool.jsonl"
    pool.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "schema_version": "bace_ours_multiseed_pool_v2",
        "input_candidate_count": input_count,
        "candidate_count": len(rows),
        "unique_fragment_count": len({row["final_fragment"] for row in rows}),
        "parent_count": len({_parent_id(row) for row in rows}),
        "dedup_key": "parent_id + canonical final_fragment",
        "dedup_preference": [
            "strict_flip",
            "higher_cf_drop",
            "smaller_atom_ratio",
            "fixed_source_and_row_order",
        ],
        "input_files": [
            {"path": str(path), "sha256": _sha(path)} for path in sources
        ],
        "candidate_pool_sha256": _sha(pool),
        "test_parent_used": False,
        "run_complete": True,
    }
    (output / "candidate_pool_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_RUN_COMPLETE.json").write_text(
        json.dumps({"run_complete": True}) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    print("[BACE_OURS_MULTISEED_POOL_MERGE_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
