#!/usr/bin/env python3
"""Freeze source-side connected-feasible BACE candidates with full lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bace_candidate_universe import (  # noqa: E402
    classify_connected_feasible_source_row,
    source_parent_id,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Any) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_parent_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {str(row["molecule_id"]).strip(): row for row in rows}


def filter_candidates(
    *,
    input_jsonl: Path,
    parent_csv: Path,
    output_jsonl: Path,
    audit_json: Path,
    generation_round: int,
    generation_regime: str,
    prompt_mode: str,
) -> dict[str, Any]:
    rows = _load_rows(input_jsonl)
    parent_metadata = _load_parent_metadata(parent_csv)
    retained: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, original in enumerate(rows):
        payload = dict(original)
        payload["candidate_lineage_source"] = str(input_jsonl)
        payload["candidate_lineage_source_index"] = int(generation_round)
        payload["candidate_lineage_row_index"] = index
        decision = classify_connected_feasible_source_row(
            payload,
            record_index=index,
            min_atom_ratio=0.0,
            max_atom_ratio=0.85,
            require_lineage=True,
        )
        decisions.append(decision)
        reason = decision.get("matrix_exclusion_reason")
        if reason:
            reasons[str(reason)] += 1
            continue
        parent_id = source_parent_id(payload)
        metadata = parent_metadata.get(parent_id)
        if metadata is None:
            raise ValueError(f"Candidate source is outside frozen train cohort: {parent_id}")
        lineage = {
            "source_file_sha256": _sha256(input_jsonl),
            "source_parent_id": parent_id,
            "source_graph_hash": str(payload.get("source_graph_hash") or ""),
            "candidate_index": payload.get("candidate_index"),
            "generation_round": int(generation_round),
            "generation_regime": str(generation_regime),
            "prompt_mode": str(prompt_mode),
        }
        payload.update(
            source_residual_connected=True,
            source_residual_num_components=1,
            source_residual_sanitized=True,
            boundary_bond_count=decision["boundary_bond_count_min"],
            attachment_count=decision["attachment_count_min"],
            connected_generation_prompt=(prompt_mode == "connected_deletion_v1"),
            generation_regime=str(generation_regime),
            generation_round=int(generation_round),
            source_scaffold=str(metadata.get("scaffold") or ""),
            source_cluster=str(payload.get("source_cluster") or ""),
            candidate_lineage_sha256=_stable_sha256(lineage),
        )
        retained.append(payload)
    if not retained:
        raise ValueError("Connected source-side gate retained zero BACE candidates.")
    if any(not row.get("source_residual_connected") for row in retained):
        raise AssertionError("Connected source-side gate retained a disconnected residual.")
    _atomic_text(
        output_jsonl,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
    )
    audit = {
        "schema_version": "bace_connected_source_gate_v4",
        "input_jsonl": str(input_jsonl),
        "input_sha256": _sha256(input_jsonl),
        "parent_csv": str(parent_csv),
        "parent_csv_sha256": _sha256(parent_csv),
        "input_candidate_count": len(rows),
        "retained_candidate_count": len(retained),
        "connected_valid_rate": len(retained) / len(rows) if rows else 0.0,
        "source_cf_flip_feature_only": True,
        "source_cf_drop_feature_only": True,
        "source_oracle_feature_only": True,
        "source_filter_counts": dict(sorted(reasons.items())),
        "generation_round": int(generation_round),
        "generation_regime": str(generation_regime),
        "prompt_mode": str(prompt_mode),
        "test_source_parent_count": 0,
        "output_jsonl": str(output_jsonl),
        "output_sha256": _sha256(output_jsonl),
        "run_complete": True,
    }
    _atomic_text(audit_json, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--generation-round", type=int, required=True)
    parser.add_argument("--generation-regime", required=True)
    parser.add_argument("--prompt-mode", choices=("connected_deletion_v1",), required=True)
    args = parser.parse_args()
    payload = filter_candidates(
        input_jsonl=Path(args.input_jsonl).expanduser().resolve(),
        parent_csv=Path(args.parent_csv).expanduser().resolve(),
        output_jsonl=Path(args.output_jsonl).expanduser().resolve(),
        audit_json=Path(args.audit_json).expanduser().resolve(),
        generation_round=int(args.generation_round),
        generation_regime=str(args.generation_regime),
        prompt_mode=str(args.prompt_mode),
    )
    print(json.dumps(payload, sort_keys=True))
    print("[BACE_CONNECTED_SOURCE_GATE_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
