#!/usr/bin/env python3
"""Audit BACE candidate attrition before calibration matrix construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bace_candidate_universe import (
    build_connected_feasible_candidate_universe,
    classify_connected_feasible_source_row,
    source_parent_id,
)
from src.eval.candidate_pool_audit import _normalize_row
from src.eval.class_counterfactual_selector import _is_failure_free
from src.eval.molclr_node_embeddings import canonicalize_smiles


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict, tuple))
                        else ("" if value is None else value)
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _old_matrix_fragments(matrix_root: Path) -> set[str]:
    for name in ("candidate_universe.jsonl", "selected_candidate_universe.jsonl"):
        path = matrix_root / name
        if not path.is_file():
            continue
        values = {
            canonical
            for row in _read_jsonl(path)
            if (
                canonical := canonicalize_smiles(
                    str(row.get("canonical_fragment") or row.get("fragment_smiles") or "")
                )
            )
        }
        if values:
            return values
    raise FileNotFoundError(f"No candidate universe under {matrix_root}")


def _legacy_exclusion_reason(
    source_rows: Sequence[dict[str, Any]],
    _decisions: Sequence[dict[str, Any]],
) -> str:
    remaining = [
        (row, _normalize_row(row, record_index=index))
        for index, row in enumerate(source_rows)
    ]
    stages = (
        ("excluded_other", lambda item: item[1].label == 1),
        ("excluded_parse", lambda item: bool(item[1].final_fragment)),
        ("excluded_substructure", lambda item: item[1].final_substructure),
        ("excluded_parse", lambda item: item[1].parse_ok and item[1].valid),
        ("excluded_parse", lambda item: item[1].connected),
    )
    for reason, predicate in stages:
        remaining = [item for item in remaining if predicate(item)]
        if not remaining:
            return reason
    remaining = [item for item in remaining if item[1].oracle_ok]
    if not remaining:
        return "excluded_source_oracle"
    remaining = [
        item
        for item in remaining
        if item[1].cf_drop is not None and float(item[1].cf_drop) >= 0.2
    ]
    if not remaining:
        return "excluded_source_low_cfdrop"
    remaining = [item for item in remaining if item[1].cf_flip]
    if not remaining:
        return "excluded_source_not_flip"
    remaining = [item for item in remaining if _is_failure_free(item[1].failure_tag)]
    if not remaining:
        return "excluded_other"
    remaining = [
        item
        for item in remaining
        if not (item[1].full_parent or item[1].near_parent or item[1].too_small)
    ]
    if not remaining:
        return "excluded_size"
    return "excluded_other"


def audit_candidate_universe(
    *,
    candidate_pool: Path,
    matrix_root: Path,
    teacher_path: Path,
    output_dir: Path,
    expected_pool_unique: int = 0,
    expected_old_matrix_candidates: int = 0,
) -> dict[str, Any]:
    pool_rows = _read_jsonl(candidate_pool)
    old_fragments = _old_matrix_fragments(matrix_root)
    decisions = [
        classify_connected_feasible_source_row(row, record_index=index)
        for index, row in enumerate(pool_rows)
    ]
    new_universe, new_stats, rebuilt_decisions = build_connected_feasible_candidate_universe(
        pool_rows
    )
    if decisions != rebuilt_decisions:
        raise AssertionError("Candidate attrition decisions are not deterministic.")
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, decision in zip(pool_rows, decisions, strict=True):
        canonical = decision.get("canonical_fragment") or str(row.get("final_fragment") or "")
        grouped_rows[str(canonical)].append(row)
        grouped_decisions[str(canonical)].append(decision)

    output_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for canonical in sorted(grouped_rows):
        rows = grouped_rows[canonical]
        fragment_decisions = grouped_decisions[canonical]
        entered_old = canonical in old_fragments
        entered_new = any(
            bool(item["entered_connected_feasible_universe"])
            for item in fragment_decisions
        )
        exclusion = None if entered_old else _legacy_exclusion_reason(rows, fragment_decisions)
        if exclusion:
            exclusion_counts[exclusion] += 1
        parent_ids = sorted({value for row in rows if (value := source_parent_id(row))})
        output_rows.append(
            {
                "candidate_id": "BACE_WNODE_"
                + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper(),
                "fragment_smiles": canonical,
                "source_parent_count": len(parent_ids),
                "source_parent_ids": parent_ids,
                "parse_ok": all(bool(item["parse_ok"]) for item in fragment_decisions),
                "connected_fragment": all(
                    bool(item["connected_fragment"]) for item in fragment_decisions
                ),
                "direct_substructure": any(
                    bool(item["direct_substructure"]) for item in fragment_decisions
                ),
                "final_substructure": any(
                    bool(item["final_substructure"]) for item in fragment_decisions
                ),
                "projection_used": any(
                    bool(item["projection_used"]) for item in fragment_decisions
                ),
                "source_residual_connected": any(
                    bool(item["source_residual_connected"]) for item in fragment_decisions
                ),
                "source_residual_sanitized": any(
                    bool(item["source_residual_sanitized"]) for item in fragment_decisions
                ),
                "source_cf_flip": any(bool(row.get("cf_flip")) for row in rows),
                "source_cf_drop": max(
                    (float(row["cf_drop"]) for row in rows if row.get("cf_drop") is not None),
                    default=None,
                ),
                "atom_ratio": min(
                    (
                        float(item["atom_ratio"])
                        for item in fragment_decisions
                        if item.get("atom_ratio") is not None
                    ),
                    default=None,
                ),
                "dedup_status": "canonical_aggregate" if len(rows) > 1 else "unique",
                "dedup_representative": min(
                    int(item["record_index"]) for item in fragment_decisions
                ),
                "entered_matrix": entered_old,
                "entered_connected_feasible_v4": entered_new,
                "matrix_exclusion_reason": exclusion,
                "v4_matrix_exclusion_reason": next(
                    (
                        str(item["matrix_exclusion_reason"])
                        for item in fragment_decisions
                        if item.get("matrix_exclusion_reason")
                    ),
                    None,
                )
                if not entered_new
                else None,
            }
        )

    pool_unique = len(grouped_rows)
    if expected_pool_unique and pool_unique != expected_pool_unique:
        raise AssertionError(
            f"Pool unique count changed: expected {expected_pool_unique}, found {pool_unique}"
        )
    if expected_old_matrix_candidates and len(old_fragments) != expected_old_matrix_candidates:
        raise AssertionError(
            "Old matrix candidate count changed: expected "
            f"{expected_old_matrix_candidates}, found {len(old_fragments)}"
        )
    stage_counts = {
        "UNIQUE_FRAGMENTS_IN_POOL": pool_unique,
        "CANDIDATES_ENTERED_MATRIX": len(old_fragments),
        "CANDIDATES_AFTER_UNIVERSE_FIX": len(new_universe),
        "excluded_parse": exclusion_counts["excluded_parse"],
        "excluded_substructure": exclusion_counts["excluded_substructure"],
        "excluded_projection": exclusion_counts["excluded_projection"],
        "excluded_source_disconnected": exclusion_counts["excluded_source_disconnected"],
        "excluded_source_not_flip": exclusion_counts["excluded_source_not_flip"],
        "excluded_source_low_cfdrop": exclusion_counts["excluded_source_low_cfdrop"],
        "excluded_size": exclusion_counts["excluded_size"],
        "excluded_dedup": exclusion_counts["excluded_dedup"],
        "excluded_missing_lineage": exclusion_counts["excluded_missing_lineage"],
        "excluded_other": sum(
            count
            for reason, count in exclusion_counts.items()
            if reason
            not in {
                "excluded_parse",
                "excluded_substructure",
                "excluded_projection",
                "excluded_source_disconnected",
                "excluded_source_not_flip",
                "excluded_source_low_cfdrop",
                "excluded_size",
                "excluded_dedup",
                "excluded_missing_lineage",
            }
        ),
    }
    payload = {
        "status": "PASS",
        "candidate_pool": str(candidate_pool),
        "candidate_pool_sha256": _sha256(candidate_pool),
        "matrix_root": str(matrix_root),
        "teacher_path": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "source_flip_hard_filter_removed": True,
        "source_cfdrop_hard_filter_removed": True,
        "source_oracle_hard_filter_removed": True,
        "test_loaded": False,
        "stage_counts": stage_counts,
        "new_policy_statistics": new_stats,
        "candidate_rows": output_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "candidate_universe_attrition.csv", output_rows)
    _write_json(output_dir / "candidate_universe_attrition.json", payload)
    _write_json(output_dir / "candidate_stage_counts.json", stage_counts)
    report = [
        "BACE candidate universe attrition",
        "=================================",
        *(f"{key}={value}" for key, value in stage_counts.items()),
        "SOURCE_FLIP_HARD_FILTER_REMOVED=true",
        "SOURCE_CFDROP_HARD_FILTER_REMOVED=true",
        "TEST_LOADED=false",
    ]
    _atomic_text(output_dir / "candidate_universe_attrition_report.txt", "\n".join(report) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--selector-input", default=None)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-pool-unique", type=int, default=0)
    parser.add_argument("--expected-old-matrix-candidates", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forbidden = " ".join(
        str(value or "") for value in (args.candidate_pool, args.matrix_root, args.selector_input)
    ).lower()
    if "test" in forbidden or "gcf" in forbidden:
        raise ValueError("Candidate-universe audit forbids test and GCF inputs.")
    payload = audit_candidate_universe(
        candidate_pool=Path(args.candidate_pool).expanduser().resolve(),
        matrix_root=Path(args.matrix_root).expanduser().resolve(),
        teacher_path=Path(args.teacher_path).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        expected_pool_unique=int(args.expected_pool_unique),
        expected_old_matrix_candidates=int(args.expected_old_matrix_candidates),
    )
    print(json.dumps(payload["stage_counts"], sort_keys=True), flush=True)
    print("[BACE_CANDIDATE_UNIVERSE_ATTRITION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
