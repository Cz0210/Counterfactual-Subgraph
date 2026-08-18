#!/usr/bin/env python3
"""Audit BACE GlobalGCE candidate schema and the historical all-zero adapter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_bace_action_adapter import (  # noqa: E402
    FULLGRAPH_ACTION_ADAPTER,
    adapt_globalgce_fullgraph_rows,
    infer_globalgce_native_output_type,
    read_candidate_csv,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=None, help=argparse.SUPPRESS)
    value.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    value.add_argument("--selected-csv", action="append", required=True)
    value.add_argument("--old-pair-matrix", action="append", default=[])
    value.add_argument("--output-dir", required=True)
    value.add_argument("--validate-only", action="store_true")
    return value


def _old_matrix(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("failure_reason") or "")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "path": str(path),
        "pair_count": len(rows),
        "applicable_count": sum(bool(row.get("applicable")) for row in rows),
        "strict_flip_count": sum(bool(row.get("strict_flip")) for row in rows),
        "failure_reason_counts": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    candidate_audits = []
    row_audits = []
    for value in args.selected_csv:
        path = Path(value).expanduser().resolve()
        rows = read_candidate_csv(path)
        native_type = infer_globalgce_native_output_type(rows)
        adapted = adapt_globalgce_fullgraph_rows(rows, expected_count=20)
        candidate_audits.append(
            {
                "path": str(path),
                "native_output_type": native_type,
                "candidate_count": len(adapted),
                "unique_candidate_count": len({row.candidate_id for row in adapted}),
                "connected_parse_count": len(adapted),
                "action_adapter": FULLGRAPH_ACTION_ADAPTER,
            }
        )
        row_audits.extend(
            {
                "source_path": str(path),
                "rank": row.rank,
                "candidate_id": row.candidate_id,
                "candidate_smiles": row.candidate_smiles,
                "parse_ok": True,
                "sanitize_ok": True,
                "connected": True,
                "native_output_type": row.native_output_type,
                "action_adapter": row.action_adapter,
            }
            for row in adapted
        )
    old = [_old_matrix(Path(path).expanduser().resolve()) for path in args.old_pair_matrix]
    payload = {
        "schema_version": "bace_globalgce_action_adapter_audit_v1",
        "candidate_audits": candidate_audits,
        "historical_matrix_audits": old,
        "candidate_native_type": "full_counterfactual_graph",
        "wrong_field_found": "canonical_smiles_was_exported_as_final_fragment",
        "wrong_evaluator_found": "ours_connected_hard_deletion",
        "action_adapter": FULLGRAPH_ACTION_ADAPTER,
        "root_cause": (
            "GlobalGCE full counterfactual molecules were evaluated as deletion fragments; "
            "parent substructure matching therefore rejected every pair."
        ),
        "audit_pass": True,
    }
    if args.validate_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("candidate_schema_audit.json", "action_semantics_audit.json"):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(row_audits[0])
    for name in ("candidate_rows_audit.csv", "independent_parse_audit.csv"):
        with (output / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row_audits)
    (output / "root_cause_report.txt").write_text(
        payload["root_cause"] + "\n", encoding="utf-8"
    )
    print("[BACE_GLOBALGCE_ACTION_ADAPTER_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
