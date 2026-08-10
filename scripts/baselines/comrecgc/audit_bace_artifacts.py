#!/usr/bin/env python3
"""Gate BACE COMRECGC slot artifacts under the common paper protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import sha256_file, write_json  # noqa: E402
from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_paper_artifacts import (  # noqa: E402
    FIGURE3_FIELDS,
    FIGURE4_FIELDS,
    TABLE2_FIELDS,
    load_bace_thresholds,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), [dict(row) for row in reader]


def audit_bace_comrecgc_artifacts(
    *, root: str | Path, thresholds_json: str | Path, expected_parent_count: int = 116
) -> dict[str, Any]:
    output = Path(root).expanduser().resolve()
    threshold_path = Path(thresholds_json).expanduser().resolve()
    required = {
        "figure3": output / "figure3_coverage_vs_k.csv",
        "figure4": output / "figure4_coverage_vs_threshold.csv",
        "table2": output / "table2_comrecgc_k10.csv",
        "summary": output / "summary.json",
        "manifest": output / "run_manifest.json",
        "audit": output / "final_artifact_audit.json",
    }
    for path in (*required.values(), threshold_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    thresholds = load_bace_thresholds(threshold_path)
    fields3, rows3 = _csv(required["figure3"])
    fields4, rows4 = _csv(required["figure4"])
    fields2, rows2 = _csv(required["table2"])
    if fields3 != FIGURE3_FIELDS or fields4 != FIGURE4_FIELDS or fields2 != TABLE2_FIELDS:
        raise ValueError("BACE COMRECGC paper CSV schema differs from the common contract.")
    if len(rows3) != 20 or [int(row["k"]) for row in rows3] != list(range(1, 21)):
        raise ValueError("BACE COMRECGC Figure 3 K grid is not exactly 1..20.")
    coverage3 = [float(row["coverage"]) for row in rows3]
    if any(not math.isfinite(value) for value in coverage3) or any(
        right + 1e-12 < left for left, right in zip(coverage3, coverage3[1:])
    ):
        raise ValueError("BACE COMRECGC Figure 3 coverage is invalid or non-monotone.")
    expected_thresholds = [float(value) for value in thresholds["thresholds"]]
    actual_thresholds = [float(row["threshold"]) for row in rows4]
    if len(actual_thresholds) != len(expected_thresholds) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual, expected in zip(actual_thresholds, expected_thresholds, strict=True)
    ):
        raise ValueError("BACE COMRECGC Figure 4 threshold grid differs.")
    coverage4 = [float(row["coverage"]) for row in rows4]
    if any(not math.isfinite(value) for value in coverage4) or any(
        right + 1e-12 < left for left, right in zip(coverage4, coverage4[1:])
    ):
        raise ValueError("BACE COMRECGC Figure 4 coverage is invalid or non-monotone.")
    if len(rows2) != 1 or int(rows2[0]["k"]) != 10:
        raise ValueError("BACE COMRECGC Table 2 is not one K=10 row.")
    summary = _json(required["summary"])
    manifest = _json(required["manifest"])
    final_audit = _json(required["audit"])
    hard_checks = {
        "dataset": summary.get("dataset") == "BACE",
        "method": summary.get("method") == "COMRECGC",
        "parent_count": int(summary.get("test_parent_count") or -1) == int(expected_parent_count),
        "strict_flip": summary.get("cf_mode") == "strict_flip",
        "connected_semantics": summary.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS,
        "connected_match_policy": summary.get("match_selection_policy") == CONNECTED_MATCH_SELECTION_POLICY,
        "candidate_set_preselected": manifest.get("candidate_set_preselected") is True,
        "selection_outside_eval": manifest.get("selection_performed_in_eval") is False,
        "test_not_used_for_selection": manifest.get("test_loaded_for_selection") is False,
        "threshold_not_fit_on_test": summary.get("threshold_fitted_on_test") is False,
        "rank_not_compacted": summary.get("rank_compaction") is False,
        "no_backfill": summary.get("invalid_slot_backfill") is False,
        "disconnected_not_used": int(summary.get("disconnected_output_used_count") or 0) == 0,
        "all_evaluated_connected": summary.get("all_evaluated_candidates_connected") is True,
        "inner_audit": final_audit.get("passed") is True,
    }
    failed = sorted(name for name, passed in hard_checks.items() if not passed)
    if failed:
        raise ValueError(f"BACE COMRECGC artifact hard checks failed: {failed}")
    result = {
        "schema_version": "bace_comrecgc_artifact_gate_v1",
        "passed": True,
        "dataset": "BACE",
        "method": "COMRECGC",
        "hard_checks": hard_checks,
        "test_parent_count": int(expected_parent_count),
        "candidate_count": int(summary.get("candidate_count") or 0),
        "valid_candidate_count": int(summary.get("valid_repaired_slot_count") or 0),
        "disconnected_output_used_count": 0,
        "strict_flip": True,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "thresholds_sha256": sha256_file(threshold_path),
        "file_sha256": {name: sha256_file(path) for name, path in required.items()},
    }
    write_json(output / "bace_comrecgc_artifact_gate.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--root", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=116)
    args = parser.parse_args(argv)
    result = audit_bace_comrecgc_artifacts(
        root=args.root,
        thresholds_json=args.thresholds_json,
        expected_parent_count=int(args.expected_parent_count),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_COMRECGC_ARTIFACT_GATE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
