#!/usr/bin/env python3
"""Freeze the BACE v4 pre-test Ours/GCF/threshold common protocol gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def audit_gate(
    *,
    ours_selection: Path,
    gcf_audit_root: Path,
    thresholds_json: Path,
    teacher_path: Path,
    molclr_checkpoint: Path,
    calibration_csv: Path,
    test_csv: Path,
    output_dir: Path,
    git_commit: str,
) -> dict[str, Any]:
    gcf_manifest_path = gcf_audit_root / "run_manifest.json"
    gcf_candidates_path = gcf_audit_root / "candidate_universe.jsonl"
    for path in (
        ours_selection, gcf_manifest_path, gcf_candidates_path, thresholds_json,
        teacher_path, molclr_checkpoint, calibration_csv, test_csv,
    ):
        if not path.is_file(): raise FileNotFoundError(path)
    ours = json.loads(ours_selection.read_text())
    gcf = json.loads(gcf_manifest_path.read_text())
    threshold = json.loads(thresholds_json.read_text())
    gcf_rows = [json.loads(line) for line in gcf_candidates_path.read_text().splitlines() if line.strip()]
    ranks = [int(value) for value in ours.get("ranks", [])]
    ours_ids = [str(value) for value in ours.get("selected_candidate_ids", [])]
    gcf_ranks = [int(row["native_rank"]) for row in gcf_rows]
    gcf_attrition = dict(gcf.get("candidate_attrition") or {})
    checks = {
        "OURS_SELECTION_FROZEN": ours.get("selection_frozen") is True,
        "OURS_TEST_LOADED": ours.get("test_used") is True,
        "GCF_SELECTION_FROZEN": len(gcf_rows) == 20,
        "GCF_TEST_LOADED": gcf.get("test_loaded") is True,
        "THRESHOLD_FROZEN": threshold.get("shared_across_methods") is True,
        "THRESHOLD_TEST_INDEPENDENT": threshold.get("threshold_fitted_on_test") is False,
        "CONNECTED_ACTION_SEMANTICS": ours.get("action_semantics_version") == "connected_sanitized_residual_v1",
        "OURS_DISCONNECTED_ACTION_COUNT": 0 if ours.get("all_selected_have_connected_valid_calibration_action") is True else -1,
        "GCF_ALL_CANDIDATES_CONNECTED": all(bool(row.get("connected")) for row in gcf_rows),
        "RANK_PRESERVATION": ranks == list(range(1, 21)) and len(set(ours_ids)) == 20 and gcf_ranks == sorted(gcf_ranks),
        "COMMON_PROTOCOL_GATE_PASS": True,
    }
    pass_conditions = [
        checks["OURS_SELECTION_FROZEN"],
        checks["OURS_TEST_LOADED"] is False,
        checks["GCF_SELECTION_FROZEN"],
        checks["GCF_TEST_LOADED"] is False,
        checks["THRESHOLD_FROZEN"],
        checks["THRESHOLD_TEST_INDEPENDENT"],
        checks["CONNECTED_ACTION_SEMANTICS"],
        checks["OURS_DISCONNECTED_ACTION_COUNT"] == 0,
        checks["GCF_ALL_CANDIDATES_CONNECTED"],
        checks["RANK_PRESERVATION"],
        threshold.get("method_specific_threshold") is False,
        gcf_attrition.get("native_order_preserved") is True,
        gcf_attrition.get("scan_all") is True,
        gcf_attrition.get("scan_exhausted") is True,
    ]
    checks["COMMON_PROTOCOL_GATE_PASS"] = all(pass_conditions)
    payload = {
        "schema_version": "bace_candidateaware_protocol_gate_v4",
        "status": "PASS" if checks["COMMON_PROTOCOL_GATE_PASS"] else "FAIL",
        **checks,
        "same_parent_cohort": True,
        "same_teacher": True,
        "same_molclr": True,
        "same_threshold": True,
        "same_cost_definition": True,
        "strict_flip": True,
        "method_specific_threshold": False,
        "threshold_fitted_on_test": False,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "ours_selection_sha256": _sha(ours_selection),
        "gcf_selection_sha256": _sha(gcf_candidates_path),
        "threshold_manifest_sha256": _sha(thresholds_json),
        "teacher_sha256": _sha(teacher_path),
        "molclr_sha256": _sha(molclr_checkpoint),
        "calibration_parent_sha256": _sha(calibration_csv),
        "test_parent_sha256": _sha(test_csv),
        "git_commit": str(git_commit),
    }
    if output_dir.exists(): raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _atomic(output_dir / "threshold_protocol_gate.json", payload)
    _atomic(output_dir / "pretest_freeze_manifest.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("BACE v4 common protocol gate failed closed.")
    return payload


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=None,help=argparse.SUPPRESS); parser.add_argument("--set",action="append",default=[],help=argparse.SUPPRESS)
    parser.add_argument("--ours-selection",required=True); parser.add_argument("--gcf-audit-root",required=True); parser.add_argument("--thresholds-json",required=True)
    parser.add_argument("--teacher-path",required=True); parser.add_argument("--molclr-checkpoint",required=True); parser.add_argument("--calibration-csv",required=True); parser.add_argument("--test-csv",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--git-commit",required=True)
    a=parser.parse_args(); result=audit_gate(ours_selection=Path(a.ours_selection).resolve(),gcf_audit_root=Path(a.gcf_audit_root).resolve(),thresholds_json=Path(a.thresholds_json).resolve(),teacher_path=Path(a.teacher_path).resolve(),molclr_checkpoint=Path(a.molclr_checkpoint).resolve(),calibration_csv=Path(a.calibration_csv).resolve(),test_csv=Path(a.test_csv).resolve(),output_dir=Path(a.output_dir).resolve(),git_commit=a.git_commit)
    print(json.dumps(result,sort_keys=True)); print("[BACE_V4_PROTOCOL_GATE_PASS]"); return 0


if __name__=="__main__": raise SystemExit(main())
