from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.audit_bace_connected_common_artifacts import main
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(
    path: Path,
    fields: list[str],
    rows: dict[str, object] | list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows if isinstance(rows, list) else [rows])


def test_common_audit_writes_cohort_and_protocol_parity(tmp_path: Path) -> None:
    common = tmp_path / "common"
    ours = common / "ours"
    gcf = common / "gcfexplainer"
    ours.mkdir(parents=True)
    gcf.mkdir()
    teacher = tmp_path / "teacher.pkl"
    molclr = tmp_path / "model.pth"
    teacher.write_bytes(b"teacher")
    molclr.write_bytes(b"molclr")
    thresholds = common / "thresholds.json"
    _write_json(
        thresholds,
        {
            "theta_star": 0.1,
            "thresholds": [0.1],
            "threshold_fitted_on_test": False,
            "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
            "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        },
    )
    threshold_sha = hashlib.sha256(thresholds.read_bytes()).hexdigest()
    figure3_schema = ["method", "k", "coverage", "cost"]
    figure4_schema = ["method", "threshold", "coverage"]
    table_schema = ["method", "k", "coverage", "cost", "flip_rate", "cf_drop"]
    for method, root, table_name in (
        ("Ours", ours, "table2_ours_k10.csv"),
        ("GCFExplainer", gcf, "table2_gcfexplainer_k10.csv"),
    ):
        _write_csv(
            root / "figure3_coverage_vs_k.csv",
            figure3_schema,
            [
                {"method": method, "k": k, "coverage": 0.0, "cost": ""}
                for k in range(1, 21)
            ],
        )
        _write_csv(
            root / "figure4_coverage_vs_threshold.csv",
            figure4_schema,
            {"method": method, "threshold": 0.1, "coverage": 0.0},
        )
        _write_csv(
            root / table_name,
            table_schema,
            {
                "method": method,
                "k": 10,
                "coverage": 0.0,
                "cost": "",
                "flip_rate": 0.0,
                "cf_drop": 0.0,
            },
        )
        _write_json(
            root / "summary.json",
            {
                "cf_mode": "strict_flip",
                "distance_line": "MolCLR-Node-Wasserstein",
                "cost_cap": 0.1,
                "test_parent_count": 116,
                "test_parent_ids_sha256": "a" * 64,
                "selection_performed_in_eval": False,
                "test_used_for_selection": False,
                "disconnected_residual_used_count": 0,
                "covered_residual_connected_rate": 1.0,
            },
        )
        _write_json(
            root / "run_manifest.json",
            {
                "thresholds_json_sha256": threshold_sha,
                "teacher_path": str(teacher),
                "molclr_checkpoint": str(molclr),
            },
        )
        _write_json(
            root / "final_artifact_audit.json",
            {
                "passed": True,
                "figure3_schema": figure3_schema,
                "figure4_schema": figure4_schema,
                "table2_schema": table_schema,
            },
        )
    candidate_audit = tmp_path / "gcf_candidates.json"
    _write_json(candidate_audit, {"all_candidates_connected": True})

    assert (
        main(
            [
                "--ours-root",
                str(ours),
                "--gcf-root",
                str(gcf),
                "--thresholds-json",
                str(thresholds),
                "--gcf-candidate-audit",
                str(candidate_audit),
                "--output-root",
                str(common),
            ]
        )
        == 0
    )
    protocol = json.loads((common / "bace_connected_protocol_audit.json").read_text())
    cohort = json.loads((common / "cohort_parity_audit.json").read_text())
    assert protocol["same_cost_definition"] is True
    assert protocol["plotting_adapter_required"] is False
    assert cohort["same_parent_cohort"] is True
    assert cohort["eligible_parent_count"] == 116
