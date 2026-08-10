from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_bace_common4_connected import audit_common4
from scripts.import_bace_v4_common4 import import_v4
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_paper_artifacts import QUANTILES


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_v4(root: Path) -> None:
    for method in ("ours", "gcfexplainer"):
        _json(root / method / "final_artifact_audit.json", {"passed": True})
        (root / method / "payload.txt").write_text(method, encoding="utf-8")
    _json(
        root / "bace_connected_protocol_audit.json",
        {
            "passed": True,
            "same_cf_mode": True,
            "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
            "test_used_for_selection": False,
        },
    )
    _json(
        root / "threshold_protocol/thresholds.json",
        {"schema_version": "placeholder"},
    )
    _json(
        root / "threshold_protocol/threshold_protocol_audit.json",
        {
            "THRESHOLD_METHOD_INDEPENDENT": True,
            "THRESHOLD_TEST_INDEPENDENT": True,
        },
    )
    _json(root / "gcf_candidate_connectivity_audit.json", {"all_candidates_connected": True})


def test_v4_import_is_exact_idempotent_and_collision_safe(tmp_path: Path) -> None:
    source = tmp_path / "v4"
    target = tmp_path / "common4"
    _source_v4(source)

    first = import_v4(source_root=source, target_root=target)
    second = import_v4(source_root=source, target_root=target)
    assert first == second
    assert first["test_reexecuted"] is False
    assert (target / "ours/payload.txt").read_text() == "ours"

    (target / "ours/payload.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError):
        import_v4(source_root=source, target_root=target)


def _method_artifacts(
    root: Path,
    *,
    slug: str,
    display: str,
    teacher: Path,
    molclr: Path,
    threshold_sha: str,
) -> None:
    method_root = root / slug
    figure3_fields = ("method", "k", "coverage", "cost")
    figure4_fields = ("method", "threshold", "coverage")
    table_fields = ("method", "k", "coverage", "cost", "flip_rate", "cf_drop")
    thresholds = [0.01 * index for index in range(1, 8)]
    _csv(
        method_root / "figure3_coverage_vs_k.csv",
        figure3_fields,
        [
            {"method": display, "k": k, "coverage": 0.0, "cost": ""}
            for k in range(1, 21)
        ],
    )
    _csv(
        method_root / "figure4_coverage_vs_threshold.csv",
        figure4_fields,
        [
            {"method": display, "threshold": threshold, "coverage": 0.0}
            for threshold in thresholds
        ],
    )
    _csv(
        method_root / f"table2_{slug}_k10.csv",
        table_fields,
        [
            {
                "method": display,
                "k": 10,
                "coverage": 0.0,
                "cost": "",
                "flip_rate": 0.0,
                "cf_drop": 0.0,
            }
        ],
    )
    summary = {
        "dataset": "BACE",
        "method": display,
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "test_parent_count": 116,
        "test_parent_ids_sha256": "a" * 64,
        "candidate_count": 20,
        "cost_cap": 0.07,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "disconnected_residual_used_count": 0,
        "all_candidates_connected": True,
        "all_evaluated_candidates_connected": True,
        "disconnected_output_used_count": 0,
    }
    _json(method_root / "summary.json", summary)
    manifest = {
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "teacher_path": str(teacher),
        "molclr_checkpoint": str(molclr),
        "thresholds_json_sha256": threshold_sha,
        "threshold_fitted_on_test": False,
    }
    if slug == "comrecgc":
        manifest.pop("thresholds_json_sha256")
        manifest["thresholds_sha256"] = threshold_sha
    _json(method_root / "run_manifest.json", manifest)
    _json(
        method_root / "final_artifact_audit.json",
        {
            "passed": True,
            "figure3_schema": list(figure3_fields),
            "figure4_schema": list(figure4_fields),
            "table2_schema": list(table_fields),
        },
    )


def test_common4_audit_accepts_scientifically_empty_four_method_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "common4"
    root.mkdir()
    teacher = tmp_path / "teacher.pkl"
    molclr = tmp_path / "model.pth"
    teacher.write_bytes(b"teacher")
    molclr.write_bytes(b"molclr")
    thresholds = [0.01 * index for index in range(1, 8)]
    _json(
        root / "thresholds.json",
        {
            "schema_version": "bace_wnode_thresholds_v2",
            "dataset": "BACE",
            "distance_line": "MolCLR-Node-Wasserstein",
            "cf_mode": "strict_flip",
            "quantiles": list(QUANTILES),
            "thresholds": thresholds,
            "theta_star": thresholds[3],
            "cost_cap": thresholds[-1],
            "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
            "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
            "threshold_fitted_on_test": False,
        },
    )
    threshold_sha = _sha(root / "thresholds.json")
    for slug, display in (
        ("ours", "Ours"),
        ("globalgce", "GlobalGCE"),
        ("gcfexplainer", "GCFExplainer"),
        ("comrecgc", "COMRECGC"),
    ):
        _method_artifacts(
            root,
            slug=slug,
            display=display,
            teacher=teacher,
            molclr=molclr,
            threshold_sha=threshold_sha,
        )
    _json(root / "v4_import_manifest.json", {"passed": True, "test_reexecuted": False})
    _json(root / "source_v4_gcf_connectivity_audit.json", {"all_candidates_connected": True})

    result = audit_common4(root=root)
    assert result["passed"] is True
    assert result["methods"] == ["Ours", "GlobalGCE", "GCFExplainer", "COMRECGC"]
    assert result["same_cost_definition"] is True
    assert result["plotting_adapter_required"] is False
    assert (root / "table2_bace_k10.csv").is_file()
