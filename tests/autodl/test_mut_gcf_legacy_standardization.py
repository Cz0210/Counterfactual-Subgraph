from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.eval.four_by_four_main_results import audit_cell
from src.eval.four_by_four_registry import AuditConfig, audit_registry
from src.eval.mut_gcf_legacy_standardization import (
    COST_CAP,
    THRESHOLD_COUNT,
    standardize_mut_gcf_legacy_cell,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    heldout = tmp_path / "heldout"
    final = heldout / "final"
    final.mkdir(parents=True)
    frozen = tmp_path / "frozen"
    (frozen / "schema_reference").mkdir(parents=True)
    values = [COST_CAP * index / (THRESHOLD_COUNT - 1) for index in range(THRESHOLD_COUNT)]
    identities = {
        "dataset": "1" * 64,
        "split": "2" * 64,
        "oracle": "3" * 64,
        "molclr": "4" * 64,
        "threshold": "5" * 64,
    }
    threshold = {
        "status": "PASS",
        "dataset": "Mutagenicity",
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "thresholds": values,
        "theta_star": 0.05,
        "cost_cap": 0.0535,
        "threshold_config_hash": identities["threshold"],
        "test_used_for_selection": False,
    }
    _json(frozen / "matched_thresholds.json", threshold)
    _json(frozen / "schema_reference/thresholds.json", threshold)

    prefix = [
        {
            "method": "GCFExplainer-Top20",
            "k": index,
            "close_cf_coverage": index / 20,
            "fixed_capped_mean_cost": 0.0535 - index * 0.001,
        }
        for index in range(1, 21)
    ]
    prefix_fields = list(prefix[0])
    _csv(final / "figure3_coverage_vs_k.csv", prefix_fields, prefix)
    _csv(final / "prefix_metrics.csv", prefix_fields, prefix)
    _json(final / "prefix_metrics.json", {"prefix_metrics": prefix})
    figure4 = [
        {
            "method": "GCFExplainer-Top20",
            "k": k,
            "threshold": value,
            "close_cf_coverage": index / (THRESHOLD_COUNT - 1),
        }
        for k in (10, 20)
        for index, value in enumerate(values)
    ]
    _csv(final / "figure4_coverage_vs_threshold.csv", list(figure4[0]), figure4)
    _csv(
        final / "table2_gcfexplainer_k10.csv",
        ["dataset", "method", "k", "coverage", "cost"],
        [
            {
                "dataset": "Mutagenicity",
                "method": "GCFExplainer-Top20",
                "k": 10,
                "coverage": 0.5,
                "cost": 0.02,
            }
        ],
    )
    _csv(
        final / "parent_best_distances.csv",
        ["parent_id", "method", "distance"],
        [{"parent_id": "parent-1", "method": "GCFExplainer-Top20", "distance": 0.02}],
    )
    summary = {
        "dataset": "Mutagenicity",
        "method": "GCFExplainer-Top20",
        "source_label": 1,
        "target_label": 0,
        "test_parent_count": 217,
        "candidate_count": 20,
        "pair_count": 4340,
        "complete_cartesian": True,
        "run_complete": True,
        "candidate_selection_performed": False,
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
        "test_used_for_selection": False,
    }
    _json(final / "summary.json", summary)
    test_audit = {
        "audit_passed": True,
        "cohort": "test",
        "parent_count": 217,
        "candidate_count": 20,
        "pair_count": 4340,
        "complete_cartesian": True,
        "strict_flip": True,
        "distance_line": "MolCLR-Node-Wasserstein",
        "candidate_selection_performed": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "dataset_csv_sha256": identities["dataset"],
        "test_cohort_hash": identities["split"],
        "teacher_sha256": identities["oracle"],
        "molclr_checkpoint_sha256": identities["molclr"],
    }
    run = {
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
        "test_used_for_selection": False,
        "teacher": {"path": "/frozen/mut-rf.pkl", "sha256": identities["oracle"]},
        "molclr_checkpoint": {"path": "/frozen/molclr.pt", "sha256": identities["molclr"]},
        "test_evaluation_audit": test_audit,
        "threshold_provenance": {
            "ours_thresholds_json_sha256": _sha(frozen / "schema_reference/thresholds.json")
        },
    }
    _json(final / "run_manifest.json", run)
    _json(
        final / "final_artifact_audit.json",
        {
            "final_artifact_audit_passed": True,
            "parent_count": 217,
            "candidate_count": 20,
            "pair_count": 4340,
            "complete_cartesian": True,
            "candidate_order_frozen": True,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        },
    )
    _json(final / "_RUN_COMPLETE.json", {"run_complete": True, "audit_passed": True})
    inventory_names = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_gcfexplainer_k10.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
    )
    _json(
        final / "artifact_manifest.json",
        {"files": {name: _sha(final / name) for name in inventory_names}},
    )
    _json(
        final / "_FINALIZED.json",
        {"finalized": True, "artifact_manifest_sha256": _sha(final / "artifact_manifest.json")},
    )
    return heldout, frozen, identities


def test_standardizes_k10_figure4_and_passes_final_cell_contract(tmp_path: Path) -> None:
    heldout, frozen, identities = _fixture(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    output = tmp_path / "standardized"
    result = standardize_mut_gcf_legacy_cell(
        heldout_root=heldout,
        frozen_root=frozen,
        output_dir=output,
        proc_root=proc,
    )
    assert result["status"] == "FROZEN_PASS"
    with (output / "figure4_coverage_vs_threshold.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 601
    assert {row["k"] for row in rows} == {"10"}
    assert {row["method"] for row in rows} == {"GCFExplainer"}

    registry = audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=tmp_path / "registry",
            expectations={
                "datasets": {
                    "Mutagenicity": {
                        "oracle_backend": "rf",
                        "classifier_family": "random_forest",
                        "oracle_hash": identities["oracle"],
                        "dataset_hash": identities["dataset"],
                        "split_hash": identities["split"],
                        "molclr_checkpoint_hash": identities["molclr"],
                        "threshold_config_hash": identities["threshold"],
                    }
                }
            },
            explicit_cells={"Mutagenicity/GCFExplainer": str(output)},
        )
    )
    row = next(
        row
        for row in registry.matrix_rows
        if row["dataset"] == "Mutagenicity" and row["method"] == "GCFExplainer"
    )
    assert row["status"] == "FROZEN_PASS", row["rerun_reason"]
    audit_cell(row)
