from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl import audit_four_methods_four_datasets as audit_cli
from src.eval.four_by_four_registry import (
    AuditConfig,
    CellStatus,
    DATASETS,
    MATRIX_FIELDS,
    METHODS,
    audit_registry,
    build_threshold_contracts,
    write_registry_outputs,
)


def _sha_token(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _method_slug(method: str) -> str:
    return "".join(character.lower() for character in method if character.isalnum())


def _complete_cell(
    root: Path,
    *,
    dataset: str,
    method: str,
    oracle_backend: str,
    classifier_family: str,
    oracle_hash: str | None = None,
    dataset_hash: str | None = None,
    split_hash: str | None = None,
    rf_oracle_used: bool | None = None,
    frozen: bool = False,
) -> None:
    oracle_hash = oracle_hash or _sha_token(f"{dataset}-oracle")
    dataset_hash = dataset_hash or _sha_token(f"{dataset}-dataset")
    split_hash = split_hash or _sha_token(f"{dataset}-test")
    oracle = root / "raw/oracle.bin"
    oracle.parent.mkdir(parents=True, exist_ok=True)
    oracle.write_bytes(b"oracle")
    (root / "pair_details.csv").write_text("parent_id,candidate_id\np0,c0\n", encoding="utf-8")
    _csv(
        root / "figure3_coverage_vs_k.csv",
        ("method", "k", "coverage", "cost"),
        [
            {"method": method, "k": k, "coverage": k / 100.0, "cost": 0.01}
            for k in range(1, 21)
        ],
    )
    _csv(
        root / "figure4_coverage_vs_threshold.csv",
        ("method", "threshold", "coverage"),
        [
            {"method": method, "threshold": index / 100.0, "coverage": index / 10.0}
            for index in range(1, 8)
        ],
    )
    _csv(
        root / f"table2_{_method_slug(method)}_k10.csv",
        ("method", "k", "coverage", "cost", "flip_rate", "cf_drop"),
        [
            {
                "method": method,
                "k": 10,
                "coverage": 0.1,
                "cost": 0.01,
                "flip_rate": 0.2,
                "cf_drop": 0.3,
            }
        ],
    )
    common = {
        "dataset": dataset,
        "method": method,
        "dataset_hash": dataset_hash,
        "test_split_hash": split_hash,
        "test_parent_ids_sha256": split_hash,
        "oracle_backend": oracle_backend,
        "classifier_family": classifier_family,
        "oracle_checkpoint": str(oracle),
        "oracle_hash": oracle_hash,
        "molclr_checkpoint_hash": _sha_token(f"{dataset}-molclr"),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "rf_oracle_used": rf_oracle_used,
        "raw_output_complete": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }
    _json(root / "summary.json", common)
    _json(root / "run_manifest.json", {**common, "raw_output_root": str(root)})
    _json(
        root / "final_artifact_audit.json",
        {**common, "passed": True, "frozen": frozen},
    )


def _nested_comrecgc_cell(container: Path) -> Path:
    standardized = container / "standardized"
    _complete_cell(
        standardized,
        dataset="AIDS",
        method="ComRecGC",
        oracle_backend="rf",
        classifier_family="random_forest",
        rf_oracle_used=True,
    )
    for name in ("summary.json", "run_manifest.json", "final_artifact_audit.json"):
        path = standardized / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dataset_key"] = "aids"
        payload["teacher_sha256"] = payload.pop("oracle_hash")
        payload["dataset_csv_sha256"] = payload.pop("dataset_hash")
        payload["parent_ids_sha256"] = payload.pop("test_split_hash")
        payload.pop("test_parent_ids_sha256")
        payload["molclr_checkpoint_sha256"] = payload.pop(
            "molclr_checkpoint_hash"
        )
        payload["thresholds_sha256"] = _sha_token("aids-thresholds")
        payload.pop("raw_output_root", None)
        payload.pop("raw_output_complete", None)
        if name == "final_artifact_audit.json":
            payload["audit_passed"] = payload.pop("passed")
            payload.pop("frozen", None)
        _json(path, payload)
    files: dict[str, dict[str, object]] = {}
    for path in sorted(standardized.iterdir()):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    _json(
        standardized / "freeze_manifest.json",
        {
            "dataset": "AIDS",
            "dataset_key": "aids",
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "files": files,
        },
    )
    _json(standardized / "_FINALIZED.json", {"finalized": True, "gate_passed": True})
    generation = container / "frozen-generation"
    generation.mkdir(parents=True)
    (generation / "counterfactuals.pt").write_bytes(b"frozen-generation")
    final = {
        "status": "PASS",
        "dataset": "AIDS",
        "method": "ComRecGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "generation_adopted": True,
        "source_generation_root": str(generation),
        "standardized_output_root": str(standardized),
        "standardized_run_manifest_sha256": hashlib.sha256(
            (standardized / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "freeze_manifest_sha256": hashlib.sha256(
            (standardized / "freeze_manifest.json").read_bytes()
        ).hexdigest(),
    }
    _json(container / "run_manifest.json", final)
    _json(container / "final_gate.json", final)
    _json(container / "_RUN_COMPLETE.json", {**final, "run_complete": True})
    (container / "PASS").write_text("PASS\n", encoding="utf-8")
    return standardized


def _cell(result: object, dataset: str, method: str) -> dict[str, object]:
    rows = getattr(result, "matrix_rows")
    return next(row for row in rows if row["dataset"] == dataset and row["method"] == method)


def test_empty_scan_emits_exact_matrix_without_fabricating_pass(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    scan.mkdir()
    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )

    assert len(result.matrix_rows) == 16
    assert [(row["dataset"], row["method"]) for row in result.matrix_rows] == [
        (dataset, method) for dataset in DATASETS for method in METHODS
    ]
    assert result.matrix_complete_cells == 0
    assert {
        row["status"] for row in result.matrix_rows if row["dataset"] == "TasteMolNet"
    } == {CellStatus.BLOCKED_LICENSE.value}
    assert {
        row["status"] for row in result.matrix_rows if row["dataset"] != "TasteMolNet"
    } == {CellStatus.MISSING.value}


def test_complete_rf_cell_is_adoptable_and_source_tree_is_read_only(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    cell_root = scan / "aids/ours"
    _complete_cell(
        cell_root,
        dataset="AIDS",
        method="Ours",
        oracle_backend="rf",
        classifier_family="random_forest",
        rf_oracle_used=True,
    )
    before = {
        path.relative_to(scan).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in scan.rglob("*")
        if path.is_file()
    }

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "AIDS", "Ours")

    assert row["status"] == CellStatus.ADOPTABLE_PASS.value
    assert row["generation_adoption_candidate"] is True
    assert result.matrix_complete_cells == 1
    assert before == {
        path.relative_to(scan).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in scan.rglob("*")
        if path.is_file()
    }


def test_bace_rf_artifact_is_stale_oracle_not_adopted(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    _complete_cell(
        scan / "bace/ours",
        dataset="BACE",
        method="Ours",
        oracle_backend="rf",
        classifier_family="random_forest",
        rf_oracle_used=True,
    )

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "BACE", "Ours")

    assert row["status"] == CellStatus.STALE_ORACLE.value
    assert "GNN_ORACLE_CONTRACT_MISMATCH" in str(row["rerun_reason"])
    assert result.matrix_complete_cells == 0


def test_taste_complete_artifact_stays_blocked_without_explicit_license_basis(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "outputs"
    _complete_cell(
        scan / "taste/ours",
        dataset="TasteMolNet",
        method="Ours",
        oracle_backend="gnn",
        classifier_family="gine",
        rf_oracle_used=False,
    )

    blocked = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "blocked")
    )
    assert _cell(blocked, "TasteMolNet", "Ours")["status"] == CellStatus.BLOCKED_LICENSE.value

    allowed = audit_registry(
        AuditConfig(
            scan_roots=(scan,),
            output_root=tmp_path / "allowed",
            taste_license_gate={
                "status": "PASS",
                "passed": True,
                "license_basis": "explicit upstream research reuse statement",
            },
        )
    )
    assert _cell(allowed, "TasteMolNet", "Ours")["status"] == CellStatus.ADOPTABLE_PASS.value


def test_cross_method_oracle_hash_conflict_fails_closed(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    _complete_cell(
        scan / "aids/ours",
        dataset="AIDS",
        method="Ours",
        oracle_backend="rf",
        classifier_family="random_forest",
        oracle_hash=_sha_token("oracle-a"),
        rf_oracle_used=True,
    )
    _complete_cell(
        scan / "aids/gcf",
        dataset="AIDS",
        method="GCFExplainer",
        oracle_backend="rf",
        classifier_family="random_forest",
        oracle_hash=_sha_token("oracle-b"),
        rf_oracle_used=True,
    )

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )

    assert _cell(result, "AIDS", "Ours")["status"] == CellStatus.STALE_ORACLE.value
    assert _cell(result, "AIDS", "GCFExplainer")["status"] == CellStatus.STALE_ORACLE.value
    assert result.matrix_complete_cells == 0


def test_conflicting_test_selection_evidence_cannot_be_hidden_by_false_value(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "outputs"
    root = scan / "aids/ours"
    _complete_cell(
        root,
        dataset="AIDS",
        method="Ours",
        oracle_backend="rf",
        classifier_family="random_forest",
        rf_oracle_used=True,
    )
    audit = json.loads((root / "final_artifact_audit.json").read_text(encoding="utf-8"))
    audit["test_used_for_selection"] = True
    _json(root / "final_artifact_audit.json", audit)

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "AIDS", "Ours")

    assert row["status"] == CellStatus.INCOMPLETE.value
    assert "TEST_SELECTION_EXCLUSION_NOT_PROVEN" in str(row["rerun_reason"])


def test_legacy_raw_evidence_is_only_generation_adoption_candidate(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    root = scan / "legacy/mut_ours"
    root.mkdir(parents=True)
    (root / "pair_details.csv").write_text("parent_id,candidate_id\np,c\n", encoding="utf-8")
    _json(root / "summary.json", {"dataset": "Mutagenicity", "method": "Ours"})
    _json(root / "run_manifest.json", {"dataset": "Mutagenicity", "method": "Ours"})
    _json(root / "final_artifact_audit.json", {"dataset": "Mutagenicity", "method": "Ours"})

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "Mutagenicity", "Ours")

    assert row["status"] == CellStatus.INCOMPLETE.value
    assert row["generation_adoption_candidate"] is True
    assert "deterministic unified re-evaluation" in str(row["generation_adoption_reason"])


def test_top_level_gates_adopt_nested_standardized_cell_without_path_identity(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "outputs"
    container = scan / "opaque-run-id"
    standardized = _nested_comrecgc_cell(container)

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "AIDS", "ComRecGC")

    assert row["status"] == CellStatus.FROZEN_PASS.value
    assert row["standardized_output_root"] == str(standardized.resolve())
    assert row["raw_output_root"] == str((container / "frozen-generation").resolve())
    assert row["generation_adoption_candidate"] is True


def test_nested_freeze_inventory_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    container = scan / "opaque-run-id"
    standardized = _nested_comrecgc_cell(container)
    (standardized / "summary.json").write_text("{}\n", encoding="utf-8")

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )
    row = _cell(result, "AIDS", "ComRecGC")

    assert row["status"] == CellStatus.STALE_METRIC.value
    assert "FROZEN_FILE_" in str(row["rerun_reason"])


def test_clear_is_inventory_only_and_never_mapped_to_comrecgc(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    root = scan / "legacy/aids_clear"
    root.mkdir(parents=True)
    _json(root / "summary.json", {"dataset": "AIDS", "method": "CLEAR"})
    _json(root / "run_manifest.json", {"dataset": "AIDS", "method": "CLEAR"})
    _json(
        root / "final_artifact_audit.json",
        {"dataset": "AIDS", "method": "CLEAR", "passed": True},
    )

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )

    assert _cell(result, "AIDS", "ComRecGC")["status"] == CellStatus.MISSING.value
    assert any(row["candidate_root"] == str(root.resolve()) for row in result.stale_rows)


def test_render_only_combined_v4_is_inventory_only(tmp_path: Path) -> None:
    scan = tmp_path / "outputs"
    root = scan / "legacy-v4"
    root.mkdir(parents=True)
    _json(
        root / "combined_manifest.json",
        {
            "render_only": True,
            "methods": ["Ours", "GlobalGCE", "CLEAR", "GCFExplainer"],
            "datasets": ["AIDS", "Mutagenicity"],
        },
    )

    result = audit_registry(
        AuditConfig(scan_roots=(scan,), output_root=tmp_path / "registry")
    )

    assert result.matrix_complete_cells == 0
    assert _cell(result, "AIDS", "ComRecGC")["status"] == CellStatus.MISSING.value
    assert any(row["file_name"] == "combined_manifest.json" for row in result.inventory_rows)


def test_writer_emits_required_outputs_and_refuses_nonempty_root(tmp_path: Path) -> None:
    scan_a = tmp_path / "outputs-a"
    scan_b = tmp_path / "outputs-b"
    scan_a.mkdir()
    scan_b.mkdir()
    result = audit_registry(
        AuditConfig(
            scan_roots=(scan_a, scan_b),
            output_root=tmp_path / "registry",
        )
    )

    output = write_registry_outputs(result, tmp_path / "registry")
    expected = {
        "matrix_status.csv",
        "matrix_status.json",
        "oracle_registry.json",
        "evaluation_contract.json",
        "artifact_inventory.csv",
        "stale_artifacts.csv",
        "adoption_report.md",
        "threshold_contracts",
    }
    assert {path.name for path in output.iterdir()} == expected
    with (output / "matrix_status.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == MATRIX_FIELDS
        assert len(list(reader)) == 16
    payload = json.loads((output / "matrix_status.json").read_text(encoding="utf-8"))
    assert payload["audit_complete"] is True
    assert payload["all_cells_complete"] is False
    assert payload["no_numeric_imputation"] is True
    assert result.evaluation_contract["final_export_gate"]["required_value"] is True
    threshold_files = sorted(
        path.name for path in (output / "threshold_contracts").iterdir()
    )
    assert threshold_files == ["aids.json", "bace.json", "mutagenicity.json", "tastemolnet.json"]
    with pytest.raises(FileExistsError):
        write_registry_outputs(result, output)


def test_threshold_contract_is_evaluator_ready_only_with_calibration_provenance() -> None:
    missing = build_threshold_contracts()
    assert missing["AIDS"]["status"] == "MISSING_NOT_INFERRED"
    assert "thresholds" not in missing["AIDS"]

    threshold_hash = _sha_token("aids-frozen-thresholds")
    contracts = build_threshold_contracts(
        {
            "datasets": {
                "AIDS": {
                    "thresholds": [0.0, 0.05, 0.0535],
                    "theta_star": 0.05,
                    "cost_cap": 0.0535,
                    "threshold_source": "frozen AIDS calibration protocol",
                    "threshold_source_split": "calibration",
                    "threshold_config_hash": threshold_hash,
                    "test_used_for_selection": False,
                }
            }
        }
    )
    aids = contracts["AIDS"]
    assert aids["status"] == "PASS"
    assert aids["thresholds"] == [0.0, 0.05, 0.0535]
    assert aids["theta_star"] == 0.05
    assert aids["cost_cap"] == 0.0535
    assert aids["threshold_config_hash"] == threshold_hash


def test_threshold_contract_rejects_test_selected_values() -> None:
    contracts = build_threshold_contracts(
        {
            "datasets": {
                "AIDS": {
                    "thresholds": [0.0, 0.05],
                    "theta_star": 0.05,
                    "cost_cap": 0.05,
                    "threshold_source": "held-out curve",
                    "threshold_source_split": "test",
                    "threshold_config_hash": _sha_token("bad-thresholds"),
                    "test_used_for_selection": True,
                }
            }
        }
    )
    assert contracts["AIDS"]["status"] == "INVALID_FAIL_CLOSED"
    assert "thresholds" not in contracts["AIDS"]


def test_threshold_contract_accepts_explicit_existing_frozen_protocol() -> None:
    contract = build_threshold_contracts(
        {
            "datasets": {
                "AIDS": {
                    "thresholds": [0.0, 0.05, 0.0535],
                    "theta_star": 0.05,
                    "cost_cap": 0.0535,
                    "threshold_source": "audited AIDS v4 frozen protocol",
                    "threshold_source_split": "existing_frozen_protocol",
                    "threshold_config_hash": "a" * 64,
                    "test_used_for_selection": False,
                }
            }
        }
    )["AIDS"]
    assert contract["status"] == "PASS"
    assert contract["threshold_source_split"] == "existing_frozen_protocol"
    assert contract["theta_star"] == 0.05


def test_final_export_gate_returns_nonzero_but_preserves_truthful_partial_matrix(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "outputs"
    scan.mkdir()
    output = tmp_path / "registry"

    return_code = audit_cli.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--scan-root",
            str(scan),
            "--output-root",
            str(output),
            "--require-complete",
        ]
    )

    assert return_code == 3
    payload = json.loads((output / "matrix_status.json").read_text(encoding="utf-8"))
    assert payload["audit_complete"] is True
    assert payload["all_cells_complete"] is False
    taste_rows = [row for row in payload["cells"] if row["dataset"] == "TasteMolNet"]
    assert {row["status"] for row in taste_rows} == {
        CellStatus.BLOCKED_LICENSE.value
    }
    assert all(row["raw_output_root"] == "" for row in taste_rows)
