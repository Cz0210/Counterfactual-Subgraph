from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.autodl.append_non_taste_matrix_authority import build_parser
from scripts.autodl.run_fast16_matrix_publisher_queue import (
    HEARTBEAT_SCHEMA,
    LOCATOR_SCHEMA,
    QUEUE_SCHEMA,
    run_queue,
)
from src.eval.bace_frozen_cell_standardization import standardize_bace_frozen_cell
from src.eval.fast16_matrix_authority_pointer import (
    MatrixAuthorityPointerError,
    POINTER_SCHEMA,
    append_under_authority_pointer,
)
from src.eval.four_by_four_registry import AuditConfig, audit_registry, write_registry_outputs
from src.eval import non_taste_matrix_append as append_module
from src.eval.non_taste_matrix_append import (
    NonTasteMatrixAppendError,
    _validate_aids_terminal,
    _validate_bace_terminal,
    _validate_mut_terminal,
    append_non_taste_matrix_cell,
)
from tests.autodl.test_append_tastemolnet_matrix_authority import _legacy_cell
from tests.autodl.test_bace_frozen_cell_standardization import _checkpoint, _source


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha(path)}


def _bace_campaign(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    explicit: dict[str, str] = {}
    for dataset in ("AIDS", "Mutagenicity"):
        for method in ("Ours", "GCFExplainer", "GlobalGCE"):
            cell = tmp_path / "legacy" / dataset / method
            _legacy_cell(cell, dataset=dataset, method=method)
            explicit[f"{dataset}/{method}"] = str(cell)
    checkpoint, checkpoint_id, test_hash = _checkpoint(tmp_path)
    cells: dict[str, Path] = {}
    for method in ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"):
        fixture = tmp_path / "sources" / method
        fixture.mkdir(parents=True)
        source = _source(
            fixture,
            method=method,
            checkpoint_id=checkpoint_id,
            test_hash=test_hash,
        )
        cell = tmp_path / "bace" / method
        standardize_bace_frozen_cell(
            method=method,
            source_final_root=source,
            gnn_checkpoint=checkpoint,
            output_dir=cell,
        )
        cells[method] = cell
        if method in {"Ours", "GCFExplainer"}:
            explicit[f"BACE/{method}"] = str(cell)
    result = audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=tmp_path / "unused",
            explicit_cells=explicit,
            expectations={
                "datasets": {"BACE": {"oracle_checkpoint": str(checkpoint.resolve())}}
            },
        )
    )
    assert result.matrix_complete_cells == 8
    return write_registry_outputs(result, tmp_path / "authority-8"), cells


def test_sequential_bace_appends_share_one_locked_pointer_and_preserve_rows(
    tmp_path: Path,
) -> None:
    prior, cells = _bace_campaign(tmp_path)
    state = tmp_path / "control/state.json"
    lock = tmp_path / "control/publish.lock"
    before = json.loads((prior / "matrix_status.json").read_text(encoding="utf-8"))

    first = append_under_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=prior,
        requested_cells=("BACE/GlobalGCE",),
        append=lambda current: append_non_taste_matrix_cell(
            prior_authority_root=current,
            dataset="BACE",
            method="GlobalGCE",
            cell_terminal_root=cells["GlobalGCE"],
            output_root=tmp_path / "authority-9",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        ),
    )
    assert first["matrix_complete_cells"] == 9
    assert first["marker"] == "[MATRIX_9_OF_16_PASS]"
    after_first = json.loads(
        (Path(first["output_root"]) / "matrix_status.json").read_text(encoding="utf-8")
    )
    old_rows = {(row["dataset"], row["method"]): row for row in before["cells"]}
    first_rows = {
        (row["dataset"], row["method"]): row for row in after_first["cells"]
    }
    for key, row in old_rows.items():
        if key != ("BACE", "GlobalGCE"):
            assert first_rows[key] == row

    second = append_under_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=None,
        requested_cells=("BACE/ComRecGC",),
        append=lambda current: append_non_taste_matrix_cell(
            prior_authority_root=current,
            dataset="BACE",
            method="ComRecGC",
            cell_terminal_root=cells["ComRecGC"],
            output_root=tmp_path / "authority-10",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        ),
    )
    assert second["matrix_complete_cells"] == 10
    pointer = json.loads(state.read_text(encoding="utf-8"))
    assert pointer["schema_version"] == POINTER_SCHEMA
    assert pointer["latest_authority_root"] == str((tmp_path / "authority-10").resolve())
    assert pointer["latest_count"] == 10
    assert pointer["applied_cells"][-2:] == ["BACE/GlobalGCE", "BACE/ComRecGC"]


def test_bace_terminal_rejects_smoke_failed_or_test_selection_drift(tmp_path: Path) -> None:
    _, cells = _bace_campaign(tmp_path)
    root = cells["GlobalGCE"]
    (root / "PASS").write_text("[BACE_GLOBALGCE_SMOKE_PASS]\n", encoding="utf-8")
    with pytest.raises(NonTasteMatrixAppendError, match="PASS bytes"):
        _validate_bace_terminal(
            root, method="GlobalGCE", proc_root=tmp_path / "proc", require_writer_audit=False
        )
    (root / "PASS").write_bytes(b"PASS\n")
    evaluation = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    evaluation["selection_frozen_before_test"] = False
    _json(root / "evaluation_manifest.json", evaluation)
    with pytest.raises(NonTasteMatrixAppendError, match="terminal contract changed"):
        _validate_bace_terminal(
            root, method="GlobalGCE", proc_root=tmp_path / "proc", require_writer_audit=False
        )
    (root / "FAILED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(NonTasteMatrixAppendError, match="failure sentinel"):
        _validate_bace_terminal(
            root, method="GlobalGCE", proc_root=tmp_path / "proc", require_writer_audit=False
        )


def test_fresh_no_replace(tmp_path: Path) -> None:
    prior, cells = _bace_campaign(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(NonTasteMatrixAppendError, match="must be fresh"):
        append_non_taste_matrix_cell(
            prior_authority_root=prior,
            dataset="BACE",
            method="GlobalGCE",
            cell_terminal_root=cells["GlobalGCE"],
            output_root=occupied,
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        )


def test_pointer_fails_closed_when_state_drifts(tmp_path: Path) -> None:
    prior, cells = _bace_campaign(tmp_path)
    state = tmp_path / "control/state.json"
    lock = tmp_path / "control/publish.lock"
    append_under_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=prior,
        requested_cells=("BACE/GlobalGCE",),
        append=lambda current: append_non_taste_matrix_cell(
            prior_authority_root=current,
            dataset="BACE",
            method="GlobalGCE",
            cell_terminal_root=cells["GlobalGCE"],
            output_root=tmp_path / "authority-9",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        ),
    )
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["latest_count"] = 99
    _json(state, payload)
    with pytest.raises(MatrixAuthorityPointerError, match="does not match"):
        append_under_authority_pointer(
            state_path=state,
            lock_path=lock,
            initial_authority_root=None,
            requested_cells=("TasteMolNet/Ours",),
            append=lambda _prior: pytest.fail("drifted pointer must not call appender"),
        )


def _aids_final_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "aids-science"
    source_eval = root / "evaluation"
    standardized = root / "standardized"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    teacher = inputs / "teacher.pkl"
    dataset_csv = inputs / "aids.csv"
    molclr = inputs / "molclr.pt"
    thresholds = inputs / "thresholds.json"
    teacher.write_bytes(b"teacher")
    dataset_csv.write_text("smiles,label\nC,1\n", encoding="utf-8")
    molclr.write_bytes(b"molclr")
    thresholds.write_text("{\"thresholds\":[0.1]}\n", encoding="utf-8")
    common = {
        "schema_version": 1,
        "dataset": "AIDS",
        "dataset_key": "aids",
        "method": "COMRECGC-Adapted-DeterministicChemRepair",
        "run_complete": True,
        "mode": "full",
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "candidate_order_unchanged": True,
        "invalid_candidates_sent_to_rf_or_wnode": False,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "distance_calculation_reimplemented": False,
        "teacher_calculation_reimplemented": False,
        "calibration_loaded": False,
        "test_loaded_for_selection": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "teacher_path": str(teacher.resolve()),
        "teacher_sha256": _sha(teacher),
        "dataset_csv": str(dataset_csv.resolve()),
        "dataset_csv_sha256": _sha(dataset_csv),
        "parent_ids_sha256": "1" * 64,
        "molclr_checkpoint": str(molclr.resolve()),
        "molclr_checkpoint_sha256": _sha(molclr),
        "thresholds_path": str(thresholds.resolve()),
        "thresholds_sha256": _sha(thresholds),
        "dataset_fingerprint": "fixture-dataset",
    }
    _json(source_eval / "run_manifest.json", common)
    _json(source_eval / "summary.json", common)
    _json(source_eval / "final_artifact_audit.json", {**common, "audit_passed": True})
    for name in (
        "pair_matrix.jsonl",
        "selected_sequence.jsonl",
        "representative_counterfactuals.jsonl",
        "selected_common_recourses.json",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_comrecgc_k10.csv",
        "table2_comrecgc_k20.csv",
    ):
        path = source_eval / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n" if name.endswith((".csv", ".jsonl")) else "{}\n", encoding="utf-8")
    gate = {
        "schema_version": 1,
        "stage": "aids_project_full_gate",
        "status": "FULL_EXECUTION_PASS",
        "audit_passed": True,
        "run_complete": True,
        "dataset": "aids",
        "source_run_dir": str(source_eval.resolve()),
        "source_manifest_sha256": _sha(source_eval / "run_manifest.json"),
    }
    gate_path = root / "evaluation_gate/gate_result.json"
    _json(gate_path, gate)
    required = [path.name for path in source_eval.iterdir() if path.is_file()]
    standardized.mkdir(parents=True)
    for name in required:
        shutil.copyfile(source_eval / name, standardized / name)
    files = {name: _identity(standardized / name) for name in sorted(required)}
    freeze = {
        "schema_version": 1,
        "dataset": "AIDS",
        "dataset_key": "aids",
        "method": "COMRECGC-Adapted-DeterministicChemRepair",
        "source_output_root": str(source_eval.resolve()),
        "standardized_output_root": str(standardized.resolve()),
        "source_run_manifest_sha256": _sha(source_eval / "run_manifest.json"),
        "source_gate_result_path": str(gate_path.resolve()),
        "source_gate_result_sha256": _sha(gate_path),
        "teacher_sha256": _sha(teacher),
        "molclr_checkpoint_sha256": _sha(molclr),
        "dataset_fingerprint": "fixture-dataset",
        "gate_return_code": 0,
        "files": files,
    }
    _json(standardized / "freeze_manifest.json", freeze)
    _json(
        standardized / "_FINALIZED.json",
        {
            "finalized": True,
            "gate_passed": True,
            "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
        },
    )
    generation = root / "generation"
    generation.mkdir()
    (generation / "counterfactuals.pt").write_bytes(b"generation")
    source_integrity = root / "source_integrity_final.json"
    _json(source_integrity, {"status": "PASS"})
    outer = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "aids",
        "method": "COMRECGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "generation_adopted": True,
        "generation_rerun": False,
        "ordering_adopted": False,
        "evaluation_adopted": False,
        "source_generation_root": str(generation.resolve()),
        "standardized_output_root": str(standardized.resolve()),
        "standardized_run_manifest_sha256": _sha(standardized / "run_manifest.json"),
        "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
        "teacher_sha256": _sha(teacher),
        "molclr_checkpoint_sha256": _sha(molclr),
        "dataset_csv_sha256": _sha(dataset_csv),
        "source_integrity_final_sha256": _sha(source_integrity),
    }
    _json(root / "run_manifest.json", outer)
    _json(root / "final_gate.json", outer)
    _json(root / "_RUN_COMPLETE.json", {**outer, "run_complete": True})
    common_terminal = root / "common_recourse/_RUN_COMPLETE.json"
    _json(common_terminal, {"run_complete": True})
    (root / "PASS").write_bytes(b"PASS\n")
    exact_receipt = root / "exact_receipt.json"
    exact_payload = {
        "status": "PASS",
        "run_complete": True,
        "dbscan_partition_proven": True,
        "ordinary_pass_dependency_eligible": False,
    }
    _json(exact_receipt, exact_payload)
    final_receipt = root / "exact_recovery_freeze_receipt.json"
    final_payload = {
        "schema_version": append_module.FINAL_STAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "dataset": "aids",
        "method": "COMRECGC",
        "continuation_terminal_sha256": _sha(root / "_RUN_COMPLETE.json"),
        "common_terminal_sha256": _sha(common_terminal),
        "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
    }
    _json(final_receipt, final_payload)
    controller_manifest = tmp_path / "controller.manifest.json"
    _json(controller_manifest, {"fixture": True})
    return {
        "root": root,
        "controller_manifest": controller_manifest,
        "exact_path": exact_receipt,
        "exact_payload": exact_payload,
        "final_path": final_receipt,
        "final_payload": final_payload,
    }


def test_aids_accepts_only_controller_bound_final_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _aids_final_fixture(tmp_path)
    monkeypatch.setattr(append_module, "load_bound_controller_manifest", lambda _path: {})
    monkeypatch.setattr(
        append_module,
        "validate_controller_terminal",
        lambda _manifest: {"schema_version": append_module.AIDS_CONTROLLER_TERMINAL_SCHEMA},
    )

    def _stage(_manifest: object, *, stage_id: str) -> dict[str, object]:
        if stage_id == append_module.EXACT_STAGE:
            return {
                "path": str(fixture["exact_path"]),
                "stage_receipt": fixture["exact_payload"],
            }
        return {"path": str(fixture["final_path"]), "manifest": fixture["final_payload"]}

    monkeypatch.setattr(append_module, "validate_stage_terminal", _stage)
    monkeypatch.setattr(
        append_module, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    evidence = _validate_aids_terminal(
        fixture["root"],
        controller_manifest_path=fixture["controller_manifest"],
        proc_root=tmp_path / "proc",
        require_writer_audit=False,
    )
    assert evidence["terminal_kind"] == "AIDS_EXACT_RECOVERY_CONTROLLER_FINAL"
    assert evidence["standardized"]["identities"]["oracle_hash"] == _sha(
        tmp_path / "inputs/teacher.pkl"
    )
    with pytest.raises(NonTasteMatrixAppendError, match="PASS"):
        _validate_aids_terminal(
            Path(fixture["root"]) / "standardized",
            controller_manifest_path=fixture["controller_manifest"],
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def test_aids_rejects_test_leakage_even_when_controller_is_mocked_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _aids_final_fixture(tmp_path)
    root = Path(fixture["root"])
    run_path = root / "standardized/run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["test_used_for_selection"] = True
    _json(run_path, run)
    monkeypatch.setattr(append_module, "load_bound_controller_manifest", lambda _path: {})
    monkeypatch.setattr(
        append_module,
        "validate_controller_terminal",
        lambda _manifest: {"schema_version": append_module.AIDS_CONTROLLER_TERMINAL_SCHEMA},
    )
    monkeypatch.setattr(
        append_module,
        "validate_stage_terminal",
        lambda _manifest, *, stage_id: (
            {"path": str(fixture["exact_path"]), "stage_receipt": fixture["exact_payload"]}
            if stage_id == append_module.EXACT_STAGE
            else {"path": str(fixture["final_path"]), "manifest": fixture["final_payload"]}
        ),
    )
    monkeypatch.setattr(
        append_module, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    with pytest.raises(NonTasteMatrixAppendError, match="terminal contract changed"):
        _validate_aids_terminal(
            root,
            controller_manifest_path=fixture["controller_manifest"],
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def _mut_final_fixture(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = _aids_final_fixture(tmp_path)
    root = Path(base["root"])
    source = root / "evaluation"
    standardized = root / "standardized"

    for directory in (source, standardized):
        run_path = directory / "run_manifest.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run.update(dataset="Mutagenicity", dataset_key="mutagenicity")
        _json(run_path, run)
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(dataset="Mutagenicity", dataset_key="mutagenicity")
        _json(summary_path, summary)
        audit_path = directory / "final_artifact_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(dataset="Mutagenicity", dataset_key="mutagenicity")
        _json(audit_path, audit)
    gate_path = root / "evaluation_gate/gate_result.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update(stage="mutagenicity_project_full_gate", dataset="mutagenicity")
    gate["source_manifest_sha256"] = _sha(source / "run_manifest.json")
    _json(gate_path, gate)
    freeze_path = standardized / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze.update(dataset="Mutagenicity", dataset_key="mutagenicity")
    freeze["source_run_manifest_sha256"] = _sha(source / "run_manifest.json")
    freeze["source_gate_result_sha256"] = _sha(gate_path)
    freeze["files"] = {
        name: _identity(standardized / name)
        for name in sorted(freeze["files"])
    }
    _json(freeze_path, freeze)
    _json(
        standardized / "_FINALIZED.json",
        {
            "finalized": True,
            "gate_passed": True,
            "freeze_manifest_sha256": _sha(freeze_path),
        },
    )

    generation_root = root / "generation"
    generation = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "mutagenicity",
        "generation_adopted": True,
        "generation_mode": "adopted_read_only_cache",
        "generation_rerun": False,
        "source_generation_root": str(generation_root.resolve()),
        "counterfactuals_sha256_claimed": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_actual": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_verified": True,
        "counterfactual_candidate_count": append_module.MUT_SOURCE_CANDIDATE_COUNT,
    }
    exact_receipt = root / "mut_exact_adoption.json"
    _json(exact_receipt, {"status": "PASS"})
    common_root = root / "exact-common"
    common_root.mkdir()
    controller_state = root / "controller-state.json"
    _json(controller_state, {"state": "PASS"})
    exact = {
        "schema_version": "mut_comrecgc_exact_read_only_adoption_v1",
        "status": "PASS",
        "adoption_receipt_path": str(exact_receipt.resolve()),
        "adoption_receipt_sha256": _sha(exact_receipt),
        "source_controller_state_path": str(controller_state.resolve()),
        "source_controller_state_sha256": _sha(controller_state),
        "common_root": str(common_root.resolve()),
        "common_terminal_sha256": "1" * 64,
        "common_manifest_sha256": "2" * 64,
        "source_generation_root": str(generation_root.resolve()),
        "source_generation_manifest_sha256": "3" * 64,
        "source_counterfactuals_sha256": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "common_recourse_count": append_module.MUT_EXPECTED_COMMON_RECOURSES,
        "common_recourse_parameters": {"fixture": True},
        "dbscan_scientific_identity_sha256": "4" * 64,
        "dbscan_next_offset": 10,
        "common_recourse_rerun": False,
        "dbscan_rerun": False,
        "pair_store_rerun": False,
    }
    parity_path = root / "trace-parity.json"
    _json(parity_path, {"status": "PASS"})
    parity = {
        "status": "PASS",
        "traced_source_root": str(generation_root.resolve()),
        "path": str(parity_path.resolve()),
        "sha256": _sha(parity_path),
    }
    prior_matrix = root / "old-prior-matrix"
    matrix_output = root / "old-mut-matrix"
    prior_matrix.mkdir()
    matrix_output.mkdir()
    _json(
        matrix_output / "append_authority.json",
        {"schema_version": append_module.MUT_MATRIX_APPEND_SCHEMA},
    )
    old_append = {
        "status": "PASS",
        "output_root": str(matrix_output.resolve()),
        "matrix_status_path": str(matrix_output.resolve() / "matrix_status.json"),
        "matrix_status_sha256": "5" * 64,
        "combined_audit_sha256": "6" * 64,
        "matrix_complete_cells": 9,
        "matrix_total_cells": 16,
        "appended_cell": "Mutagenicity/ComRecGC",
        "appended_standardized_root": str(standardized.resolve()),
        "marker": "[MATRIX_9_OF_16_PASS]",
        "adopted_after_interruption": False,
    }
    contract = {
        "schema_version": "mut_comrecgc_exact_postprocess_resume_v1",
        "expected_common_recourse_count": append_module.MUT_EXPECTED_COMMON_RECOURSES,
        "forbidden_reruns": ["pair_store", "DBSCAN", "common_recourse"],
        "exact_adoption_receipt_path": str(exact_receipt.resolve()),
        "exact_adoption_receipt_sha256": _sha(exact_receipt),
        "adopted_common_root": str(common_root.resolve()),
        "adopted_common_terminal_sha256": exact["common_terminal_sha256"],
        "adopted_common_manifest_sha256": exact["common_manifest_sha256"],
        "dbscan_scientific_identity_sha256": exact[
            "dbscan_scientific_identity_sha256"
        ],
        "trace_parity_path": str(parity_path.resolve()),
        "trace_parity_sha256": _sha(parity_path),
        "prior_matrix_root": str(prior_matrix.resolve()),
        "matrix_output_root": str(matrix_output.resolve()),
    }
    science = {
        "schema_version": "mut_comrecgc_exact_postprocess_science_v1",
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "trace_parity_passed": True,
        "generation_adopted": True,
        "generation_rerun": False,
        "common_recourse_adopted": True,
        "common_recourse_rerun": False,
        "dbscan_rerun": False,
        "pair_store_rerun": False,
        "chemistry_rerun": True,
        "wnode_evaluation_rerun": True,
        "expected_common_recourse_count": append_module.MUT_EXPECTED_COMMON_RECOURSES,
        "standardized_output_root": str(standardized.resolve()),
        "teacher_sha256": _sha(tmp_path / "inputs/teacher.pkl"),
        "calibration_loaded": False,
        "test_loaded_only_in_unified_evaluation": True,
        "completed_at": "science-time",
    }
    final = {
        **science,
        "schema_version": append_module.MUT_RUN_SCHEMA,
        "matrix_append_status": "PASS",
        "matrix_output_root": str(matrix_output.resolve()),
        "matrix_complete_cells": 9,
        "matrix_total_cells": 16,
        "matrix_status_sha256": old_append["matrix_status_sha256"],
        "run_complete": True,
        "completed_at": "final-time",
    }
    _json(root / "generation_adoption_manifest.json", generation)
    _json(root / "exact_common_adoption_manifest.json", exact)
    _json(root / "trace_parity_adoption_manifest.json", parity)
    _json(
        root / "source_generation_integrity_final.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "payload_sha256_recomputed": False,
            "payload_stat_unchanged": True,
            "critical_manifest_stat_and_hash_unchanged": True,
        },
    )
    _json(root / "source_exact_integrity_final.json", exact)
    _json(root / "continuation_resume_contract.json", contract)
    _json(root / "science_manifest.json", science)
    _json(root / "_SCIENCE_COMPLETE.json", {**science, "run_complete": True})
    _json(root / "matrix_append_receipt.json", old_append)
    _json(root / "run_manifest.json", final)
    _json(root / "_RUN_COMPLETE.json", final)
    return {
        "root": root,
        "standardized": standardized,
        "exact": exact,
        "parity": parity,
        "old_append": old_append,
    }


def _mut_parity_final_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _mut_final_fixture(tmp_path)
    root = Path(fixture["root"])
    standardized = Path(fixture["standardized"])
    generation_root = root / "generation"
    payload = generation_root / "counterfactuals.pt"

    generation = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "mutagenicity",
        "generation_adopted": True,
        "generation_mode": "adopted_read_only_cache",
        "generation_rerun": False,
        "source_generation_root": str(generation_root.resolve()),
        "counterfactuals_path": str(payload.resolve()),
        "counterfactuals_sha256_claimed": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_actual": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "counterfactuals_sha256_verified": True,
        "counterfactuals_sha256_computation_count": 1,
        "counterfactual_candidate_count": append_module.MUT_SOURCE_CANDIDATE_COUNT,
        "source_project_commit": append_module.SOURCE_PROJECT_COMMIT,
        "upstream_commit": append_module.COMRECGC_UPSTREAM_COMMIT,
        "serialization_rerun": False,
        "lineage_resolution_rerun": False,
        "source_integrity": {"fixture": True},
    }
    reopened_integrity = {
        "schema_version": 1,
        "status": "PASS",
        "payload_sha256_recomputed": False,
        "payload_stat_unchanged": True,
        "critical_manifest_stat_and_hash_unchanged": True,
        "critical_manifests": {"run_manifest.json": {"sha256": "1" * 64}},
        "payload": {"resolved_path": str(payload.resolve())},
        "live_writer_audit_before_snapshot": {"writers": []},
        "live_writer_audit_after_snapshot": {"writers": []},
        "verified_at": "fixture",
    }
    final_integrity = dict(reopened_integrity)

    parity_path = root / "parity-gate/trace_parity.json"
    _json(parity_path, {"status": "PASS"})
    parity = {
        "status": "PASS",
        "traced_source_root": str(generation_root.resolve()),
        "path": str(parity_path.resolve()),
        "sha256": _sha(parity_path),
    }
    common_root = root / "exact-common"
    common_path = root / "common-gate/common_recourse_adoption.json"
    _json(common_path, {"status": "PASS"})
    common = {
        "status": "PASS",
        "common_root": str(common_root.resolve()),
        "path": str(common_path.resolve()),
        "sha256": _sha(common_path),
    }

    upstream = root / "upstream"
    upstream.mkdir()
    required_files: dict[str, str] = {}
    for name in append_module._COMRECGC_REQUIRED_SOURCE_FILES:
        path = upstream / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        required_files[name] = _sha(path)
    _json(
        root / "upstream_checkout_audit.json",
        {
            "root": str(upstream.resolve()),
            "expected_commit": append_module.COMRECGC_UPSTREAM_COMMIT,
            "actual_commit": append_module.COMRECGC_UPSTREAM_COMMIT,
            "commit_match": True,
            "required_files": required_files,
            "vendor_manifest_present": False,
            "vendor_manifest_match": None,
            "import_pass": True,
            "network_required": False,
            "passed": True,
        },
    )
    run = {
        "schema_version": append_module._MUT_PARITY_RUN_SCHEMA,
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "generation_adopted": True,
        "generation_rerun": False,
        "traceoff_reference_rerun": True,
        "trace_parity_passed": True,
        "trace_fields_stripped": False,
        "common_recourse_adopted": True,
        "common_recourse_rerun": False,
        "chemistry_rerun": True,
        "evaluation_rerun": True,
        "source_generation_root": str(generation_root.resolve()),
        "source_common_recourse_root": str(common_root.resolve()),
        "trace_parity_path": str(parity_path.resolve()),
        "trace_parity_sha256": _sha(parity_path),
        "standardized_output_root": str(standardized.resolve()),
        "project_commit": "a" * 40,
        "source_payload_sha256": append_module.MUT_SOURCE_PAYLOAD_SHA256,
        "standardized_run_manifest_sha256": _sha(standardized / "run_manifest.json"),
        "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
        "teacher_sha256": _sha(tmp_path / "inputs/teacher.pkl"),
        "calibration_loaded": False,
        "test_loaded_only_in_unified_evaluation": True,
        "completed_at": "fixture",
    }
    _json(root / "generation_adoption_manifest.json", generation)
    _json(root / "common_recourse_adoption_manifest.json", common)
    _json(root / "trace_parity_adoption_manifest.json", parity)
    _json(root / "source_integrity_final.json", final_integrity)
    _json(root / "run_manifest.json", run)
    _json(root / "final_gate.json", run)
    _json(root / "_RUN_COMPLETE.json", {**run, "run_complete": True})
    (root / "PASS").write_bytes(b"PASS\n")
    return {
        "root": root,
        "standardized": standardized,
        "source_root": generation_root,
        "parity": parity,
        "common": common,
        "reopened_integrity": reopened_integrity,
    }


def _mut_fast_accurate_final_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _mut_parity_final_fixture(tmp_path)
    root = Path(fixture["root"])
    standardized = Path(fixture["standardized"])
    source_root = Path(fixture["source_root"])
    common_root = root / "exact-common"
    common_root.mkdir(exist_ok=True)
    evidence_root = root / "historical-evidence"
    pair_manifest = evidence_root / "pair-store.json"
    dbscan_manifest = evidence_root / "dbscan.json"
    equivalence = evidence_root / "semantic-equivalence.json"
    adoption_path = evidence_root / "historical-adoption.json"
    vectors = evidence_root / "vectors.npy"
    vectors.parent.mkdir(parents=True, exist_ok=True)
    vectors.write_bytes(b"vectors\n")
    universe = "d" * 64
    for path, payload in (
        (
            pair_manifest,
            {
                "status": "PASS",
                "kind": "historical-pair-store",
                "vectors_path": str(vectors.resolve()),
                "vectors_sha256": _sha(vectors),
                "scientific_identity": {
                    "candidate_graph_hashes_sha256": universe,
                },
            },
        ),
        (
            dbscan_manifest,
            {
                "status": "PASS",
                "kind": "historical-dbscan",
                "scientific_identity": {
                    "vectors_path": str(vectors.resolve()),
                    "vectors_sha256": _sha(vectors),
                },
            },
        ),
        (
            equivalence,
            {"status": "PASS", "steps": 500, "paper_eligible": False},
        ),
        (adoption_path, {"status": "PASS", "truthful_source": "trace-on-50k"}),
    ):
        _json(path, payload)
    historical = {
        "status": "PASS",
        "path": str(adoption_path.resolve()),
        "sha256": _sha(adoption_path),
        "common_root": str(common_root.resolve()),
        "candidate_universe_sha": universe,
        "pair_store_source_candidate_universe_sha": universe,
        "dbscan_source_candidate_universe_sha": universe,
        "pair_candidate_graph_hashes_sha256": universe,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "source_pair_store_manifest_path": str(pair_manifest.resolve()),
        "source_pair_store_manifest_sha256": _sha(pair_manifest),
        "source_dbscan_manifest_path": str(dbscan_manifest.resolve()),
        "source_dbscan_manifest_sha256": _sha(dbscan_manifest),
        "500_step_semantic_equivalence_receipt_path": str(equivalence.resolve()),
        "500_step_semantic_equivalence_receipt_sha256": _sha(equivalence),
    }
    old_run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    run = {
        **old_run,
        "schema_version": append_module.MUT_FAST_ACCURATE_RUN_SCHEMA,
        "historical_artifact_adopted": True,
        "historical_source_trace_enabled": True,
        "full_50k_rerun_performed": False,
        "traceoff_reference_rerun": False,
        "trace_parity_passed": False,
        "500_step_semantic_equivalence_passed": True,
        "adoption_without_full_50k_parity_rerun_authorized": True,
        "generation_steps": 50_000,
        "M_MAX": 50_000,
        "M_EFFECTIVE": 50_000,
        "early_stop_used": False,
        "stop_reason": "HISTORICAL_FULL_50K_ARTIFACT_ADOPTION",
        "candidate_capacity": 100_000,
        "candidate_universe_sha": universe,
        "pair_store_source_candidate_universe_sha": universe,
        "dbscan_source_candidate_universe_sha": universe,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": (
            "pair_candidate_universe_via_exact_generation_payload_and_dbscan_vectors"
        ),
        "pair_candidate_graph_hashes_sha256": universe,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "pair_store_rerun": False,
        "dbscan_rerun": False,
        "source_common_recourse_root": str(common_root.resolve()),
        "trace_parity_path": None,
        "trace_parity_sha256": None,
        "historical_adoption_path": str(adoption_path.resolve()),
        "historical_adoption_sha256": _sha(adoption_path),
        "source_pair_store_manifest_path": str(pair_manifest.resolve()),
        "source_pair_store_manifest_sha256": _sha(pair_manifest),
        "source_dbscan_manifest_path": str(dbscan_manifest.resolve()),
        "source_dbscan_manifest_sha256": _sha(dbscan_manifest),
        "500_step_semantic_equivalence_receipt_path": str(equivalence.resolve()),
        "500_step_semantic_equivalence_receipt_sha256": _sha(equivalence),
        "standardized_output_root": str(standardized.resolve()),
    }
    _json(root / "historical_adoption_manifest.json", historical)
    _json(root / "run_manifest.json", run)
    _json(root / "final_gate.json", run)
    _json(root / "_RUN_COMPLETE.json", {**run, "run_complete": True})
    return {**fixture, "historical": historical, "run": run}


def test_mut_parity_standardization_is_a_strict_terminal_without_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_parity_final_fixture(tmp_path)
    monkeypatch.setattr(
        append_module,
        "validate_mut_parity_standardization",
        lambda _path, *, source_root: fixture["parity"],
    )
    monkeypatch.setattr(
        append_module,
        "validate_mut_parity_common_adoption",
        lambda _path, *, parity: fixture["common"],
    )
    monkeypatch.setattr(
        append_module,
        "verify_mut_adopted_generation_integrity",
        lambda _generation: fixture["reopened_integrity"],
    )
    evidence = _validate_mut_terminal(
        fixture["root"],
        proc_root=tmp_path / "proc",
        require_writer_audit=False,
    )
    assert evidence["terminal_kind"] == "MUT_PARITY_STANDARDIZATION_FINAL"
    assert evidence["standardized"]["root"] == str(
        Path(fixture["standardized"]).resolve()
    )
    assert "original_matrix_authority_root" not in evidence
    assert evidence["source_integrity"]["source_payload_sha256_recomputed"] is False


def test_mut_fast_accurate_standardization_is_explicit_truthful_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_fast_accurate_final_fixture(tmp_path)
    monkeypatch.setattr(
        append_module,
        "validate_mut_historical_adoption",
        lambda _path, *, source_root: fixture["historical"],
    )
    monkeypatch.setattr(
        append_module,
        "verify_mut_adopted_generation_integrity",
        lambda _generation: fixture["reopened_integrity"],
    )

    evidence = _validate_mut_terminal(
        fixture["root"],
        proc_root=tmp_path / "proc",
        require_writer_audit=False,
    )

    assert evidence["terminal_kind"] == "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL"
    assert evidence["candidate_universe_sha"] == "d" * 64
    assert evidence["standardized"]["root"] == str(
        Path(fixture["standardized"]).resolve()
    )


def test_mut_fast_accurate_terminal_rejects_false_trace_parity_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_fast_accurate_final_fixture(tmp_path)
    root = Path(fixture["root"])
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    run["trace_parity_passed"] = True
    _json(root / "run_manifest.json", run)
    _json(root / "final_gate.json", run)
    _json(root / "_RUN_COMPLETE.json", {**run, "run_complete": True})
    monkeypatch.setattr(
        append_module,
        "validate_mut_historical_adoption",
        lambda _path, *, source_root: fixture["historical"],
    )

    with pytest.raises(NonTasteMatrixAppendError, match="terminal contract changed"):
        _validate_mut_terminal(
            root,
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def test_mut_fast_accurate_terminal_rejects_pair_universe_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_fast_accurate_final_fixture(tmp_path)
    root = Path(fixture["root"])
    historical = dict(fixture["historical"])
    pair_path = Path(historical["source_pair_store_manifest_path"])
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    pair["scientific_identity"]["candidate_graph_hashes_sha256"] = "e" * 64
    _json(pair_path, pair)
    historical["source_pair_store_manifest_sha256"] = _sha(pair_path)
    _json(root / "historical_adoption_manifest.json", historical)
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    run["source_pair_store_manifest_sha256"] = _sha(pair_path)
    _json(root / "run_manifest.json", run)
    _json(root / "final_gate.json", run)
    _json(root / "_RUN_COMPLETE.json", {**run, "run_complete": True})
    monkeypatch.setattr(
        append_module,
        "validate_mut_historical_adoption",
        lambda _path, *, source_root: historical,
    )

    with pytest.raises(NonTasteMatrixAppendError, match="strict-flip universe"):
        _validate_mut_terminal(
            root,
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def test_mut_parity_terminal_rejects_frozen_source_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_parity_final_fixture(tmp_path)
    monkeypatch.setattr(
        append_module,
        "validate_mut_parity_standardization",
        lambda _path, *, source_root: fixture["parity"],
    )
    monkeypatch.setattr(
        append_module,
        "validate_mut_parity_common_adoption",
        lambda _path, *, parity: fixture["common"],
    )
    def _drift(_generation: object) -> dict[str, object]:
        raise ValueError("SOURCE_CLOSURE_CHANGED:adoption_manifest.json")

    monkeypatch.setattr(
        append_module, "verify_mut_adopted_generation_integrity", _drift
    )
    with pytest.raises(NonTasteMatrixAppendError, match="integrity reopen failed"):
        _validate_mut_terminal(
            fixture["root"],
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def test_mut_accepts_only_full_exact_postprocess_and_reopens_old_append_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _mut_final_fixture(tmp_path)
    monkeypatch.setattr(
        append_module, "validate_mut_exact_adoption", lambda **_kwargs: fixture["exact"]
    )
    monkeypatch.setattr(
        append_module, "validate_mut_trace_parity", lambda _path: fixture["parity"]
    )
    monkeypatch.setattr(
        append_module,
        "_reopen_mut_matrix_append",
        lambda **_kwargs: {**fixture["old_append"], "adopted_after_interruption": True},
    )
    evidence = _validate_mut_terminal(
        fixture["root"],
        proc_root=tmp_path / "proc",
        require_writer_audit=False,
    )
    assert evidence["terminal_kind"] == "MUT_EXACT_POSTPROCESS_FINAL"
    assert evidence["original_matrix_used_as_terminal_evidence_only"] is True
    assert evidence["standardized"]["root"] == str(Path(fixture["standardized"]).resolve())

    (Path(fixture["root"]) / "PASS").write_text(
        "[MUT_COMRECGC_EXACT_POSTPROCESS_PASS]\n", encoding="utf-8"
    )
    with pytest.raises(NonTasteMatrixAppendError, match="PASS bytes"):
        _validate_mut_terminal(
            fixture["root"],
            proc_root=tmp_path / "proc",
            require_writer_audit=False,
        )


def test_mut_shared_pointer_append_uses_current_authority_not_old_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior, _cells = _bace_campaign(tmp_path)
    science_root = tmp_path / "mut-final"
    _legacy_cell(
        science_root / "standardized",
        dataset="Mutagenicity",
        method="ComRecGC",
    )
    shared_oracle = str(
        (tmp_path / "legacy/Mutagenicity/Ours/oracle.pkl").resolve()
    )
    for name in ("summary.json", "run_manifest.json", "final_artifact_audit.json"):
        path = science_root / "standardized" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["oracle_checkpoint"] = shared_oracle
        _json(path, payload)
    (science_root / "PASS").write_bytes(b"PASS\n")
    terminal = {
        "terminal_kind": "MUT_EXACT_POSTPROCESS_FINAL",
        "root": str(science_root.resolve()),
        "standardized": {"root": str((science_root / "standardized").resolve())},
    }
    monkeypatch.setattr(append_module, "_validate_mut_terminal", lambda *_args, **_kwargs: terminal)
    result = append_non_taste_matrix_cell(
        prior_authority_root=prior,
        dataset="Mutagenicity",
        method="ComRecGC",
        cell_terminal_root=science_root,
        output_root=tmp_path / "authority-mut-9",
        require_writer_audit=False,
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
    )
    assert result["matrix_complete_cells"] == 9
    receipt = json.loads(
        (Path(result["output_root"]) / "append_authority.json").read_text(encoding="utf-8")
    )
    assert receipt["prior_authority_root"] == str(prior.resolve())
    assert receipt["appended_cell"]["terminal_evidence"][
        "terminal_kind"
    ] == "MUT_EXACT_POSTPROCESS_FINAL"


def test_cli_exposes_shared_pointer_and_rejects_relative_paths() -> None:
    args = build_parser().parse_args(
        [
            "--dataset",
            "BACE",
            "--method",
            "GlobalGCE",
            "--cell-terminal-root",
            "/cell",
            "--output-root",
            "/output",
        ]
    )
    assert args.authority_state_path.is_absolute()
    assert args.authority_lock_path.is_absolute()
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--dataset",
                "BACE",
                "--method",
                "GlobalGCE",
                "--cell-terminal-root",
                "relative",
                "--output-root",
                "/output",
            ]
        )


def test_queue_restart_treats_cell_already_in_pointer_as_applied(tmp_path: Path) -> None:
    prior, cells = _bace_campaign(tmp_path)
    state = tmp_path / "control/state.json"
    lock = tmp_path / "control/publish.lock"
    applied = append_under_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=prior,
        requested_cells=("BACE/GlobalGCE",),
        append=lambda current: append_non_taste_matrix_cell(
            prior_authority_root=current,
            dataset="BACE",
            method="GlobalGCE",
            cell_terminal_root=cells["GlobalGCE"],
            output_root=tmp_path / "authority-9-restart",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        ),
    )
    queue = tmp_path / "queue.json"
    _json(
        queue,
        {
            "schema_version": QUEUE_SCHEMA,
            "initial_authority_root": str(prior.resolve()),
            "authority_state_path": str(state.resolve()),
            "authority_lock_path": str(lock.resolve()),
            "poll_seconds": 60,
            "cells": [
                {
                    "dataset": "BACE",
                    "method": "GlobalGCE",
                    "terminal_root": str(cells["GlobalGCE"].resolve()),
                    "output_root": str((tmp_path / "must-not-be-used").resolve()),
                }
            ],
        },
    )
    heartbeat = run_queue(
        queue_manifest=queue,
        heartbeat_path=tmp_path / "heartbeat.json",
        once=True,
    )
    assert heartbeat["schema_version"] == HEARTBEAT_SCHEMA
    assert heartbeat["state"] == "PASS"
    assert heartbeat["cells"]["BACE/GlobalGCE"] == {"state": "APPLIED"}
    assert heartbeat["latest_authority_root"] == applied["output_root"]
    assert not (tmp_path / "must-not-be-used").exists()


def test_queue_waits_for_future_t12_terminal_locator_without_accepting_generation(
    tmp_path: Path,
) -> None:
    prior, _cells = _bace_campaign(tmp_path)
    locator = tmp_path / "future-t12-locator.json"
    queue = tmp_path / "queue.json"
    _json(
        queue,
        {
            "schema_version": QUEUE_SCHEMA,
            "initial_authority_root": str(prior.resolve()),
            "authority_state_path": str((tmp_path / "control/state.json").resolve()),
            "authority_lock_path": str((tmp_path / "control/publish.lock").resolve()),
            "poll_seconds": 60,
            "taste": {
                key: str((tmp_path / f"taste-{key}").resolve())
                for key in (
                    "t3_root",
                    "policy_path",
                    "policy_receipt",
                    "prepared_root",
                    "graph_cache_root",
                )
            },
            "cells": [
                {
                    "dataset": "TasteMolNet",
                    "method": "GCFExplainer",
                    "terminal_root_locator": str(locator.resolve()),
                    "output_root": str((tmp_path / "future-authority").resolve()),
                }
            ],
        },
    )
    heartbeat = run_queue(
        queue_manifest=queue,
        heartbeat_path=tmp_path / "heartbeat.json",
        once=True,
    )
    assert LOCATOR_SCHEMA == "fast16_matrix_cell_root_locator_v1"
    assert heartbeat["state"] == "WAITING"
    assert heartbeat["cells"]["TasteMolNet/GCFExplainer"] == {
        "state": "WAITING",
        "reason": "LOCATOR_ABSENT",
    }
    assert not (tmp_path / "future-authority").exists()


def test_new_slurm_wrappers_and_status_use_pinned_hpc_runtime() -> None:
    for relative in (
        "scripts/slurm/append_non_taste_matrix_authority.sh",
        "scripts/slurm/run_fast16_matrix_publisher_queue.sh",
    ):
        script = Path(relative).read_text(encoding="utf-8")
        for required in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "--config configs/hpc.yaml",
            "--set inference.fallback_to_heuristic=false",
        ):
            assert required in script
    status = Path(
        "scripts/autodl/status_fast16_matrix_publisher_queue.sh"
    ).read_text(encoding="utf-8")
    assert "AUTODL_PYTHON=" in status
    assert '"$AUTODL_PYTHON" -m json.tool' in status
