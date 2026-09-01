from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.autodl.reconcile_aids_comrecgc_terminal_publication import build_parser
from src.eval import aids_comrecgc_terminal_reconciliation as reconciliation
from src.eval import four_by_four_registry as registry
from src.eval import non_taste_matrix_append as matrix


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _zero_source(tmp_path: Path) -> Path:
    root = tmp_path / "science"
    standardized = root / "standardized"
    common = {
        "dataset": "AIDS",
        "dataset_key": "aids",
        "method": "COMRECGC-Adapted-DeterministicChemRepair",
        "cf_mode": "strict_flip",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "run_complete": True,
    }
    _json(standardized / "run_manifest.json", common)
    _json(standardized / "summary.json", common)
    # Production build_final_audit() intentionally has no dataset/dataset_key.
    _json(
        standardized / "final_artifact_audit.json",
        {
            "method": common["method"],
            "cf_mode": "strict_flip",
            "scientific_output_empty": True,
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
            "run_complete": True,
            "audit_passed": True,
        },
    )
    gate = root / "full_gate/gate_result.json"
    _json(
        gate,
        {
            "status": "FULL_EXECUTION_PASS",
            "audit_passed": True,
            "run_complete": True,
            "dataset": "aids",
            "scientific_output_empty": True,
            "scientific_output_status": "SCIENTIFIC_OUTPUT_EMPTY",
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        },
    )
    _json(
        standardized / "freeze_manifest.json",
        {
            "source_gate_result_path": str(gate.resolve()),
            "source_gate_result_sha256": _sha(gate),
        },
    )
    _json(standardized / "_FINALIZED.json", {"finalized": True})
    prefixes = [
        {
            "method": common["method"],
            "k": k,
            "close_cf_coverage": 0.0,
            "num_any_strict_flip_parents": 0,
            "conditional_mean_cost": None,
            "conditional_median_cost": None,
            "fixed_capped_mean_cost": 1.0,
            "fixed_capped_median_cost": 1.0,
        }
        for k in range(1, 21)
    ]
    _json(standardized / "prefix_metrics.json", {"prefix_metrics": prefixes})
    _csv(standardized / "prefix_metrics.csv", prefixes)
    _csv(
        standardized / "figure3_coverage_vs_k.csv",
        [
            {
                "method": common["method"],
                "k": k,
                "close_cf_coverage": 0.0,
                "conditional_mean_cost": "",
                "conditional_median_cost": "",
                "fixed_capped_mean_cost": 1.0,
                "fixed_capped_median_cost": 1.0,
            }
            for k in range(1, 21)
        ],
    )
    _csv(
        standardized / "figure4_coverage_vs_threshold.csv",
        [
            {
                "method": common["method"],
                "threshold": 0.0,
                "close_cf_coverage": 0.0,
            },
            {
                "method": common["method"],
                "threshold": 0.1,
                "close_cf_coverage": 0.0,
            },
        ],
    )
    for k in (10, 20):
        _csv(
            standardized / f"table2_comrecgc_k{k}.csv",
            [
                {
                    "k": k,
                    "method": common["method"],
                    "dataset": "AIDS",
                    "coverage": 0.0,
                    "ccrcov": 0.0,
                    "conditional_mean_cost": "",
                    "conditional_median_cost": "",
                    "fixed_capped_mean_cost": 1.0,
                    "fixed_capped_median_cost": 1.0,
                }
            ],
        )
    return root


def _science_evidence(
    source: Path, controller: dict[str, object]
) -> dict[str, object]:
    standardized = source / "standardized"
    return {
        "terminal_kind": "AIDS_POSTHOC_SELF_CLOSED_SCIENCE_FINAL",
        "root": str(source.resolve()),
        "controller_manifest_path": controller["controller_manifest_path"],
        "controller_manifest_sha256": controller["controller_manifest_sha256"],
        "posthoc_exact_adoption_path": controller["posthoc_exact_adoption_path"],
        "posthoc_exact_adoption_sha256": controller[
            "posthoc_exact_adoption_sha256"
        ],
        "adoption_checkpoint_path": controller["adoption_checkpoint_path"],
        "adoption_checkpoint_sha256": controller["adoption_checkpoint_sha256"],
        "adoption_checkpoint_progress_rows": controller[
            "adoption_checkpoint_progress_rows"
        ],
        "checkpoint_path": controller["checkpoint_path"],
        "checkpoint_sha256": controller["checkpoint_sha256"],
        "checkpoint_progress_rows": controller["checkpoint_progress_rows"],
        "checkpoint_identity_sha256": controller[
            "checkpoint_identity_sha256"
        ],
        "checkpoint_vectors_sha256": controller["checkpoint_vectors_sha256"],
        "checkpoint_monotonic_from_adoption": controller[
            "checkpoint_monotonic_from_adoption"
        ],
        "exact_receipt_path": controller["exact_receipt"]["path"],
        "exact_receipt_sha256": controller["exact_receipt"]["sha256"],
        "exact_dbscan_manifest_path": controller["exact_receipt"][
            "dbscan_manifest_path"
        ],
        "exact_dbscan_manifest_sha256": controller["exact_receipt"][
            "dbscan_manifest_sha256"
        ],
        "continuation_terminal_sha256": "4" * 64,
        "common_terminal_sha256": "5" * 64,
        "run_manifest_sha256": "6" * 64,
        "final_gate_sha256": "7" * 64,
        "source_generation_root": "/science/generation",
        "source_integrity_final_sha256": "8" * 64,
        "dbscan_adoption_manifest_path": "/science/dbscan_adoption.json",
        "dbscan_adoption_manifest_sha256": "9" * 64,
        "standardized": {
            "root": str(standardized.resolve()),
            "source_evaluation_root": "/science/unified_eval",
            "run_manifest_sha256": _sha(standardized / "run_manifest.json"),
            "final_artifact_audit_sha256": _sha(
                standardized / "final_artifact_audit.json"
            ),
            "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
            "identities": {"oracle_hash": "a" * 64},
        },
        "zero_strict_flip_evidence": reconciliation.validate_zero_strict_flip_science(
            source
        ),
        "inventory": reconciliation.validate_zero_strict_flip_science(source)[
            "source_inventory"
        ],
    }


def test_zero_strict_flip_is_a_valid_non_imputed_science_result(tmp_path: Path) -> None:
    source = _zero_source(tmp_path)
    evidence = reconciliation.validate_zero_strict_flip_science(source)
    assert evidence["status"] == "PASS"
    assert evidence["scientific_output_empty"] is True
    assert evidence["coverage"] == 0.0
    assert evidence["conditional_cost_available"] is False
    assert evidence["numeric_imputation_used"] is False


@pytest.mark.parametrize("failure", ["coverage", "flip", "cost"])
def test_zero_reconciliation_rejects_nonzero_or_imputed_exports(
    tmp_path: Path, failure: str
) -> None:
    source = _zero_source(tmp_path)
    prefix = source / "standardized/prefix_metrics.json"
    value = json.loads(prefix.read_text(encoding="utf-8"))
    if failure == "coverage":
        value["prefix_metrics"][0]["close_cf_coverage"] = 0.1
    elif failure == "flip":
        value["prefix_metrics"][0]["num_any_strict_flip_parents"] = 1
    else:
        value["prefix_metrics"][0]["conditional_median_cost"] = 0.0
    _json(prefix, value)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="prefix contains",
    ):
        reconciliation.validate_zero_strict_flip_science(source)


def _controller_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, object],
    Path,
    Path,
    dict[str, object],
    dict[str, object],
]:
    manifest_path = tmp_path / "controller.manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / ".controller.lock").write_text("{}\n", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    checkpoint = tmp_path / "science/checkpoint.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("adoption-time checkpoint\n", encoding="utf-8")
    exact_receipt = tmp_path / "science/exact_recovery_receipt.json"
    _json(exact_receipt, {"status": "PASS"})
    dbscan = tmp_path / "science/dbscan/run_manifest.json"
    _json(dbscan, {"status": "PASS"})
    stages = [{"stage_id": stage} for stage in reconciliation.STAGE_ORDER]
    exact_stage = next(
        row for row in stages if row["stage_id"] == reconciliation.EXACT_STAGE
    )
    exact_stage.update(
        {
            "output_dir": str(checkpoint.parent.resolve()),
            "terminal_path": str(exact_receipt.resolve()),
            "progress_checkpoint_path": str(checkpoint.resolve()),
        }
    )
    manifest: dict[str, object] = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": "a" * 64,
        "controller_root": str(controller.resolve()),
        "stages": stages,
        "source_authority": {"source_vectors_sha256": "e" * 64},
    }
    exact_index = reconciliation.STAGE_ORDER.index(reconciliation.EXACT_STAGE)
    stage_states = {
        stage: (
            "PASS"
            if index < exact_index
            else "BLOCKED"
            if index == exact_index
            else "PENDING"
        )
        for index, stage in enumerate(reconciliation.STAGE_ORDER)
    }
    state = {
        "schema_version": reconciliation.STATE_SCHEMA,
        "controller_id": reconciliation.CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "status": "BLOCKED",
        "current_stage": reconciliation.EXACT_STAGE,
        "stages": stage_states,
        "controller_process": {"pid": 321, "start_ticks": 456},
        "worker": {
            "stage_id": reconciliation.EXACT_STAGE,
            "pid": 654,
            "start_ticks": 987,
            "process_group_id": 654,
        },
        "startup_barrier": {"stage_id": reconciliation.EXACT_STAGE},
    }
    _json(controller / "state.json", state)
    adopted_stat = checkpoint.stat()
    snapshot = {
        "path": str(checkpoint.resolve()),
        "sha256_at_observation": _sha(checkpoint),
        "checkpoint_payload_sha256": "b" * 64,
        "identity_sha256": "c" * 64,
        "progress_ledgers_sha256": "d" * 64,
        "progress_rows": 123,
        "vectors_sha256": "e" * 64,
        "stat_identity_at_observation": {
            "device": adopted_stat.st_dev,
            "inode": adopted_stat.st_ino,
            "mode": adopted_stat.st_mode,
            "size": adopted_stat.st_size,
            "mtime_ns": adopted_stat.st_mtime_ns,
            "ctime_ns": adopted_stat.st_ctime_ns,
            "nlink": adopted_stat.st_nlink,
        },
    }
    adoption: dict[str, object] = {
        "schema_version": reconciliation.EXACT_CHECKPOINT_ADOPTION_SCHEMA,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "stage_id": reconciliation.EXACT_STAGE,
        "checkpoint_snapshot": snapshot,
        "expected_progress_rows": 123,
        "science_writer_absent": True,
        "publication_sequence": [
            "producer_os_replace",
            "producer_parent_fsync",
            "verifier_o_nofollow_open",
            "verifier_fstat",
            "verifier_fd_sha256",
        ],
        "signals_sent": [],
        "verified_at": "2026-09-01T00:00:00+00:00",
    }
    adoption["receipt_sha256"] = reconciliation._stable_sha256(adoption)
    adoption_path = (
        controller
        / "gates"
        / f"89_exact_checkpoint_adoption_{snapshot['sha256_at_observation'][:16]}.json"
    )
    _json(adoption_path, adoption)
    adoption_path.chmod(0o600)
    final_checkpoint = checkpoint.with_suffix(".final.tmp")
    final_checkpoint.write_text("completed exact checkpoint\n", encoding="utf-8")
    os.replace(final_checkpoint, checkpoint)
    exact_evidence: dict[str, object] = {
        "status": "PASS",
        "path": str(exact_receipt.resolve()),
        "sha256": _sha(exact_receipt),
        "dbscan_manifest_path": str(dbscan.resolve()),
        "dbscan_manifest_sha256": _sha(dbscan),
        "proof_path": str((tmp_path / "science/proof.json").resolve()),
        "proof_sha256": "b" * 64,
        "linked_artifacts": {},
    }
    return (
        manifest_path,
        controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    )


def _patch_controller_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: dict[str, object],
    adoption_path: Path,
    adoption: dict[str, object],
    exact_evidence: dict[str, object],
) -> None:
    monkeypatch.setattr(
        reconciliation, "load_bound_controller_manifest", lambda _path: manifest
    )
    snapshot = adoption["checkpoint_snapshot"]
    checkpoint = Path(snapshot["path"])
    final_snapshot = {
        "path": str(checkpoint.resolve()),
        "sha256_at_observation": _sha(checkpoint),
        "identity_sha256": snapshot["identity_sha256"],
        "progress_rows": 456,
        "vectors_sha256": snapshot["vectors_sha256"],
    }
    monkeypatch.setattr(
        reconciliation,
        "_validated_exact_checkpoint_snapshot_and_artifact",
        lambda _stage: (
            dict(final_snapshot),
            {
                "path": str(checkpoint.resolve()),
                "content_sha256": _sha(checkpoint),
            },
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_validate_exact_receipt",
        lambda **_kwargs: dict(exact_evidence),
    )
    monkeypatch.setattr(
        reconciliation,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )


def test_exact_receipt_reopens_science_without_historical_typed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adoption_root = tmp_path / "adoption"
    adoption_receipt = adoption_root / "failed_selection_adoption_receipt.json"
    adoption_payload = {"status": "RECOVERY_ONLY_READY", "science": "frozen"}
    _json(adoption_receipt, adoption_payload)

    exact_root = tmp_path / "exact"
    proof_path = exact_root / "dbscan/shortcut_proof.json"
    proof = {
        "unique_seed_component_proven": True,
        "seed_component_count": 1,
        "all_points_core_proven": True,
        "exact_multicomponent_partition_proven": True,
        "all_progress_prefixes_complete": True,
    }
    _json(proof_path, proof)
    dbscan_path = exact_root / "dbscan/run_manifest.json"
    dbscan = {
        "run_complete": True,
        "clustering_path": reconciliation.ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
        "approximation_used": False,
        "num_samples": reconciliation.EXPECTED_ROWS,
        "core_count": reconciliation.EXPECTED_ROWS,
        "noise_count": 0,
        "shortcut_proof_path": str(proof_path.resolve()),
    }
    _json(dbscan_path, dbscan)
    linked_paths = {
        "promotion_manifest_path": exact_root / "dbscan/promotion.json",
        "source_evidence_receipt_path": exact_root / "source_evidence/receipt.json",
        "continuation_bootstrap_path": exact_root / "bootstrap.json",
    }
    for path in linked_paths.values():
        _json(path, {"status": "FROZEN"})
    exact_receipt = exact_root / "exact_recovery_receipt.json"
    receipt = {
        "schema_version": reconciliation.EXACT_STAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "dbscan_partition_proven": True,
        "observed_environment": {"device": "cpu"},
        "dbscan_manifest_path": str(dbscan_path.resolve()),
        "dbscan_manifest_sha256": _sha(dbscan_path),
    }
    for path_field, path in linked_paths.items():
        receipt[path_field] = str(path.resolve())
        receipt[path_field.replace("_path", "_sha256")] = _sha(path)
    _json(exact_receipt, receipt)
    manifest = {
        "stages": [
            {
                "stage_id": reconciliation.ADOPTION_STAGE,
                "terminal_path": str(adoption_receipt.resolve()),
            },
            {
                "stage_id": reconciliation.EXACT_STAGE,
                "terminal_path": str(exact_receipt.resolve()),
            },
        ]
    }
    # This is the historical gate that became operationally stale after the
    # host restart.  The posthoc path must not inspect it.
    _json(
        tmp_path / "controller/gates/01_failed_selection_adoption.json",
        {"writer_lock_identity": {"pid": 999999, "start_ticks": 1}},
    )
    monkeypatch.setattr(
        reconciliation,
        "_frozen_stage_environment",
        lambda _manifest: {"device": "cpu"},
    )
    monkeypatch.setattr(
        reconciliation,
        "_validate_component_recovery_closure",
        lambda **_kwargs: None,
    )

    def validate_adoption_directly(
        *, manifest: object, validator: object
    ) -> dict[str, object]:
        assert callable(validator)
        assert validator(output_dir=adoption_root) == adoption_payload
        return {
            "receipt_path": str(adoption_receipt.resolve()),
            "receipt_sha256": _sha(adoption_receipt),
        }

    def validate_exact_directly(
        observed_manifest: object, adoption_binding: object
    ) -> dict[str, object]:
        assert observed_manifest is manifest
        assert adoption_binding == {
            "artifact": {
                "path": str(adoption_receipt.resolve()),
                "sha256": _sha(adoption_receipt),
            }
        }
        return {
            "path": str(exact_receipt.resolve()),
            "sha256": _sha(exact_receipt),
            "stage_receipt": receipt,
            "manifest": dbscan,
            "proof": proof,
        }

    monkeypatch.setattr(
        reconciliation,
        "validate_typed_adoption_receipt",
        validate_adoption_directly,
    )
    monkeypatch.setattr(
        reconciliation,
        "_validate_exact_terminal",
        validate_exact_directly,
    )
    result = reconciliation._validate_exact_receipt(
        manifest=manifest,
        receipt_path=exact_receipt.resolve(),
    )
    assert result == {
        "status": "PASS",
        "path": str(exact_receipt.resolve()),
        "sha256": _sha(exact_receipt),
        "dbscan_manifest_path": str(dbscan_path.resolve()),
        "dbscan_manifest_sha256": _sha(dbscan_path),
        "proof_path": str(proof_path.resolve()),
        "proof_sha256": _sha(proof_path),
        "linked_artifacts": {
            path_field: {"path": str(path.resolve()), "sha256": _sha(path)}
            for path_field, path in linked_paths.items()
        },
    }


def test_historical_blocked_controller_binds_posthoc_exact_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        _controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    ) = _controller_fixture(tmp_path)
    _patch_controller_dependencies(
        monkeypatch,
        manifest=manifest,
        adoption_path=adoption_path,
        adoption=adoption,
        exact_evidence=exact_evidence,
    )
    result = reconciliation.validate_historical_controller_exact_authority(
        manifest_path,
        exact_receipt_path=exact_receipt,
        exact_adoption_gate_path=adoption_path,
        proc_root=proc,
    )
    assert result["historical_state"] == "BLOCKED_EXACT_COMPONENT_RECOVERY"
    assert result["stale_worker_projection_preserved"] is True
    assert result["stale_startup_barrier_preserved"] is True
    assert result["controller_process_alive"] is False
    assert result["exact_worker_alive"] is False
    assert result["science_writer_absent"] is True
    assert result["old_state_modified"] is False
    assert result["adoption_checkpoint_progress_rows"] == 123
    assert result["checkpoint_progress_rows"] == 456
    assert result["adoption_checkpoint_sha256"] != result["checkpoint_sha256"]
    assert result["checkpoint_monotonic_from_adoption"] is True


@pytest.mark.parametrize("failure", ["progress_regression", "identity_drift", "vector_drift"])
def test_historical_checkpoint_continuation_fails_closed_on_science_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    (
        manifest_path,
        _controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    ) = _controller_fixture(tmp_path)
    _patch_controller_dependencies(
        monkeypatch,
        manifest=manifest,
        adoption_path=adoption_path,
        adoption=adoption,
        exact_evidence=exact_evidence,
    )
    adopted = adoption["checkpoint_snapshot"]
    checkpoint = Path(adopted["path"])
    final = {
        "path": str(checkpoint.resolve()),
        "sha256_at_observation": _sha(checkpoint),
        "identity_sha256": adopted["identity_sha256"],
        "progress_rows": 456,
        "vectors_sha256": adopted["vectors_sha256"],
    }
    if failure == "progress_regression":
        final["progress_rows"] = 122
    elif failure == "identity_drift":
        final["identity_sha256"] = "f" * 64
    else:
        final["vectors_sha256"] = "f" * 64
    monkeypatch.setattr(
        reconciliation,
        "_validated_exact_checkpoint_snapshot_and_artifact",
        lambda _stage: (
            final,
            {
                "path": str(checkpoint.resolve()),
                "content_sha256": _sha(checkpoint),
            },
        ),
    )
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="regressed or changed scientific input",
    ):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )


def test_historical_exact_authority_rejects_relaxed_state_or_science_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    ) = _controller_fixture(tmp_path)
    _patch_controller_dependencies(
        monkeypatch,
        manifest=manifest,
        adoption_path=adoption_path,
        adoption=adoption,
        exact_evidence=exact_evidence,
    )
    state_path = controller / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "RUNNING"
    _json(state_path, state)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="historical BLOCKED-exact projection",
    ):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )

    state["status"] = "BLOCKED"
    _json(state_path, state)
    monkeypatch.setattr(
        reconciliation,
        "_validate_exact_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("DBSCAN proof failed")),
    )
    with pytest.raises(ValueError, match="DBSCAN proof failed"):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )


def test_historical_exact_authority_rejects_live_process_and_gate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        _controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    ) = _controller_fixture(tmp_path)
    _patch_controller_dependencies(
        monkeypatch,
        manifest=manifest,
        adoption_path=adoption_path,
        adoption=adoption,
        exact_evidence=exact_evidence,
    )
    monkeypatch.setattr(
        reconciliation,
        "_proc_start_ticks",
        lambda _proc, pid: 456 if pid == 321 else None,
    )
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="controller is still alive",
    ):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )
    monkeypatch.setattr(reconciliation, "_proc_start_ticks", lambda *_args: None)
    adoption["science_writer_absent"] = False
    adoption["receipt_sha256"] = reconciliation._stable_sha256(
        {key: value for key, value in adoption.items() if key != "receipt_sha256"}
    )
    _json(adoption_path, adoption)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="adoption contract changed",
    ):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )


def test_historical_exact_authority_rejects_ordinary_terminal_and_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        controller,
        proc,
        manifest,
        exact_receipt,
        adoption_path,
        adoption,
        exact_evidence,
    ) = _controller_fixture(tmp_path)
    _patch_controller_dependencies(
        monkeypatch,
        manifest=manifest,
        adoption_path=adoption_path,
        adoption=adoption,
        exact_evidence=exact_evidence,
    )
    (controller / "terminal.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="ordinary AIDS controller terminal exists",
    ):
        reconciliation.validate_historical_controller_exact_authority(
            manifest_path,
            exact_receipt_path=exact_receipt,
            exact_adoption_gate_path=adoption_path,
            proc_root=proc,
        )
    (controller / "terminal.json").unlink()
    with (controller / ".controller.lock").open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            reconciliation.AIDSComRecGCTerminalReconciliationError,
            match="lock is still held",
        ):
            reconciliation.validate_historical_controller_exact_authority(
                manifest_path,
                exact_receipt_path=exact_receipt,
                exact_adoption_gate_path=adoption_path,
                proc_root=proc,
            )


def _posthoc_source(
    tmp_path: Path, controller: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    root = _zero_source(tmp_path)
    (root / "PASS").write_bytes(b"PASS\n")
    source_generation = tmp_path / "generation"
    source_generation.mkdir()
    source_integrity = root / "source_integrity_final.json"
    _json(source_integrity, {"status": "PASS"})
    exact = controller["exact_receipt"]
    adoption_path = (
        root / "common_recourse/external_memory/dbscan_adoption/run_manifest.json"
    )
    adoption = {
        "status": "PASS",
        "run_complete": True,
        "source_access": "read_only",
        "source_mutated": False,
        "dbscan_recomputed": False,
        "pair_store_recomputed": False,
        "sklearn_float64_semantics_preserved": True,
        "exact_recovery_receipt_path": exact["path"],
        "exact_recovery_receipt_sha256": exact["sha256"],
        "source_manifest_path": exact["dbscan_manifest_path"],
        "source_manifest_sha256": exact["dbscan_manifest_sha256"],
    }
    _json(adoption_path, adoption)
    common_manifest = {
        "external_memory_artifacts": {
            "dbscan_adopted_read_only": True,
            "dbscan_adoption_manifest": str(adoption_path.resolve()),
            "dbscan_adoption_manifest_sha256": _sha(adoption_path),
        }
    }
    _json(root / "common_recourse/run_manifest.json", common_manifest)
    _json(root / "common_recourse/_RUN_COMPLETE.json", {"status": "PASS"})
    identities = {
        "oracle_hash": "a" * 64,
        "molclr_checkpoint_hash": "b" * 64,
        "dataset_hash": "c" * 64,
    }
    standardized = {
        "root": str((root / "standardized").resolve()),
        "source_evaluation_root": "/science/unified_eval",
        "run_manifest_sha256": _sha(root / "standardized/run_manifest.json"),
        "final_artifact_audit_sha256": _sha(
            root / "standardized/final_artifact_audit.json"
        ),
        "freeze_manifest_sha256": _sha(
            root / "standardized/freeze_manifest.json"
        ),
        "identities": identities,
    }
    continuation = {
        "schema_version": 1,
        "status": "PASS",
        "run_complete": True,
        "dataset": "aids",
        "method": "COMRECGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "generation_adopted": True,
        "generation_rerun": False,
        "ordering_adopted": False,
        "evaluation_adopted": False,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "standardized_output_root": str((root / "standardized").resolve()),
        "standardized_run_manifest_sha256": standardized[
            "run_manifest_sha256"
        ],
        "freeze_manifest_sha256": standardized["freeze_manifest_sha256"],
        "teacher_sha256": identities["oracle_hash"],
        "molclr_checkpoint_sha256": identities["molclr_checkpoint_hash"],
        "dataset_csv_sha256": identities["dataset_hash"],
        "source_generation_root": str(source_generation.resolve()),
        "source_integrity_final_sha256": _sha(source_integrity),
    }
    _json(root / "_RUN_COMPLETE.json", continuation)
    outer = dict(continuation)
    outer.pop("run_complete")
    _json(root / "run_manifest.json", outer)
    _json(root / "final_gate.json", outer)
    return root, standardized


def test_posthoc_final_self_closure_binds_exact_receipt_without_stage_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = {
        "controller_manifest_path": "/control/controller.manifest.json",
        "controller_manifest_sha256": "1" * 64,
        "posthoc_exact_adoption_path": "/control/adoption.json",
        "posthoc_exact_adoption_sha256": "2" * 64,
        "adoption_checkpoint_path": "/science/checkpoint.json",
        "adoption_checkpoint_sha256": "0" * 64,
        "adoption_checkpoint_progress_rows": 123,
        "checkpoint_path": "/science/checkpoint.json",
        "checkpoint_sha256": "3" * 64,
        "checkpoint_progress_rows": 456,
        "checkpoint_identity_sha256": "6" * 64,
        "checkpoint_vectors_sha256": "7" * 64,
        "checkpoint_monotonic_from_adoption": True,
        "exact_receipt": {
            "status": "PASS",
            "path": "/science/exact_recovery_receipt.json",
            "sha256": "4" * 64,
            "dbscan_manifest_path": "/science/dbscan/run_manifest.json",
            "dbscan_manifest_sha256": "5" * 64,
        },
    }
    root, standardized = _posthoc_source(tmp_path, controller)
    common_calls: list[dict[str, object]] = []

    def validate_common(**kwargs: object) -> dict[str, object]:
        common_calls.append(dict(kwargs))
        return {
            "pair_store_reopen_evidence": {
                "schema_version": "comrecgc_pair_store_reopen_evidence_v1",
                "policy": "AIDS_TERMINAL_RECONCILIATION_REMOUNT_DEVICE_ONLY",
                "remount_device_drift_allowed": True,
                "remount_device_drift_detected": True,
                "allowed_drift_fields": ["device"],
                "hashes_verified": True,
                "writer_scan_before_count": 0,
                "writer_scan_after_count": 0,
                "source_root_guard_verified": True,
                "stat_stable_during_reopen": True,
                "source_files": {
                    "/source/vectors.npy": {
                        "recorded_stat": {"device": 126},
                        "observed_stat": {"device": 76},
                        "device_changed": True,
                        "stable_stat_fields_match": True,
                    }
                },
            },
            "close_pair_view_reopen_evidence": {
                "schema_version": "comrecgc_close_pair_view_reopen_evidence_v1",
                "policy": "AIDS_TERMINAL_RECONCILIATION_REMOUNT_DEVICE_ONLY",
                "remount_device_drift_allowed": True,
                "remount_device_drift_detected": True,
                "allowed_drift_fields": ["device"],
                "hashes_verified": True,
                "writer_scan_before_count": 0,
                "writer_scan_after_count": 0,
                "stat_stable_during_reopen": True,
                "source_files": {
                    f"/source/{name}": {"device_changed": True}
                    for name in ("vectors.npy", "distances.npy", "pair.json")
                },
            },
            "dbscan_source_reopen_evidence": {
                "schema_version": "comrecgc_dbscan_source_reopen_evidence_v1",
                "policy": "AIDS_TERMINAL_RECONCILIATION_REMOUNT_DEVICE_ONLY",
                "remount_device_drift_allowed": True,
                "allowed_drift_fields": ["device"],
                "hashes_verified": True,
                "stat_stable_while_hashing": True,
            },
            "component_summary_reopen_evidence": {
                "schema_version": "comrecgc_component_summary_reopen_evidence_v1",
                "policy": "AIDS_TERMINAL_RECONCILIATION_REMOUNT_DEVICE_ONLY",
                "remount_device_drift_allowed": True,
                "allowed_drift_fields": ["device"],
                "no_active_writer_verified": True,
                "stat_stable_during_reopen": True,
                "close_pair_view": {"hashes_verified": True},
            },
        }

    monkeypatch.setattr(
        reconciliation, "_validate_common_recourse_completion", validate_common
    )
    monkeypatch.setattr(
        matrix, "_validate_aids_standardized", lambda _root: dict(standardized)
    )
    monkeypatch.setattr(
        matrix,
        "_writer_audit",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    result = reconciliation.validate_reconciled_final_science(
        root, controller_evidence=controller, proc_root=tmp_path
    )
    assert result["terminal_kind"] == "AIDS_POSTHOC_SELF_CLOSED_SCIENCE_FINAL"
    assert result["exact_receipt_sha256"] == "4" * 64
    assert result["zero_strict_flip_evidence"]["scientific_output_empty"] is True
    assert common_calls[0][
        "allow_pair_store_remount_device_drift_for_terminal_reconciliation"
    ] is True
    assert common_calls[0][
        "allow_close_view_and_downstream_remount_device_drift_for_aids_terminal_reconciliation"
    ] is True
    assert result["pair_store_reopen_evidence"][
        "remount_device_drift_detected"
    ] is True

    adoption_path = Path(result["dbscan_adoption_manifest_path"])
    adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
    adoption["exact_recovery_receipt_sha256"] = "0" * 64
    _json(adoption_path, adoption)
    common_path = root / "common_recourse/run_manifest.json"
    common = json.loads(common_path.read_text(encoding="utf-8"))
    common["external_memory_artifacts"][
        "dbscan_adoption_manifest_sha256"
    ] = _sha(adoption_path)
    _json(common_path, common)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="final/exact receipt binding changed",
    ):
        reconciliation.validate_reconciled_final_science(
            root, controller_evidence=controller, proc_root=tmp_path
        )


def test_fresh_wrapper_is_hash_closed_and_detects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _zero_source(tmp_path)
    zero = reconciliation.validate_zero_strict_flip_science(source)
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    exact_science_root = tmp_path / "exact-science"
    exact_science_root.mkdir()
    controller = {
        "status": "PASS",
        "controller_manifest_path": "/control/controller.manifest.json",
        "controller_manifest_sha256": "1" * 64,
        "controller_root": str(controller_root.resolve()),
        "exact_science_root": str(exact_science_root.resolve()),
        "historical_state": "BLOCKED_EXACT_COMPONENT_RECOVERY",
        "stale_worker_projection_preserved": True,
        "stale_startup_barrier_preserved": True,
        "controller_terminal_present": False,
        "controller_pass_marker_present": False,
        "controller_process_alive": False,
        "exact_worker_alive": False,
        "exact_process_group_alive": False,
        "controller_lock_held": False,
        "science_writer_absent": True,
        "old_state_modified": False,
        "posthoc_exact_adoption_path": "/control/adoption.json",
        "posthoc_exact_adoption_sha256": "2" * 64,
        "adoption_checkpoint_path": "/science/checkpoint.json",
        "adoption_checkpoint_sha256": "0" * 64,
        "adoption_checkpoint_progress_rows": 123,
        "checkpoint_path": "/science/checkpoint.json",
        "checkpoint_sha256": "3" * 64,
        "checkpoint_progress_rows": 456,
        "checkpoint_identity_sha256": "6" * 64,
        "checkpoint_vectors_sha256": "7" * 64,
        "checkpoint_monotonic_from_adoption": True,
        "exact_receipt": {
            "status": "PASS",
            "path": "/science/exact.json",
            "sha256": "4" * 64,
            "dbscan_manifest_path": "/science/dbscan.json",
            "dbscan_manifest_sha256": "5" * 64,
        },
        "controller_restart_performed": False,
    }
    science = _science_evidence(source, controller)
    monkeypatch.setattr(
        reconciliation,
        "validate_historical_controller_exact_authority",
        lambda *_args, **_kwargs: dict(controller),
    )
    monkeypatch.setattr(
        reconciliation,
        "validate_reconciled_final_science",
        lambda root, **_kwargs: _science_evidence(Path(root), controller),
    )
    wrapper = tmp_path / "wrapper"
    receipt = reconciliation.publish_reconciliation(
        output_root=wrapper,
        science_projection=reconciliation.science_terminal_projection(science),
        controller_evidence=controller,
        zero_evidence=zero,
        proc_root=tmp_path,
    )
    assert (wrapper / "PASS").read_bytes() == b"PASS\n"
    assert reconciliation.validate_reconciliation_root(wrapper, proc_root=tmp_path) == receipt
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="overlaps the read-only controller root",
    ):
        reconciliation.publish_reconciliation(
            output_root=controller_root / "forbidden-wrapper",
            science_projection=reconciliation.science_terminal_projection(science),
            controller_evidence=controller,
            zero_evidence=zero,
            proc_root=tmp_path,
        )
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="overlaps the read-only exact science root",
    ):
        reconciliation.publish_reconciliation(
            output_root=exact_science_root / "forbidden-wrapper",
            science_projection=reconciliation.science_terminal_projection(science),
            controller_evidence=controller,
            zero_evidence=zero,
            proc_root=tmp_path,
        )
    figure = source / "standardized/figure3_coverage_vs_k.csv"
    figure.write_text(figure.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="projection changed|evidence changed",
    ):
        reconciliation.validate_reconciliation_root(wrapper, proc_root=tmp_path)


def test_matrix_dispatch_accepts_only_matching_reconciliation_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "PASS").write_bytes(b"PASS\n")
    _json(wrapper / "run_manifest.json", {"schema_version": reconciliation.RECONCILIATION_SCHEMA})
    source = tmp_path / "science"
    source.mkdir()
    controller_manifest = tmp_path / "controller.json"
    controller_manifest.write_text("{}\n", encoding="utf-8")
    science = {
        "root": str(source.resolve()),
        "standardized": {"root": str((source / "standardized").resolve())},
        "inventory": {"PASS": {}},
    }
    projection = {"root": str(source.resolve()), "static": True}
    receipt = {
        "source_science_root": str(source.resolve()),
        "controller_terminal_reconciliation": {
            "controller_manifest_path": str(controller_manifest.resolve()),
            "posthoc_exact_adoption_path": "/control/adoption.json",
            "exact_receipt": {"path": "/science/exact.json"},
        },
        "science_terminal_projection": projection,
        "reconciliation_sha256": "f" * 64,
    }
    monkeypatch.setattr(matrix, "validate_aids_reconciliation_root", lambda *_a, **_k: receipt)
    monkeypatch.setattr(
        matrix,
        "validate_aids_historical",
        lambda *_a, **_k: receipt["controller_terminal_reconciliation"],
    )
    monkeypatch.setattr(
        matrix,
        "validate_aids_reconciled_final_science",
        lambda *_a, **_k: dict(science),
    )
    monkeypatch.setattr(
        matrix,
        "_validate_aids_science_terminal",
        lambda *_a, **_k: pytest.fail(
            "historical BLOCKED reconciliation must not use controller stage gates"
        ),
    )
    monkeypatch.setattr(matrix, "aids_science_terminal_projection", lambda _value: projection)
    monkeypatch.setattr(matrix, "_critical_inventory", lambda *_a, **_k: {})
    result = matrix._validate_aids_terminal(
        wrapper,
        controller_manifest_path=controller_manifest,
        proc_root=tmp_path,
        require_writer_audit=False,
    )
    assert result["root"] == str(source.resolve())
    assert result["terminal_kind"] == "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION"
    assert result["numeric_imputation_used"] is False

    receipt["science_terminal_projection"] = {"root": str(source.resolve()), "static": False}
    with pytest.raises(matrix.NonTasteMatrixAppendError, match="projection changed"):
        matrix._validate_aids_terminal(
            wrapper,
            controller_manifest_path=controller_manifest,
            proc_root=tmp_path,
            require_writer_audit=False,
        )


def test_registry_mismatch_is_only_unavailable_conditional_cost(tmp_path: Path) -> None:
    source = _zero_source(tmp_path)
    _, reasons, k_max, table2_k, _ = registry._validate_standardized_csvs(
        source / "standardized", "ComRecGC"
    )
    assert set(reasons) == {
        "FIGURE3_INVALID:ValueError",
        "TABLE2_INVALID:ValueError",
    }
    assert k_max == 20
    assert table2_k == 10


def test_registry_promotes_only_exact_reconciled_zero_cost_mismatch() -> None:
    terminal = {
        "terminal_kind": "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "coverage": 0.0,
        "conditional_cost_available": False,
        "numeric_imputation_used": False,
        "registry_numeric_imputation_used": False,
    }
    target = {
        "dataset": "AIDS",
        "method": "ComRecGC",
        "status": registry.CellStatus.INCOMPLETE.value,
        "k_max": 20,
        "table2_k": 10,
        "adoption_reason": "",
        "rerun_reason": (
            "FIGURE3_INVALID:ValueError;TABLE2_INVALID:ValueError"
        ),
    }
    promoted = matrix._reconcile_aids_zero_registry_row(target, terminal=terminal)
    assert promoted["status"] == registry.CellStatus.FROZEN_PASS.value
    assert promoted["rerun_reason"] == ""
    assert "no numeric value was imputed" in promoted["adoption_reason"]

    target["rerun_reason"] += ";MISSING_OR_EMPTY:summary.json"
    rejected = matrix._reconcile_aids_zero_registry_row(target, terminal=terminal)
    assert rejected["status"] == registry.CellStatus.INCOMPLETE.value
    assert "MISSING_OR_EMPTY" in rejected["rerun_reason"]


def _threshold_grid_row(
    root: Path,
    *,
    dataset: str,
    method: str,
    threshold_hash: str,
    thresholds: list[str],
    coverages: list[str],
) -> dict[str, object]:
    _csv(
        root / "figure4_coverage_vs_threshold.csv",
        [
            {
                "method": method,
                "threshold": threshold,
                "close_cf_coverage": coverage,
            }
            for threshold, coverage in zip(thresholds, coverages, strict=True)
        ],
    )
    return {
        "dataset": dataset,
        "method": method,
        "standardized_output_root": str(root.resolve()),
        "threshold_config_hash": threshold_hash,
    }


def test_aids_zero_threshold_grid_accepts_only_decimal_serialization_drift(
    tmp_path: Path,
) -> None:
    target_hash = "a" * 64
    reference_hash = "b" * 64
    target = _threshold_grid_row(
        tmp_path / "target",
        dataset="AIDS",
        method="ComRecGC",
        threshold_hash=target_hash,
        thresholds=[
            "0",
            "0.017833333333333333",
            "0.035666666666666666",
            "0.0535",
        ],
        coverages=["0", "0.0", "0.000", "0"],
    )
    reference = _threshold_grid_row(
        tmp_path / "reference",
        dataset="AIDS",
        method="Ours",
        threshold_hash=reference_hash,
        thresholds=[
            "0.0",
            "0.0178333333333333",
            "0.0356666666666667",
            "0.0535",
        ],
        coverages=["0.2", "0.3", "0.4", "0.5"],
    )

    evidence = matrix._validate_aids_zero_threshold_grid_equivalence(
        target, reference
    )

    assert evidence["status"] == "PASS"
    assert evidence["same_start"] is True
    assert evidence["same_end"] is True
    assert evidence["same_count"] is True
    assert evidence["both_canonical_equidistant"] is True
    assert evidence["target_all_coverages_zero"] is True
    assert evidence["target_threshold_config_hash_original"] == target_hash
    assert evidence["reference_threshold_config_hash_original"] == reference_hash
    assert evidence["hashes_rewritten"] is False


def test_aids_zero_threshold_grid_rejects_real_grid_drift(tmp_path: Path) -> None:
    target = _threshold_grid_row(
        tmp_path / "target",
        dataset="AIDS",
        method="ComRecGC",
        threshold_hash="a" * 64,
        thresholds=["0", "0.017833333333333333", "0.035666666666666666", "0.0535"],
        coverages=["0", "0", "0", "0"],
    )
    reference = _threshold_grid_row(
        tmp_path / "reference",
        dataset="AIDS",
        method="Ours",
        threshold_hash="b" * 64,
        thresholds=["0", "0.0178333333333333", "0.0356663333333333", "0.0535"],
        coverages=["0.2", "0.3", "0.4", "0.5"],
    )

    with pytest.raises(
        matrix.NonTasteMatrixAppendError,
        match="not the canonical equidistant grid",
    ):
        matrix._validate_aids_zero_threshold_grid_equivalence(target, reference)


def test_aids_zero_threshold_hash_mismatch_requires_bound_grid_receipt() -> None:
    terminal = {
        "terminal_kind": "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "coverage": 0.0,
        "conditional_cost_available": False,
        "numeric_imputation_used": False,
        "registry_numeric_imputation_used": False,
    }
    target_hash = "a" * 64
    reference_hash = "b" * 64
    target = {
        "dataset": "AIDS",
        "method": "ComRecGC",
        "status": registry.CellStatus.INCOMPLETE.value,
        "k_max": 20,
        "table2_k": 10,
        "threshold_config_hash": target_hash,
        "adoption_reason": "",
        "rerun_reason": (
            "EXPECTED_THRESHOLD_CONFIG_HASH_MISMATCH;"
            "FIGURE3_INVALID:ValueError;TABLE2_INVALID:ValueError"
        ),
    }
    assert matrix._reconcile_aids_zero_registry_row(
        target, terminal=terminal
    )["status"] == registry.CellStatus.INCOMPLETE.value

    evidence = {
        "schema_version": matrix._AIDS_ZERO_THRESHOLD_GRID_EQUIVALENCE_SCHEMA,
        "status": "PASS",
        "scope": "AIDS_COMRECGC_VALID_ZERO_RESULT_ONLY",
        "target_all_coverages_zero": True,
        "target_threshold_config_hash_original": target_hash,
        "reference_threshold_config_hash_original": reference_hash,
        "hashes_rewritten": False,
    }
    promoted = matrix._reconcile_aids_zero_registry_row(
        target,
        terminal=terminal,
        threshold_equivalence=evidence,
    )
    assert promoted["status"] == registry.CellStatus.FROZEN_PASS.value
    assert promoted["threshold_config_hash"] == target_hash

    shared = {
        field: f"value-{field}"
        for field in matrix._RF_SHARED_FIELDS
        if field != "threshold_config_hash"
    }
    compatibility = matrix._identity_compatibility(
        dataset="AIDS",
        target={**shared, "threshold_config_hash": target_hash},
        reference={
            **shared,
            "threshold_config_hash": reference_hash,
            "status": registry.CellStatus.FROZEN_PASS.value,
        },
        equivalent_mismatches={"threshold_config_hash": evidence},
    )
    assert compatibility["equivalent_mismatched_fields"] == [
        "threshold_config_hash"
    ]
    assert compatibility["equivalent_mismatch_evidence"][
        "threshold_config_hash"
    ] == evidence


def test_cli_and_slurm_keep_publication_on_shared_pointer() -> None:
    args = build_parser().parse_args(
        [
            "publish",
            "--controller-manifest",
            "/control/controller.json",
            "--exact-receipt",
            "/science/exact_recovery_receipt.json",
            "--exact-adoption-gate",
            "/control/gates/89_exact_checkpoint_adoption.json",
            "--reconciliation-root",
            "/runtime/reconciliation",
            "--matrix-output-root",
            "/runtime/authority-next",
        ]
    )
    assert args.authority_state_path == Path(
        "/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json"
    )
    script = Path("scripts/slurm/reconcile_aids_comrecgc_terminal_publication.sh").read_text(
        encoding="utf-8"
    )
    for token in (
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
        "--controller-manifest,--exact-receipt,--exact-adoption-gate",
    ):
        assert token in script
