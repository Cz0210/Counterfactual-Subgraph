from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.eval import tastemolnet_t4_oracle_smoke_v2 as t4
from src.eval import tastemolnet_t3_calibration_v2 as t3
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.terminal_publisher_v2 import seal_worker_staging
from tests.autodl.test_tastemolnet_t3_calibration_v2 import (
    _make_source_bundle,
    _make_t2_receipt,
)


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def _documents() -> dict[str, dict[str, object]]:
    smoke: dict[str, object] = {
        "schema_version": "tastemolnet_t4_bounded_oracle_smoke_metrics_v1",
        "status": "PASS",
        "selected_count": 16,
        "parent_deletion_counts_by_position": [4] * 16,
        "valid_deletion_count": 64,
        "all_selected_true_source": True,
        "all_selected_predicted_source": True,
        "all_selected_have_four_connected_deletions": True,
        "checkpoint_load_count": 1,
        "num_classes": 3,
        "source_label": 1,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "all_three_probabilities_validated": True,
        "batch_single_max_abs_difference": 0.0,
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
        "per_example_predictions_written": False,
        "smiles_written": False,
        "molecule_identifiers_written": False,
        "destination_distribution": {
            "overall": {
                "transitions": {
                    "1->0": {"count": 5},
                    "1->1": {"count": 54},
                    "1->2": {"count": 5},
                }
            }
        },
    }
    t3_binding: dict[str, object] = {
        "schema_version": "tastemolnet_t4_t3_binding_v2",
        "t3_root": "/private/seed7/calibrated-fixture",
        "t3_gate_sha256": "1" * 64,
        "t3_verification_sha256": "2" * 64,
        "t3_root_inventory_sha256": "3" * 64,
        "checkpoint_dir": "/private/seed7/calibrated-fixture/artifacts/checkpoint",
        "checkpoint_sha256s_sha256": "4" * 64,
        "checkpoint_id": "5" * 64,
        "model_sha256": "5" * 64,
        "temperature": 1.25,
        "temperature_scaling_sha256": "6" * 64,
        "feature_schema_file_sha256": "7" * 64,
        "feature_schema_sha256": "8" * 64,
        "temperature_refit_performed": True,
        "selection_split": "validation",
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
    }
    provenance: dict[str, object] = {
        "schema_version": "tastemolnet_t4_oracle_provenance_v2",
        "physical_gpu_index": 2,
        "physical_gpu_uuid": GPU_UUID,
        "visible_device": "cuda:0",
        "cuda_visible_devices": "2",
        "checkpoint_load_count": 1,
        "model_sha256": "5" * 64,
        "temperature_scaling_sha256": "6" * 64,
    }
    access: dict[str, object] = {
        "schema_version": "tastemolnet_t4_data_access_v2",
        "opened_payload_splits": ["calibration"],
        "train_payload_opened": False,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "csv_payload_opened": False,
        "per_example_output_written": False,
    }
    return {
        "oracle_smoke.json": smoke,
        "oracle_provenance.json": provenance,
        "data_access_manifest.json": access,
        "t3_binding.json": t3_binding,
    }


INPUT_HASHES = {
    "t3_gate": "1" * 64,
    "t3_verification": "2" * 64,
    "graph_cache_manifest": "9" * 64,
    "calibration_cache": "a" * 64,
}


def _runner(**_kwargs: object) -> t4.T4ScienceRun:
    return t4.T4ScienceRun(
        documents=copy.deepcopy(_documents()),
        input_hashes=dict(INPUT_HASHES),
        _revalidate=lambda: None,
        _close=lambda: None,
    )


def test_worker_source_has_no_method_pass_authority() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (
        root / "scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py"
    ).read_text(encoding="utf-8")
    verifier = (
        root / "scripts/autodl/tastemolnet_t4_oracle_smoke_verifier_v2.py"
    ).read_text(encoding="utf-8")
    assert t4.PASS_MARKER not in worker
    assert "print(PASS_MARKER" in verifier


def test_t4_holds_the_real_managed_t3_publication_shape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    seed_root = tmp_path / "outputs/gnn_oracles/tastemolnet/gine/seed7"
    bundle, hashes, row_hash = _make_source_bundle(seed_root / "full-source")
    receipt = _make_t2_receipt(
        tmp_path / "control/adoptions/T2_GINE/00000000-0000-4000-8000-000000000001",
        bundle=bundle,
        hashes=hashes,
        row_hash=row_hash,
    )
    controller_id = "taste-main-v2-test"
    git_commit = "d" * 40
    config_hash = "e" * 64
    input_hashes = {
        "t2_receipt_gate": hashlib.sha256(
            (receipt / "gate.json").read_bytes()
        ).hexdigest(),
        "t2_source_evidence": json.loads(
            (receipt / "source_evidence.json").read_text(encoding="utf-8")
        )["source_evidence_sha256"],
        "t2_source_sha256s": hashes["sha256sums.txt"],
    }
    stage_root = tmp_path / "control/T3_CALIBRATION"
    stage_root.mkdir(parents=True)
    with create_managed_attempt(
        stage_root=stage_root,
        controller_id=controller_id,
        task_id=t3.TASK_ID,
        git_commit=git_commit,
        config_hash=config_hash,
        input_hashes=input_hashes,
        attempt_id="00000000-0000-4000-8000-000000000020",
        boot_id="test-boot",
    ) as attempt:
        with create_worker_staging(
            attempt, staging_id="00000000-0000-4000-8000-000000000021"
        ) as staging:
            t3.build_t3_candidate(
                t2_receipt_root=receipt,
                source_bundle_root=bundle,
                artifact_root=staging.artifact_root,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                max_iter=10,
            )
            raw = write_worker_raw_evidence(
                staging,
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "process_lineage": {
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            raw.close()
            exited = write_worker_exit(
                staging,
                {
                    "exit_code": 0,
                    "worker_closed_artifact_writers": True,
                    "process_audit": {
                        "state": "EXITED",
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            exited.close()
            sealed = seal_worker_staging(staging)
        final = seed_root / "calibrated-fixture"
        t3.verify_and_publish_t3(
            sealed_path=sealed.staging_path,
            final_path=final,
            t2_receipt_root=receipt,
            source_bundle_root=bundle,
            expected_attempt_id=sealed.attempt_id,
            expected_generation_token=sealed.generation_token,
            expected_controller_id=controller_id,
            expected_git_commit=git_commit,
            expected_config_hash=config_hash,
            max_iter=10,
        )
    held = t4.HeldPublishedT3(final)
    try:
        assert held.binding["checkpoint_dir"] == str(final / "artifacts/checkpoint")
        assert held.binding["temperature_refit_performed"] is True
        held.verify()
    finally:
        held.close()


def test_aggregate_guard_rejects_row_level_smiles() -> None:
    documents = _documents()
    documents["oracle_smoke.json"]["rows"] = [{"smiles": "CC"}]
    with pytest.raises(t4.TasteT4OracleSmokeError, match="row-level"):
        t4._validate_documents(documents)


def test_managed_worker_and_independent_replay_publish(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    seed_root = tmp_path / "outputs/gnn_oracles/tastemolnet/gine/seed7"
    t3_root = seed_root / "calibrated-fixture"
    t3_root.mkdir(parents=True)
    graph_cache = tmp_path / "private-cache"
    graph_cache.mkdir()
    stage_root = tmp_path / "control/T4_ORACLE_SMOKE"
    stage_root.mkdir(parents=True)
    controller_id = "taste-main-v2-test"
    git_commit = "b" * 40
    config_hash = "c" * 64
    with create_managed_attempt(
        stage_root=stage_root,
        controller_id=controller_id,
        task_id=t4.TASK_ID,
        git_commit=git_commit,
        config_hash=config_hash,
        input_hashes=INPUT_HASHES,
        attempt_id="00000000-0000-4000-8000-000000000010",
        boot_id="test-boot",
    ) as attempt:
        with create_worker_staging(
            attempt, staging_id="00000000-0000-4000-8000-000000000011"
        ) as staging:
            t4.build_t4_candidate(
                t3_root=t3_root,
                graph_cache_root=graph_cache,
                artifact_root=staging.artifact_root,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                gpu_uuid=GPU_UUID,
                science_runner=_runner,
            )
            raw = write_worker_raw_evidence(
                staging,
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "process_lineage": {
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            raw.close()
            exited = write_worker_exit(
                staging,
                {
                    "exit_code": 0,
                    "worker_closed_artifact_writers": True,
                    "process_audit": {
                        "state": "EXITED",
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            exited.close()
            sealed = seal_worker_staging(staging)
        final = seed_root / "t4-oracle-smoke-fixture"
        publication, verification = t4.verify_and_publish_t4(
            sealed_path=sealed.staging_path,
            final_path=final,
            t3_root=t3_root,
            graph_cache_root=graph_cache,
            gpu_uuid=GPU_UUID,
            expected_attempt_id=sealed.attempt_id,
            expected_generation_token=sealed.generation_token,
            expected_controller_id=controller_id,
            expected_git_commit=git_commit,
            expected_config_hash=config_hash,
            science_runner=_runner,
        )
    assert publication.final_path == final
    assert verification["marker"] == t4.PASS_MARKER
    assert verification["physical_gpu_index"] == 2
    assert verification["valid_deletion_count"] == 64
    assert load_verified_gate(final)["status"] == "PASS"
    assert (final / "PASS").read_text(encoding="utf-8") == "[MANAGED_EXECUTION_V2_PASS]\n"
    assert not (final / "artifacts/gate.json").exists()


def test_independent_replay_rejects_science_drift(tmp_path: Path, monkeypatch) -> None:
    documents = _documents()
    t4._validate_documents(documents)
    changed = copy.deepcopy(documents)
    changed["oracle_smoke.json"]["batch_single_max_abs_difference"] = 1e-3
    assert not t4._equivalent(documents, changed)
