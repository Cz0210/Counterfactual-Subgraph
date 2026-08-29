from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from types import SimpleNamespace

import pytest

from src.eval import tastemolnet_t4_oracle_smoke_v2 as t4
from src.eval import tastemolnet_t3_calibration_v2 as t3
from src.eval.counterfactual_semantics import (
    compute_counterfactual_semantics,
    strict_flip,
)
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils import autodl_tastemolnet_main_v2 as main_v2
from src.utils.process_identity_v2 import ProcessSnapshotV2, capture_process_snapshot
from tests.autodl.test_tastemolnet_main_v2_controller import GPU_INVENTORY, _policy
from src.utils.terminal_publisher_v2 import seal_worker_staging
from tests.autodl.test_tastemolnet_t3_calibration_v2 import (
    _make_source_bundle,
    _make_t2_receipt,
)


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def _documents() -> dict[str, object]:
    schedule = [
        {
            "round": index,
            "parent_limit": parent_limit,
            "deletion_cap_per_parent": deletion_cap,
        }
        for index, (parent_limit, deletion_cap) in enumerate(
            t4.SEARCH_SCHEDULE, start=1
        )
    ]
    smoke: dict[str, object] = {
        "schema_version": "tastemolnet_t4_adaptive_oracle_smoke_metrics_v1",
        "status": "PASS",
        "adaptive_calibration_search": True,
        "search_schedule": schedule,
        "rounds_executed": [
            {
                "round": 1,
                "parent_limit": 16,
                "deletion_cap_per_parent": 8,
                "selected_count": 16,
                "valid_deletion_count": 128,
                "strict_flip_count": 20,
                "distinct_flipped_parent_count": 10,
                "destination_0_count": 15,
                "destination_2_count": 5,
                "gate_pass": True,
            }
        ],
        "terminal_round": 1,
        "selected_count": 16,
        "parent_deletion_counts_by_position": [8] * 16,
        "deletion_cap_per_parent": 8,
        "valid_deletion_count": 128,
        "batch_examples": 16,
        "at_least_one_connected_deletion": True,
        "all_selected_true_source": True,
        "all_selected_predicted_source": True,
        "all_selected_have_connected_deletions": True,
        "checkpoint_load_count": 1,
        "num_classes": 3,
        "source_label": 1,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "strict_flip_count": 20,
        "minimum_strict_flip_count": 16,
        "distinct_flipped_parent_count": 10,
        "minimum_distinct_flipped_parent_count": 8,
        "strict_flip_gate_pass": True,
        "all_three_probabilities_validated": True,
        "all_three_logits_validated": True,
        "three_class_api_validated": True,
        "batch_single_max_abs_difference": 0.0,
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
        "calibration_payload_loaded": True,
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "per_example_predictions_written": False,
        "smiles_written": False,
        "molecule_identifiers_written": False,
        "destination_0_count": 15,
        "destination_2_count": 5,
        "observed_destination_labels": [0, 2],
        "destination_diversity_status": "DESTINATION_DIVERSITY_PASS",
        "destination_diversity_single_class_warning": False,
        "strict_flip_to_bitter_observed": True,
        "strict_flip_to_tasteless_observed": True,
        "destination_distribution": {
            "schema_version": 1,
            "source_label": 1,
            "num_classes": 3,
            "overall": {
                "total_records": 128,
                "total_strict_flips": 20,
                "transitions": {
                    "1->0": {"count": 15},
                    "1->2": {"count": 5},
                }
            },
            "by_rule": {},
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
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU_UUID,
        "visible_device": "cuda:0",
        "cuda_visible_devices": "1",
        "checkpoint_load_count": 1,
        "model_sha256": "5" * 64,
        "temperature_scaling_sha256": "6" * 64,
        "adaptive_calibration_search": True,
        "search_schedule": schedule,
        "minimum_strict_flip_count": 16,
        "minimum_distinct_flipped_parent_count": 8,
        "destination_diversity_required": False,
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
    documents: dict[str, object] = {
        "oracle_smoke.json": smoke,
        "oracle_provenance.json": provenance,
        "data_access_manifest.json": access,
        "t3_binding.json": t3_binding,
    }
    documents[t4.DESTINATION_DISTRIBUTION_NAME] = (
        t4._destination_distribution_csv(smoke)
    )
    return documents


INPUT_HASHES = {
    "t3_gate": "1" * 64,
    "t3_verification": "2" * 64,
    "graph_cache_manifest": "9" * 64,
    "calibration_cache": "a" * 64,
}


def _controller_authority(
    tmp_path: Path,
    *,
    git_commit: str,
) -> tuple[
    main_v2.ControllerCreation,
    main_v2.GpuLeaseCreation,
    main_v2.HeartbeatCreation,
    str,
    str,
]:
    data_root = tmp_path.resolve() / "controller-authority"
    data_root.mkdir()
    controller_uuid = str(uuid.uuid4())
    controller_id = f"taste-main-v2-{controller_uuid}"
    git_tree = "f" * 40
    snapshot = capture_process_snapshot(os.getpid())
    policy = _policy(data_root)
    controllers, launchers = main_v2.ensure_controller_namespace_parents(
        policy["persistent_control_root"]
    )
    launcher_root = launchers / controller_uuid
    launcher_snapshot = ProcessSnapshotV2(
        pid=snapshot.ppid,
        ppid=1,
        pid_start_ticks=snapshot.pid_start_ticks + 1,
        boot_id=snapshot.boot_id,
        executable_realpath=snapshot.executable_realpath,
        command=snapshot.command,
        command_hash=snapshot.command_hash,
        cwd_realpath=snapshot.cwd_realpath,
        cgroup_path=snapshot.cgroup_path,
    )
    main_v2._create_fresh_namespace(
        launcher_root, children=(main_v2.PUBLICATION_STAGING_DIRECTORY,)
    )
    launcher_payload = {
        "schema_version": main_v2.LAUNCHER_RECEIPT_SCHEMA,
        "managed_taste_release_version": main_v2.MANAGED_TASTE_RELEASE_VERSION,
        "controller_id": controller_id,
        "controller_uuid": controller_uuid,
        "launcher_generation_token": str(uuid.uuid4()),
        "launcher_process": launcher_snapshot.to_dict(),
        "controller_process": snapshot.to_dict(),
        "git_commit": git_commit,
        "git_tree": git_tree,
        "project_root": str(Path.cwd().resolve()),
        "policy_facts": policy,
        "policy_facts_sha256": main_v2._sha256(main_v2._json_bytes(policy)),
        "state": "CONTROLLER_SPAWNED",
        "created_at": "2026-08-28T00:00:00Z",
        "created_at_ns": 1,
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    launcher_path = launcher_root / main_v2.LAUNCHER_RECEIPT_NAME
    launcher_data = main_v2._json_bytes(launcher_payload)
    main_v2._publish_immutable(
        launcher_path,
        launcher_data,
        staging_root=launcher_root / main_v2.PUBLICATION_STAGING_DIRECTORY,
    )
    launcher = SimpleNamespace(
        receipt_path=launcher_path,
        receipt_sha256=main_v2._sha256(launcher_data),
    )
    created = main_v2.create_controller_receipt(
        controller_root=controllers / controller_uuid,
        project_root=Path.cwd(),
        controller_id=controller_id,
        controller_uuid=controller_uuid,
        launcher_receipt_path=launcher.receipt_path,
        expected_launcher_receipt_sha256=launcher.receipt_sha256,
        git_identity=(git_commit, git_tree),
        process_snapshot=snapshot,
        policy_facts=policy,
    )
    lease = main_v2.create_gpu_lease_request(
        controller_receipt_path=created.receipt_path,
        task_id=t4.TASK_ID,
        physical_gpu_index=t4.PHYSICAL_GPU_INDEX,
        physical_gpu_uuid=GPU_UUID,
    )
    heartbeat = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=1,
        previous_heartbeat_sha256=None,
        gpu_inventory=GPU_INVENTORY,
    )
    return created, lease, heartbeat, controller_id, git_tree


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
    assert t4.SINGLE_DESTINATION_WARNING_MARKER not in worker
    assert "print(SINGLE_DESTINATION_WARNING_MARKER" in verifier
    assert "print(PASS_MARKER" in verifier


def test_managed_input_collector_requires_exact_published_t3_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(t4.TasteT4OracleSmokeError, match="exact reviewed"):
        t4.collect_t4_managed_input_hashes(
            t3_root=tmp_path.resolve(),
            graph_cache_root=tmp_path.resolve(),
            controller_launcher_receipt_sha256="1" * 64,
            controller_receipt_sha256="2" * 64,
            controller_anchor_heartbeat_sha256="3" * 64,
            gpu_lease_sha256="4" * 64,
        )


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


@pytest.mark.parametrize(
    ("pred_after", "expected"),
    [(0, True), (2, True), (1, False)],
)
def test_t4_strict_flip_unit_fixtures(pred_after: int, expected: bool) -> None:
    probabilities = {
        0: [0.8, 0.1, 0.1],
        1: [0.1, 0.8, 0.1],
        2: [0.1, 0.1, 0.8],
    }
    result = compute_counterfactual_semantics(
        source_label=1,
        pred_before=1,
        pred_after=pred_after,
        probabilities_before=probabilities[1],
        probabilities_after=probabilities[pred_after],
    )
    assert strict_flip(1, pred_after, 1) is expected
    assert result.cf_flip is expected


def test_t4_single_destination_is_a_nonblocking_warning() -> None:
    documents = _documents()
    smoke = documents["oracle_smoke.json"]
    assert isinstance(smoke, dict)
    smoke["rounds_executed"][0]["destination_0_count"] = 20
    smoke["rounds_executed"][0]["destination_2_count"] = 0
    smoke["destination_0_count"] = 20
    smoke["destination_2_count"] = 0
    smoke["observed_destination_labels"] = [0]
    smoke["destination_diversity_status"] = (
        "DESTINATION_DIVERSITY_SINGLE_CLASS_WARNING"
    )
    smoke["destination_diversity_single_class_warning"] = True
    smoke["strict_flip_to_bitter_observed"] = True
    smoke["strict_flip_to_tasteless_observed"] = False
    smoke["destination_distribution"]["overall"]["transitions"]["1->0"][
        "count"
    ] = 20
    smoke["destination_distribution"]["overall"]["transitions"]["1->2"][
        "count"
    ] = 0
    documents[t4.DESTINATION_DISTRIBUTION_NAME] = (
        t4._destination_distribution_csv(smoke)
    )
    t4._validate_documents(documents)


def test_t4_zero_strict_flip_gate_is_rejected() -> None:
    documents = _documents()
    smoke = documents["oracle_smoke.json"]
    assert isinstance(smoke, dict)
    smoke["strict_flip_count"] = 0
    with pytest.raises(t4.TasteT4OracleSmokeError, match="adaptive aggregate"):
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
    git_commit = "b" * 40
    config_hash = "c" * 64
    controller, lease, heartbeat, controller_id, git_tree = _controller_authority(
        tmp_path, git_commit=git_commit
    )
    managed_input_hashes = {
        **INPUT_HASHES,
        "controller_launcher_receipt": controller.payload[
            "launcher_receipt_sha256"
        ],
        "controller_receipt": controller.receipt_sha256,
        "controller_anchor_heartbeat": heartbeat.sha256,
        "gpu1_lease": lease.sha256,
    }
    with create_managed_attempt(
        stage_root=stage_root,
        controller_id=controller_id,
        task_id=t4.TASK_ID,
        git_commit=git_commit,
        config_hash=config_hash,
        input_hashes=managed_input_hashes,
        attempt_id="00000000-0000-4000-8000-000000000010",
        boot_id="test-boot",
    ) as attempt:
        with create_worker_staging(
            attempt, staging_id="00000000-0000-4000-8000-000000000011"
        ) as staging:
            worker_activation = main_v2.create_gpu_lease_activation(
                controller_receipt_path=controller.receipt_path,
                lease_path=lease.path,
                expected_lease_sha256=lease.sha256,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                phase="WORKER_ACTIVE",
            )
            worker_heartbeat = main_v2.write_heartbeat_generation(
                controller_receipt_path=controller.receipt_path,
                sequence=2,
                previous_heartbeat_sha256=heartbeat.sha256,
                gpu_inventory=GPU_INVENTORY,
            )
            t4.build_t4_candidate(
                t3_root=t3_root,
                graph_cache_root=graph_cache,
                artifact_root=staging.artifact_root,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                gpu_uuid=GPU_UUID,
                controller_launcher_receipt_path=controller.payload[
                    "launcher_receipt_path"
                ],
                controller_receipt_path=controller.receipt_path,
                controller_anchor_heartbeat_path=heartbeat.path,
                expected_controller_id=controller_id,
                expected_git_commit=git_commit,
                expected_git_tree=git_tree,
                expected_controller_launcher_receipt_sha256=controller.payload[
                    "launcher_receipt_sha256"
                ],
                expected_controller_receipt_sha256=controller.receipt_sha256,
                expected_controller_anchor_heartbeat_sha256=heartbeat.sha256,
                expected_gpu_lease_uuid=lease.lease_uuid,
                expected_gpu_lease_sha256=lease.sha256,
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
        waiting_activation = main_v2.create_gpu_lease_activation(
            controller_receipt_path=controller.receipt_path,
            lease_path=lease.path,
            expected_lease_sha256=lease.sha256,
            attempt_id=sealed.attempt_id,
            generation_token=sealed.generation_token,
            activation_sequence=2,
            previous_activation_sha256=worker_activation.sha256,
            phase="WAITING_VERIFIER",
        )
        verifier_activation = main_v2.create_gpu_lease_activation(
            controller_receipt_path=controller.receipt_path,
            lease_path=lease.path,
            expected_lease_sha256=lease.sha256,
            attempt_id=sealed.attempt_id,
            generation_token=sealed.generation_token,
            activation_sequence=3,
            previous_activation_sha256=waiting_activation.sha256,
            phase="VERIFIER_ACTIVE",
        )
        verifier_heartbeat = main_v2.write_heartbeat_generation(
            controller_receipt_path=controller.receipt_path,
            sequence=3,
            previous_heartbeat_sha256=worker_heartbeat.sha256,
            gpu_inventory=GPU_INVENTORY,
        )
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
            expected_git_tree=git_tree,
            expected_config_hash=config_hash,
            controller_launcher_receipt_path=controller.payload[
                "launcher_receipt_path"
            ],
            controller_receipt_path=controller.receipt_path,
            controller_anchor_heartbeat_path=heartbeat.path,
            expected_controller_launcher_receipt_sha256=controller.payload[
                "launcher_receipt_sha256"
            ],
            expected_controller_receipt_sha256=controller.receipt_sha256,
            expected_controller_anchor_heartbeat_sha256=heartbeat.sha256,
            expected_gpu_lease_uuid=lease.lease_uuid,
            expected_gpu_lease_sha256=lease.sha256,
            science_runner=_runner,
        )
    assert publication.final_path == final
    assert verification["marker"] == t4.PASS_MARKER
    assert verification["physical_gpu_index"] == 1
    assert verification["valid_deletion_count"] == 128
    assert verification["strict_flip_count"] == 20
    assert verification["distinct_flipped_parent_count"] == 10
    assert verification["destination_diversity_status"] == (
        "DESTINATION_DIVERSITY_PASS"
    )
    assert (final / "artifacts/destination_distribution.csv").is_file()
    assert verification["controller_anchor_heartbeat_sha256"] == heartbeat.sha256
    assert verification["worker_initial_heartbeat_sha256"] == worker_heartbeat.sha256
    assert verification["worker_initial_heartbeat_sha256"] != heartbeat.sha256
    assert verification["verifier_heartbeat_sha256"] == verifier_heartbeat.sha256
    assert load_verified_gate(final)["status"] == "PASS"
    assert (final / "PASS").read_text(encoding="utf-8") == "[MANAGED_EXECUTION_V2_PASS]\n"
    assert not (final / "artifacts/gate.json").exists()


def test_independent_replay_rejects_science_drift(tmp_path: Path, monkeypatch) -> None:
    documents = _documents()
    t4._validate_documents(documents)
    changed = copy.deepcopy(documents)
    changed["oracle_smoke.json"]["batch_single_max_abs_difference"] = 1e-3
    assert not t4._equivalent(documents, changed)


def test_aggregate_accepts_measured_cuda_tail_within_frozen_tolerance() -> None:
    documents = _documents()
    documents["oracle_smoke.json"]["batch_single_max_abs_difference"] = 5e-7
    t4._validate_documents(documents)
