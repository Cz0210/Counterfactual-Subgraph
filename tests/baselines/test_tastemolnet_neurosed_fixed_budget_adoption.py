from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import pytest

import src.eval.tastemolnet_neurosed_fixed_budget_adoption as adoption
from src.data.tastemolnet_neurosed_production import stable_sha256
from src.eval.tastemolnet_neurosed_official_fixed_budget import (
    DISTANCE_DIRECTION_SCHEMA,
    OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
    READINESS_SCHEMA,
)
from src.train.tastemolnet_neurosed_fixed_budget import NEUROSED_PASS_MARKER
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.terminal_publisher_v2 import (
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _fixture(root: Path) -> dict[str, str]:
    root.mkdir()
    selected = b"fixed-budget-selected-state"
    selected_sha = hashlib.sha256(selected).hexdigest()
    hashes = {
        "train_pair_sampler_manifest_sha256": "1" * 64,
        "validation_pair_sampler_manifest_sha256": "2" * 64,
        "train_pair_labels_manifest_sha256": "3" * 64,
        "validation_pair_labels_manifest_sha256": "4" * 64,
    }
    for name in ("best.pt", "model.pt"):
        (root / name).write_bytes(selected)
    checkpoint_id = str(uuid.uuid4())
    checkpoint_dir = root / "checkpoints" / checkpoint_id
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(selected)
    feature = {
        "schema_version": adoption.FIXED_BUDGET_FEATURE_SCHEMA,
        "dataset": "tastemolnet",
        "train_derived_only": True,
        "validation_unseen_atomic_numbers": [],
        "feature_atomic_numbers": [1, 6],
        "input_dim": 2,
    }
    _write_json(root / "feature_schema.json", feature)
    feature_sha = hashlib.sha256((root / "feature_schema.json").read_bytes()).hexdigest()
    selector_sha = "5" * 64
    direction_sha = "6" * 64
    model_card = {
        "schema_version": OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
        "selected_checkpoint_sha256": selected_sha,
        "feature_schema_sha256": feature_sha,
        "selector_trace_sha256": selector_sha,
        "distance_direction_trace_sha256": direction_sha,
        **hashes,
    }
    _write_json(root / "model_card.json", model_card)
    _write_json(
        root / "pair_manifest.json",
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_pair_bundle_v1",
            "train_pair_count": 5000,
            "validation_pair_count": 1000,
            "independent_pairs": True,
            "query_graph_id_differs_from_target_graph_id": True,
            "class_labels_used_as_supervision": False,
            "calibration_loaded": False,
            "test_loaded": False,
            **hashes,
        },
    )
    split_sha = "7" * 64
    _write_json(
        root / "split_manifest.json",
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_split_manifest_v1",
            "opened_payload_splits": ["train", "validation"],
            "train_pair_roles_subset_of_train": True,
            "validation_pair_roles_subset_of_validation": True,
            "train_validation_graph_id_intersection_empty": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "source_split_isolation_sha256": split_sha,
        },
    )
    _write_json(
        root / "checkpoint_manifest.json",
        {
            "selected_checkpoint_sha256": selected_sha,
            "best_pt_sha256": selected_sha,
            "model_pt_sha256": selected_sha,
            "best_and_model_bytes_identical": True,
        },
    )
    _write_json(
        root / "health_gate.json",
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_worker_health_v1",
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "worker_wrote_scientific_pass": False,
            "finite_loss": True,
            "finite_validation_metric": True,
            "no_split_leakage": True,
            "official_selector_trace": True,
            "generated_query_to_original_target_assertion": True,
            "gcf_runner_load_passed": True,
        },
    )
    _write_json(
        root / "readiness.json",
        {
            "schema_version": READINESS_SCHEMA,
            "status": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION",
            "marker": None,
            "scientific_pass_claimed": False,
            "model_card_contract_valid": True,
            "official_selector_contract_valid": True,
            "generated_query_original_target_direction_valid": True,
            "evidence_bindings": hashes,
        },
    )
    _write_json(
        root / "selector_trace.json",
        {"selected_checkpoint_sha256": selected_sha, "trace_sha256": selector_sha},
    )
    _write_json(
        root / "distance_direction_trace.json",
        {
            "schema_version": DISTANCE_DIRECTION_SCHEMA,
            "direction": "generated_query_to_original_target",
            "reverse_direction_used": False,
            "trace_sha256": direction_sha,
        },
    )
    for name in (
        "config.yaml",
        "environment.json",
        "ged_label_manifest.json",
        "git_state.json",
        "training_metrics.json",
        "validation_metrics.json",
    ):
        _write_json(root / name, {})
    scientific = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "sha256sums.txt").write_text("\n".join(scientific) + "\n")
    verification = {
        "schema_version": adoption.FIXED_BUDGET_VERIFICATION_SCHEMA,
        "status": "PASS",
        "marker": NEUROSED_PASS_MARKER,
        "independent_process_reopened_worker_root": True,
        "worker_wrote_scientific_pass": False,
        "checkpoint_reload_passed": True,
        "official_selector_trace_replayed": True,
        "batch_single_agreement_reproduced": True,
        "gcf_runner_load_reproduced": True,
        "generated_query_to_original_target_reproduced": True,
        "validation_metrics_reproduced": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "checkpoint_sha256": selected_sha,
        "selector_trace_sha256": selector_sha,
        "distance_direction_trace_sha256": direction_sha,
    }
    verification["verification_sha256"] = stable_sha256(verification)
    _write_json(root / "verification.json", verification)
    verification_file_sha = hashlib.sha256(
        (root / "verification.json").read_bytes()
    ).hexdigest()
    (root / "verification_sha256s.txt").write_text(
        f"{verification_file_sha}  verification.json\n"
    )
    (root / "PASS").write_text(NEUROSED_PASS_MARKER + "\n")
    return {"checkpoint": selected_sha, "verification": verification_file_sha}


def test_lossless_copy_and_consumer_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _fixture(source)
    monkeypatch.setattr(
        adoption, "validate_official_fixed_budget_model_card", lambda card, **_: card
    )
    inspected = adoption.inspect_fixed_budget_neurosed_pass(
        source, vendored_gcf_root=tmp_path
    )
    destination = tmp_path / "artifacts"
    destination.mkdir()
    copied = adoption.copy_fixed_budget_neurosed_pass(
        source_root=source,
        artifact_root=destination,
        expected_source_inventory_sha256=inspected["inventory_sha256"],
        vendored_gcf_root=tmp_path,
    )
    assert copied["inventory_sha256"] == inspected["inventory_sha256"]
    assert copied["t7_consumer"]["schema_version"] == (
        adoption.T7_FIXED_BUDGET_CONSUMER_SCHEMA
    )
    assert copied["t7_consumer"]["train_pair_sampler_manifest_sha256"] == "1" * 64
    assert not (source / "gate.json").exists()


def test_independent_managed_verifier_and_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    monkeypatch.setattr(
        adoption, "validate_official_fixed_budget_model_card", lambda card, **_: card
    )
    source = tmp_path / "source"
    evidence = _fixture(source)
    inspected = adoption.inspect_fixed_budget_neurosed_pass(
        source, vendored_gcf_root=tmp_path
    )
    inputs = {
        "fixed_budget_neurosed_checkpoint_sha256": evidence["checkpoint"],
        "fixed_budget_neurosed_pass_sha256": inspected["pass_sha256"],
        "fixed_budget_neurosed_source_inventory_sha256": inspected[
            "inventory_sha256"
        ],
        "fixed_budget_neurosed_verification_sha256": evidence["verification"],
    }
    stage = tmp_path / "stage"
    stage.mkdir()
    final = tmp_path / "final-parent" / "neurosed"
    final.parent.mkdir()
    with create_managed_attempt(
        stage_root=stage,
        controller_id="real-controller-fixture",
        task_id=adoption.ADOPTION_TASK_ID,
        git_commit="a" * 40,
        config_hash="b" * 64,
        input_hashes=inputs,
    ) as attempt:
        with create_worker_staging(attempt) as staging:
            adoption.copy_fixed_budget_neurosed_pass(
                source_root=source,
                artifact_root=staging.artifact_root,
                expected_source_inventory_sha256=inspected["inventory_sha256"],
                vendored_gcf_root=tmp_path,
            )
            raw = write_worker_raw_evidence(
                staging,
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "scientific_command": [
                        "python",
                        "adopt_tastemolnet_fixed_budget_neurosed_v2.py",
                        "copy",
                    ],
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
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            exited.close()
            sealed = seal_worker_staging(staging)
        with open_sealed_worker_artifact(
            sealed.seal_path,
            expected_attempt_id=attempt.attempt_id,
            expected_generation_token=sealed.generation_token,
        ) as held:
            verification = adoption.verify_fixed_budget_managed_adoption(
                held,
                source_root=source,
                expected_source_inventory_sha256=inspected["inventory_sha256"],
                vendored_gcf_root=tmp_path,
            )
            publication = verify_and_publish_sealed_attempt(
                held, final_path=final, verification=verification
            )
    assert publication.final_path == final
    assert (final / "PASS").read_bytes() == b"[MANAGED_EXECUTION_V2_PASS]\n"
    assert (final / "artifacts" / "best.pt").read_bytes() == (
        source / "best.pt"
    ).read_bytes()
