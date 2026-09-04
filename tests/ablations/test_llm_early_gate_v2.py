from __future__ import annotations

from dataclasses import asdict

from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.early_launch_gate import (
    EarlyLaunchSnapshot,
    EarlyRunAuthorizationReceipt,
    evaluate_early_launch_gate,
    main_priority_runtime_action,
    snapshot_sha256,
)


def _snapshot(**overrides) -> EarlyLaunchSnapshot:
    values = {
        "matrix_complete_cells": 13,
        "matrix_authority_path": "/runtime/fast16/matrix_authority.json",
        "matrix_authority_sha256": "a" * 64,
        "t8_t13_state": "RUNNING",
        "t8_t13_science_pid": 123,
        "t12_healthy": True,
        "t14_healthy": True,
        "mut_passed_or_gpu_released": True,
        "main_ready_waiting_gpu": (),
        "main_publishers_waiting_gpu": (),
        "idle_gpu": 0,
        "idle_gpu_seconds": 1200,
        "persistent_free_gb": 500.0,
        "minimum_persistent_free_gb": 100.0,
        "memory_available_gb": 128.0,
        "minimum_memory_available_gb": 64.0,
        "checkpoint_resume_supported": True,
        "requested_early_gpus": 1,
        "main_owner_registry_path": "/runtime/final16-owner-registry.json",
        "main_owner_registry_sha256": "d" * 64,
        "main_owner_registry_self_sha256": "e" * 64,
        "all_incomplete_main_cells_owned": True,
        "unhealthy_or_unowned_main_cells": (),
        "missing_main_publisher_cells": (),
        "active_early_llm_ablation_gpus": (),
    }
    values.update(overrides)
    return EarlyLaunchSnapshot(**values)


def _receipt(snapshot: EarlyLaunchSnapshot) -> EarlyRunAuthorizationReceipt:
    body = {
        "authorization_id": "user-early-llm-1",
        "authorized_by": "user_project_owner",
        "matrix_authority_sha256": "a" * 64,
        "snapshot_sha256": snapshot_sha256(snapshot),
        "run_contract_sha256": "b" * 64,
        "execution_commit": "c" * 40,
        "allow_early_llm_ablation": True,
        "max_gpus": 1,
        "schema_version": "early_llm_ablation_authorization_receipt_v1",
    }
    return EarlyRunAuthorizationReceipt(
        **body, authorization_sha256=canonical_json_sha256(body)
    )


def _evaluate(snapshot: EarlyLaunchSnapshot, receipt=None, **overrides):
    values = {
        "matrix_authority_sha256": "a" * 64,
        "run_contract_sha256": "b" * 64,
        "execution_commit": "c" * 40,
        "runtime_evidence_ready": True,
        "science_entrypoint_available": True,
    }
    values.update(overrides)
    return evaluate_early_launch_gate(snapshot, receipt=receipt, **values)


def test_framework_allowed_at12_but_gpu_science_blocked() -> None:
    decision = _evaluate(_snapshot(matrix_complete_cells=12), receipt=None)
    assert decision.science_launch_allowed is False
    assert "MATRIX_BELOW_13" in decision.blockers


def test_early_start_requires_receipt_and_then_allows_exactly_one_gpu() -> None:
    snapshot = _snapshot()
    preflight = _evaluate(snapshot, receipt=None)
    assert preflight.eligible_for_authorization_receipt is True
    assert preflight.science_launch_allowed is False
    decision = _evaluate(snapshot, receipt=_receipt(snapshot))
    assert decision.science_launch_allowed is True
    assert decision.assigned_gpu == 0


def test_main_ready_gpu_or_short_idle_blocks_ablation() -> None:
    for snapshot in (
        _snapshot(main_ready_waiting_gpu=("Mut publisher",)),
        _snapshot(idle_gpu_seconds=1199),
        _snapshot(requested_early_gpus=2),
        _snapshot(active_early_llm_ablation_gpus=(2,)),
        _snapshot(
            all_incomplete_main_cells_owned=False,
            unhealthy_or_unowned_main_cells=("TasteMolNet/GCFExplainer",),
        ),
        _snapshot(missing_main_publisher_cells=("TasteMolNet/ComRecGC",)),
    ):
        decision = _evaluate(snapshot, receipt=None)
        assert decision.science_launch_allowed is False
        assert decision.eligible_for_authorization_receipt is False


def test_running_ablation_yields_to_main_only_at_safe_checkpoint() -> None:
    snapshot = _snapshot(main_ready_waiting_gpu=("Taste publisher",))
    assert main_priority_runtime_action(
        snapshot, ablation_running=True, at_safe_checkpoint=False
    ) == "REQUEST_CHECKPOINT_THEN_PAUSE"
    assert main_priority_runtime_action(
        snapshot, ablation_running=True, at_safe_checkpoint=True
    ) == "GRACEFUL_PAUSE_AND_RELEASE_GPU"


def test_running_ablation_yields_when_a_main_owner_becomes_unhealthy() -> None:
    snapshot = _snapshot(
        all_incomplete_main_cells_owned=False,
        unhealthy_or_unowned_main_cells=("TasteMolNet/GlobalGCE",),
    )
    assert main_priority_runtime_action(
        snapshot, ablation_running=True, at_safe_checkpoint=True
    ) == "GRACEFUL_PAUSE_AND_RELEASE_GPU"


def test_early_receipt_binds_authority_contract_and_execution_commit() -> None:
    snapshot = _snapshot()
    receipt = _receipt(snapshot)
    cases = (
        (
            {"matrix_authority_sha256": "d" * 64},
            "EARLY_RECEIPT_MATRIX_AUTHORITY_CHANGED",
        ),
        ({"run_contract_sha256": "d" * 64}, "EARLY_RECEIPT_RUN_CONTRACT_CHANGED"),
        ({"execution_commit": "d" * 40}, "EARLY_RECEIPT_EXECUTION_COMMIT_CHANGED"),
    )
    for changed, blocker in cases:
        decision = _evaluate(snapshot, receipt=receipt, **changed)
        assert decision.science_launch_allowed is False
        assert blocker in decision.blockers


def test_config_only_or_missing_runtime_evidence_blocks_science() -> None:
    snapshot = _snapshot()
    receipt = _receipt(snapshot)
    missing = _evaluate(snapshot, receipt=receipt, runtime_evidence_ready=False)
    assert "LLM_RUNTIME_EVIDENCE_NOT_READY" in missing.blockers
    config_only = _evaluate(
        snapshot, receipt=receipt, science_entrypoint_available=False
    )
    assert "LLM_SCIENCE_ENTRYPOINT_CONFIG_ONLY" in config_only.blockers
