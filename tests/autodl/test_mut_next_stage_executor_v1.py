from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.utils.autodl_mut_first_divergence_v1 import file_sha256, stable_sha256
from src.utils.autodl_mut_next_stage_executor_v1 import (
    ADOPTION_STAGES,
    MutNextStageError,
    build_successor_spec,
    consume_next_action_once,
    validate_next_action,
    validate_successor_spec,
)
from src.utils.autodl_mut_post_ab_continuation_v1 import (
    EXECUTION_COMMIT,
    SOURCE_COMMIT,
    UPSTREAM_COMMIT,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate_tree(tmp_path: Path, *, divergent: bool = False) -> tuple[Path, Path, dict[str, object]]:
    manifest: dict[str, object] = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "dataset": "mutagenicity",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "formal_M_MAX": 50_000,
        "comparison_steps": 500,
        "post_reload_steps": 10,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": 1448,
        "batch_size": 128,
        "device": "cuda:0",
        "arms_sequential": True,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "pythonhashseed": "0",
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    manifest_path = _write(tmp_path / "input/manifest.json", manifest)
    exact_fields = {
        key: not divergent
        for key in (
            "trace_on_off_stepwise_exact",
            "step_action_trace_exact",
            "rng_state_exact",
            "classifier_probability_trace_exact",
            "step_semantic_fields_present",
            "trace_on_checkpoint_reload_pass",
            "trace_off_checkpoint_reload_pass",
            "post_reload_trace_mode_equivalence_pass",
            "step500_checkpoint_serialized_candidate_records_exact",
            "step500_checkpoint_candidate_universe_exact",
            "checkpoint_algorithm_scientific_state_exact",
            "checkpoint_rng_state_exact",
            "checkpoint_sqlite_logical_state_exact",
            "checkpoint_graph_registry_exact",
            "resolved_config_scientific_binding_exact",
        )
    }
    # A causal divergence still requires complete semantic fields and valid
    # observer logs; only the stepwise comparison itself is false.
    if divergent:
        exact_fields["step_semantic_fields_present"] = True
    gate: dict[str, object] = {
        "schema_version": "mut_trace_on_off_500_step_equivalence_v1",
        "status": "FAIL" if divergent else "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "formal_M_MAX": 50_000,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "post_walk_prefix_finalization_performed": False,
        "full_50k_trace_on_off_parity_claimed": False,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "first_semantic_divergence_step": 17 if divergent else None,
        "failures": ["step_17"] if divergent else [],
        "trace_on_observer_log_audit": {"status": "PASS"},
        "trace_off_observer_log_audit": {"status": "PASS"},
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": file_sha256(manifest_path),
        **exact_fields,
    }
    gate["summary_sha256"] = stable_sha256(gate)
    gate_path = _write(tmp_path / "gate/gate.json", gate)
    terminal = {
        "schema_version": "mut_same_contract_ab_owner_terminal_v1",
        "task_id": "mut-ab",
        "status": (
            "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED"
            if divergent
            else "PASS_TRACE_MODE_EQUIVALENCE"
        ),
        "fresh_50k_started": False,
        "equivalence_gate": str(gate_path),
        "equivalence_gate_sha256": file_sha256(gate_path),
    }
    terminal_path = _write(tmp_path / "terminal/terminal.json", terminal)
    decision: dict[str, object] = {
        "schema_version": "mut_post_same_contract_ab_decision_v1",
        "status": "READY",
        "branch": (
            "ROUTE_B_AUTHORIZATION_REQUIRED"
            if divergent
            else "HISTORICAL_ADOPTION_GATES_REQUIRED"
        ),
        "classification": "SCIENTIFIC_STATE_DIVERGENCE" if divergent else "TRACE_ALIAS_ONLY",
        "same_contract_gate": str(gate_path),
        "same_contract_gate_sha256": file_sha256(gate_path),
        "same_contract_gate_summary_sha256": gate["summary_sha256"],
        "historical_adoption_gate_eligible": not divergent,
        "historical_adoption_permitted": False,
        "route_b_evidence_eligible": divergent,
        "route_b_permitted": False,
        "requires_immutable_deployed_consumer": True,
        "fresh_50k_started": False,
    }
    decision["decision_sha256"] = stable_sha256(decision)
    return gate_path, terminal_path, decision


def _stage(tmp_path: Path, name: str) -> dict[str, object]:
    return {
        "stage": name,
        "argv": [str(Path(sys.executable).resolve()), "-c", "raise SystemExit(0)"],
        "cwd": str(tmp_path),
        "environment": {"PYTHONHASHSEED": "0"},
        "expected_terminal": str(tmp_path / f"terminal-{name}.json"),
        "expected_terminal_status": ["PASS"],
        "output_root": str(tmp_path / f"output-{name}"),
        "pair_store_recomputed": False,
        "dbscan_recomputed": False,
        "calibration_used_for_selection": False,
        "test_used_for_selection": False,
    }


def test_successor_spec_predeploys_full_adoption_and_route_b(tmp_path: Path) -> None:
    predecessor = _write(tmp_path / "predecessor/spec.json", {"immutable": True})
    authority = tmp_path / "authority"
    authority.mkdir()
    value = build_successor_spec(
        {
            "task_id": "mut-next",
            "execution_commit": "a" * 40,
            "predecessor_task_id": "mut-ab",
            "predecessor_task_spec": str(predecessor),
            "predecessor_terminal": str(tmp_path / "terminal/terminal.json"),
            "next_action_path": str(tmp_path / "action/next_action.json"),
            "runtime_root": str(tmp_path / "runtime"),
            "lease_path": str(tmp_path / "lease"),
            "publisher_id": "mut-publisher",
            "publisher_locator": str(tmp_path / "publisher.locator.json"),
            "matrix_authority_root": str(authority),
            "adoption_pipeline": [_stage(tmp_path, name) for name in ADOPTION_STAGES],
            "route_b_pipeline": [_stage(tmp_path, "ROUTE_B")],
        },
        check_files=False,
    )
    assert value["route_b_M_MAX"] == 50_000
    assert value["route_b_M_MIN"] == 20_000
    assert validate_successor_spec(value, check_files=False) == value


def test_next_action_consumed_once(tmp_path: Path) -> None:
    _gate, terminal, decision = _gate_tree(tmp_path)
    action = _write(tmp_path / "action/next_action.json", decision)
    lane, _value, consumed, receipt = consume_next_action_once(
        action_path=action,
        predecessor_terminal=terminal,
        task_spec_sha256="a" * 64,
        expected_task_id="mut-ab",
    )
    assert lane == "ADOPTION"
    assert not action.exists()
    assert consumed.is_file()
    assert receipt["status"] == "CONSUMED_ONCE"
    with pytest.raises(MutNextStageError, match="absent"):
        consume_next_action_once(
            action_path=action,
            predecessor_terminal=terminal,
            task_spec_sha256="a" * 64,
            expected_task_id="mut-ab",
        )


def test_only_scientific_divergence_selects_route_b(tmp_path: Path) -> None:
    _gate, terminal, decision = _gate_tree(tmp_path, divergent=True)
    lane, _ = validate_next_action(
        decision, predecessor_terminal=terminal, expected_task_id="mut-ab"
    )
    assert lane == "ROUTE_B"
    decision["branch"] = "ENGINEERING_REPAIR_REQUIRED"
    decision["classification"] = "COMPARATOR_BUG"
    decision["route_b_evidence_eligible"] = False
    decision["decision_sha256"] = stable_sha256(
        {key: item for key, item in decision.items() if key != "decision_sha256"}
    )
    assert validate_next_action(
        decision, predecessor_terminal=terminal, expected_task_id="mut-ab"
    )[0] == "ENGINEERING_REPAIR"


def test_adoption_stage_cannot_recompute_pair_store(tmp_path: Path) -> None:
    predecessor = _write(tmp_path / "predecessor/spec.json", {"immutable": True})
    authority = tmp_path / "authority"
    authority.mkdir()
    stages = [_stage(tmp_path, name) for name in ADOPTION_STAGES]
    stages[0]["pair_store_recomputed"] = True
    with pytest.raises(MutNextStageError, match="no-leakage"):
        build_successor_spec(
            {
                "task_id": "mut-next",
                "execution_commit": "a" * 40,
                "predecessor_task_id": "mut-ab",
                "predecessor_task_spec": str(predecessor),
                "predecessor_terminal": str(tmp_path / "terminal/terminal.json"),
                "next_action_path": str(tmp_path / "action/next_action.json"),
                "runtime_root": str(tmp_path / "runtime"),
                "lease_path": str(tmp_path / "lease"),
                "publisher_id": "mut-publisher",
                "publisher_locator": str(tmp_path / "publisher.locator.json"),
                "matrix_authority_root": str(authority),
                "adoption_pipeline": stages,
                "route_b_pipeline": [_stage(tmp_path, "ROUTE_B")],
            },
            check_files=False,
        )
