from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.utils.autodl_mut_first_divergence_v1 import (
    audit_mut_first_divergence,
    file_sha256,
    stable_sha256,
)
from src.utils.autodl_mut_route_b_v1 import (
    MutRouteBSpecError,
    ROUTE_B_EXECUTION_COMMIT,
    build_route_b_spec,
    route_b_generation_command,
    validate_route_b_evidence,
)
from src.utils.autodl_mut_same_contract_ab_v1 import (
    build_same_contract_ab_spec,
    same_contract_ab_command,
)
from src.utils.autodl_mut_post_ab_continuation_v1 import (
    MutPostABError,
    classify_same_contract_gate,
    select_post_ab_action,
    validate_ab_owner_terminal,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_arm(
    root: Path,
    *,
    project_commit: str,
    resumed: bool,
    trace_enabled: bool,
    action: list[object],
    target: str,
    candidate_count: int,
) -> None:
    trace = root / "trace"
    chunks = trace / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    selected = chunks / "part-000000.jsonl"
    selected.write_text(
        json.dumps(
            {
                "event": "selected_transition",
                "move_index": 0,
                "head_index": 0,
                "parent_id": "MUT_PARENT",
                "source_graph_sha256": "a" * 64,
                "source_official_hash": "unstable-source",
                "action": action,
                "target_graph_sha256": target,
                "target_official_hash": "unstable-target",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        trace / "selected_action_trace_manifest.json",
        {
            "row_count": 1,
            "chunks": [
                {
                    "path": "selected_action_trace_chunks/part-000000.jsonl",
                    "row_count": 1,
                    "sha256": file_sha256(selected),
                }
            ],
        },
    )
    lineage = trace / "candidate_action_lineage_index.jsonl"
    with lineage.open("w", encoding="utf-8") as stream:
        for index in range(candidate_count):
            stream.write(
                json.dumps(
                    {
                        "candidate_index": index,
                        "parent_id": f"MUT_{index % 1448:04d}",
                        "stable_graph_sha256": f"{index:064x}",
                        "action_count": 1,
                        "lineage_root_status": "frozen_source_graph_exact",
                        "lineage_storage": "selected_trace_predecessor_index",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = {
        "schema_version": 1,
        "dataset": "mutagenicity",
        "dataset_audit": {
            "dataset_fingerprint": "d" * 64,
            "generation_parent_ids_sha256": "c" * 64,
        },
        "generation_parent_ids_sha256": "c" * 64,
        "gnn": {"checkpoint_sha256": "g" * 64},
        "distance_model": {"checkpoint_sha256": "e" * 64},
        "config_sha256": "1" * 64 if not resumed else "2" * 64,
        "project_commit": project_commit,
        "upstream_commit": "u" * 40,
        "runtime_environment": {"pythonhashseed": "0"},
        "parameters": {
            "steps": 500,
            "seed": 0,
            "candidate_capacity": 100_000,
        },
        "resumed_from_checkpoint": "/checkpoint/250" if resumed else None,
        "trace_enabled": trace_enabled,
        "trace_summary": {
            "graph_identity_mode": "official_untyped_node_adjacency_v1",
            "frozen_payload_closure": {
                "graph_serialization_version": "codec-v7" if resumed else "codec-v6"
            },
        },
        "counterfactual_candidate_count": candidate_count,
        "test_loaded": False,
        "calibration_loaded": False,
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_json(
        root / "semantic_lineage_finalizer_receipt.json",
        {
            "source_algorithm_commit": "s" * 40,
            "semantic_transition_count": 1,
            "semantic_transition_alias_event_count": 0,
        },
    )


def _task_spec(path: Path) -> None:
    _write_json(
        path,
        {
            "science_contract": {"rf_oracle_sha256": "f" * 64},
            "input_hashes": {"mut_rf_oracle": "f" * 64},
        },
    )


def _dataset_summary(path: Path) -> None:
    _write_json(
        path,
        {"train_ids_hash": "t" * 64, "val_ids_hash": "v" * 64},
    )


def test_real_2250_vs_2255_cannot_be_output_order_only(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    instrumented = tmp_path / "instrumented"
    _write_arm(
        legacy,
        project_commit="1" * 40,
        resumed=False,
        trace_enabled=True,
        action=["EA", 0, 24],
        target="4" * 64,
        candidate_count=2250,
    )
    _write_arm(
        instrumented,
        project_commit="2" * 40,
        resumed=True,
        trace_enabled=True,
        action=["EA", 2, 22],
        target="5" * 64,
        candidate_count=2255,
    )
    task_spec = tmp_path / "task.json"
    dataset_summary = tmp_path / "dataset.json"
    _task_spec(task_spec)
    _dataset_summary(dataset_summary)
    result = audit_mut_first_divergence(
        legacy_root=legacy,
        instrumented_root=instrumented,
        task_spec_path=task_spec,
        dataset_summary_path=dataset_summary,
        output_dir=tmp_path / "audit",
    )
    assert result["classification"] == "SCIENTIFIC_STATE_DIVERGENCE"
    assert result["2250_vs_2255_explicitly_accounted_for"] is True
    assert result["output_order_only_permitted"] is False
    assert result["candidate_universe"]["delta_instrumented_minus_legacy"] == 5
    assert result["first_transition_divergence"]["generation_step"] == 1
    assert result["first_transition_divergence"]["differences"]["action"] == {
        "legacy": ["EA", 0, 24],
        "instrumented": ["EA", 2, 22],
    }


def test_mixed_commit_resume_and_two_trace_on_arms_do_not_authorize_route_b(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    instrumented = tmp_path / "instrumented"
    _write_arm(
        legacy,
        project_commit="1" * 40,
        resumed=False,
        trace_enabled=True,
        action=["EA", 0, 24],
        target="4" * 64,
        candidate_count=1,
    )
    _write_arm(
        instrumented,
        project_commit="2" * 40,
        resumed=True,
        trace_enabled=True,
        action=["EA", 2, 22],
        target="5" * 64,
        candidate_count=2,
    )
    result = audit_mut_first_divergence(
        legacy_root=legacy,
        instrumented_root=instrumented,
        output_dir=tmp_path / "audit",
    )
    assert result["current_pair_trace_mode_causal_claim_permitted"] is False
    assert result["route_b_gate"]["route_b_admissible"] is False
    assert result["route_b_gate"]["next_required_action"] == (
        "RUN_FRESH_SAME_COMMIT_SEQUENTIAL_TRACE_ON_OFF_A_B"
    )
    with pytest.raises(MutRouteBSpecError, match="same-contract"):
        validate_route_b_evidence(result)


def test_same_scientific_contract_allows_only_the_trace_mode_to_differ(
    tmp_path: Path,
) -> None:
    trace_on = tmp_path / "trace-on"
    trace_off = tmp_path / "trace-off"
    for root, enabled in ((trace_on, True), (trace_off, False)):
        _write_arm(
            root,
            project_commit="1" * 40,
            resumed=False,
            trace_enabled=enabled,
            action=["EA", 0, 24],
            target="4" * 64,
            candidate_count=1,
        )
    task_spec = tmp_path / "task.json"
    dataset_summary = tmp_path / "dataset.json"
    _task_spec(task_spec)
    _dataset_summary(dataset_summary)
    result = audit_mut_first_divergence(
        legacy_root=trace_on,
        instrumented_root=trace_off,
        task_spec_path=task_spec,
        dataset_summary_path=dataset_summary,
        output_dir=tmp_path / "audit",
    )
    comparison = json.loads(
        Path(result["contract_comparison_path"]).read_text(encoding="utf-8")
    )
    assert comparison["same_scientific_contract_except_trace_mode"] is True
    assert comparison["trace_pair_is_on_off"] is True
    assert comparison["eligible_as_trace_mode_equivalence_evidence"] is True


def _eligible_evidence(path: Path) -> dict[str, object]:
    gate_path, gate = _same_contract_gate(path.parent, status="FAIL", first=17)
    spec = path.parent / "ab-spec.json"
    _write_json(spec, {"sealed": True})
    value = select_post_ab_action(
        terminal={
            "status": "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED",
            "fresh_50k_started": False,
        },
        gate=gate,
        ab_spec_path=spec,
        gate_path=gate_path,
    )
    _write_json(path, value)
    return value


def _authorization(
    path: Path,
    *,
    evidence: Path,
    commit: str,
    task_id: str = "mut-route-b-test",
    attempt_uuid: str,
    output_root: str,
) -> None:
    value: dict[str, object] = {
        "schema_version": "mut_traceoff_route_b_launch_authorization_v1",
        "allow_fresh_traceoff_50k": True,
        "evidence_file_sha256": file_sha256(evidence),
        "execution_commit": commit,
        "task_id": task_id,
        "attempt_uuid": attempt_uuid,
        "output_root": output_root,
        "gpu_index": 0,
        "gpu_uuid": "GPU-test-0000",
    }
    value["receipt_sha256"] = stable_sha256(value)
    _write_json(path, value)


def test_route_b_spec_is_fixed_traceoff_50k_and_never_reuses_pair_store(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    authorization = tmp_path / "authorization.json"
    commit = ROUTE_B_EXECUTION_COMMIT
    attempt_uuid = str(uuid4())
    output_root = str(tmp_path / "generation")
    _eligible_evidence(evidence)
    _authorization(
        authorization,
        evidence=evidence,
        commit=commit,
        attempt_uuid=attempt_uuid,
        output_root=output_root,
    )
    spec = build_route_b_spec(
        {
            "task_id": "mut-route-b-test",
            "attempt_uuid": attempt_uuid,
            "execution_commit": commit,
            "repo_root": "/immutable/repo",
            "python": "/env/bin/python",
            "config_path": "/immutable/repo/configs/hpc.yaml",
            "upstream_root": "/immutable/upstream",
            "dataset_dir": "/immutable/dataset",
            "gnn_checkpoint": "/immutable/gnn.pt",
            "distance_checkpoint": "/immutable/distance.pt",
            "evidence_path": str(evidence),
            "launch_authorization_path": str(authorization),
            "output_root": output_root,
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "checkpoint_mirror_root": str(tmp_path / "mirror"),
            "lease_path": str(tmp_path / "lease.lock"),
            "gpu_lock_root": str(tmp_path / "gpu-locks"),
            "gpu_uuid": "GPU-test-0000",
            "owner_runtime_root": str(tmp_path / "owner"),
            "gpu_index": 0,
        },
        check_files=False,
    )
    assert spec["contract"]["M_MAX"] == 50_000
    assert spec["contract"]["candidate_capacity"] == 100_000
    assert spec["contract"]["trace_enabled"] is False
    assert spec["contract"]["pair_store_reuse_allowed"] is False
    assert spec["contract"]["dbscan_reuse_allowed"] is False
    assert spec["contract"]["convergence_early_stop_allowed"] is False
    command = route_b_generation_command(spec)
    assert command[command.index("--mode") + 1] == "full"
    assert "--trace-output-dir" not in command
    assert command[command.index("--checkpoint-interval-steps") + 1] == "500"


def test_route_b_requires_separate_bound_launch_authorization(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    authorization = tmp_path / "authorization.json"
    commit = ROUTE_B_EXECUTION_COMMIT
    attempt_uuid = str(uuid4())
    output_root = str(tmp_path / "generation")
    _eligible_evidence(evidence)
    _authorization(
        authorization,
        evidence=evidence,
        commit="b" * 40,
        attempt_uuid=attempt_uuid,
        output_root=output_root,
    )
    with pytest.raises(MutRouteBSpecError, match="different code"):
        build_route_b_spec(
            {
                "task_id": "mut-route-b-test",
                "attempt_uuid": attempt_uuid,
                "execution_commit": commit,
                "repo_root": "/immutable/repo",
                "python": "/env/bin/python",
                "config_path": "/immutable/repo/configs/hpc.yaml",
                "upstream_root": "/immutable/upstream",
                "dataset_dir": "/immutable/dataset",
                "gnn_checkpoint": "/immutable/gnn.pt",
                "distance_checkpoint": "/immutable/distance.pt",
                "evidence_path": str(evidence),
                "launch_authorization_path": str(authorization),
                "output_root": output_root,
                "checkpoint_root": str(tmp_path / "checkpoints"),
                "checkpoint_mirror_root": str(tmp_path / "mirror"),
                "lease_path": str(tmp_path / "lease.lock"),
                "gpu_lock_root": str(tmp_path / "gpu-locks"),
                "gpu_uuid": "GPU-test-0000",
                "owner_runtime_root": str(tmp_path / "owner"),
                "gpu_index": 0,
            },
            check_files=False,
        )


def test_route_b_authorization_cannot_be_replayed_for_another_output(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    authorization = tmp_path / "authorization.json"
    commit = ROUTE_B_EXECUTION_COMMIT
    attempt_uuid = str(uuid4())
    authorized_output = str(tmp_path / "authorized-generation")
    _eligible_evidence(evidence)
    _authorization(
        authorization,
        evidence=evidence,
        commit=commit,
        attempt_uuid=attempt_uuid,
        output_root=authorized_output,
    )
    with pytest.raises(MutRouteBSpecError, match="authorization target changed"):
        build_route_b_spec(
            {
                "task_id": "mut-route-b-test",
                "attempt_uuid": attempt_uuid,
                "execution_commit": commit,
                "repo_root": "/immutable/repo",
                "python": "/env/bin/python",
                "config_path": "/immutable/repo/configs/hpc.yaml",
                "upstream_root": "/immutable/upstream",
                "dataset_dir": "/immutable/dataset",
                "gnn_checkpoint": "/immutable/gnn.pt",
                "distance_checkpoint": "/immutable/distance.pt",
                "evidence_path": str(evidence),
                "launch_authorization_path": str(authorization),
                "output_root": str(tmp_path / "different-generation"),
                "checkpoint_root": str(tmp_path / "checkpoints"),
                "checkpoint_mirror_root": str(tmp_path / "mirror"),
                "lease_path": str(tmp_path / "lease.lock"),
                "gpu_lock_root": str(tmp_path / "gpu-locks"),
                "gpu_uuid": "GPU-test-0000",
                "owner_runtime_root": str(tmp_path / "owner"),
                "gpu_index": 0,
            },
            check_files=False,
        )


def test_same_contract_ab_is_bounded_sequential_and_never_launches_50k(
    tmp_path: Path,
) -> None:
    spec = build_same_contract_ab_spec(
        {
            "task_id": "mut-ab-test",
            "attempt_uuid": str(uuid4()),
            "controller_project_root": "/controller",
            "controller_commit": "c" * 40,
            "python": "/env/bin/python",
            "runner_path": "/controller/scripts/autodl/run_mut_trace_mode_equivalence.py",
            "legacy_project_root": "/legacy",
            "execution_project_root": "/instrumented",
            "historical_artifact_root": "/historical",
            "upstream_root": "/upstream",
            "dataset_dir": "/dataset",
            "gnn_checkpoint": "/models/gnn.pt",
            "distance_checkpoint": "/models/distance.pt",
            "rf_oracle": "/models/rf.pkl",
            "run_root": str(tmp_path / "run"),
            "output_dir": str(tmp_path / "output"),
            "control_root": str(tmp_path / "control"),
            "lease_path": str(tmp_path / "lease.lock"),
            "gpu_lock_root": str(tmp_path / "gpu-locks"),
            "gpu_uuid": "GPU-test-0000",
            "gpu_index": 0,
        },
        check_files=False,
    )
    assert spec["steps"] == 500
    assert spec["post_reload_steps"] == 10
    assert spec["trace_modes"] == ["on", "off"]
    assert spec["arms_sequential"] is True
    assert spec["resume_parity_separate"] is True
    assert spec["fresh_50k_started"] is False
    command = same_contract_ab_command(spec)
    assert "run-pair" in command
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "50000" not in command


def _same_contract_gate(tmp_path: Path, *, status: str, first: int | None) -> tuple[Path, dict[str, object]]:
    manifest: dict[str, object] = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "dataset": "mutagenicity",
        "source_algorithm_commit": "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4",
        "execution_commit": "66487c062c86d53ef2f762ce04d0fb965af5af08",
        "upstream_commit": "122f9341a360e9f06bb58a2f5823bb596021f6bf",
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
    manifest_path = tmp_path / "input.json"
    _write_json(manifest_path, manifest)
    exact = status == "PASS"
    gate: dict[str, object] = {
        "schema_version": "mut_trace_on_off_500_step_equivalence_v1",
        "status": status,
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "source_algorithm_commit": "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4",
        "execution_commit": "66487c062c86d53ef2f762ce04d0fb965af5af08",
        "formal_M_MAX": 50_000,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "trace_on_off_stepwise_exact": exact,
        "step_action_trace_exact": exact,
        "rng_state_exact": exact,
        "classifier_probability_trace_exact": exact,
        "step_semantic_fields_present": True,
        "first_semantic_divergence_step": first,
        "trace_on_checkpoint_reload_pass": True,
        "trace_off_checkpoint_reload_pass": True,
        "post_reload_trace_mode_equivalence_pass": True,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "step500_checkpoint_serialized_candidate_records_exact": exact,
        "step500_checkpoint_candidate_universe_exact": exact,
        "checkpoint_algorithm_scientific_state_exact": exact,
        "checkpoint_rng_state_exact": exact,
        "checkpoint_sqlite_logical_state_exact": exact,
        "checkpoint_graph_registry_exact": exact,
        "resolved_config_scientific_binding_exact": exact,
        "post_walk_prefix_finalization_performed": False,
        "full_50k_trace_on_off_parity_claimed": False,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": file_sha256(manifest_path),
        "trace_on_observer_log_audit": {"status": "PASS"},
        "trace_off_observer_log_audit": {"status": "PASS"},
        "failures": [] if exact else ["trace_on_off_stepwise_semantics"],
    }
    gate["summary_sha256"] = stable_sha256(gate)
    gate_path = tmp_path / "gate.json"
    _write_json(gate_path, gate)
    return gate_path, gate


def test_post_ab_pass_persists_adoption_without_starting_route_b(tmp_path: Path) -> None:
    gate_path, gate = _same_contract_gate(tmp_path, status="PASS", first=None)
    spec = tmp_path / "spec.json"
    _write_json(spec, {"sealed": True})
    decision = select_post_ab_action(
        terminal={
            "status": "PASS_TRACE_MODE_EQUIVALENCE",
            "fresh_50k_started": False,
        },
        gate=gate,
        ab_spec_path=spec,
        gate_path=gate_path,
    )
    assert decision["branch"] == "HISTORICAL_ADOPTION_GATES_REQUIRED"
    assert decision["historical_adoption_gate_eligible"] is True
    assert decision["historical_adoption_permitted"] is False
    assert decision["route_b_permitted"] is False
    assert decision["fresh_50k_started"] is False


def test_post_ab_true_same_contract_divergence_selects_route_b(tmp_path: Path) -> None:
    gate_path, gate = _same_contract_gate(tmp_path, status="FAIL", first=17)
    spec = tmp_path / "spec.json"
    _write_json(spec, {"sealed": True})
    decision = select_post_ab_action(
        terminal={
            "status": "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED",
            "fresh_50k_started": False,
        },
        gate=gate,
        ab_spec_path=spec,
        gate_path=gate_path,
    )
    assert decision["branch"] == "ROUTE_B_AUTHORIZATION_REQUIRED"
    assert decision["classification"] == "SCIENTIFIC_STATE_DIVERGENCE"
    assert decision["route_b_evidence_eligible"] is True
    assert decision["route_b_permitted"] is False
    assert decision["fresh_50k_started"] is False


def test_post_ab_engineering_failure_never_selects_route_b(tmp_path: Path) -> None:
    gate_path, gate = _same_contract_gate(tmp_path, status="FAIL", first=None)
    spec = tmp_path / "spec.json"
    _write_json(spec, {"sealed": True})
    gate["trace_on_checkpoint_reload_pass"] = False
    gate["failures"] = ["on_checkpoint_reload_step_501"]
    gate["summary_sha256"] = stable_sha256(
        {key: value for key, value in gate.items() if key != "summary_sha256"}
    )
    _write_json(gate_path, gate)
    decision = select_post_ab_action(
        terminal={"status": "FAILED_A_B_EXIT_1", "fresh_50k_started": False},
        gate=gate,
        ab_spec_path=spec,
        gate_path=gate_path,
    )
    assert decision["branch"] == "ENGINEERING_REPAIR_REQUIRED"
    assert decision["route_b_permitted"] is False
    assert decision["fresh_50k_started"] is False


def test_first_step_marker_without_valid_observer_logs_is_engineering_failure(
    tmp_path: Path,
) -> None:
    _path, gate = _same_contract_gate(tmp_path, status="FAIL", first=17)
    gate["trace_on_observer_log_audit"] = {"status": "FAIL"}
    gate["summary_sha256"] = stable_sha256(
        {key: value for key, value in gate.items() if key != "summary_sha256"}
    )
    assert classify_same_contract_gate(gate) == "ENGINEERING_REPAIR_REQUIRED"


def test_owner_terminal_must_bind_exact_gate_bytes(tmp_path: Path) -> None:
    gate_path, _gate = _same_contract_gate(tmp_path, status="PASS", first=None)
    terminal = {
        "schema_version": "mut_same_contract_ab_owner_terminal_v1",
        "task_id": "mut-ab-test",
        "fresh_50k_started": False,
        "equivalence_gate": str(gate_path),
        "equivalence_gate_sha256": file_sha256(gate_path),
    }
    validate_ab_owner_terminal(terminal, task_id="mut-ab-test", gate_path=gate_path)
    terminal["equivalence_gate_sha256"] = "0" * 64
    with pytest.raises(MutPostABError, match="bytes changed"):
        validate_ab_owner_terminal(
            terminal, task_id="mut-ab-test", gate_path=gate_path
        )


def test_gpu_owners_lock_physical_uuid_and_watcher_never_launches_science() -> None:
    repo = Path(__file__).resolve().parents[2]
    for name in (
        "run_mut_same_contract_ab_owner_v1.py",
        "run_mut_route_b_owner_v1.py",
    ):
        source = (repo / "scripts/autodl" / name).read_text(encoding="utf-8")
        assert "GPUFileLock(" in source
        assert "query_gpu_inventory()" in source
        assert "physical.processes" in source
        assert "gpu_lock.release()" in source
    watcher = (
        repo / "scripts/autodl/run_mut_post_ab_continuation_v1.py"
    ).read_text(encoding="utf-8")
    assert "subprocess" not in watcher
    assert "fresh_50k_started\": False" in watcher


def test_post_ab_rejects_contradictory_owner_terminal(tmp_path: Path) -> None:
    gate_path, gate = _same_contract_gate(tmp_path, status="PASS", first=None)
    spec = tmp_path / "spec.json"
    _write_json(spec, {"sealed": True})
    with pytest.raises(MutPostABError, match="disagrees"):
        select_post_ab_action(
            terminal={
                "status": "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED",
                "fresh_50k_started": False,
            },
            gate=gate,
            ab_spec_path=spec,
            gate_path=gate_path,
        )
