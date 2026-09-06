from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.autodl import run_mut_trace_mode_equivalence as runner
from scripts.autodl import activate_mut_recovery_binding_v1 as activator
from src.baselines.comrecgc.generation_checkpoint import save_generation_checkpoint, scientific_command_sha256
from src.utils import autodl_mut_recovery_binding_v1 as binding
from src.utils import autodl_mut_same_contract_ab_v1 as ab


def _snapshot(root, *, count=3):
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE graphs (key TEXT PRIMARY KEY, payload BLOB)")
    database.execute("INSERT INTO graphs VALUES ('graph0', X'0102')")
    database.commit()
    argv = ("mut-resource-fixture", "M_MAX=50000")
    command = scientific_command_sha256(argv)
    algorithm = {"official_state": {"counterfactual_candidates": [{"graph_hash": "graph0", "frequency": count}],
                                   "graph_index_map": {"graph0": 0}},
                 "loop_state": {"completed_step": 250, "next_step": 251},
                 "live_graph_state": {"rehydrations": 2, "store_write_count": 1}}
    try:
        saved = save_generation_checkpoint(root, completed_step=250, step_complete=True,
            algorithm_state=algorithm, trace_state={"schema_version": "comrecgc_action_trace_state_v1"},
            sqlite_source=database, scientific_argv=argv, command_sha256=command,
            provenance_fingerprints={"project_commit": "a"*40, "scientific_command_sha256": command,
                                     "total_steps": "50000"}, total_steps=50000)
    finally:
        database.close()
    return saved.checkpoint_dir


def _comparison_fixture(tmp_path, monkeypatch, *, different=False):
    random.seed(5)
    old = _snapshot(tmp_path / "old")
    newarm = tmp_path / "new"
    random.seed(5)
    new = _snapshot(newarm / "generation_checkpoints", count=4 if different else 3)
    oldobs = tmp_path / "old-observer.jsonl"
    newobs = newarm / "common_step_state.jsonl"
    rows = [{"phase": "continuous", "step": i, "next_step": i+1, "trace_mode": "on", "move": {"action": i}}
            for i in range(1,251)]
    oldobs.write_text("".join(json.dumps(r)+"\n" for r in rows[:249]))
    newobs.write_text("".join(json.dumps(r)+"\n" for r in rows))
    contract = binding.sealed({"schema_version": "mut_resource_replay_250_v1",
        "original_checkpoint": str(old), "original_checkpoint_manifest_sha256": runner.sha256_file(old/"checkpoint_manifest.json"),
        "original_observer": str(oldobs), "original_observer_sha256": runner.sha256_file(oldobs),
        "replay_range": [1,250], "last_joint_completed_step": 0, "observed_event_range": [1,249],
        "skip_event_250": False, "route_b_on_resource_failure": False,
        "source_algorithm_commit": runner.SOURCE_COMMIT, "instrumentation_commit": runner.INSTRUMENTATION_COMMIT,
        "pythonhashseed": "0"}, "contract_sha256")
    path = tmp_path/"contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(runner, "_resume_common_boundary", lambda *a, **k: {"checkpoint": {"completed_step": 250}})
    monkeypatch.setattr(runner, "_require_no_checkpoint_writer", lambda roots: None)
    monkeypatch.setattr(runner, "_git_head", lambda p: runner.INSTRUMENTATION_COMMIT)
    monkeypatch.setattr(runner, "_install_science_root", lambda p: None)
    return SimpleNamespace(recovery_contract=str(path), replayed_arm=str(newarm),
                           science_project_root=str(tmp_path), output=str(tmp_path/"comparison.json")), new


def test_real_sealed_snapshot_deserialization_compares250_before251(tmp_path, monkeypatch):
    args, _ = _comparison_fixture(tmp_path, monkeypatch)
    assert runner._compare_replayed_250(args) == 0
    receipt = json.loads(Path(args.output).read_text())
    assert receipt["status"] == "PASS"
    assert receipt["independent_checkpoint_deserialization"]
    assert receipt["runtime_restore251_not_yet_performed"]
    assert not receipt["missing_event_skipped"]


def test_replay_mismatch_not_route_b_and_records_exact_component(tmp_path, monkeypatch):
    args, _ = _comparison_fixture(tmp_path, monkeypatch, different=True)
    with pytest.raises(RuntimeError, match="no Route B"):
        runner._compare_replayed_250(args)
    receipt = json.loads(Path(args.output).read_text())
    assert receipt["status"] == "BLOCKED_REPLAY_STATE_MISMATCH"
    assert "serialized_candidate_records_sha256" in receipt["differing_components"]
    assert not receipt["resource_failure_triggers_route_b"]


def test_modified_old_events_are_not_adopted(tmp_path, monkeypatch):
    args, _ = _comparison_fixture(tmp_path, monkeypatch)
    contract = json.loads(Path(args.recovery_contract).read_text())
    Path(contract["original_observer"]).write_text("{}\n")
    with pytest.raises(ValueError, match="ledger changed"):
        runner._compare_replayed_250(args)


def test_payload_must_match_existing_manifest(tmp_path, monkeypatch):
    args, new = _comparison_fixture(tmp_path, monkeypatch)
    with (new/"generation_state.pt").open("ab") as f:
        f.write(b"uncommitted")
    with pytest.raises(ValueError, match="sealed manifest"):
        runner._compare_replayed_250(args)


def test_cannot_relabel250_as500(tmp_path):
    root = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="step-500"):
        runner._checkpoint_state_audit(root, mode="on")


def test_only250_or510_stops_authorized(tmp_path):
    with pytest.raises(ValueError, match="authorized"):
        runner._CommonStepObserver(tmp_path/"x", phase="continuous", trace_mode="on", stop_step=249)
    assert runner._CommonStepObserver(tmp_path/"x", phase="continuous", trace_mode="on", stop_step=250).stop_step == 250


def test_recovery_command_preserves_mode_and_explicit_contract(monkeypatch):
    minimal = {"python": "/python", "runner_path": "/driver/run.py", "controller_project_root": "/driver",
               "recovery_contract": "/contract.json", "legacy_project_root": "/legacy", "execution_project_root": "/science",
               "run_root": "/run", "output_dir": "/audit", "historical_artifact_root": "/historic",
               "rf_oracle": "/rf", "upstream_root": "/upstream", "dataset_dir": "/data",
               "gnn_checkpoint": "/gnn", "distance_checkpoint": "/distance"}
    monkeypatch.setattr(ab, "validate_same_contract_ab_spec", lambda x, **k: dict(x))
    argv = ab.same_contract_ab_command(minimal)
    assert argv[-2:] == ["--recovery-contract", "/contract.json"]
    assert "--resume" not in argv


def test_resource_guard_not_lowered_and_current_files_not_added_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(binding.os, "statvfs", lambda p: SimpleNamespace(f_favail=94644,
        f_bavail=200000000, f_frsize=4096, f_blocks=300000000))
    result = binding.resource_status(tmp_path)
    assert result["state"] == "SEALED_WAITING_INODE_OR_BYTES"
    assert result["fixed_required_free"] == 100160
    assert result["fixed_inode_shortfall"] == 5516
    assert result["unknown_dynamic_peak"] != 0


def test_rebind_preserves_canonical_publisher_namespace():
    old = {"publisher_locator": "/control/canonical/locator.json", "lease": "/control/executor/lease",
           "pipeline": [{"root": "/old-output", "argv": ["/old-driver/run.py", "/old-output/result"]}]}
    rebound = binding._replace_strings(copy.deepcopy(old), {"/old-output": "/fresh-output", "/old-driver": "/new-driver"})
    assert rebound["publisher_locator"] == old["publisher_locator"]
    assert rebound["lease"] == old["lease"]
    assert rebound["pipeline"][0]["argv"] == ["/new-driver/run.py", "/fresh-output/result"]


def test_inode_failure_prevents_even_identity_or_signal_work(tmp_path, monkeypatch):
    values = {"binding_sha256": "ignored", "ab_task_spec": "/ab", "executor_task_spec": "/exec"}
    monkeypatch.setattr(activator, "read_json", lambda p: values if p == tmp_path/"binding.json" else {})
    monkeypatch.setattr(activator, "sealed", lambda v, key: v)
    monkeypatch.setattr(activator, "validate_same_contract_ab_spec", lambda *a, **k: {"run_root": str(tmp_path)})
    monkeypatch.setattr(activator, "validate_successor_spec", lambda *a, **k: {})
    monkeypatch.setattr(activator, "admission", lambda *a: {"state": "SEALED_WAITING_RESOURCE"})
    def prohibited(*a, **k):
        raise AssertionError("resource-blocked activation must not inspect/retire/launch owner")
    monkeypatch.setattr(activator, "idle_executor", prohibited)
    monkeypatch.setattr(activator.os, "kill", prohibited)
    monkeypatch.setattr(activator, "_launch", prohibited)
    result = activator.activate(tmp_path/"binding.json", tmp_path/"registry.json",
                               resource_path=tmp_path/"resource.json", execute=True)
    assert result["state"] == "SEALED_WAITING_RESOURCE"
    assert result["old_executor_signaled"] is False


def test_activation_uses_existing_joint_inode_budget_not_only_mut_minimum(tmp_path, monkeypatch):
    cfg = {"main_registry_path": str(tmp_path/"registry.json"), "gpu_lock_root": str(tmp_path/"locks"),
        "persistent_root": str(tmp_path), "minimum_free_inodes": 100000, "reserved_new_inodes": 10761,
        "minimum_memory_headroom_bytes": 64*1024**3, "minimum_persistent_free_bytes": 100*1024**3,
        "cgroup_memory_root": str(tmp_path/"cgroup")}
    monkeypatch.setattr(activator, "read_json", lambda p: cfg)
    monkeypatch.setattr(activator, "validate_resource_config", lambda c: c)
    monkeypatch.setattr(activator, "read_cgroup", lambda p: {"headroom_bytes": 400*1024**3})
    monkeypatch.setattr(activator, "resource_status", lambda p: {"state": "ADMISSION_PASS",
        "path": str(tmp_path), "fixed_required_free": 100160, "free_inodes": 105000,
        "free_bytes": 1000*1024**3, "minimum_free_bytes": 50*1024**3})
    result = activator.admission({"gpu_lock_root": cfg["gpu_lock_root"], "run_root": str(tmp_path)},
        tmp_path/"registry.json", tmp_path/"resource.json")
    assert result["state"] == "SEALED_WAITING_RESOURCE"
    assert result["joint_required_free_inodes"] == 110761
    assert result["joint_inode_shortfall"] == 5761
    assert result["unknown_dynamic_peak"] == "UNKNOWN_NOT_ZERO"
    assert result["complete_peak_admission_claimed"] is False


def test_unknown_peak_is_not_zero_or_admission(tmp_path):
    assert activator.complete_peak_admission({}, {}, None)["state"] == "RESOURCE_PEAK_PLAN_INCOMPLETE"


def test_peak_plan_requires_every_current_reserved_task(tmp_path):
    path=tmp_path/"plan.json"
    path.write_text(json.dumps({"scope":"MUT_RESOURCE_REPLAY_WITH_CURRENT_MAIN_RESERVATIONS",
        "resource_config_sha256":"a"*64,"task_peak_reservations":[]}))
    result=activator.complete_peak_admission({"resource_config_sha256":"a"*64},
        {"gpu_leases":[{"task_id":"T13","state":"HELD"}]},path)
    assert result["state"]=="RESOURCE_PEAK_PLAN_INCOMPLETE"
    assert "coverage" in result["first_missing_field"]


def _idle_fixture(tmp_path, monkeypatch, *, children=(), stage=None, active_claim=False):
    import time
    from datetime import datetime, timezone
    specpath = tmp_path/"old-spec.json"
    runtime = tmp_path/"old-runtime"
    runtime.mkdir()
    script = str(tmp_path/"scripts/autodl/run_mut_next_stage_executor_v1.py")
    spec = {"task_id": "old", "adoption_pipeline": [{"cwd": str(tmp_path)}], "runtime_root": str(runtime),
            "next_action_path": str(tmp_path/"next_action.json"), "publisher_id": "canonical",
            "publisher_locator": str(tmp_path/"locator.json")}
    specpath.write_text(json.dumps(spec))
    (runtime/"heartbeat.json").write_text(json.dumps({"pid": 123, "written_at": datetime.now(timezone.utc).isoformat(),
        "state": "WAITING_FOR_NEXT_ACTION", "science_pid": None, "stage": stage, "lane": None}))
    if active_claim:
        (runtime/"next_action_consumption.json").write_text("{}")
    monkeypatch.setattr(activator, "identity", lambda pid: {"pid": pid, "start_ticks": 555,
        "argv": ["/python", script, "--task-spec", str(specpath)], "cwd": str(tmp_path), "children": list(children)})
    registry = {"tasks": [{"task_id": "old", "owner_pid": 123, "owner_start_ticks": 555}],
                "publishers": [{"publisher_id": "canonical", "active_writer_count": 0}]}
    return {"old_executor_spec": str(specpath)}, registry


def test_exact_idle_executor_identity_can_be_bound(tmp_path, monkeypatch):
    bound, registry = _idle_fixture(tmp_path, monkeypatch)
    assert activator.idle_executor(bound, registry)["start_ticks"] == 555


@pytest.mark.parametrize("case", ["child", "stage", "claim", "pid_reused"])
def test_executor_with_child_claim_or_changed_identity_cannot_be_retired(tmp_path, monkeypatch, case):
    bound, registry = _idle_fixture(tmp_path, monkeypatch, children=[124] if case == "child" else [],
        stage="ADOPTION" if case == "stage" else None, active_claim=case == "claim")
    if case == "pid_reused":
        registry["tasks"][0]["owner_start_ticks"] = 999
    with pytest.raises(RuntimeError):
        activator.idle_executor(bound, registry)
