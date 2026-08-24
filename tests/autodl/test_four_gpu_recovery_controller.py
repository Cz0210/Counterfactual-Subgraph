from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import scripts.autodl.run_four_gpu_recovery_controller as controller_module
from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    ControllerManifest,
    FixedShardSpec,
    OOMRetry,
    TaskSpec,
    _launch_instance,
    _aggregate_task_state,
    _dependency_output_context,
    _prepare_shards,
    _reconcile_instance,
    _summarize_controller_state,
    _task_manifest_payload,
    allocation_safe_gpu_uuids,
    audit_gpu_locks,
    bind_adopted_runs,
    classify_failure,
    classify_task_failure,
    controller_safety_environment,
    dependency_order,
    load_controller_manifest,
    materialize_fixed_parent_shards,
    oom_retry_allowed,
    publish_user_registry,
    scheduler_candidates,
    transient_retry_allowed,
    validate_no_test_before_freeze,
)
from scripts.autodl.status_four_gpu_recovery import _load_tasks, render_table
from src.utils.autodl_runtime import (
    FOUR_GPU_RECOVERY_LIMIT,
    GPUInventoryError,
    GPUObservation,
    build_runtime_layout,
    validate_max_gpus,
    verify_required_absolute_outputs,
)
from tests.autodl.test_gpu_colocation_benchmark_gate import build_test_gate


def _task(
    task_id: str,
    *,
    dataset: str = "bace",
    stage: str = "AUX",
    depends_on: tuple[str, ...] = (),
    shards: FixedShardSpec | None = None,
    data_splits: tuple[str, ...] = ("calibration",),
    freezes_selector: bool = False,
    selector_parameters_frozen: bool = False,
    read_only_test: bool = False,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        dataset=dataset,
        stage=stage,
        command=("{python}", "payload.py"),
        depends_on=depends_on,
        resource="gpu",
        priority=100,
        enabled=True,
        blocked_reason=None,
        skip_reason=None,
        data_splits=data_splits,
        manifest_only=False,
        runner_dataset=("bace-controller-shard" if shards else dataset),
        runner_stage=(f"{stage}-{{shard_id}}" if shards else stage),
        external_bace_stage=None,
        adopt_existing_run_id=None,
        adopt_gpu_index=None,
        adopt_gpu_uuid=None,
        adopt_project_root=None,
        adopt_git_commit=None,
        adopt_max_gpus=None,
        adopt_heavy=None,
        config_files=(),
        input_manifest="{runtime_root}/data/input.json",
        expected_output=(
            "{artifact_root}/task/{shard_id}/attempt-{attempt}" if shards
            else "{artifact_root}/task/attempt-{attempt}"
        ),
        required_output_files=("PASS.json",),
        required_output_any=(),
        required_absolute_output_files=(),
        required_log_marker="[PASS]",
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
        semantic_failure_markers=("semantic gate failed",),
        oom_retry=OOMRetry(True, "BATCH_SIZE", 8, 4),
        shards=shards,
        publish_bace_stage=False,
        freezes_selector=freezes_selector,
        selector_parameters_frozen=selector_parameters_frozen,
        read_only_test=read_only_test,
    )


def _state(task: TaskSpec, state: str) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "stage": task.stage,
        "state": state,
        "instances": {
            instance_id: {
                "instance_id": instance_id,
                "state": "NOT_STARTED" if state != "PASS" else "PASS",
                "attempt": 0,
            }
            for instance_id in task.instance_ids
        },
    }


def _gpu(index: int) -> GPUObservation:
    return GPUObservation(
        index=index,
        uuid=f"GPU-uuid-{index}",
        name="A800",
        memory_total_mb=81920,
        memory_used_mb=0,
        memory_free_mb=81920,
        utilization_gpu_percent=0,
    )


def test_four_gpu_ceiling_is_explicit_not_a_global_default() -> None:
    assert validate_max_gpus(FOUR_GPU_RECOVERY_LIMIT, hard_limit=4) == 4
    with pytest.raises(GPUInventoryError, match=r"\[1, 2\]"):
        validate_max_gpus(4)


def test_controller_thread_defaults_are_ceilings_and_reject_unsafe_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert controller_safety_environment(cpu_count=112) == {
        "OMP_NUM_THREADS": "27",
        "MKL_NUM_THREADS": "27",
        "TOKENIZERS_PARALLELISM": "false",
    }
    source = (
        Path(__file__).resolve().parents[2]
        / "configs/autodl/four_gpu_recovery.template.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["tasks"][0]["environment"]["OMP_NUM_THREADS"] = "112"
    unsafe = tmp_path / "unsafe-controller.json"
    unsafe.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ControllerError, match="may not override"):
        load_controller_manifest(unsafe)

    payload["tasks"][0]["environment"]["OMP_NUM_THREADS"] = "1"
    payload["tasks"][0]["environment"]["MKL_NUM_THREADS"] = "1"
    bounded = tmp_path / "bounded-controller.json"
    bounded.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(controller_module.os, "cpu_count", lambda: 112)
    manifest = load_controller_manifest(bounded)
    assert manifest.tasks[0].environment["OMP_NUM_THREADS"] == "1"
    assert manifest.tasks[0].environment["MKL_NUM_THREADS"] == "1"


def test_scheduler_is_work_conserving_but_never_violates_dependencies() -> None:
    first = _task("first")
    sharded = _task(
        "verify",
        depends_on=("first",),
        shards=FixedShardSpec(4, "/persistent/parents.json", "calibration"),
    )
    final = _task("final", depends_on=("verify",))
    tasks = (first, sharded, final)
    states = {
        "first": _state(first, "PASS"),
        "verify": _state(sharded, "READY"),
        "final": _state(final, "WAITING_DEPENDENCY"),
    }
    assert dependency_order(tasks) == ["first", "verify", "final"]
    assert scheduler_candidates(tasks, states) == [
        ("verify", "shard-000"),
        ("verify", "shard-001"),
        ("verify", "shard-002"),
        ("verify", "shard-003"),
    ]

    states["verify"] = _state(sharded, "PASS")
    states["final"] = _state(final, "READY")
    assert scheduler_candidates(tasks, states) == [("final", "main")]


def test_dependency_cycle_is_rejected() -> None:
    left = _task("left", depends_on=("right",))
    right = _task("right", depends_on=("left",))
    with pytest.raises(ControllerError, match="cycle"):
        dependency_order((left, right))


def test_fixed_parent_shards_are_disjoint_exhaustive_and_immutable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parents.json"
    source.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset": "bace",
                "split": "calibration",
                "parent_ids": ["p5", "p1", "p4", "p2", "p3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = materialize_fixed_parent_shards(
        source_manifest=source,
        destination_root=tmp_path / "shards",
        shard_count=4,
        expected_dataset="bace",
        expected_split="calibration",
    )
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    parent_ids = [
        parent_id for document in documents for parent_id in document["parent_ids"]
    ]
    assert sorted(parent_ids) == ["p1", "p2", "p3", "p4", "p5"]
    assert len(parent_ids) == len(set(parent_ids))
    assert len({document["assignment_sha256"] for document in documents}) == 1

    # Re-reading identical bytes is resumable; changed source bytes fail closed.
    assert materialize_fixed_parent_shards(
        source_manifest=source,
        destination_root=tmp_path / "shards",
        shard_count=4,
        expected_dataset="bace",
        expected_split="calibration",
    ) == paths
    source.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset": "bace",
                "split": "calibration",
                "parent_ids": ["p1", "p2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ControllerError, match="changed"):
        materialize_fixed_parent_shards(
            source_manifest=source,
            destination_root=tmp_path / "shards",
            shard_count=4,
            expected_dataset="bace",
            expected_split="calibration",
        )


def test_test_split_is_unreachable_until_selector_freeze() -> None:
    unsafe = _task("unsafe", stage="B11_CROSS_PARENT_VERIFIED", data_splits=("test",))
    with pytest.raises(ControllerError, match="one-shot B13"):
        validate_no_test_before_freeze((unsafe,))

    selector = _task(
        "selector", stage="B12_SELECTOR", freezes_selector=True
    )
    heldout = _task(
        "heldout",
        stage="B13_FINAL_EVAL",
        depends_on=("selector",),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    validate_no_test_before_freeze((selector, heldout))

    am_selector = _task(
        "am-selector",
        stage="AM_MUT_GCF_CALIBRATION_FREEZE",
        freezes_selector=True,
    )
    am_heldout = _task(
        "am-heldout",
        stage="AM_MUT_GCF_HELDOUT_EVAL",
        depends_on=("am-selector",),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    validate_no_test_before_freeze((am_selector, am_heldout))

    baseline_selector = _task(
        "baseline-selector",
        stage="BACE_BASELINE_SELECTOR",
        freezes_selector=True,
    )
    baseline_heldout = _task(
        "baseline-heldout",
        stage="BACE_BASELINE_TEST_VERIFY",
        depends_on=("baseline-selector",),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    validate_no_test_before_freeze((baseline_selector, baseline_heldout))

    postfreeze_manifest = _task(
        "test-manifest",
        stage="B13_TEST_PARENT_MANIFEST",
        depends_on=("selector",),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    postfreeze_shards = _task(
        "test-shards",
        stage="B13_FINAL_EVAL_SHARDS",
        depends_on=("test-manifest",),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    validate_no_test_before_freeze((selector, postfreeze_manifest, postfreeze_shards))

    missing_selector = TaskSpec(
        **{
            **postfreeze_shards.__dict__,
            "task_id": "missing-selector",
            "depends_on": (),
        }
    )
    with pytest.raises(ControllerError, match="frozen B12"):
        validate_no_test_before_freeze((missing_selector,))

    hidden = TaskSpec(
        **{
            **_task("hidden").__dict__,
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PARENT_CSV": "/persistent/data/test.csv",
            },
        }
    )
    with pytest.raises(ControllerError, match="test-looking"):
        validate_no_test_before_freeze((hidden,))


def test_only_frozen_b13_may_materialize_four_heldout_shards(
    tmp_path: Path,
) -> None:
    project = tmp_path / "controller-project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    source = layout.data_dir / "bace-test-parents.json"
    source.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset": "bace",
                "split": "test",
                "parent_ids": [f"p{index}" for index in range(9)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selector = _task(
        "selector", stage="B12_SELECTOR", freezes_selector=True
    )
    heldout = _task(
        "heldout",
        stage="B13_FINAL_EVAL",
        depends_on=("selector",),
        shards=FixedShardSpec(4, str(source), "test"),
        data_splits=("test",),
        selector_parameters_frozen=True,
        read_only_test=True,
    )
    validate_no_test_before_freeze((selector, heldout))
    paths = _prepare_shards(
        layout,
        layout.control_root / "four_gpu_recovery" / "controller",
        heldout,
        Path(sys.executable).resolve(),
    )
    assert sorted(paths) == [f"shard-{index:03d}" for index in range(4)]

    wrong_stage = TaskSpec(
        **{
            **heldout.__dict__,
            "task_id": "wrong-stage",
            "stage": "B11_CROSS_PARENT_VERIFIED",
        }
    )
    with pytest.raises(ControllerError, match="may not load held-out test"):
        _prepare_shards(
            layout,
            layout.control_root / "four_gpu_recovery" / "controller",
            wrong_stage,
            Path(sys.executable).resolve(),
        )


def test_oom_has_exactly_one_lower_batch_retry_and_semantic_never_retries() -> None:
    policy = OOMRetry(True, "BATCH_SIZE", 8, 4)
    assert classify_failure("torch.OutOfMemoryError: CUDA out of memory") == "OOM"
    assert oom_retry_allowed("OOM", 0, policy)
    assert not oom_retry_allowed("OOM", 1, policy)
    assert not oom_retry_allowed("SEMANTIC", 0, policy)
    assert classify_failure("semantic gate failed: wrong oracle") == "SEMANTIC"


def test_transient_io_has_exactly_one_fresh_attempt_retry() -> None:
    assert classify_failure("OSError: [Errno 5] Input/output error") == "TRANSIENT_IO"
    assert transient_retry_allowed(
        "TRANSIENT_IO", 0, max_transient_retries=1
    )
    assert not transient_retry_allowed(
        "TRANSIENT_IO", 1, max_transient_retries=1
    )
    assert transient_retry_allowed(
        "TRANSIENT_PROCESS_LOSS", 0, max_transient_retries=1
    )
    assert not transient_retry_allowed("EXECUTION", 0, max_transient_retries=1)


def test_signal_exit_retry_is_task_opt_in_and_semantic_first() -> None:
    base = _task("signal-loss")
    opted_in = TaskSpec(**{**base.__dict__, "retry_on_process_loss": True})
    signal_log = "[AUTODL_RUN_EXIT] exit_code=143 timestamp=now\n"
    assert classify_task_failure(signal_log, base) == "EXECUTION"
    assert classify_task_failure(signal_log, opted_in) == "TRANSIENT_PROCESS_LOSS"
    assert classify_task_failure(
        "semantic gate failed\n" + signal_log, opted_in
    ) == "SEMANTIC"


def test_workload_state_can_pass_while_raw_taste_gate_remains_blocked() -> None:
    assert _summarize_controller_state({"PASS": 3, "BLOCKED": 1}) == "BLOCKED"
    assert _summarize_controller_state({"PASS": 3}) == "PASS"


def test_dead_starting_and_running_workers_retry_once_then_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    task = _task("dead-worker", dataset="mutagenicity")
    run_root = layout.runs_root / "dead-starting"
    run_root.mkdir(parents=True)
    (run_root / "state.json").write_text(
        json.dumps(
            {
                "state": "STARTING",
                "pid": None,
                "child_pid": None,
                "updated_at": "2020-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    starting = {
        "state": "STARTING",
        "run_id": "dead-starting",
        "started_at": "2020-01-01T00:00:00Z",
        "heartbeat_at": "2099-01-01T00:00:00Z",
        "launcher_pid": None,
        "tmux_session": None,
    }
    # Even a surviving detached launcher/tmux cannot substitute for an
    # exp_run worker after the bounded launch grace expires.
    monkeypatch.setattr(
        controller_module, "_launcher_evidence_alive", lambda _instance: True
    )
    _reconcile_instance(
        layout,
        task,
        starting,
        launch_grace_seconds=60,
        now_epoch=1_700_000_000.0,
    )
    assert starting["state"] == "NOT_STARTED"
    assert starting["attempt"] == 1
    assert starting["failure_class"] == "TRANSIENT_PROCESS_LOSS_RETRY"
    assert starting["retry_kind"] == "TRANSIENT_PROCESS_LOSS"
    assert starting["heartbeat_at"] is None

    missing_state = {
        "state": "STARTING",
        "run_id": "missing-run-state",
        "started_at": "2020-01-01T00:00:00Z",
        "heartbeat_at": "2099-01-01T00:00:00Z",
    }
    _reconcile_instance(
        layout,
        task,
        missing_state,
        launch_grace_seconds=60,
        now_epoch=1_700_000_000.0,
    )
    assert missing_state["state"] == "NOT_STARTED"
    assert missing_state["retry_kind"] == "TRANSIENT_PROCESS_LOSS"
    assert missing_state["heartbeat_at"] is None

    running_root = layout.runs_root / "dead-running"
    running_root.mkdir(parents=True)
    (running_root / "state.json").write_text(
        json.dumps({"state": "RUNNING", "pid": 2_000_000_000}) + "\n",
        encoding="utf-8",
    )
    running = {
        "state": "RUNNING",
        "run_id": "dead-running",
        "started_at": "2020-01-01T00:00:00Z",
        "attempt": 1,
        "transient_retry_count": 1,
    }
    _reconcile_instance(layout, task, running)
    assert running["state"] == "FAILED"
    assert running["failure_class"] == "STALE_PROCESS"
    assert running["failure_reason"] == "RUNNING exp_run worker PID is absent"


def test_dead_worker_log_semantic_marker_preempts_global_process_loss_retry(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    log = tmp_path / "dead-worker.log"
    log.write_text("semantic gate failed: provenance differs\n", encoding="utf-8")
    task = _task("dead-semantic", dataset="mutagenicity")
    instance = {
        "state": "STARTING",
        "run_id": "missing-semantic-run",
        "started_at": "2020-01-01T00:00:00Z",
        "attempt": 0,
        "transient_retry_count": 0,
        "log_path": str(log),
    }
    _reconcile_instance(
        layout,
        task,
        instance,
        launch_grace_seconds=60,
        max_transient_retries=1,
        now_epoch=1_700_000_000.0,
    )
    assert instance["state"] == "FAILED"
    assert instance["failure_class"] == "SEMANTIC"
    assert instance["attempt"] == 0


def test_b7_transient_failure_records_latest_checkpoint_for_fresh_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    source_output = layout.artifacts_dir / "bace" / "attempt-0"
    checkpoint = source_output / "checkpoint-50"
    checkpoint.mkdir(parents=True)
    run_root = layout.runs_root / "b7-failed"
    run_root.mkdir(parents=True)
    log = layout.logs_dir / "b7-failed.log"
    log.write_text("OSError: [Errno 5] Input/output error\n", encoding="utf-8")
    (run_root / "state.json").write_text(
        json.dumps(
            {
                "state": "FAILED",
                "pid": None,
                "child_pid": None,
                "log_path": str(log),
                "failures": ["scientific command exited 1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        controller_module,
        "find_latest_stable_ppo_resume_checkpoint",
        lambda output: checkpoint,
    )
    task = _task("b7", stage="B7_PPO_FULL")
    instance = {
        "state": "RUNNING",
        "run_id": "b7-failed",
        "attempt": 0,
        "adopted": False,
        "expected_output": str(source_output),
    }
    _reconcile_instance(
        layout,
        task,
        instance,
        max_transient_retries=1,
    )
    assert instance["state"] == "NOT_STARTED"
    assert instance["attempt"] == 1
    assert instance["retry_kind"] == "TRANSIENT_IO"
    assert instance["resume_from_checkpoint"] == str(checkpoint)
    assert instance["resume_source_output"] == str(source_output.resolve())


def test_failed_shard_drains_live_siblings_before_terminal_aggregate(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    task = _task(
        "sharded",
        dataset="mutagenicity",
        shards=FixedShardSpec(4, "/persistent/parents.json", "calibration"),
    )
    state = _state(task, "RUNNING")
    instances = state["instances"]
    instances["shard-000"].update(
        {"state": "FAILED", "failure_class": "SEMANTIC"}
    )
    instances["shard-001"]["state"] = "RUNNING"
    root = layout.control_root / "four_gpu_recovery" / "controller"
    (root / "tasks" / task.task_id).mkdir(parents=True)
    manifest_path = tmp_path / "controller.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = ControllerManifest(
        path=manifest_path,
        sha256="0" * 64,
        controller_id="controller",
        tasks=(task,),
        runtime={"max_gpus": 4},
        resource_gates={},
    )
    states = {task.task_id: state}
    _aggregate_task_state(root, layout, manifest, task, state, states)
    assert state["state"] == "RUNNING"
    assert "DRAINING_ACTIVE_SIBLINGS" in state["reason"]
    assert scheduler_candidates((task,), states) == []

    instances["shard-001"]["state"] = "PASS"
    _aggregate_task_state(root, layout, manifest, task, state, states)
    assert state["state"] == "FAILED"


def test_retryable_parent_consumers_resolve_the_passing_attempt_output(
    tmp_path: Path,
) -> None:
    parent = _task("bace_b6_ppo_smoke", stage="B6_PPO_SMOKE_V2")
    child = _task(
        "bace_b7_ppo_full",
        stage="B7_PPO_FULL",
        depends_on=(parent.task_id,),
    )
    parent_state = _state(parent, "PASS")
    parent_state["instances"]["main"]["attempt"] = 1
    parent_state["instances"]["main"]["expected_output"] = str(
        tmp_path / "b6-v2" / "attempt-1"
    )
    context = _dependency_output_context(
        tmp_path,
        child,
        {
            parent.task_id: parent_state,
            child.task_id: _state(child, "WAITING_DEPENDENCY"),
        },
    )
    assert context["dep_bace_b6_ppo_smoke_output"].endswith(
        "/b6-v2/attempt-1"
    )


def test_sharded_dependency_tokens_bind_each_actual_passing_attempt(
    tmp_path: Path,
) -> None:
    shards = FixedShardSpec(4, "/persistent/parents.json", "train")
    parent = _task("bace_b8_pool_base", stage="B8_POOL_BASE", shards=shards)
    child = _task(
        "bace_b10_pool_merged",
        stage="B10_POOL_MERGED",
        depends_on=(parent.task_id,),
    )
    parent_state = _state(parent, "PASS")
    for index, instance in enumerate(parent_state["instances"].values()):
        instance["state"] = "PASS"
        instance["attempt"] = index % 2
        instance["expected_output"] = str(
            tmp_path / f"shard-{index:03d}" / f"attempt-{index % 2}"
        )
    context = _dependency_output_context(
        tmp_path,
        child,
        {parent.task_id: parent_state, child.task_id: _state(child, "READY")},
    )
    assert context["dep_bace_b8_pool_base_shard_000_output"].endswith(
        "/shard-000/attempt-0"
    )
    assert context["dep_bace_b8_pool_base_shard_001_output"].endswith(
        "/shard-001/attempt-1"
    )
    assert context["dep_bace_b8_pool_base_output"].endswith(
        "/tasks/bace_b8_pool_base"
    )


def test_prepare_shards_expands_parent_manifest_from_dependency_output(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    prep_output = layout.artifacts_dir / "bace" / "prep" / "attempt-1"
    prep_output.mkdir(parents=True)
    (prep_output / "train_parent_ids.frozen.json").write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset": "bace",
                "split": "train",
                "parent_ids": ["p3", "p1", "p2", "p4"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prep = _task("bace_b7_prep_shard_manifests", stage="B7_PREP_SHARDS")
    child = _task(
        "bace_b8_pool_base",
        stage="B8_POOL_BASE",
        depends_on=(prep.task_id,),
        shards=FixedShardSpec(
            4,
            "{dep_bace_b7_prep_shard_manifests_output}/train_parent_ids.frozen.json",
            "train",
        ),
        data_splits=("train",),
    )
    prep_state = _state(prep, "PASS")
    prep_state["instances"]["main"]["expected_output"] = str(prep_output)
    states = {
        prep.task_id: prep_state,
        child.task_id: _state(child, "READY"),
    }
    paths = _prepare_shards(
        layout,
        layout.control_root / "four_gpu_recovery" / "controller",
        child,
        Path(sys.executable).resolve(),
        states,
    )
    assert sorted(paths) == [f"shard-{index:03d}" for index in range(4)]


def test_template_clean_policy_dependencies_follow_the_passing_retry_attempt(
    tmp_path: Path,
) -> None:
    manifest = load_controller_manifest(
        Path(__file__).resolve().parents[2]
        / "configs/autodl/four_gpu_recovery.template.json"
    )
    audit = manifest.by_id["bace_policy_provenance_audit"]
    initializer = manifest.by_id["bace_clean_initializer"]
    audit_state = _state(audit, "PASS")
    audit_state["instances"]["main"].update(
        {
            "attempt": 1,
            "expected_output": str(tmp_path / "audit" / "attempt-1"),
        }
    )
    context = _dependency_output_context(
        tmp_path,
        initializer,
        {
            audit.task_id: audit_state,
            initializer.task_id: _state(initializer, "READY"),
        },
    )
    assert context["dep_bace_policy_provenance_audit_output"].endswith(
        "/audit/attempt-1"
    )
    assert "{dep_bace_policy_provenance_audit_output}" in (
        initializer.input_manifest or ""
    )


def test_stale_uuid_lock_metadata_is_audited_but_not_deleted(tmp_path: Path) -> None:
    gpu = _gpu(0)
    lock_path = tmp_path / "gpu-GPU-uuid-0.lock"
    lock_path.write_text(
        json.dumps({"state": "LOCKED", "pid": 2_000_000_000}) + "\n",
        encoding="utf-8",
    )
    rows = audit_gpu_locks(tmp_path, [gpu], probe_advisory_lock=True)
    assert rows[0]["audit"] == "STALE_METADATA"
    assert lock_path.is_file()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["state"] == "LOCKED"


def test_live_uuid_lock_metadata_is_never_reused_when_advisory_fd_is_openable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu = _gpu(0)
    owner_pid = 4242
    lock_path = tmp_path / "gpu-GPU-uuid-0.lock"
    lock_path.write_text(
        json.dumps({"state": "LOCKED", "pid": owner_pid}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        controller_module,
        "read_process_identity",
        lambda pid: (
            {"pid": pid, "start_ticks": 1, "command_sha256": "a" * 64}
            if pid == owner_pid
            else None
        ),
    )
    rows = audit_gpu_locks(tmp_path, [gpu], probe_advisory_lock=True)
    assert rows[0]["advisory_lock_available"] is True
    assert rows[0]["audit"] == "INDETERMINATE"
    assert allocation_safe_gpu_uuids(rows) == frozenset()


@pytest.mark.parametrize("owner_pid", [None, "4242", True, 0, -1])
def test_locked_uuid_metadata_requires_one_explicit_positive_integer_pid(
    tmp_path: Path,
    owner_pid: Any,
) -> None:
    gpu = _gpu(0)
    lock_path = tmp_path / "gpu-GPU-uuid-0.lock"
    lock_path.write_text(
        json.dumps({"state": "LOCKED", "pid": owner_pid}) + "\n",
        encoding="utf-8",
    )
    rows = audit_gpu_locks(tmp_path, [gpu], probe_advisory_lock=True)
    assert rows[0]["lock_owner_pid_valid"] is False
    assert rows[0]["audit"] == "MALFORMED_LOCK_METADATA"
    assert allocation_safe_gpu_uuids(rows) == frozenset()


def test_advisory_uuid_lock_never_fails_open_on_nonlocked_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu = _gpu(0)
    lock_path = tmp_path / "gpu-GPU-uuid-0.lock"
    lock_path.write_text(
        json.dumps({"state": "RELEASED", "pid": None}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(controller_module, "gpu_lock_available", lambda *_: False)
    rows = audit_gpu_locks(tmp_path, [gpu], probe_advisory_lock=True)
    assert rows[0]["audit"] == "INDETERMINATE"
    assert allocation_safe_gpu_uuids(rows) == frozenset()


def test_launch_delegates_to_exp_run_with_frozen_python_and_four_gpu_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller_module.os, "cpu_count", lambda: 112)
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    input_manifest = layout.data_dir / "input.json"
    input_manifest.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    config = project / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    task = _task("launch")
    task = TaskSpec(
        **{
            **task.__dict__,
            "runner_dataset": "bace-gnn-clean-audit",
            "runner_stage": "BACE_POLICY_PROVENANCE_AUDIT",
            "config_files": (str(config),),
            "input_manifest": str(input_manifest),
            "required_absolute_output_files": (
                str(layout.artifacts_dir / "autodl" / "audits" / "audit.csv"),
            ),
        }
    )
    state = _state(task, "READY")
    root = layout.control_root / "four_gpu_recovery" / "test"
    task_root = root / "tasks" / task.task_id
    task_root.mkdir(parents=True)
    frozen_task = _task_manifest_payload(task, "0" * 64)
    (task_root / "manifest.json").write_text(
        json.dumps(frozen_task) + "\n",
        encoding="utf-8",
    )
    assert frozen_task["controller_safety_environment"] == {
        "OMP_NUM_THREADS": "27",
        "MKL_NUM_THREADS": "27",
        "TOKENIZERS_PARALLELISM": "false",
    }
    captured: dict[str, Any] = {}

    def fake_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "run_id": "controller-launch-main-a0",
                    "launcher_pid": 123,
                    "tmux_session": None,
                }
            ),
            stderr="",
        )

    class _Manifest:
        controller_id = "controller"
        sha256 = "0" * 64
        runtime = {
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "nohup",
        }

    interpreter = Path(sys.executable).resolve()
    _launch_instance(
        layout,
        root,
        _Manifest(),  # type: ignore[arg-type]
        task,
        state,
        "main",
        python_executable=interpreter,
        gpu=_gpu(0),
        shard_manifest=None,
        extra_context=None,
        runner=fake_runner,
    )
    command = captured["command"]
    assert command[0] == str(interpreter)
    assert command[1].endswith("scripts/autodl/exp_run.py")
    assert command[command.index("--gpu-hard-limit") + 1] == "4"
    assert command[command.index("--dataset") + 1] == "bace-gnn-clean-audit"
    assert command[command.index("--stage") + 1] == "BACE_POLICY_PROVENANCE_AUDIT"
    assert command[command.index("--required-absolute-output-file") + 1].endswith(
        "/outputs/autodl/audits/audit.csv"
    )
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert "OMP_NUM_THREADS=27" in command
    assert "MKL_NUM_THREADS=27" in command
    assert "TOKENIZERS_PARALLELISM=false" in command
    payload_start = command.index("--") + 1
    assert command[payload_start] == str(interpreter)
    assert state["instances"]["main"]["state"] == "STARTING"


def test_shared_launch_revalidates_gate_bytes_after_task_freeze(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    input_manifest = layout.data_dir / "input.json"
    input_manifest.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    gate_path, gate_sha256 = build_test_gate(tmp_path)
    base = _task("shared-launch")
    task = TaskSpec(
        **{
            **base.__dict__,
            "gpu_lock_mode": "shared_lowmem_slot_0",
            "gpu_memory_reservation_mb": 12000,
            "gpu_shared_workload_class": "bace_gcfexplainer_vrrw",
            "gpu_colocation_gate": str(gate_path),
            "gpu_colocation_gate_sha256": gate_sha256,
            "input_manifest": str(input_manifest),
        }
    )
    state = _state(task, "READY")
    root = layout.control_root / "four_gpu_recovery" / "controller"
    task_root = root / "tasks" / task.task_id
    task_root.mkdir(parents=True)
    (task_root / "manifest.json").write_text(
        json.dumps(_task_manifest_payload(task, "0" * 64)) + "\n",
        encoding="utf-8",
    )
    # Schema validation already succeeded when the task was frozen.  A later
    # byte change must still fail immediately before exp_run is invoked.
    gate_path.write_text(gate_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    class _Manifest:
        controller_id = "controller"
        sha256 = "0" * 64
        runtime = {
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "nohup",
        }

    gpu = GPUObservation(
        index=0,
        uuid="GPU-shared",
        name="NVIDIA A800 80GB PCIe",
        memory_total_mb=81920,
        memory_used_mb=0,
        memory_free_mb=81920,
        utilization_gpu_percent=0,
    )
    with pytest.raises(ControllerError, match="launch-time.*SHA256 mismatch"):
        _launch_instance(
            layout,
            root,
            _Manifest(),  # type: ignore[arg-type]
            task,
            state,
            "main",
            python_executable=Path(sys.executable).resolve(),
            gpu=gpu,
            shard_manifest=None,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not start"),
        )


def test_launch_expands_task_output_before_command_and_numeric_shard_index(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    input_manifest = layout.data_dir / "input.json"
    input_manifest.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    shards = FixedShardSpec(4, str(layout.data_dir / "parents.json"), "train")
    base = _task("launch-shard", stage="B8_POOL_BASE", shards=shards)
    task = TaskSpec(
        **{
            **base.__dict__,
            "command": (
                "{python}",
                "payload.py",
                "--output-dir",
                "{task_output}",
                "--shard-index",
                "{shard_index}",
            ),
            "input_manifest": str(input_manifest),
        }
    )
    state = _state(task, "READY")
    root = layout.control_root / "four_gpu_recovery" / "controller"
    task_root = root / "tasks" / task.task_id
    task_root.mkdir(parents=True)
    (task_root / "manifest.json").write_text(
        json.dumps(_task_manifest_payload(task, "0" * 64)) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "run_id": "controller-launch-shard-003-a0",
                    "launcher_pid": 123,
                    "tmux_session": None,
                }
            ),
            stderr="",
        )

    class _Manifest:
        controller_id = "controller"
        sha256 = "0" * 64
        runtime = {
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "nohup",
        }

    _launch_instance(
        layout,
        root,
        _Manifest(),  # type: ignore[arg-type]
        task,
        state,
        "shard-003",
        python_executable=Path(sys.executable).resolve(),
        gpu=_gpu(3),
        shard_manifest=layout.data_dir / "shard-003.json",
        runner=fake_runner,
    )
    payload = captured["command"][captured["command"].index("--") + 1 :]
    expected = str(layout.artifacts_dir / "task" / "shard-003" / "attempt-0")
    assert payload[payload.index("--output-dir") + 1] == expected
    assert payload[payload.index("--shard-index") + 1] == "3"


def test_existing_exp_run_is_strictly_adopted_and_not_relaunched(
    tmp_path: Path,
) -> None:
    project = tmp_path / "controller-project"
    adopted_project = tmp_path / "adopted-project"
    data = tmp_path / "data"
    project.mkdir()
    adopted_project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    (adopted_project / ".git").mkdir()
    (adopted_project / "payload.sh").write_text("exit 0\n", encoding="utf-8")
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    input_manifest = layout.data_dir / "input.json"
    input_manifest.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    output = layout.artifacts_dir / "task" / "attempt-0"
    output.mkdir(parents=True)
    (output / "PASS.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    log = layout.logs_dir / "adopted.log"
    log.write_text("[PASS]\n", encoding="utf-8")
    interpreter = Path(sys.executable).resolve()
    base = _task("adopt", dataset="mutagenicity")
    task = TaskSpec(
        **{
            **base.__dict__,
            "adopt_existing_run_id": "existing-run",
            "resource": "cpu",
            "command": ("bash", "{project_root}/payload.sh"),
            "adopt_project_root": str(adopted_project),
            "adopt_git_commit": "1" * 40,
            "adopt_max_gpus": 2,
            "adopt_heavy": False,
            "input_manifest": str(input_manifest),
        }
    )
    run_root = layout.runs_root / "existing-run"
    run_root.mkdir(parents=True)
    spec = {
        "schema_version": 1,
        "run_id": "existing-run",
        "dataset": "mutagenicity",
        "stage": "AUX",
        "command": ["bash", str(adopted_project / "payload.sh")],
        "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "input_manifest": str(input_manifest),
        "input_hash": hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
        "expected_output": str(output),
        "required_output_files": ["PASS.json"],
        "required_output_any": [],
        "required_absolute_output_files": [],
        "required_log_marker": "[PASS]",
        "python_executable": str(interpreter),
        "project_root": str(adopted_project.resolve()),
        "data_root": str(data.resolve()),
        "control_root": str(layout.control_root),
        "git_commit": "1" * 40,
        "max_gpus": 2,
        "heavy": False,
        "config_files": [],
        "config_hash": None,
        "gpu_index": None,
        "gpu_uuid": None,
    }
    (run_root / "launch_spec.json").write_text(
        json.dumps(spec) + "\n", encoding="utf-8"
    )
    (run_root / "state.json").write_text(
        json.dumps(
            {
                "run_id": "existing-run",
                "dataset": "mutagenicity",
                "stage": "AUX",
                "state": "PASS",
                "pid": 123,
                "child_pid": 456,
                "log_path": str(log),
                "gpu_index": None,
                "gpu_uuid": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # A reused generic stage slot may still contain terminal fields from an
    # older run. Adoption authority is the exact run-specific state/spec above,
    # never this mutable compatibility projection.
    generic_stage = layout.control_root / "mutagenicity" / "stages" / "AUX"
    generic_stage.mkdir(parents=True)
    (generic_stage / "state.json").write_text(
        json.dumps(
            {
                "run_id": "old-failed-run",
                "state": "FAILED",
                "exit_code": 1,
                "completed_at": "2020-01-01T00:00:00Z",
                "failures": ["stale old stage projection"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root = layout.control_root / "four_gpu_recovery" / "controller"
    state = _state(task, "NOT_STARTED")
    states = {task.task_id: state}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = ControllerManifest(
        path=manifest_path,
        sha256="0" * 64,
        controller_id="controller",
        tasks=(task,),
        runtime={"max_gpus": 4},
        resource_gates={},
    )
    bind_adopted_runs(
        layout,
        root,
        manifest,
        states,
        python_executable=interpreter,
    )
    instance = states["adopt"]["instances"]["main"]
    assert instance["run_id"] == "existing-run"
    assert instance["adopted"] is True
    _reconcile_instance(layout, task, instance)
    assert instance["state"] == "PASS"
    assert scheduler_candidates((task,), states) == []

    spec["environment"]["UNDECLARED"] = "forbidden"
    (run_root / "launch_spec.json").write_text(
        json.dumps(spec) + "\n", encoding="utf-8"
    )
    with pytest.raises(ControllerError, match="environment mismatch"):
        bind_adopted_runs(
            layout,
            root,
            manifest,
            {task.task_id: _state(task, "NOT_STARTED")},
            python_executable=interpreter,
        )

    spec["environment"].pop("UNDECLARED")
    spec["project_root"] = str(project.resolve())
    (run_root / "launch_spec.json").write_text(
        json.dumps(spec) + "\n", encoding="utf-8"
    )
    with pytest.raises(ControllerError, match="project_root mismatch"):
        bind_adopted_runs(
            layout,
            root,
            manifest,
            {task.task_id: _state(task, "NOT_STARTED")},
            python_executable=interpreter,
        )


def test_committed_integration_template_is_valid_and_keeps_taste_blocked() -> None:
    root = Path(__file__).resolve().parents[2]
    template_path = root / "configs/autodl/four_gpu_recovery.template.json"

    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            assert key not in result, f"duplicate JSON key: {key}"
            result[key] = value
        return result

    json.loads(
        template_path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicate_keys,
    )
    manifest = load_controller_manifest(
        template_path
    )
    assert manifest.runtime["max_gpus"] == 4
    assert manifest.by_id["tastemolnet_foundation"].blocked_reason == (
        "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW"
    )
    assert manifest.by_id["bace_b11_verification_shards"].shards is not None
    assert manifest.by_id["bace_b11_cross_parent_verified"].shards is None
    assert manifest.by_id["bace_b6_ppo_smoke"].stage == "B6_PPO_SMOKE_V2"
    assert manifest.by_id["bace_b6_ppo_smoke"].depends_on == (
        "bace_gnn_ppo_adapter_canary",
        "bace_clean_initializer",
    )
    assert manifest.by_id["bace_b6_ppo_smoke"].adopt_existing_run_id is None
    assert manifest.by_id["bace_b7_ppo_full"].adopt_existing_run_id is None
    assert manifest.by_id["bace_gnn_ppo_adapter_canary"].depends_on == (
        "bace_clean_initializer",
    )
    assert manifest.by_id["bace_b9_pool_hightemp"].depends_on == (
        "bace_b7_ppo_full",
        "bace_b7_prep_shard_manifests",
        "bace_b7_prep_output_preflight",
    )
    assert manifest.by_id["bace_b13_verification_shards"].shards is not None
    assert manifest.by_id["bace_b13_verification_shards"].shards.count == 4


def test_live_candidate_accepts_exact_adoptions_but_keeps_b6_b7_fresh() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "configs/autodl/four_gpu_recovery.live_candidate.json"
    manifest = load_controller_manifest(
        manifest_path
    )

    old_controller_id = "autodl-four-gpu-recovery-20260822T033351Z-v1"
    assert manifest.controller_id == (
        "autodl-four-gpu-recovery-20260822T044445Z-v2"
    )
    assert old_controller_id not in manifest_path.read_text(encoding="utf-8")
    assert len(manifest.tasks) == 22
    assert manifest.runtime["launch_grace_seconds"] == 180
    assert manifest.runtime["max_transient_retries"] == 1
    assert manifest.by_id["mut_recovery"].adopt_existing_run_id is not None
    assert manifest.by_id["aids_recovery"].adopt_existing_run_id is not None
    assert (
        manifest.by_id["bace_gnn_ppo_adapter_canary"].adopt_existing_run_id
        is not None
    )
    assert manifest.by_id["bace_b6_ppo_smoke"].adopt_existing_run_id is None
    assert manifest.by_id["bace_b7_ppo_full"].adopt_existing_run_id is None
    assert "{shard_id}" in manifest.by_id["bace_b13_verification_shards"].runner_stage
    assert manifest.by_id["bace_b13_final_eval"].shards is None
    assert manifest.by_id["bace_b11_cross_parent_verified"].depends_on == (
        "bace_b11_verification_shards",
        "bace_b10_pool_merged",
    )
    assert manifest.by_id["bace_b13_test_parent_manifest"].depends_on == (
        "bace_b12_selector",
    )
    assert manifest.by_id["bace_b13_verification_shards"].depends_on == (
        "bace_b12_selector",
        "bace_b13_test_parent_manifest",
    )
    assert manifest.by_id["bace_b13_final_eval"].depends_on == (
        "bace_b13_verification_shards",
        "bace_b12_selector",
    )
    assert manifest.by_id["bace_b14_frozen"].manifest_only is True
    assert manifest.by_id["bace_b14_frozen"].data_splits == ()
    for prep_id in (
        "bace_b7_prep_gnn_before",
        "bace_b7_prep_molclr_parent",
        "bace_b7_prep_shard_manifests",
        "bace_b7_prep_output_preflight",
    ):
        assert manifest.by_id[prep_id].depends_on == ("bace_b6_ppo_smoke",)
        assert prep_id not in manifest.by_id["bace_b7_ppo_full"].depends_on
        assert (
            manifest.by_id[prep_id].priority
            > manifest.by_id["bace_b7_ppo_full"].priority
        )

    expected_adoptions = {
        "mut_recovery": "20260822T025620Z-mut-lineage-v3-6ddd743",
        "aids_recovery": "20260822T020238Z-aids-lineage-v2-6ddd743",
        "bace_policy_provenance_audit": (
            "20260822T030124Z-bace-policy-audit-1c889b9"
        ),
        "bace_clean_initializer": "20260822T030604Z-bace-clean-init-1c889b9",
        "bace_gnn_ppo_adapter_canary": (
            "20260822T033440Z-bace-ppo-canary-a625841"
        ),
    }
    assert {
        task.task_id: task.adopt_existing_run_id
        for task in manifest.tasks
        if task.adopt_existing_run_id is not None
    } == expected_adoptions

    controller_scoped_prefixes = (
        "/outputs/autodl/four_gpu_recovery/",
        "/cache/bace/frozen_gnn_downstream/autodl-four-gpu-recovery-",
    )
    for task in manifest.tasks:
        strings = (
            *(task.command or ()),
            task.expected_output or "",
            *task.environment.values(),
        )
        for value in strings:
            if any(prefix in value for prefix in controller_scoped_prefixes):
                assert manifest.controller_id in value, (task.task_id, value)


def test_template_matches_commit_b_wrapper_environment_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_controller_manifest(
        root / "configs/autodl/four_gpu_recovery.template.json"
    )
    audit = manifest.by_id["bace_policy_provenance_audit"]
    initializer = manifest.by_id["bace_clean_initializer"]
    canary = manifest.by_id["bace_gnn_ppo_adapter_canary"]
    b6 = manifest.by_id["bace_b6_ppo_smoke"]
    b7 = manifest.by_id["bace_b7_ppo_full"]
    assert audit.runner_dataset == "bace-gnn-clean-audit"
    assert initializer.runner_dataset == "bace-gnn-clean-init"
    assert audit.runner_dataset != "bace"
    assert initializer.runner_dataset != "bace"
    assert "BACE_POLICY_INITIALIZER" not in audit.environment
    assert audit.external_bace_stage == "B5_ORACLE_SMOKE"
    assert "BACE_POLICY_PROVENANCE_MANIFEST" not in audit.environment
    assert "BACE_POLICY_INITIALIZER" not in initializer.environment
    assert "BACE_POLICY_PROVENANCE_MANIFEST" not in initializer.environment
    assert initializer.environment["BACE_POLICY_AUDIT_SELECTION"].endswith(
        "{dep_bace_policy_provenance_audit_output}/initializer_selection.json"
    )
    assert audit.required_absolute_output_files == (
        audit.environment["BACE_POLICY_AUDIT_CSV"],
    )
    assert "/outputs/autodl/audits/" in audit.environment[
        "BACE_POLICY_AUDIT_CSV"
    ].replace("{artifact_root}", "/runtime/outputs")
    assert b6.oom_retry.batch_env == "BACE_PPO_BATCH_SIZE"
    assert b7.oom_retry.batch_env == "BACE_PPO_BATCH_SIZE"
    assert b7.input_manifest == (
        "{dep_bace_b6_ppo_smoke_output}/ppo_smoke_manifest.json"
    )
    assert b7.environment["BACE_B6_V2_MANIFEST"] == b7.input_manifest
    assert "attempt-0" not in json.dumps(
        {
            "initializer_input": initializer.input_manifest,
            "initializer_environment": initializer.environment,
            "canary_input": canary.input_manifest,
            "canary_environment": canary.environment,
            "b6_input": b6.input_manifest,
            "b6_environment": b6.environment,
            "input_manifest": b7.input_manifest,
            "environment": b7.environment,
        },
        sort_keys=True,
    )
    assert canary.stage == "BACE_GNN_PPO_ADAPTER_CANARY"
    assert all(
        task.environment["PYTHONDONTWRITEBYTECODE"] == "1"
        for task in manifest.tasks
    )
    assert "canary_connected_deletion_preflight.json" in canary.required_output_files
    for required in (
        "resume_provenance.json",
        "stable_ppo_resume_manifest.json",
        "stable_ppo_training_state.pt",
        "checkpoint-50/stable_ppo_resume_manifest.json",
        "checkpoint-300/stable_ppo_training_state.pt",
    ):
        assert required in b7.required_output_files


def test_template_injects_exact_commit_d_output_and_marker_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (root / "configs/autodl/four_gpu_recovery.template.json").read_text(
            encoding="utf-8"
        )
    )
    contract = json.loads(
        (root / "configs/autodl/bace_frozen_gnn_downstream_tasks.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = {task["id"]: task for task in template["tasks"]}
    output_contracts = {
        "bace_b7_prep_gnn_before": contract["b7_parallel_prep"][
            "required_outputs_by_action"
        ]["CALIBRATION_GNN_BEFORE_CACHE"],
        "bace_b7_prep_molclr_parent": contract["b7_parallel_prep"][
            "required_outputs_by_action"
        ]["CALIBRATION_MOLCLR_PARENT_CACHE"],
        "bace_b7_prep_shard_manifests": contract["b7_parallel_prep"][
            "required_outputs_by_action"
        ]["FIXED_SHARD_MANIFESTS"],
        "bace_b7_prep_output_preflight": contract["b7_parallel_prep"][
            "required_outputs_by_action"
        ]["OUTPUT_PREFLIGHT"],
        "bace_b8_pool_base": contract["B8_POOL_BASE"]["required_outputs"],
        "bace_b9_pool_hightemp": contract["B9_POOL_HIGHTEMP"]["required_outputs"],
        "bace_b10_pool_merged": contract["B10_POOL_MERGED"]["required_outputs"],
        "bace_b11_verification_shards": contract[
            "B11_CROSS_PARENT_VERIFIED"
        ]["required_shard_outputs"],
        "bace_b11_cross_parent_verified": contract[
            "B11_CROSS_PARENT_VERIFIED"
        ]["required_merge_outputs"],
        "bace_b12_selector": contract["B12_SELECTOR"]["required_outputs"],
        "bace_b13_test_parent_manifest": contract["B13_TEST_PARENT_MANIFEST"][
            "required_outputs"
        ],
        "bace_b13_verification_shards": contract["B13_FINAL_EVAL"][
            "required_shard_outputs"
        ],
        "bace_b13_final_eval": contract["B13_FINAL_EVAL"][
            "required_merge_outputs"
        ],
        "bace_b14_frozen": contract["B14_FROZEN"]["required_outputs"],
    }
    for task_id, expected in output_contracts.items():
        task = tasks[task_id]
        assert task["required_output_files"] == expected
        assert task["command"][:2] == [
            "bash",
            "{project_root}/scripts/autodl/run_bace_frozen_gnn_downstream.sh",
        ]
        assert "{task_output}" in task["command"]
        assert not str(task.get("blocked_reason", "")).startswith(
            "INTEGRATION_REQUIRED"
        )

    marker_contracts = {
        "bace_b8_pool_base": contract["B8_POOL_BASE"]["required_log_marker"],
        "bace_b9_pool_hightemp": contract["B9_POOL_HIGHTEMP"][
            "required_log_marker"
        ],
        "bace_b10_pool_merged": contract["B10_POOL_MERGED"][
            "required_log_marker"
        ],
        "bace_b11_verification_shards": contract[
            "B11_CROSS_PARENT_VERIFIED"
        ]["required_shard_log_marker"],
        "bace_b11_cross_parent_verified": contract[
            "B11_CROSS_PARENT_VERIFIED"
        ]["required_merge_log_marker"],
        "bace_b12_selector": contract["B12_SELECTOR"]["required_log_marker"],
        "bace_b13_test_parent_manifest": contract["B13_TEST_PARENT_MANIFEST"][
            "required_log_marker"
        ],
        "bace_b13_verification_shards": contract["B13_FINAL_EVAL"][
            "required_shard_log_marker"
        ],
        "bace_b13_final_eval": contract["B13_FINAL_EVAL"][
            "required_merge_log_marker"
        ],
        "bace_b14_frozen": contract["B14_FROZEN"]["required_log_marker"],
    }
    for task_id, expected in marker_contracts.items():
        assert tasks[task_id]["required_log_marker"] == expected


def test_absolute_audit_evidence_is_physical_nonempty_and_persistent(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    evidence = allowed / "autodl" / "audits" / "audit.csv"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("path,sha256\nmodel,abc\n", encoding="utf-8")
    assert verify_required_absolute_outputs(
        [evidence], allowed_root=allowed
    ) == []
    outside = tmp_path / "outside.csv"
    outside.write_text("not allowed\n", encoding="utf-8")
    assert "outside" in verify_required_absolute_outputs(
        [outside], allowed_root=allowed
    )[0]


def test_user_runtime_registry_is_append_only_and_status_updates_every_tick(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    task = _task("registry", dataset="mutagenicity")
    root = layout.control_root / "four_gpu_recovery" / "controller"
    task_root = root / "tasks" / task.task_id
    task_root.mkdir(parents=True)
    (task_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "controller.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = ControllerManifest(
        path=manifest_path,
        sha256="2" * 64,
        controller_id="controller",
        tasks=(task,),
        runtime={"max_gpus": 4},
        resource_gates={},
    )
    state = _state(task, "READY")
    states = {task.task_id: state}
    publish_user_registry(layout, root, manifest, states)
    publish_user_registry(layout, root, manifest, states)
    registry = layout.artifacts_dir / "autodl" / "experiment_registry"
    run_rows = (registry / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    update_rows = (
        registry / "status_updates.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(run_rows) == 1
    assert len(update_rows) == 2
    row = json.loads(run_rows[0])
    required_fields = {
        "run_id",
        "dataset",
        "stage",
        "state",
        "pid",
        "tmux_session",
        "tmux",
        "gpu_index",
        "gpu_uuid",
        "git_commit",
        "input",
        "input_root",
        "output",
        "output_root",
        "checkpoint",
        "checkpoint_hash",
        "config",
        "config_hash",
        "start_time",
        "end_time",
        "exit_code",
        "retry",
        "retry_count",
        "dependencies",
        "dependency_ids",
    }
    assert required_fields.issubset(row)
    assert required_fields.issubset(json.loads(update_rows[0]))
    log = layout.runtime_root / "docs" / "AUTODL_FOUR_GPU_EXPERIMENT_LOG.md"
    assert "AutoDL four-GPU experiment log" in log.read_text(encoding="utf-8")


def test_autodl_launcher_has_tmux_nohup_fallback_and_never_calls_slurm() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/autodl/launch_four_gpu_recovery.sh").read_text(
        encoding="utf-8"
    )
    assert "command -v tmux" in text
    assert "nohup" in text
    assert "sbatch" not in text
    assert "AUTODL_PYTHON" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text


def test_read_only_status_exposes_operational_queue_fields(tmp_path: Path) -> None:
    root = tmp_path / "controller"
    task_root = root / "tasks" / "queued"
    task_root.mkdir(parents=True)
    (task_root / "state.json").write_text(
        json.dumps(
            {
                "task_id": "queued",
                "dataset": "bace",
                "stage": "B7_PPO_FULL",
                "state": "READY",
                "instances": {
                    "main": {
                        "state": "NOT_STARTED",
                        "attempt": 0,
                        "worker_pid": 101,
                        "child_pid": 102,
                        "tmux_session": "cf-run",
                        "gpu_index": 2,
                        "gpu_uuid": "GPU-two",
                        "started_at": "2026-08-22T00:00:00+00:00",
                        "heartbeat_at": "2026-08-22T00:01:00+00:00",
                        "expected_output": "/persistent/output/b7",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = _load_tasks(root, priorities={"queued": 5})
    instance = rows[0]["instances"][0]
    assert {
        "worker_pid",
        "child_pid",
        "tmux_session",
        "duration_seconds",
        "output_root",
        "heartbeat_at",
    }.issubset(instance)
    payload = {
        "refreshed_at": "now",
        "controller": {
            "controller_id": "controller",
            "state": "RUNNING",
            "pid": 1,
            "heartbeat_age_seconds": 1.0,
        },
        "gpus": [
            {
                "gpu_index": 2,
                "gpu_uuid": "GPU-two",
                "utilization_gpu_percent": 0,
                "memory_used_mb": 4,
                "memory_free_mb": 81916,
                "memory_total_mb": 81920,
                "lock_state": "ABSENT",
                "audit": "AVAILABLE",
            }
        ],
        "gpu_error": None,
        "queue": rows,
        "next_queued": rows[0],
        "mut": {"state": "PASS", "task_id": "mut"},
        "aids": {"state": "RUNNING", "task_id": "aids"},
        "bace_b6_b14": [],
    }
    table = render_table(payload)
    for value in (
        "USED/FREE/TOTAL_MIB",
        "NEXT QUEUED",
        "pid=101/102",
        "tmux=cf-run",
        "duration=",
        "heartbeat=",
        "output=/persistent/output/b7",
    ):
        assert value in table
