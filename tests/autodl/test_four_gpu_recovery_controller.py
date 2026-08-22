from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    ControllerManifest,
    FixedShardSpec,
    OOMRetry,
    TaskSpec,
    _launch_instance,
    _dependency_output_context,
    _prepare_shards,
    _reconcile_instance,
    audit_gpu_locks,
    bind_adopted_runs,
    classify_failure,
    dependency_order,
    load_controller_manifest,
    materialize_fixed_parent_shards,
    oom_retry_allowed,
    publish_user_registry,
    scheduler_candidates,
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


def test_launch_delegates_to_exp_run_with_frozen_python_and_four_gpu_limit(
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
    config = project / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    task = _task("launch")
    task = TaskSpec(
        **{
            **task.__dict__,
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
    (task_root / "manifest.json").write_text("{}\n", encoding="utf-8")
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
    assert command[command.index("--required-absolute-output-file") + 1].endswith(
        "/outputs/autodl/audits/audit.csv"
    )
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    payload_start = command.index("--") + 1
    assert command[payload_start] == str(interpreter)
    assert state["instances"]["main"]["state"] == "STARTING"


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
    assert manifest.by_id["bace_b11_cross_parent_verified"].shards is not None
    assert manifest.by_id["bace_b6_ppo_smoke"].stage == "B6_PPO_SMOKE_V2"
    assert manifest.by_id["bace_b6_ppo_smoke"].depends_on == (
        "bace_gnn_ppo_adapter_canary",
    )
    assert manifest.by_id["bace_gnn_ppo_adapter_canary"].depends_on == (
        "bace_clean_initializer",
    )
    assert manifest.by_id["bace_b9_pool_hightemp"].depends_on == (
        "bace_b7_ppo_full",
    )
    assert manifest.by_id["bace_b13_final_eval"].shards is not None
    assert manifest.by_id["bace_b13_final_eval"].shards.count == 4
    assert "{shard_id}" in manifest.by_id["bace_b13_final_eval"].runner_stage
    assert manifest.by_id["bace_b14_frozen"].manifest_only is True
    assert manifest.by_id["bace_b14_frozen"].data_splits == ()
    for prep_id in (
        "bace_b7_prep_calibration_gnn_before",
        "bace_b7_prep_molclr_embeddings",
        "bace_b7_prep_shard_manifests",
        "bace_b7_prep_output_preflight",
    ):
        assert manifest.by_id[prep_id].depends_on == ("bace_b6_ppo_smoke",)
        assert prep_id not in manifest.by_id["bace_b7_ppo_full"].depends_on
        assert (
            manifest.by_id[prep_id].priority
            > manifest.by_id["bace_b7_ppo_full"].priority
        )


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
    assert "BACE_POLICY_INITIALIZER" not in audit.environment
    assert audit.external_bace_stage == "B5_ORACLE_SMOKE"
    assert "BACE_POLICY_PROVENANCE_MANIFEST" not in audit.environment
    assert "BACE_POLICY_INITIALIZER" not in initializer.environment
    assert "BACE_POLICY_PROVENANCE_MANIFEST" not in initializer.environment
    assert initializer.environment["BACE_POLICY_AUDIT_SELECTION"].endswith(
        "/initializer_selection.json"
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
    assert "b6-v2/attempt-0" not in json.dumps(
        {
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
