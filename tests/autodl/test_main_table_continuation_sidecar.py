from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from src.utils.autodl_main_table_continuation_sidecar import (
    SPEC_SCHEMA,
    TERMINAL_SCHEMA,
    ContinuationSidecar,
    ContinuationSidecarError,
    ProcessSnapshot,
    load_continuation_spec,
    run_child_with_terminal_receipt,
)
from src.utils.autodl_runtime import GPUObservation, GPUProcess, atomic_write_json


GPU0 = "GPU-00000000-0000-0000-0000-000000000000"
GPU1 = "GPU-11111111-1111-1111-1111-111111111111"


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class FakeLauncher:
    def __init__(self, pids: Iterator[int], snapshots: dict[int, ProcessSnapshot]) -> None:
        self.pids = pids
        self.snapshots = snapshots
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
        pid = next(self.pids)
        self.snapshots[pid] = ProcessSnapshot(
            pid=pid,
            ppid=1,
            start_ticks=pid * 10,
            command=" ".join(argv),
        )
        self.calls.append({"argv": list(argv), **kwargs})
        return FakeProcess(pid)


def _gpu(index: int, *, busy_pid: int | None = None) -> GPUObservation:
    gpu_uuid = GPU0 if index == 0 else GPU1
    processes = (
        ()
        if busy_pid is None
        else (GPUProcess(busy_pid, "python", 1024),)
    )
    return GPUObservation(
        index=index,
        uuid=gpu_uuid,
        name="A800",
        memory_total_mb=81920,
        memory_used_mb=1024 if busy_pid else 0,
        memory_free_mb=80896 if busy_pid else 81920,
        utilization_gpu_percent=1 if busy_pid else 0,
        processes=processes,
    )


def _fixture(
    tmp_path: Path,
    *,
    progress: int | None = None,
    neurosed_ready: bool = False,
) -> tuple[Path, dict[int, ProcessSnapshot]]:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "scripts/autodl").mkdir(parents=True)
    config = project / "configs/hpc.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    entrypoint = project / "scripts/autodl/run_main_table_continuation_sidecar.py"
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    wrapper = project / "scripts/autodl/run_tastemolnet_comrecgc_smoke.sh"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    (project / "scripts/autodl/gpu_lock.py").write_text("# fixture\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    data_root = tmp_path / "data"
    lock_root = runtime / "locks"
    state_root = runtime / "control/continuation"
    checkpoint = runtime / "aids/checkpoint.json"
    progress_path = runtime / "bace/comrec/progress.json"
    if progress is not None:
        progress_path.parent.mkdir(parents=True)
        progress_path.write_text(json.dumps({"step": progress}), encoding="utf-8")
    fixed_t9_environment = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "RUN_GNN_ABLATION": "0",
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_DATA_ROOT": str(data_root),
        "TASTEMOLNET_T2_ADOPTION_ROOT": "/authority/t2",
        "TASTEMOLNET_T2_ADOPTION_GATE_SHA256": "1" * 64,
        "TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256": "2" * 64,
        "TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256": "3" * 64,
        "TASTEMOLNET_T3_OUTPUT_ROOT": "/authority/t3",
        "TASTEMOLNET_T4_OUTPUT_ROOT": "/authority/t4",
        "TASTEMOLNET_TRAIN_CSV": "/authority/train.csv",
        "COMRECGC_OFFICIAL_ROOT": "/authority/comrecgc",
    }
    neurosed: dict[str, object] = {
        "label_manifest": None,
        "trainer_argv": None,
    }
    if neurosed_ready:
        label_manifest = runtime / "labels/ged_label_manifest.json"
        label_manifest.parent.mkdir(parents=True)
        label_manifest.write_text(
            json.dumps(
                {
                    "train_success_count": 5000,
                    "validation_success_count": 1000,
                    "backend": "branch",
                    "state": "PASS",
                    "calibration_loaded": False,
                    "test_loaded": False,
                }
            ),
            encoding="utf-8",
        )
        neurosed = {
            "label_manifest": str(label_manifest),
            "label_assertions": {
                "/train_success_count": 5000,
                "/validation_success_count": 1000,
                "/backend": "branch",
                "/state": "PASS",
                "/calibration_loaded": False,
                "/test_loaded": False,
            },
            "trainer_argv": [
                sys.executable,
                str(project / "scripts/train_fixed_budget_neurosed.py"),
                "--output-dir",
                "{attempt_root}",
            ],
            "fixed_environment": {"RUN_GNN_ABLATION": "0"},
            "science_process_token": "train_fixed_budget_neurosed.py",
            "attempt_parent": str(runtime / "outputs/neurosed"),
            "success_marker_template": "{attempt_root}/best.pt",
        }
    spec = {
        "schema_version": SPEC_SCHEMA,
        "controller_id": "test-continuation-v1",
        "state_root": str(state_root),
        "project_root": str(project),
        "runtime_root": str(runtime),
        "data_root": str(data_root),
        "python": sys.executable,
        "config": str(config),
        "entrypoint": str(entrypoint),
        "poll_seconds": 60,
        "run_gnn_ablation": False,
        "lock_root": str(lock_root),
        "gpus": [
            {"index": 0, "uuid": GPU0},
            {"index": 1, "uuid": GPU1},
        ],
        "aids_exact": {
            "state": "BLOCKED",
            "handover_allowed": False,
            "controller_id": "aids-exact-blocked-v1",
            "controller_pid": 100,
            "science_pid": 101,
            "checkpoint": str(checkpoint),
            "blocker": "checkpoint phase is BLOCKED; reload/progress gate is false",
        },
        "observers": {
            "bace_gcf": {
                "pid": 200,
                "start_ticks": 2000,
                "final_markers": [str(runtime / "bace/gcf/PASS.json")],
            },
            "bace_globalgce": {"pid": 201, "start_ticks": 2010},
            "bace_comrecgc": {
                "pid": 202,
                "start_ticks": 2020,
                "trigger_step": 17500,
                "progress_json": str(progress_path),
                "progress_pointer": "/step",
            },
        },
        "blocked_taste": {
            task: {
                "state": "BLOCKED_RELEASE",
                "reason": f"{task} release authority is false",
            }
            for task in ("T6", "T7", "T8")
        },
        "t9": {
            "enabled": True,
            "wrapper": str(wrapper),
            "stage_parent": str(runtime / "outputs/T9"),
            "final_parent": str(runtime / "outputs/tastemolnet/comrecgc"),
            "run_id_prefix": "taste-t9-comrecgc-m500",
            "fixed_environment": fixed_t9_environment,
        },
        "neurosed": neurosed,
    }
    spec_path = tmp_path / "continuation.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    snapshots = {
        200: ProcessSnapshot(200, 1, 2000, "bace-gcf"),
        201: ProcessSnapshot(201, 1, 2010, "bace-globalgce"),
        202: ProcessSnapshot(202, 1, 2020, "bace-comrecgc"),
    }
    return spec_path, snapshots


def _sidecar(
    spec_path: Path,
    snapshots: dict[int, ProcessSnapshot],
    launcher: FakeLauncher,
    *,
    gpu0_busy: int | None = 300,
    gpu1_busy: int | None = 301,
    uuid_values: Iterator[uuid.UUID] | None = None,
    descendants: dict[int, tuple[ProcessSnapshot, ...]] | None = None,
) -> ContinuationSidecar:
    return ContinuationSidecar(
        spec_path,
        gpu_reader=lambda: [_gpu(0, busy_pid=gpu0_busy), _gpu(1, busy_pid=gpu1_busy)],
        lock_reader=lambda _root, _uuid: True,
        process_reader=lambda pid: snapshots.get(pid),
        descendants_reader=lambda pid: (descendants or {}).get(pid, ()),
        launcher=launcher,
        command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="{\"status\":\"PASS\"}\n", stderr=""
        ),
        uuid_factory=(lambda: next(uuid_values)) if uuid_values is not None else uuid.uuid4,
    )


def test_spec_refuses_gnn_ablation_and_wrong_aids_handover(tmp_path: Path) -> None:
    spec_path, _ = _fixture(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["run_gnn_ablation"] = True
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContinuationSidecarError, match="RUN_GNN_ABLATION"):
        load_continuation_spec(spec_path)
    payload["run_gnn_ablation"] = False
    payload["aids_exact"]["handover_allowed"] = True
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContinuationSidecarError, match="handover forbidden"):
        load_continuation_spec(spec_path)


def test_fixed_queue_is_durable_and_release_blocked_tasks_are_never_launched(
    tmp_path: Path,
) -> None:
    spec_path, snapshots = _fixture(tmp_path)
    launcher = FakeLauncher(iter(range(700, 710)), snapshots)
    sidecar = _sidecar(spec_path, snapshots, launcher)
    state = sidecar.tick()
    assert launcher.calls == []
    assert state["observations"]["aids_exact"]["state"] == "BLOCKED"
    assert state["observations"]["aids_exact"]["handover_allowed"] is False
    for task in ("T6", "T7", "T8"):
        assert state["tasks"][task]["state"] == "BLOCKED_RELEASE"
        assert state["tasks"][task]["launch_attempts"] == 0
    assert state["tasks"]["NEUROSED"]["state"] == "WAITING_INPUT"
    queue = json.loads(sidecar.queue_path.read_text(encoding="utf-8"))
    assert queue["fixed_policy"]["run_gnn_ablation"] is False
    registration = json.loads(
        (sidecar.state_root / "bace_comrecgc_convergence_registration.json").read_text(
            encoding="utf-8"
        )
    )
    assert registration["status"] == "DURABLE_PENDING"
    assert registration["trigger_step"] == 17500
    assert registration["sidecar_may_stop_worker"] is False


def test_bace_step_17500_becomes_external_check_only(tmp_path: Path) -> None:
    spec_path, snapshots = _fixture(tmp_path, progress=17500)
    launcher = FakeLauncher(iter(range(700, 710)), snapshots)
    sidecar = _sidecar(spec_path, snapshots, launcher)
    state = sidecar.tick()
    observation = state["observations"]["bace_comrecgc"]
    assert observation["state"] == "READY_FOR_EXTERNAL_CONVERGENCE_CHECK"
    assert observation["external_convergence_audit_required"] is True
    assert observation["sidecar_may_stop_worker"] is False
    assert launcher.calls == []


def test_optional_convergence_hook_persists_pass_without_handover(tmp_path: Path) -> None:
    spec_path, snapshots = _fixture(tmp_path, progress=17500)
    launcher = FakeLauncher(iter(range(700, 710)), snapshots)
    calls: list[dict[str, object]] = []

    def auditor(value: dict[str, object]) -> dict[str, object]:
        calls.append(dict(value))
        return {"status": "PASS", "audit_root": "/fresh/read-only/audit"}

    sidecar = ContinuationSidecar(
        spec_path,
        gpu_reader=lambda: [_gpu(0, busy_pid=300), _gpu(1, busy_pid=301)],
        lock_reader=lambda _root, _uuid: True,
        process_reader=lambda pid: snapshots.get(pid),
        launcher=launcher,
        convergence_auditor=auditor,
    )
    state = sidecar.tick()
    assert len(calls) == 1
    assert calls[0]["latest_observed_progress"] == 17500
    assert calls[0]["sidecar_may_stop_worker"] is False
    assert (
        state["observations"]["bace_comrecgc"]["state"]
        == "AUDIT_PASS_AWAITING_SEPARATE_HANDOVER"
    )
    sidecar.tick()
    assert len(calls) == 1
    hook = json.loads(
        (
            sidecar.state_root
            / "bace_comrecgc_convergence_audits/step-17500.json"
        ).read_text(encoding="utf-8")
    )
    assert hook["handover_implemented_by_sidecar"] is False


def test_t9_rc75_abandons_every_identity_and_next_launch_is_uuid_fresh(
    tmp_path: Path,
) -> None:
    spec_path, snapshots = _fixture(tmp_path)
    values = iter(
        [
            uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        ]
    )
    launcher = FakeLauncher(iter((700, 701)), snapshots)
    sidecar = _sidecar(
        spec_path,
        snapshots,
        launcher,
        gpu1_busy=None,
        uuid_values=values,
    )
    first = sidecar.tick()["tasks"]["T9"]
    first_uuid = first["active_attempt_uuid"]
    assert first_uuid == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    first_attempt = sidecar.state_root / "attempts/t9" / first_uuid
    atomic_write_json(
        first_attempt / "terminal.json",
        {
            "schema_version": TERMINAL_SCHEMA,
            "task": "T9",
            "returncode": 75,
        },
    )
    second = sidecar.tick()["tasks"]["T9"]
    second_uuid = second["active_attempt_uuid"]
    assert second_uuid == "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    assert second_uuid != first_uuid
    abandoned = json.loads((first_attempt / "attempt.json").read_text(encoding="utf-8"))
    assert abandoned["state"] == "ABANDONED_PREFLIGHT_RC75"
    assert abandoned["reuse_allowed"] is False
    assert first_uuid in abandoned["stage_root"]
    assert first_uuid in abandoned["final_root"]
    assert first_uuid in abandoned["run_id"]
    assert second_uuid in second["stage_root"]
    assert second_uuid in second["final_root"]
    assert second_uuid in second["run_id"]
    assert len(launcher.calls) == 2


def test_t9_records_real_science_descendant_and_terminal_validation(
    tmp_path: Path,
) -> None:
    spec_path, snapshots = _fixture(tmp_path)
    value = uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    launcher = FakeLauncher(iter((700,)), snapshots)
    science = ProcessSnapshot(
        711,
        700,
        7110,
        "python tastemolnet_t9_managed_runner_v2.py --stage-root fresh",
    )
    sidecar = _sidecar(
        spec_path,
        snapshots,
        launcher,
        gpu1_busy=None,
        uuid_values=iter((value,)),
        descendants={700: (science,)},
    )
    first = sidecar.tick()["tasks"]["T9"]
    attempt_uuid = first["active_attempt_uuid"]
    running = sidecar.tick()["tasks"]["T9"]
    assert running["state"] == "RUNNING"
    assert running["science_child_pid"] == 711
    attempt_root = sidecar.state_root / "attempts/t9" / attempt_uuid
    atomic_write_json(
        attempt_root / "terminal.json",
        {"schema_version": TERMINAL_SCHEMA, "task": "T9", "returncode": 0},
    )
    terminal = sidecar.tick()["tasks"]["T9"]
    assert terminal["state"] == "PASS"
    assert terminal["active_attempt_uuid"] is None


def test_neurosed_requires_manifest_and_argv_then_uses_first_free_gpu(
    tmp_path: Path,
) -> None:
    spec_path, snapshots = _fixture(tmp_path, neurosed_ready=True)
    launcher = FakeLauncher(iter((700,)), snapshots)
    sidecar = _sidecar(
        spec_path,
        snapshots,
        launcher,
        gpu0_busy=None,
        gpu1_busy=301,
        uuid_values=iter((uuid.UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),)),
    )
    task = sidecar.tick()["tasks"]["NEUROSED"]
    assert task["state"] == "STARTING"
    assert task["gpu_index"] == 0
    assert task["active_attempt_uuid"] == "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    launched = launcher.calls[0]
    joined = " ".join(launched["argv"])
    assert "gpu_lock.py" in joined
    assert "--gpu-index 0" in joined
    assert "train_fixed_budget_neurosed.py" in joined
    assert task["active_attempt_uuid"] in joined
    assert launched["env"]["RUN_GNN_ABLATION"] == "0"


def test_child_terminal_receipt_persists_exact_rc75(tmp_path: Path) -> None:
    receipt = tmp_path / "terminal.json"
    rc = run_child_with_terminal_receipt(
        task="T9",
        terminal_receipt=receipt,
        command=[sys.executable, "-c", "raise SystemExit(75)"],
    )
    assert rc == 75
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == TERMINAL_SCHEMA
    assert payload["returncode"] == 75


def test_sidecar_source_has_no_process_termination_or_matrix_writer() -> None:
    source = Path(
        "src/utils/autodl_main_table_continuation_sidecar.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("os.kill(", "pkill", "killall", "SIGKILL", "SIGTERM"):
        assert forbidden not in source
    assert "write_registry_outputs" not in source
    assert "matrix_status.csv" not in source
