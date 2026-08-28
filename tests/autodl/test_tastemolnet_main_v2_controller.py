from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from scripts.autodl import run_taste_main_v2 as controller_cli
from scripts.autodl import tastemolnet_t4_managed_runner_v2 as t4_runner
from src.utils import autodl_tastemolnet_main_v2 as main_v2
from src.utils.process_identity_v2 import ProcessSnapshotV2, capture_process_snapshot


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU_INVENTORY = {
    0: "GPU-00000000-0000-0000-0000-000000000000",
    1: GPU_UUID,
    2: "GPU-aaaaaaaa-2222-2222-2222-222222222222",
    3: "GPU-bbbbbbbb-3333-3333-3333-333333333333",
}
ATTEMPT_ID = "00000000-0000-4000-8000-000000000010"
GENERATION_TOKEN = "00000000-0000-4000-8000-000000000011"
COMMIT = "a" * 40
TREE = "b" * 40


def _policy(root: Path) -> dict[str, object]:
    runtime_root = root / "counterfactual-subgraph-runtime"
    runtime_root.mkdir(exist_ok=True)
    control_root = runtime_root / "control"
    control_root.mkdir(exist_ok=True)
    (runtime_root / "locks").mkdir(exist_ok=True)
    return {
        "run_tastemolnet": True,
        "taste_research_compute_allowed": True,
        "taste_paper_results_allowed": True,
        "taste_data_redistribution_allowed": False,
        "primary_taste_source_label": 1,
        "minimum_free_after_reservations_gib": 100,
        "scheduler_poll_seconds": 60,
        "max_physical_gpus": 4,
        "max_concurrent_taste_full": 2,
        "gnn_ablation_enabled": False,
        "persistent_data_root": str(root),
        "persistent_storage_root": str(runtime_root),
        "persistent_control_root": str(control_root),
        "canonical_gpu_lock_root": str(runtime_root / "locks"),
        "persistent_free_bytes_at_launch": 101 * 1024**3,
        "persistent_free_gib_at_launch": 101.0,
        "observed_at": "2026-08-28T00:00:00Z",
    }


def _external_launcher(
    *,
    controller_id: str,
    controller_uuid: str,
    controller_snapshot: ProcessSnapshotV2,
    policy: dict[str, object],
):
    _controllers, launchers = main_v2.ensure_controller_namespace_parents(
        policy["persistent_control_root"]
    )
    launcher_root = launchers / controller_uuid
    launcher_snapshot = ProcessSnapshotV2(
        pid=controller_snapshot.ppid,
        ppid=1,
        pid_start_ticks=controller_snapshot.pid_start_ticks + 1,
        boot_id=controller_snapshot.boot_id,
        executable_realpath=controller_snapshot.executable_realpath,
        command=controller_snapshot.command,
        command_hash=controller_snapshot.command_hash,
        cwd_realpath=controller_snapshot.cwd_realpath,
        cgroup_path=controller_snapshot.cgroup_path,
    )
    main_v2._create_fresh_namespace(
        launcher_root, children=(main_v2.PUBLICATION_STAGING_DIRECTORY,)
    )
    payload = {
        "schema_version": main_v2.LAUNCHER_RECEIPT_SCHEMA,
        "managed_taste_release_version": main_v2.MANAGED_TASTE_RELEASE_VERSION,
        "controller_id": controller_id,
        "controller_uuid": controller_uuid,
        "launcher_generation_token": str(uuid.uuid4()),
        "launcher_process": launcher_snapshot.to_dict(),
        "controller_process": controller_snapshot.to_dict(),
        "git_commit": COMMIT,
        "git_tree": TREE,
        "project_root": str(Path.cwd().resolve()),
        "policy_facts": policy,
        "policy_facts_sha256": main_v2._sha256(main_v2._json_bytes(policy)),
        "state": "CONTROLLER_SPAWNED",
        "created_at": "2026-08-28T00:00:00Z",
        "created_at_ns": 1,
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    path = launcher_root / main_v2.LAUNCHER_RECEIPT_NAME
    data = main_v2._json_bytes(payload)
    main_v2._publish_immutable(
        path,
        data,
        staging_root=launcher_root / main_v2.PUBLICATION_STAGING_DIRECTORY,
    )
    return SimpleNamespace(
        receipt_path=path,
        receipt_sha256=main_v2._sha256(data),
        payload=payload,
    )


def _authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    training_child: ProcessSnapshotV2 | None = None,
    snapshot_reader=capture_process_snapshot,
):
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    data_root = tmp_path.resolve() / "data"
    data_root.mkdir()
    controller_uuid = str(uuid.uuid4())
    controller_id = f"taste-main-v2-{controller_uuid}"
    snapshot = capture_process_snapshot(os.getpid())
    policy = _policy(data_root)
    controllers, _launches = main_v2.ensure_controller_namespace_parents(
        policy["persistent_control_root"]
    )
    launcher = _external_launcher(
        controller_id=controller_id,
        controller_uuid=controller_uuid,
        controller_snapshot=snapshot,
        policy=policy,
    )
    created = main_v2.create_controller_receipt(
        controller_root=controllers / controller_uuid,
        project_root=Path.cwd(),
        controller_id=controller_id,
        controller_uuid=controller_uuid,
        launcher_receipt_path=launcher.receipt_path,
        expected_launcher_receipt_sha256=launcher.receipt_sha256,
        git_identity=(COMMIT, TREE),
        process_snapshot=snapshot,
        policy_facts=policy,
    )
    lease = main_v2.create_gpu_lease_request(
        controller_receipt_path=created.receipt_path,
        task_id="T4_ORACLE_SMOKE",
        physical_gpu_index=1,
        physical_gpu_uuid=GPU_UUID,
    )
    main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        training_child=training_child,
    )
    heartbeat = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=1,
        previous_heartbeat_sha256=None,
        gpu_inventory=GPU_INVENTORY,
        snapshot_reader=snapshot_reader,
    )
    return created, lease, heartbeat, controller_id


def _hold(created, lease, heartbeat, controller_id, **kwargs):
    return main_v2.hold_taste_main_v2_controller_authority(
        created.receipt_path,
        heartbeat.path,
        controller_id,
        COMMIT,
        TREE,
        35,
        expected_receipt_sha256=created.receipt_sha256,
        expected_launcher_receipt_path=created.payload["launcher_receipt_path"],
        expected_launcher_receipt_sha256=created.payload["launcher_receipt_sha256"],
        expected_heartbeat_sha256=heartbeat.sha256,
        expected_task_id="T4_ORACLE_SMOKE",
        expected_gpu_index=1,
        expected_gpu_uuid=GPU_UUID,
        expected_lease_uuid=lease.lease_uuid,
        expected_lease_sha256=lease.sha256,
        expected_attempt_id=ATTEMPT_ID,
        expected_generation_token=GENERATION_TOKEN,
        expected_activation_phase="WORKER_ACTIVE",
        **kwargs,
    )


def test_receipt_lease_and_heartbeat_are_immutable_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, controller_id = _authority(tmp_path, monkeypatch)
    second = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256=first.sha256,
        gpu_inventory=GPU_INVENTORY,
    )
    assert first.path.exists() and second.path.exists()
    assert first.path != second.path
    assert os.stat(created.receipt_path).st_nlink == 1
    assert os.stat(first.path).st_nlink == 1
    assert json.loads(second.path.read_text())["previous_heartbeat_sha256"] == first.sha256
    with _hold(created, lease, first, controller_id) as held:
        assert held.evidence["sequence"] == 2
        assert held.evidence["anchor_heartbeat_sha256"] == first.sha256
        evidence = held.revalidate()
        assert evidence["sequence"] == 2
        assert evidence["heartbeat_sha256"] == second.sha256
        assert evidence["lease_sha256"] == lease.sha256


def test_status_is_strictly_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, _lease, _heartbeat, _controller_id = _authority(tmp_path, monkeypatch)

    def inventory() -> list[tuple[str, int, int]]:
        return sorted(
            (str(path.relative_to(created.controller_root)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in created.controller_root.rglob("*")
            if path.is_file()
        )

    before = inventory()
    status = main_v2.controller_status(controller_root=created.controller_root)
    after = inventory()
    assert status["status"] == "RUNNING"
    assert status["read_only"] is True
    assert before == after


def test_status_reports_missing_authority_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path.resolve() / "missing-controller"
    result = main_v2.controller_status(controller_root=missing)
    assert result["status"] == "QUARANTINED"
    assert result["read_only"] is True
    assert not missing.exists()


def test_held_receipt_detects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    with _hold(created, lease, heartbeat, controller_id) as held:
        with created.receipt_path.open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(main_v2.TasteMainV2AuthorityError, match="changed while held"):
            held.revalidate()


def test_symlink_heartbeat_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    alias = heartbeat.path.parent / "00000000000000000002-00000000-0000-4000-8000-000000000002.json"
    alias.symlink_to(heartbeat.path)
    with pytest.raises(main_v2.TasteMainV2AuthorityError):
        main_v2.hold_taste_main_v2_controller_authority(
            created.receipt_path,
            alias,
            controller_id,
            COMMIT,
            TREE,
            35,
            expected_receipt_sha256=created.receipt_sha256,
            expected_task_id="T4_ORACLE_SMOKE",
            expected_gpu_index=1,
            expected_gpu_uuid=GPU_UUID,
            expected_lease_uuid=lease.lease_uuid,
            expected_lease_sha256=lease.sha256,
            expected_attempt_id=ATTEMPT_ID,
            expected_generation_token=GENERATION_TOKEN,
        )


def test_stale_heartbeat_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    future = heartbeat.payload["heartbeat_at_ns"] + 36_000_000_000
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="stale"):
        _hold(created, lease, heartbeat, controller_id, now_ns=lambda: future)


def test_live_process_generation_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    original = capture_process_snapshot(os.getpid())
    drifted = ProcessSnapshotV2(
        pid=original.pid,
        ppid=original.ppid,
        pid_start_ticks=original.pid_start_ticks + 1,
        boot_id=original.boot_id,
        executable_realpath=original.executable_realpath,
        command=original.command,
        command_hash=original.command_hash,
        cwd_realpath=original.cwd_realpath,
        cgroup_path=original.cgroup_path,
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="generation"):
        _hold(
            created,
            lease,
            heartbeat,
            controller_id,
            snapshot_reader=lambda _pid: drifted,
        )


def test_expected_active_worker_generation_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    worker = capture_process_snapshot(os.getpid())
    drifted = ProcessSnapshotV2(
        pid=worker.pid,
        ppid=worker.ppid,
        pid_start_ticks=worker.pid_start_ticks + 1,
        boot_id=worker.boot_id,
        executable_realpath=worker.executable_realpath,
        command=worker.command,
        command_hash=worker.command_hash,
        cwd_realpath=worker.cwd_realpath,
        cgroup_path=worker.cgroup_path,
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="worker generation"):
        _hold(
            created,
            lease,
            heartbeat,
            controller_id,
            expected_worker_process=drifted,
        )


def test_protected_gpu_and_duplicate_gpu_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, _lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="fixed Taste"):
        main_v2.create_gpu_lease_request(
            controller_receipt_path=created.receipt_path,
            task_id="BAD_TASK",
            physical_gpu_index=0,
            physical_gpu_uuid=GPU_UUID,
        )
    main_v2.create_gpu_lease_request(
        controller_receipt_path=created.receipt_path,
        task_id="T4_ORACLE_SMOKE",
        physical_gpu_index=1,
        physical_gpu_uuid=GPU_UUID,
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="duplicate"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            gpu_inventory=GPU_INVENTORY,
        )


def test_active_lease_contends_in_canonical_global_and_coordination_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, _first, _controller_id = _authority(tmp_path, monkeypatch)
    lock_root = Path(created.payload["policy_facts"]["canonical_gpu_lock_root"])
    worker_probe = main_v2.GPUFileLock(
        lock_root,
        gpu_index=1,
        gpu_uuid=GPU_UUID,
    )
    with pytest.raises(main_v2.GPULockError, match="project-locked"):
        worker_probe.acquire()
    coordination_probe = main_v2.GPUFileLock(
        lock_root,
        gpu_index=1,
        gpu_uuid=f"{GPU_UUID}.taste-main-v2-controller",
        owner={"lease_uuid": lease.lease_uuid},
    )
    with pytest.raises(main_v2.GPULockError, match="project-locked"):
        coordination_probe.acquire()


def test_controller_sources_expose_no_process_termination_primitive() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "src/utils/autodl_tastemolnet_main_v2.py",
            "scripts/autodl/run_taste_main_v2.py",
            "scripts/autodl/status_taste_main_v2.py",
            "scripts/autodl/launch_taste_main_v2.sh",
            "scripts/autodl/tastemolnet_t4_managed_runner_v2.py",
            "scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py",
            "scripts/autodl/tastemolnet_t4_oracle_smoke_verifier_v2.py",
        )
    )
    for forbidden in ("os.kill", "SIGTERM", "SIGKILL", "kill -"):
        assert forbidden not in sources
    assert "AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0" in sources


def test_historical_heartbeats_may_be_stale_when_terminal_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, controller_id = _authority(tmp_path, monkeypatch)
    previous = first
    base_ns = first.payload["heartbeat_at_ns"]
    for sequence in range(2, 33):
        previous = main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=sequence,
            previous_heartbeat_sha256=previous.sha256,
            now_ns=base_ns + (sequence - 1) * 10_000_000_000,
            gpu_inventory=GPU_INVENTORY,
        )
    with _hold(
        created,
        lease,
        first,
        controller_id,
        now_ns=lambda: previous.payload["heartbeat_at_ns"],
    ) as held:
        assert held.evidence["sequence"] == 32
        assert held.evidence["heartbeat_chain_length"] == 32
        assert held.evidence["anchor_heartbeat_sha256"] == first.sha256


def test_bad_heartbeat_predecessor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, controller_id = _authority(tmp_path, monkeypatch)
    main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256="f" * 64,
        gpu_inventory=GPU_INVENTORY,
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="predecessor"):
        _hold(created, lease, first, controller_id)


def test_activation_phase_jump_is_rejected_before_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    active = first.payload["active_tasks"][0]
    main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=2,
        previous_activation_sha256=active["activation_sha256"],
        phase="VERIFIER_ACTIVE",
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="phase chain"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            gpu_inventory=GPU_INVENTORY,
        )


def test_activation_bad_predecessor_is_rejected_before_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=2,
        previous_activation_sha256="e" * 64,
        phase="WAITING_VERIFIER",
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="predecessor"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            gpu_inventory=GPU_INVENTORY,
        )


def test_phase_transition_rejects_a_still_live_prior_science_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = capture_process_snapshot(os.getpid())
    child_snapshot = ProcessSnapshotV2(
        pid=worker.pid + 100000,
        ppid=worker.pid,
        pid_start_ticks=worker.pid_start_ticks + 1,
        boot_id=worker.boot_id,
        executable_realpath=worker.executable_realpath,
        command=worker.command,
        command_hash=worker.command_hash,
        cwd_realpath=worker.cwd_realpath,
        cgroup_path=worker.cgroup_path,
    )
    snapshots = {worker.pid: worker, child_snapshot.pid: child_snapshot}

    def reader(pid: int) -> ProcessSnapshotV2:
        return snapshots[pid]

    created, lease, first, _controller_id = _authority(
        tmp_path,
        monkeypatch,
        training_child=child_snapshot,
        snapshot_reader=reader,
    )
    active = first.payload["active_tasks"][0]
    main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=2,
        previous_activation_sha256=active["activation_sha256"],
        phase="WAITING_VERIFIER",
    )
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="remains live"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            gpu_inventory=GPU_INVENTORY,
            snapshot_reader=reader,
        )


def test_controller_heartbeat_rejects_non_controller_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, _lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    original = capture_process_snapshot(os.getpid())
    foreign = ProcessSnapshotV2(
        pid=original.pid + 100000,
        ppid=original.ppid,
        pid_start_ticks=original.pid_start_ticks,
        boot_id=original.boot_id,
        executable_realpath=original.executable_realpath,
        command=original.command,
        command_hash=original.command_hash,
        cwd_realpath=original.cwd_realpath,
        cgroup_path=original.cgroup_path,
    )
    monkeypatch.setattr(main_v2, "_validate_receipt", lambda *_args, **_kwargs: foreign)
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="only the immutable"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            gpu_inventory=GPU_INVENTORY,
        )


def test_private_half_write_is_not_visible_to_chain_scanner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, controller_id = _authority(tmp_path, monkeypatch)
    partial = (
        created.controller_root
        / main_v2.PUBLICATION_STAGING_DIRECTORY
        / "partial-heartbeat.json"
    )
    partial.write_bytes(b'{"sequence":')
    second = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256=first.sha256,
        gpu_inventory=GPU_INVENTORY,
    )
    with _hold(created, lease, first, controller_id) as held:
        assert held.evidence["heartbeat_sha256"] == second.sha256
        assert held.evidence["heartbeat_chain_length"] == 2


def test_launcher_receipt_cannot_self_sign_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    root = tmp_path.resolve() / "launcher-parent"
    root.mkdir()
    controller_uuid = str(uuid.uuid4())
    snapshot = capture_process_snapshot(os.getpid())
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="distinct live parent"):
        main_v2.create_launcher_receipt(
            launcher_root=root / controller_uuid,
            controller_id=f"taste-main-v2-{controller_uuid}",
            controller_uuid=controller_uuid,
            controller_snapshot=snapshot,
            project_root=Path.cwd(),
            git_identity=(COMMIT, TREE),
            policy_facts=_policy(root),
            launcher_snapshot=snapshot,
        )


def test_no_lease_holder_rejects_spurious_gpu_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, _lease, heartbeat, controller_id = _authority(tmp_path, monkeypatch)
    common = {
        "expected_launcher_receipt_path": created.payload["launcher_receipt_path"],
        "expected_launcher_receipt_sha256": created.payload["launcher_receipt_sha256"],
        "expected_receipt_sha256": created.receipt_sha256,
        "expected_heartbeat_sha256": heartbeat.sha256,
    }
    with main_v2.hold_taste_main_v2_controller_authority(
        created.receipt_path,
        heartbeat.path,
        controller_id,
        COMMIT,
        TREE,
        35,
        **common,
    ) as held:
        assert "lease_uuid" not in held.evidence
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="no-lease"):
        main_v2.hold_taste_main_v2_controller_authority(
            created.receipt_path,
            heartbeat.path,
            controller_id,
            COMMIT,
            TREE,
            35,
            expected_gpu_index=1,
            **common,
        )


def test_policy_capture_binds_canonical_runtime_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path.resolve() / "data"
    runtime_root = data_root / "counterfactual-subgraph-runtime"
    data_root.mkdir()
    runtime_root.mkdir()
    environment = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "PRIMARY_TASTE_SOURCE_LABEL": "1",
        "MIN_FREE_AFTER_RESERVATIONS_GB": "100",
        "SCHEDULER_POLL_SECONDS": "60",
        "AUTODL_MAX_GPUS": "4",
        "MAX_CONCURRENT_TASTE_FULL": "2",
        "RUN_GNN_ABLATION": "0",
        "AUTODL_DATA_ROOT": str(data_root),
        "AUTODL_RUNTIME_ROOT": str(runtime_root),
        "AUTODL_CONTROL_ROOT": str(runtime_root / "control"),
    }
    monkeypatch.setattr(
        main_v2.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=101 * 1024**3),
    )
    policy = main_v2.capture_policy_facts(
        persistent_storage_root=runtime_root,
        environment=environment,
    )
    assert policy["canonical_gpu_lock_root"] == str(runtime_root / "locks")
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="canonical"):
        main_v2.capture_policy_facts(
            persistent_storage_root=runtime_root,
            environment={
                **environment,
                "AUTODL_RUNTIME_ROOT": str(tmp_path / "alternate-runtime"),
            },
        )


def test_renewal_is_append_only_and_controller_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, controller_id = _authority(tmp_path, monkeypatch)
    renewal = main_v2.create_gpu_lease_renewal(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        sequence=1,
        previous_renewal_sha256=None,
    )
    main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256=first.sha256,
        gpu_inventory=GPU_INVENTORY,
    )
    with _hold(created, lease, first, controller_id) as held:
        assert held.evidence["renewal_sequence"] == 1
        assert held.evidence["renewal_sha256"] == renewal.sha256
        assert held.evidence["effective_expires_at_ns"] > held.evidence[
            "base_expires_at_ns"
        ]


def test_clean_release_allows_sequential_gpu_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    active = first.payload["active_tasks"][0]
    waiting = main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=2,
        previous_activation_sha256=active["activation_sha256"],
        phase="WAITING_VERIFIER",
    )
    verifying = main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=3,
        previous_activation_sha256=waiting.sha256,
        phase="VERIFIER_ACTIVE",
    )
    releasing = main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        attempt_id=ATTEMPT_ID,
        generation_token=GENERATION_TOKEN,
        activation_sequence=4,
        previous_activation_sha256=verifying.sha256,
        phase="RELEASE_REQUESTED",
    )
    second = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256=first.sha256,
        gpu_inventory=GPU_INVENTORY,
    )
    released = main_v2.release_registered_runner_gpu_lock_after_ack(
        controller_receipt_path=created.receipt_path,
        lease_path=lease.path,
        expected_lease_sha256=lease.sha256,
        release_activation=releasing,
    )
    assert released["state"] == "RELEASED"
    next_attempt = "00000000-0000-4000-8000-000000000020"
    next_generation = "00000000-0000-4000-8000-000000000021"
    next_lease = main_v2.create_gpu_lease_request(
        controller_receipt_path=created.receipt_path,
        task_id="T4_ORACLE_SMOKE",
        physical_gpu_index=1,
        physical_gpu_uuid=GPU_UUID,
    )
    main_v2.create_gpu_lease_activation(
        controller_receipt_path=created.receipt_path,
        lease_path=next_lease.path,
        expected_lease_sha256=next_lease.sha256,
        attempt_id=next_attempt,
        generation_token=next_generation,
    )
    third = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=3,
        previous_heartbeat_sha256=second.sha256,
        gpu_inventory=GPU_INVENTORY,
    )
    assert [task["lease_uuid"] for task in third.payload["active_tasks"]] == [
        next_lease.lease_uuid
    ]


def test_expiry_quarantine_retains_controller_coordination_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created, lease, first, _controller_id = _authority(tmp_path, monkeypatch)
    retained_before = {
        key for key, (_lock, lease_uuid) in main_v2._GPU_LOCKS.items()
        if lease_uuid == lease.lease_uuid
    }
    assert len(retained_before) == 1
    with pytest.raises(main_v2.TasteMainV2AuthorityError, match="expired"):
        main_v2.write_heartbeat_generation(
            controller_receipt_path=created.receipt_path,
            sequence=2,
            previous_heartbeat_sha256=first.sha256,
            now_ns=lease.payload["expires_at_ns"] + 1,
            gpu_inventory=GPU_INVENTORY,
        )
    quarantined = main_v2.write_heartbeat_generation(
        controller_receipt_path=created.receipt_path,
        sequence=2,
        previous_heartbeat_sha256=first.sha256,
        controller_state="QUARANTINED",
    )
    assert quarantined.payload["state"] == "QUARANTINED"
    assert retained_before <= set(main_v2._GPU_LOCKS)


def test_actual_controller_cli_requires_external_registration_and_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    root = tmp_path.resolve()
    data_root = root / "data"
    data_root.mkdir()
    policy = _policy(data_root)
    controller_parent, launcher_parent = main_v2.ensure_controller_namespace_parents(
        policy["persistent_control_root"]
    )
    controller_uuid = str(uuid.uuid4())
    controller_id = f"taste-main-v2-{controller_uuid}"
    controller_root = controller_parent / controller_uuid
    launcher_root = launcher_parent / controller_uuid
    launcher_receipt = launcher_root / main_v2.LAUNCHER_RECEIPT_NAME
    handshake_read, handshake_write = os.pipe()
    registration_read, registration_write = os.pipe()
    command = [
        sys.executable,
        "-I",
        "-B",
        str(
            Path(__file__).resolve().parents[2]
            / "scripts/autodl/run_taste_main_v2.py"
        ),
        "--config",
        str(Path(__file__).resolve().parents[2] / "configs/hpc.yaml"),
        "run",
        "--controller-root",
        str(controller_root),
        "--controller-id",
        controller_id,
        "--controller-uuid",
        controller_uuid,
        "--project-root",
        str(Path.cwd().resolve()),
        "--persistent-storage-root",
        str(policy["persistent_storage_root"]),
        "--expected-git-commit",
        COMMIT,
        "--expected-git-tree",
        TREE,
        "--launcher-receipt",
        str(launcher_receipt),
        "--launcher-handshake-fd",
        str(handshake_read),
        "--launcher-registration-fd",
        str(registration_write),
        "--heartbeat-count",
        "2",
    ]
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env={**os.environ, "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(handshake_read, registration_write),
    )
    os.close(handshake_read)
    os.close(registration_write)
    registration_chunks: list[bytes] = []
    while block := os.read(registration_read, 4096):
        registration_chunks.append(block)
    os.close(registration_read)
    registered = ProcessSnapshotV2.from_mapping(
        json.loads(b"".join(registration_chunks).decode("utf-8"))
    )
    assert registered.pid == process.pid
    assert registered.ppid == os.getpid()
    launcher = main_v2.create_launcher_receipt(
        launcher_root=launcher_root,
        controller_id=controller_id,
        controller_uuid=controller_uuid,
        controller_snapshot=registered,
        project_root=Path.cwd(),
        git_identity=(COMMIT, TREE),
        policy_facts=policy,
    )
    os.write(
        handshake_write,
        json.dumps(
            {
                "launcher_receipt_path": str(launcher.receipt_path),
                "launcher_receipt_sha256": launcher.receipt_sha256,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    os.close(handshake_write)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            anchor = main_v2.latest_heartbeat_path(controller_root)
            break
        except main_v2.TasteMainV2AuthorityError:
            time.sleep(0.05)
    else:
        stdout, stderr = process.communicate(timeout=15)
        pytest.fail(f"controller produced no heartbeat: {stdout!r} {stderr!r}")
    receipt_sha = main_v2.immutable_authority_sha256(
        controller_root / main_v2.CONTROLLER_RECEIPT_NAME,
        label="controller receipt",
    )
    anchor_sha = main_v2.immutable_authority_sha256(
        anchor, label="controller heartbeat"
    )
    with main_v2.hold_taste_main_v2_controller_authority(
        controller_root / main_v2.CONTROLLER_RECEIPT_NAME,
        anchor,
        controller_id,
        COMMIT,
        TREE,
        35,
        expected_launcher_receipt_path=launcher.receipt_path,
        expected_launcher_receipt_sha256=launcher.receipt_sha256,
        expected_receipt_sha256=receipt_sha,
        expected_heartbeat_sha256=anchor_sha,
        snapshot_reader=lambda _pid: registered,
    ) as authority:
        assert authority.evidence["pid"] == process.pid
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, (stdout, stderr)


def test_actual_launch_cli_supervises_and_attests_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise launch dispatch, inherited pipes, child, and ready receipt."""

    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    data_root = tmp_path.resolve() / "data"
    data_root.mkdir()
    policy = _policy(data_root)
    controller_parent, launcher_parent = main_v2.ensure_controller_namespace_parents(
        policy["persistent_control_root"]
    )
    controller_uuid = str(uuid.uuid4())
    controller_id = f"taste-main-v2-{controller_uuid}"
    controller_root = controller_parent / controller_uuid
    launcher_root = launcher_parent / controller_uuid
    log_root = tmp_path / "logs"
    log_root.mkdir()
    controller_log = log_root / "controller.log"

    monkeypatch.setattr(
        controller_cli,
        "capture_policy_facts",
        lambda **_kwargs: dict(policy),
    )
    monkeypatch.setattr(
        controller_cli,
        "inspect_clean_git",
        lambda _root: (COMMIT, TREE),
    )

    def local_review_holder(receipt_path, *args, **kwargs):
        receipt_payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        snapshot = main_v2._snapshot_from_payload(receipt_payload)
        return main_v2.hold_taste_main_v2_controller_authority(
            receipt_path,
            *args,
            snapshot_reader=lambda _pid: snapshot,
            **kwargs,
        )

    # Linux production independently recaptures /proc.  The shared local
    # process helper intentionally cannot inspect a different PID, so inject
    # only the already pipe-registered child snapshot for this macOS test.
    if not sys.platform.startswith("linux"):
        monkeypatch.setattr(
            controller_cli,
            "hold_taste_main_v2_controller_authority",
            local_review_holder,
        )
    result = controller_cli.main(
        [
            "--config",
            str(Path(__file__).resolve().parents[2] / "configs/hpc.yaml"),
            "launch",
            "--control-root",
            str(policy["persistent_control_root"]),
            "--controller-root",
            str(controller_root),
            "--launcher-root",
            str(launcher_root),
            "--controller-id",
            controller_id,
            "--controller-uuid",
            controller_uuid,
            "--project-root",
            str(Path.cwd().resolve()),
            "--persistent-storage-root",
            str(policy["persistent_storage_root"]),
            "--expected-git-commit",
            COMMIT,
            "--expected-git-tree",
            TREE,
            "--controller-log",
            str(controller_log),
            "--heartbeat-count",
            "2",
        ]
    )

    assert result == 0
    ready_path = launcher_root / main_v2.LAUNCHER_READY_NAME
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    assert ready["state"] == "RUNNING"
    assert ready["science_released"] is False
    assert ready["controller_id"] == controller_id
    assert ready["anchor_heartbeat_sequence"] == 1
    assert ready["controller_anchor_heartbeat_path"] == str(
        main_v2.initial_heartbeat_path(controller_root)
    )
    assert ready["terminal_heartbeat_sequence"] >= 1
    assert ready["controller_receipt_sha256"] == main_v2.immutable_authority_sha256(
        controller_root / main_v2.CONTROLLER_RECEIPT_NAME,
        label="controller receipt",
    )
    assert controller_log.is_file()


def test_t4_runner_waits_naturally_for_spawned_children_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child:
        wait_count = 0

        def wait(self) -> int:
            self.wait_count += 1
            return 0

    child = Child()
    monkeypatch.setattr(t4_runner.subprocess, "Popen", lambda *_a, **_k: child)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with t4_runner._NaturalChildLifetime() as lifetime:
            assert lifetime.spawn(["synthetic-child"]) is child
            raise RuntimeError("synthetic failure")
    assert child.wait_count == 1
