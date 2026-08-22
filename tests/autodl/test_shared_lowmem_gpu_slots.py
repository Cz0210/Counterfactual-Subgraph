from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    _parse_task,
    plan_gpu_allocations,
)
from tests.autodl.test_gpu_colocation_benchmark_gate import build_test_gate
from src.utils.autodl_runtime import (
    GPUFileLock,
    GPUObservation,
    GPUProcess,
    GPUSharedSlotLock,
    GPULockError,
    shared_gpu_slot_admission,
)


def _gpu(*, used: int = 0, processes: tuple[GPUProcess, ...] = ()) -> GPUObservation:
    return GPUObservation(
        index=0,
        uuid="GPU-shared",
        name="A800",
        memory_total_mb=81920,
        memory_used_mb=used,
        memory_free_mb=81920 - used,
        utilization_gpu_percent=0,
        processes=processes,
    )


def _identity_reader(mapping: dict[int, tuple[int, int]]):
    return lambda pid: mapping.get(pid)


def _owner(workload_class: str) -> dict[str, object]:
    pair = ["bace_comrecgc_generation", "bace_gcfexplainer_vrrw"]
    return {
        "gpu_colocation_gate_sha256": "a" * 64,
        "gpu_shared_workload_class": workload_class,
        "gpu_colocation_authorized_pair_sha256": hashlib.sha256(
            json.dumps(pair, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "gpu_colocation_authorized_pair": pair,
    }


def test_two_shared_slots_block_exclusive_and_enforce_70_percent(tmp_path: Path) -> None:
    inventory = lambda: [_gpu()]
    identities = _identity_reader({123: (1, 1000), 456: (1, 2000)})
    slot0 = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=25000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
        process_identity_reader=identities,
    )
    slot1 = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_1",
        memory_reservation_mb=25000,
        owner=_owner("bace_comrecgc_generation"),
        inventory_reader=inventory,
        process_identity_reader=identities,
    )
    exclusive = GPUFileLock(
        tmp_path, gpu_index=0, gpu_uuid="GPU-shared", owner={"run_id": "exclusive"}
    )
    with slot0:
        slot0.update_child_pid(123)
        with slot1:
            slot1.update_child_pid(456)
            with pytest.raises(GPULockError, match="project-locked"):
                exclusive.acquire()

    too_large = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=58000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
    )
    with pytest.raises(GPULockError, match="70%|ceiling"):
        too_large.acquire()


def test_shared_admission_rejects_external_compute_process(tmp_path: Path) -> None:
    observation = _gpu(
        used=1000,
        processes=(GPUProcess(pid=999, process_name="external", used_memory_mb=1000),),
    )
    admitted, reason, _detail = shared_gpu_slot_admission(
        tmp_path,
        observation,
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=1000,
    )
    assert admitted is False
    assert "not owned" in reason


def test_second_slot_accepts_only_recorded_first_child(tmp_path: Path) -> None:
    inventory = lambda: [_gpu()]
    identities = _identity_reader({123: (1, 1000)})
    first = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=10000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
        process_identity_reader=identities,
    )
    with first:
        first.update_child_pid(123)
        observation = _gpu(
            used=9000,
            processes=(GPUProcess(pid=123, process_name="python", used_memory_mb=9000),),
        )
        admitted, reason, detail = shared_gpu_slot_admission(
            tmp_path,
            observation,
            lock_mode="shared_lowmem_slot_1",
            memory_reservation_mb=10000,
            process_identity_reader=identities,
        )
        assert admitted is True, reason
        assert detail["active_shared_child_pids"] == [123]


def test_second_slot_accepts_compute_grandchild_of_recorded_launcher(
    tmp_path: Path,
) -> None:
    inventory = lambda: [_gpu()]
    identities = _identity_reader(
        {
            123: (50, 1000),
            456: (123, 2000),
            50: (1, 500),
        }
    )
    first = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=10000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
        process_identity_reader=identities,
    )
    with first:
        first.update_child_pid(123)
        observation = _gpu(
            used=9000,
            processes=(GPUProcess(pid=456, process_name="python", used_memory_mb=9000),),
        )
        admitted, reason, detail = shared_gpu_slot_admission(
            tmp_path,
            observation,
            lock_mode="shared_lowmem_slot_1",
            memory_reservation_mb=10000,
            process_identity_reader=identities,
        )
        assert admitted is True, reason
        assert detail["active_shared_child_pids"] == [123]
        assert detail["attributed_compute_pids"] == [456]


def test_second_slot_rejects_pid_reuse_in_recorded_launcher(tmp_path: Path) -> None:
    inventory = lambda: [_gpu()]
    first_identities = _identity_reader({123: (1, 1000)})
    first = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=10000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
        process_identity_reader=first_identities,
    )
    with first:
        first.update_child_pid(123)
        reused = _identity_reader({123: (1, 9999)})
        observation = _gpu(
            used=9000,
            processes=(GPUProcess(pid=123, process_name="python", used_memory_mb=9000),),
        )
        admitted, reason, detail = shared_gpu_slot_admission(
            tmp_path,
            observation,
            lock_mode="shared_lowmem_slot_1",
            memory_reservation_mb=10000,
            process_identity_reader=reused,
        )
        assert admitted is False
        assert "not owned" in reason
        assert detail["unattributed_compute_pids"] == [123]


def test_atomic_slot_lock_rejects_unbenchmarked_pair_or_gate(tmp_path: Path) -> None:
    inventory = lambda: [_gpu()]
    identities = _identity_reader({123: (1, 1000)})
    first = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=10000,
        owner=_owner("bace_gcfexplainer_vrrw"),
        inventory_reader=inventory,
        process_identity_reader=identities,
    )
    with first:
        first.update_child_pid(123)
        wrong_pair = GPUSharedSlotLock(
            tmp_path,
            gpu_index=0,
            gpu_uuid="GPU-shared",
            lock_mode="shared_lowmem_slot_1",
            memory_reservation_mb=10000,
            owner=_owner("bace_gcfexplainer_vrrw"),
            inventory_reader=inventory,
            process_identity_reader=identities,
        )
        with pytest.raises(GPULockError, match="pair differs"):
            wrong_pair.acquire()

        different_gate_owner = _owner("bace_comrecgc_generation")
        different_gate_owner["gpu_colocation_gate_sha256"] = "c" * 64
        wrong_gate = GPUSharedSlotLock(
            tmp_path,
            gpu_index=0,
            gpu_uuid="GPU-shared",
            lock_mode="shared_lowmem_slot_1",
            memory_reservation_mb=10000,
            owner=different_gate_owner,
            inventory_reader=inventory,
            process_identity_reader=identities,
        )
        with pytest.raises(GPULockError, match="different.*gates"):
            wrong_gate.acquire()


def test_controller_can_plan_two_explicit_slots_on_one_gpu(tmp_path: Path) -> None:
    gate_path, gate_sha256 = build_test_gate(tmp_path)
    tasks = {}
    for slot in (0, 1):
        task = _parse_task(
            {
                "id": f"task-{slot}",
                "dataset": "bace",
                "stage": f"PROFILE_{slot}",
                "resource": "gpu",
                "gpu_lock_mode": f"shared_lowmem_slot_{slot}",
                "gpu_memory_reservation_mb": 12000,
                "gpu_shared_workload_class": (
                    "bace_gcfexplainer_vrrw"
                    if slot == 0
                    else "bace_comrecgc_generation"
                ),
                "gpu_colocation_gate": str(gate_path),
                "gpu_colocation_gate_sha256": gate_sha256,
            }
        )
        tasks[task.task_id] = task
    allocations = plan_gpu_allocations(
        [("task-0", "main"), ("task-1", "main")],
        tasks,
        [_gpu()],
        stable_idle_uuids=frozenset({"GPU-shared"}),
        exclusive_safe_uuids=frozenset({"GPU-shared"}),
        lock_root=tmp_path,
        capacity=2,
    )
    assert [(row.task_id, row.gpu.uuid) for row in allocations] == [
        ("task-0", "GPU-shared"),
        ("task-1", "GPU-shared"),
    ]


def test_task_manifest_shared_mode_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ControllerError, match="positive reservation"):
        _parse_task(
            {
                "id": "bad",
                "dataset": "bace",
                "stage": "BAD",
                "resource": "gpu",
                "gpu_lock_mode": "shared_lowmem_slot_0",
            }
        )
    with pytest.raises(ControllerError, match="reservation=0"):
        _parse_task(
            {
                "id": "bad-exclusive",
                "dataset": "bace",
                "stage": "BAD",
                "resource": "gpu",
                "gpu_memory_reservation_mb": 100,
            }
        )
    with pytest.raises(ControllerError, match="requires workload class"):
        _parse_task(
            {
                "id": "bad-ungated-shared",
                "dataset": "bace",
                "stage": "BAD",
                "resource": "gpu",
                "gpu_lock_mode": "shared_lowmem_slot_0",
                "gpu_memory_reservation_mb": 1000,
            }
        )
    gate_path, gate_sha256 = build_test_gate(tmp_path)
    with pytest.raises(ControllerError, match="SHA256 mismatch"):
        _parse_task(
            {
                "id": "bad-gate-hash",
                "dataset": "bace",
                "stage": "BAD",
                "resource": "gpu",
                "gpu_lock_mode": "shared_lowmem_slot_0",
                "gpu_memory_reservation_mb": 12000,
                "gpu_shared_workload_class": "bace_gcfexplainer_vrrw",
                "gpu_colocation_gate": str(gate_path),
                "gpu_colocation_gate_sha256": "0" * 64,
            }
        )
