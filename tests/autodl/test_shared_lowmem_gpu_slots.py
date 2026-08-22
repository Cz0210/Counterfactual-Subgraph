from __future__ import annotations

from pathlib import Path

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    _parse_task,
    plan_gpu_allocations,
)
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


def test_two_shared_slots_block_exclusive_and_enforce_70_percent(tmp_path: Path) -> None:
    inventory = lambda: [_gpu()]
    slot0 = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=25000,
        inventory_reader=inventory,
    )
    slot1 = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_1",
        memory_reservation_mb=25000,
        inventory_reader=inventory,
    )
    exclusive = GPUFileLock(
        tmp_path, gpu_index=0, gpu_uuid="GPU-shared", owner={"run_id": "exclusive"}
    )
    with slot0, slot1:
        with pytest.raises(GPULockError, match="project-locked"):
            exclusive.acquire()

    too_large = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=58000,
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
    first = GPUSharedSlotLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-shared",
        lock_mode="shared_lowmem_slot_0",
        memory_reservation_mb=10000,
        inventory_reader=inventory,
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
        )
        assert admitted is True, reason
        assert detail["active_shared_child_pids"] == [123]


def test_controller_can_plan_two_explicit_slots_on_one_gpu(tmp_path: Path) -> None:
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


def test_task_manifest_shared_mode_is_fail_closed() -> None:
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
