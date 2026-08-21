from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    BACE_STAGES,
    GPUFileLock,
    GPULockError,
    ProjectGPUSlotLock,
    GPUInventoryError,
    GPUObservation,
    StageTransitionError,
    append_jsonl_locked,
    assert_bace_stage_can_start,
    assert_tastemolnet_launch_allowed,
    atomic_write_json,
    build_runtime_layout,
    initialize_bace_stage_tree,
    latest_registry_events,
    mark_bace_stage_pass,
    observe_stable_idle_gpus,
    parse_gpu_inventory,
    read_registry,
    resolve_passed_bace_stage_output,
    validate_max_gpus,
    verify_required_outputs,
)


def _gpu(index: int, uuid: str, *, free: int = 80_000, util: int = 0) -> GPUObservation:
    return GPUObservation(
        index=index,
        uuid=uuid,
        name="NVIDIA A800 80GB PCIe",
        memory_total_mb=81_920,
        memory_used_mb=81_920 - free,
        memory_free_mb=free,
        utilization_gpu_percent=util,
    )


def test_gpu_inventory_requires_memory_utilization_and_no_compute_process() -> None:
    rows = "\n".join(
        [
            "0, GPU-aaa, NVIDIA A800 80GB PCIe, 81920, 100, 81820, 0",
            "1, GPU-bbb, NVIDIA A800 80GB PCIe, 81920, 20000, 61920, 0",
        ]
    )
    processes = "GPU-bbb, 1234, python, 19000"
    inventory = parse_gpu_inventory(rows, processes)
    assert inventory[0].is_idle(min_free_memory_mb=16_000, max_utilization_percent=10)
    assert not inventory[1].is_idle(
        min_free_memory_mb=16_000, max_utilization_percent=10
    )
    assert inventory[1].process_count == 1


def test_stable_idle_is_intersection_across_full_window() -> None:
    samples = iter(
        [
            [_gpu(0, "GPU-a"), _gpu(1, "GPU-b")],
            [_gpu(0, "GPU-a"), _gpu(1, "GPU-b", util=90)],
            [_gpu(0, "GPU-a"), _gpu(1, "GPU-b")],
        ]
    )
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    result = observe_stable_idle_gpus(
        stable_seconds=2,
        sample_interval_seconds=1,
        min_free_memory_mb=16_000,
        max_utilization_percent=10,
        sampler=lambda: next(samples),
        monotonic=monotonic,
        sleep=sleep,
    )
    assert result.samples == 3
    assert result.stable_idle_uuids == {"GPU-a"}
    assert [gpu.uuid for gpu in result.selected(max_gpus=1)] == ["GPU-a"]


def test_more_than_two_gpus_is_rejected() -> None:
    assert validate_max_gpus(2) == 2
    with pytest.raises(GPUInventoryError, match=r"\[1, 2\]"):
        validate_max_gpus(3)


def test_uuid_gpu_lock_rejects_second_writer(tmp_path: Path) -> None:
    first = GPUFileLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-immutable-a",
        owner={"run_id": "first"},
    )
    second = GPUFileLock(
        tmp_path,
        gpu_index=0,
        gpu_uuid="GPU-immutable-a",
        owner={"run_id": "second"},
    )
    with first:
        with pytest.raises(GPULockError, match="project-locked"):
            second.acquire()


def test_project_gpu_slots_enforce_aggregate_two_gpu_cap(tmp_path: Path) -> None:
    first = ProjectGPUSlotLock(tmp_path, max_slots=2, owner={"run_id": "one"})
    second = ProjectGPUSlotLock(tmp_path, max_slots=2, owner={"run_id": "two"})
    third = ProjectGPUSlotLock(tmp_path, max_slots=2, owner={"run_id": "three"})
    with first, second:
        assert {first.slot, second.slot} == {0, 1}
        with pytest.raises(GPULockError, match="All 2 project GPU slots"):
            third.acquire()


def test_bace_stage_templates_and_predecessor_gate(tmp_path: Path) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    initialize_bace_stage_tree(layout)
    for stage in BACE_STAGES:
        root = layout.stages_root / stage
        assert (root / "state.json").is_file()
        assert (root / "manifest.json").is_file()
        assert (root / "gate.json").is_file()
    with pytest.raises(StageTransitionError, match="requires B0_AUDIT"):
        assert_bace_stage_can_start(layout, "B1_DATA_READY")

    audit = tmp_path / "audit.json"
    split = tmp_path / "split_manifest.json"
    audit.write_text('{"status":"PASS"}\n', encoding="utf-8")
    split.write_text('{"status":"PASS"}\n', encoding="utf-8")
    mark_bace_stage_pass(layout, stage="B0_AUDIT", evidence=[audit], note="audit")
    mark_bace_stage_pass(
        layout, stage="B1_DATA_READY", evidence=[split], note="frozen split"
    )
    assert_bace_stage_can_start(layout, "B2_GNN_SMOKE")


def test_control_root_defaults_persistent_and_explicit_path_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "persistent-data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    monkeypatch.delenv("AUTODL_CONTROL_ROOT", raising=False)
    default = build_runtime_layout(project_root=project, data_root=data)
    assert default.control_root == (
        data / "counterfactual-subgraph-runtime" / "control"
    ).resolve()
    assert project not in default.control_root.parents

    explicit = data / "custom-control"
    monkeypatch.setenv("AUTODL_CONTROL_ROOT", str(explicit))
    assert (
        build_runtime_layout(project_root=project, data_root=data).control_root
        == explicit.resolve()
    )

    monkeypatch.setenv("AUTODL_CONTROL_ROOT", "relative/control")
    with pytest.raises(AutoDLRuntimeError, match="absolute"):
        build_runtime_layout(project_root=project, data_root=data)
    monkeypatch.setenv("AUTODL_CONTROL_ROOT", str(tmp_path / "outside"))
    with pytest.raises(AutoDLRuntimeError, match="persistent data root"):
        build_runtime_layout(project_root=project, data_root=data)


def test_passed_stage_output_comes_only_from_frozen_persistent_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    (project / ".git").mkdir()
    monkeypatch.delenv("AUTODL_CONTROL_ROOT", raising=False)
    layout = build_runtime_layout(project_root=project, data_root=data).ensure()
    initialize_bace_stage_tree(layout)
    output = layout.artifacts_dir / "gnn_oracles" / "bace" / "full"
    output.mkdir(parents=True)
    (output / "model.pt").write_bytes(b"model")
    paths = layout.stages_root / "B3_GNN_FULL"
    atomic_write_json(paths / "state.json", {"state": "PASS"})
    atomic_write_json(paths / "gate.json", {"status": "PASS"})
    atomic_write_json(
        paths / "manifest.json",
        {"status": "FROZEN", "expected_output": str(output)},
    )
    assert resolve_passed_bace_stage_output(
        layout, "B3_GNN_FULL", required_relative=["model.pt"]
    ) == output.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.pt").write_bytes(b"model")
    atomic_write_json(
        paths / "manifest.json",
        {"status": "FROZEN", "expected_output": str(outside)},
    )
    with pytest.raises(StageTransitionError, match="escapes persistent artifacts"):
        resolve_passed_bace_stage_output(
            layout, "B3_GNN_FULL", required_relative=["model.pt"]
        )


def test_tastemolnet_heavy_run_defaults_fail_closed() -> None:
    with pytest.raises(AutoDLRuntimeError, match="HEAVY_RUN_DISABLED"):
        assert_tastemolnet_launch_allowed(
            dataset="tastemolnet", heavy=True, run_tastemolnet="0"
        )
    assert_tastemolnet_launch_allowed(
        dataset="tastemolnet", heavy=False, run_tastemolnet="0"
    ) is None
    assert_tastemolnet_launch_allowed(
        dataset="tastemolnet", heavy=True, run_tastemolnet="1"
    ) is None


def test_registry_is_append_only_and_latest_event_wins(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    append_jsonl_locked(path, {"run_id": "r1", "state": "RUNNING"})
    append_jsonl_locked(path, {"run_id": "r1", "state": "PASS"})
    append_jsonl_locked(path, {"run_id": "r2", "state": "FAILED"})
    rows = read_registry(path)
    assert len(rows) == 3
    latest = {row["run_id"]: row for row in latest_registry_events(rows)}
    assert latest["r1"]["state"] == "PASS"
    assert latest["r2"]["state"] == "FAILED"


def test_required_bundle_paths_are_nonempty_and_cannot_escape(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    output.mkdir()
    (output / "model.pt").write_bytes(b"weights")
    assert verify_required_outputs(output, ["model.pt"]) == []
    failures = verify_required_outputs(output, ["missing.json", "../escape"])
    assert any("missing" in failure for failure in failures)
    assert any("unsafe" in failure for failure in failures)
