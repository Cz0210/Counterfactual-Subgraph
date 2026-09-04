from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts.autodl.status_final16_successors_v1 import read_status
from src.utils.final16_owner_registry_v1 import (
    atomic_write_owner_registry,
    build_owner_registry,
)
from src.utils.final16_successors_v1 import (
    Final16SuccessorsError,
    HEARTBEAT_SCHEMA,
    build_snapshot,
)
from src.utils.final_four_cells_observer import REMAINING_CELLS, stable_sha256


COMMIT = "a" * 40
SHA = "b" * 64
BASE_CELLS = [f"base/cell-{index}" for index in range(12)]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _proc(proc: Path, pid: int, ticks: int) -> None:
    target = proc / str(pid)
    target.mkdir(parents=True)
    fields = ["0"] * 20
    fields[0] = "S"
    fields[19] = str(ticks)
    (target / "stat").write_text(
        f"{pid} (final16 owner) " + " ".join(fields) + "\n", encoding="ascii"
    )


def _matrix(authority: Path, cells: list[str]) -> None:
    authority.mkdir(parents=True)
    _write(
        authority / "state.json",
        {
            "schema_version": "fast16_matrix_authority_pointer_v1",
            "latest_count": len(cells),
            "applied_cells": cells,
        },
    )


def _registry(
    tmp_path: Path, *, authority: Path, owner_ticks: int = 991
) -> Path:
    heartbeat = _write(
        tmp_path / "mut-heartbeat.json",
        {"owner_pid": 123, "owner_start_ticks": 991, "state": "RUNNING"},
    )
    tasks = [
        {
            "task_id": "mut-current-ab",
            "dataset": "Mutagenicity",
            "method": "ComRecGC",
            "stage": "TRACE_ON_OFF_AB",
            "owner_state": "ADOPTED_RUNNING",
            "owner_pid": 123,
            "owner_start_ticks": owner_ticks,
            "heartbeat": str(heartbeat),
            "input_root": str(tmp_path / "mut-input"),
            "output_root": str(tmp_path / "mut-output"),
            "execution_commit": COMMIT,
            "task_spec_sha": SHA,
            "gpu": 0,
            "successor_task_id": "mut-next",
            "publisher_id": "mut-publisher",
        },
        {
            "task_id": "mut-next",
            "dataset": "Mutagenicity",
            "method": "ComRecGC",
            "stage": "POST_AB_SUCCESSOR",
            "owner_state": "PREDEPLOYED",
            "owner_pid": None,
            "owner_start_ticks": None,
            "heartbeat": None,
            "input_root": str(tmp_path / "next-action"),
            "output_root": str(tmp_path / "next-output"),
            "execution_commit": COMMIT,
            "task_spec_sha": SHA,
            "gpu": 0,
            "successor_task_id": None,
            "publisher_id": "mut-publisher",
        },
    ]
    publishers = [
        {
            "publisher_id": "mut-publisher",
            "cell_id": "Mutagenicity/ComRecGC",
            "owner_state": "PREDEPLOYED",
            "owner_pid": None,
            "owner_start_ticks": None,
            "heartbeat": None,
            "locator": str(tmp_path / "mut.locator.json"),
            "lease_path": str(tmp_path / "mut.publisher.lock"),
            "execution_commit": COMMIT,
            "claim_enabled": True,
            "active_writer_count": 0,
        }
    ]
    value = build_owner_registry(
        registry_id="final16-fixture",
        matrix_authority_root=authority,
        tasks=tasks,
        publishers=publishers,
        gpu_leases=[
            {
                "gpu": 0,
                "task_id": "mut-current-ab",
                "state": "HELD",
                "lease_path": str(tmp_path / "gpu0.lock"),
            }
        ],
        check_processes=False,
    )
    path = tmp_path / "registry/current.json"
    atomic_write_owner_registry(path, value)
    return path


def test_controller_adopts_existing_owner_without_writing_authority(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 991)
    authority = tmp_path / "authority"
    _matrix(authority, BASE_CELLS)
    registry = _registry(tmp_path, authority=authority)
    authority_before = (authority / "state.json").read_bytes()
    registry_before = registry.read_bytes()

    value = build_snapshot(
        matrix_authority_root=authority,
        owner_registry_path=registry,
        proc_root=proc,
    )

    assert value["matrix_complete_cells"] == 12
    assert value["tasks"]["mut-current-ab"]["observation"] == "ADOPT_EXISTING_OWNER"
    assert value["tasks"]["mut-next"]["observation"] == "NOT_RUNNING"
    assert value["dispatch_mode"] == "SEALED_ONE_SHOT_BINDERS_ONLY"
    assert value["controller_launches_science"] is False
    assert value["controller_launches_publishers"] is False
    assert value["matrix_write_performed"] is False
    assert value["gpu_lock_acquired"] is False
    assert (authority / "state.json").read_bytes() == authority_before
    assert registry.read_bytes() == registry_before


def test_stale_registry_owner_is_reported_but_not_restarted(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 992)
    authority = tmp_path / "authority"
    _matrix(authority, BASE_CELLS)
    registry = _registry(tmp_path, authority=authority)

    value = build_snapshot(
        matrix_authority_root=authority,
        owner_registry_path=registry,
        proc_root=proc,
    )

    assert value["state"] == "BLOCKED_STALE_REGISTRY"
    assert value["stale_owners"] == ["mut-current-ab"]
    assert value["science_restart_performed"] is False
    assert value["signal_sent"] is False


def test_registry_cannot_redirect_to_second_matrix_authority(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 991)
    authority = tmp_path / "authority"
    other = tmp_path / "other-authority"
    _matrix(authority, BASE_CELLS)
    _matrix(other, BASE_CELLS)
    registry = _registry(tmp_path, authority=other)

    with pytest.raises(Final16SuccessorsError, match="different matrix authority"):
        build_snapshot(
            matrix_authority_root=authority,
            owner_registry_path=registry,
            proc_root=proc,
        )


def test_complete_matrix_only_delegates_final_export_gate(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 991)
    authority = tmp_path / "authority"
    _matrix(authority, [*BASE_CELLS, *REMAINING_CELLS])
    registry = _registry(tmp_path, authority=authority)

    value = build_snapshot(
        matrix_authority_root=authority,
        owner_registry_path=registry,
        proc_root=proc,
    )

    assert value["state"] == "MAIN_MATRIX_COMPLETE"
    assert value["missing_cells"] == []
    assert value["gnn_ablation_gate"] == "WAITING_FINAL_EXPORT_RECEIPTS"
    assert value["llm_ablation_started"] is False
    assert value["gnn_ablation_started"] is False


def test_status_checks_heartbeat_pid_generation(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 432, 778)
    state_root = tmp_path / "state"
    heartbeat = {
        "schema_version": HEARTBEAT_SCHEMA,
        "controller_id": "controller-fixture",
        "controller_pid": 432,
        "controller_start_ticks": 778,
        "state": "RUNNING_LONG_EXPERIMENTS",
        "snapshot": {},
        "evidence_error": None,
        "sequence": 3,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    heartbeat["heartbeat_sha256"] = stable_sha256(heartbeat)
    _write(state_root / "heartbeat.json", heartbeat)

    value, code = read_status(
        state_root=state_root, max_age_seconds=180, proc_root=proc
    )

    assert code == 0
    assert value["status"] == "RUNNING"
    assert value["controller_process_live"] is True


def test_controller_source_has_no_dispatch_or_matrix_mutation() -> None:
    project = Path(__file__).resolve().parents[2]
    runner = (project / "scripts/autodl/run_final16_successors_v1.py").read_text(
        encoding="utf-8"
    )
    launcher = (project / "scripts/autodl/launch_final16_successors_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "subprocess" not in runner
    assert "os.kill" not in runner
    assert "GPUFileLock" not in runner
    assert 'export CUDA_VISIBLE_DEVICES=""' in launcher
    assert "RUN_LLM_ABLATION=0" in launcher
    assert "RUN_GNN_ABLATION=0" in launcher
