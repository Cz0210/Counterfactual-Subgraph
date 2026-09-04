from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.final16_owner_registry_v1 import (
    Final16OwnerRegistryError,
    atomic_write_owner_registry,
    build_owner_registry,
    validate_owner_registry,
)


COMMIT = "a" * 40
SHA = "b" * 64


def _proc(proc: Path, pid: int, ticks: int) -> None:
    target = proc / str(pid)
    target.mkdir(parents=True)
    fields = ["0"] * 20
    fields[0] = "S"
    fields[19] = str(ticks)
    (target / "stat").write_text(
        f"{pid} (final owner) " + " ".join(fields) + "\n", encoding="utf-8"
    )


def _task(tmp_path: Path, heartbeat: Path) -> dict[str, object]:
    return {
        "task_id": "mut-current-ab",
        "dataset": "Mutagenicity",
        "method": "ComRecGC",
        "stage": "TRACE_ON_OFF_AB",
        "owner_state": "ADOPTED_RUNNING",
        "owner_pid": 123,
        "owner_start_ticks": 991,
        "heartbeat": str(heartbeat),
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "output"),
        "execution_commit": COMMIT,
        "task_spec_sha": SHA,
        "gpu": 0,
        "successor_task_id": "mut-successor",
        "publisher_id": "mut-publisher",
    }


def _successor(tmp_path: Path) -> dict[str, object]:
    return {
        "task_id": "mut-successor",
        "dataset": "Mutagenicity",
        "method": "ComRecGC",
        "stage": "POST_AB_SUCCESSOR",
        "owner_state": "PREDEPLOYED",
        "owner_pid": None,
        "owner_start_ticks": None,
        "heartbeat": None,
        "input_root": str(tmp_path / "next-action"),
        "output_root": str(tmp_path / "successor-output"),
        "execution_commit": COMMIT,
        "task_spec_sha": SHA,
        "gpu": 0,
        "successor_task_id": None,
        "publisher_id": "mut-publisher",
    }


def _publisher(tmp_path: Path, *, publisher_id: str = "mut-publisher") -> dict[str, object]:
    return {
        "publisher_id": publisher_id,
        "cell_id": "Mutagenicity/ComRecGC",
        "owner_state": "PREDEPLOYED",
        "owner_pid": None,
        "owner_start_ticks": None,
        "heartbeat": None,
        "locator": str(tmp_path / f"{publisher_id}.locator.json"),
        "lease_path": str(tmp_path / f"{publisher_id}.lease"),
        "execution_commit": COMMIT,
        "claim_enabled": True,
        "active_writer_count": 0,
    }


def test_existing_owner_adoption_and_atomic_registry(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 991)
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps({"owner_pid": 123, "owner_start_ticks": 991, "phase": "RUNNING"}),
        encoding="utf-8",
    )
    authority = tmp_path / "authority"
    authority.mkdir()
    value = build_owner_registry(
        registry_id="final16-fixture",
        matrix_authority_root=authority,
        tasks=[_task(tmp_path, heartbeat), _successor(tmp_path)],
        publishers=[_publisher(tmp_path)],
        gpu_leases=[
            {
                "gpu": 0,
                "task_id": "mut-current-ab",
                "state": "HELD",
                "lease_path": str(tmp_path / "gpu0.lock"),
            }
        ],
        proc_root=proc,
    )
    assert value["tasks"][0]["owner_pid"] == 123
    assert value["tasks"][0]["owner_state"] == "ADOPTED_RUNNING"
    assert value["self_sha256"]
    output = tmp_path / "registry/current.json"
    atomic_write_owner_registry(output, value)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert validate_owner_registry(loaded, proc_root=proc) == loaded


def test_registry_rejects_stale_existing_owner(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 123, 992)
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps({"owner_pid": 123, "owner_start_ticks": 991}), encoding="utf-8"
    )
    authority = tmp_path / "authority"
    authority.mkdir()
    with pytest.raises(Final16OwnerRegistryError, match="not live"):
        build_owner_registry(
            registry_id="fixture",
            matrix_authority_root=authority,
            tasks=[_task(tmp_path, heartbeat), _successor(tmp_path)],
            publishers=[_publisher(tmp_path)],
            gpu_leases=[],
            proc_root=proc,
        )


def test_single_publisher_per_cell(tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    successor = _successor(tmp_path)
    successor["publisher_id"] = "publisher-a"
    with pytest.raises(Final16OwnerRegistryError, match="multiple canonical publishers"):
        build_owner_registry(
            registry_id="fixture",
            matrix_authority_root=authority,
            tasks=[successor],
            publishers=[
                _publisher(tmp_path, publisher_id="publisher-a"),
                _publisher(tmp_path, publisher_id="publisher-b"),
            ],
            gpu_leases=[],
            check_processes=False,
        )


def test_superseded_publisher_is_not_a_second_claim(tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    successor = _successor(tmp_path)
    canonical = _publisher(tmp_path)
    old = _publisher(tmp_path, publisher_id="old-mut-publisher")
    old["owner_state"] = "SUPERSEDED_DUPLICATE_CLAIM"
    old["claim_enabled"] = False
    value = build_owner_registry(
        registry_id="fixture",
        matrix_authority_root=authority,
        tasks=[successor],
        publishers=[canonical, old],
        gpu_leases=[],
        check_processes=False,
    )
    assert len(value["publishers"]) == 2
