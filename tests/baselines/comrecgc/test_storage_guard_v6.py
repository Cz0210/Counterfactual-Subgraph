from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.baselines.comrecgc import storage_guard as module
from src.baselines.comrecgc.storage_guard import (
    ComRecGCStorageGuardStop,
    StorageGuard,
    StorageGuardConfig,
)


class _Store:
    def __init__(self) -> None:
        self.checkpoint_calls = 0

    def checkpoint_wal(self, *, truncate: bool = False) -> dict[str, int]:
        assert truncate is False
        self.checkpoint_calls += 1
        return {"busy": 0, "log_pages": 2, "checkpointed_pages": 2}


class _State:
    def __init__(self) -> None:
        self.store = _Store()

    def runtime_diagnostics(self) -> dict[str, int]:
        return {"hot_cache_size": 8, "backing_store_size": 16}


def _filesystem(*, free_bytes: int, free_ratio: float, free_inodes: int):
    return {
        "filesystem_path": "/scratch",
        "total_bytes": 1_000_000,
        "used_bytes": 1_000_000 - free_bytes,
        "free_bytes": free_bytes,
        "free_ratio": free_ratio,
        "total_inodes": 1_000_000,
        "free_inodes": free_inodes,
    }


def test_storage_guard_uses_lightweight_runtime_diagnostics(tmp_path, monkeypatch) -> None:
    database = tmp_path / "graphs.sqlite3"
    database.write_bytes(b"state")
    monkeypatch.setattr(
        module,
        "_filesystem_snapshot",
        lambda _path: _filesystem(
            free_bytes=900_000, free_ratio=0.9, free_inodes=900_000
        ),
    )
    state = _State()
    guard = StorageGuard(
        StorageGuardConfig(
            root=tmp_path,
            expected_steps=100,
            check_every_steps=10,
            min_free_bytes=100,
            min_free_ratio=0.1,
            min_free_inodes=100,
        ),
        database_path=database,
    )

    guard.check(10, state)

    heartbeat = json.loads(guard.heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["storage_guard_pass"] is True
    assert heartbeat["graph_state"]["hot_cache_size"] == 8
    assert state.store.checkpoint_calls == 0


def test_disk_full_guard_checkpoints_then_fails_closed(tmp_path, monkeypatch) -> None:
    database = tmp_path / "graphs.sqlite3"
    database.write_bytes(b"state")
    monkeypatch.setattr(
        module,
        "_filesystem_snapshot",
        lambda _path: _filesystem(
            free_bytes=10, free_ratio=0.01, free_inodes=1
        ),
    )
    state = _State()
    guard = StorageGuard(
        StorageGuardConfig(
            root=tmp_path,
            expected_steps=100,
            check_every_steps=10,
            min_free_bytes=100,
            min_free_ratio=0.1,
            min_free_inodes=100,
        ),
        database_path=database,
    )

    with pytest.raises(ComRecGCStorageGuardStop):
        guard.check(20, state)

    stop = json.loads(guard.stop_path.read_text(encoding="utf-8"))
    assert stop["resume_safe"] is False
    assert stop["random_walk_resume_supported"] is False
    assert stop["checkpoint_atomic"] is True
    assert state.store.checkpoint_calls == 1


def test_storage_guard_ignores_non_boundary_steps(tmp_path) -> None:
    database = tmp_path / "graphs.sqlite3"
    database.write_bytes(b"state")
    guard = StorageGuard(
        StorageGuardConfig(root=tmp_path, expected_steps=100, check_every_steps=10),
        database_path=database,
    )
    guard.check(9, SimpleNamespace())
    assert not guard.heartbeat_path.exists()
