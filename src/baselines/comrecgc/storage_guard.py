"""Fail-closed storage monitoring for long COMRECGC random walks."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import write_json


STORAGE_GUARD_POLICY = "persistent_scratch_projected_capacity_v1"


class ComRecGCStorageGuardStop(RuntimeError):
    """Raised after durable state is flushed before storage exhaustion."""

    def __init__(self, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "[COMRECGC_STORAGE_GUARD_STOP] "
            f"step={self.diagnostics.get('current_step')} "
            f"reasons={self.diagnostics.get('stop_reasons')}"
        )


@dataclass(frozen=True, slots=True)
class StorageGuardConfig:
    root: Path
    expected_steps: int
    check_every_steps: int = 500
    min_free_bytes: int = 20 * 1024**3
    min_free_ratio: float = 0.05
    min_free_inodes: int = 100_000
    projection_safety_factor: float = 1.30

    def validate(self) -> None:
        if self.expected_steps <= 0:
            raise ValueError("Storage guard expected_steps must be positive.")
        if self.check_every_steps <= 0:
            raise ValueError("Storage guard check_every_steps must be positive.")
        if self.min_free_bytes < 0 or self.min_free_inodes < 0:
            raise ValueError("Storage guard free-space limits cannot be negative.")
        if not 0.0 <= self.min_free_ratio < 1.0:
            raise ValueError("Storage guard min_free_ratio must be in [0, 1).")
        if self.projection_safety_factor < 1.0:
            raise ValueError("Storage guard projection safety factor must be >= 1.")


def _filesystem_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    stat = os.statvfs(path)
    free_inodes = int(stat.f_favail)
    return {
        "filesystem_path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_ratio": float(usage.free / usage.total) if usage.total else 0.0,
        "total_inodes": int(stat.f_files),
        "free_inodes": free_inodes,
    }


def sqlite_state_sizes(database_path: str | Path) -> dict[str, int]:
    database = Path(database_path).expanduser().resolve()
    wal = Path(f"{database}-wal")
    shm = Path(f"{database}-shm")
    return {
        "database_bytes": database.stat().st_size if database.exists() else 0,
        "wal_bytes": wal.stat().st_size if wal.exists() else 0,
        "shm_bytes": shm.stat().st_size if shm.exists() else 0,
    }


class StorageGuard:
    """Monitor one persistent scratch state and stop before SQLite corruption."""

    def __init__(self, config: StorageGuardConfig, *, database_path: str | Path) -> None:
        config.validate()
        self.config = config
        self.root = config.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = Path(database_path).expanduser().resolve()
        try:
            self.database_path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                "COMRECGC graph database must live below the guarded scratch root."
            ) from exc
        self.heartbeat_path = self.root / "storage_guard_heartbeat.json"
        self.stop_path = self.root / "STORAGE_GUARD_STOP.json"
        self.checkpoint_path = self.root / "storage_guard_checkpoint.json"

    def snapshot(self, *, current_step: int, state: Any) -> dict[str, Any]:
        filesystem = _filesystem_snapshot(self.root)
        sizes = sqlite_state_sizes(self.database_path)
        state_bytes = sum(sizes.values())
        bytes_per_step = state_bytes / max(int(current_step), 1)
        projected_final_bytes = int(bytes_per_step * self.config.expected_steps)
        projected_remaining_bytes = max(projected_final_bytes - state_bytes, 0)
        required_headroom = max(
            int(self.config.min_free_bytes),
            int(projected_remaining_bytes * self.config.projection_safety_factor),
        )
        stop_reasons: list[str] = []
        if filesystem["free_bytes"] < required_headroom:
            stop_reasons.append("projected_free_bytes_below_required_headroom")
        if filesystem["free_ratio"] < self.config.min_free_ratio:
            stop_reasons.append("free_ratio_below_limit")
        if filesystem["free_inodes"] < self.config.min_free_inodes:
            stop_reasons.append("free_inodes_below_limit")
        audit = {
            "schema_version": "comrecgc_storage_guard_v1",
            "policy": STORAGE_GUARD_POLICY,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "current_step": int(current_step),
            "expected_steps": int(self.config.expected_steps),
            "check_every_steps": int(self.config.check_every_steps),
            **filesystem,
            **sizes,
            "state_bytes": int(state_bytes),
            "bytes_per_step": float(bytes_per_step),
            "projected_final_bytes": int(projected_final_bytes),
            "projected_remaining_bytes": int(projected_remaining_bytes),
            "required_headroom_bytes": int(required_headroom),
            "stop_reasons": stop_reasons,
            "storage_guard_pass": not stop_reasons,
            "graph_state": state.runtime_diagnostics(),
            "config": {**asdict(self.config), "root": str(self.root)},
        }
        return audit

    def check(self, current_step: int, state: Any) -> None:
        if int(current_step) % int(self.config.check_every_steps) != 0:
            return
        audit = self.snapshot(current_step=int(current_step), state=state)
        write_json(self.heartbeat_path, audit)
        print(
            "[COMRECGC_STORAGE_GUARD] "
            f"current_step={current_step} free_bytes={audit['free_bytes']} "
            f"free_ratio={audit['free_ratio']:.6f} "
            f"free_inodes={audit['free_inodes']} "
            f"database_bytes={audit['database_bytes']} wal_bytes={audit['wal_bytes']} "
            f"bytes_per_step={audit['bytes_per_step']:.3f} "
            f"projected_final_bytes={audit['projected_final_bytes']} "
            f"storage_guard_pass={audit['storage_guard_pass']}",
            flush=True,
        )
        if audit["storage_guard_pass"]:
            return

        wal_checkpoint = state.store.checkpoint_wal(truncate=False)
        checkpoint = {
            **audit,
            "wal_checkpoint": wal_checkpoint,
            "checkpoint_atomic": True,
            "random_walk_resume_supported": False,
            "resume_safe": False,
            "restart_policy": "fresh_from_step_0",
            "reason": (
                "Pinned COMRECGC does not expose complete RNG/transition checkpoints; "
                "this guard preserves a valid database and fails closed without "
                "claiming scientific resume support."
            ),
        }
        write_json(self.checkpoint_path, checkpoint)
        write_json(self.stop_path, checkpoint)
        raise ComRecGCStorageGuardStop(checkpoint)
