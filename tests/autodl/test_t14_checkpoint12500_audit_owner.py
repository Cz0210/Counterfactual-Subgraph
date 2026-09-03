from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest

from scripts.autodl import run_t14_low_memory_resume_owner as existing_owner
from src.baselines.comrecgc.contracts import stable_json_sha256
from src.baselines.tastemolnet_t14_resume import (
    MEMORY_ADMISSION_SCHEMA,
    RESUME_SPEC_SCHEMA,
)
from src.baselines import tastemolnet_t14_checkpoint12500_audit_owner as audit_owner


GIB = 1024**3


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_canonical_bytes(value))


def _owner_proc(proc_root: Path, *, start_ticks: int = 456_789) -> None:
    process = proc_root / str(os.getpid())
    process.mkdir(parents=True)
    process.joinpath("stat").write_text(
        f"{os.getpid()} (t14 audit owner) S "
        + " ".join(["0"] * 18 + [str(start_ticks)])
        + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    pickle_bytes: int = 1024,
    cgroup_limit: int = 480 * GIB,
    cgroup_current: int = 100 * GIB,
) -> tuple[Path, Path, Path, Path]:
    root = (tmp_path / "t14").resolve()
    checkpoint_root = root / "checkpoints"
    checkpoint = checkpoint_root / "step-000000012500"
    checkpoint.mkdir(parents=True)
    state = checkpoint / audit_owner.STATE_FILENAME
    with zipfile.ZipFile(state, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("generation_state/data.pkl", b"p" * pickle_bytes)
        archive.writestr("generation_state/data/0", b"tensor-storage")
        archive.writestr("generation_state/version", b"3\n")
    # It exists and is stat-checked, but the audit is forbidden to open it.
    sqlite = checkpoint / audit_owner.SQLITE_FILENAME
    sqlite.write_bytes(b"sealed-sqlite-snapshot")
    sqlite_recorded_sha = hashlib.sha256(sqlite.read_bytes()).hexdigest()
    identity = root / "checkpoint_identity.json"
    _write_json(identity, {"status": "FROZEN", "source": "fixture"})
    provenance = {
        "dataset": "tastemolnet",
        "method": "comrecgc",
        "total_steps": "25000",
        "scientific_command_sha256": "c" * 64,
    }
    manifest: dict[str, object] = {
        "schema_version": audit_owner.CHECKPOINT_SCHEMA,
        "atomic_complete": True,
        "boundary": audit_owner.CHECKPOINT_BOUNDARY,
        "state_schema_version": audit_owner.STATE_SCHEMA,
        "checkpoint_dir": checkpoint.name,
        "completed_step": 12_500,
        "next_step": 12_501,
        "total_steps": 25_000,
        "file_digest_algorithm": "sha256",
        "checkpoint_digest_scheme": "stable_json_sha256_v1",
        "files": {
            audit_owner.STATE_FILENAME: {
                "bytes": state.stat().st_size,
                "sha256": _sha(state),
            },
            audit_owner.SQLITE_FILENAME: {
                "bytes": sqlite.stat().st_size,
                "sha256": sqlite_recorded_sha,
            },
        },
        "provenance_fingerprints": provenance,
        "provenance_sha256": stable_json_sha256(provenance),
        "scientific_argv": ["fixture"],
        "command_sha256": "c" * 64,
        "rng_components": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "sqlite_snapshot": {"integrity_check": "ok"},
        "sqlite_snapshot_method": "sqlite_connection_backup_api_v1",
        "torch_cuda_device_count": 1,
    }
    manifest["checkpoint_digest"] = stable_json_sha256(manifest)
    manifest_path = checkpoint / audit_owner.MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    _write_json(
        checkpoint / audit_owner.COMPLETE_FILENAME,
        {
            "checkpoint_digest": manifest["checkpoint_digest"],
            "manifest_sha256": _sha(manifest_path),
            "schema_version": audit_owner.CHECKPOINT_SCHEMA,
        },
    )
    _write_json(
        checkpoint_root / audit_owner.LATEST_FILENAME,
        {
            "checkpoint_digest": manifest["checkpoint_digest"],
            "checkpoint_dir": checkpoint.name,
            "completed_step": 12_500,
            "schema_version": audit_owner.LATEST_SCHEMA,
        },
    )
    limit = (tmp_path / "memory.limit").resolve()
    current = (tmp_path / "memory.current").resolve()
    failcnt = (tmp_path / "memory.failcnt").resolve()
    limit.write_text(str(cgroup_limit), encoding="ascii")
    current.write_text(str(cgroup_current), encoding="ascii")
    failcnt.write_text("0", encoding="ascii")
    spec: dict[str, object] = {
        "schema_version": RESUME_SPEC_SCHEMA,
        "status": "AUTHORIZED_TRANSPORT_ONLY_PENDING_MEMORY_CANARY",
        "source_output_root": str(root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_dir": str(checkpoint),
        "completed_step": 12_500,
        "next_step": 12_501,
        "checkpoint_digest": manifest["checkpoint_digest"],
        "checkpoint_manifest_sha256": _sha(manifest_path),
        "generation_state_sha256": _sha(state),
        "generation_state_bytes": state.stat().st_size,
        "sqlite_snapshot_sha256": sqlite_recorded_sha,
        "checkpoint_identity_path": str(identity),
        "checkpoint_identity_sha256": _sha(identity),
        "source_execution_commit": "1" * 40,
        "resume_execution_commit": "2" * 40,
        "scientific_projection_sha256": "d" * 64,
        "transport_changes": [
            "single_pass_checkpoint_deserialize",
            "consumptive_state_transfer",
        ],
        "scientific_state_changes": False,
        "rng_changes": False,
        "candidate_order_changes": False,
        "auditor_concurrency_policy": "SERIAL_ONLY",
        "full_state_consumer_limit": 1,
        "full_state_lock_path": str(
            checkpoint_root / ".t14-full-state-consumer.lock"
        ),
        "resume_parity_canary_max_steps": 50,
        "memory": {
            "schema_version": MEMORY_ADMISSION_SCHEMA,
            "historical_process_peak_bytes": 386 * GIB,
            "historical_checkpoint_peak_bytes": 448 * GIB,
            "historical_required_headroom_bytes": 512 * GIB,
            "safety_margin_bytes": 64 * GIB,
            "optimized_canary_receipt_required": True,
            "cgroup_limit_path": str(limit),
            "cgroup_current_path": str(current),
            "cgroup_failcnt_path": str(failcnt),
        },
    }
    spec["spec_sha256"] = hashlib.sha256(_canonical_bytes(spec)).hexdigest()
    spec_path = (tmp_path / "resume-spec.json").resolve()
    _write_json(spec_path, spec)
    proc_root = (tmp_path / "proc").resolve()
    _owner_proc(proc_root)
    return spec_path, checkpoint, proc_root, sqlite


def test_checkpoint_audit_hashes_state_without_opening_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, _checkpoint, _proc, sqlite = _fixture(tmp_path)
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object):
        if self == sqlite or self.name.endswith(("-wal", "-shm")):
            raise AssertionError("SQLite/WAL payload must never be opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    _spec, evidence = audit_owner.audit_checkpoint_without_sqlite(spec_path)

    assert evidence["manifest_and_state_hashes_pass"] is True
    assert evidence["sqlite_payload_opened"] is False
    assert evidence["sqlite_payload_hash_recomputed"] is False
    assert evidence["torch_load_invoked"] is False


def test_monolithic_pickle_writes_blocked_owner_terminal_without_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, _checkpoint, proc_root, _sqlite = _fixture(
        tmp_path, pickle_bytes=4096
    )
    monkeypatch.setattr(audit_owner, "MAX_STREAMABLE_PICKLE_BYTES", 1024)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch must not be imported by the audit owner")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    owner_root = (tmp_path / "owner").resolve()
    result = audit_owner.main(
        [
            "--resume-spec",
            str(spec_path),
            "--owner-root",
            str(owner_root),
            "--proc-root",
            str(proc_root),
        ]
    )

    assert result == audit_owner.BLOCKED_EXIT
    heartbeat = json.loads((owner_root / "heartbeat.json").read_text())
    terminal = json.loads((owner_root / "terminal.json").read_text())
    archive = json.loads((owner_root / "archive_layout.json").read_text())
    assert heartbeat["owner_pid"] == os.getpid()
    assert terminal["reason_code"] == "BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE"
    assert terminal["scientifically_safe_to_launch"] is False
    assert terminal["science_started"] is False
    assert terminal["canary_steps_executed"] == 0
    assert terminal["torch_load_invoked"] is False
    assert archive["pickle_bytes"] == 4096
    assert archive["block_reason"] == "MONOLITHIC_DATA_PICKLE_EXCEEDS_STREAMING_LIMIT"


def test_owner_fails_closed_when_generation_state_hash_changes(
    tmp_path: Path,
) -> None:
    spec_path, checkpoint, proc_root, _sqlite = _fixture(tmp_path)
    with (checkpoint / audit_owner.STATE_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    owner_root = (tmp_path / "owner").resolve()

    result = audit_owner.main(
        [
            "--resume-spec",
            str(spec_path),
            "--owner-root",
            str(owner_root),
            "--proc-root",
            str(proc_root),
        ]
    )

    terminal = json.loads((owner_root / "terminal.json").read_text())
    assert result == audit_owner.ERROR_EXIT
    assert terminal["reason_code"] == "FAILED_T14_CHECKPOINT_AUDIT"
    assert terminal["science_started"] is False


def test_owner_preserves_64_gib_margin_before_any_canary(
    tmp_path: Path,
) -> None:
    spec_path, _checkpoint, proc_root, _sqlite = _fixture(
        tmp_path,
        pickle_bytes=128,
        cgroup_limit=100 * GIB,
        cgroup_current=50 * GIB,
    )
    owner_root = (tmp_path / "owner").resolve()

    result = audit_owner.main(
        [
            "--resume-spec",
            str(spec_path),
            "--owner-root",
            str(owner_root),
            "--proc-root",
            str(proc_root),
        ]
    )

    terminal = json.loads((owner_root / "terminal.json").read_text())
    cgroup = json.loads((owner_root / "cgroup_headroom.json").read_text())
    assert result == audit_owner.BLOCKED_EXIT
    assert terminal["reason_code"] == "BLOCKED_T14_CGROUP_HEADROOM"
    assert cgroup["cgroup_headroom_bytes"] == 50 * GIB
    assert cgroup["safety_margin_preserved_before_canary"] is False


def test_new_owner_does_not_replace_existing_production_owner() -> None:
    assert audit_owner.OWNER_SCHEMA != existing_owner.OWNER_SCHEMA
    assert "subprocess" not in audit_owner.__dict__

