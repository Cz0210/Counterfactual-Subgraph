"""Atomic current-chain pointer for the CPU-only Taste GlobalGCE offload.

The pointer is deliberately small.  It is status evidence, not a scheduler or
scientific checkpoint.  Writers serialize the whole T8 continuation through
one advisory lock and publish ``current.json`` with an embedded canonical
SHA-256 plus a detached file SHA-256.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping
import uuid


CURRENT_SCHEMA = "t8_hpc_current_chain_v1"
CURRENT_HASH_FIELD = "current_sha256"
JOB_ID_RE = re.compile(r"^[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class T8ChainPointerError(RuntimeError):
    """Raised when the current-chain pointer is malformed or regresses."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _unsigned(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != CURRENT_HASH_FIELD}


def _validate(payload: Mapping[str, Any]) -> None:
    required_strings = (
        "state",
        "active_stage",
        "updated_at",
        "output_root",
        "continuation_root",
        "controller_commit",
        "science_commit",
    )
    if payload.get("schema_version") != CURRENT_SCHEMA:
        raise T8ChainPointerError("current pointer schema mismatch")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
        raise T8ChainPointerError("current pointer has missing string fields")
    claimed = payload.get(CURRENT_HASH_FIELD)
    if not isinstance(claimed, str) or not SHA256_RE.fullmatch(claimed):
        raise T8ChainPointerError("current pointer self-hash is malformed")
    if canonical_sha256(_unsigned(payload)) != claimed:
        raise T8ChainPointerError("current pointer self-hash mismatch")
    for key in (
        "upstream_job_id",
        "canary_job_id",
        "followup_job_id",
        "array_job_id",
        "merge_job_id",
        "package_job_id",
    ):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or not JOB_ID_RE.fullmatch(value)):
            raise T8ChainPointerError(f"current pointer {key} is not a Slurm job ID")
    depth = payload.get("refinement_depth")
    if depth is not None and (type(depth) is not int or depth < 1):
        raise T8ChainPointerError("current pointer refinement depth is invalid")
    canary = payload.get("canary_job_id")
    followup = payload.get("followup_job_id")
    dependency = payload.get("followup_dependency")
    if (canary is None) != (followup is None):
        raise T8ChainPointerError("canary and follow-up IDs must be recorded together")
    if canary is not None and dependency != f"afterany:{canary}":
        raise T8ChainPointerError("follow-up dependency is not bound to the canary")
    previous = payload.get("previous_current_sha256")
    if previous is not None and (
        not isinstance(previous, str) or not SHA256_RE.fullmatch(previous)
    ):
        raise T8ChainPointerError("previous current pointer hash is malformed")


def load_current_pointer(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise T8ChainPointerError(f"current pointer is not one regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise T8ChainPointerError(f"current pointer is unreadable: {path}") from exc
    if type(payload) is not dict:
        raise T8ChainPointerError("current pointer must be a JSON object")
    _validate(payload)
    detached_path = path.with_suffix(path.suffix + ".sha256")
    if detached_path.is_symlink() or not detached_path.is_file():
        raise T8ChainPointerError("current pointer detached SHA is missing")
    detached = detached_path.read_text(encoding="ascii").strip()
    if detached != sha256_file(path):
        raise T8ChainPointerError("current pointer detached SHA mismatch")
    return dict(payload)


def _stage_order(payload: Mapping[str, Any]) -> tuple[int, int]:
    stage = payload.get("active_stage")
    if stage == "REFINEMENT_CANARY":
        return 10, int(payload.get("refinement_depth", 0))
    if stage == "FULL_CHAIN":
        return 20, 0
    if stage == "TERMINAL":
        return 30, 0
    return 0, int(payload.get("refinement_depth", 0) or 0)


def write_current_pointer(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Publish a monotonic current pointer while the caller holds ``chain_lock``.

    Replaying the same canary/follow-up pair is idempotent.  A second follow-up
    for an already recorded canary or a move back to a shallower chain stage is
    rejected before any bytes are replaced.
    """

    candidate = dict(payload)
    if CURRENT_HASH_FIELD in candidate:
        raise T8ChainPointerError(f"reserved field already set: {CURRENT_HASH_FIELD}")
    existing: dict[str, Any] | None = None
    if path.exists():
        existing = load_current_pointer(path)
        same_canary = (
            candidate.get("canary_job_id") is not None
            and candidate.get("canary_job_id") == existing.get("canary_job_id")
        )
        if same_canary and candidate.get("followup_job_id") != existing.get("followup_job_id"):
            raise T8ChainPointerError("one canary cannot own two follow-up jobs")
        if _stage_order(candidate) < _stage_order(existing):
            return existing
        if _stage_order(candidate) == _stage_order(existing):
            identity_keys = (
                "active_stage",
                "refinement_depth",
                "canary_job_id",
                "followup_job_id",
                "array_job_id",
                "merge_job_id",
                "package_job_id",
            )
            if all(candidate.get(key) == existing.get(key) for key in identity_keys):
                return existing
            raise T8ChainPointerError("current pointer stage identity conflicts")
        candidate["previous_current_sha256"] = existing[CURRENT_HASH_FIELD]
    candidate[CURRENT_HASH_FIELD] = canonical_sha256(candidate)
    _validate(candidate)
    encoded = canonical_bytes(candidate) + b"\n"
    _atomic_write(path, encoded)
    _atomic_write(
        path.with_suffix(path.suffix + ".sha256"),
        (sha256_file(path) + "\n").encode("ascii"),
    )
    return candidate


@contextmanager
def chain_lock(current_pointer: Path) -> Iterator[None]:
    """Serialize all continuation decisions sharing one stable pointer."""

    current_pointer.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = current_pointer.parent / ".chain.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def followup_for_canary(path: Path, canary_job_id: str) -> str | None:
    """Return the already recorded follow-up for ``canary_job_id`` if current."""

    if not path.exists():
        return None
    payload = load_current_pointer(path)
    if payload.get("canary_job_id") != canary_job_id:
        return None
    value = payload.get("followup_job_id")
    return str(value) if value is not None else None
