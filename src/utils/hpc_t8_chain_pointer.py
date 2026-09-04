"""Atomic current-chain pointer for the CPU-only Taste GlobalGCE offload.

The pointer is deliberately small.  It is status evidence, not a scheduler or
scientific checkpoint.  Writers serialize the whole T8 continuation through
one advisory lock and publish ``current.json`` with an embedded canonical
SHA-256 plus a detached file SHA-256.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
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
ARRAY_RANGE_RE = re.compile(r"^(?P<start>[0-9]+)-(?P<end>[0-9]+)$")


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


def _load_self_hashed_json(path: Path, *, hash_field: str) -> dict[str, Any]:
    """Load one immutable controller receipt, including its detached digest."""

    if path.is_symlink() or not path.is_file():
        raise T8ChainPointerError(f"receipt is not one regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise T8ChainPointerError(f"receipt is unreadable: {path}") from exc
    if type(payload) is not dict:
        raise T8ChainPointerError(f"receipt must be a JSON object: {path}")
    claimed = payload.get(hash_field)
    if not isinstance(claimed, str) or not SHA256_RE.fullmatch(claimed):
        raise T8ChainPointerError(f"receipt self-hash is malformed: {path}")
    unsigned = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_sha256(unsigned) != claimed:
        raise T8ChainPointerError(f"receipt self-hash mismatch: {path}")
    detached_path = path.with_suffix(path.suffix + ".sha256")
    if detached_path.is_symlink() or not detached_path.is_file():
        raise T8ChainPointerError(f"receipt detached SHA is missing: {path}")
    if detached_path.read_text(encoding="ascii").strip() != sha256_file(path):
        raise T8ChainPointerError(f"receipt detached SHA mismatch: {path}")
    return dict(payload)


def parse_array_sacct(
    text: str, *, array_job_id: str, array_range: str
) -> dict[str, Any]:
    """Classify exact Slurm array elements without counting batch/extern rows."""

    if not JOB_ID_RE.fullmatch(array_job_id):
        raise T8ChainPointerError("array job ID is invalid")
    match = ARRAY_RANGE_RE.fullmatch(array_range)
    if match is None:
        raise T8ChainPointerError("array range is invalid")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end < start:
        raise T8ChainPointerError("array range is reversed")
    expected = set(range(start, end + 1))
    states: dict[int, str] = {}
    collapsed_pending: set[int] = set()
    exact_re = re.compile(rf"^{re.escape(array_job_id)}_(?P<index>[0-9]+)$")
    range_re = re.compile(
        rf"^{re.escape(array_job_id)}_\[(?P<start>[0-9]+)-(?P<end>[0-9]+)(?:%[0-9]+)?\]$"
    )
    for raw_line in text.splitlines():
        columns = raw_line.strip().split("|")
        if len(columns) < 2 or not columns[0]:
            continue
        job, state = columns[0], columns[1].split()[0].split("+")[0].upper()
        exact = exact_re.fullmatch(job)
        if exact is not None:
            index = int(exact.group("index"))
            if index not in expected:
                raise T8ChainPointerError("sacct contains an out-of-range array task")
            previous = states.get(index)
            if previous is not None and previous != state:
                raise T8ChainPointerError("sacct contains conflicting task states")
            states[index] = state
            continue
        collapsed = range_re.fullmatch(job)
        if collapsed is not None:
            lo = int(collapsed.group("start"))
            hi = int(collapsed.group("end"))
            indices = set(range(lo, hi + 1))
            if state != "PENDING" or not indices.issubset(expected):
                raise T8ChainPointerError("collapsed array row is invalid")
            collapsed_pending.update(indices)

    for index in collapsed_pending:
        states.setdefault(index, "PENDING")
    missing = expected.difference(states)
    if missing:
        raise T8ChainPointerError(
            "sacct is missing array tasks: " + ",".join(map(str, sorted(missing)))
        )

    running_states = {"RUNNING", "COMPLETING", "CONFIGURING"}
    pending_states = {"PENDING"}
    failed_states = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
    unknown = {
        state
        for state in states.values()
        if state not in running_states | pending_states | failed_states | {"COMPLETED"}
    }
    if unknown:
        raise T8ChainPointerError(
            "sacct contains unsupported states: " + ",".join(sorted(unknown))
        )
    completed = sorted(index for index, state in states.items() if state == "COMPLETED")
    running = sorted(index for index, state in states.items() if state in running_states)
    pending = sorted(index for index, state in states.items() if state in pending_states)
    failed = sorted(index for index, state in states.items() if state in failed_states)
    return {
        "shard_count": len(expected),
        "completed_shards": len(completed),
        "running_shards": len(running),
        "pending_shards": len(pending),
        "failed_shards": len(failed),
        "completed_shard_ids": completed,
        "running_shard_ids": running,
        "pending_shard_ids": pending,
        "failed_shard_ids": failed,
    }


def _export_values(argv: Any) -> dict[str, str]:
    if type(argv) is not list:
        raise T8ChainPointerError("array sbatch argv is malformed")
    try:
        export_value = argv[argv.index("--export") + 1]
    except (ValueError, IndexError) as exc:
        raise T8ChainPointerError("array sbatch export contract is missing") from exc
    if not isinstance(export_value, str):
        raise T8ChainPointerError("array sbatch export contract is malformed")
    values: dict[str, str] = {}
    for token in export_value.split(","):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def reconcile_full_chain_pointer(
    *,
    current_pointer: Path,
    submission_receipt: Path,
    slurm_inventory: Path,
    admission_receipt: Path,
    sacct_text: str,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Rebuild the production pointer from immutable receipts plus live Slurm state.

    This function never submits, cancels, or requeues a job.  It only publishes
    one atomic, self-hashed status pointer after all IDs and dependency edges
    agree across the controller receipts.
    """

    submission = _load_self_hashed_json(
        submission_receipt, hash_field="submission_receipt_sha256"
    )
    inventory = _load_self_hashed_json(
        slurm_inventory, hash_field="slurm_inventory_sha256"
    )
    admission = _load_self_hashed_json(
        admission_receipt, hash_field="admission_receipt_sha256"
    )
    if (
        submission.get("schema_version") != "t8_hpc_stress_followup_v1"
        or submission.get("state") != "FULL_CHAIN_SUBMITTED"
        or submission.get("action") != "SUBMIT_ARRAY_AFTEROK_MERGE_AFTEROK_PACKAGE"
        or inventory.get("schema_version") != "t8_hpc_slurm_inventory_v1"
        or inventory.get("state") != "PASS"
        or admission.get("schema_version") != "t8_hpc_full_admission_v1"
        or admission.get("state") != "PASS"
        or admission.get("admission_pass") is not True
    ):
        raise T8ChainPointerError("full-chain receipt state is not publishable")

    array_job_id = str(submission.get("array_job_id", ""))
    merge_job_id = str(submission.get("merge_job_id", ""))
    package_job_id = str(submission.get("package_job_id", ""))
    if any(
        not JOB_ID_RE.fullmatch(value)
        for value in (array_job_id, merge_job_id, package_job_id)
    ):
        raise T8ChainPointerError("full-chain receipt has invalid Slurm IDs")
    if (
        str(inventory.get("array_job_id")) != array_job_id
        or str(inventory.get("merge_job_id")) != merge_job_id
        or str(inventory.get("package_job_id")) != package_job_id
        or inventory.get("dependency_chain")
        != [
            {"dependency": f"afterok:{array_job_id}", "job_id": merge_job_id},
            {"dependency": f"afterok:{merge_job_id}", "job_id": package_job_id},
        ]
    ):
        raise T8ChainPointerError("full-chain dependency receipts disagree")

    full_manifest = Path(str(submission.get("full_manifest", "")))
    if (
        full_manifest.is_symlink()
        or not full_manifest.is_file()
        or str(admission.get("full_manifest_path")) != str(full_manifest)
        or sha256_file(full_manifest) != admission.get("full_manifest_file_sha256")
    ):
        raise T8ChainPointerError("partition manifest binding is invalid")
    export = _export_values(submission.get("array_sbatch_argv"))
    required_export = {
        "T8_EXPECTED_COMMIT",
        "T8_EXPECTED_INPUT_MANIFEST_SHA256",
        "T8_CANARY_PARITY_RECEIPT",
    }
    if not required_export.issubset(export):
        raise T8ChainPointerError("full-chain science export is incomplete")
    if export["T8_EXPECTED_COMMIT"] != submission.get("science_commit"):
        raise T8ChainPointerError("science commit differs from array export")

    array_state = parse_array_sacct(
        sacct_text,
        array_job_id=array_job_id,
        array_range=str(inventory.get("array_range", "")),
    )
    range_start, range_end = map(
        int, str(inventory["array_range"]).split("-", 1)
    )
    if array_state["shard_count"] != range_end - range_start + 1:
        raise T8ChainPointerError("array shard count binding changed")
    timestamp = updated_at or datetime.now(timezone.utc).isoformat()
    state = (
        "FULL_CHAIN_FAILED"
        if array_state["failed_shards"]
        else "FULL_CHAIN_ARRAY_PASS"
        if array_state["completed_shards"] == array_state["shard_count"]
        else "FULL_CHAIN_RUNNING"
    )
    payload: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA,
        "state": state,
        "active_stage": "FULL_CHAIN",
        "updated_at": timestamp,
        "output_root": str(submission_receipt.parent.parent),
        "continuation_root": str(Path(str(submission["full_root"])).parent),
        "decision_root": str(submission_receipt.parent),
        "upstream_job_id": str(submission.get("upstream_terminal", {}).get("job_id", "")),
        "controller_commit": str(submission["controller_commit"]),
        "science_commit": str(submission["science_commit"]),
        "array_job_id": array_job_id,
        "merge_job_id": merge_job_id,
        "package_job_id": package_job_id,
        "array_dependency": None,
        "merge_dependency": f"afterok:{array_job_id}",
        "package_dependency": f"afterok:{merge_job_id}",
        "full_root": str(submission["full_root"]),
        "partition_manifest": str(full_manifest),
        "partition_manifest_sha256": str(admission["full_manifest_sha256"]),
        "partition_manifest_file_sha256": str(admission["full_manifest_file_sha256"]),
        "input_manifest_sha256": export["T8_EXPECTED_INPUT_MANIFEST_SHA256"],
        "parity_receipt": export["T8_CANARY_PARITY_RECEIPT"],
        "submission_receipt": str(submission_receipt),
        "admission_receipt": str(admission_receipt),
        "matrix_write_enabled": False,
        "gpu_requested": False,
        **array_state,
    }
    if not JOB_ID_RE.fullmatch(payload["upstream_job_id"]):
        raise T8ChainPointerError("upstream terminal job ID is invalid")
    with chain_lock(current_pointer):
        return write_current_pointer(current_pointer, payload)
