"""Exact, transport-only adoption contract for a committed T14 checkpoint.

The scientific checkpoint remains bound to its original execution commit.
This module permits a later clean commit to change only checkpoint transport
(single-pass loading, consumptive restoration, and tensor-backed future
serialization) while proving that every scientific identity field is equal.
It does not weaken checkpoint hashes or authorize a science launch by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


RESUME_SPEC_SCHEMA = "tastemolnet_t14_low_memory_resume_spec_v1"
RESUME_BINDING_SCHEMA = "tastemolnet_t14_low_memory_resume_binding_v1"
MEMORY_ADMISSION_SCHEMA = "tastemolnet_t14_memory_admission_v1"
CANARY_RECEIPT_SCHEMA = "tastemolnet_t14_low_memory_canary_receipt_v1"
SOURCE_STEP = 12_500
SAFETY_MARGIN_BYTES = 64 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class T14ResumeError(RuntimeError):
    """The T14 resume transport or memory contract is incomplete."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """One exact Linux process identity, immune to PID reuse."""

    pid: int
    start_ticks: int
    state: str

    @property
    def live(self) -> bool:
        return self.state not in {"MISSING", "Z"}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T14ResumeError(f"T14 resume JSON is unreadable: {path}") from exc
    if type(value) is not dict:
        raise T14ResumeError(f"T14 resume JSON must be one object: {path}")
    return value


def _physical(path: str | Path, *, kind: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.is_symlink():
        raise T14ResumeError(f"T14 {kind} must be one absolute physical path")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise T14ResumeError(f"T14 {kind} is unavailable: {value}") from exc
    if resolved != value:
        raise T14ResumeError(f"T14 {kind} contains an alias: {value}")
    return value


def _self_hash(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes({key: item for key, item in value.items() if key != "spec_sha256"})
    )


def _scientific_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dict(identity.get("provenance") or {})
    provenance.pop("execution_commit", None)
    provenance.pop("scientific_command_sha256", None)
    argv = [
        value
        for value in identity.get("scientific_argv") or ()
        if not str(value).startswith("execution_commit=")
    ]
    return {
        "schema_version": identity.get("schema_version"),
        "status": identity.get("status"),
        "provenance": provenance,
        "scientific_argv_without_execution_commit": argv,
        "total_steps": identity.get("total_steps"),
        "checkpoint_interval": identity.get("checkpoint_interval"),
        "transition_expanded_capacity": identity.get(
            "transition_expanded_capacity"
        ),
        "raw_neighbor_graphs_retained_unbounded": identity.get(
            "raw_neighbor_graphs_retained_unbounded"
        ),
    }


def build_resume_spec(
    *,
    output_root: str | Path,
    checkpoint_dir: str | Path,
    resume_execution_commit: str,
    historical_process_peak_bytes: int,
    historical_checkpoint_peak_bytes: int,
) -> dict[str, Any]:
    """Hash-close the exact 12,500 checkpoint and its source identity."""

    root = _physical(output_root, kind="output root")
    checkpoint = _physical(checkpoint_dir, kind="checkpoint directory")
    if checkpoint.parent != root / "checkpoints" or checkpoint.name != (
        "step-000000012500"
    ):
        raise T14ResumeError("T14 resume source must be the committed 12,500 checkpoint")
    if _GIT_SHA.fullmatch(str(resume_execution_commit)) is None:
        raise T14ResumeError("T14 resume execution commit is malformed")
    identity_path = root / "checkpoint_identity.json"
    identity = _json(_physical(identity_path, kind="checkpoint identity"))
    source_commit = str((identity.get("provenance") or {}).get("execution_commit") or "")
    if _GIT_SHA.fullmatch(source_commit) is None:
        raise T14ResumeError("T14 source execution commit is malformed")

    from src.baselines.comrecgc.generation_checkpoint import (
        validate_generation_checkpoint_envelope,
    )

    validation = validate_generation_checkpoint_envelope(
        checkpoint,
        expected_provenance=identity.get("provenance"),
        expected_scientific_argv=identity.get("scientific_argv"),
        expected_command_sha256=identity.get("command_sha256"),
        expected_total_steps=25_000,
        expected_completed_step=SOURCE_STEP,
    )
    process_peak = int(historical_process_peak_bytes)
    checkpoint_peak = int(historical_checkpoint_peak_bytes)
    if process_peak <= 0 or checkpoint_peak <= 0:
        raise T14ResumeError("T14 historical memory evidence must be positive")
    required = max(
        checkpoint_peak + SAFETY_MARGIN_BYTES,
        math.ceil(process_peak * 1.15),
    )
    payload: dict[str, Any] = {
        "schema_version": RESUME_SPEC_SCHEMA,
        "status": "AUTHORIZED_TRANSPORT_ONLY_PENDING_MEMORY_CANARY",
        "source_output_root": str(root),
        "checkpoint_root": str(checkpoint.parent),
        "checkpoint_dir": str(checkpoint),
        "completed_step": SOURCE_STEP,
        "next_step": SOURCE_STEP + 1,
        "checkpoint_digest": validation.checkpoint_digest,
        "checkpoint_manifest_sha256": _sha256_file(
            checkpoint / "checkpoint_manifest.json"
        ),
        "generation_state_sha256": validation.manifest["files"][
            "generation_state.pt"
        ]["sha256"],
        "generation_state_bytes": validation.manifest["files"][
            "generation_state.pt"
        ]["bytes"],
        "sqlite_snapshot_sha256": validation.manifest["files"][
            "authoritative_graph_store.sqlite3"
        ]["sha256"],
        "checkpoint_identity_path": str(identity_path),
        "checkpoint_identity_sha256": _sha256_file(identity_path),
        "source_execution_commit": source_commit,
        "resume_execution_commit": str(resume_execution_commit),
        "scientific_projection_sha256": _sha256_bytes(
            _canonical_bytes(_scientific_projection(identity))
        ),
        "transport_changes": [
            "single_pass_checkpoint_deserialize",
            "consumptive_state_transfer",
            "tensor_backed_future_checkpoint_numerics",
            "writer_envelope_validation_without_live_state_reload",
        ],
        "scientific_state_changes": False,
        "rng_changes": False,
        "candidate_order_changes": False,
        "auditor_concurrency_policy": "SERIAL_ONLY",
        "full_state_consumer_limit": 1,
        "full_state_lock_path": str(
            checkpoint.parent / ".t14-full-state-consumer.lock"
        ),
        "resume_parity_canary_max_steps": 50,
        "memory": {
            "schema_version": MEMORY_ADMISSION_SCHEMA,
            "historical_process_peak_bytes": process_peak,
            "historical_checkpoint_peak_bytes": checkpoint_peak,
            "safety_margin_bytes": SAFETY_MARGIN_BYTES,
            "historical_required_headroom_bytes": required,
            "optimized_canary_receipt_required": True,
            "cgroup_limit_path": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "cgroup_current_path": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "cgroup_failcnt_path": "/sys/fs/cgroup/memory/memory.failcnt",
        },
    }
    payload["spec_sha256"] = _self_hash(payload)
    return payload


def write_resume_spec(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.is_absolute() or destination.is_symlink():
        raise T14ResumeError("T14 resume spec destination must be absolute")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_resume_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _physical(path, kind="resume spec")
    value = _json(spec_path)
    if (
        value.get("schema_version") != RESUME_SPEC_SCHEMA
        or value.get("spec_sha256") != _self_hash(value)
        or value.get("completed_step") != SOURCE_STEP
        or value.get("next_step") != SOURCE_STEP + 1
        or value.get("scientific_state_changes") is not False
        or value.get("rng_changes") is not False
        or value.get("candidate_order_changes") is not False
        or value.get("auditor_concurrency_policy") != "SERIAL_ONLY"
        or value.get("full_state_consumer_limit") != 1
        or value.get("full_state_lock_path")
        != str(Path(str(value.get("checkpoint_root"))) / ".t14-full-state-consumer.lock")
        or value.get("resume_parity_canary_max_steps") != 50
        or _SHA256.fullmatch(str(value.get("checkpoint_digest") or "")) is None
        or _SHA256.fullmatch(str(value.get("generation_state_sha256") or ""))
        is None
    ):
        raise T14ResumeError("T14 resume spec is invalid")
    return value


def load_canary_receipt(
    path: str | Path,
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a hash-closed, same-checkpoint parity/memory receipt."""

    receipt_path = _physical(path, kind="optimized canary receipt")
    value = _json(receipt_path)
    detached_hash = value.get("receipt_sha256")
    calculated_hash = _sha256_bytes(
        _canonical_bytes(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    )
    if detached_hash != calculated_hash:
        raise T14ResumeError("T14 optimized canary receipt hash changed")
    # Keep the complete semantic gate in one place.  The result is discarded;
    # the owner repeats admission against live cgroup counters immediately
    # before launch.
    evaluate_memory_admission(
        spec,
        cgroup_limit_bytes=int(value.get("cgroup_limit_bytes", -1)),
        cgroup_current_bytes=int(value.get("cgroup_baseline_current_bytes", -1)),
        optimized_canary_receipt=value,
    )
    return value


def read_cgroup_counter(path: str | Path, *, allow_max: bool = False) -> int:
    """Read one physical cgroup-v1/v2 byte counter without shell expansion."""

    counter_path = _physical(path, kind="cgroup counter")
    try:
        text = counter_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise T14ResumeError(f"T14 cgroup counter is unreadable: {counter_path}") from exc
    if allow_max and text == "max":
        raise T14ResumeError("T14 requires a finite cgroup memory limit")
    if not text.isdigit():
        raise T14ResumeError(f"T14 cgroup counter is malformed: {counter_path}")
    value = int(text)
    if value < 0:
        raise T14ResumeError(f"T14 cgroup counter is negative: {counter_path}")
    return value


def inspect_process_identity(
    pid: int,
    *,
    proc_root: str | Path = "/proc",
) -> ProcessIdentity:
    """Read the exact start tick and state for a Linux PID.

    ``/proc/<pid>/stat`` puts the command in parentheses and permits spaces, so
    fields are parsed only after the final ``)``.  Missing processes are a
    normal state used to prove that the frozen full-state auditor is gone.
    """

    normalized_pid = int(pid)
    if normalized_pid <= 0:
        raise T14ResumeError("T14 process PID must be positive")
    root = Path(proc_root)
    stat_path = root / str(normalized_pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ProcessIdentity(pid=normalized_pid, start_ticks=-1, state="MISSING")
    except OSError as exc:
        raise T14ResumeError(f"T14 cannot inspect process {normalized_pid}") from exc
    closing = raw.rfind(")")
    tail = raw[closing + 2 :].split() if closing >= 0 else []
    # tail[0] is field 3 (state), tail[19] is field 22 (starttime).
    if len(tail) <= 19 or len(tail[0]) != 1 or not tail[19].isdigit():
        raise T14ResumeError(f"T14 process stat is malformed for PID {normalized_pid}")
    return ProcessIdentity(
        pid=normalized_pid,
        start_ticks=int(tail[19]),
        state=tail[0],
    )


def assert_auditor_serialized(
    *,
    auditor_pid: int,
    auditor_start_ticks: int,
    proc_root: str | Path = "/proc",
) -> ProcessIdentity:
    """Fail if the exact frozen full-state auditor is still resident."""

    expected_ticks = int(auditor_start_ticks)
    if expected_ticks <= 0:
        raise T14ResumeError("T14 auditor start ticks must be positive")
    observed = inspect_process_identity(auditor_pid, proc_root=proc_root)
    if observed.live and observed.start_ticks == expected_ticks:
        raise T14ResumeError("T14 full-state auditor is still live; SERIAL_ONLY blocks science")
    return observed


def bind_resume_identity(
    *,
    spec_path: str | Path,
    output_root: str | Path,
    current_execution_commit: str,
    current_checkpoint_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the frozen source identity only after exact transport adoption."""

    spec = load_resume_spec(spec_path)
    root = _physical(output_root, kind="output root")
    if (
        spec.get("source_output_root") != str(root)
        or spec.get("checkpoint_root") != str(root / "checkpoints")
        or spec.get("checkpoint_dir")
        != str(root / "checkpoints" / "step-000000012500")
        or spec.get("resume_execution_commit") != current_execution_commit
    ):
        raise T14ResumeError("T14 resume spec does not bind the current execution")
    identity_path = _physical(
        spec["checkpoint_identity_path"], kind="checkpoint identity"
    )
    if identity_path != root / "checkpoint_identity.json":
        raise T14ResumeError("T14 resume identity path changed")
    source_identity = _json(identity_path)
    if (
        _sha256_file(identity_path) != spec.get("checkpoint_identity_sha256")
        or (source_identity.get("provenance") or {}).get("execution_commit")
        != spec.get("source_execution_commit")
        or _sha256_bytes(_canonical_bytes(_scientific_projection(source_identity)))
        != spec.get("scientific_projection_sha256")
        or _scientific_projection(source_identity)
        != _scientific_projection(current_checkpoint_identity)
    ):
        raise T14ResumeError("T14 scientific identity changed across transport commits")
    latest = _json(root / "checkpoints" / "LATEST")
    if (
        latest.get("checkpoint_dir") != "step-000000012500"
        or latest.get("completed_step") != SOURCE_STEP
        or latest.get("checkpoint_digest") != spec.get("checkpoint_digest")
    ):
        raise T14ResumeError("T14 LATEST no longer names the adopted checkpoint")
    receipt = {
        "schema_version": RESUME_BINDING_SCHEMA,
        "status": "PASS",
        "source_execution_commit": spec["source_execution_commit"],
        "resume_execution_commit": current_execution_commit,
        "checkpoint_digest": spec["checkpoint_digest"],
        "checkpoint_identity_sha256": spec["checkpoint_identity_sha256"],
        "scientific_projection_sha256": spec["scientific_projection_sha256"],
        "transport_changes": list(spec["transport_changes"]),
        "scientific_state_changes": False,
        "rng_changes": False,
        "candidate_order_changes": False,
        "full_state_consumers_serialized": True,
    }
    return source_identity, receipt


@dataclass(frozen=True, slots=True)
class MemoryAdmission:
    state: str
    required_headroom_bytes: int
    available_headroom_bytes: int
    basis: str

    @property
    def admitted(self) -> bool:
        return self.state == "PASS"


def evaluate_memory_admission(
    spec: Mapping[str, Any],
    *,
    cgroup_limit_bytes: int,
    cgroup_current_bytes: int,
    optimized_canary_receipt: Mapping[str, Any] | None = None,
) -> MemoryAdmission:
    """Apply the historical gate until a same-checkpoint optimized canary passes."""

    memory = spec.get("memory")
    if type(memory) is not dict or memory.get("schema_version") != MEMORY_ADMISSION_SCHEMA:
        raise T14ResumeError("T14 resume memory contract changed")
    required = int(memory.get("historical_required_headroom_bytes", -1))
    basis = "HISTORICAL_CHECKPOINT_PEAK_PLUS_64_GIB"
    if optimized_canary_receipt is not None:
        receipt = dict(optimized_canary_receipt)
        steps = int(receipt.get("steps", 0))
        reference_digest = str(receipt.get("reference_final_state_sha256") or "")
        optimized_digest = str(receipt.get("optimized_final_state_sha256") or "")
        checkpoint_state_digest = str(
            receipt.get("forced_checkpoint_state_sha256") or ""
        )
        if (
            receipt.get("schema_version") != CANARY_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("checkpoint_digest") != spec.get("checkpoint_digest")
            or receipt.get("resume_spec_sha256") != spec.get("spec_sha256")
            or receipt.get("generation_state_sha256")
            != spec.get("generation_state_sha256")
            or receipt.get("resume_execution_commit")
            != spec.get("resume_execution_commit")
            or receipt.get("source_completed_step") != SOURCE_STEP
            or receipt.get("start_step") != SOURCE_STEP + 1
            or receipt.get("end_step") != SOURCE_STEP + steps
            or receipt.get("semantic_parity_pass") is not True
            or receipt.get("checkpoint_reload_pass") is not True
            or receipt.get("checkpoint_save_pass") is not True
            or receipt.get("forced_checkpoint_step") != SOURCE_STEP + steps
            or receipt.get("first_semantic_divergence_step") is not None
            or receipt.get("step_state_digests_equal") is not True
            or receipt.get("rng_state_equal") is not True
            or receipt.get("candidate_registry_equal") is not True
            or receipt.get("lineage_equal") is not True
            or receipt.get("scientific_config_equal") is not True
            or receipt.get("cgroup_failcnt_delta") != 0
            or receipt.get("cgroup_oom_kill_delta") != 0
            or receipt.get("cgroup_limit_path") != memory.get("cgroup_limit_path")
            or receipt.get("cgroup_current_path") != memory.get("cgroup_current_path")
            or receipt.get("cgroup_failcnt_path") != memory.get("cgroup_failcnt_path")
            or int(receipt.get("cgroup_limit_bytes", -1))
            != int(cgroup_limit_bytes)
            or not 0
            <= int(receipt.get("cgroup_baseline_current_bytes", -1))
            <= int(receipt.get("cgroup_peak_current_bytes", -1))
            <= int(cgroup_limit_bytes)
            or int(receipt.get("science_pid", 0)) <= 0
            or int(receipt.get("science_start_ticks", 0)) <= 0
            or int(receipt.get("reload_verifier_pid", 0)) <= 0
            or int(receipt.get("reload_verifier_start_ticks", 0)) <= 0
            or receipt.get("reload_verifier_pid") == receipt.get("science_pid")
            or _SHA256.fullmatch(reference_digest) is None
            or optimized_digest != reference_digest
            or checkpoint_state_digest != optimized_digest
            or type(receipt.get("torch_version")) is not str
            or not receipt.get("torch_version")
            or type(receipt.get("mmap_effective")) is not bool
            or not 1 <= steps <= 50
        ):
            raise T14ResumeError("T14 optimized memory canary receipt is invalid")
        resume_peak = int(receipt.get("resume_peak_bytes", -1))
        checkpoint_peak = int(receipt.get("checkpoint_peak_bytes", -1))
        if resume_peak <= 0 or checkpoint_peak <= 0:
            raise T14ResumeError("T14 optimized canary memory evidence is missing")
        required = max(
            checkpoint_peak + SAFETY_MARGIN_BYTES,
            math.ceil(resume_peak * 1.15),
        )
        basis = "OPTIMIZED_PARITY_CANARY_PEAK"
    limit = int(cgroup_limit_bytes)
    current = int(cgroup_current_bytes)
    if limit <= 0 or current < 0 or current > limit:
        raise T14ResumeError("T14 cgroup memory counters are invalid")
    available = limit - current
    return MemoryAdmission(
        state="PASS" if available >= required else "WAITING_MEMORY_HEADROOM",
        required_headroom_bytes=required,
        available_headroom_bytes=available,
        basis=basis,
    )


__all__ = [
    "CANARY_RECEIPT_SCHEMA",
    "MemoryAdmission",
    "ProcessIdentity",
    "RESUME_SPEC_SCHEMA",
    "T14ResumeError",
    "assert_auditor_serialized",
    "bind_resume_identity",
    "build_resume_spec",
    "evaluate_memory_admission",
    "inspect_process_identity",
    "load_canary_receipt",
    "load_resume_spec",
    "read_cgroup_counter",
    "write_resume_spec",
]
