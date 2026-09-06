"""Mut-only common observer/checkpoint commits; never opens a state payload.

The algorithm checkpoint and observer ledger have separate writers.  A joint
receipt is committed only after both have durably recorded the same *completed*
1-based step.  This does not retroactively invent an absent observer event.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4


SCHEMA = "mut_common_checkpoint_boundary_v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _physical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing physical boundary JSON: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Boundary JSON is not an object: {path}")
    return value, raw


def checkpoint_binding(checkpoint: Path, *, step: int) -> dict[str, Any]:
    """Read only the small manifest/marker, never SQLite or generation_state.pt."""
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Checkpoint directory must be physical")
    manifest, raw = _physical_json(checkpoint / "checkpoint_manifest.json")
    complete, _ = _physical_json(checkpoint / "_CHECKPOINT_COMPLETE.json")
    if (manifest.get("schema_version") != "comrecgc_generation_checkpoint_v2"
            or manifest.get("atomic_complete") is not True
            or manifest.get("boundary") != "after_fully_completed_step_v1"
            or manifest.get("completed_step") != step
            or manifest.get("next_step") != step + 1
            or step < 1
            or manifest.get("checkpoint_dir") != checkpoint.name):
        raise ValueError("Algorithm checkpoint completed-step boundary differs")
    digest = manifest.get("checkpoint_digest")
    checkpoint_self = _sha(json.dumps(
        {k: v for k, v in manifest.items() if k != "checkpoint_digest"},
        sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str,
    ).encode())
    if digest != checkpoint_self:
        raise ValueError("Algorithm checkpoint manifest self hash differs")
    if (complete.get("checkpoint_digest") != digest
            or complete.get("manifest_sha256") != _sha(raw)):
        raise ValueError("Algorithm checkpoint completion marker differs")
    return {"path": str(checkpoint.resolve()), "manifest_sha256": _sha(raw),
            "checkpoint_digest": digest, "completed_step": step,
            "next_step": step + 1, "command_sha256": manifest["command_sha256"],
            "provenance_sha256": manifest["provenance_sha256"],
            "payload_reload_verified": False}


def observer_prefix(path: Path, *, phase: str, trace_mode: str, stop_step: int,
                    science_digest: Callable[[Mapping[str, Any]], str],
                    require_exact_end: bool = True) -> dict[str, Any]:
    """Validate the actual complete 1-based rows and history, without inference."""
    if path.is_symlink() or not path.is_file() or stop_step < 1:
        raise ValueError("Physical nonempty observer prefix required")
    file_digest = hashlib.sha256()
    history = "0" * 64
    count = 0
    prefix_bytes = 0
    last: dict[str, Any] | None = None
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise ValueError("Observer has a partial row; never truncate the source")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("Observer row is not an object")
            if count == stop_step:
                if require_exact_end:
                    raise ValueError("Observer suffix extends beyond the committed boundary")
                break
            count += 1
            if (row.get("schema_version") != "mut_trace_common_step_state_v1"
                    or row.get("phase") != phase or row.get("trace_mode") != trace_mode
                    or row.get("step") != count or row.get("next_step") != count + 1):
                raise ValueError("Observer step/phase/mode sequence is not contiguous from 1")
            digest = science_digest(row)
            if digest != row.get("scientific_checkpoint_digest"):
                raise ValueError("Observer scientific digest differs")
            history = _sha(bytes.fromhex(history) + bytes.fromhex(digest))
            if history != row.get("history_digest"):
                raise ValueError("Observer history chain differs")
            file_digest.update(raw)
            prefix_bytes += len(raw)
            last = row
    if count != stop_step or last is None:
        raise ValueError(f"Observer missing completed step {stop_step}; no inferred event")
    return {"path": str(path.resolve()), "phase": phase, "trace_mode": trace_mode,
            "row_count": count, "prefix_bytes": prefix_bytes,
            "prefix_sha256": file_digest.hexdigest(), "completed_step": stop_step,
            "next_step": stop_step + 1, "history_digest": history,
            "scientific_checkpoint_digest": last["scientific_checkpoint_digest"]}


def _publish_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        existing, _ = _physical_json(path)
        if existing != dict(value):
            raise ValueError("Existing joint boundary differs; source will not be overwritten")
        return
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # This small receipt has the same single-writer ownership as the ledger.
    # Atomic rename is supported by the existing AutoDL persistence backend.
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def seal_joint_boundary(*, checkpoint: Path, observer: Path, step: int,
                        phase: str, trace_mode: str,
                        science_digest: Callable[[Mapping[str, Any]], str]) -> Path:
    if phase != "continuous":
        raise ValueError("Joint commits bind the continuous arm, not diagnostic reload rows")
    value = {"schema_version": SCHEMA, "status": "COMMON_BOUNDARY_COMMITTED",
             "step_numbering": "completed_1_based_next_equals_plus_one",
             "checkpoint": checkpoint_binding(checkpoint, step=step),
             "observer": observer_prefix(observer, phase=phase, trace_mode=trace_mode,
                                          stop_step=step, science_digest=science_digest),
             "source_event_reconstructed": False,
             "resource_failure_triggers_route_b": False}
    value["receipt_sha256"] = _stable(value)
    path = observer.parent / "common_boundaries" / f"step-{step:012d}.json"
    _publish_new_json(path, value)
    return path


def reopen_joint_boundary(*, receipt: Path, checkpoint: Path, observer: Path,
                          trace_mode: str,
                          science_digest: Callable[[Mapping[str, Any]], str],
                          require_exact_end: bool = True) -> dict[str, Any]:
    value, _ = _physical_json(receipt)
    if (value.get("schema_version") != SCHEMA
            or value.get("status") != "COMMON_BOUNDARY_COMMITTED"
            or value.get("source_event_reconstructed") is not False
            or value.get("resource_failure_triggers_route_b") is not False
            or value.get("receipt_sha256") != _stable(
                {k: v for k, v in value.items() if k != "receipt_sha256"})):
        raise ValueError("Joint boundary receipt is invalid")
    step = int(value["checkpoint"]["completed_step"])
    if checkpoint_binding(checkpoint, step=step) != value["checkpoint"]:
        raise ValueError("Joint checkpoint binding differs")
    reopened = observer_prefix(observer, phase="continuous", trace_mode=trace_mode,
                               stop_step=step, science_digest=science_digest,
                               require_exact_end=require_exact_end)
    if reopened != value["observer"]:
        raise ValueError("Joint observer binding differs")
    return value


def recovery_plan(*, checkpoint_steps: list[int], valid_observer_through: int,
                  joint_committed_steps: list[int]) -> dict[str, Any]:
    """Facts supplied by the caller are not an execution authorization."""
    eligible = sorted(set(checkpoint_steps).intersection(joint_committed_steps))
    eligible = [step for step in eligible if 0 < step <= valid_observer_through]
    common = max(eligible, default=0)
    latest_algorithm = max(checkpoint_steps, default=0)
    return {"status": "PLAN_ONLY_RESOURCE_ADMISSION_REQUIRED",
            "common_completed_step": common, "next_step": common + 1,
            "resume_existing_checkpoint": common > 0,
            "replay_range_to_last_algorithm_checkpoint": (
                [common + 1, latest_algorithm] if common < latest_algorithm else []),
            "missing_observer_steps": ([valid_observer_through + 1, latest_algorithm]
                                       if valid_observer_through < latest_algorithm else []),
            "old_output_modified": False, "scientific_failure": False,
            "route_b_eligible": False, "science_launch_allowed": False,
            "minimum_free_inodes": 100_000}
