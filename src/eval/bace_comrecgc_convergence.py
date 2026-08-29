"""Read-only convergence audit for the live BACE ComRecGC trajectory.

The audit deliberately consumes only small JSON checkpoint receipts and CLOSED
selected-action trace chunks.  It never deserializes ``generation_state.pt``,
never opens the authoritative SQLite payload, never writes below a running
generation/checkpoint root, and has no process-signalling API.

The public entry point is :func:`audit_bace_comrecgc_convergence`.  Callers
must provide the two frozen SHA-256 identities from ``resolved_config.json``
instead of learning authority from the live run being audited.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.baselines.comrecgc.contracts import stable_json_sha256


AUDIT_SCHEMA_VERSION = "bace_comrecgc_convergence_audit_v1"
RECEIPT_SCHEMA_VERSION = "bace_comrecgc_convergence_receipt_v1"
CHECKPOINT_SCHEMA_VERSION = "comrecgc_generation_checkpoint_v2"
CHECKPOINT_BOUNDARY = "after_fully_completed_step_v1"
CHECKPOINT_STATE_SCHEMA_VERSION = "comrecgc_generation_state_v2"
CHECKPOINT_COMPLETE_FILENAME = "_CHECKPOINT_COMPLETE.json"
CHECKPOINT_MIRRORED_FILENAME = "_CHECKPOINT_MIRRORED.json"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
CHECKPOINT_STATE_FILENAME = "generation_state.pt"
CHECKPOINT_SQLITE_FILENAME = "authoritative_graph_store.sqlite3"
RETENTION_HISTORY_DIRNAME = "retention_history"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_NAME = re.compile(r"step-(?P<step>[0-9]{12})")
_TRACE_PART_NAME = re.compile(r"part-(?P<index>[0-9]{6})\.jsonl")


class BaceComRecGCConvergenceError(RuntimeError):
    """The convergence evidence is malformed or violates the frozen contract."""


class _EvidenceNotReady(BaceComRecGCConvergenceError):
    """The requested committed/closed evidence has not been published yet."""


@dataclass(frozen=True, slots=True)
class ConvergencePolicy:
    """The preregistered BACE ComRecGC early-stop policy."""

    m_max: int = 50_000
    m_min: int = 10_000
    check_interval: int = 2_500
    patience_checks: int = 2
    top100_jaccard_min: float = 0.99
    top20_jaccard_min: float = 0.95
    rank_spearman_min: float = 0.99
    absolute_train_coverage_gain_max: float = 0.005
    minimum_valid_unique_count: int = 20
    parent_count: int = 360
    heads: int = 5
    trace_chunk_rows: int = 512
    top100_size: int = 100
    top20_size: int = 20
    missing_rank: int = 101


POLICY = ConvergencePolicy()


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _CheckpointEvidence:
    step: int
    checkpoint_digest: str
    kind: str
    evidence_paths: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    provenance_fingerprints: Mapping[str, str] | None
    declared_payload_bytes: int | None


@dataclass(frozen=True, slots=True)
class _TraceScan:
    closed_chunks: tuple[dict[str, Any], ...]
    closed_row_count: int
    closed_through_move: int
    move_rows: Mapping[int, tuple[dict[str, Any], ...]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BaceComRecGCConvergenceError(
            f"{field} must be one lowercase 64-character SHA-256 digest"
        )
    return value


def _identity(path: Path) -> _FileIdentity:
    try:
        value = path.lstat()
    except OSError as exc:
        raise BaceComRecGCConvergenceError(f"evidence path is missing: {path}") from exc
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise BaceComRecGCConvergenceError(
            f"evidence path must be a physical regular file: {path}"
        )
    return _FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        links=int(value.st_nlink),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _read_stable_bytes(path: Path, *, maximum_bytes: int) -> tuple[bytes, str]:
    before = _identity(path)
    if before.size <= 0 or before.size > int(maximum_bytes):
        raise BaceComRecGCConvergenceError(
            f"evidence file has an invalid size: path={path}, bytes={before.size}"
        )
    try:
        with path.open("rb") as handle:
            payload = handle.read(int(maximum_bytes) + 1)
    except OSError as exc:
        raise BaceComRecGCConvergenceError(f"cannot read evidence file: {path}") from exc
    after = _identity(path)
    if before != after or len(payload) != before.size or len(payload) > maximum_bytes:
        raise BaceComRecGCConvergenceError(
            f"evidence file changed while it was read: {path}"
        )
    return payload, hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = _read_stable_bytes(path, maximum_bytes=16 * 1024 * 1024)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaceComRecGCConvergenceError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise BaceComRecGCConvergenceError(
            f"JSON evidence must contain one object: {path}"
        )
    return value, digest


def _normalize_provenance(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise BaceComRecGCConvergenceError(
            "checkpoint provenance_fingerprints must be a nonempty mapping"
        )
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise BaceComRecGCConvergenceError(
                "checkpoint provenance keys and values must be strings"
            )
        key = raw_key.strip()
        item = raw_value.strip()
        if not key or not item or key in normalized:
            raise BaceComRecGCConvergenceError(
                "checkpoint provenance contains an empty or duplicate key/value"
            )
        normalized[key] = item
    return dict(sorted(normalized.items()))


def _validate_payload_stat(path: Path, declared: Mapping[str, Any]) -> int:
    """Validate only metadata; do not open or hash a large checkpoint payload."""

    if set(declared) != {"bytes", "sha256"}:
        raise BaceComRecGCConvergenceError(
            f"checkpoint payload declaration is malformed: {path.name}"
        )
    expected_size = declared.get("bytes")
    if not _is_int(expected_size) or int(expected_size) <= 0:
        raise BaceComRecGCConvergenceError(
            f"checkpoint payload byte count is invalid: {path.name}"
        )
    _require_sha256(declared.get("sha256"), field=f"files.{path.name}.sha256")
    observed = _identity(path)
    if observed.size != int(expected_size):
        raise BaceComRecGCConvergenceError(
            f"checkpoint payload size differs from its committed manifest: {path}"
        )
    return observed.size


def _validate_resolved_config(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_parent_ids_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    config, physical_sha256 = _read_json_object(path)
    recorded = _require_sha256(config.get("config_sha256"), field="config_sha256")
    if recorded != expected_config_sha256:
        raise BaceComRecGCConvergenceError(
            "resolved config differs from the caller-frozen config SHA-256"
        )
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    if stable_json_sha256(unsigned) != recorded:
        raise BaceComRecGCConvergenceError("resolved config self-hash is invalid")
    raw_parent_ids = config.get("generation_parent_ids")
    if not isinstance(raw_parent_ids, list) or len(raw_parent_ids) != POLICY.parent_count:
        raise BaceComRecGCConvergenceError(
            f"resolved config must freeze exactly {POLICY.parent_count} parent IDs"
        )
    if any(not isinstance(value, str) or not value for value in raw_parent_ids):
        raise BaceComRecGCConvergenceError("frozen parent IDs must be nonempty strings")
    parent_ids = tuple(raw_parent_ids)
    if len(set(parent_ids)) != POLICY.parent_count:
        raise BaceComRecGCConvergenceError("frozen parent IDs must be unique")
    computed_parent_sha256 = stable_json_sha256(list(parent_ids))
    recorded_parent_sha256 = _require_sha256(
        config.get("generation_parent_ids_sha256"),
        field="generation_parent_ids_sha256",
    )
    if not (
        computed_parent_sha256
        == recorded_parent_sha256
        == expected_parent_ids_sha256
    ):
        raise BaceComRecGCConvergenceError(
            "frozen parent list/hash differs from the caller-frozen authority"
        )
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise BaceComRecGCConvergenceError("resolved config parameters are absent")
    checks = {
        "dataset": str(config.get("dataset") or "").lower() == "bace",
        "mode": config.get("mode") == "full",
        "parent_limit": config.get("parent_limit") == POLICY.parent_count,
        "total_steps": config.get("total_steps") == POLICY.m_max,
        "parameter_steps": parameters.get("steps") == POLICY.m_max,
        "heads": parameters.get("heads") == POLICY.heads,
        "calibration_closed": config.get("calibration_loaded") is False,
        "test_closed": config.get("test_loaded") is False,
        "rf_absent": config.get("rf_oracle_used") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise BaceComRecGCConvergenceError(
            "resolved BACE full-run contract failed: " + ",".join(failures)
        )
    interval = config.get("generation_checkpoint_interval_steps")
    if not _is_int(interval) or int(interval) <= 0:
        raise BaceComRecGCConvergenceError(
            "generation checkpoint interval must be a positive integer"
        )
    _require_sha256(config.get("command_sha256"), field="command_sha256")
    scientific_argv = config.get("scientific_argv")
    if not isinstance(scientific_argv, list) or not scientific_argv or any(
        not isinstance(value, str) or not value for value in scientific_argv
    ):
        raise BaceComRecGCConvergenceError(
            "resolved config scientific_argv must be a nonempty string list"
        )
    return config, parent_ids, physical_sha256


def _validate_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    step: int,
    config: Mapping[str, Any],
    expected_parent_ids_sha256: str,
) -> tuple[dict[str, Any], str, str, int]:
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_FILENAME
    complete_path = checkpoint_dir / CHECKPOINT_COMPLETE_FILENAME
    manifest, manifest_file_sha256 = _read_json_object(manifest_path)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BaceComRecGCConvergenceError("checkpoint manifest schema is unsupported")
    if (
        manifest.get("file_digest_algorithm") != "sha256"
        or manifest.get("checkpoint_digest_scheme") != "stable_json_sha256_v1"
        or manifest.get("boundary") != CHECKPOINT_BOUNDARY
        or manifest.get("state_schema_version") != CHECKPOINT_STATE_SCHEMA_VERSION
        or manifest.get("atomic_complete") is not True
    ):
        raise BaceComRecGCConvergenceError(
            "checkpoint manifest is not an atomic completed-step checkpoint"
        )
    if (
        manifest.get("checkpoint_dir") != checkpoint_dir.name
        or manifest.get("completed_step") != step
        or manifest.get("next_step") != step + 1
        or manifest.get("total_steps") != POLICY.m_max
    ):
        raise BaceComRecGCConvergenceError(
            f"checkpoint step/directory contract is invalid: {checkpoint_dir}"
        )
    provenance = _normalize_provenance(manifest.get("provenance_fingerprints"))
    if stable_json_sha256(provenance) != manifest.get("provenance_sha256"):
        raise BaceComRecGCConvergenceError("checkpoint provenance digest is invalid")
    required_provenance = {
        "generation_parent_ids_sha256": expected_parent_ids_sha256,
        "parameters_sha256": stable_json_sha256(config["parameters"]),
        "scientific_command_sha256": str(config["command_sha256"]),
        "total_steps": str(POLICY.m_max),
        "dataset": "bace",
        "mode": "full",
    }
    mismatches = [
        key for key, value in required_provenance.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise BaceComRecGCConvergenceError(
            "checkpoint provenance differs from frozen config: " + ",".join(mismatches)
        )
    if manifest.get("scientific_argv") != config.get("scientific_argv") or (
        manifest.get("command_sha256") != config.get("command_sha256")
    ):
        raise BaceComRecGCConvergenceError(
            "checkpoint command contract differs from frozen config"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        CHECKPOINT_STATE_FILENAME,
        CHECKPOINT_SQLITE_FILENAME,
    }:
        raise BaceComRecGCConvergenceError("checkpoint payload inventory is incomplete")
    declared_payload_bytes = sum(
        _validate_payload_stat(checkpoint_dir / name, files[name])
        for name in (CHECKPOINT_STATE_FILENAME, CHECKPOINT_SQLITE_FILENAME)
    )
    checkpoint_digest = _require_sha256(
        manifest.get("checkpoint_digest"), field="checkpoint_digest"
    )
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "checkpoint_digest"
    }
    if checkpoint_digest != stable_json_sha256(unsigned_manifest):
        raise BaceComRecGCConvergenceError("checkpoint manifest digest is invalid")
    complete, complete_sha256 = _read_json_object(complete_path)
    if complete != {
        "checkpoint_digest": checkpoint_digest,
        "manifest_sha256": manifest_file_sha256,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
    }:
        raise BaceComRecGCConvergenceError("checkpoint completion receipt is invalid")
    return manifest, manifest_file_sha256, complete_sha256, declared_payload_bytes


def _validate_live_checkpoint_pair(
    local_dir: Path,
    mirror_dir: Path,
    *,
    step: int,
    config: Mapping[str, Any],
    expected_parent_ids_sha256: str,
) -> _CheckpointEvidence:
    local = _validate_checkpoint_manifest(
        local_dir,
        step=step,
        config=config,
        expected_parent_ids_sha256=expected_parent_ids_sha256,
    )
    mirror = _validate_checkpoint_manifest(
        mirror_dir,
        step=step,
        config=config,
        expected_parent_ids_sha256=expected_parent_ids_sha256,
    )
    if local[0] != mirror[0] or local[1] != mirror[1]:
        raise BaceComRecGCConvergenceError(
            f"local/mirror checkpoint manifests differ at step {step}"
        )
    marker_paths = (
        local_dir / CHECKPOINT_MIRRORED_FILENAME,
        mirror_dir / CHECKPOINT_MIRRORED_FILENAME,
    )
    marker_rows = [_read_json_object(path) for path in marker_paths]
    if marker_rows[0][0] != marker_rows[1][0]:
        raise BaceComRecGCConvergenceError(
            f"local/mirror checkpoint receipts differ at step {step}"
        )
    marker = marker_rows[0][0]
    checkpoint_digest = str(local[0]["checkpoint_digest"])
    if (
        marker.get("schema_version") != "comrecgc_generation_checkpoint_mirror_v1"
        or marker.get("checkpoint_mirrored") is not True
        or marker.get("completed_step") != step
        or marker.get("checkpoint_digest") != checkpoint_digest
        or Path(str(marker.get("source_checkpoint") or "")).name != local_dir.name
        or Path(str(marker.get("mirror_checkpoint") or "")).name != mirror_dir.name
    ):
        raise BaceComRecGCConvergenceError(
            f"checkpoint mirror receipt is invalid at step {step}"
        )
    return _CheckpointEvidence(
        step=step,
        checkpoint_digest=checkpoint_digest,
        kind="live_local_and_mirror",
        evidence_paths=tuple(
            str(path)
            for path in (
                local_dir / CHECKPOINT_MANIFEST_FILENAME,
                local_dir / CHECKPOINT_COMPLETE_FILENAME,
                marker_paths[0],
                mirror_dir / CHECKPOINT_MANIFEST_FILENAME,
                mirror_dir / CHECKPOINT_COMPLETE_FILENAME,
                marker_paths[1],
            )
        ),
        evidence_sha256=(
            local[1],
            local[2],
            marker_rows[0][1],
            mirror[1],
            mirror[2],
            marker_rows[1][1],
        ),
        provenance_fingerprints=_normalize_provenance(
            local[0]["provenance_fingerprints"]
        ),
        declared_payload_bytes=int(local[3]),
    )


def _validate_retention_pair(
    local_receipt: Path, mirror_receipt: Path, *, step: int
) -> _CheckpointEvidence:
    local, local_sha256 = _read_json_object(local_receipt)
    mirror, mirror_sha256 = _read_json_object(mirror_receipt)
    if local != mirror:
        raise BaceComRecGCConvergenceError(
            f"local/mirror retention histories differ at step {step}"
        )
    checkpoint_name = f"step-{step:012d}"
    checkpoint_digest = _require_sha256(
        local.get("checkpoint_digest"), field="retention.checkpoint_digest"
    )
    _require_sha256(
        local.get("mirror_marker_sha256"), field="retention.mirror_marker_sha256"
    )
    if (
        local.get("schema_version")
        != "comrecgc_generation_checkpoint_retention_v1"
        or local.get("checkpoint_mirrored") is not True
        or local.get("completed_step") != step
        or Path(str(local.get("local_checkpoint") or "")).name != checkpoint_name
        or Path(str(local.get("mirror_checkpoint") or "")).name != checkpoint_name
        or not _is_int(local.get("retention_keep_last"))
        or int(local["retention_keep_last"]) < 2
    ):
        raise BaceComRecGCConvergenceError(
            f"checkpoint retention receipt is invalid at step {step}"
        )
    return _CheckpointEvidence(
        step=step,
        checkpoint_digest=checkpoint_digest,
        kind="paired_retention_history",
        evidence_paths=(str(local_receipt), str(mirror_receipt)),
        evidence_sha256=(local_sha256, mirror_sha256),
        provenance_fingerprints=None,
        declared_payload_bytes=None,
    )


def _checkpoint_evidence_for_step(
    local_root: Path,
    mirror_root: Path,
    *,
    step: int,
    config: Mapping[str, Any],
    expected_parent_ids_sha256: str,
) -> _CheckpointEvidence:
    name = f"step-{step:012d}"
    local_dir = local_root / name
    mirror_dir = mirror_root / name
    local_history = local_root / RETENTION_HISTORY_DIRNAME / f"{name}.json"
    mirror_history = mirror_root / RETENTION_HISTORY_DIRNAME / f"{name}.json"
    live_pair = local_dir.is_dir() and not local_dir.is_symlink() and (
        mirror_dir.is_dir() and not mirror_dir.is_symlink()
    )
    retention_pair = local_history.is_file() and not local_history.is_symlink() and (
        mirror_history.is_file() and not mirror_history.is_symlink()
    )
    live: _CheckpointEvidence | None = None
    retained: _CheckpointEvidence | None = None
    if live_pair:
        live = _validate_live_checkpoint_pair(
            local_dir,
            mirror_dir,
            step=step,
            config=config,
            expected_parent_ids_sha256=expected_parent_ids_sha256,
        )
    if retention_pair:
        retained = _validate_retention_pair(local_history, mirror_history, step=step)
    if live is not None and retained is not None:
        if live.checkpoint_digest != retained.checkpoint_digest:
            raise BaceComRecGCConvergenceError(
                f"live/retained checkpoint digests differ at step {step}"
            )
        return live
    if live is not None:
        return live
    if retained is not None:
        return retained
    asymmetric_live = local_dir.exists() != mirror_dir.exists()
    asymmetric_history = local_history.exists() != mirror_history.exists()
    if asymmetric_live or asymmetric_history:
        raise BaceComRecGCConvergenceError(
            f"checkpoint evidence is asymmetric at step {step}"
        )
    raise _EvidenceNotReady(f"committed checkpoint evidence is absent at step {step}")


def _trace_parts(trace_chunks_dir: Path) -> list[tuple[int, Path, _FileIdentity]]:
    if trace_chunks_dir.is_symlink() or not trace_chunks_dir.is_dir():
        raise BaceComRecGCConvergenceError(
            f"trace chunk root must be a physical directory: {trace_chunks_dir}"
        )
    rows: list[tuple[int, Path, _FileIdentity]] = []
    for path in trace_chunks_dir.iterdir():
        match = _TRACE_PART_NAME.fullmatch(path.name)
        if match is None:
            continue
        rows.append((int(match.group("index")), path, _identity(path)))
    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        raise _EvidenceNotReady(
            "trace has no CLOSED chunk because no part has a physical successor"
        )
    indices = [row[0] for row in rows]
    if indices != list(range(indices[-1] + 1)):
        raise BaceComRecGCConvergenceError("trace chunk indices are not contiguous from zero")
    return rows


def _read_closed_trace(trace_chunks_dir: Path) -> _TraceScan:
    """Read only parts whose immediate numeric successor already exists."""

    initial = _trace_parts(trace_chunks_dir)
    closed = initial[:-1]
    chunk_evidence: list[dict[str, Any]] = []
    move_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    for index, path, before_identity in closed:
        payload, digest = _read_stable_bytes(path, maximum_bytes=64 * 1024 * 1024)
        if _identity(path) != before_identity:
            raise BaceComRecGCConvergenceError(
                f"CLOSED trace chunk identity changed during scan: {path}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BaceComRecGCConvergenceError(
                f"CLOSED trace chunk is not UTF-8: {path}"
            ) from exc
        lines = text.splitlines()
        if len(lines) != POLICY.trace_chunk_rows or any(not line.strip() for line in lines):
            raise BaceComRecGCConvergenceError(
                f"CLOSED trace chunk must contain exactly {POLICY.trace_chunk_rows} rows: {path}"
            )
        for line_index, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaceComRecGCConvergenceError(
                    f"invalid trace JSON: {path}:{line_index + 1}"
                ) from exc
            if not isinstance(value, dict):
                raise BaceComRecGCConvergenceError(
                    f"trace row must be one object: {path}:{line_index + 1}"
                )
            move = value.get("move_index")
            if not _is_int(move) or int(move) < 0:
                raise BaceComRecGCConvergenceError(
                    f"trace move_index is invalid: {path}:{line_index + 1}"
                )
            value = dict(value)
            value["_trace_part"] = index
            value["_trace_line"] = line_index + 1
            move_rows[int(move)].append(value)
        row_count += len(lines)
        chunk_evidence.append(
            {
                "index": index,
                "path": str(path),
                "row_count": len(lines),
                "bytes": len(payload),
                "sha256": digest,
                "closed_by_successor": str(trace_chunks_dir / f"part-{index + 1:06d}.jsonl"),
            }
        )
    # The writer may publish more successors while this read-only scan runs.
    # Every chunk used above must still exist with the same identity and must
    # still have its immediate successor; new later parts are harmless.
    final_by_index = {index: (path, identity) for index, path, identity in _trace_parts(trace_chunks_dir)}
    for index, path, identity in closed:
        final = final_by_index.get(index)
        if final is None or final[0] != path or final[1] != identity or index + 1 not in final_by_index:
            raise BaceComRecGCConvergenceError(
                f"CLOSED trace authority changed during scan: {path}"
            )
    return _TraceScan(
        closed_chunks=tuple(chunk_evidence),
        closed_row_count=row_count,
        closed_through_move=max(move_rows, default=-1),
        move_rows={key: tuple(value) for key, value in move_rows.items()},
    )


def _lineage_errors(row: Mapping[str, Any], frozen_parents: set[str]) -> list[str]:
    failures: list[str] = []
    if row.get("action_resolution") != "exact":
        failures.append("action_resolution_not_exact")
    action = row.get("action")
    if not isinstance(action, list) or not action:
        failures.append("action_null_or_invalid")
    parent = row.get("parent_id")
    if not isinstance(parent, str) or parent not in frozen_parents:
        failures.append("parent_not_frozen")
    for key in ("source_graph_sha256", "target_graph_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            failures.append(f"{key}_not_lowercase_sha256")
    return failures


def _validate_move(
    move: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    frozen_parents: set[str],
) -> tuple[list[tuple[str, str]], list[dict[str, Any]], bool]:
    event_types = [row.get("event") for row in rows]
    teleports = [row for row in rows if row.get("event") == "teleport"]
    selected = [row for row in rows if row.get("event") == "selected_transition"]
    unknown = [value for value in event_types if value not in {"teleport", "selected_transition"}]
    if unknown:
        raise BaceComRecGCConvergenceError(
            f"move {move} contains unsupported trace events: {unknown}"
        )
    if teleports:
        if len(rows) != 1 or len(teleports) != 1 or "head_index" in teleports[0]:
            raise BaceComRecGCConvergenceError(
                f"move {move} must contain exactly one headless teleport event"
            )
        return [], [], True
    if len(selected) != POLICY.heads:
        raise BaceComRecGCConvergenceError(
            f"move {move} must contain one teleport or {POLICY.heads} selected transitions"
        )
    heads = [row.get("head_index") for row in selected]
    if any(not _is_int(value) for value in heads) or set(heads) != set(range(POLICY.heads)):
        raise BaceComRecGCConvergenceError(
            f"move {move} selected-transition heads are incomplete or duplicated"
        )
    valid: list[tuple[str, str]] = []
    errors: list[dict[str, Any]] = []
    for row in selected:
        failures = _lineage_errors(row, frozen_parents)
        if failures:
            errors.append(
                {
                    "move_index": move,
                    "head_index": row.get("head_index"),
                    "trace_part": row.get("_trace_part"),
                    "trace_line": row.get("_trace_line"),
                    "failures": failures,
                }
            )
            continue
        valid.append((str(row["target_graph_sha256"]), str(row["parent_id"])))
    return valid, errors, False


def _candidate_ranking(counter: Counter[str]) -> list[str]:
    return [candidate for candidate, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _checkpoint_summaries(
    scan: _TraceScan,
    *,
    checkpoint_steps: Sequence[int],
    frozen_parent_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest = int(checkpoint_steps[-1])
    if scan.closed_through_move < latest - 1:
        raise _EvidenceNotReady(
            "CLOSED trace does not yet cover the requested checkpoint: "
            f"closed_through_move={scan.closed_through_move}, required={latest - 1}"
        )
    required_moves = set(range(latest))
    observed_moves = {move for move in scan.move_rows if 0 <= move < latest}
    if observed_moves != required_moves:
        missing = sorted(required_moves - observed_moves)
        extra = sorted(observed_moves - required_moves)
        raise BaceComRecGCConvergenceError(
            "CLOSED trace lacks complete move coverage: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    frozen_parents = set(frozen_parent_ids)
    frequencies: Counter[str] = Counter()
    parents_by_candidate: dict[str, set[str]] = defaultdict(set)
    lineage_errors: list[dict[str, Any]] = []
    teleport_count = 0
    selected_count = 0
    summaries: list[dict[str, Any]] = []
    checkpoint_set = set(int(value) for value in checkpoint_steps)
    for move in range(latest):
        valid, errors, teleported = _validate_move(
            move, scan.move_rows[move], frozen_parents=frozen_parents
        )
        lineage_errors.extend(errors)
        if teleported:
            teleport_count += 1
        else:
            selected_count += POLICY.heads
            for candidate, parent in valid:
                frequencies[candidate] += 1
                parents_by_candidate[candidate].add(parent)
        completed_step = move + 1
        if completed_step not in checkpoint_set:
            continue
        ranking = _candidate_ranking(frequencies)
        top100 = ranking[: POLICY.top100_size]
        top20 = ranking[: POLICY.top20_size]
        covered_parents: set[str] = set()
        for candidate in top20:
            covered_parents.update(parents_by_candidate[candidate])
        summaries.append(
            {
                "step": completed_step,
                "move_coverage_start": 0,
                "move_coverage_end": completed_step - 1,
                "move_coverage_count": completed_step,
                "selected_transition_count": selected_count,
                "teleport_count": teleport_count,
                "lineage_error_count": len(lineage_errors),
                "valid_unique_count": len(frequencies),
                "top100": top100,
                "top20": top20,
                "top100_frequency": [int(frequencies[value]) for value in top100],
                "top20_distinct_parent_count": len(covered_parents),
                "train_coverage": len(covered_parents) / POLICY.parent_count,
                "ranking_contract": "Counter(target_graph_sha256), sort=(-frequency,sha256)",
            }
        )
    if [row["step"] for row in summaries] != list(checkpoint_steps):
        raise BaceComRecGCConvergenceError(
            "checkpoint summaries were not materialized at all required steps"
        )
    return summaries, lineage_errors


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if list(left) == list(right) else 0.0
    return numerator / (left_scale * right_scale)


def _rank_spearman(left: Sequence[str], right: Sequence[str]) -> float:
    """Historical contract: direct Pearson on deterministic ranks.

    Each own top-100 receives ranks 1..100.  An item absent from one side but
    present in the sorted union receives the fixed rank 101.  The historical
    metric name is retained, but no scipy/tie reranking is performed.
    """

    left_rank = {value: index + 1 for index, value in enumerate(left)}
    right_rank = {value: index + 1 for index, value in enumerate(right)}
    union = sorted(set(left) | set(right))
    return _pearson(
        [float(left_rank.get(value, POLICY.missing_rank)) for value in union],
        [float(right_rank.get(value, POLICY.missing_rank)) for value in union],
    )


def _window_metrics(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        "from_step": int(previous["step"]),
        "to_step": int(current["step"]),
        "top100_jaccard": _jaccard(previous["top100"], current["top100"]),
        "top20_jaccard": _jaccard(previous["top20"], current["top20"]),
        "rank_spearman": _rank_spearman(previous["top100"], current["top100"]),
        "absolute_train_coverage_gain": abs(
            float(current["train_coverage"]) - float(previous["train_coverage"])
        ),
        "lineage_error_count": int(current["lineage_error_count"]),
        "valid_unique_count": int(current["valid_unique_count"]),
        "rank_contract": (
            "pearson_correlation_of_deterministic_ranks_over_top100_union_"
            "with_missing_rank_101"
        ),
    }
    gates = {
        "top100_jaccard": metrics["top100_jaccard"] >= POLICY.top100_jaccard_min,
        "top20_jaccard": metrics["top20_jaccard"] >= POLICY.top20_jaccard_min,
        "rank_spearman": metrics["rank_spearman"] >= POLICY.rank_spearman_min,
        "absolute_train_coverage_gain": metrics["absolute_train_coverage_gain"]
        <= POLICY.absolute_train_coverage_gain_max,
        "lineage_error_count": metrics["lineage_error_count"] == 0,
        "valid_unique_count": metrics["valid_unique_count"]
        >= POLICY.minimum_valid_unique_count,
    }
    metrics["gates"] = gates
    metrics["pass"] = all(gates.values())
    return metrics


def _claim_audit_root(audit_root: Path, sources: Sequence[Path]) -> Path:
    raw = audit_root.expanduser()
    if not raw.is_absolute() or raw.exists() or raw.is_symlink():
        raise BaceComRecGCConvergenceError(
            f"audit_root must be one fresh absolute path: {raw}"
        )
    parent = raw.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise BaceComRecGCConvergenceError(
            f"audit_root parent must be a physical directory: {parent}"
        )
    resolved_candidate = parent / raw.name
    for source in sources:
        resolved_source = source.expanduser().resolve(strict=False)
        if (
            resolved_candidate == resolved_source
            or resolved_candidate in resolved_source.parents
            or resolved_source in resolved_candidate.parents
        ):
            raise BaceComRecGCConvergenceError(
                "audit_root must be disjoint from every running/config/trace source"
            )
    resolved_candidate.mkdir(mode=0o700)
    return resolved_candidate


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def audit_bace_comrecgc_convergence(
    *,
    resolved_config_path: str | Path,
    trace_chunks_dir: str | Path,
    local_checkpoint_root: str | Path,
    mirror_checkpoint_root: str | Path,
    audit_root: str | Path,
    evaluation_step: int,
    expected_config_sha256: str,
    expected_parent_ids_sha256: str,
) -> dict[str, Any]:
    """Audit one preregistered BACE ComRecGC convergence checkpoint.

    ``evaluation_step`` must be a 2,500-step check boundary.  The function
    reads the three exact 500-step checkpoint generations ending at that step,
    accepting either paired live checkpoint receipts or paired retention
    histories.  Large checkpoint payloads are stat-checked against the
    committed manifest but are never opened or hashed.

    The only writes are ``convergence.json`` and, on a genuine two-window PASS,
    ``CONVERGED_EARLY_STOP.json`` below the caller-provided fresh ``audit_root``.
    The returned dictionary is exactly the convergence JSON payload.
    """

    expected_config_sha256 = _require_sha256(
        expected_config_sha256, field="expected_config_sha256"
    )
    expected_parent_ids_sha256 = _require_sha256(
        expected_parent_ids_sha256, field="expected_parent_ids_sha256"
    )
    if not _is_int(evaluation_step):
        raise BaceComRecGCConvergenceError("evaluation_step must be an integer")
    evaluation_step = int(evaluation_step)
    if (
        evaluation_step < POLICY.m_min
        or evaluation_step > POLICY.m_max
        or evaluation_step % POLICY.check_interval != 0
    ):
        raise BaceComRecGCConvergenceError(
            "evaluation_step must be a preregistered 2,500-step boundary in [10000,50000]"
        )

    config_path = Path(resolved_config_path).expanduser()
    trace_dir = Path(trace_chunks_dir).expanduser()
    local_root = Path(local_checkpoint_root).expanduser()
    mirror_root = Path(mirror_checkpoint_root).expanduser()
    output = _claim_audit_root(
        Path(audit_root), (config_path, trace_dir, local_root, mirror_root)
    )
    base_report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": "BACE",
        "method": "ComRecGC",
        "evaluation_step": evaluation_step,
        "policy": asdict(POLICY),
        "expected_config_sha256": expected_config_sha256,
        "expected_parent_ids_sha256": expected_parent_ids_sha256,
        "created_at": _utc_now(),
        "read_only_science_inputs": True,
        "running_root_modified": False,
        "signals_sent": False,
        "state_payload_opened": False,
        "sqlite_payload_opened": False,
        "test_loaded": False,
    }
    try:
        config, parent_ids, config_file_sha256 = _validate_resolved_config(
            config_path,
            expected_config_sha256=expected_config_sha256,
            expected_parent_ids_sha256=expected_parent_ids_sha256,
        )
        checkpoint_interval = int(config["generation_checkpoint_interval_steps"])
        checkpoint_steps = [
            evaluation_step - 2 * checkpoint_interval,
            evaluation_step - checkpoint_interval,
            evaluation_step,
        ]
        if checkpoint_steps[0] <= 0:
            raise BaceComRecGCConvergenceError(
                "evaluation checkpoint does not have three positive generations"
            )
        checkpoints = [
            _checkpoint_evidence_for_step(
                local_root,
                mirror_root,
                step=step,
                config=config,
                expected_parent_ids_sha256=expected_parent_ids_sha256,
            )
            for step in checkpoint_steps
        ]
        live_provenance = [
            dict(item.provenance_fingerprints)
            for item in checkpoints
            if item.provenance_fingerprints is not None
        ]
        if not live_provenance:
            raise BaceComRecGCConvergenceError(
                "at least one of the three checkpoint generations must retain a live manifest"
            )
        if any(value != live_provenance[0] for value in live_provenance[1:]):
            raise BaceComRecGCConvergenceError(
                "live checkpoint generations do not share one provenance contract"
            )
        scan = _read_closed_trace(trace_dir)
        summaries, lineage_errors = _checkpoint_summaries(
            scan,
            checkpoint_steps=checkpoint_steps,
            frozen_parent_ids=parent_ids,
        )
        windows = [
            _window_metrics(previous, current)
            for previous, current in zip(
                summaries[:-1], summaries[1:], strict=True
            )
        ]
        converged = len(windows) == POLICY.patience_checks and all(
            bool(window["pass"]) for window in windows
        )
        status = "CONVERGED_EARLY_STOP" if converged else "CONTINUE"
        report = {
            **base_report,
            "status": status,
            "converged": converged,
            "m_effective": evaluation_step if converged else None,
            "config_path": str(config_path.resolve(strict=True)),
            "config_file_sha256": config_file_sha256,
            "checkpoint_interval_steps": checkpoint_interval,
            "checkpoint_evidence": [
                {
                    "step": item.step,
                    "checkpoint_digest": item.checkpoint_digest,
                    "kind": item.kind,
                    "evidence_paths": list(item.evidence_paths),
                    "evidence_sha256": list(item.evidence_sha256),
                    "declared_payload_bytes": item.declared_payload_bytes,
                    "large_payloads_stat_checked": item.declared_payload_bytes is not None,
                    "large_payloads_content_read_or_rehashed": False,
                }
                for item in checkpoints
            ],
            "trace_contract": {
                "closure_rule": "part_is_closed_only_when_immediate_successor_exists",
                "chunk_rows": POLICY.trace_chunk_rows,
                "closed_chunk_count": len(scan.closed_chunks),
                "closed_row_count": scan.closed_row_count,
                "closed_through_move": scan.closed_through_move,
                "closed_chunks": list(scan.closed_chunks),
                "move_contract": (
                    "exactly_one_headless_teleport_or_five_selected_transitions_"
                    "with_heads_0_through_4"
                ),
            },
            "checkpoint_summaries": summaries,
            "windows": windows,
            "lineage_error_examples": lineage_errors[:20],
            "lineage_error_examples_truncated": len(lineage_errors) > 20,
            "gate": {
                "required_consecutive_windows": POLICY.patience_checks,
                "observed_consecutive_passing_windows": sum(
                    1 for window in windows if window["pass"]
                ),
                "pass": converged,
            },
        }
    except _EvidenceNotReady as exc:
        report = {
            **base_report,
            "status": "NOT_READY",
            "converged": False,
            "m_effective": None,
            "reason": str(exc),
        }
    except (BaceComRecGCConvergenceError, OSError, ValueError) as exc:
        report = {
            **base_report,
            "status": "FAIL_CLOSED",
            "converged": False,
            "m_effective": None,
            "error_class": type(exc).__name__,
            "reason": str(exc),
        }

    audit_path = output / "convergence.json"
    audit_sha256 = _atomic_write_json(audit_path, report)
    if report["status"] == "CONVERGED_EARLY_STOP":
        latest = report["checkpoint_evidence"][-1]
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "PASS",
            "marker": "[BACE_COMRECGC_CONVERGENCE_EARLY_STOP_PASS]",
            "dataset": "BACE",
            "method": "ComRecGC",
            "m_effective": evaluation_step,
            "checkpoint_step": evaluation_step,
            "checkpoint_digest": latest["checkpoint_digest"],
            "audit_path": str(audit_path),
            "audit_sha256": audit_sha256,
            "two_consecutive_windows_passed": True,
            "test_used_for_convergence": False,
            "state_payload_opened": False,
            "sqlite_payload_opened": False,
            "running_root_modified": False,
            "signals_sent": False,
            "eligible_for_caller_managed_exact_pid_graceful_stop": True,
            "created_at": _utc_now(),
        }
        _atomic_write_json(output / "CONVERGED_EARLY_STOP.json", receipt)
    return report


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "BaceComRecGCConvergenceError",
    "ConvergencePolicy",
    "POLICY",
    "RECEIPT_SCHEMA_VERSION",
    "audit_bace_comrecgc_convergence",
]
