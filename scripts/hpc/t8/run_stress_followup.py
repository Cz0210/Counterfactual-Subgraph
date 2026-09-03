#!/usr/bin/env python3
"""Continue the T8 exact-gSpan stress canary without polling or GPU use.

The command is intentionally inert by default.  It records a deterministic
plan, and invokes ``sbatch`` only when ``--submit`` is present.  A timed-out
canary is refined at the next canonical DFS depth.  A passing canary is used
only as an empirical resource estimate before an exhaustive array is admitted.
No result from this script is a paper-table publication.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    GlobalGCEHPCExactError,
    build_partition_manifest,
    canonical_sha256,
    sha256_file,
    validate_hpc_cli_contract,
    validate_merge_result,
    validate_partition_manifest,
)


FOLLOWUP_SCHEMA = "t8_hpc_stress_followup_v1"
CHILDREN_SCHEMA = "t8_hpc_refinement_children_v1"
ADMISSION_SCHEMA = "t8_hpc_full_admission_v1"
TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
PASS_FLAGS = (
    "patterns_equal",
    "supports_equal",
    "stable_preorder_equal",
    "candidate_inputs_equal",
    "rejection_events_equal",
    "all_events_equal",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[0-9]+$")
PINNED_SCIENCE_COMMIT = "481475c31d809577b791f4dd9002f5d2894c65b4"
INITIAL_STRESS_DEPTH = 3
MAX_REFINEMENT_LEVELS = 4
MIN_STORAGE_RESERVE_BYTES = 2 * 1024**3
STORAGE_RESERVE_FRACTION = 0.20
MAX_ARRAY_CONCURRENCY = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_write_self_hashed(
    path: Path, payload: Mapping[str, Any], *, hash_field: str
) -> dict[str, Any]:
    result = dict(payload)
    if hash_field in result:
        raise GlobalGCEHPCExactError(f"reserved hash field already set: {hash_field}")
    result[hash_field] = canonical_sha256(result)
    encoded = _canonical_bytes(result) + b"\n"
    atomic_write_bytes(path, encoded)
    atomic_write_bytes(path.with_suffix(path.suffix + ".sha256"), (sha256_file(path) + "\n").encode())
    return result


def load_self_hashed(path: Path, *, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict or not SHA256_RE.fullmatch(str(payload.get(hash_field, ""))):
        raise GlobalGCEHPCExactError(f"malformed self-hashed JSON: {path}")
    copy = dict(payload)
    claimed = copy.pop(hash_field)
    if canonical_sha256(copy) != claimed:
        raise GlobalGCEHPCExactError(f"self-hash mismatch: {path}")
    return payload


@contextmanager
def decision_lock(root: Path, job_id: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".upstream-{job_id}.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def normalize_slurm_state(value: str) -> str:
    return value.strip().split()[0].split("+")[0].upper()


def parse_sacct(text: str, job_id: str) -> dict[str, Any]:
    """Parse the exact base-job row; never infer success from a batch step."""

    if not JOB_ID_RE.fullmatch(job_id):
        raise GlobalGCEHPCExactError("upstream job ID must be numeric")
    fields = (
        "job_id",
        "job_name",
        "partition",
        "state_raw",
        "exit_code",
        "elapsed_seconds",
        "time_limit",
        "allocated_cpus",
        "requested_memory",
        "max_rss",
        "started_at",
        "ended_at",
    )
    matches: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        columns = raw_line.rstrip("\n").split("|")
        if columns and columns[-1] == "":
            columns.pop()
        if len(columns) != len(fields) or columns[0] != job_id:
            continue
        row = dict(zip(fields, columns, strict=True))
        try:
            elapsed = int(row["elapsed_seconds"])
            cpus = int(row["allocated_cpus"] or 0)
        except ValueError as exc:
            raise GlobalGCEHPCExactError("sacct numeric fields are malformed") from exc
        state = normalize_slurm_state(row["state_raw"])
        if state not in TERMINAL_STATES:
            raise GlobalGCEHPCExactError(f"upstream job is not terminal: {state}")
        matches.append({**row, "state": state, "elapsed_seconds": elapsed, "allocated_cpus": cpus})
    if len(matches) != 1:
        raise GlobalGCEHPCExactError(
            f"expected one exact base-job sacct row for {job_id}, found {len(matches)}"
        )
    return matches[0]


def query_sacct(job_id: str) -> tuple[str, list[str]]:
    command = [
        "sacct",
        "-j",
        job_id,
        "--format=JobIDRaw,JobName,Partition,State,ExitCode,ElapsedRaw,Timelimit,AllocCPUS,ReqMem,MaxRSS,Start,End",
        "--parsable2",
        "--noheader",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout, command


def _path_tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise GlobalGCEHPCExactError(f"symlink is not allowed in canary evidence: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _require_file_sha(path: Path, expected: str, *, label: str) -> None:
    if not SHA256_RE.fullmatch(expected):
        raise GlobalGCEHPCExactError(f"{label} expected SHA must be 64 lowercase hex")
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise GlobalGCEHPCExactError(f"{label} identity mismatch")


def _validate_execution_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_science_commit != PINNED_SCIENCE_COMMIT:
        raise GlobalGCEHPCExactError(
            f"science commit must remain pinned to {PINNED_SCIENCE_COMMIT}"
        )
    validate_hpc_cli_contract(args.config, args.set)
    if not COMMIT_RE.fullmatch(args.expected_controller_commit):
        raise GlobalGCEHPCExactError(
            "expected controller commit must be exact lowercase 40-hex"
        )
    actual_science_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.science_worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_science_commit != args.expected_science_commit:
        raise GlobalGCEHPCExactError("science worktree commit mismatch")
    actual_controller_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.controller_worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_controller_commit != args.expected_controller_commit:
        raise GlobalGCEHPCExactError("controller worktree commit mismatch")
    _require_file_sha(
        args.input_manifest,
        args.expected_input_manifest_sha256,
        label="input manifest",
    )
    _require_file_sha(args.config, args.expected_hpc_config_sha256, label="HPC config")
    _require_file_sha(
        args.science_worktree / "configs/hpc.yaml",
        args.expected_hpc_config_sha256,
        label="science HPC config",
    )
    input_payload = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if (
        input_payload.get("state") != "PASS"
        or input_payload.get("route_kind")
        != "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD"
        or input_payload.get("split_scope") != "train_only"
        or input_payload.get("calibration_payload_included") is not False
        or input_payload.get("test_payload_included") is not False
        or input_payload.get("matrix_publication_allowed_from_hpc") is not False
        or input_payload.get("mining_config_sha256") != args.expected_config_sha256
        or input_payload.get("hpc_runtime_config", {}).get("sha256")
        != args.expected_hpc_config_sha256
    ):
        raise GlobalGCEHPCExactError("input manifest violates the frozen T8 contract")
    graph_sha = sha256_file(args.graphs_jsonl)
    if input_payload.get("transaction_binding", {}).get("graph_jsonl_sha256") != graph_sha:
        raise GlobalGCEHPCExactError("graph JSONL is not bound to the input manifest")
    return input_payload


def _validate_parent_timeout(canary_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_partition_manifest(canary_root / "partition_manifest.json")
    if manifest.get("scope") != "SELECTED_PARTITION_CANARY":
        raise GlobalGCEHPCExactError("TIMEOUT refinement requires a selected-prefix canary")
    selected = [
        unit
        for unit in manifest["partitions"]
        if unit["root_index"] == 0 and unit["partition_type"] == "PREFIX_SUBTREE"
    ]
    if len(selected) != 1 or manifest.get("selected_partition_ids") != [selected[0]["partition_id"]]:
        raise GlobalGCEHPCExactError("TIMEOUT canary does not bind one fixed root-0 prefix")
    selection_path = canary_root / "canary_prefix_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    claimed_selection_sha = selection.pop("selection_receipt_sha256", None)
    observed_selection_sha = hashlib.sha256(_canonical_bytes(selection) + b"\n").hexdigest()
    selection["selection_receipt_sha256"] = claimed_selection_sha
    if (
        claimed_selection_sha != observed_selection_sha
        or selection.get("state") != "PASS"
        or selection.get("selected_partition_id") != selected[0]["partition_id"]
        or selection.get("selected_root_index") != 0
        or selection.get("partition_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise GlobalGCEHPCExactError("TIMEOUT prefix selection receipt is invalid")
    checkpoint_path = canary_root / "reference" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if (
        checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]
        or checkpoint.get("scientific_input_sha256") != manifest["scientific_input_sha256"]
        or checkpoint.get("current_unit_id") != selected[0]["partition_id"]
        or checkpoint.get("resume_boundary")
        != "COMPLETED_PERSISTENT_REFERENCE_UNIT_ONLY"
    ):
        raise GlobalGCEHPCExactError("TIMEOUT checkpoint is not bound to the fixed parent prefix")
    return manifest, selected[0]


def derive_refinement_children(
    parent_manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    parent_code = parent.get("dfs_code")
    if type(parent_code) is not list or not parent_code:
        raise GlobalGCEHPCExactError("parent DFS prefix is malformed")
    child_depth = len(parent_code) + 1
    if (
        catalog.get("scope") != "SELECTED_ROOTS_CANARY"
        or catalog.get("split_depth") != child_depth
        or catalog.get("included_root_indices") != [0]
        or catalog.get("scientific_input_sha256")
        != parent_manifest.get("scientific_input_sha256")
        or catalog.get("root_universe_sha256")
        != parent_manifest.get("root_universe_sha256")
    ):
        raise GlobalGCEHPCExactError("depth+1 catalog is not bound to the parent science input")
    parent_headers = [
        dict(unit)
        for unit in catalog["partitions"]
        if unit.get("root_index") == 0
        and unit.get("partition_type") == "PREFIX_HEADER"
        and unit.get("dfs_code") == parent_code
    ]
    if len(parent_headers) != 1:
        raise GlobalGCEHPCExactError(
            "depth+1 catalog does not contain the fixed parent event exactly once"
        )
    children = [
        dict(unit)
        for unit in catalog["partitions"]
        if unit.get("root_index") == 0
        and unit.get("partition_type") in {"PREFIX_HEADER", "PREFIX_SUBTREE"}
        and type(unit.get("dfs_code")) is list
        and len(unit["dfs_code"]) == child_depth
        and unit["dfs_code"][: len(parent_code)] == parent_code
    ]
    children.sort(key=lambda unit: int(unit["global_partition_order"]))
    if not children:
        raise GlobalGCEHPCExactError("fixed TIMEOUT prefix has no strict depth+1 children")
    refinable = [unit for unit in children if unit["partition_type"] == "PREFIX_SUBTREE"]
    if not refinable:
        raise GlobalGCEHPCExactError("all depth+1 children are terminal; no refinement canary exists")
    heaviest = sorted(
        refinable,
        key=lambda unit: (-int(unit["support_hint"]), str(unit["partition_id"])),
    )[0]
    return parent_headers[0], children, heaviest


def _build_manifest_or_adopt(path: Path, **kwargs: Any) -> dict[str, Any]:
    if path.exists():
        return validate_partition_manifest(path)
    return build_partition_manifest(output=path, **kwargs)


def validate_canary_pass(canary_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = canary_root / "partition_manifest.json"
    manifest = validate_partition_manifest(manifest_path)
    parity_path = canary_root / "exact_parity.json"
    parity = load_self_hashed(parity_path, hash_field="result_sha256")
    if (
        parity.get("status") != "PASS"
        or parity.get("manifest_sha256") != manifest["manifest_sha256"]
        or parity.get("scientific_input_sha256") != manifest["scientific_input_sha256"]
        or parity.get("search_space_scope") != "SELECTED_PARTITION_CANARY"
        or any(parity.get(flag) is not True for flag in PASS_FLAGS)
        or parity.get("first_event_divergence") is not None
        or parity.get("first_pattern_divergence") is not None
        or parity.get("scientific_search_pruned") is not False
        or parity.get("approximation_used") is not False
        or parity.get("matrix_write_enabled") is not False
    ):
        raise GlobalGCEHPCExactError("completed stress canary lacks an exact parity PASS")
    validate_merge_result(
        canary_root / "merged",
        manifest=manifest,
        allowed_scopes=("SELECTED_PARTITION_CANARY",),
    )
    return manifest, parity


def compute_admission(
    *,
    canary_elapsed_seconds: int,
    canary_bytes: int,
    canary_manifest: Mapping[str, Any],
    full_manifest: Mapping[str, Any],
    free_bytes: int,
    walltime_limit_seconds: int,
    time_safety_factor: float,
    storage_safety_factor: float,
) -> dict[str, Any]:
    if min(canary_elapsed_seconds, canary_bytes, free_bytes, walltime_limit_seconds) < 1:
        raise GlobalGCEHPCExactError("resource measurements must be positive")
    if time_safety_factor < 1.0 or storage_safety_factor < 1.0:
        raise GlobalGCEHPCExactError("resource safety factors are invalid")
    reserve_bytes = max(
        MIN_STORAGE_RESERVE_BYTES,
        math.ceil(free_bytes * STORAGE_RESERVE_FRACTION),
    )
    canary_weight = sum(max(1, int(unit["support_hint"])) for unit in canary_manifest["partitions"])
    shard_loads = [0 for _ in range(int(full_manifest["shard_count"]))]
    for unit in full_manifest["partitions"]:
        shard_loads[int(unit["shard_index"])] += max(1, int(unit["support_hint"]))
    full_weight = sum(shard_loads)
    longest_load = max(shard_loads)
    projected_longest_seconds = math.ceil(
        canary_elapsed_seconds * longest_load / canary_weight * time_safety_factor
    )
    projected_persistent_bytes = math.ceil(
        canary_bytes * full_weight / canary_weight * storage_safety_factor
    )
    available_after_reserve = max(0, free_bytes - reserve_bytes)
    walltime_pass = projected_longest_seconds <= walltime_limit_seconds
    storage_pass = projected_persistent_bytes <= available_after_reserve
    return {
        "canary_elapsed_seconds": canary_elapsed_seconds,
        "canary_persistent_bytes": canary_bytes,
        "canary_support_weight": canary_weight,
        "full_support_weight": full_weight,
        "full_shard_support_weights": shard_loads,
        "longest_shard_support_weight": longest_load,
        "time_safety_factor": time_safety_factor,
        "storage_safety_factor": storage_safety_factor,
        "projected_longest_shard_seconds": projected_longest_seconds,
        "array_walltime_limit_seconds": walltime_limit_seconds,
        "persistent_free_bytes": free_bytes,
        "persistent_reserve_bytes": reserve_bytes,
        "persistent_reserve_policy": "MAX_2_GIB_OR_20_PERCENT_OF_PATH_FREE",
        "persistent_reserve_fraction": STORAGE_RESERVE_FRACTION,
        "persistent_available_after_reserve_bytes": available_after_reserve,
        "projected_persistent_bytes": projected_persistent_bytes,
        "walltime_admission_pass": walltime_pass,
        "storage_admission_pass": storage_pass,
        "admission_pass": walltime_pass and storage_pass,
        "projection_model": "CANARY_ELAPSED_AND_BYTES_SCALED_BY_SUPPORT_HINT_WEIGHT",
        "projection_is_empirical_admission_only": True,
        "projection_is_not_scientific_equivalence_evidence": True,
    }


def _sbatch_export(values: Mapping[str, str]) -> str:
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or any(char in value for char in ",\n\r"):
            raise GlobalGCEHPCExactError(f"unsafe sbatch export value: {key}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in sorted(values.items()))


def submit_sbatch(command: Sequence[str]) -> str:
    completed = subprocess.run(list(command), check=True, capture_output=True, text=True)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not JOB_ID_RE.fullmatch(job_id):
        raise GlobalGCEHPCExactError(f"sbatch did not return a numeric job ID: {completed.stdout!r}")
    return job_id


def _process_tree_rss_bytes(root_pid: int, proc_root: Path = Path("/proc")) -> int | None:
    if not (proc_root / str(root_pid)).exists():
        return None
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            values: dict[str, str] = {}
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            pid = int(entry.name)
            parents[pid] = int(values["PPid"].split()[0])
            rss[pid] = int(values.get("VmRSS", "0 kB").split()[0]) * 1024
        except (KeyError, OSError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(pid for pid, parent in parents.items() if parent in descendants)
        changed = len(descendants) != before
    return sum(rss.get(pid, 0) for pid in descendants)


def _jsonl_lines(path: Path, cache: dict[str, tuple[int, int, int, int]]) -> int:
    stat = path.stat()
    key = str(path)
    prior = cache.get(key)
    identity = (int(stat.st_dev), int(stat.st_ino))
    if prior is not None and prior[:2] == identity and stat.st_size >= prior[2]:
        offset, count = prior[2], prior[3]
    else:
        offset, count = 0, 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while block := stream.read(1024 * 1024):
            count += block.count(b"\n")
        offset = stream.tell()
    cache[key] = (identity[0], identity[1], offset, count)
    return count


def _observe_tree(
    roots: Sequence[Path], cache: dict[str, tuple[int, int, int, int]]
) -> dict[str, Any]:
    files: dict[str, Path] = {}
    skipped_symlinks = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                skipped_symlinks += 1
            elif path.is_file():
                files[str(path)] = path
    total_bytes = 0
    latest_mtime_ns: int | None = None
    event_lines = 0
    pattern_lines = 0
    event_file_count = 0
    pattern_file_count = 0
    live_keys = set(files)
    for key in tuple(cache):
        if key not in live_keys:
            cache.pop(key, None)
    for path in files.values():
        stat = path.stat()
        total_bytes += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns or 0, stat.st_mtime_ns)
        if path.name == "events.jsonl":
            event_file_count += 1
            event_lines += _jsonl_lines(path, cache)
        elif path.name == "patterns.jsonl":
            pattern_file_count += 1
            pattern_lines += _jsonl_lines(path, cache)
    return {
        "root_count": len(roots),
        "file_count": len(files),
        "bytes": total_bytes,
        "event_file_count": event_file_count,
        "pattern_file_count": pattern_file_count,
        "event_lines": event_lines if event_file_count else None,
        "pattern_lines": pattern_lines if pattern_file_count else None,
        "latest_mtime_ns": latest_mtime_ns,
        "skipped_symlink_count": skipped_symlinks,
    }


def _current_partition_prefix_depth(canary_root: Path) -> int | None:
    try:
        manifest = json.loads(
            (canary_root / "partition_manifest.json").read_text(encoding="utf-8")
        )
        checkpoint = json.loads(
            (canary_root / "reference" / "checkpoint.json").read_text(
                encoding="utf-8"
            )
        )
        current_id = checkpoint.get("current_unit_id")
        matches = [
            row for row in manifest.get("partitions", ())
            if row.get("partition_id") == current_id
        ]
        return len(matches[0]["dfs_code"]) if len(matches) == 1 else None
    except (OSError, TypeError, ValueError, KeyError):
        return None


def collect_telemetry(
    *,
    canary_root: Path,
    scratch_roots: Sequence[Path],
    science_pid: int,
    persistent_cache: dict[str, tuple[int, int, int, int]],
    scratch_cache: dict[str, tuple[int, int, int, int]],
    previous_signature: str | None,
    last_progress_at: str | None,
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], str, str | None]:
    observed_at = utc_now()
    persistent = _observe_tree((canary_root,), persistent_cache)
    scratch = _observe_tree(tuple(scratch_roots), scratch_cache)
    rss_bytes = _process_tree_rss_bytes(science_pid, proc_root=proc_root)
    signature = canonical_sha256(
        {
            "persistent": persistent,
            "scratch": scratch,
        }
    )
    progressed = previous_signature is not None and signature != previous_signature
    if previous_signature is None or progressed:
        last_progress_at = observed_at
    payload = {
        "schema_version": "t8_hpc_refinement_telemetry_v1",
        "state": "OBSERVATION",
        "observed_at": observed_at,
        "science_pid": science_pid,
        "science_process_alive": (proc_root / str(science_pid)).exists(),
        "process_tree_rss_bytes": rss_bytes,
        "persistent": persistent,
        "scratch": scratch,
        "current_partition_prefix_depth": _current_partition_prefix_depth(
            canary_root
        ),
        "current_dfs_depth": None,
        "current_dfs_depth_unavailable_reason": (
            "PINNED_SCIENCE_ENTRYPOINT_DOES_NOT_EXPOSE_LIVE_DFS_STACK"
        ),
        "progress_since_previous_observation": progressed,
        "last_progress_at": last_progress_at,
        "progress_signature_sha256": signature,
        "interval_seconds": 60,
        "pure_observation": True,
        "used_for_algorithm_control": False,
        "matrix_write_enabled": False,
        "gpu_requested": False,
    }
    return payload, signature, last_progress_at


def monitor_refinement_canary(args: argparse.Namespace) -> int:
    validate_hpc_cli_contract(args.config, [])
    telemetry_root = args.telemetry_root.expanduser().absolute()
    telemetry_root.mkdir(parents=True, exist_ok=True)
    existing_sequences = [
        int(path.stem.rsplit("-", 1)[1])
        for path in telemetry_root.glob("observation-[0-9][0-9][0-9][0-9][0-9][0-9].json")
    ]
    sequence = max(existing_sequences, default=-1) + 1
    persistent_cache: dict[str, tuple[int, int, int, int]] = {}
    scratch_cache: dict[str, tuple[int, int, int, int]] = {}
    previous_signature: str | None = None
    last_progress_at: str | None = None
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while True:
        scratch_roots = sorted(
            path
            for path in args.scratch_base.glob(
                f"t8-gspan-canary-{args.slurm_job_id}.*"
            )
            if path.is_dir() and not path.is_symlink()
        )
        payload, previous_signature, last_progress_at = collect_telemetry(
            canary_root=args.canary_root,
            scratch_roots=scratch_roots,
            science_pid=args.science_pid,
            persistent_cache=persistent_cache,
            scratch_cache=scratch_cache,
            previous_signature=previous_signature,
            last_progress_at=last_progress_at,
        )
        payload["sequence"] = sequence
        receipt = atomic_write_self_hashed(
            telemetry_root / f"observation-{sequence:06d}.json",
            payload,
            hash_field="telemetry_sha256",
        )
        atomic_write_bytes(
            telemetry_root / "latest.json", _canonical_bytes(receipt) + b"\n"
        )
        sequence += 1
        if stop or not Path(f"/proc/{args.science_pid}").exists():
            return 0
        deadline = time.monotonic() + 60.0
        while not stop and time.monotonic() < deadline:
            if not Path(f"/proc/{args.science_pid}").exists():
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def build_telemetry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observe a refinement canary")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--canary-root", required=True, type=Path)
    parser.add_argument("--telemetry-root", required=True, type=Path)
    parser.add_argument("--scratch-base", required=True, type=Path)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--science-pid", required=True, type=int)
    return parser


def validate_refinement_telemetry(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    depth = int(manifest["split_depth"])
    if depth <= INITIAL_STRESS_DEPTH:
        return None
    selected = [
        unit
        for unit in manifest["partitions"]
        if unit["root_index"] == 0 and unit["partition_type"] == "PREFIX_SUBTREE"
    ]
    if len(selected) != 1:
        raise GlobalGCEHPCExactError("refinement telemetry requires one selected prefix")
    root = args.output_root / "refinement-telemetry" / (
        f"depth-{depth}-{selected[0]['partition_id']}"
    )
    observations = sorted(root.glob("observation-*.json"))
    if not observations:
        raise GlobalGCEHPCExactError("refinement canary has no observational telemetry")
    science_pids: set[int] = set()
    last: dict[str, Any] | None = None
    for sequence, path in enumerate(observations):
        payload = load_self_hashed(path, hash_field="telemetry_sha256")
        if (
            payload.get("schema_version") != "t8_hpc_refinement_telemetry_v1"
            or payload.get("state") != "OBSERVATION"
            or payload.get("sequence") != sequence
            or payload.get("interval_seconds") != 60
            or payload.get("pure_observation") is not True
            or payload.get("used_for_algorithm_control") is not False
            or payload.get("current_dfs_depth") is not None
            or payload.get("matrix_write_enabled") is not False
            or payload.get("gpu_requested") is not False
            or type(payload.get("science_pid")) is not int
        ):
            raise GlobalGCEHPCExactError("refinement telemetry contract is invalid")
        science_pids.add(payload["science_pid"])
        last = payload
    latest = load_self_hashed(root / "latest.json", hash_field="telemetry_sha256")
    if latest != last or len(science_pids) != 1:
        raise GlobalGCEHPCExactError("refinement telemetry sequence/latest binding changed")
    return {
        "root": str(root),
        "observation_count": len(observations),
        "first_observation_file_sha256": sha256_file(observations[0]),
        "last_observation_file_sha256": sha256_file(observations[-1]),
        "latest_telemetry_sha256": latest["telemetry_sha256"],
        "science_pid": next(iter(science_pids)),
        "current_dfs_depth_available": False,
        "current_dfs_depth_unavailable_reason": latest[
            "current_dfs_depth_unavailable_reason"
        ],
        "pure_observation": True,
        "used_for_algorithm_control": False,
    }


def _science_export(args: argparse.Namespace) -> dict[str, str]:
    return {
        "HPC_CPU_PARTITION": args.partition,
        "T8_EXECUTION_WORKTREE": str(args.science_worktree),
        "T8_EXPECTED_COMMIT": args.expected_science_commit,
        "T8_PYTHON": str(args.python),
        "T8_GRAPHS_JSONL": str(args.graphs_jsonl),
        "T8_INPUT_MANIFEST": str(args.input_manifest),
        "T8_EXPECTED_INPUT_MANIFEST_SHA256": args.expected_input_manifest_sha256,
        "T8_EXPECTED_CONFIG_SHA256": args.expected_config_sha256,
        "T8_EXPECTED_HPC_CONFIG_SHA256": args.expected_hpc_config_sha256,
        "T8_OFFICIAL_SRC": str(args.official_src),
        "T8_MIN_SUPPORT": str(args.min_support),
        "T8_MIN_VERTICES": str(args.min_vertices),
        "T8_MAX_VERTICES": str(args.max_vertices),
        "T8_TOP_K": str(args.top_k),
    }


def _controller_export(args: argparse.Namespace) -> dict[str, str]:
    return {
        "HPC_CPU_PARTITION": args.partition,
        "T8_CONTROLLER_WORKTREE": str(args.controller_worktree),
        "T8_EXPECTED_CONTROLLER_COMMIT": args.expected_controller_commit,
        "T8_PYTHON": str(args.python),
    }


def _followup_cli(args: argparse.Namespace, *, job_id: str, canary_root: Path) -> list[str]:
    values = [
        "--config", str(args.config),
        "--upstream-job-id", job_id,
        "--upstream-canary-root", str(canary_root),
        "--output-root", str(args.output_root),
        "--continuation-root", str(args.continuation_root),
        "--controller-worktree", str(args.controller_worktree),
        "--expected-controller-commit", args.expected_controller_commit,
        "--science-worktree", str(args.science_worktree),
        "--expected-science-commit", args.expected_science_commit,
        "--python", str(args.python),
        "--graphs-jsonl", str(args.graphs_jsonl),
        "--input-manifest", str(args.input_manifest),
        "--expected-input-manifest-sha256", args.expected_input_manifest_sha256,
        "--expected-config-sha256", args.expected_config_sha256,
        "--expected-hpc-config-sha256", args.expected_hpc_config_sha256,
        "--official-src", str(args.official_src),
        "--partition", args.partition,
        "--full-shard-count", str(args.full_shard_count),
        "--array-concurrency", str(args.array_concurrency),
        "--array-cpus", str(args.array_cpus),
        "--array-memory", args.array_memory,
        "--storage-path", str(args.storage_path),
        "--full-walltime-hours", str(args.full_walltime_hours),
        "--time-safety-factor", str(args.time_safety_factor),
        "--storage-safety-factor", str(args.storage_safety_factor),
        "--min-support", str(args.min_support),
        "--min-vertices", str(args.min_vertices),
        "--max-vertices", str(args.max_vertices),
        "--top-k", str(args.top_k),
        "--submit",
    ]
    for override in args.set:
        values.extend(("--set", override))
    return values


def _record_plan(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return atomic_write_self_hashed(path, payload, hash_field="plan_sha256")


def _handle_timeout(
    args: argparse.Namespace,
    terminal: Mapping[str, Any],
    decision_root: Path,
) -> dict[str, Any]:
    parent_manifest, parent = _validate_parent_timeout(args.upstream_canary_root)
    next_depth = len(parent["dfs_code"]) + 1
    current_refinement_level = len(parent["dfs_code"]) - INITIAL_STRESS_DEPTH
    next_refinement_level = next_depth - INITIAL_STRESS_DEPTH
    if current_refinement_level < 0:
        raise GlobalGCEHPCExactError(
            "TIMEOUT parent is shallower than the frozen initial stress depth"
        )
    if next_refinement_level > MAX_REFINEMENT_LEVELS:
        return _record_plan(
            decision_root / "plan.json",
            {
                "schema_version": FOLLOWUP_SCHEMA,
                "state": "BLOCKED_MAX_REFINEMENT_LEVELS",
                "created_at": utc_now(),
                "dry_run": not args.submit,
                "upstream_terminal": dict(terminal),
                "initial_stress_depth": INITIAL_STRESS_DEPTH,
                "parent_depth": len(parent["dfs_code"]),
                "requested_child_depth": next_depth,
                "current_refinement_level": current_refinement_level,
                "requested_refinement_level": next_refinement_level,
                "max_refinement_levels": MAX_REFINEMENT_LEVELS,
                "action": "NO_SUBMISSION",
                "matrix_write_enabled": False,
                "gpu_requested": False,
            },
        )
    telemetry = validate_refinement_telemetry(args, parent_manifest)
    catalog_path = decision_root / f"root0-depth-{next_depth}-catalog.json"
    catalog = _build_manifest_or_adopt(
        catalog_path,
        graph_jsonl=args.graphs_jsonl,
        input_manifest=args.input_manifest,
        expected_commit=args.expected_science_commit,
        official_src=args.official_src,
        shard_count=1,
        min_support=args.min_support,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        top_k=args.top_k,
        split_root_indices=(0,),
        split_depth=next_depth,
        canary_root_indices=(0,),
        included_root_indices=(0,),
    )
    parent_header, children, heaviest = derive_refinement_children(
        parent_manifest, parent, catalog
    )
    children_receipt = atomic_write_self_hashed(
        decision_root / "refinement_children_manifest.json",
        {
            "schema_version": CHILDREN_SCHEMA,
            "state": "PASS",
            "created_at": utc_now(),
            "upstream_job_id": terminal["job_id"],
            "upstream_terminal_state": terminal["state"],
            "controller_worktree": str(args.controller_worktree),
            "controller_commit": args.expected_controller_commit,
            "science_worktree": str(args.science_worktree),
            "science_commit": args.expected_science_commit,
            "parent_manifest_sha256": parent_manifest["manifest_sha256"],
            "parent_partition_id": parent["partition_id"],
            "parent_dfs_code": parent["dfs_code"],
            "parent_dfs_code_sha256": parent["dfs_code_sha256"],
            "parent_depth": len(parent["dfs_code"]),
            "child_depth": next_depth,
            "initial_stress_depth": INITIAL_STRESS_DEPTH,
            "refinement_level": next_refinement_level,
            "max_refinement_levels": MAX_REFINEMENT_LEVELS,
            "parent_header": parent_header,
            "catalog_file_sha256": sha256_file(catalog_path),
            "catalog_manifest_sha256": catalog["manifest_sha256"],
            "scientific_input_sha256": catalog["scientific_input_sha256"],
            "strict_direct_child_count": len(children),
            "refinable_child_count": sum(unit["partition_type"] == "PREFIX_SUBTREE" for unit in children),
            "terminal_child_count": sum(unit["partition_type"] == "PREFIX_HEADER" for unit in children),
            "children": children,
            "coverage_contract": "PARENT_HEADER_PLUS_ALL_CANONICAL_DEPTH_PLUS_ONE_CHILD_SEARCH_SPACES",
            "partition_cut": [parent_header, *children],
            "partition_cut_pairwise_disjoint": True,
            "partition_cut_complete_for_parent_subtree": True,
            "selection_order": "SUPPORT_HINT_DESC_PARTITION_ID_ASC",
            "selected_heaviest_partition_id": heaviest["partition_id"],
            "selected_heaviest_support_hint": int(heaviest["support_hint"]),
            "matrix_write_enabled": False,
            "upstream_refinement_telemetry": telemetry,
        },
        hash_field="children_manifest_sha256",
    )
    progress_path = decision_root / "submission_progress.json"
    progress = (
        load_self_hashed(progress_path, hash_field="submission_progress_sha256")
        if progress_path.exists()
        else None
    )
    suffix = uuid.uuid4().hex if args.submit else "DRY_RUN_FRESH_UUID"
    fresh_canary_root = (
        Path(progress["fresh_canary_root"])
        if progress is not None
        else args.continuation_root / f"refinement-depth-{next_depth}-{suffix}"
    )
    telemetry_root = args.output_root / "refinement-telemetry" / (
        f"depth-{next_depth}-{heaviest['partition_id']}"
    )
    plan_payload = {
        "schema_version": FOLLOWUP_SCHEMA,
        "state": "READY_REFINEMENT_CANARY",
        "created_at": utc_now(),
        "dry_run": not args.submit,
        "upstream_terminal": dict(terminal),
        "controller_worktree": str(args.controller_worktree),
        "controller_commit": args.expected_controller_commit,
        "science_worktree": str(args.science_worktree),
        "science_commit": args.expected_science_commit,
        "action": "SUBMIT_FRESH_DEPTH_PLUS_ONE_HEAVIEST_CHILD_CANARY",
        "children_manifest": str(decision_root / "refinement_children_manifest.json"),
        "children_manifest_sha256": children_receipt["children_manifest_sha256"],
        "selected_parent_partition_id": parent["partition_id"],
        "selected_child_partition_id": heaviest["partition_id"],
        "selected_child_support_hint": int(heaviest["support_hint"]),
        "fresh_canary_root": str(fresh_canary_root),
        "telemetry_root": str(telemetry_root),
        "refinement_level": next_refinement_level,
        "max_refinement_levels": MAX_REFINEMENT_LEVELS,
        "submit_requested": args.submit,
        "matrix_write_enabled": False,
        "gpu_requested": False,
        "upstream_refinement_telemetry": telemetry,
    }
    plan = _record_plan(decision_root / "plan.json", plan_payload)
    if not args.submit:
        return plan
    submission_path = decision_root / "submission_receipt.json"
    if submission_path.exists():
        return load_self_hashed(submission_path, hash_field="submission_receipt_sha256")
    if progress is None and fresh_canary_root.exists():
        raise GlobalGCEHPCExactError("fresh refinement canary root already exists")
    science_env = _science_export(args)
    canary_env = {
        **science_env,
        **_controller_export(args),
        "T8_CANARY_PREFIX_UNIT_ID": str(heaviest["partition_id"]),
        "T8_CANARY_ROOT": str(fresh_canary_root),
        "T8_CANARY_TELEMETRY_ROOT": str(telemetry_root),
        "T8_SCIENCE_WORKTREE": str(args.science_worktree),
        "T8_EXPECTED_SCIENCE_COMMIT": args.expected_science_commit,
        "T8_SPLIT_DEPTH": str(next_depth),
        "T8_CANARY_SHARD_COUNT": "2",
    }
    canary_command = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task", "8", "--mem", "64G", "--time", "01:00:00",
        "--export", _sbatch_export(canary_env),
        str(args.controller_worktree / "scripts/hpc/t8/slurm_stress_followup.sh"),
        "refinement-canary",
    ]
    canary_job_id = (
        str(progress["refinement_canary_job_id"])
        if progress is not None and progress.get("refinement_canary_job_id")
        else submit_sbatch(canary_command)
    )
    if progress is None or not progress.get("refinement_canary_job_id"):
        progress = atomic_write_self_hashed(
            progress_path,
            {
                "schema_version": "t8_hpc_submission_progress_v1",
                "state": "REFINEMENT_CANARY_SUBMITTED",
                "updated_at": utc_now(),
                "fresh_canary_root": str(fresh_canary_root),
                "refinement_canary_job_id": canary_job_id,
                "canary_sbatch_argv": canary_command,
            },
            hash_field="submission_progress_sha256",
        )
    followup_command = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--dependency", f"afterany:{canary_job_id}",
        "--export", _sbatch_export(_controller_export(args)),
        str(args.controller_worktree / "scripts/hpc/t8/slurm_stress_followup.sh"),
        "followup",
        *_followup_cli(args, job_id=canary_job_id, canary_root=fresh_canary_root),
    ]
    followup_job_id = (
        str(progress["afterany_followup_job_id"])
        if progress is not None and progress.get("afterany_followup_job_id")
        else submit_sbatch(followup_command)
    )
    atomic_write_self_hashed(
        progress_path,
        {
            "schema_version": "t8_hpc_submission_progress_v1",
            "state": "AFTERANY_FOLLOWUP_SUBMITTED",
            "updated_at": utc_now(),
            "fresh_canary_root": str(fresh_canary_root),
            "refinement_canary_job_id": canary_job_id,
            "afterany_followup_job_id": followup_job_id,
            "canary_sbatch_argv": canary_command,
            "followup_sbatch_argv": followup_command,
        },
        hash_field="submission_progress_sha256",
    )
    return atomic_write_self_hashed(
        submission_path,
        {
            **plan_payload,
            "state": "REFINEMENT_CANARY_SUBMITTED",
            "plan_sha256": plan["plan_sha256"],
            "refinement_canary_job_id": canary_job_id,
            "afterany_followup_job_id": followup_job_id,
            "canary_sbatch_argv": canary_command,
            "followup_sbatch_argv": followup_command,
        },
        hash_field="submission_receipt_sha256",
    )


def _handle_pass(
    args: argparse.Namespace,
    terminal: Mapping[str, Any],
    decision_root: Path,
) -> dict[str, Any]:
    canary_manifest, parity = validate_canary_pass(args.upstream_canary_root)
    telemetry = validate_refinement_telemetry(args, canary_manifest)
    split_depth = int(canary_manifest["split_depth"])
    full_root = args.continuation_root / (
        f"full-depth-{split_depth}-{canary_manifest['scientific_input_sha256'][:12]}"
    )
    full_root.mkdir(parents=True, exist_ok=True)
    full_manifest_path = full_root / "partition_manifest.json"
    full_manifest = _build_manifest_or_adopt(
        full_manifest_path,
        graph_jsonl=args.graphs_jsonl,
        input_manifest=args.input_manifest,
        expected_commit=args.expected_science_commit,
        official_src=args.official_src,
        shard_count=args.full_shard_count,
        min_support=args.min_support,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        top_k=args.top_k,
        split_root_indices=(0,),
        split_depth=split_depth,
        canary_root_indices=(0, 22),
        included_root_indices=None,
    )
    if (
        full_manifest["scope"] != "FULL_ROOT_UNIVERSE"
        or full_manifest["split_depth"] != split_depth
        or full_manifest["shard_count"] != args.full_shard_count
        or full_manifest["scientific_input_sha256"]
        != canary_manifest["scientific_input_sha256"]
        or full_manifest["provenance"]["provenance_sha256"]
        != canary_manifest["provenance"]["provenance_sha256"]
    ):
        raise GlobalGCEHPCExactError("full manifest does not match passing canary science")
    stat = os.statvfs(args.storage_path)
    admission_values = compute_admission(
        canary_elapsed_seconds=int(terminal["elapsed_seconds"]),
        canary_bytes=_path_tree_bytes(args.upstream_canary_root),
        canary_manifest=canary_manifest,
        full_manifest=full_manifest,
        free_bytes=int(stat.f_bavail * stat.f_frsize),
        walltime_limit_seconds=math.ceil(args.full_walltime_hours * 3600),
        time_safety_factor=args.time_safety_factor,
        storage_safety_factor=args.storage_safety_factor,
    )
    admission = atomic_write_self_hashed(
        decision_root / "admission_receipt.json",
        {
            "schema_version": ADMISSION_SCHEMA,
            "state": "PASS" if admission_values["admission_pass"] else "BLOCKED",
            "created_at": utc_now(),
            "upstream_job_id": terminal["job_id"],
            "upstream_terminal_state": terminal["state"],
            "canary_manifest_sha256": canary_manifest["manifest_sha256"],
            "canary_parity_result_sha256": parity["result_sha256"],
            "full_manifest_path": str(full_manifest_path),
            "full_manifest_file_sha256": sha256_file(full_manifest_path),
            "full_manifest_sha256": full_manifest["manifest_sha256"],
            "scientific_input_sha256": full_manifest["scientific_input_sha256"],
            "refinement_telemetry": telemetry,
            "storage_path": str(args.storage_path.resolve()),
            **admission_values,
            "matrix_write_enabled": False,
            "gpu_requested": False,
        },
        hash_field="admission_receipt_sha256",
    )
    plan_payload = {
        "schema_version": FOLLOWUP_SCHEMA,
        "state": "READY_FULL_CHAIN" if admission_values["admission_pass"] else "BLOCKED_RESOURCE_ADMISSION",
        "created_at": utc_now(),
        "dry_run": not args.submit,
        "upstream_terminal": dict(terminal),
        "controller_worktree": str(args.controller_worktree),
        "controller_commit": args.expected_controller_commit,
        "science_worktree": str(args.science_worktree),
        "science_commit": args.expected_science_commit,
        "action": "SUBMIT_ARRAY_AFTEROK_MERGE_AFTEROK_PACKAGE" if admission_values["admission_pass"] else "NO_SUBMISSION",
        "full_root": str(full_root),
        "full_manifest": str(full_manifest_path),
        "admission_receipt": str(decision_root / "admission_receipt.json"),
        "admission_receipt_sha256": admission["admission_receipt_sha256"],
        "submit_requested": args.submit,
        "matrix_write_enabled": False,
        "gpu_requested": False,
        "refinement_telemetry": telemetry,
    }
    plan = _record_plan(decision_root / "plan.json", plan_payload)
    if not args.submit or not admission_values["admission_pass"]:
        return plan
    submission_path = decision_root / "submission_receipt.json"
    if submission_path.exists():
        return load_self_hashed(submission_path, hash_field="submission_receipt_sha256")

    shards_root = full_root / "shards"
    merge_root = full_root / "merged"
    result_tar = full_root / "t8-exact-result.tar"
    result_manifest = full_root / "t8-exact-result-manifest.json"
    environment_path = decision_root / "environment_manifest.json"
    inventory_path = decision_root / "slurm_inventory.json"
    atomic_write_self_hashed(
        environment_path,
        {
            "schema_version": "t8_hpc_environment_v1",
            "state": "PASS",
            "created_at": utc_now(),
            "science_execution_commit": args.expected_science_commit,
            "science_execution_worktree": str(args.science_worktree),
            "controller_commit": args.expected_controller_commit,
            "controller_worktree": str(args.controller_worktree),
            "python": str(args.python),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config_file_sha256": args.expected_hpc_config_sha256,
            "input_manifest_file_sha256": args.expected_input_manifest_sha256,
            "cpu_only": True,
            "gpu_requested": False,
        },
        hash_field="environment_manifest_sha256",
    )
    science_env = _science_export(args)
    chain_env = {
        **science_env,
        "T8_PARTITION_MANIFEST": str(full_manifest_path),
        "T8_EXPECTED_PARTITION_MANIFEST_SHA256": sha256_file(full_manifest_path),
        "T8_CANARY_PARITY_RECEIPT": str(args.upstream_canary_root / "exact_parity.json"),
        "T8_EXPECTED_CANARY_PARITY_SHA256": sha256_file(args.upstream_canary_root / "exact_parity.json"),
        "T8_FULL_SHARDS_ROOT": str(shards_root),
        "T8_SHARD_COUNT": str(args.full_shard_count),
        "T8_ARRAY_CONCURRENCY": str(min(args.array_concurrency, args.full_shard_count)),
    }
    array_command = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task", str(args.array_cpus), "--mem", args.array_memory,
        "--time", f"{args.full_walltime_hours}:00:00",
        "--array", f"0-{args.full_shard_count - 1}%{min(args.array_concurrency, args.full_shard_count)}",
        "--export", _sbatch_export(chain_env),
        str(args.science_worktree / "scripts/hpc/t8/slurm_array.sh"),
    ]
    progress_path = decision_root / "submission_progress.json"
    progress = (
        load_self_hashed(progress_path, hash_field="submission_progress_sha256")
        if progress_path.exists()
        else {}
    )
    array_job_id = (
        str(progress["array_job_id"])
        if progress.get("array_job_id")
        else submit_sbatch(array_command)
    )
    if not progress.get("array_job_id"):
        progress = atomic_write_self_hashed(
            progress_path,
            {
                "schema_version": "t8_hpc_submission_progress_v1",
                "state": "ARRAY_SUBMITTED",
                "updated_at": utc_now(),
                "array_job_id": array_job_id,
                "array_sbatch_argv": array_command,
            },
            hash_field="submission_progress_sha256",
        )
    merge_env = {
        **chain_env,
        **_controller_export(args),
        "T8_FULL_MERGE_ROOT": str(merge_root),
    }
    merge_command = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--dependency", f"afterok:{array_job_id}",
        "--cpus-per-task", "4", "--mem", "64G", "--time", "04:00:00",
        "--export", _sbatch_export(merge_env),
        str(args.controller_worktree / "scripts/hpc/t8/slurm_stress_followup.sh"), "merge",
    ]
    merge_job_id = (
        str(progress["merge_job_id"])
        if progress.get("merge_job_id")
        else submit_sbatch(merge_command)
    )
    if not progress.get("merge_job_id"):
        progress = atomic_write_self_hashed(
            progress_path,
            {
                "schema_version": "t8_hpc_submission_progress_v1",
                "state": "MERGE_SUBMITTED",
                "updated_at": utc_now(),
                "array_job_id": array_job_id,
                "merge_job_id": merge_job_id,
                "array_sbatch_argv": array_command,
                "merge_sbatch_argv": merge_command,
            },
            hash_field="submission_progress_sha256",
        )
    package_env = {
        **merge_env,
        "T8_RESULT_BUNDLE": str(result_tar),
        "T8_RESULT_MANIFEST": str(result_manifest),
        "T8_ENVIRONMENT_MANIFEST": str(environment_path),
        "T8_SLURM_INVENTORY": str(inventory_path),
        "T8_RESOURCE_METRICS": str(decision_root / "admission_receipt.json"),
    }
    package_command = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--dependency", f"afterok:{merge_job_id}",
        "--cpus-per-task", "1", "--mem", "8G", "--time", "01:00:00",
        "--export", _sbatch_export(package_env),
        str(args.controller_worktree / "scripts/hpc/t8/slurm_stress_followup.sh"), "package",
    ]
    package_job_id = (
        str(progress["package_job_id"])
        if progress.get("package_job_id")
        else submit_sbatch(package_command)
    )
    atomic_write_self_hashed(
        progress_path,
        {
            "schema_version": "t8_hpc_submission_progress_v1",
            "state": "PACKAGE_SUBMITTED",
            "updated_at": utc_now(),
            "array_job_id": array_job_id,
            "merge_job_id": merge_job_id,
            "package_job_id": package_job_id,
            "array_sbatch_argv": array_command,
            "merge_sbatch_argv": merge_command,
            "package_sbatch_argv": package_command,
        },
        hash_field="submission_progress_sha256",
    )
    inventory = atomic_write_self_hashed(
        inventory_path,
        {
            "schema_version": "t8_hpc_slurm_inventory_v1",
            "state": "PASS",
            "created_at": utc_now(),
            "upstream_canary_job_id": terminal["job_id"],
            "array_job_id": array_job_id,
            "merge_job_id": merge_job_id,
            "package_job_id": package_job_id,
            "dependency_chain": [
                {"job_id": merge_job_id, "dependency": f"afterok:{array_job_id}"},
                {"job_id": package_job_id, "dependency": f"afterok:{merge_job_id}"},
            ],
            "partition": args.partition,
            "array_range": f"0-{args.full_shard_count - 1}",
            "array_concurrency": min(args.array_concurrency, args.full_shard_count),
            "gpu_requested": False,
        },
        hash_field="slurm_inventory_sha256",
    )
    return atomic_write_self_hashed(
        submission_path,
        {
            **plan_payload,
            "state": "FULL_CHAIN_SUBMITTED",
            "plan_sha256": plan["plan_sha256"],
            "array_job_id": array_job_id,
            "merge_job_id": merge_job_id,
            "package_job_id": package_job_id,
            "slurm_inventory_sha256": inventory["slurm_inventory_sha256"],
            "array_sbatch_argv": array_command,
            "merge_sbatch_argv": merge_command,
            "package_sbatch_argv": package_command,
        },
        hash_field="submission_receipt_sha256",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--upstream-job-id", default="2535373")
    parser.add_argument("--upstream-canary-root", required=True, type=Path)
    parser.add_argument("--sacct-file", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--continuation-root", required=True, type=Path)
    parser.add_argument("--controller-worktree", required=True, type=Path)
    parser.add_argument("--expected-controller-commit", required=True)
    parser.add_argument(
        "--science-worktree",
        default=Path("/share/home/u20526/czx/worktrees/t8-hpc-481475c3"),
        type=Path,
    )
    parser.add_argument(
        "--expected-science-commit", default=PINNED_SCIENCE_COMMIT
    )
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--graphs-jsonl", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-hpc-config-sha256", required=True)
    parser.add_argument("--official-src", required=True, type=Path)
    parser.add_argument("--partition", default="intel")
    parser.add_argument("--full-shard-count", default=16, type=int)
    parser.add_argument("--array-concurrency", default=4, type=int)
    parser.add_argument("--array-cpus", default=4, type=int)
    parser.add_argument("--array-memory", default="64G")
    parser.add_argument("--storage-path", required=True, type=Path)
    parser.add_argument("--full-walltime-hours", default=48, type=int)
    parser.add_argument("--time-safety-factor", default=2.0, type=float)
    parser.add_argument("--storage-safety-factor", default=2.0, type=float)
    parser.add_argument("--min-support", default=2, type=int)
    parser.add_argument("--min-vertices", default=3, type=int)
    parser.add_argument("--max-vertices", default=20, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--submit", action="store_true")
    return parser


def _normalize_paths(args: argparse.Namespace) -> None:
    for name in (
        "config", "upstream_canary_root", "output_root", "continuation_root",
        "controller_worktree", "science_worktree", "python", "graphs_jsonl", "input_manifest",
        "official_src", "storage_path",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().absolute())
    if args.sacct_file is not None:
        args.sacct_file = args.sacct_file.expanduser().absolute()
    if (
        args.full_shard_count < 1
        or args.array_concurrency < 1
        or args.array_cpus < 1
        or args.array_concurrency > MAX_ARRAY_CONCURRENCY
        or args.full_walltime_hours < 1
        or not re.fullmatch(r"[1-9][0-9]*[KMGTP]", args.array_memory)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition)
    ):
        raise GlobalGCEHPCExactError("Slurm resource arguments are invalid")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _normalize_paths(args)
    _validate_execution_inputs(args)
    if args.sacct_file is None:
        sacct_text, sacct_command = query_sacct(args.upstream_job_id)
    else:
        sacct_text = args.sacct_file.read_text(encoding="utf-8")
        sacct_command = ["SACCT_EVIDENCE_FILE", str(args.sacct_file)]
    terminal = parse_sacct(sacct_text, args.upstream_job_id)
    decision_root = args.output_root / f"upstream-{args.upstream_job_id}"
    with decision_lock(args.output_root, args.upstream_job_id):
        submission_path = decision_root / "submission_receipt.json"
        if submission_path.exists():
            return load_self_hashed(
                submission_path, hash_field="submission_receipt_sha256"
            )
        atomic_write_self_hashed(
            decision_root / "terminal_evidence.json",
            {
                "schema_version": "t8_hpc_slurm_terminal_evidence_v1",
                "state": "PASS",
                "captured_at": utc_now(),
                "sacct_argv": sacct_command,
                "sacct_stdout_sha256": hashlib.sha256(sacct_text.encode()).hexdigest(),
                "terminal": terminal,
            },
            hash_field="terminal_evidence_sha256",
        )
        if terminal["state"] == "TIMEOUT":
            return _handle_timeout(args, terminal, decision_root)
        if terminal["state"] == "COMPLETED":
            return _handle_pass(args, terminal, decision_root)
        return _record_plan(
            decision_root / "plan.json",
            {
                "schema_version": FOLLOWUP_SCHEMA,
                "state": "BLOCKED_UPSTREAM_TERMINAL_FAILURE",
                "created_at": utc_now(),
                "dry_run": not args.submit,
                "upstream_terminal": terminal,
                "controller_worktree": str(args.controller_worktree),
                "controller_commit": args.expected_controller_commit,
                "science_worktree": str(args.science_worktree),
                "science_commit": args.expected_science_commit,
                "action": "NO_SUBMISSION",
                "matrix_write_enabled": False,
                "gpu_requested": False,
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    if normalized_argv[:1] == ["monitor"]:
        try:
            return monitor_refinement_canary(
                build_telemetry_parser().parse_args(normalized_argv[1:])
            )
        except (GlobalGCEHPCExactError, OSError, ValueError) as exc:
            print(f"T8_REFINEMENT_TELEMETRY_BLOCKED: {exc}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(normalized_argv)
    try:
        report = run(args)
    except (GlobalGCEHPCExactError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"T8_STRESS_FOLLOWUP_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
