"""Strict policy receipts for observational Mut trace-on adoption.

This module is deliberately dataset-specific.  It does not make trace-on
artifacts generally interchangeable with trace-off artifacts; it records and
validates the one project-owner authorization dated 2026-09-01 and inventories
the exact historical/checkpoint source trees before a dynamic gate may run.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Mapping


AUTHORIZATION_SCHEMA = "mut_trace_on_50k_adoption_authorization_v1"
AUDIT_SCHEMA = "mut_trace_observational_code_audit_v1"
SOURCE_PROJECT_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
INSTRUMENTATION_PROJECT_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
SOURCE_PAYLOAD_SHA256 = (
    "fc790056e3c3267153ac3e2d707717ccec88a89e4d0ad3b677af82d5a90cd3d3"
)
EXPECTED_CANDIDATE_UNIVERSE_SHA256 = (
    "9c4d79ac61746a1dbbbefb1c7826c77a940c95b0348fb051c202f434de06ffbe"
)
CANARY_REQUIRED_HEADROOM_GIB = 64
CANARY_REQUIRED_HEADROOM_BYTES = CANARY_REQUIRED_HEADROOM_GIB * 1024**3
CANARY_RSS_STOP_GIB = 24
CANARY_RSS_STOP_BYTES = CANARY_RSS_STOP_GIB * 1024**3
CANARY_HEADROOM_STOP_GIB = 32
CANARY_HEADROOM_STOP_BYTES = CANARY_HEADROOM_STOP_GIB * 1024**3

_TRACE_TOKEN = re.compile(
    r"trace|trace_enabled|write_trace|trace_buffer|transition_trace|"
    r"lineage_trace|debug_trace",
    re.IGNORECASE,
)


class MutTraceAuthorizationError(RuntimeError):
    """Authorization or trace-code evidence failed closed."""


def _process_start_ticks(proc_root: Path, pid: int) -> int | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        return int(raw[closing + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _nested_counter(value: Mapping[str, Any], dotted: str) -> float:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise MutTraceAuthorizationError(
                f"Protected progress counter is absent: {dotted}"
            )
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise MutTraceAuthorizationError(
            f"Protected progress counter is not numeric: {dotted}"
        )
    return float(current)


def load_protected_throughput_manifest(path: Path) -> dict[str, Any]:
    value, file_sha = _read_physical_json(path)
    if value.get("schema_version") != "mut_trace_protected_throughput_manifest_v1":
        raise MutTraceAuthorizationError("Protected throughput manifest schema changed")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise MutTraceAuthorizationError("Protected throughput manifest has no tasks")
    normalized: list[dict[str, Any]] = []
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise MutTraceAuthorizationError("Protected task must be one object")
        pid = int(raw.get("pid", 0))
        start_ticks = int(raw.get("start_ticks", 0))
        progress = Path(str(raw.get("progress_path") or "")).expanduser()
        if (
            pid <= 0
            or start_ticks <= 0
            or not progress.is_absolute()
            or progress.is_symlink()
            or not progress.is_file()
        ):
            raise MutTraceAuthorizationError("Protected task identity/path is invalid")
        row = {
            "task_id": str(raw.get("task_id") or ""),
            "pid": pid,
            "start_ticks": start_ticks,
            "progress_path": str(progress.resolve(strict=True)),
            "counter_field": str(raw.get("counter_field") or ""),
            "terminal_value": float(raw.get("terminal_value", -1)),
        }
        if not row["task_id"] or not row["counter_field"]:
            raise MutTraceAuthorizationError("Protected task ID/counter field is empty")
        normalized.append(row)
    return {**value, "tasks": normalized, "manifest_file_sha256": file_sha}


def read_protected_progress(
    task: Mapping[str, Any], *, proc_root: Path
) -> dict[str, Any]:
    path = Path(str(task["progress_path"]))
    value, file_sha = _read_physical_json(path)
    counter = _nested_counter(value, str(task["counter_field"]))
    current_ticks = _process_start_ticks(proc_root, int(task["pid"]))
    alive = current_ticks == int(task["start_ticks"])
    terminal_value = float(task["terminal_value"])
    completed = terminal_value >= 0 and counter >= terminal_value
    return {
        "task_id": task["task_id"],
        "pid": int(task["pid"]),
        "start_ticks": int(task["start_ticks"]),
        "alive": alive,
        "completed": completed,
        "counter": counter,
        "terminal_value": terminal_value,
        "progress_path": str(path),
        "progress_file_sha256": file_sha,
        "sampled_at_unix": time.time(),
    }


def establish_protected_throughput_baseline(
    manifest: Mapping[str, Any],
    *,
    proc_root: Path,
    baseline_seconds: int = 300,
    poll_seconds: int = 10,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Measure real monotonic science counters for a complete five minutes."""

    if baseline_seconds < 300 or poll_seconds < 5:
        raise MutTraceAuthorizationError(
            "Protected baseline must span at least five minutes"
        )
    tasks = list(manifest["tasks"])
    first = {
        row["task_id"]: read_protected_progress(row, proc_root=proc_root)
        for row in tasks
    }
    started = time.monotonic()
    samples = [first]
    while time.monotonic() - started < baseline_seconds:
        if progress_callback is not None:
            progress_callback(int(time.monotonic() - started), first)
        time.sleep(poll_seconds)
        samples.append(
            {
                row["task_id"]: read_protected_progress(row, proc_root=proc_root)
                for row in tasks
            }
        )
    elapsed = max(time.monotonic() - started, 1e-9)
    last = samples[-1]
    baselines: dict[str, Any] = {}
    failures: list[str] = []
    for task in tasks:
        key = task["task_id"]
        before = first[key]
        after = last[key]
        if after["counter"] < before["counter"]:
            failures.append(f"counter_regressed:{key}")
            continue
        if not after["alive"] and not after["completed"]:
            failures.append(f"protected_task_exited:{key}")
            continue
        delta = after["counter"] - before["counter"]
        if after["alive"] and not after["completed"] and delta <= 0:
            failures.append(f"no_positive_baseline:{key}")
            continue
        baselines[key] = {
            "state": "COMPLETED_DURING_BASELINE" if after["completed"] else "ACTIVE",
            "counter_start": before["counter"],
            "counter_end": after["counter"],
            "counter_delta": delta,
            "elapsed_seconds": elapsed,
            "units_per_second": delta / elapsed if delta > 0 else None,
            "pid": task["pid"],
            "start_ticks": task["start_ticks"],
            "progress_path": task["progress_path"],
            "counter_field": task["counter_field"],
        }
    return {
        "schema_version": "mut_trace_protected_throughput_baseline_v1",
        "status": "PASS" if not failures else "FAIL",
        "baseline_seconds": elapsed,
        "poll_seconds": poll_seconds,
        "tasks": baselines,
        "failures": failures,
        "sample_count": len(samples),
    }


class ProtectedThroughputGate:
    """Fail if a real science counter is >10% slower for five minutes."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        baseline: Mapping[str, Any],
        *,
        proc_root: Path,
        window_seconds: int = 300,
        maximum_slowdown: float = 0.10,
    ) -> None:
        if baseline.get("status") != "PASS":
            raise MutTraceAuthorizationError("Cannot arm throughput gate without PASS baseline")
        self.tasks = list(manifest["tasks"])
        self.baseline = dict(baseline["tasks"])
        self.proc_root = proc_root
        self.window_seconds = int(window_seconds)
        self.maximum_slowdown = float(maximum_slowdown)
        if self.window_seconds < 300:
            raise MutTraceAuthorizationError(
                "Protected throughput windows must span at least five minutes"
            )
        if self.maximum_slowdown != 0.10:
            raise MutTraceAuthorizationError(
                "Protected throughput slowdown threshold must remain exactly 10%"
            )
        self.windows: dict[str, list[dict[str, Any]]] = {}
        self.checked_windows: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.completed_during_canary: set[str] = set()

    def sample(self) -> dict[str, Any]:
        now = time.time()
        failures: list[str] = []
        task_rows: dict[str, Any] = {}
        for task in self.tasks:
            key = str(task["task_id"])
            base = self.baseline.get(key) or {}
            current = read_protected_progress(task, proc_root=self.proc_root)
            task_rows[key] = current
            if base.get("state") == "COMPLETED_DURING_BASELINE":
                continue
            if not current["alive"] and not current["completed"]:
                failures.append(f"protected_task_exited:{key}")
                continue
            if current["completed"]:
                self.completed_during_canary.add(key)
                continue
            window = self.windows.setdefault(key, [])
            window.append(current)
            window[:] = [
                row
                for row in window
                if now - row["sampled_at_unix"] <= self.window_seconds + 30
            ]
            if len(window) < 2 or now - window[0]["sampled_at_unix"] < self.window_seconds:
                continue
            elapsed = current["sampled_at_unix"] - window[0]["sampled_at_unix"]
            delta = current["counter"] - window[0]["counter"]
            rate = delta / elapsed if elapsed > 0 else 0.0
            baseline_rate = float(base["units_per_second"])
            slowdown = 1.0 - rate / baseline_rate if baseline_rate > 0 else 1.0
            check = {
                "task_id": key,
                "elapsed_seconds": elapsed,
                "counter_delta": delta,
                "baseline_units_per_second": baseline_rate,
                "observed_units_per_second": rate,
                "slowdown_fraction": slowdown,
                "pass": slowdown <= self.maximum_slowdown,
            }
            self.checked_windows.append(check)
            if not check["pass"]:
                failures.append(f"protected_slowdown_gt_10_percent:{key}")
        self.failures.extend(
            failure for failure in failures if failure not in self.failures
        )
        return {
            "status": "PASS" if not failures else "FAIL",
            "tasks": task_rows,
            "failures": failures,
            "checked_window_count": len(self.checked_windows),
        }

    def receipt(self) -> dict[str, Any]:
        active = {
            key
            for key, value in self.baseline.items()
            if value.get("state") == "ACTIVE"
        }
        checked = {str(row["task_id"]) for row in self.checked_windows}
        missing = sorted(active - checked - self.completed_during_canary)
        failed = [row for row in self.checked_windows if row.get("pass") is not True]
        status = (
            "PASS"
            if not missing and not failed and not self.failures
            else "FAIL"
        )
        return {
            "schema_version": "mut_trace_protected_throughput_gate_v1",
            "status": status,
            "window_seconds": self.window_seconds,
            "maximum_slowdown": self.maximum_slowdown,
            "active_task_ids": sorted(active),
            "checked_task_ids": sorted(checked),
            "completed_during_canary_task_ids": sorted(
                self.completed_during_canary
            ),
            "missing_complete_five_minute_windows": missing,
            "checked_windows": list(self.checked_windows),
            "failed_windows": failed,
            "failures": list(self.failures),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _physical_file(value: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MutTraceAuthorizationError(
            f"{label} must be one absolute physical file: {path}"
        )
    return path.resolve(strict=True)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL production dependency
        raise MutTraceAuthorizationError(
            "PyTorch is required to bind the historical Mut candidate universe"
        ) from exc
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:  # pragma: no cover - compatibility with older pinned torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise MutTraceAuthorizationError(
            "Historical Mut counterfactual payload must contain one mapping"
        )
    return value


def _candidate_importance(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("importance_parts")
    if value is None:
        raise MutTraceAuthorizationError(
            "Historical Mut candidate is missing importance_parts"
        )
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
        first = value[0]
        if hasattr(first, "item"):
            first = first.item()
        return float(first)
    except (IndexError, TypeError, ValueError) as exc:
        raise MutTraceAuthorizationError(
            "Historical Mut candidate importance_parts is malformed"
        ) from exc


def verify_mut_candidate_pair_dbscan_binding(
    *,
    source_payload_path: Path,
    pair_manifest_path: Path,
    dbscan_manifest_path: Path,
    expected_candidate_universe_sha256: str = EXPECTED_CANDIDATE_UNIVERSE_SHA256,
    expected_source_payload_sha256: str = SOURCE_PAYLOAD_SHA256,
    expected_candidate_count: int = 50_620,
    candidate_capacity: int = 100_000,
) -> dict[str, Any]:
    """Prove the source -> pair chunks -> vectors -> exact DBSCAN chain.

    The historical DBSCAN manifest intentionally has no candidate-universe
    field.  This verifier therefore never fabricates a native DBSCAN hash.  It
    reconstructs the ordered candidate universe from the generation payload,
    proves every pair-store chunk covers the corresponding candidate slice,
    byte-verifies and logically concatenates the chunk arrays into the frozen
    consolidated arrays, then binds those exact vectors to the exact DBSCAN
    input and outputs.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - AutoDL production dependency
        raise MutTraceAuthorizationError(
            "NumPy is required to verify the Mut pair/DBSCAN binding"
        ) from exc

    source_path = _physical_file(source_payload_path, label="source payload")
    pair_path = _physical_file(pair_manifest_path, label="pair-store manifest")
    dbscan_path = _physical_file(dbscan_manifest_path, label="DBSCAN manifest")
    if _sha256_file(source_path) != expected_source_payload_sha256:
        raise MutTraceAuthorizationError("Historical Mut source payload SHA changed")

    payload = _torch_payload(source_path)
    graph_map = payload.get("graph_map")
    raw_candidates = payload.get("counterfactual_candidates")
    if not isinstance(graph_map, Mapping) or not isinstance(raw_candidates, (list, tuple)):
        raise MutTraceAuthorizationError(
            "Historical Mut payload lacks graph_map/counterfactual_candidates"
        )
    graph_hashes: list[str] = []
    generation_indices: list[int] = []
    for generation_index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise MutTraceAuthorizationError(
                f"Historical Mut candidate {generation_index} is not one mapping"
            )
        graph_hash = raw.get("graph_hash")
        if _candidate_importance(raw) >= 0.5 and graph_hash in graph_map:
            graph_hashes.append(str(graph_hash))
            generation_indices.append(generation_index)
        if len(graph_hashes) >= int(candidate_capacity):
            break
    source_universe_sha = _stable_sha256(graph_hashes)
    source_generation_indices_sha = _stable_sha256(generation_indices)
    raw_candidate_count = len(raw_candidates)
    # Drop the multi-GiB graph payload before opening any pair arrays.
    del payload, graph_map, raw_candidates
    import gc

    gc.collect()
    if (
        len(graph_hashes) != int(expected_candidate_count)
        or source_universe_sha != expected_candidate_universe_sha256
    ):
        raise MutTraceAuthorizationError(
            "Historical generation payload does not reconstruct the authorized "
            "Mut candidate universe"
        )

    pair_value, pair_file_sha = _read_physical_json(pair_path)
    scientific = pair_value.get("scientific_identity")
    chunks = pair_value.get("chunks")
    if (
        pair_value.get("schema_version") != "comrecgc_external_pair_store_v1"
        or pair_value.get("run_complete") is not True
        or pair_value.get("candidate_major_parent_minor_order") is not True
        or not isinstance(scientific, Mapping)
        or not isinstance(chunks, list)
        or int(pair_value.get("chunk_count", -1)) != len(chunks)
        or int(scientific.get("candidate_count", -1)) != len(graph_hashes)
        or scientific.get("candidate_graph_hashes_sha256") != source_universe_sha
        or scientific.get("generation_indices_sha256")
        != source_generation_indices_sha
        or scientific.get("pair_order") != "candidate_major_parent_minor"
        or pair_value.get("scientific_identity_sha256")
        != _stable_sha256(scientific)
    ):
        raise MutTraceAuthorizationError(
            "Mut pair-store top-level scientific identity is not source-bound"
        )

    pairs_path = _physical_file(pair_value.get("pairs_path"), label="pair indices")
    vectors_path = _physical_file(
        pair_value.get("vectors_path"), label="pair recourse vectors"
    )
    if (
        _sha256_file(pairs_path) != pair_value.get("pairs_sha256")
        or _sha256_file(vectors_path) != pair_value.get("vectors_sha256")
    ):
        raise MutTraceAuthorizationError("Mut consolidated pair/vector SHA changed")
    consolidated_pairs = np.load(pairs_path, mmap_mode="r", allow_pickle=False)
    consolidated_vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    row_count = int(pair_value.get("row_count", -1))
    vector_dim = int(pair_value.get("vector_dim", -1))
    if (
        consolidated_pairs.shape != (row_count, 2)
        or consolidated_vectors.shape != (row_count, vector_dim)
        or str(consolidated_vectors.dtype) != str(pair_value.get("vectors_dtype"))
    ):
        raise MutTraceAuthorizationError("Mut consolidated pair/vector shape changed")

    expected_start = 0
    row_offset = 0
    represented_candidates: set[int] = set()
    for expected_chunk_index, raw_chunk in enumerate(chunks):
        if not isinstance(raw_chunk, Mapping):
            raise MutTraceAuthorizationError("Mut pair chunk metadata is malformed")
        identity = raw_chunk.get("scientific_identity")
        if not isinstance(identity, Mapping):
            raise MutTraceAuthorizationError("Mut pair chunk lacks scientific identity")
        start = int(identity.get("candidate_start", -1))
        stop = int(identity.get("candidate_stop", -1))
        chunk_rows = int(raw_chunk.get("row_count", -1))
        if (
            int(raw_chunk.get("chunk_index", -1)) != expected_chunk_index
            or int(identity.get("chunk_index", -1)) != expected_chunk_index
            or start != expected_start
            or stop <= start
            or stop > len(graph_hashes)
            or identity.get("candidate_graph_hashes_sha256")
            != _stable_sha256(graph_hashes[start:stop])
            or identity.get("generation_indices_sha256")
            != _stable_sha256(generation_indices[start:stop])
            or raw_chunk.get("scientific_identity_sha256")
            != _stable_sha256(identity)
        ):
            raise MutTraceAuthorizationError(
                f"Mut pair chunk {expected_chunk_index} candidate binding changed"
            )
        chunk_pairs_path = _physical_file(
            raw_chunk.get("pairs_path"), label=f"pair chunk {expected_chunk_index}"
        )
        chunk_vectors_path = _physical_file(
            raw_chunk.get("vectors_path"),
            label=f"vector chunk {expected_chunk_index}",
        )
        if (
            _sha256_file(chunk_pairs_path) != raw_chunk.get("pairs_sha256")
            or _sha256_file(chunk_vectors_path) != raw_chunk.get("vectors_sha256")
        ):
            raise MutTraceAuthorizationError(
                f"Mut pair chunk {expected_chunk_index} bytes changed"
            )
        chunk_pairs = np.load(chunk_pairs_path, mmap_mode="r", allow_pickle=False)
        chunk_vectors = np.load(
            chunk_vectors_path, mmap_mode="r", allow_pickle=False
        )
        row_stop = row_offset + chunk_rows
        if (
            chunk_pairs.shape != (chunk_rows, 2)
            or chunk_vectors.shape != (chunk_rows, vector_dim)
            or str(chunk_vectors.dtype) != str(raw_chunk.get("vectors_dtype"))
            or int(raw_chunk.get("vector_dim", -1)) != vector_dim
            or not np.array_equal(
                chunk_pairs, consolidated_pairs[row_offset:row_stop]
            )
            or not np.array_equal(
                chunk_vectors, consolidated_vectors[row_offset:row_stop]
            )
        ):
            raise MutTraceAuthorizationError(
                f"Mut pair chunk {expected_chunk_index} does not concatenate exactly"
            )
        if chunk_rows:
            parent_column = np.asarray(chunk_pairs[:, 0])
            candidate_column = np.asarray(chunk_pairs[:, 1])
            if (
                int(candidate_column.min()) < start
                or int(candidate_column.max()) >= stop
                or np.any(candidate_column[1:] < candidate_column[:-1])
                or np.any(
                    (candidate_column[1:] == candidate_column[:-1])
                    & (parent_column[1:] < parent_column[:-1])
                )
                or list(map(int, chunk_pairs[0]))
                != list(map(int, raw_chunk.get("first_pair") or []))
                or list(map(int, chunk_pairs[-1]))
                != list(map(int, raw_chunk.get("last_pair") or []))
            ):
                raise MutTraceAuthorizationError(
                    f"Mut pair chunk {expected_chunk_index} row order changed"
                )
            represented_candidates.update(map(int, np.unique(candidate_column)))
        expected_start = stop
        row_offset = row_stop
        del chunk_pairs, chunk_vectors
    if expected_start != len(graph_hashes) or row_offset != row_count:
        raise MutTraceAuthorizationError(
            "Mut pair chunks do not cover the complete ordered candidate universe"
        )

    dbscan_value, dbscan_file_sha = _read_physical_json(dbscan_path)
    dbscan_identity = dbscan_value.get("scientific_identity")
    native_fields = {
        "source_candidate_universe_sha256",
        "candidate_universe_sha256",
        "candidate_graph_hashes_sha256",
    }
    native_present = any(key in dbscan_value for key in native_fields) or (
        isinstance(dbscan_identity, Mapping)
        and any(key in dbscan_identity for key in native_fields)
    )
    shortcut = (
        dbscan_identity.get("shortcut_contract")
        if isinstance(dbscan_identity, Mapping)
        else None
    )
    contract = (
        dbscan_identity.get("contract")
        if isinstance(dbscan_identity, Mapping)
        else None
    )
    if (
        dbscan_value.get("schema_version") != "comrecgc_external_memory_dbscan_v3"
        or dbscan_value.get("run_complete") is not True
        or dbscan_value.get("approximation_used") is not False
        or native_present
        or not isinstance(dbscan_identity, Mapping)
        or Path(str(dbscan_identity.get("vectors_path") or "")).resolve(strict=True)
        != vectors_path
        or dbscan_identity.get("vectors_sha256") != pair_value.get("vectors_sha256")
        or list(dbscan_identity.get("vectors_shape") or [])
        != [row_count, vector_dim]
        or dbscan_identity.get("vectors_dtype") != pair_value.get("vectors_dtype")
        or dbscan_identity.get("distance_reference_dtype") != "float64"
        or dbscan_identity.get("nearest_neighbors_algorithm") != "brute"
        or dbscan_identity.get("nearest_neighbors_metric") != "euclidean"
        or dbscan_value.get("clustering_path")
        != "sklearn_float64_exact_multi_component_v1"
        or not isinstance(shortcut, Mapping)
        or shortcut.get("reference_semantics") != "SKLEARN_FLOAT64"
        or shortcut.get("comparison") != "distance <= eps"
        or shortcut.get("failure_cap_used") is not False
        or not isinstance(contract, Mapping)
        or float(contract.get("eps", -1.0)) != 0.02
        or int(contract.get("min_samples", -1)) != 3
    ):
        raise MutTraceAuthorizationError(
            "Mut DBSCAN is not an explicit exact transitive consumer of pair vectors"
        )
    dbscan_outputs: dict[str, Any] = {}
    for prefix in ("labels", "core_mask", "neighbor_counts"):
        artifact = _physical_file(
            dbscan_value.get(f"{prefix}_path"), label=f"DBSCAN {prefix}"
        )
        if _sha256_file(artifact) != dbscan_value.get(f"{prefix}_sha256"):
            raise MutTraceAuthorizationError(f"Mut DBSCAN {prefix} SHA changed")
        array = np.load(artifact, mmap_mode="r", allow_pickle=False)
        if array.shape != (row_count,):
            raise MutTraceAuthorizationError(f"Mut DBSCAN {prefix} shape changed")
        dbscan_outputs[prefix] = {
            "path": str(artifact),
            "sha256": dbscan_value[f"{prefix}_sha256"],
            "shape": [row_count],
            "dtype": str(array.dtype),
        }
        del array

    result = {
        "schema_version": "mut_candidate_pair_dbscan_binding_v1",
        "status": "PASS",
        "binding_kind": "transitive_generation_pair_store_vectors_dbscan_v1",
        "source_payload_path": str(source_path),
        "source_payload_sha256": expected_source_payload_sha256,
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": len(graph_hashes),
        "source_native_candidate_universe_sha": source_universe_sha,
        "source_generation_indices_sha256": source_generation_indices_sha,
        "pair_store_source_candidate_universe_sha": str(
            scientific["candidate_graph_hashes_sha256"]
        ),
        "pair_store_generation_indices_sha256": str(
            scientific["generation_indices_sha256"]
        ),
        "pair_store_manifest_path": str(pair_path),
        "pair_store_manifest_sha256": pair_file_sha,
        "pair_store_chunk_count": len(chunks),
        "pair_store_row_count": row_count,
        "pair_store_represented_candidate_count": len(represented_candidates),
        "pair_store_zero_close_pair_candidate_count": (
            len(graph_hashes) - len(represented_candidates)
        ),
        "pair_indices_sha256": pair_value["pairs_sha256"],
        "recourse_vectors_path": str(vectors_path),
        "recourse_vectors_sha256": pair_value["vectors_sha256"],
        "dbscan_manifest_path": str(dbscan_path),
        "dbscan_manifest_sha256": dbscan_file_sha,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_transitively_bound_candidate_universe_sha": source_universe_sha,
        "dbscan_approximation_used": False,
        "dbscan_outputs": dbscan_outputs,
        "candidate_universe_binding_state": "PASS",
    }
    result["binding_sha256"] = _stable_sha256(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, fresh: bool = False) -> None:
    """Publish non-JSON evidence with the same physical atomicity contract."""

    target = path.expanduser()
    if not target.is_absolute() or target.is_symlink():
        raise MutTraceAuthorizationError(
            f"Evidence target must be absolute/physical: {target}"
        )
    if fresh and target.exists():
        raise FileExistsError(f"Fresh evidence target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if fresh and target.exists():
            raise FileExistsError(
                f"Fresh evidence target raced into existence: {target}"
            )
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any], *, fresh: bool = False) -> None:
    """Write one physical JSON object using fsync plus atomic rename."""

    target = path.expanduser()
    if not target.is_absolute() or target.is_symlink():
        raise MutTraceAuthorizationError(f"JSON target must be absolute/physical: {target}")
    if fresh and target.exists():
        raise FileExistsError(f"Fresh JSON target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if fresh and target.exists():
            raise FileExistsError(f"Fresh JSON target raced into existence: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_physical_json(path: Path) -> tuple[dict[str, Any], str]:
    """Read one JSON through a no-follow FD and bind the exact bytes read."""

    logical = path.expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise MutTraceAuthorizationError(
            f"Authorization path must be absolute and non-symlink: {logical}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(logical, flags)
    except OSError as exc:
        raise MutTraceAuthorizationError(f"Cannot open authorization: {logical}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise MutTraceAuthorizationError("Authorization is not one nonempty regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size:
            raise MutTraceAuthorizationError("Authorization changed while being read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MutTraceAuthorizationError("Authorization is invalid JSON") from exc
    if not isinstance(value, dict):
        raise MutTraceAuthorizationError("Authorization must be one JSON object")
    return dict(value), hashlib.sha256(raw).hexdigest()


def authorization_payload(*, controller_id: str, source_root: Path) -> dict[str, Any]:
    source = source_root.expanduser().resolve(strict=True)
    payload: dict[str, Any] = {
        "authorization_version": 1,
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_by": "user_project_owner",
        "authorization_date": "2026-09-01",
        "controller_id": str(controller_id),
        "allow_trace_on_50k_adoption": True,
        "require_500_step_equivalence": True,
        "require_trace_code_audit": True,
        "require_checkpoint_reload": True,
        "require_lineage": True,
        "require_candidate_universe_binding": True,
        "allow_full_traceoff_rerun_skip": True,
        "source_trace_enabled": True,
        "target_protocol_trace_enabled": False,
        "source_artifact_root": str(source),
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "expected_candidate_universe_sha256": EXPECTED_CANDIDATE_UNIVERSE_SHA256,
        "trace_mode_policy": "OBSERVATIONAL_INSTRUMENTATION_EQUIVALENCE",
        "canary_parent_headroom_gib": CANARY_REQUIRED_HEADROOM_GIB,
        "canary_arms_sequential": True,
        "required_disclosure": {
            "source_trace_enabled": True,
            "trace_off_full_rerun_performed": False,
            "full_trace_on_off_parity_claimed": False,
        },
        "issued_at": _utc_now(),
    }
    payload["authorization_sha256"] = _stable_sha256(payload)
    return payload


def write_authorization_receipt(
    *, path: Path, controller_id: str, source_root: Path
) -> dict[str, Any]:
    payload = authorization_payload(controller_id=controller_id, source_root=source_root)
    atomic_json(path, payload, fresh=True)
    reopened, file_sha256 = validate_authorization_receipt(
        path,
        expected_controller_id=controller_id,
        expected_source_root=source_root,
    )
    return {**reopened, "receipt_path": str(path.resolve()), "receipt_file_sha256": file_sha256}


def validate_authorization_receipt(
    path: Path,
    *,
    expected_controller_id: str,
    expected_source_root: Path,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _read_physical_json(path)
    expected = {
        "authorization_version": 1,
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_by": "user_project_owner",
        "authorization_date": "2026-09-01",
        "controller_id": str(expected_controller_id),
        "allow_trace_on_50k_adoption": True,
        "require_500_step_equivalence": True,
        "require_trace_code_audit": True,
        "require_checkpoint_reload": True,
        "require_lineage": True,
        "require_candidate_universe_binding": True,
        "allow_full_traceoff_rerun_skip": True,
        "source_trace_enabled": True,
        "target_protocol_trace_enabled": False,
        "source_artifact_root": str(expected_source_root.expanduser().resolve(strict=True)),
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "expected_candidate_universe_sha256": EXPECTED_CANDIDATE_UNIVERSE_SHA256,
        "trace_mode_policy": "OBSERVATIONAL_INSTRUMENTATION_EQUIVALENCE",
        "canary_parent_headroom_gib": CANARY_REQUIRED_HEADROOM_GIB,
        "canary_arms_sequential": True,
        "required_disclosure": {
            "source_trace_enabled": True,
            "trace_off_full_rerun_performed": False,
            "full_trace_on_off_parity_claimed": False,
        },
    }
    failures = [key for key, item in expected.items() if value.get(key) != item]
    claimed = value.get("authorization_sha256")
    unhashed = {key: item for key, item in value.items() if key != "authorization_sha256"}
    if claimed != _stable_sha256(unhashed):
        failures.append("authorization_sha256")
    if failures:
        raise MutTraceAuthorizationError(
            f"Trace-on authorization contract failed: {sorted(set(failures))}"
        )
    return value, file_sha256


def _git_head(root: Path) -> str:
    import subprocess

    value = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise MutTraceAuthorizationError(f"Malformed Git HEAD: {value!r}")
    return value


def _git_science_tree_status(root: Path) -> list[str]:
    import subprocess

    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "src/baselines/comrecgc",
            "scripts/baselines/comrecgc",
        ],
        text=True,
        timeout=30,
    )
    return [line for line in output.splitlines() if line.strip()]


def _enclosing_function(tree: ast.AST) -> dict[ast.AST, str | None]:
    result: dict[ast.AST, str | None] = {}

    def visit(node: ast.AST, current: str | None = None) -> None:
        next_name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
        result[node] = next_name
        for child in ast.iter_child_nodes(node):
            visit(child, next_name)

    visit(tree)
    return result


def _classify_trace_branch(relative: str, function: str | None, source: str) -> str:
    if relative.endswith("/graph_trace.py"):
        if function in {"export_checkpoint_state", "restore_checkpoint_state"}:
            return "CHECKPOINT_SERIALIZATION_ONLY"
        if function in {
            "assert_trace_parity",
            "_configure_output",
            "_flush_chunks",
            "_lineage_recovery_context",
        }:
            return "OBSERVATIONAL_WRITE_ONLY"
        return "UNKNOWN"
    if relative.endswith("/generation_checkpoint.py"):
        return "CHECKPOINT_SERIALIZATION_ONLY"
    if relative.endswith("/runtime.py"):
        if function in {
            "completed_step_boundary",
            "_restore_runtime_checkpoint_state",
        }:
            return "CHECKPOINT_SERIALIZATION_ONLY"
        if function == "run_project_generation":
            # This conservatively covers trace construction, trace metadata,
            # and the final graph-map closure rewrite.  Parity itself is a
            # read-only verifier.
            return (
                "OBSERVATIONAL_WRITE_ONLY"
                if "parity_reference" in source
                else "CHECKPOINT_SERIALIZATION_ONLY"
            )
        if function in {
            "patched_official_runtime",
            "wrapped",
            "lineage_neighbor_wrapper",
        }:
            return "OBSERVATIONAL_WRITE_ONLY"
    if relative.endswith("/run_generation.py") and function in {"main", "build_parser"}:
        return "OBSERVATIONAL_WRITE_ONLY"
    return "UNKNOWN"


def _tree_trace_inventory(root: Path, *, expected_commit: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve(strict=True)
    if _git_head(resolved) != expected_commit:
        raise MutTraceAuthorizationError(
            f"Trace review worktree commit changed: {resolved}"
        )
    dirty = _git_science_tree_status(resolved)
    if dirty:
        raise MutTraceAuthorizationError(
            "Trace review worktree has dirty or shadow scientific source: "
            + ";".join(dirty[:8])
        )
    branches: list[dict[str, Any]] = []
    text_hits: list[dict[str, Any]] = []
    # Inventory all trace references, while branch classification is limited
    # to the generation implementation that the trace-mode switch can reach.
    search_roots = (
        resolved / "src/baselines/comrecgc",
        resolved / "scripts/baselines/comrecgc",
    )
    generation_toggle_files = {
        "scripts/baselines/comrecgc/run_generation.py",
        "src/baselines/comrecgc/runtime.py",
        "src/baselines/comrecgc/graph_trace.py",
        "src/baselines/comrecgc/generation_checkpoint.py",
    }
    for base in search_roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                raise MutTraceAuthorizationError(f"Trace source is not physical: {path}")
            source = path.read_text(encoding="utf-8")
            relative = path.relative_to(resolved).as_posix()
            lines = source.splitlines()
            for number, line in enumerate(lines, start=1):
                if _TRACE_TOKEN.search(line):
                    text_hits.append(
                        {
                            "path": relative,
                            "line": number,
                            "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                        }
                    )
            tree = ast.parse(source, filename=str(path))
            if relative not in generation_toggle_files:
                continue
            enclosing = _enclosing_function(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.If, ast.IfExp)):
                    continue
                condition = ast.get_source_segment(source, node.test) or ast.dump(node.test)
                if not _TRACE_TOKEN.search(condition):
                    continue
                function = enclosing.get(node)
                classification = _classify_trace_branch(relative, function, condition)
                branches.append(
                    {
                        "path": relative,
                        "line": int(getattr(node, "lineno", -1)),
                        "function": function,
                        "condition": condition,
                        "classification": classification,
                    }
                )
    unknown = [row for row in branches if row["classification"] == "UNKNOWN"]
    required_sources: dict[str, str] = {}
    missing_required_sources: list[str] = []
    for relative in (
        "src/baselines/comrecgc/runtime.py",
        "src/baselines/comrecgc/graph_trace.py",
        "src/baselines/comrecgc/frozen_payload.py",
    ):
        path = resolved / relative
        if path.is_symlink() or not path.is_file():
            missing_required_sources.append(relative)
            required_sources[relative] = ""
        else:
            required_sources[relative] = path.read_text(encoding="utf-8")
    runtime_source = required_sources["src/baselines/comrecgc/runtime.py"]
    graph_trace_source = required_sources["src/baselines/comrecgc/graph_trace.py"]
    frozen_payload_source = required_sources[
        "src/baselines/comrecgc/frozen_payload.py"
    ]
    recourse_path = resolved / "src/baselines/comrecgc/recourse.py"
    recourse_source = (
        recourse_path.read_text(encoding="utf-8")
        if recourse_path.is_file() and not recourse_path.is_symlink()
        else ""
    )
    if not recourse_source:
        missing_required_sources.append("src/baselines/comrecgc/recourse.py")
    frozen_payload_tree = ast.parse(frozen_payload_source)
    closure_builder = next(
        (
            node
            for node in ast.walk(frozen_payload_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build_frozen_payload_closure"
        ),
        None,
    )
    closure_written_keys: set[str] = set()
    closure_copies_input = False
    if closure_builder is not None:
        for node in ast.walk(closure_builder):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                )
                value = node.value
                if any(
                    isinstance(target, ast.Name) and target.id == "frozen"
                    for target in targets
                ):
                    closure_copies_input = (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "dict"
                        and len(value.args) == 1
                        and isinstance(value.args[0], ast.Name)
                        and value.args[0].id == "payload"
                    ) or closure_copies_input
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "frozen"
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        closure_written_keys.add(target.slice.value)
    scientific_assertions = {
        "official_move_result_precedes_record_enumerated": (
            "result = _apply_neighbor_with_lineage(original, graph, action)"
            in runtime_source
            and runtime_source.index(
                "result = _apply_neighbor_with_lineage(original, graph, action)"
            )
            < runtime_source.index("trace_recorder.record_enumerated")
        ),
        "trace_wrap_calls_original_once_in_definition": False,
        "trace_rng_api_absent_from_recorder": True,
        "post_walk_payload_closure_rewrite_present": (
            "payload.clear()" in graph_trace_source
            and "payload.update(" in graph_trace_source
        ),
        "trace_on_and_trace_off_use_same_closure_builder": (
            "trace_recorder.write(" in runtime_source
            and "build_frozen_payload_closure(" in runtime_source
            and "build_frozen_payload_closure(" in graph_trace_source
        ),
        "closure_builder_copies_input_before_graph_only_expansion": (
            closure_builder is not None and closure_copies_input
        ),
        "closure_builder_does_not_assign_candidate_records": (
            closure_builder is not None
            and "counterfactual_candidates" not in closure_written_keys
        ),
        "closure_builder_declares_candidate_semantics_unchanged": (
            '"candidate_order_changed": False' in frozen_payload_source
            and '"candidate_payload_changed": False' in frozen_payload_source
            and '"scientific_parameters_changed": False' in frozen_payload_source
        ),
        "downstream_candidate_selection_reads_payload_not_trace_artifacts": (
            'payload.get("counterfactual_candidates")' in recourse_source
            and 'payload.get("graph_map")' in recourse_source
            and "_importance_parts(candidate)" in recourse_source
            and "trace_output_dir" not in recourse_source
            and "candidate_action_lineage.json" not in recourse_source
            and "selected_action_trace" not in recourse_source
        ),
    }
    graph_tree = ast.parse(graph_trace_source)
    for node in ast.walk(graph_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "wrap_move":
            nested = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if nested is not None:
                original_calls = [
                    call
                    for call in ast.walk(nested)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "original"
                ]
                scientific_assertions[
                    "trace_wrap_calls_original_once_in_definition"
                ] = len(original_calls) == 1
    recorder_node = next(
        (
            node
            for node in ast.walk(graph_tree)
            if isinstance(node, ast.ClassDef) and node.name == "ActionTraceRecorder"
        ),
        None,
    )
    if recorder_node is None:
        scientific_assertions["trace_rng_api_absent_from_recorder"] = False
    else:
        for call in (
            node for node in ast.walk(recorder_node) if isinstance(node, ast.Call)
        ):
            rendered = ast.unparse(call.func)
            if re.search(
                r"(^|\.)(random|rand|randn|choice|shuffle|permutation)$",
                rendered,
            ):
                scientific_assertions["trace_rng_api_absent_from_recorder"] = False
                break
    failed_assertions = sorted(
        key for key, value in scientific_assertions.items() if value is not True
    )
    return {
        "root": str(resolved),
        "commit": expected_commit,
        "trace_text_hit_count": len(text_hits),
        "trace_branch_count": len(branches),
        "text_hits": text_hits,
        "branches": branches,
        "unknown_branches": unknown,
        "scientific_assertions": scientific_assertions,
        "failed_scientific_assertions": failed_assertions,
        "missing_required_sources": missing_required_sources,
        "status": (
            "PASS"
            if branches
            and not unknown
            and not failed_assertions
            and not missing_required_sources
            else "FAIL"
        ),
    }


def audit_trace_semantics(
    *, historical_root: Path, instrumentation_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Inventory every trace branch in the exact two pinned source trees."""

    output = output_dir.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise FileExistsError(f"Trace audit output must be one fresh absolute path: {output}")
    legacy = _tree_trace_inventory(
        historical_root, expected_commit=SOURCE_PROJECT_COMMIT
    )
    instrumented = _tree_trace_inventory(
        instrumentation_root, expected_commit=INSTRUMENTATION_PROJECT_COMMIT
    )
    failures: list[str] = []
    if legacy["status"] != "PASS":
        failures.append("historical_trace_branch_inventory")
    if instrumented["status"] != "PASS":
        failures.append("instrumentation_trace_branch_inventory")
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "trace_is_observational": not failures,
        "trace_candidate_selection_is_observational": not failures,
        "trace_rng_mutation_found": False,
        "trace_algorithm_state_mutation_found": False,
        "trace_control_flow_mutation_found": False,
        "trace_operational_side_effects_found": True,
        "trace_pinning_and_synchronous_io_side_effects_found": True,
        "trace_post_walk_payload_serialization_mutation_found": True,
        "trace_post_walk_graph_closure_only": not failures and all(
            legacy["scientific_assertions"].get(key) is True
            and instrumented["scientific_assertions"].get(key) is True
            for key in (
                "trace_on_and_trace_off_use_same_closure_builder",
                "closure_builder_copies_input_before_graph_only_expansion",
                "closure_builder_does_not_assign_candidate_records",
                "closure_builder_declares_candidate_semantics_unchanged",
            )
        ),
        "historical_post_walk_trace_write_failed": True,
        "freeze_only_recovery_performed": True,
        "freeze_only_recovery_commit_cryptographically_attested": False,
        "static_audit_sufficient_for_adoption": False,
        "dynamic_500_step_equivalence_required": True,
        "full_trace_on_off_parity_claimed": False,
        "historical": legacy,
        "instrumentation": instrumented,
        "failures": failures,
        "audited_at": _utc_now(),
    }
    payload["audit_sha256"] = _stable_sha256(payload)
    output.mkdir(parents=True)
    atomic_json(output / "trace_semantics_audit.json", payload)
    rows = [
        *(dict(row, source_tree="historical") for row in legacy["branches"]),
        *(
            dict(row, source_tree="instrumentation")
            for row in instrumented["branches"]
        ),
    ]
    csv_lines = ["source_tree,path,line,function,classification,condition_sha256"]
    for row in rows:
        condition_sha = hashlib.sha256(str(row["condition"]).encode("utf-8")).hexdigest()
        csv_lines.append(
            ",".join(
                [
                    str(row["source_tree"]),
                    str(row["path"]),
                    str(row["line"]),
                    str(row.get("function") or ""),
                    str(row["classification"]),
                    condition_sha,
                ]
            )
        )
    atomic_bytes(
        output / "trace_flag_code_inventory.csv",
        ("\n".join(csv_lines) + "\n").encode("utf-8"),
    )
    atomic_bytes(
        output / "trace_semantics_audit.md",
        (
            "# Mut trace semantics audit\n\n"
            f"Status: {payload['status']}\n\n"
            "Candidate selection is post-observation in the pinned source. "
            "Trace nevertheless adds pinning and synchronous I/O, and its "
            "post-walk writer expands the serialized graph-map closure. The "
            "historical post-walk write failed and a later freeze-only recovery "
            "produced the final artifact; that recovery receipt does not attest "
            "its code commit. Dynamic same-execution trace-on/off equivalence "
            "and the prior historical-vs-instrumented gate remain mandatory. "
            "No full 50k trace-on/off parity is claimed.\n"
        ).encode("utf-8"),
    )
    if failures:
        raise MutTraceAuthorizationError(f"Trace code audit failed: {failures}")
    atomic_bytes(output / "PASS", b"PASS\n", fresh=True)
    return payload
