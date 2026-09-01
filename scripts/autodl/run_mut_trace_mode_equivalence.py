#!/usr/bin/env python3
"""Run sequential trace-on/trace-off Mut prefixes from one pinned checkout.

Both arms use the checkpoint-instrumented execution checkout whose declared
source algorithm is the historical 7f7ed51 commit.  A common observer wraps
the already-patched move function in both arms, records deterministic
scientific state digests, and is excluded from the trace-mode toggle itself.
The formal random-walk budget remains 50,000; each diagnostic is interrupted
after step 510, with a real reload from the committed step-500 checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


CONTROLLER_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from scripts.autodl.run_mut_checkpoint_instrumentation_equivalence import (  # noqa: E402
    INSTRUMENTATION_COMMIT,
    INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
    LEGACY_SOURCE_INVENTORY_SHA256,
    SOURCE_COMMIT,
    _source_delta_audit,
    _source_inventory,
)


SCHEMA = "mut_trace_on_off_500_step_equivalence_v1"
STEPS_TO_COMPARE = 500
POST_RELOAD_STEPS = 10
FORMAL_M_MAX = 50_000
HISTORICAL_CONFIG_SHA256 = (
    "5a6088a56741627cc7353d5450999d90aad18076304060c7621ea6c0cad11f34"
)
HISTORICAL_DATASET_SHA256 = (
    "6fd22a03193e772a36b608ce05e858dc76cf125f0a25c2779728cb44ccf445dd"
)
HISTORICAL_SOURCE_COHORT_SHA256 = (
    "2c5d6842cbb2f74d72b5d4ec281a59a1f327c2b4959059fa20fb78bf5974e573"
)
HISTORICAL_GNN_SHA256 = (
    "22045e5a6a833d6ed980cef9834859859136a1e2f644d19d78bd63345585f239"
)
HISTORICAL_DISTANCE_SHA256 = (
    "bc64c16340c9170388ff1b3951d2ee4cb9a372456b09691ecd6bb2a881f17648"
)
HISTORICAL_RF_ORACLE_SHA256 = (
    "af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
)
HISTORICAL_INTERNAL_PREDICTION_COUNTS = {"0": 1267, "1": 181}
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"


def _absolute(value: str | Path, *, exists: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"Physical absolute path required: {path}")
    return path.resolve(strict=exists)


def _git_head(root: Path) -> str:
    value = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        timeout=30,
    ).strip()
    if len(value) != 40:
        raise ValueError(f"Malformed Git HEAD: {value!r}")
    return value


def _git_science_status(root: Path) -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
        timeout=30,
    )
    return [line for line in output.splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


_GRAPH_DIFF_DIAGNOSTIC_FIELDS = {
    "graph_diff_action_diagnostic_only",
    "graph_diff_action_state",
    "graph_diff_action_candidates",
    "graph_diff_contains_transition_action",
    "graph_diff_diagnostic_error",
}


def _equivalence_scientific_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _equivalence_scientific_projection(item)
            for key, item in value.items()
            if str(key) not in _GRAPH_DIFF_DIAGNOSTIC_FIELDS
        }
    if isinstance(value, list):
        return [_equivalence_scientific_projection(item) for item in value]
    if isinstance(value, tuple):
        return [_equivalence_scientific_projection(item) for item in value]
    return value


def _physical_json(path: Path) -> dict[str, Any]:
    resolved = _absolute(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {resolved}")
    return value


def _assert_module_under(module: Any, root: Path) -> None:
    source = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if root not in source.parents:
        raise RuntimeError(
            f"Scientific module escaped pinned worktree: {module.__name__}={source}"
        )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _dense_array_descriptor(value: Any, *, kind: str) -> dict[str, Any]:
    """Describe exact logical C-order values without losing dtype or shape."""

    array = value
    if hasattr(array, "detach"):
        array = array.detach().cpu().contiguous()
        try:
            raw = array.numpy().tobytes(order="C")
        except (TypeError, RuntimeError):
            # NumPy does not expose every torch dtype (notably bfloat16).
            raw = array.view(__import__("torch").uint8).numpy().tobytes(
                order="C"
            )
    else:
        contiguous = __import__("numpy").ascontiguousarray(array)
        if bool(getattr(contiguous.dtype, "hasobject", False)):
            raise TypeError("Object-dtype ndarray has no stable scientific bytes")
        array = contiguous
        raw = contiguous.tobytes(order="C")
    return {
        "array_kind": kind,
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "layout": str(getattr(value, "layout", "numpy_c_contiguous")),
        "bytes_contract": "logical_c_order_values_v1",
        "byte_count": len(raw),
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _array_descriptor(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "detach") and hasattr(value, "dtype") and hasattr(
        value, "shape"
    ):
        tensor = value.detach().cpu()
        if bool(getattr(tensor, "is_sparse", False)) or str(
            getattr(tensor, "layout", "")
        ) != "torch.strided":
            original_layout = str(getattr(tensor, "layout", ""))
            if original_layout == "torch.sparse_coo":
                sparse = tensor
                indices = sparse._indices()
                values = sparse._values()
                is_coalesced = bool(sparse.is_coalesced())
            else:
                try:
                    sparse = tensor.to_sparse_coo()
                except (AttributeError, RuntimeError) as exc:
                    raise TypeError(
                        "Unsupported torch tensor layout: "
                        f"{getattr(tensor, 'layout', None)}"
                    ) from exc
                indices = sparse._indices()
                values = sparse._values()
                is_coalesced = bool(sparse.is_coalesced())
            components = {
                "indices": _dense_array_descriptor(
                    indices, kind="torch_sparse_indices"
                ),
                "values": _dense_array_descriptor(
                    values, kind="torch_sparse_values"
                ),
            }
            return {
                "array_kind": "torch_tensor",
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
                "layout": original_layout,
                "is_coalesced": is_coalesced,
                "stored_entry_count": int(values.shape[0]),
                "sparse_components": components,
                "bytes_sha256": stable_json_sha256(components),
            }
        return _dense_array_descriptor(tensor, kind="torch_tensor")
    if (
        type(value).__module__.split(".", 1)[0] == "numpy"
        and hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    ):
        return _dense_array_descriptor(value, kind="numpy_ndarray")
    return None


def _array_values(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _science_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if hasattr(value, "x") and hasattr(value, "edge_index"):
        node_array = getattr(value, "x")
        edge_array = getattr(value, "edge_index")
        nodes = _array_values(node_array)
        edge_index = _array_values(edge_array)
        if (
            not isinstance(nodes, list)
            or not isinstance(edge_index, list)
            or len(edge_index) != 2
            or len(edge_index[0]) != len(edge_index[1])
        ):
            raise TypeError("Malformed graph in scientific checkpoint state")
        payload = {
            "num_nodes": int(getattr(value, "num_nodes", len(nodes))),
            "x": nodes,
            "x_storage": _array_descriptor(node_array),
            "edge_index_storage": _array_descriptor(edge_array),
            "directed_edges": sorted(
                (
                    {"source": int(source), "target": int(target)}
                    for source, target in zip(
                        edge_index[0], edge_index[1], strict=True
                    )
                ),
                key=lambda row: (row["source"], row["target"]),
            ),
        }
        metadata = {
            name: _science_plain(getattr(value, name))
            for name in (
                "comrecgc_parent_id",
                "comrecgc_node_origin",
                "comrecgc_trace_node_ids",
            )
            if getattr(value, name, None) is not None
        }
        return {
            "graph_sha256": stable_json_sha256(payload),
            "graph_payload": payload,
            "lineage_metadata": metadata,
        }
    array = _array_descriptor(value)
    if array is not None:
        return array
    if isinstance(value, Mapping):
        return {str(key): _science_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_science_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_science_plain(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported scientific observer value: {type(value).__name__}")


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - pinned production torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected checkpoint mapping: {path}")
    return value


_LIVE_GRAPH_OPERATIONAL_FIELDS = {
    # These counters describe cache/I/O work.  They are allowed to differ when
    # trace observation pins or reads an already-authoritative graph, and none
    # is consumed by the random walk, candidate selection, or resume outcome.
    "rehydrations",
    "store_read_count",
    "max_hot_cache_size",
}


def _checkpoint_scientific_plain(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if (
                "live_graph_state" in path
                and name in _LIVE_GRAPH_OPERATIONAL_FIELDS
            ):
                continue
            result[name] = _checkpoint_scientific_plain(item, (*path, name))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _checkpoint_scientific_plain(item, (*path, str(index)))
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        rows = [_checkpoint_scientific_plain(item, (*path, "set")) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))
    return _science_plain(value)


def _sqlite_logical_audit(path: Path) -> dict[str, Any]:
    resolved = _absolute(path)
    connection = sqlite3.connect(
        f"{resolved.as_uri()}?mode=ro&immutable=1", uri=True, timeout=60.0
    )
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"Checkpoint SQLite integrity failed: {resolved}")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not str(row[0]).startswith("sqlite_")
        ]
        table_rows: list[dict[str, Any]] = []
        total_rows = 0
        for table in tables:
            if '"' in table:
                raise ValueError(f"Unsafe SQLite table name: {table!r}")
            columns = list(connection.execute(f'PRAGMA table_info("{table}")'))
            names = [str(row[1]) for row in columns]
            if not names or any('"' in name for name in names):
                raise ValueError(f"Malformed SQLite schema: {table}")
            primary = [
                str(row[1])
                for row in sorted(columns, key=lambda row: int(row[5]))
                if int(row[5]) > 0
            ]
            order = primary or names
            select = ",".join(f'"{name}"' for name in names)
            ordering = ",".join(f'"{name}"' for name in order)
            digest = hashlib.sha256()
            count = 0
            for row in connection.execute(
                f'SELECT {select} FROM "{table}" ORDER BY {ordering}'
            ):
                encoded = [_science_plain(item) for item in row]
                digest.update(
                    json.dumps(
                        encoded,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                )
                digest.update(b"\n")
                count += 1
            total_rows += count
            table_rows.append(
                {
                    "table": table,
                    "columns": names,
                    "primary_key": primary,
                    "row_count": count,
                    "logical_rows_sha256": digest.hexdigest(),
                }
            )
        return {
            "integrity_check": integrity,
            "table_count": len(table_rows),
            "row_count": total_rows,
            "tables": table_rows,
            "logical_database_sha256": stable_json_sha256(table_rows),
        }
    finally:
        connection.close()


def _checkpoint_state_audit(checkpoint_dir: Path, *, mode: str) -> dict[str, Any]:
    root = _absolute(checkpoint_dir)
    state_path = root / "generation_state.pt"
    sqlite_path = root / "authoritative_graph_store.sqlite3"
    manifest_path = root / "checkpoint_manifest.json"
    state = _torch_load(state_path)
    if (
        int(state.get("completed_step", -1)) != STEPS_TO_COMPARE
        or int(state.get("next_step", -1)) != STEPS_TO_COMPARE + 1
        or state.get("fully_completed_step") is not True
        or int(state.get("total_steps", -1)) != FORMAL_M_MAX
    ):
        raise ValueError(f"Invalid step-500 checkpoint state: {root}")
    algorithm = state.get("algorithm_state")
    rng = state.get("rng_state")
    trace_state = state.get("trace_state")
    if not isinstance(algorithm, Mapping) or not isinstance(rng, Mapping):
        raise ValueError(f"Checkpoint scientific state is incomplete: {root}")
    if not isinstance(trace_state, Mapping):
        raise ValueError(f"Checkpoint trace state is incomplete: {root}")
    if mode == "on":
        if trace_state.get("schema_version") != "comrecgc_action_trace_state_v1":
            raise ValueError("Trace-on checkpoint does not contain trace recorder state")
    elif trace_state != {
        "schema_version": "comrecgc_trace_disabled_v1",
        "enabled": False,
    }:
        raise ValueError("Trace-off checkpoint contains unexpected trace state")
    official = algorithm.get("official_state")
    if not isinstance(official, Mapping):
        raise ValueError("Checkpoint official state is absent")
    records = _science_plain(official.get("counterfactual_candidates"))
    if not isinstance(records, list):
        raise ValueError("Checkpoint candidate records are malformed")
    candidate_hashes = []
    for row in records:
        if not isinstance(row, Mapping) or row.get("graph_hash") is None:
            raise ValueError("Checkpoint candidate has no official graph hash")
        candidate_hashes.append(str(row["graph_hash"]))
    registry = official.get("graph_index_map")
    if not isinstance(registry, Mapping):
        raise ValueError("Checkpoint graph identity registry is malformed")
    registry_rows = sorted(
        (
            {
                "official_hash": str(key),
                "candidate_index": int(value),
            }
            for key, value in registry.items()
        ),
        key=lambda row: (row["official_hash"], row["candidate_index"]),
    )
    algorithm_full = _science_plain(algorithm)
    algorithm_scientific = _checkpoint_scientific_plain(algorithm)
    rng_plain = _science_plain(rng)
    sqlite_audit = _sqlite_logical_audit(sqlite_path)
    manifest = _physical_json(manifest_path)
    return {
        "checkpoint_dir": str(root),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_digest": manifest.get("checkpoint_digest"),
        "state_file_sha256": sha256_file(state_path),
        "trace_state_intentionally_excluded_from_scientific_digest": True,
        "trace_state_schema": trace_state.get("schema_version"),
        "algorithm_full_state_sha256": stable_json_sha256(algorithm_full),
        "algorithm_scientific_state_sha256": stable_json_sha256(
            algorithm_scientific
        ),
        "algorithm_operational_fields_excluded": sorted(
            _LIVE_GRAPH_OPERATIONAL_FIELDS
        ),
        "rng_state_sha256": stable_json_sha256(rng_plain),
        "candidate_count": len(records),
        "serialized_candidate_records_sha256": stable_json_sha256(records),
        "ordered_candidate_graph_hashes_sha256": stable_json_sha256(
            candidate_hashes
        ),
        "candidate_universe_sha256": stable_json_sha256(
            sorted(set(candidate_hashes))
        ),
        "candidate_duplicate_hash_count": len(candidate_hashes)
        - len(set(candidate_hashes)),
        "graph_registry_keys_sha256": stable_json_sha256(
            sorted(str(key) for key in registry)
        ),
        "graph_registry_mapping_sha256": stable_json_sha256(registry_rows),
        "sqlite": sqlite_audit,
    }


def _resolved_config_audit(arm: Path, *, mode: str) -> dict[str, Any]:
    path = _absolute(arm / "resolved_config.json")
    config = _physical_json(path)
    parameters = config.get("parameters")
    dataset_audit = config.get("dataset_audit")
    gnn = config.get("gnn")
    distance = config.get("distance_model")
    patches = config.get("official_compatibility_patches")
    argv = config.get("scientific_argv")
    if not all(
        isinstance(value, Mapping)
        for value in (parameters, dataset_audit, gnn, distance)
    ) or not isinstance(patches, list) or not isinstance(argv, list):
        raise ValueError(f"Resolved config is incomplete: {path}")
    expected_parameters = {
        "theta": 0.1,
        "teleport": 0.1,
        "steps": FORMAL_M_MAX,
        "heads": 5,
        "candidate_capacity": 100_000,
        "sample_size": 10_000,
        "seed": 0,
    }
    failures = []
    expected_scalars = {
        "dataset": "mutagenicity",
        "mode": "full",
        "project_commit": INSTRUMENTATION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "parent_limit": 1448,
        "generation_parent_ids_sha256": HISTORICAL_SOURCE_COHORT_SHA256,
        "total_steps": FORMAL_M_MAX,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    failures.extend(
        f"config.{key}"
        for key, expected in expected_scalars.items()
        if config.get(key) != expected
    )
    if dict(parameters) != expected_parameters:
        failures.append("config.parameters")
    if (
        dataset_audit.get("dataset_fingerprint") != HISTORICAL_DATASET_SHA256
        or dataset_audit.get("generation_parent_ids_sha256")
        != HISTORICAL_SOURCE_COHORT_SHA256
        or int(dataset_audit.get("node_feature_dim", -1)) != 10
        or int(dataset_audit.get("edge_feature_dim", -1)) != 0
    ):
        failures.append("config.dataset_audit")
    if gnn.get("checkpoint_sha256") != HISTORICAL_GNN_SHA256:
        failures.append("config.gnn")
    if distance.get("checkpoint_sha256") != HISTORICAL_DISTANCE_SHA256:
        failures.append("config.distance_model")
    if config.get("internal_prediction_counts") != HISTORICAL_INTERNAL_PREDICTION_COUNTS:
        failures.append("config.internal_prediction_counts")
    trace_patch = "project_runtime_action_trace_only_v1"
    if (trace_patch in patches) is not (mode == "on"):
        failures.append("config.trace_patch")
    required_argv = {
        "dataset=mutagenicity",
        f"source_algorithm_commit={SOURCE_COMMIT}",
        f"execution_commit={INSTRUMENTATION_COMMIT}",
        f"trace_mode={mode}",
        "parent_limit=1448",
        "batch_size=128",
        "device=cuda:0",
        f"M_MAX={FORMAL_M_MAX}",
        "candidate_capacity=100000",
        "seed=0",
    }
    if not required_argv.issubset({str(value) for value in argv}):
        failures.append("config.scientific_argv")
    normalized_patches = sorted(str(value) for value in patches if value != trace_patch)
    scientific_binding = {
        "dataset": config.get("dataset"),
        "mode": config.get("mode"),
        "project_commit": config.get("project_commit"),
        "upstream_commit": config.get("upstream_commit"),
        "parent_limit": config.get("parent_limit"),
        "generation_parent_ids_sha256": config.get(
            "generation_parent_ids_sha256"
        ),
        "parameters": dict(parameters),
        "dataset_fingerprint": dataset_audit.get("dataset_fingerprint"),
        "feature_schema": {
            "node_feature_dim": dataset_audit.get("node_feature_dim"),
            "edge_feature_dim": dataset_audit.get("edge_feature_dim"),
            "atom_types": dataset_audit.get("atom_types"),
            "label_semantics": dataset_audit.get("label_semantics"),
        },
        "gnn_checkpoint_sha256": gnn.get("checkpoint_sha256"),
        "distance_checkpoint_sha256": distance.get("checkpoint_sha256"),
        "internal_prediction_counts": config.get("internal_prediction_counts"),
        "official_patches_excluding_trace": normalized_patches,
        "calibration_loaded": config.get("calibration_loaded"),
        "test_loaded": config.get("test_loaded"),
    }
    return {
        "mode": mode,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "trace_patch_present": trace_patch in patches,
        "scientific_binding": scientific_binding,
        "scientific_binding_sha256": stable_json_sha256(scientific_binding),
        "allowed_config_differences": [
            "trace patch presence",
            "trace output and checkpoint/output paths",
            "scientific argv trace_mode/path tokens and command SHA",
            "timestamps and runtime environment diagnostics",
        ],
    }


def _proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError(f"Malformed /proc stat for PID {pid}")
    return int(raw[closing + 2 :].split()[19])


def _execution_overlap(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals = sorted(
        (
            float(row["started_at_unix"]),
            float(row["stopped_at_unix"]),
            str(row["trace_mode"]),
            str(row["phase"]),
            int(row["pid"]),
            int(row["pid_start_ticks"]),
        )
        for row in receipts
    )
    events: list[tuple[float, int]] = []
    overlaps: list[dict[str, Any]] = []
    for index, row in enumerate(intervals):
        started, stopped, mode, phase, pid, ticks = row
        if stopped < started:
            raise ValueError("Canary arm stop precedes start")
        events.extend(((started, 1), (stopped, -1)))
        for other in intervals[:index]:
            if started < other[1] and other[0] < stopped:
                overlaps.append(
                    {
                        "left": f"{other[2]}_{other[3]}",
                        "right": f"{mode}_{phase}",
                        "left_pid": other[4],
                        "right_pid": pid,
                    }
                )
    concurrent = 0
    maximum = 0
    # Stop events sort before start events at an exact boundary.
    for _when, delta in sorted(events, key=lambda item: (item[0], item[1])):
        concurrent += delta
        maximum = max(maximum, concurrent)
    return {
        "arms_overlapped": bool(overlaps),
        "max_concurrent_arms": maximum,
        "overlaps": overlaps,
        "intervals": [
            {
                "started_at_unix": row[0],
                "stopped_at_unix": row[1],
                "trace_mode": row[2],
                "phase": row[3],
                "pid": row[4],
                "pid_start_ticks": row[5],
            }
            for row in intervals
        ],
    }


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": _science_plain(random.getstate()),
        "numpy": _science_plain(np.random.get_state()),
        "torch_cpu": _science_plain(torch.get_rng_state()),
        "torch_cuda": (
            [_science_plain(item) for item in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _candidate_state(module: Any) -> dict[str, Any]:
    candidates = [_science_plain(row) for row in module.counterfactual_candidates]
    graph_index_rows = sorted(
        (
            {
                "official_hash": str(key),
                "candidate_index": int(value),
            }
            for key, value in module.graph_index_map.items()
        ),
        key=lambda row: (row["official_hash"], row["candidate_index"]),
    )
    frequencies = [
        {
            "graph_hash": str(row.get("graph_hash")),
            "frequency": _science_plain(
                row.get("frequency", row.get("count", row.get("importance")))
            ),
        }
        for row in module.counterfactual_candidates
    ]
    live = getattr(module, "comrecgc_live_graph_state", None)
    transitions = getattr(module, "transitions", None)
    export_transition_state = getattr(transitions, "export_checkpoint_state", None)
    if not callable(export_transition_state):
        raise TypeError(
            "Common observer requires the pinned checkpointable transition cache"
        )
    raw_transition_state = export_transition_state()
    if not isinstance(raw_transition_state, Mapping):
        raise TypeError("Transition checkpoint state is not one mapping")
    transition_entries = _science_plain(raw_transition_state.get("entries"))
    if not isinstance(transition_entries, list):
        raise TypeError("Transition checkpoint entries are malformed")
    transition_scientific_state = {
        "schema_version": raw_transition_state.get("schema_version"),
        "seed": int(raw_transition_state.get("seed", -1)),
        "expanded_capacity": int(
            raw_transition_state.get("expanded_capacity", -1)
        ),
        "current_step": int(raw_transition_state.get("current_step", -1)),
        "entry_count": len(transition_entries),
        "entries_sha256": stable_json_sha256(transition_entries),
    }
    live_audit = live.audit() if callable(getattr(live, "audit", None)) else {}
    backing_store = live_audit.get("backing_store")
    if isinstance(backing_store, Mapping):
        live_audit = {
            **live_audit,
            "backing_store": {
                key: _science_plain(backing_store.get(key))
                for key in (
                    "integrity_passed",
                    "entry_count",
                    "content_sha256",
                )
            },
        }
    live_audit = {
        key: _science_plain(value)
        for key, value in live_audit.items()
        if key
        not in {
            "process_peak_rss_bytes",
            "sqlite_bytes",
            "sqlite_wal_bytes",
        }
    }
    return {
        "candidate_count": len(candidates),
        "candidate_records_digest": stable_json_sha256(candidates),
        "candidate_frequency_digest": stable_json_sha256(frequencies),
        "registry_count": len(graph_index_rows),
        "registry_digest": stable_json_sha256(graph_index_rows),
        "registry_mapping_contract": "official_hash_to_candidate_index_v1",
        "algorithm_registry_identity": (
            "pinned_upstream_embedding_bytes_python_hash_seed0"
        ),
        "audit_graph_identity": "stable_untyped_graph_sha256",
        "covering_graphs_digest": stable_json_sha256(
            sorted(str(item) for item in module.covering_graphs)
        ),
        "input_graphs_covered_digest": stable_json_sha256(
            _science_plain(module.input_graphs_covered)
        ),
        "eviction_count": int(
            getattr(getattr(live, "graph_map", None), "eviction_committed", 0)
            or 0
        ),
        "transition_registry_digest": stable_json_sha256(
            transition_scientific_state
        ),
        "transition_registry_state": transition_scientific_state,
        "live_graph_scientific_digest": stable_json_sha256(live_audit),
        "live_graph_scientific_state": live_audit,
    }


def _candidate_frequency_map(module: Any) -> dict[str, Any]:
    return {
        str(row.get("graph_hash")): _science_plain(
            row.get("frequency", row.get("count", row.get("importance")))
        )
        for row in module.counterfactual_candidates
        if isinstance(row, Mapping) and row.get("graph_hash") is not None
    }


def _candidate_population_snapshot(module: Any) -> dict[str, Any]:
    """Constant-boundary scientific state usable inside an active move."""

    candidates = [_science_plain(row) for row in module.counterfactual_candidates]
    registry = {
        str(key): int(value) for key, value in module.graph_index_map.items()
    }
    frequencies = _candidate_frequency_map(module)
    transitions = getattr(module, "transitions", None)
    transition_entries = getattr(transitions, "_entries", transitions)
    if not isinstance(transition_entries, Mapping):
        raise TypeError("Populate observer cannot inspect transition entries")
    transition_keys = sorted(str(key) for key in transition_entries)
    live_map = getattr(
        getattr(module, "comrecgc_live_graph_state", None), "graph_map", None
    )
    public = {
        "candidate_count": len(candidates),
        "candidate_records_sha256": stable_json_sha256(candidates),
        "candidate_ordered_hashes_sha256": stable_json_sha256(
            [str(row.get("graph_hash")) for row in module.counterfactual_candidates]
        ),
        "registry_count": len(registry),
        "registry_mapping_sha256": stable_json_sha256(registry),
        "candidate_frequency_sha256": stable_json_sha256(frequencies),
        "covering_graphs_sha256": stable_json_sha256(
            sorted(str(item) for item in module.covering_graphs)
        ),
        "input_graphs_covered": _science_plain(module.input_graphs_covered),
        "transition_entry_count": len(transition_keys),
        "transition_entry_keys_sha256": stable_json_sha256(transition_keys),
        "eviction_attempts": int(getattr(live_map, "eviction_attempts", 0) or 0),
        "eviction_committed": int(
            getattr(live_map, "eviction_committed", 0) or 0
        ),
        "eviction_deferred": int(getattr(live_map, "eviction_deferred", 0) or 0),
    }
    public["scientific_state_sha256"] = stable_json_sha256(public)
    return {
        "public": public,
        "registry": registry,
        "frequencies": frequencies,
        "transition_keys": transition_keys,
    }


def _candidate_population_event(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    event_index: int,
    scope: str,
    graph_hash: str,
    importance_parts: Any,
    input_graphs_covering_list: Any,
    bypass_size: bool,
) -> dict[str, Any]:
    before_registry = dict(before["registry"])
    after_registry = dict(after["registry"])
    before_frequencies = dict(before["frequencies"])
    after_frequencies = dict(after["frequencies"])
    before_keys = set(before_registry)
    after_keys = set(after_registry)
    removed = sorted(before_keys - after_keys)
    added = sorted(after_keys - before_keys)
    remapped = [
        {
            "graph_hash": key,
            "candidate_index_before": before_registry[key],
            "candidate_index_after": after_registry[key],
        }
        for key in sorted(before_keys & after_keys)
        if before_registry[key] != after_registry[key]
    ]
    frequency_changes = [
        {
            "graph_hash": key,
            "frequency_before": before_frequencies.get(key),
            "frequency_after": after_frequencies.get(key),
        }
        for key in sorted(set(before_frequencies) | set(after_frequencies))
        if before_frequencies.get(key) != after_frequencies.get(key)
    ]
    before_transition_keys = set(before["transition_keys"])
    after_transition_keys = set(after["transition_keys"])
    before_public = dict(before["public"])
    after_public = dict(after["public"])
    if removed and added:
        decision = "CAPACITY_REPLACEMENT"
    elif added:
        decision = "NEW_CANDIDATE_ADDED"
    elif frequency_changes or remapped:
        decision = "CANDIDATE_REINFORCED_OR_REORDERED"
    else:
        decision = "NO_CANDIDATE_SCIENTIFIC_CHANGE"
    event = {
        "event_index": int(event_index),
        "scope": str(scope),
        "event": "populate_counterfactual_candidates",
        "target_algorithm_graph_hash": str(graph_hash),
        "bypass_size": bool(bypass_size),
        "importance_parts": _science_plain(importance_parts),
        "input_graphs_covering_list": _science_plain(
            input_graphs_covering_list
        ),
        "decision": decision,
        "added_graph_hashes": added,
        "removed_graph_hashes": removed,
        "remapped_registry_entries": remapped,
        "frequency_changes": frequency_changes,
        "transition_entries_added": sorted(
            after_transition_keys - before_transition_keys
        ),
        "transition_entries_removed": sorted(
            before_transition_keys - after_transition_keys
        ),
        "eviction_attempts_delta": (
            int(after_public["eviction_attempts"])
            - int(before_public["eviction_attempts"])
        ),
        "eviction_committed_delta": (
            int(after_public["eviction_committed"])
            - int(before_public["eviction_committed"])
        ),
        "eviction_deferred_delta": (
            int(after_public["eviction_deferred"])
            - int(before_public["eviction_deferred"])
        ),
        "state_before": before_public,
        "state_after": after_public,
    }
    event["event_sha256"] = stable_json_sha256(event)
    return event


class _CommonStepObserver:
    """Observe identical state boundaries in both trace modes."""

    def __init__(self, path: Path, *, phase: str, trace_mode: str) -> None:
        self.path = path
        self.phase = phase
        self.trace_mode = trace_mode
        self.last_move: dict[str, Any] | None = None
        self.model_call_events: list[dict[str, Any]] = []
        self.transition_action_snapshot: list[dict[str, Any]] = []
        self.populate_events: list[dict[str, Any]] = []
        self.populate_event_scope = "pre_step_initialization"
        self.history_digest = "0" * 64
        if phase == "reload" and path.is_file():
            continuous = [
                row
                for row in _read_jsonl(path)
                if row.get("phase") == "continuous"
                and int(row.get("step", -1)) == STEPS_TO_COMPARE
            ]
            if len(continuous) != 1:
                raise ValueError("Reload observer cannot bind the unique continuous step 500")
            self.history_digest = str(continuous[0]["history_digest"])

    def install(self, runtime: Any) -> None:
        original_loop = runtime.run_generation_loop
        observer = self

        def observed_loop(module: Any, **kwargs: Any) -> Any:
            from src.baselines.comrecgc.graph_trace import (
                enumerate_official_single_edits,
                stable_untyped_graph_sha256,
            )
            from src.baselines.comrecgc.live_graph_state import (
                pin_graphs,
                resolve_graph,
                resolve_graphs,
            )

            original_move = module.move_to_next_graph
            original_call = module.call
            original_populate = module.populate_counterfactual_candidates
            original_boundary = kwargs.get("on_step_complete")
            transition_map = module.transitions
            original_transition_end = getattr(transition_map, "end_move", None)
            if not callable(original_transition_end) or not hasattr(
                transition_map, "_entries"
            ):
                raise TypeError(
                    "Common observer requires the pinned compact transition cache"
                )

            def binary_prediction_fields(raw_target_probability: Any) -> dict[str, Any]:
                target_probability = float(raw_target_probability)
                if not 0.0 <= target_probability <= 1.0:
                    raise ValueError(
                        "Observed internal-target probability is outside [0, 1]"
                    )
                source_probability = 1.0 - target_probability
                internal_prediction = int(target_probability >= 0.5)
                return {
                    "internal_target_probability": target_probability,
                    "internal_source_probability": source_probability,
                    "project_source_probability": source_probability,
                    "internal_class_probabilities_source0_target1": [
                        source_probability,
                        target_probability,
                    ],
                    "project_class_probabilities_label0_label1": [
                        target_probability,
                        source_probability,
                    ],
                    "internal_predicted_label": internal_prediction,
                    "project_predicted_label": (
                        0 if internal_prediction == 1 else 1
                    ),
                    "project_source_label": 1,
                    "strict_flip": bool(internal_prediction == 1),
                }

            def first_numeric(value: Any) -> float:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
                while isinstance(value, (list, tuple)):
                    if not value:
                        raise ValueError("Selected importance row is empty")
                    value = value[0]
                if hasattr(value, "item"):
                    value = value.item()
                return float(value)

            def observed_transition_end() -> Any:
                snapshot: list[dict[str, Any]] = []
                entries = getattr(transition_map, "_entries")
                # Restart sampling is with replacement, so multiple heads may
                # share one source.  The transition entry is authoritative per
                # source, not per repeated head occurrence.
                for source_hash in dict.fromkeys(
                    getattr(transition_map, "_active_keys", ())
                ):
                    entry = entries.get(source_hash)
                    if entry is None:
                        continue
                    for target_hash, action in zip(
                        entry.target_hashes, entry.actions, strict=True
                    ):
                        snapshot.append(
                            {
                                "source_official_hash": str(source_hash),
                                "target_official_hash": str(target_hash),
                                "action": _science_plain(action),
                            }
                        )
                observer.transition_action_snapshot = snapshot
                return original_transition_end()

            transition_map.end_move = observed_transition_end

            def observed_populate(*args: Any, **populate_kwargs: Any) -> Any:
                graph_hash = populate_kwargs.get(
                    "graph_hash", args[0] if args else None
                )
                importance_parts = populate_kwargs.get(
                    "importance_parts", args[1] if len(args) > 1 else None
                )
                input_graphs_covering_list = populate_kwargs.get(
                    "input_graphs_covering_list",
                    args[2] if len(args) > 2 else None,
                )
                bypass_size = bool(
                    populate_kwargs.get(
                        "bypass_size", args[3] if len(args) > 3 else False
                    )
                )
                before_population = _candidate_population_snapshot(module)
                result = original_populate(*args, **populate_kwargs)
                after_population = _candidate_population_snapshot(module)
                observer.populate_events.append(
                    _candidate_population_event(
                        before=before_population,
                        after=after_population,
                        event_index=len(observer.populate_events),
                        scope=observer.populate_event_scope,
                        graph_hash=str(graph_hash),
                        importance_parts=importance_parts,
                        input_graphs_covering_list=input_graphs_covering_list,
                        bypass_size=bypass_size,
                    )
                )
                return result

            module.populate_counterfactual_candidates = observed_populate

            def observed_call(*args: Any, **call_kwargs: Any) -> Any:
                import numpy as np

                graphs = list(call_kwargs.get("graphs", args[0] if args else []))
                result = original_call(*args, **call_kwargs)
                importance = np.asarray(result[0])
                embeddings = np.asarray(result[1])
                if (
                    importance.ndim != 2
                    or importance.shape[0] != len(graphs)
                    or importance.shape[1] < 1
                    or embeddings.shape[0] != len(graphs)
                ):
                    raise ValueError("Observed COMRECGC classifier output is malformed")
                rows = []
                for graph, raw_probability in zip(
                    graphs, importance[:, 0].tolist(), strict=True
                ):
                    graph_state = _science_plain(graph)
                    rows.append(
                        {
                            "candidate_graph_sha256": graph_state["graph_sha256"],
                            **binary_prediction_fields(raw_probability),
                        }
                    )
                observer.model_call_events.append(
                    {
                        "rows": rows,
                        "rows_sha256": stable_json_sha256(rows),
                        "embedding_output_sha256": stable_json_sha256(
                            _science_plain(embeddings)
                        ),
                    }
                )
                return result

            def observed_move(*args: Any, **move_kwargs: Any) -> Any:
                observer.model_call_events = []
                observer.transition_action_snapshot = []
                observer.populate_event_scope = "official_move"
                source_hashes = list(
                    move_kwargs.get("graphs_hash", args[0] if args else [])
                )
                registry_before = {
                    str(key) for key in module.graph_index_map.keys()
                }
                frequencies_before = _candidate_frequency_map(module)
                before = _candidate_state(module)
                random_module = getattr(module, "random", None)
                original_random_choices = getattr(random_module, "choices", None)
                lead_head_draws: list[int] = []

                def observed_random_choices(
                    population: Any, *choice_args: Any, **choice_kwargs: Any
                ) -> Any:
                    result = original_random_choices(
                        population, *choice_args, **choice_kwargs
                    )
                    weights = (
                        choice_kwargs.get("weights")
                        if "weights" in choice_kwargs
                        else (choice_args[0] if choice_args else None)
                    )
                    cum_weights = choice_kwargs.get("cum_weights")
                    count = int(choice_kwargs.get("k", 1))
                    try:
                        population_values = list(population)
                    except TypeError:
                        population_values = []
                    if (
                        weights is None
                        and cum_weights is None
                        and count == 1
                        and population_values == list(range(len(source_hashes)))
                        and isinstance(result, list)
                        and len(result) == 1
                    ):
                        lead_head_draws.append(int(result[0]))
                    return result

                if callable(original_random_choices):
                    random_module.choices = observed_random_choices
                try:
                    with pin_graphs(module, source_hashes):
                        source_graphs = resolve_graphs(module, source_hashes)
                        result = original_move(*args, **move_kwargs)
                finally:
                    if callable(original_random_choices):
                        random_module.choices = original_random_choices
                    observer.populate_event_scope = (
                        "post_move_restart_before_step_boundary"
                    )
                random_lead_head = (
                    lead_head_draws[0] if len(lead_head_draws) == 1 else None
                )
                random_lead_head_state = (
                    "EXACT_WRAPPED_RANDOM_CHOICES_NO_EXTRA_DRAW"
                    if len(lead_head_draws) == 1
                    else "UNAVAILABLE_OR_AMBIGUOUS"
                )
                with pin_graphs(module, source_hashes):
                    source_graphs = resolve_graphs(module, source_hashes)
                    target_hashes = [] if result[0] is None else list(result[0])
                    selected: list[dict[str, Any]] = []
                    if not bool(result[1]):
                        selected_importance_rows = list(result[3])
                        if len(selected_importance_rows) != len(target_hashes):
                            raise ValueError(
                                "Selected importance rows do not match output heads"
                            )
                        with pin_graphs(module, target_hashes):
                            for head, (source_hash, target_hash, source_graph) in enumerate(
                                zip(source_hashes, target_hashes, source_graphs, strict=True)
                            ):
                                target_graph = resolve_graph(module, target_hash)
                                graph_diff_error = None
                                try:
                                    actions = enumerate_official_single_edits(
                                        source_graph, target_graph
                                    )
                                except Exception as exc:
                                    actions = []
                                    graph_diff_error = {
                                        "error_class": type(exc).__name__,
                                        "message": str(exc),
                                    }
                                graph_diff_actions = [
                                    _science_plain(action) for action in actions
                                ]
                                exact_action_rows = [
                                    row
                                    for row in observer.transition_action_snapshot
                                    if row["source_official_hash"]
                                    == str(source_hash)
                                    and row["target_official_hash"]
                                    == str(target_hash)
                                ]
                                if len(exact_action_rows) != 1:
                                    raise ValueError(
                                        "Common observer cannot bind the exact compact-cache action"
                                    )
                                exact_action = exact_action_rows[0]["action"]
                                graph_diff_matches = [
                                    action
                                    for action in graph_diff_actions
                                    if action == exact_action
                                ]
                                if graph_diff_error is not None:
                                    graph_diff_state = "GRAPH_DIFF_DIAGNOSTIC_ERROR"
                                elif not graph_diff_actions:
                                    graph_diff_state = "NO_GRAPH_DIFF_ACTION"
                                elif len(graph_diff_actions) > 1:
                                    graph_diff_state = (
                                        "AMBIGUOUS_GRAPH_DIFF_ACTIONS"
                                    )
                                elif graph_diff_matches:
                                    graph_diff_state = (
                                        "UNIQUE_GRAPH_DIFF_MATCHES_TRANSITION"
                                    )
                                else:
                                    graph_diff_state = (
                                        "UNIQUE_GRAPH_DIFF_DIFFERS_FROM_TRANSITION"
                                    )
                                selected.append(
                                    {
                                        "head": head,
                                        "source_official_hash": str(source_hash),
                                        "target_official_hash": str(target_hash),
                                        "source_graph_sha256": stable_untyped_graph_sha256(
                                            source_graph
                                        ),
                                        "candidate_graph_sha256": stable_untyped_graph_sha256(
                                            target_graph
                                        ),
                                        "parent_id": str(
                                            getattr(target_graph, "comrecgc_parent_id", "")
                                        ),
                                        "action": exact_action,
                                        "action_source": (
                                            "compact_transition_cache_before_end_move"
                                        ),
                                        "action_gate": "transition_cache_exact",
                                        "graph_diff_action_diagnostic_only": True,
                                        "graph_diff_action_state": graph_diff_state,
                                        "graph_diff_action_candidates": (
                                            graph_diff_actions
                                        ),
                                        "graph_diff_diagnostic_error": (
                                            graph_diff_error
                                        ),
                                        "graph_diff_contains_transition_action": bool(
                                            graph_diff_matches
                                        ),
                                        **binary_prediction_fields(
                                            first_numeric(
                                                selected_importance_rows[head]
                                            )
                                        ),
                                        "probability_source": (
                                            "move_result_selected_importance"
                                        ),
                                    }
                                )
                after = _candidate_state(module)
                frequencies_after = _candidate_frequency_map(module)
                classifier_rows = [
                    row
                    for event in observer.model_call_events
                    for row in event["rows"]
                ]
                classifier_by_graph: dict[str, list[dict[str, Any]]] = {}
                for row in classifier_rows:
                    classifier_by_graph.setdefault(
                        str(row["candidate_graph_sha256"]), []
                    ).append(row)
                selected_classifier_outputs = []
                for row in selected:
                    observations = classifier_by_graph.get(
                        str(row["candidate_graph_sha256"]), []
                    )
                    target_probability = float(
                        row["internal_target_probability"]
                    )
                    if any(
                        float(observation["internal_target_probability"])
                        != target_probability
                        for observation in observations
                    ):
                        raise ValueError(
                            "Selected move importance differs from an observed "
                            "classifier output for the same graph"
                        )
                    selected_classifier_outputs.append(
                        {
                            "head": row["head"],
                            "candidate_graph_sha256": row[
                                "candidate_graph_sha256"
                            ],
                            "internal_target_probability": target_probability,
                            "internal_source_probability": row[
                                "internal_source_probability"
                            ],
                            "project_source_probability": row[
                                "project_source_probability"
                            ],
                            "internal_predicted_label": row[
                                "internal_predicted_label"
                            ],
                            "project_predicted_label": row[
                                "project_predicted_label"
                            ],
                            "strict_flip": row["strict_flip"],
                            "probability_source": (
                                "move_result_selected_importance_cross_checked_"
                                "model_call"
                                if observations
                                else "move_result_selected_importance_no_model_"
                                "call_observed_cache_hit_compatible"
                            ),
                            "model_call_cross_check_count": len(observations),
                            "observations": observations,
                        }
                    )
                candidate_decisions = []
                for row in selected:
                    duplicate_key = str(row["target_official_hash"])
                    before_frequency = frequencies_before.get(duplicate_key)
                    after_frequency = frequencies_after.get(duplicate_key)
                    was_duplicate = duplicate_key in frequencies_before
                    candidate_decisions.append(
                        {
                            "head": row["head"],
                            "algorithm_duplicate_key": duplicate_key,
                            "algorithm_duplicate_identity": (
                                "pinned_upstream_embedding_bytes_python_hash_seed0"
                            ),
                            "was_candidate_before_move": was_duplicate,
                            "was_registered_graph_before_move": (
                                duplicate_key in registry_before
                            ),
                            "frequency_before": before_frequency,
                            "frequency_after": after_frequency,
                            "accepted_or_reinforced": (
                                after_frequency is not None
                                and after_frequency != before_frequency
                            ),
                            "decision": (
                                "DUPLICATE_REINFORCED"
                                if was_duplicate
                                and after_frequency != before_frequency
                                else (
                                    "NEW_CANDIDATE_ACCEPTED"
                                    if not was_duplicate
                                    and after_frequency is not None
                                    else "NO_CANDIDATE_CHANGE"
                                )
                            ),
                        }
                    )
                records_changed = (
                    before["candidate_records_digest"]
                    != after["candidate_records_digest"]
                )
                frequencies_changed = (
                    before["candidate_frequency_digest"]
                    != after["candidate_frequency_digest"]
                )
                if after["candidate_count"] > before["candidate_count"]:
                    duplicate_decision = "NEW_CANDIDATE_ACCEPTED"
                elif records_changed or frequencies_changed:
                    duplicate_decision = (
                        "EXISTING_OR_CAPACITY_BOUND_CANDIDATE_UPDATED"
                    )
                else:
                    duplicate_decision = "NO_CANDIDATE_STATE_CHANGE"
                observer.last_move = {
                    "selected_transitions": selected,
                    "teleported": bool(result[1]),
                    "recourse": _science_plain(result[2]),
                    "selected_importance": _science_plain(result[3]),
                    "selected_diff": _science_plain(result[4]),
                    "classifier_output_contract": (
                        "binary_probabilities_derived_from_exact_internal_target_"
                        "probability_consumed_by_upstream_move_v1"
                    ),
                    "random_lead_head": random_lead_head,
                    "random_lead_head_observation_state": random_lead_head_state,
                    "random_lead_matching_draw_count": len(lead_head_draws),
                    "selected_transition_head_semantics": (
                        "per_output_head_index_distinct_from_random_lead_head"
                    ),
                    "classifier_call_count": len(observer.model_call_events),
                    "classifier_observation_count": len(classifier_rows),
                    "classifier_calls_sha256": stable_json_sha256(
                        observer.model_call_events
                    ),
                    "selected_classifier_outputs": selected_classifier_outputs,
                    "candidate_decisions": candidate_decisions,
                    "candidate_state_before": before,
                    "candidate_state_after_move": after,
                    "candidate_accepted": records_changed or frequencies_changed,
                    "duplicate_decision": duplicate_decision,
                    "accept_reject_digest": stable_json_sha256(
                        {"before": before, "after": after}
                    ),
                    "lineage_predecessor_digest": stable_json_sha256(
                        _equivalence_scientific_projection(selected)
                    ),
                    "lineage_downstream_digest": stable_json_sha256(
                        {
                            "selected": _equivalence_scientific_projection(
                                selected
                            ),
                            "candidate_records": after[
                                "candidate_records_digest"
                            ],
                        }
                    ),
                }
                return result

            def observed_boundary(state: Any) -> None:
                if original_boundary is not None:
                    original_boundary(state)
                if observer.last_move is None:
                    raise ValueError("Completed step has no common move observation")
                populate_events = list(observer.populate_events)
                if [row.get("event_index") for row in populate_events] != list(
                    range(len(populate_events))
                ):
                    raise ValueError("Populate event indices are not contiguous")
                observer.last_move = {
                    **observer.last_move,
                    "populate_event_count": len(populate_events),
                    "populate_events": populate_events,
                    "populate_events_sha256": stable_json_sha256(
                        populate_events
                    ),
                    "candidate_state_at_step_boundary": _candidate_state(module),
                }
                scientific = {
                    "step": int(state.completed_step),
                    "next_step": int(state.next_step),
                    "start_graph_hashes": [str(item) for item in state.start_graph_hashes],
                    "current_graph_hashes": [str(item) for item in state.current_graph_hashes],
                    "restart_indices": [int(item) for item in state.restart_indices],
                    "rng_state_sha256": stable_json_sha256(_rng_state()),
                    "move": observer.last_move,
                    "candidate_state": _candidate_state(module),
                }
                step_digest = stable_json_sha256(
                    _equivalence_scientific_projection(scientific)
                )
                observer.history_digest = hashlib.sha256(
                    bytes.fromhex(observer.history_digest) + bytes.fromhex(step_digest)
                ).hexdigest()
                _append_jsonl(
                    observer.path,
                    {
                        **scientific,
                        "schema_version": "mut_trace_common_step_state_v1",
                        "phase": observer.phase,
                        "trace_mode": observer.trace_mode,
                        "scientific_checkpoint_digest": step_digest,
                        "history_digest": observer.history_digest,
                    },
                )
                observer.populate_events = []
                observer.populate_event_scope = "between_completed_steps"
                if int(state.completed_step) == (
                    STEPS_TO_COMPARE + POST_RELOAD_STEPS
                ):
                    # The complete JSONL row is fsync'd before this exact-PID
                    # self-stop.  No step-511 code can run and no broad process
                    # lookup or extra RNG draw is introduced.
                    os.kill(os.getpid(), signal.SIGTERM)

            module.move_to_next_graph = observed_move
            module.call = observed_call
            kwargs["on_step_complete"] = observed_boundary
            try:
                return original_loop(module, **kwargs)
            finally:
                module.move_to_next_graph = original_move
                module.call = original_call
                module.populate_counterfactual_candidates = original_populate
                transition_map.end_move = original_transition_end

        runtime.run_generation_loop = observed_loop


def _install_science_root(root: Path) -> None:
    cached = sorted(
        name for name in sys.modules if name == "src" or name.startswith("src.")
    )
    if cached:
        raise RuntimeError(
            "Scientific modules were imported before the pinned root was installed: "
            + ",".join(cached[:8])
        )
    retained: list[str] = []
    for value in sys.path:
        try:
            resolved = Path(value).resolve()
        except (OSError, ValueError):
            retained.append(value)
            continue
        if (resolved / "src").is_dir() or resolved.name == "autodl":
            continue
        retained.append(value)
    sys.path[:] = [str(root), *retained]


def _run_one(args: argparse.Namespace) -> int:
    science_root = _absolute(args.science_project_root)
    if _git_head(science_root) != INSTRUMENTATION_COMMIT:
        raise ValueError("Both trace arms require the exact instrumentation checkout")
    output = _absolute(args.output_root, exists=bool(args.resume))
    if not args.resume and output.exists():
        raise FileExistsError(f"Fresh trace-mode arm already exists: {output}")
    _install_science_root(science_root)
    from src.baselines.comrecgc.contracts import GenerationParameters
    from src.baselines.comrecgc.generation_checkpoint import scientific_command_sha256
    import src.baselines.comrecgc.runtime as runtime
    import src.baselines.comrecgc.contracts as contracts
    import src.baselines.comrecgc.generation_checkpoint as generation_checkpoint
    import src.baselines.comrecgc.graph_trace as graph_trace

    for module in (contracts, generation_checkpoint, graph_trace, runtime):
        _assert_module_under(module, science_root)

    parameters = GenerationParameters.for_mode("full")
    if int(parameters.steps) != FORMAL_M_MAX or int(parameters.candidate_capacity) != 100_000:
        raise ValueError("Formal trace-mode parameters changed")
    observer = _CommonStepObserver(
        _absolute(args.observer_output, exists=bool(args.resume)),
        phase=args.phase,
        trace_mode=args.trace_mode,
    )
    observer.install(runtime)
    scientific_argv = (
        "mut_trace_on_off_equivalence_v1",
        "dataset=mutagenicity",
        f"source_algorithm_commit={SOURCE_COMMIT}",
        f"execution_commit={INSTRUMENTATION_COMMIT}",
        f"trace_mode={args.trace_mode}",
        f"parent_limit={int(args.parent_limit)}",
        f"batch_size={int(args.batch_size)}",
        f"device={str(args.device)}",
        f"M_MAX={FORMAL_M_MAX}",
        f"candidate_capacity={parameters.candidate_capacity}",
        f"seed={parameters.seed}",
        f"upstream={_absolute(args.upstream_root)}",
        f"dataset_dir={_absolute(args.dataset_dir)}",
        f"gnn={_absolute(args.gnn_checkpoint)}",
        f"distance={_absolute(args.distance_checkpoint)}",
        f"output={output}",
    )
    trace_dir = output / "trace" if args.trace_mode == "on" else None
    runtime.run_project_generation(
        project_root=science_root,
        upstream_root=_absolute(args.upstream_root),
        dataset="mutagenicity",
        dataset_dir=_absolute(args.dataset_dir),
        source_csv=None,
        gnn_checkpoint=_absolute(args.gnn_checkpoint),
        distance_checkpoint=_absolute(args.distance_checkpoint),
        output_dir=output,
        mode="full",
        parent_limit=int(args.parent_limit),
        parameters=parameters,
        device=str(args.device),
        batch_size=int(args.batch_size),
        resume=bool(args.resume),
        trace_output_dir=trace_dir,
        parity_reference_path=None,
        graph_state_dir=output / "graph_state",
        storage_guard_root=output,
        storage_check_every_steps=250,
        storage_min_free_bytes=50 * 1024**3,
        storage_min_free_ratio=0.02,
        storage_min_free_inodes=100_000,
        checkpoint_root=output / "generation_checkpoints",
        checkpoint_mirror_root=_absolute(args.checkpoint_mirror_root, exists=False),
        checkpoint_interval_steps=500,
        checkpoint_keep_last=2,
        progress_interval_steps=25,
        scientific_argv=scientific_argv,
        command_sha256=scientific_command_sha256(scientific_argv),
    )
    raise RuntimeError("Trace diagnostic unexpectedly exhausted the formal 50k budget")


def _read_jsonl(
    path: Path, *, allow_unterminated_live_tail: bool = False
) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        complete = raw
    elif allow_unterminated_live_tail:
        final_newline = raw.rfind(b"\n")
        complete = b"" if final_newline < 0 else raw[: final_newline + 1]
    else:
        raise ValueError(f"Observer JSONL has an unterminated final row: {path}")
    rows: list[dict[str, Any]] = []
    for number, encoded in enumerate(complete.splitlines(), start=1):
        if not encoded:
            raise ValueError(f"Empty observer JSONL line {number}: {path}")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid observer JSONL line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Observer row is not one object: {path}:{number}")
        rows.append(dict(value))
    return rows


def _row_science(row: Mapping[str, Any]) -> dict[str, Any]:
    return _equivalence_scientific_projection(
        {
            key: value
            for key, value in row.items()
            if key not in {"phase", "trace_mode", "schema_version"}
        }
    )


def _phase_rows(
    path: Path,
    phase: str,
    *,
    allow_unterminated_live_tail: bool = False,
) -> dict[int, dict[str, Any]]:
    selected = [
        row
        for row in _read_jsonl(
            path,
            allow_unterminated_live_tail=allow_unterminated_live_tail,
        )
        if row.get("phase") == phase
    ]
    result: dict[int, dict[str, Any]] = {}
    for row in selected:
        step = int(row.get("step", -1))
        if step in result:
            raise ValueError(f"Duplicate observer phase/step: {path}:{phase}:{step}")
        result[step] = row
    return result


def _validate_observer_log(
    path: Path, *, trace_mode: str, require_reload: bool
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    expected = [
        ("continuous", step) for step in range(1, STEPS_TO_COMPARE + POST_RELOAD_STEPS + 1)
    ]
    if require_reload:
        expected.extend(
            ("reload", step)
            for step in range(
                STEPS_TO_COMPARE + 1,
                STEPS_TO_COMPARE + POST_RELOAD_STEPS + 1,
            )
        )
    actual: list[tuple[str, int]] = []
    for line_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != "mut_trace_common_step_state_v1":
            raise ValueError(
                f"Observer schema mismatch at {path}:{line_number}"
            )
        if row.get("trace_mode") != trace_mode:
            raise ValueError(
                f"Observer trace mode mismatch at {path}:{line_number}"
            )
        phase = str(row.get("phase"))
        step = int(row.get("step", -1))
        if int(row.get("next_step", -1)) != step + 1:
            raise ValueError(
                f"Observer next-step mismatch at {path}:{line_number}"
            )
        actual.append((phase, step))
    if actual != expected:
        raise ValueError(
            "Observer JSONL is not the exact phase-scoped prefix: "
            f"path={path} expected_rows={len(expected)} actual_rows={len(actual)} "
            f"first_mismatch={next((index for index, pair in enumerate(zip(actual, expected, strict=False), start=1) if pair[0] != pair[1]), None)}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "continuous_steps": [1, STEPS_TO_COMPARE + POST_RELOAD_STEPS],
        "reload_steps": (
            [STEPS_TO_COMPARE + 1, STEPS_TO_COMPARE + POST_RELOAD_STEPS]
            if require_reload
            else []
        ),
        "duplicates": 0,
        "extra_rows": 0,
        "unterminated_tail": False,
        "status": "PASS",
    }


def _stop_after(
    command: Sequence[str],
    *,
    log: Path,
    active: Path,
    mode: str,
    phase: str,
    marker: Path,
    observer: Path,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    started = time.time()
    process: subprocess.Popen[Any] | None = None
    start_ticks: int | None = None
    stop_reason: str | None = None
    timed_out = False
    returncode: int | None = None
    exact_boundary_seen_at: float | None = None
    with log.open("a", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen(
                list(command), env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            for _attempt in range(20):
                try:
                    start_ticks = _proc_start_ticks(process.pid)
                    break
                except (OSError, ValueError, IndexError):
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
            if start_ticks is None:
                raise RuntimeError(
                    f"Cannot bind {mode}/{phase} to one /proc start-tick identity"
                )
            _atomic_json(
                active,
                {
                    "schema_version": "mut_trace_active_arm_v1",
                    "trace_mode": mode,
                    "phase": phase,
                    "pid": process.pid,
                    "pid_start_ticks": start_ticks,
                    "started_at_unix": started,
                },
            )
            deadline = time.monotonic() + 72 * 60 * 60
            while process.poll() is None:
                rows = _phase_rows(
                    observer,
                    phase,
                    allow_unterminated_live_tail=True,
                )
                observed_max = max(rows, default=0)
                exact_stop_step = STEPS_TO_COMPARE + POST_RELOAD_STEPS
                if observed_max > exact_stop_step:
                    stop_reason = "observer_overran_exact_step_510_boundary"
                    break
                if marker.is_file() and observed_max == exact_stop_step:
                    if exact_boundary_seen_at is None:
                        exact_boundary_seen_at = time.monotonic()
                    elif time.monotonic() - exact_boundary_seen_at > 120:
                        stop_reason = "exact_boundary_self_stop_timeout"
                        break
                if time.monotonic() >= deadline:
                    timed_out = True
                    stop_reason = "72_hour_timeout"
                    break
                time.sleep(2)
            if process.poll() is not None and stop_reason is None:
                returncode = int(process.returncode or 0)
                if returncode in {
                    -signal.SIGTERM,
                    128 + signal.SIGTERM,
                } and marker.is_file():
                    _validate_observer_log(
                        observer,
                        trace_mode=mode,
                        require_reload=phase == "reload",
                    )
                    stop_reason = "self_signaled_exact_step_510_boundary"
                else:
                    stop_reason = "unexpected_process_exit"
        finally:
            if process is not None and process.poll() is None:
                if start_ticks is None or _proc_start_ticks(process.pid) != start_ticks:
                    raise RuntimeError(
                        f"Refusing ambiguous SIGTERM for {mode}/{phase}; PID identity changed"
                    )
                process.send_signal(signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=120)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"{mode}/{phase} ignored SIGTERM; SIGKILL is forbidden"
                    ) from exc
            elif process is not None and returncode is None:
                returncode = int(process.returncode or 0)
    if process is None or start_ticks is None or returncode is None:
        raise RuntimeError(f"{mode}/{phase} produced no exact process receipt")
    if timed_out:
        raise TimeoutError(f"{mode}/{phase} did not reach step 510 in 72 hours")
    if stop_reason != "self_signaled_exact_step_510_boundary":
        raise RuntimeError(
            f"{mode}/{phase} exited before the verified boundary: "
            f"reason={stop_reason}, returncode={returncode}"
        )
    if returncode not in {-signal.SIGTERM, 128 + signal.SIGTERM}:
        raise RuntimeError(f"{mode}/{phase} exited unexpectedly: {returncode}")
    stopped = time.time()
    receipt = {
        "trace_mode": mode,
        "phase": phase,
        "pid": process.pid,
        "pid_start_ticks": start_ticks,
        "returncode": returncode,
        "checkpoint_500": str(marker.parent),
        "checkpoint_500_complete": marker.is_file(),
        "observed_through_step": max(_phase_rows(observer, phase), default=0),
        "started_at_unix": started,
        "stopped_at_unix": stopped,
        "signal": "SIGTERM_SELF_AT_EXACT_BOUNDARY",
        "stop_reason": stop_reason,
    }
    _atomic_json(active, {**receipt, "state": "ARM_STOPPED_AT_VERIFIED_BOUNDARY"})
    return receipt


def _command(args: argparse.Namespace, *, mode: str, phase: str, arm: Path) -> list[str]:
    return [
        str(_absolute(args.python)),
        str(Path(__file__).resolve()),
        "run-one",
        "--science-project-root", str(_absolute(args.execution_project_root)),
        "--output-root", str(arm),
        "--checkpoint-mirror-root", str(arm / "checkpoint-mirror"),
        "--observer-output", str(arm / "common_step_state.jsonl"),
        "--trace-mode", mode,
        "--phase", phase,
        "--upstream-root", str(_absolute(args.upstream_root)),
        "--dataset-dir", str(_absolute(args.dataset_dir)),
        "--gnn-checkpoint", str(_absolute(args.gnn_checkpoint)),
        "--distance-checkpoint", str(_absolute(args.distance_checkpoint)),
        "--parent-limit", str(int(args.parent_limit)),
        "--device", str(args.device),
        "--batch-size", str(int(args.batch_size)),
        *(["--resume"] if phase == "reload" else []),
    ]


def _run_pair(args: argparse.Namespace) -> int:
    legacy = _absolute(args.legacy_project_root)
    execution = _absolute(args.execution_project_root)
    if _git_head(legacy) != SOURCE_COMMIT or _git_head(execution) != INSTRUMENTATION_COMMIT:
        raise ValueError("Pinned trace review worktree commit changed")
    if _git_science_status(legacy) or _git_science_status(execution):
        raise ValueError("Pinned trace review worktree has dirty/shadow source")
    upstream = _absolute(args.upstream_root)
    if _git_head(upstream) != UPSTREAM_COMMIT or _git_science_status(upstream):
        raise ValueError("Pinned upstream COMRECGC checkout changed")
    if (
        int(args.parent_limit) != 1448
        or int(args.batch_size) != 128
        or str(args.device) != "cuda:0"
    ):
        raise ValueError("Frozen Mut trace-mode launch parameters changed")
    legacy_inventory = _source_inventory(legacy)
    execution_inventory = _source_inventory(execution)
    if legacy_inventory.get("inventory_sha256") != LEGACY_SOURCE_INVENTORY_SHA256:
        raise ValueError("Historical source inventory changed")
    if execution_inventory.get("inventory_sha256") != INSTRUMENTATION_SOURCE_INVENTORY_SHA256:
        raise ValueError("Instrumentation source inventory changed")
    source_delta = _source_delta_audit(legacy_inventory, execution_inventory)
    if source_delta.get("status") != "PASS":
        raise ValueError("Checkpoint instrumentation source delta is not reviewed")
    run_root = _absolute(args.run_root, exists=False)
    output = _absolute(args.output_dir, exists=False)
    if run_root.exists() or output.exists():
        raise FileExistsError("Trace-mode roots must be fresh")
    run_root.mkdir(parents=True)
    active = run_root / "active_arm.json"
    historical_root = _absolute(args.historical_artifact_root)
    historical_manifest_path = historical_root / "run_manifest.json"
    historical_manifest = _physical_json(historical_manifest_path)
    dataset_audit = historical_manifest.get("dataset_audit")
    if not isinstance(dataset_audit, Mapping):
        raise ValueError("Historical Mut dataset audit is absent")
    expected_historical = {
        "dataset": "mutagenicity",
        "project_commit": SOURCE_COMMIT,
        "upstream_commit": (
            "122f9341a360e9f06bb58a2f5823bb596021f6bf"
        ),
        "config_sha256": HISTORICAL_CONFIG_SHA256,
        "generation_parent_ids_sha256": HISTORICAL_SOURCE_COHORT_SHA256,
        "trace_enabled": True,
        "internal_prediction_counts": HISTORICAL_INTERNAL_PREDICTION_COUNTS,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    changed_historical = sorted(
        key
        for key, expected in expected_historical.items()
        if historical_manifest.get(key) != expected
    )
    if changed_historical:
        raise ValueError(
            f"Historical input manifest changed: {changed_historical}"
        )
    if (
        dataset_audit.get("dataset_fingerprint") != HISTORICAL_DATASET_SHA256
        or dataset_audit.get("generation_parent_ids_sha256")
        != HISTORICAL_SOURCE_COHORT_SHA256
        or int(dataset_audit.get("node_feature_dim", -1)) != 10
        or int(dataset_audit.get("edge_feature_dim", -1)) != 0
    ):
        raise ValueError("Historical dataset/split/feature schema changed")
    gnn_path = _absolute(args.gnn_checkpoint)
    distance_path = _absolute(args.distance_checkpoint)
    rf_oracle_path = _absolute(args.rf_oracle)
    if sha256_file(gnn_path) != HISTORICAL_GNN_SHA256:
        raise ValueError("Historical Mut generation GNN changed")
    if sha256_file(distance_path) != HISTORICAL_DISTANCE_SHA256:
        raise ValueError("Historical Mut NeuroSED checkpoint changed")
    if sha256_file(rf_oracle_path) != HISTORICAL_RF_ORACLE_SHA256:
        raise ValueError("Frozen downstream Mut RF oracle changed")
    dataset_dir = _absolute(args.dataset_dir)
    dataset_files = {}
    for name in ("dataset_summary.json", "generation_source_graphs.pt"):
        path = dataset_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Frozen Mut dataset input is absent: {path}")
        dataset_files[name] = sha256_file(path)
    feature_schema = {
        "atom_types": dataset_audit.get("atom_types"),
        "node_feature_dim": dataset_audit.get("node_feature_dim"),
        "edge_feature_dim": dataset_audit.get("edge_feature_dim"),
        "label_semantics": dataset_audit.get("label_semantics"),
    }
    input_manifest = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "dataset": "mutagenicity",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": INSTRUMENTATION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "legacy_source_inventory_sha256": LEGACY_SOURCE_INVENTORY_SHA256,
        "instrumentation_source_inventory_sha256": INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
        "formal_M_MAX": FORMAL_M_MAX,
        "comparison_steps": STEPS_TO_COMPARE,
        "post_reload_steps": POST_RELOAD_STEPS,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": int(args.parent_limit),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "arms_sequential": True,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "historical_artifact_root": str(historical_root),
        "historical_run_manifest": str(historical_manifest_path),
        "historical_run_manifest_sha256": sha256_file(
            historical_manifest_path
        ),
        "historical_config_sha256": HISTORICAL_CONFIG_SHA256,
        "dataset_sha256": HISTORICAL_DATASET_SHA256,
        "split_source_cohort_sha256": HISTORICAL_SOURCE_COHORT_SHA256,
        "feature_schema": feature_schema,
        "feature_schema_sha256": stable_json_sha256(feature_schema),
        "algorithm_registry_identity": (
            "pinned_upstream_embedding_bytes_python_hash_seed0"
        ),
        "audit_graph_identity": "stable_untyped_graph_sha256",
        "pythonhashseed": "0",
        "candidate_predicate": {
            "project_source_label": 1,
            "internal_source_label": 0,
            "internal_target_label": 1,
            "project_to_internal_mapping": {"1": 0, "0": 1},
            "strict_flip": "internal_target_probability>=0.5",
        },
        "historical_internal_prediction_counts": (
            HISTORICAL_INTERNAL_PREDICTION_COUNTS
        ),
        "rf_oracle": {
            "path": str(rf_oracle_path),
            "sha256": sha256_file(rf_oracle_path),
            "role": "frozen_downstream_unified_evaluation_only",
            "loaded_by_generation_canary": False,
        },
        "input_files": {
            "dataset_dir": str(dataset_dir),
            "dataset_files": dataset_files,
            "gnn_checkpoint": {
                "path": str(gnn_path),
                "sha256": sha256_file(gnn_path),
            },
            "distance_checkpoint": {
                "path": str(distance_path),
                "sha256": sha256_file(distance_path),
            },
        },
    }
    input_manifest["manifest_sha256"] = stable_json_sha256(input_manifest)
    _atomic_json(run_root / "equivalence_input_manifest.json", input_manifest)
    arm_receipts: list[dict[str, Any]] = []
    for mode in ("on", "off"):
        arm = run_root / f"trace_{mode}"
        observer = arm / "common_step_state.jsonl"
        marker = arm / "checkpoint-mirror/step-000000000500/_CHECKPOINT_MIRRORED.json"
        arm_receipts.append(
            _stop_after(
                _command(args, mode=mode, phase="continuous", arm=arm),
                log=run_root / f"trace_{mode}_continuous.log",
                active=active,
                mode=mode,
                phase="continuous",
                marker=marker,
                observer=observer,
            )
        )
        arm_receipts.append(
            _stop_after(
                _command(args, mode=mode, phase="reload", arm=arm),
                log=run_root / f"trace_{mode}_reload.log",
                active=active,
                mode=mode,
                phase="reload",
                marker=marker,
                observer=observer,
            )
        )
    overlap_audit = _execution_overlap(arm_receipts)
    trace_on_config = _resolved_config_audit(
        run_root / "trace_on", mode="on"
    )
    trace_off_config = _resolved_config_audit(
        run_root / "trace_off", mode="off"
    )
    trace_on_checkpoint = _checkpoint_state_audit(
        run_root / "trace_on/checkpoint-mirror/step-000000000500",
        mode="on",
    )
    trace_off_checkpoint = _checkpoint_state_audit(
        run_root / "trace_off/checkpoint-mirror/step-000000000500",
        mode="off",
    )
    on_path = run_root / "trace_on/common_step_state.jsonl"
    off_path = run_root / "trace_off/common_step_state.jsonl"
    trace_on_observer_log_audit = _validate_observer_log(
        on_path, trace_mode="on", require_reload=True
    )
    trace_off_observer_log_audit = _validate_observer_log(
        off_path, trace_mode="off", require_reload=True
    )
    on_cont = _phase_rows(on_path, "continuous")
    off_cont = _phase_rows(off_path, "continuous")
    on_reload = _phase_rows(on_path, "reload")
    off_reload = _phase_rows(off_path, "reload")
    failures: list[str] = []
    first_divergence: int | None = None
    artifact_diff: list[dict[str, Any]] = []
    for step in range(1, STEPS_TO_COMPARE + 1):
        left = on_cont.get(step)
        right = off_cont.get(step)
        exact = left is not None and right is not None and _row_science(left) == _row_science(right)
        if not exact:
            first_divergence = step if first_divergence is None else first_divergence
            artifact_diff.append({"step": step, "comparison": "trace_on_vs_trace_off", "exact": False})
    for mode, continuous, reloaded in (
        ("on", on_cont, on_reload),
        ("off", off_cont, off_reload),
    ):
        for step in range(STEPS_TO_COMPARE + 1, STEPS_TO_COMPARE + POST_RELOAD_STEPS + 1):
            exact = (
                continuous.get(step) is not None
                and reloaded.get(step) is not None
                and _row_science(continuous[step]) == _row_science(reloaded[step])
            )
            if not exact:
                failures.append(f"{mode}_checkpoint_reload_step_{step}")
                artifact_diff.append({"step": step, "comparison": f"trace_{mode}_reload", "exact": False})
    for step in range(STEPS_TO_COMPARE + 1, STEPS_TO_COMPARE + POST_RELOAD_STEPS + 1):
        if (
            on_reload.get(step) is None
            or off_reload.get(step) is None
            or _row_science(on_reload[step]) != _row_science(off_reload[step])
        ):
            failures.append(f"post_reload_trace_mode_step_{step}")
    if first_divergence is not None:
        failures.append("trace_on_off_stepwise_semantics")
    for mode, path in (("on", on_path), ("off", off_path)):
        rows = _read_jsonl(path)
        if not rows:
            failures.append(f"trace_{mode}_observer_empty")
    def has_semantic_fields(row: Mapping[str, Any]) -> bool:
        move = row.get("move")
        if not isinstance(move, Mapping):
            return False
        selected = move.get("selected_transitions") or []
        outputs = move.get("selected_classifier_outputs") or []
        populate_events = move.get("populate_events") or []
        if (
            move.get("classifier_calls_sha256") is None
            or move.get("duplicate_decision")
            not in {
                "NEW_CANDIDATE_ACCEPTED",
                "EXISTING_OR_CAPACITY_BOUND_CANDIDATE_UPDATED",
                "NO_CANDIDATE_STATE_CHANGE",
            }
            or move.get("selected_transition_head_semantics")
            != "per_output_head_index_distinct_from_random_lead_head"
            or move.get("random_lead_head_observation_state")
            not in {
                "EXACT_WRAPPED_RANDOM_CHOICES_NO_EXTRA_DRAW",
                "UNAVAILABLE_OR_AMBIGUOUS",
            }
            or len(outputs) != len(selected)
            or move.get("populate_event_count") != len(populate_events)
            or not populate_events
            or move.get("populate_events_sha256")
            != stable_json_sha256(populate_events)
            or [event.get("event_index") for event in populate_events]
            != list(range(len(populate_events)))
        ):
            return False
        required_output_fields = {
            "internal_target_probability",
            "internal_source_probability",
            "project_source_probability",
            "internal_predicted_label",
            "project_predicted_label",
            "strict_flip",
            "probability_source",
            "model_call_cross_check_count",
            "observations",
        }
        classifier_fields_present = all(
            isinstance(output, Mapping)
            and required_output_fields.issubset(output)
            and isinstance(output.get("observations"), list)
            and output.get("probability_source")
            in {
                "move_result_selected_importance_cross_checked_model_call",
                "move_result_selected_importance_no_model_call_observed_"
                "cache_hit_compatible",
            }
            for output in outputs
        )
        action_fields_present = all(
            isinstance(transition, Mapping)
            and transition.get("action_source")
            == "compact_transition_cache_before_end_move"
            and transition.get("action_gate") == "transition_cache_exact"
            and transition.get("graph_diff_action_diagnostic_only") is True
            and isinstance(
                transition.get("graph_diff_action_candidates"), list
            )
            for transition in selected
        )
        populate_fields_present = all(
            isinstance(event, Mapping)
            and event.get("event") == "populate_counterfactual_candidates"
            and event.get("scope")
            in {
                "pre_step_initialization",
                "official_move",
                "post_move_restart_before_step_boundary",
            }
            and event.get("decision")
            in {
                "CAPACITY_REPLACEMENT",
                "NEW_CANDIDATE_ADDED",
                "CANDIDATE_REINFORCED_OR_REORDERED",
                "NO_CANDIDATE_SCIENTIFIC_CHANGE",
            }
            and event.get("event_sha256")
            == stable_json_sha256(
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_sha256"
                }
            )
            and isinstance(event.get("state_before"), Mapping)
            and isinstance(event.get("state_after"), Mapping)
            for event in populate_events
        )
        return (
            classifier_fields_present
            and action_fields_present
            and populate_fields_present
        )

    semantic_fields_present = all(
        has_semantic_fields(row) for row in on_cont.values()
    ) and all(has_semantic_fields(row) for row in off_cont.values())
    if not semantic_fields_present:
        failures.append("step_semantic_field_coverage")
    rng_state_exact = all(
        on_cont.get(step, {}).get("rng_state_sha256") is not None
        and on_cont[step].get("rng_state_sha256")
        == off_cont.get(step, {}).get("rng_state_sha256")
        for step in range(1, STEPS_TO_COMPARE + 1)
    )
    step_action_trace_exact = all(
        _equivalence_scientific_projection(
            on_cont.get(step, {}).get("move", {}).get("selected_transitions")
        )
        == _equivalence_scientific_projection(
            off_cont.get(step, {}).get("move", {}).get("selected_transitions")
        )
        for step in range(1, STEPS_TO_COMPARE + 1)
    )
    classifier_probability_trace_exact = all(
        on_cont.get(step, {}).get("move", {}).get("classifier_calls_sha256")
        is not None
        and on_cont[step]["move"].get("classifier_calls_sha256")
        == off_cont.get(step, {}).get("move", {}).get(
            "classifier_calls_sha256"
        )
        for step in range(1, STEPS_TO_COMPARE + 1)
    )
    step500_candidate_records_exact = (
        trace_on_checkpoint["serialized_candidate_records_sha256"]
        == trace_off_checkpoint["serialized_candidate_records_sha256"]
    )
    step500_candidate_universe_exact = (
        trace_on_checkpoint["candidate_universe_sha256"]
        == trace_off_checkpoint["candidate_universe_sha256"]
    )
    checkpoint_algorithm_scientific_state_exact = (
        trace_on_checkpoint["algorithm_scientific_state_sha256"]
        == trace_off_checkpoint["algorithm_scientific_state_sha256"]
    )
    checkpoint_algorithm_full_state_exact = (
        trace_on_checkpoint["algorithm_full_state_sha256"]
        == trace_off_checkpoint["algorithm_full_state_sha256"]
    )
    checkpoint_rng_state_exact = (
        trace_on_checkpoint["rng_state_sha256"]
        == trace_off_checkpoint["rng_state_sha256"]
    )
    checkpoint_sqlite_logical_state_exact = (
        trace_on_checkpoint["sqlite"]["logical_database_sha256"]
        == trace_off_checkpoint["sqlite"]["logical_database_sha256"]
    )
    checkpoint_graph_registry_exact = (
        trace_on_checkpoint["graph_registry_mapping_sha256"]
        == trace_off_checkpoint["graph_registry_mapping_sha256"]
    )
    resolved_config_scientific_binding_exact = (
        trace_on_config["status"] == "PASS"
        and trace_off_config["status"] == "PASS"
        and trace_on_config["scientific_binding_sha256"]
        == trace_off_config["scientific_binding_sha256"]
    )
    checkpoint_gates = {
        "step500_checkpoint_serialized_candidate_records_exact": (
            step500_candidate_records_exact
        ),
        "step500_checkpoint_candidate_universe_exact": (
            step500_candidate_universe_exact
        ),
        "checkpoint_algorithm_scientific_state_exact": (
            checkpoint_algorithm_scientific_state_exact
        ),
        "checkpoint_rng_state_exact": checkpoint_rng_state_exact,
        "checkpoint_sqlite_logical_state_exact": (
            checkpoint_sqlite_logical_state_exact
        ),
        "checkpoint_graph_registry_exact": checkpoint_graph_registry_exact,
        "resolved_config_scientific_binding_exact": (
            resolved_config_scientific_binding_exact
        ),
        "arms_sequential_from_process_intervals": (
            overlap_audit["arms_overlapped"] is False
            and int(overlap_audit["max_concurrent_arms"]) == 1
        ),
        "trace_on_observer_log_exact": (
            trace_on_observer_log_audit["status"] == "PASS"
        ),
        "trace_off_observer_log_exact": (
            trace_off_observer_log_audit["status"] == "PASS"
        ),
    }
    failures.extend(
        key for key, passed in checkpoint_gates.items() if passed is not True
    )
    if not rng_state_exact:
        failures.append("stepwise_rng_state")
    if not step_action_trace_exact:
        failures.append("stepwise_selected_action")
    if not classifier_probability_trace_exact:
        failures.append("stepwise_classifier_probability")
    output.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": INSTRUMENTATION_COMMIT,
        "formal_M_MAX": FORMAL_M_MAX,
        "steps_compared": STEPS_TO_COMPARE,
        "post_reload_steps_compared": POST_RELOAD_STEPS,
        "trace_on_off_stepwise_exact": first_divergence is None,
        "step_action_trace_exact": step_action_trace_exact,
        "rng_state_exact": rng_state_exact,
        "classifier_probability_trace_exact": classifier_probability_trace_exact,
        "step_semantic_fields_present": semantic_fields_present,
        "first_semantic_divergence_step": first_divergence,
        "trace_on_checkpoint_reload_pass": not any(item.startswith("on_checkpoint") for item in failures),
        "trace_off_checkpoint_reload_pass": not any(item.startswith("off_checkpoint") for item in failures),
        "post_reload_trace_mode_equivalence_pass": not any(item.startswith("post_reload") for item in failures),
        "trace_on_observer": str(on_path),
        "trace_off_observer": str(off_path),
        "trace_on_observer_sha256": sha256_file(on_path),
        "trace_off_observer_sha256": sha256_file(off_path),
        "trace_on_observer_log_audit": trace_on_observer_log_audit,
        "trace_off_observer_log_audit": trace_off_observer_log_audit,
        "trace_on_checkpoint": arm_receipts[0]["checkpoint_500"],
        "trace_off_checkpoint": arm_receipts[2]["checkpoint_500"],
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "timestamps_pid_runtime_excluded_from_scientific_digest": True,
        "step500_checkpoint_serialized_candidate_records_exact": (
            step500_candidate_records_exact
        ),
        "trace_on_step500_serialized_candidate_records_sha256": (
            trace_on_checkpoint["serialized_candidate_records_sha256"]
        ),
        "trace_off_step500_serialized_candidate_records_sha256": (
            trace_off_checkpoint["serialized_candidate_records_sha256"]
        ),
        "step500_checkpoint_candidate_universe_exact": (
            step500_candidate_universe_exact
        ),
        "trace_on_step500_candidate_universe_sha256": (
            trace_on_checkpoint["candidate_universe_sha256"]
        ),
        "trace_off_step500_candidate_universe_sha256": (
            trace_off_checkpoint["candidate_universe_sha256"]
        ),
        "checkpoint_algorithm_scientific_state_exact": (
            checkpoint_algorithm_scientific_state_exact
        ),
        "checkpoint_algorithm_full_state_exact": (
            checkpoint_algorithm_full_state_exact
        ),
        "checkpoint_full_state_exact_required": False,
        "checkpoint_rng_state_exact": checkpoint_rng_state_exact,
        "checkpoint_sqlite_logical_state_exact": (
            checkpoint_sqlite_logical_state_exact
        ),
        "checkpoint_graph_registry_exact": checkpoint_graph_registry_exact,
        "resolved_config_scientific_binding_exact": (
            resolved_config_scientific_binding_exact
        ),
        "post_walk_prefix_finalization_performed": False,
        "post_walk_candidate_semantics_bound_by_static_audit": True,
        "full_50k_trace_on_off_parity_claimed": False,
        "arms_overlapped": overlap_audit["arms_overlapped"],
        "max_concurrent_arms": overlap_audit["max_concurrent_arms"],
        "execution_overlap_audit": overlap_audit,
        "execution_order": ["trace_on_continuous", "trace_on_reload", "trace_off_continuous", "trace_off_reload"],
        "calibration_loaded": False,
        "test_loaded": False,
        "input_manifest": str(run_root / "equivalence_input_manifest.json"),
        "input_manifest_sha256": sha256_file(run_root / "equivalence_input_manifest.json"),
        "source_delta_audit": source_delta,
        "trace_on_resolved_config_audit": trace_on_config,
        "trace_off_resolved_config_audit": trace_off_config,
        "trace_on_checkpoint_state_audit": trace_on_checkpoint,
        "trace_off_checkpoint_state_audit": trace_off_checkpoint,
        "checkpoint_gates": checkpoint_gates,
        "arm_receipts": arm_receipts,
        "failures": failures,
    }
    result["summary_sha256"] = stable_json_sha256(result)
    _atomic_json(output / "trace_on_off_500_step_equivalence.json", result)
    _atomic_json(
        output / "trace_on_off_first_divergence.json",
        {"first_semantic_divergence_step": first_divergence, "status": result["status"]},
    )
    with (output / "trace_on_off_artifact_diff.csv").open("w", encoding="utf-8") as handle:
        handle.write("step,comparison,exact\n")
        for row in artifact_diff:
            handle.write(f"{row['step']},{row['comparison']},{str(row['exact']).lower()}\n")
    _atomic_json(
        output / "trace_on_checkpoint_reload.json",
        {"status": "PASS" if result["trace_on_checkpoint_reload_pass"] else "FAIL", "checkpoint": result["trace_on_checkpoint"]},
    )
    _atomic_json(
        output / "trace_off_checkpoint_reload.json",
        {"status": "PASS" if result["trace_off_checkpoint_reload_pass"] else "FAIL", "checkpoint": result["trace_off_checkpoint"]},
    )
    _atomic_json(
        output / "post_reload_trace_mode_equivalence.json",
        {"status": "PASS" if result["post_reload_trace_mode_equivalence_pass"] else "FAIL", "steps": list(range(501, 511))},
    )
    if failures:
        _atomic_json(output / "FAIL.json", result)
        raise RuntimeError(f"Trace-mode semantic equivalence failed: {failures[:8]}")
    (output / "PASS").write_bytes(b"PASS\n")
    print(json.dumps(result, sort_keys=True))
    print("[MUT_TRACE_ON_OFF_500_STEP_EQUIVALENCE_PASS]", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    one = commands.add_parser("run-one")
    one.add_argument("--science-project-root", required=True)
    one.add_argument("--output-root", required=True)
    one.add_argument("--checkpoint-mirror-root", required=True)
    one.add_argument("--observer-output", required=True)
    one.add_argument("--trace-mode", choices=("on", "off"), required=True)
    one.add_argument("--phase", choices=("continuous", "reload"), required=True)
    one.add_argument("--resume", action="store_true")
    pair = commands.add_parser("run-pair")
    pair.add_argument("--python", required=True)
    pair.add_argument("--legacy-project-root", required=True)
    pair.add_argument("--execution-project-root", required=True)
    pair.add_argument("--run-root", required=True)
    pair.add_argument("--output-dir", required=True)
    pair.add_argument("--active-arm-path")
    pair.add_argument("--historical-artifact-root", required=True)
    pair.add_argument("--rf-oracle", required=True)
    for target in (one, pair):
        target.add_argument("--upstream-root", required=True)
        target.add_argument("--dataset-dir", required=True)
        target.add_argument("--gnn-checkpoint", required=True)
        target.add_argument("--distance-checkpoint", required=True)
        target.add_argument("--parent-limit", type=int, default=1448)
        target.add_argument("--device", default="cuda:0")
        target.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_one(args) if args.action == "run-one" else _run_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
