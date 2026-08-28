"""Resumable wrappers around the unmodified GlobalGCE training semantics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pickle
import random
import resource
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION = (
    "globalgce_training_resume_identity_v2"
)
GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1 = "globalgce_epoch_checkpoint_v1"
GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2 = "globalgce_epoch_checkpoint_v2"
_TRAINING_RESUME_IDENTITY_KEYS = {
    "schema_version",
    "dataset",
    "num_classes",
    "source_label",
    "target_label",
    "oracle_identity",
    "native_train_cohort",
    "source_train_cohort",
    "official_source_identity",
    "training_config",
}
_COHORT_IDENTITY_KEYS = {
    "count",
    "ordered_sha256",
    "train_count",
    "train_ordered_sha256",
    "val_count",
    "val_ordered_sha256",
}
_FROZEN_GINE_IDENTITY_KEYS = {
    "schema_version",
    "backend",
    "checkpoint_root",
    "checkpoint_id",
    "dataset",
    "num_classes",
    "source_label",
    "temperature_hex",
    "temperature_scaling_sha256",
    "sha256sums_sha256",
    "inventory",
    "inventory_sha256",
    "identity_sha256",
}
_NATIVE_BINARY_IDENTITY_KEYS = {
    "schema_version",
    "backend",
    "num_classes",
    "native_train_csv",
    "identity_sha256",
}
_FILE_IDENTITY_KEYS = {"path", "bytes", "sha256"}
_INVENTORY_ENTRY_KEYS = {"name", "bytes", "sha256"}
_OFFICIAL_SOURCE_IDENTITY_KEYS = {
    "schema_version",
    "root",
    "files",
    "identity_sha256",
}
_OFFICIAL_SOURCE_FILE_KEYS = {"bytes", "sha256"}
_TRAINING_CONFIG_IDENTITY_KEYS = {
    "seed",
    "epochs",
    "top_k_native",
    "learning_rate_hex",
    "dropout_hex",
    "min_freq",
    "gspan_flush_every",
    "gspan_max_in_memory_candidates",
    "gspan_exact_top_k_pruning",
    "gspan_adoption_identity",
}


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "GlobalGCE training resume identity is not canonical JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_float_hex(value: Any, *, positive: bool = False) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return False
    return (
        math.isfinite(parsed)
        and parsed.hex() == value
        and (not positive or parsed > 0.0)
    )


def _dataset_identity_token(value: Any) -> str:
    if type(value) is not str:
        return ""
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_file_identity(payload: Any) -> None:
    if (
        type(payload) is not dict
        or set(payload) != _FILE_IDENTITY_KEYS
        or type(payload.get("path")) is not str
        or not payload["path"]
        or type(payload.get("bytes")) is not int
        or payload["bytes"] < 0
        or not _is_sha256(payload.get("sha256"))
    ):
        raise ValueError("GlobalGCE native checkpoint file identity is invalid.")


def _validate_oracle_identity(
    payload: Any,
    *,
    dataset: str,
    num_classes: int,
    source_label: int,
) -> None:
    if type(payload) is not dict:
        raise ValueError("GlobalGCE training resume oracle identity is invalid.")
    backend = payload.get("backend")
    if backend == "frozen_gine":
        if (
            set(payload) != _FROZEN_GINE_IDENTITY_KEYS
            or payload.get("schema_version")
            != "globalgce_frozen_gine_resume_identity_v1"
            or type(payload.get("checkpoint_root")) is not str
            or not payload["checkpoint_root"]
            or not _is_sha256(payload.get("checkpoint_id"))
            or _dataset_identity_token(payload.get("dataset"))
            != _dataset_identity_token(dataset)
            or type(payload.get("num_classes")) is not int
            or payload["num_classes"] != num_classes
            or type(payload.get("source_label")) is not int
            or payload["source_label"] != source_label
            or not _canonical_float_hex(payload.get("temperature_hex"), positive=True)
            or not _is_sha256(payload.get("temperature_scaling_sha256"))
            or not _is_sha256(payload.get("sha256sums_sha256"))
            or not _is_sha256(payload.get("inventory_sha256"))
            or not _is_sha256(payload.get("identity_sha256"))
        ):
            raise ValueError("GlobalGCE frozen-GINE resume identity is invalid.")
        inventory = payload.get("inventory")
        if type(inventory) is not list or not inventory:
            raise ValueError("GlobalGCE frozen-GINE inventory is empty.")
        names: set[str] = set()
        inventory_by_name: dict[str, dict[str, Any]] = {}
        for entry in inventory:
            if (
                type(entry) is not dict
                or set(entry) != _INVENTORY_ENTRY_KEYS
                or type(entry.get("name")) is not str
                or not entry["name"]
                or Path(entry["name"]).name != entry["name"]
                or entry["name"] in names
                or type(entry.get("bytes")) is not int
                or entry["bytes"] < 0
                or not _is_sha256(entry.get("sha256"))
            ):
                raise ValueError("GlobalGCE frozen-GINE inventory is invalid.")
            names.add(entry["name"])
            inventory_by_name[entry["name"]] = entry
        model_entry = inventory_by_name.get("model.pt")
        temperature_entry = inventory_by_name.get("temperature_scaling.json")
        if (
            model_entry is None
            or model_entry["sha256"] != payload["checkpoint_id"]
            or temperature_entry is None
            or temperature_entry["sha256"]
            != payload["temperature_scaling_sha256"]
            or payload["inventory_sha256"]
            != _canonical_json_sha256({"files": inventory})
            or payload["identity_sha256"]
            != _canonical_json_sha256(
                {key: value for key, value in payload.items() if key != "identity_sha256"}
            )
        ):
            raise ValueError("GlobalGCE frozen-GINE identity hashes are inconsistent.")
        return
    if backend == "official_native_gtgnn":
        if (
            set(payload) != _NATIVE_BINARY_IDENTITY_KEYS
            or payload.get("schema_version")
            != "globalgce_native_binary_gtgnn_resume_identity_v1"
            or num_classes != 2
            or type(payload.get("num_classes")) is not int
            or payload.get("num_classes") != 2
            or not _is_sha256(payload.get("identity_sha256"))
        ):
            raise ValueError("GlobalGCE native binary resume identity is invalid.")
        _validate_file_identity(payload.get("native_train_csv"))
        if payload["identity_sha256"] != _canonical_json_sha256(
            {key: value for key, value in payload.items() if key != "identity_sha256"}
        ):
            raise ValueError("GlobalGCE native binary identity hash is inconsistent.")
        return
    raise ValueError("GlobalGCE training resume oracle backend is invalid.")


def _validate_official_source_identity(payload: Any) -> None:
    if (
        type(payload) is not dict
        or set(payload) != _OFFICIAL_SOURCE_IDENTITY_KEYS
        or payload.get("schema_version")
        != "globalgce_official_source_resume_identity_v1"
        or type(payload.get("root")) is not str
        or not payload["root"]
        or type(payload.get("files")) is not dict
        or not payload["files"]
        or not _is_sha256(payload.get("identity_sha256"))
    ):
        raise ValueError("GlobalGCE official source resume identity is invalid.")
    for name, file_identity in payload["files"].items():
        if (
            type(name) is not str
            or not name
            or type(file_identity) is not dict
            or set(file_identity) != _OFFICIAL_SOURCE_FILE_KEYS
            or type(file_identity.get("bytes")) is not int
            or file_identity["bytes"] < 0
            or not _is_sha256(file_identity.get("sha256"))
        ):
            raise ValueError("GlobalGCE official source file identity is invalid.")
    if payload["identity_sha256"] != _canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    ):
        raise ValueError("GlobalGCE official source identity hash is inconsistent.")


def _validate_training_config_identity(payload: Any) -> None:
    if (
        type(payload) is not dict
        or set(payload) != _TRAINING_CONFIG_IDENTITY_KEYS
        or type(payload.get("seed")) is not int
        or payload["seed"] < 0
        or type(payload.get("epochs")) is not int
        or payload["epochs"] < 0
        or type(payload.get("top_k_native")) is not int
        or payload["top_k_native"] <= 0
        or not _canonical_float_hex(payload.get("learning_rate_hex"), positive=True)
        or not _canonical_float_hex(payload.get("dropout_hex"))
        or (
            payload.get("min_freq") is not None
            and (
                type(payload["min_freq"]) is not int
                or payload["min_freq"] < 2
            )
        )
        or type(payload.get("gspan_flush_every")) is not int
        or payload["gspan_flush_every"] <= 0
        or type(payload.get("gspan_max_in_memory_candidates")) is not int
        or payload["gspan_max_in_memory_candidates"] <= 0
        or type(payload.get("gspan_exact_top_k_pruning")) is not bool
        or (
            payload.get("gspan_adoption_identity") is not None
            and (
                type(payload["gspan_adoption_identity"]) is not dict
                or not payload["gspan_adoption_identity"]
            )
        )
    ):
        raise ValueError("GlobalGCE training resume configuration identity is invalid.")


def normalize_globalgce_training_resume_identity(
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return one typed, canonical identity for an epoch/terminal checkpoint.

    The legacy unversioned checkpoint surface is retained only when callers do
    not provide an identity.  A v2 caller must provide the complete contract;
    it can never reinterpret a v1 checkpoint as a v2 Taste or BACE run.
    """

    if type(identity) is not dict or set(identity) != _TRAINING_RESUME_IDENTITY_KEYS:
        raise ValueError("GlobalGCE training resume identity has an invalid schema.")
    if identity.get("schema_version") != GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION:
        raise ValueError("GlobalGCE training resume identity version changed.")
    dataset = identity.get("dataset")
    num_classes = identity.get("num_classes")
    source_label = identity.get("source_label")
    target_label = identity.get("target_label")
    if (
        type(dataset) is not str
        or not dataset.strip()
        or not _dataset_identity_token(dataset)
    ):
        raise ValueError("GlobalGCE training resume dataset is invalid.")
    if (
        type(num_classes) is not int
        or num_classes < 2
        or type(source_label) is not int
        or type(target_label) is not int
        or not 0 <= source_label < num_classes
        or not 0 <= target_label < num_classes
        or source_label == target_label
    ):
        raise ValueError("GlobalGCE training resume class contract is invalid.")
    for key in ("native_train_cohort", "source_train_cohort"):
        cohort = identity.get(key)
        if type(cohort) is not dict or set(cohort) != _COHORT_IDENTITY_KEYS:
            raise ValueError(f"GlobalGCE training resume {key} schema is invalid.")
        for count_key in ("count", "train_count", "val_count"):
            if type(cohort.get(count_key)) is not int or cohort[count_key] <= 0:
                raise ValueError(
                    f"GlobalGCE training resume {key}.{count_key} is invalid."
                )
        if cohort["train_count"] + cohort["val_count"] != cohort["count"]:
            raise ValueError(f"GlobalGCE training resume {key} is not a partition.")
        for hash_key in (
            "ordered_sha256",
            "train_ordered_sha256",
            "val_ordered_sha256",
        ):
            if not _is_sha256(cohort.get(hash_key)):
                raise ValueError(
                    f"GlobalGCE training resume {key}.{hash_key} is invalid."
                )
    _validate_oracle_identity(
        identity.get("oracle_identity"),
        dataset=dataset,
        num_classes=num_classes,
        source_label=source_label,
    )
    _validate_official_source_identity(identity.get("official_source_identity"))
    _validate_training_config_identity(identity.get("training_config"))
    try:
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("GlobalGCE training resume identity is not canonical JSON.") from exc
    normalized = json.loads(encoded.decode("utf-8"))
    if type(normalized) is not dict or normalized != identity:
        raise ValueError("GlobalGCE training resume identity changed during canonicalization.")
    return normalized, hashlib.sha256(encoded).hexdigest()


def validate_globalgce_epoch_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a partial checkpoint from another dataset, cohort, or target."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("GlobalGCE training checkpoint must be a mapping.")
    expected, expected_sha256 = normalize_globalgce_training_resume_identity(
        expected_identity
    )
    if checkpoint.get("checkpoint_schema_version") != GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2:
        raise ValueError("GlobalGCE training checkpoint is not identity-bound v2.")
    actual = checkpoint.get("resume_identity")
    if type(actual) is not dict:
        raise ValueError("GlobalGCE training checkpoint has no resume identity.")
    normalized, actual_sha256 = normalize_globalgce_training_resume_identity(actual)
    if (
        normalized != expected
        or actual_sha256 != expected_sha256
        or checkpoint.get("resume_identity_sha256") != expected_sha256
    ):
        raise ValueError("GlobalGCE training checkpoint resume identity mismatch.")
    return {
        "schema_version": GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2,
        "resume_identity": expected,
        "resume_identity_sha256": expected_sha256,
    }


def _serialize_numpy_rng_state(numpy_module: Any) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = (
        numpy_module.random.get_state()
    )
    return {
        "schema_version": "globalgce_numpy_rng_state_v1",
        "algorithm": str(algorithm),
        "keys": [int(value) for value in keys.tolist()],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(numpy_module: Any, payload: Mapping[str, Any]) -> None:
    if (
        type(payload) is not dict
        or set(payload)
        != {
            "schema_version",
            "algorithm",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        or payload.get("schema_version") != "globalgce_numpy_rng_state_v1"
        or type(payload.get("algorithm")) is not str
        or type(payload.get("keys")) is not list
        or not payload["keys"]
        or any(type(value) is not int or not 0 <= value < 2**32 for value in payload["keys"])
        or type(payload.get("position")) is not int
        or type(payload.get("has_gauss")) is not int
        or payload["has_gauss"] not in (0, 1)
        or type(payload.get("cached_gaussian")) is not float
    ):
        raise ValueError("GlobalGCE NumPy RNG checkpoint state is invalid.")
    numpy_module.random.set_state(
        (
            payload["algorithm"],
            numpy_module.asarray(payload["keys"], dtype=numpy_module.uint32),
            payload["position"],
            payload["has_gauss"],
            payload["cached_gaussian"],
        )
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(),
    )


def _atomic_pickle(path: Path, payload: Any) -> None:
    _atomic_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _typed_fingerprint_value(value: Any) -> Any:
    """Encode values without collapsing types that affect native traversal."""

    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "value": [_typed_fingerprint_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "value": [_typed_fingerprint_value(item) for item in value],
        }
    if isinstance(value, Mapping):
        encoded_items = [
            (_typed_fingerprint_value(key), _typed_fingerprint_value(item))
            for key, item in value.items()
        ]
        encoded_items.sort(
            key=lambda row: json.dumps(
                row[0], sort_keys=True, separators=(",", ":")
            )
        )
        return {"type": "mapping", "value": encoded_items}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _graph_input_fingerprint(graphs: list[Any], settings: dict[str, Any]) -> str:
    """Bind every ordering axis consumed by pinned NetworkX/gSpan traversal."""

    rows = []
    for graph_index, graph in enumerate(graphs):
        nodes = [
            {
                "node_insertion_index": node_index,
                "node": _typed_fingerprint_value(node),
                "label": _typed_fingerprint_value(attributes.get("label")),
            }
            for node_index, (node, attributes) in enumerate(graph.nodes(data=True))
        ]
        edges = [
            {
                "edge_traversal_index": edge_index,
                "left": _typed_fingerprint_value(left),
                "right": _typed_fingerprint_value(right),
                "label": _typed_fingerprint_value(attributes.get("label")),
            }
            for edge_index, (left, right, attributes) in enumerate(
                graph.edges(data=True)
            )
        ]
        rows.append(
            {
                "graph_list_index": graph_index,
                "nodes_in_insertion_order": nodes,
                "edges_in_native_traversal_order": edges,
            }
        )
    encoded = json.dumps(
        {
            "fingerprint_schema": "globalgce_native_traversal_order_v2",
            "settings": _typed_fingerprint_value(settings),
            "graphs_in_input_list_order": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _selected_rows_from_database(
    connection: sqlite3.Connection, *, top_k: int
) -> list[dict[str, Any]]:
    selected = connection.execute(
        "SELECT support, root_index, local_index, payload FROM patterns "
        "ORDER BY support DESC, root_index ASC, local_index ASC LIMIT ?",
        (int(top_k),),
    ).fetchall()
    return [
        {
            "rank": rank,
            "support": int(row[0]),
            "root_index": int(row[1]),
            "local_index": int(row[2]),
            "payload_sha256": hashlib.sha256(bytes(row[3])).hexdigest(),
        }
        for rank, row in enumerate(selected, start=1)
    ]


def validate_exact_top_k_audit(audit_path: str | Path) -> dict[str, Any]:
    """Recompute the terminal exact-top-k proof from checkpoint and SQLite."""

    audit_file = Path(audit_path).expanduser().resolve(strict=True)
    if audit_file.name != "exact_top_k_audit.json":
        raise ValueError("GlobalGCE exact-top-k proof path has an invalid filename.")
    checkpoint_path = audit_file.parent / "checkpoint.json"
    database_path = audit_file.parent / "frequent_patterns.sqlite3"
    if not checkpoint_path.is_file() or not database_path.is_file():
        raise FileNotFoundError(
            "GlobalGCE exact-top-k proof requires checkpoint.json and SQLite."
        )
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or not isinstance(checkpoint, dict):
        raise ValueError("GlobalGCE exact-top-k proof files must be JSON objects.")
    if (
        audit.get("schema_version")
        != "globalgce_exact_stable_topk_audit_v2"
        or audit.get("status") != "PASS"
        or audit.get("run_complete") is not True
    ):
        raise ValueError("GlobalGCE exact-top-k audit is not a v2 terminal PASS.")
    if (
        checkpoint.get("schema_version")
        != "globalgce_gspan_exact_stable_topk_v2"
        or checkpoint.get("stage") != "complete"
        or checkpoint.get("exact_top_k_pruning") is not True
    ):
        raise ValueError("GlobalGCE exact-top-k checkpoint is not complete.")
    checkpoint_identity = _file_identity(checkpoint_path)
    if audit.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("GlobalGCE exact-top-k checkpoint hash binding failed.")
    if audit.get("input_fingerprint") != checkpoint.get("input_fingerprint"):
        raise ValueError("GlobalGCE exact-top-k fingerprint closure failed.")
    if Path(str(checkpoint.get("sqlite_path") or "")).resolve() != database_path:
        raise ValueError("GlobalGCE exact-top-k SQLite path closure failed.")
    top_k = int(audit.get("top_k") or 0)
    if top_k <= 0:
        raise ValueError("GlobalGCE exact-top-k audit has an invalid top_k.")
    with sqlite3.connect(database_path, timeout=120) as connection:
        stored_fingerprint = connection.execute(
            "SELECT value FROM metadata WHERE key='input_fingerprint'"
        ).fetchone()
        active_root = connection.execute(
            "SELECT value FROM metadata WHERE key='active_root_index'"
        ).fetchone()
        root_count, complete_count = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(complete), 0) FROM roots"
        ).fetchone()
        retained_count = int(
            connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        )
        reported_count, pruned_count = connection.execute(
            "SELECT COALESCE(SUM(reported_pattern_count), 0), "
            "COALESCE(SUM(pruned_branch_count), 0) FROM root_stats"
        ).fetchone()
        selected_rows = _selected_rows_from_database(connection, top_k=top_k)
    if stored_fingerprint is None or str(stored_fingerprint[0]) != str(
        checkpoint["input_fingerprint"]
    ):
        raise ValueError("GlobalGCE exact-top-k SQLite fingerprint mismatch.")
    if active_root is not None:
        raise ValueError("GlobalGCE exact-top-k proof retains an active root.")
    if (
        int(root_count) != int(checkpoint.get("root_count", -1))
        or int(complete_count) != int(root_count)
        or int(checkpoint.get("completed_root_count", -1)) != int(root_count)
    ):
        raise ValueError("GlobalGCE exact-top-k root completion closure failed.")
    selected_identity = hashlib.sha256(
        json.dumps(selected_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if (
        audit.get("selected_rows") != selected_rows
        or int(audit.get("selected_count", -1)) != len(selected_rows)
        or audit.get("selected_identity_sha256") != selected_identity
        or int(audit.get("retained_pattern_count", -1)) != retained_count
        or int(audit.get("reported_pattern_count", -1)) != int(reported_count)
        or int(audit.get("pruned_branch_count", -1)) != int(pruned_count)
    ):
        raise ValueError("GlobalGCE exact-top-k selected-payload closure failed.")
    return {
        "schema_version": "globalgce_exact_top_k_proof_identity_v1",
        "status": "PASS",
        "input_fingerprint": str(checkpoint["input_fingerprint"]),
        "top_k": top_k,
        "selected_identity_sha256": selected_identity,
        "audit": _file_identity(audit_file),
        "checkpoint": checkpoint_identity,
        "sqlite_path": str(database_path),
    }


def validate_exact_top_k_proof_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a persisted proof identity still matches its bytes."""

    if not isinstance(identity, Mapping):
        raise ValueError("GlobalGCE exact-top-k proof identity is missing.")
    audit = identity.get("audit")
    if not isinstance(audit, Mapping) or not audit.get("path"):
        raise ValueError("GlobalGCE exact-top-k proof has no audit path.")
    observed = validate_exact_top_k_audit(str(audit["path"]))
    if dict(identity) != observed:
        raise ValueError("GlobalGCE exact-top-k proof identity/hash mismatch.")
    return observed


class _Heartbeat:
    def __init__(self, path: Path, state: dict[str, Any]) -> None:
        self.path = path
        self.state = state
        self.interval = max(float(os.environ.get("GLOBALGCE_HEARTBEAT_SECONDS", "60")), 1.0)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        _atomic_json(
            self.path,
            {**self.state, "heartbeat_epoch_seconds": time.time()},
        )

    def _run(self) -> None:
        self._write()
        while not self.stop_event.wait(self.interval):
            self._write()

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 1.0)
        self._write()


@contextmanager
def resumable_gspan_root_chunks(
    gspan_module: Any,
    *,
    checkpoint_root: str | Path,
    top_k: int | None = None,
    flush_every: int | None = None,
    max_in_memory_candidates: int | None = None,
    exact_top_k_pruning: bool = False,
) -> Iterator[dict[str, Any]]:
    """Spill deterministic gSpan root reports and retain only official top-k.

    The official implementation builds every frequent graph in memory and then
    performs a stable support-descending sort before slicing ``topk``.  The
    default route stores the same traversal order in SQLite and applies the
    equivalent SQL ordering.  The opt-in exact-top-k route retains only the
    stable top-k and prunes a DFS branch only after its support upper bound can
    no longer enter that top-k.  Both routes restart an interrupted gSpan root
    from its beginning; neither adopts a partial traversal.
    """

    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    gspan_class = gspan_module.gSpan
    original_run = gspan_class.run
    original_report = gspan_class._report
    original_subgraph_mining = gspan_class._subgraph_mining
    globals_ = original_run.__globals__
    projected_class = globals_["Projected"]
    pdfs_class = globals_["PDFS"]
    dfsedge_class = globals_["DFSedge"]
    configured_flush = int(
        flush_every
        if flush_every is not None
        else os.environ.get("GLOBALGCE_GSPAN_FLUSH_EVERY", "256")
    )
    configured_max = int(
        max_in_memory_candidates
        if max_in_memory_candidates is not None
        else os.environ.get("GLOBALGCE_GSPAN_MAX_IN_MEMORY_CANDIDATES", "256")
    )
    if configured_flush <= 0 or configured_max <= 0:
        raise ValueError("GlobalGCE spill limits must be positive.")
    if exact_top_k_pruning and (top_k is None or int(top_k) <= 0):
        raise ValueError("Exact GlobalGCE top-k pruning requires a positive top_k.")
    commit_every = min(configured_flush, configured_max)
    min_free_bytes = int(
        float(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_GIB", "50")) * 1024**3
    )
    min_free_ratio = float(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_RATIO", "0.02"))
    min_free_inodes = int(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_INODES", "100000"))
    active: dict[int, dict[str, Any]] = {}
    session: dict[str, Any] = {
        "exact_top_k_pruning": bool(exact_top_k_pruning),
        "completed_exact_top_k_proofs": [],
    }

    def _stable_key(row: tuple[int, int, int]) -> tuple[int, int, int]:
        support, root_index, local_index = row
        return (-int(support), int(root_index), int(local_index))

    def _load_retained_top_k(connection: sqlite3.Connection) -> list[tuple[int, int, int]]:
        rows = connection.execute(
            "SELECT support, root_index, local_index FROM patterns "
            "ORDER BY support DESC, root_index ASC, local_index ASC"
        ).fetchall()
        return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]

    def peak_rss_mib() -> float:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.  HPC uses the Linux branch.
        return value / (1024.0 if value < 1024**3 else 1024.0**2)

    def storage_snapshot(path: Path) -> dict[str, Any]:
        usage = shutil.disk_usage(path)
        filesystem = os.statvfs(path)
        free_inodes = int(filesystem.f_favail)
        free_ratio = float(usage.free / usage.total) if usage.total else 0.0
        return {
            "free_bytes": int(usage.free),
            "free_ratio": free_ratio,
            "free_inodes": free_inodes,
            "storage_guard_pass": (
                int(usage.free) >= min_free_bytes
                and free_ratio >= min_free_ratio
                and free_inodes >= min_free_inodes
            ),
        }

    def optimized_report(self: Any, projected: Any) -> None:
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        context = active.get(id(self))
        if context is None:
            raise RuntimeError("GlobalGCE spill report invoked outside an active root.")
        if context.pop("suppress_next_report", False):
            return
        local_index = int(context["local_index"])
        report_gid = next(self._counter)
        row = (int(self._support), int(context["root_index"]), local_index)
        retain = True
        retained = context.get("retained_top_k")
        if retained is not None:
            limit = int(context["top_k"])
            retain = len(retained) < limit or _stable_key(row) < _stable_key(retained[-1])
        if retain:
            graph = self._DFScode.to_graph(
                gid=report_gid, is_undirected=self._is_undirected
            )
            graph = self._from_Graph_to_nx_Graph(graph)
            context["connection"].execute(
                "INSERT INTO patterns(root_index, local_index, support, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    int(context["root_index"]),
                    local_index,
                    int(self._support),
                    sqlite3.Binary(
                        pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
                    ),
                ),
            )
            if retained is not None:
                retained.append(row)
                retained.sort(key=_stable_key)
                if len(retained) > int(context["top_k"]):
                    discarded = retained.pop()
                    context["connection"].execute(
                        "DELETE FROM patterns WHERE root_index=? AND local_index=?",
                        (int(discarded[1]), int(discarded[2])),
                    )
        context["local_index"] = local_index + 1
        context["uncommitted"] = int(context["uncommitted"]) + 1
        context["state"]["frequent_subgraph_count"] = int(
            context["state"].get("frequent_subgraph_count") or 0
        ) + 1
        if int(context["uncommitted"]) >= commit_every:
            context["connection"].commit()
            context["uncommitted"] = 0
            context["state"]["peak_rss_mib"] = peak_rss_mib()
            context["state"].update(storage_snapshot(context["support_root"]))
            if context["state"]["storage_guard_pass"] is not True:
                context["state"]["stage"] = "storage_guard_stop"
                _atomic_json(context["checkpoint_path"], context["state"])
                raise RuntimeError(
                    "GLOBALGCE_STORAGE_GUARD_STOP: scratch free-space or inode "
                    "reserve was reached after a committed SQLite checkpoint."
                )

    def exact_subgraph_mining(self: Any, projected: Any) -> Any:
        """Apply only the anti-monotone pruning that preserves stable top-k."""

        context = active.get(id(self))
        if context is None or context.get("retained_top_k") is None:
            return original_subgraph_mining(self, projected)
        self._support = self._get_support(projected)
        if self._support < self._min_support:
            return None
        if not self._is_min():
            return None
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return original_subgraph_mining(self, projected)
        optimized_report(self, projected)
        retained = context["retained_top_k"]
        if len(retained) == int(context["top_k"]):
            cutoff_support = int(retained[-1][0])
            # gSpan projected support is anti-monotone under every extension.
            # Equal-support descendants are later in the stable traversal and
            # therefore cannot displace the already retained kth row either.
            if cutoff_support >= int(self._support):
                context["pruned_branch_count"] = int(
                    context.get("pruned_branch_count") or 0
                ) + 1
                return self
        context["suppress_next_report"] = True
        try:
            return original_subgraph_mining(self, projected)
        finally:
            context.pop("suppress_next_report", None)

    def resumable_run(self: Any) -> tuple[list[Any], list[int]]:
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        if self._max_num_vertices < 2:
            return self.fs_collection, self.freq_collection
        top_roots: dict[Any, Any] = defaultdict(projected_class)
        for graph_id, graph in self.graphs.items():
            for vertex_id, vertex in graph.vertices.items():
                for edge in self._get_forward_root_edges(graph, vertex_id):
                    top_roots[
                        (vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)
                    ].append(pdfs_class(graph_id, edge, None))
        settings = {
            "min_support": self._min_support,
            "min_vertices": self._min_num_vertices,
            "max_vertices": self._max_num_vertices,
            "is_undirected": self._is_undirected,
            "root_order": [repr(value) for value in top_roots],
            "top_k": int(top_k) if top_k is not None else None,
            "spill_schema": (
                "sqlite_exact_stable_topk_antimonotone_v2"
                if exact_top_k_pruning
                else "sqlite_stable_support_topk_v2"
            ),
        }
        fingerprint = _graph_input_fingerprint(self._nx_graph_list, settings)
        support_root = root / f"support_{int(self._min_support)}_{fingerprint[:16]}"
        support_root.mkdir(parents=True, exist_ok=True)
        database_path = support_root / "frequent_patterns.sqlite3"
        connection = sqlite3.connect(database_path, timeout=120)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS roots("
            "root_index INTEGER PRIMARY KEY, root_label TEXT NOT NULL, "
            "complete INTEGER NOT NULL, pattern_count INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS patterns("
            "root_index INTEGER NOT NULL, local_index INTEGER NOT NULL, "
            "support INTEGER NOT NULL, payload BLOB NOT NULL, "
            "PRIMARY KEY(root_index, local_index))"
        )
        if exact_top_k_pruning:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS root_snapshot("
                "snapshot_for_root INTEGER NOT NULL, root_index INTEGER NOT NULL, "
                "local_index INTEGER NOT NULL, support INTEGER NOT NULL, "
                "payload BLOB NOT NULL, "
                "PRIMARY KEY(snapshot_for_root, root_index, local_index))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS root_stats("
                "root_index INTEGER PRIMARY KEY, reported_pattern_count INTEGER NOT NULL, "
                "pruned_branch_count INTEGER NOT NULL)"
            )
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key='input_fingerprint'"
        ).fetchone()
        if existing is not None and str(existing[0]) != fingerprint:
            connection.close()
            raise ValueError("GlobalGCE SQLite spill fingerprint mismatch.")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('input_fingerprint', ?)",
            (fingerprint,),
        )
        connection.commit()
        state = {
            "schema_version": (
                "globalgce_gspan_exact_stable_topk_v2"
                if exact_top_k_pruning
                else "globalgce_gspan_sqlite_chunks_v2"
            ),
            "stage": "mining",
            "input_fingerprint": fingerprint,
            "root_count": len(top_roots),
            "completed_root_count": 0,
            "current_root_index": None,
            "frequent_subgraph_count": 0,
            "sqlite_path": str(database_path),
            "flush_every": configured_flush,
            "max_in_memory_candidates": configured_max,
            "exact_top_k_pruning": bool(exact_top_k_pruning),
            "peak_rss_mib": peak_rss_mib(),
            **storage_snapshot(support_root),
        }
        try:
            with _Heartbeat(support_root / "heartbeat.json", state):
                for root_index, (vevlb, projected) in enumerate(top_roots.items()):
                    state["current_root_index"] = root_index
                    completed = connection.execute(
                        "SELECT complete, pattern_count FROM roots WHERE root_index=?",
                        (root_index,),
                    ).fetchone()
                    if completed is None or int(completed[0]) != 1:
                        if exact_top_k_pruning:
                            active_root = connection.execute(
                                "SELECT value FROM metadata WHERE key='active_root_index'"
                            ).fetchone()
                            if completed is not None:
                                if active_root is None or int(active_root[0]) != root_index:
                                    raise RuntimeError(
                                        "GlobalGCE exact top-k resume snapshot identity mismatch."
                                    )
                                connection.execute("DELETE FROM patterns")
                                connection.execute(
                                    "INSERT INTO patterns(root_index, local_index, support, payload) "
                                    "SELECT root_index, local_index, support, payload "
                                    "FROM root_snapshot WHERE snapshot_for_root=?",
                                    (root_index,),
                                )
                            else:
                                if active_root is not None:
                                    raise RuntimeError(
                                        "GlobalGCE exact top-k database has a stale active root."
                                    )
                                connection.execute("DELETE FROM root_snapshot")
                                connection.execute(
                                    "INSERT INTO root_snapshot("
                                    "snapshot_for_root, root_index, local_index, support, payload) "
                                    "SELECT ?, root_index, local_index, support, payload FROM patterns",
                                    (root_index,),
                                )
                                connection.execute(
                                    "INSERT INTO metadata(key, value) "
                                    "VALUES('active_root_index', ?) ",
                                    (str(root_index),),
                                )
                        else:
                            connection.execute(
                                "DELETE FROM patterns WHERE root_index=?", (root_index,)
                            )
                        connection.execute(
                            "INSERT OR REPLACE INTO roots(root_index, root_label, complete, pattern_count) "
                            "VALUES (?, ?, 0, 0)",
                            (root_index, repr(vevlb)),
                        )
                        connection.commit()
                        context = {
                            "connection": connection,
                            "root_index": root_index,
                            "local_index": 0,
                            "uncommitted": 0,
                            "state": state,
                            "support_root": support_root,
                            "checkpoint_path": support_root / "checkpoint.json",
                            "retained_top_k": (
                                _load_retained_top_k(connection)
                                if exact_top_k_pruning
                                else None
                            ),
                            "top_k": int(top_k) if exact_top_k_pruning else None,
                            "pruned_branch_count": 0,
                        }
                        active[id(self)] = context
                        self.fs_collection = []
                        self.freq_collection = []
                        self._frequent_subgraphs = []
                        self._DFScode.append(dfsedge_class(0, 1, vevlb))
                        try:
                            self._subgraph_mining(projected)
                        finally:
                            self._DFScode.pop()
                            active.pop(id(self), None)
                        connection.commit()
                        pattern_count = int(context["local_index"])
                        if exact_top_k_pruning:
                            connection.execute(
                                "INSERT OR REPLACE INTO root_stats("
                                "root_index, reported_pattern_count, pruned_branch_count) "
                                "VALUES (?, ?, ?)",
                                (
                                    root_index,
                                    pattern_count,
                                    int(context["pruned_branch_count"]),
                                ),
                            )
                            connection.execute(
                                "UPDATE roots SET complete=1, pattern_count=? "
                                "WHERE root_index=?",
                                (pattern_count, root_index),
                            )
                            connection.execute(
                                "DELETE FROM root_snapshot WHERE snapshot_for_root=?",
                                (root_index,),
                            )
                            connection.execute(
                                "DELETE FROM metadata WHERE key='active_root_index'"
                            )
                        else:
                            connection.execute(
                                "UPDATE roots SET complete=1, pattern_count=? "
                                "WHERE root_index=?",
                                (pattern_count, root_index),
                            )
                        connection.commit()
                    state.update(
                        {
                            "completed_root_count": int(
                                connection.execute(
                                    "SELECT COUNT(*) FROM roots WHERE complete=1"
                                ).fetchone()[0]
                            ),
                            "frequent_subgraph_count": int(
                                connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
                            ),
                            "peak_rss_mib": peak_rss_mib(),
                            "sqlite_bytes": database_path.stat().st_size,
                        }
                    )
                    if exact_top_k_pruning:
                        state.update(
                            {
                                "reported_pattern_count": int(
                                    connection.execute(
                                        "SELECT COALESCE(SUM(reported_pattern_count), 0) "
                                        "FROM root_stats"
                                    ).fetchone()[0]
                                ),
                                "pruned_branch_count": int(
                                    connection.execute(
                                        "SELECT COALESCE(SUM(pruned_branch_count), 0) "
                                        "FROM root_stats"
                                    ).fetchone()[0]
                                ),
                                "retained_pattern_count": int(
                                    connection.execute(
                                        "SELECT COUNT(*) FROM patterns"
                                    ).fetchone()[0]
                                ),
                            }
                        )
                    _atomic_json(support_root / "checkpoint.json", state)
            limit = int(top_k) if top_k is not None else -1
            query = (
                "SELECT support, root_index, local_index, payload FROM patterns "
                "ORDER BY support DESC, root_index ASC, local_index ASC"
            )
            parameters: tuple[int, ...] = ()
            if limit >= 0:
                query += " LIMIT ?"
                parameters = (limit,)
            selected = connection.execute(query, parameters).fetchall()
            self.freq_collection = [int(row[0]) for row in selected]
            self.fs_collection = [pickle.loads(row[3]) for row in selected]
            self._frequent_subgraphs = []
            if exact_top_k_pruning:
                selected_rows = _selected_rows_from_database(
                    connection, top_k=int(top_k)
                )
                state.update(
                    {
                        "stage": "complete",
                        "current_root_index": None,
                        "selected_top_k_count": len(selected),
                        "peak_rss_mib": peak_rss_mib(),
                    }
                )
                # The complete checkpoint closes first.  The exact audit is the
                # terminal PASS publication and therefore can never claim a
                # run whose latest checkpoint still says ``mining``.
                checkpoint_path = support_root / "checkpoint.json"
                _atomic_json(checkpoint_path, state)
                audit_payload = {
                    "schema_version": "globalgce_exact_stable_topk_audit_v2",
                    "status": "PASS",
                    "run_complete": True,
                    "input_fingerprint": fingerprint,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "top_k": int(top_k),
                    "selected_count": len(selected_rows),
                    "selected_rows": selected_rows,
                    "selected_identity_sha256": hashlib.sha256(
                        json.dumps(
                            selected_rows,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "ordering": "support_desc_root_index_asc_local_index_asc",
                    "pruning_proof": (
                        "projected_support_is_antimonotone_and_equal_support_"
                        "descendants_are_later_in_stable_dfs_order"
                    ),
                    "reported_pattern_count": int(
                        connection.execute(
                            "SELECT COALESCE(SUM(reported_pattern_count), 0) "
                            "FROM root_stats"
                        ).fetchone()[0]
                    ),
                    "pruned_branch_count": int(
                        connection.execute(
                            "SELECT COALESCE(SUM(pruned_branch_count), 0) "
                            "FROM root_stats"
                        ).fetchone()[0]
                    ),
                    "retained_pattern_count": int(
                        connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
                    ),
                }
                audit_path = support_root / "exact_top_k_audit.json"
                _atomic_json(audit_path, audit_payload)
                session["completed_exact_top_k_proofs"].append(
                    validate_exact_top_k_audit(audit_path)
                )
            else:
                state.update(
                    {
                        "stage": "complete",
                        "current_root_index": None,
                        "selected_top_k_count": len(selected),
                        "peak_rss_mib": peak_rss_mib(),
                    }
                )
                _atomic_json(support_root / "checkpoint.json", state)
            return self.fs_collection, self.freq_collection
        finally:
            active.pop(id(self), None)
            connection.close()

    gspan_class._report = optimized_report
    gspan_class.run = resumable_run
    if exact_top_k_pruning:
        gspan_class._subgraph_mining = exact_subgraph_mining
    try:
        yield session
    finally:
        gspan_class.run = original_run
        gspan_class._report = original_report
        gspan_class._subgraph_mining = original_subgraph_mining


def _atomic_torch_save(torch_module: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch_module.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _get_fs_expanded_data_from_adoption(
    *,
    model: Any,
    train_loader: Any,
    proof_path: str | Path,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any]]:
    """Run official expansion with an independently adopted exhaustive top-k.

    Only one invocation of ``find_fs`` is replaced.  Pinned official
    ``get_fs``/``run_fsg`` still fixes graph sizes, expands the train-only
    dataset, tensorizes the graphs, and creates the native decoders.
    """

    from src.baselines.globalgce_mining_adoption import (
        load_adopted_globalgce_top_k,
        validate_globalgce_gspan_adoption_proof,
    )

    identity = validate_globalgce_gspan_adoption_proof(proof_path)
    graphs, supports = load_adopted_globalgce_top_k(
        proof_path,
        validated_identity=identity,
    )
    expected_top_k = int(model.fsg.topk)
    if len(graphs) != expected_top_k or len(supports) != expected_top_k:
        raise ValueError(
            "GlobalGCE adopted mining payload does not contain configured top-k"
        )
    normalized_supports = [int(value) for value in supports]
    if normalized_supports != sorted(normalized_supports, reverse=True):
        raise ValueError("GlobalGCE adopted top-k is not in stable support order")
    if int(identity.get("top_k") or -1) != expected_top_k:
        raise ValueError("GlobalGCE adoption top-k/config mismatch")

    fsg = model.fsg
    had_override = "find_fs" in vars(fsg)
    previous_override = vars(fsg).get("find_fs")
    call_count = 0

    def adopted_find_fs(min_freq: int) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count != 1:
            raise RuntimeError("GlobalGCE adopted top-k was requested more than once")
        if int(identity.get("min_freq") or -1) != int(min_freq):
            raise ValueError("GlobalGCE adoption min_freq/config mismatch")
        return list(graphs)

    fsg.find_fs = adopted_find_fs
    try:
        expanded = model.get_fs_expanded_data(train_loader)
    finally:
        if had_override:
            fsg.find_fs = previous_override
        else:
            delattr(fsg, "find_fs")
    if call_count != 1:
        raise RuntimeError("Official GlobalGCE did not consume adopted top-k exactly once")
    return expanded, identity


def train_globalgce_resumable(
    *,
    epochs: int,
    pred_model: Any,
    model: Any,
    learning_rate: float,
    train_loader: Any,
    val_loader: Any,
    save_rule_path: str | Path,
    save_model_path: str | Path,
    checkpoint_dir: str | Path,
    torch_module: Any,
    numpy_module: Any,
    test_globalgce: Any,
    gspan_module: Any,
    resume: bool,
    gspan_flush_every: int = 256,
    gspan_max_in_memory_candidates: int = 256,
    gspan_exact_top_k_pruning: bool = False,
    gspan_adoption_proof: str | Path | None = None,
    on_exact_top_k_proof: Callable[[dict[str, Any]], None] | None = None,
    on_gspan_adoption_proof: Callable[[dict[str, Any]], None] | None = None,
    resume_identity: Mapping[str, Any] | None = None,
) -> Any:
    """Run the official loop with atomic epoch checkpoints and exact RNG state."""

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "training_checkpoint.pt"
    heartbeat_path = checkpoint_root / "training_heartbeat.json"
    if gspan_adoption_proof is not None and gspan_exact_top_k_pruning:
        raise ValueError(
            "GlobalGCE mining adoption and fresh exact-top-k mining are mutually exclusive"
        )
    config = {"epochs": int(epochs), "learning_rate": float(learning_rate)}
    normalized_resume_identity: dict[str, Any] | None = None
    resume_identity_sha256: str | None = None
    if resume_identity is not None:
        normalized_resume_identity, resume_identity_sha256 = (
            normalize_globalgce_training_resume_identity(resume_identity)
        )
    checkpoint: Mapping[str, Any] | None = None
    if resume and checkpoint_path.is_file():
        checkpoint = torch_module.load(checkpoint_path, map_location=model.device)
        if checkpoint.get("config") != config:
            raise ValueError("GlobalGCE training checkpoint configuration mismatch.")
        if normalized_resume_identity is None:
            if checkpoint.get("checkpoint_schema_version") not in (
                None,
                GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1,
            ):
                raise ValueError(
                    "Legacy GlobalGCE caller cannot consume a v2 identity-bound checkpoint."
                )
        else:
            validate_globalgce_epoch_checkpoint_identity(
                checkpoint,
                normalized_resume_identity,
            )
    if gspan_adoption_proof is not None:
        (
            (fss, expanded_train, expanded_val, expanded_test),
            adoption_identity,
        ) = _get_fs_expanded_data_from_adoption(
            model=model,
            train_loader=train_loader,
            proof_path=gspan_adoption_proof,
        )
        if on_gspan_adoption_proof is not None:
            on_gspan_adoption_proof(adoption_identity)
    else:
        with resumable_gspan_root_chunks(
            gspan_module,
            checkpoint_root=checkpoint_root / "gspan",
            top_k=int(model.fsg.topk),
            flush_every=int(gspan_flush_every),
            max_in_memory_candidates=int(gspan_max_in_memory_candidates),
            exact_top_k_pruning=bool(gspan_exact_top_k_pruning),
        ) as gspan_session:
            fss, expanded_train, expanded_val, expanded_test = (
                model.get_fs_expanded_data(train_loader)
            )
    if gspan_exact_top_k_pruning and gspan_adoption_proof is None:
        proofs = list(gspan_session["completed_exact_top_k_proofs"])
        if not proofs:
            raise RuntimeError(
                "GlobalGCE exact-top-k training produced no terminal audit proof."
            )
        proof = validate_exact_top_k_proof_identity(proofs[-1])
        if on_exact_top_k_proof is not None:
            on_exact_top_k_proof(proof)
    optimizer = torch_module.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-5
    )
    scheduler = torch_module.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9
    )
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    next_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_loss = float(checkpoint["best_loss"])
        # The pinned official implementation keeps a shallow ``state_dict``
        # reference, so its selected payload tracks the uninterrupted final
        # parameters.  Rebind after loading to preserve that exact behavior.
        best_state = model.state_dict() if checkpoint.get("best_state_seen") else None
        next_epoch = int(checkpoint["next_epoch"])
        random.setstate(checkpoint["python_rng_state"])
        if normalized_resume_identity is not None:
            _restore_numpy_rng_state(
                numpy_module,
                checkpoint["numpy_rng_state"],
            )
        else:
            numpy_module.random.set_state(checkpoint["numpy_rng_state"])
        torch_module.set_rng_state(checkpoint["torch_rng_state"])
        if torch_module.cuda.is_available() and checkpoint.get("cuda_rng_state"):
            torch_module.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    for epoch in range(next_epoch, int(epochs) + 1):
        model.train()
        model.gt_gnn.eval()
        loss = loss_kl = loss_sim = loss_cfe = 0.0
        rules = model.get_rules(fss)
        for batch_index, data in enumerate(expanded_train):
            if batch_index >= 5:
                break
            values = model.run_one_batch(rules, data)
            loss += values[0]
            loss_kl += values[1]
            loss_sim += values[2]
            loss_cfe += values[3]
        (loss_cfe if epoch < 35 else loss).backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        metrics: dict[str, Any] = {"epoch": epoch}
        if epoch % 5 == 0:
            with torch_module.no_grad():
                model.eval()
                evaluated = test_globalgce(expanded_val, model, pred_model, rules)
                val_loss = float(evaluated["loss"].detach().cpu())
                metrics.update(
                    {
                        "val_loss": val_loss,
                        "val_loss_kl": float(evaluated["loss_kl"].detach().cpu()),
                        "val_loss_sim": float(evaluated["loss_sim"].detach().cpu()),
                        "val_loss_cfe": float(evaluated["loss_cfe"].detach().cpu()),
                    }
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = model.state_dict()
        state = {
            "config": config,
            "checkpoint_schema_version": (
                GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
                if normalized_resume_identity is not None
                else GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1
            ),
            **(
                {
                    "resume_identity": normalized_resume_identity,
                    "resume_identity_sha256": resume_identity_sha256,
                }
                if normalized_resume_identity is not None
                else {}
            ),
            "next_epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_state_seen": best_state is not None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (
                _serialize_numpy_rng_state(numpy_module)
                if normalized_resume_identity is not None
                else numpy_module.random.get_state()
            ),
            "torch_rng_state": torch_module.get_rng_state(),
            "cuda_rng_state": (
                torch_module.cuda.get_rng_state_all()
                if torch_module.cuda.is_available()
                else None
            ),
        }
        _atomic_torch_save(torch_module, state, checkpoint_path)
        _atomic_json(
            heartbeat_path,
            {
                "schema_version": (
                    GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
                    if normalized_resume_identity is not None
                    else GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1
                ),
                "stage": "training",
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "best_loss": best_loss,
                "metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                **(
                    {
                        "resume_identity": normalized_resume_identity,
                        "resume_identity_sha256": resume_identity_sha256,
                    }
                    if normalized_resume_identity is not None
                    else {}
                ),
                "updated_at_epoch_seconds": time.time(),
            },
        )
    if best_state is None:
        raise RuntimeError("GlobalGCE training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    best_rules = model.get_rules(fss)
    _atomic_torch_save(torch_module, best_state, Path(save_model_path).resolve())
    _atomic_torch_save(torch_module, best_rules, Path(save_rule_path).resolve())
    _atomic_json(
        heartbeat_path,
        {
            "schema_version": (
                GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
                if normalized_resume_identity is not None
                else GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V1
            ),
            "stage": "complete",
            "next_epoch": int(epochs) + 1,
            "best_loss": best_loss,
            **(
                {
                    "resume_identity": normalized_resume_identity,
                    "resume_identity_sha256": resume_identity_sha256,
                }
                if normalized_resume_identity is not None
                else {}
            ),
            "updated_at_epoch_seconds": time.time(),
        },
    )
    return expanded_test
