"""Fail-closed adoption of completed exhaustive GlobalGCE gSpan mining.

The failed BACE v5 run completed its *legacy exhaustive* gSpan traversal before
rule training reached the differentiable classifier bridge.  This module can
adopt only that immutable mining result.  It never upgrades the failed parent
run to PASS and never opens the source SQLite database in ordinary read-only
mode: every query uses ``mode=ro&immutable=1``.

Adoption is deliberately expensive.  Checkpoint and SQLite bytes, source
dataset/GINE/config identity, traversal-order-v2 input identity, all completed
roots, and the stable official top-k are revalidated before any selected graph
is returned to training.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from src.baselines.globalgce_resumable import (
    _graph_input_fingerprint,
    _typed_fingerprint_value,
)
from src.eval.bace_frozen_gnn_contracts import stable_sha256


ADOPTION_SCHEMA_VERSION = "globalgce_gspan_exhaustive_v2_adoption_v1"
EXPECTED_OFFICIAL_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"
CHECKPOINT_SCHEMA_VERSION = "globalgce_gspan_sqlite_chunks_v2"
ORDERING = "support_desc_root_index_asc_local_index_asc"


class GlobalGCEMiningAdoptionError(RuntimeError):
    """A v5 mining source cannot be adopted without weakening provenance."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "path": str(path),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _identity(path: Path) -> dict[str, Any]:
    return {**_stat(path), "sha256": _sha256_file(path)}


def _same_stat(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return all(
        before.get(key) == after.get(key)
        for key in ("path", "device", "inode", "bytes", "mtime_ns", "ctime_ns")
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GlobalGCEMiningAdoptionError(f"{description} must be a JSON object")
    return payload


def _resolve_file(value: str | Path, *, description: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    return path


def _resolve_dir(value: str | Path, *, description: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def _sqlite_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=120)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _fd_flags(fdinfo: Path) -> int | None:
    try:
        for line in fdinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("flags:"):
                return int(line.split(":", 1)[1].strip(), 8)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def _writable_fd_holders(paths: Sequence[Path]) -> list[dict[str, Any]]:
    targets = {str(path.resolve(strict=False)) for path in paths}
    holders: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return holders
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            fds = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            normalized = target.removesuffix(" (deleted)")
            if normalized not in targets:
                continue
            flags = _fd_flags(process / "fdinfo" / fd.name)
            if flags is None or (flags & os.O_ACCMODE) == os.O_RDONLY:
                continue
            try:
                command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                command = ""
            holders.append(
                {
                    "pid": int(process.name),
                    "fd": int(fd.name),
                    "flags": flags,
                    "target": normalized,
                    "command": command,
                }
            )
    return holders


def _sidecar_snapshot(database: Path) -> dict[str, Any]:
    wal = Path(str(database) + "-wal")
    shm = Path(str(database) + "-shm")
    result: dict[str, Any] = {}
    for name, path in (("wal", wal), ("shm", shm)):
        result[name] = _identity(path) if path.exists() else {"path": str(path), "exists": False}
    if wal.exists() and wal.stat().st_size != 0:
        raise GlobalGCEMiningAdoptionError(
            "Source SQLite WAL is non-empty; exhaustive mining is not immutable"
        )
    return result


def _official_commit(official_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _task_failure_closure(task_state: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    encoded = json.dumps(task_state, sort_keys=True)
    terminal_values = {
        str(value).upper()
        for key, value in task_state.items()
        if str(key).lower() in {"state", "status", "workload_state", "task_state"}
    }
    nested_states: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in {"state", "status", "workload_state", "task_state"}:
                    nested_states.append(str(item).upper())
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(task_state)
    states = sorted(terminal_values | set(nested_states))
    if "FAILED" not in states:
        raise GlobalGCEMiningAdoptionError("Source controller task is not FAILED")
    if str(source_root) not in encoded:
        raise GlobalGCEMiningAdoptionError(
            "Source controller task does not bind the exact v5 output root"
        )
    if "globalgce" not in encoded.lower() or "train_candidates" not in encoded.lower():
        raise GlobalGCEMiningAdoptionError(
            "Source controller task is not the failed GlobalGCE candidate-training task"
        )
    return {"states": states, "exact_output_root_bound": True}


def _graph_semantics(graph: Any) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "index": index,
                "node": _typed_fingerprint_value(node),
                "label": _typed_fingerprint_value(attributes.get("label")),
            }
            for index, (node, attributes) in enumerate(graph.nodes(data=True))
        ],
        "edges": [
            {
                "index": index,
                "left": _typed_fingerprint_value(left),
                "right": _typed_fingerprint_value(right),
                "label": _typed_fingerprint_value(attributes.get("label")),
            }
            for index, (left, right, attributes) in enumerate(graph.edges(data=True))
        ],
    }


def _query_exhaustive_database(
    database: Path, *, top_k: int
) -> dict[str, Any]:
    with _sqlite_connection(database) as connection:
        metadata = connection.execute(
            "SELECT value FROM metadata WHERE key='input_fingerprint'"
        ).fetchone()
        root_count, completed_root_count, root_pattern_count = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(complete),0), "
            "COALESCE(SUM(pattern_count),0) FROM roots"
        ).fetchone()
        pattern_count = int(
            connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        )
        rows = connection.execute(
            "SELECT support, root_index, local_index, payload FROM patterns "
            "ORDER BY support DESC, root_index ASC, local_index ASC LIMIT ?",
            (int(top_k),),
        ).fetchall()
    selected_rows: list[dict[str, Any]] = []
    graphs: list[Any] = []
    supports: list[int] = []
    for rank, row in enumerate(rows, start=1):
        payload = bytes(row[3])
        graph = pickle.loads(payload)
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise GlobalGCEMiningAdoptionError("Selected SQLite payload is not a graph")
        graphs.append(graph)
        supports.append(int(row[0]))
        selected_rows.append(
            {
                "rank": rank,
                "support": int(row[0]),
                "root_index": int(row[1]),
                "local_index": int(row[2]),
                "payload_sha256": _sha256_bytes(payload),
            }
        )
    selected_identity = stable_sha256(selected_rows)
    semantic_rows = [
        {"rank": index + 1, "support": supports[index], "graph": _graph_semantics(graph)}
        for index, graph in enumerate(graphs)
    ]
    return {
        "input_fingerprint": str(metadata[0]) if metadata else "",
        "root_count": int(root_count),
        "completed_root_count": int(completed_root_count),
        "root_pattern_count": int(root_pattern_count),
        "pattern_count": pattern_count,
        "selected_rows": selected_rows,
        "selected_identity_sha256": selected_identity,
        "selected_semantic_identity_sha256": stable_sha256(semantic_rows),
        "graphs": graphs,
        "supports": supports,
    }


def _recompute_bace_traversal_fingerprint(
    *,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gine_checkpoint: Path,
    min_freq: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Rebuild only the train-loader graph order and gSpan root order."""

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.baselines.globalgce_bace_adapter import (
        OfficialGlobalGCEBACEGenerator,
        audit_bace_globalgce_train_contract,
    )
    from src.baselines.globalgce_mutagenicity_adapter import (
        _import_official_modules,
        _prepare_native_and_source_datasets,
    )
    from src.baselines.bace_gnn_baseline_contracts import validate_bace_frozen_gine

    contract = audit_bace_globalgce_train_contract(
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
    )
    checkpoint, _card, _schema = validate_bace_frozen_gine(gine_checkpoint)
    generator = OfficialGlobalGCEBACEGenerator(
        official_root,
        native_train_csv=native_train_csv,
        min_freq=int(min_freq),
        frozen_gine_checkpoint=checkpoint,
        native_train_parent_ids=contract.native_train_parent_ids,
    )
    modules = _import_official_modules(generator.official_src)
    (
        _native_parents,
        _native_train_idx,
        _native_val_idx,
        _native_dataset,
        source_train_idx,
        _source_val_idx,
        source_dataset,
    ) = _prepare_native_and_source_datasets(
        native_train_csv=native_train_csv,
        parents=contract.source_parents,
        seed=int(seed),
        torch_module=torch,
        dataset_name="BACE",
        native_train_parent_ids=contract.native_train_parent_ids,
    )
    loader = DataLoader(
        Subset(source_dataset, source_train_idx),
        batch_size=500,
        shuffle=False,
    )
    fsg_class = modules["fsg_module"].FrequentSubgraphGenerator
    fsg = fsg_class(3, 20, str(source_manifest.parent / "unused_fs.pkl"), int(top_k), False)
    graphs = fsg.get_nx_graphs(loader)
    gspan_module = modules["gspan_module"]
    miner = gspan_module.gSpan(
        graphs,
        int(min_freq),
        3,
        20,
        len(graphs),
        where=str(source_manifest.parent / "unused_gspan.txt"),
    )
    miner._read_graphs()
    miner._generate_1edge_frequent_subgraphs()
    globals_ = miner.run.__globals__
    top_roots: dict[Any, Any] = defaultdict(globals_["Projected"])
    for graph_id, graph in miner.graphs.items():
        for vertex_id, vertex in graph.vertices.items():
            for edge in miner._get_forward_root_edges(graph, vertex_id):
                top_roots[(vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)].append(
                    globals_["PDFS"](graph_id, edge, None)
                )
    settings = {
        "min_support": int(min_freq),
        "min_vertices": 3,
        "max_vertices": 20,
        "is_undirected": True,
        "root_order": [repr(value) for value in top_roots],
        "top_k": int(top_k),
        "spill_schema": "sqlite_stable_support_topk_v2",
    }
    return {
        "fingerprint_schema": "globalgce_native_traversal_order_v2",
        "input_fingerprint": _graph_input_fingerprint(graphs, settings),
        "graph_count": len(graphs),
        "root_count": len(top_roots),
        "settings": settings,
        "source_train_index_sha256": stable_sha256(list(source_train_idx)),
    }


def _validate_source_manifest_contract(
    *,
    source_run_manifest: Path,
    source_task_state: Path,
    source_checkpoint: Path,
    source_sqlite: Path,
    official_root: Path,
    native_train_csv: Path,
    source_manifest: Path,
    gine_checkpoint: Path,
    expected_official_commit: str,
    expected_pattern_count: int,
    expected_root_count: int,
    min_freq: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_root = source_run_manifest.parent.resolve(strict=True)
    expected_checkpoint_parent = (
        source_root / "native" / "globalgce_training_checkpoints" / "gspan"
    )
    if expected_checkpoint_parent not in source_checkpoint.parents:
        raise GlobalGCEMiningAdoptionError("Checkpoint is outside the v5 gSpan root")
    if source_checkpoint.parent != source_sqlite.parent:
        raise GlobalGCEMiningAdoptionError("Checkpoint and SQLite roots differ")
    if source_checkpoint.name != "checkpoint.json" or source_sqlite.name != "frequent_patterns.sqlite3":
        raise GlobalGCEMiningAdoptionError("Unexpected gSpan source filenames")
    if (source_root / "PASS").exists() or (source_root / "_RUN_COMPLETE.json").exists():
        raise GlobalGCEMiningAdoptionError("Failed v5 source unexpectedly claims terminal PASS")

    run = _json_object(source_run_manifest, description="v5 run manifest")
    task = _json_object(source_task_state, description="v5 task state")
    checkpoint = _json_object(source_checkpoint, description="v5 gSpan checkpoint")
    task_closure = _task_failure_closure(task, source_root)
    if run.get("run_complete") is not False or str(run.get("status") or "").upper() == "PASS":
        raise GlobalGCEMiningAdoptionError("v5 parent run must remain an incomplete failure")
    config = dict(run.get("config") or {})
    required_config = {
        "seed": int(seed),
        "top_k_native": int(top_k),
        "min_freq": int(min_freq),
        "gspan_exact_top_k_pruning": False,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise GlobalGCEMiningAdoptionError(
                f"v5 run config mismatch for {key}: {config.get(key)!r} != {expected!r}"
            )
    if (
        run.get("oracle_backend") != "gnn"
        or run.get("classifier_family") != "gine"
        or run.get("rf_oracle_used") is not False
        or run.get("calibration_loaded") is not False
        or run.get("test_loaded") is not False
    ):
        raise GlobalGCEMiningAdoptionError("v5 run is not frozen-GINE train-only")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("stage") != "complete"
        or checkpoint.get("exact_top_k_pruning") is not False
        or int(checkpoint.get("root_count", -1)) != int(expected_root_count)
        or int(checkpoint.get("completed_root_count", -1)) != int(expected_root_count)
        or int(checkpoint.get("frequent_subgraph_count", -1))
        != int(expected_pattern_count)
    ):
        raise GlobalGCEMiningAdoptionError("v5 exhaustive checkpoint is not complete")

    commit = _official_commit(official_root)
    if commit != str(expected_official_commit):
        raise GlobalGCEMiningAdoptionError(
            f"Pinned official commit mismatch: {commit} != {expected_official_commit}"
        )
    traversal = _recompute_bace_traversal_fingerprint(
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
        official_root=official_root,
        gine_checkpoint=gine_checkpoint,
        min_freq=int(min_freq),
        top_k=int(top_k),
        seed=int(seed),
    )
    if traversal["input_fingerprint"] != checkpoint.get("input_fingerprint"):
        raise GlobalGCEMiningAdoptionError("v5 traversal-order-v2 fingerprint mismatch")
    database = _query_exhaustive_database(source_sqlite, top_k=int(top_k))
    if (
        database["input_fingerprint"] != checkpoint.get("input_fingerprint")
        or database["root_count"] != int(expected_root_count)
        or database["completed_root_count"] != int(expected_root_count)
        or database["root_pattern_count"] != int(expected_pattern_count)
        or database["pattern_count"] != int(expected_pattern_count)
        or len(database["graphs"]) != int(top_k)
    ):
        raise GlobalGCEMiningAdoptionError("v5 SQLite exhaustive closure failed")
    return (
        {
            "source_root": str(source_root),
            "parent_run_complete": False,
            "parent_status": run.get("status"),
            "task_failure": task_closure,
            "official_commit": commit,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "calibration_loaded": False,
            "test_loaded": False,
        },
        traversal,
        database,
    )


def build_globalgce_gspan_adoption(
    *,
    source_run_manifest: str | Path,
    source_task_state: str | Path,
    source_checkpoint: str | Path,
    source_sqlite: str | Path,
    official_root: str | Path,
    native_train_csv: str | Path,
    source_manifest: str | Path,
    gine_checkpoint: str | Path,
    output_dir: str | Path,
    expected_official_commit: str = EXPECTED_OFFICIAL_COMMIT,
    expected_pattern_count: int = 5_441_858,
    expected_root_count: int = 19,
    min_freq: int = 7,
    top_k: int = 20,
    seed: int = 13,
) -> dict[str, Any]:
    """Create a fresh proof and selected top-k without modifying v5 bytes."""

    paths = {
        "source_run_manifest": _resolve_file(source_run_manifest, description="source run manifest"),
        "source_task_state": _resolve_file(source_task_state, description="source task state"),
        "source_checkpoint": _resolve_file(source_checkpoint, description="source checkpoint"),
        "source_sqlite": _resolve_file(source_sqlite, description="source SQLite"),
        "official_root": _resolve_dir(official_root, description="official root"),
        "native_train_csv": _resolve_file(native_train_csv, description="native train CSV"),
        "source_manifest": _resolve_file(source_manifest, description="source manifest"),
        "gine_checkpoint": _resolve_dir(gine_checkpoint, description="GINE checkpoint"),
    }
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Adoption output must be fresh: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    database = paths["source_sqlite"]
    checkpoint_path = paths["source_checkpoint"]
    sidecars_before = _sidecar_snapshot(database)
    source_targets = [
        database,
        checkpoint_path,
        Path(str(database) + "-wal"),
        Path(str(database) + "-shm"),
    ]
    holders_before = _writable_fd_holders([path for path in source_targets if path.exists()])
    if holders_before:
        raise GlobalGCEMiningAdoptionError("Source mining files retain a writable FD")
    database_before = _identity(database)
    checkpoint_before = _identity(checkpoint_path)
    source_scope, traversal, query = _validate_source_manifest_contract(
        source_run_manifest=paths["source_run_manifest"],
        source_task_state=paths["source_task_state"],
        source_checkpoint=checkpoint_path,
        source_sqlite=database,
        official_root=paths["official_root"],
        native_train_csv=paths["native_train_csv"],
        source_manifest=paths["source_manifest"],
        gine_checkpoint=paths["gine_checkpoint"],
        expected_official_commit=expected_official_commit,
        expected_pattern_count=int(expected_pattern_count),
        expected_root_count=int(expected_root_count),
        min_freq=int(min_freq),
        top_k=int(top_k),
        seed=int(seed),
    )
    database_after = _identity(database)
    checkpoint_after = _identity(checkpoint_path)
    sidecars_after = _sidecar_snapshot(database)
    holders_after = _writable_fd_holders([path for path in source_targets if path.exists()])
    if (
        database_before != database_after
        or checkpoint_before != checkpoint_after
        or sidecars_before != sidecars_after
        or holders_after
    ):
        raise GlobalGCEMiningAdoptionError("Source mining bytes/stat/writer state drifted")

    selected_payload = {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "ordering": ORDERING,
        "graphs": query.pop("graphs"),
        "supports": query.pop("supports"),
    }
    selected_path = destination / "selected_top20.pkl"
    _atomic_bytes(selected_path, pickle.dumps(selected_payload, protocol=pickle.HIGHEST_PROTOCOL))
    source_payload = {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "status": "SOURCE_FAILED_AFTER_MINING",
        "source_scope": source_scope,
        "inputs": {
            key: _identity(value / "model.pt" if key == "gine_checkpoint" else value)
            if key != "official_root"
            else {"path": str(value), "commit": _official_commit(value)}
            for key, value in paths.items()
        },
        "source_checkpoint": checkpoint_before,
        "source_sqlite": database_before,
        "sqlite_sidecars": sidecars_before,
        "no_writable_fd": True,
    }
    source_path = destination / "source_manifest.json"
    _atomic_json(source_path, source_payload)
    proof = {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "status": "PASS",
        "proof": {
            "dataset_gine_config_train_only_closure": True,
            "checkpoint_sqlite_complete_closure": True,
            "stable_top20_recomputed": True,
            "source_no_writer": True,
            "parent_run_inherited_pass": False,
        },
        "source_manifest": _identity(source_path),
        "selected_top20": _identity(selected_path),
        "source_checkpoint": checkpoint_before,
        "source_sqlite": database_before,
        "sqlite_sidecars": sidecars_before,
        "input_fingerprint": traversal["input_fingerprint"],
        "traversal_identity": traversal,
        "selected_rows": query["selected_rows"],
        "selected_identity_sha256": query["selected_identity_sha256"],
        "selected_semantic_identity_sha256": query[
            "selected_semantic_identity_sha256"
        ],
        "root_count": int(query["root_count"]),
        "pattern_count": int(query["pattern_count"]),
        "min_freq": int(min_freq),
        "top_k": int(top_k),
        "seed": int(seed),
        "ordering": ORDERING,
        "source_scope": source_scope,
    }
    proof_path = destination / "adoption_proof.json"
    _atomic_json(proof_path, proof)
    _atomic_bytes(destination / "ADOPTION_PASS", b"[BACE_GLOBALGCE_V5_GSPAN_ADOPTION_PASS]\n")
    _atomic_bytes(destination / "PASS", b"PASS\n")
    return validate_globalgce_gspan_adoption_proof(proof_path)


def validate_globalgce_gspan_adoption_proof(
    proof_path: str | Path,
) -> dict[str, Any]:
    """Deeply revalidate source bytes, immutable SQLite, and selected payload."""

    path = _resolve_file(proof_path, description="adoption proof")
    root = path.parent
    if not (root / "PASS").is_file() or not (root / "ADOPTION_PASS").is_file():
        raise GlobalGCEMiningAdoptionError("Adoption proof lacks PASS-last markers")
    proof = _json_object(path, description="adoption proof")
    if proof.get("schema_version") != ADOPTION_SCHEMA_VERSION or proof.get("status") != "PASS":
        raise GlobalGCEMiningAdoptionError("Adoption proof schema/status mismatch")
    source_path = root / "source_manifest.json"
    selected_path = root / "selected_top20.pkl"
    if proof.get("source_manifest") != _identity(source_path):
        raise GlobalGCEMiningAdoptionError("Adoption source-manifest bytes changed")
    if proof.get("selected_top20") != _identity(selected_path):
        raise GlobalGCEMiningAdoptionError("Adopted selected-top20 bytes changed")
    source = _json_object(source_path, description="adoption source manifest")
    inputs = dict(source.get("inputs") or {})
    for name, stored in inputs.items():
        if name == "official_root":
            continue
        current_path = Path(str(stored.get("path") or "")).resolve(strict=True)
        if stored != _identity(current_path):
            raise GlobalGCEMiningAdoptionError(
                f"Adoption input bytes/stat changed: "
                f"{'source SQLite' if name == 'source_sqlite' else name}"
            )
    source_root = Path(str(source["source_scope"]["source_root"])).resolve(strict=True)
    _task_failure_closure(
        _json_object(
            Path(str(inputs["source_task_state"]["path"])).resolve(strict=True),
            description="v5 task state",
        ),
        source_root,
    )
    current_run = _json_object(
        Path(str(inputs["source_run_manifest"]["path"])).resolve(strict=True),
        description="v5 run manifest",
    )
    if current_run.get("run_complete") is not False or str(
        current_run.get("status") or ""
    ).upper() == "PASS":
        raise GlobalGCEMiningAdoptionError("v5 parent failure status changed")
    checkpoint = Path(str(proof["source_checkpoint"]["path"])).resolve(strict=True)
    database = Path(str(proof["source_sqlite"]["path"])).resolve(strict=True)
    if proof.get("source_checkpoint") != _identity(checkpoint):
        raise GlobalGCEMiningAdoptionError("Source checkpoint bytes/stat changed")
    if proof.get("source_sqlite") != _identity(database):
        raise GlobalGCEMiningAdoptionError("Source SQLite bytes/stat changed")
    sidecars = _sidecar_snapshot(database)
    if proof.get("sqlite_sidecars") != sidecars:
        raise GlobalGCEMiningAdoptionError("Source SQLite sidecar state changed")
    targets = [database, checkpoint, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
    if _writable_fd_holders([item for item in targets if item.exists()]):
        raise GlobalGCEMiningAdoptionError("Source mining files now have a writable FD")
    query = _query_exhaustive_database(database, top_k=int(proof["top_k"]))
    graphs = query.pop("graphs")
    supports = query.pop("supports")
    if (
        query["input_fingerprint"] != proof.get("input_fingerprint")
        or query["selected_rows"] != proof.get("selected_rows")
        or query["selected_identity_sha256"] != proof.get("selected_identity_sha256")
        or query["selected_semantic_identity_sha256"]
        != proof.get("selected_semantic_identity_sha256")
        or query["root_count"] != int(proof.get("root_count", -1))
        or query["completed_root_count"] != int(proof.get("root_count", -1))
        or query["pattern_count"] != int(proof.get("pattern_count", -1))
        or query["root_pattern_count"] != int(proof.get("pattern_count", -1))
    ):
        raise GlobalGCEMiningAdoptionError("Source SQLite proof closure changed")
    selected = pickle.loads(selected_path.read_bytes())
    if (
        not isinstance(selected, dict)
        or selected.get("schema_version") != ADOPTION_SCHEMA_VERSION
        or selected.get("ordering") != ORDERING
        or list(selected.get("supports") or []) != supports
        or stable_sha256(
            [
                {"rank": index + 1, "support": supports[index], "graph": _graph_semantics(graph)}
                for index, graph in enumerate(selected.get("graphs") or [])
            ]
        )
        != proof.get("selected_semantic_identity_sha256")
        or len(selected.get("graphs") or []) != int(proof.get("top_k", -1))
    ):
        raise GlobalGCEMiningAdoptionError("Fresh selected-top20 payload closure failed")
    # Rebuild the train traversal on every deep validation.  This binds the
    # proof to the current dataset/GINE/official bytes, not merely v5 JSON.
    official = Path(str(inputs["official_root"]["path"])).resolve(strict=True)
    traversal = _recompute_bace_traversal_fingerprint(
        source_manifest=Path(str(inputs["source_manifest"]["path"])).resolve(strict=True),
        native_train_csv=Path(str(inputs["native_train_csv"]["path"])).resolve(strict=True),
        official_root=official,
        gine_checkpoint=Path(str(inputs["gine_checkpoint"]["path"])).parent.resolve(strict=True),
        min_freq=int(proof["min_freq"]),
        top_k=int(proof["top_k"]),
        seed=int(proof["seed"]),
    )
    if traversal != proof.get("traversal_identity"):
        raise GlobalGCEMiningAdoptionError("Current traversal identity differs from v5")
    if _official_commit(official) != EXPECTED_OFFICIAL_COMMIT:
        raise GlobalGCEMiningAdoptionError("Pinned official commit changed")
    return {
        key: proof[key]
        for key in (
            "schema_version",
            "status",
            "proof",
            "source_manifest",
            "selected_top20",
            "source_checkpoint",
            "source_sqlite",
            "input_fingerprint",
            "selected_identity_sha256",
            "selected_semantic_identity_sha256",
            "root_count",
            "pattern_count",
            "min_freq",
            "top_k",
            "ordering",
            "source_scope",
        )
    }


def load_adopted_globalgce_top_k(
    proof_path: str | Path,
    *,
    validated_identity: Mapping[str, Any] | None = None,
) -> tuple[list[Any], list[int]]:
    """Return selected graphs only after a fresh deep proof validation."""

    observed = validate_globalgce_gspan_adoption_proof(proof_path)
    if validated_identity is not None and dict(validated_identity) != observed:
        raise GlobalGCEMiningAdoptionError("Prevalidated adoption identity changed")
    selected_path = Path(proof_path).expanduser().resolve(strict=True).parent / "selected_top20.pkl"
    payload = pickle.loads(selected_path.read_bytes())
    return list(payload["graphs"]), [int(value) for value in payload["supports"]]


__all__ = [
    "ADOPTION_SCHEMA_VERSION",
    "EXPECTED_OFFICIAL_COMMIT",
    "GlobalGCEMiningAdoptionError",
    "build_globalgce_gspan_adoption",
    "load_adopted_globalgce_top_k",
    "validate_globalgce_gspan_adoption_proof",
]
