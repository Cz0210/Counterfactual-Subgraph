"""Resumable wrappers around the unmodified GlobalGCE training semantics."""

from __future__ import annotations

import hashlib
import itertools
import json
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
from typing import Any, Iterator


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


def _graph_input_fingerprint(graphs: list[Any], settings: dict[str, Any]) -> str:
    rows = []
    for graph in graphs:
        nodes = sorted(
            (str(node), str(attributes.get("label")))
            for node, attributes in graph.nodes(data=True)
        )
        edges = sorted(
            (
                min(str(left), str(right)),
                max(str(left), str(right)),
                str(attributes.get("label")),
            )
            for left, right, attributes in graph.edges(data=True)
        )
        rows.append({"nodes": nodes, "edges": edges})
    encoded = json.dumps(
        {"settings": settings, "graphs": rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
) -> Iterator[None]:
    """Spill deterministic gSpan root reports and retain only official top-k.

    The official implementation builds every frequent graph in memory and then
    performs a stable support-descending sort before slicing ``topk``.  We store
    the same traversal order in SQLite and apply the equivalent SQL ordering,
    allowing an interrupted root to be replayed without retaining the universe.
    """

    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    gspan_class = gspan_module.gSpan
    original_run = gspan_class.run
    original_report = gspan_class._report
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
    commit_every = min(configured_flush, configured_max)
    min_free_bytes = int(
        float(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_GIB", "50")) * 1024**3
    )
    min_free_ratio = float(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_RATIO", "0.02"))
    min_free_inodes = int(os.environ.get("GLOBALGCE_STORAGE_MIN_FREE_INODES", "100000"))
    active: dict[int, dict[str, Any]] = {}

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
        graph = self._DFScode.to_graph(
            gid=next(self._counter), is_undirected=self._is_undirected
        )
        graph = self._from_Graph_to_nx_Graph(graph)
        local_index = int(context["local_index"])
        context["connection"].execute(
            "INSERT INTO patterns(root_index, local_index, support, payload) "
            "VALUES (?, ?, ?, ?)",
            (
                int(context["root_index"]),
                local_index,
                int(self._support),
                sqlite3.Binary(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
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
            "spill_schema": "sqlite_stable_support_topk_v2",
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
            "schema_version": "globalgce_gspan_sqlite_chunks_v2",
            "stage": "mining",
            "input_fingerprint": fingerprint,
            "root_count": len(top_roots),
            "completed_root_count": 0,
            "current_root_index": None,
            "frequent_subgraph_count": 0,
            "sqlite_path": str(database_path),
            "flush_every": configured_flush,
            "max_in_memory_candidates": configured_max,
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
                        connection.execute(
                            "UPDATE roots SET complete=1, pattern_count=? WHERE root_index=?",
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
                    _atomic_json(support_root / "checkpoint.json", state)
            limit = int(top_k) if top_k is not None else -1
            query = (
                "SELECT support, payload FROM patterns "
                "ORDER BY support DESC, root_index ASC, local_index ASC"
            )
            parameters: tuple[int, ...] = ()
            if limit >= 0:
                query += " LIMIT ?"
                parameters = (limit,)
            selected = connection.execute(query, parameters).fetchall()
            self.freq_collection = [int(row[0]) for row in selected]
            self.fs_collection = [pickle.loads(row[1]) for row in selected]
            self._frequent_subgraphs = []
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
    try:
        yield
    finally:
        gspan_class.run = original_run
        gspan_class._report = original_report


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
) -> Any:
    """Run the official loop with atomic epoch checkpoints and exact RNG state."""

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "training_checkpoint.pt"
    heartbeat_path = checkpoint_root / "training_heartbeat.json"
    with resumable_gspan_root_chunks(
        gspan_module,
        checkpoint_root=checkpoint_root / "gspan",
        top_k=int(model.fsg.topk),
        flush_every=int(gspan_flush_every),
        max_in_memory_candidates=int(gspan_max_in_memory_candidates),
    ):
        fss, expanded_train, expanded_val, expanded_test = model.get_fs_expanded_data(
            train_loader
        )
    optimizer = torch_module.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-5
    )
    scheduler = torch_module.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9
    )
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    next_epoch = 0
    config = {"epochs": int(epochs), "learning_rate": float(learning_rate)}
    if resume and checkpoint_path.is_file():
        checkpoint = torch_module.load(checkpoint_path, map_location=model.device)
        if checkpoint.get("config") != config:
            raise ValueError("GlobalGCE training checkpoint configuration mismatch.")
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
            "next_epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_state_seen": best_state is not None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": numpy_module.random.get_state(),
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
                "schema_version": "globalgce_epoch_checkpoint_v1",
                "stage": "training",
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "best_loss": best_loss,
                "metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
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
            "schema_version": "globalgce_epoch_checkpoint_v1",
            "stage": "complete",
            "next_epoch": int(epochs) + 1,
            "best_loss": best_loss,
            "updated_at_epoch_seconds": time.time(),
        },
    )
    return expanded_test
