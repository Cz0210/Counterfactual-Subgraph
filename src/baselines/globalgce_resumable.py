"""Resumable wrappers around the unmodified GlobalGCE training semantics."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import pickle
import random
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
) -> Iterator[None]:
    """Checkpoint independent top-level DFS roots and drop unused O(n^2) reports."""

    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    gspan_class = gspan_module.gSpan
    original_run = gspan_class.run
    original_report = gspan_class._report
    globals_ = original_run.__globals__
    projected_class = globals_["Projected"]
    pdfs_class = globals_["PDFS"]
    dfsedge_class = globals_["DFSedge"]

    def optimized_report(self: Any, projected: Any) -> None:
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        graph = self._DFScode.to_graph(
            gid=next(self._counter), is_undirected=self._is_undirected
        )
        graph = self._from_Graph_to_nx_Graph(graph)
        self.fs_collection.append(graph)
        self.freq_collection.append(self._support)

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
        }
        fingerprint = _graph_input_fingerprint(self._nx_graph_list, settings)
        support_root = root / f"support_{int(self._min_support)}_{fingerprint[:16]}"
        support_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "globalgce_gspan_root_chunks_v1",
            "stage": "mining",
            "input_fingerprint": fingerprint,
            "root_count": len(top_roots),
            "completed_root_count": 0,
            "current_root_index": None,
            "frequent_subgraph_count": 0,
        }
        completed_fs: list[Any] = []
        completed_freq: list[int] = []
        with _Heartbeat(support_root / "heartbeat.json", state):
            for root_index, (vevlb, projected) in enumerate(top_roots.items()):
                part = support_root / f"root-{root_index:04d}.pkl"
                state["current_root_index"] = root_index
                if part.is_file():
                    saved = pickle.loads(part.read_bytes())
                    if saved.get("input_fingerprint") != fingerprint:
                        raise ValueError("GlobalGCE gSpan root checkpoint fingerprint mismatch.")
                    root_fs = list(saved["frequent_subgraphs"])
                    root_freq = list(saved["frequencies"])
                else:
                    self.fs_collection = []
                    self.freq_collection = []
                    self._DFScode.append(dfsedge_class(0, 1, vevlb))
                    try:
                        self._subgraph_mining(projected)
                    finally:
                        self._DFScode.pop()
                    root_fs = list(self.fs_collection)
                    root_freq = list(self.freq_collection)
                    _atomic_pickle(
                        part,
                        {
                            "input_fingerprint": fingerprint,
                            "root_index": root_index,
                            "root_label": repr(vevlb),
                            "frequent_subgraphs": root_fs,
                            "frequencies": root_freq,
                        },
                    )
                completed_fs.extend(root_fs)
                completed_freq.extend(root_freq)
                state.update(
                    {
                        "completed_root_count": root_index + 1,
                        "frequent_subgraph_count": len(completed_fs),
                    }
                )
                _atomic_json(support_root / "checkpoint.json", state)
        self.fs_collection = completed_fs
        self.freq_collection = completed_freq
        state.update({"stage": "complete", "current_root_index": None})
        _atomic_json(support_root / "checkpoint.json", state)
        return self.fs_collection, self.freq_collection

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
) -> Any:
    """Run the official loop with atomic epoch checkpoints and exact RNG state."""

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "training_checkpoint.pt"
    heartbeat_path = checkpoint_root / "training_heartbeat.json"
    with resumable_gspan_root_chunks(
        gspan_module, checkpoint_root=checkpoint_root / "gspan"
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
