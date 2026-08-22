"""Equivalence-gated acceleration helpers for the native GCFExplainer VRRW.

The helpers in this module deliberately do not own any scientific choices.  In
particular they never enumerate a different edit neighbourhood, consume a
random number, change the VRRW budget, or reorder an action.  The optimized
runtime is opt-in and a full run is allowed only after independent 500- and
1000-step canonical-equivalence checks plus a same-GPU throughput gate.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, TypeVar


LEGACY_ACCELERATION_MODE = "legacy"
ORDERED_ACCELERATION_MODE = "ordered_v2"
SUPPORTED_ACCELERATION_MODES = frozenset(
    {LEGACY_ACCELERATION_MODE, ORDERED_ACCELERATION_MODE}
)
EQUIVALENCE_BUDGETS = (500, 1000)
MINIMUM_AB_SPEEDUP_FRACTION = 0.20
MAXIMUM_SHARED_VRAM_FRACTION = 0.70

T = TypeVar("T")
U = TypeVar("U")


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write fresh gate evidence without replacing another run's evidence."""

    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Acceleration evidence must be fresh: {path}")
    temporary = path.with_name(f".{path.name}.{random.SystemRandom().randrange(1 << 32):08x}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GCFAccelerationConfig:
    """Runtime-only knobs whose semantics are checked by replay gates."""

    mode: str = LEGACY_ACCELERATION_MODE
    gine_batch_size: int = 256
    graph_cache_capacity: int = 0
    cpu_neighbor_workers: int = 1
    progress_every: int = 1000

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_ACCELERATION_MODES:
            raise ValueError(f"Unsupported GCF acceleration mode: {self.mode}")
        for field_name in (
            "gine_batch_size",
            "cpu_neighbor_workers",
            "progress_every",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.graph_cache_capacity, bool)
            or not isinstance(self.graph_cache_capacity, int)
            or self.graph_cache_capacity < 0
        ):
            raise ValueError("graph_cache_capacity cannot be negative")
        if self.mode == LEGACY_ACCELERATION_MODE:
            if self.graph_cache_capacity != 0 or self.cpu_neighbor_workers != 1:
                raise ValueError(
                    "legacy mode forbids graph caching and parallel neighbour construction"
                )
        elif self.graph_cache_capacity <= 0:
            raise ValueError("ordered_v2 requires a positive graph cache capacity")

    @property
    def fingerprint(self) -> str:
        return _stable_json_sha256(asdict(self))


class OrderedLRU:
    """Small deterministic LRU used only for pure graph/model results."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._values: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._values:
            return None
        value = self._values.pop(key)
        self._values[key] = value
        return value

    def put(self, key: str, value: Any) -> None:
        if self.capacity <= 0:
            return
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)


def ordered_parallel_map(
    function: Callable[[T], U],
    values: Sequence[T],
    *,
    workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[U]:
    """Evaluate a proven-pure function concurrently and restore input order."""

    if workers <= 1 or len(values) <= 1:
        return [function(value) for value in values]
    if executor is not None:
        return list(executor.map(function, values))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gcf-pure") as pool:
        return list(pool.map(function, values))


class OrderedNeighbourAcceleration(AbstractContextManager["OrderedNeighbourAcceleration"]):
    """Patch official neighbour builders without changing their action order.

    Only ``neighbor_graph_access`` calls are concurrent.  Action enumeration,
    bridge discovery, sampling, random choices, importance calculation and
    transition insertion remain in the official main thread.
    """

    _NAMES = ("edge_change", "node_label_change", "node_addition", "node_removal")

    def __init__(self, vrrw: Any, *, workers: int) -> None:
        if workers <= 1:
            raise ValueError("Ordered neighbour acceleration requires workers > 1")
        self.vrrw = vrrw
        self.workers = int(workers)
        self._originals: dict[str, Callable[..., Any]] = {}
        self._pool: ThreadPoolExecutor | None = None

    def _graphs(self, graph: Any, actions: Sequence[tuple[Any, ...]]) -> list[Any]:
        assert self._pool is not None
        return ordered_parallel_map(
            lambda action: self.vrrw.neighbor_graph_access(graph, action),
            list(actions),
            workers=self.workers,
            executor=self._pool,
        )

    def _edge_change(
        self, graph: Any, keep_bridge: bool = True, only_removal: bool = False
    ) -> tuple[list[tuple[Any, ...]], list[Any]]:
        nxg = self.vrrw.torch_utils.to_networkx(graph, to_undirected=True)
        bridges = set(self.vrrw.nx.bridges(nxg)) if keep_bridge else set()
        actions: list[tuple[Any, ...]] = []
        for i in range(int(graph.x.shape[0])):
            for j in range(i + 1, int(graph.x.shape[0])):
                if nxg.has_edge(i, j):
                    action = "ER" if keep_bridge and (i, j) not in bridges else "ERR"
                    actions.append((action, i, j))
                elif not only_removal:
                    actions.append(("EA", i, j))
        return actions, self._graphs(graph, actions)

    def _node_label_change(self, graph: Any) -> tuple[list[tuple[Any, ...]], list[Any]]:
        actions = [
            ("NLC", i, j)
            for i in range(int(graph.x.shape[0]))
            for j in range(int(graph.x.shape[1]))
            if graph.x[i, j] != 1
        ]
        return actions, self._graphs(graph, actions)

    def _node_addition(self, graph: Any) -> tuple[list[tuple[Any, ...]], list[Any]]:
        actions = [
            ("NA", i, j)
            for i in range(int(graph.x.shape[0]))
            for j in range(int(graph.x.shape[1]))
        ]
        return actions, self._graphs(graph, actions)

    def _node_removal(self, graph: Any) -> tuple[list[tuple[Any, ...]], list[Any]]:
        degree = self.vrrw.torch_utils.degree(
            graph.edge_index[0], num_nodes=graph.num_nodes
        )
        actions = [
            ("NR", i, i) for i in range(len(degree)) if degree[i] == 1
        ]
        return actions, self._graphs(graph, actions)

    def __enter__(self) -> "OrderedNeighbourAcceleration":
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="gcf-neighbour"
        )
        self._originals = {name: getattr(self.vrrw, name) for name in self._NAMES}
        self.vrrw.edge_change = self._edge_change
        self.vrrw.node_label_change = self._node_label_change
        self.vrrw.node_addition = self._node_addition
        self.vrrw.node_removal = self._node_removal
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for name, function in self._originals.items():
            setattr(self.vrrw, name, function)
        self._originals.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None


class VRRWPhaseProfiler(AbstractContextManager["VRRWPhaseProfiler"]):
    """In-process timings/progress with buffered, semantics-free reporting."""

    def __init__(
        self,
        *,
        vrrw: Any,
        importance: Any,
        progress_every: int,
        emit: Callable[..., None] = print,
    ) -> None:
        self.vrrw = vrrw
        self.importance = importance
        self.progress_every = int(progress_every)
        self.emit = emit
        self.phase_seconds: dict[str, float] = {}
        self.phase_calls: dict[str, int] = {}
        self.steps = 0
        self.started = 0.0
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []
        self._lock = threading.Lock()

    def _wrap(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                with self._lock:
                    self.phase_seconds[label] = self.phase_seconds.get(label, 0.0) + elapsed
                    self.phase_calls[label] = self.phase_calls.get(label, 0) + 1

        self._originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def __enter__(self) -> "VRRWPhaseProfiler":
        self.started = time.perf_counter()
        for name in ("edge_change", "node_label_change", "node_addition", "node_removal"):
            self._wrap(self.vrrw, name, f"neighbour.{name}")
        self._wrap(self.importance, "call", "importance.call")
        original_move = self.vrrw.move_to_next_graph

        def progress_move(*args: Any, **kwargs: Any) -> Any:
            result = original_move(*args, **kwargs)
            self.steps += 1
            if self.steps % self.progress_every == 0:
                self.emit(
                    "[GCF_VRRW_PROGRESS] "
                    + json.dumps(
                        {
                            "steps": self.steps,
                            "elapsed_seconds": round(
                                time.perf_counter() - self.started, 3
                            ),
                            "visited_graphs": len(self.vrrw.graph_map),
                            "cached_transitions": len(self.vrrw.transitions),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return result

        self._originals.append((self.vrrw, "move_to_next_graph", original_move))
        self.vrrw.move_to_next_graph = progress_move
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for owner, name, function in reversed(self._originals):
            setattr(owner, name, function)
        self._originals.clear()

    def report(self) -> dict[str, Any]:
        wall = max(0.0, time.perf_counter() - self.started)
        return {
            "wall_seconds": wall,
            "steps": self.steps,
            "steps_per_second": self.steps / wall if wall > 0 else 0.0,
            "phase_seconds": dict(sorted(self.phase_seconds.items())),
            "phase_calls": dict(sorted(self.phase_calls.items())),
        }


class BufferedVRRWLogging(AbstractContextManager["BufferedVRRWLogging"]):
    """Replace per-step tqdm writes; the profiler emits fixed-interval JSON."""

    def __init__(self, vrrw: Any) -> None:
        self.vrrw = vrrw
        self._original: Callable[..., Any] | None = None

    def __enter__(self) -> "BufferedVRRWLogging":
        self._original = self.vrrw.tqdm
        self.vrrw.tqdm = lambda iterable, *args, **kwargs: iterable
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            self.vrrw.tqdm = self._original
            self._original = None


def _tensor_digest(value: Any) -> str:
    tensor = value.detach().cpu()
    if getattr(tensor, "is_sparse", False):
        tensor = tensor.to_dense()
    tensor = tensor.contiguous()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_graph_tensor_digest(graph: Any) -> str:
    """Hash exact graph tensors without process-randomized Python hashes."""

    payload = {
        "num_nodes": int(graph.num_nodes),
        "x": _tensor_digest(graph.x),
        "edge_index": _tensor_digest(graph.edge_index),
    }
    for name in ("gcf_origin_index", "gcf_node_origin"):
        value = getattr(graph, name, None)
        if value is not None:
            payload[name] = _tensor_digest(value)
    return _stable_json_sha256(payload)


class OrderedImportanceAcceleration(
    AbstractContextManager["OrderedImportanceAcceleration"]
):
    """Cache pure GINE/NeuroSED rows and reassemble every batch in input order."""

    def __init__(self, importance: Any, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Ordered importance cache requires positive capacity")
        self.importance = importance
        self.cache = OrderedLRU(capacity)
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self._original: Callable[..., Any] | None = None

    def __enter__(self) -> "OrderedImportanceAcceleration":
        original = self.importance.call
        self._original = original

        def cached_call(graphs: Sequence[Any], wargs: Mapping[str, Any]) -> Any:
            self.calls += 1
            keys = [canonical_graph_tensor_digest(graph) for graph in graphs]
            rows: dict[str, tuple[Any, Any, Any]] = {}
            missing_keys: list[str] = []
            missing_graphs: list[Any] = []
            for key, graph in zip(keys, graphs, strict=True):
                if key in rows:
                    self.hits += 1
                    continue
                value = self.cache.get(key)
                if value is not None:
                    rows[key] = value
                    self.hits += 1
                    continue
                rows[key] = None  # type: ignore[assignment]
                missing_keys.append(key)
                missing_graphs.append(graph)
                self.misses += 1
            if missing_graphs:
                importance_rows, embeddings, coverage = original(
                    missing_graphs, wargs
                )
                for index, key in enumerate(missing_keys):
                    value = (
                        importance_rows[index].copy(),
                        embeddings[index].copy(),
                        coverage[index].detach().cpu().clone(),
                    )
                    rows[key] = value
                    self.cache.put(key, value)
            try:
                import numpy as np
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Ordered importance cache requires NumPy/Torch") from exc
            return (
                np.stack([rows[key][0] for key in keys]),
                np.stack([rows[key][1] for key in keys]),
                torch.stack([rows[key][2] for key in keys]),
            )

        self.importance.call = cached_call
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            self.importance.call = self._original
            self._original = None

    def report(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_entries": len(self.cache),
        }


def _random_state_hashes(torch: Any, np: Any) -> dict[str, str | None]:
    numpy_state = np.random.get_state()
    numpy_payload = [
        str(numpy_state[0]),
        hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4]),
    ]
    cuda_hash: str | None = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        cuda_hash = _stable_json_sha256([_tensor_digest(state) for state in cuda_states])
    return {
        "python_random_sha256": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
        "numpy_random_sha256": _stable_json_sha256(numpy_payload),
        "torch_random_sha256": _tensor_digest(torch.get_rng_state()),
        "torch_cuda_random_sha256": cuda_hash,
    }


def build_vrrw_equivalence_trace(
    *, payload: Mapping[str, Any], torch: Any, np: Any, budget: int
) -> dict[str, Any]:
    graph_map = dict(payload.get("graph_map", {}))
    identities = {
        key: canonical_graph_tensor_digest(graph) for key, graph in graph_map.items()
    }
    traversed = [identities[value] for value in payload.get("traversed_hashes", [])]
    candidates: list[dict[str, Any]] = []
    for row in payload.get("counterfactual_candidates", []):
        covering = row.get("input_graphs_covering_list")
        candidates.append(
            {
                "graph": identities[row["graph_hash"]],
                "frequency": int(row.get("frequency", 0)),
                "importance": [float(value) for value in row.get("importance_parts", [])],
                "covering": _tensor_digest(covering) if covering is not None else None,
            }
        )
    if len(traversed) != int(budget):
        raise ValueError(
            f"VRRW trace length {len(traversed)} does not match budget {budget}"
        )
    trace = {
        "schema_version": 1,
        "equivalence": "CANONICAL_EXACT_GRAPH_TENSORS_V1",
        "budget": int(budget),
        "traversed_count": len(traversed),
        "traversed_canonical_sha256": _stable_json_sha256(traversed),
        "candidate_count": len(candidates),
        "candidate_canonical_sha256": _stable_json_sha256(candidates),
        "graph_map_count": len(graph_map),
        "graph_identity_multiset_sha256": _stable_json_sha256(sorted(identities.values())),
        **_random_state_hashes(torch, np),
    }
    trace["trace_sha256"] = _stable_json_sha256(trace)
    return trace


_SCIENTIFIC_CONFIG_KEYS = (
    "dataset",
    "dataset_name",
    "gnn_checkpoint_sha256",
    "neurosed_checkpoint_sha256",
    "neurosed_manifest_sha256",
    "parent_limit",
    "generation_parent_ids_sha256",
    "generation_source_cohort_hash",
    "M",
    "alpha",
    "theta",
    "teleport",
    "candidate_capacity",
    "sample",
    "sample_size",
    "seed",
)

_FULL_COMPATIBILITY_KEYS = (
    "dataset",
    "dataset_name",
    "gnn_checkpoint_sha256",
    "neurosed_checkpoint_sha256",
    "neurosed_manifest_sha256",
    "alpha",
    "theta",
    "teleport",
    "candidate_capacity",
    "sample",
    "sample_size",
    "seed",
    "node_feature_dim",
    "dataset_fingerprint",
    "official_compatibility_patches",
)


def scientific_replay_contract_sha256(manifest: Mapping[str, Any]) -> str:
    """Identity shared by 64-parent smokes and the 360-parent full run."""

    return _stable_json_sha256(
        {key: manifest.get(key) for key in _FULL_COMPATIBILITY_KEYS}
    )


def compare_vrrw_equivalence(
    legacy_root: Path,
    optimized_root: Path,
    *,
    budget: int,
) -> dict[str, Any]:
    if int(budget) not in EQUIVALENCE_BUDGETS:
        raise ValueError(f"Equivalence budget must be one of {EQUIVALENCE_BUDGETS}")
    documents: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for root in (legacy_root, optimized_root):
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        trace = json.loads((root / "equivalence_trace.json").read_text(encoding="utf-8"))
        documents.append((manifest, trace))
    legacy, optimized = documents
    failures: list[str] = []
    for key in _SCIENTIFIC_CONFIG_KEYS:
        if legacy[0].get(key) != optimized[0].get(key):
            failures.append(f"scientific_config_mismatch:{key}")
    if legacy[0].get("acceleration", {}).get("mode") != LEGACY_ACCELERATION_MODE:
        failures.append("baseline_not_legacy")
    if optimized[0].get("acceleration", {}).get("mode") != ORDERED_ACCELERATION_MODE:
        failures.append("candidate_not_ordered_v2")
    for _manifest, trace in (legacy, optimized):
        if int(trace.get("budget", -1)) != int(budget):
            failures.append("trace_budget_mismatch")
    compared = (
        "traversed_count",
        "traversed_canonical_sha256",
        "candidate_count",
        "candidate_canonical_sha256",
        "graph_map_count",
        "graph_identity_multiset_sha256",
        "python_random_sha256",
        "numpy_random_sha256",
        "torch_random_sha256",
        "torch_cuda_random_sha256",
    )
    mismatches = [key for key in compared if legacy[1].get(key) != optimized[1].get(key)]
    failures.extend(f"canonical_mismatch:{key}" for key in mismatches)
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAILED",
        "budget": int(budget),
        "equivalence": "CANONICAL_EXACT" if not failures else "NOT_EQUIVALENT",
        "legacy_root": str(legacy_root.resolve()),
        "optimized_root": str(optimized_root.resolve()),
        "legacy_trace_sha256": legacy[1].get("trace_sha256"),
        "optimized_trace_sha256": optimized[1].get("trace_sha256"),
        "optimized_config_fingerprint": optimized[0].get("acceleration", {}).get(
            "fingerprint"
        ),
        "scientific_replay_contract_sha256": scientific_replay_contract_sha256(
            optimized[0]
        ),
        "compared_fields": list(compared),
        "failures": failures,
    }


def build_acceleration_gate(
    *,
    equivalence_markers: Sequence[Path],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in equivalence_markers]
    failures: list[str] = []
    if sorted(int(row.get("budget", -1)) for row in markers) != list(EQUIVALENCE_BUDGETS):
        failures.append("requires_exact_500_and_1000_equivalence_markers")
    if any(row.get("status") != "PASS" for row in markers):
        failures.append("equivalence_marker_failed")
    fingerprints = {row.get("optimized_config_fingerprint") for row in markers}
    if len(fingerprints) != 1 or None in fingerprints:
        failures.append("optimized_config_fingerprint_mismatch")
    scientific_contracts = {
        row.get("scientific_replay_contract_sha256") for row in markers
    }
    if len(scientific_contracts) != 1 or None in scientific_contracts:
        failures.append("scientific_replay_contract_mismatch")
    if benchmark.get("status") != "PASS":
        failures.append("benchmark_failed")
    speedup = float(benchmark.get("speedup_fraction", -1.0))
    if speedup < MINIMUM_AB_SPEEDUP_FRACTION:
        failures.append("speedup_below_20_percent")
    if benchmark.get("same_gpu_uuid") is not True:
        failures.append("benchmark_not_same_physical_gpu")
    peak_fraction = float(benchmark.get("peak_vram_fraction", 1.0))
    if peak_fraction > MAXIMUM_SHARED_VRAM_FRACTION:
        failures.append("peak_vram_exceeds_70_percent")
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAILED",
        "budgets": list(EQUIVALENCE_BUDGETS),
        "optimized_config_fingerprint": next(iter(fingerprints), None),
        "scientific_replay_contract_sha256": next(
            iter(scientific_contracts), None
        ),
        "minimum_speedup_fraction": MINIMUM_AB_SPEEDUP_FRACTION,
        "maximum_vram_fraction": MAXIMUM_SHARED_VRAM_FRACTION,
        "benchmark": dict(benchmark),
        "equivalence_markers": [str(path.resolve()) for path in equivalence_markers],
        "failures": failures,
    }


def compare_same_gpu_profiles(
    *,
    legacy_root: Path,
    optimized_root: Path,
    equivalence_marker: Path,
) -> dict[str, Any]:
    marker = json.loads(equivalence_marker.read_text(encoding="utf-8"))
    legacy = json.loads(
        (legacy_root / "performance_profile.json").read_text(encoding="utf-8")
    )
    optimized = json.loads(
        (optimized_root / "performance_profile.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    if marker.get("status") != "PASS":
        failures.append("equivalence_marker_failed")
    legacy_seconds = float(legacy.get("random_walk_seconds") or 0.0)
    optimized_seconds = float(optimized.get("random_walk_seconds") or 0.0)
    if legacy_seconds <= 0.0 or optimized_seconds <= 0.0:
        failures.append("nonpositive_benchmark_duration")
        speedup = -1.0
    else:
        speedup = legacy_seconds / optimized_seconds - 1.0
    legacy_uuid = legacy.get("gpu_uuid")
    optimized_uuid = optimized.get("gpu_uuid")
    same_gpu = bool(legacy_uuid and legacy_uuid == optimized_uuid)
    if not same_gpu:
        failures.append("benchmark_not_same_physical_gpu")
    peak_fraction = max(
        float(legacy.get("peak_vram_fraction", 0.0)),
        float(optimized.get("peak_vram_fraction", 0.0)),
    )
    if peak_fraction > MAXIMUM_SHARED_VRAM_FRACTION:
        failures.append("peak_vram_exceeds_70_percent")
    if speedup < MINIMUM_AB_SPEEDUP_FRACTION:
        failures.append("speedup_below_20_percent")
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAILED",
        "legacy_root": str(legacy_root.resolve()),
        "optimized_root": str(optimized_root.resolve()),
        "equivalence_marker": str(equivalence_marker.resolve()),
        "gpu_uuid": legacy_uuid if same_gpu else None,
        "same_gpu_uuid": same_gpu,
        "legacy_random_walk_seconds": legacy_seconds,
        "optimized_random_walk_seconds": optimized_seconds,
        "speedup_fraction": speedup,
        "minimum_speedup_fraction": MINIMUM_AB_SPEEDUP_FRACTION,
        "peak_vram_fraction": peak_fraction,
        "maximum_vram_fraction": MAXIMUM_SHARED_VRAM_FRACTION,
        "mps_enabled": False,
        "failures": failures,
    }


def validate_full_acceleration_gate(
    path: Path,
    *,
    config: GCFAccelerationConfig,
    scientific_contract_sha256: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("GCF acceleration gate is not PASS")
    if payload.get("budgets") != list(EQUIVALENCE_BUDGETS):
        raise ValueError("GCF acceleration gate lacks exact 500/1000 evidence")
    if payload.get("optimized_config_fingerprint") != config.fingerprint:
        raise ValueError("GCF acceleration config differs from equivalence evidence")
    if (
        scientific_contract_sha256 is not None
        and payload.get("scientific_replay_contract_sha256")
        != scientific_contract_sha256
    ):
        raise ValueError("GCF scientific inputs differ from equivalence evidence")
    benchmark = payload.get("benchmark", {})
    if (
        benchmark.get("same_gpu_uuid") is not True
        or float(benchmark.get("speedup_fraction", -1.0))
        < MINIMUM_AB_SPEEDUP_FRACTION
        or float(benchmark.get("peak_vram_fraction", 1.0))
        > MAXIMUM_SHARED_VRAM_FRACTION
    ):
        raise ValueError("GCF acceleration benchmark gate is invalid")
    return payload


__all__ = [
    "EQUIVALENCE_BUDGETS",
    "BufferedVRRWLogging",
    "GCFAccelerationConfig",
    "LEGACY_ACCELERATION_MODE",
    "MAXIMUM_SHARED_VRAM_FRACTION",
    "MINIMUM_AB_SPEEDUP_FRACTION",
    "ORDERED_ACCELERATION_MODE",
    "OrderedLRU",
    "OrderedImportanceAcceleration",
    "OrderedNeighbourAcceleration",
    "VRRWPhaseProfiler",
    "build_acceleration_gate",
    "build_vrrw_equivalence_trace",
    "canonical_graph_tensor_digest",
    "compare_vrrw_equivalence",
    "compare_same_gpu_profiles",
    "ordered_parallel_map",
    "scientific_replay_contract_sha256",
    "validate_full_acceleration_gate",
    "write_fresh_json",
]
