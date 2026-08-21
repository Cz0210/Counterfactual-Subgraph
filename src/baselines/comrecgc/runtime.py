"""Execution bridge into the unmodified COMRECGC random walk."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .cache_trust import load_aids_tensor_payload
from .contracts import (
    CF_MODE,
    METHOD,
    RecourseParameters,
    UPSTREAM_COMMIT,
    GenerationParameters,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .model_adapter import (
    AIDSGreedEmbeddingAdapter,
    load_aids_gnn,
    load_bace_gnn,
    load_mutagenicity_gnn,
)
from .graph_trace import (
    ActionTraceRecorder,
    assert_trace_parity,
    iter_selected_trace,
    stable_graph_sha256,
)
from .generation_checkpoint import (
    GenerationCheckpointError,
    load_generation_checkpoint,
    mirror_generation_checkpoint,
    prune_mirrored_generation_checkpoints,
    restore_rng_state,
    restore_sqlite_snapshot,
    save_generation_checkpoint,
    scientific_command_sha256,
)
from .generation_loop import (
    GenerationLoopState,
    restore_official_state,
    run_generation_loop,
    save_official_payload,
    snapshot_official_state,
)
from .frozen_payload import (
    build_frozen_payload_closure,
    payload_file_audit,
)
from .live_graph_state import (
    GRAPH_STATE_POLICY,
    LiveGraphState,
    current_rss_mib,
)
from .storage_guard import StorageGuard, StorageGuardConfig
from .project_dataset import (
    GraphListDataset,
    ProjectDatasetBundle,
    load_aids_generation_bundle,
    load_bace_generation_bundle,
    load_mutagenicity_generation_bundle,
)
from .transition_cache import (
    COMPACT_TRANSITION_CACHE_PATCH,
    CompactMoveScopedTransitionMap,
)
from .upstream import imported_upstream

OFFICIAL_RUNTIME_PATCHES = (
    "project_dataset_injection_v1",
    "project_internal_label_mapping_v1",
    "bounded_batch_call_oom_safe_v1",
    "source_graph_lineage_v1",
    "candidate_map_unmaterialized_eviction_none_safe_v1",
)

ACTIVE_MOVE_TRANSITION_PATCH = "active_move_transition_eviction_deferred_v1"
LIVE_GRAPH_STATE_PATCH = "live_graph_authoritative_backing_resolution_v2"
PROJECT_OWNED_LOOP_PATCH = "project_owned_completed_step_checkpoint_loop_v1"


@dataclass(frozen=True, slots=True)
class PatchedRuntimeHandles:
    live_graph_state: LiveGraphState | None
    transition_map: Any | None


class _EndpointSafeGraphMap(dict[Any, Any]):
    """Make only the official unmaterialized-tail eviction idempotent."""

    def __init__(self, module: Any, values: Mapping[Any, Any]) -> None:
        super().__init__(values)
        self._module = module
        self.missing_unmaterialized_eviction_count = 0

    def __delitem__(self, key: Any) -> None:
        if key in self:
            super().__delitem__(key)
            return
        candidates = getattr(self._module, "counterfactual_candidates", ())
        tail_hash = candidates[-1].get("graph_hash") if candidates else None
        index_map = getattr(self._module, "graph_index_map", {})
        if key != tail_hash or key in index_map:
            raise KeyError(key)
        self.missing_unmaterialized_eviction_count += 1

    def __reduce__(self) -> tuple[Any, tuple[dict[Any, Any]]]:
        # Official torch.save artifacts must contain a plain dict.
        return dict, (dict(self),)


class _MoveScopedTransitionMap(dict[Any, Any]):
    """Keep current-head transitions alive until one official move completes.

    Upstream candidate-capacity eviction can remove a non-lead head after all
    head transitions have been built but before followers consume them.  The
    deferred deletion is applied immediately after the move, so cache capacity
    and subsequent algorithm behavior stay unchanged.
    """

    def __init__(
        self,
        module: Any,
        values: Mapping[Any, Any],
        *,
        seed: int,
        live_graph_state: LiveGraphState | None = None,
    ) -> None:
        super().__init__(values)
        self._module = module
        self._seed = int(seed)
        self._active_keys: tuple[Any, ...] = ()
        self._deferred_deletions: set[Any] = set()
        self._current_step = 0
        self.move_count = 0
        self.deferred_deletion_count = 0
        self.applied_deferred_deletion_count = 0
        self.cancelled_deferred_deletion_count = 0
        self.missing_lookup_count = 0
        self.max_transition_size = len(self)
        self._live_graph_state = live_graph_state

    def begin_move(self, graph_hashes: Sequence[Any]) -> int:
        if self._active_keys or self._deferred_deletions:
            raise RuntimeError("COMRECGC transition move scopes cannot be nested.")
        self.move_count += 1
        self._current_step = self.move_count
        self._active_keys = tuple(graph_hashes)
        if self._live_graph_state is not None:
            self._live_graph_state.graph_map.begin_move(
                graph_hashes, current_step=self._current_step
            )
        return self._current_step

    def end_move(self) -> None:
        index_map = getattr(self._module, "graph_index_map", {})
        for key in tuple(self._deferred_deletions):
            if key in index_map:
                self.cancelled_deferred_deletion_count += 1
            elif key in self:
                super().__delitem__(key)
                self.applied_deferred_deletion_count += 1
        self._deferred_deletions.clear()
        self._active_keys = ()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.max_transition_size = max(self.max_transition_size, len(self))

    def __delitem__(self, key: Any) -> None:
        if key in self._active_keys and key in self:
            if key not in self._deferred_deletions:
                self.deferred_deletion_count += 1
            self._deferred_deletions.add(key)
            return
        super().__delitem__(key)

    def __getitem__(self, key: Any) -> Any:
        if key not in self:
            self.missing_lookup_count += 1
            head = self._active_keys.index(key) if key in self._active_keys else None
            graph_map = getattr(self._module, "graph_map", {})
            raise RuntimeError(
                "[COMRECGC_TRANSITION_STATE_ERROR] "
                f"current_step={self._current_step} head={head} seed={self._seed} "
                f"graph_hash={key} transition_size={len(self)} "
                f"cache_size={len(graph_map)} active_head_count={len(self._active_keys)}"
            )
        return super().__getitem__(key)

    def wrap_move(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            graph_hashes = list(kwargs.get("graphs_hash", args[0] if args else []))
            current_step = self.begin_move(graph_hashes)
            try:
                return original(*args, **kwargs)
            finally:
                self.end_move()
                report_steps = {1_000, 45_000, 46_000, 46_600, 47_000, 48_000, 50_000}
                if current_step in report_steps or current_step % 1_000 == 0:
                    live_map = (
                        self._live_graph_state.graph_map
                        if self._live_graph_state is not None
                        else None
                    )
                    rss_mib = current_rss_mib()
                    print(
                        "[COMRECGC_TRANSITION_STATE] "
                        f"current_step={current_step} seed={self._seed} "
                        f"transition_size={len(self)} "
                        f"cache_size={len(getattr(self._module, 'graph_map', {}))} "
                        f"deferred_deletions={self.deferred_deletion_count} "
                        f"missing_lookups={self.missing_lookup_count} "
                        f"backing_store_size={live_map.store.count() if live_map else 0} "
                        f"pins={sum(live_map.pin_counts.values()) if live_map else 0} "
                        f"graph_deferred_deletions={len(live_map.deferred_deletions) if live_map else 0} "
                        f"eviction_committed={live_map.eviction_committed if live_map else 0} "
                        f"rehydrations={live_map.rehydrations if live_map else 0} "
                        f"unresolved_graph_lookups={live_map.unresolved_lookups if live_map else 0} "
                        f"max_rss_mib={rss_mib if rss_mib is not None else 'unavailable'}",
                        flush=True,
                    )
                    if current_step in {45_000, 46_000, 46_600, 47_000, 48_000, 50_000}:
                        print(
                            "[COMRECGC_GRAPH_STATE_INTEGRITY_SNAPSHOT] "
                            f"current_step={current_step} "
                            f"all_current_hashes_resolvable="
                            f"{all(live_map.contains_resolvable(key) for key in self._active_keys) if live_map else True} "
                            f"recent_eviction={live_map.recent_evictions[-1] if live_map and live_map.recent_evictions else None}",
                            flush=True,
                        )

        return wrapped

    def audit(self) -> dict[str, Any]:
        return {
            "patch": ACTIVE_MOVE_TRANSITION_PATCH,
            "policy": "defer_current_head_eviction_until_move_complete",
            "move_count": self.move_count,
            "deferred_deletion_count": self.deferred_deletion_count,
            "applied_deferred_deletion_count": self.applied_deferred_deletion_count,
            "cancelled_deferred_deletion_count": self.cancelled_deferred_deletion_count,
            "missing_lookup_count": self.missing_lookup_count,
            "max_transition_size": self.max_transition_size,
            "rng_calls_added": 0,
            "candidate_order_changed": False,
            "scientific_parameters_changed": False,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[Any, Any]]]:
        return dict, (dict(self),)


def _torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        from torch_geometric.data import Batch
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("COMRECGC runtime requires torch and torch_geometric.") from exc
    return torch, Batch


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _torch_load(path: Path) -> Any:
    torch, _Batch = _torch_stack()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save_atomic(payload: Any, path: Path) -> None:
    torch, _Batch = _torch_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_native_aids_gnn_from_trusted_features(
    *,
    gnn_module: Any,
    upstream_root: str | Path,
    num_features: int,
    device: str,
) -> Any:
    """Load the pinned official GNN without reopening the trusted PyG cache."""

    torch, _Batch = _torch_stack()
    model = gnn_module.GNN(
        num_features=int(num_features),
        num_classes=2,
        num_layers=3,
        dim=20,
        dropout=0.0,
    ).to(device)
    checkpoint = (
        Path(upstream_root).expanduser().resolve()
        / "data/aids/gnn/model_best.pth"
    )
    try:
        state_dict = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - older pinned torch compatibility
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


def validate_counterfactual_payload(payload: Any) -> tuple[dict[Any, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise RuntimeError("COMRECGC counterfactual payload must be a dictionary.")
    graph_map = payload.get("graph_map")
    candidates = payload.get("counterfactual_candidates")
    if not isinstance(graph_map, dict) or not graph_map:
        raise RuntimeError("COMRECGC counterfactual payload has no graph_map.")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("COMRECGC counterfactual payload has no candidates.")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("COMRECGC candidate records must be dictionaries.")
    return graph_map, candidates


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _materialize_dataset_indices(dataset: Any, indices: Sequence[int]) -> list[Any]:
    """Load lazy PyG rows before changing away from the dataset's cwd."""

    return [dataset[int(index)] for index in indices]


def _apply_neighbor_with_lineage(
    original: Callable[[Any, tuple[Any, ...]], Any],
    graph: Any,
    action: tuple[Any, ...],
) -> Any:
    """Apply one official edit while preserving project source-node lineage."""

    torch, _Batch = _torch_stack()
    result = original(graph, action)
    origins = _as_list(getattr(graph, "comrecgc_node_origin"))
    action_name = str(action[0])
    if action_name in {"NA", "INA"}:
        origins.append(-1)
    elif action_name in {"NR", "INR"}:
        remove_index = int(action[1])
        origins = [value for index, value in enumerate(origins) if index != remove_index]
    if len(origins) != int(result.num_nodes):
        raise RuntimeError(
            "COMRECGC lineage length changed independently of graph nodes: "
            f"action={action_name}, origins={len(origins)}, nodes={int(result.num_nodes)}"
        )
    result.comrecgc_node_origin = torch.tensor(origins, dtype=torch.long)
    for name in (
        "comrecgc_parent_id",
        "comrecgc_source_index",
        "comrecgc_source_smiles",
        "comrecgc_project_label",
        "comrecgc_source_record",
    ):
        if hasattr(graph, name):
            setattr(result, name, getattr(graph, name))
    return result


def lineage_neighbor_wrapper(
    original: Callable[[Any, tuple[Any, ...]], Any],
    *,
    trace_recorder: ActionTraceRecorder | None = None,
    transition_cache: CompactMoveScopedTransitionMap | None = None,
) -> Callable[[Any, tuple[Any, ...]], Any]:
    """Preserve source-node lineage without changing graph tensors."""

    def wrapped(graph: Any, action: tuple[Any, ...]) -> Any:
        result = _apply_neighbor_with_lineage(original, graph, action)
        if transition_cache is not None:
            transition_cache.record_enumerated(result, action)
        if trace_recorder is not None:
            trace_recorder.record_enumerated(
                source_graph=graph,
                target_graph=result,
                action=action,
            )
        return result

    return wrapped


def reset_official_state(module: Any, *, candidate_capacity: int, sample_size: int) -> None:
    torch, _Batch = _torch_stack()
    module.MAX_COUNTERFACTUAL_SIZE = int(candidate_capacity)
    module.graph_map = {}
    module.graph_index_map = {}
    module.counterfactual_candidates = []
    module.input_graphs_covered = torch.zeros(0, dtype=torch.float32)
    module.covering_graphs = set()
    module.transitions = {}
    module.start = {}
    module.is_sample = True
    module.starting_step = 1
    module.traversed_hashes = []
    module.sample_size = int(sample_size)


def _safe_call_factory(
    *,
    model: Any,
    embedding_model: Any,
    gnn_device: str,
    embedding_device: str,
    batch_size: int,
) -> Callable[[Sequence[Any], Mapping[str, Any]], tuple[np.ndarray, np.ndarray]]:
    torch, Batch = _torch_stack()

    def safe_call(graphs: Sequence[Any], _unused: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        values = list(graphs)
        predictions: list[Any] = []
        embeddings: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(values), max(1, int(batch_size))):
                chunk = values[start : start + max(1, int(batch_size))]
                gnn_batch = Batch.from_data_list(chunk).to(gnn_device)
                log_probs = model(gnn_batch)[-1]
                predictions.append(torch.exp(log_probs)[:, 1].detach().cpu())
                embedding_batch = Batch.from_data_list(chunk).to(embedding_device)
                embeddings.append(embedding_model.embed_model(embedding_batch).detach().cpu())
        if not predictions:
            return np.empty((0, 2), dtype=float), np.empty((0, 0), dtype=float)
        prediction_array = torch.cat(predictions).numpy()
        embedding_array = torch.cat(embeddings).numpy()
        if not np.isfinite(prediction_array).all() or not np.isfinite(embedding_array).all():
            raise RuntimeError("COMRECGC model call produced NaN/Inf.")
        coverage = np.ones_like(prediction_array)
        return np.stack([prediction_array, coverage], axis=1), embedding_array

    return safe_call


@contextmanager
def patched_official_runtime(
    module: Any,
    *,
    model: Any,
    embedding_model: Any,
    gnn_device: str,
    embedding_device: str,
    batch_size: int,
    trace_recorder: ActionTraceRecorder | None = None,
    compatibility_audit: dict[str, Any] | None = None,
    preserve_active_transitions: bool = False,
    compact_transitions: bool = False,
    transition_expanded_capacity: int = 5,
    seed: int = 0,
    graph_state_dir: str | Path | None = None,
    graph_store_path: str | Path | None = None,
    storage_guard: StorageGuard | None = None,
) -> Iterator[PatchedRuntimeHandles]:
    originals = {
        "call": module.call,
        "neighbor_graph_access": module.neighbor_graph_access,
        "move_to_next_graph": module.move_to_next_graph,
    }
    original_neighbor = module.neighbor_graph_access
    live_graph_state = (
        LiveGraphState(
            module,
            module.graph_map,
            store_path=(
                Path(graph_store_path).expanduser().resolve()
                if graph_store_path is not None
                else Path(graph_state_dir).expanduser().resolve()
                / "authoritative_graph_store.sqlite3"
            ),
            seed=seed,
            # Storage checks run from the project-owned outer loop after
            # teleport/restart and both upstream assertions, never here.
            on_step=None,
        )
        if preserve_active_transitions and graph_state_dir is not None
        else None
    )
    endpoint_safe_graph_map = (
        live_graph_state.graph_map
        if live_graph_state is not None
        else _EndpointSafeGraphMap(module, module.graph_map)
    )
    module.graph_map = endpoint_safe_graph_map
    if live_graph_state is not None:
        module.comrecgc_live_graph_state = live_graph_state
    if compact_transitions:
        transition_map: _MoveScopedTransitionMap | CompactMoveScopedTransitionMap | None = (
            CompactMoveScopedTransitionMap(
                module,
                module.transitions,
                seed=seed,
                expanded_capacity=transition_expanded_capacity,
                rebuild_target=lambda graph, action: _apply_neighbor_with_lineage(
                    original_neighbor, graph, action
                ),
            )
        )
    else:
        transition_map = (
            _MoveScopedTransitionMap(
                module,
                module.transitions,
                seed=seed,
                live_graph_state=live_graph_state,
            )
            if preserve_active_transitions
            else None
        )
    if transition_map is not None:
        module.transitions = transition_map
    module.call = _safe_call_factory(
        model=model,
        embedding_model=embedding_model,
        gnn_device=gnn_device,
        embedding_device=embedding_device,
        batch_size=batch_size,
    )
    module.neighbor_graph_access = lineage_neighbor_wrapper(
        original_neighbor,
        trace_recorder=trace_recorder,
        transition_cache=(
            transition_map
            if isinstance(transition_map, CompactMoveScopedTransitionMap)
            else None
        ),
    )
    patched_move = module.move_to_next_graph
    if trace_recorder is not None:
        patched_move = trace_recorder.wrap_move(patched_move, module)
    if transition_map is not None:
        # Keep the transition scope outside trace capture so selected-action
        # resolution can still inspect the transition before deferred cleanup.
        patched_move = transition_map.wrap_move(patched_move)
    if live_graph_state is not None:
        # Outermost scope: pins cover trace reads, the official move,
        # transition updates, streamed trace writes, and deferred cleanup.
        patched_move = live_graph_state.wrap_move(patched_move)
    module.move_to_next_graph = patched_move
    try:
        yield PatchedRuntimeHandles(
            live_graph_state=live_graph_state,
            transition_map=transition_map,
        )
    finally:
        try:
            if compatibility_audit is not None:
                compatibility_audit.update(
                    {
                        "patch": "candidate_map_unmaterialized_eviction_none_safe_v1",
                        "missing_unmaterialized_eviction_count": int(
                            endpoint_safe_graph_map.missing_unmaterialized_eviction_count
                        ),
                        "rng_calls_added": 0,
                        "candidate_order_changed": False,
                    }
                )
                if transition_map is not None:
                    compatibility_audit["transition_state"] = transition_map.audit()
                if live_graph_state is not None:
                    compatibility_audit["live_graph_state"] = live_graph_state.audit()
        finally:
            module.graph_map = dict(endpoint_safe_graph_map)
            if transition_map is not None:
                module.transitions = (
                    {}
                    if isinstance(transition_map, CompactMoveScopedTransitionMap)
                    else dict(transition_map)
                )
            module.call = originals["call"]
            module.neighbor_graph_access = originals["neighbor_graph_access"]
            module.move_to_next_graph = originals["move_to_next_graph"]
            if live_graph_state is not None:
                delattr(module, "comrecgc_live_graph_state")
                live_graph_state.close()


def _predict_internal(model: Any, graphs: Sequence[Any], *, device: str, batch_size: int = 128) -> list[int]:
    torch, Batch = _torch_stack()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(list(graphs[start : start + batch_size])).to(device)
            predictions.extend(
                int(value) for value in model(batch)[-1].argmax(dim=-1).detach().cpu().tolist()
            )
    return predictions


def _materialize_official_result(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite result: {destination}")
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        mode = "atomic_copy"
    if sha256_file(source) != sha256_file(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Materialized counterfactual artifact failed SHA256 verification.")
    return mode


def _load_bundle(
    dataset: str,
    *,
    dataset_dir: Path,
    source_csv: Path | None,
    parent_limit: int,
) -> ProjectDatasetBundle:
    if dataset == "aids":
        if source_csv is None:
            raise ValueError("AIDS project adaptation requires --source-csv.")
        return load_aids_generation_bundle(
            dataset_dir=dataset_dir,
            source_csv=source_csv,
            parent_limit=parent_limit,
        )
    if dataset == "mutagenicity":
        return load_mutagenicity_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    if dataset == "bace":
        return load_bace_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    raise ValueError(f"Unsupported project dataset: {dataset}")


_RUNTIME_CHECKPOINT_STATE_SCHEMA = "comrecgc_bace_runtime_checkpoint_v1"
_RESOLVED_CONFIG_BINDING_SCHEMA = "comrecgc_resolved_config_binding_v1"
_RESOLVED_CONFIG_FILENAME = "resolved_config.json"
_RESOLVED_CONFIG_BINDING_FILENAME = "resolved_config.binding.json"


def _runtime_environment(torch: Any) -> dict[str, Any]:
    try:
        import torch_geometric

        pyg_version = str(torch_geometric.__version__)
    except Exception:
        pyg_version = "unavailable"
    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": pyg_version,
        "torch_cuda_runtime": str(torch.version.cuda),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_device_count": cuda_count,
        "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(cuda_count)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _runtime_checkpoint_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    """Build stable fingerprints; timestamps and output paths are excluded."""

    environment = config["runtime_environment"]
    compatible_environment = {
        key: environment[key]
        for key in (
            "python_version",
            "python_implementation",
            "pythonhashseed",
            "numpy_version",
            "torch_version",
            "torch_geometric_version",
            "torch_cuda_runtime",
            "cudnn_version",
            "cuda_device_count",
            "cuda_device_names",
        )
    }
    return {
        "project_commit": str(config["project_commit"]),
        "upstream_commit": str(config["upstream_commit"]),
        "dataset": str(config["dataset"]),
        "mode": str(config["mode"]),
        "generation_parent_ids_sha256": str(
            config["generation_parent_ids_sha256"]
        ),
        "parameters_sha256": stable_json_sha256(config["parameters"]),
        "dataset_fingerprint": str(config["dataset_audit"]["dataset_fingerprint"]),
        "dataset_contract_sha256": stable_json_sha256(
            {
                key: config["dataset_audit"][key]
                for key in (
                    "dataset",
                    "dataset_source",
                    "num_graphs",
                    "num_label0",
                    "num_label1",
                    "node_feature_dim",
                    "edge_feature_dim",
                    "atom_types",
                    "label_semantics",
                    "generation_source_parent_rows",
                    "generation_parent_ids_sha256",
                )
            }
        ),
        "gnn_checkpoint_sha256": str(config["gnn"]["checkpoint_sha256"]),
        "distance_checkpoint_sha256": str(
            config["distance_model"]["checkpoint_sha256"]
        ),
        "runtime_patch": PROJECT_OWNED_LOOP_PATCH,
        "runtime_environment_sha256": stable_json_sha256(compatible_environment),
        "scientific_command_sha256": str(config["command_sha256"]),
        "total_steps": str(int(config["total_steps"])),
    }


def _resolved_config_content_sha256(config: Mapping[str, Any]) -> str:
    return stable_json_sha256(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )


def _validate_resolved_config_binding(
    config: Mapping[str, Any],
    *,
    expected_scientific_argv: Sequence[str],
    expected_command_sha256: str,
    expected_total_steps: int,
) -> None:
    argv = tuple(str(value) for value in config.get("scientific_argv") or ())
    if argv != tuple(expected_scientific_argv):
        raise GenerationCheckpointError(
            "Resolved config scientific argv differs from the current runtime."
        )
    if config.get("command_sha256") != expected_command_sha256:
        raise GenerationCheckpointError(
            "Resolved config command SHA256 differs from the current runtime."
        )
    if scientific_command_sha256(argv) != expected_command_sha256:
        raise GenerationCheckpointError(
            "Resolved config command SHA256 does not match canonical argv."
        )
    if int(config.get("total_steps", -1)) != int(expected_total_steps):
        raise GenerationCheckpointError(
            "Resolved config total_steps differs from the current runtime."
        )
    expected_config_sha256 = _resolved_config_content_sha256(config)
    if config.get("config_sha256") != expected_config_sha256:
        raise GenerationCheckpointError("Resolved config content SHA256 mismatch.")


def _persistent_resolved_config_paths(mirror_root: Path) -> tuple[Path, Path]:
    parent = mirror_root.parent
    return (
        parent / _RESOLVED_CONFIG_FILENAME,
        parent / _RESOLVED_CONFIG_BINDING_FILENAME,
    )


def _publish_persistent_resolved_config(
    config: Mapping[str, Any],
    *,
    mirror_root: Path,
) -> tuple[Path, Path]:
    """Atomically publish resume metadata before any checkpoint can be mirrored."""

    config_path, binding_path = _persistent_resolved_config_paths(mirror_root)
    command_sha256 = str(config["command_sha256"])
    config_sha256 = str(config["config_sha256"])
    binding = {
        "schema_version": _RESOLVED_CONFIG_BINDING_SCHEMA,
        "resolved_config": _RESOLVED_CONFIG_FILENAME,
        "resolved_config_sha256": None,
        "config_sha256": config_sha256,
        "scientific_argv": list(config["scientific_argv"]),
        "command_sha256": command_sha256,
        "total_steps": int(config["total_steps"]),
    }
    if config_path.is_symlink():
        raise GenerationCheckpointError(
            "Persistent resolved_config path must not be a symbolic link."
        )
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != dict(config):
            raise GenerationCheckpointError(
                "Persistent resolved_config differs from the current run."
            )
    elif config_path.exists() or config_path.is_symlink():
        raise GenerationCheckpointError(
            "Persistent resolved_config path is not a physical file."
        )
    else:
        write_json(config_path, dict(config))
    binding["resolved_config_sha256"] = sha256_file(config_path)
    if binding_path.is_symlink():
        raise GenerationCheckpointError(
            "Persistent resolved_config binding path must not be a symbolic link."
        )
    if binding_path.is_file():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding != binding:
            raise GenerationCheckpointError(
                "Persistent resolved_config binding differs from the current run."
            )
    elif binding_path.exists() or binding_path.is_symlink():
        raise GenerationCheckpointError(
            "Persistent resolved_config binding path is not a physical file."
        )
    else:
        write_json(binding_path, binding)
    # The formal AutoDL stage historically restores from
    # ``generation_resume_metadata/resolved_config.json`` while the profile
    # stage restores from the checkpoint mirror parent. Publish both aliases
    # before generation can reach checkpoint 1; both are physical, atomically
    # written files with an adjacent binding, never symlinks.
    compatibility_root = mirror_root.parent / "generation_resume_metadata"
    compatibility_config = compatibility_root / _RESOLVED_CONFIG_FILENAME
    compatibility_binding = compatibility_root / _RESOLVED_CONFIG_BINDING_FILENAME
    for destination, payload in (
        (compatibility_config, dict(config)),
        (compatibility_binding, binding),
    ):
        if destination.is_symlink():
            raise GenerationCheckpointError(
                "Persistent resolved_config compatibility path is a symbolic link."
            )
        if destination.is_file():
            existing_payload = json.loads(destination.read_text(encoding="utf-8"))
            if existing_payload != payload:
                raise GenerationCheckpointError(
                    "Persistent resolved_config compatibility copy differs."
                )
        elif destination.exists():
            raise GenerationCheckpointError(
                "Persistent resolved_config compatibility path is not a file."
            )
        else:
            write_json(destination, payload)
    return config_path, binding_path


def _load_persistent_resolved_config(
    *,
    mirror_root: Path,
    expected_scientific_argv: Sequence[str],
    expected_command_sha256: str,
    expected_total_steps: int,
) -> dict[str, Any]:
    config_path, binding_path = _persistent_resolved_config_paths(mirror_root)
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or not binding_path.is_file()
        or binding_path.is_symlink()
    ):
        raise GenerationCheckpointError(
            "Persistent checkpoint mirror has no complete resolved_config metadata."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(binding, dict):
        raise GenerationCheckpointError("Persistent resolved_config metadata is invalid.")
    _validate_resolved_config_binding(
        config,
        expected_scientific_argv=expected_scientific_argv,
        expected_command_sha256=expected_command_sha256,
        expected_total_steps=expected_total_steps,
    )
    expected_binding = {
        "schema_version": _RESOLVED_CONFIG_BINDING_SCHEMA,
        "resolved_config": _RESOLVED_CONFIG_FILENAME,
        "resolved_config_sha256": sha256_file(config_path),
        "config_sha256": config["config_sha256"],
        "scientific_argv": list(expected_scientific_argv),
        "command_sha256": expected_command_sha256,
        "total_steps": int(expected_total_steps),
    }
    if binding != expected_binding:
        raise GenerationCheckpointError("Persistent resolved_config binding mismatch.")
    return config


def _checkpoint_algorithm_state(
    module: Any,
    *,
    loop_state: GenerationLoopState,
    handles: PatchedRuntimeHandles,
) -> dict[str, Any]:
    if not isinstance(handles.transition_map, CompactMoveScopedTransitionMap):
        raise GenerationCheckpointError(
            "Exact full-generation checkpoints require compact transition storage."
        )
    if handles.live_graph_state is None:
        raise GenerationCheckpointError(
            "Exact full-generation checkpoints require authoritative live graph state."
        )
    transition_state = handles.transition_map.export_checkpoint_state()
    live_graph_state = handles.live_graph_state.export_checkpoint_state()
    required_hashes = {
        *loop_state.start_graph_hashes,
        *loop_state.current_graph_hashes,
        *module.graph_index_map.keys(),
        *module.covering_graphs,
        *(row.get("graph_hash") for row in module.counterfactual_candidates),
        *(row["source_hash"] for row in transition_state["entries"]),
    }
    missing = [
        value
        for value in required_hashes
        if value is not None and not handles.live_graph_state.contains(value)
    ]
    if missing:
        raise GenerationCheckpointError(
            "COMRECGC checkpoint graph closure is incomplete: "
            f"missing_count={len(missing)}, first={missing[0]!r}."
        )
    return {
        "schema_version": _RUNTIME_CHECKPOINT_STATE_SCHEMA,
        "loop_state": loop_state.to_checkpoint_state(),
        "official_state": snapshot_official_state(module),
        "transition_state": transition_state,
        "live_graph_state": live_graph_state,
    }


def _restore_runtime_checkpoint_state(
    module: Any,
    *,
    algorithm_state: Mapping[str, Any],
    trace_state: Mapping[str, Any],
    handles: PatchedRuntimeHandles,
    trace_recorder: ActionTraceRecorder | None,
) -> GenerationLoopState:
    if algorithm_state.get("schema_version") != _RUNTIME_CHECKPOINT_STATE_SCHEMA:
        raise GenerationCheckpointError(
            "Unsupported COMRECGC runtime checkpoint state schema."
        )
    if not isinstance(handles.transition_map, CompactMoveScopedTransitionMap):
        raise GenerationCheckpointError(
            "Checkpoint restore requires compact transition storage."
        )
    if handles.live_graph_state is None:
        raise GenerationCheckpointError(
            "Checkpoint restore requires authoritative live graph state."
        )
    handles.transition_map.restore_checkpoint_state(
        algorithm_state["transition_state"]
    )
    handles.live_graph_state.restore_checkpoint_state(
        algorithm_state["live_graph_state"]
    )
    if trace_recorder is None:
        if trace_state.get("enabled") is not False:
            raise GenerationCheckpointError(
                "Checkpoint contains trace state but tracing is disabled."
            )
    else:
        trace_recorder.restore_checkpoint_state(trace_state)
    state = GenerationLoopState.from_checkpoint_state(
        algorithm_state["loop_state"]
    )
    if handles.live_graph_state.move_count != state.completed_step:
        raise GenerationCheckpointError(
            "Checkpoint walk step differs from live graph move count."
        )
    if handles.transition_map.move_count != state.completed_step:
        raise GenerationCheckpointError(
            "Checkpoint walk step differs from transition move count."
        )
    if trace_recorder is not None and trace_recorder.move_index != state.completed_step:
        raise GenerationCheckpointError(
            "Checkpoint walk step differs from trace move count."
        )
    return state


def _progress_payload(
    *,
    current_step: int,
    max_steps: int,
    config_sha256: str,
    run_complete: bool,
    checkpoint_dir: str | None = None,
    last_checkpoint_step: int | None = None,
    started_monotonic: float,
    process_start_step: int,
    gpu_id: str,
    code_commit: str,
) -> dict[str, Any]:
    completed = int(current_step)
    total = int(max_steps)
    elapsed = max(time.monotonic() - float(started_monotonic), 0.0)
    process_steps = max(completed - int(process_start_step), 0)
    steps_per_hour = (
        float(process_steps * 3600.0 / elapsed) if elapsed > 0.0 else 0.0
    )
    heartbeat_at = datetime.now(timezone.utc).isoformat()
    return {
        "stage": "generation",
        "current_step": completed,
        "max_steps": total,
        "completed_step": completed,
        "next_step": min(completed + 1, total + 1),
        "total_steps": total,
        "steps_per_hour": steps_per_hour,
        "elapsed_seconds": elapsed,
        "last_checkpoint_step": (
            int(last_checkpoint_step) if last_checkpoint_step is not None else None
        ),
        "latest_checkpoint": checkpoint_dir,
        "heartbeat_at": heartbeat_at,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "gpu_id": str(gpu_id),
        "code_commit": str(code_commit),
        "run_complete": bool(run_complete),
        "config_sha256": str(config_sha256),
        "checkpoint_dir": checkpoint_dir,
        "updated_at": heartbeat_at,
    }


def model_counterfactual_graphs(
    payload: Mapping[str, Any], *, limit: int
) -> list[Any]:
    """Resolve actual model-counterfactual graphs in official candidate order."""

    from .recourse import _importance_parts

    graph_map, candidates = validate_counterfactual_payload(payload)
    resolved: list[Any] = []
    for candidate in candidates:
        importance = _importance_parts(candidate)
        graph_hash = candidate.get("graph_hash")
        if float(importance[0]) >= 0.5 and graph_hash in graph_map:
            resolved.append(graph_map[graph_hash][0])
        if len(resolved) >= int(limit):
            break
    if not resolved:
        raise RuntimeError("Native smoke has no model-counterfactual graph candidates.")
    return resolved


def _run_native_common_recourse(
    *,
    modules: Mapping[str, Any],
    sources: Sequence[Any],
    payload: Mapping[str, Any],
    embedding_model: Any,
    output_dir: Path,
    device: str,
    mode: str,
    allow_empty: bool,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Exercise official clustering/summary on real native random-walk output."""

    from sklearn.cluster import DBSCAN

    from .recourse import stable_graph_id, trace_official_cluster_order

    parameters = RecourseParameters.for_mode(mode)
    candidate_graphs = model_counterfactual_graphs(payload, limit=parameters.cf_size)
    torch, Batch = _torch_stack()
    with torch.no_grad():
        source_embeddings = embedding_model.embed_model(
            Batch.from_data_list(list(sources)).to(device)
        ).detach().cpu()
    embedding_model.embed_targets(list(sources))
    source_counts = modules["util"].graph_element_counts(list(sources)).cpu()
    pair_indices: list[tuple[int, int]] = []
    recourse_vectors: list[np.ndarray] = []
    distance_pair_count = 0
    for start in range(0, len(candidate_graphs), max(1, int(batch_size))):
        chunk = candidate_graphs[start : start + max(1, int(batch_size))]
        with torch.no_grad():
            distances = embedding_model.predict_outer_with_queries(
                chunk, batch_size=batch_size
            ).cpu()
            candidate_embeddings = embedding_model.embed_model(
                Batch.from_data_list(chunk).to(device)
            ).detach().cpu()
        candidate_counts = modules["util"].graph_element_counts(chunk).cpu()
        scale = candidate_counts[:, None] + source_counts[None, :]
        normalized = distances / scale
        valid_pairs = torch.nonzero(
            normalized <= float(parameters.theta), as_tuple=False
        )
        distance_pair_count += int(normalized.numel())
        for local_candidate, source_index in valid_pairs.tolist():
            candidate_index = start + int(local_candidate)
            pair_indices.append((int(source_index), candidate_index))
            vector = (
                candidate_embeddings[int(local_candidate)]
                - source_embeddings[int(source_index)]
            ) / scale[int(local_candidate), int(source_index)]
            recourse_vectors.append(vector.numpy())
    if recourse_vectors:
        recourse_array = np.asarray(recourse_vectors)
        if not np.isfinite(recourse_array).all():
            raise RuntimeError("Native common-recourse embeddings contain NaN/Inf.")
        clustering = DBSCAN(
            eps=float(parameters.delta), min_samples=int(parameters.cluster_size)
        ).fit(recourse_array)
        official_result = modules["common_recourse"].coverage_summary(
            db_2=clustering,
            rec=torch.tensor(recourse_array),
            idxs=pair_indices,
            radius=float(parameters.delta),
            threshold_theta=float(parameters.theta),
            recourse_size=int(parameters.recourse_size),
        )
        selected = trace_official_cluster_order(
            labels=np.asarray(clustering.labels_),
            recourse_vectors=recourse_array,
            pair_indices=pair_indices,
            radius=float(parameters.delta),
            theta=float(parameters.theta),
            recourse_size=int(parameters.recourse_size),
            official_greedy=modules[
                "common_recourse"
            ].greedy_counterfactual_summary_from_covering_sets,
        )
        labels = np.asarray(clustering.labels_)
    else:
        recourse_array = np.empty((0, 0), dtype=float)
        official_result = ([], [], [])
        selected = []
        labels = np.asarray([], dtype=int)
    diagnostics = {
        "model_counterfactual_candidate_count": len(candidate_graphs),
        "distance_pair_count": distance_pair_count,
        "theta_eligible_pair_count": len(pair_indices),
        "dbscan_cluster_count": len({int(value) for value in labels if int(value) >= 0}),
        "dbscan_noise_point_count": int(np.count_nonzero(labels < 0)),
        "official_coverage_summary_invoked": True,
        "official_coverage_summary_result": [list(value) for value in official_result],
        "selected_common_recourse_count": len(selected),
    }
    if not selected and not allow_empty:
        write_json(output_dir / "native_common_recourse_failure.json", diagnostics)
        raise RuntimeError("Native smoke common-recourse summary is empty.")
    representatives = [
        candidate_graphs[int(row["representative_counterfactual_index"])]
        for row in selected
    ]
    representative_path = output_dir / "native_representative_counterfactuals.pt"
    _torch_save_atomic(representatives, representative_path)
    reloaded = _torch_load(representative_path)
    if not isinstance(reloaded, list) or len(reloaded) != len(representatives):
        raise RuntimeError("Native common-recourse representatives are not reloadable.")
    rows = [
        {
            **row,
            "candidate_id": stable_graph_id(representatives[index]),
            "source_graph_id": str(
                getattr(sources[int(row["representative_source_index"])], "comrecgc_parent_id")
            ),
        }
        for index, row in enumerate(selected)
    ]
    summary = {
        "schema_version": 1,
        "route": "native_reproduction",
        "parameters": parameters.__dict__,
        **diagnostics,
        "common_recourse_count": len(rows),
        "scientific_output_empty": not bool(rows),
        "execution_status": (
            "EMPTY_COMMON_RECOURSE" if not rows else "FULL_EXECUTION_PASS"
        ),
        "native_cost": None if not rows else rows[-1]["native_cumulative_cost"],
        "official_greedy_order_preserved": True,
        "representative_policy": "real_pair_nearest_cluster_center",
        "representative_counterfactuals_path": str(representative_path),
        "representative_counterfactuals_sha256": sha256_file(representative_path),
        "serialization_reloadable": True,
        "no_nan_or_inf": True,
        "selected_common_recourses": rows,
        "run_complete": True,
    }
    write_json(output_dir / "native_common_recourse.json", summary)
    return summary


def run_project_generation(
    *,
    project_root: str | Path,
    upstream_root: str | Path,
    dataset: str,
    dataset_dir: str | Path,
    source_csv: str | Path | None,
    gnn_checkpoint: str | Path,
    distance_checkpoint: str | Path,
    output_dir: str | Path,
    mode: str,
    parent_limit: int,
    parameters: GenerationParameters,
    device: str = "cuda:0",
    batch_size: int = 128,
    resume: bool = False,
    trace_output_dir: str | Path | None = None,
    parity_reference_path: str | Path | None = None,
    graph_state_dir: str | Path | None = None,
    storage_guard_root: str | Path | None = None,
    storage_check_every_steps: int = 500,
    storage_min_free_bytes: int = 20 * 1024**3,
    storage_min_free_ratio: float = 0.05,
    storage_min_free_inodes: int = 100_000,
    checkpoint_root: str | Path | None = None,
    checkpoint_mirror_root: str | Path | None = None,
    checkpoint_interval_steps: int = 500,
    checkpoint_keep_last: int = 2,
    progress_interval_steps: int = 25,
    scientific_argv: Sequence[str] | None = None,
    command_sha256: str | None = None,
) -> dict[str, Any]:
    parameters.validate(mode)
    if int(checkpoint_interval_steps) <= 0:
        raise ValueError("checkpoint_interval_steps must be positive.")
    if not 10 <= int(progress_interval_steps) <= 50:
        raise ValueError("progress_interval_steps must be between 10 and 50.")
    if resume and mode != "full":
        raise ValueError("Exact COMRECGC resume is supported only for full generation.")
    if resume and checkpoint_root is None:
        raise ValueError("--resume requires --checkpoint-root.")
    if mode == "full" and dataset == "bace" and checkpoint_mirror_root is None:
        raise ValueError(
            "BACE full generation requires --checkpoint-mirror-root on persistent storage."
        )
    if int(checkpoint_keep_last) < 2:
        raise ValueError("checkpoint_keep_last must be at least two.")
    normalized_scientific_argv = tuple(str(value) for value in scientific_argv or ())
    if not normalized_scientific_argv:
        raise ValueError("Generation requires canonical redacted scientific argv.")
    normalized_command_sha256 = str(command_sha256 or "")
    if scientific_command_sha256(normalized_scientific_argv) != normalized_command_sha256:
        raise ValueError("Generation command SHA256 does not match scientific argv.")
    project = Path(project_root).expanduser().resolve()
    root = require_empty_output(output_dir, resume=resume)
    bundle = _load_bundle(
        dataset,
        dataset_dir=Path(dataset_dir).expanduser().resolve(),
        source_csv=Path(source_csv).expanduser().resolve() if source_csv else None,
        parent_limit=int(parent_limit),
    )
    if len(bundle.graphs) != int(parent_limit):
        raise RuntimeError(
            f"Project generation parent count mismatch: actual={len(bundle.graphs)}, "
            f"expected={int(parent_limit)}"
        )
    torch, _Batch = _torch_stack()
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("A CUDA device was requested but is not available.")
    if mode == "full" and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "Exact COMRECGC checkpoints require PYTHONHASHSEED=0 before Python starts."
        )
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parameters.seed)
    dataset_key = {
        "aids": "project_aids",
        "mutagenicity": "project_mutagenicity",
        "bace": "project_bace",
    }.get(dataset)
    if dataset_key is None:
        raise ValueError(f"Unsupported project dataset: {dataset}")
    runtime_root = root / "official_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    graph_state_root = (
        Path(graph_state_dir).expanduser().resolve()
        if graph_state_dir is not None
        else root / "graph_state"
    )
    if mode == "full":
        graph_state_root.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint_root = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root is not None
        else (root / "generation_checkpoints" if mode == "full" else None)
    )
    resolved_checkpoint_mirror_root = (
        Path(checkpoint_mirror_root).expanduser().resolve()
        if checkpoint_mirror_root is not None
        else None
    )
    active_graph_store_path = graph_state_root / "authoritative_graph_store.sqlite3"
    storage_guard: StorageGuard | None = None
    if mode == "full" and storage_guard_root is not None:
        guard_root = Path(storage_guard_root).expanduser().resolve()
        storage_guard = StorageGuard(
            StorageGuardConfig(
                root=guard_root,
                expected_steps=int(parameters.steps),
                check_every_steps=int(storage_check_every_steps),
                min_free_bytes=int(storage_min_free_bytes),
                min_free_ratio=float(storage_min_free_ratio),
                min_free_inodes=int(storage_min_free_inodes),
            ),
            database_path=graph_state_root / "authoritative_graph_store.sqlite3",
            exact_resume_supported=resolved_checkpoint_root is not None,
            generation_checkpoint_root=resolved_checkpoint_root,
        )
    started = datetime.now(timezone.utc).isoformat()
    process_started_monotonic = time.monotonic()
    process_start_step = 0
    progress_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES") or str(device)
    trace_recorder = (
        ActionTraceRecorder(
            output_dir=trace_output_dir,
            compact_enumeration=mode == "full",
        )
        if trace_output_dir is not None
        else None
    )
    compatibility_audit: dict[str, Any] = {}
    try:
        with imported_upstream(upstream_root) as modules:
            official = modules["comrecgc"]
            reset_official_state(
                official,
                candidate_capacity=parameters.candidate_capacity,
                sample_size=parameters.sample_size,
            )
            official.input_graphs_covered = torch.zeros(len(bundle.graphs), dtype=torch.float32)
            if dataset == "aids":
                model, model_provenance = load_aids_gnn(
                    gnn_checkpoint,
                    num_features=bundle.node_feature_dim,
                    device=device,
                )
                embedding_model = AIDSGreedEmbeddingAdapter(
                    distance_checkpoint,
                    atom_vocabulary=[str(value) for value in bundle.atom_vocabulary],
                    device=device,
                ).eval()
                distance_provenance = embedding_model.provenance()
            else:
                loader = (
                    load_bace_gnn if dataset == "bace" else load_mutagenicity_gnn
                )
                model, model_provenance = loader(
                    gnn_checkpoint,
                    num_features=bundle.node_feature_dim,
                    official_gnn_class=modules["gnn"].GNN,
                    device=device,
                )
                embedding_model = modules["distance"].load_neurosed(
                    bundle.graphs,
                    neurosed_model_path=str(Path(distance_checkpoint).expanduser().resolve()),
                    device=device,
                ).to(device).eval()
                distance_provenance = {
                    "checkpoint_path": str(Path(distance_checkpoint).expanduser().resolve()),
                    "checkpoint_sha256": sha256_file(distance_checkpoint),
                    "distance_model": f"{dataset}_neurosed",
                    "checkpoint_retrained": False,
                }
            predictions = _predict_internal(model, bundle.graphs, device=device)
            internal_counts = {str(key): int(value) for key, value in Counter(predictions).items()}
            config = {
                "schema_version": 1,
                "method": METHOD,
                "dataset": dataset,
                "route": "project_adapted",
                "mode": mode,
                "project_commit": _git_commit(project),
                "upstream_commit": UPSTREAM_COMMIT,
                "dataset_audit": bundle.audit(),
                "parent_limit": int(parent_limit),
                "generation_parent_ids": bundle.parent_ids,
                "generation_parent_ids_sha256": stable_json_sha256(bundle.parent_ids),
                "parameters": parameters.__dict__,
                "scientific_argv": list(normalized_scientific_argv),
                "command_sha256": normalized_command_sha256,
                "total_steps": int(parameters.steps),
                "gnn": model_provenance,
                "distance_model": distance_provenance,
                "internal_prediction_counts": internal_counts,
                "runtime_environment": _runtime_environment(torch),
                "cf_mode": CF_MODE,
                "calibration_loaded": False,
                "test_loaded": False,
                "official_compatibility_patches": [
                    *OFFICIAL_RUNTIME_PATCHES,
                    PROJECT_OWNED_LOOP_PATCH,
                    *([ACTIVE_MOVE_TRANSITION_PATCH] if mode == "full" else []),
                    *([LIVE_GRAPH_STATE_PATCH] if mode == "full" else []),
                    *([COMPACT_TRANSITION_CACHE_PATCH] if mode == "full" else []),
                    *(
                        [
                            "project_runtime_action_trace_only_v1",
                            *(
                                ["full_selected_transition_weak_action_index_v1"]
                                if mode == "full"
                                else []
                            ),
                        ]
                        if trace_recorder is not None
                        else []
                    ),
                ],
                "official_source_modified": False,
                "generation_resume_supported": mode == "full",
                "generation_checkpoint_root": (
                    str(resolved_checkpoint_root)
                    if resolved_checkpoint_root is not None
                    else None
                ),
                "generation_checkpoint_mirror_root": (
                    str(resolved_checkpoint_mirror_root)
                    if resolved_checkpoint_mirror_root is not None
                    else None
                ),
                "generation_checkpoint_interval_steps": (
                    int(checkpoint_interval_steps) if mode == "full" else None
                ),
                "generation_checkpoint_keep_last": int(checkpoint_keep_last),
                "generation_progress_interval_steps": int(progress_interval_steps),
                "generation_checkpoint_boundary": (
                    "after_move_trace_teleport_restart_and_assertions_v1"
                    if mode == "full"
                    else None
                ),
                "transition_cache_policy": (
                    "authoritative_backing_plus_exact_action_replay_lru_v3"
                    if mode == "full"
                    else "pinned_upstream_in_memory_transitions_v1"
                ),
                "graph_state_dir": str(graph_state_root) if mode == "full" else None,
                "storage_guard": (
                    {
                        "enabled": True,
                        "root": str(storage_guard.root),
                        "check_every_steps": int(storage_check_every_steps),
                        "min_free_bytes": int(storage_min_free_bytes),
                        "min_free_ratio": float(storage_min_free_ratio),
                        "min_free_inodes": int(storage_min_free_inodes),
                    }
                    if storage_guard is not None
                    else {"enabled": False}
                ),
                "transition_expanded_capacity": (
                    int(parameters.heads) if mode == "full" else None
                ),
                "transition_model_recomputation": False,
                "started_at": started,
            }
            checkpoint_provenance = _runtime_checkpoint_provenance(config)
            config["checkpoint_provenance"] = checkpoint_provenance
            existing_config_path = root / _RESOLVED_CONFIG_FILENAME
            if resume:
                if existing_config_path.is_file() and not existing_config_path.is_symlink():
                    existing_config = json.loads(
                        existing_config_path.read_text(encoding="utf-8")
                    )
                elif resolved_checkpoint_mirror_root is not None:
                    existing_config = _load_persistent_resolved_config(
                        mirror_root=resolved_checkpoint_mirror_root,
                        expected_scientific_argv=normalized_scientific_argv,
                        expected_command_sha256=normalized_command_sha256,
                        expected_total_steps=parameters.steps,
                    )
                    write_json(existing_config_path, existing_config)
                else:
                    raise GenerationCheckpointError(
                        "Resume has no local or persistent resolved_config.json."
                    )
                if not isinstance(existing_config, dict):
                    raise GenerationCheckpointError("Resume resolved_config is invalid.")
                _validate_resolved_config_binding(
                    existing_config,
                    expected_scientific_argv=normalized_scientific_argv,
                    expected_command_sha256=normalized_command_sha256,
                    expected_total_steps=parameters.steps,
                )
                if existing_config.get("checkpoint_provenance") != checkpoint_provenance:
                    raise GenerationCheckpointError(
                        "Resume inputs differ from the original resolved runtime config."
                    )
                config = existing_config
            else:
                config["config_sha256"] = _resolved_config_content_sha256(config)
                write_json(existing_config_path, config)
            if resolved_checkpoint_mirror_root is not None:
                _publish_persistent_resolved_config(
                    config,
                    mirror_root=resolved_checkpoint_mirror_root,
                )
            if not resume:
                write_json(
                    root / "progress.json",
                    _progress_payload(
                        current_step=0,
                        max_steps=parameters.steps,
                        run_complete=False,
                        config_sha256=config["config_sha256"],
                        last_checkpoint_step=None,
                        started_monotonic=process_started_monotonic,
                        process_start_step=process_start_step,
                        gpu_id=progress_gpu_id,
                        code_commit=config["project_commit"],
                    ),
                )

            loaded_checkpoint = None
            initial_loop_state: GenerationLoopState | None = None
            if resume:
                assert resolved_checkpoint_root is not None
                loaded_checkpoint = load_generation_checkpoint(
                    resolved_checkpoint_root,
                    expected_provenance=checkpoint_provenance,
                    expected_scientific_argv=normalized_scientific_argv,
                    expected_command_sha256=normalized_command_sha256,
                    expected_total_steps=parameters.steps,
                )
                if resolved_checkpoint_mirror_root is not None:
                    mirror_generation_checkpoint(
                        loaded_checkpoint.validation.checkpoint_dir,
                        resolved_checkpoint_mirror_root,
                        expected_provenance=checkpoint_provenance,
                    )
                    prune_mirrored_generation_checkpoints(
                        resolved_checkpoint_root,
                        resolved_checkpoint_mirror_root,
                        keep_last=checkpoint_keep_last,
                        expected_provenance=checkpoint_provenance,
                    )
                active_graph_store_path = graph_state_root / (
                    "authoritative_graph_store."
                    f"resume-{loaded_checkpoint.completed_step:012d}-{os.getpid()}.sqlite3"
                )
                restore_sqlite_snapshot(
                    loaded_checkpoint.sqlite_snapshot_path,
                    active_graph_store_path,
                )
                restore_official_state(
                    official, loaded_checkpoint.algorithm_state["official_state"]
                )
                initial_loop_state = GenerationLoopState.from_checkpoint_state(
                    loaded_checkpoint.algorithm_state["loop_state"]
                )
                process_start_step = initial_loop_state.completed_step
                if initial_loop_state.completed_step > int(parameters.steps):
                    raise GenerationCheckpointError(
                        "Resume checkpoint is beyond the configured final step."
                    )
            if storage_guard is not None:
                storage_guard.database_path = active_graph_store_path

            old_cwd = Path.cwd()
            try:
                os.chdir(runtime_root)
                with patched_official_runtime(
                    official,
                    model=model,
                    embedding_model=embedding_model,
                    gnn_device=device,
                    embedding_device=device,
                    batch_size=batch_size,
                    trace_recorder=trace_recorder,
                    compatibility_audit=compatibility_audit,
                    preserve_active_transitions=mode == "full",
                    compact_transitions=mode == "full",
                    transition_expanded_capacity=parameters.heads,
                    seed=parameters.seed,
                    graph_state_dir=graph_state_root if mode == "full" else None,
                    graph_store_path=(active_graph_store_path if mode == "full" else None),
                    storage_guard=storage_guard,
                ) as runtime_handles:
                    if loaded_checkpoint is not None:
                        initial_loop_state = _restore_runtime_checkpoint_state(
                            official,
                            algorithm_state=loaded_checkpoint.algorithm_state,
                            trace_state=loaded_checkpoint.trace_state,
                            handles=runtime_handles,
                            trace_recorder=trace_recorder,
                        )
                        # Restoring RNG is deliberately the final mutation
                        # before entering completed_step + 1.
                        restore_rng_state(loaded_checkpoint.rng_state)

                    latest_checkpoint_dir: str | None = (
                        str(loaded_checkpoint.validation.checkpoint_dir)
                        if loaded_checkpoint is not None
                        else None
                    )
                    last_checkpoint_step: int | None = (
                        loaded_checkpoint.completed_step
                        if loaded_checkpoint is not None
                        else None
                    )

                    def completed_step_boundary(loop_state: GenerationLoopState) -> None:
                        nonlocal latest_checkpoint_dir, last_checkpoint_step
                        checkpoint_due = bool(
                            mode == "full"
                            and resolved_checkpoint_root is not None
                            and (
                                loop_state.completed_step
                                % int(checkpoint_interval_steps)
                                == 0
                                or (
                                    storage_guard is not None
                                    and loop_state.completed_step
                                    % int(storage_check_every_steps)
                                    == 0
                                )
                                or loop_state.completed_step == int(parameters.steps)
                            )
                        )
                        if checkpoint_due:
                            algorithm_state = _checkpoint_algorithm_state(
                                official,
                                loop_state=loop_state,
                                handles=runtime_handles,
                            )
                            trace_state = (
                                trace_recorder.export_checkpoint_state()
                                if trace_recorder is not None
                                else {
                                    "schema_version": "comrecgc_trace_disabled_v1",
                                    "enabled": False,
                                }
                            )
                            validation = save_generation_checkpoint(
                                resolved_checkpoint_root,
                                completed_step=loop_state.completed_step,
                                step_complete=True,
                                algorithm_state=algorithm_state,
                                trace_state=trace_state,
                                sqlite_source=(
                                    runtime_handles.live_graph_state.store.checkpoint_connection
                                ),
                                provenance_fingerprints=checkpoint_provenance,
                                scientific_argv=normalized_scientific_argv,
                                command_sha256=normalized_command_sha256,
                                total_steps=parameters.steps,
                            )
                            if resolved_checkpoint_mirror_root is not None:
                                mirror_generation_checkpoint(
                                    validation.checkpoint_dir,
                                    resolved_checkpoint_mirror_root,
                                    expected_provenance=checkpoint_provenance,
                                )
                                prune_mirrored_generation_checkpoints(
                                    resolved_checkpoint_root,
                                    resolved_checkpoint_mirror_root,
                                    keep_last=checkpoint_keep_last,
                                    expected_provenance=checkpoint_provenance,
                                )
                            latest_checkpoint_dir = str(validation.checkpoint_dir)
                            last_checkpoint_step = loop_state.completed_step
                        if storage_guard is not None:
                            storage_guard.check(
                                loop_state.completed_step,
                                runtime_handles.live_graph_state,
                            )
                        if (
                            loop_state.completed_step % int(progress_interval_steps) == 0
                            or checkpoint_due
                            or loop_state.completed_step == int(parameters.steps)
                        ):
                            write_json(
                                root / "progress.json",
                                _progress_payload(
                                    current_step=loop_state.completed_step,
                                    max_steps=parameters.steps,
                                    run_complete=False,
                                    config_sha256=config["config_sha256"],
                                    checkpoint_dir=latest_checkpoint_dir,
                                    last_checkpoint_step=last_checkpoint_step,
                                    started_monotonic=process_started_monotonic,
                                    process_start_step=process_start_step,
                                    gpu_id=progress_gpu_id,
                                    code_commit=config["project_commit"],
                                ),
                            )

                    run_generation_loop(
                        official,
                        input_graphs=GraphListDataset(
                            bundle.graphs, bundle.node_feature_dim
                        ),
                        importance_args={
                            "gnn_model": model,
                            "neurosed_model": embedding_model,
                            "gnn_device": device,
                            "neurosed_device": device,
                        },
                        teleport_probability=parameters.teleport,
                        max_steps=parameters.steps,
                        heads=parameters.heads,
                        initial_state=initial_loop_state,
                        on_step_complete=completed_step_boundary,
                    )
                    save_official_payload(
                        official, dataset_name=dataset_key, heads=parameters.heads
                    )
            finally:
                os.chdir(old_cwd)
            official_result = (
                runtime_root / f"results/{dataset_key}/counterfactuals/comrecgc_k_{parameters.heads}.pt"
            )
            if not official_result.is_file() or official_result.stat().st_size <= 0:
                raise RuntimeError("Official COMRECGC did not serialize counterfactual candidates.")
            result_path = root / "counterfactuals.pt"
            if result_path.exists():
                if not resume:
                    raise FileExistsError(
                        f"Refusing to overwrite result: {result_path}"
                    )
                # Finalization may have been interrupted after materialization.
                # Rebuild the project copy atomically from the checkpoint-restored
                # official state instead of trusting or deleting the old file.
                _torch_save_atomic(_torch_load(official_result), result_path)
                materialization = "atomic_resume_finalization_rewrite"
            else:
                materialization = _materialize_official_result(
                    official_result, result_path
                )
            payload = _torch_load(result_path)
            graph_map, candidates = validate_counterfactual_payload(payload)
            graph_state_audit_path: Path | None = None
            if "live_graph_state" in compatibility_audit:
                graph_state_audit_path = root / "graph_state_audit.json"
                write_json(graph_state_audit_path, compatibility_audit["live_graph_state"])
            trace_summary: dict[str, Any] | None = None
            parity_summary: dict[str, Any] | None = None
            frozen_payload_audit_path: Path | None = None
            frozen_payload_audit: dict[str, Any] | None = None
            backing_store_path = active_graph_store_path
            if trace_recorder is not None:
                frozen_payload_audit_path = (
                    root / "frozen_payload_closure_audit.json"
                    if mode == "full"
                    else None
                )
                trace_summary = trace_recorder.write(
                    trace_output_dir,
                    payload,
                    source_graphs_by_parent_id=dict(
                        zip(bundle.parent_ids, bundle.graphs, strict=True)
                    ),
                    compact_candidate_lineage=mode == "full",
                    frozen_payload_backing_store=(
                        backing_store_path if mode == "full" else None
                    ),
                    frozen_payload_audit_path=frozen_payload_audit_path,
                )
                frozen_payload_audit = trace_summary.get("frozen_payload_closure")
            elif mode == "full":
                frozen_payload_audit_path = root / "frozen_payload_closure_audit.json"
                payload, frozen_payload_audit = build_frozen_payload_closure(
                    payload, (), backing_store_path=backing_store_path
                )
                write_json(frozen_payload_audit_path, frozen_payload_audit)
            if mode == "full":
                # The official file contains only the bounded hot map. Replace
                # the project copy atomically with its referentially complete
                # closure, then reload it once before permitting downstream use.
                _torch_save_atomic(payload, result_path)
                verified_payload = _torch_load(result_path)
                selected_events = (
                    iter_selected_trace(
                        Path(trace_output_dir) / "selected_action_trace_manifest.json"
                    )
                    if trace_recorder is not None
                    else ()
                )
                _verified_payload, verification = build_frozen_payload_closure(
                    verified_payload,
                    selected_events,
                    backing_store_path=None,
                )
                if not verification["closure_complete"]:
                    raise RuntimeError(
                        "COMRECGC frozen payload failed its post-write closure audit."
                    )
                frozen_payload_audit = {
                    **(frozen_payload_audit or {}),
                    **payload_file_audit(result_path),
                    "post_write_reload_verified": True,
                    "post_write_required_hash_count": verification[
                        "required_hash_count"
                    ],
                }
                if frozen_payload_audit_path is not None:
                    write_json(frozen_payload_audit_path, frozen_payload_audit)
                payload = verified_payload
                graph_map, candidates = validate_counterfactual_payload(payload)
                materialization = f"{materialization}_then_atomic_closure_rewrite"
            if parity_reference_path is not None:
                reference_path = Path(parity_reference_path).expanduser().resolve()
                parity_summary = assert_trace_parity(_torch_load(reference_path), payload)
                parity_summary["reference_path"] = str(reference_path)
                parity_summary["reference_sha256"] = sha256_file(reference_path)
                write_json(root / "trace_parity.json", parity_summary)
            manifest = {
                **config,
                "counterfactuals_path": str(result_path),
                "counterfactuals_sha256": sha256_file(result_path),
                "counterfactuals_bytes": result_path.stat().st_size,
                "artifact_materialization_mode": materialization,
                "counterfactual_candidate_count": len(candidates),
                "visited_graph_count": len(graph_map),
                "traversed_step_count": len(payload.get("traversed_hashes") or []),
                "trace_enabled": trace_recorder is not None,
                "trace_summary": trace_summary,
                "trace_parity": parity_summary,
                "official_compatibility_audit": compatibility_audit,
                "graph_state_audit_path": (
                    str(graph_state_audit_path) if graph_state_audit_path else None
                ),
                "graph_state_audit_sha256": (
                    sha256_file(graph_state_audit_path) if graph_state_audit_path else None
                ),
                "frozen_payload_closure_audit_path": (
                    str(frozen_payload_audit_path)
                    if frozen_payload_audit_path is not None
                    else None
                ),
                "frozen_payload_closure_audit_sha256": (
                    sha256_file(frozen_payload_audit_path)
                    if frozen_payload_audit_path is not None
                    else None
                ),
                "frozen_payload_closure": frozen_payload_audit,
                "candidate_order_source": "official_frequency_reinforced_order",
                "algorithm_rerun": True,
                "resumed_from_checkpoint": (
                    str(loaded_checkpoint.validation.checkpoint_dir)
                    if loaded_checkpoint is not None
                    else None
                ),
                "active_graph_store_path": (
                    str(active_graph_store_path) if mode == "full" else None
                ),
                "run_complete": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(root / "run_manifest.json", manifest)
            write_json(
                root / "progress.json",
                _progress_payload(
                    current_step=parameters.steps,
                    max_steps=parameters.steps,
                    run_complete=True,
                    config_sha256=config["config_sha256"],
                    checkpoint_dir=latest_checkpoint_dir,
                    last_checkpoint_step=last_checkpoint_step,
                    started_monotonic=process_started_monotonic,
                    process_start_step=process_start_step,
                    gpu_id=progress_gpu_id,
                    code_commit=config["project_commit"],
                ),
            )
            (root / "_RUN_FAILED.json").unlink(missing_ok=True)
            write_json(
                root / "_RUN_COMPLETE.json",
                {
                    "run_complete": True,
                    "counterfactuals_sha256": manifest["counterfactuals_sha256"],
                },
            )
            return manifest
    except Exception as exc:
        failure = {
            "stage": "project_generation",
            "dataset": dataset,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": False,
            "official_compatibility_audit": compatibility_audit,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(root / "failure_summary.json", failure)
        write_json(root / "_RUN_FAILED.json", failure)
        raise


def run_native_smoke(
    *,
    project_root: str | Path,
    upstream_root: str | Path,
    dataset: str,
    output_dir: str | Path,
    parameters: GenerationParameters,
    parent_limit: int = 32,
    device: str = "cuda:0",
    mode: str = "smoke",
    trusted_dataset_payload: str | Path | None = None,
    expected_cache_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Exercise the official TU dataset/model/NeuroSED/random-walk route."""

    parameters.validate(mode)
    project = Path(project_root).expanduser().resolve()
    trusted_payload_path: Path | None = None
    if trusted_dataset_payload is not None:
        trusted_payload_path = Path(trusted_dataset_payload).expanduser()
        if not trusted_payload_path.is_absolute():
            trusted_payload_path = project / trusted_payload_path
        trusted_payload_path = trusted_payload_path.resolve(strict=True)
    root = require_empty_output(output_dir)
    torch, _Batch = _torch_stack()
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parameters.seed)
    try:
        runtime_root = root / "official_runtime"
        runtime_root.mkdir(parents=True)
        old_cwd = Path.cwd()
        compatibility_audit: dict[str, Any] = {}
        try:
            os.chdir(Path(upstream_root).expanduser().resolve())
            with imported_upstream(upstream_root) as modules:
                if dataset == "aids":
                    if (
                        trusted_dataset_payload is None
                        or expected_cache_inventory_sha256 is None
                    ):
                        raise ValueError(
                            "Native AIDS execution requires a scoped trusted-cache payload."
                        )
                    graphs, _dataset_payload = load_aids_tensor_payload(
                        trusted_payload_path,
                        expected_inventory_sha256=expected_cache_inventory_sha256,
                    )
                    num_features = int(graphs[0].x.shape[1])
                else:
                    graphs = modules["data"].load_dataset(dataset)
                    num_features = int(graphs.num_features)
                model = (
                    _load_native_aids_gnn_from_trusted_features(
                        gnn_module=modules["gnn"],
                        upstream_root=upstream_root,
                        num_features=num_features,
                        device=device,
                    )
                    if dataset == "aids"
                    else modules["gnn"].load_trained_gnn(dataset, device=device).eval()
                )
                predictions = modules["gnn"].load_trained_prediction(dataset, device=device).cpu()
                all_source_indices = torch.where(predictions == 0)[0]
                source_indices = (
                    all_source_indices
                    if mode == "full"
                    else all_source_indices[: int(parent_limit)]
                )
                sources = _materialize_dataset_indices(
                    graphs, [int(value) for value in source_indices.tolist()]
                )
                if mode == "smoke" and len(sources) != int(parent_limit):
                    raise RuntimeError("Native TU smoke source cohort is smaller than requested.")
                for index, graph in enumerate(sources):
                    graph.comrecgc_parent_id = f"TU_{dataset.upper()}_{int(source_indices[index]):06d}"
                    graph.comrecgc_source_index = int(source_indices[index])
                    graph.comrecgc_source_smiles = ""
                    graph.comrecgc_project_label = -1
                    graph.comrecgc_node_origin = torch.arange(int(graph.num_nodes), dtype=torch.long)
                embedding = modules["distance"].load_neurosed(
                    sources,
                    neurosed_model_path=str(
                        Path(upstream_root).expanduser().resolve() / f"data/{dataset}/neurosed/best_model.pt"
                    ),
                    device=device,
                ).to(device).eval()
                official = modules["comrecgc"]
                reset_official_state(
                    official,
                    candidate_capacity=parameters.candidate_capacity,
                    sample_size=parameters.sample_size,
                )
                official.input_graphs_covered = torch.zeros(len(sources), dtype=torch.float32)
                os.chdir(runtime_root)
                try:
                    with patched_official_runtime(
                        official,
                        model=model,
                        embedding_model=embedding,
                        gnn_device=device,
                        embedding_device=device,
                        batch_size=128,
                        compatibility_audit=compatibility_audit,
                    ):
                        official.counterfactual_summary_with_randomwalk(
                            dataset_name=f"native_{dataset}",
                            input_graphs=GraphListDataset(sources, num_features),
                            importance_args={},
                            teleport_probability=parameters.teleport,
                            max_steps=parameters.steps,
                            heads=parameters.heads,
                        )
                finally:
                    os.chdir(Path(upstream_root).expanduser().resolve())
        finally:
            os.chdir(old_cwd)
        source_result = (
            runtime_root
            / f"results/native_{dataset}/counterfactuals/comrecgc_k_{parameters.heads}.pt"
        )
        result = root / "counterfactuals.pt"
        materialization_mode = _materialize_official_result(source_result, result)
        payload = _torch_load(result)
        graph_map, candidates = validate_counterfactual_payload(payload)
        native_common = _run_native_common_recourse(
            modules=modules,
            sources=sources,
            payload=payload,
            embedding_model=embedding,
            output_dir=root,
            device=device,
            mode=mode,
            allow_empty=mode == "full",
        )
        manifest = {
            "method": METHOD,
            "route": "native_reproduction",
            "dataset": f"TU/{dataset}",
            "mode": mode,
            "project_commit": _git_commit(project),
            "upstream_commit": UPSTREAM_COMMIT,
            "parameters": parameters.__dict__,
            "parent_limit": len(sources),
            "full_parent_universe": mode == "full",
            "counterfactual_candidate_count": len(candidates),
            "visited_graph_count": len(graph_map),
            "common_recourse_count": int(native_common["common_recourse_count"]),
            "scientific_output_empty": bool(native_common["scientific_output_empty"]),
            "execution_status": str(native_common["execution_status"]),
            "native_cost": native_common["native_cost"],
            "native_common_recourse_path": str(root / "native_common_recourse.json"),
            "native_common_recourse_sha256": sha256_file(
                root / "native_common_recourse.json"
            ),
            "representative_counterfactuals_path": native_common[
                "representative_counterfactuals_path"
            ],
            "representative_counterfactuals_sha256": native_common[
                "representative_counterfactuals_sha256"
            ],
            "serialization_reloadable": native_common["serialization_reloadable"],
            "no_nan_or_inf": native_common["no_nan_or_inf"],
            "counterfactuals_path": str(result),
            "counterfactuals_sha256": sha256_file(result),
            "artifact_materialization_mode": materialization_mode,
            "official_compatibility_patches": list(OFFICIAL_RUNTIME_PATCHES),
            "official_compatibility_audit": compatibility_audit,
            "not_eligible_for_project_figures": True,
            "run_complete": True,
            "full_execution_pass": mode == "full",
        }
        write_json(root / "run_manifest.json", manifest)
        write_json(root / "_RUN_COMPLETE.json", {"run_complete": True})
        return manifest
    except Exception as exc:
        failure = {
            "stage": f"native_{mode}",
            "dataset": dataset,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "run_complete": False,
        }
        write_json(root / "failure_summary.json", failure)
        write_json(root / "_RUN_FAILED.json", failure)
        raise
