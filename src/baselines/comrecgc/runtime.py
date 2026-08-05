"""Execution bridge into the unmodified COMRECGC random walk."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

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
    load_mutagenicity_gnn,
)
from .project_dataset import (
    GraphListDataset,
    ProjectDatasetBundle,
    load_aids_generation_bundle,
    load_mutagenicity_generation_bundle,
)
from .upstream import imported_upstream

OFFICIAL_RUNTIME_PATCHES = (
    "project_dataset_injection_v1",
    "project_internal_label_mapping_v1",
    "bounded_batch_call_oom_safe_v1",
    "source_graph_lineage_v1",
)


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


def lineage_neighbor_wrapper(original: Callable[[Any, tuple[Any, ...]], Any]) -> Callable[[Any, tuple[Any, ...]], Any]:
    """Preserve source-node lineage without changing graph tensors."""

    def wrapped(graph: Any, action: tuple[Any, ...]) -> Any:
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
) -> Iterator[None]:
    originals = {
        "call": module.call,
        "neighbor_graph_access": module.neighbor_graph_access,
    }
    module.call = _safe_call_factory(
        model=model,
        embedding_model=embedding_model,
        gnn_device=gnn_device,
        embedding_device=embedding_device,
        batch_size=batch_size,
    )
    module.neighbor_graph_access = lineage_neighbor_wrapper(module.neighbor_graph_access)
    try:
        yield
    finally:
        module.call = originals["call"]
        module.neighbor_graph_access = originals["neighbor_graph_access"]


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
    raise ValueError(f"Unsupported project dataset: {dataset}")


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


def _run_native_common_recourse_smoke(
    *,
    modules: Mapping[str, Any],
    sources: Sequence[Any],
    payload: Mapping[str, Any],
    embedding_model: Any,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Exercise official clustering/summary on real native random-walk output."""

    from sklearn.cluster import DBSCAN

    from .recourse import stable_graph_id, trace_official_cluster_order

    parameters = RecourseParameters.for_mode("smoke")
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
    if not recourse_vectors:
        raise RuntimeError("Native smoke has no pairs inside the official theta gate.")
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
    diagnostics = {
        "model_counterfactual_candidate_count": len(candidate_graphs),
        "distance_pair_count": distance_pair_count,
        "theta_eligible_pair_count": len(pair_indices),
        "dbscan_cluster_count": len(
            {int(value) for value in clustering.labels_ if int(value) >= 0}
        ),
        "official_coverage_summary_invoked": True,
        "official_coverage_summary_result": [list(value) for value in official_result],
        "selected_common_recourse_count": len(selected),
    }
    if not selected:
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
) -> dict[str, Any]:
    parameters.validate(mode)
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
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parameters.seed)
    dataset_key = "project_aids" if dataset == "aids" else "project_mutagenicity"
    runtime_root = root / "official_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
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
                model, model_provenance = load_mutagenicity_gnn(
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
                    "distance_model": "mutagenicity_neurosed",
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
                "gnn": model_provenance,
                "distance_model": distance_provenance,
                "internal_prediction_counts": internal_counts,
                "cf_mode": CF_MODE,
                "calibration_loaded": False,
                "test_loaded": False,
                "official_compatibility_patches": list(OFFICIAL_RUNTIME_PATCHES),
                "official_source_modified": False,
                "started_at": started,
            }
            config["config_sha256"] = stable_json_sha256(config)
            write_json(root / "resolved_config.json", config)
            write_json(
                root / "progress.json",
                {
                    "stage": "generation",
                    "current_step": 0,
                    "max_steps": parameters.steps,
                    "run_complete": False,
                    "config_sha256": config["config_sha256"],
                },
            )
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
                ):
                    official.counterfactual_summary_with_randomwalk(
                        dataset_name=dataset_key,
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
                    )
            finally:
                os.chdir(old_cwd)
            official_result = (
                runtime_root / f"results/{dataset_key}/counterfactuals/comrecgc_k_{parameters.heads}.pt"
            )
            if not official_result.is_file() or official_result.stat().st_size <= 0:
                raise RuntimeError("Official COMRECGC did not serialize counterfactual candidates.")
            result_path = root / "counterfactuals.pt"
            materialization = _materialize_official_result(official_result, result_path)
            payload = _torch_load(result_path)
            graph_map, candidates = validate_counterfactual_payload(payload)
            manifest = {
                **config,
                "counterfactuals_path": str(result_path),
                "counterfactuals_sha256": sha256_file(result_path),
                "counterfactuals_bytes": result_path.stat().st_size,
                "artifact_materialization_mode": materialization,
                "counterfactual_candidate_count": len(candidates),
                "visited_graph_count": len(graph_map),
                "traversed_step_count": len(payload.get("traversed_hashes") or []),
                "candidate_order_source": "official_frequency_reinforced_order",
                "algorithm_rerun": True,
                "run_complete": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(root / "run_manifest.json", manifest)
            write_json(
                root / "progress.json",
                {
                    "stage": "generation",
                    "current_step": parameters.steps,
                    "max_steps": parameters.steps,
                    "run_complete": True,
                    "config_sha256": config["config_sha256"],
                },
            )
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
) -> dict[str, Any]:
    """Exercise the official TU dataset/model/NeuroSED/random-walk route."""

    parameters.validate("smoke")
    project = Path(project_root).expanduser().resolve()
    root = require_empty_output(output_dir)
    torch, _Batch = _torch_stack()
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    try:
        runtime_root = root / "official_runtime"
        runtime_root.mkdir(parents=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(Path(upstream_root).expanduser().resolve())
            with imported_upstream(upstream_root) as modules:
                graphs = modules["data"].load_dataset(dataset)
                num_features = int(graphs.num_features)
                model = modules["gnn"].load_trained_gnn(dataset, device=device).eval()
                predictions = modules["gnn"].load_trained_prediction(dataset, device=device).cpu()
                source_indices = torch.where(predictions == 0)[0][: int(parent_limit)]
                sources = _materialize_dataset_indices(
                    graphs, [int(value) for value in source_indices.tolist()]
                )
                if len(sources) != int(parent_limit):
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
        native_common = _run_native_common_recourse_smoke(
            modules=modules,
            sources=sources,
            payload=payload,
            embedding_model=embedding,
            output_dir=root,
            device=device,
        )
        manifest = {
            "method": METHOD,
            "route": "native_reproduction",
            "dataset": f"TU/{dataset}",
            "project_commit": _git_commit(project),
            "upstream_commit": UPSTREAM_COMMIT,
            "parameters": parameters.__dict__,
            "parent_limit": int(parent_limit),
            "counterfactual_candidate_count": len(candidates),
            "visited_graph_count": len(graph_map),
            "common_recourse_count": int(native_common["common_recourse_count"]),
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
            "not_eligible_for_project_figures": True,
            "run_complete": True,
        }
        write_json(root / "run_manifest.json", manifest)
        write_json(root / "_RUN_COMPLETE.json", {"run_complete": True})
        return manifest
    except Exception as exc:
        failure = {
            "stage": "native_smoke",
            "dataset": dataset,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "run_complete": False,
        }
        write_json(root / "failure_summary.json", failure)
        write_json(root / "_RUN_FAILED.json", failure)
        raise
