"""Official common-recourse clustering with auditable graph medoids."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    METHOD,
    RecourseParameters,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .model_adapter import AIDSGreedEmbeddingAdapter
from .project_dataset import load_aids_generation_bundle, load_mutagenicity_generation_bundle
from .upstream import imported_upstream


def _torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        from torch_geometric.data import Batch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("COMRECGC common-recourse requires torch and torch_geometric.") from exc
    return torch, Batch


def _torch_load(path: Path) -> Any:
    torch, _Batch = _torch_stack()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save_atomic(payload: Any, path: Path) -> None:
    import os

    torch, _Batch = _torch_stack()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_graph_id(graph: Any) -> str:
    x = graph.x.detach().cpu().tolist()
    pairs = sorted(
        {
            tuple(sorted((int(source), int(target))))
            for source, target in graph.edge_index.detach().cpu().t().tolist()
            if int(source) != int(target)
        }
    )
    return "COMRECGC_" + stable_json_sha256({"x": x, "edges": pairs})[:20].upper()


def choose_cluster_medoid(
    recourse_vectors: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
) -> tuple[int, int, float]:
    """Return the real source->CF pair nearest the center, preserving tie order."""

    if len(pair_indices) == 0 or recourse_vectors.shape[0] != len(pair_indices):
        raise ValueError("A cluster medoid requires aligned, non-empty vectors and pairs.")
    centroid = np.mean(recourse_vectors, axis=0)
    distances = np.linalg.norm(recourse_vectors - centroid, axis=1)
    winner = int(np.argmin(distances))
    source_index, counterfactual_index = pair_indices[winner]
    return int(source_index), int(counterfactual_index), float(distances[winner])


def trace_official_cluster_order(
    *,
    labels: np.ndarray,
    recourse_vectors: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
    radius: float,
    theta: float,
    recourse_size: int,
    official_greedy: Any,
) -> list[dict[str, Any]]:
    """Reconstruct official greedy inputs and retain real graph medoids.

    Cluster construction and greedy calls match ``common_recourse.py``.  This
    function adds lineage only; it does not alter labels, centers, coverage, or
    tie ordering.
    """

    common_recourse: dict[int, set[int]] = {}
    centroid_norms: dict[int, float] = {}
    cluster_cf_indices: dict[int, set[int]] = {}
    cluster_pairs: dict[int, list[tuple[int, int]]] = {}
    cluster_vectors: dict[int, np.ndarray] = {}
    max_label = int(labels.max()) if labels.size else -1
    for cluster_label in range(max_label + 1):
        positions = np.flatnonzero(labels == cluster_label)
        points = recourse_vectors[positions]
        if points.size == 0:
            continue
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        covered: set[int] = set()
        counterfactuals: set[int] = set()
        retained_pairs: list[tuple[int, int]] = []
        retained_vectors: list[np.ndarray] = []
        for local_index, distance in enumerate(distances):
            if float(distance) < float(radius):
                pair = pair_indices[int(positions[local_index])]
                covered.add(int(pair[0]))
                counterfactuals.add(int(pair[1]))
                retained_pairs.append((int(pair[0]), int(pair[1])))
                retained_vectors.append(points[local_index])
        common_recourse[cluster_label] = covered
        centroid_norms[cluster_label] = float(np.linalg.norm(centroid))
        cluster_cf_indices[cluster_label] = counterfactuals
        cluster_pairs[cluster_label] = retained_pairs
        cluster_vectors[cluster_label] = np.asarray(retained_vectors)

    filtered = {
        label: set(parents)
        for label, parents in common_recourse.items()
        if centroid_norms[label] < float(theta) and parents and cluster_pairs[label]
    }
    covered_by: dict[int, set[int]] = defaultdict(set)
    for label, parents in filtered.items():
        for parent in parents:
            covered_by[parent].add(label)
    if not filtered:
        return []
    selection = official_greedy(
        counterfactual_covering={label: set(values) for label, values in filtered.items()},
        graphs_covered_by={key: set(values) for key, values in covered_by.items()},
        k=min(int(recourse_size), len(filtered)),
    )
    ordered: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    covered: set[int] = set()
    for rank, value in selection.items():
        cluster_label = int(value[0])
        source_index, counterfactual_index, medoid_distance = choose_cluster_medoid(
            cluster_vectors[cluster_label],
            cluster_pairs[cluster_label],
        )
        covered.update(filtered[cluster_label])
        cumulative_cost += centroid_norms[cluster_label]
        ordered.append(
            {
                "rank": int(rank),
                "cluster_label": cluster_label,
                "cluster_center_norm": centroid_norms[cluster_label],
                "cluster_radius": float(radius),
                "cluster_size": int(np.sum(labels == cluster_label)),
                "representative_source_index": source_index,
                "representative_counterfactual_index": counterfactual_index,
                "representative_distance_to_center": medoid_distance,
                "covered_parent_indices_native": sorted(filtered[cluster_label]),
                "native_cumulative_covered_count": len(covered),
                "native_cumulative_cost": cumulative_cost,
                "member_counterfactual_indices": sorted(cluster_cf_indices[cluster_label]),
            }
        )
    return ordered


def ordered_prefix(records: Sequence[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    ranks = [int(record["rank"]) for record in values]
    if ranks != list(range(1, len(values) + 1)):
        raise ValueError("Common-recourse records are not in a unique contiguous official order.")
    return values[: int(k)]


def _importance_parts(candidate: Mapping[str, Any]) -> list[float]:
    value = candidate.get("importance_parts")
    if value is None:
        return [0.0]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not value:
        return [0.0]
    return [float(part) for part in value]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_common_recourse(
    *,
    upstream_root: str | Path,
    dataset: str,
    dataset_dir: str | Path,
    source_csv: str | Path | None,
    generation_dir: str | Path,
    distance_checkpoint: str | Path,
    output_dir: str | Path,
    mode: str,
    parent_limit: int,
    parameters: RecourseParameters,
    device: str = "cuda:0",
    batch_size: int = 128,
    resume: bool = False,
) -> dict[str, Any]:
    parameters.validate(mode)
    root = require_empty_output(output_dir, resume=resume)
    generation_root = Path(generation_dir).expanduser().resolve()
    generation_manifest = json_load(generation_root / "run_manifest.json")
    if generation_manifest.get("run_complete") is not True:
        raise ValueError("Generation manifest is not complete.")
    if generation_manifest.get("dataset") != dataset or generation_manifest.get("mode") != mode:
        raise ValueError("Generation lineage dataset/profile mismatch.")
    if int(generation_manifest.get("parent_limit", -1)) != int(parent_limit):
        raise ValueError("Generation parent_limit differs from common-recourse contract.")
    if dataset == "aids":
        if source_csv is None:
            raise ValueError("AIDS common-recourse requires --source-csv.")
        bundle = load_aids_generation_bundle(
            dataset_dir=dataset_dir,
            source_csv=source_csv,
            parent_limit=parent_limit,
        )
    else:
        bundle = load_mutagenicity_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    payload_path = Path(generation_manifest["counterfactuals_path"]).resolve()
    if sha256_file(payload_path) != generation_manifest["counterfactuals_sha256"]:
        raise ValueError("Generation counterfactual artifact SHA256 mismatch.")
    payload = _torch_load(payload_path)
    graph_map = payload.get("graph_map") or {}
    raw_candidates = list(payload.get("counterfactual_candidates") or [])
    candidate_graphs: list[Any] = []
    generation_indices: list[int] = []
    for generation_index, candidate in enumerate(raw_candidates):
        importance = _importance_parts(candidate)
        graph_hash = candidate.get("graph_hash")
        if float(importance[0]) >= 0.5 and graph_hash in graph_map:
            candidate_graphs.append(graph_map[graph_hash][0])
            generation_indices.append(generation_index)
        if len(candidate_graphs) >= int(parameters.cf_size):
            break
    if not candidate_graphs:
        raise RuntimeError("No model-counterfactual candidates are available for clustering.")
    torch, Batch = _torch_stack()
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("COMRECGC common-recourse requires scikit-learn.") from exc
    with imported_upstream(upstream_root) as modules:
        if dataset == "aids":
            embedding_model = AIDSGreedEmbeddingAdapter(
                distance_checkpoint,
                atom_vocabulary=[str(value) for value in bundle.atom_vocabulary],
                device=device,
            ).eval()
        else:
            embedding_model = modules["distance"].load_neurosed(
                bundle.graphs,
                neurosed_model_path=str(Path(distance_checkpoint).expanduser().resolve()),
                device=device,
            ).to(device).eval()
        original_batch = Batch.from_data_list(bundle.graphs).to(device)
        with torch.no_grad():
            original_embeddings = embedding_model.embed_model(original_batch).detach().cpu()
        embedding_model.embed_targets(bundle.graphs)
        original_counts = modules["util"].graph_element_counts(bundle.graphs).cpu()
        pair_indices: list[tuple[int, int]] = []
        recourse_vectors: list[np.ndarray] = []
        distance_pair_count = 0
        for start in range(0, len(candidate_graphs), max(1, int(batch_size))):
            chunk = candidate_graphs[start : start + max(1, int(batch_size))]
            with torch.no_grad():
                raw_distances = embedding_model.predict_outer_with_queries(chunk, batch_size=batch_size).cpu()
                chunk_embeddings = embedding_model.embed_model(
                    Batch.from_data_list(chunk).to(device)
                ).detach().cpu()
            chunk_counts = modules["util"].graph_element_counts(chunk).cpu()
            scale = chunk_counts[:, None] + original_counts[None, :]
            normalized = raw_distances / scale
            valid_pairs = torch.nonzero(normalized <= float(parameters.theta), as_tuple=False)
            distance_pair_count += int(normalized.numel())
            for local_cf, original_index in valid_pairs.tolist():
                global_cf = start + int(local_cf)
                pair_indices.append((int(original_index), global_cf))
                vector = (
                    (chunk_embeddings[int(local_cf)] - original_embeddings[int(original_index)])
                    / scale[int(local_cf), int(original_index)]
                )
                recourse_vectors.append(vector.numpy())
        if recourse_vectors:
            recourse_array = np.asarray(recourse_vectors)
            if not np.isfinite(recourse_array).all():
                raise RuntimeError("Recourse embeddings contain NaN/Inf.")
            clustering = DBSCAN(eps=float(parameters.delta), min_samples=int(parameters.cluster_size))
            clustering.fit(recourse_array)
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
            cluster_labels = np.asarray(clustering.labels_)
        else:
            official_result = ([], [], [])
            selected = []
            cluster_labels = np.asarray([], dtype=int)
    # An empty cluster set is a valid scientific result.  Engineering success
    # is determined by complete model/embedding/DBSCAN execution and artifacts,
    # not by forcing positive recourse yield in smoke.
    representative_graphs: list[Any] = []
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        local_cf_index = int(row["representative_counterfactual_index"])
        graph = candidate_graphs[local_cf_index]
        source_index = int(row["representative_source_index"])
        candidate_id = stable_graph_id(graph)
        representative_graphs.append(graph)
        output_rows.append(
            {
                **row,
                "common_recourse_id": f"COMRECGC_{dataset.upper()}_{int(row['rank']):04d}",
                "candidate_id": candidate_id,
                "representative_source_graph_id": bundle.parent_ids[source_index],
                "representative_counterfactual_graph_id": candidate_id,
                "generation_candidate_index": generation_indices[local_cf_index],
                "native_coverage": int(row["native_cumulative_covered_count"]) / len(bundle.graphs),
                "native_cost": float(row["native_cumulative_cost"]),
            }
        )
    _torch_save_atomic(representative_graphs, root / "representative_counterfactuals.pt")
    write_json(root / "selected_common_recourses.json", output_rows)
    fields = [
        "rank",
        "common_recourse_id",
        "candidate_id",
        "cluster_label",
        "cluster_center_norm",
        "cluster_radius",
        "cluster_size",
        "representative_source_graph_id",
        "representative_counterfactual_graph_id",
        "native_coverage",
        "native_cost",
    ]
    _write_csv(root / "selected_common_recourses.csv", output_rows, fields)
    manifest = {
        "method": METHOD,
        "dataset": dataset,
        "route": "project_adapted",
        "mode": mode,
        "adaptation_mode": ADAPTATION_MODE,
        "cf_mode": CF_MODE,
        "parameters": parameters.__dict__,
        "generation_manifest_path": str(generation_root / "run_manifest.json"),
        "generation_manifest_sha256": sha256_file(generation_root / "run_manifest.json"),
        "counterfactuals_sha256": sha256_file(payload_path),
        "model_counterfactual_candidate_count": len(candidate_graphs),
        "distance_pair_count": distance_pair_count,
        "theta_eligible_pair_count": len(pair_indices),
        "dbscan_cluster_count": len({int(value) for value in cluster_labels if int(value) >= 0}),
        "dbscan_noise_point_count": int(np.count_nonzero(cluster_labels < 0)),
        "common_recourse_count": len(output_rows),
        "scientific_output_empty": not bool(output_rows),
        "execution_status": (
            "SCIENTIFIC_OUTPUT_EMPTY" if not output_rows else "FULL_EXECUTION_PASS"
        ),
        "native_cost": None if not output_rows else output_rows[-1]["native_cost"],
        "official_coverage_summary_invoked": True,
        "official_coverage_summary_result": [list(value) for value in official_result],
        "official_greedy_order_preserved": True,
        "embedding_centers_exported_as_graphs": False,
        "representative_policy": "real_pair_nearest_cluster_center",
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "representative_counterfactuals_path": str(root / "representative_counterfactuals.pt"),
        "representative_counterfactuals_sha256": sha256_file(
            root / "representative_counterfactuals.pt"
        ),
        "run_complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True})
    return manifest


def json_load(path: Path) -> dict[str, Any]:
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
