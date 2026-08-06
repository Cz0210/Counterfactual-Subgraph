"""Read-only DBSCAN funnel audit for frozen native COMRECGC candidates."""

from __future__ import annotations

import ast
import csv
import io
import math
import os
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .graph_trace import stable_graph_sha256
from .runtime import _materialize_dataset_indices, _torch_load, _torch_stack, model_counterfactual_graphs
from .upstream import imported_upstream, validate_upstream_checkout


@dataclass(frozen=True)
class DBSCANContract:
    theta: float
    eps: float
    min_samples: int
    recourse_size: int
    cf_size: int
    metric: str = "euclidean"
    algorithm: str = "auto"


@contextmanager
def _working_directory(path: str | Path):
    previous = Path.cwd()
    try:
        os.chdir(Path(path).expanduser().resolve())
        yield
    finally:
        os.chdir(previous)


def resolve_upstream_contract(upstream_root: str | Path) -> DBSCANContract:
    root = validate_upstream_checkout(upstream_root)
    source = (root / "common_recourse.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = ast.literal_eval(node.args[0])
        if option not in {"--theta", "--delta", "--cluster_size", "--recourse_size", "--cf_size"}:
            continue
        default_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        if default_node is not None:
            defaults[str(option)] = ast.literal_eval(default_node)
    required = {"--theta", "--delta", "--cluster_size", "--recourse_size", "--cf_size"}
    if set(defaults) != required:
        raise ValueError(f"Could not resolve pinned upstream DBSCAN defaults: {defaults}")
    return DBSCANContract(
        theta=float(defaults["--theta"]),
        eps=float(defaults["--delta"]),
        min_samples=int(defaults["--cluster_size"]),
        recourse_size=int(defaults["--recourse_size"]),
        cf_size=int(defaults["--cf_size"]),
    )


def distribution_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
    }
    names = ("min", "q01", "q05", "q10", "q25", "median", "q75", "q90", "q95", "q99", "max")
    if finite.size == 0:
        result.update({name: None for name in names})
        result.update({"mean": None, "std": None, "num_zero": 0, "num_exact_duplicates": 0})
        return result
    quantiles = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    result.update({name: float(value) for name, value in zip(names, quantiles, strict=True)})
    result.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "num_zero": int(np.count_nonzero(finite == 0.0)),
            "num_exact_duplicates": int(finite.size - np.unique(finite).size),
        }
    )
    return result


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    atomic_write_bytes(path, _csv_bytes(rows, fields))


def _epsilon_components(adjacency: np.ndarray) -> list[list[int]]:
    unseen = set(range(int(adjacency.shape[0])))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            neighbors = set(int(value) for value in np.flatnonzero(adjacency[current]))
            discovered = neighbors & unseen
            unseen.difference_update(discovered)
            queue.extend(sorted(discovered))
        components.append(sorted(component))
    return components


def audit_geometry(
    *,
    recourse_vectors: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
    parent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    contract: DBSCANContract,
) -> dict[str, Any]:
    from sklearn.cluster import DBSCAN

    recourse = np.asarray(recourse_vectors, dtype=np.float64)
    if recourse.ndim != 2 or recourse.shape[0] != len(pair_indices) or not np.isfinite(recourse).all():
        raise ValueError("Finite recourse vectors must align exactly with pair indices.")
    differences = recourse[:, None, :] - recourse[None, :, :]
    pairwise = np.linalg.norm(differences, axis=-1).astype(np.float64, copy=False)
    adjacency = pairwise <= float(contract.eps)
    neighbor_counts = adjacency.sum(axis=1).astype(int)
    clustering = DBSCAN(
        eps=float(contract.eps),
        min_samples=int(contract.min_samples),
        metric=contract.metric,
        algorithm=contract.algorithm,
    ).fit(recourse)
    labels = np.asarray(clustering.labels_, dtype=int)
    core_indices = set(int(value) for value in clustering.core_sample_indices_.tolist())
    point_rows: list[dict[str, Any]] = []
    for index, (parent_index, candidate_index) in enumerate(pair_indices):
        sorted_distances = np.sort(pairwise[index])
        nonself = sorted_distances[sorted_distances > 0.0]
        point_rows.append(
            {
                "point_id": index,
                "parent_index": int(parent_index),
                "parent_id": str(parent_ids[int(parent_index)]),
                "candidate_index": int(candidate_index),
                "candidate_id": str(candidate_ids[int(candidate_index)]),
                "recourse_norm": float(np.linalg.norm(recourse[index])),
                "epsilon_neighbor_count_including_self": int(neighbor_counts[index]),
                "nearest_nonself_distance": float(nonself[0]) if nonself.size else None,
                "second_nearest_nonself_distance": float(nonself[1]) if nonself.size > 1 else None,
                "min_samples_th_neighbor_distance_including_self": float(
                    sorted_distances[min(int(contract.min_samples) - 1, len(sorted_distances) - 1)]
                ),
                "is_core": index in core_indices,
                "is_border": bool(labels[index] >= 0 and index not in core_indices),
                "is_noise": bool(labels[index] < 0),
                "dbscan_label": int(labels[index]),
            }
        )
    components = _epsilon_components(adjacency)
    component_rows: list[dict[str, Any]] = []
    for component_id, members in enumerate(components):
        parents = sorted({str(parent_ids[pair_indices[index][0]]) for index in members})
        candidates = sorted({str(candidate_ids[pair_indices[index][1]]) for index in members})
        degrees = [int(neighbor_counts[index] - 1) for index in members]
        component_rows.append(
            {
                "component_id": component_id,
                "size": len(members),
                "unique_parent_count": len(parents),
                "unique_candidate_count": len(candidates),
                "parent_ids": ";".join(parents),
                "candidate_ids": ";".join(candidates),
                "contains_core": any(index in core_indices for index in members),
                "max_degree": max(degrees, default=0),
                "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            }
        )
    cluster_rows: list[dict[str, Any]] = []
    postfilter_rows: list[dict[str, Any]] = []
    for cluster_label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        positions = np.flatnonzero(labels == cluster_label)
        points = recourse[positions]
        centroid = np.mean(points, axis=0)
        centroid_distances = np.linalg.norm(points - centroid, axis=1)
        accepted_lt = positions[centroid_distances < float(contract.eps)]
        accepted_le = positions[centroid_distances <= float(contract.eps)]
        accepted_parents = {
            int(pair_indices[int(index)][0]) for index in accepted_lt.tolist()
        }
        accepted_candidates = {
            int(pair_indices[int(index)][1]) for index in accepted_lt.tolist()
        }
        row = {
            "cluster_label": cluster_label,
            "cluster_size": int(len(positions)),
            "num_core": int(sum(int(index) in core_indices for index in positions)),
            "num_border": int(sum(int(index) not in core_indices for index in positions)),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "min_distance_to_centroid": float(np.min(centroid_distances)),
            "median_distance_to_centroid": float(np.median(centroid_distances)),
            "max_distance_to_centroid": float(np.max(centroid_distances)),
            "num_points_with_distance_to_centroid_lt_delta": int(len(accepted_lt)),
            "num_points_with_distance_to_centroid_le_delta": int(len(accepted_le)),
            "unique_parents_inside_radius": len(accepted_parents),
            "unique_candidates_inside_radius": len(accepted_candidates),
            "passed_centroid_norm_theta": bool(np.linalg.norm(centroid) < float(contract.theta)),
            "coverage_parent_count": len(accepted_parents),
        }
        cluster_rows.append(row)
        if row["passed_centroid_norm_theta"] and accepted_parents:
            postfilter_rows.append(row)
    upper = pairwise[np.triu_indices(len(recourse), k=1)]
    k_distances = np.asarray(
        [row["min_samples_th_neighbor_distance_including_self"] for row in point_rows],
        dtype=np.float64,
    )
    return {
        "labels": labels,
        "point_rows": point_rows,
        "component_rows": component_rows,
        "cluster_rows": cluster_rows,
        "postfilter_rows": postfilter_rows,
        "pairwise": pairwise,
        "statistics": {
            "recourse_norm": distribution_summary(np.linalg.norm(recourse, axis=1)),
            "pairwise_euclidean": distribution_summary(upper),
            "k_distance": distribution_summary(k_distances),
        },
        "dbscan_core_points": len(core_indices),
        "dbscan_border_points": int(sum(row["is_border"] for row in point_rows)),
        "dbscan_noise_points": int(sum(row["is_noise"] for row in point_rows)),
        "dbscan_non_noise_clusters": len(cluster_rows),
        "postfilter_cluster_count": len(postfilter_rows),
        "num_points_with_neighbor_count_ge_min_samples": int(
            np.count_nonzero(neighbor_counts >= int(contract.min_samples))
        ),
        "num_connected_components": len(components),
        "largest_component_size": max((len(value) for value in components), default=0),
        "component_size_histogram": dict(Counter(len(value) for value in components)),
        "num_components_size_ge_min_samples": sum(
            len(value) >= int(contract.min_samples) for value in components
        ),
        "num_components_with_core_point": sum(
            bool(row["contains_core"]) for row in component_rows
        ),
    }


def run_aids_native_dbscan_audit(
    *,
    project_root: str | Path,
    upstream_root: str | Path,
    counterfactuals_path: str | Path,
    output_dir: str | Path,
    parent_limit: int,
    expected_sha256: str,
    expected_candidates: int = 31,
    expected_distance_pairs: int | None = 1984,
    expected_eligible_pairs: int | None = 28,
    full_reject_parent_universe: bool = False,
    preregistration_path: str | Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 128,
) -> dict[str, Any]:
    root = require_empty_output(output_dir)
    project = Path(project_root).expanduser().resolve()
    artifact = Path(counterfactuals_path).expanduser().resolve()
    if sha256_file(artifact) != expected_sha256:
        raise ValueError("AIDS native counterfactual artifact SHA256 mismatch.")
    contract = resolve_upstream_contract(upstream_root)
    payload = _torch_load(artifact)
    raw_candidates = list(payload.get("counterfactual_candidates") or [])
    candidate_graphs = model_counterfactual_graphs(payload, limit=contract.cf_size)
    if len(candidate_graphs) != int(expected_candidates):
        raise ValueError(
            f"AIDS model-counterfactual count mismatch: {len(candidate_graphs)} != {expected_candidates}"
        )
    torch, Batch = _torch_stack()
    with imported_upstream(upstream_root) as modules, _working_directory(upstream_root):
        dataset = modules["data"].load_dataset("aids")
        predictions = modules["gnn"].load_trained_prediction("aids", device=device).cpu()
        reject_indices = [int(value) for value in torch.where(predictions == 0)[0].tolist()]
        if not full_reject_parent_universe:
            reject_indices = reject_indices[: int(parent_limit)]
        else:
            reject_indices = sorted(reject_indices, key=lambda value: f"TU_AIDS_{value:06d}")
        sources = _materialize_dataset_indices(dataset, reject_indices)
        parent_ids = [f"TU_AIDS_{value:06d}" for value in reject_indices]
        embedding = modules["distance"].load_neurosed(
            sources,
            neurosed_model_path=str(
                Path(upstream_root).expanduser().resolve() / "data/aids/neurosed/best_model.pt"
            ),
            device=device,
        ).to(device).eval()
        with torch.no_grad():
            source_embeddings = embedding.embed_model(Batch.from_data_list(sources).to(device)).detach().cpu()
        embedding.embed_targets(sources)
        source_counts = modules["util"].graph_element_counts(sources).cpu()
        candidate_ids = [stable_graph_sha256(graph) for graph in candidate_graphs]
        all_distance_rows: list[dict[str, Any]] = []
        eligible_rows: list[dict[str, Any]] = []
        pair_indices: list[tuple[int, int]] = []
        recourse_vectors: list[np.ndarray] = []
        normalized_values: list[float] = []
        for start in range(0, len(candidate_graphs), max(1, int(batch_size))):
            chunk = candidate_graphs[start : start + max(1, int(batch_size))]
            with torch.no_grad():
                distances = embedding.predict_outer_with_queries(chunk, batch_size=batch_size).cpu()
                candidate_embeddings = embedding.embed_model(
                    Batch.from_data_list(chunk).to(device)
                ).detach().cpu()
            candidate_counts = modules["util"].graph_element_counts(chunk).cpu()
            scale = candidate_counts[:, None] + source_counts[None, :]
            normalized = distances / scale
            for local_candidate in range(len(chunk)):
                candidate_index = start + local_candidate
                for source_index in range(len(sources)):
                    value = float(normalized[local_candidate, source_index])
                    normalized_values.append(value)
                    eligible = bool(math.isfinite(value) and value <= float(contract.theta))
                    row = {
                        "parent_index": source_index,
                        "parent_id": parent_ids[source_index],
                        "candidate_index": candidate_index,
                        "candidate_id": candidate_ids[candidate_index],
                        "normalized_distance": value,
                        "finite": math.isfinite(value),
                        "theta_eligible": eligible,
                    }
                    all_distance_rows.append(row)
                    if eligible:
                        pair_indices.append((source_index, candidate_index))
                        vector = (
                            candidate_embeddings[local_candidate] - source_embeddings[source_index]
                        ) / scale[local_candidate, source_index]
                        recourse_vectors.append(vector.numpy().astype(np.float64, copy=False))
                        eligible_rows.append(row)
    if expected_distance_pairs is not None and len(all_distance_rows) != int(expected_distance_pairs):
        raise ValueError("AIDS native distance-pair count differs from the frozen artifact evidence.")
    if expected_eligible_pairs is not None and len(pair_indices) != int(expected_eligible_pairs):
        raise ValueError("AIDS native theta-eligible count differs from the frozen artifact evidence.")
    geometry = audit_geometry(
        recourse_vectors=np.asarray(recourse_vectors, dtype=np.float64),
        pair_indices=pair_indices,
        parent_ids=parent_ids,
        candidate_ids=candidate_ids,
        contract=contract,
    )
    points = geometry["point_rows"]
    funnel = {
        "raw_generated_candidates": len(raw_candidates),
        "importance_ge_threshold_candidates": len(candidate_graphs),
        "distance_pairs": len(all_distance_rows),
        "finite_distance_pairs": sum(bool(row["finite"]) for row in all_distance_rows),
        "theta_eligible_pairs": len(pair_indices),
        "finite_recourse_embeddings": len(recourse_vectors),
        "dbscan_input_points": len(recourse_vectors),
        "dbscan_core_points": geometry["dbscan_core_points"],
        "dbscan_border_points": geometry["dbscan_border_points"],
        "dbscan_noise_points": geometry["dbscan_noise_points"],
        "dbscan_non_noise_clusters": geometry["dbscan_non_noise_clusters"],
        "centroid_radius_accepted_points": sum(
            int(row["num_points_with_distance_to_centroid_lt_delta"])
            for row in geometry["cluster_rows"]
        ),
        "centroid_norm_accepted_clusters": sum(
            bool(row["passed_centroid_norm_theta"]) for row in geometry["cluster_rows"]
        ),
        "nonempty_coverage_clusters": geometry["postfilter_cluster_count"],
        "greedy_selected_clusters": min(
            int(contract.recourse_size), geometry["postfilter_cluster_count"]
        ),
    }
    _write_csv(root / "funnel.csv", [{"stage": key, "count": value} for key, value in funnel.items()], ["stage", "count"])
    _write_csv(root / "eligible_pairs.csv", eligible_rows, list(eligible_rows[0]) if eligible_rows else ["parent_index", "parent_id", "candidate_index", "candidate_id", "normalized_distance", "finite", "theta_eligible"])
    recourse_rows = [
        {
            "point_id": index,
            "parent_id": parent_ids[parent_index],
            "candidate_id": candidate_ids[candidate_index],
            "recourse_norm": float(np.linalg.norm(recourse_vectors[index])),
        }
        for index, (parent_index, candidate_index) in enumerate(pair_indices)
    ]
    _write_csv(root / "recourse_points.csv", recourse_rows, ["point_id", "parent_id", "candidate_id", "recourse_norm"])
    point_fields = list(points[0]) if points else []
    _write_csv(root / "epsilon_neighbors.csv", points, point_fields)
    _write_csv(root / "dbscan_points.csv", points, point_fields)
    component_fields = list(geometry["component_rows"][0]) if geometry["component_rows"] else []
    _write_csv(root / "epsilon_components.csv", geometry["component_rows"], component_fields)
    cluster_fields = list(geometry["cluster_rows"][0]) if geometry["cluster_rows"] else ["cluster_label", "cluster_size"]
    _write_csv(root / "dbscan_clusters.csv", geometry["cluster_rows"], cluster_fields)
    _write_csv(root / "postfilter_clusters.csv", geometry["postfilter_rows"], cluster_fields)
    statistics = {
        "all_distance_pairs": distribution_summary(normalized_values),
        "theta_eligible_pairs": distribution_summary(
            [row["normalized_distance"] for row in eligible_rows]
        ),
        **geometry["statistics"],
    }
    write_json(root / "pairwise_distance_summary.json", statistics)
    source_sha = sha256_file(artifact)
    preregistration_sha = None
    if preregistration_path is not None:
        preregistration_sha = sha256_file(preregistration_path)
    audit = {
        "schema_version": 1,
        "audit_passed": True,
        "scientific_output_empty": geometry["postfilter_cluster_count"] == 0,
        "source_counterfactuals_path": str(artifact),
        "source_counterfactuals_sha256": source_sha,
        "source_counterfactuals_bytes": artifact.stat().st_size,
        "upstream_commit": UPSTREAM_COMMIT,
        "project_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "parent_count": len(parent_ids),
        "parent_ids_sha256": stable_json_sha256(parent_ids),
        "candidate_count": len(candidate_graphs),
        "candidate_set_sha256": stable_json_sha256(candidate_ids),
        "candidate_regeneration": False,
        "full_reject_parent_universe": full_reject_parent_universe,
        "upstream_contract": contract.__dict__,
        "original_smoke_contract_difference": {"cluster_size": 2},
        "funnel": funnel,
        "dbscan": {
            key: value
            for key, value in geometry.items()
            if key
            in {
                "dbscan_core_points",
                "dbscan_border_points",
                "dbscan_noise_points",
                "dbscan_non_noise_clusters",
                "postfilter_cluster_count",
                "num_points_with_neighbor_count_ge_min_samples",
                "num_connected_components",
                "largest_component_size",
                "component_size_histogram",
                "num_components_size_ge_min_samples",
                "num_components_with_core_point",
            }
        },
        "preregistration_path": str(Path(preregistration_path).resolve()) if preregistration_path else None,
        "preregistration_sha256": preregistration_sha,
        "used_for_final_metrics": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "audit.json", audit)
    write_json(
        root / "provenance.json",
        {
            "upstream_commit": UPSTREAM_COMMIT,
            "project_commit": audit["project_commit"],
            "source_artifact_sha256": source_sha,
            "candidate_set_sha256": audit["candidate_set_sha256"],
            "preregistration_sha256": preregistration_sha,
        },
    )
    (root / "audit.txt").write_text(
        "COMRECGC AIDS native DBSCAN audit\n"
        f"parents={len(parent_ids)} candidates={len(candidate_graphs)}\n"
        f"eligible_points={len(pair_indices)} core_points={geometry['dbscan_core_points']}\n"
        f"clusters={geometry['dbscan_non_noise_clusters']} postfilter={geometry['postfilter_cluster_count']}\n"
        f"largest_epsilon_component={geometry['largest_component_size']}\n"
        "[COMRECGC_AIDS_DBSCAN_AUDIT_PASS]\n",
        encoding="utf-8",
    )
    write_json(
        root / "manifest.json",
        {
            "run_complete": True,
            "audit_passed": True,
            "scientific_output_empty": audit["scientific_output_empty"],
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(root.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return audit
