"""Official common-recourse clustering with auditable graph medoids."""

from __future__ import annotations

import csv
import gc
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
from .external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalDBSCANContract,
    ExternalMemoryDBSCANError,
    fit_external_memory_dbscan,
)
from .external_component_summary import (
    summarize_proven_all_core_components_external,
)
from .external_memory_recourse import (
    ExternalPairStore,
    adopt_external_pair_store_read_only,
    invoke_official_coverage_summary_external,
    summarize_proven_one_cluster_external,
    trace_external_cluster_order,
)
from .external_pair_chunk_cache import (
    DEFAULT_LOCAL_FREE_FLOOR_BYTES,
    materialize_cartesian_chunk_vector_cache,
)
from .close_pair_view import (
    NORMALIZED_DISTANCE_CONTRACT,
    SCALE_CONTRACT,
    ThetaClosePairContract,
    validate_theta_close_pair_view,
)
from .project_dataset import (
    load_aids_generation_bundle,
    load_bace_generation_bundle,
    load_mutagenicity_generation_bundle,
)
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
                "selected_rank": int(rank),
                "cluster_label": cluster_label,
                "cluster_id": cluster_label,
                "cluster_center_norm": centroid_norms[cluster_label],
                "centroid_norm": centroid_norms[cluster_label],
                "cluster_radius": float(radius),
                "cluster_size": int(np.sum(labels == cluster_label)),
                "representative_source_index": source_index,
                "representative_counterfactual_index": counterfactual_index,
                "representative_distance_to_center": medoid_distance,
                "covered_parent_indices_native": sorted(filtered[cluster_label]),
                "native_cumulative_covered_count": len(covered),
                "cumulative_covered_count": len(covered),
                "native_cumulative_cost": cumulative_cost,
                "member_counterfactual_indices": sorted(cluster_cf_indices[cluster_label]),
                "representative_candidate_ids": [int(counterfactual_index)],
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
    engine: str = "legacy_in_memory",
    external_max_rss_bytes: int = 96 * 1024**3,
    external_query_block_size: int = 8,
    external_checkpoint_interval_blocks: int = 1,
    external_dbscan_shortcut_mode: str = "disabled",
    external_shortcut_seed_count: int = 3,
    external_shortcut_failure_cap: int = 4_096,
    external_shortcut_query_block_size: int = 65_536,
    external_exact_fallback_max_samples: int = 100_000,
    external_summary_block_size: int = 65_536,
    external_pair_store_source_manifest: str | Path | None = None,
    external_pair_store_source_checkpoint: str | Path | None = None,
    external_pair_store_source_owner_root: str | Path | None = None,
    external_close_pair_view_manifest: str | Path | None = None,
    external_vector_cache_root: str | Path | None = None,
    external_vector_cache_lock: str | Path | None = None,
    external_vector_cache_route_lock: str | Path | None = None,
    external_vector_cache_min_free_bytes: int = DEFAULT_LOCAL_FREE_FLOOR_BYTES,
    external_vector_cache_proc_root: str | Path = "/proc",
    expected_sklearn_version: str = "1.7.2",
) -> dict[str, Any]:
    parameters.validate(mode)
    if engine not in {"legacy_in_memory", "external_memory_exact_v1"}:
        raise ValueError(f"Unsupported common-recourse engine: {engine}")
    if engine == "external_memory_exact_v1" and (
        dataset != "aids" or device != "cpu"
    ):
        raise ValueError(
            "external_memory_exact_v1 is released only for CPU-backed AIDS"
        )
    chunk_source_requested = external_pair_store_source_checkpoint is not None
    if chunk_source_requested and external_close_pair_view_manifest is None:
        raise ValueError(
            "UNPROVEN_CARTESIAN_DBSCAN_INPUT: the physical candidate-by-parent "
            "snapshot is not bound to a validated theta-close logical view"
        )
    if (
        not chunk_source_requested
        and external_pair_store_source_manifest is None
        and external_close_pair_view_manifest is not None
    ):
        raise ValueError(
            "a theta-close view manifest requires an adopted physical pair source"
        )
    if (
        external_pair_store_source_manifest is not None or chunk_source_requested
    ) and engine != "external_memory_exact_v1":
        raise ValueError(
            "external pair-store adoption requires external_memory_exact_v1"
        )
    if external_pair_store_source_manifest is not None and chunk_source_requested:
        raise ValueError("terminal and chunk pair-store sources are mutually exclusive")
    chunk_source_values = (
        external_pair_store_source_checkpoint,
        external_vector_cache_root,
        external_vector_cache_lock,
        external_vector_cache_route_lock,
    )
    if any(value is not None for value in chunk_source_values) and not all(
        value is not None for value in chunk_source_values
    ):
        raise ValueError(
            "chunk-source checkpoint, local cache root, and lock are required together"
        )
    source_requested = (
        external_pair_store_source_manifest is not None or chunk_source_requested
    )
    if source_requested and external_pair_store_source_owner_root is None:
        raise ValueError("external pair-store source requires its old owner root")
    if not source_requested and external_pair_store_source_owner_root is not None:
        raise ValueError("pair-store owner root was provided without a source")
    root = require_empty_output(output_dir, resume=resume)
    if external_pair_store_source_manifest is not None or chunk_source_requested:
        if (
            str(external_dbscan_shortcut_mode)
            != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        ):
            raise ValueError(
                "read-only pair-store adoption requires the exact adaptive shortcut"
            )
        source_pair_artifact = Path(
            external_pair_store_source_manifest
            if external_pair_store_source_manifest is not None
            else external_pair_store_source_checkpoint  # type: ignore[arg-type]
        ).expanduser().resolve(strict=True)
        try:
            source_pair_artifact.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError("pair-store source must be outside the fresh output root")
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
    elif dataset == "mutagenicity":
        bundle = load_mutagenicity_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    elif dataset == "bace":
        bundle = load_bace_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    else:
        raise ValueError(f"Unsupported project dataset: {dataset}")
    payload_path = Path(generation_manifest["counterfactuals_path"]).resolve()
    payload_sha256 = sha256_file(payload_path)
    if payload_sha256 != generation_manifest["counterfactuals_sha256"]:
        raise ValueError("Generation counterfactual artifact SHA256 mismatch.")
    payload = _torch_load(payload_path)
    graph_map = payload.get("graph_map") or {}
    raw_candidates = list(payload.get("counterfactual_candidates") or [])
    candidate_graphs: list[Any] = []
    generation_indices: list[int] = []
    candidate_graph_hashes: list[str] = []
    for generation_index, candidate in enumerate(raw_candidates):
        importance = _importance_parts(candidate)
        graph_hash = candidate.get("graph_hash")
        if float(importance[0]) >= 0.5 and graph_hash in graph_map:
            candidate_graphs.append(graph_map[graph_hash][0])
            generation_indices.append(generation_index)
            candidate_graph_hashes.append(str(graph_hash))
        if len(candidate_graphs) >= int(parameters.cf_size):
            break
    if not candidate_graphs:
        raise RuntimeError("No model-counterfactual candidates are available for clustering.")
    # Keep only selected graph objects and their frozen order.  The 4.9 GiB
    # AIDS payload otherwise retains every raw candidate/map entry throughout
    # clustering and compounds DBSCAN's former neighborhood peak.
    del payload, graph_map, raw_candidates
    gc.collect()
    torch, Batch = _torch_stack()
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("COMRECGC common-recourse requires scikit-learn.") from exc
    with imported_upstream(upstream_root) as modules:
        pair_source_requested = (
            external_pair_store_source_manifest is not None
            or chunk_source_requested
        )
        embedding_model = None
        original_embeddings = None
        original_counts = None
        if not pair_source_requested:
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
        external_artifacts: dict[str, Any] | None = None
        official_audit: dict[str, Any] = {
            "official_coverage_summary_invoked": True,
            "official_coverage_semantics_derived_for_single_label_zero": False,
        }
        if engine == "external_memory_exact_v1":
            chunk_size = max(1, int(batch_size))
            dataset_audit = bundle.audit()
            pair_identity = {
                "schema_version": "comrecgc_external_pair_materialization_v1",
                "dataset": dataset,
                "mode": mode,
                "parameters": parameters.__dict__,
                "generation_manifest_sha256": sha256_file(
                    generation_root / "run_manifest.json"
                ),
                "counterfactuals_sha256": generation_manifest[
                    "counterfactuals_sha256"
                ],
                "distance_checkpoint_sha256": sha256_file(distance_checkpoint),
                "candidate_count": len(candidate_graphs),
                "candidate_graph_hashes_sha256": stable_json_sha256(
                    candidate_graph_hashes
                ),
                "generation_indices_sha256": stable_json_sha256(
                    generation_indices
                ),
                "parent_ids_sha256": stable_json_sha256(list(bundle.parent_ids)),
                "parent_count": len(bundle.graphs),
                "dataset_fingerprint": bundle.dataset_fingerprint,
                "dataset_audit": dataset_audit,
                "dataset_audit_sha256": stable_json_sha256(dataset_audit),
                "batch_size": chunk_size,
                "pair_order": "candidate_major_parent_minor",
                "vector_expression": "(candidate_embedding-parent_embedding)/(candidate_count+parent_count)",
                "device": "cpu",
            }
            expected_chunk_identities = []
            for chunk_index, start in enumerate(
                range(0, len(candidate_graphs), chunk_size)
            ):
                stop = min(len(candidate_graphs), start + chunk_size)
                expected_chunk_identities.append(
                    {
                        "chunk_index": chunk_index,
                        "candidate_start": start,
                        "candidate_stop": stop,
                        "candidate_graph_hashes_sha256": stable_json_sha256(
                            candidate_graph_hashes[start:stop]
                        ),
                        "generation_indices_sha256": stable_json_sha256(
                            generation_indices[start:stop]
                        ),
                    }
                )
            pair_adoption = None
            chunk_cache = None
            close_pair_view = None
            physical_pair_row_count = None
            pair_authority_manifest_path = None
            pair_authority_manifest_sha256 = None
            if chunk_source_requested:
                chunk_cache = materialize_cartesian_chunk_vector_cache(
                    source_checkpoint_path=external_pair_store_source_checkpoint,  # type: ignore[arg-type]
                    source_owner_root=external_pair_store_source_owner_root,  # type: ignore[arg-type]
                    persistent_root=root / "external_memory/chunk_vector_cache",
                    local_cache_root=external_vector_cache_root,  # type: ignore[arg-type]
                    scratch_lock_path=external_vector_cache_lock,  # type: ignore[arg-type]
                    route_lock_path=external_vector_cache_route_lock,
                    expected_scientific_identity=pair_identity,
                    expected_chunk_identities=expected_chunk_identities,
                    parent_count=len(bundle.graphs),
                    candidate_count=len(candidate_graphs),
                    min_local_free_bytes=int(external_vector_cache_min_free_bytes),
                    proc_root=external_vector_cache_proc_root,
                    resume=resume,
                )
                pair_indices_external = chunk_cache.pairs
                physical_pair_row_count = int(chunk_cache.row_count)
                close_pair_view = validate_theta_close_pair_view(
                    external_close_pair_view_manifest,  # type: ignore[arg-type]
                    expected_contract=ThetaClosePairContract(
                        theta=float(parameters.theta),
                        parent_count=len(bundle.graphs),
                        candidate_count=len(candidate_graphs),
                        distance_checkpoint_sha256=sha256_file(distance_checkpoint),
                        embedding_checkpoint_sha256=sha256_file(distance_checkpoint),
                        scale_contract=SCALE_CONTRACT,
                        normalized_distance_contract=NORMALIZED_DISTANCE_CONTRACT,
                    ),
                    expected_physical_vectors_path=chunk_cache.vectors_path,
                    expected_physical_vectors_sha256=chunk_cache.vectors_sha256,
                    require_pair_semantics_authority=True,
                )
                if not close_pair_view.eligible_for_dbscan:
                    raise ExternalMemoryDBSCANError(
                        close_pair_view.blocking_reason
                        or "CLOSE_VIEW_NOT_DBSCAN_ELIGIBLE"
                    )
                if (
                    close_pair_view.all_pairs_close
                    and close_pair_view.pairs_sha256
                    != chunk_cache.pairs.logical_npy_sha256
                ):
                    raise ExternalMemoryDBSCANError(
                        "CLOSE_VIEW_PHYSICAL_PAIR_IDENTITY_MISMATCH"
                    )
                pair_indices_external = (
                    chunk_cache.pairs
                    if close_pair_view.pairs_path is None
                    else np.load(
                        close_pair_view.pairs_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                )
                recourse_array_external = np.load(
                    close_pair_view.vectors_path, mmap_mode="r", allow_pickle=False
                )
                pair_row_count = close_pair_view.logical_close_rows
                pair_indices_sha256 = close_pair_view.pairs_sha256
                recourse_vectors_sha256 = close_pair_view.vectors_sha256
                recourse_vectors_path = close_pair_view.vectors_path
                pair_manifest_path = chunk_cache.manifest_path
                pair_manifest_sha256 = chunk_cache.manifest_sha256
                pair_authority_manifest_path = close_pair_view.manifest_path
                pair_authority_manifest_sha256 = close_pair_view.manifest_sha256
            elif external_pair_store_source_manifest is not None:
                pair_adoption = adopt_external_pair_store_read_only(
                    source_manifest_path=external_pair_store_source_manifest,
                    source_owner_root=external_pair_store_source_owner_root,
                    adoption_root=root / "external_memory/pair_store_adoption",
                    expected_scientific_identity=pair_identity,
                    resume=resume,
                )
                pair_result = pair_adoption.pair_store
                physical_pair_row_count = int(pair_result.row_count)
                terminal_is_cartesian = physical_pair_row_count == (
                    len(candidate_graphs) * len(bundle.graphs)
                )
                if terminal_is_cartesian and external_close_pair_view_manifest is None:
                    raise ExternalMemoryDBSCANError(
                        "UNPROVEN_CARTESIAN_DBSCAN_INPUT: terminal physical pair "
                        "store requires a validated theta-close logical view"
                    )
                if external_close_pair_view_manifest is not None:
                    close_pair_view = validate_theta_close_pair_view(
                        external_close_pair_view_manifest,
                        expected_contract=ThetaClosePairContract(
                            theta=float(parameters.theta),
                            parent_count=len(bundle.graphs),
                            candidate_count=len(candidate_graphs),
                            distance_checkpoint_sha256=sha256_file(
                                distance_checkpoint
                            ),
                            embedding_checkpoint_sha256=sha256_file(
                                distance_checkpoint
                            ),
                            scale_contract=SCALE_CONTRACT,
                            normalized_distance_contract=(
                                NORMALIZED_DISTANCE_CONTRACT
                            ),
                        ),
                        expected_physical_vectors_path=pair_result.vectors_path,
                        expected_physical_vectors_sha256=pair_result.vectors_sha256,
                        require_pair_semantics_authority=True,
                    )
                    if not close_pair_view.eligible_for_dbscan:
                        raise ExternalMemoryDBSCANError(
                            close_pair_view.blocking_reason
                            or "CLOSE_VIEW_NOT_DBSCAN_ELIGIBLE"
                        )
                    if (
                        close_pair_view.all_pairs_close
                        and close_pair_view.pairs_sha256
                        != pair_result.pairs_sha256
                    ):
                        raise ExternalMemoryDBSCANError(
                            "CLOSE_VIEW_PHYSICAL_PAIR_IDENTITY_MISMATCH"
                        )
                    pair_indices_external = (
                        np.load(
                            pair_result.pairs_path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                        if close_pair_view.pairs_path is None
                        else np.load(
                            close_pair_view.pairs_path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                    )
                    recourse_array_external = np.load(
                        close_pair_view.vectors_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    pair_row_count = close_pair_view.logical_close_rows
                    pair_indices_sha256 = close_pair_view.pairs_sha256
                    recourse_vectors_sha256 = close_pair_view.vectors_sha256
                    recourse_vectors_path = close_pair_view.vectors_path
                    pair_manifest_path = pair_result.manifest_path
                    pair_manifest_sha256 = pair_result.manifest_sha256
                    pair_authority_manifest_path = close_pair_view.manifest_path
                    pair_authority_manifest_sha256 = (
                        close_pair_view.manifest_sha256
                    )
                else:
                    pair_indices_external = np.load(
                        pair_result.pairs_path, mmap_mode="r", allow_pickle=False
                    )
                    recourse_array_external = np.load(
                        pair_result.vectors_path, mmap_mode="r", allow_pickle=False
                    )
                    pair_row_count = pair_result.row_count
                    pair_indices_sha256 = pair_result.pairs_sha256
                    recourse_vectors_sha256 = pair_result.vectors_sha256
                    recourse_vectors_path = pair_result.vectors_path
                    pair_manifest_path = pair_result.manifest_path
                    pair_manifest_sha256 = pair_result.manifest_sha256
            else:
                pair_store = ExternalPairStore(
                    root=root / "external_memory/pair_store",
                    scientific_identity=pair_identity,
                    max_rss_bytes=int(external_max_rss_bytes),
                    resume=resume,
                )
                if pair_store.complete:
                    pair_result = pair_store.finalize()
                else:
                    if embedding_model is None or original_embeddings is None or original_counts is None:
                        raise RuntimeError("pair materialization model state is unavailable")
                    for chunk_identity in expected_chunk_identities:
                        chunk_index = int(chunk_identity["chunk_index"])
                        start = int(chunk_identity["candidate_start"])
                        stop = int(chunk_identity["candidate_stop"])
                        if chunk_index < pair_store.next_chunk_index:
                            pair_store.verify_completed_chunk(
                                chunk_index=chunk_index,
                                chunk_identity=chunk_identity,
                            )
                            continue
                        if chunk_index != pair_store.next_chunk_index:
                            raise RuntimeError("External pair-store checkpoint has a gap.")
                        chunk = candidate_graphs[start:stop]
                        with torch.no_grad():
                            raw_distances = embedding_model.predict_outer_with_queries(
                                chunk, batch_size=batch_size
                            ).cpu()
                            chunk_embeddings = embedding_model.embed_model(
                                Batch.from_data_list(chunk).to(device)
                            ).detach().cpu()
                        chunk_counts = modules["util"].graph_element_counts(chunk).cpu()
                        scale = chunk_counts[:, None] + original_counts[None, :]
                        normalized = raw_distances / scale
                        valid_pairs = torch.nonzero(
                            normalized <= float(parameters.theta), as_tuple=False
                        )
                        pair_rows: list[tuple[int, int]] = []
                        vector_rows: list[np.ndarray] = []
                        for local_cf, original_index in valid_pairs.tolist():
                            global_cf = start + int(local_cf)
                            pair_rows.append((int(original_index), global_cf))
                            vector = (
                                (
                                    chunk_embeddings[int(local_cf)]
                                    - original_embeddings[int(original_index)]
                                )
                                / scale[int(local_cf), int(original_index)]
                            )
                            vector_rows.append(vector.numpy())
                        pairs_chunk = np.asarray(pair_rows, dtype=np.int64).reshape(-1, 2)
                        if vector_rows:
                            vectors_chunk = np.asarray(vector_rows)
                        else:
                            vectors_chunk = np.empty(
                                (0, int(original_embeddings.shape[1])),
                                dtype=original_embeddings.numpy().dtype,
                            )
                        pair_store.append(
                            chunk_index=chunk_index,
                            pairs=pairs_chunk,
                            vectors=vectors_chunk,
                            chunk_identity=chunk_identity,
                        )
                    pair_result = pair_store.finalize()
                pair_indices_external = np.load(
                    pair_result.pairs_path, mmap_mode="r", allow_pickle=False
                )
                recourse_array_external = np.load(
                    pair_result.vectors_path, mmap_mode="r", allow_pickle=False
                )
                pair_row_count = pair_result.row_count
                pair_indices_sha256 = pair_result.pairs_sha256
                recourse_vectors_sha256 = pair_result.vectors_sha256
                recourse_vectors_path = pair_result.vectors_path
                pair_manifest_path = pair_result.manifest_path
                pair_manifest_sha256 = pair_result.manifest_sha256
            distance_pair_count = len(candidate_graphs) * len(bundle.graphs)
            if pair_row_count:
                dbscan_result = fit_external_memory_dbscan(
                    vectors_path=recourse_vectors_path,
                    work_dir=root / "external_memory/dbscan",
                    contract=ExternalDBSCANContract(
                        eps=float(parameters.delta),
                        min_samples=int(parameters.cluster_size),
                        query_block_size=int(external_query_block_size),
                        checkpoint_interval_blocks=int(
                            external_checkpoint_interval_blocks
                        ),
                        max_rss_bytes=int(external_max_rss_bytes),
                        expected_sklearn_version=expected_sklearn_version,
                        shortcut_mode=str(external_dbscan_shortcut_mode),
                        shortcut_seed_count=int(external_shortcut_seed_count),
                        shortcut_failure_cap=int(external_shortcut_failure_cap),
                        shortcut_query_block_size=int(
                            external_shortcut_query_block_size
                        ),
                        exact_fallback_max_samples=int(
                            external_exact_fallback_max_samples
                        ),
                    ),
                    expected_vectors_sha256=recourse_vectors_sha256,
                    resume=resume,
                )
                cluster_labels = np.load(
                    dbscan_result.labels_path, mmap_mode="r", allow_pickle=False
                )
                one_cluster_summary_manifest = None
                one_cluster_summary_manifest_sha256 = None
                all_core_component_summary_manifest = None
                all_core_component_summary_manifest_sha256 = None
                completed_dbscan_manifest = json_load(dbscan_result.manifest_path)
                if (
                    completed_dbscan_manifest.get("clustering_path")
                    == ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
                ):
                    exact_summary = summarize_proven_all_core_components_external(
                        work_dir=(
                            root / "external_memory/all_core_component_summary"
                        ),
                        dbscan_manifest_path=dbscan_result.manifest_path,
                        dbscan_manifest_sha256=dbscan_result.manifest_sha256,
                        labels=cluster_labels,
                        recourse_vectors=recourse_array_external,
                        pair_indices=pair_indices_external,
                        pairs_sha256=pair_indices_sha256,
                        pair_authority_manifest_path=pair_authority_manifest_path,
                        pair_authority_manifest_sha256=(
                            pair_authority_manifest_sha256
                        ),
                        radius=float(parameters.delta),
                        theta=float(parameters.theta),
                        recourse_size=int(parameters.recourse_size),
                        official_greedy=modules[
                            "common_recourse"
                        ].greedy_counterfactual_summary_from_covering_sets,
                        torch_module=torch,
                        max_rss_bytes=int(external_max_rss_bytes),
                        block_size=int(external_summary_block_size),
                        resume=resume,
                    )
                    official_result = exact_summary.official_result
                    selected = exact_summary.selected
                    summary_manifest = json_load(exact_summary.manifest_path)
                    official_audit = {
                        "schema_version": summary_manifest["schema_version"],
                        "official_coverage_summary_invoked": False,
                        "official_coverage_semantics_streamed_for_all_components": True,
                        "cluster_count": summary_manifest["cluster_count"],
                        "peak_rss_bytes_observed": summary_manifest[
                            "peak_rss_bytes_observed"
                        ],
                    }
                    trace_audit = {
                        "schema_version": summary_manifest["schema_version"],
                        "selected_count": len(selected),
                        "official_mask_is_primary": True,
                        "numpy_and_float64_are_audit_only": True,
                        "official_greedy_invoked": summary_manifest[
                            "official_greedy_invoked"
                        ],
                        "peak_rss_bytes_observed": summary_manifest[
                            "peak_rss_bytes_observed"
                        ],
                    }
                    all_core_component_summary_manifest = str(
                        exact_summary.manifest_path
                    )
                    all_core_component_summary_manifest_sha256 = (
                        exact_summary.manifest_sha256
                    )
                elif (
                    dbscan_result.shortcut_proof_path is not None
                    and dbscan_result.cluster_count == 1
                ):
                    exact_summary = summarize_proven_one_cluster_external(
                        work_dir=root / "external_memory/one_cluster_summary",
                        dbscan_manifest_path=dbscan_result.manifest_path,
                        dbscan_manifest_sha256=dbscan_result.manifest_sha256,
                        recourse_vectors=recourse_array_external,
                        pair_indices=pair_indices_external,
                        pairs_sha256=pair_indices_sha256,
                        pair_authority_manifest_path=pair_authority_manifest_path,
                        pair_authority_manifest_sha256=(
                            pair_authority_manifest_sha256
                        ),
                        radius=float(parameters.delta),
                        theta=float(parameters.theta),
                        recourse_size=int(parameters.recourse_size),
                        official_greedy=modules[
                            "common_recourse"
                        ].greedy_counterfactual_summary_from_covering_sets,
                        torch_module=torch,
                        max_rss_bytes=int(external_max_rss_bytes),
                        block_size=int(external_summary_block_size),
                        resume=resume,
                    )
                    official_result = exact_summary.official_result
                    selected = exact_summary.selected
                    summary_manifest = json_load(exact_summary.manifest_path)
                    official_audit = {
                        "schema_version": summary_manifest["schema_version"],
                        "official_coverage_summary_invoked": False,
                        "official_coverage_semantics_derived_for_single_label_zero": True,
                        "legacy_torch_reduction_order_preserved": True,
                        "peak_rss_bytes_observed": summary_manifest[
                            "peak_rss_bytes_observed"
                        ],
                    }
                    trace_audit = {
                        "schema_version": summary_manifest["schema_version"],
                        "selected_count": len(selected),
                        "legacy_numpy_reduction_order_preserved": True,
                        "official_greedy_invoked": summary_manifest[
                            "official_greedy_invoked_for_trace"
                        ],
                        "peak_rss_bytes_observed": summary_manifest[
                            "peak_rss_bytes_observed"
                        ],
                    }
                    one_cluster_summary_manifest = str(exact_summary.manifest_path)
                    one_cluster_summary_manifest_sha256 = (
                        exact_summary.manifest_sha256
                    )
                else:
                    official_result, official_audit = (
                        invoke_official_coverage_summary_external(
                            labels=cluster_labels,
                            recourse_vectors=recourse_array_external,
                            pair_indices=pair_indices_external,
                            radius=float(parameters.delta),
                            theta=float(parameters.theta),
                            recourse_size=int(parameters.recourse_size),
                            official_coverage_summary=modules[
                                "common_recourse"
                            ].coverage_summary,
                            torch_module=torch,
                            max_rss_bytes=int(external_max_rss_bytes),
                        )
                    )
                    selected, trace_audit = trace_external_cluster_order(
                        labels=cluster_labels,
                        recourse_vectors=recourse_array_external,
                        pair_indices=pair_indices_external,
                        radius=float(parameters.delta),
                        theta=float(parameters.theta),
                        recourse_size=int(parameters.recourse_size),
                        official_greedy=modules[
                            "common_recourse"
                        ].greedy_counterfactual_summary_from_covering_sets,
                        max_rss_bytes=int(external_max_rss_bytes),
                    )
                dbscan_manifest = str(dbscan_result.manifest_path)
                dbscan_manifest_sha256 = dbscan_result.manifest_sha256
            else:
                official_result = ([], [], [])
                selected = []
                cluster_labels = np.asarray([], dtype=np.intp)
                official_audit = {
                    "official_coverage_summary_invoked": False,
                    "reason": "no_theta_eligible_pairs",
                }
                trace_audit = {
                    "selected_count": 0,
                    "reason": "no_theta_eligible_pairs",
                }
                dbscan_manifest = None
                dbscan_manifest_sha256 = None
                one_cluster_summary_manifest = None
                one_cluster_summary_manifest_sha256 = None
                all_core_component_summary_manifest = None
                all_core_component_summary_manifest_sha256 = None
            external_artifacts = {
                "engine": engine,
                "pair_store_manifest": str(pair_manifest_path),
                "pair_store_manifest_sha256": pair_manifest_sha256,
                "pair_store_scientific_identity_sha256": stable_json_sha256(
                    pair_identity
                ),
                "pair_store_adopted_read_only": pair_adoption is not None,
                "pair_chunks_adopted_read_only": chunk_cache is not None,
                "physical_pair_count": physical_pair_row_count,
                "logical_close_pair_count": pair_row_count,
                "dbscan_input_count": pair_row_count,
                "dbscan_input": "theta_close_recourse_only",
                "close_pair_view_manifest": (
                    None
                    if close_pair_view is None
                    else str(close_pair_view.manifest_path)
                ),
                "close_pair_view_manifest_sha256": (
                    None
                    if close_pair_view is None
                    else close_pair_view.manifest_sha256
                ),
                "pair_indices_materialized": (
                    chunk_cache is None
                    or (
                        close_pair_view is not None
                        and close_pair_view.pairs_path is not None
                    )
                ),
                "pair_indices_formula": (
                    None
                    if chunk_cache is None
                    or (
                        close_pair_view is not None
                        and close_pair_view.pairs_path is not None
                    )
                    else "parent=row%parent_count;candidate=row//parent_count"
                ),
                "chunk_vector_cache_manifest": (
                    None if chunk_cache is None else str(chunk_cache.manifest_path)
                ),
                "chunk_vector_cache_manifest_sha256": (
                    None if chunk_cache is None else chunk_cache.manifest_sha256
                ),
                "local_vector_cache_is_scientific_authority": False,
                "pair_store_adoption_manifest": (
                    None
                    if pair_adoption is None
                    else str(pair_adoption.adoption_manifest_path)
                ),
                "pair_store_adoption_manifest_sha256": (
                    None
                    if pair_adoption is None
                    else pair_adoption.adoption_manifest_sha256
                ),
                "pair_indices_sha256": pair_indices_sha256,
                "recourse_vectors_sha256": recourse_vectors_sha256,
                "dbscan_manifest": dbscan_manifest,
                "dbscan_manifest_sha256": dbscan_manifest_sha256,
                "one_cluster_summary_manifest": one_cluster_summary_manifest,
                "one_cluster_summary_manifest_sha256": (
                    one_cluster_summary_manifest_sha256
                ),
                "all_core_component_summary_manifest": (
                    all_core_component_summary_manifest
                ),
                "all_core_component_summary_manifest_sha256": (
                    all_core_component_summary_manifest_sha256
                ),
                "official_coverage_audit": official_audit,
                "trace_audit": trace_audit,
                "max_rss_bytes": int(external_max_rss_bytes),
                "resume_enabled": bool(resume),
                "dbscan_shortcut_mode": str(external_dbscan_shortcut_mode),
                "adaptive_shortcut_requested": (
                    str(external_dbscan_shortcut_mode)
                    == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
                ),
            }
            theta_eligible_pair_count = pair_row_count
        else:
            pair_indices: list[tuple[int, int]] = []
            recourse_vectors: list[np.ndarray] = []
            distance_pair_count = 0
            for start in range(0, len(candidate_graphs), max(1, int(batch_size))):
                chunk = candidate_graphs[start : start + max(1, int(batch_size))]
                with torch.no_grad():
                    raw_distances = embedding_model.predict_outer_with_queries(
                        chunk, batch_size=batch_size
                    ).cpu()
                    chunk_embeddings = embedding_model.embed_model(
                        Batch.from_data_list(chunk).to(device)
                    ).detach().cpu()
                chunk_counts = modules["util"].graph_element_counts(chunk).cpu()
                scale = chunk_counts[:, None] + original_counts[None, :]
                normalized = raw_distances / scale
                valid_pairs = torch.nonzero(
                    normalized <= float(parameters.theta), as_tuple=False
                )
                distance_pair_count += int(normalized.numel())
                for local_cf, original_index in valid_pairs.tolist():
                    global_cf = start + int(local_cf)
                    pair_indices.append((int(original_index), global_cf))
                    vector = (
                        (
                            chunk_embeddings[int(local_cf)]
                            - original_embeddings[int(original_index)]
                        )
                        / scale[int(local_cf), int(original_index)]
                    )
                    recourse_vectors.append(vector.numpy())
            if recourse_vectors:
                recourse_array = np.asarray(recourse_vectors)
                if not np.isfinite(recourse_array).all():
                    raise RuntimeError("Recourse embeddings contain NaN/Inf.")
                clustering = DBSCAN(
                    eps=float(parameters.delta),
                    min_samples=int(parameters.cluster_size),
                )
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
            theta_eligible_pair_count = len(pair_indices)
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
        "oracle_backend": generation_manifest.get("oracle_backend"),
        "classifier_family": generation_manifest.get("classifier_family"),
        "rf_oracle_used": generation_manifest.get("rf_oracle_used"),
        "oracle_checkpoint_hash": generation_manifest.get(
            "oracle_checkpoint_hash"
        ),
        "eligible_for_bace_gnn_main_results": bool(
            dataset == "bace"
            and generation_manifest.get("eligible_for_bace_gnn_main_results") is True
            and generation_manifest.get("oracle_backend") == "gnn"
            and generation_manifest.get("classifier_family") == "gine"
            and generation_manifest.get("rf_oracle_used") is False
        ),
        "counterfactuals_sha256": payload_sha256,
        "model_counterfactual_candidate_count": len(candidate_graphs),
        "distance_pair_count": distance_pair_count,
        "theta_eligible_pair_count": theta_eligible_pair_count,
        "dbscan_cluster_count": len({int(value) for value in cluster_labels if int(value) >= 0}),
        "dbscan_noise_point_count": int(np.count_nonzero(cluster_labels < 0)),
        "common_recourse_count": len(output_rows),
        "scientific_output_empty": not bool(output_rows),
        "execution_status": (
            "SCIENTIFIC_OUTPUT_EMPTY" if not output_rows else "FULL_EXECUTION_PASS"
        ),
        "native_cost": None if not output_rows else output_rows[-1]["native_cost"],
        "official_coverage_summary_invoked": bool(
            official_audit.get("official_coverage_summary_invoked")
        ),
        "official_coverage_semantics_derived_for_single_label_zero": bool(
            official_audit.get(
                "official_coverage_semantics_derived_for_single_label_zero"
            )
        ),
        "official_coverage_summary_result": [list(value) for value in official_result],
        "official_greedy_order_preserved": True,
        "embedding_centers_exported_as_graphs": False,
        "representative_policy": "real_pair_nearest_cluster_center",
        "common_recourse_engine": engine,
        "external_memory_artifacts": external_artifacts,
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
    closure_files = {
        "run_manifest.json": sha256_file(root / "run_manifest.json"),
        "selected_common_recourses.json": sha256_file(
            root / "selected_common_recourses.json"
        ),
        "selected_common_recourses.csv": sha256_file(
            root / "selected_common_recourses.csv"
        ),
        "representative_counterfactuals.pt": sha256_file(
            root / "representative_counterfactuals.pt"
        ),
    }
    if external_artifacts is not None:
        if external_artifacts.get("pair_chunks_adopted_read_only") is True:
            closure_files[
                "external_memory/chunk_vector_cache/run_manifest.json"
            ] = str(external_artifacts["chunk_vector_cache_manifest_sha256"])
        elif external_artifacts.get("pair_store_adopted_read_only") is True:
            closure_files[
                "external_memory/pair_store_adoption/run_manifest.json"
            ] = str(external_artifacts["pair_store_adoption_manifest_sha256"])
        else:
            closure_files["external_memory/pair_store/run_manifest.json"] = str(
                external_artifacts["pair_store_manifest_sha256"]
            )
        if external_artifacts.get("dbscan_manifest") is not None:
            closure_files["external_memory/dbscan/run_manifest.json"] = str(
                external_artifacts["dbscan_manifest_sha256"]
            )
        if external_artifacts.get("one_cluster_summary_manifest") is not None:
            closure_files[
                "external_memory/one_cluster_summary/run_manifest.json"
            ] = str(external_artifacts["one_cluster_summary_manifest_sha256"])
        if (
            external_artifacts.get("all_core_component_summary_manifest")
            is not None
        ):
            closure_files[
                "external_memory/all_core_component_summary/run_manifest.json"
            ] = str(
                external_artifacts[
                    "all_core_component_summary_manifest_sha256"
                ]
            )
    write_json(
        root / "_RUN_COMPLETE.json",
        {
            "schema_version": "comrecgc_common_recourse_terminal_v2",
            "run_complete": True,
            "common_recourse_engine": engine,
            "artifact_sha256": closure_files,
        },
    )
    return manifest


def json_load(path: Path) -> dict[str, Any]:
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
