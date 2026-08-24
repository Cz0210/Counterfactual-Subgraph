"""Audit and materialize the production AIDS COMRECGC theta-close pair view.

The command is intentionally CPU-only.  It adopts the completed physical pair
store read-only, reloads the exact frozen AIDS GREED checkpoint and candidate
order, and obtains every scalar distance through
``AIDSGreedEmbeddingAdapter.predict_outer_with_queries``.  It never treats a
recourse-vector norm as the distance authority and never copies the 25 GB pair
arrays.

Run with ``python -m src.baselines.comrecgc.aids_pair_semantics --help``.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import shutil
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .close_pair_scan import (
    ClosePairScanError,
    PairChunk,
    normalize_distance_block,
    pair_for_cartesian_row,
    recourse_vector_formula,
    scan_theta_close_pairs,
    utc_now,
)
from .contracts import (
    RecourseParameters,
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
)
from .model_adapter import AIDSGreedEmbeddingAdapter
from .project_dataset import load_aids_generation_bundle
from .recourse import _importance_parts, _torch_load, _torch_stack
from .upstream import imported_upstream, read_upstream_commit, validate_upstream_checkout


AIDS_PAIR_SEMANTICS_SCHEMA = "aids_comrecgc_pair_semantics_audit_v1"
ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA = "comrecgc_all_pairs_close_certificate_v1"
EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
EXPECTED_PHYSICAL_PAIR_COUNT = 91_916_686
EXPECTED_THETA = 0.1


@dataclass(frozen=True)
class CandidateSelection:
    graphs: list[Any]
    graph_hashes: list[str]
    generation_indices: list[int]
    raw_candidate_count: int
    classifier_pass_count: int
    graph_resolved_classifier_pass_count: int
    cap: int


class AIDSPairSemanticsError(RuntimeError):
    """Raised when production provenance or recomputed semantics fail closed."""


def _resolve_pair_chunk_authority(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[list[PairChunk], dict[str, Any]]:
    chunks_raw = manifest.get("chunks")
    if not isinstance(chunks_raw, list):
        raise AIDSPairSemanticsError("pair-store manifest lacks chunks")
    if chunks_raw:
        if int(manifest.get("chunk_count", -1)) != len(chunks_raw):
            raise AIDSPairSemanticsError("pair-store chunk count differs")
        return [PairChunk.from_manifest(row) for row in chunks_raw], {
            "physical_snapshot": False,
            "chunk_metadata_manifest": str(manifest_path),
            "chunk_metadata_manifest_sha256": sha256_file(manifest_path),
            "source_chunk_count": len(chunks_raw),
            "source_chunks_sha256": stable_json_sha256(chunks_raw),
        }
    if manifest.get("physical_snapshot") is not True:
        raise AIDSPairSemanticsError(
            "empty chunk metadata is allowed only for a physical snapshot wrapper"
        )
    if int(manifest.get("chunk_count", -1)) != 0:
        raise AIDSPairSemanticsError("physical snapshot wrapper chunk_count must be zero")
    source_value = manifest.get("source_manifest_path")
    if not isinstance(source_value, str) or not source_value:
        raise AIDSPairSemanticsError(
            "physical snapshot source manifest path is missing"
        )
    source_unresolved = Path(source_value).expanduser()
    if not source_unresolved.is_absolute():
        raise AIDSPairSemanticsError(
            "physical snapshot source manifest path must be absolute"
        )
    if source_unresolved.is_symlink():
        raise AIDSPairSemanticsError(
            "physical snapshot source manifest must not be a symlink"
        )
    source_lstat = source_unresolved.lstat()
    if not stat.S_ISREG(source_lstat.st_mode):
        raise AIDSPairSemanticsError(
            "physical snapshot source manifest is not a file"
        )
    source_path = source_unresolved.resolve(strict=True)
    source_hash = sha256_file(source_path)
    if source_hash != manifest.get("source_manifest_sha256"):
        raise AIDSPairSemanticsError("physical snapshot source manifest SHA256 differs")
    if source_path == manifest_path:
        raise AIDSPairSemanticsError("physical snapshot source manifest is recursive")
    source = _json_load(source_path)
    source_chunks = source.get("chunks")
    if not isinstance(source_chunks, list) or not source_chunks:
        raise AIDSPairSemanticsError(
            "physical snapshot source lacks closed chunk metadata"
        )
    source_chunks_hash = stable_json_sha256(source_chunks)
    binding_checks = {
        "source_run_complete": source.get("run_complete") is True,
        "source_chunk_count": int(manifest.get("source_chunk_count", -1))
        == len(source_chunks)
        == int(source.get("chunk_count", -1)),
        "source_chunks_sha256": manifest.get("source_chunks_sha256")
        == source_chunks_hash,
        "scientific_identity": source.get("scientific_identity")
        == manifest.get("scientific_identity"),
        "row_count": source.get("row_count") == manifest.get("row_count"),
        "pairs_sha256": source.get("pairs_sha256")
        == manifest.get("pairs_sha256"),
        "vectors_sha256": source.get("vectors_sha256")
        == manifest.get("vectors_sha256"),
        "pair_order": source.get("candidate_major_parent_minor_order") is True,
    }
    failed = [key for key, passed in binding_checks.items() if not passed]
    if failed:
        raise AIDSPairSemanticsError(
            "physical snapshot source binding failed: " + ", ".join(failed)
        )
    return [PairChunk.from_manifest(row) for row in source_chunks], {
        "physical_snapshot": True,
        "physical_snapshot_schema": manifest.get("physical_snapshot_schema"),
        "chunk_metadata_manifest": str(source_path),
        "chunk_metadata_manifest_sha256": source_hash,
        "source_chunk_count": len(source_chunks),
        "source_chunks_sha256": source_chunks_hash,
    }


def _json_load(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AIDSPairSemanticsError(f"expected JSON object: {source}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def _file_stat(path: str | Path) -> dict[str, int]:
    value = Path(path).expanduser().resolve(strict=True).stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "mode": int(value.st_mode),
    }


def _current_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()


@contextmanager
def _exclusive_audit_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AIDSPairSemanticsError(
                f"another pair-semantics audit owns {path}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _progress(path: Path, *, state: str, **values: Any) -> None:
    payload = {
        "schema_version": AIDS_PAIR_SEMANTICS_SCHEMA,
        "state": state,
        "pid": os.getpid(),
        "updated_at": utc_now(),
        **values,
    }
    _atomic_json(path, payload)
    summary = " ".join(
        f"{key}={payload[key]}"
        for key in ("state", "rows_processed", "rolling_throughput_rows_per_second", "eta_seconds")
        if key in payload
    )
    print(f"[AIDS_CLOSE_PAIR_AUDIT] {summary}", flush=True)


def _write_terminal_pass(
    *, root: Path, progress_path: Path, final: Mapping[str, Any]
) -> None:
    """Write terminal metadata first and the exact PASS sentinel last."""

    _atomic_json(root / "pair_semantics_audit.json", final)
    _progress(
        progress_path,
        state="PASS",
        rows_processed=int(final["physical_store_rows"]),
        logical_close_pair_count=int(final["logical_close_rows"]),
    )
    atomic_write_bytes(root / "PASS", b"PASS\n")


def _select_candidates(
    payload: Mapping[str, Any], *, cap: int
) -> CandidateSelection:
    graph_map = payload.get("graph_map") or {}
    raw_candidates = list(payload.get("counterfactual_candidates") or [])
    classifier_pass = 0
    resolved_pass = 0
    graphs: list[Any] = []
    graph_hashes: list[str] = []
    generation_indices: list[int] = []
    for generation_index, candidate in enumerate(raw_candidates):
        importance = _importance_parts(candidate)
        graph_hash = candidate.get("graph_hash")
        passed = bool(importance and float(importance[0]) >= 0.5)
        if passed:
            classifier_pass += 1
        if passed and graph_hash in graph_map:
            resolved_pass += 1
            if len(graphs) < int(cap):
                graphs.append(graph_map[graph_hash][0])
                graph_hashes.append(str(graph_hash))
                generation_indices.append(generation_index)
    return CandidateSelection(
        graphs=graphs,
        graph_hashes=graph_hashes,
        generation_indices=generation_indices,
        raw_candidate_count=len(raw_candidates),
        classifier_pass_count=classifier_pass,
        graph_resolved_classifier_pass_count=resolved_pass,
        cap=int(cap),
    )


def _validate_pair_store(
    *,
    manifest_path: Path,
    generation_manifest_path: Path,
    generation_manifest: Mapping[str, Any],
    distance_checkpoint: Path,
    selection: CandidateSelection,
    parent_ids: Sequence[str],
    theta: float,
) -> tuple[dict[str, Any], list[PairChunk]]:
    manifest = _json_load(manifest_path)
    scientific = manifest.get("scientific_identity")
    if not isinstance(scientific, dict):
        raise AIDSPairSemanticsError("pair-store manifest lacks scientific_identity")
    chunks, chunk_authority = _resolve_pair_chunk_authority(
        manifest_path, manifest
    )
    expected_parameters = asdict(RecourseParameters.for_mode("full"))
    failures: list[str] = []
    checks = {
        "run_complete": manifest.get("run_complete") is True,
        "candidate_major_parent_minor_order": manifest.get(
            "candidate_major_parent_minor_order"
        )
        is True,
        "physical_row_count": int(manifest.get("row_count", -1))
        == EXPECTED_PHYSICAL_PAIR_COUNT,
        "candidate_count": int(scientific.get("candidate_count", -1))
        == len(selection.graphs),
        "parent_count": int(scientific.get("parent_count", -1)) == len(parent_ids),
        "full_cartesian_count": int(manifest.get("row_count", -1))
        == len(selection.graphs) * len(parent_ids),
        "candidate_graph_hashes": scientific.get("candidate_graph_hashes_sha256")
        == stable_json_sha256(selection.graph_hashes),
        "generation_indices": scientific.get("generation_indices_sha256")
        == stable_json_sha256(selection.generation_indices),
        "parent_ids": scientific.get("parent_ids_sha256")
        == stable_json_sha256(list(parent_ids)),
        "generation_manifest": scientific.get("generation_manifest_sha256")
        == sha256_file(generation_manifest_path),
        "counterfactuals": scientific.get("counterfactuals_sha256")
        == generation_manifest.get("counterfactuals_sha256"),
        "distance_checkpoint": scientific.get("distance_checkpoint_sha256")
        == sha256_file(distance_checkpoint),
        "parameters": scientific.get("parameters") == expected_parameters,
        "theta": math.isclose(float(theta), EXPECTED_THETA, rel_tol=0.0, abs_tol=0.0),
        "chunk_count": int(chunk_authority["source_chunk_count"]) == len(chunks),
    }
    failures.extend(key for key, passed in checks.items() if not passed)
    if failures:
        raise AIDSPairSemanticsError(
            "pair-store provenance contract failed: " + ", ".join(failures)
        )
    pair_path = Path(str(manifest["pairs_path"])).expanduser().resolve(strict=True)
    vector_path = Path(str(manifest["vectors_path"])).expanduser().resolve(strict=True)
    pairs = np.load(pair_path, mmap_mode="r", allow_pickle=False)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    if (
        pairs.shape != (EXPECTED_PHYSICAL_PAIR_COUNT, 2)
        or pairs.dtype != np.dtype(np.int64)
        or vectors.shape != (EXPECTED_PHYSICAL_PAIR_COUNT, 64)
        or vectors.dtype != np.dtype(np.float32)
    ):
        raise AIDSPairSemanticsError("production pair-store array schema differs")
    manifest = {**manifest, "resolved_chunk_metadata_authority": chunk_authority}
    return manifest, chunks


def _sample_rows(
    *,
    total_rows: int,
    parent_count: int,
    chunks: Sequence[PairChunk],
    boundary_rows: Sequence[int],
    seed: int,
    random_count: int,
) -> tuple[np.ndarray, dict[int, set[str]]]:
    rng = np.random.default_rng(int(seed))
    random_rows = rng.choice(total_rows, size=min(random_count, total_rows), replace=False)
    sources: dict[int, set[str]] = {}
    for row in random_rows:
        sources.setdefault(int(row), set()).add("random")
    for chunk in chunks:
        first = chunk.candidate_start * int(parent_count)
        last = chunk.candidate_stop * int(parent_count) - 1
        sources.setdefault(int(first), set()).add("chunk_first")
        sources.setdefault(int(last), set()).add("chunk_last")
    for row in boundary_rows:
        sources.setdefault(int(row), set()).add("theta_boundary_closest")
    rows = np.asarray(sorted(sources), dtype=np.int64)
    return rows, sources


def _formula_audit(
    *,
    sample_rows: np.ndarray,
    sample_sources: Mapping[int, set[str]],
    parent_count: int,
    candidate_graphs: Sequence[Any],
    parent_embeddings: Any,
    parent_counts: Any,
    adapter: AIDSGreedEmbeddingAdapter,
    upstream_util: Any,
    pair_indices_path: Path,
    recourse_vectors_path: Path,
    normalized_distances_path: Path,
    batch_class: Any,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    pairs = np.load(pair_indices_path, mmap_mode="r", allow_pickle=False)
    stored_vectors = np.load(recourse_vectors_path, mmap_mode="r", allow_pickle=False)
    distances = np.load(normalized_distances_path, mmap_mode="r", allow_pickle=False)
    candidate_indices = sorted(
        {
            pair_for_cartesian_row(int(row), parent_count=parent_count)[1]
            for row in sample_rows
        }
    )
    embedding_by_candidate: dict[int, np.ndarray] = {}
    count_by_candidate: dict[int, int] = {}
    for start in range(0, len(candidate_indices), max(1, int(batch_size))):
        indices = candidate_indices[start : start + max(1, int(batch_size))]
        graphs = [candidate_graphs[index] for index in indices]
        embedded = (
            adapter.embed_model(batch_class.from_data_list(graphs).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        counts = upstream_util.graph_element_counts(graphs).detach().cpu().numpy()
        for local, candidate_index in enumerate(indices):
            embedding_by_candidate[candidate_index] = np.asarray(embedded[local])
            count_by_candidate[candidate_index] = int(counts[local])
    parent_embedding_array = parent_embeddings.detach().cpu().numpy()
    parent_count_array = parent_counts.detach().cpu().numpy()
    rows_out: list[dict[str, Any]] = []
    max_formula_error = 0.0
    max_reverse_error = 0.0
    max_distance_norm_error = 0.0
    pair_mismatch_count = 0
    for row_value in sample_rows:
        row = int(row_value)
        parent_index, candidate_index = pair_for_cartesian_row(
            row, parent_count=parent_count
        )
        actual_pair = (int(pairs[row, 0]), int(pairs[row, 1]))
        if actual_pair != (parent_index, candidate_index):
            pair_mismatch_count += 1
        candidate_count = count_by_candidate[candidate_index]
        parent_element_count = int(parent_count_array[parent_index])
        calculated = recourse_vector_formula(
            embedding_by_candidate[candidate_index],
            parent_embedding_array[parent_index],
            candidate_element_count=candidate_count,
            parent_element_count=parent_element_count,
        ).astype(np.float32, copy=False)
        stored = np.asarray(stored_vectors[row], dtype=np.float32)
        formula_error = float(
            np.max(np.abs(stored.astype(np.float64) - calculated.astype(np.float64)))
        )
        reverse_error = float(
            np.max(np.abs(stored.astype(np.float64) + calculated.astype(np.float64)))
        )
        normalized = float(distances[row])
        formula_norm = float(np.linalg.norm(calculated.astype(np.float64)))
        distance_norm_error = abs(normalized - formula_norm)
        max_formula_error = max(max_formula_error, formula_error)
        max_reverse_error = max(max_reverse_error, reverse_error)
        max_distance_norm_error = max(max_distance_norm_error, distance_norm_error)
        rows_out.append(
            {
                "row": row,
                "sample_sources": sorted(sample_sources[row]),
                "pair_from_store": list(actual_pair),
                "pair_from_candidate_major_formula": [parent_index, candidate_index],
                "candidate_element_count": candidate_count,
                "parent_element_count": parent_element_count,
                "scale": candidate_count + parent_element_count,
                "normalized_distance_from_frozen_interface": normalized,
                "stored_recourse_norm_float64": float(
                    np.linalg.norm(stored.astype(np.float64))
                ),
                "recomputed_recourse_norm_float64": formula_norm,
                "candidate_minus_parent_max_abs_error": formula_error,
                "parent_minus_candidate_max_abs_error": reverse_error,
                "distance_vs_recomputed_norm_abs_error": distance_norm_error,
            }
        )
    return {
        "schema_version": "aids_comrecgc_pair_formula_sample_v1",
        "sample_count": len(rows_out),
        "random_sample_count_requested": 1_000,
        "chunk_endpoint_source_count": sum(
            "chunk_first" in values or "chunk_last" in values
            for values in sample_sources.values()
        ),
        "theta_boundary_source_count": sum(
            "theta_boundary_closest" in values for values in sample_sources.values()
        ),
        "pair_axis_mismatch_count": pair_mismatch_count,
        "candidate_minus_parent_max_absolute_error": max_formula_error,
        "parent_minus_candidate_max_absolute_error": max_reverse_error,
        "distance_vs_recomputed_recourse_norm_max_absolute_error": max_distance_norm_error,
        "axis_orientation": "col0=parent_index,col1=candidate_index",
        "recourse_formula": (
            "(embedding(candidate)-embedding(parent))/"
            "(element_count(candidate)+element_count(parent))"
        ),
        "rows": rows_out,
    }


def _build_all_pairs_close_certificate(
    *,
    physical_pair_count: int,
    logical_close_pair_count: int,
    theta: float,
    count_distance_eq_theta: int,
    physical_vectors_sha256: str,
    normalized_distances_sha256: str,
    close_bitmap_sha256: str,
    distance_checkpoint_sha256: str,
    normalization_audit: Mapping[str, Any],
    formula_audit: Mapping[str, Any],
    formula_tolerance: float,
    distance_norm_consistency_tolerance: float,
) -> dict[str, Any]:
    normalization_pass = bool(
        normalization_audit.get("official_torch_vs_independent_numpy_exact")
    ) and float(normalization_audit.get("max_absolute_error", math.inf)) == 0.0
    official_sample_comparison_pass = bool(
        int(formula_audit.get("pair_axis_mismatch_count", -1)) == 0
        and float(
            formula_audit.get(
                "candidate_minus_parent_max_absolute_error", math.inf
            )
        )
        <= float(formula_tolerance)
        and float(
            formula_audit.get(
                "distance_vs_recomputed_recourse_norm_max_absolute_error",
                math.inf,
            )
        )
        <= float(distance_norm_consistency_tolerance)
    )
    all_close = int(logical_close_pair_count) == int(physical_pair_count)
    if not all_close or not normalization_pass or not official_sample_comparison_pass:
        raise AIDSPairSemanticsError(
            "all-pairs-close certificate prerequisites are not proven"
        )
    return {
        "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
        "status": "PASS",
        "all_pairs_close_proven": True,
        "full_distance_scan_complete": True,
        "official_sample_comparison_pass": True,
        "official_sample_comparison_scope": (
            "frozen GREED cdist, official element-count normalization, and "
            "stored recourse-vector formula on production samples"
        ),
        "normalization_audit_pass": True,
        "filter_operator": "<=",
        "pair_orientation": "col0_parent_col1_candidate",
        "pair_columns": ["parent_index", "candidate_index"],
        "pair_order": "candidate_major_parent_minor",
        "physical_pair_count": int(physical_pair_count),
        "physical_rows": int(physical_pair_count),
        "physical_store_rows": int(physical_pair_count),
        "count_distance_le_theta": int(logical_close_pair_count),
        "count_distance_lte_theta": int(logical_close_pair_count),
        "count_distance_gt_theta": int(physical_pair_count)
        - int(logical_close_pair_count),
        "count_distance_eq_theta": int(count_distance_eq_theta),
        "theta": float(theta),
        "physical_vectors_sha256": str(physical_vectors_sha256),
        "normalized_distances_sha256": str(normalized_distances_sha256),
        "close_bitmap_sha256": str(close_bitmap_sha256),
        "distance_checkpoint_sha256": str(distance_checkpoint_sha256),
        "distance_checkpoint_hash": str(distance_checkpoint_sha256),
        "embedding_checkpoint_sha256": str(distance_checkpoint_sha256),
        "embedding_checkpoint_hash": str(distance_checkpoint_sha256),
        "scale_contract": (
            "element_count(parent)+element_count(candidate)"
        ),
        "distance_contract": (
            "AIDSGreedEmbeddingAdapter.predict_outer_with_queries -> "
            "torch.cdist(candidate_embeddings,parent_embeddings,p=2)"
        ),
        "normalized_distance_contract": (
            "GREED/NeuroSED.predict_outer(parent,candidate)/"
            "(element_count(parent)+element_count(candidate))"
        ),
        "distance_authority": "frozen_project_GREED_checkpoint",
        "official_backend_difference": "PROJECT_EXTENSION",
        "approximation_used": False,
        "completed_at": utc_now(),
    }


def run_aids_pair_semantics_audit(
    *,
    project_root: str | Path,
    upstream_root: str | Path,
    dataset_dir: str | Path,
    source_csv: str | Path,
    generation_dir: str | Path,
    distance_checkpoint: str | Path,
    pair_store_manifest: str | Path,
    output_dir: str | Path,
    expected_pair_store_manifest_sha256: str,
    parent_limit: int = EXPECTED_PARENT_COUNT,
    theta: float = EXPECTED_THETA,
    device: str = "cpu",
    distance_batch_size: int = 128,
    resume: bool = False,
    max_chunks: int | None = None,
    verify_source_array_hashes: bool = True,
) -> dict[str, Any]:
    if device != "cpu":
        raise AIDSPairSemanticsError("AIDS pair-semantics audit is CPU-only")
    if int(parent_limit) != EXPECTED_PARENT_COUNT:
        raise AIDSPairSemanticsError("production AIDS parent count must remain 1283")
    if not math.isclose(float(theta), EXPECTED_THETA, rel_tol=0.0, abs_tol=0.0):
        raise AIDSPairSemanticsError("production AIDS theta must remain 0.1")
    if not verify_source_array_hashes and max_chunks is None:
        raise AIDSPairSemanticsError(
            "source-array hash verification may be skipped only for a bounded benchmark"
        )
    if max_chunks is not None and int(max_chunks) <= 0:
        raise AIDSPairSemanticsError("max_chunks must be positive when provided")
    if int(distance_batch_size) <= 0:
        raise AIDSPairSemanticsError("distance_batch_size must be positive")
    project = Path(project_root).expanduser().resolve(strict=True)
    upstream = validate_upstream_checkout(upstream_root)
    root = Path(output_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    materialized_bytes = EXPECTED_PHYSICAL_PAIR_COUNT * (
        np.dtype(np.float32).itemsize + np.dtype(np.uint8).itemsize
    )
    required_free_bytes = materialized_bytes + 64 * 1024**2
    storage_preflight = {
        "output_filesystem_total_bytes": int(disk.total),
        "output_filesystem_used_bytes": int(disk.used),
        "output_filesystem_free_bytes": int(disk.free),
        "materialized_distance_and_bitmap_bytes": int(materialized_bytes),
        "required_free_bytes_with_safety_margin": int(required_free_bytes),
    }
    if int(disk.free) < required_free_bytes:
        raise AIDSPairSemanticsError(
            "insufficient free space for GREED distance and close bitmap materialization"
        )
    progress_path = root / "progress.json"
    lock_path = root / "pair_semantics.lock"
    generation_root = Path(generation_dir).expanduser().resolve(strict=True)
    generation_manifest_path = generation_root / "run_manifest.json"
    generation_manifest = _json_load(generation_manifest_path)
    pair_manifest_path = Path(pair_store_manifest).expanduser().resolve(strict=True)
    checkpoint_path = Path(distance_checkpoint).expanduser().resolve(strict=True)
    if sha256_file(pair_manifest_path) != str(expected_pair_store_manifest_sha256):
        raise AIDSPairSemanticsError("pair-store owner manifest SHA256 differs")
    with _exclusive_audit_lock(lock_path):
        _progress(progress_path, state="LOADING_FROZEN_INPUTS")
        bundle = load_aids_generation_bundle(
            dataset_dir=dataset_dir,
            source_csv=source_csv,
            parent_limit=parent_limit,
        )
        if len(bundle.graphs) != EXPECTED_PARENT_COUNT:
            raise AIDSPairSemanticsError("frozen AIDS parent bundle count differs")
        counterfactuals_path = Path(
            str(generation_manifest["counterfactuals_path"])
        ).expanduser().resolve(strict=True)
        if sha256_file(counterfactuals_path) != generation_manifest.get(
            "counterfactuals_sha256"
        ):
            raise AIDSPairSemanticsError("frozen counterfactual payload SHA256 differs")
        load_started = time.monotonic()
        payload = _torch_load(counterfactuals_path)
        parameters = RecourseParameters.for_mode("full")
        selection = _select_candidates(payload, cap=parameters.cf_size)
        del payload
        gc.collect()
        if len(selection.graphs) != EXPECTED_CANDIDATE_COUNT:
            raise AIDSPairSemanticsError(
                f"production candidate count {len(selection.graphs)} != 71642"
            )
        manifest, chunks = _validate_pair_store(
            manifest_path=pair_manifest_path,
            generation_manifest_path=generation_manifest_path,
            generation_manifest=generation_manifest,
            distance_checkpoint=checkpoint_path,
            selection=selection,
            parent_ids=bundle.parent_ids,
            theta=theta,
        )
        pair_indices_path = Path(str(manifest["pairs_path"])).resolve(strict=True)
        recourse_vectors_path = Path(str(manifest["vectors_path"])).resolve(strict=True)
        chunk_metadata_manifest_path = Path(
            str(
                manifest["resolved_chunk_metadata_authority"][
                    "chunk_metadata_manifest"
                ]
            )
        ).resolve(strict=True)
        source_stats_before = {
            "pair_manifest": _file_stat(pair_manifest_path),
            "chunk_metadata_manifest": _file_stat(chunk_metadata_manifest_path),
            "pair_indices": _file_stat(pair_indices_path),
            "recourse_vectors": _file_stat(recourse_vectors_path),
        }
        _progress(
            progress_path,
            state="EMBEDDING_PARENTS",
            candidate_count=len(selection.graphs),
            parent_count=len(bundle.graphs),
            payload_load_seconds=time.monotonic() - load_started,
        )
        torch, Batch = _torch_stack()
        adapter = AIDSGreedEmbeddingAdapter(
            checkpoint_path,
            atom_vocabulary=[str(value) for value in bundle.atom_vocabulary],
            device=device,
        ).eval()
        with imported_upstream(upstream) as modules:
            original_batch = Batch.from_data_list(bundle.graphs).to(device)
            with torch.no_grad():
                parent_embeddings = adapter.embed_model(original_batch).detach().cpu()
            adapter.embed_targets(bundle.graphs)
            target_reembed_max_error = float(
                torch.max(torch.abs(parent_embeddings - adapter._targets.detach().cpu())).item()
            )
            parent_counts = modules["util"].graph_element_counts(bundle.graphs).cpu()
            normalization_audit_path = root / "normalization_audit.json"
            normalization_identity = {
                "schema_version": "comrecgc_normalization_audit_v1",
                "distance_checkpoint_sha256": sha256_file(checkpoint_path),
                "theta": float(theta),
                "official_expression": (
                    "raw_distances / (candidate_counts[:,None] + "
                    "parent_counts[None,:])"
                ),
            }
            if resume and normalization_audit_path.exists():
                normalization_payload = _json_load(normalization_audit_path)
                if normalization_payload.get("identity") != normalization_identity:
                    raise AIDSPairSemanticsError(
                        "normalization-audit resume identity differs"
                    )
                records_raw = normalization_payload.get("records")
                if not isinstance(records_raw, dict):
                    raise AIDSPairSemanticsError(
                        "normalization-audit resume records are invalid"
                    )
                normalization_records = dict(records_raw)
            else:
                if normalization_audit_path.exists():
                    raise FileExistsError(
                        f"fresh normalization audit exists: {normalization_audit_path}"
                    )
                normalization_records: dict[str, Any] = {}

            def normalization_summary() -> dict[str, Any]:
                records = list(normalization_records.values())
                return {
                    "identity": normalization_identity,
                    "blocks_compared": len(records),
                    "official_torch_vs_independent_numpy_exact": all(
                        bool(record["exact_equal"]) for record in records
                    ),
                    "max_absolute_error": max(
                        (float(record["max_absolute_error"]) for record in records),
                        default=0.0,
                    ),
                    "records": dict(normalization_records),
                    "updated_at": utc_now(),
                }

            def provide(candidate_start: int, candidate_stop: int) -> np.ndarray:
                chunk = selection.graphs[candidate_start:candidate_stop]
                with torch.no_grad():
                    raw = adapter.predict_outer_with_queries(
                        chunk, batch_size=distance_batch_size
                    ).detach().cpu()
                candidate_counts = modules["util"].graph_element_counts(chunk).cpu()
                scale = candidate_counts[:, None] + parent_counts[None, :]
                official_normalized = (raw / scale).numpy()
                independent_normalized = normalize_distance_block(
                    raw.numpy(),
                    candidate_element_counts=candidate_counts.numpy(),
                    parent_element_counts=parent_counts.numpy(),
                )
                exact = bool(
                    np.array_equal(
                        official_normalized,
                        independent_normalized,
                        equal_nan=True,
                    )
                )
                finite = np.isfinite(official_normalized) & np.isfinite(
                    independent_normalized
                )
                block_error = (
                    float(
                        np.max(
                            np.abs(
                                official_normalized[finite].astype(np.float64)
                                - independent_normalized[finite].astype(np.float64)
                            )
                        )
                    )
                    if np.any(finite)
                    else 0.0
                )
                key = f"{candidate_start}:{candidate_stop}"
                normalization_records[key] = {
                    "candidate_start": int(candidate_start),
                    "candidate_stop": int(candidate_stop),
                    "exact_equal": exact,
                    "max_absolute_error": block_error,
                    "official_dtype": str(official_normalized.dtype),
                    "independent_dtype": str(independent_normalized.dtype),
                }
                _atomic_json(normalization_audit_path, normalization_summary())
                return np.asarray(official_normalized, dtype=np.float32)

            scan_identity = {
                "project_commit": _current_commit(project),
                "official_comrecgc_commit": read_upstream_commit(upstream),
                "pair_store_manifest": str(pair_manifest_path),
                "pair_store_manifest_sha256": sha256_file(pair_manifest_path),
                "chunk_metadata_authority": manifest[
                    "resolved_chunk_metadata_authority"
                ],
                "generation_manifest": str(generation_manifest_path),
                "generation_manifest_sha256": sha256_file(generation_manifest_path),
                "counterfactuals_sha256": generation_manifest[
                    "counterfactuals_sha256"
                ],
                "distance_checkpoint": str(checkpoint_path),
                "distance_checkpoint_sha256": sha256_file(checkpoint_path),
                "distance_interface": (
                    "AIDSGreedEmbeddingAdapter.predict_outer_with_queries -> "
                    "torch.cdist(p=2)"
                ),
                "normalization": (
                    "raw_distance/(element_count(candidate)+element_count(parent))"
                ),
                "candidate_graph_hashes_sha256": stable_json_sha256(
                    selection.graph_hashes
                ),
                "generation_indices_sha256": stable_json_sha256(
                    selection.generation_indices
                ),
                "parent_ids_sha256": stable_json_sha256(list(bundle.parent_ids)),
            }

            def report_scan_progress(value: Mapping[str, Any]) -> None:
                _progress(
                    progress_path,
                    state="SCANNING_FROZEN_GREED_DISTANCES",
                    rows_processed=int(value["rows_processed"]),
                    logical_close_pair_count=int(value["logical_close_pair_count"]),
                    rolling_throughput_rows_per_second=float(
                        value["rolling_throughput_rows_per_second"]
                    ),
                    eta_seconds=value.get("eta_seconds"),
                    last_checkpoint=int(value["next_chunk_index"]) - 1,
                )

            scan = scan_theta_close_pairs(
                output_dir=root / "distance_scan",
                pair_indices_path=pair_indices_path,
                pair_chunks=chunks,
                parent_count=len(bundle.graphs),
                candidate_count=len(selection.graphs),
                theta=theta,
                scientific_identity=scan_identity,
                distance_provider=provide,
                resume=resume,
                max_chunks=max_chunks,
                boundary_sample_size=1_000,
                progress_callback=report_scan_progress,
            )
            if scan is None:
                checkpoint = _json_load(root / "distance_scan/checkpoint.json")
                source_stats_after_benchmark = {
                    "pair_manifest": _file_stat(pair_manifest_path),
                    "chunk_metadata_manifest": _file_stat(
                        chunk_metadata_manifest_path
                    ),
                    "pair_indices": _file_stat(pair_indices_path),
                    "recourse_vectors": _file_stat(recourse_vectors_path),
                }
                if source_stats_after_benchmark != source_stats_before:
                    raise AIDSPairSemanticsError(
                        "read-only source stat identity changed during benchmark"
                    )
                chunk_metadata_final_sha256 = sha256_file(
                    chunk_metadata_manifest_path
                )
                if chunk_metadata_final_sha256 != manifest[
                    "resolved_chunk_metadata_authority"
                ]["chunk_metadata_manifest_sha256"]:
                    raise AIDSPairSemanticsError(
                        "chunk-metadata authority changed during benchmark"
                    )
                benchmark = {
                    "schema_version": AIDS_PAIR_SEMANTICS_SCHEMA,
                    "status": "BENCHMARK_COMPLETE_NOT_SCIENTIFIC_PASS",
                    "rows_processed": int(checkpoint["rows_processed"]),
                    "physical_pair_count": EXPECTED_PHYSICAL_PAIR_COUNT,
                    "rolling_throughput_rows_per_second": checkpoint.get(
                        "rolling_throughput_rows_per_second"
                    ),
                    "eta_seconds": checkpoint.get("eta_seconds"),
                    "resume_command_required": True,
                    "storage_preflight": storage_preflight,
                    "source_stats_before": source_stats_before,
                    "source_stats_after": source_stats_after_benchmark,
                    "source_mutated": False,
                    "chunk_metadata_final_sha256": (
                        chunk_metadata_final_sha256
                    ),
                    "normalization_audit": normalization_summary(),
                    "completed_at": utc_now(),
                }
                _atomic_json(root / "benchmark_result.json", benchmark)
                _progress(progress_path, state="BENCHMARK_COMPLETE", **benchmark)
                return benchmark

            scan_manifest = _json_load(scan.manifest_path)
            expected_normalization_keys = {
                f"{chunk.candidate_start}:{chunk.candidate_stop}"
                for chunk in chunks
            }
            if set(normalization_records) != expected_normalization_keys:
                raise AIDSPairSemanticsError(
                    "normalization audit does not cover every physical chunk"
                )
            normalization_audit = normalization_summary()
            if (
                not normalization_audit[
                    "official_torch_vs_independent_numpy_exact"
                ]
                or float(normalization_audit["max_absolute_error"]) != 0.0
            ):
                raise AIDSPairSemanticsError(
                    "official torch and independently recomputed normalization differ"
                )
            nonfinite_count = int(
                scan_manifest["distance_statistics"]["nonfinite_count"]
            )
            if nonfinite_count:
                raise AIDSPairSemanticsError(
                    "frozen GREED distance scan produced non-finite values: "
                    f"count={nonfinite_count}"
                )
            sample_rows, sample_sources = _sample_rows(
                total_rows=EXPECTED_PHYSICAL_PAIR_COUNT,
                parent_count=len(bundle.graphs),
                chunks=chunks,
                boundary_rows=scan_manifest["boundary_rows"],
                seed=20260824,
                random_count=1_000,
            )
            _progress(
                progress_path,
                state="VERIFYING_PAIR_AXIS_AND_RECOURSE_FORMULA",
                rows_processed=EXPECTED_PHYSICAL_PAIR_COUNT,
            )
            formula = _formula_audit(
                sample_rows=sample_rows,
                sample_sources=sample_sources,
                parent_count=len(bundle.graphs),
                candidate_graphs=selection.graphs,
                parent_embeddings=parent_embeddings,
                parent_counts=parent_counts,
                adapter=adapter,
                upstream_util=modules["util"],
                pair_indices_path=pair_indices_path,
                recourse_vectors_path=recourse_vectors_path,
                normalized_distances_path=scan.distance_path,
                batch_class=Batch,
                device=device,
                batch_size=distance_batch_size,
            )
        formula_path = root / "recourse_formula_samples.json"
        _atomic_json(formula_path, formula)
        formula_tolerance = 1e-6
        distance_norm_consistency_tolerance = 1e-5
        if (
            int(formula["pair_axis_mismatch_count"]) != 0
            or float(formula["candidate_minus_parent_max_absolute_error"])
            > formula_tolerance
            or float(formula["distance_vs_recomputed_recourse_norm_max_absolute_error"])
            > distance_norm_consistency_tolerance
        ):
            raise AIDSPairSemanticsError("pair-axis/formula sample audit failed")

        source_hashes: dict[str, Any] = {
            "pair_store_manifest_sha256": sha256_file(pair_manifest_path),
            "pair_indices_manifest_sha256": manifest["pairs_sha256"],
            "recourse_vectors_manifest_sha256": manifest["vectors_sha256"],
            "direct_array_hash_verification_performed": bool(
                verify_source_array_hashes
            ),
            "chunk_metadata_manifest_sha256": sha256_file(
                chunk_metadata_manifest_path
            ),
        }
        if source_hashes["chunk_metadata_manifest_sha256"] != manifest[
            "resolved_chunk_metadata_authority"
        ]["chunk_metadata_manifest_sha256"]:
            raise AIDSPairSemanticsError(
                "chunk-metadata authority changed during full scan"
            )
        if verify_source_array_hashes:
            _progress(
                progress_path,
                state="VERIFYING_SOURCE_ARRAY_SHA256",
                rows_processed=EXPECTED_PHYSICAL_PAIR_COUNT,
            )
            source_hashes.update(
                {
                    "pair_indices_direct_sha256": sha256_file(pair_indices_path),
                    "recourse_vectors_direct_sha256": sha256_file(
                        recourse_vectors_path
                    ),
                }
            )
            if (
                source_hashes["pair_indices_direct_sha256"]
                != manifest["pairs_sha256"]
                or source_hashes["recourse_vectors_direct_sha256"]
                != manifest["vectors_sha256"]
            ):
                raise AIDSPairSemanticsError("direct pair-store SHA256 differs")
        source_stats_after = {
            "pair_manifest": _file_stat(pair_manifest_path),
            "chunk_metadata_manifest": _file_stat(chunk_metadata_manifest_path),
            "pair_indices": _file_stat(pair_indices_path),
            "recourse_vectors": _file_stat(recourse_vectors_path),
        }
        if source_stats_after != source_stats_before:
            raise AIDSPairSemanticsError("read-only source stat identity changed")
        logical_close = int(scan_manifest["logical_close_pair_count"])
        all_pairs_close_certificate_path: Path | None = None
        all_pairs_close_certificate_sha256: str | None = None
        if logical_close == EXPECTED_PHYSICAL_PAIR_COUNT:
            all_pairs_close_certificate = _build_all_pairs_close_certificate(
                physical_pair_count=EXPECTED_PHYSICAL_PAIR_COUNT,
                logical_close_pair_count=logical_close,
                theta=theta,
                count_distance_eq_theta=int(
                    scan_manifest["distance_statistics"][
                        "count_distance_eq_theta"
                    ]
                ),
                physical_vectors_sha256=str(
                    source_hashes["recourse_vectors_direct_sha256"]
                ),
                normalized_distances_sha256=scan.distance_sha256,
                close_bitmap_sha256=scan.close_bitmap_sha256,
                distance_checkpoint_sha256=sha256_file(checkpoint_path),
                normalization_audit=normalization_audit,
                formula_audit=formula,
                formula_tolerance=formula_tolerance,
                distance_norm_consistency_tolerance=(
                    distance_norm_consistency_tolerance
                ),
            )
            all_pairs_close_certificate_path = (
                root / "all_pairs_close_certificate.json"
            )
            _atomic_json(
                all_pairs_close_certificate_path,
                all_pairs_close_certificate,
            )
            all_pairs_close_certificate_sha256 = sha256_file(
                all_pairs_close_certificate_path
            )
        contract = {
            "schema_version": AIDS_PAIR_SEMANTICS_SCHEMA,
            "status": "PASS",
            "physical_pair_store_adopted": True,
            "pair_store_regenerated": False,
            "physical_store_rows": EXPECTED_PHYSICAL_PAIR_COUNT,
            "logical_close_rows": logical_close,
            "close_pair_rate": logical_close / EXPECTED_PHYSICAL_PAIR_COUNT,
            "physical_store_is_full_cartesian": True,
            "all_pairs_close": logical_close == EXPECTED_PHYSICAL_PAIR_COUNT,
            "all_pairs_close_certificate": (
                None
                if all_pairs_close_certificate_path is None
                else str(all_pairs_close_certificate_path)
            ),
            "all_pairs_close_certificate_sha256": (
                all_pairs_close_certificate_sha256
            ),
            "theta": float(theta),
            "filter_operator": "<=",
            "distance_checkpoint_hash": sha256_file(checkpoint_path),
            "embedding_checkpoint_hash": sha256_file(checkpoint_path),
            "distance_backend": "project_greed_hiv_ged",
            "official_backend_difference": "PROJECT_EXTENSION",
            "scale_contract": (
                "element_count(parent)+element_count(candidate)"
            ),
            "pair_orientation": ["parent_index", "candidate_index"],
            "pair_order": "candidate_major_parent_minor",
            "dtype": "float32",
            "chunk_order": [chunk.chunk_index for chunk in chunks],
            "chunk_count": len(chunks),
            "close_bitmap": str(scan.close_bitmap_path),
            "close_bitmap_hash": scan.close_bitmap_sha256,
            "normalized_distances": str(scan.distance_path),
            "normalized_distances_hash": scan.distance_sha256,
            "distance_statistics": scan_manifest["distance_statistics"],
            "normalization_audit": normalization_audit,
            "normalization_audit_path": str(normalization_audit_path),
            "normalization_audit_sha256": sha256_file(
                normalization_audit_path
            ),
            "dbscan_input_count_must_equal": logical_close,
            "pair_axis_all_rows_checked": True,
            "pair_axis_mismatch_count": 0,
            "recourse_formula_sample_count": formula["sample_count"],
            "recourse_formula_max_absolute_error": formula[
                "candidate_minus_parent_max_absolute_error"
            ],
            "distance_vs_recourse_norm_sample_max_absolute_error": formula[
                "distance_vs_recomputed_recourse_norm_max_absolute_error"
            ],
            "recourse_formula_tolerance": formula_tolerance,
            "distance_norm_consistency_tolerance": (
                distance_norm_consistency_tolerance
            ),
            "parent_target_reembedding_max_absolute_error": target_reembed_max_error,
            "candidate_cap_source": (
                "official generation comrecgc.py --k -> MAX_COUNTERFACTUAL_SIZE"
            ),
            "official_candidate_cap_source": (
                "comrecgc.py --k -> MAX_COUNTERFACTUAL_SIZE"
            ),
            "official_common_recourse_cf_size_applied": False,
            "project_extension_slice": (
                "post-predicate first RecourseParameters.full.cf_size candidates"
            ),
            "project_extension_slice_present": True,
            "candidate_cap_applied": (
                selection.graph_resolved_classifier_pass_count > selection.cap
            ),
            "candidate_cap_binding": (
                selection.graph_resolved_classifier_pass_count > selection.cap
            ),
            "candidate_cap_value": selection.cap,
            "raw_candidate_count": selection.raw_candidate_count,
            "candidate_count_before": (
                selection.graph_resolved_classifier_pass_count
            ),
            "candidate_classifier_pass_count": selection.classifier_pass_count,
            "candidate_graph_resolved_classifier_pass_count": (
                selection.graph_resolved_classifier_pass_count
            ),
            "candidate_count_after": len(selection.graphs),
            "candidate_predicate": (
                "importance_parts[0] >= 0.5 and graph_hash in graph_map"
            ),
            "source_pair_store_manifest": str(pair_manifest_path),
            "source_pair_store_manifest_sha256": sha256_file(pair_manifest_path),
            "chunk_metadata_authority": manifest[
                "resolved_chunk_metadata_authority"
            ],
            "source_hashes": source_hashes,
            "source_stats_before": source_stats_before,
            "source_stats_after": source_stats_after,
            "source_mutated": False,
            "storage_preflight": storage_preflight,
            "generation_manifest": str(generation_manifest_path),
            "generation_manifest_sha256": sha256_file(generation_manifest_path),
            "official_comrecgc_commit": read_upstream_commit(upstream),
            "project_commit": _current_commit(project),
            "formula_samples": str(formula_path),
            "formula_samples_sha256": sha256_file(formula_path),
            "distance_scan_manifest": str(scan.manifest_path),
            "distance_scan_manifest_sha256": scan.manifest_sha256,
            "completed_at": utc_now(),
        }
        contract_path = root / "close_pair_contract.json"
        _atomic_json(contract_path, contract)
        final = {
            **contract,
            "close_pair_contract": str(contract_path),
            "close_pair_contract_sha256": sha256_file(contract_path),
            "markers": [
                "[AIDS_CLOSE_PAIR_FILTER_PASS]",
                "[AIDS_RECOURSE_VECTOR_FORMULA_PASS]",
            ],
        }
        _write_terminal_pass(
            root=root,
            progress_path=progress_path,
            final=final,
        )
        return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hpc.yaml",
        help="Accepted for repository wrapper compatibility; no config values are read.",
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--distance-checkpoint", required=True)
    parser.add_argument("--pair-store-manifest", required=True)
    parser.add_argument("--expected-pair-store-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, default=EXPECTED_PARENT_COUNT)
    parser.add_argument("--theta", type=float, default=EXPECTED_THETA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Bounded benchmark only; a partial scan never emits scientific PASS.",
    )
    parser.add_argument(
        "--skip-source-array-hash-verification",
        action="store_true",
        help="Allowed only for a bounded benchmark, never for a full PASS.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_source_array_hash_verification and args.max_chunks is None:
        raise SystemExit(
            "--skip-source-array-hash-verification is allowed only with --max-chunks"
        )
    result = run_aids_pair_semantics_audit(
        project_root=args.project_root,
        upstream_root=args.upstream_root,
        dataset_dir=args.dataset_dir,
        source_csv=args.source_csv,
        generation_dir=args.generation_dir,
        distance_checkpoint=args.distance_checkpoint,
        pair_store_manifest=args.pair_store_manifest,
        output_dir=args.output_dir,
        expected_pair_store_manifest_sha256=(
            args.expected_pair_store_manifest_sha256
        ),
        parent_limit=args.parent_limit,
        theta=args.theta,
        device=args.device,
        distance_batch_size=args.distance_batch_size,
        resume=args.resume,
        max_chunks=args.max_chunks,
        verify_source_array_hashes=not args.skip_source_array_hash_verification,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
