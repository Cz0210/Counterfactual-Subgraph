"""Bounded exact global-neighbor witnesses for the frozen AIDS RF pair store.

The old anchor-only failure is inconclusive.  This reader uses the identical
sklearn float32 brute-radius kernel, never regenerates vectors, and never
substitutes an unproven DBSCAN partition.  A completed seed-failure ledger is
the only authority for non-failure rows; 55 core anchors alone are insufficient.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from . import external_memory_dbscan as engine
from .rf_aligned_pool import atomic_json, digest, file_sha

GIB = 1024**3
SCHEMA = "aids_global_radius_witness_v1"


def memory_plan(binding: dict[str, Any]) -> dict[str, Any]:
    components = {
        "new_source_vector_window_pages": 1048576 * 64 * 4,
        "sparse_anchor_file_pages": 55 * 4096,
        "bounded_radius_workspace": 64 * 1024**2,
        "python_sklearn_and_static_ledger": GIB,
        "owner_sampler_and_atomic_outputs": 128 * 1024**2,
    }
    if sum(components.values()) > 2 * GIB:
        raise ValueError("Witness pilot exceeds reviewed 2GiB bound")
    return {"phase": "AIDS_GLOBAL_RADIUS_WITNESS", "components_bytes": components,
            "component_sum_bytes": sum(components.values()), "charged_peak_bound_bytes": 2*GIB,
            "workers": 1, "cpu_thread_limit": 2, "query_rows_max": 4096,
            "max_new_files": 64, "max_seconds": 7200, "gpu_requested": False,
            "maximum_new_rows_this_process": 1048576,
            "all_next_window_pagecache_counted": True, "full_scan_admission_claimed": False,
            "full_n_squared_graph": False}


def source_binding(config: dict[str, Any], recourse: Path, pairs: dict[str, Any]):
    """Consume complete static ledgers once, without rehashing 10GB arrays."""
    root = recourse / "dbscan"
    checkpoint_path = root / "checkpoint.json"
    before = engine._source_stat_identity(checkpoint_path)
    state = engine._load_checkpoint(checkpoint_path)
    identity = state["identity"]
    if (identity["vectors_sha256"] != pairs["arrays"]["vectors"]["sha256"]
            or identity["vectors_path"] != pairs["arrays"]["vectors"]["path"]
            or identity["contract"]["eps"] != .02
            or identity["contract"]["min_samples"] != 3):
        raise ValueError("Witness scientific source differs from frozen RF pairs")
    vector_path = Path(identity["vectors_path"])
    engine._assert_source_stat_identity(vector_path, expected_stat=identity["vectors_stat_identity"], phase="global witness adoption")
    ledgers = engine._load_progress_ledgers(state, identity=identity, num_samples=pairs["rows"])
    selection_path = root / "adaptive_anchor_selection.json"
    anchors, selection = engine._validate_adaptive_selection_manifest(
        path=selection_path, expected_sha256=file_sha(selection_path), root=root,
        identity=identity, progress_ledgers=ledgers)
    details = selection["selection_identity"]
    failures = np.load(details["failure_indices_path"], allow_pickle=False).astype(np.int64)
    if len(anchors) != 55 or len(failures) != 52:
        raise ValueError("Current source is not authorized 55-anchor/52-failure witness scope")
    if engine._source_stat_identity(checkpoint_path) != before:
        raise ValueError("Prior completed ledger changed during adoption")
    result = {"schema": SCHEMA, "old_shortcut_state": "INCONCLUSIVE",
              "vectors_path": str(vector_path), "vectors_sha256": identity["vectors_sha256"],
              "vectors_stat_identity": identity["vectors_stat_identity"],
              "rows": pairs["rows"], "dimension": 64, "dtype": "float32",
              "eps": .02, "min_samples": 3, "self_counts": True,
              "sklearn_version": details["sklearn_version"],
              "selection_sha256": file_sha(selection_path), "anchor_ids": anchors.tolist(),
              "failure_ids": failures.tolist(), "seed_ids": details["seed_indices"],
              "seed_failure_ledger_complete": True,
              "seed_ledger_sha": details["adaptive_seed_progress_ledger_sha256"],
              "failure_ledger_sha": details["adaptive_failure_progress_ledger_sha256"],
              "nonfailure_core_proof": "own row plus at least two actual frozen seed neighbors",
              "pair_rows_recomputed": 0, "large_hash_receipts_reused": True}
    return result


def scan(vectors: np.ndarray, *, anchors: list[int], failures: list[int], seeds: list[int],
         eps: float, min_samples: int, expected_version: str, output: Path,
         binding: dict[str, Any], max_seconds: float = 7200, block_rows: int = 4096,
         boundary: Callable[[], None] | None = None, stop_after_blocks: int | None = None,
         max_new_rows: int = 1048576):
    """Invert bounded source->anchor queries; retain IDs/distances, not lists of N neighbors."""
    if vectors.dtype != np.float32 or vectors.ndim != 2 or not 1 <= block_rows <= 4096:
        raise ValueError("Only bounded frozen float32 source representation is supported")
    if not 0 < max_seconds <= 7200 or not 1 <= len(anchors) <= 55:
        raise ValueError("Witness budget exceeds authorization")
    if not 1 <= max_new_rows <= 1048576:
        raise ValueError("Witness window exceeds memory admission")
    output.mkdir(parents=True, exist_ok=True)
    key = digest({"binding": binding, "anchors": anchors, "failures": failures,
                  "seeds": seeds, "eps": eps, "min_samples": min_samples,
                  "kernel": "sklearn NearestNeighbors brute euclidean float32"})
    cp = output / "witness_checkpoint.json"
    started = time.monotonic()
    if cp.exists():
        state = json.loads(cp.read_text())
        checkpoint_sha = state.pop("checkpoint_sha256", None)
        if checkpoint_sha != digest(state):
            raise ValueError("Witness checkpoint content digest mismatch")
        if state.get("binding_sha") != key:
            raise ValueError("Witness checkpoint input binding differs")
    else:
        state = {"schema": SCHEMA, "binding_sha": key, "next_offset": 0,
                 "elapsed_seconds": 0., "comparisons": 0, "blocks": 0,
                 "witnesses": {str(i): [] for i in anchors},
                 "outside_failure_neighbor": {}, "failure_edges": {str(i): [] for i in anchors}}
    failure_set, anchor_set = set(failures), set(anchors)
    model, version = engine._fit_anchor_neighbors(np.asarray(vectors[anchors]), eps=eps)
    if version != expected_version:
        raise ValueError(f"sklearn version mismatch: {version} != {expected_version}")
    # All original seeds must be one connected component; their own mutual
    # edges are measured under the same kernel, not inferred from small norms.
    distances, neighborhoods = model.radius_neighbors(np.asarray(vectors[anchors]), return_distance=True)
    for a, ids, ds in zip(anchors, neighborhoods, distances):
        for local, distance in zip(ids, ds):
            b = anchors[int(local)]
            if b in anchor_set and b != a and b not in state["failure_edges"][str(a)]:
                state["failure_edges"][str(a)].append(b)
    elapsed_prior = state["elapsed_seconds"]
    blocks_this_run = 0
    def save_checkpoint():
        atomic_json(cp, dict(state, checkpoint_sha256=digest(state)))
    def finish_state():
        # Every nonfailure is core and connected to the original seed component
        # iff the complete adopted failure scan says so. A failure gets that
        # attachment only from an actual radius neighbor, never by borrowing a
        # seed's neighbor counts.
        core = {a for a in anchors if len(state["witnesses"][str(a)]) >= min_samples}
        connected = set(seeds)
        for a in anchors:
            if str(a) in state["outside_failure_neighbor"] and a in core:
                connected.add(a)
        changed = True
        while changed:
            changed = False
            for a in core:
                if a not in connected and any(b in connected for b in state["failure_edges"][str(a)]):
                    connected.add(a); changed = True
        seed_connected = {seeds[0]}
        for _ in seeds:
            for a in list(seed_connected):
                seed_connected.update(b for b in state["failure_edges"].get(str(a), []) if b in seeds)
        all_core = len(core) == len(anchors)
        complete_certificate = (all_core and set(failures).issubset(connected)
                                and set(seeds).issubset(seed_connected)
                                and binding.get("seed_failure_ledger_complete") is True)
        return core, connected, complete_certificate
    attempt_start = int(state["next_offset"])
    attempt_stop = min(len(vectors), attempt_start+max_new_rows)
    for start in range(attempt_start, attempt_stop, block_rows):
        if time.monotonic()-started+elapsed_prior >= max_seconds:
            break
        if boundary is not None and blocks_this_run % 16 == 0:
            if boundary() is False:
                break
        stop = min(attempt_stop, start+block_rows)
        block = np.asarray(vectors[start:stop])
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite source row in {start}:{stop}; not silently filtered")
        distances, neighborhoods = model.radius_neighbors(block, return_distance=True)
        for local_row, (ids, ds) in enumerate(zip(neighborhoods, distances)):
            row = start+local_row
            for local_anchor, distance in zip(ids, ds):
                a = anchors[int(local_anchor)]; records = state["witnesses"][str(a)]
                if len(records) < min_samples and all(x["row_id"] != row for x in records):
                    records.append({"row_id": row, "distance": float(distance),
                                    "distance_hex": float(distance).hex(), "self": row == a})
                if row not in failure_set and row != a:
                    state["outside_failure_neighbor"].setdefault(str(a), {"row_id": row, "distance": float(distance)})
                if row in anchor_set and row != a and row not in state["failure_edges"][str(a)]:
                    state["failure_edges"][str(a)].append(row)
        state.update(next_offset=stop, blocks=state["blocks"]+1,
                     comparisons=state["comparisons"]+(stop-start)*len(anchors),
                     elapsed_seconds=elapsed_prior+time.monotonic()-started)
        blocks_this_run += 1
        core, connected, certificate = finish_state()
        if blocks_this_run % 16 == 0 or certificate or stop == len(vectors):
            save_checkpoint()
            atomic_json(output/"progress.json", {"state": "GLOBAL_WITNESS_RUNNING", "pid": os.getpid(),
                "next_offset": stop, "rows": len(vectors), "verified_core_anchors": len(core),
                "connected_to_seed_anchors": len(connected), "elapsed_seconds": state["elapsed_seconds"],
                "sampled_at_unix": time.time(), "pair_rows_recomputed": 0})
        if certificate or (stop_after_blocks is not None and blocks_this_run >= stop_after_blocks):
            break
    save_checkpoint()
    core, connected, certificate = finish_state()
    exhausted = state["next_offset"] == len(vectors)
    result = {"schema": SCHEMA, "state": "GLOBAL_ALL_CORE_ONE_COMPONENT_CERTIFIED" if certificate else "WITNESS_INCONCLUSIVE",
              "binding_sha": key, "checkpoint": str(cp), "verified_core_anchors": len(core),
              "anchor_count": len(anchors), "not_yet_proven_core": sorted(set(anchors)-core),
              "proven_noncore": sorted(set(anchors)-core) if exhausted else [],
              "noncore_requires_complete_domain": True,
              "connected_to_seed_anchors": len(connected), "complete_dbscan_certificate": certificate,
              "certificate_scope": "all rows core and one component via complete seed/failure ledger plus actual witness edges" if certificate else "anchor core/attachment evidence only; no labels adopted",
              "scanned_rows": state["next_offset"], "total_rows": len(vectors),
              "new_rows_this_process": state["next_offset"]-attempt_start,
              "maximum_new_rows_this_process": max_new_rows,
              "exhaustive_source_scan": exhausted, "elapsed_seconds": state["elapsed_seconds"],
              "comparisons": state["comparisons"], "pair_rows_recomputed": 0,
              "numeric_kernel": "sklearn NearestNeighbors brute euclidean float32",
              "eps": eps, "min_samples": min_samples, "approximation_used": False,
              "all_neighbors_materialized": False, "created_at_unix": time.time()}
    atomic_json(output/"witness_result.json", result)
    return result


def run_witness(config, *, recourse_root: Path, evidence_root: Path):
    from .rf_aligned_cluster_phase import sealed_pairs, require_start_admission, PhaseObserver
    from threadpoolctl import threadpool_limits
    pairs = sealed_pairs(config, recourse_root)
    plan = memory_plan(pairs)
    atomic_json(evidence_root/"phase_memory_plan.json", plan)
    require_start_admission(config, plan, evidence_root)
    binding = source_binding(config, recourse_root, pairs)
    atomic_json(evidence_root/"witness_input_binding.json", binding)
    vectors = np.load(binding["vectors_path"], mmap_mode="r", allow_pickle=False)
    with threadpool_limits(limits=2), PhaseObserver(config, plan, evidence_root) as observer:
        result = scan(vectors, anchors=binding["anchor_ids"], failures=binding["failure_ids"],
            seeds=binding["seed_ids"], eps=.02, min_samples=3, expected_version=binding["sklearn_version"],
            output=evidence_root, binding=binding,
            max_seconds=float(config.get("witness_max_seconds", 7200)),
            block_rows=int(config.get("witness_block_rows",4096)),
            boundary=lambda: observer.sample()["state"] == "PASS")
        result["peak_tree_rss_bytes"] = observer.peak_rss
        result["peak_cgroup_usage_bytes"] = observer.peak_usage
        result["minimum_cgroup_headroom_bytes"] = observer.minimum_headroom
    engine._assert_source_stat_identity(Path(binding["vectors_path"]), expected_stat=binding["vectors_stat_identity"], phase="global witness closeout")
    atomic_json(evidence_root/"terminal.json", result)
    return result
