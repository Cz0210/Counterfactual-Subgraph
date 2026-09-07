"""Rebuild native common-recourse on the new, frozen RF0 candidate universe.

This module retains native GREED normalization, DBSCAN parameters, and greedy
cluster medoids. Old cluster labels are never input. Stage resource admission
precedes pair-store creation; insufficient storage cannot truncate science.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from .rf_aligned_pool import atomic_json, compact_graph, digest, file_sha, rf_predict


def restore_graph(record, parents):
    import torch
    from .graph_trace import stable_untyped_graph_sha256
    compact = record["graph"]
    graph = parents[compact["source_parent_id"]].clone()
    graph.x = torch.nn.functional.one_hot(torch.tensor(compact["labels"]), compact["feature_count"]).to(getattr(torch, compact["x_dtype"].split(".")[-1]))
    graph.edge_index = torch.tensor(compact["edge_index"], dtype=torch.long)
    graph.edge_attr = None
    graph.num_nodes = len(compact["labels"])
    graph.comrecgc_node_origin = torch.tensor(compact["node_origin"], dtype=torch.long)
    graph.gcf_node_origin = graph.comrecgc_node_origin
    if stable_untyped_graph_sha256(graph) != record["stable_graph_sha256"]:
        raise ValueError("Compact candidate does not round-trip to frozen native identity")
    return graph


def storage_plan(*, parent_count: int, candidate_count: int, vector_dim: int,
                 free_bytes: int) -> dict[str, Any]:
    pairs_upper = parent_count * candidate_count
    # ExternalPairStore retains chunk arrays plus final arrays. DBSCAN stores
    # bounded labels/core/count/checkpoint arrays; the summary's vector copies
    # are RAM, not a third persistent vector file. Keep these domains separate.
    peak = pairs_upper * (vector_dim * 4 * 2 + 16 * 2 + 64) + 512 * 1024**2
    reserve = max(2 * 1024**3, free_bytes // 5)
    return {"pair_rows_upper_bound": pairs_upper, "projected_peak_bytes": peak,
            "free_bytes": free_bytes, "reserve_bytes": reserve,
            "state": "PASS" if peak <= free_bytes - reserve else "BLOCKED_STORAGE",
            "pair_universe_truncated": False}


def run_recourse(config: Mapping[str, Any], *, pool_root: Path, output_root: Path):
    import numpy as np
    import torch
    import sklearn
    from torch_geometric.data import Batch
    from .project_dataset import load_aids_generation_bundle
    from .model_adapter import AIDSGreedEmbeddingAdapter
    from .upstream import imported_upstream
    from .external_memory_recourse import ExternalPairStore, trace_external_cluster_order
    from .external_memory_dbscan import ExternalDBSCANContract, fit_external_memory_dbscan, ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
    from src.rewards.reward_calculator import load_oracle_bundle

    terminal = json.loads((pool_root / "terminal.json").read_text())
    if terminal["state"] != "POOL_SCREEN_COMPLETE" or terminal["counts"].get("CACHE_PROVENANCE_GAP", 0):
        raise ValueError("Full pool provenance is not closed; no partial-universe clustering")
    pool_config = json.loads((pool_root / "contract.json").read_text())
    if terminal["contract_sha"] != digest(pool_config) or any(config.get(key) != value for key, value in pool_config.items()):
        raise ValueError("Pool screening contract binding differs")
    if not terminal["unique_rf_target0"]:
        atomic_json(output_root / "terminal.json", {"state": "POOL_NO_RF0_COUNTERFACTUALS", "next_required_stage": "ONE_RF_GUIDED_NATIVE_VRRW", "new_search_started": False})
        return
    output_root.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    source = load_aids_generation_bundle(dataset_dir=config["dataset_dir"], source_csv=config["source_csv"])
    parents = dict(zip(source.parent_ids, source.graphs, strict=True))
    predictions = json.loads((pool_root / "source_predictions.json").read_text())
    if [x["parent_id"] for x in predictions["rows"]] != source.parent_ids:
        raise ValueError("RF source mask does not bind to original 1283 order")
    source_positions = [i for i, x in enumerate(predictions["rows"]) if x["prediction"] == 1]
    graph_sources = [source.graphs[i] for i in source_positions]
    records = []
    for path in sorted((pool_root / "segments").glob("segment-*.json")):
        segment = json.loads(path.read_text())
        if segment["contract_sha"] != terminal["contract_sha"]:
            raise ValueError("Candidate segment binding differs")
        records.extend(row for row in segment["rows"] if row["state"] == "RF_TARGET0")
    graphs = [restore_graph(row, parents) for row in records]
    torch.set_num_threads(int(config.get("threads", 2)))
    batch_size = 128
    model = AIDSGreedEmbeddingAdapter(config["greed_path"], atom_vocabulary=source.atom_vocabulary, device="cpu").eval()
    # Preserve the original all-parent batching before applying the RF source
    # mask, so padding and graph embedding numerical context stay frozen.
    all_embeddings = model.embed_model(Batch.from_data_list(source.graphs)).detach().cpu()
    source_embeddings = all_embeddings[source_positions]
    fs = os.statvfs(output_root)
    plan = storage_plan(parent_count=len(graph_sources), candidate_count=len(graphs), vector_dim=source_embeddings.shape[1], free_bytes=fs.f_bavail * fs.f_frsize)
    plan.update(source_denominator=1283, rf_source1_count=len(graph_sources), candidate_count=len(graphs), graph_unique_count=len({r["stable_graph_sha256"] for r in records}), chemical_unique_count=len({r["canonical_smiles"] for r in records}), vector_dim=source_embeddings.shape[1])
    atomic_json(output_root / "resource_admission.json", plan)
    if plan["state"] != "PASS":
        return plan
    identity = {"schema": "aids_rf_aligned_native_recourse_v2", "pool_contract": terminal["contract_sha"], "pool_terminal_sha": file_sha(pool_root / "terminal.json"), "candidate_graphs": [r["stable_graph_sha256"] for r in records], "source_positions": source_positions, "source_denominator": 1283, "greed_sha": config["greed_sha256"], "rf_sha": config["rf_sha256"], "theta": .1, "eps": .02, "min_samples": 3, "metric": "euclidean", "pair_order": "candidate_major_parent_minor", "old_dbscan_labels_reused": False}
    atomic_json(output_root / "universe_manifest.json", identity)
    max_rss = 28 * 1024**3
    store = ExternalPairStore(root=output_root / "pair_store", scientific_identity=identity, max_rss_bytes=max_rss, resume=True)
    with imported_upstream(config["upstream_root"]) as modules:
        source_counts = modules["util"].graph_element_counts(source.graphs).cpu()[source_positions]
        for chunk_index, begin in enumerate([] if store.complete else range(0, len(graphs), batch_size)):
            stop = min(len(graphs), begin + batch_size)
            chunk_identity = {"start": begin, "stop": stop, "candidate_ids": [r["stable_graph_sha256"] for r in records[begin:stop]]}
            if chunk_index < store.next_chunk_index:
                store.verify_completed_chunk(chunk_index=chunk_index, chunk_identity=chunk_identity)
                continue
            chunk = graphs[begin:stop]
            embeddings = model.embed_model(Batch.from_data_list(chunk)).detach().cpu()
            scale = modules["util"].graph_element_counts(chunk).cpu()[:, None] + source_counts[None, :]
            distances = torch.cdist(embeddings, source_embeddings, p=2) / scale
            selected = torch.nonzero(distances <= .1, as_tuple=False)
            vectors = (embeddings[selected[:, 0]] - source_embeddings[selected[:, 1]]) / scale[selected[:, 0], selected[:, 1], None]
            pairs = np.asarray([(source_positions[int(parent)], begin + int(cf)) for cf, parent in selected.tolist()], dtype=np.int64).reshape(-1, 2)
            store.append(chunk_index=chunk_index, pairs=pairs, vectors=vectors.numpy(), chunk_identity=chunk_identity)
            if len(pairs) and not (output_root / "first_valid_native_pair.json").exists():
                parent_position, cf_index = pairs[0]
                atomic_json(output_root / "first_valid_native_pair.json", {"stage": "RF_VALID_NATIVE_PAIR_NOT_YET_CLUSTER_MEDOID", "parent_id": source.parent_ids[parent_position], "candidate_index": records[cf_index]["candidate_index"], "candidate_smiles": records[cf_index]["canonical_smiles"], "pred_before": 1, "pred_after": 0, "rf": records[cf_index]["rf"], "normalized_greed_distance": float(distances[selected[0, 0], selected[0, 1]]), "candidate_graph_sha": records[cf_index]["stable_graph_sha256"]})
            atomic_json(output_root / "progress.json", {"stage": "PAIR_STORE", "candidates_completed": stop, "candidate_count": len(graphs), "elapsed_seconds": time.monotonic() - start_time})
        pair_result = store.finalize()
        if pair_result.row_count == 0:
            atomic_json(output_root / "terminal.json", {"state": "POOL_NO_THETA_CLOSE_RF_RECOURSE", "next_required_stage": "ONE_RF_GUIDED_NATIVE_VRRW"})
            return
        contract = ExternalDBSCANContract(eps=.02, min_samples=3, query_block_size=8, checkpoint_interval_blocks=1, max_rss_bytes=max_rss, expected_sklearn_version=config["expected_sklearn_version"], shortcut_mode=ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT, exact_fallback_max_samples=100000)
        # For >100k points this engine must prove its adaptive exact shortcut;
        # failure is explicit, never silent N-squared brute-force fallback.
        cluster = fit_external_memory_dbscan(vectors_path=pair_result.vectors_path, work_dir=output_root / "dbscan", contract=contract, expected_vectors_sha256=pair_result.vectors_sha256, resume=True)
        vectors = np.load(pair_result.vectors_path, mmap_mode="r")
        pairs = np.load(pair_result.pairs_path, mmap_mode="r")
        labels = np.load(cluster.labels_path, mmap_mode="r")
        selected, summary_audit = trace_external_cluster_order(labels=labels, recourse_vectors=vectors, pair_indices=pairs, radius=.02, theta=.1, recourse_size=100, official_greedy=modules["common_recourse"].greedy_counterfactual_summary_from_covering_sets, max_rss_bytes=max_rss)
    rf = load_oracle_bundle(config["rf_path"])
    rows = []
    for item in selected:
        record = records[item["representative_counterfactual_index"]]
        score = rf_predict([record["canonical_smiles"]], rf)[0]
        if score["prediction"] != 0:
            raise ValueError("Cluster medoid no longer passes frozen RF target0")
        rows.append(dict(item, canonical_smiles=record["canonical_smiles"], rf=score, original_candidate_index=record["candidate_index"], original_parent_id=record["parent_id"], graph=record["graph"]))
    atomic_json(output_root / "selected_native_recourses.json", rows)
    atomic_json(output_root / "terminal.json", {"state": "RF_ALIGNED_NATIVE_SUMMARY_COMPLETE" if rows else "POOL_NO_VALID_COMMON_SUMMARY", "selected_count": len(rows), "source_denominator": 1283, "source1_count": len(source_positions), "candidate_count": len(records), "pair_rows": pair_result.row_count, "dbscan_cluster_count": cluster.cluster_count, "summary_audit": summary_audit, "old_cluster_labels_reused": False, "evaluation_complete": False, "test_loaded": False, "elapsed_seconds": time.monotonic() - start_time})
