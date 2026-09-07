"""Rebuild native common-recourse on the new, frozen RF0 candidate universe.

This module retains native GREED normalization, DBSCAN parameters, and greedy
cluster medoids. Old cluster labels are never input. Stage resource admission
precedes pair-store creation; insufficient storage cannot truncate science.
"""
from __future__ import annotations

import json
import os
import shutil
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
                 free_bytes: int, exact_pair_count: int | None = None) -> dict[str, Any]:
    pairs_upper = parent_count * candidate_count
    pairs_needed = pairs_upper if exact_pair_count is None else exact_pair_count
    if not 0 <= pairs_needed <= pairs_upper:
        raise ValueError("Exact pair count is outside the candidate/source universe")
    # ExternalPairStore retains chunk arrays plus final arrays. DBSCAN stores
    # bounded labels/core/count/checkpoint arrays; the summary's vector copies
    # are RAM, not a third persistent vector file. Keep these domains separate.
    peak = pairs_needed * (vector_dim * 4 * 2 + 16 * 2 + 64) + 512 * 1024**2
    reserve = max(2 * 1024**3, free_bytes // 5)
    return {"pair_rows_upper_bound": pairs_upper, "exact_pair_count": exact_pair_count, "projected_peak_bytes": peak,
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
    from .rf_aligned_count import count_exact_pairs, decode_mask, rss_bytes
    from src.rewards.reward_calculator import load_oracle_bundle

    terminal = json.loads((pool_root / "terminal.json").read_text())
    if terminal["state"] != "POOL_SCREEN_COMPLETE" or terminal["counts"].get("CACHE_PROVENANCE_GAP", 0):
        raise ValueError("Full pool provenance is not closed; no partial-universe clustering")
    pool_config = json.loads((pool_root / "contract.json").read_text())
    if terminal["contract_sha"] != digest(pool_config) or any(config.get(key) != value for key, value in pool_config.items()):
        raise ValueError("Pool screening contract binding differs")
    original_config = dict(config)
    runtime_paths = dict(config.get("runtime_paths", {}))
    allowed_paths = {"dataset_dir", "source_csv", "rf_path", "greed_path", "upstream_root"}
    if set(runtime_paths) - allowed_paths or any(not Path(value).is_absolute() for value in runtime_paths.values()):
        raise ValueError('Unsupported runtime path remapping')
    config = dict(config, **runtime_paths)
    if not terminal["unique_rf_target0"]:
        atomic_json(output_root / "terminal.json", {"state": "POOL_NO_RF0_COUNTERFACTUALS", "next_required_stage": "ONE_RF_GUIDED_NATIVE_VRRW", "new_search_started": False})
        return
    output_root.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    max_rss = int(config.get('max_rss_bytes', 28 * 1024**3))
    # Bound deserialization before creating a full graph list. This reserves
    # model/runtime overhead and expansion of PyG and JSON inputs, not host RAM.
    input_bytes = Path(config["dataset_dir"], "graphs.pt").stat().st_size + sum(p.stat().st_size for p in (pool_root / "segments").glob("segment-*.json"))
    initial_ram = {"current_rss_bytes": rss_bytes(), "input_bytes": input_bytes, "projected_input_expansion_bytes": input_bytes * 12 + 2 * 1024**3, "max_rss_bytes": max_rss}
    initial_ram["state"] = "PASS" if initial_ram["projected_input_expansion_bytes"] + rss_bytes() <= max_rss else "BLOCKED_RAM"
    atomic_json(output_root / "input_ram_admission.json", initial_ram)
    if initial_ram["state"] != "PASS":
        return initial_ram
    source = load_aids_generation_bundle(dataset_dir=config["dataset_dir"], source_csv=config["source_csv"])
    prior_source = json.loads((pool_root / 'source_input_binding.json').read_text())
    if source.dataset_fingerprint != prior_source['dataset_fingerprint']:
        raise ValueError('Remapped source dataset is not the frozen native source')
    atomic_json(output_root / 'runtime_path_binding.json', {'original_paths': {k: original_config[k] for k in runtime_paths}, 'effective_paths': runtime_paths, 'dataset_fingerprint': source.dataset_fingerprint, 'pool_contract_sha': terminal['contract_sha'], 'original_contract_changed': False})
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
    tensor_bytes = sum((len(r["graph"]["labels"]) * (r["graph"]["feature_count"] * 4 + 8) + len(r["graph"]["edge_index"][0]) * 16) for r in records)
    max_nodes = max(int(g.num_nodes) for g in source.graphs)
    batch_bound = len(source.graphs) * max_nodes**2 * 16 + 2 * 1024**3
    graph_ram = {"current_rss_bytes": rss_bytes(), "candidate_tensor_bytes": tensor_bytes, "candidate_expansion_bound": tensor_bytes * 4, "all_parent_batch_bound": batch_bound, "max_rss_bytes": max_rss}
    graph_ram["state"] = "PASS" if rss_bytes() + tensor_bytes * 4 + batch_bound <= max_rss else "BLOCKED_RAM"
    atomic_json(output_root / "graph_ram_admission.json", graph_ram)
    if graph_ram["state"] != "PASS":
        return graph_ram
    graphs = [restore_graph(row, parents) for row in records]
    torch.set_num_threads(int(config.get("threads", 2)))
    batch_size = 128
    model = AIDSGreedEmbeddingAdapter(config["greed_path"], atom_vocabulary=source.atom_vocabulary, device="cpu").eval()
    identity = {"schema": "aids_rf_aligned_native_recourse_v2", "pool_contract": terminal["contract_sha"], "pool_terminal_sha": file_sha(pool_root / "terminal.json"), "candidate_graphs": [r["stable_graph_sha256"] for r in records], "source_positions": source_positions, "source_denominator": 1283, "greed_sha": config["greed_sha256"], "rf_sha": config["rf_sha256"], "theta": .1, "eps": .02, "min_samples": 3, "metric": "euclidean", "pair_order": "candidate_major_parent_minor", "old_dbscan_labels_reused": False}
    identity_path = output_root / "universe_manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resumed RF recourse universe differs")
    atomic_json(identity_path, identity)
    reuse_root = config.get('count_reuse_root')
    if reuse_root and not (output_root / 'exact_count').exists():
        previous_count = Path(reuse_root)
        previous_manifest = json.loads((previous_count / 'manifest.json').read_text())
        if previous_manifest.get('identity_sha') != digest(identity):
            raise ValueError('Transferred count is from a different scientific universe')
        shutil.copytree(previous_count, output_root / 'exact_count')
        atomic_json(output_root / 'count_adoption.json', {'source': str(previous_count), 'destination': str(output_root / 'exact_count'), 'source_manifest_sha': file_sha(previous_count / 'manifest.json'), 'identity_sha': digest(identity), 'original_preserved': True, 'model_inference_required': False})
    with imported_upstream(config["upstream_root"]) as modules:
        count, source_embeddings, source_counts = count_exact_pairs(graphs=graphs, all_sources=source.graphs, source_positions=source_positions, model=model, element_counts=modules["util"].graph_element_counts, identity=identity, root=output_root / "exact_count", max_rss_bytes=max_rss, batch_size=batch_size)
        first_chunk = next((x for x in count["chunks"] if x["first_close_pair"] is not None), None)
        if first_chunk is not None:
            first = first_chunk["first_close_pair"]
            candidate_index = first_chunk["begin"] + first["local_candidate"]
            parent_position = source_positions[first["local_parent"]]
            atomic_json(output_root / "first_valid_native_pair.json", {"stage": "RF_VALID_NATIVE_PAIR_NOT_YET_CLUSTER_MEDOID", "parent_id": source.parent_ids[parent_position], "candidate_index": records[candidate_index]["candidate_index"], "candidate_smiles": records[candidate_index]["canonical_smiles"], "pred_before": 1, "pred_after": 0, "rf": records[candidate_index]["rf"], "normalized_greed_distance": first["normalized_greed_distance"], "candidate_graph_sha": records[candidate_index]["stable_graph_sha256"]})
        fs = os.statvfs(output_root)
        plan = storage_plan(parent_count=len(graph_sources), candidate_count=len(graphs), vector_dim=source_embeddings.shape[1], free_bytes=fs.f_bavail * fs.f_frsize, exact_pair_count=count["pair_count"])
        plan.update(source_denominator=1283, rf_source1_count=len(graph_sources), candidate_count=len(graphs), vector_dim=source_embeddings.shape[1], rss_bytes=rss_bytes(), count_manifest=str(output_root / "exact_count/manifest.json"))
        atomic_json(output_root / "resource_admission.json", plan)
        if plan["state"] != "PASS":
            return plan
        store = ExternalPairStore(root=output_root / "pair_store", scientific_identity=identity, max_rss_bytes=max_rss, resume=True)
        for chunk_index, begin in enumerate([] if store.complete else range(0, len(graphs), batch_size)):
            stop = min(len(graphs), begin + batch_size)
            chunk_identity = {"start": begin, "stop": stop, "candidate_ids": [r["stable_graph_sha256"] for r in records[begin:stop]]}
            if chunk_index < store.next_chunk_index:
                store.verify_completed_chunk(chunk_index=chunk_index, chunk_identity=chunk_identity)
                continue
            counted_chunk = count["chunks"][chunk_index]
            if file_sha(Path(counted_chunk["path"])) != counted_chunk["sha256"]:
                raise ValueError("Frozen exact-count arrays changed")
            with np.load(counted_chunk["path"], allow_pickle=False) as saved:
                embeddings = torch.from_numpy(saved["embeddings"].copy())
                candidate_counts = torch.from_numpy(saved["counts"].copy())
                mask = decode_mask(saved["mask"], counted_chunk["mask_shape"])
            scale = candidate_counts[:, None] + source_counts[None, :]
            selected = torch.from_numpy(np.argwhere(mask))
            vectors = (embeddings[selected[:, 0]] - source_embeddings[selected[:, 1]]) / scale[selected[:, 0], selected[:, 1], None]
            pairs = np.asarray([(source_positions[int(parent)], begin + int(cf)) for cf, parent in selected.tolist()], dtype=np.int64).reshape(-1, 2)
            store.append(chunk_index=chunk_index, pairs=pairs, vectors=vectors.numpy(), chunk_identity=chunk_identity)
            if len(pairs) and not (output_root / "first_valid_native_pair.json").exists():
                parent_position, cf_index = pairs[0]
                atomic_json(output_root / "first_valid_native_pair.json", {"stage": "RF_VALID_NATIVE_PAIR_NOT_YET_CLUSTER_MEDOID", "parent_id": source.parent_ids[parent_position], "candidate_index": records[cf_index]["candidate_index"], "candidate_smiles": records[cf_index]["canonical_smiles"], "pred_before": 1, "pred_after": 0, "rf": records[cf_index]["rf"], "normalized_greed_distance": counted_chunk["first_close_pair"]["normalized_greed_distance"], "candidate_graph_sha": records[cf_index]["stable_graph_sha256"]})
            atomic_json(output_root / "progress.json", {"stage": "PAIR_STORE", "candidates_completed": stop, "candidate_count": len(graphs), "elapsed_seconds": time.monotonic() - start_time})
        pair_result = store.finalize()
        if config.get('stop_after_pair_store', False):
            atomic_json(output_root / 'pair_stage_terminal.json', {'state': 'PAIR_STORE_COMPLETE_WAITING_DBSCAN_RESOURCE', 'pair_rows': pair_result.row_count, 'pair_manifest': str(pair_result.manifest_path), 'pair_manifest_sha': pair_result.manifest_sha256, 'max_rss_bytes': max_rss, 'dbscan_started': False})
            return
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
        clustering_receipt = json.loads(cluster.manifest_path.read_text())
        greedy = modules["common_recourse"].greedy_counterfactual_summary_from_covering_sets
        if clustering_receipt["clustering_path"] == "all_core_adaptive_anchor_component_recovery_v1" or (cluster.shortcut_proof_path is not None and cluster.cluster_count == 1):
            from .external_component_summary import summarize_proven_all_core_components_external
            from .external_memory_recourse import summarize_proven_one_cluster_external
            common = dict(work_dir=output_root / "streamed_native_summary", dbscan_manifest_path=cluster.manifest_path, dbscan_manifest_sha256=cluster.manifest_sha256, recourse_vectors=vectors, pair_indices=pairs, pairs_sha256=pair_result.pairs_sha256, pair_authority_manifest_path=pair_result.manifest_path, pair_authority_manifest_sha256=pair_result.manifest_sha256, radius=.02, theta=.1, recourse_size=100, official_greedy=greedy, torch_module=torch, max_rss_bytes=max_rss, block_size=65536, resume=True)
            if clustering_receipt["clustering_path"] == "all_core_adaptive_anchor_component_recovery_v1":
                result = summarize_proven_all_core_components_external(labels=labels, **common)
            else:
                result = summarize_proven_one_cluster_external(**common)
            selected = result.selected
            summary_audit = {"streaming_summary_manifest": str(result.manifest_path), "manifest_sha256": result.manifest_sha256}
        else:
            selected, summary_audit = trace_external_cluster_order(labels=labels, recourse_vectors=vectors, pair_indices=pairs, radius=.02, theta=.1, recourse_size=100, official_greedy=greedy, max_rss_bytes=max_rss)
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
