"""BACE train-only Reach-v2 stages; no implicit test access or main publisher.

The main exact evaluator is reused with an explicit attributed-graph enumerator.
Each parent/pass is a committed recovery boundary. Calibration cannot feed the
search: its function requires an already sealed candidate universe.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.chem.bace_reach_search import SearchBudget, deletion_outcomes, retain_train_pool, search_parent
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file, stable_sha256, load_bace_parents, utc_now
from src.eval.bace_reach_selector import ReachMasks, select_nested

SCHEMA = "bace_ours_reach_v2_20260907"


def seal(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["self_sha256"] = stable_sha256(value)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("IMMUTABLE_REACH_RECEIPT_EXISTS:" + str(path))
    else:
        atomic_json(path, value)
    return value


def unseal(path: Path) -> dict[str, Any]:
    value = read_json(path)
    digest = value.pop("self_sha256")
    if stable_sha256(value) != digest:
        raise ValueError("REACH_RECEIPT_HASH_CONFLICT:" + str(path))
    return {**value, "self_sha256": digest}


def plan(reference_path: Path, output: Path, *, proposal_source: str,
         proposal_path: Path | None = None) -> dict[str, Any]:
    """Freeze inputs before scientific execution; do not open test payloads."""
    ref = read_json(reference_path)
    if ref.get("status") != "PASS" or ref.get("dataset") != "bace" or ref.get("method") != "ours":
        raise ValueError("BACE_FROZEN_REFERENCE_REQUIRED")
    down = ref["frozen_downstream"]
    if proposal_source not in {"OURS_MAIN_PPO_66", "L0", "L1", "L2", "L3"}:
        raise ValueError("EXPLICIT_PROPOSAL_SOURCE_REQUIRED")
    if proposal_source != "OURS_MAIN_PPO_66":
        # Public graph/search/evaluation functions are reusable, but a variant
        # runner must supply its OWN pre-search selection and calibration rows.
        # Reject here rather than sealing a misleading Ours-derived LLM spec.
        raise ValueError("VARIANT_SPEC_REQUIRES_OWN_SELECTOR_AND_CALIBRATION_BINDING_NOT_IMPLEMENTED")
    if proposal_source == "OURS_MAIN_PPO_66":
        expected = Path(ref["candidate_generation"]["merge_manifest"]["path"]).parent / "candidate_universe.jsonl"
        if proposal_path and proposal_path.resolve() != expected.resolve():
            raise ValueError("OURS_MAIN_POOL_SOURCE_CHANGED")
        proposal_path = expected
        if sha256_file(proposal_path) != ref["candidate_generation"]["candidate_universe_sha256"]:
            raise ValueError("OURS_MAIN_POOL_HASH_CONFLICT")
    elif proposal_path is None:
        raise ValueError("ABLATION_MUST_PROVIDE_ITS_OWN_POOL_NO_OURS_INJECTION")
    rows = read_jsonl(proposal_path)
    ids = [str(x["candidate_id"]) for x in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("PROPOSAL_UNIVERSE_EMPTY_OR_DUPLICATE")
    if proposal_source == "OURS_MAIN_PPO_66" and len(rows) != 66:
        raise ValueError("EXPECTED_OURS_OLD_66")
    selector = down["selector_contract"]
    paths = {"proposal": str(proposal_path), "reference": str(reference_path),
             "oracle": str(Path(down["gine_checkpoint"]).parent),
             "molclr_checkpoint": down["molclr_root"],
             "molclr_source": str(Path(down["molclr_root"]).parents[3]),
             "train": down["dataset_split_paths"]["train"],
             "calibration": down["dataset_split_paths"]["calibration"],
             "thresholds": selector["thresholds"]["path"],
             "old_selector": selector["selector_manifest"]["path"],
             "old_calibration_pairs": str(Path(selector["verified_matrix_manifest"]["path"]).parent / "pair_matrix.jsonl")}
    for key in ("train", "calibration"):
        if sha256_file(paths[key]) != down["dataset_split_hashes"][key]:
            raise ValueError("SPLIT_HASH_CONFLICT:" + key)
    for key, descriptor in (("thresholds", selector["thresholds"]), ("old_selector", selector["selector_manifest"])):
        if sha256_file(paths[key]) != descriptor["sha256"]:
            raise ValueError("FROZEN_SELECTOR_INPUT_CONFLICT:" + key)
    if not (Path(paths["molclr_source"]) / "models" / "gin.py").is_file():
        raise ValueError("MOLCLR_SOURCE_LAYOUT_UNRESOLVED:" + paths["molclr_source"])
    output.mkdir(parents=True, exist_ok=True)
    budget = asdict(SearchBudget())
    return seal(output / "search_contract.json", {
        "schema": SCHEMA, "method_version": "Ours-Reach-v2", "proposal_source": proposal_source,
        "paths": paths, "reference_sha256": sha256_file(reference_path),
        "proposal_sha256": sha256_file(proposal_path), "old_candidate_ids": ids,
        "search_budget": budget, "search_budget_sha256": stable_sha256(budget),
        "source_label": ref["source_label"], "oracle_weight_sha256": down["gine_checkpoint_sha"],
        "temperature_sha256": down["temperature_sha"], "temperature": down["temperature"],
        "molclr_sha256": down["molclr_sha"], "wnode_config": down["wnode_config"],
        "oracle_binding": stable_sha256({k: down[k] for k in ("gine_checkpoint_sha", "temperature_sha")}),
        "size_contract_source": "main B8/B9 connected hard-deletion kernel: >=1 deletion atom, nonempty valid connected residual; no additional fraction filter",
        "pool_freeze_precedes_calibration": True, "test_opened": False,
        "source_science_commit": ref["execution_commit"],
        "conditional_ppo": {"enabled_now": False, "max_additional_updates": 300,
            "max_campaigns": 1, "required_train_reach_below": 0.9,
            "requires_bounded_search_complete": True, "requires_proposal_gap_dominance_evidence": True,
            "adapter": ref["ppo"]["adapter_weights"], "original_updates": ref["ppo"]["optimizer_updates"],
            "prompt": "native_chat", "projection_disabled": True,
            "reward_change": "original_reward + 2 * frozen_rollout_undercoverage_weight * valid_strict_flip",
            "execution_state": "SEALED_CONDITIONAL_NOT_DISPATCHABLE_UNTIL_TRAIN_GATE"},
        "controls": ["old_pool_old_selector", "new_pool_old_selector", "new_pool_reach_first"],
        "predetermined_final_control": "new_pool_reach_first", "matrix_write_authorized": False})


def evaluate_pairs(parents, candidates, *, oracle, featurizer, distance_provider,
                   split, oracle_checkpoint_id, oracle_batch_size=64, parent_prediction_cache=None):
    """Reusable v2 attributed pattern adapter; all old evaluator semantics stay."""
    from src.eval.bace_frozen_gnn_verification import _evaluate_rows
    pairs, matches = _evaluate_rows(parents, candidates, oracle=oracle, featurizer=featurizer,
        distance_provider=distance_provider, oracle_batch_size=oracle_batch_size,
        split=split, oracle_checkpoint_id=oracle_checkpoint_id,
        parent_prediction_cache=parent_prediction_cache, outcome_enumerator=deletion_outcomes)
    gaps = [m for m in matches if m["teacher_strict_flip"] and not m["distance_ok"]]
    if gaps:
        raise ValueError("CACHE_PROVENANCE_OR_DISTANCE_GAP:" + json.dumps(gaps[0], sort_keys=True))
    return pairs, matches


def load_runtime(contract, output, device):
    from src.eval.bace_frozen_gnn_pool import _checkpoint_contract
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
    from src.oracles.gnn_oracle import GNNOracle
    from src.eval.node_wasserstein_distance import MolCLRNodeWassersteinConfig, MolCLRNodeWassersteinDistance
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    paths = contract["paths"]
    _, schema = _checkpoint_contract(Path(paths["oracle"]))
    featurizer = MolecularGraphFeaturizer(schema=schema)
    # Immutable receipt already binds the weights. Do not rehash all model files.
    oracle = GNNOracle.from_checkpoint(paths["oracle"], device=device, verify_hashes=False)
    if oracle.temperature != contract["temperature"]:
        raise ValueError("FROZEN_GINE_TEMPERATURE_CHANGED")
    cfg = contract["wnode_config"]
    if cfg["solver"] != "exact_emd2" or cfg["size_penalty_beta"] != 0:
        raise ValueError("FROZEN_WNODE_CONTRACT_NOT_SUPPORTED")
    distance = MolCLRNodeWassersteinDistance(MolCLRNodeWassersteinConfig(
        molclr_root=paths["molclr_source"], molclr_ckpt=paths["molclr_checkpoint"],
        cache_db=output / "cache" / "distance.sqlite", node_emb_cache_dir=output / "cache" / "nodes",
        device=device, feature_cost=cfg["feature_cost"], node_mass=cfg["node_mass"],
        size_penalty_beta=cfg["size_penalty_beta"], distance_namespace=cfg["distance_namespace"]))
    install_compact_node_cache(distance)
    return oracle, featurizer, distance


def predict_smiles(oracle, featurizer, smiles, split):
    from src.eval.bace_frozen_gnn_verification import _graph
    return oracle.predict_records([_graph(featurizer, smiles=s, molecule_id=str(i), split=split)
                                   for i, s in enumerate(smiles)], batch_size=64)


def summary(pairs, parent_ids, ordered, thresholds):
    by_pair = {(r["parent_id"], r["candidate_id"]): r for r in pairs}
    result = {}
    for name, ids in (("S10", ordered[:10]), ("S20", ordered[:20]), ("all", sorted({r["candidate_id"] for r in pairs}))):
        ds = [min((float(by_pair[p, c]["distance_for_selection"]) for c in ids), default=math.inf) for p in parent_ids]
        result[name] = {"parents": len(ds), "finite_strict_flip": sum(math.isfinite(x) for x in ds),
            "high_cap_covered": sum(x <= thresholds["cost_cap"] for x in ds),
            "theta_star_covered": sum(x <= thresholds["theta_star"] for x in ds)}
    return result


def run_train(output: Path, *, device: str, boundary_check=lambda: None, canary_parents: int = 0):
    contract = unseal(output / "search_contract.json")
    if (output / "candidate_freeze.json").exists():
        return unseal(output / "candidate_freeze.json")
    if canary_parents not in (0, 1, 2):
        raise ValueError("BOUNDED_CANARY_PARENT_LIMIT")
    work = output / ("canary" if canary_parents else "train")
    work.mkdir(parents=True, exist_ok=True)
    old = read_jsonl(contract["paths"]["proposal"])
    if sha256_file(contract["paths"]["proposal"]) != contract["proposal_sha256"]:
        raise ValueError("PROPOSAL_SOURCE_CHANGED")
    parents = load_bace_parents(contract["paths"]["train"], source_label=contract["source_label"])
    if canary_parents:
        parents = parents[:canary_parents]
    oracle, features, distance = load_runtime(contract, work, device)
    thresholds = read_json(contract["paths"]["thresholds"])
    selected = read_json(contract["paths"]["old_selector"])["ordered_rule_ids"]
    if contract["proposal_source"] != "OURS_MAIN_PPO_66":
        raise ValueError("ABLATION_OLD_SELECTOR_MUST_BE_EXPLICIT_VARIANT_SELECTOR_NOT_OURS")
    budget = SearchBudget(**contract["search_budget"])
    states, all_pairs = [], []
    started = time.monotonic()
    for i, parent in enumerate(parents):
        boundary_check()
        token = stable_sha256({"id": parent.parent_id, "smiles": parent.smiles})[:24]
        file = work / (token + ".json")
        if file.exists():
            saved = unseal(file)
            if saved["search_contract_sha256"] != contract["self_sha256"]:
                raise ValueError("TRAIN_PARENT_CHECKPOINT_CONTRACT_CHANGED")
        else:
            before = predict_smiles(oracle, features, [parent.smiles], "train")[0]
            pairs, matches = evaluate_pairs([parent], old, oracle=oracle, featurizer=features,
                distance_provider=distance, split="train", oracle_checkpoint_id=contract["oracle_binding"])
            covered = any(r["pair_strict_flip"] for r in pairs)
            state = None
            if int(before["predicted_label"]) == contract["source_label"] and not covered:
                state = search_parent(parent_id=parent.parent_id, parent_smiles=parent.smiles,
                    before=before, predict=lambda ss: predict_smiles(oracle, features, ss, "train_reach_search"),
                    old_candidates=old, oracle_binding=contract["oracle_binding"], budget=budget,
                    maximum_new_queries=budget.initial_queries_per_parent)
            saved = seal(file, {"search_contract_sha256": contract["self_sha256"], "parent_id": parent.parent_id,
                "before": before, "old_pool_pairs": pairs, "old_pool_matches": matches,
                "old_pool_covered": covered, "search": state, "test_opened": False, "calibration_opened": False})
        all_pairs.extend(saved["old_pool_pairs"])
        if saved["search"]:
            states.append(saved["search"])
        atomic_json(output / "progress.json", {"state": "TRAIN_SEARCH_RUNNING", "completed_parents": i + 1,
            "total_parents": len(parents), "new_graph_oracle_queries": sum(s["new_graph_oracle_queries"] for s in states),
            "elapsed_seconds": time.monotonic() - started, "updated_at": utc_now(), "science_pid": os.getpid()})
    if canary_parents:
        return seal(work / "canary_receipt.json", {"state": "BOUNDED_TRAIN_CANARY_COMPLETE", "parents": len(parents),
            "old_pool": summary(all_pairs, [p.parent_id for p in parents], selected, thresholds),
            "new_graph_queries": sum(s["new_graph_oracle_queries"] for s in states),
            "elapsed_seconds": time.monotonic() - started, "test_opened": False,
            "budget_exceeded": any(s["new_graph_oracle_queries"] > 128 for s in states),
            "search_contract_sha256": contract["self_sha256"]})
    extra_ids = sorted(s["parent_id"] for s in states if not s["witnesses"])[:budget.extra_parent_limit]
    for j, state in enumerate(list(states)):
        if state["parent_id"] not in extra_ids:
            continue
        boundary_check()
        path = work / (stable_sha256(state["parent_id"])[:24] + "-extra.json")
        if path.exists():
            extra = unseal(path)
            if extra["search_contract_sha256"] != contract["self_sha256"] or extra["initial_binding"] != stable_sha256(state):
                raise ValueError("EXTRA_PARENT_RESUME_BINDING_CHANGED")
            states[j] = extra["search"]
        else:
            expanded = search_parent(parent_id=state["parent_id"], parent_smiles=state["parent_smiles"],
                before=state["before"], predict=lambda ss: predict_smiles(oracle, features, ss, "train_reach_search"),
                old_candidates=old, oracle_binding=contract["oracle_binding"], budget=budget,
                maximum_new_queries=budget.extra_queries_per_parent, previous=state)
            seal(path, {"search_contract_sha256": contract["self_sha256"], "initial_binding": stable_sha256(state), "search": expanded})
            states[j] = expanded
    pool = retain_train_pool(old, states, budget.pool_limit)
    atomic_jsonl(output / "candidate_universe.jsonl", pool)
    atomic_json(output / "old_train_diagnostic.json", summary(all_pairs, [p.parent_id for p in parents], selected, thresholds))
    return seal(output / "candidate_freeze.json", {"state": "TRAIN_ONLY_POOL_FROZEN", "schema": SCHEMA,
        "search_contract_sha256": contract["self_sha256"], "proposal_source": contract["proposal_source"],
        "candidate_universe_sha256": sha256_file(output / "candidate_universe.jsonl"), "candidate_count": len(pool),
        "old_candidate_count": len(old), "train_parent_count": len(parents), "extra_parent_ids": extra_ids,
        "new_graph_oracle_queries": sum(s["new_graph_oracle_queries"] for s in states),
        "actual_query_budget_exceeded": any(s["new_graph_oracle_queries"] > (512 if s["parent_id"] in extra_ids else 128) for s in states),
        "calibration_opened_during_search": False, "test_opened": False,
        "impossibility_proven": False, "conditional_ppo_started": False})


def run_calibration(output: Path, *, device: str = "cpu", boundary_check=lambda: None):
    contract = unseal(output / "search_contract.json")
    frozen = unseal(output / "candidate_freeze.json")
    if sha256_file(output / "candidate_universe.jsonl") != frozen["candidate_universe_sha256"]:
        raise ValueError("CANDIDATE_FREEZE_CONFLICT")
    if (output / "selector_freeze.json").exists():
        return unseal(output / "selector_freeze.json")
    pool = read_jsonl(output / "candidate_universe.jsonl")
    parents = load_bace_parents(contract["paths"]["calibration"], source_label=contract["source_label"])
    work = output / "calibration"
    work.mkdir(parents=True, exist_ok=True)
    oldids = set(contract["old_candidate_ids"])
    oldpairs = read_jsonl(contract["paths"]["old_calibration_pairs"])
    ref = read_json(contract["paths"]["reference"])
    if sha256_file(contract["paths"]["old_calibration_pairs"]) != ref["frozen_downstream"]["selector_contract"]["calibration_input_sha"]:
        raise ValueError("SEALED_OLD_CALIBRATION_MATRIX_CHANGED")
    new = [r for r in pool if r["candidate_id"] not in oldids]
    oracle, features, distance = load_runtime(contract, work, device)
    pairs = list(oldpairs)
    for i, parent in enumerate(parents):
        boundary_check()
        path = work / (stable_sha256(parent.parent_id)[:24] + ".json")
        if path.exists():
            saved = unseal(path)
            if saved["pool_sha256"] != frozen["candidate_universe_sha256"]:
                raise ValueError("CALIBRATION_PARENT_POOL_CHANGED")
        else:
            rows, matches = evaluate_pairs([parent], new, oracle=oracle, featurizer=features,
                distance_provider=distance, split="calibration", oracle_checkpoint_id=contract["oracle_binding"])
            saved = seal(path, {"pool_sha256": frozen["candidate_universe_sha256"], "pairs": rows, "matches": matches})
        pairs.extend(saved["pairs"])
        atomic_json(output / "progress.json", {"state": "CALIBRATION_RUNNING", "completed_parents": i + 1,
                    "total_parents": len(parents), "updated_at": utc_now(), "science_pid": os.getpid()})
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs, select_calibration
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict, VariantConfig
    matrix = matrix_from_pairs([p.parent_id for p in parents], pool, pairs, root=output, split="calibration")
    thresholds = read_json(contract["paths"]["thresholds"])
    oldselection = read_json(contract["paths"]["old_selector"])
    configs = read_json(Path(contract["paths"]["old_selector"]).parent / "variant_configs.json")
    original = {"variant": VariantConfig(**configs["variants"][oldselection["selected_variant"]]),
        "thresholds": threshold_bundle_from_dict(thresholds), "prefix_weights": tuple(configs["prefix_weights"]),
        "local_swap_passes": int(configs["local_swap_passes"])}
    control, control_metrics = select_calibration(matrix, original)
    ids = [str(r["candidate_id"]) for r in pool]
    masks = ReachMasks.from_distances(ids, matrix.distances, thresholds)
    selected = select_nested(masks, oldselection["ordered_rule_ids"])
    atomic_jsonl(output / "calibration_pairs.jsonl", pairs)
    atomic_json(output / "old_calibration_diagnostic.json", summary(oldpairs, [p.parent_id for p in parents], oldselection["ordered_rule_ids"], thresholds))
    return seal(output / "selector_freeze.json", {"state": "CALIBRATION_SELECTOR_FROZEN", "schema": SCHEMA,
        "candidate_freeze_sha256": frozen["self_sha256"], "candidate_universe_sha256": frozen["candidate_universe_sha256"],
        "calibration_pairs_sha256": sha256_file(output / "calibration_pairs.jsonl"), "reach_first": selected,
        "controls": {"old_pool_old_selector": oldselection["ordered_rule_ids"],
                     "new_pool_old_selector": [ids[i] for i in control], "new_pool_old_selector_metrics": control_metrics},
        "predetermined_test_treatment": "new_pool_reach_first", "test_opened": False,
        "test_stage": "SEALED_WAITING_SEPARATE_PREDECLARED_DESCRIPTIVE_EVALUATION", "pool_generation_reopened": False})
