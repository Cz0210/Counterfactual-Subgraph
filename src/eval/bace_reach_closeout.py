"""Post-search train gate and one predeclared descriptive test, no new search.

This is a successor to immutable search artifacts. It never modifies the
running search driver, chooses a test variant, or fits a classifier/threshold.
"""
from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.eval.bace_frozen_gnn_contracts import (
    atomic_csv, atomic_json, atomic_jsonl, load_bace_parents,
    read_json, read_jsonl, sha256_file, stable_sha256, utc_now,
)
from src.eval.bace_reach_v2 import seal, unseal


def verify_witness_application(parent_smiles, witness, candidate):
    """A stored mask is not enough: the retained rule must reproduce it."""
    from src.chem.bace_reach_search import deletion_outcomes
    wanted = tuple(sorted(witness["match_atom_indices"]))
    matching = [outcome for outcome in deletion_outcomes(parent_smiles, candidate, "train_binding_check")
                if tuple(sorted(outcome.match_atom_indices)) == wanted]
    if (len(matching) != 1 or not matching[0].valid
        or matching[0].residual_smiles != witness["residual_smiles"]
        or not witness.get("strict_flip") or not witness.get("valid")
        or witness.get("before", {}).get("predicted_label") != 1
        or witness.get("after", {}).get("predicted_label") != 0):
        raise ValueError("RETAINED_RULE_OWN_WITNESS_NOT_REPLAYABLE")
    return True


def retained_support_gate(rows, pool, source_label=1):
    """A retained-pool witness is a lower bound, never an impossibility claim."""
    retained = {r["candidate_id"] for r in pool}
    parents, covered, counts = [], [], {}
    for row in rows:
        if int(row["before"]["predicted_label"]) != source_label:
            continue
        pid = row["parent_id"]
        if pid in counts:
            raise ValueError("DUPLICATE_COMMITTED_TRAIN_PARENT")
        parents.append(pid)
        ids = {p["candidate_id"] for p in row["old_pool_pairs"]
               if p["pair_strict_flip"] and p["candidate_id"] in retained}
        ids.update(w["pattern"]["candidate_id"] for w in (row.get("search") or {}).get("witnesses", [])
                   if w["pattern"]["candidate_id"] in retained and w.get("valid")
                   and w.get("strict_flip") and w.get("after", {}).get("predicted_label") != source_label)
        counts[pid] = len(ids)
        if ids:
            covered.append(pid)
    if not parents:
        raise ValueError("NO_SOURCE_ELIGIBLE_TRAIN_PARENTS")
    lower = len(covered) / len(parents)
    return {"source_eligible_count": len(parents), "source_eligible_parent_ids": parents,
            "retained_verified_reach_lower_count": len(covered),
            "retained_verified_reach_lower_bound": lower, "verified_rule_count_by_parent": counts,
            "state": "NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE" if lower >= .9
                     else "NEEDS_RETAINED_POOL_CROSS_PARENT_REACH_NOT_PPO_AUTHORIZED",
            "full_pool_reach_exact": len(covered) == len(parents),
            "uncovered_parent_impossibility_claimed": False, "test_opened": False}


def train_gate(campaign: Path):
    contract = unseal(campaign / "search_contract.json")
    frozen = unseal(campaign / "candidate_freeze.json")
    if frozen["search_contract_sha256"] != contract["self_sha256"] or frozen["actual_query_budget_exceeded"]:
        raise ValueError("SEARCH_COMPLETION_BINDING_INVALID")
    pool = read_jsonl(campaign / "candidate_universe.jsonl")
    if sha256_file(campaign / "candidate_universe.jsonl") != frozen["candidate_universe_sha256"]:
        raise ValueError("FROZEN_RETAINED_POOL_CHANGED")
    parents = load_bace_parents(contract["paths"]["train"], source_label=contract["source_label"])
    by_id = {r["candidate_id"]: r for r in pool}
    application_checks = 0
    rows = []
    extra_ids = set(frozen["extra_parent_ids"])
    for parent in parents:
        key = stable_sha256({"id": parent.parent_id, "smiles": parent.smiles})[:24]
        row = unseal(campaign / "train" / (key + ".json"))
        if row["search_contract_sha256"] != contract["self_sha256"] or row["parent_id"] != parent.parent_id:
            raise ValueError("TRAIN_PARENT_INPUT_BINDING_CHANGED")
        if parent.parent_id in extra_ids:
            extra = unseal(campaign / "train" / (stable_sha256(parent.parent_id)[:24] + "-extra.json"))
            if extra["search_contract_sha256"] != contract["self_sha256"] or extra["initial_binding"] != stable_sha256(row["search"]):
                raise ValueError("TRAIN_EXTRA_PASS_BINDING_CHANGED")
            row = {**row, "search": extra["search"]}
        for witness in (row.get("search") or {}).get("witnesses", []):
            candidate = by_id.get(witness["pattern"]["candidate_id"])
            if candidate is not None:
                verify_witness_application(parent.smiles, witness, candidate)
                application_checks += 1
        rows.append(row)
    result = retained_support_gate(rows, pool, contract["source_label"])
    selected = read_json(contract["paths"]["old_selector"])["ordered_rule_ids"]
    thresholds = read_json(contract["paths"]["thresholds"])
    eligible_rows = [r for r in rows if r["before"]["predicted_label"] == contract["source_label"]]
    funnel = {}
    for name, ids in (("S10", selected[:10]), ("S20", selected[:20]), ("all66", contract["old_candidate_ids"])):
        chosen = set(ids)
        groups = [[p for p in r["old_pool_pairs"] if p["candidate_id"] in chosen] for r in eligible_rows]
        funnel[name] = {"source_eligible_parents": len(eligible_rows),
            "parents_with_match": sum(any(p["num_matches"] for p in g) for g in groups),
            "parents_with_valid_deletion": sum(any(p["num_valid_residuals"] for p in g) for g in groups),
            "parents_with_strict_flip": sum(any(p["pair_strict_flip"] for p in g) for g in groups),
            "theta_star_covered": sum(any(float(p["distance_for_selection"]) <= thresholds["theta_star"] for p in g) for g in groups),
            "high_cap_covered": sum(any(float(p["distance_for_selection"]) <= thresholds["cost_cap"] for p in g) for g in groups),
            "match_count": sum(p["num_matches"] for g in groups for p in g),
            "valid_deletion_count": sum(p["num_valid_residuals"] for g in groups for p in g),
            "strict_flip_match_count": sum(p["num_strict_flip_matches"] for g in groups for p in g)}
    return seal(campaign / "train_reach_gate.json", {**result,
        "train_parent_count": len(rows), "candidate_freeze_sha256": frozen["self_sha256"],
        "old_pool_train_funnel": funnel, "all_matches_enumerated_in_old_diagnostic": True,
        "retained_own_rule_application_checks": application_checks,
        "candidate_universe_sha256": frozen["candidate_universe_sha256"],
        "search_contract_sha256": contract["self_sha256"],
        "new_oracle_calls": 0, "ot_recomputed": 0,
        "canary_query_count_separate": unseal(campaign / "canary" / "canary_receipt.json")["new_graph_queries"],
        "formal_search_new_queries": frozen["new_graph_oracle_queries"],
        "old66_diagnostic_graph_evaluations": frozen["old_pool_diagnostic_classifier_graph_evaluations"],
        "lower_bound_not_substituted_for_upper_bound": True})


def reach_parent(*, parent, candidates: Sequence[Mapping], before: Mapping,
                 predict: Callable, oracle_binding: str, known_rows=(), source_label=1):
    """Existential full-pool reach: no WNode; a witnessed parent may stop early.

    For an unwitnessed source parent every match in the retained finite pool is
    checked. This is not an impossibility certificate for all graph deletions.
    """
    from src.chem.bace_reach_search import deletion_outcomes, new_graph_key
    if int(before["predicted_label"]) != source_label:
        return {"parent_id": parent.parent_id, "source_eligible": False, "reachable": False,
                "new_graph_oracle_queries": 0, "all_retained_matches_exhausted": False}
    cache = {}
    for row in known_rows:
        if row.get("delete_valid") and row.get("residual_smiles") and row.get("p_after") is not None:
            if row.get("oracle_checkpoint_hash") != oracle_binding:
                raise ValueError("REACH_KNOWN_ORACLE_BINDING_CHANGED")
            cache[new_graph_key(row["residual_smiles"], oracle_binding)] = {
                "predicted_label": row["pred_after"], "probabilities": row["p_after"]}
    queries, hits, matches, valid = 0, 0, 0, 0
    for candidate in candidates:
        for outcome in deletion_outcomes(parent.smiles, candidate, parent.parent_id):
            matches += 1
            if not outcome.valid:
                continue
            valid += 1
            key = new_graph_key(outcome.residual_smiles, oracle_binding)
            if key not in cache:
                cache[key] = dict(predict([outcome.residual_smiles])[0])
                queries += 1
            else:
                hits += 1
            if int(cache[key]["predicted_label"]) != source_label:
                return {"parent_id": parent.parent_id, "source_eligible": True, "reachable": True,
                        "candidate_id": candidate["candidate_id"], "witness": outcome.as_dict(),
                        "before": dict(before), "after": cache[key], "oracle_binding": oracle_binding,
                        "new_graph_oracle_queries": queries, "cache_hits": hits,
                        "matches_examined": matches, "valid_residuals_examined": valid,
                        "all_retained_matches_exhausted": False, "existential_witness_sufficient": True}
    return {"parent_id": parent.parent_id, "source_eligible": True, "reachable": False,
            "new_graph_oracle_queries": queries, "cache_hits": hits, "matches_examined": matches,
            "valid_residuals_examined": valid, "all_retained_matches_exhausted": True,
            "full_action_space_impossibility_claimed": False}


def freeze_final(campaign: Path, output: Path, *, old_test_pair_source=None):
    """Seal the one test location and all three controls before opening test."""
    contract = unseal(campaign / "search_contract.json")
    pool = unseal(campaign / "candidate_freeze.json")
    selector = unseal(campaign / "selector_freeze.json")
    gate = unseal(campaign / "train_reach_gate.json")
    if gate["state"] != "NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE":
        raise ValueError("TRAIN_PPO_DECISION_NOT_CLOSED")
    if (selector["candidate_freeze_sha256"] != pool["self_sha256"]
        or gate["candidate_freeze_sha256"] != pool["self_sha256"]
        or selector["test_opened"] is not False):
        raise ValueError("POOL_SELECTOR_TRAIN_GATE_BINDING_CHANGED")
    ref = read_json(contract["paths"]["reference"])
    down = ref["frozen_downstream"]
    controls = {"old_pool_old_selector": selector["controls"]["old_pool_old_selector"],
                "new_pool_old_selector": selector["controls"]["new_pool_old_selector"],
                "new_pool_reach_first": selector["reach_first"]["ordered_rule_ids"]}
    for ids in controls.values():
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError("THREE_CONTROLS_REQUIRE_THEIR_FROZEN_UNIQUE_20")
    if not old_test_pair_source:
        raise ValueError("OLD_COMPLIANT_TEST_OT_SOURCE_BINDING_REQUIRED_NO_BLIND_RECOMPUTATION")
    if set(old_test_pair_source) != {"path", "sha256", "receipt_path", "receipt_sha256"}:
        raise ValueError("OLD_TEST_PAIR_SOURCE_DESCRIPTOR_INCOMPLETE")
    value = {"state": "REACH_V2_FINAL_CONFIGURATION_FROZEN", "campaign": str(campaign.resolve()),
             "test_output_root": str(output.resolve()), "controls": controls,
             "selected_control": "new_pool_reach_first", "selected_using_test": False,
             "original_test_previously_seen": True, "claim_new_untouched_test": False,
             "test_opened": False, "test_campaigns_max": 1,
             "search_contract_sha256": contract["self_sha256"],
             "candidate_freeze_sha256": pool["self_sha256"],
             "selector_freeze_sha256": selector["self_sha256"],
             "train_gate_sha256": gate["self_sha256"],
             "test_path": down["dataset_split_paths"]["test"],
             "test_sha256": down["dataset_split_hashes"]["test"],
             "old_test_pair_source": old_test_pair_source,
             "thresholds_sha256": sha256_file(contract["paths"]["thresholds"]),
             "main_matrix_write": False, "new_model_training": False}
    # A second output/version with the same campaign is rejected, not chosen
    # after observing test. Resume of this same binding is idempotent.
    return seal(campaign / "final_test_binding.json", value)


def _curve_report(matrix, sequence, thresholds):
    from src.ablations.gnn.cpu_evaluation import explanation_metrics
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    import numpy as np
    result = explanation_metrics(matrix, sequence, threshold_bundle_from_dict(thresholds))
    for k in (10, 20):
        best = np.min(matrix.distances[:, sequence[:k]], axis=1)
        result[f"K{k}_finite_reach_count"] = int(np.isfinite(best).sum())
        result[f"K{k}_high_cap_count"] = int((best <= thresholds["cost_cap"]).sum())
    return result


def run_final_test(campaign: Path, output: Path, *, boundary_check=lambda: None):
    """Resume one descriptive evaluation; never select or dispatch a variant."""
    binding = unseal(campaign / "final_test_binding.json")
    if str(output.resolve()) != binding["test_output_root"]:
        raise ValueError("SECOND_FINAL_TEST_ROOT_FORBIDDEN")
    contract = unseal(campaign / "search_contract.json")
    selector = unseal(campaign / "selector_freeze.json")
    gate = unseal(campaign / "train_reach_gate.json")
    if (contract["self_sha256"] != binding["search_contract_sha256"]
        or selector["self_sha256"] != binding["selector_freeze_sha256"]
        or gate["self_sha256"] != binding["train_gate_sha256"]):
        raise ValueError("FINAL_CONFIGURATION_CHANGED_AFTER_FREEZE")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "final_audit.json").exists():
            return unseal(output / "final_audit.json")
        return _run_final_test_locked(campaign, output, binding, contract, boundary_check)


def _run_final_test_locked(campaign, output, binding, contract, boundary_check):
    from src.eval.bace_reach_v2 import evaluate_pairs, load_runtime, predict_smiles
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    source = binding["old_test_pair_source"]
    # Only this post-freeze stage may inspect old test records. Reuse immutable
    # receipt once; no hashing all model packages and no OT recomputation.
    for path_key, sha_key in (("path", "sha256"), ("receipt_path", "receipt_sha256")):
        if sha256_file(source[path_key]) != source[sha_key]:
            raise ValueError("OLD_COMPLIANT_TEST_SOURCE_CHANGED")
    receipt = read_json(source["receipt_path"])
    if (receipt.get("schema_version") != "bace_frozen_gnn_test_evaluation_manifest_v1"
        or receipt.get("status") != "PASS" or receipt.get("stage") != "B13_FINAL_EVAL"
        or receipt.get("oracle_checkpoint_hash") != contract["oracle_weight_sha256"]
        or receipt.get("molclr_checkpoint_hash") != contract["molclr_sha256"]
        or receipt.get("pair_matrix_identity", {}).get("path") != source["path"]
        or receipt["pair_matrix_identity"]["sha256"] != source["sha256"]
        or receipt.get("selector_refit_on_test") is not False
        or receipt.get("selection_frozen_before_test") is not True):
        raise ValueError("OLD_COMPLIANT_TEST_RECEIPT_NOT_SAME_FROZEN_SCIENCE")
    old_ids = set(contract["old_candidate_ids"])
    reusable_ids = set(receipt["ordered_rule_ids"])
    chosen = set().union(*map(set, binding["controls"].values()))
    missing_old_cache = chosen.intersection(old_ids).difference(reusable_ids)
    if missing_old_cache:
        raise ValueError("OLD_UNSELECTED_CANDIDATE_RAW_OT_REUSE_BINDING_REQUIRED:" + ",".join(sorted(missing_old_cache)))
    if sha256_file(binding["test_path"]) != binding["test_sha256"]:
        raise ValueError("FROZEN_TEST_SPLIT_CHANGED")
    pool = read_jsonl(campaign / "candidate_universe.jsonl")
    frozen = unseal(campaign / "candidate_freeze.json")
    if frozen["self_sha256"] != binding["candidate_freeze_sha256"] or sha256_file(campaign / "candidate_universe.jsonl") != frozen["candidate_universe_sha256"]:
        raise ValueError("FINAL_POOL_CHANGED")
    old = read_jsonl(source["path"])
    for r in old:
        if r["candidate_id"] not in reusable_ids or r["oracle_checkpoint_hash"] != contract["oracle_weight_sha256"]:
            raise ValueError("OLD_PAIR_ORACLE_OR_CANDIDATE_BINDING_MISMATCH")
    parents = load_bace_parents(binding["test_path"], source_label=contract["source_label"])
    old_by_parent = {p.parent_id: [] for p in parents}
    for row in old:
        if row["parent_id"] not in old_by_parent:
            raise ValueError("OLD_TEST_PAIR_COHORT_MISMATCH")
        old_by_parent[row["parent_id"]].append(row)
    matrix_from_pairs([p.parent_id for p in parents], [r for r in pool if r["candidate_id"] in reusable_ids], old, root=output, split="test")
    selected = [r for r in pool if r["candidate_id"] in chosen]
    new_selected = [r for r in selected if r["candidate_id"] not in old_ids]
    boundary_check()
    oracle, features, distance = load_runtime(contract, output, "cpu")
    pairs, reach = [], []
    for i, parent in enumerate(parents):
        boundary_check()
        path = output / "parents" / (stable_sha256(parent.parent_id)[:24] + ".json")
        if path.exists():
            saved = unseal(path)
            if saved["final_binding_sha256"] != binding["self_sha256"]:
                raise ValueError("TEST_PARENT_CHECKPOINT_BINDING_CHANGED")
        else:
            previous = old_by_parent[parent.parent_id]
            old_before = {(r["pred_before"], r["p1_before"]) for r in previous}
            if len(old_before) != 1:
                raise ValueError("OLD_TEST_PARENT_PREDICTION_RECORDS_DISAGREE")
            pred, p1 = next(iter(old_before))
            before = {"predicted_label": pred, "probabilities": [1.0-p1, p1]}
            rows, matches = evaluate_pairs([parent], new_selected, oracle=oracle, featurizer=features,
                distance_provider=distance, split="test", oracle_checkpoint_id=contract["oracle_binding"],
                parent_prediction_cache={parent.parent_id: {"parent_smiles": parent.smiles,
                    "p_before": before["probabilities"], "pred_before": before["predicted_label"]}})
            full = reach_parent(parent=parent, candidates=pool, before=before,
                predict=lambda ss: predict_smiles(oracle, features, ss, "test_fullpool_reach"),
                oracle_binding=contract["oracle_binding"], known_rows=matches)
            saved = seal(path, {"final_binding_sha256": binding["self_sha256"], "parent_id": parent.parent_id,
                "new_selected_pairs": rows, "new_selected_match_witnesses": matches, "full_pool_reach": full,
                "old_pair_source_reused": source, "test_used_for_selection": False})
        pairs.extend(r for r in old_by_parent[parent.parent_id] if r["candidate_id"] in chosen)
        pairs.extend(saved["new_selected_pairs"])
        reach.append(saved["full_pool_reach"])
        atomic_json(output / "progress.json", {"state": "DESCRIPTIVE_TEST_RUNNING", "science_pid": os.getpid(),
            "completed_parents": i+1, "total_parents": len(parents), "updated_at": utc_now()})
    matrix = matrix_from_pairs([p.parent_id for p in parents], selected, pairs, root=output, split="test")
    thresholds = read_json(contract["paths"]["thresholds"])
    if sha256_file(contract["paths"]["thresholds"]) != binding["thresholds_sha256"]:
        raise ValueError("FINAL_THRESHOLDS_CHANGED")
    reports = {}
    for control, ids in binding["controls"].items():
        report = _curve_report(matrix, [matrix.candidate_index[c] for c in ids], thresholds)
        reports[control] = report
        atomic_csv(output / control / "prefix_metrics.csv", report["prefix_rows"])
        atomic_csv(output / control / "parent_prefix_distances.csv", report["parent_rows"])
        atomic_csv(output / control / "table2_k10.csv", [report["prefix_rows"][9]])
        atomic_json(output / control / "explanation_metrics.json", report)
    atomic_jsonl(output / "selected_pairs.jsonl", pairs)
    atomic_jsonl(output / "full_pool_reach.jsonl", reach)
    return seal(output / "final_audit.json", {"state": "EXECUTION_VALID", "final_binding_sha256": binding["self_sha256"],
        "test_campaigns": 1, "test_parent_count": len(parents), "source_eligible_count": sum(r["source_eligible"] for r in reach),
        "full_pool_reach_count": sum(r["reachable"] for r in reach), "selected_control": binding["selected_control"],
        "three_controls_evaluated": list(reports), "test_selected_variant": False, "new_untouched_test_claimed": False,
        "old_compliant_ot_recomputed": 0, "source_receipt": source, "main_matrix_write": False,
        "reports": {k: str(output / k / "explanation_metrics.json") for k in reports}})
