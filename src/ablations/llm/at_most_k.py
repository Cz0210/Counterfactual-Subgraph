"""LLM-only at-most-20 reporting; no rule padding or GNN protocol changes."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


POLICY = {
    "policy": "AT_MOST_K", "K_MAX": 20, "Table2_K": 10,
    "effective_prefix": "min(requested_K,valid_unique_rule_count,20)",
    "padding_allowed": False, "empty_universe_is_valid_zero": True,
    "applies_to": "ALL_FOUR_BACE_LLM_VARIANTS",
}


def select_calibration(matrix: Any, selector: Mapping[str, Any]):
    """Same frozen main kernels/weights; only cap top_k at actual availability."""
    from src.eval.mutagenicity_wnode_selector import (
        build_candidate_chemistry, build_coverage_redundancy_matrix, _objective_callable,
        greedy_select, optimize_insertion_order, local_swap_search,
    )
    if matrix.manifest.get("split") != "calibration" or matrix.manifest.get("test_loaded") is not False:
        raise ValueError("LLM_SELECTOR_REQUIRES_CALIBRATION_ONLY")
    if not matrix.parent_ids:
        raise ValueError("BLOCKED_EMPTY_CALIBRATION_COHORT")
    if not matrix.candidate_rows:
        return [], {"reason": "EMPTY_TRAIN_UNIVERSE"}
    chemistry = build_candidate_chemistry(matrix.candidate_rows, size_normalization_rows=matrix.full_candidate_rows)
    red = build_coverage_redundancy_matrix(matrix.distances, selector["thresholds"].levels)
    variant = selector["variant"]
    objective = _objective_callable(matrix=matrix, thresholds=selector["thresholds"],
        prefix_weights=selector["prefix_weights"], variant=variant,
        coverage_redundancy_matrix=red, structural_similarity_matrix=chemistry.structural_similarity,
        normalized_sizes=chemistry.normalized_sizes)
    sequence, trace = greedy_select(range(len(matrix.candidate_rows)), top_k=min(20, len(matrix.candidate_rows)),
        objective_fn=objective, candidate_ids=matrix.candidate_ids)
    insertion, swap = [], []
    if variant.insertion_reorder:
        sequence, insertion = optimize_insertion_order(sequence, objective_fn=objective, candidate_ids=matrix.candidate_ids)
    if variant.local_swap:
        sequence, swap = local_swap_search(sequence, all_candidate_indices=range(len(matrix.candidate_rows)),
            objective_fn=objective, candidate_ids=matrix.candidate_ids, max_passes=selector["local_swap_passes"])
    return sequence, {"greedy": trace, "insertion": insertion, "swap": swap, "objective": objective(sequence)}


def explanation_metrics(matrix: Any, sequence: Sequence[int], thresholds: Any) -> dict[str, Any]:
    from src.eval.mutagenicity_wnode_selector import (
        build_candidate_chemistry, build_coverage_redundancy_matrix, compute_prefix_metrics,
    )
    sequence = list(sequence)
    if len(sequence) > 20 or len(sequence) != len(set(sequence)) or any(
            i < 0 or i >= len(matrix.candidate_rows) for i in sequence):
        raise ValueError("LLM_AT_MOST_K_REQUIRES_DISTINCT_EXISTING_RULES")
    effective = len(sequence)
    n = len(matrix.parent_ids)
    if sequence and n:
        chemistry = build_candidate_chemistry(matrix.candidate_rows, size_normalization_rows=matrix.full_candidate_rows)
        red = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
        raw, parent_rows = compute_prefix_metrics(sequence, matrix=matrix, thresholds=thresholds,
            coverage_redundancy_matrix=red, structural_similarity_matrix=chemistry.structural_similarity)
        rows = [{**raw[min(k, effective) - 1], "k": k, "effective_k": min(k, effective),
                 "available_rule_count": effective, "plateau_after_available_rules": k > effective}
                for k in range(1, 21)]
        best = np.min(matrix.distances[:, sequence], axis=1)
    else:
        parent_rows = []
        rows = [{"k": k, "effective_k": 0, "available_rule_count": effective,
                 "ccrcov_theta_star": 0.0 if n else None, "conditional_median_cost": None,
                 "strict_flip_parent_count": 0, "applicable_rate": 0.0 if n else None,
                 "structural_redundancy": None, "plateau_after_available_rules": True}
                for k in range(1, 21)]
        best = np.full(n, np.inf)
    curve = [row["ccrcov_theta_star"] for row in rows]
    auc = float(sum((a + b) / 2 for a, b in zip(curve[:-1], curve[1:]))) if n else None
    last = rows[-1]
    return {"state": "PASS" if n else "VALID_EMPTY_COHORT", "cohort_size": n,
        "K_MAX": 20, "K_EFFECTIVE": effective, "selection_policy": POLICY,
        "CCRCov@10": curve[9], "CCRCov@20": curve[19], "AUC_over_K_1_20": auc,
        "AUC_over_K_1_20_normalized": auc / 19 if auc is not None else None,
        "conditional_median_WNode": last["conditional_median_cost"],
        "strict_flip_rate": last["strict_flip_parent_count"] / n if n else None,
        "applicable_rate": last["applicable_rate"],
        "selected_rule_diversity": 1 - last["structural_redundancy"] if last["structural_redundancy"] is not None else None,
        "covered_parent_ids": [p for p, value in zip(matrix.parent_ids, best) if value <= thresholds.theta_star],
        "prefix_rows": rows, "parent_rows": parent_rows,
        "threshold_rows": [{"threshold": float(x), "CCRCov": float(np.mean(best <= x)) if n else None,
                            "K": 20, "effective_k": effective} for x in sorted(set(thresholds.raw_thresholds))]}
