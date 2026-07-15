from __future__ import annotations

import math
from pathlib import Path

from src.eval.gcf_style_recourse_report import (
    CandidateRank,
    MethodRun,
    PairRecourse,
    aggregate_detail_rows,
    best_recourse_by_parent,
    compute_prefix_metrics,
)


def _candidate(rank: int, candidate_id: str) -> CandidateRank:
    return CandidateRank(rank, candidate_id, f"C{rank}", f"C{rank}", rank - 1)


def test_match_instances_keep_minimum_finite_strict_flip_and_ignore_nonflip() -> None:
    candidates = [_candidate(1, "c1"), _candidate(2, "c2")]
    rows = [
        {"method": "ours", "parent_id": "p1", "candidate_id": "c1", "label": 1, "pred_before": 1, "pred_after": 0, "distance": 0.4, "cf_drop": 0.1},
        {"method": "ours", "parent_id": "p1", "candidate_id": "c1", "label": 1, "pred_before": 1, "pred_after": 0, "distance": 0.2, "cf_drop": 0.3},
        {"method": "ours", "parent_id": "p1", "candidate_id": "c1", "label": 1, "pred_before": 1, "pred_after": 1, "distance": 0.01, "cf_drop": 0.9},
        {"method": "ours", "parent_id": "p1", "candidate_id": "c2", "label": 1, "pred_before": 1, "pred_after": 1, "distance": 0.0},
        {"method": "ours", "parent_id": "p2", "candidate_id": "c1", "label": 1, "pred_before": 1, "pred_after": 1, "distance": ""},
        {"method": "ours", "parent_id": "p2", "candidate_id": "c2", "label": 1, "pred_before": 1, "pred_after": 1, "distance": ""},
    ]
    parents, recourse, audit, _methods = aggregate_detail_rows(rows, candidates=candidates, source=Path("test.csv"))
    assert recourse[("p1", "c1")].distance == 0.2
    assert recourse[("p1", "c1")].cf_drop == 0.3
    assert ("p1", "c2") not in recourse
    assert audit["num_multi_match_parent_candidate_pairs"] == 1
    run = MethodRun(
        display_name="Ours", run_dir=Path("."), method="ours", config={}, summary_rows=[], cache_stats={},
        candidates=candidates, candidate_path=Path("selected.csv"), rank_source="rank", selection_method="external",
        parent_ids=parents, recourse_by_pair=recourse, num_detail_rows=len(rows),
        num_unique_parent_candidate_pairs=audit["num_unique_parent_candidate_pairs"],
        num_valid_parent_candidate_pairs=audit["num_valid_parent_candidate_pairs"],
        num_multi_match_parent_candidate_pairs=audit["num_multi_match_parent_candidate_pairs"],
    )
    distances, _ = best_recourse_by_parent(run, k=1)
    assert distances["p1"] == 0.2
    assert math.isinf(distances["p2"])
    metrics = compute_prefix_metrics(run, k=1, threshold=0.3)
    assert math.isinf(metrics["median_cost"])
    assert metrics["conditional_median_cost"] == 0.2


def test_candidate_prefix_uses_rank_not_distance() -> None:
    candidates = [_candidate(1, "first"), _candidate(2, "second")]
    run = MethodRun(
        display_name="X", run_dir=Path("."), method="x", config={}, summary_rows=[], cache_stats={},
        candidates=candidates, candidate_path=Path("selected.csv"), rank_source="rank", selection_method="external",
        parent_ids=("p",), recourse_by_pair={
            ("p", "first"): PairRecourse("p", "first", 1, 0.8, None),
            ("p", "second"): PairRecourse("p", "second", 2, 0.1, None),
        }, num_detail_rows=2, num_unique_parent_candidate_pairs=2,
        num_valid_parent_candidate_pairs=2, num_multi_match_parent_candidate_pairs=0,
    )
    assert best_recourse_by_parent(run, k=1)[0]["p"] == 0.8
    assert best_recourse_by_parent(run, k=2)[0]["p"] == 0.1
    threshold_coverages = [
        compute_prefix_metrics(run, k=2, threshold=threshold)["coverage"]
        for threshold in (0.05, 0.1, 0.2)
    ]
    assert threshold_coverages == sorted(threshold_coverages)
