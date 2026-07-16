from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from src.eval.gcf_style_recourse_report import (
    CandidateRank,
    MethodRun,
    PairRecourse,
    _compact_table2_rows,
    _theta_covered_table2_rows,
    add_figure3_plotted_cost,
    aggregate_detail_rows,
    best_recourse_by_parent,
    compute_prefix_metrics,
    figure3_cost_ylim,
    report_artifact_bases,
    resolve_cost_metric,
    resolve_report_thetas,
    validate_figure3_rows,
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


def _complete_figure3_rows(*, theta: float = 0.2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")):
        for k in range(1, 21):
            conditional = 0.6 - 0.01 * k + 0.05 * method_index
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "theta": theta,
                    "coverage": k / 20.0,
                    "num_covered": k,
                    "num_applicable_parents": 20,
                    "num_strict_recourse_applicable_parents": 20,
                    "num_theta_covered_parents": k,
                    "median_cost": conditional + 0.2,
                    "conditional_median_cost": conditional,
                    "applicable_parent_median_cost": conditional,
                    "theta_covered_conditional_median_cost": min(theta, conditional),
                }
            )
    return rows


def test_conditional_median_maps_only_to_conditional_median_cost() -> None:
    assert resolve_cost_metric("median_cost", "conditional_median") == "conditional_median_cost"
    rows = add_figure3_plotted_cost(
        _complete_figure3_rows(), cost_metric="conditional_median_cost"
    )
    source = np.asarray([row["conditional_median_cost"] for row in rows], dtype=float)
    plotted = np.asarray([row["plotted_cost"] for row in rows], dtype=float)
    assert np.allclose(plotted, source, equal_nan=True)


def test_globalgce_first_six_conditional_medians_remain_finite() -> None:
    rows = add_figure3_plotted_cost(
        _complete_figure3_rows(), cost_metric="conditional_median_cost"
    )
    first_six = [
        row["plotted_cost"]
        for row in rows
        if row["method"] == "GlobalGCE" and int(row["k"]) <= 6
    ]
    assert len(first_six) == 6
    assert all(math.isfinite(float(value)) for value in first_six)


def test_theta_covered_cost_remains_nan_when_no_parent_is_covered() -> None:
    source = _complete_figure3_rows()[0]
    source["num_covered"] = 0
    source["num_theta_covered_parents"] = 0
    source["theta_covered_conditional_median_cost"] = float("nan")
    row = add_figure3_plotted_cost(
        [source], cost_metric="theta_covered_conditional_median_cost"
    )[0]
    assert math.isnan(float(row["plotted_cost"]))


def test_figure3_complete_prefix_and_full_ylim_cover_every_finite_cost() -> None:
    rows = add_figure3_plotted_cost(
        _complete_figure3_rows(), cost_metric="conditional_median_cost"
    )
    audit = validate_figure3_rows(
        rows,
        max_k=20,
        figure3_theta=0.2,
        cost_metric="conditional_median_cost",
    )
    assert audit["actual_rows"] == 80
    for method in ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer"):
        assert [row["k"] for row in rows if row["method"] == method] == list(range(1, 21))
    y_min, y_max = figure3_cost_ylim(rows, max_k=20, mode="full")
    finite = [float(row["plotted_cost"]) for row in rows]
    assert y_min <= min(finite)
    assert y_max >= max(finite)


def test_table2_and_figure3_thetas_are_independent_with_legacy_fallback() -> None:
    explicit = argparse.Namespace(theta_star=0.1, table2_theta=0.2, figure3_theta=0.3)
    assert resolve_report_thetas(explicit) == (0.2, 0.3)
    legacy = argparse.Namespace(theta_star=0.1, table2_theta=None, figure3_theta=None)
    assert resolve_report_thetas(legacy) == (0.1, 0.1)


def test_wnode_primary_names_have_no_fgw_and_default_has_no_legacy_aliases() -> None:
    artifacts = report_artifact_bases(
        prefix="final",
        distance_label="MolCLR-Node-Wasserstein",
        write_legacy_aliases=False,
    )
    assert artifacts["primary"]["figure3"] == "final_figure3_wnode_coverage_cost_vs_k"
    assert artifacts["primary"]["figure4"] == "final_figure4_wnode_coverage_vs_threshold"
    assert all("fgw" not in value for value in artifacts["primary"].values())
    assert all(not aliases for aliases in artifacts["legacy"].values())


def test_compact_and_theta_covered_tables_use_distinct_cost_columns() -> None:
    full_rows = [
        {
            "Method": "Ours",
            "Coverage": 0.4,
            "Num Covered": 40,
            "Conditional Median Cost": 0.3,
            "Theta-covered Conditional Median Cost": 0.1,
        }
    ]
    compact = _compact_table2_rows(full_rows)
    theta_audit = _theta_covered_table2_rows(full_rows)
    assert compact == [
        {"Method": "Ours", "Coverage": 0.4, "Conditional Median Cost": 0.3}
    ]
    assert theta_audit == [
        {
            "Method": "Ours",
            "Coverage": 0.4,
            "Num Covered": 40,
            "Theta-covered Conditional Median Cost": 0.1,
        }
    ]
