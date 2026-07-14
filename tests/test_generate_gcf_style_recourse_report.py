from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval import gcf_style_recourse_report as report


def _candidate(rank: int) -> report.CandidateRank:
    return report.CandidateRank(
        rank=rank,
        candidate_id=str(rank),
        smiles=f"C{rank}",
        canonical_identity=f"C{rank}",
        row_index=rank - 1,
    )


def _run(
    recourse: dict[tuple[str, str], tuple[float, float | None]],
    *,
    num_parents: int = 4,
    num_candidates: int = 3,
) -> report.MethodRun:
    records = {
        key: report.PairRecourse(
            parent_id=key[0],
            candidate_id=key[1],
            candidate_rank=int(key[1]),
            distance=value[0],
            cf_drop=value[1],
        )
        for key, value in recourse.items()
    }
    return report.MethodRun(
        display_name="Ours",
        run_dir=Path("run"),
        method="ours_selected_subgraphs",
        config={},
        summary_rows=[],
        cache_stats={},
        candidates=[_candidate(rank) for rank in range(1, num_candidates + 1)],
        candidate_path=Path("selected.csv"),
        rank_source="rank",
        selection_method="test",
        parent_ids=tuple(str(index) for index in range(num_parents)),
        recourse_by_pair=records,
        num_detail_rows=0,
        num_unique_parent_candidate_pairs=0,
        num_valid_parent_candidate_pairs=0,
        num_multi_match_parent_candidate_pairs=0,
    )


class PairAggregationTests(unittest.TestCase):
    def test_multi_match_uses_minimum_strict_flip_distance_and_matching_drop(self) -> None:
        candidates = [_candidate(1)]
        rows = [
            {
                "method": "ours_selected_subgraphs",
                "parent_id": "p1",
                "candidate_id": "1",
                "label": "1",
                "pred_before": "1",
                "pred_after": "0",
                "cf_flip": "true",
                "distance": "0.4",
                "cf_drop": "0.3",
            },
            {
                "method": "ours_selected_subgraphs",
                "parent_id": "p1",
                "candidate_id": "1",
                "label": "1",
                "pred_before": "1",
                "pred_after": "0",
                "cf_flip": "true",
                "distance": "0.1",
                "cf_drop": "0.7",
            },
            {
                "method": "ours_selected_subgraphs",
                "parent_id": "p1",
                "candidate_id": "1",
                "label": "1",
                "pred_before": "1",
                "pred_after": "1",
                "cf_flip": "false",
                "distance": "0.01",
                "cf_drop": "0.9",
            },
        ]
        parents, recourse, audit, methods = report.aggregate_detail_rows(rows, candidates=candidates)
        self.assertEqual(parents, ("p1",))
        self.assertEqual(methods, {"ours_selected_subgraphs"})
        self.assertEqual(recourse[("p1", "1")].distance, 0.1)
        self.assertEqual(recourse[("p1", "1")].cf_drop, 0.7)
        self.assertEqual(audit["num_detail_rows"], 3)
        self.assertEqual(audit["num_unique_parent_candidate_pairs"], 1)
        self.assertEqual(audit["num_valid_parent_candidate_pairs"], 1)
        self.assertEqual(audit["num_multi_match_parent_candidate_pairs"], 1)


class CandidateRankingTests(unittest.TestCase):
    def test_rank_column_controls_order_instead_of_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["rank", "candidate_id", "candidate_smiles", "canonical_smiles"],
                )
                writer.writeheader()
                writer.writerow({"rank": 2, "candidate_id": "b", "candidate_smiles": "CC", "canonical_smiles": "CC"})
                writer.writerow({"rank": 1, "candidate_id": "a", "candidate_smiles": "C", "canonical_smiles": "C"})
            candidates, rank_source = report.load_candidate_ranking(path, ours=False, expected_top_k=2)
        self.assertEqual(rank_source, "rank")
        self.assertEqual([candidate.candidate_id for candidate in candidates], ["a", "b"])


class PrefixMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = _run(
            {
                ("0", "1"): (0.4, 0.1),
                ("0", "2"): (0.1, 0.5),
                ("1", "1"): (0.2, 0.4),
                ("2", "3"): (0.05, 0.8),
            }
        )

    def test_prefix_k_is_respected_and_coverage_is_monotone(self) -> None:
        curve = report.compute_k_curve(self.run, threshold=0.15, max_k=3)
        self.assertEqual([row["num_covered"] for row in curve], [0, 1, 2])
        self.assertTrue(all(left <= right for left, right in zip(
            [row["coverage"] for row in curve],
            [row["coverage"] for row in curve][1:],
        )))

    def test_median_cost_is_monotone_non_increasing(self) -> None:
        curve = report.compute_k_curve(self.run, threshold=1.0, max_k=3)
        medians = [row["median_cost"] for row in curve]
        self.assertTrue(all(right <= left for left, right in zip(medians, medians[1:])))

    def test_unavailable_parent_is_infinite_unconditional_but_not_conditional(self) -> None:
        metrics = report.compute_prefix_metrics(self.run, k=2, threshold=1.0)
        self.assertTrue(math.isinf(metrics["median_cost"]))
        self.assertAlmostEqual(metrics["conditional_median_cost"], 0.15)
        self.assertAlmostEqual(metrics["applicable_rate"], 0.5)


class ThresholdAndBootstrapTests(unittest.TestCase):
    def test_coverage_is_monotone_in_threshold(self) -> None:
        curve = report.bootstrap_coverage_curve(
            [0.1, 0.2, math.inf],
            [0.0, 0.1, 0.15, 0.2],
            bootstrap_samples=20,
            seed=3,
        )
        coverage = [row["coverage"] for row in curve]
        self.assertEqual(coverage, sorted(coverage))

    def test_methods_receive_the_exact_same_threshold_grid(self) -> None:
        grid = report.build_threshold_grid("0,0.01,0.02", minimum=0.0, maximum=1.0, points=2)
        first = report.bootstrap_coverage_curve([0.01, math.inf], grid, bootstrap_samples=10, seed=0)
        second = report.bootstrap_coverage_curve([0.02, 0.03], grid, bootstrap_samples=10, seed=0)
        self.assertEqual([row["threshold"] for row in first], [row["threshold"] for row in second])

    def test_bootstrap_is_reproducible_for_fixed_seed(self) -> None:
        first = report.bootstrap_coverage_curve(
            [0.1, 0.2, math.inf, 0.3],
            [0.1, 0.2, 0.3],
            bootstrap_samples=50,
            seed=7,
        )
        second = report.bootstrap_coverage_curve(
            [0.1, 0.2, math.inf, 0.3],
            [0.1, 0.2, 0.3],
            bootstrap_samples=50,
            seed=7,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
