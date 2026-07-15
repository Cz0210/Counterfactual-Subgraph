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

    def test_legacy_weak_flip_is_recomputed_as_teacher_strict(self) -> None:
        candidates = [_candidate(1)]
        rows = [
            {
                "method": "globalgce_frequency_top20",
                "parent_id": "p1",
                "candidate_id": "1",
                "label": "1",
                "pred_before": "0",
                "pred_after": "0",
                "cf_flip": "true",
                "distance": "0.01",
            }
        ]
        _parents, recourse, audit, _methods = report.aggregate_detail_rows(
            rows, candidates=candidates
        )
        self.assertEqual(recourse, {})
        self.assertEqual(audit["strict_flip_mismatch_rows"], 1)


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
        self.assertAlmostEqual(metrics["applicable_parent_median_cost"], 0.15)
        self.assertAlmostEqual(metrics["applicable_rate"], 0.5)

    def test_covered_parent_median_is_conditioned_by_threshold(self) -> None:
        metrics = report.compute_prefix_metrics(self.run, k=2, threshold=0.12)
        self.assertAlmostEqual(metrics["covered_parent_median_cost"], 0.1)
        self.assertLessEqual(metrics["covered_parent_median_cost"], 0.12)
        self.assertEqual(
            metrics["theta_covered_conditional_median_cost"],
            metrics["covered_parent_median_cost"],
        )

    def test_table_uses_exact_theta_even_when_plot_grid_does_not_contain_it(self) -> None:
        theta = 0.12
        grid = report.build_threshold_grid("0,0.1,0.2", minimum=0.0, maximum=1.0, points=2)
        self.assertNotIn(theta, grid.tolist())
        plotting_grid = report.include_exact_threshold(grid, theta)
        self.assertIn(theta, plotting_grid.tolist())
        table = report._table_rows([self.run], k=2, theta=theta)[0]
        direct = report.compute_prefix_metrics(self.run, k=2, threshold=theta)
        self.assertEqual(table["Coverage"], direct["coverage"])
        self.assertEqual(
            table["Theta-covered conditional median cost"],
            direct["covered_parent_median_cost"],
        )


class Figure3CostSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_row = {
            "method": "GlobalGCE",
            "k": 10,
            "theta": 0.0328,
            "coverage": 0.042868277474668745,
            "num_covered": 55,
            "median_cost": 0.04677763864948642,
            "applicable_parent_median_cost": 0.04504489141648487,
            "theta_covered_conditional_median_cost": 0.030951724750759534,
        }

    def test_default_figure3_cost_is_theta_covered_conditional_median(self) -> None:
        row = report.add_figure3_plotted_cost([self.source_row])[0]
        self.assertEqual(
            row["plotted_cost_metric"],
            "theta_covered_conditional_median_cost",
        )
        self.assertAlmostEqual(row["plotted_cost"], 0.030951724750759534)

    def test_explicit_median_cost_is_used_only_when_requested(self) -> None:
        row = report.add_figure3_plotted_cost(
            [self.source_row],
            cost_metric="median_cost",
        )[0]
        self.assertEqual(row["plotted_cost_metric"], "median_cost")
        self.assertAlmostEqual(row["plotted_cost"], 0.04677763864948642)

    def test_no_covered_parent_produces_nan_without_fallback(self) -> None:
        source = {**self.source_row, "num_covered": 0}
        row = report.add_figure3_plotted_cost(
            [source],
            cost_metric="median_cost",
        )[0]
        self.assertTrue(math.isnan(row["plotted_cost"]))

    def test_default_finite_cost_must_not_exceed_theta(self) -> None:
        source = {
            **self.source_row,
            "theta_covered_conditional_median_cost": 0.04,
        }
        with self.assertRaisesRegex(AssertionError, "exceeds theta"):
            report.add_figure3_plotted_cost([source])

    def test_figure3_writes_explicit_k10_and_k20_versions(self) -> None:
        rows = report.add_figure3_plotted_cost(
            [
                {
                    **self.source_row,
                    "method": "Ours",
                    "k": k,
                    "num_covered": 1,
                    "coverage": k / 20.0,
                    "theta_covered_conditional_median_cost": 0.02,
                }
                for k in range(1, 21)
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report.write_figure3_plots(
                rows,
                output_dir=output,
                bases=["figure3_fgw_coverage_cost_vs_k"],
                distance_label="MolCLR-Node-FGW",
                theta=0.0328,
                cost_metric="theta_covered_conditional_median_cost",
                inset_max_k=10,
            )
            for suffix in ("1_10", "1_20"):
                self.assertTrue(
                    (output / f"figure3_fgw_coverage_cost_vs_k_{suffix}.png").is_file()
                )
                self.assertTrue(
                    (output / f"figure3_fgw_coverage_cost_vs_k_{suffix}.pdf").is_file()
                )


class Table2SelectionTests(unittest.TestCase):
    def test_default_table_contains_only_final_three_columns(self) -> None:
        run = _run({("0", "1"): (0.01, 0.4)}, num_candidates=10)
        rows = report._table_rows([run], k=10, theta=0.0328)
        fields = report._table_fields(
            cost_metric="theta_covered_conditional_median_cost",
            include_applicable_rate=False,
            include_median_cost=False,
        )
        self.assertEqual(
            fields,
            [
                "Method",
                "Coverage",
                "Theta-covered conditional median cost",
            ],
        )
        self.assertEqual(set(rows[0]), set(fields))
        self.assertNotIn("Median cost", rows[0])
        self.assertNotIn("Applicable rate", rows[0])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report._plot_table2(
                rows,
                fields=fields,
                png=output / "table2_global_recourse.png",
                pdf=output / "table2_global_recourse.pdf",
            )
            self.assertTrue((output / "table2_global_recourse.png").is_file())
            self.assertTrue((output / "table2_global_recourse.pdf").is_file())

    def test_globalgce_k10_regression_values(self) -> None:
        expected_cost = 0.030951724750759534
        recourse = {
            (str(parent_index), "1"): (expected_cost, 0.2)
            for parent_index in range(55)
        }
        run = _run(recourse, num_parents=1283, num_candidates=10)
        run.display_name = "GlobalGCE"
        row = report._table_rows([run], k=10, theta=0.0328)[0]
        self.assertAlmostEqual(row["Coverage"], 0.042868277474668745)
        self.assertAlmostEqual(
            row["Theta-covered conditional median cost"],
            expected_cost,
        )

    def test_default_cli_metrics_are_theta_covered(self) -> None:
        args = report.build_parser().parse_args([])
        self.assertEqual(
            args.figure3_cost_metric,
            "theta_covered_conditional_median_cost",
        )
        self.assertEqual(
            args.table_cost_metric,
            "theta_covered_conditional_median_cost",
        )
        self.assertEqual(args.table_include_applicable_rate, 0)
        self.assertEqual(args.table_include_median_cost, 0)


class ParentCohortTests(unittest.TestCase):
    def test_same_count_but_different_parent_ids_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 1 reference parent IDs"):
            report.validate_reference_parent_set(
                ["p1", "p3"],
                ["p1", "p2"],
                source=Path("synthetic.csv"),
            )

    def test_extra_parent_ids_are_allowed_for_explicit_reference_filter(self) -> None:
        result = report.validate_reference_parent_set(
            ["p1", "p2", "extra"],
            ["p1", "p2"],
            source=Path("synthetic.csv"),
        )
        self.assertEqual(result["extra_ids"], ["extra"])
        self.assertFalse(result["exact_set_match"])


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
