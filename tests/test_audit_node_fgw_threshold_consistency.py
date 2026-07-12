from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_node_fgw_threshold_consistency as audit


class NodeFGWThresholdAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "eval"
        self.output = self.root / "audits" / "node_fgw_threshold_consistency"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_run(
        self,
        name: str,
        *,
        methods: tuple[str, ...] = ("ours",),
        thresholds: tuple[float, ...] = (0.1, 0.2),
        quantiles: tuple[float | None, ...] = (0.1, 0.2),
        threshold_source: str = "auto_quantile",
        fgw_lambda: float = 0.5,
        dataset_csv: str = "/project/parents.csv",
        num_parents: int = 100,
        num_candidates: int = 20,
        write_summary: bool = True,
    ) -> Path:
        run_dir = self.root / name
        combined = run_dir / "combined"
        combined.mkdir(parents=True, exist_ok=True)
        config = {
            "distance_type": "node_fgw",
            "distance_line": "MolCLR-Node-FGW",
            "fgw_lambda": fgw_lambda,
            "structure_mode": "shortest_path_unweighted",
            "feature_cost": "cosine",
            "atom_penalty": 0.0,
            "cf_mode": "strict_flip",
            "dataset_csv": dataset_csv,
            "teacher_path": "/project/aids_rf_model.pkl",
            "skip_redundancy": True,
            "threshold_source": threshold_source,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "selection_method": "test_selector",
        }
        (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "cache_stats.json").write_text(
            json.dumps(
                {
                    "pair_distance_cache_hit_rate": 0.5,
                    "node_embedding_cache_hit_rate": 0.6,
                    "num_invalid_smiles": 1,
                    "num_nan_distances": 2,
                    "runtime_seconds": 3.0,
                }
            ),
            encoding="utf-8",
        )
        if write_summary:
            fields = [
                "method",
                "distance_type",
                "distance_line",
                "fgw_lambda",
                "structure_mode",
                "feature_cost",
                "atom_penalty",
                "threshold",
                "threshold_source",
                "quantile",
                "num_parents",
                "num_candidates",
                "close_only_coverage",
                "close_cf_coverage",
                "skip_redundancy",
                "candidate_set_preselected",
                "selection_performed_in_eval",
                "selection_method",
            ]
            with (combined / "combined_threshold_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for method in methods:
                    for quantile, threshold in zip(quantiles, thresholds):
                        writer.writerow(
                            {
                                "method": method,
                                "distance_type": "node_fgw",
                                "distance_line": "MolCLR-Node-FGW",
                                "fgw_lambda": fgw_lambda,
                                "structure_mode": "shortest_path_unweighted",
                                "feature_cost": "cosine",
                                "atom_penalty": 0.0,
                                "threshold": threshold,
                                "threshold_source": threshold_source,
                                "quantile": "" if quantile is None else quantile,
                                "num_parents": num_parents,
                                "num_candidates": num_candidates,
                                "close_only_coverage": 0.5,
                                "close_cf_coverage": 0.25,
                                "skip_redundancy": True,
                                "candidate_set_preselected": True,
                                "selection_performed_in_eval": False,
                                "selection_method": "test_selector",
                            }
                        )
        return run_dir

    def _discover(self) -> list[audit.RunRecord]:
        records, _warnings, _count = audit.discover_runs(self.root, output_dir=self.output)
        return records

    def test_same_quantile_grid_and_thresholds_are_directly_comparable(self) -> None:
        self._write_run("ours_full")
        self._write_run("clear_full")
        left, right = self._discover()
        result = audit.compare_runs(left, right, atol=1e-12, rtol=1e-9)
        self.assertTrue(result["same_quantile_grid"])
        self.assertTrue(result["same_absolute_thresholds"])
        self.assertTrue(result["direct_threshold_comparison_ok"])

    def test_same_quantile_grid_with_different_thresholds_is_not_direct(self) -> None:
        self._write_run("ours_full", thresholds=(0.1, 0.2))
        self._write_run("clear_full", thresholds=(0.11, 0.25))
        left, right = self._discover()
        result = audit.compare_runs(left, right, atol=1e-12, rtol=1e-9)
        self.assertTrue(result["same_quantile_grid"])
        self.assertFalse(result["same_absolute_thresholds"])
        self.assertFalse(result["direct_threshold_comparison_ok"])

    def test_lambda_mismatch_is_detected(self) -> None:
        self._write_run("ours_full", fgw_lambda=0.5)
        self._write_run("clear_full", fgw_lambda=0.7)
        result = audit.compare_runs(*self._discover(), atol=1e-12, rtol=1e-9)
        self.assertFalse(result["same_fgw_lambda"])
        self.assertFalse(result["same_fgw_config"])

    def test_dataset_csv_mismatch_is_detected(self) -> None:
        self._write_run("ours_full", dataset_csv="/project/a.csv")
        self._write_run("clear_full", dataset_csv="/project/b.csv")
        result = audit.compare_runs(*self._discover(), atol=1e-12, rtol=1e-9)
        self.assertFalse(result["same_dataset_csv"])
        self.assertFalse(result["same_eval_protocol"])

    def test_num_parents_mismatch_is_detected(self) -> None:
        self._write_run("ours_full", num_parents=100)
        self._write_run("clear_full", num_parents=99)
        result = audit.compare_runs(*self._discover(), atol=1e-12, rtol=1e-9)
        self.assertFalse(result["same_num_parents"])
        self.assertFalse(result["direct_threshold_comparison_ok"])

    def test_smoke_and_full_paths_are_classified_and_full_filter_works(self) -> None:
        self._write_run("ours_smoke")
        self._write_run("ours_full")
        records = self._discover()
        self.assertEqual(sum(record.is_smoke_path for record in records), 1)
        self.assertEqual(sum(record.is_full_path for record in records), 1)
        full_records, _warnings, _count = audit.discover_runs(
            self.root, output_dir=self.output, full_only=True
        )
        self.assertEqual(len(full_records), 1)
        self.assertTrue(full_records[0].is_full_path)

    def test_one_directory_with_multiple_methods_creates_multiple_run_ids(self) -> None:
        self._write_run("combined_full", methods=("ours", "clear"))
        records = self._discover()
        self.assertEqual(len(records), 2)
        self.assertEqual({record.method for record in records}, {"ours", "clear"})
        self.assertEqual(len({record.run_id for record in records}), 2)

    def test_missing_summary_is_warned_without_aborting_scan(self) -> None:
        self._write_run("missing_full", write_summary=False)
        self._write_run("valid_full")
        records, warnings, discovered = audit.discover_runs(self.root, output_dir=self.output)
        self.assertEqual(discovered, 2)
        self.assertEqual(len(records), 1)
        self.assertTrue(any("missing_threshold_summary:" in warning for warning in warnings))

    def test_fixed_threshold_source_without_quantiles(self) -> None:
        self._write_run(
            "ours_full",
            threshold_source="fixed",
            quantiles=(None, None),
            thresholds=(0.05, 0.1),
        )
        self._write_run(
            "clear_full",
            threshold_source="fixed",
            quantiles=(None, None),
            thresholds=(0.05, 0.1),
        )
        left, right = self._discover()
        result = audit.compare_runs(left, right, atol=1e-12, rtol=1e-9)
        self.assertEqual(left.threshold_source, "fixed")
        self.assertEqual(left.quantiles, [])
        self.assertTrue(result["same_absolute_thresholds"])
        self.assertTrue(result["direct_threshold_comparison_ok"])

    def test_auto_quantile_difference_emits_leakage_warning(self) -> None:
        self._write_run("ours_full", thresholds=(0.1, 0.2))
        self._write_run("clear_full", thresholds=(0.12, 0.24))
        result = audit.run_audit(eval_root=self.root, output_dir=self.output)
        self.assertIn(audit.AUTO_QUANTILE_WARNING, result["summary"]["warnings"])
        report = (self.output / "node_fgw_threshold_audit_report.txt").read_text(encoding="utf-8")
        self.assertIn(audit.AUTO_QUANTILE_WARNING, report)

    def test_reference_output_contains_all_other_runs(self) -> None:
        self._write_run("ours_full")
        self._write_run("clear_full")
        records = self._discover()
        reference = records[0].run_id
        result = audit.run_audit(
            eval_root=self.root,
            output_dir=self.output,
            reference_run_id=reference,
        )
        self.assertEqual(result["summary"]["num_reference_comparisons"], 1)
        self.assertTrue((self.output / "node_fgw_threshold_vs_reference.csv").is_file())


if __name__ == "__main__":
    unittest.main()
