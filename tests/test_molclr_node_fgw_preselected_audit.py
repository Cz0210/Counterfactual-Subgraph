from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_ccrcov_with_molclr_node_fgw as evaluator


class MolCLRNodeFGWPreselectedAuditTests(unittest.TestCase):
    def _write_ours_directory(
        self,
        root: Path,
        *,
        count: int = 20,
        duplicate_fragment: bool = False,
        duplicate_candidate_id: bool = False,
        duplicate_rank: bool = False,
        write_summary: bool = True,
    ) -> Path:
        directory = root / "stable300_label1_top20_mmr_cov20"
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["rank", "candidate_id", "fragment", "score"]
        with (directory / "selected_subgraphs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(1, count + 1):
                writer.writerow(
                    {
                        "rank": 1 if duplicate_rank and index == count else index,
                        "candidate_id": "candidate-1"
                        if duplicate_candidate_id and index == count
                        else f"candidate-{index}",
                        "fragment": "C"
                        if duplicate_fragment and index == count
                        else "C" * index,
                        "score": 1.0 / index,
                    }
                )
        if write_summary:
            (directory / "selector_summary.json").write_text(
                json.dumps(
                    {
                        "metadata": {"top_k": 20, "beta_coverage": 20.0},
                        "selected_count": count,
                    }
                ),
                encoding="utf-8",
            )
        return directory

    def test_exactly_twenty_selected_subgraphs_pass_and_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = self._write_ours_directory(Path(tmpdir))
            audit = evaluator.validate_preselected_ours_directory(directory, 20)
        self.assertEqual(audit["num_rows"], 20)
        self.assertEqual(audit["num_unique_fragments"], 20)
        self.assertEqual(audit["selection_method"], "greedy_mmr_cov20")
        self.assertEqual(audit["ordered_fragments"][0], "C")
        self.assertEqual(audit["ordered_fragments"][-1], "C" * 20)
        self.assertTrue(audit["order_preserved"])

    def test_nineteen_or_twenty_one_selected_subgraphs_fail(self) -> None:
        for count in (19, 21):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmpdir:
                directory = self._write_ours_directory(Path(tmpdir), count=count)
                with self.assertRaisesRegex(ValueError, "exactly 20 rows"):
                    evaluator.validate_preselected_ours_directory(directory, 20)

    def test_duplicate_fragment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = self._write_ours_directory(Path(tmpdir), duplicate_fragment=True)
            with self.assertRaisesRegex(ValueError, "20 unique fragments"):
                evaluator.validate_preselected_ours_directory(directory, 20)

    def test_duplicate_candidate_identifier_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = self._write_ours_directory(Path(tmpdir), duplicate_candidate_id=True)
            with self.assertRaisesRegex(ValueError, "duplicate candidate identifiers"):
                evaluator.validate_preselected_ours_directory(directory, 20)

    def test_duplicate_rank_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = self._write_ours_directory(Path(tmpdir), duplicate_rank=True)
            with self.assertRaisesRegex(ValueError, "ranks must be exactly"):
                evaluator.validate_preselected_ours_directory(directory, 20)

    def test_missing_selection_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "missing selected_subgraphs"):
                evaluator.validate_preselected_ours_directory(Path(tmpdir), 20)

    def test_unknown_selector_metadata_uses_external_selector_fallback(self) -> None:
        self.assertEqual(
            evaluator._infer_ours_selection_method(Path("plain_selector_directory"), {}),
            "ours_external_selector",
        )

    def test_match_instances_are_not_candidate_selection(self) -> None:
        details = [
            {
                "parent_id": "p1",
                "candidate_id": "c1",
                "label": 1,
                "match": True,
                "delete_valid": True,
                "distance": 0.1,
                "pred_after": 0,
                "cf_flip": True,
                "cf_drop": 0.4,
            },
            {
                "parent_id": "p1",
                "candidate_id": "c1",
                "label": 1,
                "match": True,
                "delete_valid": True,
                "distance": 0.2,
                "pred_after": 0,
                "cf_flip": True,
                "cf_drop": 0.3,
            },
            {
                "parent_id": "p1",
                "candidate_id": "c2",
                "label": 1,
                "match": False,
                "delete_valid": False,
                "distance": None,
                "pred_after": None,
                "cf_flip": False,
            },
        ]
        row_audit = evaluator.build_evaluation_row_audit(
            details,
            evaluation_row_unit="match_instance",
        )
        self.assertEqual(row_audit["num_unique_parent_candidate_pairs"], 2)
        self.assertEqual(row_audit["num_detail_rows"], 3)
        self.assertEqual(row_audit["num_valid_match_instances"], 2)

        summary = evaluator.summarize_method(
            method="ours_selected_subgraphs",
            details=details,
            threshold_rows=[{"threshold_source": "explicit", "quantile": None, "threshold": 0.3}],
            total_parents=1,
            total_candidates=20,
            fgw_lambda=0.5,
            structure_mode="shortest_path_unweighted",
            feature_cost="cosine",
            atom_penalty=0.0,
            cf_mode="strict_flip",
            min_cf_drop=0.0,
            cache_hit_rate=0.0,
            node_embedding_cache_hit_rate=0.0,
            skip_redundancy=True,
            selection_performed_in_eval=False,
            candidate_set_preselected=True,
            selection_method="greedy_mmr_cov20",
            **row_audit,
        )[0]
        self.assertFalse(summary["selection_performed_in_eval"])
        self.assertTrue(summary["candidate_set_preselected"])
        self.assertEqual(summary["evaluation_row_unit"], "match_instance")
        self.assertEqual(summary["num_unique_parent_candidate_pairs"], 2)
        self.assertEqual(summary["num_detail_rows"], 3)
        self.assertEqual(summary["total_pairs"], 3)

    def test_fullgraph_preselected_validation_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fullgraph_top20.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["rank", "candidate_smiles", "rf_strict_flip", "selection_mode"],
                )
                writer.writeheader()
                for index in range(1, 21):
                    writer.writerow(
                        {
                            "rank": index,
                            "candidate_smiles": f"candidate-{index}",
                            "rf_strict_flip": True,
                            "selection_mode": "parent_frequency",
                        }
                    )
            with mock.patch.object(evaluator, "canonicalize_smiles", side_effect=lambda value: value):
                audit = evaluator.validate_preselected_candidate_csv(path, 20)
        self.assertEqual(audit["input_kind"], "fullgraph")
        self.assertEqual(audit["num_unique_canonical_smiles"], 20)
        self.assertEqual(audit["selection_method"], "parent_frequency")
        self.assertTrue(audit["order_preserved"])

    def test_ours_only_main_writes_preselected_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directory = self._write_ours_directory(root)
            output = root / "eval"
            candidates = [SimpleNamespace(smiles="C" * index) for index in range(1, 21)]
            details = [
                {
                    "parent_id": "p1",
                    "candidate_id": "1",
                    "label": 1,
                    "match": True,
                    "delete_valid": True,
                    "distance": 0.1,
                    "pred_after": 0,
                    "cf_flip": True,
                    "cf_drop": 0.4,
                },
                {
                    "parent_id": "p1",
                    "candidate_id": "1",
                    "label": 1,
                    "match": True,
                    "delete_valid": True,
                    "distance": 0.15,
                    "pred_after": 0,
                    "cf_flip": True,
                    "cf_drop": 0.3,
                },
            ]
            fake_provider = SimpleNamespace(
                cache=SimpleNamespace(stats_dict=lambda: {"pair_distance_cache_hit_rate": 0.0}),
                stats_dict=lambda: {"node_embedding_cache_hit_rate": 0.0},
            )
            argv = [
                "evaluate_ccrcov_with_molclr_node_fgw.py",
                "--dataset-csv",
                str(root / "parents.csv"),
                "--teacher-path",
                str(root / "teacher.pkl"),
                "--molclr-root",
                str(root / "molclr"),
                "--molclr-checkpoint",
                str(root / "model.pth"),
                "--ours-selected-path",
                str(directory),
                "--output-dir",
                str(output),
                "--run-ours",
                "1",
                "--run-fullgraph",
                "0",
                "--preselected-topk",
                "20",
                "--require-preselected-topk",
                "1",
                "--max-candidates",
                "20",
                "--fgw-thresholds",
                "0.2",
                "--cf-mode",
                "strict_flip",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "MolCLRNodeFGWDistanceProvider", return_value=fake_provider
            ), mock.patch.object(
                evaluator,
                "_load_parent_records",
                return_value=(root / "parents.csv", [SimpleNamespace()], "label"),
            ), mock.patch.object(
                evaluator,
                "_load_candidate_records",
                return_value=(directory / "selected_subgraphs.csv", candidates),
            ), mock.patch.object(
                evaluator, "TeacherSemanticScorer", return_value=SimpleNamespace()
            ), mock.patch.object(
                evaluator, "_evaluate_ours", return_value=details
            ):
                self.assertEqual(evaluator.main(), 0)

            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["candidate_set_preselected"])
            self.assertFalse(config["selection_performed_in_eval"])
            self.assertEqual(config["selection_method"], "greedy_mmr_cov20")
            self.assertEqual(config["preselected_topk"], 20)
            self.assertEqual(config["evaluation_row_unit"], "match_instance")
            self.assertEqual(config["num_unique_parent_candidate_pairs"], 1)
            self.assertEqual(config["num_detail_rows"], 2)
            self.assertEqual(config["num_valid_match_instances"], 2)


if __name__ == "__main__":
    unittest.main()
