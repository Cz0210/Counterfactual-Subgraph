from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.eval import close_counterfactual_coverage as ccc


class _FakeMol:
    def __init__(self, atoms: int, bonds: int) -> None:
        self._atoms = atoms
        self._bonds = bonds

    def GetNumAtoms(self) -> int:
        return self._atoms

    def GetNumBonds(self) -> int:
        return self._bonds


class _MockEmbeddingTeacher:
    def get_graph_embedding(self, smiles: str, embedding_layer: str = "penultimate"):
        del embedding_layer
        if smiles == "A":
            return [1.0, 0.0]
        return [0.0, 1.0]


class _FakeTeacher:
    available = True
    availability_reason = "ok"

    def score_smiles(self, smiles: str, label: int, parent_smiles=None, meta=None):
        del parent_smiles, meta
        if smiles == "CC":
            return {
                "teacher_result_ok": True,
                "teacher_prob": 0.9,
                "teacher_label": int(label),
                "teacher_reason": "ok",
            }
        return {
            "teacher_result_ok": True,
            "teacher_prob": 0.2,
            "teacher_label": 0,
            "teacher_reason": "ok",
        }


class CloseCounterfactualCoverageTests(unittest.TestCase):
    @unittest.skipIf(ccc.Chem is None, "RDKit is not available")
    def test_hard_delete_substructure_any_match_deletes_simple_fragment(self) -> None:
        candidates = ccc.hard_delete_substructure_any_match("CCO", "CO")
        self.assertTrue(candidates)
        self.assertTrue(any(candidate["delete_valid"] for candidate in candidates))
        self.assertTrue(any(candidate["residual_smiles"] == "C" for candidate in candidates))

    @unittest.skipIf(ccc.Chem is None, "RDKit is not available")
    def test_hard_delete_substructure_any_match_returns_multiple_matches(self) -> None:
        candidates = ccc.hard_delete_substructure_any_match("CCC", "C")
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(any(candidate["delete_valid"] for candidate in candidates))

    def test_normalized_delete_ged_distance_is_bounded(self) -> None:
        distance = ccc.normalized_delete_ged_distance(
            _FakeMol(atoms=4, bonds=3),
            _FakeMol(atoms=2, bonds=1),
            num_removed_atoms=2,
            num_removed_bonds=2,
        )
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)

    def test_threshold_coverage_is_deduplicated_by_parent(self) -> None:
        details = [
            {
                "parent_id": "p1",
                "label": 1,
                "match": True,
                "delete_valid": True,
                "distance": 0.05,
                "cf_drop": 0.1,
                "cf_flip": False,
            },
            {
                "parent_id": "p1",
                "label": 1,
                "match": True,
                "delete_valid": True,
                "distance": 0.04,
                "cf_drop": 0.2,
                "cf_flip": True,
            },
        ]
        summary = ccc.build_threshold_summary(
            details,
            method="ours_selected_subgraphs",
            distance_type="ged",
            ged_mode="delete",
            thresholds=[0.1],
            total_parents=1,
            total_candidates=2,
            require_flip_only=False,
            min_cf_drop=0.0,
        )
        self.assertEqual(summary[0]["num_close_only_covered"], 1)
        self.assertEqual(summary[0]["num_close_cf_covered"], 1)
        self.assertEqual(summary[0]["close_cf_coverage"], 1.0)

    def test_embedding_distance_uses_one_minus_cosine(self) -> None:
        result = ccc.embedding_distance_from_teacher(_MockEmbeddingTeacher(), "A", "B")
        self.assertTrue(result["embedding_ok"])
        self.assertAlmostEqual(result["cosine_similarity"], 0.0)
        self.assertAlmostEqual(result["embedding_distance"], 1.0)

    def test_gcf_baseline_does_not_call_hard_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.csv"
            candidates = root / "gcf.csv"
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles", "label"])
                writer.writeheader()
                writer.writerow({"smiles": "CC", "label": "1"})
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["candidate_smiles"])
                writer.writeheader()
                writer.writerow({"candidate_smiles": "C"})

            with mock.patch.object(ccc, "TeacherSemanticScorer", return_value=_FakeTeacher()), mock.patch.object(
                ccc,
                "normalized_networkx_ged_distance",
                return_value=0.1,
            ), mock.patch.object(
                ccc,
                "hard_delete_substructure_any_match",
                side_effect=AssertionError("GCF mode must not call hard deletion"),
            ):
                result = ccc.evaluate_gcf_counterfactual_graphs(
                    dataset_csv=dataset,
                    gcf_candidates_path=candidates,
                    teacher_path=root / "teacher.pkl",
                    label=1,
                    distance_type="ged",
                    thresholds=[0.2],
                    output_dir=root / "out",
                    require_flip_only=True,
                )
            self.assertEqual(result["threshold_summary"][0]["close_cf_coverage"], 1.0)
            self.assertTrue((root / "out" / "details.csv").exists())


if __name__ == "__main__":
    unittest.main()
