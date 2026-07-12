from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import evaluate_ccrcov_with_molclr_node_fgw as node_fgw_eval
from scripts.eval import compare_hiv_recourse_baselines as comparison
from src.eval import ccrcov_distance_eval as distance_eval
from src.eval.close_counterfactual_coverage import CandidateRecord, ParentRecord
from src.eval.flip_semantics import old_weak_flip, teacher_strict_flip


class TeacherStrictFlipDefinitionTests(unittest.TestCase):
    def test_required_prediction_cases(self) -> None:
        cases = [
            (1, 1, 0, True),
            (1, 0, 0, False),
            (1, 1, 1, False),
            (0, 0, 1, True),
        ]
        for target, pred_before, pred_after, expected in cases:
            with self.subTest(
                target=target,
                pred_before=pred_before,
                pred_after=pred_after,
            ):
                self.assertEqual(
                    teacher_strict_flip(pred_before, pred_after, target),
                    expected,
                )

    def test_legacy_weak_flip_remains_audit_only(self) -> None:
        self.assertTrue(old_weak_flip(0, 1))
        self.assertFalse(teacher_strict_flip(0, 0, 1))
        row = {"pred_before": 0, "pred_after": 0, "cf_drop": 0.0}
        self.assertFalse(
            node_fgw_eval._cf_condition(
                row,
                label=1,
                cf_mode="strict_flip",
                min_cf_drop=0.0,
            )
        )


class NodeFGWStrictSummaryTests(unittest.TestCase):
    def test_main_coverage_is_strict_with_all_parent_denominator(self) -> None:
        details = [
            {
                "parent_id": "teacher-target",
                "candidate_id": "c1",
                "label": 1,
                "pred_before": 1,
                "pred_after": 0,
                "cf_flip": True,
                "teacher_strict_flip": True,
                "old_weak_flip": True,
                "distance": 0.1,
                "cf_drop": 0.5,
            },
            {
                "parent_id": "already-nontarget",
                "candidate_id": "c1",
                "label": 1,
                "pred_before": 0,
                "pred_after": 0,
                "cf_flip": False,
                "teacher_strict_flip": False,
                "old_weak_flip": True,
                "distance": 0.1,
                "cf_drop": 0.0,
            },
        ]
        summary = node_fgw_eval.summarize_method(
            method="baseline",
            details=details,
            threshold_rows=[
                {"threshold_source": "explicit", "quantile": None, "threshold": 0.2}
            ],
            total_parents=2,
            total_candidates=1,
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
            selection_method="external",
            evaluation_row_unit="parent_candidate_pair",
            num_unique_parent_candidate_pairs=2,
            num_detail_rows=2,
            num_valid_match_instances=None,
        )[0]

        self.assertEqual(summary["num_teacher_target_parents"], 1)
        self.assertEqual(summary["num_close_cf_covered"], 1)
        self.assertEqual(summary["close_cf_coverage"], 0.5)
        self.assertEqual(summary["old_weak_num_close_cf_covered"], 2)
        self.assertEqual(summary["old_weak_close_cf_coverage"], 1.0)
        self.assertEqual(summary["flip_rate_among_covered"], 1.0)
        self.assertEqual(summary["main_ccrcov_uses"], "teacher_strict_flip")
        self.assertEqual(summary["old_weak_ccrcov_status"], "audit_only")


class SharedPairGenerationStrictFlipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = ParentRecord("p1", "parent", 1, {})
        self.candidate = CandidateRecord("c1", "candidate", {})
        self.provider = SimpleNamespace(
            distance=lambda _a, _b: {
                "distance": 0.1,
                "cosine_similarity": None,
                "ok": True,
                "error": None,
            }
        )

    @staticmethod
    def _prediction(_teacher: object, smiles: str, _label: int) -> dict[str, object]:
        pred = 0
        return {"pred_label": pred, "p_label": 0.1, "ok": True, "error": None}

    def test_fullgraph_pair_marks_weak_but_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            distance_eval,
            "predict_with_teacher",
            side_effect=self._prediction,
        ):
            rows = distance_eval._evaluate_gt_fullgraph(
                parents=[self.parent],
                candidates=[self.candidate],
                teacher=object(),
                provider=self.provider,
                distance_type="node_fgw",
                distance_name="molclr_node_fgw",
                method="baseline",
                partial_path=Path(tmpdir) / "partial.csv",
                partial_every=0,
            )
        self.assertFalse(rows[0]["cf_flip"])
        self.assertFalse(rows[0]["teacher_strict_flip"])
        self.assertTrue(rows[0]["old_weak_flip"])

    def test_ours_pair_marks_weak_but_not_strict(self) -> None:
        deletion = {
            "match_index": 0,
            "match_atoms": [0],
            "residual_smiles": "residual",
            "delete_valid": True,
            "num_components": 1,
            "num_match_atoms": 1,
            "num_removed_atoms": 1,
            "num_removed_bonds": 0,
            "residual_atom_count": 1,
            "residual_bond_count": 0,
            "atom_delete_ratio": 0.5,
            "bond_delete_ratio": 0.0,
            "error": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            distance_eval,
            "predict_with_teacher",
            side_effect=self._prediction,
        ), mock.patch.object(
            distance_eval,
            "hard_delete_substructure_any_match",
            return_value=[deletion],
        ):
            rows = distance_eval._evaluate_ours(
                parents=[self.parent],
                candidates=[self.candidate],
                teacher=object(),
                provider=self.provider,
                distance_type="node_fgw",
                distance_name="molclr_node_fgw",
                partial_path=Path(tmpdir) / "partial.csv",
                partial_every=0,
            )
        self.assertFalse(rows[0]["cf_flip"])
        self.assertFalse(rows[0]["teacher_strict_flip"])
        self.assertTrue(rows[0]["old_weak_flip"])


class ComparisonStrictFlipTests(unittest.TestCase):
    def test_fullgraph_matrix_rejects_already_nontarget_parent(self) -> None:
        records = [
            SimpleNamespace(
                teacher_ok=True,
                p_target=0.1,
                teacher_label=0,
                canonical_smiles="parent",
            )
        ]
        candidates = [
            SimpleNamespace(
                teacher_ok=True,
                p_target=0.1,
                teacher_label=0,
                canonical_smiles="candidate",
            )
        ]
        proxy = SimpleNamespace(
            distance=lambda _a, _b: {
                "ok": True,
                "cost": 0.1,
                "proxy_edit": 0.1,
            }
        )
        matrix, _ = comparison.build_gt_candidate_matrix(
            records,
            candidates,
            target_label=1,
            distance_proxy=proxy,
            disable_tqdm=True,
        )
        self.assertFalse(matrix[(0, 0)]["cf_flip"])


if __name__ == "__main__":
    unittest.main()
