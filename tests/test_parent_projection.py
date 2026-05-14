import unittest
from unittest.mock import patch

import numpy as np

from src.chem import (
    build_parent_projection_candidates,
    is_parent_substructure,
    is_rdkit_available,
    project_fragment_to_parent_subgraph,
)
from src.rewards.reward_wrapper import ChemRLRewarder

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


class _FakeOracleModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([0.8, 0.2], dtype=np.float32)
        return np.repeat(probabilities[None, :], repeats=features.shape[0], axis=0)


class _FakeCounterfactualTeacherScorer:
    def __init__(self) -> None:
        self.available = True
        self.calls: list[dict[str, object]] = []

    def score_counterfactual(
        self,
        parent_smiles,
        core_fragment_smiles,
        label,
        raw_fragment_smiles=None,
        meta=None,
    ):
        self.calls.append(
            {
                "parent_smiles": parent_smiles,
                "core_fragment_smiles": core_fragment_smiles,
                "label": label,
                "raw_fragment_smiles": raw_fragment_smiles,
                "meta": meta,
            }
        )
        return {
            "teacher_available": True,
            "teacher_result_ok": True,
            "teacher_reason": "ok",
            "parent_without_fragment_smiles": "CCO",
            "p_before": 0.8,
            "p_after": 0.2,
            "pred_before": int(label),
            "pred_after": 0,
            "cf_drop": 0.6,
            "cf_flip": True,
            "counterfactual_sem": 1.6,
        }


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for projection tests")
class ParentProjectionTests(unittest.TestCase):
    PARENT = "CC(=O)Oc1ccccc1C(=O)O"

    def _build_rewarder(
        self,
        *,
        counterfactual_teacher=None,
        enable_projected_cf_reward: bool = False,
        enable_substructure_distance_reward: bool = True,
        substructure_distance_reward_weight: float = 0.5,
        projection_penalty: float = 0.5,
    ) -> ChemRLRewarder:
        oracle_bundle = {
            "model": _FakeOracleModel(),
            "fingerprint_radius": 2,
            "fingerprint_bits": 128,
        }
        with patch(
            "src.rewards.reward_wrapper.load_oracle_bundle",
            return_value=oracle_bundle,
        ):
            return ChemRLRewarder(
                oracle_path="unused.pkl",
                counterfactual_teacher_scorer=counterfactual_teacher,
                enable_parent_projection=True,
                projection_min_score=0.35,
                projection_max_candidates=128,
                projection_min_atoms=3,
                projection_max_atom_ratio=0.70,
                projection_penalty=projection_penalty,
                projection_mcs_timeout=1,
                enable_substructure_distance_reward=enable_substructure_distance_reward,
                substructure_distance_reward_weight=substructure_distance_reward_weight,
                enable_projected_cf_reward=enable_projected_cf_reward,
                disable_projected_cf_reward=not enable_projected_cf_reward,
            )

    def test_candidate_pool_filters_to_strict_parent_subgraphs(self) -> None:
        assert Chem is not None
        parent_mol = Chem.MolFromSmiles(self.PARENT)

        candidates = build_parent_projection_candidates(
            parent_mol,
            parent_smiles=self.PARENT,
            max_candidates=128,
            min_atoms=3,
            max_atom_ratio=0.70,
        )

        self.assertGreater(len(candidates), 0)
        parent_atom_count = parent_mol.GetNumAtoms()
        for candidate in candidates:
            self.assertGreaterEqual(candidate.atom_count, 3)
            self.assertLess(candidate.atom_count, parent_atom_count)
            self.assertLessEqual(candidate.atom_ratio, 0.70)
            self.assertTrue(is_parent_substructure(self.PARENT, candidate.smiles))

    def test_projection_retrieves_parent_subgraph_for_parseable_mismatch(self) -> None:
        raw_fragment = "CC(=O)N"

        result = project_fragment_to_parent_subgraph(
            self.PARENT,
            raw_fragment,
            min_score=0.35,
            max_candidates=128,
            min_atoms=3,
            max_atom_ratio=0.70,
            mcs_timeout=1,
        )

        self.assertTrue(result.attempted)
        self.assertTrue(result.success)
        self.assertEqual(result.projection_method, "retrieval")
        self.assertIsNotNone(result.projected_fragment_smiles)
        self.assertTrue(is_parent_substructure(self.PARENT, result.projected_fragment_smiles or ""))
        self.assertGreaterEqual(result.projection_score or 0.0, 0.35)
        self.assertGreater(result.candidate_count, 0)

    def test_projection_refuses_parse_failed_raw_fragment(self) -> None:
        result = project_fragment_to_parent_subgraph(self.PARENT, "C?C")

        self.assertFalse(result.attempted)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "parse_failed")

    def test_rewarder_uses_distance_reward_without_projected_cf_reward(self) -> None:
        counterfactual_teacher = _FakeCounterfactualTeacherScorer()
        rewarder = self._build_rewarder(counterfactual_teacher=counterfactual_teacher)

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.projection_attempted)
        self.assertTrue(trace.projection_success)
        self.assertEqual(trace.projection_method, "nearest_parent_subgraph")
        self.assertEqual(trace.raw_fragment_smiles, "CC(=O)N")
        self.assertTrue(trace.core_fragment_smiles)
        self.assertFalse(trace.direct_substructure)
        self.assertFalse(trace.counterfactual_teacher_called)
        self.assertEqual(trace.counterfactual_teacher_reason, "not_direct_substructure")
        self.assertEqual(trace.cf_reward_skipped_reason, "not_direct_substructure")
        self.assertFalse(trace.used_projected_subgraph_for_reward)
        self.assertGreater(trace.substructure_similarity, 0.0)
        self.assertGreater(trace.substructure_distance_reward, 0.0)
        self.assertAlmostEqual(trace.substructure_distance_weight, 0.5)
        self.assertGreater(trace.substructure_distance_contribution, 0.0)
        self.assertAlmostEqual(trace.projection_penalty_applied, 0.5)
        self.assertAlmostEqual(trace.projection_penalty, 0.5)
        self.assertAlmostEqual(trace.projection_penalty_config, 0.5)
        self.assertAlmostEqual(
            trace.reward_before_projection_penalty - trace.reward_after_projection_penalty,
            0.5,
        )
        self.assertAlmostEqual(trace.reward_after_projection_penalty, trace.reward)
        self.assertAlmostEqual(
            trace.substructure_distance_contribution,
            trace.breakdown["subdist_contribution"],
        )
        self.assertAlmostEqual(
            trace.substructure_distance_contribution,
            trace.breakdown["subdist_weighted_r"],
        )
        self.assertEqual(len(counterfactual_teacher.calls), 0)
        expected_total = (
            trace.breakdown["format_r"]
            + trace.breakdown["valid_r"]
            + trace.breakdown["length_r"]
            + trace.breakdown["size_window_r"]
            + trace.breakdown["dummy_r"]
            + trace.breakdown["subdist_weighted_r"]
            + trace.breakdown["sem_r"]
        )
        self.assertAlmostEqual(trace.reward, expected_total)

    def test_rewarder_keeps_zero_subdist_contribution_when_disabled(self) -> None:
        rewarder = self._build_rewarder(
            enable_substructure_distance_reward=False,
            substructure_distance_reward_weight=0.3,
        )

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]

        self.assertFalse(trace.direct_substructure)
        self.assertGreater(trace.substructure_distance_reward, 0.0)
        self.assertAlmostEqual(trace.substructure_distance_weight, 0.0)
        self.assertAlmostEqual(trace.substructure_distance_contribution, 0.0)
        self.assertAlmostEqual(trace.projection_penalty_applied, 0.5)
        self.assertAlmostEqual(trace.breakdown["subdist_weight"], 0.0)
        self.assertAlmostEqual(trace.breakdown["subdist_contribution"], 0.0)
        self.assertAlmostEqual(trace.breakdown["subdist_weighted_r"], 0.0)

    def test_parse_failed_fragment_does_not_get_positive_subdist_contribution(self) -> None:
        rewarder = self._build_rewarder(
            enable_substructure_distance_reward=True,
            substructure_distance_reward_weight=0.3,
        )

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["C?C"],
            parent_labels=[1],
        )[0]

        self.assertFalse(trace.raw_parse_ok)
        self.assertIsNotNone(trace.failure_tag)
        self.assertLessEqual(trace.substructure_distance_contribution, 0.0)
        self.assertLessEqual(trace.breakdown["subdist_contribution"], 0.0)
        self.assertAlmostEqual(trace.projection_penalty_applied, 0.0)

    def test_direct_substructure_does_not_apply_projection_penalty(self) -> None:
        rewarder = self._build_rewarder(projection_penalty=1.0)

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)O"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.direct_substructure)
        self.assertAlmostEqual(trace.projection_penalty_applied, 0.0)
        self.assertAlmostEqual(trace.projection_penalty, 0.0)
        self.assertAlmostEqual(trace.reward_before_projection_penalty, trace.reward)
        self.assertAlmostEqual(trace.reward_after_projection_penalty, trace.reward)

    def test_projection_penalty_reduces_not_direct_reward_by_one(self) -> None:
        counterfactual_teacher = _FakeCounterfactualTeacherScorer()
        rewarder_without_penalty = self._build_rewarder(
            counterfactual_teacher=counterfactual_teacher,
            enable_projected_cf_reward=True,
            substructure_distance_reward_weight=0.3,
            projection_penalty=0.0,
        )
        rewarder_with_penalty = self._build_rewarder(
            counterfactual_teacher=_FakeCounterfactualTeacherScorer(),
            enable_projected_cf_reward=True,
            substructure_distance_reward_weight=0.3,
            projection_penalty=1.0,
        )

        trace_without_penalty = rewarder_without_penalty.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]
        trace_with_penalty = rewarder_with_penalty.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]

        self.assertFalse(trace_with_penalty.direct_substructure)
        self.assertTrue(trace_with_penalty.used_projected_subgraph_for_reward)
        self.assertAlmostEqual(trace_without_penalty.projection_penalty_applied, 0.0)
        self.assertAlmostEqual(trace_with_penalty.projection_penalty_applied, 1.0)
        self.assertAlmostEqual(trace_with_penalty.projection_penalty, 1.0)
        self.assertAlmostEqual(trace_with_penalty.projection_penalty_config, 1.0)
        self.assertAlmostEqual(
            trace_with_penalty.reward_before_projection_penalty,
            trace_without_penalty.reward_before_projection_penalty,
        )
        self.assertAlmostEqual(
            trace_with_penalty.reward_after_projection_penalty,
            trace_with_penalty.reward_before_projection_penalty - 1.0,
        )
        self.assertAlmostEqual(
            trace_without_penalty.reward - trace_with_penalty.reward,
            1.0,
        )

    def test_rewarder_uses_projected_fragment_for_counterfactual_reward_when_enabled(self) -> None:
        counterfactual_teacher = _FakeCounterfactualTeacherScorer()
        rewarder = self._build_rewarder(
            counterfactual_teacher=counterfactual_teacher,
            enable_projected_cf_reward=True,
            substructure_distance_reward_weight=0.3,
            projection_penalty=1.0,
        )

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.projection_attempted)
        self.assertTrue(trace.projection_success)
        self.assertEqual(trace.projection_method, "nearest_parent_subgraph")
        self.assertTrue(trace.projected_fragment_smiles)
        self.assertFalse(trace.direct_substructure)
        self.assertTrue(trace.used_projected_subgraph_for_reward)
        self.assertIsNone(trace.cf_reward_skipped_reason)
        self.assertTrue(trace.counterfactual_teacher_called)
        self.assertTrue(trace.oracle_ok)
        self.assertEqual(trace.counterfactual_teacher_reason, "ok")
        self.assertEqual(trace.teacher_input_smiles, trace.projected_fragment_smiles)
        self.assertEqual(trace.parent_without_fragment_smiles, "CCO")
        self.assertEqual(len(counterfactual_teacher.calls), 1)
        self.assertEqual(
            counterfactual_teacher.calls[0]["core_fragment_smiles"],
            trace.projected_fragment_smiles,
        )
        self.assertAlmostEqual(trace.substructure_distance_weight, 0.3)
        self.assertGreater(trace.substructure_distance_contribution, 0.0)
        self.assertAlmostEqual(trace.projection_penalty_applied, 1.0)
        self.assertAlmostEqual(trace.reward_after_projection_penalty, trace.reward)
        self.assertAlmostEqual(trace.cf_drop or 0.0, 0.6)
        self.assertTrue(trace.cf_flip)
        self.assertGreater(trace.breakdown["cf_r"], 0.0)
        self.assertGreater(trace.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
