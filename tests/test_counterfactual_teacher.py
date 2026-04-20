import unittest
from unittest.mock import patch

from src.chem import is_rdkit_available
from src.rewards.counterfactual_oracle import (
    CounterfactualTeacherScorer,
    delete_one_substructure,
)
from src.rewards.reward_wrapper import ChemRLRewarder


class _FakeOracleModel:
    classes_ = [0, 1]

    def predict_proba(self, features):
        raise AssertionError("Reward wrapper oracle bundle should not be used in this test.")


class _FakeFragmentTeacherScorer:
    def __init__(self, probability: float = 0.8) -> None:
        self.available = True
        self.availability_reason = "ok"
        self.teacher_format = "fake_fragment_teacher"
        self.probability = float(probability)
        self.calls: list[dict[str, object]] = []

    def score_smiles(self, smiles, label=None, parent_smiles=None, meta=None):
        self.calls.append(
            {
                "smiles": smiles,
                "label": label,
                "parent_smiles": parent_smiles,
                "meta": meta,
            }
        )
        return {
            "teacher_available": True,
            "teacher_input_smiles": smiles,
            "teacher_parse_ok": True,
            "teacher_prob": self.probability,
            "teacher_label": int(label) if label is not None else 0,
            "teacher_sem": 2.0 * self.probability - 1.0,
            "teacher_reason": "ok",
            "teacher_result_ok": True,
        }


class _FakeProbabilityTeacherScorer:
    def __init__(self, probabilities_by_smiles, predicted_labels=None) -> None:
        self.available = True
        self.availability_reason = "ok"
        self.teacher_format = "fake_predict_proba"
        self._probabilities_by_smiles = dict(probabilities_by_smiles)
        self._predicted_labels = dict(predicted_labels or {})
        self.calls: list[dict[str, object]] = []

    def score_smiles(self, smiles, label=None, parent_smiles=None, meta=None):
        self.calls.append(
            {
                "smiles": smiles,
                "label": label,
                "parent_smiles": parent_smiles,
                "meta": meta,
            }
        )
        probability = float(self._probabilities_by_smiles[smiles])
        predicted_label = int(self._predicted_labels.get(smiles, label if label is not None else 0))
        return {
            "teacher_available": True,
            "teacher_input_smiles": smiles,
            "teacher_parse_ok": True,
            "teacher_prob": probability,
            "teacher_label": predicted_label,
            "teacher_sem": 2.0 * probability - 1.0,
            "teacher_reason": "ok",
            "teacher_result_ok": True,
        }


class _FakeCounterfactualTeacherScorer:
    def __init__(self, counterfactual_sem: float = 0.5) -> None:
        self.available = True
        self.availability_reason = "ok"
        self.counterfactual_sem = float(counterfactual_sem)
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
            "teacher_input_parent_smiles": parent_smiles,
            "teacher_input_fragment_smiles": core_fragment_smiles,
            "parent_parse_ok": True,
            "fragment_parse_ok": True,
            "substructure_ok": True,
            "deletion_ok": True,
            "parent_without_fragment_smiles": "c1ccccc1",
            "p_before": 0.9,
            "p_after": 0.4,
            "pred_before": int(label),
            "pred_after": 0,
            "cf_drop": self.counterfactual_sem,
            "cf_flip": False,
            "counterfactual_sem": self.counterfactual_sem,
            "teacher_sem": self.counterfactual_sem,
            "teacher_reason": "ok",
            "teacher_result_ok": True,
        }


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for counterfactual-teacher tests")
class CounterfactualTeacherTests(unittest.TestCase):
    PARENT = "O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1"
    CORE_FRAGMENT = "CC(=O)O"
    RAW_FRAGMENT = "*CC(=O)O"

    def _build_rewarder(
        self,
        *,
        fragment_teacher=None,
        counterfactual_teacher=None,
    ) -> ChemRLRewarder:
        oracle_bundle = {
            "model": _FakeOracleModel(),
            "fingerprint_radius": 2,
            "fingerprint_bits": 16,
        }
        with patch(
            "src.rewards.reward_wrapper.load_oracle_bundle",
            return_value=oracle_bundle,
        ):
            return ChemRLRewarder(
                oracle_path="unused.pkl",
                teacher_scorer=fragment_teacher,
                counterfactual_teacher_scorer=counterfactual_teacher,
                teacher_sem_scale=1.0,
                teacher_sem_missing_penalty=-5.0,
            )

    def test_delete_one_substructure_removes_single_match(self) -> None:
        result = delete_one_substructure(self.PARENT, self.CORE_FRAGMENT)

        self.assertTrue(result["parent_parse_ok"])
        self.assertTrue(result["fragment_parse_ok"])
        self.assertTrue(result["substructure_ok"])
        self.assertTrue(result["deletion_ok"])
        self.assertTrue(result["parent_without_fragment_smiles"])
        self.assertNotEqual(result["parent_without_fragment_smiles"], self.PARENT)
        self.assertTrue(result["match_atom_indices"])

    def test_counterfactual_teacher_scorer_uses_parent_and_residual_probabilities(self) -> None:
        fake_teacher = _FakeProbabilityTeacherScorer(
            {
                self.PARENT: 0.9,
                "c1ccccc1": 0.4,
            },
            predicted_labels={
                self.PARENT: 1,
                "c1ccccc1": 0,
            },
        )
        scorer = CounterfactualTeacherScorer(
            teacher_path="unused.pkl",
            teacher_scorer=fake_teacher,
            flip_bonus=1.0,
            missing_penalty=-5.0,
        )

        with patch(
            "src.rewards.counterfactual_oracle.delete_one_substructure",
            return_value={
                "parent_parse_ok": True,
                "fragment_parse_ok": True,
                "substructure_ok": True,
                "deletion_ok": True,
                "parent_without_fragment_smiles": "c1ccccc1",
                "match_atom_indices": [0, 1, 2, 3],
                "reason": "ok",
            },
        ):
            result = scorer.score_counterfactual(
                parent_smiles=self.PARENT,
                core_fragment_smiles=self.CORE_FRAGMENT,
                label=1,
            )

        self.assertTrue(result["teacher_result_ok"])
        self.assertAlmostEqual(result["p_before"], 0.9)
        self.assertAlmostEqual(result["p_after"], 0.4)
        self.assertAlmostEqual(result["cf_drop"], 0.5)
        self.assertTrue(result["cf_flip"])
        self.assertAlmostEqual(result["counterfactual_sem"], 1.5)
        self.assertAlmostEqual(result["teacher_sem"], result["counterfactual_sem"])
        self.assertEqual(result["teacher_reason"], "ok")

    def test_wrapper_uses_counterfactual_sem_in_total(self) -> None:
        fragment_teacher = _FakeFragmentTeacherScorer(probability=0.8)
        counterfactual_teacher = _FakeCounterfactualTeacherScorer(counterfactual_sem=0.5)
        rewarder = self._build_rewarder(
            fragment_teacher=fragment_teacher,
            counterfactual_teacher=counterfactual_teacher,
        )

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            [self.RAW_FRAGMENT],
            parent_labels=[1],
        )[0]

        self.assertEqual(trace.breakdown["sem_r"], 0.5)
        self.assertEqual(trace.breakdown["teacher_sem_r"], 0.5)
        self.assertEqual(trace.breakdown["cf_r"], 0.5)
        self.assertGreater(trace.breakdown["fragment_teacher_sem_r"], 0.0)
        expected_total = (
            trace.breakdown["format_r"]
            + trace.breakdown["valid_r"]
            + trace.breakdown["subgraph_r"]
            + trace.breakdown["length_r"]
            + trace.breakdown["cf_r"]
        )
        self.assertAlmostEqual(trace.reward, expected_total)
        self.assertEqual(trace.fragment_teacher_sem, trace.breakdown["fragment_teacher_sem_r"])
        self.assertTrue(trace.counterfactual_teacher_called)
        self.assertEqual(trace.counterfactual_teacher_reason, "ok")

    def test_invalid_fragment_does_not_call_counterfactual_oracle(self) -> None:
        counterfactual_teacher = _FakeCounterfactualTeacherScorer(counterfactual_sem=0.5)
        rewarder = self._build_rewarder(counterfactual_teacher=counterfactual_teacher)

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["not_a_smiles"],
            parent_labels=[0],
        )[0]

        self.assertEqual(len(counterfactual_teacher.calls), 0)
        self.assertFalse(trace.counterfactual_teacher_called)
        self.assertEqual(trace.counterfactual_teacher_reason, "invalid_or_not_substructure")

    def test_non_substructure_fragment_does_not_call_counterfactual_oracle(self) -> None:
        counterfactual_teacher = _FakeCounterfactualTeacherScorer(counterfactual_sem=0.5)
        rewarder = self._build_rewarder(counterfactual_teacher=counterfactual_teacher)

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["N#N"],
            parent_labels=[0],
        )[0]

        self.assertEqual(len(counterfactual_teacher.calls), 0)
        self.assertFalse(trace.counterfactual_teacher_called)
        self.assertEqual(trace.counterfactual_teacher_reason, "invalid_or_not_substructure")

    def test_deletion_failure_has_clear_reason(self) -> None:
        fake_teacher = _FakeProbabilityTeacherScorer({self.PARENT: 0.9})
        scorer = CounterfactualTeacherScorer(
            teacher_path="unused.pkl",
            teacher_scorer=fake_teacher,
            flip_bonus=1.0,
            missing_penalty=-5.0,
        )

        with patch(
            "src.rewards.counterfactual_oracle.delete_one_substructure",
            return_value={
                "parent_parse_ok": True,
                "fragment_parse_ok": True,
                "substructure_ok": True,
                "deletion_ok": False,
                "parent_without_fragment_smiles": None,
                "match_atom_indices": [0, 1],
                "reason": "residual_sanitize_failed:boom",
            },
        ):
            result = scorer.score_counterfactual(
                parent_smiles=self.PARENT,
                core_fragment_smiles=self.CORE_FRAGMENT,
                label=1,
            )

        self.assertFalse(result["teacher_result_ok"])
        self.assertEqual(result["teacher_reason"], "residual_sanitize_failed:boom")


if __name__ == "__main__":
    unittest.main()
