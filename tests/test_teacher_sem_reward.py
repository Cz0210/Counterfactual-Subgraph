import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.chem import is_rdkit_available
from src.rewards.reward_wrapper import ChemRLRewarder
from src.rewards.teacher_semantic import (
    TeacherSemanticScorer,
    require_teacher_semantic_scorer,
)


class _FakeOracleModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self._probabilities[None, :], repeats=features.shape[0], axis=0)


class _FakeTeacherScorer:
    def __init__(self, probability: float = 0.8) -> None:
        self.available = True
        self.availability_reason = "ok"
        self.teacher_format = "fake_predict_proba"
        self.probability = float(probability)
        self.calls: list[dict[str, object]] = []

    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        parent_smiles: str | None = None,
        meta: dict | None = None,
    ) -> dict[str, object]:
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
            "teacher_label": int(label) if label is not None else None,
            "teacher_sem": 2.0 * self.probability - 1.0,
            "teacher_reason": "ok",
            "teacher_result_ok": True,
            "teacher_format": self.teacher_format,
        }


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for teacher-semantic reward tests")
class TeacherSemanticRewardTests(unittest.TestCase):
    def _build_rewarder(self, *, teacher_scorer=None) -> ChemRLRewarder:
        oracle_bundle = {
            "model": _FakeOracleModel([0.2, 0.8]),
            "fingerprint_radius": 2,
            "fingerprint_bits": 16,
        }
        with patch(
            "src.rewards.reward_wrapper.load_oracle_bundle",
            return_value=oracle_bundle,
        ):
            return ChemRLRewarder(
                oracle_path="unused.pkl",
                teacher_scorer=teacher_scorer,
                teacher_sem_scale=1.0,
                teacher_sem_missing_penalty=-5.0,
            )

    def test_dummy_atom_fragment_uses_core_smiles_for_teacher_sem(self) -> None:
        teacher = _FakeTeacherScorer(probability=0.8)
        rewarder = self._build_rewarder(teacher_scorer=teacher)
        parent = "O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1"
        fragment = "*CC(=O)O"

        trace = rewarder.calculate_reward_details_batch(
            [parent],
            [fragment],
            parent_labels=[0],
        )[0]

        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(teacher.calls[0]["smiles"], "CC(=O)O")
        self.assertTrue(trace.raw_parse_ok)
        self.assertTrue(trace.core_parse_ok)
        self.assertEqual(trace.core_fragment_smiles, "CC(=O)O")
        self.assertTrue(trace.is_subgraph)
        self.assertTrue(trace.teacher_called)
        self.assertEqual(trace.teacher_input_smiles, "CC(=O)O")
        self.assertEqual(trace.teacher_probability, 0.8)
        self.assertEqual(trace.teacher_reason, "ok")
        self.assertGreater(trace.breakdown["fragment_teacher_sem_r"], 0.0)

    def test_invalid_fragment_does_not_call_teacher(self) -> None:
        teacher = _FakeTeacherScorer(probability=0.8)
        rewarder = self._build_rewarder(teacher_scorer=teacher)

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["not_a_smiles"],
            parent_labels=[0],
        )[0]

        self.assertEqual(len(teacher.calls), 0)
        self.assertFalse(trace.teacher_called)
        self.assertEqual(trace.teacher_reason, "invalid_or_not_substructure")
        self.assertEqual(
            trace.breakdown["fragment_teacher_sem_r"],
            rewarder.teacher_sem_missing_penalty,
        )

    def test_non_substructure_does_not_call_teacher(self) -> None:
        teacher = _FakeTeacherScorer(probability=0.8)
        rewarder = self._build_rewarder(teacher_scorer=teacher)

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["N#N"],
            parent_labels=[0],
        )[0]

        self.assertEqual(len(teacher.calls), 0)
        self.assertFalse(trace.teacher_called)
        self.assertEqual(trace.teacher_reason, "invalid_or_not_substructure")
        self.assertEqual(
            trace.breakdown["fragment_teacher_sem_r"],
            rewarder.teacher_sem_missing_penalty,
        )


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for teacher-semantic scorer tests")
class TeacherSemanticScorerTests(unittest.TestCase):
    def test_predict_proba_bundle_scoring(self) -> None:
        fake_bundle = {
            "model": _FakeOracleModel([0.8, 0.2]),
            "fingerprint_radius": 2,
            "fingerprint_bits": 16,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            teacher_path = Path(tmpdir) / "teacher.pkl"
            teacher_path.write_bytes(b"placeholder")
            with patch(
                "src.rewards.teacher_semantic.load_oracle_bundle",
                return_value=fake_bundle,
            ):
                scorer = TeacherSemanticScorer(teacher_path)
                result = scorer.score_smiles("CC(=O)O", label=0)

        self.assertTrue(scorer.available)
        self.assertEqual(scorer.teacher_format, "sklearn_bundle")
        self.assertTrue(result["teacher_result_ok"])
        self.assertAlmostEqual(float(result["teacher_prob"]), 0.8)
        self.assertGreater(float(result["teacher_sem"]), 0.0)

    def test_require_teacher_semantic_scorer_raises_when_missing(self) -> None:
        scorer = TeacherSemanticScorer("missing_teacher.pkl")

        with self.assertRaises(RuntimeError):
            require_teacher_semantic_scorer(
                scorer,
                teacher_path="missing_teacher.pkl",
            )


if __name__ == "__main__":
    unittest.main()
