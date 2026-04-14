import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.chem.types import DeletionResult
from src.rewards.reward_calculator import CounterfactualReward, load_oracle_bundle


class _FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self._probabilities[None, :], repeats=features.shape[0], axis=0)


class _FakeChemEngine:
    def __init__(self, *, validity_score: float, subgraph_score: float) -> None:
        self.validity_score = validity_score
        self.subgraph_score = subgraph_score

    def check_validity(self, generated_smiles: str) -> float:
        return self.validity_score

    def check_subgraph(self, original_smiles: str, generated_smiles: str) -> float:
        return self.subgraph_score


class RewardCalculatorTests(unittest.TestCase):
    def test_load_oracle_bundle_accepts_minimal_valid_dictionary(self) -> None:
        bundle = {
            "model": _FakeModel([0.9, 0.1]),
            "fingerprint_radius": 2,
            "fingerprint_bits": 2048,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.pkl"
            with path.open("wb") as handle:
                pickle.dump(bundle, handle)

            loaded = load_oracle_bundle(path)

        self.assertEqual(loaded["fingerprint_radius"], 2)
        self.assertEqual(loaded["fingerprint_bits"], 2048)

    def test_compute_reward_early_stops_on_invalid_smiles(self) -> None:
        reward = CounterfactualReward(
            oracle_bundle={
                "model": _FakeModel([0.8, 0.2]),
                "fingerprint_radius": 2,
                "fingerprint_bits": 2048,
            },
            chem_engine=_FakeChemEngine(validity_score=-1.0, subgraph_score=1.0),
        )

        total_reward, breakdown = reward.compute_reward("CCO", "not_a_smiles", 1)

        self.assertEqual(total_reward, reward.invalid_smiles_penalty)
        self.assertEqual(
            breakdown,
            {
                "valid_r": reward.invalid_smiles_penalty,
                "subgraph_r": 0.0,
                "cf_r": 0.0,
            },
        )

    def test_compute_reward_early_stops_on_non_subgraph(self) -> None:
        reward = CounterfactualReward(
            oracle_bundle={
                "model": _FakeModel([0.8, 0.2]),
                "fingerprint_radius": 2,
                "fingerprint_bits": 2048,
            },
            chem_engine=_FakeChemEngine(validity_score=1.0, subgraph_score=-1.0),
        )

        total_reward, breakdown = reward.compute_reward("CCO", "N#N", 1)

        self.assertEqual(
            total_reward,
            reward.valid_pass_reward + reward.invalid_subgraph_penalty,
        )
        self.assertEqual(breakdown["cf_r"], 0.0)

    def test_compute_reward_uses_residual_probability_for_counterfactual_term(self) -> None:
        reward = CounterfactualReward(
            oracle_bundle={
                "model": _FakeModel([0.75, 0.25]),
                "fingerprint_radius": 2,
                "fingerprint_bits": 2048,
            },
            chem_engine=_FakeChemEngine(validity_score=1.0, subgraph_score=1.0),
            flip_bonus=1.0,
            flip_threshold=0.5,
        )

        deletion_result = DeletionResult(
            parent_smiles="CCO",
            fragment_smiles="CO",
            residual_smiles="C",
            success=True,
        )

        with patch(
            "src.rewards.reward_calculator.delete_fragment_from_parent",
            return_value=deletion_result,
        ):
            with patch(
                "src.rewards.reward_calculator.prepare_smiles_for_oracle",
                return_value="C",
            ):
                with patch(
                    "src.rewards.reward_calculator.smiles_to_morgan_array",
                    return_value=np.asarray([1.0, 0.0], dtype=np.float32),
                ):
                    total_reward, breakdown = reward.compute_reward("CCO", "CO", 1)

        expected_cf_r = (0.75 - 0.25) + reward.flip_bonus
        self.assertAlmostEqual(breakdown["cf_r"], expected_cf_r)
        self.assertAlmostEqual(
            total_reward,
            reward.valid_pass_reward + reward.subgraph_pass_reward + expected_cf_r,
        )


if __name__ == "__main__":
    unittest.main()
