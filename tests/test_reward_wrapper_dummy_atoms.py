import unittest
from unittest.mock import patch

import numpy as np

from src.chem import is_rdkit_available
from src.rewards.reward_wrapper import ChemRLRewarder, normalize_fragment_with_dummy_atoms


class _FakeOracleModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self._probabilities[None, :], repeats=features.shape[0], axis=0)


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for dummy-atom reward tests")
class RewardWrapperDummyAtomTests(unittest.TestCase):
    def _build_rewarder(self) -> ChemRLRewarder:
        oracle_bundle = {
            "model": _FakeOracleModel([0.8, 0.2]),
            "fingerprint_radius": 2,
            "fingerprint_bits": 16,
        }
        with patch(
            "src.rewards.reward_wrapper.load_oracle_bundle",
            return_value=oracle_bundle,
        ):
            return ChemRLRewarder(oracle_path="unused.pkl")

    def test_normalize_fragment_with_dummy_atoms_keeps_aliphatic_core(self) -> None:
        info = normalize_fragment_with_dummy_atoms("*CC(=O)O")

        self.assertTrue(info["raw_parse_ok"])
        self.assertTrue(info["has_dummy"])
        self.assertEqual(info["dummy_count"], 1)
        self.assertTrue(info["core_parse_ok"])
        self.assertEqual(info["core_smiles"], "CC(=O)O")
        self.assertEqual(info["core_atom_count"], 4)

    def test_normalize_fragment_with_dummy_atoms_keeps_aromatic_core(self) -> None:
        info = normalize_fragment_with_dummy_atoms("*c1ccccc1")

        self.assertTrue(info["raw_parse_ok"])
        self.assertTrue(info["has_dummy"])
        self.assertEqual(info["dummy_count"], 1)
        self.assertTrue(info["core_parse_ok"])
        self.assertEqual(info["core_smiles"], "c1ccccc1")
        self.assertEqual(info["core_atom_count"], 6)

    def test_reward_trace_uses_core_fragment_for_dummy_atom_substructure(self) -> None:
        rewarder = self._build_rewarder()
        parent = "O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1"
        fragment = "*CC(=O)O"

        trace = rewarder.calculate_reward_details_batch(
            [parent],
            [fragment],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.raw_parse_ok)
        self.assertTrue(trace.has_dummy_atoms)
        self.assertEqual(trace.dummy_count, 1)
        self.assertTrue(trace.core_parse_ok)
        self.assertEqual(trace.core_fragment_smiles, "CC(=O)O")
        self.assertEqual(trace.teacher_input_smiles, "CC(=O)O")
        self.assertTrue(trace.is_subgraph)
        self.assertGreater(trace.breakdown["valid_r"], 0.0)
        self.assertGreater(trace.breakdown["subgraph_r"], 0.0)
        self.assertGreaterEqual(trace.breakdown["length_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
