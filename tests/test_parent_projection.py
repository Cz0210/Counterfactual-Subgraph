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


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for projection tests")
class ParentProjectionTests(unittest.TestCase):
    PARENT = "CC(=O)Oc1ccccc1C(=O)O"

    def _build_rewarder(self) -> ChemRLRewarder:
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
                enable_parent_projection=True,
                projection_min_score=0.35,
                projection_max_candidates=128,
                projection_min_atoms=3,
                projection_max_atom_ratio=0.70,
                projection_penalty=0.5,
                projection_mcs_timeout=1,
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

    def test_rewarder_scores_projected_fragment_and_applies_penalty(self) -> None:
        rewarder = self._build_rewarder()

        trace = rewarder.calculate_reward_details_batch(
            [self.PARENT],
            ["CC(=O)N"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.projection_attempted)
        self.assertTrue(trace.projection_success)
        self.assertEqual(trace.projection_method, "retrieval")
        self.assertEqual(trace.raw_fragment_smiles, "CC(=O)N")
        self.assertTrue(trace.core_fragment_smiles)
        self.assertTrue(is_parent_substructure(self.PARENT, trace.core_fragment_smiles or ""))
        self.assertAlmostEqual(trace.projection_penalty, 0.5)
        unpenalized = (
            trace.breakdown["format_r"]
            + trace.breakdown["valid_r"]
            + trace.breakdown["subgraph_r"]
            + trace.breakdown["length_r"]
            + trace.breakdown["size_window_r"]
            + trace.breakdown["dummy_r"]
            + trace.breakdown["sem_r"]
        )
        self.assertAlmostEqual(trace.reward, unpenalized - 0.5)


if __name__ == "__main__":
    unittest.main()
