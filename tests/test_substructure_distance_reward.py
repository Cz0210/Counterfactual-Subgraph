import unittest

from src.chem import compute_substructure_distance_reward, is_rdkit_available


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for substructure-distance tests")
class SubstructureDistanceRewardTests(unittest.TestCase):
    PARENT = "CCOc1ccc(N)cc1"

    def test_direct_match_returns_identity_reward(self) -> None:
        result = compute_substructure_distance_reward(
            self.PARENT,
            "c1ccc(N)cc1",
        )

        self.assertTrue(result["parse_ok"])
        self.assertTrue(result["direct_substructure"])
        self.assertAlmostEqual(float(result["substructure_similarity"]), 1.0)
        self.assertAlmostEqual(float(result["substructure_distance"]), 0.0)
        self.assertAlmostEqual(float(result["substructure_distance_reward"]), 1.0)
        self.assertEqual(result["projection_method"], "direct_match")

    def test_similar_but_not_direct_fragment_gets_dense_reward(self) -> None:
        result = compute_substructure_distance_reward(
            self.PARENT,
            "c1ccc(Cl)cc1",
        )

        self.assertTrue(result["parse_ok"])
        self.assertFalse(result["direct_substructure"])
        self.assertGreater(float(result["substructure_similarity"]), 0.0)
        self.assertLess(float(result["substructure_distance"]), 1.0)
        self.assertTrue(result["nearest_parent_subgraph_smiles"])
        self.assertEqual(
            result["failure_tag"],
            "parse_ok_but_not_direct_substructure",
        )
        self.assertFalse(bool(result["used_projected_subgraph_for_reward"]))

    def test_unrelated_fragment_scores_lower_than_similar_case(self) -> None:
        similar = compute_substructure_distance_reward(
            self.PARENT,
            "c1ccc(Cl)cc1",
        )
        unrelated = compute_substructure_distance_reward(
            self.PARENT,
            "C1CCCCC1",
        )

        self.assertTrue(unrelated["parse_ok"])
        self.assertFalse(unrelated["direct_substructure"])
        self.assertLess(
            float(unrelated["substructure_similarity"]),
            float(similar["substructure_similarity"]),
        )


if __name__ == "__main__":
    unittest.main()
