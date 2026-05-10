import unittest
from unittest.mock import patch

import numpy as np

from src.chem import is_rdkit_available
from src.rewards.reward_wrapper import ChemRLRewarder, normalize_fragment_with_dummy_atoms

try:
    import torch
except ImportError:  # pragma: no cover - depends on local test environment
    torch = None


class _FakeOracleModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self._probabilities[None, :], repeats=features.shape[0], axis=0)


@unittest.skipUnless(is_rdkit_available(), "RDKit is required for dummy-atom reward tests")
class RewardWrapperDummyAtomTests(unittest.TestCase):
    def _build_rewarder(self, **kwargs) -> ChemRLRewarder:
        oracle_bundle = {
            "model": _FakeOracleModel([0.8, 0.2]),
            "fingerprint_radius": 2,
            "fingerprint_bits": 16,
        }
        with patch(
            "src.rewards.reward_wrapper.load_oracle_bundle",
            return_value=oracle_bundle,
        ):
            return ChemRLRewarder(oracle_path="unused.pkl", **kwargs)

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

    def test_normalize_fragment_with_dummy_atoms_preserves_raw_dummy_info_on_parse_failure(self) -> None:
        info = normalize_fragment_with_dummy_atoms("*C?")

        self.assertFalse(info["raw_parse_ok"])
        self.assertTrue(info["raw_has_dummy"])
        self.assertEqual(info["raw_dummy_count"], 1)
        self.assertTrue(info["has_dummy"])
        self.assertEqual(info["dummy_count"], 1)
        self.assertEqual(info["parse_stage"], "raw_with_dummy")
        self.assertFalse(info["dummy_removed_before_parse"])
        self.assertFalse(info["parsed_raw_with_dummy"])
        self.assertFalse(info["parsed_core"])
        self.assertEqual(info["parse_failed_reason"], "parse_failed_raw_with_dummy")

    def test_normalize_fragment_without_dummy_uses_raw_without_dummy_bucket(self) -> None:
        info = normalize_fragment_with_dummy_atoms("C?C")

        self.assertFalse(info["raw_parse_ok"])
        self.assertFalse(info["raw_has_dummy"])
        self.assertEqual(info["raw_dummy_count"], 0)
        self.assertFalse(info["has_dummy"])
        self.assertEqual(info["dummy_count"], 0)
        self.assertEqual(info["parse_stage"], "raw_without_dummy")
        self.assertEqual(info["parse_failed_reason"], "parse_failed_raw_without_dummy")

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
        self.assertLess(trace.breakdown["dummy_r"], 0.0)
        self.assertGreater(trace.breakdown["valid_r"], 0.0)
        self.assertGreater(trace.breakdown["subgraph_r"], 0.0)
        self.assertGreaterEqual(trace.breakdown["length_r"], 0.0)

    def test_reward_trace_marks_parse_failed_after_dummy_removal(self) -> None:
        rewarder = self._build_rewarder()
        parent = "CCO"
        fragment = "*"

        trace = rewarder.calculate_reward_details_batch(
            [parent],
            [fragment],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.raw_parse_ok)
        self.assertTrue(trace.raw_has_dummy)
        self.assertEqual(trace.raw_dummy_count, 1)
        self.assertEqual(trace.failure_tag, "parse_failed_after_dummy_removal")
        self.assertEqual(trace.parse_stage, "core_after_dummy_removal")
        self.assertTrue(trace.parsed_raw_with_dummy)
        self.assertFalse(trace.parsed_core)
        self.assertFalse(trace.dummy_removed_before_parse)
        self.assertEqual(trace.parse_failed_reason, "parse_failed_after_dummy_removal")

    def test_tiny_fragment_hard_fail_overrides_positive_terms(self) -> None:
        rewarder = self._build_rewarder(
            min_fragment_atoms=3,
            tiny_fragment_hard_fail_penalty=-6.0,
        )

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["O"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.tiny_fragment_hard_fail)
        self.assertEqual(trace.fragment_atom_count, 1)
        self.assertEqual(trace.min_fragment_atoms, 3)
        self.assertEqual(trace.failure_tag, "tiny_fragment_hard_fail")
        self.assertAlmostEqual(trace.reward, -6.0)

    def test_component_salvage_runs_before_not_connected_failure(self) -> None:
        rewarder = self._build_rewarder(
            component_salvage_min_atoms=2,
            min_residual_atoms=0,
        )

        trace = rewarder.calculate_reward_details_batch(
            ["CCCCO"],
            ["CCC.O"],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.component_salvage_attempted)
        self.assertTrue(trace.component_salvage_success)
        self.assertGreaterEqual(trace.raw_component_count, 2)
        self.assertEqual(trace.component_salvage_stage, "raw")
        self.assertTrue(trace.salvaged_fragment)

    def test_minimal_syntax_repair_can_flow_into_projection(self) -> None:
        rewarder = self._build_rewarder(
            enable_parent_projection=True,
            projection_min_score=0.35,
            projection_min_atoms=3,
            projection_max_atom_ratio=0.70,
            projection_mcs_timeout=1,
            enable_minimal_syntax_repair=True,
            syntax_repair_min_atoms=3,
        )
        parent = "CC(=O)Oc1ccccc1C(=O)O"

        trace = rewarder.calculate_reward_details_batch(
            [parent],
            ["CC(=O)N("],
            parent_labels=[1],
        )[0]

        self.assertTrue(trace.repair_attempted)
        self.assertTrue(trace.repair_success)
        self.assertTrue(trace.repair_candidate_accepted)
        self.assertGreater(trace.repair_candidate_count, 0)
        self.assertGreater(trace.repair_candidates_parse_ok, 0)
        self.assertGreater(trace.repair_candidates_core_ok, 0)
        self.assertEqual(trace.repair_accept_stage, "projection")
        self.assertTrue(trace.projection_success)

    def test_size_window_reward_prefers_medium_fragment(self) -> None:
        rewarder = self._build_rewarder(
            enable_size_window_reward=True,
            size_window_low=0.15,
            size_window_high=0.65,
            size_window_bonus=0.4,
            size_window_small_penalty=-0.4,
            size_window_large_penalty=-0.4,
            min_residual_atoms=0,
            min_residual_ratio=0.0,
        )

        trace = rewarder.calculate_reward_details_batch(
            ["CCCCCC"],
            ["CCC"],
            parent_labels=[1],
        )[0]

        self.assertEqual(trace.size_window_bucket, "in_window")
        self.assertAlmostEqual(trace.size_window_reward, 0.4)
        self.assertEqual(trace.final_fragment_atom_count, 3)
        self.assertAlmostEqual(trace.final_fragment_atom_ratio or 0.0, 0.5)

    def test_full_parent_fail_keeps_component_fields_without_keyword_collision(self) -> None:
        rewarder = self._build_rewarder()

        trace = rewarder.calculate_reward_details_batch(
            ["CCO"],
            ["CCO"],
            parent_labels=[1],
        )[0]

        self.assertEqual(trace.failure_tag, "full_parent_fragment")
        self.assertEqual(trace.raw_component_count, 1)
        self.assertEqual(trace.core_component_count, 1)
        self.assertTrue(hasattr(trace, "size_window_bucket"))

    def test_near_parent_fail_keeps_size_window_fields_without_keyword_collision(self) -> None:
        rewarder = self._build_rewarder()

        trace = rewarder.calculate_reward_details_batch(
            ["CCCCCCC"],
            ["CCCCCC"],
            parent_labels=[1],
        )[0]

        self.assertEqual(trace.failure_tag, "near_parent_fragment")
        self.assertEqual(trace.raw_component_count, 1)
        self.assertEqual(trace.core_component_count, 1)
        self.assertEqual(trace.size_window_bucket, "hard_failed_near_parent")

    @unittest.skipIf(torch is None, "torch is required for decoded reward-wrapper tests")
    def test_compute_rewards_from_decoded_not_direct_substructure_returns_failure_log(self) -> None:
        rewarder = self._build_rewarder(
            enable_parent_projection=True,
            enable_substructure_distance_reward=True,
            substructure_distance_reward_weight=0.0,
            substructure_distance_sim_threshold=0.95,
            enable_size_window_reward=False,
            format_pass_reward=0.0,
            valid_pass_reward=0.0,
            partial_valid_reward=0.0,
            compactness_bonus=0.0,
            min_fragment_atoms=0,
            min_residual_atoms=0,
            min_residual_ratio=0.0,
        )
        parent = "CCOc1ccc(N)cc1"
        fragment = "C1CCCCC1"

        reward_tensor, reward_logs = rewarder.compute_rewards_from_decoded(
            parent_smiles=[parent],
            generated_fragments=[fragment],
            raw_outputs=[fragment],
            labels=[1],
            metas=[{"id": "projection-failure-regression"}],
            device="cpu",
        )

        self.assertEqual(tuple(reward_tensor.shape), (1,))
        self.assertEqual(len(reward_logs), 1)

        reward_log = reward_logs[0]
        self.assertEqual(
            reward_log["failure_tag"],
            "parse_ok_but_not_direct_substructure",
        )
        self.assertEqual(reward_log["invalid_detail"], "not_parent_substructure")
        self.assertFalse(reward_log["direct_substructure"])
        self.assertFalse(reward_log["substructure"])
        self.assertTrue(reward_log["valid"] or reward_log["parse_ok"])
        self.assertIn("projection_attempted", reward_log)
        self.assertIn("projection_success", reward_log)
        self.assertIn("projection_method", reward_log)
        self.assertLessEqual(float(reward_log["reward_total"]), 0.5)

    def test_merge_failure_fields_avoids_duplicate_keyword_collisions(self) -> None:
        rewarder = self._build_rewarder()

        failure_kwargs = rewarder._merge_failure_fields(
            {
                "direct_substructure": False,
                "substructure_similarity": 0.2,
                "substructure_distance": 0.8,
                "substructure_distance_reward": 0.0,
                "used_projected_subgraph_for_reward": False,
                "projection_attempted": True,
                "projection_success": False,
                "projection_method": "nearest_parent_subgraph",
                "projection_score": 0.2,
                "projection_source": "debug",
                "projected_fragment_smiles": None,
                "projection_reason": "projection_failed_low_score",
                "failure_tag": "from_trace_dict",
                "invalid_detail": "from_trace_dict",
            },
            parent_smiles="CCO",
            generated_smiles="N1CCCCC1",
            normalized_generated="N1CCCCC1",
            original_label=1,
            failure_stage="subgraph",
            error_message="Simulated non-substructure failure.",
            breakdown=rewarder._build_breakdown(
                format_reward=0.0,
                valid_reward=0.0,
                subgraph_reward=0.0,
                length_reward=0.0,
                semantic_reward=0.0,
                fragment_teacher_reward=0.0,
                counterfactual_reward=0.0,
            ),
            valid_smiles=True,
            connected_fragment=True,
            is_subgraph=False,
            direct_substructure=False,
            failure_tag="parse_ok_but_not_direct_substructure",
            invalid_detail="not_parent_substructure",
        )

        trace = rewarder._fail(**failure_kwargs)

        self.assertEqual(trace.failure_tag, "parse_ok_but_not_direct_substructure")
        self.assertEqual(trace.invalid_detail, "not_parent_substructure")
        self.assertFalse(trace.direct_substructure)
        self.assertTrue(trace.projection_attempted)
        self.assertFalse(trace.projection_success)
        self.assertEqual(trace.projection_reason, "projection_failed_low_score")


if __name__ == "__main__":
    unittest.main()
