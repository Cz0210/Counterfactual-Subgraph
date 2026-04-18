import unittest

from scripts.train_ppo import (
    build_prompt_example_from_json_row,
    ensure_score_head_for_experimental_ppo,
    extract_parent_smiles_from_prompt,
    normalize_hiv_label,
)
from src.rewards.reward_wrapper import shape_probability_reward


class _DummyVHead:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states)
        return {"hidden_states": hidden_states}


class TrainPPOHelperTests(unittest.TestCase):
    def test_extract_parent_smiles_from_counterfactual_prompt(self) -> None:
        prompt = (
            "You are given a molecule SMILES.\n"
            "ORIGINAL_LABEL: 1\n"
            "MOLECULE_SMILES: CCO\n"
            "FRAGMENT_SMILES:"
        )

        self.assertEqual(extract_parent_smiles_from_prompt(prompt), "CCO")

    def test_normalize_hiv_label_supports_nci_and_binary_forms(self) -> None:
        self.assertEqual(normalize_hiv_label("CI"), 0)
        self.assertEqual(normalize_hiv_label("CM"), 1)
        self.assertEqual(normalize_hiv_label("CA"), 1)
        self.assertEqual(normalize_hiv_label("0"), 0)
        self.assertEqual(normalize_hiv_label(1), 1)
        self.assertIsNone(normalize_hiv_label("UNKNOWN"))

    def test_build_prompt_example_from_json_row_uses_defaults(self) -> None:
        row = {
            "instruction": "MOLECULE_SMILES: CCN\nFRAGMENT_SMILES:",
        }

        example = build_prompt_example_from_json_row(
            row,
            index=3,
            default_parent_label=1,
            only_positive=True,
        )

        assert example is not None
        self.assertEqual(example.index, 3)
        self.assertEqual(example.parent_smiles, "CCN")
        self.assertEqual(example.original_label, 1)

    def test_probability_reward_keeps_dense_exploration_signal(self) -> None:
        self.assertAlmostEqual(shape_probability_reward(0.2), 0.2)
        self.assertAlmostEqual(shape_probability_reward(0.8), 9.0)

    def test_score_adapter_keeps_existing_score(self) -> None:
        model = type("ModelWithScore", (), {})()
        existing = lambda hidden_states: ("existing", hidden_states)
        model.score = existing

        adapted = ensure_score_head_for_experimental_ppo(model, "value_model")

        self.assertIs(adapted.score, existing)
        self.assertEqual(adapted.score("hs"), ("existing", "hs"))

    def test_score_adapter_uses_top_level_v_head(self) -> None:
        model = type("ModelWithVHead", (), {})()
        model.v_head = _DummyVHead()

        adapted = ensure_score_head_for_experimental_ppo(model, "value_model")
        result = adapted.score("hidden_states")

        self.assertTrue(hasattr(adapted, "score"))
        self.assertEqual(result, {"hidden_states": "hidden_states"})
        self.assertEqual(model.v_head.calls, ["hidden_states"])

    def test_score_adapter_finds_nested_v_head(self) -> None:
        wrapped = type("WrappedModel", (), {})()
        wrapped.pretrained_model = type("InnerModel", (), {})()
        wrapped.pretrained_model.v_head = _DummyVHead()

        adapted = ensure_score_head_for_experimental_ppo(wrapped, "value_model")
        result = adapted.score("nested_hidden_states")

        self.assertTrue(hasattr(adapted, "score"))
        self.assertEqual(result, {"hidden_states": "nested_hidden_states"})
        self.assertEqual(
            wrapped.pretrained_model.v_head.calls,
            ["nested_hidden_states"],
        )


if __name__ == "__main__":
    unittest.main()
