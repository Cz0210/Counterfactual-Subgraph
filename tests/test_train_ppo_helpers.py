import argparse
import unittest

from scripts.train_ppo import (
    apply_decoded_chem_generation_defaults,
    build_parser,
    build_prompt_example_from_json_row,
    diagnose_eval_dataloader_for_generate_completions,
    disable_generate_completions_if_needed,
    ensure_reward_model_for_experimental_ppo,
    ensure_score_head_for_experimental_ppo,
    extract_fragment_smiles,
    extract_parent_smiles_from_prompt,
    normalize_hiv_label,
    resolve_decoded_chem_generation_config,
)
from src.rewards.reward_wrapper import (
    detect_obvious_parse_failure_detail,
    preprocess_generated_fragment,
    shape_probability_reward,
)

try:
    import torch
except ImportError:  # pragma: no cover - depends on local test environment
    torch = None


class _DummyVHead:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states)
        return {"hidden_states": hidden_states}


class _ListLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.messages.append(message)


if torch is not None:
    class _DummyBackbone(torch.nn.Module):
        def __init__(self, hidden_size: int = 4) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size, "torch_dtype": torch.float32})()
            self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}


    class _RewardWrapperWithBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _DummyBackbone()


    class _ChemistryOnlyRewardWrapper(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            return torch.zeros((1, 1), dtype=torch.float32)


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

    def test_extract_fragment_smiles_uses_first_non_empty_line(self) -> None:
        raw_text = "\n CCO \nExplanation should be ignored"
        self.assertEqual(extract_fragment_smiles(raw_text), "CCO")

    def test_preprocess_generated_fragment_keeps_only_first_line(self) -> None:
        fragment, char_count = preprocess_generated_fragment("\n CCO \nN#N\n")

        self.assertEqual(fragment, "CCO")
        self.assertEqual(char_count, 3)

    def test_detect_obvious_parse_failure_detail_flags_unclosed_ring(self) -> None:
        self.assertEqual(
            detect_obvious_parse_failure_detail("c1ccccc"),
            "parse_failed_unclosed_ring",
        )

    def test_apply_decoded_chem_generation_defaults_tightens_decoded_length(self) -> None:
        args = argparse.Namespace(
            ppo_loop="decoded_chem",
            gen_max_new_tokens=None,
            max_new_tokens=64,
        )

        apply_decoded_chem_generation_defaults(args, argv=[])

        self.assertEqual(args.gen_max_new_tokens, 48)

    def test_apply_decoded_chem_generation_defaults_respects_legacy_max_new_tokens(self) -> None:
        args = argparse.Namespace(
            ppo_loop="decoded_chem",
            gen_max_new_tokens=None,
            max_new_tokens=40,
        )

        apply_decoded_chem_generation_defaults(args, argv=["--max-new-tokens", "40"])

        self.assertIsNone(args.gen_max_new_tokens)

    def test_resolve_decoded_chem_generation_config_prefers_gen_overrides(self) -> None:
        args = argparse.Namespace(
            gen_max_new_tokens=32,
            max_new_tokens=64,
            gen_temperature=0.6,
            temperature=0.8,
            gen_top_p=0.9,
            top_p=0.95,
            gen_do_sample=False,
        )

        config = resolve_decoded_chem_generation_config(args)

        self.assertEqual(config.max_new_tokens, 32)
        self.assertAlmostEqual(config.temperature, 0.6)
        self.assertAlmostEqual(config.top_p, 0.9)
        self.assertFalse(config.do_sample)

    def test_parser_accepts_parent_repair_flags(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--enable-parent-aware-repair",
                "--repair-min-similarity",
                "0.45",
                "--repair-max-candidates",
                "32",
            ]
        )

        self.assertTrue(args.enable_parent_aware_repair)
        self.assertAlmostEqual(args.repair_min_similarity, 0.45)
        self.assertEqual(args.repair_max_candidates, 32)

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

    def test_eval_dataloader_diagnosis_reports_missing_data_source(self) -> None:
        trainer = type("Trainer", (), {})()
        trainer.eval_dataloader = type("EvalDataLoader", (), {})()
        trainer.eval_dataloader.dataset = [1, 2, 3]
        trainer.eval_dataloader.sampler = type("Sampler", (), {"data_source": None})()

        reason = diagnose_eval_dataloader_for_generate_completions(trainer)

        self.assertEqual(reason, "ppo_trainer.eval_dataloader.sampler.data_source is None")

    def test_disable_generate_completions_installs_no_op_patch(self) -> None:
        trainer = type("Trainer", (), {})()

        def _original_generate_completions(self, *args, **kwargs):
            raise AssertionError("original generate_completions should have been patched out")

        trainer.generate_completions = _original_generate_completions.__get__(trainer, trainer.__class__)
        logger = _ListLogger()

        disable_generate_completions_if_needed(
            trainer,
            logger,
            reason="skip flag enabled",
        )
        result = trainer.generate_completions()

        self.assertIsNone(result)
        self.assertTrue(any("[PPO_GENERATE_COMPLETIONS_SKIPPED]" in message for message in logger.messages))
        self.assertTrue(any("reason=skip flag enabled" in message for message in logger.messages))

    @unittest.skipIf(torch is None, "torch is required for reward adapter helper tests")
    def test_reward_model_adapter_uses_existing_backbone(self) -> None:
        reward_model = _RewardWrapperWithBackbone()

        adapted = ensure_reward_model_for_experimental_ppo(
            reward_model,
            fallback_lm_model=None,
            name="reward_model",
        )

        self.assertIs(adapted, reward_model)
        self.assertTrue(hasattr(adapted, "base_model_prefix"))
        self.assertEqual(adapted.base_model_prefix, "backbone")
        self.assertIs(getattr(adapted, adapted.base_model_prefix), adapted.backbone)
        self.assertTrue(hasattr(adapted, "score"))
        score_output = adapted.score(torch.zeros((2, 3, adapted.backbone.config.hidden_size)))
        self.assertEqual(tuple(score_output.shape), (2, 3, 1))

    @unittest.skipIf(torch is None, "torch is required for reward adapter helper tests")
    def test_reward_model_adapter_builds_fallback_trl_wrapper(self) -> None:
        chemistry_reward_model = _ChemistryOnlyRewardWrapper()
        fallback_backbone = _DummyBackbone(hidden_size=6)

        adapted = ensure_reward_model_for_experimental_ppo(
            chemistry_reward_model,
            fallback_lm_model=fallback_backbone,
            name="reward_model",
        )

        self.assertIsNot(adapted, chemistry_reward_model)
        self.assertTrue(hasattr(adapted, "base_model_prefix"))
        self.assertEqual(adapted.base_model_prefix, "pretrained_model")
        self.assertIs(adapted.pretrained_model, fallback_backbone)
        self.assertTrue(hasattr(adapted, "score"))
        self.assertTrue(getattr(adapted, "interface_compatibility_only", False))
        self.assertIs(getattr(adapted, "chemistry_reward_component"), chemistry_reward_model)
        score_output = adapted.score(torch.zeros((1, 5, fallback_backbone.config.hidden_size)))
        self.assertEqual(tuple(score_output.shape), (1, 5, 1))


if __name__ == "__main__":
    unittest.main()
