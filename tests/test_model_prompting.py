import unittest

from src.models import (
    FEW_SHOT_EXAMPLES,
    build_chemllm_prompt,
    build_counterfactual_system_prompt,
    clean_generated_smiles,
)


class ModelPromptingTests(unittest.TestCase):
    def test_system_prompt_forbids_dummy_atoms_and_mentions_connected_subgraphs(self) -> None:
        system_prompt = build_counterfactual_system_prompt()

        self.assertIn("do not use dummy atoms", system_prompt.lower())
        self.assertIn("connected", system_prompt)
        self.assertIn("subgraph", system_prompt)

    def test_prompt_builder_includes_few_shot_examples(self) -> None:
        prompt = build_chemllm_prompt("Cc1ccccc1", label=1)

        self.assertIn("PARENT_SMILES: Cc1ccccc1", prompt)
        self.assertIn("ORIGINAL_LABEL: 1", prompt)
        self.assertGreaterEqual(len(FEW_SHOT_EXAMPLES), 3)
        self.assertIn("FRAGMENT_SMILES: C", prompt)
        self.assertIn("FRAGMENT_SMILES: CO", prompt)

    def test_clean_generated_smiles_strips_verbose_prefixes(self) -> None:
        raw_text = "The SMILES is: c1ccccc1"
        self.assertEqual(clean_generated_smiles(raw_text), "c1ccccc1")

    def test_clean_generated_smiles_prefers_core_token(self) -> None:
        raw_text = "Answer:\nFRAGMENT_SMILES: O\nThis is the final answer."
        self.assertEqual(clean_generated_smiles(raw_text), "O")


if __name__ == "__main__":
    unittest.main()
