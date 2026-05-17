import unittest

from scripts.make_stratified_ppo_prompts import stratify_and_shuffle_rows


class StratifiedPromptsTests(unittest.TestCase):
    def test_stratified_shuffle_keeps_all_rows_and_builds_block_summary(self) -> None:
        rows = [
            {"parent_smiles": "CCO", "label": "1"},
            {"parent_smiles": "CC.C", "label": "1"},
            {"parent_smiles": "c1ccccc1", "label": "1"},
            {"parent_smiles": "CCCCCCCCCCCCCCCCCCCC", "label": "1"},
        ]

        shuffled_rows, summary = stratify_and_shuffle_rows(
            rows,
            seed=13,
            smiles_col="parent_smiles",
        )

        self.assertEqual(len(shuffled_rows), 4)
        self.assertEqual(summary["num_rows"], 4)
        self.assertEqual(len(summary["blocks"]), 1)
        self.assertIn("bucket_sizes", summary)

