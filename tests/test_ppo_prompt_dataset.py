from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.ppo_prompt_dataset import load_ppo_prompt_records


class PPOPromptDatasetTests(unittest.TestCase):
    def test_load_ppo_prompt_records_handles_prompt_fallback_and_rebuilds_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "prompts.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["parent_smiles", "label", "prompt"],
                )
                writer.writeheader()
                writer.writerow({"parent_smiles": "CCO", "label": "1", "prompt": ""})
                writer.writerow(
                    {
                        "parent_smiles": "",
                        "label": "1",
                        "prompt": "ORIGINAL_LABEL: 1\nMOLECULE_SMILES: CCN\nFRAGMENT_SMILES:",
                    }
                )
                writer.writerow({"parent_smiles": "CCC", "label": "0", "prompt": "skip"})

            records, metadata = load_ppo_prompt_records(
                dataset_path,
                label_col="label",
                smiles_col="parent_smiles",
                target_label=1,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].parent_smiles, "CCO")
            self.assertIn("ORIGINAL_LABEL: 1", records[0].prompt)
            self.assertEqual(records[1].parent_smiles, "CCN")
            self.assertEqual(records[1].label, 1)
            self.assertEqual(metadata["resolved_smiles_col"], "parent_smiles")
            self.assertEqual(metadata["resolved_label_col"], "label")
            self.assertEqual(metadata["usable_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
