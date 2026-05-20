from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.unified_ppo_prompts import (
    PromptBuildConfig,
    UnifiedPromptBuildConfig,
    build_label_specific_prompt_rows,
    build_unified_prompt_rows,
    check_unified_prompt_balance,
    write_prompt_csv_and_summary,
)


class UnifiedPPOPromptTests(unittest.TestCase):
    def test_builds_label_specific_and_unified_prompt_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "parents.csv"
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles", "HIV_active"])
                writer.writeheader()
                writer.writerow({"smiles": "CCO", "HIV_active": "1"})
                writer.writerow({"smiles": "CCN", "HIV_active": "1"})
                writer.writerow({"smiles": "CCC", "HIV_active": "0"})
                writer.writerow({"smiles": "c1ccccc1O", "HIV_active": "0"})

            header0, rows0, summary0 = build_label_specific_prompt_rows(
                input_csv,
                config=PromptBuildConfig(
                    label_col="HIV_active",
                    smiles_col="smiles",
                    label=0,
                ),
            )
            header1, rows1, summary1 = build_label_specific_prompt_rows(
                input_csv,
                config=PromptBuildConfig(
                    label_col="HIV_active",
                    smiles_col="smiles",
                    label=1,
                ),
            )

            self.assertEqual(summary0["kept_count"], 2)
            self.assertEqual(summary1["kept_count"], 2)
            self.assertIn("Original class label: 0", rows0[0]["prompt"])
            self.assertIn("FRAGMENT_SMILES:", rows1[0]["prompt"])

            label0_csv = Path(tmpdir) / "label0.csv"
            label0_json = Path(tmpdir) / "label0.summary.json"
            write_prompt_csv_and_summary(
                header=header0,
                rows=rows0,
                summary=summary0,
                out_csv=label0_csv,
                out_json=label0_json,
            )
            label1_csv = Path(tmpdir) / "label1.csv"
            label1_json = Path(tmpdir) / "label1.summary.json"
            write_prompt_csv_and_summary(
                header=header1,
                rows=rows1,
                summary=summary1,
                out_csv=label1_csv,
                out_json=label1_json,
            )

            unified_header, unified_rows, unified_summary = build_unified_prompt_rows(
                label0_csv,
                label1_csv,
                config=UnifiedPromptBuildConfig(balance_labels=True, seed=13),
            )
            self.assertEqual(unified_summary["selected_label0_count"], 2)
            self.assertEqual(unified_summary["selected_label1_count"], 2)
            self.assertEqual(len(unified_rows), 4)
            self.assertEqual(int(unified_rows[0]["label"]), 0)
            self.assertEqual(int(unified_rows[1]["label"]), 1)
            self.assertIn("0", unified_summary["prompt_examples_by_label"])
            self.assertIn("1", unified_summary["prompt_examples_by_label"])

            unified_csv = Path(tmpdir) / "unified.csv"
            unified_json = Path(tmpdir) / "unified.summary.json"
            write_prompt_csv_and_summary(
                header=unified_header,
                rows=unified_rows,
                summary=unified_summary,
                out_csv=unified_csv,
                out_json=unified_json,
            )
            balance_summary = check_unified_prompt_balance(unified_csv, block_size=2)
            self.assertEqual(balance_summary["label_counts"]["0"], 2)
            self.assertEqual(balance_summary["label_counts"]["1"], 2)
            self.assertEqual(balance_summary["blocks"][0]["label0_count"], 1)
            self.assertEqual(balance_summary["blocks"][0]["label1_count"], 1)


if __name__ == "__main__":
    unittest.main()
