import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_label_ppo_prompt_csv import build_label_prompt_csv


class BuildLabelPPOPromptCSVTests(unittest.TestCase):
    def test_builds_minimal_label_specific_csv_from_shared_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "train.csv"
            out_csv = Path(tmpdir) / "label0.csv"
            out_json = Path(tmpdir) / "label0.summary.json"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles", "HIV_active"])
                writer.writeheader()
                writer.writerow({"smiles": "CCO", "HIV_active": "1"})
                writer.writerow({"smiles": "CCN", "HIV_active": "0"})
                writer.writerow({"smiles": "CCC", "HIV_active": "0"})

            summary = build_label_prompt_csv(
                source_path=source,
                out_csv=out_csv,
                out_json=out_json,
                target_label=0,
                label_col="HIV_active",
                smiles_col="smiles",
            )

            with out_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(["smiles", "label"], list(rows[0].keys()))
            self.assertEqual([row["label"] for row in rows], ["0", "0"])
            self.assertEqual(summary["num_rows"], 2)
            self.assertEqual(json.loads(out_json.read_text(encoding="utf-8"))["target_label"], 0)

    def test_builds_from_sft_jsonl_with_label_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sft_v3_hiv_train.jsonl"
            out_csv = Path(tmpdir) / "label1.csv"
            out_json = Path(tmpdir) / "label1.summary.json"
            rows = [
                {"smiles": "CCO", "label": 1},
                {"parent_smiles": "CCN", "label": 0},
                {"smiles": "CCC", "label": 1},
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            build_label_prompt_csv(
                source_path=source,
                out_csv=out_csv,
                out_json=out_json,
                target_label=1,
                label_col="HIV_active",
                smiles_col="smiles",
            )

            with out_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                output_rows = list(csv.DictReader(handle))

            self.assertEqual([row["smiles"] for row in output_rows], ["CCO", "CCC"])
            self.assertEqual([row["label"] for row in output_rows], ["1", "1"])


if __name__ == "__main__":
    unittest.main()
