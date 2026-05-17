import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.teacher_confidence_filter import (
    TeacherConfidenceFilterConfig,
    filter_prompt_rows_by_teacher_confidence,
    write_filtered_prompt_outputs,
)


class _FakeTeacherScorer:
    def __init__(self, outputs_by_smiles):
        self.available = True
        self.availability_reason = "ok"
        self.teacher_format = "fake_predict_proba"
        self.teacher_path = Path("/tmp/fake_teacher.pkl")
        self._outputs_by_smiles = dict(outputs_by_smiles)

    def score_smiles(self, smiles, label=None, parent_smiles=None, meta=None):
        del label, parent_smiles, meta
        payload = dict(self._outputs_by_smiles[smiles])
        payload.setdefault("teacher_available", True)
        payload.setdefault("teacher_result_ok", True)
        payload.setdefault("teacher_reason", "ok")
        payload.setdefault("teacher_input_smiles", smiles)
        payload.setdefault("teacher_format", self.teacher_format)
        return payload


class TeacherConfidenceFilterTests(unittest.TestCase):
    def test_filter_falls_back_from_parent_smiles_to_smiles_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "ppo_prompts.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["smiles", "label", "prompt", "extra"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "smiles": "CCO",
                        "label": "1",
                        "prompt": "MOLECULE_SMILES: CCO\nFRAGMENT_SMILES:",
                        "extra": "keep",
                    }
                )
                writer.writerow(
                    {
                        "smiles": "CCN",
                        "label": "1",
                        "prompt": "MOLECULE_SMILES: CCN\nFRAGMENT_SMILES:",
                        "extra": "drop_low_conf",
                    }
                )
                writer.writerow(
                    {
                        "smiles": "",
                        "label": "1",
                        "prompt": "MOLECULE_SMILES: CCC\nFRAGMENT_SMILES:",
                        "extra": "drop_wrong_label",
                    }
                )
                writer.writerow(
                    {
                        "smiles": "CCCC",
                        "label": "0",
                        "prompt": "MOLECULE_SMILES: CCCC\nFRAGMENT_SMILES:",
                        "extra": "non_target",
                    }
                )

            scorer = _FakeTeacherScorer(
                {
                    "CCO": {"teacher_prob": 0.9, "teacher_label": 1},
                    "CCN": {"teacher_prob": 0.4, "teacher_label": 1},
                    "CCC": {"teacher_prob": 0.8, "teacher_label": 0},
                }
            )
            result = filter_prompt_rows_by_teacher_confidence(
                dataset_path,
                teacher_path="/tmp/fake_teacher.pkl",
                config=TeacherConfidenceFilterConfig(
                    label_col="label",
                    smiles_col="parent_smiles",
                    target_label=1,
                    min_p_label=0.5,
                    require_teacher_correct=True,
                ),
                teacher_scorer=scorer,
            )

            self.assertEqual(result.summary["resolved_smiles_col"], "smiles")
            self.assertEqual(result.summary["target_label_count"], 3)
            self.assertEqual(result.summary["kept_count"], 1)
            self.assertAlmostEqual(float(result.summary["kept_rate"]), 1.0 / 3.0, places=6)
            self.assertAlmostEqual(
                float(result.summary["teacher_correct_rate_before"]),
                2.0 / 3.0,
                places=6,
            )
            self.assertEqual(result.summary["low_confidence_removed_count"], 1)
            self.assertEqual(result.summary["very_low_confidence_removed_count"], 0)
            self.assertEqual(result.summary["drop_reason_counts"]["low_teacher_confidence"], 1)
            self.assertEqual(result.summary["drop_reason_counts"]["teacher_incorrect"], 1)
            self.assertEqual(len(result.kept_rows), 1)
            self.assertEqual(result.kept_rows[0]["extra"], "keep")

    def test_write_filtered_outputs_preserves_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "ppo_prompts.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles", "label", "prompt"])
                writer.writeheader()
                writer.writerow(
                    {
                        "smiles": "CCO",
                        "label": "1",
                        "prompt": "MOLECULE_SMILES: CCO\nFRAGMENT_SMILES:",
                    }
                )

            scorer = _FakeTeacherScorer({"CCO": {"teacher_prob": 0.7, "teacher_label": 1}})
            result = filter_prompt_rows_by_teacher_confidence(
                dataset_path,
                teacher_path="/tmp/fake_teacher.pkl",
                config=TeacherConfidenceFilterConfig(
                    target_label=1,
                    min_p_label=0.5,
                    require_teacher_correct=True,
                ),
                teacher_scorer=scorer,
            )

            out_csv = Path(tmpdir) / "filtered.csv"
            out_json = Path(tmpdir) / "summary.json"
            write_filtered_prompt_outputs(result, out_csv=out_csv, out_json=out_json)

            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            summary = json.loads(out_json.read_text(encoding="utf-8"))

            self.assertEqual(reader.fieldnames, ["smiles", "label", "prompt"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["smiles"], "CCO")
            self.assertEqual(summary["kept_count"], 1)
            self.assertEqual(summary["target_label_count"], 1)


if __name__ == "__main__":
    unittest.main()
