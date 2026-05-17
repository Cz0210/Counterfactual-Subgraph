import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.eval.reward_teacher_audit import (
    RewardTeacherAuditConfig,
    audit_teacher_parent_reliability,
    load_dataset_parent_records,
    run_reward_teacher_audit,
)


class _FakeTeacherScorer:
    def __init__(self, outputs_by_smiles):
        self.teacher_path = Path("/tmp/fake_teacher.pkl")
        self.teacher_format = "fake_predict_proba"
        self.available = True
        self.availability_reason = "ok"
        self._outputs_by_smiles = dict(outputs_by_smiles)

    def score_smiles(self, smiles, label=None, parent_smiles=None, meta=None):
        del label, parent_smiles, meta
        payload = dict(self._outputs_by_smiles[smiles])
        payload.setdefault("teacher_available", True)
        payload.setdefault("teacher_result_ok", True)
        payload.setdefault("teacher_reason", "ok")
        payload.setdefault("teacher_format", self.teacher_format)
        payload.setdefault("teacher_input_smiles", smiles)
        return payload


class RewardTeacherAuditTests(unittest.TestCase):
    def test_teacher_parent_reliability_uses_prompt_fallback_and_groups_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "parents.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt", "label"])
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCO\nFRAGMENT_SMILES:",
                        "label": 1,
                    }
                )
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCN\nFRAGMENT_SMILES:",
                        "label": 0,
                    }
                )
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCC\nFRAGMENT_SMILES:",
                        "label": 1,
                    }
                )

            records, metadata = load_dataset_parent_records(
                dataset_path,
                label_col="label",
                smiles_col="parent_smiles",
            )
            scorer = _FakeTeacherScorer(
                {
                    "CCO": {"teacher_prob": 0.9, "teacher_label": 1, "teacher_sem": 0.8},
                    "CCN": {"teacher_prob": 0.8, "teacher_label": 0, "teacher_sem": 0.6},
                    "CCC": {"teacher_prob": 0.3, "teacher_label": 0, "teacher_sem": -0.4},
                }
            )
            summary = audit_teacher_parent_reliability(records, teacher_scorer=scorer)

        self.assertEqual(metadata["resolved_smiles_col"], "prompt")
        self.assertEqual(len(records), 3)
        self.assertEqual(summary["num_total"], 3)
        self.assertAlmostEqual(float(summary["teacher_correct_rate"]), 2.0 / 3.0, places=6)
        self.assertEqual(summary["low_confidence_count"], 1)
        self.assertEqual(summary["very_low_confidence_count"], 0)
        self.assertIn("0", summary["by_label"])
        self.assertIn("1", summary["by_label"])
        self.assertAlmostEqual(float(summary["by_label"]["0"]["teacher_correct_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(summary["by_label"]["1"]["teacher_correct_rate"]), 0.5, places=6)

    def test_full_reward_teacher_audit_writes_outputs_and_flags_loopholes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "parents.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt", "label"])
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCOCC\nFRAGMENT_SMILES:",
                        "label": 1,
                    }
                )
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCOCC\nFRAGMENT_SMILES:",
                        "label": 1,
                    }
                )
                writer.writerow(
                    {
                        "prompt": "MOLECULE_SMILES: CCNCC\nFRAGMENT_SMILES:",
                        "label": 1,
                    }
                )

            pool_path = tmp_path / "candidate_pool.jsonl"
            pool_rows = [
                {
                    "parent_smiles": "CCOCC",
                    "original_label": 1,
                    "reward_total": 0.5,
                    "cf_drop": 0.2,
                    "cf_flip": False,
                    "p_before": 0.9,
                    "p_after": 0.7,
                    "counterfactual_called": True,
                    "counterfactual_reason": "ok",
                    "parent_without_fragment_smiles": "CCC",
                    "direct_substructure": True,
                    "final_substructure": True,
                    "used_projected_subgraph_for_reward": False,
                    "parse_ok": True,
                    "valid": True,
                    "atom_ratio": 0.2,
                    "fragment_atom_count": 2,
                    "projection_score": None,
                },
                {
                    "parent_smiles": "CCOCC",
                    "label": 1,
                    "reward_total": 1.2,
                    "teacher_p_before": 0.95,
                    "teacher_p_after": 0.10,
                    "teacher_cf_drop": 0.85,
                    "counterfactual_flip": True,
                    "counterfactual_called": True,
                    "counterfactual_reason": "ok",
                    "parent_without_fragment_smiles": "CC",
                    "direct_substructure": False,
                    "final_substructure": True,
                    "used_projected_subgraph_for_reward": True,
                    "parse_ok": True,
                    "valid": True,
                    "atom_ratio": 0.55,
                    "fragment_atom_count": 6,
                    "projection_score": 0.95,
                },
                {
                    "parent_smiles": "CCNCC",
                    "label": 1,
                    "reward_total": -0.5,
                    "counterfactual_called": False,
                    "counterfactual_reason": "invalid_or_not_substructure",
                    "direct_substructure": False,
                    "final_substructure": False,
                    "used_projected_subgraph_for_reward": False,
                    "parse_ok": False,
                    "valid": False,
                    "atom_ratio": 0.03,
                    "fragment_atom_count": 1,
                    "failure_tag": "parse_ok_but_core_unusable",
                    "invalid_detail": "core_unusable_after_dummy_removal",
                },
                {
                    "parent_smiles": "CCOCC",
                    "label": 1,
                    "reward_total": 1.8,
                    "cf_drop": 0.9,
                    "cf_flip": True,
                    "p_before": 0.92,
                    "p_after": 0.02,
                    "counterfactual_called": True,
                    "counterfactual_reason": "ok",
                    "parent_without_fragment_smiles": "C",
                    "direct_substructure": True,
                    "final_substructure": True,
                    "used_projected_subgraph_for_reward": False,
                    "parse_ok": True,
                    "valid": True,
                    "atom_ratio": 0.9,
                    "fragment_atom_count": 8,
                },
            ]
            with pool_path.open("w", encoding="utf-8") as handle:
                for row in pool_rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            out_dir = tmp_path / "audit_out"
            scorer = _FakeTeacherScorer(
                {
                    "CCOCC": {"teacher_prob": 0.95, "teacher_label": 1, "teacher_sem": 0.9},
                    "CCNCC": {"teacher_prob": 0.90, "teacher_label": 1, "teacher_sem": 0.8},
                }
            )
            summary = run_reward_teacher_audit(
                dataset_path=dataset_path,
                candidate_pool=pool_path,
                teacher_path="/tmp/fake_teacher.pkl",
                out_dir=out_dir,
                config=RewardTeacherAuditConfig(
                    label_col="label",
                    smiles_col="parent_smiles",
                    sim_sample_size=100,
                ),
                teacher_scorer=scorer,
            )

            oracle_summary = summary["candidate_pool_oracle_validity"]
            projection_summary = summary["projection_loophole_audit"]
            size_summary = summary["size_loophole_audit"]
            final_judgment = summary["final_judgment"]

            self.assertAlmostEqual(float(oracle_summary["cf_oracle_called_rate"]), 0.75, places=6)
            self.assertAlmostEqual(float(oracle_summary["cf_oracle_skipped_rate"]), 0.25, places=6)
            self.assertTrue(projection_summary["possible_projection_loophole"])
            self.assertTrue(size_summary["possible_size_loophole"])
            self.assertEqual(final_judgment["primary_diagnosis"], "reward_or_teacher_issue_more_likely")
            self.assertTrue((out_dir / "teacher_parent_reliability.json").exists())
            self.assertTrue((out_dir / "candidate_pool_oracle_validity.json").exists())
            self.assertTrue((out_dir / "reward_component_correlation.json").exists())
            self.assertTrue((out_dir / "projection_loophole_audit.json").exists())
            self.assertTrue((out_dir / "size_loophole_audit.json").exists())
            self.assertTrue((out_dir / "audit_summary.json").exists())
            self.assertTrue((out_dir / "audit_report.txt").exists())


if __name__ == "__main__":
    unittest.main()
