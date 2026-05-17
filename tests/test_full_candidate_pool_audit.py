from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.eval.full_candidate_pool_audit import FullPoolAuditConfig, audit_full_candidate_pool


class FullCandidatePoolAuditTests(unittest.TestCase):
    def test_audit_full_candidate_pool_writes_selector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["parent_smiles", "label", "prompt"],
                )
                writer.writeheader()
                for smiles in ("CCO", "CCN", "CCC"):
                    writer.writerow(
                        {
                            "parent_smiles": smiles,
                            "label": "1",
                            "prompt": f"ORIGINAL_LABEL: 1\nMOLECULE_SMILES: {smiles}\nFRAGMENT_SMILES:",
                        }
                    )

            pool_path = Path(tmpdir) / "candidate_pool.jsonl"
            rows = [
                {
                    "parent_index": 0,
                    "label": 1,
                    "parent_smiles": "CCO",
                    "raw_fragment": "CC",
                    "core_fragment": "CC",
                    "final_fragment": "CC",
                    "parse_ok": True,
                    "valid": True,
                    "connected": True,
                    "direct_substructure": True,
                    "final_substructure": True,
                    "projection_used": False,
                    "projection_method": "direct_match",
                    "projection_attempted": False,
                    "projection_success": False,
                    "atom_count": 2,
                    "atom_ratio": 0.4,
                    "oracle_ok": True,
                    "counterfactual_called": True,
                    "counterfactual_reason": "ok",
                    "cf_flip": True,
                    "cf_drop": 0.6,
                    "p_before": 0.9,
                    "p_after": 0.3,
                    "reward_total": 4.0,
                },
                {
                    "parent_index": 1,
                    "label": 1,
                    "parent_smiles": "CCN",
                    "raw_fragment": "CC",
                    "core_fragment": "CC",
                    "final_fragment": "CC",
                    "parse_ok": True,
                    "valid": True,
                    "connected": True,
                    "direct_substructure": True,
                    "final_substructure": True,
                    "projection_used": False,
                    "projection_method": "direct_match",
                    "projection_attempted": False,
                    "projection_success": False,
                    "atom_count": 2,
                    "atom_ratio": 0.4,
                    "oracle_ok": True,
                    "counterfactual_called": True,
                    "counterfactual_reason": "ok",
                    "cf_flip": False,
                    "cf_drop": 0.2,
                    "p_before": 0.8,
                    "p_after": 0.6,
                    "reward_total": 2.0,
                },
                {
                    "parent_index": 2,
                    "label": 1,
                    "parent_smiles": "CCC",
                    "raw_fragment": "bad",
                    "core_fragment": "",
                    "final_fragment": "",
                    "parse_ok": False,
                    "valid": False,
                    "connected": False,
                    "direct_substructure": False,
                    "final_substructure": False,
                    "projection_used": False,
                    "projection_method": "none",
                    "projection_attempted": True,
                    "projection_success": False,
                    "failure_tag": "parse_failed",
                    "invalid_detail": "parse_failed",
                    "atom_count": 1,
                    "atom_ratio": 0.05,
                    "oracle_ok": False,
                    "counterfactual_called": False,
                    "counterfactual_reason": "invalid_or_not_substructure",
                    "cf_flip": False,
                    "reward_total": -6.0,
                },
            ]
            with pool_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            out_dir = Path(tmpdir) / "audit"
            summary = audit_full_candidate_pool(
                pool_jsonl=pool_path,
                dataset_path=dataset_path,
                teacher_path=Path(tmpdir) / "teacher.pkl",
                out_dir=out_dir,
                config=FullPoolAuditConfig(
                    label_col="label",
                    smiles_col="parent_smiles",
                    target_label=1,
                    sim_sample_size=10,
                    topk_show=5,
                ),
            )

            overall = summary["overall"]
            self.assertEqual(overall["num_rows"], 3)
            self.assertAlmostEqual(overall["cf_flip_rate"], 1.0 / 3.0)
            self.assertAlmostEqual(overall["projection_failed_rate"], 1.0 / 3.0)
            self.assertTrue((out_dir / "audit_summary.json").exists())
            self.assertTrue((out_dir / "audit_report.txt").exists())
            self.assertTrue((out_dir / "fragment_frequency_topk.csv").exists())
            self.assertTrue((out_dir / "parent_coverage_summary.json").exists())
            self.assertTrue((out_dir / "diversity_summary.json").exists())
            self.assertTrue((out_dir / "failure_cases.jsonl").exists())

            coverage_summary = json.loads(
                (out_dir / "parent_coverage_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(coverage_summary["top_fragments"][0]["final_fragment"], "CC")
            self.assertIn(
                coverage_summary["top_fragments"][0]["coverage_count"],
                {0, 3},
            )


if __name__ == "__main__":
    unittest.main()
