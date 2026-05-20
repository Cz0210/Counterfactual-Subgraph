from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.class_counterfactual_selector import (
    SelectorConfig,
    select_class_counterfactual_subgraphs,
)


class ClassCounterfactualSelectorTests(unittest.TestCase):
    def test_selector_filters_failures_and_prefers_diverse_shared_fragments(self) -> None:
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
                "oracle_ok": True,
                "cf_flip": True,
                "cf_drop": 0.7,
                "reward_total": 4.0,
                "atom_ratio": 0.30,
                "projection_used": False,
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
                "oracle_ok": True,
                "cf_flip": True,
                "cf_drop": 0.6,
                "reward_total": 3.8,
                "atom_ratio": 0.31,
                "projection_used": False,
            },
            {
                "parent_index": 2,
                "label": 1,
                "parent_smiles": "c1ccccc1O",
                "raw_fragment": "c1ccccc1",
                "core_fragment": "c1ccccc1",
                "final_fragment": "c1ccccc1",
                "parse_ok": True,
                "valid": True,
                "connected": True,
                "direct_substructure": True,
                "final_substructure": True,
                "oracle_ok": True,
                "cf_flip": True,
                "cf_drop": 0.55,
                "reward_total": 3.2,
                "atom_ratio": 0.42,
                "projection_used": False,
            },
            {
                "parent_index": 3,
                "label": 1,
                "parent_smiles": "CCCC",
                "raw_fragment": "C",
                "core_fragment": "C",
                "final_fragment": "C",
                "parse_ok": True,
                "valid": True,
                "connected": True,
                "direct_substructure": True,
                "final_substructure": True,
                "oracle_ok": True,
                "cf_flip": False,
                "cf_drop": 0.1,
                "reward_total": 1.0,
                "atom_ratio": 0.10,
                "projection_used": False,
            },
            {
                "parent_index": 4,
                "label": 1,
                "parent_smiles": "CCCl",
                "raw_fragment": "bad",
                "core_fragment": "",
                "final_fragment": "",
                "parse_ok": False,
                "valid": False,
                "connected": False,
                "direct_substructure": False,
                "final_substructure": False,
                "oracle_ok": False,
                "cf_flip": False,
                "cf_drop": -1.0,
                "reward_total": -5.0,
                "atom_ratio": 0.05,
                "failure_tag": "parse_failed",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "candidate_pool.jsonl"
            pool_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "selector"
            result = select_class_counterfactual_subgraphs(
                pool_path,
                out_dir=out_dir,
                config=SelectorConfig(
                    label=1,
                    top_k=2,
                    min_cf_drop=0.2,
                    require_cf_flip=True,
                    require_final_substructure=True,
                    dedup_by_final_fragment=True,
                ),
            )

            summary = result["summary"]
            selected_rows = result["selected_rows"]
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(summary["valid_candidate_count_after_filter"], 3)
            self.assertAlmostEqual(summary["final_cumulative_coverage"], 0.6)
            self.assertEqual(selected_rows[0]["fragment"], "CC")
            self.assertTrue((out_dir / "selected_subgraphs.json").exists())
            self.assertTrue((out_dir / "selected_subgraphs.csv").exists())
            self.assertTrue((out_dir / "selector_summary.json").exists())
            self.assertTrue((out_dir / "selector_report.txt").exists())


if __name__ == "__main__":
    unittest.main()
