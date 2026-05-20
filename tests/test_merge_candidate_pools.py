from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.candidate_pool_merge import MergeConfig, merge_candidate_pools


class MergeCandidatePoolsTests(unittest.TestCase):
    def test_merge_prefers_higher_reward_then_cf_drop(self) -> None:
        pool_a_rows = [
            {
                "parent_smiles": "CCO",
                "label": 1,
                "final_fragment": "CC",
                "reward_total": 3.0,
                "cf_drop": 0.4,
            },
            {
                "parent_smiles": "CCN",
                "label": 1,
                "final_fragment": "CN",
                "reward_total": 2.0,
                "cf_drop": 0.3,
            },
        ]
        pool_b_rows = [
            {
                "parent_smiles": "CCO",
                "label": 1,
                "final_fragment": "CC",
                "reward_total": 4.0,
                "cf_drop": 0.2,
            },
            {
                "parent_smiles": "CCN",
                "label": 1,
                "final_fragment": "CN",
                "reward_total": 2.0,
                "cf_drop": 0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pool_a = Path(tmpdir) / "pool_a.jsonl"
            pool_b = Path(tmpdir) / "pool_b.jsonl"
            out_jsonl = Path(tmpdir) / "merged.jsonl"
            pool_a.write_text(
                "".join(json.dumps(row) + "\n" for row in pool_a_rows),
                encoding="utf-8",
            )
            pool_b.write_text(
                "".join(json.dumps(row) + "\n" for row in pool_b_rows),
                encoding="utf-8",
            )

            summary = merge_candidate_pools(
                [pool_a, pool_b],
                out_jsonl=out_jsonl,
                config=MergeConfig(
                    dedup_key=("final_fragment", "parent_smiles"),
                    keep_best_by="reward_total",
                ),
            )

            merged_rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(summary["merged_count_before_dedup"], 4)
        self.assertEqual(summary["merged_count_after_dedup"], 2)
        self.assertEqual(summary["dedup_removed_count"], 2)
        self.assertEqual(len(merged_rows), 2)
        merged_by_parent = {row["parent_smiles"]: row for row in merged_rows}
        self.assertAlmostEqual(merged_by_parent["CCO"]["reward_total"], 4.0)
        self.assertAlmostEqual(merged_by_parent["CCN"]["cf_drop"], 0.5)


if __name__ == "__main__":
    unittest.main()
