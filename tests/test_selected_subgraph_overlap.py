from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.selected_subgraph_overlap import compare_selected_subgraph_overlap


class SelectedSubgraphOverlapTests(unittest.TestCase):
    def test_compares_exact_and_soft_overlap(self) -> None:
        label0_rows = [
            {"fragment": "CC"},
            {"fragment": "c1ccccc1O"},
        ]
        label1_rows = [
            {"fragment": "CC"},
            {"fragment": "c1ccccc1N"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            label0_json = Path(tmpdir) / "label0.json"
            label1_json = Path(tmpdir) / "label1.json"
            out_dir = Path(tmpdir) / "overlap"
            label0_json.write_text(json.dumps(label0_rows), encoding="utf-8")
            label1_json.write_text(json.dumps(label1_rows), encoding="utf-8")
            summary = compare_selected_subgraph_overlap(
                label0_json,
                label1_json,
                out_dir=out_dir,
                sim_thresholds=[0.5, 0.7],
            )

            self.assertEqual(summary["exact_overlap"]["exact_intersection_count"], 1)
            self.assertGreater(summary["exact_overlap"]["exact_jaccard"], 0.0)
            self.assertIn("0.5", summary["soft_overlap"]["count_sim_ge"])
            self.assertTrue((out_dir / "overlap_summary.json").exists())
            self.assertTrue((out_dir / "overlap_report.txt").exists())
            self.assertTrue((out_dir / "overlap_pairs.csv").exists())


if __name__ == "__main__":
    unittest.main()
