import json
import tempfile
import unittest
from pathlib import Path

from src.eval.candidate_pool_audit import AuditConfig, audit_candidate_pool, render_audit_report


class CandidatePoolAuditTests(unittest.TestCase):
    def test_audit_candidate_pool_handles_projection_and_failure_compat_fields(self) -> None:
        rows = [
            {
                "parent_smiles": "CCOCC",
                "original_label": 1,
                "raw_fragment": "CO",
                "core_fragment": "CO",
                "direct_substructure": True,
                "final_substructure": True,
                "parse_ok": True,
                "valid": True,
                "connected_ok": True,
                "oracle_ok": True,
                "cf_drop": 0.2,
                "cf_flip": False,
                "p_before": 0.8,
                "p_after": 0.6,
                "atom_count": 2,
                "atom_ratio": 0.4,
            },
            {
                "parent_smiles": "CCOCC",
                "label": 1,
                "fragment": "C1COCCN1",
                "core_fragment_smiles": "C1COCCN1",
                "projected_fragment": "CO",
                "projection_attempted": True,
                "projection_success": True,
                "used_projected_subgraph_for_reward": True,
                "final_substructure": True,
                "parse_ok": True,
                "valid": True,
                "connected_ok": True,
                "oracle_ok": True,
                "counterfactual_drop": 0.8,
                "counterfactual_flip": True,
                "teacher_p_before": 0.9,
                "teacher_p_after": 0.1,
                "projection_penalty_applied": 1.0,
            },
            {
                "parent_smiles": "CCN",
                "original_label": 0,
                "raw_fragment": "C?C",
                "parse_ok": False,
                "valid": False,
                "connected_ok": False,
                "failure_tag": "parse_ok_but_core_unusable",
                "invalid_detail": "core_unusable_after_dummy_removal",
                "tiny_fragment_hard_fail": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "candidate_pool.jsonl"
            with pool_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            summary = audit_candidate_pool(
                pool_path,
                config=AuditConfig(group_by_label=True, sim_sample_size=100, topk_show=5),
            )

        self.assertEqual(summary["overall"]["num_total"], 3)
        self.assertEqual(summary["overall"]["num_unique_parent"], 2)
        self.assertAlmostEqual(summary["overall"]["projection_used_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["overall"]["projection_retrieval_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["overall"]["final_substructure_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["overall"]["core_unusable_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["overall"]["too_small_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["overall"]["cf_flip_rate"], 1.0 / 3.0)
        self.assertEqual(summary["overall"]["unique_final_fragment_count"], 2)
        self.assertAlmostEqual(summary["overall"]["top1_final_fragment_ratio"], 2.0 / 3.0)
        self.assertIn("0", summary["by_label"])
        self.assertIn("1", summary["by_label"])

    def test_render_audit_report_mentions_selector_readiness(self) -> None:
        summary = {
            "metadata": {
                "pool_jsonl": "/tmp/pool.jsonl",
                "generated_at_utc": "2026-05-14T00:00:00+00:00",
            },
            "overall": {
                "num_total": 2,
                "num_by_label": {"1": 2},
                "num_unique_parent": 2,
                "avg_candidates_per_parent": 1.0,
                "valid_rate": 1.0,
                "parse_ok_rate": 1.0,
                "connected_rate": 1.0,
                "direct_substructure_rate": 0.5,
                "final_substructure_rate": 1.0,
                "projection_used_rate": 0.0,
                "projection_identity_rate": 0.0,
                "projection_retrieval_rate": 0.0,
                "projection_failed_rate": 0.0,
                "oracle_ok_rate": 1.0,
                "cf_flip_rate": 0.5,
                "cf_drop_mean": 0.8,
                "cf_drop_median": 0.8,
                "atom_ratio_mean": 0.3,
                "atom_ratio_median": 0.3,
                "unique_final_fragment_rate": 1.0,
                "top5_final_fragment_ratio": 0.5,
                "mean_pairwise_tanimoto": 0.4,
                "median_pairwise_tanimoto": 0.4,
                "top_final_fragments": [{"fragment": "CO", "count": 1, "ratio": 0.5}],
                "atom_ratio_histogram": {
                    "0-0.05": {"count": 0, "rate": 0.0},
                    "0.05-0.1": {"count": 0, "rate": 0.0},
                    "0.1-0.2": {"count": 0, "rate": 0.0},
                    "0.2-0.4": {"count": 2, "rate": 1.0},
                    "0.4-0.6": {"count": 0, "rate": 0.0},
                    "0.6-0.8": {"count": 0, "rate": 0.0},
                    "0.8-1.0": {"count": 0, "rate": 0.0},
                    "full-parent": {"count": 0, "rate": 0.0},
                },
                "atom_ratio_missing_count": 0,
            },
            "judgment": {
                "recommend_start_selector": True,
                "mode_collapse_risk": False,
                "projection_dependency_high": False,
                "strong_cf_but_low_diversity": False,
                "atom_ratio_out_of_range": False,
                "recommend_continue_long_ppo": False,
                "recommend_sampling_tuning": False,
                "heuristic_checks": {
                    "final_substructure_rate>=0.9": True,
                    "cf_flip_rate>=0.5": True,
                },
            },
            "by_label": {},
        }

        report = render_audit_report(summary)

        self.assertIn("suitable_for_selector: yes", report)
        self.assertIn("recommend_continue_long_ppo: no", report)


if __name__ == "__main__":
    unittest.main()
