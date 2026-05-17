from pathlib import Path
import tempfile
import unittest

from scripts.analyze_stable_ppo_log import _extract_step_metrics, _summarize_ranges


class AnalyzeStablePPOLogTests(unittest.TestCase):
    def test_extracts_stable_update_and_gate_metrics(self) -> None:
        content = """2026-05-17 12:00:00 | INFO | train_ppo_stable | [STABLE_PPO_TEACHER_CONF_GATE] step=1 p_before=0.3 min_p_before=0.5 applied=True low_conf_cf_weight=0.3 reward_before_gate=4.0 reward_after_gate=2.8
2026-05-17 12:00:01 | INFO | train_ppo_stable | [STABLE_PPO_UPDATE] step=1 reward_mean=3.4 reward_min=1.0 reward_max=5.0 policy_loss=0.1 value_loss=0.2 total_loss=0.3 approx_kl=0.2 parse_ok_rate=0.9 valid_rate=0.8 direct_substructure_rate=0.7 final_substructure_rate=0.8 projection_used_rate=0.2 oracle_ok_rate=0.9 cf_flip_rate=0.85 cf_drop_mean=0.7 core_unusable_count=1 parse_failed_count=2 atom_ratio_mean=0.33
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "stable.err"
            log_path.write_text(content, encoding="utf-8")
            per_step = _extract_step_metrics(log_path)
            ranges = _summarize_ranges(per_step)

        self.assertIn(1, per_step)
        self.assertEqual(per_step[1]["teacher_conf_gate_applied_count"], 1)
        self.assertAlmostEqual(per_step[1]["approx_kl"], 0.2)
        self.assertAlmostEqual(ranges["1-50"]["reward_mean"], 3.4)
