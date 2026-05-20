from pathlib import Path
import tempfile
import unittest

from scripts.analyze_stable_ppo_log import (
    _extract_step_metrics,
    _extract_unified_sample_metrics,
    _summarize_ranges,
    _summarize_unified_label_ranges,
)


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

    def test_extracts_unified_sample_metrics_by_label(self) -> None:
        content = """2026-05-17 12:00:00 | INFO | train_ppo_stable | [UNIFIED_PPO_SAMPLE] step=1 sample_index=0 label=0 parent_smiles=CCC raw_fragment=CC final_fragment=CC parse_ok=True valid=True direct_substructure=True projection_used=False p_before=0.8 p_after=0.1 cf_drop=0.7 cf_flip=True reward_total=4.2
2026-05-17 12:00:00 | INFO | train_ppo_stable | [UNIFIED_PPO_SAMPLE] step=1 sample_index=1 label=1 parent_smiles=CCN raw_fragment=CN final_fragment=CN parse_ok=True valid=True direct_substructure=False projection_used=True p_before=0.9 p_after=0.4 cf_drop=0.5 cf_flip=False reward_total=2.5
2026-05-17 12:00:01 | INFO | train_ppo_stable | [STABLE_PPO_UPDATE] step=1 reward_mean=3.4 reward_min=1.0 reward_max=5.0 policy_loss=0.1 value_loss=0.2 total_loss=0.3 approx_kl=0.2 parse_ok_rate=0.9 valid_rate=0.8 direct_substructure_rate=0.7 final_substructure_rate=0.8 projection_used_rate=0.2 oracle_ok_rate=0.9 cf_flip_rate=0.85 cf_drop_mean=0.7 core_unusable_count=1 parse_failed_count=2 atom_ratio_mean=0.33
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "stable.err"
            log_path.write_text(content, encoding="utf-8")
            per_step = _extract_step_metrics(log_path)
            sample_rows = _extract_unified_sample_metrics(log_path)
            by_label, label_mix = _summarize_unified_label_ranges(sample_rows, per_step)

        self.assertEqual(len(sample_rows), 2)
        self.assertEqual(label_mix["1-50"]["label0_count"], 1)
        self.assertEqual(label_mix["1-50"]["label1_count"], 1)
        self.assertAlmostEqual(by_label["0"]["1-50"]["reward_mean"], 4.2)
        self.assertAlmostEqual(by_label["1"]["1-50"]["projection_used_rate"], 1.0)
        self.assertAlmostEqual(by_label["0"]["1-50"]["approx_kl_mean"], 0.2)
