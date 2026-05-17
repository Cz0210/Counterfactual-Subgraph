import unittest

from src.rewards.reward_wrapper_stable import (
    StableTeacherConfidenceGateConfig,
    apply_teacher_confidence_gate_to_reward_logs,
)


class StableRewardWrapperTests(unittest.TestCase):
    def test_teacher_conf_gate_only_scales_counterfactual_component(self) -> None:
        logs, adjusted = apply_teacher_confidence_gate_to_reward_logs(
            [
                {
                    "p_before": 0.3,
                    "counterfactual_sem": 4.0,
                    "reward_total": 6.5,
                    "total": 6.5,
                }
            ],
            config=StableTeacherConfidenceGateConfig(
                enabled=True,
                min_teacher_p_before=0.5,
                low_conf_cf_weight=0.25,
            ),
            step_index=12,
        )

        self.assertEqual(len(logs), 1)
        self.assertAlmostEqual(adjusted[0], 3.5)
        self.assertTrue(logs[0]["stable_teacher_conf_gate_applied"])
        self.assertAlmostEqual(logs[0]["counterfactual_sem"], 1.0)
        self.assertAlmostEqual(logs[0]["reward_total"], 3.5)

    def test_teacher_conf_gate_keeps_reward_when_confidence_is_high(self) -> None:
        logs, adjusted = apply_teacher_confidence_gate_to_reward_logs(
            [
                {
                    "p_before": 0.9,
                    "counterfactual_sem": 4.0,
                    "reward_total": 6.5,
                }
            ],
            config=StableTeacherConfidenceGateConfig(
                enabled=True,
                min_teacher_p_before=0.5,
                low_conf_cf_weight=0.25,
            ),
            step_index=12,
        )

        self.assertAlmostEqual(adjusted[0], 6.5)
        self.assertFalse(logs[0]["stable_teacher_conf_gate_applied"])

