import pytest

from src.eval.counterfactual_semantics import (
    compute_counterfactual_semantics,
    destination_distribution,
)


def _record(after: int, rule_id: str):
    probabilities = {
        0: (0.70, 0.15, 0.15),
        1: (0.10, 0.80, 0.10),
        2: (0.15, 0.20, 0.65),
    }
    return compute_counterfactual_semantics(
        source_label=1,
        pred_before=1,
        pred_after=after,
        probabilities_before=(0.10, 0.80, 0.10),
        probabilities_after=probabilities[after],
        rule_id=rule_id,
    )


def test_destination_distribution_reports_both_taste_destinations_per_rule() -> None:
    summary = destination_distribution(
        [_record(0, "rule-a"), _record(2, "rule-a"), _record(2, "rule-b"), _record(1, "rule-b")],
        source_label=1,
        num_classes=3,
        label_map={0: "Bitter", 1: "Sweet", 2: "Tasteless"},
    )
    overall = summary["overall"]
    assert overall["total_strict_flips"] == 3
    assert overall["transitions"]["1->0"]["count"] == 1
    assert overall["transitions"]["1->0"]["rate"] == pytest.approx(1 / 3)
    assert overall["transitions"]["1->2"]["count"] == 2
    assert overall["transitions"]["1->2"]["rate"] == pytest.approx(2 / 3)
    assert summary["by_rule"]["rule-a"]["total_strict_flips"] == 2
    assert summary["by_rule"]["rule-b"]["total_strict_flips"] == 1


def test_destination_distribution_rejects_stale_flip_field() -> None:
    with pytest.raises(ValueError, match="cf_flip disagrees"):
        destination_distribution(
            [
                {
                    "source_label": 1,
                    "pred_before": 1,
                    "pred_after": 0,
                    "cf_flip": False,
                }
            ],
            source_label=1,
            num_classes=3,
        )
