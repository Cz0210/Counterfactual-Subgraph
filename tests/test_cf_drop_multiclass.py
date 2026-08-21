import pytest

from src.eval.counterfactual_semantics import (
    compute_counterfactual_semantics,
    counterfactual_drop,
    source_class_margin,
)


def test_multiclass_drop_and_margin_use_source_probability() -> None:
    before = (0.10, 0.75, 0.15)
    after = (0.55, 0.20, 0.25)

    assert counterfactual_drop(before, after, source_label=1) == pytest.approx(0.55)
    assert source_class_margin(before, source_label=1) == pytest.approx(0.60)
    assert source_class_margin(after, source_label=1) == pytest.approx(-0.35)

    record = compute_counterfactual_semantics(
        source_label=1,
        pred_before=1,
        pred_after=0,
        probabilities_before=before,
        probabilities_after=after,
    )
    assert record.cf_flip is True
    assert record.destination_label == 0
    assert record.cf_drop == pytest.approx(0.55)
    assert record.margin_drop == pytest.approx(0.95)


def test_probability_vectors_fail_closed() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        counterfactual_drop((0.2, 0.2, 0.2), (0.2, 0.3, 0.5), source_label=1)
    with pytest.raises(ValueError, match="expected 3"):
        counterfactual_drop((0.2, 0.3, 0.5), (0.4, 0.6), source_label=1)
