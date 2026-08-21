from src.eval.counterfactual_semantics import multiclass_flip


def test_sweet_to_either_destination_is_a_strict_flip() -> None:
    assert multiclass_flip(pred_before=1, pred_after=0, source_label=1)
    assert multiclass_flip(pred_before=1, pred_after=2, source_label=1)
    assert not multiclass_flip(pred_before=1, pred_after=1, source_label=1)


def test_prediction_must_begin_in_source_class() -> None:
    assert not multiclass_flip(pred_before=0, pred_after=2, source_label=1)
