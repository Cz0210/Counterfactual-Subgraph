from src.eval.counterfactual_semantics import strict_flip


def test_binary_strict_flip_is_source_relative() -> None:
    assert strict_flip(pred_before=1, pred_after=0, source_label=1)
    assert strict_flip(pred_before=0, pred_after=1, source_label=0)
    assert not strict_flip(pred_before=1, pred_after=1, source_label=1)
    assert not strict_flip(pred_before=0, pred_after=0, source_label=1)
