"""Small synthetic numeric fixtures; never model/OT/scientific PASS tests."""
import json

import numpy as np
import pytest

from src.baselines.cm_crem_selection import (
    PrefixEvaluation, SelectionFreeze, evaluate_frozen_test, select_calibration,
)

CONTRACT = "a" * 64
POOL = "b" * 64


def select(matrix, *, candidates=None, mask=None, theta=.2, cap=1., statuses=None):
    matrix = np.asarray(matrix, dtype=float)
    n, m = matrix.shape
    mask = np.ones(n, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    if statuses is None:
        statuses = np.where(np.isfinite(matrix), "OK", "NO_STRICT_FLIP")
        statuses[~mask] = "NON_SOURCE"
    return select_calibration(matrix, pair_status=statuses,
        parent_ids=[f"cal-{i}" for i in range(n)],
        candidate_ids=candidates if candidates is not None else [f"candidate-{i:04d}" for i in range(m)],
        source_mask=mask, theta=theta, cap=cap, contract_sha256=CONTRACT, frozen_pool_sha256=POOL)


def evaluate(freeze, matrix, *, mask=None, statuses=None, candidates=None, contract=CONTRACT):
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    mask = np.ones(n, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    if statuses is None:
        statuses = np.where(np.isfinite(matrix), "OK", "NO_STRICT_FLIP")
        statuses[~mask] = "NON_SOURCE"
    return evaluate_frozen_test(freeze, matrix, pair_status=statuses,
        parent_ids=[f"test-{i}" for i in range(n)],
        candidate_ids=(freeze.selected_candidate_ids if isinstance(freeze, SelectionFreeze)
                       else freeze["selected_candidate_ids"]) if candidates is None else candidates,
        source_mask=mask, contract_sha256=contract)


def test_coverage_first_even_if_other_candidate_reduces_cost_more():
    frozen = select([[.1, .21], [1., .21], [1., .21]], candidates=["coverage", "cost"])
    assert frozen.selected_candidate_ids[0] == "coverage"
    assert frozen.steps[0].marginal_covered_count == 1
    assert frozen.steps[0].marginal_capped_mean_decrease == pytest.approx(.3)


def test_equal_coverage_uses_cost_then_canonical_id_not_column_order():
    frozen = select([[.1, .1, .1], [.8, .3, .3]], candidates=["z", "b", "a"])
    assert frozen.selected_candidate_ids == ("a", "b", "z")
    # After a, both remaining candidates have zero gain and zero cost gain.
    # Canonical b wins, not the original column z.


def test_zero_gain_continues_to_min_20_with_no_padding():
    frozen = select(np.full((2, 25), np.inf), candidates=[f"c{i:02d}" for i in reversed(range(25))])
    assert frozen.selected_candidate_ids == tuple(f"c{i:02d}" for i in range(20))
    assert all(s.marginal_covered_count == 0 and s.marginal_capped_mean_decrease == 0 for s in frozen.steps)
    assert len(frozen.steps) == 20


def test_fixed_denominator_and_non_source_never_becomes_covered():
    frozen = select([[.1], [np.inf]], mask=[True, False], cap=.5)
    assert frozen.steps[0].marginal_capped_mean_decrease == pytest.approx(.2)
    result = evaluate(frozen, [[.1], [np.inf]], mask=[True, False])
    row = result.prefix_metrics()[0]
    assert row["base_parent_count"] == 2 and row["source_parent_count"] == 1
    assert row["coverage"] == .5 and row["fixed_capped_mean_cost"] == pytest.approx(.3)


def test_empty_pool_is_valid_zero_cap_and_undefined_conditional_median():
    frozen = select(np.empty((3, 0)))
    result = evaluate(frozen, np.empty((4, 0)))
    assert frozen.selected_candidate_ids == ()
    assert len(result.prefix_metrics()) == 20
    assert all(r["effective_k"] == 0 and r["coverage"] == 0 and r["cost"] == 1.
               and r["conditional_median_cost"] is None for r in result.prefix_metrics())
    assert result.exact_ecdf(20) == [{"k": 20, "effective_k": 0, "distance": 0., "covered_count": 0,
        "coverage": 0., "base_parent_count": 4, "finite_recourse_count": 0, "unresolved_count": 4}]
    assert len(result.parent_best_rows()) == 80


def test_at_most_k_plateau_and_uncapped_median_ecdf():
    frozen = select([[.1]], cap=.5)
    result = evaluate(frozen, [[2.], [2.], [.1], [np.inf]])
    rows = result.prefix_metrics()
    assert all(r["effective_k"] == 1 for r in rows)
    assert all(r["cost"] == pytest.approx(.4) and r["conditional_median_cost"] == 2. for r in rows)
    assert result.best_distances[19, 0] == 2.  # Never replace uncapped distance with cap.
    ecdf = result.exact_ecdf(10)
    assert [(r["distance"], r["coverage"]) for r in ecdf] == [(0., 0.), (.1, .25), (2., .75)]
    assert ecdf[-1]["unresolved_count"] == 1


def test_exact_threshold_and_zero_distance():
    frozen = select([[.2]], theta=.2)
    result = evaluate(frozen, [[.2], [np.nextafter(.2, np.inf)], [0.]])
    assert result.prefix_metrics()[0]["covered_count"] == 2
    assert result.exact_ecdf(10)[0]["coverage"] == 1/3
    assert len(result.exact_ecdf(10)) == 3


@pytest.mark.parametrize("bad", [np.nan, -np.inf, -.01])
def test_numeric_errors_never_become_semantic_infinity(bad):
    with pytest.raises(ValueError):
        select([[bad]])


@pytest.mark.parametrize("status", ["ERROR", "TIMEOUT", "PENDING", "UNCOMPUTED", "OK"])
def test_infinity_requires_explicit_semantic_status(status):
    with pytest.raises(ValueError):
        select([[np.inf]], statuses=[[status]])


def test_source_mask_inconsistency_rejected_before_any_imputation():
    with pytest.raises(ValueError):
        select([[.1]], mask=[False])
    with pytest.raises(ValueError):
        select([[np.nan]], mask=[False])
    with pytest.raises(ValueError):
        select([[np.inf]], statuses=[["NON_SOURCE"]], mask=[True])
    with pytest.raises(ValueError):
        select([[np.inf]], statuses=[["INVALID"]], mask=[False])


def test_pool_and_contract_binding_and_json_roundtrip():
    frozen = select([[.1, .2]], candidates=["z", "a"])
    assert SelectionFreeze.from_dict(json.loads(json.dumps(frozen.to_dict()))) == frozen
    assert frozen.contract_sha256 == CONTRACT and frozen.frozen_pool_sha256 == POOL
    raw = frozen.to_dict()
    raw["theta"] = .9
    with pytest.raises(ValueError, match="hash mismatch"):
        SelectionFreeze.from_dict(raw)
    with pytest.raises(ValueError, match="Test contract"):
        evaluate(frozen, [[.1, .2]], contract="c"*64)


def test_test_rejects_reordered_or_unselected_candidate_columns():
    frozen = select(np.zeros((1, 25)))
    with pytest.raises(ValueError, match="exactly the ordered"):
        evaluate(frozen, np.zeros((2, 20)), candidates=list(reversed(frozen.selected_candidate_ids)))
    with pytest.raises(ValueError, match="exactly the ordered"):
        evaluate(frozen, np.zeros((2, 25)), candidates=frozen.pool_candidate_ids)


def test_oversized_or_duplicate_train_pool_rejected():
    with pytest.raises(ValueError, match="<=2000"):
        select(np.empty((1, 2001)))
    with pytest.raises(ValueError, match="unique"):
        select([[.1, .2]], candidates=["same", "same"])


def test_scalar_reference_global_greedy_and_column_permutation():
    rng = np.random.default_rng(7)
    matrix = np.round(rng.uniform(0, 2, size=(9, 23)), 2)
    matrix[rng.random(matrix.shape) < .2] = np.inf
    ids = [f"c-{i:02d}" for i in reversed(range(matrix.shape[1]))]
    actual = select(matrix, candidates=ids, theta=.35, cap=.8)
    remaining, chosen, best = list(range(len(ids))), [], [np.inf]*len(matrix)
    for _ in range(20):
        def key(j):
            updates = [min(best[i], matrix[i, j]) for i in range(len(matrix))]
            gain = sum(value <= .35 and best[i] > .35 for i, value in enumerate(updates))
            decrease = sum(min(best[i], .8)-min(value, .8) for i, value in enumerate(updates))/len(matrix)
            return -gain, -decrease, ids[j]
        winner = min(remaining, key=key)
        remaining.remove(winner)
        chosen.append(ids[winner])
        best = [min(best[i], matrix[i, winner]) for i in range(len(matrix))]
    assert actual.selected_candidate_ids == tuple(chosen)
    order = rng.permutation(len(ids))
    permuted = select(matrix[:, order], candidates=[ids[i] for i in order], theta=.35, cap=.8)
    assert permuted.selected_candidate_ids == actual.selected_candidate_ids


def test_test_evaluation_does_not_call_selector(monkeypatch):
    import src.baselines.cm_crem_selection as module
    frozen = select([[.1]])
    monkeypatch.setattr(module, "select_calibration", lambda *args, **kwargs: pytest.fail("Test must not select"))
    assert evaluate(frozen, [[.2]]).prefix_metrics()[0]["coverage"] == 1.


def test_evaluation_strict_json_roundtrip_and_driver_envelope():
    frozen = select([[.1]])
    result = evaluate({"science_hash": CONTRACT, **frozen.to_dict()}, [[.2], [np.inf]])
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["best_distances_uncapped"][0] == [.2, "inf"]
    reopened = PrefixEvaluation.from_dict({"science_hash": CONTRACT, **payload})
    assert reopened.prefix_metrics() == result.prefix_metrics()
    assert reopened.parent_best_rows() == result.parent_best_rows()
    payload["best_distances_uncapped"][0][0] = .1
    with pytest.raises(ValueError, match="hash mismatch"):
        PrefixEvaluation.from_dict(payload)


def test_driver_envelope_must_not_override_frozen_science_contract():
    frozen = select([[.1]])
    with pytest.raises(ValueError, match="science envelope"):
        SelectionFreeze.from_dict({"science_hash": "c"*64, **frozen.to_dict()})
