import random
import pytest
from src.eval.ecdf_display import Point, staircase, simplify


def test_zero_is_not_epsilon_and_zero_plateau_remains():
    points = staircase([(0, 0), (1, 0), (2, .25), (3, .25)], [1.5])
    display, audit = simplify(points, keys=[1.5])
    assert Point(1.5, 0) in display and Point(2, 0) in display
    assert Point(2, .25) in display
    assert audit['maximum_absolute_error'] <= .005


def test_large_jumps_and_keys_exact():
    points = staircase([(0, 0), (.01, .8), (.04, .85)], [.02])
    display, audit = simplify(points, keys=[.02])
    assert Point(.01, 0) in display and Point(.01, .8) in display
    assert Point(.02, .8) in display and display[-1].y == .85
    assert audit['display_used_for_metrics_auc_or_ranking'] is False


def test_small_jumps_bounded_at_both_limits_and_intervals():
    rng = random.Random(7)
    xs = sorted(rng.random() for _ in range(1000))
    points = staircase([(0, 0)] + [(x, (i + 1) / 1200) for i, x in enumerate(xs)] + [(1, 1000 / 1200)], [.2])
    display, audit = simplify(points, keys=[.2])
    assert len(display) < len(points)
    assert audit['maximum_absolute_error'] <= .005
    assert display[0] == points[0] and display[-1] == points[-1]
    # Independent bound check at every original vertex. Include both one-sided
    # limits at duplicate-x discontinuities, not merely threshold right values.
    for a, b in zip(display, display[1:]):
        if a.x == b.x:
            continue
        for p in points:
            if a.x < p.x < b.x:
                value = a.y + (b.y - a.y) * (p.x - a.x) / (b.x - a.x)
                assert abs(value - p.y) <= .005 + 1e-15


def test_identical_policy_is_method_independent():
    points = staircase([(0, 0), (.1, .002), (.2, .004), (.3, .9)])
    assert simplify(points) == simplify(points)


@pytest.mark.parametrize('rows', [[], [(0, .1), (0, .2)], [(0, .2), (1, .1)], [(0, float('nan'))]])
def test_invalid_exact_input_rejected(rows):
    with pytest.raises(ValueError):
        staircase(rows)


def test_key_outside_range_rejected():
    with pytest.raises(ValueError):
        staircase([(0, 0), (1, .8)], [2])


def test_error_cannot_be_relaxed():
    with pytest.raises(ValueError):
        simplify([Point(0, 0), Point(1, 1)], max_error=.01)
