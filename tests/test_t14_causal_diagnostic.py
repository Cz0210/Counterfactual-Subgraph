import random
from types import SimpleNamespace

from src.baselines.t14_causal_diagnostic import (
    TOTAL_TRANSITION_CAP, START_STEP, END_STEP, SamplingObserver,
    first_difference, semantic,
)


def test_total_170_is_not_64_plus_170():
    assert 2 * (END_STEP - START_STEP) == TOTAL_TRANSITION_CAP == 170


def test_first_difference_values_not_pickle_bytes():
    assert first_difference({"a": [1, 2]}, {"a": [1, 3]})["path"] == "$.a[1]"
    assert first_difference(semantic({3, 2}), semantic({2, 3})) is None


def test_scalar_storage_wrappers_not_scientific_differences():
    import numpy as np
    assert semantic(np.int64(7)) == semantic(7)
    assert semantic(np.float64(0.25)) == semantic(0.25)


def test_observer_does_not_change_rng_or_selected_actions():
    def choose(hashes, importances, importance_args):
        probabilities = [row[0] for row in importances]
        importance_values = probabilities
        return random.choices(range(len(hashes)), weights=probabilities)[0]
    module = SimpleNamespace(move_from_known_graph=choose, graph_index_map={}, counterfactual_candidates=[])
    random.seed(7)
    initial = random.getstate()
    expected = [random.uniform(0, 1), choose(["a", "b"], [[0.4], [0.6]], {}), random.sample(range(100), 5)]
    final = random.getstate()
    random.setstate(initial)
    with SamplingObserver(module) as observer:
        actual = [random.uniform(0, 1), choose(["a", "b"], [[0.4], [0.6]], {}), random.sample(range(100), 5)]
    assert actual == expected
    assert random.getstate() == final
    native = next(row for row in observer.events if row["api"] == "move_from_known_graph.return")
    assert native["candidate_order"] == ["a", "b"]
    assert native["actual_probabilities"] == [0.4, 0.6]
    assert any(row["api"] == "Random.random" and "u" in row for row in observer.events)


def test_observer_restores_rng_method_on_failure():
    def choose():
        return 0
    module = SimpleNamespace(move_from_known_graph=choose)
    original = random._inst.random
    try:
        with SamplingObserver(module):
            raise RuntimeError("expected")
    except RuntimeError:
        pass
    assert random._inst.random == original
