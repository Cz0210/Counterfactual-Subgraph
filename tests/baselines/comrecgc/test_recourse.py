from __future__ import annotations

import numpy as np
import pytest

from src.baselines.comrecgc.recourse import (
    _importance_parts,
    choose_cluster_medoid,
    ordered_prefix,
    trace_official_cluster_order,
)


def test_cluster_medoid_is_real_pair_and_ties_keep_input_order() -> None:
    vectors = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    assert choose_cluster_medoid(vectors, [(5, 10), (6, 11)]) == (5, 10, 1.0)


def test_ordered_prefix_does_not_rerank() -> None:
    rows = [{"rank": 1, "candidate_id": "b"}, {"rank": 2, "candidate_id": "a"}]
    assert [row["candidate_id"] for row in ordered_prefix(rows, 2)] == ["b", "a"]
    with pytest.raises(ValueError):
        ordered_prefix([{"rank": 2}], 1)


def test_numpy_importance_parts_do_not_use_ambiguous_truth_value() -> None:
    assert _importance_parts(
        {"importance_parts": np.asarray([1.0, 0.25], dtype=np.float32)}
    ) == [1.0, 0.25]
    assert _importance_parts({"importance_parts": None}) == [0.0]


def _official_greedy(counterfactual_covering, graphs_covered_by, k):
    selected = {}
    covered = set()
    for rank in range(1, k + 1):
        cluster, gains = max(counterfactual_covering.items(), key=lambda item: len(item[1]))
        covered.update(gains)
        counterfactual_covering.pop(cluster)
        for parent in gains:
            for other in graphs_covered_by[parent] - {cluster}:
                if other in counterfactual_covering:
                    counterfactual_covering[other].discard(parent)
        selected[rank] = (cluster, len(covered))
    return selected


def test_trace_preserves_official_greedy_order_and_medoid() -> None:
    vectors = np.asarray([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
    labels = np.asarray([0, 0, 1])
    rows = trace_official_cluster_order(
        labels=labels,
        recourse_vectors=vectors,
        pair_indices=[(0, 4), (1, 5), (2, 6)],
        radius=1.0,
        theta=1.0,
        recourse_size=2,
        official_greedy=_official_greedy,
    )
    assert [row["cluster_label"] for row in rows] == [0, 1]
    assert rows[0]["representative_counterfactual_index"] == 4
