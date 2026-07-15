from __future__ import annotations

import math

from scripts.compare_node_distance_ablation import (
    average_ranks,
    minimum_quantile_for_coverage,
    normalized_pauc,
)


def test_average_rank_uses_average_for_ties() -> None:
    ranks = average_ranks({"Ours": 0.7, "CLEAR": 0.5, "GCF": 0.5, "Global": 0.1})
    assert ranks == {"Ours": 1.0, "CLEAR": 2.5, "GCF": 2.5, "Global": 4.0}


def test_normalized_pauc_and_coverage_target_position() -> None:
    curve = {
        0.05: {"coverage": 0.1}, 0.10: {"coverage": 0.2},
        0.20: {"coverage": 0.4}, 0.30: {"coverage": 0.6},
    }
    assert 0.1 <= normalized_pauc(curve, 0.05, 0.30) <= 0.6
    assert minimum_quantile_for_coverage(curve, 0.5) == 0.30
    assert math.isnan(minimum_quantile_for_coverage(curve, 0.9))
