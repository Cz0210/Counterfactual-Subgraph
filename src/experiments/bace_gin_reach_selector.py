"""Predeclared A+ reach-aware selection; no test or candidate generation."""
from __future__ import annotations

import math
from typing import Sequence

from src.eval.bace_reach_selector import ReachMasks

POLICY = "GIN_A_PLUS_025_REACH_075_NORMALIZED_GRID_GREEDY_1SWAP"


def objective(masks: ReachMasks, chosen: Sequence[int]) -> float:
    total = sum(masks.weights)
    if not total or any(not math.isfinite(x) or x <= 0 for x in masks.weights):
        raise ValueError("INVALID_ORIGINAL_GRID_WEIGHTS")
    r = masks.union(masks.reach, chosen).bit_count()
    grid = sum(w * masks.union(level, chosen).bit_count()
               for w, level in zip(masks.weights, masks.threshold)) / total
    return (.25 * r + .75 * grid) / masks.parent_count


def select(masks: ReachMasks, old_order: Sequence[str], *, swap_passes=2):
    """Old S10 supplies a feasible floor; preserve S10 when extending to S20.

    The objective and two deterministic 1-swap passes are sealed before new
    calibration outcomes. This is not the old lexicographic Reach-v2 selector.
    No optimality theorem or test-selected solver configuration is claimed.
    """
    if swap_passes != 2:
        raise ValueError("PREDECLARED_SWAP_PASSES_CHANGED")
    lookup = {v: i for i, v in enumerate(masks.candidate_ids)}
    if not old_order or len(set(old_order)) != len(old_order) or any(x not in lookup for x in old_order):
        raise ValueError("OLD_CALIBRATION_ORDER_NOT_RETAINED")
    old10 = [lookup[x] for x in old_order[:10]]
    floor = masks.low_count(old10)
    available = sorted(lookup.values(), key=lambda i: masks.candidate_ids[i])

    def best_add(chosen, candidates):
        return min(candidates, key=lambda i: (-objective(masks, [*chosen, i]), masks.candidate_ids[i]))

    def improve(initial, k, fixed=()):
        chosen = list(initial)
        while len(chosen) < min(k, len(available)):
            chosen.append(best_add(chosen, [i for i in available if i not in chosen]))
        for _ in range(swap_passes):
            score = objective(masks, chosen)
            best = None
            for outgoing in [i for i in chosen if i not in fixed]:
                kept = [i for i in chosen if i != outgoing]
                for incoming in available:
                    if incoming in chosen:
                        continue
                    trial = [*kept, incoming]
                    if masks.low_count(trial) < floor:
                        continue
                    value = objective(masks, trial)
                    key = tuple(sorted(masks.candidate_ids[i] for i in trial))
                    if value > score and (best is None or (-value, key) < best[0]):
                        best = ((-value, key), trial)
            if best is None:
                break
            chosen = best[1]
        return chosen

    def order(chosen, prefix=()):
        result = list(prefix)
        while len(result) < len(chosen):
            result.append(best_add(result, [i for i in chosen if i not in result]))
        return result

    ten = order(improve(old10, 10))
    twenty = order(improve(ten, 20, ten), ten)
    assert masks.low_count(ten) >= floor
    return {"policy": POLICY, "ordered_rule_ids": [masks.candidate_ids[i] for i in twenty],
            "old_S10_theta_count": floor, "new_S10_theta_count": masks.low_count(ten),
            "S10_objective": objective(masks, ten), "S20_objective": objective(masks, twenty),
            "reach_weight": .25, "grid_weight": .75, "threshold_weights_normalized": True,
            "swap_passes": swap_passes, "global_optimality_claimed": False,
            "solver": "deterministic_greedy_then_1swap", "test_loaded": False}
