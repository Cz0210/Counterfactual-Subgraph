"""Calibration-only, fixed nested reach-first BACE rule selection.

The scalar MILP objective encodes the same lexicographic objective as the
bitset fallback. This is a new method version, not the old greedy theorem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReachMasks:
    candidate_ids: tuple[str, ...]
    parent_count: int
    reach: tuple[int, ...]
    high: tuple[int, ...]
    low: tuple[int, ...]
    threshold: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...]

    @classmethod
    def from_distances(cls, candidate_ids: Sequence[str], distances: Any,
                       thresholds: Mapping[str, Any]) -> "ReachMasks":
        d = np.asarray(distances, dtype=np.float64)
        if d.ndim != 2 or d.shape[1] != len(candidate_ids) or not d.shape[0]:
            raise ValueError("REACH_MATRIX_SHAPE_OR_EMPTY_CALIBRATION")
        if len(set(candidate_ids)) != len(candidate_ids) or np.isnan(d).any() or (d < 0).any():
            raise ValueError("REACH_DUPLICATE_IDS_OR_INVALID_DISTANCE")
        def bits(mask: np.ndarray) -> tuple[int, ...]:
            return tuple(sum(1 << int(p) for p in np.flatnonzero(mask[:, c])) for c in range(d.shape[1]))
        levels = thresholds["merged_thresholds"]
        weights = tuple(float(x["weight"]) for x in levels)
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("FROZEN_THRESHOLD_WEIGHTS_INVALID")
        return cls(tuple(candidate_ids), len(d), bits(np.isfinite(d)),
                   bits(d <= float(thresholds["cost_cap"])), bits(d <= float(thresholds["theta_star"])),
                   tuple(bits(d <= float(x["threshold"])) for x in levels), weights)

    @staticmethod
    def union(masks: Sequence[int], selected: Sequence[int]) -> int:
        result = 0
        for i in selected:
            result |= masks[i]
        return result

    def score(self, selected: Sequence[int]) -> tuple[int, int, float]:
        return (self.union(self.reach, selected).bit_count(),
                self.union(self.high, selected).bit_count(),
                sum(w * self.union(level, selected).bit_count() for w, level in zip(self.weights, self.threshold)))

    def low_count(self, selected: Sequence[int]) -> int:
        return self.union(self.low, selected).bit_count()

    def ordered(self, selected: Sequence[int], fixed_prefix: Sequence[int] = ()) -> list[int]:
        result = list(fixed_prefix)
        remaining = set(selected) - set(result)
        while remaining:
            chosen = min(remaining, key=lambda i: (tuple(-v for v in self.score([*result, i])), self.candidate_ids[i]))
            result.append(chosen)
            remaining.remove(chosen)
        return result


def improve_feasible(masks: ReachMasks, initial: Sequence[int], k: int, low_floor: int,
                     fixed: Sequence[int] = ()) -> list[int]:
    """Deterministic positive greedy additions then bounded exact-mask swaps."""
    selected = list(initial)
    if len(set(selected)) != len(selected) or len(selected) > k or masks.low_count(selected) < low_floor:
        raise ValueError("OLD_S10_NOT_FEASIBLE_CHECK_COHORT_BINDING")
    all_indices = set(range(len(masks.candidate_ids)))
    while len(selected) < min(k, len(all_indices)):
        candidate = min(all_indices - set(selected), key=lambda i: (
            tuple(-v for v in masks.score([*selected, i])), masks.candidate_ids[i]))
        selected.append(candidate)
    for _ in range(4):
        best = selected
        best_score = masks.score(best)
        best_key = tuple(sorted(masks.candidate_ids[i] for i in best))
        for outgoing in sorted(set(selected) - set(fixed), key=lambda i: masks.candidate_ids[i]):
            kept = [i for i in selected if i != outgoing]
            for incoming in sorted(all_indices - set(selected), key=lambda i: masks.candidate_ids[i]):
                trial = [*kept, incoming]
                if masks.low_count(trial) < low_floor:
                    continue
                score = masks.score(trial)
                key = tuple(sorted(masks.candidate_ids[i] for i in trial))
                if score > best_score or (score == best_score and key < best_key):
                    best, best_score, best_key = trial, score, key
        if set(best) == set(selected):
            break
        selected = best
    return selected


def milp_improve(masks: ReachMasks, selected: Sequence[int], k: int, low_floor: int,
                 fixed: Sequence[int], seconds: float) -> tuple[list[int], dict[str, Any]]:
    """Use already-installed free SciPy/HiGHS only, with a hard solver time limit."""
    if not 0 <= seconds <= 120:
        raise ValueError("SOLVER_TIME_LIMIT_EXCEEDS_AUTHORIZATION")
    report: dict[str, Any] = {"solver": "scipy.optimize.milp/HiGHS", "time_limit_seconds": seconds,
                              "globally_optimal": False, "fallback": "deterministic_greedy_swap"}
    if seconds == 0:
        return list(selected), {**report, "state": "DISABLED_EXPLICITLY", "gap": None}
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return list(selected), {**report, "state": "SOLVER_UNAVAILABLE", "gap": None}
    n, p = len(masks.candidate_ids), masks.parent_count
    levels = (masks.reach, masks.high, masks.low, *masks.threshold)
    count = n + len(levels) * p
    matrix = lil_matrix((1 + len(levels) * p + 1, count), dtype=np.float64)
    lower = np.full(matrix.shape[0], -np.inf)
    upper = np.full(matrix.shape[0], np.inf)
    matrix[0, :n] = 1
    upper[0] = k
    row = 1
    for li, level in enumerate(levels):
        for pi in range(p):
            matrix[row, n + li * p + pi] = 1
            for ci in range(n):
                if (level[ci] >> pi) & 1:
                    matrix[row, ci] = -1
            upper[row] = 0
            row += 1
    matrix[row, n + 2*p:n + 3*p] = 1
    lower[row] = low_floor
    f_bound = p * sum(masks.weights)
    h_coeff = f_bound + 1
    r_coeff = p * h_coeff + f_bound + 1
    objective = np.zeros(count)
    objective[n:n+p] = -r_coeff
    objective[n+p:n+2*p] = -h_coeff
    for li, weight in enumerate(masks.weights):
        objective[n + (3+li)*p:n + (4+li)*p] = -weight
    lb, ub = np.zeros(count), np.ones(count)
    for i in fixed:
        lb[i] = 1
    result = milp(objective, integrality=np.ones(count), bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": float(seconds), "mip_rel_gap": 0.0})
    report.update({"state": str(result.message), "status": int(result.status),
                   "gap": getattr(result, "mip_gap", None),
                   "scalar_objective_upper_bound": (-float(result.mip_dual_bound)
                        if getattr(result, "mip_dual_bound", None) is not None else None)})
    adopted = list(selected)
    if result.x is not None:
        trial = [i for i in range(n) if result.x[i] > .5]
        # Never trust the solver's coverage variables in lieu of actual masks.
        if len(trial) <= k and set(fixed) <= set(trial) and masks.low_count(trial) >= low_floor:
            if masks.score(trial) > masks.score(adopted):
                adopted = trial
            report["solver_mask_recomputed"] = True
        else:
            report["state"] = "SOLVER_SOLUTION_REJECTED_MASK_VALIDATION"
    if len(adopted) < min(k, n):
        adopted = improve_feasible(masks, adopted, k, low_floor, fixed)
    r, h, f = masks.score(adopted)
    report.update({"scalar_objective_lower_bound": r*r_coeff+h*h_coeff+f,
                   "feasible_objective_R_H_F_counts": [r, h, f],
                   "globally_optimal": bool(result.success and report.get("solver_mask_recomputed"))})
    return adopted, report


def select_nested(masks: ReachMasks, old_order: Sequence[str], *, solver_seconds: float = 120) -> dict[str, Any]:
    lookup = {key: i for i, key in enumerate(masks.candidate_ids)}
    if len(old_order) < 10 or len(set(old_order)) != len(old_order) or any(i not in lookup for i in old_order):
        raise ValueError("OLD_FROZEN_SEQUENCE_NOT_RETAINED")
    old10 = [lookup[i] for i in old_order[:10]]
    floor = masks.low_count(old10)
    ten = improve_feasible(masks, old10, 10, floor)
    ten, solve10 = milp_improve(masks, ten, 10, floor, (), solver_seconds)
    prefix = masks.ordered(ten)
    twenty = improve_feasible(masks, prefix, 20, floor, prefix)
    twenty, solve20 = milp_improve(masks, twenty, 20, floor, prefix, solver_seconds)
    ordered = masks.ordered(twenty, prefix)
    return {"selection_policy": "CALIBRATION_LEXICOGRAPHIC_R_H_F_WITH_OLD_S10_L_FLOOR_V2",
            "ordered_rule_ids": [masks.candidate_ids[i] for i in ordered],
            "prefixes": {str(k): [masks.candidate_ids[i] for i in ordered[:k]] for k in range(1, 21)},
            "old_S10_low_count": floor, "new_S10_low_count": masks.low_count(ordered[:10]),
            "S10_R_H_F_counts": list(masks.score(ordered[:10])), "S20_R_H_F_counts": list(masks.score(ordered)),
            "solver_K10": solve10, "solver_K20": solve20, "test_loaded": False,
            "within_set_order": "marginal_lexicographic_R_H_F_then_canonical_id",
            "global_nested_sequence_optimality_claimed": False}
