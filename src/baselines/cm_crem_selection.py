"""CM-CReM calibration-only global summary over already frozen train prototypes.

No oracle, encoder, OT, generation, or test-time selection is performed here.
Positive infinity is a *semantic* value and always requires an explicit status;
incomplete or failed distance computations are rejected, never imputed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

METHOD_ID = "CM-CReM-Global-Budgeted-v1"
PAPER_LABEL = "CM-CReM-Global"
K_MAX = 20
POOL_MAX = 2000
FINITE_STATUSES = frozenset({"OK", "FINITE"})
NON_SOURCE_STATUSES = frozenset({"NON_SOURCE", "BEFORE_NOT_SOURCE"})
SEMANTIC_STATUSES = NON_SOURCE_STATUSES | {"INVALID", "SEMANTIC_INVALID", "NO_STRICT_FLIP"}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(x, str) or not x or x.strip() != x for x in result):
        raise ValueError(f"{name} must contain nonempty canonical string IDs")
    if len(set(result)) != len(result) or (not result and not allow_empty):
        raise ValueError(f"{name} must be unique and nonempty")
    return result


def _thresholds(theta: float, cap: float) -> tuple[float, float]:
    if isinstance(theta, bool) or isinstance(cap, bool):
        raise ValueError("Boolean thresholds are invalid")
    theta, cap = float(theta), float(cap)
    if not math.isfinite(theta) or not math.isfinite(cap) or theta < 0 or cap <= 0:
        raise ValueError("theta must be finite/nonnegative; cap finite/positive")
    return theta, cap


def _matrix(distances: Any, pair_status: Any, parent_ids: Sequence[str],
            candidate_ids: Sequence[str], source_mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(distances, dtype=np.float64)
    statuses = np.asarray(pair_status)
    mask = np.asarray(source_mask)
    shape = (len(parent_ids), len(candidate_ids))
    if matrix.shape != shape or statuses.shape != shape:
        raise ValueError(f"Full pair matrix and status matrix must both have shape {shape}")
    if mask.shape != (shape[0],) or mask.dtype != np.dtype(bool):
        raise ValueError("source_mask must be an aligned Boolean vector; never prefilter parents")
    if np.isnan(matrix).any() or np.isneginf(matrix).any() or (matrix < 0).any():
        raise ValueError("NaN, negative infinity, and negative distances are not scientific outcomes")
    finite = np.isfinite(matrix)
    semantic = np.isin(statuses, tuple(SEMANTIC_STATUSES))
    valid_finite = np.isin(statuses, tuple(FINITE_STATUSES))
    if np.any(finite & ~valid_finite) or np.any(~finite & ~semantic):
        raise ValueError("Finite pairs require OK/FINITE; infinity requires explicit semantic invalidity")
    non_source = np.isin(statuses, tuple(NON_SOURCE_STATUSES))
    if np.any(non_source[mask]) or np.any(~non_source[~mask]) or np.any(finite[~mask]):
        raise ValueError("Non-source status must agree exactly with the frozen source mask")
    # Take private, read-only copies, including statuses: callers cannot mutate a
    # validated matrix during selection/evaluation through an aliased buffer.
    matrix, statuses, mask = matrix.copy(), statuses.astype(str, copy=True), mask.copy()
    for value in (matrix, statuses, mask):
        value.flags.writeable = False
    return matrix, statuses, mask


def _matrix_sha(matrix: np.ndarray, statuses: np.ndarray) -> str:
    return canonical_sha256({"shape": list(matrix.shape), "dtype": "float64-le",
        "values_sha256": hashlib.sha256(matrix.astype("<f8").tobytes(order="C")).hexdigest(),
        "statuses": statuses.tolist()})


@dataclass(frozen=True)
class SelectionStep:
    k: int
    candidate_id: str
    marginal_covered_count: int
    marginal_capped_mean_decrease: float
    covered_count: int
    fixed_capped_mean_cost: float


@dataclass(frozen=True)
class SelectionFreeze:
    schema_version: str
    method_id: str
    contract_sha256: str
    frozen_pool_sha256: str
    pool_candidate_ids: tuple[str, ...]
    calibration_parent_ids: tuple[str, ...]
    calibration_source_mask: tuple[bool, ...]
    calibration_matrix_sha256: str
    theta: float
    cap: float
    selected_candidate_ids: tuple[str, ...]
    steps: tuple[SelectionStep, ...]
    freeze_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Portable strict JSON; no model tensors or infinity-valued metrics."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectionFreeze":
        value = dict(raw)
        if value.pop("science_hash", value.get("contract_sha256")) != value.get("contract_sha256"):
            raise ValueError("Driver science envelope differs from the frozen contract")
        for name in ("pool_candidate_ids", "calibration_parent_ids", "calibration_source_mask",
                     "selected_candidate_ids"):
            value[name] = tuple(value[name])
        value["steps"] = tuple(SelectionStep(**row) for row in value["steps"])
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        raw = self.to_dict()
        claimed = raw.pop("freeze_sha256")
        if _sha(claimed, "freeze_sha256") != canonical_sha256(raw):
            raise ValueError("Selection freeze content hash mismatch")
        v2 = self.method_id == "CM-Global-K20-v2" and self.schema_version == "cm_crem_selection_freeze_k20_v2"
        if not v2 and (self.schema_version != "cm_crem_selection_freeze_v1" or self.method_id != METHOD_ID):
            raise ValueError("Unknown CM-CReM selection contract")
        for name in ("contract_sha256", "frozen_pool_sha256", "calibration_matrix_sha256"):
            _sha(getattr(self, name), name)
        _thresholds(self.theta, self.cap)
        pool = _ids(self.pool_candidate_ids, "pool_candidate_ids", allow_empty=True)
        _ids(self.calibration_parent_ids, "calibration_parent_ids")
        chosen = _ids(self.selected_candidate_ids, "selected_candidate_ids", allow_empty=True)
        if len(pool) > (6000 if v2 else POOL_MAX) or len(chosen) != min(K_MAX, len(pool)) or not set(chosen) <= set(pool):
            raise ValueError("Frozen selection must contain min(20,M) unique train-pool prototypes")
        if len(self.calibration_source_mask) != len(self.calibration_parent_ids) or any(
            type(x) is not bool for x in self.calibration_source_mask
        ):
            raise ValueError("Frozen calibration source mask is malformed")
        if tuple(s.candidate_id for s in self.steps) != chosen or tuple(s.k for s in self.steps) != tuple(range(1, len(chosen)+1)):
            raise ValueError("Frozen step records do not match the selected prefix")


def select_calibration(distances: Any, *, pair_status: Any, parent_ids: Sequence[str],
                       candidate_ids: Sequence[str], source_mask: Any, theta: float,
                       cap: float, contract_sha256: str, frozen_pool_sha256: str) -> SelectionFreeze:
    """Choose once on the full calibration Cartesian product, never on test.

    Key: maximal newly theta-covered *count*, maximal decrease of the capped
    mean over the fixed base-parent denominator, then canonical candidate ID.
    Zero-gain candidates are still selected until min(20, pool size).
    The driver must authenticate the train-only pool receipt before this call.
    """
    parents = _ids(parent_ids, "parent_ids")
    candidates = _ids(candidate_ids, "candidate_ids", allow_empty=True)
    if len(candidates) > POOL_MAX:
        raise ValueError("Train pool must be frozen at <=2000 before calibration is loaded")
    theta, cap = _thresholds(theta, cap)
    _sha(contract_sha256, "contract_sha256")
    _sha(frozen_pool_sha256, "frozen_pool_sha256")
    matrix, statuses, mask = _matrix(distances, pair_status, parents, candidates, source_mask)
    best = np.full(len(parents), np.inf, dtype=np.float64)
    remaining = list(range(len(candidates)))
    selected: list[str] = []
    steps: list[SelectionStep] = []
    for k in range(1, min(K_MAX, len(candidates)) + 1):
        updated = np.minimum(best[:, None], matrix[:, remaining])
        gains = np.count_nonzero((updated <= theta) & (best[:, None] > theta), axis=0)
        decreases = np.mean(np.minimum(best[:, None], cap) - np.minimum(updated, cap), axis=0)
        # No tolerance, approximate shortlist, Ours Reach objective or early
        # zero-gain stopping; reductions use the original fixed parent order.
        local = min(range(len(remaining)), key=lambda j: (-int(gains[j]), -float(decreases[j]), candidates[remaining[j]]))
        winner = remaining.pop(local)
        best = updated[:, local].copy()
        selected.append(candidates[winner])
        steps.append(SelectionStep(k, candidates[winner], int(gains[local]), float(decreases[local]),
                                   int(np.count_nonzero(best <= theta)), float(np.minimum(best, cap).mean())))
    raw = dict(schema_version="cm_crem_selection_freeze_v1", method_id=METHOD_ID,
        contract_sha256=contract_sha256, frozen_pool_sha256=frozen_pool_sha256,
        pool_candidate_ids=candidates, calibration_parent_ids=parents,
        calibration_source_mask=tuple(bool(x) for x in mask),
        calibration_matrix_sha256=_matrix_sha(matrix, statuses), theta=theta, cap=cap,
        selected_candidate_ids=tuple(selected), steps=tuple(steps))
    freeze = SelectionFreeze(**raw, freeze_sha256="")
    payload = freeze.to_dict()
    payload.pop("freeze_sha256")
    return SelectionFreeze.from_dict({**payload, "freeze_sha256": canonical_sha256(payload)})


@dataclass(frozen=True)
class PrefixEvaluation:
    selection: SelectionFreeze
    parent_ids: tuple[str, ...]
    source_mask: tuple[bool, ...]
    test_matrix_sha256: str
    best_distances: np.ndarray  # K_MAX x N; uncapped and includes semantic inf.
    best_candidate_ids: tuple[tuple[str | None, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Strict portable JSON; semantic infinity is the literal string inf."""
        if np.isnan(self.best_distances).any() or (self.best_distances < 0).any():
            raise ValueError("Invalid parent-best distances cannot be serialized as semantic infinity")
        payload = {"schema_version": "cm_crem_prefix_evaluation_v1",
            "contract_sha256": self.contract_sha256,
            "selection": self.selection.to_dict(), "parent_ids": list(self.parent_ids),
            "source_mask": list(self.source_mask), "test_matrix_sha256": self.test_matrix_sha256,
            "best_distances_uncapped": [[float(x) if np.isfinite(x) else "inf" for x in row]
                                        for row in self.best_distances],
            "best_candidate_ids": [list(row) for row in self.best_candidate_ids]}
        return {**payload, "evaluation_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PrefixEvaluation":
        payload = dict(raw)
        if payload.pop("science_hash", payload.get("contract_sha256")) != payload.get("contract_sha256"):
            raise ValueError("Driver science envelope differs from the evaluation contract")
        claimed = payload.pop("evaluation_sha256")
        if _sha(claimed, "evaluation_sha256") != canonical_sha256(payload):
            raise ValueError("Prefix evaluation content hash mismatch")
        if payload["schema_version"] != "cm_crem_prefix_evaluation_v1":
            raise ValueError("Unknown CM-CReM evaluation schema")
        freeze = SelectionFreeze.from_dict(payload["selection"])
        if payload["contract_sha256"] != freeze.contract_sha256:
            raise ValueError("Evaluation contract and frozen selection differ")
        parents = _ids(payload["parent_ids"], "parent_ids")
        mask = tuple(payload["source_mask"])
        if len(mask) != len(parents) or any(type(x) is not bool for x in mask):
            raise ValueError("Malformed evaluation source mask")
        _sha(payload["test_matrix_sha256"], "test_matrix_sha256")
        matrix = np.asarray(payload["best_distances_uncapped"], dtype=np.float64)
        if matrix.shape != (K_MAX, len(parents)) or np.isnan(matrix).any() or (matrix < 0).any():
            raise ValueError("Malformed uncapped parent-best history")
        ids = tuple(tuple(row) for row in payload["best_candidate_ids"])
        if len(ids) != K_MAX or any(len(row) != len(parents) for row in ids):
            raise ValueError("Malformed parent-best candidate history")
        if np.any(matrix[1:] > matrix[:-1]) or np.any(np.isfinite(matrix[:, np.logical_not(mask)])):
            raise ValueError("Parent-best history is not a source-valid monotone prefix")
        for k, row in enumerate(ids, start=1):
            allowed = set(freeze.selected_candidate_ids[:k])
            for i, candidate in enumerate(row):
                if (np.isfinite(matrix[k-1, i]) and candidate not in allowed) or (
                    not np.isfinite(matrix[k-1, i]) and candidate is not None
                ):
                    raise ValueError("Parent-best identity is outside its frozen prefix")
        size = len(freeze.selected_candidate_ids)
        if size < K_MAX and not all(np.array_equal(matrix[k], matrix[max(0, size-1)])
                                   and ids[k] == ids[max(0, size-1)] for k in range(size, K_MAX)):
            raise ValueError("At-most-K exhaustion must preserve the exact plateau")
        matrix.flags.writeable = False
        return cls(freeze, parents, mask, payload["test_matrix_sha256"], matrix, ids)

    @property
    def contract_sha256(self) -> str:
        return self.selection.contract_sha256

    def prefix_metrics(self) -> list[dict[str, Any]]:
        rows = []
        n = len(self.parent_ids)
        for k, best in enumerate(self.best_distances, start=1):
            finite = best[np.isfinite(best)]
            covered = int(np.count_nonzero(best <= self.selection.theta))
            rows.append({"k": k, "effective_k": min(k, len(self.selection.selected_candidate_ids)),
                "base_parent_count": n, "source_parent_count": sum(self.source_mask),
                "finite_recourse_count": len(finite), "covered_count": covered,
                "coverage": covered / n, "CCRCov": covered / n,
                "fixed_capped_mean_cost": float(np.minimum(best, self.selection.cap).mean()),
                "cost": float(np.minimum(best, self.selection.cap).mean()),
                "conditional_median_cost": float(np.median(finite)) if len(finite) else None,
                "theta": self.selection.theta, "cap": self.selection.cap})
        return rows

    def parent_best_rows(self) -> list[dict[str, Any]]:
        return [{"k": k, "effective_k": min(k, len(self.selection.selected_candidate_ids)),
                 "parent_id": parent, "source_member": self.source_mask[i],
                 "best_candidate_id": self.best_candidate_ids[k-1][i],
                 "best_distance_uncapped": float(self.best_distances[k-1, i]),
                 "finite_strict_flip": bool(np.isfinite(self.best_distances[k-1, i])),
                 "covered": bool(self.best_distances[k-1, i] <= self.selection.theta)}
                for k in range(1, K_MAX+1) for i, parent in enumerate(self.parent_ids)]

    def exact_ecdf(self, k: int) -> list[dict[str, Any]]:
        if type(k) is not int or not 1 <= k <= K_MAX:
            raise ValueError("ECDF K must be in 1..20")
        best = self.best_distances[k-1]
        finite = np.sort(best[np.isfinite(best)])
        # Keep the unresolved mass out of finite theta; never put it at cap,
        # renormalize to finite parents, or interpolate on a rounded grid.
        xs = np.unique(np.concatenate((np.array([0.0]), finite)))
        return [{"k": k, "effective_k": min(k, len(self.selection.selected_candidate_ids)),
                 "distance": float(x), "covered_count": int(np.searchsorted(finite, x, side="right")),
                 "coverage": int(np.searchsorted(finite, x, side="right"))/len(best),
                 "base_parent_count": len(best), "finite_recourse_count": len(finite),
                 "unresolved_count": len(best)-len(finite)} for x in xs]


def evaluate_frozen_test(freeze: SelectionFreeze | Mapping[str, Any], distances: Any, *,
                         pair_status: Any, parent_ids: Sequence[str], candidate_ids: Sequence[str],
                         source_mask: Any, contract_sha256: str) -> PrefixEvaluation:
    """Evaluate only the authenticated selected columns; never call a selector.

    This interface intentionally rejects a full test x 2000 pool matrix. The
    driver verifies frozen split/oracle identities through contract_sha256.
    """
    if not isinstance(freeze, SelectionFreeze):
        freeze = SelectionFreeze.from_dict(freeze)
    freeze.validate()
    if _sha(contract_sha256, "contract_sha256") != freeze.contract_sha256:
        raise ValueError("Test contract differs from the calibration freeze")
    parents = _ids(parent_ids, "parent_ids")
    candidates = _ids(candidate_ids, "candidate_ids", allow_empty=True)
    if candidates != freeze.selected_candidate_ids:
        raise ValueError("Test columns must be exactly the ordered frozen at-most-20 prototypes")
    matrix, statuses, mask = _matrix(distances, pair_status, parents, candidates, source_mask)
    best = np.full(len(parents), np.inf, dtype=np.float64)
    best_ids: list[str | None] = [None] * len(parents)
    history = []
    id_history = []
    for k in range(1, K_MAX+1):
        if k <= len(candidates):
            values, candidate = matrix[:, k-1], candidates[k-1]
            for i in range(len(parents)):
                if values[i] < best[i] or (np.isfinite(values[i]) and values[i] == best[i]
                                           and (best_ids[i] is None or candidate < best_ids[i])):
                    best[i], best_ids[i] = values[i], candidate
        history.append(best.copy())
        id_history.append(tuple(best_ids))
    distances_out = np.stack(history)
    distances_out.flags.writeable = False
    return PrefixEvaluation(freeze, parents, tuple(bool(x) for x in mask),
                            _matrix_sha(matrix, statuses), distances_out, tuple(id_history))
