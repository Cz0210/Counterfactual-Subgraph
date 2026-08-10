"""BACE candidate-universe policies and attrition accounting.

The legacy BACE matrix admitted a fragment only when its source molecule already
passed the teacher counterfactual gates.  That is useful as a historical policy,
but it is too restrictive for a class-level recourse matrix: effectiveness must
be measured across calibration parents.  The v4 policy therefore admits source
rows using chemistry and lineage only and retains source teacher outcomes as
features.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Sequence

from src.chem.hard_deletion import enumerate_connected_hard_deletions
from src.eval.candidate_pool_audit import _normalize_row
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.mutagenicity_wnode_matrix import CANDIDATE_ORDER_SOURCE_SUPPORT

try:  # pragma: no cover - deployment dependency
    from rdkit import rdBase
except ImportError:  # pragma: no cover
    rdBase = None


LEGACY_SOURCE_EFFECT_POLICY = "legacy_source_effect_v1"
CONNECTED_FEASIBLE_V4_POLICY = "connected_feasible_v4"
CANDIDATE_UNIVERSE_POLICIES = (
    LEGACY_SOURCE_EFFECT_POLICY,
    CONNECTED_FEASIBLE_V4_POLICY,
)


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def source_parent_id(row: dict[str, Any]) -> str:
    for field in ("molecule_id", "parent_id", "parent_index", "source_parent_id"):
        if field not in row or row[field] is None:
            continue
        value = str(row[field]).strip()
        if value:
            return value
    return ""


def candidate_lineage_complete(row: dict[str, Any]) -> bool:
    source = str(row.get("candidate_lineage_source") or "").strip()
    source_index_present = (
        "candidate_lineage_source_index" in row
        and row.get("candidate_lineage_source_index") is not None
    )
    return bool(
        source_parent_id(row)
        and str(row.get("parent_smiles") or row.get("smiles") or "").strip()
        and str(row.get("final_fragment") or "").strip()
        and source
        and source_index_present
    )


def _outcome_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    method = getattr(value, "as_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, dict):
            return dict(payload)
    raise TypeError(f"Unsupported connected-deletion outcome: {type(value)!r}")


def classify_connected_feasible_source_row(
    row: dict[str, Any],
    *,
    record_index: int,
    min_atom_ratio: float = 0.0,
    max_atom_ratio: float = 0.85,
    require_lineage: bool = True,
    deletion_fn: Callable[[str, str], Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Classify one source row without using source teacher effect as a gate."""

    if not 0.0 <= float(min_atom_ratio) < float(max_atom_ratio) < 1.0:
        raise ValueError("Candidate atom-ratio bounds must satisfy 0 <= min < max < 1.")
    normalized = _normalize_row(row, record_index=record_index)
    parent_smiles = str(normalized.parent_smiles or "").strip()
    fragment = str(normalized.final_fragment or "").strip()
    canonical = canonicalize_smiles(fragment) if fragment else None
    lineage_ok = candidate_lineage_complete(row) if require_lineage else bool(
        source_parent_id(row) and parent_smiles and fragment
    )
    atom_ratio = _finite(normalized.atom_ratio)
    hard_reason: str | None = None
    outcomes: list[dict[str, Any]] = []

    if not fragment or not normalized.parse_ok or not normalized.valid or canonical is None:
        hard_reason = "excluded_parse"
    elif not normalized.connected or "." in canonical:
        hard_reason = "excluded_connected_fragment"
    elif not lineage_ok:
        hard_reason = "excluded_missing_lineage"
    elif not parent_smiles or not normalized.final_substructure:
        hard_reason = "excluded_substructure"
    elif atom_ratio is None or not (
        float(min_atom_ratio) < atom_ratio < float(max_atom_ratio)
    ):
        hard_reason = "excluded_size"
    else:
        evaluator = deletion_fn or enumerate_connected_hard_deletions
        log_block = rdBase.BlockLogs() if rdBase is not None else nullcontext()
        with log_block:
            outcomes = [
                _outcome_payload(value)
                for value in evaluator(parent_smiles, canonical)
            ]
        if not outcomes:
            hard_reason = "excluded_substructure"
        elif not any(_truth(item.get("delete_valid", item.get("valid"))) for item in outcomes):
            reasons = {
                str(item.get("invalid_reason") or item.get("error") or "")
                for item in outcomes
            }
            if any("disconnected" in reason for reason in reasons):
                hard_reason = "excluded_source_disconnected"
            elif any("sanitize" in reason for reason in reasons):
                hard_reason = "excluded_source_unsanitized"
            elif any("empty" in reason for reason in reasons):
                hard_reason = "excluded_source_empty"
            else:
                hard_reason = "excluded_other"

    valid_outcomes = [
        item
        for item in outcomes
        if _truth(item.get("delete_valid", item.get("valid")))
    ]
    boundary_counts = [
        int(item.get("boundary_bond_count") or 0) for item in valid_outcomes
    ]
    return {
        "record_index": int(record_index),
        "source_parent_id": source_parent_id(row),
        "fragment_smiles": fragment or None,
        "canonical_fragment": canonical,
        "parse_ok": bool(normalized.parse_ok and normalized.valid and canonical),
        "connected_fragment": bool(normalized.connected and canonical and "." not in canonical),
        "direct_substructure": bool(normalized.direct_substructure),
        "final_substructure": bool(normalized.final_substructure),
        "projection_used": bool(normalized.projection_used),
        "lineage_complete": lineage_ok,
        "atom_ratio": atom_ratio,
        "source_cf_flip": bool(normalized.cf_flip),
        "source_cf_drop": normalized.cf_drop,
        "source_oracle_ok": bool(normalized.oracle_ok),
        "source_connected_deletion_count": len(valid_outcomes),
        "source_residual_connected": bool(valid_outcomes),
        "source_residual_sanitized": bool(valid_outcomes),
        "boundary_bond_count_min": min(boundary_counts) if boundary_counts else None,
        "attachment_count_min": min(boundary_counts) if boundary_counts else None,
        "entered_connected_feasible_universe": hard_reason is None,
        "matrix_exclusion_reason": hard_reason,
    }


def _mean(values: Sequence[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _maximum(values: Sequence[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return max(finite) if finite else None


def build_connected_feasible_candidate_universe(
    pool_rows: Sequence[dict[str, Any]],
    *,
    candidate_order: str = CANDIDATE_ORDER_SOURCE_SUPPORT,
    min_atom_ratio: float = 0.0,
    max_atom_ratio: float = 0.85,
    require_lineage: bool = True,
    deletion_fn: Callable[[str, str], Sequence[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Aggregate chemistry-feasible BACE fragments without source-effect gates."""

    if candidate_order != CANDIDATE_ORDER_SOURCE_SUPPORT:
        raise ValueError(f"Unsupported candidate order: {candidate_order!r}")
    decisions = [
        classify_connected_feasible_source_row(
            row,
            record_index=index,
            min_atom_ratio=min_atom_ratio,
            max_atom_ratio=max_atom_ratio,
            require_lineage=require_lineage,
            deletion_fn=deletion_fn,
        )
        for index, row in enumerate(pool_rows)
    ]
    eligible_rows = [
        row
        for row, decision in zip(pool_rows, decisions, strict=True)
        if decision["entered_connected_feasible_universe"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        canonical = canonicalize_smiles(str(row.get("final_fragment") or "").strip())
        if canonical is None:
            raise AssertionError("A v4 chemistry-feasible fragment lost canonical identity.")
        grouped[canonical].append(row)

    candidates: list[dict[str, Any]] = []
    for canonical, rows in grouped.items():
        parent_ids = sorted(
            {parent_id for row in rows if (parent_id := source_parent_id(row))}
        )
        candidates.append(
            {
                "canonical_fragment": canonical,
                "source_row_count": len(rows),
                "source_parent_count": len(parent_ids),
                "source_parent_ids": parent_ids,
                "source_cf_drop_mean": _mean([row.get("cf_drop") for row in rows]),
                "source_cf_drop_max": _maximum([row.get("cf_drop") for row in rows]),
                "source_cf_flip_rate": sum(_truth(row.get("cf_flip")) for row in rows)
                / len(rows),
                "source_oracle_ok_rate": sum(_truth(row.get("oracle_ok")) for row in rows)
                / len(rows),
                "source_reward_mean": _mean([row.get("reward_total") for row in rows]),
                "source_reward_max": _maximum([row.get("reward_total") for row in rows]),
                "source_atom_ratio_mean": _mean([row.get("atom_ratio") for row in rows]),
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["source_parent_count"]),
            -float(row["source_cf_drop_mean"])
            if row["source_cf_drop_mean"] is not None
            else float("inf"),
            str(row["canonical_fragment"]),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_order_index"] = index
        candidate["candidate_order"] = candidate_order

    hard_counts: dict[str, int] = defaultdict(int)
    for decision in decisions:
        reason = decision.get("matrix_exclusion_reason")
        if reason:
            hard_counts[str(reason)] += 1
    statistics = {
        "input_pool_rows": len(pool_rows),
        "source_eligible_rows": len(eligible_rows),
        "source_eligible_raw_unique_fragments": len(
            {str(row.get("final_fragment") or "").strip() for row in eligible_rows}
        ),
        "canonical_unique_candidates": len(candidates),
        "source_filter_counts": dict(sorted(hard_counts.items())),
        "source_feature_counts": {
            "source_cf_flip_true_rows": sum(_truth(row.get("cf_flip")) for row in eligible_rows),
            "source_cf_drop_ge_0_2_rows": sum(
                (value := _finite(row.get("cf_drop"))) is not None and value >= 0.2
                for row in eligible_rows
            ),
            "source_oracle_ok_rows": sum(_truth(row.get("oracle_ok")) for row in eligible_rows),
        },
        "source_filter_contract": {
            "policy": CONNECTED_FEASIBLE_V4_POLICY,
            "require_parseable_canonical_connected_fragment": True,
            "require_actual_source_substructure": True,
            "require_source_residual_nonempty_sanitized_connected": True,
            "min_atom_ratio_exclusive": float(min_atom_ratio),
            "max_atom_ratio_exclusive": float(max_atom_ratio),
            "require_candidate_lineage": bool(require_lineage),
            "source_cf_flip_is_feature_not_gate": True,
            "source_cf_drop_is_feature_not_gate": True,
            "source_oracle_is_feature_not_gate": True,
        },
    }
    return candidates, statistics, decisions


__all__ = [
    "CANDIDATE_UNIVERSE_POLICIES",
    "CONNECTED_FEASIBLE_V4_POLICY",
    "LEGACY_SOURCE_EFFECT_POLICY",
    "build_connected_feasible_candidate_universe",
    "candidate_lineage_complete",
    "classify_connected_feasible_source_row",
    "source_parent_id",
]
