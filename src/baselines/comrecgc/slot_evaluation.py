"""Preserve official COMRECGC rank slots around the shared WNode evaluator.

The shared evaluator only receives chemically valid repaired medoids.  This
module maps those pair rows back onto the immutable official rank sequence and
represents invalid medoids as unavailable slots.  It performs no candidate
selection, reordering, distance calculation, or teacher inference.
"""

from __future__ import annotations

import csv
from collections import Counter
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.fullgraph_wnode_artifacts import summarize_wnode_thresholds

from .contracts import atomic_write_bytes, sha256_file, write_json


METHOD = "COMRECGC-Adapted-DeterministicChemRepair"
ADAPTATION_MODE = "official_cluster_original_medoid_deterministic_chemical_repair"
SELECTION_METHOD = "official_cluster_rank_original_medoid_no_backfill"
FLOAT_TOLERANCE = 1e-12


def _candidate_slot_id(rank: int) -> str:
    """Return the evaluator-only identity for one immutable official rank slot."""

    return f"COMRECGC_OFFICIAL_SLOT_{int(rank):06d}"


def _slot_id(slot: Mapping[str, Any]) -> str:
    value = _text(slot.get("candidate_slot_id"))
    if value:
        return value
    rank = _integer(slot.get("official_cluster_rank"))
    if rank is None or rank <= 0:
        raise ValueError("Official medoid slot has no stable positive rank identity.")
    return _candidate_slot_id(rank)


def _source_candidate_id(slot: Mapping[str, Any]) -> str:
    value = _text(slot.get("source_candidate_id") or slot.get("candidate_id"))
    if not value:
        raise ValueError("Official medoid slot has no source candidate identity.")
    return value


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in values:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True, ensure_ascii=True)
                if isinstance(value, (dict, list, tuple))
                else ""
                if value is None
                else value
                for key, value in row.items()
            }
        )
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True, default=str) + "\n"
        for row in rows
    )
    atomic_write_bytes(path, payload.encode("utf-8"))


def load_official_slots(path: str | Path) -> list[dict[str, Any]]:
    """Load the immutable official medoid sequence without compacting it."""

    rows = read_csv(path)
    slots: list[dict[str, Any]] = []
    for row in rows:
        rank = _integer(row.get("official_cluster_rank"))
        candidate_id = _text(row.get("candidate_id"))
        cluster_id = _text(row.get("cluster_id"))
        if rank is None or rank <= 0 or not candidate_id or not cluster_id:
            raise ValueError(
                "Official medoid rows require a positive rank, cluster_id, and candidate_id."
            )
        if _bool(row.get("invalid_slot_backfill")):
            raise ValueError(f"Official rank {rank} was backfilled; evaluation is forbidden.")
        if _bool(row.get("rank_compaction")):
            raise ValueError(f"Official rank {rank} was compacted; evaluation is forbidden.")
        repair_success = _bool(row.get("repair_success"))
        smiles = _text(row.get("repaired_smiles"))
        slot_valid = bool(repair_success and smiles)
        slots.append(
            {
                **row,
                "official_cluster_rank": rank,
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "source_candidate_id": candidate_id,
                "candidate_slot_id": _candidate_slot_id(rank),
                "repair_success": repair_success,
                "repaired_smiles": smiles,
                "candidate_slot_valid": slot_valid,
                "slot_status": "REPAIRED_VALID" if slot_valid else "REPAIRED_INVALID",
                "slot_rejection_reason": "" if slot_valid else "deterministic_repair_invalid",
                "invalid_slot_backfill": False,
                "rank_compaction": False,
            }
        )
    ranks = [int(row["official_cluster_rank"]) for row in slots]
    if ranks != list(range(1, len(slots) + 1)):
        raise ValueError(
            "Official medoid ranks must remain the contiguous upstream sequence 1..N."
        )
    cluster_ids = [str(row["cluster_id"]) for row in slots]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Official cluster IDs must be unique across rank slots.")
    slot_ids = [_slot_id(row) for row in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("Official candidate slot IDs must be unique.")
    source_counts = Counter(_source_candidate_id(row) for row in slots)
    source_signatures: dict[str, tuple[bool, str, str]] = {}
    evaluation_id_by_smiles: dict[str, str] = {}
    for slot in slots:
        source_id = _source_candidate_id(slot)
        signature = (
            bool(slot["repair_success"]),
            str(slot["repaired_smiles"]),
            _text(slot.get("repaired_graph_sha256")),
        )
        previous = source_signatures.setdefault(source_id, signature)
        if previous != signature:
            raise ValueError(
                f"Reused source candidate {source_id!r} has inconsistent repair lineage."
            )
        repaired_smiles = str(slot["repaired_smiles"])
        evaluation_id = (
            evaluation_id_by_smiles.setdefault(repaired_smiles, _slot_id(slot))
            if bool(slot["candidate_slot_valid"])
            else _slot_id(slot)
        )
        slot["evaluation_candidate_id"] = evaluation_id
        slot["evaluation_compute_reused"] = evaluation_id != _slot_id(slot)
        slot["source_candidate_slot_count"] = source_counts[source_id]
        slot["source_candidate_reused_across_slots"] = source_counts[source_id] > 1
    return slots


def build_internal_valid_candidates(
    slots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a compute-only CSV while retaining each candidate's native rank."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for slot in slots:
        if not bool(slot.get("candidate_slot_valid")):
            continue
        grouped.setdefault(str(slot["evaluation_candidate_id"]), []).append(slot)
    rows: list[dict[str, Any]] = []
    for evaluation_candidate_id, group in grouped.items():
        slot = group[0]
        rows.append(
            {
                "rank": len(rows) + 1,
                "native_rank": int(slot["official_cluster_rank"]),
                "candidate_id": evaluation_candidate_id,
                "candidate_slot_id": evaluation_candidate_id,
                "source_candidate_id": _source_candidate_id(slot),
                "cluster_id": str(slot["cluster_id"]),
                "official_rank_slots": [
                    int(item["official_cluster_rank"]) for item in group
                ],
                "candidate_slot_ids": [_slot_id(item) for item in group],
                "evaluation_compute_reuse_count": len(group),
                "smiles": str(slot["repaired_smiles"]),
                "canonical_smiles": str(slot["repaired_smiles"]),
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": SELECTION_METHOD,
                "adaptation_mode": ADAPTATION_MODE,
            }
        )
    return rows


def expand_pair_rows(
    *,
    parent_ids: Sequence[str],
    slots: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand valid evaluator rows into a Cartesian parent x official-slot audit."""

    valid_ids = {
        str(slot["evaluation_candidate_id"])
        for slot in slots
        if bool(slot.get("candidate_slot_valid"))
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for source in evaluated_rows:
        row = dict(source)
        key = (_text(row.get("parent_id")), _text(row.get("candidate_id")))
        if not all(key) or key[1] not in valid_ids:
            raise ValueError(f"Unexpected shared-evaluator pair row: {key!r}")
        if key in indexed:
            raise ValueError(f"Duplicate shared-evaluator pair row: {key!r}")
        indexed[key] = row
    expected_valid = len(parent_ids) * len(valid_ids)
    if len(indexed) != expected_valid:
        raise ValueError(
            f"Shared evaluator pair count mismatch: actual={len(indexed)}, "
            f"expected={expected_valid}."
        )
    output: list[dict[str, Any]] = []
    parent_set = {str(value) for value in parent_ids}
    if len(parent_set) != len(parent_ids):
        raise ValueError("Parent IDs must be unique.")
    for parent_id in parent_ids:
        for slot in slots:
            candidate_id = _slot_id(slot)
            source_candidate_id = _source_candidate_id(slot)
            rank = int(slot["official_cluster_rank"])
            if bool(slot.get("candidate_slot_valid")):
                evaluation_candidate_id = str(slot["evaluation_candidate_id"])
                row = dict(indexed[(str(parent_id), evaluation_candidate_id)])
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "candidate_slot_id": candidate_id,
                        "evaluation_candidate_id": evaluation_candidate_id,
                        "evaluation_compute_reused": bool(
                            slot.get("evaluation_compute_reused")
                        ),
                        "source_candidate_id": source_candidate_id,
                        "cluster_id": str(slot["cluster_id"]),
                        "official_cluster_rank": rank,
                        "candidate_slot_valid": True,
                        "slot_status": "REPAIRED_VALID",
                        "slot_rejection_reason": "",
                        "invalid_slot_backfill": False,
                        "rank_compaction": False,
                    }
                )
            else:
                row = {
                    "method": METHOD,
                    "parent_id": str(parent_id),
                    "candidate_id": candidate_id,
                    "candidate_slot_id": candidate_id,
                    "evaluation_candidate_id": None,
                    "evaluation_compute_reused": False,
                    "source_candidate_id": source_candidate_id,
                    "cluster_id": str(slot.get("cluster_id") or ""),
                    "official_cluster_rank": rank,
                    "candidate_slot_valid": False,
                    "slot_status": "REPAIRED_INVALID",
                    "slot_rejection_reason": str(
                        slot.get("slot_rejection_reason")
                        or "deterministic_repair_invalid"
                    ),
                    "match": False,
                    "delete_valid": False,
                    "applicable": False,
                    "distance": None,
                    "wnode_distance": None,
                    "pred_before": None,
                    "pred_after": None,
                    "teacher_strict_flip": False,
                    "cf_flip": False,
                    "cf_drop": None,
                    "error": "candidate_not_sent_to_rf_or_wnode",
                    "invalid_slot_backfill": False,
                    "rank_compaction": False,
                }
            output.append(row)
    return output


def _strict_flip(row: Mapping[str, Any], source_label: int, target_label: int) -> bool:
    before = _integer(row.get("pred_before"))
    after = _integer(row.get("pred_after"))
    return before == int(source_label) and after == int(target_label)


def _distance(row: Mapping[str, Any]) -> float | None:
    primary = _finite(row.get("distance"))
    return primary if primary is not None else _finite(row.get("wnode_distance"))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def compute_slot_metrics(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    parent_ids: Sequence[str],
    thresholds: Sequence[float],
    theta_star: float,
    cost_cap: float,
    max_k: int,
    source_label: int = 1,
    target_label: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute frozen K prefixes using original rank slots and shared summaries."""

    if max_k <= 0:
        raise ValueError("max_k must be positive.")
    threshold_values = [float(value) for value in thresholds]
    if threshold_values != sorted(set(threshold_values)):
        raise ValueError("Thresholds must be sorted and unique.")
    rank_by_id = {
        _slot_id(slot): int(slot["official_cluster_rank"])
        for slot in slots
    }
    source_id_by_slot = {_slot_id(slot): _source_candidate_id(slot) for slot in slots}
    valid_by_rank = {
        int(slot["official_cluster_rank"]): bool(slot.get("candidate_slot_valid"))
        for slot in slots
    }
    rows_by_parent: dict[str, list[dict[str, Any]]] = {
        str(parent_id): [] for parent_id in parent_ids
    }
    for source in pair_rows:
        row = dict(source)
        parent_id = _text(row.get("parent_id"))
        candidate_id = _text(row.get("candidate_id"))
        if parent_id not in rows_by_parent or candidate_id not in rank_by_id:
            raise ValueError("Pair rows do not match the frozen parent/slot universe.")
        rows_by_parent[parent_id].append(row)
    expected_rows = len(parent_ids) * len(slots)
    if len(pair_rows) != expected_rows:
        raise ValueError(
            f"Expanded slot matrix is incomplete: {len(pair_rows)} != {expected_rows}."
        )
    prefixes: list[dict[str, Any]] = []
    threshold_metrics: list[dict[str, Any]] = []
    parent_best_rows: list[dict[str, Any]] = []
    threshold_rows = [
        {
            "threshold": value,
            "threshold_source": "frozen_calibration",
            "quantile": None,
        }
        for value in threshold_values
    ]
    for requested_k in range(1, int(max_k) + 1):
        available_rank_slots = min(requested_k, len(slots))
        prefix_rows = [
            dict(row)
            for row in pair_rows
            if rank_by_id[str(row["candidate_id"])] <= requested_k
        ]
        shared = summarize_wnode_thresholds(
            method=METHOD,
            details=prefix_rows,
            threshold_rows=threshold_rows,
            total_parents=len(parent_ids),
            total_candidates=requested_k,
            source_label=source_label,
            target_label=target_label,
            group_audit={
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": SELECTION_METHOD,
                "evaluation_row_unit": "parent_official_rank_slot",
                "num_unique_parent_candidate_pairs": len(prefix_rows),
                "num_detail_rows": len(prefix_rows),
                "num_valid_match_instances": None,
            },
        )
        shared_rows = [
            {
                **row,
                "k": requested_k,
                "requested_k": requested_k,
                "available_rank_slots": available_rank_slots,
                "valid_k": sum(
                    bool(valid_by_rank.get(rank, False))
                    for rank in range(1, requested_k + 1)
                ),
                "invalid_or_missing_k": requested_k
                - sum(
                    bool(valid_by_rank.get(rank, False))
                    for rank in range(1, requested_k + 1)
                ),
                "invalid_slot_backfill": False,
                "rank_compaction": False,
            }
            for row in shared
        ]
        threshold_metrics.extend(shared_rows)
        theta_summary = next(
            (
                row
                for row in shared_rows
                if math.isclose(
                    float(row["threshold"]),
                    float(theta_star),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            ),
            None,
        )
        if theta_summary is None:
            theta_summary = {
                **summarize_wnode_thresholds(
                    method=METHOD,
                    details=prefix_rows,
                    threshold_rows=[
                        {
                            "threshold": float(theta_star),
                            "threshold_source": "frozen_calibration_theta_star",
                            "quantile": 0.30,
                        }
                    ],
                    total_parents=len(parent_ids),
                    total_candidates=requested_k,
                    source_label=source_label,
                    target_label=target_label,
                )[0],
                "k": requested_k,
            }
        conditional: list[float] = []
        applicable_parent_count = 0
        for parent_id in parent_ids:
            parent_rows = [
                row
                for row in rows_by_parent[str(parent_id)]
                if rank_by_id[str(row["candidate_id"])] <= requested_k
            ]
            finite_rows = [
                (distance, row)
                for row in parent_rows
                if (distance := _distance(row)) is not None
                and _strict_flip(row, source_label, target_label)
            ]
            if any(_distance(row) is not None for row in parent_rows):
                applicable_parent_count += 1
            if finite_rows:
                best_distance, best_row = min(
                    finite_rows,
                    key=lambda value: (
                        value[0],
                        rank_by_id[str(value[1]["candidate_id"])],
                    ),
                )
                conditional.append(float(best_distance))
                best_candidate_id: str | None = str(best_row["candidate_id"])
                best_rank: int | None = rank_by_id[best_candidate_id]
                best_source_candidate_id: str | None = source_id_by_slot[best_candidate_id]
                cf_drop = _finite(best_row.get("cf_drop"))
            else:
                best_distance = None
                best_candidate_id = None
                best_rank = None
                best_source_candidate_id = None
                cf_drop = None
            parent_best_rows.append(
                {
                    "k": requested_k,
                    "requested_k": requested_k,
                    "parent_id": str(parent_id),
                    "best_candidate_id": best_candidate_id,
                    "best_source_candidate_id": best_source_candidate_id,
                    "best_official_cluster_rank": best_rank,
                    "best_distance": best_distance,
                    "capped_distance": min(best_distance, cost_cap)
                    if best_distance is not None
                    else float(cost_cap),
                    "strict_recourse_available": best_distance is not None,
                    "theta_star_covered": bool(
                        best_distance is not None and best_distance <= theta_star
                    ),
                    "cf_drop": cf_drop,
                }
            )
        valid_k = sum(
            bool(valid_by_rank.get(rank, False))
            for rank in range(1, requested_k + 1)
        )
        capped = [min(value, cost_cap) for value in conditional] + [
            float(cost_cap)
        ] * (len(parent_ids) - len(conditional))
        prefixes.append(
            {
                **theta_summary,
                "k": requested_k,
                "requested_k": requested_k,
                "available_rank_slots": available_rank_slots,
                "valid_k": valid_k,
                "invalid_or_missing_k": requested_k - valid_k,
                "num_applicable_parents": applicable_parent_count,
                "applicable_coverage": applicable_parent_count / len(parent_ids)
                if parent_ids
                else 0.0,
                "num_any_strict_flip_parents": len(conditional),
                "any_strict_flip_coverage": len(conditional) / len(parent_ids)
                if parent_ids
                else 0.0,
                "conditional_mean_cost": sum(conditional) / len(conditional)
                if conditional
                else None,
                "conditional_median_cost": _median(conditional),
                "fixed_capped_mean_cost": sum(capped) / len(capped) if capped else None,
                "fixed_capped_median_cost": _median(capped),
                "coverage_redundancy": None,
                "structural_redundancy": None,
                "adaptation_mode": ADAPTATION_MODE,
                "invalid_slot_backfill": False,
                "rank_compaction": False,
            }
        )
    _assert_monotonic(prefixes, threshold_metrics, max_k=max_k)
    return prefixes, threshold_metrics, parent_best_rows


def _assert_monotonic(
    prefixes: Sequence[Mapping[str, Any]],
    threshold_metrics: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
) -> None:
    coverage = [float(row["close_cf_coverage"]) for row in prefixes]
    if any(right + FLOAT_TOLERANCE < left for left, right in zip(coverage, coverage[1:])):
        raise ValueError("Coverage versus K is not monotonic nondecreasing.")
    rows = sorted(
        (
            row
            for row in threshold_metrics
            if int(row["k"]) == int(max_k)
        ),
        key=lambda row: float(row["threshold"]),
    )
    threshold_coverage = [float(row["close_cf_coverage"]) for row in rows]
    if any(
        right + FLOAT_TOLERANCE < left
        for left, right in zip(threshold_coverage, threshold_coverage[1:])
    ):
        raise ValueError("Coverage versus threshold is not monotonic nondecreasing.")


def table_row(
    prefix: Mapping[str, Any], *, theta_star: float, dataset: str = "Mutagenicity"
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "dataset": dataset,
        "source_label": 1,
        "target_label": 0,
        "k": int(prefix["k"]),
        "requested_k": int(prefix["requested_k"]),
        "valid_k": int(prefix["valid_k"]),
        "theta": float(theta_star),
        "coverage": float(prefix["close_cf_coverage"]),
        "ccrcov": float(prefix["close_cf_coverage"]),
        "applicable_rate": float(prefix["applicable_coverage"]),
        "mean_cf_drop": prefix.get("avg_cf_drop_among_covered"),
        "conditional_mean_cost": prefix.get("conditional_mean_cost"),
        "conditional_median_cost": prefix.get("conditional_median_cost"),
        "fixed_capped_mean_cost": prefix.get("fixed_capped_mean_cost"),
        "fixed_capped_median_cost": prefix.get("fixed_capped_median_cost"),
        "num_parents": int(prefix["num_parents"]),
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "adaptation_mode": ADAPTATION_MODE,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
    }


def build_final_audit(
    *,
    root: Path,
    prefixes: Sequence[Mapping[str, Any]],
    figure4: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    parent_count: int,
    thresholds: Sequence[float],
    evaluator_invoked: bool,
    interface_probe_invoked: bool,
) -> dict[str, Any]:
    valid_slots = sum(bool(row.get("candidate_slot_valid")) for row in slots)
    source_candidate_ids = [_source_candidate_id(row) for row in slots]
    evaluated_candidate_ids = {
        str(row["evaluation_candidate_id"])
        for row in slots
        if bool(row.get("candidate_slot_valid"))
    }
    strict_flip_observed = any(
        int(row.get("num_any_strict_flip_parents") or 0) > 0 for row in prefixes
    )
    failures: list[str] = []
    if len(prefixes) != 20 or [int(row["k"]) for row in prefixes] != list(range(1, 21)):
        failures.append("figure3_k_grid")
    if [float(row["threshold"]) for row in figure4] != [float(value) for value in thresholds]:
        failures.append("figure4_threshold_grid")
    if any(bool(row.get("invalid_slot_backfill")) for row in slots):
        failures.append("invalid_slot_backfill")
    if any(bool(row.get("rank_compaction")) for row in slots):
        failures.append("rank_compaction")
    audit = {
        "schema_version": 1,
        "audit_passed": not failures,
        "run_complete": not failures,
        "failed_hard_checks": failures,
        "method": METHOD,
        "adaptation_mode": ADAPTATION_MODE,
        "distance_line": "MolCLR-Node-Wasserstein",
        "distance_type": "node_wasserstein",
        "cf_mode": "strict_flip",
        "parent_count": int(parent_count),
        "official_rank_slot_count": len(slots),
        "unique_candidate_slot_count": len({_slot_id(row) for row in slots}),
        "unique_source_candidate_count": len(set(source_candidate_ids)),
        "source_candidate_duplicate_slot_count": len(source_candidate_ids)
        - len(set(source_candidate_ids)),
        "source_candidate_reuse_preserved": True,
        "unique_evaluated_chemical_candidate_count": len(evaluated_candidate_ids),
        "evaluation_compute_reuse_slot_count": valid_slots
        - len(evaluated_candidate_ids),
        "evaluation_compute_reuse_semantics": (
            "identical_repaired_smiles_scored_once_then_expanded_to_official_slots"
        ),
        "valid_repaired_slot_count": valid_slots,
        "invalid_repaired_slot_count": len(slots) - valid_slots,
        "strict_flip_status": "STRICT_FLIP_OBSERVED"
        if strict_flip_observed
        else "STRICT_FLIP_NOT_OBSERVED",
        "scientific_output_empty": not strict_flip_observed,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "invalid_candidates_sent_to_rf_or_wnode": False,
        "shared_evaluator_invoked": bool(evaluator_invoked),
        "interface_probe_invoked": bool(interface_probe_invoked),
        "rf_callable": bool(evaluator_invoked or interface_probe_invoked),
        "wnode_callable": bool(evaluator_invoked or interface_probe_invoked),
        "figure3_prefix_nested": True,
        "coverage_vs_k_monotonic": True,
        "coverage_vs_threshold_monotonic": True,
        "calibration_loaded": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "file_sha256": {
            path.name: sha256_file(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name not in {"final_artifact_audit.json", "run_manifest.json"}
        },
    }
    write_json(root / "final_artifact_audit.json", audit)
    return audit
