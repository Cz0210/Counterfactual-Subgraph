"""Frozen full-graph WNode metrics and final-artifact export utilities.

This module is intentionally post-processing only.  It reads an existing
Cartesian parent/candidate evaluation and never calls a teacher, selector, or
distance provider.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.eval.flip_semantics import (
    OLD_WEAK_FLIP_DEFINITION,
    TEACHER_STRICT_FLIP_DEFINITION,
    old_weak_flip,
)
from src.eval.mutagenicity_wnode_selector import (
    ThresholdLevel,
    build_candidate_chemistry,
    build_coverage_redundancy_matrix,
)


FLOAT_TOLERANCE = 1e-12
OFFICIAL_FIELDS = (
    "num_parents",
    "num_candidates",
    "num_valid_pairs",
    "num_close_only_covered",
    "close_only_coverage",
    "num_close_cf_covered",
    "close_cf_coverage",
    "avg_best_distance",
    "median_best_distance",
    "avg_cf_drop_among_covered",
    "flip_rate_among_covered",
)
TABLE_REQUIRED_FIELDS = (
    "method",
    "dataset",
    "source_label",
    "target_label",
    "k",
    "theta",
    "ccrcov",
    "applicable_coverage",
    "any_strict_flip_coverage",
    "flip_rate_among_covered",
    "avg_cf_drop_among_covered",
    "conditional_mean_cost",
    "conditional_median_cost",
    "fixed_capped_mean_cost",
    "fixed_capped_median_cost",
    "coverage_redundancy",
    "structural_redundancy",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_distance(row: dict[str, Any]) -> float | None:
    for field in ("distance", "wnode_distance"):
        value = _as_float(row.get(field))
        if value is not None:
            return value
    return None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [number for value in values if (number := _as_float(value)) is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _median(values: Iterable[Any]) -> float | None:
    clean = sorted(
        number for value in values if (number := _as_float(value)) is not None
    )
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _strict_flip(
    row: dict[str, Any],
    *,
    source_label: int | None = None,
    target_label: int | None = None,
) -> bool:
    label = source_label
    if label is None:
        label = _as_int(row.get("label"))
    before = _as_int(row.get("pred_before"))
    after = _as_int(row.get("pred_after"))
    if label is not None and before is not None and after is not None:
        if target_label is not None:
            return before == int(label) and after == int(target_label)
        return before == int(label) and after != int(label)
    return _as_bool(row.get("teacher_strict_flip") or row.get("cf_flip"))


def _applicable(row: dict[str, Any]) -> bool:
    if "applicable" in row and row.get("applicable") not in (None, ""):
        return _as_bool(row.get("applicable"))
    if "match" in row or "delete_valid" in row:
        return _as_bool(row.get("match")) and _as_bool(row.get("delete_valid"))
    return _finite_distance(row) is not None


def _best_strict_row(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    source_label: int | None,
    target_label: int | None,
) -> dict[str, Any] | None:
    best: tuple[float, int, dict[str, Any]] | None = None
    for position, row in enumerate(rows):
        distance = _finite_distance(row)
        if distance is None:
            continue
        if threshold is not None and distance > float(threshold):
            continue
        if not _strict_flip(
            row,
            source_label=source_label,
            target_label=target_label,
        ):
            continue
        key = (distance, position, row)
        if best is None or key[:2] < best[:2]:
            best = key
    return best[2] if best is not None else None


def summarize_wnode_thresholds(
    *,
    method: str,
    details: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    source_label: int | None = None,
    target_label: int | None = None,
    feature_cost: str = "cosine",
    node_mass: str = "uniform",
    size_penalty_beta: float = 0.0,
    cf_mode: str = "strict_flip",
    cache_hit_rate: float = 0.0,
    node_embedding_cache_hit_rate: float = 0.0,
    skip_redundancy: bool = True,
    group_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Production WNode threshold aggregation shared by evaluator/exporter."""

    if cf_mode != "strict_flip":
        raise ValueError("summarize_wnode_thresholds requires cf_mode=strict_flip.")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        parent_id = _text(row.get("parent_id"))
        if not parent_id:
            raise ValueError("A detail row has an empty parent_id.")
        by_parent.setdefault(parent_id, []).append(dict(row))
    teacher_target = {
        parent_id
        for parent_id, rows in by_parent.items()
        if any(
            _as_int(row.get("pred_before"))
            == (
                int(source_label)
                if source_label is not None
                else _as_int(row.get("label"))
            )
            for row in rows
        )
    }
    valid_pairs = sum(_finite_distance(row) is not None for row in details)
    audit = dict(group_audit or {})
    output: list[dict[str, Any]] = []
    for threshold_row in threshold_rows:
        threshold = float(threshold_row["threshold"])
        if not math.isfinite(threshold):
            continue
        close_only = {
            parent_id
            for parent_id, rows in by_parent.items()
            if any(
                (distance := _finite_distance(row)) is not None
                and distance <= threshold
                for row in rows
            )
        }
        best_rows = [
            best
            for rows in by_parent.values()
            if (
                best := _best_strict_row(
                    rows,
                    threshold=threshold,
                    source_label=source_label,
                    target_label=target_label,
                )
            )
            is not None
        ]
        close_cf = {_text(row.get("parent_id")) for row in best_rows}
        weak = {
            parent_id
            for parent_id, rows in by_parent.items()
            if any(
                (distance := _finite_distance(row)) is not None
                and distance <= threshold
                and old_weak_flip(
                    row.get("pred_after"),
                    int(
                        source_label
                        if source_label is not None
                        else (_as_int(row.get("label")) or 0)
                    ),
                )
                for row in rows
            )
        }
        output.append(
            {
                "method": method,
                "distance_type": "node_wasserstein",
                "distance_line": "MolCLR-Node-Wasserstein",
                "feature_cost": feature_cost,
                "node_mass": node_mass,
                "size_penalty_beta": float(size_penalty_beta),
                "solver": "exact_emd2",
                "threshold": threshold,
                "threshold_source": threshold_row.get("threshold_source"),
                "quantile": threshold_row.get("quantile"),
                "cf_mode": cf_mode,
                "main_ccrcov_uses": "teacher_strict_flip",
                "teacher_strict_flip_definition": TEACHER_STRICT_FLIP_DEFINITION,
                "old_weak_flip_definition": OLD_WEAK_FLIP_DEFINITION,
                "old_weak_ccrcov_status": "audit_only",
                "num_parents": int(total_parents),
                "num_teacher_target_parents": len(teacher_target),
                "num_candidates": int(total_candidates),
                "num_valid_pairs": int(valid_pairs),
                "num_close_only_covered": len(close_only),
                "close_only_coverage": _rate(len(close_only), total_parents),
                "num_close_cf_covered": len(close_cf),
                "close_cf_coverage": _rate(len(close_cf), total_parents),
                "old_weak_num_close_cf_covered": len(weak),
                "old_weak_close_cf_coverage": _rate(len(weak), total_parents),
                "avg_best_distance": _mean(
                    _finite_distance(row) for row in best_rows
                ),
                "median_best_distance": _median(
                    _finite_distance(row) for row in best_rows
                ),
                "avg_cf_drop_among_covered": _mean(
                    row.get("cf_drop") for row in best_rows
                ),
                "flip_rate_among_covered": _mean(
                    1.0
                    if _strict_flip(
                        row,
                        source_label=source_label,
                        target_label=target_label,
                    )
                    else 0.0
                    for row in best_rows
                ),
                "total_pairs": len(details),
                "cache_hit_rate": float(cache_hit_rate),
                "node_embedding_cache_hit_rate": float(
                    node_embedding_cache_hit_rate
                ),
                "skip_redundancy": bool(skip_redundancy),
                **audit,
            }
        )
    return output


def read_csv(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _candidate_smiles(row: dict[str, Any]) -> str:
    for field in (
        "canonical_smiles",
        "candidate_smiles",
        "smiles",
        "graph_smiles",
        "cf_smiles",
        "final_smiles",
    ):
        value = _text(row.get(field))
        if value:
            return value
    return ""


def load_ranked_candidates(
    path: str | Path,
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, fields = read_csv(path)
    if "rank" not in fields:
        raise ValueError(f"Frozen candidate CSV requires a rank column: {path}")
    ranked: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        rank = _as_int(row.get("rank"))
        candidate_id = _text(row.get("candidate_id"))
        smiles = _candidate_smiles(row)
        if rank is None or rank <= 0 or not candidate_id or not smiles:
            raise ValueError(f"Invalid frozen candidate row: {row}")
        normalized = dict(row)
        normalized["rank"] = rank
        normalized["candidate_id"] = candidate_id
        normalized["candidate_smiles"] = smiles
        ranked.append((rank, normalized))
    ranked.sort(key=lambda item: item[0])
    ordered = [row for _, row in ranked]
    expected_ranks = list(range(1, int(expected_count) + 1))
    if [rank for rank, _ in ranked] != expected_ranks:
        raise ValueError(
            f"Frozen candidate ranks must be 1..{expected_count}: "
            f"{[rank for rank, _ in ranked]}"
        )
    ids = [str(row["candidate_id"]) for row in ordered]
    smiles = [str(row["candidate_smiles"]) for row in ordered]
    if len(set(ids)) != len(ids) or len(set(smiles)) != len(smiles):
        raise ValueError("Frozen candidates contain duplicate IDs or SMILES.")
    return ordered, fields


def locate_test_inputs(test_run_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(test_run_dir).expanduser().resolve()
    pair_candidates = (
        root / "details" / "pair_details.csv",
        root / "pair_details.csv",
        root / "test_pair_details.csv",
    )
    summary_candidates = (
        root / "combined" / "combined_threshold_summary.csv",
        root / "combined_threshold_summary.csv",
        root / "test_threshold_summary.csv",
    )
    config_candidates = (root / "run_config.json", root / "run_manifest.json")
    pair = next((path for path in pair_candidates if path.is_file()), None)
    summary = next((path for path in summary_candidates if path.is_file()), None)
    config = next((path for path in config_candidates if path.is_file()), None)
    if pair is None or summary is None or config is None:
        raise FileNotFoundError(
            f"Missing pair details, combined summary, or run config under {root}."
        )
    return pair, summary, config


def validate_complete_cartesian(
    details: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    expected_parent_count: int,
    expected_pair_count: int,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    candidate_set = set(candidate_ids)
    parent_order: list[str] = []
    seen_parents: set[str] = set()
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in details:
        parent_id = _text(row.get("parent_id"))
        candidate_id = _text(row.get("candidate_id"))
        if not parent_id or not candidate_id:
            raise ValueError("Pair details contain an empty parent/candidate ID.")
        if candidate_id not in candidate_set:
            raise ValueError(f"Pair details contain unfrozen candidate {candidate_id!r}.")
        key = (parent_id, candidate_id)
        if key in seen_pairs:
            raise ValueError(f"Duplicate parent-candidate pair: {key}")
        seen_pairs.add(key)
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_order.append(parent_id)
        matrix.setdefault(parent_id, {})[candidate_id] = dict(row)
    if len(parent_order) != int(expected_parent_count):
        raise ValueError(
            f"Parent count mismatch: {len(parent_order)} != {expected_parent_count}"
        )
    expected = int(expected_parent_count) * len(candidate_ids)
    if expected != int(expected_pair_count) or len(seen_pairs) != expected:
        raise ValueError(
            f"Cartesian pair count mismatch: rows={len(seen_pairs)}, "
            f"expected={expected}, CLI_expected={expected_pair_count}."
        )
    missing = [
        (parent_id, candidate_id)
        for parent_id in parent_order
        for candidate_id in candidate_ids
        if candidate_id not in matrix[parent_id]
    ]
    if missing:
        raise ValueError(f"Incomplete Cartesian matrix; missing sample={missing[:5]}")
    return parent_order, matrix


def _pairwise_prefix_mean(matrix: np.ndarray, k: int) -> float:
    if int(k) < 2:
        return 0.0
    selected = matrix[: int(k), : int(k)]
    upper = selected[np.triu_indices(int(k), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def compute_prefix_artifacts(
    *,
    details: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    parent_ids: Sequence[str],
    thresholds: Sequence[float],
    theta_star: float,
    cost_cap: float,
    source_label: int,
    target_label: int,
    method_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    parent_index = {parent_id: index for index, parent_id in enumerate(parent_ids)}
    distances = np.full((len(parent_ids), len(candidates)), np.inf, dtype=np.float64)
    cf_drops = np.full_like(distances, np.nan)
    applicable = np.zeros_like(distances, dtype=bool)
    rows_by_parent: dict[str, list[dict[str, Any]]] = {
        parent_id: [] for parent_id in parent_ids
    }
    for row in details:
        parent_id = _text(row.get("parent_id"))
        candidate_id = _text(row.get("candidate_id"))
        rows_by_parent[parent_id].append(dict(row))
        i = parent_index[parent_id]
        j = candidate_index[candidate_id]
        applicable[i, j] = _applicable(row)
        distance = _finite_distance(row)
        if distance is not None and _strict_flip(
            row,
            source_label=source_label,
            target_label=target_label,
        ):
            distances[i, j] = distance
            drop = _as_float(row.get("cf_drop"))
            if drop is not None:
                cf_drops[i, j] = drop

    coverage_redundancy = build_coverage_redundancy_matrix(
        distances,
        (
            ThresholdLevel(
                threshold_id="theta_star",
                threshold=float(theta_star),
                weight=1.0,
                quantiles=(),
                quantile_labels=(),
            ),
        ),
    )
    chemistry = build_candidate_chemistry(
        [
            {
                "candidate_id": row["candidate_id"],
                "canonical_fragment": row["candidate_smiles"],
            }
            for row in candidates
        ]
    )

    prefix_metrics: list[dict[str, Any]] = []
    threshold_metrics: list[dict[str, Any]] = []
    parent_best_rows: list[dict[str, Any]] = []
    for k in range(1, len(candidates) + 1):
        prefix_details = [
            row
            for row in details
            if candidate_index[_text(row.get("candidate_id"))] < k
        ]
        threshold_rows = [
            {
                "threshold": float(threshold),
                "threshold_source": "frozen_calibration",
                "quantile": None,
            }
            for threshold in thresholds
        ]
        summaries = summarize_wnode_thresholds(
            method=method_name,
            details=prefix_details,
            threshold_rows=threshold_rows,
            total_parents=len(parent_ids),
            total_candidates=k,
            source_label=source_label,
            target_label=target_label,
            group_audit={
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": "frozen_rank_prefix",
                "evaluation_row_unit": "parent_candidate",
                "num_unique_parent_candidate_pairs": len(parent_ids) * k,
                "num_detail_rows": len(prefix_details),
                "num_valid_match_instances": None,
            },
        )
        threshold_metrics.extend({**row, "k": k} for row in summaries)
        theta_summary = next(
            (
                row
                for row in summaries
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
            theta_summary = summarize_wnode_thresholds(
                method=method_name,
                details=prefix_details,
                threshold_rows=[
                    {
                        "threshold": float(theta_star),
                        "threshold_source": "frozen_calibration_theta_star",
                        "quantile": 0.30,
                    }
                ],
                total_parents=len(parent_ids),
                total_candidates=k,
                source_label=source_label,
                target_label=target_label,
            )[0]
        best = np.min(distances[:, :k], axis=1)
        best_candidate_positions = np.argmin(distances[:, :k], axis=1)
        finite = np.isfinite(best)
        capped = np.minimum(best, float(cost_cap))
        capped[~finite] = float(cost_cap)
        conditional = best[finite]
        applicable_parent = np.any(applicable[:, :k], axis=1)
        prefix_row = {
            "k": k,
            **theta_summary,
            "num_applicable_parents": int(np.count_nonzero(applicable_parent)),
            "applicable_coverage": float(np.mean(applicable_parent)),
            "num_any_strict_flip_parents": int(np.count_nonzero(finite)),
            "any_strict_flip_coverage": float(np.mean(finite)),
            "conditional_mean_cost": (
                float(np.mean(conditional)) if conditional.size else None
            ),
            "conditional_median_cost": (
                float(np.median(conditional)) if conditional.size else None
            ),
            "fixed_capped_mean_cost": float(np.mean(capped)),
            "fixed_capped_median_cost": float(np.median(capped)),
            "coverage_redundancy": _pairwise_prefix_mean(
                coverage_redundancy, k
            ),
            "structural_redundancy": _pairwise_prefix_mean(
                chemistry.structural_similarity, k
            ),
        }
        prefix_metrics.append(prefix_row)
        for i, parent_id in enumerate(parent_ids):
            selected_position = int(best_candidate_positions[i]) if finite[i] else -1
            selected_drop = (
                _as_float(cf_drops[i, selected_position])
                if selected_position >= 0
                else None
            )
            parent_best_rows.append(
                {
                    "k": k,
                    "parent_id": parent_id,
                    "best_candidate_id": (
                        candidate_ids[selected_position]
                        if selected_position >= 0
                        else None
                    ),
                    "best_distance": float(best[i]) if finite[i] else None,
                    "capped_distance": float(capped[i]),
                    "strict_recourse_available": bool(finite[i]),
                    "theta_star_covered": bool(best[i] <= float(theta_star)),
                    "applicable": bool(applicable_parent[i]),
                    "cf_drop": selected_drop,
                }
            )
    return prefix_metrics, threshold_metrics, parent_best_rows


def _same_value(expected: Any, actual: Any, field: str) -> bool:
    if field.startswith("num_"):
        return _as_int(expected) == _as_int(actual)
    left = _as_float(expected)
    right = _as_float(actual)
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def reconstruct_official_summary(
    *,
    recomputed_k20: Sequence[dict[str, Any]],
    official_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    theta_star: float,
    expected_theta_star_covered: int | None = None,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for threshold in thresholds:
        recomputed = next(
            row
            for row in recomputed_k20
            if math.isclose(
                float(row["threshold"]),
                float(threshold),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        )
        matches = [
            row
            for row in official_rows
            if (_as_float(row.get("threshold")) is not None)
            and math.isclose(
                float(row["threshold"]),
                float(threshold),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Official summary must contain one row for threshold={threshold}; "
                f"found={len(matches)}."
            )
        official = matches[0]
        field_results = {
            field: _same_value(official.get(field), recomputed.get(field), field)
            for field in OFFICIAL_FIELDS
        }
        comparisons.append(
            {
                "threshold": float(threshold),
                "all_fields_match": all(field_results.values()),
                "field_matches": field_results,
                "official": {field: official.get(field) for field in OFFICIAL_FIELDS},
                "recomputed": {
                    field: recomputed.get(field) for field in OFFICIAL_FIELDS
                },
            }
        )
    failures = [row for row in comparisons if not row["all_fields_match"]]
    if failures:
        raise RuntimeError(
            "Official K20 threshold summary reconstruction failed: "
            f"{json.dumps(failures[:2], ensure_ascii=False)}"
        )
    theta_row = next(
        row
        for row in recomputed_k20
        if math.isclose(
            float(row["threshold"]),
            float(theta_star),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    )
    if expected_theta_star_covered is not None:
        actual = int(theta_row["num_close_cf_covered"])
        if actual != int(expected_theta_star_covered):
            raise RuntimeError(
                f"Theta-star covered count mismatch: {actual} != "
                f"{expected_theta_star_covered}."
            )
        expected_coverage = int(expected_theta_star_covered) / int(
            theta_row["num_parents"]
        )
        if not math.isclose(
            float(theta_row["close_cf_coverage"]),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError("Theta-star coverage is not covered/num_parents.")
    return {
        "official_summary_reconstruction_passed": True,
        "float_abs_tolerance": FLOAT_TOLERANCE,
        "float_rel_tolerance": 0.0,
        "threshold_count": len(thresholds),
        "comparisons": comparisons,
        "theta_star": float(theta_star),
        "theta_star_num_close_cf_covered": int(
            theta_row["num_close_cf_covered"]
        ),
        "theta_star_close_cf_coverage": float(theta_row["close_cf_coverage"]),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), ensure_ascii=False)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else ("" if row.get(field) is None else row.get(field))
                    )
                    for field in fields
                }
            )


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _deep_find(payload: Any, names: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in names and value not in (None, ""):
                return value
        for value in payload.values():
            found = _deep_find(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _deep_find(value, names)
            if found not in (None, ""):
                return found
    return None


def _inherited_file_identity(
    config: dict[str, Any],
    path_names: set[str],
    hash_names: set[str],
) -> dict[str, Any]:
    path_value = _deep_find(config, path_names)
    inherited_hash = _deep_find(config, hash_names)
    result = {
        "path": str(path_value) if path_value not in (None, "") else None,
        "sha256": (
            str(inherited_hash) if inherited_hash not in (None, "") else None
        ),
        "sha256_source": "run_config",
    }
    if path_value not in (None, ""):
        path = Path(str(path_value)).expanduser()
        if path.is_file():
            result["path"] = str(path.resolve())
            result["sha256"] = sha256_file(path)
            result["sha256_source"] = "file"
    return result


def _resolve_table_fields(
    ours_schema_root: Path,
    k: int,
) -> list[str]:
    reference = ours_schema_root / f"table2_ours_k{k}.csv"
    _, fields = read_csv(reference)
    resolved = list(fields)
    for field in TABLE_REQUIRED_FIELDS:
        if field not in resolved:
            resolved.append(field)
    return resolved


def _extract_threshold_values(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("raw_quantile_thresholds")
    if isinstance(raw, list):
        values = [
            _as_float(item.get("threshold"))
            for item in raw
            if isinstance(item, dict)
        ]
        if values and all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    raw = payload.get("thresholds")
    if isinstance(raw, list):
        values = [_as_float(value) for value in raw]
        if values and all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return []


def validate_frozen_threshold_provenance(
    *,
    ours_schema_root: str | Path,
    calibration_run_dir: str | Path,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    ours_root = Path(ours_schema_root).expanduser().resolve()
    threshold_path = ours_root / "thresholds.json"
    frozen = read_json(threshold_path)
    frozen_theta = _as_float(frozen.get("theta_star"))
    frozen_cap = _as_float(frozen.get("cost_cap"))
    frozen_thresholds = _extract_threshold_values(frozen)
    if frozen_theta is None or frozen_cap is None or not frozen_thresholds:
        raise ValueError(
            f"Ours frozen thresholds schema is incomplete: {threshold_path}"
        )
    if not math.isclose(
        frozen_theta,
        float(theta_star),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("CLI theta_star differs from frozen calibration theta_star.")
    if not math.isclose(
        frozen_cap,
        float(cost_cap),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("CLI cost_cap differs from frozen calibration cost_cap.")
    requested = [float(value) for value in thresholds]
    if len(requested) != len(frozen_thresholds) or any(
        not math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
        for left, right in zip(requested, frozen_thresholds)
    ):
        raise ValueError("CLI thresholds differ from frozen calibration thresholds.")

    calibration_root = Path(calibration_run_dir).expanduser().resolve()
    calibration_values: list[float] = []
    quantile_csv = calibration_root / "distance_quantiles.csv"
    if quantile_csv.is_file():
        rows, _ = read_csv(quantile_csv)
        calibration_values = [
            value
            for row in rows
            if (value := _as_float(row.get("threshold"))) is not None
        ]
    else:
        config_path = calibration_root / "run_config.json"
        if config_path.is_file():
            calibration_values = _extract_threshold_values(read_json(config_path))
    if calibration_values and (
        len(calibration_values) != len(requested)
        or any(
            not math.isclose(
                left,
                right,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for left, right in zip(calibration_values, requested)
        )
    ):
        raise ValueError(
            "Calibration run thresholds differ from the frozen threshold list."
        )
    return {
        "threshold_source": "frozen_calibration",
        "ours_thresholds_json": str(threshold_path),
        "ours_thresholds_json_sha256": sha256_file(threshold_path),
        "theta_star_matches": True,
        "cost_cap_matches": True,
        "thresholds_match": True,
        "calibration_run_thresholds_checked": bool(calibration_values),
    }


def _table_row(
    metric: dict[str, Any],
    *,
    method_name: str,
    dataset: str,
    source_label: int,
    target_label: int,
    theta_star: float,
) -> dict[str, Any]:
    return {
        "method": method_name,
        "dataset": dataset,
        "source_label": int(source_label),
        "target_label": int(target_label),
        "k": int(metric["k"]),
        "theta": float(theta_star),
        "coverage": metric["close_cf_coverage"],
        "ccrcov": metric["close_cf_coverage"],
        "applicable_rate": metric["applicable_coverage"],
        "applicable_coverage": metric["applicable_coverage"],
        "any_strict_flip_coverage": metric["any_strict_flip_coverage"],
        "flip_rate_among_covered": metric["flip_rate_among_covered"],
        "avg_cf_drop_among_covered": metric["avg_cf_drop_among_covered"],
        "mean_cf_drop": metric["avg_cf_drop_among_covered"],
        "conditional_mean_cost": metric["conditional_mean_cost"],
        "conditional_median_cost": metric["conditional_median_cost"],
        "fixed_capped_mean_cost": metric["fixed_capped_mean_cost"],
        "fixed_capped_median_cost": metric["fixed_capped_median_cost"],
        "coverage_redundancy": metric["coverage_redundancy"],
        "structural_redundancy": metric["structural_redundancy"],
        "num_test_parents": metric["num_parents"],
        "num_candidates": int(metric["k"]),
        "selected_variant": "frozen_fullgraph_rank",
    }


def _check_run_semantics(
    config: dict[str, Any],
    *,
    forbid_selection: bool,
    forbid_fitting: bool,
) -> None:
    cf_mode = _deep_find(config, {"cf_mode"})
    if cf_mode is not None and _text(cf_mode) != "strict_flip":
        raise ValueError(f"Test run is not strict_flip: cf_mode={cf_mode!r}")
    if forbid_selection:
        preselected = _deep_find(config, {"candidate_set_preselected"})
        selection = _deep_find(config, {"selection_performed_in_eval"})
        if preselected is None or not _as_bool(preselected):
            raise ValueError("Test run does not declare preselected candidates.")
        if selection is None or _as_bool(selection):
            raise ValueError("Test evaluator performed candidate selection.")
    if forbid_fitting:
        source = _deep_find(config, {"threshold_source"})
        if source is None:
            raise ValueError("Test run does not declare its threshold_source.")
        if _text(source).lower() in {
            "auto",
            "auto_quantile",
            "test",
            "test_quantile",
        }:
            raise ValueError(f"Test thresholds were fitted in test run: {source!r}")


def export_final_artifacts(
    *,
    test_run_dir: str | Path,
    calibration_run_dir: str | Path,
    frozen_candidates_csv: str | Path,
    ours_schema_root: str | Path,
    output_dir: str | Path,
    method_name: str,
    dataset: str,
    source_label: int,
    target_label: int,
    test_job_id: str,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
    k_values: Sequence[int],
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
    forbid_selection: bool,
    forbid_fitting: bool,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not thresholds:
        raise ValueError("At least one frozen threshold is required.")
    calibration_root = Path(calibration_run_dir).expanduser().resolve()
    if not calibration_root.is_dir():
        raise FileNotFoundError(
            f"Calibration provenance directory does not exist: {calibration_root}"
        )
    frozen_thresholds = [float(value) for value in thresholds]
    if any(not math.isfinite(value) for value in frozen_thresholds):
        raise ValueError("Thresholds must all be finite.")
    threshold_provenance = validate_frozen_threshold_provenance(
        ours_schema_root=ours_schema_root,
        calibration_run_dir=calibration_root,
        theta_star=theta_star,
        cost_cap=cost_cap,
        thresholds=frozen_thresholds,
    )
    if not any(
        math.isclose(value, float(theta_star), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
        for value in frozen_thresholds
    ):
        raise ValueError("theta_star must be present in the frozen threshold list.")
    requested_k = sorted(set(int(value) for value in k_values))
    if requested_k != list(range(1, int(expected_candidate_count) + 1)):
        raise ValueError(
            f"k_values must be 1..{expected_candidate_count}; got={requested_k}"
        )

    pair_path, official_path, config_path = locate_test_inputs(test_run_dir)
    details, detail_fields = read_csv(pair_path)
    strict_mismatches = []
    for row in details:
        if row.get("cf_flip") in (None, ""):
            continue
        expected = _strict_flip(
            row,
            source_label=int(source_label),
            target_label=int(target_label),
        )
        if _as_bool(row.get("cf_flip")) != expected:
            strict_mismatches.append(
                (_text(row.get("parent_id")), _text(row.get("candidate_id")))
            )
    if strict_mismatches:
        raise ValueError(
            "pair_details cf_flip does not match strict source-to-target flip; "
            f"sample={strict_mismatches[:5]}"
        )
    official_rows, _ = read_csv(official_path)
    run_config = read_json(config_path)
    _check_run_semantics(
        run_config,
        forbid_selection=forbid_selection,
        forbid_fitting=forbid_fitting,
    )
    candidates, candidate_fields = load_ranked_candidates(
        frozen_candidates_csv,
        expected_count=expected_candidate_count,
    )
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=expected_parent_count,
        expected_pair_count=expected_pair_count,
    )

    prefix_metrics, threshold_metrics, parent_best_rows = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parent_ids,
        thresholds=frozen_thresholds,
        theta_star=float(theta_star),
        cost_cap=float(cost_cap),
        source_label=int(source_label),
        target_label=int(target_label),
        method_name=method_name,
    )
    k20_rows = [
        row for row in threshold_metrics if int(row["k"]) == expected_candidate_count
    ]
    expected_theta_count = (
        69
        if method_name == "GlobalGCE-Frequency-Top20"
        and int(expected_parent_count) == 217
        else None
    )
    reconstruction = reconstruct_official_summary(
        recomputed_k20=k20_rows,
        official_rows=official_rows,
        thresholds=frozen_thresholds,
        theta_star=float(theta_star),
        expected_theta_star_covered=expected_theta_count,
    )

    ordered_details = sorted(
        details,
        key=lambda row: (
            parent_ids.index(_text(row.get("parent_id"))),
            int(next(
                item["rank"]
                for item in candidates
                if item["candidate_id"] == _text(row.get("candidate_id"))
            )),
        ),
    )
    by_k = {int(row["k"]): row for row in prefix_metrics}
    figure4_rows = [
        row
        for row in threshold_metrics
        if int(row["k"]) in {10, int(expected_candidate_count)}
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        selected_fields = list(candidate_fields)
        for field in ("rank", "candidate_id", "candidate_smiles"):
            if field not in selected_fields:
                selected_fields.append(field)
        _write_csv(temp / "selected_top20.csv", candidates, selected_fields)
        _write_csv(temp / "test_pair_details.csv", ordered_details, detail_fields)
        _write_csv(temp / "test_threshold_summary.csv", k20_rows)
        _write_csv(temp / "parent_best_distances.csv", parent_best_rows)
        _write_csv(temp / "prefix_metrics.csv", prefix_metrics)
        _write_json(temp / "prefix_metrics.json", {"prefix_metrics": prefix_metrics})
        _write_csv(temp / "figure3_coverage_vs_k.csv", prefix_metrics)
        _write_csv(temp / "figure4_coverage_vs_threshold.csv", figure4_rows)
        for k in (10, int(expected_candidate_count)):
            table_row = _table_row(
                by_k[k],
                method_name=method_name,
                dataset=dataset,
                source_label=source_label,
                target_label=target_label,
                theta_star=theta_star,
            )
            fields = _resolve_table_fields(Path(ours_schema_root).resolve(), k)
            slug = "globalgce" if "globalgce" in method_name.lower() else method_name.lower().replace(" ", "_")
            _write_csv(temp / f"table2_{slug}_k{k}.csv", [table_row], fields)

        k10 = by_k[10]
        k20 = by_k[int(expected_candidate_count)]
        summary = {
            "method": method_name,
            "dataset": dataset,
            "source_label": int(source_label),
            "target_label": int(target_label),
            "test_parent_count": len(parent_ids),
            "candidate_count": len(candidates),
            "pair_count": len(details),
            "complete_cartesian": True,
            "theta_star": float(theta_star),
            "cost_cap": float(cost_cap),
            "thresholds": frozen_thresholds,
            "threshold_provenance": threshold_provenance,
            "k10_ccrcov_theta_star": float(k10["close_cf_coverage"]),
            "k20_ccrcov_theta_star": float(k20["close_cf_coverage"]),
            "k10_conditional_mean_cost": k10["conditional_mean_cost"],
            "k10_conditional_median_cost": k10["conditional_median_cost"],
            "k10_fixed_capped_mean_cost": k10["fixed_capped_mean_cost"],
            "k10_fixed_capped_median_cost": k10["fixed_capped_median_cost"],
            "k20_conditional_mean_cost": k20["conditional_mean_cost"],
            "k20_conditional_median_cost": k20["conditional_median_cost"],
            "k20_fixed_capped_mean_cost": k20["fixed_capped_mean_cost"],
            "k20_fixed_capped_median_cost": k20["fixed_capped_median_cost"],
            "k10_coverage_redundancy": k10["coverage_redundancy"],
            "k10_structural_redundancy": k10["structural_redundancy"],
            "k20_coverage_redundancy": k20["coverage_redundancy"],
            "k20_structural_redundancy": k20["structural_redundancy"],
            "selection_used_calibration": False,
            "selection_used_test": False,
            "threshold_fitted_on_test": False,
            "test_used_for_selection": False,
            "official_summary_reconstruction_passed": True,
            "run_complete": True,
        }
        _write_json(temp / "summary.json", summary)
        _write_json(
            temp / "official_summary_reconstruction_audit.json",
            reconstruction,
        )
        repo_root = Path(__file__).resolve().parents[2]
        manifest = {
            "test_job_id": str(test_job_id),
            "generation_input_split": "train",
            "candidate_selection_source": "train_only_frozen_candidates",
            "candidate_selection_performed": False,
            "selection_used_calibration": False,
            "selection_used_test": False,
            "threshold_fitted_on_test": False,
            "test_used_for_selection": False,
            "test_parent_count": len(parent_ids),
            "candidate_count": len(candidates),
            "pair_count": len(details),
            "theta_star": float(theta_star),
            "cost_cap": float(cost_cap),
            "thresholds": frozen_thresholds,
            "threshold_provenance": threshold_provenance,
            "candidate_csv": str(Path(frozen_candidates_csv).resolve()),
            "candidate_csv_sha256": sha256_file(frozen_candidates_csv),
            "pair_details": str(pair_path),
            "pair_details_sha256": sha256_file(pair_path),
            "official_threshold_summary": str(official_path),
            "official_threshold_summary_sha256": sha256_file(official_path),
            "calibration_run_dir": str(calibration_root),
            "ours_schema_root": str(Path(ours_schema_root).resolve()),
            "teacher": _inherited_file_identity(
                run_config,
                {"teacher_path", "teacher_model_path"},
                {"teacher_sha256", "teacher_hash"},
            ),
            "molclr_checkpoint": _inherited_file_identity(
                run_config,
                {"molclr_checkpoint", "molclr_ckpt"},
                {"molclr_checkpoint_sha256", "molclr_checkpoint_hash"},
            ),
            "git_commit": _git_commit(repo_root),
            "forbid_selection": bool(forbid_selection),
            "forbid_fitting": bool(forbid_fitting),
        }
        _write_json(temp / "run_manifest.json", manifest)
        final_audit = audit_final_artifacts(
            run_dir=temp,
            frozen_candidates_csv=frozen_candidates_csv,
            ours_schema_root=ours_schema_root,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            expected_pair_count=expected_pair_count,
            theta_star=theta_star,
            cost_cap=cost_cap,
            thresholds=frozen_thresholds,
            check_manifest=False,
        )
        _write_json(temp / "final_artifact_audit.json", final_audit)
        artifact_hashes = {
            path.relative_to(temp).as_posix(): sha256_file(path)
            for path in sorted(temp.rglob("*"))
            if path.is_file()
        }
        _write_json(
            temp / "artifact_manifest.json",
            {
                "files": artifact_hashes,
                "file_count": len(artifact_hashes),
                "all_hashes_generated": True,
                "self_excluded": "artifact_manifest.json",
                "finalization_marker_excluded": "_FINALIZED.json",
            },
        )
        _write_json(
            temp / "_FINALIZED.json",
            {
                "finalized": True,
                "artifact_manifest_sha256": sha256_file(
                    temp / "artifact_manifest.json"
                ),
                "official_summary_reconstruction_passed": True,
                "final_artifact_audit_passed": True,
            },
        )
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return summary


def audit_final_artifacts(
    *,
    run_dir: str | Path,
    frozen_candidates_csv: str | Path,
    ours_schema_root: str | Path,
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
    check_manifest: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    required = (
        "selected_top20.csv",
        "test_pair_details.csv",
        "test_threshold_summary.csv",
        "parent_best_distances.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "summary.json",
        "run_manifest.json",
        "official_summary_reconstruction_audit.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Final artifact files missing: {missing}")
    candidates, _ = load_ranked_candidates(
        root / "selected_top20.csv",
        expected_count=expected_candidate_count,
    )
    frozen, _ = load_ranked_candidates(
        frozen_candidates_csv,
        expected_count=expected_candidate_count,
    )
    if [row["candidate_id"] for row in candidates] != [
        row["candidate_id"] for row in frozen
    ]:
        raise ValueError("Exported candidate order differs from frozen rank order.")
    details, _ = read_csv(root / "test_pair_details.csv")
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=expected_parent_count,
        expected_pair_count=expected_pair_count,
    )
    prefix, _ = read_csv(root / "prefix_metrics.csv")
    if [_as_int(row.get("k")) for row in prefix] != list(
        range(1, expected_candidate_count + 1)
    ):
        raise ValueError("Figure 3/prefix metrics do not contain K=1..20.")
    previous_coverage = -math.inf
    previous_cost = math.inf
    for row in prefix:
        coverage = float(row["close_cf_coverage"])
        capped = float(row["fixed_capped_mean_cost"])
        if coverage + FLOAT_TOLERANCE < previous_coverage:
            raise ValueError("Prefix CCRCov decreases with K.")
        if capped > previous_cost + FLOAT_TOLERANCE:
            raise ValueError("Fixed capped mean cost increases with K.")
        previous_coverage = coverage
        previous_cost = capped
    summary = read_json(root / "summary.json")
    recomputed_prefix, recomputed_thresholds, _ = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parent_ids,
        thresholds=thresholds,
        theta_star=theta_star,
        cost_cap=cost_cap,
        source_label=int(summary["source_label"]),
        target_label=int(summary["target_label"]),
        method_name=str(summary["method"]),
    )
    recomputed_by_k = {int(row["k"]): row for row in recomputed_prefix}
    metric_fields = (
        *OFFICIAL_FIELDS,
        "num_applicable_parents",
        "applicable_coverage",
        "num_any_strict_flip_parents",
        "any_strict_flip_coverage",
        "conditional_mean_cost",
        "conditional_median_cost",
        "fixed_capped_mean_cost",
        "fixed_capped_median_cost",
        "coverage_redundancy",
        "structural_redundancy",
    )
    for stored in prefix:
        k = int(stored["k"])
        recomputed = recomputed_by_k[k]
        mismatched = [
            field
            for field in metric_fields
            if not _same_value(stored.get(field), recomputed.get(field), field)
        ]
        if mismatched:
            raise ValueError(
                f"Prefix metric reconstruction failed for K={k}: {mismatched}"
            )
    figure4, _ = read_csv(root / "figure4_coverage_vs_threshold.csv")
    expected_figure4 = 2 * len(thresholds)
    if len(figure4) != expected_figure4:
        raise ValueError(
            f"Figure 4 rows={len(figure4)} != expected={expected_figure4}."
        )
    k_values = {_as_int(row.get("k")) for row in figure4}
    if k_values != {10, expected_candidate_count}:
        raise ValueError(f"Figure 4 K values are invalid: {k_values}")
    expected_figure4_rows = [
        row
        for row in recomputed_thresholds
        if int(row["k"]) in {10, expected_candidate_count}
    ]
    for stored, recomputed in zip(figure4, expected_figure4_rows):
        if int(stored["k"]) != int(recomputed["k"]) or not math.isclose(
            float(stored["threshold"]),
            float(recomputed["threshold"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError("Figure 4 row ordering or frozen threshold changed.")
        mismatched = [
            field
            for field in OFFICIAL_FIELDS
            if not _same_value(stored.get(field), recomputed.get(field), field)
        ]
        if mismatched:
            raise ValueError(
                f"Figure 4 metric reconstruction failed: {mismatched}"
            )
    reconstruction = read_json(root / "official_summary_reconstruction_audit.json")
    if not _as_bool(reconstruction.get("official_summary_reconstruction_passed")):
        raise ValueError("Official summary reconstruction did not pass.")
    if _as_bool(summary.get("selection_used_test")):
        raise ValueError("Final artifacts declare test candidate selection.")
    if _as_bool(summary.get("threshold_fitted_on_test")):
        raise ValueError("Final artifacts declare test threshold fitting.")
    if not math.isclose(
        float(summary["theta_star"]),
        float(theta_star),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ) or not math.isclose(
        float(summary["cost_cap"]),
        float(cost_cap),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("Frozen theta_star or cost_cap changed.")
    for k in (10, expected_candidate_count):
        table_candidates = list(root.glob(f"table2_*_k{k}.csv"))
        if len(table_candidates) != 1:
            raise ValueError(f"Expected one Table 2 artifact for K={k}.")
        _, fields = read_csv(table_candidates[0])
        reference_fields = _resolve_table_fields(Path(ours_schema_root), k)
        if fields != reference_fields:
            raise ValueError(f"Table 2 schema mismatch for K={k}.")
    manifest_verified = None
    if check_manifest:
        finalized = read_json(root / "_FINALIZED.json")
        if not _as_bool(finalized.get("finalized")):
            raise ValueError("Run is not finalized.")
        artifact_manifest = read_json(root / "artifact_manifest.json")
        for relative, digest in artifact_manifest.get("files", {}).items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != str(digest):
                raise ValueError(f"Artifact hash mismatch: {relative}")
        if sha256_file(root / "artifact_manifest.json") != str(
            finalized.get("artifact_manifest_sha256")
        ):
            raise ValueError("artifact_manifest.json hash mismatch.")
        run_manifest = read_json(root / "run_manifest.json")
        threshold_provenance = run_manifest.get("threshold_provenance") or {}
        threshold_source = Path(
            str(threshold_provenance.get("ours_thresholds_json") or "")
        ).expanduser()
        if (
            not threshold_source.is_file()
            or sha256_file(threshold_source)
            != str(threshold_provenance.get("ours_thresholds_json_sha256"))
        ):
            raise ValueError("Frozen threshold provenance hash mismatch.")
        for path_field, hash_field in (
            ("candidate_csv", "candidate_csv_sha256"),
            ("pair_details", "pair_details_sha256"),
            (
                "official_threshold_summary",
                "official_threshold_summary_sha256",
            ),
        ):
            source_path = Path(str(run_manifest[path_field])).expanduser()
            if source_path.is_file() and sha256_file(source_path) != str(
                run_manifest[hash_field]
            ):
                raise ValueError(f"Source provenance hash mismatch: {path_field}")
        official_source = Path(
            str(run_manifest["official_threshold_summary"])
        ).expanduser()
        if official_source.is_file():
            official_rows, _ = read_csv(official_source)
            exported_k20, _ = read_csv(root / "test_threshold_summary.csv")
            reconstruct_official_summary(
                recomputed_k20=exported_k20,
                official_rows=official_rows,
                thresholds=thresholds,
                theta_star=theta_star,
                expected_theta_star_covered=(
                    69
                    if str(summary["method"])
                    == "GlobalGCE-Frequency-Top20"
                    and int(expected_parent_count) == 217
                    else None
                ),
            )
        manifest_verified = True
    return {
        "final_artifact_audit_passed": True,
        "parent_count": len(parent_ids),
        "candidate_count": len(candidates),
        "pair_count": len(details),
        "complete_cartesian": True,
        "candidate_order_frozen": True,
        "top10_is_rank_1_to_10": True,
        "top20_is_rank_1_to_20": True,
        "coverage_monotonic_nondecreasing": True,
        "fixed_capped_cost_monotonic_nonincreasing": True,
        "prefix_metrics_recomputed": True,
        "figure4_metrics_recomputed": True,
        "test_selection": False,
        "test_threshold_fitting": False,
        "manifest_hashes_verified": manifest_verified,
    }


__all__ = [
    "FLOAT_TOLERANCE",
    "OFFICIAL_FIELDS",
    "TABLE_REQUIRED_FIELDS",
    "audit_final_artifacts",
    "compute_prefix_artifacts",
    "export_final_artifacts",
    "load_ranked_candidates",
    "locate_test_inputs",
    "reconstruct_official_summary",
    "sha256_file",
    "summarize_wnode_thresholds",
    "validate_frozen_threshold_provenance",
    "validate_complete_cartesian",
]
