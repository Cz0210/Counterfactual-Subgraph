#!/usr/bin/env python3
"""Greedily select full-graph candidates by strict-flip Node-FGW coverage.

The input ``pair_details.csv`` may contain several evaluation methods.  This
selector intentionally consumes only rows whose ``method`` exactly matches
``--method-name`` (``globalgce`` by default), so selected GlobalGCE molecules
cannot inherit coverage from ours selected-subgraph rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METHOD = "globalgce"
DEFAULT_QUANTILE = 0.20
MISSING_ORDER = 10**18


@dataclass
class CandidateCoverage:
    """One full-graph candidate and its best close-flip distance per parent."""

    candidate_id: str
    candidate_smiles: str
    original_order: int = MISSING_ORDER
    first_detail_order: int = MISSING_ORDER
    coverage_distances: dict[str, float] = field(default_factory=dict)
    flip_close_pair_count: int = 0


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def _as_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "t", "yes", "y", "on"}


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def _candidate_id(row: dict[str, Any], row_index: int) -> str:
    for field_name in ("candidate_id", "id", "rank", "candidate_index", "index"):
        value = _text(row.get(field_name))
        if value:
            return value
    return str(row_index)


def _quantile(values: array, quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a threshold quantile from zero valid distances.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"threshold quantile must be within [0, 1], got {quantile}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _read_candidate_pool(
    path: Path,
    *,
    candidate_smiles_col: str,
) -> tuple[dict[str, CandidateCoverage], dict[str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate CSV does not exist: {path}")
    candidates: dict[str, CandidateCoverage] = {}
    ids_by_smiles: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or candidate_smiles_col not in reader.fieldnames:
            raise ValueError(
                f"Candidate CSV must contain {candidate_smiles_col!r}; "
                f"available columns: {reader.fieldnames or []}"
            )
        for row_index, row in enumerate(reader):
            smiles = _text(row.get(candidate_smiles_col))
            if not smiles:
                continue
            candidate_id = _candidate_id(row, row_index)
            existing = candidates.get(candidate_id)
            if existing is None:
                candidates[candidate_id] = CandidateCoverage(
                    candidate_id=candidate_id,
                    candidate_smiles=smiles,
                    original_order=row_index,
                )
            ids_by_smiles.setdefault(smiles, candidate_id)
    return candidates, ids_by_smiles, len(candidates)


def _resolve_candidate(
    *,
    row: dict[str, Any],
    row_index: int,
    candidates: dict[str, CandidateCoverage],
    ids_by_smiles: dict[str, str],
    candidate_smiles_col: str,
    pool_candidate_count: int,
) -> CandidateCoverage | None:
    smiles = _text(row.get(candidate_smiles_col))
    if not smiles:
        return None
    raw_id = _candidate_id(row, row_index)
    candidate_id = raw_id if raw_id in candidates else ids_by_smiles.get(smiles, raw_id)
    candidate = candidates.get(candidate_id)
    if candidate is None:
        candidate = CandidateCoverage(
            candidate_id=candidate_id,
            candidate_smiles=smiles,
            original_order=pool_candidate_count + row_index,
        )
        candidates[candidate_id] = candidate
        ids_by_smiles.setdefault(smiles, candidate_id)
    elif not candidate.candidate_smiles:
        candidate.candidate_smiles = smiles
    candidate.first_detail_order = min(candidate.first_detail_order, row_index)
    return candidate


def _read_distance_quantile(pair_details: Path, requested_quantile: float) -> tuple[float | None, Path | None]:
    """Prefer the evaluator's saved threshold when its requested quantile exists."""

    candidates = (
        pair_details.parent / "distance_quantiles.csv",
        pair_details.parent.parent / "distance_quantiles.csv",
    )
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                quantile = _as_float(row.get("quantile"))
                threshold = _as_float(row.get("threshold"))
                if (
                    quantile is not None
                    and threshold is not None
                    and math.isclose(quantile, requested_quantile, rel_tol=0.0, abs_tol=1e-12)
                ):
                    return threshold, path
    return None, None


def _scan_pair_details(
    *,
    pair_details: Path,
    method_name: str,
    candidates: dict[str, CandidateCoverage],
    ids_by_smiles: dict[str, str],
    candidate_smiles_col: str,
    pool_candidate_count: int,
    threshold: float | None,
) -> dict[str, Any]:
    """Stream the detail table once; optionally populate close-flip coverage."""

    total_rows = 0
    method_rows = 0
    valid_distance_rows = 0
    rows_with_candidate_smiles = 0
    rows_missing_parent_id = 0
    parent_ids: set[str] = set()
    distances = array("d")
    flip_close_pairs = 0
    with pair_details.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"method", "parent_id", "candidate_id", candidate_smiles_col, "distance", "cf_flip"}
        missing = sorted(column for column in required_columns if column not in (reader.fieldnames or []))
        if missing:
            raise ValueError(
                "pair_details.csv is missing required columns: "
                f"{missing}; available columns: {reader.fieldnames or []}"
            )
        for row_index, row in enumerate(reader):
            total_rows += 1
            if _text(row.get("method")) != method_name:
                continue
            method_rows += 1
            parent_id = _text(row.get("parent_id"))
            if parent_id:
                parent_ids.add(parent_id)
            else:
                rows_missing_parent_id += 1
            candidate = _resolve_candidate(
                row=row,
                row_index=row_index,
                candidates=candidates,
                ids_by_smiles=ids_by_smiles,
                candidate_smiles_col=candidate_smiles_col,
                pool_candidate_count=pool_candidate_count,
            )
            if candidate is None:
                continue
            rows_with_candidate_smiles += 1
            distance = _as_float(row.get("distance"))
            if distance is None:
                continue
            valid_distance_rows += 1
            distances.append(distance)
            if threshold is None or not parent_id or not _as_bool(row.get("cf_flip")) or distance > threshold:
                continue
            flip_close_pairs += 1
            candidate.flip_close_pair_count += 1
            previous = candidate.coverage_distances.get(parent_id)
            if previous is None or distance < previous:
                candidate.coverage_distances[parent_id] = distance
    return {
        "num_rows_total": total_rows,
        "num_rows_after_method_filter": method_rows,
        "num_rows_with_candidate_smiles": rows_with_candidate_smiles,
        "num_valid_distance_rows": valid_distance_rows,
        "num_rows_missing_parent_id": rows_missing_parent_id,
        "parent_ids": parent_ids,
        "distances": distances,
        "num_flip_close_pairs": flip_close_pairs,
    }


def _candidate_sort_key(candidate: CandidateCoverage, covered: set[str]) -> tuple[float, float, int, int, str]:
    marginal_distances = [
        distance for parent_id, distance in candidate.coverage_distances.items() if parent_id not in covered
    ]
    marginal_gain = len(marginal_distances)
    marginal_mean = _mean(marginal_distances)
    return (
        -float(marginal_gain),
        float(marginal_mean) if marginal_mean is not None else float("inf"),
        len(candidate.candidate_smiles),
        min(candidate.original_order, candidate.first_detail_order),
        candidate.candidate_id,
    )


def _greedy_select(
    candidates: list[CandidateCoverage],
    *,
    top_k: int,
    num_parents: int,
    threshold: float,
    method_name: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    remaining = list(candidates)
    for rank in range(1, min(top_k, len(remaining)) + 1):
        chosen = min(remaining, key=lambda candidate: _candidate_sort_key(candidate, covered))
        remaining.remove(chosen)
        new_parent_ids = set(chosen.coverage_distances) - covered
        marginal_distances = [chosen.coverage_distances[parent_id] for parent_id in new_parent_ids]
        covered.update(new_parent_ids)
        all_distances = list(chosen.coverage_distances.values())
        selected.append(
            {
                "rank": rank,
                "candidate_id": chosen.candidate_id,
                "candidate_smiles": chosen.candidate_smiles,
                "marginal_coverage": len(new_parent_ids),
                "cumulative_coverage": len(covered),
                "cumulative_coverage_rate": float(len(covered) / num_parents) if num_parents else 0.0,
                "covered_parent_count": len(chosen.coverage_distances),
                "mean_distance_on_covered": _mean(all_distances),
                "mean_distance_on_marginal_covered": _mean(marginal_distances),
                "min_distance": min(all_distances) if all_distances else None,
                "flip_close_pair_count": chosen.flip_close_pair_count,
                "method": method_name,
                "threshold": threshold,
                "original_candidate_order": min(chosen.original_order, chosen.first_detail_order),
            }
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Accepted for consistency with existing Slurm wrappers; selector behavior does not use them.
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pair-details", required=True, help="Node-FGW details/pair_details.csv")
    parser.add_argument("--candidates-csv", required=True, help="Original GlobalGCE fullgraph candidate pool CSV")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float, default=None)
    threshold_group.add_argument("--threshold-quantile", type=float, default=None)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--candidate-smiles-col", default="candidate_smiles")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("[ERROR] --top-k must be positive.")
    if args.threshold is not None and (not math.isfinite(args.threshold) or args.threshold < 0.0):
        raise SystemExit("[ERROR] --threshold must be a finite non-negative value.")

    pair_details = Path(args.pair_details).expanduser().resolve()
    candidates_csv = Path(args.candidates_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not pair_details.is_file():
        raise SystemExit(f"[ERROR] pair details CSV does not exist: {pair_details}")

    requested_quantile = DEFAULT_QUANTILE if args.threshold is None and args.threshold_quantile is None else args.threshold_quantile
    if requested_quantile is not None and not 0.0 <= requested_quantile <= 1.0:
        raise SystemExit("[ERROR] --threshold-quantile must be within [0, 1].")

    print("[FGW_FULLGRAPH_SELECTOR_CONFIG]", flush=True)
    print(f"pair_details={pair_details}", flush=True)
    print(f"candidates_csv={candidates_csv}", flush=True)
    print(f"method_name={args.method_name}", flush=True)
    print(f"top_k={args.top_k}", flush=True)
    print(f"threshold={args.threshold}", flush=True)
    print(f"threshold_quantile={requested_quantile}", flush=True)

    candidates, ids_by_smiles, pool_candidate_count = _read_candidate_pool(
        candidates_csv,
        candidate_smiles_col=args.candidate_smiles_col,
    )
    first_scan = _scan_pair_details(
        pair_details=pair_details,
        method_name=args.method_name,
        candidates=candidates,
        ids_by_smiles=ids_by_smiles,
        candidate_smiles_col=args.candidate_smiles_col,
        pool_candidate_count=pool_candidate_count,
        threshold=None,
    )
    if first_scan["num_rows_after_method_filter"] == 0:
        raise SystemExit(
            f"[ERROR] No rows with method={args.method_name!r} were found in {pair_details}. "
            "The selector refuses to mix methods."
        )
    if first_scan["num_valid_distance_rows"] == 0:
        raise SystemExit("[ERROR] No finite distances remain after the required method/candidate filtering.")

    quantile_source_path: Path | None = None
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "explicit"
    else:
        threshold, quantile_source_path = _read_distance_quantile(pair_details, float(requested_quantile))
        if threshold is None:
            threshold = _quantile(first_scan["distances"], float(requested_quantile))
            threshold_source = "computed_quantile"
        else:
            threshold_source = "distance_quantiles_csv"

    # The second stream stores only close strict-flip rows, which is bounded by
    # candidate x parent coverage rather than every raw pair-detail row.
    for candidate in candidates.values():
        candidate.coverage_distances.clear()
        candidate.flip_close_pair_count = 0
    second_scan = _scan_pair_details(
        pair_details=pair_details,
        method_name=args.method_name,
        candidates=candidates,
        ids_by_smiles=ids_by_smiles,
        candidate_smiles_col=args.candidate_smiles_col,
        pool_candidate_count=pool_candidate_count,
        threshold=float(threshold),
    )

    num_parents = len(first_scan["parent_ids"])
    selected = _greedy_select(
        list(candidates.values()),
        top_k=int(args.top_k),
        num_parents=num_parents,
        threshold=float(threshold),
        method_name=args.method_name,
    )
    coverage_rows = []
    for candidate in sorted(
        candidates.values(),
        key=lambda value: (min(value.original_order, value.first_detail_order), value.candidate_id),
    ):
        distances = list(candidate.coverage_distances.values())
        coverage_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_smiles": candidate.candidate_smiles,
                "covered_parent_count": len(candidate.coverage_distances),
                "coverage_rate": float(len(candidate.coverage_distances) / num_parents) if num_parents else 0.0,
                "mean_distance_on_covered": _mean(distances),
                "min_distance": min(distances) if distances else None,
                "flip_close_pair_count": candidate.flip_close_pair_count,
                "method": args.method_name,
                "threshold": float(threshold),
                "original_candidate_order": min(candidate.original_order, candidate.first_detail_order),
            }
        )

    selected_for_eval = [
        {
            "candidate_smiles": row["candidate_smiles"],
            "method": "GlobalGCE",
            "fullgraph_method": "globalgce_selected20",
            "rank": row["rank"],
            "candidate_id": row["candidate_id"],
        }
        for row in selected
    ]
    final_coverage = int(selected[-1]["cumulative_coverage"]) if selected else 0
    summary = {
        "method_name": args.method_name,
        "top_k": int(args.top_k),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "threshold_quantile": requested_quantile,
        "distance_quantiles_csv": str(quantile_source_path) if quantile_source_path else None,
        "num_rows_total": int(first_scan["num_rows_total"]),
        "num_rows_after_method_filter": int(first_scan["num_rows_after_method_filter"]),
        "num_rows_with_candidate_smiles": int(first_scan["num_rows_with_candidate_smiles"]),
        "num_valid_distance_rows": int(first_scan["num_valid_distance_rows"]),
        "num_rows_missing_parent_id": int(first_scan["num_rows_missing_parent_id"]),
        "num_candidates_from_input_csv": int(pool_candidate_count),
        "num_candidates": len(candidates),
        "num_parents": num_parents,
        "num_flip_close_pairs": int(second_scan["num_flip_close_pairs"]),
        "selected_count": len(selected),
        "final_cumulative_coverage": final_coverage,
        "final_cumulative_coverage_rate": float(final_coverage / num_parents) if num_parents else 0.0,
        "input_pair_details": str(pair_details),
        "input_candidates_csv": str(candidates_csv),
        "candidate_smiles_col": args.candidate_smiles_col,
        "coverage_definition": "cf_flip == true and distance <= threshold",
        "selection_policy": "greedy_maximum_marginal_coverage",
        "tie_break_order": [
            "lower_mean_distance_on_marginal_covered_parents",
            "shorter_candidate_smiles",
            "earlier_original_candidate_order",
        ],
    }
    _write_csv(
        out_dir / "coverage_by_candidate.csv",
        coverage_rows,
        [
            "candidate_id",
            "candidate_smiles",
            "covered_parent_count",
            "coverage_rate",
            "mean_distance_on_covered",
            "min_distance",
            "flip_close_pair_count",
            "method",
            "threshold",
            "original_candidate_order",
        ],
    )
    _write_csv(
        out_dir / "selected_top20.csv",
        selected,
        [
            "rank",
            "candidate_id",
            "candidate_smiles",
            "marginal_coverage",
            "cumulative_coverage",
            "cumulative_coverage_rate",
            "covered_parent_count",
            "mean_distance_on_covered",
            "mean_distance_on_marginal_covered",
            "min_distance",
            "flip_close_pair_count",
            "method",
            "threshold",
            "original_candidate_order",
        ],
    )
    _write_csv(
        out_dir / "selected_top20_for_eval.csv",
        selected_for_eval,
        ["candidate_smiles", "method", "fullgraph_method", "rank", "candidate_id"],
    )
    _write_json(out_dir / "selector_summary.json", summary)

    print("[FGW_FULLGRAPH_SELECTOR_DONE]", flush=True)
    print(f"selected_count={len(selected)}", flush=True)
    print(f"final_cumulative_coverage={final_coverage}", flush=True)
    print(f"final_cumulative_coverage_rate={summary['final_cumulative_coverage_rate']:.6f}", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
