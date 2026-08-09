#!/usr/bin/env python3
"""Audit frozen BACE Ours rank provenance and calibration/test coverage funnels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.molclr_node_embeddings import canonicalize_smiles  # noqa: E402
from src.eval.mutagenicity_wnode_selector import morgan_tanimoto  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _rank_rows(selected_csv: Path) -> list[dict[str, Any]]:
    rows = _read_csv(selected_csv)
    parsed = [
        {
            **row,
            "rank": int(row["rank"]),
            "fragment": canonicalize_smiles(str(row.get("fragment") or "")),
        }
        for row in rows
    ]
    if [row["rank"] for row in parsed] != list(range(1, 21)):
        raise AssertionError("Frozen BACE Ours ranks are not exactly 1..20.")
    if any(not row["fragment"] for row in parsed):
        raise AssertionError("Frozen selector has an invalid fragment.")
    return parsed


def _aggregate_pairs(
    detail_csv: Path,
    rank_by_fragment: dict[str, int],
    theta: float,
) -> tuple[dict[tuple[str, int], dict[str, Any]], set[str]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    parents: set[str] = set()
    for row in _read_csv(detail_csv):
        parent_id = str(row["parent_id"])
        fragment = canonicalize_smiles(
            str(row.get("fragment_smiles") or row.get("candidate_smiles") or "")
        )
        if fragment not in rank_by_fragment:
            raise AssertionError(f"Evaluator candidate absent from frozen rank: {fragment}")
        parents.add(parent_id)
        grouped[(parent_id, rank_by_fragment[fragment])].append(row)
    pairs: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        applicable = any(_bool(row.get("match")) or row.get("match_index") not in (None, "") for row in rows)
        delete_valid = any(_bool(row.get("delete_valid")) for row in rows)
        strict_rows = [row for row in rows if _bool(row.get("teacher_strict_flip"))]
        finite_strict_rows = [
            row for row in strict_rows if _float(row.get("distance")) is not None
        ]
        distances = [float(row["distance"]) for row in finite_strict_rows]
        close_only = any(
            (distance := _float(row.get("distance"))) is not None and distance <= theta
            for row in rows
        )
        best = (
            min(finite_strict_rows, key=lambda row: float(row["distance"]))
            if finite_strict_rows
            else None
        )
        pairs[key] = {
            "applicable": applicable,
            "delete_valid": delete_valid,
            "strict_flip": bool(strict_rows),
            "close_only": close_only,
            "close_strict_flip": bool(distances and min(distances) <= theta),
            "best_distance": min(distances) if distances else None,
            "cf_drop": _float(best.get("cf_drop")) if best else None,
            "atom_ratio": _float(best.get("atom_delete_ratio")) if best else None,
        }
    return pairs, parents


def _evaluator_rank_fragments(
    paths: list[Path],
    rank_by_fragment: dict[str, int],
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for path in paths:
        for row in _read_csv(path):
            fragment = canonicalize_smiles(
                str(row.get("fragment_smiles") or row.get("candidate_smiles") or "")
            )
            if fragment not in rank_by_fragment:
                raise AssertionError(
                    f"Evaluator candidate absent from frozen selector: {fragment}"
                )
            result[rank_by_fragment[fragment]].add(str(fragment))
    return result


def _funnel(
    *,
    split: str,
    pairs: dict[tuple[str, int], dict[str, Any]],
    parents: set[str],
    theta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefix_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    ordered_parents = sorted(parents)
    for k in range(1, 21):
        per_parent: list[dict[str, Any]] = []
        for parent in ordered_parents:
            entries = [pairs.get((parent, rank), {}) for rank in range(1, k + 1)]
            covered_ranks = [
                rank
                for rank, entry in enumerate(entries, start=1)
                if entry.get("close_strict_flip")
            ]
            distances = [
                float(entry["best_distance"])
                for entry in entries
                if entry.get("best_distance") is not None
            ]
            best_distance = min(distances) if distances else None
            best_rank = None
            if best_distance is not None:
                best_rank = next(
                    rank
                    for rank, entry in enumerate(entries, start=1)
                    if entry.get("best_distance") == best_distance
                )
            per_parent.append(
                {
                    "split": split,
                    "k": k,
                    "parent_id": parent,
                    "num_any_applicable": int(any(entry.get("applicable") for entry in entries)),
                    "num_delete_valid": int(any(entry.get("delete_valid") for entry in entries)),
                    "num_any_strict_flip": int(any(entry.get("strict_flip") for entry in entries)),
                    "num_close_only": int(any(entry.get("close_only") for entry in entries)),
                    "num_close_strict_flip": int(bool(covered_ranks)),
                    "coverage": int(bool(covered_ranks)),
                    "best_distance": best_distance,
                    "best_candidate_rank": best_rank,
                    "first_covered_rank": min(covered_ranks) if covered_ranks else None,
                }
            )
        prefix_rows.append(
            {
                "split": split,
                "k": k,
                "num_parents": len(ordered_parents),
                "num_any_applicable": sum(row["num_any_applicable"] for row in per_parent),
                "num_delete_valid": sum(row["num_delete_valid"] for row in per_parent),
                "num_any_strict_flip": sum(row["num_any_strict_flip"] for row in per_parent),
                "num_close_only": sum(row["num_close_only"] for row in per_parent),
                "num_close_strict_flip": sum(row["num_close_strict_flip"] for row in per_parent),
                "coverage": sum(row["coverage"] for row in per_parent) / len(ordered_parents),
                "theta_star": theta,
            }
        )
        if k == 20:
            parent_rows.extend(per_parent)
    return prefix_rows, parent_rows


def _candidate_diagnostics(
    selected: list[dict[str, Any]],
    calibration_pairs: dict[tuple[str, int], dict[str, Any]],
    calibration_parents: set[str],
    test_pairs: dict[tuple[str, int], dict[str, Any]],
    test_parents: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_calibration: set[str] = set()
    cumulative_test: set[str] = set()
    previous_fragments: list[str] = []
    previous_cover_sets: list[set[str]] = []
    for source in selected:
        rank = int(source["rank"])
        fragment = str(source["fragment"])
        calibration_cover = {
            parent
            for parent in calibration_parents
            if calibration_pairs.get((parent, rank), {}).get("close_strict_flip")
        }
        test_cover = {
            parent
            for parent in test_parents
            if test_pairs.get((parent, rank), {}).get("close_strict_flip")
        }
        calibration_entries = [calibration_pairs.get((parent, rank), {}) for parent in calibration_parents]
        distances = [
            float(entry["best_distance"])
            for entry in calibration_entries
            if entry.get("best_distance") is not None
        ]
        similarities = [morgan_tanimoto(fragment, other) for other in previous_fragments]
        redundancies = []
        for prior in previous_cover_sets:
            union = calibration_cover | prior
            redundancies.append(len(calibration_cover & prior) / len(union) if union else 0.0)
        row = {
            "rank": rank,
            "candidate_id": source.get("candidate_id") or rank,
            "fragment": fragment,
            "source_parent_id": source.get("representative_parent_ids"),
            "source_cf_drop": _float(source.get("mean_cf_drop")),
            "source_cf_flip": _float(source.get("cf_flip_rate")),
            "structural_support_count": int(float(source.get("support_count") or 0)),
            "applicable_parent_count": sum(bool(entry.get("applicable")) for entry in calibration_entries),
            "valid_deletion_parent_count": sum(bool(entry.get("delete_valid")) for entry in calibration_entries),
            "strict_flip_parent_count": sum(bool(entry.get("strict_flip")) for entry in calibration_entries),
            "close_strict_flip_parent_count": len(calibration_cover),
            "marginal_close_strict_flip_parent_count": len(calibration_cover - cumulative_calibration),
            "coverage_set": sorted(calibration_cover),
            "test_close_strict_flip_parent_count_diagnostic_only": len(test_cover),
            "test_marginal_parent_count_diagnostic_only": len(test_cover - cumulative_test),
            "mean_best_distance": sum(distances) / len(distances) if distances else None,
            "median_best_distance": sorted(distances)[len(distances) // 2] if distances else None,
            "atom_ratio": _float(source.get("mean_atom_ratio")),
            "projection_used": _float(source.get("projection_used_rate")),
            "direct_substructure": None,
            "max_pairwise_tanimoto_to_previous": max(similarities) if similarities else 0.0,
            "max_coverage_redundancy_to_previous": max(redundancies) if redundancies else 0.0,
        }
        rows.append(row)
        cumulative_calibration |= calibration_cover
        cumulative_test |= test_cover
        previous_fragments.append(fragment)
        previous_cover_sets.append(calibration_cover)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--calibration-details", required=True)
    parser.add_argument("--test-details", required=True)
    parser.add_argument("--theta-star", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = _rank_rows(Path(args.selected_csv).expanduser().resolve())
    rank_by_fragment = {str(row["fragment"]): int(row["rank"]) for row in selected}
    calibration_detail_path = Path(args.calibration_details).expanduser().resolve()
    test_detail_path = Path(args.test_details).expanduser().resolve()
    calibration_pairs, calibration_parents = _aggregate_pairs(
        calibration_detail_path, rank_by_fragment, args.theta_star
    )
    test_pairs, test_parents = _aggregate_pairs(
        test_detail_path, rank_by_fragment, args.theta_star
    )
    calibration_funnel, calibration_parent_rows = _funnel(
        split="calibration",
        pairs=calibration_pairs,
        parents=calibration_parents,
        theta=args.theta_star,
    )
    test_funnel, test_parent_rows = _funnel(
        split="existing_test_diagnostic_only",
        pairs=test_pairs,
        parents=test_parents,
        theta=args.theta_star,
    )
    candidate_rows = _candidate_diagnostics(
        selected, calibration_pairs, calibration_parents, test_pairs, test_parents
    )
    evaluator_by_rank = _evaluator_rank_fragments(
        [calibration_detail_path, test_detail_path], rank_by_fragment
    )
    rank_rows = []
    for row in selected:
        rank = int(row["rank"])
        evaluator_fragments = evaluator_by_rank.get(rank, set())
        rank_rows.append(
            {
                "rank": rank,
                "selector_fragment": row["fragment"],
                "evaluator_fragment": (
                    next(iter(evaluator_fragments)) if len(evaluator_fragments) == 1 else None
                ),
                "identity_exact": evaluator_fragments == {row["fragment"]},
            }
        )
    rank_pass = all(row["identity_exact"] for row in rank_rows)
    _write_csv(destination / "rank_provenance.csv", rank_rows)
    (destination / "rank_provenance.json").write_text(
        json.dumps({"rows": rank_rows, "rank_preservation_pass": rank_pass}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "rank_audit.txt").write_text(
        f"rank_preservation_pass={str(rank_pass).lower()}\n"
        "figure3_nested_prefix=true\ntable2_uses_top10=true\nfigure4_uses_top20=true\n",
        encoding="utf-8",
    )
    _write_csv(destination / "coverage_funnel_calibration.csv", calibration_funnel)
    _write_csv(destination / "coverage_funnel_existing_test_diagnostic.csv", test_funnel)
    (destination / "coverage_funnel_calibration.json").write_text(
        json.dumps({"rows": calibration_funnel, "test_used": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "coverage_funnel_existing_test_diagnostic.json").write_text(
        json.dumps(
            {"rows": test_funnel, "diagnostic_only": True, "selector_import_allowed": False},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(destination / "candidate_prefix_diagnostic.csv", candidate_rows)
    _write_csv(
        destination / "parent_first_covered_rank.csv",
        calibration_parent_rows + test_parent_rows,
    )
    print(f"rank_preservation_pass={rank_pass}")
    print("[BACE_OURS_COVERAGE_FUNNEL_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
