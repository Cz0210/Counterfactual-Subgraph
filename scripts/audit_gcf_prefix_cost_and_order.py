#!/usr/bin/env python3
"""Audit GCFExplainer-HIVCSV prefix costs and candidate-order preservation.

This tool only reads existing candidate and pair-detail artifacts. It never
loads MolCLR, computes FGW distances, or mutates evaluator outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RANK_FIELDS = ("rank", "selected_rank", "selection_rank", "candidate_rank")
SMILES_FIELDS = ("candidate_smiles", "canonical_smiles", "smiles", "fragment_smiles")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "ok"}


def _as_float(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_float_or_default(value: Any, default: float) -> float:
    """Parse a finite float while preserving a genuine zero value."""

    parsed = _as_float(value)
    return float(default) if parsed is None else parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None or not float(parsed).is_integer():
        return None
    return int(parsed)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonicalize_smiles(smiles: str) -> str:
    text = _text(smiles)
    if not text:
        return ""
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(text)
        if molecule is not None:
            return Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        pass
    return text


def _candidate_smiles(row: dict[str, Any]) -> str:
    return next((_text(row.get(field)) for field in SMILES_FIELDS if _text(row.get(field))), "")


def _candidate_id(row: dict[str, Any], row_index: int) -> str:
    return _text(row.get("candidate_id") or row.get("id") or row.get("graph_hash")) or f"row_{row_index}"


def _ranked_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    rank_field = next(
        (field for field in RANK_FIELDS if any(_text(row.get(field)) for row in rows)),
        None,
    )
    ranked: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        rank = _as_int(row.get(rank_field)) if rank_field else index + 1
        if rank is None:
            raise ValueError(f"Invalid {rank_field!r} in candidate row {index + 2}")
        row["_audit_rank"] = rank
        row["_audit_row_order"] = index + 1
        row["_audit_candidate_id"] = _candidate_id(row, index)
        row["_audit_smiles"] = _candidate_smiles(row)
        row["_audit_canonical_smiles"] = canonicalize_smiles(row["_audit_smiles"])
        ranked.append(row)
    if rank_field:
        ranked.sort(key=lambda row: int(row["_audit_rank"]))
    return ranked, rank_field or "row_order"


def _sanitize_ok(row: dict[str, Any]) -> bool:
    if "sanitize_ok" in row and _text(row.get("sanitize_ok")):
        return _as_bool(row.get("sanitize_ok"))
    if "convert_ok" in row and _text(row.get("convert_ok")):
        return _as_bool(row.get("convert_ok"))
    return bool(_candidate_smiles(row))


def valid_rows_preserving_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter invalid converted rows without sorting the survivors."""

    return [dict(row) for row in rows if _sanitize_ok(row) and _candidate_smiles(row)]


def _normalize_method(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def filter_method_rows(
    rows: Sequence[dict[str, Any]], requested_method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = [_text(row.get("method")) for row in rows if _text(row.get("method"))]
    unique_methods = list(dict.fromkeys(methods))
    if not methods:
        return list(rows), {"requested_method": requested_method, "matched_methods": [], "mode": "no_method_column"}
    requested = _normalize_method(requested_method)
    exact = [row for row in rows if _normalize_method(row.get("method")) == requested]
    if exact:
        matched = sorted({_text(row.get("method")) for row in exact})
        return exact, {"requested_method": requested_method, "matched_methods": matched, "mode": "normalized_exact"}
    aliases = [
        method
        for method in unique_methods
        if requested and (requested in _normalize_method(method) or _normalize_method(method) in requested)
    ]
    if aliases:
        selected = [row for row in rows if _text(row.get("method")) in set(aliases)]
        return selected, {"requested_method": requested_method, "matched_methods": aliases, "mode": "normalized_contains"}
    if len(unique_methods) == 1:
        return list(rows), {
            "requested_method": requested_method,
            "matched_methods": unique_methods,
            "mode": "single_method_fallback",
        }
    raise ValueError(f"Cannot uniquely match method={requested_method!r}; available={unique_methods}")


@dataclass(frozen=True)
class PrefixCandidate:
    rank: int
    candidate_id: str
    canonical_smiles: str


@dataclass
class PairAudit:
    parents: tuple[str, ...]
    candidates: tuple[PrefixCandidate, ...]
    strict_distance: dict[tuple[str, str], float]
    raw_distance: dict[tuple[str, str], float]
    row_counts: Counter[tuple[str, str]]
    rows_by_pair: set[tuple[str, str]]
    nan_rows_by_candidate: Counter[str]
    unmatched_rows: int
    strict_source_counts: Counter[str]


def _strict_flip(row: dict[str, Any]) -> tuple[bool, str]:
    label = _as_int(row.get("label"))
    pred_before = _as_int(row.get("pred_before"))
    pred_after = _as_int(row.get("pred_after"))
    if label is not None and pred_before is not None and pred_after is not None:
        return pred_before == label and pred_after != label, "recomputed_teacher_strict"
    if _text(row.get("teacher_strict_flip")):
        return _as_bool(row.get("teacher_strict_flip")), "teacher_strict_flip"
    return _as_bool(row.get("cf_flip")), "cf_flip_fallback"


def _candidate_for_pair_row(
    row: dict[str, Any],
    *,
    by_id: dict[str, PrefixCandidate],
    by_smiles: dict[str, PrefixCandidate],
) -> PrefixCandidate | None:
    candidate = by_id.get(_text(row.get("candidate_id")))
    if candidate is not None:
        return candidate
    smiles = _candidate_smiles(row)
    return by_smiles.get(canonicalize_smiles(smiles)) if smiles else None


def aggregate_pair_details(
    rows: Sequence[dict[str, Any]], candidates: Sequence[PrefixCandidate]
) -> PairAudit:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    by_smiles = {
        candidate.canonical_smiles: candidate for candidate in candidates if candidate.canonical_smiles
    }
    strict_distance: dict[tuple[str, str], float] = {}
    raw_distance: dict[tuple[str, str], float] = {}
    row_counts: Counter[tuple[str, str]] = Counter()
    nan_rows_by_candidate: Counter[str] = Counter()
    strict_sources: Counter[str] = Counter()
    parents: set[str] = set()
    unmatched = 0
    for row in rows:
        parent_id = _text(row.get("parent_id"))
        if not parent_id:
            continue
        parents.add(parent_id)
        candidate = _candidate_for_pair_row(row, by_id=by_id, by_smiles=by_smiles)
        if candidate is None:
            unmatched += 1
            continue
        key = (parent_id, candidate.candidate_id)
        row_counts[key] += 1
        distance = _as_float(row.get("distance"))
        if distance is None:
            nan_rows_by_candidate[candidate.candidate_id] += 1
            continue
        previous_raw = raw_distance.get(key)
        if previous_raw is None or distance < previous_raw:
            raw_distance[key] = distance
        strict, source = _strict_flip(row)
        strict_sources[source] += 1
        if strict:
            previous = strict_distance.get(key)
            if previous is None or distance < previous:
                strict_distance[key] = distance
    return PairAudit(
        parents=tuple(sorted(parents)),
        candidates=tuple(candidates),
        strict_distance=strict_distance,
        raw_distance=raw_distance,
        row_counts=row_counts,
        rows_by_pair=set(row_counts),
        nan_rows_by_candidate=nan_rows_by_candidate,
        unmatched_rows=unmatched,
        strict_source_counts=strict_sources,
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    left, right = ordered[middle - 1], ordered[middle]
    if math.isinf(left) or math.isinf(right):
        return math.inf if left > 0 or right > 0 else -math.inf
    return (left + right) / 2.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index or lower == upper:
        return lower
    if math.isinf(lower) or math.isinf(upper):
        return upper if math.isinf(upper) else lower
    weight = position - lower_index
    return lower * (1.0 - weight) + upper * weight


def _mean_all(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    if any(math.isinf(value) for value in values):
        return math.inf
    return sum(values) / len(values)


def compute_prefix_metrics(pair_audit: PairAudit, *, theta: float, max_k: int) -> list[dict[str, Any]]:
    parents = pair_audit.parents
    strict_best = {parent: math.inf for parent in parents}
    raw_best = {parent: math.inf for parent in parents}
    rows: list[dict[str, Any]] = []
    previous_strict = dict(strict_best)
    previous_raw = dict(raw_best)
    for candidate in pair_audit.candidates[: int(max_k)]:
        for parent in parents:
            key = (parent, candidate.candidate_id)
            if key in pair_audit.strict_distance:
                strict_best[parent] = min(strict_best[parent], pair_audit.strict_distance[key])
            if key in pair_audit.raw_distance:
                raw_best[parent] = min(raw_best[parent], pair_audit.raw_distance[key])
        if any(strict_best[parent] > previous_strict[parent] for parent in parents):
            raise AssertionError("Strict best distance increased under a nested candidate prefix.")
        if any(raw_best[parent] > previous_raw[parent] for parent in parents):
            raise AssertionError("Raw best distance increased under a nested candidate prefix.")
        previous_strict = dict(strict_best)
        previous_raw = dict(raw_best)
        all_strict = list(strict_best.values())
        all_raw = list(raw_best.values())
        finite_strict = [value for value in all_strict if math.isfinite(value)]
        covered = [value for value in all_strict if value <= float(theta)]
        prefix_ids = {item.candidate_id for item in pair_audit.candidates[: candidate.rank]}
        any_pair_parents = {
            parent for parent, candidate_id in pair_audit.rows_by_pair if candidate_id in prefix_ids
        }
        rows.append(
            {
                "K": candidate.rank,
                "num_parents_total": len(parents),
                "num_parents_with_any_pair": len(any_pair_parents),
                "num_strict_flip_parents": len(finite_strict),
                "num_theta_covered": len(covered),
                "coverage": len(covered) / len(parents) if parents else 0.0,
                "unconditional_median_best_distance_all_parents": _median(all_strict),
                "conditional_median_best_distance_theta_covered": _median(covered),
                "strict_flip_applicable_parent_median_distance": _median(finite_strict),
                "mean_best_distance_all_parents": _mean_all(all_strict),
                "p25_best_distance_all_parents": _quantile(all_strict, 0.25),
                "p75_best_distance_all_parents": _quantile(all_strict, 0.75),
                "num_missing_best_distance": sum(not math.isfinite(value) for value in all_strict),
                "num_nan_distance": sum(
                    count
                    for candidate_id, count in pair_audit.nan_rows_by_candidate.items()
                    if candidate_id in prefix_ids
                ),
                "num_duplicate_candidates": len(pair_audit.candidates)
                - len({item.canonical_smiles for item in pair_audit.candidates}),
                "raw_distance_unconditional_median_all_parents": _median(all_raw),
                "raw_distance_conditional_median_finite_parents": _median(
                    [value for value in all_raw if math.isfinite(value)]
                ),
                "raw_distance_num_missing": sum(not math.isfinite(value) for value in all_raw),
            }
        )
    return rows


def _nonincreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(right <= left + 1e-12 for left, right in zip(sequence, sequence[1:]))


def _nondecreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(right + 1e-12 >= left for left, right in zip(sequence, sequence[1:]))


def _identity_maps(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    by_id: dict[str, int] = {}
    by_canonical: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        candidate_id = _candidate_id(row, index - 1)
        canonical = canonicalize_smiles(_candidate_smiles(row))
        by_id.setdefault(candidate_id, index)
        if canonical:
            by_canonical.setdefault(canonical, index)
    return by_id, by_canonical


def _identity_row_maps(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(rows):
        row = dict(source)
        candidate_id = _candidate_id(row, index)
        canonical = canonicalize_smiles(_candidate_smiles(row))
        by_id.setdefault(candidate_id, row)
        if canonical:
            by_canonical.setdefault(canonical, row)
    return by_id, by_canonical


def _lookup_position(
    candidate_id: str,
    canonical: str,
    by_id: dict[str, int],
    by_canonical: dict[str, int],
) -> int | None:
    return by_id.get(candidate_id) or (by_canonical.get(canonical) if canonical else None)


def build_candidate_order_audit(
    candidate_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    converted_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    *,
    max_k: int,
    selected_pt_rows: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked_candidates, rank_source = _ranked_rows(candidate_rows)
    if len(ranked_candidates) < int(max_k):
        raise ValueError(f"Candidate CSV has {len(ranked_candidates)} rows, expected at least {max_k}.")
    prefix = ranked_candidates[: int(max_k)]
    if [int(row["_audit_rank"]) for row in prefix] != list(range(1, int(max_k) + 1)):
        raise ValueError("Candidate prefix ranks must be exactly 1..max_k.")

    ranked_selected, _ = _ranked_rows(selected_rows) if selected_rows else ([], "missing")
    valid_converted = valid_rows_preserving_order(converted_rows)
    selected_id, selected_canonical = _identity_maps(ranked_selected)
    selected_row_by_id, selected_row_by_canonical = _identity_row_maps(ranked_selected)
    selected_pt_id, selected_pt_canonical = _identity_maps(selected_pt_rows)
    selected_pt_row_by_id, selected_pt_row_by_canonical = _identity_row_maps(selected_pt_rows)
    valid_id, valid_canonical = _identity_maps(valid_converted)

    pair_first_rows: list[dict[str, Any]] = []
    seen_pair_candidates: set[str] = set()
    for row in pair_rows:
        candidate_id = _text(row.get("candidate_id"))
        canonical = canonicalize_smiles(_candidate_smiles(row))
        identity = candidate_id or canonical
        if identity and identity not in seen_pair_candidates:
            seen_pair_candidates.add(identity)
            pair_first_rows.append(row)
    pair_id, pair_canonical = _identity_maps(pair_first_rows)

    selected_positions: list[int | None] = []
    output: list[dict[str, Any]] = []
    for row in prefix:
        candidate_id = _text(row["_audit_candidate_id"])
        canonical = _text(row["_audit_canonical_smiles"])
        selected_rank = _lookup_position(candidate_id, canonical, selected_id, selected_canonical)
        selected_metadata = selected_row_by_id.get(candidate_id) or selected_row_by_canonical.get(
            canonical, {}
        )
        min_distance_value = row.get("min_distance_seen")
        if min_distance_value is None or min_distance_value == "":
            min_distance_value = selected_metadata.get("min_distance_seen")
        selected_pt_index = _lookup_position(
            candidate_id, canonical, selected_pt_id, selected_pt_canonical
        )
        selected_pt_record = selected_pt_row_by_id.get(
            candidate_id
        ) or selected_pt_row_by_canonical.get(canonical, {})
        selected_pt_hash_match = selected_pt_record.get("_audit_graph_hash_match")
        valid_rank = _lookup_position(candidate_id, canonical, valid_id, valid_canonical)
        pair_index = _lookup_position(candidate_id, canonical, pair_id, pair_canonical)
        prefix_rank = int(row["_audit_rank"])
        selected_positions.append(selected_rank)
        output.append(
            {
                "selected_rank": selected_rank,
                "selected_graphs_pt_index": selected_pt_index,
                "selected_graphs_pt_graph_hash_match": selected_pt_hash_match,
                "valid_filtered_rank": valid_rank,
                "prefix_rank": prefix_rank,
                "candidate_id": candidate_id,
                "graph_hash": _text(row.get("graph_hash") or selected_metadata.get("graph_hash")),
                "frequency": _as_int(row.get("frequency") or selected_metadata.get("frequency")),
                "min_distance_seen": finite_float_or_default(
                    min_distance_value,
                    999.0,
                ),
                "covered_count": _as_int(
                    row.get("covered_count")
                    or row.get("cumulative_covered_count")
                    or selected_metadata.get("covered_count")
                ),
                "sanitize_ok": _sanitize_ok(row),
                "canonical_smiles": canonical,
                "pair_details_candidate_index": pair_index,
                # Sanitization can create gaps in the original selected ranks.
                # It must preserve their relative order, while valid/pair ranks
                # must match the actual Top-K prefix exactly.
                "order_match": (
                    valid_rank == prefix_rank
                    and pair_index == prefix_rank
                    and (not selected_pt_rows or selected_pt_index == selected_rank)
                    and selected_pt_hash_match is not False
                ),
            }
        )
    candidate_ids = [row["candidate_id"] for row in output]
    canonicals = [row["canonical_smiles"] for row in output if row["canonical_smiles"]]
    graph_hashes = [row["graph_hash"] for row in output if row["graph_hash"]]
    selected_present = all(value is not None for value in selected_positions)
    selected_relative_order = selected_present and all(
        right > left
        for left, right in zip(
            (int(value) for value in selected_positions[:-1] if value is not None),
            (int(value) for value in selected_positions[1:] if value is not None),
        )
    )
    exact = selected_relative_order and all(row["order_match"] for row in output) and all(
        row["pair_details_candidate_index"] is not None for row in output
    )
    return output, {
        "rank_source": rank_source,
        "prefix_is_nested": [row["prefix_rank"] for row in output] == list(range(1, int(max_k) + 1)),
        "candidate_order_exact_match": exact,
        "selected_metadata_relative_order_preserved": selected_relative_order,
        "selected_graphs_pt_order_match": (
            all(row["selected_graphs_pt_index"] == row["selected_rank"] for row in output)
            if selected_pt_rows
            else None
        ),
        "selected_graphs_pt_graph_hash_match": (
            all(row["selected_graphs_pt_graph_hash_match"] is not False for row in output)
            if selected_pt_rows
            else None
        ),
        "duplicate_candidate_id_count": len(candidate_ids) - len(set(candidate_ids)),
        "canonical_smiles_duplicate_count": len(canonicals) - len(set(canonicals)),
        "graph_hash_duplicate_count": len(graph_hashes) - len(set(graph_hashes)),
        "num_valid_converted_rows": len(valid_converted),
        "top20_is_first_valid_prefix": all(
            row["valid_filtered_rank"] == row["prefix_rank"] for row in output
        ),
    }


def _load_optional(path_text: str | None) -> tuple[list[str], list[dict[str, str]]]:
    if not path_text:
        return [], []
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_csv(path)


def _load_selected_graph_records(path_text: str | None) -> tuple[list[dict[str, Any]], int | None]:
    if not path_text:
        return [], None
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Reading --selected-graphs requires torch in the audit environment.") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Selected graphs payload must be a dict: {path}")
    records = [dict(row) for row in payload.get("selected_records") or [] if isinstance(row, dict)]
    graphs = list(payload.get("selected_graphs") or [])
    if records and len(records) != len(graphs):
        raise ValueError(
            f"selected_records/selected_graphs length mismatch: records={len(records)} graphs={len(graphs)}"
        )
    for index, graph in enumerate(graphs):
        if index >= len(records):
            break
        try:
            hash_payload = {
                "x": graph.x.detach().cpu().numpy().tobytes().hex(),
                "edge_index": graph.edge_index.detach().cpu().numpy().tobytes().hex(),
                "num_nodes": int(graph.num_nodes),
            }
            computed = hashlib.sha1(
                json.dumps(hash_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
        except Exception:
            computed = ""
        recorded = _text(records[index].get("graph_hash"))
        records[index]["_audit_computed_graph_hash"] = computed
        records[index]["_audit_graph_hash_match"] = (
            computed == recorded if computed and recorded else None
        )
    return records, len(graphs)


def _find_plot_matches(
    figure3_csv: str | None,
    metrics: Sequence[dict[str, Any]],
    method: str,
) -> tuple[bool | None, bool | None]:
    if not figure3_csv:
        return None, None
    _, rows = _read_csv(Path(figure3_csv).expanduser().resolve())
    by_k = {int(row["K"]): row for row in metrics}
    conditional_match = True
    unconditional_match = True
    matched = 0
    for row in rows:
        if _normalize_method(row.get("method")) != _normalize_method(method):
            continue
        k = _as_int(row.get("k") or row.get("K"))
        plotted = _as_float(row.get("plotted_cost"))
        if k not in by_k or plotted is None:
            continue
        matched += 1
        conditional = by_k[k]["conditional_median_best_distance_theta_covered"]
        unconditional = by_k[k]["unconditional_median_best_distance_all_parents"]
        conditional_match &= math.isfinite(conditional) and math.isclose(plotted, conditional, abs_tol=1e-10)
        unconditional_match &= math.isfinite(unconditional) and math.isclose(plotted, unconditional, abs_tol=1e-10)
    return (conditional_match, unconditional_match) if matched else (None, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper compatibility; not read.")
    parser.add_argument("--set", action="append", default=[], help="Accepted for HPC wrapper compatibility.")
    parser.add_argument("--pair-details", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--selected-metadata", required=True)
    parser.add_argument("--selected-graphs", default=None, help="Optional selected_counterfactual_graphs.pt order audit.")
    parser.add_argument("--converted-smiles", required=True)
    parser.add_argument("--theta", type=float, default=0.0328)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--method", default="GCFExplainer")
    parser.add_argument("--figure3-csv", default=None, help="Optional existing report CSV for exact value comparison.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hpc/audits/gcf_prefix_cost_and_order",
    )
    return parser


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    pair_path = Path(args.pair_details).expanduser().resolve()
    candidate_path = Path(args.candidate_csv).expanduser().resolve()
    selected_path = Path(args.selected_metadata).expanduser().resolve()
    converted_path = Path(args.converted_smiles).expanduser().resolve()
    selected_graphs_arg = getattr(args, "selected_graphs", None)
    selected_graphs_path = (
        Path(selected_graphs_arg).expanduser().resolve() if selected_graphs_arg else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, all_pair_rows = _read_csv(pair_path)
    _, candidate_rows = _read_csv(candidate_path)
    _, selected_rows = _read_csv(selected_path)
    _, converted_rows = _read_csv(converted_path)
    selected_pt_rows, selected_pt_graph_count = _load_selected_graph_records(selected_graphs_arg)
    pair_rows, method_audit = filter_method_rows(all_pair_rows, args.method)

    order_rows, order_summary = build_candidate_order_audit(
        candidate_rows,
        selected_rows,
        converted_rows,
        pair_rows,
        max_k=args.max_k,
        selected_pt_rows=selected_pt_rows,
    )
    candidates = tuple(
        PrefixCandidate(
            rank=int(row["prefix_rank"]),
            candidate_id=str(row["candidate_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
        )
        for row in order_rows
    )
    pair_audit = aggregate_pair_details(pair_rows, candidates)
    metrics = compute_prefix_metrics(pair_audit, theta=float(args.theta), max_k=int(args.max_k))

    unconditional_values = [row["unconditional_median_best_distance_all_parents"] for row in metrics]
    coverage_values = [row["coverage"] for row in metrics]
    conditional_match, unconditional_match = _find_plot_matches(
        args.figure3_csv, metrics, args.method
    )
    expected_pairs = len(pair_audit.parents) * len(candidates)
    duplicate_pair_rows = sum(max(0, count - 1) for count in pair_audit.row_counts.values())
    suspected: list[str] = []
    if not order_summary["candidate_order_exact_match"]:
        suspected.append("candidate_order_mismatch")
    if (
        order_summary["duplicate_candidate_id_count"]
        or order_summary["canonical_smiles_duplicate_count"]
        or order_summary["graph_hash_duplicate_count"]
    ):
        suspected.append("duplicate_candidates")
    if len(pair_audit.rows_by_pair) != expected_pairs:
        suspected.append("incomplete_fullgraph_parent_candidate_matrix")
    if not _nonincreasing(unconditional_values):
        suspected.append("unconditional_cost_not_monotone")
    if not _nondecreasing(coverage_values):
        suspected.append("coverage_not_monotone")
    suspected.append("paper_cost_vs_theta_covered_cost_definition_mismatch")

    summary = {
        "inputs": {
            "pair_details": str(pair_path),
            "candidate_csv": str(candidate_path),
            "selected_metadata": str(selected_path),
            "selected_graphs": str(selected_graphs_path) if selected_graphs_path else None,
            "converted_smiles": str(converted_path),
            "figure3_csv": str(Path(args.figure3_csv).expanduser().resolve()) if args.figure3_csv else None,
        },
        "theta": float(args.theta),
        "max_k": int(args.max_k),
        "method_filter": method_audit,
        "num_pair_rows_before_method_filter": len(all_pair_rows),
        "num_pair_rows_after_method_filter": len(pair_rows),
        "num_parents": len(pair_audit.parents),
        "num_candidates": len(candidates),
        "num_unique_parent_candidate_pairs": len(pair_audit.rows_by_pair),
        "num_duplicate_parent_candidate_rows": duplicate_pair_rows,
        "unmatched_pair_rows": pair_audit.unmatched_rows,
        "strict_flip_source_counts": dict(pair_audit.strict_source_counts),
        "unconditional_cost_monotonic_nonincreasing": _nonincreasing(unconditional_values),
        "per_parent_best_monotonic_nonincreasing": True,
        "coverage_monotonic_nondecreasing": _nondecreasing(coverage_values),
        "prefix_is_nested": order_summary["prefix_is_nested"],
        "candidate_order_exact_match": order_summary["candidate_order_exact_match"],
        "all_parents_have_fullgraph_pairs": len(pair_audit.rows_by_pair) == expected_pairs,
        "duplicate_candidate_count": order_summary["duplicate_candidate_id_count"],
        "canonical_smiles_duplicate_count": order_summary["canonical_smiles_duplicate_count"],
        "graph_hash_duplicate_count": order_summary["graph_hash_duplicate_count"],
        "selected_pt_graph_count": selected_pt_graph_count,
        "current_plot_matches_conditional_cost": conditional_match,
        "current_plot_matches_unconditional_cost": unconditional_match,
        "order_audit": order_summary,
        "suspected_bug_list": suspected,
        "distance_recomputed": False,
        "metric_definitions": {
            "paper_unconditional_cost": "median over all parents of prefix minimum strict-valid distance; unavailable parents are +inf",
            "theta_covered_conditional_cost": "median over parents whose prefix minimum strict-valid distance is <= theta",
            "raw_distance_diagnostic": "prefix minimum finite distance without the strict-flip requirement",
        },
    }

    metric_fields = list(metrics[0].keys()) if metrics else []
    _write_csv(output_dir / "prefix_metrics_audit.csv", metrics, metric_fields)
    order_fields = list(order_rows[0].keys()) if order_rows else []
    _write_csv(output_dir / "candidate_order_audit.csv", order_rows, order_fields)
    _write_json(output_dir / "audit_summary.json", summary)

    final_row = metrics[-1] if metrics else {}
    report = f"""# GCF prefix cost and order audit

Method filter: {method_audit}
Pair rows: {len(pair_rows)} (before filter: {len(all_pair_rows)})
Parents: {len(pair_audit.parents)}
Candidates: {len(candidates)}
Theta: {float(args.theta)}
Distance recomputed: false

## Metric result

Paper-style unconditional all-parent median is monotone non-increasing: {summary['unconditional_cost_monotonic_nonincreasing']}
Theta-covered conditional median changes its conditioning parent set at every K and is not required to be monotone.
Final K coverage: {final_row.get('coverage')}
Final K unconditional median: {final_row.get('unconditional_median_best_distance_all_parents')}
Final K theta-covered conditional median: {final_row.get('conditional_median_best_distance_theta_covered')}

## Order result

Prefix nested: {summary['prefix_is_nested']}
Candidate order exact match: {summary['candidate_order_exact_match']}
Full parent-candidate matrix present: {summary['all_parents_have_fullgraph_pairs']}
Candidate ID duplicates: {summary['duplicate_candidate_count']}
Canonical SMILES duplicates: {summary['canonical_smiles_duplicate_count']}
Graph-hash duplicates: {summary['graph_hash_duplicate_count']}

## Suspected issues

{chr(10).join('- ' + item for item in suspected)}
"""
    (output_dir / "audit_report.txt").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run_audit(args)
    print("[GCF_PREFIX_COST_ORDER_AUDIT_DONE]", flush=True)
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
