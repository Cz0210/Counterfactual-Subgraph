#!/usr/bin/env python3
"""Generate GCFExplainer-style tables and curves from completed recourse runs.

This entrypoint is deliberately post-processing only. It reads existing
``pair_details.csv`` files and the externally selected candidate files recorded
in each run config. It never recomputes embeddings, FGW distances, teacher
predictions, or candidate rankings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    "Ours": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_ours_top20_lam05_final"
    ),
    "GlobalGCE": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_"
        "globalgce_frequency_top20_lam05_final"
    ),
    "CLEAR": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_"
        "clear_parent_frequency_top20_lam05_final"
    ),
    "GCFExplainer": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_"
        "gcfexplainer_top20_normalized_lam05_final"
    ),
}
DISPLAY_ORDER = tuple(DEFAULT_RUNS)
OURS_SMILES_FIELDS = (
    "fragment",
    "fragment_smiles",
    "final_fragment",
    "core_fragment",
    "selected_fragment",
    "subgraph_smiles",
    "smiles",
)
FULLGRAPH_SMILES_FIELDS = (
    "candidate_smiles",
    "canonical_smiles",
    "counterfactual_smiles",
    "cf_smiles",
    "graph_smiles",
    "smiles",
    "final_smiles",
)
TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off", ""}


@dataclass(frozen=True, slots=True)
class CandidateRank:
    rank: int
    candidate_id: str
    smiles: str
    canonical_identity: str
    row_index: int


@dataclass(frozen=True, slots=True)
class PairRecourse:
    parent_id: str
    candidate_id: str
    candidate_rank: int
    distance: float
    cf_drop: float | None


@dataclass(slots=True)
class MethodRun:
    display_name: str
    run_dir: Path
    method: str
    config: dict[str, Any]
    summary_rows: list[dict[str, str]]
    cache_stats: dict[str, Any]
    candidates: list[CandidateRank]
    candidate_path: Path
    rank_source: str
    selection_method: str
    parent_ids: tuple[str, ...]
    recourse_by_pair: dict[tuple[str, str], PairRecourse]
    num_detail_rows: int
    num_unique_parent_candidate_pairs: int
    num_valid_parent_candidate_pairs: int
    num_multi_match_parent_candidate_pairs: int


def _text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"none", "null", "nan"} else rendered


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = _text(value).lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _normalize_candidate_id(value: Any) -> str:
    text = _text(value)
    numeric = _as_float(text)
    if numeric is not None and numeric.is_integer():
        return str(int(numeric))
    return text


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_csv_value(row.get(field))
                    for field in fields
                }
            )


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _resolve_recorded_path(raw: Any, *, run_dir: Path) -> Path:
    value = _text(raw)
    if not value:
        raise ValueError(f"Run config in {run_dir} does not record a candidate path.")
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        marker = "counterfactual-subgraph/"
        if marker in path.as_posix():
            suffix = path.as_posix().split(marker, 1)[1]
            candidates.append(REPO_ROOT / suffix)
    else:
        candidates.extend((REPO_ROOT / path, run_dir / path, path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _candidate_file_from_directory(directory: Path, *, ours: bool) -> Path:
    names = (
        ("selected_subgraphs.csv", "selected_subgraphs.json")
        if ours
        else (
            "selected_top20_for_eval.csv",
            "valid_greedy_top20_smiles_for_fgw.csv",
            "selected_counterfactual_metadata.csv",
        )
    )
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    csv_candidates = sorted(directory.glob("*top20*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    raise FileNotFoundError(
        f"Could not resolve exactly one selected candidate file in {directory}; "
        f"checked={list(names)} candidates={[str(path) for path in csv_candidates]}"
    )


def _read_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return [dict(row) for row in _read_csv(path)[1]]
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported candidate ranking file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("selected_subgraphs", "selected_rows", "rows", "candidates"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported candidate JSON schema: {path}")


def _canonical_smiles_identity(smiles: str) -> str:
    normalized = _text(smiles)
    if not normalized:
        return ""
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(normalized)
        if molecule is not None:
            return str(Chem.MolToSmiles(molecule, canonical=True))
    except (ImportError, Exception):
        pass
    return normalized


def load_candidate_ranking(
    path: Path,
    *,
    ours: bool,
    expected_top_k: int = 20,
) -> tuple[list[CandidateRank], str]:
    """Load external candidate order without deriving any rank from distances."""

    if path.is_dir():
        path = _candidate_file_from_directory(path, ours=ours)
    rows = _read_candidate_rows(path)
    if len(rows) != int(expected_top_k):
        raise ValueError(
            f"Selected candidate file must contain exactly {expected_top_k} rows; "
            f"found {len(rows)}: {path}"
        )
    rank_present = any(_text(row.get("rank")) for row in rows)
    ranked: list[CandidateRank] = []
    smiles_fields = OURS_SMILES_FIELDS if ours else FULLGRAPH_SMILES_FIELDS
    for row_index, row in enumerate(rows):
        rank = _as_int(row.get("rank")) if rank_present else row_index + 1
        if rank is None:
            raise ValueError(f"Missing or invalid rank in selected candidate row {row_index}: {path}")
        smiles = next((_text(row.get(field)) for field in smiles_fields if _text(row.get(field))), "")
        if not smiles:
            raise ValueError(f"Missing candidate/fragment SMILES in row {row_index}: {path}")
        canonical = _text(row.get("canonical_smiles")) or _canonical_smiles_identity(smiles)
        candidate_id = _normalize_candidate_id(
            row.get("candidate_id")
            or row.get("id")
            or row.get("rank")
            or row.get("candidate_index")
            or row.get("index")
            or row_index
        )
        ranked.append(
            CandidateRank(
                rank=int(rank),
                candidate_id=candidate_id,
                smiles=smiles,
                canonical_identity=canonical,
                row_index=row_index,
            )
        )
    if rank_present:
        ranked.sort(key=lambda candidate: candidate.rank)
        rank_source = "rank"
    else:
        rank_source = "row_order"
    expected_ranks = list(range(1, int(expected_top_k) + 1))
    if [candidate.rank for candidate in ranked] != expected_ranks:
        raise ValueError(f"Candidate ranks must be exactly 1..{expected_top_k}: {path}")
    ids = [candidate.candidate_id for candidate in ranked]
    canonical = [candidate.canonical_identity for candidate in ranked]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Selected candidate IDs are not unique: {path}")
    if not all(canonical) or len(set(canonical)) != len(canonical):
        raise ValueError(f"Selected canonical SMILES/fragments are empty or duplicated: {path}")
    return ranked, rank_source


def _selector_name(config: dict[str, Any], display_name: str) -> str:
    audits = config.get("method_evaluation_audits")
    if isinstance(audits, dict):
        for payload in audits.values():
            if isinstance(payload, dict) and _text(payload.get("selection_method")):
                return _text(payload["selection_method"])
    value = _text(config.get("selection_method"))
    if value and value not in {"mixed", "not_preselected"}:
        return value
    defaults = {
        "Ours": "greedy_mmr_cov20",
        "GlobalGCE": "frequency_top20",
        "CLEAR": "parent_frequency",
        "GCFExplainer": "normalized_top20",
    }
    return defaults.get(display_name, "external_selector")


def _strict_flip_from_row(row: dict[str, Any], *, source: Path, row_number: int) -> bool:
    label = _as_int(row.get("label"))
    pred_before = _as_int(row.get("pred_before"))
    pred_after = _as_int(row.get("pred_after"))
    computed: bool | None = None
    if label is not None and pred_before is not None and pred_after is not None:
        computed = pred_before == label and pred_after != label
    teacher_field = _text(row.get("teacher_strict_flip"))
    cf_field = _text(row.get("cf_flip"))
    recorded: bool | None = None
    if teacher_field:
        recorded = _as_bool(teacher_field)
    elif cf_field:
        recorded = _as_bool(cf_field)
    if computed is not None and recorded is not None and computed != recorded:
        raise ValueError(
            "Strict-flip audit failed in "
            f"{source} row={row_number}: label={label} pred_before={pred_before} "
            f"pred_after={pred_after} recorded={recorded} computed={computed}"
        )
    if computed is not None:
        return computed
    return bool(recorded)


def aggregate_detail_rows(
    rows: Iterable[dict[str, Any]],
    *,
    candidates: Sequence[CandidateRank],
    source: Path = Path("pair_details.csv"),
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str], PairRecourse],
    dict[str, int],
    set[str],
]:
    """Collapse match-instance rows to strict-flip parent-candidate recourse."""

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    by_smiles = {candidate.canonical_identity: candidate for candidate in candidates}
    group_row_counts: dict[tuple[str, str], int] = {}
    group_has_finite: set[tuple[str, str]] = set()
    recourse: dict[tuple[str, str], PairRecourse] = {}
    parent_ids: set[str] = set()
    methods: set[str] = set()
    observed_candidate_ids: set[str] = set()
    detail_count = 0
    for row_number, row in enumerate(rows, start=2):
        detail_count += 1
        method = _text(row.get("method"))
        if method:
            methods.add(method)
        parent_id = _text(row.get("parent_id"))
        if not parent_id:
            raise ValueError(f"Missing parent_id in {source} row={row_number}")
        parent_ids.add(parent_id)
        raw_candidate_id = _normalize_candidate_id(row.get("candidate_id"))
        candidate = by_id.get(raw_candidate_id)
        if candidate is None:
            raw_smiles = _text(row.get("fragment_smiles")) or _text(row.get("candidate_smiles"))
            candidate = by_smiles.get(_canonical_smiles_identity(raw_smiles))
        if candidate is None:
            raise ValueError(
                f"Detail row references a candidate outside the external Top{len(candidates)}: "
                f"candidate_id={raw_candidate_id!r} source={source} row={row_number}"
            )
        observed_candidate_ids.add(candidate.candidate_id)
        key = (parent_id, candidate.candidate_id)
        group_row_counts[key] = group_row_counts.get(key, 0) + 1
        distance = _as_float(row.get("distance"))
        if distance is None:
            continue
        group_has_finite.add(key)
        if not _strict_flip_from_row(row, source=source, row_number=row_number):
            continue
        cf_drop = _as_float(row.get("cf_drop"))
        previous = recourse.get(key)
        if previous is None or distance < previous.distance:
            recourse[key] = PairRecourse(
                parent_id=parent_id,
                candidate_id=candidate.candidate_id,
                candidate_rank=candidate.rank,
                distance=distance,
                cf_drop=cf_drop,
            )
    missing = sorted(set(by_id) - observed_candidate_ids)
    if missing:
        raise ValueError(f"External selected candidates missing from pair details: {missing}; source={source}")
    audit = {
        "num_detail_rows": detail_count,
        "num_unique_parent_candidate_pairs": len(group_row_counts),
        "num_valid_parent_candidate_pairs": len(group_has_finite),
        "num_multi_match_parent_candidate_pairs": sum(
            count > 1 for count in group_row_counts.values()
        ),
    }
    return tuple(sorted(parent_ids, key=_natural_key)), recourse, audit, methods


def _natural_key(value: str) -> tuple[Any, ...]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _validate_external_selection(config: dict[str, Any], *, run_dir: Path, expected_top_k: int) -> None:
    if not _as_bool(config.get("candidate_set_preselected")):
        raise ValueError(f"Run is not marked candidate_set_preselected=true: {run_dir}")
    if _as_bool(config.get("selection_performed_in_eval")):
        raise ValueError(f"Run reports evaluator-side candidate selection: {run_dir}")
    configured_top_k = _as_int(config.get("preselected_topk"))
    if configured_top_k is not None and configured_top_k != int(expected_top_k):
        raise ValueError(
            f"Run preselected_topk={configured_top_k}, expected {expected_top_k}: {run_dir}"
        )
    if _text(config.get("cf_mode")) != "strict_flip":
        raise ValueError(f"Report requires cf_mode=strict_flip: {run_dir}")
    main_uses = _text(config.get("main_ccrcov_uses"))
    if main_uses and main_uses != "teacher_strict_flip":
        raise ValueError(f"Run does not use teacher-strict flip for main CCRCOV: {run_dir}")


def load_method_run(
    display_name: str,
    run_dir_like: str | Path,
    *,
    expected_top_k: int = 20,
    expected_num_parents: int = 1283,
) -> MethodRun:
    run_dir = Path(run_dir_like).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir.resolve()
    config = _read_json(run_dir / "run_config.json")
    cache_stats = _read_json(run_dir / "cache_stats.json")
    summary_fields, summary_rows = _read_csv(run_dir / "combined" / "combined_threshold_summary.csv")
    if not summary_rows:
        raise ValueError(f"Combined threshold summary is empty: {run_dir}")
    _validate_external_selection(config, run_dir=run_dir, expected_top_k=expected_top_k)
    ours = display_name.lower() == "ours"
    recorded = config.get("ours_selected_path") if ours else config.get("fullgraph_candidates_path")
    candidate_path = _resolve_recorded_path(recorded, run_dir=run_dir)
    candidates, rank_source = load_candidate_ranking(
        candidate_path,
        ours=ours,
        expected_top_k=expected_top_k,
    )
    detail_path = run_dir / "details" / "pair_details.csv"
    _fields, detail_rows = _read_csv(detail_path)
    parent_ids, recourse, audit, methods = aggregate_detail_rows(
        detail_rows,
        candidates=candidates,
        source=detail_path,
    )
    if len(methods) != 1:
        raise ValueError(f"Expected one evaluator method in {detail_path}, found {sorted(methods)}")
    method = next(iter(methods))
    if expected_num_parents > 0 and len(parent_ids) != int(expected_num_parents):
        raise ValueError(
            f"Expected {expected_num_parents} parents, found {len(parent_ids)}: {detail_path}"
        )
    summary_methods = {_text(row.get("method")) for row in summary_rows if _text(row.get("method"))}
    if summary_methods != {method}:
        raise ValueError(
            f"Summary/detail method mismatch: summary={sorted(summary_methods)} detail={method!r}"
        )
    if "num_candidates" in summary_fields:
        counts = {_as_int(row.get("num_candidates")) for row in summary_rows}
        if counts != {int(expected_top_k)}:
            raise ValueError(f"Summary candidate count is not exactly {expected_top_k}: {counts}")
    return MethodRun(
        display_name=display_name,
        run_dir=run_dir,
        method=method,
        config=config,
        summary_rows=summary_rows,
        cache_stats=cache_stats,
        candidates=candidates,
        candidate_path=candidate_path,
        rank_source=rank_source,
        selection_method=_selector_name(config, display_name),
        parent_ids=parent_ids,
        recourse_by_pair=recourse,
        num_detail_rows=audit["num_detail_rows"],
        num_unique_parent_candidate_pairs=audit["num_unique_parent_candidate_pairs"],
        num_valid_parent_candidate_pairs=audit["num_valid_parent_candidate_pairs"],
        num_multi_match_parent_candidate_pairs=audit["num_multi_match_parent_candidate_pairs"],
    )


def best_recourse_by_parent(
    run: MethodRun,
    *,
    k: int,
) -> tuple[dict[str, float], dict[str, float | None]]:
    if not 1 <= int(k) <= len(run.candidates):
        raise ValueError(f"K must be within 1..{len(run.candidates)}, got {k}")
    candidate_ids = {candidate.candidate_id for candidate in run.candidates if candidate.rank <= int(k)}
    distances = {parent_id: math.inf for parent_id in run.parent_ids}
    drops: dict[str, float | None] = {parent_id: None for parent_id in run.parent_ids}
    for (parent_id, candidate_id), recourse in run.recourse_by_pair.items():
        if candidate_id not in candidate_ids:
            continue
        if recourse.distance < distances[parent_id]:
            distances[parent_id] = recourse.distance
            drops[parent_id] = recourse.cf_drop
    return distances, drops


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


def compute_prefix_metrics(run: MethodRun, *, k: int, threshold: float) -> dict[str, Any]:
    distances, drops = best_recourse_by_parent(run, k=k)
    all_distances = list(distances.values())
    finite = [distance for distance in all_distances if math.isfinite(distance)]
    covered = [parent_id for parent_id, distance in distances.items() if distance <= float(threshold)]
    covered_drops = [drops[parent_id] for parent_id in covered if drops[parent_id] is not None]
    denominator = len(run.parent_ids)
    return {
        "method": run.display_name,
        "k": int(k),
        "theta": float(threshold),
        "coverage": len(covered) / denominator if denominator else 0.0,
        "num_covered": len(covered),
        "median_cost": _median(all_distances),
        "conditional_median_cost": _median(finite),
        "applicable_rate": len(finite) / denominator if denominator else 0.0,
        "mean_cf_drop_among_covered": (
            float(sum(covered_drops) / len(covered_drops)) if covered_drops else float("nan")
        ),
        "flip_rate_among_covered": 1.0 if covered else float("nan"),
    }


def compute_k_curve(run: MethodRun, *, threshold: float, max_k: int = 20) -> list[dict[str, Any]]:
    rows = [compute_prefix_metrics(run, k=k, threshold=threshold) for k in range(1, max_k + 1)]
    _assert_non_decreasing([row["coverage"] for row in rows], "coverage vs K", run.display_name)
    _assert_non_decreasing([row["applicable_rate"] for row in rows], "applicable rate vs K", run.display_name)
    _assert_non_increasing([row["median_cost"] for row in rows], "median cost vs K", run.display_name)
    return rows


def _assert_non_decreasing(values: Sequence[float], metric: str, method: str) -> None:
    if any(right + 1e-15 < left for left, right in zip(values, values[1:])):
        raise AssertionError(f"{metric} is not monotone non-decreasing for {method}: {values}")


def _assert_non_increasing(values: Sequence[float], metric: str, method: str) -> None:
    if any(right > left + 1e-15 for left, right in zip(values, values[1:])):
        raise AssertionError(f"{metric} is not monotone non-increasing for {method}: {values}")


def build_threshold_grid(raw: str | None, *, minimum: float, maximum: float, points: int) -> np.ndarray:
    if raw:
        grid = np.asarray([float(part.strip()) for part in raw.split(",") if part.strip()], dtype=float)
    else:
        if points < 2:
            raise ValueError("threshold-points must be at least 2")
        grid = np.linspace(float(minimum), float(maximum), int(points), dtype=float)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("Threshold grid must contain finite values.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("Threshold grid must be strictly increasing and duplicate-free.")
    return grid


def bootstrap_coverage_curve(
    best_distances: Sequence[float],
    thresholds: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
    bootstrap_indices: np.ndarray | None = None,
) -> list[dict[str, float]]:
    distances = np.asarray(best_distances, dtype=float)
    grid = np.asarray(thresholds, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("best_distances must be a non-empty vector")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if bootstrap_indices is None:
        rng = np.random.default_rng(int(seed))
        bootstrap_indices = rng.integers(
            0,
            distances.size,
            size=(int(bootstrap_samples), distances.size),
        )
    if bootstrap_indices.shape != (int(bootstrap_samples), distances.size):
        raise ValueError("bootstrap_indices shape does not match samples x parents")
    sampled = distances[bootstrap_indices]
    output: list[dict[str, float]] = []
    for threshold in grid:
        estimates = np.mean(sampled <= threshold, axis=1)
        output.append(
            {
                "threshold": float(threshold),
                "coverage": float(np.mean(distances <= threshold)),
                "mean": float(np.mean(estimates)),
                "lower": float(np.quantile(estimates, 0.025)),
                "upper": float(np.quantile(estimates, 0.975)),
            }
        )
    _assert_non_decreasing([row["coverage"] for row in output], "coverage vs threshold", "bootstrap curve")
    return output


def _threshold_at_coverage(best_distances: Sequence[float], target: float) -> float:
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"Coverage target must be in [0, 1], got {target}")
    required = int(math.ceil(target * len(best_distances)))
    if required == 0:
        return 0.0
    ordered = sorted(float(value) for value in best_distances if math.isfinite(float(value)))
    if len(ordered) < required:
        return float("nan")
    return ordered[required - 1]


def _auc_over_range(
    thresholds: Sequence[float],
    coverages: Sequence[float],
    *,
    lower: float,
    upper: float,
) -> float:
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(coverages, dtype=float)
    if lower < x[0] - 1e-15 or upper > x[-1] + 1e-15 or upper <= lower:
        raise ValueError(f"AUC interval [{lower}, {upper}] lies outside threshold grid.")
    inside = (x > lower) & (x < upper)
    xs = np.concatenate(([lower], x[inside], [upper]))
    ys = np.interp(xs, x, y)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(ys, xs))


def _table_rows(runs: Sequence[MethodRun], *, k: int, theta: float) -> list[dict[str, Any]]:
    return [
        {
            "Method": run.display_name,
            "K": int(k),
            "Theta": float(theta),
            "Coverage": metrics["coverage"],
            "Num covered": metrics["num_covered"],
            "Median cost": metrics["median_cost"],
            "Conditional median cost": metrics["conditional_median_cost"],
            "Applicable rate": metrics["applicable_rate"],
            "Mean CFDrop among covered": metrics["mean_cf_drop_among_covered"],
            "Flip rate among covered": metrics["flip_rate_among_covered"],
        }
        for run in runs
        for metrics in [compute_prefix_metrics(run, k=k, threshold=theta)]
    ]


def _display_number(value: Any, *, digits: int = 6) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = _as_float(value)
    if number is None:
        raw = _text(value)
        if raw.lower() in {"inf", "+inf", "infinity"} or value == math.inf:
            return "inf"
        return "NaN" if isinstance(value, float) and math.isnan(value) else raw
    return f"{number:.{digits}g}"


def _write_markdown_table(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display_number(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")


def _write_latex_table(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    alignment = "l" + "r" * (len(fields) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(_latex_escape(field) for field in fields) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(
            " & ".join(_latex_escape(_display_number(row.get(field))) for field in fields) + " \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prefixed(prefix: str, name: str) -> str:
    cleaned = prefix.strip().strip("_-")
    return f"{cleaned}_{name}" if cleaned else name


def _plot_figure3(
    rows: Sequence[dict[str, Any]],
    *,
    png: Path,
    pdf: Path,
    distance_label: str,
    inset_max_k: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Ours": "#1b1b1b", "GlobalGCE": "#d97706", "CLEAR": "#18864b", "GCFExplainer": "#2563a8"}
    markers = {"Ours": "o", "GlobalGCE": "s", "CLEAR": "^", "GCFExplainer": "D"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True)
    for method in DISPLAY_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        xs = [row["k"] for row in method_rows]
        axes[0].plot(xs, [row["coverage"] for row in method_rows], label=method, color=colors[method], marker=markers[method], linewidth=1.8, markersize=4)
        costs = [row["median_cost"] if math.isfinite(row["median_cost"]) else np.nan for row in method_rows]
        axes[1].plot(xs, costs, label=method, color=colors[method], marker=markers[method], linewidth=1.8, markersize=4)
    axes[0].set_ylabel("Coverage")
    axes[1].set_ylabel(f"Median cost ({distance_label})")
    axes[1].set_xlabel("Prefix K")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(1, 20)
        axis.set_xticks([1, 5, 10, 15, 20])
    axes[0].legend(ncol=2, frameon=False)
    if inset_max_k and 1 < int(inset_max_k) < 20:
        inset = axes[0].inset_axes([0.53, 0.12, 0.43, 0.42])
        for method in DISPLAY_ORDER:
            method_rows = [row for row in rows if row["method"] == method and row["k"] <= int(inset_max_k)]
            if method_rows:
                inset.plot([row["k"] for row in method_rows], [row["coverage"] for row in method_rows], color=colors[method], linewidth=1.2)
        inset.set_xlim(1, int(inset_max_k))
        inset.grid(True, alpha=0.2)
        inset.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_figure4(
    rows: Sequence[dict[str, Any]],
    *,
    png: Path,
    pdf: Path,
    distance_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Ours": "#1b1b1b", "GlobalGCE": "#d97706", "CLEAR": "#18864b", "GCFExplainer": "#2563a8"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in DISPLAY_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        x = np.asarray([row["threshold"] for row in method_rows])
        mean = np.asarray([row["mean"] for row in method_rows])
        lower = np.asarray([row["lower"] for row in method_rows])
        upper = np.asarray([row["upper"] for row in method_rows])
        ax.plot(x, mean, label=method, color=colors[method], linewidth=2.0)
        ax.fill_between(x, lower, upper, color=colors[method], alpha=0.16, linewidth=0)
    ax.set_xlabel(f"Absolute threshold ({distance_label})")
    ax.set_ylabel("Coverage")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _parse_run_specs(raw_specs: Sequence[str]) -> dict[str, str]:
    if not raw_specs:
        return dict(DEFAULT_RUNS)
    runs: dict[str, str] = {}
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"--run must use DISPLAY_NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path or name in runs:
            raise ValueError(f"Invalid or duplicate --run specification: {spec!r}")
        runs[name] = path
    return runs


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = _parse_run_specs(args.run)
    if len(run_specs) != 4:
        raise ValueError(f"Exactly four method runs are required, found {len(run_specs)}")
    runs = [
        load_method_run(
            name,
            path,
            expected_top_k=int(args.max_k),
            expected_num_parents=int(args.expected_num_parents),
        )
        for name, path in run_specs.items()
    ]
    reference_parents = runs[0].parent_ids
    for run in runs[1:]:
        if run.parent_ids != reference_parents:
            raise ValueError(
                f"Parent set mismatch between {runs[0].display_name} and {run.display_name}."
            )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table_fields = [
        "Method",
        "K",
        "Theta",
        "Coverage",
        "Num covered",
        "Median cost",
        "Conditional median cost",
        "Applicable rate",
        "Mean CFDrop among covered",
        "Flip rate among covered",
    ]
    table_rows = _table_rows(runs, k=int(args.k), theta=float(args.theta_star))
    table_base = _prefixed(args.table_prefix, "table2_gcf_style_fgw")
    _write_csv(output_dir / f"{table_base}.csv", table_rows, table_fields)
    _write_markdown_table(output_dir / f"{table_base}.md", table_rows, table_fields)
    _write_latex_table(output_dir / f"{table_base}.tex", table_rows, table_fields)

    k_rows: list[dict[str, Any]] = []
    for run in runs:
        for row in compute_k_curve(run, threshold=float(args.theta_star), max_k=int(args.max_k)):
            k_rows.append({"distance_label": args.distance_label, **row})
    figure3_fields = [
        "method",
        "distance_label",
        "k",
        "theta",
        "coverage",
        "num_covered",
        "median_cost",
        "conditional_median_cost",
        "applicable_rate",
        "mean_cf_drop_among_covered",
        "flip_rate_among_covered",
    ]
    figure3_base = _prefixed(args.table_prefix, "figure3_fgw_coverage_cost_vs_k")
    _write_csv(output_dir / f"{figure3_base}.csv", k_rows, figure3_fields)
    _plot_figure3(
        k_rows,
        png=output_dir / f"{figure3_base}.png",
        pdf=output_dir / f"{figure3_base}.pdf",
        distance_label=args.distance_label,
        inset_max_k=args.inset_max_k,
    )

    threshold_grid = build_threshold_grid(
        args.threshold_grid,
        minimum=float(args.threshold_min),
        maximum=float(args.threshold_max),
        points=int(args.threshold_points),
    )
    rng = np.random.default_rng(int(args.seed))
    bootstrap_indices = rng.integers(
        0,
        len(reference_parents),
        size=(int(args.bootstrap_samples), len(reference_parents)),
    )
    threshold_rows: list[dict[str, Any]] = []
    best_k10_by_method: dict[str, list[float]] = {}
    for run in runs:
        distances, _drops = best_recourse_by_parent(run, k=int(args.k))
        ordered_distances = [distances[parent_id] for parent_id in reference_parents]
        best_k10_by_method[run.display_name] = ordered_distances
        curve = bootstrap_coverage_curve(
            ordered_distances,
            threshold_grid,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
            bootstrap_indices=bootstrap_indices,
        )
        threshold_rows.extend(
            {
                "method": run.display_name,
                "distance_label": args.distance_label,
                "k": int(args.k),
                "bootstrap_samples": int(args.bootstrap_samples),
                "bootstrap_seed": int(args.seed),
                **row,
            }
            for row in curve
        )
    figure4_fields = [
        "method",
        "distance_label",
        "k",
        "threshold",
        "coverage",
        "mean",
        "lower",
        "upper",
        "bootstrap_samples",
        "bootstrap_seed",
    ]
    figure4_base = _prefixed(args.table_prefix, "figure4_fgw_coverage_vs_threshold")
    _write_csv(output_dir / f"{figure4_base}.csv", threshold_rows, figure4_fields)
    _plot_figure4(
        threshold_rows,
        png=output_dir / f"{figure4_base}.png",
        pdf=output_dir / f"{figure4_base}.pdf",
        distance_label=args.distance_label,
    )

    curve_summary: list[dict[str, Any]] = []
    for run in runs:
        method_curve = [row for row in threshold_rows if row["method"] == run.display_name]
        x = [row["threshold"] for row in method_curve]
        y = [row["coverage"] for row in method_curve]
        distances = best_k10_by_method[run.display_name]
        k20 = compute_prefix_metrics(run, k=int(args.max_k), threshold=float(args.theta_star))
        k10 = compute_prefix_metrics(run, k=int(args.k), threshold=float(args.theta_star))
        curve_summary.append(
            {
                "method": run.display_name,
                "distance_label": args.distance_label,
                "low_threshold_pauc": _auc_over_range(
                    x,
                    y,
                    lower=0.0,
                    upper=float(args.low_threshold_auc_max),
                ),
                "low_threshold_pauc_range": f"[0,{float(args.low_threshold_auc_max):.16g}]",
                "full_auc": _auc_over_range(
                    x,
                    y,
                    lower=0.0,
                    upper=float(args.threshold_max),
                ),
                "full_auc_range": f"[0,{float(args.threshold_max):.16g}]",
                "threshold_at_30pct_coverage": _threshold_at_coverage(distances, 0.30),
                "threshold_at_50pct_coverage": _threshold_at_coverage(distances, 0.50),
                "threshold_at_70pct_coverage": _threshold_at_coverage(distances, 0.70),
                "applicable_rate_at_k10": k10["applicable_rate"],
                "applicable_rate_at_k20": k20["applicable_rate"],
                "num_detail_rows": run.num_detail_rows,
                "num_unique_parent_candidate_pairs": run.num_unique_parent_candidate_pairs,
                "num_valid_parent_candidate_pairs": run.num_valid_parent_candidate_pairs,
                "num_multi_match_parent_candidate_pairs": run.num_multi_match_parent_candidate_pairs,
                "rank_source": run.rank_source,
                "selection_method": run.selection_method,
            }
        )
    curve_fields = [
        "method",
        "distance_label",
        "low_threshold_pauc",
        "low_threshold_pauc_range",
        "full_auc",
        "full_auc_range",
        "threshold_at_30pct_coverage",
        "threshold_at_50pct_coverage",
        "threshold_at_70pct_coverage",
        "applicable_rate_at_k10",
        "applicable_rate_at_k20",
        "num_detail_rows",
        "num_unique_parent_candidate_pairs",
        "num_valid_parent_candidate_pairs",
        "num_multi_match_parent_candidate_pairs",
        "rank_source",
        "selection_method",
    ]
    curve_base = _prefixed(args.table_prefix, "fgw_curve_summary")
    _write_csv(output_dir / f"{curve_base}.csv", curve_summary, curve_fields)

    audit = {
        "report_type": "gcf_style_recourse_postprocessing",
        "distance_label": args.distance_label,
        "table_prefix": args.table_prefix,
        "strict_flip_only": True,
        "candidate_ranking_uses_distance": False,
        "selection_performed_in_report": False,
        "expected_num_parents": int(args.expected_num_parents),
        "expected_top_k": int(args.max_k),
        "table_k": int(args.k),
        "theta_star": float(args.theta_star),
        "threshold_grid": threshold_grid.tolist(),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "runs": [
            {
                "display_name": run.display_name,
                "method": run.method,
                "run_dir": str(run.run_dir),
                "candidate_path": str(run.candidate_path),
                "rank_source": run.rank_source,
                "selection_method": run.selection_method,
                "candidate_set_preselected": run.config.get("candidate_set_preselected"),
                "selection_performed_in_eval": run.config.get("selection_performed_in_eval"),
                "num_candidates": len(run.candidates),
                "num_parents": len(run.parent_ids),
                "num_detail_rows": run.num_detail_rows,
                "num_unique_parent_candidate_pairs": run.num_unique_parent_candidate_pairs,
                "num_valid_parent_candidate_pairs": run.num_valid_parent_candidate_pairs,
                "num_multi_match_parent_candidate_pairs": run.num_multi_match_parent_candidate_pairs,
                "cache_stats": run.cache_stats,
                "candidate_order": [asdict(candidate) for candidate in run.candidates],
            }
            for run in runs
        ],
    }
    audit_path = output_dir / _prefixed(args.table_prefix, "gcf_style_report_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Accepted for shared HPC wrapper compatibility.")
    parser.add_argument("--set", action="append", default=[], help="Accepted for shared HPC wrapper compatibility.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="DISPLAY_NAME=PATH",
        help="Repeat exactly four times to replace the built-in final-run paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hpc/eval/reports/gcf_style_molclr_node_fgw_final",
    )
    parser.add_argument("--distance-label", default="MolCLR-Node-FGW")
    parser.add_argument("--table-prefix", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--expected-num-parents", type=int, default=1283)
    parser.add_argument("--theta-star", type=float, default=0.0545395671276376)
    parser.add_argument("--threshold-grid", default=None, help="Comma-separated absolute thresholds.")
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=0.0985011840378189)
    parser.add_argument("--threshold-points", type=int, default=101)
    parser.add_argument("--low-threshold-auc-max", type=float, default=0.0328363645853374)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inset-max-k", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= int(args.k) <= int(args.max_k):
        raise SystemExit("[ERROR] --k must be within 1..--max-k")
    print("[GCF_STYLE_REPORT_CONFIG]", flush=True)
    print(f"distance_label={args.distance_label}", flush=True)
    print(f"k={args.k}", flush=True)
    print(f"theta_star={args.theta_star}", flush=True)
    print(f"bootstrap_samples={args.bootstrap_samples}", flush=True)
    result = generate_report(args)
    print("[GCF_STYLE_REPORT_DONE]", flush=True)
    print(f"output_dir={result['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
