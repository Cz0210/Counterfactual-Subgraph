"""Calibration-only WNode-aware nested prefix selection.

This module reuses the production matrix and metric primitives.  It never
loads a test cohort and rejects GCFExplainer inputs at its public boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_paper_artifacts import load_bace_thresholds
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.mutagenicity_wnode_selector import (
    DEFAULT_PREFIX_WEIGHTS,
    DEFAULT_THRESHOLD_WEIGHTS,
    ChemistryData,
    MatrixData,
    ThresholdBundle,
    ThresholdLevel,
    build_candidate_chemistry,
    build_coverage_redundancy_matrix,
    compute_prefix_metrics,
    fixed_denominator_capped_cost,
    greedy_select,
    load_calibration_matrix,
    local_swap_search,
    optimize_insertion_order,
    single_threshold_coverage,
    weighted_multi_threshold_utility,
)

try:  # pragma: no cover - environment dependent
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


TOP_K = 20
TABLE_K = 10
FLOAT_TOLERANCE = 1e-12
VARIANT_NAMES = (
    "A0_current",
    "A1_coverage_theta",
    "A2_multi_threshold",
    "A3_multi_threshold_prefix",
    "A4_prefix_covred_swap",
)


@dataclass(frozen=True, slots=True)
class PrefixVariant:
    name: str
    mode: str
    lambda_table2: float = 0.0
    lambda_covred: float = 0.0
    lambda_struct: float = 0.0
    lambda_size: float = 0.0
    lambda_cost: float = 0.0
    insertion_reorder: bool = False
    local_swap: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, tuple, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _forbidden_path(path_like: str | Path) -> bool:
    tokens: set[str] = set()
    for part in Path(path_like).expanduser().parts:
        normalized = part.lower().replace("-", "_").replace(".", "_")
        tokens.update(token for token in normalized.split("_") if token)
    return "test" in tokens or "gcf" in tokens or "gcfexplainer" in tokens


def assert_calibration_selector_inputs(*paths: str | Path) -> None:
    for path in paths:
        if _forbidden_path(path):
            raise ValueError(f"Selector input may not reference test or GCF data: {path}")


def threshold_bundle_from_manifest(
    path: str | Path,
    *,
    finite_distance_count: int,
    weights: Sequence[float] = DEFAULT_THRESHOLD_WEIGHTS,
) -> ThresholdBundle:
    contract = load_bace_thresholds(path)
    quantiles = tuple(float(value) for value in contract["quantiles"])
    thresholds = tuple(float(value) for value in contract["thresholds"])
    parsed_weights = tuple(float(value) for value in weights)
    if len(parsed_weights) != len(thresholds):
        raise ValueError("Threshold weight count differs from frozen BACE grid.")
    labels = tuple(f"q{int(round(value * 100)):02d}" for value in quantiles)
    grouped: dict[float, dict[str, Any]] = {}
    order: list[float] = []
    for quantile, label, threshold, weight in zip(
        quantiles, labels, thresholds, parsed_weights, strict=True
    ):
        if threshold not in grouped:
            grouped[threshold] = {"weight": 0.0, "quantiles": [], "labels": []}
            order.append(threshold)
        grouped[threshold]["weight"] += weight
        grouped[threshold]["quantiles"].append(quantile)
        grouped[threshold]["labels"].append(label)
    levels = tuple(
        ThresholdLevel(
            threshold_id=str(grouped[value]["labels"][0]),
            threshold=value,
            weight=float(grouped[value]["weight"]),
            quantiles=tuple(grouped[value]["quantiles"]),
            quantile_labels=tuple(grouped[value]["labels"]),
        )
        for value in order
    )
    return ThresholdBundle(
        finite_distance_count=int(finite_distance_count),
        requested_quantiles=quantiles,
        requested_weights=parsed_weights,
        raw_thresholds=thresholds,
        quantile_labels=labels,
        levels=levels,
        theta_star_quantile=float(contract["theta_star_quantile"]),
        theta_star=float(contract["theta_star"]),
        cost_cap_quantile=float(contract["cost_cap_quantile"]),
        cost_cap=float(contract["cost_cap"]),
    )


def _pairwise_mean(matrix: np.ndarray, sequence: Sequence[int]) -> float:
    if len(sequence) < 2:
        return 0.0
    values = matrix[np.ix_(sequence, sequence)]
    upper = values[np.triu_indices(len(sequence), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def prefix_objective(
    sequence: Sequence[int],
    *,
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    prefix_weights: Sequence[float],
    variant: PrefixVariant,
    coverage_redundancy: np.ndarray,
    chemistry: ChemistryData,
) -> float:
    if not sequence:
        return 0.0
    best = np.full(len(matrix.parent_ids), np.inf, dtype=np.float64)
    utilities: list[float] = []
    costs: list[float] = []
    coverage_redundancies: list[float] = []
    structural_redundancies: list[float] = []
    size_penalties: list[float] = []
    rhos: list[float] = []
    prefix: list[int] = []
    table_coverage = 0.0
    for position, candidate_index in enumerate(sequence):
        prefix.append(int(candidate_index))
        best = np.minimum(best, matrix.distances[:, int(candidate_index)])
        if variant.mode == "single":
            utility = single_threshold_coverage(best, thresholds.theta_star)
        else:
            utility = weighted_multi_threshold_utility(best, thresholds.levels)
        capped_mean, _, _ = fixed_denominator_capped_cost(best, thresholds.cost_cap)
        utilities.append(float(utility))
        costs.append(float(capped_mean))
        coverage_redundancies.append(_pairwise_mean(coverage_redundancy, prefix))
        structural_redundancies.append(
            _pairwise_mean(chemistry.structural_similarity, prefix)
        )
        size_penalties.append(
            float(
                np.mean(
                    chemistry.normalized_sizes[
                        np.asarray(prefix, dtype=np.int64)
                    ]
                )
            )
        )
        rhos.append(float(prefix_weights[position]))
        if position + 1 == min(TABLE_K, len(sequence)):
            table_coverage = single_threshold_coverage(best, thresholds.theta_star)
    if variant.mode == "prefix":
        base = float(np.average(utilities, weights=rhos))
        capped_cost = float(np.average(costs, weights=rhos))
        coverage_penalty = float(np.average(coverage_redundancies, weights=rhos))
        structural_penalty = float(np.average(structural_redundancies, weights=rhos))
        size_penalty = float(np.average(size_penalties, weights=rhos))
    else:
        base = utilities[-1]
        capped_cost = costs[-1]
        coverage_penalty = coverage_redundancies[-1]
        structural_penalty = structural_redundancies[-1]
        size_penalty = size_penalties[-1]
    cost_utility = (
        1.0 - capped_cost / thresholds.cost_cap
        if thresholds.cost_cap > 0.0
        else float(capped_cost <= 0.0)
    )
    return float(
        base
        + variant.lambda_table2 * table_coverage
        + variant.lambda_cost * cost_utility
        - variant.lambda_covred * coverage_penalty
        - variant.lambda_struct * structural_penalty
        - variant.lambda_size * size_penalty
    )


def _objective(
    *,
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    prefix_weights: Sequence[float],
    variant: PrefixVariant,
    coverage_redundancy: np.ndarray,
    chemistry: ChemistryData,
) -> Callable[[Sequence[int]], float]:
    return lambda sequence: prefix_objective(
        sequence,
        matrix=matrix,
        thresholds=thresholds,
        prefix_weights=prefix_weights,
        variant=variant,
        coverage_redundancy=coverage_redundancy,
        chemistry=chemistry,
    )


def select_sequence(
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    chemistry: ChemistryData,
    variant: PrefixVariant,
    *,
    top_k: int = TOP_K,
    prefix_weights: Sequence[float] = DEFAULT_PREFIX_WEIGHTS,
    local_swap_passes: int = 2,
    eligible_candidate_indices: Sequence[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    covred = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
    objective = _objective(
        matrix=matrix,
        thresholds=thresholds,
        prefix_weights=prefix_weights,
        variant=variant,
        coverage_redundancy=covred,
        chemistry=chemistry,
    )
    universe = (
        [int(value) for value in eligible_candidate_indices]
        if eligible_candidate_indices is not None
        else list(range(len(matrix.candidate_rows)))
    )
    if len(universe) < int(top_k):
        raise ValueError(
            f"Connected selector needs {top_k} eligible actions, found {len(universe)}."
        )
    sequence, greedy_trace = greedy_select(
        universe,
        top_k=int(top_k),
        objective_fn=objective,
        candidate_ids=matrix.candidate_ids,
    )
    insertion_trace: list[dict[str, Any]] = []
    if variant.insertion_reorder:
        sequence, insertion_trace = optimize_insertion_order(
            sequence,
            objective_fn=objective,
            candidate_ids=matrix.candidate_ids,
        )
    before_swap = float(objective(sequence))
    swap_trace: list[dict[str, Any]] = []
    if variant.local_swap:
        sequence, swap_trace = local_swap_search(
            sequence,
            all_candidate_indices=universe,
            objective_fn=objective,
            candidate_ids=matrix.candidate_ids,
            max_passes=min(5, max(0, int(local_swap_passes))),
        )
    after_swap = float(objective(sequence))
    if after_swap + FLOAT_TOLERANCE < before_swap:
        raise AssertionError("Local swap lowered the registered prefix objective.")
    return sequence, {
        "greedy_trace": greedy_trace,
        "insertion_trace": insertion_trace,
        "swap_trace": swap_trace,
        "objective_before_swap": before_swap,
        "objective_after_swap": after_swap,
    }


def _subset_matrix(matrix: MatrixData, positions: Sequence[int]) -> MatrixData:
    indices = np.asarray(positions, dtype=np.int64)
    return MatrixData(
        matrix_run_dir=matrix.matrix_run_dir,
        parent_ids=tuple(matrix.parent_ids[index] for index in indices),
        candidate_rows=matrix.candidate_rows,
        distances=matrix.distances[indices, :],
        cf_drops=matrix.cf_drops[indices, :],
        applicable=matrix.applicable[indices, :],
        full_finite_distances=matrix.full_finite_distances,
        full_parent_count=matrix.full_parent_count,
        full_candidate_count=matrix.full_candidate_count,
        full_pair_count=matrix.full_pair_count,
        full_strict_flip_pair_count=matrix.full_strict_flip_pair_count,
        summary=matrix.summary,
        manifest=matrix.manifest,
        full_candidate_rows=matrix.full_candidate_rows,
    )


def _parent_smiles(matrix: MatrixData) -> dict[str, str]:
    result: dict[str, str] = {}
    with (matrix.matrix_run_dir / "pair_matrix.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result.setdefault(str(row["parent_id"]), str(row.get("parent_smiles") or ""))
    return result


def build_calibration_folds(matrix: MatrixData, *, fold_count: int = 5) -> list[list[int]]:
    if fold_count < 2 or len(matrix.parent_ids) < fold_count:
        raise ValueError("Calibration CV requires at least one parent per fold.")
    smiles = _parent_smiles(matrix)
    groups: dict[str, list[int]] = {}
    for index, parent_id in enumerate(matrix.parent_ids):
        scaffold = ""
        if Chem is not None and MurckoScaffold is not None:
            molecule = Chem.MolFromSmiles(smiles.get(parent_id, ""))
            if molecule is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=molecule, includeChirality=True
                )
        key = scaffold or f"fallback:{_stable_sha256(parent_id)[:12]}"
        groups.setdefault(key, []).append(index)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for _group, positions in ordered:
        target = min(range(fold_count), key=lambda value: (len(folds[value]), value))
        folds[target].extend(positions)
    if any(not fold for fold in folds):
        ordered_indices = sorted(range(len(matrix.parent_ids)), key=lambda i: matrix.parent_ids[i])
        folds = [ordered_indices[index::fold_count] for index in range(fold_count)]
    return [sorted(fold) for fold in folds]


def _sequence_metrics(
    sequence: Sequence[int],
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    chemistry: ChemistryData,
) -> dict[str, float]:
    covred = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
    rows, _ = compute_prefix_metrics(
        sequence,
        matrix=matrix,
        thresholds=thresholds,
        coverage_redundancy_matrix=covred,
        structural_similarity_matrix=chemistry.structural_similarity,
    )
    first_ten = rows[: min(TABLE_K, len(rows))]
    prefix = rows[: min(TOP_K, len(rows))]
    return {
        "prefix_auc_theta_k1_10": float(
            np.mean([row["ccrcov_theta_star"] for row in first_ten])
        ),
        "multi_threshold_prefix_auc": float(
            np.average(
                [row["weighted_multi_threshold_utility"] for row in prefix],
                weights=DEFAULT_PREFIX_WEIGHTS[: len(prefix)],
            )
        ),
        "k10_ccrcov_theta_star": float(first_ten[-1]["ccrcov_theta_star"]),
        "k20_ccrcov_theta_star": float(prefix[-1]["ccrcov_theta_star"]),
        "k10_fixed_capped_mean_cost": float(first_ten[-1]["fixed_capped_mean_cost"]),
        "coverage_redundancy": float(prefix[-1]["coverage_redundancy"]),
        "structural_redundancy": float(prefix[-1]["structural_redundancy"]),
        "size_penalty": float(
            np.mean(chemistry.normalized_sizes[np.asarray(sequence, dtype=np.int64)])
        ),
    }


def _read_a0_sequence(path: str | Path, matrix: MatrixData) -> list[int]:
    source = Path(path).expanduser().resolve()
    assert_calibration_selector_inputs(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if [int(row["rank"]) for row in rows] != list(range(1, TOP_K + 1)):
        raise ValueError("A0 selected sequence must contain exact ranks 1..20.")
    by_fragment = {
        str(row["canonical_fragment"]): index
        for index, row in enumerate(matrix.candidate_rows)
    }
    sequence: list[int] = []
    for row in rows:
        fragment = canonicalize_smiles(str(row.get("fragment") or ""))
        if fragment not in by_fragment:
            raise ValueError(f"A0 fragment is absent from the full matrix: {fragment}")
        sequence.append(by_fragment[fragment])
    if len(set(sequence)) != TOP_K:
        raise ValueError("A0 maps to duplicate canonical matrix candidates.")
    return sequence


def _base_variants() -> tuple[PrefixVariant, ...]:
    return (
        PrefixVariant("A1_coverage_theta", "single"),
        PrefixVariant("A2_multi_threshold", "multi"),
        PrefixVariant(
            "A3_multi_threshold_prefix",
            "prefix",
            lambda_table2=1.0,
        ),
    )


def a4_grid() -> tuple[PrefixVariant, ...]:
    return tuple(
        PrefixVariant(
            "A4_prefix_covred_swap",
            "prefix",
            lambda_table2=1.0,
            lambda_covred=covred,
            lambda_struct=structural,
            lambda_size=size,
            lambda_cost=cost,
            local_swap=True,
        )
        for covred in (0.1, 0.3, 0.5)
        for structural in (0.05, 0.1, 0.2)
        for size in (0.05, 0.1)
        for cost in (0.0, 0.1)
    )


def _cv_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_prefix_auc_theta_k1_10"]),
        -float(row["mean_multi_threshold_prefix_auc"]),
        float(row["mean_k10_fixed_capped_mean_cost"]),
        float(row["mean_coverage_redundancy"]),
        float(row["mean_structural_redundancy"]),
        float(row["mean_size_penalty"]),
        str(row["variant"]),
        str(row["config_sha256"]),
    )


def _cross_validate(
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    chemistry: ChemistryData,
    folds: Sequence[Sequence[int]],
    variant: PrefixVariant,
    *,
    a0_sequence: Sequence[int] | None = None,
    local_swap_passes: int = 2,
    eligible_candidate_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_positions = set(range(len(matrix.parent_ids)))
    for fold_index, validation_positions in enumerate(folds):
        train_positions = sorted(all_positions - set(validation_positions))
        train = _subset_matrix(matrix, train_positions)
        validation = _subset_matrix(matrix, validation_positions)
        if a0_sequence is None:
            sequence, _trace = select_sequence(
                train,
                thresholds,
                chemistry,
                variant,
                local_swap_passes=local_swap_passes,
                eligible_candidate_indices=eligible_candidate_indices,
            )
        else:
            sequence = list(a0_sequence)
        metrics = _sequence_metrics(sequence, validation, thresholds, chemistry)
        rows.append(
            {
                "variant": variant.name,
                "config_sha256": _stable_sha256(asdict(variant)),
                "fold": fold_index,
                "train_parent_count": len(train_positions),
                "validation_parent_count": len(validation_positions),
                **asdict(variant),
                **metrics,
            }
        )
    summary = {
        "variant": variant.name,
        "config_sha256": _stable_sha256(asdict(variant)),
        **asdict(variant),
        **{
            f"mean_{field}": float(np.mean([row[field] for row in rows]))
            for field in (
                "prefix_auc_theta_k1_10",
                "multi_threshold_prefix_auc",
                "k10_ccrcov_theta_star",
                "k20_ccrcov_theta_star",
                "k10_fixed_capped_mean_cost",
                "coverage_redundancy",
                "structural_redundancy",
                "size_penalty",
            )
        },
    }
    return summary, rows


def _candidate_rows(
    sequence: Sequence[int],
    matrix: MatrixData,
    metrics: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_covered = 0
    for rank, candidate_index in enumerate(sequence, start=1):
        source = matrix.candidate_rows[int(candidate_index)]
        covered = int(metrics[rank - 1]["num_theta_star_covered"])
        rows.append(
            {
                "rank": rank,
                "candidate_id": str(source["candidate_id"]),
                "fragment": str(source["canonical_fragment"]),
                "canonical_fragment": str(source["canonical_fragment"]),
                "source_parent_id": (source.get("source_parent_ids") or [None])[0],
                "source_parent_count": int(source.get("source_parent_count") or 0),
                "source_cf_drop": source.get("source_cf_drop_mean"),
                "source_cf_flip": True,
                "atom_ratio": source.get("source_atom_ratio_mean"),
                "projection_used": bool(
                    float(source.get("source_projection_used_rate") or 0.0) > 0.0
                ),
                "direct_substructure": bool(
                    float(source.get("source_direct_substructure_rate") or 0.0) > 0.0
                ),
                "calibration_coverage": float(metrics[rank - 1]["ccrcov_theta_star"]),
                "marginal_close_strict_flip_parent_count": covered - previous_covered,
            }
        )
        previous_covered = covered
    return rows


def _candidate_limitation(
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    parent_smiles: dict[str, str],
    *,
    connected_valid_candidate_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    finite = np.isfinite(matrix.distances)
    close = matrix.distances <= thresholds.theta_star
    rows: list[dict[str, Any]] = []
    scaffold_groups: dict[str, list[int]] = {}
    for index, parent_id in enumerate(matrix.parent_ids):
        scaffold = ""
        if Chem is not None and MurckoScaffold is not None:
            molecule = Chem.MolFromSmiles(parent_smiles.get(parent_id, ""))
            if molecule is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        scaffold = scaffold or "NO_SCAFFOLD"
        scaffold_groups.setdefault(scaffold, []).append(index)
        if bool(np.any(close[index])):
            group = "A_low_threshold_covered"
        elif bool(np.any(finite[index])):
            group = "B_only_high_threshold"
        elif bool(np.any(matrix.applicable[index])):
            group = "C_applicable_not_flip"
        else:
            group = "D_no_applicable"
        rows.append(
            {
                "parent_id": parent_id,
                "scaffold": scaffold,
                "limitation_group": group,
                "num_applicable_candidates": int(np.count_nonzero(matrix.applicable[index])),
                "num_strict_flip_candidates": int(np.count_nonzero(finite[index])),
                "num_close_strict_flip_candidates": int(np.count_nonzero(close[index])),
            }
        )
    scaffold_rows: list[dict[str, Any]] = []
    undercovered: set[str] = set()
    for scaffold, positions in sorted(scaffold_groups.items()):
        coverage = float(np.mean(np.any(close[np.asarray(positions), :], axis=1)))
        if coverage < 0.5:
            undercovered.add(scaffold)
        scaffold_rows.append(
            {
                "scaffold": scaffold,
                "parent_count": len(positions),
                "all_pool_close_strict_flip_coverage": coverage,
                "undercovered_scaffold": coverage < 0.5,
            }
        )
    for row in rows:
        row["undercovered_scaffold"] = row["scaffold"] in undercovered
        row["hard_parent_group"] = (
            "E_undercovered_scaffold"
            if row["undercovered_scaffold"]
            and row["limitation_group"] != "A_low_threshold_covered"
            else row["limitation_group"]
        )
    close_union = float(np.mean(np.any(close, axis=1)))
    strict_union = float(np.mean(np.any(finite, axis=1)))
    applicable_union = float(np.mean(np.any(matrix.applicable, axis=1)))
    no_effective = float(
        np.mean(
            [
                row["limitation_group"]
                in {"C_applicable_not_flip", "D_no_applicable"}
                for row in rows
            ]
        )
    )
    candidates_with_strict = int(np.count_nonzero(np.any(finite, axis=0)))
    candidates_with_close = int(np.count_nonzero(np.any(close, axis=0)))
    required = bool(
        connected_valid_candidate_count < TOP_K
        or candidates_with_strict < TOP_K
        or (close_union < 0.80 and no_effective >= 0.20)
    )
    audit = {
        "schema_version": "bace_candidate_pool_limitation_v1",
        "parent_count": len(matrix.parent_ids),
        "candidate_count": len(matrix.candidate_rows),
        "all_candidate_union_coverage": applicable_union,
        "all_candidate_strict_flip_union": strict_union,
        "all_candidate_close_strict_flip_union": close_union,
        "num_unique_candidates_with_any_connected_strict_flip": candidates_with_strict,
        "num_unique_candidates_with_any_close_connected_strict_flip": candidates_with_close,
        "num_unique_candidates_with_any_connected_valid_action": connected_valid_candidate_count,
        "candidate_expansion_rule": (
            "connected_valid_candidates < 20 or connected_strict_candidates < 20 "
            "or (close_union < 0.80 and no_effective >= 0.20)"
        ),
        "candidate_expansion_required": required,
        "test_used": False,
    }
    return audit, rows, scaffold_rows


def _candidate_connectivity_stats(
    matrix: MatrixData,
    thresholds: ThresholdBundle,
) -> list[dict[str, Any]]:
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in matrix.candidate_ids
    }
    with (matrix.matrix_run_dir / "pair_matrix.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_candidate[str(row["candidate_id"])].append(row)
    output: list[dict[str, Any]] = []
    for source in matrix.candidate_rows:
        candidate_id = str(source["candidate_id"])
        rows = rows_by_candidate[candidate_id]
        connected_valid = sum(
            int(row.get("num_connected_valid_matches") or 0) > 0 for row in rows
        )
        strict = sum(bool(row.get("pair_strict_flip")) for row in rows)
        close = sum(
            bool(row.get("pair_strict_flip"))
            and row.get("wnode_distance") is not None
            and float(row["wnode_distance"]) <= thresholds.theta_star
            for row in rows
        )
        output.append(
            {
                "candidate_id": candidate_id,
                "canonical_fragment": source.get("canonical_fragment"),
                "calibration_parent_count": len(rows),
                "connected_valid_parent_count": connected_valid,
                "connected_strict_flip_parent_count": strict,
                "close_connected_strict_flip_parent_count": close,
                "disconnected_match_count": sum(
                    int(row.get("num_disconnected_matches") or 0) for row in rows
                ),
                "has_connected_valid_calibration_action": connected_valid > 0,
                "has_connected_strict_flip_calibration_action": strict > 0,
            }
        )
    return output


def _connected_valid_candidate_indices(
    matrix: MatrixData,
    connectivity_rows: Sequence[dict[str, Any]],
) -> list[int]:
    by_id = {str(row["candidate_id"]): row for row in connectivity_rows}
    return [
        index
        for index, candidate_id in enumerate(matrix.candidate_ids)
        if bool(by_id[candidate_id]["has_connected_valid_calibration_action"])
    ]


def run_bace_wnode_prefix_selector(
    *,
    matrix_run_dir: str | Path,
    thresholds_json: str | Path,
    current_selected_csv: str | Path,
    output_dir: str | Path,
    local_swap_passes: int = 2,
    fold_count: int = 5,
    require_connected: bool = False,
) -> dict[str, Any]:
    assert_calibration_selector_inputs(
        matrix_run_dir, thresholds_json, current_selected_csv, output_dir
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Selector output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    matrix = load_calibration_matrix(matrix_run_dir, forbid_test=True)
    if require_connected:
        if matrix.manifest.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
            raise ValueError("BACE connected selector rejects a legacy action matrix.")
        if matrix.manifest.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("BACE connected selector rejects a mismatched match policy.")
    if len(matrix.parent_ids) != 60:
        raise ValueError(f"Expected 60 BACE calibration parents, found {len(matrix.parent_ids)}")
    if len(matrix.candidate_rows) < TOP_K:
        raise ValueError("BACE candidate universe has fewer than 20 unique actions.")
    thresholds = threshold_bundle_from_manifest(
        thresholds_json,
        finite_distance_count=len(matrix.full_finite_distances),
    )
    threshold_contract = load_bace_thresholds(thresholds_json)
    if require_connected:
        if threshold_contract.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
            raise ValueError("BACE connected selector rejects legacy thresholds.")
        if threshold_contract.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise ValueError("BACE connected selector threshold policy mismatch.")
    chemistry = build_candidate_chemistry(
        matrix.candidate_rows,
        size_normalization_rows=matrix.full_candidate_rows,
    )
    if require_connected:
        connectivity_rows = _candidate_connectivity_stats(matrix, thresholds)
        eligible_candidate_indices = _connected_valid_candidate_indices(
            matrix, connectivity_rows
        )
    else:
        connectivity_rows = [
            {
                "candidate_id": candidate_id,
                "canonical_fragment": matrix.candidate_rows[index].get(
                    "canonical_fragment"
                ),
                "has_connected_valid_calibration_action": True,
            }
            for index, candidate_id in enumerate(matrix.candidate_ids)
        ]
        eligible_candidate_indices = list(range(len(matrix.candidate_ids)))
    if require_connected and len(eligible_candidate_indices) < TOP_K:
        raise ValueError(
            "Connected candidate universe has fewer than 20 actions with a "
            "connected-valid calibration deletion."
        )
    folds = build_calibration_folds(matrix, fold_count=int(fold_count))
    a0 = _read_a0_sequence(current_selected_csv, matrix)

    cv_summaries: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    a0_variant = PrefixVariant("A0_current", "prefix")
    a0_connected_valid = set(a0).issubset(set(eligible_candidate_indices))
    summary, rows = _cross_validate(
        matrix, thresholds, chemistry, folds, a0_variant, a0_sequence=a0
    )
    summary["eligible_under_connected_policy"] = a0_connected_valid
    cv_summaries.append(summary)
    cv_rows.extend(rows)
    for variant in _base_variants():
        summary, rows = _cross_validate(
            matrix,
            thresholds,
            chemistry,
            folds,
            variant,
            local_swap_passes=0,
            eligible_candidate_indices=eligible_candidate_indices,
        )
        cv_summaries.append(summary)
        cv_rows.extend(rows)
    a4_summaries: list[dict[str, Any]] = []
    for variant in a4_grid():
        summary, rows = _cross_validate(
            matrix,
            thresholds,
            chemistry,
            folds,
            variant,
            local_swap_passes=0,
            eligible_candidate_indices=eligible_candidate_indices,
        )
        a4_summaries.append(summary)
        cv_rows.extend(rows)
    selected_a4_summary = min(a4_summaries, key=_cv_sort_key)
    selected_a4 = next(
        variant
        for variant in a4_grid()
        if _stable_sha256(asdict(variant)) == selected_a4_summary["config_sha256"]
    )
    cv_summaries.append(selected_a4_summary)
    chosen_cv = min(
        [
            row
            for row in cv_summaries
            if row["variant"] != "A0_current" or a0_connected_valid
        ],
        key=_cv_sort_key,
    )

    variants = [a0_variant, *_base_variants(), selected_a4]
    sequences: dict[str, list[int]] = {}
    calibration_rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    covred = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
    for variant in variants:
        if variant.name == "A0_current":
            sequence = list(a0)
            trace = {"adoption_mode": "frozen_current_selector"}
        else:
            sequence, trace = select_sequence(
                matrix,
                thresholds,
                chemistry,
                variant,
                local_swap_passes=local_swap_passes,
                eligible_candidate_indices=eligible_candidate_indices,
            )
        sequences[variant.name] = sequence
        traces[variant.name] = trace
        metrics, parents = compute_prefix_metrics(
            sequence,
            matrix=matrix,
            thresholds=thresholds,
            coverage_redundancy_matrix=covred,
            structural_similarity_matrix=chemistry.structural_similarity,
        )
        variant_dir = destination / "variants" / variant.name
        variant_dir.mkdir(parents=True, exist_ok=False)
        _write_jsonl(
            variant_dir / "selected_sequence.jsonl",
            _candidate_rows(sequence, matrix, metrics),
        )
        _write_csv(variant_dir / "prefix_metrics.csv", metrics)
        _write_csv(variant_dir / "parent_best_distances.csv", parents)
        _write_json(variant_dir / "trace.json", trace)
        summary_row = {
            "variant": variant.name,
            **asdict(variant),
            **_sequence_metrics(sequence, matrix, thresholds, chemistry),
            "selected_candidate_ids": [matrix.candidate_ids[index] for index in sequence],
        }
        calibration_rows.append(summary_row)

    selected_variant = str(chosen_cv["variant"])
    selected_sequence = sequences[selected_variant]
    selected_metrics, selected_parent_rows = compute_prefix_metrics(
        selected_sequence,
        matrix=matrix,
        thresholds=thresholds,
        coverage_redundancy_matrix=covred,
        structural_similarity_matrix=chemistry.structural_similarity,
    )
    selected_rows = _candidate_rows(selected_sequence, matrix, selected_metrics)
    if [row["rank"] for row in selected_rows] != list(range(1, TOP_K + 1)):
        raise AssertionError("Final selected rank is not exactly 1..20.")
    if len({row["candidate_id"] for row in selected_rows}) != TOP_K:
        raise AssertionError("Final selection has duplicate candidate IDs.")
    if len({row["fragment"] for row in selected_rows}) != TOP_K:
        raise AssertionError("Final selection has duplicate canonical fragments.")
    connectivity_by_id = {
        str(row["candidate_id"]): row for row in connectivity_rows
    }
    selected_connectivity = [
        connectivity_by_id[str(row["candidate_id"])] for row in selected_rows
    ]
    all_selected_connected_valid = all(
        bool(row["has_connected_valid_calibration_action"])
        for row in selected_connectivity
    )
    if require_connected and not all_selected_connected_valid:
        raise ValueError(
            "Connected selector chose a candidate without a connected-valid calibration action."
        )

    selected_csv = destination / "selected_top20.csv"
    _write_csv(selected_csv, selected_rows)
    selected_sha = _sha256_file(selected_csv)
    _atomic_text(destination / "selected_top20.sha256", f"{selected_sha}  selected_top20.csv\n")
    _write_json(destination / "selected_top20.json", {"candidates": selected_rows})
    _write_csv(destination / "selected_subgraphs.csv", selected_rows)
    _write_json(destination / "selected_subgraphs.json", selected_rows)
    _write_csv(destination / "prefix_marginal_gain.csv", selected_rows)
    _write_csv(destination / "selector_variant_calibration.csv", calibration_rows)
    _write_json(
        destination / "selector_variant_calibration.json",
        {"variants": calibration_rows},
    )
    _write_csv(destination / "selector_cv_results.csv", cv_rows)
    _write_json(destination / "objective_traces.json", traces)
    _write_csv(destination / "parent_best_distances.csv", selected_parent_rows)
    _write_csv(destination / "candidate_connectivity_stats.csv", connectivity_rows)

    parent_smiles = _parent_smiles(matrix)
    limitation, hard_parents, scaffold_rows = _candidate_limitation(
        matrix,
        thresholds,
        parent_smiles,
        connected_valid_candidate_count=len(eligible_candidate_indices),
    )
    _write_json(destination / "candidate_pool_limitation_audit.json", limitation)
    _write_csv(destination / "hard_parent_groups.csv", hard_parents)
    _write_csv(destination / "scaffold_coverage.csv", scaffold_rows)
    decision = {
        "schema_version": "bace_wnode_prefix_selection_v2",
        "selected_variant": selected_variant,
        "selected_cv_metrics": chosen_cv,
        "selected_a4_hyperparameters": asdict(selected_a4),
        "selection_rule": [
            "max mean calibration CV prefix AUC at theta_star for K=1..10",
            "max mean calibration CV multi-threshold prefix AUC",
            "min capped cost",
            "min coverage redundancy",
            "min structural redundancy",
            "min deletion-size penalty",
            "stable config SHA256",
        ],
        "fold_count": int(fold_count),
        "cv_search": "deterministic_greedy_without_local_swap",
        "final_selected_a4_local_swap_passes": int(local_swap_passes),
        "folds": [[matrix.parent_ids[index] for index in fold] for fold in folds],
        "test_used": False,
        "gcf_result_used": False,
        "action_semantics_version": matrix.manifest.get(
            "action_semantics_version", "hard_delete_all_matches_v1"
        ),
        "match_selection_policy": matrix.manifest.get(
            "match_selection_policy",
            "min_wnode_then_cfdrop_then_match_index_v1",
        ),
    }
    _write_json(destination / "selected_variant_manifest.json", decision)

    matrix_root = Path(matrix_run_dir).expanduser().resolve()
    threshold_path = Path(thresholds_json).expanduser().resolve()
    matrix_manifest = matrix.manifest
    selection_record = {
        "schema_version": "bace_ours_wnode_frozen_selection_v2",
        "selection_frozen": not limitation["candidate_expansion_required"],
        "selection_split": "calibration",
        "test_used": False,
        "gcf_result_used": False,
        "selected_sequence_sha256": selected_sha,
        "selected_candidate_ids": [row["candidate_id"] for row in selected_rows],
        "ranks": list(range(1, TOP_K + 1)),
        "candidate_pool_identity": matrix_manifest.get("inputs", {}).get("candidate_pool"),
        "calibration_matrix_sha256": _sha256_file(matrix_root / "pair_matrix.jsonl"),
        "teacher_identity": matrix_manifest.get("inputs", {}).get("teacher_path"),
        "molclr_identity": matrix_manifest.get("inputs", {}).get("molclr_checkpoint"),
        "threshold_manifest": str(threshold_path),
        "threshold_manifest_sha256": _sha256_file(threshold_path),
        "theta_star": thresholds.theta_star,
        "selector_config": decision,
        "created_at": _utc_now(),
        "action_semantics_version": decision["action_semantics_version"],
        "match_selection_policy": decision["match_selection_policy"],
        "all_selected_have_connected_valid_calibration_action": all_selected_connected_valid,
    }
    if limitation["candidate_expansion_required"]:
        selection_record["freeze_block_reason"] = (
            "candidate_pool_limitation_audit requires preregistered expansion"
        )
        _write_json(destination / "provisional_selection.json", selection_record)
    else:
        _write_json(destination / "frozen_selection.json", selection_record)
    rank_audit = {
        "rank_preservation_pass": True,
        "ranks_exact_1_to_20": True,
        "candidate_ids_unique": True,
        "fragments_unique": True,
        "selected_sequence_sha256": selected_sha,
        "selection_performed_in_eval": False,
        "test_used": False,
        "action_semantics_version": decision["action_semantics_version"],
        "match_selection_policy": decision["match_selection_policy"],
    }
    _write_json(destination / "rank_preservation_audit.json", rank_audit)
    summary_payload = {
        "schema_version": "bace_wnode_prefix_selector_summary_v2",
        "parent_count": len(matrix.parent_ids),
        "candidate_count": len(matrix.candidate_rows),
        "selected_variant": selected_variant,
        "theta_star": thresholds.theta_star,
        "cost_cap": thresholds.cost_cap,
        "candidate_expansion_required": limitation["candidate_expansion_required"],
        "test_loaded": False,
        "test_used": False,
        "gcf_result_used": False,
        "selection_frozen": not limitation["candidate_expansion_required"],
        "action_semantics_version": decision["action_semantics_version"],
        "match_selection_policy": decision["match_selection_policy"],
        "all_selected_have_connected_valid_calibration_action": all_selected_connected_valid,
        "disconnected_residual_used_count": 0 if require_connected else None,
        "run_complete": True,
    }
    _write_json(destination / "summary.json", summary_payload)
    _write_json(
        destination / "run_manifest.json",
        {
            "created_at": _utc_now(),
            "matrix_run_dir": str(matrix_root),
            "thresholds_json": str(threshold_path),
            "current_selected_csv": str(Path(current_selected_csv).expanduser().resolve()),
            "selection_split": "calibration",
            "selection_performed_in_eval": False,
            "threshold_fitted_on_test": False,
            "test_loaded": False,
            "gcf_result_used": False,
            "action_semantics_version": decision["action_semantics_version"],
            "match_selection_policy": decision["match_selection_policy"],
            "run_complete": True,
        },
    )
    selector_audit = {
        "schema_version": "bace_connected_selector_audit_v3",
        "passed": True,
        "selected_count": len(selected_rows),
        "ranks_exact_1_to_20": True,
        "unique_candidates": True,
        "test_loaded": False,
        "test_used": False,
        "gcf_result_used": False,
        "all_selected_have_connected_valid_calibration_action": all_selected_connected_valid,
        "all_calibration_winners_connected": True if require_connected else None,
        "disconnected_residual_used_count": 0 if require_connected else None,
        "action_semantics_version": decision["action_semantics_version"],
        "match_selection_policy": decision["match_selection_policy"],
        "threshold_fitted_on_test": False,
        "selection_performed_in_eval": False,
        "selection_frozen": not limitation["candidate_expansion_required"],
    }
    _write_json(destination / "selector_manifest.json", selection_record)
    _write_json(destination / "selector_audit.json", selector_audit)
    _write_json(destination / "_RUN_COMPLETE.json", summary_payload)
    return summary_payload


__all__ = [
    "PrefixVariant",
    "a4_grid",
    "assert_calibration_selector_inputs",
    "build_calibration_folds",
    "prefix_objective",
    "run_bace_wnode_prefix_selector",
    "select_sequence",
    "threshold_bundle_from_manifest",
]
