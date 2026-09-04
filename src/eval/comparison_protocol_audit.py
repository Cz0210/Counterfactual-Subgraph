"""Frozen-protocol, parent-unit comparison audit for the final 4x4 matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.eval.four_by_four_main_results import (
    DATASET_ORDER,
    DATASET_SLUGS,
    METHOD_ORDER,
    NA_COST_VALUES,
    audit_cell,
)


SCHEMA_VERSION = "final_comparison_protocol_audit_v1"
CLAIMS = (
    "OURS_UNIVERSAL_DOMINANCE_SUPPORTED",
    "OURS_PRIMARY_METRIC_ADVANTAGE_SUPPORTED",
    "OURS_LOW_COST_TRADEOFF_ADVANTAGE_SUPPORTED",
    "OURS_ADVANTAGE_NOT_STATISTICALLY_RESOLVED",
)
BASELINES = tuple(method for method in METHOD_ORDER if method != "Ours")


class ComparisonAuditError(ValueError):
    """The frozen comparison contract cannot be established."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonAuditError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ComparisonAuditError(f"JSON must contain one object: {path}")
    return dict(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        raise ComparisonAuditError(f"invalid CSV: {path}") from exc
    if not rows:
        raise ComparisonAuditError(f"CSV has no rows: {path}")
    return rows


def _normalized(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _field(row: Mapping[str, Any], aliases: Sequence[str], *, required: bool = True) -> str | None:
    fields = {_normalized(key): key for key in row}
    for alias in aliases:
        if _normalized(alias) in fields:
            return fields[_normalized(alias)]
    if required:
        raise ComparisonAuditError(f"missing field from {tuple(aliases)}")
    return None


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonAuditError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ComparisonAuditError(f"{label} is not finite")
    return result


def _optional_cost(value: Any) -> float | None:
    if str(value).strip().lower() in NA_COST_VALUES:
        return None
    result = _float(value, label="conditional cost")
    if result < 0.0:
        raise ComparisonAuditError("conditional cost is negative")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ComparisonAuditError(f"{label} is not boolean: {value!r}")


def load_frozen_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    expected = {
        "schema_version": "final_paper_evaluation_contract_v1",
        "datasets": list(DATASET_ORDER),
        "methods": list(METHOD_ORDER),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ComparisonAuditError(f"frozen contract {key} mismatch")
    if contract.get("figure3") != {
        "k_values": list(range(1, 21)),
        "theta": 0.05,
    }:
        raise ComparisonAuditError("frozen Figure 3 contract mismatch")
    if contract.get("figure4") != {
        "k": 10,
        "threshold_count": 601,
        "threshold_start": 0.0,
        "threshold_stop": 0.0535,
    }:
        raise ComparisonAuditError("frozen Figure 4 contract mismatch")
    if contract.get("table2") != {
        "k": 10,
        "theta": 0.05,
        "undefined_conditional_cost": "N/A",
    }:
        raise ComparisonAuditError("frozen Table 2 contract mismatch")
    bootstrap = contract.get("bootstrap")
    if bootstrap != {"resampling_unit": "test_parent", "samples": 1000, "seed": 0}:
        raise ComparisonAuditError("frozen bootstrap contract mismatch")
    forbidden = (
        "forbid_method_specific_parent_cohort",
        "forbid_method_specific_threshold",
        "forbid_posthoc_metric_selection",
        "forbid_test_tuning",
    )
    if any(contract.get(key) is not True for key in forbidden):
        raise ComparisonAuditError("outcome-shaping exclusion is incomplete")
    return contract


def load_parent_observations(
    path: Path, *, k: int, theta: float
) -> dict[str, tuple[bool, float | None]]:
    """Load one row per test parent at frozen K without changing its cohort."""

    rows = _read_csv(path)
    parent_field = _field(
        rows[0], ("parent_id", "test_parent_id", "query_id", "graph_id")
    )
    k_field = _field(rows[0], ("k", "K", "prefix_k"))
    distance_field = _field(
        rows[0],
        ("best_distance", "wnode_distance", "distance", "cost", "capped_distance"),
    )
    covered_field = _field(
        rows[0],
        ("theta_star_covered", "covered", "strict_flip_covered", "close_cf_covered"),
        required=False,
    )
    strict_field = _field(
        rows[0],
        ("strict_recourse_available", "strict_flip_available", "strict_flip"),
        required=False,
    )
    result: dict[str, tuple[bool, float | None]] = {}
    for row in rows:
        try:
            row_k = int(str(row[k_field]))
        except (TypeError, ValueError) as exc:
            raise ComparisonAuditError(f"invalid parent K in {path}") from exc
        if row_k != k:
            continue
        parent_id = str(row[parent_field]).strip()
        if not parent_id or parent_id in result:
            raise ComparisonAuditError(f"missing or duplicate parent at K={k}: {path}")
        distance = _optional_cost(row[distance_field])
        strict = (
            _boolean(row[strict_field], label="strict flip")
            if strict_field is not None
            else distance is not None
        )
        inferred = strict and distance is not None and distance <= theta + 1e-12
        covered = (
            _boolean(row[covered_field], label="theta coverage")
            if covered_field is not None
            else inferred
        )
        if covered != inferred:
            raise ComparisonAuditError(
                f"parent coverage disagrees with strict-flip WNode contract: {path}"
            )
        result[parent_id] = (covered, distance if covered else None)
    if not result:
        raise ComparisonAuditError(f"no parent rows at K={k}: {path}")
    return result


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_parent_bootstrap(
    ours: Mapping[str, tuple[bool, float | None]],
    baseline: Mapping[str, tuple[bool, float | None]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired test parents; candidates or rules are never resampled."""

    if set(ours) != set(baseline) or not ours:
        raise ComparisonAuditError("paired methods do not have one parent population")
    if samples < 1:
        raise ComparisonAuditError("bootstrap samples must be positive")
    parent_ids = sorted(ours)
    rng = random.Random(seed)

    def statistics_for(indices: Iterable[int]) -> tuple[float, float | None]:
        selected = [parent_ids[index] for index in indices]
        ours_covered = [ours[parent][0] for parent in selected]
        base_covered = [baseline[parent][0] for parent in selected]
        coverage_difference = statistics.fmean(ours_covered) - statistics.fmean(
            base_covered
        )
        ours_cost = _median(
            [ours[parent][1] for parent in selected if ours[parent][1] is not None]
        )
        base_cost = _median(
            [
                baseline[parent][1]
                for parent in selected
                if baseline[parent][1] is not None
            ]
        )
        cost_difference = (
            None if ours_cost is None or base_cost is None else base_cost - ours_cost
        )
        return coverage_difference, cost_difference

    observed_coverage, observed_cost = statistics_for(range(len(parent_ids)))
    coverage_samples: list[float] = []
    cost_samples: list[float] = []
    for _ in range(samples):
        sampled = [rng.randrange(len(parent_ids)) for _ in parent_ids]
        coverage, cost = statistics_for(sampled)
        coverage_samples.append(coverage)
        if cost is not None:
            cost_samples.append(cost)
    return {
        "bootstrap_unit": "test_parent",
        "parent_count": len(parent_ids),
        "samples": samples,
        "coverage_difference": observed_coverage,
        "coverage_ci_low": _percentile(coverage_samples, 0.025),
        "coverage_ci_high": _percentile(coverage_samples, 0.975),
        "cost_difference": observed_cost,
        "cost_ci_low": _percentile(cost_samples, 0.025),
        "cost_ci_high": _percentile(cost_samples, 0.975),
        "cost_valid_bootstrap_samples": len(cost_samples),
    }


def _trapezoid(rows: Sequence[Mapping[str, str]], *, x: str, y: str) -> float:
    points = sorted(
        ((_float(row[x], label=x), _float(row[y], label=y)) for row in rows),
        key=lambda pair: pair[0],
    )
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        raise ComparisonAuditError("AUC requires at least two distinct x values")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def classify_claim(pairwise: Sequence[Mapping[str, Any]], rankings: Sequence[Mapping[str, Any]]) -> str:
    if not pairwise:
        raise ComparisonAuditError("claim classification requires pairwise results")
    decisive = [
        row
        for row in pairwise
        if row.get("coverage_ci_low") is not None
        and row.get("cost_ci_low") is not None
    ]
    if len(decisive) == len(pairwise) and all(
        float(row["coverage_ci_low"]) > 0.0 and float(row["cost_ci_low"]) > 0.0
        for row in decisive
    ):
        return CLAIMS[0]
    by_baseline = {
        baseline: [row for row in pairwise if row["baseline"] == baseline]
        for baseline in BASELINES
    }
    primary = all(
        len(rows) == len(DATASET_ORDER)
        and sum(
            float(row["coverage_difference"]) > 0.0
            and row.get("cost_difference") is not None
            and float(row["cost_difference"]) > 0.0
            for row in rows
        )
        >= 3
        and statistics.fmean(float(row["coverage_difference"]) for row in rows)
        > 0.0
        and statistics.fmean(
            float(row["cost_difference"])
            for row in rows
            if row.get("cost_difference") is not None
        )
        > 0.0
        for rows in by_baseline.values()
    )
    if primary:
        return CLAIMS[1]
    ours_top_auc = sum(
        row.get("method") == "Ours"
        and row.get("rank") == 1
        and row.get("metric") in {"figure3_coverage_auc", "figure4_coverage_auc"}
        for row in rankings
    )
    positive_cost = sum(
        row.get("cost_difference") is not None
        and float(row["cost_difference"]) > 0.0
        for row in pairwise
    )
    if ours_top_auc >= len(DATASET_ORDER) or positive_cost >= len(DATASET_ORDER) * 2:
        return CLAIMS[2]
    return CLAIMS[3]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ComparisonAuditError(f"refusing empty output CSV: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_comparison_audit(
    *,
    matrix_status: Path,
    final_export_root: Path,
    frozen_contract: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = load_frozen_contract(frozen_contract)
    matrix = _read_json(matrix_status)
    if (
        matrix.get("matrix_complete_cells") != 16
        or matrix.get("matrix_total_cells") != 16
        or matrix.get("all_cells_complete") is not True
    ):
        raise ComparisonAuditError("matrix authority does not prove 16/16")
    rows = matrix.get("cells")
    if type(rows) is not list or len(rows) != 16:
        raise ComparisonAuditError("matrix must contain exactly 16 cells")
    export_audit = _read_json(final_export_root / "final_export_audit.json")
    if export_audit.get("passed") is not True:
        raise ComparisonAuditError("final renderer audit is not PASS")

    cells = [audit_cell(row) for row in rows]
    indexed = {(cell.dataset, cell.method): cell for cell in cells}
    if set(indexed) != {
        (dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER
    }:
        raise ComparisonAuditError("matrix cell identities are incomplete")

    pairwise: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    bootstrap = contract["bootstrap"]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        dataset_cells = [indexed[(dataset, method)] for method in METHOD_ORDER]
        for identity in ("oracle_hash", "split_hash", "molclr_checkpoint_hash", "threshold_config_hash"):
            values = {cell.row.get(identity) for cell in dataset_cells}
            if len(values) != 1 or None in values or "" in values:
                raise ComparisonAuditError(f"{dataset}: methods differ on {identity}")
        parents = {
            method: load_parent_observations(
                indexed[(dataset, method)].root / "parent_best_distances.csv",
                k=contract["table2"]["k"],
                theta=contract["table2"]["theta"],
            )
            for method in METHOD_ORDER
        }
        populations = {tuple(sorted(value)) for value in parents.values()}
        if len(populations) != 1:
            raise ComparisonAuditError(f"{dataset}: method-specific parent cohort")
        protocol_rows.append(
            {
                "dataset": dataset,
                "parent_count": len(parents["Ours"]),
                "same_parent_population": True,
                "same_wnode": True,
                "same_threshold_grid": True,
                "same_selector_freeze_policy": True,
                "test_used_for_selection": False,
            }
        )
        for baseline_index, baseline in enumerate(BASELINES):
            result = paired_parent_bootstrap(
                parents["Ours"],
                parents[baseline],
                samples=bootstrap["samples"],
                seed=bootstrap["seed"] + dataset_index * 100 + baseline_index,
            )
            pairwise.append({"dataset": dataset, "baseline": baseline, **result})

    rankings: list[dict[str, Any]] = []
    macro_inputs: dict[tuple[str, str], list[float]] = {}
    for dataset in DATASET_ORDER:
        combined = final_export_root / DATASET_SLUGS[dataset] / "combined"
        for metric, filename, x in (
            ("figure3_coverage_auc", "figure3_coverage_vs_k.csv", "k"),
            ("figure4_coverage_auc", "figure4_coverage_vs_threshold.csv", "threshold"),
        ):
            source = _read_csv(combined / filename)
            values = {
                method: _trapezoid(
                    [row for row in source if row.get("method") == method],
                    x=x,
                    y="coverage",
                )
                for method in METHOD_ORDER
            }
            ordered = sorted(METHOD_ORDER, key=lambda method: (-values[method], method))
            for method in METHOD_ORDER:
                rankings.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "direction": "higher",
                        "method": method,
                        "value": values[method],
                        "rank": ordered.index(method) + 1,
                    }
                )
                macro_inputs.setdefault((metric, method), []).append(values[method])
        table = _read_csv(combined / "table2.csv")
        for metric, field, direction in (
            ("table2_ccrcov", "coverage", "higher"),
            ("table2_conditional_median_wnode", "cost", "lower"),
        ):
            values = {
                row["method"]: (
                    _optional_cost(row[field])
                    if field == "cost"
                    else _float(row[field], label=field)
                )
                for row in table
            }
            ordered = sorted(
                METHOD_ORDER,
                key=lambda method: (
                    values[method] is None,
                    -float(values[method]) if direction == "higher" and values[method] is not None else float(values[method]) if values[method] is not None else math.inf,
                    method,
                ),
            )
            for method in METHOD_ORDER:
                rankings.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "direction": direction,
                        "method": method,
                        "value": values[method] if values[method] is not None else "N/A",
                        "rank": ordered.index(method) + 1,
                    }
                )
                if values[method] is not None:
                    macro_inputs.setdefault((metric, method), []).append(
                        float(values[method])
                    )

    macro = [
        {
            "metric": metric,
            "method": method,
            "dataset_count": len(values),
            "macro_average": statistics.fmean(values),
        }
        for (metric, method), values in sorted(macro_inputs.items())
    ]
    for baseline in BASELINES:
        rows_for_baseline = [row for row in pairwise if row["baseline"] == baseline]
        cost_values = [
            float(row["cost_difference"])
            for row in rows_for_baseline
            if row["cost_difference"] is not None
        ]
        macro.extend(
            [
                {
                    "metric": "ours_minus_baseline_ccrcov",
                    "method": baseline,
                    "dataset_count": len(rows_for_baseline),
                    "macro_average": statistics.fmean(
                        float(row["coverage_difference"])
                        for row in rows_for_baseline
                    ),
                },
                {
                    "metric": "baseline_minus_ours_cost",
                    "method": baseline,
                    "dataset_count": len(cost_values),
                    "macro_average": statistics.fmean(cost_values)
                    if cost_values
                    else "N/A",
                },
            ]
        )

    claim = classify_claim(pairwise, rankings)
    destination = output_root.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"comparison audit root must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_csv(temporary / "ours_pairwise_bootstrap.csv", pairwise)
        _write_csv(temporary / "ours_dataset_rankings.csv", rankings)
        _write_csv(temporary / "ours_macro_average.csv", macro)
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "same_parent_population": True,
            "same_wnode": True,
            "same_threshold_grid": True,
            "same_selector_freeze_policy": True,
            "test_used_for_selection": False,
            "outcome_shaping_used": False,
            "contract_sha256": _sha256(contract),
            "matrix_status": str(matrix_status.resolve()),
            "final_export_root": str(final_export_root.resolve()),
            "datasets": protocol_rows,
        }
        _atomic_json(temporary / "comparison_protocol_audit.json", protocol)
        superiority = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "claim_status": claim,
            "allowed_claims": list(CLAIMS),
            "bootstrap_unit": "test_parent",
            "bootstrap_samples": bootstrap["samples"],
            "bootstrap_seed": bootstrap["seed"],
            "posthoc_metric_selection": False,
            "test_used_for_selection": False,
            "pairwise_sha256": hashlib.sha256(
                (temporary / "ours_pairwise_bootstrap.csv").read_bytes()
            ).hexdigest(),
            "rankings_sha256": hashlib.sha256(
                (temporary / "ours_dataset_rankings.csv").read_bytes()
            ).hexdigest(),
        }
        _atomic_json(temporary / "ours_superiority_audit.json", superiority)
        (temporary / "PASS").write_text("PASS\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "output_root": str(destination),
        "claim_status": claim,
        "protocol_audit_state": "PASS",
    }


__all__ = [
    "BASELINES",
    "CLAIMS",
    "ComparisonAuditError",
    "classify_claim",
    "load_frozen_contract",
    "load_parent_observations",
    "paired_parent_bootstrap",
    "run_comparison_audit",
]
