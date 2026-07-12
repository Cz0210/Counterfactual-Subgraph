#!/usr/bin/env python3
"""Audit MolCLR-Node-FGW threshold consistency across evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


INVENTORY_FIELDS = [
    "run_id",
    "run_dir",
    "method",
    "is_smoke_path",
    "is_full_path",
    "audit_valid",
    "distance_type",
    "distance_line",
    "fgw_lambda",
    "structure_mode",
    "feature_cost",
    "atom_penalty",
    "cf_mode",
    "dataset_csv",
    "teacher_path",
    "skip_redundancy",
    "num_parents",
    "num_candidates",
    "threshold_source",
    "quantiles_json",
    "thresholds_json",
    "candidate_set_preselected",
    "selection_performed_in_eval",
    "selection_method",
    "pair_distance_cache_hit_rate",
    "node_embedding_cache_hit_rate",
    "num_invalid_smiles",
    "num_nan_distances",
    "runtime_seconds",
    "warnings",
]

THRESHOLD_LONG_FIELDS = [
    "run_id",
    "run_dir",
    "method",
    "quantile",
    "threshold",
    "threshold_source",
    "num_parents",
    "num_candidates",
    "close_only_coverage",
    "close_cf_coverage",
]

PAIRWISE_FIELDS = [
    "run_a",
    "run_b",
    "method_a",
    "method_b",
    "same_distance_type",
    "same_distance_line",
    "same_fgw_lambda",
    "same_structure_mode",
    "same_feature_cost",
    "same_atom_penalty",
    "same_fgw_config",
    "same_cf_mode",
    "same_dataset_csv",
    "same_teacher_path",
    "same_skip_redundancy",
    "same_num_parents",
    "same_num_candidates",
    "same_candidate_set_preselected",
    "same_selection_performed_in_eval",
    "same_eval_protocol",
    "same_threshold_source",
    "same_quantile_grid",
    "same_absolute_thresholds",
    "num_common_quantiles",
    "max_abs_threshold_diff",
    "mean_abs_threshold_diff",
    "max_relative_threshold_diff",
    "mismatched_config_fields",
    "direct_threshold_comparison_ok",
]

AUTO_QUANTILE_WARNING = (
    "These runs use the same quantile labels but different absolute FGW radii; "
    "their coverage values should not be treated as measurements at the same threshold."
)


@dataclass
class RunRecord:
    run_id: str
    run_dir: str
    method: str
    is_smoke_path: bool
    is_full_path: bool
    distance_type: str | None = None
    distance_line: str | None = None
    fgw_lambda: float | None = None
    structure_mode: str | None = None
    feature_cost: str | None = None
    atom_penalty: float | None = None
    cf_mode: str | None = None
    dataset_csv: str | None = None
    teacher_path: str | None = None
    skip_redundancy: bool | None = None
    num_parents: int | None = None
    num_candidates: int | None = None
    threshold_source: str | None = None
    candidate_set_preselected: bool | None = None
    selection_performed_in_eval: bool | None = None
    selection_method: str | None = None
    pair_distance_cache_hit_rate: float | None = None
    node_embedding_cache_hit_rate: float | None = None
    num_invalid_smiles: int | None = None
    num_nan_distances: int | None = None
    runtime_seconds: float | None = None
    threshold_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_valid: bool = True

    @property
    def quantiles(self) -> list[float]:
        return sorted(
            {
                float(row["quantile"])
                for row in self.threshold_rows
                if _as_float(row.get("quantile")) is not None
            }
        )

    @property
    def thresholds(self) -> list[float]:
        rows_with_quantiles = [
            row for row in self.threshold_rows if _as_float(row.get("quantile")) is not None
        ]
        if rows_with_quantiles:
            ordered = sorted(rows_with_quantiles, key=lambda row: float(row["quantile"]))
            return [float(row["threshold"]) for row in ordered]
        return sorted(
            float(row["threshold"])
            for row in self.threshold_rows
            if _as_float(row.get("threshold")) is not None
        )

    def inventory_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "method": self.method,
            "is_smoke_path": self.is_smoke_path,
            "is_full_path": self.is_full_path,
            "audit_valid": self.audit_valid,
            "distance_type": self.distance_type,
            "distance_line": self.distance_line,
            "fgw_lambda": self.fgw_lambda,
            "structure_mode": self.structure_mode,
            "feature_cost": self.feature_cost,
            "atom_penalty": self.atom_penalty,
            "cf_mode": self.cf_mode,
            "dataset_csv": self.dataset_csv,
            "teacher_path": self.teacher_path,
            "skip_redundancy": self.skip_redundancy,
            "num_parents": self.num_parents,
            "num_candidates": self.num_candidates,
            "threshold_source": self.threshold_source,
            "quantiles_json": json.dumps(self.quantiles),
            "thresholds_json": json.dumps(self.thresholds),
            "candidate_set_preselected": self.candidate_set_preselected,
            "selection_performed_in_eval": self.selection_performed_in_eval,
            "selection_method": self.selection_method,
            "pair_distance_cache_hit_rate": self.pair_distance_cache_hit_rate,
            "node_embedding_cache_hit_rate": self.node_embedding_cache_hit_rate,
            "num_invalid_smiles": self.num_invalid_smiles,
            "num_nan_distances": self.num_nan_distances,
            "runtime_seconds": self.runtime_seconds,
            "warnings": "; ".join(self.warnings),
        }


def _as_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    result = _as_float(value)
    return int(result) if result is not None else None


def _as_bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _normalize_path(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))


def _first_value(rows: Sequence[dict[str, Any]], config: dict[str, Any], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return config.get(key)


def _constant_value(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    key: str,
    warnings: list[str],
) -> Any:
    values = [str(row[key]).strip() for row in rows if row.get(key) not in (None, "")]
    unique = list(dict.fromkeys(values))
    if len(unique) > 1:
        warnings.append(f"inconsistent_{key}:{unique}")
    return unique[0] if unique else config.get(key)


def _is_node_fgw(config: dict[str, Any], rows: Sequence[dict[str, Any]]) -> bool:
    distance_types = {_text(config.get("distance_type"))}
    distance_lines = {_text(config.get("distance_line"))}
    distance_types.update(_text(row.get("distance_type")) for row in rows)
    distance_lines.update(_text(row.get("distance_line")) for row in rows)
    return "node_fgw" in distance_types or "MolCLR-Node-FGW" in distance_lines


def _path_kind(path: Path) -> tuple[bool, bool]:
    normalized = path.as_posix().lower()
    smoke = re.search(r"(^|[/_.-])smoke([/_.-]|$)", normalized) is not None
    full = re.search(r"(^|[/_.-])full([/_.-]|$)", normalized) is not None
    return smoke, full


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _method_run_id(eval_root: Path, run_dir: Path, method: str) -> str:
    try:
        relative = run_dir.relative_to(eval_root).as_posix()
    except ValueError:
        relative = run_dir.as_posix()
    relative = relative or run_dir.name
    return f"{relative}::{method}"


def _build_record(
    *,
    eval_root: Path,
    run_dir: Path,
    method: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    cache: dict[str, Any],
) -> RunRecord:
    warnings: list[str] = []
    smoke, full = _path_kind(run_dir)
    valid_rows: list[dict[str, Any]] = []
    seen_quantiles: set[float] = set()
    for row_index, row in enumerate(rows):
        threshold = _as_float(row.get("threshold"))
        if threshold is None:
            warnings.append(f"invalid_threshold_row:{row_index}")
            continue
        quantile = _as_float(row.get("quantile"))
        if quantile is not None and quantile in seen_quantiles:
            warnings.append(f"duplicate_quantile:{quantile}")
        if quantile is not None:
            seen_quantiles.add(quantile)
        valid_rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "threshold_source": _text(row.get("threshold_source")),
                "num_parents": _as_int(row.get("num_parents")),
                "num_candidates": _as_int(row.get("num_candidates")),
                "close_only_coverage": _as_float(row.get("close_only_coverage")),
                "close_cf_coverage": _as_float(row.get("close_cf_coverage")),
            }
        )
    if not valid_rows:
        warnings.append("no_valid_threshold_rows")

    threshold_source = _text(_constant_value(rows, config, "threshold_source", warnings))
    record = RunRecord(
        run_id=_method_run_id(eval_root, run_dir, method),
        run_dir=str(run_dir),
        method=method,
        is_smoke_path=smoke,
        is_full_path=full,
        distance_type=_text(_constant_value(rows, config, "distance_type", warnings)),
        distance_line=_text(_constant_value(rows, config, "distance_line", warnings)),
        fgw_lambda=_as_float(_constant_value(rows, config, "fgw_lambda", warnings)),
        structure_mode=_text(_constant_value(rows, config, "structure_mode", warnings)),
        feature_cost=_text(_constant_value(rows, config, "feature_cost", warnings)),
        atom_penalty=_as_float(_constant_value(rows, config, "atom_penalty", warnings)),
        cf_mode=_text(_first_value(rows, config, "cf_mode")),
        dataset_csv=_normalize_path(_first_value(rows, config, "dataset_csv")),
        teacher_path=_normalize_path(_first_value(rows, config, "teacher_path")),
        skip_redundancy=_as_bool(_constant_value(rows, config, "skip_redundancy", warnings)),
        num_parents=_as_int(_constant_value(rows, config, "num_parents", warnings)),
        num_candidates=_as_int(_constant_value(rows, config, "num_candidates", warnings)),
        threshold_source=threshold_source,
        candidate_set_preselected=_as_bool(
            _constant_value(rows, config, "candidate_set_preselected", warnings)
        ),
        selection_performed_in_eval=_as_bool(
            _constant_value(rows, config, "selection_performed_in_eval", warnings)
        ),
        selection_method=_text(_constant_value(rows, config, "selection_method", warnings)),
        pair_distance_cache_hit_rate=_as_float(cache.get("pair_distance_cache_hit_rate")),
        node_embedding_cache_hit_rate=_as_float(cache.get("node_embedding_cache_hit_rate")),
        num_invalid_smiles=_as_int(cache.get("num_invalid_smiles")),
        num_nan_distances=_as_int(cache.get("num_nan_distances")),
        runtime_seconds=_as_float(cache.get("runtime_seconds")),
        threshold_rows=valid_rows,
        warnings=warnings,
    )
    required = {
        "distance_type": record.distance_type,
        "distance_line": record.distance_line,
        "fgw_lambda": record.fgw_lambda,
        "structure_mode": record.structure_mode,
        "feature_cost": record.feature_cost,
        "atom_penalty": record.atom_penalty,
        "cf_mode": record.cf_mode,
        "dataset_csv": record.dataset_csv,
        "teacher_path": record.teacher_path,
        "skip_redundancy": record.skip_redundancy,
        "num_parents": record.num_parents,
        "num_candidates": record.num_candidates,
        "threshold_source": record.threshold_source,
    }
    for key, value in required.items():
        if value is None:
            record.warnings.append(f"missing_required_field:{key}")
    provenance = {
        "candidate_set_preselected": record.candidate_set_preselected,
        "selection_performed_in_eval": record.selection_performed_in_eval,
        "selection_method": record.selection_method,
    }
    for key, value in provenance.items():
        if value is None:
            record.warnings.append(f"missing_protocol_provenance:{key}")
    record.audit_valid = bool(valid_rows) and not any(
        warning.startswith("missing_required_field:") for warning in record.warnings
    )
    return record


def discover_runs(
    eval_root: str | Path,
    *,
    output_dir: str | Path,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    full_only: bool = False,
    method_regex: str | None = None,
) -> tuple[list[RunRecord], list[str], int]:
    root = Path(eval_root).expanduser().resolve()
    excluded_output = Path(output_dir).expanduser().resolve()
    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None
    method_pattern = re.compile(method_regex) if method_regex else None
    warnings: list[str] = []
    records: list[RunRecord] = []
    if not root.is_dir():
        return [], [f"eval_root_missing:{root}"], 0
    config_paths = sorted(root.rglob("run_config.json"))
    if not config_paths:
        warnings.append(f"no_run_config_found:{root}")
    for config_path in config_paths:
        run_dir = config_path.parent.resolve()
        if _is_within(run_dir, excluded_output):
            continue
        match_text = run_dir.as_posix()
        if include_pattern and not include_pattern.search(match_text):
            continue
        if exclude_pattern and exclude_pattern.search(match_text):
            continue
        smoke, full = _path_kind(run_dir)
        if full_only and not full:
            continue
        try:
            config = _read_json(config_path)
        except Exception as exc:
            warnings.append(f"invalid_run_config:{config_path}:{exc}")
            continue
        summary_path = run_dir / "combined" / "combined_threshold_summary.csv"
        if not summary_path.is_file():
            if _is_node_fgw(config, []):
                warnings.append(f"missing_threshold_summary:{run_dir}")
            continue
        try:
            summary_rows = _read_csv(summary_path)
        except Exception as exc:
            warnings.append(f"invalid_threshold_summary:{summary_path}:{exc}")
            continue
        if not _is_node_fgw(config, summary_rows):
            continue
        if not summary_rows:
            warnings.append(f"empty_threshold_summary:{summary_path}")
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row_index, row in enumerate(summary_rows):
            method = _text(row.get("method"))
            if method is None:
                warnings.append(f"missing_method:{summary_path}:row={row_index}")
                continue
            if method_pattern and not method_pattern.search(method):
                continue
            grouped[method].append(row)
        cache: dict[str, Any] = {}
        cache_path = run_dir / "cache_stats.json"
        if cache_path.is_file():
            try:
                cache = _read_json(cache_path)
            except Exception as exc:
                warnings.append(f"invalid_cache_stats:{cache_path}:{exc}")
        for method, method_rows in sorted(grouped.items()):
            record = _build_record(
                eval_root=root,
                run_dir=run_dir,
                method=method,
                config=config,
                rows=method_rows,
                cache=cache,
            )
            records.append(record)
            warnings.extend(f"{record.run_id}:{warning}" for warning in record.warnings)
    return records, warnings, len(config_paths)


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _same_scalar(left: Any, right: Any) -> bool:
    return _present(left) and _present(right) and left == right


def _same_number(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    left_number = _as_float(left)
    right_number = _as_float(right)
    return bool(
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, abs_tol=atol, rel_tol=rtol)
    )


def _allclose(left: Sequence[float], right: Sequence[float], *, atol: float, rtol: float) -> bool:
    return len(left) == len(right) and bool(left) and all(
        math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
        for a, b in zip(left, right)
    )


def _threshold_pairs_by_common_quantile(
    left: RunRecord,
    right: RunRecord,
    *,
    atol: float,
    rtol: float,
) -> tuple[list[tuple[float, float]], int]:
    left_map = {
        float(row["quantile"]): float(row["threshold"])
        for row in left.threshold_rows
        if _as_float(row.get("quantile")) is not None
    }
    right_map = {
        float(row["quantile"]): float(row["threshold"])
        for row in right.threshold_rows
        if _as_float(row.get("quantile")) is not None
    }
    pairs: list[tuple[float, float]] = []
    matched_right: set[float] = set()
    for left_quantile, left_threshold in sorted(left_map.items()):
        for right_quantile, right_threshold in sorted(right_map.items()):
            if right_quantile in matched_right:
                continue
            if math.isclose(left_quantile, right_quantile, abs_tol=atol, rel_tol=rtol):
                pairs.append((left_threshold, right_threshold))
                matched_right.add(right_quantile)
                break
    if pairs:
        return pairs, len(pairs)
    if len(left.thresholds) == len(right.thresholds):
        return list(zip(sorted(left.thresholds), sorted(right.thresholds))), 0
    return [], 0


def compare_runs(left: RunRecord, right: RunRecord, *, atol: float, rtol: float) -> dict[str, Any]:
    same_distance_type = _same_scalar(left.distance_type, right.distance_type)
    same_distance_line = _same_scalar(left.distance_line, right.distance_line)
    same_fgw_lambda = _same_number(left.fgw_lambda, right.fgw_lambda, atol=atol, rtol=rtol)
    same_structure_mode = _same_scalar(left.structure_mode, right.structure_mode)
    same_feature_cost = _same_scalar(left.feature_cost, right.feature_cost)
    same_atom_penalty = _same_number(left.atom_penalty, right.atom_penalty, atol=atol, rtol=rtol)
    same_fgw_config = all(
        (
            same_distance_type,
            same_distance_line,
            same_fgw_lambda,
            same_structure_mode,
            same_feature_cost,
            same_atom_penalty,
        )
    )
    same_cf_mode = _same_scalar(left.cf_mode, right.cf_mode)
    same_dataset_csv = _same_scalar(left.dataset_csv, right.dataset_csv)
    same_teacher_path = _same_scalar(left.teacher_path, right.teacher_path)
    same_skip_redundancy = _same_scalar(left.skip_redundancy, right.skip_redundancy)
    same_num_parents = _same_scalar(left.num_parents, right.num_parents)
    same_num_candidates = _same_scalar(left.num_candidates, right.num_candidates)
    same_candidate_set_preselected = _same_scalar(
        left.candidate_set_preselected, right.candidate_set_preselected
    )
    same_selection_performed = _same_scalar(
        left.selection_performed_in_eval, right.selection_performed_in_eval
    )
    same_eval_protocol = all(
        (
            same_cf_mode,
            same_dataset_csv,
            same_teacher_path,
            same_skip_redundancy,
            same_num_candidates,
            same_candidate_set_preselected,
            same_selection_performed,
        )
    )
    same_threshold_source = _same_scalar(left.threshold_source, right.threshold_source)
    same_quantile_grid = len(left.quantiles) == len(right.quantiles) and all(
        math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        for a, b in zip(left.quantiles, right.quantiles)
    )
    same_absolute_thresholds = _allclose(
        sorted(left.thresholds), sorted(right.thresholds), atol=atol, rtol=rtol
    )
    threshold_pairs, num_common_quantiles = _threshold_pairs_by_common_quantile(
        left, right, atol=atol, rtol=rtol
    )
    absolute_differences = [abs(a - b) for a, b in threshold_pairs]
    relative_differences = [
        abs(a - b) / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 0.0 else 0.0
        for a, b in threshold_pairs
    ]
    checks = {
        "distance_type": same_distance_type,
        "distance_line": same_distance_line,
        "fgw_lambda": same_fgw_lambda,
        "structure_mode": same_structure_mode,
        "feature_cost": same_feature_cost,
        "atom_penalty": same_atom_penalty,
        "cf_mode": same_cf_mode,
        "dataset_csv": same_dataset_csv,
        "teacher_path": same_teacher_path,
        "skip_redundancy": same_skip_redundancy,
        "num_parents": same_num_parents,
        "num_candidates": same_num_candidates,
        "candidate_set_preselected": same_candidate_set_preselected,
        "selection_performed_in_eval": same_selection_performed,
        "threshold_source": same_threshold_source,
        "quantile_grid": same_quantile_grid,
        "absolute_thresholds": same_absolute_thresholds,
    }
    if left.selection_method != right.selection_method:
        checks["selection_method"] = False
    mismatches = [field for field, matches in checks.items() if not matches]
    direct_ok = bool(
        same_fgw_config
        and same_eval_protocol
        and same_num_parents
        and same_absolute_thresholds
    )
    return {
        "run_a": left.run_id,
        "run_b": right.run_id,
        "method_a": left.method,
        "method_b": right.method,
        "same_distance_type": same_distance_type,
        "same_distance_line": same_distance_line,
        "same_fgw_lambda": same_fgw_lambda,
        "same_structure_mode": same_structure_mode,
        "same_feature_cost": same_feature_cost,
        "same_atom_penalty": same_atom_penalty,
        "same_fgw_config": same_fgw_config,
        "same_cf_mode": same_cf_mode,
        "same_dataset_csv": same_dataset_csv,
        "same_teacher_path": same_teacher_path,
        "same_skip_redundancy": same_skip_redundancy,
        "same_num_parents": same_num_parents,
        "same_num_candidates": same_num_candidates,
        "same_candidate_set_preselected": same_candidate_set_preselected,
        "same_selection_performed_in_eval": same_selection_performed,
        "same_eval_protocol": same_eval_protocol,
        "same_threshold_source": same_threshold_source,
        "same_quantile_grid": same_quantile_grid,
        "same_absolute_thresholds": same_absolute_thresholds,
        "num_common_quantiles": num_common_quantiles,
        "max_abs_threshold_diff": max(absolute_differences) if absolute_differences else None,
        "mean_abs_threshold_diff": mean(absolute_differences) if absolute_differences else None,
        "max_relative_threshold_diff": max(relative_differences) if relative_differences else None,
        "mismatched_config_fields": ";".join(mismatches),
        "direct_threshold_comparison_ok": direct_ok,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _threshold_long_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        for threshold_row in record.threshold_rows:
            output.append(
                {
                    "run_id": record.run_id,
                    "run_dir": record.run_dir,
                    "method": record.method,
                    "quantile": threshold_row.get("quantile"),
                    "threshold": threshold_row.get("threshold"),
                    "threshold_source": threshold_row.get("threshold_source") or record.threshold_source,
                    "num_parents": threshold_row.get("num_parents") or record.num_parents,
                    "num_candidates": threshold_row.get("num_candidates") or record.num_candidates,
                    "close_only_coverage": threshold_row.get("close_only_coverage"),
                    "close_cf_coverage": threshold_row.get("close_cf_coverage"),
                }
            )
    return output


def _format_pairs(rows: Sequence[dict[str, Any]], predicate: str) -> list[str]:
    selected = [row for row in rows if bool(row.get(predicate))]
    return [f"- {row['run_a']} <-> {row['run_b']}" for row in selected] or ["- None"]


def build_report(
    records: Sequence[RunRecord],
    pairwise: Sequence[dict[str, Any]],
    warnings: Sequence[str],
    *,
    reference_run_id: str | None = None,
    reference_rows: Sequence[dict[str, Any]] = (),
) -> str:
    full_records = [record for record in records if record.audit_valid and record.is_full_path]
    fgw_mismatches = [row for row in pairwise if not bool(row["same_fgw_config"])]
    protocol_mismatches = [row for row in pairwise if not bool(row["same_eval_protocol"])]
    quantile_mismatches = [row for row in pairwise if not bool(row["same_quantile_grid"])]
    threshold_mismatches = [row for row in pairwise if not bool(row["same_absolute_thresholds"])]
    leakage_rows = [
        row
        for row in pairwise
        if bool(row["same_quantile_grid"])
        and not bool(row["same_absolute_thresholds"])
        and next((r.threshold_source for r in records if r.run_id == row["run_a"]), None)
        == "auto_quantile"
        and next((r.threshold_source for r in records if r.run_id == row["run_b"]), None)
        == "auto_quantile"
    ]
    lines = [
        "Node-FGW Threshold Consistency Audit",
        "",
        "1. Summary",
        f"- Valid Node-FGW runs: {sum(record.audit_valid for record in records)}",
        f"- Pairwise comparisons: {len(pairwise)}",
        f"- Directly comparable pairs: {sum(bool(row['direct_threshold_comparison_ok']) for row in pairwise)}",
        f"- Warnings: {len(warnings)}",
        "",
        "2. Full-run inventory",
    ]
    lines.extend(
        f"- {record.run_id}: method={record.method}, parents={record.num_parents}, "
        f"candidates={record.num_candidates}, source={record.threshold_source}"
        for record in full_records
    )
    if not full_records:
        lines.append("- None")
    lines.extend(["", "3. FGW configuration mismatches"])
    lines.extend(
        f"- {row['run_a']} <-> {row['run_b']}: {row['mismatched_config_fields']}"
        for row in fgw_mismatches
    )
    if not fgw_mismatches:
        lines.append("- None")
    lines.extend(["", "4. Parent/evaluation protocol mismatches"])
    lines.extend(
        f"- {row['run_a']} <-> {row['run_b']}: {row['mismatched_config_fields']}"
        for row in protocol_mismatches
    )
    if not protocol_mismatches:
        lines.append("- None")
    lines.extend(["", "5. Quantile-grid consistency"])
    lines.extend(
        f"- Mismatch: {row['run_a']} <-> {row['run_b']}" for row in quantile_mismatches
    )
    if not quantile_mismatches:
        lines.append("- All audited pairs use matching quantile grids.")
    lines.extend(["", "6. Absolute-threshold consistency"])
    lines.extend(
        f"- Mismatch: {row['run_a']} <-> {row['run_b']}; "
        f"max_abs_diff={row['max_abs_threshold_diff']}"
        for row in threshold_mismatches
    )
    if not threshold_mismatches:
        lines.append("- All audited pairs use matching absolute thresholds.")
    lines.extend(["", "7. Auto-quantile leakage warnings"])
    if leakage_rows:
        lines.append(AUTO_QUANTILE_WARNING)
        lines.extend(f"- {row['run_a']} <-> {row['run_b']}" for row in leakage_rows)
    else:
        lines.append("- No same-grid/different-radius auto-quantile pairs detected.")
    lines.extend(["", "8. Directly comparable run pairs"])
    lines.extend(_format_pairs(pairwise, "direct_threshold_comparison_ok"))
    if reference_run_id:
        lines.append(f"- Explicit reference: {reference_run_id}")
        for row in reference_rows:
            status = "directly comparable" if row["direct_threshold_comparison_ok"] else "not directly comparable"
            if row["same_quantile_grid"] and not row["same_absolute_thresholds"]:
                status = "same quantile grid, different absolute thresholds"
            lines.append(f"- Reference vs {row['run_b']}: {status}")
    lines.extend(
        [
            "",
            "9. Recommended next action",
            "- For a fair main table, choose one explicit reference run and rerun all methods with the same absolute FGW thresholds.",
            "- Keep auto_quantile for within-run diagnostics and curve exploration, not as evidence of a shared distance radius.",
            "- Verify FGW lambda, structure mode, feature cost, atom penalty, parent CSV, teacher, CF mode, and parent count before comparison.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    *,
    eval_root: str | Path,
    output_dir: str | Path,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    full_only: bool = False,
    method_regex: str | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-9,
    reference_run_id: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    records, warnings, discovered_count = discover_runs(
        eval_root,
        output_dir=output,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        full_only=full_only,
        method_regex=method_regex,
    )
    valid_records = [record for record in records if record.audit_valid]
    pairwise = [
        compare_runs(left, right, atol=float(atol), rtol=float(rtol))
        for left, right in combinations(valid_records, 2)
    ]
    leakage_pairs = [
        row
        for row in pairwise
        if bool(row["same_quantile_grid"])
        and not bool(row["same_absolute_thresholds"])
        and next(record for record in valid_records if record.run_id == row["run_a"]).threshold_source
        == "auto_quantile"
        and next(record for record in valid_records if record.run_id == row["run_b"]).threshold_source
        == "auto_quantile"
    ]
    if leakage_pairs:
        warnings.append(AUTO_QUANTILE_WARNING)

    _write_csv(output / "node_fgw_run_inventory.csv", [record.inventory_row() for record in records], INVENTORY_FIELDS)
    _write_csv(output / "node_fgw_threshold_long.csv", _threshold_long_rows(records), THRESHOLD_LONG_FIELDS)
    _write_csv(output / "node_fgw_threshold_pairwise.csv", pairwise, PAIRWISE_FIELDS)

    reference_rows: list[dict[str, Any]] = []
    if reference_run_id:
        reference = next((record for record in valid_records if record.run_id == reference_run_id), None)
        if reference is None:
            warnings.append(f"reference_run_id_not_found:{reference_run_id}")
        else:
            for record in valid_records:
                if record.run_id == reference.run_id:
                    continue
                reference_rows.append(compare_runs(reference, record, atol=float(atol), rtol=float(rtol)))
        _write_csv(output / "node_fgw_threshold_vs_reference.csv", reference_rows, PAIRWISE_FIELDS)

    audit_json = {
        "num_discovered_configs": discovered_count,
        "num_valid_node_fgw_runs": len(valid_records),
        "num_full_runs": sum(record.is_full_path for record in valid_records),
        "num_smoke_runs": sum(record.is_smoke_path for record in valid_records),
        "num_auto_quantile_runs": sum(record.threshold_source == "auto_quantile" for record in valid_records),
        "num_pairwise_comparisons": len(pairwise),
        "num_pairwise_same_quantile_grid": sum(bool(row["same_quantile_grid"]) for row in pairwise),
        "num_pairwise_same_absolute_thresholds": sum(
            bool(row["same_absolute_thresholds"]) for row in pairwise
        ),
        "num_pairwise_directly_comparable": sum(
            bool(row["direct_threshold_comparison_ok"]) for row in pairwise
        ),
        "reference_run_id": reference_run_id,
        "num_reference_comparisons": len(reference_rows),
        "atol": float(atol),
        "rtol": float(rtol),
        "warnings": warnings,
    }
    _write_json(output / "node_fgw_threshold_audit.json", audit_json)
    (output / "node_fgw_threshold_audit_report.txt").write_text(
        build_report(
            valid_records,
            pairwise,
            warnings,
            reference_run_id=reference_run_id,
            reference_rows=reference_rows,
        ),
        encoding="utf-8",
    )
    return {"records": records, "pairwise": pairwise, "summary": audit_json}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--eval-root", default="outputs/hpc/eval")
    parser.add_argument(
        "--output-dir",
        default="outputs/hpc/eval/audits/node_fgw_threshold_consistency",
    )
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--method-regex", default=None)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--reference-run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("[ERROR] --atol and --rtol must be non-negative.")
    try:
        result = run_audit(
            eval_root=args.eval_root,
            output_dir=args.output_dir,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            full_only=bool(args.full_only),
            method_regex=args.method_regex,
            atol=float(args.atol),
            rtol=float(args.rtol),
            reference_run_id=args.reference_run_id,
        )
    except re.error as exc:
        raise SystemExit(f"[ERROR] invalid regular expression: {exc}") from exc
    summary = result["summary"]
    print("[NODE_FGW_THRESHOLD_AUDIT_DONE]", flush=True)
    print(f"num_valid_node_fgw_runs={summary['num_valid_node_fgw_runs']}", flush=True)
    print(f"num_pairwise_comparisons={summary['num_pairwise_comparisons']}", flush=True)
    print(f"num_pairwise_directly_comparable={summary['num_pairwise_directly_comparable']}", flush=True)
    print(f"output_dir={Path(args.output_dir).expanduser().resolve()}", flush=True)
    if args.strict and summary["warnings"]:
        print(f"[NODE_FGW_THRESHOLD_AUDIT_STRICT_FAILED] warnings={len(summary['warnings'])}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
