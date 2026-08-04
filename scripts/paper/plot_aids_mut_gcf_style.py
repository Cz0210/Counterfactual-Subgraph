#!/usr/bin/env python3
"""Plot the frozen AIDS and Mutagenicity four-method WNode comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


DISTANCE_LABEL = "MolCLR-Node-Wasserstein"
DISTANCE_TYPE = "node_wasserstein"
CF_MODE = "strict_flip"
METHODS = ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
DATASETS = ("AIDS", "Mutagenicity")
EXPECTED_PARENT_COUNTS = {"AIDS": 1283, "Mutagenicity": 217}
EXPECTED_FIGURE4_COUNTS = {"AIDS": 102, "Mutagenicity": 7}
FORBIDDEN_SOURCE_TOKENS = (
    "ccrcov_molclr_node_fgw_",
    "node_fgw",
    "lam05",
    "gt_fullgraph",
    "opposite_fullgraph",
    "opposite-label",
)

METHOD_STYLE = {
    "Ours": {"color": "#202020", "marker": "o"},
    "GlobalGCE": {"color": "#e68613", "marker": "s"},
    "CLEAR": {"color": "#2e7d32", "marker": "^"},
    "GCFExplainer": {"color": "#2563a8", "marker": "D"},
}
K_MARKERS = {1, 3, 5, 10, 15, 20}


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    lookup = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalize(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Required column not found. candidates={list(candidates)}, "
            f"columns={frame.columns.tolist()}"
        )
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty CSV: {path}")
    return frame


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in column {column}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_scalars(value: object, prefix: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _json_scalars(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _json_scalars(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _json_strings(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                yield key
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)
    elif isinstance(value, str):
        yield value


def _referenced_provenance_files(root: Path, project_root: Path) -> list[Path]:
    """Follow only local manifest/audit references needed to prove provenance."""

    queue: list[Path] = []
    seen: set[Path] = set()

    def metadata_files(directory: Path) -> list[Path]:
        resolved = directory.expanduser().resolve()
        if not resolved.is_dir() or not _path_is_within(resolved, project_root):
            return []
        names = (
            "*manifest*.json",
            "*audit*.json",
            "summary.json",
            "run_manifest.json",
            "run_config.json",
        )
        matches: list[Path] = []
        for pattern in names:
            matches.extend(sorted(resolved.glob(pattern)))
        return sorted(set(matches))

    def referenced_path(value: str, *, relative_to: Path) -> Path | None:
        if not (
            Path(value).is_absolute()
            or "/" in value
            or "\\" in value
            or Path(value).suffix.lower() in {".csv", ".json", ".jsonl", ".pt"}
        ):
            return None
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            project_candidate = project_root / candidate
            local_candidate = relative_to / candidate
            candidate = (
                project_candidate
                if project_candidate.exists() or not local_candidate.exists()
                else local_candidate
            )
        try:
            candidate = candidate.resolve()
        except OSError:
            return None
        return candidate if _path_is_within(candidate, project_root) else None

    def manifest_references_root(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        for scalar in _json_strings(payload):
            if not scalar.strip():
                continue
            candidate = referenced_path(scalar, relative_to=path.parent)
            if candidate is not None and (
                candidate == root or _path_is_within(candidate, root)
            ):
                return True
        return False

    queue.extend(metadata_files(root))
    queue.extend(
        path for path in metadata_files(root.parent) if manifest_references_root(path)
    )
    output: list[Path] = []
    while queue and len(seen) < 96:
        path = queue.pop(0).resolve()
        if path in seen or not path.is_file() or not _path_is_within(path, project_root):
            continue
        seen.add(path)
        output.append(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for scalar in _json_strings(payload):
            if not scalar.strip():
                continue
            candidate = referenced_path(scalar, relative_to=path.parent)
            if candidate is None:
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".json":
                queue.append(candidate)
            elif candidate.is_file():
                queue.extend(metadata_files(candidate.parent))
            elif candidate.is_dir():
                queue.extend(metadata_files(candidate))
    return sorted(set(output))


def _metadata_evidence(
    *,
    root: Path,
    project_root: Path,
    csv_paths: Sequence[Path],
) -> dict[str, Any]:
    evidence: dict[str, list[Any]] = {
        "distance_labels": [],
        "distance_types": [],
        "cf_modes": [],
        "parent_counts": [],
        "candidate_counts": [],
        "candidate_set_preselected": [],
        "selection_performed_in_eval": [],
        "strict_flip_evidence": [],
        "provenance_files": [],
    }
    forbidden_hits: list[str] = []

    def record(key: str, value: object) -> None:
        if value is None or value == "":
            return
        if value not in evidence[key]:
            evidence[key].append(value)

    text_sources = [str(root)]
    for path in csv_paths:
        frame = read_csv(path)
        for column in frame.columns:
            key = normalize(column)
            values = frame[column].dropna().unique().tolist()
            if key in {"distance", "distance_label", "distance_line"}:
                for value in values:
                    record("distance_labels", str(value))
            elif key == "distance_type":
                for value in values:
                    record("distance_types", str(value))
            elif key == "cf_mode":
                for value in values:
                    record("cf_modes", str(value))
            elif key in {"num_parents", "test_parent_count"}:
                for value in values:
                    record("parent_counts", int(float(value)))
            elif key == "candidate_set_preselected":
                for value in values:
                    parsed = _as_bool(value)
                    if parsed is not None:
                        record("candidate_set_preselected", parsed)
            elif key == "selection_performed_in_eval":
                for value in values:
                    parsed = _as_bool(value)
                    if parsed is not None:
                        record("selection_performed_in_eval", parsed)
            elif "strict_flip" in key:
                record("strict_flip_evidence", f"{path.name}:{column}")

    provenance_files = _referenced_provenance_files(root, project_root)
    for path in provenance_files:
        record("provenance_files", {"path": str(path), "sha256": sha256(path)})
        payload = json.loads(path.read_text(encoding="utf-8"))
        text_sources.append(json.dumps(payload, sort_keys=True))
        for key_path, value in _json_scalars(payload):
            key_components = [
                normalize(component.split("[", 1)[0])
                for component in key_path.split(".")
            ]
            if any(
                component in {
                    "checks",
                    "semantic_checks",
                    "required_fields",
                    "forbidden_fields",
                }
                for component in key_components[:-1]
            ):
                continue
            key = key_components[-1]
            if key in {"distance", "distance_label", "distance_line"}:
                record("distance_labels", str(value))
            elif key == "distance_type":
                record("distance_types", str(value))
            elif key == "cf_mode":
                record("cf_modes", str(value))
            elif key in {
                "expected_num_parents",
                "num_parents",
                "parent_count",
                "reference_parent_count",
                "test_parent_count",
            }:
                try:
                    record("parent_counts", int(value))
                except (TypeError, ValueError):
                    pass
            elif key in {"candidate_count", "num_candidates"}:
                try:
                    record("candidate_counts", int(value))
                except (TypeError, ValueError):
                    pass
            elif key in {
                "candidate_set_preselected",
                "candidate_order_frozen",
                "candidate_order_exact_match",
                "candidate_order_matches_frozen",
            }:
                parsed = _as_bool(value)
                if parsed is True:
                    record("candidate_set_preselected", parsed)
            elif key in {
                "selection_performed_in_eval",
                "candidate_selection_performed",
                "test_candidate_selection",
            }:
                parsed = _as_bool(value)
                if parsed is not None:
                    record("selection_performed_in_eval", parsed)
            elif "strict_flip" in key:
                parsed = _as_bool(value)
                try:
                    positive_count = float(value) > 0
                except (TypeError, ValueError):
                    positive_count = False
                if parsed is True or positive_count:
                    record("strict_flip_evidence", f"{path}:{key_path}")
            elif key == "fgw_lambda":
                forbidden_hits.append(f"{path}:{key_path}")

    combined_text = "\n".join(text_sources).lower()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in combined_text:
            forbidden_hits.append(token)
    if "molclr-node-fgw" in combined_text or '"distance_type": "node_fgw"' in combined_text:
        forbidden_hits.append("Node-FGW distance metadata")
    if forbidden_hits:
        raise ValueError(
            f"Forbidden non-WNode provenance under {root}: {sorted(set(forbidden_hits))}"
        )

    labels = {normalize(value) for value in evidence["distance_labels"]}
    types = {normalize(value) for value in evidence["distance_types"]}
    if normalize(DISTANCE_LABEL) not in labels:
        raise ValueError(
            f"WNode distance label is not proven for {root}; observed={evidence['distance_labels']}"
        )
    if types and types != {DISTANCE_TYPE}:
        raise ValueError(f"Unexpected distance_type for {root}: {evidence['distance_types']}")
    if not types:
        raise ValueError(f"WNode distance_type is not proven for {root}.")
    modes = {normalize(value) for value in evidence["cf_modes"]}
    if modes and modes != {CF_MODE}:
        raise ValueError(f"Unexpected cf_mode for {root}: {evidence['cf_modes']}")

    evidence["distance_label_verified"] = True
    evidence["distance_type_verified"] = True
    evidence["strict_flip_verified"] = (
        modes == {CF_MODE} or bool(evidence["strict_flip_evidence"])
    )
    return evidence


def table2_path(root: Path, method: str) -> Path:
    preferred = {
        "Ours": "table2_ours_k10.csv",
        "GlobalGCE": "table2_globalgce_k10.csv",
        "CLEAR": "table2_clear_k10.csv",
        "GCFExplainer": "table2_gcfexplainer_k10.csv",
    }[method]
    candidate = root / preferred
    if candidate.is_file():
        return candidate
    matches = sorted(root.glob("table2_*_k10.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{method}: expected one K=10 Table 2 CSV under {root}, "
            f"found {[str(path) for path in matches]}"
        )
    return matches[0]


def _standard_plot_paths(root: Path, method: str) -> list[Path]:
    return [
        root / "figure3_coverage_vs_k.csv",
        root / "figure4_coverage_vs_threshold.csv",
        table2_path(root, method),
    ]


def _is_raw_aids_gcf_run(root: Path, *, dataset: str, method: str) -> bool:
    if dataset != "AIDS" or method != "GCFExplainer" or not root.is_dir():
        return False
    if (root / "figure3_coverage_vs_k.csv").is_file():
        return False
    return (root / "run_config.json").is_file() and (
        root / "combined" / "combined_threshold_summary.csv"
    ).is_file()


def parse_roots(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    return {
        "AIDS": {
            "Ours": Path(args.aids_ours_root).resolve(),
            "GlobalGCE": Path(args.aids_globalgce_root).resolve(),
            "CLEAR": Path(args.aids_clear_root).resolve(),
            "GCFExplainer": Path(args.aids_gcf_root).resolve(),
        },
        "Mutagenicity": {
            "Ours": Path(args.mut_ours_root).resolve(),
            "GlobalGCE": Path(args.mut_globalgce_root).resolve(),
            "CLEAR": Path(args.mut_clear_root).resolve(),
            "GCFExplainer": Path(args.mut_gcf_root).resolve(),
        },
    }


def validate_root(
    root: Path,
    *,
    project_root: Path,
    dataset: str,
    method: str,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing frozen plotting root: {root}")
    paths = _standard_plot_paths(root, method)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    evidence = _metadata_evidence(root=root, project_root=project_root, csv_paths=paths)
    expected_parents = EXPECTED_PARENT_COUNTS[dataset]
    observed_parent_counts = set(evidence["parent_counts"])
    if expected_parents not in observed_parent_counts:
        raise ValueError(
            f"{dataset}/{method}: expected parent count {expected_parents} is not proven; "
            f"observed={sorted(observed_parent_counts)}"
        )
    if True not in evidence["candidate_set_preselected"]:
        raise ValueError(f"{dataset}/{method}: frozen candidate order is not proven.")
    if True in evidence["selection_performed_in_eval"]:
        raise ValueError(f"{dataset}/{method}: selection was performed in evaluation.")
    if False not in evidence["selection_performed_in_eval"]:
        raise ValueError(f"{dataset}/{method}: no-selection provenance is not proven.")
    if evidence["strict_flip_verified"] is not True:
        raise ValueError(f"{dataset}/{method}: strict_flip provenance is not proven.")
    return evidence


def _json_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def standardize_raw_aids_gcf_run(
    root: Path,
    *,
    project_root: Path,
    theta_star: float,
    figure4_thresholds: Sequence[float],
    figure4_threshold_source: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Aggregate one frozen GCFExplainer WNode run without recomputing pairs."""

    from src.eval.gcf_style_recourse_report import (
        compute_k_curve,
        compute_prefix_metrics,
        load_method_run,
    )

    if not _is_raw_aids_gcf_run(root, dataset="AIDS", method="GCFExplainer"):
        raise ValueError(f"Not a supported raw AIDS GCFExplainer run root: {root}")
    combined_path = root / "combined" / "combined_threshold_summary.csv"
    config_path = root / "run_config.json"
    details_path = root / "details" / "pair_details.csv"
    cache_stats_path = root / "cache_stats.json"
    complete_path = root / "_RUN_COMPLETE.json"
    for path in (config_path, combined_path, details_path, cache_stats_path, complete_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    evidence = _metadata_evidence(
        root=root,
        project_root=project_root,
        csv_paths=[combined_path],
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(complete, dict):
        raise ValueError(f"Invalid GCFExplainer run JSON under {root}")
    if complete.get("complete") is not True and complete.get("run_complete") is not True:
        raise ValueError(f"GCFExplainer WNode run is not complete: {root}")
    if config.get("distance_line") != DISTANCE_LABEL:
        raise ValueError(f"Unexpected GCFExplainer distance_line: {config.get('distance_line')}")
    if config.get("distance_type") != DISTANCE_TYPE:
        raise ValueError(f"Unexpected GCFExplainer distance_type: {config.get('distance_type')}")
    if config.get("cf_mode") != CF_MODE:
        raise ValueError(f"Unexpected GCFExplainer cf_mode: {config.get('cf_mode')}")

    run = load_method_run(
        "GCFExplainer",
        root,
        expected_top_k=20,
        expected_num_parents=EXPECTED_PARENT_COUNTS["AIDS"],
    )
    if run.num_unique_parent_candidate_pairs != EXPECTED_PARENT_COUNTS["AIDS"] * 20:
        raise ValueError(
            "GCFExplainer pair details are not a complete 1283x20 Cartesian matrix: "
            f"found={run.num_unique_parent_candidate_pairs}"
        )
    candidate_ranks = [candidate.rank for candidate in run.candidates]
    if candidate_ranks != list(range(1, 21)):
        raise ValueError(f"GCFExplainer frozen ranks must be 1..20: {candidate_ranks}")

    summary = read_csv(combined_path)
    threshold_col = find_column(summary, ["threshold", "theta"])
    summary_coverage_col = find_column(
        summary,
        ["close_cf_coverage", "ccrcov", "coverage"],
    )
    summary_thresholds = np.sort(numeric(summary, threshold_col).unique().astype(float))
    if len(summary_thresholds) == 0:
        raise ValueError(
            "GCFExplainer combined summary does not contain any frozen thresholds"
        )
    if not np.any(np.isclose(summary_thresholds, theta_star, rtol=0.0, atol=1e-12)):
        raise ValueError(f"GCFExplainer summary does not include theta*={theta_star:.17g}")
    reconstructed_summary: list[dict[str, float]] = []
    for threshold in summary_thresholds:
        matching = summary.loc[
            np.isclose(
                numeric(summary, threshold_col).to_numpy(dtype=float),
                float(threshold),
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                "GCFExplainer combined summary must have one row per threshold: "
                f"threshold={threshold:.17g} rows={len(matching)}"
            )
        expected_coverage = float(matching.iloc[0][summary_coverage_col])
        actual_coverage = float(
            compute_prefix_metrics(run, k=20, threshold=float(threshold))["coverage"]
        )
        if not math.isclose(actual_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "GCFExplainer saved-pair reconstruction disagrees with official summary: "
                f"threshold={threshold:.17g} expected={expected_coverage:.17g} "
                f"actual={actual_coverage:.17g}"
            )
        reconstructed_summary.append(
            {
                "threshold": float(threshold),
                "expected_coverage": expected_coverage,
                "actual_coverage": actual_coverage,
            }
        )

    dense_thresholds = np.asarray(list(figure4_thresholds), dtype=float)
    if (
        dense_thresholds.ndim != 1
        or len(dense_thresholds) != EXPECTED_FIGURE4_COUNTS["AIDS"]
        or not np.isfinite(dense_thresholds).all()
    ):
        raise ValueError(
            "AIDS standardized Figure 4 reference must contain 102 finite thresholds"
        )
    dense_thresholds = np.sort(dense_thresholds)
    if len(np.unique(dense_thresholds)) != len(dense_thresholds):
        raise ValueError("AIDS standardized Figure 4 reference thresholds are duplicated")

    curve_rows = compute_k_curve(run, threshold=theta_star, max_k=20)
    figure3 = pd.DataFrame(
        {
            "Dataset": "AIDS",
            "Method": "GCFExplainer",
            "K": [int(row["k"]) for row in curve_rows],
            "Coverage": [float(row["coverage"]) for row in curve_rows],
            "ConditionalMedianCost": [
                float(row["conditional_median_cost"]) for row in curve_rows
            ],
        }
    )
    if not np.isfinite(
        figure3[["Coverage", "ConditionalMedianCost"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("GCFExplainer Figure 3 contains non-finite frozen metrics")
    if not figure3["Coverage"].between(0, 1).all():
        raise ValueError("GCFExplainer Figure 3 coverage is outside [0,1]")
    threshold_metrics = [
        compute_prefix_metrics(run, k=10, threshold=float(threshold))
        for threshold in dense_thresholds
    ]
    figure4 = pd.DataFrame(
        {
            "Dataset": "AIDS",
            "Method": "GCFExplainer",
            "K": 10,
            "ThresholdName": "",
            "Threshold": dense_thresholds,
            "Coverage": [float(row["coverage"]) for row in threshold_metrics],
        }
    )
    if not figure4["Coverage"].between(0, 1).all():
        raise ValueError("GCFExplainer Figure 4 coverage is outside [0,1]")
    table_metrics = compute_prefix_metrics(run, k=10, threshold=theta_star)
    table = {
        "Dataset": "AIDS",
        "Method": "GCFExplainer",
        "K": 10,
        "Theta": float(theta_star),
        "Coverage": float(table_metrics["coverage"]),
        "CoveragePercent": 100.0 * float(table_metrics["coverage"]),
        "Cost": float(table_metrics["median_cost"]),
    }
    if not math.isfinite(table["Cost"]) or table["Cost"] < 0:
        raise ValueError("GCFExplainer K=10 frozen median cost is not finite and non-negative")

    candidate_order = [
        {"rank": candidate.rank, "candidate_id": candidate.candidate_id}
        for candidate in run.candidates
    ]
    parent_order = [{"parent_id": parent_id} for parent_id in run.parent_ids]
    config_sha256 = sha256(config_path)
    combined_sha256 = sha256(combined_path)
    details_sha256 = sha256(details_path)
    candidate_sha256 = sha256(run.candidate_path)
    common_source = {
        "raw_run_root": str(root),
        "standardized_from_saved_pairs": True,
        "distance_recomputed": False,
        "teacher_recomputed": False,
        "candidate_order_changed": False,
        "run_config": {"path": str(config_path), "sha256": config_sha256},
        "combined_summary": {"path": str(combined_path), "sha256": combined_sha256},
        "official_summary_threshold_count": len(summary_thresholds),
        "official_summary_reconstruction": reconstructed_summary,
        "figure4_threshold_grid_source": dict(figure4_threshold_source),
        "pair_details": {"path": str(details_path), "sha256": details_sha256},
        "candidate_file": {
            "path": str(run.candidate_path),
            "sha256": candidate_sha256,
        },
        "candidate_rank_source": run.rank_source,
        "candidate_order_sha256": _json_order_sha256(candidate_order),
        "parent_ids_sha256": _json_order_sha256(parent_order),
        "num_parents": len(run.parent_ids),
        "num_candidates": len(run.candidates),
        "num_unique_parent_candidate_pairs": run.num_unique_parent_candidate_pairs,
    }
    figure3_source = {
        **common_source,
        "path": str(details_path),
        "sha256": details_sha256,
        "columns": {
            "k": "frozen_candidate_rank_prefix",
            "coverage": "strict_flip_distance_le_theta",
            "cost": "conditional_median_cost",
        },
    }
    figure4_source = {
        **common_source,
        "path": str(combined_path),
        "sha256": combined_sha256,
        "duplicate_rows_removed": 0,
        "columns": {
            "k": "frozen_candidate_rank_prefix_k10",
            "threshold_name": "",
            "threshold": threshold_col,
            "coverage": "strict_flip_distance_le_threshold",
        },
    }
    table_source = {
        **common_source,
        "path": str(details_path),
        "sha256": details_sha256,
        "columns": {
            "k": "frozen_candidate_rank_prefix_k10",
            "theta": "AIDS_frozen_theta_star",
            "coverage": "strict_flip_distance_le_theta",
            "cost": "median_cost",
        },
    }
    evidence.update(
        {
            "raw_run_standardized": True,
            "candidate_order_sha256": common_source["candidate_order_sha256"],
            "parent_ids_sha256": common_source["parent_ids_sha256"],
            "top20_frozen_ranking_verified": True,
            "complete_cartesian_verified": True,
        }
    )
    return figure3, figure4, table, figure3_source, figure4_source, table_source, evidence


def read_figure3(root: Path, dataset: str, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / "figure3_coverage_vs_k.csv"
    frame = read_csv(path)
    k_col = find_column(frame, ["k", "prefix_k", "num_candidates", "candidate_count"])
    coverage_col = find_column(
        frame,
        [
            "ccrcov_theta_star",
            "coverage_at_theta_star",
            "close_cf_coverage_at_theta_star",
            "close_cf_coverage",
            "ccrcov",
            "coverage",
        ],
    )
    cost_col = find_column(
        frame,
        [
            "conditional_median_cost",
            "applicable_parent_median_cost",
            "covered_parent_median_cost",
            "cost",
        ],
    )
    output = pd.DataFrame(
        {
            "Dataset": dataset,
            "Method": method,
            "K": numeric(frame, k_col).astype(int),
            "Coverage": numeric(frame, coverage_col).astype(float),
            "ConditionalMedianCost": numeric(frame, cost_col).astype(float),
        }
    ).sort_values("K")
    output = output.loc[output["K"].between(1, 20)].reset_index(drop=True)
    if output["K"].tolist() != list(range(1, 21)):
        raise ValueError(
            f"{dataset}/{method}: Figure 3 requires K=1,...,20; "
            f"observed={output['K'].tolist()}"
        )
    if not output["Coverage"].between(0, 1).all():
        raise ValueError(f"{dataset}/{method}: coverage outside [0,1]")
    if (output["ConditionalMedianCost"] < 0).any():
        raise ValueError(f"{dataset}/{method}: negative conditional median cost")
    return output, {
        "path": str(path),
        "sha256": sha256(path),
        "columns": {"k": k_col, "coverage": coverage_col, "cost": cost_col},
    }


def _deduplicate_threshold_rows(
    frame: pd.DataFrame,
    *,
    threshold_col: str,
    coverage_col: str,
    dataset: str,
    method: str,
) -> tuple[pd.DataFrame, int]:
    keep: list[int] = []
    removed = 0
    for _threshold, group in frame.groupby(threshold_col, sort=False):
        coverage = numeric(group, coverage_col).to_numpy(dtype=float)
        if not np.allclose(coverage, coverage[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"{dataset}/{method}: conflicting duplicate threshold rows")
        keep.append(int(group.index[0]))
        removed += len(group) - 1
    return frame.loc[keep].copy(), removed


def read_figure4(root: Path, dataset: str, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / "figure4_coverage_vs_threshold.csv"
    frame = read_csv(path)
    k_col = find_column(frame, ["k", "prefix_k", "num_candidates", "candidate_count"], required=False)
    if k_col is not None:
        k_values = numeric(frame, k_col).astype(int)
        if 10 not in set(k_values):
            raise ValueError(f"{dataset}/{method}: Figure 4 has no K=10")
        frame = frame.loc[k_values == 10].copy()
    threshold_col = find_column(frame, ["threshold", "theta", "distance_threshold"])
    coverage_col = find_column(frame, ["ccrcov", "coverage", "close_cf_coverage", "strict_flip_coverage"])
    frame, duplicate_rows_removed = _deduplicate_threshold_rows(
        frame,
        threshold_col=threshold_col,
        coverage_col=coverage_col,
        dataset=dataset,
        method=method,
    )
    threshold_name_col = find_column(
        frame,
        ["threshold_name", "quantile_label"],
        required=False,
    )
    output = pd.DataFrame(
        {
            "Dataset": dataset,
            "Method": method,
            "K": 10,
            "ThresholdName": (
                frame[threshold_name_col].astype(str)
                if threshold_name_col is not None
                else ""
            ),
            "Threshold": numeric(frame, threshold_col).astype(float),
            "Coverage": numeric(frame, coverage_col).astype(float),
        }
    ).sort_values("Threshold").reset_index(drop=True)
    if output["Threshold"].duplicated().any():
        raise ValueError(f"{dataset}/{method}: duplicate thresholds remain")
    if not output["Coverage"].between(0, 1).all():
        raise ValueError(f"{dataset}/{method}: coverage outside [0,1]")
    if not output["Coverage"].is_monotonic_increasing:
        raise ValueError(f"{dataset}/{method}: Figure 4 coverage is not monotonic")
    expected_count = EXPECTED_FIGURE4_COUNTS[dataset]
    if len(output) != expected_count:
        raise ValueError(
            f"{dataset}/{method}: expected {expected_count} frozen thresholds, found {len(output)}"
        )
    return output, {
        "path": str(path),
        "sha256": sha256(path),
        "duplicate_rows_removed": duplicate_rows_removed,
        "columns": {
            "k": k_col or "",
            "threshold_name": threshold_name_col or "",
            "threshold": threshold_col,
            "coverage": coverage_col,
        },
    }


def read_table2(root: Path, dataset: str, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = table2_path(root, method)
    frame = read_csv(path)
    k_col = find_column(frame, ["k"])
    frame = frame.loc[numeric(frame, k_col).astype(int) == 10].copy()
    if len(frame) != 1:
        raise ValueError(f"{dataset}/{method}: Table 2 needs one K=10 row")
    theta_col = find_column(frame, ["theta", "theta_star", "threshold"])
    coverage_col = find_column(frame, ["coverage", "ccrcov_theta_star", "close_cf_coverage", "ccrcov"])
    cost_col = find_column(
        frame,
        ["conditional_median_cost", "applicable_parent_median_cost", "cost"],
    )
    row = frame.iloc[0]
    result = {
        "Dataset": dataset,
        "Method": method,
        "K": 10,
        "Theta": float(row[theta_col]),
        "Coverage": float(row[coverage_col]),
        "CoveragePercent": 100.0 * float(row[coverage_col]),
        "Cost": float(row[cost_col]),
    }
    if not 0.0 <= result["Coverage"] <= 1.0:
        raise ValueError(f"{dataset}/{method}: invalid Table 2 coverage")
    if not math.isfinite(result["Cost"]) or result["Cost"] < 0:
        raise ValueError(f"{dataset}/{method}: invalid Table 2 cost")
    return result, {
        "path": str(path),
        "sha256": sha256(path),
        "columns": {
            "k": k_col,
            "theta": theta_col,
            "coverage": coverage_col,
            "cost": cost_col,
        },
    }


def marker_indices_for_k(values: pd.Series) -> list[int]:
    return [index for index, value in enumerate(values.tolist()) if int(value) in K_MARKERS]


def marker_indices_evenly(count: int, target: int = 6) -> list[int]:
    if count <= target:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, target).round().astype(int).tolist()))


def dataset_threshold_xlim(values: np.ndarray) -> tuple[float, float]:
    minimum = min(0.0, float(np.min(values)))
    maximum = float(np.max(values))
    if maximum <= 0:
        return 0.0, 1.0
    step = 0.005 if maximum <= 0.06 else 0.01
    return minimum, math.ceil(maximum / step) * step


def _validate_dataset_alignment(
    *,
    figure4: pd.DataFrame,
    table: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        table_rows = table.loc[table["Dataset"] == dataset]
        thetas = table_rows["Theta"].to_numpy(dtype=float)
        if len(thetas) != len(METHODS) or not np.allclose(thetas, thetas[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"{dataset}: Table 2 methods do not share one frozen theta")
        method_grids: list[np.ndarray] = []
        for method in METHODS:
            rows = figure4.loc[(figure4["Dataset"] == dataset) & (figure4["Method"] == method)]
            method_grids.append(rows.sort_values("Threshold")["Threshold"].to_numpy(dtype=float))
        first = method_grids[0]
        if any(
            len(grid) != len(first) or not np.allclose(grid, first, rtol=0.0, atol=1e-12)
            for grid in method_grids[1:]
        ):
            raise ValueError(f"{dataset}: Figure 4 threshold grids differ across methods")
        result[dataset] = {
            "parent_count": EXPECTED_PARENT_COUNTS[dataset],
            "theta_star": float(thetas[0]),
            "figure4_threshold_count": len(first),
            "figure4_threshold_min": float(first.min()),
            "figure4_threshold_max": float(first.max()),
        }
    return result


def write_table_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    pivot = table.pivot(index="Method", columns="Dataset")
    compact = pd.DataFrame(
        [
            {
                "Method": method,
                "AIDS Coverage (%)": float(pivot.loc[method, ("CoveragePercent", "AIDS")]),
                "AIDS Cost": float(pivot.loc[method, ("Cost", "AIDS")]),
                "Mutagenicity Coverage (%)": float(
                    pivot.loc[method, ("CoveragePercent", "Mutagenicity")]
                ),
                "Mutagenicity Cost": float(pivot.loc[method, ("Cost", "Mutagenicity")]),
            }
            for method in METHODS
        ]
    )
    compact.to_csv(output_dir / "table2_aids_mut_gcf_style.csv", index=False)
    md = [
        "| Method | AIDS Coverage (%) | AIDS Cost | Mutagenicity Coverage (%) | Mutagenicity Cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in compact.iterrows():
        md.append(
            f"| {row['Method']} | {row['AIDS Coverage (%)']:.2f} | "
            f"{row['AIDS Cost']:.4f} | "
            f"{row['Mutagenicity Coverage (%)']:.2f} | "
            f"{row['Mutagenicity Cost']:.4f} |"
        )
    (output_dir / "table2_aids_mut_gcf_style.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12.5, 3.3))
    ax.axis("off")
    cell_text = [
        [
            row["Method"],
            f"{row['AIDS Coverage (%)']:.2f}%",
            f"{row['AIDS Cost']:.4f}",
            f"{row['Mutagenicity Coverage (%)']:.2f}%",
            f"{row['Mutagenicity Cost']:.4f}",
        ]
        for _, row in compact.iterrows()
    ]
    artist = ax.table(
        cellText=cell_text,
        colLabels=[
            "Method",
            "AIDS Coverage",
            "AIDS Cost",
            "Mutagenicity Coverage",
            "Mutagenicity Cost",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    artist.scale(1.0, 1.55)
    for dataset, coverage_column, cost_column in (("AIDS", 1, 2), ("Mutagenicity", 3, 4)):
        best_coverage = int(compact[f"{dataset} Coverage (%)"].idxmax()) + 1
        best_cost = int(compact[f"{dataset} Cost"].idxmin()) + 1
        artist[(best_coverage, coverage_column)].get_text().set_weight("bold")
        artist[(best_cost, cost_column)].get_text().set_weight("bold")
    theta_aids = table.loc[table["Dataset"] == "AIDS", "Theta"].iloc[0]
    theta_mut = table.loc[table["Dataset"] == "Mutagenicity", "Theta"].iloc[0]
    ax.set_title(
        "WNode Global Recourse Comparison at K=10\n"
        f"AIDS theta*={theta_aids:.6f}; Mutagenicity theta*={theta_mut:.6f}",
        fontsize=13,
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "table2_aids_mut_gcf_style.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "table2_aids_mut_gcf_style.pdf", bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AIDS + Mutagenicity four-method WNode Figure 3, Figure 4, and Table 2."
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--aids-ours-root", required=True)
    parser.add_argument("--aids-globalgce-root", required=True)
    parser.add_argument("--aids-clear-root", required=True)
    parser.add_argument("--aids-gcf-root", required=True)
    parser.add_argument("--mut-ours-root", required=True)
    parser.add_argument("--mut-globalgce-root", required=True)
    parser.add_argument("--mut-clear-root", required=True)
    parser.add_argument("--mut-gcf-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coverage-ymax", type=float, default=100.0)
    parser.add_argument("--cost-ymin", type=float, default=0.0)
    parser.add_argument("--cost-ymax", type=float, default=None)
    return parser


def _write_audit(
    *,
    output_dir: Path,
    datasets: Mapping[str, Mapping[str, Path]],
    dataset_audit: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    lines = [
        "AIDS + Mutagenicity four-method WNode plot audit",
        f"distance_label={DISTANCE_LABEL}",
        f"distance_type={DISTANCE_TYPE}",
        f"cf_mode={CF_MODE}",
        f"methods={','.join(METHODS)}",
    ]
    for dataset in DATASETS:
        info = dataset_audit[dataset]
        lines.extend(
            (
                f"{dataset}.parent_count={info['parent_count']}",
                f"{dataset}.theta_star={info['theta_star']:.17g}",
                f"{dataset}.figure4_threshold_count={info['figure4_threshold_count']}",
                f"{dataset}.figure4_threshold_min={info['figure4_threshold_min']:.17g}",
                f"{dataset}.figure4_threshold_max={info['figure4_threshold_max']:.17g}",
            )
        )
        for method in METHODS:
            lines.append(f"{dataset}.{method}.root={datasets[dataset][method]}")
            lines.append(
                f"{dataset}.{method}.distance_verified="
                f"{sources['provenance'][dataset][method]['distance_type_verified']}"
            )
    lines.append("[AIDS_MUT_GCF_STYLE_PLOT_OK]")
    (output_dir / "combined_audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    datasets = parse_roots(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if output_dir.exists():
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, dict[str, Any]] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        if tuple(datasets[dataset]) != METHODS:
            raise ValueError(f"{dataset}: methods must be exactly {METHODS}")

    figure3_parts: list[pd.DataFrame] = []
    figure4_parts: list[pd.DataFrame] = []
    table_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {
        "figure3": [],
        "figure4": [],
        "table2": [],
        "provenance": provenance,
    }
    for dataset in DATASETS:
        for method in METHODS:
            root = datasets[dataset][method]
            if _is_raw_aids_gcf_run(root, dataset=dataset, method=method):
                aids_table_rows = [
                    row
                    for row in table_rows
                    if row["Dataset"] == "AIDS" and row["Method"] == "Ours"
                ]
                if len(aids_table_rows) != 1:
                    raise RuntimeError(
                        "AIDS Ours Table 2 must be loaded before raw GCFExplainer standardization"
                    )
                aids_figure4_rows = [
                    frame
                    for frame in figure4_parts
                    if set(frame["Dataset"]) == {"AIDS"}
                    and set(frame["Method"]) == {"Ours"}
                ]
                aids_figure4_sources = [
                    source
                    for source in source_manifest["figure4"]
                    if source["dataset"] == "AIDS" and source["method"] == "Ours"
                ]
                if len(aids_figure4_rows) != 1 or len(aids_figure4_sources) != 1:
                    raise RuntimeError(
                        "AIDS Ours Figure 4 grid must be loaded before raw GCFExplainer "
                        "standardization"
                    )
                (
                    figure3,
                    figure4,
                    table_row,
                    source3,
                    source4,
                    source2,
                    raw_evidence,
                ) = standardize_raw_aids_gcf_run(
                    root,
                    project_root=project_root,
                    theta_star=float(aids_table_rows[0]["Theta"]),
                    figure4_thresholds=aids_figure4_rows[0]["Threshold"].tolist(),
                    figure4_threshold_source=aids_figure4_sources[0],
                )
                provenance[dataset][method] = raw_evidence
            else:
                provenance[dataset][method] = validate_root(
                    root,
                    project_root=project_root,
                    dataset=dataset,
                    method=method,
                )
                figure3, source3 = read_figure3(root, dataset, method)
                figure4, source4 = read_figure4(root, dataset, method)
                table_row, source2 = read_table2(root, dataset, method)
            figure3_parts.append(figure3)
            figure4_parts.append(figure4)
            table_rows.append(table_row)
            source_manifest["figure3"].append({"dataset": dataset, "method": method, **source3})
            source_manifest["figure4"].append({"dataset": dataset, "method": method, **source4})
            source_manifest["table2"].append({"dataset": dataset, "method": method, **source2})

    figure3 = pd.concat(figure3_parts, ignore_index=True)
    figure4 = pd.concat(figure4_parts, ignore_index=True)
    table = pd.DataFrame(table_rows)
    dataset_audit = _validate_dataset_alignment(figure4=figure4, table=table)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    try:
        figure3.to_csv(temp_dir / "figure3_aids_mut_source.csv", index=False)
        figure4.to_csv(temp_dir / "figure4_aids_mut_source.csv", index=False)
        table.to_csv(temp_dir / "table2_aids_mut_full.csv", index=False)

        plt.rcParams.update(
            {
                "font.family": "serif",
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )
        cost_max = args.cost_ymax
        if cost_max is None:
            cost_max = 1.08 * float(figure3["ConditionalMedianCost"].max())

        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex="col")
        for column, dataset in enumerate(DATASETS):
            for method in METHODS:
                rows = figure3.loc[
                    (figure3["Dataset"] == dataset) & (figure3["Method"] == method)
                ].sort_values("K")
                style = METHOD_STYLE[method]
                markevery = marker_indices_for_k(rows["K"])
                axes[0, column].plot(
                    rows["K"],
                    100.0 * rows["Coverage"],
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    markevery=markevery,
                    linewidth=2.0,
                    markersize=5.5,
                )
                axes[1, column].plot(
                    rows["K"],
                    rows["ConditionalMedianCost"],
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    markevery=markevery,
                    linewidth=2.0,
                    markersize=5.5,
                )
            parent_count = EXPECTED_PARENT_COUNTS[dataset]
            theta = dataset_audit[dataset]["theta_star"]
            axes[0, column].set_title(f"{dataset} (n={parent_count}, theta*={theta:.5f})")
            for axis in (axes[0, column], axes[1, column]):
                axis.set_xlim(1, 20)
                axis.set_xticks([1, 5, 10, 15, 20])
                axis.axvline(10, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
                axis.grid(alpha=0.25)
            axes[0, column].set_ylim(0, args.coverage_ymax)
            axes[1, column].set_ylim(args.cost_ymin, cost_max)
            axes[1, column].set_xlabel("Prefix size K")
        axes[0, 0].set_ylabel("CCRCov (%)")
        axes[1, 0].set_ylabel("Conditional median cost")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=True)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        fig.savefig(temp_dir / "figure3_aids_mut_gcf_style.png", dpi=300, bbox_inches="tight")
        fig.savefig(temp_dir / "figure3_aids_mut_gcf_style.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
        for column, dataset in enumerate(DATASETS):
            all_thresholds: list[float] = []
            for method in METHODS:
                rows = figure4.loc[
                    (figure4["Dataset"] == dataset) & (figure4["Method"] == method)
                ].sort_values("Threshold")
                style = METHOD_STYLE[method]
                axes[column].plot(
                    rows["Threshold"],
                    100.0 * rows["Coverage"],
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    markevery=marker_indices_evenly(len(rows)),
                    linewidth=2.0,
                    markersize=5.5,
                )
                all_thresholds.extend(rows["Threshold"].tolist())
            theta = dataset_audit[dataset]["theta_star"]
            axes[column].axvline(theta, color="#555555", linestyle="--", linewidth=1.2, label="Frozen theta*")
            x_min, x_max = dataset_threshold_xlim(np.asarray(all_thresholds, dtype=float))
            axes[column].set_xlim(x_min, x_max)
            axes[column].set_ylim(0, args.coverage_ymax)
            axes[column].set_title(dataset)
            axes[column].set_xlabel("WNode threshold")
            axes[column].grid(alpha=0.25)
            axes[column].xaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[column].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axes[0].set_ylabel("CCRCov (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=5,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(temp_dir / "figure4_aids_mut_gcf_style.png", dpi=300, bbox_inches="tight")
        fig.savefig(temp_dir / "figure4_aids_mut_gcf_style.pdf", bbox_inches="tight")
        plt.close(fig)

        write_table_outputs(table, temp_dir)
        _write_audit(
            output_dir=temp_dir,
            datasets=datasets,
            dataset_audit=dataset_audit,
            sources=source_manifest,
        )
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "distance_label": DISTANCE_LABEL,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "selection_performed_in_plot": False,
            "distance_recomputed": False,
            "teacher_recomputed": False,
            "candidate_order_changed": False,
            "dataset_audit": dataset_audit,
            "roots": {
                dataset: {method: str(datasets[dataset][method]) for method in METHODS}
                for dataset in DATASETS
            },
            "sources": source_manifest,
            "outputs": {},
        }
        for path in sorted(temp_dir.iterdir()):
            if path.is_file() and path.name not in {"combined_manifest.json", "_RUN_COMPLETE.json"}:
                manifest["outputs"][path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        (temp_dir / "combined_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        complete = {
            "run_complete": True,
            "distance_label": DISTANCE_LABEL,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "method_count": len(METHODS),
            "dataset_count": len(DATASETS),
            "manifest_sha256": sha256(temp_dir / "combined_manifest.json"),
        }
        (temp_dir / "_RUN_COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"output_dir={output_dir}")
    print(f"methods={','.join(METHODS)}")
    print(f"distance_label={DISTANCE_LABEL}")
    print(f"distance_type={DISTANCE_TYPE}")
    print("[AIDS_MUT_GCF_STYLE_PLOT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
