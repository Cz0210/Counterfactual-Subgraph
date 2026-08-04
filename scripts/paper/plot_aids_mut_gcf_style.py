#!/usr/bin/env python3
"""Render the frozen AIDS and Mutagenicity WNode results in GCF style.

The script is deliberately presentation-only.  AIDS Figure 3 and Figure 4
are read from their audited CSVs, AIDS Table 2 is reduced from saved pair
details, and Mutagenicity is read from frozen plotting artifacts.  It never
computes embeddings, distances, teacher predictions, or candidate rankings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


DISTANCE_LABEL = "MolCLR-Node-Wasserstein"
DISTANCE_TYPE = "node_wasserstein"
CF_MODE = "strict_flip"

METHODS = ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
DATASET_ORDER = ("AIDS", "NCI1", "Mutagenicity", "Proteins")
ACTIVE_DATASET_COLUMNS = {"AIDS": 0, "Mutagenicity": 2}
EXPECTED_PARENT_COUNTS = {"AIDS": 1283, "Mutagenicity": 217}

AIDS_FIGURE3_RELATIVE_PATH = Path(
    "outputs/hpc/eval/paper/molclr_node_wasserstein_figure3_theta005_raw/"
    "wnode_fig3_theta005_figure3_wnode_coverage_cost_vs_k.csv"
)
AIDS_FIGURE4_RELATIVE_PATH = Path(
    "outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10/"
    "wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv"
)
AIDS_TABLE_ROOTS = {
    "Ours": Path(
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_ours_top20_final"
    ),
    "GlobalGCE": Path(
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_"
        "globalgce_frequency_top20_final"
    ),
    "CLEAR": Path(
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_"
        "clear_parent_frequency_top20_final"
    ),
    "GCFExplainer": Path(
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_"
        "gcfexplainer_top20_normalized_final"
    ),
}

AIDS_FIGURE3_THETA = 0.05
AIDS_TABLE2_THETA = 0.05
AIDS_TABLE2_K = 10
AIDS_FIGURE3_ROWS = 80
AIDS_FIGURE4_ROWS = 2404
AIDS_FIGURE4_K = 10
AIDS_FIGURE4_THRESHOLD_MIN = 0.0
AIDS_FIGURE4_THRESHOLD_MAX = 0.0535
AIDS_FIGURE4_POINTS_PER_METHOD = 601
MUT_FIGURE4_POINTS_PER_METHOD = AIDS_FIGURE4_POINTS_PER_METHOD
MUT_EXPECTED_PAIR_ROWS = EXPECTED_PARENT_COUNTS["Mutagenicity"] * 20

MUT_FROZEN_PAIR_SPECS = {
    "Ours": {
        "pairs": Path("pair_matrix.jsonl"),
        "order": Path("selected_sequence.jsonl"),
        "strict": "pair_strict_flip",
        "distance": "wnode_distance",
    },
    "GlobalGCE": {
        "pairs": Path("test_pair_details.csv"),
        "order": Path("selected_top20.csv"),
        "strict": "teacher_strict_flip",
        "distance": "distance",
    },
    "CLEAR": {
        "pairs": Path("test/k20_pair_details.csv"),
        "order": Path("selected_candidates.csv"),
        "strict": "teacher_strict_flip",
        "distance": "distance",
    },
    "GCFExplainer": {
        "pairs": Path("test_pair_details.csv"),
        "order": Path("selected_sequence.jsonl"),
        "strict": "teacher_strict_flip",
        "distance": "distance",
    },
}

AIDS_TABLE2_EXPECTED = {
    "Ours": (0.7490257209664848, 0.0148861954639967),
    "GlobalGCE": (0.4489477786438036, 0.0493276937150003),
    "CLEAR": (0.2166796570537802, 0.0553405022830596),
    "GCFExplainer": (0.642244738893219, 0.041917737627761),
}

METHOD_STYLES = {
    "Ours": {"color": "black", "marker": "s", "label": "Ours"},
    "GlobalGCE": {"color": "#E53935", "marker": "x", "label": "GlobalGCE"},
    "CLEAR": {"color": "#2E7D32", "marker": "*", "label": "CLEAR"},
    "GCFExplainer": {
        "color": "#B02BC7",
        "marker": "^",
        "label": "GCFExplainer",
    },
}
FIGURE3_MARKER_K = (1, 3, 5, 10, 15, 20)
FIGURE3_MARKER_INDICES = (0, 2, 4, 9, 14, 19)
FIGURE4_AIDS_MARKEVERY = 100
FIGURE3_FIGSIZE = (16.0, 6.3)
FIGURE4_FIGSIZE = (16.0, 3.8)
TABLE2_FIGSIZE = (15.8, 3.3)

FIGURE3_OUTPUT_STEM = "figure3_gcf_style_aids_mut"
FIGURE4_OUTPUT_STEM = "figure4_gcf_style_aids_mut"
TABLE2_OUTPUT_STEM = "table2_gcf_style_aids_mut"

FORBIDDEN_PROVENANCE_TOKENS = (
    "ccrcov_molclr_node_fgw_",
    "molclr-node-fgw",
    "node_fgw",
    "fgw_lambda",
    "lam05",
    "gt_fullgraph",
    "opposite_fullgraph",
    "opposite-label",
)

_METHOD_ALIASES = {
    "ours": "Ours",
    "ours_selected_subgraphs": "Ours",
    "globalgce": "GlobalGCE",
    "globalgce_frequency_top20": "GlobalGCE",
    "clear": "CLEAR",
    "clear_parentfrequency_top20": "CLEAR",
    "clear_parent_frequency_top20": "CLEAR",
    "gcfexplainer": "GCFExplainer",
    "gcfexplainer_top20": "GCFExplainer",
}


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonical_method(value: object) -> str:
    key = normalize(value)
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unknown method label: {value!r}")
    return _METHOD_ALIASES[key]


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    lookup = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
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
        raise ValueError(f"Non-finite values in {column}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"})
    if unknown.any():
        raise ValueError(f"Unrecognized boolean values: {sorted(normalized[unknown].unique())}")
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _assert_exact_project_path(path: Path, project_root: Path, expected: Path) -> None:
    expected_path = (project_root / expected).resolve()
    if path != expected_path:
        raise ValueError(f"Expected frozen source {expected_path}, received {path}")


def _reject_forbidden_provenance(*values: object) -> None:
    combined = "\n".join(str(value) for value in values).lower()
    hits = sorted(token for token in FORBIDDEN_PROVENANCE_TOKENS if token in combined)
    if hits:
        raise ValueError(f"Forbidden non-WNode provenance: {hits}")


def _validate_frame_provenance(
    frame: pd.DataFrame,
    *,
    expected_parents: int,
    context: str,
    require_distance_label: bool,
) -> dict[str, list[object]]:
    evidence: dict[str, list[object]] = {
        "distance_labels": [],
        "distance_types": [],
        "cf_modes": [],
        "parent_counts": [],
        "candidate_set_preselected": [],
        "selection_performed_in_eval": [],
    }

    def values(candidates: Sequence[str]) -> list[object]:
        column = find_column(frame, candidates, required=False)
        return [] if column is None else frame[column].dropna().unique().tolist()

    evidence["distance_labels"] = values(("distance_label", "distance_line"))
    evidence["distance_types"] = values(("distance_type",))
    evidence["cf_modes"] = values(("cf_mode",))
    evidence["parent_counts"] = values(
        ("num_parents", "num_test_parents", "test_parent_count")
    )
    evidence["candidate_set_preselected"] = values(("candidate_set_preselected",))
    evidence["selection_performed_in_eval"] = values(("selection_performed_in_eval",))

    _reject_forbidden_provenance(context, evidence)
    labels = {normalize(value) for value in evidence["distance_labels"]}
    if require_distance_label and labels != {normalize(DISTANCE_LABEL)}:
        raise ValueError(f"{context}: unexpected distance labels {sorted(labels)}")
    if labels and labels != {normalize(DISTANCE_LABEL)}:
        raise ValueError(f"{context}: unexpected distance labels {sorted(labels)}")
    types = {normalize(value) for value in evidence["distance_types"]}
    if types and types != {DISTANCE_TYPE}:
        raise ValueError(f"{context}: unexpected distance types {sorted(types)}")
    modes = {normalize(value) for value in evidence["cf_modes"]}
    if modes and modes != {CF_MODE}:
        raise ValueError(f"{context}: unexpected cf modes {sorted(modes)}")
    if evidence["parent_counts"]:
        counts = {int(float(value)) for value in evidence["parent_counts"]}
        if counts != {expected_parents}:
            raise ValueError(f"{context}: unexpected parent counts {sorted(counts)}")
    if evidence["candidate_set_preselected"]:
        states = set(normalize_bool(pd.Series(evidence["candidate_set_preselected"])))
        if states != {True}:
            raise ValueError(f"{context}: candidates are not proven preselected")
    if evidence["selection_performed_in_eval"]:
        states = set(normalize_bool(pd.Series(evidence["selection_performed_in_eval"])))
        if states != {False}:
            raise ValueError(f"{context}: selection was performed in evaluation")
    return evidence


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


def _root_metadata_evidence(root: Path) -> dict[str, Any]:
    paths: set[Path] = set()
    for pattern in (
        "*manifest*.json",
        "*audit*.json",
        "summary.json",
        "run_config.json",
        "_RUN_COMPLETE.json",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    evidence: dict[str, Any] = {
        "distance_labels": [],
        "distance_types": [],
        "cf_modes": [],
        "parent_counts": [],
        "candidate_set_preselected": [],
        "selection_performed_in_eval": [],
        "strict_flip_provenance": [],
        "files": [],
    }

    def record(key: str, value: object) -> None:
        if value is None or value == "" or value in evidence[key]:
            return
        evidence[key].append(value)

    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid provenance JSON {path}: {error}") from error
        _reject_forbidden_provenance(path, json.dumps(payload, sort_keys=True))
        evidence["files"].append({"path": str(path), "sha256": sha256(path)})
        for key_path, value in _json_scalars(payload):
            components = [
                normalize(component.split("[", 1)[0]) for component in key_path.split(".")
            ]
            if any(
                component in {"checks", "semantic_checks", "required_fields", "forbidden_fields"}
                for component in components[:-1]
            ):
                continue
            key = components[-1]
            if key in {"distance_label", "distance_line"}:
                record("distance_labels", str(value))
            elif key == "distance_type":
                record("distance_types", str(value))
            elif key == "cf_mode":
                record("cf_modes", str(value))
            elif key in {
                "num_parents",
                "num_test_parents",
                "test_parent_count",
                "reference_parent_count",
            }:
                try:
                    record("parent_counts", int(value))
                except (TypeError, ValueError):
                    pass
            elif key in {
                "candidate_set_preselected",
                "candidate_order_frozen",
                "candidate_order_exact_match",
            }:
                record("candidate_set_preselected", value)
            elif key in {
                "selection_performed_in_eval",
                "candidate_selection_performed",
                "test_candidate_selection",
            }:
                record("selection_performed_in_eval", value)
            elif "strict_flip" in key:
                record("strict_flip_provenance", f"{path.name}:{key_path}")
    return evidence


def _ordered_parts(parts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([parts[method] for method in METHODS if method in parts], ignore_index=True)


def load_aids_figure3(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_csv(path)
    if len(frame) != AIDS_FIGURE3_ROWS:
        raise ValueError(f"AIDS Figure 3 row_count={len(frame)}, expected={AIDS_FIGURE3_ROWS}")
    method_col = find_column(frame, ("method", "Method"))
    k_col = find_column(frame, ("k", "K"))
    theta_col = find_column(frame, ("theta", "threshold"))
    coverage_col = find_column(frame, ("coverage", "Coverage"))
    cost_col = find_column(frame, ("conditional_median_cost",))
    _validate_frame_provenance(
        frame,
        expected_parents=EXPECTED_PARENT_COUNTS["AIDS"],
        context="AIDS Figure 3",
        require_distance_label=True,
    )

    work = frame.copy()
    work["_method"] = work[method_col].map(canonical_method)
    parts: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        rows = work.loc[work["_method"] == method].copy()
        if len(rows) != 20:
            raise ValueError(f"AIDS Figure 3 {method} rows={len(rows)}, expected=20")
        k = numeric(rows, k_col).astype(int).to_numpy()
        if not np.array_equal(k, np.arange(1, 21)):
            raise ValueError(f"AIDS Figure 3 {method} K must be exactly 1,...,20")
        theta = numeric(rows, theta_col).to_numpy(dtype=float)
        if not np.allclose(theta, AIDS_FIGURE3_THETA, rtol=0.0, atol=1e-12):
            raise ValueError(f"AIDS Figure 3 {method} theta must be {AIDS_FIGURE3_THETA}")
        coverage = numeric(rows, coverage_col).to_numpy(dtype=float)
        cost = numeric(rows, cost_col).to_numpy(dtype=float)
        if not np.all((coverage >= 0.0) & (coverage <= 1.0)):
            raise ValueError(f"AIDS Figure 3 {method} coverage outside [0,1]")
        if np.any(cost < 0.0):
            raise ValueError(f"AIDS Figure 3 {method} cost is negative")
        parts[method] = pd.DataFrame(
            {
                "Dataset": "AIDS",
                "Method": method,
                "K": k,
                "Theta": theta,
                "Coverage": coverage,
                "Cost": cost,
            }
        )
    output = _ordered_parts(parts)
    if output.duplicated(("Method", "K")).any():
        raise ValueError("AIDS Figure 3 contains duplicate method/K rows")
    return output, {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(frame),
        "theta": AIDS_FIGURE3_THETA,
        "cost_source_column": cost_col,
        "distance_recomputed": False,
        "candidate_order_changed": False,
    }


def load_aids_figure4(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_csv(path)
    if len(frame) != AIDS_FIGURE4_ROWS:
        raise ValueError(f"AIDS Figure 4 row_count={len(frame)}, expected={AIDS_FIGURE4_ROWS}")
    method_col = find_column(frame, ("method", "Method"))
    k_col = find_column(frame, ("k", "K"))
    threshold_col = find_column(frame, ("threshold", "theta"))
    coverage_col = find_column(frame, ("coverage", "mean", "close_cf_coverage"))
    _validate_frame_provenance(
        frame,
        expected_parents=EXPECTED_PARENT_COUNTS["AIDS"],
        context="AIDS Figure 4",
        require_distance_label=True,
    )

    work = frame.copy()
    work["_method"] = work[method_col].map(canonical_method)
    parts: dict[str, pd.DataFrame] = {}
    grids: list[np.ndarray] = []
    for method in METHODS:
        rows = work.loc[work["_method"] == method].copy()
        if len(rows) != AIDS_FIGURE4_POINTS_PER_METHOD:
            raise ValueError(
                f"AIDS Figure 4 {method} points={len(rows)}, "
                f"expected={AIDS_FIGURE4_POINTS_PER_METHOD}"
            )
        k = numeric(rows, k_col).astype(int).to_numpy()
        if set(k.tolist()) != {AIDS_FIGURE4_K}:
            raise ValueError(f"AIDS Figure 4 {method} K must be {AIDS_FIGURE4_K}")
        thresholds = numeric(rows, threshold_col).to_numpy(dtype=float)
        if not np.all(np.diff(thresholds) > 0.0):
            raise ValueError(f"AIDS Figure 4 {method} threshold order is not strictly increasing")
        if not np.isclose(thresholds[0], AIDS_FIGURE4_THRESHOLD_MIN, rtol=0.0, atol=1e-15):
            raise ValueError(f"AIDS Figure 4 {method} threshold_min is not 0.0")
        if not np.isclose(thresholds[-1], AIDS_FIGURE4_THRESHOLD_MAX, rtol=0.0, atol=1e-15):
            raise ValueError(f"AIDS Figure 4 {method} threshold_max is not 0.0535")
        coverage = numeric(rows, coverage_col).to_numpy(dtype=float)
        if not np.all((coverage >= 0.0) & (coverage <= 1.0)):
            raise ValueError(f"AIDS Figure 4 {method} coverage outside [0,1]")
        grids.append(thresholds)
        parts[method] = pd.DataFrame(
            {
                "Dataset": "AIDS",
                "Method": method,
                "K": k,
                "Threshold": thresholds,
                "Coverage": coverage,
            }
        )
    first = grids[0]
    if any(not np.allclose(grid, first, rtol=0.0, atol=1e-15) for grid in grids[1:]):
        raise ValueError("AIDS Figure 4 threshold grids differ across methods")
    output = _ordered_parts(parts)
    if output.duplicated(("Method", "Threshold")).any():
        raise ValueError("AIDS Figure 4 contains duplicate method/threshold rows")
    return output, {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(frame),
        "k": AIDS_FIGURE4_K,
        "points_per_method": AIDS_FIGURE4_POINTS_PER_METHOD,
        "threshold_min": float(first[0]),
        "threshold_max": float(first[-1]),
        "interpolation_performed": False,
        "distance_recomputed": False,
        "candidate_order_changed": False,
    }


def candidate_rank(details: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ("rank", "selection_rank", "candidate_rank", "native_rank"):
        if column not in details.columns:
            continue
        rank = pd.to_numeric(details[column], errors="coerce")
        finite = rank.dropna()
        if finite.empty:
            continue
        if finite.min() == 0 and finite.max() <= 19:
            rank = rank + 1
        return rank, column
    candidate_col = find_column(details, ("candidate_id",))
    order = list(dict.fromkeys(details[candidate_col].astype(str)))
    mapping = {candidate_id: index for index, candidate_id in enumerate(order, start=1)}
    return details[candidate_col].astype(str).map(mapping), "stable_first_occurrence"


def load_parent_best_distances(run_dir: Path, *, k: int) -> tuple[pd.Series, dict[str, Any]]:
    _reject_forbidden_provenance(run_dir)
    path = run_dir / "details" / "pair_details.csv"
    details = read_csv(path)
    parent_col = find_column(details, ("parent_id", "molecule_id"))
    candidate_col = find_column(details, ("candidate_id",))
    distance_col = find_column(details, ("distance", "wnode_distance"))
    strict_col = find_column(details, ("cf_flip", "teacher_strict_flip", "strict_flip"))
    rank, rank_source = candidate_rank(details)

    parent_count = details[parent_col].astype(str).nunique()
    candidate_count = details[candidate_col].astype(str).nunique()
    if parent_count != EXPECTED_PARENT_COUNTS["AIDS"]:
        raise ValueError(f"{path}: parent_count={parent_count}, expected=1283")
    if candidate_count != 20:
        raise ValueError(f"{path}: candidate_count={candidate_count}, expected=20")
    observed_ranks = sorted(set(pd.to_numeric(rank, errors="coerce").dropna().astype(int)))
    if observed_ranks != list(range(1, 21)):
        raise ValueError(f"{path}: frozen candidate ranks are not 1,...,20")

    work = details.assign(
        _rank=rank,
        _distance=pd.to_numeric(details[distance_col], errors="coerce"),
        _strict_flip=normalize_bool(details[strict_col]),
    )
    valid = work.loc[
        work["_rank"].notna()
        & work["_rank"].between(1, k)
        & work["_distance"].notna()
        & np.isfinite(work["_distance"])
        & work["_strict_flip"]
    ].copy()
    if valid.empty:
        raise ValueError(f"{path}: no valid strict-flip rows at K={k}")
    pair_best = (
        valid.groupby([parent_col, candidate_col], as_index=False, sort=False)["_distance"]
        .min()
    )
    parent_best = pair_best.groupby(parent_col, sort=False)["_distance"].min().astype(float)
    if not np.isfinite(parent_best.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite parent-best distances")
    return parent_best, {
        "pair_details_path": str(path),
        "pair_details_sha256": sha256(path),
        "rank_source": rank_source,
        "num_raw_rows": len(details),
        "num_parent_candidate_pairs": len(pair_best),
        "num_applicable_parents": len(parent_best),
        "candidate_count": candidate_count,
        "candidate_order_changed": False,
        "distance_recomputed": False,
        "teacher_recomputed": False,
        "strict_flip_column": strict_col,
    }


def load_aids_table2(roots: Mapping[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for method in METHODS:
        root = roots[method]
        if not root.is_dir():
            raise FileNotFoundError(root)
        parent_best, metadata = load_parent_best_distances(root, k=AIDS_TABLE2_K)
        coverage = float(
            np.count_nonzero(parent_best.to_numpy(dtype=float) <= AIDS_TABLE2_THETA)
            / EXPECTED_PARENT_COUNTS["AIDS"]
        )
        cost = float(np.median(parent_best.to_numpy(dtype=float)))
        expected_coverage, expected_cost = AIDS_TABLE2_EXPECTED[method]
        if not np.isclose(coverage, expected_coverage, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"AIDS Table 2 {method} coverage={coverage:.17g}, "
                f"expected={expected_coverage:.17g}"
            )
        if not np.isclose(cost, expected_cost, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"AIDS Table 2 {method} cost={cost:.17g}, expected={expected_cost:.17g}"
            )
        row = {
            "Dataset": "AIDS",
            "Method": method,
            "K": AIDS_TABLE2_K,
            "Theta": AIDS_TABLE2_THETA,
            "Coverage": coverage,
            "Cost": cost,
            "NumParents": EXPECTED_PARENT_COUNTS["AIDS"],
        }
        rows.append(row)
        audit[method] = {**metadata, **row, "root": str(root)}
    return pd.DataFrame(rows), audit


def table2_path(root: Path, method: str) -> Path:
    slug = {
        "Ours": "ours",
        "GlobalGCE": "globalgce",
        "CLEAR": "clear",
        "GCFExplainer": "gcfexplainer",
    }[method]
    preferred = root / f"table2_{slug}_k10.csv"
    if preferred.is_file():
        return preferred
    matches = sorted(root.glob("table2_*_k10.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{method}: expected one K=10 Table 2 CSV under {root}, found={matches}"
        )
    return matches[0]


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return pd.DataFrame(rows)


def _read_frozen_frame(path: Path) -> pd.DataFrame:
    return _read_jsonl(path) if path.suffix == ".jsonl" else read_csv(path)


def _sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_mut_candidate_order(path: Path, method: str) -> tuple[list[str], dict[str, Any]]:
    frame = _read_frozen_frame(path)
    candidate_col = find_column(frame, ("candidate_id",))
    rank_col = find_column(frame, ("rank", "selection_rank"))
    ranks = numeric(frame, rank_col).astype(int).tolist()
    candidate_ids = frame[candidate_col].astype(str).tolist()
    if ranks != list(range(1, 21)):
        raise ValueError(f"Mutagenicity/{method}: frozen ranks must be ordered 1,...,20")
    if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20:
        raise ValueError(f"Mutagenicity/{method}: frozen candidate IDs must be 20 unique values")
    preselected_col = find_column(frame, ("candidate_set_preselected",), required=False)
    if preselected_col is not None and set(normalize_bool(frame[preselected_col])) != {True}:
        raise ValueError(f"Mutagenicity/{method}: candidate set is not frozen")
    selected_in_eval_col = find_column(frame, ("selection_performed_in_eval",), required=False)
    if selected_in_eval_col is not None and set(normalize_bool(frame[selected_in_eval_col])) != {False}:
        raise ValueError(f"Mutagenicity/{method}: selection was performed in evaluation")
    return candidate_ids, {
        "path": str(path),
        "sha256": sha256(path),
        "candidate_count": len(candidate_ids),
        "candidate_order_sha256": _sequence_sha256(candidate_ids),
        "candidate_order_changed": False,
    }


def _load_mut_distance_matrix(
    root: Path,
    method: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    _reject_forbidden_provenance(root)
    spec = MUT_FROZEN_PAIR_SPECS[method]
    pair_path = root / spec["pairs"]
    order_path = root / spec["order"]
    candidate_ids, order_evidence = _load_mut_candidate_order(order_path, method)
    pairs = _read_frozen_frame(pair_path)
    if len(pairs) != MUT_EXPECTED_PAIR_ROWS:
        raise ValueError(
            f"Mutagenicity/{method}: pair_count={len(pairs)}, expected={MUT_EXPECTED_PAIR_ROWS}"
        )
    parent_col = find_column(pairs, ("parent_id", "molecule_id"))
    candidate_col = find_column(pairs, ("candidate_id",))
    if spec["strict"] not in pairs.columns or spec["distance"] not in pairs.columns:
        raise ValueError(f"Mutagenicity/{method}: frozen pair schema does not match contract")

    parent_ids = list(dict.fromkeys(pairs[parent_col].astype(str)))
    observed_candidates = set(pairs[candidate_col].astype(str))
    if len(parent_ids) != EXPECTED_PARENT_COUNTS["Mutagenicity"]:
        raise ValueError(f"Mutagenicity/{method}: expected 217 unique parents")
    if observed_candidates != set(candidate_ids):
        raise ValueError(f"Mutagenicity/{method}: pair candidates differ from frozen order")
    pair_keys = pairs[parent_col].astype(str) + "\0" + pairs[candidate_col].astype(str)
    if pair_keys.duplicated().any() or pair_keys.nunique() != MUT_EXPECTED_PAIR_ROWS:
        raise ValueError(f"Mutagenicity/{method}: pair matrix is not a complete Cartesian product")
    counts = pairs.groupby(parent_col, sort=False)[candidate_col].nunique().to_numpy(dtype=int)
    if not np.array_equal(counts, np.full(len(parent_ids), 20, dtype=int)):
        raise ValueError(f"Mutagenicity/{method}: one or more parents do not have 20 candidates")

    labels: set[str] = set()
    types: set[str] = set()
    distance_line_col = find_column(pairs, ("distance_line",), required=False)
    distance_type_col = find_column(pairs, ("distance_type",), required=False)
    if distance_line_col is not None:
        labels.update(normalize(value) for value in pairs[distance_line_col].dropna().unique())
    if distance_type_col is not None:
        types.update(normalize(value) for value in pairs[distance_type_col].dropna().unique())
    metadata = _root_metadata_evidence(root)
    labels.update(normalize(value) for value in metadata["distance_labels"])
    types.update(normalize(value) for value in metadata["distance_types"])
    if labels != {normalize(DISTANCE_LABEL)} or types != {DISTANCE_TYPE}:
        raise ValueError(
            f"Mutagenicity/{method}: frozen pair WNode provenance mismatch: "
            f"labels={sorted(labels)}, types={sorted(types)}"
        )

    strict = normalize_bool(pairs[spec["strict"]]).to_numpy(dtype=bool)
    distances = pd.to_numeric(pairs[spec["distance"]], errors="coerce").to_numpy(dtype=float)
    if np.any(strict & (~np.isfinite(distances) | (distances < 0.0))):
        raise ValueError(f"Mutagenicity/{method}: strict-flip pair has invalid saved distance")
    effective = np.where(strict, distances, np.inf)
    rank_map = {candidate_id: rank for rank, candidate_id in enumerate(candidate_ids, start=1)}
    work = pd.DataFrame(
        {
            "parent_id": pairs[parent_col].astype(str),
            "rank": pairs[candidate_col].astype(str).map(rank_map),
            "distance": effective,
        }
    )
    matrix_frame = work.pivot(index="parent_id", columns="rank", values="distance")
    matrix_frame = matrix_frame.reindex(index=parent_ids, columns=list(range(1, 21)))
    matrix = matrix_frame.to_numpy(dtype=float)
    if matrix.shape != (EXPECTED_PARENT_COUNTS["Mutagenicity"], 20):
        raise ValueError(f"Mutagenicity/{method}: unexpected frozen matrix shape {matrix.shape}")
    evidence = {
        "root": str(root),
        "pair_path": str(pair_path),
        "pair_sha256": sha256(pair_path),
        "pair_count": len(pairs),
        "complete_cartesian": True,
        "parent_count": len(parent_ids),
        "parent_ids_sha256": _sequence_sha256(parent_ids),
        "strict_flip_field": spec["strict"],
        "distance_field": spec["distance"],
        "distance_recomputed": False,
        "teacher_recomputed": False,
        **order_evidence,
    }
    return matrix, parent_ids, evidence


def load_mut_method(
    root: Path,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    matrix, _parent_ids, source = _load_mut_distance_matrix(root, method)
    best = np.full(matrix.shape[0], np.inf, dtype=float)
    figure3_rows: list[dict[str, Any]] = []
    best_at_k10: np.ndarray | None = None
    for candidate_index in range(matrix.shape[1]):
        best = np.minimum(best, matrix[:, candidate_index])
        finite = np.isfinite(best)
        if not np.any(finite):
            raise ValueError(f"Mutagenicity/{method}: no strict-flip parent at K={candidate_index + 1}")
        figure3_rows.append(
            {
                "Dataset": "Mutagenicity",
                "Method": method,
                "K": candidate_index + 1,
                "Theta": AIDS_FIGURE3_THETA,
                "Coverage": float(np.mean(best <= AIDS_FIGURE3_THETA)),
                "Cost": float(np.median(best[finite])),
            }
        )
        if candidate_index + 1 == AIDS_TABLE2_K:
            best_at_k10 = best.copy()
    if best_at_k10 is None:
        raise AssertionError("K=10 prefix was not produced")

    thresholds = np.linspace(
        AIDS_FIGURE4_THRESHOLD_MIN,
        AIDS_FIGURE4_THRESHOLD_MAX,
        AIDS_FIGURE4_POINTS_PER_METHOD,
    )
    figure4 = pd.DataFrame(
        {
            "Dataset": "Mutagenicity",
            "Method": method,
            "K": AIDS_FIGURE4_K,
            "Threshold": thresholds,
            "Coverage": [float(np.mean(best_at_k10 <= threshold)) for threshold in thresholds],
        }
    )
    finite_k10 = np.isfinite(best_at_k10)
    if not np.any(finite_k10):
        raise ValueError(f"Mutagenicity/{method}: no strict-flip parent at K=10")
    table = {
        "Dataset": "Mutagenicity",
        "Method": method,
        "K": AIDS_TABLE2_K,
        "Theta": AIDS_TABLE2_THETA,
        "Coverage": float(np.mean(best_at_k10 <= AIDS_TABLE2_THETA)),
        "Cost": float(np.median(best_at_k10[finite_k10])),
        "NumParents": EXPECTED_PARENT_COUNTS["Mutagenicity"],
    }
    source.update(
        {
            "threshold_mode": "match-aids",
            "aggregation_only": True,
            "figure3_theta": AIDS_FIGURE3_THETA,
            "table2_theta": AIDS_TABLE2_THETA,
            "table2_k": AIDS_TABLE2_K,
            "figure4_k": AIDS_FIGURE4_K,
            "figure4_threshold_min": AIDS_FIGURE4_THRESHOLD_MIN,
            "figure4_threshold_max": AIDS_FIGURE4_THRESHOLD_MAX,
            "figure4_points_per_method": AIDS_FIGURE4_POINTS_PER_METHOD,
        }
    )
    return pd.DataFrame(figure3_rows), figure4, table, source


def _filter_method(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    method_col = find_column(frame, ("method", "Method"), required=False)
    if method_col is None:
        return frame.copy()
    canonical = frame[method_col].map(canonical_method)
    return frame.loc[canonical == method].copy()


def _deduplicate_native_thresholds(
    frame: pd.DataFrame,
    *,
    threshold_col: str,
    coverage_col: str,
    context: str,
) -> tuple[pd.DataFrame, int]:
    keep: list[int] = []
    removed = 0
    for _, group in frame.groupby(threshold_col, sort=False):
        coverage = numeric(group, coverage_col).to_numpy(dtype=float)
        if not np.allclose(coverage, coverage[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"{context}: conflicting duplicate threshold rows")
        keep.append(int(group.index[0]))
        removed += len(group) - 1
    return frame.loc[keep].copy(), removed


def load_mut_native_method(
    root: Path,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    _matrix, _parent_ids, pair_evidence = _load_mut_distance_matrix(root, method)
    figure3_path = root / "figure3_coverage_vs_k.csv"
    figure4_path = root / "figure4_coverage_vs_threshold.csv"
    table_path = table2_path(root, method)
    raw3 = _filter_method(read_csv(figure3_path), method)
    raw4 = _filter_method(read_csv(figure4_path), method)
    raw2 = _filter_method(read_csv(table_path), method)

    table_k_col = find_column(raw2, ("k", "K"))
    table_rows = raw2.loc[numeric(raw2, table_k_col).astype(int) == 10].copy()
    if len(table_rows) != 1:
        raise ValueError(f"Mutagenicity/{method}: native Table 2 must have one K=10 row")
    row = table_rows.iloc[0]
    theta_col = find_column(table_rows, ("theta", "theta_star", "threshold"))
    coverage2_col = find_column(table_rows, ("coverage", "ccrcov_theta_star", "ccrcov"))
    cost2_col = find_column(table_rows, ("conditional_median_cost",))
    parent_col = find_column(
        table_rows, ("num_parents", "num_test_parents", "test_parent_count")
    )
    theta = float(row[theta_col])
    table = {
        "Dataset": "Mutagenicity",
        "Method": method,
        "K": 10,
        "Theta": theta,
        "Coverage": float(row[coverage2_col]),
        "Cost": float(row[cost2_col]),
        "NumParents": int(row[parent_col]),
    }
    if table["NumParents"] != EXPECTED_PARENT_COUNTS["Mutagenicity"]:
        raise ValueError(f"Mutagenicity/{method}: native Table 2 parent count mismatch")

    k3_col = find_column(raw3, ("k", "K"))
    coverage3_col = find_column(
        raw3,
        ("ccrcov_theta_star", "coverage_at_theta_star", "coverage", "close_cf_coverage"),
    )
    cost3_col = find_column(raw3, ("conditional_median_cost",))
    theta3_col = find_column(raw3, ("theta", "theta_star", "threshold"), required=False)
    k_values = numeric(raw3, k3_col).astype(int).to_numpy()
    if not np.array_equal(k_values, np.arange(1, 21)):
        raise ValueError(f"Mutagenicity/{method}: native Figure 3 K must be 1,...,20")
    theta_values = (
        np.full(20, theta, dtype=float)
        if theta3_col is None
        else numeric(raw3, theta3_col).to_numpy(dtype=float)
    )
    if not np.allclose(theta_values, theta, rtol=0.0, atol=1e-12):
        raise ValueError(f"Mutagenicity/{method}: native Figure 3 theta mismatch")
    figure3 = pd.DataFrame(
        {
            "Dataset": "Mutagenicity",
            "Method": method,
            "K": k_values,
            "Theta": theta_values,
            "Coverage": numeric(raw3, coverage3_col).to_numpy(dtype=float),
            "Cost": numeric(raw3, cost3_col).to_numpy(dtype=float),
        }
    )

    k4_col = find_column(raw4, ("k", "K"))
    raw4 = raw4.loc[numeric(raw4, k4_col).astype(int) == 10].copy()
    threshold_col = find_column(raw4, ("threshold", "theta"))
    coverage4_col = find_column(raw4, ("coverage", "ccrcov", "close_cf_coverage", "mean"))
    raw4, removed = _deduplicate_native_thresholds(
        raw4,
        threshold_col=threshold_col,
        coverage_col=coverage4_col,
        context=f"Mutagenicity/{method} native Figure 4",
    )
    thresholds = numeric(raw4, threshold_col).to_numpy(dtype=float)
    coverage4 = numeric(raw4, coverage4_col).to_numpy(dtype=float)
    order = np.argsort(thresholds, kind="stable")
    thresholds = thresholds[order]
    coverage4 = coverage4[order]
    if len(thresholds) != 7 or len(np.unique(thresholds)) != 7:
        raise ValueError(f"Mutagenicity/{method}: native Figure 4 must have 7 thresholds")
    figure4 = pd.DataFrame(
        {
            "Dataset": "Mutagenicity",
            "Method": method,
            "K": 10,
            "Threshold": thresholds,
            "Coverage": coverage4,
        }
    )
    source = {
        **pair_evidence,
        "threshold_mode": "native",
        "figure3": {"path": str(figure3_path), "sha256": sha256(figure3_path)},
        "figure4": {
            "path": str(figure4_path),
            "sha256": sha256(figure4_path),
            "duplicate_rows_removed": removed,
        },
        "table2": {"path": str(table_path), "sha256": sha256(table_path)},
    }
    return figure3, figure4, table, source


def validate_mut_alignment(
    figure4: pd.DataFrame,
    table: pd.DataFrame,
) -> dict[str, Any]:
    methods = tuple(table["Method"])
    if methods != METHODS:
        raise ValueError(f"Mutagenicity methods must be exactly {METHODS}; observed={methods}")
    theta_values = table["Theta"].to_numpy(dtype=float)
    if not np.allclose(theta_values, AIDS_TABLE2_THETA, rtol=0.0, atol=1e-12):
        raise ValueError("Mutagenicity Table 2 must use the AIDS theta=0.05 contract")
    grids = [
        figure4.loc[figure4["Method"] == method, "Threshold"].to_numpy(dtype=float)
        for method in METHODS
    ]
    first = grids[0]
    if len(first) != AIDS_FIGURE4_POINTS_PER_METHOD:
        raise ValueError("Mutagenicity Figure 4 must contain 601 thresholds per method")
    if any(not np.array_equal(grid, first) for grid in grids[1:]):
        raise ValueError("Mutagenicity Figure 4 threshold grids differ across methods")
    if not np.isclose(first[0], AIDS_FIGURE4_THRESHOLD_MIN, rtol=0.0, atol=1e-15):
        raise ValueError("Mutagenicity Figure 4 threshold_min must be 0.0")
    if not np.isclose(first[-1], AIDS_FIGURE4_THRESHOLD_MAX, rtol=0.0, atol=1e-15):
        raise ValueError("Mutagenicity Figure 4 threshold_max must be 0.0535")
    return {
        "parent_count": EXPECTED_PARENT_COUNTS["Mutagenicity"],
        "theta": AIDS_TABLE2_THETA,
        "figure4_threshold_count": len(first),
        "figure4_threshold_min": float(first[0]),
        "figure4_threshold_max": float(first[-1]),
        "methods_present": list(METHODS),
        "uses_same_parameters_and_range_as_aids": True,
    }


def validate_mut_native_alignment(
    figure4: pd.DataFrame,
    table: pd.DataFrame,
) -> dict[str, Any]:
    methods = tuple(table["Method"])
    if methods != METHODS:
        raise ValueError(f"Mutagenicity methods must be exactly {METHODS}; observed={methods}")
    theta_values = table["Theta"].to_numpy(dtype=float)
    if not np.allclose(theta_values, theta_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Mutagenicity native Table 2 methods do not share one frozen theta")
    grids = [
        figure4.loc[figure4["Method"] == method, "Threshold"].to_numpy(dtype=float)
        for method in METHODS
    ]
    first = grids[0]
    if len(first) != 7 or any(
        not np.allclose(grid, first, rtol=0.0, atol=1e-12) for grid in grids[1:]
    ):
        raise ValueError("Mutagenicity native Figure 4 grids must share 7 frozen thresholds")
    return {
        "parent_count": EXPECTED_PARENT_COUNTS["Mutagenicity"],
        "theta": float(theta_values[0]),
        "figure4_threshold_count": len(first),
        "figure4_threshold_min": float(first[0]),
        "figure4_threshold_max": float(first[-1]),
        "methods_present": list(METHODS),
        "uses_same_parameters_and_range_as_aids": False,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
        }
    )


def _save_png_pdf(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _placeholder(axis: plt.Axes, title: str) -> None:
    axis.clear()
    axis.axis("off")
    axis.set_title(title, fontweight="bold", pad=8)


def _method_rows(frame: pd.DataFrame, dataset: str, method: str, sort: str) -> pd.DataFrame:
    return frame.loc[
        (frame["Dataset"] == dataset) & (frame["Method"] == method)
    ].sort_values(sort, kind="stable")


def render_figure3(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    mut_matches_aids: bool,
    coverage_ymax: float = 80.0,
) -> None:
    if not np.isfinite(coverage_ymax) or coverage_ymax <= 0:
        raise ValueError("coverage_ymax must be a finite positive percentage.")
    coverage_ticks = [0, 20, 40, 60, 80]
    if coverage_ymax > coverage_ticks[-1]:
        coverage_ticks.append(float(coverage_ymax))
    configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGURE3_FIGSIZE,
        gridspec_kw={"hspace": 0.18, "wspace": 0.22},
    )
    for dataset, column in ACTIVE_DATASET_COLUMNS.items():
        top = axes[0, column]
        bottom = axes[1, column]
        for method in METHODS:
            rows = _method_rows(frame, dataset, method, "K")
            if rows.empty:
                continue
            style = METHOD_STYLES[method]
            top.plot(
                rows["K"],
                100.0 * rows["Coverage"],
                color=style["color"],
                marker=style["marker"],
                markevery=list(FIGURE3_MARKER_INDICES),
                markersize=5.5,
                markeredgewidth=0.9,
                label=style["label"],
            )
            bottom.plot(
                rows["K"],
                rows["Cost"],
                color=style["color"],
                marker=style["marker"],
                markevery=list(FIGURE3_MARKER_INDICES),
                markersize=5.5,
                markeredgewidth=0.9,
            )
        top.set_title(dataset, fontweight="bold", pad=8)
        top.set_xlim(0, 20)
        bottom.set_xlim(0, 20)
        top.set_xticks([0, 5, 10, 15, 20])
        bottom.set_xticks([0, 5, 10, 15, 20])
        top.grid(alpha=0.42, linewidth=0.7)
        bottom.grid(alpha=0.42, linewidth=0.7)

    aids_top = axes[0, ACTIVE_DATASET_COLUMNS["AIDS"]]
    aids_bottom = axes[1, ACTIVE_DATASET_COLUMNS["AIDS"]]
    aids_top.set_ylabel("Coverage (%)")
    aids_top.set_ylim(0, coverage_ymax)
    aids_top.set_yticks(coverage_ticks)
    aids_bottom.set_ylabel("Cost")
    aids_bottom.set_ylim(0.008, 0.11)
    aids_bottom.set_yticks([0.02, 0.04, 0.06, 0.08, 0.10])

    mut_top = axes[0, ACTIVE_DATASET_COLUMNS["Mutagenicity"]]
    mut_bottom = axes[1, ACTIVE_DATASET_COLUMNS["Mutagenicity"]]
    mut_coverage = frame.loc[frame["Dataset"] == "Mutagenicity", "Coverage"]
    mut_cost = frame.loc[frame["Dataset"] == "Mutagenicity", "Cost"]
    if mut_matches_aids:
        mut_top.set_ylim(0, coverage_ymax)
        mut_top.set_yticks(coverage_ticks)
        mut_bottom.set_ylim(0.008, 0.11)
        mut_bottom.set_yticks([0.02, 0.04, 0.06, 0.08, 0.10])
    elif not mut_coverage.empty:
        ymax = max(80.0, 10.0 * np.ceil(100.0 * float(mut_coverage.max()) / 10.0))
        mut_top.set_ylim(0, min(100.0, ymax))
    if not mut_matches_aids and not mut_cost.empty:
        span = max(0.002, 0.06 * float(mut_cost.max() - mut_cost.min()))
        mut_bottom.set_ylim(max(0.0, float(mut_cost.min()) - span), float(mut_cost.max()) + span)

    for column, dataset in ((1, "NCI1"), (3, "Proteins")):
        _placeholder(axes[0, column], dataset)
        axes[1, column].axis("off")

    handles, labels = aids_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.015),
        columnspacing=2.0,
        handlelength=2.2,
    )
    fig.supxlabel("Size ($k$)", y=0.075, fontsize=14)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.16)
    _save_png_pdf(fig, output_dir, FIGURE3_OUTPUT_STEM)
    plt.close(fig)


def render_figure4(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    mut_matches_aids: bool,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE4_FIGSIZE, gridspec_kw={"wspace": 0.24})
    for dataset, column in ACTIVE_DATASET_COLUMNS.items():
        axis = axes[column]
        for method in METHODS:
            rows = _method_rows(frame, dataset, method, "Threshold")
            if rows.empty:
                continue
            style = METHOD_STYLES[method]
            axis.plot(
                rows["Threshold"],
                100.0 * rows["Coverage"],
                color=style["color"],
                marker=style["marker"],
                markevery=(
                    FIGURE4_AIDS_MARKEVERY
                    if dataset == "AIDS" or mut_matches_aids
                    else 1
                ),
                markersize=5.5,
                markeredgewidth=0.9,
                label=style["label"],
            )
        axis.set_title(dataset, fontweight="bold", pad=8)
        axis.set_xlabel(r"Distance threshold ($\theta$)")
        axis.grid(alpha=0.42, linewidth=0.7)

    aids_axis = axes[ACTIVE_DATASET_COLUMNS["AIDS"]]
    aids_axis.set_ylabel("Coverage (%)")
    aids_axis.set_xlim(AIDS_FIGURE4_THRESHOLD_MIN, AIDS_FIGURE4_THRESHOLD_MAX)
    aids_axis.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    aids_axis.set_ylim(0, 79.5)
    aids_axis.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

    mut_axis = axes[ACTIVE_DATASET_COLUMNS["Mutagenicity"]]
    mut_rows = frame.loc[frame["Dataset"] == "Mutagenicity"]
    if mut_matches_aids:
        mut_axis.set_xlim(AIDS_FIGURE4_THRESHOLD_MIN, AIDS_FIGURE4_THRESHOLD_MAX)
        mut_axis.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
        mut_axis.set_ylim(0, 79.5)
        mut_axis.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    elif not mut_rows.empty:
        mut_axis.set_xlim(float(mut_rows["Threshold"].min()), float(mut_rows["Threshold"].max()))
        mut_ymax = max(80.0, 10.0 * np.ceil(100.0 * float(mut_rows["Coverage"].max()) / 10.0))
        mut_axis.set_ylim(0, min(100.0, mut_ymax))

    for column, dataset in ((1, "NCI1"), (3, "Proteins")):
        _placeholder(axes[column], dataset)

    handles, labels = aids_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=2.0,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.24)
    _save_png_pdf(fig, output_dir, FIGURE4_OUTPUT_STEM)
    plt.close(fig)


def distinct_best_and_second(
    values: pd.Series,
    *,
    higher_is_better: bool,
) -> tuple[float, float | None]:
    finite = [float(value) for value in values if pd.notna(value)]
    unique = sorted(set(finite), reverse=higher_is_better)
    if not unique:
        return float("nan"), None
    return unique[0], unique[1] if len(unique) > 1 else None


def table_cell_styles(table: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    styles: dict[tuple[str, str, str], str] = {}
    for dataset in ("AIDS", "Mutagenicity"):
        rows = table.loc[table["Dataset"] == dataset]
        if rows.empty:
            continue
        for metric, higher in (("Coverage", True), ("Cost", False)):
            best, second = distinct_best_and_second(rows[metric], higher_is_better=higher)
            for _, row in rows.iterrows():
                value = float(row[metric])
                if np.isclose(value, best, rtol=0.0, atol=1e-12):
                    styles[(dataset, str(row["Method"]), metric)] = "best"
                elif second is not None and np.isclose(value, second, rtol=0.0, atol=1e-12):
                    styles[(dataset, str(row["Method"]), metric)] = "second"
                else:
                    styles[(dataset, str(row["Method"]), metric)] = "normal"
    return styles


def _table_matrix(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        row: dict[str, Any] = {"Method": method}
        for dataset in DATASET_ORDER:
            match = table.loc[(table["Dataset"] == dataset) & (table["Method"] == method)]
            row[f"{dataset} Coverage"] = np.nan if match.empty else float(match.iloc[0]["Coverage"])
            row[f"{dataset} Cost"] = np.nan if match.empty else float(match.iloc[0]["Cost"])
        rows.append(row)
    return pd.DataFrame(rows)


def _display_metric(value: float, *, coverage: bool) -> str:
    if pd.isna(value):
        return "—"
    return f"{100.0 * value:.2f}%" if coverage else f"{value:.4f}"


def write_table_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    configure_matplotlib()
    matrix = _table_matrix(table)
    matrix.to_csv(output_dir / f"{TABLE2_OUTPUT_STEM}.csv", index=False)
    markdown = [
        "| Method | AIDS Coverage | AIDS Cost | NCI1 Coverage | NCI1 Cost | "
        "Mutagenicity Coverage | Mutagenicity Cost | Proteins Coverage | Proteins Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in matrix.iterrows():
        cells = [str(row["Method"])]
        for dataset in DATASET_ORDER:
            cells.append(_display_metric(float(row[f"{dataset} Coverage"]), coverage=True))
            cells.append(_display_metric(float(row[f"{dataset} Cost"]), coverage=False))
        markdown.append("| " + " | ".join(cells) + " |")
    (output_dir / f"{TABLE2_OUTPUT_STEM}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    styles = table_cell_styles(table)
    fig, axis = plt.subplots(figsize=TABLE2_FIGSIZE)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 1)
    axis.axis("off")
    method_x = 1.0
    metric_x = {
        ("AIDS", "Coverage"): 2.45,
        ("AIDS", "Cost"): 3.25,
        ("NCI1", "Coverage"): 4.25,
        ("NCI1", "Cost"): 5.05,
        ("Mutagenicity", "Coverage"): 6.15,
        ("Mutagenicity", "Cost"): 6.95,
        ("Proteins", "Coverage"): 8.05,
        ("Proteins", "Cost"): 8.85,
    }
    dataset_center = {"AIDS": 2.85, "NCI1": 4.65, "Mutagenicity": 6.55, "Proteins": 8.45}
    for y, linewidth in ((0.94, 1.4), (0.59, 0.9), (0.06, 1.4)):
        axis.plot([0.15, 9.75], [y, y], color="black", linewidth=linewidth)
    axis.text(method_x, 0.75, "Method", ha="center", va="center", fontsize=16, fontweight="bold")
    for dataset in DATASET_ORDER:
        axis.text(
            dataset_center[dataset],
            0.84,
            dataset,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        for metric in ("Coverage", "Cost"):
            axis.text(
                metric_x[(dataset, metric)],
                0.68,
                metric,
                ha="center",
                va="center",
                fontsize=14,
            )

    for row_index, method in enumerate(METHODS):
        y = (0.48, 0.35, 0.22, 0.09)[row_index]
        axis.text(method_x, y, method, ha="center", va="center", fontsize=15, fontvariant="small-caps")
        for dataset in DATASET_ORDER:
            for metric in ("Coverage", "Cost"):
                match = table.loc[(table["Dataset"] == dataset) & (table["Method"] == method)]
                value = np.nan if match.empty else float(match.iloc[0][metric])
                style = styles.get((dataset, method, metric), "normal")
                axis.text(
                    metric_x[(dataset, metric)],
                    y,
                    _display_metric(value, coverage=metric == "Coverage"),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold" if style == "best" else "normal",
                )
                if style == "second":
                    half_width = 0.25 if metric == "Coverage" else 0.22
                    x = metric_x[(dataset, metric)]
                    axis.plot(
                        [x - half_width, x + half_width],
                        [y - 0.035, y - 0.035],
                        color="black",
                        linewidth=0.8,
                    )
    _save_png_pdf(fig, output_dir, TABLE2_OUTPUT_STEM)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot frozen AIDS and Mutagenicity WNode artifacts in the GCF-style layout."
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--aids-figure3-csv", required=True)
    parser.add_argument("--aids-figure4-csv", required=True)
    parser.add_argument("--aids-ours-root", required=True)
    parser.add_argument("--aids-globalgce-root", required=True)
    parser.add_argument("--aids-clear-root", required=True)
    parser.add_argument("--aids-gcf-root", required=True)
    parser.add_argument("--mut-ours-root", required=True)
    parser.add_argument("--mut-globalgce-root", required=True)
    parser.add_argument("--mut-clear-root", required=True)
    parser.add_argument("--mut-gcf-root", required=True)
    parser.add_argument(
        "--mut-threshold-mode",
        choices=("native", "match-aids"),
        default="native",
        help="Keep MUT frozen thresholds or aggregate saved pair distances on the AIDS grid.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _args_paths(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    aids_figure3 = _resolve(project_root, args.aids_figure3_csv)
    aids_figure4 = _resolve(project_root, args.aids_figure4_csv)
    _assert_exact_project_path(aids_figure3, project_root, AIDS_FIGURE3_RELATIVE_PATH)
    _assert_exact_project_path(aids_figure4, project_root, AIDS_FIGURE4_RELATIVE_PATH)
    aids_roots = {
        "Ours": _resolve(project_root, args.aids_ours_root),
        "GlobalGCE": _resolve(project_root, args.aids_globalgce_root),
        "CLEAR": _resolve(project_root, args.aids_clear_root),
        "GCFExplainer": _resolve(project_root, args.aids_gcf_root),
    }
    for method, expected in AIDS_TABLE_ROOTS.items():
        _assert_exact_project_path(aids_roots[method], project_root, expected)
    mut_roots: dict[str, Path] = {
        "Ours": _resolve(project_root, args.mut_ours_root),
        "GlobalGCE": _resolve(project_root, args.mut_globalgce_root),
        "CLEAR": _resolve(project_root, args.mut_clear_root),
        "GCFExplainer": _resolve(project_root, args.mut_gcf_root),
    }
    return {
        "aids_figure3": aids_figure3,
        "aids_figure4": aids_figure4,
        "aids_roots": aids_roots,
        "mut_roots": mut_roots,
    }


def _write_audit(
    output_dir: Path,
    *,
    sources: Mapping[str, Any],
    mut_audit: Mapping[str, Any],
    mut_threshold_mode: str,
) -> None:
    lines = [
        "AIDS + Mutagenicity GCF-style WNode plot audit",
        f"distance_line={DISTANCE_LABEL}",
        f"distance_type={DISTANCE_TYPE}",
        f"cf_mode={CF_MODE}",
        f"dataset_order={','.join(DATASET_ORDER)}",
        f"method_order={','.join(METHODS)}",
        f"AIDS.figure3_source={sources['aids_figure3']['path']}",
        f"AIDS.figure3_rows={AIDS_FIGURE3_ROWS}",
        f"AIDS.figure3_theta={AIDS_FIGURE3_THETA}",
        f"AIDS.figure4_source={sources['aids_figure4']['path']}",
        f"AIDS.figure4_rows={AIDS_FIGURE4_ROWS}",
        f"AIDS.figure4_k={AIDS_FIGURE4_K}",
        f"AIDS.figure4_points_per_method={AIDS_FIGURE4_POINTS_PER_METHOD}",
        f"AIDS.figure4_threshold_min={AIDS_FIGURE4_THRESHOLD_MIN}",
        f"AIDS.figure4_threshold_max={AIDS_FIGURE4_THRESHOLD_MAX}",
        f"AIDS.table2_k={AIDS_TABLE2_K}",
        f"AIDS.table2_theta={AIDS_TABLE2_THETA}",
        f"AIDS.num_parents={EXPECTED_PARENT_COUNTS['AIDS']}",
        f"Mutagenicity.num_parents={EXPECTED_PARENT_COUNTS['Mutagenicity']}",
        f"Mutagenicity.threshold_mode={mut_threshold_mode}",
        f"Mutagenicity.theta={mut_audit['theta']:.17g}",
        f"Mutagenicity.figure4_threshold_count={mut_audit['figure4_threshold_count']}",
        f"Mutagenicity.figure4_threshold_min={mut_audit['figure4_threshold_min']:.17g}",
        f"Mutagenicity.figure4_threshold_max={mut_audit['figure4_threshold_max']:.17g}",
        "distance_recomputed=false",
        "teacher_recomputed=false",
        "candidate_order_changed=false",
        "[AIDS_MUT_WNODE_GCF_STYLE_V2_OK]",
    ]
    (output_dir / "combined_audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    paths = _args_paths(args, project_root)
    output_dir = _resolve(project_root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if output_dir.exists():
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    aids_figure3, aids_figure3_source = load_aids_figure3(paths["aids_figure3"])
    aids_figure4, aids_figure4_source = load_aids_figure4(paths["aids_figure4"])
    aids_table, aids_table_source = load_aids_table2(paths["aids_roots"])

    mut_figure3_parts: list[pd.DataFrame] = []
    mut_figure4_parts: list[pd.DataFrame] = []
    mut_table_rows: list[dict[str, Any]] = []
    mut_sources: dict[str, Any] = {}
    mut_matches_aids = args.mut_threshold_mode == "match-aids"
    mut_loader = load_mut_method if mut_matches_aids else load_mut_native_method
    for method in METHODS:
        root = paths["mut_roots"][method]
        figure3, figure4, table_row, source = mut_loader(root, method)
        mut_figure3_parts.append(figure3)
        mut_figure4_parts.append(figure4)
        mut_table_rows.append(table_row)
        mut_sources[method] = source

    mut_figure3 = pd.concat(mut_figure3_parts, ignore_index=True)
    mut_figure4 = pd.concat(mut_figure4_parts, ignore_index=True)
    mut_table = pd.DataFrame(mut_table_rows)
    mut_audit = (
        validate_mut_alignment(mut_figure4, mut_table)
        if mut_matches_aids
        else validate_mut_native_alignment(mut_figure4, mut_table)
    )
    figure3 = pd.concat((aids_figure3, mut_figure3), ignore_index=True)
    figure4 = pd.concat((aids_figure4, mut_figure4), ignore_index=True)
    table = pd.concat((aids_table, mut_table), ignore_index=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    try:
        figure3.to_csv(temp_dir / f"{FIGURE3_OUTPUT_STEM}_data.csv", index=False)
        figure4.to_csv(temp_dir / f"{FIGURE4_OUTPUT_STEM}_data.csv", index=False)
        render_figure3(figure3, temp_dir, mut_matches_aids=mut_matches_aids)
        render_figure4(figure4, temp_dir, mut_matches_aids=mut_matches_aids)
        write_table_outputs(table, temp_dir)
        sources = {
            "aids_figure3": aids_figure3_source,
            "aids_figure4": aids_figure4_source,
            "aids_table2": aids_table_source,
            "mutagenicity": mut_sources,
        }
        _write_audit(
            temp_dir,
            sources=sources,
            mut_audit=mut_audit,
            mut_threshold_mode=args.mut_threshold_mode,
        )
        manifest = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_order": list(DATASET_ORDER),
            "active_dataset_columns": ACTIVE_DATASET_COLUMNS,
            "method_order": list(METHODS),
            "distance_line": DISTANCE_LABEL,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "aids": {
                "num_parents": EXPECTED_PARENT_COUNTS["AIDS"],
                "figure3_theta": AIDS_FIGURE3_THETA,
                "table2_theta": AIDS_TABLE2_THETA,
                "table2_k": AIDS_TABLE2_K,
                "figure4_k": AIDS_FIGURE4_K,
                "figure4_threshold_min": AIDS_FIGURE4_THRESHOLD_MIN,
                "figure4_threshold_max": AIDS_FIGURE4_THRESHOLD_MAX,
                "figure4_points_per_method": AIDS_FIGURE4_POINTS_PER_METHOD,
            },
            "mutagenicity": mut_audit,
            "mutagenicity_threshold_mode": args.mut_threshold_mode,
            "sources": sources,
            "distance_recomputed": False,
            "teacher_recomputed": False,
            "candidate_ranking_recomputed": False,
            "candidate_order_changed": False,
            "selection_performed_in_plot": False,
            "outputs": {},
        }
        for path in sorted(temp_dir.iterdir()):
            if path.is_file() and path.name not in {"combined_manifest.json", "_RUN_COMPLETE.json"}:
                manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        (temp_dir / "combined_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        complete = {
            "run_complete": True,
            "distance_line": DISTANCE_LABEL,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "manifest_sha256": sha256(temp_dir / "combined_manifest.json"),
        }
        (temp_dir / "_RUN_COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"output_dir={output_dir}")
    print(f"distance_line={DISTANCE_LABEL}")
    print("[AIDS_MUT_WNODE_GCF_STYLE_V2_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
