#!/usr/bin/env python3
"""Evaluate CCRCov with MolCLR node-level FGW distance.

This is an evaluation-only auxiliary distance line. It does not modify training
losses, PPO, selector logic, or redundancy calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.ccrcov_distance_eval import (  # noqa: E402
    CF_MODES,
    GT_DIRECTORY_CANDIDATES,
    OURS_DIRECTORY_CANDIDATES,
    _evaluate_gt_fullgraph,
    _evaluate_ours,
    _parse_int_label,
)
from src.eval.close_counterfactual_coverage import (  # noqa: E402
    DETAIL_FIELDS,
    _as_bool,
    _as_float,
    _load_candidate_records,
    _load_parent_records,
    canonicalize_smiles,
)
from src.eval.greed_distance.pair_generation import GT_FULLGRAPH_FIELDS, OURS_FRAGMENT_FIELDS  # noqa: E402
from src.eval.node_fgw_distance import (  # noqa: E402
    DEFAULT_NODE_EMB_CACHE_DIR,
    MolCLRNodeFGWDistanceProvider,
    NodeFGWConfig,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402
from src.utils.io import ensure_directory  # noqa: E402


DISTANCE_LINE = "MolCLR-Node-FGW"
DISTANCE_TYPE = "node_fgw"
DISTANCE_NAME = "molclr_node_fgw"
DEFAULT_QUANTILES = "0.05,0.10,0.20,0.30,0.50,0.70,0.90"
DEFAULT_CACHE_DB = "outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite"
SUMMARY_FIELDS = [
    "method",
    "distance_type",
    "distance_line",
    "fgw_lambda",
    "structure_mode",
    "feature_cost",
    "atom_penalty",
    "threshold",
    "threshold_source",
    "quantile",
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
    "total_pairs",
    "cache_hit_rate",
    "node_embedding_cache_hit_rate",
    "skip_redundancy",
    "selection_performed_in_eval",
    "candidate_set_preselected",
    "selection_method",
    "evaluation_row_unit",
    "num_unique_parent_candidate_pairs",
    "num_detail_rows",
    "num_valid_match_instances",
]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    destination = Path(path).expanduser()
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else ("" if row.get(field) is None else row.get(field))
                    for field in fieldnames
                }
            )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser()
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_smiles_from_row(row: dict[str, Any]) -> str:
    for field in GT_FULLGRAPH_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def validate_preselected_candidate_csv(path_like: str | Path, expected_top_k: int) -> dict[str, Any]:
    """Require an exact, ordered, RDKit-valid unique candidate CSV."""

    path = Path(path_like).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"Preselected fullgraph input must be a CSV file: {path}")
    canonical: list[str] = []
    strict_flip_values: list[bool] = []
    ranks: list[int] = []
    selection_modes: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        has_strict_flip_field = "rf_strict_flip" in (reader.fieldnames or [])
        has_rank_field = "rank" in (reader.fieldnames or [])
        for row_index, row in enumerate(reader):
            smiles = _candidate_smiles_from_row(row)
            if not smiles:
                raise ValueError(f"Preselected candidate row {row_index} has no recognized SMILES field: {path}")
            normalized = canonicalize_smiles(smiles)
            if not normalized:
                raise ValueError(f"Preselected candidate row {row_index} is not RDKit-valid: {smiles!r}")
            canonical.append(normalized)
            if has_strict_flip_field:
                strict_flip_values.append(_as_bool(row.get("rf_strict_flip")))
            if has_rank_field:
                rank = _parse_int_label(row.get("rank"))
                if rank is None:
                    raise ValueError(f"Preselected candidate row {row_index} has an invalid rank: {path}")
                ranks.append(int(rank))
            selection_mode = str(row.get("selection_mode") or "").strip()
            if selection_mode:
                selection_modes.add(selection_mode)
    if len(canonical) != int(expected_top_k):
        raise ValueError(
            f"Preselected candidate CSV must contain exactly {expected_top_k} rows; "
            f"found {len(canonical)}: {path}"
        )
    if len(set(canonical)) != int(expected_top_k):
        raise ValueError(
            f"Preselected candidate CSV must contain {expected_top_k} unique canonical SMILES; "
            f"found {len(set(canonical))}: {path}"
        )
    if strict_flip_values and not all(strict_flip_values):
        raise ValueError(f"Preselected candidate CSV contains rf_strict_flip=false rows: {path}")
    if ranks and ranks != list(range(1, int(expected_top_k) + 1)):
        raise ValueError(f"Preselected candidate ranks must be exactly 1..{expected_top_k}: {path}")
    if len(selection_modes) > 1:
        raise ValueError(f"Preselected candidate CSV contains mixed selection_mode values: {path}")
    return {
        "input_kind": "fullgraph",
        "path": str(path),
        "expected_top_k": int(expected_top_k),
        "num_rows": len(canonical),
        "num_unique_canonical_smiles": len(set(canonical)),
        "rf_strict_flip_validated": bool(strict_flip_values),
        "rank_order_validated": bool(ranks),
        "order_preserved": True,
        "selection_method": next(iter(selection_modes), "preselected_external"),
    }


def _read_selected_subgraph_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("selected_subgraphs", "selected_rows", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported selected-subgraph JSON schema: {path}")


def _selected_fragment_from_row(row: dict[str, Any]) -> str:
    for field in OURS_FRAGMENT_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def _fragment_identity(smiles: str) -> str:
    return canonicalize_smiles(smiles) or str(smiles or "").strip()


def _infer_ours_selection_method(directory: Path, summary: dict[str, Any]) -> str:
    for container in (summary, summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}):
        for key in ("selection_method", "selector_name", "selector_version"):
            value = str(container.get(key) or "").strip()
            if value:
                return value
    match = re.search(r"mmr_cov([0-9]+(?:p[0-9]+)?)", directory.name.lower())
    if match:
        return f"greedy_mmr_cov{match.group(1).replace('p', '.')}"
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    beta_coverage = _as_float(metadata.get("beta_coverage"))
    if beta_coverage is not None:
        rendered = str(int(beta_coverage)) if float(beta_coverage).is_integer() else str(beta_coverage)
        return f"greedy_mmr_cov{rendered}"
    return "ours_external_selector"


def validate_preselected_ours_directory(path_like: str | Path, expected_top_k: int) -> dict[str, Any]:
    """Validate an externally selected Ours fragment directory without reordering it."""

    directory = Path(path_like).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Ours preselected input must be a directory: {directory}")
    selected_path = next(
        (candidate for candidate in (directory / "selected_subgraphs.csv", directory / "selected_subgraphs.json") if candidate.is_file()),
        None,
    )
    if selected_path is None:
        raise ValueError(
            f"Ours preselected directory is missing selected_subgraphs.csv or selected_subgraphs.json: {directory}"
        )
    rows = _read_selected_subgraph_rows(selected_path)
    fragments: list[str] = []
    fragment_keys: list[str] = []
    candidate_ids: list[str] = []
    ranks: list[int] = []
    has_rank = bool(rows) and any(str(row.get("rank") or "").strip() for row in rows)
    for row_index, row in enumerate(rows):
        fragment = _selected_fragment_from_row(row)
        if not fragment:
            raise ValueError(f"Ours selected row {row_index} has no recognized fragment field: {selected_path}")
        fragments.append(fragment)
        fragment_keys.append(_fragment_identity(fragment))
        rank_value = _parse_int_label(row.get("rank")) if has_rank else None
        if has_rank:
            if rank_value is None:
                raise ValueError(f"Ours selected row {row_index} has an invalid or missing rank: {selected_path}")
            ranks.append(int(rank_value))
        candidate_id = str(
            row.get("candidate_id")
            or row.get("id")
            or row.get("rank")
            or row.get("selected_step")
            or row_index
        ).strip()
        candidate_ids.append(candidate_id)
    if len(rows) != int(expected_top_k):
        raise ValueError(
            f"Ours selected directory must contain exactly {expected_top_k} rows; found {len(rows)}: {selected_path}"
        )
    if len(set(fragment_keys)) != int(expected_top_k):
        raise ValueError(
            f"Ours selected directory must contain {expected_top_k} unique fragments; "
            f"found {len(set(fragment_keys))}: {selected_path}"
        )
    if len(set(candidate_ids)) != int(expected_top_k):
        raise ValueError(
            f"Ours selected directory contains duplicate candidate identifiers: {selected_path}"
        )
    if ranks and ranks != list(range(1, int(expected_top_k) + 1)):
        raise ValueError(f"Ours selected ranks must be exactly 1..{expected_top_k} in file order: {selected_path}")
    selector_summary_path = directory / "selector_summary.json"
    selector_summary: dict[str, Any] = {}
    summary_warning: str | None = None
    if selector_summary_path.is_file():
        try:
            payload = json.loads(selector_summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                selector_summary = payload
            else:
                summary_warning = "selector_summary_root_not_object"
        except Exception as exc:
            summary_warning = f"selector_summary_read_failed:{exc}"
    selection_method = _infer_ours_selection_method(directory, selector_summary)
    return {
        "input_kind": "ours_selected_subgraphs",
        "path": str(directory),
        "selected_file": str(selected_path),
        "selector_summary_path": str(selector_summary_path) if selector_summary_path.is_file() else None,
        "selector_summary_warning": summary_warning,
        "expected_top_k": int(expected_top_k),
        "num_rows": len(rows),
        "num_unique_candidate_ids": len(set(candidate_ids)),
        "num_unique_fragments": len(set(fragment_keys)),
        "rank_order_validated": bool(ranks),
        "order_preserved": True,
        "selection_method": selection_method,
        "ordered_fragments": fragments,
        "candidate_selection_stage": "external_before_evaluation",
        "subgraph_matching_stage": "inside_evaluation_not_candidate_selection",
    }


def build_evaluation_row_audit(
    details: Sequence[dict[str, Any]],
    *,
    evaluation_row_unit: str,
) -> dict[str, Any]:
    unique_pairs = {
        (str(row.get("parent_id") or ""), str(row.get("candidate_id") or ""))
        for row in details
    }
    valid_match_instances: int | None = None
    if evaluation_row_unit == "match_instance":
        valid_match_instances = sum(
            1
            for row in details
            if _as_bool(row.get("match")) and _as_bool(row.get("delete_valid"))
        )
    return {
        "evaluation_row_unit": evaluation_row_unit,
        "num_unique_parent_candidate_pairs": len(unique_pairs),
        "num_detail_rows": len(details),
        "num_valid_match_instances": valid_match_instances,
    }


def _finite_distance(row: dict[str, Any]) -> float | None:
    value = _as_float(row.get("distance"))
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _cf_condition(row: dict[str, Any], *, label: int, cf_mode: str, min_cf_drop: float) -> bool:
    pred_after = _parse_int_label(row.get("pred_after"))
    strict_flip = pred_after is not None and int(pred_after) != int(label)
    cf_drop = _as_float(row.get("cf_drop"))
    drop_ok = cf_drop is not None and float(cf_drop) >= float(min_cf_drop)
    if cf_mode == "strict_flip":
        return bool(strict_flip)
    if cf_mode == "drop_or_flip":
        return bool(strict_flip or drop_ok)
    if cf_mode == "drop_only":
        return bool(drop_ok)
    raise ValueError(f"Unsupported cf_mode={cf_mode!r}")


def _row_flip(row: dict[str, Any]) -> float:
    return 1.0 if _as_bool(row.get("cf_flip")) else 0.0


def _best_row(rows: list[dict[str, Any]], *, threshold: float, label: int, cf_mode: str, min_cf_drop: float) -> tuple[dict[str, Any] | None, bool, bool]:
    best: tuple[float, float, dict[str, Any]] | None = None
    close_only = False
    close_cf = False
    for row in rows:
        distance = _finite_distance(row)
        if distance is None or distance > float(threshold):
            continue
        close_only = True
        if not _cf_condition(row, label=label, cf_mode=cf_mode, min_cf_drop=min_cf_drop):
            continue
        close_cf = True
        cf_drop = _as_float(row.get("cf_drop"))
        candidate_key = (float(distance), -float(cf_drop if cf_drop is not None else -1e9), row)
        if best is None or candidate_key[:2] < best[:2]:
            best = candidate_key
    return (best[2] if best else None), close_only, close_cf


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantiles from an empty distance list.")
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = float(q) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def build_threshold_table(raw_thresholds: str | None, quantiles_raw: str, details: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
    distances = [distance for distance in (_finite_distance(row) for row in details) if distance is not None]
    if raw_thresholds and str(raw_thresholds).strip() not in {"", "auto", "auto_quantile"}:
        thresholds = _parse_float_list(raw_thresholds)
        rows = [{"threshold_source": "explicit", "quantile": None, "threshold": threshold} for threshold in thresholds]
        return rows, thresholds
    quantiles = _parse_float_list(quantiles_raw or DEFAULT_QUANTILES)
    rows: list[dict[str, Any]] = []
    thresholds: list[float] = []
    for quantile in quantiles:
        threshold = _quantile(distances, quantile) if distances else float("nan")
        rows.append({"threshold_source": "auto_quantile", "quantile": float(quantile), "threshold": threshold})
        if math.isfinite(float(threshold)):
            thresholds.append(float(threshold))
    return rows, thresholds


def summarize_method(
    *,
    method: str,
    details: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    fgw_lambda: float,
    structure_mode: str,
    feature_cost: str,
    atom_penalty: float,
    cf_mode: str,
    min_cf_drop: float,
    cache_hit_rate: float,
    node_embedding_cache_hit_rate: float,
    skip_redundancy: bool,
    selection_performed_in_eval: bool,
    candidate_set_preselected: bool,
    selection_method: str,
    evaluation_row_unit: str,
    num_unique_parent_candidate_pairs: int,
    num_detail_rows: int,
    num_valid_match_instances: int | None,
) -> list[dict[str, Any]]:
    rows_by_parent: dict[str, list[dict[str, Any]]] = {}
    labels_by_parent: dict[str, int] = {}
    for row in details:
        parent_id = str(row.get("parent_id") or "")
        rows_by_parent.setdefault(parent_id, []).append(row)
        label = _parse_int_label(row.get("label"))
        if label is not None:
            labels_by_parent[parent_id] = label
    num_valid_pairs = sum(1 for row in details if _finite_distance(row) is not None)
    output: list[dict[str, Any]] = []
    for threshold_row in threshold_rows:
        threshold = threshold_row["threshold"]
        if not math.isfinite(float(threshold)):
            continue
        close_only: set[str] = set()
        close_cf: set[str] = set()
        best_rows: list[dict[str, Any]] = []
        for parent_id, rows in rows_by_parent.items():
            best, is_close, is_cf = _best_row(
                rows,
                threshold=float(threshold),
                label=labels_by_parent.get(parent_id, 0),
                cf_mode=cf_mode,
                min_cf_drop=float(min_cf_drop),
            )
            if is_close:
                close_only.add(parent_id)
            if is_cf:
                close_cf.add(parent_id)
            if best is not None:
                best_rows.append(best)
        output.append(
            {
                "method": method,
                "distance_type": DISTANCE_TYPE,
                "distance_line": DISTANCE_LINE,
                "fgw_lambda": float(fgw_lambda),
                "structure_mode": structure_mode,
                "feature_cost": feature_cost,
                "atom_penalty": float(atom_penalty),
                "threshold": float(threshold),
                "threshold_source": threshold_row.get("threshold_source"),
                "quantile": threshold_row.get("quantile"),
                "num_parents": int(total_parents),
                "num_candidates": int(total_candidates),
                "num_valid_pairs": int(num_valid_pairs),
                "num_close_only_covered": len(close_only),
                "close_only_coverage": _rate(len(close_only), total_parents),
                "num_close_cf_covered": len(close_cf),
                "close_cf_coverage": _rate(len(close_cf), total_parents),
                "avg_best_distance": _mean(_finite_distance(row) for row in best_rows),
                "median_best_distance": _median(_finite_distance(row) for row in best_rows),
                "avg_cf_drop_among_covered": _mean(_as_float(row.get("cf_drop")) for row in best_rows),
                "flip_rate_among_covered": _mean(_row_flip(row) for row in best_rows),
                "total_pairs": len(details),
                "cache_hit_rate": float(cache_hit_rate),
                "node_embedding_cache_hit_rate": float(node_embedding_cache_hit_rate),
                "skip_redundancy": bool(skip_redundancy),
                "selection_performed_in_eval": bool(selection_performed_in_eval),
                "candidate_set_preselected": bool(candidate_set_preselected),
                "selection_method": selection_method,
                "evaluation_row_unit": evaluation_row_unit,
                "num_unique_parent_candidate_pairs": int(num_unique_parent_candidate_pairs),
                "num_detail_rows": int(num_detail_rows),
                "num_valid_match_instances": num_valid_match_instances,
            }
        )
    return output


def _detail_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = list(DETAIL_FIELDS)
    for extra in (
        "distance_line",
        "fgw_lambda",
        "structure_mode",
        "feature_cost",
        "atom_penalty",
        "skip_redundancy",
    ):
        if extra not in fields:
            fields.append(extra)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def infer_fullgraph_method(path: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    text = str(path or "").lower()
    if "clear" in text:
        return "CLEAR-RF-FullGraph"
    if "gcf" in text and "gt_fullgraph" not in text:
        return "GCFExplainer-FullGraph"
    return "gt_fullgraph"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-csv", default=_env("HIV_CSV", "data/raw/AIDS/HIV.csv"))
    parser.add_argument("--ours-selected-path", default=_env("OURS_SELECTED_PATH"))
    parser.add_argument("--gt-fullgraph-candidates-path", default=_env("GT_FULLGRAPH_CANDIDATES_PATH"))
    parser.add_argument("--gcf-candidates-path", default=_env("GCF_CANDIDATES_PATH"))
    parser.add_argument("--clear-fullgraph-candidates-path", default=_env("CLEAR_FULLGRAPH_CANDIDATES_PATH"))
    parser.add_argument("--fullgraph-method-name", default=_env("FULLGRAPH_METHOD_NAME"))
    parser.add_argument("--teacher-path", default=_env("TEACHER_PATH"))
    parser.add_argument("--molclr-root", default=_env("MOLCLR_ROOT"))
    parser.add_argument("--molclr-checkpoint", default=_env("MOLCLR_CKPT"))
    parser.add_argument("--label", type=int, default=int(_env("TARGET_LABEL", "1") or 1))
    parser.add_argument("--smiles-col", default=_env("SMILES_COL", "smiles"))
    parser.add_argument("--label-col", default=_env("LABEL_COL", _env("LABEL_COLUMN", "HIV_active")))
    parser.add_argument("--cf-mode", choices=CF_MODES, default=_env("CF_MODE", "strict_flip"))
    parser.add_argument("--min-cf-drop", type=float, default=float(_env("MIN_CF_DROP", "0.0") or 0.0))
    parser.add_argument("--output-dir", default=_env("OUTPUT_DIR", "outputs/hpc/eval/ccrcov_molclr_node_fgw_smoke"))
    parser.add_argument("--max-parents", type=int, default=int(_env("MAX_PARENTS", "50") or 50))
    parser.add_argument("--max-candidates", type=int, default=int(_env("MAX_CANDIDATES", "20") or 20))
    parser.add_argument("--fgw-lambda", type=float, default=float(_env("FGW_LAMBDA", "0.5") or 0.5))
    parser.add_argument("--fgw-thresholds", default=_env("FGW_THRESHOLDS", "auto_quantile"))
    parser.add_argument("--fgw-quantiles", default=_env("FGW_QUANTILES", DEFAULT_QUANTILES))
    parser.add_argument("--fgw-cache-db", default=_env("FGW_CACHE_DB", DEFAULT_CACHE_DB))
    parser.add_argument("--node-emb-cache-dir", default=_env("NODE_EMB_CACHE_DIR", DEFAULT_NODE_EMB_CACHE_DIR))
    parser.add_argument("--structure-mode", default=_env("STRUCTURE_MODE", "shortest_path_unweighted"))
    parser.add_argument("--feature-cost", default=_env("FEATURE_COST", "cosine"))
    parser.add_argument("--atom-penalty", type=float, default=float(_env("ATOM_PENALTY", "0.0") or 0.0))
    parser.add_argument("--device", default=_env("DEVICE", "cuda"))
    parser.add_argument("--encoder-type", default=_env("MOLCLR_ENCODER_TYPE", "gin"))
    parser.add_argument("--fgw-max-iter", type=int, default=int(_env("FGW_MAX_ITER", "100") or 100))
    parser.add_argument("--fgw-tol", type=float, default=float(_env("FGW_TOL", "1e-7") or 1e-7))
    parser.add_argument("--partial-every", type=int, default=int(_env("PARTIAL_EVERY", "500") or 500))
    parser.add_argument("--skip-redundancy", action="store_true", default=True)
    parser.add_argument(
        "--run-ours",
        type=int,
        choices=(0, 1),
        default=1 if _env_bool("RUN_OURS", True) else 0,
        help="Evaluate ours selected-subgraph deletion candidates.",
    )
    parser.add_argument(
        "--run-fullgraph",
        type=int,
        choices=(0, 1),
        default=1 if _env_bool("RUN_FULLGRAPH", True) else 0,
        help="Evaluate fullgraph candidates, including GT/GCF/CLEAR fullgraph paths.",
    )
    parser.add_argument(
        "--preselected-topk",
        type=int,
        default=int(_env("PRESELECTED_TOPK", "0") or 0),
        help="Expected size of each preselected fullgraph or Ours selected-subgraph set; 0 disables the check.",
    )
    parser.add_argument(
        "--require-preselected-topk",
        type=int,
        choices=(0, 1),
        default=1 if _env_bool("REQUIRE_PRESELECTED_TOPK", False) else 0,
        help="Fail unless active fullgraph/Ours inputs are exact preselected sets of --preselected-topk candidates.",
    )
    return parser


def main() -> int:
    started = time.time()
    args = build_parser().parse_args()
    if not args.teacher_path:
        raise SystemExit("[ERROR] TEACHER_PATH is required.")
    if not args.molclr_root or not args.molclr_checkpoint:
        raise SystemExit("[ERROR] MOLCLR_ROOT and MOLCLR_CKPT are required.")
    output = ensure_directory(Path(args.output_dir).expanduser())
    details_dir = ensure_directory(output / "details")
    combined_dir = ensure_directory(output / "combined")
    fullgraph_path = args.gt_fullgraph_candidates_path or args.gcf_candidates_path
    fullgraph_method = infer_fullgraph_method(fullgraph_path, args.fullgraph_method_name)
    preselected_requested = int(args.preselected_topk) > 0
    if bool(args.require_preselected_topk) and not preselected_requested:
        raise SystemExit("[ERROR] REQUIRE_PRESELECTED_TOPK=1 requires PRESELECTED_TOPK > 0.")
    if bool(args.require_preselected_topk) and args.cf_mode != "strict_flip":
        raise SystemExit("[ERROR] Preselected final evaluation requires CF_MODE=strict_flip.")
    preselected_audits: list[dict[str, Any]] = []
    preselected_audit_by_input: dict[str, dict[str, Any]] = {}
    if preselected_requested:
        if bool(args.run_ours) and not args.ours_selected_path and bool(args.require_preselected_topk):
            raise SystemExit("[ERROR] Preselected Ours evaluation requires OURS_SELECTED_PATH.")
        try:
            if bool(args.run_ours) and args.ours_selected_path:
                ours_audit = validate_preselected_ours_directory(
                    args.ours_selected_path,
                    int(args.preselected_topk),
                )
                preselected_audits.append(ours_audit)
                preselected_audit_by_input["ours"] = ours_audit
            if bool(args.run_fullgraph) and fullgraph_path:
                fullgraph_audit = validate_preselected_candidate_csv(
                    fullgraph_path,
                    int(args.preselected_topk),
                )
                preselected_audits.append(fullgraph_audit)
                preselected_audit_by_input["fullgraph"] = fullgraph_audit
            if bool(args.run_fullgraph) and args.clear_fullgraph_candidates_path:
                clear_audit = validate_preselected_candidate_csv(
                    args.clear_fullgraph_candidates_path,
                    int(args.preselected_topk),
                )
                preselected_audits.append(clear_audit)
                preselected_audit_by_input["clear"] = clear_audit
        except ValueError as exc:
            raise SystemExit(f"[ERROR] preselected candidate validation failed: {exc}") from exc
        if bool(args.require_preselected_topk) and not preselected_audits:
            raise SystemExit("[ERROR] Preselected mode requires at least one active candidate input.")
    candidate_set_preselected = bool(preselected_audits)
    selection_methods = {
        str(audit.get("selection_method") or "preselected_external")
        for audit in preselected_audits
    }
    selection_method = (
        next(iter(selection_methods))
        if len(selection_methods) == 1
        else ("mixed_preselected" if selection_methods else "not_preselected")
    )

    print("[MOLCLR_NODE_FGW_CONFIG]", flush=True)
    print(f"distance_line={DISTANCE_LINE}", flush=True)
    print(f"dataset_csv={args.dataset_csv}", flush=True)
    print(f"teacher_path={args.teacher_path}", flush=True)
    print(f"molclr_root={args.molclr_root}", flush=True)
    print(f"molclr_checkpoint={args.molclr_checkpoint}", flush=True)
    print(f"fgw_lambda={args.fgw_lambda}", flush=True)
    print(f"skip_redundancy={args.skip_redundancy}", flush=True)
    print(f"run_ours={bool(args.run_ours)}", flush=True)
    print(f"run_fullgraph={bool(args.run_fullgraph)}", flush=True)
    print(f"preselected_topk={args.preselected_topk}", flush=True)
    print(f"require_preselected_topk={bool(args.require_preselected_topk)}", flush=True)
    print(f"candidate_set_preselected={candidate_set_preselected}", flush=True)
    print(f"selection_method={selection_method}", flush=True)

    provider = MolCLRNodeFGWDistanceProvider(
        NodeFGWConfig(
            molclr_root=args.molclr_root,
            molclr_ckpt=args.molclr_checkpoint,
            fgw_lambda=float(args.fgw_lambda),
            structure_mode=args.structure_mode,
            feature_cost=args.feature_cost,
            atom_penalty=float(args.atom_penalty),
            max_iter=int(args.fgw_max_iter),
            tol=float(args.fgw_tol),
            device=args.device,
            encoder_type=args.encoder_type,
            cache_db=args.fgw_cache_db,
            node_emb_cache_dir=args.node_emb_cache_dir,
        )
    )
    _dataset_path, parents, _actual_label_col = _load_parent_records(
        args.dataset_csv,
        label=int(args.label),
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        max_parents=(int(args.max_parents) if args.max_parents is not None and int(args.max_parents) > 0 else None),
    )
    teacher = TeacherSemanticScorer(args.teacher_path)
    all_details: list[dict[str, Any]] = []
    detail_groups: list[dict[str, Any]] = []

    if bool(args.run_ours) and args.ours_selected_path:
        _ours_path, ours_candidates = _load_candidate_records(
            args.ours_selected_path,
            fields=OURS_FRAGMENT_FIELDS,
            directory_candidates=OURS_DIRECTORY_CANDIDATES,
        )
        ours_count_before_limit = len(ours_candidates)
        ours_preselected_audit = preselected_audit_by_input.get("ours")
        if ours_preselected_audit is None and args.max_candidates is not None and int(args.max_candidates) > 0:
            ours_candidates = ours_candidates[: int(args.max_candidates)]
        if ours_preselected_audit is not None:
            if len(ours_candidates) != int(args.preselected_topk):
                raise SystemExit(
                    f"[ERROR] Loaded Ours candidate count changed after directory validation: "
                    f"expected={args.preselected_topk} loaded={len(ours_candidates)}"
                )
            expected_fragments = [
                _fragment_identity(value)
                for value in ours_preselected_audit.get("ordered_fragments", [])
            ]
            loaded_fragments = [_fragment_identity(candidate.smiles) for candidate in ours_candidates]
            if loaded_fragments != expected_fragments:
                raise SystemExit("[ERROR] Ours candidate loader did not preserve selected-subgraph order.")
        ours_details = _evaluate_ours(
            parents=parents,
            candidates=ours_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=DISTANCE_TYPE,
            distance_name=DISTANCE_NAME,
            partial_path=details_dir / "ours.partial.csv",
            partial_every=int(args.partial_every),
        )
        ours_row_audit = build_evaluation_row_audit(
            ours_details,
            evaluation_row_unit="match_instance",
        )
        detail_groups.append(
            {
                "method": "ours_selected_subgraphs",
                "details": ours_details,
                "num_candidates": len(ours_candidates),
                "candidate_set_preselected": ours_preselected_audit is not None,
                "selection_performed_in_eval": (
                    False if ours_preselected_audit is not None else len(ours_candidates) < ours_count_before_limit
                ),
                "selection_method": (
                    str(ours_preselected_audit.get("selection_method"))
                    if ours_preselected_audit is not None
                    else "not_preselected"
                ),
                **ours_row_audit,
            }
        )
        all_details.extend(ours_details)

    if bool(args.run_fullgraph) and fullgraph_path:
        _full_path, full_candidates = _load_candidate_records(
            fullgraph_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=GT_DIRECTORY_CANDIDATES,
        )
        full_count_before_limit = len(full_candidates)
        full_preselected_audit = preselected_audit_by_input.get("fullgraph")
        if full_preselected_audit is None and args.max_candidates is not None and int(args.max_candidates) > 0:
            full_candidates = full_candidates[: int(args.max_candidates)]
        if full_preselected_audit is not None and len(full_candidates) != int(args.preselected_topk):
            raise SystemExit(
                f"[ERROR] Loaded fullgraph candidate count changed after CSV validation: "
                f"expected={args.preselected_topk} loaded={len(full_candidates)}"
            )
        full_details = _evaluate_gt_fullgraph(
            parents=parents,
            candidates=full_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=DISTANCE_TYPE,
            distance_name=DISTANCE_NAME,
            method=fullgraph_method,
            partial_path=details_dir / "fullgraph.partial.csv",
            partial_every=int(args.partial_every),
        )
        full_row_audit = build_evaluation_row_audit(
            full_details,
            evaluation_row_unit="parent_candidate_pair",
        )
        detail_groups.append(
            {
                "method": fullgraph_method,
                "details": full_details,
                "num_candidates": len(full_candidates),
                "candidate_set_preselected": full_preselected_audit is not None,
                "selection_performed_in_eval": (
                    False if full_preselected_audit is not None else len(full_candidates) < full_count_before_limit
                ),
                "selection_method": (
                    str(full_preselected_audit.get("selection_method"))
                    if full_preselected_audit is not None
                    else "not_preselected"
                ),
                **full_row_audit,
            }
        )
        all_details.extend(full_details)

    if bool(args.run_fullgraph) and args.clear_fullgraph_candidates_path:
        _clear_path, clear_candidates = _load_candidate_records(
            args.clear_fullgraph_candidates_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=GT_DIRECTORY_CANDIDATES,
        )
        clear_count_before_limit = len(clear_candidates)
        clear_preselected_audit = preselected_audit_by_input.get("clear")
        if clear_preselected_audit is None and args.max_candidates is not None and int(args.max_candidates) > 0:
            clear_candidates = clear_candidates[: int(args.max_candidates)]
        if clear_preselected_audit is not None and len(clear_candidates) != int(args.preselected_topk):
            raise SystemExit(
                f"[ERROR] Loaded CLEAR candidate count changed after CSV validation: "
                f"expected={args.preselected_topk} loaded={len(clear_candidates)}"
            )
        clear_details = _evaluate_gt_fullgraph(
            parents=parents,
            candidates=clear_candidates,
            teacher=teacher,
            provider=provider,
            distance_type=DISTANCE_TYPE,
            distance_name=DISTANCE_NAME,
            method="CLEAR-RF-FullGraph",
            partial_path=details_dir / "clear_rf_fullgraph.partial.csv",
            partial_every=int(args.partial_every),
        )
        clear_row_audit = build_evaluation_row_audit(
            clear_details,
            evaluation_row_unit="parent_candidate_pair",
        )
        detail_groups.append(
            {
                "method": "CLEAR-RF-FullGraph",
                "details": clear_details,
                "num_candidates": len(clear_candidates),
                "candidate_set_preselected": clear_preselected_audit is not None,
                "selection_performed_in_eval": (
                    False if clear_preselected_audit is not None else len(clear_candidates) < clear_count_before_limit
                ),
                "selection_method": (
                    str(clear_preselected_audit.get("selection_method"))
                    if clear_preselected_audit is not None
                    else "not_preselected"
                ),
                **clear_row_audit,
            }
        )
        all_details.extend(clear_details)

    if not detail_groups:
        raise SystemExit("[ERROR] No candidate inputs were provided.")

    for row in all_details:
        row.update(
            {
                "distance_line": DISTANCE_LINE,
                "fgw_lambda": float(args.fgw_lambda),
                "structure_mode": args.structure_mode,
                "feature_cost": args.feature_cost,
                "atom_penalty": float(args.atom_penalty),
                "skip_redundancy": bool(args.skip_redundancy),
            }
        )

    threshold_rows, thresholds = build_threshold_table(args.fgw_thresholds, args.fgw_quantiles, all_details)
    _write_csv(output / "distance_quantiles.csv", threshold_rows, ["threshold_source", "quantile", "threshold"])
    pair_stats = provider.cache.stats_dict()
    cache_hit_rate = float(pair_stats.get("pair_distance_cache_hit_rate") or 0.0)
    node_hit_rate = float(provider.stats_dict().get("node_embedding_cache_hit_rate") or 0.0)
    summaries: list[dict[str, Any]] = []
    for group in detail_groups:
        method = str(group["method"])
        details = list(group["details"])
        num_candidates = int(group["num_candidates"])
        summaries.extend(
            summarize_method(
                method=method,
                details=details,
                threshold_rows=threshold_rows,
                total_parents=len(parents),
                total_candidates=num_candidates,
                fgw_lambda=float(args.fgw_lambda),
                structure_mode=args.structure_mode,
                feature_cost=args.feature_cost,
                atom_penalty=float(args.atom_penalty),
                cf_mode=args.cf_mode,
                min_cf_drop=float(args.min_cf_drop),
                cache_hit_rate=cache_hit_rate,
                node_embedding_cache_hit_rate=node_hit_rate,
                skip_redundancy=bool(args.skip_redundancy),
                selection_performed_in_eval=bool(group["selection_performed_in_eval"]),
                candidate_set_preselected=bool(group["candidate_set_preselected"]),
                selection_method=str(group["selection_method"]),
                evaluation_row_unit=str(group["evaluation_row_unit"]),
                num_unique_parent_candidate_pairs=int(group["num_unique_parent_candidate_pairs"]),
                num_detail_rows=int(group["num_detail_rows"]),
                num_valid_match_instances=group["num_valid_match_instances"],
            )
        )
    _write_csv(details_dir / "pair_details.csv", all_details, _detail_fields(all_details))
    _write_csv(combined_dir / "combined_threshold_summary.csv", summaries, SUMMARY_FIELDS)
    _write_json(combined_dir / "combined_threshold_summary.json", {"threshold_summary": summaries})

    all_groups_preselected = bool(detail_groups) and all(
        bool(group["candidate_set_preselected"]) for group in detail_groups
    )
    any_selection_in_eval = any(bool(group["selection_performed_in_eval"]) for group in detail_groups)
    group_selection_methods = {str(group["selection_method"]) for group in detail_groups}
    run_selection_method = (
        next(iter(group_selection_methods))
        if len(group_selection_methods) == 1
        else "mixed"
    )
    row_units = {str(group["evaluation_row_unit"]) for group in detail_groups}
    valid_match_counts = [
        int(group["num_valid_match_instances"])
        for group in detail_groups
        if group["num_valid_match_instances"] is not None
    ]
    run_config = {
        "distance_line": DISTANCE_LINE,
        "distance_type": DISTANCE_TYPE,
        "fgw_lambda": float(args.fgw_lambda),
        "structure_mode": args.structure_mode,
        "feature_cost": args.feature_cost,
        "atom_penalty": float(args.atom_penalty),
        "thresholds": thresholds,
        "threshold_source": "explicit" if args.fgw_thresholds not in {None, "", "auto", "auto_quantile"} else "auto_quantile",
        "fgw_quantiles": _parse_float_list(args.fgw_quantiles),
        "skip_redundancy": bool(args.skip_redundancy),
        "run_ours": bool(args.run_ours),
        "run_fullgraph": bool(args.run_fullgraph),
        "selection_performed_in_eval": any_selection_in_eval,
        "candidate_set_preselected": all_groups_preselected,
        "selection_method": run_selection_method,
        "preselected_topk": int(args.preselected_topk),
        "require_preselected_topk": bool(args.require_preselected_topk),
        "preselected_candidate_audits": preselected_audits,
        "evaluation_row_unit": next(iter(row_units)) if len(row_units) == 1 else "mixed",
        "num_unique_parent_candidate_pairs": sum(
            int(group["num_unique_parent_candidate_pairs"]) for group in detail_groups
        ),
        "num_detail_rows": sum(int(group["num_detail_rows"]) for group in detail_groups),
        "num_valid_match_instances": sum(valid_match_counts) if valid_match_counts else None,
        "method_evaluation_audits": {
            str(group["method"]): {
                "candidate_set_preselected": bool(group["candidate_set_preselected"]),
                "selection_performed_in_eval": bool(group["selection_performed_in_eval"]),
                "selection_method": str(group["selection_method"]),
                "evaluation_row_unit": str(group["evaluation_row_unit"]),
                "num_unique_parent_candidate_pairs": int(group["num_unique_parent_candidate_pairs"]),
                "num_detail_rows": int(group["num_detail_rows"]),
                "num_valid_match_instances": group["num_valid_match_instances"],
            }
            for group in detail_groups
        },
        "max_parents": args.max_parents,
        "max_candidates": args.max_candidates,
        "cf_mode": args.cf_mode,
        "dataset_csv": args.dataset_csv,
        "ours_selected_path": args.ours_selected_path,
        "fullgraph_candidates_path": fullgraph_path,
        "fullgraph_method": fullgraph_method,
        "teacher_path": args.teacher_path,
        "molclr_root": args.molclr_root,
        "molclr_checkpoint": args.molclr_checkpoint,
    }
    _write_json(output / "run_config.json", run_config)
    cache_stats = provider.stats_dict()
    cache_stats["runtime_seconds"] = float(time.time() - started)
    _write_json(output / "cache_stats.json", cache_stats)
    print(f"[TASK_DONE] output_dir={output} runtime_seconds={cache_stats['runtime_seconds']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
