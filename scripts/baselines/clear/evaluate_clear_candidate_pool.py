#!/usr/bin/env python3
"""Evaluate a CLEAR candidate/action pool under the unified CCRCov contract.

CLEAR exports local counterfactual graph actions. This evaluator keeps the
official CLEAR fields as diagnostics, but final flip/drop metrics are populated
only from a project teacher prediction source. When the pool lacks SMILES or
full graph arrays needed by a teacher adapter, the script fails clearly unless
``--allow-action-only`` is set for cost-only smoke diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.ccrcov_distance_eval import CF_MODES, normalize_cf_mode  # noqa: E402
from src.eval.close_counterfactual_coverage import predict_with_teacher  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


DEFAULT_CANDIDATE_POOL = (
    "outputs/hpc/baselines/clear/ogbg_molhiv/candidate_pool/"
    "clear_ogbg_molhiv_candidate_pool.jsonl"
)
DEFAULT_TEACHER_PATH = "outputs/hpc/oracle/aids_rf_model.pkl"
DEFAULT_OUT_DIR = "outputs/hpc/baselines/clear/ogbg_molhiv/eval"

SMILES_ORIGINAL_FIELDS = ("original_smiles", "parent_smiles", "smiles")
SMILES_CF_FIELDS = ("cf_smiles", "counterfactual_smiles", "candidate_smiles", "action_smiles")
FULL_GRAPH_FIELDS = ("original_adj", "cf_adj", "original_x", "cf_x")
PRECOMPUTED_TEACHER_ORIG = ("teacher_original_pred", "teacher_original_pred_label", "teacher_pred_before")
PRECOMPUTED_TEACHER_CF = ("teacher_cf_pred", "teacher_cf_pred_label", "teacher_pred_after")
PRECOMPUTED_TEACHER_P_ORIG = ("teacher_original_p_label", "teacher_p_before", "teacher_original_prob_label")
PRECOMPUTED_TEACHER_P_CF = ("teacher_cf_p_label", "teacher_p_after", "teacher_cf_prob_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLEAR candidate/action pools with unified teacher-facing metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-pool", default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--dataset", default="ogbg_molhiv")
    parser.add_argument("--teacher-path", default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cf-mode", choices=CF_MODES, default="strict_flip")
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--top-k", default="1,5,10,20")
    parser.add_argument(
        "--thresholds",
        default="5,10,20,50,100,200",
        help="Cost thresholds for action-distance CCRCov summaries. Use values matching distance-method scale.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--deduplicate-by", choices=("none", "instance_index"), default="none")
    parser.add_argument("--distance-method", choices=("action", "ged", "molclr"), default="action")
    parser.add_argument("--rank-by", choices=("total_cost", "edge_cost", "input_order"), default="total_cost")
    parser.add_argument(
        "--allow-action-only",
        action="store_true",
        help=(
            "Allow cost-only diagnostics when no unified teacher prediction source is available. "
            "Final strict FlipRate/CFDrop/CCRCov will be null/zero and must not be reported as final."
        ),
    )
    parser.add_argument("--config", default=None, help="Accepted for Slurm/config compatibility; not used.")
    parser.add_argument("--set", action="append", default=[], help="Accepted for Slurm/config compatibility; not used.")
    return parser.parse_args()


def parse_csv_numbers(raw: str, *, as_int: bool = False) -> list[Any]:
    values: list[Any] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in {"inf", "infinity"}:
            values.append(math.inf)
        elif as_int:
            values.append(int(float(token)))
        else:
            values.append(float(token))
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                payload["_input_order"] = len(rows)
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else ("" if row.get(key) is None else row.get(key))
                    for key in fields
                }
            )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else "inf"
    return value


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def numeric_values(values: Iterable[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        number = as_float(value)
        if number is not None:
            clean.append(number)
    return clean


def mean(values: Iterable[Any]) -> float | None:
    clean = numeric_values(values)
    return float(statistics.mean(clean)) if clean else None


def median(values: Iterable[Any]) -> float | None:
    clean = numeric_values(values)
    return float(statistics.median(clean)) if clean else None


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def normalize_prob_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    probs: list[float] = []
    for item in value:
        number = as_float(item)
        if number is None:
            return None
        probs.append(number)
    return probs


def p_for_label(prob_vector: list[float] | None, label: int | None) -> float | None:
    if prob_vector is None or label is None or label < 0 or label >= len(prob_vector):
        return None
    return float(prob_vector[label])


def compute_action_distance(row: dict[str, Any]) -> tuple[float | None, str | None]:
    total_cost = as_float(row.get("total_cost"))
    if total_cost is not None:
        return total_cost, None
    edge_cost = as_float(row.get("edge_cost"))
    feature_cost = as_float(row.get("feature_l1_cost"))
    if edge_cost is not None and feature_cost is not None:
        return float(edge_cost + feature_cost), None
    if edge_cost is not None:
        return edge_cost, None
    return None, "missing_action_cost"


def has_full_graph_arrays(row: dict[str, Any]) -> bool:
    return all(name in row for name in FULL_GRAPH_FIELDS)


def has_smiles_pair(row: dict[str, Any]) -> bool:
    return first_present(row, SMILES_ORIGINAL_FIELDS) is not None and first_present(row, SMILES_CF_FIELDS) is not None


def has_precomputed_teacher(row: dict[str, Any]) -> bool:
    return first_present(row, PRECOMPUTED_TEACHER_ORIG) is not None and first_present(row, PRECOMPUTED_TEACHER_CF) is not None


def teacher_eval_from_precomputed(row: dict[str, Any], label: int | None) -> dict[str, Any]:
    pred_before = as_int(first_present(row, PRECOMPUTED_TEACHER_ORIG))
    pred_after = as_int(first_present(row, PRECOMPUTED_TEACHER_CF))
    p_before = as_float(first_present(row, PRECOMPUTED_TEACHER_P_ORIG))
    p_after = as_float(first_present(row, PRECOMPUTED_TEACHER_P_CF))
    return {
        "teacher_eval_ok": pred_before is not None and pred_after is not None,
        "teacher_eval_source": "precomputed_teacher_fields",
        "teacher_original_pred": pred_before,
        "teacher_cf_pred": pred_after,
        "teacher_p_before": p_before,
        "teacher_p_after": p_after,
        "teacher_flip": bool(pred_before != pred_after) if pred_before is not None and pred_after is not None else None,
        "cf_drop": (p_before - p_after) if p_before is not None and p_after is not None else None,
        "teacher_error": None,
        "label_used_for_p_label": label,
    }


def teacher_eval_from_smiles(
    row: dict[str, Any],
    *,
    teacher: TeacherSemanticScorer,
    label: int,
) -> dict[str, Any]:
    original_smiles = str(first_present(row, SMILES_ORIGINAL_FIELDS) or "")
    cf_smiles = str(first_present(row, SMILES_CF_FIELDS) or "")
    before = predict_with_teacher(teacher, original_smiles, label)
    after = predict_with_teacher(teacher, cf_smiles, label)
    if not before.get("ok"):
        return {
            "teacher_eval_ok": False,
            "teacher_eval_source": "smiles_teacher",
            "teacher_original_pred": None,
            "teacher_cf_pred": None,
            "teacher_p_before": None,
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": f"before_failed:{before.get('error')}",
            "label_used_for_p_label": label,
        }
    if not after.get("ok"):
        return {
            "teacher_eval_ok": False,
            "teacher_eval_source": "smiles_teacher",
            "teacher_original_pred": before.get("pred_label"),
            "teacher_cf_pred": None,
            "teacher_p_before": before.get("p_label"),
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": f"after_failed:{after.get('error')}",
            "label_used_for_p_label": label,
        }
    p_before = before.get("p_label")
    p_after = after.get("p_label")
    return {
        "teacher_eval_ok": True,
        "teacher_eval_source": "smiles_teacher",
        "teacher_original_pred": before.get("pred_label"),
        "teacher_cf_pred": after.get("pred_label"),
        "teacher_p_before": p_before,
        "teacher_p_after": p_after,
        "teacher_flip": bool(before.get("pred_label") != after.get("pred_label")),
        "cf_drop": (float(p_before) - float(p_after)) if p_before is not None and p_after is not None else None,
        "teacher_error": None,
        "label_used_for_p_label": label,
    }


def cf_condition(row: dict[str, Any], *, cf_mode: str, min_cf_drop: float) -> bool | None:
    mode = normalize_cf_mode(cf_mode)
    teacher_flip = as_bool(row.get("teacher_flip"))
    cf_drop = as_float(row.get("cf_drop"))
    drop_ok = cf_drop is not None and cf_drop >= float(min_cf_drop)
    if teacher_flip is None and mode in {"strict_flip", "drop_or_flip"} and not drop_ok:
        return None
    if mode == "strict_flip":
        return bool(teacher_flip)
    if mode == "drop_or_flip":
        return bool(teacher_flip or drop_ok)
    if mode == "drop_only":
        return bool(drop_ok)
    raise ValueError(f"Unsupported cf_mode={mode}")


def evaluate_candidate(
    row: dict[str, Any],
    *,
    teacher: TeacherSemanticScorer | None,
    distance_method: str,
) -> dict[str, Any]:
    label = as_int(row.get("original_label"))
    if label is None:
        label = as_int(row.get("label"))

    distance_error: str | None = None
    if distance_method == "action":
        distance, distance_error = compute_action_distance(row)
    elif distance_method in {"ged", "molclr"}:
        if not has_smiles_pair(row) and not has_full_graph_arrays(row):
            raise RuntimeError(
                f"distance-method={distance_method} requires original/counterfactual graph content. "
                "The current CLEAR candidate pool does not contain full graph arrays by default. "
                "Regenerate it with: scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py "
                "--include-full-graphs, then add the corresponding graph-distance adapter."
            )
        raise RuntimeError(
            f"distance-method={distance_method} is reserved for a future CLEAR graph-distance adapter. "
            "Use --distance-method action for the current CLEAR action pool."
        )
    else:
        raise ValueError(f"Unsupported distance_method={distance_method}")

    teacher_eval: dict[str, Any]
    if has_precomputed_teacher(row):
        teacher_eval = teacher_eval_from_precomputed(row, label)
    elif has_smiles_pair(row) and teacher is not None and teacher.available and label is not None:
        teacher_eval = teacher_eval_from_smiles(row, teacher=teacher, label=label)
    else:
        teacher_eval = {
            "teacher_eval_ok": False,
            "teacher_eval_source": "unavailable",
            "teacher_original_pred": None,
            "teacher_cf_pred": None,
            "teacher_p_before": None,
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": (
                "no_unified_teacher_prediction_source: candidate lacks original_smiles/cf_smiles "
                "or precomputed teacher fields; default CLEAR pool also omits full graph arrays"
            ),
            "label_used_for_p_label": label,
        }

    candidate = {
        "candidate_id": row.get("candidate_id"),
        "source": row.get("source", "CLEAR"),
        "dataset": row.get("dataset"),
        "exp_id": row.get("exp_id"),
        "split": row.get("split"),
        "instance_index": row.get("instance_index"),
        "original_label": label,
        "target_cf_label": row.get("target_cf_label"),
        "official_original_pred_label": row.get("official_original_pred_label"),
        "official_cf_pred_label": row.get("official_cf_pred_label"),
        "official_flip": row.get("official_flip"),
        "official_target_success": row.get("official_target_success"),
        "official_original_correct": row.get("official_original_correct"),
        "distance_method": distance_method,
        "distance": distance,
        "distance_ok": distance is not None,
        "distance_error": distance_error,
        "edge_cost": row.get("edge_cost"),
        "total_cost": row.get("total_cost"),
        "num_edge_added": row.get("num_edge_added"),
        "num_edge_deleted": row.get("num_edge_deleted"),
        "num_edge_changed": row.get("num_edge_changed"),
        "action_edges_added": row.get("action_edges_added"),
        "action_edges_deleted": row.get("action_edges_deleted"),
        "feature_l1_cost": row.get("feature_l1_cost"),
        "feature_l2_cost": row.get("feature_l2_cost"),
        "num_node_feature_changed": row.get("num_node_feature_changed"),
        "changed_node_indices": row.get("changed_node_indices"),
        **teacher_eval,
        "has_smiles_pair": has_smiles_pair(row),
        "has_full_graph_arrays": has_full_graph_arrays(row),
    }
    return candidate


def load_and_prepare_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.candidate_pool).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CLEAR candidate pool not found: {path}")
    raw_rows = read_jsonl(path)
    if args.max_candidates is not None:
        raw_rows = raw_rows[: int(args.max_candidates)]

    if args.deduplicate_by == "instance_index":
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in raw_rows:
            key = str(row.get("instance_index"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        raw_rows = deduped
    return raw_rows


def rank_candidates(rows: list[dict[str, Any]], rank_by: str) -> list[dict[str, Any]]:
    if rank_by == "input_order":
        return sorted(rows, key=lambda row: int(row.get("_eval_order", row.get("_input_order", 0)) or 0))
    if rank_by == "edge_cost":
        return sorted(rows, key=lambda row: (as_float(row.get("edge_cost")) is None, as_float(row.get("edge_cost")) or 1e18))
    return sorted(rows, key=lambda row: (as_float(row.get("distance")) is None, as_float(row.get("distance")) or 1e18))


def action_signature(row: dict[str, Any]) -> set[str]:
    signature: set[str] = set()
    for edge in row.get("action_edges_added") or []:
        signature.add(f"+e:{edge}")
    for edge in row.get("action_edges_deleted") or []:
        signature.add(f"-e:{edge}")
    for node in row.get("changed_node_indices") or []:
        signature.add(f"xf:{node}")
    return signature


def structural_redundancy(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return 0.0
    signatures = [action_signature(row) for row in rows]
    total = 0.0
    count = 0
    for i, left in enumerate(signatures):
        for right in signatures[i + 1 :]:
            union = len(left | right)
            total += (len(left & right) / union) if union else 0.0
            count += 1
    return float(total / count) if count else 0.0


def coverage_redundancy(rows: list[dict[str, Any]], *, threshold: float, cf_mode: str, min_cf_drop: float) -> float:
    cover_sets: list[set[str]] = []
    for row in rows:
        distance = as_float(row.get("distance"))
        if distance is None or distance > threshold:
            continue
        condition = cf_condition(row, cf_mode=cf_mode, min_cf_drop=min_cf_drop)
        if condition is not True:
            continue
        instance = row.get("instance_index")
        if instance is not None:
            cover_sets.append({str(instance)})
    if len(cover_sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, left in enumerate(cover_sets):
        for right in cover_sets[i + 1 :]:
            union = len(left | right)
            total += (len(left & right) / union) if union else 0.0
            count += 1
    return float(total / count) if count else 0.0


def build_summary(
    evaluated_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    *,
    top_ks: list[int],
    thresholds: list[float],
    cf_mode: str,
    min_cf_drop: float,
    dataset: str,
    distance_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_ids = {str(row.get("instance_index")) for row in evaluated_rows if row.get("instance_index") is not None}
    num_parents = len(parent_ids)
    teacher_eval_count = sum(1 for row in evaluated_rows if row.get("teacher_eval_ok") is True)
    teacher_available = teacher_eval_count > 0
    threshold_rows: list[dict[str, Any]] = []

    for k in top_ks:
        top_rows = ranked_rows[: min(int(k), len(ranked_rows))]
        for threshold in thresholds:
            close_rows = [
                row
                for row in top_rows
                if as_float(row.get("distance")) is not None and float(row["distance"]) <= float(threshold)
            ]
            close_parent_ids = {str(row.get("instance_index")) for row in close_rows if row.get("instance_index") is not None}
            cf_rows = [
                row
                for row in close_rows
                if cf_condition(row, cf_mode=cf_mode, min_cf_drop=min_cf_drop) is True
            ]
            cf_parent_ids = {str(row.get("instance_index")) for row in cf_rows if row.get("instance_index") is not None}
            threshold_rows.append(
                {
                    "method": "CLEAR",
                    "dataset": dataset,
                    "distance_method": distance_method,
                    "cf_mode": cf_mode,
                    "min_cf_drop": float(min_cf_drop),
                    "top_k": int(k),
                    "threshold": float(threshold) if math.isfinite(float(threshold)) else "inf",
                    "num_parents": num_parents,
                    "num_candidates": len(evaluated_rows),
                    "num_selected_candidates": len(top_rows),
                    "num_close_only_covered": len(close_parent_ids),
                    "close_only_coverage": rate(len(close_parent_ids), num_parents),
                    "num_close_cf_covered": len(cf_parent_ids),
                    "close_cf_coverage": rate(len(cf_parent_ids), num_parents) if teacher_available else None,
                    "SuppCov": rate(len(close_parent_ids), num_parents),
                    "CCRCov@K": rate(len(cf_parent_ids), num_parents) if teacher_available else None,
                    "FlipRate": mean(1.0 if row.get("teacher_flip") is True else 0.0 for row in close_rows if row.get("teacher_eval_ok") is True),
                    "CFDrop": mean(row.get("cf_drop") for row in cf_rows),
                    "CostMean": mean(row.get("distance") for row in cf_rows) if cf_rows else mean(row.get("distance") for row in close_rows),
                    "CostMedian": median(row.get("distance") for row in cf_rows) if cf_rows else median(row.get("distance") for row in close_rows),
                    "StructRed": structural_redundancy(top_rows),
                    "CovRed": coverage_redundancy(top_rows, threshold=float(threshold), cf_mode=cf_mode, min_cf_drop=min_cf_drop),
                    "ValidRate": rate(sum(1 for row in top_rows if row.get("distance_ok")), len(top_rows)),
                    "teacher_eval_ok_rate": rate(sum(1 for row in top_rows if row.get("teacher_eval_ok")), len(top_rows)),
                    "official_flip_rate_selected": mean(1.0 if as_bool(row.get("official_flip")) else 0.0 for row in top_rows),
                    "mean_edge_changed_selected": mean(row.get("num_edge_changed") for row in top_rows),
                    "mean_feature_l1_selected": mean(row.get("feature_l1_cost") for row in top_rows),
                }
            )

    overall = {
        "method": "CLEAR",
        "dataset": dataset,
        "distance_method": distance_method,
        "cf_mode": cf_mode,
        "min_cf_drop": float(min_cf_drop),
        "num_candidates": len(evaluated_rows),
        "num_parents": num_parents,
        "teacher_eval_count": teacher_eval_count,
        "teacher_eval_ok_rate": rate(teacher_eval_count, len(evaluated_rows)),
        "teacher_available_for_final_metrics": teacher_available,
        "official_flip_count": sum(1 for row in evaluated_rows if as_bool(row.get("official_flip")) is True),
        "official_flip_rate": mean(1.0 if as_bool(row.get("official_flip")) else 0.0 for row in evaluated_rows),
        "mean_distance": mean(row.get("distance") for row in evaluated_rows),
        "median_distance": median(row.get("distance") for row in evaluated_rows),
        "mean_edge_changed": mean(row.get("num_edge_changed") for row in evaluated_rows),
        "mean_feature_l1_cost": mean(row.get("feature_l1_cost") for row in evaluated_rows),
        "note": (
            "Final FlipRate/CFDrop/CCRCov use unified teacher fields only. "
            "CLEAR official_flip is retained as a diagnostic and never used as final strict flip."
        ),
    }
    return threshold_rows, overall


def write_report(path: Path, *, overall: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CLEAR Unified Candidate Pool Evaluation",
        "",
        f"- dataset: {overall.get('dataset')}",
        f"- distance_method: {overall.get('distance_method')}",
        f"- CF mode: {overall.get('cf_mode')}",
        f"- num_candidates: {overall.get('num_candidates')}",
        f"- num_parents: {overall.get('num_parents')}",
        f"- teacher_eval_ok_rate: {overall.get('teacher_eval_ok_rate')}",
        f"- official_flip_rate_diagnostic_only: {overall.get('official_flip_rate')}",
        "",
        "CLEAR official flip/validity fields are diagnostics only. Final FlipRate, CFDrop, and CCRCov must use the unified teacher/oracle.",
        "",
        "## Top-K Summary",
        "",
        "| top_k | threshold | SuppCov | CCRCov@K | FlipRate | CFDrop | CostMean | teacher_eval_ok_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_rows:
        lines.append(
            "| {top_k} | {threshold} | {SuppCov} | {ccrcov} | {flip} | {drop} | {cost} | {teacher} |".format(
                top_k=row.get("top_k"),
                threshold=row.get("threshold"),
                SuppCov=_fmt(row.get("SuppCov")),
                ccrcov=_fmt(row.get("CCRCov@K")),
                flip=_fmt(row.get("FlipRate")),
                drop=_fmt(row.get("CFDrop")),
                cost=_fmt(row.get("CostMean")),
                teacher=_fmt(row.get("teacher_eval_ok_rate")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def main() -> int:
    args = parse_args()
    candidate_pool = Path(args.candidate_pool).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cf_mode = normalize_cf_mode(args.cf_mode)
    top_ks = parse_csv_numbers(args.top_k, as_int=True)
    thresholds = parse_csv_numbers(args.thresholds)
    if not top_ks:
        raise ValueError("--top-k must contain at least one K value")
    if not thresholds:
        thresholds = [math.inf]

    print("[CLEAR_EVAL_CONFIG]")
    print(f"candidate_pool={candidate_pool}")
    print(f"dataset={args.dataset}")
    print(f"teacher_path={args.teacher_path}")
    print(f"out_dir={out_dir}")
    print(f"cf_mode={cf_mode}")
    print(f"top_k={top_ks}")
    print(f"thresholds={thresholds}")
    print(f"distance_method={args.distance_method}")
    print(f"allow_action_only={args.allow_action_only}")

    raw_rows = load_and_prepare_candidates(args)
    if not raw_rows:
        raise ValueError(f"No candidates found in {candidate_pool}")

    has_any_smiles = any(has_smiles_pair(row) for row in raw_rows)
    has_any_full_graph = any(has_full_graph_arrays(row) for row in raw_rows)
    has_any_precomputed_teacher = any(has_precomputed_teacher(row) for row in raw_rows)

    teacher = None
    if has_any_smiles and args.teacher_path:
        teacher = TeacherSemanticScorer(args.teacher_path)

    if args.distance_method != "action" and not (has_any_smiles or has_any_full_graph):
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] CLEAR candidate pool lacks full graph arrays or SMILES required for non-action "
            "distance evaluation. Regenerate the pool with --include-full-graphs, or use --distance-method action."
        )

    if not has_any_smiles and not has_any_precomputed_teacher and not args.allow_action_only:
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] Unified teacher/oracle evaluation cannot run on this CLEAR candidate pool because it lacks "
            "original_smiles/cf_smiles and precomputed teacher_* fields. The default conversion omits full graph arrays, "
            "so no graph-teacher adapter can reconstruct predictions from this JSONL. Re-run conversion with "
            "--include-full-graphs and provide a graph-teacher adapter, or use --allow-action-only for diagnostic "
            "cost/SuppCov summaries only. CLEAR official_flip is not used as final flip."
        )

    evaluated: list[dict[str, Any]] = []
    for eval_order, row in enumerate(raw_rows):
        result = evaluate_candidate(row, teacher=teacher, distance_method=args.distance_method)
        result["_eval_order"] = eval_order
        evaluated.append(result)

    if not args.allow_action_only and not any(row.get("teacher_eval_ok") for row in evaluated):
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] No candidate received a successful unified teacher evaluation. This usually means the pool has no SMILES "
            "or the supplied teacher cannot score the candidate representation. Final CLEAR metrics are not available."
        )

    ranked = rank_candidates(evaluated, args.rank_by)
    threshold_rows, overall = build_summary(
        evaluated,
        ranked,
        top_ks=[int(k) for k in top_ks],
        thresholds=[float(t) for t in thresholds],
        cf_mode=cf_mode,
        min_cf_drop=float(args.min_cf_drop),
        dataset=args.dataset,
        distance_method=args.distance_method,
    )
    overall.update(
        {
            "candidate_pool": str(candidate_pool),
            "out_dir": str(out_dir),
            "rank_by": args.rank_by,
            "deduplicate_by": args.deduplicate_by,
            "max_candidates": args.max_candidates,
            "allow_action_only": args.allow_action_only,
            "has_any_smiles_pair": has_any_smiles,
            "has_any_full_graph_arrays": has_any_full_graph,
            "has_any_precomputed_teacher": has_any_precomputed_teacher,
        }
    )

    per_candidate_path = out_dir / "per_candidate_eval.jsonl"
    threshold_summary_path = out_dir / "threshold_summary.csv"
    summary_json_path = out_dir / "summary.json"
    summary_csv_path = out_dir / "summary.csv"
    report_path = out_dir / "report.md"
    write_jsonl(per_candidate_path, evaluated)
    write_csv(threshold_summary_path, threshold_rows)
    write_json(summary_json_path, overall)
    write_csv(summary_csv_path, [overall])
    write_report(report_path, overall=overall, threshold_rows=threshold_rows)

    print("[CLEAR_EVAL_PREVIEW]")
    for row in evaluated[:3]:
        print(
            json.dumps(
                {
                    "candidate_id": row.get("candidate_id"),
                    "instance_index": row.get("instance_index"),
                    "distance": row.get("distance"),
                    "official_flip": row.get("official_flip"),
                    "teacher_eval_ok": row.get("teacher_eval_ok"),
                    "teacher_flip": row.get("teacher_flip"),
                    "cf_drop": row.get("cf_drop"),
                },
                sort_keys=True,
            )
        )
    print("[CLEAR_EVAL_DONE]")
    print(f"per_candidate_eval={per_candidate_path}")
    print(f"threshold_summary={threshold_summary_path}")
    print(f"summary_json={summary_json_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
