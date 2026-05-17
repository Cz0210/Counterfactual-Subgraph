"""Teacher/reward audit helpers for decoded PPO candidate pools."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from src.chem import is_parent_substructure
from src.rewards.teacher_semantic import TeacherSemanticScorer
from src.utils.io import ensure_directory, read_jsonl

try:  # pragma: no cover - optional dependency
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover - optional dependency
    pearsonr = None
    spearmanr = None


_SIZE_BUCKETS = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.1", 0.05, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0000001),
)
_DEFAULT_SMILES_COLUMNS = ("parent_smiles", "smiles", "SMILES", "prompt", "instruction", "input")
_DEFAULT_LABEL_COLUMNS = ("label", "original_label", "y", "HIV_active", "HIV", "class")
_PROMPT_SMILES_PATTERNS = (
    re.compile(
        r"PARENT_SMILES:\s*(?P<smiles>.+?)(?:\n\s*\n\[Output\]|\n\s*\[Output\]|\Z)",
        flags=re.DOTALL,
    ),
    re.compile(
        r"MOLECULE_SMILES:\s*(?P<smiles>.+?)(?:\nFRAGMENT_SMILES:|\Z)",
        flags=re.DOTALL,
    ),
    re.compile(
        r"SMILES:\s*(?P<smiles>.+?)(?:\nReturn ONE connected substructure|\n\n\[Assistant\]|\Z)",
        flags=re.DOTALL,
    ),
)
_SKIP_CF_REASONS = frozenset(
    {
        "invalid_or_not_substructure",
        "not_direct_substructure",
        "invalid_generation_too_long",
        "full_parent_fragment",
        "near_parent_fragment",
        "tiny_residual_fragment",
        "tiny_fragment_hard_fail",
        "empty_response",
    }
)


@dataclass(frozen=True, slots=True)
class RewardTeacherAuditConfig:
    """Execution knobs for teacher/reward audits."""

    label_col: str = "label"
    smiles_col: str = "parent_smiles"
    sim_sample_size: int = 5000


@dataclass(frozen=True, slots=True)
class DatasetParentRecord:
    """One dataset parent molecule row adapted for teacher reliability scoring."""

    record_index: int
    parent_smiles: str
    label: int
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NormalizedPoolRow:
    """One candidate-pool row normalized for reward/teacher auditing."""

    record_index: int
    parent_smiles: str | None
    label: int | None
    reward_total: float | None
    cf_drop: float | None
    cf_flip: bool
    p_before: float | None
    p_after: float | None
    counterfactual_reason: str | None
    counterfactual_called: bool
    parent_without_fragment_smiles: str | None
    residual_sanitize_failed: bool
    invalid_or_not_substructure: bool
    direct_substructure: bool
    final_substructure: bool
    projection_used: bool
    atom_ratio: float | None
    fragment_atom_count: int | None
    parse_ok: bool
    valid: bool
    core_unusable: bool
    projection_score: float | None
    full_parent: bool
    failure_tag: str | None
    invalid_detail: str | None
    raw_payload: dict[str, Any]


def _coalesce(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    try:
        return int(numeric)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    numeric = _as_float(value)
    if numeric is not None:
        return bool(numeric)
    return None


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _coerce_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value) if value in (0, 1) else None
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "positive", "pos", "active", "hiv_active"}:
        return 1
    if text in {"0", "false", "no", "negative", "neg", "inactive", "hiv_inactive"}:
        return 0
    numeric = _as_int(value)
    if numeric in (0, 1):
        return numeric
    return None


def _distribution(counter: Counter[str], denominator: int) -> list[dict[str, Any]]:
    return [
        {
            "value": key,
            "count": int(count),
            "rate": _round_or_none(_safe_rate(int(count), denominator)),
        }
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _resolve_dataset_columns(
    field_names: list[str],
    *,
    preferred_smiles_col: str,
    preferred_label_col: str,
) -> tuple[str, str]:
    available = {name.strip().lower(): name for name in field_names if str(name).strip()}

    def _resolve_one(preferred: str, fallbacks: tuple[str, ...], kind: str) -> str:
        preferred_key = str(preferred or "").strip().lower()
        if preferred_key and preferred_key in available:
            return available[preferred_key]
        for candidate in fallbacks:
            candidate_key = candidate.strip().lower()
            if candidate_key in available:
                return available[candidate_key]
        raise ValueError(
            f"Could not resolve {kind} column. preferred={preferred!r} available={field_names!r}"
        )

    return (
        _resolve_one(preferred_smiles_col, _DEFAULT_SMILES_COLUMNS, "smiles"),
        _resolve_one(preferred_label_col, _DEFAULT_LABEL_COLUMNS, "label"),
    )


def _extract_parent_smiles_from_prompt(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in _PROMPT_SMILES_PATTERNS:
        match = pattern.search(text)
        if match:
            smiles = str(match.group("smiles") or "").strip()
            if smiles:
                return smiles
    if "\n" not in text and " " not in text:
        return text
    return None


def _load_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def load_dataset_parent_records(
    dataset_path: str | Path,
    *,
    label_col: str,
    smiles_col: str,
) -> tuple[list[DatasetParentRecord], dict[str, Any]]:
    """Load one CSV/JSONL dataset and normalize parent rows for teacher audit."""

    path = Path(dataset_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw_rows, field_names = _load_csv_rows(path)
    elif suffix in {".jsonl", ".json"}:
        raw_rows = read_jsonl(path)
        field_names = list(raw_rows[0].keys()) if raw_rows else []
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    if not raw_rows:
        raise ValueError(f"No dataset rows found in {path}")

    resolved_smiles_col, resolved_label_col = _resolve_dataset_columns(
        field_names,
        preferred_smiles_col=smiles_col,
        preferred_label_col=label_col,
    )

    usable_records: list[DatasetParentRecord] = []
    dropped_counts: Counter[str] = Counter()
    for index, row in enumerate(raw_rows):
        parent_smiles = _extract_parent_smiles_from_prompt(
            _normalize_text(row.get(resolved_smiles_col))
        )
        if parent_smiles is None:
            for candidate_col in _DEFAULT_SMILES_COLUMNS:
                parent_smiles = _extract_parent_smiles_from_prompt(
                    _normalize_text(row.get(candidate_col))
                )
                if parent_smiles:
                    break
        if not parent_smiles:
            dropped_counts["missing_parent_smiles"] += 1
            continue

        label = _coerce_binary_label(row.get(resolved_label_col))
        if label is None:
            for candidate_col in _DEFAULT_LABEL_COLUMNS:
                label = _coerce_binary_label(row.get(candidate_col))
                if label is not None:
                    break
        if label is None:
            dropped_counts["missing_or_invalid_label"] += 1
            continue

        usable_records.append(
            DatasetParentRecord(
                record_index=index,
                parent_smiles=parent_smiles,
                label=label,
                raw_payload=dict(row),
            )
        )

    metadata = {
        "dataset_path": str(path),
        "dataset_format": suffix.lstrip("."),
        "input_row_count": len(raw_rows),
        "usable_row_count": len(usable_records),
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "resolved_smiles_col": resolved_smiles_col,
        "resolved_label_col": resolved_label_col,
        "requested_smiles_col": smiles_col,
        "requested_label_col": label_col,
    }
    return usable_records, metadata


def _resolve_final_fragment(row: dict[str, Any]) -> str | None:
    explicit_final = _normalize_text(_coalesce(row, "final_fragment", "final_fragment_smiles"))
    if explicit_final:
        return explicit_final
    projection_used = bool(_as_bool(_coalesce(row, "used_projected_subgraph_for_reward")))
    projection_success = bool(_as_bool(_coalesce(row, "projection_success")))
    projected_fragment = _normalize_text(
        _coalesce(row, "projected_fragment", "projected_fragment_smiles")
    )
    if (projection_used or projection_success) and projected_fragment:
        return projected_fragment
    return _normalize_text(
        _coalesce(
            row,
            "core_fragment",
            "core_fragment_smiles",
            "raw_fragment",
            "fragment",
            "fragment_smiles",
            "generated_fragment",
        )
    )


def _infer_parse_ok(row: dict[str, Any]) -> bool:
    explicit = _as_bool(_coalesce(row, "parse_ok", "raw_parse_ok", "core_parse_ok"))
    if explicit is not None:
        return explicit
    failure_tag = str(_coalesce(row, "failure_tag") or "")
    invalid_detail = str(_coalesce(row, "invalid_detail") or "")
    return not (
        failure_tag.startswith("parse_failed")
        or "parse_failed" in invalid_detail
        or "unclosed_ring" in invalid_detail
    )


def _infer_valid(row: dict[str, Any]) -> bool:
    explicit = _as_bool(_coalesce(row, "valid", "valid_smiles", "sanitize_ok"))
    if explicit is not None:
        return explicit
    invalid_detail = str(_coalesce(row, "invalid_detail") or "").lower()
    failure_tag = str(_coalesce(row, "failure_tag") or "").lower()
    return not ("sanitize" in invalid_detail or "sanitize" in failure_tag)


def _infer_core_unusable(row: dict[str, Any]) -> bool:
    explicit = _as_bool(_coalesce(row, "core_unusable"))
    if explicit is not None:
        return explicit
    failure_tag = str(_coalesce(row, "failure_tag") or "").lower()
    invalid_detail = str(_coalesce(row, "invalid_detail") or "").lower()
    return (
        failure_tag == "parse_ok_but_core_unusable"
        or "core_unusable" in invalid_detail
        or "unusable_after" in invalid_detail
    )


def _infer_direct_substructure(row: dict[str, Any]) -> bool:
    explicit = _as_bool(_coalesce(row, "direct_substructure", "direct_substructure_success"))
    return bool(explicit) if explicit is not None else False


def _infer_projection_used(row: dict[str, Any], direct_substructure: bool) -> bool:
    explicit = _as_bool(_coalesce(row, "projection_used", "used_projected_subgraph_for_reward"))
    if explicit is not None:
        return explicit
    projection_success = bool(_as_bool(_coalesce(row, "projection_success")))
    projected_fragment = _normalize_text(
        _coalesce(row, "projected_fragment", "projected_fragment_smiles")
    )
    return bool(projection_success and projected_fragment and not direct_substructure)


def _infer_final_substructure(
    row: dict[str, Any],
    *,
    parent_smiles: str | None,
    final_fragment: str | None,
    direct_substructure: bool,
    projection_used: bool,
) -> bool:
    explicit = _as_bool(_coalesce(row, "final_substructure"))
    if explicit is not None:
        return explicit
    if parent_smiles and final_fragment:
        try:
            return bool(is_parent_substructure(parent_smiles, final_fragment))
        except Exception:
            return bool(direct_substructure or projection_used)
    return bool(direct_substructure or projection_used)


def normalize_candidate_pool_row(record_index: int, row: dict[str, Any]) -> NormalizedPoolRow:
    """Normalize one candidate-pool row into the audit contract."""

    parent_smiles = _normalize_text(_coalesce(row, "parent_smiles"))
    label = _coerce_binary_label(_coalesce(row, "label", "original_label"))
    reward_total = _as_float(_coalesce(row, "reward_total", "total"))
    cf_drop = _as_float(_coalesce(row, "cf_drop", "counterfactual_drop", "teacher_cf_drop"))
    cf_flip = bool(_as_bool(_coalesce(row, "cf_flip", "counterfactual_flip")))
    p_before = _as_float(_coalesce(row, "p_before", "teacher_p_before"))
    p_after = _as_float(_coalesce(row, "p_after", "teacher_p_after"))
    counterfactual_reason = _normalize_text(
        _coalesce(row, "counterfactual_reason", "cf_reward_skipped_reason")
    )
    parent_without_fragment_smiles = _normalize_text(_coalesce(row, "parent_without_fragment_smiles"))
    direct_substructure = _infer_direct_substructure(row)
    projection_used = _infer_projection_used(row, direct_substructure)
    final_fragment = _resolve_final_fragment(row)
    final_substructure = _infer_final_substructure(
        row,
        parent_smiles=parent_smiles,
        final_fragment=final_fragment,
        direct_substructure=direct_substructure,
        projection_used=projection_used,
    )
    parse_ok = _infer_parse_ok(row)
    valid = _infer_valid(row)
    core_unusable = _infer_core_unusable(row)
    failure_tag = _normalize_text(_coalesce(row, "failure_tag"))
    invalid_detail = _normalize_text(_coalesce(row, "invalid_detail"))
    projection_score = _as_float(_coalesce(row, "projection_score"))
    fragment_atom_count = _as_int(
        _coalesce(row, "fragment_atom_count", "final_fragment_atom_count", "atom_count")
    )
    atom_ratio = _as_float(_coalesce(row, "atom_ratio", "final_fragment_atom_ratio"))
    full_parent = bool(_as_bool(_coalesce(row, "full_parent")))
    if not full_parent and atom_ratio is not None and atom_ratio >= 0.999999:
        full_parent = True
    explicit_called = _as_bool(
        _coalesce(row, "counterfactual_called", "counterfactual_teacher_called")
    )
    if explicit_called is not None:
        counterfactual_called = explicit_called
    else:
        counterfactual_called = bool(
            p_before is not None
            or p_after is not None
            or cf_drop is not None
            or parent_without_fragment_smiles
        )
    residual_sanitize_failed = bool(
        counterfactual_reason and "residual_sanitize_failed" in counterfactual_reason
    )
    invalid_or_not_substructure = bool(
        counterfactual_reason == "invalid_or_not_substructure"
        or (
            not counterfactual_called
            and (
                (failure_tag or "").startswith("parse_failed")
                or failure_tag == "parse_ok_but_not_direct_substructure"
                or failure_tag == "parse_ok_but_core_unusable"
                or core_unusable
            )
        )
    )
    return NormalizedPoolRow(
        record_index=record_index,
        parent_smiles=parent_smiles,
        label=label,
        reward_total=reward_total,
        cf_drop=cf_drop,
        cf_flip=cf_flip,
        p_before=p_before,
        p_after=p_after,
        counterfactual_reason=counterfactual_reason,
        counterfactual_called=counterfactual_called,
        parent_without_fragment_smiles=parent_without_fragment_smiles,
        residual_sanitize_failed=residual_sanitize_failed,
        invalid_or_not_substructure=invalid_or_not_substructure,
        direct_substructure=direct_substructure,
        final_substructure=final_substructure,
        projection_used=projection_used,
        atom_ratio=atom_ratio,
        fragment_atom_count=fragment_atom_count,
        parse_ok=parse_ok,
        valid=valid,
        core_unusable=core_unusable,
        projection_score=projection_score,
        full_parent=full_parent,
        failure_tag=failure_tag,
        invalid_detail=invalid_detail,
        raw_payload=dict(row),
    )


def load_candidate_pool_rows(candidate_pool_path: str | Path) -> list[NormalizedPoolRow]:
    rows = read_jsonl(candidate_pool_path)
    return [normalize_candidate_pool_row(index, row) for index, row in enumerate(rows)]


def _summarize_probability_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _round_or_none(_safe_mean(values)),
        "median": _round_or_none(_safe_median(values)),
        "p25": _round_or_none(_safe_quantile(values, 0.25)),
        "p75": _round_or_none(_safe_quantile(values, 0.75)),
    }


def audit_teacher_parent_reliability(
    dataset_records: list[DatasetParentRecord],
    *,
    teacher_scorer: TeacherSemanticScorer,
) -> dict[str, Any]:
    """Audit teacher reliability on original dataset parents."""

    teacher_reason_counter: Counter[str] = Counter()
    scored_rows: list[dict[str, Any]] = []
    for record in dataset_records:
        result = teacher_scorer.score_smiles(record.parent_smiles, label=record.label)
        teacher_prob = _as_float(result.get("teacher_prob"))
        teacher_pred = _coerce_binary_label(result.get("teacher_label"))
        teacher_ok = bool(result.get("teacher_result_ok"))
        teacher_reason = str(result.get("teacher_reason") or "")
        teacher_reason_counter[teacher_reason or "<empty>"] += 1
        scored_rows.append(
            {
                "label": record.label,
                "teacher_prob": teacher_prob,
                "teacher_pred": teacher_pred,
                "teacher_result_ok": teacher_ok,
                "teacher_reason": teacher_reason,
                "correct": bool(teacher_ok and teacher_pred == record.label),
            }
        )

    def _build_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        probability_values = [
            float(row["teacher_prob"])
            for row in rows
            if row["teacher_prob"] is not None
        ]
        low_confidence_count = sum(
            1 for row in rows if row["teacher_prob"] is not None and float(row["teacher_prob"]) < 0.5
        )
        very_low_confidence_count = sum(
            1 for row in rows if row["teacher_prob"] is not None and float(row["teacher_prob"]) < 0.2
        )
        return {
            "num_total": total,
            "teacher_result_ok_rate": _round_or_none(
                _safe_rate(sum(1 for row in rows if row["teacher_result_ok"]), total)
            ),
            "teacher_correct_rate": _round_or_none(
                _safe_rate(sum(1 for row in rows if row["correct"]), total)
            ),
            "p_label_mean": _round_or_none(_safe_mean(probability_values)),
            "p_label_median": _round_or_none(_safe_median(probability_values)),
            "p_label_p25": _round_or_none(_safe_quantile(probability_values, 0.25)),
            "p_label_p75": _round_or_none(_safe_quantile(probability_values, 0.75)),
            "low_confidence_count": low_confidence_count,
            "very_low_confidence_count": very_low_confidence_count,
            "low_confidence_rate": _round_or_none(_safe_rate(low_confidence_count, total)),
            "very_low_confidence_rate": _round_or_none(
                _safe_rate(very_low_confidence_count, total)
            ),
        }

    by_label: dict[str, Any] = {}
    for label in sorted({record.label for record in dataset_records}):
        group_rows = [row for row in scored_rows if row["label"] == label]
        by_label[str(label)] = _build_group(group_rows)

    return {
        "teacher_path": str(teacher_scorer.teacher_path) if teacher_scorer.teacher_path else None,
        "teacher_available": bool(teacher_scorer.available),
        "teacher_format": teacher_scorer.teacher_format,
        "teacher_availability_reason": teacher_scorer.availability_reason,
        "num_total": len(dataset_records),
        **_build_group(scored_rows),
        "teacher_reason_distribution": _distribution(
            teacher_reason_counter,
            len(scored_rows),
        ),
        "by_label": by_label,
    }


def _render_teacher_parent_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Teacher Parent Reliability Audit",
        f"- teacher_available: {summary['teacher_available']}",
        f"- teacher_format: {summary.get('teacher_format') or 'n/a'}",
        f"- teacher_availability_reason: {summary.get('teacher_availability_reason') or 'n/a'}",
        f"- num_total: {summary['num_total']}",
        f"- teacher_correct_rate: {summary['teacher_correct_rate']}",
        f"- p_label_mean: {summary['p_label_mean']}",
        f"- p_label_median: {summary['p_label_median']}",
        f"- p_label_p25: {summary['p_label_p25']}",
        f"- p_label_p75: {summary['p_label_p75']}",
        f"- low_confidence_count: {summary['low_confidence_count']}",
        f"- very_low_confidence_count: {summary['very_low_confidence_count']}",
        "",
        "By Label",
    ]
    for label, label_summary in sorted(summary.get("by_label", {}).items()):
        lines.append(
            f"- label={label}: num_total={label_summary['num_total']} "
            f"teacher_correct_rate={label_summary['teacher_correct_rate']} "
            f"p_label_mean={label_summary['p_label_mean']} "
            f"low_confidence_rate={label_summary['low_confidence_rate']} "
            f"very_low_confidence_rate={label_summary['very_low_confidence_rate']}"
        )
    return "\n".join(lines).strip() + "\n"


def audit_candidate_pool_oracle_validity(rows: list[NormalizedPoolRow]) -> dict[str, Any]:
    """Audit counterfactual-oracle coverage and failure modes from candidate_pool."""

    total = len(rows)
    called_rows = [row for row in rows if row.counterfactual_called]
    skipped_rows = [row for row in rows if not row.counterfactual_called]
    p_before_values = [row.p_before for row in called_rows if row.p_before is not None]
    p_after_values = [row.p_after for row in called_rows if row.p_after is not None]
    cf_drop_values = [row.cf_drop for row in called_rows if row.cf_drop is not None]
    reason_counter = Counter(
        row.counterfactual_reason or "<missing>"
        for row in rows
    )
    parent_without_fragment_nonempty_count = sum(
        1 for row in rows if row.parent_without_fragment_smiles
    )
    residual_sanitize_failed_count = sum(1 for row in rows if row.residual_sanitize_failed)
    invalid_or_not_substructure_count = sum(
        1 for row in rows if row.invalid_or_not_substructure
    )

    return {
        "num_total": total,
        "cf_oracle_called_rate": _round_or_none(_safe_rate(len(called_rows), total)),
        "cf_oracle_skipped_rate": _round_or_none(_safe_rate(len(skipped_rows), total)),
        "p_before": _summarize_probability_values([float(value) for value in p_before_values]),
        "p_after": _summarize_probability_values([float(value) for value in p_after_values]),
        "cf_drop": _summarize_probability_values([float(value) for value in cf_drop_values]),
        "cf_flip_rate": _round_or_none(
            _safe_rate(sum(1 for row in called_rows if row.cf_flip), len(called_rows))
        ),
        "counterfactual_reason_distribution": _distribution(reason_counter, total),
        "parent_without_fragment_smiles_nonempty_rate": _round_or_none(
            _safe_rate(parent_without_fragment_nonempty_count, total)
        ),
        "residual_sanitize_failed_rate": _round_or_none(
            _safe_rate(residual_sanitize_failed_count, total)
        ),
        "invalid_or_not_substructure_rate": _round_or_none(
            _safe_rate(invalid_or_not_substructure_count, total)
        ),
    }


def _sample_pairs(
    xs: list[float],
    ys: list[float],
    *,
    sample_size: int,
) -> tuple[list[float], list[float]]:
    if sample_size <= 0 or len(xs) <= sample_size:
        return xs, ys
    rng = random.Random(0)
    indices = sorted(rng.sample(range(len(xs)), sample_size))
    return [xs[index] for index in indices], [ys[index] for index in indices]


def _compute_correlations(
    xs: list[float],
    ys: list[float],
    *,
    sample_size: int,
) -> dict[str, Any]:
    if pearsonr is None or spearmanr is None:
        return {
            "correlations_available": False,
            "pearson": None,
            "spearman": None,
            "num_pairs": len(xs),
        }
    sampled_xs, sampled_ys = _sample_pairs(xs, ys, sample_size=sample_size)
    if len(sampled_xs) < 2 or len(set(sampled_xs)) < 2 or len(set(sampled_ys)) < 2:
        return {
            "correlations_available": True,
            "pearson": None,
            "spearman": None,
            "num_pairs": len(sampled_xs),
        }
    try:
        pearson_value = float(pearsonr(sampled_xs, sampled_ys).statistic)
    except Exception:
        pearson_value = None
    try:
        spearman_value = float(spearmanr(sampled_xs, sampled_ys).statistic)
    except Exception:
        spearman_value = None
    return {
        "correlations_available": True,
        "pearson": _round_or_none(pearson_value),
        "spearman": _round_or_none(spearman_value),
        "num_pairs": len(sampled_xs),
    }


def _build_group_stat(rows: list[NormalizedPoolRow]) -> dict[str, Any]:
    reward_values = [row.reward_total for row in rows if row.reward_total is not None]
    return {
        "count": len(rows),
        "reward_mean": _round_or_none(_safe_mean([float(value) for value in reward_values])),
        "reward_median": _round_or_none(_safe_median([float(value) for value in reward_values])),
    }


def _quantile_bucket_rows(
    rows: list[NormalizedPoolRow],
    values_by_index: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    if not values_by_index:
        return []
    ordered = sorted(values_by_index, key=lambda item: item[1])
    total = len(ordered)
    buckets: list[dict[str, Any]] = []
    boundaries = [0, total // 4, total // 2, (3 * total) // 4, total]
    for bucket_index in range(4):
        start = boundaries[bucket_index]
        end = boundaries[bucket_index + 1]
        if start >= end:
            continue
        row_indices = [ordered[position][0] for position in range(start, end)]
        bucket_rows = [rows[index] for index in row_indices]
        bucket_values = [ordered[position][1] for position in range(start, end)]
        bucket_summary = _build_group_stat(bucket_rows)
        bucket_summary.update(
            {
                "group": f"q{bucket_index + 1}",
                "min_value": _round_or_none(min(bucket_values)),
                "max_value": _round_or_none(max(bucket_values)),
            }
        )
        buckets.append(bucket_summary)
    return buckets


def audit_reward_component_correlation(
    rows: list[NormalizedPoolRow],
    *,
    sim_sample_size: int,
) -> dict[str, Any]:
    """Audit how reward_total aligns with structural/counterfactual fields."""

    field_extractors: OrderedDict[str, Any] = OrderedDict(
        (
            ("cf_drop", lambda row: row.cf_drop),
            ("cf_flip", lambda row: 1.0 if row.cf_flip else 0.0),
            ("direct_substructure", lambda row: 1.0 if row.direct_substructure else 0.0),
            ("final_substructure", lambda row: 1.0 if row.final_substructure else 0.0),
            ("projection_used", lambda row: 1.0 if row.projection_used else 0.0),
            ("atom_ratio", lambda row: row.atom_ratio),
            ("fragment_atom_count", lambda row: float(row.fragment_atom_count) if row.fragment_atom_count is not None else None),
            ("parse_ok", lambda row: 1.0 if row.parse_ok else 0.0),
            ("core_unusable", lambda row: 1.0 if row.core_unusable else 0.0),
        )
    )
    summary: dict[str, Any] = {}
    for field_name, extractor in field_extractors.items():
        paired_rows: list[tuple[int, float, float]] = []
        for index, row in enumerate(rows):
            if row.reward_total is None:
                continue
            field_value = extractor(row)
            if field_value is None:
                continue
            paired_rows.append((index, float(field_value), float(row.reward_total)))

        xs = [value for _, value, _ in paired_rows]
        ys = [reward for _, _, reward in paired_rows]
        field_summary: dict[str, Any] = {
            "num_pairs": len(paired_rows),
            "grouped_reward_stats": [],
        }
        correlations = _compute_correlations(xs, ys, sample_size=sim_sample_size)
        field_summary.update(correlations)

        if field_name in {
            "cf_flip",
            "direct_substructure",
            "final_substructure",
            "projection_used",
            "parse_ok",
            "core_unusable",
        }:
            false_rows = [rows[index] for index, value, _ in paired_rows if value < 0.5]
            true_rows = [rows[index] for index, value, _ in paired_rows if value >= 0.5]
            field_summary["grouped_reward_stats"] = [
                {"group": "false", **_build_group_stat(false_rows)},
                {"group": "true", **_build_group_stat(true_rows)},
            ]
        else:
            field_summary["grouped_reward_stats"] = _quantile_bucket_rows(
                rows,
                [(index, value) for index, value, _ in paired_rows],
            )

        summary[field_name] = field_summary
    return summary


def _summarize_subset(rows: list[NormalizedPoolRow]) -> dict[str, Any]:
    reward_values = [float(row.reward_total) for row in rows if row.reward_total is not None]
    cf_drop_values = [float(row.cf_drop) for row in rows if row.cf_drop is not None]
    atom_ratio_values = [float(row.atom_ratio) for row in rows if row.atom_ratio is not None]
    projection_score_values = [
        float(row.projection_score) for row in rows if row.projection_score is not None
    ]
    return {
        "count": len(rows),
        "reward_mean": _round_or_none(_safe_mean(reward_values)),
        "cf_drop_mean": _round_or_none(_safe_mean(cf_drop_values)),
        "cf_flip_rate": _round_or_none(_safe_rate(sum(1 for row in rows if row.cf_flip), len(rows))),
        "atom_ratio_mean": _round_or_none(_safe_mean(atom_ratio_values)),
        "projection_score_mean": _round_or_none(_safe_mean(projection_score_values)),
    }


def audit_projection_loophole(rows: list[NormalizedPoolRow]) -> dict[str, Any]:
    """Compare direct vs projected vs invalid branches for reward shortcuts."""

    direct_rows = [row for row in rows if row.direct_substructure]
    projection_rows = [row for row in rows if row.projection_used]
    invalid_rows = [
        row
        for row in rows
        if (not row.parse_ok) or row.core_unusable or (not row.valid)
    ]
    direct_summary = _summarize_subset(direct_rows)
    projection_summary = _summarize_subset(projection_rows)
    invalid_summary = _summarize_subset(invalid_rows)
    direct_reward = direct_summary.get("reward_mean")
    projection_reward = projection_summary.get("reward_mean")
    possible_loophole = bool(
        direct_reward is not None
        and projection_reward is not None
        and float(projection_reward) >= float(direct_reward)
        and projection_summary["count"] > 0
    )
    return {
        "direct_substructure_true": direct_summary,
        "projection_used_true": projection_summary,
        "invalid_core_unusable_parse_failed": invalid_summary,
        "possible_projection_loophole": possible_loophole,
    }


def _atom_ratio_bucket(row: NormalizedPoolRow) -> str:
    if row.full_parent:
        return "full-parent"
    atom_ratio = row.atom_ratio
    if atom_ratio is None:
        return "missing"
    for bucket_name, low, high in _SIZE_BUCKETS:
        if low <= float(atom_ratio) < high:
            return bucket_name
    return "full-parent" if float(atom_ratio) >= 1.0 else "missing"


def audit_size_loophole(rows: list[NormalizedPoolRow]) -> dict[str, Any]:
    """Audit whether reward/cf-flip increasingly favors overly large fragments."""

    buckets: OrderedDict[str, list[NormalizedPoolRow]] = OrderedDict(
        (name, []) for name, _, _ in _SIZE_BUCKETS
    )
    buckets["full-parent"] = []
    buckets["missing"] = []
    for row in rows:
        buckets[_atom_ratio_bucket(row)].append(row)

    bucket_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for bucket_name, bucket_rows in buckets.items():
        reward_values = [float(row.reward_total) for row in bucket_rows if row.reward_total is not None]
        cf_drop_values = [float(row.cf_drop) for row in bucket_rows if row.cf_drop is not None]
        bucket_summaries[bucket_name] = {
            "count": len(bucket_rows),
            "reward_mean": _round_or_none(_safe_mean(reward_values)),
            "cf_drop_mean": _round_or_none(_safe_mean(cf_drop_values)),
            "cf_flip_rate": _round_or_none(
                _safe_rate(sum(1 for row in bucket_rows if row.cf_flip), len(bucket_rows))
            ),
            "direct_substructure_rate": _round_or_none(
                _safe_rate(sum(1 for row in bucket_rows if row.direct_substructure), len(bucket_rows))
            ),
            "projection_used_rate": _round_or_none(
                _safe_rate(sum(1 for row in bucket_rows if row.projection_used), len(bucket_rows))
            ),
        }

    mid_rows = buckets["0.1-0.2"] + buckets["0.2-0.4"]
    high_rows = buckets["0.4-0.6"] + buckets["0.6-0.8"] + buckets["0.8-1.0"] + buckets["full-parent"]
    mid_reward = _safe_mean([float(row.reward_total) for row in mid_rows if row.reward_total is not None])
    high_reward = _safe_mean([float(row.reward_total) for row in high_rows if row.reward_total is not None])
    mid_flip = _safe_rate(sum(1 for row in mid_rows if row.cf_flip), len(mid_rows))
    high_flip = _safe_rate(sum(1 for row in high_rows if row.cf_flip), len(high_rows))
    reasons: list[str] = []
    if high_rows and mid_rows and high_reward is not None and mid_reward is not None and high_reward >= mid_reward + 0.5:
        reasons.append(
            f"high_atom_ratio_reward_mean({round(high_reward, 4)}) >= mid_atom_ratio_reward_mean({round(mid_reward, 4)}) + 0.5"
        )
    if high_rows and mid_rows and high_flip >= mid_flip + 0.15:
        reasons.append(
            f"high_atom_ratio_cf_flip_rate({round(high_flip, 4)}) >= mid_atom_ratio_cf_flip_rate({round(mid_flip, 4)}) + 0.15"
        )
    return {
        "by_bucket": bucket_summaries,
        "possible_size_loophole": bool(reasons),
        "possible_size_loophole_reasons": reasons,
        "mid_atom_ratio_reward_mean": _round_or_none(mid_reward),
        "high_atom_ratio_reward_mean": _round_or_none(high_reward),
        "mid_atom_ratio_cf_flip_rate": _round_or_none(mid_flip),
        "high_atom_ratio_cf_flip_rate": _round_or_none(high_flip),
    }


def _group_reward_mean(correlation_summary: dict[str, Any], field_name: str, group: str) -> float | None:
    for item in correlation_summary.get(field_name, {}).get("grouped_reward_stats", []):
        if item.get("group") == group:
            return _as_float(item.get("reward_mean"))
    return None


def build_final_judgment(
    teacher_summary: dict[str, Any],
    oracle_summary: dict[str, Any],
    correlation_summary: dict[str, Any],
    projection_summary: dict[str, Any],
    size_summary: dict[str, Any],
) -> dict[str, Any]:
    """Render the final diagnosis requested by the PPO drift audit."""

    teacher_correct_rate = _as_float(teacher_summary.get("teacher_correct_rate")) or 0.0
    low_confidence_rate = _as_float(teacher_summary.get("low_confidence_rate")) or 0.0
    very_low_confidence_rate = _as_float(teacher_summary.get("very_low_confidence_rate")) or 0.0
    teacher_reliable = (
        bool(teacher_summary.get("teacher_available"))
        and teacher_correct_rate >= 0.8
        and low_confidence_rate <= 0.2
        and very_low_confidence_rate <= 0.1
    )

    cf_called_rate = _as_float(oracle_summary.get("cf_oracle_called_rate")) or 0.0
    cf_skipped_rate = _as_float(oracle_summary.get("cf_oracle_skipped_rate")) or 0.0
    residual_sanitize_failed_rate = _as_float(
        oracle_summary.get("residual_sanitize_failed_rate")
    ) or 0.0
    invalid_or_not_substructure_rate = _as_float(
        oracle_summary.get("invalid_or_not_substructure_rate")
    ) or 0.0
    cf_reward_heavily_impacted = (
        cf_called_rate < 0.6
        or cf_skipped_rate > 0.4
        or residual_sanitize_failed_rate > 0.05
        or invalid_or_not_substructure_rate > 0.25
    )

    cf_drop_corr = correlation_summary.get("cf_drop", {})
    cf_drop_pearson = _as_float(cf_drop_corr.get("pearson"))
    cf_drop_spearman = _as_float(cf_drop_corr.get("spearman"))
    cf_flip_reward_false = _group_reward_mean(correlation_summary, "cf_flip", "false")
    cf_flip_reward_true = _group_reward_mean(correlation_summary, "cf_flip", "true")
    reward_driven_by_cf = bool(
        (
            (cf_drop_pearson is not None and cf_drop_pearson >= 0.25)
            or (cf_drop_spearman is not None and cf_drop_spearman >= 0.25)
        )
        or (
            cf_flip_reward_true is not None
            and cf_flip_reward_false is not None
            and cf_flip_reward_true >= cf_flip_reward_false + 0.3
        )
    )

    possible_projection_loophole = bool(projection_summary.get("possible_projection_loophole"))
    possible_size_loophole = bool(size_summary.get("possible_size_loophole"))

    if not teacher_reliable:
        primary_diagnosis = "teacher_problem_more_likely"
    elif cf_reward_heavily_impacted or not reward_driven_by_cf or possible_projection_loophole or possible_size_loophole:
        primary_diagnosis = "reward_or_teacher_issue_more_likely"
    else:
        primary_diagnosis = "ppo_drift_more_likely"

    if primary_diagnosis == "ppo_drift_more_likely":
        secondary_diagnosis = "data_order_or_prompt_difficulty_shift_possible_but_secondary"
    elif primary_diagnosis == "teacher_problem_more_likely":
        secondary_diagnosis = "ppo_drift_still_possible_after_teacher_fix"
    else:
        secondary_diagnosis = "ppo_drift_can_still_amplify_existing_reward_shortcuts"

    return {
        "teacher_reliable": teacher_reliable,
        "cf_reward_heavily_impacted": cf_reward_heavily_impacted,
        "reward_total_reasonably_cf_driven": reward_driven_by_cf,
        "possible_projection_loophole": possible_projection_loophole,
        "possible_size_loophole": possible_size_loophole,
        "primary_diagnosis": primary_diagnosis,
        "secondary_diagnosis": secondary_diagnosis,
    }


def render_reward_teacher_report(summary: dict[str, Any]) -> str:
    """Render the final human-readable audit report."""

    teacher = summary["teacher_parent_reliability"]
    oracle = summary["candidate_pool_oracle_validity"]
    correlations = summary["reward_component_correlation"]
    projection = summary["projection_loophole_audit"]
    size = summary["size_loophole_audit"]
    judgment = summary["final_judgment"]

    lines = [
        "Reward / Teacher Audit Report",
        f"generated_at_utc: {summary['metadata']['generated_at_utc']}",
        f"dataset_path: {summary['metadata']['dataset_path']}",
        f"candidate_pool: {summary['metadata']['candidate_pool']}",
        f"teacher_path: {summary['metadata']['teacher_path']}",
        "",
        "1. Teacher 在原始 parent 上是否可靠",
        (
            f"- answer: {'yes' if judgment['teacher_reliable'] else 'no'}; "
            f"teacher_correct_rate={teacher['teacher_correct_rate']} "
            f"low_confidence_rate={teacher['low_confidence_rate']} "
            f"very_low_confidence_rate={teacher['very_low_confidence_rate']}"
        ),
        "",
        "2. cf reward 是否被大量 skipped 或 deletion failed 影响",
        (
            f"- answer: {'yes' if judgment['cf_reward_heavily_impacted'] else 'no'}; "
            f"cf_oracle_called_rate={oracle['cf_oracle_called_rate']} "
            f"cf_oracle_skipped_rate={oracle['cf_oracle_skipped_rate']} "
            f"residual_sanitize_failed_rate={oracle['residual_sanitize_failed_rate']} "
            f"invalid_or_not_substructure_rate={oracle['invalid_or_not_substructure_rate']}"
        ),
        "",
        "3. reward_total 是否主要由 cf_drop / cf_flip 合理驱动",
        (
            f"- answer: {'yes' if judgment['reward_total_reasonably_cf_driven'] else 'no'}; "
            f"cf_drop_pearson={correlations['cf_drop']['pearson']} "
            f"cf_drop_spearman={correlations['cf_drop']['spearman']} "
            f"cf_flip_reward_mean(false)={_group_reward_mean(correlations, 'cf_flip', 'false')} "
            f"cf_flip_reward_mean(true)={_group_reward_mean(correlations, 'cf_flip', 'true')}"
        ),
        "",
        "4. projection 是否可能成为 reward shortcut",
        (
            f"- answer: {'yes' if judgment['possible_projection_loophole'] else 'no'}; "
            f"direct_reward_mean={projection['direct_substructure_true']['reward_mean']} "
            f"projection_reward_mean={projection['projection_used_true']['reward_mean']}"
        ),
        "",
        "5. reward 是否鼓励过大片段",
        (
            f"- answer: {'yes' if judgment['possible_size_loophole'] else 'no'}; "
            f"mid_atom_ratio_reward_mean={size['mid_atom_ratio_reward_mean']} "
            f"high_atom_ratio_reward_mean={size['high_atom_ratio_reward_mean']} "
            f"mid_atom_ratio_cf_flip_rate={size['mid_atom_ratio_cf_flip_rate']} "
            f"high_atom_ratio_cf_flip_rate={size['high_atom_ratio_cf_flip_rate']}"
        ),
        "",
        "6. 当前 100-step 后退化更可能来自哪里",
        f"- primary_diagnosis: {judgment['primary_diagnosis']}",
        f"- secondary_diagnosis: {judgment['secondary_diagnosis']}",
    ]
    return "\n".join(lines).strip() + "\n"


def run_reward_teacher_audit(
    *,
    dataset_path: str | Path,
    candidate_pool: str | Path,
    teacher_path: str | Path,
    out_dir: str | Path,
    config: RewardTeacherAuditConfig,
    teacher_scorer: TeacherSemanticScorer | None = None,
) -> dict[str, Any]:
    """Run the full teacher/reward audit and write all outputs."""

    dataset_records, dataset_metadata = load_dataset_parent_records(
        dataset_path,
        label_col=config.label_col,
        smiles_col=config.smiles_col,
    )
    scorer = teacher_scorer or TeacherSemanticScorer(teacher_path)
    pool_rows = load_candidate_pool_rows(candidate_pool)

    teacher_summary = audit_teacher_parent_reliability(
        dataset_records,
        teacher_scorer=scorer,
    )
    oracle_summary = audit_candidate_pool_oracle_validity(pool_rows)
    correlation_summary = audit_reward_component_correlation(
        pool_rows,
        sim_sample_size=config.sim_sample_size,
    )
    projection_summary = audit_projection_loophole(pool_rows)
    size_summary = audit_size_loophole(pool_rows)
    final_judgment = build_final_judgment(
        teacher_summary,
        oracle_summary,
        correlation_summary,
        projection_summary,
        size_summary,
    )

    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "candidate_pool": str(Path(candidate_pool).expanduser().resolve()),
            "teacher_path": str(Path(teacher_path).expanduser().resolve()),
            "out_dir": str(Path(out_dir).expanduser().resolve()),
            "label_col": config.label_col,
            "smiles_col": config.smiles_col,
            "sim_sample_size": config.sim_sample_size,
            "dataset_loader": dataset_metadata,
        },
        "teacher_parent_reliability": teacher_summary,
        "candidate_pool_oracle_validity": oracle_summary,
        "reward_component_correlation": correlation_summary,
        "projection_loophole_audit": projection_summary,
        "size_loophole_audit": size_summary,
        "final_judgment": final_judgment,
    }
    write_audit_outputs(summary, out_dir=out_dir)
    return summary


def write_audit_outputs(summary: dict[str, Any], *, out_dir: str | Path) -> None:
    """Write the requested JSON/TXT audit artifacts."""

    target_dir = ensure_directory(out_dir)

    def _write_json(name: str, payload: dict[str, Any]) -> None:
        (target_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    teacher_summary = summary["teacher_parent_reliability"]
    _write_json("teacher_parent_reliability.json", teacher_summary)
    (target_dir / "teacher_parent_reliability.txt").write_text(
        _render_teacher_parent_summary(teacher_summary),
        encoding="utf-8",
    )
    _write_json("candidate_pool_oracle_validity.json", summary["candidate_pool_oracle_validity"])
    _write_json("reward_component_correlation.json", summary["reward_component_correlation"])
    _write_json("projection_loophole_audit.json", summary["projection_loophole_audit"])
    _write_json("size_loophole_audit.json", summary["size_loophole_audit"])
    _write_json("audit_summary.json", summary)
    (target_dir / "audit_report.txt").write_text(
        render_reward_teacher_report(summary),
        encoding="utf-8",
    )


__all__ = [
    "RewardTeacherAuditConfig",
    "audit_candidate_pool_oracle_validity",
    "audit_projection_loophole",
    "audit_reward_component_correlation",
    "audit_size_loophole",
    "audit_teacher_parent_reliability",
    "load_candidate_pool_rows",
    "load_dataset_parent_records",
    "normalize_candidate_pool_row",
    "render_reward_teacher_report",
    "run_reward_teacher_audit",
    "write_audit_outputs",
]
