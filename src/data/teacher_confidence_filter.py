"""Teacher-confidence filtering for PPO prompt CSV files."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from src.rewards.teacher_semantic import TeacherSemanticScorer
from src.utils.io import ensure_directory


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


@dataclass(frozen=True, slots=True)
class TeacherConfidenceFilterConfig:
    """Runtime knobs for teacher-confidence prompt filtering."""

    label_col: str = "label"
    smiles_col: str = "parent_smiles"
    target_label: int = 1
    min_p_label: float = 0.5
    require_teacher_correct: bool = False


@dataclass(frozen=True, slots=True)
class TeacherConfidenceFilterResult:
    """Result of filtering one PPO prompt CSV by teacher confidence."""

    header: tuple[str, ...]
    kept_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


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


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


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
    try:
        numeric = int(float(text))
    except Exception:
        return None
    return numeric if numeric in (0, 1) else None


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


def _resolve_one_column(
    header: list[str],
    *,
    preferred: str,
    fallbacks: tuple[str, ...],
    kind: str,
) -> str:
    available = {name.strip().lower(): name for name in header if str(name).strip()}
    preferred_key = str(preferred or "").strip().lower()
    if preferred_key and preferred_key in available:
        return available[preferred_key]
    for candidate in fallbacks:
        candidate_key = candidate.strip().lower()
        if candidate_key in available:
            return available[candidate_key]
    raise ValueError(
        f"Could not resolve {kind} column. preferred={preferred!r} available={header!r}"
    )


def _summarize_probabilities(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _round_or_none(_safe_mean(values)),
        "median": _round_or_none(median(values) if values else None),
        "p25": _round_or_none(_safe_quantile(values, 0.25)),
        "p75": _round_or_none(_safe_quantile(values, 0.75)),
    }


def load_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load a CSV file into a header list plus row dictionaries."""

    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not header:
        raise ValueError(f"CSV is missing a header row: {csv_path}")
    return header, rows


def filter_prompt_rows_by_teacher_confidence(
    dataset_path: str | Path,
    *,
    teacher_path: str | Path,
    config: TeacherConfidenceFilterConfig,
    teacher_scorer: TeacherSemanticScorer | None = None,
) -> TeacherConfidenceFilterResult:
    """Filter PPO prompt CSV rows by teacher correctness and confidence."""

    header, rows = load_csv_rows(dataset_path)
    resolved_smiles_col = _resolve_one_column(
        header,
        preferred=config.smiles_col,
        fallbacks=_DEFAULT_SMILES_COLUMNS,
        kind="smiles",
    )
    resolved_label_col = _resolve_one_column(
        header,
        preferred=config.label_col,
        fallbacks=_DEFAULT_LABEL_COLUMNS,
        kind="label",
    )

    scorer = teacher_scorer or TeacherSemanticScorer(teacher_path)

    target_rows: list[dict[str, Any]] = []
    target_probabilities_before: list[float] = []
    kept_probabilities: list[float] = []
    kept_rows: list[dict[str, Any]] = []
    teacher_correct_before_count = 0
    teacher_result_ok_before_count = 0
    low_confidence_removed_count = 0
    very_low_confidence_removed_count = 0
    dropped_reason_counter: Counter[str] = Counter()

    for row in rows:
        label = _coerce_binary_label(row.get(resolved_label_col))
        if label is None:
            dropped_reason_counter["invalid_label"] += 1
            continue
        if label != int(config.target_label):
            dropped_reason_counter["non_target_label"] += 1
            continue

        target_rows.append(dict(row))
        parent_smiles = _extract_parent_smiles_from_prompt(_normalize_text(row.get(resolved_smiles_col)))
        if not parent_smiles and resolved_smiles_col.lower() != "smiles":
            parent_smiles = _extract_parent_smiles_from_prompt(_normalize_text(row.get("smiles")))
        if not parent_smiles and resolved_smiles_col.lower() != "parent_smiles":
            parent_smiles = _extract_parent_smiles_from_prompt(_normalize_text(row.get("parent_smiles")))
        if not parent_smiles:
            for fallback_col in ("prompt", "instruction", "input"):
                parent_smiles = _extract_parent_smiles_from_prompt(
                    _normalize_text(row.get(fallback_col))
                )
                if parent_smiles:
                    break
        if not parent_smiles:
            dropped_reason_counter["missing_parent_smiles"] += 1
            continue
        teacher_result = scorer.score_smiles(parent_smiles, label=label)
        teacher_result_ok = bool(teacher_result.get("teacher_result_ok"))
        pred_label = _coerce_binary_label(teacher_result.get("teacher_label"))
        p_label = _as_float(teacher_result.get("teacher_prob"))

        if teacher_result_ok:
            teacher_result_ok_before_count += 1
        if teacher_result_ok and pred_label == label:
            teacher_correct_before_count += 1
        if p_label is not None:
            target_probabilities_before.append(float(p_label))

        if not teacher_result_ok:
            dropped_reason_counter["teacher_result_not_ok"] += 1
            continue
        if config.require_teacher_correct and pred_label != label:
            dropped_reason_counter["teacher_incorrect"] += 1
            if p_label is not None and p_label < float(config.min_p_label):
                low_confidence_removed_count += 1
                if p_label < 0.2:
                    very_low_confidence_removed_count += 1
            continue
        if p_label is None:
            dropped_reason_counter["missing_teacher_probability"] += 1
            continue
        if p_label < float(config.min_p_label):
            dropped_reason_counter["low_teacher_confidence"] += 1
            low_confidence_removed_count += 1
            if p_label < 0.2:
                very_low_confidence_removed_count += 1
            continue

        kept_rows.append(dict(row))
        kept_probabilities.append(float(p_label))

    target_label_count = len(target_rows)
    kept_count = len(kept_rows)
    dropped_count = target_label_count - kept_count

    summary = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "teacher_path": str(Path(teacher_path).expanduser().resolve()),
        "teacher_available": bool(scorer.available),
        "teacher_format": scorer.teacher_format,
        "teacher_availability_reason": scorer.availability_reason,
        "requested_smiles_col": config.smiles_col,
        "requested_label_col": config.label_col,
        "resolved_smiles_col": resolved_smiles_col,
        "resolved_label_col": resolved_label_col,
        "target_label": int(config.target_label),
        "min_p_label": float(config.min_p_label),
        "require_teacher_correct": bool(config.require_teacher_correct),
        "input_count": len(rows),
        "target_label_count": target_label_count,
        "kept_count": kept_count,
        "dropped_count": dropped_count,
        "kept_rate": _round_or_none(_safe_rate(kept_count, target_label_count)),
        "teacher_result_ok_rate_before": _round_or_none(
            _safe_rate(teacher_result_ok_before_count, target_label_count)
        ),
        "teacher_correct_rate_before": _round_or_none(
            _safe_rate(teacher_correct_before_count, target_label_count)
        ),
        "p_label_mean_before": _summarize_probabilities(target_probabilities_before)["mean"],
        "p_label_median_before": _summarize_probabilities(target_probabilities_before)["median"],
        "p_label_p25_before": _summarize_probabilities(target_probabilities_before)["p25"],
        "p_label_p75_before": _summarize_probabilities(target_probabilities_before)["p75"],
        "p_label_mean_after": _summarize_probabilities(kept_probabilities)["mean"],
        "p_label_median_after": _summarize_probabilities(kept_probabilities)["median"],
        "p_label_p25_after": _summarize_probabilities(kept_probabilities)["p25"],
        "p_label_p75_after": _summarize_probabilities(kept_probabilities)["p75"],
        "low_confidence_removed_count": low_confidence_removed_count,
        "very_low_confidence_removed_count": very_low_confidence_removed_count,
        "drop_reason_counts": dict(sorted(dropped_reason_counter.items())),
    }
    return TeacherConfidenceFilterResult(
        header=tuple(header),
        kept_rows=tuple(kept_rows),
        summary=summary,
    )


def write_filtered_prompt_outputs(
    result: TeacherConfidenceFilterResult,
    *,
    out_csv: str | Path,
    out_json: str | Path,
) -> None:
    """Write the filtered CSV and summary JSON outputs."""

    csv_path = Path(out_csv).expanduser().resolve()
    json_path = Path(out_json).expanduser().resolve()
    ensure_directory(csv_path.parent)
    ensure_directory(json_path.parent)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.header))
        writer.writeheader()
        for row in result.kept_rows:
            writer.writerow({column: row.get(column, "") for column in result.header})

    json_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "TeacherConfidenceFilterConfig",
    "TeacherConfidenceFilterResult",
    "filter_prompt_rows_by_teacher_confidence",
    "load_csv_rows",
    "write_filtered_prompt_outputs",
]
