"""Normalized dataset loading for PPO prompt CSV / JSONL files."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.utils.io import read_jsonl


_DEFAULT_SMILES_COLUMNS = (
    "parent_smiles",
    "smiles",
    "SMILES",
    "prompt",
    "instruction",
    "query",
    "text",
    "input",
)
_DEFAULT_LABEL_COLUMNS = ("label", "original_label", "y", "HIV_active", "HIV", "activity", "class")
_DEFAULT_PROMPT_COLUMNS = ("prompt", "instruction", "query", "text", "input")
_PROMPT_SMILES_PATTERNS = (
    re.compile(r"MOLECULE_SMILES:\s*(?P<smiles>[^\n\r]+)"),
    re.compile(r"PARENT_SMILES:\s*(?P<smiles>[^\n\r]+)"),
)
_LABEL_PATTERN = re.compile(r"ORIGINAL_LABEL:\s*(?P<label>[01])")


@dataclass(frozen=True, slots=True)
class PPOPromptRecord:
    """One normalized PPO prompt record for generation / audit pipelines."""

    parent_index: int
    parent_smiles: str
    label: int
    prompt: str
    raw_payload: dict[str, Any]


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "positive": 1,
        "pos": 1,
        "active": 1,
        "hiv_active": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "negative": 0,
        "neg": 0,
        "inactive": 0,
        "hiv_inactive": 0,
    }
    if text in mapping:
        return mapping[text]
    try:
        integer_value = int(float(text))
    except Exception:
        return None
    return integer_value if integer_value in (0, 1) else None


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


def _extract_label_from_prompt(value: str | None) -> int | None:
    match = _LABEL_PATTERN.search(str(value or ""))
    if not match:
        return None
    return int(match.group("label"))


def _load_raw_rows(path: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            field_names = list(reader.fieldnames or [])
        return rows, field_names, "csv"
    if suffix in {".jsonl", ".json"}:
        rows = read_jsonl(path)
        field_names = list(rows[0].keys()) if rows else []
        return rows, field_names, suffix.lstrip(".")
    raise ValueError(f"Unsupported PPO prompt dataset format: {path}")


def _resolve_optional_column(
    field_names: list[str],
    *,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> str | None:
    available = {name.strip().lower(): name for name in field_names if str(name).strip()}
    preferred_key = str(preferred or "").strip().lower()
    if preferred_key and preferred_key in available:
        return available[preferred_key]
    for candidate in fallbacks:
        candidate_key = candidate.strip().lower()
        if candidate_key in available:
            return available[candidate_key]
    return None


def load_ppo_prompt_records(
    dataset_path: str | Path,
    *,
    label_col: str = "label",
    smiles_col: str = "parent_smiles",
    target_label: int | None = None,
    limit: int = 0,
) -> tuple[list[PPOPromptRecord], dict[str, Any]]:
    """Load one PPO prompt dataset with robust prompt/smiles/label fallbacks."""

    path = Path(dataset_path).expanduser().resolve()
    raw_rows, field_names, dataset_format = _load_raw_rows(path)
    if not raw_rows:
        raise ValueError(f"No rows found in PPO prompt dataset: {path}")

    resolved_smiles_col = _resolve_optional_column(
        field_names,
        preferred=smiles_col,
        fallbacks=_DEFAULT_SMILES_COLUMNS,
    )
    resolved_label_col = _resolve_optional_column(
        field_names,
        preferred=label_col,
        fallbacks=_DEFAULT_LABEL_COLUMNS,
    )
    resolved_prompt_col = _resolve_optional_column(
        field_names,
        preferred="prompt",
        fallbacks=_DEFAULT_PROMPT_COLUMNS,
    )

    records: list[PPOPromptRecord] = []
    dropped_counts: dict[str, int] = {
        "missing_parent_smiles": 0,
        "missing_label": 0,
        "non_target_label": 0,
        "missing_prompt": 0,
    }

    for index, row in enumerate(raw_rows):
        prompt = _normalize_text(row.get(resolved_prompt_col)) if resolved_prompt_col else None
        if prompt is None:
            for candidate_col in _DEFAULT_PROMPT_COLUMNS:
                prompt = _normalize_text(row.get(candidate_col))
                if prompt:
                    break

        parent_smiles = (
            _extract_parent_smiles_from_prompt(_normalize_text(row.get(resolved_smiles_col)))
            if resolved_smiles_col
            else None
        )
        if parent_smiles is None:
            for candidate_col in _DEFAULT_SMILES_COLUMNS:
                parent_smiles = _extract_parent_smiles_from_prompt(
                    _normalize_text(row.get(candidate_col))
                )
                if parent_smiles:
                    break
        if parent_smiles is None and prompt is not None:
            parent_smiles = _extract_parent_smiles_from_prompt(prompt)
        if parent_smiles is None:
            dropped_counts["missing_parent_smiles"] += 1
            continue

        label = _coerce_binary_label(row.get(resolved_label_col)) if resolved_label_col else None
        if label is None:
            for candidate_col in _DEFAULT_LABEL_COLUMNS:
                label = _coerce_binary_label(row.get(candidate_col))
                if label is not None:
                    break
        if label is None and prompt is not None:
            label = _extract_label_from_prompt(prompt)
        if label is None:
            dropped_counts["missing_label"] += 1
            continue
        if target_label is not None and int(label) != int(target_label):
            dropped_counts["non_target_label"] += 1
            continue

        if prompt is None:
            prompt = build_counterfactual_prompt(
                MoleculeRecord(record_id=index, smiles=parent_smiles, label=int(label)),
                include_label=True,
            )
        if not prompt:
            dropped_counts["missing_prompt"] += 1
            continue

        records.append(
            PPOPromptRecord(
                parent_index=index,
                parent_smiles=parent_smiles,
                label=int(label),
                prompt=prompt,
                raw_payload=dict(row),
            )
        )
        if limit > 0 and len(records) >= int(limit):
            break

    metadata = {
        "dataset_path": str(path),
        "dataset_format": dataset_format,
        "input_row_count": len(raw_rows),
        "usable_row_count": len(records),
        "requested_smiles_col": smiles_col,
        "requested_label_col": label_col,
        "resolved_smiles_col": resolved_smiles_col,
        "resolved_label_col": resolved_label_col,
        "resolved_prompt_col": resolved_prompt_col,
        "target_label": int(target_label) if target_label is not None else None,
        "limit": int(limit),
        "dropped_counts": dropped_counts,
    }
    return records, metadata


__all__ = ["PPOPromptRecord", "load_ppo_prompt_records"]
