"""Helpers for label-specific and unified PPO prompt CSV construction."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chem import parse_smiles
from src.utils.io import ensure_directory

try:  # pragma: no cover - depends on runtime env
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on runtime env
    Chem = None


DEFAULT_SMILES_COLUMNS = (
    "parent_smiles",
    "smiles",
    "SMILES",
    "smiles_raw",
    "prompt",
    "instruction",
    "query",
    "text",
    "input",
)
DEFAULT_LABEL_COLUMNS = (
    "label",
    "original_label",
    "y",
    "HIV_active",
    "HIV",
    "activity",
    "class",
)
PROMPT_SMILES_PATTERNS = (
    re.compile(r"MOLECULE_SMILES:\s*(?P<smiles>[^\n\r]+)"),
    re.compile(r"PARENT_SMILES:\s*(?P<smiles>[^\n\r]+)"),
    re.compile(r"Parent molecule:\s*(?P<smiles>[^\n\r]+)", flags=re.IGNORECASE),
)
PROMPT_LABEL_PATTERNS = (
    re.compile(r"ORIGINAL_LABEL:\s*(?P<label>[01])"),
    re.compile(r"Original class label:\s*(?P<label>[01])", flags=re.IGNORECASE),
)
PROMPT_TEMPLATE = (
    "Parent molecule: {smiles}\n"
    "Original class label: {label}\n"
    "Generate a connected molecular subgraph whose removal would reduce the "
    "teacher model confidence for label {label} or flip the prediction away "
    "from label {label}. Return only the subgraph SMILES.\n"
    "ORIGINAL_LABEL: {label}\n"
    "MOLECULE_SMILES: {smiles}\n"
    "FRAGMENT_SMILES:"
)


@dataclass(frozen=True, slots=True)
class PromptBuildConfig:
    """Shared prompt-building knobs."""

    label_col: str = "label"
    smiles_col: str = "parent_smiles"
    label: int = 1


@dataclass(frozen=True, slots=True)
class UnifiedPromptBuildConfig:
    """Knobs for unified label01 prompt construction."""

    balance_labels: bool = True
    seed: int = 13
    max_per_label: int = 0


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
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "active": 1,
        "positive": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "inactive": 0,
        "negative": 0,
    }
    if text in mapping:
        return mapping[text]
    try:
        numeric = int(float(text))
    except Exception:
        return None
    return numeric if numeric in (0, 1) else None


def _extract_parent_smiles_from_prompt(text: str | None) -> str | None:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    for pattern in PROMPT_SMILES_PATTERNS:
        match = pattern.search(normalized)
        if match:
            smiles = str(match.group("smiles") or "").strip()
            if smiles:
                return smiles
    if "\n" not in normalized and " " not in normalized:
        return normalized
    return None


def _extract_label_from_prompt(text: str | None) -> int | None:
    normalized = str(text or "")
    for pattern in PROMPT_LABEL_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return int(match.group("label"))
    return None


def _resolve_column(
    fieldnames: list[str],
    *,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> str | None:
    available = {name.strip().lower(): name for name in fieldnames if str(name).strip()}
    preferred_key = str(preferred or "").strip().lower()
    if preferred_key and preferred_key in available:
        return available[preferred_key]
    for candidate in fallbacks:
        candidate_key = str(candidate or "").strip().lower()
        if candidate_key and candidate_key in available:
            return available[candidate_key]
    return None


def _canonicalize_parent_smiles(smiles: str | None) -> str | None:
    normalized = _normalize_text(smiles)
    if normalized is None:
        return None
    parsed = parse_smiles(normalized, sanitize=True, canonicalize=True)
    if parsed.parseable and parsed.sanitized and parsed.canonical_smiles:
        return str(parsed.canonical_smiles)
    return normalized


def build_label_conditioned_prompt(parent_smiles: str, label: int) -> str:
    """Build one explicit label-conditioned PPO prompt."""

    return PROMPT_TEMPLATE.format(smiles=str(parent_smiles).strip(), label=int(label))


def load_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load a CSV file into header plus row dictionaries."""

    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not header:
        raise ValueError(f"CSV is missing a header row: {csv_path}")
    return header, rows


def _compute_smiles_features(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    has_dot_or_salt = "." in text
    component_count = max(1, len([chunk for chunk in text.split(".") if chunk]))
    sanitize_fail = False
    atom_count = 0
    if Chem is None or not text:
        return {
            "atom_count": atom_count,
            "has_dot_or_salt": has_dot_or_salt,
            "component_count": component_count,
            "sanitize_fail": True,
        }

    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return {
            "atom_count": atom_count,
            "has_dot_or_salt": has_dot_or_salt,
            "component_count": component_count,
            "sanitize_fail": True,
        }

    atom_count = int(mol.GetNumAtoms())
    try:
        component_count = int(len(Chem.GetMolFrags(mol)))
    except Exception:
        component_count = component_count
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        sanitize_fail = True
    return {
        "atom_count": atom_count,
        "has_dot_or_salt": has_dot_or_salt or component_count > 1,
        "component_count": component_count,
        "sanitize_fail": sanitize_fail,
    }


def build_label_specific_prompt_rows(
    input_csv: str | Path,
    *,
    config: PromptBuildConfig,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Filter one CSV to a single label and attach explicit label-conditioned prompts."""

    header, rows = load_csv_rows(input_csv)
    resolved_smiles_col = _resolve_column(
        header,
        preferred=config.smiles_col,
        fallbacks=DEFAULT_SMILES_COLUMNS,
    )
    resolved_label_col = _resolve_column(
        header,
        preferred=config.label_col,
        fallbacks=DEFAULT_LABEL_COLUMNS,
    )
    if resolved_label_col is None:
        raise ValueError(f"Could not resolve label column from header={header!r}")

    kept_rows: list[dict[str, Any]] = []
    dropped_counts = Counter()
    label_counter = Counter()
    for index, row in enumerate(rows):
        parent_smiles = (
            _normalize_text(row.get(resolved_smiles_col)) if resolved_smiles_col else None
        )
        if parent_smiles is not None:
            parent_smiles = _extract_parent_smiles_from_prompt(parent_smiles) or parent_smiles
        prompt_fallback = _normalize_text(
            row.get("prompt") or row.get("instruction") or row.get("query") or row.get("text")
        )
        if parent_smiles is None:
            parent_smiles = _extract_parent_smiles_from_prompt(prompt_fallback)
        parent_smiles = _canonicalize_parent_smiles(parent_smiles)
        if parent_smiles is None:
            dropped_counts["missing_parent_smiles"] += 1
            continue

        label = _coerce_binary_label(row.get(resolved_label_col))
        if label is None:
            label = _extract_label_from_prompt(prompt_fallback)
        if label is None:
            dropped_counts["missing_label"] += 1
            continue
        label_counter[int(label)] += 1
        if int(label) != int(config.label):
            dropped_counts["label_mismatch"] += 1
            continue

        prompt = build_label_conditioned_prompt(parent_smiles, int(label))
        output_row = dict(row)
        output_row["source_row_index"] = str(index)
        output_row["parent_smiles"] = parent_smiles
        output_row["label"] = int(label)
        output_row["original_label"] = int(label)
        output_row["prompt"] = prompt
        kept_rows.append(output_row)

    output_header = list(header)
    for required_name in ("source_row_index", "parent_smiles", "label", "original_label", "prompt"):
        if required_name not in output_header:
            output_header.append(required_name)

    summary = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "target_label": int(config.label),
        "input_count": len(rows),
        "kept_count": len(kept_rows),
        "resolved_smiles_col": resolved_smiles_col,
        "resolved_label_col": resolved_label_col,
        "input_label_counts": {
            str(label): int(count)
            for label, count in sorted(label_counter.items())
        },
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "prompt_examples_by_label": {
            str(config.label): [row["prompt"] for row in kept_rows[:2]],
        },
    }
    return output_header, kept_rows, summary


def write_prompt_csv_and_summary(
    *,
    header: list[str],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_csv: str | Path,
    out_json: str | Path,
) -> None:
    """Persist one prompt CSV plus its summary JSON."""

    out_csv_path = Path(out_csv).expanduser().resolve()
    out_json_path = Path(out_json).expanduser().resolve()
    ensure_directory(out_csv_path.parent)
    ensure_directory(out_json_path.parent)
    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in header})
    out_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _block_label_counts(rows: list[dict[str, Any]], *, block_size: int = 50) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block_start in range(0, len(rows), block_size):
        block_rows = rows[block_start : block_start + block_size]
        label0_count = 0
        label1_count = 0
        for row in block_rows:
            label = _coerce_binary_label(row.get("label"))
            if label == 0:
                label0_count += 1
            elif label == 1:
                label1_count += 1
        blocks.append(
            {
                "block_start": block_start + 1,
                "block_end": block_start + len(block_rows),
                "label0_count": int(label0_count),
                "label1_count": int(label1_count),
            }
        )
    return blocks


def build_unified_prompt_rows(
    label0_csv: str | Path,
    label1_csv: str | Path,
    *,
    config: UnifiedPromptBuildConfig,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Combine label0 and label1 prompt CSVs into one balanced unified prompt CSV."""

    header0, rows0 = load_csv_rows(label0_csv)
    header1, rows1 = load_csv_rows(label1_csv)
    normalized_rows0 = [dict(row, label=int(_coerce_binary_label(row.get("label")) or 0)) for row in rows0]
    normalized_rows1 = [dict(row, label=int(_coerce_binary_label(row.get("label")) or 1)) for row in rows1]
    normalized_rows0 = [row for row in normalized_rows0 if int(row["label"]) == 0]
    normalized_rows1 = [row for row in normalized_rows1 if int(row["label"]) == 1]

    rng0 = random.Random(int(config.seed))
    rng1 = random.Random(int(config.seed) + 1)
    rng0.shuffle(normalized_rows0)
    rng1.shuffle(normalized_rows1)

    if config.balance_labels:
        per_label_limit = min(len(normalized_rows0), len(normalized_rows1))
    else:
        per_label_limit = max(len(normalized_rows0), len(normalized_rows1))
    if int(config.max_per_label) > 0:
        per_label_limit = min(per_label_limit, int(config.max_per_label))

    selected0 = normalized_rows0[:per_label_limit]
    selected1 = normalized_rows1[:per_label_limit] if config.balance_labels else normalized_rows1[: int(config.max_per_label or len(normalized_rows1))]
    if not config.balance_labels and int(config.max_per_label) > 0:
        selected0 = normalized_rows0[: min(len(normalized_rows0), int(config.max_per_label))]
        selected1 = normalized_rows1[: min(len(normalized_rows1), int(config.max_per_label))]

    unified_rows: list[dict[str, Any]] = []
    max_len = max(len(selected0), len(selected1))
    for index in range(max_len):
        if index < len(selected0):
            unified_rows.append(selected0[index])
        if index < len(selected1):
            unified_rows.append(selected1[index])

    unified_header = list(header0)
    for field in header1:
        if field not in unified_header:
            unified_header.append(field)
    for required_name in ("source_row_index", "parent_smiles", "label", "original_label", "prompt"):
        if required_name not in unified_header:
            unified_header.append(required_name)

    summary = {
        "label0_csv": str(Path(label0_csv).expanduser().resolve()),
        "label1_csv": str(Path(label1_csv).expanduser().resolve()),
        "input_label0_count": len(rows0),
        "input_label1_count": len(rows1),
        "selected_label0_count": len(selected0),
        "selected_label1_count": len(selected1),
        "balance_labels": bool(config.balance_labels),
        "seed": int(config.seed),
        "max_per_label": int(config.max_per_label),
        "num_rows": len(unified_rows),
        "block_label_distribution": _block_label_counts(unified_rows, block_size=50),
        "prompt_examples_by_label": {
            "0": [row["prompt"] for row in selected0[:2]],
            "1": [row["prompt"] for row in selected1[:2]],
        },
    }
    return unified_header, unified_rows, summary


def check_unified_prompt_balance(
    dataset_path: str | Path,
    *,
    block_size: int = 50,
) -> dict[str, Any]:
    """Summarize label balance and parent-chemistry mix for a unified prompt CSV."""

    header, rows = load_csv_rows(dataset_path)
    resolved_smiles_col = _resolve_column(
        header,
        preferred="parent_smiles",
        fallbacks=DEFAULT_SMILES_COLUMNS,
    )
    resolved_label_col = _resolve_column(
        header,
        preferred="label",
        fallbacks=DEFAULT_LABEL_COLUMNS,
    )
    if resolved_smiles_col is None or resolved_label_col is None:
        raise ValueError(f"Could not resolve smiles/label columns from header={header!r}")

    label_counts = Counter()
    atom_counts: list[int] = []
    dot_salt_count = 0
    sanitize_fail_count = 0
    block_rows: list[dict[str, Any]] = []
    for row in rows:
        label = _coerce_binary_label(row.get(resolved_label_col))
        if label is not None:
            label_counts[int(label)] += 1
        smiles = _normalize_text(row.get(resolved_smiles_col))
        features = _compute_smiles_features(smiles or "")
        atom_counts.append(int(features["atom_count"]))
        dot_salt_count += int(bool(features["has_dot_or_salt"]))
        sanitize_fail_count += int(bool(features["sanitize_fail"]))
        enriched = dict(row)
        enriched["_atom_count"] = int(features["atom_count"])
        enriched["_has_dot_or_salt"] = bool(features["has_dot_or_salt"])
        enriched["_sanitize_fail"] = bool(features["sanitize_fail"])
        block_rows.append(enriched)

    blocks: list[dict[str, Any]] = []
    for block_start in range(0, len(block_rows), int(block_size)):
        chunk = block_rows[block_start : block_start + int(block_size)]
        label0_count = 0
        label1_count = 0
        for row in chunk:
            label = _coerce_binary_label(row.get(resolved_label_col))
            if label == 0:
                label0_count += 1
            elif label == 1:
                label1_count += 1
        chunk_atom_counts = [int(row["_atom_count"]) for row in chunk]
        blocks.append(
            {
                "block_start": block_start + 1,
                "block_end": block_start + len(chunk),
                "label0_count": int(label0_count),
                "label1_count": int(label1_count),
                "atom_count_mean": (sum(chunk_atom_counts) / len(chunk_atom_counts)) if chunk_atom_counts else 0.0,
                "dot_salt_count": int(sum(1 for row in chunk if bool(row["_has_dot_or_salt"]))),
                "sanitize_fail_count": int(sum(1 for row in chunk if bool(row["_sanitize_fail"]))),
            }
        )

    return {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "resolved_smiles_col": resolved_smiles_col,
        "resolved_label_col": resolved_label_col,
        "total_rows": len(rows),
        "label_counts": {
            str(label): int(count)
            for label, count in sorted(label_counts.items())
        },
        "atom_count_mean": (sum(atom_counts) / len(atom_counts)) if atom_counts else 0.0,
        "dot_salt_count": int(dot_salt_count),
        "sanitize_fail_count": int(sanitize_fail_count),
        "block_size": int(block_size),
        "blocks": blocks,
    }


__all__ = [
    "PromptBuildConfig",
    "UnifiedPromptBuildConfig",
    "build_label_conditioned_prompt",
    "build_label_specific_prompt_rows",
    "build_unified_prompt_rows",
    "check_unified_prompt_balance",
    "write_prompt_csv_and_summary",
]
