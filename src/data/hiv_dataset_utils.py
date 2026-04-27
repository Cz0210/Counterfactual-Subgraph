"""Utilities for normalizing HIV.csv records and scaffold-aware parent sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional build dependency
    pd = None

from src.chem import is_rdkit_available, parse_smiles

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - depends on local runtime
    MurckoScaffold = None


_SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES")
_LABEL_COLUMN_CANDIDATES = ("label", "y", "HIV_active", "HIV", "class")


@dataclass(frozen=True, slots=True)
class HIVParentRecord:
    """One normalized parent molecule candidate from the raw HIV CSV."""

    sample_id: str
    source_row_index: int
    source_smiles: str
    parent_smiles: str
    label: int
    raw_label: Any
    parent_atom_count: int
    scaffold_smiles: str
    size_bin: str

    @property
    def stratum_key(self) -> str:
        return f"{self.scaffold_smiles}::{self.size_bin}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_row_index": self.source_row_index,
            "source_smiles": self.source_smiles,
            "parent_smiles": self.parent_smiles,
            "label": self.label,
            "raw_label": self.raw_label,
            "parent_atom_count": self.parent_atom_count,
            "scaffold_smiles": self.scaffold_smiles,
            "size_bin": self.size_bin,
            "stratum_key": self.stratum_key,
        }


def load_hiv_dataframe(csv_path: str | Path):
    """Load the raw HIV CSV with pandas."""

    _require_pandas()
    return pd.read_csv(Path(csv_path).expanduser().resolve())


def resolve_hiv_column_names(dataframe) -> tuple[str, str]:
    """Resolve the SMILES and label columns from common HIV.csv aliases."""

    lower_to_actual: dict[str, str] = {}
    for column in dataframe.columns:
        normalized = str(column).strip().lower()
        if normalized and normalized not in lower_to_actual:
            lower_to_actual[normalized] = str(column)

    smiles_column = _resolve_column_candidate(
        lower_to_actual,
        _SMILES_COLUMN_CANDIDATES,
        kind="SMILES",
    )
    label_column = _resolve_column_candidate(
        lower_to_actual,
        _LABEL_COLUMN_CANDIDATES,
        kind="label",
    )
    return smiles_column, label_column


def normalize_hiv_records(
    dataframe,
    *,
    positive_label: Any = 1,
) -> tuple[list[HIVParentRecord], dict[str, Any]]:
    """Normalize one raw HIV dataframe into canonical parent records."""

    _require_pandas()
    if not is_rdkit_available():
        raise ValueError("RDKit is required for HIV dataset normalization.")

    smiles_column, label_column = resolve_hiv_column_names(dataframe)
    positive_token = _normalize_label_token(positive_label)
    if positive_token is None:
        raise ValueError(f"Could not normalize positive_label={positive_label!r}.")

    unique_label_tokens: list[Any] = []
    for value in dataframe[label_column].tolist():
        token = _normalize_label_token(value)
        if token is None:
            continue
        if token not in unique_label_tokens:
            unique_label_tokens.append(token)

    if positive_token not in unique_label_tokens:
        raise ValueError(
            "The configured positive label was not found in the HIV CSV. "
            f"positive_label={positive_label!r} normalized={positive_token!r} "
            f"available_labels={unique_label_tokens!r}"
        )

    negative_tokens = [token for token in unique_label_tokens if token != positive_token]
    if len(negative_tokens) > 1:
        raise ValueError(
            "Expected a binary HIV label column but found multiple non-positive values. "
            f"positive_label={positive_label!r} available_labels={unique_label_tokens!r}"
        )
    negative_token = negative_tokens[0] if negative_tokens else None

    dropped_counts: Counter[str] = Counter()
    records: list[HIVParentRecord] = []
    for row_index, row in enumerate(dataframe.itertuples(index=False)):
        row_dict = row._asdict()
        source_smiles = str(row_dict.get(smiles_column, "") or "").strip()
        if not source_smiles:
            dropped_counts["empty_smiles"] += 1
            continue

        raw_label = row_dict.get(label_column)
        label_token = _normalize_label_token(raw_label)
        if label_token is None:
            dropped_counts["missing_label"] += 1
            continue
        if label_token == positive_token:
            normalized_label = 1
        elif negative_token is not None and label_token == negative_token:
            normalized_label = 0
        else:
            dropped_counts["unexpected_label_value"] += 1
            continue

        parsed_parent = parse_smiles(
            source_smiles,
            sanitize=True,
            canonicalize=True,
            allow_capped_fragments=False,
        )
        if not parsed_parent.sanitized or parsed_parent.mol is None:
            dropped_counts["invalid_parent_smiles"] += 1
            continue

        parent_smiles = str(parsed_parent.canonical_smiles or source_smiles).strip()
        parent_atom_count = int(parsed_parent.atom_count or parsed_parent.mol.GetNumAtoms())
        scaffold_smiles = murcko_scaffold_smiles(parsed_parent.mol)
        records.append(
            HIVParentRecord(
                sample_id=f"hiv:{row_index}",
                source_row_index=row_index,
                source_smiles=source_smiles,
                parent_smiles=parent_smiles,
                label=normalized_label,
                raw_label=raw_label,
                parent_atom_count=parent_atom_count,
                scaffold_smiles=scaffold_smiles,
                size_bin=parent_atom_count_bin(parent_atom_count),
            )
        )

    label_counts = Counter(record.label for record in records)
    summary = {
        "input_rows": int(len(dataframe)),
        "valid_rows": int(len(records)),
        "dropped_rows": int(sum(dropped_counts.values())),
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "smiles_column": smiles_column,
        "label_column": label_column,
        "positive_label": positive_label,
        "positive_label_normalized": positive_token,
        "negative_label_normalized": negative_token,
        "valid_label_counts": {
            "0": int(label_counts.get(0, 0)),
            "1": int(label_counts.get(1, 0)),
        },
    }
    return records, summary


def murcko_scaffold_smiles(parent_mol: object) -> str:
    """Return a stable Bemis-Murcko scaffold key or ACYCLIC fallback."""

    if MurckoScaffold is None or parent_mol is None:
        return "ACYCLIC"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=parent_mol,
            includeChirality=True,
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        scaffold = ""
    normalized = str(scaffold or "").strip()
    return normalized or "ACYCLIC"


def parent_atom_count_bin(atom_count: int) -> str:
    """Bucket one parent molecule by atom count for stratified sampling."""

    if atom_count <= 15:
        return "atoms_00_15"
    if atom_count <= 25:
        return "atoms_16_25"
    if atom_count <= 35:
        return "atoms_26_35"
    if atom_count <= 45:
        return "atoms_36_45"
    return "atoms_46_plus"


def stratified_round_robin_order(
    records: list[HIVParentRecord],
    *,
    seed: int,
    key_fn: Callable[[HIVParentRecord], str] | None = None,
) -> list[HIVParentRecord]:
    """Return records in a scaffold+size-diversity-first round-robin order."""

    if len(records) <= 1:
        return list(records)

    rng = random.Random(seed)
    group_key = key_fn or (lambda record: record.stratum_key)
    grouped: dict[str, list[HIVParentRecord]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    active_keys = sorted(grouped)
    for key in active_keys:
        rng.shuffle(grouped[key])

    ordered: list[HIVParentRecord] = []
    while active_keys:
        cycle_keys = list(active_keys)
        rng.shuffle(cycle_keys)
        next_active_keys: list[str] = []
        for key in cycle_keys:
            bucket = grouped[key]
            if not bucket:
                continue
            ordered.append(bucket.pop())
            if bucket:
                next_active_keys.append(key)
        active_keys = sorted(next_active_keys)
    return ordered


def sample_records_by_strata(
    records: list[HIVParentRecord],
    *,
    sample_size: int,
    seed: int,
    key_fn: Callable[[HIVParentRecord], str] | None = None,
) -> list[HIVParentRecord]:
    """Downsample records while preserving scaffold+size diversity."""

    if sample_size <= 0 or not records:
        return []
    if sample_size >= len(records):
        return stratified_round_robin_order(
            list(records),
            seed=seed,
            key_fn=key_fn,
        )
    ordered = stratified_round_robin_order(
        list(records),
        seed=seed,
        key_fn=key_fn,
    )
    return ordered[:sample_size]


def label_counts(records: list[HIVParentRecord]) -> dict[str, int]:
    """Return stable stringified label counts."""

    counts = Counter(record.label for record in records)
    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
    }


def _require_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for HIV dataset loading.")


def _resolve_column_candidate(
    lower_to_actual: dict[str, str],
    candidates: tuple[str, ...],
    *,
    kind: str,
) -> str:
    for candidate in candidates:
        actual = lower_to_actual.get(candidate.strip().lower())
        if actual:
            return actual
    available = ", ".join(sorted(lower_to_actual.values()))
    expected = ", ".join(candidates)
    raise ValueError(
        f"Could not resolve the {kind} column. "
        f"Tried [{expected}]. Available columns: {available}"
    )


def _normalize_label_token(value: Any) -> Any:
    if value is None:
        return None
    if pd is not None and bool(pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return 1
    if lowered == "false":
        return 0
    try:
        numeric_value = float(text)
    except ValueError:
        return lowered
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


__all__ = [
    "HIVParentRecord",
    "label_counts",
    "load_hiv_dataframe",
    "murcko_scaffold_smiles",
    "normalize_hiv_records",
    "parent_atom_count_bin",
    "resolve_hiv_column_names",
    "sample_records_by_strata",
    "stratified_round_robin_order",
]
