"""Typed dataset records for the counterfactual fragment task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


RecordId = str | int


@dataclass(frozen=True, slots=True)
class MoleculeRecord:
    """One parent molecule example used throughout the project."""

    record_id: RecordId
    smiles: str
    label: int


@dataclass(frozen=True, slots=True)
class FragmentExample:
    """A prompt-ready example with an optional weak fragment target."""

    record: MoleculeRecord
    prompt: str
    target_fragment: str | None = None


def normalize_molecule_record(raw: Mapping[str, Any]) -> MoleculeRecord:
    """Validate the minimal JSONL schema defined by the v3 docs."""

    required_keys = {"id", "smiles", "label"}
    missing_keys = required_keys.difference(raw.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required molecule fields: {missing}")

    label = raw["label"]
    if not isinstance(label, int) or label not in (0, 1):
        raise ValueError("Expected binary integer label in {0, 1}")

    smiles = str(raw["smiles"]).strip()
    if not smiles:
        raise ValueError("SMILES must be a non-empty string")

    return MoleculeRecord(record_id=raw["id"], smiles=smiles, label=label)
