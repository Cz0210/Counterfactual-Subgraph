"""Dataset adapters that keep JSONL loading separate from training code."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from src.data.schemas import MoleculeRecord, normalize_molecule_record
from src.utils.io import read_jsonl


class JsonlMoleculeDataset(Sequence[MoleculeRecord]):
    """In-memory dataset for the minimal molecule JSONL contract."""

    def __init__(self, records: Iterable[MoleculeRecord]) -> None:
        self._records = tuple(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "JsonlMoleculeDataset":
        rows = read_jsonl(path)
        return cls(normalize_molecule_record(row) for row in rows)

    def __getitem__(self, index: int) -> MoleculeRecord:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)
