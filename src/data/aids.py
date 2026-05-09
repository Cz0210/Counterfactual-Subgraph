"""Local CSV adapters for the AIDS/HIV benchmark data."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from src.data.schemas import MoleculeRecord


@dataclass(frozen=True, slots=True)
class AIDSHIVRecord(MoleculeRecord):
    """One normalized row from the local AIDS/HIV CSV file."""

    activity: str = ""


class AIDSHIVCsvDataset(Sequence[AIDSHIVRecord]):
    """In-memory view over the local AIDS/HIV CSV file."""

    def __init__(self, records: Sequence[AIDSHIVRecord]) -> None:
        self._records = tuple(records)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        smiles_column: str = "smiles",
        label_column: str = "HIV_active",
        activity_column: str = "activity",
    ) -> "AIDSHIVCsvDataset":
        return cls(
            tuple(
                iter_aids_hiv_records(
                    path,
                    smiles_column=smiles_column,
                    label_column=label_column,
                    activity_column=activity_column,
                )
            )
        )

    def __getitem__(self, index: int) -> AIDSHIVRecord:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def sample_one(self, *, seed: int | None = None) -> AIDSHIVRecord:
        if not self._records:
            raise ValueError("AIDS/HIV dataset is empty after CSV normalization.")
        rng = random.Random(seed)
        return rng.choice(self._records)


def iter_aids_hiv_records(
    path: str | Path,
    *,
    smiles_column: str = "smiles",
    label_column: str = "HIV_active",
    activity_column: str = "activity",
) -> Iterator[AIDSHIVRecord]:
    """Yield normalized AIDS/HIV records from a local CSV file."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"AIDS/HIV CSV file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"AIDS/HIV CSV file is missing a header row: {csv_path}")

        required_columns = {smiles_column, label_column}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise KeyError(f"AIDS/HIV CSV file is missing required columns: {missing}")

        for row_index, row in enumerate(reader):
            if row is None:
                continue
            smiles = str(row.get(smiles_column, "") or "").strip()
            if not smiles:
                continue

            raw_label = str(row.get(label_column, "") or "").strip()
            if raw_label not in {"0", "1"}:
                raise ValueError(
                    f"Expected binary label in column '{label_column}' for row {row_index + 2}."
                )

            activity = str(row.get(activity_column, "") or "").strip()
            yield AIDSHIVRecord(
                record_id=row_index,
                smiles=smiles,
                label=int(raw_label),
                activity=activity,
            )


def sample_random_aids_hiv_record(
    path: str | Path,
    *,
    seed: int | None = None,
    smiles_column: str = "smiles",
    label_column: str = "HIV_active",
    activity_column: str = "activity",
) -> AIDSHIVRecord:
    """Load the local AIDS/HIV CSV file and return one deterministic random record."""

    dataset = AIDSHIVCsvDataset.from_csv(
        path,
        smiles_column=smiles_column,
        label_column=label_column,
        activity_column=activity_column,
    )
    return dataset.sample_one(seed=seed)
