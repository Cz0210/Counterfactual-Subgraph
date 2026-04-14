"""Utilities for evaluating base-model prediction logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(
        "RDKit is required for base metric evaluation. "
        "Please run inside the smiles_pip118 environment."
    ) from exc


@dataclass(frozen=True, slots=True)
class BaseMetricSummary:
    """Aggregate metrics for a base-model JSONL prediction log."""

    total_records: int
    valid_records: int
    capped_records: int
    skipped_records: int

    @property
    def validity_rate(self) -> float:
        """Return the RDKit-valid fraction in percentage space."""

        if self.total_records == 0:
            return 0.0
        return 100.0 * self.valid_records / self.total_records

    @property
    def capping_rate(self) -> float:
        """Return the fraction containing at least one dummy atom marker."""

        if self.total_records == 0:
            return 0.0
        return 100.0 * self.capped_records / self.total_records


def _is_valid_smiles(smiles: str) -> bool:
    """Return True when RDKit can parse and sanitize the SMILES string."""

    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return False
    return mol is not None


def compute_base_metrics(log_path: Path, *, prediction_field: str = "prediction") -> BaseMetricSummary:
    """Compute capping and validity metrics from a JSONL prediction log."""

    total_records = 0
    valid_records = 0
    capped_records = 0
    skipped_records = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                skipped_records += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_records += 1
                continue

            if not isinstance(record, dict):
                skipped_records += 1
                continue

            prediction = str(record.get(prediction_field, "") or "").strip()
            total_records += 1

            if "*" in prediction:
                capped_records += 1
            if _is_valid_smiles(prediction):
                valid_records += 1

    return BaseMetricSummary(
        total_records=total_records,
        valid_records=valid_records,
        capped_records=capped_records,
        skipped_records=skipped_records,
    )
