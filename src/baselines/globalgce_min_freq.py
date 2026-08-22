"""Calibration-only GlobalGCE minimum-frequency configuration."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_MIN_FREQ: dict[str, int] = {
    "AIDS": 50,
    "Mutagenicity": 200,
    "NCI1": 100,
    "PROTEINS": 15,
    "ENZYMES": 35,
}
FALLBACK_RATIO_GRID = (0.005, 0.01, 0.02, 0.05)
# Primary BACE route is preregistered from train-cohort size only: round(2% *
# 360)=7.  It is never selected from held-out curves, and remains one member
# of the immutable train-derived grid returned by ``bace_min_freq_grid``.
BACE_PRIMARY_MIN_FREQ = 7


class GlobalGCEMinFreqConfigurationError(ValueError):
    """Raised when a dataset has no leakage-safe minimum-frequency value."""


@dataclass(frozen=True, slots=True)
class MinFreqResolution:
    dataset: str
    value: int
    source: str
    manifest_path: str | None = None


def bace_min_freq_grid(source_train_parent_count: int) -> tuple[int, ...]:
    """Return the fixed ratio grid mapped onto the BACE train cohort."""

    count = int(source_train_parent_count)
    if count < 2:
        raise GlobalGCEMinFreqConfigurationError(
            "BACE source-train parent count must be at least two."
        )
    values = {
        max(2, min(count, int(round(ratio * count))))
        for ratio in FALLBACK_RATIO_GRID
    }
    return tuple(sorted(values))


def _load_manifest(path: Path, dataset: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GlobalGCEMinFreqConfigurationError(
            f"Minimum-frequency manifest must be a JSON object: {path}"
        )
    if str(payload.get("dataset") or "") != dataset:
        raise GlobalGCEMinFreqConfigurationError(
            "Minimum-frequency manifest dataset mismatch: "
            f"actual={payload.get('dataset')!r}, expected={dataset!r}."
        )
    if str(payload.get("selection_split") or "") != "calibration":
        raise GlobalGCEMinFreqConfigurationError(
            "BACE minimum frequency must be selected on calibration."
        )
    if payload.get("test_loaded") is not False:
        raise GlobalGCEMinFreqConfigurationError(
            "BACE minimum-frequency manifest must prove test_loaded=false."
        )
    return payload


def resolve_globalgce_min_freq(
    dataset: str,
    *,
    explicit_min_freq: int | None = None,
    calibration_manifest: str | Path | None = None,
    official_values: Mapping[str, int] = OFFICIAL_MIN_FREQ,
) -> MinFreqResolution:
    """Resolve one explicit, frozen, or official GlobalGCE frequency."""

    name = str(dataset).strip()
    if not name:
        raise GlobalGCEMinFreqConfigurationError("Dataset name is required.")
    if explicit_min_freq is not None:
        value = int(explicit_min_freq)
        if value < 2:
            raise GlobalGCEMinFreqConfigurationError(
                "GlobalGCE minimum frequency must be at least two."
            )
        return MinFreqResolution(name, value, "explicit")
    if calibration_manifest is not None:
        path = Path(calibration_manifest).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_manifest(path, name)
        value = int(payload.get("selected_min_freq") or 0)
        if value < 2:
            raise GlobalGCEMinFreqConfigurationError(
                "Calibration manifest selected_min_freq must be at least two."
            )
        return MinFreqResolution(name, value, "calibration_manifest", str(path))
    if name in official_values:
        value = int(official_values[name])
        if value < 2:
            raise GlobalGCEMinFreqConfigurationError(
                f"Official minimum frequency is invalid for {name}: {value}."
            )
        return MinFreqResolution(name, value, "official")
    raise GlobalGCEMinFreqConfigurationError(
        f"No GlobalGCE minimum frequency is configured for {name}. "
        "Provide an explicit calibration candidate or a frozen calibration manifest."
    )


def select_bace_min_freq(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select from calibration metrics using the preregistered ordering."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise GlobalGCEMinFreqConfigurationError(
            "BACE minimum-frequency calibration produced no candidates."
        )
    for row in candidates:
        if str(row.get("selection_split") or "") != "calibration":
            raise GlobalGCEMinFreqConfigurationError(
                "All minimum-frequency metrics must use calibration."
            )
        if row.get("test_loaded") not in (False, "false", "False", 0, "0"):
            raise GlobalGCEMinFreqConfigurationError(
                "Minimum-frequency selection cannot consume test metrics."
            )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["prefix_auc_k1_k10"]),
            -float(row["multi_threshold_prefix_auc"]),
            float(row["cost"]),
            float(row["coverage_redundancy"]),
            int(row["rule_count"]),
            int(row["min_freq"]),
        ),
    )
    return ranked[0]


def read_calibration_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
