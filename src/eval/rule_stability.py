"""Compare two frozen rule selections without reopening held-out molecules."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.chem import parse_smiles
from src.data.hiv_dataset_utils import murcko_scaffold_smiles

try:  # pragma: no cover - runtime dependent
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - runtime dependent
    DataStructs = None
    rdFingerprintGenerator = None


RULE_LIST_KEYS = ("ordered_rules", "selected_rules", "rules")
RULE_ID_KEYS = ("rule_hash", "rule_id", "hash", "id")
SMILES_KEYS = ("core_fragment", "fragment", "rhs_smiles", "smiles")
COVERAGE_KEYS = ("covered_parent_ids", "coverage_parent_ids", "coverage")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen selection must be one JSON object: {source}")
    return payload


def _rule_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw: Any = None
    for key in RULE_LIST_KEYS:
        if isinstance(payload.get(key), list):
            raw = payload[key]
            break
    if raw is None and isinstance(payload.get("ordered_rule_ids"), list):
        lookup = payload.get("rules_by_id")
        if not isinstance(lookup, Mapping):
            raise ValueError("ordered_rule_ids requires rules_by_id for stability audit")
        raw = [lookup.get(str(rule_id)) for rule_id in payload["ordered_rule_ids"]]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Frozen selection contains no ordered rule rows")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if isinstance(value, Mapping):
            row = dict(value)
        else:
            row = {"rule_id": str(value)}
        row["_rank"] = index + 1
        rows.append(row)
    return rows


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _rule_id(row: Mapping[str, Any]) -> str:
    value = _first(row, RULE_ID_KEYS)
    return str(value) if value not in (None, "") else _stable_hash(
        {key: value for key, value in row.items() if key != "_rank"}
    )


def _canonical_smiles(row: Mapping[str, Any]) -> tuple[str, str] | None:
    value = _first(row, SMILES_KEYS)
    if value in (None, ""):
        return None
    parsed = parse_smiles(str(value), sanitize=True, canonicalize=True)
    if not parsed.parseable or not parsed.sanitized or not parsed.canonical_smiles:
        raise ValueError(f"Unparseable selected-rule SMILES: {value!r}")
    return str(parsed.canonical_smiles), murcko_scaffold_smiles(parsed.mol)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _morgan_mean_max(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    if not left or not right:
        return {"status": "NOT_AVAILABLE", "reason": "missing_rule_smiles"}
    if DataStructs is None or rdFingerprintGenerator is None:
        return {"status": "NOT_AVAILABLE", "reason": "rdkit_unavailable"}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles: str) -> Any:
        parsed = parse_smiles(smiles, sanitize=True, canonicalize=True)
        if parsed.mol is None:
            raise ValueError(f"Could not fingerprint rule SMILES: {smiles}")
        return generator.GetFingerprint(parsed.mol)

    left_fp = [fingerprint(value) for value in left]
    right_fp = [fingerprint(value) for value in right]
    left_max = [max(float(DataStructs.TanimotoSimilarity(a, b)) for b in right_fp) for a in left_fp]
    right_max = [max(float(DataStructs.TanimotoSimilarity(b, a)) for a in left_fp) for b in right_fp]
    return {
        "status": "PASS",
        "left_to_right_mean_max": sum(left_max) / len(left_max),
        "right_to_left_mean_max": sum(right_max) / len(right_max),
        "bidirectional_mean_max": sum(left_max + right_max) / len(left_max + right_max),
    }


def _coverage(rows: Sequence[Mapping[str, Any]]) -> set[str] | None:
    found = False
    values: set[str] = set()
    for row in rows:
        raw = _first(row, COVERAGE_KEYS)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple, set)):
            raise ValueError("Rule coverage must be a sequence of parent IDs")
        found = True
        values.update(str(value) for value in raw)
    return values if found else None


def _destination(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    raw = payload.get("destination_distribution")
    totals: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            totals[str(key)] = float(value)
    else:
        for row in rows:
            distribution = row.get("destination_distribution")
            if not isinstance(distribution, Mapping):
                continue
            for key, value in distribution.items():
                totals[str(key)] = totals.get(str(key), 0.0) + float(value)
    if not totals:
        return None
    if any(not math.isfinite(value) or value < 0.0 for value in totals.values()):
        raise ValueError("Destination distribution must be finite and nonnegative")
    total = sum(totals.values())
    if total <= 0.0:
        raise ValueError("Destination distribution has zero mass")
    return {key: value / total for key, value in sorted(totals.items())}


def compare_frozen_rule_selections(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    """Return exact, chemical, scaffold, coverage, and destination similarities."""

    left_payload = _load_object(left_path)
    right_payload = _load_object(right_path)
    left_rows = _rule_rows(left_payload)
    right_rows = _rule_rows(right_payload)
    left_ids = {_rule_id(row) for row in left_rows}
    right_ids = {_rule_id(row) for row in right_rows}
    left_chem = [value for row in left_rows if (value := _canonical_smiles(row)) is not None]
    right_chem = [value for row in right_rows if (value := _canonical_smiles(row)) is not None]
    left_smiles = [value[0] for value in left_chem]
    right_smiles = [value[0] for value in right_chem]
    left_scaffolds = {value[1] for value in left_chem}
    right_scaffolds = {value[1] for value in right_chem}
    left_coverage = _coverage(left_rows)
    right_coverage = _coverage(right_rows)
    left_destination = _destination(left_payload, left_rows)
    right_destination = _destination(right_payload, right_rows)
    destination_keys = sorted(set(left_destination or {}) | set(right_destination or {}))
    destination_similarity: dict[str, Any]
    if left_destination is None or right_destination is None:
        destination_similarity = {"status": "NOT_AVAILABLE", "reason": "missing_destination_distribution"}
    else:
        total_variation = 0.5 * sum(
            abs(left_destination.get(key, 0.0) - right_destination.get(key, 0.0))
            for key in destination_keys
        )
        destination_similarity = {
            "status": "PASS",
            "metric": "one_minus_total_variation",
            "similarity": 1.0 - total_variation,
            "left": left_destination,
            "right": right_destination,
        }
    coverage_similarity = (
        {"status": "PASS", "jaccard": _jaccard(left_coverage, right_coverage)}
        if left_coverage is not None and right_coverage is not None
        else {"status": "NOT_AVAILABLE", "reason": "missing_coverage_sets"}
    )
    return {
        "schema_version": "selected_rule_stability_v1",
        "left_manifest": str(Path(left_path).expanduser().resolve(strict=True)),
        "right_manifest": str(Path(right_path).expanduser().resolve(strict=True)),
        "left_rule_count": len(left_rows),
        "right_rule_count": len(right_rows),
        "exact_rule_jaccard": _jaccard(left_ids, right_ids),
        "morgan_mean_max_similarity": _morgan_mean_max(left_smiles, right_smiles),
        "scaffold_overlap": {
            "left_count": len(left_scaffolds),
            "right_count": len(right_scaffolds),
            "jaccard": _jaccard(left_scaffolds, right_scaffolds),
        },
        "coverage_set_overlap": coverage_similarity,
        "destination_distribution_similarity": destination_similarity,
    }


__all__ = ["compare_frozen_rule_selections"]
