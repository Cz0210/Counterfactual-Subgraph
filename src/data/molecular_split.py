"""Deterministic molecular split helpers and fail-closed overlap audits."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SPLIT_NAMES = ("train", "val", "calibration", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.10, 0.10, 0.10)


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_ratios(values: Sequence[float]) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in values)
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(
            f"Expected {len(SPLIT_NAMES)} train/val/calibration/test ratios; "
            f"received {ratios}."
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("Split ratios must be finite and positive.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split ratios must sum to one: {ratios}")
    return ratios


def resolve_unique_column(
    fieldnames: Sequence[str],
    *,
    explicit: str | None,
    aliases: Sequence[str],
    role: str,
) -> str:
    """Resolve one CSV column without silently choosing between aliases."""

    available = {str(value) for value in fieldnames}
    if explicit:
        if explicit not in available:
            raise ValueError(
                f"Configured {role} column {explicit!r} is absent; "
                f"available={sorted(available)}"
            )
        return explicit
    matches = [alias for alias in aliases if alias in available]
    if len(matches) != 1:
        reason = "missing" if not matches else "ambiguous"
        raise ValueError(
            f"BBBP {role} column is {reason}; matches={matches}, "
            f"aliases={list(aliases)}. Set the explicit column override."
        )
    return matches[0]


def hashed_group_split(
    group: str,
    *,
    seed: int,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> str:
    validated = validate_split_ratios(ratios)
    if not str(group):
        raise ValueError("Molecular split group must be non-empty.")
    digest = hashlib.sha256(f"{int(seed)}\0{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    cumulative = 0.0
    for split, ratio in zip(SPLIT_NAMES, validated, strict=True):
        cumulative += ratio
        if value < cumulative or split == SPLIT_NAMES[-1]:
            return split
    raise AssertionError("unreachable split assignment")


def audit_split_overlap(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fields: Sequence[str] = ("molecule_id", "canonical_smiles"),
    require_scaffold_disjoint: bool = False,
    scaffold_field: str = "scaffold_smiles",
) -> dict[str, Any]:
    """Audit every split pair and raise when any protected identity overlaps."""

    missing_splits = sorted(set(SPLIT_NAMES) - set(rows_by_split))
    if missing_splits:
        raise ValueError(f"Split audit is missing cohorts: {missing_splits}")
    checked_fields = list(fields)
    if require_scaffold_disjoint:
        checked_fields.append(scaffold_field)
    sets: dict[str, dict[str, set[str]]] = {}
    for split in SPLIT_NAMES:
        rows = list(rows_by_split[split])
        if not rows:
            raise ValueError(f"Split audit rejects empty cohort: {split}")
        sets[split] = {}
        for field in checked_fields:
            values: list[str] = []
            for row in rows:
                value = str(row.get(field) or "").strip()
                if not value:
                    raise ValueError(
                        f"Split={split} has an empty protected field={field}."
                    )
                values.append(value)
            if len(set(values)) != len(values) and field in fields:
                raise ValueError(
                    f"Split={split} contains duplicate protected field={field}."
                )
            sets[split][field] = set(values)
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            for field in checked_fields:
                shared = sorted(sets[left][field] & sets[right][field])
                if shared:
                    overlaps.append(
                        {
                            "left_split": left,
                            "right_split": right,
                            "field": field,
                            "count": len(shared),
                            "examples": shared[:10],
                        }
                    )
    audit = {
        "schema_version": "molecular_split_leakage_audit_v1",
        "passed": not overlaps,
        "split_names": list(SPLIT_NAMES),
        "split_sizes": {
            split: len(rows_by_split[split]) for split in SPLIT_NAMES
        },
        "protected_fields": checked_fields,
        "require_scaffold_disjoint": bool(require_scaffold_disjoint),
        "overlaps": overlaps,
        "test_usage": "final_evaluation_only",
    }
    if overlaps:
        raise ValueError(
            "Molecular split leakage detected: "
            + "; ".join(
                f"{item['left_split']}/{item['right_split']}:{item['field']}="
                f"{item['count']}"
                for item in overlaps
            )
        )
    return audit


def validate_repository_relative_path(value: str, *, field: str) -> str:
    """Validate a portable repository-relative POSIX file path."""

    rendered = str(value)
    if not rendered or rendered.endswith("/"):
        raise ValueError(f"{field} must be a non-empty file path without trailing '/'.")
    path = PurePosixPath(rendered)
    if path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be a repository-relative POSIX path: {value!r}")
    if any(character in rendered for character in "*?[]{}"):
        raise ValueError(f"{field} does not permit glob syntax: {value!r}")
    return path.as_posix()


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "SPLIT_NAMES",
    "audit_split_overlap",
    "file_sha256",
    "hashed_group_split",
    "resolve_unique_column",
    "stable_json_sha256",
    "validate_repository_relative_path",
    "validate_split_ratios",
]
