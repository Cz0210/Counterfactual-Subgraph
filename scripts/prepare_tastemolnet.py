#!/usr/bin/env python3
"""Prepare a provenance-bound, scaffold-disjoint TasteMolNet dataset.

This command is deliberately offline.  It accepts a user-provided local CSV or
an already materialized upstream/official CSV, records its provenance, and
never downloads or fabricates labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import get_dataset_spec  # noqa: E402


LABEL_MAP = {0: "Bitter", 1: "Sweet", 2: "Tasteless"}
LABEL_NAME_TO_ID = {value.lower(): key for key, value in LABEL_MAP.items()}
SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.10, 0.10, 0.10)
SOURCE_MODES = ("local", "upstream_processed", "official_supplement")
STANDARDIZATION_VERSION = "tastemolnet_rdkit_isomeric_v1"
LICENSE_REVIEW_REQUIRED = "LICENSE_REVIEW_REQUIRED"
SUPPORTED_ATOMIC_NUMBERS = frozenset(range(1, 119))

_SMILES_CANDIDATES = (
    "processed_smiles",
    "canonical_smiles",
    "model_smiles",
    "smiles",
    "smile",
    "mol",
    "molecule",
)
_LABEL_CANDIDATES = (
    "target",
    "group",
    "label",
    "taste",
    "taste_label",
    "class",
)
_ID_CANDIDATES = (
    "compound_id",
    "molecule_id",
    "id",
    "compound",
)
_AMBIGUOUS_LABEL_TOKENS = {
    "ambiguous",
    "unknown",
    "multi",
    "multi_taste",
    "multitaste",
    "mixed",
    "other",
    "nan",
    "none",
    "",
}
_MULTI_LABEL_SEPARATOR = re.compile(r"(?:[,;/+|]|\band\b|\bor\b)", re.IGNORECASE)


CLEAN_FIELDS = (
    "molecule_id",
    "raw_smiles",
    "canonical_smiles",
    "model_smiles",
    "label",
    "label_name",
    "scaffold",
    "num_atoms",
    "num_bonds",
    "formal_charge",
    "fragment_count",
    "supported_atom_check",
    "molclr_compatible",
    "chemlm_compatible",
    "has_stereochemistry",
    "source",
    "source_row_id",
    "standardization_version",
    "component_policy",
    "split",
    "exclusion_reason",
)
EXCLUDED_FIELDS = (
    "source_row_id",
    "raw_smiles",
    "raw_label",
    "canonical_smiles",
    "model_smiles",
    "normalized_label",
    "normalized_label_name",
    "exclusion_reason",
    "exclusion_detail",
)
CONFLICT_FIELDS = (
    "conflict_identity_type",
    "conflict_identity",
    "labels",
    "label_names",
    "source_row_ids",
    "raw_smiles_values",
)


class TasteMolNetPreparationError(RuntimeError):
    """Raised when the data contract cannot be satisfied safely."""


class TasteLabelError(ValueError):
    """A label is not one of the three admitted single-label classes."""

    def __init__(self, reason: str, value: Any) -> None:
        self.reason = reason
        self.value = value
        super().__init__(f"{reason}: {value!r}")


@dataclass(frozen=True, slots=True)
class DetectedSchema:
    smiles_column: str
    label_column: str
    id_column: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles_column": self.smiles_column,
            "label_column": self.label_column,
            "id_column": self.id_column,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _column_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _resolve_column(
    fieldnames: Sequence[str],
    *,
    explicit: str | None,
    candidates: Sequence[str],
    kind: str,
    required: bool,
) -> str | None:
    token_to_columns: dict[str, list[str]] = defaultdict(list)
    for field in fieldnames:
        token_to_columns[_column_token(field)].append(str(field))
    duplicate_tokens = {
        token: names for token, names in token_to_columns.items() if len(names) > 1
    }
    if duplicate_tokens:
        raise ValueError(
            "CSV contains case/punctuation-ambiguous columns: "
            + json.dumps(duplicate_tokens, sort_keys=True)
        )

    if explicit:
        resolved = token_to_columns.get(_column_token(explicit))
        if not resolved:
            raise ValueError(
                f"Explicit {kind} column {explicit!r} is absent; "
                f"available={list(fieldnames)!r}."
            )
        return resolved[0]

    for candidate in candidates:
        resolved = token_to_columns.get(_column_token(candidate))
        if resolved:
            return resolved[0]
    if required:
        raise ValueError(
            f"Could not detect {kind} column; available={list(fieldnames)!r}, "
            f"candidates={list(candidates)!r}."
        )
    return None


def detect_schema(
    fieldnames: Sequence[str],
    *,
    smiles_column: str | None = None,
    label_column: str | None = None,
    id_column: str | None = None,
) -> DetectedSchema:
    """Detect common upstream TasteMolNet columns without hard-coding one CSV."""

    fields = tuple(str(value) for value in fieldnames if str(value).strip())
    if not fields:
        raise ValueError("TasteMolNet CSV has no header columns.")
    return DetectedSchema(
        smiles_column=str(
            _resolve_column(
                fields,
                explicit=smiles_column,
                candidates=_SMILES_CANDIDATES,
                kind="SMILES",
                required=True,
            )
        ),
        label_column=str(
            _resolve_column(
                fields,
                explicit=label_column,
                candidates=_LABEL_CANDIDATES,
                kind="label",
                required=True,
            )
        ),
        id_column=_resolve_column(
            fields,
            explicit=id_column,
            candidates=_ID_CANDIDATES,
            kind="ID",
            required=False,
        ),
    )


def normalize_taste_label(value: Any) -> tuple[int, str]:
    """Normalize only Bitter/Sweet/Tasteless single-label ground truth."""

    if value is None:
        raise TasteLabelError("AMBIGUOUS_LABEL", value)
    text = str(value).strip()
    lowered = re.sub(r"\s+", "_", text.lower()).strip("_")
    if lowered in _AMBIGUOUS_LABEL_TOKENS:
        raise TasteLabelError("AMBIGUOUS_LABEL", value)
    if _MULTI_LABEL_SEPARATOR.search(lowered):
        found = {
            label_name
            for label_name in LABEL_NAME_TO_ID
            if re.search(rf"\b{re.escape(label_name)}\b", lowered)
        }
        if len(found) != 1 or lowered not in LABEL_NAME_TO_ID:
            raise TasteLabelError("AMBIGUOUS_LABEL", value)
    if lowered in LABEL_NAME_TO_ID:
        label = LABEL_NAME_TO_ID[lowered]
        return label, LABEL_MAP[label]
    numeric = re.fullmatch(r"([012])(?:\.0+)?", lowered)
    if numeric:
        label = int(numeric.group(1))
        return label, LABEL_MAP[label]
    raise TasteLabelError("AMBIGUOUS_LABEL", value)


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("TasteMolNet preparation requires RDKit.") from exc
    return Chem, MurckoScaffold


def _largest_organic_fragment(Chem: Any, molecule: Any) -> Any | None:
    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    organic = [
        fragment
        for fragment in fragments
        if any(int(atom.GetAtomicNum()) == 6 for atom in fragment.GetAtoms())
    ]
    if not organic:
        return None
    return max(
        organic,
        key=lambda fragment: (
            int(fragment.GetNumHeavyAtoms()),
            int(fragment.GetNumBonds()),
            str(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)),
        ),
    )


def _has_stereochemistry(Chem: Any, molecule: Any) -> bool:
    try:
        chiral_centers = Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
            includeCIP=True,
        )
    except TypeError:  # Older RDKit does not expose includeCIP.
        chiral_centers = Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
        )
    if chiral_centers:
        return True
    return any(str(bond.GetStereo()) != "STEREONONE" for bond in molecule.GetBonds())


def _murcko_scaffold(Chem: Any, MurckoScaffold: Any, molecule: Any) -> str:
    value = str(
        MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=True,
        )
        or ""
    )
    return value


def _normalize_chemical_row(
    *,
    raw_smiles: str,
    label: int,
    label_name: str,
    source_row_id: str,
    component_policy: str,
    max_chemlm_atoms: int,
    source_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    Chem, MurckoScaffold = _require_rdkit()
    audit = {
        "canonical_only_valid": False,
        "largest_organic_fragment_valid": False,
        "was_multicomponent": False,
        "canonical_smiles": None,
        "canonical_num_atoms": None,
        "canonical_molclr_compatible": False,
        "largest_organic_fragment_smiles": None,
        "largest_organic_fragment_num_atoms": None,
        "largest_organic_fragment_molclr_compatible": False,
    }

    def excluded(reason: str, detail: str = "") -> tuple[None, dict[str, Any], dict[str, Any]]:
        return None, {
            "source_row_id": source_row_id,
            "raw_smiles": raw_smiles,
            "raw_label": label_name,
            "canonical_smiles": "",
            "model_smiles": "",
            "normalized_label": label,
            "normalized_label_name": label_name,
            "exclusion_reason": reason,
            "exclusion_detail": detail,
        }, audit

    if not raw_smiles:
        return excluded("PARSE_FAILED", "empty_smiles")
    try:
        molecule = Chem.MolFromSmiles(raw_smiles, sanitize=False)
    except Exception as exc:
        return excluded("PARSE_FAILED", f"{type(exc).__name__}:{exc}")
    if molecule is None:
        return excluded("PARSE_FAILED", "rdkit_returned_none")
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        return excluded("SANITIZE_FAILED", f"{type(exc).__name__}:{exc}")
    canonical = str(
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    )
    fragment_count = int(len(Chem.GetMolFrags(molecule)))
    audit["canonical_only_valid"] = True
    audit["was_multicomponent"] = fragment_count > 1
    audit["canonical_smiles"] = canonical
    audit["canonical_num_atoms"] = int(molecule.GetNumAtoms())
    audit["canonical_molclr_compatible"] = all(
        int(atom.GetAtomicNum()) in SUPPORTED_ATOMIC_NUMBERS
        for atom in molecule.GetAtoms()
    ) and {
        str(bond.GetBondType()).upper() for bond in molecule.GetBonds()
    }.issubset({"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"})
    largest_organic = _largest_organic_fragment(Chem, molecule)
    audit["largest_organic_fragment_valid"] = largest_organic is not None
    if largest_organic is not None:
        largest_smiles = str(
            Chem.MolToSmiles(
                largest_organic,
                canonical=True,
                isomericSmiles=True,
            )
        )
        audit["largest_organic_fragment_smiles"] = largest_smiles
        audit["largest_organic_fragment_num_atoms"] = int(
            largest_organic.GetNumAtoms()
        )
        audit["largest_organic_fragment_molclr_compatible"] = all(
            int(atom.GetAtomicNum()) in SUPPORTED_ATOMIC_NUMBERS
            for atom in largest_organic.GetAtoms()
        ) and {
            str(bond.GetBondType()).upper() for bond in largest_organic.GetBonds()
        }.issubset({"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"})

    if component_policy == "canonical_only":
        if fragment_count != 1:
            return excluded(
                "MULTICOMPONENT_UNSUPPORTED",
                "canonical_only policy rejects dot-disconnected molecules",
            )
        model_molecule = molecule
    elif component_policy == "largest_organic_fragment":
        if largest_organic is None:
            return excluded("NO_ORGANIC_COMPONENT")
        model_molecule = largest_organic
    else:  # pragma: no cover - argparse and the public entrypoint validate it.
        raise ValueError(f"Unsupported component_policy={component_policy!r}.")

    model_smiles = str(
        Chem.MolToSmiles(model_molecule, canonical=True, isomericSmiles=True)
    )
    atomic_numbers = [int(atom.GetAtomicNum()) for atom in model_molecule.GetAtoms()]
    supported_atoms = bool(atomic_numbers) and all(
        value in SUPPORTED_ATOMIC_NUMBERS for value in atomic_numbers
    )
    if not supported_atoms:
        return excluded(
            "UNSUPPORTED_ATOM",
            json.dumps(sorted(set(atomic_numbers)), separators=(",", ":")),
        )
    bond_names = {str(bond.GetBondType()).upper() for bond in model_molecule.GetBonds()}
    molclr_compatible = bond_names.issubset(
        {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
    )
    if not molclr_compatible:
        return excluded("MOLCLR_UNSUPPORTED", ",".join(sorted(bond_names)))
    num_atoms = int(model_molecule.GetNumAtoms())
    chemlm_compatible = num_atoms <= int(max_chemlm_atoms) and len(model_smiles) <= 512
    if not chemlm_compatible:
        return excluded(
            "TOO_LARGE_FOR_CHEMLLM",
            f"num_atoms={num_atoms};smiles_length={len(model_smiles)}",
        )
    scaffold = _murcko_scaffold(Chem, MurckoScaffold, model_molecule)
    record = {
        "molecule_id": "TASTE_" + hashlib.sha256(
            model_smiles.encode("utf-8")
        ).hexdigest()[:20].upper(),
        "raw_smiles": raw_smiles,
        "canonical_smiles": canonical,
        "model_smiles": model_smiles,
        "label": int(label),
        "label_name": label_name,
        "scaffold": scaffold,
        "num_atoms": num_atoms,
        "num_bonds": int(model_molecule.GetNumBonds()),
        "formal_charge": int(
            sum(int(atom.GetFormalCharge()) for atom in model_molecule.GetAtoms())
        ),
        "fragment_count": fragment_count,
        "supported_atom_check": True,
        "molclr_compatible": True,
        "chemlm_compatible": True,
        "has_stereochemistry": _has_stereochemistry(Chem, model_molecule),
        "source": source_name,
        "source_row_id": source_row_id,
        "standardization_version": STANDARDIZATION_VERSION,
        "component_policy": component_policy,
        "split": "",
        "exclusion_reason": "",
    }
    return record, None, audit


def _conflict_and_duplicate_filter(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identity_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        canonical = str(record["canonical_smiles"])
        model = str(record["model_smiles"])
        identity_groups[("canonical_smiles", canonical)].append(index)
        if model != canonical:
            identity_groups[("model_smiles", model)].append(index)

    conflict_indices: set[int] = set()
    conflict_rows: list[dict[str, Any]] = []
    for (identity_type, identity), indices in sorted(identity_groups.items()):
        labels = sorted({int(records[index]["label"]) for index in indices})
        if len(labels) <= 1:
            continue
        conflict_indices.update(indices)
        conflict_rows.append(
            {
                "conflict_identity_type": identity_type,
                "conflict_identity": identity,
                "labels": json.dumps(labels, separators=(",", ":")),
                "label_names": json.dumps(
                    [LABEL_MAP[label] for label in labels],
                    separators=(",", ":"),
                ),
                "source_row_ids": json.dumps(
                    [str(records[index]["source_row_id"]) for index in indices],
                    separators=(",", ":"),
                ),
                "raw_smiles_values": json.dumps(
                    [str(records[index]["raw_smiles"]) for index in indices],
                    separators=(",", ":"),
                ),
            }
        )

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_model_smiles: dict[str, int] = {}
    for index, record in enumerate(records):
        if index in conflict_indices:
            excluded.append(
                {
                    "source_row_id": record["source_row_id"],
                    "raw_smiles": record["raw_smiles"],
                    "raw_label": record["label_name"],
                    "canonical_smiles": record["canonical_smiles"],
                    "model_smiles": record["model_smiles"],
                    "normalized_label": record["label"],
                    "normalized_label_name": record["label_name"],
                    "exclusion_reason": "CROSS_LABEL_DUPLICATE",
                    "exclusion_detail": "canonical/model identity has multiple labels",
                }
            )
            continue
        model_smiles = str(record["model_smiles"])
        previous = seen_model_smiles.get(model_smiles)
        if previous is not None:
            excluded.append(
                {
                    "source_row_id": record["source_row_id"],
                    "raw_smiles": record["raw_smiles"],
                    "raw_label": record["label_name"],
                    "canonical_smiles": record["canonical_smiles"],
                    "model_smiles": model_smiles,
                    "normalized_label": record["label"],
                    "normalized_label_name": record["label_name"],
                    "exclusion_reason": "DUPLICATE_SAME_LABEL",
                    "exclusion_detail": f"first_source_index={previous}",
                }
            )
            continue
        seen_model_smiles[model_smiles] = index
        kept.append(dict(record))
    return kept, excluded, conflict_rows


def filter_cross_label_conflicts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Public, side-effect-free duplicate/conflict filter used by audits/tests."""

    return _conflict_and_duplicate_filter([dict(record) for record in records])


def _component_strategy_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_labelled_rows: int,
) -> dict[str, Any]:
    modes = {
        "canonical_only": (
            "canonical_smiles",
            "canonical_num_atoms",
            "canonical_molclr_compatible",
        ),
        "largest_organic_fragment": (
            "largest_organic_fragment_smiles",
            "largest_organic_fragment_num_atoms",
            "largest_organic_fragment_molclr_compatible",
        ),
    }
    result: dict[str, Any] = {}
    for mode, (identity_field, atom_field, molclr_field) in modes.items():
        eligible = [row for row in rows if row.get(identity_field)]
        atom_counts = [int(row[atom_field]) for row in eligible]
        labels_by_identity: dict[str, set[int]] = defaultdict(set)
        for row in eligible:
            labels_by_identity[str(row[identity_field])].add(int(row["label"]))
        conflicts = sorted(
            identity for identity, labels in labels_by_identity.items() if len(labels) > 1
        )
        molclr_successes = sum(bool(row.get(molclr_field)) for row in eligible)
        result[mode] = {
            "eligible_rows": len(eligible),
            "retention_rate": (
                float(len(eligible) / input_labelled_rows)
                if input_labelled_rows
                else 0.0
            ),
            "cross_label_conflict_identities": len(conflicts),
            "cross_label_conflict_examples": conflicts[:20],
            "atom_count": {
                "min": min(atom_counts) if atom_counts else None,
                "mean": (
                    float(sum(atom_counts) / len(atom_counts)) if atom_counts else None
                ),
                "max": max(atom_counts) if atom_counts else None,
            },
            "molclr_compatible_rows": molclr_successes,
            "molclr_compatible_rate": (
                float(molclr_successes / len(eligible)) if eligible else 0.0
            ),
        }
    return result


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in ratios)
    if len(normalized) != len(SPLIT_NAMES):
        raise ValueError(f"Expected four split ratios, observed {normalized!r}.")
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("Split ratios must be finite and positive.")
    if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split ratios must sum to one: {normalized!r}.")
    return normalized


def _effective_scaffold(record: Mapping[str, Any]) -> str:
    scaffold = str(record.get("scaffold") or "").strip()
    if scaffold:
        return scaffold
    canonical = str(record["model_smiles"])
    return "ACYCLIC::" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def scaffold_disjoint_split(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign complete scaffold groups while balancing size and three classes."""

    ratios = _validate_ratios(split_ratios)
    if not records:
        raise ValueError("Cannot split an empty TasteMolNet dataset.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_record in records:
        record = dict(raw_record)
        label = int(record["label"])
        if label not in LABEL_MAP:
            raise ValueError(f"Unexpected TasteMolNet label: {label}.")
        groups[_effective_scaffold(record)].append(record)

    total_count = len(records)
    overall_labels = Counter(int(record["label"]) for record in records)
    targets_total = {
        split: total_count * ratio
        for split, ratio in zip(SPLIT_NAMES, ratios, strict=True)
    }
    targets_labels = {
        split: {
            label: overall_labels[label] * ratio
            for label in LABEL_MAP
        }
        for split, ratio in zip(SPLIT_NAMES, ratios, strict=True)
    }
    split_groups: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}
    split_counts = Counter({split: 0 for split in SPLIT_NAMES})
    split_labels = {
        split: Counter({label: 0 for label in LABEL_MAP}) for split in SPLIT_NAMES
    }

    def group_order(item: tuple[str, list[dict[str, Any]]]) -> tuple[Any, ...]:
        scaffold, members = item
        label_counts = Counter(int(member["label"]) for member in members)
        tie_hash = hashlib.sha256(f"{seed}\0{scaffold}".encode("utf-8")).hexdigest()
        return (-len(members), -max(label_counts.values()), tie_hash, scaffold)

    for scaffold, members in sorted(groups.items(), key=group_order):
        member_labels = Counter(int(member["label"]) for member in members)

        def assignment_score(split: str) -> tuple[float, float, int]:
            projected_counts = {
                name: split_counts[name] + (len(members) if name == split else 0)
                for name in SPLIT_NAMES
            }
            size_error = sum(
                (projected_counts[name] - targets_total[name]) ** 2
                for name in SPLIT_NAMES
            )
            class_error = 0.0
            for name in SPLIT_NAMES:
                for label in LABEL_MAP:
                    projected = split_labels[name][label]
                    if name == split:
                        projected += member_labels[label]
                    class_error += (projected - targets_labels[name][label]) ** 2
            overflow = sum(
                max(0.0, projected_counts[name] - targets_total[name]) ** 2
                for name in SPLIT_NAMES
            )
            fill_ratio = split_counts[split] / max(targets_total[split], 1.0)
            return (
                size_error + 1.5 * class_error + 4.0 * overflow,
                fill_ratio,
                SPLIT_NAMES.index(split),
            )

        selected_split = min(SPLIT_NAMES, key=assignment_score)
        split_groups[selected_split].append(scaffold)
        split_counts[selected_split] += len(members)
        split_labels[selected_split].update(member_labels)

    scaffold_to_split = {
        scaffold: split
        for split, scaffold_values in split_groups.items()
        for scaffold in scaffold_values
    }
    assigned: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        record["split"] = scaffold_to_split[_effective_scaffold(record)]
        assigned.append(record)
    assigned.sort(key=lambda row: (SPLIT_NAMES.index(str(row["split"])), str(row["molecule_id"])))
    audit = scaffold_overlap_audit(assigned)
    if not audit["passed"]:
        raise AssertionError("TasteMolNet scaffold split leakage was detected.")
    split_statistics = {
        split: {
            "rows": int(split_counts[split]),
            "scaffolds": len(split_groups[split]),
            "class_counts": {
                str(label): int(split_labels[split][label]) for label in LABEL_MAP
            },
        }
        for split in SPLIT_NAMES
    }
    return assigned, {
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "empty_scaffold_policy": "canonical_smiles_specific_acyclic_group",
        "split_statistics": split_statistics,
        "scaffold_overlap_audit": audit,
    }


def scaffold_overlap_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scaffold_sets = {
        split: {
            _effective_scaffold(record)
            for record in records
            if str(record.get("split")) == split
        }
        for split in SPLIT_NAMES
    }
    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            shared = sorted(scaffold_sets[left] & scaffold_sets[right])
            overlaps[f"{left}__{right}"] = shared
    smiles_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        smiles_to_splits[str(record["model_smiles"])].add(str(record["split"]))
    duplicate_smiles_leakage = {
        smiles: sorted(splits)
        for smiles, splits in smiles_to_splits.items()
        if len(splits) > 1
    }
    return {
        "passed": not any(overlaps.values()) and not duplicate_smiles_leakage,
        "scaffold_counts": {
            split: len(scaffold_sets[split]) for split in SPLIT_NAMES
        },
        "pairwise_overlaps": overlaps,
        "canonical_smiles_cross_split": duplicate_smiles_leakage,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_source_rows(
    source_csv: Path,
    *,
    smiles_column: str | None,
    label_column: str | None,
    id_column: str | None,
) -> tuple[list[dict[str, str]], DetectedSchema]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        schema = detect_schema(
            reader.fieldnames or (),
            smiles_column=smiles_column,
            label_column=label_column,
            id_column=id_column,
        )
        return [dict(row) for row in reader], schema


def prepare_tastemolnet_dataset(
    *,
    source_csv: str | Path,
    output_dir: str | Path,
    source_mode: str = "local",
    source_url: str | None = None,
    upstream_commit: str | None = None,
    license_id: str | None = None,
    license_reviewed: bool = False,
    smiles_column: str | None = None,
    label_column: str | None = None,
    id_column: str | None = None,
    component_policy: str = "largest_organic_fragment",
    split_seed: int = 7,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    max_chemlm_atoms: int = 128,
    require_all_classes_per_split: bool = True,
) -> dict[str, Any]:
    """Prepare one fresh TasteMolNet output root without remote access."""

    get_dataset_spec("tastemolnet", allow_historical=False)
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {SOURCE_MODES!r}.")
    if component_policy not in {"canonical_only", "largest_organic_fragment"}:
        raise ValueError("Unsupported TasteMolNet component policy.")
    if source_mode == "upstream_processed" and not upstream_commit:
        raise ValueError("upstream_processed mode requires an immutable upstream_commit.")
    if source_mode in {"upstream_processed", "official_supplement"} and not source_url:
        raise ValueError(f"{source_mode} mode requires source_url provenance.")
    if license_reviewed and not str(license_id or "").strip():
        raise ValueError("A reviewed license requires --license-id.")
    source = Path(source_csv).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"TasteMolNet source CSV must be a regular file: {source}")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"TasteMolNet output must be a fresh absent path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        raw_rows, schema = _read_source_rows(
            source,
            smiles_column=smiles_column,
            label_column=label_column,
            id_column=id_column,
        )
        if not raw_rows:
            raise TasteMolNetPreparationError("TasteMolNet source CSV has no data rows.")

        normalized_candidates: list[dict[str, Any]] = []
        excluded_rows: list[dict[str, Any]] = []
        component_audit_rows: list[dict[str, Any]] = []
        normalized_label_row_count = 0
        for row_index, raw_row in enumerate(raw_rows):
            source_row_value = (
                raw_row.get(schema.id_column, "") if schema.id_column is not None else ""
            )
            source_row_id = str(source_row_value or f"row:{row_index}").strip()
            raw_smiles = str(raw_row.get(schema.smiles_column) or "").strip()
            raw_label = raw_row.get(schema.label_column)
            try:
                label, label_name = normalize_taste_label(raw_label)
            except TasteLabelError as exc:
                excluded_rows.append(
                    {
                        "source_row_id": source_row_id,
                        "raw_smiles": raw_smiles,
                        "raw_label": "" if raw_label is None else str(raw_label),
                        "canonical_smiles": "",
                        "model_smiles": "",
                        "normalized_label": "",
                        "normalized_label_name": "",
                        "exclusion_reason": exc.reason,
                        "exclusion_detail": str(exc.value),
                    }
                )
                continue
            normalized_label_row_count += 1
            record, excluded, audit = _normalize_chemical_row(
                raw_smiles=raw_smiles,
                label=label,
                label_name=label_name,
                source_row_id=source_row_id,
                component_policy=component_policy,
                max_chemlm_atoms=max_chemlm_atoms,
                source_name=source_mode,
            )
            component_audit_rows.append({**audit, "label": label})
            if excluded is not None:
                excluded["raw_label"] = str(raw_label)
                excluded_rows.append(excluded)
            elif record is not None:
                normalized_candidates.append(record)

        clean_records, duplicate_exclusions, conflict_rows = filter_cross_label_conflicts(
            normalized_candidates
        )
        excluded_rows.extend(duplicate_exclusions)
        if not clean_records:
            raise TasteMolNetPreparationError(
                "No single-label, conflict-free TasteMolNet molecules remained."
            )
        assigned, split_summary = scaffold_disjoint_split(
            clean_records,
            seed=split_seed,
            split_ratios=split_ratios,
        )
        missing_split_classes = {
            split: [
                label
                for label in LABEL_MAP
                if int(split_summary["split_statistics"][split]["class_counts"][str(label)])
                == 0
            ]
            for split in SPLIT_NAMES
        }
        missing_split_classes = {
            split: labels
            for split, labels in missing_split_classes.items()
            if labels
        }
        if require_all_classes_per_split and missing_split_classes:
            raise TasteMolNetPreparationError(
                "Scaffold split lacks one or more TasteMolNet classes: "
                + json.dumps(missing_split_classes, sort_keys=True)
            )
        if len({str(row["molecule_id"]) for row in assigned}) != len(assigned):
            raise AssertionError("TasteMolNet stable molecule IDs are not unique.")

        _atomic_write_csv(stage / "tastemolnet_clean.csv", assigned, CLEAN_FIELDS)
        split_root = stage / "splits"
        for split in SPLIT_NAMES:
            _atomic_write_csv(
                split_root / f"{split}.csv",
                [row for row in assigned if row["split"] == split],
                CLEAN_FIELDS,
            )
        _atomic_write_csv(split_root / "excluded_rows.csv", excluded_rows, EXCLUDED_FIELDS)
        _atomic_write_csv(stage / "cross_label_conflicts.csv", conflict_rows, CONFLICT_FIELDS)

        label_counts = Counter(int(row["label"]) for row in assigned)
        exclusion_counts = Counter(str(row["exclusion_reason"]) for row in excluded_rows)
        split_statistics = {
            "total_clean_rows": len(assigned),
            "overall_class_counts": {
                str(label): int(label_counts[label]) for label in LABEL_MAP
            },
            "splits": split_summary["split_statistics"],
            "true_sweet_test_rows": sum(
                1 for row in assigned if row["split"] == "test" and int(row["label"]) == 1
            ),
            "correctly_classified_sweet_test_rows": None,
            "correctly_classified_note": "available only after the frozen GNN is trained",
        }
        split_manifest = {
            "schema_version": 1,
            "dataset": "tastemolnet",
            "task_type": "multiclass_graph_classification",
            "num_classes": 3,
            "label_map": {str(key): value for key, value in LABEL_MAP.items()},
            "source_label": 1,
            "source_label_name": "Sweet",
            "counterfactual_mode": "untargeted_flip",
            "seed": int(split_seed),
            "ratios": split_summary["ratios"],
            "empty_scaffold_policy": split_summary["empty_scaffold_policy"],
            "same_canonical_smiles_cross_split_forbidden": True,
            "scaffold_overlap_gate_passed": split_summary[
                "scaffold_overlap_audit"
            ]["passed"],
            "all_classes_present_per_split": not missing_split_classes,
            "missing_split_classes": missing_split_classes,
            "sparse_split_override_used": bool(
                missing_split_classes and not require_all_classes_per_split
            ),
        }
        _atomic_write_json(split_root / "split_manifest.json", split_manifest)
        _atomic_write_json(split_root / "split_statistics.json", split_statistics)
        _atomic_write_json(
            split_root / "scaffold_overlap_audit.json",
            split_summary["scaffold_overlap_audit"],
        )
        _atomic_write_json(
            stage / "component_strategy_audit.json",
            {
                "selected_policy": component_policy,
                "input_labelled_rows": normalized_label_row_count,
                "multicomponent_rows": sum(
                    bool(row["was_multicomponent"]) for row in component_audit_rows
                ),
                "strategies": _component_strategy_summary(
                    component_audit_rows,
                    input_labelled_rows=normalized_label_row_count,
                ),
                "comparison_note": (
                    "Both representations were audited; the selected policy is explicit "
                    "and no tautomer normalization, neutralization, or stereo removal ran."
                ),
            },
        )

        license_status = (
            f"REVIEWED:{str(license_id).strip()}"
            if license_reviewed
            else LICENSE_REVIEW_REQUIRED
        )
        provenance = {
            "schema_version": 1,
            "dataset": "tastemolnet",
            "source_mode": source_mode,
            "source_csv_path": str(source),
            "source_csv_sha256": sha256_file(source),
            "source_csv_bytes": int(source.stat().st_size),
            "source_url": source_url,
            "upstream_commit": upstream_commit,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "download_performed": False,
            "raw_data_copied_into_output": False,
            "raw_data_commit_allowed": bool(license_reviewed),
            "license_id": license_id,
            "license_status": license_status,
            "schema_detection": schema.to_dict(),
            "standardization_version": STANDARDIZATION_VERSION,
            "component_policy": component_policy,
            "tautomer_normalization": False,
            "neutralization": False,
            "stereochemistry_removed": False,
            "run_tastemolnet_heavy": False,
            "foundation_status": "READY_NOT_RUN",
        }
        _atomic_write_json(stage / "provenance_manifest.json", provenance)
        if license_status == LICENSE_REVIEW_REQUIRED:
            _atomic_write_text(
                stage / LICENSE_REVIEW_REQUIRED,
                "TasteMolNet source license requires human review before raw or processed "
                "data may be committed.\n",
            )
        summary = {
            "status": "READY_NOT_RUN",
            "dataset": "tastemolnet",
            "input_rows": len(raw_rows),
            "clean_rows": len(assigned),
            "excluded_rows": len(excluded_rows),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "cross_label_conflict_identities": len(conflict_rows),
            "label_counts": {str(label): int(label_counts[label]) for label in LABEL_MAP},
            "split_statistics": split_statistics["splits"],
            "scaffold_overlap_gate_passed": True,
            "all_classes_present_per_split": not missing_split_classes,
            "missing_split_classes": missing_split_classes,
            "license_status": license_status,
            "source_csv_sha256": provenance["source_csv_sha256"],
            "output_dir": str(destination),
            "run_tastemolnet": False,
        }
        _atomic_write_json(stage / "prepare_summary.json", summary)

        manifest_files = sorted(
            path for path in stage.rglob("*") if path.is_file() and path.name != "output_manifest.json"
        )
        output_manifest = {
            "schema_version": 1,
            "files": {
                str(path.relative_to(stage)): {
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
                for path in manifest_files
            },
        }
        output_manifest["manifest_digest"] = _stable_json_hash(output_manifest["files"])
        _atomic_write_json(stage / "output_manifest.json", output_manifest)
        os.replace(stage, destination)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parse_ratios(value: str) -> tuple[float, ...]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
        return _validate_ratios(ratios)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-csv",
        default=os.environ.get("TASTEMOLNET_CSV"),
        help="Existing local CSV; no network download is attempted.",
    )
    parser.add_argument(
        "--source-mode",
        choices=SOURCE_MODES,
        default=os.environ.get("TASTEMOLNET_SOURCE_MODE", "local"),
    )
    parser.add_argument("--source-url", default=os.environ.get("TASTEMOLNET_SOURCE_URL"))
    parser.add_argument(
        "--upstream-commit",
        default=os.environ.get("TASTEMOLNET_UPSTREAM_COMMIT"),
    )
    parser.add_argument("--license-id", default=os.environ.get("TASTEMOLNET_LICENSE_ID"))
    parser.add_argument("--license-reviewed", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="data/processed/tastemolnet",
    )
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument(
        "--component-policy",
        choices=("canonical_only", "largest_organic_fragment"),
        default="largest_organic_fragment",
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument(
        "--split-ratios",
        type=_parse_ratios,
        default=DEFAULT_SPLIT_RATIOS,
    )
    parser.add_argument("--max-chemlm-atoms", type=int, default=128)
    parser.add_argument(
        "--allow-sparse-smoke-splits",
        action="store_true",
        help="Allow a class-missing split only for tiny engineering fixtures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_csv:
        raise SystemExit(
            "TASTEMOLNET_CSV/--input-csv is required. The command is offline and "
            "will not invent or download a dataset."
        )
    summary = prepare_tastemolnet_dataset(
        source_csv=args.input_csv,
        output_dir=args.output_dir,
        source_mode=args.source_mode,
        source_url=args.source_url,
        upstream_commit=args.upstream_commit,
        license_id=args.license_id,
        license_reviewed=args.license_reviewed,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        id_column=args.id_column,
        component_policy=args.component_policy,
        split_seed=args.split_seed,
        split_ratios=args.split_ratios,
        max_chemlm_atoms=args.max_chemlm_atoms,
        require_all_classes_per_split=not args.allow_sparse_smoke_splits,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["license_status"] == LICENSE_REVIEW_REQUIRED:
        print(f"[{LICENSE_REVIEW_REQUIRED}]", flush=True)
    print("[TASTEMOLNET_FOUNDATION_READY_NOT_RUN]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
