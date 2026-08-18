"""BACE GlobalGCE native-output classification and action adaptation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FULL_COUNTERFACTUAL_GRAPH = "full_counterfactual_graph"
LHS_RHS_RULE = "lhs_rhs_rule"
DELETION_FRAGMENT = "deletion_fragment"
FULLGRAPH_ACTION_ADAPTER = "connected_sanitized_fullgraph_counterfactual_v1"


class GlobalGCEActionAdapterError(ValueError):
    """Raised when native candidate semantics cannot be established safely."""


@dataclass(frozen=True, slots=True)
class AdaptedGlobalGCECandidate:
    rank: int
    candidate_id: str
    candidate_smiles: str
    canonical_smiles: str
    native_output_type: str
    action_adapter: str
    source_row: Mapping[str, Any]


def _text(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def infer_globalgce_native_output_type(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise GlobalGCEActionAdapterError("GlobalGCE candidate input is empty.")
    kinds: set[str] = set()
    for row in rows:
        lhs = _text(row, "lhs_smiles", "lhs_smarts", "rule_lhs")
        rhs = _text(row, "rhs_smiles", "rhs_smarts", "rule_rhs")
        action_type = _text(row, "action_type", "native_output_type").lower()
        fragment = _text(row, "fragment_smiles", "deletion_fragment")
        fullgraph = _text(
            row,
            "fullgraph_smiles",
            "counterfactual_smiles",
            "candidate_smiles",
            "canonical_smiles",
            "smiles",
        )
        if lhs or rhs:
            if not lhs or not rhs:
                raise GlobalGCEActionAdapterError(
                    "GlobalGCE rule row must contain both LHS and RHS."
                )
            kinds.add(LHS_RHS_RULE)
        elif action_type in {"deletion", "delete", DELETION_FRAGMENT} or fragment:
            if not fragment:
                raise GlobalGCEActionAdapterError(
                    "GlobalGCE deletion row has no fragment SMILES."
                )
            kinds.add(DELETION_FRAGMENT)
        elif fullgraph:
            kinds.add(FULL_COUNTERFACTUAL_GRAPH)
        else:
            raise GlobalGCEActionAdapterError(
                "GlobalGCE row has no rule, deletion-fragment, or fullgraph field."
            )
    if len(kinds) != 1:
        raise GlobalGCEActionAdapterError(
            f"Mixed GlobalGCE native candidate semantics are forbidden: {sorted(kinds)}"
        )
    return next(iter(kinds))


def _canonical_connected_smiles(value: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - HPC chemistry dependency
        raise RuntimeError("GlobalGCE action adaptation requires RDKit.") from exc
    molecule = Chem.MolFromSmiles(value, sanitize=False)
    if molecule is None:
        raise GlobalGCEActionAdapterError(f"Invalid GlobalGCE fullgraph SMILES: {value!r}")
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise GlobalGCEActionAdapterError(
            f"GlobalGCE fullgraph sanitization failed: {value!r}"
        ) from exc
    if molecule.GetNumAtoms() <= 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise GlobalGCEActionAdapterError(
            f"GlobalGCE fullgraph must be nonempty and connected: {value!r}"
        )
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    if not canonical or "." in canonical:
        raise GlobalGCEActionAdapterError(
            f"GlobalGCE canonical fullgraph is disconnected: {canonical!r}"
        )
    return canonical


def adapt_globalgce_fullgraph_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int | None = None
) -> list[AdaptedGlobalGCECandidate]:
    kind = infer_globalgce_native_output_type(rows)
    if kind != FULL_COUNTERFACTUAL_GRAPH:
        raise GlobalGCEActionAdapterError(
            f"BACE v7 fullgraph adapter cannot consume native type {kind!r}."
        )
    if expected_count is not None and len(rows) != int(expected_count):
        raise GlobalGCEActionAdapterError(
            f"Expected {expected_count} GlobalGCE candidates, found {len(rows)}."
        )
    result: list[AdaptedGlobalGCECandidate] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    for index, row in enumerate(rows, start=1):
        rank = int(row.get("rank") or index)
        if rank != index:
            raise GlobalGCEActionAdapterError(
                f"GlobalGCE ranks must be contiguous 1..K; row {index} has {rank}."
            )
        candidate_id = _text(row, "candidate_id")
        if not candidate_id or candidate_id in seen_ids:
            raise GlobalGCEActionAdapterError(
                f"GlobalGCE candidate IDs must be nonempty and unique: {candidate_id!r}."
            )
        raw = _text(
            row,
            "fullgraph_smiles",
            "counterfactual_smiles",
            "candidate_smiles",
            "canonical_smiles",
            "smiles",
        )
        canonical = _canonical_connected_smiles(raw)
        if canonical in seen_smiles:
            raise GlobalGCEActionAdapterError(
                f"Duplicate canonical GlobalGCE fullgraph: {canonical!r}."
            )
        seen_ids.add(candidate_id)
        seen_smiles.add(canonical)
        result.append(
            AdaptedGlobalGCECandidate(
                rank=rank,
                candidate_id=candidate_id,
                candidate_smiles=canonical,
                canonical_smiles=canonical,
                native_output_type=kind,
                action_adapter=FULLGRAPH_ACTION_ADAPTER,
                source_row=dict(row),
            )
        )
    return result


def read_candidate_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def assert_nonzero_fullgraph_applicability(
    rows: Sequence[Mapping[str, Any]], *, expected_pairs: int | None = None
) -> dict[str, int]:
    if expected_pairs is not None and len(rows) != int(expected_pairs):
        raise GlobalGCEActionAdapterError(
            f"Expected {expected_pairs} GlobalGCE pairs, found {len(rows)}."
        )
    applicable = sum(
        str(row.get("match") or "").strip().lower() in {"1", "true", "yes"}
        for row in rows
    )
    valid = sum(
        str(row.get("delete_valid") or "").strip().lower() in {"1", "true", "yes"}
        for row in rows
    )
    strict_flip = sum(
        str(row.get("teacher_strict_flip") or row.get("cf_flip") or "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
        for row in rows
    )
    if rows and applicable == 0:
        raise GlobalGCEActionAdapterError(
            "SCHEMA_OR_ACTION_ADAPTER_FAILURE: every GlobalGCE calibration pair "
            "has applicable=0."
        )
    return {
        "pair_count": len(rows),
        "applicable_count": applicable,
        "valid_count": valid,
        "strict_flip_count": strict_flip,
    }
