"""Minimal single-sample inference helpers without training dependencies."""

from __future__ import annotations

from typing import Any

from src.chem import (
    ParsedMolecule,
    FragmentValidationResult,
    is_connected_fragment,
    is_parent_substructure,
    is_valid_capped_subgraph,
    is_rdkit_available,
    parse_smiles,
    validate_fragment_candidate,
)
from src.data import sample_random_aids_hiv_record
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.models import ChemLLMGenerator
from src.models.interfaces import GenerationRequest, GenerationResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


def _parsed_to_dict(parsed: ParsedMolecule) -> dict[str, Any]:
    return {
        "smiles": parsed.smiles,
        "parseable": parsed.parseable,
        "canonical_smiles": parsed.canonical_smiles,
        "atom_count": parsed.atom_count,
        "sanitized": parsed.sanitized,
        "contains_dummy_atoms": parsed.contains_dummy_atoms,
        "used_relaxed_sanitization": parsed.used_relaxed_sanitization,
        "failure_type": parsed.failure_type.value if parsed.failure_type else None,
        "failure_reason": parsed.failure_reason,
    }


def _validation_to_dict(validation: FragmentValidationResult) -> dict[str, Any]:
    return {
        "parseable": validation.parseable,
        "chemically_valid": validation.chemically_valid,
        "connected": validation.connected,
        "is_substructure": validation.is_substructure,
        "deletion_supported": validation.deletion_supported,
        "parent_parseable": validation.parent_parseable,
        "parent_chemically_valid": validation.parent_chemically_valid,
        "residual_smiles": validation.residual_smiles,
        "failure_types": [failure_type.value for failure_type in validation.failure_types],
        "failure_reasons": list(validation.failure_reasons),
    }


def _substructure_check(parent_smiles: str, fragment_smiles: str) -> bool:
    fragment = parse_smiles(
        fragment_smiles,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    if not fragment.sanitized:
        return False
    if fragment.contains_dummy_atoms:
        return is_valid_capped_subgraph(parent_smiles, fragment_smiles)
    return is_parent_substructure(parent_smiles, fragment_smiles)


def build_fragment_checks(parent_smiles: str, fragment_candidate: str) -> dict[str, Any]:
    """Build a consistent structural validation payload for one candidate."""

    parent_parse = parse_smiles(parent_smiles, sanitize=True, canonicalize=True)
    fragment_parse = parse_smiles(
        fragment_candidate,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=True,
    )
    connected = is_connected_fragment(fragment_candidate)
    is_substructure = _substructure_check(parent_smiles, fragment_candidate)
    validation = validate_fragment_candidate(parent_smiles, fragment_candidate)

    return {
        "parent_parse": _parsed_to_dict(parent_parse),
        "fragment_parse": _parsed_to_dict(fragment_parse),
        "connected": connected,
        "is_substructure": is_substructure,
        "validation": _validation_to_dict(validation),
    }


def _fragment_smiles_from_atoms(mol: object, atom_indices: tuple[int, ...]) -> str | None:
    if Chem is None:
        return None
    try:
        fragment_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(atom_indices),
            canonical=True,
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None
    candidate = fragment_smiles.strip()
    return candidate or None


def _iter_candidate_smiles(parent: ParsedMolecule) -> list[tuple[str, str]]:
    if not is_rdkit_available() or Chem is None or parent.mol is None:
        return []

    mol = parent.mol
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_candidate(strategy: str, atom_indices: tuple[int, ...]) -> None:
        candidate = _fragment_smiles_from_atoms(mol, atom_indices)
        if not candidate:
            return
        if candidate == parent.smiles or candidate == parent.canonical_smiles:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append((candidate, strategy))

    for bond in mol.GetBonds():
        atom_indices = tuple(
            sorted({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()})
        )
        if bond.GetBeginAtom().GetDegree() == 1 or bond.GetEndAtom().GetDegree() == 1:
            add_candidate("terminal_bond_fragment", atom_indices)
    for atom in mol.GetAtoms():
        if atom.GetDegree() <= 1 and not atom.GetIsAromatic():
            add_candidate("terminal_atom_fragment", (atom.GetIdx(),))
    for bond in mol.GetBonds():
        atom_indices = tuple(
            sorted({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()})
        )
        add_candidate("bond_fragment", atom_indices)
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            add_candidate("atom_fragment", (atom.GetIdx(),))

    return candidates


def propose_fragment_candidate(parent_smiles: str) -> tuple[str, str, tuple[str, ...]]:
    """Generate one heuristic fragment candidate without any trained model."""

    normalized_parent = str(parent_smiles).strip()
    parent = parse_smiles(normalized_parent, sanitize=True, canonicalize=True)
    fallback_candidate = parent.canonical_smiles or normalized_parent

    if not parent.sanitized or parent.mol is None:
        return (
            fallback_candidate,
            "parent_fallback",
            ("Parent could not be sanitized; using the parent SMILES as the fallback candidate.",),
        )

    best_valid_with_deletion: tuple[int, int, str, str] | None = None
    best_valid_any: tuple[int, int, str, str] | None = None
    evaluation_parent = parent.canonical_smiles or normalized_parent

    for candidate, strategy in _iter_candidate_smiles(parent):
        validation = validate_fragment_candidate(evaluation_parent, candidate)
        if not (
            validation.parseable
            and validation.chemically_valid
            and validation.connected
            and validation.is_substructure
        ):
            continue
        parsed_candidate = parse_smiles(candidate, sanitize=True, canonicalize=True)
        atom_count = parsed_candidate.atom_count or 9999
        ranking = (atom_count, len(candidate), candidate, strategy)
        if best_valid_any is None or ranking < best_valid_any:
            best_valid_any = ranking
        if validation.deletion_supported and (
            best_valid_with_deletion is None or ranking < best_valid_with_deletion
        ):
            best_valid_with_deletion = ranking

    if best_valid_with_deletion is not None:
        atom_count, _, candidate, strategy = best_valid_with_deletion
        return (
            candidate,
            strategy,
            (
                "Heuristic candidate selected from a small connected parent substructure.",
                f"candidate_atom_count={atom_count}",
                "deletion_supported=true",
            ),
        )

    if best_valid_any is not None:
        atom_count, _, candidate, strategy = best_valid_any
        return (
            candidate,
            strategy,
            (
                "Heuristic candidate selected from a structurally valid parent substructure.",
                f"candidate_atom_count={atom_count}",
                "deletion_supported=false",
            ),
        )

    return (
        fallback_candidate,
        "parent_fallback",
        ("No smaller valid connected substructure candidate was found; using the parent SMILES.",),
    )


def run_minimal_inference(
    parent_smiles: str,
    *,
    label: int | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """Run a minimal inference loop and return a structured result."""

    normalized_parent = str(parent_smiles).strip()
    record = MoleculeRecord(record_id="cli", smiles=normalized_parent, label=int(label or 0))
    prompt = build_counterfactual_prompt(record, include_label=label is not None)
    request = GenerationRequest(
        parent_smiles=record.smiles,
        label=label,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    fragment_candidate, strategy, notes = propose_fragment_candidate(normalized_parent)
    return {
        "mode": "minimal_heuristic_inference",
        "uses_trained_model": False,
        "parent_smiles": normalized_parent,
        "fragment_candidate": fragment_candidate,
        "candidate_strategy": strategy,
        "candidate_notes": list(notes),
        "generation_request": {
            "parent_smiles": request.parent_smiles,
            "label": request.label,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        },
        "prompt": prompt,
        "checks": build_fragment_checks(normalized_parent, fragment_candidate),
    }


def sample_inference_record_from_aids(
    dataset_path: str,
    *,
    seed: int | None = None,
    smiles_column: str = "smiles",
    label_column: str = "HIV_active",
    activity_column: str = "activity",
) -> MoleculeRecord:
    """Sample one real molecule from the configured local AIDS/HIV CSV file."""

    sampled = sample_random_aids_hiv_record(
        dataset_path,
        seed=seed,
        smiles_column=smiles_column,
        label_column=label_column,
        activity_column=activity_column,
    )
    return MoleculeRecord(
        record_id=sampled.record_id,
        smiles=sampled.smiles,
        label=sampled.label,
    )


def run_chemllm_inference(
    record: MoleculeRecord,
    *,
    generator: ChemLLMGenerator,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """Run local ChemLLM inference for one record and return a structured result."""

    request = GenerationRequest(
        parent_smiles=record.smiles,
        label=record.label,
        prompt=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generation = generator.generate(request)
    return _result_from_generation(
        record=record,
        generation=generation,
        uses_trained_model=True,
        mode="chemllm_local_inference",
    )


def _result_from_generation(
    *,
    record: MoleculeRecord,
    generation: GenerationResult,
    uses_trained_model: bool,
    mode: str,
) -> dict[str, Any]:
    prompt = generation.metadata.get("prompt") or build_counterfactual_prompt(
        record,
        include_label=True,
    )
    fragment_candidate = generation.fragment_smiles.strip()
    return {
        "mode": mode,
        "uses_trained_model": uses_trained_model,
        "parent_smiles": record.smiles,
        "parent_label": record.label,
        "record_id": record.record_id,
        "fragment_candidate": fragment_candidate,
        "generation_request": {
            "parent_smiles": record.smiles,
            "label": record.label,
            "max_new_tokens": generation.metadata.get("max_new_tokens"),
            "temperature": generation.metadata.get("temperature"),
            "top_p": generation.metadata.get("top_p"),
        },
        "prompt": prompt,
        "raw_generation": generation.raw_text,
        "checks": build_fragment_checks(record.smiles, fragment_candidate),
    }
