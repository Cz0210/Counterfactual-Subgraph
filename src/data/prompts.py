"""Prompt builders aligned to the deletion-based counterfactual objective."""

from __future__ import annotations

from src.data.schemas import MoleculeRecord


_PROMPT_PREFIX = (
    "You are given a molecule SMILES. Output ONE connected substructure "
    "SMILES whose deletion is most likely to flip the molecule label.\n"
    "The output fragment must be a valid connected substructure of the "
    "molecule.\n"
    "Output SMILES only, no extra text."
)

_DECODED_CHEM_EXACT_PARENT_PROMPT_PREFIX = (
    "You are given a molecule SMILES. Output ONE connected substructure "
    "SMILES whose deletion is most likely to flip the molecule label.\n"
    "The fragment must be an exact connected substructure of the parent "
    "molecule.\n"
    "Do not invent atoms, rings, branches, or substituents.\n"
    "Prefer one functional group or one ring neighborhood from the parent.\n"
    "Output only one fragment SMILES, no extra text."
)


def build_counterfactual_prompt(
    record: MoleculeRecord,
    *,
    include_label: bool = False,
) -> str:
    """Construct the canonical text prompt for one generation example."""

    lines = [_PROMPT_PREFIX]
    if include_label:
        lines.append(f"ORIGINAL_LABEL: {record.label}")
    lines.extend(
        [
            f"MOLECULE_SMILES: {record.smiles}",
            "FRAGMENT_SMILES:",
        ]
    )
    return "\n".join(lines)


def build_exact_parent_substructure_prompt(
    record: MoleculeRecord,
    *,
    include_label: bool = False,
) -> str:
    """Construct a stricter decoded-chem PPO prompt for exact parent substructures."""

    lines = [_DECODED_CHEM_EXACT_PARENT_PROMPT_PREFIX]
    if include_label:
        lines.append(f"ORIGINAL_LABEL: {record.label}")
    lines.extend(
        [
            f"MOLECULE_SMILES: {record.smiles}",
            "FRAGMENT_SMILES:",
        ]
    )
    return "\n".join(lines)
