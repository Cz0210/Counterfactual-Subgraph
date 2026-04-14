"""Safe SMILES parsing and canonicalization hooks.

This layer now treats dummy atoms (``*``) as a first-class part of the
counterfactual fragment contract. The parser still insists on basic RDKit
soundness, but it can relax a narrow subset of sanitization steps for capped
fragments when aromaticity or hybridization perception around dummy atoms is
the only blocker.
"""

from __future__ import annotations

from typing import Protocol

from src.chem.types import ChemistryFailureType, ParsedMolecule

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None

if Chem is not None:  # pragma: no branch - import-time constant
    _RELAXABLE_DUMMY_SANITIZE_OPS = int(
        Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    )
else:  # pragma: no cover - RDKit unavailable
    _RELAXABLE_DUMMY_SANITIZE_OPS = 0


class MoleculeParser(Protocol):
    """Backend contract for RDKit-based parsing and canonicalization."""

    def parse(
        self,
        smiles: str,
        *,
        sanitize: bool = True,
        allow_capped_fragments: bool = True,
    ) -> ParsedMolecule:
        """Parse and optionally sanitize a SMILES string."""

    def canonicalize(self, smiles: str) -> str:
        """Return a canonical SMILES representation when available."""


def is_rdkit_available() -> bool:
    """Return whether the RDKit backend is available."""

    return Chem is not None


def _build_failure(
    smiles: object,
    *,
    failure_type: ChemistryFailureType,
    failure_reason: str,
) -> ParsedMolecule:
    text = smiles if isinstance(smiles, str) else repr(smiles)
    return ParsedMolecule(
        smiles=text,
        parseable=False,
        sanitized=False,
        failure_type=failure_type,
        failure_reason=failure_reason,
        mol=None,
    )


def _mol_contains_dummy_atoms(mol: object | None) -> bool:
    if Chem is None or mol is None:
        return False

    try:
        return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return False


def _sanitize_flag_name(flag: object) -> str:
    name = getattr(flag, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(flag)


def sanitize_molecule(
    mol: object,
    *,
    allow_capped_fragments: bool = True,
) -> tuple[object | None, bool, tuple[str, ...], str | None]:
    """Sanitize an already-parsed RDKit molecule.

    Returns a tuple of:
    1. sanitized molecule or ``None`` when sanitization failed;
    2. whether relaxed dummy-aware sanitization was used;
    3. the skipped sanitize op names, if any;
    4. a failure reason when sanitization failed.
    """

    if Chem is None:  # pragma: no cover - guarded by caller
        return None, False, (), "RDKit is not installed."

    base_mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(base_mol)
        return base_mol, False, (), None
    except Exception as exc:
        if not allow_capped_fragments or not _mol_contains_dummy_atoms(base_mol):
            return None, False, (), str(exc)

        remaining_ops = int(Chem.SanitizeFlags.SANITIZE_ALL)
        skipped_ops: list[str] = []
        max_relaxed_steps = bin(_RELAXABLE_DUMMY_SANITIZE_OPS).count("1")

        for _ in range(max_relaxed_steps):
            candidate = Chem.Mol(mol)
            failed_op = Chem.SanitizeMol(
                candidate,
                sanitizeOps=remaining_ops,
                catchErrors=True,
            )
            failed_value = int(failed_op)
            if failed_value == int(Chem.SanitizeFlags.SANITIZE_NONE):
                return candidate, True, tuple(skipped_ops), None
            if failed_value == 0:
                return candidate, True, tuple(skipped_ops), None
            if failed_value & _RELAXABLE_DUMMY_SANITIZE_OPS == 0:
                return None, False, tuple(skipped_ops), str(exc)
            remaining_ops &= ~failed_value
            skipped_ops.append(_sanitize_flag_name(failed_op))

        return None, False, tuple(skipped_ops), str(exc)


def parse_smiles(
    smiles: str,
    *,
    sanitize: bool = True,
    canonicalize: bool = True,
    allow_capped_fragments: bool = True,
) -> ParsedMolecule:
    """Parse a SMILES string and return structured failure information."""

    if not isinstance(smiles, str):
        return _build_failure(
            smiles,
            failure_type=ChemistryFailureType.INVALID_INPUT_TYPE,
            failure_reason="Expected SMILES to be a string.",
        )

    normalized = smiles.strip()
    if not normalized:
        return _build_failure(
            smiles,
            failure_type=ChemistryFailureType.EMPTY_SMILES,
            failure_reason="SMILES string is empty after stripping whitespace.",
        )

    if Chem is None:
        return _build_failure(
            normalized,
            failure_type=ChemistryFailureType.RDKIT_UNAVAILABLE,
            failure_reason="RDKit is required for chemistry utilities but is not installed.",
        )

    try:
        mol = Chem.MolFromSmiles(normalized, sanitize=False)
    except Exception as exc:  # pragma: no cover - depends on RDKit internals
        return _build_failure(
            normalized,
            failure_type=ChemistryFailureType.PARSE_FAILED,
            failure_reason=f"RDKit raised while parsing SMILES: {exc}",
        )

    if mol is None:
        return _build_failure(
            normalized,
            failure_type=ChemistryFailureType.PARSE_FAILED,
            failure_reason="RDKit could not parse the SMILES string.",
        )

    atom_count = mol.GetNumAtoms()
    contains_dummy_atoms = _mol_contains_dummy_atoms(mol)
    if not sanitize:
        canonical_smiles = None
        if canonicalize:
            try:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                canonical_smiles = None
        return ParsedMolecule(
            smiles=normalized,
            parseable=True,
            canonical_smiles=canonical_smiles,
            atom_count=atom_count,
            sanitized=False,
            contains_dummy_atoms=contains_dummy_atoms,
            mol=mol,
        )

    sanitized_mol, used_relaxed_sanitization, skipped_ops, failure_reason = sanitize_molecule(
        mol,
        allow_capped_fragments=allow_capped_fragments,
    )
    if sanitized_mol is None:
        skipped_detail = ""
        if skipped_ops:
            skipped_detail = (
                " Dummy-aware fallback skipped sanitize ops: "
                + ", ".join(skipped_ops)
                + "."
            )
        return ParsedMolecule(
            smiles=normalized,
            parseable=True,
            canonical_smiles=None,
            atom_count=atom_count,
            sanitized=False,
            contains_dummy_atoms=contains_dummy_atoms,
            failure_type=ChemistryFailureType.SANITIZE_FAILED,
            failure_reason=(
                "RDKit parsed the SMILES but sanitization failed: "
                f"{failure_reason}{skipped_detail}"
            ),
            mol=mol,
        )

    canonical_smiles = None
    if canonicalize:
        canonical_smiles = Chem.MolToSmiles(sanitized_mol, canonical=True)

    return ParsedMolecule(
        smiles=normalized,
        parseable=True,
        canonical_smiles=canonical_smiles,
        atom_count=sanitized_mol.GetNumAtoms(),
        sanitized=True,
        contains_dummy_atoms=contains_dummy_atoms,
        used_relaxed_sanitization=used_relaxed_sanitization,
        mol=sanitized_mol,
    )


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical SMILES or raise a clear error."""

    parsed = parse_smiles(smiles, sanitize=True, canonicalize=True)
    if not parsed.sanitized or parsed.canonical_smiles is None:
        detail = parsed.failure_reason or "unknown chemistry parsing failure"
        raise ValueError(f"Could not canonicalize SMILES: {detail}")
    return parsed.canonical_smiles
