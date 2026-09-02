"""Deterministic train-only BRICS vocabulary and fixed proposer."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contracts import (
    LLMAblationContractError,
    LLMProposerVariant,
    ProposalRequest,
    ProposalResult,
    canonical_json_sha256,
)

try:  # imported lazily by the module; no model or oracle is involved
    from rdkit import Chem
    from rdkit.Chem import BRICS
except ImportError:  # pragma: no cover - exercised only without RDKit
    Chem = None
    BRICS = None


@dataclass(frozen=True, slots=True)
class TrainingMolecule:
    molecule_id: str
    smiles: str
    split: str
    label: int

    def __post_init__(self) -> None:
        if not str(self.molecule_id).strip() or not str(self.smiles).strip():
            raise LLMAblationContractError("BRICS input ID/SMILES must be non-empty")
        if str(self.split) != "train":
            raise LLMAblationContractError(
                f"BRICS vocabulary is train-only; found split={self.split!r}"
            )
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise LLMAblationContractError("BRICS input label must be an integer")


@dataclass(frozen=True, slots=True)
class BRICSFragmentRecord:
    """One train-derived, parent-matchable BRICS core."""

    fragment_smiles: str
    train_frequency: int
    atom_count: int
    source_parent_count: int
    vocabulary_rank: int

    def __post_init__(self) -> None:
        if not str(self.fragment_smiles).strip():
            raise LLMAblationContractError("BRICS fragment SMILES must be non-empty")
        for field in ("train_frequency", "atom_count", "source_parent_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LLMAblationContractError(f"{field} must be a positive integer")
        if (
            isinstance(self.vocabulary_rank, bool)
            or not isinstance(self.vocabulary_rank, int)
            or self.vocabulary_rank < 1
        ):
            raise LLMAblationContractError("vocabulary_rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class BRICSVocabulary:
    records: tuple[BRICSFragmentRecord, ...]
    train_molecule_count: int
    train_input_sha256: str
    schema_version: str = "llm_train_only_brics_vocabulary_v1"
    split_policy: str = "strict_train_only"
    ordering_policy: str = "train_frequency_desc_then_canonical_smiles"
    ranking_policy: str = "train_frequency_only_no_oracle"
    oracle_fields_read: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.records:
            raise LLMAblationContractError("BRICS vocabulary must be non-empty")
        fragments = tuple(record.fragment_smiles for record in self.records)
        expected = tuple(
            sorted(
                self.records,
                key=lambda record: (-record.train_frequency, record.fragment_smiles),
            )
        )
        if len(fragments) != len(set(fragments)) or self.records != expected:
            raise LLMAblationContractError(
                "BRICS records must be unique and ordered by train frequency then SMILES"
            )
        if tuple(record.vocabulary_rank for record in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise LLMAblationContractError("BRICS vocabulary ranks must be contiguous from one")
        if self.ranking_policy != "train_frequency_only_no_oracle" or self.oracle_fields_read:
            raise LLMAblationContractError("BRICS vocabulary may not use oracle ranking")

    @property
    def fragments(self) -> tuple[str, ...]:
        return tuple(record.fragment_smiles for record in self.records)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "records": [asdict(record) for record in self.records],
                "train_input_sha256": self.train_input_sha256,
                "split_policy": self.split_policy,
                "ordering_policy": self.ordering_policy,
                "ranking_policy": self.ranking_policy,
                "oracle_fields_read": list(self.oracle_fields_read),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [asdict(record) for record in self.records]
        payload["fragments"] = list(self.fragments)
        payload["oracle_fields_read"] = list(self.oracle_fields_read)
        payload["vocabulary_sha256"] = self.sha256
        return payload


def training_molecules_from_mappings(
    records: Iterable[Mapping[str, Any]],
) -> tuple[TrainingMolecule, ...]:
    """Project arbitrary rows onto the four allowed BRICS input fields.

    Extra columns, including predictions, probabilities, rewards, and oracle
    scores, are deliberately invisible to vocabulary construction.
    """

    molecules: list[TrainingMolecule] = []
    for row in records:
        raw_label = row.get("label")
        if isinstance(raw_label, bool):
            raise LLMAblationContractError("BRICS input label must not be bool")
        try:
            label = int(str(raw_label))
        except (TypeError, ValueError) as exc:
            raise LLMAblationContractError(f"invalid BRICS input label: {raw_label!r}") from exc
        molecules.append(
            TrainingMolecule(
                molecule_id=str(row.get("molecule_id") or row.get("parent_id") or ""),
                smiles=str(row.get("smiles") or row.get("parent_smiles") or ""),
                split=str(row.get("split") or ""),
                label=label,
            )
        )
    return tuple(molecules)


def _canonical_brics_core(fragment_mol: Any) -> tuple[str, int] | None:
    """Remove BRICS attachment dummies so entries can match real parents."""

    editable = Chem.RWMol(fragment_mol)
    dummy_indices = [
        atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0
    ]
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    core = editable.GetMol()
    if core.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(core)
    except Exception:
        return None
    components = Chem.GetMolFrags(core, asMols=True, sanitizeFrags=True)
    if len(components) != 1:
        return None
    canonical = Chem.MolToSmiles(
        components[0],
        canonical=True,
        isomericSmiles=True,
    )
    return canonical, int(components[0].GetNumHeavyAtoms())


def build_train_only_brics_vocabulary(
    molecules: Iterable[TrainingMolecule | Mapping[str, Any]],
) -> BRICSVocabulary:
    """Build a deterministic vocabulary without classifier/oracle access."""

    if Chem is None or BRICS is None:
        raise LLMAblationContractError("RDKit BRICS is required to build the vocabulary")

    materialized: list[TrainingMolecule] = []
    for item in molecules:
        if isinstance(item, TrainingMolecule):
            materialized.append(item)
        elif isinstance(item, Mapping):
            materialized.extend(training_molecules_from_mappings((item,)))
        else:
            raise LLMAblationContractError("BRICS inputs must be TrainingMolecule/mapping rows")
    if not materialized:
        raise LLMAblationContractError("BRICS vocabulary requires at least one train molecule")

    by_id: dict[str, TrainingMolecule] = {}
    for item in materialized:
        prior = by_id.get(item.molecule_id)
        if prior is not None and prior != item:
            raise LLMAblationContractError(
                f"conflicting duplicate BRICS molecule ID: {item.molecule_id}"
            )
        by_id[item.molecule_id] = item

    normalized_inputs: list[dict[str, Any]] = []
    frequencies: Counter[str] = Counter()
    source_parents: dict[str, set[str]] = defaultdict(set)
    atom_counts: dict[str, int] = {}
    for item in sorted(by_id.values(), key=lambda row: row.molecule_id):
        mol = Chem.MolFromSmiles(item.smiles)
        if mol is None:
            raise LLMAblationContractError(
                f"invalid train SMILES for BRICS molecule {item.molecule_id}"
            )
        canonical_parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        normalized_inputs.append(
            {
                "molecule_id": item.molecule_id,
                "canonical_smiles": canonical_parent,
                "split": "train",
                "label": item.label,
            }
        )
        try:
            broken = BRICS.BreakBRICSBonds(mol)
            raw_fragments = Chem.GetMolFrags(
                broken,
                asMols=True,
                sanitizeFrags=True,
            )
        except Exception as exc:  # pragma: no cover - defensive around RDKit
            raise LLMAblationContractError(
                f"RDKit BRICS decomposition failed for {item.molecule_id}"
            ) from exc
        for fragment_mol in raw_fragments:
            core = _canonical_brics_core(fragment_mol)
            if core is None:
                continue
            canonical, atom_count = core
            # An undecomposed whole molecule is not a BRICS vocabulary entry for
            # the deletion-fragment baseline.
            if canonical == canonical_parent:
                continue
            frequencies[canonical] += 1
            source_parents[canonical].add(item.molecule_id)
            atom_counts[canonical] = atom_count

    if not frequencies:
        raise LLMAblationContractError(
            "train molecules produced no proper BRICS fragments; no fallback is allowed"
        )
    input_sha = canonical_json_sha256({"train_molecules": normalized_inputs})
    ordered = sorted(frequencies, key=lambda value: (-frequencies[value], value))
    records = tuple(
        BRICSFragmentRecord(
            fragment_smiles=fragment,
            train_frequency=frequencies[fragment],
            atom_count=atom_counts[fragment],
            source_parent_count=len(source_parents[fragment]),
            vocabulary_rank=rank,
        )
        for rank, fragment in enumerate(ordered, start=1)
    )
    return BRICSVocabulary(
        records=records,
        train_molecule_count=len(normalized_inputs),
        train_input_sha256=input_sha,
    )


class BRICSFixedGenerator:
    """Attempt-matched deterministic proposer over one frozen vocabulary."""

    variant = LLMProposerVariant.BRICS_FIXED

    def __init__(self, vocabulary: BRICSVocabulary) -> None:
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> BRICSVocabulary:
        return self._vocabulary

    def generate(self, request: ProposalRequest) -> ProposalResult:
        if Chem is None:
            raise LLMAblationContractError("RDKit is required for BRICS parent matching")
        parent = Chem.MolFromSmiles(request.parent_smiles)
        if parent is None:
            raise LLMAblationContractError(
                f"invalid proposal parent SMILES: {request.parent_smiles!r}"
            )
        matching_records = tuple(
            record
            for record in self._vocabulary.records
            if parent.HasSubstructMatch(
                Chem.MolFromSmiles(record.fragment_smiles),
                useChirality=True,
            )
        )
        if request.attempt_index >= len(matching_records):
            return ProposalResult(
                variant=self.variant,
                fragment_smiles="",
                raw_text="",
                finish_reason="proposal_shortfall",
                metadata={
                    "vocabulary_sha256": self._vocabulary.sha256,
                    "matching_fragment_count": len(matching_records),
                    "proposal_shortfall": True,
                    "selection_policy": "parent_match_then_train_frequency",
                    "oracle_used": False,
                },
            )
        record = matching_records[request.attempt_index]
        fragment = record.fragment_smiles
        return ProposalResult(
            variant=self.variant,
            fragment_smiles=fragment,
            raw_text=fragment,
            finish_reason="deterministic_vocabulary_lookup",
            metadata={
                "vocabulary_sha256": self._vocabulary.sha256,
                "vocabulary_rank": record.vocabulary_rank,
                "train_frequency": record.train_frequency,
                "source_parent_count": record.source_parent_count,
                "matching_fragment_count": len(matching_records),
                "proposal_shortfall": False,
                "selection_policy": "parent_match_then_train_frequency",
                "oracle_used": False,
            },
        )


__all__ = [
    "BRICSFixedGenerator",
    "BRICSFragmentRecord",
    "BRICSVocabulary",
    "TrainingMolecule",
    "build_train_only_brics_vocabulary",
    "training_molecules_from_mappings",
]
