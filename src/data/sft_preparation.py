"""Balanced SFT data preparation helpers for capped and core-only targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import random
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional local dependency for data building
    pd = None

from src.chem import match_core_fragment_to_parent, normalize_core_fragment
from src.utils.io import write_jsonl

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on runtime environment
    Chem = None


LEGACY_CAPPED_INSTRUCTION_TEMPLATE = (
    "[System]\n"
    "Generate a valid, chemically capped subgraph for the following parent molecule. "
    "Output only the fragment SMILES.\n\n"
    "[Input]\n"
    "PARENT_SMILES: {parent_smiles}\n\n"
    "[Output]\n"
)

CORE_SFT_PROMPT_TEMPLATE = (
    "[System]\n"
    "You are a chemistry assistant. Output ONLY one valid connected substructure "
    "SMILES of the input molecule. Do not output dummy atoms such as '*'. "
    "No extra words, no explanations, no quotes.\n\n"
    "[User]\n"
    "SMILES: {parent_smiles}\n"
    "Return ONE connected substructure as a valid SMILES fragment. "
    "Do not use dummy atom '*'.\n\n"
    "[Assistant]\n"
)

RELAXABLE_DUMMY_SANITIZE_OPS = int(
    Chem.SanitizeFlags.SANITIZE_KEKULIZE
    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    | Chem.SanitizeFlags.SANITIZE_ADJUSTHS
) if Chem is not None else 0


@dataclass(frozen=True, slots=True)
class BalancedSamplingResult:
    """Balanced base-pool sampling outcome before fragment generation."""

    valid_positive_count: int
    valid_negative_count: int
    selected_positive_count: int
    selected_negative_count: int
    base_records: Any
    refill_negative_records: Any


@dataclass(frozen=True, slots=True)
class PreparedSFTExample:
    """One SFT example plus stable metadata."""

    sample_id: str
    graph_id: str
    parent_smiles: str
    label: int
    prompt: str
    response: str
    meta: dict[str, Any]

    @property
    def instruction(self) -> str:
        return self.prompt

    @property
    def output(self) -> str:
        return self.response

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "graph_id": self.graph_id,
            "smiles": self.parent_smiles,
            "label": int(self.label),
            "prompt": self.prompt,
            "response": self.response,
            "instruction": self.prompt,
            "output": self.response,
            "task_type": "concept_smiles_core_sft",
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    """Dataset-build audit counters shared by CLI and tests."""

    successful_examples: int
    failed_fragment_records: int
    refill_records_attempted: int
    total_generation_attempts: int
    total_raw_candidates: int
    dropped_samples: int
    parseable_core_candidates: int
    substructure_core_candidates: int
    full_parent_candidates: int
    too_small_candidates: int
    drop_by_failure_tag: dict[str, int] = field(default_factory=dict)


def build_sft_instruction(
    parent_smiles: str,
    *,
    target_format: str = "core",
) -> str:
    """Render the SFT prompt for either capped or core-only targets."""

    normalized_parent = str(parent_smiles).strip()
    if str(target_format).strip().lower() == "capped":
        return LEGACY_CAPPED_INSTRUCTION_TEMPLATE.format(parent_smiles=normalized_parent)
    return CORE_SFT_PROMPT_TEMPLATE.format(parent_smiles=normalized_parent)


def load_hiv_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw HIV CSV."""

    _require_pandas()
    return pd.read_csv(Path(csv_path).expanduser().resolve())


def filter_valid_hiv_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid parent SMILES and binary labels."""

    _require_pandas()
    required_columns = {"smiles", "HIV_active"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    normalized = dataframe.loc[:, ["smiles", "HIV_active"]].copy()
    normalized["source_row_index"] = list(range(len(normalized)))
    normalized["smiles"] = normalized["smiles"].astype(str).str.strip()
    normalized["HIV_active"] = pd.to_numeric(normalized["HIV_active"], errors="coerce")
    normalized = normalized[normalized["smiles"] != ""]
    normalized = normalized[normalized["HIV_active"].isin([0, 1])]
    normalized["HIV_active"] = normalized["HIV_active"].astype(int)
    normalized.reset_index(drop=True, inplace=True)

    valid_rows: list[dict[str, Any]] = []
    for row in normalized.itertuples(index=False):
        canonical_smiles = canonicalize_parent_smiles(row.smiles)
        if canonical_smiles is None:
            continue
        valid_rows.append(
            {
                "source_row_index": int(row.source_row_index),
                "parent_smiles": canonical_smiles,
                "HIV_active": int(row.HIV_active),
            }
        )

    return pd.DataFrame(
        valid_rows,
        columns=["source_row_index", "parent_smiles", "HIV_active"],
    )


def build_balanced_candidate_pool(
    valid_records: pd.DataFrame,
    *,
    total_examples: int,
    seed: int,
) -> BalancedSamplingResult:
    """Keep all positives when possible and fill the remainder with negatives."""

    _require_pandas()
    positive_records = valid_records[valid_records["HIV_active"] == 1].copy()
    negative_records = valid_records[valid_records["HIV_active"] == 0].copy()
    if positive_records.empty:
        raise ValueError("No valid positive molecules remained after RDKit filtering.")
    if negative_records.empty:
        raise ValueError("No valid negative molecules remained after RDKit filtering.")

    rng = random.Random(seed)
    selected_positives = positive_records.copy()
    if len(selected_positives) > total_examples:
        selected_positives = selected_positives.sample(
            n=total_examples,
            random_state=seed,
            replace=False,
        )

    negative_quota = max(0, int(total_examples) - len(selected_positives))
    if len(negative_records) < negative_quota:
        raise ValueError(
            "Not enough valid negative molecules to satisfy the requested example count. "
            f"Required {negative_quota}, found {len(negative_records)}."
        )

    selected_negatives = (
        negative_records.sample(
            n=negative_quota,
            random_state=seed,
            replace=False,
        )
        if negative_quota > 0
        else negative_records.iloc[0:0].copy()
    )
    refill_negatives = negative_records.drop(selected_negatives.index)

    base_records = pd.concat(
        [selected_positives, selected_negatives],
        ignore_index=True,
    )
    base_records = base_records.sample(
        frac=1.0,
        random_state=rng.randint(0, 10**9),
    ).reset_index(drop=True)
    refill_negatives = refill_negatives.sample(
        frac=1.0,
        random_state=rng.randint(0, 10**9),
    ).reset_index(drop=True)

    return BalancedSamplingResult(
        valid_positive_count=len(positive_records),
        valid_negative_count=len(negative_records),
        selected_positive_count=len(selected_positives),
        selected_negative_count=len(selected_negatives),
        base_records=base_records,
        refill_negative_records=refill_negatives,
    )


def prepare_balanced_sft_examples(
    valid_records: pd.DataFrame,
    *,
    total_examples: int,
    seed: int,
    target_format: str = "core",
    show_progress: bool = False,
    min_real_atoms: int = 4,
    max_cut_attempts: int = 24,
    fragment_builder: Callable[[str, random.Random], str | None] | None = None,
) -> tuple[list[PreparedSFTExample], PreparationSummary]:
    """Generate balanced SFT examples and keep an audit summary."""

    sampling = build_balanced_candidate_pool(
        valid_records,
        total_examples=total_examples,
        seed=seed,
    )
    rng = random.Random(seed)
    builder = fragment_builder or (
        lambda parent_smiles, local_rng: generate_capped_fragment(
            parent_smiles,
            local_rng,
            min_real_atoms=min_real_atoms,
            max_cut_attempts=max_cut_attempts,
        )
    )

    prepared_examples: list[PreparedSFTExample] = []
    drop_counter: Counter[str] = Counter()
    failed_fragment_records = 0
    refill_attempts = 0
    total_generation_attempts = 0
    total_raw_candidates = 0
    parseable_core_candidates = 0
    substructure_core_candidates = 0
    full_parent_candidates = 0
    too_small_candidates = 0

    base_iterator = (
        sampling.base_records.itertuples(index=False)
        if show_progress
        else sampling.base_records.itertuples(index=False)
    )
    for row in base_iterator:
        total_generation_attempts += 1
        maybe_example, failure_tag, audit_counts = _prepare_single_example(
            source_row_index=int(row.source_row_index),
            parent_smiles=str(row.parent_smiles),
            label=int(row.HIV_active),
            rng=rng,
            builder=builder,
            target_format=target_format,
        )
        total_raw_candidates += audit_counts["has_raw_candidate"]
        parseable_core_candidates += audit_counts["parseable_core_candidate"]
        substructure_core_candidates += audit_counts["substructure_core_candidate"]
        full_parent_candidates += audit_counts["full_parent_candidate"]
        too_small_candidates += audit_counts["too_small_candidate"]
        if maybe_example is not None:
            prepared_examples.append(maybe_example)
            continue
        failed_fragment_records += 1
        drop_counter[failure_tag] += 1

    for row in sampling.refill_negative_records.itertuples(index=False):
        if len(prepared_examples) >= total_examples:
            break
        total_generation_attempts += 1
        refill_attempts += 1
        maybe_example, failure_tag, audit_counts = _prepare_single_example(
            source_row_index=int(row.source_row_index),
            parent_smiles=str(row.parent_smiles),
            label=int(row.HIV_active),
            rng=rng,
            builder=builder,
            target_format=target_format,
        )
        total_raw_candidates += audit_counts["has_raw_candidate"]
        parseable_core_candidates += audit_counts["parseable_core_candidate"]
        substructure_core_candidates += audit_counts["substructure_core_candidate"]
        full_parent_candidates += audit_counts["full_parent_candidate"]
        too_small_candidates += audit_counts["too_small_candidate"]
        if maybe_example is not None:
            prepared_examples.append(maybe_example)
            continue
        failed_fragment_records += 1
        drop_counter[failure_tag] += 1

    summary = PreparationSummary(
        successful_examples=len(prepared_examples),
        failed_fragment_records=failed_fragment_records,
        refill_records_attempted=refill_attempts,
        total_generation_attempts=total_generation_attempts,
        total_raw_candidates=total_raw_candidates,
        dropped_samples=sum(drop_counter.values()),
        parseable_core_candidates=parseable_core_candidates,
        substructure_core_candidates=substructure_core_candidates,
        full_parent_candidates=full_parent_candidates,
        too_small_candidates=too_small_candidates,
        drop_by_failure_tag=dict(sorted(drop_counter.items())),
    )
    return prepared_examples[:total_examples], summary


def split_examples(
    examples: list[PreparedSFTExample],
    *,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[PreparedSFTExample], list[PreparedSFTExample]]:
    """Shuffle deterministically and split into train/validation examples."""

    if train_size + val_size > len(examples):
        raise ValueError(
            f"Requested train+val={train_size + val_size} exceeds available examples={len(examples)}."
        )
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size : train_size + val_size]


def label_ratio(examples: list[PreparedSFTExample]) -> dict[int, float]:
    """Return class ratios keyed by label."""

    if not examples:
        return {0: 0.0, 1: 0.0}
    total = float(len(examples))
    negative_count = sum(example.label == 0 for example in examples)
    positive_count = sum(example.label == 1 for example in examples)
    return {
        0: negative_count / total,
        1: positive_count / total,
    }


def save_sft_jsonl(path: str | Path, examples: list[PreparedSFTExample]) -> None:
    """Write SFT examples to JSONL."""

    write_jsonl(path, (example.to_json() for example in examples))


def build_core_sft_audit_payload(
    *,
    examples: list[PreparedSFTExample],
    summary: PreparationSummary,
    train_output: str | Path,
    val_output: str | Path,
) -> dict[str, Any]:
    """Build the core-only dataset audit payload requested by the user."""

    atom_counts = [
        int(example.meta.get("core_atom_count") or 0)
        for example in examples
        if int(example.meta.get("core_atom_count") or 0) > 0
    ]
    atom_ratios = [
        float(example.meta.get("atom_ratio"))
        for example in examples
        if example.meta.get("atom_ratio") is not None
    ]
    dummy_in_response_count = sum("*" in example.response for example in examples)

    total_raw_candidates = max(1, int(summary.total_raw_candidates))
    return {
        "train_output": str(Path(train_output).expanduser().resolve()),
        "val_output": str(Path(val_output).expanduser().resolve()),
        "total_generation_attempts": int(summary.total_generation_attempts),
        "total_raw_candidates": int(summary.total_raw_candidates),
        "kept_samples": len(examples),
        "dropped_samples": int(summary.dropped_samples),
        "drop_by_failure_tag": dict(summary.drop_by_failure_tag),
        "parse_rate": summary.parseable_core_candidates / total_raw_candidates,
        "substructure_rate": summary.substructure_core_candidates / total_raw_candidates,
        "avg_atom_count": (sum(atom_counts) / len(atom_counts)) if atom_counts else 0.0,
        "avg_atom_ratio": (sum(atom_ratios) / len(atom_ratios)) if atom_ratios else 0.0,
        "too_small_rate_atoms_le_2": summary.too_small_candidates / total_raw_candidates,
        "too_small_rate": summary.too_small_candidates / total_raw_candidates,
        "full_parent_rate": summary.full_parent_candidates / total_raw_candidates,
        "dummy_in_response_count": int(dummy_in_response_count),
    }


def save_audit_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def canonicalize_parent_smiles(smiles: str) -> str | None:
    """Return a canonical parent SMILES if the molecule is valid."""

    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip(), sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def contains_dummy_atoms(mol: object) -> bool:
    """Return whether a molecule contains any dummy atoms."""

    if Chem is None:
        return False
    try:
        return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    except Exception:
        return False


def clear_dummy_atom_isotopes(mol: object) -> None:
    """Strip RDKit's default isotope labels from dummy atoms."""

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope():
            atom.SetIsotope(0)


def sanitize_capped_fragment(mol: object) -> object | None:
    """Sanitize a fragment, allowing limited fallback around dummy atoms."""

    if Chem is None:
        return None
    try:
        base_mol = Chem.Mol(mol)
        Chem.SanitizeMol(base_mol)
        return base_mol
    except Exception:
        if not contains_dummy_atoms(mol):
            return None

    remaining_ops = int(Chem.SanitizeFlags.SANITIZE_ALL)
    max_relaxed_steps = max(bin(RELAXABLE_DUMMY_SANITIZE_OPS).count("1"), 1)
    for _ in range(max_relaxed_steps):
        candidate = Chem.Mol(mol)
        try:
            failed_op = Chem.SanitizeMol(
                candidate,
                sanitizeOps=remaining_ops,
                catchErrors=True,
            )
        except Exception:
            return None

        failed_value = int(failed_op)
        if failed_value == 0 or failed_value == int(Chem.SanitizeFlags.SANITIZE_NONE):
            return candidate
        if failed_value & RELAXABLE_DUMMY_SANITIZE_OPS == 0:
            return None
        remaining_ops &= ~failed_value
    return None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required for SFT data preparation but is not installed."
        )


def count_real_atoms(mol: object) -> int:
    """Count non-dummy atoms in a molecule."""

    return sum(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())


def is_connected_molecule(mol: object) -> bool:
    """Return whether a molecule has exactly one connected component."""

    if Chem is None:
        return False
    try:
        return len(Chem.GetMolFrags(mol)) == 1 and mol.GetNumAtoms() > 0
    except Exception:
        return False


def find_acyclic_single_bonds(mol: object) -> list[int]:
    """Return indices of non-ring single bonds."""

    if Chem is None:
        return []
    bond_indices: list[int] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        bond_indices.append(int(bond.GetIdx()))
    return bond_indices


def fragment_to_capped_smiles(
    fragment_mol: object,
    *,
    min_real_atoms: int,
) -> str | None:
    """Convert a cut fragment to a valid capped fragment SMILES."""

    if Chem is None:
        return None
    try:
        working_fragment = Chem.Mol(fragment_mol)
        clear_dummy_atom_isotopes(working_fragment)
        sanitized_fragment = sanitize_capped_fragment(working_fragment)
        if sanitized_fragment is None:
            return None
        clear_dummy_atom_isotopes(sanitized_fragment)
        if not is_connected_molecule(sanitized_fragment):
            return None
        if count_real_atoms(sanitized_fragment) < min_real_atoms:
            return None
        if not contains_dummy_atoms(sanitized_fragment):
            return None
        fragment_smiles = Chem.MolToSmiles(sanitized_fragment, canonical=True)
    except Exception:
        return None

    if not fragment_smiles or "." in fragment_smiles or "*" not in fragment_smiles:
        return None
    return fragment_smiles


def generate_capped_fragment(
    parent_smiles: str,
    rng: random.Random,
    *,
    min_real_atoms: int,
    max_cut_attempts: int,
) -> str | None:
    """Generate one capped fragment by cutting 1-2 acyclic single bonds."""

    if Chem is None:
        return None
    try:
        parent_mol = Chem.MolFromSmiles(parent_smiles, sanitize=False)
        if parent_mol is None:
            return None
        Chem.SanitizeMol(parent_mol)
    except Exception:
        return None

    bond_indices = find_acyclic_single_bonds(parent_mol)
    if not bond_indices:
        return None

    max_cuts = min(2, len(bond_indices))
    for _ in range(max_cut_attempts):
        cut_count = 1 if max_cuts == 1 else rng.choice((1, 2))
        try:
            selected_bonds = tuple(sorted(rng.sample(bond_indices, cut_count)))
        except ValueError:
            continue

        try:
            fragmented = Chem.FragmentOnBonds(
                Chem.Mol(parent_mol),
                list(selected_bonds),
                addDummies=True,
            )
            clear_dummy_atom_isotopes(fragmented)
            fragment_mols = Chem.GetMolFrags(
                fragmented,
                asMols=True,
                sanitizeFrags=False,
            )
        except Exception:
            continue

        valid_fragments: list[str] = []
        seen_fragments: set[str] = set()
        for fragment_mol in fragment_mols:
            fragment_smiles = fragment_to_capped_smiles(
                fragment_mol,
                min_real_atoms=min_real_atoms,
            )
            if not fragment_smiles or fragment_smiles in seen_fragments:
                continue
            seen_fragments.add(fragment_smiles)
            valid_fragments.append(fragment_smiles)

        if valid_fragments:
            return rng.choice(valid_fragments)
    return None


def _prepare_single_example(
    *,
    source_row_index: int,
    parent_smiles: str,
    label: int,
    rng: random.Random,
    builder: Callable[[str, random.Random], str | None],
    target_format: str,
) -> tuple[PreparedSFTExample | None, str, dict[str, int]]:
    raw_fragment = builder(parent_smiles, rng)
    audit_counts = {
        "has_raw_candidate": 0,
        "parseable_core_candidate": 0,
        "substructure_core_candidate": 0,
        "full_parent_candidate": 0,
        "too_small_candidate": 0,
    }
    if not raw_fragment:
        return None, "raw_fragment_missing", audit_counts

    audit_counts["has_raw_candidate"] = 1
    target_mode = str(target_format).strip().lower()
    if target_mode == "capped":
        prompt = build_sft_instruction(parent_smiles, target_format="capped")
        example = PreparedSFTExample(
            sample_id=f"sft_capped_{source_row_index:07d}",
            graph_id=str(source_row_index),
            parent_smiles=parent_smiles,
            label=int(label),
            prompt=prompt,
            response=str(raw_fragment).strip(),
            meta={
                "raw_fragment": str(raw_fragment).strip(),
                "core_fragment": None,
                "target_format": "capped_with_dummy",
                "strategy": "acyclic_single_bond_cut",
                "dummy_removed": False,
            },
        )
        return example, "ok", audit_counts

    normalized = normalize_core_fragment(raw_fragment, keep_largest_component=True)
    if normalized.core_parse_ok:
        audit_counts["parseable_core_candidate"] = 1
    if not normalized.core_parse_ok or not normalized.core_fragment_smiles:
        failure_tag = normalized.failure_tag or "sanitize_failed"
        return None, failure_tag, audit_counts

    match = match_core_fragment_to_parent(
        parent_smiles,
        normalized.core_fragment_smiles,
    )
    if not match.matched:
        return None, match.reason or "not_substructure", audit_counts
    audit_counts["substructure_core_candidate"] = 1

    if match.full_parent:
        audit_counts["full_parent_candidate"] = 1
        return None, "full_parent", audit_counts
    if normalized.core_atom_count <= 2:
        audit_counts["too_small_candidate"] = 1
        return None, "too_small", audit_counts

    response = str(normalized.core_fragment_smiles).strip()
    prompt = build_sft_instruction(parent_smiles, target_format="core")
    strategy = "acyclic_single_bond_cut"
    if normalized.kept_largest_component:
        strategy += "_largest_component"
    if normalized.raw_has_dummy:
        strategy += "_dummy_removed"

    example = PreparedSFTExample(
        sample_id=f"sft_v3_core_{source_row_index:07d}",
        graph_id=str(source_row_index),
        parent_smiles=parent_smiles,
        label=int(label),
        prompt=prompt,
        response=response,
        meta={
            "raw_fragment": str(raw_fragment).strip(),
            "core_fragment": response,
            "target_format": "core_no_dummy",
            "strategy": strategy,
            "parent_atom_indices": list(match.match_atom_indices),
            "attachment_points": list(match.attachment_points),
            "boundary_bonds": match.boundary_bonds_as_dicts(),
            "explanation_fragment_with_dummy": match.explanation_fragment_with_dummy,
            "dummy_removed": bool(normalized.raw_has_dummy),
            "raw_has_dummy": bool(normalized.raw_has_dummy),
            "raw_dummy_count": int(normalized.raw_dummy_count),
            "kept_largest_component": bool(normalized.kept_largest_component),
            "component_count_before_selection": int(
                normalized.component_count_before_selection
            ),
            "core_atom_count": int(normalized.core_atom_count),
            "atom_ratio": match.atom_ratio,
            "match_reason": match.reason,
        },
    )
    return example, "ok", audit_counts


__all__ = [
    "BalancedSamplingResult",
    "PreparedSFTExample",
    "PreparationSummary",
    "build_balanced_candidate_pool",
    "build_core_sft_audit_payload",
    "build_sft_instruction",
    "filter_valid_hiv_records",
    "generate_capped_fragment",
    "label_ratio",
    "load_hiv_dataframe",
    "prepare_balanced_sft_examples",
    "save_audit_json",
    "save_sft_jsonl",
    "split_examples",
]
