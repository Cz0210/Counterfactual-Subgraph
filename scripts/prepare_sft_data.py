#!/usr/bin/env python3
"""Prepare balanced SFT data with capped molecular subgraphs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - depends on HPC runtime
    raise SystemExit(
        "RDKit is required for scripts/prepare_sft_data.py. "
        "Please run this script inside the smiles_pip118 conda environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "raw" / "AIDS" / "HIV.csv"
DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "sft_train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "sft_val.jsonl"
DEFAULT_NEGATIVE_SAMPLE_SIZE = 3700
DEFAULT_TARGET_EXAMPLES = 5000
DEFAULT_TRAIN_SIZE = 4500
DEFAULT_VAL_SIZE = 500
INSTRUCTION_TEMPLATE = (
    "[System]\n"
    "Generate a valid, chemically capped subgraph for the following parent molecule. "
    "Output only the fragment SMILES.\n\n"
    "[Input]\n"
    "PARENT_SMILES: {parent_smiles}\n\n"
    "[Output]\n"
)
RELAXABLE_DUMMY_SANITIZE_OPS = int(
    Chem.SanitizeFlags.SANITIZE_KEKULIZE
    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    | Chem.SanitizeFlags.SANITIZE_ADJUSTHS
)


@dataclass(frozen=True, slots=True)
class SFTExample:
    """One minimal instruction/output training example."""

    instruction: str
    output: str
    label: int

    def to_json(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "output": self.output,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the raw HIV CSV file.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_TRAIN_JSONL),
        help="Path to the train JSONL output.",
    )
    parser.add_argument(
        "--val-output",
        default=str(DEFAULT_VAL_JSONL),
        help="Path to the validation JSONL output.",
    )
    parser.add_argument(
        "--negative-sample-size",
        type=int,
        default=DEFAULT_NEGATIVE_SAMPLE_SIZE,
        help="Number of valid negative molecules to sample into the base pool.",
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=DEFAULT_TARGET_EXAMPLES,
        help="Target number of successful SFT examples to build.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="Number of records to save into sft_train.jsonl.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help="Number of records to save into sft_val.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling, shuffling, and fragment selection.",
    )
    parser.add_argument(
        "--min-real-atoms",
        type=int,
        default=4,
        help="Minimum number of non-dummy atoms required in a fragment.",
    )
    parser.add_argument(
        "--max-cut-attempts",
        type=int,
        default=24,
        help="How many random cut attempts to try per molecule.",
    )
    return parser


def canonicalize_parent_smiles(smiles: str) -> str | None:
    """Return a canonical parent SMILES if the molecule is valid."""

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


def count_real_atoms(mol: object) -> int:
    """Count non-dummy atoms in a molecule."""

    return sum(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())


def is_connected_molecule(mol: object) -> bool:
    """Return whether a molecule has exactly one connected component."""

    try:
        return len(Chem.GetMolFrags(mol)) == 1 and mol.GetNumAtoms() > 0
    except Exception:
        return False


def find_acyclic_single_bonds(mol: object) -> list[int]:
    """Return indices of non-ring single bonds."""

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
    """Generate one capped fragment by randomly cutting 1 to 2 acyclic single bonds."""

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


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a list of JSON objects to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_instruction(parent_smiles: str) -> str:
    """Render the exact minimal instruction template."""

    return INSTRUCTION_TEMPLATE.format(parent_smiles=parent_smiles)


def load_and_filter_valid_records(csv_path: Path) -> pd.DataFrame:
    """Load the HIV CSV and keep only RDKit-valid parent molecules."""

    dataframe = pd.read_csv(csv_path)
    required_columns = {"smiles", "HIV_active"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    normalized = dataframe.loc[:, ["smiles", "HIV_active"]].copy()
    normalized["smiles"] = normalized["smiles"].astype(str).str.strip()
    normalized["HIV_active"] = pd.to_numeric(normalized["HIV_active"], errors="coerce")
    normalized = normalized[normalized["smiles"] != ""]
    normalized = normalized[normalized["HIV_active"].isin([0, 1])]
    normalized["HIV_active"] = normalized["HIV_active"].astype(int)
    normalized.reset_index(drop=True, inplace=True)

    valid_rows: list[dict[str, object]] = []
    iterator = tqdm(
        normalized.itertuples(index=False),
        total=len(normalized),
        desc="Filtering valid SMILES",
    )
    for row in iterator:
        try:
            canonical_smiles = canonicalize_parent_smiles(row.smiles)
        except Exception:
            canonical_smiles = None
        if canonical_smiles is None:
            continue
        valid_rows.append(
            {
                "parent_smiles": canonical_smiles,
                "HIV_active": int(row.HIV_active),
            }
        )

    return pd.DataFrame(valid_rows, columns=["parent_smiles", "HIV_active"])


def sample_balanced_base_pool(
    valid_records: pd.DataFrame,
    *,
    negative_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep all positives and sample negatives for the base pool."""

    positive_records = valid_records[valid_records["HIV_active"] == 1].copy()
    negative_records = valid_records[valid_records["HIV_active"] == 0].copy()
    if positive_records.empty:
        raise ValueError("No valid positive molecules were found after RDKit filtering.")
    if len(negative_records) < negative_sample_size:
        raise ValueError(
            "Not enough valid negative molecules to satisfy the requested sampling size. "
            f"Required {negative_sample_size}, found {len(negative_records)}."
        )

    sampled_negative_records = negative_records.sample(
        n=negative_sample_size,
        random_state=seed,
        replace=False,
    )
    remaining_negative_records = negative_records.drop(sampled_negative_records.index)
    base_pool = pd.concat(
        [positive_records, sampled_negative_records],
        ignore_index=True,
    )
    base_pool = base_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    remaining_negative_records = remaining_negative_records.sample(
        frac=1.0,
        random_state=seed + 1,
    ).reset_index(drop=True)
    return base_pool, positive_records.reset_index(drop=True), remaining_negative_records


def build_examples_from_pool(
    records: pd.DataFrame,
    rng: random.Random,
    *,
    min_real_atoms: int,
    max_cut_attempts: int,
    description: str,
) -> tuple[list[SFTExample], int]:
    """Convert a dataframe of molecules into SFT examples."""

    examples: list[SFTExample] = []
    failed_records = 0
    iterator = tqdm(
        records.itertuples(index=False),
        total=len(records),
        desc=description,
    )
    for row in iterator:
        fragment_smiles: str | None
        try:
            fragment_smiles = generate_capped_fragment(
                str(row.parent_smiles),
                rng,
                min_real_atoms=min_real_atoms,
                max_cut_attempts=max_cut_attempts,
            )
        except Exception:
            fragment_smiles = None

        if not fragment_smiles:
            failed_records += 1
            continue

        examples.append(
            SFTExample(
                instruction=build_instruction(str(row.parent_smiles)),
                output=fragment_smiles,
                label=int(row.HIV_active),
            )
        )
    return examples, failed_records


def backfill_examples(
    initial_examples: list[SFTExample],
    initial_failed_records: int,
    remaining_negative_records: pd.DataFrame,
    rng: random.Random,
    *,
    target_examples: int,
    min_real_atoms: int,
    max_cut_attempts: int,
) -> tuple[list[SFTExample], int, int]:
    """Fill any missing examples with additional unused negatives."""

    examples = list(initial_examples)
    failed_records = initial_failed_records
    refill_attempts = 0
    if len(examples) >= target_examples:
        return examples[:target_examples], failed_records, refill_attempts

    iterator = tqdm(
        remaining_negative_records.itertuples(index=False),
        total=len(remaining_negative_records),
        desc="Backfilling failed examples",
    )
    for row in iterator:
        if len(examples) >= target_examples:
            break

        refill_attempts += 1
        fragment_smiles: str | None
        try:
            fragment_smiles = generate_capped_fragment(
                str(row.parent_smiles),
                rng,
                min_real_atoms=min_real_atoms,
                max_cut_attempts=max_cut_attempts,
            )
        except Exception:
            fragment_smiles = None

        if not fragment_smiles:
            failed_records += 1
            continue

        examples.append(
            SFTExample(
                instruction=build_instruction(str(row.parent_smiles)),
                output=fragment_smiles,
                label=int(row.HIV_active),
            )
        )
    return examples, failed_records, refill_attempts


def compute_label_ratio(examples: list[SFTExample]) -> tuple[int, int, float, float]:
    """Return counts and ratios for negatives and positives."""

    negative_count = sum(example.label == 0 for example in examples)
    positive_count = sum(example.label == 1 for example in examples)
    total = len(examples)
    if total == 0:
        return negative_count, positive_count, 0.0, 0.0
    return (
        negative_count,
        positive_count,
        negative_count / total,
        positive_count / total,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.train_size + args.val_size != args.target_examples:
        raise SystemExit(
            "--train-size + --val-size must equal --target-examples "
            f"({args.train_size} + {args.val_size} != {args.target_examples})."
        )

    input_csv = Path(args.input_csv).expanduser().resolve()
    train_output = Path(args.train_output).expanduser().resolve()
    val_output = Path(args.val_output).expanduser().resolve()
    rng = random.Random(args.seed)

    valid_records = load_and_filter_valid_records(input_csv)
    if valid_records.empty:
        raise SystemExit("No valid molecules remained after RDKit filtering.")

    base_pool, positive_records, remaining_negative_records = sample_balanced_base_pool(
        valid_records,
        negative_sample_size=args.negative_sample_size,
        seed=args.seed,
    )
    if len(base_pool) != args.target_examples:
        print(
            "Warning: base pool size does not equal the target example count. "
            f"base_pool={len(base_pool)}, target={args.target_examples}. "
            "The script will continue and try to backfill from unused negatives."
        )

    initial_examples, initial_failed_records = build_examples_from_pool(
        base_pool,
        rng,
        min_real_atoms=args.min_real_atoms,
        max_cut_attempts=args.max_cut_attempts,
        description="Generating capped fragments",
    )
    all_examples, failed_records, refill_attempts = backfill_examples(
        initial_examples,
        initial_failed_records,
        remaining_negative_records,
        rng,
        target_examples=args.target_examples,
        min_real_atoms=args.min_real_atoms,
        max_cut_attempts=args.max_cut_attempts,
    )
    if len(all_examples) < args.target_examples:
        raise SystemExit(
            "Could not build enough successful SFT examples. "
            f"Built {len(all_examples)} examples, expected {args.target_examples}."
        )

    all_examples = all_examples[: args.target_examples]
    rng.shuffle(all_examples)
    train_examples = all_examples[: args.train_size]
    val_examples = all_examples[args.train_size : args.train_size + args.val_size]

    write_jsonl(train_output, [example.to_json() for example in train_examples])
    write_jsonl(val_output, [example.to_json() for example in val_examples])

    overall_neg, overall_pos, overall_neg_ratio, overall_pos_ratio = compute_label_ratio(all_examples)
    train_neg, train_pos, train_neg_ratio, train_pos_ratio = compute_label_ratio(train_examples)
    val_neg, val_pos, val_neg_ratio, val_pos_ratio = compute_label_ratio(val_examples)
    base_positive_count = int((base_pool["HIV_active"] == 1).sum())
    base_negative_count = int((base_pool["HIV_active"] == 0).sum())
    total_attempts = len(base_pool) + refill_attempts
    success_rate = len(all_examples) / total_attempts if total_attempts else 0.0

    print("SFT data preparation completed.")
    print(f"Input CSV: {input_csv}")
    print(f"Valid molecules: {len(valid_records)}")
    print(
        "Valid class counts: "
        f"positive={len(positive_records)}, "
        f"negative={int((valid_records['HIV_active'] == 0).sum())}"
    )
    print(
        "Base pool class counts: "
        f"positive={base_positive_count}, "
        f"negative={base_negative_count}, "
        f"total={len(base_pool)}"
    )
    print(
        "Fragment generation stats: "
        f"successful={len(all_examples)}, "
        f"failed={failed_records}, "
        f"refill_attempts={refill_attempts}, "
        f"success_rate={success_rate:.4f}"
    )
    print(
        "Overall label ratio: "
        f"positive={overall_pos} ({overall_pos_ratio:.4f}), "
        f"negative={overall_neg} ({overall_neg_ratio:.4f})"
    )
    print(
        "Train label ratio: "
        f"positive={train_pos} ({train_pos_ratio:.4f}), "
        f"negative={train_neg} ({train_neg_ratio:.4f})"
    )
    print(
        "Validation label ratio: "
        f"positive={val_pos} ({val_pos_ratio:.4f}), "
        f"negative={val_neg} ({val_neg_ratio:.4f})"
    )
    print(f"Saved train JSONL: {train_output}")
    print(f"Saved val JSONL: {val_output}")


if __name__ == "__main__":
    main()
