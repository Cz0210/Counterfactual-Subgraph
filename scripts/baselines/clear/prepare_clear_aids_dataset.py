#!/usr/bin/env python3
"""Prepare AIDS/HIV CSV data in the official CLEAR pickle format.

This script is intentionally project-owned. It does not modify CLEAR official
source code; it only writes runtime dataset pickles under
``baselines/clear_official/dataset/`` so the patched official loader can read
``dataset=aids``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_PATH = "outputs/hpc/baselines/clear/aids/dataset/clear_aids_dataset_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert data/raw/AIDS/HIV.csv into CLEAR GraphData pickles for dataset=aids."
    )
    parser.add_argument("--aids-csv", default="data/raw/AIDS/HIV.csv", help="Input AIDS/HIV CSV path.")
    parser.add_argument("--smiles-column", default="smiles", help="SMILES column in the input CSV.")
    parser.add_argument("--label-column", default="HIV_active", help="Binary label column in the input CSV.")
    parser.add_argument("--out-dir", default="baselines/clear_official/dataset", help="CLEAR dataset output dir.")
    parser.add_argument("--dataset", default="aids", help="CLEAR dataset name to write. Default: aids.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split/feature seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--exp-num", type=int, default=3, help="Number of CLEAR split repetitions.")
    parser.add_argument("--max-num-nodes", type=int, default=30, help="Drop molecules larger than this node cap.")
    parser.add_argument("--x-dim", type=int, default=11, help="Node feature dimension. Default matches OGBG export shape.")
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH, help="Summary JSON path.")
    parser.add_argument("--config", default=None, help="Accepted for Slurm/config compatibility; not used.")
    return parser.parse_args()


def require_runtime_imports(repo_root: Path) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required to prepare CLEAR AIDS pickles") from exc

    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("RDKit is required to parse AIDS/HIV SMILES") from exc

    clear_src = repo_root / "baselines" / "clear_official" / "src"
    sys.path.insert(0, str(clear_src.resolve()))
    try:
        from data_sampler import GraphData
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import CLEAR data_sampler.GraphData. "
            "Run from the project root inside the CLEAR/HPC Python environment."
        ) from exc

    return np, Chem, GraphData


def parse_binary_label(value: str) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0", "False", "false"}:
        return 0
    if text in {"1", "1.0", "True", "true"}:
        return 1
    try:
        number = float(text)
    except ValueError:
        return None
    if number == 0.0:
        return 0
    if number == 1.0:
        return 1
    return None


def atom_features(atom: Any, x_dim: int) -> list[float]:
    """Return atom descriptor slots after the first two CLEAR causal features."""
    descriptors = [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        max(-3.0, min(3.0, float(atom.GetFormalCharge()))) / 3.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass() / 200.0,
        1.0 if atom.GetSymbol() == "C" else 0.0,
        1.0 if atom.GetSymbol() not in {"C", "H"} else 0.0,
        atom.GetTotalValence() / 8.0,
    ]
    needed = max(0, x_dim - 2)
    if len(descriptors) < needed:
        descriptors.extend([0.0] * (needed - len(descriptors)))
    return descriptors[:needed]


def mol_to_clear_arrays(
    *,
    mol: Any,
    label: int,
    rng: Any,
    np: Any,
    x_dim: int,
) -> tuple[Any, Any, Any]:
    num_atoms = mol.GetNumAtoms()
    adj = np.eye(num_atoms, dtype=float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    u_value = float(rng.randint(0, 9))
    u = np.asarray([u_value], dtype=float)

    label_proxy = 0.2 + 0.6 * float(label) + rng.uniform(-0.03, 0.03)
    label_proxy = max(0.0, min(1.0, label_proxy))
    confounder = rng.uniform(0.15, 1.0) + 0.05 * float(label)
    confounder = max(0.0, min(1.0, confounder))

    features = []
    for atom in mol.GetAtoms():
        row = [confounder, label_proxy]
        row.extend(atom_features(atom, x_dim))
        if len(row) < x_dim:
            row.extend([0.0] * (x_dim - len(row)))
        features.append(row[:x_dim])

    return adj, np.asarray(features, dtype=float), u


def stratified_split_once(labels: list[int], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label_indices in by_label.values():
        indices = list(label_indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train.extend(indices[:n_train])
        val.extend(indices[n_train : n_train + n_val])
        test.extend(indices[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def make_splits(
    *,
    labels: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    exp_num: int,
    np: Any,
) -> tuple[list[Any], list[Any], list[Any], str]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []
    source = "manual_stratified_shuffle"
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        source = "sklearn_stratified_shuffle"
        labels_np = np.asarray(labels)
        indices = np.arange(len(labels_np))
        for exp_i in range(exp_num):
            split_seed = seed + exp_i
            train_split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=split_seed)
            train_idx, temp_idx = next(train_split.split(indices, labels_np))
            temp_labels = labels_np[temp_idx]
            val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
            val_split = StratifiedShuffleSplit(n_splits=1, train_size=val_fraction_of_temp, random_state=split_seed + 10000)
            val_rel, test_rel = next(val_split.split(temp_idx, temp_labels))
            idx_train_list.append(np.asarray(train_idx, dtype=int))
            idx_val_list.append(np.asarray(temp_idx[val_rel], dtype=int))
            idx_test_list.append(np.asarray(temp_idx[test_rel], dtype=int))
        return idx_train_list, idx_val_list, idx_test_list, source
    except Exception:
        pass

    for exp_i in range(exp_num):
        train, val, test = stratified_split_once(labels, train_ratio, val_ratio, seed + exp_i)
        idx_train_list.append(np.asarray(train, dtype=int))
        idx_val_list.append(np.asarray(val, dtype=int))
        idx_test_list.append(np.asarray(test, dtype=int))
    return idx_train_list, idx_val_list, idx_test_list, source


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    aids_csv = Path(args.aids_csv)
    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary_path)
    if not aids_csv.is_absolute():
        aids_csv = repo_root / aids_csv
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path

    np, Chem, GraphData = require_runtime_imports(repo_root)

    if not aids_csv.exists():
        raise FileNotFoundError(f"Missing AIDS/HIV CSV: {aids_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    adj_all = []
    features_all = []
    u_all = []
    labels_all = []
    smiles_all: list[str] = []
    row_indices: list[int] = []
    activity_all: list[str | None] = []
    num_atoms_all: list[int] = []
    invalid_smiles: list[dict[str, Any]] = []
    invalid_label_count = 0
    rdkit_valid_smiles_count = 0
    skipped_too_large: list[dict[str, Any]] = []
    label_distribution_raw: Counter[str] = Counter()
    label_distribution_used: Counter[str] = Counter()
    rng = random.Random(args.seed)

    with aids_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if args.smiles_column not in fieldnames:
            raise ValueError(f"SMILES column {args.smiles_column!r} not found in {aids_csv}. Columns: {fieldnames}")
        if args.label_column not in fieldnames:
            raise ValueError(f"Label column {args.label_column!r} not found in {aids_csv}. Columns: {fieldnames}")

        rows = list(reader)

    for row_idx, row in enumerate(rows):
        label = parse_binary_label(row.get(args.label_column, ""))
        if label is None:
            invalid_label_count += 1
            invalid_smiles.append({"row_index": row_idx, "smiles": row.get(args.smiles_column), "error": "invalid_label"})
            continue
        label_distribution_raw[str(label)] += 1

        smiles = (row.get(args.smiles_column) or "").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            invalid_smiles.append({"row_index": row_idx, "smiles": smiles, "error": "rdkit_parse_failed"})
            continue
        rdkit_valid_smiles_count += 1
        num_atoms = int(mol.GetNumAtoms())
        if num_atoms == 0:
            invalid_smiles.append({"row_index": row_idx, "smiles": smiles, "error": "empty_molecule"})
            continue
        if num_atoms > args.max_num_nodes:
            skipped_too_large.append({"row_index": row_idx, "smiles": smiles, "num_atoms": num_atoms})
            continue

        adj, features, u = mol_to_clear_arrays(
            mol=mol,
            label=label,
            rng=rng,
            np=np,
            x_dim=args.x_dim,
        )
        adj_all.append(adj)
        features_all.append(features)
        u_all.append(u)
        labels_all.append(np.asarray([float(label)], dtype=float))
        smiles_all.append(smiles)
        row_indices.append(row_idx)
        activity_all.append(row.get("activity"))
        num_atoms_all.append(num_atoms)
        label_distribution_used[str(label)] += 1

    if not adj_all:
        raise RuntimeError("No usable AIDS molecules remained after RDKit parsing and node-count filtering.")

    labels_flat = [int(float(label.reshape(-1)[0])) for label in labels_all]
    idx_train_list, idx_val_list, idx_test_list, split_source = make_splits(
        labels=labels_flat,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        exp_num=args.exp_num,
        np=np,
    )

    data = GraphData(adj_all, features_all, u_all, labels_all, args.max_num_nodes, padded=True)
    data.smiles_all = smiles_all
    data.csv_row_index_all = row_indices
    data.activity_all = activity_all
    data.num_atoms_all = num_atoms_all
    data.source_csv = str(aids_csv)
    data.smiles_column = args.smiles_column
    data.label_column = args.label_column
    data.clear_dataset_name = args.dataset

    import pickle

    full_pickle = out_dir / f"{args.dataset}_full.pickle"
    datasplit_pickle = out_dir / f"{args.dataset}_datasplit.pickle"
    with full_pickle.open("wb") as handle:
        pickle.dump({"data": data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with datasplit_pickle.open("wb") as handle:
        pickle.dump(
            {
                "idx_train_list": idx_train_list,
                "idx_val_list": idx_val_list,
                "idx_test_list": idx_test_list,
                "split_source": split_source,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary = {
        "input_csv": str(aids_csv),
        "smiles_column": args.smiles_column,
        "label_column": args.label_column,
        "dataset": args.dataset,
        "num_rows": len(rows),
        "num_valid_smiles": rdkit_valid_smiles_count,
        "invalid_smiles_count": len(invalid_smiles),
        "rdkit_parse_failed_count": len([item for item in invalid_smiles if item.get("error") == "rdkit_parse_failed"]),
        "invalid_label_count": invalid_label_count,
        "skipped_too_large_count": len(skipped_too_large),
        "num_used_graphs": len(adj_all),
        "label_distribution": dict(sorted(label_distribution_raw.items())),
        "used_label_distribution": dict(sorted(label_distribution_used.items())),
        "max_num_nodes": args.max_num_nodes,
        "observed_max_num_nodes": max(num_atoms_all),
        "x_dim": args.x_dim,
        "train_size": int(len(idx_train_list[0])),
        "val_size": int(len(idx_val_list[0])),
        "test_size": int(len(idx_test_list[0])),
        "exp_num": args.exp_num,
        "split_source": split_source,
        "seed": args.seed,
        "output_full_pickle": str(full_pickle),
        "output_datasplit_pickle": str(datasplit_pickle),
        "summary_path": str(summary_path),
    }
    write_json(summary_path, summary)

    print("[CLEAR_AIDS_PREP_DONE]")
    print(f"input_csv={aids_csv}")
    print(f"num_rows={len(rows)}")
    print(f"num_used_graphs={len(adj_all)}")
    print(f"invalid_smiles_count={len(invalid_smiles)}")
    print(f"skipped_too_large_count={len(skipped_too_large)}")
    print(f"full_pickle={full_pickle}")
    print(f"datasplit_pickle={datasplit_pickle}")
    print(f"summary_path={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
