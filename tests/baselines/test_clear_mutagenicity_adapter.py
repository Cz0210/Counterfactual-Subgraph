from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from src.baselines.clear_mutagenicity_adapter import (
    DEFAULT_EXPECTED_COUNTS,
    ClearMutagenicityCodecError,
    build_clear_dataset_payload,
    build_train_schema,
    encode_atom_feature,
    load_phase_a_cohorts,
    molecule_to_clear_graph,
    project_binary_graph_to_molecule,
    required_category_probe_selection,
    round_trip_source_graph,
    run_codec_probe,
    validate_validation_schema,
)


TRAIN_POSITIVE = (
    ("tp_br", "CCBr"),
    ("tp_cl", "CCl"),
    ("tp_p", "CP(=O)(O)O"),
    ("tp_i", "CI"),
    ("tp_charge", "O=[N+]([O-])c1ccccc1"),
    ("tp_triple", "C#N"),
)
TRAIN_NEGATIVE = (
    ("tn_f", "CCF"),
    ("tn_s", "CS"),
    ("tn_neutral", "CCN"),
    ("tn_double", "C=C"),
)
VAL_POSITIVE = (("vp_br", "CCCBr"), ("vp_p", "CCP(=O)(O)O"))
VAL_NEGATIVE = (("vn_cl", "CCCl"), ("vn_i", "CCI"))


class FakeGraphData:
    def __init__(
        self,
        adj_all,
        features_all,
        u_all,
        labels_all,
        max_num_nodes,
        padded=False,
        index=None,
    ):
        self.max_num_nodes = int(max_num_nodes)
        self.adj_all = list(adj_all)
        self.feature_all = list(features_all)
        self.u_all = list(u_all)
        self.labels_all = list(labels_all)
        self.len_all = [int(adj.shape[0]) for adj in self.adj_all]
        self.padded = bool(padded)
        self.index = list(range(len(self.adj_all))) if index is None else index

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, index):
        return {
            "adj": self.adj_all[index],
            "features": self.feature_all[index],
            "u": self.u_all[index],
            "labels": self.labels_all[index],
            "num_node_real": self.len_all[index],
            "index": self.index[index],
        }


def _write_cohort(
    path: Path,
    molecules: tuple[tuple[str, str], ...],
    *,
    split: str,
    label: int,
) -> None:
    rows = [
        {
            "molecule_id": molecule_id,
            "smiles": smiles,
            "label": label,
            "split": split,
            "teacher_pred": label,
            "teacher_correct": "true",
        }
        for molecule_id, smiles in molecules
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def phase_a_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "train_positive_csv": tmp_path / "train_positive.csv",
        "train_negative_csv": tmp_path / "train_negative.csv",
        "val_positive_csv": tmp_path / "val_positive.csv",
        "val_negative_csv": tmp_path / "val_negative.csv",
    }
    _write_cohort(
        paths["train_positive_csv"], TRAIN_POSITIVE, split="train", label=1
    )
    _write_cohort(
        paths["train_negative_csv"], TRAIN_NEGATIVE, split="train", label=0
    )
    _write_cohort(
        paths["val_positive_csv"], VAL_POSITIVE, split="val", label=1
    )
    _write_cohort(
        paths["val_negative_csv"], VAL_NEGATIVE, split="val", label=0
    )
    return paths


def _load(paths: dict[str, Path]):
    return load_phase_a_cohorts(
        **paths,
        expected_counts={
            "train_positive": len(TRAIN_POSITIVE),
            "train_negative": len(TRAIN_NEGATIVE),
            "val_positive": len(VAL_POSITIVE),
            "val_negative": len(VAL_NEGATIVE),
        },
    )


def test_protocol_expected_counts_are_fixed() -> None:
    assert DEFAULT_EXPECTED_COUNTS["train_positive"] == 1448
    assert DEFAULT_EXPECTED_COUNTS["train_negative"] == 1437
    assert DEFAULT_EXPECTED_COUNTS["val_positive"] == 260
    assert DEFAULT_EXPECTED_COUNTS["val_negative"] == 95
    assert (
        DEFAULT_EXPECTED_COUNTS["train_positive"]
        + DEFAULT_EXPECTED_COUNTS["train_negative"]
        == 2885
    )
    assert (
        DEFAULT_EXPECTED_COUNTS["val_positive"]
        + DEFAULT_EXPECTED_COUNTS["val_negative"]
        == 355
    )


def test_dataset_payload_uses_train_val_only_and_deterministic_order(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    data, splits, summary = build_clear_dataset_payload(
        cohorts=cohorts,
        graph_data_class=FakeGraphData,
    )
    assert summary["train_rows"] == len(TRAIN_POSITIVE) + len(TRAIN_NEGATIVE)
    assert summary["val_rows"] == len(VAL_POSITIVE) + len(VAL_NEGATIVE)
    assert summary["generation_source_parent_rows"] == len(TRAIN_POSITIVE)
    assert summary["label_counts_train"] == {
        "0": len(TRAIN_NEGATIVE),
        "1": len(TRAIN_POSITIVE),
    }
    assert summary["calibration_loaded"] is False
    assert summary["test_loaded"] is False
    assert data.calibration_loaded is False
    assert data.test_loaded is False
    expected_ids = [
        molecule_id for molecule_id, _ in sorted(TRAIN_POSITIVE)
    ] + [molecule_id for molecule_id, _ in sorted(TRAIN_NEGATIVE)]
    assert data.molecule_id_all[: len(expected_ids)] == expected_ids
    assert all(
        np.array_equal(split, splits["idx_train_list"][0])
        for split in splits["idx_train_list"]
    )


def test_train_only_schema_and_max_nodes_from_train_plus_val(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    train = cohorts["train_positive"] + cohorts["train_negative"]
    schema = build_train_schema(train)
    assert schema.atom_vocabulary == (6, 7, 8, 9, 15, 16, 17, 35, 53)
    assert schema.formal_charge_vocabulary == (-1, 0, 1)
    assert schema.aromaticity_vocabulary == (False, True)
    assert schema.bond_type_vocabulary == (
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    )
    _, _, summary = build_clear_dataset_payload(
        cohorts=cohorts,
        graph_data_class=FakeGraphData,
    )
    all_rows = [row for values in cohorts.values() for row in values]
    assert summary["max_num_nodes"] == max(
        row.mol.GetNumAtoms() for row in all_rows
    )
    assert summary["vocabulary_source"] == "strict_train_only"
    assert summary["max_num_nodes_source"] == "strict_train_plus_validation"


def test_validation_unseen_atom_fails_instead_of_mapping_to_carbon(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    schema = build_train_schema(
        cohorts["train_positive"] + cohorts["train_negative"]
    )
    unseen_path = phase_a_files["val_negative_csv"]
    _write_cohort(
        unseen_path,
        (("vn_si", "[SiH4]"),),
        split="val",
        label=0,
    )
    unseen = load_phase_a_cohorts(
        **phase_a_files,
        expected_counts={
            "train_positive": len(TRAIN_POSITIVE),
            "train_negative": len(TRAIN_NEGATIVE),
            "val_positive": len(VAL_POSITIVE),
            "val_negative": 1,
        },
    )
    with pytest.raises(ClearMutagenicityCodecError, match="unseen"):
        validate_validation_schema(unseen["val_negative"], schema)
    with pytest.raises(ClearMutagenicityCodecError, match="no fallback"):
        encode_atom_feature(
            atomic_num=14,
            formal_charge=0,
            is_aromatic=False,
            schema=schema,
        )


@pytest.mark.parametrize("atomic_num", [15, 35, 53])
def test_br_p_i_have_exact_discrete_atom_categories(
    phase_a_files: dict[str, Path],
    atomic_num: int,
) -> None:
    cohorts = _load(phase_a_files)
    schema = build_train_schema(
        cohorts["train_positive"] + cohorts["train_negative"]
    )
    feature = encode_atom_feature(
        atomic_num=atomic_num,
        formal_charge=0,
        is_aromatic=False,
        schema=schema,
    )
    atom_block = feature[schema.atom_feature_start : schema.atom_feature_end]
    assert int(np.argmax(atom_block)) == schema.atom_vocabulary.index(atomic_num)
    assert float(atom_block.sum()) == 1.0


def test_source_round_trip_restores_charge_aromatic_and_bond_types(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    train = cohorts["train_positive"] + cohorts["train_negative"]
    schema = build_train_schema(train)
    for row in train:
        result, checks = round_trip_source_graph(row, schema=schema)
        assert result.ok, (row.molecule_id, result.error)
        assert checks["round_trip_passed"], (row.molecule_id, checks)
    charged = next(row for row in train if row.molecule_id == "tp_charge")
    result, _ = round_trip_source_graph(charged, schema=schema)
    assert -1 in result.decoded_formal_charges
    assert 1 in result.decoded_formal_charges
    assert "AROMATIC" in result.decoded_bond_types
    assert "DOUBLE" in result.decoded_bond_types
    triple = next(row for row in train if row.molecule_id == "tp_triple")
    triple_result, _ = round_trip_source_graph(triple, schema=schema)
    assert "TRIPLE" in triple_result.decoded_bond_types


def test_binary_adjacency_projection_inherits_and_adds_bonds(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    train = cohorts["train_positive"] + cohorts["train_negative"]
    schema = build_train_schema(train)
    double = next(row for row in train if row.molecule_id == "tn_double")
    adjacency, features, sidecar = molecule_to_clear_graph(
        double, schema=schema
    )
    retained = project_binary_graph_to_molecule(
        features=features,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
    )
    assert retained.ok
    assert retained.decoded_bond_types == ["DOUBLE"]
    assert retained.projected_new_edge_count == 0

    chain_mol = Chem.MolFromSmiles("CCC")
    chain = type(double)(
        molecule_id="chain",
        original_smiles="CCC",
        canonical_smiles="CCC",
        label=0,
        split="train",
        source_row_index=0,
        mol=chain_mol,
        atom_categories=tuple(
            atom.GetAtomicNum() for atom in chain_mol.GetAtoms()
        ),
        charge_categories=tuple(
            atom.GetFormalCharge() for atom in chain_mol.GetAtoms()
        ),
        aromatic_categories=tuple(
            atom.GetIsAromatic() for atom in chain_mol.GetAtoms()
        ),
        bond_categories=tuple(
            str(bond.GetBondType()).upper() for bond in chain_mol.GetBonds()
        ),
    )
    chain_adj, chain_features, chain_sidecar = molecule_to_clear_graph(
        chain, schema=schema
    )
    chain_adj[0, 2] = chain_adj[2, 0] = 1.0
    projected = project_binary_graph_to_molecule(
        features=chain_features,
        adjacency=chain_adj,
        schema=schema,
        parent_sidecar=chain_sidecar,
    )
    assert projected.ok
    assert projected.projected_new_edge_count == 1
    assert projected.decoded_bond_types.count("SINGLE") == 3


def test_required_category_probe_is_deterministic_and_complete(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    all_rows = [row for values in cohorts.values() for row in values]
    selected_a, audit_a = required_category_probe_selection(
        all_rows, limit=len(all_rows)
    )
    selected_b, audit_b = required_category_probe_selection(
        list(reversed(all_rows)), limit=len(all_rows)
    )
    assert [row.molecule_id for row in selected_a] == [
        row.molecule_id for row in selected_b
    ]
    assert audit_a == audit_b
    assert set(audit_a["required_category_representatives"]) == {
        "atom_Br",
        "atom_Cl",
        "atom_P",
        "atom_I",
        "formal_charge_-1",
        "formal_charge_+1",
        "bond_AROMATIC",
        "bond_DOUBLE",
        "bond_TRIPLE",
        "near_max_nodes",
    }
    probe_rows, summary = run_codec_probe(
        cohorts=cohorts, limit=len(all_rows)
    )
    assert summary["probe_passed"] is True
    assert summary["probe_failed_rows"] == 0
    assert len(probe_rows) == len(all_rows)
    assert all(row["round_trip_passed"] for row in probe_rows)


def test_calibration_and_test_paths_are_rejected(
    phase_a_files: dict[str, Path],
) -> None:
    forbidden = dict(phase_a_files)
    forbidden["val_negative_csv"] = (
        phase_a_files["val_negative_csv"].parent
        / "test_target_label0_teacher_correct.csv"
    )
    forbidden["val_negative_csv"].write_text(
        phase_a_files["val_negative_csv"].read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_phase_a_cohorts(
            **forbidden,
            expected_counts={
                "train_positive": len(TRAIN_POSITIVE),
                "train_negative": len(TRAIN_NEGATIVE),
                "val_positive": len(VAL_POSITIVE),
                "val_negative": len(VAL_NEGATIVE),
            },
        )
