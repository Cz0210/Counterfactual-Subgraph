from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from src.baselines.clear_mutagenicity_adapter import (
    ATOM_SIDECAR_SCHEMA_VERSION,
    DEFAULT_EXPECTED_COUNTS,
    ClearMutagenicityCodecError,
    PreparedMolecule,
    build_clear_dataset_payload,
    build_train_schema,
    decode_generated_atom_feature,
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

HISTORICAL_NH_FAILURES = (
    (
        "MUT_3019BBA5D04D0B43C486",
        "C=C(C)C(C)NCCCNC(=O)c1csc(-c2csc(CCNC(=O)C(NC(=O)C(C)C(O)"
        "C(C)NC(=O)C(NC(=O)c3nc(C(CC(N)=O)NCC(N)C(N)=O)nc(N)c3C)"
        "C(OC3OC(CO)C(O)C(O)C3OC3OC(CO)C(O)C(OC(N)=O)C3O)"
        "c3c[nH]cn3)C(C)O)n2)n1",
    ),
    ("MUT_006730B97E307FF9A8DF", "O=C(O)CCCc1c[nH]c2ccccc12"),
    ("MUT_02E516F5A6F36C5747AD", "O=c1[nH][nH]c2ccccc12"),
    (
        "MUT_045954270C58B62ACB6D",
        "O=C(Nc1cccc2c(=O)c3c(ccc4c5cc(NC(=O)c6ccccc6)c6c(=O)c7ccccc7"
        "c(=O)c6c5[nH]c43)c(=O)c12)c1ccccc1",
    ),
    ("MUT_050530D5640387950B07", "Cc1cc2c(nc1N)[nH]c1ccccc12"),
)


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


def _prepared(molecule_id: str, smiles: str) -> PreparedMolecule:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return PreparedMolecule(
        molecule_id=molecule_id,
        original_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        ),
        label=1,
        split="train",
        source_row_index=0,
        mol=mol,
        atom_categories=tuple(
            atom.GetAtomicNum() for atom in mol.GetAtoms()
        ),
        charge_categories=tuple(
            atom.GetFormalCharge() for atom in mol.GetAtoms()
        ),
        aromatic_categories=tuple(
            atom.GetIsAromatic() for atom in mol.GetAtoms()
        ),
        bond_categories=tuple(
            str(bond.GetBondType()).upper() for bond in mol.GetBonds()
        ),
        explicit_h_categories=tuple(
            atom.GetNumExplicitHs() for atom in mol.GetAtoms()
        ),
        implicit_h_categories=tuple(
            atom.GetNumImplicitHs() for atom in mol.GetAtoms()
        ),
        no_implicit_categories=tuple(
            atom.GetNoImplicit() for atom in mol.GetAtoms()
        ),
        chiral_tag_categories=tuple(
            int(atom.GetChiralTag()) for atom in mol.GetAtoms()
        ),
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


@pytest.mark.parametrize(("molecule_id", "smiles"), HISTORICAL_NH_FAILURES)
def test_all_historical_aromatic_nh_failures_round_trip(
    molecule_id: str,
    smiles: str,
) -> None:
    row = _prepared(molecule_id, smiles)
    schema = build_train_schema([row])
    result, checks = round_trip_source_graph(row, schema=schema)
    assert result.ok, (molecule_id, result.error_type, result.error)
    assert checks["round_trip_passed"], (molecule_id, checks)
    assert checks["explicit_hs_exact"]
    assert checks["no_implicit_exact"]
    assert checks["chiral_tags_exact"]
    assert checks["mismatch_fields"] == []


def test_aromatic_n_and_nh_remain_distinct() -> None:
    row = _prepared("aromatic_n_pair", "c1ncc[nH]1")
    schema = build_train_schema([row])
    result, checks = round_trip_source_graph(row, schema=schema)
    assert checks["round_trip_passed"]
    nitrogens = [
        atom for atom in result.molecule.GetAtoms() if atom.GetAtomicNum() == 7
    ]
    assert [atom.GetNumExplicitHs() for atom in nitrogens] == [0, 1]
    assert all(atom.GetNoImplicit() is False for atom in nitrogens)


def test_source_sidecar_restores_explicit_h_noimplicit_and_chirality() -> None:
    nh_row = _prepared("indole", "c1c[nH]c2ccccc12")
    chiral_row = _prepared("chiral", "N[C@@H](C)C(=O)O")
    schema = build_train_schema([nh_row, chiral_row])

    _, _, nh_sidecar = molecule_to_clear_graph(nh_row, schema=schema)
    nh_atom = next(
        atom
        for atom in nh_sidecar["atoms"]
        if atom["atomic_num"] == 7 and atom["is_aromatic"]
    )
    assert nh_atom["num_explicit_hs"] == 1
    assert nh_atom["num_implicit_hs"] == 0
    assert nh_atom["no_implicit"] is False
    assert nh_sidecar["atom_sidecar_schema_version"] == (
        ATOM_SIDECAR_SCHEMA_VERSION
    )

    result, checks = round_trip_source_graph(chiral_row, schema=schema)
    assert result.ok
    assert checks["chiral_tags_exact"]
    assert any(
        tag != int(Chem.ChiralType.CHI_UNSPECIFIED)
        for tag in result.decoded_chiral_tags
    )


@pytest.mark.parametrize(
    "smiles",
    ("[nH+]1ccccc1", "[n-]1cccc1"),
)
def test_charged_aromatic_n_round_trip(smiles: str) -> None:
    row = _prepared("charged_aromatic_n", smiles)
    schema = build_train_schema([row])
    result, checks = round_trip_source_graph(row, schema=schema)
    assert result.ok
    assert checks["round_trip_passed"]
    assert checks["formal_charges_exact"]
    assert checks["explicit_hs_exact"]
    assert checks["no_implicit_exact"]


def test_unchanged_generated_atom_inherits_parent_hydrogen_state() -> None:
    row = _prepared("generated_indole", "c1c[nH]c2ccccc12")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    result = project_binary_graph_to_molecule(
        features=features,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok
    source_explicit_hs = [
        atom.GetNumExplicitHs() for atom in row.mol.GetAtoms()
    ]
    assert result.decoded_num_explicit_hs == source_explicit_hs
    assert max(result.decoded_num_explicit_hs) == 1


def test_changed_generated_atom_is_sanitized_instead_of_preemptively_rejected() -> None:
    row = _prepared("generated_changed", "c1c[nH]c2ccccc12")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    nitrogen_slot = next(
        atom.GetIdx()
        for atom in row.mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
    )
    changed = features.copy()
    changed[
        nitrogen_slot,
        schema.atom_feature_start : schema.atom_feature_end,
    ] = 0.0
    changed[
        nitrogen_slot,
        schema.atom_feature_start + schema.atom_vocabulary.index(6),
    ] = 1.0
    result = project_binary_graph_to_molecule(
        features=changed,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok is False
    assert result.error_type == "generated_kekulization_failed"
    assert result.error_type != "ambiguous_generated_atom_hydrogen_state"
    assert result.decoded_num_explicit_hs[nitrogen_slot] == 0
    assert result.decoded_no_implicit[nitrogen_slot] is False
    assert result.decoded_chiral_tags[nitrogen_slot] == int(
        Chem.ChiralType.CHI_UNSPECIFIED
    )


def _change_generated_atom_identity(
    features: np.ndarray,
    *,
    slot: int,
    schema,
    atomic_num: int | None = None,
    formal_charge: int | None = None,
    is_aromatic: bool | None = None,
) -> np.ndarray:
    changed = features.copy()
    if atomic_num is not None:
        changed[slot, schema.atom_feature_start : schema.atom_feature_end] = 0.0
        changed[
            slot,
            schema.atom_feature_start
            + schema.atom_vocabulary.index(atomic_num),
        ] = 1.0
    if formal_charge is not None:
        changed[
            slot,
            schema.charge_feature_start : schema.charge_feature_end,
        ] = 0.0
        changed[
            slot,
            schema.charge_feature_start
            + schema.formal_charge_vocabulary.index(formal_charge),
        ] = 1.0
    if is_aromatic is not None:
        changed[slot, schema.aromatic_feature_index] = float(is_aromatic)
    return changed


@pytest.mark.parametrize("source_smiles", ("OC", "NC", "ClC", "BrC"))
def test_generated_atom_identity_change_uses_decoded_atom_without_source_h(
    source_smiles: str,
) -> None:
    row = _prepared("identity_change", source_smiles)
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    changed = _change_generated_atom_identity(
        features, slot=0, schema=schema, atomic_num=6
    )
    result = project_binary_graph_to_molecule(
        features=changed,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok, (result.error_type, result.error)
    assert result.canonical_smiles == "CC"
    assert result.decoded_atomic_numbers[0] == 6
    assert result.decoded_num_explicit_hs[0] == 0
    assert result.decoded_no_implicit[0] is False
    assert result.atom_state_sources[0] == "generated_deterministic_default"


def test_source_mode_still_rejects_atom_identity_change() -> None:
    row = _prepared("source_exact", "OC")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    changed = _change_generated_atom_identity(
        features, slot=0, schema=schema, atomic_num=6
    )
    result = project_binary_graph_to_molecule(
        features=changed,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="source",
    )
    assert result.ok is False
    assert result.error_type == "source_atom_attribute_mapping_failed"


def test_generated_aromatic_to_nonaromatic_does_not_inherit_source_h() -> None:
    row = _prepared("aromatic_change", "c1ccccc1")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    changed = _change_generated_atom_identity(
        features, slot=0, schema=schema, is_aromatic=False
    )
    result = project_binary_graph_to_molecule(
        features=changed,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok is False
    assert result.error_type == "generated_aromaticity_bond_inconsistent"
    assert result.decoded_aromaticity[0] is False
    assert result.decoded_num_explicit_hs[0] == 0
    assert result.error_type != "ambiguous_generated_atom_hydrogen_state"


def test_generated_formal_charge_change_uses_deterministic_hydrogen_state() -> None:
    charged = _prepared("charged", "[NH4+]")
    neutral = _prepared("neutral", "N")
    schema = build_train_schema([charged, neutral])
    adjacency, features, sidecar = molecule_to_clear_graph(
        charged, schema=schema
    )
    changed = _change_generated_atom_identity(
        features, slot=0, schema=schema, formal_charge=0
    )
    result = project_binary_graph_to_molecule(
        features=changed,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok, (result.error_type, result.error)
    assert result.decoded_formal_charges == [0]
    assert result.decoded_num_explicit_hs == [0]
    assert result.decoded_no_implicit == [False]
    assert result.decoded_chiral_tags == [int(Chem.ChiralType.CHI_UNSPECIFIED)]


def test_unchanged_identity_with_changed_incident_edge_does_not_inherit_h() -> None:
    row = _prepared("changed_edge", "[NH2]C")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    adjacency = adjacency.copy()
    adjacency[0, 1] = adjacency[1, 0] = 0.0
    result = project_binary_graph_to_molecule(
        features=features,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
        require_connected=False,
    )
    assert result.ok, (result.error_type, result.error)
    assert result.decoded_num_explicit_hs[0] == 0
    assert result.atom_state_sources[0] == "generated_deterministic_default"


def test_generated_impossible_valence_is_rejected_by_rdkit_sanitize() -> None:
    row = _prepared("overbonded", "CCCCCC")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    adjacency = np.ones_like(adjacency)
    result = project_binary_graph_to_molecule(
        features=features,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok is False
    assert result.error_type == "generated_valence_sanitize_failed"


def test_generated_padded_node_is_not_decoded_as_carbon() -> None:
    row = _prepared("padded", "CC")
    schema = build_train_schema([row])
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    padded_features = np.zeros((3, schema.feature_dim), dtype=np.float64)
    padded_features[:2] = features
    padded_adjacency = np.zeros((3, 3), dtype=np.float64)
    padded_adjacency[:2, :2] = adjacency
    result = project_binary_graph_to_molecule(
        features=padded_features,
        adjacency=padded_adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="generated",
    )
    assert result.ok
    assert result.num_real_nodes == 2
    assert result.feature_decoding_audit["padded_node_count"] == 1
    assert result.feature_decoding_audit["padded_node_slots"] == [2]


def test_generated_logit_argmax_uses_exact_train_vocabulary_mapping(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    schema = build_train_schema(
        cohorts["train_positive"] + cohorts["train_negative"]
    )
    feature = np.full(schema.feature_dim, -9.0, dtype=np.float64)
    atom_index = schema.atom_vocabulary.index(35)
    charge_index = schema.formal_charge_vocabulary.index(-1)
    feature[schema.atom_feature_start + atom_index] = -1.0
    feature[schema.charge_feature_start + charge_index] = -2.0
    feature[schema.aromatic_feature_index] = 0.7
    feature[schema.node_present_feature_index] = 1.0
    decoded = decode_generated_atom_feature(feature, schema=schema)
    assert decoded is not None
    assert decoded["atom_argmax_index"] == atom_index
    assert decoded["charge_argmax_index"] == charge_index
    assert decoded["atomic_num"] == 35
    assert decoded["formal_charge"] == -1
    assert decoded["is_aromatic"] is True


def test_dataset_summary_versions_atom_sidecar(
    phase_a_files: dict[str, Path],
) -> None:
    cohorts = _load(phase_a_files)
    data, _, summary = build_clear_dataset_payload(
        cohorts=cohorts,
        graph_data_class=FakeGraphData,
    )
    assert summary["atom_sidecar_schema_version"] == (
        ATOM_SIDECAR_SCHEMA_VERSION
    )
    assert summary["stores_num_explicit_hs"] is True
    assert summary["stores_no_implicit"] is True
    assert summary["stores_chiral_tag"] is True
    assert data.atom_sidecar_schema_version == ATOM_SIDECAR_SCHEMA_VERSION


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
