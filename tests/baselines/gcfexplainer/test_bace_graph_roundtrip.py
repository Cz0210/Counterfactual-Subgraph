from __future__ import annotations

import pytest
from rdkit import Chem

from src.baselines.gcfexplainer_bace_adapter import (
    BACE_FEATURE_ATOMIC_NUMBERS,
    BACEGraphSchema,
)
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    StrictMolecule,
    encode_source_graph,
    reconstruct_source_graph,
)


SCHEMA = BACEGraphSchema(
    atom_vocabulary=BACE_FEATURE_ATOMIC_NUMBERS,
    feature_atomic_numbers=BACE_FEATURE_ATOMIC_NUMBERS,
    formal_charge_vocabulary=(-1, 0, 1),
    aromaticity_vocabulary=(False, True),
    bond_type_vocabulary=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
    max_num_nodes=256,
)


@pytest.mark.parametrize(
    "smiles",
    (
        "CC(=O)Oc1ccccc1C(=O)O",
        "C[NH2+]CC(=O)[O-]",
        "Clc1ncc(Br)c(I)c1",
    ),
)
def test_bace_source_graph_roundtrip_is_chemically_exact(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    record = StrictMolecule(
        molecule_id="BACE_TEST",
        smiles=smiles,
        canonical_smiles=canonical,
        label=1,
        split="train",
        semantic_label="active",
        source_row_index=0,
        source_path="synthetic",
    )
    encoded = encode_source_graph(record, SCHEMA)
    _molecule, checks = reconstruct_source_graph(encoded, SCHEMA)
    assert checks["round_trip_passed"] is True
    assert checks["formal_charges_exact"] is True
    assert checks["aromaticity_exact"] is True
    assert checks["bond_types_exact"] is True
    assert checks["canonical_smiles_exact"] is True
