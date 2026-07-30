from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from src.baselines.globalgce_mutagenicity_adapter import (
    GlobalGCECodecMetadata,
    GlobalGCEEmptyCandidateUniverseError,
    GlobalGCEMutagenicityCodecError,
    NativeGenerationResult,
    PoolBuildConfig,
    TrainParent,
    _add_bond_once,
    _build_dense_dataset,
    attach_globalgce_generation_dataset,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
    decode_globalgce_molecule,
    globalgce_tensors_to_graph_record,
    load_strict_train_parents,
    probe_source_graph_codec,
    require_source_codec_gate,
    stable_candidate_id,
    validate_globalgce_generation_loader,
)


def _write_train(path: Path) -> None:
    rows = [
        {
            "molecule_id": "p3",
            "smiles": "CCCl",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
        {
            "molecule_id": "p1",
            "smiles": "CCO",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
        {
            "molecule_id": "p2",
            "smiles": "CCN",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_official_root(root: Path) -> None:
    for relative in ("main.py", "models/GlobalGCE.py", "data/data_preprocess.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake official layout\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _FakeTeacher:
    def score_smiles(self, smiles: str, label: int | None = None, **_kwargs):
        del label
        prediction = 1 if smiles == "CO" else 0
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
        }


class _FakeGenerator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate(self, parents, **_kwargs):
        if self.fail:
            raise RuntimeError("intentional generation interruption")
        assert [parent.parent_id for parent in parents] == ["p1", "p2", "p3"]
        records = [
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "CC",
                "generator_rank": 1,
                "generator_score": 3.0,
                "native_rule_id": "r1",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "C-C",
                "generator_rank": 2,
                "generator_score": 2.0,
                "native_rule_id": "r2",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "not_a_smiles",
                "generator_rank": 3,
                "native_rule_id": "r3",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "CO",
                "generator_rank": 4,
                "native_rule_id": "r4",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p2",
                "source_parent_smiles": "CCN",
                "source_split": "train",
                "raw_smiles": "CC",
                "generator_rank": 1,
                "native_rule_id": "r1",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p2",
                "source_parent_smiles": "CCN",
                "source_split": "train",
                "raw_smiles": "CCC",
                "generator_rank": 2,
                "native_rule_id": "r5",
                "native_run_id": "fake",
            },
        ]
        return NativeGenerationResult(
            records,
            {
                "internal_train_ids_hash": "train-hash",
                "internal_val_ids_hash": "val-hash",
                "native_gnn_required": True,
                "source_codec_checked_rows": len(parents),
                "source_codec_rdkit_valid_rows": len(parents),
                "source_codec_structure_match_rows": len(parents),
                "source_codec_invalid_valence_rows": 0,
                "source_codec_attribute_mapping_failed_rows": 0,
                "source_codec_passed": True,
                "source_codec_failure_examples": [],
                "atom_attribute_source": "source_anchored",
                "formal_charge_encoded_by_native": False,
                "source_atom_mapping_method": (
                    "rdkit_atom_index_preserved_by_dense_builder"
                ),
                "source_atom_mapping_unique": True,
                "source_formal_charge_nonzero_atom_count": 0,
            },
        )


class _FakeNativeDataset:
    def __init__(self, max_num_nodes: int = 17) -> None:
        self.max_num_nodes = max_num_nodes


class _FakeAugmentedDataset:
    def __init__(self) -> None:
        self.rows = [
            {"parent_id": "p2", "value": 2},
            {"parent_id": "p1", "value": 1},
            {"parent_id": "p3", "value": 3},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return dict(self.rows[index])


def _native_graph_tensors(
    atom_labels: list[int],
    edges: list[tuple[int, int, int]],
    *,
    node_class_count: int,
    edge_class_count: int = 4,
    adjacency_value: float = 1.0,
):
    torch = pytest.importorskip("torch")
    size = len(atom_labels)
    features = torch.zeros((size, node_class_count), dtype=torch.float32)
    for index, label in enumerate(atom_labels):
        features[index, int(label)] = 1.0
    adjacency = torch.zeros((size, size), dtype=torch.float32)
    edge_attr = torch.zeros(
        (size * (size - 1) // 2, edge_class_count),
        dtype=torch.float32,
    )
    edge_attr[:, 0] = 1.0
    for left, right, edge_label in edges:
        adjacency[left, right] = adjacency_value
        adjacency[right, left] = adjacency_value
        high, low = max(left, right), min(left, right)
        position = (high - 1) * high // 2 + low
        edge_attr[position, 0] = 0.0
        edge_attr[position, edge_label] = 1.0
    return features, adjacency, edge_attr


def _atom_sidecar(
    atom_labels: list[int],
    metadata: GlobalGCECodecMetadata,
    *,
    formal_charges: dict[int, int] | None = None,
    explicit_hs: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    periodic_table = Chem.GetPeriodicTable()
    charges = formal_charges or {}
    hydrogens = explicit_hs or {}
    rows = []
    for index, label in enumerate(atom_labels):
        if label <= 0:
            continue
        symbol = metadata.atom_symbols[label - 1]
        rows.append(
            {
                "native_node_index": index,
                "source_atom_index": index,
                "atomic_num": periodic_table.GetAtomicNumber(symbol),
                "formal_charge": charges.get(index, 0),
                "is_aromatic": False,
                "num_explicit_hs": hydrogens.get(index, 0),
                "isotope": 0,
                "chiral_tag": int(Chem.ChiralType.CHI_UNSPECIFIED),
                "no_implicit": bool(hydrogens.get(index, 0)),
                "attribute_source": "source_anchored",
            }
        )
    return rows


@pytest.fixture()
def fixture_paths(tmp_path: Path) -> dict[str, Path]:
    train = tmp_path / "train.csv"
    teacher = tmp_path / "teacher.pkl"
    official = tmp_path / "official"
    output = tmp_path / "output"
    _write_train(train)
    teacher.write_bytes(b"fake-teacher")
    _write_official_root(official)
    return {
        "train": train,
        "teacher": teacher,
        "official": official,
        "output": output,
    }


def _build(paths: dict[str, Path], **config_overrides):
    defaults = {
        "expected_parent_count": 3,
        "device": "cpu",
        "epochs": 1,
        "top_k_native": 2,
    }
    defaults.update(config_overrides)
    return build_mutagenicity_train_pool(
        train_csv=paths["train"],
        teacher_path=paths["teacher"],
        official_root=paths["official"],
        output_dir=paths["output"],
        teacher=_FakeTeacher(),
        generator=_FakeGenerator(),
        config=PoolBuildConfig(**defaults),
    )


def test_train_only_parent_loading_and_deterministic_parent_limit(
    fixture_paths: dict[str, Path],
) -> None:
    all_parents, selected = load_strict_train_parents(
        fixture_paths["train"],
        parent_limit=2,
        expected_parent_count=3,
    )
    assert [parent.parent_id for parent in all_parents] == ["p1", "p2", "p3"]
    assert [parent.parent_id for parent in selected] == ["p1", "p2"]


def test_source_graph_round_trip_uses_native_metadata() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("benzene", "c1ccccc1", 1, "train"),
        TrainParent("bromoethane", "CCBr", 1, "train"),
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[1],
        test_idx=[],
        torch_module=torch,
    )
    summary = probe_source_graph_codec(dataset, parents)

    assert summary["source_codec_checked_rows"] == 2
    assert summary["source_codec_rdkit_valid_rows"] == 2
    assert summary["source_codec_structure_match_rows"] == 2
    assert summary["source_codec_passed"] is True
    assert summary["source_codec_failure_examples"] == []
    node_mapping = summary["codec_metadata"]["node_label_mapping"]
    edge_mapping = summary["codec_metadata"]["edge_label_mapping"]
    assert node_mapping == {"0": "padding", "1": "C", "2": "Br"}
    assert edge_mapping == {
        "0": "no_edge",
        "1": "single",
        "2": "double",
        "3": "triple",
    }


@pytest.mark.parametrize(
    "smiles",
    [
        "[N+](=O)[O-]",
        "O=[N+]([O-])c1ccccc1",
        "Nc1ccc(O)c([N+](=O)[O-])c1",
    ],
)
def test_nitro_source_graph_round_trip_restores_formal_charge(
    smiles: str,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    parents = [TrainParent("nitro", smiles, 1, "train")]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    audit_path = tmp_path / "source_atom_attribute_audit.jsonl"
    summary = probe_source_graph_codec(
        dataset,
        parents,
        atom_attribute_audit_path=audit_path,
    )

    assert summary["formal_charge_encoded_by_native"] is False
    assert summary["atom_attribute_source"] == "source_anchored"
    assert summary["source_atom_mapping_unique"] is True
    assert summary["source_codec_invalid_valence_rows"] == 0
    assert summary["source_codec_attribute_mapping_failed_rows"] == 0
    assert summary["source_codec_structure_match_rows"] == 1
    assert summary["source_codec_passed"] is True
    charges = {
        int(row["formal_charge"])
        for row in map(json.loads, audit_path.read_text().splitlines())
    }
    assert charges == {-1, 1}


def test_source_sidecar_preserves_charged_atoms_and_neutral_amine() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train"),
        TrainParent("amine", "CCN", 1, "train"),
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[1],
        test_idx=[],
        torch_module=torch,
    )
    nitro_sidecar = dataset.source_atom_attributes[0]
    amine_sidecar = dataset.source_atom_attributes[1]

    assert any(
        row["atomic_num"] == 7 and row["formal_charge"] == 1
        for row in nitro_sidecar
    )
    assert any(
        row["atomic_num"] == 8 and row["formal_charge"] == -1
        for row in nitro_sidecar
    )
    assert all(
        row["formal_charge"] == 0
        for row in amine_sidecar
        if row["atomic_num"] == 7
    )


def test_source_atom_mapping_ambiguity_is_rejected() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    tensors = _native_graph_tensors(
        [1, 1],
        [(0, 1, 1)],
        node_class_count=2,
    )
    ambiguous = _atom_sidecar([1, 1], metadata)
    ambiguous[1]["source_atom_index"] = 0
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="source",
        source_atom_attributes=ambiguous,
    )

    assert result.ok is False
    assert result.error_type == "source_codec_attribute_mapping_failed"
    assert "not unique" in str(result.error_message)


def test_generated_same_atom_type_inherits_source_charge() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train")
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    graph = dataset[0]
    result = decode_globalgce_molecule(
        graph["feature"],
        graph["adj"],
        graph["edge_attr"],
        metadata=GlobalGCECodecMetadata.from_dataset(dataset),
        num_nodes=int(graph["num_nodes"]),
        graph_role="generated",
        source_atom_attributes=dataset.source_atom_attributes[0],
    )

    assert result.ok
    assert sorted(
        atom.GetFormalCharge()
        for atom in result.mol.GetAtoms()
        if atom.GetFormalCharge()
    ) == [-1, 1]


def test_generated_atom_type_change_does_not_inherit_charge() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train")
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    graph = dataset[0]
    metadata = GlobalGCECodecMetadata.from_dataset(dataset)
    feature = graph["feature"].clone()
    nitrogen_index = next(
        int(row["native_node_index"])
        for row in dataset.source_atom_attributes[0]
        if row["formal_charge"] == 1
    )
    carbon_label = metadata.atom_symbols.index("C") + 1
    feature[nitrogen_index].zero_()
    feature[nitrogen_index, carbon_label] = 1.0
    result = decode_globalgce_molecule(
        feature,
        graph["adj"],
        graph["edge_attr"],
        metadata=metadata,
        num_nodes=int(graph["num_nodes"]),
        graph_role="generated",
        source_atom_attributes=dataset.source_atom_attributes[0],
    )

    assert result.ok is False
    assert result.error_type == "generated_attribute_ambiguous"
    changed = result.graph_record["atom_attribute_audit"][-1]
    assert changed["attributes_inherited"] is False
    assert changed["formal_charge"] is None
    assert (
        changed["ambiguity_reason"]
        == "generated_atom_type_differs_from_source"
    )


def test_generated_new_node_without_source_identity_is_ambiguous() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    tensors = _native_graph_tensors(
        [1, 1, 1],
        [(0, 1, 1), (1, 2, 1)],
        node_class_count=2,
    )
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 1], metadata),
    )

    assert result.ok is False
    assert result.error_type == "generated_attribute_ambiguous"
    assert (
        result.graph_record["atom_attribute_audit"][-1][
            "ambiguity_reason"
        ]
        == "new_native_node_without_source_identity"
    )


def test_source_and_generated_valence_errors_use_distinct_types() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C", "Br"),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    labels = [2, 1, 1, 1, 1, 1, 1]
    tensors = _native_graph_tensors(
        labels,
        [(0, index, 1) for index in range(1, 7)],
        node_class_count=3,
    )
    sidecar = _atom_sidecar(labels, metadata)
    source_result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="source",
        source_atom_attributes=sidecar,
    )
    generated_result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )

    assert source_result.error_type == "source_codec_invalid_valence"
    assert generated_result.error_type == "generated_invalid_valence"
    assert all(
        row["formal_charge"] == 0
        for row in generated_result.graph_record["atom_attribute_audit"]
    )


def test_adjacency_is_only_edge_mask_and_edge_class_controls_bond_type() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    single = _native_graph_tensors(
        [1, 1],
        [(0, 1, 1)],
        node_class_count=2,
        adjacency_value=9.0,
    )
    double = _native_graph_tensors(
        [1, 1],
        [(0, 1, 2)],
        node_class_count=2,
        adjacency_value=9.0,
    )
    sidecar = _atom_sidecar([1, 1], metadata)
    single_result = decode_globalgce_molecule(
        *single,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )
    double_result = decode_globalgce_molecule(
        *double,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )

    assert single_result.ok and double_result.ok
    assert single_result.mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE
    assert double_result.mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE


def test_padding_self_loops_and_undirected_edges_are_handled_once() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    feature, adjacency, edge_attr = _native_graph_tensors(
        [1, 1, 0],
        [(0, 1, 1)],
        node_class_count=2,
    )
    adjacency[0, 0] = 1.0
    record = globalgce_tensors_to_graph_record(
        feature,
        adjacency,
        edge_attr,
        metadata=metadata,
    )
    result = decode_globalgce_molecule(
        feature,
        adjacency,
        edge_attr,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 1, 0], metadata),
    )

    assert record["padding_node_indices"] == [2]
    assert record["self_loop_count_ignored"] == 1
    assert result.ok
    assert result.num_atoms == 2
    assert result.num_bonds == 1


def test_duplicate_bond_is_rejected_explicitly() -> None:
    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom("C"))
    molecule.AddAtom(Chem.Atom("C"))
    seen: set[tuple[int, int]] = set()
    _add_bond_once(molecule, 0, 1, Chem.BondType.SINGLE, seen)
    with pytest.raises(GlobalGCEMutagenicityCodecError, match="Duplicate"):
        _add_bond_once(molecule, 1, 0, Chem.BondType.SINGLE, seen)


def test_aromatic_bond_metadata_sets_consistent_aromatic_atoms() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple", "aromatic"),
    )
    edges = [
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 4, 4),
        (4, 5, 4),
        (5, 0, 4),
    ]
    tensors = _native_graph_tensors(
        [1] * 6,
        edges,
        node_class_count=2,
        edge_class_count=5,
    )
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1] * 6, metadata),
    )

    assert result.ok
    assert result.smiles == "c1ccccc1"
    assert all(atom.GetIsAromatic() for atom in result.mol.GetAtoms())
    assert all(bond.GetIsAromatic() for bond in result.mol.GetBonds())


def test_bromine_single_bond_is_valid_but_overvalent_bromine_is_not_repaired() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C", "Br"),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    valid = _native_graph_tensors(
        [1, 2],
        [(0, 1, 1)],
        node_class_count=3,
    )
    valid_result = decode_globalgce_molecule(
        *valid,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 2], metadata),
    )
    assert valid_result.ok
    assert valid_result.smiles == "CBr"

    invalid = _native_graph_tensors(
        [2, 1, 1, 1, 1, 1, 1],
        [(0, index, 1) for index in range(1, 7)],
        node_class_count=3,
    )
    invalid_result = decode_globalgce_molecule(
        *invalid,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar(
            [2, 1, 1, 1, 1, 1, 1],
            metadata,
        ),
    )
    assert invalid_result.ok is False
    assert invalid_result.codec_decoded is True
    assert invalid_result.error_type == "generated_invalid_valence"
    assert invalid_result.smiles is None
    assert invalid_result.num_bonds == 6


def test_source_codec_gate_hard_fails_on_round_trip_mismatch() -> None:
    summary = {
        "source_codec_passed": False,
        "source_codec_failure_examples": [{"parent_id": "p1"}],
    }
    with pytest.raises(
        GlobalGCEMutagenicityCodecError,
        match="before training",
    ):
        require_source_codec_gate(summary)


def test_generation_loader_retains_real_native_dataset_contract() -> None:
    torch = pytest.importorskip("torch")
    native = _FakeNativeDataset(max_num_nodes=17)
    augmented = _FakeAugmentedDataset()
    rows_before = [augmented[index] for index in range(len(augmented))]
    length_before = len(augmented)

    returned = attach_globalgce_generation_dataset(augmented, native)
    loader = torch.utils.data.DataLoader(
        augmented,
        batch_size=2,
        shuffle=False,
    )

    assert returned is augmented
    assert loader.dataset is augmented
    assert loader.dataset.dataset is native
    assert loader.dataset.dataset is not loader.dataset
    assert loader.dataset.dataset.max_num_nodes == 17
    assert validate_globalgce_generation_loader(loader) == 17
    assert len(augmented) == length_before
    assert [augmented[index] for index in range(len(augmented))] == rows_before

    # Identity, rather than a copied/fabricated max_num_nodes, is observable.
    native.max_num_nodes = 19
    assert loader.dataset.dataset.max_num_nodes == 19
    assert validate_globalgce_generation_loader(loader) == 19


def test_fake_official_generate_cfs_can_use_nested_dataset_contract() -> None:
    torch = pytest.importorskip("torch")
    native = _FakeNativeDataset(max_num_nodes=11)
    augmented = attach_globalgce_generation_dataset(
        _FakeAugmentedDataset(),
        native,
    )
    loader = torch.utils.data.DataLoader(augmented, batch_size=1, shuffle=False)

    def fake_generate_cfs(dataloader):
        max_num_nodes = dataloader.dataset.dataset.max_num_nodes
        parent_order = [
            parent_id
            for batch in dataloader
            for parent_id in batch["parent_id"]
        ]
        return max_num_nodes, parent_order

    max_num_nodes, parent_order = fake_generate_cfs(loader)
    assert max_num_nodes == native.max_num_nodes
    assert parent_order == ["p2", "p1", "p3"]


def test_generation_loader_validator_reports_missing_underlying_dataset() -> None:
    class LoaderWithoutDataset:
        pass

    with pytest.raises(
        ValueError,
        match=r"LoaderWithoutDataset.*missing attribute dataset",
    ):
        validate_globalgce_generation_loader(LoaderWithoutDataset())

    class Loader:
        dataset = _FakeAugmentedDataset()

    with pytest.raises(
        ValueError,
        match=r"_FakeAugmentedDataset.*missing underlying attribute dataset",
    ):
        validate_globalgce_generation_loader(Loader())


def test_generation_loader_validator_reports_missing_max_num_nodes() -> None:
    class NativeWithoutMaximum:
        pass

    class Loader:
        dataset = _FakeAugmentedDataset()

    Loader.dataset.dataset = NativeWithoutMaximum()
    with pytest.raises(
        ValueError,
        match=r"NativeWithoutMaximum.*missing required attribute max_num_nodes",
    ):
        validate_globalgce_generation_loader(Loader())


def test_generation_loader_rejects_self_reference() -> None:
    augmented = _FakeAugmentedDataset()
    augmented.dataset = augmented

    class Loader:
        dataset = augmented

    with pytest.raises(ValueError, match="self-referential"):
        validate_globalgce_generation_loader(Loader())
    with pytest.raises(ValueError, match="cannot reference itself"):
        attach_globalgce_generation_dataset(augmented, augmented)


@pytest.mark.parametrize("name", ["calibration_source.csv", "test_source.csv"])
def test_forbidden_cohort_paths_are_rejected(
    fixture_paths: dict[str, Path],
    name: str,
) -> None:
    forbidden = fixture_paths["train"].with_name(name)
    forbidden.write_bytes(fixture_paths["train"].read_bytes())
    with pytest.raises(ValueError, match="forbidden|train-only"):
        load_strict_train_parents(forbidden, expected_parent_count=3)


def test_build_isolates_invalid_and_non_target_and_deduplicates(
    fixture_paths: dict[str, Path],
) -> None:
    summary = _build(fixture_paths)
    root = fixture_paths["output"]
    raw = _read_jsonl(root / "raw_generated_candidates.jsonl")
    pool = _read_jsonl(root / "candidate_pool.jsonl")
    universe = _read_jsonl(root / "candidate_universe.jsonl")
    invalid = _read_jsonl(root / "invalid_candidates.jsonl")
    non_target = _read_jsonl(root / "non_target_candidates.jsonl")

    assert len(raw) == 6
    assert len(invalid) == 1
    assert len(non_target) == 1
    assert {(row["source_parent_id"], row["canonical_smiles"]) for row in pool} == {
        ("p1", "CC"),
        ("p2", "CC"),
        ("p2", "CCC"),
    }
    assert [row["canonical_smiles"] for row in universe] == ["CC", "CCC"]
    cc = next(row for row in universe if row["canonical_smiles"] == "CC")
    assert cc["source_parent_count"] == 2
    assert cc["source_occurrence_count"] == 2
    assert cc["candidate_id"] == stable_candidate_id("CC")
    assert summary["raw_generated_rows"] == 6
    assert summary["rdkit_valid_rows"] == 5
    assert summary["teacher_target_rows"] == 4
    assert summary["candidate_pool_rows"] == 3
    assert summary["canonical_unique_candidates"] == 2
    assert summary["calibration_loaded"] is False
    assert summary["test_loaded"] is False

    audit = audit_mutagenicity_train_pool(
        root,
        train_csv=fixture_paths["train"],
        expected_parent_count=3,
    )
    assert audit["audit_passed"] is True


def test_generated_attribute_ambiguity_is_excluded_from_candidate_pool(
    fixture_paths: dict[str, Path],
) -> None:
    class AmbiguousGenerator:
        def generate(self, parents, **_kwargs):
            return NativeGenerationResult(
                [
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": None,
                        "generator_rank": 1,
                        "native_codec_decoded": False,
                        "native_conversion_ok": False,
                        "native_conversion_error_type": (
                            "generated_attribute_ambiguous"
                        ),
                        "native_conversion_error": (
                            "new node has no source identity"
                        ),
                    },
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": "CC",
                        "generator_rank": 2,
                        "native_codec_decoded": True,
                        "native_conversion_ok": True,
                    },
                ],
                {
                    "source_codec_checked_rows": len(parents),
                    "source_codec_rdkit_valid_rows": len(parents),
                    "source_codec_structure_match_rows": len(parents),
                    "source_codec_invalid_valence_rows": 0,
                    "source_codec_attribute_mapping_failed_rows": 0,
                    "source_codec_passed": True,
                    "source_codec_failure_examples": [],
                    "atom_attribute_source": "source_anchored",
                    "formal_charge_encoded_by_native": False,
                    "source_atom_mapping_method": (
                        "rdkit_atom_index_preserved_by_dense_builder"
                    ),
                    "source_atom_mapping_unique": True,
                    "source_formal_charge_nonzero_atom_count": 0,
                },
            )

    summary = build_mutagenicity_train_pool(
        train_csv=fixture_paths["train"],
        teacher_path=fixture_paths["teacher"],
        official_root=fixture_paths["official"],
        output_dir=fixture_paths["output"],
        teacher=_FakeTeacher(),
        generator=AmbiguousGenerator(),
        config=PoolBuildConfig(
            expected_parent_count=3,
            device="cpu",
            epochs=1,
            top_k_native=2,
        ),
    )
    invalid = _read_jsonl(
        fixture_paths["output"] / "invalid_candidates.jsonl"
    )
    pool = _read_jsonl(fixture_paths["output"] / "candidate_pool.jsonl")

    assert summary["generated_attribute_ambiguous_rows"] == 1
    assert len(invalid) == 1
    assert invalid[0]["invalid_reason"] == "generated_attribute_ambiguous"
    assert len(pool) == 1
    assert pool[0]["canonical_smiles"] == "CC"


def test_empty_generated_universe_is_incomplete_and_has_no_completion_marker(
    fixture_paths: dict[str, Path],
) -> None:
    class AllInvalidGenerator:
        def generate(self, parents, **_kwargs):
            return NativeGenerationResult(
                [
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": None,
                        "generator_rank": 1,
                        "native_codec_decoded": True,
                        "native_conversion_ok": False,
                        "native_conversion_error_type": (
                            "generated_invalid_valence"
                        ),
                        "native_conversion_error": "explicit valence failure",
                    }
                ],
                {
                    "source_codec_checked_rows": 3,
                    "source_codec_rdkit_valid_rows": 3,
                    "source_codec_structure_match_rows": 3,
                    "source_codec_passed": True,
                    "source_codec_failure_examples": [],
                },
            )

    with pytest.raises(
        GlobalGCEEmptyCandidateUniverseError,
        match=r"source_codec_passed=True.*generated_invalid_valence_rows=1",
    ):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=AllInvalidGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
            ),
        )

    summary = json.loads(
        (fixture_paths["output"] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["raw_generated_rows"] == 1
    assert summary["generated_graph_rows"] == 1
    assert summary["generated_codec_decoded_rows"] == 1
    assert summary["generated_rdkit_valid_rows"] == 0
    assert summary["generated_invalid_valence_rows"] == 1
    assert summary["generated_invalid_other_rows"] == 0
    assert summary["candidate_pool_rows"] == 0
    assert summary["canonical_unique_candidates"] == 0
    assert summary["source_codec_passed"] is True
    assert summary["run_complete"] is False
    assert not (fixture_paths["output"] / "_RUN_COMPLETE.json").exists()


def test_codec_probe_only_does_not_construct_teacher_or_call_training(
    fixture_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.baselines.globalgce import (
        build_mutagenicity_train_pool as build_cli,
    )

    calls = {"probe": 0, "train": 0}

    class ProbeGenerator:
        def __init__(self, _official_root, *, native_train_csv=None):
            assert native_train_csv == fixture_paths["train"].resolve()

        def probe_codec(self, parents, *, seed, output_path):
            calls["probe"] += 1
            assert [parent.parent_id for parent in parents] == ["p1", "p2"]
            assert seed == 13
            payload = {
                "source_codec_checked_rows": 2,
                "source_codec_rdkit_valid_rows": 2,
                "source_codec_structure_match_rows": 2,
                "source_codec_passed": True,
                "source_codec_failure_examples": [],
                "calibration_loaded": False,
                "test_loaded": False,
            }
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return payload

    def forbidden_teacher(*_args, **_kwargs):
        raise AssertionError("Codec probe must not load the RF teacher.")

    def forbidden_train(*_args, **_kwargs):
        calls["train"] += 1
        raise AssertionError("Codec probe must not invoke training.")

    monkeypatch.setattr(
        build_cli,
        "OfficialGlobalGCEMutagenicityGenerator",
        ProbeGenerator,
    )
    monkeypatch.setattr(build_cli, "TeacherSemanticScorer", forbidden_teacher)
    monkeypatch.setattr(
        build_cli,
        "build_mutagenicity_train_pool",
        forbidden_train,
    )
    exit_code = build_cli.main(
        [
            "--train-csv",
            str(fixture_paths["train"]),
            "--teacher-path",
            str(
                fixture_paths["output"].parent
                / "teacher-must-not-be-read.pkl"
            ),
            "--official-root",
            str(fixture_paths["official"]),
            "--native-train-csv",
            str(fixture_paths["train"]),
            "--output-dir",
            str(fixture_paths["output"]),
            "--expected-parent-count",
            "3",
            "--parent-limit",
            "2",
            "--codec-probe-only",
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert calls == {"probe": 1, "train": 0}
    assert "[MUTAGENICITY_GLOBALGCE_CODEC_PROBE_OK]" in captured
    assert "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_BUILD_OK]" not in captured
    assert (fixture_paths["output"] / "codec_probe_summary.json").is_file()


def test_build_cli_does_not_print_success_after_empty_pool_error(
    fixture_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.baselines.globalgce import (
        build_mutagenicity_train_pool as build_cli,
    )

    class AvailableTeacher:
        available = True
        availability_reason = "ok"

        def __init__(self, _path):
            pass

    class Generator:
        def __init__(self, _official_root, *, native_train_csv=None):
            del native_train_csv

    def empty_build(**_kwargs):
        raise GlobalGCEEmptyCandidateUniverseError("empty")

    monkeypatch.setattr(build_cli, "TeacherSemanticScorer", AvailableTeacher)
    monkeypatch.setattr(
        build_cli,
        "OfficialGlobalGCEMutagenicityGenerator",
        Generator,
    )
    monkeypatch.setattr(build_cli, "build_mutagenicity_train_pool", empty_build)
    with pytest.raises(GlobalGCEEmptyCandidateUniverseError):
        build_cli.main(
            [
                "--train-csv",
                str(fixture_paths["train"]),
                "--teacher-path",
                str(fixture_paths["teacher"]),
                "--official-root",
                str(fixture_paths["official"]),
                "--output-dir",
                str(fixture_paths["output"]),
                "--expected-parent-count",
                "3",
            ]
        )
    assert (
        "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_BUILD_OK]"
        not in capsys.readouterr().out
    )


def test_generator_cannot_return_parent_outside_train(
    fixture_paths: dict[str, Path],
) -> None:
    class BadGenerator(_FakeGenerator):
        def generate(self, parents, **kwargs):
            result = super().generate(parents, **kwargs)
            result.records[0]["source_parent_id"] = "not-in-train"
            return result

    with pytest.raises(ValueError, match="non-train source parent"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=BadGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
            ),
        )


def test_completed_run_refuses_overwrite(fixture_paths: dict[str, Path]) -> None:
    _build(fixture_paths)
    with pytest.raises(FileExistsError, match="cannot be rerun"):
        _build(fixture_paths)


def test_resume_rejects_configuration_mismatch(
    fixture_paths: dict[str, Path],
) -> None:
    with pytest.raises(RuntimeError, match="intentional"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=_FakeGenerator(fail=True),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
                seed=13,
            ),
        )
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=_FakeGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
                seed=99,
            ),
        )
