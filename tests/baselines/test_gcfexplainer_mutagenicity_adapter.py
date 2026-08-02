from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    EXPECTED_GENERATION_SOURCE_ROWS,
    EXPECTED_MODEL_TRAIN_ROWS,
    EXPECTED_MODEL_VAL_ROWS,
    GCFExplainerMutagenicityCodecError,
    GCFExplainerVRRWConfigError,
    MutagenicityGraphSchema,
    OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS,
    StrictMolecule,
    checkpoint_is_aids,
    decode_generated_fullgraph,
    derive_schema,
    encode_source_graph,
    filter_native_rank_candidates,
    graph_lineage_neighbor_wrapper,
    get_vrrw_profile_contract,
    load_strict_molecules,
    reconstruct_source_graph,
    select_codec_probe_records,
    stable_graph_candidate_id,
    validate_gnn_profile,
    validate_vrrw_profile,
)
from src.baselines.gcfexplainer_mutagenicity_runtime import (
    _clear_vrrw_config_failure_for_resume,
)


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _row(smiles: str, index: int = 0, *, label: int = 1, split: str = "train") -> StrictMolecule:
    return StrictMolecule(
        molecule_id=f"MUT_{index:04d}",
        smiles=smiles,
        canonical_smiles=_canonical(smiles),
        label=label,
        split=split,
        semantic_label="mutagenic" if label == 1 else "non_mutagenic",
        source_row_index=index,
        source_path="strict_train.csv" if split == "train" else "strict_val.csv",
    )


@pytest.fixture()
def schema() -> MutagenicityGraphSchema:
    return MutagenicityGraphSchema(
        atom_vocabulary=OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS,
        feature_atomic_numbers=OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS,
        formal_charge_vocabulary=(-1, 0, 1),
        aromaticity_vocabulary=(False, True),
        bond_type_vocabulary=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
        max_num_nodes=256,
    )


def _graph_from_record(
    record: dict,
    *,
    atom_changes: dict[int, int] | None = None,
    remove_origins: set[int] | None = None,
    remove_edges: set[tuple[int, int]] | None = None,
    add_edges: set[tuple[int, int]] | None = None,
    num_nodes_override: int | None = None,
) -> SimpleNamespace:
    atom_changes = atom_changes or {}
    removed = remove_origins or set()
    kept = [index for index in range(record["num_nodes"]) if index not in removed]
    remap = {origin: index for index, origin in enumerate(kept)}
    x = [list(record["x"][origin]) for origin in kept]
    feature_atoms = list(OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS)
    for origin, atomic_num in atom_changes.items():
        generated_index = remap[origin]
        x[generated_index] = [0.0] * len(feature_atoms)
        x[generated_index][feature_atoms.index(atomic_num)] = 1.0
    source_pairs = {
        (int(item["begin"]), int(item["end"]))
        for item in record["bond_sidecar"]
    }
    source_pairs -= {tuple(sorted(pair)) for pair in (remove_edges or set())}
    source_pairs |= {tuple(sorted(pair)) for pair in (add_edges or set())}
    pairs = [
        (remap[a], remap[b])
        for a, b in sorted(source_pairs)
        if a in remap and b in remap
    ]
    directed = sorted([edge for a, b in pairs for edge in ((a, b), (b, a))])
    return SimpleNamespace(
        x=x,
        edge_index=[
            [edge[0] for edge in directed],
            [edge[1] for edge in directed],
        ],
        num_nodes=len(x) if num_nodes_override is None else num_nodes_override,
        gcf_node_origin=kept,
        gcf_origin_index=[0],
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "CCl",
        "CBr",
        "CP",
        "CI",
        "[NH4+]",
        "O=C[O-]",
        "c1cc[nH]c1",
        "C#N",
        "F/C=C/F",
        "N[C@@H](C)C(=O)O",
    ],
)
def test_source_graph_round_trip_is_chemically_exact(schema, smiles: str) -> None:
    record = encode_source_graph(_row(smiles), schema)
    _mol, checks = reconstruct_source_graph(record, schema)
    assert checks["round_trip_passed"] is True
    assert checks["formal_charges_exact"] is True
    assert checks["aromaticity_exact"] is True
    assert checks["explicit_hs_exact"] is True
    assert checks["no_implicit_exact"] is True
    assert checks["chiral_tags_exact"] is True
    assert checks["bond_stereo_exact"] is True


def test_official_feature_vocabulary_is_train_derived_and_ordered() -> None:
    train_smiles = ["C", "O", "Cl", "N", "F", "Br", "S", "P", "I", "[NH4+]", "[O-]"]
    train = [_row(smiles, index) for index, smiles in enumerate(train_smiles)]
    val = [_row("CO", 100, split="val")]
    schema = derive_schema(train, val)
    assert schema.feature_atomic_numbers == OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS
    assert schema.formal_charge_vocabulary == (-1, 0, 1)
    assert schema.node_feature_dim == 10


def test_validation_cannot_expand_train_only_vocabulary() -> None:
    train = [
        _row(smiles, index)
        for index, smiles in enumerate(("C", "O", "Cl", "N", "F", "Br", "S", "P", "I"))
    ]
    with pytest.raises(ValueError, match="train-unseen"):
        derive_schema(train, [_row("[Na+]", 100, split="val")])


def test_strict_counts_and_generation_contract_are_frozen() -> None:
    assert EXPECTED_MODEL_TRAIN_ROWS == 2885
    assert EXPECTED_MODEL_VAL_ROWS == 355
    assert EXPECTED_GENERATION_SOURCE_ROWS == 1448
    assert validate_gnn_profile("full", epochs=1000, train_rows=2885, val_rows=355).value == "full"
    assert validate_vrrw_profile(
        "full",
        parent_limit=1448,
        m=50000,
        alpha=1.0,
        theta=0.05,
        seed=13,
    ).value == "full"


def test_smoke_and_full_profile_guards() -> None:
    validate_gnn_profile("smoke", epochs=5, train_rows=64, val_rows=32)
    validate_vrrw_profile(
        "smoke",
        parent_limit=64,
        m=500,
        alpha=1.0,
        theta=0.05,
        seed=13,
    )
    validate_vrrw_profile(
        "smoke",
        parent_limit=64,
        m=1000,
        alpha=1.0,
        theta=0.05,
        seed=13,
    )
    with pytest.raises(ValueError):
        validate_gnn_profile("full", epochs=5, train_rows=2885, val_rows=355)
    with pytest.raises(GCFExplainerVRRWConfigError):
        validate_vrrw_profile(
            "full",
            parent_limit=64,
            m=50000,
            alpha=1.0,
            theta=0.05,
            seed=13,
        )
    with pytest.raises(GCFExplainerVRRWConfigError):
        validate_vrrw_profile(
            "full",
            parent_limit=1448,
            m=50000,
            alpha=0.5,
            theta=0.05,
            seed=13,
        )


def test_vrrw_profile_defaults_and_actual_values_are_auditable() -> None:
    smoke = get_vrrw_profile_contract("smoke")
    full = get_vrrw_profile_contract("full")
    assert (smoke.parent_limit, smoke.default_m, smoke.allowed_m) == (
        64,
        500,
        (500, 1000),
    )
    assert (full.parent_limit, full.default_m, full.allowed_m) == (
        1448,
        50000,
        (50000,),
    )
    with pytest.raises(GCFExplainerVRRWConfigError) as caught:
        validate_vrrw_profile(
            "smoke",
            parent_limit=1448,
            m=50000,
            alpha=1.0,
            theta=0.05,
            seed=13,
        )
    assert caught.value.details["parent_limit"] == 1448
    assert caught.value.details["M"] == 50000
    assert caught.value.details["expected_parent_limit"] == 64
    assert caught.value.details["expected_M"] == [500, 1000]
    assert "parent_limit=1448" in str(caught.value)
    assert "M=50000" in str(caught.value)


@pytest.mark.parametrize(
    ("profile", "parent_limit", "m", "mismatch"),
    (
        ("smoke", 64, 50000, "M"),
        ("smoke", 1448, 500, "parent_limit"),
        ("full", 1448, 500, "M"),
        ("full", 64, 50000, "parent_limit"),
    ),
)
def test_vrrw_profile_rejects_cross_profile_values(
    profile: str,
    parent_limit: int,
    m: int,
    mismatch: str,
) -> None:
    with pytest.raises(GCFExplainerVRRWConfigError) as caught:
        validate_vrrw_profile(
            profile,
            parent_limit=parent_limit,
            m=m,
            alpha=1.0,
            theta=0.05,
            seed=13,
        )
    assert mismatch in caught.value.details["mismatched_fields"]


def test_resume_discards_only_stale_pre_generation_config_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "vrrw"
    run_dir.mkdir()
    (run_dir / "_RUN_FAILED.json").write_text(
        json.dumps({"stage": "vrrw_config", "M": 50000}),
        encoding="utf-8",
    )
    (run_dir / "failure_summary.json").write_text(
        json.dumps({"stage": "vrrw_config", "M": 50000}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"profile": "smoke", "M": 50000}),
        encoding="utf-8",
    )
    assert _clear_vrrw_config_failure_for_resume(run_dir, resume=True) is True
    assert list(run_dir.iterdir()) == []


def test_resume_never_discards_started_vrrw_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "vrrw"
    run_dir.mkdir()
    (run_dir / "_RUN_FAILED.json").write_text(
        json.dumps({"stage": "vrrw_config"}),
        encoding="utf-8",
    )
    (run_dir / "vrrw_progress.json").write_text(
        json.dumps({"current_step": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="will not discard"):
        _clear_vrrw_config_failure_for_resume(run_dir, resume=True)


def test_csv_loader_enforces_split_teacher_and_forbidden_paths(tmp_path: Path) -> None:
    path = tmp_path / "train_source.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("molecule_id", "smiles", "label", "split", "teacher_pred", "teacher_correct"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "molecule_id": "M1",
                "smiles": "C",
                "label": 1,
                "split": "train",
                "teacher_pred": 1,
                "teacher_correct": "true",
            }
        )
    rows = load_strict_molecules(path, expected_split="train", expected_label=1, expected_rows=1)
    assert [row.molecule_id for row in rows] == ["M1"]
    forbidden = tmp_path / "test_source_label1.csv"
    forbidden.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_strict_molecules(forbidden, expected_split="train", expected_label=1, expected_rows=1)


def test_probe_sampling_is_category_first_and_deterministic(schema) -> None:
    smiles = ["CCl", "CBr", "CP", "CI", "[NH4+]", "c1cc[nH]c1", "C#N", "C=C"]
    records = [encode_source_graph(_row(value, index), schema) for index, value in enumerate(smiles)]
    selected1, coverage1 = select_codec_probe_records(records, limit=len(records), require_all=False)
    selected2, coverage2 = select_codec_probe_records(list(reversed(records)), limit=len(records), require_all=False)
    assert [row["molecule_id"] for row in selected1] == [row["molecule_id"] for row in selected2]
    assert coverage1 == coverage2
    assert coverage1["br"] and coverage1["i"] and coverage1["n_h"]


@pytest.mark.parametrize("source,decoded", [("O", 6), ("N", 6), ("Cl", 6), ("Br", 6)])
def test_generated_identity_change_uses_decoded_atom_without_source_hydrogen_state(schema, source: str, decoded: int) -> None:
    record = encode_source_graph(_row(source), schema)
    graph = _graph_from_record(record, atom_changes={0: decoded})
    result = decode_generated_fullgraph(graph, source_record=record, schema=schema)
    assert result.decode_ok is True
    assert result.reset_atom_state_count >= 1
    assert "explicit-H state cannot be inferred" not in result.failure_reason


def test_unchanged_graph_inherits_source_atom_state(schema) -> None:
    record = encode_source_graph(_row("c1cc[nH]c1"), schema)
    graph = _graph_from_record(record)
    result = decode_generated_fullgraph(graph, source_record=record, schema=schema)
    assert result.decode_ok is True
    assert result.inherited_atom_state_count == record["num_nodes"]
    assert result.canonical_smiles == record["canonical_smiles"]


def test_changed_incident_environment_resets_hydrogen_and_uses_provisional_single(schema) -> None:
    record = encode_source_graph(_row("CCC"), schema)
    hydrogen_by_carbon = {
        atom["attached_original_atom_index"]: atom["graph_node_index"]
        for atom in record["atom_sidecar"]
        if atom["atomic_num"] == 1 and atom["attached_original_atom_index"] is not None
    }
    graph = _graph_from_record(
        record,
        remove_origins={hydrogen_by_carbon[0], hydrogen_by_carbon[2]},
        add_edges={(0, 2)},
    )
    result = decode_generated_fullgraph(graph, source_record=record, schema=schema)
    assert result.decode_ok is True
    assert result.projected_new_edge_count == 1
    assert result.reset_atom_state_count >= 2


def test_invalid_generated_valence_is_rejected_after_sanitize(schema) -> None:
    record = encode_source_graph(_row("O"), schema)
    graph = _graph_from_record(record, atom_changes={0: 9})
    result = decode_generated_fullgraph(graph, source_record=record, schema=schema)
    assert result.decode_ok is False
    assert result.failure_reason == "generated_valence_sanitize_failed"


def test_padded_or_mismatched_nodes_are_not_decoded_as_carbon(schema) -> None:
    record = encode_source_graph(_row("C"), schema)
    graph = _graph_from_record(record, num_nodes_override=record["num_nodes"] + 1)
    result = decode_generated_fullgraph(graph, source_record=record, schema=schema)
    assert result.decode_ok is False
    assert result.failure_reason == "generated_invalid_active_node_mask"


def test_atom_feature_index_mapping_is_exact(schema) -> None:
    record = encode_source_graph(_row("O"), schema)
    for atomic_num in OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS:
        graph = _graph_from_record(record, atom_changes={0: atomic_num})
        active = graph.x[0]
        assert active.index(1.0) == list(OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS).index(atomic_num)


def test_aids_checkpoint_is_rejected() -> None:
    assert checkpoint_is_aids("outputs/aids/gnn/model_best.pth") is True
    assert checkpoint_is_aids("outputs/mutagenicity/gnn/model_best.pth") is False


class _FakeTeacher:
    available = True

    def score_smiles(self, smiles: str, label: int | None = None, **_kwargs):
        target = "C" in smiles
        return {
            "teacher_result_ok": True,
            "teacher_label": 0 if target else 1,
            "teacher_prob": 0.9 if label == (0 if target else 1) else 0.1,
        }


def test_rf_filter_preserves_native_rank_and_stable_candidate_ids(schema) -> None:
    source = encode_source_graph(_row("C"), schema)
    graph = _graph_from_record(source)
    native = [
        {"native_rank": 2, "candidate_id": "second"},
        {"native_rank": 1, "candidate_id": "first"},
    ]
    selected, skipped = filter_native_rank_candidates(
        native,
        [graph, graph],
        [source],
        schema,
        _FakeTeacher(),
        top_k=1,
    )
    assert [row["skip_reason"] for row in skipped] == ["canonical_duplicate"]
    assert selected[0]["native_rank"] == 1
    assert selected[0]["selection_method"] == "native_gcf_summary_rank_filtered_by_validity"
    assert selected[0]["selection_performed_in_eval"] is False
    assert selected[0]["candidate_id"].startswith("GCFMOL_")
    assert stable_graph_candidate_id(graph) == stable_graph_candidate_id(graph)


def test_lineage_wrapper_does_not_change_official_graph_tensors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class Graph:
        def __init__(self):
            self.x = torch.tensor([[1.0, 0.0]])
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.num_nodes = 1
            self.gcf_node_origin = torch.tensor([7])
            self.gcf_origin_index = torch.tensor([3])

    def official(graph, _action):
        candidate = Graph()
        candidate.x = torch.vstack((graph.x, torch.tensor([[0.0, 1.0]])))
        candidate.edge_index = torch.tensor([[0, 1], [1, 0]])
        candidate.num_nodes = 2
        return candidate

    candidate = graph_lineage_neighbor_wrapper(official)(Graph(), ("NA", 0, 1))
    assert candidate.x.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert candidate.edge_index.tolist() == [[0, 1], [1, 0]]
    assert candidate.gcf_node_origin.tolist() == [7, -1]
