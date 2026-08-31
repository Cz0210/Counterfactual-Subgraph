from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src.baselines.comrecgc.exporter import _aids_schema_and_record
from src.baselines.comrecgc.mutagenicity_chemistry_audit import (
    AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV,
    AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY,
    _aids_source_noop_identity,
)
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    decode_generated_fullgraph,
    reconstruct_source_graph,
)


class _Graph(SimpleNamespace):
    def clone(self) -> "_Graph":
        return copy.deepcopy(self)


def _aids_graph(smiles: str) -> tuple[_Graph, tuple[str, ...]]:
    torch = pytest.importorskip("torch")
    Chem = pytest.importorskip("rdkit.Chem")

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    vocabulary = tuple(
        sorted({str(atom.GetSymbol()) for atom in molecule.GetAtoms()})
    )
    symbol_to_index = {symbol: index for index, symbol in enumerate(vocabulary)}
    x = torch.zeros((molecule.GetNumAtoms(), len(vocabulary)), dtype=torch.float32)
    for atom in molecule.GetAtoms():
        x[atom.GetIdx(), symbol_to_index[atom.GetSymbol()]] = 1.0
    directed_edges: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        directed_edges.extend(((begin, end), (end, begin)))
    edge_index = (
        torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
        if directed_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    lineage = torch.arange(molecule.GetNumAtoms(), dtype=torch.long)
    graph = _Graph(
        x=x,
        edge_index=edge_index,
        num_nodes=int(molecule.GetNumAtoms()),
        gcf_node_origin=lineage.clone(),
        comrecgc_node_origin=lineage.clone(),
        smiles=smiles,
        comrecgc_source_smiles=smiles,
        comrecgc_parent_id="AIDS_HIV_MULTICOMPONENT_FIXTURE",
        comrecgc_source_index=0,
        comrecgc_project_label=1,
    )
    return graph, vocabulary


def _identity(
    graph: _Graph,
    vocabulary: tuple[str, ...],
    *,
    authorized: bool,
    clone_graph: _Graph | None = None,
    loaded_graph: _Graph | None = None,
    batched_graph: _Graph | None = None,
    record_updates: dict[str, object] | None = None,
) -> tuple[dict[str, object], object]:
    Chem = pytest.importorskip("rdkit.Chem")

    schema, record = _aids_schema_and_record(graph, vocabulary)
    if record_updates:
        record = {**record, **record_updates}
    molecule, _diagnostics = reconstruct_source_graph(record, schema)
    reconstructed = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    decoded = decode_generated_fullgraph(
        graph,
        source_record=record,
        schema=schema,
    )
    result = _aids_source_noop_identity(
        graph=graph,
        clone_graph=clone_graph or graph.clone(),
        loaded_graph=loaded_graph or graph.clone(),
        batched_graph=batched_graph or graph.clone(),
        record=record,
        reconstructed_smiles=reconstructed,
        decoded=decoded,
        multicomponent_authorized=authorized,
    )
    return result, decoded


def test_authorized_multicomponent_source_noop_preserves_multiplicity() -> None:
    graph, vocabulary = _aids_graph("C.C.O")

    identity, decoded = _identity(graph, vocabulary, authorized=True)

    assert decoded.decode_ok is False
    assert decoded.failure_reason == "generated_disconnected_or_empty"
    assert identity["identity_mode"] == AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY
    assert identity["canonical_isomeric_component_multiset"] == ["C", "C", "O"]
    assert identity["component_count"] == 3
    assert identity["graph_component_count"] == 3
    assert identity["component_multiplicity_preserved"] is True
    assert identity["noop_roundtrip_ok"] is True


def test_single_component_source_retains_existing_decoder_contract() -> None:
    graph, vocabulary = _aids_graph("CCO")

    identity, decoded = _identity(graph, vocabulary, authorized=False)

    assert decoded.decode_ok is True
    assert identity["identity_mode"] == "single_component_generated_decoder_v1"
    assert identity["component_count"] == 1
    assert identity["noop_roundtrip_ok"] is True


def test_multicomponent_source_noop_fails_without_exact_authorization() -> None:
    graph, vocabulary = _aids_graph("CC.O")

    with pytest.raises(ValueError, match=AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV):
        _identity(graph, vocabulary, authorized=False)


@pytest.mark.parametrize("closure_name", ["clone", "save_load", "batch_unbatch"])
def test_source_noop_rejects_atom_tensor_drift(closure_name: str) -> None:
    graph, vocabulary = _aids_graph("CC.O")
    changed = graph.clone()
    changed.x[0] = 0.0
    variants = {
        "clone_graph": graph.clone(),
        "loaded_graph": graph.clone(),
        "batched_graph": graph.clone(),
    }
    variants[
        {
            "clone": "clone_graph",
            "save_load": "loaded_graph",
            "batch_unbatch": "batched_graph",
        }[closure_name]
    ] = changed

    identity, _decoded = _identity(
        graph,
        vocabulary,
        authorized=True,
        **variants,
    )

    assert identity["atom_tensor_closure_exact"] is False
    assert identity["graph_hash_closure_exact"] is False
    assert identity["noop_roundtrip_ok"] is False


def test_source_noop_rejects_node_lineage_drift_even_when_graph_hash_matches() -> None:
    graph, vocabulary = _aids_graph("CC.O")
    changed = graph.clone()
    changed.comrecgc_node_origin[0] = -1

    identity, _decoded = _identity(
        graph,
        vocabulary,
        authorized=True,
        clone_graph=changed,
    )

    assert identity["node_lineage_exact"] is False
    assert identity["graph_hash_closure_exact"] is True
    assert identity["noop_roundtrip_ok"] is False


def test_source_noop_rejects_component_multiplicity_drift() -> None:
    graph, vocabulary = _aids_graph("C.C.O")

    identity, _decoded = _identity(
        graph,
        vocabulary,
        authorized=True,
        record_updates={"original_smiles": "C.O"},
    )

    assert identity["component_multiset_exact"] is False
    assert identity["component_count_exact"] is False
    assert identity["noop_roundtrip_ok"] is False


def test_connected_non_noop_candidate_is_not_admitted_by_source_identity() -> None:
    torch = pytest.importorskip("torch")
    graph, vocabulary = _aids_graph("CC.O")
    _schema, source_record = _aids_schema_and_record(graph, vocabulary)
    candidate = graph.clone()
    candidate.edge_index = torch.cat(
        (
            candidate.edge_index,
            torch.tensor([[1, 2], [2, 1]], dtype=torch.long).T,
        ),
        dim=1,
    )
    schema, _candidate_record = _aids_schema_and_record(graph, vocabulary)
    decoded = decode_generated_fullgraph(
        candidate,
        source_record=source_record,
        schema=schema,
    )
    assert decoded.decode_ok is True

    Chem = pytest.importorskip("rdkit.Chem")
    source_molecule, _diagnostics = reconstruct_source_graph(source_record, schema)
    identity = _aids_source_noop_identity(
        graph=candidate,
        clone_graph=candidate.clone(),
        loaded_graph=candidate.clone(),
        batched_graph=candidate.clone(),
        record=source_record,
        reconstructed_smiles=Chem.MolToSmiles(
            source_molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        decoded=decoded,
        multicomponent_authorized=True,
    )

    assert identity["component_count_exact"] is False
    assert identity["bond_tensor_record_exact"] is False
    assert identity["decoder_contract_exact"] is False
    assert identity["noop_roundtrip_ok"] is False


def test_generated_decoder_remains_single_component_with_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, vocabulary = _aids_graph("CC.O")
    monkeypatch.setenv(AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV, "1")
    schema, record = _aids_schema_and_record(graph, vocabulary)

    decoded = decode_generated_fullgraph(
        graph,
        source_record=record,
        schema=schema,
    )

    assert decoded.decode_ok is False
    assert decoded.failure_reason == "generated_disconnected_or_empty"
