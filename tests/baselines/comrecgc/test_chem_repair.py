from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from src.baselines.comrecgc.chem_repair import (  # noqa: E402
    REPAIR_METHOD,
    apply_action_to_graph,
    repair_candidate,
)
from src.baselines.comrecgc.graph_trace import stable_graph_sha256  # noqa: E402
from src.baselines.gcfexplainer_mutagenicity_adapter import (  # noqa: E402
    MutagenicityGraphSchema,
    StrictMolecule,
    decode_generated_fullgraph,
    encode_source_graph,
)


@dataclass
class Graph:
    x: object
    edge_index: object
    num_nodes: int
    comrecgc_node_origin: object
    gcf_node_origin: object
    comrecgc_trace_node_ids: list[str]
    edge_attr: object | None = None

    def clone(self) -> "Graph":
        return copy.deepcopy(self)


def schema() -> MutagenicityGraphSchema:
    return MutagenicityGraphSchema(
        atom_vocabulary=(6, 8),
        feature_atomic_numbers=(6, 8, 1),
        formal_charge_vocabulary=(0,),
        aromaticity_vocabulary=(False,),
        bond_type_vocabulary=("SINGLE",),
        max_num_nodes=64,
    )


def source_record(smiles: str = "C") -> dict:
    return encode_source_graph(
        StrictMolecule(
            molecule_id="fixture",
            smiles=smiles,
            canonical_smiles=smiles,
            label=1,
            split="train",
            semantic_label="mutagenic",
            source_row_index=0,
            source_path="fixture.csv",
        ),
        schema(),
    )


def source_graph(record: dict) -> Graph:
    count = int(record["num_nodes"])
    origins = torch.arange(count, dtype=torch.long)
    return Graph(
        x=torch.tensor(record["x"], dtype=torch.float32),
        edge_index=torch.tensor(record["edge_index"], dtype=torch.long),
        num_nodes=count,
        comrecgc_node_origin=origins,
        gcf_node_origin=origins.clone(),
        comrecgc_trace_node_ids=[f"fixture:source:{index}" for index in range(count)],
    )


def transition(
    action: list[object], source_ids: list[str], target_ids: list[str]
) -> dict[str, object]:
    return {
        "action_resolution": "exact",
        "action": action,
        "source_node_ids": source_ids,
        "target_node_ids": target_ids,
    }


def test_invalid_action_rolls_back_and_dependent_action_is_skipped() -> None:
    record = source_record("C")
    graph = source_graph(record)
    initial_sha = stable_graph_sha256(graph)
    initial_ids = list(graph.comrecgc_trace_node_ids)
    added_ids = [*initial_ids, "fixture:new:carbon"]
    actions = [
        transition(["NA", 0, 0], initial_ids, added_ids),
        transition(["NLC", len(initial_ids), 1], added_ids, added_ids),
    ]

    result = repair_candidate(
        source_graph=graph,
        source_record=record,
        schema=schema(),
        actions=actions,
    )

    assert result.repair_success is True
    assert result.repair_noop is True
    assert result.retained_action_count == 0
    assert result.skipped_action_count == 2
    assert result.dependent_action_skip_count == 1
    assert result.output_graph_sha256 == initial_sha
    assert [row["status"] for row in result.action_records] == [
        "skipped_invalid_action",
        "skipped_dependent_action",
    ]


def test_repair_is_deterministic_and_one_output_per_raw_candidate() -> None:
    record = source_record("C")
    graph = source_graph(record)
    ids = list(graph.comrecgc_trace_node_ids)
    action = transition(["NLC", 1, 1], ids, ids)

    first = repair_candidate(
        source_graph=graph,
        source_record=record,
        schema=schema(),
        actions=[action],
    )
    second = repair_candidate(
        source_graph=graph,
        source_record=record,
        schema=schema(),
        actions=[action],
    )

    assert first.output_graph_sha256 == second.output_graph_sha256
    assert first.canonical_smiles == second.canonical_smiles
    assert first.action_records == second.action_records
    assert isinstance(first.graph, Graph)


def test_node_lineage_after_remove_is_exact() -> None:
    record = source_record("C")
    graph = source_graph(record)
    source_ids = list(graph.comrecgc_trace_node_ids)
    target_ids = [value for index, value in enumerate(source_ids) if index != 1]
    removed = apply_action_to_graph(
        graph,
        ["NR", 1],
        target_node_ids=target_ids,
    )

    assert removed.num_nodes == graph.num_nodes - 1
    assert removed.comrecgc_trace_node_ids == target_ids
    assert removed.comrecgc_node_origin.tolist() == [0, 2, 3, 4]
    assert int(removed.edge_index.max()) < removed.num_nodes


def test_new_untyped_edge_uses_shared_single_bond_decoder_policy() -> None:
    record = source_record("CCCC")
    graph = source_graph(record)
    # Remove one explicit H from each terminal carbon before closing the ring.
    atom_sidecar = list(record["atom_sidecar"])
    terminal_hydrogens = []
    for terminal in (0, 3):
        terminal_hydrogens.append(
            next(
                int(atom["graph_node_index"])
                for atom in atom_sidecar
                if int(atom["atomic_num"]) == 1
                and int(atom.get("attached_original_atom_index", -1)) == terminal
            )
        )
    for origin in sorted(terminal_hydrogens, reverse=True):
        current_origins = graph.comrecgc_node_origin.tolist()
        current_index = current_origins.index(origin)
        target_ids = [
            value
            for index, value in enumerate(graph.comrecgc_trace_node_ids)
            if index != current_index
        ]
        graph = apply_action_to_graph(
            graph,
            ["NR", current_index],
            target_node_ids=target_ids,
        )
    positions = {
        origin: index for index, origin in enumerate(graph.comrecgc_node_origin.tolist())
    }
    graph = apply_action_to_graph(
        graph,
        ["EA", positions[0], positions[3]],
        target_node_ids=graph.comrecgc_trace_node_ids,
    )
    decoded = decode_generated_fullgraph(graph, source_record=record, schema=schema())

    assert decoded.decode_ok is True
    assert decoded.projected_new_edge_count == 1
    assert decoded.canonical_smiles == "C1CCC1"


def test_repair_module_has_no_rf_or_wnode_selection_dependency() -> None:
    import src.baselines.comrecgc.chem_repair as module

    source = inspect.getsource(module).lower()
    assert REPAIR_METHOD == "COMRECGC-Adapted-DeterministicChemRepair"
    assert "teachersemanticscorer" not in source
    assert "molclr" not in source
    assert "wnode" not in source
    assert "rf_prob" not in source
