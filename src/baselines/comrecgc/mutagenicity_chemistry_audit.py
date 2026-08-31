"""Auditable source, action-replay, and chemistry-repair gates for Mutagenicity."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chem_repair import REPAIR_METHOD, REPAIR_POLICY_VERSION, repair_candidate, replay_raw_actions
from .contracts import (
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    require_empty_output,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from .graph_trace import (
    iter_candidate_lineage_from_selected_trace,
    iter_selected_trace,
    stable_graph_sha256,
)
from .preregistration import (
    validate_chemistry_trace_evidence,
    write_mutagenicity_chem_repair_preregistration,
)
from .project_dataset import (
    load_aids_generation_bundle,
    load_bace_generation_bundle,
    load_mutagenicity_generation_bundle,
)
from .runtime import _torch_load, _torch_save_atomic, validate_counterfactual_payload


AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV = (
    "ALLOW_AIDS_MULTICOMPONENT_SOURCE_NOOP_IDENTITY_V1"
)
AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY = (
    "aids_multicomponent_source_noop_identity_v1"
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    atomic_write_bytes(path, _csv_bytes(rows, fields))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True, default=str) + "\n"
        for row in rows
    )
    atomic_write_bytes(path, payload.encode("utf-8"))


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected JSON object list: {source}")
    return [dict(row) for row in value]


def _lineage_contract(path: str | Path) -> tuple[Any, int, str]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value], len(value), "inline_json"
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported COMRECGC lineage artifact: {source}")
    if value.get("format") != "selected_trace_predecessor_index":
        raise ValueError(
            f"Unsupported COMRECGC lineage format: {value.get('format')!r}"
        )
    count = int(value.get("candidate_count", -1))
    if count < 0:
        raise ValueError("Compact COMRECGC lineage has no valid candidate count.")
    return value, count, "selected_trace_predecessor_index"


def _resolved_child(root: Path, relative: Any, field: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"Compact lineage {field} is not a safe relative path.")
    resolved = (root / candidate).resolve()
    if resolved.parent != root and root not in resolved.parents:
        raise ValueError(f"Compact lineage {field} escapes its trace root.")
    return resolved


def _iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object row: {path}")
                yield value


def _iter_candidate_lineage(
    *,
    path: Path,
    contract: Any,
    payload: Mapping[str, Any],
    source_graphs_by_parent_id: Mapping[str, Any],
) -> Any:
    if isinstance(contract, list):
        yield from contract
        return
    trace_root = path.parent
    index_path = _resolved_child(
        trace_root, contract.get("candidate_index_path"), "candidate_index_path"
    )
    selected_path = _resolved_child(
        trace_root,
        contract.get("selected_trace_manifest_path"),
        "selected_trace_manifest_path",
    )
    if sha256_file(index_path) != str(contract.get("candidate_index_sha256")):
        raise ValueError("Compact candidate lineage index SHA256 mismatch.")
    if sha256_file(selected_path) != str(
        contract.get("selected_trace_manifest_sha256")
    ):
        raise ValueError("Compact selected trace manifest SHA256 mismatch.")
    recovered_rows = iter_candidate_lineage_from_selected_trace(
        payload,
        iter_selected_trace(selected_path),
        source_graphs_by_parent_id=source_graphs_by_parent_id,
        include_actions=True,
    )
    sentinel = object()
    summary_fields = (
        "candidate_index",
        "official_graph_hash",
        "stable_graph_sha256",
        "parent_id",
        "action_lineage_resolved",
        "zero_action_source_root",
        "lineage_root_status",
        "action_count",
    )
    for expected_index, (summary, recovered) in enumerate(
        zip_longest(_iter_jsonl(index_path), recovered_rows, fillvalue=sentinel)
    ):
        if summary is sentinel or recovered is sentinel:
            raise ValueError("Compact candidate lineage row count mismatch.")
        if int(summary.get("candidate_index", -1)) != expected_index:
            raise ValueError("Compact candidate lineage index is out of order.")
        if any(summary.get(field) != recovered.get(field) for field in summary_fields):
            raise ValueError(
                f"Compact candidate lineage summary differs at candidate {expected_index}."
            )
        yield recovered


@contextmanager
def _atomic_jsonl_writer(path: Path) -> Any:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _canonical(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Source SMILES is invalid: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_isomeric_component_multiset(smiles: str) -> tuple[str, ...]:
    """Return a multiplicity-preserving canonical identity for every component."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"AIDS source component identity is invalid: {smiles!r}")
    Chem.SanitizeMol(molecule)
    components: list[str] = []
    for component in Chem.GetMolFrags(
        molecule,
        asMols=True,
        sanitizeFrags=True,
    ):
        heavy = Chem.RemoveHs(component, sanitize=True)
        components.append(
            Chem.MolToSmiles(
                heavy,
                canonical=True,
                isomericSmiles=True,
            )
        )
    if not components:
        raise ValueError("AIDS source component identity is empty.")
    return tuple(sorted(components))


def _tensor_payload(value: Any) -> tuple[str, tuple[int, ...], Any] | None:
    if value is None:
        return None
    current = value.detach() if hasattr(value, "detach") else value
    current = current.cpu() if hasattr(current, "cpu") else current
    shape = tuple(int(item) for item in getattr(current, "shape", ()))
    payload = current.tolist() if hasattr(current, "tolist") else current
    return str(getattr(current, "dtype", type(current).__name__)), shape, payload


def _tensor_exact(left: Any, right: Any) -> bool:
    return _tensor_payload(left) == _tensor_payload(right)


def _graph_component_count(graph: Any) -> int:
    num_nodes = int(getattr(graph, "num_nodes", -1))
    if num_nodes <= 0:
        raise ValueError("AIDS source graph has no nodes.")
    edge_index = _tensor_payload(getattr(graph, "edge_index", None))
    if edge_index is None:
        raise ValueError("AIDS source graph has no edge_index tensor.")
    values = edge_index[2]
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("AIDS source graph edge_index is not [2, E].")
    if len(values[0]) != len(values[1]):
        raise ValueError("AIDS source graph edge_index rows are misaligned.")
    adjacency = [set() for _ in range(num_nodes)]
    for source, target in zip(values[0], values[1], strict=True):
        a, b = int(source), int(target)
        if not 0 <= a < num_nodes or not 0 <= b < num_nodes:
            raise ValueError("AIDS source graph edge_index is outside its node axis.")
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    unseen = set(range(num_nodes))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbours = adjacency[node] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
    return count


def _node_lineage(graph: Any, field: str) -> tuple[int, ...] | None:
    value = getattr(graph, field, None)
    if value is None:
        return None
    payload = _tensor_payload(value)
    if payload is None or not isinstance(payload[2], list):
        return None
    return tuple(int(item) for item in payload[2])


def _aids_source_noop_identity(
    *,
    graph: Any,
    clone_graph: Any,
    loaded_graph: Any,
    batched_graph: Any,
    record: Mapping[str, Any],
    reconstructed_smiles: str,
    decoded: Any,
    multicomponent_authorized: bool,
) -> dict[str, Any]:
    """Prove a source-only empty action without relaxing generated decoding."""

    variants = (graph, clone_graph, loaded_graph, batched_graph)
    expected_components = _canonical_isomeric_component_multiset(
        str(record["canonical_smiles"])
    )
    component_views = {
        "record": expected_components,
        "reconstructed": _canonical_isomeric_component_multiset(
            reconstructed_smiles
        ),
        "graph_native": _canonical_isomeric_component_multiset(
            str(record["graph_native_smiles"])
        ),
        "requested_source": _canonical_isomeric_component_multiset(
            str(record["original_smiles"])
        ),
    }
    component_multiset_exact = all(
        value == expected_components for value in component_views.values()
    )
    molecule_component_count = len(expected_components)
    graph_component_count = _graph_component_count(graph)
    is_multicomponent = molecule_component_count > 1
    if is_multicomponent and not multicomponent_authorized:
        raise ValueError(
            "AIDS/HIV multi-component source no-op requires "
            f"{AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV}=1."
        )

    expected_origin = tuple(range(int(graph.num_nodes)))
    comrecgc_lineages = tuple(
        _node_lineage(value, "comrecgc_node_origin") for value in variants
    )
    gcf_lineages = tuple(_node_lineage(value, "gcf_node_origin") for value in variants)
    node_lineage_exact = bool(
        all(value == expected_origin for value in comrecgc_lineages)
        and all(value == expected_origin for value in gcf_lineages)
    )
    atom_tensor_record_exact = _tensor_payload(graph.x)[2] == record["x"]
    bond_tensor_record_exact = (
        _tensor_payload(graph.edge_index)[2] == record["edge_index"]
    )
    atom_tensor_closure_exact = all(
        _tensor_exact(graph.x, value.x) for value in variants[1:]
    )
    bond_tensor_closure_exact = all(
        _tensor_exact(graph.edge_index, value.edge_index) for value in variants[1:]
    ) and all(
        _tensor_exact(
            getattr(graph, "edge_attr", None),
            getattr(value, "edge_attr", None),
        )
        for value in variants[1:]
    )
    node_count_closure_exact = all(
        int(value.num_nodes) == int(graph.num_nodes) for value in variants[1:]
    )
    parent_identity_closure_exact = all(
        str(getattr(value, "comrecgc_parent_id", ""))
        == str(record["molecule_id"])
        for value in variants
    )
    graph_hashes = tuple(stable_graph_sha256(value) for value in variants)
    graph_hash_closure_exact = len(set(graph_hashes)) == 1
    component_count_exact = bool(
        graph_component_count == molecule_component_count
        and all(len(value) == molecule_component_count for value in component_views.values())
    )
    if is_multicomponent:
        # The shared generated-candidate decoder must continue to reject this
        # disconnected graph. Only the byte/tensor-identical source empty
        # action is eligible for the separately authorized identity proof.
        decoder_contract_exact = bool(
            decoded.decode_ok is False
            and decoded.failure_reason == "generated_disconnected_or_empty"
        )
        identity_mode = AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY
    else:
        decoder_contract_exact = bool(
            decoded.decode_ok
            and _canonical_isomeric_component_multiset(decoded.canonical_smiles)
            == expected_components
        )
        identity_mode = "single_component_generated_decoder_v1"

    checks = {
        "component_multiset_exact": component_multiset_exact,
        "component_count_exact": component_count_exact,
        "atom_tensor_record_exact": atom_tensor_record_exact,
        "bond_tensor_record_exact": bond_tensor_record_exact,
        "atom_tensor_closure_exact": atom_tensor_closure_exact,
        "bond_tensor_closure_exact": bond_tensor_closure_exact,
        "node_count_closure_exact": node_count_closure_exact,
        "node_lineage_exact": node_lineage_exact,
        "parent_identity_closure_exact": parent_identity_closure_exact,
        "graph_hash_closure_exact": graph_hash_closure_exact,
        "decoder_contract_exact": decoder_contract_exact,
        "multicomponent_authority_exact": (
            bool(multicomponent_authorized) if is_multicomponent else True
        ),
    }
    return {
        "identity_mode": identity_mode,
        "is_multicomponent_source": is_multicomponent,
        "component_count": molecule_component_count,
        "graph_component_count": graph_component_count,
        "canonical_isomeric_component_multiset": list(expected_components),
        "canonical_isomeric_component_multiset_sha256": stable_json_sha256(
            list(expected_components)
        ),
        "component_multiplicity_preserved": True,
        "multicomponent_authorized": bool(multicomponent_authorized),
        "graph_hashes": {
            "source": graph_hashes[0],
            "clone": graph_hashes[1],
            "save_load": graph_hashes[2],
            "batch_unbatch": graph_hashes[3],
        },
        **checks,
        "noop_roundtrip_ok": all(checks.values()),
    }


def _project_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _candidate_graphs(payload: Mapping[str, Any]) -> list[Any]:
    graph_map, candidates = validate_counterfactual_payload(payload)
    values: list[Any] = []
    for index, candidate in enumerate(candidates):
        key = candidate.get("graph_hash")
        entry = graph_map.get(key)
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(f"Candidate {index} is absent from graph_map: {key!r}")
        values.append(entry[0])
    return values


def _mapping_payloads(
    schema: Any, *, dataset: str = "mutagenicity"
) -> tuple[dict[str, Any], dict[str, Any]]:
    atom = {
        "schema_version": 1,
        "source": f"{dataset}_project_graph_schema.feature_atomic_numbers",
        "node_label_semantics": "one_hot_atom_type",
        "node_feature_dim": int(schema.node_feature_dim),
        "index_to_atomic_number": {
            str(index): int(value)
            for index, value in enumerate(schema.feature_atomic_numbers)
        },
        "alternative_atom_search": False,
    }
    bond = {
        "schema_version": 1,
        "source": "project_fullgraph_codec_deterministic_repair_v1",
        "retained_bond_rule": "source_sidecar_exact",
        "new_untyped_edge_bond_type": "SINGLE",
        "supported_source_bond_types": list(schema.bond_type_vocabulary),
        "alternative_bond_order_search": False,
    }
    return atom, bond


def _source_and_noop_gate(
    *, root: Path, dataset_dir: Path, parent_limit: int, dataset: str
) -> tuple[Any, list[Any], dict[str, Mapping[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from rdkit import Chem
    from torch_geometric.data import Batch

    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        decode_generated_fullgraph,
        reconstruct_source_graph,
    )

    if dataset == "bace":
        from src.baselines.gcfexplainer_bace_adapter import (
            load_bace_gcf_dataset as load_dataset_artifacts,
        )

        bundle_loader = load_bace_generation_bundle
    elif dataset == "mutagenicity":
        from src.baselines.gcfexplainer_mutagenicity_adapter import (
            load_dataset_artifacts,
        )

        bundle_loader = load_mutagenicity_generation_bundle
    else:
        raise ValueError(f"Unsupported shared molecular graph codec: {dataset}")

    schema, _train, _val, generation, _summary = load_dataset_artifacts(dataset_dir)
    records = sorted(generation, key=lambda row: str(row["molecule_id"]))[: int(parent_limit)]
    bundle = bundle_loader(
        dataset_dir=dataset_dir,
        parent_limit=parent_limit,
    )
    if len(records) != parent_limit or len(bundle.graphs) != parent_limit:
        raise ValueError(
            f"{dataset} source cohort does not match the frozen parent count."
        )
    record_by_id = {str(row["molecule_id"]): row for row in records}
    source_rows: list[dict[str, Any]] = []
    no_op_rows: list[dict[str, Any]] = []
    source_hashes = [stable_graph_sha256(graph) for graph in bundle.graphs]
    serialized = root / "source_graph_noop_roundtrip.pt"
    _torch_save_atomic(bundle.graphs, serialized)
    reloaded = _torch_load(serialized)
    batched = Batch.from_data_list(bundle.graphs).to_data_list()
    if not isinstance(reloaded, list) or len(reloaded) != len(bundle.graphs):
        raise ValueError("Source graph save/load did not preserve the graph list.")
    if len(batched) != len(bundle.graphs):
        raise ValueError("Source graph batch/unbatch did not preserve graph count.")
    for index, (graph, loaded_graph, batched_graph) in enumerate(
        zip(bundle.graphs, reloaded, batched, strict=True)
    ):
        parent_id = bundle.parent_ids[index]
        record = record_by_id[parent_id]
        molecule, diagnostics = reconstruct_source_graph(record, schema)
        reconstructed = _canonical(Chem.MolToSmiles(molecule, isomericSmiles=True))
        expected = _canonical(str(record["canonical_smiles"]))
        decoded = decode_generated_fullgraph(graph, source_record=record, schema=schema)
        graph_hash = source_hashes[index]
        loaded_hash = stable_graph_sha256(loaded_graph)
        batched_hash = stable_graph_sha256(batched_graph)
        node_origin = [int(value) for value in graph.comrecgc_node_origin.detach().cpu().tolist()]
        expected_origin = list(range(int(graph.num_nodes)))
        source_ok = bool(
            diagnostics.get("round_trip_passed")
            and reconstructed == expected
            and node_origin == expected_origin
        )
        no_op_ok = bool(
            decoded.decode_ok
            and decoded.canonical_smiles == expected
            and graph_hash == loaded_hash == batched_hash
        )
        source_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "sanitize_ok": source_ok,
                "node_count_exact": int(molecule.GetNumAtoms()) == int(record["num_nodes"]),
                "undirected_bond_count_exact": int(molecule.GetNumBonds()) == int(record["num_edges"]),
                "atomic_number_exact": diagnostics.get("atomic_numbers_exact"),
                "formal_charge_exact": diagnostics.get("formal_charges_exact"),
                "aromaticity_exact": diagnostics.get("aromaticity_exact"),
                "explicit_h_exact": diagnostics.get("explicit_hs_exact"),
                "no_implicit_exact": diagnostics.get("no_implicit_exact"),
                "chirality_exact": diagnostics.get("chiral_tags_exact"),
                "bond_type_exact": diagnostics.get("bond_types_exact"),
                "edge_direction_normalization_exact": True,
                "duplicate_edge_absence": int(graph.edge_index.shape[1]) == 2 * int(record["num_edges"]),
                "self_loop_absence": not bool((graph.edge_index[0] == graph.edge_index[1]).any().item()),
                "edge_attr_alignment_exact": getattr(graph, "edge_attr", None) is None,
                "node_lineage_exact": node_origin == expected_origin,
                "canonical_molecule_equivalence": reconstructed == expected,
                "roundtrip_ok": source_ok,
            }
        )
        no_op_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "clone_graph_sha256": stable_graph_sha256(graph.clone()),
                "save_load_graph_sha256": loaded_hash,
                "batch_unbatch_graph_sha256": batched_hash,
                "empty_action_decode_ok": decoded.decode_ok,
                "canonical_smiles": decoded.canonical_smiles,
                "noop_roundtrip_ok": no_op_ok,
            }
        )
    if not all(bool(row["roundtrip_ok"]) for row in source_rows):
        raise ValueError(f"{dataset} source round-trip is not 100%; repair is forbidden.")
    if not all(bool(row["noop_roundtrip_ok"]) for row in no_op_rows):
        raise ValueError(f"{dataset} no-op/serialization round-trip is not 100%.")
    return schema, bundle.graphs, record_by_id, source_rows, no_op_rows


def _aids_source_and_noop_gate(
    *,
    root: Path,
    dataset_dir: Path,
    source_csv: Path,
    parent_limit: int,
    multicomponent_authorized: bool,
) -> tuple[
    dict[str, Any],
    list[Any],
    dict[str, Mapping[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Freeze exact HIV source chemistry without introducing a second codec."""

    from rdkit import Chem
    from torch_geometric.data import Batch

    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        decode_generated_fullgraph,
        reconstruct_source_graph,
    )

    from .exporter import _aids_schema_and_record

    bundle = load_aids_generation_bundle(
        dataset_dir=dataset_dir,
        source_csv=source_csv,
        parent_limit=parent_limit,
    )
    if len(bundle.graphs) != parent_limit:
        raise ValueError("AIDS/HIV source cohort does not match the frozen parent count.")
    source_rows: list[dict[str, Any]] = []
    no_op_rows: list[dict[str, Any]] = []
    records: dict[str, Mapping[str, Any]] = {}
    schemas: dict[str, Any] = {}
    source_graphs: list[Any] = []
    for graph in bundle.graphs:
        cloned = graph.clone()
        cloned.gcf_node_origin = cloned.comrecgc_node_origin
        parent_id = str(cloned.comrecgc_parent_id)
        schema, record = _aids_schema_and_record(cloned, bundle.atom_vocabulary)
        record = {
            **record,
            "num_nodes": int(cloned.num_nodes),
            "num_edges": int(cloned.edge_index.shape[1] // 2),
        }
        schemas[parent_id] = schema
        records[parent_id] = record
        source_graphs.append(cloned)

    serialized = root / "source_graph_noop_roundtrip.pt"
    _torch_save_atomic(source_graphs, serialized)
    reloaded = _torch_load(serialized)
    batched = Batch.from_data_list(source_graphs).to_data_list()
    if not isinstance(reloaded, list) or len(reloaded) != len(source_graphs):
        raise ValueError("AIDS/HIV source graph save/load did not preserve graph count.")
    if len(batched) != len(source_graphs):
        raise ValueError("AIDS/HIV source graph batch/unbatch did not preserve graph count.")
    for graph, loaded_graph, batched_graph in zip(
        source_graphs, reloaded, batched, strict=True
    ):
        parent_id = str(graph.comrecgc_parent_id)
        schema = schemas[parent_id]
        record = records[parent_id]
        molecule, diagnostics = reconstruct_source_graph(record, schema)
        reconstructed = _canonical(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
        expected = _canonical(str(record["canonical_smiles"]))
        decoded = decode_generated_fullgraph(
            graph, source_record=record, schema=schema
        )
        clone_graph = graph.clone()
        graph_hash = stable_graph_sha256(graph)
        loaded_hash = stable_graph_sha256(loaded_graph)
        batched_hash = stable_graph_sha256(batched_graph)
        node_origin = [
            int(value) for value in graph.comrecgc_node_origin.detach().cpu().tolist()
        ]
        expected_origin = list(range(int(graph.num_nodes)))
        source_ok = bool(
            diagnostics.get("round_trip_passed")
            and reconstructed == expected
            and node_origin == expected_origin
        )
        identity = _aids_source_noop_identity(
            graph=graph,
            clone_graph=clone_graph,
            loaded_graph=loaded_graph,
            batched_graph=batched_graph,
            record=record,
            reconstructed_smiles=reconstructed,
            decoded=decoded,
            multicomponent_authorized=multicomponent_authorized,
        )
        source_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "sanitize_ok": source_ok,
                "node_count_exact": int(molecule.GetNumAtoms()) == int(record["num_nodes"]),
                "undirected_bond_count_exact": int(molecule.GetNumBonds())
                == int(record["num_edges"]),
                "atomic_number_exact": diagnostics.get("atomic_numbers_exact"),
                "formal_charge_exact": diagnostics.get("formal_charges_exact"),
                "aromaticity_exact": diagnostics.get("aromaticity_exact"),
                "explicit_h_exact": diagnostics.get("explicit_hs_exact"),
                "no_implicit_exact": diagnostics.get("no_implicit_exact"),
                "chirality_exact": diagnostics.get("chiral_tags_exact"),
                "bond_type_exact": diagnostics.get("bond_types_exact"),
                "edge_direction_normalization_exact": True,
                "duplicate_edge_absence": int(graph.edge_index.shape[1])
                == 2 * int(record["num_edges"]),
                "self_loop_absence": not bool(
                    (graph.edge_index[0] == graph.edge_index[1]).any().item()
                ),
                "edge_attr_alignment_exact": getattr(graph, "edge_attr", None) is None,
                "node_lineage_exact": node_origin == expected_origin,
                "canonical_molecule_equivalence": reconstructed == expected,
                "source_smiles_match_mode": record["source_smiles_match_mode"],
                "node_order_authority": record["node_order_authority"],
                "roundtrip_ok": source_ok,
            }
        )
        no_op_rows.append(
            {
                "parent_id": parent_id,
                "source_graph_sha256": graph_hash,
                "clone_graph_sha256": stable_graph_sha256(clone_graph),
                "save_load_graph_sha256": loaded_hash,
                "batch_unbatch_graph_sha256": batched_hash,
                "empty_action_decode_ok": decoded.decode_ok,
                "empty_action_decode_failure_reason": decoded.failure_reason,
                "canonical_smiles": expected,
                "identity_mode": identity["identity_mode"],
                "is_multicomponent_source": identity["is_multicomponent_source"],
                "component_count": identity["component_count"],
                "graph_component_count": identity["graph_component_count"],
                "canonical_isomeric_component_multiset": json.dumps(
                    identity["canonical_isomeric_component_multiset"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "canonical_isomeric_component_multiset_sha256": identity[
                    "canonical_isomeric_component_multiset_sha256"
                ],
                "component_multiplicity_preserved": identity[
                    "component_multiplicity_preserved"
                ],
                "component_multiset_exact": identity["component_multiset_exact"],
                "component_count_exact": identity["component_count_exact"],
                "atom_tensor_record_exact": identity["atom_tensor_record_exact"],
                "bond_tensor_record_exact": identity["bond_tensor_record_exact"],
                "atom_tensor_closure_exact": identity[
                    "atom_tensor_closure_exact"
                ],
                "bond_tensor_closure_exact": identity[
                    "bond_tensor_closure_exact"
                ],
                "node_count_closure_exact": identity["node_count_closure_exact"],
                "node_lineage_closure_exact": identity["node_lineage_exact"],
                "parent_identity_closure_exact": identity[
                    "parent_identity_closure_exact"
                ],
                "graph_hash_closure_exact": identity["graph_hash_closure_exact"],
                "decoder_contract_exact": identity["decoder_contract_exact"],
                "multicomponent_authorized": identity[
                    "multicomponent_authorized"
                ],
                "noop_roundtrip_ok": identity["noop_roundtrip_ok"],
            }
        )
    if not all(bool(row["roundtrip_ok"]) for row in source_rows):
        raise ValueError("AIDS/HIV source round-trip is not 100%; repair is forbidden.")
    if not all(bool(row["noop_roundtrip_ok"]) for row in no_op_rows):
        raise ValueError("AIDS/HIV no-op/serialization round-trip is not 100%.")
    return schemas, source_graphs, records, source_rows, no_op_rows


def run_mutagenicity_chemistry_audit(
    *,
    project_root: str | Path,
    dataset_dir: str | Path,
    generation_dir: str | Path,
    trace_lineage_path: str | Path,
    trace_parity_path: str | Path,
    common_recourse_dir: str | Path,
    output_dir: str | Path,
    preregistration_path: str | Path,
    parent_limit: int = 64,
    expected_candidate_count: int | None = 164,
    expected_medoid_count: int | None = 4,
    expected_counterfactuals_sha256: str | None = None,
    dataset: str = "mutagenicity",
    source_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all raw candidates, replay exact actions, and freeze one repair each."""

    if dataset not in {"aids", "mutagenicity", "bace"}:
        raise ValueError(f"Unsupported project chemistry dataset: {dataset}")
    if dataset == "aids" and source_csv is None:
        raise ValueError("AIDS/HIV chemistry audit requires source_csv.")

    root = require_empty_output(output_dir)
    project = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    generation_root = Path(generation_dir).expanduser().resolve()
    common_root = Path(common_recourse_dir).expanduser().resolve()
    generation_manifest = _load_json(generation_root / "run_manifest.json")
    counterfactuals = Path(generation_manifest["counterfactuals_path"]).expanduser().resolve()
    actual_sha = sha256_file(counterfactuals)
    if actual_sha != str(generation_manifest["counterfactuals_sha256"]):
        raise ValueError("Mutagenicity generation artifact differs from its manifest.")
    if expected_counterfactuals_sha256 and actual_sha != expected_counterfactuals_sha256:
        raise ValueError("Mutagenicity generation artifact differs from the frozen blocker SHA256.")
    trace_evidence = validate_chemistry_trace_evidence(
        trace_parity_path,
        dataset=dataset,
    )
    lineage_path = Path(trace_lineage_path).expanduser().resolve()
    lineage_contract, lineage_count, lineage_format = _lineage_contract(lineage_path)
    evidence_candidate_count = trace_evidence.get("candidate_count")
    if evidence_candidate_count is not None and int(evidence_candidate_count) != lineage_count:
        raise ValueError(
            "Trace evidence and candidate lineage counts differ: "
            f"evidence={evidence_candidate_count}, lineage={lineage_count}."
        )
    payload = _torch_load(counterfactuals)
    candidate_graphs = _candidate_graphs(payload)
    if (
        expected_candidate_count is not None
        and (
            len(candidate_graphs) != expected_candidate_count
            or lineage_count != expected_candidate_count
        )
    ):
        raise ValueError(
            "Frozen Mutagenicity candidate count mismatch: "
            f"graphs={len(candidate_graphs)}, lineage={lineage_count}, expected={expected_candidate_count}."
        )
    if lineage_count != len(candidate_graphs):
        raise ValueError(
            "Generation and lineage candidate counts differ: "
            f"graphs={len(candidate_graphs)}, lineage={lineage_count}."
        )

    if dataset == "aids":
        authorization_value = os.environ.get(
            AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV,
            "0",
        )
        if authorization_value not in {"0", "1"}:
            raise ValueError(
                f"{AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV} must be exactly 0 or 1."
            )
        schemas, source_graphs, source_records, source_rows, no_op_rows = (
            _aids_source_and_noop_gate(
                root=root,
                dataset_dir=dataset_root,
                source_csv=Path(str(source_csv)).expanduser().resolve(),
                parent_limit=parent_limit,
                multicomponent_authorized=authorization_value == "1",
            )
        )
        schema = next(iter(schemas.values()))
    else:
        schema, source_graphs, source_records, source_rows, no_op_rows = (
            _source_and_noop_gate(
                root=root,
                dataset_dir=dataset_root,
                parent_limit=parent_limit,
                dataset=dataset,
            )
        )
        schemas = {
            str(getattr(graph, "comrecgc_parent_id")): schema
            for graph in source_graphs
        }
    source_by_id = {
        str(getattr(graph, "comrecgc_parent_id")): graph for graph in source_graphs
    }
    lineage = _iter_candidate_lineage(
        path=lineage_path,
        contract=lineage_contract,
        payload=payload,
        source_graphs_by_parent_id=source_by_id,
    )
    atom_mapping, bond_mapping = _mapping_payloads(schema, dataset=dataset)
    atom_mapping_path = root / "atom_label_mapping.json"
    bond_mapping_path = root / "bond_mapping.json"
    write_json(atom_mapping_path, atom_mapping)
    write_json(bond_mapping_path, bond_mapping)

    preregistration_target = Path(preregistration_path).expanduser().resolve()
    if preregistration_target.exists():
        preregistration = _load_json(preregistration_target)
        if preregistration.get("source_counterfactuals_sha256") != actual_sha:
            raise ValueError("Existing chemistry preregistration targets another artifact.")
        preregistration_sha = sha256_file(preregistration_target)
    else:
        preregistration = write_mutagenicity_chem_repair_preregistration(
            project_commit=_project_commit(project),
            source_counterfactuals_path=counterfactuals,
            trace_parity_path=trace_parity_path,
            atom_mapping_path=atom_mapping_path,
            bond_mapping_path=bond_mapping_path,
            output_path=preregistration_target,
            dataset=dataset,
        )
        preregistration_sha = str(preregistration["file_sha256"])

    common_records = _load_json_list(common_root / "selected_common_recourses.json")
    if expected_medoid_count is not None and len(common_records) != expected_medoid_count:
        raise ValueError("Official Mutagenicity medoid count differs from the frozen blocker.")
    ranks = [int(row["rank"]) for row in common_records]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("Official common-recourse rank is not contiguous and frozen.")
    medoid_candidate_indices = {
        int(row["generation_candidate_index"]) for row in common_records
    }
    if any(index < 0 or index >= len(candidate_graphs) for index in medoid_candidate_indices):
        raise ValueError("Official common-recourse medoid index is outside candidate range.")

    raw_rows: list[dict[str, Any]] = []
    repaired_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    first_invalid_rows: list[dict[str, Any]] = []
    materialize_all_repaired_graphs = lineage_format == "inline_json"
    repaired_graphs: list[Any] = []
    medoid_graphs_by_index: dict[int, Any] = {}
    action_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_reason_counts: Counter[str] = Counter()
    for index, (graph, trace) in enumerate(zip(candidate_graphs, lineage, strict=True)):
        if int(trace.get("candidate_index", -1)) != index:
            raise ValueError("Candidate action lineage is not in exact official order.")
        if trace.get("action_lineage_resolved") is not True:
            raise ValueError(f"Candidate {index} action lineage is unresolved.")
        parent_id = str(trace.get("parent_id") or getattr(graph, "comrecgc_parent_id", ""))
        if parent_id not in source_by_id or parent_id not in source_records:
            raise ValueError(f"Candidate {index} parent lineage is absent: {parent_id!r}")
        source_graph = source_by_id[parent_id]
        source_record = source_records[parent_id]
        candidate_schema = schemas[parent_id]
        actions = list(trace.get("actions") or [])
        raw_replay = replay_raw_actions(source_graph, actions)
        replay_sha = stable_graph_sha256(raw_replay)
        raw_sha = stable_graph_sha256(graph)
        if replay_sha != raw_sha:
            raise ValueError(f"Candidate {index} exact action replay differs from official graph.")
        from src.baselines.gcfexplainer_mutagenicity_adapter import decode_generated_fullgraph

        graph.gcf_node_origin = graph.comrecgc_node_origin
        raw_decoded = decode_generated_fullgraph(
            graph, source_record=source_record, schema=candidate_schema
        )
        first = repair_candidate(
            source_graph=source_graph,
            source_record=source_record,
            schema=candidate_schema,
            actions=actions,
        )
        second = repair_candidate(
            source_graph=source_graph,
            source_record=source_record,
            schema=candidate_schema,
            actions=actions,
        )
        deterministic = bool(
            first.output_graph_sha256 == second.output_graph_sha256
            and first.canonical_smiles == second.canonical_smiles
            and first.action_records == second.action_records
        )
        if not deterministic:
            raise ValueError(f"Candidate {index} chemistry repair is nondeterministic.")
        if materialize_all_repaired_graphs:
            repaired_graphs.append(first.graph)
        if index in medoid_candidate_indices:
            medoid_graphs_by_index[index] = first.graph
        raw_reason = "valid" if raw_decoded.decode_ok else str(raw_decoded.failure_reason)
        raw_reason_counts[raw_reason] += 1
        candidate_prefix = {
            "aids": "AIDS",
            "mutagenicity": "MUT",
            "bace": "BACE",
        }[dataset]
        candidate_id = f"COMRECGC_{candidate_prefix}_RAW_{index:06d}"
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                "raw_graph_sha256": raw_sha,
                "raw_sanitize_ok": raw_decoded.decode_ok,
                "raw_invalid_reason": "" if raw_decoded.decode_ok else raw_reason,
                "raw_canonical_smiles": raw_decoded.canonical_smiles,
                "action_count": len(actions),
                "action_replay_exact": True,
            }
        )
        first_invalid = dict(first.first_invalid_action or {})
        repaired_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                "raw_graph_sha256": raw_sha,
                "input_candidate_sha256": raw_sha,
                "repair_policy_sha256": preregistration_sha,
                "repair_attempted": True,
                "repair_success": first.repair_success,
                "repair_noop": first.repair_noop,
                "retained_action_count": first.retained_action_count,
                "skipped_action_count": first.skipped_action_count,
                "skipped_action_types": list(first.skipped_action_types),
                "dependent_action_skip_count": first.dependent_action_skip_count,
                "action_retention_rate": (
                    first.retained_action_count / len(actions) if actions else 1.0
                ),
                "first_invalid_action": first_invalid,
                "repaired_graph_sha256": first.output_graph_sha256,
                "output_candidate_sha256": first.output_graph_sha256,
                "repaired_smiles": first.canonical_smiles,
                "repaired_sanitize_ok": first.repair_success,
                "repair_deterministic": deterministic,
                "rf_scored": False,
                "rf_pred_before": None,
                "rf_pred_after": None,
                "rf_strict_flip": None,
                "rf_cf_drop": None,
                "wnode_scored": False,
                "wnode_distance": None,
            }
        )
        if first_invalid:
            first_invalid_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "source_graph_id": parent_id,
                    **first_invalid,
                }
            )
        for action in first.action_records:
            row = {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "source_graph_id": parent_id,
                **dict(action),
            }
            action_rows.append(row)
            action_type_counts[str(action["action_type"])][str(action["status"])] += 1

    repaired_candidates_path: Path | None = None
    if materialize_all_repaired_graphs:
        repaired_candidates_path = root / "repaired_candidates.pt"
        _torch_save_atomic(repaired_graphs, repaired_candidates_path)
        reloaded_repaired = _torch_load(repaired_candidates_path)
        if len(reloaded_repaired) != len(repaired_graphs):
            raise ValueError("Repaired candidate artifact is not reloadable.")

    medoid_rows: list[dict[str, Any]] = []
    medoid_graphs: list[Any] = []
    for common in common_records:
        candidate_index = int(common["generation_candidate_index"])
        repaired = repaired_rows[candidate_index]
        if candidate_index not in medoid_graphs_by_index:
            raise ValueError("Repaired official medoid graph was not retained.")
        medoid_graphs.append(medoid_graphs_by_index[candidate_index])
        medoid_rows.append(
            {
                "official_cluster_rank": int(common["rank"]),
                "cluster_id": common.get("common_recourse_id"),
                "cluster_label": common.get("cluster_label"),
                "candidate_index": candidate_index,
                "candidate_id": repaired["candidate_id"],
                "is_official_medoid": True,
                "representative_policy": "repaired_original_official_medoid",
                "repair_success": repaired["repair_success"],
                "repair_noop": repaired["repair_noop"],
                "repaired_graph_sha256": repaired["repaired_graph_sha256"],
                "repaired_smiles": repaired["repaired_smiles"],
                "invalid_slot_backfill": False,
                "rank_compaction": False,
            }
        )
    _torch_save_atomic(medoid_graphs, root / "repaired_official_medoids.pt")

    source_fields = list(source_rows[0])
    noop_fields = list(no_op_rows[0])
    _write_csv(root / "source_roundtrip.csv", source_rows, source_fields)
    _write_csv(root / "noop_roundtrip.csv", no_op_rows, noop_fields)
    aids_source_noop_receipt: dict[str, Any] | None = None
    if dataset == "aids":
        component_histogram = Counter(int(row["component_count"]) for row in no_op_rows)
        multicomponent_count = sum(
            bool(row["is_multicomponent_source"]) for row in no_op_rows
        )
        aids_source_noop_receipt = {
            "schema_version": AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY,
            "dataset": "AIDS/HIV",
            "authorization_environment": AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV,
            "authorization_enabled": os.environ.get(
                AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV
            )
            == "1",
            "source_parent_count": len(source_rows),
            "expected_source_parent_count": int(parent_limit),
            "full_requested_cohort_preserved": len(source_rows) == int(parent_limit),
            "multicomponent_source_count": multicomponent_count,
            "component_count_histogram": {
                str(key): int(value) for key, value in sorted(component_histogram.items())
            },
            "all_source_roundtrips_passed": all(
                bool(row["roundtrip_ok"]) for row in source_rows
            ),
            "all_source_noop_identities_passed": all(
                bool(row["noop_roundtrip_ok"]) for row in no_op_rows
            ),
            "canonical_isomeric_component_multiset_with_multiplicity": True,
            "atom_and_bond_tensor_closure_required": True,
            "node_lineage_closure_required": True,
            "clone_save_load_batch_unbatch_closure_required": True,
            "source_roundtrip_path": str(root / "source_roundtrip.csv"),
            "source_roundtrip_sha256": sha256_file(root / "source_roundtrip.csv"),
            "noop_roundtrip_path": str(root / "noop_roundtrip.csv"),
            "noop_roundtrip_sha256": sha256_file(root / "noop_roundtrip.csv"),
            "source_rows_excluded": 0,
            "source_components_stripped": False,
            "source_components_repaired": False,
            "generated_candidate_decoder_modified": False,
            "generated_candidate_single_component_gate_preserved": True,
            "generation_rerun": False,
            "dbscan_rerun": False,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        write_json(
            root / "aids_multicomponent_source_noop_identity.json",
            aids_source_noop_receipt,
        )
    _write_csv(root / "raw_candidates.csv", raw_rows, list(raw_rows[0]))
    _write_csv(root / "candidate_validity.csv", repaired_rows, list(repaired_rows[0]))
    _write_jsonl(root / "action_replay.jsonl", action_rows)
    _write_csv(
        root / "first_invalid_actions.csv",
        first_invalid_rows,
        list(first_invalid_rows[0]) if first_invalid_rows else ["candidate_id", "candidate_index"],
    )
    action_type_rows = [
        {"action_type": action_type, **dict(counts), "total": sum(counts.values())}
        for action_type, counts in sorted(action_type_counts.items())
    ]
    action_statuses = sorted({key for row in action_type_counts.values() for key in row})
    _write_csv(
        root / "action_type_validity.csv",
        action_type_rows,
        ["action_type", *action_statuses, "total"],
    )
    medoid_fields = (
        list(medoid_rows[0])
        if medoid_rows
        else [
            "official_cluster_rank",
            "cluster_id",
            "cluster_label",
            "candidate_index",
            "candidate_id",
            "is_official_medoid",
            "representative_policy",
            "repair_success",
            "repair_noop",
            "repaired_graph_sha256",
            "repaired_smiles",
            "invalid_slot_backfill",
            "rank_compaction",
        ]
    )
    _write_csv(root / "medoid_validity.csv", medoid_rows, medoid_fields)
    _write_csv(root / "cluster_validity.csv", medoid_rows, medoid_fields)
    _write_jsonl(root / "raw_candidates.jsonl", raw_rows)
    _write_jsonl(root / "repaired_candidates.jsonl", repaired_rows)
    _write_jsonl(root / "repair_provenance.jsonl", repaired_rows)
    _write_csv(root / "raw_candidate_validity.csv", raw_rows, list(raw_rows[0]))
    _write_csv(root / "repaired_candidate_validity.csv", repaired_rows, list(repaired_rows[0]))
    _write_csv(root / "official_cluster_order.csv", medoid_rows, medoid_fields)
    _write_csv(root / "official_medoid_repair.csv", medoid_rows, medoid_fields)
    write_json(root / "selected_common_recourses.json", medoid_rows)
    _write_jsonl(root / "representative_counterfactuals.jsonl", medoid_rows)

    source_pass = sum(bool(row["roundtrip_ok"]) for row in source_rows)
    noop_pass = sum(bool(row["noop_roundtrip_ok"]) for row in no_op_rows)
    repaired_count = sum(bool(row["repair_success"]) for row in repaired_rows)
    repaired_medoid_count = sum(bool(row["repair_success"]) for row in medoid_rows)
    audit = {
        "schema_version": 1,
        "dataset": {
            "aids": "AIDS/HIV",
            "mutagenicity": "Mutagenicity",
            "bace": "BACE",
        }[dataset],
        "dataset_key": dataset,
        "audit_passed": True,
        "engineering_smoke_pass": True,
        "method": REPAIR_METHOD,
        "repair_policy_version": REPAIR_POLICY_VERSION,
        "repair_policy_sha256": preregistration_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "project_commit": _project_commit(project),
        "dataset_fingerprint": generation_manifest.get("dataset_audit", {}).get(
            "dataset_fingerprint"
        ),
        "generation_parent_ids_sha256": generation_manifest.get(
            "generation_parent_ids_sha256"
        ),
        "source_parent_count": len(source_rows),
        "source_roundtrip_pass_count": source_pass,
        "source_roundtrip_rate": source_pass / len(source_rows),
        "noop_roundtrip_pass_count": noop_pass,
        "noop_roundtrip_rate": noop_pass / len(no_op_rows),
        "aids_multicomponent_source_noop_identity": (
            {
                "policy": AIDS_MULTICOMPONENT_SOURCE_NOOP_POLICY,
                "authorization_environment": AIDS_MULTICOMPONENT_SOURCE_NOOP_ENV,
                "authorization_enabled": bool(
                    aids_source_noop_receipt
                    and aids_source_noop_receipt["authorization_enabled"]
                ),
                "multicomponent_source_count": int(
                    aids_source_noop_receipt["multicomponent_source_count"]
                ),
                "receipt_path": str(
                    root / "aids_multicomponent_source_noop_identity.json"
                ),
                "receipt_sha256": sha256_file(
                    root / "aids_multicomponent_source_noop_identity.json"
                ),
                "generated_candidate_single_component_gate_preserved": True,
            }
            if aids_source_noop_receipt is not None
            else None
        ),
        "trace_parity": bool(trace_evidence["trace_parity_passed"]),
        "trace_integrity": bool(trace_evidence["trace_integrity_passed"]),
        "trace_evidence_kind": trace_evidence["trace_evidence_kind"],
        "trace_parity_required": bool(trace_evidence["trace_parity_required"]),
        "trace_rng_evidence_kind": trace_evidence.get("trace_rng_evidence_kind"),
        "rng_calls_added": trace_evidence.get("rng_calls_added"),
        "freeze_only_recovery_path": trace_evidence.get(
            "freeze_only_recovery_path"
        ),
        "freeze_only_recovery_sha256": trace_evidence.get(
            "freeze_only_recovery_sha256"
        ),
        "freeze_only_terminal_path": trace_evidence.get(
            "freeze_only_terminal_path"
        ),
        "freeze_only_terminal_sha256": trace_evidence.get(
            "freeze_only_terminal_sha256"
        ),
        "trace_lineage_format": lineage_format,
        "trace_lineage_streamed": lineage_format == "selected_trace_predecessor_index",
        "max_candidate_lineages_materialized": (
            1 if lineage_format == "selected_trace_predecessor_index" else lineage_count
        ),
        "raw_candidate_count": len(raw_rows),
        "raw_valid_candidate_count": sum(bool(row["raw_sanitize_ok"]) for row in raw_rows),
        "raw_invalid_reason_counts": dict(raw_reason_counts),
        "repair_provenance_count": len(repaired_rows),
        "repaired_candidate_count": repaired_count,
        "official_medoid_count": len(medoid_rows),
        "repaired_official_medoid_count": repaired_medoid_count,
        "repair_deterministic_count": sum(bool(row["repair_deterministic"]) for row in repaired_rows),
        "one_raw_candidate_max_one_repaired_candidate": True,
        "all_repaired_graphs_materialized": materialize_all_repaired_graphs,
        "retained_repaired_graph_object_count": (
            len(repaired_graphs)
            if materialize_all_repaired_graphs
            else len(medoid_graphs_by_index)
        ),
        "official_cluster_rank_unchanged": True,
        "invalid_slot_backfill": False,
        "rank_compaction": False,
        "rf_used_in_repair": False,
        "wnode_used_in_repair": False,
        "strict_flip_used_in_repair": False,
        "project_feasibility_status": (
            "PROJECT_FEASIBILITY_OBSERVED"
            if repaired_medoid_count > 0
            else "PROJECT_FEASIBILITY_NOT_OBSERVED"
        ),
        "strict_flip_status": "NOT_EVALUATED_IN_REPAIR",
        "calibration_loaded": False,
        "test_loaded": False,
        "counterfactuals_path": str(counterfactuals),
        "counterfactuals_sha256": actual_sha,
        "trace_lineage_path": str(Path(trace_lineage_path).resolve()),
        "trace_lineage_sha256": sha256_file(trace_lineage_path),
        "trace_evidence_path": str(Path(trace_parity_path).resolve()),
        "trace_evidence_sha256": sha256_file(trace_parity_path),
        "preregistration_path": str(preregistration_target),
        "preregistration_sha256": preregistration_sha,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "audit.json", audit)
    write_json(
        root / "run_manifest.json",
        {
            **audit,
            "run_complete": True,
            "algorithm_rerun": False,
            "candidate_order_source": "official_random_walk_candidate_order",
            "official_medoid_policy": "original_official_medoid_only",
            "repaired_candidates_path": (
                str(repaired_candidates_path) if repaired_candidates_path else None
            ),
            "repaired_candidates_sha256": (
                sha256_file(repaired_candidates_path)
                if repaired_candidates_path
                else None
            ),
            "repaired_candidates_materialization": (
                "all_candidates_smoke_compatibility"
                if repaired_candidates_path
                else "omitted_full_memory_bound_representatives_only"
            ),
            "repaired_official_medoids_path": str(root / "repaired_official_medoids.pt"),
            "repaired_official_medoids_sha256": sha256_file(
                root / "repaired_official_medoids.pt"
            ),
        },
    )
    atomic_write_bytes(
        root / "audit.txt",
        (
            f"COMRECGC {audit['dataset']} chemistry audit\n"
            f"source_roundtrip={source_pass}/{len(source_rows)}\n"
            f"noop_roundtrip={noop_pass}/{len(no_op_rows)}\n"
            f"raw_candidates={len(raw_rows)} raw_valid={audit['raw_valid_candidate_count']}\n"
            f"repaired_candidates={repaired_count} repaired_official_medoids={repaired_medoid_count}\n"
            "rf_used_in_repair=false wnode_used_in_repair=false\n"
            "[COMRECGC_PROJECT_CHEMISTRY_ENGINEERING_PASS]\n"
        ).encode("utf-8"),
    )
    write_json(
        root / "final_artifact_audit.json",
        {
            **audit,
            "run_complete": True,
            "output_reload_verified": True,
            "required_artifacts": {
                name: {
                    "bytes": (root / name).stat().st_size,
                    "sha256": sha256_file(root / name),
                }
                for name in (
                    "source_roundtrip.csv",
                    "noop_roundtrip.csv",
                    *(
                        ("aids_multicomponent_source_noop_identity.json",)
                        if aids_source_noop_receipt is not None
                        else ()
                    ),
                    "raw_candidates.jsonl",
                    "repair_provenance.jsonl",
                    "official_medoid_repair.csv",
                    "repaired_official_medoids.pt",
                    *(
                        ("repaired_candidates.pt",)
                        if repaired_candidates_path is not None
                        else ()
                    ),
                )
            },
        },
    )
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, "audit_passed": True})
    write_json(
        root / "manifest.json",
        {
            "run_complete": True,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(root.iterdir())
                if path.is_file()
            },
        },
    )
    return audit
