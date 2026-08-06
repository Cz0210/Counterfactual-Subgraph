"""Deterministic, one-output chemistry projection for COMRECGC graph actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .graph_trace import stable_graph_sha256, trace_node_ids

REPAIR_METHOD = "COMRECGC-Adapted-DeterministicChemRepair"
REPAIR_POLICY_VERSION = 1


@dataclass(frozen=True)
class ChemRepairResult:
    graph: Any
    repair_success: bool
    repair_noop: bool
    retained_action_count: int
    skipped_action_count: int
    dependent_action_skip_count: int
    skipped_action_types: tuple[str, ...]
    first_invalid_action: Mapping[str, Any] | None
    canonical_smiles: str
    output_graph_sha256: str
    action_records: tuple[Mapping[str, Any], ...]


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("COMRECGC chemistry repair requires PyTorch.") from exc
    return torch


def _values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _origins(graph: Any) -> list[int]:
    return [int(value) for value in _values(getattr(graph, "comrecgc_node_origin"))]


def _set_lineage(graph: Any, origins: Sequence[int], node_ids: Sequence[str]) -> None:
    torch = _torch()
    graph.comrecgc_node_origin = torch.tensor(
        [int(value) for value in origins],
        dtype=torch.long,
        device=graph.x.device,
    )
    graph.gcf_node_origin = graph.comrecgc_node_origin
    graph.comrecgc_trace_node_ids = [str(value) for value in node_ids]


def _action_node_indices(action: Sequence[Any]) -> list[int]:
    name = str(action[0])
    if name in {"NLC", "NA", "NR", "INR"}:
        return [int(action[1])]
    if name in {"ER", "ERR", "EA"}:
        return [int(action[1]), int(action[2])]
    return []


def _atom_state_snapshot(
    graph: Any,
    *,
    source_record: Mapping[str, Any],
    schema: Any,
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Describe affected atom state without sanitizing or changing the graph."""

    x = _values(graph.x)
    origins = _origins(graph)
    source_atoms = list(source_record["atom_sidecar"])
    source_bonds = {
        (int(row["begin"]), int(row["end"])): str(row["bond_type"])
        for row in source_record["bond_sidecar"]
    }
    bond_orders = {"SINGLE": 1.0, "DOUBLE": 2.0, "TRIPLE": 3.0, "AROMATIC": 1.5}
    edges = {
        tuple(sorted((int(first), int(second))))
        for first, second in zip(
            _values(graph.edge_index[0]),
            _values(graph.edge_index[1]),
            strict=True,
        )
        if int(first) != int(second)
    }
    rows: list[dict[str, Any]] = []
    for index in sorted(set(int(value) for value in indices)):
        if not 0 <= index < int(graph.num_nodes):
            continue
        feature = [float(value) for value in x[index]]
        label = max(range(len(feature)), key=feature.__getitem__)
        atomic_number = int(schema.feature_atomic_numbers[label])
        origin = int(origins[index])
        source_atom = source_atoms[origin] if 0 <= origin < len(source_atoms) else None
        identity_unchanged = bool(
            source_atom is not None and atomic_number == int(source_atom["atomic_num"])
        )
        formal_charge = int(source_atom["formal_charge"]) if identity_unchanged else 0
        bond_order_sum = 0.0
        for first, second in edges:
            if index not in {first, second}:
                continue
            other = second if first == index else first
            other_origin = int(origins[other])
            key = tuple(sorted((origin, other_origin)))
            bond_order_sum += bond_orders.get(source_bonds.get(key, "SINGLE"), 1.0)
        rows.append(
            {
                "node_index": index,
                "stable_node_id": trace_node_ids(graph)[index],
                "origin": origin,
                "atomic_number": atomic_number,
                "formal_charge": formal_charge,
                "bond_order_sum": bond_order_sum,
            }
        )
    return rows


def apply_action_to_graph(
    graph: Any,
    action: Sequence[Any],
    *,
    target_node_ids: Sequence[str] | None = None,
) -> Any:
    """Mirror the pinned upstream edit operation for the untyped graph space."""

    torch = _torch()
    candidate = graph.clone()
    if getattr(candidate, "edge_attr", None) is not None:
        raise ValueError("COMRECGC Mutagenicity repair expects the frozen untyped-edge graph space.")
    name = str(action[0])
    origins = _origins(graph)
    node_ids = trace_node_ids(graph)
    if name == "NOTHING":
        pass
    elif name == "NLC":
        node, label = int(action[1]), int(action[2])
        if not (0 <= node < int(graph.num_nodes) and 0 <= label < int(graph.x.shape[1])):
            raise IndexError("NLC action index is outside the current graph.")
        candidate.x[node] = 0
        candidate.x[node][label] = 1
    elif name in {"NA", "INA"}:
        node, label = int(action[1]), int(action[2])
        if not (0 <= label < int(graph.x.shape[1])):
            raise IndexError("Node-addition label is outside the frozen atom mapping.")
        if name == "NA" and not 0 <= node < int(graph.num_nodes):
            raise IndexError("Node-addition attachment index is outside the current graph.")
        new_feature = torch.nn.functional.one_hot(
            torch.tensor(label, device=graph.x.device),
            int(graph.x.shape[1]),
        ).to(dtype=graph.x.dtype)
        candidate.x = torch.vstack([graph.x, new_feature])
        if name == "NA":
            new_index = int(graph.num_nodes)
            added = torch.tensor(
                [[node, new_index], [new_index, node]],
                dtype=graph.edge_index.dtype,
                device=graph.edge_index.device,
            )
            candidate.edge_index = torch.hstack([graph.edge_index, added])
        candidate.num_nodes = int(graph.num_nodes) + 1
        origins.append(-1)
        if target_node_ids is None or len(target_node_ids) != int(candidate.num_nodes):
            raise ValueError("Node addition requires exact traced target node IDs.")
        node_ids = [str(value) for value in target_node_ids]
    elif name in {"NR", "INR"}:
        node = int(action[1])
        if not 0 <= node < int(graph.num_nodes):
            raise IndexError("Node-removal index is outside the current graph.")
        keep_nodes = [index for index in range(int(graph.num_nodes)) if index != node]
        candidate.x = graph.x[torch.tensor(keep_nodes, dtype=torch.long, device=graph.x.device)]
        keep_edges = (graph.edge_index[0] != node) & (graph.edge_index[1] != node)
        edges = graph.edge_index[:, keep_edges].clone()
        edges[edges > node] -= 1
        candidate.edge_index = edges
        candidate.num_nodes = int(graph.num_nodes) - 1
        origins.pop(node)
        node_ids.pop(node)
    elif name in {"ER", "ERR"}:
        first, second = int(action[1]), int(action[2])
        keep = ~(
            ((graph.edge_index[0] == first) & (graph.edge_index[1] == second))
            | ((graph.edge_index[0] == second) & (graph.edge_index[1] == first))
        )
        candidate.edge_index = graph.edge_index[:, keep]
    elif name == "EA":
        first, second = int(action[1]), int(action[2])
        if not (0 <= first < int(graph.num_nodes) and 0 <= second < int(graph.num_nodes)):
            raise IndexError("Edge-addition endpoint is outside the current graph.")
        added = torch.tensor(
            [[first, second], [second, first]],
            dtype=graph.edge_index.dtype,
            device=graph.edge_index.device,
        )
        candidate.edge_index = torch.hstack([graph.edge_index, added])
    else:
        raise ValueError(f"Unsupported pinned COMRECGC action: {name}")
    if target_node_ids is not None and name not in {"NA", "INA"}:
        if [str(value) for value in target_node_ids] != node_ids:
            raise ValueError("Traced target node IDs disagree with deterministic action semantics.")
    _set_lineage(candidate, origins, node_ids)
    return candidate


def _map_action_to_repaired_graph(
    action: Sequence[Any],
    *,
    raw_source_node_ids: Sequence[str],
    raw_target_node_ids: Sequence[str],
    repaired_node_ids: Sequence[str],
) -> tuple[list[Any] | None, list[str] | None]:
    name = str(action[0])
    values = list(action)
    node_positions = {value: index for index, value in enumerate(repaired_node_ids)}

    def mapped(raw_index: int) -> int | None:
        if not 0 <= raw_index < len(raw_source_node_ids):
            return None
        return node_positions.get(str(raw_source_node_ids[raw_index]))

    if name in {"NLC", "NA", "NR", "INR"}:
        resolved = mapped(int(values[1]))
        if resolved is None:
            return None, None
        values[1] = resolved
    elif name in {"ER", "ERR", "EA"}:
        first, second = mapped(int(values[1])), mapped(int(values[2]))
        if first is None or second is None:
            return None, None
        values[1], values[2] = first, second
    elif name not in {"NOTHING", "INA"}:
        raise ValueError(f"Unsupported action in trace: {name}")
    target_ids = list(repaired_node_ids)
    if name in {"NA", "INA"}:
        added_ids = [value for value in raw_target_node_ids if value not in raw_source_node_ids]
        if len(added_ids) != 1:
            raise ValueError("Traced node addition does not introduce exactly one stable node ID.")
        target_ids.append(str(added_ids[0]))
    elif name in {"NR", "INR"}:
        target_ids.pop(int(values[1]))
    return values, target_ids


def replay_raw_actions(source_graph: Any, actions: Sequence[Mapping[str, Any]]) -> Any:
    current = source_graph.clone()
    if not hasattr(current, "comrecgc_trace_node_ids"):
        current.comrecgc_trace_node_ids = trace_node_ids(current)
    for step, record in enumerate(actions):
        if record.get("action_resolution") != "exact" or record.get("action") is None:
            raise ValueError(f"Action trace is unresolved at step {step}.")
        expected_source = [str(value) for value in record["source_node_ids"]]
        if trace_node_ids(current) != expected_source:
            raise ValueError(f"Raw action lineage diverges at step {step}.")
        current = apply_action_to_graph(
            current,
            record["action"],
            target_node_ids=[str(value) for value in record["target_node_ids"]],
        )
    return current


def repair_candidate(
    *,
    source_graph: Any,
    source_record: Mapping[str, Any],
    schema: Any,
    actions: Sequence[Mapping[str, Any]],
) -> ChemRepairResult:
    """Replay one official path and rollback only actions that fail sanitize."""

    from src.baselines.gcfexplainer_mutagenicity_adapter import decode_generated_fullgraph

    current = source_graph.clone()
    if not hasattr(current, "comrecgc_trace_node_ids"):
        current.comrecgc_trace_node_ids = trace_node_ids(current)
    current.gcf_node_origin = current.comrecgc_node_origin
    initial = stable_graph_sha256(current)
    action_rows: list[dict[str, Any]] = []
    retained = 0
    skipped = 0
    dependent = 0
    skipped_types: list[str] = []
    first_invalid: dict[str, Any] | None = None
    for step, trace in enumerate(actions):
        if trace.get("action_resolution") != "exact" or trace.get("action") is None:
            raise ValueError(f"Action trace is unresolved at step {step}.")
        raw_action = list(trace["action"])
        action_name = str(raw_action[0])
        mapped_action, target_ids = _map_action_to_repaired_graph(
            raw_action,
            raw_source_node_ids=[str(value) for value in trace["source_node_ids"]],
            raw_target_node_ids=[str(value) for value in trace["target_node_ids"]],
            repaired_node_ids=trace_node_ids(current),
        )
        pre_sha = stable_graph_sha256(current)
        source_node_ids = trace_node_ids(current)
        mapped_indices = [] if mapped_action is None else _action_node_indices(mapped_action)
        pre_atom_state = _atom_state_snapshot(
            current,
            source_record=source_record,
            schema=schema,
            indices=mapped_indices,
        )
        if mapped_action is None or target_ids is None:
            skipped += 1
            dependent += 1
            skipped_types.append(action_name)
            action_rows.append(
                {
                    "step": step,
                    "action_type": action_name,
                    "action_arguments": raw_action[1:],
                    "status": "skipped_dependent_action",
                    "pre_graph_sha256": pre_sha,
                    "post_graph_sha256": pre_sha,
                    "source_node_ids": source_node_ids,
                    "current_node_ids": source_node_ids,
                    "affected_atoms": [],
                    "pre_valence": [],
                    "post_valence": [],
                    "pre_formal_charge": [],
                    "post_formal_charge": [],
                    "pre_bond_order_sum": [],
                    "post_bond_order_sum": [],
                    "pre_sanitize_ok": True,
                    "post_sanitize_ok": True,
                    "sanitize_ok": True,
                    "failure_reason": "dependency_missing_after_prior_rollback",
                }
            )
            continue
        try:
            tentative = apply_action_to_graph(
                current,
                mapped_action,
                target_node_ids=target_ids,
            )
            post_atom_state = _atom_state_snapshot(
                tentative,
                source_record=source_record,
                schema=schema,
                indices=_action_node_indices(mapped_action),
            )
            decoded = decode_generated_fullgraph(
                tentative,
                source_record=source_record,
                schema=schema,
            )
        except Exception as exc:
            decoded = None
            post_atom_state = []
            failure_reason = str(exc) or type(exc).__name__
        else:
            failure_reason = str(decoded.failure_reason)
        if decoded is not None and bool(decoded.decode_ok):
            current = tentative
            retained += 1
            action_rows.append(
                {
                    "step": step,
                    "action_type": action_name,
                    "action_arguments": raw_action[1:],
                    "status": "retained",
                    "pre_graph_sha256": pre_sha,
                    "post_graph_sha256": stable_graph_sha256(current),
                    "source_node_ids": source_node_ids,
                    "current_node_ids": trace_node_ids(current),
                    "affected_atoms": post_atom_state,
                    "pre_valence": [row["bond_order_sum"] for row in pre_atom_state],
                    "post_valence": [row["bond_order_sum"] for row in post_atom_state],
                    "pre_formal_charge": [row["formal_charge"] for row in pre_atom_state],
                    "post_formal_charge": [row["formal_charge"] for row in post_atom_state],
                    "pre_bond_order_sum": [row["bond_order_sum"] for row in pre_atom_state],
                    "post_bond_order_sum": [row["bond_order_sum"] for row in post_atom_state],
                    "pre_sanitize_ok": True,
                    "post_sanitize_ok": True,
                    "sanitize_ok": True,
                    "failure_reason": "",
                }
            )
        else:
            skipped += 1
            skipped_types.append(action_name)
            row = {
                "step": step,
                "action_type": action_name,
                "action_arguments": raw_action[1:],
                "status": "skipped_invalid_action",
                "pre_graph_sha256": pre_sha,
                "post_graph_sha256": pre_sha,
                "source_node_ids": source_node_ids,
                "current_node_ids": source_node_ids,
                "affected_atoms": post_atom_state,
                "pre_valence": [row["bond_order_sum"] for row in pre_atom_state],
                "post_valence": [row["bond_order_sum"] for row in post_atom_state],
                "pre_formal_charge": [row["formal_charge"] for row in pre_atom_state],
                "post_formal_charge": [row["formal_charge"] for row in post_atom_state],
                "pre_bond_order_sum": [row["bond_order_sum"] for row in pre_atom_state],
                "post_bond_order_sum": [row["bond_order_sum"] for row in post_atom_state],
                "pre_sanitize_ok": True,
                "post_sanitize_ok": False,
                "sanitize_ok": False,
                "failure_reason": failure_reason,
            }
            action_rows.append(row)
            if first_invalid is None:
                first_invalid = dict(row)
    final_decoded = decode_generated_fullgraph(current, source_record=source_record, schema=schema)
    success = bool(final_decoded.decode_ok)
    return ChemRepairResult(
        graph=current,
        repair_success=success,
        repair_noop=stable_graph_sha256(current) == initial,
        retained_action_count=retained,
        skipped_action_count=skipped,
        dependent_action_skip_count=dependent,
        skipped_action_types=tuple(skipped_types),
        first_invalid_action=first_invalid,
        canonical_smiles=str(final_decoded.canonical_smiles) if success else "",
        output_graph_sha256=stable_graph_sha256(current),
        action_records=tuple(action_rows),
    )
