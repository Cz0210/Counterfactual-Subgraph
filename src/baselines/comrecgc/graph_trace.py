"""Stable graph identities and side-effect-free COMRECGC action tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import append_jsonl, stable_json_sha256, write_json


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def normalized_graph_payload(graph: Any) -> dict[str, Any]:
    """Return a canonical tensor-only graph payload suitable for SHA256.

    Metadata and Python object identity are deliberately excluded. Directed
    edges are normalized with their aligned edge attributes, so serialization
    order cannot change the identity while an edge-feature misalignment does.
    """

    nodes = _plain(getattr(graph, "x"))
    edge_index = _plain(getattr(graph, "edge_index"))
    if not isinstance(nodes, list) or not isinstance(edge_index, list) or len(edge_index) != 2:
        raise ValueError("Graph identity requires x and a [2, E] edge_index.")
    sources, targets = edge_index
    if len(sources) != len(targets):
        raise ValueError("Graph edge_index rows are not aligned.")
    edge_attr_value = getattr(graph, "edge_attr", None)
    edge_attrs = _plain(edge_attr_value) if edge_attr_value is not None else None
    if edge_attrs is not None and len(edge_attrs) != len(sources):
        raise ValueError("Graph edge_attr is not aligned with edge_index.")
    edges = [
        {
            "source": int(source),
            "target": int(target),
            "attr": None if edge_attrs is None else edge_attrs[index],
        }
        for index, (source, target) in enumerate(zip(sources, targets, strict=True))
    ]
    edges.sort(
        key=lambda row: (
            int(row["source"]),
            int(row["target"]),
            json.dumps(row["attr"], sort_keys=True, separators=(",", ":")),
        )
    )
    return {
        "num_nodes": int(getattr(graph, "num_nodes", len(nodes))),
        "x": nodes,
        "directed_edges": edges,
    }


def stable_graph_sha256(graph: Any) -> str:
    return stable_json_sha256(normalized_graph_payload(graph))


def normalized_action(action: Sequence[Any]) -> list[Any]:
    return [_plain(value) for value in action]


def trace_node_ids(graph: Any) -> list[str]:
    existing = getattr(graph, "comrecgc_trace_node_ids", None)
    if existing is not None:
        values = [str(value) for value in existing]
        if len(values) != int(graph.num_nodes):
            raise ValueError("COMRECGC trace node IDs are not aligned with graph nodes.")
        return values
    parent = str(getattr(graph, "comrecgc_parent_id", "unknown_parent"))
    origins = getattr(graph, "comrecgc_node_origin", None)
    if origins is None:
        return [f"{parent}:node:{index}" for index in range(int(graph.num_nodes))]
    values = _plain(origins)
    return [
        f"{parent}:source:{int(origin)}"
        if int(origin) >= 0
        else f"{parent}:unknown:{index}"
        for index, origin in enumerate(values)
    ]


@dataclass
class ActionTraceRecorder:
    """Record enumerated and actually selected actions without changing RNG."""

    enumerated: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    selected: list[dict[str, Any]] = field(default_factory=list)
    predecessor_by_official_hash: dict[str, dict[str, Any]] = field(default_factory=dict)
    move_index: int = 0

    def record_enumerated(
        self,
        *,
        source_graph: Any,
        target_graph: Any,
        action: Sequence[Any],
        source_node_ids: Sequence[str],
        target_node_ids: Sequence[str],
    ) -> None:
        source_sha = stable_graph_sha256(source_graph)
        target_sha = stable_graph_sha256(target_graph)
        record = {
            "source_graph_sha256": source_sha,
            "target_graph_sha256": target_sha,
            "action": normalized_action(action),
            "source_node_ids": [str(value) for value in source_node_ids],
            "target_node_ids": [str(value) for value in target_node_ids],
        }
        self.enumerated.setdefault((source_sha, target_sha), []).append(record)

    def wrap_move(self, original: Callable[..., Any], module: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            graphs_hash = list(kwargs.get("graphs_hash", args[0] if args else []))
            source_graphs = [module.graph_map[value][0] for value in graphs_hash]
            result = original(*args, **kwargs)
            next_hashes, teleported = result[0], bool(result[1])
            if teleported or next_hashes is None:
                self.selected.append(
                    {
                        "move_index": self.move_index,
                        "event": "teleport",
                        "source_official_hashes": [str(value) for value in graphs_hash],
                    }
                )
                consumed_sources = {stable_graph_sha256(graph) for graph in source_graphs}
                self.enumerated = {
                    key: value
                    for key, value in self.enumerated.items()
                    if key[0] not in consumed_sources
                }
                self.move_index += 1
                return result
            for head_index, (source_hash, target_hash, source_graph) in enumerate(
                zip(graphs_hash, list(next_hashes), source_graphs, strict=True)
            ):
                target_graph = module.graph_map[target_hash][0]
                source_sha = stable_graph_sha256(source_graph)
                target_sha = stable_graph_sha256(target_graph)
                candidates = self.enumerated.get((source_sha, target_sha), [])
                unique: dict[str, dict[str, Any]] = {
                    json.dumps(row["action"], separators=(",", ":")): row
                    for row in candidates
                }
                action_record = next(iter(unique.values())) if len(unique) == 1 else None
                event = {
                    "move_index": self.move_index,
                    "head_index": head_index,
                    "event": "selected_transition",
                    "source_official_hash": str(source_hash),
                    "target_official_hash": str(target_hash),
                    "source_graph_sha256": source_sha,
                    "target_graph_sha256": target_sha,
                    "action_resolution": "exact" if action_record else (
                        "missing" if not unique else "ambiguous"
                    ),
                    "action": None if action_record is None else action_record["action"],
                    "source_node_ids": trace_node_ids(source_graph),
                    "target_node_ids": trace_node_ids(target_graph),
                    "parent_id": str(getattr(target_graph, "comrecgc_parent_id", "")),
                }
                self.selected.append(event)
                if action_record is not None:
                    self.predecessor_by_official_hash.setdefault(str(target_hash), event)
            consumed_sources = {stable_graph_sha256(graph) for graph in source_graphs}
            self.enumerated = {
                key: value
                for key, value in self.enumerated.items()
                if key[0] not in consumed_sources
            }
            self.move_index += 1
            return result

        return wrapped

    def _lineage_for_hash(self, official_hash: str) -> list[dict[str, Any]] | None:
        reversed_path: list[dict[str, Any]] = []
        current = str(official_hash)
        seen: set[str] = set()
        while current in self.predecessor_by_official_hash:
            if current in seen:
                raise ValueError("COMRECGC trace predecessor graph contains a cycle.")
            seen.add(current)
            event = self.predecessor_by_official_hash[current]
            reversed_path.append(event)
            current = str(event["source_official_hash"])
        if not reversed_path:
            return None
        return list(reversed(reversed_path))

    def candidate_lineage(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        graph_map = payload.get("graph_map") or {}
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(payload.get("counterfactual_candidates") or []):
            official_hash = str(candidate.get("graph_hash"))
            graph_entry = graph_map.get(candidate.get("graph_hash"))
            graph = graph_entry[0] if graph_entry else None
            path = self._lineage_for_hash(official_hash)
            rows.append(
                {
                    "candidate_index": index,
                    "official_graph_hash": official_hash,
                    "stable_graph_sha256": stable_graph_sha256(graph) if graph is not None else None,
                    "parent_id": str(getattr(graph, "comrecgc_parent_id", "")) if graph is not None else "",
                    "action_lineage_resolved": bool(
                        path is not None
                        and all(row.get("action_resolution") == "exact" for row in path)
                    ),
                    "actions": [] if path is None else path,
                }
            )
        return rows

    def write(self, output_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        selected_path = root / "selected_action_trace.jsonl"
        selected_path.unlink(missing_ok=True)
        for row in self.selected:
            append_jsonl(selected_path, row)
        lineage = self.candidate_lineage(payload)
        lineage_path = root / "candidate_action_lineage.json"
        write_json(lineage_path, lineage)
        summary = {
            "trace_schema_version": 1,
            "trace_only": True,
            "rng_calls_added": 0,
            "enumerated_transition_pair_count": len(self.enumerated),
            "selected_transition_count": sum(
                row.get("event") == "selected_transition" for row in self.selected
            ),
            "teleport_count": sum(row.get("event") == "teleport" for row in self.selected),
            "candidate_count": len(lineage),
            "candidate_lineage_resolved_count": sum(
                bool(row["action_lineage_resolved"]) for row in lineage
            ),
            "selected_trace_path": str(selected_path),
            "candidate_lineage_path": str(lineage_path),
        }
        write_json(root / "trace_summary.json", summary)
        return summary


def _importance(value: Any) -> list[float]:
    return [float(item) for item in _plain(value or [])]


def normalized_candidate_sequence(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    graph_map = payload.get("graph_map") or {}
    rows: list[dict[str, Any]] = []
    for candidate in payload.get("counterfactual_candidates") or []:
        official_hash = candidate.get("graph_hash")
        graph_entry = graph_map.get(official_hash)
        if graph_entry is None:
            raise ValueError(f"Candidate graph is absent from graph_map: {official_hash!r}")
        rows.append(
            {
                "stable_graph_sha256": stable_graph_sha256(graph_entry[0]),
                "frequency": int(candidate.get("frequency", 0)),
                "importance_parts": _importance(candidate.get("importance_parts")),
            }
        )
    return rows


def assert_trace_parity(
    reference_payload: Mapping[str, Any], traced_payload: Mapping[str, Any]
) -> dict[str, Any]:
    reference = normalized_candidate_sequence(reference_payload)
    traced = normalized_candidate_sequence(traced_payload)
    if reference != traced:
        raise ValueError("Trace-on/off candidate topology, features, frequency, importance, or order differ.")
    return {
        "trace_parity_passed": True,
        "candidate_count": len(reference),
        "candidate_sequence_sha256": stable_json_sha256(reference),
        "compared_fields": [
            "stable_graph_sha256",
            "frequency",
            "importance_parts",
            "order",
        ],
    }
