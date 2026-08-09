"""Stable graph identities and side-effect-free COMRECGC action tracing."""

from __future__ import annotations

import json
import math
import os
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import atomic_write_bytes, sha256_file, stable_json_sha256, write_json


TRACE_IMPORTANCE_ABS_TOLERANCE = 1e-6
MODEL_CF_IMPORTANCE_THRESHOLD = 0.5


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


def _normalized_nodes_and_edges(graph: Any) -> tuple[list[Any], list[int], list[int]]:
    nodes = _plain(getattr(graph, "x"))
    edge_index = _plain(getattr(graph, "edge_index"))
    if not isinstance(nodes, list) or not isinstance(edge_index, list) or len(edge_index) != 2:
        raise ValueError("Graph identity requires x and a [2, E] edge_index.")
    sources, targets = edge_index
    if len(sources) != len(targets):
        raise ValueError("Graph edge_index rows are not aligned.")
    return nodes, sources, targets


def normalized_graph_payload(graph: Any) -> dict[str, Any]:
    """Return a canonical typed graph payload suitable for SHA256.

    Metadata and Python object identity are deliberately excluded. Directed
    edges are normalized with their aligned edge attributes, so serialization
    order cannot change the identity while an edge-feature misalignment does.
    """

    nodes, sources, targets = _normalized_nodes_and_edges(graph)
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


def normalized_untyped_graph_payload(graph: Any) -> dict[str, Any]:
    """Return the canonical node/adjacency identity used by untyped COMRECGC.

    Pinned upstream COMRECGC mutates ``edge_index`` but does not consume or
    consistently update the TU dataset's bond-label ``edge_attr`` sidecar.
    Callers must opt into this identity explicitly; the typed identity above
    remains strict and continues to reject a stale or misaligned sidecar.
    """

    nodes, sources, targets = _normalized_nodes_and_edges(graph)
    edges = sorted(
        (
            {"source": int(source), "target": int(target)}
            for source, target in zip(sources, targets, strict=True)
        ),
        key=lambda row: (int(row["source"]), int(row["target"])),
    )
    return {
        "num_nodes": int(getattr(graph, "num_nodes", len(nodes))),
        "x": nodes,
        "directed_edges": edges,
    }


def stable_graph_sha256(graph: Any) -> str:
    return stable_json_sha256(normalized_graph_payload(graph))


def stable_untyped_graph_sha256(graph: Any) -> str:
    return stable_json_sha256(normalized_untyped_graph_payload(graph))


def _recorded_trace_sha_matches(graph: Any, expected: str) -> bool:
    """Accept the current untyped trace identity and legacy aligned typed traces."""

    if stable_untyped_graph_sha256(graph) == str(expected):
        return True
    try:
        return stable_graph_sha256(graph) == str(expected)
    except ValueError:
        return False


def normalized_action(action: Sequence[Any]) -> list[Any]:
    return [_plain(value) for value in action]


def apply_action_to_normalized_payload(
    source_graph: Any, action: Sequence[Any]
) -> dict[str, Any]:
    """Apply one pinned-upstream edit in the canonical untyped graph space."""

    payload = normalized_untyped_graph_payload(source_graph)
    nodes = [list(row) for row in payload["x"]]
    edges = [dict(row) for row in payload["directed_edges"]]
    name = str(action[0])
    if name == "NOTHING":
        pass
    elif name == "NLC":
        node, label = int(action[1]), int(action[2])
        if not 0 <= node < len(nodes) or not 0 <= label < len(nodes[node]):
            raise ValueError("Recovered NLC action is outside the source graph.")
        nodes[node] = [0.0] * len(nodes[node])
        nodes[node][label] = 1.0
    elif name in {"NA", "INA"}:
        attachment, label = int(action[1]), int(action[2])
        if not nodes or not 0 <= label < len(nodes[0]):
            raise ValueError("Recovered node-addition label is outside the vocabulary.")
        new_node = len(nodes)
        feature = [0.0] * len(nodes[0])
        feature[label] = 1.0
        nodes.append(feature)
        if name == "NA":
            if not 0 <= attachment < new_node:
                raise ValueError("Recovered node attachment is outside the source graph.")
            edges.extend(
                [
                    {"source": attachment, "target": new_node},
                    {"source": new_node, "target": attachment},
                ]
            )
    elif name in {"NR", "INR"}:
        removed = int(action[1])
        if not 0 <= removed < len(nodes):
            raise ValueError("Recovered node removal is outside the source graph.")
        nodes.pop(removed)
        retained: list[dict[str, Any]] = []
        for edge in edges:
            source, target = int(edge["source"]), int(edge["target"])
            if removed in {source, target}:
                continue
            retained.append(
                {
                    **edge,
                    "source": source - 1 if source > removed else source,
                    "target": target - 1 if target > removed else target,
                }
            )
        edges = retained
    elif name in {"ER", "ERR"}:
        first, second = int(action[1]), int(action[2])
        edges = [
            edge
            for edge in edges
            if (int(edge["source"]), int(edge["target"]))
            not in {(first, second), (second, first)}
        ]
    elif name == "EA":
        first, second = int(action[1]), int(action[2])
        if not (0 <= first < len(nodes) and 0 <= second < len(nodes)):
            raise ValueError("Recovered edge addition is outside the source graph.")
        edges.extend(
            [
                {"source": first, "target": second},
                {"source": second, "target": first},
            ]
        )
    else:
        raise ValueError(f"Unsupported recovered COMRECGC action: {name}")
    edges.sort(
        key=lambda row: (
            int(row["source"]),
            int(row["target"]),
        )
    )
    return {"num_nodes": len(nodes), "x": nodes, "directed_edges": edges}


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
    """Record selected actions without retaining a second in-memory walk history.

    The official runtime still owns its graph state.  This recorder keeps only
    the first predecessor needed to reconstruct candidate lineages and streams
    audit events to deterministic chunks.  A resumed deterministic replay
    reuses byte-identical completed chunks instead of appending duplicates.
    """

    output_dir: str | Path | None = None
    chunk_size: int = 512
    compact_enumeration: bool = False
    enumerated: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    predecessor_by_official_hash: dict[str, dict[str, Any]] = field(default_factory=dict)
    move_index: int = 0
    enumerated_transition_count: int = 0
    selected_transition_count: int = 0
    teleport_count: int = 0
    transition_cache_hit_count: int = 0
    transition_cache_miss_count: int = 0
    _trace_root: Path | None = field(default=None, init=False, repr=False)
    _pending_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _chunks: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _enumerated_by_target_object: dict[
        int, tuple[Any, list[list[Any]]]
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.chunk_size) <= 0:
            raise ValueError("COMRECGC trace chunk_size must be positive.")
        if self.output_dir is not None:
            self._configure_output(self.output_dir)

    def _configure_output(self, output_dir: str | Path) -> Path:
        root = Path(output_dir).expanduser().resolve()
        if self._trace_root is not None and self._trace_root != root:
            raise ValueError(
                "COMRECGC trace recorder cannot change output directories: "
                f"{self._trace_root} != {root}"
            )
        (root / "selected_action_trace_chunks").mkdir(parents=True, exist_ok=True)
        self._trace_root = root
        return root

    def _flush_chunks(self, *, final: bool) -> None:
        if self._trace_root is None:
            return
        while len(self._pending_events) >= int(self.chunk_size) or (
            final and self._pending_events
        ):
            count = min(len(self._pending_events), int(self.chunk_size))
            rows = self._pending_events[:count]
            encoded = (
                "".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n"
                    for row in rows
                )
            ).encode("utf-8")
            index = len(self._chunks)
            relative = Path("selected_action_trace_chunks") / f"part-{index:06d}.jsonl"
            destination = self._trace_root / relative
            materialization = "atomic_write"
            if destination.exists():
                if destination.read_bytes() != encoded:
                    raise ValueError(
                        "Existing COMRECGC trace chunk differs during deterministic resume: "
                        f"{destination}"
                    )
                materialization = "adopt_existing_identical"
            else:
                atomic_write_bytes(destination, encoded)
            self._chunks.append(
                {
                    "index": index,
                    "path": relative.as_posix(),
                    "row_count": count,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "materialization": materialization,
                }
            )
            del self._pending_events[:count]

    def _stream_event(self, event: dict[str, Any]) -> None:
        if event.get("event") == "selected_transition":
            self.selected_transition_count += 1
        elif event.get("event") == "teleport":
            self.teleport_count += 1
        self._pending_events.append(event)
        self._flush_chunks(final=False)

    def _discard_enumerated_sources(self, source_shas: set[str]) -> None:
        for key in [key for key in self.enumerated if key[0] in source_shas]:
            self.enumerated.pop(key, None)

    def _forget_enumerated_target(self, object_id: int, reference: Any) -> None:
        current = self._enumerated_by_target_object.get(object_id)
        if current is not None and current[0] is reference:
            self._enumerated_by_target_object.pop(object_id, None)

    def _record_compact_enumerated_target(
        self, target_graph: Any, action: Sequence[Any]
    ) -> None:
        object_id = id(target_graph)
        current = self._enumerated_by_target_object.get(object_id)
        if current is not None and current[0]() is target_graph:
            current[1].append(normalized_action(action))
            return
        try:
            reference = weakref.ref(
                target_graph,
                lambda resolved, key=object_id: self._forget_enumerated_target(
                    key, resolved
                ),
            )
        except TypeError as exc:
            raise TypeError(
                "Compact COMRECGC action tracing requires weak-referenceable graph objects."
            ) from exc
        self._enumerated_by_target_object[object_id] = (
            reference,
            [normalized_action(action)],
        )

    def _compact_transition_records(
        self,
        module: Any,
        *,
        source_hash: Any,
        target_hash: Any,
    ) -> list[dict[str, Any]]:
        transition = getattr(module, "transitions", {}).get(source_hash)
        if not isinstance(transition, tuple) or len(transition) < 2:
            return []
        target_hashes, target_graphs = transition[0], transition[1]
        records: list[dict[str, Any]] = []
        for resolved_hash, graph in zip(target_hashes, target_graphs, strict=True):
            if resolved_hash != target_hash:
                continue
            current = self._enumerated_by_target_object.get(id(graph))
            if current is None or current[0]() is not graph:
                continue
            records.extend({"action": list(action)} for action in current[1])
        return records

    def record_enumerated(
        self,
        *,
        source_graph: Any,
        target_graph: Any,
        action: Sequence[Any],
    ) -> None:
        if self.compact_enumeration:
            self._record_compact_enumerated_target(target_graph, action)
            self.enumerated_transition_count += 1
            return
        source_sha = stable_untyped_graph_sha256(source_graph)
        target_sha = stable_untyped_graph_sha256(target_graph)
        record = {
            "source_graph_sha256": source_sha,
            "target_graph_sha256": target_sha,
            "action": normalized_action(action),
        }
        self.enumerated.setdefault((source_sha, target_sha), []).append(record)
        self.enumerated_transition_count += 1

    def wrap_move(self, original: Callable[..., Any], module: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Local import avoids a module cycle: live_graph_state uses the
            # canonical graph identity defined above.
            from .live_graph_state import pin_graphs, resolve_graph, resolve_graphs

            graphs_hash = list(kwargs.get("graphs_hash", args[0] if args else []))
            with pin_graphs(module, graphs_hash):
                source_graphs = resolve_graphs(module, graphs_hash)
                transitions = getattr(module, "transitions", {})
                transition_cache_before = [
                    value in transitions for value in graphs_hash
                ]
                result = original(*args, **kwargs)
                next_hashes, teleported = result[0], bool(result[1])
                if teleported or next_hashes is None:
                    self._stream_event(
                        {
                            "move_index": self.move_index,
                            "event": "teleport",
                            "source_official_hashes": [str(value) for value in graphs_hash],
                        }
                    )
                    if not self.compact_enumeration:
                        consumed_sources = {
                            stable_untyped_graph_sha256(graph) for graph in source_graphs
                        }
                        self._discard_enumerated_sources(consumed_sources)
                    self.move_index += 1
                    return result
                resolved_next_hashes = list(next_hashes)
                self.transition_cache_hit_count += sum(transition_cache_before)
                self.transition_cache_miss_count += len(transition_cache_before) - sum(
                    transition_cache_before
                )
                with pin_graphs(module, resolved_next_hashes):
                    for head_index, (source_hash, target_hash, source_graph) in enumerate(
                        zip(graphs_hash, resolved_next_hashes, source_graphs, strict=True)
                    ):
                        target_graph = resolve_graph(module, target_hash)
                        source_sha = stable_untyped_graph_sha256(source_graph)
                        target_sha = stable_untyped_graph_sha256(target_graph)
                        candidates = (
                            self._compact_transition_records(
                                module,
                                source_hash=source_hash,
                                target_hash=target_hash,
                            )
                            if self.compact_enumeration
                            else self.enumerated.get((source_sha, target_sha), [])
                        )
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
                            "parent_id": str(getattr(target_graph, "comrecgc_parent_id", "")),
                        }
                        self._stream_event(event)
                        if action_record is not None:
                            self.predecessor_by_official_hash.setdefault(str(target_hash), event)
                if not self.compact_enumeration:
                    consumed_sources = {
                        stable_untyped_graph_sha256(graph) for graph in source_graphs
                    }
                    self._discard_enumerated_sources(consumed_sources)
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
        graph_map_by_string = {str(key): value for key, value in graph_map.items()}
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(payload.get("counterfactual_candidates") or []):
            official_hash = str(candidate.get("graph_hash"))
            graph_entry = graph_map.get(candidate.get("graph_hash"))
            graph = graph_entry[0] if graph_entry else None
            path = self._lineage_for_hash(official_hash)
            enriched_path: list[dict[str, Any]] = []
            node_lineage_resolved = path is not None
            if path:
                source_entry = graph_map_by_string.get(str(path[0]["source_official_hash"]))
                if source_entry:
                    current_node_ids = trace_node_ids(source_entry[0])
                    for path_index, event in enumerate(path):
                        enriched = dict(event)
                        source_graph_entry = graph_map_by_string.get(
                            str(event["source_official_hash"])
                        )
                        target_graph_entry = graph_map_by_string.get(
                            str(event["target_official_hash"])
                        )
                        source_node_ids = list(current_node_ids)
                        action = list(event.get("action") or [])
                        target_node_ids = list(source_node_ids)
                        if not action:
                            node_lineage_resolved = False
                        elif str(action[0]) in {"NA", "INA"}:
                            target_node_ids.append(
                                "new:"
                                + str(getattr(graph, "comrecgc_parent_id", ""))
                                + f":move:{int(event['move_index'])}:head:{int(event['head_index'])}"
                                + f":path:{path_index}:target:{event['target_official_hash']}"
                            )
                        elif str(action[0]) in {"NR", "INR"}:
                            remove_index = int(action[1])
                            if not 0 <= remove_index < len(target_node_ids):
                                node_lineage_resolved = False
                            else:
                                target_node_ids.pop(remove_index)
                        replay_exact = bool(source_graph_entry and target_graph_entry)
                        if replay_exact:
                            replay_exact = (
                                apply_action_to_normalized_payload(
                                    source_graph_entry[0], action
                                )
                                == normalized_untyped_graph_payload(target_graph_entry[0])
                            )
                        enriched["action_replay_exact"] = replay_exact
                        node_lineage_resolved = node_lineage_resolved and replay_exact
                        enriched["source_node_ids"] = source_node_ids
                        enriched["target_node_ids"] = target_node_ids
                        enriched_path.append(enriched)
                        current_node_ids = target_node_ids
                    if graph is not None and len(current_node_ids) != int(graph.num_nodes):
                        node_lineage_resolved = False
                else:
                    node_lineage_resolved = False
            rows.append(
                {
                    "candidate_index": index,
                    "official_graph_hash": official_hash,
                    "stable_graph_sha256": (
                        stable_untyped_graph_sha256(graph) if graph is not None else None
                    ),
                    "parent_id": str(getattr(graph, "comrecgc_parent_id", "")) if graph is not None else "",
                    "action_lineage_resolved": bool(
                        node_lineage_resolved
                        and path is not None
                        and all(row.get("action_resolution") == "exact" for row in path)
                    ),
                    "actions": enriched_path,
                }
            )
        return rows

    def write(
        self,
        output_dir: str | Path,
        payload: Mapping[str, Any],
        *,
        source_graphs_by_parent_id: Mapping[str, Any] | None = None,
        compact_candidate_lineage: bool = False,
    ) -> dict[str, Any]:
        root = self._configure_output(output_dir)
        self._flush_chunks(final=True)
        expected_parts = {Path(row["path"]).name for row in self._chunks}
        actual_parts = {
            path.name for path in (root / "selected_action_trace_chunks").glob("part-*.jsonl")
        }
        if actual_parts != expected_parts:
            raise ValueError(
                "COMRECGC trace resume found stale or missing chunks: "
                f"expected={sorted(expected_parts)}, actual={sorted(actual_parts)}"
            )
        selected_manifest_path = root / "selected_action_trace_manifest.json"
        write_json(
            selected_manifest_path,
            {
                "schema_version": 1,
                "format": "chunked_jsonl",
                "graph_identity_mode": "official_untyped_node_adjacency_v1",
                "chunk_size": int(self.chunk_size),
                "row_count": self.selected_transition_count + self.teleport_count,
                "chunks": self._chunks,
                "resume_policy": "reuse_byte_identical_completed_chunks",
            },
        )
        lineage_path = root / "candidate_action_lineage.json"
        lineage_index_path: Path | None = None
        if compact_candidate_lineage:
            if source_graphs_by_parent_id is None:
                raise ValueError(
                    "Compact COMRECGC lineage requires frozen source graphs."
                )
            lineage_index_path = root / "candidate_action_lineage_index.jsonl"
            descriptor, temporary_name = tempfile.mkstemp(
                dir=root,
                prefix=f".{lineage_index_path.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            lineage_count = 0
            lineage_resolved_count = 0
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    for expected_index, row in enumerate(
                        iter_candidate_lineage_from_selected_trace(
                            payload,
                            iter_selected_trace(selected_manifest_path),
                            source_graphs_by_parent_id=source_graphs_by_parent_id,
                            include_actions=False,
                        )
                    ):
                        if int(row["candidate_index"]) != expected_index:
                            raise ValueError(
                                "Compact COMRECGC lineage index is not in candidate order."
                            )
                        handle.write(
                            json.dumps(
                                row,
                                sort_keys=True,
                                ensure_ascii=True,
                                default=str,
                            )
                        )
                        handle.write("\n")
                        lineage_count += 1
                        lineage_resolved_count += int(
                            row["action_lineage_resolved"] is True
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                if lineage_index_path.exists():
                    if lineage_index_path.read_bytes() != temporary.read_bytes():
                        raise ValueError(
                            "Existing compact COMRECGC lineage index differs during resume."
                        )
                    temporary.unlink()
                else:
                    os.replace(temporary, lineage_index_path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            write_json(
                lineage_path,
                {
                    "schema_version": 2,
                    "format": "selected_trace_predecessor_index",
                    "candidate_count": lineage_count,
                    "candidate_lineage_resolved_count": lineage_resolved_count,
                    "candidate_index_path": lineage_index_path.name,
                    "candidate_index_sha256": sha256_file(lineage_index_path),
                    "selected_trace_manifest_path": selected_manifest_path.name,
                    "selected_trace_manifest_sha256": sha256_file(
                        selected_manifest_path
                    ),
                    "candidate_actions_inlined": False,
                    "reconstruction_policy": (
                        "stream_one_candidate_from_selected_trace_v1"
                    ),
                },
            )
        else:
            lineage = (
                recover_candidate_lineage_from_selected_trace(
                    payload,
                    iter_selected_trace(selected_manifest_path),
                    source_graphs_by_parent_id=source_graphs_by_parent_id,
                )
                if source_graphs_by_parent_id is not None
                else self.candidate_lineage(payload)
            )
            write_json(lineage_path, lineage)
            lineage_count = len(lineage)
            lineage_resolved_count = sum(
                bool(row["action_lineage_resolved"]) for row in lineage
            )
        summary = {
            "trace_schema_version": 1,
            "trace_only": True,
            "graph_identity_mode": "official_untyped_node_adjacency_v1",
            "rng_calls_added": 0,
            "candidate_payload_mutated": False,
            "enumeration_trace_mode": (
                "weak_target_object_action_index_v1"
                if self.compact_enumeration
                else "stable_graph_pair_v1"
            ),
            "enumerated_transition_count": self.enumerated_transition_count,
            "live_enumerated_transition_pair_count": (
                len(self._enumerated_by_target_object)
                if self.compact_enumeration
                else len(self.enumerated)
            ),
            "transition_cache_hit_count": self.transition_cache_hit_count,
            "transition_cache_miss_count": self.transition_cache_miss_count,
            "transition_cache_policy": "pinned_upstream_in_memory_transitions_v1",
            "selected_transition_count": self.selected_transition_count,
            "teleport_count": self.teleport_count,
            "candidate_count": lineage_count,
            "candidate_lineage_resolved_count": lineage_resolved_count,
            "selected_trace_path": str(selected_manifest_path),
            "selected_trace_chunk_count": len(self._chunks),
            "max_buffered_event_count": int(self.chunk_size),
            "candidate_lineage_path": str(lineage_path),
            "candidate_lineage_format": (
                "selected_trace_predecessor_index"
                if compact_candidate_lineage
                else "inline_json"
            ),
            "candidate_lineage_index_path": (
                str(lineage_index_path) if lineage_index_path is not None else None
            ),
            "max_materialized_candidate_lineages": (
                1 if compact_candidate_lineage else lineage_count
            ),
            "lineage_recovery_policy": (
                "pinned_upstream_official_hash_source_root_v2"
                if source_graphs_by_parent_id is not None
                else "runtime_recorded_predecessor_v1"
            ),
        }
        write_json(root / "trace_summary.json", summary)
        write_json(
            root / "_TRACE_COMPLETE.json",
            {
                "trace_complete": True,
                "selected_trace_manifest_sha256": sha256_file(selected_manifest_path),
                "candidate_lineage_sha256": sha256_file(lineage_path),
            },
        )
        return summary


def iter_selected_trace(manifest_path: str | Path) -> Any:
    """Stream a completed chunked trace in exact walk order."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "chunked_jsonl":
        raise ValueError(f"Unsupported COMRECGC selected trace format: {manifest.get('format')!r}")
    total = 0
    for expected_index, chunk in enumerate(manifest.get("chunks") or []):
        if int(chunk.get("index", -1)) != expected_index:
            raise ValueError("COMRECGC selected trace chunks are not contiguous.")
        chunk_path = path.parent / str(chunk["path"])
        if sha256_file(chunk_path) != str(chunk["sha256"]):
            raise ValueError(f"COMRECGC selected trace chunk SHA256 mismatch: {chunk_path}")
        chunk_count = 0
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk_count += 1
                total += 1
                yield json.loads(line)
        if chunk_count != int(chunk["row_count"]):
            raise ValueError(f"COMRECGC selected trace chunk row count mismatch: {chunk_path}")
    if total != int(manifest.get("row_count", -1)):
        raise ValueError("COMRECGC selected trace total row count mismatch.")


def load_selected_trace(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Reload a small completed trace as a list for compatibility/tests."""

    return list(iter_selected_trace(manifest_path))


def _feature_rows(graph: Any) -> list[list[float]]:
    values = _plain(getattr(graph, "x"))
    if not isinstance(values, list):
        raise ValueError("COMRECGC action recovery requires a matrix-valued x tensor.")
    return [[float(item) for item in row] for row in values]


def _undirected_edges(graph: Any) -> set[tuple[int, int]]:
    edge_index = _plain(getattr(graph, "edge_index"))
    if not isinstance(edge_index, list) or len(edge_index) != 2:
        raise ValueError("COMRECGC action recovery requires a [2, E] edge_index.")
    return {
        tuple(sorted((int(first), int(second))))
        for first, second in zip(edge_index[0], edge_index[1], strict=True)
        if int(first) != int(second)
    }


def _removed_node_edges(
    edges: set[tuple[int, int]], removed: int
) -> set[tuple[int, int]]:
    def shifted(value: int) -> int:
        return value - 1 if value > removed else value

    return {
        tuple(sorted((shifted(first), shifted(second))))
        for first, second in edges
        if removed not in {first, second}
    }


def _is_bridge(
    edges: set[tuple[int, int]], edge: tuple[int, int], num_nodes: int
) -> bool:
    remaining = edges - {edge}
    adjacency = {index: set() for index in range(num_nodes)}
    for first, second in remaining:
        adjacency[first].add(second)
        adjacency[second].add(first)
    first, second = edge
    pending = [first]
    seen = {first}
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return second not in seen


def infer_official_single_edit(source_graph: Any, target_graph: Any) -> list[Any]:
    """Recover one pinned-upstream neighbor action from an exact graph delta."""

    source_x = _feature_rows(source_graph)
    target_x = _feature_rows(target_graph)
    source_edges = _undirected_edges(source_graph)
    target_edges = _undirected_edges(target_graph)
    source_nodes = len(source_x)
    target_nodes = len(target_x)

    if source_nodes == target_nodes:
        changed_features = [
            index
            for index, (source_row, target_row) in enumerate(
                zip(source_x, target_x, strict=True)
            )
            if source_row != target_row
        ]
        added_edges = target_edges - source_edges
        removed_edges = source_edges - target_edges
        if len(changed_features) == 1 and not added_edges and not removed_edges:
            node = changed_features[0]
            active = [index for index, value in enumerate(target_x[node]) if value == 1.0]
            if len(active) != 1 or any(
                value not in {0.0, 1.0} for value in target_x[node]
            ):
                raise ValueError("Recovered NLC target is not one-hot encoded.")
            return ["NLC", node, active[0]]
        if not changed_features and len(added_edges) == 1 and not removed_edges:
            first, second = next(iter(added_edges))
            return ["EA", first, second]
        if not changed_features and len(removed_edges) == 1 and not added_edges:
            edge = next(iter(removed_edges))
            action = "ERR" if _is_bridge(source_edges, edge, source_nodes) else "ER"
            return [action, edge[0], edge[1]]
        if not changed_features and not added_edges and not removed_edges:
            return ["NOTHING", 0, 0]
    elif target_nodes == source_nodes + 1:
        if target_x[:-1] != source_x:
            raise ValueError("Recovered node addition changes retained node features.")
        active = [index for index, value in enumerate(target_x[-1]) if value == 1.0]
        if len(active) != 1 or any(value not in {0.0, 1.0} for value in target_x[-1]):
            raise ValueError("Recovered node addition is not one-hot encoded.")
        if not source_edges <= target_edges:
            raise ValueError("Recovered node addition also removes an edge.")
        added_edges = target_edges - source_edges
        if not added_edges:
            return ["INA", active[0], active[0]]
        if len(added_edges) == 1:
            first, second = next(iter(added_edges))
            new_node = source_nodes
            if new_node in {first, second}:
                attachment = second if first == new_node else first
                return ["NA", attachment, active[0]]
    elif target_nodes == source_nodes - 1:
        matches: list[list[Any]] = []
        for removed in range(source_nodes):
            if source_x[:removed] + source_x[removed + 1 :] != target_x:
                continue
            if _removed_node_edges(source_edges, removed) != target_edges:
                continue
            degree = sum(removed in edge for edge in source_edges)
            if degree == 0:
                matches.append(["INR", removed, removed])
            elif degree == 1:
                matches.append(["NR", removed, removed])
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        "Selected COMRECGC transition is not one unique pinned-upstream single edit."
    )


def _lineage_recovery_context(
    payload: Mapping[str, Any],
    selected_events: Any,
    source_graphs_by_parent_id: Mapping[str, Any] | None,
) -> dict[str, Any]:
    graph_map = payload.get("graph_map") or {}
    graph_by_stable_key: dict[tuple[str, str], Any] = {}
    graph_by_official_key: dict[tuple[str, str], Any] = {}
    official_matches: dict[str, list[tuple[tuple[str, str], Any]]] = {}
    for official_hash, entry in graph_map.items():
        graph = entry[0]
        parent_id = str(getattr(graph, "comrecgc_parent_id", ""))
        key = (parent_id, stable_untyped_graph_sha256(graph))
        existing_stable = graph_by_stable_key.get(key)
        if existing_stable is not None and (
            normalized_untyped_graph_payload(existing_stable)
            != normalized_untyped_graph_payload(graph)
        ):
            raise ValueError("Stable COMRECGC graph identity collision during trace recovery.")
        graph_by_stable_key[key] = graph
        official_key = (parent_id, str(official_hash))
        existing = graph_by_official_key.get(official_key)
        if (
            existing is not None
            and normalized_untyped_graph_payload(existing)
            != normalized_untyped_graph_payload(graph)
        ):
            raise ValueError("Official COMRECGC graph identity collision during trace recovery.")
        graph_by_official_key[official_key] = graph
        official_matches.setdefault(str(official_hash), []).append((official_key, graph))

    frozen_sources = {
        str(parent_id): graph
        for parent_id, graph in (source_graphs_by_parent_id or {}).items()
    }
    for parent_id, graph in frozen_sources.items():
        graph_parent_id = str(getattr(graph, "comrecgc_parent_id", ""))
        if graph_parent_id and graph_parent_id != parent_id:
            raise ValueError(
                "Frozen COMRECGC source graph parent identity mismatch: "
                f"key={parent_id!r}, graph={graph_parent_id!r}."
            )

    predecessor: dict[tuple[str, str], dict[str, Any]] = {}
    observed_source_keys: set[tuple[str, str]] = set()
    for raw_event in selected_events:
        if raw_event.get("event") != "selected_transition":
            continue
        event = dict(raw_event)
        parent_id = str(event.get("parent_id") or "")
        if not parent_id:
            raise ValueError("Selected trace event has no COMRECGC parent identity.")
        source_key = (parent_id, str(event["source_official_hash"]))
        target_key = (parent_id, str(event["target_official_hash"]))
        source_graph = graph_by_official_key.get(source_key)
        target_graph = graph_by_official_key.get(target_key)
        if source_graph is None or target_graph is None:
            raise ValueError("Selected trace references a graph absent from the frozen payload.")
        if not _recorded_trace_sha_matches(
            source_graph, str(event["source_graph_sha256"])
        ):
            raise ValueError("Selected trace source graph SHA256 differs from the frozen payload.")
        if not _recorded_trace_sha_matches(
            target_graph, str(event["target_graph_sha256"])
        ):
            raise ValueError("Selected trace target graph SHA256 differs from the frozen payload.")
        inferred = infer_official_single_edit(source_graph, target_graph)
        replayed = apply_action_to_normalized_payload(source_graph, inferred)
        if replayed != normalized_untyped_graph_payload(target_graph):
            raise ValueError(
                "Recovered selected action does not replay to the exact target graph."
            )
        recorded = event.get("action")
        if recorded is not None and list(recorded) != inferred:
            raise ValueError(
                "Recorded selected action disagrees with its exact source/target graph delta."
            )
        event["action"] = inferred
        event["action_resolution"] = "exact"
        event["action_recovery"] = (
            "recorded_exact" if recorded is not None else "inferred_exact_graph_delta_v1"
        )
        event["action_replay_exact"] = True
        existing = predecessor.get(target_key)
        if existing is not None:
            comparable_fields = (
                "parent_id",
                "source_official_hash",
                "target_official_hash",
                "source_graph_sha256",
                "target_graph_sha256",
                "action",
            )
            if any(existing.get(field) != event.get(field) for field in comparable_fields):
                raise ValueError(
                    "Ambiguous COMRECGC predecessor events target the same graph."
                )
            if (
                existing.get("action_recovery") != "recorded_exact"
                and event.get("action_recovery") == "recorded_exact"
            ):
                predecessor[target_key] = event
        else:
            predecessor[target_key] = event
        observed_source_keys.add(source_key)

    return {
        "graph_by_official_key": graph_by_official_key,
        "official_matches": official_matches,
        "frozen_sources": frozen_sources,
        "predecessor": predecessor,
        "observed_source_keys": observed_source_keys,
        "source_graphs_required": source_graphs_by_parent_id is not None,
    }


def iter_candidate_lineage_from_selected_trace(
    payload: Mapping[str, Any],
    selected_events: Any,
    *,
    source_graphs_by_parent_id: Mapping[str, Any] | None = None,
    include_actions: bool = True,
) -> Any:
    """Yield candidate paths one at a time from one compact predecessor index."""

    context = _lineage_recovery_context(
        payload, selected_events, source_graphs_by_parent_id
    )
    graph_by_official_key = context["graph_by_official_key"]
    official_matches = context["official_matches"]
    frozen_sources = context["frozen_sources"]
    predecessor = context["predecessor"]
    observed_source_keys = context["observed_source_keys"]
    source_graphs_required = bool(context["source_graphs_required"])
    for candidate_index, candidate in enumerate(
        payload.get("counterfactual_candidates") or []
    ):
        official_hash = str(candidate.get("graph_hash"))
        candidate_matches = official_matches.get(official_hash, [])
        if len(candidate_matches) != 1:
            raise ValueError(
                "Candidate official graph identity is absent or ambiguous during trace recovery: "
                f"{official_hash}."
            )
        (candidate_key, graph) = candidate_matches[0]
        if graph is None:
            raise ValueError(f"Candidate graph is absent during trace recovery: {official_hash}")
        parent_id = str(getattr(graph, "comrecgc_parent_id", ""))
        candidate_sha = stable_untyped_graph_sha256(graph)
        current_key = candidate_key
        reversed_path: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        while current_key in predecessor:
            if current_key in seen:
                raise ValueError("Recovered COMRECGC predecessor graph contains a cycle.")
            seen.add(current_key)
            event = predecessor[current_key]
            reversed_path.append(event)
            current_key = (parent_id, str(event["source_official_hash"]))
        path = list(reversed(reversed_path))
        root_graph = graph_by_official_key.get(current_key)
        observed_root = current_key in observed_source_keys
        frozen_source = frozen_sources.get(parent_id)
        frozen_source_exact = bool(
            root_graph is not None
            and frozen_source is not None
            and normalized_untyped_graph_payload(root_graph)
            == normalized_untyped_graph_payload(frozen_source)
        )
        resolved = bool(
            root_graph is not None
            and (
                frozen_source_exact
                if source_graphs_required
                else observed_root
            )
        )
        enriched_path: list[dict[str, Any]] = []
        node_ids = trace_node_ids(root_graph) if root_graph is not None else []
        for path_index, event in enumerate(path):
            enriched = dict(event)
            source_node_ids = list(node_ids)
            action = list(event["action"])
            target_node_ids = list(source_node_ids)
            if str(action[0]) in {"NA", "INA"}:
                target_node_ids.append(
                    f"new:{parent_id}:move:{int(event['move_index'])}:"
                    f"head:{int(event['head_index'])}:path:{path_index}:"
                    f"target:{event['target_graph_sha256']}"
                )
            elif str(action[0]) in {"NR", "INR"}:
                removed = int(action[1])
                if not 0 <= removed < len(target_node_ids):
                    resolved = False
                else:
                    target_node_ids.pop(removed)
            enriched["source_node_ids"] = source_node_ids
            enriched["target_node_ids"] = target_node_ids
            enriched_path.append(enriched)
            node_ids = target_node_ids
        if len(node_ids) != int(graph.num_nodes):
            resolved = False
        yield {
            "candidate_index": candidate_index,
            "official_graph_hash": official_hash,
            "stable_graph_sha256": candidate_sha,
            "parent_id": parent_id,
            "action_lineage_resolved": resolved,
            "zero_action_source_root": bool(not path and frozen_source_exact),
            "lineage_root_status": (
                "frozen_source_graph_exact_zero_action"
                if not path and frozen_source_exact
                else "frozen_source_graph_exact"
                if frozen_source_exact
                else "observed_trace_source"
                if observed_root
                else "unresolved"
            ),
            "action_count": len(enriched_path),
            "lineage_storage": (
                "inline_actions" if include_actions else "selected_trace_predecessor_index"
            ),
            "actions": enriched_path if include_actions else [],
        }


def recover_candidate_lineage_from_selected_trace(
    payload: Mapping[str, Any],
    selected_events: Any,
    *,
    source_graphs_by_parent_id: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recover complete candidate paths from streamed source/target graph deltas."""

    return list(
        iter_candidate_lineage_from_selected_trace(
            payload,
            selected_events,
            source_graphs_by_parent_id=source_graphs_by_parent_id,
            include_actions=True,
        )
    )


def _importance(value: Any) -> list[float]:
    if value is None:
        return []
    resolved = _plain(value)
    if resolved is None:
        return []
    if not isinstance(resolved, list):
        resolved = [resolved]
    return [float(item) for item in resolved]


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
                "candidate_id": stable_untyped_graph_sha256(graph_entry[0]),
                "stable_graph_sha256": stable_untyped_graph_sha256(graph_entry[0]),
                "frequency": int(candidate.get("frequency", 0)),
                "importance_parts": _importance(candidate.get("importance_parts")),
            }
        )
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _derived_model_cf_ids(
    rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> tuple[list[bool], list[str]]:
    mask: list[bool] = []
    identifiers: list[str] = []
    for row in rows:
        importance = list(row["importance_parts"])
        selected = bool(importance and float(importance[0]) >= float(threshold))
        mask.append(selected)
        if selected:
            identifiers.append(str(row["candidate_id"]))
    return mask, identifiers


def assert_trace_parity(
    reference_payload: Mapping[str, Any],
    traced_payload: Mapping[str, Any],
    *,
    importance_threshold: float = MODEL_CF_IMPORTANCE_THRESHOLD,
    reference_model_cf_ids: Sequence[str] | None = None,
    traced_model_cf_ids: Sequence[str] | None = None,
    reference_dbscan_input_ids: Sequence[str] | None = None,
    traced_dbscan_input_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    reference = normalized_candidate_sequence(reference_payload)
    traced = normalized_candidate_sequence(traced_payload)
    if len(reference) != len(traced):
        raise ValueError(
            "Trace-on/off candidate count differs: "
            f"reference={len(reference)}, traced={len(traced)}."
        )

    exact_importance_mismatch_count = 0
    absolute_differences: list[float] = []
    relative_differences: list[float] = []
    for index, (reference_row, traced_row) in enumerate(
        zip(reference, traced, strict=True)
    ):
        for field in ("stable_graph_sha256", "frequency"):
            if reference_row[field] != traced_row[field]:
                raise ValueError(
                    "Trace-on/off candidate topology, features, frequency, or order "
                    f"differs at candidate {index}, field={field}."
                )
        reference_importance = reference_row["importance_parts"]
        traced_importance = traced_row["importance_parts"]
        if len(reference_importance) != len(traced_importance):
            raise ValueError(
                "Trace-on/off importance shape differs at candidate "
                f"{index}: reference={len(reference_importance)}, "
                f"traced={len(traced_importance)}."
            )
        for part_index, (reference_value, traced_value) in enumerate(
            zip(reference_importance, traced_importance, strict=True)
        ):
            if not math.isfinite(reference_value) or not math.isfinite(traced_value):
                raise ValueError(
                    "Trace-on/off importance contains NaN/Inf at candidate "
                    f"{index}, part={part_index}."
                )
            difference = abs(reference_value - traced_value)
            absolute_differences.append(difference)
            denominator = max(abs(reference_value), abs(traced_value), 1e-12)
            relative_differences.append(difference / denominator)
            if difference != 0.0:
                exact_importance_mismatch_count += 1
            if difference > TRACE_IMPORTANCE_ABS_TOLERANCE:
                raise ValueError(
                    "Trace-on/off importance exceeds the audited CUDA float32 "
                    f"replay tolerance at candidate {index}, part={part_index}: "
                    f"abs_difference={difference}, "
                    f"tolerance={TRACE_IMPORTANCE_ABS_TOLERANCE}."
                )

    reference_mask, derived_reference_model_ids = _derived_model_cf_ids(
        reference, threshold=importance_threshold
    )
    traced_mask, derived_traced_model_ids = _derived_model_cf_ids(
        traced, threshold=importance_threshold
    )
    importance_threshold_mask_exact = reference_mask == traced_mask
    resolved_reference_model_ids = list(
        derived_reference_model_ids
        if reference_model_cf_ids is None
        else map(str, reference_model_cf_ids)
    )
    resolved_traced_model_ids = list(
        derived_traced_model_ids
        if traced_model_cf_ids is None
        else map(str, traced_model_cf_ids)
    )
    model_cf_id_set_exact = set(resolved_reference_model_ids) == set(
        resolved_traced_model_ids
    )
    model_cf_order_exact = resolved_reference_model_ids == resolved_traced_model_ids
    resolved_reference_dbscan_ids = list(
        resolved_reference_model_ids
        if reference_dbscan_input_ids is None
        else map(str, reference_dbscan_input_ids)
    )
    resolved_traced_dbscan_ids = list(
        resolved_traced_model_ids
        if traced_dbscan_input_ids is None
        else map(str, traced_dbscan_input_ids)
    )
    dbscan_input_id_set_exact = set(resolved_reference_dbscan_ids) == set(
        resolved_traced_dbscan_ids
    )
    dbscan_input_order_exact = (
        resolved_reference_dbscan_ids == resolved_traced_dbscan_ids
    )
    discrete_checks = {
        "importance_threshold_mask_exact": importance_threshold_mask_exact,
        "model_cf_id_set_exact": model_cf_id_set_exact,
        "model_cf_order_exact": model_cf_order_exact,
        "dbscan_input_id_set_exact": dbscan_input_id_set_exact,
        "dbscan_input_order_exact": dbscan_input_order_exact,
    }
    failed_discrete = [key for key, value in discrete_checks.items() if not value]
    if failed_discrete:
        raise ValueError(
            "Trace-on/off discrete decision parity differs: "
            + ", ".join(failed_discrete)
        )

    max_importance_abs_difference = max(absolute_differences, default=0.0)
    max_importance_relative_difference = max(relative_differences, default=0.0)
    reference_threshold_distances = [
        abs(float(row["importance_parts"][0]) - float(importance_threshold))
        for row in reference
        if row["importance_parts"]
    ]

    structural_sequence = [
        {
            "stable_graph_sha256": row["stable_graph_sha256"],
            "frequency": row["frequency"],
        }
        for row in reference
    ]
    return {
        "trace_parity_passed": True,
        "graph_identity_mode": "official_untyped_node_adjacency_v1",
        "candidate_count": len(reference),
        "candidate_sequence_sha256": stable_json_sha256(reference),
        "traced_candidate_sequence_sha256": stable_json_sha256(traced),
        "structural_candidate_sequence_sha256": stable_json_sha256(
            structural_sequence
        ),
        "importance_comparison_policy": "float32_cuda_replay_abs_tolerance_v1",
        "importance_abs_tolerance": TRACE_IMPORTANCE_ABS_TOLERANCE,
        "importance_exact_match": exact_importance_mismatch_count == 0,
        "importance_exact_mismatch_count": exact_importance_mismatch_count,
        "importance_max_abs_difference": max_importance_abs_difference,
        "max_abs_diff": max_importance_abs_difference,
        "max_relative_diff": max_importance_relative_difference,
        "mean_abs_diff": (
            sum(absolute_differences) / len(absolute_differences)
            if absolute_differences
            else 0.0
        ),
        "q99_abs_diff": _percentile(absolute_differences, 0.99),
        "num_diff_gt_1e_7": sum(value > 1e-7 for value in absolute_differences),
        "num_diff_gt_1e_6": sum(value > 1e-6 for value in absolute_differences),
        "minimum_distance_to_importance_threshold": min(
            reference_threshold_distances, default=None
        ),
        "num_values_within_1e_6_of_threshold": sum(
            value <= 1e-6 for value in reference_threshold_distances
        ),
        "importance_threshold": float(importance_threshold),
        **discrete_checks,
        "model_cf_count": len(resolved_reference_model_ids),
        "dbscan_input_count": len(resolved_reference_dbscan_ids),
        "compared_fields": [
            "stable_graph_sha256",
            "frequency",
            "importance_parts",
            "order",
            "importance_threshold_mask",
            "model_cf_ids",
            "dbscan_input_ids",
        ],
    }
