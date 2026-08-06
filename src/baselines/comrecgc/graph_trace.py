"""Stable graph identities and side-effect-free COMRECGC action tracing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import atomic_write_bytes, sha256_file, stable_json_sha256, write_json


TRACE_IMPORTANCE_ABS_TOLERANCE = 1e-6


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
    """Record selected actions without retaining a second in-memory walk history.

    The official runtime still owns its graph state.  This recorder keeps only
    the first predecessor needed to reconstruct candidate lineages and streams
    audit events to deterministic chunks.  A resumed deterministic replay
    reuses byte-identical completed chunks instead of appending duplicates.
    """

    output_dir: str | Path | None = None
    chunk_size: int = 512
    enumerated: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    predecessor_by_official_hash: dict[str, dict[str, Any]] = field(default_factory=dict)
    move_index: int = 0
    enumerated_transition_count: int = 0
    selected_transition_count: int = 0
    teleport_count: int = 0
    _trace_root: Path | None = field(default=None, init=False, repr=False)
    _pending_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _chunks: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

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

    def record_enumerated(
        self,
        *,
        source_graph: Any,
        target_graph: Any,
        action: Sequence[Any],
    ) -> None:
        source_sha = stable_graph_sha256(source_graph)
        target_sha = stable_graph_sha256(target_graph)
        record = {
            "source_graph_sha256": source_sha,
            "target_graph_sha256": target_sha,
            "action": normalized_action(action),
        }
        self.enumerated.setdefault((source_sha, target_sha), []).append(record)
        self.enumerated_transition_count += 1

    def wrap_move(self, original: Callable[..., Any], module: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            graphs_hash = list(kwargs.get("graphs_hash", args[0] if args else []))
            source_graphs = [module.graph_map[value][0] for value in graphs_hash]
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
                consumed_sources = {stable_graph_sha256(graph) for graph in source_graphs}
                self._discard_enumerated_sources(consumed_sources)
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
                    "parent_id": str(getattr(target_graph, "comrecgc_parent_id", "")),
                }
                self._stream_event(event)
                if action_record is not None:
                    self.predecessor_by_official_hash.setdefault(str(target_hash), event)
            consumed_sources = {stable_graph_sha256(graph) for graph in source_graphs}
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
                    "stable_graph_sha256": stable_graph_sha256(graph) if graph is not None else None,
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

    def write(self, output_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
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
                "chunk_size": int(self.chunk_size),
                "row_count": self.selected_transition_count + self.teleport_count,
                "chunks": self._chunks,
                "resume_policy": "reuse_byte_identical_completed_chunks",
            },
        )
        lineage = self.candidate_lineage(payload)
        lineage_path = root / "candidate_action_lineage.json"
        write_json(lineage_path, lineage)
        summary = {
            "trace_schema_version": 1,
            "trace_only": True,
            "rng_calls_added": 0,
            "enumerated_transition_count": self.enumerated_transition_count,
            "live_enumerated_transition_pair_count": len(self.enumerated),
            "selected_transition_count": self.selected_transition_count,
            "teleport_count": self.teleport_count,
            "candidate_count": len(lineage),
            "candidate_lineage_resolved_count": sum(
                bool(row["action_lineage_resolved"]) for row in lineage
            ),
            "selected_trace_path": str(selected_manifest_path),
            "selected_trace_chunk_count": len(self._chunks),
            "max_buffered_event_count": int(self.chunk_size),
            "candidate_lineage_path": str(lineage_path),
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


def load_selected_trace(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Reload a completed chunked trace in exact walk order."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "chunked_jsonl":
        raise ValueError(f"Unsupported COMRECGC selected trace format: {manifest.get('format')!r}")
    rows: list[dict[str, Any]] = []
    for expected_index, chunk in enumerate(manifest.get("chunks") or []):
        if int(chunk.get("index", -1)) != expected_index:
            raise ValueError("COMRECGC selected trace chunks are not contiguous.")
        chunk_path = path.parent / str(chunk["path"])
        if sha256_file(chunk_path) != str(chunk["sha256"]):
            raise ValueError(f"COMRECGC selected trace chunk SHA256 mismatch: {chunk_path}")
        chunk_rows = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(chunk_rows) != int(chunk["row_count"]):
            raise ValueError(f"COMRECGC selected trace chunk row count mismatch: {chunk_path}")
        rows.extend(chunk_rows)
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError("COMRECGC selected trace total row count mismatch.")
    return rows


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
    if len(reference) != len(traced):
        raise ValueError(
            "Trace-on/off candidate count differs: "
            f"reference={len(reference)}, traced={len(traced)}."
        )

    exact_importance_mismatch_count = 0
    max_importance_abs_difference = 0.0
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
            max_importance_abs_difference = max(
                max_importance_abs_difference, difference
            )
            if difference != 0.0:
                exact_importance_mismatch_count += 1
            if difference > TRACE_IMPORTANCE_ABS_TOLERANCE:
                raise ValueError(
                    "Trace-on/off importance exceeds the audited CUDA float32 "
                    f"replay tolerance at candidate {index}, part={part_index}: "
                    f"abs_difference={difference}, "
                    f"tolerance={TRACE_IMPORTANCE_ABS_TOLERANCE}."
                )

    structural_sequence = [
        {
            "stable_graph_sha256": row["stable_graph_sha256"],
            "frequency": row["frequency"],
        }
        for row in reference
    ]
    return {
        "trace_parity_passed": True,
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
        "compared_fields": [
            "stable_graph_sha256",
            "frequency",
            "importance_parts",
            "order",
        ],
    }
