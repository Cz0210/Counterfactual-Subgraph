"""Referentially complete COMRECGC payload freezing and verification."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from .contracts import sha256_file, stable_json_sha256
from .graph_trace import stable_graph_sha256, stable_untyped_graph_sha256
from .live_graph_state import AuthoritativeGraphStore


FROZEN_PAYLOAD_CLOSURE_POLICY = "selected_trace_and_runtime_reference_closure_v3"


class ComRecGCFrozenPayloadClosureError(RuntimeError):
    """Raised when a frozen payload cannot resolve every required graph."""

    def __init__(self, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "[COMRECGC_FROZEN_PAYLOAD_CLOSURE_ERROR] "
            + str(self.diagnostics)
        )


def _graph_from_entry(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError("COMRECGC graph_map entry must contain a graph at index 0.")
    return value[0]


def _recorded_sha_matches(graph: Any, expected: str) -> bool:
    if stable_untyped_graph_sha256(graph) == str(expected):
        return True
    try:
        return stable_graph_sha256(graph) == str(expected)
    except ValueError:
        return False


def _iter_nested_hashes(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_nested_hashes(item)
    elif value is not None:
        yield value


def payload_graphs_by_official_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return every graph frozen in the active map or graph-only closure."""

    result: dict[str, Any] = {}
    for key, entry in (payload.get("graph_map") or {}).items():
        result[str(key)] = _graph_from_entry(entry)
    for key, graph in (payload.get("frozen_graph_closure") or {}).items():
        key_string = str(key)
        existing = result.get(key_string)
        if existing is not None and (
            stable_untyped_graph_sha256(existing)
            != stable_untyped_graph_sha256(graph)
        ):
            raise ComRecGCFrozenPayloadClosureError(
                {
                    "reason": "official_hash_collision_between_payload_sections",
                    "graph_hash": key_string,
                }
            )
        result[key_string] = graph
    return result


def _collect_requirements(
    payload: Mapping[str, Any], selected_events: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Counter[str]]:
    required: dict[str, dict[str, Any]] = {}
    inline_transition_graphs: dict[str, Any] = {}
    kind_counts: Counter[str] = Counter()

    def add(value: Any, kind: str, expected_sha: str | None = None) -> None:
        key = str(value)
        row = required.setdefault(
            key,
            {"official_hash": key, "kinds": set(), "expected_sha256": set()},
        )
        if kind not in row["kinds"]:
            kind_counts[kind] += 1
        row["kinds"].add(kind)
        if expected_sha:
            row["expected_sha256"].add(str(expected_sha))

    for candidate in payload.get("counterfactual_candidates") or []:
        if isinstance(candidate, Mapping) and candidate.get("graph_hash") is not None:
            add(candidate["graph_hash"], "recourse_candidate")

    for field, kind in (
        ("traversed_hashes", "traversed_state"),
        ("current_graph_hashes", "current_state"),
        ("graphs_hash", "current_state"),
        ("frontier_hashes", "final_frontier"),
        ("selected_summary_hashes", "selected_summary"),
    ):
        for value in _iter_nested_hashes(payload.get(field)):
            add(value, kind)

    transitions = payload.get("transitions") or {}
    if isinstance(transitions, Mapping):
        for source_hash, transition in transitions.items():
            add(source_hash, "transition_source")
            if not isinstance(transition, tuple) or len(transition) < 2:
                raise ComRecGCFrozenPayloadClosureError(
                    {
                        "reason": "malformed_transition_entry",
                        "source_hash": str(source_hash),
                    }
                )
            target_hashes, target_graphs = transition[0], transition[1]
            if len(target_hashes) != len(target_graphs):
                raise ComRecGCFrozenPayloadClosureError(
                    {
                        "reason": "transition_hash_graph_length_mismatch",
                        "source_hash": str(source_hash),
                        "hash_count": len(target_hashes),
                        "graph_count": len(target_graphs),
                    }
                )
            for target_hash, target_graph in zip(
                target_hashes, target_graphs, strict=True
            ):
                key = str(target_hash)
                add(target_hash, "transition_destination")
                existing = inline_transition_graphs.get(key)
                if existing is not None and (
                    stable_untyped_graph_sha256(existing)
                    != stable_untyped_graph_sha256(target_graph)
                ):
                    raise ComRecGCFrozenPayloadClosureError(
                        {
                            "reason": "transition_destination_hash_collision",
                            "graph_hash": key,
                        }
                    )
                inline_transition_graphs[key] = target_graph

    for event in selected_events:
        if event.get("event") != "selected_transition":
            continue
        add(
            event["source_official_hash"],
            "selected_trace_source",
            str(event["source_graph_sha256"]),
        )
        add(
            event["target_official_hash"],
            "selected_trace_target",
            str(event["target_graph_sha256"]),
        )

    return required, inline_transition_graphs, kind_counts


def materialize_frozen_payload_closure(
    payload: Mapping[str, Any],
    selected_events: Iterable[Mapping[str, Any]],
    *,
    backing_store_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rehydrate every required graph without altering candidate semantics."""

    graph_map = dict(payload.get("graph_map") or {})
    if not graph_map:
        raise ComRecGCFrozenPayloadClosureError({"reason": "missing_graph_map"})
    frozen_graph_closure = dict(payload.get("frozen_graph_closure") or {})
    hot_key_index = {str(key): key for key in graph_map}
    closure_key_index = {str(key): key for key in frozen_graph_closure}
    if len(hot_key_index) != len(graph_map) or len(closure_key_index) != len(
        frozen_graph_closure
    ):
        raise ComRecGCFrozenPayloadClosureError(
            {"reason": "ambiguous_stringified_official_hash"}
        )

    required, inline_graphs, kind_counts = _collect_requirements(
        payload, selected_events
    )
    store: AuthoritativeGraphStore | None = None
    store_key_index: dict[str, Any] = {}
    store_audit: dict[str, Any] | None = None
    if backing_store_path is not None:
        store_path = Path(backing_store_path).expanduser().resolve()
        if not store_path.is_file():
            raise FileNotFoundError(f"COMRECGC authoritative graph store missing: {store_path}")
        store = AuthoritativeGraphStore(store_path)
        stored_keys = store.stored_keys()
        store_key_index = {str(key): key for key in stored_keys}
        if len(store_key_index) != len(stored_keys):
            store.close()
            raise ComRecGCFrozenPayloadClosureError(
                {"reason": "ambiguous_backing_store_official_hash"}
            )

    resolved_from_hot = 0
    resolved_from_backing = 0
    resolved_from_frozen = 0
    resolved_from_transition = 0
    unresolved: list[dict[str, Any]] = []
    sha_mismatches: list[dict[str, Any]] = []
    try:
        if store is not None:
            store_audit = store.integrity_audit()
            if not store_audit["integrity_passed"]:
                raise ComRecGCFrozenPayloadClosureError(
                    {"reason": "backing_store_integrity_failed", "audit": store_audit}
                )
        for key_string, requirement in required.items():
            graph: Any | None = None
            if key_string in hot_key_index:
                graph = _graph_from_entry(graph_map[hot_key_index[key_string]])
                resolved_from_hot += 1
            elif key_string in closure_key_index:
                graph = frozen_graph_closure[closure_key_index[key_string]]
                resolved_from_frozen += 1
            elif store is not None and key_string in store_key_index:
                store_key = store_key_index[key_string]
                entry = store.get(store_key)
                graph_map[store_key] = entry
                hot_key_index[key_string] = store_key
                graph = _graph_from_entry(entry)
                resolved_from_backing += 1
            elif key_string in inline_graphs:
                graph = inline_graphs[key_string]
                frozen_graph_closure[key_string] = graph
                closure_key_index[key_string] = key_string
                resolved_from_transition += 1
            else:
                unresolved.append(
                    {
                        "graph_hash": key_string,
                        "kinds": sorted(requirement["kinds"]),
                    }
                )
                continue
            expected_values = set(requirement["expected_sha256"])
            if expected_values and not all(
                _recorded_sha_matches(graph, value) for value in expected_values
            ):
                sha_mismatches.append(
                    {
                        "graph_hash": key_string,
                        "kinds": sorted(requirement["kinds"]),
                        "expected_sha256": sorted(expected_values),
                        "actual_untyped_sha256": stable_untyped_graph_sha256(graph),
                    }
                )
    finally:
        if store is not None:
            store.close()

    serialized_requirements = [
        {
            "official_hash": key,
            "kinds": sorted(value["kinds"]),
            "expected_sha256": sorted(value["expected_sha256"]),
        }
        for key, value in sorted(required.items())
    ]
    audit = {
        "schema_version": "comrecgc_frozen_payload_closure_v3",
        "policy": FROZEN_PAYLOAD_CLOSURE_POLICY,
        "required_hash_count": len(required),
        "resolved_from_hot_cache": resolved_from_hot,
        "resolved_from_backing_store": resolved_from_backing,
        "resolved_from_existing_frozen_closure": resolved_from_frozen,
        "resolved_from_inline_transition": resolved_from_transition,
        "unresolved_hash_count": len(unresolved),
        "unresolved_hashes": unresolved[:100],
        "sha_mismatch_count": len(sha_mismatches),
        "sha_mismatches": sha_mismatches[:100],
        "selected_trace_hash_count": sum(
            1
            for row in required.values()
            if {"selected_trace_source", "selected_trace_target"} & row["kinds"]
        ),
        "transition_hash_count": sum(
            1
            for row in required.values()
            if {"transition_source", "transition_destination"} & row["kinds"]
        ),
        "payload_graph_count": len(graph_map) + len(frozen_graph_closure),
        "graph_map_count": len(graph_map),
        "graph_only_closure_count": len(frozen_graph_closure),
        "requirement_kind_counts": dict(sorted(kind_counts.items())),
        "requirements_sha256": stable_json_sha256(serialized_requirements),
        "backing_store": store_audit,
        "closure_complete": not unresolved and not sha_mismatches,
        "candidate_order_changed": False,
        "candidate_payload_changed": False,
        "scientific_parameters_changed": False,
    }
    if not audit["closure_complete"]:
        raise ComRecGCFrozenPayloadClosureError(audit)

    frozen = dict(payload)
    frozen["graph_map"] = graph_map
    frozen["frozen_graph_closure"] = frozen_graph_closure
    frozen["frozen_payload_closure"] = {
        key: value
        for key, value in audit.items()
        if key not in {"unresolved_hashes", "sha_mismatches"}
    }
    return frozen, audit


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("COMRECGC payload freezing requires PyTorch.") from exc
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def torch_load_payload(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("COMRECGC payload loading requires PyTorch.") from exc
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older pinned torch
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("COMRECGC frozen counterfactual payload must be a dictionary.")
    return payload


def payload_file_audit(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        "payload_path": str(source),
        "payload_bytes": source.stat().st_size,
        "payload_checksum": sha256_file(source),
    }
