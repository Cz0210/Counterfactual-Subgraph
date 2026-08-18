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


FROZEN_PAYLOAD_CLOSURE_POLICY = "canonical_graph_alias_roundtrip_closure_v7"


class FrozenPayloadClosureError(RuntimeError):
    """Raised when a frozen payload cannot resolve every required graph."""

    def __init__(self, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "[COMRECGC_FROZEN_PAYLOAD_CLOSURE_ERROR] "
            + str(self.diagnostics)
        )


# Keep the published v6 exception name import-compatible while exposing the
# method-neutral name used by validator and recovery in v7.
ComRecGCFrozenPayloadClosureError = FrozenPayloadClosureError


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


def _follow_alias(alias_to_canonical: Mapping[str, str], graph_hash: str) -> str:
    current = str(graph_hash)
    seen: set[str] = set()
    while current in alias_to_canonical:
        if current in seen:
            raise ComRecGCFrozenPayloadClosureError(
                {"reason": "graph_alias_cycle", "graph_hash": str(graph_hash)}
            )
        seen.add(current)
        current = str(alias_to_canonical[current])
    return current


def payload_graphs_by_official_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical graphs plus every persisted official-hash alias."""

    result: dict[str, Any] = {}
    for key, entry in (payload.get("graph_map") or {}).items():
        result[str(key)] = _graph_from_entry(entry)
    closure_sections = (
        payload.get("frozen_graph_closure") or {},
        payload.get("canonical_graph_records") or {},
    )
    for section in closure_sections:
        for key, graph in section.items():
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
    aliases = {
        str(alias): str(canonical)
        for alias, canonical in (payload.get("alias_to_canonical") or {}).items()
    }
    for alias in sorted(aliases):
        canonical = _follow_alias(aliases, alias)
        graph = result.get(canonical)
        if graph is None:
            raise ComRecGCFrozenPayloadClosureError(
                {
                    "reason": "graph_alias_target_absent",
                    "graph_hash": alias,
                    "canonical_hash": canonical,
                }
            )
        existing = result.get(alias)
        if existing is not None and (
            stable_untyped_graph_sha256(existing)
            != stable_untyped_graph_sha256(graph)
        ):
            raise ComRecGCFrozenPayloadClosureError(
                {
                    "reason": "graph_alias_collision",
                    "graph_hash": alias,
                    "canonical_hash": canonical,
                }
            )
        result[alias] = graph
    return result


def _collect_requirements(
    payload: Mapping[str, Any], selected_events: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Counter[str]]:
    required: dict[str, dict[str, Any]] = {}
    inline_transition_graphs: dict[str, Any] = {}
    kind_counts: Counter[str] = Counter()

    def add(
        value: Any,
        kind: str,
        expected_sha: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        key = str(value)
        row = required.setdefault(
            key,
            {
                "official_hash": key,
                "kinds": set(),
                "expected_sha256": set(),
                "parent_ids": set(),
            },
        )
        if kind not in row["kinds"]:
            kind_counts[kind] += 1
        row["kinds"].add(kind)
        if expected_sha:
            row["expected_sha256"].add(str(expected_sha))
        if parent_id:
            row["parent_ids"].add(str(parent_id))

    # A reloaded frozen payload must audit exactly the same closure as the
    # payload that was written.  Persisting the complete requirement rows also
    # protects hashes referenced by provenance sections that an older runtime
    # may no longer expose as a top-level field.
    for persisted in payload.get("frozen_payload_closure_requirements") or []:
        if not isinstance(persisted, Mapping) or persisted.get("official_hash") is None:
            raise FrozenPayloadClosureError(
                {"reason": "malformed_persisted_closure_requirement"}
            )
        kinds = list(persisted.get("kinds") or [])
        if not kinds:
            raise FrozenPayloadClosureError(
                {
                    "reason": "persisted_closure_requirement_has_no_kind",
                    "graph_hash": str(persisted["official_hash"]),
                }
            )
        expected = [str(value) for value in persisted.get("expected_sha256") or []]
        parents = [str(value) for value in persisted.get("parent_ids") or []]
        for kind in kinds:
            add(persisted["official_hash"], str(kind))
        row = required[str(persisted["official_hash"])]
        row["expected_sha256"].update(expected)
        row["parent_ids"].update(parents)

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
            str(event.get("parent_id") or "") or None,
        )
        add(
            event["target_official_hash"],
            "selected_trace_target",
            str(event["target_graph_sha256"]),
            str(event.get("parent_id") or "") or None,
        )

    return required, inline_transition_graphs, kind_counts


def build_frozen_payload_closure(
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
    canonical_graph_records = dict(payload.get("canonical_graph_records") or {})
    alias_to_canonical = {
        str(alias): str(canonical)
        for alias, canonical in (payload.get("alias_to_canonical") or {}).items()
    }
    hot_key_index = {str(key): key for key in graph_map}
    closure_key_index = {str(key): key for key in frozen_graph_closure}
    canonical_key_index = {str(key): key for key in canonical_graph_records}
    for index, values in (
        (hot_key_index, graph_map),
        (closure_key_index, frozen_graph_closure),
        (canonical_key_index, canonical_graph_records),
    ):
        if len(index) != len(values):
            raise ComRecGCFrozenPayloadClosureError(
                {"reason": "ambiguous_stringified_official_hash"}
            )

    required, inline_graphs, kind_counts = _collect_requirements(
        payload, selected_events
    )
    original_trace_hashes = sorted(
        key
        for key, row in required.items()
        if {"selected_trace_source", "selected_trace_target"} & row["kinds"]
    )
    store: AuthoritativeGraphStore | None = None
    store_key_index: dict[str, Any] = {}
    store_audit: dict[str, Any] | None = None
    if backing_store_path is not None:
        store_path = Path(backing_store_path).expanduser().resolve()
        if not store_path.is_file():
            raise FileNotFoundError(
                f"COMRECGC authoritative graph store missing: {store_path}"
            )
        store = AuthoritativeGraphStore(store_path)
        stored_keys = store.stored_keys()
        store_key_index = {str(key): key for key in stored_keys}
        if len(store_key_index) != len(stored_keys):
            store.close()
            raise ComRecGCFrozenPayloadClosureError(
                {"reason": "ambiguous_backing_store_official_hash"}
            )

    fingerprint_index: dict[str, list[tuple[str, Any, str]]] = {}

    def index_graph(key: str, graph: Any, source: str) -> None:
        fingerprints = {stable_untyped_graph_sha256(graph)}
        try:
            fingerprints.add(stable_graph_sha256(graph))
        except ValueError:
            pass
        for fingerprint in fingerprints:
            fingerprint_index.setdefault(fingerprint, []).append(
                (str(key), graph, source)
            )

    for key, entry in graph_map.items():
        index_graph(str(key), _graph_from_entry(entry), "hot")
    for key, graph in frozen_graph_closure.items():
        index_graph(str(key), graph, "frozen")
    for key, graph in canonical_graph_records.items():
        index_graph(str(key), graph, "canonical")
    for key, graph in inline_graphs.items():
        index_graph(str(key), graph, "transition")

    resolved_from_hot = 0
    resolved_from_backing = 0
    resolved_from_frozen = 0
    resolved_from_transition = 0
    resolved_from_canonical = 0
    resolved_by_alias = 0
    unresolved: list[dict[str, Any]] = []
    sha_mismatches: list[dict[str, Any]] = []

    def exact_graph(key_string: str) -> tuple[Any | None, str | None]:
        nonlocal resolved_from_hot, resolved_from_backing
        nonlocal resolved_from_frozen, resolved_from_transition
        nonlocal resolved_from_canonical
        if key_string in hot_key_index:
            resolved_from_hot += 1
            return _graph_from_entry(graph_map[hot_key_index[key_string]]), "hot"
        if key_string in closure_key_index:
            resolved_from_frozen += 1
            return frozen_graph_closure[closure_key_index[key_string]], "frozen"
        if key_string in canonical_key_index:
            resolved_from_canonical += 1
            return canonical_graph_records[canonical_key_index[key_string]], "canonical"
        if store is not None and key_string in store_key_index:
            resolved_from_backing += 1
            store_key = store_key_index[key_string]
            entry = store.get(store_key)
            graph_map[store_key] = entry
            hot_key_index[key_string] = store_key
            return _graph_from_entry(entry), "backing"
        if key_string in inline_graphs:
            resolved_from_transition += 1
            return inline_graphs[key_string], "transition"
        return None, None

    try:
        if store is not None:
            store_audit = store.integrity_audit()
            if not store_audit["integrity_passed"]:
                raise ComRecGCFrozenPayloadClosureError(
                    {"reason": "backing_store_integrity_failed", "audit": store_audit}
                )
        for key_string, requirement in required.items():
            expected_values = set(requirement["expected_sha256"])
            parent_ids = set(requirement["parent_ids"])
            canonical_key = _follow_alias(alias_to_canonical, key_string)
            graph, source = exact_graph(canonical_key)
            if graph is None and expected_values:
                matches: dict[str, tuple[Any, str]] = {}
                for expected in sorted(expected_values):
                    for candidate_key, candidate_graph, candidate_source in (
                        fingerprint_index.get(expected) or []
                    ):
                        graph_parent = str(
                            getattr(candidate_graph, "comrecgc_parent_id", "")
                        )
                        if parent_ids and graph_parent and graph_parent not in parent_ids:
                            continue
                        if all(
                            _recorded_sha_matches(candidate_graph, value)
                            for value in expected_values
                        ):
                            matches[candidate_key] = (candidate_graph, candidate_source)
                    if store is not None:
                        for store_key in store.find_keys_by_graph_sha256(expected):
                            candidate_key = str(store_key)
                            candidate_graph = _graph_from_entry(store.get(store_key))
                            graph_parent = str(
                                getattr(candidate_graph, "comrecgc_parent_id", "")
                            )
                            if parent_ids and graph_parent and graph_parent not in parent_ids:
                                continue
                            if all(
                                _recorded_sha_matches(candidate_graph, value)
                                for value in expected_values
                            ):
                                matches[candidate_key] = (candidate_graph, "backing")
                if matches:
                    graph_digests = {
                        stable_untyped_graph_sha256(value[0])
                        for value in matches.values()
                    }
                    if len(graph_digests) != 1:
                        raise ComRecGCFrozenPayloadClosureError(
                            {
                                "reason": "ambiguous_trace_fingerprint_alias",
                                "graph_hash": key_string,
                                "candidate_canonical_hashes": sorted(matches),
                            }
                        )
                    canonical_key = sorted(matches)[0]
                    graph, source = matches[canonical_key]
                    existing_alias = alias_to_canonical.get(key_string)
                    if existing_alias is not None and (
                        _follow_alias(alias_to_canonical, existing_alias)
                        != canonical_key
                    ):
                        raise ComRecGCFrozenPayloadClosureError(
                            {
                                "reason": "graph_alias_reassignment",
                                "graph_hash": key_string,
                                "old_canonical": existing_alias,
                                "new_canonical": canonical_key,
                            }
                        )
                    if key_string != canonical_key:
                        alias_to_canonical[key_string] = canonical_key
                        resolved_by_alias += 1
                    if source == "backing":
                        resolved_from_backing += 1
            if graph is None:
                unresolved.append(
                    {
                        "graph_hash": key_string,
                        "canonical_hash": canonical_key,
                        "kinds": sorted(requirement["kinds"]),
                        "expected_sha256": sorted(expected_values),
                        "parent_ids": sorted(parent_ids),
                    }
                )
                continue
            if expected_values and not all(
                _recorded_sha_matches(graph, value) for value in expected_values
            ):
                sha_mismatches.append(
                    {
                        "graph_hash": key_string,
                        "canonical_hash": canonical_key,
                        "kinds": sorted(requirement["kinds"]),
                        "expected_sha256": sorted(expected_values),
                        "actual_untyped_sha256": stable_untyped_graph_sha256(graph),
                    }
                )
                continue
            canonical_graph_records[str(canonical_key)] = graph
            canonical_key_index[str(canonical_key)] = str(canonical_key)
            if source in {"backing", "transition"}:
                frozen_graph_closure[str(canonical_key)] = graph
                closure_key_index[str(canonical_key)] = str(canonical_key)
    finally:
        if store is not None:
            store.close()

    alias_to_canonical = {
        alias: _follow_alias(alias_to_canonical, alias)
        for alias in sorted(alias_to_canonical)
    }

    serialized_requirements = [
        {
            "official_hash": key,
            "kinds": sorted(value["kinds"]),
            "expected_sha256": sorted(value["expected_sha256"]),
            "parent_ids": sorted(value["parent_ids"]),
        }
        for key, value in sorted(required.items())
    ]
    audit = {
        "schema_version": "comrecgc_frozen_payload_closure_v7",
        "policy": FROZEN_PAYLOAD_CLOSURE_POLICY,
        "required_hash_count": len(required),
        "resolved_from_hot_cache": resolved_from_hot,
        "resolved_from_backing_store": resolved_from_backing,
        "resolved_from_existing_frozen_closure": resolved_from_frozen,
        "resolved_from_inline_transition": resolved_from_transition,
        "resolved_from_existing_canonical_records": resolved_from_canonical,
        "resolved_by_fingerprint_alias": resolved_by_alias,
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
        "payload_graph_count": len(canonical_graph_records),
        "graph_map_count": len(graph_map),
        "graph_only_closure_count": len(frozen_graph_closure),
        "canonical_graph_record_count": len(canonical_graph_records),
        "alias_count": len(alias_to_canonical),
        "original_trace_hash_count": len(original_trace_hashes),
        "requirement_kind_counts": dict(sorted(kind_counts.items())),
        "requirements_sha256": stable_json_sha256(serialized_requirements),
        "required_hashes_sha256": stable_json_sha256(sorted(required)),
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
    frozen["canonical_graph_records"] = canonical_graph_records
    frozen["alias_to_canonical"] = alias_to_canonical
    frozen["original_trace_hashes"] = original_trace_hashes
    frozen["required_hashes"] = sorted(required)
    frozen["frozen_payload_closure_requirements"] = serialized_requirements
    frozen["frozen_payload_closure"] = {
        key: value
        for key, value in audit.items()
        if key not in {"unresolved_hashes", "sha_mismatches"}
    }
    return frozen, audit


def materialize_frozen_payload_closure(
    payload: Mapping[str, Any],
    selected_events: Iterable[Mapping[str, Any]],
    *,
    backing_store_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible name for the shared pure closure builder."""

    return build_frozen_payload_closure(
        payload,
        selected_events,
        backing_store_path=backing_store_path,
    )


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
