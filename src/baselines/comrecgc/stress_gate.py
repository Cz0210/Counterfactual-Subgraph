"""Scaled eviction stress gate for COMRECGC live graph state."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .graph_trace import stable_untyped_graph_sha256
from .live_graph_state import LiveGraphState


def _graph(index: int) -> Any:
    return SimpleNamespace(
        x=np.asarray([[index % 11], [(index + 1) % 11]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
    )


def run_transition_eviction_stress_gate(
    *, output_root: str | Path, steps: int = 2_048, cache_max_entries: int = 64
) -> dict[str, Any]:
    if steps <= cache_max_entries * 2:
        raise ValueError("Stress gate must exceed twice the hot cache bound.")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    module = SimpleNamespace(
        graph_map={},
        graph_index_map={},
        counterfactual_candidates=[],
        transitions={},
    )
    state = LiveGraphState(
        module,
        {},
        store_path=root / "stress_authoritative_graph_store.sqlite3",
        seed=0,
    )
    module.graph_map = state.graph_map
    module.comrecgc_live_graph_state = state
    active_order: deque[int] = deque()
    reference_sha: dict[int, str] = {}
    try:
        for step in range(steps):
            while active_order and active_order[0] not in module.graph_map:
                active_order.popleft()
            if len(module.graph_map) >= cache_max_entries:
                victim = active_order.popleft()
                module.graph_index_map.pop(victim, None)
                del module.graph_map[victim]

            graph = _graph(step)
            entry = [
                graph,
                np.asarray([float(step), float(step + 1)], dtype=np.float64),
                np.asarray([2.0, 2.0], dtype=np.float64),
            ]
            module.graph_map[step] = entry
            module.graph_index_map[step] = step
            module.transitions[step] = ([step], [graph], [[0.5, 1.0]], [entry[1]])
            active_order.append(step)
            reference_sha[step] = stable_untyped_graph_sha256(graph)

            if step > 0 and step % 17 == 0 and active_order:
                pinned_victim = active_order.popleft()
                with state.pin_many([pinned_victim]):
                    module.graph_index_map.pop(pinned_victim, None)
                    del module.graph_map[pinned_victim]
                    assert (
                        stable_untyped_graph_sha256(state.resolve_graph(pinned_victim))
                        == reference_sha[pinned_victim]
                    )
            if step > cache_max_entries and step % 31 == 0:
                historical = step - cache_max_entries
                assert (
                    stable_untyped_graph_sha256(state.resolve_graph(historical))
                    == reference_sha[historical]
                )

        parity = all(
            stable_untyped_graph_sha256(state.resolve_graph(key)) == expected
            for key, expected in reference_sha.items()
        )
        audit = state.audit()
        result = {
            "schema_version": "comrecgc_transition_eviction_stress_gate_v1",
            "completed": True,
            "steps": steps,
            "cache_max_entries": cache_max_entries,
            "hot_cache_size": len(module.graph_map),
            "max_hot_cache_size": audit["max_hot_cache_size"],
            "backing_store_size": audit["backing_store_size"],
            "unresolved_lookups": audit["unresolved_lookups"],
            "active_eviction_prevented": audit["active_eviction_prevented"],
            "eviction_committed": audit["eviction_committed"],
            "eviction_deferred": audit["eviction_deferred"],
            "deferred_flushed": audit["deferred_flushed"],
            "deferred_queue_empty": audit["deferred_deletions"] == 0,
            "all_live_hashes_resolvable": all(
                state.contains(key) for key in state.graph_map.live_reference_hashes()
            ),
            "result_parity_with_unbounded_reference": parity,
            "cache_bound_respected": audit["max_hot_cache_size"] <= cache_max_entries,
            "backing_store_checksum_pass": audit["backing_store"]["integrity_passed"],
            "transition_source_resolution_pass": audit[
                "unresolved_transition_source_count"
            ]
            == 0,
            "transition_destination_integrity_pass": audit[
                "invalid_transition_destination_count"
            ]
            == 0,
            "audit": audit,
        }
        result["stress_gate_passed"] = all(
            (
                result["unresolved_lookups"] == 0,
                result["active_eviction_prevented"] > 0,
                result["eviction_committed"] > 0,
                result["deferred_flushed"] > 0,
                result["deferred_queue_empty"],
                result["all_live_hashes_resolvable"],
                result["result_parity_with_unbounded_reference"],
                result["cache_bound_respected"],
                result["backing_store_checksum_pass"],
                result["transition_source_resolution_pass"],
                result["transition_destination_integrity_pass"],
            )
        )
        return result
    finally:
        state.close()
