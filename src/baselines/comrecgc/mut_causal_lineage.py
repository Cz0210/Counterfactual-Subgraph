"""Opt-in Mut selected-action receipts, separate from optional debug tracing.

The pinned move has already selected and applied its lead/follower actions when
the observer runs. The existing compact transition cache keeps the deduplicated
target order and exact action at each retained target index. We read that
authority; we never infer a different action or enumerate alternatives.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import sha256_file, write_json
from .graph_trace import (
    ActionTraceRecorder,
    apply_action_to_normalized_payload,
    normalized_untyped_graph_payload,
)


CAUSAL_SCHEMA = "mut_observational_selected_action_causal_lineage_v1"
CAUSAL_CHECKPOINT_SCHEMA = "mut_observational_causal_checkpoint_v1"
_NO_ACTION_SUBSTITUTION_COUNTERS = (
    "recorded_action_replay_mismatch_count", "recorded_action_index_remap_count",
    "legacy_missing_action_count", "legacy_inference_called_count",
    "semantic_transition_lineage_replacement_count", "semantic_transition_excluded_count",
    "semantic_transition_failure_count", "predecessor_recorded_upgrade_count",
    "predecessor_unverified_conflict_count", "predecessor_unresolved_legacy_conflict_count",
)


class MutCausalLineageError(RuntimeError):
    """The actual selected action cannot be proved; never invent a lineage."""


def validate_causal_recovery_audit(summary: Mapping[str, Any]) -> None:
    """Legacy readers may recover historical actions; a new producer may not.

    Their compact serializer is reused, but any inferred action, index remap or
    lineage-friendly semantic alias replacement prevents the new causal seal.
    Later exact convergence remains visible under the existing deterministic
    first-recorded-predecessor policy; it does not substitute an action.
    """
    audit = summary.get("lineage_recovery_audit")
    if (not isinstance(audit, Mapping)
            or audit.get("schema_version") != "comrecgc_recorded_action_first_v3"
            or audit.get("predecessor_selection_policy") != "global_first_recorded_exact_event_in_selected_trace_order_v1"):
        raise MutCausalLineageError("causal closure requires the actual selected-event replay audit")
    for key in _NO_ACTION_SUBSTITUTION_COUNTERS:
        if type(audit.get(key)) is not int or audit[key] != 0:
            raise MutCausalLineageError(f"causal closure cannot infer or replace selected actions: {key}")
    selected = summary.get("selected_transition_count")
    if (type(selected) is not int or selected < 0
            or any(type(audit.get(key)) is not int or audit[key] != selected for key in (
                "selected_transition_count", "recorded_action_present_count",
                "recorded_action_replay_verified_count",
            ))):
        raise MutCausalLineageError("causal closure did not verify every selected action")


class MutCausalLineageRecorder(ActionTraceRecorder):
    """Small selected-event stream using the already-owned transition cache.

    No per-neighbor graph/object index is allocated by this observer. Chunked
    selected events and the existing compact candidate lineage index are the
    only persisted data. It is deliberately opt-in so healthy old runs and
    their checkpoint schemas are unchanged.
    """

    def __init__(self, output_dir: str | Path, *, chunk_size: int = 512) -> None:
        super().__init__(output_dir=output_dir, chunk_size=chunk_size,
                         compact_enumeration=True)

    def record_enumerated(
        self, *, source_graph: Any, target_graph: Any, action: Sequence[Any],
    ) -> None:
        # CompactMoveScopedTransitionMap already records this exact action.
        # Do not retain an additional weakref/object map or touch graph state.
        return None

    def _compact_transition_records(
        self, module: Any, *, source_hash: Any, target_hash: Any,
    ) -> list[dict[str, Any]]:
        from .live_graph_state import resolve_graph
        from .transition_cache import CompactMoveScopedTransitionMap

        transitions = getattr(module, "transitions", None)
        if not isinstance(transitions, CompactMoveScopedTransitionMap):
            raise MutCausalLineageError("causal capture requires the exact compact transition cache")
        # action_records reads the retained indexed actions without expanding
        # the cache, mutating its LRU, changing any set/dict order or drawing RNG.
        records = transitions.action_records(source_hash, target_hash)
        if len(records) != 1:
            raise MutCausalLineageError(
                "selected target must have exactly one retained action; "
                f"source={source_hash!s} target={target_hash!s} count={len(records)}"
            )
        source = resolve_graph(module, source_hash)
        target = resolve_graph(module, target_hash)
        action = deepcopy(records[0]["action"])
        replay = apply_action_to_normalized_payload(source, action)
        if replay != normalized_untyped_graph_payload(target):
            # This also rejects embedding-hash collisions between scientifically
            # different graphs. No heuristic representative is substituted.
            raise MutCausalLineageError("actual selected action does not exactly replay the stored target")
        return [{"action": action}]

    def _stream_event(self, event: dict[str, Any]) -> None:
        event = deepcopy(event)
        event["causal_schema"] = CAUSAL_SCHEMA
        event["capture_boundary"] = "after_official_move_state_update"
        if event.get("event") == "selected_transition":
            if event.get("action_resolution") != "exact" or event.get("action") is None:
                raise MutCausalLineageError("selected causal event is not exact")
            event["selected_action_source"] = "retained_official_dedup_target_index"
            event["action_replay_exact"] = True
        super()._stream_event(event)

    def export_checkpoint_state(self) -> dict[str, Any]:
        # The serializer/next move must not share mutable event lists with the
        # committed receipt; the source scientific state is never mutated.
        return {"schema_version": CAUSAL_CHECKPOINT_SCHEMA,
                "debug_trace_enabled": False,
                "causal_state": deepcopy(super().export_checkpoint_state())}

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        if (value.get("schema_version") != CAUSAL_CHECKPOINT_SCHEMA
                or value.get("debug_trace_enabled") is not False):
            raise MutCausalLineageError("causal checkpoint schema/mode changed")
        state = value.get("causal_state")
        if not isinstance(state, Mapping):
            raise MutCausalLineageError("causal checkpoint has no selected-event state")
        super().restore_checkpoint_state(deepcopy(state))

    def write(self, output_dir: str | Path, payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        summary = super().write(output_dir, payload, **kwargs)
        root = Path(output_dir).resolve()
        validate_causal_recovery_audit(summary)
        if summary["candidate_lineage_resolved_count"] != summary["candidate_count"]:
            raise MutCausalLineageError("candidate causal lineage is incomplete")
        summary.update({
            "causal_schema": CAUSAL_SCHEMA, "debug_trace_enabled": False,
            "causal_lineage_enabled": True, "trace_only": False,
            "observational_instrumentation": True, "rng_calls_added": 0,
            "enumeration_trace_mode": "existing_compact_selected_index_only",
            "candidate_universe_mutated": False,
            "action_inference_or_substitution_permitted": False,
        })
        # Legacy consumers read these established filenames inside the explicit
        # causal_lineage root. They are not a /trace debug directory.
        write_json(root / "trace_summary.json", summary)
        write_json(root / "causal_lineage_manifest.json", {
            "schema_version": CAUSAL_SCHEMA,
            "debug_trace_enabled": False, "causal_lineage_enabled": True,
            "selected_event_manifest_sha256": sha256_file(root / "selected_action_trace_manifest.json"),
            "candidate_lineage_sha256": sha256_file(root / "candidate_action_lineage.json"),
            "summary_sha256": sha256_file(root / "trace_summary.json"),
            "candidate_count": summary["candidate_count"],
            "candidate_lineage_resolved_count": summary["candidate_lineage_resolved_count"],
            "generation_production_parity_claimed": False,
        })
        return summary


def validate_causal_scope(
    *, dataset: str, mode: str, output_root: str | Path,
    debug_trace_root: str | Path | None, causal_root: str | Path | None,
) -> Path | None:
    if causal_root is None:
        return None
    root = Path(causal_root)
    expected = Path(output_root).expanduser().resolve() / "causal_lineage"
    if (dataset != "mutagenicity" or mode != "full" or debug_trace_root is not None
            or root.is_symlink() or root.expanduser().resolve() != expected):
        raise MutCausalLineageError("causal capture is Mut full trace-off only at output_root/causal_lineage")
    return expected
