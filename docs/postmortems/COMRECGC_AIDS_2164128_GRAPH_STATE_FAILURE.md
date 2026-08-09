# COMRECGC AIDS job 2164128 graph-state failure

## Incident

- Job: `2164128`
- Project commit: `6daa6d08dae45b6f4531f31c6e3bb920725bfdd5`
- Failed step: approximately `46690 / 50000`
- Exception: `KeyError: -5763365003180206704`
- Failing read: `src/baselines/comrecgc/graph_trace.py` read the hash directly
  from `module.graph_map`.
- Resource diagnosis: this was a Python state-consistency failure, not CPU,
  GPU, or host-memory exhaustion.

The failed output and Slurm evidence remain immutable under
`outputs/hpc/baselines/comrecgc/aids/project_full_comrecgc_end_to_end_retry5_transition_fix`
and `outputs/hpc/postmortems/comrecgc_aids_2164128`.

## Root cause

Pinned upstream COMRECGC removes the lowest-frequency candidate from
`graph_index_map`, `graph_map`, and `transitions` when its candidate array is
at capacity. During a multi-head restart, one head can select a hash and a
later head can evict that same hash before `restart_randomwalk` returns. The
returned `graphs_hash` list therefore still has a live reference after the
active candidate map has removed the entry.

The previous `pinned_upstream_active_move_deferred_eviction_v1` patch protected
transition-map keys only while `move_to_next_graph` was executing. It did not
cover graph-map reads after restart and before the move scope began. The trace
wrapper also read `module.graph_map[value][0]` directly rather than going
through a resolver. Consequently the graph-map lookup raised first and the
transition map's `missing_lookup_count` remained zero.

Negative Python graph hashes are valid. No hash collision was observed; the
new runtime nevertheless verifies every reused hash against a stable node/edge
fingerprint and fails closed on a collision.

## Fix

Project-owned runtime code now keeps upstream's active-map membership and
candidate ordering unchanged while persisting every logically evicted graph
entry to a checksum-verified SQLite backing store. A unified resolver serves
hot entries or losslessly reloads evicted entries. Move-scoped pins cover trace
reads, the official move, transition updates, trace persistence, and deferred
cleanup. Direct trace dictionary access has been removed.

The runtime records eviction, deferral, rehydration, unresolved lookup, cache,
pin, transition, backing-store integrity, and RSS diagnostics. An unresolved
live reference or backing-store integrity failure raises a diagnostic
`ComRecGCLiveGraphResolutionError`; it is never skipped or substituted. The
patch adds no RNG calls and does not alter proposals, importance, DBSCAN,
greedy ordering, seed, heads, or step count. Pinned upstream source is
unchanged.

## Resume decision

The failed directory contains trace/progress evidence but no atomic checkpoint
containing Python, NumPy and Torch RNG state, the current heads, transition
state, graph map, backing-store closure, and deferred deletions. The checkpoint
auditor therefore reports `RESUME_SAFE=false` with
`reason=missing_checkpoint_manifest`.

Reusing the first 46690 steps would make the continuation scientifically
non-reproducible. Retry 6 must start from step zero in a new versioned output
root with the identical scientific parameters. The 19-hour failed run remains
preserved as evidence.
