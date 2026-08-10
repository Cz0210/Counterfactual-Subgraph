# COMRECGC resolver and frozen-payload closure failures

## Scope

This postmortem covers project generation jobs `2222958` (AIDS/HIV) and
`2171296` (Mutagenicity). Neither failure was an OOM condition.

## Mutagenicity failure

The long-lived graph authority was already capable of rehydrating an evicted
graph from SQLite, but one compact-transition path still copied active entries
with a hot-map membership test followed by direct dictionary indexing. An
evicted current-head hash therefore looked absent even though its exact graph
remained in the authoritative store. The old `missing_lookups` metric stayed at
zero because this path never called the resolver that owns that counter.

The compact transition map now resolves and pins active source graphs through
the same fail-closed resolver used by the trace wrapper. It does not change
neighbor enumeration, transition ordering, model calls, RNG calls, importance,
candidate payloads, DBSCAN, or greedy ordering.

## AIDS failure

The AIDS random walk completed all 50,000 moves. The failure occurred later,
while constructing compact candidate lineage:

`Selected trace references a graph absent from the frozen payload.`

`LiveGraphMap.__reduce__` intentionally serialized only the bounded hot map.
Selected trace chunks can still reference graphs that were losslessly evicted
to SQLite. Once the patched runtime context closed, lineage recovery received a
plain payload dictionary and had no route back to the authoritative store.
Move-scoped pins correctly protected the move itself, but could not protect a
post-generation serialization phase. Random-walk `unresolved_lookups` therefore
remained zero even though the later frozen-payload lookup failed.

## Resolution

Before candidate lineage or downstream artifacts are admitted, the project now
computes the frozen reference closure over candidate, traversed/current,
selected-trace, frontier, and available transition references. It resolves hot
entries first, then checksum-verifies and rehydrates SQLite entries. Inline
transition destinations are retained in a graph-only closure. Missing entries,
SHA mismatches, malformed transitions, and official-hash collisions fail
closed.

The project copy of `counterfactuals.pt` is atomically rewritten with the
closure and reloaded for a second verification. The upstream checkout and its
algorithm are unchanged. A separate completed-walk audit decides whether the
old AIDS output is safe for freeze-only recovery; unresolved transition audit
evidence blocks reuse and forces a fresh versioned run.

## Regression gates

- no raw `module.graph_map[...]` read remains in the trace layer;
- evicted compact-transition sources resolve from SQLite;
- selected trace source/target closure replays exactly;
- missing closure entries fail closed;
- transition destinations remain available in the frozen graph closure;
- bounded-cache stress preserves parity and reports zero unresolved lookups;
- the generation integrity gate requires the closure audit and payload SHA.
