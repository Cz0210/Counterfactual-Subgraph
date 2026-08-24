# COMRECGC exact-route semantic review

Date: 2026-08-25

Reviewed release tip: `2b5d3d423b9f6b9f607a87ba3f87c38389b2953a`

## Finding

`status=REJECTED_UNPROVEN_DBSCAN_INPUT` applies to the unmodified release
route. The 91,916,686-row snapshot is physically the complete
71,642-candidate by 1,283-parent Cartesian matrix. Release `2b5d3d42` adopted
that matrix directly but did not bind DBSCAN input to either an independently
validated `normalized_distance <= theta` logical view or a complete
`ALL_PAIRS_CLOSE` certificate. This is an absence-of-proof finding, not a
claim that any row is known to exceed `theta`: the generation implementation
does append only rows passing the inclusive predicate, so a full Cartesian
store could be valid if a complete distance scan proves every row close.

The repaired consumer fails closed for an adopted Cartesian terminal or chunk
snapshot without a hash-closed close-pair manifest. A partial close set is a
bitmap/index view by default and reports
`BLOCKED_STORAGE_INDEXED_DBSCAN_ENGINE_REQUIRED`; it is not silently copied to
a second vector matrix. An explicit compact-byte budget can authorize a small
selected-row copy. A complete `ALL_PAIRS_CLOSE` proof uses the physical vector
mmap and physical/implicit Cartesian pairs without copying them. The view
binds the pair-semantics authority, scalar-distance array, physical vectors,
embedding/distance checkpoint hashes, scale formula, inclusive predicate,
pair axes, row order, and all-close certificate.

## Exact DBSCAN contract

- input: only logical theta-close recourse vectors;
- metric: Euclidean;
- epsilon: `delta` (`0.02` for the AIDS protocol);
- minimum samples: `3`;
- comparison: exact `distance <= eps`;
- self-neighbor: counted exactly once;
- approximate-neighbor use: forbidden.

The all-core/single-component shortcut now publishes and terminally revalidates
four separately hash-bound artifacts:

- `all_core_certificate.json` proves every input row has at least
  `min_samples - 1` distinct exact anchor neighbors in addition to itself;
- `connectivity_certificate.json` proves the exact anchor graph is connected
  and every non-anchor row has an exact epsilon attachment to it;
- `boundary_certificate.json` performs a complete float64 point-to-anchor
  replay, requires exact agreement with the sklearn-radius witness counts, and
  fails closed on any precision disagreement;
- `cluster_partition.json` binds the all-zero labels and all-true core mask,
  establishing one cluster and zero noise only when all preceding proofs pass.

General external-memory DBSCAN remains available when that certificate cannot
be established. Synthetic focused tests compare its partition with sklearn,
including one and multiple clusters, noise, duplicates, exact-epsilon edges,
self-neighbor semantics, and an ambiguous border point adjacent to two core
components.

## Downstream summary contract

The certified one-cluster summary no longer materializes retained positions or
retained vectors. It uses a one-byte mask and deterministic streaming scans in
global row order. It records:

- the official-style Torch float32 centroid and stable float64 audit centroid;
- maximum centroid difference and centroid-threshold decision disagreement;
- strict `distance < delta` membership, exact-at-delta count, and float64
  membership disagreements;
- strict official centroid-norm `< theta` decision and exact-at-theta count;
- official covered parent IDs, first-per-parent candidate IDs, all
  radius-passing candidate IDs, and parent-to-cluster mapping using
  `col0=parent`, `col1=candidate`;
- deterministic first-argmin representative scan without retained-vector copy;
- official greedy cumulative parent coverage, with ascending canonical cluster
  ID tie-break and no duplication to fill `R=100`.

Standardized selected rows explicitly distinguish `cluster_id`,
`selected_rank`, `cumulative_covered_count`,
`representative_candidate_ids`, and `centroid_norm`. In upstream
`selected[label]`, element zero is the cluster label and element one is the
cumulative covered-parent count; it is not a counterfactual ID.

## Validation boundary

Local synthetic and closed integration fixtures pass. They include terminal
physical-snapshot plus close-view coexistence, rejection of Cartesian terminal
adoption without a close view, close-view CLI PASS-last publication, exact
partition/certificate closure, strict downstream boundaries, resume replay,
and wrapper forwarding.

No production GREED full scan or production first/random/dense/sparse/
top-distance subset was run in this development worktree. Therefore this
review does not emit any AIDS production PASS marker and does not authorize
stopping the old route. Production status remains gated on the independently
generated distance scan, close-pair contract, subset equivalence, and full
certificate completion.
