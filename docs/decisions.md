# Decisions Log

## [2026-08-24] Prefer a closed AIDS pair store and authenticate chunk-cache allocation

### Background

The first fresh exact-Cartesian route always described the 560 closed chunks as
its source.  The old repair may, however, finish promotion before v5 is
released, in which case reconstructing another 23.53 GiB vector cache is both
unnecessary and riskier.  The production continuation wrapper also did not
forward the adaptive shortcut/source/cache controls, and the chunk cache could
not distinguish a crash after physical allocation from an unauthenticated
sparse partial.  A nominal terminal manifest could coexist with a partial or a
writable sibling inode.

### Decision

At runtime, inspect one frozen automatic pair-store root.  A physical nonempty
terminal manifest has strict priority and is adopted by read-only reference;
its invalidity never authorizes chunk fallback.  Before adoption, recursively
reject symlinks and partial artifacts in the pair store, writable FD/mapping
references to every sibling inode, and any live command that names the old
owner root.  Revalidate the same guard, exact manifest/array hashes, and stat
identities at terminal closure.

Use closed chunks only when no terminal manifest exists.  Split local cache
creation into authenticated `allocate_cache`, `allocation_complete`,
`copy_chunks`, and `cache_ready` phases.  Compute remaining physical allocation
from `st_blocks`, require exact remaining bytes plus the 3 GiB floor, run
`posix_fallocate`, verify physical allocation/header/floor, and bind that
evidence into every later checkpoint.  Replaying `posix_fallocate` after a
pre-checkpoint crash is safe.  A missing cache after terminal publication
requires a fresh root rather than rewriting the terminal checkpoint.

For the actual v5 release, terminal promotion is already complete.  Freeze a
stronger `require_promoted_final=1` gate, omit all chunk/cache fallback
variables, and set the source owner to the exact pair-store root.  The old
DBSCAN process may retain a read-only mmap concurrently; a full pair-tree
writable-FD/mapping/partial scan still blocks any mutable producer.  If the
terminal disappears or becomes invalid, v5 stops instead of switching routes.

This coexistence is not a general high-memory-lock bypass.  Freeze the old
consumer's PID, Linux start ticks, raw cmdline hash, exact output argument, and
execution cwd in the v5 spec.  At build it must be the only common-recourse
process.  Before every attempt, scan procfs again and allow either that exact
generation or an empty set after its natural exit.  Reject PID reuse, command
or cwd drift, and every second common-recourse process.  In parallel require at
least 128 GiB live cgroup headroom (which includes the old RSS), cap the new
route at 96 GiB RSS, and hold a distinct v5 route flock.  Mut remains blocked
on v5 PASS, so it cannot overlap this exception.

Queue a monitored helper on the same physical global high-memory flock before
starting the first v5 science child.  While old v4 is alive that helper waits;
when v4 exits naturally it acquires and retains the lock until the v5
supervisor generation exits.  The supervisor binds the helper's own Linux
generation and monitors it while each science child runs in a fresh process
group.  Helper failure terminates only that new process group and fails v5;
cleanup verifies the helper generation before signalling it, so PID reuse
cannot redirect a signal.

Continue the process-set scan while the science group runs.  In that interval
the only additional common-recourse command may be the exact script/output/cwd
child of the PID/start-tick-bound v5 science root.  A non-descendant, a reused
science-root PID, a second child, or command/output drift terminates only v5
and fails the fresh controller.

For the retained generic chunk route, distinguish an unauthenticated allocation
artifact from authenticated data.  Only `phase=allocate_cache`, under the
allocation flock plus an independently held route flock, may discard and
atomically rebuild a zero-byte, truncated, or wrong-schema local NPY partial.
`allocation_complete` and later phases always fail closed on the same damage.

Publish v5 through its dedicated two-task persistent manifest: a deterministic
hash-only adoption gate for the already frozen calibration selector, followed
by the terminal-only CPU science task.  The second task is the sole Mut
dependency authority.  Mut must bind its exact controller manifest SHA, task
ID, and attempt-0 output in a fresh controller; existing wait controllers are
immutable.

Freeze the production AutoDL wrapper to the reviewed adaptive shortcut,
`eps=0.02`, `min_samples=3`, sklearn self-neighbour semantics, dense fallback
zero, CPU-only execution, one route-wide scratch flock, and one bounded
process-loss-only same-root resume.  Keep the paired Slurm entrypoint static and
non-runnable; no HPC experiment is launched.

### Consequences

- A promoted final source avoids the local 23.53 GiB cache entirely.
- Active/ambiguous promotion and storage allocation fail before DBSCAN or
  downstream selection; no partial becomes scientific authority.
- Chunk fallback preserves candidate-major/parent-minor bytes and the exact
  sklearn/official selector semantics already covered by the production-shaped
  equivalence fixture.
- Old repair-v4 processes and outputs remain untouched.  V5 needs a fresh
  execution worktree, output root, controller/spec/manifest, source audit,
  smoke, and independent release review.
- The old process may exit naturally, but no signal or write is authorized by
  this route and its PID can never be reused as evidence for a new generation.

### Status

Accepted (code/tests only; remote launch remains separately gated)

## [2026-08-23] Interpret pinned GlobalGCE edge decoder outputs as categorical scores

### Background

The first BACE GlobalGCE frozen-GINE full run completed exhaustive gSpan mining
but failed as soon as official rule training called the differentiable bridge.
The bridge rejected negative `edge_attributes` because it assumed they were
nonnegative class weights. In pinned official commit
`157e65c2850bc787f229a1ee8c60564906b933f2`, however, the apparent final
`nn.Sigmoid()` is passed as the third positional argument of `nn.Linear`; it
sets the truthy bias flag and is not appended to `decoder_edge_attr`. The
decoder therefore emits unrestricted affine class scores, and the official
hard graph codec decodes them with `argmax`.

### Decision

Keep the official decoder and its LHS-to-RHS semantics unchanged. At the
categorical frozen-GINE boundary, map each finite edge score vector through a
softmax over its class dimension before taking expected bond embeddings. This
preserves the official hard `argmax` class for every unique maximum, admits
negative scores, and provides gradients to the official transformation and
decoder variables. Do not clamp scores or replace the frozen calibrated GINE
with RF, GTGNN, or a surrogate. Continue to normalize node values separately,
because the pinned node decoder does contain a real terminal sigmoid.

The bridge smoke must include production-shaped negative edge scores in
addition to the one-hot identity case, and must prove hard-oracle parity,
nonzero edge-score gradients, no classifier gradients, unchanged frozen
parameters/checkpoint, and finite outputs.

### Consequences

- The failed v5 training root remains immutable and cannot be relabelled PASS.
- A corrected full attempt requires a fresh output root and immutable execution
  commit.
- Negative finite edge scores are valid bridge inputs; malformed rank/class
  shape and NaN/Inf remain fail-closed errors.
- Final hard transformations are still sanitized and reverified by the same
  calibrated frozen GINE.

### Status

Accepted

## [2026-08-23] Gate the three-dataset release with an external-owner supervisor

### Background

The twelve AIDS/Mutagenicity/BACE cells are produced by several already
running persistent controllers. Starting another aggregate controller would
compete for their locks, while rendering from a historical handoff or a
candidate path could promote a stale or raw result. In particular, the BACE
GlobalGCE v5 native final is not yet the standardized Figure 3/Figure 4/Table 2
closure. AIDS and Mutagenicity ComRecGC may also select fresh repair routes.

### Decision

Add an independent CPU-only release supervisor that never mutates a scientific
controller or cell root. A builder freezes each standardized root together
with the SHA of its external owner manifest, owner task ID, and exact output
binding. Unsettled routes remain explicit placeholders and cannot be inferred
from hints. BACE GlobalGCE is activated only through the existing fresh CPU
`bace_globalgce_standardized` task, never from its raw final directory.

At fewer than twelve passing cells, the supervisor writes only PID/control/
heartbeat state and no numeric registry, figure, or table. At exactly twelve,
it applies the existing canonical registry and three-dataset audits, binds all
closure file identities including `PASS`, publishes a sixteen-row matrix with
four TasteMolNet license blocks, and invokes the existing staging-only
exporter. Fresh directories use atomic no-replace publication; process-loss
transactions are restartable and never duplicate a still-live exporter.

### Consequences

- Six user-approved v4 cells remain render-only and are not scientifically
  recomputed.
- TasteMolNet remains `BLOCKED_LICENSE_REVIEW`; 12/16 is explicitly partial.
- Result and paper-staging trees are byte-identical under runtime, and
  `paper/` remains unchanged.
- The code and template can be integrated now, but deployment remains blocked
  until the three placeholder owner routes are immutable and a runnable spec
  is reviewed.

### Status

Accepted

## [2026-08-23] Queue BACE GCF equivalence in an immutable UUID-lock sidecar

### Background

The active four-by-four and repair controller manifests are frozen by exact
SHA and task topology after first launch, so appending GCF replay tasks would
invalidate their persistent identity.  Four protected jobs already hold real
exclusive GPU UUID locks: legacy GCF full, GlobalGCE v5 full, the existing
ComRecGC M=500 legacy/optimized pair, and legacy ComRecGC full.  None may be
stopped or duplicated merely to make a card available.

### Decision

Build a fresh, GCF-only sidecar controller with three strictly dependent tasks:
quick M=50, quick M=100, then formal M=500.  Every task uses the fixed
duplicate-preserving GINE batch implementation, a fresh attempt-qualified
output, an exclusive global UUID lock, and a PASS-last equivalence gate.  The
sidecar audits all four protected run/worker/UUID-lock identities before its
manifest is frozen and records that it sent no signal.

Do not enqueue ComRecGC again.  Its already-running M=500 exp-run is itself the
approved sequential legacy-to-optimized pair and will publish its independent
audit if it succeeds.  The sidecar records that launch-spec identity only as
protected read-only evidence.  Quick GCF PASS cannot authorize a full run, and
M=500 alone cannot replace the separately required M=1000 and performance
gates.

### Consequences

- Existing controller manifests and all old full roots remain immutable.
- A sidecar task starts only after a physical GPU is stably idle and its global
  exclusive UUID lock is available.
- A failed quick replay blocks later GCF replay tasks without stopping any old
  full process or generating a false aggregate PASS.
- The running ComRecGC M=500 pair remains the sole M=500 pair; no duplicate
  scientific workload is introduced.

### Status

Accepted

## [2026-08-23] Bind BACE GlobalGCE native training to the frozen 869-ID view

### Background

The first full frozen-GINE route stopped before training because the processed
`BACE/train.csv` correctly contains all 959 train rows, while the previously
prepared train/validation graph bundle freezes a teacher-consistent 869-row
native train vocabulary.  The same graph manifest contains 360 project-label-1
sources, 509 project-label-0 targets, and 162 validation rows.  Treating the
raw CSV row count as the vocabulary count was therefore incorrect.

### Decision

Audit both dataset manifests and artifact hashes, derive the native vocabulary
from the exact 869 train molecule IDs, and join those IDs to the 959-row
processed train CSV by molecule ID, label, and canonical SMILES.  Select the
360 generation parents only where project label is 1 and frozen-GINE label is
0.  Validation rows are audited but never loaded into the native trainer;
calibration/test rows are rejected.  The 90 excluded rows remain recorded
train-only exclusions.  The generic GlobalGCE loader accepts this exact-ID
filter as an opt-in input, leaving Mutagenicity behavior unchanged.

### Consequences

- No row is truncated by position and no oracle, budget, rule semantics, or
  `min_freq=7` setting changes.
- A missing ID, label/SMILES drift, changed order hash, split leakage, or
  manifest/artifact hash drift fails before an output root is published.
- The failed v3 controller/root remains immutable; any retry uses a fresh v4
  controller and output root.

### Status

Accepted

## 2026-08-23 — Stage three-dataset results without weakening the final matrix

### Decision

Add a separate staging-only exporter for AIDS, Mutagenicity, and BACE.  It
accepts only the canonical 16-cell registry with exactly twelve paper-pass
non-TasteMolNet cells and four explicit `BLOCKED_LICENSE_REVIEW` TasteMolNet
cells.  It reuses the same per-cell artifact/hash/oracle/split/threshold audit
as the final exporter, copies reported CSV values verbatim, and renders only
raw empirical points.  It writes `paper_figure{3,4}_three_datasets.pdf` and
`paper_table2_three_datasets.tex` under runtime staging roots, never under
`paper/`, and explicitly records `PAPER_FROZEN_PARTIAL`.

The existing four-dataset exporter remains unchanged and still requires
16/16.  Missing cells are never zero-filled, interpolated, or promoted from a
TasteMolNet smoke run.

### Status

Accepted

## [2026-08-23] Separate optimization code from scientific release authority

### Background

The reviewed ComRecGC preprocessing and shared-GPU slot implementations were
opt-in, but a manifest could still request an optimized 50k run without one
aggregate exact-500/exact-1000 replay closure, or request `shared_lowmem` from
a positive VRAM reservation without a real co-location A/B result. Either path
could turn an engineering optimization into an unsupported scientific result.

### Decision

Require an immutable ComRecGC full-acceleration gate which revalidates both raw
replay roots, equivalence audit self-hashes, payload/completion/trace hashes,
frozen GINE and distance hashes, parent/dataset identity, batch size, runtime
configuration, strict flip, no calibration/test access, and the exact 50,000
parameters immediately before creating the formal output root.

Independently require a measured co-location gate for every shared-lowmem task.
The evidence consists of two 10--15 minute single-task profiles and one
same-GPU two-task profile. Aggregate throughput is recomputed from per-task
measurements and must improve by at least 20%; benchmark key, scientific config
and canonical result identities must match; OOM/error counts are zero; CPU and
disk health gates pass; MPS is disabled; and peak VRAM plus margin is below 70%.
Controller schema and launch, exp-run launch and worker acquisition all recheck
the exact gate bytes and source evidence. Both slots must use the same gate and
the exact benchmarked workload pair.

Existing exclusive diagnostic/legacy tasks are neither stopped nor migrated.
No real PASS is generated by this code change; formal optimized/shared launch
remains blocked until the ongoing AutoDL evidence produces the required roots.

### Consequences

- A status label or manually copied PASS file cannot authorize either route.
- Gate/source/config/oracle/result drift fails before scientific output starts.
- Low instantaneous utilization and free VRAM alone never authorize co-location.
- Slurm parity entrypoints exist, but no HPC or AutoDL job is deployed here.

### Status

Accepted

## 2026-08-23 — GlobalGCE bridge smoke log marker

### Context

The first AutoDL frozen-GINE bridge smoke completed every scientific check,
wrote `PASS` and `BRIDGE_PASS`, and exited zero. The generic controller still
classified the task as failed because its separate log-evidence contract
required `[BACE_GLOBALGCE_BRIDGE_PASS]`, while the CLI printed only the JSON
result.

### Decision

After the JSON result, the thin `globalgce-bridge-smoke` CLI prints the exact
controller marker. The evaluator remains side-effect-free apart from its
atomic artifacts. A fresh task/controller root is required; the failed
controller state is not rewritten or relabelled.

### Consequences

- Frozen-GINE weights, gradients, native LHS-to-RHS semantics, and scientific
  outputs are unchanged.
- PASS-last artifact checks and the independent log marker now agree.
- The previous failed attempt remains immutable diagnostic evidence.

### Status

Accepted
## [2026-08-23] Authenticate and replay every shortcut scan prefix before PASS

### Background

The first fixed/adaptive anchor implementation checkpointed a mutable
`next_offset` and running minima, while the lower-bound array was preallocated
with zeroes.  A reviewer reproducer interrupted a fixed 12-by-64 scan after
rows 0--2, changed only `next_offset` from 3 to 12, and resumed.  That state
could reach a false all-core PASS even though row 10 was outside epsilon and
sklearn correctly labelled it noise.  Adaptive seed and failure scans had the
same unauthenticated-prefix class of risk.

### Decision

External schema v3, adaptive-selection schema v2, and shortcut-proof schema
v2 give every checkpoint a payload SHA and every committed adaptive
seed, adaptive failure, and final anchor-lower block a scientific-identity-
bound forward hash-chain entry.  Seed entries bind the block-local float64
top-k rows, failure entries bind every insufficient source index, and lower
entries bind canonical little-endian uint32 values plus block minima.  The
checkpoint binds the complete ledger map and its committed offset.

Any resume replays all committed entries at their original boundaries from
the immutable vector source and frozen sklearn brute models.  It compares the
recomputed seed rows, failure indices, lower values, and stored partial array
before continuing; changing `next_offset`, a ledger entry, an aggregate, or a
committed lower slot fails closed.  The completed failure ledger is persisted
before publishing adaptive-selection artifacts so a rename/checkpoint crash
window remains resumable and cannot create an unbound selection.  The frozen
failure cap is asserted again on every replay-valid ledger and immediately
before selection publication, so coordinated state re-signing cannot turn a
cap-exceeded terminal prefix into a complete first pass.

The vector source identity now includes device, inode, mode, byte size, mtime,
and ctime captured around the initial full SHA-256.  Opening the mmap must
preserve that snapshot.  Both shortcut and ordinary exact routes perform a
second complete SHA-256 with matching before/after stat identity immediately
before PASS, followed by a final stat check at manifest publication.  Terminal
reopen repeats the same content/stat closure.  A long uninterrupted scan can
therefore no longer publish evidence for bytes that changed after the entry
hash.

Before labels are materialized, the finalizer verifies that lower-ledger
coverage is contiguous and exact over every source row, rehashes every lower
slice, recomputes the global and non-anchor minima, and enforces the core and
attachment thresholds.  It freezes the completed ledgers as a separate
checksum-bound proof artifact and verifies that labels are all zero and the
core mask is all true.  Terminal reopen repeats artifact, complete-ledger,
full-lower, selection, connectivity, and constant-output closure.  Exact full
neighbor counts remain explicitly unavailable rather than synthesized.

### Consequences

- The reviewer 12-by-64 false-positive reproducer and independently
  reauthenticated seed/failure offset tampering are rejected before PASS.
- Existing external-schema-v2 checkpoints and terminals are deliberately
  non-resumable under schema v3; a future formal run must use a fresh output
  root.
- Resume may spend one bounded linear replay pass validating its committed
  prefix.  This is required release evidence, not optional diagnostic work.
- The real evenly-spaced-64 failure remains immutable evidence.  The adaptive
  route is still not wired to, deployed on, or run on AutoDL by this change.

### Status

Accepted (core closure only; fresh AutoDL release remains separately gated)

## [2026-08-23] Adopt an exact Cartesian AIDS pair source into a fresh route

### Background

The running repair-v4 source has closed 560 chunks before completing its slow
network-filesystem consolidation.  Those chunks contain 71,642 candidates by
1,283 parents: exactly 91,916,686 rows.  The adaptive shortcut and streaming
summary belong to a newer commit and must never resume inside that old attempt.

### Decision

Allow a specialized chunk source only for the CPU AIDS external engine and
exact adaptive shortcut.  Reconstruct the complete pair scientific identity
from the frozen dataset/source audit, generation payload, candidate and parent
order, distance checkpoint, parameters, and batch contract.  Rehash every
checkpoint/chunk file, freeze physical stat identities, reject writable
references, and prove each pair row elementwise as
`[row % parent_count, row // parent_count]` with no gaps or overlaps.

Copy only raw vector-array bytes, in source chunk order, into one physically
allocated contiguous local-XFS `.npy` cache.  Derive and recheck the target
header/file SHA from persistent chunks.  Keep pair indices implicit using the
proven Cartesian formula, so the route does not allocate a second pair array.
Never compute chunk-local centroids, distance reductions, medoids, or coverage:
the exact DBSCAN proof and summary consume the reconstructed contiguous vector
layout and therefore preserve pinned floating reduction order.

The persistent chunks plus fresh audit are scientific authority; the local
vector file is reconstructible operational cache.  Require free space for its
exact size plus a 3 GiB floor, `posix_fallocate`, and a route-wide scratch flock.
Both normal completion and same-root resume revalidate hashes/stats, source
identity, zero writers/owners, cache content, proof, and summary.  A preliminary
audit may record a live old owner but cannot authorize adoption.  The old
controller/root remain immutable, and stopping it requires a separately
authorized graceful stage-boundary action after code, smoke, audit, and review.

### Consequences

- Fresh DBSCAN/summary work consumes one contiguous local vector memmap while
  the old persistent chunks remain read-only authority.
- A changed checkpoint/chunk/stat/hash, live writer/owner, Cartesian mismatch,
  local-space shortfall, or procfs audit failure stops before DBSCAN.
- Chunkwise floating reductions and active partial consolidation arrays remain
  forbidden; only raw-byte contiguous reconstruction is allowed.
- This mechanism does not authorize a launch: the adaptive proof and route
  still require independent review and a fresh controller manifest.

### Status

Accepted (route implementation in review; source audit and fresh launch pending)

## [2026-08-23] Replay a proven one-cluster summary with ordered disk-backed blocks

### Background

The exact adaptive DBSCAN certificate can prove that every one of the
91,916,686 AIDS recourse rows is core and has sklearn label zero.  Calling the
pinned upstream summary unchanged after that proof is still infeasible: it
creates a 91.9-million-element Python index list, iterates one Torch scalar at
a time, and the lineage trace makes repeated 23.5 GiB advanced-index copies.
Those operations add no clustering information once the sole label is proven.

### Decision

Add a specialized path that is legal only after the complete DBSCAN manifest
and anchor-proof artifact closure validates.  Reproduce the upstream Torch
centroid independently from the lineage trace's NumPy centroid.  For an
all-true label-zero selection, compute each centroid from the direct contiguous
read-only memmap view; frozen NumPy 2.2.6 and Torch 2.7.1 fixtures establish
bit equality with the legacy all-row advanced-index copy.  Compute strict
`distance < radius` membership in source-ordered blocks because every row norm
is independent, while preserving the first counterfactual observed for each
parent in the upstream coverage result.

For the lineage trace, persist the retained mask, original global positions,
and retained vectors in original order.  Compute the retained centroid from
that disk-backed contiguous array with the unchanged NumPy reduction, then
scan medoid distances in order and update the winner only on strict
improvement so the first-`argmin` tie rule is unchanged.  Preserve the native
parent/counterfactual sets, official one-cluster greedy call, cumulative cost,
and selected-row schema.  Every phase is resumable and every promoted array,
centroid, input manifest, and result is checksum closed.  Record honestly that
the general upstream coverage function was analytically replayed rather than
called; no approximation is allowed.

### Consequences

- The route removes Python work proportional to 91.9 million objects and
  bounds resident scratch by one block; retained vectors may use disk but keep
  exact row and reduction order.
- Torch and NumPy radius masks are not interchanged, since their boundary
  arithmetic belongs to different frozen legacy consumers.
- The general multi-cluster route remains unchanged.  A missing/tampered
  all-core proof or summary closure fails before final PASS.
- The source pair store is opened read-only.  Production adoption and launch
  must use a fresh output/controller and never write the old v4 attempt.

### Status

Accepted (implementation and fixtures complete; fresh source-adoption release
remains separately gated)

## [2026-08-23] Expand an exact DBSCAN witness by all deterministic failures

### Background

The full read-only evenly-spaced-64 diagnostic completed all 91,916,686 rows.
Its anchor graph was connected and all anchors saw all 64 anchors including
self, but 43 input rows still had insufficient anchor coverage and the minimum
excluding-self lower bound was zero.  The fixed witness therefore correctly
fails closed and cannot release the AIDS result.  An exploratory expansion
showed that a small exceptional set exists, but its ad hoc representatives are
not reusable scientific evidence.

### Decision

Add a separate opt-in
`all_core_one_component_adaptive_anchor_v1` selection.  First scan every
promoted vector row and select the globally smallest three squared L2 norms,
computed in float64 and ordered by `(norm, sample_index)`.  This algorithm,
seed count, complete vector SHA/dtype/shape, exact seed indices, hexadecimal
norms, and index-list hash are frozen.

Run a complete sklearn-brute seed-radius pass.  Record every sample index whose
distinct-seed lower bound cannot yet prove core status or anchor attachment.
The list is source ordered, unique, persisted exactly, and closed both by its
`.npy` SHA and a portable index-list SHA.  Exceeding the declared failure cap
is terminal and never truncates, samples, approximates, or falls back.

For a cap-compliant list, define the final anchors without a heuristic:
`sorted(unique(seed_indices union all_failure_indices))`.  Freeze the exact
anchor-index array and exact source anchor rows.  Then rerun the full input
scan against this final set and apply the unchanged all-core/single-component
proof.  The final proof binds the adaptive selection-manifest hash, seed and
failure identities, anchor-row hash, connected anchor graph, and complete
second-pass lower-bound array.

### Consequences

- The exploratory representatives and fixed-64 negative cannot be promoted to
  formal evidence.
- Selection and proof are reproducible from the promoted vector file alone;
  every tie and union ordering is specified.
- A large or poorly connected exceptional set stops with an explicit
  complexity block instead of changing DBSCAN semantics.
- Existing controllers and output roots remain immutable; AutoDL wiring and a
  fresh full proof are separate release steps.

### Status

Accepted (core only; fresh full proof still required)

## [2026-08-23] Shortcut dense DBSCAN only with a complete anchor proof

### Background

The exact three-pass external DBSCAN bounds memory but not work.  For the
91,916,686-row, 64-feature AIDS recourse array, a dense epsilon graph would
still require quadratic brute-force distance work even with query blocks of
eight.  A constant-label shortcut is scientifically acceptable only when a
small witness proves the result; observing a dense diagnostic sample is not
such a proof.

### Decision

Add an opt-in `all_core_one_component_anchor_v1` route.  Select distinct
sample-index anchors deterministically by
`floor(i * (N - 1) / (A - 1))`, bind that exact index-list hash, the complete
promoted vector-file SHA-256, dtype/shape, frozen sklearn version, epsilon,
`min_samples`, and every shortcut parameter into the scientific identity.
The shortcut is allowed only when sklearn's full-data `algorithm=auto` route
resolves to the same brute Euclidean kernel used for the anchor scan.

For every input row, count distinct anchor sample indices at distance
`<= float(eps)`, excluding the row's own sample index only when it is itself
an anchor.  Duplicate vectors remain distinct sample indices.  A lower bound
of at least `min_samples - 1` proves the row is core because sklearn also
counts its own index.  Independently require the anchor epsilon graph to be
connected and every non-anchor row to touch an anchor.  These conditions prove
that every row is core and the entire ordered input is one component, so
sklearn's exact labels are elementwise zero and its core set is every row.

Save the anchor indices, canonical undirected anchor edges, and per-row anchor
lower bounds as checksum-closed proof artifacts.  Do not create or claim an
exact `neighbor_counts.npy`; the manifest records that it is unavailable and
why.  A failed proof may enter the original exact three-pass route only when
`N` is at or below an explicit fallback limit.  Above that limit it emits
`EXACT_DBSCAN_COMPLEXITY_BLOCKED`, with an immutable inconclusive witness, and
never approximates labels.

### Consequences

- A successful witness changes quadratic dense-radius work to `O(N * A)` and
  keeps memory bounded by one sample block and the finite anchor set.
- Boundary, self, duplicate, label-order, resume, input-hash, and RSS behavior
  remain fail-closed and are compared directly with sklearn fixtures.
- Exploratory anchor diagnostics are not adopted as formal evidence; a fresh
  run must recompute the proof against the promoted pair-store vectors.
- Existing AutoDL worktrees, checkpoints, and output roots remain untouched.

### Status

Accepted (core implemented; fresh AutoDL release remains separately gated)

## [2026-08-23] Bound AIDS ComRecGC DBSCAN memory without changing labels

### Background

The serialized AIDS generation payload contains 100,262 candidates and is
4,913,145,399 bytes.  Repair-v3 loaded the frozen 1,283-parent contract and
then died in `run_common_recourse.py` when the AutoDL cgroup reached its
515,396,075,520-byte limit.  The legacy implementation retained every
theta-eligible pair/vector as Python objects and sklearn DBSCAN retained every
epsilon neighborhood simultaneously.  A second exclusive run reached the
same limit, proving that serialization rather than task colocation was the
remaining blocker.

### Decision

Keep sklearn 1.7.2's Euclidean radius query and the original float32/float64
recourse vectors, but query it in bounded deterministic blocks.  Determine
core status from an exact first pass, union only epsilon-adjacent core points
in a resumable second pass, then assign every border point to the earliest
numbered adjacent core component in a third pass.  Components are numbered by
their minimum core sample index.  This is exactly the ordered
`dbscan_inner` result, including ambiguous borders, without retaining the
radius graph.

Materialize candidate-major/parent-minor pair/vector chunks as atomic `.npy`
files and consolidate them with memory maps.  Preserve the legacy NumPy
centroid and medoid reduction on one cluster at a time and invoke the pinned
official greedy routine unchanged.  Every phase has a hash-bound checkpoint;
the last incomplete block is idempotently replayed after interruption.  A
hard RSS budget is checked before every worst-case radius query and cluster
copy.  Version, array dtype/shape/hash, epsilon, `min_samples`, block contract,
labels, and all checkpoints are fail-closed.

Pair consolidation, neighbor/core promotion, and final-label promotion use a
two-phase ready checkpoint.  Recovery accepts either the checksum-bound
partial name or its already-renamed final name, including a crash between two
promotions, and rejects mixed or tampered artifacts.  Pair-store identity also
binds the project dataset fingerprint and complete bundle audit, so same-path
changes to `graphs.pt`, `dataset_summary.json`, or the AIDS source CSV cannot
reuse earlier chunks.

The standardized continuation freezes hashes for all scientific inputs and
all stage commands.  A child-complete/parent-interrupted window is reconciled
only after the common terminal closes its run manifest, selected rows,
representative payload, pair arrays, DBSCAN arrays, and nested manifests by
checksum.  Other partial downstream stages fail closed.

The external-memory engine is opt-in and the legacy engine remains the CLI
default.  A full runner fixture already proves identical pair order, labels,
official summary, selected rows, and selected-row hash.  It remains unreleased
for a full AIDS cell until that gate also passes in the AutoDL environment and
a fresh repair-v4 controller is built.  Repair-v2 and repair-v3 roots remain
immutable.

### Consequences

- Neighbor storage is bounded by one query block rather than all points.
- Scientific output is independent of resource block size.
- A resume cannot change the vector file, sklearn version, clustering
  contract, chunk identity, or pair order.
- Any cluster too large for the declared exact NumPy reduction budget fails
  closed instead of switching to an approximate or differently rounded mean.

### Status

Accepted (algorithm core; full-route release still gated)

## [2026-08-23] Supervise AIDS external-memory repair in one persistent run root

### Background

The generic controller creates a new attempt root for an ordinary retry, while
the external-memory implementation can resume only when its exact pair-store,
DBSCAN, dataset, input-content, argv, and stage identities remain bound to the
same output root.  A process loss after an atomic child completion can also
occur before the parent writes its stage PASS checkpoint.

### Decision

Build a dedicated three-task AutoDL controller named
`four_methods_four_datasets_aids_comrecgc_repair_v4`.  Two CPU manifest-only
tasks revalidate the immutable repair-v2 generation and threshold sources.  A
single CPU scientific task then runs the unchanged full AIDS parameters under
the shared high-memory flock, at least 128 GiB of cgroup headroom, a 96 GiB
external-engine RSS budget, query blocks of eight, and one OMP/MKL thread.

Wrap that scientific task in a bounded same-process supervisor.  It may invoke
the exact `--resume` route once, in the same exp-run, attempt, and output root,
only after a hash-bound common-stage checkpoint and explicit SIGKILL/SIGTERM
evidence pass the retry gate.  Semantic, dataset/input drift, sklearn, RSS,
lineage, and leakage failures are never retried.  A second process loss is
terminal.  Restarting the outer controller reconciles a still-live supervisor
by PID start-time and launch identity instead of allocating a new attempt.

Both fresh completion and resumed completion validate the same terminal-v2
closure before publishing a stage PASS: the run manifest, selected JSON/CSV,
representative payload, pair arrays/manifest, and (when present) DBSCAN
arrays/manifest must all match their recorded SHA-256 values.  The builder is
pinned to the complete recovery-core commit
`d5c1d67339df4b9642beaf2b10908ed92bac30de` and fails when only an earlier
external-memory implementation is present.

The full repair remains gated on a fresh, diagnostic-only AutoDL
legacy-versus-external equivalence smoke.  The builder binds its physical
PASS-last gate by SHA-256, requires the same integrated execution commit and
frozen source-payload hash, requires all nine pair/vector/label/selection
checks to be true, rehashes every diagnostic evidence artifact, and rejects a
live writer.  The smoke cannot be used as a paper result.  Repair-v4 writes a
new root and never resumes or mutates repair-v2 or repair-v3.  Paired Slurm
scripts are static CLI documentation only and must not be submitted.

The migrated-checkout safe-Git fix and the external-memory recovery core use
separate exact-commit ancestry validators.  The safe-Git validator remains
restricted to its reviewed commit; the external validator independently
requires the complete recovery-core commit.  Neither helper accepts an
arbitrary caller-provided ancestry token.

### Consequences

- A resumable interruption cannot silently switch inputs, output root,
  scientific parameters, or external-memory artifacts.
- Controller restart and child-process restart are separate, bounded recovery
  cases with explicit evidence.
- The repair uses no GPU and does not consume a GPU lock.
- A successful diagnostic smoke is necessary but is not a paper result.

### Status

Accepted (deployment remains smoke-gated)

## [2026-08-23] Serialize the AIDS ComRecGC retry on CPU under a cgroup RAM gate

### Background

AM repair-v2's AIDS `run_common_recourse.py` child received `SIGKILL` while the
Mutagenicity common-recourse stage ran concurrently. The AutoDL cgroup-v1
limit was 480 GiB; its recorded maximum exceeded that limit, failure count was
nonzero, and `oom_kill=1`. Mutagenicity independently completed its clustering
but correctly failed chemistry preregistration because lineage-v3 has streamed
trace integrity rather than a true trace-on/trace-off parity artifact.

### Decision

Create an AIDS-only repair-v3 controller with two exact repair-v2 source-gate
adoptions and one fresh standardized continuation. Run every task as CPU,
clear CUDA visibility, set `gpu_required=false`, and limit controller CPU
concurrency to one. Hold a project-persistent exclusive high-memory lock for
the whole scientific task, require explicit cgroup-v1 headroom before creating
the output root, and reject any already-running legacy common-recourse process.

Use the immutable exp-run registry terminal as the authoritative wrapper exit
record. The long-lived controller's reconciled task cache is allowed to omit
its redundant `exit_code`, but if present it must be `1`; the exp-run terminal
must always be FAILED with exit `1` and match the exact controller run, output,
PIDs, log, and launch specification. This evidence is accepted only jointly
with the child `SIGKILL: 9` failure artifact and cgroup OOM proof.

Do not include a Mutagenicity retry and do not reinterpret trace integrity as
trace parity. A no-generation-rerun parity preflight is possible only if an
independently frozen trace-disabled full-budget payload with identical
scientific identity is located; the audited AutoDL inputs contain no such
reference.

### Consequences

- Repair-v2 roots and its `SIGKILL` evidence remain immutable.
- AIDS consumes no GPU lock and cannot overlap another cooperating full
  common-recourse job in host memory.
- Mutagenicity remains a scientific evidence blocker rather than an
  engineering retry.
- BACE, TasteMolNet, `paper/`, and HPC execution remain outside this repair.

### Status

Accepted

## 2026-08-23 — Require an independent trace-off Mutagenicity reference

### Decision

The frozen Mutagenicity ComRecGC lineage-v3 payload remains the immutable
trace-on authority and honestly records algorithm commit
`7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4`.  Exact resume requires the
separate reviewed checkpoint-instrumentation release
`66487c062c86d53ef2f762ce04d0fb965af5af08`; neither its manifest nor the paper
cell may claim that this execution commit is the historical 7f commit. Before
the full run, an exclusive-GPU, fresh 500-step legacy/instrumented prefix gate
freezes both worktrees' scientific source and top-level AST hashes and requires
identical candidate payload, step/action trace, prediction/coverage state, and
Python/NumPy/Torch RNG state. All source deltas stay visible and are classified;
runtime equivalence, not a false source-identity assertion, constrains their
effect on the Mut trace-off route to completed-step checkpoint/resume support.

Chemistry standardization may resume only after a fresh 50,000-step, seed-0,
trace-disabled reference uses the same frozen data order, RF-backed project
route, GNN/distance checkpoints, and generation parameters, then passes the
existing full normalized trace-parity assertion against the 7f trace-on
payload. The reference is an exclusive-GPU job with completed-step checkpoints
and a separately mirrored checkpoint authority; trace fields are neither
removed nor synthesized.

The continuation adopts the already-complete repair-v2 common-recourse output
read-only only after parity passes, then reruns chemistry/evaluation/freeze in
a fresh CPU-only root.  Every retry writes an immutable attempt gate while the
scientific generation root stays fixed so `--resume` retains one canonical
command identity.  The historical lineage-v3, repair-v2, and threshold roots
are never modified.

The dependency on AIDS is injected as an exact future repair manifest,
controller, task, output, wrapper, and manifest SHA-256.  The OOM-failed
repair-v3 controller is explicitly rejected and cannot release Mutagenicity.
Both routes use the same host-memory flock/cgroup/proc contract, but their
headroom thresholds are independently frozen: bounded external-memory AIDS-v4
uses 128 GiB, while the Mut continuation retains its conservative generation
gate. They are never required to lie about having the same peak-memory model.

### Consequences

- No copied payload or self-comparison can satisfy the trace-parity gate.
- The 50k replay cannot start unless the independent 500-step instrumentation
  gate passes and is revalidated against the source inventories frozen in the
  controller manifest.
- Calibration and held-out test data remain unloaded during both 50k runs.
- The controller can recover from a fully mirrored generation checkpoint, and
  publishes PASS only after checkpoint, provenance, payload, and parity audits.
- Deployment is blocked until a memory-bounded AIDS repair succeeds and its
  immutable manifest SHA is supplied to the Mutagenicity spec.

### Status

Accepted; implementation tested, deployment dependency pending.

## 2026-08-23 — Replace AIDS dense common-recourse materialization

The serial CPU-only repair-v3 exceeded the 480 GiB cgroup limit and was killed;
exclusive scheduling cannot solve that scientific-stage memory requirement.
The next AIDS repair must preserve the exact DBSCAN/greedy semantics while
using disk-backed pair/vector shards, bounded exact radius-neighbor passes,
disk-backed union/find labels, and streaming coverage/centroid/medoid
aggregation.  Independent per-shard clustering, approximate neighbors,
OPTICS/BIRCH substitution, reduced candidate budgets, and reuse of the failed
root are forbidden.  Release requires legacy parity on core/border/noise/tie
fixtures, chunk-size invariance, crash/resume equivalence, and a bounded-RSS
dense stress gate in a fresh repair root.

This file records major design decisions for the counterfactual subgraph v3 project.

It should be updated whenever a meaningful implementation, algorithmic, or interface decision is made.

## 2026-08-23 — Adopt exactly six user-approved AIDS/Mut frozen-v4 cells

### Background

The generic four-by-four registry correctly rejected the combined legacy v4
tables because they are render-only artifacts and do not embed complete raw
method outputs, dataset/test-split hashes, RF checkpoint hashes, MolCLR hashes,
or selector-before-test evidence. The user has now explicitly designated the
exact `aids_mutagenicity_wnode_gcf_style_matched_aids_v4` values as frozen
results that must not be rerun. Silently weakening the generic adoption gate or
inventing the missing identities would make the matrix look more complete than
its evidence.

### Decision

Add one checksum-pinned exception named `USER_APPROVED_FROZEN_V4`, scoped only
to AIDS and Mutagenicity crossed with Ours, GCFExplainer, and GlobalGCE. The
five approved CSV/JSON files are read once each, hashed from those cached bytes,
validated against their frozen source/manifests, and copied byte-for-byte into
a fresh persistent source bundle. Exact source numeric strings are then
schema-projected into six fresh standardized roots; no metric, distance,
candidate order, rule order, threshold, or scientific value is recomputed.

The exception publishes its absent raw/dataset/split/oracle/MolCLR and
selector/test-order provenance as unavailable rather than synthesizing hashes
or claims. Registry and final-export code may waive only those enumerated
legacy-evidence requirements after independently revalidating the pinned
source copy, deterministic row projection, output hash closure, RF backend,
WNode/strict-flip contract, K grids, and exception scope. The resulting status
is `ADOPTABLE_PASS`, not an ordinary provenance-complete `FROZEN_PASS`.

CLEAR is source inventory only and is never adopted or mapped to ComRecGC. The
adoption emits a future-controller supersession manifest for not-started
duplicate work, but it neither mutates an existing controller nor stops a live
task. The active route is AutoDL-only; its paired Slurm file exits before the
documented CLI and exists solely for repository parity.

### Alternatives considered

1. Rerun all six methods despite the user's explicit frozen-result boundary.
2. Mark the combined render-only root as a generic PASS without a cell closure.
3. Fabricate legacy dataset, split, RF, or MolCLR hashes to satisfy the current
   schema.
4. Substitute CLEAR for the still-separate ComRecGC cell.

### Consequences

- Six A/M cells can enter the paper matrix as visibly exception-backed
  `ADOPTABLE_PASS` artifacts without changing any frozen value.
- Missing legacy identities remain machine-readable blockers/waivers rather
  than false provenance.
- Any source hash, scope, numeric projection, RF/action identity, or output
  closure change fails the exception.
- ComRecGC remains a separate native cell and ongoing ComRecGC work is not
  interrupted.
## [2026-08-23] Preserve the complete legacy GINE batch in BACE GCF replay

### Background

The first fresh 500-step `ordered_v2` replay was not equivalent to legacy.
The random-number-generator end states and the first selected graph were
identical, but the second transition diverged.  A replay of the first source
neighbourhood proved that CPU action construction, positional restoration, and
RDKit decoding preserved all 5,913 actions and graph tensors exactly.  Of the
555 valid decoded rows, however, only 311 canonical SMILES were unique.
`ordered_v2` deduplicated those rows and evaluated them in 256/55-row chunks,
whereas legacy evaluated all 555 rows, including duplicates, in one batch.
The official implementation uses raw graph-embedding bytes as transition
identity, so low-bit changes caused by the different GINE batch shape changed
the graph hash used by VRRW.

### Decision

Keep parallel RDKit decode only, restore its output by original position, and
then reproduce the complete duplicate-preserving legacy featurization and GINE
batch.  Do not perform per-SMILES prediction reuse or GINE chunking in the
equivalence path.  The importance cache may reuse only an exact, complete,
ordered batch with the same call context; on any partial miss it delegates the
original unmodified sequence.

Add diagnostic-only 50/100-step profiles for quick replay after code changes.
These runs can reject a patch early but can never authorize the 50,000-step
run: fresh 500/1000 replay plus the throughput/VRAM gate remains mandatory.

### Consequences

- CPU neighbour parallelism remains available without changing action order or
  the RNG stream.
- The unsafe canonical-SMILES deduplication, partial-row cache, and GINE
  chunking optimizations are removed from `ordered_v2`.
- The prior failed 500-step root and the running legacy full root remain
  immutable; no formal replay or full task is launched by this code change.
- The paired Slurm entry remains explicitly pinned to legacy semantics.

### Status

Accepted

## [2026-08-23] Gate BACE GCF acceleration by exact replay and shared-slot safety

### Background

The official BACE GCFExplainer VRRW retains the preregistered 50,000-step
budget but spends substantial wall time constructing edit neighbours, decoding
the same molecular graphs, and repeatedly scoring them through the frozen
GINE.  Its GINE call already batches each individual neighbourhood, so simply
raising GPU concurrency would not address the CPU/cache bottleneck and could
silently change random-walk order.

### Decision

Keep legacy mode as the default. Add an opt-in `ordered_v2` path that enumerates
the exact official actions in the exact official order, restores concurrent
pure neighbour results by input position, caches canonical graph/lineage
feature, GINE, and NeuroSED coverage results, chunks GINE inference, and
buffers progress reporting. It must not
consume randomness or modify official vendored source.

Refuse a full optimized run unless fresh legacy/optimized 500- and 1000-step
runs have identical canonical graph-transition/candidate/RNG digests and a
sequential A/B on one physical GPU is at least 20% faster with peak reserved
VRAM at or below 70%. Preserve `M=50000` and reject a gate/config fingerprint
mismatch.

Add explicit `exclusive`, `shared_lowmem_slot_0`, and
`shared_lowmem_slot_1` lock modes. Shared workers retain a shared advisory lock
on the legacy UUID file, so any legacy/new exclusive owner still excludes
them. At most two shared slots exist, compute PIDs must be attributable to
active slot metadata, and both scheduler and worker independently enforce the
70% reservation ceiling. CUDA MPS remains disabled.

### Consequences

- The already-running legacy GCF process is not stopped or modified.
- Low GPU utilization alone never authorizes colocation; manifests must opt in
  with a positive reservation and pass admission.
- Failed equivalence or throughput evidence cannot emit a PASS gate and the
  legacy result remains authoritative.
- This change is AutoDL-only. No HPC job was submitted or contacted, and the
  task-specific no-HPC boundary supersedes creation of a new Slurm wrapper.

### Status

Accepted

## 2026-08-23 — Attribute shared-GPU CUDA grandchildren without weakening PID safety

- A shared-slot launcher may legitimately execute a shell which then starts the
  CUDA Python process.  Exact equality with the recorded direct child PID was
  therefore too conservative and prevented the second slot from being used.
- Slot metadata now records the direct child's Linux procfs start-time tick.
  Every `nvidia-smi` compute PID must equal that child or reach it through a
  bounded, live `/proc/<pid>/stat` parent chain; the recorded start tick must
  still match at the root.
- Missing procfs data, broken/cyclic ancestry, PID reuse, and unrelated compute
  processes remain fail-closed.  This is attribution only; the two-task and
  70%-VRAM limits are unchanged.

### Status

Accepted

## [2026-08-22] Reconstruct an off-grid Mutagenicity GCF theta exactly

### Background

The repair-v1 Mutagenicity GCF held-out evaluator completed and audited all
`217 x 20 = 4,340` WNode pairs.  Its final exporter then raised
`StopIteration`.  The frozen matched protocol contains 601 empirical Figure-4
thresholds from 0 through 0.0535, but `theta_star=0.05` is deliberately not a
member of that grid.  The prefix artifact builder already computes the exact
theta-star metric separately from the same pair matrix; the reconstruction
helper incorrectly assumed it could always find that row in the Figure-4 grid.

### Decision

Preserve every frozen grid row and official-summary comparison unchanged.  If
theta star is absent from the grid, require the caller to supply the exact
full-K prefix row that `compute_prefix_artifacts` recomputed at theta star.
Accept that compatibility path only when its threshold is exactly theta star,
its provenance is `frozen_calibration_theta_star`, it contains the historical
official fields, `k == num_candidates`, its parent/candidate/valid-pair
identities equal the K20 grid, and coverage equals covered parents divided by
all parents.  The exporter and its independent final audit both pass their
freshly recomputed K20 prefix row.

Never choose the nearest grid threshold, interpolate, insert an extra Figure-4
row, alter the selector, or inspect test results to choose theta.  A missing,
ambiguous, incomplete, or inconsistent exact row raises a descriptive
`RuntimeError` rather than leaking `StopIteration`.

### Consequences

- The official 601-row reconstruction and exported Figure-4 schema remain
  unchanged.
- Table 2 and prefix metrics use the preregistered exact `theta_star=0.05`, not
  either neighboring empirical-grid value.
- RF predictions, WNode distances, candidate order, calibration freeze, test
  cohort, and scientific metrics are unchanged.
- The failed repair-v1 attempt remains immutable; recovery requires a fresh
  controller task/output root on a checkout containing this fix.

### Status

Accepted

## 2026-08-23 — Keep BACE ComRecGC as one trajectory and parallelize only pure preprocessing

### Background

The live 50,000-step BACE ComRecGC generation was profiled read-only at roughly
one saturated CPU core while the assigned GPU was almost entirely idle. Splitting
`0..49999` into independent generation-index jobs was considered as a possible
eight-way acceleration.

### Decision

Generation-index sharding is rejected. Pinned upstream ComRecGC consumes one
shared RNG stream and mutates shared graph, transition, candidate frequency/order,
coverage, and restart state at every step. Independent ranges or seeds are
different trajectories and have no exact frequency/order/lineage merge.

The only admitted parallel boundary is the pure native-graph decode and RDKit
featurization below a single random-walk producer. The opt-in engine uses a
bounded spawn process pool, returns results in input order, keeps CUDA and RNG
out of workers, and retains Frozen-GINE batching in the main process. Separate
bounded source/candidate caches bind graph content to source-sidecar lineage and
frozen feature/checkpoint provenance. Historical sequential preprocessing stays
the default.

An optimized 50k run is not scientifically released until fresh 500-step and
1000-step legacy/optimized replays both pass candidate order/frequency/importance,
graph-state, coverage, trajectory, and selected-action-trace parity. Diagnostic
runs are permanently paper-ineligible and PASS markers are written only after
the fail-closed audit.

### Consequences

- Existing live processes, outputs, and legacy checkpoints remain immutable.
- A legacy checkpoint is not a resume source for a fresh optimized command.
- One failed equivalence gate blocks the optimized full route but does not alter
  the legacy run.
- The resource scheduler may reserve a shared low-memory slot for the gates,
  but this change does not modify scheduler/lock semantics.

### Status

Accepted

## [2026-08-23] Release BACE GlobalGCE through the exact frozen-GINE bridge

### Context

The previous BACE route correctly preserved GlobalGCE's native attachment-aware
LHS-to-RHS action, but stopped before training because the official decoder is
continuous while the paper GINE consumes categorical molecular features. The
project owner explicitly approved an auditable differentiable bridge, while
forbidding an RF, surrogate classifier, trainable explainee, or action rewrite.

### Decision

Use the physical calibrated BACE GINE weights directly. Freeze every classifier
parameter, keep the classifier in evaluation mode, and exclude it from the
GlobalGCE optimizer. A straight-through expected-embedding view carries
gradients from the official classifier loss to soft node, adjacency, bond, and
decoder variables; its hard one-hot value is the normal project GINE forward.
The bridge may release full rule training only after proving hard-forward
parity, nonzero transformation gradients, zero classifier gradients, stable
checkpoint bytes, and finite outputs.

For the primary BACE run, freeze `min_freq=7` before execution. This is the
train-only value `round(0.02 * 360)` from the registered BACE frequency grid;
it is not selected from calibration or test curves. The full 360-parent source
cohort and 869-row native train vocabulary are both fail-closed inputs.

Rules remain native LHS-to-RHS tensors. Calibration enumerates every exact
labelled LHS match, applies the official attachment-preserving mask/RHS write,
scores hard sanitized products with the same frozen GINE in batches, and takes
the minimum legal WNode match. The selector carries the complete native rule
payload and uses an explicit aligned node/edge-transition fingerprint only for
its redundancy term; it never presents a rule as a deletion fragment or full
counterfactual graph. Test remains inaccessible until the calibration selector
is frozen. The final cell is standardized only from frozen terminal artifacts.

### Consequences

- GlobalGCE now has CPU parity preflight, exclusive-GPU bridge smoke and full
  training, four calibration shards, CPU freeze, four held-out shards, final
  freeze, and deterministic standardization tasks.
- Official GTGNN, RF, trainable/surrogate classifiers, full-graph replacement,
  deletion replacement, and test-based selection remain fail-closed.
- A bridge smoke PASS is necessary but is not a paper-cell PASS; final hard
  products and the complete calibration/test closure are still mandatory.

### Status

Accepted

## [2026-08-22] Isolate the two A/M ComRecGC retries in a six-task controller

### Background

The bounded repair-v1 controller successfully froze the Mutagenicity and AIDS
recovered-generation adoptions and their shared-protocol threshold contracts,
but both fresh standardized continuations encountered the same Git ownership
compatibility bug in `verify_comrecgc_checkout`. Retrying the larger repair-v1
graph would mix unrelated completed BACE/GCF work with this narrow code repair.

### Decision

Add `four_methods_four_datasets_am_repair_v2` with exactly four read-only
repair-v1 PASS source gates followed by two fresh A/M ComRecGC standardized
held-out jobs. Copy scientific paths from repair-v1's immutable task
definitions and cross-check them against the adopted generation roots; do not
accept a second hand-entered set of dataset, RF, MolCLR, distance, or upstream
paths. Exact-adopt the repair-v1 601-point threshold outputs without refitting.

Require the spec to name the reviewed Git-safety fix commit
`d8b113281d24e9340bfe2379e7451ffa8adff70a` exactly, verify that it is an
ancestor of the execution checkout while building, and repeat that check in
each runtime source gate. Share the global GPU UUID locks, cap CPU tasks at
two, retain fresh outputs, and omit BACE, GCFExplainer, TasteMolNet,
final-export tasks, and every old continuation guard.

### Consequences

- Passing repair-v1 source and threshold attempts remain immutable evidence.
- Only the two failed A/M standardized cells are recomputed.
- Test access remains downstream of an exact PASS threshold-freeze gate.
- A worktree missing the reviewed checkout fix fails before either GPU task can
  become READY.
- The AutoDL-only campaign has a paired Slurm wrapper solely for CLI parity;
  it is not submitted to HPC and `paper/` remains untouched.

### Status

Accepted

## [2026-08-22] Reuse process-scoped Git trust in every COMRECGC checkout gate

### Background

The shared COMRECGC validator correctly accepted the immutable migrated
checkout through a process-private exact-path `safe.directory`.  The
standalone checkout gate then repeated `git rev-parse HEAD` with a separate raw
subprocess.  Git rejected that second query as dubious ownership, so both AIDS
and Mutagenicity standardization failed after the shared validation had passed.

### Decision

Expose one semantic commit reader from the shared upstream module.  Both the
shared validator and the standalone gate use that reader, which retains the
private temporary global-config file, exact resolved checkout path, disabled
system config, and automatic cleanup.  Remove the gate's duplicate raw Git
implementation; do not change global Git configuration, checkout ownership,
or frozen payloads.

### Consequences

- Migrated-owner checkout verification is consistent across preflight and
  standardized continuation paths.
- Wrong commits, dirty tracked source, corrupt vendor manifests, and missing
  required files still fail closed.
- Scientific lineage, RF oracle, dataset, split, threshold, and evaluation
  semantics are unchanged.
- Existing failed attempts remain immutable and require fresh controller tasks.

### Status

Accepted

## [2026-08-22] Discover persistent AutoDL controllers in a loopback-only dashboard

### Background

The standalone read-only task board remained alive and generated a fresh API
timestamp, but it was launched without `--run`. It therefore continued to read
its hard-coded `autodl_three_lines_20260821_v1` tree and two fixed auxiliary
launchers while the active main and repair controllers wrote a different
persistent namespace. The page refresh label made a fresh HTTP sample look like
fresh scientific state even though the selected source was historical.

### Decision

Own the dashboard in the repository and discover every physical controller
directory under the exact `four_methods_four_datasets_continuation` namespace.
Reject symlinks and manifest/directory identity mismatches. Reuse the hardened
task/status reader, query `nvidia-smi` and UUID locks once per snapshot, and
display controller process identity, heartbeat age, queue state, task PID,
output, and failure/dependency reason. Keep status tokens and scientific IDs
unchanged while translating ordinary UI labels to Chinese.

The HTTP service remains GET-only and refuses every non-loopback bind. Friends
access it through independent SSH local-forwarding sessions. Direct public bind
is not an acceptable substitute for authentication/TLS, and sharing root
credentials is explicitly documented as full-machine access rather than
dashboard-only access. Browser refresh, source sampling, and controller
freshness are shown as distinct clocks.

The dashboard is an AutoDL-only persistent operations service. No new Slurm
wrapper is created for this long-lived Web listener; adding such a wrapper
would conflict with the explicit no-HPC scope and create an inappropriate
network service on a compute allocation. Existing read-only Slurm status CLI
parity is unchanged.

### Consequences

- Main, repair, and later fresh controllers can be viewed together without
  editing the dashboard launch command for every controller ID.
- A healthy page can no longer silently present the old three-line run as the
  current campaign.
- Multiple viewers are supported by loopback SSH tunnels without exposing the
  unauthenticated port publicly.
- The service remains independent of scientific controller ownership and
  cannot mutate or signal their processes.

### Status

Accepted

## [2026-08-22] Pin Mutagenicity GCF evaluation to the AutoDL controller Python

### Background

The four-by-four AutoDL continuation launches the frozen Mutagenicity GCF
calibration and held-out evaluators through their portable Slurm wrappers. A
non-interactive AutoDL worker does not expose the `conda` shell function after
reading `.bashrc`, even though the controller already supplies an absolute,
validated `AUTODL_PYTHON`. Calibration therefore exited before creating its
fresh output root; no scientific computation or partial artifact was written.

### Decision

When `AUTODL_PYTHON` is present, both shared evaluators require it to be an
absolute executable, prepend its `bin` directory to `PATH`, and fail unless
`command -v python` identifies the same file. They skip shell-level Conda
activation in that branch. When the variable is absent, the existing Slurm
path continues to source `.bashrc` and activate `smiles_pip118` unchanged.

### Consequences

- AutoDL uses the controller-pinned interpreter without relying on interactive
  shell initialization.
- HPC keeps its existing Conda activation contract.
- Calibration and held-out evaluation share one fail-closed runtime rule;
  candidate generation, RF/MolCLR inputs, thresholds, and metrics are unchanged.
- The failed AutoDL attempt remains immutable evidence and must be superseded
  only by a fresh continuation task and output root.

### Status

Accepted

## [2026-08-22] Recognize the original B13 shard freeze-boundary spelling

### Background

The frozen BACE Ours B12, B13, and B14 artifacts passed their scientific
gates, but the later paper-cell standardizer rejected every B13 verification
shard because it required `selection_frozen_before_test=true`.  The production
`bace_frozen_gnn_verification_shard_v1` writer used the earlier, more precise
field name `selector_frozen_before_split_load=true`.  The final B13 manifest
separately records `selection_frozen_before_test=true` and SHA256-binds the
physical B12 selection manifest.

### Decision

Accept the original shard spelling only for BACE Ours and only for the exact
`bace_frozen_gnn_verification_shard_v1` schema.  The compatibility path also
requires a test cohort, no calibration load, the B13 merge's
`test_used_only_after_freeze=true`, an exact physical B12 predecessor root,
the complete B12 ordered-rule/hash/candidate-source identity, and identical
policy, GINE, and MolCLR hashes.  The existing B13 top-level B12 path/size/SHA
binding remains mandatory.  A missing or false shard boundary, a different
predecessor, or any candidate/model identity drift fails closed.

### Consequences

- The immutable PASS artifacts are consumed without editing or rerunning B12
  or B13.
- The compatibility does not infer a freeze from stage names or controller
  status and does not weaken test-leakage protection.
- Newer baseline schemas still require their explicit
  `selection_frozen_before_test` evidence.

### Status

Accepted

## [2026-08-22] Isolate failed four-by-four closures in a bounded repair controller

### Background

The first four-by-four continuation retained valid PASS evidence, but four
recoverable closures needed fresh execution: BACE ComRecGC native generation,
Mutagenicity GCF calibration/test export, AIDS/Mutagenicity ComRecGC
standardization, and the artifact-only BACE Ours standardizer. Restarting the
original controller or editing failed attempts would mix execution code and
immutable evidence. The original controller can also remain alive with a GCF
GPU task, so a second scheduler must coordinate resource ownership globally.

### Decision

Build a new `four_methods_four_datasets_repair_v1` manifest containing only:

- the existing generic BACE ComRecGC native GINE task fragment plus its
  standardized cell terminal;
- Mutagenicity GCF calibration, held-out evaluation, and standardization from
  the exact passing v1 freeze;
- fresh Mutagenicity/AIDS ComRecGC threshold verification and standardized
  continuation from recovered read-only generation roots; and
- artifact-only standardization of the exact passing BACE Ours B14 root.

At build time and again as controller tasks, verify each adopted controller
terminal has state/gate PASS, one exact passing attempt root, required physical
files, and no writable procfs descriptor below it. Apply an equivalent small
five-manifest/payload-stat/writer closure to recovered generations; those
historical roots do not need or receive a synthetic bare `PASS`. Require
cross-manifest agreement on the claimed payload SHA-256, but leave the actual
large-payload SHA-256 computation to the scientific continuation's existing
exactly-once gate. Do not depend on the append-only experiment registry for
terminal identity. Write every repair under a fresh root and never include
Taste or final result rendering.

Run the repair controller in the same AutoDL runtime layout so both controllers
share UUID locks. Fix `runtime.max_cpu_tasks=2`. Do not copy a `continuation`
object into the repair manifest, so it does not acquire the old BACE
predecessor guard. Require the execution worktree to contain explicitly listed
AutoDL-Python, COMRECGC safe-directory, and BACE standardizer fix commits.

### Consequences

- The main v1 controller and repair controller may coexist without assigning
  the same physical GPU.
- Old PASS and FAILED attempts remain read-only evidence.
- No B0--B14 scientific task is repeated for BACE Ours; only its deterministic
  artifact exporter is retried.
- Test access remains downstream of a fresh calibration/threshold freeze.
- A missing PASS terminal, output-anchor mismatch, live writer, missing fix
  commit, or non-fresh destination blocks manifest publication.

### Status

Accepted

## [2026-08-22] Bind continuation predecessors to the source manifest namespace

### Decision

Derive the predecessor controller root from the exact persistent layout
`<control_root>/<source-namespace>/manifests/<manifest>` and the manifest's
validated `controller_id`. Verify the persisted controller snapshot binds the
same ID, manifest path, and SHA-256. The active alias continues to determine
only the fresh continuation root. Paths outside the control root, malformed or
symlinked namespace components, and snapshot mismatches fail closed.

### Consequences

- The four-by-four alias reads the completed `four_gpu_recovery` predecessor
  instead of incorrectly searching its new namespace.
- No namespace name is hard-coded, and path traversal cannot redirect adoption.

### Status

Accepted

## [2026-08-22] Standardize the frozen Mutagenicity GCF held-out export before matrix adoption

### Background

The AutoDL Mutagenicity GCF continuation writes its audited held-out artifacts
under `<task-output>/final`. The four-by-four registry accepts either a cell
root or its `standardized/` child, and the raw export retains both K=10 and
K=20 Figure-4 threshold blocks. Binding the held-out task directly would
therefore be neither a complete standardized cell nor a valid one-grid Figure-4
input. Its command also referenced the frozen-candidate task through a template
token without declaring that task as a direct dependency.

### Decision

Keep generation, candidate order, teacher predictions, WNode distances, and
held-out metrics unchanged. Add one CPU, manifest-only task after the held-out
and frozen-candidate tasks. It verifies source checksums and no-live-writer
evidence, copies K=1..20 prefix artifacts, selects the preregistered K=10 rows
from the saved 601-point Figure-4 matrix, normalizes only the public method
identity, and publishes the complete common cell schema with frozen RF,
MolCLR, cohort, threshold, and leakage hashes.

The held-out task now declares the frozen-candidate task as a direct dependency.
The final matrix builder rejects `mut_gcf_legacy_heldout` as a cell root;
`mut_gcf_legacy_standardized` is the only accepted binding.

### Consequences

- No GCF generation, prediction, distance, candidate selection, or threshold
  fitting is repeated.
- Controller dependency tokens are available when the held-out task launches.
- Figure 4 uses exactly the shared K=10 601-point empirical grid.
- A raw held-out container cannot masquerade as a paper-ready standardized cell.

### Status

Accepted

## [2026-08-22] Re-export Mutagenicity Ours on the frozen matched protocol

### Decision

Treat the exact historical 217-by-20 pair/match bundle as immutable evaluated
evidence, not as permission to rerun generation, the selector, the RF oracle,
or MolCLR. After the original-protocol adoption is checksum-verified, aggregate
those saved strict-flip WNode distances into a fresh standardized cell using
only the preregistered 601 thresholds from 0 through 0.0535, theta 0.05,
cost cap 0.0535, and Table/Figure K=10. Candidate order and every pair distance
remain byte-identical inputs.

Publish `FROZEN_PASS` only after a PASS-last fresh-root freeze closes Figure 3
K=1..20, the exact dense Figure 4 grid, Table 2 K=10, oracle/split/distance
identities, `test_used_for_selection=false`, and all artifact hashes. The raw
test CSV is not opened: this task reads only the already-frozen held-out pair
artifact. An old 14-point curve or any source/output tamper fails closed.

Place the CPU re-export after the manifest-only original adoption verification
and before the A/M inventory. Represent the four currently missing terminal
cells (Mutagenicity GlobalGCE and AIDS Ours/GCFExplainer/GlobalGCE) as distinct
static `command=null` blocked tasks so dependency graphs can reference them
without allocating CPU or GPU.

### Consequences

- Mutagenicity Ours no longer remains `STALE_METRIC` once this matched closure
  passes, while its old native-protocol root remains immutable evidence.
- The common threshold grid is reused without interpolation or test selection.
- Missing A/M cells remain explicit and cannot be mistaken for READY work.
- The paired Slurm entrypoint remains static CLI parity and is not submitted;
  the active campaign remains AutoDL-only.

### Status

Accepted

---

## [2026-08-22] Separate BACE scientific freezes from paper-matrix cell freezes

### Background

The new BACE Ours B14 and native GCFExplainer/ComRecGC final roots prove the
scientific classifier, selector, and held-out-test boundaries, but do not
directly implement the complete four-by-four standardized cell schema. Mapping
those raw terminal tasks into the final matrix would leave Figure 3, Figure 4,
Table 2, auxiliary CSVs, and their file-hash closure unproven.

### Decision

Add one artifact-only standardization layer after each eligible BACE terminal.
It follows SHA256-pinned selection, test, verification-shard, pair-matrix, and
final-metrics identities; validates the frozen BACE GINE and RF exclusion; and
replays only deterministic prefix aggregation. The raw held-out test CSV is
never opened, and no rule ordering, threshold, prediction, embedding, or
distance is recomputed. Missing auxiliary values are explicit `N/A`, never
numeric substitutions.

The final matrix builder rejects direct BACE mappings to `bace_b14_frozen`,
`bace_gcfexplainer_final_freeze`, or `bace_comrecgc_final_freeze`. It must bind
the corresponding `*_standardized` terminal task. GlobalGCE remains
`BLOCKED_CODE` and is not fabricated.

### Consequences

- Scientific and presentation freezes remain separately auditable.
- BACE standardized cells carry the same complete hash-closed schema as the
  other datasets.
- Calibration/test leakage cannot be introduced by the exporter.
- Any threshold-grid disagreement between methods remains visible to the
  cross-cell registry gate instead of being silently normalized.

### Status

Accepted

---

## [2026-08-22] Separate native GlobalGCE action parity from frozen-GINE training compatibility

### Background

Pinned GlobalGCE commit `157e65c2850bc787f229a1ee8c60564906b933f2`
does not define its local recourse as an arbitrary maximum-common-subgraph
replacement. Its `generate_fs_mask` / `get_graph_masks` mapping and
`concate_inputs_with_local_recourse` tensor write overwrite the labelled LHS
mask square with a reconstructed RHS, append required nodes, and preserve
attachments from matched nodes to the rest of the parent graph. The former
BACE route lacked a production implementation of this action.

The official training loss also differentiates through continuous decoded
feature, adjacency, and edge tensors into its own dense ground-truth GNN. The
frozen BACE GINE has a different input boundary: RDKit sanitization followed by
discrete categorical integer node and bond features. An exact calibrated GINE
forward is available after hard decoding, but that operation has no exact
gradient back to the continuous GlobalGCE RHS decoder.

### Decision

Implement the native action as an exact labelled LHS subgraph-isomorphism and
official-order RHS tensor overwrite. Verify it in production against functions
AST-extracted from a commit- and SHA-256-validated explicit upstream checkout,
so tensor parity does not require importing unrelated PyG training modules.
Preserve every distinct match, boundary attachments, atom/bond labels, and
provenance; fail closed on invalid shapes, asymmetric tensors, ambiguous or
colliding match identities, disconnected products, or RDKit sanitization
failure.

Provide a loaded-once frozen BACE GINE forward evaluator for native products,
including strict-flip provenance and a selector-freeze guard before held-out
test access. Keep full rule training statically blocked as
`BLOCKED_GLOBALGCE_FROZEN_GINE_DIFFERENTIABLE_RULE_TRAINING_UNAVAILABLE`.
Do not substitute the official GTGNN, RF, full-graph/deletion actions, or an
unreviewed straight-through estimator. Deployment must pass an explicit
official root because the final project bundle does not populate that checkout.

### Consequences

- Native application and calibrated-GINE forward evaluation can be audited
  independently of the unresolved training bridge.
- The generic controller runs one bounded CPU parity preflight and then reaches
  a static blocker; it exposes no READY GlobalGCE GPU task.
- Releasing the reserved priority-82 rule-training stage requires a reviewed
  differentiable design and new scientific tests, not a scheduler-only change.

### Status

Accepted

---

## [2026-08-22] Freeze TasteMolNet multiclass baseline adapters behind a fresh-release license gate

### Background

TasteMolNet is a three-class Bitter/Sweet/Tasteless dataset with Sweet as the
source class. Historical binary assumptions such as `1-label` would discard
valid Sweet-to-Tasteless counterfactuals. At the same time, the exact prepared
CSV still has no approved research-reuse basis, so even a correct adapter must
not cause candidate generation, inference, training, or test access.

### Decision

Add pure, training-free contracts for GCFExplainer, GlobalGCE, and ComRecGC.
All three require the same frozen three-class GINE and define a strict flip as
`pred_before == 1 and pred_after != 1`. Preserve GCFExplainer full-graph
actions. Merge GlobalGCE target-0 and target-2 native rules by exact action
identity before calibration, failing on hash/action disagreement. Preserve
ComRecGC global graph identity and unique pinned-upstream single-edit lineage;
parent metadata is provenance only.

Publish only terminal `BLOCKED_LICENSE_REVIEW`, CPU/manifest-only controller
tasks with `command=null` and no data-split declarations. Record an all-of
fresh-release contract in the fragment. The blocked-fragment builder refuses a
PASS license gate so an old blocked artifact cannot be relabeled in place; a
future authorized route must create a fresh fragment bound to the exact PASS
gate, frozen GINE hashes, shared split/MolCLR identities, and calibration-only
selector freeze before test.

The audit gate also publishes an explicit boolean `passed` plus its
`license_basis`/approval-file identity. This is the same fail-closed schema the
matrix registry consumes; public availability alone never populates those
fields. A PASS still requires a new runnable fragment and never mutates the
blocked fragment in place.

### Consequences

- No current TasteMolNet task can consume a GPU or access train/test bytes.
- Sweet-to-Bitter and Sweet-to-Tasteless remain first-class destinations and
  are reported overall and per rule.
- RF provenance and separate binary explainees fail closed.
- Approval of the data license is necessary but not sufficient to release a
  route; classifier and native-input provenance must also close.
- The static Slurm wrapper exists only for CLI parity and is not submitted by
  the AutoDL-only campaign.

### Status

Accepted

---

## [2026-08-22] Adapt native BACE baseline fragments at the controller boundary

### Background

The BACE GCFExplainer, ComRecGC, and fail-closed GlobalGCE route builder emits
a method-facing fragment with `task_id`, `argv`, resource objects, native
output roots, and file-marker lists.  The persistent generic four-GPU
controller instead requires `id`, `command`, scalar resources, passing-attempt
dependency tokens, immutable retry roots, input manifests, log markers, and
explicit calibration/test access declarations.  Feeding the native fragment
to the generic composer therefore failed schema validation or would have bound
downstream work to a mutable non-attempt path.

### Decision

Keep the native fragment unchanged and add a one-way generic adapter plus an
explicit `generic-task-fragment` CLI.  Translate every predecessor path to
`{dep_<task_id>_output}`, every task output to `{task_output}`, and every
expected output to a fresh `attempt-{attempt}` root.  Fold native auxiliary
checkpoint/cache paths into the owning attempt, require stage-specific files
and stdout markers, and use a non-primary `runner_dataset` so baseline work
cannot publish primary BACE recovery state.

Treat each baseline calibration selector as an explicit selector freeze and
allow only named baseline test verification, merge, and final-freeze stages
after that ancestor, still requiring frozen-selector and read-only-test flags.
Prioritize the two native train routes before priority-90 B11 shards, while
placing later four-way baseline verification after B11. For GlobalGCE, permit
one bounded CPU native-action parity preflight and follow it with a static
`command=null` task carrying the independent exact training `BLOCKED_CODE`;
never create a READY GPU task while that scientific blocker remains.

### Consequences

- OOM/transient retries cannot leave a downstream baseline task bound to an
  earlier failed attempt directory.
- The original method-facing fragment remains available for direct inspection
  and no method action semantics are changed.
- GCFExplainer and ComRecGC can share the work-conserving controller with B11
  without overwriting primary BACE stage state.
- GlobalGCE consumes only the bounded CPU parity preflight, then remains
  honestly blocked rather than consuming a GPU or being substituted with a
  full-graph/deletion action.

### Status

Accepted

---

## [2026-08-22] Gate the four-by-four paper renderer on complete standardized closure

### Decision

Add one presentation-only exporter for the four datasets and four named
methods. It reads `matrix_status.json` and emits per-dataset Figure 3,
Figure 4, and Table 2 files plus four-dataset panels only after the registry
proves 16/16 `FROZEN_PASS`/`ADOPTABLE_PASS` cells and each standardized root
independently closes its manifest identities, file hashes, calibration-only
selection, and test-after-freeze evidence. The exporter copies reported values;
it does not recompute science. Figure 4 uses raw empirical threshold rows with
no interpolation or smoothing, and Taste destination-distribution fields are
preserved.

`CLEAR` is rejected and cannot satisfy the ComRecGC cell. A partial or invalid
matrix produces only a non-numeric staging audit and a blocked marker. No zero,
blank-looking number, image, TeX table, or final PASS marker is emitted. The
export path is forbidden below `paper/`.

The generic controller task is named
`four_by_four_main_results_export`. It is CPU-only and depends on 16 distinct
cell terminal PASS task IDs plus a final matrix-audit task. Therefore a Taste
license block or any code-blocked cell naturally prevents final rendering
without consuming a GPU or manufacturing workload.

### Consequences

- The final renderer cannot hide missing cells behind presentation artifacts.
- Oracle, held-out split, distance, threshold, and strict-flip drift fail before
  plotting.
- The user-owned paper tree remains frozen until a separate authorized update.
- The paired Slurm wrapper remains static CLI parity; the active run is
  AutoDL-only.

### Status

Accepted

## [2026-08-22] Continue BACE B11--B14 through flattened exact adoptions

### Background

The first four-GPU controller completed B6--B10 but retained the original
MolCLR-parent preparation failure as immutable evidence.  A corrected MolCLR
preparation later passed in a separate fresh run.  B11 could not be released
inside the original dependency graph, and the controller intentionally rejects
binding one historical run to a multi-instance sharded task.  Therefore B8 and
B9 could not be adopted by copying their aggregate task definitions.

### Decision

Build a new continuation manifest only after the source controller is
quiescent and its B10 task and gate are both `PASS`.  Exact-adopt B6, B7, the
three passing original preparation runs, corrected MolCLR preparation, B10,
and every B8/B9 shard.  Represent the eight B8/B9 runs as eight ordinary
single-instance adopted evidence tasks; never synthesize an aggregate run or
rewrite source state.  Run B11--B14 as fresh tasks under a new controller ID,
output root, and WNode cache, using the corrected MolCLR node cache.

The continuation controller acquires the predecessor controller lock before
initialization and retains it for its full lifetime.  This prevents the old
controller from restarting concurrently.  A new optional boolean runtime flag,
`keep_alive_when_blocked` (default `false`), keeps a controller heartbeat and
poll loop alive after all non-Taste tasks become terminal without launching a
dummy task.  Generated BACE continuation manifests enable it explicitly.

### Consequences

- Every adopted run is revalidated against its exact launch spec, hashes,
  output contract, GPU identity, log marker, and immutable attempt path.
- The failed v2 MolCLR attempt remains failed and unmodified; only the separate
  corrected PASS run becomes B11 evidence.
- B11 starts only after a fresh continuation-output preflight and cannot share
  a writer namespace with the old controller.
- B13 retains its post-B12-only test dependency; TasteMolNet remains
  license-blocked and paper access remains forbidden.
- Invalid non-boolean keep-alive values fail manifest validation.

### Status

Accepted

---

## [2026-08-22] Gate the four-by-four paper matrix through read-only artifact evidence

### Background

The active paper matrix now spans four methods and four datasets, while the
persistent output trees contain formal runs, intermediate artifacts, failed
attempts, and legacy render-only presentations. Similar path names and complete
looking CSVs are not enough to prove oracle, split, distance, threshold, or
test-leakage compatibility. In particular, the historical combined four-method
presentation contains CLEAR, which must not be relabeled as ComRecGC.

### Decision

Add a dependency-light, read-only registry that always emits the exact sixteen
cells for AIDS, Mutagenicity, BACE, and TasteMolNet crossed with Ours,
GCFExplainer, GlobalGCE, and ComRecGC. Directory names are inventory hints only.
A passing cell requires self-identifying manifests, raw evidence, a passing
final artifact audit, complete Figure 3/Figure 4/Table 2 CSV contracts, frozen
dataset/test/oracle/MolCLR/threshold identities, strict flip, and explicit
test-selection exclusion. Cross-method identity disagreement within a dataset
fails closed. CLEAR remains an unassigned inventory artifact.

Record raw legacy evidence as a `generation_adoption_candidate` only when it
can plausibly feed deterministic re-evaluation; this is never a scientific or
paper PASS. TasteMolNet remains license-blocked unless an explicit PASS gate
contains a reuse basis for the exact data. Scan any number of output roots, but
hash only bounded evidence files and read large-model identities from frozen
manifests. Write the registry to a fresh root atomically and never mutate a
scanned artifact.

Treat a continuation container with top-level `final_gate.json`,
`_RUN_COMPLETE.json`, and last-published `PASS` plus a nested `standardized/`
freeze as one candidate. Promotion requires cross-checking the recorded nested
root, manifest hashes, freeze inventory, and final artifact audit; the layout
or marker name alone is not evidence. Require an explicit raw-completeness or
generation-adoption gate in addition to finding raw files.

Emit one evaluator-ready threshold JSON per dataset. Numeric `thresholds`,
`theta_star`, and `cost_cap` appear only when frozen expectations explicitly
bind them to a calibration source, or to an existing frozen protocol, with a
SHA-256 identity and `test_used_for_selection=false`. The registry never
reconstructs protocol values from test curves. Missing contracts remain
`MISSING_NOT_INFERRED`; test-derived or malformed inputs become
`INVALID_FAIL_CLOSED` and omit all numeric threshold fields.

The emitted evaluation contract fixes MolCLR Node-Wasserstein, strict flip,
K=1..20, Table 2 K=10, method-native actions, shared within-dataset classifier
identity, and the standardized export filenames. It is a schema and promotion
boundary, not a substitute evaluator and not permission to fabricate missing
results. Paired Slurm wrappers remain static CLI-parity files required by
repository policy; the active campaign does not submit them or access HPC.

### Consequences

- Legacy render-only figures and CSVs cannot silently enter the new matrix.
- A missing or stale cell is explicit and can be scheduled without rerunning a
  valid cell.
- Mixed RF/GNN classifier families remain visible while all methods within one
  dataset must share one exact classifier.
- The final figure/table gate can distinguish 16/16 complete from a valid
  partial audit without filling blocked cells with zeros.
### Status

Accepted

---
## [2026-08-22] Accept explicit CUDA indices in the shared MolCLR loader

### Background

The BACE frozen-GNN preparation route assigns a concrete GPU to MolCLR node
embedding work and therefore passes devices such as `cuda:0`. The shared MolCLR
loader accepted only `auto`, `cpu`, or the unindexed `cuda` spelling, so the
preparation step failed before loading its checkpoint.

### Decision

Treat `cuda:N` as part of the shared MolCLR device contract. Before constructing
the PyTorch device, require CUDA availability and verify that `N` is within the
visible device count. Continue to reject malformed device strings and invisible
indices rather than silently falling back to another GPU.

### Consequences

- AutoDL and Slurm callers can retain their explicit GPU assignment.
- A bad or unavailable CUDA index fails before checkpoint loading with a clear
  error.
- MolCLR embeddings, checkpoints, cache keys, distance semantics, and BACE
  scientific gates are unchanged.

### Status

Accepted

## [2026-08-22] Admit only the controller-owned tokenizer scheduling key

### Background

The four-GPU controller deliberately injects `TOKENIZERS_PARALLELISM=false`
to bound CPU pressure.  `exp_run` rejected that launch before scientific code
started because its credential-name detector matched the `TOKEN` substring in
`TOKENIZERS`.  Removing the scheduling limit or broadly weakening token-name
screening would make the controller or credential boundary less safe.

### Decision

Allow exactly the case-sensitive key `TOKENIZERS_PARALLELISM` through
`exp_run` environment parsing.  Keep the existing credential detector for all
other keys, including `TOKEN`, `API_TOKEN`, `SECRET`, `PASSWORD`, and
`AUTHORIZATION`; case variants of the scheduler key are not implicitly
accepted.

### Consequences

- Controller-owned bounded thread settings can reach a fresh AutoDL task.
- The failed pre-science launch remains evidence and needs no artifact rewrite.
- Credential-like environment keys continue to fail before a run root is
  created.

### Status

Accepted

## [2026-08-22] Bind downstream controller edges to passing attempt outputs

### Background

The first four-GPU controller could expose a single aggregate dependency path,
but the BACE Frozen-GNN downstream route consumes eight B8/B9 shard outputs and
four B11/B13 verification outputs independently.  It also materializes parent
shards from manifests produced by earlier controller tasks.  A retry can change
the immutable attempt directory, so a literal attempt-zero path or a controller
task directory is not sufficient scientific input evidence.

### Decision

Expand each task's immutable `expected_output` before expanding its command and
environment, and expose that resolved path as `task_output`.  Expose a numeric
`shard_index` separately from the stable `shard-000` instance ID.  For every
dependency instance in `PASS`, expose an instance-specific token such as
`dep_bace_b8_pool_base_shard_000_output`; the token always names the output of
the attempt that actually passed.  Expand a fixed-shard task's parent manifest
with the same dependency context before reading or partitioning it.

Represent B11 and B13 as non-publishing four-shard tasks followed by one
official deterministic merge.  The held-out test path is permitted only for
the post-B12 test-parent manifest, B13 shards, and B13 merge, each of which must
declare read-only access and have the frozen B12 selector as an ancestor.  B14
accepts only the frozen B12/B13 artifact roots and remains a manifest-only gate.

### Consequences

- OOM retries cannot leave a downstream consumer bound to a failed attempt.
- B10/B11/B13 merges receive exact immutable shard roots rather than directory
  scans or aggregate task folders.
- The controller can materialize train, calibration, and post-freeze test
  shards from upstream outputs without hard-coded attempt paths.
- Test bytes remain unreachable until B12 has passed and frozen its selector.

### Status

Accepted

---

## [2026-08-22] Freeze the BACE GNN downstream route at stage boundaries

### Background

The first AutoDL BACE frozen-GNN driver intentionally stopped after a real GNN
scoring diagnostic because the historical PPO, pool, verification, selector,
and held-out evaluator were tied to Morgan-RF provenance.  B6-v2 and B7 now
provide a train-only GNN-reward policy, but B8--B14 still need a route that can
reuse the stable chemistry, generation, GNN, and MolCLR kernels without
promoting any RF artifact or opening test before selector freeze.

### Decision

Run B8 and B9 as eight fixed train-parent shards.  Parent assignment is the
position of `parent_id` in the globally sorted source-parent list modulo four,
and therefore never changes with available GPU count.  Each shard binds the
current final LoRA adapter config and weights by path, size, SHA256, and the B7
on-disk `policy_checkpoint_hash`; B10 requires all eight PASS manifests, equal
policy/GNN identities, equal parent closure, and equal within-stage generation
config before a deterministic merge.

Run B11 and B13 as complete parent-by-rule-by-match hard-deletion shards.  The
frozen BACE GINE predicts parents and residuals in batches; every exact valid
match is retained; a pair uses the minimum MolCLR Node-Wasserstein distance
among connected strict flips and otherwise has selection distance `+inf`.
B11 reads calibration only.  B12 reuses the oracle-neutral prefix selector,
freezes exactly 20 ordered rule IDs, rule hashes, all K=1..20 prefixes, and its
calibration-derived threshold/config identities.  B13 may resolve and open the
held-out test split only after validating that B12 freeze.  A dedicated CPU
gate then freezes `test_parent_ids.frozen.json` for controller shard
materialization; test identities are never prepared alongside B7.  B14 accepts no raw
split argument and checks only frozen B12/B13 manifests and declared artifact
identities.

Permit four bounded, read-only actions after B6-v2 while B7 trains: calibration
GNN-before cache, calibration original-graph MolCLR cache, fixed train and
calibration shard manifests, and fresh-output/disk preflight.  These actions
never load a policy checkpoint, generate candidates, select a rule, or open
test.  Every scientific stage publishes data and manifests first and an atomic
fsynced `PASS` marker last; a failed fresh invocation retains atomic
`FAIL.json`/`FAILED` evidence.

### Consequences

- RF and unknown provenance fail closed at every promotion boundary.
- B8/B9 can consume only the exact adapter bytes frozen by B7.
- B11 and B13 preserve all match evidence while the selector sees one
  deterministic minimum-distance pair value.
- Test cannot influence policy, candidate generation, calibration thresholds,
  variant choice, rule order, or hashes.
- B14 can be rerun without reopening either raw calibration or raw test data.
- The four-GPU controller receives a foreground command/output contract and
  remains the sole owner of locks, retry policy, logs, and process lifetime.

### Status

Accepted

---

## [2026-08-09] Expand BACE GCF native ranking from the frozen VRRW pool

### Background

The completed 50,000-step BACE GCFExplainer VRRW artifact contains 45,488
saved candidates and 38,637 model-counterfactual candidates. The first native
summary inherited upstream's one-candidate-per-parent shortlist size and ranked
only 360 candidates. Chemistry and RF filtering of those ranks retained 2 of
the required 20, even though the much larger immutable VRRW pool was never
examined by the summary.

### Decision

Keep the completed VRRW artifact, GNN, NeuroSED distance, theta, teacher, and
official greedy objective fixed. Permit BACE summary generation to include all
saved model-counterfactual candidates when the explicit candidate limit is
zero. Preserve the exact greedy tie behavior and fast-forward only its
mathematically deterministic all-zero coverage tail. Store large ranked graph
sequences as hash references to the SHA-frozen VRRW graph map.

Filter the resulting native ranks sequentially until 20 unique, chemically
valid RF-target candidates are found or the pool is exhausted. Emit structured
candidate attrition and fail closed when insufficient. Candidate chemistry,
native order, RF predictions, and WNode values never alter generation or native
ranking; no repair, copying, rank compaction, or backfill is allowed.

Treat native rank as the unique summary-row identity. Structural candidate IDs
may repeat when distinct VRRW records encode the same graph; these records stay
in sequence and are accounted for by the canonical-SMILES deduplication audit.

### Consequences

- The 50,000-step VRRW job is reused rather than repeated.
- BACE source-graph codec identity remains independently gated.
- The final selected CSV keeps the existing paper evaluator schema.
- Expanded internal summary storage does not duplicate hundreds of megabytes
  of graph objects.
- AIDS and Mutagenicity GCFExplainer behavior is unchanged.

### Status

Accepted

---

## [2026-08-08] Defer active-head transition eviction within one COMRECGC move

### Background

The fixed AIDS/HIV project full run reached step 34,939 before the pinned
upstream `move_to_next_graph` raised a transition lookup `KeyError`. Upstream
materializes transitions for all current random-walk heads, then the selected
lead reinforces a candidate. At full candidate capacity that reinforcement can
evict a non-lead current head and delete its transition before the same move
consumes it for follower matching.

### Decision

Wrap only project full generation's upstream transition dictionary. A deletion
of a current move head is deferred until the wrapped move returns; then the
deletion is applied if the candidate remains evicted. Non-active transition
deletions remain immediate. A missing lookup raises a diagnostic error with
step, head, seed, graph hash, transition size, and graph-cache size. The wrapper
adds no random calls and leaves neighbor enumeration, candidate capacity,
importance, DBSCAN, and ranking untouched. It serializes as a plain dictionary
and emits bounded progress diagnostics at step 1,000 and every 10,000 steps.

### Consequences

- Every current head retains its already-built transition for the complete
  official move that consumes it.
- Evicted historical transitions are still released after that move, avoiding
  an unbounded full-run cache.
- Smoke and native routes keep their previous transition behavior.
- Cross-job random-walk resume remains unsupported; retries use a fresh
  versioned output with the same frozen scientific parameters.

### Status

Accepted

---

## [2026-08-04] Restore the audited AIDS GCF-style WNode presentation contract

### Background

The first combined AIDS and Mutagenicity renderer used the wrong AIDS primary
threshold and a shortened threshold grid. A previously audited AIDS renderer,
its source CSVs, and its Figure 3, Figure 4, and Table 2 outputs establish the
correct presentation contract.

### Decision

Read AIDS Figure 3 directly from the audited theta-0.05 80-row CSV and AIDS
Figure 4 directly from the audited K=10 2404-row CSV. Figure 4 retains all 601
empirical thresholds per method from 0 through 0.0535, with no interpolation.
Compute only the Table 2 presentation reduction from the four frozen WNode
pair-detail roots at K=10 and theta=0.05, then require exact agreement with the
audited four-method values. Reuse the original renderer's serif typography,
method styles, four-column panel layout, sparse markers, and booktabs-like
table drawing.

Expose two explicit Mutagenicity presentation profiles. `native` reads its
frozen dataset-specific threshold and seven-point grid. `match-aids` aggregates
the already-saved 217x20 strict-flip pair matrices at theta=0.05 and on the
same 601-point 0..0.0535 grid used by AIDS. This second profile recomputes no
distance or prediction and preserves each method's frozen candidate order.
The profiles write separate output roots and can coexist.

The renderer may read and aggregate saved artifacts, but it must not recompute
embeddings, distances, teacher predictions, candidate selection, or candidate
order. Both profiles share distance semantics, strict-flip semantics, method
order, and visual design; their threshold relationship is explicit in the
profile name and manifest.

### Consequences

- AIDS Figure 3 and Figure 4 can no longer be silently rebuilt from another
  threshold or threshold grid.
- The four AIDS Table 2 values are protected by a numerical regression Gate.
- Native-threshold and AIDS-matched Mutagenicity presentations cannot
  overwrite or masquerade as one another.
- Non-Wasserstein and deletion-fragment provenance remains rejected.
- The final manifest records that distance, teacher prediction, ranking, and
  selection were not recomputed.

### Status

Accepted

---

## [2026-08-07] Isolate COMRECGC full preflight artifacts and trace untyped edits

### Background

The first end-to-end full submission exposed two engineering-only failures.
The native AIDS wrapper wrote its preregistration into the fresh generation
root before invoking the generation runner, so the runner correctly rejected
the now non-empty directory.  Project AIDS reached the upstream random walk,
but pinned COMRECGC edits `edge_index` without consistently updating the
unused bond-label `edge_attr` sidecar; the project trace identity therefore
rejected the first stale sidecar even though upstream does not consume it.
Early full throughput also showed that the fixed 50,000-step protocol can
exceed the wrappers' 48-hour request, while the assigned A800 QOS permits seven
days.

### Decision

Store native-full preregistration under the append-only automation state root,
outside the fresh generation output.  Keep the chemistry-facing typed graph
identity strict, but make the side-effect-free upstream action trace and its
parity/replay identity explicitly node-and-adjacency based.  This matches the
pinned algorithm's actual graph semantics and never repairs or mutates its
candidate payload.  Request the QOS-supported seven-day wall time only for
full generation wrappers; all seeds, parents, steps, heads, candidate budgets,
importance, clustering, and ranking parameters remain unchanged.

### Consequences

- Native preflight evidence no longer invalidates the generation fresh-output
  gate.
- A stale upstream bond sidecar remains visible to strict chemistry checks but
  cannot crash read-only random-walk tracing.
- Full generation has enough wall time without reducing the preregistered
  50,000-step budget or changing scientific behavior.
- A later end-to-end retry may adopt each explicitly enumerated, passing smoke
  Gate by hashing every file in that stage root. Full stages remain
  non-adoptable through this interface.
- Native AIDS resolves the trusted tensor payload before entering the pinned
  upstream working directory and constructs the identical official GNN from
  that payload's frozen feature dimension, avoiding an unsafe second cache
  read under PyTorch 2.6 without modifying upstream code or model weights.

### Status

Accepted

---

## [2026-08-07] Distinguish COMRECGC rank slots from reused source medoids

### Background

The frozen Mutagenicity common-recourse result contains four distinct official
clusters, while three cluster slots legitimately select the same original
candidate graph as their medoid.  The first unified-evaluation adapter treated
the source candidate ID as the rank-slot identity and rejected this valid
lineage before RF or WNode evaluation.

### Decision

Keep each original candidate ID unchanged as `source_candidate_id`, and assign
an evaluator-only `candidate_slot_id` from the immutable official cluster rank.
Require rank and cluster IDs to be unique and preserve repeated source candidate
IDs in their exact official order. Identical repaired SMILES are scored once by
the shared evaluator and that exact result is expanded back to every official
slot; rank slots are never deduplicated, compacted, or backfilled.

### Consequences

- Reused medoid graphs remain separate official prefix slots with explicit
  shared lineage.
- Shared evaluator pair keys are unambiguous, while output manifests retain
  hashes for both slot order and original candidate order.
- Candidate graphs, chemistry repair, RF/WNode semantics, DBSCAN, and official
  greedy ordering are unchanged.

### Status

Accepted

---

## [2026-08-06] Scope trusted TU AIDS loading and preserve untyped COMRECGC identities

### Background

The retry3 native AIDS audit exposed two compatibility properties of the pinned
upstream artifacts. PyTorch 2.6+ requires an explicit trusted-pickle boundary
for the 1,837 cached PyG objects, and upstream graph edits can leave the TU bond
label sidecar shorter than the edited adjacency even though COMRECGC consumes
only node labels and `edge_index`. A separate Mutagenicity wrapper also checked
an obsolete success marker after producing a complete chemistry audit.

### Decision

Load the frozen AIDS cache only in one compatibility-scoped child process and
materialize a tensor-only, weights-only-reloadable payload whose cache inventory
SHA256 is checked before and after loading. Native model, NeuroSED, RF, and
MolCLR processes do not inherit the compatibility variable. Keep the general
typed graph hash strict, while native AIDS DBSCAN uses an explicit canonical
`official_untyped_x_edge_index` identity that ignores the unused stale sidecar.
Allow an exact, hash-frozen completed smoke stage to satisfy a later `afterok`
dependency without rerunning that stage. Fix the chemistry wrapper to recognize
the project-wide engineering-pass marker.

### Consequences

- No upstream source, candidate tensor, candidate order, distance, DBSCAN
  parameter, or greedy ordering changes.
- A completed stage can be adopted only from COMRECGC output roots and is
  rehashed on every refresh; any mutation blocks continuation.
- Failed Slurm history remains intact while retries use fresh versioned roots.

### Status

Accepted

---
## [2026-08-06] Gate COMRECGC end-to-end promotion and stream full lineage

### Background

The recovered smoke trace proved candidate/order/frequency and discrete
importance decisions unchanged, but materializing every candidate's complete
action path and repaired graph would multiply memory at the frozen full budget
of 50,000 steps, five heads, and capacity 100,000. The project owner also
authorized full execution after, and only after, each dataset's engineering
smoke Gate passes.

### Decision

Freeze an authorization file and exact job DAG at the project commit. Require
the dataset-specific smoke Gate before any full node and use `afterok` for every
edge. AIDS native full and project AIDS/HIV full remain separate; only the
project route can enter paper artifacts. Scientific empty output is a passing
execution with coverage zero and conditional cost unavailable.

For full project generation, stream selected transitions to atomic bounded
JSONL chunks, write a compact candidate index, and reconstruct one lineage at a
time during chemistry audit. Retain repaired graph objects only for original
official medoids. Keep inline lineage for existing smoke/recovery artifacts.

### Consequences

- Full execution cannot precede its dataset's smoke engineering Gate.
- Empty clusters, invalid repaired medoids, or zero strict flips do not trigger
  parameter changes, rank backfill, seed search, or engineering retries.
- Trace capture adds no RNG calls and does not alter neighbor enumeration,
  importance, DBSCAN inputs, candidate order, or official greedy rank.
- AIDS uses the same 1283-parent CSV and frozen dense WNode threshold grid as
  the existing final AIDS evaluator; Mutagenicity uses its frozen 217-parent
  test cohort and threshold artifact.

### Status

Accepted

---
## [2026-08-06] Harden COMRECGC retry3 as an authorization-scoped smoke replay

### Background

The retry2 recovery jobs exposed four control-plane risks before retry3: the
trusted AIDS PyG cache compatibility variable was exported process-wide, the
Mutagenicity smoke gate used `afterany`, the recovery driver did not consume a
run-specific authorization artifact, and trace parity did not freeze the
discrete model-CF and DBSCAN input decisions.

### Decision

Require an exact `authorization.json` bound to the current project and pinned
upstream commits before any retry3 submission. Limit that authorization to two
AIDS and four Mutagenicity smoke jobs, and require separate positive booleans
for every full submission path. Use `afterok` throughout the Mutagenicity smoke
chain. Audit the immutable AIDS cache before and after loading, and scope
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to only the Python process that loads that
cache. Extend trace parity with exact importance-mask, model-CF set/order, and
DBSCAN input set/order gates while retaining the audited `1e-6` floating-point
tolerance.

### Consequences

- Retry3 cannot auto-promote to full and cannot exceed the authorized 2/4/6
  job limits.
- Existing retry jobs are adopted from state and the experiment registry
  instead of being resubmitted after a client interruption.
- A writable or symlinked AIDS cache blocks only the AIDS chain with explicit
  evidence; it does not weaken trusted pickle loading or block the independent
  Mutagenicity smoke chain.
- Random-walk candidates, importance math, DBSCAN parameters, greedy order,
  and frozen artifacts are unchanged.

### Status

Accepted

---

## [2026-08-06] Separate COMRECGC smoke interface evidence from final yield

### Background

The controlled COMRECGC smoke budget produced a chemically valid AIDS medoid
that the RF classified as `1 -> 1`, so the formal strict-flip candidate CSV was
correctly empty. Requiring that final CSV as the only WNode input prevented the
smoke run from demonstrating that the distance interface itself works.

### Decision

For smoke only, when the frozen strict-flip CSV is empty, evaluate decoded and
RF-scored medoids in unchanged official native-rank order as an explicitly
labeled interface cohort. The evaluator still uses `strict_flip`, so non-target
medoids contribute zero coverage. The cohort is marked
`smoke_interface_only=true` and `eligible_for_final_results=false`. Full runs
continue to require exactly 20 frozen strict-flip candidates and never use this
fallback.

### Consequences

- Smoke can independently verify RF and WNode plumbing without claiming
  scientific candidate yield.
- `candidate_yield_gate_passed` remains false when no strict-flip medoid exists.
- Invalid chemistry is never admitted: at least one decoded, RF-scored medoid
  is required before the smoke interface cohort can be formed.

### Status

Accepted

---
## [2026-08-06] Treat COMRECGC node lineage as authoritative during export

### Background

Project Mutagenicity source graphs already contain `gcf_node_origin`. Official
COMRECGC graph edits clone that field, while the project-owned neighbor wrapper
updates `comrecgc_node_origin` as nodes are added or removed. The exporter was
therefore passing stale source lineage to the shared generated-fullgraph codec.

### Decision

Immediately before chemical decoding, always expose the updated
`comrecgc_node_origin` as `gcf_node_origin`. Missing COMRECGC lineage remains a
hard failure. This synchronization changes metadata only; it does not modify
node features, edges, official ordering, chemistry policy, or RF strict-flip.

### Consequences

- Generated Mutagenicity medoids are checked against their actual edited node
  count instead of stale source-node lineage.
- The shared fullgraph codec and all scientific validity gates remain intact.
- Native COMRECGC graph content and upstream source remain unchanged.

### Status

Accepted

---

## [2026-08-06] Keep COMRECGC native reproduction separate from project adaptation

### Background

COMRECGC's `--dataset aids` means the filtered PyG TU AIDS benchmark, whereas
the project paper protocol uses `data/raw/AIDS/HIV.csv`. The two datasets,
labels, graph IDs, and parent universes are not interchangeable. The upstream
repository also has no clear redistribution license.

### Decision

Pin upstream commit `122f9341a360e9f06bb58a2f5823bb596021f6bf` in an ignored
external checkout and do not vendor or modify it. Maintain a native TU smoke
route for reproducibility only. For project results, inject the frozen HIV.csv
and strict Mutagenicity graph artifacts, reuse their validated GNN and
GREED/NeuroSED checkpoints, map labels explicitly to COMRECGC's internal
source-0/target-1 convention, and preserve upstream edit, transition,
reinforcement, teleportation, clustering, and greedy ordering. Represent a
common-recourse cluster by its nearest real source-to-counterfactual pair, not
by a fictional embedding-center graph. Apply the unified RF strict-flip and
MolCLR-Node-Wasserstein evaluator only after candidate order is frozen.

### Consequences

- Native TU metrics cannot appear in AIDS/HIV or Mutagenicity project figures.
- Upstream source and algorithm semantics remain separately auditable.
- The project records model and dataset fingerprints and never uses test data
  to generate, rank, or select COMRECGC candidates.
- Smoke/full jobs are dependency-chained and resumable; full remains gated by
  successful dataset-specific smoke jobs.
- Existing CLEAR, GCFExplainer, GlobalGCE, Ours, RF, WNode, and paper artifacts
  are not modified or overwritten.
- The one permitted interface-only smoke budget retry raises generation from
  50 steps/64 samples to 100 steps/128 samples after both first runs produced
  a single common-recourse representative and no unified-evaluation candidate.
  This remains within the predeclared smoke bounds and does not alter full
  parameters, clustering thresholds, candidate order, or final evaluation.

### Status

Accepted

---

## [2026-08-04] Re-render the combined AIDS/MUT figure set from frozen CSV values

### Background

The completed `matched_aids_v2_copy` presentation directory contains the three
frozen CSV inputs for Figure 3, Figure 4, and Table 2. Its current Figure 3 CSV
contains intentional presentation-value edits and therefore no longer matches
the older output hash recorded by the copied manifest. Recomputing pair
distances or rankings would discard those explicit CSV values.

### Decision

Provide a presentation-only CSV replay command that validates the complete
two-dataset/four-method grids, renders with the existing GCF-style functions,
and records the current file hashes as the numeric source of truth. The replay
does not read pair details, models, predictions, or candidate rankings. V3 uses
a 90 percent Figure 3 coverage ceiling because the frozen MUT curves exceed 80
percent; the existing renderer keeps its 80 percent default for older runs.

### Consequences

- `matched_aids_v2_copy` remains read-only and V3 is written to a new root.
- Figure 3 and Figure 4 data CSVs and Table 2 CSV are copied byte-for-byte into
  V3 after validation.
- A stale copied-manifest hash is retained as advisory evidence but cannot
  replace the current CSV hash or values.
- No scientific result, distance, teacher prediction, or ranking is recomputed.

### Status

Accepted

---

## [2026-07-24] Preserve explicit PPO CLI values during config merge

### Background

The PPO config adapter inferred whether an argument was explicit by comparing
its parsed value with the parser default. Mutagenicity full PPO explicitly
passed its task output directory, but that value equaled the Mutagenicity
parser default and was therefore replaced by the generic HPC
`outputs/hpc/rl_checkpoints` config path.

### Decision

Determine explicit arguments from the actual argv option strings. Config values
may only replace destinations absent from argv and still at their parser
defaults. The Mutagenicity adapter additionally audits all task-critical CLI
fields and prints the CLI, config candidate, post-config, and resolved output
paths before creating the output directory. An explicit Mutagenicity output
that resolves to the legacy shared RL directory is rejected.

Keep one full epoch as the wrapper default by deriving
`ceil(1448 / 64) = 23` updates. An explicitly supplied `MAX_UPDATES` remains an
override rather than being rejected by the Python adapter.

### Consequences

- Parameter precedence is now CLI, then config, then parser default.
- Existing AIDS/HIV PPO entrypoints retain their CLI and config interfaces.
- Mutagenicity full and smoke paths cannot be silently redirected by
  `configs/hpc.yaml`.
- Reward, teacher, policy initialization, sampling order, and optimizer
  semantics are unchanged.

### Status

Accepted

---

## [2026-07-24] Define stable-PPO size ratio from the deleted fragment

### Background

The stable decoded-chemistry reward computed its size-window term from the
normalized raw/core fragment before parent-subgraph projection. When projection
replaced an oversized generation with the fragment actually used for hard
deletion, the reward log and candidate pool still reported the raw/core ratio.

### Decision

After direct matching or projection is resolved, compute the stable-PPO
`atom_ratio` as final-fragment heavy atoms divided by parent heavy atoms.
Recompute the size-window reward from that same ratio before the teacher
confidence gate and PPO update. Preserve `raw_atom_ratio` only as a diagnostic,
record both atom counts and `atom_ratio_source=final_fragment`, and fail if a
claimed final parent substructure produces a ratio outside `(0, 1]`.

For Mutagenicity full PPO, validate every 100 updates by default instead of
every 5 updates. Keep the existing final validation pass so a non-divisible
last update is still evaluated and considered for best-checkpoint selection.

### Consequences

- Candidate-pool rows, stable update metrics, and the size reward now use one
  final-fragment ratio.
- Oversized raw generations can retain `raw_atom_ratio > 1` for diagnosis
  without corrupting the PPO size term.
- The shared stable wrapper applies the corrected semantics to Mutagenicity and
  remains compatible with AIDS/HIV callers.
- Teacher scoring, strict flip, CFDrop, PPO loss, KL control, learning rate,
  parent sampling, and policy initialization are unchanged.

### Status

Accepted

---

## [2026-07-23] Continue exactly one existing PEFT adapter on Mutagenicity

### Background

The Mutagenicity continued-SFT smoke run completed successfully, but PEFT
warned that the model already had a `peft_config` and could receive multiple
adapters. The previous runtime check only required at least one trainable
parameter, so it could not prove that the ChemLLM base was unwrapped or that
exactly one existing AIDS adapter was active.

### Decision

Load the adapter configuration from the AIDS `checkpoint-500`, verify that its
declared ChemLLM base matches the requested base model, load that base without
an adapter, and invoke `PeftModel.from_pretrained(..., is_trainable=True)`
exactly once. Before training, require exactly one configured and active LoRA
adapter, positive trainable LoRA parameters, and zero trainable non-LoRA base
parameters. Persist this result as `adapter_audit.json` and include it in the
training log and report.

### Consequences

- Continued SFT updates the recovered AIDS LoRA weights rather than creating a
  second random adapter.
- Existing or ambiguous adapter state on the requested base fails before
  Trainer construction.
- Data, optimization hyperparameters, PPO, selectors, WNode, baselines, and
  evaluation semantics are unchanged.

### Status

Accepted

---
## [2026-07-24] Isolate stable-PPO validation generation RNG

### Background

The ChemLLM/InternLM2 PEFT stack rejects a `torch.Generator` passed through
`model.generate()`, because this Transformers version validates it as an
unused model keyword. Mutagenicity stable-PPO therefore failed during its
first validation pass after completing a successful rollout and update.

### Decision

Do not forward `generator` to validation `generate()`. Derive a deterministic
seed from the run seed, validation step, and batch index, then run only that
validation batch inside `torch.random.fork_rng`. Seed CPU and available CUDA
generators inside the context so their previous states are restored afterward.
Leave PPO rollout generation and all training, reward, teacher, sampling, and
optimization logic unchanged.

### Consequences

- Validation remains reproducible for the same seed, step, and batch order.
- Validation sampling does not advance the RNG state used by later PPO
  rollouts.
- The fix is shared by Mutagenicity and AIDS/HIV stable-PPO validation.

### Status

Accepted

---
## [2026-07-23] Adapt the shared stable PPO loop to Mutagenicity 1 -> 0

### Background

The AIDS stable decoded-chemistry PPO loop already implements generation,
chemistry validation, parent projection, deletion-based RF scoring, PPO
clipping, adaptive KL, and value-head updates. Its update counter is
DataLoader-batch based, however, and its generic counterfactual scorer defines
a flip only from the post-intervention prediction. Mutagenicity requires an
auditable source-label `1` to target-label `0` direction and one complete
no-replacement pass over 1,448 train parents.

### Decision

Keep `run_stable_decoded_chem_ppo_loop()` as the only PPO algorithm and add an
optional run observer plus stable `molecule_id` propagation. The
Mutagenicity adapter deterministically orders the teacher-correct train view,
uses a nominal rollout batch of 64, derives `ceil(1448 / 64) = 23` updates,
and stops after the first exhausted DataLoader pass. Load the pure ChemLLM
base plus exactly one continued-SFT `checkpoint-200` LoRA for both policy and
frozen reference policy. Use a Mutagenicity-specific scorer with strict flip
`pred_before == 1 and pred_after == 0` and `cf_drop = p1_before - p1_after`.

### Consequences

- Existing AIDS callers do not pass the observer and retain their current
  sampling and reward behavior.
- The decoded stable loop still does one optimizer update per PPO epoch for
  each rollout batch; its legacy mini-batch and gradient-accumulation CLI
  values do not subdivide this local loop.
- Parent coverage, update metrics, candidates, validation samples, model
  freezing, teacher direction, and checkpoint selection are persisted as
  first-class run artifacts.
- Calibration and test data are rejected before model loading.

### Status

Accepted

---

## [2026-07-15] Plot and report theta-covered conditional FGW cost by default

### Background
The GCF-style report CSV already contained the correct theta-covered
conditional median cost, but Figure 3 still plotted the unconditional
`median_cost`. This made the plotted GlobalGCE K=10 cost exceed the displayed
theta even though the correctly conditioned value was available.

### Decision
Make `theta_covered_conditional_median_cost` the explicit default for Figure 3
and final Table 2. Record the selected Figure 3 value as `plotted_cost`, leave
zero-coverage points as NaN, and assert that every finite default plotted/table
cost is no greater than theta. Emit separate K=1..10 and K=1..20 Figure 3 files.
The final `table2_global_recourse` artifact contains only method, coverage, and
theta-covered conditional median cost; legacy audit-oriented fields remain in
the compatibility CSV.

### Consequences
- Figure 3 and Table 2 now use the same strict-close conditioning event as
  coverage at the stated theta.
- Applicable-parent and unconditional medians remain available only through
  explicit metric parameters or audit columns.
- Figure 4 coverage, strict flip, parent cohort, candidate order, and all saved
  Node-FGW distances are unchanged.

### Status
Accepted

---
## [2026-07-23] Continue the AIDS SFT-v3 adapter on Mutagenicity

### Background

Mutagenicity now has fixed teacher-consistent SFT train/validation data, but
the repository had no training entrypoint for it. The stable AIDS
`checkpoint-500` is a PEFT LoRA adapter; treating it as a complete base model
or resuming its completed optimizer/global step would give incorrect
continued-training semantics.

### Decision

Load the same 4-bit ChemLLM base used by AIDS SFT-v3, attach the stable AIDS
adapter with `is_trainable=True`, and start a fresh Mutagenicity Trainer state.
Retain the AIDS learning rate, batch/accumulation, scheduler, warmup, bf16, and
500-step full schedule. Make the previously implicit prompt/completion
supervision explicit: prompt tokens are masked with `-100`, while completion
and retained EOS tokens participate in causal-LM loss. Validate the complete
1,317/250 train/validation contract before any smoke sampling and never load
calibration or test.

### Consequences

- Continued SFT inherits learned AIDS adapter weights without silently falling
  back to another model or random LoRA initialization.
- Tokenization, truncation, parent coverage, checkpoint selection, and
  generation sanity checks are persisted as auditable artifacts.
- Validation loss selects checkpoints; calibration and test cannot influence
  training or selection.
- PPO, reward, selector, teacher, WNode, baselines, and unified evaluation code
  remain unchanged.

### Status

Accepted

---

## [2026-07-15] Require explicit parent-cohort inputs in saved FGW audits

### Background
The GlobalGCE Frequency-Top20 audit populated an empty `--comparison-run`
list with production output paths. As a result, an otherwise self-contained
unit test could discover an unrelated 1283-parent Ours run and fail before it
audited its two temporary parents.

### Decision
Treat comparison runs and reference cohorts strictly as data inputs: only open
them when supplied explicitly through `--reference-parent-ids`,
`--comparison-run Ours=...`, `--reference-ours-run`, or
`--auto-reference-from-ours`. Without one of these inputs, audit the current
run as an all-label-parent diagnostic and emit a warning. Explicit reference
CSVs are already in the current GlobalGCE ID namespace; explicit Ours-run
references are mapped into that namespace by a one-to-one canonical-SMILES
crosswalk before missing-ID validation.

### Consequences
- Unit tests and ad hoc audits no longer depend on files under `outputs/hpc`.
- Final 1283-parent audits retain their exact explicit crosswalk behavior.
- No MolCLR embedding, FGW distance, strict-flip, ranking, coverage, or Table 2
  calculation changes.

### Status
Accepted

---

## [2026-07-14] Use one explicit parent-ID cohort and corrected strict flip for final FGW reports

### Background
The raw GlobalGCE Frequency-Top20 run contains 1443 label parents, while the
final Ours reference cohort contains 1283 parents. Historical GlobalGCE pair
details also recorded the old weak flip (`pred_after != target_label`). Counting
mismatches before filtering by method mixed unrelated rows into the audit.

### Decision
Define the final comparison cohort by the exact `parent_id` set from the final
Ours run's `details/pair_details.csv`, or by an explicit reference-parent CSV.
Filter every method by this set before strict-flip, cost, prefix-K, threshold,
or bootstrap aggregation. Extra raw parents may be discarded by ID; missing
reference IDs are fatal, even when the raw parent count happens to match.

Recompute teacher-strict flip from saved `label`, `pred_before`, and
`pred_after`, retaining the old weak field only for audit. This correction is
post-processing only and reuses every saved Node-FGW distance. Final Table 2
reports only coverage and theta-covered conditional median cost at the exact
requested theta; the latter is asserted not to exceed theta.

### Consequences
- Raw 1443-parent GlobalGCE output remains an all-label-parent diagnostic.
- Final figures and tables use the same 1283 parent IDs across all methods.
- Historical weak-flip pair details can be corrected without loading MolCLR,
  running POT, or changing the distance cache.
- Applicable-parent median cost remains available as an audit metric but is not
  presented as theta-conditional cost in the final table.

### Status
Accepted

---

## [2026-07-14] Keep strict-flip confusion summaries self-contained and backward compatible

### Background
The GlobalGCE Frequency-Top20 FGW audit computed the historical strict-flip
confusion matrix correctly, but downstream automation could report a null
mismatch count when a JSON reader expected one exact top-level field name.
Older audit files may also contain the complete four-cell matrix without the
redundant totals.

### Decision
Write the four confusion cells, `recorded_true_pairs`,
`expected_strict_pairs`, and `mismatch_rows` at the top level of
`strict_flip_confusion.json`. Preserve the previous field aliases, and allow
readers to infer missing redundant totals from an arithmetically consistent
four-cell matrix. Such legacy inputs are `PASS_WITH_WARNINGS`; contradictory
provided totals remain failures.

### Consequences
- Automated checks no longer interpret a missing redundant field as a failed
  core experiment.
- The mismatch count is always `TF + FT` and is guarded by explicit arithmetic
  assertions.
- Corrected pair details, parent cohorts, FGW distances, coverage, candidate
  ranking, and corrected Table 2 metrics are unchanged.

### Status
Accepted

---

## [2026-07-14] Audit GlobalGCE Frequency-Top20 from saved Node-FGW artifacts

### Background
GlobalGCE Frequency-Top20 shows plateaus and jumps in prefix-K coverage. Such a
shape can be caused by candidate marginal coverage, but it can also expose a
stale weak-flip artifact, rank drift, or inconsistent post-processing. The
saved pair details already contain all required Node-FGW distances, so an audit
must not recompute MolCLR embeddings or FGW transport.

### Decision
Add a read-only audit that verifies teacher-strict flip, candidate order,
frequency provenance, prefix/threshold monotonicity, exact-theta consistency,
fullgraph evaluation semantics, and per-rank marginal coverage. It reads the
external Frequency-Top20 order and never ranks candidates by FGW.

The report metric previously called `Conditional median cost` is the median
best strict-recourse distance over parents with any finite strict recourse; it
is not conditioned on `distance <= theta`. Keep its compatibility field, but
label it `Applicable-parent median cost`, add a distinct
`Covered-parent median cost`, and assert that the latter cannot exceed theta.
Likewise, label applicability as `Strict-recourse applicable rate` so it is not
confused with subgraph match applicability or teacher-target parent rate.

### Consequences
- Existing evaluator outputs, Node-FGW distances, caches, and candidate ranks
  remain unchanged.
- Historical tables using the ambiguous labels remain reproducible but are
  identified by the audit as reporting-label issues.
- Frequency-ranked candidates may legitimately produce coverage plateaus;
  they are accepted as data-driven only after strict-flip, order, and summary
  consistency checks pass.

### Status
Accepted

---

## [2026-07-14] Generate paper-style recourse reports without reevaluating candidates

### Background
The four final MolCLR-Node-FGW runs already contain teacher-strict pair details
for externally selected Top20 candidate sets. Reusing evaluator summaries alone
cannot produce prefix-K curves, and ranking candidates by their measured FGW
distance during reporting would leak the evaluation metric back into selection.

### Decision
Add a read-only GCFExplainer-style reporting entrypoint that restores each
candidate order from its recorded external selector file, validates exactly 20
unique ranked candidates and no evaluator-side selection, and aggregates Ours
match instances to the minimum finite strict-flip distance for each
parent-candidate pair. The report uses all 1283 parents as the denominator,
represents unavailable unconditional recourse cost as positive infinity, and
uses one shared absolute-threshold grid and paired parent-bootstrap indices for
all four methods.

### Consequences
- Existing MolCLR embeddings, FGW distances, caches, strict-flip semantics, and
  candidate sets remain untouched.
- Prefix-K and threshold curves are reproducible from final artifacts alone.
- Candidate selection remains external to evaluation and reporting; FGW values
  never determine candidate rank.
- The same reporting implementation can be reused for another distance line by
  supplying four run paths, a distance label, and a table prefix.

### Status
Accepted

---

## [2026-07-12] Filter legal GCF-HIVCSV molecules before greedy Top-K export

### Background
The original HIVCSV summary export ranked all generated graphs before the
graph-to-SMILES legality audit. Invalid graphs could therefore consume ranks
and update the covered-parent set. Its fallback expression also interpreted a
real `min_distance_seen=0.0` as missing.

### Decision
Keep the historical export unchanged as an experiment artifact, but add a
validity-first export path. Convert and sanitize the complete raw candidate
pool, discard illegal or empty molecules, and then apply the existing greedy
key `(marginal_coverage_gain, frequency, -min_distance_seen)`. Preserve one
shared order across metadata, graph tensors, and FGW-ready SMILES.

### Consequences
- Invalid candidates cannot influence native coverage selection.
- A real zero distance wins the expected tie-break instead of becoming 999.
- The valid Top-K export is deterministic and written beside, rather than over,
  the historical `summary_export` files.

### Status
Accepted

---

## [2026-07-12] Treat Ours Top20 as externally preselected in Node-FGW

### Background
One selected fragment can match a parent molecule at several atom mappings.
The Node-FGW evaluator expands those mappings into multiple detail rows, but
that expansion is evaluation work and is not a new candidate-selection step.

### Decision
When `PRESELECTED_TOPK` is enabled, validate Ours selector directories using
`selected_subgraphs.csv` or `selected_subgraphs.json`, preserve their rank
order, and record the external selector identity from `selector_summary.json`
or directory metadata. Ours evaluation rows use
`evaluation_row_unit=match_instance`, while candidate provenance remains
`candidate_set_preselected=true` and `selection_performed_in_eval=false`.

### Consequences
Run summaries separately report unique parent-candidate pairs, detail rows, and
valid match instances. Multiple `match_index` values no longer imply that the
evaluator optimized or reordered the selected Top20. Fullgraph preselection
validation remains unchanged.

### Status
Accepted

---

## [2026-07-12] Require a teacher transition for strict-flip CCRCOV

### Background
The MolCLR-Node-FGW evaluator treated every candidate whose post-intervention
prediction differed from the dataset target label as a flip. This counted
parents that the teacher already classified as non-target before intervention,
inflating CCRCOV without an actual prediction transition.

### Decision
Define the main strict-flip condition consistently across shared CCRCov pair
generation, MolCLR-Node-FGW aggregation, and baseline comparison as:

`pred_before == target_label and pred_after != target_label`.

The earlier condition, `pred_after != target_label`, is retained only as an
explicit `old_weak_flip` audit field. Main CCRCOV continues to use all evaluated
parents as its denominator, while `num_teacher_target_parents` is reported
separately.

### Consequences
- Parents already predicted as non-target cannot create strict-flip coverage.
- Pair details and summaries expose both definitions without mixing them.
- Existing FGW distances, caches, thresholds, and candidate selection remain
  unchanged; affected evaluations must be rerun to refresh their metrics.

### Status
Accepted

---

## [2026-07-12] Audit absolute Node-FGW radii across methods

### Background
Method-local `auto_quantile` sweeps can use the same quantile labels while
producing different absolute MolCLR-Node-FGW thresholds. Coverage at equal
quantile labels is therefore not necessarily coverage at an equal distance
radius.

### Decision
Add a read-only Node-FGW threshold consistency audit. Each `run_dir + method`
is treated as a distinct run. The audit compares FGW definition, teacher and
parent protocol, quantile grid, and absolute thresholds independently. A pair
is directly comparable only when FGW configuration, evaluation protocol,
parent count, and absolute thresholds all match.

### Consequences
Auto-quantile remains suitable for method-local diagnostics. Final fair tables
must use shared explicit absolute FGW thresholds. Ours is never selected as an
implicit reference; reference comparison requires an explicit run id.

### Status
Accepted

---

## [2026-07-12] Add CLEAR Parent-Frequency Top20 as a parallel selector

### Background
CLEAR full-molecule generation yields repeated canonical candidates across
source parents and experiment repetitions. Greedy-MMR is one valid global
selection protocol, but a direct frequency baseline is useful for separating
generation recurrence from coverage-proxy optimization.

### Decision
Add `selection_mode=parent_frequency` to the shared CLEAR selector. It reuses
RDKit validation, canonical deduplication, and the AIDS/HIV RF strict-flip
filter. For each canonical candidate it records raw row frequency, distinct
`source_instance_index` frequency, distinct experiment frequency, minimum
action cost, and mean action cost. Parent frequency intentionally excludes
`source_exp_id` from its key.

The exact ranking is parent frequency descending, raw frequency descending,
RF label-0 probability descending, minimum total action cost ascending, then
canonical SMILES ascending. Node-FGW, GED, MolCLR embeddings, and iterative
coverage gain do not participate in this ranking.

### Consequences
`CLEAR Parent-Frequency Top20` is reported separately from CLEAR Greedy-MMR.
Its Top20 CSV is a preselected candidate set: Node-FGW preserves row order,
requires exactly 20 unique RDKit-valid strict-flip molecules, and records
`selection_method=parent_frequency`, `selection_performed_in_eval=false`, and
`candidate_set_preselected=true`.

### Status
Accepted

---

## [2026-07-12] Preselect CLEAR Top20 before MolCLR Node-FGW evaluation

### Background
CLEAR produces 9,184 RDKit-valid full-molecule candidates after RF-unified
conversion. Evaluating all label-1 parents against the entire pool with
MolCLR-Node-FGW is computationally impractical and would also conflate global
candidate selection with final distance evaluation.

### Decision
Add a dedicated CLEAR fullgraph selector with the following pipeline:

```text
CLEAR candidate generation
-> RDKit validation and canonical deduplication
-> shared RF strict-flip filter
-> Morgan/Tanimoto greedy MMR Top20
-> MolCLR-Node-FGW final evaluation
```

The selector reuses the accepted coverage-heavy Ours weights
(`w_cf=0.8`, `w_cov=20.0`, `w_cost=0.3`, `w_red=0.7`) and the same weighted MMR
score helper. Full-molecule coverage is represented by packed Morgan/Tanimoto
parent bitsets; candidate redundancy is Morgan Tanimoto. Because the Ours
selector uses exact fragment support and defines no full-molecule similarity
threshold, `COVERAGE_THRESHOLD` must be supplied explicitly.

Node-FGW preselected mode requires exactly 20 unique RDKit-valid candidates,
preserves CSV order, performs no in-evaluator selection, and evaluates every
target parent against those 20 candidates. It records
`selection_performed_in_eval=false` and `candidate_set_preselected=true`.

### Consequences
The final CLEAR Node-FGW workload decreases from `parents x 9184` to
`parents x 20`. Node-FGW remains an evaluation-only distance and never enters
the CLEAR selector. CLEAR native total-action-cost ordering is diagnostic and
does not replace RF strict-flip greedy MMR selection in the fair table.

### Status
Accepted

---

## [2026-07-10] Select GlobalGCE fullgraphs by strict-flip MolCLR Node-FGW coverage

### Background
The MolCLR Node-FGW evaluator can write pair-level distance and teacher-flip
details for both `ours_selected_subgraphs` and `globalgce` in one CSV. A
GlobalGCE top2000 fullgraph pool needs a project-owned top-K selection step
that reflects its actual strict-flip explanatory coverage, rather than a
frequency-only or arbitrary first-K choice.

### Decision
Add `scripts/select_fullgraph_candidates_by_fgw_coverage.py`. The selector
filters the input detail table by exact `method=globalgce` before computing any
distance quantile or coverage. A candidate covers a parent only when:

```text
cf_flip == true and Node-FGW distance <= threshold
```

It greedily maximizes marginal parent coverage, breaking ties by lower mean
distance on newly covered parents, shorter SMILES, and earlier source-candidate
order. The resulting `selected_top20_for_eval.csv` uses:

```text
method = GlobalGCE
fullgraph_method = globalgce_selected20
```

and can be supplied as a fullgraph candidate input to the Node-FGW evaluator.

### Consequences
Ours rows are never used to select GlobalGCE fullgraph candidates. The
selection remains evaluation-only: it does not modify GlobalGCE, the teacher,
PPO, selector training, or the existing Node-FGW distance calculation.

### Status
Accepted

---

## [2026-07-07] Add MolCLR Node-FGW as evaluation-only CCRCOV distance line

### Background
Graph-level MolCLR cosine distance can be too coarse for some AIDS/HIV CCRCOV
threshold sweeps. A node-level distance can preserve more local molecular
structure without changing the generator, selector, or training objective.

### Decision
Add `molclr_node_fgw` as an auxiliary CCRCOV evaluation distance line. The line
uses the existing MolCLR pretrained GIN checkpoint but extracts node-level
embeddings before graph pooling. Molecules are compared with Fused
Gromov-Wasserstein distance using normalized unweighted shortest-path structure
matrices and cosine node feature cost:

```text
distance_line = MolCLR-Node-FGW
distance_type = node_fgw
FGW_LAMBDA = 0.5
```

The implementation caches both SMILES-level node artifacts and pairwise FGW
distances:

```text
outputs/hpc/cache/molclr_node_embeddings/
outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite
```

Thresholds default to auto quantiles instead of graph-level MolCLR cosine
thresholds because FGW has a different scale.

### Consequences
`molclr_node_fgw` is evaluation-only. It does not modify loss functions, PPO,
candidate generation, or selector logic. It also skips `StructRed`, `CovRed`,
and pairwise candidate redundancy. GREED-GED remains the main GED-style
distance line; MolCLR Node-FGW is an embedding-matrix auxiliary CCRCOV line.

### Status
Accepted

---

## [2026-07-05] Evaluate CLEAR full-graph pools with explicit graphPred teacher adapter

### Background
CLEAR `export_test` and candidate-pool conversion now produce AIDS full-graph
records with `original_adj`, `cf_adj`, `original_x`, and `cf_x`. The converted
pool also preserves CLEAR official prediction diagnostics, but those fields are
not the final unified metrics. The historical AIDS RF oracle is SMILES-based and
cannot directly score CLEAR's continuous graph tensors.

### Decision
Extend `scripts/baselines/clear/evaluate_clear_candidate_pool.py` with
`TEACHER_KIND=clear_graphpred`. This adapter loads the CLEAR graph prediction
checkpoint:

```text
baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt
```

and recomputes predictions for each original/counterfactual graph pair. The
evaluator records `strict_flip_eval`, `strict_flip_vs_original_label_eval`, and
`cf_drop_eval`, while keeping `official_flip` only as a diagnostic comparison.
`TEACHER_KIND=none` / `action_only` remain cost-only diagnostics and must not be
reported as final CLEAR FlipRate, CFDrop, or CCRCov.

### Consequences
Final CLEAR AIDS reporting must explicitly record `TEACHER_KIND=clear_graphpred`
or another documented unified teacher path. Official CLEAR flip/validity values
are never used as the final strict-flip condition.

### Status
Accepted

---

## [2026-07-05] Add CLEAR-RF-FullGraph path for final fair CCRCOV tables

### Background
The CLEAR AIDS pipeline now produces a full-graph candidate pool and can be
evaluated with `TEACHER_KIND=clear_graphpred`. That native diagnostic uses
CLEAR's own graph prediction checkpoint and action-distance costs. It is not
directly comparable to Ours and GT-FullGraph when those methods are evaluated
with the shared AIDS/HIV RF oracle and learned/embedding distances.

### Decision
Add a separate `CLEAR-RF-FullGraph` adaptation path. The path first audits
whether CLEAR's `original_adj`, `cf_adj`, `original_x`, and `cf_x` arrays can be
conservatively converted into valid RDKit SMILES. If conversion is feasible, it
writes RF-readable fullgraph candidates and evaluates them through the same
GREED-GED / MolCLR CCRCov pipeline used by Ours and GT-FullGraph:

```text
parent set = outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
teacher = outputs/hpc/oracle/aids_rf_model.pkl
CF_MODE = strict_flip
method = CLEAR-RF-FullGraph
```

If conversion is not reliable, the audit records `rf_oracle_usable=false` and
the native `clear_graphpred` result remains diagnostic only.

The converter treats CLEAR `cf_x` as a continuous decoder tensor rather than an
atom vocabulary. It must never use a raw float value as an atom-type key. For
the current AIDS pickles, atom identity is recovered from the AIDS descriptor
slot `original_x[:, 2] = atomic_num / 100`; true one-hot or soft categorical
rows, if introduced later, are decoded by row-wise argmax over the fixed AIDS
atom vocabulary (`C`, `N`, `O`, `F`, `S`, `Cl`). Counterfactual topology comes
from symmetrized/thresholded `cf_adj`, with single-bond valence checks used to
avoid obviously invalid RDKit molecules. The converter records feature-schema
statistics, argmax distributions, decode-mode counts, `cf_adj` statistics, and
CSV-level RDKit validation. It enforces a quality gate
(`MIN_VALID_CANDIDATES=20`, `MIN_VALID_RATE=0.001` by default); failing either
the conversion gate or CSV validation exits non-zero and blocks downstream fair
evaluation.

MolCLR-Node-FGW can run CLEAR fullgraph candidates without re-running ours by
setting:

```text
RUN_OURS=0
RUN_FULLGRAPH=1
RUN_GT_FULLGRAPH=0
CLEAR_FULLGRAPH_CANDIDATES_PATH=<clear_rf_fullgraph_candidates.csv>
FULLGRAPH_METHOD_NAME=CLEAR-RF-FullGraph
```

### Consequences
The final fair table must not substitute CLEAR native graphPred/action-distance
metrics for RF-oracle CCRCov. CLEAR can enter the final table only through
`CLEAR-RF-FullGraph` or another explicitly documented adapter that shares the
same parent set, teacher, distance system, thresholds, and strict-flip mode as
the other methods.

### Status
Accepted

---

## [2026-07-03] Train GCFExplainer-HIVCSV GNN with imbalance-aware metrics

### Background
The adapted GCFExplainer-HIVCSV path uses the canonical project source
`data/raw/AIDS/HIV.csv` with `LABEL_COLUMN=HIV_active`. The label distribution
is highly imbalanced (`0: 39684`, `1: 1443`), so overall accuracy can be
misleading and can hide majority-class collapse.

### Decision
Add the adapted HIVCSV scaffold:

- `scripts/gcf_hiv_csv_prepare_dataset.py` converts the canonical CSV into
  RDKit/PyG `graphs.pt` without external graph benchmark downloads;
- `scripts/gcf_hiv_csv_train_gnn.py` trains the adapted HIVCSV GNN teacher;
- `scripts/gcf_hiv_csv_run_vrrw.py` runs a project-owned lightweight
  GCF-style VRRW over the HIVCSV graphs;
- `scripts/gcf_hiv_csv_export_summary.py` and
  `scripts/evaluate_gcf_hiv_csv_native.py` produce top-K summaries and native
  close-CF coverage;
- `scripts/convert_gcf_hiv_csv_graphs_to_smiles.py` is a diagnostic conversion
  path only.

The training script uses deterministic stratified train/validation/test splits
and enables class-weighted `CrossEntropyLoss` by default:

```text
weight_c = total_train / (num_classes * count_c)
```

Checkpoint selection defaults to macro-F1, and `gnn_train_summary.json`
records overall accuracy, per-class precision/recall/F1, macro-F1, balanced
accuracy, ROC-AUC, prediction counts, class weights, and split label counts.
If test label-1 recall is below the configured threshold, the summary includes
a warning.

### Consequences
The adapted HIVCSV path is separate from the official AIDS graph-benchmark
reproduction and must be reported as `GCFExplainer-HIVCSV` or
`GCFExplainer-adapted-HIVCSV`. Accuracy alone is not accepted as evidence that
the HIVCSV GNN teacher is usable. The adapted path does not invoke external
graph benchmark downloads; it reads only the project CSV-derived `graphs.pt`.

### Status
Accepted

---

## [2026-07-03] Add official GCFExplainer native fullgraph baseline path

### Background
The project already contains a GT-FullGraph proxy baseline, but that proxy is
not the official GCFExplainer reproduction. Official GCFExplainer outputs a set
of complete counterfactual graphs and writes to fixed relative paths inside its
repository, which makes alpha sweeps unsafe unless each run is isolated.

### Decision
Add project-owned official GCFExplainer adapters and Slurm entrypoints without
modifying the official source. The new path resolves the official checkout from
`GCF_OFFICIAL_REPO`, `third_party/GCFExplainer`, or the legacy
`baselines/gcfexplainer_official` directory. VRRW runs execute inside an
isolated per-run workdir and write results under
`outputs/hpc/gcfexplainer_official`. Native evaluation uses official GNN
predictions, official NeuroSED distance, `GCF_MODE=official_native`,
`TEACHER_TYPE=official_gnn`, `DISTANCE_TYPE=official_native`, and
`CF_MODE=strict_flip`.

### Consequences
GT-FullGraph remains a project proxy and must not be named official
GCFExplainer. Graph-to-SMILES-to-RF evaluation is available only as a diagnostic
because official graph artifacts may not preserve safe atom/bond mapping.
NetworkX GED is not used for large fullgraph GCFExplainer evaluation; GREED-GED
and MolCLR diagnostics reuse the existing distance pipelines only when valid
SMILES candidates are available.

### Status
Accepted

---

## [2026-07-03] Audit GlobalGCE AIDS/HIV edge-label conversion modes

### Background
The first `native-cf-fullgraph` GlobalGCE AIDS/HIV evaluation ran successfully,
but graph-to-SMILES conversion produced very low validity and many sanitized
SMILES with implausible cumulene-like double-bond chains. This indicated that
the exported GlobalGCE edge labels may not always be raw zero-based bond labels;
for example, an internal label value of `1` can mean a single bond rather than
a double bond.

### Decision
Keep GlobalGCE official source unchanged and make the project adapter
edge-label interpretation explicit and auditable. The converter now supports
`raw_zero_based`, `internal_one_based`, `adjacency_only_single`, and default
`auto`. In `auto`, each graph tries internal one-based labels, raw zero-based
labels, and adjacency-only single bonds, then selects the first RDKit-sanitized
result by that priority. The evaluator records raw conversion ok/fail counts,
unique valid candidates before top-K, selected candidates after top-K, edge
label values seen, and conversion success/failure by edge-label mode.

### Consequences
GlobalGCE `native-cf-fullgraph` remains a diagnostic fullgraph candidate
evaluation. Strict CCRCOV now requires teacher-strict flipping:
`distance <= threshold`, `pred_before == target_label`, and
`pred_after != target_label`. The old weaker condition
`pred_after != target_label` is retained only as `old_weak_CCRCOV` /
`old_weak_flip` audit output. If `distance_mode=tanimoto`, reports continue to
label the distance as `tanimoto_fingerprint`; it must not be presented as GED.

### Status
Accepted

---

## [2026-07-03] Use weighted CLEAR graphPred training for AIDS/HIV imbalance

### Background
The canonical AIDS/HIV source is `data/raw/AIDS/HIV.csv` with
`SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`, and `TARGET_LABEL=1`.
The raw distribution is strongly imbalanced (`HIV_active=0: 39684`,
`HIV_active=1: 1443`). The CLEAR AIDS max100 x10 dataset preserves this
natural imbalance, and the initial CLEAR graph prediction run degenerated into
an almost majority-class predictor.

An audit found no existing balanced parent molecule classification dataset that
can be directly used for CLEAR `pred`. Existing balanced or label-conditioned
artifacts are SFT/PPO prompt files, label-specific candidate pools, selector
outputs, or `hiv_quick` evaluation outputs. They are not CLEAR graphPred
training data and must not be used as a substitute for the prepared AIDS
pickles.

### Decision
Keep the prepared CLEAR AIDS dataset:

```text
baselines/clear_official/dataset/aids_full.pickle
baselines/clear_official/dataset/aids_datasplit.pickle
```

Add `patches/clear_official/004_aids_weighted_graphpred.patch` so the official
CLEAR `train_pred.py` runtime copy uses class-weighted cross entropy only for
`dataset=aids`. The class weights are computed from the current training split:

```text
weight_c = total_train / (num_classes * count_c)
```

The patch also changes graphPred metrics to be computed over the full
validation/test split instead of averaging batch-level AUC/F1. AIDS pred logs
now report training label counts, class weights, `y_true_counts`,
`y_pred_counts`, `positive_pred_rate`, `balanced_accuracy`, F1, and ROC-AUC.
For AIDS pred, checkpoint selection prefers validation F1 rather than validation
loss alone.

### Consequences
The canonical AIDS/HIV raw dataset and CLEAR max100 x10 pickles remain the
source for CLEAR AIDS pred. No new balanced evaluation dataset is introduced.
SFT/PPO prompt files, label1 candidate pools, and `hiv_quick` evaluation
outputs remain forbidden as CLEAR graphPred training data. The fix is isolated
to the CLEAR patch workflow and does not change PPO, selector, candidate pool,
or unified evaluation logic.

### Status
Accepted

---

## [2026-07-03] Evaluate GlobalGCE on canonical AIDS/HIV labels

### Background
GlobalGCE official AIDS top30 reproduction and export produce
`globalgce_rules.jsonl` and `globalgce_cfs_graphs.jsonl`. The official AIDS
graph format has its own preprocessing and label alignment caveats, so its
internal graph labels must not be treated as final project labels.

### Decision
Add a project-facing GlobalGCE evaluator path for the canonical AIDS/HIV
dataset:

- final dataset display name is `AIDS/HIV`;
- raw source is `data/raw/AIDS/HIV.csv`;
- labels come from `HIV_active`;
- target label is `1`;
- GlobalGCE official graph outputs are treated as baseline-generated candidate
  artifacts;
- `native-cf-fullgraph` converts GlobalGCE CF graphs to RDKit molecules and
  canonical SMILES, then evaluates them with the project teacher;
- strict CCRCOV uses `distance <= threshold`,
  `pred_before == target_label`, and `pred_after != target_label`;
- smoke distance is explicitly named `distance_type=tanimoto_fingerprint` and
  must not be reported as GED.

### Consequences
SuppCov is skipped for `native-cf-fullgraph` because complete CF graph
candidates are not support rules. `native-cf-delta-action` and `rule-action`
remain safety/audit modes until reliable source-parent atom mapping and
attachment-aware LHS/RHS replacement are available.

### Status
Accepted

---

## Decision: AIDS/HIV is the canonical main dataset

### Background
Project scripts and baseline adapters historically use several names around the
same benchmark: `hiv`, `hiv_quick`, `aids`, and `ogbg_molhiv`. This creates a
risk that engineering validation runs are mixed with final AIDS/HIV baseline
results.

### Decision
AIDS and HIV are not two separate main datasets in this project. The canonical
main dataset is the AIDS/HIV dataset backed by the single raw CSV:

```text
data/raw/AIDS/HIV.csv
```

The canonical columns are `SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`,
and `TARGET_LABEL=1`. Different modules may keep different internal dataset
keys:

- `hiv` / `hiv_quick` are legacy internal names for the same raw CSV;
- `aids` is the official graph-baseline dataset key for CLEAR and GCF-style
  graph-format adapters;
- `ogbg_molhiv` is engineering validation only.

Final comparison must be unified to `data/raw/AIDS/HIV.csv` and must record the
metadata required by `docs/DATASET_CONTRACT.md`.

### Consequences
Do not report `ogbg_molhiv` CLEAR results as final AIDS/HIV baseline results.
All final CCRCOV, CFDrop, FlipRate, Cost, and Redundancy tables must be
traceable to the canonical CSV, label column, target label, baseline dataset
key, teacher/oracle path or teacher kind, and `CF_MODE=strict_flip`.

### Status
Accepted

---

## [2026-07-02] Add CLEAR AIDS dataset support

### Background
The AIDS/HIV main experiment uses `data/raw/AIDS/HIV.csv` with
`smiles` as the molecule column and `HIV_active` as the binary label. Previous
CLEAR engineering smoke runs used `ogbg_molhiv`, but that dataset is not the
AIDS/HIV main-result dataset. CLEAR official source only supports
`community`, `ogbg_molhiv`, and `imdb_m` out of the box.

### Decision
Add a project-owned AIDS preparation and patch workflow:

- `scripts/baselines/clear/prepare_clear_aids_dataset.py` converts
  `data/raw/AIDS/HIV.csv` into CLEAR-compatible `aids_full.pickle` and
  `aids_datasplit.pickle`, using `max_num_nodes=100` and `10` deterministic
  stratified split repetitions by default;
- `scripts/slurm/prepare_clear_aids_dataset.sh` provides the HPC sbatch
  entrypoint for deterministic stratified CLEAR-internal split preparation;
- `patches/clear_official/003_support_aids_dataset.patch` adds
  `dataset=aids` support to CLEAR official loaders, CLI choices, molecular
  evaluation branches, and graph prediction model behavior through the existing
  idempotent patch mechanism;
- CLEAR wrappers now recognize `aids` dataset files and keep all generated
  pickles/checkpoints/exports under ignored runtime paths.

### Consequences
`ogbg_molhiv` remains only a CLEAR engineering validation dataset. AIDS
baseline runs should use `dataset=aids`, `SMILES_COLUMN=smiles`, and
`LABEL_COLUMN=HIV_active`. CLEAR official flip/validity remain diagnostic only.
The historical AIDS RF oracle at `outputs/hpc/oracle/aids_rf_model.pkl` is a
SMILES/Morgan-fingerprint oracle and cannot directly consume CLEAR continuous
graph counterfactual tensors; final strict-flip CCRCov requires a full-graph
candidate pool and an explicitly documented unified teacher/adapter path.

### Status
Accepted

---

## [2026-07-02] Add CLEAR candidate/action pool unified evaluation entrypoint

### Background
The CLEAR reproduction pipeline now produces a converted candidate/action pool
under `outputs/hpc/baselines/clear/<dataset>/candidate_pool/`. The converted
pool preserves CLEAR official prediction diagnostics, but final baseline
comparison must use the project's unified teacher/oracle and native-action
CCRCov convention.

### Decision
Add `scripts/baselines/clear/evaluate_clear_candidate_pool.py` and
`scripts/slurm/evaluate_clear_candidate_pool.sh`. The evaluator reads CLEAR
action-pool JSONL files, computes action-distance costs, writes
`per_candidate_eval.jsonl`, `summary.json`, `summary.csv`, `threshold_summary.csv`,
and `report.md`, and exposes `strict_flip`, `drop_or_flip`, and `drop_only`
counterfactual modes. CLEAR official `official_flip` is diagnostic only and is
never used as final strict flip. If the candidate pool lacks SMILES,
precomputed unified-teacher fields, or full graph arrays needed by a graph
teacher adapter, the evaluator fails clearly unless `--allow-action-only` is
used for smoke diagnostics.

### Consequences
CLEAR official source and training remain unchanged. The current default CLEAR
candidate pool can be smoke-checked for cost/action summaries, but final
`FlipRate`, `CFDrop`, and `CCRCov` require a unified teacher prediction source
for CLEAR original/counterfactual graph pairs.

### Status
Accepted

---

## [2026-07-02] Convert CLEAR exports into unified candidate/action pools

### Background
CLEAR `export_test` now produces per-instance original/counterfactual graph
pairs under `outputs/hpc/baselines/clear/<dataset>/test_exports/`. These files
preserve full graph arrays and CLEAR official prediction diagnostics, but they
are not yet in the action-pool format consumed by the project's unified
CCRCov/action-rule evaluation.

### Decision
Add `scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py` to
convert CLEAR export pickles into a project-owned JSONL candidate/action pool.
The conversion keeps official CLEAR flip and target-success diagnostics but
does not filter non-flips by default, because final `FlipRate`, `CFDrop`, and
`CCRCov` must be recomputed by the unified frozen teacher/oracle. Each
candidate records edge additions/deletions and continuous node-feature changes
from the original graph to the CLEAR counterfactual graph. A Slurm wrapper,
`scripts/slurm/convert_clear_exports_to_candidate_pool.sh`, provides the HPC
entrypoint.

### Consequences
CLEAR official source, model structure, training, and export logic remain
unchanged. Runtime candidate pools stay under
`outputs/hpc/baselines/clear/<dataset>/candidate_pool/` and must not be
committed. The resulting JSONL can feed the next CLEAR adapter/evaluator stage
for unified SuppCov, CCRCov, CFDrop, FlipRate, Cost, StructRed, and CovRed.

### Status
Accepted

---

## [2026-07-02] Add CLEAR per-instance counterfactual export

### Background
CLEAR official `test` loads trained CFE checkpoints and reports aggregate
metrics, but it does not persist per-instance original/counterfactual graph
outputs. The official entrypoint also maps `experiment_type == test` to
`test_small`, so the default printed metrics cover a small test subset. Unified
CCRCov/action-rule evaluation needs per-instance counterfactual graph records.

### Decision
Add a second project-owned CLEAR patch:

- `patches/clear_official/002_export_test_counterfactuals.patch` adds the
  marker `CLEAR_WRAPPER_EXPORT_TEST_COUNTERFACTUALS`;
- the patch adds opt-in CLI flags `--export_counterfactuals`,
  `--export_full_test`, `--export_max_items`, and `--export_dir`;
- official aggregate test behavior is preserved unless export flags are
  explicitly passed;
- `scripts/baselines/clear/run_clear.sh` adds `export_test` for full test
  split export and `export_test_small` for debugging;
- export files are written under
  `outputs/hpc/baselines/clear/<dataset>/test_exports/` as pickle arrays plus
  JSONL metadata.

### Consequences
CLEAR model structure, loss, optimizer, training logic, dataset loading, and
official aggregate metrics remain unchanged. The exported per-instance graph
records can be converted into a CLEAR candidate/action pool for unified
SuppCov, CCRCov, CFDrop, FlipRate, Cost, StructRed, and CovRed evaluation.

### Status
Accepted

---

## [2026-07-01] Patch CLEAR to save CFE checkpoints for test

### Background
CLEAR `pred` successfully saves the graph prediction model, but the official
`train` path in `baselines/clear_official/src/main.py` passes
`save_model=False`, while the official `test` path always loads
`../models_save/weights_graphCFE_CLEAR_<dataset>_exp<i>_epoch900.pt`. This can
make a completed CLEAR train run unusable for test because no CFE generator
checkpoint exists.

### Decision
Keep CLEAR algorithm code isolated and add a project-owned patch workflow:

- `patches/clear_official/001_save_cfe_checkpoints.patch` enables CFE
  checkpoint saving without changing model structure, losses, optimizer,
  dataset loading, or metrics;
- train now saves epoch 900 and final-epoch CFE `state_dict()` files with
  `[CLEAR_CKPT_SAVE]` logs;
- `scripts/baselines/clear/apply_clear_patches.sh` applies the patch
  idempotently by checking the `CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT` marker;
- `scripts/hpc_pull_clear.sh`, `scripts/baselines/clear/slurm_clear.sbatch`,
  and `scripts/baselines/clear/run_clear.sh` apply the patch before CLEAR runs;
- wrappers check for exp0/exp1/exp2 epoch-900 CFE checkpoints and create an
  epoch-900 symlink from the highest available epoch when needed.

### Consequences
The submodule does not need to be committed dirty as the sole source of the
fix. Runtime artifacts remain ignored. CLEAR test fails early with a clear
checkpoint error if train has not produced any usable CFE checkpoint.

### Status
Accepted

---

## [2026-07-01] Make GREED/MolCLR CCRCov default to strict flip

### Background
GREED-GED and MolCLR-Embedding CCRCov smoke outputs could report
`close_cf_coverage > 0` while `flip_rate_among_covered = 0` because the
counterfactual condition allowed probability-drop coverage through
`cf_drop >= min_cf_drop`.

### Decision
For GREED/MolCLR distance-based CCRCov evaluation, add explicit `cf_mode`
support:

- `strict_flip`: `distance <= theta` and `pred_after != label`;
- `drop_or_flip`: `distance <= theta` and either strict flip or
  `cf_drop >= min_cf_drop`;
- `drop_only`: `distance <= theta` and `cf_drop >= min_cf_drop`.

The default is now `strict_flip`. `min_cf_drop` remains recorded and is used
only by drop-based modes. Slurm wrappers expose `CF_MODE` and `MIN_CF_DROP`,
and threshold summaries/reports record the selected mode.

### Consequences
The main GREED/MolCLR CCRCov result now matches the paper-facing
`phi(G^a) != y` strict flip definition by default. GREED training, MolCLR
encoding, PPO, selector, and candidate generation remain unchanged.

### Status
Accepted

---

## [2026-06-29] Add CLEAR official baseline HPC wrappers

### Background
CLEAR / GraphCFE is kept as an official baseline under
`baselines/clear_official`. Its official code relies on relative paths such as
`../dataset` and `../models_save`, so project-owned wrappers must run from the
official `src` directory while keeping datasets, checkpoints, logs, and outputs
out of ordinary Git.

### Decision
Add project-owned CLEAR workflow files:

- shared shell helpers for dataset checks, runtime directory creation, and
  environment diagnostics;
- a stage wrapper for `pred`, `train`, `test`, CLEAR baselines, and `all`;
- a Slurm wrapper that activates the HPC default `smiles_pip118` conda
  environment by default, allows `CLEAR_CONDA_ENV` overrides, requests the A800
  GPU queue with one `gpu:a800:1` allocation, and delegates to the stage
  wrapper;
- an HPC pull helper that syncs submodules and prepares runtime directories
  without downloading data;
- documentation and `.gitignore` rules for CLEAR datasets, checkpoints, and
  logs.

### Consequences
`baselines/clear_official/src/` remains untouched. CLEAR can be launched from
HPC via `sbatch` after `git pull`, while large runtime artifacts remain outside
normal Git history.

### Status
Accepted

---

## [2026-06-29] Add GREED-GED and MolCLR distance lines for CCRCov

### Background
Fullgraph CCRCov evaluation cannot scale if every parent-candidate pair is sent
to NetworkX GED. The current HIV comparison needs a distance protocol that can
evaluate Ours and the GT-FullGraph proxy baseline without blocking on exact GED.

### Decision
Add two evaluation-only distance lines:

- GREED-GED prepares HIV graph pairs, labels deletion pairs exactly, labels
  fullgraph/random pairs with a scalable bounded approximation unless an
  explicit debug mode is requested, trains a Siamese GIN-style distance model,
  and evaluates CCRCov with predicted normalized GED;
- MolCLR-Embedding precomputes parent, hard-deletion residual, and GT-FullGraph
  candidate embeddings with an explicit runtime MolCLR checkpoint and evaluates
  CCRCov using `1 - cosine_similarity`;
- NetworkX GED remains only a small debug option and is not the default
  fullgraph distance path;
- GT-FullGraph is treated as a project proxy baseline, not as official
  GCFExplainer.

### Consequences
Training PPO, selector logic, and candidate generation remain unchanged. The
new files provide sbatch-first workflows for smoke/full GREED, smoke/full
MolCLR, and final comparison plots under the native-action CCRCov convention.

### Status
Accepted

---

## [2026-06-26] Add GlobalGCE baseline reproduction and unified evaluation wrappers

### Background
GlobalGCE is a relevant global counterfactual explanation baseline, but its
official outputs and metrics are not directly comparable to the project's
native-action CCRCov protocol. The official code should remain isolated under
`baselines/globalgce_official`, while project-owned wrappers should control
HPC execution, artifact export, and unified re-evaluation.

### Decision
Add GlobalGCE support without modifying official source code:

- layout checking for `baselines/globalgce_official`;
- a wrapper that copies official `src` into `outputs/hpc/globalgce/...` and runs
  `main.py` from the copied tree;
- an exporter that records official metrics, introspects rules/CF pickles, and
  writes project-owned JSON/JSONL artifacts;
- a `src.baselines.globalgce_adapter` module for AIDS label maps, CF graph
  conversion, rule descriptors, structural redundancy, coverage redundancy, and
  label-alignment warnings;
- a unified evaluator that supports first-stage native-CF CCRCov and a
  rule-action audit mode that reports SuppCov/StructRed/CovRed while explicitly
  marking safe RHS replacement as unsupported;
- Slurm wrappers for smoke, official top30, export, and label-specific CCRCov
  evaluation;
- baseline documentation in `docs/BASELINE_GLOBALGCE.md`.

### Consequences
GlobalGCE official code remains untouched. All generated GlobalGCE run outputs
live under `outputs/hpc/globalgce/...`, and unified evaluation outputs live under
`outputs/hpc/eval/globalgce/...`. Official validity/proximity metrics remain
reproduction diagnostics, while final comparison metrics are recomputed by the
project's frozen teacher and CCRCov protocol.

### Status
Accepted

---

## [2026-06-26] Add Slurm experiment tracking entrypoint

### Background
The project has many Slurm jobs for PPO, candidate-pool generation, selectors,
baseline evaluation, CCRCov sweeps, and visualization. Direct `sbatch`
submission makes it easy to lose the job id, command, output root, git commit,
and notes needed to reconstruct a result later.

### Decision
Add a lightweight submission and tracking layer:

- `scripts/exp_sbatch.py` calls the real `sbatch` without `shell=True`, records
  successful and failed submissions, and supports dry-run inspection;
- `scripts/exp_sbatch.sh` provides a repository-root shell wrapper;
- `scripts/sync_experiment_status.py` appends Slurm status snapshots using
  `sacct` with `squeue` fallback;
- `docs/EXPERIMENT_LOG.md` stores append-only human-readable records;
- `outputs/hpc/experiment_registry/jobs.jsonl` is the runtime machine-readable
  registry path;
- `docs/EXPERIMENT_TRACKING.md` documents the standard workflow and optional
  shell alias.

### Consequences
Training, selector, and evaluation logic remain unchanged. Existing Slurm
scripts are not modified; future submissions should prefer the tracking wrapper
so that experiment provenance is preserved.

### Status
Accepted

---

## [2026-06-26] Add close counterfactual coverage evaluation workflow

### Background
The selected-subgraph method needs to be evaluated under a GCFExplainer-style
close counterfactual graph protocol while preserving the existing PPO,
candidate-pool, audit, overlap, and selector workflows. The comparison must
support both our selected fragments, which become counterfactual graphs only
after hard deletion from a parent, and GCF-style baselines that already provide
full counterfactual graph candidates.

### Decision
Add evaluation-only code:

- a close counterfactual coverage module that computes hard-deletion residuals,
  normalized deletion-GED, optional NetworkX GED, teacher prediction deltas, and
  teacher-embedding distance when the teacher exposes an embedding API;
- CLI entrypoints for single-mode evaluation, four-way ours/GCF by GED/embedding
  evaluation, and matplotlib threshold-sweep visualization;
- a label-1 Slurm wrapper for the VSCode -> git push -> HPC git pull -> sbatch
  workflow;
- focused tests for deletion, threshold de-duplication, embedding-distance
  semantics, and GCF no-deletion behavior.

### Consequences
- Training logic, reward logic, candidate-pool generation, selector scoring, and
  overlap analysis remain unchanged.
- Our selected fragments are evaluated by hard deletion with any-match
  semantics; GCF candidates are evaluated as full counterfactual graphs.
- GED defaults to the fast deletion-cost upper bound for our hard-deletion
  residuals, while GCF GED uses NetworkX graph edit distance because it has no
  deletion action.
- Embedding distance is available only when the supplied teacher/model exposes a
  graph-embedding method; otherwise rows record `embedding_ok=false` with an
  explicit error.

### Status
Accepted

---

## [2026-06-22] Add no-GNN GT-fullgraph Tanimoto baseline trajectory for Pareto plots

### Background
MolCLR-GNN selector sweeps provide one trajectory for the current method, but
the Pareto frontier plot also needs a comparable baseline trajectory that does
not use GNN embeddings. The available clean GT-fullgraph action motif pool is
`camc_gt_fullgraph_motif_pool.csv`; it can be re-selected with the same
top-k/gamma sweep idea using Morgan/Tanimoto redundancy and then evaluated by
the unchanged legacy HIV quick CAMC evaluator.

### Decision
Add evaluation-only scripts:

- a GT-fullgraph motif-pool selector that aggregates action motifs, scores them
  with CF/support/size proxies, and applies greedy MMR with RDKit Morgan
  Tanimoto redundancy;
- a Slurm wrapper for gamma/beta sweeps that writes legacy-evaluator-compatible
  `selected_subgraphs.csv` and `selected_subgraphs.json`;
- a manifest-driven plotting script that reads legacy evaluator
  `camc_comparison_table.csv` outputs for Ours-MolCLR-GNN and
  Baseline-noGNN-Tanimoto trajectories, marks three-objective Pareto points, and
  exports PNG/PDF figures.

### Consequences
- The legacy evaluator remains unchanged.
- The new baseline does not use GNN embeddings; redundancy is the original
  Morgan/Tanimoto structural similarity.
- Selected motif outputs are compatible with
  `scripts/eval/compare_hiv_recourse_baselines.py --ours-selected-dir`.

### Status
Accepted

---

## [2026-06-21] Make MolCLR-GNN skip policy produce selector-ready pools

### Background
The MolCLR-GNN embedding job can encode most candidate fragments while a small
number of invalid fragment SMILES fail RDKit/PyG graph construction. The
original `invalid_policy=skip` behavior omitted those SMILES from the embedding
map but still wrote their original rows to the output JSONL, leaving rows
without `final_fragment_gnn_embedding` and causing the selector's
`--embedding-missing-policy error` mode to fail.

### Decision
Keep selector behavior unchanged and fix only the embedding preparation layer:

- have the MolCLR encoder expose invalid-SMILES details alongside successful
  embeddings;
- make `scripts/add_candidate_pool_molclr_embeddings.py` skip failed rows from
  the output JSONL when `--invalid-policy skip` is used;
- retain zero-vector behavior only for `--invalid-policy zero`, with an explicit
  `molclr_embedding_status` marker;
- expand the summary with input/output row counts, skipped rows, zero rows, and
  missing-embedding checks.

### Consequences
- `INVALID_POLICY=skip` now creates selector-ready JSONL files whose retained
  rows all contain `final_fragment_gnn_embedding`.
- Failed rows remain auditable through `molclr_gnn_embedding_failed_rows.jsonl`.
- Existing ChemLLM text embeddings, Morgan/Tanimoto redundancy, and selector
  logic remain unchanged.

### Status
Accepted

---

## [2026-06-15] Add MolCLR-GNN embedding redundancy workflow for selector experiments

### Background
The selector already supports embedding-cosine redundancy through
`--sim-metric embedding`, but the existing learned embedding field is generated
from the ChemLLM text model. The current experiment needs a fragment-level graph
embedding from a pretrained MolCLR GIN/GCN encoder while keeping Morgan/Tanimoto
and ChemLLM embedding paths unchanged.

### Decision
Add an evaluation/selection workflow only:

- introduce `src.embeddings.molclr_gnn_embedding`, which converts fragment
  SMILES to MolCLR/PyG molecular graphs and loads MolCLR code/checkpoints from
  explicit runtime paths;
- add `scripts/add_candidate_pool_molclr_embeddings.py`, which writes
  `final_fragment_gnn_embedding` into a derived candidate-pool JSONL;
- add Ours and GT seed-13 Slurm wrappers for MolCLR-GNN embedding generation,
  embedding-redundancy selection, and legacy HIV quick CAMC re-evaluation;
- document that MolCLR code/checkpoints are external assets and must not be
  downloaded implicitly on HPC.

### Consequences
- Selector defaults remain unchanged; Morgan/Tanimoto remains available unless
  scripts explicitly pass `--sim-metric embedding`.
- The new `final_fragment_gnn_embedding` field affects only the redundancy
  similarity term, not coverage gain, counterfactual scoring, size penalties,
  training, rewards, or selected-subgraph generation.
- Ours and GT can be compared with the same MolCLR checkpoint and the same
  selector redundancy semantics.

### Status
Accepted

---

## [2026-06-11] Re-evaluate embedding selector sets with the legacy HIV quick CAMC evaluator

### Background
The candidate-pool sanity check can show protocol drift, but it still does not
answer whether the embedding selector top20 sets perform poorly under the exact
legacy CAMC evaluator that produced the old PPT table. The old reference table
was generated by `scripts/eval/compare_hiv_recourse_baselines.py` using the HIV
CSV, RF teacher, seed 13, and CAMC top-k/delta settings.

### Decision
Add an evaluation-only Slurm workflow that runs the unchanged legacy HIV quick
CAMC evaluator three times:

- old Morgan-MMR selector top20 as a reproduction control;
- embedding conservative wide-grid selector top20;
- embedding low-redundancy wide-grid selector top20.

Add a small summary script that reads each run's `camc_comparison_table.csv`,
extracts `method=ours_selected_subgraph, k=20`, compares embedding rows against
old Morgan, and checks whether old Morgan reproduces the old PPT values.

### Consequences
- The legacy evaluator metrics and implementation remain unchanged.
- The result directly distinguishes true embedding-selector coverage loss from
  the earlier candidate-pool-evidence protocol difference.
- All long-running work remains submitted through Slurm rather than executed on
  a login node.

### Status
Accepted

---

## [2026-06-11] Add selected-set sanity check for CAMC protocol drift diagnosis

### Background
The legacy HIV quick CAMC table reported much higher Ours coverage than the
new embedding-selector final table. The new table is computed from selected
top20 fragments and candidate-pool evidence, while the legacy CAMC evaluator
uses full target inputs plus RF-teacher deletion evaluation. A dedicated sanity
check is needed to distinguish true coverage loss from evaluator/protocol
differences.

### Decision
Add an evaluation-only selected-set sanity check that:

- reads multiple selector directories, including the old Morgan-MMR set and new
  embedding-MMR wide-grid sets;
- evaluates all selected sets under one shared candidate-pool evidence protocol;
- dumps each selected fragment with selector score, support evidence,
  cf-drop/flip evidence, atom-ratio evidence, and redundancy diagnostics;
- records the legacy CAMC generator location
  (`scripts/eval/compare_hiv_recourse_baselines.py`) and explains why this
  lightweight sanity command does not rerun teacher-based CAMC unless that full
  evaluator is launched separately.

### Consequences
- The check can reveal whether the embedding selector truly sacrificed coverage
  relative to the old Morgan selected set under the same evidence source.
- If the old Morgan set is also much lower under candidate-pool evidence than
  in the legacy CAMC table, the observed drop is mainly due to evaluator or
  evidence-source differences.
- Training code, selector defaults, selected-subgraph artifacts, and candidate
  pools remain unchanged.

### Status
Accepted

---

## [2026-06-10] Formalize embedding-cosine redundancy selector wide-grid CAMC workflow

### Background
The gamma-only embedding selector sweep showed that both ours and the relaxed
GT-fullgraph proxy baseline can run with `--sim-metric embedding`, but the
first pass did not find an ours gamma that simultaneously preserved
coverage/flip/cf-drop and reduced embedding redundancy below the GT mean. The
experiment therefore needs a wider coverage-vs-redundancy search and a final
CAMC-style table computed from the selected top20 fragments.

### Decision
Keep selector defaults and all training code unchanged, and add explicit
evaluation-only Slurm workflows:

- run Ours and GT-fullgraph relaxed selectors with embedding-cosine redundancy
  over a beta/gamma grid;
- summarize the grid without requiring identical beta/gamma alignment, while
  also reporting same-parameter deltas;
- identify Ours Pareto candidates by maximizing coverage, keeping flip high,
  preferring higher cf-drop, and minimizing embedding cosine redundancy;
- write conservative, balanced, and low-redundancy recommended Ours configs;
- compute the final selected-top20 CAMC-style table from selector outputs and
  candidate-pool evidence, explicitly flagging GT `cf_drop=0.0` rows as proxy
  filled rather than teacher re-evaluated strength.

### Consequences
- The official experiment scripts now make embedding cosine similarity the
  redundancy term by passing `--sim-metric embedding` explicitly.
- Morgan/Tanimoto remains available as the selector default and as a reporting
  diagnostic, but it is no longer the redundancy objective in this formal
  comparison workflow.
- Final CAMC table generation is reproducible from selected selector artifacts
  and candidate pools, with clear warnings about proxy GT cf-drop and
  theta-coverage fallback sources.

### Status
Accepted

---

## [2026-06-10] Relax GT-fullgraph embedding selector flow after candidate-pool diagnosis

### Background
The first GT-fullgraph embedding selector sweep produced zero coverage and null
selected metrics. The converted GT CAMC motif pools can lack teacher-recomputed
`cf_drop` / `cf_flip` fields, while the strict selector wrapper required
`--require-cf-flip` and the selector default `min_cf_drop=0.2`; this can filter
all GT proxy candidates before MMR selection.

### Decision
Keep selector defaults and training code unchanged, but add an explicit GT proxy
diagnosis and relaxed evaluation path:

- add a selector-pool diagnosis script that reports required-field coverage,
  embedding availability, strict-filter reasons, and relaxed-GT-filter reasons;
- make CAMC-to-candidate-pool conversion write `cf_drop=0.0` with
  `cf_drop_missing=true` when the CAMC motif pool lacks true `cf_drop`;
- make missing `cf_flip` default to `true` with `cf_flip_missing=true` for this
  GT proxy candidate-pool conversion;
- add a relaxed GT embedding sweep wrapper that removes `--require-cf-flip` and
  sets `--min-cf-drop -999` while keeping embedding redundancy,
  final-substructure filtering, deduplication, and the shared MMR weights;
- add a relaxed summary wrapper that compares ours against the relaxed GT sweep
  root.

### Consequences
- The original strict GT sweep script remains available for auditing.
- The relaxed path treats GT CAMC motifs as an already constructed baseline
  action pool rather than as teacher-rescored generated fragments.
- Ours and GT still use the same embedding redundancy selector once their
  candidate pools pass the minimum structural/embedding filters.

### Status
Accepted

---

## [2026-06-10] Add embedding-MMR comparison workflow for GT-fullgraph CAMC motifs

### Background
The embedding-based selector is available for our merged stable300 candidate
pool, but the GT-fullgraph CAMC baseline currently exists as action-motif pool
CSV files rather than selector-readable candidate pools with learned fragment
embeddings. A fair redundancy comparison requires both ours and GT-fullgraph to
run the same class-level MMR selector with `sim_metric=embedding`.

### Decision
Add evaluation/Slurm workflow code only:

- convert the three clean GT-fullgraph CAMC motif pools
  `label1_1594411`, `label1_1594412`, and `label1_1594413` into
  selector-readable JSONL pools, explicitly excluding the older `1593189` run;
- reuse `scripts/add_candidate_pool_embeddings.py` to add
  `final_fragment_embedding` to the converted GT pools;
- run embedding-redundancy gamma sweeps for both ours merged and GT-fullgraph
  proxy pools with identical MMR selector weights except for gamma;
- summarize ours-vs-GT sweep results by gamma, including GT mean/std over the
  three clean seeds and a simple pass/fail recommendation rule.

### Consequences
- Existing SFT, PPO, reward, selector defaults, selected-subgraph artifacts, and
  original candidate pools remain unchanged.
- The default selector redundancy metric remains Morgan/Tanimoto unless an
  experiment script explicitly passes `--sim-metric embedding`.
- GT-fullgraph CAMC motif pools can now participate in selector-level embedding
  redundancy comparisons after an explicit conversion and embedding-generation
  preparation step.

### Status
Accepted

---

## [2026-06-02] Add offline candidate-pool embedding generation for embedding-MMR selection

### Background
The class-level selector now supports `--sim-metric embedding`, but the current
stable300 merged candidate pool does not yet contain `final_fragment_embedding`.
The embedding selector should therefore consume a derived JSONL with learned
fragment embeddings instead of mutating or regenerating the original
`candidate_pool.jsonl`.

### Decision
Add an evaluation/inference utility layer only:

- introduce `scripts/add_candidate_pool_embeddings.py`, which reads an existing
  candidate pool, embeds each resolved fragment SMILES with the same ChemLLM
  base-model plus optional SFT/PPO PEFT adapter loading path used by
  `src.eval.full_candidate_pool`, and writes a new
  `candidate_pool_with_embeddings.jsonl`;
- keep `final_fragment` as the primary text source, with fallbacks through
  `core_fragment`, `final_fragment_smiles`, `candidate_smiles`, and
  `raw_fragment`;
- use attention-mask-aware mean pooling by default over the last hidden state,
  L2-normalize the vector, and record summary/failed-row sidecar files;
- add stable300 and generic Slurm wrappers for HPC embedding generation, then
  point the embedding-MMR selector wrapper at the derived embedded pool.

### Consequences
- Existing SFT, PPO, reward, selector training, selected-subgraph artifacts, and
  original candidate pools remain unchanged.
- The embedding selector now has a reproducible upstream preparation step before
  `--embedding-missing-policy error` is used.
- The default Morgan/Tanimoto selector path remains unaffected.

### Status
Accepted

---

## [2026-06-02] Add embedding-based redundancy mode for class-level selector

### Background
The class-level counterfactual subgraph selector currently uses greedy MMR with
Morgan/Tanimoto similarity as the redundancy penalty. For candidate pools that
also contain learned subgraph embeddings, we need an optional way to compute
redundancy from embedding cosine similarity while preserving the default Morgan
behavior and the deletion-based counterfactual objective.

### Decision
Extend only the selector/evaluation layer:

- keep `sim_metric=morgan` as the default and preserve the existing Morgan
  fingerprint Tanimoto redundancy path;
- add `sim_metric=embedding`, where redundancy is
  `max(0, cosine(candidate_embedding, selected_embedding))`;
- parse embeddings from `final_fragment_embedding` by default, with fallback
  fields for `embedding`, `fragment_embedding`, `subgraph_embedding`, and
  `graph_embedding`;
- add `embedding_missing_policy={error,skip}` so missing embeddings either fail
  clearly or are filtered explicitly;
- record MMR component diagnostics in selected outputs and add pairwise
  embedding cosine statistics to selector summaries/reports;
- add an HPC Slurm wrapper for the stable300 label-1 merged pool embedding
  selector and a lightweight embedding-field checker.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- CF score, coverage gain, size penalty, and candidate filters remain the same;
  only the redundancy similarity source changes when explicitly requested.
- Existing Morgan/Tanimoto reports remain present for backward compatibility,
  with embedding statistics reported separately.

### Status
Accepted

---

## [2026-05-29] Replace eval Morgan fingerprint calls and add CAMC motif overlap diagnostics

### Background
The HIV quick comparison plus CAMC run completed, but the Slurm log contained a
large number of RDKit deprecation warnings:
`DEPRECATION WARNING: please use MorganGenerator`. These warnings came from
legacy Morgan fingerprint APIs used by evaluation and similarity helpers, not
from a training objective change.

### Decision
Keep the change evaluation/helper scoped:

- replace legacy Morgan bit-vector calls in HIV comparison, selector/audit
  similarity helpers, selected-subgraph overlap, and chemistry projection/repair
  Tanimoto helpers with cached `rdFingerprintGenerator.GetMorganGenerator`
  instances;
- keep `src/rewards/reward_calculator.py` unchanged because reward code is out
  of scope and already prefers the newer generator when available;
- add `--suppress-rdkit-warnings` / `--no-suppress-rdkit-warnings` to the HIV
  comparison script as a fallback log-control option;
- add CAMC `motif_overlap_diagnostics` comparing ours and GT selected motifs by
  exact overlap, max Tanimoto, atom counts, aromatic motifs, and hetero-atom
  motifs;
- make the HIV quick Slurm wrapper count MorganGenerator deprecation warnings in
  `progress.log` at the end of the job.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- CAMC metrics are unchanged; the new motif-overlap block is diagnostic only.
- Future HIV quick comparison logs should show zero MorganGenerator deprecation
  warnings unless a separate code path still uses RDKit's legacy API.

### Status
Accepted

---

## [2026-05-28] Add theta-aware recourse coverage diagnostics and CAMC action-motif comparison

### Background
The HIV quick comparison ran end to end, but long runs lacked enough progress
logging for Slurm monitoring. The ours recourse coverage could also show
`k=20 < k=10`, which indicated that the evaluator was choosing one action before
applying theta instead of computing theta-aware existential coverage over the
top-k action set. The current analysis also needs a second, method-aligned view
that compares shared counterfactual action motifs while remaining applicable to
fullgraph baselines.

### Decision
Update only the evaluation and Slurm layers:

- rebuild `scripts/eval/compare_hiv_recourse_baselines.py` around explicit
  ours action candidates and theta-aware existential aggregation;
- keep the original recourse-level outputs while adding
  `ours_action_candidates.csv`, `diagnostic_counts.json`, and `progress.log`;
- add CAMC output files that evaluate action motifs from our selected fragments
  and MCS-deleted motifs extracted from GT or extra selected fullgraph SMILES;
- add flushed logging, tqdm progress, MCS timing diagnostics, and recourse/CAMC
  monotonicity warning lists;
- update `scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh`
  to tee logs to the run directory and pass CAMC/progress controls.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- Recourse coverage is now computed as `exists feasible action` under each
  theta, so it should be monotone in both `k` and theta.
- CAMC is more aligned with class-level counterfactual subgraph selection, but
  fullgraph methods can still participate when they provide selected fullgraph
  SMILES. Official graph benchmark outputs still require a graph-level CAMC
  evaluator or a SMILES adapter.

### Status
Accepted

---

## [2026-05-28] Add HIV quick recourse-level comparison evaluator

### Background
The official GCFExplainer AIDS reproduction is now available as a sanity check,
but it is not directly comparable to the current HIV/SMILES counterfactual
fragment system. The next practical need is a fast recourse-level comparison in
the current RF-teacher setting that can compare our selected subgraphs with a
simple opposite-label full-molecule baseline without changing training code.

### Decision
Add an evaluation-only quick comparison path:

- `scripts/eval/compare_hiv_recourse_baselines.py` evaluates
  `ours_selected_subgraph` by deleting selected fragments from each target
  molecule and evaluates `gt_fullgraph_greedy` by greedily choosing
  opposite-label full molecules;
- both methods are normalized to per-input recourse candidates `G_i'` and scored
  with the same RF teacher for `p_before`, `p_after`, `cf_drop`, and `cf_flip`;
- both methods use the same RDKit MCS proxy distance for approximate recourse
  cost;
- `scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh` provides
  an HPC wrapper with `smiles_pip118`, environment diagnostics, HIV CSV
  auto-discovery, and explicit failure when the CSV is ambiguous;
- `docs/baselines/hiv_quick_recourse_comparison.md` documents the scope,
  limitations, metrics, and commands.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool training code remains
  unchanged.
- The comparison is explicitly a quick HIV/SMILES RF-teacher analysis, not an
  official GCFExplainer reproduction and not a learned/exact GED evaluation.
- Ours-style subgraph match coverage is retained as an internal diagnostic, but
  headline comparison uses recourse-level coverage, cost, drop, and flip metrics.

### Status
Accepted

---

## [2026-05-28] Harden official GCFExplainer conda activation and add AIDS GNN training wrapper

### Background
The official GCFExplainer AIDS reproduction needs a temporary upstream GNN
checkpoint before `vrrw.py` can run. The first Slurm attempt failed before
training because `source ~/.bashrc` transitively loaded `/etc/bashrc`, where the
HPC shell referenced `BASHRCSOURCED` while `set -u` behavior could treat it as
an unbound-variable error.

### Decision
Keep the fix isolated to the official GCFExplainer scaffold:

- add `scripts/slurm/gcfexplainer/train_aids_gnn.sh` to train upstream
  `gnn.py --dataset aids` in the separate `gcfexplainer_py38` environment;
- source `/share/home/u20526/anaconda3/etc/profile.d/conda.sh` directly in all
  GCFExplainer Slurm wrappers instead of `~/.bashrc`;
- enable `set -u` only after conda activation in these wrappers;
- fail fast in the `vrrw` and all-in-one wrappers when
  `baselines/gcfexplainer_official/data/aids/gnn/model_best.pth` is missing,
  with a direct pointer to the GNN training sbatch command.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool code paths remain
  unchanged.
- The GCFExplainer baseline no longer depends on user shell startup files for
  conda activation.
- Missing official AIDS GNN checkpoints now produce a clear remediation command
  instead of a later opaque upstream failure.

### Status
Accepted

---

## [2026-05-26] Add isolated official GCFExplainer AIDS reproduction scaffold

### Background
The project needs a reproducible way to run the official GCFExplainer AIDS
baseline from the frozen third-party source under
`baselines/gcfexplainer_official/` while preserving the current v3 SMILES
counterfactual objective and avoiding dependency contamination in the main
`smiles_pip118` environment.

### Decision
Add an isolated HPC reproduction scaffold:

- `scripts/slurm/gcfexplainer/reproduce_aids_vrrw.sh` runs upstream
  `vrrw.py --dataset aids` and syncs official `results/aids/` artifacts into a
  per-job output directory.
- `scripts/slurm/gcfexplainer/reproduce_aids_summary.sh` runs upstream
  `summary.py --dataset aids`, then parses coverage/cost text into JSON and CSV.
- `scripts/slurm/gcfexplainer/reproduce_aids_all.sh` runs both stages in one
  Slurm job.
- `scripts/eval/collect_gcf_official_results.py` parses known and partially
  unknown summary formats without failing silently.
- `docs/baselines/gcfexplainer_reproduction.md` documents local checks, separate
  conda environment setup, Slurm commands, and result inspection.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool code paths remain
  unchanged.
- Official GCFExplainer dependencies stay in a separate `gcfexplainer_py38`
  environment by default, with `CONDA_ENV` override support.
- The official AIDS run is recorded as a sanity check, not as a fair comparison
  with the current HIV/SMILES method.
- Missing upstream `vrrw.py` or `summary.py` fails fast with explicit errors, and
  unparseable summary logs still produce machine-readable failure artifacts.

### Status
Accepted

---

## [2026-05-22] Add label=1 Base/SFT/PPO ablation wrappers

### Background
The current stable300 label=1 model is selector-ready, but the project still
needs a controlled ablation to separate the contribution of SFT from the
additional contribution of PPO. The comparison must keep the label=1 parent set,
teacher/oracle, generation count, projection settings, audit tooling, and
selector settings fixed so the measured differences are attributable to model
stage rather than sampling or downstream configuration changes.

### Decision
Add a label=1 ablation layer that reuses the existing full-pool generator,
candidate-pool audit, and class-level selector:

- `scripts/generate_full_candidate_pool.py` can now treat `--sft-lora-path NONE`
  as a base-model-only inference run, skipping PEFT adapter loading while
  preserving existing SFT-only and PPO adapter paths.
- Three Slurm wrappers generate and audit comparable `n=4` candidate pools for
  Base ChemLLM, SFT-only, and SFT+PPO stable300.
- Three selector Slurm wrappers apply the same coverage-heavy MMR selector
  settings to each pool.
- `scripts/export_candidate_pool_audit_artifacts.py` materializes audit sidecar
  artifacts expected by ablation bookkeeping.
- `scripts/summarize_label1_sft_ppo_ablation.py` combines audit and selector
  summaries into CSV and Markdown reports.

### Consequences
- The ablation does not modify PPO training code, stable PPO training code, or
  label=0 logic.
- Base ChemLLM can now be evaluated through the same generation/audit path as
  SFT and PPO checkpoints.
- The high-temperature merged pool remains reserved for the final complete
  system and is excluded from this main ablation to avoid confounding sampling
  diversity with training-stage effects.

### Status
Accepted

---

## [2026-05-21] Add a same-source label0 PPO prompt builder for unified label01 runs

### Background
The existing label=1 PPO prompt CSV in the SFT v3 HIV dataset directory is a
minimal `smiles,label` file and is consumed by downstream PPO, pool generation,
and selector jobs. The corresponding label=0 CSV was missing, which prevented
the unified label01 prompt build from proceeding. Copying label=1 rows and
changing the label would violate the counterfactual data contract because
label=0 prompts must come from genuine label=0 parent molecules.

### Decision
Add a small reusable prompt-CSV builder around the existing PPO prompt dataset
loader:

- `scripts/build_label_ppo_prompt_csv.py` reads a shared source CSV/JSONL,
  resolves SMILES and label columns with existing fallbacks, filters by
  `--target-label`, and writes only `smiles,label`;
- `scripts/slurm/build_sft_v3_hiv_ppo_prompts_label0_same_as_label1.sh`
  builds the missing label=0 file from the same SFT v3 train split source and
  also emits the stratified `shuffle_seed13` variant;
- `scripts/slurm/build_unified_ppo_prompts_label01.sh` now uses the same
  minimal builder when label0 is missing, and merges label0/label1 into a
  shuffled two-column unified CSV without requiring a separate source CSV when
  both label-specific prompt files already exist.

### Consequences
- The existing label=1 prompt file and downstream label=1 pipeline remain
  untouched.
- Unified label-conditioned PPO can build its input from genuine label=0 and
  label=1 parents while preserving the minimal prompt CSV schema expected by
  current generation/training loaders.
- The label0 build is reproducible from Slurm and produces the same style of
  stratified shuffle companion file used by stable label=1 PPO runs.

### Status
Accepted

---

## [2026-05-21] Harden unified label01 prompt Slurm input resolution

### Background
The unified label-conditioned PPO prompt build wrapper previously required an
original source training CSV even when label0 and label1 PPO prompt CSVs already
existed. On HPC this caused `build_unified_ppo_prompts_label01.sh` to fail with
an unresolved `SOURCE_INPUT_CSV`, even though the intended next step could have
been completed by merging the existing label-specific prompt files.

### Decision
Keep the Python prompt builders unchanged and harden only the Slurm wrapper:

- preserve explicit `SOURCE_INPUT_CSV` support when the user supplies a valid
  source CSV;
- if `SOURCE_INPUT_CSV` is empty and both label-specific prompt CSVs already
  exist, merge them directly into the unified label01 prompt CSV;
- otherwise, auto-discover an original training CSV under the dataset directory
  while excluding derived prompt, audit, candidate, balance, and summary files;
- emit structured `[UNIFIED_PROMPT_*]` diagnostics and a richer unified summary
  JSON with source mode and label counts.

### Consequences
- Existing label=1 prompt artifacts are not rebuilt unless explicitly requested
  with `FORCE_REBUILD_PROMPTS=true`.
- HPC users can recover from missing source-path configuration without manual
  intervention when label0 and label1 prompt CSVs are already available.
- Failure logs now include dataset CSV listings and column previews, making
  source CSV selection issues much easier to diagnose.

### Status
Accepted

---

## [2026-05-17] Add a dedicated Slurm wrapper for auditing the stable300 candidate pool with the existing selector-facing audit script

### Background
The repository already has a selector-facing candidate-pool audit entrypoint in
`scripts/audit_candidate_pool.py`, and the stable300 run
`decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`
has completed. The immediate need is to audit its saved
`candidate_pool.jsonl` on HPC without introducing a new audit implementation or
changing any PPO or stable-PPO training code.

The existing audit CLI was confirmed to support the required arguments:

- `--pool_jsonl`
- `--out_json`
- `--out_txt`
- `--group_by_label`
- `--sim_sample_size`
- `--topk_show`

### Decision
Add only a thin Slurm wrapper:

- `scripts/slurm/audit_candidate_pool_stable300.sh`

This wrapper:

- targets the stable300 candidate pool path directly;
- prints environment and path diagnostics;
- checks that both the candidate pool and the audit script exist;
- writes outputs into
  `outputs/hpc/audits/<RUN_NAME>_candidate_pool_audit/`;
- reuses the existing `scripts/audit_candidate_pool.py` unchanged.

### Alternatives considered
1. Write a new stable300-specific audit script.
2. Hardcode the stable300 path inside the existing audit Python entrypoint.
3. Run the audit manually from an interactive shell without a Slurm wrapper.

### Consequences
- Stable300 can now be audited with a single `sbatch` command.
- The result remains directly comparable with earlier candidate-pool audits
  because the same Python audit implementation is reused.

### Status
Accepted

---

## [2026-05-20] Add unified label-conditioned PPO prompt, training-wrapper, and overlap-analysis pipeline

### Background
The stable300 label=1 pool already became selector-ready, so the next question
is no longer whether more PPO steps help. Instead, we now need to test whether
one shared policy can condition on the parent label and produce useful
counterfactual fragments for both label=0 and label=1 parents, while keeping
the existing selector and pool-audit tooling intact.

That requires:

- explicit label-conditioned prompt construction for label0, label1, and mixed
  label01 training sets;
- a unified stable PPO submission path that keeps per-sample labels visible in
  logs;
- separate full-pool generation and selection for unified label0 and label1
  outputs;
- a final overlap analysis over the selected category-level fragment sets.

### Decision
Extend the repository with a unified label-conditioned PPO experiment layer
without rewriting selector/merge tooling or the legacy PPO entrypoint:

- add `src/data/unified_ppo_prompts.py` plus thin CLIs for:
  - building label-specific PPO prompts,
  - building balanced unified label01 prompts,
  - checking unified prompt balance by 50-row blocks;
- keep `scripts/train_ppo_stable.py` as the stable PPO entrypoint, but add an
  opt-in `[UNIFIED_PPO_SAMPLE]` per-sample logging path so unified runs can be
  audited label-wise without creating a parallel trainer;
- extend `scripts/analyze_stable_ppo_log.py` to parse the new sample logs and
  report label0/label1 metrics by training block while preserving old log
  behavior when the new tag is absent;
- reuse the existing full-pool generator and selector for unified label0 and
  label1 pools through new Slurm wrappers only;
- add `src/eval/selected_subgraph_overlap.py` and a thin CLI that compares
  selected fragment sets by exact canonical overlap, soft Morgan overlap, and
  Murcko scaffold overlap.

### Alternatives considered
1. Fork `train_ppo_stable.py` into a second unified trainer.
2. Compare full-pool overlap only and skip selector-level overlap.
3. Build unified prompts with a new schema that drops compatibility anchors like
   `ORIGINAL_LABEL`, `MOLECULE_SMILES`, and `FRAGMENT_SMILES`.

### Consequences
- Unified PPO can now be tested with minimal risk to the existing label=1
  stable300 path because the main trainer is reused rather than duplicated.
- Selector-level overlap becomes a first-class analysis artifact instead of an
  ad hoc notebook step.
- The unified prompt format explicitly conditions on the original label while
  still preserving compatibility anchors expected by current PPO data loaders
  and generation helpers.

### Status
Accepted

---

## [2026-05-20] Add class-level MMR selector and diversity-side pool merge tooling

### Background
Stable PPO training has already converged far enough for offline pool building,
and the stable300 full candidate pool audit now says the pool is suitable for a
selector. At the same time, the audit still flags high mode-collapse risk and
recommends sampling tuning rather than more PPO steps. That means the next phase
needs two things:

- a class-level selector that can turn a large candidate pool into a shared,
  low-redundancy fragment set;
- a safe way to compare the current base pool against a higher-temperature pool
  and optionally merge them without touching any PPO training code.

### Decision
Add a selector-and-merge layer around the existing full-pool generation and
audit pipeline:

- `src/eval/class_counterfactual_selector.py` implements a greedy MMR selector
  over filtered counterfactual candidates, scoring fragments by shared
  counterfactual strength, marginal parent coverage, redundancy penalty, and
  atom-ratio size regularization.
- `scripts/select_class_counterfactual_subgraphs.py` exposes that selector as a
  thin CLI and writes JSON, CSV, summary JSON, and TXT report artifacts.
- `src/eval/candidate_pool_merge.py` and
  `scripts/merge_candidate_pools.py` merge multiple pool JSONLs while
  deduplicating by `(final_fragment, parent_smiles)` and keeping the
  higher-scoring candidate.
- New Slurm wrappers cover:
  - base-pool selector runs,
  - high-temperature stable300 full-pool generation plus audit,
  - merged-pool creation plus audit,
  - merged-pool selector runs.

The implementation explicitly reuses the existing candidate-pool normalization
contract from `src/eval/candidate_pool_audit.py` so selector filtering stays
compatible with current and future pool field aliases.

### Alternatives considered
1. Continue PPO training to chase diversity improvements.
2. Pick fragments only by raw `cf_drop` without redundancy control.
3. Rebuild the pool schema specifically for selector consumption.

### Consequences
- Selector development is now decoupled from PPO training and can iterate on
  existing stable300 pools.
- Diversity recovery can be tested through higher-temperature sampling and pool
  merging without changing stable300 checkpoints.
- Shared field normalization between audit and selector reduces schema drift
  risk when pools come from slightly different generation paths.

### Status
Accepted

## 2026-05-17 Stable300 Full Candidate Pool Wrapper

### Background
The stable PPO diagnostic path has already converged on
`decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`
as the current selector-ready checkpoint. We do not want to continue PPO
training, rewrite the existing pool audit, or fork new reward logic just to
prepare the full label=1 candidate pool.

### Decision
Reuse the existing full-pool generation and candidate-pool audit entrypoints:

- `scripts/generate_full_candidate_pool.py`
- `scripts/audit_candidate_pool.py`

and add a single Slurm wrapper:

- `scripts/slurm/generate_and_audit_full_pool_stable300_label1_n4.sh`

The wrapper generates a 4-candidate-per-parent full pool for the complete
label=1 PPO prompt CSV, skips regeneration when a non-empty pool already
exists, and then audits the resulting JSONL with the existing
selector-facing audit script.

### Alternatives considered
1. Add a new stable-specific Python generation pipeline.
2. Re-run PPO or tune stable300 before building the full pool.
3. Write a second full-pool audit implementation tailored to selector prep.

### Consequences
- Stable300 full-pool generation stays aligned with the same chemistry,
  projection, oracle, and candidate-pool schema already used elsewhere.
- The workflow remains `git pull` + `sbatch`, with no checkpoint mutation and
  no PPO code changes.
- Existing `candidate_pool.jsonl` runs can be resumed safely because the Slurm
  wrapper only regenerates when `FORCE_REGEN=true` or the pool is missing /
  empty.

### Status
Accepted

---

## [2026-05-17] Add a parallel conservative stable-PPO path without modifying the existing PPO entrypoint

### Background
The repository already had a working decoded-chem PPO path and a best short-run
checkpoint from the original shuffled prompt order, but longer 150/200/300-step
runs showed drift symptoms:

- later-step reward and `cf_flip` dropped;
- `approx_kl` rose in the back half of longer runs;
- short shuffled PPO remained more reliable than simply extending ordinary PPO.

The user explicitly required that the original PPO code remain untouched so the
team could run apples-to-apples control experiments between:

1. the existing PPO path, and
2. a new conservative / stable PPO path with stronger KL control and lower
   update aggressiveness.

### Decision
Add a new stable-only training and analysis path that stays fully parallel to
the original PPO entrypoint:

- `scripts/train_ppo_stable.py` is a new decoded-chem PPO entrypoint that
  reuses existing loaders / model builders / reward helpers but implements its
  own conservative PPO update behavior;
- `src/rewards/reward_wrapper_stable.py` adds a stable-only post-processing
  layer around the existing reward wrapper so teacher-confidence gating can be
  applied without changing default reward behavior in ordinary PPO;
- the stable PPO path now supports optional environment / CLI overrides for:
  lower learning rate, smaller clip range, fewer PPO epochs, explicit gradient
  clipping, reward clipping, optional reward / advantage normalization, target
  and hard KL monitoring, adaptive KL penalty, teacher-confidence gating, and
  validation-based best-checkpoint / early-stop logic;
- new Slurm wrappers were added for:
  - a 5-step smoke run, and
  - a 200-step conservative shuffled-label1 run;
- `scripts/make_stratified_ppo_prompts.py` adds a new stratified shuffle tool
  for PPO prompt CSVs;
- `scripts/analyze_stable_ppo_log.py` adds a stable-PPO segment analyzer for
  1-50 / 51-100 / 101-150 / 151-200 blocks.

### Alternatives considered
1. Modify `scripts/train_ppo.py` in place to add stable flags.
2. Change the default reward wrapper behavior globally.
3. Keep using only the original shuffled short-run PPO path and skip a
   conservative long-run branch entirely.

### Consequences
- The old PPO entrypoint and its paired Slurm script remain unchanged, which
  preserves backward compatibility and protects current baselines.
- Conservative long-run PPO can now be tested as a parallel branch rather than
  as a behavior change hidden behind new flags in the original script.
- Teacher-confidence gating and stable-KL logic are isolated to the new stable
  path, which keeps the default reward semantics unchanged for existing runs.

### Status
Accepted

---

## [2026-05-17] Add full-dataset candidate-pool generation and selector-facing audit for original shuffle100 before any further PPO training

### Background
By this point the project had already completed:

- SFT v3;
- decoded PPO diagnostics;
- teacher-confidence filtering;
- ordered-vs-shuffle control experiments.

The working conclusion shifted from "keep extending PPO" to "treat the current
best shuffled short-run checkpoint as a candidate generator and measure whether
its full label=1 pool is good enough for the downstream class-level selector."

The current best checkpoint is the original shuffled 100-step run
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`.
The next priority is therefore not more PPO optimization, but:

1. generate a full label=1 candidate pool with multiple candidates per parent,
   without any PPO updates;
2. compare that pool against an SFT-only baseline;
3. audit legality, counterfactual utility, diversity, redundancy, and parent
   coverage in a selector-facing format.

### Decision
Add a separate full-pool inference and audit path that stays outside SFT and
PPO training logic:

- `src/data/ppo_prompt_dataset.py` now provides one normalized loader for PPO
  prompt CSV / JSONL files, including fallbacks for `parent_smiles`, `smiles`,
  prompt-text SMILES extraction, and prompt reconstruction when needed;
- `src/eval/full_candidate_pool.py` now provides:
  - checkpoint inspection helpers for adapter-root vs `checkpoint-*` layouts,
  - adapter load-path resolution,
  - offline multi-candidate generation over the full prompt pool,
  - reward/evaluator reuse through `ChemRLRewarder.compute_rewards_from_decoded(...)`,
  - JSONL row enrichment so the output is compatible with existing
    `candidate_pool.jsonl` consumers while also exposing new selector-facing
    aliases such as `final_fragment`, `projection_used`, `final_substructure`,
    and `cf_oracle_called`;
- `scripts/generate_full_candidate_pool.py` is the thin CLI entrypoint for
  full-dataset inference with either:
  - SFT-only mode: base model + SFT LoRA, or
  - PPO mode: base model + resolved PPO adapter;
- `src/eval/full_candidate_pool_audit.py` now computes a richer audit than the
  earlier checkpoint-level candidate-pool audit, including:
  - pool scale,
  - legality,
  - counterfactual-oracle usage,
  - size statistics,
  - diversity / redundancy,
  - selector-facing fragment coverage over the full label=1 parent set,
  - explicit failure-case export;
- `scripts/audit_full_candidate_pool.py` is the thin CLI wrapper for that
  selector-facing full-pool audit;
- new Slurm wrappers now support one-command HPC generation / audit for:
  - PPO original shuffle100,
  - optional chained generate+audit,
  - SFT-only baseline;
- the shuffle200 path is kept as a future template only; it is not executed by
  default.

### Alternatives considered
1. Wait for shuffle200 to finish before building any full-pool generation path.
2. Reuse only partial `candidate_pool.jsonl` artifacts from training steps
   instead of running dedicated full-dataset inference.
3. Build a separate reward implementation for inference-time pool scoring.

### Consequences
- The repository now has a dedicated offline candidate-generation path that
  reuses the existing decoded-chem reward/evaluator instead of duplicating the
  reward logic.
- Selector readiness can now be judged from saved full-pool artifacts rather
  than from short PPO logs or early-step candidate traces alone.
- The checkpoint-inspection helper makes the saved-adapter assumption explicit:
  decoded-chem PPO saves the final adapter at the run root when adapter files
  are present there; only if those files are missing do we need to fall back to
  a `checkpoint-*` subdirectory.
- The work remains completely outside SFT data construction, SFT training, PPO
  optimization logic, and reward semantics.

### Status
Accepted

---

## [2026-05-17] Add a teacher-confidence filter for PPO prompt pools before continuing long decoded-chem PPO runs

### Background
The reward/teacher audit on the current best short-run checkpoint
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500`
showed a split conclusion:

- decoded PPO itself still looked structurally healthy, with
  `cf_oracle_called_rate` near 1.0 and no obvious projection or size loophole;
- but teacher reliability on the label=1 parent pool was only moderately
  trustworthy, with `teacher_correct_rate≈0.855`, `low_confidence_rate≈0.144`,
  and `very_low_confidence_rate≈0.104`.

That means the next priority is not to keep lengthening PPO blindly, nor to
rewrite reward shaping immediately. The more controlled next step is to reduce
teacher-side noise in the PPO prompt pool and compare short filtered runs
against the current baseline.

### Decision
Add a standalone teacher-confidence filtering path for PPO prompt CSV files:

- new filtering logic now lives in `src/data/teacher_confidence_filter.py`;
- `scripts/filter_ppo_prompts_by_teacher_confidence.py` provides a thin CLI
  wrapper that:
  - reads a PPO prompt CSV,
  - resolves `parent_smiles` with fallback to `smiles` and prompt text,
  - scores each parent with `TeacherSemanticScorer`,
  - keeps only rows satisfying target-label, `teacher_result_ok`,
    optional teacher-correctness, and minimum `p_label`;
- `scripts/slurm/filter_ppo_prompts_teacher_p05_label1.sh` hardcodes the
  current label=1 PPO prompt CSV, the AIDS RF teacher, and the
  `p_label >= 0.5 && teacher_correct` filter so HPC usage stays one-command
  simple via
  `sbatch scripts/slurm/filter_ppo_prompts_teacher_p05_label1.sh`;
- `scripts/slurm/train_ppo.sh` now accepts an optional `DATASET_PATH`
  environment variable and forwards it to `--dataset-path`, so filtered prompt
  files can be used without changing PPO training code.

### Alternatives considered
1. Continue training 150/200/300-step PPO on the unfiltered prompt pool first.
2. Immediately redesign the reward function despite the audit not showing a
   clear reward loophole yet.
3. Create a separate custom PPO training wrapper instead of teaching the
   existing Slurm wrapper to accept a dataset override.

### Consequences
- The project can now run a cleaner apples-to-apples experiment:
  unfiltered PPO-100/150 versus teacher-filtered PPO-100/150.
- Teacher-side uncertainty is reduced before spending more A800 time on longer
  decoded-chem PPO runs.
- The change remains outside SFT dataset construction, SFT training, and PPO
  reward logic; it only filters prompt inputs and improves Slurm parameter
  plumbing.

### Status
Accepted

---

## [2026-05-17] Add an independent reward/teacher audit entrypoint for diagnosing PPO-100 vs long-run degradation

### Background
The current decoded PPO workflow reached a point where `PPO-100` looked better
than 150/200/300-step checkpoints on reward, direct-substructure rate, and
parse/core usability, but that trend alone could not distinguish between:

- benign PPO policy drift after longer optimization;
- data-order / prompt-difficulty effects from short sequential training;
- an actual problem in the reward function or teacher/oracle behavior.

The team therefore needed a standalone audit path that could inspect teacher
reliability on original parents and analyze `candidate_pool.jsonl` for
counterfactual-oracle coverage, reward alignment, projection shortcuts, and
size shortcuts, without changing SFT v3 data construction, SFT training, or
PPO main training logic.

### Decision
Add a separate reward/teacher diagnosis entrypoint that stays fully outside the
training loop:

- new audit logic now lives in `src/eval/reward_teacher_audit.py`;
- `scripts/audit_reward_teacher.py` provides a thin CLI for dataset-backed
  teacher reliability plus candidate-pool reward auditing;
- `scripts/slurm/audit_reward_teacher.sh` hardcodes the current best 100-step
  candidate pool, the label=1 PPO prompt dataset, and the AIDS random-forest
  teacher so HPC usage stays one-command simple via
  `sbatch scripts/slurm/audit_reward_teacher.sh`;
- the audit explicitly tolerates candidate-pool schema drift by accepting
  compatibility aliases such as `p_before/teacher_p_before`,
  `p_after/teacher_p_after`, `cf_drop/counterfactual_drop/teacher_cf_drop`,
  `cf_flip/counterfactual_flip`, and
  `counterfactual_reason/cf_reward_skipped_reason`;
- final outputs now separate six questions the team cares about most:
  teacher reliability, cf-oracle skip/deletion failure pressure, reward-to-cf
  alignment, projection loopholes, size loopholes, and whether current
  degradation looks more like PPO drift or reward/teacher trouble.

### Alternatives considered
1. Reuse only PPO training logs and avoid any new structured audit.
2. Fold reward/teacher diagnosis into `train_ppo.py`, making the training
   entrypoint even more coupled to analysis logic.
3. Jump directly to shuffle-100 or multi-seed-100 experiments before first
   checking whether the current reward/teacher pipeline is itself suspicious.

### Consequences
- PPO checkpoint diagnosis becomes reproducible from saved artifacts instead of
  depending on partial training logs.
- Teacher/oracle reliability on original parents can now be audited directly,
  which helps separate model-side label noise from PPO optimization effects.
- Projection and oversized-fragment shortcuts become explicit audit outputs
  instead of vague hypotheses during long-run PPO comparisons.
- The change remains evaluation-only and does not alter SFT data, SFT training,
  or PPO optimization behavior.

### Status
Accepted

---

## [2026-05-14] Add a selector-facing candidate-pool audit entrypoint for checkpoint-level PPO evaluation

### Background
The current decoded PPO workflow has already reached the stage where the
important question is no longer "can PPO run?" but "is a short-run checkpoint a
good candidate-pool generator for the downstream class-level selector?" The
team identified the 100-step run
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500` as a
better candidate than the longer 300-step run, which showed more policy drift.
At that point, continuing long PPO training was less urgent than auditing the
quality, diversity, projection dependence, and counterfactual utility of the
already generated `candidate_pool.jsonl`.

### Decision
Add a dedicated audit path for PPO candidate pools without touching SFT v3
dataset construction, SFT training, PPO loss, teacher/oracle logic, or
projection search:

- new selector-facing audit logic now lives in
  `src/eval/candidate_pool_audit.py`;
- `scripts/audit_candidate_pool.py` provides a thin CLI wrapper for JSON/TXT
  reports over `candidate_pool.jsonl`;
- `scripts/slurm/audit_candidate_pool.sh` is pinned to the current 100-step run
  so HPC usage stays one-command simple via
  `sbatch scripts/slurm/audit_candidate_pool.sh`;
- the audit uses field-compatibility fallbacks across reward-trace schema
  variants instead of assuming one rigid JSONL shape, because the pool rows
  evolved during recent projection-penalty and distance-reward debugging;
- final recommendations are driven by selector-oriented heuristics such as
  final-substructure rate, projection-used rate, cf-flip rate, diversity, and
  Morgan-Tanimoto redundancy.

### Alternatives considered
1. Reuse training logs only and skip a structured `candidate_pool.jsonl` audit.
2. Fold candidate-pool analysis into `train_ppo.py`, mixing evaluation logic
   back into the training entrypoint.
3. Continue toward 150-step/200-step/1000-step PPO runs first and postpone
   selector-oriented auditing.

### Consequences
- Short PPO runs can now be evaluated as generator checkpoints before spending
  more compute on longer RL training.
- Candidate-pool readiness for the downstream selector becomes explicit and
  reproducible through saved JSON and TXT reports.
- The audit remains decoupled from training so the class-level selector can
  iterate independently of PPO runtime changes.

### Status
Accepted

---

## [2026-05-14] Make decoded PPO failure traces tolerate forward-compatible reward-debug fields

### Background
After enabling both projection penalty and substructure distance reward in the
decoded PPO diagnostic, normal reward paths worked for the first few samples,
but a later parseable non-direct fragment crashed inside
`ChemRLRewarder._fail(...)` with:

`TypeError: ChemRLRewarder._fail() got an unexpected keyword argument 'projection_penalty_config'`

The reward path had already been extended to merge richer debug dictionaries
containing fields such as `projection_penalty_config`,
`projection_penalty_applied`, `reward_before_projection_penalty`,
`reward_after_projection_penalty`, and `subdist_contribution`. Successful
traces handled those fields, but failure branches still routed everything
through an older `_fail(...)` signature with a closed keyword list. As soon as a
failure path inherited a new trace field, PPO aborted instead of returning a
penalized failure trace.

### Decision
Keep reward semantics unchanged and only harden the failure-trace plumbing:

- `_fail(...)` now accepts the currently expanded projection-penalty fields
  explicitly and also captures future trace extensions through
  `**extra_trace_fields`;
- failure trace construction now uses a small merge helper that appends only
  real `RewardTrace` dataclass fields and never lets unknown logging keys crash
  PPO;
- explicit/core `_fail(...)` arguments still have priority, while future fields
  can populate newly added `RewardTrace` slots without requiring every old call
  site to be rewritten immediately;
- parse-failed, core-unusable, and parseable-but-not-direct branches now all
  keep returning structured failure traces even when projection/distance debug
  fields are present.

### Alternatives considered
1. Add only `projection_penalty_config` to the `_fail(...)` signature and leave
   the rest of the trace field flow unchanged.
2. Strip all non-core debug fields before failure handling and accept reduced
   observability on error paths.
3. Move failure-trace construction out of `_fail(...)` entirely and duplicate
   trace assembly across call sites.

### Consequences
- PPO no longer aborts when a failure branch inherits newer reward-debug fields.
- Failure rows in `CHEM_REWARD_COMPONENTS` and `candidate_pool.jsonl` retain the
  same projection and distance diagnostics as success rows, which keeps reward
  debugging consistent through bad generations.
- Future reward-trace extensions are less likely to require emergency fixes in
  `_fail(...)`, as long as they also become `RewardTrace` fields.

### Status
Accepted

---

## [2026-05-14] Apply projection penalty inside decoded PPO reward breakdown whenever a non-direct fragment needs a successful parent projection

### Background
After enabling both projected-cf reward and substructure distance reward, the
decoded PPO diagnostic logs correctly showed
`[PROJECTED_CF_REWARD_CONFIG] enabled=True`,
`[SUBSTRUCTURE_DISTANCE_REWARD_CONFIG] enabled=True`, and non-zero
`subdist_contribution`. However, `projection_penalty` still stayed at `0.0` in
`[CHEM_REWARD_COMPONENTS]` even when `PROJECTION_PENALTY=1.0` was passed through
the Slurm environment and parseable non-direct fragments were being rescued by a
nearest parent subgraph or by the projected counterfactual path.

The core issue was that projection diagnostics were being carried only as trace
metadata, while reward aggregation never subtracted the configured penalty from
`reward_total`. In the main non-direct branch, later trace merges also replaced
the projection-debug fields with a distance-reward trace that defaulted the
penalty back to `0.0`.

### Decision
Keep the projection search algorithm, teacher/oracle calls, and PPO loss logic
unchanged, and fix only the reward aggregation and observability layer:

- decoded reward breakdowns now carry explicit
  `projection_penalty_config`, `projection_penalty_applied`,
  `reward_before_projection_penalty`, and
  `reward_after_projection_penalty` fields;
- `reward_total` is now the post-penalty value:
  `reward_after_projection_penalty =
  reward_before_projection_penalty - projection_penalty_applied`;
- the penalty is applied whenever a fragment is not a direct parent
  substructure and the nearest-parent projection path succeeded
  (`projection_attempted=True` and `projection_success=True`);
- direct-substructure examples keep `projection_penalty_applied=0.0`;
- parse-failed / core-unusable examples that never reached a successful
  projection path also keep `projection_penalty_applied=0.0`;
- `scripts/train_ppo.py` now also accepts `PROJECTION_PENALTY` as an env-backed
  default for direct local launches, while `scripts/slurm/train_ppo.sh`
  continues forwarding `--projection-penalty`.

### Alternatives considered
1. Apply projection penalty only when
   `used_projected_subgraph_for_reward=True`, leaving other successful nearest
   parent projections unpenalized.
2. Push the penalty into PPO loss code instead of the chemistry reward
   breakdown.
3. Keep penalty logging separate and ask users to post-process logs to estimate
   the effective reward.

### Consequences
- Projection dependence now affects `reward_total` directly instead of being a
  logging-only diagnostic.
- Logs and `candidate_pool.jsonl` rows now show both the configured penalty and
  the actually applied deduction, making it easier to audit whether projected
  rescue is being overused.
- The fix remains local to reward composition and runtime config plumbing; it
  does not touch SFT datasets, PPO update math, teacher/oracle scoring, or the
  parent-projection retrieval algorithm itself.

### Status
Accepted

---

## [2026-05-14] Make substructure distance reward a first-class PPO shaping term with explicit runtime config and contribution logging

### Background
The decoded PPO diagnostic path already computed nearest-parent-subgraph
distance fields such as `substructure_similarity`,
`substructure_distance_reward`, and the shorter `subdist_*` aliases, but recent
projected-cf reward runs still surfaced `subdist_weight=0.0` inside
`[CHEM_REWARD_COMPONENTS]`. In practice that meant the dense distance reward was
being logged without actually contributing to `reward_total`, mostly because the
generic HPC wrapper `scripts/slurm/train_ppo.sh` never forwarded the distance
reward enable/weight knobs even though `scripts/train_ppo.py` and the rewarder
already knew about them.

### Decision
Keep the decoded PPO loss flow, SFT pipeline, teacher/oracle stack, and
projection semantics unchanged, and fix only the reward configuration,
parameter plumbing, and diagnostics:

- `scripts/train_ppo.py` now resolves the canonical runtime switch
  `--enable-substructure-distance-reward` /
  `--no-enable-substructure-distance-reward` explicitly, rejects a CLI conflict,
  supports the env alias `SUBDIST_WEIGHT` alongside
  `SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT`, and logs the resolved runtime state via
  `[SUBSTRUCTURE_DISTANCE_REWARD_CONFIG]`;
- when the feature is disabled, the effective runtime weight is forced to
  `0.0`, so logs and reward traces cannot silently show a stale positive weight;
- `src/rewards/reward_wrapper.py` now surfaces
  `subdist_contribution = subdist_weight * subdist_reward` explicitly in both
  breakdowns and decoded reward logs, while preserving the legacy
  `subdist_weighted_r` key for compatibility;
- `scripts/slurm/train_ppo.sh` now exports, echoes, and forwards the full
  distance-reward knob family so local Codex edits and HPC `sbatch` runs stay in
  sync;
- the recommended default shaping weight is now `0.3`, which keeps distance
  reward active as a conservative continuous constraint without overriding hard
  failure penalties or projected-cf reward behavior.

### Alternatives considered
1. Leave the reward wrapper unchanged and only document that users must switch
   to a dedicated `train_decoded_chem_ppo_subdist_reward.sh` script.
2. Broaden the change into a larger reward refactor that changes PPO loss
   semantics or projection behavior.
3. Introduce a second negative CLI spelling such as
   `--disable-substructure-distance-reward`.

### Consequences
- Enabling distance reward now makes `subdist_weight > 0` and
  `subdist_contribution != 0` visible in reward traces when the fragment earns
  non-zero dense similarity reward.
- Disabled runs remain explicit and unambiguous because both the config log and
  the trace fields resolve to `weight=0.0`.
- Projected counterfactual reward and distance reward can now be diagnosed
  together: projected fragments may still power deletion-based cf reward, while
  the raw/core fragment keeps its own nearest-parent-subgraph shaping term.

### Status
Accepted

---

## [2026-05-10] Add an explicit enable switch for projected counterfactual reward and route successful projections into the deletion teacher

### Background
The decoded PPO stack already surfaced projection diagnostics such as
`projection_attempted`, `projection_success`, and `projected_fragment`, but the
projected counterfactual reward path never actually became live in practice.
Two gaps caused that:

- the CLI/HPC workflow only exposed the legacy negative flag name
  `disable_projected_cf_reward`, so omitting the flag still left the feature
  disabled by default;
- `ChemRLRewarder` logged projected subgraphs for non-direct fragments, but the
  non-direct reward branch still skipped the counterfactual teacher
  unconditionally and never consumed the projected fragment even when a legal
  parent subgraph was available.

### Decision
Keep projected counterfactual reward disabled by default, preserve the old
negative flag for compatibility, and add one explicit positive path:

- `scripts/train_ppo.py` now accepts `--enable-projected-cf-reward` and resolves
  it against the legacy `--disable-projected-cf-reward` flag with an explicit
  conflict error;
- `scripts/slurm/train_ppo.sh` now forwards
  `ENABLE_PROJECTED_CF_REWARD=true/false` and the legacy disable env override;
- `ChemRLRewarder` now uses a projected parent subgraph for deletion-based
  counterfactual reward only when all gating conditions are satisfied:
  projected-cf reward enabled, parent projection enabled, projection attempted
  and successful, and the projected fragment revalidates as a legal parent
  substructure;
- the raw model output stays unchanged in logs and candidate-pool rows; the
  projected fragment is tracked separately through
  `projected_fragment_smiles` / `used_projected_subgraph_for_reward`.

### Alternatives considered
1. Keep the old negative-only flag and rely on `--no-disable-...` style usage.
2. Treat projection diagnostics as logging-only forever and never reuse them for
   counterfactual reward.
3. Replace the model output with the projected fragment downstream.

### Consequences
- Default behavior remains unchanged: projected counterfactual reward is off
  unless the user explicitly enables it.
- When enabled, parseable non-direct fragments can now receive deletion-based
  counterfactual reward through a legal projected parent subgraph without
  pretending that the model originally emitted that projected fragment.
- Logs now make the resolved runtime state explicit via
  `[PROJECTED_CF_REWARD_CONFIG]`, and relevant reward traces surface
  `used_projected_subgraph_for_reward=True` when the projected path is actually
  used.

### Status
Accepted

---

## [2026-05-10] Deduplicate decoded PPO failure-trace kwargs before calling `_fail`

### Background
The decoded chemistry PPO reward path correctly handled early success cases, but
some reward-failure branches accumulated debug fields from multiple trace
dictionaries and then passed them into `ChemRLRewarder._fail(...)` alongside
explicit keyword arguments. When a failure dict already contained fields such as
`direct_substructure`, `projection_attempted`, or
`substructure_distance_reward`, Python raised
`TypeError: ... got multiple values for keyword argument ...` instead of
returning a penalized failure trace.

### Decision
Keep the decoded PPO objective, reward semantics, projection behavior, and
training CLI unchanged, and harden only the failure-trace assembly:

- add a dedicated `_merge_failure_fields(...)` helper in
  `src/rewards/reward_wrapper.py` so `_fail(...)` kwargs are merged into one
  dict before the call;
- update every `_fail(...)` call site that mixed explicit kwargs with trace-dict
  expansion to use the merged call pattern;
- add regression tests covering the public
  `compute_rewards_from_decoded(...)` path for parseable-but-not-direct
  fragments, plus a direct `_fail(...)` assembly test where the extra trace dict
  already contains `direct_substructure=False`.

### Alternatives considered
1. Remove only the current duplicated `direct_substructure` field from one
   failing branch.
2. Drop projection/subdistance diagnostics from failure traces entirely.
3. Broaden the change into a larger PPO reward refactor.

### Consequences
- Reward failure branches such as parseable non-substructures now return logged
  negative/low-value traces instead of aborting PPO with a keyword-collision
  exception.
- Existing diagnostics remain visible in `[CHEM_REWARD_COMPONENTS]`, including
  `failure_tag`, `invalid_detail`, `direct_substructure`,
  `projection_attempted`, `projection_success`, `projection_method`, and
  `reward_total`.
- The fix stays local to reward-trace assembly and does not change SFT data,
  PPO loss flow, teacher scoring logic, or HPC launch semantics.

### Status
Accepted

---

## [2026-05-10] Standardize decoded PPO initialization on SFT_LORA_PATH with INIT_LORA_PATH alias and explicit init logging

### Background
The decoded chemistry PPO path already expected an SFT-initialized LoRA policy,
but the repository exposed that checkpoint through a mix of names depending on
which layer a user looked at: `--sft-lora-path` in `scripts/train_ppo.py`,
`SFT_LORA_PATH` in the generic Slurm wrapper, and several older diagnostic
scripts that still hardcoded a specific checkpoint path. This made it too easy
to launch PPO from the wrong adapter when switching from an older checkpoint to
the newer SFT v3 HIV runs.

### Decision
Keep `SFT_LORA_PATH` / `--sft-lora-path` as the canonical decoded PPO
initialization path, and add one narrow compatibility alias plus stronger
runtime logging:

- `scripts/train_ppo.py` now accepts `--init-lora-path` as a compatibility
  alias, with precedence `--sft-lora-path` > `--init-lora-path` >
  `--sft-adapter-path`;
- the PPO runtime manifest and logs record both the raw init-path arguments and
  the final resolved checkpoint path plus its source field;
- `scripts/slurm/train_ppo.sh` and
  `scripts/slurm/train_decoded_chem_ppo_full.sh` now accept `INIT_LORA_PATH`
  as an environment-variable alias, resolve one final init LoRA path, echo it,
  and warn if both alias names are set to different values.

### Alternatives considered
1. Leave the existing `SFT_LORA_PATH` support unchanged and rely on users to
   inspect each Slurm script manually.
2. Rename every PPO entrypoint to one new variable and break the older wrappers.
3. Broaden the compatibility layer to ambiguous names such as `CHECKPOINT_PATH`.

### Consequences
- The canonical answer for decoded PPO initialization remains:
  use `SFT_LORA_PATH` / `--sft-lora-path`.
- Existing workflows keep working, while `INIT_LORA_PATH` can be used as a
  clear compatibility alias in shared shell snippets.
- PPO logs now make it obvious which LoRA checkpoint was actually loaded for
  policy/reference initialization.

### Status
Accepted

---

## [2026-05-10] Normalize legacy SFT JSONL columns to TRL prompt-completion format at train time

### Background
The rebuilt SFT v3 datasets and some older SFT exports were still centered on
`instruction` / `output` audit fields, and the current HIV builder also emitted
`prompt` plus `response` without a `completion` alias. On HPC, TRL's
`SFTTrainer` entered prompt-completion tokenization mode and failed early with
`KeyError: 'completion'`, even though the train/validation splits themselves
were valid and readable.

### Decision
Keep the existing SFT data objective, candidate generation, filtering, and PPO
logic unchanged, and add a narrow compatibility layer for SFT-only training:

- `scripts/train_sft.py` now normalizes loaded JSONL rows before constructing
  `SFTTrainer`, preserving legacy audit fields while materializing
  `prompt` / `completion` when possible;
- the normalization supports three input shapes:
  direct `prompt` / `completion`, legacy `instruction` / `output` with optional
  `input`, and the current builder's `prompt` / `response` alias;
- completions are prefixed with a separator newline when synthesized so prompt
  and fragment text do not concatenate silently;
- train/eval startup now logs normalized column names plus prompt/completion
  previews, and raises a clear `ValueError` with available columns when a split
  still lacks the required fields;
- the SFT v3 builder now writes a `completion` alias in new JSONL outputs so
  future datasets are directly compatible with TRL prompt-completion mode.

### Alternatives considered
1. Require every existing SFT dataset to be rebuilt before retraining.
2. Patch TRL usage to rely only on a concatenated `text` column and skip
   prompt-completion compatibility entirely.
3. Broaden the fix into a larger SFT data-schema refactor.

### Consequences
- Existing `instruction` / `output` JSONL files can be trained directly without
  a dataset rebuild.
- Current builder outputs are now safer for TRL because `completion` is written
  explicitly in addition to the preserved legacy fields.
- SFT failures around missing text columns now surface with actionable dataset
  diagnostics instead of an internal TRL `KeyError`.

### Status
Accepted

---

## [2026-05-09] Add nearest-parent-subgraph distance reward and stop using projected fragments for counterfactual reward

### Background
The decoded PPO path previously had one high-risk mismatch with the v3
counterfactual objective: when a parseable core fragment was not a strict
parent substructure, the reward wrapper could retrieve one projected legal
parent subgraph and then continue deletion-based reward computation on that
projected fragment instead of on the model's own output. This hid the true
failure mode and turned the projection module into an answer-rewriting path
rather than a structural-distance diagnostic.

### Decision
Keep the strict exact-substructure reward, but separate it from a new dense
auxiliary reward based on the nearest legal connected parent subgraph:

- direct parent matches keep the exact binary substructure reward and remain the
  only cases allowed to call the deletion-based counterfactual teacher;
- parseable but non-direct fragments now compute
  `substructure_similarity / substructure_distance / substructure_distance_reward`
  against the nearest legal parent subgraph built from the existing
  parent-derived candidate pool;
- the nearest legal parent subgraph is logged for debugging only and is never
  substituted back into the reward path for counterfactual deletion scoring;
- non-direct fragments explicitly log
  `used_projected_subgraph_for_reward=False` and
  `cf_reward_skipped_reason=not_direct_substructure`;
- the decoded PPO CLI and Slurm path expose dedicated knobs for enabling the
  new dense reward and tuning its candidate window and MCS settings.

### Alternatives considered
1. Keep the projection-retrieval path as the effective reward fragment and only
   add more logging around it.
2. Treat all parseable non-substructure outputs as hard failures with zero
   dense structural feedback.
3. Replace the distance reward with fragment-only teacher semantics.

### Consequences
- Reward logs now distinguish exact direct substructure success from
  non-direct-but-similar fragments.
- Deletion-based counterfactual reward is again aligned with the model output
  instead of a projected replacement fragment.
- PPO can still receive a dense structural signal for near-miss fragments
  without allowing reward leakage through projection.

### Status
Accepted

---

## [2026-05-09] Replace the SFT v3 HIV scaffold split with a label-stratified scaffold holdout

### Background
The rebuilt raw-HIV SFT v3 dataset was already structurally healthy, but the
existing `scaffold_group_greedy` split could severely distort validation label
balance. In the observed full build, validation drifted toward almost all
positives (`{'0': 24, '1': 404}`), making the split unrepresentative even
though scaffold overlap remained zero.

### Decision
Keep candidate generation, fragment filtering, oracle ranking, and text target
format unchanged, and only replace the train/val split logic in
`src/data/sft_v3_builder.py`:

- make the default split objective explicitly label-stratified at the scaffold
  group level, so validation selection optimizes per-label target counts before
  raw total-count closeness;
- preserve scaffold-level holdout by assigning each effective scaffold group to
  exactly one split;
- treat missing/acyclic scaffolds as stable per-example pseudo-scaffolds during
  splitting so they do not collapse into one oversized group;
- surface split diagnostics such as total/target/actual label counts,
  per-label target error, actual val ratio, and per-label val ratio.

### Alternatives considered
1. Keep the old global scaffold greedy split and only tweak weights slightly.
2. Fall back to pure label-stratified random splitting and give up scaffold
   holdout.
3. Rebuild the dataset again with different parent sampling instead of fixing
   the split itself.

### Consequences
- Validation should remain much closer to the global 2:1 label mix while still
  keeping scaffold overlap at zero in normal cases.
- Large scaffold groups can still introduce small count error, but the error is
  now explicit in the split summary/report instead of being silent.
- The rebuild path stays fully compatible with the existing SFT build, audit,
  train, and eval scripts.

### Status
Accepted

---

## [2026-05-08] Restore and harden the raw HIV -> SFT v3 rebuild path as a first-class workflow

### Background
The repository already had a scaffold-aware SFT v3 rebuild pipeline, but the
active worktree had lost `src/data/hiv_dataset_utils.py`, which broke the raw
HIV.csv normalization and parent-sampling path outright. At the same time, the
HPC build wrapper and human-readable reports needed a bit more explicit
bookkeeping so the negative-pool sampling behavior remained easy to audit when
rebuilding a larger SFT initializer for later PPO runs.

### Decision
Keep the existing raw-HIV -> parent-derived reference build strategy, and make
the following operational hardening changes instead of inventing a new data
objective:

- restore `src/data/hiv_dataset_utils.py` as the source of truth for flexible
  HIV column detection, label normalization, scaffold extraction, and
  scaffold+size diversity sampling;
- preserve the current parent-derived candidate path in
  `src/data/sft_v3_builder.py`, but expose clearer selection/report metadata
  for positive/negative queue sizes, stratum counts, and raw label tokens;
- standardize the HPC default path layout under
  `outputs/hpc/sft_v3_hiv_runs/<RUN_NAME>/...` so dataset build, audit, train,
  and eval can share one experiment name without hand-editing multiple paths;
- add a login-node submission helper that emits the full Slurm dependency graph
  from one command, keeping build as the root stage, launching audit and train
  after build, and launching eval after train;
- keep the paired Slurm builder wrapper in sync by ensuring the warn-log parent
  directory is created before stderr teeing;
- extend tests so the rebuild path is checked against both numeric HIV labels
  and string-valued class aliases.

### Alternatives considered
1. Re-implement the builder around the older `prepare_sft_data.py` flow.
2. Patch only the import error and leave the reporting/slurm path unchanged.
3. Introduce a new sampling objective before the current v3 path was even
   operational again.

### Consequences
- The repo once again has a working, auditable `HIV.csv -> SFT v3` builder
  compatible with the current training/eval scripts.
- Sampling summaries now make it clearer how many negative parents are
  available versus how many are actually targeted for successful selection.
- The default HPC workflow is now easier to operate because `RUN_NAME` can
  identify the full dataset/audit/train/eval artifact tree.
- The HPC workflow is now one-command submit friendly without giving up the
  stage-specific Slurm wrappers.
- The HPC `sbatch` path is less brittle because warn-log teeing no longer
  depends on the log directory already existing.

### Status
Accepted

---

## [2026-04-27] Rebuild SFT v3 from raw HIV.csv with scaffold-aware parent sampling and parent-derived reference ranking

### Background
The repository's earlier SFT data path was built from a much smaller weak-target
pool and inherited two distribution problems: too many near-parent large
fragments and a non-trivial tail of trivial tiny fragments. For the current v3
counterfactual objective, the SFT initializer should expose the model to more
strict parent substructures whose deletion leaves a non-empty residual and
whose size distribution is centered in a usable mid-range before PPO starts.

### Decision
Add a new `scripts/build_sft_v3_from_hiv.py` pipeline backed by
`src/data/sft_v3_builder.py` and `src/data/hiv_dataset_utils.py`:

- read raw `HIV.csv` directly with flexible field-name detection for SMILES and
  labels;
- normalize parent molecules with RDKit canonicalization and build scaffold +
  parent-size metadata;
- keep positive-class parents aggressively while downsampling negatives with a
  scaffold-and-size-diversity-first round-robin strategy instead of raw
  full-retention or rigid 1:1 balancing;
- generate reference candidates only from parent-derived connected fragments,
  primarily through the existing projection candidate pool
  (ring systems, functional-group neighborhoods, BRICS components, atom/bond
  k-hop fragments) plus a lightweight Murcko-like scaffold path;
- filter candidates by strict parent-substructure matching, non-empty deletion
  residual, non-full-parent status, and a default mid-size atom-ratio window of
  `[0.10, 0.55]`;
- if an oracle bundle is provided, weak-rank filtered candidates by
  `cf_flip -> cf_drop -> size closeness`; otherwise fall back to a size-aware
  heuristic ranking;
- keep the final text target as core-only fragment SMILES while preserving the
  recovered dummy-capped explanation fragment in metadata.

### Alternatives considered
1. Keep extending the old `prepare_sft_data.py` weak-target path.
2. Rebuild SFT references from arbitrary fragment-only teacher semantics rather
   than deletion-based parent-derived candidates.
3. Preserve all negatives and rely on later SFT/PPO reweighting to correct the
   data imbalance.

### Consequences
- The project now has a direct raw-HIV -> SFT-v3 rebuild path that is aligned
  with the residual-graph counterfactual objective.
- SFT train/val JSONL outputs remain compatible with `scripts/train_sft.py`,
  `scripts/eval_sft_fragment_quality.py`, and
  `scripts/analyze_sft_fragment_distribution.py`.
- HPC workflows can rebuild, train, and evaluate the new dataset through
  dedicated Slurm wrappers without changing the local VSCode -> `git push` ->
  HPC `git pull` -> `sbatch` loop.

### Status
Accepted

---

## [2026-04-26] Switch SFT and decoded PPO text targets to core-only fragments

### Background
Decoded PPO diagnostics showed repeated failure buckets around raw dummy-atom
targets: `parse_failed`, `invalid_or_not_substructure`, and
`core_fragment_unusable_after_dummy_normalization`. The project objective is
still deletion-based counterfactual subgraph generation, but requiring the LLM
to emit capped `*...*` fragments was unnecessarily expanding the text search
space and entangling text generation with RDKit attachment-point bookkeeping.

### Decision
Adopt `v3_core` / `core_no_dummy` as the default text target for the current
SFT and decoded PPO path:

- SFT dataset responses now store no-dummy `core_fragment` strings while
  preserving the original dummy-bearing fragment as metadata;
- PPO prompts and core-mode prompt builders now instruct the model to emit only
  connected core-fragment SMILES without `*`;
- RDKit remains responsible for strict parent-substructure matching,
  parent-constrained projection, boundary-bond detection, and optional recovery
  of an explanation fragment with dummy attachment markers;
- deletion-based teacher-oracle scoring continues to operate on the strict or
  projected parent subgraph, not on fragment-only teacher semantics;
- decoded PPO keeps projection and repair scaffolding, but dummy output is now
  treated as a warning plus light penalty in core mode instead of being the
  desired text format.

### Alternatives considered
1. Keep dummy-bearing targets and only patch parse/salvage heuristics.
2. Remove dummy handling from the codebase entirely.
3. Switch to a graph-only generator and bypass SMILES decoding altogether.

### Consequences
- New `data/sft_v3_core_train.jsonl` and `data/sft_v3_core_val.jsonl` datasets
  can coexist with legacy dummy-target datasets.
- Core-only eval summaries now report dummy-output and stripped-core recovery
  metrics explicitly.
- Decoded PPO candidate pools now retain both core fragments and RDKit-restored
  explanation fragments with dummy attachment points, so diagnostics remain
  available without making dummy atoms part of the model target.

### Status
Accepted

---

## Template

```md
## [YYYY-MM-DD] Decision title

### Background
Why was this decision needed?

### Decision
What was decided?

### Alternatives considered
What other options were considered?

### Consequences
What changes because of this decision?

### Status
Proposed / Accepted / Deprecated / Superseded
```

---

## [2026-04-26] v4 minimal patch for decoded PPO repair, salvage semantics, and size-window reward

### Background
The `decoded_chem_diag50_parsefix_connectfix_v3` diagnose run kept the
projection-v1 retrieval path alive and activated the tiny-fragment hard guard,
but three problems remained: minimal syntax repair still failed mostly at
`repair_candidate_parse_failed`, component salvage logs still mixed raw/core
disconnects with core-unusable cases, and the policy could still oscillate
between tiny fragments and near-parent fragments.

### Decision
Keep projection-v1, the decoded PPO main loop, and the existing Slurm argument
chain intact, and apply only a narrow v4 repair-path patch:

- upgrade minimal syntax repair from single accepted candidate to
  multi-candidate generation with staged diagnostics
  (parse/core/strict-parent/projection);
- let repair candidates prove they are either strict parent subgraphs or
  projectable through the existing parent-constrained retrieval path before
  counting as `repair_success=True`;
- restrict component salvage to true raw/core disconnected fragments, and label
  core-unusable normalization failures explicitly instead of routing them
  through `fragment_not_connected`;
- add a soft size-window reward on the final accepted fragment atom ratio while
  preserving tiny-fragment, near-parent, and tiny-residual hard fails;
- add a dedicated v4 diagnose Slurm wrapper with all arguments encoded in the
  script.

### Alternatives considered
1. Add a global nearest-valid-molecule repairer after parse failure.
2. Rewrite projection-v1 instead of keeping the existing retrieval path.
3. Rework the entire reward framework instead of patching the decoded PPO
   reward wrapper in place.

### Consequences
- Repair logs now distinguish whether failure happened at parse, core
  normalization, strict-parent validation, or repair-time projection.
- Component salvage logs now distinguish raw/core disconnected inputs from
  non-salvageable core normalization failures.
- The final accepted fragment now carries explicit size-window diagnostics
  without weakening existing hard guards.

### Status
Accepted

---

## [2026-04-26] Tighten decoded PPO parse/connect diagnostics and tiny-fragment guard

### Background
The `decoded_chem_diag50_parsefix_connectfix_v2` diagnose run showed that
minimal syntax repair was attempted but never surfaced successful repaired
candidates, disconnected fragments were not reliably entering component
salvage, and the policy was beginning to exploit very small fragments such as
`O`, `S`, and `N=O`.

### Decision
Keep projection-v1 and the existing decoded PPO Slurm parameter chain intact,
and make a narrow reward-path patch:

- expand minimal syntax repair diagnostics with candidate counts, acceptance,
  and fine-grained failure reasons;
- detect raw and core disconnected components before strict substructure checks
  and run component salvage on the disconnected representation;
- add a diagnose-configurable tiny-fragment hard fail after strict/projected
  fragment resolution, so projection success cannot bypass the minimum atom
  constraint;
- add a v3 50-step Slurm wrapper that enables the new guard with
  `MIN_FRAGMENT_ATOMS=3` and `TINY_FRAGMENT_HARD_FAIL_PENALTY=-6.0`.

### Alternatives considered
1. Rewrite parse-failed outputs into nearest valid molecules.
2. Move tiny-fragment suppression into the prompt only.
3. Rebuild projection-v1 instead of preserving the existing retrieval path.

### Consequences
- Repaired parseable candidates now continue into the same strict parent
  subgraph / retrieval projection path rather than being treated as a separate
  unconstrained repair objective.
- Diagnose logs distinguish raw/core component counts and salvage failure
  stages.
- Very small strict or projected fragments receive a fixed hard-fail reward in
  the new diagnose script, preventing positive reward terms from masking tiny
  fragment collapse.

### Status
Accepted

---

## [2026-04-25] Add parent-constrained retrieval projection for decoded PPO non-substructure fragments

### Background
Decoded-chem PPO now surfaces parse failures separately from parseable fragments
that are not valid parent substructures. The latter failure bucket is actionable:
the raw fragment already has a chemically parseable shape, but the exact graph
does not occur in the parent molecule. For the current instance-level candidate
generator, these cases should be repaired by projecting onto strict
parent-derived candidates before reward and oracle scoring.

### Decision
Add an optional parent-constrained candidate retrieval projection path for
decoded PPO:

- keep parse-failed raw fragments on the existing failure path;
- mark parseable strict parent substructures as `projection_method=identity`;
- when a parseable connected core is not a parent substructure, build a
  parent-derived candidate pool from ring systems, SMARTS functional-group
  neighborhoods, atom-centered k-hop neighborhoods, bond-centered k-hop
  neighborhoods, and stable parent-index BRICS components;
- filter candidates to connected, RDKit-parseable, non-full-parent subgraphs
  whose deletion leaves a non-empty residual;
- score candidates with Morgan Tanimoto, MCS atom coverage, functional-group
  overlap, atom-count difference, and a large-fragment penalty;
- if the best score passes the configured threshold, continue reward and
  deletion-oracle scoring on the projected fragment and subtract a projection
  penalty from `reward_total`;
- expose projection controls through `scripts/train_ppo.py`, decoded PPO logs,
  and Slurm wrappers.

### Alternatives considered
1. Reuse the existing parent-aware repair path without adding k-hop candidates,
   strict deletion filtering, or projection-specific logs.
2. Penalize all parseable non-substructure outputs without attempting a
   parent-constrained projection.
3. Rewrite raw fragments directly with string heuristics instead of retrieving
   from parent atom-index subgraphs.

### Consequences
- The dominant `parse_ok_but_not_substructure` failure bucket can now produce
  rewardable parent-derived fragments without changing parse-failed behavior.
- Logs now record projection attempt/success, retrieval score/source, projected
  fragment, atom statistics, candidate count, and applied penalty.
- HPC diagnosis can run fixed 50-step and 200-step projection jobs through
  dedicated Slurm scripts without hand-written `sbatch --export` arguments.

### Status
Accepted

---

## [2026-04-25] Wire parent-aware repair controls through decoded PPO CLI and Slurm wrappers

### Background
The decoded PPO rewarder already supported one optional parent-aware repair
attempt for broken decoded fragments, but the training entrypoint and the HPC
Slurm wrappers did not expose those controls. As a result, users could set
environment variables such as `ENABLE_PARENT_AWARE_REPAIR=true` in `sbatch`
commands without the settings ever reaching `ChemRLRewarder`.

### Decision
Expose the existing repair controls end-to-end without changing the reward
objective:

- add `--enable-parent-aware-repair`,
  `--repair-min-similarity`, and `--repair-max-candidates` to
  `scripts/train_ppo.py`;
- pass the parsed values into `ChemRLRewarder`;
- log the resolved repair configuration at startup;
- forward the corresponding environment variables through
  `scripts/slurm/train_decoded_chem_ppo_full.sh` and
  `scripts/slurm/train_ppo.sh`.

### Alternatives considered
1. Leave repair available only as a code-level option in `reward_wrapper`.
2. Ask users to edit the Slurm shell scripts manually for each experiment.
3. Rework repair behavior inside the rewarder without exposing it through the
   CLI.

### Consequences
- `sbatch --export=ALL,ENABLE_PARENT_AWARE_REPAIR=...` style launches now
  actually affect decoded PPO runs.
- HPC diagnose jobs can sweep repair settings in the same way they already
  sweep decoded generation settings.
- The reward behavior itself is unchanged unless the user explicitly enables
  repair.

### Status
Accepted

---

## [2026-04-25] Preserve raw dummy-atom evidence in decoded PPO parse-failure logs

### Background
The decoded-chem PPO reward path already used dummy-aware normalization for
successful capped fragments such as `*CC(=O)O`, but parse-failed fragments were
still difficult to diagnose from logs alone. In particular, once raw parsing
failed, the existing trace fields could lose the evidence that the original
fragment string actually contained `*`, which made it hard to tell whether
failures mostly came from uncapped raw fragments or from the dummy-atom path
itself.

### Decision
Keep the decoded PPO chemistry objective unchanged and apply a minimal logging /
normalization refinement only in `reward_wrapper` and `train_ppo`:

- preserve the raw fragment string as the source of truth for dummy presence and
  dummy count before any RDKit parsing happens;
- continue to parse the raw fragment with `*` intact and never do string-level
  `replace("*", "")` before `MolFromSmiles`;
- keep `core_fragment` as a derived post-parse view used for teacher scoring and
  deletion checks only;
- surface explicit parse metadata in decoded PPO logs, including
  `raw_has_dummy`, `raw_dummy_count`, `parse_stage`,
  `parsed_raw_with_dummy`, `parsed_core`, `dummy_removed_before_parse`, and
  `parse_failed_reason`;
- split parse-failure buckets into
  `parse_failed_raw_with_dummy`, `parse_failed_raw_without_dummy`,
  `parse_failed_after_dummy_removal`, plus the existing obvious closure buckets
  such as `parse_failed_unclosed_ring` and
  `parse_failed_unbalanced_parentheses`;
- add per-batch counters for parse failures with and without raw dummy atoms.

### Alternatives considered
1. Keep the existing reward path unchanged and infer dummy-related failures only
   from ad hoc grep patterns.
2. Strip dummy atoms from strings before parsing so that all failures collapse
   onto core-fragment syntax.
3. Attempt automatic repair of ring digits or parentheses during reward-time
   normalization.

### Consequences
- Diagnose logs can now answer whether parse failures mostly come from raw
  fragments without `*` or from dummy-aware normalization paths.
- Successful capped fragments still use the same raw-then-core workflow, so the
  counterfactual objective and deletion logic do not change.
- The codebase now makes it explicit that dummy removal happens after raw
  parsing, not before it.

### Status
Accepted

---

## [2026-04-25] Harden decoded-chem PPO against overlong invalid fragments and surface failure buckets explicitly

### Background
After the second SFT round, decoded-chem PPO could initialize from
`checkpoint-300`, keep policy/reference aligned, and run a full 200-step
diagnose loop. The dominant failure mode, however, was no longer empty
responses. Instead, many generations failed as invalid or non-substructure
fragments, often because the decoded fragment was too long, truncated, or left
rings / brackets / parentheses unclosed. At the same time, full-parent and
empty-residual failures were still present, but their penalties were not strong
enough to clearly dominate those degenerate behaviors.

### Decision
Keep candidate-pool, selector, and SFT logic unchanged, and apply a minimal
decoded-chem PPO hardening pass only on generation / reward constraints:

- add decoded-generation-specific CLI knobs
  (`--gen-max-new-tokens`, `--gen-temperature`, `--gen-top-p`,
  `--gen-do-sample`) while keeping the legacy PPO generation flags usable;
- when decoded-chem PPO is launched without an explicit generation-length
  override, tighten its default `max_new_tokens` to `48`;
- preprocess decoded fragments before chemistry reward by stripping whitespace,
  keeping only the first line, and rejecting overlong fragments as
  `invalid_generation_too_long`;
- keep parse failures on the normal invalid path, but add an explicit
  `invalid_detail` field for obvious closure issues such as unbalanced
  parentheses, brackets, or ring digits;
- increase the default full-parent / empty-residual penalties to `-6.0` and
  `-4.0`, respectively;
- log `failure_tag`, `invalid_detail`, and generated fragment length alongside
  the existing `CHEM_REWARD_COMPONENTS` fields so bad cases can be grepped
  directly from decoded PPO diagnose runs;
- forward the new decoded-generation controls through the main HPC Slurm
  wrappers using `sbatch --export=ALL,...`.

### Alternatives considered
1. Leave generation settings untouched and only adjust the reward penalties.
2. Solve the issue in the candidate pool or selector instead of in decoded PPO.
3. Add aggressive chemistry-aware truncation that rewrites generated fragments
   rather than rejecting obviously bad strings early.

### Consequences
- Decoded PPO now pushes back earlier on the observed overlong / truncated
  invalid fragments without changing the project objective.
- Logs can distinguish `invalid_generation_too_long`,
  `invalid_or_not_substructure`, `full_parent_fragment`, and `empty_response`
  directly.
- HPC diagnose runs can sweep decoded generation settings through Slurm exports
  instead of editing shell scripts by hand.

### Status
Accepted

---

## [2026-04-25] Let decoded-chem PPO initialize both policy and reference from one explicit SFT LoRA checkpoint

### Background
The decoded chemistry PPO path is meant to start from an SFT policy rather than
from the raw base model. That becomes especially important once we want to run
PPO from a chosen SFT v2 checkpoint such as `checkpoint-300`. If the trainable
policy starts from that checkpoint but the KL reference remains the bare base
model, the KL term is misaligned and can exaggerate drift. At the same time,
collapsed generations such as empty responses, full-parent fragments, and
empty-residual deletions need to be surfaced explicitly in logs rather than
appearing as opaque chemistry failures.

### Decision
Keep the decoded-chem PPO objective unchanged, but make initialization and
anti-collapse diagnostics explicit:

- add `--sft-lora-path` as the preferred CLI name for the SFT initialization
  checkpoint while keeping the old `--sft-adapter-path` path for backward
  compatibility;
- resolve one effective SFT LoRA path and use it for both the trainable PPO
  policy and the frozen KL reference model;
- log the resolved policy/reference initialization path so HPC runs can verify
  that both models start from the same checkpoint;
- treat empty decoded responses as empty fragments instead of letting prompt
  echo accidentally fall back to the parent molecule;
- add explicit `full_parent` and `empty_residual` handling with configurable
  penalties and stable log fields (`empty_response`, `full_parent`,
  `empty_residual`, `oracle_ok`, `cf_drop`, `cf_flip`, `reward_total`).

### Alternatives considered
1. Keep using only `--sft-adapter-path` and rely on users to infer whether the
   reference model matches the policy.
2. Let the KL reference remain the raw base model even when PPO starts from an
   SFT adapter.
3. Leave empty-response and full-parent cases to be inferred indirectly from
   teacher-oracle failure reasons.

### Consequences
- PPO runs can now be launched from an explicit SFT v2 checkpoint such as
  `checkpoint-300` with policy/reference alignment preserved.
- Decoded-chem logs are easier to grep for collapse-related cases without
  changing the underlying deletion-based counterfactual objective.
- The full-training Slurm wrapper can forward the chosen SFT LoRA checkpoint
  and the new explicit penalties through `sbatch --export=ALL,...`.

### Status
Accepted

---

## [2026-04-25] Make SFT fragment-distribution audits chunkable and existence-first for HPC runs

### Background
The new SFT audit scripts are intended to characterize whether weak labels and
SFT generations collapse toward near-parent fragments or tiny trivial pieces.
On larger files, however, a few symmetric molecules caused audit runs to stall
for a long time inside RDKit substructure enumeration, which made whole-file
audits unreliable on both local machines and HPC nodes.

### Decision
Keep the audit objective unchanged, but make the audit path operationally safe:

- add chunk/window controls and progress logging to
  `scripts/analyze_sft_fragment_distribution.py`;
- isolate per-sample audit exceptions so one bad molecule does not abort the
  entire batch unless `--fail-fast` is explicitly requested;
- prefer existence-first substructure checks (`HasSubstructMatch` or capped
  queries limited to `maxMatches=1`) when the audit only needs to know whether
  a match exists;
- add cheap pruning before expensive chemistry work, including parse failures,
  fragment-larger-than-parent checks, and a full-parent shortcut based on
  canonical core equality;
- emit slow-sample records so long-running parent/fragment pairs can be
  inspected and re-run in isolated chunks on HPC.

### Alternatives considered
1. Keep the original all-in-one audit and rely on manual interruption when a
   job stalls.
2. Disable substructure and deletion checks globally, even for manageable
   molecules.
3. Rewrite the chemistry layer around a separate matching backend.

### Consequences
- SFT audits can now be submitted in bounded chunks through Slurm.
- Large runs produce explicit progress, warning, and slow-sample artifacts
  instead of appearing silently hung.
- The chemistry layer still enforces the same counterfactual-fragment
  definition, but audit-time existence checks avoid enumerating every symmetric
  match when only a yes/no answer is needed.

### Status
Accepted

---

## [2026-04-19] Add explicit teacher-semantic scoring on core fragments in the decoded chemistry PPO path

### Background
The decoded chemistry PPO loop now proves that generated text is decoded,
normalized, scored by chemistry utilities, and then used in a PPO update.
However, the logs still showed:

- `[CHEM_REWARD_COMPONENTS_MISSING] missing=teacher_sem`

That made it impossible to tell whether any auxiliary fragment-level semantic
signal was actually being applied after dummy-atom normalization.

### Decision
Add a dedicated `TeacherSemanticScorer` and wire it into the decoded chemistry
reward path only.

Key rules:

- the teacher always receives `core_fragment_smiles`, never the raw capped
  fragment with `*`;
- invalid or non-substructure fragments do not call the teacher and instead log
  a skip reason;
- when the teacher backend is unavailable, the code uses an explicit fallback
  penalty and logs the unavailability rather than pretending a real score
  exists;
- the repository's existing residual-molecule counterfactual term remains in
  place, so the teacher-semantic term is an auxiliary signal rather than a
  replacement for the deletion-based objective.

Because the repository currently ships one concrete classifier artifact at
`outputs/hpc/oracle/aids_rf_model.pkl`, the scorer first supports that
scikit-learn style bundle format (`predict_proba` plus fingerprint metadata).
Torch checkpoints are only accepted when they carry equally explicit
fingerprint configuration.

### Alternatives considered
1. Continue logging `teacher_sem` as missing.
2. Treat the residual-molecule oracle as if it already satisfied the teacher
   role and hide the distinction.
3. Require a new `teacher/teacher.pt` artifact before any teacher-semantic work
   could proceed.

### Consequences
- decoded PPO logs now expose `[TEACHER_SEM_CALLED]`,
  `[TEACHER_SEM_RESULT]`, `[TEACHER_SEM_SKIPPED]`, and
  `[TEACHER_SEM_UNAVAILABLE]`;
- `CHEM_REWARD_COMPONENTS` now shows both `teacher_sem` and the residual
  counterfactual term separately;
- the decoded chemistry smoke-test Slurm script now checks for a teacher file
  before submitting training.

### Status
Accepted

---

## [2026-04-19] Treat dummy-atom attachment points as valid decoded-fragment syntax in PPO chemistry rewards

### Background
The decoded chemistry PPO path now makes reward computation explicit, but the
rewarder was still too harsh on fragment strings containing `*`, for example
`*CC(=O)O`. In this project those stars are not arbitrary garbage characters;
they encode attachment points created by fragment cutting. If the rewarder
treated them as invalid text, PPO would incorrectly learn that many chemically
meaningful capped fragments were malformed.

### Decision
Keep two fragment views inside `src/rewards/reward_wrapper.py`:

- `raw_fragment_smiles`: the exact decoded fragment candidate, which may contain
  dummy atoms such as `*`;
- `core_fragment_smiles`: a dummy-free core used for substructure checks,
  compactness statistics, and any future fragment-level teacher signal.

The rewarder now:

- parses capped fragments with dummy-aware RDKit sanitization;
- removes dummy atoms through molecule editing instead of string replacement;
- checks parent substructure on the core fragment rather than on the raw capped
  string;
- counts fragment size on non-dummy atoms only;
- exposes raw/core parse flags and dummy counts in reward traces and decoded PPO
  logs.

### Alternatives considered
1. Keep treating all `*` tokens as invalid output.
2. Strip `*` with naive string replacement before every reward computation.
3. Move the dummy-aware normalization into TRL adapters instead of the chemistry
   rewarder.

### Consequences
- Decoded PPO logs can now distinguish raw capped fragments from their
  dummy-free core.
- Validity and substructure rewards no longer collapse to zero solely because a
  fragment uses attachment-point notation.
- The chemistry reward path remains honest about what is still missing, such as
  any dedicated teacher-semantic term.

### Status
Accepted

---

## [2026-04-18] Add a decoded-SMILES chemistry reward PPO loop alongside the TRL compatibility baseline

### Background
The repository's TRL experimental PPO smoke test now runs end to end, but the
successful path still relies on a hidden-state reward adapter that only
validates trainer interface compatibility. The logs already make this
limitation explicit:

- `ChemRewardModelWrapper remains the chemistry reward component and is not equivalent to TRL hidden-state reward head`

That means the baseline does not prove that decoded fragment strings are being
scored by the chemistry reward and then used for PPO updates.

### Decision
Keep the TRL experimental path as the engineering baseline, but add a second
training mode in `scripts/train_ppo.py`:

- `--ppo-loop decoded_chem`

This mode performs the reward flow explicitly:

- prompt batch
- `policy_model.generate()`
- decode generated response text
- extract one fragment candidate
- call `ChemRLRewarder.compute_rewards_from_decoded(...)`
- run a local PPO-style update with policy logprobs, reference logprobs,
  token-aligned value predictions, KL-shaped rewards, clipped policy loss, and
  clipped value loss

The CLI also now supports:

- `--require-chemistry-reward-path`
- `--decoded-chem-smoke-test`

and the repository includes a dedicated Slurm smoke-test script:

- `scripts/slurm/debug_decoded_chem_ppo_smoketest.sh`

### Alternatives considered
1. Keep using the TRL hidden-state reward adapter and treat it as good enough.
2. Patch TRL site-packages until they can consume the chemistry wrapper
   directly.
3. Block all further work until a legacy `PPOTrainer.step(...)` API is proven
   available in every environment.

### Consequences
- The repository now has one baseline path for TRL interface compatibility and
  one separate path that makes decoded-SMILES chemistry rewards enter PPO
  updates explicitly.
- Smoke-test logs can now distinguish “trainer compatibility succeeded” from
  “decoded chemistry reward was called and used in an update”.
- The local PPO loop stays inside repository code, so no external environment
  patching is required.

### Status
Accepted

---

## [2026-04-18] Skip trainer-managed completion previews when experimental PPO has no usable eval dataloader

### Background
After the ChemLLM cache fix, the value-model `.score` adapter, and the
reward-model compatibility adapter were all in place, the PPO smoke test
finally entered the trainer loop and emitted PPO metrics. The next failure came
from `trl.experimental.ppo.PPOTrainer.generate_completions()`, which tried to
iterate `self.eval_dataloader` even though the smoke-test path had no usable
evaluation data source. That eventually surfaced as:

- `TypeError: object of type 'NoneType' has no len()`

### Decision
Keep the main PPO training loop intact, but add a local repository-side guard
in `scripts/train_ppo.py` for the trainer-managed completion-preview stage:

- add `--skip-generate-completions` as an explicit CLI escape hatch;
- add `--diagnose-reward-flow` as a smoke-test-friendly debug flag;
- detect unusable completion-preview loaders such as missing
  `eval_dataloader`, missing `dataset`, or `sampler.data_source is None`;
- replace `ppo_trainer.generate_completions()` with a no-op logger only when
  the skip flag is enabled or the evaluation loader is clearly unusable.

The HPC smoke-test script now always passes both diagnostic flags so it can
focus on initialization and the core PPO loop without failing in the preview
generation branch.

### Alternatives considered
1. Patch `trl.experimental` directly in site-packages to special-case missing
   eval loaders.
2. Fabricate a fake evaluation dataset just to satisfy completion previews.
3. Catch and suppress broad exceptions around `ppo_trainer.train()`.

### Consequences
- The smoke test can keep exercising PPO initialization and training steps even
  when the trainer's optional preview-generation path has no evaluation data.
- Main training errors are still surfaced normally because only
  `generate_completions()` is replaced, not the full trainer loop.
- Slurm logs now include explicit `[PPO_GENERATE_COMPLETIONS_SKIPPED]` markers
  so it is obvious when this guard was applied.

### Status
Accepted

---

## [2026-04-18] Separate chemistry reward logic from TRL experimental reward-model interface compatibility

### Background
After the local `.score` adapter fixed the PPO critic-side value-model crash,
the smoke test progressed further and then failed inside
`trl.experimental.utils.get_reward()` with:

- `AttributeError: 'ChemRewardModelWrapper' object has no attribute 'base_model_prefix'`

The repository's `ChemRewardModelWrapper` computes chemistry-aware rewards by
decoding generated text back into parent / fragment pairs and calling
`ChemRLRewarder`. That is not the same interface as the Hugging Face-style
reward model expected by this experimental TRL path, which assumes:

- `base_model_prefix`
- a forwardable LM backbone accessible through that prefix
- `score(hidden_states)`

### Decision
Keep `ChemRewardModelWrapper` as the repository's chemistry reward component,
but stop passing it directly to experimental PPO when the runtime expects a
hidden-state reward model.

Instead, `scripts/train_ppo.py` now adds
`ensure_reward_model_for_experimental_ppo(...)`, which:

- reuses an existing reward-side backbone if one is already exposed;
- otherwise builds a lightweight TRL-compatible reward adapter around a
  fallback LM backbone such as `value_model.pretrained_model`;
- adds `base_model_prefix` and a token-level `score` head for interface
  compatibility;
- logs explicitly that the fallback adapter is only for smoke-test /
  interface-validation purposes and is not equivalent to the repository's
  deletion-based chemistry reward objective.

### Alternatives considered
1. Patch TRL directly in site-packages so it can accept the chemistry wrapper.
2. Pretend the chemistry wrapper is a native hidden-state reward model by
   adding only one missing attribute at a time.
3. Remove the chemistry reward component entirely and silently replace it with
   a generic reward head.

### Consequences
- The smoke test can progress past TRL's stricter `reward_model` interface
  checks without mutating the external conda environment.
- Repository code now makes the mismatch between chemistry rewards and TRL's
  hidden-state reward-model contract explicit instead of hiding it behind
  brittle monkey patches.
- Future work can reconnect true chemistry rewards to trainer-managed PPO more
  cleanly because the current limitation is now documented in code and docs.

### Status
Accepted

---

## [2026-04-18] Attach a local `.score` adapter for TRL value-head critics in experimental PPO

### Background
The ChemLLM PPO smoke test moved past the earlier InternLM2 cache failure, but
then failed inside `trl.experimental.utils.get_reward()` when the trainer tried
to evaluate the critic with:

- `model.score(output.hidden_states[-1])`

Our repository-side `value_model` was still
`AutoModelForCausalLMWithValueHead`, which exposes `v_head` rather than a
top-level `.score` method. Patching TRL in site-packages was explicitly out of
scope for the VS Code plus Git plus HPC workflow.

### Decision
Add a repository-local compatibility helper in `scripts/train_ppo.py`:

- `ensure_score_head_for_experimental_ppo(model, name=...)`

The helper now:

- leaves models that already expose `.score` unchanged;
- searches the top-level object and common wrapper layers such as
  `pretrained_model`, `base_model`, and `model` for a reachable `v_head`;
- attaches `model.score(hidden_states)` dynamically when only `v_head` exists;
- logs which wrapper layer supplied the adapted value head.

The adapter is applied to `value_model` before `PPOTrainer` construction. The
policy and reference models remain untouched so generation behavior stays
isolated from this critic-side interface patch.

### Alternatives considered
1. Patch `trl.experimental` directly inside the conda environment.
2. Replace the value wrapper again with another custom critic type.
3. Fork the PPO rollout path away from the trainer-managed `get_reward()`
   utility.

### Consequences
- Experimental PPO can keep using the existing TRL value-head wrapper while
  satisfying the newer `.score(hidden_states)` critic contract.
- The compatibility layer stays local to repository code and is therefore easy
  to review, sync to HPC, and remove later if upstream APIs converge.
- Smoke-test logs now make it explicit whether the value model has both
  `v_head` and the adapted `.score` interface before training begins.

### Status
Accepted

---

## [2026-04-09] Rebuild repository from scratch with documentation-first workflow

### Background
The previous project evolved through multiple experimental iterations. It likely accumulated script coupling, outdated assumptions, and mixed objectives inherited from earlier versions. To prevent repeated confusion during reconstruction, the repository needs a stable written definition before code implementation begins.

### Decision
Rebuild the repository from an empty root using a documentation-first workflow. The initial authoritative files are:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

These files define the objective, engineering rules, roadmap, and future design log.

### Alternatives considered
1. Start coding immediately and add documentation later.
2. Recover the old repository structure first and refactor afterward.
3. Keep all design notes only in chat history.

### Consequences
- The repository gets a clear source of truth from the start.
- Codex can be guided by repository-local instructions instead of relying on conversational memory.
- Early progress is slower in appearance but more stable in direction.

### Status
Accepted

---

## [2026-04-16] Make ChemLLM cache patch tool classify guarded vs unguarded accesses

### Background
The first repository-side ChemLLM cache patch tool could add the helper and
patch the forward-path cache-length logic, but its multiline matching for
`prepare_inputs_for_generation()` was too brittle. As soon as the target block
contained nested indentation, comments, or slightly different formatting, the
prepare-path patch silently failed. The checker output also counted every
`past_key_values[0][0].shape[2]` occurrence as equally dangerous even after it
had been moved under the new guard.

### Decision
Rework `tools/check_or_patch_chemllm_cache.py` to use indentation-aware,
line-based patching and reporting:

- patch the forward cache-length block and the
  `prepare_inputs_for_generation()` block independently;
- insert `else: past_key_values = None` / `past_length = 0` for the prepare
  path;
- detect already-patched blocks and skip them cleanly;
- classify dangerous accesses as either `guarded` or `unguarded`, and treat
  `unguarded_count=0` as the real success criterion.

### Alternatives considered
1. Keep the old regexes and only tweak them slightly.
2. Require manual patching for the prepare path on HPC.
3. Keep counting all `shape[2]` accesses as equally dangerous.

### Consequences
- The repository-side helper can now patch both critical cache branches before
  HPC smoke testing.
- Patch results are easier to interpret because protected accesses are no
  longer reported as unresolved failures.
- HPC documentation can now point users to `unguarded_count=0` instead of
  incorrectly suggesting that all `shape[2]` accesses must disappear.

### Status
Accepted

---

## [2026-04-16] Add PPO runtime import-path introspection for ChemLLM cache debugging

### Background
The ChemLLM / InternLM2 PPO path hit a cache-related generation crash inside
`modeling_internlm2.py`, but local repository edits alone were not enough to
prove which dynamically cached file the Slurm job was actually importing at
runtime. In a VS Code plus Git plus HPC workflow, the repository copy, the
Hugging Face dynamic module cache, and the job's working directory can diverge.

### Decision
Add lightweight runtime introspection to `scripts/train_ppo.py` so every PPO
run logs:

- the wrapped policy / reference / value model module names;
- the resolved module source files;
- the resolved `prepare_inputs_for_generation` source files;
- key environment variables such as `PYTHONPATH`, `HF_HOME`,
  `TRANSFORMERS_CACHE`, and `HUGGINGFACE_HUB_CACHE`.

Also add a dedicated Slurm smoke-test entrypoint:

- `scripts/slurm/debug_check_chemllm_runtime_path.sh`

that reuses the normal HPC environment bootstrap, prints repository and Python
runtime information, and runs a tiny PPO smoke test through
`scripts/train_rl.py`.

### Alternatives considered
1. Keep reasoning about cache behavior from static local files only.
2. Patch more InternLM2 code blindly without first proving the runtime import
   path.
3. Ask users to manually add debug prints on the HPC side.

### Consequences
- Future Slurm logs can show exactly which `modeling_internlm2.py` was imported
  during PPO generation.
- Runtime path mismatches between repository code and Hugging Face dynamic cache
  are easier to detect before chasing deeper trainer bugs.
- The repository now has a reusable HPC-first smoke test for ChemLLM runtime
  path debugging.

### Status
Accepted

---

## [2026-04-15] Implement first PPO training path with residual-based reward wrapper

### Background
The repository already had a trained SFT adapter, a lightweight AIDS oracle,
and a chemistry layer for capped fragment validation and deletion. What was
still missing was a real PPO training path that could optimize ChemLLM outputs
toward the deletion-based counterfactual objective on HPC hardware.

### Decision
Add a concrete PPO training entrypoint in `scripts/train_ppo.py` together with a
unified reward wrapper in `src/rewards/reward_wrapper.py`.

The reward wrapper uses a three-stage early-stop flow:

- parseability and connectedness checks;
- parent-subgraph validation for capped fragments;
- residual-molecule scoring after deleting the fragment from the parent.

The semantic reward term is computed on the residual graph rather than on the
fragment alone, because the v3 objective is label flipping after deletion.

The PPO script loads:

- ChemLLM-7B-Chat in 4-bit mode;
- the SFT LoRA checkpoint as the initial policy;
- a frozen LoRA-backed reference model for KL control;
- prompts from either the raw HIV CSV or a JSONL prompt file.

The training loop logs reward statistics, structural pass rates, simple
collapse diagnostics, and representative generations into the run directory.

### Alternatives considered
1. Keep RL as a runtime-preparation placeholder only.
2. Score the fragment alone with the oracle instead of the residual molecule.
3. Hardcode one dataset schema and refuse either CSV or JSONL prompts.

### Consequences
- The repository now has a runnable PPO-stage backbone aligned with the
  counterfactual v3 objective.
- KL control remains anchored to the SFT adapter rather than drifting from the
  base model directly.
- PPO runs surface chemistry and collapse signals explicitly instead of hiding
  them inside trainer internals.

### Status
Accepted

---

## [2026-04-15] Switch PPO policy loading back to native causal LM for experimental TRL

### Background
After the first PPO entrypoint landed, newer `trl.experimental.ppo` builds
showed constructor-time incompatibilities with explicit value-head wrappers,
including failures around missing wrapper attributes such as
`base_model_prefix`. This indicated that the trainer now expects native causal
LM policies and prefers to manage policy/value wrapping internally.

### Decision
Update `scripts/train_ppo.py` so that PPO loads native
`AutoModelForCausalLM`-style PEFT models for both the trainable policy and the
frozen reference policy, without constructing
`AutoModelForCausalLMWithValueHead`.

The PPO trainer initialization is now version-adaptive in three ways:

- `PPOConfig` kwargs are filtered against the runtime signature;
- trainer kwargs map across `args/config`, `model/policy`, and
  `ref_model/ref_policy` variants;
- external `value_model` is omitted or forced to `None` so TRL can own value
  wrapping internally.

The script also provides a lightweight reward adapter that exposes
`ChemRLRewarder` as either `reward_model` or `reward_funcs` when those newer
experimental hooks are present.

### Alternatives considered
1. Keep the explicit value-head wrapper and patch missing attributes one by one.
2. Pin the repository to one older TRL version instead of adapting the code.
3. Move the whole PPO path back to a handwritten optimizer loop.

### Consequences
- The PPO script is better aligned with current experimental TRL architecture.
- Fewer wrapper-specific attribute mismatches should appear when the trainer is
  upgraded.
- The script still keeps the repository's deletion-based counterfactual reward
  logic outside trainer internals.

### Status
Accepted

---

## [2026-04-15] Use explicit value and reward models for experimental PPOTrainer

### Background
Follow-up PPO integration work revealed another compatibility shift in newer
`trl.experimental.ppo.PPOTrainer` builds: the trainer now expects a real
transformers-style `value_model` object and also routes internal scoring
through a PyTorch `reward_model(input_ids=..., attention_mask=..., ...)`
interface.

### Decision
Keep the policy and reference networks as native causal LMs, but explicitly add:

- a 4-bit `AutoModelForSequenceClassification` value model with `num_labels=1`;
- a torch `reward_model` wrapper that decodes generated sequences back to text,
  reconstructs the parent / fragment pair, and calls `ChemRLRewarder`.

The trainer initialization in `scripts/train_ppo.py` now wires:

- `model`: native causal LM policy with the SFT LoRA adapter;
- `ref_model`: frozen native causal LM reference policy;
- `value_model`: explicit scalar sequence-classification model;
- `reward_model`: deletion-based chemistry reward wrapper.

### Alternatives considered
1. Continue passing `None` for the value model and rely on implicit trainer behavior.
2. Keep reward scoring entirely outside the trainer and ignore new internal hooks.
3. Reintroduce custom non-transformers wrapper classes around the value head.

### Consequences
- PPO initialization is better aligned with stricter experimental TRL releases.
- Internal trainer scoring can now call a PyTorch-compatible reward interface
  without losing the repository's residual-graph counterfactual objective.
- The value network contract is explicit instead of version-dependent.

### Status
Accepted

---

## [2026-04-15] Fall back to TRL value-head wrapper for InternLM2 PPO value model

### Background
The explicit sequence-classification critic introduced for experimental PPO
alignment turned out to be incompatible with the ChemLLM / InternLM2 base
checkpoint in environments where the InternLM2 config is not registered under
Hugging Face's `AutoModelForSequenceClassification` auto-mapping.

### Decision
Replace the PPO `value_model` in `scripts/train_ppo.py` with
`trl.AutoModelForCausalLMWithValueHead` loaded on top of the same quantized
ChemLLM base weights, and monkey-patch:

- `value_model.base_model_prefix = "pretrained_model"`

so that stricter `trl.experimental.ppo.PPOTrainer` builds can still treat the
wrapper like a native model during critic setup.

Only the wrapper's value head remains trainable; the wrapped causal LM backbone
stays frozen.

### Alternatives considered
1. Keep using `AutoModelForSequenceClassification` and require a newer
   InternLM2 registration patch from transformers.
2. Drop experimental PPOTrainer and fully hand-roll critic/value optimization.
3. Try to emulate a sequence-classification head with another custom wrapper.

### Consequences
- PPO critic initialization no longer depends on InternLM2 being registered in
  the sequence-classification auto-model registry.
- The project keeps the reward path aligned to the deletion-based
  counterfactual objective while staying closer to TRL's expected interfaces.
- The monkey patch is intentionally local to the PPO script and should be
  revisited if upstream TRL or transformers support improves.

### Status
Accepted

---

## [2026-04-15] Let experimental PPOTrainer own rollout and optimization

### Background
Once the experimental PPO trainer, explicit value model, and reward model were
all wired successfully, the remaining failures came from the old handwritten
training loop that still tried to call trainer-side generation and manual PPO
updates directly.

### Decision
Remove the legacy step-by-step PPO loop from `scripts/train_ppo.py` and switch
the entrypoint to the trainer-managed flow:

- initialize policy, reference, value, and reward models;
- construct `PPOTrainer`;
- call `ppo_trainer.train()`;
- save the final checkpoint via the trainer's own save path.

The PPO config assembly now also forwards `max_steps` and generation-related
kwargs when the runtime `PPOConfig` signature supports them.

### Alternatives considered
1. Keep maintaining a hybrid script that mixes experimental trainer internals
   with a manual rollout loop.
2. Revert fully to a classic step-based TRL API.
3. Move rollout back outside the trainer and bypass `reward_model`.

### Consequences
- The PPO entrypoint now matches the ownership model of newer
  `trl.experimental.ppo` releases more closely.
- Fewer incompatibilities should appear around missing `generate()` or `step()`
  methods on experimental trainer objects.
- Reward evaluation stays aligned with the repository's residual-graph
  counterfactual objective, but execution control is delegated to TRL.

### Status
Accepted

---

## [2026-04-15] Run PPO WandB logging in offline mode for HPC nodes

### Background
The PPO training entrypoint is designed for HPC execution, but compute nodes in
the target environment do not have outbound internet access. Direct online
Weights & Biases logging would therefore stall or fail during trainer startup.

### Decision
Configure `scripts/train_ppo.py` to force WandB offline mode via environment
variables:

- `WANDB_MODE=offline`
- `WANDB_SILENT=true`

At the same time, keep the trainer-side reporting target aligned to WandB when
supported by the runtime PPO config, so metrics are still written locally into
the standard `wandb/` directory for later sync.

The PPO config builder now forwards:

- `report_to="wandb"` when supported;
- `log_with="wandb"` for older compatible signatures;
- `run_name="ppo_aids_rl_v1"` as the semantic experiment label.

### Alternatives considered
1. Disable WandB entirely on HPC and rely only on stdout or custom JSON logs.
2. Require a separate shell wrapper to export WandB offline variables.
3. Keep online WandB enabled and tolerate repeated timeout failures.

### Consequences
- PPO runs on air-gapped or internal-only HPC nodes no longer depend on live
  WandB connectivity.
- Local WandB artifacts remain available for later `wandb sync`.
- Experiment naming is more stable across local logs and offline WandB runs.

### Status
Accepted

---

## [2026-04-16] Force PPO datasets to emit tensors before entering experimental trainer

### Background
After the PPO entrypoint successfully crossed model and trainer initialization,
the first runtime failure inside `trl.experimental.ppo.PPOTrainer.train()`
occurred when the trainer tried to move `data["input_ids"]` onto the device.
The immediate cause was that the Hugging Face dataset / collator path was still
yielding Python lists instead of PyTorch tensors.

### Decision
Update `scripts/train_ppo.py` so that the PPO training dataset explicitly calls:

- `dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)`

after tokenization, and replace the custom collator with a tensor-aware version
that:

- pads `input_ids` through `tokenizer.pad(..., return_tensors="pt")`;
- emits `input_ids` and `attention_mask` as tensors;
- preserves text metadata fields such as `query` and `parent_smiles` as Python
  lists for downstream reward reconstruction.

The reward wrapper's decoding path also now detaches tensor inputs to CPU
before `batch_decode`, while keeping the returned reward tensor on the same
device as `input_ids`.

### Alternatives considered
1. Rely only on `set_format("torch")` and keep the old collator unchanged.
2. Drop the custom collator entirely and hope the trainer default handles mixed
   text-and-tensor batches correctly.
3. Convert data inside the trainer loop instead of fixing the dataset contract.

### Consequences
- Experimental PPOTrainer can now consume `input_ids` with a valid `.to(device)`
  path.
- The batch contract is more explicit and robust against future TRL internal
  assumptions.
- Reward evaluation remains device-safe even when the trainer feeds tensor
  batches directly from GPU-backed training steps.

### Status
Accepted

---

## [2026-04-16] Use DataCollatorWithPadding-backed PPO collator for tensor-safe batches

### Background
After forcing the PPO dataset into torch format, the next trainer failure still
occurred at batch materialization time because the custom collator path was not
reliably returning a dictionary of tensors. In practice, the trainer ended up
receiving `None` instead of a batch payload.

### Decision
Replace the fragile handcrafted padding path in `scripts/train_ppo.py` with a
wrapper around Hugging Face's standard `DataCollatorWithPadding`, configured
with `return_tensors="pt"`.

The PPO collator now:

- delegates token padding to `DataCollatorWithPadding`;
- guarantees a returned batch dictionary;
- validates that `input_ids` exists and is a tensor;
- preserves non-model metadata fields such as `query`, `parent_smiles`, and
  `original_label` as Python lists for downstream reward reconstruction.

### Alternatives considered
1. Keep the custom collator and only add a missing `return batch`.
2. Drop metadata preservation and use a raw Hugging Face collator directly.
3. Remove the collator override and rely fully on trainer defaults.

### Consequences
- PPO batches now have a much stronger contract before they enter
  `trl.experimental.ppo.PPOTrainer`.
- Standard padding behavior is delegated to a well-tested transformers utility.
- Reward logic can still reconstruct prompt context without coupling that logic
  to the token padding implementation.

### Status
Accepted

---

## [2026-04-16] Monkey-patch experimental PPO wrapper to expose gradient checkpointing hooks

### Background
After the PPO data path was repaired, the next runtime failure came from
`trl.experimental.ppo` itself: the trainer's internal `PolicyAndValueWrapper`
was missing `gradient_checkpointing_disable` and
`gradient_checkpointing_enable`, even though the wrapped policy model already
implemented them.

### Decision
Patch the trainer-managed wrapper immediately after PPO trainer construction in
`scripts/train_ppo.py`:

- if `ppo_trainer.model.policy_model.gradient_checkpointing_disable` exists,
  bind it onto `ppo_trainer.model.gradient_checkpointing_disable`;
- do the same for `gradient_checkpointing_enable`.

In parallel, the PPO config builder now explicitly forwards
`gradient_checkpointing=False` when the runtime config signature supports that
field, so trainer-side generation does not try to rely on checkpoint toggling
more than necessary.

### Alternatives considered
1. Wait for an upstream TRL patch and leave the local training script broken.
2. Disable wrapper-managed generation entirely and fall back to a handwritten
   PPO rollout loop again.
3. Monkey-patch TRL package files directly in the environment.

### Consequences
- The repository no longer depends on an immediate upstream TRL fix for this
  wrapper attribute bug.
- The workaround stays local to the PPO entrypoint instead of mutating
  site-packages.
- Generation-time checkpoint toggling is less likely to derail short HPC PPO
  smoke tests.

### Status
Accepted

---

## [2026-04-16] Escalate PPO wrapper gradient-checkpointing fix from instance patch to class patch

### Background
The first local workaround for the experimental TRL wrapper bug patched the
trainer instance after construction. That turned out to be insufficient because
the runtime path could still recreate or access a wrapper object that had not
received the instance-local method bindings.

### Decision
Move the gradient-checkpointing workaround into the TRL import phase inside
`scripts/train_ppo.py` by patching the wrapper class itself:

- import `trl.experimental.ppo.ppo_trainer` when available;
- if `PolicyAndValueWrapper` exists, inject no-op
  `gradient_checkpointing_disable` and `gradient_checkpointing_enable` methods
  onto the class when they are missing.

This class-level patch supersedes the earlier instance-level workaround. The
config-side guard `gradient_checkpointing=False` remains in place as a second
line of defense.

### Alternatives considered
1. Keep stacking more instance-local patches after trainer construction.
2. Patch TRL directly inside site-packages on the target machine.
3. Revert to non-experimental PPO code paths entirely.

### Consequences
- Any `PolicyAndValueWrapper` instantiated after the patch inherits the missing
  methods automatically.
- The PPO script becomes less sensitive to internal wrapper recreation inside
  experimental TRL.
- The workaround remains local to the repository instead of mutating external
  package files.

### Status
Accepted

---

## [2026-04-16] Disable InternLM2 KV cache across PPO trainer wrappers before rollout

### Background
After the PPO entrypoint finally entered the trainer-managed batch generation
loop, ChemLLM's InternLM2 generation stack still failed inside
`modeling_internlm2.py` when the runtime attempted to consume incompatible
`past_key_values`. This failure occurred during PPO rollout rather than during
model initialization.

### Decision
Add a trainer-side runtime patch in `scripts/train_ppo.py` that recursively
walks the trainer model and common wrapper attributes such as:

- `policy_model`
- `pretrained_model`
- `model`
- `base_model`

For each discovered layer, the patch:

- forces `config.use_cache = False` when available;
- forces `generation_config.use_cache = False` when available;
- synchronizes `pad_token_id` and `eos_token_id` with the tokenizer.

This patch is applied immediately after PPO trainer construction and before
`ppo_trainer.train()`.

### Alternatives considered
1. Rely only on the earlier model-loading-time `use_cache=False` settings.
2. Try to sanitize or rewrite `past_key_values` formats for InternLM2.
3. Revert back to a completely manual rollout loop outside experimental TRL.

### Consequences
- PPO rollout is less likely to trigger InternLM2 KV-cache compatibility bugs
  through deeply wrapped trainer models.
- Token id settings stay aligned between tokenizer and generation config at the
  exact point where trainer-managed generation begins.
- The workaround remains local to the repository and does not require patching
  upstream model files.

### Status
Accepted

---

## [2026-04-16] Replace static InternLM2 cache patch with generate-method hijacking

### Background
Even after synchronizing `use_cache=False` into wrapped model configs, the PPO
runtime could still reintroduce cache-related generation kwargs dynamically
through experimental TRL batch generation. As a result, InternLM2 continued to
receive incompatible cache inputs during rollout.

### Decision
Change the PPO runtime workaround in `scripts/train_ppo.py` from static
generation-config edits to method hijacking:

- keep tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization;
- wrap `generate()` on the trainer model, policy model, and base model when
  available;
- force `kwargs["use_cache"] = False` on every generation call;
- drop `past_key_values` from generation kwargs before delegating to the
  original method.

### Alternatives considered
1. Keep stacking more static `config.use_cache=False` assignments.
2. Patch InternLM2 model code directly.
3. Revert back to a custom manual rollout loop outside experimental TRL.

### Consequences
- Cache disabling now applies at the exact generation call boundary where TRL
  injects kwargs.
- The workaround is less sensitive to internal trainer overrides of static
  generation config.
- The script keeps a narrow, local compatibility layer without mutating
  external package files.

### Status
Accepted

---

## [2026-04-16] Override InternLM2 prepare_inputs_for_generation at the deepest wrapped class

### Background
The generate-method hijack still proved insufficient once TRL, PEFT, and the
current transformers cache stack interacted through multiple wrapper layers.
InternLM2 continued to fail inside its own `prepare_inputs_for_generation`
implementation because the expected tuple-style cache structure no longer
matched what the runtime was trying to pass.

### Decision
Replace the generate-method interception in `scripts/train_ppo.py` with a
deeper class-level patch that:

- unwraps the trainer model through common wrapper attributes such as
  `policy_model`, `base_model`, and `model`;
- identifies the actual underlying causal LM class;
- overrides `prepare_inputs_for_generation` on that class;
- forces `past_key_values=None` and `use_cache=False` on every call.

Tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization remains in
place before the patch is applied.

### Alternatives considered
1. Keep layering more `generate()` wrappers on top of TRL and PEFT.
2. Patch InternLM2 source files directly in the runtime environment.
3. Abandon trainer-managed generation and return to a manual PPO loop.

### Consequences
- The cache-breaking branch is cut off at the exact InternLM2 method where the
  incompatibility manifests.
- The fix is more resilient to outer wrapper churn because it targets the
  effective model class rather than only the outermost object.
- The workaround stays local to repository code and can be removed later if
  upstream InternLM2 / transformers compatibility improves.

### Status
Accepted

---

## [2026-04-09] Treat counterfactual fragment generation as the sole primary objective

### Background
Earlier project stages and surrounding discussions may involve concept extraction, class-indicative subgraphs, or rationale-like objectives. Those formulations are related but not identical to the present v3 goal.

### Decision
Define the project strictly as counterfactual fragment generation: the generated fragment should be useful because deleting it is likely to flip the label.

### Alternatives considered
1. Optimize for fragment-only label predictiveness.
2. Optimize for class-shared concept extraction.
3. Optimize for rationale sufficiency rather than deletion-based effect.

### Consequences
- Reward design must include deletion-based semantics.
- Evaluation must include flip-related metrics.
- Old code paths aligned to non-counterfactual targets should be treated as legacy behavior.

### Status
Accepted

---

## [2026-04-09] Use modular repository structure rather than monolithic training script

### Background
The project involves chemistry checks, prompt generation, SFT, RL, evaluation, and logging. A single large script would make the system difficult to debug and easy to break.

### Decision
Adopt a modular structure with separate folders for data, models, rewards, training, evaluation, chemistry utilities, and general utilities.

### Alternatives considered
1. Keep everything in one file for convenience.
2. Split only by training stage.
3. Organize by experiments rather than functionality.

### Consequences
- Initial setup requires more files.
- The system becomes easier to test, replace, and extend.
- Reward and chemistry behavior can be validated independently.

### Status
Accepted

---

## [2026-04-09] Prioritize chemistry and reward layers before large-scale training code

### Background
The most brittle parts of this project are likely to be chemistry correctness and reward semantics. If these layers are unstable, training results will be misleading.

### Decision
Implement chemistry utilities and reward computation before building full SFT/RL training pipelines.

### Alternatives considered
1. Start with full training script and fill utility functions later.
2. Start with model wrapper only.
3. Start with evaluation only.

### Consequences
- Early iteration focuses on correctness rather than speed.
- The train loop can remain thinner and easier to debug.
- Reward failures and chemistry mismatches can be unit-tested.

### Status
Accepted

---

## [2026-04-09] Bootstrap the repository with typed module interfaces before any training implementation

### Background
After the documentation-first reset, the repository still had no source tree, no config folders, and no stable interface boundaries. If training code were introduced at that point, chemistry logic, reward semantics, and CLI behavior would likely become coupled again.

### Decision
Create the repository skeleton first and define minimum typed interfaces for:

- `src/data/`
- `src/chem/`
- `src/rewards/`
- `src/models/`
- `src/train/`
- `src/eval/`
- `src/utils/`

Also add thin placeholder CLI files under `scripts/` and minimal smoke tests for prompt and reward contracts.

The bootstrap intentionally implements only safe low-level helpers such as JSONL IO, prompt construction, reward-term aggregation, and text-level collapse diagnostics. RDKit-backed chemistry logic and all training loops remain deferred.

### Alternatives considered
1. Start with one large training script and refactor later.
2. Implement RDKit parsing and training code immediately without freezing interfaces first.
3. Leave the directory layout undocumented and decide file boundaries ad hoc.

### Consequences
- Future chemistry, reward, inference, and training work now has a stable import surface.
- The repository stays aligned to the deletion-based counterfactual objective instead of drifting toward concept extraction.
- Scripts remain thin by design, because domain logic now has clear module ownership.
- The next implementation phase can focus on chemistry correctness rather than file organization.

### Status
Accepted

---

## [2026-04-09] Implement chemistry utilities with RDKit-first behavior and explicit failure types

### Background
The v3 objective depends on structural correctness before any reward or training logic can be trusted. The repository therefore needs a chemistry layer that can safely parse SMILES, check connectedness and parent-substructure relations, and attempt fragment deletion without forcing train code to interpret raw RDKit exceptions.

### Decision
Implement the first chemistry utility layer in:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

The layer is RDKit-first when RDKit is available, but it must also degrade safely when RDKit is missing by returning structured failure types instead of crashing. The shared result dataclasses now carry normalized failure categories, and `validate_fragment_candidate(...)` aggregates these into one interface for downstream reward and evaluation code.

### Alternatives considered
1. Raise raw exceptions from RDKit and let train or eval scripts handle them.
2. Hard-require RDKit at import time and fail the whole repository if it is absent.
3. Delay chemistry implementation until reward or inference code exists.

### Consequences
- Reward and evaluation modules can consume chemistry results without duplicating error handling.
- The repository remains usable in environments where RDKit has not been installed yet.
- Deletion behavior is deterministic at the interface level, with the first matched connected fragment removed and failures surfaced explicitly.

### Status
Accepted

---

## [2026-04-10] Add a config-driven local and HPC runtime layer without hardcoded paths

### Background
The rebuilt repository now has modular source folders and chemistry utilities, but it still needs a stable way to run from a local workstation and from an HPC cluster. Without a shared runtime layer, path handling, log directories, and environment-specific behavior would drift into ad hoc script logic.

### Decision
Add a runtime adaptation layer based on repository-relative config files and thin CLI entrypoints.

This layer includes:

- base, local, hpc, sft, rl, and eval config files under `configs/`
- environment detection and config merging in `src/utils/env.py`
- repository-relative path resolution in `src/utils/paths.py`
- file-backed logging helpers in `src/utils/logging_utils.py`
- run entrypoints in `scripts/run_sft.py`, `scripts/run_rl.py`, `scripts/run_eval.py`, and `scripts/run_infer.py`
- Slurm templates for single-node single-GPU execution under `scripts/slurm/`

Model and tokenizer handling must support local filesystem paths and avoid silent remote downloads by default.

### Alternatives considered
1. Hardcode local and HPC paths directly inside the stage scripts.
2. Depend on an external YAML package before adding any runtime config support.
3. Jump directly to distributed training setup before the single-node path is stable.

### Consequences
- Local development and HPC submission now share one config and path resolution story.
- The codebase remains portable because configs stay relative and resolved at runtime.
- The repository can prepare deterministic run manifests before full SFT, RL, and evaluation logic is implemented.
- Distributed training remains intentionally out of scope for this phase.

### Status
Accepted

---

## [2026-04-10] Implement a minimal single-sample inference loop with heuristic fragment generation

### Background
The repository already has chemistry utilities and runtime adaptation, but it still needs a runnable end-to-end path from one parent SMILES to one fragment candidate. This is necessary to validate the IO contract before wiring any full SFT or RL training logic.

### Decision
Implement a minimal inference closure centered on `scripts/run_infer.py` and `src/eval/inference.py`.

This path:

- accepts one parent SMILES from CLI or config
- produces one heuristic fragment candidate without using a trained policy
- runs chemistry utilities for parseability, connectedness, and parent-substructure checks
- prints a structured JSON result
- stays independent of the full training stack

The heuristic prefers small connected parent substructures when chemistry backends are available, and falls back to the parent SMILES when no smaller valid candidate can be established.

### Alternatives considered
1. Keep `run_infer.py` as config-only runtime preparation.
2. Wait for a trained model before implementing any inference path.
3. Print only free-form text instead of a machine-readable result.

### Consequences
- The project now has a minimal runnable contract for one-sample inference.
- Chemistry utilities can be exercised from a real CLI path without invoking training code.
- The current fragment proposal is intentionally heuristic and not yet counterfactual-optimal, but it keeps the repository moving toward the v3 objective with a reviewable baseline.

### Status
Accepted

---

## [2026-04-14] Add a lightweight RandomForest Oracle and residual-based PPO reward wrapper

### Background
The repository has completed the SFT stage and is moving toward PPO. At that point the project needs a reward path that is both chemically constrained and fast enough to run inside RL rollouts. The existing codebase already has RDKit-backed capped-subgraph validation and deletion, but it still lacked:

- a lightweight local Oracle for fast activity scoring;
- a PPO-facing reward wrapper that combines validity, subgraph, and counterfactual terms;
- a clear decision on whether the semantic reward should score the fragment itself or the residual molecule after deletion.

### Decision
Add:

- `scripts/train_aids_oracle.py` to train a RandomForest classifier on Morgan fingerprints from `data/aids_dataset.csv`;
- `src/rewards/chem_rules.py` as a small structural reward engine for validity and parent-subgraph checks;
- `src/rewards/reward_calculator.py` as the PPO-facing reward wrapper that loads the Oracle bundle and computes `R_valid + R_subgraph + R_counterfactual`.

The counterfactual term is explicitly defined on the residual molecule `x \ g` after deleting the generated fragment, not on the fragment alone. Dummy atoms (`*`) are cleaned through RDKit graph editing before Morgan fingerprint extraction.

### Alternatives considered
1. Score the generated fragment by itself and treat that as the counterfactual term.
2. Put Oracle logic directly inside the PPO training script.
3. Save a custom Python class inside the pickle bundle instead of a plain dictionary.

### Consequences
- PPO reward computation stays aligned with the v3 deletion-based counterfactual objective.
- Oracle scoring remains fast enough for rollout-time use because it relies on RandomForest plus Morgan fingerprints.
- Structural chemistry checks stay modular and reusable instead of being buried in the train loop.
- The saved Oracle bundle is easier to reload across scripts because it stores plain metadata alongside the fitted sklearn model.

### Status
Accepted

---

## [2026-04-13] Add base-model metric tooling and presentation visualization for the SFT stage

### Background
The SFT stage now has stable headline numbers for validity, capping behavior, and token accuracy, but the project still lacked two practical tools: a deterministic way to compute baseline metrics from JSONL inference logs, and a reusable plotting pipeline for lab-meeting summaries.

### Decision
Add a small evaluation utility layer for base-model JSONL logs and extend the SFT visualization module with presentation-ready outputs.

This change introduces:

- `src/eval/base_metrics.py` for RDKit-backed capping/validity statistics
- `src/eval/base_inference.py` for deterministic base-model batch inference over `sft_val.jsonl`
- `scripts/eval_base_metrics.py` as a thin CLI entrypoint for base-model log evaluation
- `scripts/run_infer_base.py` as a thin CLI entrypoint that saves base-model predictions to JSONL
- an upgraded `src/eval/sft_visualization.py` that renders:
  - base-vs-SFT comparison bars
  - a dual-axis simulated training dynamics figure
  - a high-resolution RDKit rendering of a capped fragment
- the updated `scripts/visualize_sft_summary.py` CLI for parameterized figure generation

### Alternatives considered
1. Compute baseline metrics manually in notebooks.
2. Keep visualization code in a single ad hoc script under `scripts/`.
3. Hardcode baseline numbers directly into plotting code without a reusable evaluation helper.

### Consequences
- The repository now has a reproducible path from base-model inference logs to group-meeting figures.
- Evaluation logic remains modular under `src/eval/`, while CLI scripts stay thin and HPC-friendly.
- Presentation plots can be regenerated with different baseline numbers without editing source code.
- The training dynamics figure now uses discrete epoch-level line charts instead of dense smooth curves, which makes the plot easier to explain during presentations.

### Status
Accepted

---

## [2026-04-11] Add balanced capped-fragment SFT data preparation for the HIV dataset

### Background
The rebuilt repository already has RDKit-backed capped-subgraph utilities and thin runtime entrypoints, but it still lacked a concrete path for constructing supervised fine-tuning data from the local HIV CSV. Because the HIV benchmark is heavily label-imbalanced, a naive sample would underexpose positive molecules during SFT.

### Decision
Add a new balanced SFT data-preparation path centered on `scripts/prepare_sft_data.py` and `src/data/sft_preparation.py`.

This path:

- loads `data/raw/AIDS/HIV.csv` with pandas
- filters out invalid parent SMILES using the existing chemistry layer
- keeps all valid positive molecules and fills the remaining target size with sampled negatives
- constructs capped fragment targets by cutting one or two acyclic single bonds with RDKit dummy-atom capping
- validates generated fragments with the shared capped-subgraph checks
- writes minimal `instruction` / `output` JSONL files for ChemLLM SFT

### Alternatives considered
1. Reuse the placeholder `scripts/prepare_data.py` without adding reusable source modules.
2. Sample the HIV dataset according to its natural class distribution.
3. Generate weak targets without validating capped-subgraph correctness against the parent molecule.

### Consequences
- The repository now has an end-to-end path for building balanced SFT supervision data aligned with the capped-fragment objective.
- Positive molecules are intentionally overrepresented relative to the raw dataset so ChemLLM sees enough active-molecule structure during SFT.
- Fragment generation can still fail for some molecules, so the script explicitly reports success rate and observed label ratios after construction.

### Status
Accepted

---

## [2026-04-10] Treat dummy atoms as attachment-point caps in the chemistry layer

### Background
The project now needs chemically valid graph cutting. Plain atom deletion breaks
valence and makes the fragment contract ambiguous whenever the generated
subgraph has open attachment points to the rest of the parent molecule.

### Decision
Adopt dummy atoms (`*`) as the canonical capping representation inside
`src/chem/`.

This means:

- capped fragment SMILES such as `*c1ccccc1` are parsed and sanitized through a
  dummy-aware RDKit path;
- dummy atoms are treated as attachment-point queries when checking whether a
  fragment is a valid parent subgraph;
- capped deletion uses RDKit core replacement semantics so the remainder graph
  is also capped with dummy atoms instead of leaving broken chemistry.

### Alternatives considered
1. Keep using uncapped raw atom deletion and repair broken valence later.
2. Represent cut points with text markers outside SMILES instead of dummy atoms.
3. Treat dummy atoms as ordinary fragment atoms that should also be deleted from
   the parent match.

### Consequences
- The chemistry layer now distinguishes between real fragment atoms and dummy
  attachment points.
- Validation and deletion stay aligned with the v3 counterfactual objective,
  because both fragment extraction and residual construction use the same capped
  semantics.
- Future model outputs can use capped fragment SMILES directly without adding an
  extra post-processing representation.

### Status
Accepted

---

## [2026-04-10] Bootstrap local-only ChemLLM inference with dataset-backed sampling

### Background
The chemistry layer can now validate capped fragment SMILES, but the repository
still needs a real local-model inference path before SFT or RL code is added.
At the same time, the user needs a safe way to confirm that the local AIDS/HIV
CSV file and local ChemLLM checkpoint are usable without any accidental network
requests.

### Decision
Add a local-only ChemLLM inference bootstrap consisting of:

- `scripts/test_assets.py` for local dataset and model asset validation;
- a ChemLLM-specific prompt builder with hard-coded capped-fragment few-shot
  examples;
- a lightweight Hugging Face `ChemLLMGenerator` that always loads with
  `local_files_only=True`;
- dataset-backed `run_infer.py` behavior that samples one real molecule from the
  local AIDS/HIV CSV file when no explicit SMILES is provided.

If the local transformers stack is unavailable at runtime, the CLI is allowed to
fall back to the existing heuristic inference path, but it must surface the
model-side failure explicitly in the structured result.

### Alternatives considered
1. Wait for training code before integrating any real LLM backend.
2. Hardcode one demo SMILES instead of sampling a real local dataset row.
3. Allow `from_pretrained` to use default remote resolution behavior.

### Consequences
- The project now has a true local model inference integration point without
  depending on vLLM or distributed serving.
- Asset validation becomes safer because the repository refuses to reach out to
  Hugging Face when local files are incomplete.
- The inference CLI stays usable in lightweight dev environments because it can
  degrade to heuristic generation while still reporting why the model path did
  not run.

### Status
Accepted

---

## [2026-04-19] Make decoded chemistry PPO use a deletion-based counterfactual teacher oracle

### Background
The decoded chemistry PPO smoke test already proved that a fragment-level
teacher score on `core_fragment_smiles` could be computed, but the semantic
term that actually entered `total` still came from a fixed
`counterfactual_sem=-5.0` missing penalty whenever the deletion-based branch
was unavailable. That meant the PPO loop still was not aligned tightly enough
with the v3 counterfactual objective.

### Decision
Introduce an explicit deletion-based counterfactual teacher scorer that:

- deletes exactly one matched instance of the core fragment from the parent
  molecule;
- scores both the original parent and the residual parent with the teacher
  classifier;
- computes `cf_drop = p_before - p_after` and adds a configurable flip bonus
  when the residual prediction no longer matches the original label;
- uses this deletion-based `counterfactual_sem` as the default semantic term
  that enters the decoded PPO reward and total score.

The earlier fragment-level teacher score is retained only as an auxiliary
diagnostic field (`fragment_teacher_sem`) so we can still inspect whether the
generated fragment itself looks label-associated.

### Alternatives considered
1. Keep using the fragment-level teacher score as the main semantic reward.
2. Continue using the old residual-oracle fallback plus a fixed missing penalty.
3. Collapse fragment-level and counterfactual teacher semantics into one field.

### Consequences
- The decoded PPO path is now explicitly aligned with the repository's
  counterfactual deletion objective instead of a fragment-only semantic proxy.
- Logs can distinguish fragment-level diagnostics from the real
  counterfactual reward that enters PPO.
- Deletion failures, unavailable teacher backends, and disabled counterfactual
  teacher scoring are surfaced explicitly instead of being hidden behind an
  unexplained `-5.0`.

### Status
Accepted
## [2026-07-15] Use exact MolCLR node Wasserstein as the primary learned CCRCov distance

### Background
Graph-level pooled MolCLR distance can hide local correspondence, while the
existing Node-FGW line mixes learned node features with an explicit
shortest-path structure term. The final comparison needs a primary node-feature
transport line and a separately named structure-aware ablation.

### Decision
Use exact uniform-mass `ot.emd2` over MolCLR node cosine costs as
`MolCLR-Node-Wasserstein`. Keep Node-FGW (`lambda=0.5`) as an ablation. Share a
structure-independent v2 node embedding cache, but use independent symmetric
pair-cache namespaces. Calibrate thresholds from Ours only and use the resulting
absolute thresholds unchanged for every final baseline. Candidate selection is
external; the evaluator never changes Top20 order and does not compute
redundancy.

### Consequences
- WNode does not call shortest-path, GW/FGW, Sinkhorn, or networkx GED.
- Partial output now has a fingerprinted completed-pair resume contract.
- GCF-style reporting continues to aggregate match instances before Top-K
  prefix evaluation and can explicitly plot finite strict-recourse conditional
  median cost.
- Baseline training and candidate generation remain unchanged.

### Status
Accepted

---

## [2026-07-16] Make FGW presentation figures read-only and use the reported conditional-cost field

### Background
FGW final evaluation outputs already contain the Figure 3 prefix curve and
the dense Figure 4 threshold curve. Recomputing them merely to create paper
figures risks changing an evaluation artifact and is not permitted on an HPC
login node. Earlier ad hoc plotting also failed to recognize the actual
`conditional_median_cost` column.

### Decision
Use `scripts/plot_fgw_sota_figures.py` as a read-only post-processing tool.
It prioritizes `conditional_median_cost` (with documented compatibility
aliases), draws Figure 3 at the fixed q30 threshold, and emits separate
`K=1..10` and `K=1..20` figures. Figure 4 is read only from a dense `K=20`
curve; its normalized low-cost AUC always integrates over `[0, q30]`.

Submit plotting through `scripts/slurm/plot_fgw_sota_figures_gpu.sh`, which
uses the confirmed `A800` and `gpu:a800:1` resource combination from the
successful CLEAR Slurm template. No account, qos, or constraint is invented
because none is present in the verified project allocation pattern.

### Consequences
- The figure layer cannot alter strict-flip semantics, candidates, distances,
  or evaluator output.
- The displayed cost is explicitly described as the unified evaluator's
  conditional cost, not as original-paper unconditional GCFExplainer cost.
- A presentation audit permits only the narrowly scoped low-cost,
  compact-budget SOTA statement when all three configured checks pass.

### Status
Accepted

---
## Decision: GCF-style reports separate table and prefix thresholds

**Date:** 2026-07-16

GCF-style recourse post-processing now treats the Table 2 threshold and the
Figure 3 prefix threshold as separate report parameters. Both continue to fall
back to `--theta-star` for command compatibility, while `--table2-theta` and
`--figure3-theta` make the intended protocol explicit.

Figure 3 cost statistics use a fixed one-to-one mapping. In particular,
`conditional_median` means `conditional_median_cost`; it is not an alias for a
theta-covered cost. Full-range cost axes are the default, WNode artifacts use a
`wnode` filename slug, and historical unprefixed or `fgw` aliases are written
only when `--write-legacy-aliases` is requested. The compact Table 2 reports
overall conditional median cost, while theta-covered cost remains in a
separately named audit table.
## [2026-07-22] Use one curated SMILES benchmark for Mutagenicity baselines

### Background
Mutagenicity is available both as a raw/TU graph dataset and as raw, removed,
and curated SMILES tables. Allowing each baseline to choose its own source or
label encoding would make cross-method recourse results incomparable.

### Decision
Use the 4,247-row curated SMILES CSV as the primary v1 benchmark source, with
`1=mutagenic`, `0=non_mutagenic`, and main recourse direction `1 -> 0`. Audit
the 4,337-row raw CSV against TU graph order using the uniquely verified
inverse TU mapping. Preserve isomeric chemistry, reject invalid or disconnected
molecules, deduplicate canonical isomeric SMILES, exclude label conflicts, and
build one deterministic label-aware 70/10/10/10 scaffold-group split shared by
all methods.

### Consequences
- Raw files remain immutable provenance inputs.
- No neutralization, tautomer canonicalization, or stereochemistry removal is
  performed in v1.
- Ours and every baseline must use the same processed benchmark and split
  manifest for final Mutagenicity comparisons.
- Smoke outputs are isolated from the canonical full processed directory.

### Status
Accepted

---
## [2026-07-22] Fit the Mutagenicity RF teacher on train only

### Background
The unified Mutagenicity benchmark provides fixed train, validation,
calibration, and test splits. Reusing the older AIDS script's random holdout
would break this dataset contract and risk calibration/test leakage.

### Decision
Train a Morgan fingerprint RandomForest only on the fixed train split, select
its hyperparameters only on validation balanced accuracy, and reserve the
calibration and test splits for probability/threshold calibration and final
teacher-quality reporting respectively. Persist the model in the existing
oracle bundle format so shared teacher consumers can read it without changing
their core logic.

### Consequences
- The teacher data root is
  `outputs/hpc/datasets/final/mutagenicity_v1_processed`, not the parent full
  run directory.
- Every split receives the same probability and classification metric audit.
- Calibration and test metrics cannot influence model selection.

### Status
Accepted

---
## [2026-07-22] Reuse the AIDS SFT v3 target constructor for Mutagenicity

### Background

The processed Mutagenicity benchmark and RF teacher now provide fixed,
teacher-consistent source-label train and validation views. A new ChemLLM SFT
and stable-PPO data path must preserve the existing counterfactual-fragment
objective without inventing molecule-level pseudo-targets or admitting
calibration/test examples.

### Decision

Build Mutagenicity SFT targets by directly reusing
`select_reference_candidate_for_parent()` from the AIDS SFT v3 implementation.
This preserves the existing projection and Murcko candidate sources, core
normalization, dummy-atom audit representation, exact parent matching, size and
non-empty-residual filters, and optional deletion-based RF ranking. Build PPO
prompts one-to-one from every validated teacher-correct source parent and retain
the stable `molecule_id`. Read calibration/test files only as exclusion
manifests and fail on molecule, canonical-SMILES, or scaffold leakage.

### Consequences

- Mutagenicity uses the same weak target semantics as AIDS SFT v3 rather than a
  newly invented fragment label.
- Parents for which that constructor finds no target are explicit SFT misses;
  they are not assigned fallback pseudo-fragments.
- The PPO prompt set can cover all valid source parents independently of SFT
  target coverage.
- Existing AIDS, SFT trainer, stable PPO, selector, WNode, and CCRCov code paths
  remain unchanged.

### Status

Accepted

---
## [2026-07-22] Rebuild Mutagenicity teacher views from processed metadata

### Background

The first teacher-consistent Mutagenicity files were filtered directly from RF
prediction CSVs. Those files contain IDs, SMILES, labels, and predictions but
not the processed benchmark's `semantic_label`, `scaffold_smiles`, or `split`,
so they cannot satisfy the SFT/PPO parent and leakage-audit contract.

### Decision

Treat each fixed processed split as the authoritative metadata source and
strictly one-to-one join its corresponding RF prediction CSV by
`molecule_id`. Reject duplicate or missing IDs, row-set differences, SMILES or
label mismatches, and split mismatches. Preserve all processed columns, append
teacher fields and the fixed `source_label=1,target_label=0` direction, then
derive source-all, source teacher-correct, and target teacher-correct views.
The SFT/PPO smoke and full wrappers rebuild these views before data generation.

### Consequences

- Scaffold and semantic metadata are preserved from the fixed benchmark; they
  are never fabricated or recomputed from prediction-only files.
- No inner join may silently discard a molecule.
- Calibration and test source views remain exclusion manifests and do not
  become SFT/PPO training inputs.
- Existing teacher, ChemLLM, stable PPO, selector, WNode, and baseline logic is
  unchanged.

### Status

Accepted

---
## [2026-08-06] Recover COMRECGC with trace-only lineage and deterministic chemistry projection

### Background

The pinned COMRECGC native AIDS smoke produced 31 counterfactual candidates but
no common-recourse cluster, while the project Mutagenicity smoke produced four
official medoids that all failed RDKit decoding before RF inference. Changing
DBSCAN or random-walk parameters after seeing those outcomes would confound the
baseline, and replacing an invalid medoid with another cluster member would
change official greedy selection.

### Decision

Keep upstream commit `122f9341a360e9f06bb58a2f5823bb596021f6bf`
unmodified. Add project-owned action tracing that is parity-checked against the
frozen smoke payload, a read-only float64 DBSCAN geometry audit, and one
preregistered AIDS parent-density diagnostic using the unchanged 31-candidate
set and upstream parameters. Run native full with upstream defaults regardless
of density-retry yield, treating an empty common-recourse result as valid
science with N/A cost.

Persist selected-transition trace events as bounded, atomic JSONL chunks. Keep
only the first predecessor needed for candidate replay, release consumed
neighbor/action mappings after every move, and reuse only byte-identical chunks
on deterministic resume. This prevents the trace layer from retaining a second
full graph/path history or duplicating a completed trace prefix.

Run native AIDS data/model reads from the pinned upstream checkout so compute
jobs reuse the existing TU cache without network access. Trace parity requires
exact candidate tensors, order, and frequency. Repeated CUDA float32 importance
values may differ by at most `1e-6` absolute; the mismatch count and maximum
difference are persisted, and any larger or non-finite difference is fatal.
If that gate blocked only after a complete trace was serialized, adopt the
failed trace into a fresh directory instead of rerunning the random walk.
Recover actions omitted by cached upstream neighborhoods only when a unique
pinned-upstream single-edit graph delta reproduces the saved source and target;
record inferred actions separately and preserve the frozen candidate order.

For project Mutagenicity, replay the exact official action lineage and apply one
deterministic chemistry projection per raw candidate: retained chemistry comes
from source sidecars, new untyped edges are SINGLE, each action is sanitized
once, invalid actions are rolled back, and dependent actions are skipped. RF,
strict-flip, and WNode are unavailable to the repair decision. Preserve the
original official medoid and rank slot without compaction or backfill.

### Consequences

- COMRECGC-Native, COMRECGC-Raw-Project, and
  COMRECGC-Adapted-DeterministicChemRepair remain separate report routes.
- Zero repaired medoids or zero strict flips do not fail the engineering smoke
  when source/no-op, trace, determinism, lineage, and serialization gates pass.
- Unified evaluation delegates RF and WNode computation to the existing shared
  evaluator, then maps valid pair rows back to immutable official rank slots;
  invalid slots remain unavailable and are never compacted or backfilled.
- Full jobs are submitted only through the experiment registry and remain
  resumable from append-only recovery state. Mutagenicity full submission is
  deferred until the smoke gate reaches `COMPLETED/0:0`.
- Existing Ours, GCFExplainer, GlobalGCE, CLEAR, evaluator, RF, MolCLR, WNode,
  and frozen result artifacts are unchanged.

### Status

Accepted

---
## 2026-08-07: COMRECGC unmaterialized-tail eviction compatibility

The pinned upstream random walk can index a newly observed non-lead candidate before
materializing it in `graph_map`. If another candidate evicts that tail in the same
move, upstream performs `del graph_map[tail]` although the intended final state is
already absent. The project runtime now makes only that exact deletion idempotent:
the key must be the current candidate tail and must already be absent from
`graph_index_map`. All other missing deletions still fail. The scoped patch adds no
RNG calls, preserves candidate ordering/content, restores a plain dictionary on
exit, and records its activation count in the run manifest. Upstream source remains
unchanged.

---

## 2026-08-07: Compact full-generation action indexing

The first project AIDS full run remained CPU-bound and reached roughly 55 GiB
RSS while the pinned random walk was still populating its per-source transition
cache. The project trace layer was hashing both the source and target graph for
every enumerated neighbor, even though only five selected transitions per walk
step are part of the lineage evidence.

Full generation now associates each enumerated action with the exact target
object already retained by the pinned upstream transition cache through a weak
reference. Stable source/target SHA256 values are computed only after upstream
selects a transition. The selected trace, action replay, RNG calls, neighbor
enumeration/order, model calls, candidate payload, importance, DBSCAN inputs,
and greedy ordering are unchanged. Cache hit/miss counts and the compact trace
mode are recorded for audit. Smoke retains the prior stable-graph-pair trace so
the two implementations remain independently covered.

The AIDS project generation wrapper requests 192 GiB on the existing one-A800,
seven-CPU allocation and keeps the seven-day limit. Cross-job random-walk
resume remains disabled because the pinned runtime does not serialize all
Python/NumPy/Torch RNG and transition state; a fresh versioned output is
required instead of claiming an unsafe resume.

### Registered submission log-directory preflight

Slurm resolves relative `#SBATCH --output` and `--error` paths before the
batch script can run. A fresh Git worktree does not contain the ignored
`logs/` directory, so an otherwise valid registered job can fail at launch
without entering Python or producing a log. The shared `exp_sbatch.sh` entry
now creates and verifies the worktree-local `logs/` directory before invoking
the registry/submission client. This changes no Slurm resources, scientific
parameters, runtime algorithm, or output artifact contract.

---

## 2026-08-07: BACE Ours end-to-end paper artifact contract

### Background

The existing AIDS and Mutagenicity Ours paths already share candidate-pool,
coverage-heavy MMR selection, and strict-flip MolCLR-Node-Wasserstein
evaluation components. BACE must enter those components without changing
their algorithms or either existing dataset route.

### Decision

Add a project-owned BACE adapter that normalizes a local raw CSV to
`smiles`/`label`, canonicalizes molecules, records stable molecule and graph
hashes, and freezes deterministic scaffold-level train, validation,
calibration, and test splits. Train an independent Morgan-RF BACE teacher on
train only, select it on validation only, and materialize calibration/test
teacher-correct cohorts only after the model is frozen.

Reuse the stable300 checkpoint and its established generation parameters to
generate candidates from the teacher-correct BACE train-source cohort. A
lineage-only adapter adds stable BACE parent IDs without modifying or
reordering scientific candidate fields. The existing candidate audit must
pass its structural selector gate before the unchanged coverage-heavy MMR
selector freezes Top20.

Use Ours calibration distances to freeze the existing q05, q10, q20, q30,
q50, q70, q90 protocol, with q30 as `theta_star` and q90 as the cost cap. The
selected Top20 then enters the existing MolCLR-Node-Wasserstein evaluator on
the teacher-correct BACE test cohort under `strict_flip`; evaluation never
invokes a selector or changes candidate order.

Write the plotting columns directly at artifact creation time:

- Figure 3: `method,k,coverage,cost`;
- Figure 4: `method,threshold,coverage`;
- Table 2: `method,k,coverage,cost,flip_rate,cf_drop`.

### Consequences

- The repository does not download BACE implicitly; `data/raw/BACE/bace.csv`
  is an explicit, auditable input.
- Generation, audit, selector, evaluation, and artifact audit have separate
  Slurm stages and fresh-output gates. Calibration/test data are not loaded by
  generation or selection.
- The final artifact audit rejects schema, order, method, threshold,
  parent-ID, teacher, or MolCLR drift; it never recomputes metrics.
- This decision authorizes BACE Ours only. No BACE GlobalGCE, GCFExplainer, or
  COMRECGC experiment is implemented or submitted in this stage.
- Existing AIDS and Mutagenicity adapters, evaluators, candidates, and paper
  artifacts are unchanged.

### Generation persistence recovery

The shared generator already emits an integer `parent_id` equal to its
`parent_index`. The BACE lineage adapter preserves and verifies that field,
then adds the stable project `molecule_id`, graph hash, and candidate ID. It
never overwrites candidate values or changes row order.

If generation completes but lineage persistence fails, the wrapper may adopt
the raw pool only when its recorded source job, exact SHA256, expected row
count, and otherwise-empty persistence state all match. That recovery skips
model generation, records `algorithm_rerun=false`, and writes the normal
lineage and completion manifests. It cannot be used to resume a partial raw
pool or overwrite any finalized output.

---

## 2026-08-08: BACE official GCFExplainer project adaptation

### Decision

Run BACE through the existing project-owned official GCFExplainer boundary:
freeze the teacher-consistent train/validation cohorts, train the unchanged
official three-layer GNN using train and validation only, run official VRRW on
the 360 train-source parents, preserve official greedy summary order, then
filter that order only for molecular validity and BACE RF target label 0.
Neither RF probabilities nor WNode distances re-rank candidates.

The frozen BACE train/validation atom alphabet has nine channels. The only
available verified project NeuroSED checkpoint has the official ten-channel
Mutagenicity order. Its BACE transfer therefore deterministically removes the
phosphorus input column, which BACE does not contain, while preserving every
other weight byte-for-byte. The projection, source/output hashes, channel
orders, and absence of training/calibration/test access are recorded in a
manifest. This is an explicit transfer limitation, not a newly trained BACE
graph-edit distance model.

The BACE codec probe keeps every established chemistry and round-trip category
except phosphorus. Phosphorus is excluded explicitly because the frozen BACE
train vocabulary does not contain it; requiring the Mutagenicity-only `p`
category would reject the correct nine-channel dataset before generation.

Final comparison directly reuses the BACE fullgraph evaluator with the frozen
Ours threshold contract, BACE RF teacher, MolCLR-Node-Wasserstein distance,
and `strict_flip`. It writes the existing Figure 3, Figure 4, and Table 2
schemas directly; no post-experiment compatibility conversion is permitted.

### Consequences

- Official GCFExplainer algorithm files, GNN architecture, edit map, VRRW,
  importance, graph distance, dynamic teleportation, and greedy ordering are
  unchanged.
- Generation reads no calibration or test cohort; test is loaded only by the
  shared final evaluator after Top20 is frozen.
- AIDS and Mutagenicity GCFExplainer routes and all existing BACE Ours
  artifacts remain unchanged.

---

## 2026-08-09: BACE Ours calibration-only WNode prefix selection

### Decision

Treat the existing BACE Ours candidate pool, RF teacher, MolCLR checkpoint,
q05--q90 threshold contract, and test result as immutable provenance. Diagnose
the flat K=3--10 prefix before changing selection, then build a complete
calibration parent by unique-action matrix with the production hard-deletion,
all-match, RF `strict_flip`, and MolCLR-Node-Wasserstein implementations.

Select one nested Top20 outside evaluation using only the 60 frozen
calibration parents. Compare the existing selector with single-threshold,
multi-threshold, prefix-weighted, and coverage/structure-redundancy variants.
Use deterministic five-fold scaffold-grouped calibration CV and the fixed
pre-registered hyperparameter grid. The frozen BACE threshold manifest is an
input; Ours may not refit a method-specific threshold.

The selected sequence is immutable before the only new test evaluation.
Figure 3 uses its nested K=1--20 prefixes, Table 2 uses its first ten actions,
and Figure 4 uses its first twenty actions. The existing paper CSV schemas and
cost definition remain unchanged.

### Consequences

- Selector entrypoints reject test- or GCFExplainer-named inputs and record
  `test_used=false` and `gcf_result_used=false` in frozen provenance.
- Candidate expansion is conditional on a calibration-only limitation gate.
  If required, it reuses the frozen stable300 checkpoint and fixed
  multi-seed/multi-temperature regimes; it never reads test parents.
- Existing BACE Ours, BACE GCFExplainer, AIDS, and Mutagenicity artifacts are
  not overwritten or promoted automatically.
- A new BACE matrix module owns dataset identity, resume, and artifacts while
  delegating hard deletion, teacher scoring, all-match aggregation, and WNode
  distance to the existing production functions; the Mutagenicity matrix is
  unchanged.
- Older evaluator run manifests may record the teacher path, size, and mtime
  without an inline SHA256. The paper exporter computes the missing digest
  from that immutable file and still requires exact agreement with the frozen
  selection; it never accepts path equality as a substitute for hash equality.

---

## 2026-08-10: Connected molecular residuals for hard-deletion CCRCOV

### Decision

Version the molecular hard-deletion action as
`connected_sanitized_residual_v1`. An exact RDKit substructure match is a
feasible action only when deleting exactly those atoms leaves a nonempty,
sanitized molecule with exactly one connected component and no dot-separated
SMILES. The evaluator does not retain a largest component, repair chemistry,
or impose an additional attachment-count heuristic.

For multiple feasible matches, use
`existential_min_wnode_among_valid_connected_strict_flips_v1`: choose the
minimum MolCLR-Node-Wasserstein distance among connected, sanitized, RF
strict-flip matches, then break ties by higher CFDrop and the canonical atom
tuple. Distance, prediction, CFDrop, residual, and edit cost must all come from
that same match row.

The corrected BACE protocol refits the existing q05--q90 common calibration
grid from connected calibration actions, rebuilds the calibration matrix and
selector, and evaluates the frozen test cohort once. Ours and GCFExplainer use
the same new threshold manifest. Old BACE v1/v2 artifacts and caches remain
immutable evidence and are not promoted.

### Consequences

- Teacher and MolCLR calls occur only after connected residual validation.
- Connected distances use the independent
  `molclr_node_wasserstein_connected_residual_v3` cache namespace and logical
  action keys include parent, residual, candidate, match, model, and policy
  identities.
- Legacy action semantics remain the default for existing AIDS/Mutagenicity
  entrypoints; this change does not silently reinterpret their frozen files.
- The old BACE Ours result is invalid for paper use because its rank-1 action
  derived almost all apparent coverage from disconnected residuals.

The first connected calibration pass found only nine candidates with any
strict flip inside the common q30 threshold and a six-of-sixty parent union.
This activates the pre-registered, one-shot candidate expansion: the frozen
checkpoint is sampled at exactly three fixed seed/temperature/top-p regimes.
The merged formal pool rejects any row whose recorded source-parent deletion
is empty, unsanitized, or disconnected. After that fixed expansion the best
calibration-only sequence is frozen once even if the limitation diagnostic
remains true; no extra seed, test-guided retry, or adaptive budget is allowed.
# 2026-08-10: BACE v4 candidate universe uses chemistry feasibility, not source effect

- The historical BACE WNode matrix required `oracle_ok`, source `cf_drop >= 0.2`,
  and source `cf_flip` before a fragment could be evaluated across calibration
  parents.  This reduced the connected v3 pool from 151 canonical fragments to
  55 and conflated source-parent performance with class-level utility.
- `connected_feasible_v4` admits only parseable/canonical connected fragments
  with complete lineage, an exact source-parent match, a nonempty sanitized
  single-component source residual, and the frozen `(0.0, 0.85)` atom-ratio
  window.  Source oracle, flip, and CFDrop remain recorded ranking features.
- The legacy policy remains the default, so AIDS, Mutagenicity, and all frozen
  BACE v3 artifacts retain their prior behavior.
- A threshold protocol audit must run before BACE v4 final test evaluation.  An
  Ours-derived Q30 threshold is not treated as method-independent merely because
  downstream methods reuse it; if no cross-dataset preregistered rule can be
  proven, pooled calibration Q30/Q50 thresholds must be frozen before test use.

## 2026-08-10: BACE v4 connected-aware generation and complete GCF audit

### Decision

The BACE v4 expansion is an opt-in generation protocol named
`connected_deletion_v1`. It reuses the frozen ChemLLM/SFT/PPO checkpoints and
the train-only source cohort, while explicitly asking for an exact connected
substructure whose hard deletion leaves one nonempty sanitized component.
Round 1 has exactly three preregistered temperature/top-p/seed regimes with
eight samples per source. Every generated row is replayed through the same
connected hard-deletion implementation before entering the merged pool.

Source-parent RF flip, CFDrop, and oracle outcomes remain provenance and
ranking features, not matrix-admission gates. The default dataset prompt and
legacy merge behavior remain unchanged for all existing experiments.

The BACE GCFExplainer v4 audit scans every available native rank but freezes
the same first twenty valid connected RF counterfactuals in native order. Rows
after the frozen twentieth candidate are attrition evidence only; they cannot
change the sequence or trigger RF/WNode reranking.

### Consequences

- Test parents and GCF results are absent from Ours generation and selection.
- Existing BACE v3 Ours/GCF artifacts remain immutable.
- A full native-rank GCF audit can establish whether the old summary stopped
  early without silently changing official ordering semantics.
- Final test remains blocked until a method-independent calibration threshold
  contract and both frozen selections pass the common protocol gate.

Round 2 is also fixed before test. It runs only when the Round-1 calibration
matrix remains below either preregistered candidate-limitation bound. Its
source cohort is derived from calibration hard groups B/C/D and mapped to
train-only molecules by exact Bemis-Murcko scaffold, then deterministic Morgan
similarity. No frozen MolCLR source-cluster artifact exists for BACE, so the
manifest records that absence rather than inventing cluster identities.

If no cross-dataset threshold preregistration can be proven, v4 freezes a
method-balanced pooled calibration distribution: one minimum connected
strict-flip distance per parent and method, with total weight 0.5 for Ours and
0.5 for GCFExplainer. Pooled Q30 is the strict primary and pooled Q50 is the
standard sensitivity threshold. Both are frozen before the sole v4 test job.

---

## 2026-08-09: Lossless COMRECGC live-graph backing state

AIDS project full generation job `2164128` failed near step 46690 because a
multi-head restart retained a hash after upstream candidate-capacity eviction
removed its graph-map entry. The prior move-scoped transition protection did
not cover this restart-to-trace interval, and direct trace dictionary access
bypassed its missing-lookup diagnostics.

Keep the pinned upstream algorithm unchanged. Project runtime wrappers now
preserve logically evicted graph entries in a checksum-verified SQLite backing
store, resolve all trace graph reads through one fail-closed API, and pin the
complete move/trace lifecycle. Active candidate-map membership, RNG calls,
proposal order, importance, DBSCAN, greedy rank, seed, heads, and steps remain
unchanged. A bounded-cache stress gate must match the unbounded reference and
the full downstream chain must wait for a generation-integrity gate.

The failed job does not contain a complete atomic RNG/transition/graph-state
checkpoint, so its partial progress is evidence only and cannot be resumed.
Retry 6 uses a fresh versioned root and identical scientific parameters. See
`docs/postmortems/COMRECGC_AIDS_2164128_GRAPH_STATE_FAILURE.md`.

---

## 2026-08-08: Bounded COMRECGC full transition graph cache

Mutagenicity full generation exhausted its 96 GiB host-memory allocation after
22,156 of 50,000 steps. The streamed action trace occupied only about 42 MiB;
the dominant retained state was the pinned upstream `transitions` dictionary,
which stores every complete neighbor PyG graph alongside already-computed
hashes, importance values, and embeddings.

Full project generation now keeps the exact hashes, importance arrays,
embeddings, and enumerated edit actions, while retaining complete reconstructed
neighbor graphs only for a heads-sized LRU. An evicted expanded transition is
reconstructed from its exact action without another model call, random draw,
neighbor enumeration, or numerical recomputation. Current-head deletion remains
deferred until the move ends. Cache size, hits, misses, reconstruction counts,
and compact numeric bytes are recorded in the compatibility audit.

The Mutagenicity wrapper receives 128 GiB of host-memory headroom on the same
single A800 and seven CPUs. Seed, steps, heads, candidate capacity, sample size,
teleport probability, importance, DBSCAN parameters, candidate order, and
official source remain unchanged. Cross-job resume remains disabled because the
failed process did not persist complete RNG and candidate state.

---

## 2026-08-10: Close COMRECGC trace and frozen-payload graph references

A completed full walk can outlive the bounded hot `graph_map`: selected trace
chunks and compact transitions may still reference exact graphs held only in
the authoritative SQLite store. All project trace and transition-source reads
now use one fail-closed resolver. Before freezing, required graph references are
rehydrated, checksum-verified, atomically serialized, and verified again after
reload. Missing or colliding graph state is never skipped or substituted.

Freeze-only reuse is allowed only after a separate audit proves that the walk
completed, the fixed generation configuration matches, the backing store is
sound, selected-trace closure is complete, and transition integrity has no
unresolved evidence. A completed walk needs no RNG state for deterministic
freeze-only postprocessing; incomplete random-walk resume still requires a
complete atomic RNG/transition checkpoint and otherwise starts from step zero.
See `docs/postmortems/COMRECGC_RESOLVER_FREEZE_CLOSURE_20260810.md`.

---

## 2026-08-10: BACE connected four-method integration

BACE GlobalGCE reuses the established project-owned Mutagenicity adapter only
at its generic dense-molecule boundary. Dataset identity, the two-class BACE
train set, the 360 teacher-consistent train-source parents, checkpoints, stable
candidate IDs, manifests, and output roots remain BACE-specific. The official
GlobalGCE training and generation routines are unchanged. Its paper sequence
is frozen once by deterministic train-only frequency/support ranking after
parse, sanitize, connectivity, uniqueness, and frozen-teacher target checks;
calibration, test, WNode, and other methods do not rerank it.

BACE COMRECGC uses the fixed upstream full random-walk, importance, DBSCAN, and
greedy ordering parameters through the unified resolver/backing-store runtime.
The original medoid rank remains an immutable slot. Deterministic chemistry
repair may invalidate a slot, and the connected protocol may invalidate a
repaired full molecule, but neither operation compacts ranks or backfills from
another cluster. Only connected, sanitized candidates enter RF/WNode scoring;
an empty valid sequence remains an engineering PASS with zero coverage and
undefined conditional cost.

The completed BACE v4 Ours and GCFExplainer method directories are imported
into the new common4 root by exact per-file SHA256, without rerunning test or
rewriting CSV files. GlobalGCE and COMRECGC must use the same frozen v4 parent
cohort, teacher, MolCLR checkpoint, threshold manifest, strict-flip rule, and
cost schema. The common4 audit fails closed on any identity, connectivity,
ordering, or schema mismatch and emits plotting-ready Figure 3, Figure 4, and
Table 2 combined files without an adapter.

---

## 2026-08-11: BACE v4 launch and offline dependency gates

BACE GlobalGCE no longer relies on the official dataset-name dictionary having
a BACE entry. Its minimum frequency is either an explicit calibration candidate
or a frozen calibration-only manifest selected from the fixed train-cohort ratio
grid. Test metrics are rejected by the resolver and selector. The official
GlobalGCE source remains unchanged; the resolved value is injected only into the
isolated training process and recorded in its generator identity.

BACE COMRECGC now receives an explicit absolute checkout root and expected fixed
upstream commit. A short gate verifies source files, Git cleanliness, optional
vendor-manifest hashes, and imports before full generation. Offline immutable
vendor checkouts are allowed to carry only their integrity manifest as an
untracked file. No formal BACE job performs an online checkout.

Zero-runtime Slurm exit `0:53` is handled as launch infrastructure failure. The
three observed roots did not share one node, so retry jobs use `--requeue`
without excluding a node and depend on a short launch preflight.

---

## 2026-08-17: Persistent experiment state and two single-GPU lanes

The failed Mutagenicity and BACE COMRECGC runs exhausted the `/share` project
filesystem while writing their authoritative SQLite graph stores. Long-lived
SQLite state, WAL/SHM files, trace payloads, checkpoints, and WNode pair caches
therefore live on compute-visible persistent project scratch. `/share` retains
only versioned links, manifests, logs, audits, and final compact artifacts. A
500-move storage guard records free bytes, free inodes, database growth, and
projected final size. It checkpoints the WAL and fails closed before exhaustion;
because the pinned random walk still lacks a complete atomic RNG/transition
checkpoint, a guard stop never claims unsupported scientific resume.

The completed AIDS walk exposed a separate freeze alias mismatch: validator and
recovery could resolve the same graph under different official hashes. Both now
use one pure closure builder that persists canonical graph records, complete
alias chains, and original trace hashes, then verifies them after serialization.
This is a code-only repair in this recovery round; no AIDS job is submitted.

BACE GCF reads the already frozen, method-shared calibration threshold manifest
and never invokes `auto_quantile`; its threshold-independent pair cache can be
copied to scratch and reused. BACE GlobalGCE remains CPU-only. Its official DFS,
support, loss, epoch, and rule ordering stay fixed, while unused per-pattern
pandas report concatenation is removed and independent top-level gSpan roots and
training epochs receive atomic checkpoints and heartbeats.

New Mutagenicity and BACE GPU stages are represented as two serialized lanes.
Each GPU stage requests exactly one A800; CPU integrity, chemistry, gate, freeze,
checkpoint, and GlobalGCE stages request none. The static plan includes active
legacy jobs and fails closed unless each lane uses at most one GPU and their
combined theoretical concurrency is at most two. Protected jobs are reported as
account occupancy only and never become cancellation or dependency targets.
Because the cluster rejects zero-GPU jobs in the A800 partition, these CPU-only
stages use the default `intel` partition rather than reserving an idle GPU.

The BACE candidate-aware selector preserves a complete 20-rule A0 sequence
verbatim when adapting it to an expanded candidate universe. Fallback indices
are considered only while fewer than 20 A0 rules resolve; the selector never
extends an already complete prefix. This is an engineering boundary fix and
does not alter candidate ordering, calibration objectives, thresholds, or test
isolation.

GPU plan validation reconciles current Slurm allocations with an optional
`submitted_job_id` on planned stages. An active job that is the already
submitted instance of a plan stage is counted once and remains subject to the
plan's transitive lane ordering. Any active project GPU job that cannot be
matched to a planned stage still requires an explicit `afterany` resource-lane
dependency before a new job may enter that lane.

---

## 2026-08-18: Fullgraph GlobalGCE actions and fail-closed generation slices

BACE GlobalGCE frequency candidates are complete generated counterfactual
molecules. They are not deletion fragments. The previous calibration adapter
copied `canonical_smiles` into `final_fragment` and routed every pair through
Ours' hard-deletion primitive, which correctly produced zero substructure
matches but measured the wrong action. GlobalGCE now uses the same fullgraph
contract as the established GCF path: parse, sanitize, require one connected
component, query the frozen teacher, and compute parent-to-candidate WNode.
An all-zero applicability matrix fails as a schema/action-adapter error.

At low BACE support, official gSpan emits enough frequent patterns to exhaust
192 GiB before GlobalGCE performs its stable support sort and top-k slice. The
project wrapper now records deterministic traversal indices in a scratch SQLite
store, commits each top-level root, resumes only completed roots, and loads only
the official stable top-k order. This changes storage complexity, not mining,
support, rule ranking, or model training semantics.

COMRECGC continuation is deliberately stricter than a nonempty-output check.
A slice no-ops when the 50,000-step completion contract is present and resumes
only an atomic checkpoint carrying RNG, transition, trace, backing-store, and
referential-closure evidence. Without that evidence it fails closed and never
silently starts at step zero. The currently running BACE v6 process cannot gain
new checkpoint behavior retroactively; its continuation jobs therefore protect
completed output and expose an honest blocker if it times out without a safe
manifest.

The project GPU plan includes the two live one-GPU roots. Future MUT/AIDs work
shares one serialized lane and future BACE work shares the other; no stage may
request more than one GPU and total project concurrency remains at most two.

Completed-walk freeze recovery distinguishes serialized transition state from
the runtime proposal cache. If a payload persists transitions, their complete
source/destination closure remains mandatory. If it does not, the selected
trace is the authoritative record consumed by lineage reconstruction, and
historical runtime-cache mismatch counters remain visible diagnostics rather
than forcing a scientifically unnecessary 50,000-step rerun. This exception is
limited to completed walks; checkpoint resume still requires atomic RNG,
transition, trace, backing-store, and closure state.

---

## 2026-08-21: Treat AutoDL recovery lanes as durable local processes

The three-line recovery uses four local A800 processes rather than Slurm jobs.
Representing a local PID through the HPC experiment registry's numeric Slurm
contract would make status, cancellation, and restart evidence ambiguous. A
single monolithic launcher would also allow one failed lane to lose the other
lanes' process identity or to start BACE common4 before both required inputs
were scientifically complete.

The AutoDL controller therefore records `backend=autodl`, an operating-system
PID, and an explicit null Slurm job id. Every lane owns a persistent state
document, PID record, advisory writer lock, heartbeat, log, provenance record,
input-manifest digest, output-manifest digest, and orchestration sentinel.
State and JSON sentinels use fsync plus atomic replacement. Scientific stage
success is never inferred from exit code alone: a configured nonempty JSON
sentinel and its exact required fields must pass before the controller publishes
its own success sentinel. BACE common4 waits for both the BACE COMRECGC final
and GlobalGCE WNode sentinels. MUT and AIDS stages are rejected unless their
specification and runtime environment both prohibit generation.

The persistent filesystem owns state, logs, frozen inputs, outputs, and process
registry records. Disposable NVMe owns only independent cache/active roots,
including the fresh checkpointed BACE walk. `resume` skips already proven
stages, refuses a live second writer or orphan child, and delegates scientific
checkpoint selection to the checkpoint-aware command. The paired Slurm script
is intentionally read-only status plumbing required by repository CLI parity;
it cannot start or resume AutoDL work. These control-plane rules change no
candidate order, RNG use, random-walk state, graph semantics, threshold, or
evaluation protocol.

Scientific stages use one production runner with immutable primary,
`static_project`, and recursively enumerated Step0 input manifests. MUT and
AIDS formal freeze stages additionally require a non-formal preserved-lineage
smoke: full closure validation, recorded-action replay without legacy
inference, and the AIDS original-to-canonical serialization round trip. Smoke
evidence is bound to a deterministic repair-code content closure rather than a
pre-commit Git identity, so identical content remains valid after the repair
commit while any later code change fails closed.

BACE writes trace chunks directly to persistent storage, keeps active SQLite
and graph state on fast NVMe, and mirrors the latest two atomic checkpoints to
persistent storage. Its mandatory profile gate compares a formal-configuration
0-to-1000 run against a checkpoint at 500, continued execution beyond step 525,
SIGKILL, complete fast-state quarantine, persistent-only restore, and
500-to-1000 resume. The comparison ignores trace
materialization labels but requires identical published trace-row and pending-
event digests, counters, algorithm state, RNG, logical SQLite contents, and
candidate sequence. The profile report records progress-derived per-step
timing, GPU/CPU/I/O observations, and named cProfile aggregates; unavailable
platform observations are marked `NOT_OBSERVED` rather than synthesized.
Final BACE publication also requires the dedicated artifact audit; common4
verifies both upstream scientific sentinels and manifests.

Every reusable scientific substage now publishes a separate immutable proof.
The proof binds the physical completion marker and its required fields, a
content-verified output manifest, all three input-manifest digests, normalized
scientific argv, explicit scientific environment, the pinned external commit,
and a content digest covering all project Python/shell sources plus the HPC
config and production AutoDL spec.  Top-level sentinels repeat the same input
and code-closure binding, so a stale success file cannot bypass substage
verification.  Pre-commit integration smoke may cross only the Git-commit
identity boundary: its full code/config closure and external commit must remain
exact.  Marker-only legacy outputs, incomplete proof publication windows, and
changed config or dirty code all fail closed instead of being adopted.

Child-process control is PID-reuse resistant.  A signalable child is bound to
its kernel process start time, raw command-line digest, process group, run,
lane, stage, and normalized command digest.  Status, resume, and stop re-read
and compare that complete identity; a stale or reused numeric PID is retained
as audit evidence but is never signalled.  Worker environments also remove
password, token, API-key, authorization, credential, and private-key variables,
and the same names are rejected in explicit stage commands and overrides.

The completed Mutagenicity recovery also keeps two frozen cardinalities
separate.  Its serialized candidate payload contains 100,235 unique
counterfactual candidates, while the selected trace contains 224,690 recorded
transitions whose actions replay exactly.  Repeated selection of one candidate
is valid walk evidence; the formal Gate checks both cardinalities independently
instead of requiring the unique candidate population to equal transition
multiplicity.

An AIDS frozen closure may legitimately persist an empty
`alias_to_canonical` mapping when every original selected-trace official hash
already names its canonical graph record directly.  The Gate therefore requires
the alias field to exist with mapping type, permits `alias_count=0`, and still
requires zero alias cycles/dangling targets.  More importantly, every original
trace hash must resolve to identical normalized graph content before and after
real Torch serialization.  Nonempty alias maps remain subject to the same
round-trip check; recovery must not invent an alias merely to satisfy a
cardinality test.

Formal AutoDL lanes may now be activated incrementally with repeatable
`start/resume --lane`.  Omitted lanes remain durably `NOT_STARTED`; selecting
one for the first time is not a retry, and omitting `--lane` preserves the
four-lane launch.  Every incremental control action rechecks the global
code/vendor/GPU gates and only the selected lanes' immutable inputs and fresh
roots.  Cross-lane release requires both the existing scientific proof and a
matching persisted producer stage plus terminal producer-lane success.  The
lane sentinel is published before top-level `SUCCEEDED`, so a crash window can
wait safely but cannot expose a success state without its physical proof.

Worker control now uses the same fail-closed identity standard as scientific
children.  A worker PID record binds kernel start time, raw command-line
digest, process group, expected argv digest, run, lane, and spec digest; Linux
formal execution requires procfs and a private worker process group.  Resume,
status, and stop never adopt or signal a live numeric PID whose complete
identity differs, and stop revalidates immediately before signalling.  The
first run-state publication also binds the exact spec bytes, schema, path, and
all normalized roots.  The expanded persisted-state contract is explicitly
`state_schema_version=2` while the production spec/sentinel schema remains v1.
Later publication, resume, status, stop, and worker
startup reject spec/root drift instead of rewriting stale state under a new
configuration.

The AutoDL Python 3.10 runtime may expose neither `os.pidfd_open` nor
`signal.pidfd_send_signal`, and its glibc may expose neither corresponding C
wrapper.  Linux signalling therefore prefers the Python wrapper, then the libc
wrapper, then the architecture-allow-listed kernel syscalls (`pidfd_open=434`,
`pidfd_send_signal=424`) on x86_64/aarch64.  Unknown architectures and
`ENOSYS`, or an unavailable Linux procfs identity, make direct signalling
unavailable; Linux never falls back to `kill(numeric_pid, ...)`.  Only a
genuinely non-Linux host may use the existing double-identity-checked numeric
signal path.  `stop` publishes every lane's durable stop marker before
attempting a signal.  A live worker with no safe signalling support remains
`STOPPING` and observes that marker through its heartbeat loop, where its owned
`Popen` performs bounded child cleanup.  A persisted orphan with no live worker
cannot use that cooperative path, so it remains `ORPHANED_CHILD` with
`manual_stop_required=true` and receives no signal.  When pidfd is available,
persisted orphan-stage shutdown uses the exact pidfd-bound leader signal
instead of `killpg` from a recorded PGID.  The production stage runner owns
signal forwarding to its live scientific process group, closing the
controller-side PID/PGID reuse window without changing scientific work.

---

## 2026-08-21: Mirror the live global first-recorded COMRECGC predecessor during freeze recovery

The completed Mutagenicity trace can select the same normalized transition
more than once under different official source-hash aliases.  The live
`ActionTraceRecorder` indexes a predecessor by the global official target hash
and uses `setdefault` only for events carrying an exact recorded action.  The
freeze recovery had instead indexed by `(parent_id, target_hash)` and rejected
any repeated target whose source official hash differed.  That mismatch made
valid preserved walk evidence fail even though both source SHA, target SHA,
and recorded-action replay proved the same transition.  It also would have
missed the live global behavior for a target hash reached from two parents.

Freeze recovery now consumes the verified selected-trace order and retains the
first exact recorded event for each global official target hash, matching the
live recorder.  Every later event is still independently resolved from the
frozen payload, checked against its recorded source and target SHA, and exact-
replayed before it can be classified.  Exact repeats, normalized-source
aliases, distinct exact predecessor events, and cross-parent convergence are
reported separately, and a deterministic digest binds the selected
predecessor index.  A later recorded event supersedes only a legacy inferred
placeholder; unresolved competing legacy-only predecessors continue to fail
closed.  Frozen-payload official-hash collisions, SHA mismatches, replay
mismatches, cycles, and candidate paths that would cross parent identity also
remain fatal.

The graph stored for one global official hash is only a normalized-content
representative.  Its `comrecgc_parent_id` metadata can come from a later
content-equivalent parent because parent metadata is deliberately excluded
from the untyped graph identity.  Recovery therefore audits representative
source/target parent mismatches but never uses that metadata to select or
reject the first predecessor.  A candidate with recorded lineage takes its
parent from the selected predecessor event, exactly like the live recorder;
the complete selected-event chain must still have one consistent parent or
recovery fails closed.  Zero-action candidates continue to use their frozen
graph parent because they have no recorded predecessor event.

This is a recovery-parity repair only.  It changes no random-walk proposal,
selection, RNG use, candidate order, graph operation, threshold, model, or
evaluation behavior.  It enables the completed Mutagenicity walk to be frozen
without rerunning generation and applies the same evidence standard to AIDS.
No BACE scientific module or workflow is changed.

---

## 2026-08-22: Use task-specific frozen GNNs for BACE and TasteMolNet

The active fourth dataset changes from BBBP to the three-class TasteMolNet
task. Historical BBBP artifacts remain immutable; the migration changes only
the active registry, new automation, and paper-facing templates. TasteMolNet
uses labels Bitter=0, Sweet=1, and Tasteless=2, with Sweet as the source class.
Both `1 -> 0` and `1 -> 2` are strict untargeted counterfactuals, so new code
must use `pred_before == source_label and pred_after != source_label` rather
than `1 - label`.

BACE and TasteMolNet now require independent task-specific frozen molecular
GNN classifiers. GINE is the primary backbone; GIN, GCN, and GATv2 share one
registry and feature/checkpoint contract for later sensitivity studies. The
classifier, ChemLLM proposer, and MolCLR WNode encoder remain separate model
roles and have separately recorded checkpoint identities.

The existing BACE Morgan-RF teacher and all candidate, verification, selector,
or final artifacts bound to it are historical `RF_CONTAMINATED` evidence. They
cannot enter the new route. The existing scaffold-disjoint BACE
train/validation/calibration/test split is retained exactly, while the new
classifier is selected and temperature-calibrated using validation only.
Calibration remains exclusive to thresholds and selector fitting; test is
held out until final evaluation.

TasteMolNet data may come from an explicit local CSV, official supplementary
material, or a fixed upstream processed repository. The initial public source
is fixed at `MujeebOnawole/Taste_Prediction_RGCN` commit
`16af8ead8a17b6bd3941d9eb5879c5be75c14114`. Because that repository has no
standalone license file, its CSV remains untracked and the route records
`LICENSE_REVIEW_REQUIRED`. This task authorizes data preparation, graph
foundation, configs, tests, and tiny forward smoke only; heavy TasteMolNet
training remains disabled by `RUN_TASTEMOLNET=0`.

AutoDL uses a new lightweight GNN run registry and GPU locks instead of the
three-line recovery controller. It never invokes Slurm, takes at most two GPUs
that are stably idle, and does not cancel existing work. Paired Slurm wrappers
are maintained solely to keep repository CLI contracts synchronized.

The frozen BACE CSV is grouped by label, so bounded smoke subsets are selected
deterministically across all classes rather than taking a raw row prefix.
Formal BACE checkpoint selection uses validation ROC-AUC. A full classifier is
not allowed to publish its pass marker unless validation ROC-AUC is at least
0.65, predictions cover more than one class, source-class recall is positive,
and all reported probabilities and metrics are finite. The smoke also reloads
the published bundle and checks batch/single inference and deletion-forward
contracts; file presence alone is not a scientific pass.

---

## 2026-08-22: Persist the AutoDL GNN control plane and separate B4/B5 evidence

Fast NVMe clones are execution copies, not durable control roots. The frozen-
GNN runner now derives its default control root from the selected persistent
data root, rejects relative, out-of-data-root, symlink-final, and code-worktree
control paths, and stores the resolved control root in every detached launch
spec. Detached workers also require the exact launch-spec Python executable;
AutoDL shell entrypoints pin the `smiles_pip118` interpreter instead of relying
on a non-interactive shell's `python` resolution.

B4 never calibrates the B3 directory in place. It copies the verified,
uncalibrated B3 bundle to a fresh persistent output and fits temperature using
validation only, after requiring the current validation path and SHA-256 to
equal B3's frozen split manifest. B5 consumes only a PASS/FROZEN B4 manifest output. It selects
exactly 16 correctly-predicted source-class parents from calibration, loads the
GNN once for all deletion pairs, and records calibrated parent/residual
predictions under connected hard-deletion semantics. Every one of those 16
parents must have at least one valid connected deletion. Test is not loaded, RF
provenance is rejected, and B6 remains blocked until both B4 and B5 state and
gate documents are PASS.

A TasteMolNet `LICENSE_REVIEW_REQUIRED` marker is a blocker, not readiness
evidence. The full launcher exits nonzero without starting work and emits only
the license-blocked marker; setting the heavy-run switch cannot bypass it.

The BACE/Taste graph cache is a separate offline foundation artifact. It uses
only tensors and Python primitives, is loaded with `weights_only=True`, binds
all four split CSV identities and the molecular feature schema in one
manifest, and must be written to a fresh destination. Training does not yet
silently substitute this cache for CSV featurization; cache consumption will
be an explicit later change so provenance cannot drift unnoticed.

The molecular-GNN training bundle is versioned as
`molecular_gnn_checkpoint_v2`. Training no longer parses, featurizes, predicts,
or reports metrics for the held-out test CSV. It freezes only the test path and
streaming SHA-256 in `test_evaluation_status.json`, which must declare
`NOT_EVALUATED` and `test_loaded=false` and must match the split manifest.
`test_predictions.csv` is no longer a checkpoint artifact. Final frozen-model
evaluation remains the only route allowed to load the test split.

---

## 2026-08-22: Keep B6 honest and block legacy-teacher downstream kernels

The repository's stable PPO reward loop and candidate generator instantiate
the legacy teacher scorers directly. The historical BACE WNode action matrix
calls `predict_with_teacher`, and its selector/final artifacts preserve that
teacher identity. Passing a GNN checkpoint path through their existing CLI
would neither inject `GNNOracle` nor remove the old reward and verification
semantics.

Until those integrations are implemented, B6 runs a bounded calibrated-GNN
scoring preflight over B5's real connected deletion records. It performs one
batched oracle call, recomputes the complete counterfactual record, and proves
batch/single and B5 probability agreement. It explicitly records that no PPO
training or PPO reward occurred, then publishes
`BLOCKED_MISSING_GNN_PPO_INTEGRATION` with state/gate `BLOCKED`. A passing
diagnostic does not make the `B6_PPO_SMOKE` stage PASS and does not authorize
B7. B7 accepts only a future manifest proving at least one real PPO update with
the frozen GNN reward backend and checkpoint identity.

The same B6 evidence records a second independent blocker: there is no safely
reusable BACE policy initialization. Historical BACE PPO/LoRA artifacts are
RF-contaminated, unknown-provenance LoRA is rejected, and the oracle-neutral
ChemLLM base cannot directly satisfy the current PPO entrypoint's required
LoRA initialization. A GNN-aware reward adapter alone is therefore
insufficient to claim B6 PASS.

B7--B14 share one fail-closed stage driver. Each action checks the passing
predecessor's required output contract and the frozen B4 GNN provenance, then
publishes the exact missing scientific interface. A reserved child exit code
78 is registered as `BLOCKED` only when its declared log marker and blocker
evidence are complete; malformed or incomplete blockers remain `FAILED`.
Historical BACE PPO, candidate, verification, selector, and final artifacts
remain `RF_CONTAMINATED` and cannot satisfy any new predecessor gate.

Consequences:

- a GNN scoring probe cannot be presented as a PPO smoke or full result;
- downstream state names are backed by executable input/output checks rather
  than placeholder PASS markers;
- deterministic oracle-neutral merge/selector math may be adapted later, but
  only after new GNN-clean input schemas and provenance exist;
- the held-out test CSV is not parsed by B6 or any blocked preflight.

---

## 2026-08-22: Resolve COMRECGC trace actions against global graph identity

COMRECGC official hashes identify global graph content; representative parent
metadata remains provenance only. A recorded node-label-change may name an
equivalent node index from another representative. Recovery may remap that
index only when the pinned source and downstream SHA identities already match,
the official graph delta enumerates exactly one single edit, that edit is also
`NLC` with the same label, and replay produces the exact downstream payload.
Every other mismatch remains fail closed, and both recorded and resolved node
indices are persisted.

Recovery manifests now distinguish selected trace multiplicity from unique
candidate population. Fresh-root adoption records source checksums and proves
that serialization, lineage resolution, and freeze were rerun without a bare
symlink; historical output roots remain untouched.

An unsafe completed-generation validation now atomically persists its complete
audit before the recovery command exits nonzero.  The recovery path passes the
single in-memory validation result directly into that failure evidence, so a
large frozen graph/trace closure is never scanned a second time merely to learn
which checks failed.  A failed fresh root remains immutable evidence and a later
repair must use another fresh versioned root.

The first Mutagenicity fresh-root v2 attempt failed only because its launch
supplied an expected source commit with one extra trailing character (41
characters versus the 40-character commit in the checksum-bound source
configuration).  The strict commit gate and science resolver remain unchanged.
The audit now records actual/expected values, types, representations, and
lengths, and the historical mismatch/config checksum are frozen as a
reproducer.  The retry must use another fresh root and the exact 40-character
source commit.

---

## 2026-08-22: Add a provenance-clean BACE GNN-PPO route without rewriting stable PPO

The historical `B6_PPO_SMOKE` diagnostic remains immutable and BLOCKED.  A
new, fresh-root `B6_PPO_SMOKE_V2` route is additive: it injects a frozen,
validation-temperature-calibrated BACE GINE reward adapter into the existing
`run_stable_decoded_chem_ppo_loop`.  The shared loop still owns generation,
policy/reference/value forwards, clipping, KL control, gradients, AdamW, and
checkpointing.  The BACE entrypoint contains no private or substitute PPO
optimizer.

Policy initialization now fails closed on provenance.  An audit classifies
every supplied candidate as `CLEAN_CHEMLLM_BASE`,
`CLEAN_ORACLE_NEUTRAL_SFT`, `RF_CONTAMINATED`, `UNKNOWN`, or `MISSING`.
Unknown adapters are ineligible.  The raw ChemLLM base may be converted to a
fresh, zero-update LoRA needed by the stable loop, or a single bounded LoRA SFT
may be built from the exact frozen BACE train CSV.  That optional SFT uses only
deterministic chemistry targets, an internal scaffold/parent-disjoint
validation split, and no RF, GNN ranking, formal validation, calibration, or
test input.  Its manifest distinguishes the train-derived internal validation
that it does load from the untouched formal validation split that it does not.
Adapter bytes, source identity, and training-data identity are
bound in `policy_provenance.json`; later PPO stages reuse that manifest rather
than repeatedly hashing the base model.  The AutoDL build requires the passing
audit selection, verifies the audit CSV/selection hashes and exact selected
path, and reuses that one formal base-model content hash.

The reward adapter loads the GINE checkpoint once, caches calibrated parent
predictions by canonical SMILES plus checkpoint/temperature/schema identity,
and batches all valid connected hard-deletion residuals in each rollout.  It
uses source-class confidence drop plus strict-flip bonus and records complete
candidate-level chemistry, deletion, prediction, reward, checkpoint, policy,
and reference provenance.  Parent predictions and the classifier are detached
from policy gradients; RF is rejected before training.  Provenance likewise
distinguishes `calibration_dataset_loaded=false` from the required
`frozen_temperature_calibration_loaded=true` and binds the latter's hash.

`B6_PPO_SMOKE_V2` requires five to ten observed callbacks after real shared-
loop optimizer steps, changed trainable policy bytes, unchanged reference
bytes, a reloadable LoRA checkpoint, a saved candidate pool and reward
manifest, at least one valid GNN-scored deletion, finite rewards/metrics, no
held-out split, and KL below the hard limit.  `B7_PPO_FULL` is released only by
that exact passing manifest and runs 300 train-only updates with conservative
stable settings and checkpoints at 50, 100, 150, 200, 250, and 300.  A one-step
adapter canary is explicitly non-formal and cannot release B7.
Final and last-periodic LoRA config/weight artifacts are each deserialized,
checked finite, and bound by absolute path, byte size, and SHA-256.  A stable
`bace_lora_checkpoint_identity_v1` hash over file names/sizes/hashes is the
policy identity consumed by downstream candidate shards; an in-memory
parameter hash is never substituted for this disk identity.

B8--B14 retain their existing scientific kernels/blockers.  The new boundary
module can publish only `READY` after exact PASS dependencies and split rules;
it cannot publish a scientific PASS.  Test access requires a complete frozen
B12 calibration selector with ordered top-20 rule and model/input hashes.
B14 is manifest-only: it verifies B13 and frozen hashes without reopening raw
test bytes.

The AutoDL stage shell is a foreground payload.  The persistent controller
alone owns nohup, PID/heartbeat, GPU UUID locks, and retries.  `OUTPUT_ROOT`
must be the controller's exact fresh expected output below the persistent
runtime root.  No new HPC/Slurm launcher is added in this change because the
active execution path is AutoDL-only; any repository-paired Slurm wrapper is a
static CLI contract and is not invoked by this route.

---

## 2026-08-22: Reuse `exp_run` beneath one persistent four-GPU DAG

The four-GPU recovery controller is a scheduler, not a second scientific
worker implementation. Task commands, dependencies, immutable inputs, output
contracts, resource class, bounded OOM policy, and fixed parent sharding are
declared in a frozen persistent manifest. Every detached task is launched
through the existing AutoDL `exp_run` boundary, which remains responsible for
physical GPU UUID locks, project slots, tmux/nohup workers, atomic stage/run
documents, gates, and the canonical append-only registry.

The ordinary frozen-GNN launchers retain their two-GPU hard ceiling. Only the
new controller explicitly opts into the audited four-GPU ceiling. It samples
idle GPUs for at least 60 seconds, gates CPU/RAM/persistent disk, and audits PID
generation plus GPU-lock metadata without deleting locks or signalling any
process. One CUDA OOM may retry with a lower manifest-bound batch and a fresh
attempt-qualified output; semantic failures never retry.

B11-style parallelism uses a single frozen parent-ID manifest to materialize
deterministic, disjoint, exhaustive shard manifests. Held-out test access is
rejected before a frozen B12 selector and is allowed only for the one-shot,
read-only B13 task. TasteMolNet stays blocked on license review, and all paper
paths remain frozen. The default integration template is intentionally
BLOCKED until Commit A/B foreground argv and evidence contracts are filled, so
the controller cannot accidentally execute placeholder or legacy-RF science.
The clean PPO route is named `B6_PPO_SMOKE_V2` and uses controller/`exp_run`
generic stage documents; it never overwrites the legacy blocked B6 record.
After B7, B8 base and B9 high-temperature pools expose four fixed train-parent
shards each and may fill the four-card queue concurrently; B10 alone joins both
passing pools.
`B14_FROZEN` is manifest-only: it verifies the passing B13 bundle, selector
freeze, and bound hashes without loading calibration or held-out test bytes.

MUT/AIDS jobs that were launched before controller deployment may be bound by
an explicit `adopt_existing_run_id`. Adoption is not a loose PID import: the
controller verifies the immutable `exp_run` launch spec, input SHA, output and
gate contract, interpreter, environment, dataset/stage, and CPU/GPU binding.
It then monitors or accepts that one run and never schedules a second writer.
The adopted run's immutable execution worktree is verified independently and
need not equal the newer controller worktree. Exact launch environment equality
is required; undeclared extra variables reject adoption.

The clean BACE queue includes the provenance audit, clean initializer, adapter
canary, fresh `B6_PPO_SMOKE_V2`, and B7. The canary is required before B6 but
cannot itself release B7. Commit-D preparation work may fan out after B6 beside
B7 without becoming a B7 dependency. B13 alone receives four frozen test-parent
shards after B12; test-looking argv, config, environment, and manifest paths are
rejected everywhere else. B14 remains manifest-only.

The controller mirrors each transition and each tick to the user-facing
runtime registry under `outputs/autodl/experiment_registry` and appends a
human-readable runtime experiment log. External timestamped policy-audit CSVs
are first-class required outputs, while large checkpoint identity is frozen
from the launch spec instead of re-hashed on each tick. AutoDL shells and child
specs use `PYTHONDONTWRITEBYTECODE=1` so immutable execution clones remain
byte-for-byte unchanged by imports.

---

## 2026-08-22: Separate deterministic adapter coverage from formal B6 yield

The one-update BACE adapter canary previously required its very small
PPO-generated sample to contain a connected deletion.  A run could therefore
complete the real optimizer, parameter-change, checkpoint, and reload checks
yet fail solely because two sampled strings happened not to yield a legal
deletion.  Increasing that sample count would reduce but not remove the
probabilistic gate.

The canary now inspects the exact eight fixed source-class parents already read
from the frozen BACE train CSV.  It deterministically enumerates bounded
parent-derived fragments, selects the first fragment with a valid connected
hard deletion under the production chemistry primitive, and sends the selected
rows through the same already-loaded `BatchedGNNPPORewardAdapter` used by the
subsequent stable-PPO update.  PASS requires an observed residual prediction
batch, an increased GNN-scored-deletion counter, one unchanged oracle load, the
train CSV hash and parent identities, and explicit absence of RF,
formal-calibration-dataset, and test access.  It does not read B5 evidence.

This preflight is integration evidence only: it is marked non-formal and cannot
release B7.  The canary still requires a real stable optimizer update, changed
policy bytes, unchanged reference bytes, and reloadable checkpoints.  Formal
`B6_PPO_SMOKE_V2` ignores the preflight and retains its existing requirement
that at least one of B6's own PPO-generated candidates receives a real GNN
deletion score.

---

## 2026-08-22: Fail closed at the A/B/C release boundary and make B7 resumable

The four-GPU controller allocates only UUIDs whose advisory-lock audit is
positively `AVAILABLE` or whose stale metadata is proven to have neither a
live owner PID nor a compute process. Non-LOCKED JSON metadata cannot override
a held advisory lock. STARTING has a bounded launch grace, and RUNNING is
healthy only while its exact PID generation remains alive; controller-written
heartbeats do not substitute for worker evidence. A lost worker or recognized
transient I/O failure receives at most one independent transient retry, while
OOM retains its separate one-time lower-batch retry. Every retry writes a new
attempt-qualified output. Once one shard becomes terminal, already-running
siblings drain, no new sibling launches, and the aggregate becomes terminal
only after all active siblings exit.

Controller-owned `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`TOKENIZERS_PARALLELISM=false` limits are computed from the four-worker CPU
budget, frozen in launch evidence, and cannot be overridden by task input.
They apply to newly launched jobs only: exact adoption continues to compare
the historical environment byte-for-byte and never claims defaults that were
not present. Custom BACE policy audit/initializer workers use non-primary
dataset namespaces so the primary B0--B14 parser cannot reject their auxiliary
stage names. Dependency inputs resolve the exact passing retry output instead
of a hard-coded attempt zero. The dashboard preserves the raw Taste license
BLOCKED state and additionally reports `workload_state`, which excludes that
immutable license gate from executable-science completion.

B7 now rejects B6 based on the complete physical predecessor bundle rather
than manifest labels alone. It requires the same execution commit as B7,
recomputes final and periodic LoRA identities from bytes, rebuilds reward
summary from the candidate JSONL, and binds GNN/checkpoint, initializer,
reference, RF, calibration, and test provenance. Consequently a live B6 from
an earlier pre-release commit remains diagnostic evidence and cannot release a
final-commit B7; the final controller must launch a fresh B6 and B7 from one
immutable project root/commit.

Formal B7 periodic checkpoints now atomically publish policy, value head,
optimizer, KL/validation control, observer/candidate history, and Python/Torch
RNG state under an exact resume contract. A same-batch transient retry may
adopt only the latest complete, hash-valid, incomplete periodic checkpoint
into a fresh output root. OOM retry changes batch size and therefore restarts
from the clean initializer. The final B7 gate binds all six periodic resume
states plus the final state before writing PASS.
# 2026-08-22 — TasteMolNet publication route remains explicitly license-gated

- Added an offline, fresh-output license audit for the exact prepared
  TasteMolNet source. Public repository/paper/CSV availability is evidence of
  accessibility, not permission to reuse the compiled data in a formal paper.
- The heavy route remains `BLOCKED_LICENSE_REVIEW` unless the prepared
  provenance already binds an explicit reviewed license or the user supplies a
  nonempty approval/terms file through `TASTEMOLNET_LICENSE_APPROVAL_FILE`.
- The audit records the exact prepared provenance and approval-file identity;
  it never downloads labels, rewrites the prepared dataset, or unlocks work by
  inference from an open-access article license.
## 2026-08-22 — Adopt recovered COMRECGC generations read-only for paper-cell continuation

- AIDS and Mutagenicity lineage recovery roots are immutable generation inputs;
  the continuation does not regenerate random walks or write into those roots.
- `scripts/autodl/run_comrecgc_standardized_continuation.py` validates the
  recovery, closure, and upstream-commit gates, records a small-file adoption
  manifest, then reruns native common-recourse clustering, deterministic
  chemistry validation, unified RF/WNode held-out evaluation, the project-full
  gate, and atomic standardized freeze under one fresh output root.
- The multi-gigabyte counterfactual payload is not repeatedly hashed by the
  adoption preflight. Its frozen claimed identity is cross-checked across the
  existing recovery gates; the first scientific consumer still performs its
  own exact payload verification once.
- Parent metadata remains provenance only. Graph content and transition lineage
  retain the previously frozen global-graph-identity resolver and uniqueness
  checks.
- This campaign runs only on AutoDL. The paired Slurm wrapper is retained solely
  as static CLI parity required by repository policy and is not submitted.

## 2026-08-22 — Isolate the four-by-four controller from recovery history

- The four-method × four-dataset campaign reuses the hardened manifest-driven
  scheduler and status implementation, but persists under the separate
  `four_methods_four_datasets_continuation` control namespace.
- Its launcher never stops or mutates an earlier recovery controller. A new
  controller ID, manifest, state tree, heartbeat, and run registry are required
  for every continuation release.
- Taste heavy work remains an explicit manifest gate; setting a launcher
  environment variable cannot synthesize license approval.
- The controller runs the Taste license inspection as a CPU audit task. Audit
  completion may return zero for orchestration while the immutable gate remains
  `BLOCKED_LICENSE_REVIEW`; only `heavy_route_authorized=true` can release Taste
  training or evaluation.
- The final live manifest is composed from independently reviewed task
  fragments. Composition rejects duplicate IDs and every unresolved
  `__CONFIGURE_` placeholder, records fragment hashes once, validates the
  resulting manifest with the production loader, and removes an invalid
  partial manifest rather than leaving a launchable file.
- When the BACE B11--B14 continuation is the base fragment, composition also
  preserves its validated predecessor-quiescence policy. This keeps the old v2
  controller lock guard active for the wider four-by-four controller instead
  of silently dropping the anti-double-writer boundary.

## 2026-08-22 — Freeze exact core tasks for the partial four-by-four run

- A fresh task-fragment builder resolves the immutable AIDS/Mutagenicity
  recovery roots, RF teachers, distance models, MolCLR inputs, threshold
  contracts, and Taste prepared provenance before publishing controller tasks.
- AIDS/Mutagenicity COMRECGC continue only from the recovered generations and
  write fresh standardized cells; they do not rerun generation.
- The Taste license audit is a CPU evidence task. All four Taste heavy cells
  remain static `BLOCKED_LICENSE_REVIEW` tasks until an explicit approval is
  supplied in a later fresh manifest.
- The paired Slurm wrapper is static repository CLI parity only. This campaign
  neither connects to nor submits work on HPC.

## [2026-08-22] BACE native baselines share the frozen calibrated GINE

### Background

The historical BACE baseline routes were bound to method-local classifiers or
RF filtering and therefore cannot enter the new same-classifier main matrix.

### Decision

- GCFExplainer VRRW and ComRecGC generation now call the same calibrated BACE
  GINE bundle through a project-owned native one-hot-graph decoder. Invalid
  edited graphs fail to the source class; there is no RF or repair fallback.
- Native train generation is followed by calibration-only WNode selection and
  held-out test access only after an immutable top-20 freeze. Full graphs remain
  full-graph actions; ComRecGC lineage/common-recourse semantics are retained.
- GlobalGCE initially failed closed because its attachment-aware LHS→RHS rule
  application engine was absent. That action engine and pinned-source tensor
  parity are now implemented, but full training remains fail-closed with
  `BLOCKED_GLOBALGCE_FROZEN_GINE_DIFFERENTIABLE_RULE_TRAINING_UNAVAILABLE`;
  the accepted decision above records the exact continuous-decoder/discrete-
  GINE gradient boundary. Historical RF/full-graph conversions remain invalid
  substitutes.
- AutoDL foreground commands and terminal artifact contracts are documented in
  `docs/AUTODL_BACE_BASELINE_GNN_ROUTES.md`. Its paired Slurm wrapper is static
  CLI parity only and is never submitted by this AutoDL-only campaign.

### Consequences

GCFExplainer and ComRecGC can now produce provenance-clean BACE cells while
retaining their native actions. GlobalGCE native application and GINE forward
evaluation are auditable, while its full cell remains visibly incomplete
instead of contaminating the matrix with a semantically different substitute.

### Status

Accepted

## 2026-08-22 — Separate legacy raw reuse from frozen-cell adoption

Legacy AIDS/Mutagenicity artifacts now record three independent facts:
generation reuse, calibration/order reuse, and held-out evaluation reuse. A
candidate pool is not a paper cell. A PNG or an old combined CSV is not a
checksum/provenance closure. Only the exact Mutagenicity Ours final result is
eligible for immediate adoption because it has a frozen selector, complete
held-out Cartesian matrix, fixed thresholds, RF/MolCLR identities, and a
`final_result_manifest.json` file closure that can be independently audited.
This preserves the original scientific closure but does not make it a final
common-protocol cell: its historical theta/cap and 14-point Figure 4 differ
from the matched 601-point protocol, so its standardized status is explicitly
`STALE_METRIC` pending deterministic pair-matrix re-export.

That result is copied into a fresh root only after checking all source bytes,
the absence of live writers, selector-before-test provenance, strict-flip
semantics, and deterministic metric reconstruction. The original output is
never amended with a retroactive audit. Physical-file hashes are cached during
one adoption so the multi-file closure is not pointlessly recomputed.

The reconstruction necessarily reads the already-frozen held-out cohort. It
therefore runs once before controller launch and is not mislabeled as a
manifest-only controller task. The controller receives only the resulting
persistent standardized manifest, runs a manifest-only closure verification,
and then inventories the remaining raw roots. The source-spec copy is also
persisted outside the fast execution worktree before launch.
The task fragment addresses those files through the full
`{runtime_root}/outputs/autodl/paper_matrix/four_methods_four_datasets_v1`
prefix. The shorter `{artifact_root}` placeholder is not interchangeable: in
the production scheduler it denotes `{runtime_root}/outputs`, not the matrix
root chosen by the foreground adoption command.

Mutagenicity GCFExplainer retains an exact completed Top20 export but lacks its
standalone frozen-candidate manifest. A deterministic, generation-free freeze
step now verifies the exporter completion/filter audit, exact CSV/order/native
ranks, and frozen RF identity before publishing a fresh package for the
existing calibration/test kernels. Mutagenicity GlobalGCE and AIDS Ours remain
`INCOMPLETE`. The locally observed AIDS GCFExplainer and GlobalGCE evidence was
not transferred in the AutoDL Step0 payload, so the AutoDL source registry
marks both `MISSING_SOURCE_NOT_TRANSFERRED`; GlobalGCE additionally remains
code-blocked by absent native LHS-to-RHS attachment mapping after any future
transfer. It may not be coerced into deletion semantics. CLEAR remains
excluded and can never stand in for ComRecGC.
# 2026-08-22 — Preserve dormant backbone and selected-rule stability axes

- The four-by-four primary controller does not schedule backbone ablations.
  `scripts/autodl/run_backbone_ablation.py` records the full
  GINE/GIN/GCN/GATv2 × seed/config axes and emits tasks only with an explicit
  enable flag; TasteMolNet additionally requires an explicit license PASS.
- Frozen selector comparisons use `scripts/compare_selected_rule_stability.py`.
  Exact rule Jaccard, Morgan bidirectional mean-max similarity, scaffold
  Jaccard, coverage-set Jaccard, and destination-distribution similarity are
  computed from two frozen manifests without opening held-out molecules.
- These entry points are intentionally absent from the primary controller
  manifest. Static Slurm wrappers exist only for repository CLI parity and are
  not submitted by the AutoDL-only campaign.

The matched Mutagenicity expectation is tracked in
`configs/autodl/mutagenicity_matched_protocol_v1.json`: a hash-bound 601-point
linear 0..0.0535 grid, theta 0.05, cap 0.0535,
`existing_frozen_protocol`, and `test_used_for_selection=false`. The persistent
fragment schedules generation-free GCF freeze, calibration-only selector
freeze, and post-freeze held-out evaluation in dependency order. The generic
controller recognizes those AM stages in the same no-test-before-freeze gate
used by B12/B13.
# 2026-08-22 — Bind the final matrix audit to passing controller outputs

- The registry CLI accepts repeatable explicit `DATASET/METHOD=ROOT` bindings
  so the controller can pass exact successful attempt roots without a mutable
  intermediate map.
- `build_four_by_four_final_tasks.py` requires all sixteen distinct terminal
  task IDs, emits one post-cell CPU audit, and writes the dependency contract
  consumed by the no-fabrication exporter. A blocked cell keeps both tasks
  non-READY; it is never represented by a numeric zero.
# 2026-08-22 — Keep BACE primary routes ahead of legacy post-processing

- Mutagenicity GCF candidate freeze remains an early CPU task, but its
  calibration and held-out GPU stages use priorities 300/301. BACE GCF,
  ComRecGC, and B11 therefore receive the initial four cards as preregistered;
  the generation-free legacy evaluation enters the queue only after the BACE
  primary chain releases capacity.
# 2026-08-22 — Materialize the shared AIDS/Mut matched threshold contract once

- The tracked 601-point Mutagenicity protocol already identifies the audited
  matched AIDS/Mut frozen protocol. A small fresh-output builder clones that
  exact grid/theta/cap/hash to the AIDS expectation before the one-time matrix
  audit; it does not inspect or derive thresholds from held-out curves.

# 2026-08-22 — Gate AIDS/Mut ComRecGC held-out evaluation behind threshold freeze

- The combined registry emits the canonical
  `MolCLR-Node-Wasserstein` 601-point frozen threshold contract. A small
  manifest-only task validates and republishes that exact contract before the
  ComRecGC continuation is allowed to open the held-out cohort.
- The ComRecGC continuation is therefore a post-freeze, read-only test task;
  it no longer declares calibration access inside the held-out task. Taste
  licensing and its four blocked placeholders are manifest-only and declare
  no raw split access while the license gate remains blocked.

# 2026-08-22 — Verify adopted COMRECGC payload bytes before continuation

- The fresh standardized continuation computes the actual SHA-256 of the
  adopted `counterfactuals.pt` exactly once at its entry gate and requires it
  to equal the identity independently recorded by the run, completion, and
  freeze-recovery manifests.
- Critical frozen-closure manifests are protected by stat and SHA-256 snapshots
  before and after that payload pass, then checked again before final `PASS`.
  The large payload is not rehashed at the final check; its device, inode,
  mode, size, mtime, and ctime must remain unchanged.
- Linux procfs is mandatory. Any process holding a writable file descriptor
  below the adopted root, or to a protected inode through another path, blocks
  adoption. Missing or unreadable procfs evidence also fails closed.

## 2026-08-22 — Scope migrated ComRecGC checkout trust to each Git subprocess

### Background

The AutoDL Step0 vendor checkout is intentionally read-only and retains its
archive owner (`501:50`), while the controller runs as `root`. Git therefore
rejected the otherwise pinned checkout as dubious ownership before BACE
ComRecGC generation loaded any data or model. The generic BACE route preflight
validated only the frozen GINE and incorrectly published `READY` without
opening the ComRecGC checkout.

### Decision

AutoDL's Git 2.34.1 ownership-check backport does not honor `safe.directory`
from `git -c`. Every Git query made by the shared ComRecGC checkout validator
therefore redirects `GIT_CONFIG_GLOBAL` for that child process to a private,
short-lived config containing only the resolved exact checkout and disables
system-config lookup. The file is deleted when the subprocess scope closes.
Do not write the user's global Git configuration and do not change the
immutable vendor payload's ownership or mode. Continue to require the pinned
commit, clean tracked source, allowed runtime-data exceptions, required files,
and vendor-manifest hashes.

The BACE ComRecGC preflight now receives the same explicit upstream root as
generation and calls the same checkout validator before publishing `READY`.
Its controller fragment records that checkout as an input. The existing
`--official-root` CLI option and generic paired Slurm argument forwarding are
unchanged; only the Slurm comment is synchronized, and this AutoDL-only repair
does not submit any HPC job.

### Consequences

- Migrated ownership no longer causes a false execution failure.
- A missing, dirty, corrupt, or wrong-commit checkout fails on CPU before a GPU
  is allocated.
- Repository trust is limited to the exact resolved path and lifetime of each
  validation subprocess.
- Classifier, split, lineage, generation, and evaluation semantics are
  unchanged.

### Status

Accepted
## 2026-08-23: GlobalGCE gSpan may use exact stable-top-k anti-monotone pruning

- The running BACE GlobalGCE v5 route showed millions of frequent subgraphs in
  its first gSpan root while the official consumer only uses the stable top 20.
- We added an opt-in route that retains the exact stable top-k and prunes a DFS
  branch only when projected-support anti-monotonicity proves that neither the
  branch nor an equal-support later descendant can enter that top-k.
- The legacy all-pattern SQLite spill remains the default.  The optimized route
  has a distinct fingerprint/schema, atomic pre-root snapshots, whole-root
  replay after interruption, and a selected-payload hash audit.
- This decision does not authorize stopping or modifying the active v5 writer.
  A fresh route requires exact monolithic parity, crash-resume validation, an
  immutable execution commit, and a fresh output root.

## 2026-08-23: Close traversal-order and terminal-proof boundaries for exact gSpan

- The exact-route fingerprint is versioned again and binds graph-list order,
  node insertion order, and the exact NetworkX edge traversal order.  Sorting
  graph content is invalid because the pinned official stable top-k can change
  under equal-support traversal ties even when the unordered graph set is the
  same.
- A terminal exact audit may be published only after an atomic checkpoint says
  `stage=complete`.  The audit is the PASS-last proof and binds that checkpoint
  hash plus the selected SQLite payload identities.  Checkpoint or audit write
  failure leaves a resumable non-terminal state.
- Frozen-GINE BACE training persists the exact proof identity in its training
  summary, summary, run manifest, and completion gate.  All four publications
  are reopened, hash-checked, and the exact audit is recomputed before the
  outer `PASS`; deleted or modified proof bytes fail closed.
- The exact BACE CLI is only the GINE route
  `run_bace_baseline_gnn_route.py globalgce-train-rules`.  The historical
  RF-backed pool builder cannot enable exact mining and remains ineligible for
  the BACE paper cell.
- These changes do not authorize mutation or termination of the active v5
  output.  Any diagnostic or replacement begins from a fresh immutable root.
