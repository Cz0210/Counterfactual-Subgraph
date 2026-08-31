# Decisions Log

## [2026-09-01] Compare T12 native results by exact scientific content

### Motivation

The real three-process canary attempt `91b` completed both terminal workers
with 210 strict counterfactuals.  Its uninterrupted and resumed
`scientific_state` dictionaries are exactly equal, including the canonical
native-result SHA-256
`4c6d4df28e9435905bd22c95bb55abcc9f7e367b762f425024a24d8758da9f10`.
The two raw `torch.save` archives nevertheless have different SHA-256 values.
The prior verifier compared the archive bytes first and therefore conflated a
serialization representation difference with scientific divergence.

### Decision

Keep both raw archive hashes as evidence, but do not use their equality as the
scientific replay contract.  Gate replay on a recursive canonical native-result
snapshot that covers every mapping key/value, exact sequence and tensor element
order, tensor dtype and shape, and exact scalar values.  Mapping insertion order
is canonicalized by key because it is not scientific content.  No tolerance or
`allclose` is allowed.  If canonical hashes or any other scientific-state field
differ, fail closed.  If canonical state is exact and only raw archive bytes
differ, classify it explicitly as
`NON_SEMANTIC_SERIALIZATION_REPRESENTATION_ONLY`.

### Consequences

- The already committed `91b` observations can be independently re-verified;
  the observation schema is unchanged and the gate schema is bumped to v2.
- A gate records both raw hashes, their equality flag, the canonical contract
  and canonical digest.  It cannot silently discard the byte-level difference.
- This repairs the verifier only.  The receipt remains
  `production_released=false`, and no T12 cell or full-run PASS is implied.

## [2026-09-01] Bound T12 bridge history without weakening official full parameters

### Motivation

The exact three-process canary retains every complete collision payload,
embedding vector, coverage vector and lineage counter in one Python mapping.
That is intentionally auditable at 16 steps but cannot be extrapolated to the
pinned full route.  The full scientific parameters remain
`M=20000`, `sample_size=10000` and `candidate_capacity=100000`; only changing
the latter two to the canary's 128/512 values would be an unauthorized method
change.

### Decision

Add a Taste-T12-only production bridge mode.  Keep complete first-row records
only for the official live `graph_map`/ordered-candidate/current domain, whose
20k upper bound is `min(k, M+1)=20001`, plus at most one 10k neighbour batch as
transient state.  Every scored observation instead enters a fixed 272-byte,
append-only SHA-256 chain.  The compact record contains graph and semantic
digests, three probabilities, discrete outcome flags, covered-parent count and
a lineage digest; it contains no historical graph payload, embedding values,
coverage vector or lineage payload.  A bounded-page-cache SQLite index is
derived from the journal and rebuilt from the authenticated committed prefix
after restart.  It is never authority.  Re-entry of an evicted graph requires
exact embedding/coverage/failure digests and the reviewed probability envelope;
drift blocks rather than choosing a new first row.

Bind the resource proof and checkpoint identity to the official 10k/100k
parameters and 10k/20k cursors.  The worst scored-row count is
`1 + 20000*10000 = 200000001`; the journal prefix bound is 54,400,008,488
bytes under a 64-GiB cap.  With explicit 512-KiB deep/256-KiB serialized
per-live-row gates, the bridge RAM formula is 15,863,382,016 bytes under
16 GiB and its checkpoint component formula is 5,310,251,008 bytes under
8 GiB.  A complete checkpoint is hard-capped at 32 GiB before publication.
The narrow orchestrator permits only fresh 1--10k and resumed 10001--20k
segments and only 10k/20k manifests.

Do not claim that this bridge proof closes the whole official state.  At the
pinned parameters and 3,778 parents, the raw official transition dictionary
can reference 200,020,000 coverage rows: about 3.02 TB as dense float32 and
94,459,445,000 bytes even if coverage alone is bit-packed.  Production commit
therefore fails closed unless a dataset-specific external compact transition
store with bounded expanded LRU and exact checkpoint export is installed.
This remaining item must preserve sampled actions, importance, coverage,
ordering and RNG; lowering sample size or candidate capacity is not allowed.

### Consequences

- The all-history canary remains unchanged and continues to be the required
  exact A800 replay gate.
- Bridge history is reopenable across the 10k process boundary without
  retaining historical payloads in RAM or copying the journal into the model
  checkpoint.
- T12 full is still not launch-ready: a real A800 canary PASS, bounded official
  transition store, candidate persistence, calibration/test/export wiring and
  an independent terminal verifier remain required.
- No remote process, GPU task, calibration/test payload or matrix cell is
  touched by this implementation.

### Status

Implemented and focused-tested locally; production deliberately remains
fail-closed at the raw transition-state gate.

## [2026-09-01] Bind T12 threshold reuse to official source content, not checkout path

### Motivation

The calibration-only threshold selector was published from the clean
`ef89a09` execution worktree.  The T12 replay implementation lives in a later
clean worktree whose integrated official GCF directory has the same pinned
inventory.  Requiring those two absolute checkout paths to be identical made
the real A800 canary impossible: the old path failed the current integrated-
source hold, while the current path failed the selector's path comparison.

### Decision

Reopen both physical roots, but allow clean-worktree relocation when the
selector's official-source inventory SHA-256 exactly equals the inventory
derived from the currently held integrated source.  Model hashes, selector
receipt, calibration pair inventory, method semantics, threshold value, split
isolation, and every official source file digest remain mandatory.

### Consequences

This removes a non-scientific filesystem-path dependency only.  It does not
permit source drift, a hand-written threshold, test selection, symlinks, or a
missing old authority root.

---

## [2026-09-01] Bound the T8 zero-candidate recovery at 25 train-only epochs

### Motivation

Fresh deadline attempt `4376be2b-42de-46d4-a3c6-ad291dd3f9f0` completed both
native GlobalGCE branches and produced 4,860 raw generations per branch, but
none survived the existing parseability, chemistry, substructure, connectivity,
and strict-flip gates. Replaying the hard catalogs on the same 16 frozen train
parents also produced no valid candidate. Target 0 was rejected by native
adjacency/deletion connectivity constraints and target 2 by sanitization. The
failure is therefore not an empty generator, record-field mismatch, or
classifier/split/oracle error; the five-epoch decoder was under-trained for
this fixed cohort.

### Decision

Keep the historical smoke default at five epochs and permit exactly one
explicit zero-candidate recovery budget of 25 epochs, the minimum training
budget already accepted by the Taste GlobalGCE full configuration. The
deadline CLI requires both `--zero-candidate-recovery` and the canonical UUIDv4
of the failed source attempt, while the new attempt ID and state/output roots
remain fresh. The recovery records its source attempt, reason, and exact
science configuration in preflight, manifest, and gate evidence.

The recovery keeps the same 16-of-64 train-only cohort, seed 7, calibrated
three-class GINE, Sweet source label, independent targets 0 and 2, native
Top20/min-frequency/action semantics, checkpoint resume proof, and all strict
candidate filters. It does not load validation, calibration, or test, and it
does not repair, coerce, or relax invalid native graphs. Any epoch value other
than 5 or 25 fails before execution. If the fixed 25-epoch fresh attempt still
has no valid connected candidate, T8 remains a scientific failure; no further
automatic budget tier is authorized.

### Consequences

- The training-resource change is explicit and hash-visible, while the
  classifier, split, oracle, target branches, native action, and acceptance
  semantics remain unchanged.
- Existing five-epoch results cannot be relabeled or resumed as recovery.
- The measured single-GPU estimate is approximately eight to nine hours; the
  estimate is operational only and is not part of the acceptance gate.

### Status

Accepted

---

## [2026-09-01] Give T7 the stable attributed-graph identity already used by T12

### Motivation

Fresh T7 attempt `32126b5d-e288-4a14-8f3d-9c6382b22f51` reached the official
importance bridge and failed because two edited graphs shared one embedding
hash while their graph semantics differed. The vendored VRRW identity is
Python's process-local `hash(embedding.tobytes())`; invalid native graphs also
deliberately share a zero GINE row. Embedding bytes therefore cannot be a graph
identity even within one process.

### Decision

Apply only T12's already implemented stable-identity bridge to the bounded T7
dataset boundary. For every ordered importance batch, compute the SHA-256 of
the canonical parent-free attributed native graph and queue it beside the
corresponding embedding SHA. Patch official `calculate_hash` during the walk
so it consumes the structural identity in exact order and verifies the
embedding SHA only as a call-order assertion. Persist the stable identity
contract and counters through T7's existing eight-plus-eight checkpoint.

Allow one structural identity to reuse a previously scored row only under the
same T12 low-bit GINE envelope and exact discrete prediction, candidate,
validity, failure, coverage, and canonical collision evidence. Different
structural identities remain distinct even when their embedding bytes are
identical. Do not use Python built-in hash, raw embedding identity, parent
metadata, or lineage as the registry key.

Official mutation enumeration, VRRW movement/restart/frequency behavior, the
same calibrated three-class GINE, `1-p(Sweet)` importance, `argmax != Sweet`
candidate predicate, generated-query-to-original-target NeuroSED, and
full-graph semantics are unchanged.

### Consequences

- The failed attempt remains immutable and is not resumable under the new
  checkpoint identity contract; rerun requires fresh UUID and output roots.
- Equal embeddings from different edited graphs no longer alias their
  counterfactual semantics.
- A missing, extra, or reordered official hash request remains fail-closed.
- This implementation performs no deployment or remote science launch.

### Status

Implemented with focused local tests; fresh AutoDL deployment/run pending.

---

## [2026-09-01] Bound Taste T14 transition memory without changing the walk

### Motivation

The first fresh T14 attempt froze the authorized 3,778-parent train cohort but
disappeared before its first 2,500-step checkpoint after about five hours.  It
wrote no Python traceback or semantic failure.  The host cgroup reports OOM
kills and a historical high-water mark equal to its 480-GiB limit; that counter
is cumulative, so it is supporting rather than PID-specific evidence.  The
code audit found a deterministic defect regardless: unlike the established
full ComRecGC runtime, T14 retained every expanded PyG neighbour graph in every
transition and had no authoritative backing store for graph-map eviction.

### Decision

Use the existing exact full-walk state substrate in T14:

- retain transition hashes, exact enumerated actions, importance values and
  embeddings in `CompactMoveScopedTransitionMap`, with at most five expanded
  transition entries and deterministic action replay after LRU eviction;
- retain the official hot graph-map semantics while placing evicted graph
  payloads in the existing authoritative SQLite store;
- checkpoint the hot official state, compact transitions, bridge records,
  graph-state counters, complete CPU/CUDA RNG and an atomic SQLite snapshot at
  every existing 2,500-step boundary; and
- expose explicit fresh/resume launcher modes.  Resume requires a complete
  checkpoint and cannot adopt the checkpoint-free failed root.

Release the temporary all-Sweet cohort-selection adapter and graphs before the
full walk, and write RSS/cache diagnostics every 100 completed steps.  These
are lifetime and observability changes only.  The frozen cohort, official edit
enumeration and ordering, GINE scores, canonical attributed-graph identity,
lineage, `M=20,000`, one `25,000` fallback, candidate capacity 50,000, sample
size 10,000, and minimum ten valid unique rules are unchanged.

### Consequences

- Raw neighbour `Data` objects are bounded by a five-entry expanded LRU rather
  than growing with every visited transition; no model call, random draw,
  neighbour enumeration, candidate ordering or scientific parameter is added
  or removed.
- The old `c87c0a7a-...` root remains immutable and cannot resume because it
  has no complete checkpoint.  Production must start in a fresh root; after
  the first 2,500-step closure, the same immutable execution commit may resume
  that root with `TASTEMOLNET_T14_RESUME=1`.
- `TASTEMOLNET_T14_GPU_INDEX=0..3` selects an explicitly scheduled physical
  GPU (default `1`); UUID discovery and the existing exclusive GPU lock remain
  mandatory.  The selected index and UUID are checkpoint-bound, so any exact
  resume returns to the same physical device.
- Checkpoint validation now includes the compact transition state and SQLite
  snapshot.  Calibration and test remain unopened during generation.

### Status

Implemented and locally focused-tested; requires immutable AutoDL deployment
and a fresh production attempt.

---

## [2026-09-01] Authorize strict multi-component AIDS source no-op identity

### Motivation

The frozen AIDS cohort contains 1,283 source parents, of which 236 contain two
to four disconnected components.  The shared full-graph decoder correctly
rejects disconnected generated COMRECGC candidates, but the chemistry audit
also reused that generated-candidate connectivity gate for the source graph's
empty action.  A byte- and tensor-identical source therefore failed with
`generated_disconnected_or_empty` even though no component or edge had been
changed.  Stripping salts, excluding those parents, or relaxing generated
candidate connectivity would each change the frozen cohort or method
semantics.

### Decision

Accept `ALLOW_AIDS_MULTICOMPONENT_SOURCE_NOOP_IDENTITY_V1=1` only as an
AIDS source/no-op audit authority.  A multi-component source passes the empty
action only when all of the following independently close:

- canonical isomeric component multisets match the requested source, frozen
  graph-native spelling, reconstructed molecule, and record, with duplicate
  components retained as multiplicities;
- molecular and graph-topology component counts match;
- node-feature, edge-index, and edge-attribute tensors match the frozen record
  and remain exact across clone, save/load, and batch/unbatch;
- both COMRECGC and GCF node-lineage tensors remain the exact source identity;
- parent identity, node count, and stable graph SHA-256 remain exact across all
  four representations; and
- the unchanged shared generated-candidate decoder still rejects the same
  disconnected source shape specifically as `generated_disconnected_or_empty`.

Publish the per-parent closure in `noop_roundtrip.csv` and a separate
`aids_multicomponent_source_noop_identity.json` receipt.  Without the exact
environment authority, any multi-component source continues to fail closed.

### Consequences

- All 1,283 frozen parents remain in the cohort; no component is stripped,
  repaired, reordered, or excluded.
- The completed 91,916,686-pair exact DBSCAN result is adopted unchanged and
  is not regenerated by this source-only audit.
- No generated candidate obtains this exception.  The shared COMRECGC
  full-graph decoder and chemistry repair retain the single-component gate,
  and a connected but tensor-different candidate fails the source identity.
- Classifier, split, oracle, lineage, calibration, held-out test, and matrix
  publication semantics are unchanged.  A fresh postprocess root is required;
  the prior failed root remains immutable.

### Status

Authorized and implemented; awaiting a fresh CPU-only AutoDL postprocess.

---

## [2026-09-01] Authorize a stable attributed-graph identity for Taste T12

### Motivation

The T12 audit proved that official `hash(graph_embedding.tobytes())` cannot
survive a new isolated Python process.  The user has now explicitly authorized
the recommended deterministic implementation so the 20k native GCFExplainer
walk can use a real persistent checkpoint without changing classifier, split,
edit, score, coverage, or test-isolation semantics.

### Decision

For TasteMolNet T12 only, patch the official registry/hash callback at the
project boundary to return the existing canonical parent-free attributed-
graph SHA-256.  Keep a queued digest of each returned embedding solely to
prove official scorer/hash call order.  Do not use raw embedding bytes,
parent metadata, or Python built-in hash as identity.  Apply the same reviewed
first canonical GINE row and low-bit reuse envelope as Taste ComRecGC, while
rejecting any discrete prediction, validity, candidate, coverage, or graph
change.

Persist the complete official walk/bridge/adapter/action state and Python,
NumPy, Torch CPU and CUDA RNG.  Bind it to fresh attempt/generation identity,
20k budget with 10k/20k cursors, source cohort/split, GINE, NeuroSED and
threshold, official source, execution tree, runtime and GPU.  A bounded real-
GPU uninterrupted-versus-distinct-process exact replay gate is mandatory
before production; an `allclose` terminal comparison is insufficient.

### Consequences

- The stable identity/checkpoint/replay-gate substrate is implemented without
  modifying vendored official source or creating a generic controller.
- The dataset-specific producer now reconstructs the real held T7
  GINE/NeuroSED/VRRW route.  Its checkpoint prefix, resume suffix and
  uninterrupted reference must be three distinct Linux process instances in
  one A800 allocation; the independent gate rehashes the manifest and exact
  official mutable state.
- CUDA execution is pinned to deterministic-error mode with cuDNN benchmarking,
  TF32 and supported reduced-precision reductions disabled.  An unsupported
  deterministic PyG kernel is a real blocker, not grounds for `allclose`.
- The canary marker is not a T12 cell PASS and explicitly does not release
  production.
- The producer consumes the calibration-only T7 threshold selector artifact,
  its input authority, receipt and inventory; scalar/default thresholds are
  rejected.  The full runner still requires a real A800 PASS, lossless native
  candidate persistence, the shared WNode authority, calibration/test/export
  wiring, and independent verification.
- The bounded bridge intentionally retains complete replay rows.  It must not
  be scaled directly to 20k: production first needs a deterministic compact
  live-registry/history-hash layout and a measured RAM/checkpoint-size bound.

### Status

Identity, restart substrate and bounded real-route producer accepted; real A800
canary execution and production remain gated.

---

## [2026-09-01] Fit Taste distance thresholds on the frozen T4 calibration cohort

### Motivation

Taste T7 still required a typed NeuroSED distance threshold and the full
methods still required one method-shared WNode grid.  No reviewed authority
existed: the matrix placeholder explicitly said `MISSING_NOT_INFERRED`, while
the only numeric values in tests or upstream defaults were not fitted on this
dataset.  The frozen T3 split manifest reserves `calibration` for threshold
and selector work; the NeuroSED validation pairs already selected its model
checkpoint and therefore are not a substitute for that task split.

### Decision

Replay the independently published T4 terminal calibration cohort exactly:
the first 64 calibration Sweet parents predicted Sweet, at most 16 connected
one-/two-atom deletions per parent, 733 valid deletions, and all 38 strict
flips.  Measure every strict-flip residual as the generated query and its
parent as the original target.  Select the official normalized NeuroSED
threshold as the float64 linear q30 of those 38 values.  On the same pairs,
select the shared MolCLR node-Wasserstein q05/q10/q20/q30/q50/q70/q90 grid,
with theta=q30 and cost cap=q90.  Identical WNode quantiles merge into one
level, retaining the earliest lower-quantile name and summed weights.

The selector opens only the frozen calibration graph cache, verifies the T3,
T4, NeuroSED, GCF, and MolCLR inputs, and publishes receipts rather than a
paper method cell.  Train, validation, and test payloads remain unopened;
there is no upstream-default or test-derived fallback.

### Consequences

- T7 receives a generated-query-to-original-target threshold fitted to the
  same method-independent strict-flip calibration semantics used downstream.
- T11/T12/T13/T14 can consume the exact same WNode contract without
  method-specific tuning.
- One T4 replay produces both authorities, avoiding duplicate GINE work.
- The selector does not repair or relax the separate T12 cross-process VRRW
  blocker and does not publish any matrix result.

### Status

Implemented; requires one short AutoDL A800 selector run to materialize the
numeric authorities.

---

## [2026-09-01] Keep Taste T12 blocked on deterministic cross-process VRRW resume

### Motivation

The requested TasteMolNet T12 main-table route needs a real 10,000--20,000
step native GCFExplainer walk that can survive worker exit and resume in a new
process.  The existing T7 proof is intentionally narrower: it saves at step
eight, erases live state, and reloads for steps nine through sixteen while the
same Python interpreter, imported official modules, model instances, private
temporary directory, and held file descriptors remain alive.  Treating that
in-process proof as production restart would make a bounded smoke look like a
full experiment.

The vendored official implementation also defines graph identity as
`hash(graph_embedding.tobytes())`.  The production T7 chain starts workers
with Python isolated mode (`-I`), which ignores `PYTHONHASHSEED`.  A local
four-process probe with the same bytes and `PYTHONHASHSEED=7` produced four
different identities under `-I`, while four non-isolated processes produced
one common identity.  A restored `graph_map`, `graph_index_map`, candidate
list, transition map, traversed trace, and current cursor would therefore use
the previous process's integer namespace while newly evaluated neighbours use
the new process's namespace.  Reinforcement, frequency ordering, and walk
transitions can silently diverge.

### Decision

Do not add a T12 runner, CLI, Slurm wrapper, PASS marker, or placeholder.  Keep
T12 explicitly unimplemented until the native walk has a reviewed
cross-process identity and checkpoint contract.  In particular:

- do not change the official embedding-byte identity to a project hash without
  an explicit scientific-method decision;
- do not claim that setting `PYTHONHASHSEED` repairs the current isolated
  production command, and do not remove `-I` without a separate execution and
  reproducibility review;
- do not persist or reopen the T7 private checkpoint as-is.  Its holder has no
  `open_existing` path, relies on live descriptors and inode ancestry, lives in
  a `TemporaryDirectory`, states
  `checkpoint_payload_persisted_to_terminal_output=false`, and is deleted with
  the worker runtime;
- do not generalize the T7 schema by relabeling it.  Its validator is fixed to
  T7, an 8/16 step split, and one checkpoint; it has no 10k/20k cursor,
  interval, attempt, source-cohort, release, GPU/runtime, or official-source
  binding;
- do not promote T7 terminal evidence into a T12 candidate pool.  T7
  deliberately omits graph tensors, SMILES, and molecule payloads and records
  selector/full-route status as `NOT_EVALUATED` and paper eligibility as
  false.

A future implementation must first prove an exact uninterrupted-versus-
process-restarted trace on the real GPU path, including graph identities,
candidate frequencies/order, transitions, current cursor, bridge and scorer
state, Python/NumPy/Torch CPU and CUDA RNG, generated-to-original NeuroSED
coverage, and the official native result.  This proof must account for the
official raw GPU-embedding-byte identity; prior BACE replay work showed that
batch/process floating-point differences can change those bytes and therefore
the walk.  Only after that gate may a persistent 10k/20k checkpoint be used by
calibration ordering, held-out test evaluation, standardized export, and an
independent terminal verifier.  The NeuroSED distance threshold remains a
required external typed pin and must never receive a default.

### Consequences

- T7 remains a valid 16-step in-process checkpoint/reload smoke and no more.
- No classifier, split, NeuroSED, threshold, official VRRW, or graph-identity
  semantics are changed by this audit.
- No local or remote science is started, and no T12 result or matrix cell is
  claimed.
- T12 remains a main-table blocker rather than a mislabeled bounded run.

### Status

Blocked pending a reviewed deterministic cross-process VRRW identity and
checkpoint protocol.

---

## [2026-08-31] Preserve frozen AIDS graph node order across canonical SMILES fallback

### Motivation

The fresh AIDS ComRecGC v4 postprocess completed all 91,916,686 exact
common-recourse pairs, produced one DBSCAN cluster with zero noise, and then
failed its source chemistry round trip before calibration or test access.  The
codec paired the frozen graph one-hot rows with atom sidecars reparsed from the
source CSV spelling.  Thirteen of 1,283 parents required the already-authorized
unique canonical-SMILES fallback, and three of those equivalent spellings use a
different RDKit atom enumeration.

### Decision

For AIDS only, construct node-indexed atom and bond sidecars from the frozen
graph's own `smiles`, because that exact spelling generated its `x`,
`edge_index`, atom symbols, and bond order.  Preserve the requested source CSV
SMILES as provenance and independently require its canonical isomeric identity
to equal the graph-native molecule.  Record whether matching was exact or a
canonical fallback and name `frozen_graph_smiles` as the node-order authority.
Reject an invalid native spelling or any identity mismatch before decoding.

### Consequences

- Canonically equivalent source spellings no longer corrupt node-indexed
  sidecars merely by enumerating the same atoms in another order.
- The source/no-op round-trip remains strict; this change does not skip a row,
  reorder a graph, expand chemistry tolerance, or repair a non-equivalent
  molecule.
- Classifier, split, common-recourse, calibration, and held-out test semantics
  are unchanged.  The failed v4 root remains immutable and requires a fresh
  downstream-only postprocess attempt after deployment.
- A read-only full-cohort preflight exposed a separate next gate: 236 of 1,283
  frozen AIDS source graphs contain two to four disconnected components and
  fail the unchanged `generated_disconnected_or_empty` no-op rule.  This fix
  neither hides nor resolves that data-contract issue; salt handling,
  source-cohort exclusion, or multi-component full-graph eligibility requires
  an explicit scientific decision before another production attempt.

### Status

Accepted

---

## [2026-09-01] Accept hash-bound empty Python package sources in T8 provenance

### Motivation

The fresh T8 target-0 attempt
`ebb64920-9ca7-4122-b9c6-b7e4870c07c9` completed both bounded generation
chunks and produced a nonempty native-rule catalog, but the terminal startup
validator rejected its provenance.  Rehashing all 19 recorded module files
found no source mismatch.  The only structurally rejected rows were the pinned
official checkout's legitimate zero-byte `data/__init__.py`,
`models/__init__.py`, and `models/gSpan/__init__.py`, each correctly recorded
with SHA-256 of the empty byte string.

The startup document is create-once.  On the resumed generator call, a fresh
capture must produce the exact same canonical document or
`_write_canonical_identity_once` fails before science continues.  Therefore
lazy imports did not add a second unbound closure in this attempt; the failure
was the terminal validator's incorrect assumption that every regular Python
source must contain at least one byte.

### Decision

Permit a provenance row with `bytes == 0` only when its source SHA-256 is the
canonical SHA-256 of the empty byte string.  Continue to reject negative byte
counts, reject the empty-source hash for a positive-length file, and retain the
exact document hash, module path/root, inode/device, package version, official
commit, isolated-interpreter, and required-module checks.  Treat the top-level
`models` and `data` package rows as official-commit-bound entries as well.

Do not alter official GlobalGCE source, API signatures, classifier, split,
target branch, checkpoint seal, resume identity, or native-rule semantics.
The failed attempt remains non-adoptable; any rerun requires a fresh UUID and
root from a reviewed immutable successor.

### Consequences

- Empty pinned `__init__.py` files remain byte-for-byte identified instead of
  being mistaken for absent evidence.
- A zero-length row with any other digest fails closed.
- Source changes, module additions/removals between captures, and resume drift
  retain their existing rejection paths.

### Status

Accepted

---

## [2026-09-01] Remove only the artificial AIDS edge from Mut trace-off parity v2

### Motivation

The deployed Mut trace-off parity v1 controller froze
`mut_wait_aids_comrecgc_pass` into its manifest and task DAG.  That task has
failed terminally even though the Mut 500-step instrumentation gate, 50k
trace-disabled reference, and real trace parity are scientifically independent
of the AIDS paper cell.  Editing the frozen v1 manifest or manufacturing an
AIDS receipt would violate persistent-controller and parity provenance.

### Decision

Add an explicit v2 spec/controller identity with a fresh output root.  It
requires the recorded authority
`REMOVE_ARTIFICIAL_MUT_WAITS_FOR_AIDS_FINAL=1`, rejects any `aids_dependency`
field, and emits exactly seven tasks.  Only the two dependency lists change:
the instrumentation equivalence depends on the frozen trace-on source, and the
50k reference depends on that source plus the equivalence PASS.

The legacy algorithm worktree remains pinned to `7f7ed51`, the checkpointed
execution worktree remains pinned to `66487c0`, the prefix remains 500 steps,
and the reference remains a fresh seed-0, trace-disabled 50k exclusive-GPU
run.  The same normalized real parity assertion is still mandatory before
common-recourse adoption or held-out standardization.

### Consequences

- The existing v1 controller, manifest, task state, and output root remain
  immutable and are never reused by v2.
- The new builder cannot silently retain an AIDS dependency or accept copied,
  self-compared, or synthetic parity evidence.
- Deployment still requires a fresh immutable controller worktree, an
  exclusive idle GPU, and the unchanged 440-GiB cgroup-free-memory setting.
- This change builds and validates the route only; it does not launch AutoDL
  science.

### Status

Accepted and implemented; AutoDL deployment pending resource preflight.

## [2026-08-31] Compare T6 PEFT target modules by their native set semantics

### Motivation

The first T6 run to complete all five real PPO updates wrote byte-valid final
and periodic safetensors checkpoints, then failed reload because PEFT 0.18.1
serialized its in-memory `target_modules` set in a different JSON-list order.
The saved and in-memory authorities contained exactly the same five module
names; all other fields matched.

### Decision

At checkpoint reload, normalize only a nonempty unique-string
`target_modules` list to sorted set order on both sides before comparing the
complete adapter configuration.  Continue to hash-bind the raw
`adapter_config.json` bytes and all safetensors/value-head bytes.  Reject any
missing, added, duplicate, empty, or non-string module and report changed
fields instead of weakening the reload gate.

### Consequences

- PEFT hash-table iteration order is not misclassified as a scientific config
  change.
- The exact target-module membership and every other adapter field remain
  fail-closed.
- The failed five-step attempt remains non-adoptable; the fix requires a fresh
  UUID/root.

### Status

Accepted

---

## [2026-08-31] Harden Mut exact adoption and matrix publication fail-closed

### Motivation

The first read-only Mut exact continuation hash-closed its source artifacts but
did not pin the complete production DBSCAN/scientific parameter identity.  It
also accepted a compatibility parity reader, allowed source/output containment,
and exposed matrix files one at a time at the final pathname.  A crash could
therefore leave an ambiguous partial successor, while resume checked only a
small subset of the append receipt.

### Decision

Require the exact common-recourse parameters, 100 selected rows, and the full
sklearn-1.7.2 float64 multi-component DBSCAN contract (`eps=0.02`,
`min_samples=3`, brute Euclidean, four workers, no approximation or failure
cap).  Validate the terminal controller schema and require both recorded PIDs
absent.  Use the canonical Mut trace-on/off parity gate and bind its traced root
to the adopted generation.

Reject containment among either output and every frozen source authority, and
between the two outputs.  Publish the matrix successor only after complete
same-parent staging and verification, using an atomic no-replace directory
rename.  Resume must revalidate all non-target rows, the target and Mut/Ours
shared identity, both prior hashes, standardized inventory, writer evidence,
execution identity, and every non-recomputation/non-imputation flag.

### Consequences

- No pair, DBSCAN, or common-recourse stage is added or rerun.
- Exact production semantics and held-out-test boundaries are unchanged.
- A pre-publication crash leaves no final matrix path; a post-publication crash
  can adopt only a fully equivalent closed successor.
- The remaining external blocker is still a canonical
  `mut_trace_on_off_parity_v1` PASS receipt for the frozen generation.

### Status

Accepted

---

## [2026-08-31] Close the reviewed T6 GPU0 policy authority after relocation

### Motivation

The reviewed downstream policy was changed from physical GPU1 to GPU0 for the
authorized T6 release, but its fail-closed in-code expected value and raw-file
SHA remained pinned to the pre-relocation bytes.  The real T6 child therefore
exited before loading any model or creating its output root.

### Decision

Bind the loader to the already-reviewed tracked JSON exactly as committed:
T6 physical GPU index zero and raw SHA-256
`29bc6779af9b1d60784d76643f11a9d32213e04412818ce36ac4400ab2af46da`.
Do not alter the JSON, any model, split, reward, optimizer, or GPU UUID release
authority.  Retain exact-path, raw-hash, canonical-payload, and base-policy
verification.

### Consequences

- T6 can pass its pre-science policy gate on the authorized GPU0 route.
- Any byte or semantic drift from the tracked policy still fails closed.
- The failed attempt remains non-adoptable; a retry must use a fresh UUID/root.

### Status

Accepted

---
## [2026-08-31] Decode T8 affine edge scores at the native catalog boundary

### Motivation

Fresh T8 attempt `1ed059be-3f95-4334-80e4-3bb35d82ea4d` completed target-0
exact Top20 mining, the planned epoch checkpoint/reload, five training epochs,
and both eight-parent generation chunks.  Its terminal catalog path existed
but was empty: twelve learned rules retained an ambiguous node maximum and the
other eight were rejected only because the catalog validator treated the
pinned official edge decoder's unrestricted affine scores as probabilities in
`[0,1]`.  The latter contradicts the already accepted official hard-codec
contract, which decodes those finite scores with row-wise `argmax`.

### Decision

Keep the official model, training, raw rule checkpoint, continuous generation,
and attachment-aware action unchanged.  When materializing the private hard
native-rule catalog, decode only `edge_attrs_reconst` with the pinned official
row-wise `argmax` into one-hot edge labels and record the named decode contract.
Continue to reject non-finite/malformed edge scores, ambiguous node labels,
invalid adjacency, disconnected LHS rules, and every other existing native
rule failure.

Top20 remains the mined and optimized smoke surface.  The bounded T8 smoke
requires a nonempty hard-valid catalog from each target branch, matching its
pre-registered at-least-one-rule/strict-flip purpose; it does not require all
twenty learned slots to hard-validate.  An absent or empty catalog is accepted
at neither terminal branch capture nor independent verification.  The catalog
is required only after the deliberate resume reaches the generator completion
boundary, not at the epoch-zero planned interruption.

### Consequences

- The preserved failed attempt remains non-adoptable and no state file is
  modified; a retry must use a fresh UUID, state root, scratch root, and final
  root.
- The observed target-0 checkpoint yields eight hard-valid unique rules under
  the already frozen edge `argmax` semantics while the twelve node-tie rules
  remain rejected.
- Empty catalogs, silent clamping, node-tie coercion, RF/binary fallback, and
  a lowered Top20 training surface remain forbidden.

### Status

Accepted for the narrow T8 fresh-retry repair.

## [2026-08-31] Bind omitted AIDS freeze-only RNG evidence to its recovery receipt

### Motivation

The completed AIDS exact postprocess successfully adopted all 91,916,686
distance pairs and produced the one-cluster, zero-noise common-recourse result.
Its chemistry stage then failed before candidate replay because the historical
freeze-only lineage summary omitted `rng_calls_added`. The producer's sibling
v4 recovery audit already proves that the 50,000-step walk was complete and
that freeze-only reconstruction performed no proposal or RNG call, but the
chemistry validator previously recognized only the redundant summary field.

### Decision

Keep explicit `rng_calls_added=0` as the normal contract and add it to future
freeze-only summaries. For the preserved omission, permit one dataset-scoped
compatibility path only when the exact
`authoritative_backing_freeze_only_v3` summary reopens all of the following:
the v4 `FREEZE_ONLY_RECOVERY_SAFE` receipt and its PASS terminal hash, completed
walk/no-RNG reason, all-true recovery checks, candidate/resolved counts,
lineage and selected-trace marker hashes, frozen-payload v7 no-drift closure,
zero replay/fallback errors, and the matching completed run manifest. Record
the recovery and terminal hashes in chemistry and preregistration outputs so
the compatibility is never silent.

Do not accept a generic missing field, synthesize or modify the historical
trace summary, rerun generation, rerun DBSCAN, or reinterpret trace integrity
as trace-on/off parity.

### Consequences

- The existing immutable AIDS lineage and completed exact common-recourse
  artifacts are sufficient for a fresh downstream-only postprocess attempt.
- Any missing, mutated, symlinked, unhashed, incomplete, or non-freeze-only
  evidence still fails closed.
- Future freeze-only artifacts satisfy the ordinary explicit-zero path.

### Status

Accepted for the minimal AIDS postprocess compatibility repair.

---

## [2026-08-31] Close T13 over the real T8 and all resumable science bytes

### Motivation

The first T13 integration accepted a legacy flat `PASS` shape although the
real T8 is a managed-execution-v2 terminal with its typed T8 verification
nested in `verification.json`. It also trusted completed branch and
per-parent checkpoint paths more narrowly than the independent terminal
verifier could replay, and applied the ten-rule paper gate to each target
branch instead of to their canonical merged rule set.

### Decision

Require T13 to descriptor-reopen the published managed-v2 T8 directory,
validate the nested typed T8 receipt, and cross-bind its frozen three-class
GINE checkpoint. Hash-close every target-branch file, including official
model/rule/epoch checkpoints, and independently parse the typed training
resume identity. Bind every adopted evaluation chunk to the complete split,
parent, ordered-rule, GINE, MolCLR, threshold, distance, and counterfactual
configuration. Freeze every raw file, then have the separate verifier re-hash
the exact raw closure and replay branch validation, canonical merge/dedup,
calibration-only selection, chunk reconstruction, and standardized exports.

Allow either target branch to contribute fewer than ten native rules. Require
at least ten unique rules only after the two branches are canonically merged;
do not copy rules or weaken the final paper gate.

### Consequences

- The real managed-v2 T8 publication is consumable without a legacy adapter.
- A resumed chunk or branch can be adopted only when its complete content and
  science configuration still match.
- The terminal verifier no longer passes placeholder branch, merge, selector,
  or checkpoint evidence.
- Matrix publication remains an explicit downstream registry/publisher step;
  T13 publishes only its independently verified cell artifacts.

### Status

Accepted

---

## [2026-08-31] Bound transient procfs permission races during frozen-source adoption

### Motivation

The fresh AIDS ComRecGC postprocess verified the complete exact source, then
failed while globally auditing open descriptors because a short-lived,
unrelated process temporarily returned `EACCES` for `/proc/<pid>/fd/0` and
exited.  Treating this exec-time procfs race as a source writer is not evidence
of mutation, but ignoring a persistent unreadable descriptor would weaken the
read-only adoption gate.

### Decision

Retry only the exact unreadable descriptor for a bounded 0.75 seconds.  Reopen
and compare its target device/inode before accepting it.  If the descriptor
closes, record the resolved permission race; if it becomes readable, continue
the existing source-path/inode and writable-flags checks; if it remains
unreadable or unstable, fail closed.  Keep the source snapshots and SHA checks
before and after adoption unchanged.

### Consequences

- A process that has exited or closed the descriptor cannot spuriously abort
  AIDS postprocess.
- A persistent inaccessible FD, a writable source FD, or an inode-changing FD
  still prevents adoption.
- No DBSCAN, generation, classifier, split, threshold, or evaluation semantics
  change.

### Status

Accepted

---

## [2026-08-31] Release the dataset-specific TasteMolNet T11 Ours full route

### Motivation

The T6 route proves that the three-class frozen-GINE reward, decoded-chem PPO,
policy update, frozen reference, and adapter reload work together, but it is
intentionally bounded to 5--10 updates and 8--16 parents.  It cannot be renamed
or wrapped as a paper cell: it has neither a full optimization schedule nor the
base/high-temperature pool, calibration selector, held-out test, standardized
exports, or restartable downstream evaluation required by T11.

### Decision

Reuse the already-audited stable decoded-chem PPO primitive for one additional
300-update train-only Taste run, starting from the independently validated T6
adapter and holding that T6 adapter fixed as the reference.  Save complete
policy/optimizer/RNG/observer/candidate-history state every 50 updates and
permit resume only into a fresh root with the exact T6, GINE, train CSV,
parent-cohort, and optimizer/generation contract.

Add a dataset-specific T11 downstream runner rather than a generic controller.
Generate four candidates per frozen-GINE-predicted Sweet train parent at the
preregistered base and high-temperature settings, with a deterministic
per-parent RNG stream and per-parent restart chunks.  Canonicalize and dedupe
fragments, evaluate their connected hard deletions on calibration with the
same three-class GINE and MolCLR-Node-Wasserstein, greedily freeze 10--20
ordered rules using calibration only, and open held-out test only after that
freeze is fsynced.  Export Figure 3, Figure 4, Table 2 K=10, prefix, parent,
destination, oracle, evaluation, and raw pair evidence.  The science process
may publish only `SEALED`; a distinct invocation replays every metric and
atomically publishes a fresh terminal root.

The shared Taste WNode threshold remains an external frozen experiment input.
The runner requires its existing calibration/frozen-protocol JSON and refuses
missing, test-fitted, or hash-inconsistent values; it does not infer a threshold
from any held-out output.

### Consequences

- T11 performs real full PPO and full candidate generation instead of adopting
  T6 smoke candidates.
- PPO and downstream parent chunks have usable, identity-bound restart paths.
- Policy training and both generation modes remain train-only; calibration is
  selector-only and test access is ordered after the durable freeze.
- Final output is directly checked by the ordinary 4-by-4 registry contract.
- AutoDL deployment still needs the real shared Taste threshold-contract path,
  MolCLR paths/caches, frozen splits, and fresh PPO/science/final roots; local
  code/tests are not a scientific PASS.

### Status

Accepted

---

## [2026-08-31] Close T11 threshold, resume, and terminal evidence

### Motivation

The first dataset-specific T11 implementation correctly ran the full PPO and
kept selection before test, but three evidence gaps remained. Its local
threshold parser could accept a manifest that the shared-threshold contract
would reject and then emit safe-looking leakage fields. Existing generation
and pair chunks were checked only by a few metadata fields, so a finite forged
WNode value or internally consistent probability row could survive resume.
Finally, the terminal freeze omitted the base/high chunk closure and did not
reconstruct their merge or require complete held-out counts and hashes.

### Decision

Load the threshold through `load_shared_frozen_thresholds` and additionally
require the Taste calibration/frozen-protocol source and explicit
`test_used_for_selection=false`. The accepted authority must say
`threshold_fitted_on_test=false`, `shared_across_methods=true`, and
`cf_mode=strict_flip`; the T11 manifests may report those facts only after the
loader has established them.

Give every generation and pair chunk a typed receipt binding its byte hash,
closed row schema, parent/candidate inventory, split, model identities, and
evaluation configuration. On resume, recompute generation rows with the same
frozen GINE and recompute pair rows with the same GINE plus MolCLR WNode path;
adopt only byte-identical replay. At sealing, freeze all aggregate manifests
and every chunk/receipt hash. The separate terminal verifier must reopen the
frozen train/calibration/test authorities, reconstruct both generation pools,
replay canonical merge and calibration selection, reconstruct both pair
matrices, validate the full held-out test manifest, recompute paper metrics,
and carry the exact science freeze into the fresh PASS root.

### Consequences

- T11 cannot turn a method-specific or test-fitted threshold into apparently
  shared, test-independent output.
- Resume remains per-parent, while scientific probabilities, WNode values,
  action semantics, and raw-to-canonical identities are fail-closed.
- A self-rehashed aggregate, merge, or test manifest is insufficient without
  replay against the frozen chunk and split evidence.
- These are dataset-specific T11 checks; no new controller or generic trust
  framework is introduced, and real AutoDL science is still required.

### Status

Accepted

---

## [2026-08-31] Add the executable TasteMolNet T13 native GlobalGCE full route

### Motivation

The T8 implementation proves the two-target multiclass bridge and one planned
checkpoint/reload, but its fixed 16-parent, five-epoch bounds intentionally
cannot produce the paper main-table cell. Wrapping that smoke in a full-stage
name would leave calibration, held-out test evaluation, standardized export,
and process-restart behavior absent.

### Decision

Add one dataset-specific T13 runner that uses the existing official GlobalGCE
generator for independent Sweet-to-Bitter and Sweet-to-Tasteless train-only
branches. Use its existing epoch checkpoint/resume implementation, merge and
deduplicate native rules by tensor-content identity, order at most twenty rules
on calibration only, freeze that order before opening held-out test, and
evaluate native LHS-to-RHS applications with the same three-class GINE and
MolCLR-Node-Wasserstein. Require one explicit frozen TasteMolNet threshold
contract shared by the four methods. Persist per-parent evaluation chunks for
restart, export the complete registry schema, and let a separate verifier
process replay the frozen metrics and publish PASS.

Do not add T11/T12 wrappers that merely call their currently bounded smoke
loops; those require real full-scale generation primitives before they can be
truthfully released.

### Consequences

- T8 PASS can release a real T13 science process without waiting for T10.
- Official GlobalGCE model/optimizer/scheduler/RNG checkpoints and the T13
  stage checkpoint are both usable after interruption.
- Test bytes remain unopened until calibration selection is durably frozen.
- The output is directly auditable by the existing 4-by-4 registry and reports
  destinations 0 and 2 without binary projection or RF fallback.
- T11 and T12 remain explicit implementation blockers rather than mislabeled
  smoke results.

### Status

Accepted

---

## [2026-08-31] Release Taste T7 from adopted NeuroSED and managed T3 pins

### Motivation

The fixed-budget NeuroSED source and its managed adoption already passed, and
the calibrated three-class Taste GINE is already published by T3. Retraining
NeuroSED or waiting for the legacy static T7 authority would change or block
the authorized deadline route without adding scientific evidence.

### Decision

Add one dataset-specific `TasteGCFReleasePinsV1` successor. A CPU candidate
writer derives the exact official-GCF/NeuroSED/T3/model/temperature/dataset and
four-split hashes from retained real files. A separate invocation reopens all
sources and alone publishes `[TASTE_T7_TYPED_RELEASE_PASS]`. The runtime then
executes the existing native 16-step full-graph VRRW smoke on physical GPU0;
its worker can publish only aggregate SEALED evidence, and a separate verifier
reopens the typed release before atomically publishing the generic managed-v2
final with the T7 domain marker.

The NeuroSED distance threshold is retained as runtime source authority, not
as an extra field in the prescribed typed-pins schema. No NeuroSED training,
validation/calibration/test payload load, RF oracle, or generic controller
framework is introduced.

### Consequences

- The legacy `run_tastemolnet_gcf_smoke.sh` remains disabled and historical;
  production uses the fresh `run_tastemolnet_gcf_smoke_v2.sh` successor.
- T7 fails closed if the adopted NeuroSED, T3 GINE, temperature, any split,
  checkout, official source inventory, GPU0 UUID, or generated-to-original
  direction changes.
- Local code and tests do not constitute either T7 marker; deployment still
  requires real AutoDL roots, the frozen distance threshold, and fresh paths.

### Status

Accepted and implemented locally; not deployed.

---

## [2026-08-31] Adopt completed AIDS exact DBSCAN read-only for postprocessing

### Motivation

AIDS has completed all 91,916,686 exact rows. Recomputing DBSCAN or its pair
store would waste the dominant runtime and create an unnecessary second
science identity; only streamed summary and standardized evaluation remain.

### Decision

Add one AIDS-only adoption interface. It reopens the typed controller-bound
exact terminal and requires exact vector/hash/contract/component closure,
all-core, zero-noise, and sklearn-float64 evidence. A fresh root writes only a
small adoption manifest and the existing streamed summary, chemistry, WNode,
Figure 3/4, Table 2, gate, and freeze outputs. It cannot create a fresh DBSCAN
tree, regenerate the pair store, use CUDA, exceed eight CPU workers, or refresh
the matrix before scientific PASS.

### Consequences

- Any source identity drift fails without fallback recomputation.
- Same-root resume retains the existing frozen stage/checkpoint contract.
- Local implementation and tests are not an AIDS cell PASS or deployment;
  AutoDL launch waits for normal operator ownership of the execution checkout.

### Status

Accepted for implementation; not deployed.

---

## [2026-08-31] Adopt the managed T5 generic base directly for Taste T6

### Background

The verified T5 result is an unchanged generic ChemLLM base with zero optimizer
steps and no Taste split or adapter. The older T6 candidate instead required a
never-released T5 zero-step adapter root, so treating the managed base as that
root would fabricate predecessor semantics.

### Decision

T6 holds the published managed-v2 T5 final and its complete source inventory,
while a separately pinned external authority carries the exact common
T2/T3/T4 frozen-oracle identity. T6 deterministically materializes matching
rank-8 zero-step LoRAs in memory for policy and reference; it does not train or
copy the base and does not relabel the T5 adoption as SFT. The policy must
change after at least five real PPO steps and the reference must remain
unchanged.

After the trainer publishes its strict terminal root, a separate process calls
the existing held T6 consumer and atomically publishes a fresh verifier
receipt. The managed task adopts that verifier receipt, not a path-only PASS
check. No validation, calibration payload, test, RF, or GNN ablation is opened.

### Consequences

- T5's real `ADOPTED_CLEAN_GENERIC_BASE` semantics remain truthful.
- T6 retains the existing three-class GINE reward and stable decoded-chem PPO
  implementation instead of adding a training framework.
- Release still requires exact immutable predecessor, checkout, controller,
  GPU, output, and storage pins plus a fresh attempt UUID.

### Status

Accepted for the deadline T6 dataset-specific release.

## [2026-08-31] Bind Taste T6 directly to the real managed-v2 T2/T3/T4 receipts

### Background

The first deadline T6 integration still reopened legacy held-stage roots, while
the real Taste predecessors are the T2 receipt bundle, the published managed-v2
T3 root, and the managed-final T4 root. That mismatch leaves T6 unable to hold
the exact frozen checkpoint and oracle evidence used by production Taste runs.

### Decision

Expose one dataset-specific managed-v2 binding entrypoint from the existing T5
release module and make T6 consume that retained authority directly. The held
binding remains exact to the real seven-file `HeldT2Receipt`, `HeldPublishedT3`, and
`hold_t4_managed_final` contracts, preserves the same checkpoint payload reads,
and keeps the wrapper frozen to physical GPU0. No generic controller or new
cross-dataset abstraction is introduced.

The current-campaign T2 receipt is not the historical five-file downstream
receipt and has no `manifest.json`. T6 therefore binds its `gate.json`, source
evidence, complete receipt inventory, source model, feature-schema file, and
split-manifest file through `HeldT2Receipt`; silently substituting the legacy
`hold_t2_gine_pass_adoption` schema is rejected. The managed T5 adoption is
likewise registered through its published `verification.json`, because that
adoption truthfully has no legacy `output_hashes.json`.

The same managed receipt identity is carried through the runtime input closure,
the smoke gate, and the independent terminal-output validator. In particular,
the current receipt-inventory digest replaces the absent legacy receipt-file
digest at every one of those boundaries; legacy artifacts remain readable only
through their explicitly tagged historical schema.

### Consequences

- T6 now revalidates the real managed-v2 predecessor layout instead of a stale
  legacy stage-output schema.
- The frozen GINE checkpoint payloads remain descriptor-held through the same
  no-test, no-RF, three-class Taste contract.
- Execution still remains release-gated until fresh external pins are filled
  and a clean AutoDL checkout launches the science run.

### Status

Accepted for local integration and focused verification.

## [2026-08-31] Advance the matrix only by strict authority append

### Background

The shared-threshold BACE GCFExplainer selection, held-out shards, merge,
freeze, and standardization completed successfully.  A separate fresh audit
scanned only that cell, however, and truthfully reported one passing cell; it
did not inherit the already frozen seven-cell authority and therefore cannot
be called the eight-cell matrix.

### Decision

Add one dataset-specific CPU-only append command.  It reopens the complete
hash closure of the existing 7/16 authority, re-audits exactly its seven
passing standardized roots plus the new BACE GCFExplainer standardized root,
and requires all fifteen non-target rows to remain identical.  The new BACE
cell must be `FROZEN_PASS`, retain K=20/Table-2 K=10, and match frozen BACE Ours
on dataset, split, GINE, MolCLR, distance, strict-flip, and shared-threshold
identities.  No broad scan, scientific recomputation, candidate reordering, or
raw-test access is allowed.

The successor is a fresh append-only root.  Its combined audit hash-closes an
append receipt and explicit supersession references before publishing
`matrix_status.json` last.  Superseded incomplete snapshots are referenced by
their exact matrix/combined hashes and observed cell count; their directories
and bytes are never edited or deleted.  If a historical top-level snapshot
predates combined-audit closure, the successor records that absence plus the
physical `matrix_status.json` identity instead of fabricating a combined hash.

### Consequences

- A one-cell or zero-cell re-audit cannot replace the seven-cell authority.
- Any predecessor drift or BACE cross-method identity conflict blocks before
  the fresh output root is published.
- A successful run advances exactly 7/16 to 8/16 without rerunning science.

### Status

Accepted for immediate AutoDL deployment.

## [2026-08-31] Bind BACE GCFExplainer to the frozen Ours B12 threshold grid

### Background

The completed BACE GCFExplainer calibration shards were scientifically valid,
but the native selector re-derived its seven WNode thresholds from the GCF
calibration matrix.  That produced threshold identity `b4aaf264...`, while the
already-frozen BACE Ours cell and the paper matrix use the method-shared B12
numeric grid identity `37d7a265...`.  The registry therefore correctly rejected
the otherwise standardized GCF cell as a cross-method threshold conflict.

### Decision

The GCFExplainer `select` stage must receive an explicit `--thresholds-json`
pointing to the immutable `bace/ours/b12-selector` output.  Before selection it
re-opens and hash-validates that file, its sibling frozen-selection manifest,
and the referenced Ours B11 calibration manifest.  The source must prove the
frozen BACE GINE and MolCLR identities, calibration-only fitting,
`test_loaded=false`, `test_used=false`, and numeric threshold hash
`37d7a265ee53fc0c31edaf59f8b412f41c79c62af4941d4ddf1f3e66c4afa427`.

The selector adopts the frozen bundle exactly; it does not re-fit thresholds,
open test data, regenerate candidates, or alter the calibration candidate
universe.  Its output records immutable source identities and is independently
reload-auditable.  Because the shared grid changes the GCF selection objective,
the prior GCF test/freeze/standardized roots are not adopted: selection, held-out
test evaluation, freeze, and standardization require fresh output roots.

### Consequences

- Ours and GCFExplainer now share exactly one BACE threshold grid.
- Missing, copied-look-alike, mutated, test-derived, wrong-GINE, or wrong-MolCLR
  threshold sources fail closed before selector output is created.
- Existing calibration shards remain reusable because neither their pair
  matrix nor candidate order is modified.
- This is a local code repair only; it does not deploy or start an experiment.

### Status

Accepted

---

## [2026-08-31] Bound TasteMolNet ComRecGC full generation to a frozen train cohort

### Motivation

The completed T9 smoke proves the native three-class bridge, but its 64-row
cohort and 500-step budget are not a paper result.  The deadline protocol now
fixes both the full cohort and a disclosed resource cap, so the full route no
longer needs a new framework or any held-out-data decision.

### Decision

T14 selects every training row with true label 1 whose frozen T3 calibrated
GINE prediction is also 1.  It applies no canonical-graph deduplication and
orders rows by molecule identifier and canonical attributed-graph hash before
writing a SHA-bound cohort manifest.  Generation uses only that train cohort,
the verified T9 bridge, and the official native ComRecGC implementation.

Write private reloadable checkpoints every 2,500 steps.  At 20,000, stop and
post-process when at least ten canonical valid unique, lineage-closed strict
train-side flips exist.  Otherwise extend once to 25,000; fewer than ten there
is a scientific failure.  Validation, calibration, and test remain unopened
during generation and the resource decision.

### Consequences

- The result discloses configured 20k, fallback 25k, effective M, cap use, and
  stop reason.
- Resume restores official collector, transition, bridge, loop, and RNG state;
  it cannot change the cohort.
- An independent terminal reopen validates cohort hashes, the effective
  checkpoint, valid-rule count, resource receipt, and held-out isolation.

### Status

Accepted

---

## [2026-08-31] Restrict fast-16of16-v2 to eight concrete stage observations

### Motivation

The old deadline heartbeat is stale and the Taste controller exited, while the
remaining science already has dataset-specific launchers and exact process
identities.  A new scheduler platform would add no scientific capability.

### Decision

Fast-v2 accepts exactly AIDS postprocess, Mut exact, BACE ComRecGC, BACE
GlobalGCE, and Taste T6/T7/T8/T14 bindings.  It validates exact PID generation
and command tokens when present, observes only caller-pinned progress and
terminal files, and writes a 60-second heartbeat.  It cannot launch or signal
science, publish the matrix, change a root, or enable GNN ablation.

### Consequences

- Existing and independently launched science remains owned by its dataset
  runner; fast-v2 supplies one restartable status authority.
- Missing future PIDs remain visibly `QUEUED` rather than being inferred from
  fuzzy process matching.
- Stage launch and any graceful stop remain separate, explicitly authorized
  actions.

### Status

Accepted

## [2026-08-31] Losslessly adopt the fixed-budget NeuroSED PASS for T7

### Motivation

The independently verified fixed-budget Taste NeuroSED result publishes its
scientific files at the root, while T7 consumes a generic managed-v2 final with
the same files under `artifacts/`. The legacy T7 consumer names split graph-ID
hashes that are not the fixed-budget pair authority.

### Decision

Add one dataset-specific adoption boundary. Its worker copies the complete
fixed-budget PASS tree byte-for-byte into managed-v2 `artifacts/`. A separate
verifier reopens and hashes the source and copy, validates the scientific PASS,
checkpoint, feature schema, selector, generated-query/original-target
direction, split isolation, and sampler/label bindings, then invokes the
existing atomic publisher. Consumer-v2 records the real sampler/label hashes;
it does not retrain, rewrite a model, invent a controller receipt, or change
GCF, GINE, split, or oracle semantics.

### Consequences

- The original fixed-budget root remains unchanged.
- T7 can consume `artifacts/best.pt` without a lossy conversion.
- A real managed attempt and a separately pinned typed T7 release remain
  mandatory; this implementation is not an execution release.

### Status

Accepted for implementation; not deployed or released.

---

## [2026-08-31] Separate Taste GlobalGCE live mining scratch from durable proof

### Background

The fresh T8 route reached real exact-top-k gSpan mining, then stopped at the
unchanged storage guard because `/autodl-fs/data` had fewer than 100,000 free
inodes.  `/root/autodl-tmp` has ample inodes but only about 18 GiB free, so it
also fails the existing 50 GiB free-space floor.  Lowering either guard would
hide the resource condition, while storing terminal proof only on tmpfs would
make the branch unreloadable after process exit.

### Decision

Give exact-top-k mining one explicit, science-neutral scratch root.  Keep the
live SQLite database and its WAL/SHM companions there and apply the unchanged
50 GiB, 2 percent, and 100,000-inode guard to that filesystem.  For the current
AutoDL host, a fresh attempt-specific `/dev/shm/fast16-t8-<UUID>` root satisfies
those floors; `/root/autodl-tmp` does not.

Before exact mining can return, create a transactionally consistent SQLite
backup in a temporary file on the persistent checkpoint filesystem, fsync it,
and atomically replace the durable database name.  Then publish the complete
checkpoint and heartbeat, and publish the exact-top-k audit last.  Exact proof
identities written through a held `/proc/self/fd` directory carry the verified
stable named persistent path, never the procfs or tmpfs spelling.  A later
reload validates and consumes the durable audit/database directly without
requiring the live scratch root.

The deadline T8 runner requires an absolute fresh `--gspan-scratch-root` and
gives targets 0 and 2 disjoint children.  It records only that external scratch
was used and durable proof was published; terminal manifests contain no
scratch path.  The failed b466 attempt had zero completed roots and is not
adopted or resumed; a deployment must use a fresh UUID/state/output/scratch
namespace.

### Consequences

- The official GlobalGCE traversal, stable Top20 ordering, train split, frozen
  three-class GINE, target branches, and guard thresholds do not change.
- Loss of `/dev/shm` after durable proof publication cannot invalidate branch
  reload or terminal verification.
- A storage stop before durable proof remains non-PASS and may retain only
  scratch-local partial progress.
- This change performs no deployment, cleanup, GPU launch, or scientific
  adoption.

### Status

Accepted

---

## [2026-08-31] Canonicalize the AIDS continuation interpreter path

### Background

The adopted AIDS checkpoint and all frozen scientific inputs matched, but the
resume launcher named the same Python executable through its `python` symlink
instead of the frozen run's physical `python3.10` path.  Because argv hashes
are part of the continuation contract, this lexical-only difference rejected
all five otherwise identical stages.

### Decision

Resolve `sys.executable` to its strict physical path before constructing and
hashing continuation stage argv.  Continue comparing the complete argv and all
existing checkpoint, input, schema, and project hashes without exemptions.

### Consequences

- Symlink and physical spellings of the same interpreter yield one frozen argv
  contract.
- Different interpreters or any scientific argument still fail closed.
- The verified checkpoint remains adopted at offset 20,512,768; no scientific
  payload is rewritten by this launcher-only repair.

### Status

Accepted

---

## [2026-08-31] Keep invalid Taste ComRecGC graphs outside the GINE model-graph channel

### Motivation

The Taste ComRecGC native adapter can reject a generated graph before frozen
GINE inference.  In that case it deliberately returns an aligned
`model_graph_payload=None`, `valid_fullgraph=false`, the fixed source-class
probability row, and a zero embedding.  The bridge previously conflated this
explicit unscored value with an older adapter that exposes no model-payload
channel.  It therefore substituted node-order-sensitive native tensors as if
they had been sent to GINE.  Two permutations of the same canonical invalid
graph then appeared to change model semantics even though neither was scored.

### Decision

Represent only the explicit `valid_fullgraph=false` plus aligned
`model_graph_payload=None` case with a fixed
`tastemolnet_gine_invalid_unscored_graph_v1` sentinel.  A valid graph with an
explicitly missing model payload is an error, as is an invalid graph carrying
GINE model evidence.  Adapters that do not expose a model-payload channel keep
the existing ordered native-tensor authority.

Do not change the valid-graph model payload hash, probability, prediction,
candidate, embedding dtype/shape, or numerical agreement gates.

### Consequences

- Node permutations of the same invalid, unscored graph share one stable
  parent-free identity without claiming that GINE evaluated either graph.
- Every valid generated graph still carries the complete lossless model input
  and must reproduce frozen-GINE semantics.
- Midpoint checkpoint/reload preserves the sentinel exactly; no classifier,
  split, oracle, or counterfactual decision changes.

### Status

Accepted

---

## [2026-08-31] Keep ComRecGC exception cleanup bounded

### Background

The BACE generation route committed a complete, resume-safe step-18000
checkpoint and then raised its storage guard because the shared filesystem had
fewer than 100,000 free inodes.  Exception unwinding nevertheless invoked the
full live-graph integrity audit, which reconstructs historical compact
transition targets and can consume a CPU indefinitely after science has
already stopped.

### Decision

Run the full live-graph audit only after normal completion.  On an exception,
record constant-time runtime diagnostics and explicitly mark the integrity
audit incomplete.  Checkpoint validation and normal-completion auditing remain
unchanged; a resumed attempt must still pass the full terminal audit.

### Consequences

- Fail-closed storage stops can terminate promptly and preserve their real
  error instead of appearing to be slow science.
- The committed checkpoint and scientific state are not changed or adopted by
  this cleanup-only fix.
- No failed run can be published from the abbreviated exception diagnostic.

### Status

Accepted

---

## [2026-08-31] Align the fixed-budget NeuroSED writer with its pair contract

### Background

The first real 5000/1000 NeuroSED run completed 1214 epochs and the official
batch-interleaved selector stopped normally, but final publication failed
before writing `model_card.json`.  The validator required nine explicit pair
and resource-budget declarations that its production writer omitted, even
though the authenticated pair manifests and training path already satisfied
all nine declarations.

### Decision

Write those existing facts into the production model card: the fixed-budget
extension is documented; GREED independent query/target role semantics are
preserved without claiming its sampler is byte-for-byte unchanged; pair search
is neither exhaustive nor Cartesian; query and target are independent and
distinct; no own-parent shortcut is used; and class labels are not training
supervision.  Keep the 5000/1000 inventories, GED labels, model/loss/optimizer,
selector, checkpoint ordering, split isolation, and runtime direction
unchanged.

### Consequences

- The writer and fail-closed validator now express the same frozen contract.
- The failed root remains non-PASS evidence; recovery uses a fresh attempt.
- No checkpoint, result, split, oracle, or numerical tolerance is adopted or
  relaxed by this metadata repair.

### Status

Accepted

---

## [2026-08-31] Keep Taste ComRecGC cohort identity authority stable on replay

### Motivation

The frozen GINE adapter selects the Taste source cohort with its decoded,
attributed identity graph.  A later replay check incorrectly replaced that
identity with the native explicit-hydrogen graph, so all selected parents were
rejected even though their frozen-GINE predictions and supplied identities were
unchanged.

### Decision

When the frozen adapter supplies identity graph payloads, require aligned
payloads on replay and use them for both cohort selection and replay identity
validation.  Continue to use the original attributed model graph for GINE
inference and chemical intervention.  The native-graph fallback remains only
for adapters that supply no identity payloads at either step.

### Consequences

- Cohort deduplication and replay now use one stable authority.
- GINE logits, probabilities, model graph features, and the frozen train pool
  are unchanged.
- Missing or misaligned replay identity evidence fails closed.

### Status

Accepted

---

## [2026-08-31] Give Taste GlobalGCE a lossless train-observed atom vocabulary

### Motivation

TasteMolNet train molecules contain elements outside the historical
Mutagenicity TU node-label table.  Rejecting those molecules blocks the T8
route, while dropping them or mapping distinct elements to one category would
change the frozen split and classifier semantics.

### Decision

For TasteMolNet only, derive the official GlobalGCE dense node categories from
the exact elements observed in the frozen train CSV, ordered deterministically
by atomic number.  Keep each element as a separate category and require its
atomic number to be an explicit (non-unknown) token in the same frozen
three-class GINE feature schema loaded from the checkpoint payload.  Bind the
schema hash and vocabulary source into codec metadata.  Mutagenicity and BACE
retain their existing vocabulary route.

### Consequences

- No molecule or atom is discarded, substituted, or collapsed into an unknown
  bucket.
- The official GlobalGCE model and rule algorithm remain unchanged; only its
  dataset-specific input dimensionality follows the frozen Taste train data.
- Calibration/test data are not consulted, and both target branches share the
  same exact vocabulary and GINE schema.
- Each branch writes its source-atom audit inside that branch's retained
  directory authority.  Direct-child publication uses temporary creation and
  atomic replacement relative to the already-open directory descriptor; it
  never treats `/proc/self/fd` itself as the output parent.

### Status

Accepted

---

## [2026-08-31] Finalize BACE ComRecGC from an exact resource-cap checkpoint

### Background

The live BACE ComRecGC trajectory was configured for 50,000 steps, but the
deadline policy now fixes a 20,000-step main budget with a 25,000-step fallback
and at least ten clean unique rules.  A stale progress JSON did not prove a
stall: a bounded read-only sample showed roughly one full CPU core of positive
work.  The prior cap observer could write a request but intentionally had no
signal or checkpoint-finalization authority.

### Decision

Keep any positive CPU, output, progress, checkpoint, or checkpoint-write
evidence typed `RUNNING_SLOW` and send no signal.  After an eligible committed
checkpoint request, a BACE-only executor reopens the checkpoint, binds its
digest and rule/lineage counts, revalidates the exact PID/start-ticks/raw
command/cwd/output/controller receipt immediately before signalling, and sends
SIGTERM to that PID only.  SIGKILL, fuzzy matching, process-group signals, and
test-derived decisions are absent.

Finalize the official checkpoint state directly in a fresh namespace.  Copy
and verify its selected-action trace chunks, rebuild the referentially complete
payload from the checkpoint's authoritative store, and run no further walk
step.  Retain the original 50k command as source provenance while recording
`M_configured_max=20000`, `M_fallback_max=25000`, `M_effective`, cap/early-stop
flags, and stop reason.  Persist the existing downstream BACE task fragment.
For ComRecGC only, accept 10--19 frozen rules and hold every K>R metric at the
complete R-rule prefix without copying rules; all other BACE method gates remain
unchanged.

### Consequences

- The currently CPU-active worker is not terminated merely because its progress
  file is stale.
- A committed checkpoint can become a complete resource-capped generation
  without changing RNG/model/data semantics or loading calibration/test data.
- The exact signal and postprocess receipts are durable and fresh-root bound.
- Deployment still requires a real >=20k eligible observer request; code and a
  waiting controller do not claim BACE ComRecGC PASS.

### Status

Accepted

## [2026-08-31] Bind Taste GlobalGCE to the frozen model-SMILES column

### Background

The fresh T8 run opened the authenticated Taste train split but rejected it
because the dataset carries `model_smiles`, `canonical_smiles`, and
`raw_smiles` rather than the legacy `smiles`/`parent_smiles` names.  After that
schema gate was repaired, the descriptor-held CSV path exposed a missing
standard-library `io` import before native rule generation.

### Decision

Use the same SMILES-column precedence as the frozen molecular-GNN loader:
`model_smiles`, then `canonical_smiles`, then legacy `smiles` and
`parent_smiles`.  Keep the exact train-file hash, row/label counts, split
checks, and three-class GINE authority unchanged.
Import `io` for the already-defined in-memory CSV reader; do not replace the
descriptor-held bytes with a path reopen.

### Consequences

- T8 consumes the representation on which the frozen GINE was trained.
- A missing supported column still fails closed.
- Native generation continues from the exact held CSV bytes.
- No dataset, split, oracle, or GlobalGCE algorithm setting changes.

### Status

Accepted

## [2026-08-31] Separate Taste COMRECGC identity and model graphs

### Background

T9 used a canonicalized native atom/edge graph as both the stable COMRECGC
identity and the authority for reusing a frozen-GINE row.  The actual GINE
input is decoded with source lineage and then featurized with complete atom and
bond attributes, so two simplified native graphs could share a key while
representing different model inputs.

### Decision

Keep a canonical chemistry payload solely for deduplication, registry,
lineage, and the official cache key.  Separately retain a canonical-JSON,
lossless model payload containing the ordered node features, directed edge
index, edge attributes, feature-schema hash, and graph hash that were sent to
GINE.  Bind that payload into the bridge checkpoint and require the same
stable identity to keep the exact model-graph digest plus the existing
logit/probability/embedding agreement envelope.  Never reconstruct a model
graph from the identity hash.

### Consequences

- Canonical identity no longer substitutes a simplified graph for GINE input.
- Checkpoint reload verifies complete model-graph bytes and semantic replay.
- The existing semantic assertion and tolerance remain unchanged; fresh T9
  attempts use checkpoint schema v3.

### Status

Accepted

## [2026-08-31] Normalize fixed-budget NeuroSED CLI paths before dispatch

### Background

The production trainer accepted `--config` as a string but called `resolve`
on the unconverted value.  The fixed GED labels and split protocol were
already complete; this type error alone prevented the real trainer launch.

### Decision

Normalize string and `Path` inputs through
`Path(value).expanduser().resolve(strict=...)`.  Anchor the relative HPC
configuration to the immutable checkout, require every scientific input and
symlink target to exist, and permit only the fresh output root to be absent.
Keep the 5000/1000 labels, non-MIP backend, selector, and generated-to-original
inference direction unchanged.

### Consequences

- The production runner no longer dispatches `resolve` on a string.
- Missing inputs fail before training; no labels or protocol choices are
  recomputed.
- Existing Slurm CLI arguments remain unchanged.

### Status

Accepted

## [2026-08-30] Add a narrow deadline-continuation heartbeat

### Background

The active 4x4 campaign now runs several already-owned science processes plus
deadline-specific AIDS/Mut repairs, BACE resource caps, and Taste preflights.
The existing continuation sidecar cannot accept these new observations without
a restart, while restarting science owners would risk output completeness.

### Decision

Add a minimal read-only deadline sidecar.  Its immutable spec names exact
process identities and terminal artifacts; it atomically writes one state and
heartbeat every 60 seconds.  It cannot start or stop science, publish matrix
cells, enable GNN ablations, or inspect calibration/test payloads.

### Consequences

- Existing science controllers keep ownership of their workers.
- GED adoption, NeuroSED readiness, resource-cap receipts, and fresh Taste
  attempts can be followed from one restartable heartbeat.
- The sidecar introduces no general scheduler, trust model, or controller
  protocol.

### Status

Accepted

## [2026-08-30] Directly recheck identity-bound AIDS DBSCAN self-neighbors

### Motivation

The production-shaped sparse subset boundary pass uses the exact Gram identity
to evaluate 2,000 float64 vectors against 273 anchors.  For 113 anchor rows,
different reduction rounding left positive self squared distances from
`2^-63` through `2^-61` (direct distances about `3.29e-10` through
`6.59e-10`).  The direct norm between the sample row and its identity-bound
anchor row was exactly zero in every case.  Because these values are near zero
rather than near `eps=0.02`, the existing epsilon-boundary direct recheck did
not cover them and the exact self-neighbor assertion failed closed.

### Decision

Keep Euclidean `eps=0.02`, inclusive `distance <= eps`, self-neighbor, strict
centroid-radius, coverage, and greedy semantics unchanged.  In the terminal
float64 boundary pass, identify self only through the frozen global sample
index to anchor-column mapping, recompute that one pair by direct norm, require
the result to be finite and exactly zero, and only then remove it from the
distinct-other-anchor count.  Record the number of direct identity rechecks
and require it to equal the proof's authenticated anchor count on reopen.
All non-self pairs continue to use the existing Gram result and only the
existing near-epsilon direct-norm recheck.

### Consequences

- Exact mathematical self membership is restored without a tolerance,
  approximation, epsilon widening, or forced diagonal assignment.
- A source/anchor identity mismatch still fails; non-self Gram or epsilon
  membership drift remains fail closed.
- Existing radius, centroid, coverage, partition, and official-greedy science
  is unchanged.  Deployment requires a new immutable checkout and fresh AIDS
  attempt; no protected process signal is authorized here.

### Status

Accepted

---

## [2026-08-30] Wire the exact-budget Taste NeuroSED production path

### Motivation

AutoDL already has an authenticated branch GEDLIB build, two independently
verified deterministic canary replays, and frozen split-local pair inventories.
The inventories contain exactly 5000 train and 1000 validation rows, not the
historical 5500/1100 reserve shape.  Requiring an absent reserve would block
before science, while claiming one would falsify provenance.

### Decision

Add a compact pickle-free NumPy GED-label writer and a separate fixed-budget
trainer.  Accept either an exact-budget inventory or a physically present 10%
reserve, and record which mode was opened.  For the active exact inventory set
reserve fraction and surplus to zero; any failed GED call is terminal
`BLOCKED_GEDLIB_LABEL_YIELD`.  Reopen both selected branch canary observation
files, compare all pair/role/hash/status/bound fields, prove their 100 IDs are
the train-inventory prefix, and adopt successful directional cache rows.

Train with the pinned GREED-expts AIDS notebook parameters: shuffled batches
200/1000, AdamW at `1e-3` learning rate and weight decay, CyclicLR 2000/2000,
cycle patience 5, and gradient clipping 0.1.  Use only the existing model
helper and `OfficialBatchInterleavedSelector`, never the historical own-parent
epoch-selector trainer.  Every improved checkpoint is captured before its
paired update in a fresh UUID root.  `best.pt` and `model.pt` are identical
copies of the selected pre-update bytes.  Explicitly verify the directional
NormSED training forward and different NormGED runner forward, per-model
batch/single agreement, runner loading, and generated-query to original-target
API direction.

The worker may write only verifier readiness.  A second invocation in a new
Python process reopens the checksum inventory, labels, pair manifests,
selection receipt, checkpoint, model card, selector trace, direction trace,
validation metrics, and vendored GCF source; it replays the selector state
machine and writes `[TASTE_NEUROSED_FIXED_BUDGET_PASS]` last.  This change does
not deploy code, run GEDLIB, allocate a GPU, or claim a runtime PASS.  The
single-argv `--train-and-verify` mode launches this same verifier as a separate
Python process and reports the exact PASS-file path to managed scheduling.

### Consequences

- No new solver or pair-budget research is introduced.
- Exact 5000/1000 failures cannot silently resample absent reserve rows.
- The selected 100-pair canary is reused without recomputation.
- T7 still requires a real independently verified AutoDL checkpoint.

### Status

Accepted

---

## [2026-08-29] Separate Taste T9 checkpoint authority from model loading

### Motivation

The T9 holder correctly retained `config.yaml` as part of its exact T3
checkpoint authority, but passed that complete eight-file mapping to the
frozen-GINE in-memory loader.  That loader deliberately accepts exactly seven
runtime payloads and rejected the extra configuration evidence before science.

### Decision

Continue to retain, hash, and revalidate all eight authority payloads.  Before
constructing the frozen GINE, require that exact authority key set and project
only the loader's seven exact keys, excluding `config.yaml`.  Missing, empty,
or injected authority payloads remain fail closed.

### Consequences

- The immutable checkpoint authority is not weakened or rewritten.
- The model loader receives exactly its existing contract; no permissive
  extra-key behavior is introduced.
- The failed T9 stage remains unusable.  Runtime recovery requires a fresh
  managed attempt from a newly integrated immutable execution commit.

## [2026-08-30] Bound AIDS subset RSS above its measured mmap-scan baseline and reap exited workers

### Motivation

The fresh exact-recovery route reached the production-subset preflight, but its
first 2,000-row external DBSCAN failed before doing work because the process
high-water RSS was 24,293,740,544 bytes while the stage hard-coded an 8 GiB
absolute limit. The high-water value is expected: deterministic selection has
already scanned the complete file-backed vector and distance mappings. The
worker then exited with code 1, but the controller compared only PID start
ticks, treated the unreaped zombie as live, and kept the stage `RUNNING`.

### Decision

Keep the exact DBSCAN 96 GiB scope unchanged. Give the subset process an
independent 32 GiB authorized ceiling and derive its actual absolute limit once,
immediately before subset DBSCAN, as measured `max(VmRSS, VmHWM)` plus exactly
8 GiB. Persist this complete calculation in aggregate and per-subset manifests
and bind it in the controller terminal validator. Poll directly owned workers
through `Popen.poll()`, retain `wait()` and process-group quiescence checks, and
treat procfs `Z`, `X`, and `x` states as dead for reattachment and status.

### Consequences

- The five subset definitions, sklearn comparison, exact DBSCAN parameters,
  source authority, and full-production non-claim are unchanged.
- A baseline-plus-margin value above 32 GiB still fails closed; the fix is not
  an unbounded memory increase and does not borrow the exact-only 96 GiB scope.
- A nonzero child exit reaches the existing controller exception path, which
  persists controller and current stage as retryable `BLOCKED`; a dead leader
  with live process-group descendants still blocks reattachment.
- The already failed CID/attempt remains immutable. Any deployment must use a
  newly integrated immutable checkout and fresh adoption/spec/manifest/CID;
  this decision authorizes no signal to the protected legacy brute process.

### Status

Accepted

---

## [2026-08-29] Preserve canonical AIDS anchor-component order

### Motivation

The frozen, SHA-bound AIDS anchor-edge array has three connected components.
Canonical IDs are assigned by increasing minimum selected-anchor position, so
their sizes are `(114, 149, 3)`.  The frozen shortcut-failure certificate
independently records that the traversal from anchor position zero reached 114
anchors.  The adoption authority mistakenly pinned the same sizes in
descending-first order as `(149, 114, 3)`, causing a genuine source to fail
after all earlier byte and semantic checks passed.

### Decision

Pin the production authority to canonical component order `(114, 149, 3)`.
Continue to recompute every component from the immutable edge array and bind
the shortcut failure's `anchor_component_reached_count` to canonical component
zero.  Use an asymmetric, interleaved graph regression so sorting component
sizes instead of preserving canonical graph order cannot pass accidentally.

### Consequences

- The authority now agrees with both the frozen graph topology and the
  independently hashed shortcut-failure certificate.
- No edge, anchor, seed, selection, distance, or exact-recovery science is
  changed or regenerated.
- The failed empty adoption child remains unusable.  Deployment still requires
  a new reviewed immutable worktree and a fresh direct child, and this change
  authorizes no signal to the protected old process.

### Status

Accepted

---

## [2026-08-29] Use pair-keyed WNode for native full-graph BACE candidates

### Motivation

The adoption-only BACE GCFExplainer postprocess preserved the completed 50k
generation and candidate freeze, but every calibration shard stopped before
its first WNode result. The evaluator passed a complete generated molecule to
the deletion-action cache API, whose truthful contract requires match atom
indices and a match-selection policy. Neither exists for a native full-graph
GCFExplainer or ComRecGC candidate.

### Decision

Evaluate native full-graph candidates with the exact MolCLR node-Wasserstein
pair API. Keep the action-aware API for methods that actually apply a matched
action. Do not invent empty match indices or a synthetic deletion policy to
make the cache key pass.

### Consequences

- The WNode value remains the exact full-graph-to-full-graph EMD under the same
  frozen MolCLR checkpoint and method-specific distance namespace.
- Canonically identical graph pairs may share a symmetric cache entry, which
  is scientifically correct because candidate IDs do not alter WNode.
- Failed calibration shard roots remain terminal; postprocessing must use
  fresh shard/controller namespaces while reusing the completed 50k generation
  and PASS candidate universe.

### Status

Accepted

---

## [2026-08-29] Bound only the observed Taste T4 CUDA replay aggregates

### Motivation

A SEALED adaptive T4 worker and two independent GPU1 replays agreed on every
discrete scientific field: 38 strict flips, 17 distinct parents, and the full
`1 -> 0` / `1 -> 2` destination distribution.  The verifier quarantined the
attempt because three reduction-derived aggregates differed by at most about
`1.35e-8`, below the already frozen `1e-6` batch/single CUDA tolerance.

### Decision

Use the existing `T4_BATCH_SINGLE_ATOL=1e-6` absolute, zero-relative envelope
only at the exact replay paths
`oracle_smoke.json.batch_single_max_abs_difference`,
`oracle_smoke.json.cf_drop.mean`, and
`oracle_smoke.json.cf_drop.minimum`.  Require those leaves to remain finite.
Keep the prior comparator for every other float and exact equality for types,
keys, list shape, discrete values, destination counts, and authority hashes.

### Consequences

- CUDA reduction-order tails no longer create a false replay quarantine.
- Any change above `1e-6`, any non-finite value, or any unlisted scientific or
  authority change remains a hard failure.
- The quarantined attempt remains immutable and is not promoted; a runtime
  PASS still requires a fresh managed worker and independent verifier.

### Status

Accepted

---

## [2026-08-29] Compare AIDS close bitmaps by typed row semantics

### Motivation

The pinned pair-semantics scan stores its 91,916,686 decisions as binary
`uint8`, while the zero-copy close-view materializer stores the same decisions
as NumPy `bool`.  Both files have their own canonical contract hashes.  The
adoption validator incorrectly required those byte hashes to match and also
required the scan artifact itself to be `bool`, so it rejected the genuine
authority even though a bounded full scan proved every row equivalent.

### Decision

Validate each bitmap against the hash in its own owning contract.  Require the
pair-semantics scan to be one-dimensional `uint8` with value domain `{0, 1}`
and the materialized close bitmap to be one-dimensional `bool`, both at the
exact physical row count.  Compare `uint8.astype(bool)` with the materialized
bitmap row for row in bounded one-million-row blocks, and continue to require
every row to agree with the all-pairs-close authority.  Do not compare byte
hashes across the two different storage representations.

### Consequences

- The canonical scan and materialized view remain independently content-bound
  while their common scientific meaning is checked exactly over every row.
- Wrong dtype, non-binary scan values, row disagreement, false rows, shape
  drift, hash drift, and path/inode replacement remain fail closed.
- This changes only adoption validation.  It does not rewrite either source,
  alter pair order or distance/DBSCAN science, authorize deployment, or permit
  signalling the protected old worker.

### Status

Accepted

---

## [2026-08-29] Treat the AIDS pair-semantics bitmap as a contained artifact tree

### Motivation

The pinned production pair-semantics contract lives at
`pair_semantics_science/close_pair_contract.json` and canonically references
its scan artifact at
`pair_semantics_science/distance_scan/close_pair_bitmap.greed.uint8.npy`.
The adoption validator incorrectly required the bitmap and contract to share
one immediate parent, so a complete real authority scan failed after hashing
the frozen 91,916,686-row store even though the bitmap is a physical descendant
of the contract-owned root.

### Decision

Require the bitmap to be one regular, no-symlink physical path beneath the
pair-semantics contract parent rather than requiring the same immediate parent.
Apply the same containment check during the pre-write source-overlap discovery
and the full authority scan. Keep the existing content hash, link-count,
pre/post stat, held-directory, terminal-reopen, and receipt-stat bindings.

### Consequences

- The canonical `distance_scan/` layout can be adopted without changing or
  regenerating any frozen scientific artifact.
- Siblings, symlinked files or directories, lexical `..` aliases, hardlink
  aliases, and inode/content replacement remain rejected.
- This changes only authority-path validation; pair/vector bytes, exact DBSCAN
  science, release pins, worker count, handover gates, and signal policy are
  unchanged. Deployment still requires a newly reviewed immutable execution
  commit and a new adoption child.

---

## [2026-08-29] Release the bounded Taste T9 smoke through managed v2 on GPU1

### Motivation

The Taste T9 scientific core already froze native COMRECGC at M=500 with a
real step-250 checkpoint/reload, but its only launcher referenced an unfinished
GPU2 controller/result-dispatch contract and a nonexistent runner. The project
owner now authorizes the shortest bounded smoke on GPU1 after T4 and permits a
trusted single operator root, while still requiring exact provenance, separate
verification, fresh publication, and no test access.

### Decision

Keep every scientific parameter and the pinned official source unchanged. Add
`run_tastemolnet_comrecgc_smoke` as a managed-v2 worker route. Both worker and
verifier retain the receipt-only T2 adoption, T3/T4 managed outputs, their one
common frozen GINE, checkpoint payload hashes, checkpoint-declared train CSV,
`configs/hpc.yaml`, clean execution commit/tree, and all seven official source
files. The worker writes only aggregate science/input evidence, worker-exit
evidence, and SEALED inventory. A different process reopens that closure and
every input before using the existing atomic no-replace terminal publisher.

Use the explicit launch semantic `TRUSTED_SINGLE_OPERATOR_ROOT` instead of
inventing a replacement controller receipt. The AutoDL wrapper takes the
existing UUID-scoped lock for physical GPU1 and holds it across worker,
SEALED handoff, verifier, and publication. It requires T4 first, performs only
one bounded idle check, never signals another process, and leaves GPU0/GPU3
untouched. The old disabled GPU2 release config remains historical evidence,
not an alternate path.

### Consequences

- The generic managed-v2 PASS is the physical terminal marker; the exact T9
  method marker is nested in independent verification and printed afterward.
- Terminal artifacts remain aggregate-only and explicitly record no validation
  payload, calibration payload, test, RF, redistribution, or paper eligibility.
- M=50000 remains unrun, and code/tests cannot claim a scientific PASS.
- Paired Slurm entrypoints remain static AutoDL-only refusals.

### Status

Implemented for focused review; not deployed or executed.

---

## [2026-08-29] Remove ETA and relative-speedup gates from AIDS old-brute handover

### Motivation

The latest operator authorization permits the exact old brute to be retired
after the new exact route has produced a recoverable durable checkpoint,
survived an independent controller reload/reattachment, and demonstrated at
least ten continuous minutes of positive authenticated progress.  The older
handover contract additionally required a projected ETA of at most 48 hours
or unavailable 100x evidence, which could block an otherwise authorized safe
handover indefinitely.

### Decision

Bump the handover schema to v3 and make ETA/relative-speedup values diagnostic
only.  Eligibility continues to require reviewed release authority, source and
pair-store closure, exact old/new PID-generation bindings, durable checkpoint
and independent reload evidence, ten continuous minutes of fresh progress,
positive throughput, and a non-blocked controller.  The controller still has
no signal API; a separately reviewed executor must reopen the exact old
generation before issuing the one authorized graceful `SIGTERM`.

### Consequences

- A healthy new exact route is not rejected solely because its conservative
  full-run ETA exceeds 48 hours.
- Checkpoint, reload, duration, progress, process identity, and release gates
  remain fail closed.
- This change authorizes no signal by itself and does not stop the live old
  process.

## [2026-08-29] Expose explicit trusted-operator AIDS release pins

### Motivation

The production-spec generator intentionally emitted only an unlaunchable
template, leaving six release pins unset and deployment authorization false.
The owner has now explicitly authorized the independently reviewed eight-worker
exact route, so deployment needs a reviewable CLI boundary rather than editing
generated JSON by hand.

### Decision

Allow `generate-production` to receive each reviewed component commit and one
explicit `--authorize-production-deployment` flag. Authorization fails unless
all pins are full Git SHAs, the immutable science pin is unchanged, and the
controller pin exactly equals the clean execution worktree HEAD. Repeat that
exact equality at manifest build, manifest validation, launch-time load, and
handover review; ancestry alone is insufficient. Existing ancestry,
entrypoint-hash, adoption, source, and manifest gates remain in force.
Omitting the flag preserves the historical release-unready behavior.

### Consequences

- A trusted single operator can create a launchable spec without mutating it
  after generation.
- Partial pins or an old controller pin cannot accompany an authorized spec.
- This interface does not weaken scientific or process-identity validation.

## [2026-08-29] Start the reviewed AIDS exact-component route with eight CPU workers

### Motivation

The fast main-table completion policy authorizes the already reviewed AIDS
exact-component science and asks the fresh route to start with eight CPU
workers, while retaining twelve as the maximum bounded CPU allocation.  A
read-only AutoDL audit reconfirmed the physical scientific input: 91,916,686
rows from 1,283 parents by 71,642 candidates, 560 closed source chunks in
strict order, column 0 parent / column 1 candidate, and candidate-major /
parent-minor row order.  The adopted DBSCAN and downstream implementation
already freeze Euclidean `eps=0.02`, `min_samples=3`, self-neighbour semantics,
exact multi-component partitioning, streaming centroids, strict `<` radius and
centroid-norm filters, coverage, and the official greedy selector.

### Decision

Make eight the generated production-spec default and accept only the closed
integer range 8--12 in the manifest and launcher.  The selected count remains
part of the immutable manifest and the frozen BLAS/OpenMP environment; it
cannot be changed inside an existing attempt.  Scaling to twelve therefore
requires a fresh not-yet-started attempt or a separately reviewed resume
contract, never an in-place manifest edit.

Do not change rows, chunk/input authority, pair orientation, DBSCAN parameters,
checkpoint identity, component algorithm, streaming summary, radius, coverage,
or greedy semantics.  Production authorization and release pins remain
explicit; this change does not create an automatic release bypass or signal
the protected old process.

### Consequences

- The fresh CPU route can coexist with the live BACE GPU jobs and begin with
  the requested eight-worker reservation.
- Values below eight or above twelve fail before a controller log/PID is
  created.
- The protected old AIDS process remains untouched until the dedicated
  handover verifier and exact-generation stop executor are separately
  authorized and run.

## [2026-08-29] Use one bounded pinned non-MIP GEDLIB selector for Taste NeuroSED

### Motivation

The pinned GREED wrapper names F2/BLP methods only when its own `GUROBI`
preprocessor flag is enabled. The AutoDL host has no approved Gurobi 9.1.1
runtime or licence, and waiting for or installing one is no longer part of the
main-table path. Removing only the wrapper define is invalid because its
F2/BLP enum references then fail compilation.

### Decision

Build one isolated wrapper from the exact pinned GREED and GEDLIB commits while
removing the Gurobi define and the F2/BLP mapper branches. Retain exactly the
one authenticated deterministic non-MIP candidate, `branch`. Both exposed
IPFP and `anchor_aware_ged` are excluded: IPFP's default initializer uses an
unseeded C++ `random_device` path, and anchor-aware GED invokes that IPFP path
internally without setting a deterministic initializer. On one fixed seed-7
cohort of 100 real independent train query-target
pairs, run each candidate twice with single-thread deterministic settings, at
most ten minutes per candidate and thirty minutes total. An eligible candidate
must replay identical statuses/bounds, succeed on at least 95 pairs, and return
finite `lower <= upper`; select the highest eligible successful-pair throughput.
An independent verifier reopens both observations JSONL files and both
benchmark manifests for every eligible candidate, checks their SHA-256 values,
recomputes outcome hashes, success counts, determinism, throughput, and the
final choice, then emits a separate receipt.

Use selected throughput to project 6000 labels. Select 5000 train and 1000
validation pairs when the projection is at most 24 hours; otherwise select the
explicit resource-reduced 2000/500 budget. Do not run the historical
5k/10k/20k search or make 500/1000 tiers separate release blockers.
Given the ten-minute and >=95/100 selector gates, the reduced branch is a
defensive policy and may be dormant for a passing candidate; it remains legal
for a separately observed projection above 24 hours and is tested directly at
the model-card boundary.

### Consequences

- Manifests state `GED_LABEL_BACKEND_VARIANT=NON_MIP_GEDLIB`,
  `F2_BLP_USED=false`, and `GUROBI_USED=false`; this is never reported as the
  upstream F2/BLP configuration.
- Model cards set `full_official_neurosed_semantics_claimed=false`. They may
  separately attest the preserved upstream model/loss/interleaved-selection
  semantics, but never the complete upstream GED backend configuration.
- Pair sources remain split-local, independent, and label-agnostic;
  calibration and test are not opened.
- Only label solving changes. Official GREED model/loss/interleaved validation,
  checkpoint selection, and generated-query to original-target GCF distance
  direction remain release requirements.

## [2026-08-29] Replace T4 destination diversity with an adaptive calibration gate

### Motivation

The fixed sixteen-parent/four-deletion smoke made simultaneous observation of
Sweet-to-Bitter and Sweet-to-Tasteless a release requirement. That small-sample
diversity condition is not part of strict counterfactual correctness and can
block all downstream Taste experiments even when the frozen three-class oracle
produces many real Sweet-to-non-Sweet flips. The project owner has authorized a
bounded adaptive calibration-only successor and explicitly made destination
diversity diagnostic rather than terminal.

### Decision

Keep the existing managed-v2 worker/verifier and managed release-v3 GPU-1
authority. Replace only the T4 scientific search with deterministic rounds of
16 parents and at most 8 deletions, 64 and at most 16, then 128 and at most 32.
Every parent comes from the authenticated calibration cache, has true label 1,
and is predicted as label 1 before deletion. Stop at the first round containing
at least 16 strict flips across at least 8 distinct parents, where both
`1 -> 0` and `1 -> 2` satisfy `pred_before == 1 and pred_after != 1`.

Continue to validate real connected deletions, three-class probabilities,
batch/single parity, fail-closed invalid/full-parent controls, one model load per
scientific process, and no train/validation/test/RF access. Publish only
aggregate documents, including `destination_distribution.csv`. Both
destinations produce `DESTINATION_DIVERSITY_PASS`; exactly one produces
`DESTINATION_DIVERSITY_SINGLE_CLASS_WARNING` and remains releasable. A search
that never reaches the flip and parent-coverage minima fails. Deterministic unit
fixtures cover `1 -> 0`, `1 -> 2`, and `1 -> 1` independently.

### Consequences

- The 2026-08-28 fixed `16 x 4` destination-diversity decision remains
  historical evidence but is superseded for new T4 executions.
- The warning cannot block T4, T6, or the main Taste experiment queue.
- No test or RF payload is opened, and no row-level identifier, SMILES, or
  prediction is published.
- This implementation change does not itself claim a runtime PASS, launch an
  ablation, or authorize GPU 0/GPU 3 use. A fresh reviewed immutable execution
  and independent verifier publication remain required.

### Status

Implementation, focused tests, documentation, and paired Slurm synchronization
are complete locally. Independent review and fresh AutoDL execution are
pending.
## [2026-08-29] Accept hash-matched empty official Python initializers for K20

### Motivation

The third immutable AutoDL K20 startup reached the full official-source audit
and then failed before output creation because the pinned clean GlobalGCE
checkout contains a tracked zero-byte `src/__init__.py`. Git records that file
as an ordinary `100644` blob with the SHA-256 of empty content. The runtime
closure incorrectly treated every zero-byte Python source as missing even
though it subsequently compares each file's size and digest with its exact Git
blob. No GPU2 science child was created and the controller identity is retired.

### Decision

Permit byte size zero in the complete tracked-Python authority and in its
downstream normalizer. Continue to require Git mode `100644`, a clean checkout,
one regular filesystem leaf with link count one, exact Git blob length and
SHA-256, stable inode/device/size on descriptor reopen, and no bytecode,
native, ignored, untracked, skip-worktree, or assume-unchanged runtime code.
Critical official source files retain their separate nonempty/hash pins.

### Consequences

- Exact empty package initializers no longer block the pinned official commit.
- A missing, symlinked, linked, dirty, truncated, grown, or hash-mismatched
  source still fails closed.
- The next AutoDL attempt requires a fresh reviewed commit, detached clean
  worktree, controller UUID, output root, and log. No runtime launch or PASS is
  claimed by this code change.

## [2026-08-29] Install the K20 controller signal mask before science imports

### Motivation

The first immutable AutoDL launch failed closed before creating its output
root: imported numerical/scientific dependencies had already created native OS
threads when `run_extension` attempted the single-thread signal-mask gate.
Local hostile tests did not reproduce that runtime import behavior.

### Decision

The thin K20 CLI now classifies the exact controller/raw-round command before
importing the scientific core.  A controller blocks SIGINT, SIGTERM, and
SIGHUP while `/proc/self/task` still contains only its initial task and passes
the predecessor mask into `run_extension`.  After heavy imports, the core
reopens every live task's `/proc/.../status` and requires all `SigBlk` values to
contain the complete release mask.  It repeats that process-wide proof at each
controller revalidation.  A raw-round exec unblocks the same signals before
its scientific imports, so the controller still has no child-signal channel.
The bootstrap rejects an ignored release-signal disposition; therefore plain
`nohup` is not a valid launcher, while `tmux` or a detached `setsid` wrapper
that restores HUP/INT/TERM defaults is valid.

### Consequences

- The failed controller emitted only a structured BLOCKED log and no science
  output root or GPU2 child; its controller identity is not reused.
- The production guarantee now covers native-library threads created during
  imports instead of assuming those imports are single-threaded.
- A fresh reviewed commit, detached AutoDL worktree, controller UUID, log, and
  output root remain required before another launch.

## [2026-08-29] Restore the dedicated AIDS old-brute handover certificate

### Motivation

The reviewed private commit `4b7fcde15a7b2dd4ba0249651cec18bffaddec72`
contains a fail-closed, read-only old-route handover gate, but the reviewed
tree-preserving lineage integration intentionally kept the current successor
tree. Object-graph reachability therefore did not deploy that gate. The
protected brute worker remains the exact PID generation `273939/687141119` and
must receive no signal merely because the private commits are ancestors.

### Decision

Port only the handover certificate, authenticated exact-checkpoint reopen,
second-controller reattachment receipt, and continuous-progress monitor needed
by the current twelve-worker controller. Freeze the old process identity in the
generated and validated production resource contract, including its raw
cmdline SHA-256. Eligibility requires the exact old generation and command to
remain live, reviewed release ancestry and a clean
execution tree, typed adoption/source closure, a new-route hash-chained durable
checkpoint observed across a real controller restart, at least ten continuous
minutes of fresh exact progress, and positive throughput. The later v3 policy
above makes conservative ETA a diagnostic rather than an eligibility gate.

The original contract's optional 100x branch remains fail closed here: this
controller does not accept caller-supplied old-route throughput. A future
separately reviewed stop executor may provide exact-generation live speedup
evidence when the conservative ETA branch is unavailable. This controller only
reports `eligible_to_request_old_brute_stop`; it never calls a signal API.

### Consequences

- PID reuse, exec/cmdline replacement, a dead old generation, checkpoint
  tamper, a same-controller pseudo-resume, stale/discontinuous progress, or
  nonpositive throughput all keep the certificate `NOT_ELIGIBLE`.
- A later executor must reopen the exact old PID/start-tick generation and own
  the one graceful `SIGTERM`; no general kill or `SIGKILL` is authorized.
- This successor is code, tests, and documentation only. It does not deploy,
  launch science, start handover, or signal the protected process.

## [2026-08-29] Match AIDS adoption structure to the pinned real manifest

### Motivation

The production adoption authority already pins the fixed controller manifest
to SHA-256
`7b2987bc2d223ebe3262cc15bc43bd1c0b030c6706a1c074959d154af5fd84d7`.
A fresh AutoDL adoption nevertheless failed after that hash reopened because
the validator required a synthetic `run_tastemolnet: 0` field. The pinned real
manifest has no such field, while the unit fixture had accidentally invented
it, making the structural check impossible to satisfy in production.

### Decision

Require `run_tastemolnet` to be absent from the fixed source manifest, matching
the exact pinned bytes. Do not use a default that could make an absent field
look equivalent to an explicit value. Make the fixture use the real shape and
reject an injected field even when a test rebinds both the manifest SHA and the
persistent snapshot.

### Consequences

- The hash-pinned real c766 manifest can pass its intended structural check.
- Adding a Taste execution flag remains a fail-closed authority change.
- This change does not authorize deployment, launch science, or signal the old
  brute process; release review and the dedicated handover gate remain
  mandatory.

## [2026-08-29] Replace the AIDS zero-byte root preclaim with a content-bound claim

### Motivation

The exact private lineage `a8c42be4a4f7df89cb4dc2fc79713d6ce8fae923`
-> `4b7fcde15a7b2dd4ba0249651cec18bffaddec72` ->
`f4fcfcd8a126efb483402e9a815ce2dd08bb746e` was imported through the verified
bundle whose SHA-256 is
`4e6536c6ade21dc5d35deed285c25580e0be14a1dea18e7a269d9f063876c137`.
Independent AutoDL replay retained 84 passing focused tests but exposed one
real fail-open: unlinking and recreating the empty 0600 controller-root claim
could reuse its inode, so the old device/inode/mode/owner/link/size receipt
could accept the replacement. The original imported objects remain immutable
review evidence and are not merged as a private branch.

The `141b37c60d43a17c49c93f96b99904748c77aa0d` integration base already
contains the later equivalent twelve-worker AIDS controller. Apply the repair
to that current controller instead of replacing it with the older imported
tree.

### Decision

Make every fresh root claim a bounded JSON document created with `O_EXCL`.
Before any root is finalized, write and fsync a fresh random attempt id and
nonce together with the exact controller id, controller root, controller-
manifest SHA, schema, and creation time; then fsync the parent directory. Bump
the owner receipt schema and bind the claim's content SHA-256, nonce SHA-256,
attempt id, device, inode, mode, uid, gid, link count, size, nanosecond mtime,
and nanosecond ctime.

Fresh initialization and resume both hold the claim flock, read through the
opened no-follow descriptor, compare the named path before and after, validate
the strict payload, and recheck the complete content/stat binding before
unlock. A zero/partial claim left by a crash is retained but cannot be resumed;
manual diagnosis and a fresh CID are required rather than filling or replacing
it. Copying the original bytes is insufficient because the owner receipt also
binds the physical inode and ctime. A hardlink/unlink/relink ABA keeps the inode
and bytes but changes ctime and therefore also fails closed.

### Consequences

- Zero-byte replacement, copied-content replacement, rapid inode reuse,
  same-inode content mutation, and same-inode ABA cannot satisfy resume.
- Ordinary same-CID resume retains the exact content/stat receipt and remains
  idempotent after root or owner-publication interruption.
- Historical zero-byte-claim roots require their historical code and cannot be
  silently adopted by this successor.
- This change creates no execution pin, launches no science, and sends no
  signal to the old AIDS brute process. Independent review remains mandatory.

### Status

Local successor implementation and focused tests; deployment not authorized.

## [2026-08-29] Match the official GREED GEDLIB include as an exact line

### Motivation

The first authorized isolated AutoDL build stopped before compilation with
`official CMake GEDLIB include anchor changed`.  The pinned GREED CMake file
contains one standalone GEDLIB-root include plus seven valid include
subpaths and three valid link subpaths.  Counting the root string as a raw
substring therefore counted every subpath and rejected the authenticated
official file.

### Decision

Authenticate exactly one complete standalone
`${CMAKE_SOURCE_DIR}/ext/gedlib` line and assign its replacement by the exact
line index, never by first-substring replacement.  Then replace every
slash-delimited GEDLIB subpath and reject any remaining legacy GEDLIB-root
reference.  Retain the existing fail-closed behavior for a missing or
duplicate standalone line and retain the complete Gurobi-removal check.  Cover
the official root-plus-seven-include-subpath shape, three link subpaths placed
before the root, and missing, duplicate, and comment-hidden tamper cases in a
focused regression.

### Consequences

- The overlay accepts the byte-authenticated pinned official CMake layout
  without weakening its unique-root anchor.
- Missing or duplicated root includes, comment-hidden legacy roots, and any
  remaining Gurobi reference still block the build.
- The failed AutoDL prefix remains non-PASS evidence.  A fresh retry and all
  import, `ldd`, deterministic-fixture, and `python -I` checks remain required
  before either provisioning PASS marker may be emitted.

## [2026-08-29] Authenticate the vendored GCFExplainer source against upstream

### Motivation

The official fixed-budget NeuroSED readiness gate correctly rejected the
vendored GCFExplainer snapshot because it had critical file hashes but no
authenticated upstream commit. A free-form 40-character model-card value was
not sufficient provenance for a production Taste GCF route.

### Decision

Pin the official repository commit
`cc7ca30eb2026c57f20cd6afe2ee621f486fcf2e`. A recursive comparison against
that checkout proves every retained file in `baselines/gcfexplainer_official/`
is byte-identical; the project snapshot only omits upstream dataset/model
artifacts. Require this exact commit and the existing five critical executable
file hashes in every fixed-budget model card. At readiness time,
descriptor-reopen the exact 17-file retained inventory, reject symlinks and
extra entries, hash the real bytes, and bind the complete inventory digest;
model-card strings alone are not source authentication.

### Consequences

- The missing-GCF-identity blocker is closed without changing executable
  science code.
- GEDLIB/pybind11 provisioning, the real build, disjoint throughput benchmark,
  compact labels, training, managed verification, and T7 remain blocked or
  unrun; this provenance result emits no scientific marker.
- The pinned GREED README prescribes GEDLIB branch/tag v1.0. The build gate
  resolves and freezes that tag to
  `120856f670e013f080b116c0be4cc6bd72fc935d`; an operator-supplied alternative
  commit is rejected rather than treated as equivalent.

## [2026-08-29] Make GEDLIB worker selection a replayable manifest authority

### Motivation

The fixed-budget planner accepted an operator-provided `--selected-workers`
integer and a separate CPU-contention boolean. Those values were not closed by
the real worker trials, so a healthy 1/2/4/8 benchmark set could be bypassed or
partially omitted before budget projection.

### Decision

Require one fresh, mutually disjoint, at-least-100-pair real GEDLIB report for
every legal 1/2/4/8 candidate available on the runtime physical cores. Embed
all reports in a machine-generated manifest, exclude timeout/error, unhealthy
load/iowait, or greater-than-10% BACE/AIDS contention candidates, and select
the highest reproducible pairs/hour with a deterministic lower-worker tie
break. Rebuild that manifest inside the budget planner; remove the manual
worker and CPU-contention CLI inputs.

No reviewed producer currently authenticates the protected BACE/AIDS process
generations and samples their pre/during progress counters. Freeze its source
pin as null: benchmark preflight emits a self-hashed blocker evidence document
and exits before pyged/workers, worker selection returns
`BLOCKED_GEDLIB_RESOURCE_EVIDENCE`, and a
self-declared evidence PASS is rejected.
Even after that producer is reviewed, one missing/unauthenticated required
candidate blocks selection globally; only authenticated-but-unhealthy trials
may be excluded while another authenticated candidate is selected.

The current sampler has only the unique 1600-pair 100/500/1000 planning
partition and no official path for four additional disjoint worker cohorts.
Record `WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED` as a second machine
blocker and remove the misleading four-report example from the static Slurm
wrapper.

### Consequences

- The selected integer is evidence output, never operator authority.
- Missing candidates, reused pairs, backend drift, manifest tampering, or a
  selected-worker mismatch against the 1000-pair report fail closed.
- Implementing and reviewing the protected-job resource evidence producer is
  now an explicit prerequisite to any non-null worker selection.
- Implementing a deterministic, hash-closed worker-trial cohort builder that
  is disjoint from planning is a separate prerequisite.
- Real AutoDL trials remain unrun, so this closes code review only and emits no
  scientific marker.

## [2026-08-28] Adopt the existing frozen BACE Ours cell by receipt only

### Motivation

The exact BACE Ours standardized `attempt-0` already satisfies the ordinary
four-by-four registry as `FROZEN_PASS`.  Re-running its candidate generation,
oracle, calibration, distance computation, or test evaluation would waste GPU
time and could create a second scientific identity.  A paper-matrix successor
still needs durable evidence that the exact historical bytes were deliberately
adopted from a quiescent source.

### Decision

Pin the physical standardized root, its separate raw-writer guard root, all
sixteen source file SHA-256 values, and the BACE/GINE/oracle/dataset/split/
MolCLR/threshold/temperature/feature identities in one checked-in policy.
Before publication, hash and stat every source file twice, audit Linux procfs
for writable descriptors under both roots, and invoke the same ordinary
registry candidate gate used by the matrix audit.  Publish only a three-file
receipt (`adoption_manifest.json`, `verification.json`, and the exact
`[BACE_OURS_FREEZE_ADOPTION_PASS]` marker) under a fresh
`matrix/adoptions/bace_ours_frozen_*` directory.  Atomically publish the two
receipt JSON files first, reopen them and every external gate at the final
path, and only then durably create `PASS` as the last terminal operation.  A
post-publish verification failure may retain the two diagnostic JSON files but
must not leave `PASS`.  Do not copy scientific
CSVs, open raw test data, or recompute any numeric value.

### Consequences

- The receipt records adoption provenance; the authoritative scientific files
  remain at the checksum-pinned standardized root.
- A fresh matrix audit must still explicitly scan that source and independently
  reach `FROZEN_PASS`; the receipt alone cannot fill a cell.
- Each fresh matrix audit writes `combined_audit.json` with the size/SHA closure
  of every sibling, fsyncs the sibling directories, and publishes
  `matrix_status.json` last; the old six-cell audit is never overwritten.
- Source drift, a live writer, identity mismatch, dirty execution checkout,
  non-fresh destination, or registry downgrade blocks the marker.
- This decision authorizes only BACE Ours receipt publication.  It does not
  rerun BACE Ours or alter the protected BACE GCFExplainer/ComRecGC processes.

## [2026-08-28] Cap AIDS exact-component recovery at twelve CPU workers

### Motivation

The fixed-budget NeuroSED campaign will benchmark GEDLIB concurrently with the
CPU-only AIDS recovery and the two protected BACE legacy processes.  The new
resource contract caps the AIDS exact route at twelve workers; the previously
staged controller froze sixteen and therefore cannot be released unchanged.

### Decision

Freeze `DEFAULT_THREAD_COUNT=12` in the exact-component recovery manifest and
all stage environments.  Freeze the outer launcher preflight to the same
literal and regression-test that 12 passes while 0, 8, and 16 fail before a
controller log or PID is created.  This changes only the CPU reservation: the pinned
91,916,686-row authority, exact component algorithm, block order, checkpoints,
and downstream numerical contracts remain unchanged.  The old brute process
remains read-only and receives no signal from this change.

### Consequences

- A fresh controller/review/release commit is required; old unready specs are
  not patched or adopted.
- GEDLIB worker selection must still prove that neither AIDS nor protected
  BACE throughput drops by more than ten percent.
- This decision alone does not authorize launch or old-brute handover.

## [2026-08-28] Freeze T6/T8/T9 smoke markers to the T0--T16 contract

### Motivation

The unrun Taste smoke implementations used descriptive marker names that did
not carry their assigned T-stage numbers. T6 also stored an unbracketed value
in structured evidence while its stdout and `PASS` leaf added brackets at the
call site. That made controller configuration and terminal verification prone
to a silent one- or two-bracket mismatch.

### Decision

Freeze the exact stage markers to `[TASTE_T6_OURS_PPO_SMOKE_PASS]`,
`[TASTE_T8_GLOBALGCE_SMOKE_PASS]`, and
`[TASTE_T9_COMRECGC_SMOKE_PASS]`. Each scientific implementation stores the
same already-bracketed value in structured `marker` fields and prints that
value unchanged. A method-specific `PASS` leaf contains the same bytes plus
exactly one trailing newline. Managed-v2 publication retains its generic outer
marker; the method marker remains inside independent verification. T7 is not
changed because its successor is owned by the concurrent NeuroSED/GCF work.

### Consequences

- Controller log-marker pins can use the same literal as scientific evidence.
- Consumers reject legacy marker spellings and bracket-normalization tricks.
- This change authorizes no release, deployment, or experiment.

### Status

Implemented in code, focused tests, and operator documentation; all three
science routes remain unrun and release disabled.

## [2026-08-28] Freeze managed Taste release v3 and move T4 to GPU 1

### Motivation

The historical T4 path consumed an older T3 adoption layout and let the
science process write its own terminal marker. The
fresh T3 result is instead a managed-v2 publication containing a newly fitted
validation-only temperature under `artifacts/checkpoint`, and the active Taste
schedule assigns the bounded oracle smoke to physical GPU 1. The previous
authority draft also lacked a complete external launcher trust root, phase
barrier, and auditable GPU lease lifecycle.

### Decision

Add a successor which descriptor-retains the exact managed T3 tree, verifies
its generic gate and nested scientific PASS, and loads only its selected GINE
checkpoint plus graph-cache `manifest.json` and `calibration.pt`. Bind physical
GPU index 1, its UUID, `CUDA_VISIBLE_DEVICES=1`, and visible `cuda:0`. Run the
fixed sixteen-parent/four-deletion smoke with full three-class probability and
batch/single checks, real connected deletions, invalid/full-parent controls,
and observed Sweet-to-Bitter and Sweet-to-Tasteless flips.

The worker emits only aggregate candidate documents. A different method
verifier repeats the science, compares the SEALED evidence, and is the only
caller of the managed-v2 terminal publisher. PASS remains generic managed-v2
evidence with method semantics nested in independent verification. Attempt,
staging, and generation identities are UUIDs; input authorities remain held;
ABA or lineage drift quarantines without signals; automatic child termination
is false.

The compatibility `taste_main_v2` files implement the managed Taste
release-v3 foundation, not the complete `main_completion_v4` scheduler. A
distinct supervisor registers and independently verifies the controller child
before publishing an immutable launcher receipt. Heartbeats, activations,
renewals, and releases are append-only chains. The outer T4 runner holds the
canonical GPU UUID lock continuously through scientific worker, SEALED
handoff, independent verifier, and terminal publication. T4 is fixed to GPU 1;
the reserved NeuroSED authority lane is GPU 2. GPU 0 and GPU 3 remain protected.

### Consequences

- T4 opens no train, validation, test, or CSV payload and publishes no SMILES,
  molecule IDs, per-example predictions, graph payloads, or paper-cell claim.
- The model is constructed once per scientific process; the independent
  verifier deliberately performs its own separately loaded replay.
- The prior T4 implementation remains historical and is not a valid successor
  terminal for this route.
- This code change performs no deployment, science run, HPC submission, or GNN
  ablation; those require the clean committed execution identity.

### Status

Implemented and locally tested; AutoDL deployment/execution remains pending.

## [2026-08-28] Freeze the release-disabled TasteMolNet T9 COMRECGC core

### Motivation

TasteMolNet T9 needs a bounded native COMRECGC smoke without reviving the
binary-classifier assumptions used by earlier routes.  The implementation
candidate must preserve the official random-walk, candidate-frequency, and
common-recourse mechanisms while using the same frozen three-class GINE as the
other Taste stages.  Freezing those scientific semantics separately from the
managed execution runner keeps review of the algorithmic boundary distinct
from authorization to run it.

### Decision

Freeze the release-disabled T9 core around the following boundaries:

- score native walk importance as `1 - p[:, 1]`, but classify a candidate only
  when the three-class argmax is not Sweet (`1`);
- identify native states by a canonical attributed-graph SHA-256 that excludes
  GINE embeddings, Python `hash()`, parent metadata, and lineage, while keeping
  lineage as separate native state;
- retain the official serial stateful heads, frequency collection, DBSCAN,
  coverage, and greedy common-recourse selection at the exact smoke settings,
  including `M=500`, a durable checkpoint/reload after completed step 250,
  eight train-only predicted-Sweet sources, and seed 7;
- load the seven executable files from official COMRECGC commit
  `122f9341a360e9f06bb58a2f5823bb596021f6bf` only through retained file
  descriptors whose SHA-256 values equal the checked-in reviewed map; and
- expose a strict aggregate-only seven-file terminal consumer ending in
  `[TASTE_T9_COMRECGC_SMOKE_PASS]`, with no molecule rows, SMILES, graph payload,
  or checkpoint payload persisted.

The tracked release config remains a native false with null mutable pins.  The
CLI can strictly validate an existing result while disabled, but its science
branch intentionally names a not-yet-implemented
`run_tastemolnet_comrecgc_smoke`.  A later integration must acquire and retain
the reviewed managed ACTIVE authority, T2 receipt-only authority, T3/T4 stage
authorities, frozen checkpoint, train CSV, official checkout, GPU-2 lease, and
fresh output authority through the last fallible pre-PASS callback.  It must
also register the `taste_t9_v1` strict result dispatch and controller task
before a separately reviewed one-parent release successor may fill any pin.

### Consequences

- The frozen core is testable without authorizing AutoDL science or presenting
  the stage as runnable.
- A self-signed upstream checkout, altered smoke parameter, binary candidate
  shortcut, identity based on model embeddings, incomplete midpoint restore,
  or row-level output fails closed.
- GPU-2 lane ordering after T8 remains a controller scheduling dependency;
  T9's scientific predecessors remain T2, T3, and T4.
- Full-budget `M=50000`, deployment, controller mutation, and scientific
  execution are outside this freeze and remain pending.

### Status

Local release-disabled core, strict consumer, static release perimeter,
focused/adjacent tests, and documentation only.  The managed runner, result
dispatch, controller integration, immutable release receipt, deployment, and
science have not been implemented or performed.

## [2026-08-28] Fail closed on T8 self-signed consumption and cached official code

### Background

Fresh review found that the real official generator did not expose the two
resume/completion parameters already required by the T8 protocol, public
consumption derived all expectations from the terminal itself, and Python
could reuse a preloaded `models.*` closure. It also confirmed that the
inherited zero-link marker relink is not a viable Linux publication primitive.

### Decision

Make the real generator signature exactly carry the planned checkpoint and
completion callback surface, validate both checkpoint evidence and callback
types before execution, and keep the completion callback at one normal-return
boundary. Public terminal holding now requires a live independent authority
provider and exact full-authority equality, including managed task/run/GPU,
ACTIVE/child, T2--T4, GINE, official, train, and policy bindings.

Before every official import, validate any predecessor `models.*` closure
against exact held source origin/inode/hash evidence, remove it, and import a
fresh closure with bytecode writes disabled. Reject bytecode/native shadows,
and make the Git/source holder reject untracked or ignored files in the
official runtime source closure.

### Consequences

- A self-consistent rehashed terminal is no longer consumable without the
  independently held pre-publication authority.
- Two target branches still use the same three-class GINE and the existing
  canonical merge plus untargeted strict-flip gate; no binary/RF route is
  introduced.
- `--validate-only`, release, deployment, and science remain disabled until a
  reviewed managed COMPLETION/ACTIVE adapter exists.
- T8 does not adopt the rejected zero-link marker primitive. A corrected
  permanent-authority final-hardlink design remains an integration blocker.

## [2026-08-28] Implement T8 as a resumed two-target native GlobalGCE smoke

### Background

The reviewed three-class foundation made one calibrated Taste GINE usable by
GlobalGCE without collapsing the oracle to binary labels. T8 still required a
real bounded execution boundary: both non-Sweet destinations had to exercise
native training and resume, final acceptance had to use the original
three-class order, and no molecule-level output could cross the terminal
boundary. A wrapper PID, a generic required-file check, or a caller-provided
GPU index is not sufficient execution authority.

### Decision

Implement two independent official GlobalGCE branches with Sweet source label
1 and target labels 0 and 2. Both consume identical descriptor-authorized
bytes for the frozen seven-file calibrated GINE and prepared train CSV. Each
branch intentionally stops only after its durable epoch-0 checkpoint, then a
second native call restores the same model, optimizer, scheduler, and
Python/NumPy/Torch RNG identity and completes the bounded run. The apparent
validation cohort inside GlobalGCE is a deterministic branch-local partition
of the already authorized train rows; T8 never opens a dataset validation,
calibration, or test payload.

Each freshly created target directory is retained by descriptor. Its epoch-0
checkpoint directory, checkpoint leaf, and heartbeat leaf remain physically
held across the deliberate interruption and reload; the native loader checks
and deserializes that exact checkpoint inode. The terminal branch tree is
captured inside the generator completion callback, before control returns to
path-based orchestration. Same-byte leaf swaps and target-directory swaps are
therefore rejected instead of being adopted as resume or terminal proof.

Merge branch rule catalogs by exact canonical LHS/RHS action content, apply
the established attachment-aware native rewrite, and canonical-deduplicate
generated residuals. Reload the same frozen GINE in original Bitter/Sweet/
Tasteless order and accept only `pred_before == 1 and pred_after != 1`, with at
least one accepted strict flip attributable to each target branch. Two binary
classifiers, native GTGNN fallback, RF, BACE adapters, heuristic fallback,
per-example terminal evidence, and data redistribution are rejected.

The following predecessor terminal design is retained here as decision history
and is superseded by the managed-v2 decision later in this file. Its terminal
authority was exactly six files: `input_hashes.json`,
`state.json`, `manifest.json`, `gate.json`, `output_hashes.json`, and `PASS`.
The first five are canonical hash-closed aggregate evidence; `PASS` contains
`[TASTE_T8_GLOBALGCE_SMOKE_PASS]` and is the final no-replace publication syscall.
The producer performs no fallible validation, fsync, pathname reopen, cleanup,
or logging after that commit. Downstream adoption must use the strict retained
consumer, which rejects unknown files and physical replacement even when bytes
are unchanged.

In that predecessor design, T8 consumed only the receipt-only T2 adoption plus held T3 and T4 outputs that
all bind the same checkpoint, along with held train, checkpoint, policy, and
pinned official-source authorities. Its managed execution literals are
`taste_t8_gpu2_v1`, `tastemolnet_t8_globalgce_smoke`, and `taste_t8_v1`, with
predecessors T2/T3/T4 and exclusive physical GPU 2 exposed alone as logical
`cuda:0`. The implementation deliberately imports the reviewed managed-child
API; it does not create a controller, lease, or receipt.

### Consequences

- The checked-in release config and AutoDL wrapper remain disabled, so no
  caller-selected path, GPU probe, output, or science is touched by default.
- The paired Slurm wrapper is CLI parity only and exits unconditionally before
  Python because this Taste route is AutoDL-only.
- Private branch state may contain rules/checkpoints needed for resume, but the
  public root contains aggregates only and cannot redistribute train rows,
  SMILES, IDs, rules, or per-example predictions.
- A later reviewed one-parent release successor must pin the immutable
  implementation, exact predecessor/input identities, official source,
  managed GPU2 child, and fresh state/output parents, then require strict
  consumer success before controller adoption.

### Status

Accepted for implementation and local no-cache tests only. Two integration
blockers are explicit: this base lacks the reviewed
`PreparedTerminalOutput.commit_final_rename`, and it lacks a managed
`taste_t8_gpu2_v1` registry row plus `taste_t8_v1` completion dispatch. The
release config/wrapper remain disabled; independent integration review,
release, deployment, and science remain separate gates.

## [2026-08-28] Adopt the existing Taste GINE calibration and smoke only the calibration cache

### Background

The formal three-class TasteMolNet GINE already fits one positive scalar
temperature on validation inside T2 and publishes the logits, calibration
metrics, model/last-checkpoint closure, and reload evidence in one immutable
bundle. Reusing the BACE B4 implementation would incorrectly require an
uncalibrated checkpoint, refit the same validation logits, and copy the model
bundle. Reusing BACE B5 would also read a CSV and publish molecule-level
prediction/deletion rows.

### Decision

Add a separately hashed downstream policy that remains bound to active policy
v2 and authorizes the two implemented AutoDL stages plus one typed future T6
boundary. T3 verifies and adopts the existing
validation fit in place: it recomputes raw/calibrated NLL, ECE, Brier score, and
argmax invariance from the immutable bundle's validation predictions, never
calls a fitter, never copies a checkpoint, and proves the source inventory is
unchanged. T4 is frozen to physical GPU 1 and one loaded `model.pt` oracle. It
opens only the graph-cache `manifest.json` and authenticated
`calibration.pt`, chooses the first sixteen true/predicted Sweet parents with
exactly four connected one/two-atom deletions, applies the shared three-class
strict-flip semantics, and requires observed flips to both Bitter and
Tasteless. T3 is CPU-only and claims no physical GPU ownership.

T4 writes aggregate private evidence only: an opaque ordered-cohort digest,
position-only deletion counts, aggregate flip destinations and probability
drop statistics, policy/provenance/access documents, a controller-facing gate,
and complete SHA closure under `[TASTE_MULTICLASS_ORACLE_PASS]`. It writes no
CSV, SMILES, molecule identifiers, or per-example predictions. Train,
validation, and test cache payloads remain unopened; test remains
metadata-hash-only. The paired Slurm file is static CLI parity and exits before
science because this campaign is AutoDL-only.
Checkpoint, cache, and stage-output child reads are anchored with retained
directory descriptors and `openat`; temporary root swap-and-restore attempts
fail closed instead of redirecting temperature, model, or gate reads.
The successor additionally retains every directory component, restricts T3/T4
outputs to exact direct fresh children of the AutoDL Taste/GINE/seed-7 artifact
root, and revalidates all source/policy/output authorities after publication.
It exposes held stage and held T2-checkpoint APIs with exact checkpoint path,
ID, full hash inventory, stat inventory, and manifest SHA evidence so a later
consumer cannot substitute an equal-byte copy or symlink alias while loading.

The supplemental policy explicitly authorizes `T6_OURS_SMOKE` only as a
train-only, minimum-five-step PPO smoke that consumes only the frozen prepared
train CSV, with frozen-GINE reward, immutable T3/T4/T5 predecessors, no RF, and
no validation/calibration/test payload. This is authority for a later
implementation, not a T6 implementation or launch.

### Consequences

- T3 means existing-fit adoption/verification, never a second calibration fit.
- The selected inference asset remains the best `model.pt`; `last.pt` remains
  terminal/reload evidence rather than replacing the selected model.
- T4 is a bounded interface and multiclass-semantics smoke, not a method
  efficacy result; its PASS nevertheless requires both authorized strict-flip
  destinations to be exercised by real connected deletions.
- The standalone T3/T4 gates can be consumed by the main controller only after
  an explicit predecessor-gated controller action is released separately.
- T6 must retain and revalidate the public stage/checkpoint/policy authorities
  across its model/reward load and output publication; this change does not add
  T6 runtime or controller integration.

### Status

Accepted for code/tests; remote deployment and science remain separately gated.

## [2026-08-28] Reject the first Taste T5 initializer freeze and require a physical successor

### Rejected candidate evidence

The first isolated T5 candidate was reviewed at base commit
`3a90fd8697b58bad4f95f3be9347b327d5c51043`, staged tree
`ed84fa2f4cb57b0f5f45861e54da144c8489c70f`, and full-index binary patch
SHA-256 `faa5eebeb2ac76ac5c503e5c2ebb3b195fd15d05636245f8eefe7464d2cbbe0e`.
That identity is retained here as rejected evidence and must not be released.

### Reason

The candidate accepted arbitrary non-empty bytes as safetensors, did not bind
the T3/T4 gates back to the common GINE checkpoint, exposed path-based model
and adapter loads outside its retained file-descriptor authorities, did not
bind the published name to the retained staging inode, mislabeled the bare
base hash as the reference-policy hash, and asserted GPU-lock ownership
without a physical execution receipt.

### Successor requirement

The successor must parse and reload the real zero-step LoRA, consume the typed
T3/T4 held interfaces, keep source/adapter paths descriptor-backed throughout
loads, publish and validate through the retained staging inode, distinguish
base-model and base-plus-adapter identities, and describe controller identity
as declaration-only until a separate execution-receipt contract is reviewed.
No production pin is enabled by this local successor work.

### Successor resolution

The local successor now performs real safetensors parsing, exact LoRA A/B
key/rank/shape/dtype/finite checks, zero-B proof, and a fresh-base PEFT reload.
It consumes the descriptor-held typed T3/T4 APIs and binds their gates, root
inventories, full/stat/SHA checkpoint inventories, config, feature, label, and
temperature identities back to one common GINE. Source, adapter, and published
roots retain both descriptors and full inode/ctime inventories, so
swap-load-restore does not disappear merely because bytes are restored.

Publication binds the named parent/staging/adapter inodes, validates external
authorities after no-replace rename, and writes PASS last. The reference policy
uses a distinct canonical base-plus-adapter identity. T6 receives a combined
held-load token and verifies both loaded policy/reference PEFT states. GPU state
is only `controller_declared_only`; there is no lock-ownership claim, and the
public builder stops on
`RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT` before opening
an authority or model. A later physical execution-receipt contract requires a
new review.

## [2026-08-25] Separate GREED scan science state from controller receipts

### Background

The complete AIDS GREED theta-close scan traverses 91,916,686 pairs. Generic
controller outputs are immutable and attempt-qualified, so a worker, host, or
controller loss could not safely resume the scan's existing checkpoint without
either violating no-clobber launch semantics or discarding hours of verified
work.

### Decision

Keep one campaign-owned `pair_semantics_science` root with no attempt token and
make the full task's `expected_output` a fresh
`pair_semantics_receipt/attempt-{attempt}` root. A dedicated CPU supervisor owns
an inode-bound flock adjacent to the science root. Attempt zero launches the
reviewed child without `--resume`; one transient child retry or a later
controller attempt may resume only after checkpoint identity, committed prefix,
PID generations, root/lock inodes, PASS absence, source hashes, and writable
FD/mapping absence are revalidated. Resume authorization is two-phase so a
pre-spawn supervisor crash does not consume the single allowance. Semantic and
provenance markers are terminal and remain semantic-first even when the process
also exits by signal.

At completion, rehash the terminal science manifests and large arrays, freeze a
terminal supervisor manifest, and write its PASS last. Fresh receipts bind that
immutable manifest rather than mutable state/events. The close-view task uses
the dependency receipt JSON as `input_manifest`, validates the receipt through
the fixed science-root inode and exact terminal file hashes, then reads the
fixed-root contract and distance array. Enable one generic transient retry only
for fresh receipt publication; the science root itself never gains an attempt
suffix.

### Consequences

- Host/process loss can produce receipt attempt 1 without recomputing an
  authenticated scan prefix.
- Receipt, terminal-manifest, lock-inode, science-inode, artifact, symlink,
  partial/final coexistence, or live-writer drift fails closed.
- The controller and Slurm entrypoints remain thin; the Slurm wrapper is parity
  evidence only and explicitly refuses HPC execution.

### Status

Accepted for immutable release pinning; deployment remains separately gated.

## [2026-08-24] Adopt the completed AIDS physical snapshot into a fresh route

### Background

The corrected `pair_order_v1` controller completed and froze its 25 GB physical
snapshot, then its science task failed before DBSCAN progress.  The failure was
not scientific: a procfs guard searched the decoded command-line string and
mistook a read-only `bash` diagnostic containing the literal entrypoint name
for another common-recourse worker.  Recopying the already closed snapshot
would waste persistent capacity and create a second unnecessary data authority.

### Decision

Publish a new controller/root, never resume the failed science attempt.  Its
middle task adopts the existing snapshot by read-only reference.  Authority is
bound first to the exact persistent `control_root/four_methods_four_datasets_continuation`
namespace, then to the controller-manifest SHA and the task gate derived from
that namespace.  The gate must contain one `main` PASS attempt zero with the
exact output.  Snapshot/DBSCAN/pair
manifest SHA values, and pair/vector SHA values.  Adoption performs the same
full source/destination/stat/writer/partial closure as snapshot publication,
writes only a small fresh adoption manifest, and publishes PASS last.  Science
repeats that adoption validator before spawning its child.  It never copies,
hardlinks, rewrites, or opens the old snapshot for writing.

Procfs classification now parses raw NUL-delimited argv.  A worker is visible
only when argv is a direct physical entrypoint or a CPython interpreter plus
valid interpreter flags and the first script operand.  Relative operands are
resolved against that process's `/proc/<pid>/cwd`.  Shell/grep/regex literals,
`python -c`, `python -m`, and later arguments to another script are ignored;
real absolute/relative/direct workers remain visible.  Once the first script
operand has the exact entrypoint basename, missing, symlinked, or unreadable
identity fails closed instead of disappearing from the process set.

### Consequences

- The earlier selector/snapshot PASS artifacts remain immutable evidence; its
  failed science output is never reused.
- The new science output and controller namespace are fresh, CPU-only, and
  continue the 128 GiB cgroup / 96 GiB RSS / global high-memory handover gates.
- Mut requires another fresh continuation bound to the new authoritative AIDS
  controller manifest SHA and attempt-zero terminal output.

### Status

Accepted for code/review; deployment still requires an immutable pinned commit.

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

## 2026-08-24: Bind AIDS v5 to the integrated equivalent of the reviewed DBSCAN core

- The independently reviewed exact-shortcut source commit `645c6e51...` was
  cherry-picked into the production lineage as `8c371b1c...`; the former is
  not a Git ancestor of the v5 execution branch.
- The v5 builder now requires `8c371b1c...` as the true ancestor and separately
  freezes the exact Git blob identities and current SHA-256 values of
  `external_memory_dbscan.py` and its focused test.
- The manifest records both commit identities and per-file evidence.  This is
  an explicit tree-equivalent integration gate, not a silent commit
  substitution, and drift in either integrated blobs or working content fails
  closed.

## [2026-08-24] Monitor unviable AutoDL work without taking task ownership

The root-cause acceleration continuation adds a read-only persistent monitor
whose spec freezes external PIDs, procfs start ticks, commands, output roots,
progress probes, and GPU indices. It classifies progress as progressing, slow,
unviable, stalled, exited, or observation-failed and records rolling throughput
and ETA every 60 seconds. These states are diagnostic only: the monitor owns
no scientific process, takes no GPU lock, never emits a cell PASS, and never
signals a task. Existing component controllers remain the only launch owners.

## [2026-08-24] Adopt failed-v5 GlobalGCE mining only through an exhaustive-v2 proof

The failed BACE GlobalGCE v5 controller cannot become PASS merely because its
19 gSpan roots and SQLite payload exist. V6 may reuse those bytes only through
the `globalgce_gspan_exhaustive_v2_adoption_v1` proof, which opens SQLite with
`mode=ro&immutable=1` and binds the failed task/root, train-only cohort, frozen
GINE, official commit, traversal order, 19 roots, stable top-20, source bytes
and stats, sidecars, and no-writer closure. Any unclosed identity selects an
explicit `fresh_exact_top_k_v2` route and consumes no v5 pattern bytes. V5
remains immutable and FAILED.

## 2026-08-24: Diagnose COMRECGC generation divergence from JSON traces first

- A failed legacy/optimized payload-count gate must still publish the earliest
  actionable selected-transition and candidate-lineage difference.
- The diagnostic reads only hash-bound JSON/JSONL trace artifacts; it does not
  unpickle a failed scientific payload.
- Python hashes of raw GNN embedding bytes are reported separately from stable
  canonical graph SHA-256 differences. Official-hash drift alone cannot be
  called a structural counterfactual difference.
- The diagnostic is evidence only and is permanently paper-ineligible. It
  cannot release an optimized 50k run or weaken the formal M500 gate.

## 2026-08-24 — Diagnose GCF raw-byte identity before optimized replacement

- Quick-50/100 records a read-only lockstep trace at every restart, importance
  call and move. It binds RNG state, canonical ordered neighbours, exact
  frozen-GINE batch tensors and row outputs, coverage and selected transition,
  so a failed replay identifies the first field rather than only a final
  candidate digest.
- Frozen-GINE inference uses one shared ordered-batch scorer. Its optional
  cache is whole-batch-only; a miss scores the original duplicate-preserving
  batch. Partial-row reuse, deduplication and chunking remain forbidden.
- CPU/GPU and repeated-cold byte identities are diagnostic evidence, not an
  equivalence waiver. Allclose output cannot be promoted to exact VRRW
  equivalence when it changes an official raw embedding hash.
- Existing 50k writers remain untouched. Replays use immutable commits and
  fresh roots; optimized full remains ineligible until all exact gates pass.

## 2026-08-24: Separate deterministic GINE identity from NeuroSED execution

- COMRECGC's official graph key hashes the raw frozen-GINE graph embedding.
  Repeated identical CUDA batches can differ in low bits because graph pooling
  uses nondeterministic CUDA scatter reductions; those bytes must not be
  mistaken for an RDKit/order/RNG change.
- The route accepts separate classifier and distance devices. A replay may
  keep the frozen GINE on CPU for byte-stable graph identity while keeping
  NeuroSED on CUDA. Both A/B roles must use the same split contract, which is
  part of the scientific command/config identity.
- Existing roots and running 50k processes are never migrated. This is not an
  equivalence waiver: formal M500 still requires exact graph order, trace,
  lineage, payload, checkpoint, and serialization parity.

## 2026-08-24: Run GCF deterministic replay on CPU after CUDA raw-byte drift

- A repeated-cold audit established that frozen GINE CPU hidden/logits are
  byte-exact while CUDA hidden/logits vary at low bits for the same full
  ordered batch, despite identical labels and allclose logits.
- The deterministic diagnostic route is CPU-only with CUDA hidden. It requires
  legacy-A/legacy-B and legacy/patched lockstep at Quick-50 before Quick-100,
  and then M500. CPU evidence cannot replace the protected 50k GPU route
  without later ETA and semantics gates.
- The fixed batch matrix 1/8/32/128/512 separates collation, prepared-batch
  model time, end-to-end time, argmax/allclose, calibrated probability, and
  raw-byte identity. It is launched by a persistent deferred controller only
  after the controller-owned GPU2 pair worker releases its UUID lock.
## 2026-08-24: Snapshot the promoted AIDS pair store into fresh physical inodes

- The protected repair-v4 DBSCAN process retains an `O_RDWR` mapping of its
  already-promoted pair arrays.  Exact-route v5 must neither signal that
  process nor consume its writable inode directly.
- A dedicated CPU-only snapshot stage now brackets a sequential physical copy
  with full source SHA-256/stat closure.  Each destination is copied through a
  non-authoritative `.partial`, synchronized, atomically promoted, then
  reopened and hash/schema checked.  Source and destination device/inode pairs
  must differ, so source hardlinks are forbidden.
- Same-root recovery may discard only an incomplete partial.  Already promoted
  arrays are adopted only after their complete content, size, schema, and
  distinct-inode proof pass.  Terminal publication is PASS-last and reuses the
  same whole-closure validator after a manifest-to-PASS crash window.
- The old process is required when the builder freezes its generation, but a
  queued snapshot accepts its verified natural exit before the first copy.
  PID reuse and other common-recourse processes remain terminal errors.  A
  fixed-name regular partial is removed and its directory fsynced before the
  resume headroom calculation; symlink/non-regular partials and any partial
  beside a promoted final remain corruption.
- The snapshot contract freezes all 91,916,686 candidate-major/parent-minor
  rows and the 64-dimensional float32 vectors.  `pair_indices` is row
  provenance, never a precomputed distance/adjacency graph.  The downstream
  DBSCAN contract remains Euclidean `eps=0.02`, `min_samples=3`, includes the
  sample itself, and follows frozen sklearn 1.7.2 border/label ordering.
- The old v4 root stays read-only and receives no signal.  Any source drift,
  unexpected writer, symlink, partial, insufficient persistent-disk floor, or
  terminal artifact drift fails closed.
## 2026-08-24 — AIDS physical snapshot freezes pair-index column semantics

- The promoted AIDS pair store is candidate-major by row, but each
  `pair_indices.npy` row is stored as `(parent_index, candidate_index)`.
- The exact production identity is therefore
  `parent_index = row % 1283` and `candidate_index = row // 1283` for all
  91,916,686 rows.  Snapshot publication checks every row in bounded blocks.
- The first fresh v5 controller correctly failed closed before copying because
  its preflight assumed the opposite column order.  That controller/root remain
  immutable failure evidence; a corrected route must use a fresh namespace.

## [2026-08-25] Recompute the AIDS logical close-pair view from frozen GREED

### Decision

Treat the 91,916,686-row pair/vector array as a physical store, not by itself
as the logical DBSCAN input. Recompute every normalized scalar distance through
the frozen AIDS GREED `torch.cdist` adapter and official element-count scale,
then materialize only the inclusive `distance <= theta` predicate in a small,
separately named bitmap. Recourse-vector norms remain a consistency audit and
never become the filtering authority.

Bind physical-snapshot wrappers to their hash-closed source chunk metadata,
verify `(parent_index, candidate_index)` columns for every scanned row, and
publish `PASS` last. A bounded benchmark remains resumable but non-terminal.
When all rows are close, publish a separate exact-consumer-compatible
`comrecgc_all_pairs_close_certificate_v1`; otherwise the bitmap is the sole
logical close-view contract.

The nominal official candidate cap comes from generation `--k` and
`MAX_COUNTERFACTUAL_SIZE`. The current common-recourse `--cf_size` parameter is
not claimed as an applied official slice. The project post-predicate slice is
recorded explicitly as `PROJECT_EXTENSION` and whether it was binding.

### Consequences

- No 25 GB pair store is regenerated or copied by this audit.
- Exact DBSCAN cannot consume the physical Cartesian row count unless the full
  frozen-distance certificate proves every row is theta-close.
- Benchmark output cannot be mistaken for a scientific PASS.

### Status

Accepted for implementation; production completion requires the immutable
AutoDL execution and full scan.

## 2026-08-25: Adopted Cartesian COMRECGC stores require a theta-close authority

- Treat the 91,916,686-row AIDS artifact as a physical Cartesian store, not as
  proof that every row is a logical DBSCAN sample. Adoption fails closed unless
  a hash-closed `normalized_distance <= theta` view is supplied; a complete
  `ALL_PAIRS_CLOSE` certificate permits zero-copy use of the physical mmap.
- Default partial-close storage is a bitmap plus physical row indices. It is
  not eligible for the current path-based DBSCAN engine unless a declared disk
  budget permits compact selected rows; this prevents an implicit second
  approximately 25 GiB vector copy.
- The close view binds the pair-semantics authority and physical vector/pair
  identities. Terminal snapshot and close-view inputs may coexist, while an
  adopted Cartesian terminal without the view is rejected.
- A successful all-core shortcut must publish separate all-core,
  connectivity, boundary, and partition certificates. The one-cluster
  downstream summary remains streaming and records official float32 and
  stable float64 centroid/radius decisions without retained-vector copies.
- These changes authorize no live deployment or process signal. Production
equivalence remains gated on the complete GREED scan and closed production
subsets.

## [2026-08-25] Separate restartable GREED science from fresh controller receipts

### Decision

Keep the production full GREED scan in one fixed campaign science root while
each generic controller attempt owns a fresh receipt root. The supervisor
recoverably promotes and inode-freezes the empty science root before the first
child, holds and revalidates one descriptor-backed flock, resets inherited
SIGTERM handling before Linux PDEATHSIG arming, and permits only one
checkpoint-authenticated same-root resume. Semantic or provenance evidence is
terminal and always takes precedence over process-loss recovery.

Terminal adoption reopens the float32 distance and uint8 bitmap arrays in
bounded mmap blocks, proves shape/dtype/nonnegative finite domain and the exact
`distance <= theta` predicate, and reconciles physical/finite/close counts with
both science manifests. Resume metadata is frozen in the terminal supervisor
manifest; the fresh receipt copies only that frozen authority and publishes
`PASS` last.

Downstream Mut no longer assumes controller attempt zero. Its v5 dependency is
the exact controller manifest plus task gate; the builder resolves one unique
PASS state/run, validates the recorded attempt output against the manifest
template, and hash-binds the gate, state, output, and terminal evidence.

### Consequences

- A dead worker or host may consume generic attempt one without changing the
  science inode; a forged replacement root, lock, bitmap, count, or receipt is
  rejected.
- Terminal success revalidates the held flock descriptor against its named
  inode after long science validation, after terminal freeze, and after receipt
  publication. A post-publication mismatch revokes receipt `PASS` before the
  task can return success.
- Existing global dead-worker recovery remains available, but a semantic marker
  in the persisted run log prevents that retry.
- No deployment, process signal, or mutation of historical AutoDL roots is
  authorized by this change.

## [2026-08-25] Separate AutoDL controller liveness from scientific route viability

### Decision

The read-only root-cause monitor retains its monitor-v1 `health_state` and
`pid_alive` fields for compatibility, but now also publishes four unambiguous
observations: `controller_process_alive`, `scientific_worker_alive`,
`scientific_progress_state`, and `route_viability`.  A live controller is
therefore not evidence that a scientific worker is alive, progressing, or on a
viable route.

`RUNNING_PROGRESSING`, `RUNNING_SLOW`, `RUNNING_UNVIABLE`, and
`RUNNING_STALLED` map respectively to `VIABLE`, `SLOW`, `UNVIABLE`, and
`STALLED` route viability.  A missing worker or failed probe maps to `UNKNOWN`,
never PASS.  `SUPERSEDED` is emitted only when a new monitor spec pins the
SHA256 of a physical no-symlink JSON handover receipt.  That receipt must prove
graceful checkpoint and stop completion, old-worker exit, `sigkill_used=false`,
and a replacement task gate in `PASS`.  It binds the old task ID, PID,
start-ticks, and output root plus the replacement controller/task/output root,
controller-manifest hash, task-gate hash, and final-manifest hash.  The monitor
rechecks the physical replacement files and every receipt field, rejects
zombie/terminal `/proc` states as live, and refuses `SUPERSEDED` while the
frozen old worker generation is still alive.

Receipt booleans use exact JSON boolean identity (`true`/`false`); numeric
`1`/`0` are rejected. PID and start-ticks use positive JSON integers only, so
booleans and integral floating-point values cannot impersonate process
identity fields.

The monitor does not probe a potentially retired progress file after this full
proof, and it never signals a scientific process.

### Consequences

- The dashboard/status payload can report process ownership and scientific
  health independently without changing existing consumers of `health_state`.
- Graceful retirement remains an operator/controller action outside this
  monitor; this code records the completed action but cannot initiate it.
- Existing monitor specs remain valid because `supersession` is optional.
- A missing/mutated receipt, symlink, non-JSON file, hash mismatch, incomplete
  stop proof, non-PASS replacement gate, or still-live worker fails closed as
  `OBSERVATION_FAILED`/`UNKNOWN`.

### Status

Accepted for the next immutable AutoDL controller release; not a deployment or
permission to stop any existing worker.

## [2026-08-25] Recover exact DBSCAN partitions from disconnected adaptive anchors

### Background

The AIDS adaptive certificate selected the complete union of three minimum-norm
seeds and every first-pass seed-lower-bound failure.  Its 266-anchor epsilon
graph had three exact components, so the former single-anchor-component check
stopped before scanning whether a nonfailure core row connected those
components.  This is an inconclusive single-cluster shortcut, not evidence that
the full DBSCAN graph is disconnected.

### Decision

Retain the complete authenticated seed/failure ledgers and promote every
failure row to its already-frozen anchor.  Recheck the entire anchor graph in
float64 and require every failure anchor to have at least `min_samples`
neighbors including self.  Every nonfailure row is independently core by the
complete seed lower bound and touches at least one seed.  First stream all rows
against every seed plus one deterministic nearest-to-seed boundary anchor per
non-seed anchor component.  A nonfailure row within inclusive epsilon of
anchors in two components is an exact core bridge witness; record both edges,
their float64 distances, and the component union in source-row order.

If those sufficient witnesses connect every component, publish the one-cluster
partition without scanning the remaining anchors.  Otherwise perform one
complete exact all-anchor component scan.  Because all failure rows are
anchors and all other rows are seed-attached cores, the resulting component
unions are the complete DBSCAN core partition: a component without a witnessed
nonfailure edge is a genuine separate cluster.  Number clusters by their
minimum global core row, matching sklearn's ordered all-core traversal.  This
route has zero noise; fixtures containing noncore failures use the ordinary
three-pass exact engine only below the frozen small-sample gate.  A large
noncore case stops with `EXACT_DBSCAN_GENERAL_EXTERNAL_REQUIRED` and never
silently starts the old quadratic brute route.

Both bridge passes are resumable through source-identity-bound forward hash
ledgers.  Every queried membership must agree elementwise between the frozen
sklearn brute kernel and direct float64 recheck.  Publish separate all-core,
connectivity, boundary, and canonical-partition certificates plus attachment,
component-root, bridge-witness, labels, and core-mask hash closure.  Terminal
reopen reconstructs labels and re-evaluates every retained bridge edge from the
immutable vector source.

### Consequences

- A disconnected anchor subgraph no longer falsely terminates an otherwise
  provable exact all-core result.
- The targeted production path costs one full scan against five anchors in the
  observed three-component case; the 266-anchor scan is reserved for a failed
  targeted connection proof.
- Exact multi-cluster outcomes remain reportable rather than being coerced to
  one cluster or approximation.
- Historical failed roots remain immutable.  Selection adoption, the reviewed
  downstream boundary chain, controller DAG, and any AutoDL launch require a
  separate fresh immutable release and independent review.

### Status

Accepted for implementation and focused review; not yet deployed.

## [2026-08-25] Keep production-subset DBSCAN evidence distinct from full proof

### Decision

Add a CPU-only AutoDL audit that derives five deterministic, hash-closed AIDS
inputs from the authority-bound theta-close view: first, seeded random, dense,
sparse, and theta-boundary subsets. Preserve production logical-row order and
bind the physical pair/vector/distance sources, bitmap, selection indices,
seed, pivot, and canonical partition rule.

For every subset, require exact agreement between sklearn DBSCAN and the
general external-memory engine for core mask, noise, partition, centroid,
strict centroid-radius membership, parent coverage, and stable-tie greedy
selection. Attempt the all-core certificate with fallback disabled; an
inconclusive certificate is recorded as not applicable, never as PASS.
The NumPy production trace casts `delta` to the distance-array dtype before
strict comparison, matching upstream Torch scalar promotion at exact float32
boundaries instead of widening the distance to Python float64 first.

### Consequences

- Subset evidence can catch production schema, ordering, border, numerical,
  coverage, and greedy regressions before full clustering.
- `PASS` is terminal and last, but explicitly carries
  `full_production_dbscan_equivalence_claimed=false`.
- The synchronized Slurm wrapper always refuses execution because this audit
  is AutoDL-only and the current campaign forbids HPC use.
- No active execution root, pair store, controller, GPU lock, or old brute
  process is modified by this implementation.

### Status

Accepted for code review; production execution remains a separate gate.

## [2026-08-25] Gate c766 one-cluster adoption on exact radius-mask replay

### Decision

Future one-cluster lineage traces cast the Python `delta` scalar to the NumPy
distance dtype before applying strict `<`, matching upstream Torch behavior at
an exact float32 boundary. Add a read-only post-hoc gate for the already
running immutable c766 route rather than changing or restarting it.

The gate binds the terminal summary, source vectors and pairs, saved centroids,
and terminal block size. It simultaneously replays the historical widened
NumPy mask, corrected dtype-cast NumPy mask, and official Torch mask, then
compares mask digests, differing rows, parent/candidate sets, retained
centroids, medoids, and selected trace. It also proves that the historical and
Torch replays reproduce their respective terminal artifacts.

### Consequences

- Zero historical-versus-corrected mask differences permits adoption of the
  live terminal without a DBSCAN rerun.
- Any difference blocks final standardization. The audit emits the corrected
  downstream trace for a fresh downstream-only replay bound to the existing
  DBSCAN manifest; generation, theta filtering, and DBSCAN remain immutable.
- The audit is CPU-only, fresh-root, PASS-last, and forbidden on HPC.

### Status

Accepted for implementation; live execution remains untouched.

## [2026-08-25] Fail closed while recovering disconnected exact AIDS components

### Decision

Supersede the first disconnected-anchor recovery draft with an exact,
hash-closed component theorem that is valid only when every frozen seed anchor
belongs to one exact initial anchor component. Every nonfailure row must retain
the completed seed-ledger lower bound, every failure row is an exact core
anchor, and primary plus (when needed) all-anchor scans attach other anchor
components through float64-revalidated `distance <= eps` witnesses. Split seed
components route to the general external exact engine instead of publishing a
component shortcut.

For a genuine multi-component all-core result, stream every cluster in frozen
global-row blocks. The fixed-block Torch-float32 reduction is explicitly a
`PROJECT_EXTENSION`, not bit-identical to upstream's one-shot `torch.mean`;
float64 centroids are retained as an audit and any radius or theta decision
disagreement fails closed. Coverage and greedy selection use the authoritative
Torch-style strict masks without copying the 91.9M-row principal component.

Theta-close pair inputs are always reconstructed from a validated close-view
authority when one is supplied. Progress arrays, query sets, label promotion,
certificates, checkpoints, writer locks, and terminal manifests are reopened
through content hashes and deterministic replay. Resume never overwrites a
tampered partial or certificate, and terminal PASS binds the current
`O_NOFOLLOW` writer-lock inode.

### Consequences

- Nonanchor-to-nonanchor bridges cannot be silently missed under the unique
  seed-component gate; outside that gate the shortcut is inapplicable.
- A true two- or three-component exact partition can complete centroid,
  strict-radius coverage, medoid lineage, and stable greedy selection with
  bounded memory.
- Numerical differences that could change a paper decision block publication
  for an explicit audit rather than being hidden by a tolerance.
- This commit is implementation and focused-test evidence only. It does not
  adopt c766 artifacts, launch an AutoDL controller, stop the old brute route,
  or claim an AIDS matrix PASS.

### Status

Accepted for detached review; not deployed.

## [2026-08-25] Isolate c766 recovery behind a typed five-stage controller

### Decision

Use a dedicated CPU-only controller for the disconnected AIDS exact-DBSCAN
recovery. Its immutable stage order is failed-selection evidence adoption,
production-derived five-subset equivalence, exact component recovery,
streaming multi-component downstream/boundary replay, and standardized freeze.
The subset gate deliberately precedes the full 266-anchor expansion and makes
no full-production partition claim.

The c766 terminal FAILED state is never converted into an ordinary task PASS.
Its adoption receipt uses the independently reviewed recovery-only
schema and canonical scientific-state projection validator. The mutable state
byte hash is audit evidence only; the validator must prove its fixed projection,
mutable-field allowlist, double-read consistency, source-tree allowlist, and
immutable gate/manifest/process closure. Only the exact-recovery DAG can consume
the typed receipt. Matrix, Mutagenicity, and generic dependency readers reject
it; only the final hash-closed controller terminal becomes ordinary-PASS
eligible.

The controller freezes the reviewed v3 projection digests emitted by that same
production authority profile (`f2bcde0b...` for close and `b455b618...` for
final). A cross-module regression compares the controller values with
`PRODUCTION_AUTHORITY` directly. This replaces the earlier pre-v3 projection
digests, whose smaller schema omitted static state fields added by the hardened
adoption contract and would make every real production spec fail closed.

The controller uses a fresh CID/root and an independent adoption-authority
parent with one direct child. It uses immutable typed gates, inode-bound
`O_NOFOLLOW` locks, exclusive fresh-root claims, PASS-last terminal publication,
PID-generation-aware restart, and a read-only status command. Production launch
is refused until all release pins and explicit deployment authorization are
present. The superseding adoption API passed fresh detached review and is an
actual integration ancestor; its release pin remains intentionally unset until
the combined controller integration commit also passes review.

The fresh failed-selection promotion also publishes its claim, three small
arrays, rewritten selection, and terminal manifest through deterministic
two-name hardlink windows. Same-CID resume may rewrite an authenticated
temp-only prefix or collapse a final-plus-same-inode temp back to one link;
unrelated temp inodes fail closed. A process exit during the first claim write
therefore cannot strand the fresh exact root or create unbounded random temps.

Each typed gate also contains the exact terminal-validation projection and a
recursive closure inventory. Reopen rehashes bounded artifacts and verifies
publish-time SHA plus exact physical stat identity for large arrays, so a
post-gate mutation cannot survive controller restart/status merely because the
outer receipt is unchanged. Controller-level exact progress is the monotonic
sum of the primary and expansion ledgers. Its host baseline is persisted before
worker spawn and reused across worker generations; a fast terminal path binds
the exact terminal, DBSCAN manifest, and recorded peak RSS instead of silently
losing the coexistence gate.

Root initialization is itself restartable: an empty parent-side preclaim,
named by CID and controller-manifest SHA, is acquired with `O_EXCL`, fsync, and
flock before any root directory is created. The final owner and controller
terminal bind that inode, while same-CID resume may finish only the known
root/gates/logs/owner prefix after an interruption. Output usage is a hard
*publication* cap, not merely a free-space estimate: controller-owned state,
gate, terminal, and PASS writes reserve their exact net growth before publish;
science subprocesses are checked periodically and cannot publish a controller
PASS after an over-cap excursion. The controller gracefully terminates a
still-bound worker on breach. This is not a filesystem quota, so an individual
science write may cross the cap before the next poll and its failed evidence is
retained rather than destructively removed.

The controller does not promise unlimited same-CID retries. Exactly one
non-checkpointed partial per downstream stage may be archived, and that archive
must be at most 1 GiB. A second interruption of the same stage or a larger
partial is retained and blocks for manual diagnosis/new CID. Likewise,
append-only logs are checked periodically rather than constrained by a kernel
quota. These boundaries preserve fail-closed scientific publication: neither
case can yield PASS over the formula-derived cap.

The output-space gate is formula-derived from 91,916,686 rows: two uint32
component arrays, int64 labels, core and radius masks, the largest atomic
transient, bounded five-subset dense ledgers, block ledgers, standardized files,
and logs, plus at most one retained 1 GiB interrupted archive for each of the
four non-common downstream stages, all eight retained subset attempts, and the
two-name startup-record crash window. This is about 8.97 GiB of new-output budget
plus an 8 GiB safety floor; the adopted 25 GB pair/vector authority is read-only
and zero-copy. The 96 GiB RSS ceiling applies only to the exact DBSCAN process
and is closed by its native guard and terminal peak certificate; it is not
claimed as a cgroup-enforced route-wide peak for the later subprocess tree.
OMP, MKL, OpenBLAS, and NumExpr are frozen to 16 threads, CUDA is hidden, no GPU
lock is acquired, and a first-checkpoint load/iowait/RSS coexistence probe must
pass before the long exact scan continues.

### Consequences

- A production-specific subset mismatch blocks before the expensive component
  scan.
- One-cluster historical radius post-hoc evidence is not reused as if it were a
  multi-component terminal; the d891 streaming summary performs the applicable
  strict `<` float32/float64 fail-closed replay.
- A controller crash may reattach to the same PID generation or validate a
  completed terminal, but it cannot start a second writer in the same root.
- No AutoDL process, failed c766 root, old brute route, GPU task, matrix cell, or
  HPC job is changed by this implementation commit.

### Status

Implemented for focused review. Release pins remain unset; not deployed or
pushed.
## [2026-08-25] Keep c766 failed selection separate from scientific PASS

### Decision

Permit a future AIDS recovery route to adopt only the completed adaptive
seed/failure selection from the failed c766 route. The primitive is pinned to
one physical AutoDL control root, namespace, controller ID, controller
manifest, PASS close-view gate, and FAILED/SEMANTIC final gate. It derives both
task outputs from their unique `main` run/attempt closure and accepts neither
alternate authority paths nor copied controller trees.

The v3 authority treats controller state as mutable transport rather than a
byte-immutable file. It requires exact top-level, unique-main, launcher-
identity, and worker-identity key sets and freezes every static value,
including `created_at`, launch/GPU/log/retry/output fields, command, child,
PID generations, and the length/SHA of the long failure reason. Only
top-level `updated_at` and `instances.main.heartbeat_at` may vary, and both
must remain nonempty UTC timestamp strings; their concrete values alone are
replaced by projection sentinels. Both states are read before and after every
full authority scan. The close/final gates remain byte-pinned, and the failed
attempt is an exact 14-file
relative-path/SHA allowlist: unknown, missing, symlink, and terminal-looking
extras are rejected rather than adopted by TOFU.

The receipt rehashes the referenced closure through `O_NOFOLLOW` descriptors,
holds the control/namespace/controller/task/output and artifact-parent
directory inodes throughout each scan, proves that the recorded worker
generation exited without sending a signal, and rejects writable source
descriptors. It adopts bytes read-only: the 25 GB vectors and pair store are
neither regenerated nor copied. The output is a fresh direct child of the
dedicated fixed parent
`outputs/autodl/recovery_evidence/aids_c766_failed_selection_v1`; output and
lock must be disjoint from every source root/file before either is created.
At call entry the fixed output parent is opened with
`O_DIRECTORY|O_NOFOLLOW`; sibling lock and output creation then use only its
held dirfd/openat identity. The output-directory descriptor is held from
`mkdir` through both authority scans and publication. Each failed-tree walk is
repeated after all 14 tracked hashes, again at the end of each full authority
scan, and immediately before terminal publication against the originally
recorded failed-root inode.

Publication writes a hidden preterminal receipt, completes the second full
source reopen, records both scans' state-byte observations in the final
receipt, and fsyncs a prepared recovery marker. The no-clobber hard-link is
named `RECOVERY_EVIDENCE_READY`; while still holding the same output lock, the
publisher must then run the complete typed terminal reopen over the controller
manifest, both gates and states, procfs exit evidence, every tracked source
artifact/directory, and the exact failed-tree inventory. The marker is not
authoritative until that locked reopen returns. Failure revokes only the READY
name whose device/inode equals the prepared marker's recorded inode; the
prepared evidence remains diagnostic. No file named `PASS` is created.
Terminal reopen uses the same typed v3 verifier; source drift revokes only the
exact marker inode in the held receipt-bound output.
Renamed/copied outputs, locks, or replacement marker inodes are never deleted.
The receipt's top-level and every nested key set/container shape are exact;
adding and re-signing a generic-looking `PASS: true` or any nested field cannot
turn recovery evidence into a consumable dependency.

The failed graph remains negative evidence: the receipt says
`source_final_status=FAILED`,
`failed_evidence_adopted_for_recovery_only=true`, and
`ordinary_pass_dependency_eligible=false`. It freezes canonical initial
component labels, seed component IDs, and the hash/minimum of exact
self-inclusive anchor degrees. All three production seeds must occupy the same
canonical size-three component, and every anchor must retain at least three
epsilon neighbors including itself.

### Consequences

- No ordinary controller dependency may consume this receipt as a PASS, and no
  DBSCAN partition has been proved by adoption. The controller integration is
  a typed external-receipt dependency on
  `aids_c766_failed_selection_recovery_evidence_v3`, never a generic task PASS.
- The later recovery route may plan from the exact 266 selected rows without
  silently rerunning or relabelling c766.
- PID reuse, live original generations, symlinks, path escapes, alternate
  gates, state-projection tampering, partial outputs, source drift,
  namespace/output/lock replacement, unexpected failed-tree files, and
  writable source handles all fail closed.
- The paired Slurm file is static AutoDL-only CLI parity and exits before its
  documentation command; this decision authorizes no HPC job or deployment.

### Status

Superseding v3 implementation and focused tests passed independent detached
review and were merged as an actual integration ancestor. Production adoption
and recovery execution remain separate explicit actions; this change performs
neither deployment nor SSH.

## [2026-08-25] Close c766 READY publication and process-exit receipts

### Decision

Supersede the initial v3 publisher with a locked post-READY validation barrier.
Immediately after the READY hardlink and directory fsync, the same output-lock
holder performs a complete `require_ready=True` reopen. This third full scan is
the only successful return path. Controller manifest, close/final gates,
projected states, procfs, all 28 fixture source artifacts, all 14 fixture source
directory authorities, and the failed-tree allowlist are therefore checked
after publication as well as before it. Any failure removes only a READY name
that still resolves to the exact recorded prepared-marker device/inode.

Validate all six `process_exit` receipt fields rather than treating the worker
identity alone as stable. The expected worker PID/start-ticks/command hash and
recorded child PID come from the frozen final task authority;
`old_science_worker_exited` must be the JSON boolean `true`, and
`signals_sent` must remain the empty list in both the receipt and current full
scan. Worker observations are limited to absent, generation-proven PID reuse,
or zombie. A positive recorded child PID is limited to absent or zombie because
there is no frozen child generation with which to prove a live reuse.

The two observation strings may change across scans only within those safe
sets. Procfs can legitimately move from zombie to absent, from absent to a
different worker generation, or between absent and zombie for an unowned child
PID. These dynamic transitions do not weaken the stable identity fields and a
live original worker or live/unprovably-reused child continues to fail closed.

### Consequences

- A drift injected after the second scan cannot survive as a successful READY.
- Rebinding a tampered receipt to freshly computed marker bytes cannot change
  any process identity, child PID, exit boolean, signal list, or observation
  domain.
- This commit remains implementation/test evidence only. It performs no
  adoption, controller launch, deployment, SSH action, or process signal.

### Status

Passed fresh independent superseding-commit review and merged into the
release-disabled controller integration; no production receipt was created.

## [2026-08-25] Separate TasteMolNet scoped use from upstream licence status

### Decision

Record `NOT_EXPLICITLY_STATED` as the immutable upstream-terms observation and
never convert project authorization into a `LICENSE_PASS`. Replace the old
binary licence-gate model for future work with a typed scoped policy that
separately controls private research computation, aggregate paper reporting,
and dataset redistribution. Raw or cleaned tables, reconstructable datasets,
SMILES/label records, graph caches, per-example predictions, and trained-model
artifacts remain forbidden for public release. Only sanitized aggregate
metrics/tables/figures, configuration, and provenance hashes may be candidates
for publication, and each candidate root requires an independent
manifest-closed no-dataset-redistribution audit.

Preserve the historical `LICENSE_REVIEW_REQUIRED` artifact unchanged. The
legacy `audit_tastemolnet_license.py` remains a non-authorizing historical
audit and may emit only `BLOCKED_LICENSE_REVIEW`; approval-shaped input cannot
turn it into PASS.

Freeze the future science contract as a genuine three-class GINE with
Bitter=0, Sweet=1, Tasteless=2 and an untargeted strict flip
`pred_before == 1 and pred_after != 1`. RF, binary projection, held-out-test
selection, HPC, data re-preparation, cache rebuilding, and non-fresh roots are
forbidden. The dedicated future route is exclusive to physical GPU 2.

This implementation commit deliberately checks in only an inactive policy and
controller fragment: `PENDING_ROOT_ACTIVATION`, `RUN_TASTEMOLNET=0`,
`enabled=false`, and `command=null`. It cannot start training. A separate
root-authorized activation must change the exact policy state, bind raw and
canonical policy hashes plus the read-only prepared/cache receipt, pass fresh
independent review, and use fresh AutoDL roots.

### Consequences

- Upstream terms remain unresolved as an observation; no success marker may be
  interpreted as a licence determination.
- Existing private data/cache bytes are validated in place and are never
  copied into policy or public audit outputs.
- The public-artifact audit is content/schema/provenance enforcement, not a
  licence audit.
- The original blocked three-baseline fragment stays blocked and immutable;
  the scoped GINE foundation is a separate typed route.
- Paired Slurm files are static CLI parity and exit before execution because
  this campaign is AutoDL-only.

### Status

Inactive implementation and focused tests only. No activation, deployment,
SSH access, GPU allocation, dataset read/rebuild, or experiment launch was
performed.

## [2026-08-25] Activate scoped TasteMolNet research without redistribution

### Decision

Apply the project owner's explicit instruction as a scoped execution policy:
private TasteMolNet research computation and aggregate paper reporting are
allowed, while upstream terms remain `NOT_EXPLICITLY_STATED` and every raw,
cleaned, molecule-level, cache, or reconstructable dataset release remains
forbidden. The historical `LICENSE_REVIEW_REQUIRED` file stays immutable and
is no longer used as a computation blocker for this dedicated route.

The full oracle is one task-specific three-class GINE on physical GPU 2. It
requires a raw-SHA-pinned active policy, a typed read-only prepared/cache audit
receipt, a fresh output, at least 20 GiB persistent free space, train and
validation cache loading only, validation Macro OvR ROC-AUC checkpoint
selection with Macro-F1 tie-break, validation-only temperature scaling, all
three validation recalls positive, and no RF provenance. The held-out test is
recorded only by path and SHA during training.

The training bundle writes policy, cache, and oracle manifests with
`paper_result_reporting_allowed=true`, `dataset_redistributed=false`, and
`upstream_license_not_explicit=true`; all are included in the checkpoint SHA
inventory and reopened before the terminal marker. Public reporting still
requires the independent no-dataset-redistribution audit.

Because the 200-epoch route is a multi-day job, training state is not held only
in RAM. A separate, private, fresh state root freezes the exact data/policy/
model/training contract and publishes one atomic checkpoint per completed
epoch. The checkpoint includes current and best model states, optimizer,
early-stop state, history, and Python/NumPy/Torch RNG states. An inode-bound
single-writer lock rejects concurrent writers, interrupted checkpoint
publication is reconciled from the fully fsynced state, and safe superseded
checkpoint deletion is auditable. The immutable oracle output remains absent
until a fully verified sibling staging bundle is renamed into place.

### Consequences

- `TASTE_RESEARCH_AND_PAPER_REPORTING_AUTHORIZED` and
  `TASTE_NO_DATA_REDISTRIBUTION_GUARD_PASS` mean only that the scoped project
  policy and no-redistribution boundary were validated.
- No code may emit `TASTE_LICENSE_PASS` or reinterpret these markers as an
  upstream licence conclusion.
- The legacy blocked fragment remains historical evidence; the dedicated
  policy-receipt-bound fragment is the only runnable Taste GINE route.
- HPC remains forbidden; the paired Slurm entrypoint is validation-only and
  exits before any training command.

### Status

Implementation and focused tests, including checkpoint/restart and terminal
reopen, are local pending independent review and an immutable AutoDL
deployment. No Taste GPU worker was started by this decision.

## [2026-08-25] Give TasteMolNet a persistent fail-closed supervisor

### Motivation

Epoch checkpoints alone do not close controller death, worker-registration,
output-parent replacement, or final-bundle publication windows. A direct
foreground wrapper can be lost while the GPU worker continues, and a generic
`PASS` marker cannot safely summarize mutable output and checkpoint trees.
The original resume fingerprint also did not bind every merged config byte,
override, clean source identity, or physical GPU UUID.

### Decision

Route the active Taste GINE fragment through a dedicated persistent AutoDL
controller with one fresh CID/root. Freeze the clean commit/tree, reviewed
worker program/wrapper paths and SHAs, Python, config-file SHAs, exact argv,
allowlisted scientific environment, policy/receipt/private-data authority,
physical GPU-2 contract, and exact output/state paths in its immutable spec.
Use the durable exec-startup barrier before every worker generation, adopt only
the recorded live PID/start-ticks generation after controller loss, and allow
one genuine process-loss retry against the same inode-bound training root.
Resource waiting remains bounded and does not consume that scientific retry.

Bind the trainer resume contract to the complete canonical merged config,
config files, dotlist and CLI overrides, clean source identity/hashes, runtime,
and GPU UUID. Reject symlink components and output/state overlap with prepared
or cache roots before any training input is loaded. Hold an output-parent
dirfd, named lock, sentinel, and contract claim for the full route. Use one
contract-derived finalization sibling, recover only the empty
mkdir-before-claim window, receipt bounded cleanup of owned partial contents,
and publish a verified complete inventory with Linux
`renameat2(RENAME_NOREPLACE)` relative to the held parent descriptor.

The controller's terminal publication holds the training-state root and named
writer lock, output parent, and finalization authority while repeatedly
reopening the complete bundle, policy/receipt, root identities, and SHA/stat
inventories. It freezes terminal evidence and final state first, then writes
controller `PASS` with no replace as the last publication. Terminal status
uses the same typed reopen rather than trusting marker presence.

Keep `NOT_EXPLICITLY_STATED` as the exact policy status everywhere and record
scoped project permission separately as `authorization_status`.

### Consequences

- Empty controller/finalization/output-authority creation windows are
  same-contract recoverable; nonempty unclaimed roots fail closed.
- Same-byte staging, state-root, named-lock, controller-root, and output-parent
  replacement cannot publish a valid controller terminal.
- Claim/completion sidecars remain terminally hash-bound instead of becoming
  unverified external authority, and private intermediates are bounded.
- The worker explicitly calls `exp_run` with `--max-gpus 4` and
  `--gpu-hard-limit 4`; the paired Slurm entrypoint remains a static HPC
  refusal.

### Status

Local implementation, hostile crash/replacement tests, and documentation only.
No commit, deployment, SSH action, controller launch, GPU allocation, or
TasteMolNet experiment was performed by this decision.

## [2026-08-25] Harden the TasteMolNet release controller against terminal and restart gaps

### Decision

Require the four-GPU inventory route to state both `--max-gpus 4` and the
reviewed `--gpu-hard-limit 4`. Freeze one durable controller-wide resource
deadline and keep the persistent controller responsible for exit 75 until that
deadline, without spending the single scientific process-loss retry. A crash
in `RELEASE_AUTHORIZED` before the startup token is sent re-arms the same
attempt. Worker adoption now binds PID, Linux start ticks, cwd, argv, command
bytes, executable path, and executable identity while permitting only the
reviewed launcher-to-wrapper-to-`exp_run` exec phases.

Reserve event-log capacity for terminal transitions and never let a diagnostic
cap prevent durable PASS/FAILED state. A worker log that reaches its bound is
inode-checked and bounded while the controller continues supervising the live
generation; no signal is sent and ownership is not abandoned. Terminal status
and controller reopen share one strict read-only state/terminal/PASS validator.
PASS publication retains all controller, output, and state locks for a complete
post-marker source rescan; failure removes the marker only when its name still
resolves to the recorded device/inode.

Resume cleanup preserves the first durable `CLEANUP_PREPARED` inventory across
partial deletion crashes and deletes only through the already-held staging
directory descriptor. The immutable training resume contract additionally
binds NumPy, RDKit, PyG, cuDNN, CUDA-driver, and an allowlisted environment
manifest.

### Consequences

- Resource scarcity cannot reset its deadline by restarting a wrapper or
  consume a scientific retry.
- A live science generation remains supervised when diagnostics reach their
  bound or process identity becomes untrusted; the controller never signals it.
- Status inspection cannot reconcile, publish, or otherwise mutate controller
  state.
- A path replacement cannot redirect cleanup or PASS revocation to a different
  inode.

### Status

Local implementation and crash/negative tests only. No commit, push, SSH,
deployment, AutoDL controller start, GPU allocation, or TasteMolNet experiment
was performed.

## [2026-08-26] Close TasteMolNet completion-adoption and exact-runtime release gaps

### Decision

Treat `state.phase=PASS` or the presence of any terminal-named controller
artifact as an irreversible switch to the shared strict read-only validator.
This switch occurs before a resumed controller creates/acquires a writer lock
or reconciles publication temporaries; missing PASS/state/terminal peers and
untyped terminal fields therefore fail without repair.

Bind `training_contract.json` as a held physical inode, complete file SHA,
recomputed canonical contract SHA, and exact content object. Carry that
evidence through every epoch checkpoint, latest/heartbeat record, completion,
terminal scan, and read authority. A self-declared hash can no longer validate
changed contract content, and failed opens release all held descriptors and
locks.

Authorize the narrow finalization-published/completion-missing crash window
with one immutable controller receipt. `exp_run` keeps its ordinary fresh
output gate and accepts a pre-existing output only with that exact receipt,
state root, and trainer argv. The trainer validates the same read-only closure
and may write only the missing completion for the unchanged resume contract.
Controller heartbeat replacement is allowed only when every stable receipt,
CID/spec/root/deadline/attempt/launch field remains exact.

Register the real trainer child behind a second durable exec barrier before
release. Freeze its PID, Linux start ticks, parent PID/start, cwd, argv and
cmdline hashes, executable path/inode, command hash, barrier record, and
launcher-to-target phase bindings. If `exp_run` is lost, the persistent
controller adopts that exact child and forbids a concurrent retry until it
exits; identity drift remains supervised fail-closed without signalling the
science process.

Freeze the formal route to the verified GINE config, seed 7, and a persistent
free-space threshold of at least 20 GiB. Require error-mode PyTorch
deterministic algorithms, deterministic cuDNN, disabled benchmarking and TF32,
and the fixed CUBLAS/Python hash environment. The production Taste route still
requires exactly one masked CUDA device on physical GPU 2 and does not emulate
CUDA on a non-GPU host. Resume binds NumPy, RDKit, PyG, cuDNN, driver, and the
allowlisted environment manifest.

Finally, freeze the private train/validation graph-cache manifest path/SHA and
each cache path/SHA/inode. Hold descriptors across both cache loads, verify the
manifest and both named inodes before and after deserialization, and carry the
resulting cache contract into resume, output provenance, and controller
terminal evidence. Calibration and test caches remain unopened.

### Consequences

- A published model bundle cannot bypass `exp_run` freshness or consume a
  science retry merely because its completion write crashed.
- Parent loss cannot create a second trainer while the registered child is
  alive, including across controller restart.
- Contract, cache, runtime, config, or deterministic-backend drift prevents
  resume/PASS rather than being reduced to a warning.
- No redistribution, held-out-test loading, HPC eligibility, or upstream
  licence conclusion is introduced.

### Status

Local implementation plus real subprocess/crash and hostile negative tests
only. No commit, push, SSH, deployment, AutoDL controller start, GPU
allocation, TasteMolNet dataset read, or scientific experiment was performed.

## [2026-08-26] Bind Taste cache deserialization and stale-child adoption to held authority

### Decision

Deserialize both Taste train and validation graph caches directly from
duplicated binary streams backed by the already authenticated, held file
descriptors. Path-based loading remains available to ordinary callers, but the
full Taste route never reopens the cache pathname inside `torch.load`. Recheck
the manifest and both held-file/named-file inode and content bindings around
deserialization, so a graph-cache root swap either yields bytes from the held
inode and then fails the named-path closure or fails closed without consuming
replacement bytes.

Split trainer-child discovery into generation-independent authority validation
and current-worker binding. Before liveness classification, require the exact
authority/process/barrier schemas, strict integer process identities, argv and
command hashes, canonical run-state paths, owner-only authority file, physical
executable identity shape, and parent/child/barrier structural relationships.
Only an authority whose declared child PID/start pair is conclusively absent,
reused, or zombie may be ignored as historical. An unreadable or malformed
process observation, malformed authority, or still-live generation continues
through the full current-parent/barrier/phase binding and therefore prevents a
concurrent retry when mismatched. Reopen the same authority inode/hash after
the liveness observation before either exclusion or adoption.

### Consequences

- Replacing the graph-cache directory cannot redirect train or validation
  deserialization to a newly named inode.
- A valid dead historical trainer authority no longer blocks adoption of the
  one current live trainer merely because its parent belongs to an older
  worker generation.
- A live stale trainer, malformed stale record, unreadable `/proc` record, or
  authority replacement remains a hard controller failure; no second science
  worker is launched and no process is signalled.

### Status

Local implementation and focused root-swap/dead-stale/live-stale/malformed
negative tests only. No commit, push, SSH, deployment, AutoDL controller start,
GPU allocation, TasteMolNet dataset read, or scientific experiment was
performed.

## [2026-08-26] Classify exact Linux process exit before Taste argv phases

### Motivation

A mandatory Linux code-only check exposed a short-process exit window: the
registered PID/start generation could still have a `/proc/<pid>` entry while
its command line had already been cleared. Treating that empty command line as
a live argv/executable mismatch incorrectly converted a natural worker exit
into controller identity drift.

### Decision

Parse Linux proc-stat PID, state, parent PID, and start ticks as one strict
typed observation. Around every outer-worker and real-trainer process
snapshot, require the declared start ticks and reject PID reuse. Treat only an
absent entry or the exact generation's `Z`/`X` exit state as natural death
before argv phase classification. Recheck the exact proc state after a
snapshot and before classification, so an exit that clears `cmdline` cannot be
misclassified as a live phase drift.

Retry transient snapshot reads for a bounded interval while an exact
generation remains nonterminal. If its command line or executable identity is
still empty, malformed, unreadable, or unstable while proc-stat says it is
live, fail closed. Permission errors, malformed proc-stat, phase regression,
PID reuse, executable binding, and ancestry checks remain strict. Apply the
same observation primitive to the `exp_run` worker and adopted trainer child.

### Consequences

- A naturally exiting short worker or trainer does not trigger a false
  argv/executable identity-drift failure merely because it remains briefly
  visible as a zombie.
- A genuinely live process with empty or malformed argv is never treated as
  successful or dead, and cannot authorize a concurrent retry.
- Deterministic regression coverage includes live-to-zombie, PID-reuse, live
  empty/malformed argv, and a real Linux launcher-to-target-to-zombie sequence
  without sending the worker a signal.

### Status

Local superseding implementation and focused tests pending fresh independent
review. No new commit, bundle, push, SSH, deployment, process signal, GPU
allocation, TasteMolNet dataset read, or scientific experiment was performed.

## [2026-08-27] Supersede the stale Taste execution block with policy v2 and a fresh main controller

### Motivation

The historical four-dataset controller correctly recorded the former
licence-review decision, but no Taste heavy science was started. The project
owner has since explicitly authorized private research compute and aggregate
paper reporting while continuing to forbid redistribution of raw, cleaned,
row-level, or reconstructable Taste data. Rewriting the old controller would
destroy provenance; continuing to treat it as current would incorrectly block
authorized science.

### Decision

Keep every historical manifest/state/gate byte outside this route read-only.
Introduce policy schema v2 and a typed fresh adoption receipt that records
`SUPERSEDED_POLICY_V1`, `old_science_adopted=false`, and
`READY_FOR_MAIN_ROUTE`. Policy v1 remains readable as history but cannot
authorize the main route. Nested policy booleans and counts use exact native
JSON types, so Python bool/int or int/float equality cannot authorize drift.

Run the formal three-class GINE only on physical GPU 1 from a clean immutable
execution tree. Freeze a 20-GiB planning reservation and a 100-GiB
post-reservation floor. Publish `last.pt` plus a typed real checkpoint-reload
receipt, and emit first-batch progress. Create a separate fresh main-controller
namespace with an owner lock, immutable controller spec, T0--T16 evidence,
event logs, a GPU-2 classifier-independent READY lane, protected GPU-0/GPU-3
lanes, and `RUN_GNN_ABLATION=0`. The classifier is never a matrix method cell.

### Consequences

- The old licence-review record remains true historical evidence but no longer
  outranks live policy-v2/controller/science evidence.
- GPU 1 may run the formal GINE without touching the BACE processes on GPU 0
  or GPU 3; GPU 2 is not silently assigned classifier-dependent work.
- Dataset redistribution remains forbidden even when compute and aggregate
  reporting are allowed.
- Later calibration, oracle smoke, four-method smoke/full, matrix registration,
  and paper artifacts remain dependent on explicit typed PASS gates; this
  change does not fabricate them.

### Status

Implementation and focused verification precede immutable AutoDL deployment.
Runtime PIDs, GPU UUIDs, checkpoints, and PASS markers must be reported only
after direct live verification.

## [2026-08-28] Preserve frozen Python argv and narrowly adopt completed Taste identity drift

### Motivation

The deployed Taste GINE worker used the frozen launcher token
`.../bin/python` at both Python positions in the nested `exp_run` command,
while that path physically resolved to `.../bin/python3.10`. The controller
resolved the expected argv tokens but correctly compared `/proc/exe` to the
physical executable. This inconsistent normalization rejected an otherwise
exact reviewed command and left the controller `FAILED` with
`WORKER_PROCESS_IDENTITY_DRIFT` after the completed trainer exited.

### Decision

Preserve `AUTODL_PYTHON` exactly in both frozen argv positions and continue to
validate `/proc/exe` separately against its resolved physical executable.
Neither spelling is interchangeable: a process must retain the reviewed raw
argv token and the reviewed physical executable identity.

Permit one narrow terminal-only reconciliation when the durable controller
state is exactly `FAILED/WORKER_PROCESS_IDENTITY_DRIFT`. At least one matching
trainer-child authority must bind both the controller CID and root. Every
matching authority must retain the exact reviewed `exp_run` parent and trainer
startup-launcher snapshots, remain inode/content stable across inspection, and
prove both PID/start generations exited. Only then may the already complete
terminal closure be reopened and `_publish_terminal()` called with the frozen
attempt and launch index. The controller never launches, retries, resumes, or
signals science in this path. Missing terminal evidence, any live generation,
missing/malformed/partial-collision authority, or any other FAILED reason
remains return code 2.

### Consequences

- A reviewed Python symlink no longer creates false argv drift, while a changed
  raw argv token or changed physical executable still fails closed.
- A scientifically complete output can receive only its pre-existing terminal
  publication; this exception cannot become a generic FAILED recovery route.
- Main-controller namespace, policy-v2 adoption, GPU lanes, matrix registry,
  output/training roots, and the historical licence block are unchanged.

### Status

Local isolated successor implementation, deterministic hostile tests, and
static validation only. The production diagnosis was read-only. No remote
write, process signal, restart, controller/GPU-lock/matrix mutation, deployment,
or new scientific computation was performed by this decision.

## [2026-08-28] Publish Taste T3/T4 PASS only after retained closure

### Motivation

Taste T3 and T4 must write evidence before performing the final retained-input
closure.  Publishing the PASS marker at that earlier point made error cleanup
security-sensitive: a check-then-unlink sequence could delete a replacement
inode, while preserving replacement terminal files could leave a stage that
still reopened as PASS.

### Decision

Use a two-phase, marker-last publication.  The producer first writes all JSON
evidence and `sha256sums.txt`; the inventory already commits the expected hash
of the future marker, but the marker itself is absent and the public verifier
must reject the prepared root.  While the root remains non-terminal, the
producer reopens every checkpoint, cache, policy, execution-source, and T3
predecessor authority.  Only after that closure succeeds does it create and
fsync the exact PASS marker through the held output directory as the final
commit operation.  The marker payload also binds the physical identities of
every prepared JSON document and `sha256sums.txt`, so an equal-byte inode
replacement cannot reopen as the producer's terminal stage.  There is no
terminal cleanup path and no unlink of a possibly replaced file.

The held T2 consumer API now exposes only two additional metadata payloads to
T6: `split_manifest.json` and `test_evaluation_status.json`.  This lets T6 bind
the frozen train path/hash and prove the test split remains unevaluated while
continuing to reject every split CSV, validation predictions, training
metrics, and terminal training checkpoint.

### Consequences

- A producer exception during preparation or retained closure leaves no PASS
  marker and cannot leave a reusable T3/T4 terminal stage.
- Publication never tries to unlink an attacker-replaced marker or hash
  inventory.
- T6 can load its GINE and train metadata through retained descriptors without
  reopening validation, calibration, or test payloads.
- This change does not release a controller, deploy code, or start science.

## [2026-08-28] Keep Taste T5 dataset-independent and release-authority bound

### Motivation

The mature BACE initializer contains useful loader mechanics, but its schema,
dataset claims, audit selection, and terminal marker are BACE authority. A
renamed BACE manifest would not prove a Taste policy is generic, and writing
`split_used=train_only` for a zero-step adapter that opened no split would be a
false provenance claim. T6 also needs a stable read-only identity for the exact
policy and reference model rather than a mutable adapter pathname.

### Decision

Add a Taste-owned T5 contract that implements only a fresh zero-optimizer-step
LoRA from an explicitly generic ChemLLM base. Record
`initializer_data_split_used=none`, `taste_split_access_max=train_only`, and an
empty loaded-split list as separate facts. Do not implement the optional
train-only SFT fallback until a later authority explicitly requests it.

Keep the tracked configuration disabled until a separately reviewed file and
its expected raw SHA bind policy-v2/receipt, the generic source inventory, the
clean execution commit/tree, and the common three-class GINE plus T3/T4 PASS
identities. A controller declaration is not lock ownership: the public builder
also remains disabled until a separately reviewed physical execution receipt
is implemented. Reopen the complete content authority around every internal
adapter materialization/load step. RF, GNN reward, validation, calibration,
test, HPC, and data redistribution remain absent.

Publish only to a fresh private timestamp root with an atomic directory
no-replace operation. The root contains a complete LoRA, Taste provenance, the
five stage evidence files, and a PASS-last
`[TASTE_CLEAN_POLICY_INITIALIZER_PASS]` marker. Expose a stable T6 consumer
module with both one-shot validation and a combined descriptor-held
source/adapter load token. The closure uses hash identity while its live
authority additionally retains inode/ctime state to detect swap-load-restore.

### Consequences

- A generic zero-step initializer is directly loadable by the existing stable
  PPO/PEFT runtime but is never validated through the BACE schema.
- `train_only` remains an access ceiling, not a fabricated claim that this
  route read Taste train rows.
- Missing production T3/T4 or Git/source pins keep T5 explicitly disabled and
  cannot create the final output root.
- Missing physical execution-receipt authority also keeps the public builder
  disabled before authority/model loading.
- T5 does not itself count as training science or a matrix cell.

### Status

Local implementation, documentation, and hostile tests only. No commit,
bundle, SSH, deployment, controller/GPU-lock/matrix write, production model
load, adapter creation from the real base, dataset access, or scientific
experiment was performed. One tiny local CPU PEFT model was used only for a
real safetensors save/reload format test.

## [2026-08-28] Replace in-place Taste T2 reconciliation with fresh read-only adoption

### Motivation

The completed Taste GINE result and its failed controller express two
different facts: science reached a closed PASS bundle, while the old
controller persisted an identity-drift failure under its then-deployed
comparison logic. Rewriting that failed controller in place would merge those
facts and retain write authority over scientific/controller roots that must
now be historical. It would also introduce repair and crash windows into the
source of truth.

### Decision

Do not reconcile, resume, or otherwise write the old GINE controller,
training-state, output, registry, main controller, matrix, or GPU locks. Add a
versioned successor that holds all source pathname edges and files through
`openat`/`O_NOFOLLOW` descriptors, validates exact
`583bf668896142d8cc292cd624fbbffc20faf688` execution identity and
`3a90fd8697b58bad4f95f3be9347b327d5c51043` identity-fix source, and requires
the controller to remain exactly `FAILED/WORKER_PROCESS_IDENTITY_DRIFT`.

The adoption also binds every declared PID as absent, the typed trainer-child
authority, source run's final registry PASS, runtime PASS state, training
log PASS/OK/exit-0 markers, completion and canonical contract, latest
checkpoint, exact eighteen-entry
checkpoint hash closure, and full physical output/training-state inventories.
It rechecks three-class GINE semantics, label map, Sweet source label,
health/reload gates, no-RF provenance, held-out/cache boundaries, scoped
research/reporting permission, and no redistribution.

Publish only to deterministic fresh root
`<control_root>/tastemolnet-t2-gine-pass-adoption-v1/<source_cid>`. The root
contains exactly `input_hashes.json`, `state.json`, `manifest.json`,
`output_hashes.json`, and PASS-last `gate.json`; `manifest.json` is the receipt
whose SHA later T3/T4 stages must bind. There is no generic PASS marker and no
same-root recovery: an existing or interrupted root requires a newly versioned
successor.

Keep the historical controller failure and the scientific PASS as separate,
typed facts: the former is an explicit control-plane scientific false negative,
while registry/runtime/training-complete/formal-output authorities remain PASS.
Retain the four preterminal receipt files and revalidate every held old-source
identity before every receipt write and through the terminal gate commit. Bind
the four files' physical identities and held directory in the gate, prepare and
sync the gate under a non-authorizing name, and expose it only with the final
`renameat2(RENAME_NOREPLACE)` operation. No validation or fsync follows that
commit; cleanup is non-throwing. Main-controller and matrix paths remain outside the publisher's open
and write sets. T3 may consume only a validated fresh T2 gate/receipt binding
plus the exact formal-output inventory recorded by that receipt; neither the
old controller nor main/matrix state is an alternate T2 authority.

Retain and revalidate the fresh namespace and final CID directory path edges
as well, including the exact prefix listing after every no-clobber write. A
rename/replacement or injected destination file must stop publication before a
gate. Read-only status recomputes those physical identities and may endorse
PASS only under the same physically held external release authority;
canonical-looking JSON or equal-byte inode replacement cannot manufacture a
valid status.

Freeze this implementation with a tracked release config whose exact native
Boolean authorization is false and whose external-authority path/hash are
null. Replace the former release-normalized self-pin with an independent
reviewed receipt: it must bind the clean implementation commit/tree, exact
critical blobs, and every exact native source pin. A future clean one-parent
release commit may change only the release config to bind that receipt. Audit
all three checkouts with fixed root-owned `/usr/bin/git`, explicit retained
gitdir/worktree paths, isolated Git config/environment, replacement objects
disabled, and rejection of dirty/staged/untracked/ignored, skip-worktree,
assume-unchanged, and bytecode-cache state. Production PID checks always use
literal physical `/proc`; the CLI has no proc-root override. Preflight and
status are read-only; publish returns the blocked exit before destination
creation. The paired Slurm entrypoint is static AutoDL-only refusal.

### Consequences

- Scientific completion can be represented without falsifying or mutating the
  old controller's historical failure.
- A source swap, symlink, hardlink, content/stat drift, live declared PID,
  checkpoint omission, policy/classifier drift, receipt whitespace change, or
  partial output fails closed.
- T3/T4 gain a deterministic receipt-hash dependency without writing the main
  controller or matrix.
- Publication needs a later independent evidence capture, external-receipt
  review, and exact one-config release child; this implementation cannot be
  activated by runtime input.

### Status

Local isolated implementation, focused/static tests, and documentation only.
No commit, push, SSH, deployment, process action, controller/GPU-lock/matrix
mutation, scientific-root write, model load, or new science was performed.
## [2026-08-28] Implement Taste T6 Ours PPO behind an external release authority

### Motivation

TasteMolNet T6 must exercise the existing stable decoded-chemistry PPO loop
against the frozen three-class GINE oracle. A configuration-only or synthetic
smoke is not scientific evidence, while a direct runnable script would bypass
the T3--T5, policy-v2, GPU, output, and controller authorities.

### Decision

Add a Taste-specific T6 runner that reuses
`run_stable_decoded_chem_ppo_loop` for five to ten real optimizer updates. It
loads only frozen train prompts, keeps validation/calibration/test payloads
closed, uses GINE with `num_classes=3` and `source_label=1`, and defines a
strict flip as `pred_before == 1 and pred_after != 1`. Candidate reward rows,
the frozen reference policy, LoRA adapter tensors, value-head tensors, and the
final/periodic checkpoints are cross-bound to the in-memory computation.

Use descriptor-held T3, T4, T5, GINE, train-file, policy, release-receipt, and
fresh-output authorities. Runtime model loads may use `/proc/self/fd/N`, but
checkpoint metadata is rebound to the reviewed lexical source-model identity,
and the shared-loop logger rewrites its descriptor-backed I/O root to the
reviewed lexical output root. No durable output may contain an FD path. T6
checkpoints contain
only the adapter, value head, and PEFT model card; the tokenizer remains the
separately held T5 source-model authority and is deliberately not copied into
T6. The terminal tree has an exact file/directory layout, durable leaf and
bottom-up directory flush, exact candidate JSONL bytes, a physical inventory,
and a held-inode marker-last commit. Public consumers must use
`hold_taste_ppo_output()` or `validate_taste_ppo_output()` rather than testing
for a `PASS` pathname.

Keep the checked-in release bit false and every deployment/evidence pin null.
Activation requires a later immutable integration commit and a typed external
controller receipt that binds the controller generation, GPU-1 lease, exact
execution tree/config/wrapper, T3--T5 gates, fresh output root, and storage
authority. The controller must run the strict retained terminal consumer
before it adopts T6 PASS. The paired Slurm script remains a static AutoDL-only
refusal.

### Consequences

- A real T6 run cannot silently fall back to RF, CPU, another GPU, another
  dataset split, a synthetic candidate pool, or a rewritten PPO algorithm.
- Equal-byte inode replacement, unknown checkpoint directories/files,
  candidate JSON ambiguities, checkpoint config drift, and failed marker
  publication remain fail closed.
- This implementation commit is not itself an execution authorization. No T6
  science should be started merely by changing an environment variable.

### Status

Local implementation, focused/affected regression tests, and static AutoDL/
Slurm checks are complete. External controller-receipt integration, strict
post-child adoption, an `/autodl-fs` link/fsync micro-probe, immutable deploy,
and T6 science remain pending.

## [2026-08-28] Make the fresh T2 receipt the only downstream GINE authority

### Motivation

T3 and T4 originally reopened the formal GINE bundle directly. That verified
the bundle bytes but did not prove that the reviewed five-file T2 adoption had
accepted those bytes, so an equal legacy bundle could bypass the new T2
authority while later T5/T6 artifacts claimed a continuous provenance chain.

### Decision

Expose a receipt-only held consumer taking the fresh adoption root and exact
gate, receipt, and embedded-source SHA-256 pins. It validates the exact
five-file canonical hash DAG, physical publication binding, frozen source
CID/run/commits, and recorded 19-file formal GINE inventory. It intentionally
does not reopen the historical failed controller, training-state, execution,
or identity-fix roots. T3 holds and records the complete binding through its
terminal commit; T4 must exact-match it and T3; T5 freezes and reopens it; T6
reopens it and persists the complete binding plus all three pins.

### Consequences

- Historical control-plane failure remains immutable evidence, not a runtime
  dependency or alternate PASS source.
- Any T2 receipt/hash/inode replacement or cross-stage binding mismatch fails
  before a downstream PASS marker.
- T3--T6 release bits, GPU ownership, matrix state, and science status remain
  unchanged and disabled until separately reviewed execution authorities exist.

## [2026-08-28] Preserve one three-class GINE across GlobalGCE target branches

### Motivation

Official GlobalGCE optimizes an internal target class `1`, while TasteMolNet
requires two destination branches from Sweet (`1`) to Bitter (`0`) and
Tasteless (`2`). Training separate binary classifiers or collapsing
`1 - p(Sweet)` into one binary target would change the reviewed oracle and
could accept a candidate whose calibrated argmax is still Sweet.

### Rejected predecessor

Independent review rejected the first local foundation candidate at base HEAD
`31dd9f46f9ed0fdf88ee533a6c5b0584ed428c3e`, base tree
`186aa30d710fc3a5038f3998657961334405327f`, staged tree
`d9f0385e51d38fe1db7b610925c75c31cbe3240f`, ordinary patch SHA-256
`7fd64231399500eda72fa65e488f769c2d52388fa497e456fad86eb640ca1ff6`,
and full-index patch SHA-256
`8c9b4c1f53f8be8e602ed99aa01d3aa79bec77d9ed826bf5220146160932de97`.
It truncated finite non-integer CSV labels, admitted a three-class request into
the binary native GTGNN path, and bound target/class identity only after a
training run had already reached its terminal checkpoint.

### Decision

Parameterize the differentiable GINE bridge by an exact frozen class count and
introduce a target-class view that orders logits as source, requested target,
then all remaining original classes. Official internal `0/1` therefore maps to
one explicit frozen source/target pair, but the underlying checkpoint,
temperature, features, and original three-class logits remain unchanged.
Generalize native train metadata to require every frozen class and bind dataset,
class count, source, and target into resume state so branches cannot reuse one
another's learned GlobalGCE state. Class labels now use canonical non-negative
integer text and every row must explicitly declare `split=train`; no float
coercion or implicit split default remains. Multiclass and every Taste route
require a frozen GINE, while the historical two-class native GTGNN surface is
retained as an explicitly versioned legacy path.

The successor also binds every epoch checkpoint and its terminal heartbeat to
one canonical v2 identity covering dataset/classes/source/destination, ordered
native and source train/validation cohorts, official source files, full frozen
GINE checksum inventory and calibrated temperature, and all rule-training and
gSpan settings. A v2 caller cannot adopt an unbound v1 checkpoint. The legacy
no-identity API remains typed v1 for historical binary callers and cannot read
a v2 checkpoint.

### Consequences

- Sweet-to-Bitter and Sweet-to-Tasteless can share the exact same frozen GINE
  while retaining target-specific official optimization.
- Final candidate acceptance still requires an independent original-order
  three-class oracle and untargeted `pred_after != 1`; the adapter is only a
  differentiable loss view.
- A target-0 partial training checkpoint cannot be resumed as target 2, and
  train-cohort, frozen-checkpoint, or temperature drift fails before model or
  optimizer state adoption.
- This prerequisite does not implement or release T8. No controller, GPU,
  dataset, output, or scientific process is started by this change.
## [2026-08-28] Supersede mutable terminal links with managed execution v2

### Motivation

The uncommitted managed-v1 candidate still coupled terminal authority to a
mutable named inode and did not make a UUID attempt/checkpoint generation plus
an independent verifier the only route to PASS. A worker or path ABA could
therefore be confused with scientific release authority even when its bytes
looked plausible.

### Decision

Introduce isolated v2 modules for UUIDv4 attempts/checkpoints, complete
launcher and worker process lineage, and quarantine-only anomaly handling with
`AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0`. Workers may produce only raw
evidence, worker exit evidence, and a descriptor-scanned SEALED inventory.
Only an independent verifier may add verification/gate/PASS and atomically
publish the complete directory. Same-filesystem publication uses atomic
no-replace directory rename. Cross-filesystem publication copies into a unique
destination-side directory, fsyncs and rehashes it, then uses the same atomic
rename. Mutable file links and copytruncate are excluded.

### Consequences

- PID disappearance or legitimate re-parenting is runtime audit evidence, not
  the sole scientific-adoption criterion.
- Drift, orphan, heartbeat, child, or terminal anomalies become QUARANTINED
  without SIGTERM/SIGKILL and cannot release dependencies.
- Attempt/checkpoint directory UUIDs are permanently burned, including partial
  failures, so fixed-path delete/recreate ABA is not an authorized retry.
- T3--T9 remain unreleased until their workers and independent verifiers use
  this API; this code freeze runs no science and asserts no scientific PASS.

## [2026-08-28] Adopt the completed Taste GINE by replay, not controller repair

### Motivation

The completed TasteMolNet GINE bundle is scientifically closed, while its old
controller correctly retains `FAILED/WORKER_PROCESS_IDENTITY_DRIFT`. Rerunning
the classifier would discard valid work, but rewriting that control history or
trusting its worker-authored records alone would create false provenance.

### Decision

Add an independent v2 verifier that holds the exact bundle and declared input
files by physical inode, validates the complete hash/config/class/data-use
closure, reloads the frozen GINE, and replays every validation prediction from
the held validation cache. Publish only a fresh UUID receipt below
`tastemolnet-main-v2/adoptions/T2_GINE`; retain the historical controller and
training evidence byte-for-byte. Authenticate the old validation temperature
as evidence only and require a fresh validation-only T3 calibration bundle.

### Consequences

- T2 can become `ADOPTED_SCIENTIFIC_PASS` without another GINE training run.
- The old process failure is superseded only for the scientific artifact and
  remains the authoritative control-plane history.
- Calibration/test are not loaded for adoption replay, RF remains absent, and
  the receipt is not a method-matrix cell.

## [2026-08-28] Bind T2 adoption to the bundle's canonical checksum filename

### Motivation

The first AutoDL invocation stopped before receipt creation because the
verifier opened `sha256s.txt`, while the immutable historical bundle and its
exact inventory use `sha256sums.txt`.

### Decision

Open, parse, and exclude only `sha256sums.txt` when validating the checksum
manifest. Keep the independently generated adoption-receipt checksum file
named `sha256s.txt`; the two names belong to different schemas.

### Consequences

- The verifier now consumes the actual held historical manifest.
- The pre-publication failure created no PASS receipt and changed no source
  artifact, so a corrected independent invocation is safe.

The next pre-publication AutoDL check measured a maximum raw-softmax versus
stored-probability difference of `1.26654272070148e-07`, matching the stored
probability-sum float32 roundoff (`1.2665987014770508e-07`). The CSV boundary
therefore uses an explicit tolerance of two float32 epsilons. This does not
change the separately frozen model-replay tolerances or accept calibrated
probabilities as raw probabilities.

## [2026-08-28] Refit Taste T3 temperature under managed execution v2

### Motivation

The existing Taste T3 candidate only authenticated and adopted the temperature
already stored by T2. The main experiment decision instead requires a fresh
scalar fit using validation only, with no calibration/test payload access.

### Decision

Add a worker-only T3 candidate builder and a separate scientific verifier. The
worker writes a fresh checkpoint candidate below its managed artifact root;
the verifier retains that SEALED tree, the exact T2 adoption receipt, and the
19-file source bundle, repeats the fit, and publishes with atomic no-replace
managed-v2 semantics. T2 training/reload metadata remains historical, while
the new temperature document and oracle manifest become downstream authority.

### Consequences

- Model and feature-schema bytes remain identical to adopted T2.
- T3 records NLL, ECE, Brier, argmax invariance, row-order hash, and separate
  model/temperature/schema hashes.
- The worker cannot emit the T3 PASS marker; only the independent verifier can
  print it after publication.
- T3 is auxiliary infrastructure and never counts as a four-method matrix cell.

## [2026-08-28] Implement Taste T7 as a classifier-interface native-GCF smoke

### Motivation

TasteMolNet T7 must exercise GCFExplainer's native complete-graph edit and
random-walk behavior against the same frozen calibrated three-class GINE used
by T3/T4. The existing BACE adapter is binary and its official reinforcement
predicate treats an importance score of at least 0.5 as a counterfactual. For
three classes, `1-p(Sweet)>0.5` does not imply that Sweet ceased to be the
argmax. The BACE route also binds a BACE-specific graph schema and adapted
NeuroSED artifact that are not Taste authorities.

### Decision

Reuse only the vendored official full-graph node/edge mutation enumeration and
VRRW loop. Add a Taste adapter that decodes each complete edited graph, scores
it with the exact held calibrated GINE, returns importance `1-p(Sweet)`, and
maintains a separate `argmax != Sweet` predicate for official reinforcement.
At the bounded alpha-one smoke endpoint, provide an all-ones neutral coverage
matrix so official transition importance is exactly the classifier score.
This is a classifier-interface/native-VRRW smoke only: NeuroSED distance,
global selection/order, full readiness, and paper eligibility are all
`NOT_EVALUATED`.

Use only the frozen train CSV for a deterministic bounded Sweet source pool.
Require the exact five-file T2 adoption and physically held T3/T4 authorities;
require both stages to bind one checkpoint and read its model, schema, label
map, split manifest, untouched test status, and validation-fitted temperature
through the retained checkpoint API. Keep validation, calibration, and test
CSV payloads unopened. RF and every BACE dataset/model/distance/output artifact
are forbidden.

Keep runtime release-disabled until a clean one-parent release child binds a
typed external authority, live controller identity, exclusive physical-GPU-1
lease, exact output root, implementation commit/tree, critical blobs, and all
T2/T3/T4 hashes. Neither CLI inputs nor environment variables are release
capabilities. The AutoDL wrapper refuses before GPU discovery; the paired Slurm
script is static refusal.

Publish only an exact private summary tree. Validate the official temporary
native `counterfactuals.pt` in memory, then discard it with the temporary
runtime. Persist no SMILES, molecule IDs, graph tensors, or reconstructable
dataset rows. The marker-last output retains opaque graph identities and
probability/decision evidence so a strict held consumer can recompute every
score and candidate predicate and reject extra files or inode drift.

### Consequences

- A binary 0.5 shortcut, deletion-only conversion, synthetic candidate list,
  BACE artifact, RF oracle, or self-declared PASS cannot satisfy T7.
- A bounded PASS proves native official mutation/random-walk execution and at
  least one valid non-Sweet complete graph; it does not prove the full method,
  NeuroSED distance, global selector, or a paper result.
- The no-redistribution policy remains enforceable without discarding enough
  opaque evidence to audit the smoke's multiclass semantics.
- T7 remains unavailable until integration and external authority review; this
  implementation does not start or authorize science.

### Status

Implemented in an isolated worktree with release pins null. No commit, push,
SSH, deployment, controller/GPU-lock/matrix mutation, scientific-root write,
model load, or science run was performed. Fresh review and stage-freeze remain
required.

## [2026-08-28] Harden the rejected Taste T7 candidate as a receipt-only successor

### Motivation

Independent review found that the first T7 stage-freeze did not install the
official module-global `importance_args` used by the real restart path, reopened
historical T2 sources, accepted Python Boolean/numeric-string aliases in parts
of its terminal schema, and linked PASS before removing a hidden prepared
marker. Its final callback also retained input descriptors but did not repeat
the immutable checkout, critical-blob, or controller full-cmdline checks.

### Decision

Base the successor on the reviewed receipt-only T2-to-T4 implementation. Hold
the exact five-file T2 root with root/gate/receipt/embedded-source pins for the
entire run; require T3 and T4 to carry its same canonical binding and checkpoint
closure. Preserve and export `TasteFrozenGINENativeAdapter` so later Taste
method smokes reuse the same original-order classifier and native graph
identity.

Scope the official `vrrw.importance_args` module global around the real walk,
restoring an old value or deleting a previously absent attribute in `finally`.
At the final marker callback, repeat clean immutable checkout validation,
reject hidden index flags, rehash critical blobs, and repeat controller
PID-generation/full-cmdline and GPU checks. Fsync the complete prepared tree,
then make PASS visible with one final `RENAME_NOREPLACE`; successful publication
has no later fallible syscall whose error can replace PASS.

The strict output consumer now rejects Boolean-as-integer and numeric-string
aliases, requires exact adapter and batch-scorer schemas, binds the adapter
checkpoint to the input checkpoint, and closes call/decode/scored-row counts.
Output paths may not overlap either the repository or official-source root in
either ancestor direction. The static release literal is checked before the
AutoDL wrapper sources `common.sh`.

### Consequences

- Historical T2 controller/training/execution roots are evidence embedded in
  the adopted receipt only; they are no longer runtime dependencies.
- The checked-in release remains `false` with every pin `null`; this hardening
  neither creates an execution lease nor starts science.
- A future T9 successor can import the adapter and held predecessor evidence
  instead of copying classification or graph-identity behavior.

## [2026-08-28] Require a physical mid-walk checkpoint for the Taste T7 smoke

### Motivation

The first bounded T7 successor executed sixteen official VRRW steps in one
uninterrupted call. Although its terminal summary could report the final step
count, it did not satisfy the four-method smoke gate's checkpoint/resume
requirement. Re-seeding and replaying from step one would not be resume: it
would repeat the initial official restart, scoring, reinforcement, and RNG
consumption while discarding the live step-eight cursor and accumulated VRRW
frequency/transition state.

### Decision

Split the same official bounded walk at exactly eight of sixteen steps. After
step eight, capture the next graph cursor and the complete mutable official
VRRW state, bridge records/counters, adapter and exact-batch-scorer counters,
native action counts, and Python/NumPy/Torch/available-CUDA RNG states. Write
that payload once to a mode-0600 file in a fresh mode-0700 private temporary
directory, fsync file and directory, and bind its SHA-256 plus complete
device/inode/mode/link/owner/size/time identity.

Raise and catch one typed planned-interruption exception only after that
durable write. Drop the in-memory checkpoint payload, reset every official,
bridge, scorer, action-count, and RNG progress component to a provably
different state, then reload only from the revalidated held checkpoint inode.
Require exact saved/restored semantic and RNG digests before continuing. Call
the real official loop for steps nine through sixteen, substituting the saved
cursor only for its initial resume entry so no fresh restart is scored or
reinforced. All later official move, teleport, restart, sampling, frequency,
and candidate-order semantics remain unchanged.

Publish aggregate-only resume evidence: native Boolean proofs for checkpoint
write, planned interruption, reload, and resume; exact `8 + 8 = 16` counts;
physical checkpoint/SHA binding; saved/reset/restored state and RNG digests;
prefix, suffix, full-trace, and boundary commitments; and a recomputable
continuity digest. The strict retained consumer requires the entire exact
schema and rejects Boolean/integer aliases, numeric strings, false counts,
restore drift, or cursor discontinuity.

Do not persist the private `.pt` payload, molecule rows, or graph tensors in
the terminal output. Keep the checkpoint inode held through continuation and
evidence construction. Do not implement a checked-path `unlink`: replacement
between identity check and deletion could target a foreign same-user inode.
Descriptor close is non-destructive; the enclosing private temporary runtime
owns ordinary lifecycle cleanup, and terminal evidence states this honestly.

### Consequences

- T7 now exercises actual progress serialization and restoration rather than
  deterministic replay from the original seed.
- A missing, byte-drifted, equal-byte-replaced, or restore-mismatched
  checkpoint fails before terminal publication.
- The final root remains summary-only and non-redistributing; checkpoint SHA,
  physical identity, and aggregate trace commitments are evidence, not graph
  payload.
- Managed controller integration, release flags/pins, deployment, GPU work,
  science, and full/paper readiness remain unchanged and disabled.

### Status

Accepted for the isolated T7 stage-freeze and hostile local tests. No commit,
push, deployment, controller mutation, or science run is authorized by this
decision.

## [2026-08-28] Replace the rejected T7 detached-marker and path-only runtime designs

**Status: REJECTED AND SUPERSEDED.** The hardlink portion of this historical
decision must not be implemented or released. The runtime-namespace finding is
retained and strengthened by the next decision.

### Motivation

Fresh review and an exact AutoDL target-filesystem probe invalidated the first
terminal primitive. A normal named file unlinked to link count zero cannot be
relinked there: `linkat(AT_EMPTY_PATH)` returned `ENOENT`, while `O_TMPFILE`
was unsupported. Only a held normal inode linked through `/proc/self/fd` with
`AT_SYMLINK_FOLLOW` succeeded. Review also found that holding only the
checkpoint file/directory did not close replacement of the complete private
runtime namespace.

### Decision

Keep one fixed mode-0600 `.PASS.authority` file named for the lifetime of the
terminal output. Fsync it and every payload before the final retained-input
closure. Publish `PASS` by exactly one no-replace proc-fd hardlink syscall;
never unlink the authority. No stat, fsync, revalidation, close, or input
cleanup error is allowed to propagate after that syscall succeeds. The strict
consumer holds both names and requires identical device/inode/bytes and link
count exactly two. The exact root contains the hidden authority explicitly,
while the payload inventory excludes authority, `PASS`, and
`output_hashes.json` itself. A cleaned scratch preflight runs on the exact
output filesystem before any model/data load or GPU science and binds its
receipt to the later held parent inode/device.

Create a dedicated `runtime` child under one private temporary envelope. Hold
the full runtime ancestor chain and the temporary parent/name, and create/open
the direct checkpoint directory and file relative to those held descriptors.
Revalidate the namespace after write and load, immediately after continuation,
and before terminal evidence. Bind the checkpoint-directory physical identity
so whole-runtime and whole-checkpoint-directory rename/equal-copy/restore
attacks fail closed. Descriptor close remains non-destructive; the restored
private envelope is removed by its owner and no checkpoint payload escapes.

### Consequences

- Pre-link failures or a preoccupied `PASS` cannot authorize the output.
- Missing, replaced, byte-drifted, or extra-hardlinked authority names and
  extra root files fail the public retained consumer.
- The implementation depends only on the exact primitive proven on AutoDL's
  `fuse.autofs`; it does not silently fall back to an unsupported kernel path.
- Release remains disabled with null pins. This decision authorizes no commit,
  push, deployment, controller mutation, GPU work, or science.

## [2026-08-28] Move T7 to managed execution v2 and real Taste NeuroSED

### Motivation

The final security review forbids any mutable or worker-owned hardlink terminal
primitive. It also requires T7 to preserve the official GCFExplainer distance
semantics: a neutral coverage matrix is not a Taste GCF smoke. Worker-generated
scientific evidence cannot authorize its own adoption.

### Decision

T7 now depends on the frozen managed execution v2 API. Each run creates one
unreused UUIDv4 attempt and worker staging generation. The worker is limited to
`raw_evidence.json`, `worker_exit.json`, and `SEALED.json`; it imports neither
the SEALED opener nor the atomic verifier/publisher. Raw evidence binds the
expected final path and the path/kind/canonical SHA of the independently
verified Taste NeuroSED predecessor. A pure T7 method verifier recomputes the
three-class GINE, official full-graph, NeuroSED, split-isolation,
checkpoint/resume, and predecessor bindings. Only a separate process may pass
that verification to managed v2 for same-filesystem no-replace rename or
cross-filesystem copy/fsync/rehash plus no-replace atomic rename.

The scientific bridge retains the same calibrated three-class GINE and exact
`predicted_label != 1` candidate predicate with `1-p_source` classifier score.
It loads the held Taste-specific `NormGEDModel(8,input_dim,64,64)` checkpoint
through descriptor authority and calls the vendored official normalized
threshold-coverage implementation. Calibration and test remain unopened by
T7, and the NeuroSED PASS must prove train-only fitting and validation-only
selection.

Every VRRW progress checkpoint now lives under a fresh
`checkpoints/<checkpoint_uuid>/` with a canonical UUIDv4 and generation token.
The temporary parent, runtime root, checkpoint container, UUID directory, and
file identities remain held and are revalidated after write/load, immediately
after continuation, and before raw evidence. Runtime/container/directory
rename-plus-equal-copy-plus-restore is rejected, and temporary cleanup leaves
no graph/checkpoint payload in managed evidence.

### Consequences

- T7 contains no terminal hardlink publisher and cannot issue PASS, a final
  gate, verification, adoption, or release authority.
- A missing managed-v2 PASS, NeuroSED PASS/checkpoint, or API signature blocks
  before model/data science.
- The checked-in release remains disabled and all deployment/predecessor pins
  remain null. No science, deployment, remote command, or GPU launch is part of
  this code-only change.

## [2026-08-28] Adopt the unchanged generic ChemLLM base as current-campaign T5

### Motivation

The current TasteMolNet campaign needs a clean policy predecessor, but a
zero-step LoRA would add randomly initialized adapter state without learning
from any example. Calling that artifact SFT would overstate the science. The
operator explicitly permits a clean generic-base adoption PASS instead of a
training run. The existing T5 initializer remains useful historical code but
is release-disabled and requires predecessor/controller authorities that are
not needed to prove the identity of an unchanged dataset-independent base.

### Decision

Add a separate managed-execution-v2 route whose only accepted semantic state
is `ADOPTED_CLEAN_GENERIC_BASE`. It opens no Taste file, performs zero optimizer
steps, uses no RF or GINE reward, creates no adapter, copies no model weights,
and does not count as one of the sixteen method/dataset cells. Its operational
verifier marker is `[TASTE_T5_CLEAN_SFT_PASS]`, but the structured evidence
must always record `training_performed=false`, `optimizer_steps=0`, and
`taste_splits_loaded=[]`; the marker must never be interpreted as an SFT claim.

The worker hashes every physical file in the external ChemLLM tree and writes
only `source_inventory.json` and `clean_base_adoption_candidate.json` below the
managed artifact directory. Source validation requires an InternLM causal-LM
config, one InternLM tokenizer asset, a safetensors index that exactly closes
all and only the model shards, exact agreement between the index tensor map and
every shard header, and agreement between indexed total size and tensor payload
bytes. Adapter/PEFT/LoRA names and metadata, Taste/RF identifiers, and dataset
payload paths or file types are rejected.

The worker has no terminal API and stops after managed-v2 raw evidence,
zero-exit evidence, and `SEALED.json`. A separate clean-checkout verifier holds
the source tree by descriptor, repeats the complete content inventory, checks
the launcher/worker EXITED lineage and exact attempt inputs, and revalidates
the source stat/ctime closure immediately before invoking the frozen atomic
no-replace publisher. Only that verifier writes `verification.json`,
`gate.json`, and `PASS`. The terminal receipt contains hashes and metadata,
never safetensors or tokenizer/model payloads.

### Consequences

- T5 can close without occupying GPU 1/2 and without opening train,
  validation, calibration, or test data.
- The external ChemLLM directory remains the downstream read-only model
  authority; moving or changing any file invalidates the receipt's source pin.
- This adoption does not inherit the old LoRA initializer's T3/T4 binding.
  A later T6 release must independently hold and cross-bind this T5 source
  receipt with the selected T3/T4/GINE authorities before model loading.
- The historical initializer and its disabled release contract are preserved;
  this decision neither rewrites it nor labels it PASS.
- The implementation commit authorizes no deployment, GPU work, Taste data
  access, or scientific/matrix result. A real PASS requires the independent
  AutoDL verifier run against the exact private source tree.

## [2026-08-28] Harden the T8 official boundary and adopt managed execution v2

### Motivation

Fresh review rejected the T8 successor because the concrete official adapter
did not yet expose the reviewed resume/completion callback contract, an output
could derive its public expectations from worker-controlled evidence, and
official Python imports could reuse a preloaded or bytecode-shadowed module.
The frozen rejected identity was base commit
`6db268e4ef6d3a0c4b4d80f3133476815c8d2b9c`, staged tree
`9ef28ebf6d1b1c6b8e5d17387253008b2425d86f`, and cached-patch SHA-256
`1c423f0697b37b328336f2a9edfb189ed5400cc55c382b5c1e507805823ea3fa`.

### Decision

Pin official GlobalGCE commit
`157e65c2850bc787f229a1ee8c60564906b933f2` and compare the exact reviewed
constructor/function signatures with `inspect.signature` before training.
The concrete generator now has explicit checkpoint and completion callbacks;
planned interruption, held-checkpoint reload, and the terminal completion
callback are tested without a `TypeError` retry or variadic fallback.

Run the official entrypoint with `python -I -B` and `PYTHONNOUSERSITE=1`.
Descriptor-rooted imports reject foreign preloads, bytecode/native shadows,
ignored or untracked runtime code, loader/origin drift, and inode/hash changes.
Both target branches record the exact API document and full module provenance
for official sources, globalgce, torch, torch_geometric, the project adapter,
bridge, and oracle, and must agree on those documents.

Adopt the exact managed-execution-v2 API frozen at commit
`3405ae1d24fdaeb7a4af40b14823b36051966a35`. The scientific worker can write
only raw evidence, exit evidence, and SEALED inventory. A separate verifier
must hold and revalidate external task/run/GPU/ACTIVE plus T2/T3/T4/GINE/train/
official/policy authority, validate the two destination branches and one
three-class GINE, and then invoke the verifier-only atomic directory publisher.
No hardlink, worker-created gate, or worker-created PASS is an accepted route.
The release boundary exposes only a narrow held external-authority protocol;
it rejects raw mappings and managed-v1 holder shapes and recomputes a closure
hash over task/run/GPU/ACTIVE child, execution, fresh roots, T2--T4, GINE,
train, official, and policy identity. No controller-side implementation is
claimed or included in this commit. The same held adapter must provide an
official API/import expectation captured outside the worker; the verifier
requires exact equality with the full startup bundle before it can publish.

### Consequences

- Sweet-to-Bitter and Sweet-to-Tasteless remain separate native GlobalGCE
  branches over the same frozen three-class GINE; merge, canonical dedup, and
  original-order untargeted strict-flip validation are unchanged.
- A wholly rehashed worker tree cannot supply its own expected authority, and
  preloaded `models.*`/official `utils` or malicious `.pyc` origins fail closed.
- Release remains disabled. A reviewed managed-v2 controller adapter for the
  exact GPU/ACTIVE authority and an AutoDL target-filesystem atomic-rename
  preflight are still required. The blocked predecessor marker primitive and
  managed-v1 registry are deliberately not connected.
- This change performs no deployment, science run, controller mutation, or
  downstream release.

## [2026-08-28] Train a Taste-specific NeuroSED without calibration/test leakage

### Motivation

Projecting an AIDS, Mutagenicity, or BACE NeuroSED checkpoint into Taste's
feature space would not provide a dataset-specific auxiliary distance model,
while replacing NeuroSED with deletion-only distance would change the official
GCFExplainer method. Taste therefore needs a fresh auxiliary model without
changing the frozen three-class calibrated GINE or exposing held-out data.

### Decision

Pin GREED commit `1c756f49625abb62c9f6de5b0059876a4c7499c1` and its
experiments commit `e85423dc943fda1979811e7449846efffec2a1e1`. Preserve the
eight-layer GIN, 64-dimensional hidden/output representations, directional
NormSED training forward, interval criterion, AdamW, CyclicLR, and exact 0.1
gradient clipping. Export a plain state dictionary whose parameter schema is
strictly isomorphic to the bundled GCF fork's NormGED loader. Preserve the
fork's downstream division by the sum of graph element counts.

Derive one-hot explicit-hydrogen atom channels only from train and reject
validation-unseen atoms. The initial implementation constructed deterministic
connected induced BFS subgraph-to-own-parent pairs separately within train and
validation and treated the omitted-node-plus-omitted-edge count as an exact
bound. The successor decision immediately below records why that ordering is
invalid under the pinned directional costs and blocks it from launch. Labels
and the GINE are not used. Train only on train and use validation only for
early stopping and checkpoint selection. Never open a calibration/test
payload, ID, SMILES, graph hash, pair, label, or embedding.

Create every selected checkpoint under a new UUIDv4 directory. Publish the
selected bytes once as `best.pt` for GCF. Run the route on physical AutoDL GPU
1 under the shared UUID lock and managed execution v2. The worker produces
scientific files plus raw/exit/SEALED evidence but cannot sign PASS. A separate
verifier closes hashes, reloads both training and runner schemas, checks finite
validation error/rank, batch/single and CPU/GPU tolerance on synthetic probes,
and atomically publishes the terminal directory. Failures quarantine without
signals. Slurm remains an explicit static refusal.

### Consequences

- NeuroSED remains an auxiliary distance model and never becomes a classifier
  or matrix method cell.
- The architecture and GCF checkpoint loader remain compatible, but the
  successor below blocks any claim that pair sampling/direction provides full
  official NeuroSED training semantics.
- Public artifacts expose aggregate hashes/metrics only, not reconstructable
  Taste rows or pair data.
- This implementation does not launch training or assert a scientific PASS;
  T7/T12 remain blocked until a fresh independent-verifier terminal is bound.

## [2026-08-28] Block NeuroSED launch pending directional-pair and controller review

### Motivation

Successor review found that the initial nested-pair order was scientifically
invalid. GREED assigns zero insertion cost and unit deletion cost, so the
recorded omitted-node-plus-omitted-edge target is not the SED of
`(subgraph, parent)`; that ordered distance is zero. Review also found that a
published managed-v2 final cannot be opened with the staging-only SEALED
consumer, that T2/T3 lineage and independent data replay needed stronger
closure, and that an immutable attempt heartbeat H1 must not be confused with
the later worker/verifier heartbeat generation.

### Decision

Represent the exact nested deletion target only as the explicit Taste
adaptation `directional_exact_deletion_v1`, ordered `(parent, subgraph)`, with
unit node/edge deletion, zero node/edge insertion, unit node relabel, and zero
edge relabel costs. Keep the checked-in value
`PENDING_SCIENTIFIC_REVIEW`; the launcher fails before GPU discovery unless a
reviewed configuration and an explicit environment selection both choose the
adaptation. Do not implement or silently select an independent-pair/pyged
route in this successor.

Pinned upstream `make_inner_dataset` instead independently samples a query
subgraph and random target and obtains SED bounds from `pyged`. T7 runtime
embeds original parent/targets and evaluates generated graphs as queries, so
its direction is generated-query to original-target, the opposite of the
exact-deletion adaptation's training order. Record both mismatches in pair
manifests, the model card, and the independent gate. Passing loader/reload
tests proves state-dictionary compatibility only and must not be called full
official NeuroSED semantics.

Keep `reviewed_taste_epoch_level_adaptation_v1` honest: full-validation,
epoch-level selection changes the checkpoint/stopping cadence and can change
the optimization trajectory relative to GREED's batch-interleaved loop. It is
not described as an unchanged upstream loop and also requires scientific
review.

Add a descriptor-retained published-final consumer that validates the generic
managed gate, verification, PASS, generation, SEALED source inventory,
published inventory, directory digests, and required artifacts without
retaining one duplicate ancestor chain per checkpoint. T7 consumes that one
generic NeuroSED final and no NeuroSED-specific PASS type. Bind the authentic
T2 adoption/source and T3 managed final, require byte-identical split
manifests, hold train/validation and configuration bytes across parse/hash/use,
and have the verifier independently reconstruct the train vocabulary/pairs and
validation pairs/metrics from those held bytes.

Record the worker-initial heartbeat H1 as an immutable attempt input and worker
latest/verifier terminal generations separately. Require stable receipt and
process identity plus monotonic `H1 <= worker_latest <= verifier_terminal`, not
equal heartbeat hashes. Do not duplicate the main-v2 controller implementation:
launch remains disabled until the shared holder verifies the external launcher
receipt, full heartbeat chain, and an ACTIVE GPU1 lease bound to physical
index/UUID and the actual managed worker PID generation, attempt, and
generation token.

### Consequences

- This successor is code/test/documentation only and is safe to cherry-pick,
  but it is intentionally not a production launch commit.
- The user must decide whether to approve the directional nested-deletion pair
  adaptation despite its upstream-sampling and T7-runtime-direction mismatch,
  and whether to approve the epoch-level selection adaptation, or commission
  an upstream-faithful alternative.
- Under the current full-official T12 requirement the adaptation is a research
  draft/negative control only: the verifier requires exact upstream pair,
  pyged, direction, and batch-interleaved-selector provenance and therefore
  hard-rejects this bundle before generic PASS.
- Root integration must adapt the NeuroSED launcher, worker, and verifier to
  the final shared controller-holder API before any GPU allocation or science.
- No checkpoint, scientific PASS, T7 release, or matrix result is claimed.

## [2026-08-28] Supersede the own-parent NeuroSED route with independent fixed-budget pairs

### Motivation

The production Taste GCFExplainer requirement is full GREED/NeuroSED pair and
label semantics under a bounded GEDLIB cost. The previous research adaptation
paired a parent with its own sampled subgraph and used an exact deletion count.
That is neither upstream `make_inner_dataset` sampling nor the generated-query
to original-target direction consumed by GCF. Running an exhaustive Taste
train-by-train product is unnecessary and operationally unsafe, but replacing
it with the own-parent shortcut would change the scientific method.

### Decision

Supersede the own-parent route for production with deterministic seed-7
fixed-budget independent query-source/target sampling inside train or inside
validation. Draw both roles with replacement from the complete same-split
sequence through separate deterministic RNG streams and require distinct graph
IDs. Compute size/class strata only after the ordered pair decisions are
complete: neither field may select, filter, rebalance, reroll, order, or enter
the scientific pair identity. Reject any top-level seed other than exactly 7,
self-hash both sampler manifests, and require the readiness verifier to reopen
and cross-bind their contents. Never materialize the Cartesian product, and
leave every pair unlabeled until authenticated `pyged.sed` returns its lower
and upper F2 bounds. Preserve the directional zero-insertion/unit-deletion SED
costs and generated-query-to-original-target runtime direction.

Treat the finite pair count as the only resource-control extension. Determine
it from disjoint real 100/500/1000 GEDLIB benchmarks, worker contention gates,
and the 24-hour p95 projection. Only 5k/10k/20k train budgets and their
1k/2k/4k validation budgets are legal. A failed minimum tier produces
`BLOCKED_GEDLIB_THROUGHPUT`; a missing real build produces
`BLOCKED_GEDLIB_BUILD`. Neither condition authorizes approximate, neural,
own-parent, symmetric-cache, or fabricated labels.

Build only from pre-provisioned pinned sources in a fresh environment. The
local source snapshot lacks GEDLIB and pybind11, so this commit records a real
blocked smoke with no PASS marker and starts no remote science. Keep the
existing epoch-level trainer blocked as a research adaptation; production
still requires the pinned GREED batch-interleaved selector, compact directional
label cache/reserve handling, independent verification, and managed release.

### Consequences

- The deterministic sampler, benchmark authority, and budget planner can be
  reviewed and tested without RDKit, PyTorch, pyged, or network access.
- No build, benchmark, pair-builder, fixed-budget model, T7, or T12 scientific
  PASS is claimed by code integration.
- AutoDL must provide and approve an exact GEDLIB commit and pybind11 package,
  then run the real benchmark tiers before a budget or training run exists.
- The absent upstream GCF commit remains explicit: the current vendored
  snapshot is bound only by critical source hashes and cannot be assigned a
  fabricated commit identity.

Pure local contracts now make the remaining scientific boundary executable
without weakening it. Successful pyged observations retain lower and upper
bounds plus the exact/bound flag and undergo only the upstream float32 storage
cast. Reserve replacement is first-success-in-sampler-order; asymmetric SED
keys never share reverse entries. A GREED selector state machine enforces one
validation event before each training batch, strict-lower checkpointing, and
the upstream `>` patience stop before its paired update. The GCF binding embeds
original inputs as targets and exposes generated candidates only as queries.

These components emit readiness metadata, not PASS. The trainer, compact label
writer/reopener, official fixture comparison, T7 binding, and managed verifier
still need integration with real artifacts. The readiness model-card gate also
requires a full upstream GCF commit; `UNAVAILABLE_FROM_VENDORED_SNAPSHOT` is an
intentional hard failure rather than a value that critical blob hashes can
silently replace.

## [2026-08-28] Authenticate Taste method control authority with generations

### Motivation

Managed-v2 attempts previously cross-checked an operator-provided
`controller_id`, which did not prove that a live controller generation held the
GPU assignment at worker or verifier release time.

### Decision

Create one immutable main-v2 receipt for the clean Git controller process and
append a new UUID/sequence heartbeat every ten seconds. Bind each heartbeat to
the receipt SHA, prior heartbeat SHA, process generation, policy facts, and at
most two fixed task/GPU leases. Consumers hold exact receipt, heartbeat, and
lease files with `O_NOFOLLOW`, verify the live process generation, and adopt
only a complete append-only successor chain. T4 holds this authority throughout
worker science and independent verification and revalidates immediately before
terminal publication.

### Consequences

- An arbitrary controller string can no longer release T4.
- This historical main-v2 draft assigned NeuroSED/GPU1 and T4/GPU2; the
  release-v3 successor decision at the top of this file supersedes that map
  with T4/GPU1 and NeuroSED/GPU2 while GPU0/GPU3 remain protected.
- Status is read-only and every anomaly is quarantined without process signals.
- The controller may monitor dependencies, but it is not yet the full T6--T16
  main-table scheduler.

## [2026-08-29] Treat RF as a provenance token in BACE GINE manifests

### Motivation

The BACE GCFExplainer worker naturally completed all 50,000 generation steps
and its official summary passed, but candidate freeze rejected that clean
summary. The generic provenance visitor searched for the two letters `rf`
anywhere in a key, so ordinary fields such as `counterfactuals_path`,
`counterfactuals_sha256`, and `available_model_counterfactual_count` were
misclassified as RandomForest evidence. This was a post-generation gate bug;
the completed scientific generation and summary were not corrupt.

### Decision

Interpret an RF key only when normalized underscore-delimited key tokens contain
the standalone token `rf`, or when the key explicitly names `random_forest` or
`randomforest`. Preserve the existing exact forbidden RF string values and
serialized model/file-name checks. Keep `rf_oracle_used=false` mandatory and
retain the frozen GINE checkpoint, train-only, calibration-closed, and
test-closed requirements.

Resume only the missing candidate-freeze and downstream stages from the held
50k generation and passed summary. Do not regenerate any VRRW step.

### Consequences

- Legitimate counterfactual field names no longer block a GINE-clean manifest.
- Keys such as `historical_rf_checkpoint`, explicit `teacher_backend=rf`, and
  RF model filenames still fail closed.
- The repair changes no candidates, ranking, classifier, dataset, split,
  calibration, test, or matrix result; a real cell PASS still requires the
  existing downstream gates and atomic matrix refresh.

### Status

Accepted

---

## [2026-08-29] Keep BACE GlobalGCE K fixed while extending raw native rules

### Motivation

The previous BACE GlobalGCE science attempt trained the pinned official model
against the frozen GINE but its twenty decoded tensors did not yield twenty
valid native LHS-to-RHS rules. Treating that failed catalog as K20, lowering K,
duplicating rules, or choosing a retry from test behavior would corrupt the
paper cell. A new run also needs a real GPU2 science child rather than a live
controller with no child.

### Decision

Keep final K exactly 20 and use one bounded train-only raw-rule escalation:
seed 7 contributes 80 raw slots; if needed seed 17 contributes 120 more for a
cumulative 200; if still needed seed 27 contributes 300 more for a cumulative
500. Each increment runs in a fresh exact-top-k root with the same frozen
360-parent BACE train cohort, min-frequency 7, 100 epochs, pinned official
GlobalGCE, and frozen BACE GINE. Calibration and test paths are not accepted by
the controller and cannot affect continuation.

Hold physical GPU2's one hard-coded canonical non-blocking `flock` for every
child lifetime. Bind protected GPU0/GPU3 process generations to the live GPU
UUID/PID inventory and expected BACE method commands. At every heartbeat GPU2's
compute set may contain only the bound science child, and it must be empty after
child exit and before release. The controller has no
child-signal API. It must enter with one OS thread, block SIGINT/SIGTERM/SIGHUP
before later threads can inherit a different mask, and synchronously drain
pending signals into deferred stop requests; the raw-round process unblocks
them only for itself after exec. The lock remains held until the child exits
naturally. Nonzero science exit is accepted only as exact code 20 plus a
structured hash-closed shortfall receipt and last-written marker, never a log
phrase.

Reopen each catalog row through the native rule decoder, recompute its content
hash and selector chemistry, reject identity or provenance drift, and
deduplicate by canonical transformation content excluding caller-controlled
candidate ID and native index. Repeat clean Git, config, Python, complete train
contract, every tracked official Python source/import authority, every required
frozen-GINE bundle file, GPU lease, and protected-role checks immediately before
the final marker. Pre-marker artifacts use only
`SEALED_CANDIDATE`/`RELEASE`; no `_RUN_COMPLETE.json` or PASS heartbeat is
written. Publish the first 20 transformations in deterministic
round/seed/native order, then write PASS as the sole and final commit point.

### Consequences

- `[BACE_GLOBALGCE_K20_EXTENSION_LAUNCHED]` requires a real child PID/start
  ticks and fresh root; it is not a scientific PASS.
- `[BACE_GLOBALGCE_K20_PASS]` requires exactly twenty unique hard-validated
  rules, an unchanged frozen checkpoint, and train-only provenance.
- Exhausting budget 500 below K20 is an explicit blocker; it does not permit a
  smaller, duplicated, invalid, or test-selected universe.
- This is a candidate-pool resource extension, not a change to GlobalGCE's
  paper selector and not a GNN-backbone ablation.

## [2026-08-29] Replace mutable AIDS handover assertions with durable evidence

### Motivation

The first dedicated old-brute handover gate stored its restart smoke and
ten-minute progress window in mutable `state.json`.  Their hashes were public
and recomputable, so a same-authority caller could replace the state, backdate
the claimed window, and re-sign it without proving a second controller
reattachment or real elapsed progress.

### Decision

Remove mutable resume receipts entirely; retain the mutable progress monitor
only for ordinary route diagnostics and exclude it from handover eligibility.
Under the already bound controller owner/root claim and
physical `gates/` inode, publish each controller generation, each exact-worker
reattachment, and each increasing authenticated checkpoint observation as a
0600 `O_EXCL` record with a complete write, file `fsync`, and parent-directory
`fsync`.  Bind every successor to the prior record's content SHA-256 and full
device/inode/mode/uid/gid/nlink/size/mtime/ctime identity, plus unpredictable
attempt and nonce values.  Each observation reopens the live controller and
exact-worker PID/start/cmdline identities and the authenticated DBSCAN
checkpoint hashes.

Use a separate read-only verifier that accepts no `state.json` argument.  It
reopens the latest two controller generations, immutable resume receipt,
append-only observation chain, live controller/worker procfs identities, and
current checkpoint closure.  Continuous time, cadence, freshness, throughput,
and ETA come only from filesystem ctime differences and strictly increasing
authenticated checkpoint progress; payload timestamps cannot contribute.  The
chain head is only a successor seal, so every observation used for a decision
is itself content/stat-bound by a later immutable record.

### Consequences

- Re-signing mutable state can no longer make the old-brute gate eligible.
- Zero/partial files, copy replacement, symlink/hardlink substitution, inode
  reuse/ABA, chain gaps, and payload timestamp backdating fail closed.
- A fresh route must survive a genuine controller restart and then accumulate
  at least ten minutes of fresh, at-most-two-minute-gap observations.
- The output budget now reserves the bounded generation/resume/observation
  inventory.  This controller still sends no old-route signal; deployment and
  any one-time graceful stop remain separately reviewed actions.
## [2026-08-29] Materialize the Taste NeuroSED feature schema before fixed-budget sampling

### Motivation

The available T3 GNN cache uses the separate `molecular_graph_v1` categorical
schema and cannot serve as the explicit-hydrogen, one-hot atom vocabulary that
the fixed-budget NeuroSED pair builder requires.  Train and validation pair
sampling therefore lacked one concrete, split-closed schema authority even
though the pinned non-MIP GEDLIB build was available.

### Decision

Add one thin producer that opens only SHA- and role-bound Taste train and
validation CSVs through the shared split loader, rejects cross-split molecule
ID overlap, and calls the existing `derive_feature_schema` implementation.
Publish exactly one canonical `tastemolnet_gcf_neurosed_feature_schema_v1`
document with atomic no-replace semantics.  Its aggregate receipt records that
calibration and test were not opened and contains no molecule IDs or SMILES.
The CLI accepts only the tracked `configs/hpc.yaml` plus
`inference.fallback_to_heuristic=false`.

Freeze `n_hops=5` and `traversal=0.5` as project-preregistered fixed-budget
sampling choices.  They are not represented as upstream GREED/AIDS defaults.

### Consequences

- Train and validation pair builders can bind one real feature-schema SHA
  without opening calibration or test payloads.
- This producer does not create GED labels, a NeuroSED checkpoint, or a T7
  result and therefore cannot publish a scientific PASS.
- AutoDL must still materialize the schema from the immutable execution
  checkout and pin its hash into both split-local pair manifests.

### Status

Accepted

---

## [2026-08-29] Bound Taste T4 CUDA batch roundoff without accepting class drift

### Motivation

The first real A800 adaptive T4 attempt reached the frozen three-class GINE
oracle but failed its batch-versus-single probability check.  A direct replay
over the same parent and residual graphs measured a maximum absolute difference
of `1.0575430420267651e-7`; every argmax class agreed.  The former `1e-7`
absolute threshold therefore rejected CUDA/PyG reduction-order roundoff rather
than a scientific prediction change.

### Decision

Freeze the T4 batch/single absolute tolerance at `1e-6` with zero relative
tolerance, finite values, and exact shape closure.  Independently require every
three-class argmax to agree.  Apply the same constant in bounded science,
adaptive science, and aggregate verification.  Any class drift remains a hard
failure even when the probability difference is below `1e-6`.

### Consequences

- The measured A800 tail is accepted without changing the checkpoint, cohort,
  deletion search, strict-flip thresholds, or destination policy.
- NaN/Inf, shape drift, probability drift above `1e-6`, and any argmax drift
  still fail closed.
- The failed output namespace and quarantined controller are not reused; a
  runtime PASS still requires a fresh managed attempt and independent verifier.

### Status

Accepted

---

## [2026-08-30] Use one fixed-task continuation sidecar for the live main table

### Motivation

The live AutoDL routes already have independent owners. The existing Taste-v2
controller cannot accept the required T6/T8/T9 queue, while invoking unreleased
wrappers on every poll would create noise and stale attempts. T9 additionally
requires a new stage, final, and run UUID after every GPU-preflight return code
75.

### Decision

Add one bounded sidecar whose task table is compiled as NeuroSED, T9,
T6/T7/T8, and T10. Keep T6/T7/T8 terminally `BLOCKED_RELEASE`; keep T10
waiting; launch NeuroSED only from a spec-pinned label manifest and argv under
the existing UUID lock; and launch only the existing T9 GPU1 wrapper with one
UUIDv4 shared across its fresh stage/final/run identities. Persist the child
return code independently so return code 75 abandons those identities across
sidecar restarts.

Observe AIDS exact, BACE GCF, GlobalGCE, and ComRecGC without writing their
roots. Persist the ComRecGC step-17500 registration. When the immutable spec
also pins the resolved config, trace, checkpoint roots, and their frozen
hashes, invoke the read-only convergence library at 2,500-step committed
boundaries. Retry `NOT_READY`, advance only after `CONTINUE`, and retain a
genuine convergence result for a separately managed exact-PID handover. Reject
GNN ablation and provide no matrix publication or process termination
operation.

### Consequences

- GPU0/GPU1 locks and compute PIDs are both required idle before launch.
- A missing NeuroSED trainer argv remains explicit `WAITING_INPUT` rather than
  starting a guessed command.
- Every T9 preflight return code 75 is auditable and the next attempt is fresh.
- The convergence sidecar reads only CLOSED trace/checkpoint evidence and
  writes each attempt to a fresh audit root; it never signals the live worker.
- The sidecar is specific to this continuation and is not a new controller
  platform.

### Status

Accepted

---

## [2026-08-30] Audit BACE ComRecGC convergence from committed trace evidence only

### Motivation

The protected BACE ComRecGC trajectory is expensive, but a convergence check
must not deserialize its multi-gigabyte checkpoint state, mutate its output,
or infer stability from a file that the live writer may still be publishing.
The preregistered gate also requires deterministic candidate ranks and two
consecutive windows rather than a single favorable snapshot.

### Decision

Add `src/eval/bace_comrecgc_convergence.py` as a pure, importable audit library.
The caller must freeze the exact resolved-config and ordered 360-parent hashes
and request a 2,500-step evaluation boundary. The audit accepts the three
required 500-step checkpoint generations only through paired local/mirror
completion evidence or paired retention histories. It validates declared
payload sizes but never opens or rehashes `generation_state.pt` or the SQLite
graph store.

A selected-action trace part is CLOSED only when its immediate numeric
successor exists. Every covered move is exactly one headless teleport or five
selected transitions with heads 0 through 4. Selected rows require exact action
resolution, a nonempty recorded action, a frozen parent, and lowercase source
and target graph SHA-256 identities. Candidate frequency is reconstructed with
`Counter` and ranked by `(-frequency, SHA-256)`. The historical
`rank_spearman` field remains the direct Pearson correlation of deterministic
Top100-union ranks with missing rank 101; it does not call SciPy or rerank ties.
Coverage is the distinct frozen parents attached to the current Top20 divided
by 360.

The library writes `convergence.json` only below a caller-provided fresh audit
root. It writes `CONVERGED_EARLY_STOP.json` only when both latest windows meet
all preregistered thresholds. It has no signal API and cannot modify the live
generation, trace, checkpoint, controller, or test artifacts.

### Consequences

- A last trace part without a successor is never consumed, even when it already
  contains 512 parseable rows.
- Pruned checkpoint generations remain usable only through matching paired
  retention receipts; at least one of the three generations must still expose
  a live provenance manifest.
- A convergence receipt authorizes only the caller's separately reviewed
  exact-PID graceful-stop boundary; the evaluation library itself sends no
  signal.
- Test data, state tensors, RNG state, and the authoritative graph store are
  absent from the convergence decision.

### Status

Accepted

---

## [2026-08-30] Persist the BACE ComRecGC 20k/25k resource-cap gate read-only

### Motivation

The BACE ComRecGC run has an explicitly bounded resource policy in addition to
the pre-registered convergence gate.  The first committed checkpoint at or
above 20,000 steps may be adopted when lineage errors are zero and at least ten
valid unique candidates exist.  An insufficient 20,000-step checkpoint may
continue only to 25,000; an insufficient 25,000-step checkpoint is a scientific
failure.  This decision must be durable before any separately authorized
process handover.

### Decision

Add a pure decision function and a narrow read-only observer.  They consume
only the existing convergence hook receipts, bind the exact committed step and
checkpoint digest, and write a handover request outside the live generation
root.  Pre-20k termination remains possible only through the unchanged
two-window convergence PASS.  The observer exposes no signal or subprocess
surface and records `signals_sent=false` and `postprocess_started=false`.

Keep exact-PID `SIGTERM`, checkpoint materialization, and downstream
post-processing outside this read-only component.  A handover request is not a
cell PASS and requires a separate executor with explicit authority and an
implemented cap-finalization path.

### Consequences

- The 2,500-step convergence cadence and its calibration/test isolation remain
  unchanged.
- A 20k gate failure cannot silently extend past 25k.
- The observer can be deployed now without touching or stopping the live run.
- Automatic handover remains explicitly blocked until its separate execution
  boundary is reviewed and installed.

### Status

Accepted

---

# 2026-08-30: Deadline heartbeat uses atomic rename without FUSE fsync

- Scope: the read-only `deadline_main_completion_v1` controller receipt and
  heartbeat only; science checkpoints, matrix publication, and result exports
  retain their existing durability rules.
- Motivation: AutoDL mounts the persistent control root through `fuse.autofs`.
  The first 723-byte receipt blocked indefinitely in `fsync()` while unrelated
  multi-gigabyte checkpoint I/O was active, preventing any heartbeat.
- Decision: durable one-time receipts retain a same-directory temporary and
  atomic rename. The replaceable heartbeat is written directly because FUSE
  rename can also block behind unrelated transfers; a partially observed
  heartbeat is ignored and replaced on the next 60-second cycle. No science or
  matrix artifact uses this relaxed heartbeat-only path.

---

## [2026-08-31] Resume AIDS from a descriptor-bound checkpoint and run Mut as exact multi-component DBSCAN

### Motivation

The AIDS exact worker reached 20,512,768 committed recovery rows, but its
controller observed the checkpoint pathname during a legitimate atomic
replacement and stopped with `path identity changed`.  The payload hash,
scientific identity, vector hash, and progress ledger remained closed.  The
completed Mutagenicity pair store is scientifically different: its first
adaptive scan found 64,821 failures in 65,536 rows, so the AIDS
single-component shortcut and its failure cap cannot represent that dataset.

### Decision

For AIDS, retain strict inode/mode/size/mtime/ctime and SHA checks, but bind an
already-promoted checkpoint by opening it with `O_NOFOLLOW`, reading and
hashing the opened descriptor, and requiring the final pathname to still name
that exact `fstat` generation.  Retry only when a concurrent atomic generation
is rejected.  A stopped nonzero checkpoint must receive an independently
published verifier receipt before the resume-only exact-stage entrypoint may
consume it.

For Mutagenicity, adopt the completed pair store read-only and use the named
`sklearn_float64_exact_multi_component_v1` route.  It converts the production
vectors to float64, uses sklearn brute Euclidean radius queries with at most
four workers, and runs the existing exact neighbor-count, core-union, and
border-assignment passes.  It never invokes a single-component shortcut or a
failure cap.  Deterministic production-derived subsets must match sklearn
float64 labels and pass an independent terminal reload before full launch.

### Consequences

- AIDS retains the existing 20,512,768-row checkpoint; neither epsilon nor
  checkpoint identity is relaxed.
- Mutagenicity may contain any number of exact DBSCAN components and discloses
  the float64 reference route and four-worker ceiling in its manifests.
- Neither route regenerates the completed pair stores, reads test data, or
  changes classifier/split/oracle semantics.

### Status

Accepted

---

## [2026-08-31] Bind the Taste T7 smoke successor to physical GPU0

### Motivation

GPU1 is reserved for the independent Taste T14 full-generation route.  The
authorized shortest schedule runs T6, T8, and T7 serially on GPU0, while the
already-adopted NeuroSED model makes T7 independent of any new training.

### Decision

Change only the dataset-specific T7 physical-device contract from index 1 to
index 0 across its frozen smoke constant, typed controller/GPU receipts,
runtime environment check, disabled release candidate, AutoDL wrapper, tests,
and documentation.  Keep logical model device `cuda:0`, exclusive UUID lock,
official GCF/VRRW semantics, frozen three-class GINE, and generated-query to
original-target NeuroSED direction unchanged.  This implementation commit
does not enable the release; a separate pinned successor is still required.

### Consequences

- T7 can follow T8 on GPU0 without competing with T14 on GPU1.
- A GPU1 lease, mismatched UUID, or environment remap fails before science.
- No NeuroSED retraining, split change, oracle change, or paper-result claim is
  introduced.

### Status

Accepted

---

## [2026-08-31] Stabilize T8 checkpoint metadata before publishing resume evidence

### Motivation

A fresh TasteMolNet GlobalGCE target-0 run completed exact top-k mining and
wrote a valid epoch-0 checkpoint, but the planned-resume holder rejected it
after the callback unwound.  The preserved checkpoint reloads with the expected
`next_epoch=1` and resume identity, no temporary or alternate checkpoint exists,
and both checkpoint leaves show the same approximately three-millisecond gap
between write mtime and final ctime on the AutoDL persistent filesystem.  The
old error combined every stat field and SHA under the misleading message
`planned checkpoint bytes changed`.

### Decision

Before publishing checkpoint-file evidence to the T8 callback, retain the
opened leaf and require two consecutive identical observations.  During this
short bounded barrier, permit only a monotone `ctime_ns` settlement.  Device,
inode, mode, owner, link count, byte count, mtime, and SHA-256 must remain exact;
any change fails immediately.  Once settled evidence is published, the existing
T8 retained-descriptor and named-leaf comparisons remain fully strict.

Do not reuse the failed attempt, alter the official GlobalGCE algorithm, move
training checkpoints into scratch, or relax checkpoint hashes.

### Consequences

- Delayed metadata publication cannot be mistaken for checkpoint corruption.
- In-place byte changes, atomic leaf swaps, and SHA mismatches still fail.
- Persistent checkpoint/result semantics and the frozen GINE/split/oracle are
  unchanged.

### Status

Accepted

---

## [2026-08-31] Seal T8 epoch checkpoints only after callback unwind

### Motivation

The pre-callback ctime barrier did not close the AutoFS writer lifecycle.  A
second fresh T8 run produced the same valid epoch-zero checkpoint and failed at
the same post-callback comparison: the file content, size, inode, and mtime were
stable, while final filesystem metadata became visible only after the callback
exception unwound through the official training stack.  Reusing either failed
attempt or weakening the checkpoint hash would make resume evidence ambiguous.

### Decision

Treat the callback event as provisional writer evidence.  After the official
generator has fully unwound, move its mutable checkpoint directory through a
unique same-parent seal name, require a bounded quiet period in which every
field except monotone ctime remains exact, fsync the retained tree, and publish
the persistent `sealed-planned-checkpoint` directory atomically.  Reopen that
directory with `O_NOFOLLOW`, retain its full inode/SHA inventory, and make a
fresh single-link resume copy whose content inventory must equal the seal.
Atomically publish that copy at the official checkpoint pathname, verify it
through a separate reopen, and only then allow the official resume loader to
consume it.  Keep the sealed epoch-zero tree unchanged through terminal branch
capture.

### Consequences

- Callback-time metadata is never treated as final filesystem authority.
- The checkpoint and heartbeat SHA-256 values must remain byte-identical across
  mutable output, persistent seal, resume copy, and official loader adoption.
- A post-callback byte, mode, owner, link-count, size, mtime, device, or inode
  change fails closed; only the observed monotone ctime settlement is recorded.
- T8 scientific configuration, target branches, frozen GINE, split isolation,
  and official GlobalGCE semantics are unchanged.

### Status

Accepted

---

## [2026-08-31] Continue Mutagenicity ComRecGC from completed exact science read-only

### Motivation

The adopted Mutagenicity full root has already completed candidate labeling,
the exact multi-component DBSCAN partition, centroid/radius calculation,
coverage, and official greedy selection.  Re-running any of those stages would
create a second scientific writer and could diverge from the hash-closed result.
The remaining paper-cell work is deterministic chemistry, standardized WNode
evaluation, freeze, and matrix publication.  The generation trace is present,
but the current trace-only summary does not prove Mut trace-on/off parity.

### Decision

Add one Mutagenicity-specific read-only continuation.  It must independently
reopen the exact adoption receipt, source controller terminal, common-recourse
terminal, DBSCAN artifacts, generation manifest, and all frozen hashes before
running downstream work.  Chemistry must consume exactly the 100 selected
common recourses.  Pair-store, DBSCAN, and common-recourse commands are absent
from the runner and recorded as not rerun.

Require an explicit `mut_trace_on_off_parity_v1` PASS receipt before creating a
fresh output root.  Do not apply the AIDS/BACE trace-integrity waiver to Mut.
After freeze, publish only a strict one-cell `Mutagenicity/ComRecGC` successor
whose other 15 matrix rows are unchanged and whose shared identity agrees with
the frozen Mut/Ours cell.

### Consequences

- Completed exact science remains immutable and cannot acquire a second writer.
- Missing parity is reported at its actual scientific gate rather than hidden
  behind a chemistry or matrix failure.
- Interrupted chemistry/WNode/freeze stages can resume under an exact frozen
  input/argv contract without rerunning common recourse.
- Matrix publication cannot silently alter another method cell or impute a
  missing result.

### Status

Accepted
