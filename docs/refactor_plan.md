# Refactor Plan

## 2026-08-28: TasteMolNet T5 clean-policy initializer

- [x] Add a Taste-owned typed initializer schema instead of relabeling BACE
  provenance, manifests, or markers.
- [x] Implement only the generic ChemLLM fresh zero-step LoRA path and record
  `initializer_data_split_used=none` separately from the train-only access
  ceiling; leave train-only SFT explicitly unimplemented without authority.
- [x] Require a raw-SHA-pinned release authority over policy v2, its receipt,
  immutable Git identity, full source-model physical authority, and the typed,
  descriptor-held common GINE/T3/T4 interface.
- [x] Publish a fresh private root with the T0--T16 five-file evidence closure,
  adapter provenance, output inventory, and PASS-last marker.
- [x] Parse and audit real safetensors tensors, prove LoRA B is still zero,
  perform a fresh-base `PeftModel.from_pretrained` reload, and distinguish the
  bare-base identity from the base-plus-adapter reference-policy identity.
- [x] Add a combined descriptor-held T5/source loading token for T6 with
  loaded-policy verification and physical stat/ctime swap-restore detection.
- [x] Bind staging/adapter/parent inodes through no-replace publication, repeat
  external authority validation after publication, and write PASS last.
- [x] Add an AutoDL foreground wrapper, static-refusal paired Slurm wrapper,
  hostile publication/tensor/cross-binding tests, and a runbook.
- [ ] Fill and independently review the final T3/T4/source pins and define a
  physical controller execution receipt before enabling any public T5 model
  load or output creation; the tracked config and public builder remain
  disabled.

## 2026-08-25 release blocker: restart-safe GREED full scan

- [x] Separate the long-lived pair-semantics science root from fresh controller
  receipt roots.
- [x] Add a PID/lock/checkpoint/hash-authenticated supervisor with one bounded
  same-root transient resume and semantic-failure exclusion.
- [x] Make controller signal-loss classification opt-in and semantic-first.
- [x] Bind the close-view dependency to the receipt while reading only the
  fixed-root science artifacts after deep receipt validation.
- [x] Add the paired non-runnable Slurm parity wrapper and focused restart,
  writer, signal, and receipt-tamper tests.

## 2026-08-24: Fresh AIDS snapshot-adoption continuation

- [x] Replace decoded-command substring matching with raw NUL argv parsing,
  CPython flag/first-script semantics, proc-cwd relative resolution, and exact
  physical entrypoint identity.
- [x] Ignore bash/grep/regex literals and `python -c/-m` while rejecting real
  absolute, relative, flagged, direct, missing, or symlinked rogue workers.
- [x] Bind the completed `pair_order_v1` snapshot to its exact owner manifest,
  exact persistent namespace root, namespace-derived task-gate path/SHA,
  unique `main` PASS attempt/output, all terminal manifest hashes, and both
  large-array hashes; copied manifest/gate trees are not authority.
- [x] Add a PASS-last adoption task that fully reopens the immutable snapshot,
  publishes no arrays, and performs no copy, hardlink, or source write.
- [x] Make fresh science consume the original immutable snapshot path while
  depending on and revalidating the fresh adoption gate before child spawn.
- [x] Preserve CPU-only/cgroup/RSS/high-memory-lock/process-watchdog contracts
  and create a new CID, route lock, output root, and Mut dependency identity.
- [x] Add CLI/tamper/authority-path/process/parser/shell-preflight tests and a
  static paired Slurm wrapper.
- [x] Pin implementation commit
  `98c5125b8b68df8a8797c0228e85d9c8f45e1aed` in a builder-only follow-up.
- [ ] Obtain an independent no-blocker review and deploy only an immutable
  fresh worktree.

## 2026-08-24: Release-gate the fresh AIDS exact-Cartesian v5 route

- [x] Forward every adaptive DBSCAN, promoted-source, chunk-cache, headroom,
  procfs, and resume control through the production AutoDL continuation wrapper.
- [x] Prefer a physically promoted final pair store; reject an invalid final
  instead of silently falling back to chunks.
- [x] Freeze production v5 to terminal-only adoption with the exact pair-store
  owner root, allowing the old DBSCAN's read-only mmap but retaining full-tree
  partial/writable-inode rejection; terminal disappearance cannot select chunks.
- [x] Freeze the sole old read-only DBSCAN generation by PID/start ticks/raw
  command hash/exact output/cwd and re-scan procfs before every attempt; allow
  natural exit but reject PID reuse, identity drift, or a second common task.
- [x] Reject terminal partial/symlink artifacts, writable sibling inodes,
  unbound common-recourse processes, scientific/hash/stat drift, and source
  path escape.
- [x] Authenticate `posix_fallocate` completion and NPY header/size/headroom,
  replay allocation safely across the pre-checkpoint crash window, and bind the
  evidence through copy, promotion, and terminal validation.
- [x] Rebuild a malformed pre-allocation NPY only at authenticated
  `allocate_cache` while both allocation and outer route flocks are held;
  reject the same damage after `allocation_complete` without deleting it.
- [x] Require a fresh root if a terminal chunk-cache manifest loses its local
  reconstructible artifact; never rewrite a published closure.
- [x] Freeze the v5 supervisor to CPU-only adaptive exact DBSCAN with
  `eps=0.02`, `min_samples=3`, self-neighbour semantics, dense fallback zero,
  route-wide scratch locking, a 128 GiB per-attempt cgroup-free gate, a 96 GiB
  child RSS ceiling, and one process-loss-only same-root retry.
- [x] Queue and monitor a same-inode global high-memory flock handover before
  science starts; retain it after old-v4 natural exit, and fail/terminate only
  the new process group if the helper generation disappears or drifts.
- [x] Re-scan common-recourse processes during science and permit only the
  frozen old generation plus one exact descendant of the bound v5 science
  generation; reject mid-run rogue tasks and root PID reuse.
- [x] Keep the paired Slurm wrapper static/non-runnable and document the exact
  fresh AutoDL task/environment recipe.
- [x] Add a fresh-only v5 manifest builder with physical terminal-array audit,
  immutable repair-v4 input binding, selector-adoption dependency, exact Mut
  dependency identity, and persistent controller launch/restart commands.
- [x] Cover promoted-final priority, real shell forwarding, invalid-final,
  partial/writer/owner, allocation crash/tamper, Cartesian order, sklearn label,
  official greedy/medoid/selected-row, and terminal resume closures.
- [ ] Obtain independent review of the final route commit, then run a fresh
  source audit and production-shaped subset smoke before building/deploying a
  fresh v5 controller.  Never signal or write repair-v4 during this step.

## 2026-08-23: Accept pinned-official GlobalGCE affine edge scores

- [x] Trace the negative edge values to the pinned official decoder, where the
  apparent sigmoid is passed as `nn.Linear`'s positional bias flag and is not a
  module in the decoder sequence.
- [x] Treat decoder edge outputs as finite categorical scores and apply a
  class-axis softmax only at the frozen-GINE expected-embedding boundary.
- [x] Preserve the official hard `argmax` edge codec and reject non-finite or
  malformed score tensors rather than clamping them.
- [x] Extend the production smoke and focused tests with negative scores,
  hard-oracle parity, nonzero transformation gradients, zero classifier
  gradients, and unchanged checkpoint/parameter hashes.
- [x] Keep the AutoDL CLI and paired Slurm wrapper in sync; any failed v5 root
  remains immutable and a corrected run must use a fresh execution root.

## 2026-08-23: Persistent three-dataset release supervisor

- [x] Catalog the nine settled standardized roots and keep AIDS ComRecGC,
  Mutagenicity ComRecGC, and BACE GlobalGCE as explicit fail-closed
  placeholders until their exact routes are frozen.
- [x] Require every non-v4 cell to bind an immutable external owner manifest,
  task ID, and exact output path/template; bind PASS and closure hashes at
  publication time.
- [x] Keep the GlobalGCE native v5 final root outside the matrix and require
  the existing CPU `bace_globalgce_standardized` task to create a fresh paper
  closure.
- [x] Add a CPU-only, flock/PID-bound persistent monitor with a 60-second
  heartbeat, restart reconciliation, no GPU lock, and no writes to active
  controllers or scientific roots.
- [x] Publish neither registry nor numeric output below 12/16; at 12/16,
  atomically publish the canonical sixteen-row matrix and use the existing
  three-dataset staging exporter under fresh/no-clobber destinations.
- [x] Add exporter process-loss recovery, paired static Slurm parity wrappers,
  focused drift/restart/heartbeat/no-output tests, and an AutoDL runbook.
- [ ] Build the runnable spec only after all three placeholder routes have
  immutable owner manifests; review its SHA before deploying the sidecar.

## 2026-08-23: BACE equivalence UUID-lock sidecar

- [x] Prove active controller manifests cannot accept new task bytes after
  their manifest SHA and instance topology are frozen.
- [x] Audit the four protected live exp-runs and exclusive UUID locks without
  signalling or writing their output roots.
- [x] Add a fresh GCF-only queue for quick M=50, quick M=100, and formal M=500
  with strict dependencies and attempt-qualified output roots.
- [x] Keep the existing ComRecGC M=500 legacy-to-optimized pair outside the
  sidecar so it is observed but never duplicated.
- [ ] Build, validate, and persist the sidecar manifest from an immutable
  AutoDL execution worktree, then launch it while all GPUs remain protected.
- [ ] Publish GCF PASS markers only from the real replay gates; M=500 alone
  remains insufficient to release optimized full.

## 2026-08-23: BACE GlobalGCE frozen train-view repair

- [x] Distinguish the 959-row processed train input from the frozen 869-ID
  teacher-consistent native train vocabulary.
- [x] Recompute the 360 source / 509 target / 162 validation cohort contract
  and bind ordered ID hashes plus both dataset manifests and artifact hashes.
- [x] Add an opt-in exact parent-ID filter to native GlobalGCE dataset loading;
  preserve the existing default for other datasets.
- [x] Reject calibration/test rows, validation loading, missing IDs, and
  label/canonical-SMILES drift before training.
- [x] Add focused mapping and leakage tests and keep AutoDL/Slurm CLI parity.
- [ ] Build a fresh immutable execution worktree and v4 controller; never
  resume or rewrite the failed v3 train root.

## 2026-08-23: Acceleration release gates

- [x] Require one immutable exact-500 + exact-1000 aggregate gate before an
  optimized BACE ComRecGC 50,000-step run.
- [x] Recheck raw payload/completion/trace/audit hashes, frozen oracle,
  distance, cohort, batch size, and preprocessing config before output-root
  creation; keep legacy sequential full runs unchanged.
- [x] Require two 10--15 minute single-task profiles plus one same-GPU paired
  profile before any `shared_lowmem` task can be declared.
- [x] Compute aggregate throughput from the per-task values, bind benchmark
  keys to scientific-config and canonical-result SHA, and require >=20% gain,
  no result drift/OOM/error/CPU saturation/disk instability, and no MPS.
- [x] Revalidate the co-location gate at controller schema, controller launch,
  exp-run launch, and worker lock acquisition; bind both slots to the same
  gate and exact authorized workload pair.
- [x] Add focused negative tests, CLI docs, and paired Slurm wrappers without
  deploying or interrupting existing AutoDL diagnostic/legacy jobs.
- [ ] Publish either PASS gate only after the real AutoDL evidence completes;
  until then optimized full/shared routes remain blocked.
## 2026-08-23: Close shortcut checkpoint-prefix false positives

- [x] Authenticate each atomic checkpoint payload and bind `next_offset` to a
  contiguous, scientific-identity-bound forward hash-chain ledger.
- [x] Record block-local float64 top-k evidence for adaptive seed selection,
  the complete per-block adaptive failure set, and canonical uint32 lower
  values/minima for the final anchor scan.
- [x] Replay every committed prefix from source/model bytes before resume and
  reject offset, aggregate, ledger, partial-array, or state tampering.
- [x] Reassert the adaptive failure cap after replay and before selection so a
  coordinated, reauthenticated complete-ledger mutation still fails closed.
- [x] Bind a stable source-file stat snapshot around the entry SHA, repeat the
  complete vector SHA before PASS, and reject any mmap/final-publication stat
  drift in shortcut, fallback, and terminal-reopen paths.
- [x] Persist first-pass completion before adaptive-selection publication and
  cover the selection-rename/checkpoint crash window with a resume fixture.
- [x] Before PASS, validate contiguous exact lower coverage for every row,
  global/non-anchor minima, core/attachment thresholds, and constant
  labels/core outputs; repeat this closure on terminal reopen.
- [x] Add the original 12-by-64 reviewer reproducer, adaptive seed/failure
  coordinated offset tampering, committed-lower tampering, pre-PASS full-array
  corruption, and publish-crash recovery tests.
- [ ] Wire or run only from a later fresh AutoDL execution commit; all
  external-schema-v2 checkpoints/terminals and the real failed fixed-64
  witness remain immutable.

## 2026-08-23: Fresh exact-Cartesian adoption of AIDS pair chunks

- [x] Reconstruct and compare the complete pair scientific identity before
  adoption; bind the exact source manifest and array hashes plus physical stat
  identity.
- [x] Reject every live writable FD/mapping through Linux procfs and recheck
  source stats around full checksum validation.
- [x] Prove the complete 71,642-by-1,283 Cartesian row formula elementwise;
  represent pair indices implicitly without a second 1.47 GiB array.
- [x] Reconstruct exactly one contiguous vector `.npy` on local XFS from raw
  chunk bytes; derive its target header/hash without chunkwise reductions.
- [x] Enforce `posix_fallocate`, target-size plus 3 GiB admission, post-reserve
  floor, reconstructible-cache semantics, and exclusive scratch locking.
- [x] Close the fresh adoption manifest through normal and resumed terminal
  validation and add identity/writer/stat-drift and end-to-end fixtures.
- [x] Keep the paired Slurm CLI in sync without submitting an HPC job.
- [x] Integrate independently reviewed adaptive core
  `645c6e51b7abcdc5dd4a9e0a1226d71d020880da`.
- [ ] Obtain independent release review of the chunk/cache wiring and build a
  fresh v5 controller manifest; old v4 remains read-only until every gate passes.

## 2026-08-23: Exact disk-backed one-cluster summary replay

- [x] Admit the specialized route only from a hash-closed exact all-core,
  one-component DBSCAN proof; retain the general multi-cluster implementation.
- [x] Preserve separate frozen Torch and NumPy centroid/reduction semantics and
  strict per-row radius comparisons with block-size-independent fixtures.
- [x] Reproduce upstream first-parent counterfactual coverage and the official
  one-cluster greedy result without a 91.9-million-element Python loop.
- [x] Persist retained mask, source positions, and vectors in source order;
  preserve NumPy retained-centroid and first-argmin medoid behavior exactly.
- [x] Add resumable two-phase promotion, input/result hashes, centroid tamper
  rejection, RSS admission, empty-coverage parity, and terminal continuation
  closure.
- [x] Wire the adaptive shortcut and summary bounds through the common-recourse
  CLI and paired Slurm parity wrapper while leaving the default route disabled.
- [x] Implement both terminal pair-store adoption and the specialized
  exact-Cartesian chunk/cache route through fresh manifests.
- [ ] Run a fresh real-data adaptive proof and summary sidecar; never continue
  into the old v4 attempt or publish PASS from exploratory evidence.

## 2026-08-23: Deterministic adaptive AIDS DBSCAN witness

- [x] Retain the fixed evenly-spaced 64-anchor route as a fail-closed negative;
  the full 91,916,686-row diagnostic found 43 unproved rows.
- [x] Select three seeds by a complete global minimum squared-L2 scan with
  deterministic sample-index tie breaking and resumable bounded blocks.
- [x] Run a complete first seed-radius pass, freeze every insufficient row
  index and its hash, and block rather than truncate when the failure cap is
  exceeded.
- [x] Define the final anchors as the sorted unique union of the seeds and all
  first-pass failures; freeze exact anchor row/index hashes.
- [x] Run the ordinary complete anchor lower-bound proof as a second pass and
  retain its exact self, duplicate-index, inclusive-epsilon, and connectivity
  gates.
- [x] Add adaptive sklearn parity, resume, failure-cap, proof tamper, RSS, and
  fixed-64 negative fixtures.
- [ ] Release only through a fresh promoted-vector AutoDL proof and separate
  downstream closure; never reuse the exploratory witness or old v4 root.

## 2026-08-23: Exact all-core DBSCAN anchor shortcut

- [x] Add a deterministic, hash-bound finite-anchor witness before the
  quadratic dense-radius route.
- [x] Prove all rows core from distinct anchor sample-index lower bounds,
  including sklearn self counting, duplicate-vector identity, and inclusive
  epsilon semantics.
- [x] Prove one component from a connected anchor epsilon graph plus an anchor
  attachment for every non-anchor row; emit exact labels zero in source order.
- [x] Publish lower-bound/anchor proof artifacts and explicitly mark exact
  full neighbor counts unavailable instead of fabricating them.
- [x] Resume the linear witness by hash-bound blocks and reject tampering,
  scientific-input drift, sklearn drift, or RSS overflow.
- [x] Fall back only below an explicit exact-complexity limit; otherwise stop
  with `EXACT_DBSCAN_COMPLEXITY_BLOCKED` and no approximate labels.
- [x] Cover positive sklearn label/core parity, epsilon-boundary and duplicate
  semantics, negative fallback/block cases, resume/hash closure, and RSS.
- [ ] Wire the opt-in contract through a fresh AutoDL continuation only after
  independent real promoted-vector proof; do not mutate the running v4 root.

## 2026-08-23: AIDS ComRecGC exact external-memory repair-v4

- [x] Reconstruct sklearn DBSCAN labels from bounded radius-query passes and
  prove elementwise equality, including ambiguous border assignment.
- [x] Add atomic candidate-major pair/vector chunks, consolidation, checksums,
  interruption resume, and a hard RSS admission gate.
- [x] Bind resume identity to the complete dataset/source audit and reconcile
  every partial-to-final rename through a checksum-validated two-phase state.
- [x] Preserve legacy cluster order, strict-radius filtering, official greedy
  calls, and retained-pair medoid selection on an exact fixture.
- [x] Wire the engine into the full AIDS `run_common_recourse.py` route without
  changing the default legacy engine.
- [x] Prove a full runner fixture has identical pair order, sklearn labels,
  official summary, selected rows, and stable selected-row hash across legacy
  and external engines.
- [x] Add hash-bound whole-stage resume, including child-complete/parent-killed
  reconciliation and same-path scientific-input drift rejection.
- [x] Require the same nested artifact hash closure on the fresh-success path
  before the common-recourse stage can publish a PASS checkpoint.
- [x] Add an AIDS-only three-task repair-v4 builder plus a bounded supervisor
  that permits exactly one evidence-gated same-root retry after process loss;
  semantic, contract, sklearn, and RSS failures remain terminal.
- [x] Prove the supervisor's real shell behavior and controller restart
  reconciliation with production-shaped tests: one retry can finish PASS-last,
  a second process loss terminates, and restart adopts the live run ID/attempt.
- [x] Run end-to-end legacy/external smoke equivalence on real AutoDL AIDS data
  and freeze the exact pair/vector/label/selected-row digests in a
  diagnostic-only PASS-last gate.
- [x] Make the repair-v4 builder adopt that physical smoke gate by SHA, exact
  integrated commit, frozen source payload, nine true equivalence checks, and
  no-live-writer evidence; a missing or changed gate blocks manifest creation.
- [x] Keep the scoped safe-Git ancestry gate and external-memory recovery-core
  ancestry gate separate; each accepts only its own reviewed exact commit.
- [ ] Build and launch a fresh CPU-only repair-v4 controller; never resume or
  mutate the failed repair-v2/repair-v3 roots.

## 2026-08-23: User-approved frozen v4 AIDS/Mutagenicity adoption

- [x] Pin the exact five machine-readable v4 source files and six authorized
  AIDS/Mutagenicity cells in a tracked policy; do not open image/PDF files.
- [x] Read and SHA each adopted source file once, reuse the cached bytes for
  validation/copying/manifests, and reject any stat drift or writable AutoDL
  file descriptor.
- [x] Project exact numeric strings into fresh standardized roots without
  reranking, recomputing metrics, selecting candidates, or rendering figures.
- [x] Keep absent raw, dataset, split, oracle-checkpoint, and MolCLR identities
  explicitly unavailable under the narrow `USER_APPROVED_FROZEN_V4` exception.
- [x] Extend registry and final-export verification to accept only that exact
  checksum/scope-bound exception while retaining every normal fail-closed gate.
- [x] Exclude CLEAR and ComRecGC from the exception; a new controller may omit
  not-started duplicate repair work but this adoption never stops a live task.
- [x] Add AutoDL execution and protected static Slurm CLI-parity wrappers,
  focused tamper/source-hash/fresh-root/registry/export tests, and a deployment
  runbook.
## 2026-08-23: Repair BACE GCF ordered-v2 batch-shape equivalence

- [x] Locate the first m500 divergence and verify that RNG state, action order,
  graph tensor order, and parallel RDKit decode remain exact.
- [x] Identify canonical-SMILES deduplication and 256-row GINE chunking as the
  first semantic change: legacy scores all duplicate-preserving valid rows in
  one batch and hashes raw embedding bytes.
- [x] Preserve the complete legacy GINE batch and allow only exact whole-batch
  importance cache hits.
- [x] Add diagnostic-only 50/100 profiles and a fresh quick replay wrapper that
  cannot produce the formal acceleration gate.
- [x] Keep the paired Slurm full route explicit in legacy mode.
- [ ] Run fresh 50/100 quick replay on AutoDL, followed by the mandatory fresh
  500/1000 replay only if both quick gates pass.

## 2026-08-23: BACE GCFExplainer replay-gated acceleration

- [x] Instrument fresh VRRW roots with phase timings, throughput, RSS, GPU peak
  memory, cache counters, and canonical equivalence digests.
- [x] Preserve edit and transition order while adding ordered CPU neighbour
  construction and buffered progress.  Subsequent m500 evidence removed
  canonical-row GINE caching/chunking because it changed raw embedding bytes.
- [x] Require 500/1000 exact canonical replay and a same-GPU >=20% A/B gate
  before an optimized 50,000-step launch.
- [x] Add two opt-in low-memory slots per UUID, legacy-exclusive compatibility,
  process ownership checks, positive reservations, and a strict 70% VRAM cap.
- [x] Keep MPS disabled, legacy mode default, existing outputs immutable, and
  all deployment commands AutoDL-only.
- [ ] Run the fresh 500/1000 same-card smoke on AutoDL after the current legacy
  full job releases a suitable GPU/lock; deploy 50k only if the gate passes.

## 2026-08-23: AIDS ComRecGC CPU/high-memory repair v3

- [x] Classify repair-v2 AIDS `SIGKILL` from cgroup limit/peak/fail/OOM evidence
  as a host-memory scheduling failure rather than a scientific failure.
- [x] Add an exact three-task AIDS-only fresh controller with
  `resource=cpu`, `gpu_required=false`, and `max_cpu_tasks=1`.
- [x] Hold one persistent common-recourse high-memory lock, require cgroup-v1
  headroom, and reject a legacy uncoordinated common-recourse process.
- [x] Preserve repair-v2 source gates and the failed attempt as immutable
  evidence; do not reuse or overwrite its partial output.
- [x] Keep Mutagenicity blocked on real trace-on/trace-off parity and document
  the exact missing independent reference input instead of weakening the gate.
- [ ] Build and launch the fresh controller from an immutable AutoDL execution
  worktree after integration approval.

## 2026-08-22: Mutagenicity GCF off-grid theta-star export compatibility

- [x] Reproduce the repair-v1 failure against the real 601-point frozen grid,
  which brackets but does not contain `theta_star=0.05`.
- [x] Keep all 601 official grid rows byte-semantically unchanged and source
  the exact theta-star metric only from the already-recomputed full-prefix row.
- [x] Reject a missing explicit theta row, nearest-grid substitution, wrong
  threshold provenance, incomplete fields, or parent/candidate identity drift.
- [x] Cover both direct reconstruction and the complete exporter/final-audit
  round trip with a production-shaped 601-point fixture.
- [x] Keep repair-v1 FAILED roots immutable and document a fresh held-out plus
  standardization continuation recipe.

## 2026-08-22: AIDS/Mutagenicity-only ComRecGC repair v2

- [x] Add a six-task fresh controller fragment containing only four exact
  repair-v1 PASS source gates and two A/M standardized continuations.
- [x] Reuse the repair-v1 scientific environment instead of accepting another
  manually configured dataset/RF/MolCLR/distance/upstream path set.
- [x] Gate build and runtime execution on ancestry of the reviewed
  `verify_comrecgc_checkout` safe-Git fix.
- [x] Preserve threshold-before-test dependencies, shared GPU UUID locks,
  `max_cpu_tasks=2`, fresh outputs, and the absence of old continuation guards.
- [x] Exclude BACE, GCFExplainer, TasteMolNet, final exports, HPC execution, and
  `paper/` changes from this bounded retry.

## 2026-08-22: Reuse scoped Git trust in the standalone COMRECGC gate

- [x] Expose the shared process-private COMRECGC commit reader used by checkout
  validation.
- [x] Remove the standalone verifier's duplicate raw `git rev-parse` call.
- [x] Add a migrated-owner regression proving the verifier supplies an exact
  temporary `safe.directory` without changing global Git configuration.

## 2026-08-22: Repository-owned AutoDL four-by-four dashboard

- [x] Replace the standalone fixed `autodl_three_lines` data source with
  physical-directory discovery under the persistent four-by-four namespace.
- [x] Reuse the controller status/UUID-lock model and query GPU inventory once
  per page snapshot across every discovered controller.
- [x] Add a predominantly Chinese, GET-only loopback UI with controller/task/
  output/PID/reason fields and explicit heartbeat freshness.
- [x] Refresh every five seconds and immediately after tab visibility or
  network restoration without caching API responses.
- [x] Add a nohup AutoDL launcher, health endpoint, terminal/JSON once mode,
  SSH-tunnel documentation, and focused security/discovery tests.
- [x] Keep the dashboard AutoDL-only and deliberately omit a long-lived HPC
  Web-service wrapper.

## 2026-08-22: BACE Ours historical B13 standardization compatibility

- [x] Audit the immutable B12 selection, B13 top-level, merge, and four shard
  manifests from the failed AutoDL standardization attempt.
- [x] Recognize `selector_frozen_before_split_load=true` only for the exact
  original Ours verification-shard schema and require the full B12 candidate,
  predecessor, policy, GINE, and MolCLR identity closure.
- [x] Add a recorded historical structure fixture plus a negative regression
  proving that absent freeze evidence still fails closed.

## 2026-08-22: Bounded four-by-four repair continuation

- [x] Add a reproducible repair-only manifest builder with build-time and
  runtime source-terminal PASS/output/writer verification.
- [x] Reuse the generic native BACE ComRecGC GINE fragment and append only its
  artifact standardization terminal.
- [x] Retry Mutagenicity GCF calibration/test/standardization from the exact
  passing v1 freeze without rerunning candidate generation.
- [x] Retry AIDS/Mutagenicity COMRECGC threshold verification and standardized
  continuation from immutable recovered generations.
- [x] Retry only BACE Ours artifact standardization from the passing B14 root.
- [x] Share global GPU UUID locks with v1, cap repair CPU task concurrency at
  two, omit Taste/final rendering, and inherit no old continuation guard.
- [x] Match the historical COMRECGC recovery terminal contract: five immutable
  manifests plus physical payload stat and claimed-SHA agreement, with no
  synthetic bare `PASS` or mutable-registry dependency.

## 2026-08-22: BACE four-by-four cell closure

- [x] Keep B14 and native baseline final roots as immutable scientific
  terminals rather than treating them as paper-ready cells.
- [x] Add deterministic, CPU-only BACE Ours/GCFExplainer/ComRecGC
  standardization tasks with SHA256 identity traversal and no raw test access.
- [x] Export the complete common cell schema, file inventory, freeze marker,
  final audit, and PASS-last marker under fresh roots.
- [x] Reject final-matrix mappings that bypass the standardization layer.
- [x] Preserve GlobalGCE's reviewed code blocker without generating a
  substitute result.

## 2026-08-22: Continuation predecessor namespace binding

- [x] Resolve the BACE predecessor controller from the exact persistent source
  manifest namespace while keeping the new four-by-four controller in its own
  namespace.
- [x] Reject source manifests outside the control root, malformed namespace
  layouts, symlinked roots, and controller snapshot identity mismatches.

## 1. Purpose

This document records the intended roadmap for rebuilding the counterfactual subgraph v3 project from an empty repository.

The goal is not merely to write working code, but to build a clean research codebase that remains faithful to the counterfactual objective and is easy to evolve.

---

## 2. Rebuild Strategy

The project should be rebuilt incrementally.

The guiding principle is:

> First stabilize interfaces and responsibilities, then implement training logic.

This is important because earlier versions were likely affected by script-level coupling, implicit assumptions, and reward/training entanglement.

---

## 3. Phase Overview

## Phase 0: Documentation-first bootstrap

Objective:

- establish the research objective in writing;
- define repository conventions;
- ensure Codex and future contributors follow the same target.

Deliverables:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

Status:

- completed on 2026-04-09.

---

## Phase 1: Repository skeleton

Objective:

- create the core directory structure;
- define code boundaries;
- prepare CLI and config folders.

Deliverables:

```text
configs/
data/
scripts/
src/
tests/
outputs/
```

Recommended first-level modules:

```text
src/data/
src/models/
src/rewards/
src/train/
src/eval/
src/chem/
src/utils/
```

Success criteria:

- all major concerns have a dedicated location;
- no business logic lives in random top-level files.

Status:

- bootstrap skeleton implemented on 2026-04-09.
- training logic intentionally deferred.

### Suggested target directory structure

The repository should now grow toward the following structure:

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── README.md
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── reward/
│   └── eval/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
│   ├── cf_subgraph_v3_spec.md
│   ├── decisions.md
│   └── refactor_plan.md
├── outputs/
├── scripts/
│   ├── README.md
│   ├── prepare_data.py
│   ├── infer_single.py
│   ├── train_sft.py
│   ├── train_rl.py
│   └── eval_model.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   ├── dataset.py
│   │   └── collators.py
│   ├── chem/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── smiles_utils.py
│   │   ├── substructure.py
│   │   ├── deletion.py
│   │   └── validation.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── aggregation.py
│   │   ├── anti_collapse.py
│   │   └── counterfactual_reward.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── interfaces.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   └── diagnostics.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   ├── metrics.py
│   │   └── reporting.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging.py
│       └── seed.py
└── tests/
    ├── README.md
    ├── test_prompt_contract.py
    └── test_reward_breakdown.py
```

### Module responsibilities and minimum interfaces

#### `src/data/`

Responsibility:

- define the canonical JSONL schema;
- centralize prompt construction;
- expose dataset and batch contracts that can be reused by SFT, RL, and evaluation.

Minimum interface:

- `MoleculeRecord` and `FragmentExample` dataclasses;
- `normalize_molecule_record(raw)` for schema validation;
- `build_counterfactual_prompt(record, include_label=False)` for prompt generation;
- `JsonlMoleculeDataset.from_jsonl(path)` for deterministic loading;
- `CounterfactualPromptCollator` returning a `PromptBatch`.

#### `src/chem/`

Responsibility:

- own all chemistry-specific behavior;
- keep parsing, validation, substructure checks, and deletion out of train scripts;
- serve as the only place where future RDKit logic should live.

Minimum interface:

- `ParsedMolecule`, `FragmentValidationResult`, and `DeletionResult` dataclasses;
- `parse_smiles(smiles)` and `canonicalize_smiles(smiles)` placeholders;
- `is_parent_substructure(parent_smiles, fragment_smiles)` placeholder;
- `is_connected_fragment(fragment_smiles)` placeholder;
- `delete_fragment_from_parent(parent_smiles, fragment_smiles)` placeholder;
- `validate_fragment_candidate(parent_smiles, fragment_smiles)` placeholder.

#### `src/rewards/`

Responsibility:

- represent reward terms explicitly;
- keep counterfactual scoring distinct from structural checks;
- surface anti-collapse diagnostics without folding them into training code.

Minimum interface:

- `RewardWeights`, `RewardTerm`, and `RewardBreakdown` dataclasses;
- `RewardContext` for one candidate reward computation;
- `build_reward_breakdown(context, weights)` for structured reward assembly;
- `aggregate_reward_terms(terms)` for scalar aggregation;
- `analyze_batch_collapse(outputs)` and `collapse_penalty_from_diagnostics(...)`.

#### `src/models/`

Responsibility:

- define the generation contract between prompts and fragment outputs;
- stay backend-agnostic so the same interface can wrap local checkpoints or HF models later.

Minimum interface:

- `GenerationRequest` dataclass;
- `GenerationResult` dataclass;
- `FragmentGenerator` protocol.

#### `src/train/`

Responsibility:

- define stage-level training contracts without implementing optimization yet;
- keep diagnostics first-class so RL collapse signals are visible from day one.

Minimum interface:

- `TrainStage` enum for format SFT, weak-supervision SFT, and counterfactual RL;
- `TrainingRunRequest` and `TrainingStatus` dataclasses;
- `Trainer` protocol;
- `TrainingDiagnosticsSnapshot` dataclass.

#### `src/eval/`

Responsibility:

- define checkpoint evaluation outputs independently of training code;
- standardize metric computation and reporting for structural and counterfactual views.

Minimum interface:

- `EvaluationExample` and `EvaluationSummary` dataclasses;
- `Evaluator` protocol;
- `safe_rate(...)` and `mean_metric(...)` helpers;
- `render_summary(summary)` formatter.

#### `src/utils/`

Responsibility:

- hold reusable generic helpers that are not themselves chemistry or reward logic;
- support reproducibility, IO, and logging across local and HPC runs.

Minimum interface:

- `read_jsonl(path)` and `write_jsonl(path, rows)`;
- `ensure_directory(path)`;
- `RunContext` dataclass and `get_logger(name)`;
- `set_global_seed(seed)`.

---

## Phase 2: Chemistry utility layer

Objective:

- build reliable molecule and fragment utilities before model training.

Modules to implement first:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

Target capabilities:

- parse SMILES safely;
- sanitize molecules;
- canonicalize fragment strings where appropriate;
- test whether fragment is a parent substructure;
- test whether fragment is connected;
- perform fragment deletion or approximate deletion logic;
- report failure types clearly.

Success criteria:

- chemistry checks are deterministic and testable;
- training code does not need to reimplement chemistry logic inline.

---

## Phase 2.5: Local/HPC runtime adaptation layer

Objective:

- make the modular repository runnable in both local development and HPC settings;
- keep all path handling config-driven and repository-relative;
- support single-machine or single-node single-GPU execution only for now.

Modules and files:

- `configs/base.yaml`
- `configs/local.yaml`
- `configs/hpc.yaml`
- `configs/sft.yaml`
- `configs/rl.yaml`
- `configs/eval.yaml`
- `src/utils/paths.py`
- `src/utils/env.py`
- `src/utils/logging_utils.py`
- `src/utils/seed.py`
- `scripts/run_sft.py`
- `scripts/run_rl.py`
- `scripts/run_eval.py`
- `scripts/run_infer.py`
- `scripts/slurm/*.slurm`

Target capabilities:

- merge stage and environment configs deterministically;
- resolve all runtime paths without hardcoded absolute paths;
- support local model and tokenizer paths;
- create per-run log and manifest directories;
- provide Slurm templates for single-node single-GPU jobs;
- keep CLI entrypoints thin and compatible with later training logic.

Success criteria:

- a local or HPC user can prepare a run from config and CLI only;
- scripts save a resolved manifest for reproducibility;
- the runtime layer does not assume distributed training.

---

## Phase 3: Reward subsystem

Objective:

- implement reward logic as a standalone subsystem.

Suggested files:

- `src/rewards/types.py`
- `src/rewards/counterfactual_reward.py`
- `src/rewards/anti_collapse.py`
- `src/rewards/aggregation.py`

Target capabilities:

- compute individual reward terms;
- return structured reward breakdowns;
- support configurable weights;
- expose penalties for collapse patterns;
- isolate counterfactual scoring from train-loop code.

Success criteria:

- reward logic is testable outside RL training;
- each term has a clear name, meaning, and expected range.

---

## Phase 4: Data and prompt subsystem

Objective:

- build clean dataset loaders and prompt builders.

Suggested files:

- `src/data/schemas.py`
- `src/data/jsonl_dataset.py`
- `src/data/prompts.py`
- `src/data/collators.py`

Target capabilities:

- read raw dataset JSONL;
- validate required fields;
- construct SFT and RL prompts consistently;
- support separate train/eval/test splits;
- keep prompt format versioned and documented.

Success criteria:

- data loading is deterministic;
- prompt generation is centralized rather than duplicated.

---

## Phase 5: Inference baseline

Objective:

- implement the simplest full-path runnable workflow.

Suggested entrypoint:

- `scripts/infer_single.py`

Target capabilities:

- load tokenizer/model/checkpoint;
- take one SMILES as input;
- produce one fragment output;
- run structural validation;
- save interpretable results.

Success criteria:

- one can test the contract “parent SMILES → fragment SMILES” before any large-scale training.

Status:

- minimal heuristic single-sample inference implemented on 2026-04-10 in `scripts/run_infer.py` and `src/eval/inference.py`
- trained-model inference remains a later step

---

## Phase 6: SFT subsystem

Objective:

- implement Stage A and Stage B supervised fine-tuning.

Suggested files:

- `src/train/train_sft.py`
- `scripts/train_sft.py`

Target capabilities:

- format-oriented SFT;
- weak-supervision SFT;
- config-driven hyperparameters;
- periodic evaluation;
- checkpoint saving.

Success criteria:

- the model learns to output structured fragment candidates with low parse failure rate.

---

## Phase 7: RL subsystem

Objective:

- implement Stage C RL for counterfactual optimization.

Suggested files:

- `src/train/train_rl.py`
- `src/train/rollout.py`
- `src/train/logging.py`
- `scripts/train_rl.py`

Target capabilities:

- policy rollout;
- reward computation and aggregation;
- KL/reference policy control;
- checkpointing;
- heartbeat logging for HPC runs;
- periodic validation.

Success criteria:

- RL training is stable enough to monitor;
- reward terms and failures are observable;
- obvious collapse is surfaced quickly.

---

## Phase 8: Evaluation subsystem

Objective:

- build a standalone evaluation path.

Suggested files:

- `src/eval/metrics.py`
- `src/eval/run_eval.py`
- `src/eval/reporting.py`
- `scripts/eval_model.py`

Target capabilities:

- run structural metrics;
- run deletion-based counterfactual metrics;
- collect qualitative examples;
- compare checkpoints;
- save machine-readable reports.

Success criteria:

- model quality can be assessed independently of training scripts.

---

## Phase 9: Testing and reproducibility

Objective:

- add the minimum research-grade reliability layer.

Suggested tests:

- chemistry parser test;
- substructure match test;
- deletion behavior test;
- reward term test;
- prompt formatting test;
- inference smoke test.

Suggested reproducibility measures:

- config snapshots;
- saved CLI commands;
- seed logging;
- environment notes.

Success criteria:

- changes can be checked without rerunning the entire project blindly.

---

## 4. Immediate Build Order

When starting from zero, the first concrete implementation order should be:

1. create the folder skeleton and typed module boundaries;
2. freeze the prompt and JSONL schema contracts;
3. implement RDKit-backed parsing, connectivity, substructure, and deletion in `src/chem/`;
4. implement reward term calculators on top of `src/chem/`, keeping counterfactual scoring explicit;
5. implement single-example inference using `src/models/`, `src/data/`, and `src/chem/`;
6. extend dataset and collator support for SFT and RL-specific batching;
7. implement Stage A and Stage B SFT entrypoints while preserving output-only-SMILES behavior;
8. implement Stage C RL entrypoints with reward breakdown logging and anti-collapse diagnostics;
9. implement standalone evaluation and checkpoint comparison;
10. expand tests from interface smoke tests to chemistry, reward, and inference coverage.

### Immediate next implementation steps after this bootstrap

1. Replace the chemistry placeholders in `src/chem/` with deterministic RDKit-backed implementations.
2. Wire `src/rewards/counterfactual_reward.py` to real structural checks and deletion-based flip scoring.
3. Implement `scripts/infer_single.py` as the first runnable end-to-end contract.
4. Add versioned config files under `configs/` once interfaces stop moving.

---

## 5. Risk Register

### Risk 1: Objective drift

The project may accidentally revert to concept extraction or rationale extraction.

Mitigation:

- keep the objective explicit in docs and comments;
- ensure evaluation includes deletion-based flip metrics.

### Risk 2: RL instability

The policy may collapse during RL.

Mitigation:

- expose per-term reward logging;
- control KL;
- monitor repeated-token behavior;
- save representative outputs periodically.

### Risk 3: Chemistry utility inconsistency

If chemistry logic is duplicated across files, behavior will drift.

Mitigation:

- centralize RDKit-related logic in `src/chem/`.

### Risk 4: Overcoupled scripts

A monolithic script will be hard to debug.

Mitigation:

- keep scripts thin and modules cohesive.

---

## 6. Definition of “Good First Version”

A good first rebuilt version should support the following end-to-end workflow:

1. load a JSONL molecule dataset;
2. construct prompts;
3. run model inference for one sample;
4. check whether output is valid and connected;
5. verify whether it is a parent substructure;
6. compute reward breakdown for a candidate;
7. run a minimal training/evaluation command.

If these are achieved with clean module boundaries, the rebuild is on the right path.

---

## 7. BACE WNode Prefix Optimization Extension

The BACE paper path now has an additive, versioned optimization route:

1. audit frozen rank and coverage funnels;
2. precompute a calibration-only WNode action matrix;
3. compare frozen selector variants with grouped calibration CV;
4. conditionally expand candidates only when the calibration limitation gate
   identifies a candidate-limited pool;
5. freeze one rank-preserving Top20 sequence;
6. run one new test evaluation and a non-promoting paper artifact audit.

This extension reuses the production evaluator and does not alter the AIDS or
Mutagenicity roadmaps.
# BACE Connected Candidate-Aware v4 (2026-08-10)

- [x] Preserve legacy matrix admission as an explicit default policy.
- [x] Add chemistry-only `connected_feasible_v4` candidate admission.
- [x] Add 151-to-matrix attrition and cross-dataset threshold protocol audits.
- [x] Add a versioned full connected calibration matrix wrapper and union report.
- [ ] Freeze a method-independent pooled calibration threshold contract.
- [ ] Run the preregistered connected-aware generation rounds only when the
  calibration union gate reports candidate limitation.
- [x] Add an opt-in connected-deletion prompt and source-side chemistry gate.
- [x] Add fixed Round-1 generation, merge, and calibration-matrix wrappers.
- [x] Add a complete native-rank GCF attrition audit that preserves Top20.
- [x] Add the calibration-only hard-parent Round-2 cohort and fixed regimes.
- [x] Add method-balanced pooled calibration Q30/Q50 threshold freezing.
- [x] Add pre-test selection/protocol gates and a one-shot Ours/GCF test job.
- [ ] Freeze Ours/GCF calibration selections before exactly one v4 test run.

# Storage-Safe Two-Lane Recovery (2026-08-17)

- [x] Add persistent-scratch and SQLite WAL preflight checks.
- [x] Add fail-closed projected-capacity monitoring for COMRECGC full walks.
- [x] Unify AIDS validator/recovery frozen graph and alias closure.
- [x] Load BACE GCF thresholds exclusively from a frozen shared manifest.
- [x] Add resumable, deterministic GlobalGCE root/epoch checkpoints without
  changing the official mining or training objective.
- [x] Add current-queue GPU accounting and a static two-lane plan validator.
- [x] Make checkpoint, integrity, chemistry, gate, and freeze wrappers CPU-only.
- [ ] Run HPC scratch/checkpoint/plan gates from the committed recovery worktree.
- [ ] Submit fresh MUT retry8 and BACE retry chains with one GPU per lane.
- [ ] Complete the downstream connected four-method artifact audits.

# Three-Line Recovery v7 (2026-08-18)

- [x] Persist complete AIDS original-hash, alias, transition, frontier, and
  recourse closure requirements across payload reload.
- [x] Add a fail-closed COMRECGC resume-or-finalize decision and two-slice BACE
  continuation wrapper.
- [x] Classify BACE GlobalGCE native candidates as full counterfactual graphs
  and remove the deletion-fragment matrix adaptation.
- [x] Spill low-support gSpan reports to resumable scratch SQLite and preserve
  official stable support top-k semantics.
- [x] Include the live MUT and BACE allocations in a two-GPU project planner.
- [ ] Revalidate and recover the completed AIDS walk on CPU only.
- [ ] Recompute GlobalGCE calibration matrices for min-freq 18/7/4 and stream
  min-freq 2 without a GPU.
- [ ] Complete GlobalGCE and COMRECGC BACE artifacts, then run common4 audit.

# AutoDL Four-Lane Three-Line Recovery (2026-08-21)

- [x] Add a fixed-run, data-driven four-lane AutoDL process orchestrator.
- [x] Keep persistent run state, logs, process provenance, heartbeats, locks,
  and scientific success/failure sentinels outside the disposable NVMe root.
- [x] Reserve the fast root for lane-local caches and the BACE active state.
- [x] Enforce one independent GPU, PID file, writer lock, cache, input, output,
  and command per lane; never represent an AutoDL PID as a Slurm job id.
- [x] Require `DISALLOW_GENERATION=1` for every preserved MUT/AIDS stage.
- [x] Gate BACE common4 on both BACE COMRECGC final and GlobalGCE WNode
  scientific success sentinels.
- [x] Provide one persistent `start/status/resume/stop` interface and reject
  an unconfigured command, writable input snapshot, unknown dependency,
  second writer, or untracked nonempty first-start output root.
- [x] Fill the exact recovery/downstream commands and add the production stage
  runner plus paired Slurm wrapper.
- [x] Add immutable primary/static/Step0 input gates, SHA-bound atomic stage
  sentinels, and fail-closed partial-output handling.
- [x] Add mandatory MUT/AIDS preserved-lineage smoke gates and a BACE
  formal-configuration SIGKILL/fast-loss profile gate bound to repair content.
- [x] Persist BACE trace chunks and latest-two checkpoint mirrors, and require
  the BACE artifact gate before common4 publication.
- [x] Bind every reusable substage and top-level sentinel to current input,
  command, environment, code/config closure, vendor commit, marker, and output
  SHA evidence; reject marker-only crash windows and stale outputs.
- [x] Bind local children to kernel start time, command digest, and process
  group; never signal a stale or reused PID, and scrub inherited credentials.
- [x] Support fail-closed incremental `start/resume --lane` activation while
  keeping omitted lanes `NOT_STARTED` and preserving no-flag four-lane launch.
- [x] Bind worker identity and persisted run/lane state to exact process,
  spec-byte digest, schema, run, lane, and normalized roots.
- [x] Require persisted stage and producer-lane success in addition to
  scientific dependency proofs, with sentinel-first success publication.
- [x] Provide Python-3.10/glibc-compatible exact pidfd signalling, cooperative
  stop markers when the kernel lacks pidfd support, and fail-closed manual
  handling instead of persisted-orphan `killpg` targeting.
- [x] Make completed-walk freeze recovery mirror the live global
  first-recorded predecessor index, with exact replay for every repeated event
  and explicit alias/convergence/conflict audit counters.
- [x] Treat frozen global-hash graph parent metadata as audit-only and bind
  recovered lineage ownership to a parent-consistent selected-event chain.
- [x] Strictly remap a uniquely determined recorded NLC representative index,
  separate selected-transition and unique-candidate recovery counts, and emit
  a checksum-bound fresh-root adoption manifest.
- [ ] Pass the AutoDL integration smokes, then start the four formal lanes.

# BACE and TasteMolNet Frozen-GNN Route (2026-08-22)

- [x] Audit the BACE frozen split and classify legacy oracle provenance.
- [x] Replace BBBP with TasteMolNet in the active dataset contract while
  preserving historical artifacts.
- [x] Add the generic molecular GNN registry, checkpoint bundle, calibrated
  batched oracle API, and BACE/Taste RF guard.
- [x] Add shared binary/multiclass strict-flip, CFDrop, margin, and destination
  distribution semantics.
- [x] Prepare the fixed-commit TasteMolNet upstream data with conflict,
  standardization, license, and scaffold-leakage audits.
- [x] Add the bounded AutoDL GPU inventory/lock/experiment registry and BACE
  gated state machine; keep `RUN_TASTEMOLNET=0` by default.
- [x] Move the AutoDL control plane outside fast code worktrees, freeze the
  persistent control root and `smiles_pip118` interpreter into detached specs,
  and add predecessor-bound B4 temperature/B5 oracle-smoke launchers.
- [x] Make GNN training hold test fully unopened, freezing only path/SHA
  provenance and an explicit `NOT_EVALUATED` status until final evaluation.
- [ ] Execute and pass B4 on validation and B5 on the 16-parent correctly
  predicted source cohort from calibration after B3 passes.
- [ ] Pass CPU/PyG tests and BACE GINE smoke, then launch the single seed-7
  BACE full classifier on an idle AutoDL GPU.
- [x] Add an honest B6 calibrated-GNN scoring diagnostic that leaves the PPO
  stage BLOCKED, plus executable B7--B14 blocker contracts; never label that
  diagnostic PPO or use it to release B7.
- [x] Add fail-closed initializer provenance, a fresh raw-base LoRA and bounded
  train-only oracle-neutral SFT path for BACE; reject unknown/RF adapters.
- [x] Inject one cached/batched frozen GINE reward adapter into the existing
  stable decoded-chemistry PPO loop without adding a second optimizer stack.
- [x] Add a real fresh-root B6-v2 five-update gate and a conservative 300-step
  B7 contract with checkpoints 50--300; retain old B6 blocker evidence.
- [x] Add explicit B6--B14 split-access and dependency-release contracts that
  cannot turn READY evidence into a scientific PASS.
- [ ] Run and pass the real 7B LoRA B6-v2 on AutoDL, then release B7.
- [x] Decouple the non-formal adapter canary from stochastic generated-deletion
  yield by adding a same-adapter, eight-parent train-only connected-deletion
  GNN preflight; keep formal B6 dependent on PPO-generated deletion evidence.
- [x] Implement the provenance-clean B8/B9 fixed train-parent shards, B10
  deterministic merge, batched-GINE/all-match/MolCLR-WNode B11 and B13
  verification, calibration-only B12 freeze, and manifest-only B14 gate.
- [x] Add B6-released B7-parallel calibration caches, fixed shard manifests,
  output preflight, and a foreground command/output contract for the AutoDL
  four-GPU controller.
- [ ] Execute B6-v2 and B7, then let the controller advance the implemented
  B8--B14 route without using RF-contaminated artifacts or pre-freeze test data.
- [ ] Obtain explicit TasteMolNet data-license approval before committing data
  or enabling any heavy TasteMolNet experiment.
- [x] Add a manifest-driven, persistent four-GPU AutoDL recovery controller
  that reuses `exp_run`, UUID locks, atomic gates, and append-only registry
  semantics; include deterministic train/calibration sharding and gated
  four-shard B13 held-out evaluation.
- [x] Bind existing Commit-A writers by exact launch-spec provenance, publish
  the user-facing JSONL/Markdown registry mirrors, and keep execution clones
  free of Python bytecode writes.
- [x] Allow the exact scheduler-owned `TOKENIZERS_PARALLELISM` environment key
  through `exp_run` without weakening credential-like environment rejection.
- [x] Integrate the Frozen-GNN downstream foreground contract into the
  controller with passing-attempt shard tokens, dependency-produced parent
  manifests, explicit B11/B13 shard-to-merge joins, and a post-B12-only test
  boundary.
- [x] Add a fresh B11--B14 continuation builder that exact-adopts B6--B10,
  flattens eight historical B8/B9 shard runs into single-instance evidence
  tasks, and substitutes the passing MolCLR repair without mutating v2.
- [x] Hold the predecessor controller lock for the full continuation lifetime
  and support an opt-in terminal heartbeat/poll mode without dummy work.
- [ ] Fill the persistent controller manifest with Commit A MUT/AIDS and Commit
  B BACE foreground argv/evidence contracts, validate it, and launch on AutoDL.
# 2026-08-22 — Four-by-four main experiment continuation

- Keep TasteMolNet heavy stages behind the new offline license gate. Continue
  BACE and adoptable AIDS/Mutagenicity work while that independent gate is
  blocked; do not synthesize a Taste row in the paper matrix.
- A later explicit approval can release the existing three-class pipeline
  without changing the dataset, label semantics, or frozen evaluation schema.
## 2026-08-22 — Four-method matrix continuation

- [x] Add a fresh-root, PASS-last continuation from immutable AIDS and
  Mutagenicity COMRECGC recovery payloads into standardized Figure 3/Figure 4/
  Table 2 artifacts.
- [ ] Register the resulting cells only after their project-full gate and freeze
  manifests pass the common 4×4 provenance contract.
- [ ] Schedule the two dataset continuations independently so a semantic failure
  in one dataset releases its GPU without blocking the other dataset.
- [x] Add isolated persistent controller/status/launcher entrypoints for the
  four-method matrix while retaining the audited UUID-lock scheduler engine.
- [x] Add a fresh, exact-path core task-fragment builder for the Taste license
  audit, four blocked Taste cells, and recovered AIDS/Mutagenicity COMRECGC
  standardization without generation reruns.
- [x] Preserve BACE continuation predecessor-lock metadata when composing the
  wider four-by-four controller manifest.

## BACE native baseline Frozen-GINE continuation (2026-08-22)

- [x] Add one calibrated-GINE adapter for official GCFExplainer and ComRecGC
  one-hot graph edit runtimes.
- [x] Freeze train-only GINE-clean native candidate universes without RF
  ranking or calibration/test access.
- [x] Add four-shard native full-graph GINE/WNode verification, calibration-only
  selector freeze, held-out evaluation, and PASS-last final freeze.
- [x] Publish controller-readable resource/marker contracts and AutoDL commands.
- [x] Add a one-way native-to-generic fragment adapter with passing-attempt
  dependency tokens, non-primary runner datasets, baseline-specific
  test-after-selector gates, and a bounded CPU GlobalGCE native-action
  preflight followed by a static no-resource training block.
- [x] Implement and validate pinned-official-parity GlobalGCE attachment-aware
  LHS→RHS application plus exact frozen-GINE forward scoring.
- [x] Admit a migrated read-only ComRecGC checkout through an exact,
  process-scoped Git `safe.directory` override while preserving the pinned
  commit, dirty-tree, and vendor-manifest gates; run that same checkout gate in
  BACE preflight before allocating a GPU.
- [ ] Design and scientifically approve a differentiable bridge from the
  official continuous RHS decoder to the frozen RDKit/categorical GINE input;
  until then keep BACE GlobalGCE full training fail-closed as
  `BLOCKED_GLOBALGCE_FROZEN_GINE_DIFFERENTIABLE_RULE_TRAINING_UNAVAILABLE`.

# Four-method × four-dataset paper matrix (2026-08-22)

- [x] Add an exact 16-cell registry with a closed state enum.
- [x] Add bounded, read-only multi-root artifact inventory and explicit
  candidate-root support.
- [x] Keep CLEAR distinct from ComRecGC and prevent render-only legacy CSVs from
  becoming paper PASS evidence.
- [x] Cross-check top-level continuation gates against nested standardized
  freeze manifests without inferring dataset or method from directory names.
- [x] Gate adoption on dataset/test/oracle/distance/threshold provenance and
  cross-method identity parity within each dataset.
- [x] Emit the unified WNode/strict-flip/K=1..20/Table2-K10 evaluation and
  standardized-export contract without synthesizing missing metrics.
- [x] Emit per-dataset evaluator-ready threshold contracts only from explicit
  calibration-frozen provenance; omit numerics when the contract is missing or
  test-derived.
- [x] Keep TasteMolNet blocked unless an explicit exact-data reuse basis passes.
- [ ] Populate the registry on AutoDL from persistent artifacts and exact
  expectations, then let the continuation controller schedule only missing or
  stale cells.
- [x] Add strict, fresh-root adoption for the exact frozen Mutagenicity Ours
  final result, including checksum closure, independent frozen-test replay
  audit, RF/MolCLR identity, and selector-before-test provenance.
- [x] Inventory the remaining AIDS/Mutagenicity legacy raw cells without
  rerunning generation; keep missing calibration/evaluation evidence
  `INCOMPLETE` and missing native GlobalGCE attachment semantics
  `BLOCKED_CODE`.
- [x] Freeze the exact Mutagenicity GCF Top20 exporter output into a fresh
  checksum-closed candidate package without rerunning generation; keep AIDS
  baseline roots absent from the AutoDL payload explicitly `MISSING`.
- [x] Register persistent Mut GCF freeze -> calibration selector freeze ->
  held-out evaluation tasks and extend the production no-test-before-freeze
  stage gate for that AM route.
- [x] Freeze and hash-check the shared AIDS/Mut 601-point threshold contract
  before either ComRecGC held-out continuation, with production controller
  enforcement of the selector-before-test dependency.
- [x] Track the matched 601-point Mutagenicity threshold expectation; mark the
  audited historical Ours bundle `STALE_METRIC` until its frozen pair matrix is
  deterministically re-exported under that common protocol.
- [x] Deterministically re-export the frozen Mutagenicity Ours 217-by-20 pair
  bundle on the matched 601-point protocol with no generation, selector,
  oracle, or MolCLR rerun and a fresh PASS-last standardized freeze.
- [x] Exclude CLEAR from every adoption and inventory path and provide distinct
  static blocked tasks for every known missing A/M terminal cell.
# 2026-08-22 — Deferred main-result ablation hooks

- [x] Keep a single registered GNN backbone axis (`gine`, `gin`, `gcn`,
  `gatv2`) and a dormant AutoDL task-plan builder.
- [x] Add a frozen-rule stability comparator covering exact, chemical,
  scaffold, coverage, and multiclass destination overlap.
- [ ] Run backbone/selector/reward/candidate-pool ablations only after the
  primary four-by-four matrix is scientifically unblocked and frozen.
- [x] Add a presentation-only Figure 3/Figure 4/Table 2 exporter that requires
  16/16 registry PASS plus per-cell hash/provenance closure, rejects CLEAR,
  preserves raw thresholds and Taste destinations, and emits no plausible
  numeric output for a partial matrix.
- [x] Add a generic CPU controller fragment whose final export task depends on
  16 distinct cell terminal PASS tasks and one post-cell matrix audit.
- [ ] Run the final exporter only after Taste licensing and every code gate
  release the full 16-cell matrix.

# 2026-08-25 — AIDS production-subset exact equivalence gate

- [x] Validate the terminal theta-close view plus physical pair/vector/distance
  authority before selecting any production-derived audit row.
- [x] Materialize deterministic first, random, dense, sparse, and theta-boundary
  induced subsets with original logical order and SHA256 provenance.
- [x] Compare sklearn with general external exact DBSCAN for partition, core,
  noise, centroid, strict radius, coverage, and greedy semantics.
- [x] Attempt the exact all-core certificate without allowing an inconclusive
  proof to fall back silently, and record applicability explicitly.
- [ ] Run this gate from the immutable AutoDL execution checkout after the
  production close-view PASS; never promote its subset PASS to full DBSCAN PASS.

# 2026-08-25 — One-cluster strict-radius post-hoc gate

- [x] Cast the NumPy trace radius to the distance dtype before strict `<`,
  matching official Torch scalar promotion without tolerance.
- [x] Add a terminal-manifest-bound A/B replay for widened NumPy, corrected
  NumPy, and official Torch masks, coverage sets, medoids, and selected trace.
- [x] Publish PASS last only when the live widened mask and corrected mask are
  identical; otherwise publish a corrected downstream-only replay artifact.
- [ ] Run the audit after c766 one-cluster summary completion and before AIDS
  standardization; never restart pair generation, close filtering, or DBSCAN.
# 2026-08-22 — Final matrix dependency closure

- [x] Bind the post-cell registry audit to sixteen distinct successful attempt
  roots using controller dependency tokens.
- [x] Generate the strict final-export dependency contract from the same cell
  mapping, preventing a reporting task from silently changing cell identity.

# 2026-08-22 — AutoDL Mutagenicity GCF runtime portability

- [x] Make the calibration and held-out Slurm evaluators use the absolute
  controller-pinned Python on AutoDL without depending on a non-interactive
  `conda` shell function.
- [x] Preserve the existing `.bashrc` plus `smiles_pip118` activation path when
  the same wrappers run under Slurm without `AUTODL_PYTHON`.
- [ ] Relaunch calibration from a fresh continuation task/output root; retain
  the exit-127 attempt as immutable failure evidence.

# 2026-08-23 — Shared low-memory GPU process attribution

- [x] Record direct child PID plus Linux start-time ticks in shared-slot
  metadata.
- [x] Attribute CUDA child/grandchild processes through a bounded live procfs
  ancestry walk while rejecting PID reuse and unrelated processes.
- [x] Keep two tasks per physical GPU, strict 70% admission, UUID locking, and
  MPS-disabled behavior unchanged.

# 2026-08-23 — BACE ComRecGC equivalent generation acceleration

- [x] Profile the live generation without signals, ptrace, or writes to its root;
  identify the single-core CPU/RDKit boundary and persist the evidence.
- [x] Prove from the pinned upstream state machine that generation-index/seed
  shards cannot be merged into the original 50k trajectory.
- [x] Add an opt-in ordered bounded RDKit process pool below one producer,
  with no worker RNG/CUDA state and separate provenance-bound LRU caches.
- [x] Add fail-closed 500/1000 diagnostic replay and artifact-parity auditors;
  diagnostic outputs can never become paper cells.
- [x] Add production-like RDKit fixtures, cache/order/shardability tests, and
  static CLI/Slurm parity without submitting to HPC.
- [ ] Run fresh 500 and 1000 legacy-vs-optimized BACE gates under the AutoDL
  shared-lowmem scheduler; do not mark the optimized full route ready until both
  persistent audits pass.
- [ ] After both gates pass, launch a fresh optimized 50k root with the existing
  full-state 500-step checkpoint/mirror contract; retain the live legacy root.

# 2026-08-23 — BACE GlobalGCE frozen-GINE bridge

- [x] Preserve pinned official attachment-aware LHS-to-RHS tensor parity.
- [x] Add a no-surrogate differentiable bridge over the exact calibrated GINE;
  freeze classifier parameters and keep hard-forward numerical parity.
- [x] Gate full training behind gradient/checkpoint/finite-output bridge smoke.
- [x] Preregister BACE `min_freq=7` from the 360-parent train-only grid and
  require the exact 869-row native train vocabulary.
- [x] Freeze native rule tensors from train only and preserve their payload
  through the calibration selector and held-out evaluator.
- [x] Add rule-native redundancy fingerprints rather than relabeling rules as
  deletion fragments or full counterfactual graphs.
- [x] Add four-shard calibration/test application with batched frozen-GINE
  scoring, min-legal-WNode match selection, and test-after-freeze enforcement.
- [x] Add the fourth BACE deterministic standardization task and final-matrix
  raw-terminal guard.
- [ ] Run the bridge smoke on AutoDL before releasing exclusive full training.
- [ ] Freeze and standardize the BACE GlobalGCE cell after all scientific gates
  pass; do not treat bridge smoke alone as a paper result.

# 2026-08-23 — Mut trace-off parity continuation

- [x] Bind the exact lineage-v3 trace-on source by recomputed config, dataset,
  parent order, classifier, distance, payload, and trace identities.
- [x] Add an exclusive-GPU trace-disabled 50k reference with stable scientific
  root, completed-step checkpoint mirror, bounded transient resume, and no
  trace/parity arguments in generation.
- [x] Separate the 7f algorithm identity from the reviewed 66487c0 checkpoint
  execution identity; freeze source/AST inventories and require a fresh
  same-seed 500-step candidate/action/RNG equivalence gate before the 50k run.
- [x] Reject self-comparison, copied claims without real run/checkpoint proof,
  trace stripping, test/calibration access, changed execution commit, and
  mutable AIDS dependency manifests.
- [x] Adopt repair-v2 common recourse only through its exact FAILED controller
  task and resume chemistry/evaluation/freeze CPU-only in a fresh root.
- [x] Keep all controller task outputs attempt-qualified while downstream tasks
  consume dependency-selected attempt roots.
- [ ] Fill the future AIDS repair manifest SHA only after its bounded-memory
  implementation and terminal output pass; then build and launch the persistent
  Mut controller from an immutable AutoDL worktree.

# 2026-08-23 — AIDS bounded-memory common recourse

- [ ] Persist theta-eligible `(parent, candidate, vector)` rows in immutable,
  hashed, resumable disk shards without per-row Python objects.
- [ ] Implement exact external-memory DBSCAN with bounded radius-neighbor
  batches, deterministic core connectivity, and sklearn-compatible border/tie
  assignment and label canonicalization.
- [ ] Stream cluster coverage, centroid, and medoid aggregation without a full
  `torch.tensor(recourse_array)` or per-cluster vector copy.
- [ ] Prove legacy/new exact parity on adversarial small fixtures, chunk-size
  invariance, checkpoint/resume equivalence, dense bounded-RSS behavior, and
  PASS-last failure injection before launching a fresh AIDS repair-v4 root.

# 2026-08-23 — GlobalGCE bridge controller evidence

- [x] Preserve the evaluator's atomic `PASS` and `BRIDGE_PASS` artifacts.
- [x] Emit the generic controller's exact bridge PASS marker from the thin CLI.
- [x] Cover the JSON result and marker with a focused CLI regression test.
- [ ] Launch only in a fresh controller/output root; retain the original
  marker-missing attempt as immutable evidence.

# 2026-08-23 — Three-dataset staging-only main results

- [x] Require one canonical 16-cell registry with exactly 12 paper-pass
  AIDS/Mutagenicity/BACE cells and four explicit TasteMolNet licence blocks.
- [x] Reuse the strict standardized-cell closure and within-dataset
  oracle/split/distance/threshold identity gates.
- [x] Render three dataset-specific Figure 3/Figure 4/Table 2 artifacts and
  three-dataset panels without recomputation, interpolation, smoothing, or
  numeric imputation.
- [x] Reject destinations under the user-owned `paper/` tree and publish only
  to fresh runtime result/staging roots with `PAPER_FROZEN_PARTIAL` provenance.
- [ ] Run the exporter only after the persistent controllers establish real
  12/16 PASS; never use its staging outputs as the final four-dataset result.
## 2026-08-23: BACE GlobalGCE exact top-k mining follow-up

- [x] Add an opt-in, stable top-k gSpan spill schema with anti-monotone branch
  pruning and bounded retained storage.
- [x] Preserve the legacy route as the default and bind the optimization into
  CLI/config fingerprints.
- [x] Add stable-tie, bounded-storage, and interrupted-root replay fixtures.
- [ ] Run a pinned official-gSpan monolithic-versus-pruned smoke on AutoDL and
  freeze its input/output hashes.
- [ ] Obtain an independent release review before creating any fresh full v6
  controller.
- [ ] Keep v5 running until it completes naturally or a reviewed replacement
  is projected to finish at least 30% earlier without losing a safe prefix.
- [x] Bind graph-list, node-insertion, and edge-traversal order in the exact
  resume fingerprint; reject cross-order adoption with a pinned-official
  counterexample.
- [x] Publish a complete checkpoint before the terminal exact audit and cover
  both writes with recoverable failure injection.
- [x] Remove exact mining from the historical RF BACE CLI/Slurm surface and
  expose it only through the frozen-GINE generic route.
- [x] Hash-bind and revalidate the exact proof through training summary,
  summary, run manifest, completion gate, and final PASS.

## 2026-08-24 — AIDS v5 reviewed-core integration identity

- [x] Retain `645c6e51...` as the independent-review source identity.
- [x] Require tree-equivalent integration commit `8c371b1c...` as an actual
  execution ancestor.
- [x] Validate frozen implementation/test Git blobs and current content
  SHA-256 in the manifest builder, with positive real-HEAD and negative drift
  tests.

## 2026-08-24: Root-cause acceleration progress health

- [x] Add procfs-start-tick-bound, read-only external task monitoring.
- [x] Separate RUNNING_PROGRESSING/SLOW/UNVIABLE/STALLED from PID liveness.
- [x] Persist heartbeat, state, queue, ownership, runs and status JSONL without
  acquiring a GPU lock or signalling scientific workers.
- [x] Deploy from an immutable worktree with a 60-second persistent heartbeat.

## 2026-08-24: BACE GlobalGCE v6 mining adoption and fresh fallback

- [x] Add a deep exhaustive-v2 adoption proof for failed-v5 completed mining,
  using SQLite `mode=ro&immutable=1` only.
- [x] Bind failed task/root, train-only data, frozen GINE, official commit,
  traversal order, 19 roots, stable top-20, source identities, sidecars, and
  no-writer closure.
- [x] Select adoption only on complete proof; otherwise run fresh exact-top-k
  v2 without consuming any v5 pattern payload.
- [x] Keep decision/bridge CPU-only and formal training on one exclusive GPU.
- [x] Deploy a fresh immutable v6 controller/root; never resume or relabel v5.
- [ ] Complete training, calibration, held-out evaluation, standardization,
  and final frozen-GINE closure.

## 2026-08-24: COMRECGC deterministic divergence trace

- [x] Add a safe JSON-only first-divergence report for selected transitions and
  candidate lineage.
- [x] Split frozen-GINE and NeuroSED devices without changing the legacy
  single-device default; bind both devices into command/config provenance.
- [ ] Complete the fresh deterministic legacy/optimized M500 pair and require
  exact trace, lineage, payload, checkpoint, and serialization parity.
- [ ] Decide sharding only after M500 parity; keep the old formal 50k writer
  running until equivalence, benchmark, and ETA gates pass.

## 2026-08-25: Root-cause controller health semantics

- [x] Preserve the read-only monitor's existing state schema while adding
  explicit controller liveness, worker liveness, scientific progress, and
  route-viability fields.
- [x] Add the SHA-pinned, physical-JSON receipt-gated `SUPERSEDED` terminal
  observation. It field-binds the old PID generation/root, graceful
  checkpoint/stop evidence, no-SIGKILL evidence, and replacement PASS
  gate/controller/task/root/manifest hashes. Boolean and PID-generation fields
  are type-strict JSON values; the monitor cannot send a process signal.
- [ ] Roll the fields into a future immutable AutoDL execution release only
  after the active AIDS full scan reaches its own terminal protocol gate.

## 2026-08-24 — BACE GCF lockstep and frozen-GINE scoring

- [x] Add Quick-50/100 per-call lockstep traces and an exact first-divergence
  comparator without consuming RNG.
- [x] Centralize duplicate-preserving full-batch frozen-GINE scoring with an
  optional exact-complete-batch cache.
- [x] Prove repeated-cold CPU byte identity and diagnose CUDA raw-byte drift.
- [x] Pass CPU-only legacy-A/legacy-B/patched Quick-50 lockstep exactly.
- [ ] Complete Quick-100 and M500 before releasing any optimized full route.
- [x] Add the 1/8/32/128/512 CPU/GPU benchmark and a persistent deferred
  controller bound to the ComRecGC pair owner and physical GPU2 UUID.
- [ ] Run that matrix only after GPU2 naturally releases; do not preempt any
  protected writer.
## 2026-08-24 — AIDS v5 promoted-pair physical snapshot

- [x] Add a CPU-only, no-hardlink physical snapshot primitive with full
  pre/post source hashing, per-array fsync/atomic promotion, and a terminal
  whole-closure validator.
- [x] Make incomplete partials non-authoritative and prove same-root recovery
  across array-promotion and terminal-manifest/PASS crash windows.
- [x] Close the builder-to-task natural-exit window and discard only fixed-name
  regular partials before recalculating remaining-byte disk headroom.
- [x] Freeze the Cartesian row formula, array shape/dtype, and exact sklearn
  DBSCAN semantics in a hash-bound `dbscan_contract.json`.
- [ ] Wire the snapshot as a dependency of the fresh persistent v5 science
  task and pin this reviewed snapshot implementation in its builder.
- [ ] Run only from a fresh immutable AutoDL worktree/root after independent
  review; retain repair-v4 as untouched read-only evidence.
- [x] Freeze the AIDS promoted pair-store column contract as
  `(parent_index, candidate_index)` under candidate-major/parent-minor row
  order, with a full bounded row-formula gate before physical snapshot copy.
- [x] Fail closed when an adopted Cartesian pair store is not bound to a
  validated theta-close logical view or complete `ALL_PAIRS_CLOSE` proof.
- [x] Add zero-copy bitmap/index close views, an explicit compact-storage gate,
  split exact DBSCAN certificates, and streaming one-cluster coverage output.
- [ ] Run the production GREED distance scan and first/random/dense/sparse/
  top-distance closed-subset comparisons before releasing the fresh route.
- [x] Make the full GREED scan survive controller/worker/host loss through one
  inode-frozen science root plus fresh attempt receipts; validate terminal
  arrays and bind later Mut consumption to the controller-authoritative unique
  PASS task gate rather than standardized attempt zero.

## 2026-08-25 — AIDS disconnected-anchor exact recovery

- [x] Prove the unique-seed-component recovery theorem and fail closed to a
  general external exact route when frozen seed anchors span components.
- [x] Add exact primary/full-anchor bridge scans, promoted-ledger replay,
  float64 boundary rechecks, canonical component labels, and separate
  core/connectivity/boundary/partition certificates.
- [x] Add bounded multi-component centroid/radius/coverage/medoid/greedy
  streaming with official-float32 decisions, float64 audit, and fail-closed
  discrete-decision disagreement checks.
- [x] Add a public c766 failed-selection promotion API that rebinds only small
  authenticated arrays/ledgers into a fresh root and never replays the complete
  seed/failure scan or copies the 25 GB source vectors; make every promotion
  artifact's two-name publication crash-reconcilable under the route lock.
- [x] Add a typed five-stage CPU-only recovery controller, code-generated
  production spec, CID-local no-clobber launch/status/restart commands, and a
  formula-derived disk/RSS/coexistence contract.
- [x] Place exact DBSCAN and component summary in the native standardized
  continuation layout so one restartable final task can resume chemistry,
  WNode evaluation, exports, and freeze.
- [x] Put every controller worker and standardized-continuation child behind a
  durable exec startup barrier: the target cannot execute until PID/process-
  group identity is fsynced, and parent death before release yields EOF plus an
  inode/flock quiescence proof instead of an unbound writer.
- [x] Treat the formula-derived disk limit as a hard PASS-publication gate,
  with exact reservation for controller writes and periodic graceful-stop
  checks for science subprocesses; do not misstate it as a filesystem quota or
  the exact-DBSCAN-only 96 GiB native RSS certificate as a route cgroup limit.
- [x] Document the finite recovery boundary: one <=1 GiB archive per
  non-checkpointed downstream stage is same-CID recoverable; a second or
  oversized interruption blocks for manual diagnosis/fresh CID and may not
  publish PASS.
- [ ] Pin a fresh independently reviewed adoption-v3 commit and the final
  controller integration commit; keep production authorization false until
  both reviews pass.
- [ ] Build an immutable execution worktree, run production spec generation
  and dry CLI parsing, then launch one fresh AutoDL CPU controller.
- [ ] Require subset, exact, downstream replay, standardized freeze, and final
  controller gates before updating AIDS/Mut or stopping the old brute route.

## 2026-08-25 — TasteMolNet scoped data-use and three-class GINE activation

- [x] Add the exact machine-readable research/reporting-without-redistribution
  policy interface while retaining `NOT_EXPLICITLY_STATED` as the upstream
  terms status and forbidding every `LICENSE_PASS` claim.
- [x] Preserve the historical `LICENSE_REVIEW_REQUIRED` artifact and make the
  legacy binary audit permanently non-authorizing.
- [x] Add read-only checksum closure for the existing prepared split and graph
  cache; forbid another download, preparation pass, cache rebuild, or source
  copy.
- [x] Add a manifest-closed public-artifact audit that rejects raw/cleaned or
  reconstructable dataset content, molecule-level records, per-example
  predictions, hidden archives, and protected hashes.
- [x] Supersede that v1 execution placement with a policy-v2,
  GPU-1-exclusive, fresh-root, true three-class GINE
  task fragment with Sweet=1 strict untargeted flips, no RF, no test loading,
  and no HPC eligibility.
- [x] Activate only private research computation and aggregate paper reporting:
  `RUN_TASTEMOLNET=1`, while preserving `NOT_EXPLICITLY_STATED` and forbidding
  all dataset redistribution.
- [x] Require exact policy/receipt/prepared/cache closure at training start and
  terminal, validation Macro OvR ROC-AUC selection with Macro-F1 tie-break,
  validation-only temperature scaling, and held-out-test path/hash only.
- [x] Add a separate exact-contract epoch state root with atomic model/
  optimizer/best/RNG checkpoints, same-root resume, checkpoint cleanup audit,
  terminal reopen, and staged immutable-output publication.
- [x] Bind resume to the complete canonical config and config-file/override
  hashes, clean commit/tree/source identity, Python/Torch/CUDA runtime, and
  physical GPU-1 UUID; prove uninterrupted and crash/resume RNG/model parity.
- [x] Hold a physical output-parent dirfd/lock/sentinel/claim for the long
  route, reject output/state overlap with prepared/cache roots before loading,
  and publish the deterministic contract staging root with atomic no-replace.
- [x] Add the dedicated persistent Taste controller, durable startup barrier,
  same-generation orphan adoption, one process-loss-only same-state retry,
  bounded GPU-1 waiting/logs, terminal babysit, and typed terminal status.
- [x] Keep terminal source/state/output/finalization authorities held through
  stable full inventories and publish controller `PASS` last; reject same-byte
  staging, state-root, named-lock, and output-parent replacement.
- [x] Make PASS or any terminal-named artifact enter the shared strict
  read-only validator before resumed-controller writer reconciliation.
- [x] Bind the physical training-contract inode/content/file SHA and
  recomputed canonical SHA through checkpoints, completion, and terminal
  evidence.
- [x] Add the controller-issued completion-only adoption receipt for the
  finalization-published/completion-missing crash window while preserving the
  ordinary `exp_run` fresh-output gate.
- [x] Register the real trainer child behind a durable inner startup barrier
  and retain exact PID/start/cwd/argv/cmd/exe/ancestry ownership across
  `exp_run` parent loss without concurrent retry.
- [x] Fail closed on deterministic CUDA/PyG parity drift and freeze the exact
  GINE/seed-7/20-GiB-reservation/100-GiB-post-reservation route plus
  NumPy/RDKit/PyG/cuDNN/driver/runtime
  manifest closure.
- [x] Hold and verify manifest-bound train/validation cache path/SHA/inode
  before and after deserialization, carry it across resume/terminal, and keep
  calibration/test caches unopened.
- [x] Deserialize both train and validation caches from duplicated streams of
  the held authenticated descriptors; prove a hostile cache-root swap cannot
  redirect `torch.load` and is rejected by the post-load named-inode closure.
- [x] Strictly validate trainer authority/process/barrier structure before
  liveness filtering, ignore only conclusively dead declared PID/start
  generations, and reject live stale, malformed, unreadable, or concurrently
  live generations before retry.
- [x] Classify exact Linux PID/start zombie/exit state before worker or trainer
  argv-phase validation; boundedly retry transient teardown/exec snapshots but
  fail closed when an exact live generation retains empty or malformed argv.
- [ ] Complete independent review, immutable AutoDL deployment, and the first
  durable three-class GINE checkpoint on physical GPU 1.
- [ ] Deploy only from an immutable execution worktree and only to fresh AutoDL
  controller/science roots; do not modify existing experiments.
- [x] Add a fresh policy-v2 main-controller namespace, typed old-block adoption,
  T0--T16 evidence skeleton, main-controller owner lock, event logs, GPU-2
  classifier-independent READY lane, and an explicit no-ablation contract.
- [x] Add an exact supplemental downstream policy bound to policy v2: T3 may
  only adopt the already validation-fitted temperature, T4 may open only the
  authenticated graph-cache manifest and `calibration.pt` on GPU 1, and a
  separately typed future T6 smoke uses only the frozen prepared train CSV,
  is train-only with frozen-GINE reward, and has no
  RF/validation/calibration/test access.
- [x] Recompute T3 NLL/ECE/Brier/argmax evidence from the immutable bundle,
  forbid refit/copy/mutation, retain `model.pt` as the selected asset, and
  publish a fresh hash-closed controller-facing gate.
- [x] Add the deterministic sixteen-parent multiclass T4 interface smoke with
  bounded connected deletions, one loaded GINE oracle, batch/single parity,
  Sweet-to-Bitter/Sweet-to-Tasteless destination aggregation, and fail-closed
  invalid/full-parent controls.
- [x] Keep T4 outputs aggregate-only: no CSV, SMILES, molecule IDs, or
  per-example predictions; keep train/validation/test payloads unopened and
  record `test_payload_opened=false`.
- [x] Add thin AutoDL wrappers, an AutoDL-only non-runnable paired Slurm parity
  entrypoint, exact-type hostile tests, cache-open guards, and output-hash
  closure tests.
- [x] Anchor checkpoint, cache, and stage-output child reads to retained
  physical directory descriptors; reject permanent and temporary
  swap-and-restore attempts during temperature/model/gate reads.
- [x] Constrain T3/T4 output to exact direct fresh `calibrated-*` and
  `t4-oracle-smoke-*` children of the AutoDL Taste/GINE/seed-7 artifact root;
  retain root-to-leaf output descriptors, reclose checkpoint/cache/T3/policy
  authorities while the output remains non-terminal, and publish PASS marker
  last without any check-then-unlink cleanup.
- [x] Expose retained stage, checkpoint, and supplemental-policy authorities so
  T6 can bind the exact checkpoint path/ID/full/stat/manifest hashes across its
  model/reward load; equal-byte copies and symlink aliases fail closed.
- [ ] Add an explicit main-controller resume action that may run T3 only from
  strict T2 PASS/T3 READY, then T4 only from T3 PASS/T4 READY; it must never
  fall back to T2 or launch another classifier training run.

## 2026-08-25: c766 failed-selection recovery authority

- [x] Add an independent read-only primitive pinned to the exact controller
  namespace/CID, close PASS, FAILED/SEMANTIC final task, and fixed selection,
  failure, pair, vector, and close-view hashes.
- [x] Close the adaptive seed/failure ledgers and disconnected anchor graph;
  freeze canonical component labels, the unique size-three seed component,
  and self-inclusive anchor-degree hash/minimum without asserting a partition.
- [x] Publish an O_EXCL receipt under an inode-bound physical lock/output,
  perform the second complete authority scan before a recovery-only READY-last
  marker, revoke only an exact receipt-bound READY inode on terminal drift,
  and make typed terminal reopen idempotent without creating generic `PASS`.
- [x] Freeze canonical non-heartbeat close/final state projections and record
  every observed whole-state SHA; require exact top/main/PID-identity schemas,
  freeze every static field, and allow only well-typed UTC `updated_at` and
  `instances.main.heartbeat_at` values to drift. Hold control/namespace/
  controller/task and source/output directory inodes across scans so
  rename-plus-copy cannot become authority.
- [x] Require the failed attempt's exact 14-file path/SHA inventory and reject
  missing, extra, symlink, special, or terminal-looking injected artifacts.
- [x] Restrict output to a direct fresh child of the dedicated fixed
  `outputs/autodl/recovery_evidence/aids_c766_failed_selection_v1` parent and
  reject ancestor/descendant overlap with every discovered source root/file
  before creating the lock or output. Hold that parent at call entry and use
  its dirfd/openat identity for the sibling lock and output creation.
- [x] Require exact top-level and nested typed-receipt schemas, repeat the
  descriptor-based 14-file inventory after tracked hashes and before READY,
  and reject re-signed extra fields or files injected between full scans.
- [x] Add focused tamper, symlink, gate/status/path, PID-reuse/live-worker,
  writable-FD, state-projection churn/tamper, exact-inventory injection,
  partial-output, stale/replaced-lock, namespace/output rename-copy,
  preterminal crash, terminal drift/revocation, and replacement-inode tests
  plus an AutoDL CLI and static-refusal paired Slurm script.
- [x] Require a third complete authority reopen after the READY hardlink while
  the same output lock remains held; revoke only the exact recorded READY inode
  on any controller/gate/state/proc/source-directory/source-artifact/failed-tree
  drift. Exhaustively exercise every fixture source artifact and directory.
- [x] Validate every `process_exit` receipt field against the frozen final-task
  worker identity and child PID. Permit cross-scan procfs changes only among
  the safe absent/proven-reuse/zombie worker observations and absent/zombie
  child observations, with no signals and an exact exited boolean.
- [x] Merge into the independently reviewed recovery branch as an actual
  ancestor and bind its builder to the typed v3 external receipt without ordinary PASS,
  matrix, or Mut dependency consumption.
- [x] Bind the controller to the reviewed v3 canonical task-state projection
  hashes and add a cross-module profile equality test so a stale pre-v3 digest
  cannot make production spec generation deterministically reject the receipt.

## 2026-08-28: TasteMolNet T2 completed-result adoption boundary

- [x] Add a versioned, read-only source validator for the completed Taste GINE
  run that preserves the old controller's exact
  `FAILED/WORKER_PROCESS_IDENTITY_DRIFT` record.
- [x] Bind the deployed `583bf` execution source, `3a90` identity fix, typed
  trainer authority, dead PID set, registry/runtime PASS records,
  held runtime-log PASS/OK/exit-0 markers, `training_complete`, exact 18-hash
  checkpoint closure, and full
  output/training-state inventories under retained `openat`/`O_NOFOLLOW`
  authorities.
- [x] Define a deterministic five-file `T2_GINE_FULL_PASS_ADOPTED` receipt in
  the exact independent control namespace, with `manifest.json` as the receipt
  and a physically bound `gate.json` exposed by the final no-replace rename.
- [x] Preserve the old controller FAILED false-negative and the independent
  registry/runtime/training-complete/formal-bundle PASS facts explicitly;
  revalidate held old sources around every write, keep main/matrix outside the
  open/write set, and constrain future T3 authority to the fresh T2
  gate/receipt plus its exact formal-bundle inventory.
- [x] Keep the tracked release config at exact native `authorization=false`
  with null external-authority path/hash; expose only usable read-only
  preflight/status behavior in this stage-freeze.
- [x] Bind all four preterminal documents and their held directory by physical
  identity in the terminal gate; require no validation/fsync after publication
  and non-throwing cleanup, and reject equal-byte leaf/root replacement in
  status.
- [x] Require literal production `/proc`, exact FAILED-state and process
  schemas, native integer PID fields, and fail-closed live/PID-reuse handling.
- [x] Harden immutable Git audits with fixed `/usr/bin/git`, isolated config and
  environment, explicit gitdir/worktree authorities, critical committed blobs,
  and rejection of dirty/staged/untracked/ignored/index-hidden/bytecode state.
- [x] Add the paired AutoDL CLI, static-refusal Slurm parity, focused hostile
  tests, and operator documentation.
- [ ] After a separate production evidence capture and review, create the
  exact external release receipt, then a clean one-parent commit changing only
  the release config. Never patch or reconcile this version's partial/final
  root.
- [ ] Implement T3 so its only T2 authority is the validated fresh gate/receipt
  SHA plus the exact formal-bundle inventory in that receipt; require T4 to
  retain that receipt binding and consume T3's own gate before PASS.
