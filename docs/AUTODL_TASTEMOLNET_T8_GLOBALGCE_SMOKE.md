# AutoDL TasteMolNet T8 native GlobalGCE smoke

## Status

T8 is implemented but deliberately release disabled. The authoritative file
is `configs/autodl/tastemolnet_t8_globalgce_smoke_release_v1.json`; its release
bit is false and every production pin is null. The tracked AutoDL wrapper exits
78 before sourcing runtime helpers, reading caller-selected paths, checking
storage, or probing a GPU. The paired Slurm wrapper always exits 64 before its
otherwise CLI-parity Python command because Taste policy-v2 science is
AutoDL-only.

Do not flip either release literal by hand. A later clean one-parent successor
may enable the release config and wrapper together only after the immutable
integration tree and external authorities have independent review.

## Scientific boundary

The smoke is one bounded native GlobalGCE execution with these fixed choices:

- dataset `tastemolnet`, seed 7, class order Bitter=0, Sweet=1, Tasteless=2;
- one frozen calibrated three-class GINE, never two binary classifiers;
- true/predicted-Sweet source cohort selected from at most 64 prepared train
  rows, with exactly 16 parents retained;
- independent target branches 0 and 2;
- `min_freq=2`, native Top20, training parameter `epochs=5`, learning rate
  `0.1`, dropout `0.5`, bounded generation/oracle/gSpan chunks;
- established native attachment-aware LHS-to-RHS rewrite semantics;
- original-order three-class final acceptance
  `pred_before == 1 and pred_after != 1`;
- at least one accepted strict flip attributable to each branch.

Each branch is called twice. The first call is deliberately interrupted only
after the epoch-0 checkpoint and heartbeat are durable. The second call must
restore the same identity-bound model, optimizer, scheduler, and
Python/NumPy/Torch RNG state and then reach the terminal rule catalog. Cross-
target, cohort, GINE, official-source, or configuration resume is rejected.

The freshly created target directory and its epoch-0 checkpoint/heartbeat
leaves remain open by descriptor across interruption and reload. The resumed
loader deserializes the exact held physical checkpoint; the complete branch
tree is captured inside the generator completion boundary. Target-directory
replacement or same-byte checkpoint-leaf replacement therefore fails closed.

Rules from both branches are merged by exact canonical LHS/RHS action content.
Generated residuals are canonicalized and deduplicated before one batched
reload of the same frozen GINE verifies all source and destination labels.

## Data and input authority

Only the prepared train CSV is loaded. GlobalGCE's internal train/validation
partition is made entirely from those held train rows; it is not the dataset
validation split. T8 does not stat or open validation, calibration, or test
payloads, and it has no RF, native binary-GTGNN, BACE, heuristic, CPU fallback,
or GNN-ablation route.

The runner retains and revalidates through the terminal commit:

- the exact receipt-only T2 adoption root, gate, receipt, source evidence, and
  formal GINE inventory;
- exact held T3 and T4 terminal roots, whose complete T2 binding and frozen
  checkpoint identity must match each other and the supplied checkpoint;
- all seven frozen GINE payloads, loaded from held bytes;
- the exact prepared train CSV, also parsed from held bytes;
- the supplemental and base no-redistribution policies;
- every tracked Python source file in the clean pinned official GlobalGCE
  checkout, with imports rooted at a held checkout descriptor on Linux;
- the immutable implementation/release/wrapper/config identities;
- the injected managed child execution authority.

The worker consumes an independent held authority that cross-binds its managed
run, task, GPU UUID and live ACTIVE child generation to T2/T3/T4, the one
three-class GINE, official GlobalGCE, prepared train split, policies, execution
commit, and fresh roots. A mapping embedded in worker evidence is not that
authority. The public consumer and the independent verifier both revalidate
the external held authority and reject a wholly rehashed fake root whose
expectations come from the worker itself.

The only preparatory consumer surface is the narrow
`TasteT8ExternalManagedV2Authority` protocol. It accepts a held object exposing
`revalidate_t8_managed_v2_authority()` and an independently captured
`revalidate_t8_official_startup_authority()` expectation; raw mappings and the
legacy generic `revalidate()` holder shape are rejected. Its exact evidence binds managed-v2
source commit, task/run, held authority record, ACTIVE generation, child process
identity and lineage, exclusive GPU UUID, disabled auto-termination, and a
controller-supplied closure hash. T8 recomputes that closure from the retained
execution, fresh roots, T2--T4, GINE, train, official, and policy authorities.
This repository intentionally contains no provider implementation for that
protocol.

Managed execution v2 commit
`3405ae1d24fdaeb7a4af40b14823b36051966a35` supplies only the frozen generic
attempt/staging/sealing and independent atomic publisher API. T8 now uses that
typed boundary, but no reviewed adapter yet derives T8's held GPU/ACTIVE
authority from a managed-v2 controller. The tracked wrapper therefore exits
before GPU preflight or science. It does not use the rejected marker primitive,
does not pretend that the old managed-v1 receipt/registry is managed-v2
authority, and does not synthesize a local controller, lease, or receipt.

## Official API and import authority

The official checkout is pinned to commit
`157e65c2850bc787f229a1ee8c60564906b933f2`. Startup runs with `python -I -B`
and `PYTHONNOUSERSITE=1`. Before any official training call, the adapter uses
`inspect.signature` to compare the exact reviewed constructors and functions,
including `GTGNN`, `GlobalGCE`, `FrequentSubgraphGenerator`, `gSpan`, train,
test, generation, expansion, rule, batch, and concatenation APIs. Variadic
parameters, signature drift, `TypeError` retry, and `**kwargs` compatibility
fallbacks are forbidden.

Imports are rooted at the descriptor-held checkout. Preloaded `models.*`,
`data.*`, or official top-level `utils` modules are verified before removal and
fresh loading; a foreign preload, bytecode/native shadow, `__pycache__` origin,
ignored/untracked runtime source, origin/loader mismatch, inode replacement, or
hash change fails closed. Both target branches persist the same canonical API
signature document and a provenance document containing path, realpath,
device/inode, size, and SHA-256 evidence for every official source module plus
`globalgce`, `torch`, `torch_geometric`, the project adapter/bridge, and oracle.
The worker carries those full documents, not just worker-chosen digests, into
sealed raw evidence for independent verification.

## Private state and managed-v2 terminal output

Branch checkpoints, rule catalogs, and any molecule-level working data stay in
one fresh private state root. That root is not a terminal/public artifact.

The worker creates one UUIDv4 managed attempt and one unique UUIDv4 staging
generation. Its protocol writes only:

```text
raw_evidence.json
worker_exit.json
SEALED.json
artifacts/
```

T8 currently publishes no scientific payload below `artifacts/`; branch
checkpoints and molecule-level state remain in the separate private state root.
Raw evidence contains aggregate science, the full external authority snapshot,
the exact attempt manifest, and the full official startup documents. It records
that no private state was published and no data was redistributed. The worker
cannot write `verification.json`, `gate.json`, or `PASS`.

A separate verifier opens `SEALED.json` by held descriptors, rehashes the
inventory, revalidates its independently supplied authority, repeats the
science/resume/GINE/import checks, and only then supplies a method-specific
`status=PASS` verification to managed v2. The generic publisher writes
`verification.json`, `gate.json`, and
`[MANAGED_EXECUTION_V2_PASS]\n` in the still-private directory, then publishes
the complete directory with atomic no-replace rename. Cross-filesystem mode
copies from held descriptors into a unique destination-side directory, fsyncs
and rehashes it, then performs the same rename. No hardlink is used.

The legacy six-file T8 terminal candidate and its marker-last consumer remain
testable predecessor code only; they are not a production publication route.
Checking only file names, worker raw evidence, or stdout is never authority.
Its stage marker is nevertheless frozen by the T0--T16 contract as exactly
`[TASTE_T8_GLOBALGCE_SMOKE_PASS]`: legacy structured `marker` fields, stdout,
and the legacy `PASS` leaf use that same already-bracketed string, with exactly
one trailing newline only in the leaf. The active managed-v2 terminal continues
to use its generic `[MANAGED_EXECUTION_V2_PASS]` outer marker; T8 success
remains nested method verification and is not inferred from stdout.

## Release checklist

A later release successor must complete all of the following without editing
this stage-frozen implementation:

1. freeze and independently review the managed-v2 controller adapter that
   supplies the held T8 GPU/ACTIVE child and predecessor authority; do not wire
   the rejected marker primitive or managed-v1 registry;
2. pass fresh focused and adjacent no-cache tests plus independent review of
   the exact integration tree;
3. pin one immutable implementation commit/tree and critical blob map;
4. pin the T2 receipt-only authority, matching T3/T4 roots, exact GINE/train/
   policy/official-source identities, and fresh state/output parents;
5. issue a controller-owned managed-v2 held child authority for exclusive
   physical GPU 2 and preflight atomic no-replace directory rename on the exact
   AutoDL target filesystem;
6. change only the release config and tracked wrapper literal in one clean
   successor, then make the independent verifier publish and the controller
   adopt the exact managed-v2 gate/PASS.

Until then the truthful outcome is `TASTE_T8_GLOBALGCE_WRAPPER_NOT_RELEASED`.
There is intentionally no launch command in this document.
