# T14 retry2: candidate-first storage repair (2026-09-06)

## Observed failure and preserved evidence

The existing retry2 owner at
`/autodl-fs/data/counterfactual-subgraph-runtime/control/t14_route_c/owners/route-c-59f101cd-f30b-458d-aa8c-2eb93ae82609`
wrote `terminal.json` at `2026-09-06T02:23:56.296226+00:00`:
`FAILED`, `lowmemory-continuous-510`, exit 1. Its last heartbeat is older than
the terminal and must not be interpreted as a healthy owner.

The exception is `T14RouteCFreshError: Route C candidate graph is not in graph
store`, originating in `RouteCMMapCandidateState._append_record`. The low-memory
stage's progress is step 0 / COHORT_FROZEN; its checkpoint directory is empty
and no completed-step ledger exists. This is not an OOM or a demonstrated
scientific parity divergence.

Preserve without modification:

- Reference 500:
  `canaries/reference_500/20d02bb8-f69e-4726-9ce4-cc1358b9b870/science-20d02bb8-f69e-4726-9ce4-cc1358b9b870`
  beneath that owner. It has checkpoint500 and a 500-step ledger.
- Failed low-memory stage:
  `canaries/low_memory_continuous_510/19af6213-e51b-4bf8-819d-c87ab66606f1/science-19af6213-e51b-4bf8-819d-c87ab66606f1`.
- The existing reference specification, owner plan, logs, failed terminal,
  generation specification and continuation binding.

No science, replacement stage or owner is launched by this change. No legacy
42.6-GB checkpoint is adopted; no third full retry is requested.

## Exact cause

The frozen official COMRECGC source is commit
`122f9341a360e9f06bb58a2f5823bb596021f6bf`.

`comrecgc.py:402` calls `populate_counterfactual_candidates` for a non-lead
head *before* `comrecgc.py:410` writes its actual graph-map entry. A lead head
uses the opposite ordering. The former Route C append path required an already
materialized graph ID and therefore failed at the first selected non-lead
candidate. Moving graph-map insertion earlier would be incorrect: it changes
the subsequent `if hash not in graph_map` and `bypass_size`/frequency behavior.

## Minimal representation repair

Candidate append now reserves only `(stable numeric ID, exact key bytes,
key SHA)` in a compact `graph_keys` table in the **existing** graph-index SQLite
file. Reserving a key does not append a graph blob, add an active graph-map
entry, satisfy `contains`/`graph_id`, consume a scientific sequence ID, invoke
an oracle, or consume RNG. The actual graph is written at the unmodified
official graph-map assignment. Missing payloads remain missing until then.

The reservation also handles a transient candidate evicted before graph
materialization: its immutable record remains valid, but no graph is fabricated.
Numeric ID collisions across pending and materialized keys fail closed.

Only the physical graph-store checkpoint schema changes to
`tastemolnet_t14_route_c_append_only_graph_store_v2`; candidate record and
scientific digest schemas are unchanged. New checkpoint state records the
reservation count/policy. Independent external-state recovery checks the
sealed reserved-key table and binds graph/candidate locators and mmap IDs to
those keys before applying the existing recovery operation. The existing sealed
SQLite snapshot already contains this table; there are no extra checkpoint files.

A populated v1 store cannot be upgraded in place. The replacement low-memory
stage must use a fresh root; the old failed root is preserved. This schema change
does not require rerunning the reference500 (which uses reference storage).

## Verification and remaining production gate

Run locally, without model/GPU/dataset loading:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
T14_OFFICIAL_SOURCE=/absolute/pinned/comrecgc.py \
python -m pytest -q \
  tests/autodl/test_t14_route_c_candidate_registration.py \
  tests/autodl/test_t14_route_c_fresh.py
```

Observed: **33 passed**, including six new focused cases. The optional official
integration test was actually executed, not skipped, using the frozen source
prefix read from AutoDL. It AST-loads nine exact unmodified official functions;
their normalized AST SHA is
`59d9523a2e1d74658475b4fbac830f0d19eb56ed2c6bb57c394de08162c51e0e`.
Four tiny two-head moves compare plain-list versus low-memory candidate records,
frequencies, ordering, active graph identities, move outputs and exact RNG state.
The graph serializer processes real CPU tensor graph payloads, not a mocked
graph hash. No learned oracle or production pair data is loaded.

Other tests cover pending ID collision, no premature materialization/RNG/sequence
change, candidate replacement before materialization, compact checkpoint/reopen,
uncommitted key-only suffix rejection and existing external rollback behavior.

These tests are **not** production 500/510-step parity or full resume PASS.
The authorized failed-stage replacement still requires:

1. A new immutable execution commit and a fresh low-memory child root.
2. Explicit binding to the existing retry2/reference500 and preserved failed root.
3. New low-memory continuous510 and independent reload250→510.
4. Exact reference1–500, continuous/reload1–500 and501–510 parity.
5. Existing memory/storage/inode and GPU-owner admission; no guard is lowered.
6. Existing final20k generation and publisher path only after those gates.

The old owner command alone cannot do this: its sealed plan names the crashed
child, and that child's nonempty root lacks sealed510. A separate small
failed-stage replacement binding is needed, retaining retry index2 and the old
reference. Do not reset the old plan, delete the failed root or rerun reference500.

## Resource impact

The key table reuses the already-required index/WAL/checkpoint files, so its
steady-state additional inode count is zero. Index bytes grow with distinct
observed keys and must be included in the replacement stage's measured compact
peak; no numerical byte-admission PASS is inferred from the tiny tests. Fresh
child/owner/receipt roots still consume additional inodes. The existing100000
inode guard remains in force; no AutoDL/HPC/Mac data are deleted by this change.
