# Mut observational causal-lineage repair

Status: **producer implemented and tiny tests passed; conditional Route B is
still BLOCKED_CAUSAL_PRODUCTION_PARITY_REQUIRED, not end-to-end READY**.

This change does not deploy, submit, restart or modify any healthy main-table
task. The existing A/B experiment must finish. Only its reopened, byte-bound
actual scientific failure may select Route B. An engineering failure, missing
logs or an incomplete comparison cannot select it.

## Actual action authority

The pinned official `move_to_next_graph` selects a lead target index and
deterministic follower indices from the embedding-hash-deduplicated target
lists, then updates the candidate population before returning. The existing
`CompactMoveScopedTransitionMap` retains the action at each of those exact
indices. The new observer reads `action_records(source_hash, target_hash)`
before that move's cache scope is cleaned up. It never infers an alternate
action, re-enumerates neighbors, touches target order, calls a classifier or
draws random numbers.

Every selected source/action must replay exactly to the actually stored target.
Missing/ambiguous actions and embedding identities that do not map to that
exact graph fail closed. The original walker may continue to be scientifically
valid in such a case, but this recorder cannot claim a valid causal proof.
This is one reason the production comparison is still mandatory.
The reused final serializer's replay audit must additionally show zero action
inference, node-index remapping or semantic-alias action substitution, and
exactly account for every selected event. Legacy recovery capabilities cannot
silently change this new producer's actual recorded action before causal seal.

Events and exported checkpoint state are deep snapshots. No mutable action
list is shared back with the transition authority. Only selected events are
persisted (512 per compact chunk), plus the existing compact candidate lineage
index and typed manifest. `trace_enabled=false` continues to mean no optional
debug trace; `causal_lineage_enabled=true` explicitly declares the newly
authorized scientific receipt. No legacy `/trace` directory is created.

The new argument is:

```text
--mut-causal-lineage-output-dir <fresh-generation-root>/causal_lineage
```

It is accepted only by the project Mut full route, with no `--trace-output-dir`.
The default adds neither output nor scientific argv fields to existing runs.
It is not silently accepted by the native route. The synchronized entrypoint is
`scripts/slurm/run_generation.sh`; it is not submitted by this change.

## Evidence already established locally

- Deterministic 500-step tiny indexed-move fixture: observer off, new causal
  observer and existing debug observer have identical RNG, selected actions,
  heads, graphs, candidate registry, candidate frequencies and transition order.
- Step 500 is serialized by the real generation checkpoint writer (including
  its standalone SQLite snapshot), reloaded into a fresh fixture instance and
  compared against uninterrupted steps 501--510, including causal cursor/state.
- Mutating source event lists or an exported checkpoint cannot mutate the
  recorder; missing, ambiguous and wrong-replay actions are rejected.
- Tiny final candidate lineage is complete and exactly replays, with candidate
  payload unchanged. The manifest expressly declines production parity claims.
- Existing trace-disabled CLI identity is unchanged when the flag is omitted.

These tests use no 7B model, real Mut oracle, production data or GPU. They must
not be represented as the real 500-step gate, full 50k parity or a method PASS.

## Remaining conditional closeout sequence

1. Keep the current A/B task untouched. Reopen its existing genuine scientific
   failure receipt before selecting any Route B. A/B PASS follows adoption.
2. On a new immutable commit, run the new causal observer versus the same
   trace-off producer using identical real Mut inputs, 1,448 train parents,
   RF/NeuroSED/config/seed/order, original 50k configured budget and 100k
   capacity. Stop the canary only at the existing safe diagnostic endpoint;
   compare 1--500 and reloaded 501--510. Bind the source/input/checkpoint chain.
   Old A/B receipts for another producer commit are not this proof.
3. Add explicit typed acceptance of that evidence in
   `validate_chemistry_trace_evidence`. Current Mut chemistry only accepts its
   historical parity/adoption contracts. Do not set `trace_parity_passed=true`
   in a causal summary to bypass this mismatch.
4. Bind the proven producer and opt-in flag into a **fresh** owner/spec. The
   existing `66487c...` owner execution pin remains unchanged in this patch;
   it would not activate the new producer. Do not launch a generation-only
   backup before its consumers and full resource admission are closed.
5. Freeze fresh generation payload + ordered candidate universe. Run new pair
   construction and exact DBSCAN, whose existing identity binds that manifest,
   payload, ordered hashes/indices, parent IDs and distance checkpoint. Do not
   import the historical pair/DBSCAN/cache when equality has not been proven.
6. Perform fixed deterministic chemistry, unified evaluation, gate and freeze.
   Bind a fresh typed Route-B terminal in the existing canonical publisher and
   matrix validator. Do not impersonate a historical trace-on adoption terminal
   or create another publisher/authority.

`draft_fresh_closeout_commands` reuses the actual shared five-stage builder and
only adapts its chemistry paths to `causal_lineage`. It forces CPU pair-building,
exact sklearn float64 DBSCAN and fresh outputs; it rejects historical external
pair/DBSCAN/vector-cache references and resume. It returns `dispatchable=false`
with the precise evidence/publication blockers, even if all command paths exist.
It does not create the missing preregistration or publication receipt.

## Resources

At the 2026-09-06 01:08 CST parent audit, free inodes were 95,191. This is 4,809
below the unchanged 100,000 guard. The known compact peak is now 2,569 new
inodes (1,564 pair files, checkpoint/retention/fixed artifacts and 489 causal
chunks plus twelve causal directory/fixed/atomic-temporary slots), so the minimum
gap preserving the guard is 7,378 **before** unmeasured evaluation/chemistry cache
peaks. No cleanup or
threshold lowering is performed. Causal event bytes, in-memory predecessor
state and mirrored checkpoint amplification require actual canary measurement.

The read-only CLI remains:

```bash
python scripts/autodl/preflight_mut_route_b_closeout_v1.py \
  --config configs/hpc.yaml \
  --resource-path /autodl-fs/data/counterfactual-subgraph-runtime
```

No active SQLite/WAL is opened by that preflight. Local tests create only their
own tiny database and immutable snapshot fixtures.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/baselines/comrecgc/test_mut_causal_lineage.py \
  tests/autodl/test_mut_route_b_closeout_preflight_v1.py
bash -n scripts/slurm/run_generation.sh
```
