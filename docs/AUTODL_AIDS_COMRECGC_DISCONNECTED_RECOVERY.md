# AIDS ComRecGC disconnected-anchor exact recovery

## Scope

This route recovers from the c766 production result
`anchor_epsilon_graph_disconnected` without converting that failed task into a
scientific PASS. It is CPU-only, uses a fresh controller CID/root, reads the
91,916,686-row pair/vector authority zero-copy, and does not replay the
completed seed/failure scan.

The source failure remains immutable. Thirteen small non-`FAILED.json`
evidence files are byte-checked into a fresh evidence archive; the active
selection manifest and checkpoint are rebuilt with fresh paths. The 25 GB
vectors and pair store are never copied or regenerated.

The fresh promotion claim and copied small arrays use deterministic two-name
hardlink publication. Resume rewrites only a validated temp-only prefix or
removes the extra name when final and temp are the same inode; any unrelated
temp inode blocks. This closes an interruption during the very first claim
write without replaying the seed/failure scan.

## Exactness contract

The recovery is valid only after all of the following reopen successfully:

1. every frozen seed lies in one exact initial anchor component;
2. the complete authenticated failure ledger contains every row lacking the
   seed lower bound, and every nonfailure row has an exact edge into that seed
   component;
3. every selected anchor is core with self counted under
   `distance <= 0.02` and `min_samples=3`;
4. deterministic primary bridge queries are scanned first, followed by the
   full all-anchor scan whenever primary queries do not close every component;
5. all certifying edges are rechecked directly in float64 with no tolerance;
6. the final core, connectivity, boundary, and partition certificates and all
   progress ledgers pass terminal replay.

If the unique-seed-component premise fails, the component shortcut is rejected
as `GENERAL_EXTERNAL_REQUIRED`. It is never generalized by assumption.

For a genuine multi-component all-core result, centroids and strict
`distance < 0.02` membership are streamed in frozen global row order. The
fixed-block Torch-float32 reduction is recorded as a `PROJECT_EXTENSION`, not
as bit-identical upstream `torch.mean`. Float64 results are audit values, and
any theta/radius decision disagreement blocks publication. Coverage, medoid,
and stable greedy selection are fully replayed; clusters are not duplicated to
fill `R=100`.

## Controller DAG

The immutable dependency order is:

```text
typed c766 failed-selection evidence (recovery-only)
  -> production first/random/dense/sparse/theta-boundary subset preflight
  -> exact component recovery and partition certificates
  -> streamed component downstream and numeric-boundary replay
  -> standardized continuation, WNode/export/freeze, final controller gate
```

The adoption receipt uses a dedicated `RECOVERY_ONLY_READY` type and
`RECOVERY_EVIDENCE_READY` marker. It creates no generic `PASS`; matrix, Mut,
and ordinary dependency readers must reject it. Only the final controller
terminal is ordinary-PASS eligible.

The resource identity freezes the selected 8--12 CPU threads (the authorized
fresh-route default is 8), hides CUDA, acquires no GPU lock,
and derives a formula-bound maximum new-output budget plus an 8 GiB safety
floor. The 96 GiB RSS contract is deliberately scoped to the exact DBSCAN
process: its native memory guard and terminal peak-RSS certificate close that
limit. It is not presented as a cgroup-enforced peak for the later short-lived
standardization subprocess tree. A first-checkpoint coexistence probe records
load, iowait, RSS, and progress before the long scan may continue. Its first host
sample and monotonic component-scan offset are persisted before `Popen` and
survive worker generations. A worker that finishes before PID binding or
within one block of a restart can close the probe only by binding the exact
terminal and DBSCAN SHA plus the DBSCAN-observed peak RSS.

Every typed stage gate freezes its validator projection and a recursive file
inventory. Small files are rehashed on every reopen; large arrays retain their
publish-time SHA and must keep the exact inode, size, mtime, ctime, mode, and
link count. This makes status/restart reject post-gate certificate, label,
summary, or freeze mutation without replaying the 91.9M-row downstream scan on
every dashboard poll. A missing final `PASS` can be reconstructed only after
all scientific stage validators, including the full downstream replay, reopen.

The controller root is claimed by a CID-and-manifest-SHA parent-side JSON file
created with `O_EXCL`, flocked, written with a fresh random attempt id and
nonce, and fsynced with its parent during initialization. The immutable owner
receipt binds its content SHA-256 plus complete inode/mode/owner/link/size/
mtime/ctime identity. The root, `gates/`, `logs/`, and owner receipt are then
idempotently finalized; same-CID `resume` reopens and verifies the exact claim
content and physical receipt. A zero/partial claim or any replacement/ABA is a
manual-diagnosis blocker rather than a repairable empty-prefix window. Output
usage is a hard controller-publication gate: exact net growth is reserved before
controller-owned state/gate/terminal/PASS writes, and usage is also checked
every 60 seconds while a worker is live. A live bound worker that crosses the
cap receives graceful `SIGTERM`, never `SIGKILL`, and no controller PASS may be
published over cap. This is deliberately not described as a filesystem quota:
a single science-subprocess write can cross the limit before the next poll, and
that failed evidence is retained. The formula includes at most one retained
1 GiB interrupted archive for each of the four non-common downstream stages,
for about 8.97 GiB of new output plus the 8 GiB free-space floor. This includes
all eight retained subset attempts, both names in a startup-record publication
crash window, and one byte-capped 1 GiB archive for each of four downstream
stages.

The bound is a publication/recovery contract, not unlimited retry storage.
One interrupted archive per non-common downstream stage is recoverable in the
same CID only when that archive is at most 1 GiB; a second interruption of the
same non-checkpointed stage, or an oversized partial, fails closed and requires
manual diagnosis plus a fresh CID. Append-only worker logs are polled rather
than filesystem-quota constrained, so an over-cap excursion may retain failed
evidence but can never publish controller PASS.

## Production commands

These commands are intentionally non-runnable until the reviewed adoption and
controller release commits are pinned and
`production_deployment_authorized=true` in the generated release spec.
Scientific paths are derived by code from the canonical adoption receipt; do
not hand-author them.

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
PROJECT=/root/autodl-tmp/worktrees/<immutable-recovery-execution>
ADOPTION=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery_evidence/aids_c766_failed_selection_v1/<fresh-child>
PARENT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/repairs
SPEC=/autodl-fs/data/counterfactual-subgraph-runtime/control/<fresh-cid>.spec.json
MANIFEST=/autodl-fs/data/counterfactual-subgraph-runtime/control/<fresh-cid>.manifest.json
EXEC_COMMIT=$(git -C "$PROJECT" rev-parse HEAD)
ADOPTION_COMMIT=7370006da6175851def0f151ca6fb4dfb44f2ab7
RUNNER_COMMIT=ab14be7c70803384eb6904d85bbf87b070d8d961

"$PY" "$PROJECT/scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py" \
  --config "$PROJECT/configs/hpc.yaml" generate-production \
  --adoption-output "$ADOPTION" \
  --controller-parent "$PARENT" \
  --python "$PY" \
  --project-root "$PROJECT" \
  --controller-manifest "$MANIFEST" \
  --thread-count 8 \
  --adoption-commit "$ADOPTION_COMMIT" \
  --controller-commit "$EXEC_COMMIT" \
  --exact-runner-commit "$RUNNER_COMMIT" \
  --subset-runner-commit "$RUNNER_COMMIT" \
  --downstream-runner-commit "$RUNNER_COMMIT" \
  --standardization-runner-commit "$RUNNER_COMMIT" \
  --authorize-production-deployment \
  --output "$SPEC"

"$PY" "$PROJECT/scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py" \
  --config "$PROJECT/configs/hpc.yaml" validate --spec "$SPEC"

"$PY" "$PROJECT/scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py" \
  --config "$PROJECT/configs/hpc.yaml" build \
  --spec "$SPEC" --output "$MANIFEST"

"$PROJECT/scripts/autodl/launch_aids_comrecgc_exact_recovery_v1.sh" \
  "$MANIFEST" fresh
```

Read-only status and same-CID restart are:

```bash
"$PY" "$PROJECT/scripts/autodl/status_aids_comrecgc_exact_recovery.py" \
  --config "$PROJECT/configs/hpc.yaml" --manifest "$MANIFEST"

"$PROJECT/scripts/autodl/launch_aids_comrecgc_exact_recovery_v1.sh" \
  "$MANIFEST" resume
```

The paired `scripts/slurm/` files exist for repository CLI parity but
deliberately exit before doing work. This recovery must not be submitted to
HPC.

## Current release status

- Science recovery commit `d8912ccb0901840ee1f0458ef66f630312024b0b` passed
  fresh detached review.
- Adoption-v3 superseding commit
  `7370006da6175851def0f151ca6fb4dfb44f2ab7` passed fresh detached review and
  is an actual integration ancestor.
- The adoption and combined controller integration pins remain intentionally
  unset pending fresh review of the merge commit.
- No fresh recovery controller has been deployed by this implementation.
- The c766 failed root and old brute route remain untouched.
