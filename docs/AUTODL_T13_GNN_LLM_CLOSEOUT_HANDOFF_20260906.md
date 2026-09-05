# T13 recovery / corrected GNN / LLM owner closeout — 2026-09-06

## Verified outcomes and current limits

- Main matrix: 12/16 at 2026-09-06 02:01 CST. Missing: Mutagenicity/ComRecGC,
  TasteMolNet/GCFExplainer, TasteMolNet/GlobalGCE, TasteMolNet/ComRecGC.
- Corrected five-model GNN seed7 passed independent portable verification,
  completed the existing Mac relay/import, and was published to the independent
  GNN ablation registry. No fit, model training, residual inference, selector or
  OT computation was repeated this turn. Preserve the old provisional package.
- GNN package SHA: `10f7a32e9bcb95c529d52cf8b3c442085dd16fd98d1355178866186a3d28e579`.
  Portable verifier job 2560832 COMPLETED 0:0. Relay 82454 finished normally;
  do not restart it or transfer the 6 GB T8 package.
- L0 job 2560839 FAILED 75:0 after completing all 386×8 train attempts. It
  produced 15 valid unique rules, below its current required 20. This is not
  an engineering retry. No test was loaded and no candidates were padded.
  The user has been asked whether to authorize effective K=15 and a plateau
  above 15 without changing attempts/selector/test. No change is authorized
  merely by this handoff; preserve the failed result and completed train ledger.
- T13 old attempt ended with return code -9 and no training checkpoint. This
  proves process termination, not PID-specific OOM. Old failure evidence remains
  at its original location and in the new narrow failure audit.
- The new T13 owner is running the real sequential two-target canary, **not yet
  formal training**. It owns the existing GPU1 lease; only all parity and memory
  checks can consume the single full-start claim. Do not start a second owner.
- A new main-table blocker appeared during this turn: Mut stopped at committed
  step 250 due to `free_inodes_below_limit`, not A/B semantic divergence or OOM.
  Owner 193161, science 193180 and post-AB continuation 193450 are now absent.
  Step250 checkpoint and mirror exist. Trace-off has not started. GPU0 remains
  reserved for Mut recovery; it is not an LLM idle GPU. Never trigger Route B
  from this storage/engineering stop. Observer last row249 versus committed250
  requires checkpoint/log reconciliation before an exact resume claim. Both
  compact57,628,093-byte checkpoint manifests say completed250/next251. The
  observer calls the storage-guarded checkpoint boundary before writing row250;
  its current continuous resume resets history, while its reload path requires
  an existing continuous500. Thus old `--resume` / `--phase reload` is not a
  valid complete-proof continuation. Repair the failure-safe observer boundary
  and segment resume, then deterministically recover/replay the missing250
  evidence; never fabricate that row or skip it. No249 checkpoint was found.
- T12 owner162844/science173495 and T14 owner268102/science268321 retain their
  identities. T12 remains diagnostic reference with sealed250, no full parity;
  T14 retry2 reference353/500 at snapshot. Neither was restarted or signalled.

## Immutable code and tests

- T13 execution: `708dd59f3ab476169df4716473005655041118ec`, immutable worktree
  `/root/autodl-tmp/worktrees/t13-lazy-708dd59f`.
- Integration / LLM / conditional Mut: `ec22f552`, branch
  `feat/early-gnn-first-ablation-20260905`, immutable AutoDL worktree
  `/root/autodl-tmp/worktrees/llm-mut-ec22f552`.
- Corrected GNN paper publisher: `b5e5d8d8`, deployed separately. Existing
  execution commits were not modified. Private branch fast-forward pushed;
  AutoDL received small incremental source bundles because its origin still
  points to an old local bundle. No weights or data were committed.
- T13 integrated relevant tests: 65 PASS using the existing pinned official
  source via GLOBALGCE_OFFICIAL_ROOT. LLM+Mut changed tests: 38 PASS. Agent
  broader regressions: LLM30 PASS and Mut49 PASS. Counts overlap, not additive.
  Compilation, shell syntax and diff checks passed.
- CPU/tiny-fixture tests are not production parity or GPU smoke. T13 short
  canary does not cover a full 100-epoch trajectory, epoch>=35 objective switch,
  or all validation events. Formal epoch-boundary checkpoints bind model,
  optimizer, scheduler, RNG, index/mask identity and sampler next_epoch.

## T13 actual owner and one-start contract

Control root:
`/autodl-fs/data/counterfactual-subgraph-runtime/control/t13-lazy-repair-20260906`

Narrow authorization: `authorization.json`; failure audit: `failure_audit.json`;
deployment: `deployment.json`. Owner PID294806/startticks36489077, canary
PID294825/startticks36489166 at snapshot. Owner path:
`/autodl-fs/data/counterfactual-subgraph-runtime/control/t13-lazy-repair-20260906/owner-46356adf-15b9-49fb-a2a4-7b12ac9112c8`.

Fresh formal output (reserved, not yet evidence of formal start):
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t13-indexed/46356adf-15b9-49fb-a2a4-7b12ac9112c8`.

Same imported20patterns, same train/GINE/targets0,2/seed7/original configured100
epochs and decoder are retained. Indexed augmentation retains exact masks,
sample order and multiplicity; only batch-local tensors materialize. The first
formal checkpoint follows the first original optimizer update (upstream groups
up to five batches per update); no microbatch semantics were changed.

Gate: 384 GiB starting cgroup headroom, 192 GiB protected-main reserve, 96 GiB
canary RSS ceiling, and full headroom also covering twice observed canary peak.
Boundary probes cover assembly/forward/backward/save/reload and transient
failcnt/headroom. This does not lower Mut's separate100000-inode guard.
Formal `full_start.json` is absent until successful canary admission. Once
consumed, no second fresh full start is allowed; checkpoint-specific recovery
must be separately validated if that attempt fails.

## LLM existing interface; GPU science has not started

FDs are actually inherited via pass_fds with independent lock-contention tests,
owner/child generations, UUID-to-cuda:0 mapping and current resource sampling.
No new locks/controller/trust platform and no GPU borrowing were added.

Fresh CPU preparation:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-ec22f552-20260906`.
The `gnn_owner_acceptance.json` binds the accepted corrected GNN receipt without
repeated unpacking. `execution_gate_status.json` truthfully says GPU smoke not
run, no GPU science, and no dispatch-spec resource evidence invented.

L1/L2/L3 native task specs are real. L2 retains the300-update PPO-LoRA, needs
matched native-prompt regeneration, and must not adopt the old plain prompt
pool. L3 retains the prior isolated CPU4-token proof, not a GPU smoke claim.

Remaining launch requirements: real GPU smoke/save/resume, complete current
main READY/publisher sources (including Mut's missing source), unreserved idle
GPU for1200s, measured resource admission, and a fresh sealed dispatch through
the existing owner. Its one-shot entrypoint is implemented; no live outer queue
continuously dispatching all three variants was demonstrated. Do not describe
configuration preparation as unattended LLM execution.

## Conditional Mut producer and resources

The actual selected-action observational producer is implemented/tested and
deployed only in the new isolated code. Existing active science was not patched.
It records compact immutable selected-action events without new RNG calls.
See `docs/MUT_OBSERVATIONAL_CAUSAL_LINEAGE_REPAIR.md`.

Not fully dispatchable: real production500/510+reload proof for this producer,
typed chemistry acceptance, fresh execution pin and Route-B terminal/publisher
binding remain. Do not substitute old trace parity or fixture tests. Pair-store
and DBSCAN must match the actual new universe; no recomputation was launched.

At02:01 CST: persistent free bytes1683968868352, free inodes95107. The guard is
still100000. Conditional RouteB known peak2569 additional inodes implies at
least7462 missing inodes before unmeasured chemistry/evaluation caches and other
reservations. Current Mut recovery already requires restoring its own guard;
do not wait for a hypothetical RouteB to address it. No files were deleted or
archived this turn. Scientific results and checkpoints remain protected.

The bounded read-only temporary-file inventory found no sufficient safe new
cleanup target. Conventional tmp/scratch roots were absent; transfers hold
current inputs; code_bundles has only three bundles. The old BACE cache's65724
inodes were already released on2026-08-31; the remaining tar/manifests cannot
release them again. At02:04 CST free inodes95113, current guard deficit4887;
with RouteB's known2569 peak, minimum7456 before other reservations. User action:
increase inode quota or identify additional regenerable terminal cache for
scoped review. Engineering recovery of the observer is already authorized.
Diagnosis: `/private/tmp/gnn-temp-repair-main-audit-20260905/mut-exit-diagnosis-20260905T180429Z.json`.

T13 remained healthy at02:09:43 CST: observed process-tree RSS peak4656508928
bytes, minimum measured headroom498435874816 bytes; formal_start still absent.
These are interim measurements, not final memory or parity PASS.

## Paper artifacts and exact read-only commands

GNN AutoDL paper tables:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/seed7-corrected-20260905T144600Z/paper-20260906`.
Mac small-table copy:
`/private/tmp/t13-gnn-llm-closeout-20260906/paper-gnn-corrected`.
Registry:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/registry/gnn_result_registry.json`.

Existing AIDS/BACE PARTIAL Figure3/4/Table2:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_staging/partial-aids-bace-fd98c5f2-20260905`.
Missing main cells remain PENDING; no final16/16 figures and no complete LLM
ablation table are claimed. Seed17/27 remain deferred, not an LLM prerequisite.

```bash
/Users/cz0210/miniconda3/envs/smiles_local/bin/python /private/tmp/status_t13_gnn_llm_closeout_20260906.py
ssh tongji-hpc 'sacct -X -j 2560832,2560839 --format=JobID,State,ExitCode,Elapsed -P'
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/t13-lazy-repair-20260906/owner-46356adf-15b9-49fb-a2a4-7b12ac9112c8/heartbeat.json'
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-ec22f552-20260906/execution_gate_status.json'
ssh autodl-a800 'cd /root/autodl-tmp/worktrees/llm-mut-ec22f552 && /root/miniconda3/envs/smiles_pip118/bin/python -I -B scripts/autodl/preflight_mut_route_b_closeout_v1.py --config configs/hpc.yaml --resource-path /autodl-fs/data/counterfactual-subgraph-runtime'
```

Do not use a T13 launch command again while owner294806 is healthy. Do not
re-run verifier2560832 or L0's completed train verification to fill markers.
Prioritize safe Mut storage/checkpoint recovery and the running T13 canary.
