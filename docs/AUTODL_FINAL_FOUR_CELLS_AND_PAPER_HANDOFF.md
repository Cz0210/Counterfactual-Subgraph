# AutoDL final-four main-table handoff (2026-09-04)

## Scope and authority

The only active objective is the 4 methods x 4 datasets main table.  LLM and
GNN ablation science remain disabled.  AutoDL is the only matrix writer; HPC
may produce train-only GlobalGCE intermediate artifacts but cannot run the
GINE oracle, calibration, test evaluation, or matrix publication.

- Matrix pointer: `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Current registered count at the last audit: `12/16`
- Missing cells: Mutagenicity/ComRecGC, TasteMolNet/GCFExplainer,
  TasteMolNet/GlobalGCE, TasteMolNet/ComRecGC
- Read-only closeout observer state:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/final-four-cells-v1-d6ff6d26`
- Observer PID at the 2026-09-04 06:31 UTC audit: `192758`; it is
  `observer_only=true` and never writes the matrix.

Never replace the matrix pointer, create another matrix authority, or treat a
generation-only PASS as a method-cell PASS.

## Mutagenicity / ComRecGC

The old `2250` versus `2255` candidate comparison is a real scientific-state
difference at step 1, but it is not a valid trace-on/off causal experiment:
the roots used different commits and fresh/resume modes and both were
trace-enabled.  It therefore cannot authorize Route B.

- Bounded audit:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/mut_first_divergence_v1_20260904T050904Z`
- Fresh same-contract A/B spec:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/mut_same_contract_ab_task_spec_20260904T052257Z_7c9a0159.json`
- Fresh run root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_same_contract_trace_ab_v1_20260904T052257Z`
- Owner PID at launch: `193161`
- Post-A/B routing watcher control root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/mut_post_ab_continuation_v1_20260904T052257Z_7c9a0159`

The two trace modes run sequentially.  The watcher may publish only one of
`HISTORICAL_50K_ADOPTION`, `FRESH_TRACEOFF_ROUTE_B`, or
`ENGINEERING_REPAIR_REQUIRED`; it must not start Route B from the invalid old
comparison.  Route B must use a fresh root and must not reuse the old pair
store or DBSCAN when the candidate-universe hash differs.

At 2026-09-04 07:09 UTC the trace-on arm was healthy at step 57; trace-off had
not started.  PID `193180` remained the science process on GPU0.  This is a
long sequential A/B computation and must not be restarted merely because its
step rate is low.

## TasteMolNet / GCFExplainer (T12)

The live reference task must not be restarted or modified.  Its sealed
checkpoint-250 is usable as a common fork point, but the current compact
journal lacks per-step selected action, pre-softmax logits, and normalized
NeuroSED distance.  In addition, the 510-step diagnostic checkpoint identity
cannot be relabelled as a 20k production identity.

The accelerated preparation is therefore deliberately fail-closed:

- status: `BLOCKED_UNSUPPORTED_BY_CURRENT_STATE_SCHEMA`
- no accelerated full/postprocess/publisher task is dispatchable
- the Mut GPU0 release receipt is bound into the sealed spec and rechecked by
  the owner before output creation, lease acquisition, or Torch import
- endpoint agreement is recorded only as `ENDPOINT_STATE_MATCH`, never as
  per-step parity

Required future work is a checkpoint-250 shadow-reference ledger plus a
fresh-root, lossless journal/identity reframe for the production checkpoint
schedule.  Until that exists, keep the reference owner/science alive and rely
on its natural checkpoint-500 completion.

At 2026-09-04 06:34 UTC owner `162844` and science PID `173495` were still
alive.  The active segment SQLite had grown to `296812544` bytes with a fresh
mtime, so the process's D state represented continuing I/O rather than a dead
task.  Do not restart it.

## TasteMolNet / GlobalGCE (T8/T13)

HPC chain (do not resubmit completed shards):

- array: `2536781` (`0-15%4`)
- merge: `2536786`, dependency `afterok:2536781`
- package: `2536787`, dependency `afterok:2536786`
- canonical pointer:
  `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/control/t8-production-chain/current.json`
- science commit: `481475c31d809577b791f4dd9002f5d2894c65b4`

At the 2026-09-04 06:31 UTC audit the array had 12 completed, 4 running,
0 pending, and 0 failed shards (active shard IDs 3, 13, 14, and 15).  AutoDL's
old T8 remains an untouched fallback.
Only after the package independently passes may AutoDL import it into a fresh
root and run chemical validation, calibrated GINE inference, calibration,
held-out test evaluation, T13 closure, and matrix publication.  Do not stop
the old AutoDL route before independent import PASS.

The original storage admission is invalid: completed shard events and
patterns already occupy at least `33061713744` bytes, whereas it projected
only `225136431` bytes.  The original merge would persist another estimated
42--50 GB and the original uncompressed package another 42--50 GB, while the
user filesystem had only `6454157312` bytes free at the 07:08 UTC audit.  Jobs `2536786` and `2536787`
were therefore placed in reversible `JobHeldUser` state before the array
finished.  Do not release them.  They must be replaced by a hash-bound,
node-local exact merge that persists only a deterministic losslessly
compressed bundle and manifest.  The array itself was not held or changed.

The unused `/share/home/u20526/czx/conda_pkgs_gcfexplainer` package cache is
being streamed to
`/Volumes/DireRaven/counterfactual-hpc-offload/hpc-archive-20260904T070000Z/`.
Do not remove its HPC source until the external `.tar.zst` is finalized and
the decompressed raw-tar SHA matches the recorded SHA.

## TasteMolNet / ComRecGC (T14)

Route A and Route B are terminally unsuitable.  Route C is a fresh,
dataset-specific disk-backed implementation using an append-only graph store,
stable numeric graph IDs, mmap-backed candidate state, compact transitions,
bounded graph LRU, and one scientific state updater.  It must never load the
42.6 GB legacy step-12500 state.

The launch gate requires all of the following:

1. exact reference-versus-low-memory steps 1-500 parity over the full
   scientific projection;
2. low-memory continuous versus checkpoint-250/reload parity through step 500
   and exact steps 501-510;
3. crash recovery that rolls uncommitted disk-backed suffixes back to the last
   promoted boundary before resuming the same root;
4. a promotable low-memory step-500 checkpoint;
5. the frozen train-side convergence gate from step 10000 onward;
6. a persisted postprocess/final-verifier/matrix-publication continuation;
7. GPU2 ownership, no live T14 owner, no same-root writer, and no ablation
   process.

The historical scientific inputs are:

- T2 receipt root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-main-v2/adoptions/T2_GINE/4495ac87-1396-43f3-bb68-1ee0b1053d09`
- T3 calibrated GINE:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed`
- T4 oracle smoke:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/t4-oracle-smoke-03dd2e64-ab20-4df7-b514-fdafaeed0e52`
- train split:
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/prepared/16af8ead8a17b6bd3941d9eb5879c5be75c14114/splits/train.csv`
- official ComRecGC root:
  `/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/vendor/COMRECGC/122f9341a360e9f06bb58a2f5823bb596021f6bf`

The old root is read-only and forbidden as a Route C input:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/comrecgc/t14-full/attempt-f3b2e5f2-9f20-4c12-bd26-3d7cc8e0d9ab`.

Route C was deployed from immutable commit
`fd2a4860fcb6d04ee121d4d7b9ad32698747a2b2` in the detached, clean AutoDL
worktree `/root/autodl-tmp/worktrees/t14-route-c-fd2a4860`.  The deployment
uses fresh attempt UUID `c088bb48-32cd-4c83-a073-81349987ea27`, output root
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/comrecgc_route_c/route-c-c088bb48-32cd-4c83-a073-81349987ea27`, and task spec
`/autodl-fs/data/counterfactual-subgraph-runtime/control/t14_route_c/owners/route-c-c088bb48-32cd-4c83-a073-81349987ea27/T14_ROUTE_C_TASK_SPEC.json`.

At the single post-launch audit the owner was PID `196830` (start ticks
`23916755`), its heartbeat was `SCIENCE_RUNNING` in stage `reference-500`,
the managed GPU lock child was PID `196901`, and the Python science child was
PID `196909` on physical GPU2 UUID
`GPU-901b50ea-30b2-4a0c-505f-bf94980e1484`.  This proves that the fresh
Route C owner and parity gate are running; it is not a parity PASS or a method
cell PASS.  Do not launch a second owner.  Local integrated focused tests were
`146 passed`; the immutable AutoDL worktree's Route C suite was `62 passed`.

After generation freezes, the sealed continuation binds the existing unique
T14 publisher queue at
`/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16-matrix-publisher-952a80c-taste3-v2-20260901T145221Z/queue.json`
and the existing publisher PID `17420`.  It never creates another publisher
or writes the matrix directly.

## Final rendering and claims

The frozen renderer uses Figure 3 K=1..20, the exact 601-point Figure 4 grid
from 0 through 0.0535, and Table 2 at K=10/theta=0.05.  Undefined cost remains
N/A when coverage is zero; it is never imputed as zero.

The comparison audit is fail-closed.  Six legacy adopted AIDS/Mutagenicity
Ours/GCFExplainer/GlobalGCE artifacts currently expose placeholder
`parent_best_distances.csv` rows and omit the dataset/oracle/split/MolCLR
hashes needed for a same-parent paired bootstrap.  This does not invalidate
their already-authorized numerical cells, but it blocks a universal/fairness
superiority claim until parent-level provenance is reconstructed.  Do not
weaken the audit or claim that Ours wins unless the final evidence supports it.

## Status commands

```bash
# Mac (read-only; SSH ControlPath is resolved from the user's config)
bash scripts/local/status_hpc_autodl_offload.sh

# HPC
ssh tongji-hpc 'squeue -j 2536781,2536786,2536787; sacct -j 2536781,2536786,2536787 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P'

# AutoDL
ssh autodl-a800 '/root/miniconda3/envs/smiles_pip118/bin/python -I -B /root/autodl-tmp/worktrees/final-four-d6ff6d26/scripts/autodl/status_final_four_cells_v1.py --config /root/autodl-tmp/worktrees/final-four-d6ff6d26/configs/hpc.yaml --set inference.fallback_to_heuristic=false --state-root /autodl-fs/data/counterfactual-subgraph-runtime/control/final-four-cells-v1-d6ff6d26'
```

If a task enters an hour-scale healthy computation after its checkpoint and
continuation owner are present, stop interactive polling and let the owner
continue unattended.
