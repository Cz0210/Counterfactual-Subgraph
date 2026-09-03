# AutoDL main-16 and HPC T8 continuation handoff

Snapshot time: 2026-09-04 00:05 CST. This file records a live operational
snapshot, not a substitute for the status commands below.

## Authority and code identities

- The sole matrix authority remains
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority`.
- Its current state is 12/16. The missing cells are Mutagenicity/ComRecGC and
  TasteMolNet/GCFExplainer, GlobalGCE, and ComRecGC.
- The AutoDL task-spec execution worktree is
  `/root/autodl-tmp/worktrees/main-ready-task-spec-binding-5945f127`, pinned to
  `5945f12777cad71b75c2417a9cfb8e2b924eef73` and clean at deployment.
- The HPC science worktree remains
  `/share/home/u20526/czx/worktrees/t8-hpc-481475c3`, pinned to
  `481475c31d809577b791f4dd9002f5d2894c65b4`.
- The repaired HPC controller worktree is
  `/share/home/u20526/czx/worktrees/t8-continuation-ad7b0ffd`, pinned to
  `ad7b0ffd1861a765ff80139dedc6c4dcbf2fe766`.
- LLM and GNN ablation science is disabled.

## AutoDL main-ready task specs

The immutable bundle is:

`/autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T143321Z`

Its manifest file SHA-256 is
`fd9068cc88c533e4907f475db957e31088262d1ff327b62d2b1d6ea3b20b5fa5`
and its semantic self-hash is
`16dc3f4bf813338427c922d2ae87be0ea8cb13146d5cae8f1273d993d6a79072`.

The three sealed specs are:

- Mut: `mut-clean-equivalence-e37666ec.json`, spec hash
  `372a55511603218191b0625c4cab21c8fceb5bce870b3f3e8040f87e2eee2819`.
  It binds fresh trace-on and trace-off arms, `PYTHONHASHSEED=0`, source
  algorithm commit `7f7ed51a...`, instrumentation commit `66487c06...`, the
  correct semantic-finalizer commit `582bc4b4...`, and the identical
  historical/pair-store/DBSCAN candidate-universe hash `9c4d79ac...`.
- T14: `t14-checkpoint12500-audit-adf09ad9.json`, spec hash
  `e65c1db5e84b8bd20bc7404137a72088365d64845e3104da1b8e1b88be5b8da8`.
  It binds the sealed step-12,500 checkpoint digest `1c71ca73...`, the
  42,602,133,879-byte state, and a serial metadata/convergence audit before
  any resume.
- T12: `t12-reference-500-8de744db.json`, spec hash
  `8fca56666380eccb16724b411226e72de8b26074b8ce73ecda95bd410b4b545e`.
  It binds the 3,778-parent train-only cohort, T3 GINE, managed NeuroSED,
  threshold authority, exact replay gate, official GCF inventory, and fresh
  checkpoints at steps 250, 500, and 510.

The existing sidecar is still healthy at PID 141824, start ticks 15172670,
with command hash
`31f5374407456ce69c1c17dabf30935547f43024bbda5f1e9456dc0802a1b534`.
It did not acknowledge the new pointer after two native poll intervals, so
the hot-bind result is `HOT_BIND_UNSUPPORTED_FALLBACK_REQUIRED`. It was not
restarted.

After explicit user approval, the sealed one-shot binder completed with status
`PASS`. All three outcomes report `duplicate_started=false`, no failures, and
`OWNER_CONFIRMED`. The healthy existing sidecar was not restarted.

- Mut owner PID 161697 (start ticks 18437020) launched reviewed worker PID
  161700. Its heartbeat phase is `RUNNING_REVIEWED_MUT_ADOPTION_WORKER`; fresh
  trace-on/off arms are bound and neither pair-store nor DBSCAN is recomputed.
- T14 owner PID 161704 (start ticks 18437224) completed its no-deserialization
  audit and published `BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE`.  No science PID
  or GPU lock was created; Torch and SQLite/WAL were not opened.
- The original T12 owner PID 161714 reached terminal `FAILED_AT_250` with zero
  completed steps.  Its child had persisted exactly 511 uncommitted
  first-seen embeddings before a resource-bound bug rejected record 512.  The
  failed root is retained read-only and has no resumable checkpoint.

T12 has a fresh repaired successor.  Commit
`1ad12b560d3ad8533f47e3bc3fd1e6ee315a895a` changes only the first-seen count
gate from the live-cache bound to the existing cumulative scored-observation
bound; 11 focused tests pass locally and on AutoDL.  The sealed successor spec
is
`/autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T160132Z-t12-firstseenfix/t12-reference-500-firstseen-c6fdce9f.json`
(spec SHA
`9882a3ada23ad79fea0d727c534fdb3d181221ac174274cbdabf1834059873b5`).
Its owner PID is 162844 (start ticks 18577652) and science PID 162847 is
running on GPU3 from the fresh root
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/t12-production/reference-500-firstseen-c6fdce9f-7e01-4481-b97e-372ae851a0f8`.
At the final post-fix canary snapshot the child had remained healthy for 254
seconds, held 23,022 MiB VRAM, and its first-seen segment contained exactly
11,817 complete records (14,736,512 bytes).  It therefore passed both the old
511-record limit and the previous failure window.

## Main-line blockers

### Mutagenicity/ComRecGC

The old mixed-condition comparison is not adopted. The new spec uses fresh
A/B arms under one algorithm contract and separately checks resume parity.
The reviewed adoption worker is now running under the new owner. GPU0 was
still at zero VRAM in the immediate post-launch snapshot, so its current
phase is CPU/preflight or internal GPU admission; do not infer a failure from
that single early snapshot and do not create another owner.

### TasteMolNet/ComRecGC (T14)

The sealed step-12,500 checkpoint is intact.  The completed metadata/archive
audit confirms its hashes without opening the payload, SQLite, or Torch.  It
is a PyTorch zip with a 41,863,324,712-byte monolithic `data.pkl`, so safe
streaming restore is not proven.  Its historical safe restore
contract requires 549,943,914,496 bytes of cgroup headroom, greater than the
entire 515,396,075,520-byte cgroup limit. The owner therefore performs only a
metadata/archive audit and must emit the typed low-memory blocker without
loading Torch or active SQLite/WAL. This is a real resource blocker, not a
missing GPU. A science resume is forbidden until a lower-memory loader is
proven or the job moves to a cgroup with sufficient measured headroom.

### TasteMolNet/GCFExplainer (T12)

The previous 417-step root has no durable checkpoint and cannot be resumed.
The first fresh reference exposed a non-scientific capacity bug: the
510-step diagnostic profile reduced `max_full_live_records` to 511, and that
live-cache bound was incorrectly reused for cumulative first-seen embeddings.
The exact theoretical count bound is instead
`1 + 510 * 10,000 = 5,100,001`; the independent actual-byte disk cap remains
in force.  The repaired fresh successor is running on GPU3 and will commit
steps 250, 500, and reload 501--510. The measured historical speed is roughly
218 seconds per step, so reference step 250 is approximately 15 hours and
step 500 is approximately 30 hours. Do not poll it continuously after launch.

### TasteMolNet/GlobalGCE (T8/T13)

The protected AutoDL route remains alive at controller PID 82588 and science
PID 82680 on GPU1. It is doing positive single-core work in target-0 exact
mining but has no durable intermediate gSpan checkpoint. It has not been
signalled.

The original HPC stress job 2535373 ended in `TIMEOUT` after 3,627 seconds.
The first automatic refinement jobs 2535893/2535894 failed before script
execution because their relative `logs/` destination did not exist; they
produced no science artifacts. Commit `ad7b0ffd...` fixes every generated
refinement, follow-up, array, merge, and package job to use a pre-created
absolute log root.

The repaired continuation job 2536032 completed successfully and submitted:

- exact depth-4 refinement canary 2536033;
- `afterany:2536033` follow-up 2536034.

The selected canonical child remains
`r0000-prefix_subtree-1fbb8ef323886a1532f9`. Job 2536033 uses the pinned
science commit, 8 CPUs, 64 GiB, one hour, node-local scratch, and no GPU.
At the 23:43 CST read-only snapshot, 2536033 is still running on `cpui048`
with 8 CPUs and 64 GiB (14:20 elapsed of a one-hour limit). Every recorded
minute has positive progress: 144,587 pattern lines, 154,818 event lines,
about 398.6 MiB process-tree RSS, about 288.8 MiB node-local scratch, and only
15,805 persistent bytes. It requests no GPU and has no matrix-write ability.
There is not yet a terminal/PASS/FAILED artifact. Job 2536034 remains
dependency-pending on `afterany:2536033` and will automatically run the next
admission/submission decision. Leave this chain under Slurm ownership and do
not poll it continuously.

## Status commands

Mac connectivity:

```bash
ssh -O check tongji-hpc
ssh -o BatchMode=yes autodl-a800 true
lsof -nP -iTCP:7897 -sTCP:LISTEN
```

HPC continuation:

```bash
ssh tongji-hpc \
  "squeue -j 2536033,2536034 -o '%i|%T|%M|%R'; \
   sacct -j 2536033,2536034 \
     --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList -P"
```

AutoDL immutable specs and owner state:

```bash
ssh -o BatchMode=yes autodl-a800 '
cd /root/autodl-tmp/worktrees/main-ready-task-spec-binding-5945f127
/root/miniconda3/envs/smiles_pip118/bin/python -I -B \
  scripts/autodl/status_main_ready_task_specs.py \
  --task-spec /autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T143321Z/mut-clean-equivalence-e37666ec.json \
  --task-spec /autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T143321Z/t14-checkpoint12500-audit-adf09ad9.json \
  --task-spec /autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T143321Z/t12-reference-500-8de744db.json
'
```

Repaired T12 successor:

```bash
ssh -o BatchMode=yes autodl-a800 '
cd /root/autodl-tmp/worktrees/t12-firstseen-bound-1ad12b56
/root/miniconda3/envs/smiles_pip118/bin/python -I -B \
  scripts/autodl/status_main_ready_task_specs.py \
  --task-spec /autodl-fs/data/counterfactual-subgraph-runtime/control/main-ready-task-specs/20260903T160132Z-t12-firstseenfix/t12-reference-500-firstseen-c6fdce9f.json
'
```

Do not read active SQLite/WAL, restart the healthy sidecar or old T8, submit a
second HPC follow-up, start ablations, or write the matrix from HPC.
