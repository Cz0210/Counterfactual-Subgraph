# T13 formal launch / Mut recovery binding handoff

Snapshot: 2026-09-07 00:23:50 CST. Main authority remains **12/16**. Missing: Mutagenicity/ComRecGC and TasteMolNet/GlobalGCE, GCFExplainer, ComRecGC. GNN and L0 ablations do not change this count.

## Implementation and execution identities

Integration branch: `feat/early-gnn-first-ablation-20260905`. Integration includes T13 commits `eb232e4e`, `c0eb892d`, T12 future cache `a54be06a`, T14 bindings `38b73061`/`65a9ddca`, Mut bindings `1b8a1951`/`69c47be0`. These are control/repair identities, not replacements for historical science commits.

Focused integration validation: **122 passed**; affected compileall, shell syntax and diff checks passed. T13 deployed CPU/interface regression: **44 passed**. Previously passed matched-deterministic scientific canary was reused, not rerun. No model, optimizer, lazy dataset or scientific backend was silently changed.

## T13: one formal attempt actually started

Execution commit: `c0eb892dd13ef05a5891c4acf1c5f4fef3966f67`.

Execution worktree: `/root/autodl-tmp/worktrees/t13-deterministic-c0eb892d`.

Formal control root:

`/autodl-fs/data/counterfactual-subgraph-runtime/control/t13-lazy-repair-20260906/deterministic-formal-1eda4517-75a6-4ea4-a008-1cf68f868fe3`

This contains `deterministic_execution_contract.json`, narrow authorization, immutable spec bundle, dispatch command and owner receipts. Contract file SHA: `1cf8d27318298d8716e8259755a061df8609bd4301a9a952cda186e4b400ddda`. The contract's internal self-hash is a distinct field and is not interchangeable with file SHA.

Science output:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t13-deterministic/1eda4517-75a6-4ea4-a008-1cf68f868fe3`

Owner PID **348088**, start ticks **44532505**; science PID **348107**, start ticks **44532678**. GPU1 UUID `GPU-244f35a8-354a-ef1e-f589-bde7f8a7a690`. Canonical task `t13-from-hpc-7b647a43-1919-4d8e-a05a-0f6071255f2e` now owns the original GPU1 reservation; no reservation was cleared and no second lock was created.

The original quota ledger `control/t13-lazy-repair-20260906/full_start.json` records **1/1 consumed**. Never start a second fresh full. Resume only from a valid checkpoint if needed.

Actual science child wrote `owner/runtime_backend_receipt.json` before CUDA initialization: Torch2.7.1+cu118, CUBLAS`:4096:8`, deterministic true/warn_only false, cuDNN deterministic true/benchmark false, matmul TF32 false, **cuDNN TF32 true**. Actual thread counts56/56; OMP/MKL/OpenBLAS overrides absent. Original canary did not numerically log thread counts; this omission is disclosed rather than inventing evidence. Complete source/config bindings and recorded RNG/sampler evidence remain attached to the contract.

Admission passed using T13's own384GiB headroom,100GiB persistent and8192 free-inode floor **including4096 compact reserve**. It includes the measured prior process-tree peak43,507,916,800 bytes, not just single-PID RSS. At launch: headroom498,457,686,016 bytes,94615 free inodes. Live process-tree peak at00:23 was4,659,433,472 bytes; minimum observed cgroup headroom492,616,654,848 bytes; failcnt unchanged4306.

At00:26:51: phase `TARGET_0_RUNNING`, index preprocessing1676/3823 (index progress updated00:25:02); no optimizer checkpoint confirmed. This is a real formal process, **not yet evidence of an optimizer update or completed training**. Actual first checkpoint is `raw/target_0/globalgce_training_checkpoints/training_checkpoint.pt`, saved after epoch1's optimizer update. Original target0/2, seed7 and100epoch schedule remain unchanged. No mining or6.10GB transfer was repeated. Native nondeterministic failure and all diagnostics are preserved; diagnostic checkpoints are not promoted.

Existing executable chain remains training → rules → chemistry/GINE → calibration selector/freeze → test/export → verifier → original main publisher. The old owner was terminal; new registry row and reservation were updated by narrow identity-checked CAS. Registry SHA after CAS: `56ec94b3148efd4375425d84983e43e5b4d2c9fd24b60b312230b74101ea7576`.

## Mut: sealed, not running

Sealed root: `/autodl-fs/data/counterfactual-subgraph-runtime/control/mut-resource-recovery-306d9414-20260907`.

`binding.json` binds A replay/resume, B fresh sequential arm, post-AB decision, adoption/evaluation/publisher and the existing executor namespace. Scientific checkpoint250/next251 has events1..249; no jointly sealed positive-step boundary exists. Required replay is **1..250**, comparison against old scientific/RNG/registry/candidate state250 and existing event prefix, independent reload, then251..500 and required reload interval. Event250 is not synthesized. Replay/restore engineering failure does not trigger RouteB.

Activation implementation is deployed at `/root/autodl-tmp/worktrees/mut-recovery-activate-dc123b73`, commit `dc123b731fa322fb0e80c04a00fef17aa898e5dc`; binding implementation at `mut-recovery-binding-306d9414`. Agent focused validation87PASS. These are not production equivalence results.

Old executor222378/start28623524 remains alive, idle `WAITING_FOR_NEXT_ACTION`; old AB and postAB owners are dead. **No signal was sent.** Activation is default read-only; `--execute` requires resources, complete task peak plan, exact old identity/no-child/no-claim, existing lock and live registry CAS. No stale whole-registry overwrite is permitted.

Current blockers: inode guard/joint reserve, and missing complete `task_peak_reservations` evidence (`RESOURCE_PEAK_PLAN_INCOMPLETE`). Unknown peaks are not zero. Resource failure cannot launch RouteB. The historic conditional RouteB science adapter still has a `BLOCKED_ADAPTER_MISSING` gap; successful adoption route is bound, but do not claim unconditional 16/16 autonomy if a genuine future A/B scientific failure requires that adapter.

## T14: preserve current low-memory canary

Owner326543, live underlying science326689 (intermediate wrapper326591). Current low-memory continuous510 canary latest recorded step300/checkpoint250; original reference500 is preserved. Do not describe this as a new reference500 run. No retry3 or old42.6GB load.

Existing owner root:

`/autodl-fs/data/counterfactual-subgraph-runtime/control/t14_route_c/owners/route-c-59f101cd-f30b-458d-aa8c-2eb93ae82609`

Three formal roles are individually bound in `formal_execution_binding/binding.json`, with executable `dispatch.json`, task and continuation specs. Deployed successor `/root/autodl-tmp/worktrees/t14-formal-4e867c20`; binding SHA `b65ff3d551b1f2cf10be9a1a4ddf56c90657516d7ed28e9e6bdb9b903dcad029`. Science AST/config checks preserve original behavior, not mechanical HEAD replacement.

Current owner code is unchanged and not dynamically reloadable. **No duplicate waiting owner was started.** Once current stage terminal/parity and natural lock release satisfy the sealed guard, execute the actual dispatch command in that JSON and perform the existing unique-owner handoff. Until then formal is `SEALED_WAITING_CANARY_TERMINAL_AND_PARITY`. The current diagnostic-only510 checkpoint is not promoted. The original unstarted promotable branch preserves its own legal checkpoint/reload/promotion sequence.

## T12: reader remains untouched

Owner162844, science173495, confirmed reference step250. Phase `REFERENCE_RESUME_TO_500`; no full parity or formal fresh-zero dispatch yet.

Three permitted FD observations were taken, and no more are needed this turn. History242,191,887 bytes (not the5,343,635-byte checkpoint): offset227,679,231 →227,725,135 →227,760,399 between23:42:30 and23:44:03 CST; last remaining14,431,488 bytes. No backward jump in this sample window. 81,168-byte advance over93.4seconds is I/O evidence, **not experiment percentage or a reliable completion ETA**. Process stateD/read syscall and scientific step250 are reported separately.

Future-only immutable history cache implementation/test is integrated;21focused tests passed. Original record order/codec/first-seen bytes and state are preserved; one copy/content check, bounded sequential read, no repeated hot-cache SHA scan. It requires sealed/no-writer proof and local capacity before use. Current cache is not materialized and the active reader/worktree/path was not changed. Local NVMe was only1.8783GiB free at the mount audit; do not assume it can hold full-state outputs.

## GNN and L0: final scientific results remain PASS

GNN corrected registry: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/registry/gnn_result_registry.json`.

GNN paper root: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/seed7-corrected-20260905T144600Z/paper-20260906`.

L0 result: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/bace-l0-at-most-k-c6d5f0b5-20260906/import-68TBki/result`.

HPC2560832,2567257,2567271 are COMPLETED0:0. Old2560839FAILED remains historical. No GNN training/temperature/OT, L0 vocabulary/generation/evaluation, packaging or completed transfer was repeated.

L0 has15 eligible rules,386×8=3088 attempts,472 vocabulary entries; at-most-K applies, K16..20 do not enlarge the15-rule prefix. CCRCov@10/@20=0.19858156028368795,28/141 test parents, conditional medianWNode@10=0.004638647893874328. Missing metrics stayN/A, unfinished variantsPENDING, genuine zero stays0.

GNN claims remain single seed7, proposal-fixed backbone sensitivity; GCN is smaller, parent bootstrap is not cross-training-seed variation, GIN Brier's slight worsening is preserved. No scientific ranking claim is inferred from status.

Derived status now explicitly prioritizes final audit/publication over stale intermediate L0RUNNING; sealed progress is unchanged. No separate proven broken consumer was modified unnecessarily.

Paper summary with source CSVs: `/private/tmp/t13-formal-mut-rebind-20260907/paper-results-summary.md`. Existing AIDS/BACE PARTIAL outputs: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_staging/partial-aids-bace-fd98c5f2-20260905`. Final four-dataset Figure3/4/Table2 remain pending missing cells.

## LLM waiting owner and resource blockers

Owner325928 is alive at `/root/autodl-tmp/worktrees/llm-dispatch-3b2605d6`. Specs/queue: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-dispatch-3b2605d6-20260906`.

L1→L2→L3 are real sealed specs with tested FD/UUID/resource interfaces. L2 uses existing300update PPO-LoRA with matched native generation, not old plain-prompt pool adoption or retraining. No projectSFT/20B/newdimension.

State `WAITING_RESOURCE`; actual GPU smoke and generation have not run. No main-cell-count or secondary-seed gate. GPU0 remains Mut reservation, GPU1 now T13formal, GPU2 T14 andGPU3 T12. No borrowing, co-location or reservation clearing. Current quota, complete peak plan, main-owner health and legal unreserved1200second idle interval still apply. Existing owner re-reads real resource evidence; no second owner is needed.

## Platform action and remaining limitations

At00:23: mount `/autodl-fs/data`, source `AutoFS:fsnmafirst1132530`, `fuse.autofs`; client f_files200000/f_favail94581. Available bytes1,679,868,190,720. Mut guard deficit5419; plus fixed160 deficit5579; known joint110761 deficit16180; dynamic peaks UNKNOWN. This is below safety guard, not observed ENOSPC/EDQUOT.

Prepared request: `/private/tmp/t13-formal-mut-rebind-20260907/PLATFORM_INODE_REQUEST.md`. **User/platform action required:** ask AutoDL to verify and increase file-count quota online, no restart/format/remount/moving active processes, no automatic paid change.250000+ is an evaluation target, not guaranteed sufficiency. Request has not been sent; quota is not adjusted. No deletion was performed.

T13 can continue independently under its own valid resource contract. Mut cannot be called unattended-ready until quota and full task peak evidence pass. T14 future entry is executable/sealed but not yet a live successor; do not claim a duplicate waiter exists.

## Exact read-only status commands

Mac combined bounded status (does not load SQLite/WAL/checkpoints or sample T12FD):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 autodl-a800 '/root/miniconda3/envs/smiles_pip118/bin/python -I -B -' < /private/tmp/t13-formal-mut-rebind-20260907/status_autodl_readonly.py
```

Mut admission and identity status; deliberately no `--execute` while blocked:

```bash
ssh -o BatchMode=yes autodl-a800 '/root/miniconda3/envs/smiles_pip118/bin/python -I -B /root/autodl-tmp/worktrees/mut-recovery-activate-dc123b73/scripts/autodl/activate_mut_recovery_binding_v1.py --config configs/hpc.yaml --binding /autodl-fs/data/counterfactual-subgraph-runtime/control/mut-resource-recovery-306d9414-20260907/binding.json --owner-registry /autodl-fs/data/counterfactual-subgraph-runtime/control/final16-owner-registry/current.json --resource-config /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-dispatch-3b2605d6-20260906/resource_config.json'
```

HPC completed results, read-only:

```bash
ssh -o BatchMode=yes tongji-hpc 'sacct -X -j 2560832,2567257,2567271 --format=JobID,State,ExitCode,Elapsed -P'
```

No restart command is appropriate for healthy T13/T14/T12 or the LLM waiting owner. Formal T13quota is already consumed; never rerun its launcher as a status action.
