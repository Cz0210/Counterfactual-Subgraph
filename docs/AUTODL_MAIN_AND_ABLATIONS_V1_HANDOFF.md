# AutoDL main-table and ablation handoff (v1)

Last live audit: 2026-09-03 14:49:01 CST.  The immutable evidence snapshot is
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/main_acceleration_and_ablations_20260903T064546Z`
(30 files plus `SHA256SUMS`).  Re-run the status commands below; the values in
this note are a handoff snapshot, not a substitute for the unique matrix
authority.

## Priority and authority

The immutable priority is `MAIN_16_OF_16 > LLM_ABLATION > GNN_ABLATION`.
The only matrix authority is:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json
```

At the audit time it reported 12/16.  The missing cells were
`Mutagenicity/ComRecGC`, `TasteMolNet/GCFExplainer`,
`TasteMolNet/GlobalGCE`, and `TasteMolNet/ComRecGC`.

## Live main owners and recovery gates

- At 14:49:01 CST, Mut continuation owner PID 139038, start ticks 14835914,
  had one live instrumented/B science child PID 141424 on GPU0.  It had reached
  step 425/500, adopted the complete A arm, restarted only B from step zero,
  and remained in the equivalence route.  Historical 50k generation, pair
  store, and DBSCAN were not being rebuilt.
- Taste GlobalGCE manager PID 82588 and science PID 82680 (start ticks 7319071)
  remain the sole seed-7/100-epoch recovery on GPU1.  Target 0 is still inside
  gSpan root 0/50; its heartbeat reported 5,235,200 frequent subgraphs at this
  audit.  The old worker remains protected.  A real-input exact sharding canary
  is separately bound to production fingerprint
  `ab67a7a92fe1bab62cd8be1ef29e0dc427b946de51e6b4acedf55302eb8391e3`;
  it is preliminary evidence only and cannot authorize replacement.
- The old Taste GCF T12 processes are no longer live.  The run failed at step
  417 before a durable generation checkpoint after one evicted identity's
  recomputed GINE embedding bytes/hash differed.  The probability difference
  was only `2.9802322387695312e-08`; NeuroSED drift was not established.
  Existing buffered-I/O code is fixture-level only.  A real 500+10 production
  parity route must bind first-seen embedding bytes into the bridge/checkpoint
  before any new full launch; GPU3 being physically idle is not permission to
  start an ablation.
- The old Taste ComRecGC T14 science is no longer live.  Its complete step
  12,500 checkpoint is hash-bound by resume spec
  `455f63cd3b4c1311cacf3f01dd81b8e8c03556f02d8a4def1145434d9209148e`.
  Historical admission needs 512.175 GiB, above the 480 GiB cgroup limit, so
  GPU2 remains reserved but science is fail-closed at
  `WAITING_T14_PARITY_CANARY`.  A real <=50-step save/reload parity and memory
  receipt is required; the heavyweight auditor PID 107645 was gracefully
  stopped under `SERIAL_ONLY`.

At the 14:49:01 CST snapshot, the repaired v1 sidecar successor ran from commit
`b65bc403` with PID 141824/start ticks 15172670 and state root
`control/main-and-ablations-v1-b65bc40`.  It adopted the live Mut owner,
reported missing T14/T8 evidence rather than inventing owners, and kept both
ablations blocked.  The superseded PID 109258 had no science child and was
stopped with SIGTERM after exact PID/start-ticks/cwd/command verification.  Do
not reuse any PID in this paragraph for a signal; re-read `/proc` and the audit
snapshot first.

## LLM proposer ablation

The actual BACE/Ours path is `BASE_PLUS_PPO_LORA`.  There is no independent
project SFT checkpoint.  The four core rows are:

```text
BRICS_FIXED
CHEMLLM_7B_OFF_THE_SHELF
CHEMLLM_7B_PPO_LORA_MAIN
CHEMLLM_2B_OFF_THE_SHELF
```

The 7B PPO row is adoption-only.  The scale claim is limited to 2B versus 7B
off-the-shelf under the same proposal budget and downstream evaluation.  The
matched-SFT study stays disabled and must be reported as not applicable to the
current main pipeline (`state=N/A`,
`reason=NO_INDEPENDENT_MATCHED_PROJECT_SFT_CHECKPOINT`).

The train-only BRICS vocabulary has 472 entries.  Its candidate pool and
shortfall receipt are already present under the BACE stage-v2 output root.
The 2B snapshot is pinned at revision
`215c0dbc89417a06bbc3bae43a3ad61e58f0a56e`; it contains 1,889,110,016
parameters by safetensors headers.  Its isolated audit failed closed before
import/load because the pinned remote tokenizer implements a filesystem write
in `save_vocabulary` (`tokenization_internlm2.py:167`, `open(..., "wb")`).
No actual-loaded-weight parameter claim was emitted.  The 2B science row is
therefore blocked; this does not block the three non-2B stage rows.

Early LLM science may use at most one GPU only after matrix >=13, Mut no longer
needs GPU0, every remaining main cell has a healthy owner, no main task is
waiting for a GPU, the target has runtime evidence and checkpoint/resume, and
one GPU has been idle for 1200 seconds.  A new main GPU waiter pauses the LLM
run at its next committed stage boundary.

## GNN proposal-fixed ablation

The five core rows are `gine,gin,gcn,gatv2,gatedgcn_plus`.  GatedGCN+ is a
project-specific molecular adaptation of edge-gated message passing, residual
FFN, normalization, dropout, and RWSE components pinned from
`LUOyk1999/GNNPlus` commit `0e02ad9a`.  Its five layers, hidden width 160,
dropout 0.2, RWSE length 16, and two-layer readout are parameter-matched project
choices, not an upstream BACE recipe.  The CPU dry-run counted 1,219,138
parameters versus the reloaded GINE's 1,432,583 (14.8993% difference); no
validation or test metric selected the width.  GraphGPS remains an optional
registered backbone and is not one of the five core rows.

GNN science remains blocked until matrix 16/16 and final matrix audit, Figure
3, Figure 4, and Table 2 receipts are all PASS.  Seed 7 runs first with at most
two GPUs; seeds 17 and 27 are extensions only when the measured per-model ETA
is no more than two hours.  Graph-Mamba is pinned metadata only and never runs
under this controller.

## Recovery task-spec and dispatch contract

The repaired sidecar no longer forwards its ambient shell to component
launchers.  Bind each component through its corresponding absolute JSON path:

```text
MUT_CONTINUATION_TASK_SPEC
T14_RESUME_TASK_SPEC
T8_ZERO_FINALIZER_TASK_SPEC
LLM_ABLATION_TASK_SPEC
GNN_ABLATION_TASK_SPEC
```

Every `main_and_ablations_task_spec_v1` object names the exact task ID/type,
immutable repository and commit, Python/config, manifest, input/output root,
GPU/CPU/memory request, complete `required_environment`, owner heartbeat, and
terminal receipt.  Fresh outputs and run IDs may contain `{attempt_uuid}` and
`{attempt_number}`.  If the science command contains those tokens, specify its
exact argv as `owner.command_argv`; the sidecar resolves it and hashes the raw
Linux `/proc/<pid>/cmdline` byte contract before launch.  A fixed live command
may instead provide `owner.command_sha256`.

Mut specs must include every variable required by
`launch_mut_throttled_continuation_v1.sh`, request GPU0/two CPU workers, and
bind `MUT_TRACE_OUTPUT_ROOT` to the task output.  T14 specs must include
`T14_AUDITOR_REPO_ROOT`, `T14_CHECKPOINT_ROOT`, `T14_RESUME_SPEC`, the complete
existing T14 full-run environment, physical GPU2, resume mode, and measured
cgroup headroom paths/requirement.  `serial_auditor.active=true` binds the
existing heavy relay heartbeat and start ticks; while that process is live the
science resume remains `BLOCKED_SERIAL_AUDITOR_ACTIVE`.

An attempt is not accepted because `launch.json` exists.  The sidecar waits 60
seconds for the true science owner and jointly verifies live PID, start ticks,
fresh heartbeat, output root/cwd, and command SHA.  Missing ownership produces
`FAILED_TO_START` and preserved evidence, then 60/120/300-second backoff.  The
third failed attempt becomes `BLOCKED_LAUNCHER_RETRY_EXHAUSTED`.  A strict
terminal receipt, a held GPU lease, any prior-output writer, or invalid owner
evidence prevents another launch.

## Status commands

For the live snapshot sidecar, use its recorded state root and execution
worktree.  For code-only LLM/GNN status, use the final immutable worktree at
commit `5b37ce0c81105cc6649213a0ddc60ca8d87025cd`:

```bash
MAIN_AND_ABLATIONS_STATE_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control/main-and-ablations-v1-b65bc40 \
  /root/autodl-tmp/worktrees/main-acceleration-and-ablations-b65bc40/scripts/autodl/launch_main_and_ablations_v1.sh status
/root/miniconda3/envs/smiles_pip118/bin/python \
  /root/autodl-tmp/worktrees/main-acceleration-and-ablations-5b37ce0/scripts/autodl/status_llm_ablation_core_v1.py --help
/root/miniconda3/envs/smiles_pip118/bin/python \
  /root/autodl-tmp/worktrees/main-acceleration-and-ablations-5b37ce0/scripts/autodl/status_gnn_five_backbone_ablation_v1.py --help
```

The active science PIDs must be checked by exact PID plus `/proc/<pid>/stat`
start ticks.  Do not infer ownership from fuzzy command matching, and do not
query the active T12/T14/T8 SQLite files.

The closeout sidecar may adopt the live T14 relay only with
`T14_AUDITOR_RELAY_HEARTBEAT` and its exact
`T14_AUDITOR_RELAY_START_TICKS`.  Taste GlobalGCE valid-zero publication stays
fail-closed until `TASTE_GLOBALGCE_ATTEMPT_RECEIPT` names the physical
authoritative receipt for the sole recovery attempt; do not fabricate that
receipt from an authorization file.
