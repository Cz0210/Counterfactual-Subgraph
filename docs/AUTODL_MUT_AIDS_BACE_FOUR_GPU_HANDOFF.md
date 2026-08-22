# AutoDL Mutagenicity / AIDS / BACE four-GPU handoff

Date: 2026-08-22  
Scope: AutoDL only; no HPC GPU or CPU; TasteMolNet heavy work disabled; `paper/`
frozen.  This document is a deployment handoff, not a claim that every
scientific stage has already passed.

The authoritative controller is now the v2 deployment
`autodl-four-gpu-recovery-20260822T044445Z-v2`.  Its frozen 22-task manifest is
read-only at:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json
```

The active v2 controller is a persistent `nohup` process with PID `120431` and
kernel start ticks `680917595`.  At this handoff it has adopted the five exact
historical runs, retained Mutagenicity `PASS`, continued monitoring AIDS, and
launched a fresh final-commit B6-v2 on GPU 0.  B7 and B8--B14 are correctly
waiting for the formal B6-v2 gate; no downstream PASS is claimed here.

## 1. Baseline and final development commit

- Frozen starting authority: `8b17fb1096666852b0680f899073dd82f207cce1`.
- Manifest/handoff integration base:
  `2dc911168ee647c1cd19a16efe07d58d7ab0f1d3`.
- Deterministic canary repair:
  `a6258413619fe2f762980c7172ed20a9917a0e2f`.
- Isolated v2 controller roots:
  `3abb9bb2440292938b9cd822c92d4f5d65874b7e`.
- Final deployed science/controller commit:
  `0ad149420577c683baa2ef03f78f70ee6841f3a1`.
- Final immutable controller/BACE execution worktree:
  `/root/autodl-tmp/worktrees/run-four-gpu-recovery-0ad149420577`.

Do not edit that execution worktree after launching scientific tasks.  Continue
later development in a separate worktree.

## 2. Common lineage repair commits

- Primary resolver repair:
  `6ddd74339dbd9b1f0e57ba341ae4529cc2864fce`
  (`fix: resolve COMRECGC lineage by global graph identity`).
- Fail-closed audit persistence follow-up:
  `1c889b971d290f7d8425fca24085dc8a212387f9`
  (`fix: persist COMRECGC freeze validation failures`).

The repair treats canonical/global graph hash as content identity, parent
metadata as provenance, and the pinned upstream plus one explicit edit plus
downstream hash as transition identity.  It does not bypass uniqueness,
single-edit, replay, collision, or parent-chain gates.

## 3. BACE clean initialization and GNN-PPO commits

- Clean initializer and real stable-loop GNN reward adapter:
  `049ecaea026ec34567b5defb3644a884084e6188`.
- Complete B8--B14 Frozen-GNN downstream route:
  `4a511fe7dc29603476071a738d796c33a29b2183`.
- Deterministic connected-deletion canary repair:
  `a6258413619fe2f762980c7172ed20a9917a0e2f`.

The old diagnostic B6 root remains immutable `BLOCKED`; none of its six
diagnostic flips is PPO success evidence.

## 4. Four-GPU controller commits

- Persistent dependency-aware controller:
  `7ad80f91f8a494856024fdfccfbbe84bf8d17a64`.
- Passing-attempt tokens, fixed shards, B11/B13 joins, and post-B12 test edge:
  `2dc911168ee647c1cd19a16efe07d58d7ab0f1d3`.
- Fresh v2 output/cache namespace isolation:
  `3abb9bb2440292938b9cd822c92d4f5d65874b7e`.
- Exact allowlist for the controller-owned
  `TOKENIZERS_PARALLELISM=false` scheduling key:
  `0ad149420577c683baa2ef03f78f70ee6841f3a1`.

The controller delegates workers to `exp_run`, uses UUID locks, samples idle
GPUs for at least 60 seconds, permits at most one OOM down-batch retry, never
retries semantic failures, and keeps TasteMolNet and paper blocked.  The final
environment repair allows only the exact safe scheduling key above; token,
secret, password, authorization, and API-key-like environment names remain
rejected.

## 5. Exact implementation diff file list

`6ddd743`:

```text
M docs/decisions.md
M docs/refactor_plan.md
A scripts/autodl/run_comrecgc_fresh_recovery.sh
M scripts/baselines/comrecgc/recover_completed_generation_freeze.py
M scripts/slurm/recover_completed_generation_freeze.sh
M src/baselines/comrecgc/freeze_recovery.py
M src/baselines/comrecgc/graph_trace.py
M tests/baselines/comrecgc/test_freeze_only_recovery.py
A tests/fixtures/comrecgc_lineage/aids_move_37600.json
A tests/fixtures/comrecgc_lineage/mutagenicity_recovery_counts.json
A tests/test_aids_lineage_historical_replay.py
A tests/test_mut_lineage_historical_replay.py
```

`1c889b9`:

```text
M docs/decisions.md
M scripts/baselines/comrecgc/recover_completed_generation_freeze.py
M src/baselines/comrecgc/freeze_recovery.py
M tests/baselines/comrecgc/test_freeze_only_recovery.py
M tests/fixtures/comrecgc_lineage/mutagenicity_recovery_counts.json
M tests/test_mut_lineage_historical_replay.py
```

`049ecae`:

```text
M docs/BACE_TASTEMOLNET_GNN_AUTODL.md
M docs/decisions.md
M docs/refactor_plan.md
A scripts/audit_bace_policy_initializers.py
A scripts/autodl/run_bace_gnn_ppo_stage.sh
A scripts/build_bace_clean_policy_initializer.py
A scripts/check_bace_gnn_downstream_release.py
A scripts/train_bace_gnn_ppo.py
A src/rewards/gnn_ppo_reward.py
A src/train/bace_gnn_ppo.py
A src/train/bace_policy_init.py
A src/train/bace_stage_boundaries.py
A tests/test_bace_b6_v2_contract.py
A tests/test_bace_clean_policy_provenance.py
A tests/test_bace_gnn_ppo_reward.py
A tests/test_no_test_before_selector_freeze.py
```

`4a511fe`:

```text
M README.md
A configs/autodl/bace_frozen_gnn_downstream_tasks.json
A docs/AUTODL_BACE_FROZEN_GNN_DOWNSTREAM.md
M docs/decisions.md
M docs/refactor_plan.md
A scripts/autodl/bace_frozen_gnn_downstream.py
A scripts/autodl/run_bace_frozen_gnn_downstream.sh
A scripts/slurm/bace_frozen_gnn_downstream.sh
A src/eval/bace_frozen_gnn_contracts.py
A src/eval/bace_frozen_gnn_pool.py
A src/eval/bace_frozen_gnn_prep.py
A src/eval/bace_frozen_gnn_selection.py
A src/eval/bace_frozen_gnn_verification.py
A tests/autodl/test_bace_frozen_gnn_downstream.py
A tests/test_bace_downstream_no_test_before_selector_freeze.py
A tests/test_fixed_parent_shards.py
```

`7ad80f9`:

```text
A configs/autodl/four_gpu_recovery.template.json
A docs/AUTODL_FOUR_GPU_RECOVERY.md
M docs/decisions.md
M docs/refactor_plan.md
M scripts/autodl/common.sh
M scripts/autodl/exp_run.py
A scripts/autodl/launch_four_gpu_recovery.sh
A scripts/autodl/run_four_gpu_recovery_controller.py
A scripts/autodl/status_four_gpu_recovery.py
A scripts/slurm/run_four_gpu_recovery_controller.sh
A scripts/slurm/status_four_gpu_recovery.sh
M src/utils/autodl_runtime.py
M tests/README.md
A tests/autodl/test_four_gpu_recovery_controller.py
M tests/autodl/test_gnn_runner_contract.py
M tests/autodl/test_gnn_runtime.py
```

`2dc9111`:

```text
M configs/autodl/bace_frozen_gnn_downstream_tasks.json
M configs/autodl/four_gpu_recovery.template.json
M docs/AUTODL_BACE_FROZEN_GNN_DOWNSTREAM.md
M docs/decisions.md
M docs/refactor_plan.md
M scripts/autodl/run_four_gpu_recovery_controller.py
M tests/autodl/test_bace_frozen_gnn_downstream.py
M tests/autodl/test_four_gpu_recovery_controller.py
```

`a625841`:

```text
M docs/BACE_TASTEMOLNET_GNN_AUTODL.md
M docs/decisions.md
M docs/refactor_plan.md
M scripts/train_bace_gnn_ppo.py
M src/train/bace_gnn_ppo.py
M tests/test_bace_b6_v2_contract.py
M tests/test_bace_gnn_ppo_reward.py
```

`3abb9bb`:

```text
M configs/autodl/four_gpu_recovery.live_candidate.json
M tests/autodl/test_four_gpu_recovery_controller.py
```

`0ad1494`:

```text
M docs/decisions.md
M docs/refactor_plan.md
M scripts/autodl/exp_run.py
M tests/autodl/test_exp_run.py
```

The deployed manifest is the exact frozen copy named at the top of this
handoff.  Do not edit it in place; a future semantic manifest change requires
a new controller ID and fresh output/cache namespace.

## 6. MUT/AIDS historical replay tests

The focused lineage suite covers the frozen Mutagenicity and AIDS historical
payloads, global graph identity, representative-parent mismatch, pinned
upstream identity, exactly-one-edit, true ambiguity, collision/corruption,
legacy unique payload, and deterministic round trip.  The release evidence for
the `1c889b9` follow-up records 15 focused tests passing.

Expected positive test markers are:

```text
[COMMON_LINEAGE_FIX_TEST_PASS]
[MUT_HISTORICAL_LINEAGE_REPLAY_PASS]
[AIDS_HISTORICAL_LINEAGE_REPLAY_PASS]
```

The focused release suite in the final immutable AutoDL worktree completed with
`65 passed`.  That includes the historical replay, controller, BACE gate,
leakage, environment-key, and downstream contract checks.  Static/controller
tests are engineering evidence; the stage states below still come only from
their scientific gates.

## 7. Mutagenicity fresh recovery

- Adopted run ID: `20260822T025620Z-mut-lineage-v3-6ddd743`.
- Immutable execution worktree:
  `/root/autodl-tmp/worktrees/run-mut-lineage-6ddd743`.
- Worktree commit: `6ddd74339dbd9b1f0e57ba341ae4529cc2864fce`.
- Fresh output root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z`.
- Generation source:
  `/autodl-fs/data/runs/autodl_three_lines_20260821_v1/inputs/mut_generation`.
- Expected source commit:
  `7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4b`.
- Controller state at this handoff: `PASS` at `2026-08-22T04:36:05Z`; no MUT
  worker remains and its GPU was released.

The failed v2 fresh root remains immutable failure evidence and is not reused.

## 8. AIDS fresh recovery

- Adopted run ID: `20260822T020238Z-aids-lineage-v2-6ddd743`.
- Immutable execution worktree:
  `/root/autodl-tmp/worktrees/run-aids-lineage-6ddd743`.
- Worktree commit: `6ddd74339dbd9b1f0e57ba341ae4529cc2864fce`.
- Fresh output root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/aids_comrecgc_lineage_v2_20260822T020238Z`.
- Generation source:
  `/autodl-fs/data/runs/autodl_three_lines_20260821_v1/inputs/aids_generation`.
- Expected source commit:
  `a418692b75b888297222d31d87f49148505e10d0`.
- Controller state at this handoff: `RUNNING` on GPU 1.  The adopted `exp_run`
  worker is PID `111811` and its scientific child is PID `111879`.  This route
  is CPU-heavy while retaining its adopted GPU UUID lock; it has not failed.

## 9. Read-only generation-cache adoption

MUT and AIDS use `generation_mode=adopted_read_only_cache`; generation itself
is disabled with `DISALLOW_GENERATION=1`.  The new roots must bind source file
checksums and rerun serialization, lineage resolution, freeze, and final gate.
There is no bare symlink to a mutable source.  Any dataset/model/config hash,
candidate closure, shard-manifest, source checksum, or active-writer mismatch
must fail closed instead of regenerating silently inside an adopted root.

## 10. BACE clean initializer source

- Audit run: `20260822T030124Z-bace-policy-audit-1c889b9`.
- Audit output:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/provenance-audit/20260822T030124Z-bace-policy-audit-1c889b9`.
- Audit CSV:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_policy_initializer_provenance_20260822T030124Z.csv`.
- Initializer run: `20260822T030604Z-bace-clean-init-1c889b9`.
- Initializer output:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/clean-initializer/20260822T030604Z-bace-clean-init-1c889b9`.
- Selected initializer mode: `raw-base` from
  `/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/ChemLLM-7B-Chat`.
- Final observed states: audit `PASS`; initializer `PASS`.

## 11. Initializer and oracle provenance

The admissible initializer class is `CLEAN_CHEMLLM_BASE`, converted into the
fresh LoRA required by the stable PPO loop.  Required provenance remains:

```text
rf_reference_count=0
gnn_reward_used=false
calibration_loaded=false
test_loaded=false
oracle_backend=gnn   # PPO reward stages only
rf_oracle_used=false
```

BACE split root:
`/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/data/processed/BACE`.
Frozen B4 GINE bundle:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689`.
The policy initializer, GINE classifier, and MolCLR WNode encoder have separate
identities; neither RF nor MolCLR is allowed to masquerade as the classifier.

## 12. Fresh canary and B6-v2 gate

The earlier canary failure is evidence only and must neither be deleted nor
adopted as PASS.  The fresh canary is:

- run ID: `20260822T033440Z-bace-ppo-canary-a625841`;
- output:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/adapter-canary/20260822T033440Z-bace-ppo-canary-a625841`;
- execution worktree: `/root/autodl-tmp/worktrees/run-bace-ppo-a625841`;
- commit: `a6258413619fe2f762980c7172ed20a9917a0e2f`;
- exact launch environment additionally binds
  `PYTHONPATH=/root/autodl-tmp/worktrees/run-bace-ppo-a625841`;
- required deterministic preflight:
  `canary_connected_deletion_preflight.json`;
- terminal state: `PASS`.

The v2 controller revalidated the canary's exact launch spec and full PASS
contract before importing that PASS.  Formal B6-v2 is a **fresh controller
task**, with input evidence closed over
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/adapter-canary/20260822T033440Z-bace-ppo-canary-a625841/canary_manifest.json`;
its environment separately binds the adopted clean initializer.  B6-v2 must
prove 5--10 real optimizer updates, changed policy bytes, unchanged reference
bytes, reloadable checkpoint, finite rewards, at least one valid GNN-scored
deletion, saved pool/reward provenance, no RF, and no calibration/test loading.

The pre-release B6 run
`20260822T034345Z-bace-b6-v2-a625841` is terminal `PASS`.  It is bounded
pre-release validation evidence only: do not adopt it and do not let it publish
controller B6 authority.  The hardened controller has now launched formal B6
from the final immutable commit; B7 will be launched by that same worktree only
after B6 passes.

B6-v2 controller run ID:
`autodl-four-gpu-recovery-20260822T044445Z-v2-bace_b6_ppo_smoke-main-a0`.
Current fresh output:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b6-v2/attempt-0`.
State at this handoff: `RUNNING` on GPU 0, with `exp_run` worker PID `120622`
and scientific child PID `120630`.  No formal B6-v2 `PASS` is claimed yet.

The immutable v1 controller evidence remains at controller ID
`autodl-four-gpu-recovery-20260822T033351Z-v1`: its formal B6 launch failed
before scientific code started because `exp_run` rejected the safe scheduling
key `TOKENIZERS_PARALLELISM`.  The v1 failure and its empty/fresh attempt
evidence must remain unchanged.  Commit `0ad1494` fixes that launch boundary
without weakening credential-key rejection.

## 13. B7 configuration and state

B7 is released only by B6-v2 PASS and uses BACE train source-class parents:

```text
max_steps=300
learning_rate=1e-6
clip_range=0.05
ppo_epochs=1
max_grad_norm=0.5
target_kl=0.3
hard_kl=0.8
adaptive_kl=true
reward_clip=[-5,5]
checkpoints=50,100,150,200,250,300
```

B7 output pattern:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b7-full/attempt-{attempt}`.
State at this handoff: `PENDING`, waiting only for formal B6-v2 `PASS`; B7 has
not been launched and no B7 `PASS` is claimed.

## 14. Four GPU bindings, PID, and launcher

| GPU | UUID | Current role | PID | launcher/session | state |
|---:|---|---|---|---|---|
| 0 | `GPU-0e4e08dd-f7cc-da83-c0f6-a663440c0732` | `bace_b6_ppo_smoke` | `120622 / 120630` | v2 `exp_run` / child | `RUNNING` |
| 1 | `GPU-244f35a8-354a-ef1e-f589-bde7f8a7a690` | adopted `aids_recovery` | `111811 / 111879` | adopted `exp_run` / child | `RUNNING` |
| 2 | `GPU-901b50ea-30b2-4a0c-505f-bf94980e1484` | no READY task until B6 gate | `-` | `-` | `IDLE` |
| 3 | `GPU-2803b403-c056-187e-6047-683d02d3693b` | no READY task until B6 gate | `-` | `-` | `IDLE` |

AutoDL previously had no `tmux`; the launcher therefore uses `nohup` unless
`tmux` becomes available.  Never kill a PID based only on this table: validate
the controller state, kernel process identity, and GPU UUID process first.
Authoritative v2 controller PID: `120431`; kernel start ticks: `680917595`;
launcher: `nohup` (no tmux session).

The old v1 PID `119510` is monitor-only: it owns no new BACE science after its
pre-science launch failure and is only observing the already-adopted AIDS run.
The host returned `pidfd` `ENOSYS`, so no signal was sent.  Leave it alone; it
will exit naturally when AIDS becomes terminal.  Do not use PID-only killing
for either controller or any worker.

## 15. Dependency-aware current queue

The frozen queue is:

```text
MUT v3 adoption
AIDS v2 adoption
BACE audit adoption -> clean initializer adoption -> fresh canary adoption
  -> B6-v2
  -> B7 plus calibration GNN/MolCLR caches, fixed parent manifests, preflight
  -> B8 base[4] + B9 high-temperature[4]
  -> B10 merge
  -> B11 calibration verification[4] -> B11 merge
  -> B12 calibration selector freeze
  -> B13 test-parent manifest -> B13 test verification[4] -> B13 merge
  -> B14 manifest-only freeze
```

At this handoff the only freshly launched BACE task is B6-v2.  B7 and its four
bounded prep tasks wait for B6 `PASS`; B8--B14 remain dependency-gated and
unstarted.  When B6 passes, the controller can immediately fill newly eligible
GPU slots without waiting for AIDS.  Always use the status command for the live
next-READY task rather than inferring it from this snapshot.

## 16. TasteMolNet license block

`RUN_TASTEMOLNET=0` is mandatory.  The only Taste task has `command=null` and
`blocked_reason=TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW`.  Do not launch
TasteMolNet full GNN, PPO, candidate generation, verification, selector, or
baseline work until explicit license approval changes this boundary.

## 17. Paper freeze

`paper_frozen=true`.  Do not modify, regenerate, stage, or bulk-stage
`paper/`.  Current status is
`PAPER_FROZEN_PENDING_BACE_FINAL_AND_TASTE_LICENSE`.

## 18. Persistent roots and model/cache inputs

```text
runtime_root=/autodl-fs/data/counterfactual-subgraph-runtime
control_root=/autodl-fs/data/counterfactual-subgraph-runtime/control
python=/root/miniconda3/envs/smiles_pip118/bin/python
bace_split=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/data/processed/BACE
chemllm=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/ChemLLM-7B-Chat
bace_gnn=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
molclr_root=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/MolCLR
molclr_checkpoint=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth
new_science_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2
fresh_wnode_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/autodl-four-gpu-recovery-20260822T044445Z-v2/wnode/wnode_cache.sqlite3
fresh_node_embedding_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/autodl-four-gpu-recovery-20260822T044445Z-v2/node_embeddings
registry=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiment_registry/runs.jsonl
status_registry=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiment_registry/status_updates.jsonl
experiment_log=/autodl-fs/data/counterfactual-subgraph-runtime/docs/AUTODL_FOUR_GPU_EXPERIMENT_LOG.md
runtime_handoff=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/handoffs/AUTODL_MUT_AIDS_BACE_FOUR_GPU_HANDOFF.md
controller_root=/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2
controller_manifest=/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json
```

The WNode DB and node-embedding directory are fresh persistent targets.  Their
nonexistence/freshness and the MolCLR/B4/split file existence must be checked
read-only immediately before launch; absence is a blocker, never evidence to
invent or redirect a path.

## 19. Status command

After the final immutable worktree and manifest are deployed:

```bash
PROJECT=/root/autodl-tmp/worktrees/run-four-gpu-recovery-0ad149420577
PY=/root/miniconda3/envs/smiles_pip118/bin/python
DATA=/autodl-fs/data
CONTROL=/autodl-fs/data/counterfactual-subgraph-runtime/control
CID=autodl-four-gpu-recovery-20260822T044445Z-v2

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/status_four_gpu_recovery.py" \
  --project-root "$PROJECT" \
  --data-root "$DATA" \
  --control-root "$CONTROL" \
  --controller-id "$CID" \
  --watch 60
```

For one JSON snapshot, replace `--watch 60` with `--format json`.

## 20. Deployment, launch, and safe restart

The v2 manifest has already passed validation and is frozen read-only.  For a
restart, first use section 19 and verify PID `120431` with start ticks
`680917595`, heartbeat freshness, and the controller lock.  Do not restart
while that identity is live.  Once it is proven dead and the controller lock is
naturally released, reuse exactly the same manifest and controller ID:

```bash
PROJECT=/root/autodl-tmp/worktrees/run-four-gpu-recovery-0ad149420577
PY=/root/miniconda3/envs/smiles_pip118/bin/python
DATA=/autodl-fs/data
CONTROL=/autodl-fs/data/counterfactual-subgraph-runtime/control
MANIFEST="$CONTROL/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json"

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/run_four_gpu_recovery_controller.py" \
  validate --manifest "$MANIFEST"

AUTODL_DATA_ROOT="$DATA" \
AUTODL_CONTROL_ROOT="$CONTROL" \
AUTODL_PYTHON="$PY" \
AUTODL_MAX_GPUS=4 \
GLOBAL_MAX_CONCURRENT_GPU_JOBS=4 \
RUN_TASTEMOLNET=0 \
PYTHONDONTWRITEBYTECODE=1 \
"$PROJECT/scripts/autodl/launch_four_gpu_recovery.sh" "$MANIFEST"
```

If the foreground connection is lost, do not launch a second controller until
the status/heartbeat and process identity prove the first controller dead.
Once dead and its lock is naturally released, run the exact same launcher
command with the same manifest and controller ID.  It resumes from persistent
`controller_state.json`, registry records, and passing-attempt manifests; it
must not rewrite a completed shard or adopt a different writer.

The remaining v1 monitor PID `119510` is not the v2 restart target.  Do not
signal it solely to make the process list look clean; its v1 state and launch
failure are historical evidence, and it should finish naturally with AIDS.

Handoff snapshot:

```text
development_commit=0ad149420577c683baa2ef03f78f70ee6841f3a1
lineage_run_commit=6ddd74339dbd9b1f0e57ba341ae4529cc2864fce
bace_run_commit=0ad149420577c683baa2ef03f78f70ee6841f3a1
controller_id=autodl-four-gpu-recovery-20260822T044445Z-v2
controller_pid=120431
controller_start_ticks=680917595
controller_tmux=nohup
controller_manifest=/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json
gpu0_task=bace_b6_ppo_smoke RUNNING worker=120622 child=120630
gpu1_task=aids_recovery RUNNING worker=111811 child=111879
gpu2_task=IDLE waiting_for_b6_gate
gpu3_task=IDLE waiting_for_b6_gate
mut_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z
aids_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/aids_comrecgc_lineage_v2_20260822T020238Z
bace_initializer=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/clean-initializer/20260822T030604Z-bace-clean-init-1c889b9/adapter
bace_gnn_checkpoint=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
bace_b6_v2_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b6-v2/attempt-0
bace_b7_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b7-full/attempt-{passing-attempt}
bace_final_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/frozen_gnn_downstream/b14-frozen/attempt-{passing-attempt}
taste_status=BLOCKED_LICENSE_REVIEW
paper_status=PAPER_FROZEN_PENDING_BACE_FINAL_AND_TASTE_LICENSE
handoff_path=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/handoffs/AUTODL_MUT_AIDS_BACE_FOUR_GPU_HANDOFF.md
status_command=see section 19
controller_restart_command=repeat section 20 exact v2 launcher only after dead-controller proof
```
