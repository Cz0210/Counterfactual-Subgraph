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
historical runs, retained Mutagenicity `PASS`, continued monitoring AIDS,
passed the fresh final-commit B6-v2 gate, and is running B7 on GPU 0.  Three B7
preparation tasks passed.  The original v2 MolCLR-parent preparation attempt
failed before scientific computation because the shared loader rejected the
valid explicit device `cuda:0`; its failure evidence is preserved.  A bounded
fresh-root repair on commit `d26fe279fcb7778e8e021864878c7851d8900a12`
passed, but it does not rewrite the v2 dependency state.  No B7 or downstream
scientific PASS is claimed before its own gate.

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
- MolCLR explicit-device repair commit:
  `d26fe279fcb7778e8e021864878c7851d8900a12`.
- Immutable MolCLR repair execution worktree:
  `/root/autodl-tmp/worktrees/run-bace-molclr-prep-d26fe279fcb7`.

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
- Fail-closed support for indexed MolCLR CUDA devices:
  `d26fe279fcb7778e8e021864878c7851d8900a12`.

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

`d26fe27`:

```text
M docs/decisions.md
M src/embeddings/molclr_gnn_embedding.py
A tests/embeddings/test_molclr_device_resolution.py
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
controller B6 authority.  The hardened controller separately launched and
passed formal B6 from the final immutable commit, and only that formal result
released B7.

B6-v2 controller run ID:
`autodl-four-gpu-recovery-20260822T044445Z-v2-bace_b6_ppo_smoke-main-a0`.
Current fresh output:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b6-v2/attempt-0`.
State at this handoff: formal `PASS`.  Its gate records:

```text
ppo_training_performed=true
ppo_update_count=5
optimizer_step_count=5
candidate_count=20
valid_candidate_count=9
gnn_scored_deletion_count=3
strict_flip_count=1
policy_parameter_hash_before != policy_parameter_hash_after
reference_parameter_hash_before == reference_parameter_hash_after
checkpoint_saved=true
checkpoint_reload_pass=true
candidate_pool_saved=true
reward_manifest_saved=true
oracle_backend=gnn
rf_oracle_used=false
calibration_loaded=false
test_loaded=false
failures=[]
```

The remaining formal contract checks also passed: checkpoint save/reload,
finite rewards, candidate/reward provenance, GNN oracle, RF guard, and
calibration/test leakage guards.  PIDs `120622 / 120630` are historical B6
worker/child identities and no longer own GPU 0.

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

B7 current output:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b7-full/attempt-0`.
B7 controller run ID:
`autodl-four-gpu-recovery-20260822T044445Z-v2-bace_b7_ppo_full-main-a0`.
State at this handoff: `RUNNING` on GPU 0, with `exp_run` worker PID `121554`
and scientific child PID `121562`.  No B7 `PASS` is claimed yet.

### B7 preparation and MolCLR repair

The following v2 preparation attempts are terminal `PASS`:

```text
bace_b7_prep_gnn_before:
  /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/frozen_gnn_downstream/b7-prep/gnn-before/attempt-0
bace_b7_prep_shard_manifests:
  /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/frozen_gnn_downstream/b7-prep/shard-manifests/attempt-0
bace_b7_prep_output_preflight:
  /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/frozen_gnn_downstream/b7-prep/output-preflight/attempt-0
```

The original v2 `bace_b7_prep_molclr_parent` attempt is terminal `FAILED` and
must stay immutable:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/frozen_gnn_downstream/b7-prep/molclr-parent/attempt-0
```

It failed at device parsing because the shared MolCLR loader accepted `cuda`
but rejected the scheduler's valid explicit device `cuda:0`.  It did not
justify changing MolCLR weights, embeddings, cache identity, or scientific
semantics.  Commit `d26fe279fcb7778e8e021864878c7851d8900a12` adds bounded,
fail-closed `cuda:N` validation.  The corrected isolated run is:

```text
run_id=bace-molclr-parent-repair-v2-d26fe27-20260822T052100Z
state=PASS
worktree=/root/autodl-tmp/worktrees/run-bace-molclr-prep-d26fe279fcb7
output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery_repairs/bace-molclr-parent-v2-20260822T052100Z-d26fe27
node_embedding_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/molclr-device-v2-d26fe27/node_embeddings
parent_count=66
test_loaded=false
```

This repair is positive replacement evidence, not permission to mutate the
failed v2 task state or its attempt root.

## 14. Four GPU bindings, PID, and launcher

| GPU | UUID | Current role | PID | launcher/session | state |
|---:|---|---|---|---|---|
| 0 | `GPU-0e4e08dd-f7cc-da83-c0f6-a663440c0732` | `bace_b7_ppo_full` | `121554 / 121562` | v2 `exp_run` / child | `RUNNING`; about 21.6 GiB, 99% util at snapshot |
| 1 | `GPU-244f35a8-354a-ef1e-f589-bde7f8a7a690` | adopted `aids_recovery` | `111811 / 111879` | adopted `exp_run` / child | `RUNNING`; UUID lock retained |
| 2 | `GPU-901b50ea-30b2-4a0c-505f-bf94980e1484` | no READY task until B7 gate | `-` | `-` | `IDLE` |
| 3 | `GPU-2803b403-c056-187e-6047-683d02d3693b` | no READY task until B7 gate | `-` | `-` | `IDLE` |

AutoDL previously had no `tmux`; the launcher therefore uses `nohup` unless
`tmux` becomes available.  Never kill a PID based only on this table: validate
the controller state, kernel process identity, and GPU UUID process first.
Authoritative v2 controller PID: `120431`; kernel start ticks: `680917595`;
launcher: `nohup` (no tmux session).

At the `2026-08-22T05:32:47Z` snapshot, PID `120431` was alive and its
heartbeat age was 18 seconds.  The dashboard's raw/workload aggregate state is
`FAILED` solely because the immutable original MolCLR attempt is terminal
`FAILED`; this is not evidence that B7 or AIDS failed.  Use task-level states
and the heartbeat to interpret this mixed terminal/running controller.

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

At this handoff B6-v2 is `PASS`, B7 is `RUNNING`, three bounded prep tasks are
`PASS`, and the original v2 MolCLR prep is `FAILED`.  Once B7 passes, the
current v2 controller can safely and automatically advance B8, B9, and B10;
these stages do not need the failed MolCLR-parent dependency.  Let that
controller finish its B8/B9 fixed shards and B10 merge without interference.

The exact downstream snapshot is:

```text
B8 base shards=WAITING_DEPENDENCY
B9 high-temperature shards=WAITING_DEPENDENCY
B10 merge=WAITING
B11 verification through B14 freeze=BLOCKED
```

The current v2 graph cannot release B11 because B11 still has the historical
failed `bace_b7_prep_molclr_parent` task as a dependency.  Do not rewrite that
task to `PASS`, inject the repaired output into its attempt directory, or start
a parallel duplicate controller.  After the current v2 controller has reached
B10 `PASS` and has no eligible duplicate work left, create one new continuation
controller with a new controller ID and fresh controller/output namespace.  It
must exact-adopt the passing B6, B7, three original prep outputs, all B8/B9
shards, B10, and the corrected MolCLR repair output, then release B11--B14 from
those immutable identities.  Exact adoption must fail closed on any path,
commit, config, checksum, split, or gate mismatch.  Always use the status
command for live state rather than inferring it from this snapshot.

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
molclr_repair_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery_repairs/bace-molclr-parent-v2-20260822T052100Z-d26fe27
molclr_repair_node_embedding_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/molclr-device-v2-d26fe27/node_embeddings
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
development_commit=d26fe279fcb7778e8e021864878c7851d8900a12
lineage_run_commit=6ddd74339dbd9b1f0e57ba341ae4529cc2864fce
bace_run_commit=0ad149420577c683baa2ef03f78f70ee6841f3a1
molclr_repair_commit=d26fe279fcb7778e8e021864878c7851d8900a12
controller_id=autodl-four-gpu-recovery-20260822T044445Z-v2
controller_pid=120431
controller_start_ticks=680917595
controller_tmux=nohup
controller_snapshot_time=2026-08-22T05:32:47Z
controller_heartbeat_age=18s
controller_raw_workload_state=FAILED solely_from_preserved_molclr_attempt_0
controller_manifest=/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json
gpu0_task=bace_b7_ppo_full RUNNING worker=121554 child=121562 memory_about_21.6GiB util_about_99pct
gpu1_task=aids_recovery RUNNING worker=111811 child=111879 uuid_lock_retained
gpu2_task=IDLE waiting_for_b7_gate
gpu3_task=IDLE waiting_for_b7_gate
mut_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z
aids_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/aids_comrecgc_lineage_v2_20260822T020238Z
bace_initializer=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/clean-initializer/20260822T030604Z-bace-clean-init-1c889b9/adapter
bace_gnn_checkpoint=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
bace_b6_v2_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b6-v2/attempt-0
bace_b6_v2_state=PASS
bace_b7_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T044445Z-v2/bace/gnn_ppo/b7-full/attempt-0
bace_b7_state=RUNNING
bace_molclr_v2_attempt_state=FAILED_PRESERVED
bace_molclr_repair_run=bace-molclr-parent-repair-v2-d26fe27-20260822T052100Z
bace_molclr_repair_worktree=/root/autodl-tmp/worktrees/run-bace-molclr-prep-d26fe279fcb7
bace_molclr_repair_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery_repairs/bace-molclr-parent-v2-20260822T052100Z-d26fe27
bace_molclr_repair_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/molclr-device-v2-d26fe27/node_embeddings
bace_molclr_repair_state=PASS parent_count=66 test_loaded=false
bace_current_controller_safe_limit=B10
bace_b8_b9_state=WAITING_DEPENDENCY
bace_b10_state=WAITING
bace_b11_state=BLOCKED_FAILED_HISTORICAL_MOLCLR_DEPENDENCY
bace_b12_b14_state=BLOCKED
bace_final_output=not_created; future continuation controller fresh root required after B10 PASS
taste_status=BLOCKED_LICENSE_REVIEW
paper_status=PAPER_FROZEN_PENDING_BACE_FINAL_AND_TASTE_LICENSE
handoff_path=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/handoffs/AUTODL_MUT_AIDS_BACE_FOUR_GPU_HANDOFF.md
status_command=see section 19
controller_restart_command=repeat section 20 exact v2 launcher only after dead-controller proof
continuation_controller=build only after current v2 B10 PASS; exact-adopt immutable PASS outputs and corrected MolCLR repair; never run in parallel
```
