# AutoDL Mutagenicity / AIDS / BACE four-GPU handoff

Date: 2026-08-22  
Scope: AutoDL only; no HPC GPU or CPU; TasteMolNet heavy work disabled; `paper/`
frozen.  This document is a deployment handoff, not a claim that every
scientific stage has already passed.

The complete local candidate controller manifest is
`configs/autodl/four_gpu_recovery.live_candidate.json`.  Its local static
validation passed with 22 tasks, a four-GPU ceiling, no `__CONFIGURE_*`
placeholders, and exactly one blocked task: TasteMolNet license review.  The
candidate manifest SHA-256 is
`e041c268973ee42ecaea0c51f863b3d434f6f053de03ea61d823430a66d44fb1`.
AutoDL path existence and exact `exp_run` adoption equality were deliberately
not fabricated by this local-only drafting step; they must pass the controller's
read-only launch validation before deployment.

## 1. Baseline and final development commit

- Frozen starting authority: `8b17fb1096666852b0680f899073dd82f207cce1`.
- Manifest/handoff integration base: `2dc911168ee647c1cd19a16efe07d58d7ab0f1d3`.
- Deterministic canary repair to include before deployment:
  `a6258413619fe2f762980c7172ed20a9917a0e2f`.
- Final integrated development commit: `__FINAL_DEVELOPMENT_COMMIT__`.
- Final immutable controller/BACE execution worktree:
  `/root/autodl-tmp/worktrees/run-four-gpu-__FINAL_COMMIT_SHORT__`.

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

The controller delegates workers to `exp_run`, uses UUID locks, samples idle
GPUs for at least 60 seconds, permits at most one OOM down-batch retry, never
retries semantic failures, and keeps TasteMolNet and paper blocked.

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

The live candidate manifest is intentionally not part of the reusable code
commit.  Freeze and copy its exact bytes to the persistent control root only
after the final commit and read-only AutoDL preflight.

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

This handoff worktree did not rerun pytest because its local Python lacks the
`pytest` package.  It did run the stdlib-backed controller manifest validator,
which returned `status=PASS`.  Do not infer a new scientific PASS from that
static validation.

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
- Controller state at final handoff: `__CURRENT_MUT_STATE__`.

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
- Controller state at final handoff: `__CURRENT_AIDS_STATE__`.

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
- Final observed states: audit `__CURRENT_BACE_AUDIT_STATE__`, initializer
  `__CURRENT_BACE_INITIALIZER_STATE__`.

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
- required deterministic preflight:
  `canary_connected_deletion_preflight.json`;
- current state: `__CURRENT_BACE_CANARY_STATE__`.

The controller adopts and observes this writer; it does not infer PASS.  B6-v2
starts only after the canary's complete PASS contract.  B6-v2 must prove 5--10
real optimizer updates, changed policy bytes, unchanged reference bytes,
reloadable checkpoint, finite rewards, at least one valid GNN-scored deletion,
saved pool/reward provenance, no RF, and no calibration/test loading.

B6-v2 output pattern:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1/bace/gnn_ppo/b6-v2/attempt-{attempt}`.
Final state: `__CURRENT_BACE_B6_V2_STATE__`.

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
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1/bace/gnn_ppo/b7-full/attempt-{attempt}`.
Final state: `__CURRENT_BACE_B7_STATE__`.

## 14. Four GPU bindings, PID, and launcher

| GPU | UUID | Adopted/current role | PID | launcher/session | state |
|---:|---|---|---|---|---|
| 0 | `GPU-0e4e08dd-f7cc-da83-c0f6-a663440c0732` | MUT v3 adoption / `__CURRENT_GPU0_TASK__` | `__CURRENT_GPU0_PID__` | `__CURRENT_GPU0_LAUNCHER__` | `__CURRENT_GPU0_STATE__` |
| 1 | `GPU-244f35a8-354a-ef1e-f589-bde7f8a7a690` | AIDS v2 adoption / `__CURRENT_GPU1_TASK__` | `__CURRENT_GPU1_PID__` | `__CURRENT_GPU1_LAUNCHER__` | `__CURRENT_GPU1_STATE__` |
| 2 | `GPU-901b50ea-30b2-4a0c-505f-bf94980e1484` | clean initializer historical / `__CURRENT_GPU2_TASK__` | `__CURRENT_GPU2_PID__` | `__CURRENT_GPU2_LAUNCHER__` | `__CURRENT_GPU2_STATE__` |
| 3 | `GPU-2803b403-c056-187e-6047-683d02d3693b` | fresh canary adoption / `__CURRENT_GPU3_TASK__` | `__CURRENT_GPU3_PID__` | `__CURRENT_GPU3_LAUNCHER__` | `__CURRENT_GPU3_STATE__` |

AutoDL previously had no `tmux`; the launcher therefore uses `nohup` unless
`tmux` becomes available.  Never kill a PID based only on this table: validate
the controller state, kernel process identity, and GPU UUID process first.
Controller PID: `__CONTROLLER_PID__`; tmux/session:
`__CONTROLLER_TMUX_OR_NOHUP__`.

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

The next READY task and current B8--B14 states must be taken from the status
command, not inferred here: `__CURRENT_NEXT_READY_TASK__` and
`__CURRENT_BACE_B8_B14_STATES__`.

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
new_science_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1
fresh_wnode_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/autodl-four-gpu-recovery-20260822T033351Z-v1/wnode/wnode_cache.sqlite3
fresh_node_embedding_cache=/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/frozen_gnn_downstream/autodl-four-gpu-recovery-20260822T033351Z-v1/node_embeddings
registry=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiment_registry/runs.jsonl
status_registry=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiment_registry/status_updates.jsonl
experiment_log=/autodl-fs/data/counterfactual-subgraph-runtime/docs/AUTODL_FOUR_GPU_EXPERIMENT_LOG.md
runtime_handoff=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/handoffs/AUTODL_MUT_AIDS_BACE_FOUR_GPU_HANDOFF.md
```

The WNode DB and node-embedding directory are fresh persistent targets.  Their
nonexistence/freshness and the MolCLR/B4/split file existence must be checked
read-only immediately before launch; absence is a blocker, never evidence to
invent or redirect a path.

## 19. Status command

After the final immutable worktree and manifest are deployed:

```bash
PROJECT=/root/autodl-tmp/worktrees/run-four-gpu-__FINAL_COMMIT_SHORT__
PY=/root/miniconda3/envs/smiles_pip118/bin/python
DATA=/autodl-fs/data
CONTROL=/autodl-fs/data/counterfactual-subgraph-runtime/control
CID=autodl-four-gpu-recovery-20260822T033351Z-v1

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

Freeze the candidate bytes at:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/control/four_gpu_recovery_manifest_20260822T033351Z.json
```

Before the first launch, run the same validator with the final worktree and
verify all five adoption specs against the persistent `launch_spec.json`; this
is especially important for the currently running canary because adoption
requires exact command, environment, output contract, interpreter, commit,
GPU, and input hash equality.

```bash
PROJECT=/root/autodl-tmp/worktrees/run-four-gpu-__FINAL_COMMIT_SHORT__
PY=/root/miniconda3/envs/smiles_pip118/bin/python
DATA=/autodl-fs/data
CONTROL=/autodl-fs/data/counterfactual-subgraph-runtime/control
MANIFEST="$CONTROL/four_gpu_recovery_manifest_20260822T033351Z.json"

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

Final dynamic summary to fill immediately before returning to the user:

```text
development_commit=__FINAL_DEVELOPMENT_COMMIT__
lineage_run_commit=6ddd74339dbd9b1f0e57ba341ae4529cc2864fce
bace_run_commit=__FINAL_DEVELOPMENT_COMMIT__
controller_pid=__CONTROLLER_PID__
controller_tmux=__CONTROLLER_TMUX_OR_NOHUP__
gpu0_task=__CURRENT_GPU0_TASK__
gpu1_task=__CURRENT_GPU1_TASK__
gpu2_task=__CURRENT_GPU2_TASK__
gpu3_task=__CURRENT_GPU3_TASK__
mut_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z
aids_output_root=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/aids_comrecgc_lineage_v2_20260822T020238Z
bace_initializer=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/gnn_ppo/clean-initializer/20260822T030604Z-bace-clean-init-1c889b9/adapter
bace_gnn_checkpoint=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
bace_b6_v2_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1/bace/gnn_ppo/b6-v2/attempt-{passing-attempt}
bace_b7_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1/bace/gnn_ppo/b7-full/attempt-{passing-attempt}
bace_final_output=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/four_gpu_recovery/autodl-four-gpu-recovery-20260822T033351Z-v1/bace/frozen_gnn_downstream/b14-frozen/attempt-{passing-attempt}
taste_status=BLOCKED_LICENSE_REVIEW
paper_status=PAPER_FROZEN_PENDING_BACE_FINAL_AND_TASTE_LICENSE
handoff_path=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/handoffs/AUTODL_MUT_AIDS_BACE_FOUR_GPU_HANDOFF.md
status_command=see section 19
controller_restart_command=repeat section 20 launcher only after dead-controller proof
```
