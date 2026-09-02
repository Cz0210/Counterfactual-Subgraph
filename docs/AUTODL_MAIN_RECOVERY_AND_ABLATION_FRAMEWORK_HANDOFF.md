# AutoDL Main Recovery and Ablation Framework Handoff

Last read-only main-state audit: 2026-09-02 17:31:27 CST. T8/T13 was
repaired and relaunched at approximately 16:46 CST. This document
separates main-table execution facts from ablation preparation. The matrix
authority and task heartbeats remain authoritative over this handoff.

## 1. Main-table authority

- Authority: `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Authority mtime at audit: `2026-09-02 11:36:37.792215344 +0800`
- Registered cells: `12 / 16`
- Complete: all four AIDS cells, all four BACE cells,
  Mutagenicity/Ours, Mutagenicity/GCFExplainer,
  Mutagenicity/GlobalGCE, and TasteMolNet/Ours.
- Missing: Mutagenicity/ComRecGC and TasteMolNet/GCFExplainer,
  TasteMolNet/GlobalGCE, TasteMolNet/ComRecGC.

The framework work did not write the matrix authority, active roots, or GPU
leases. No LLM or GNN ablation science was launched.

## 2. Main workers and recovery state

| Line | Last verified state | GPU | Evidence |
|---|---|---:|---|
| Taste T8/T13 GlobalGCE | One allowed grade-recovery attempt running after a provenance-only checkpoint resume; controller PID `82588`, science PID `82680`, checkpoint `TARGET_0_RUNNING`, output 126,323,403 bytes, seed 7, 100 epochs, ordinal `1/1` | 1 | `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t8-t13-grade-recovery-once-20260902T084623Z-99cad1d6` |
| Taste T12 GCFExplainer | Healthy original science PID `66459`; output 777,498,920 bytes and growing; not restarted or reconfigured | 3 | Existing T12 production root/relay |
| Taste T14 ComRecGC | Healthy original science PID `7224`; progress 7,900/20,000 (39.5%), committed checkpoint 7,500 | 2 | Existing T14 production root/relay |
| Mut ComRecGC | Worker PID `67797` absent; latest equivalence attempt failed its protected-throughput watchdog because T14 slowdown exceeded 10% | none | `/autodl-fs/data/counterfactual-subgraph-runtime/control/mut_fast_accurate_v2/mut_fast_accurate_v2_20260901T043250Z/trace_on_adoption_worker_heartbeat.json` |

T8/T13 originally failed before training because the loader discarded the
already-validated official `runtime_source_authority`. Commit
`b12c9c80bf4ed5df8ae3f75e6d9c2992a5de9474` retains and passes that authority.
The resumed process uses the same UUID
`18675079-382b-41a9-b6ac-f5aa6e79babf`, the same output root, the same
checkpoint, and the same one-shot ordinal. It is not a second scientific
attempt. Local focused tests were 109 PASS and remote focused tests were 7
PASS before relaunch.

Mut was deliberately not restarted because the current instruction explicitly
protects it from restart. Its science gate did not fail trace equivalence; the
resource watchdog stopped it to protect T14. Mut therefore remains a main-table
blocker and must be resumed by the existing main release policy after the
protected-resource condition changes.

## 3. Main critical path

The remaining four cells depend on T12, T14, the one-shot T8/T13 recovery, and
a later safe Mut continuation. T12/T14/T8 are long-running main science and
must not be restarted for observability. No defensible 16/16 wall-clock time is
available from the current checkpoints. When any task finishes, its existing
postprocess/publisher relay must append through the same matrix authority.

## 4. LLM framework branch and deployment

- Branch: `feat/llm-stage-scale-ablation-v2`
- Framework commits: `443437f`, `e60bc67`, `2ab8ecbf`
- Final commit: `2ab8ecbf436b796283122041d06b3bcc12708ee6`
- Immutable AutoDL worktree:
  `/root/autodl-tmp/worktrees/llm-stage-scale-ablation-2ab8ecb-20260902T093000Z`
- Local and AutoDL focused tests: `76 passed`
- Python compile, `git diff --check`, and AutoDL/Slurm shell syntax: PASS

The v2 launcher is deliberately config-only and exits blocked. It cannot
acquire a GPU or start science. This is safer than exposing an incomplete
launcher while runtime model evidence and main-table gates are unresolved.

## 5. BACE/Ours LLM reference

- Reference v2:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/contracts/bace_ours_llm_reference_v2.json`
- File SHA-256:
  `775ab902fa02c7be8748f0d8ce9514e4aba0f57e6c61fc4f6c40e829b37b46c6`
- Contract self SHA-256:
  `344f4815d327f7bdbc5c276f7dbf35ac4f671f07833f49d8208c32ee457ecdd2`

The real main lineage is:

`AI4Chem/ChemLLM-7B-Chat -> fresh LoRA initializer -> PPO (300 updates)`

There is no independently matched BACE project-SFT checkpoint. Consequently:

- A0 `BRICS_FIXED`: CPU preparation is allowed.
- A1 `CHEMLLM_7B_OFF_THE_SHELF`: topology/reference framework is available,
  but GPU science remains gated.
- A2 `CHEMLLM_7B_PROJECT_SFT`: `BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT`.
- A3 `CHEMLLM_7B_PROJECT_SFT_PPO`: blocked because the existing policy is
  fresh-LoRA PPO, not project-SFT+PPO. It must not be relabelled.

## 6. Model-scale evidence

### 2B

- Model: `AI4Chem/CHEMLLM-2b-1_5`
- Revision: `215c0dbc89417a06bbc3bae43a3ad61e58f0a56e`
- Snapshot:
  `/autodl-fs/data/counterfactual-subgraph-runtime/models/chemllm/CHEMLLM-2b-1_5/215c0dbc89417a06bbc3bae43a3ad61e58f0a56e`
- Exact safetensors-header count: `1,889,110,016` BF16 parameters.
- Four weight shards match their pinned LFS hashes.
- State: `SNAPSHOT_READY_SCIENCE_LOAD_BLOCKED`; isolated remote-code import has
  not passed, so header evidence is not called an actual-loaded parameter
  report.

### 7B main reference

- Revision: `b8b2ea19e48f53d190fe8dced94572717f8e89a2`
- Exact safetensors-header base count: `7,737,708,544` BF16 parameters.
- PPO LoRA header count: `18,874,368` FP32 parameters.
- Header total: `7,756,582,912`.
- These are header-exact counts, not a newly generated actual-loaded report.

### Optional 20B

- Model: `AI4Chem/ChemLLM-20B-Chat-SFT`
- Revision: `e8d0f503e00f143f6787263765ff6ee5f3fe3998`
- Metadata manifest:
  `/autodl-fs/data/counterfactual-subgraph-runtime/models/chemllm/ChemLLM-20B-Chat-SFT/e8d0f503e00f143f6787263765ff6ee5f3fe3998/metadata/metadata_manifest.json`
- `weights_downloaded=false`, `run_enabled=false`.
- The model-index estimate is metadata only and is never reported as an
  actual-loaded parameter count.

The requested matched 2B-full versus 7B-full scale study is blocked by the
missing 7B project-SFT lineage. The supported fallback design is only
`2B off-the-shelf vs 7B off-the-shelf`, labelled
`MODEL_SCALE_PROPOSAL_SENSITIVITY`. It is not a full-method parameter ablation.

## 7. BRICS contract and CPU artifact

The BRICS vocabulary reads the complete frozen BACE train split (959 rows).
The separate 386-parent source cohort is used only for proposal generation.
The primary budget is eight attempts per parent; every attempt and shortfall is
written, and no candidate is copied to fill a shortfall.

The audited main B10 freeze has no active numeric fragment-atom or atom-ratio
threshold. Its structural predicate is parseable, valid, connected,
chirality-aware direct-substructure with a valid oracle outcome. The shared
downstream hard deletion additionally requires a non-empty, sanitizable,
single-component residual. PPO reward windows and disabled projection limits
are not final candidate filters. The BRICS proposer therefore records all
numeric size bounds as `NONE`, uses `useChirality=True` for proposal-time
matching, and leaves hard deletion/oracle/strict flip to the common downstream.

CPU output root:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/bace_ours_stage_v2/BRICS_FIXED/seed7/attempt-de7345e8-c1e1-448a-8154-41628d28a0a9`

The terminal is PASS: 472 vocabulary entries, 386 parents, 3,088 proposal
attempts, 3,088 candidates, and zero shortfalls. Calibration/test/oracle were
not loaded, `gpu_used=false`, and all six listed artifacts pass
`sha256sum -c`. The CPU command acquired no GPU and used one low-priority
worker.

## 8. Launch gates

Early LLM GPU science currently remains blocked because:

1. the registered matrix is below 13/16;
2. Mut is not PASS and still needs a future main GPU continuation;
3. main-table science occupies/reserves the remaining GPU path;
4. no valid early-run receipt exists;
5. 2B runtime loading and actual-loaded parameter reports are incomplete;
6. framework v2 intentionally has no science entrypoint.

The receipt schema binds the physical matrix-authority SHA, runtime run-contract
SHA, and exact Git execution commit. Caller-supplied booleans cannot override
those bindings.

The persisted decision is:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/main_llm_scale_ablation_20260902T093127Z/ablation_start_decision.json`

It reports `BLOCKED_MAIN_PRIORITY`, `science_started=false`, and
`gpu_lock_acquired=false`.

## 9. GNN framework

The BACE/Ours proposal-fixed framework for GINE, GIN, GCN, and GATv2 remains
config-only and rejects science launch before 16/16. Native and common cohort
contracts are retained. Optional end-to-end execution is not implemented in
the current planner and must not be reported as framework PASS. No GNN process
or GPU lease was created.

## 10. Status commands

Main authority:

```bash
/root/miniconda3/envs/smiles_pip118/bin/python \
  scripts/autodl/status_fast_16of16_v2.py \
  --config configs/hpc.yaml
```

LLM config/evidence status from the immutable worktree:

```bash
export BACE_LLM_REFERENCE_CONTRACT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/contracts/bace_ours_llm_reference_v2.json
export BACE_LLM_REFERENCE_CONTRACT_SHA256=775ab902fa02c7be8748f0d8ce9514e4aba0f57e6c61fc4f6c40e829b37b46c6
export LLM_ABLATION_MAIN_SNAPSHOT=/path/to/a/fresh/read-only-main-snapshot.json
/root/autodl-tmp/worktrees/llm-stage-scale-ablation-2ab8ecb-20260902T093000Z/scripts/autodl/launch_llm_ablation_v2.sh
```

The latter is expected to finish with
`BLOCKED_CONFIG_ONLY_NO_SCIENCE_ENTRYPOINT`; that is a safe framework state,
not a failed experiment.
