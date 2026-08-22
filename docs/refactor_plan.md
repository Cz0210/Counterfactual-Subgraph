# Refactor Plan

## 2026-08-22: BACE four-by-four cell closure

- [x] Keep B14 and native baseline final roots as immutable scientific
  terminals rather than treating them as paper-ready cells.
- [x] Add deterministic, CPU-only BACE Ours/GCFExplainer/ComRecGC
  standardization tasks with SHA256 identity traversal and no raw test access.
- [x] Export the complete common cell schema, file inventory, freeze marker,
  final audit, and PASS-last marker under fresh roots.
- [x] Reject final-matrix mappings that bypass the standardization layer.
- [x] Preserve GlobalGCE's reviewed code blocker without generating a
  substitute result.

## 2026-08-22: Continuation predecessor namespace binding

- [x] Resolve the BACE predecessor controller from the exact persistent source
  manifest namespace while keeping the new four-by-four controller in its own
  namespace.
- [x] Reject source manifests outside the control root, malformed namespace
  layouts, symlinked roots, and controller snapshot identity mismatches.

## 1. Purpose

This document records the intended roadmap for rebuilding the counterfactual subgraph v3 project from an empty repository.

The goal is not merely to write working code, but to build a clean research codebase that remains faithful to the counterfactual objective and is easy to evolve.

---

## 2. Rebuild Strategy

The project should be rebuilt incrementally.

The guiding principle is:

> First stabilize interfaces and responsibilities, then implement training logic.

This is important because earlier versions were likely affected by script-level coupling, implicit assumptions, and reward/training entanglement.

---

## 3. Phase Overview

## Phase 0: Documentation-first bootstrap

Objective:

- establish the research objective in writing;
- define repository conventions;
- ensure Codex and future contributors follow the same target.

Deliverables:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

Status:

- completed on 2026-04-09.

---

## Phase 1: Repository skeleton

Objective:

- create the core directory structure;
- define code boundaries;
- prepare CLI and config folders.

Deliverables:

```text
configs/
data/
scripts/
src/
tests/
outputs/
```

Recommended first-level modules:

```text
src/data/
src/models/
src/rewards/
src/train/
src/eval/
src/chem/
src/utils/
```

Success criteria:

- all major concerns have a dedicated location;
- no business logic lives in random top-level files.

Status:

- bootstrap skeleton implemented on 2026-04-09.
- training logic intentionally deferred.

### Suggested target directory structure

The repository should now grow toward the following structure:

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── README.md
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── reward/
│   └── eval/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
│   ├── cf_subgraph_v3_spec.md
│   ├── decisions.md
│   └── refactor_plan.md
├── outputs/
├── scripts/
│   ├── README.md
│   ├── prepare_data.py
│   ├── infer_single.py
│   ├── train_sft.py
│   ├── train_rl.py
│   └── eval_model.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   ├── dataset.py
│   │   └── collators.py
│   ├── chem/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── smiles_utils.py
│   │   ├── substructure.py
│   │   ├── deletion.py
│   │   └── validation.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── aggregation.py
│   │   ├── anti_collapse.py
│   │   └── counterfactual_reward.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── interfaces.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   └── diagnostics.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   ├── metrics.py
│   │   └── reporting.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging.py
│       └── seed.py
└── tests/
    ├── README.md
    ├── test_prompt_contract.py
    └── test_reward_breakdown.py
```

### Module responsibilities and minimum interfaces

#### `src/data/`

Responsibility:

- define the canonical JSONL schema;
- centralize prompt construction;
- expose dataset and batch contracts that can be reused by SFT, RL, and evaluation.

Minimum interface:

- `MoleculeRecord` and `FragmentExample` dataclasses;
- `normalize_molecule_record(raw)` for schema validation;
- `build_counterfactual_prompt(record, include_label=False)` for prompt generation;
- `JsonlMoleculeDataset.from_jsonl(path)` for deterministic loading;
- `CounterfactualPromptCollator` returning a `PromptBatch`.

#### `src/chem/`

Responsibility:

- own all chemistry-specific behavior;
- keep parsing, validation, substructure checks, and deletion out of train scripts;
- serve as the only place where future RDKit logic should live.

Minimum interface:

- `ParsedMolecule`, `FragmentValidationResult`, and `DeletionResult` dataclasses;
- `parse_smiles(smiles)` and `canonicalize_smiles(smiles)` placeholders;
- `is_parent_substructure(parent_smiles, fragment_smiles)` placeholder;
- `is_connected_fragment(fragment_smiles)` placeholder;
- `delete_fragment_from_parent(parent_smiles, fragment_smiles)` placeholder;
- `validate_fragment_candidate(parent_smiles, fragment_smiles)` placeholder.

#### `src/rewards/`

Responsibility:

- represent reward terms explicitly;
- keep counterfactual scoring distinct from structural checks;
- surface anti-collapse diagnostics without folding them into training code.

Minimum interface:

- `RewardWeights`, `RewardTerm`, and `RewardBreakdown` dataclasses;
- `RewardContext` for one candidate reward computation;
- `build_reward_breakdown(context, weights)` for structured reward assembly;
- `aggregate_reward_terms(terms)` for scalar aggregation;
- `analyze_batch_collapse(outputs)` and `collapse_penalty_from_diagnostics(...)`.

#### `src/models/`

Responsibility:

- define the generation contract between prompts and fragment outputs;
- stay backend-agnostic so the same interface can wrap local checkpoints or HF models later.

Minimum interface:

- `GenerationRequest` dataclass;
- `GenerationResult` dataclass;
- `FragmentGenerator` protocol.

#### `src/train/`

Responsibility:

- define stage-level training contracts without implementing optimization yet;
- keep diagnostics first-class so RL collapse signals are visible from day one.

Minimum interface:

- `TrainStage` enum for format SFT, weak-supervision SFT, and counterfactual RL;
- `TrainingRunRequest` and `TrainingStatus` dataclasses;
- `Trainer` protocol;
- `TrainingDiagnosticsSnapshot` dataclass.

#### `src/eval/`

Responsibility:

- define checkpoint evaluation outputs independently of training code;
- standardize metric computation and reporting for structural and counterfactual views.

Minimum interface:

- `EvaluationExample` and `EvaluationSummary` dataclasses;
- `Evaluator` protocol;
- `safe_rate(...)` and `mean_metric(...)` helpers;
- `render_summary(summary)` formatter.

#### `src/utils/`

Responsibility:

- hold reusable generic helpers that are not themselves chemistry or reward logic;
- support reproducibility, IO, and logging across local and HPC runs.

Minimum interface:

- `read_jsonl(path)` and `write_jsonl(path, rows)`;
- `ensure_directory(path)`;
- `RunContext` dataclass and `get_logger(name)`;
- `set_global_seed(seed)`.

---

## Phase 2: Chemistry utility layer

Objective:

- build reliable molecule and fragment utilities before model training.

Modules to implement first:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

Target capabilities:

- parse SMILES safely;
- sanitize molecules;
- canonicalize fragment strings where appropriate;
- test whether fragment is a parent substructure;
- test whether fragment is connected;
- perform fragment deletion or approximate deletion logic;
- report failure types clearly.

Success criteria:

- chemistry checks are deterministic and testable;
- training code does not need to reimplement chemistry logic inline.

---

## Phase 2.5: Local/HPC runtime adaptation layer

Objective:

- make the modular repository runnable in both local development and HPC settings;
- keep all path handling config-driven and repository-relative;
- support single-machine or single-node single-GPU execution only for now.

Modules and files:

- `configs/base.yaml`
- `configs/local.yaml`
- `configs/hpc.yaml`
- `configs/sft.yaml`
- `configs/rl.yaml`
- `configs/eval.yaml`
- `src/utils/paths.py`
- `src/utils/env.py`
- `src/utils/logging_utils.py`
- `src/utils/seed.py`
- `scripts/run_sft.py`
- `scripts/run_rl.py`
- `scripts/run_eval.py`
- `scripts/run_infer.py`
- `scripts/slurm/*.slurm`

Target capabilities:

- merge stage and environment configs deterministically;
- resolve all runtime paths without hardcoded absolute paths;
- support local model and tokenizer paths;
- create per-run log and manifest directories;
- provide Slurm templates for single-node single-GPU jobs;
- keep CLI entrypoints thin and compatible with later training logic.

Success criteria:

- a local or HPC user can prepare a run from config and CLI only;
- scripts save a resolved manifest for reproducibility;
- the runtime layer does not assume distributed training.

---

## Phase 3: Reward subsystem

Objective:

- implement reward logic as a standalone subsystem.

Suggested files:

- `src/rewards/types.py`
- `src/rewards/counterfactual_reward.py`
- `src/rewards/anti_collapse.py`
- `src/rewards/aggregation.py`

Target capabilities:

- compute individual reward terms;
- return structured reward breakdowns;
- support configurable weights;
- expose penalties for collapse patterns;
- isolate counterfactual scoring from train-loop code.

Success criteria:

- reward logic is testable outside RL training;
- each term has a clear name, meaning, and expected range.

---

## Phase 4: Data and prompt subsystem

Objective:

- build clean dataset loaders and prompt builders.

Suggested files:

- `src/data/schemas.py`
- `src/data/jsonl_dataset.py`
- `src/data/prompts.py`
- `src/data/collators.py`

Target capabilities:

- read raw dataset JSONL;
- validate required fields;
- construct SFT and RL prompts consistently;
- support separate train/eval/test splits;
- keep prompt format versioned and documented.

Success criteria:

- data loading is deterministic;
- prompt generation is centralized rather than duplicated.

---

## Phase 5: Inference baseline

Objective:

- implement the simplest full-path runnable workflow.

Suggested entrypoint:

- `scripts/infer_single.py`

Target capabilities:

- load tokenizer/model/checkpoint;
- take one SMILES as input;
- produce one fragment output;
- run structural validation;
- save interpretable results.

Success criteria:

- one can test the contract “parent SMILES → fragment SMILES” before any large-scale training.

Status:

- minimal heuristic single-sample inference implemented on 2026-04-10 in `scripts/run_infer.py` and `src/eval/inference.py`
- trained-model inference remains a later step

---

## Phase 6: SFT subsystem

Objective:

- implement Stage A and Stage B supervised fine-tuning.

Suggested files:

- `src/train/train_sft.py`
- `scripts/train_sft.py`

Target capabilities:

- format-oriented SFT;
- weak-supervision SFT;
- config-driven hyperparameters;
- periodic evaluation;
- checkpoint saving.

Success criteria:

- the model learns to output structured fragment candidates with low parse failure rate.

---

## Phase 7: RL subsystem

Objective:

- implement Stage C RL for counterfactual optimization.

Suggested files:

- `src/train/train_rl.py`
- `src/train/rollout.py`
- `src/train/logging.py`
- `scripts/train_rl.py`

Target capabilities:

- policy rollout;
- reward computation and aggregation;
- KL/reference policy control;
- checkpointing;
- heartbeat logging for HPC runs;
- periodic validation.

Success criteria:

- RL training is stable enough to monitor;
- reward terms and failures are observable;
- obvious collapse is surfaced quickly.

---

## Phase 8: Evaluation subsystem

Objective:

- build a standalone evaluation path.

Suggested files:

- `src/eval/metrics.py`
- `src/eval/run_eval.py`
- `src/eval/reporting.py`
- `scripts/eval_model.py`

Target capabilities:

- run structural metrics;
- run deletion-based counterfactual metrics;
- collect qualitative examples;
- compare checkpoints;
- save machine-readable reports.

Success criteria:

- model quality can be assessed independently of training scripts.

---

## Phase 9: Testing and reproducibility

Objective:

- add the minimum research-grade reliability layer.

Suggested tests:

- chemistry parser test;
- substructure match test;
- deletion behavior test;
- reward term test;
- prompt formatting test;
- inference smoke test.

Suggested reproducibility measures:

- config snapshots;
- saved CLI commands;
- seed logging;
- environment notes.

Success criteria:

- changes can be checked without rerunning the entire project blindly.

---

## 4. Immediate Build Order

When starting from zero, the first concrete implementation order should be:

1. create the folder skeleton and typed module boundaries;
2. freeze the prompt and JSONL schema contracts;
3. implement RDKit-backed parsing, connectivity, substructure, and deletion in `src/chem/`;
4. implement reward term calculators on top of `src/chem/`, keeping counterfactual scoring explicit;
5. implement single-example inference using `src/models/`, `src/data/`, and `src/chem/`;
6. extend dataset and collator support for SFT and RL-specific batching;
7. implement Stage A and Stage B SFT entrypoints while preserving output-only-SMILES behavior;
8. implement Stage C RL entrypoints with reward breakdown logging and anti-collapse diagnostics;
9. implement standalone evaluation and checkpoint comparison;
10. expand tests from interface smoke tests to chemistry, reward, and inference coverage.

### Immediate next implementation steps after this bootstrap

1. Replace the chemistry placeholders in `src/chem/` with deterministic RDKit-backed implementations.
2. Wire `src/rewards/counterfactual_reward.py` to real structural checks and deletion-based flip scoring.
3. Implement `scripts/infer_single.py` as the first runnable end-to-end contract.
4. Add versioned config files under `configs/` once interfaces stop moving.

---

## 5. Risk Register

### Risk 1: Objective drift

The project may accidentally revert to concept extraction or rationale extraction.

Mitigation:

- keep the objective explicit in docs and comments;
- ensure evaluation includes deletion-based flip metrics.

### Risk 2: RL instability

The policy may collapse during RL.

Mitigation:

- expose per-term reward logging;
- control KL;
- monitor repeated-token behavior;
- save representative outputs periodically.

### Risk 3: Chemistry utility inconsistency

If chemistry logic is duplicated across files, behavior will drift.

Mitigation:

- centralize RDKit-related logic in `src/chem/`.

### Risk 4: Overcoupled scripts

A monolithic script will be hard to debug.

Mitigation:

- keep scripts thin and modules cohesive.

---

## 6. Definition of “Good First Version”

A good first rebuilt version should support the following end-to-end workflow:

1. load a JSONL molecule dataset;
2. construct prompts;
3. run model inference for one sample;
4. check whether output is valid and connected;
5. verify whether it is a parent substructure;
6. compute reward breakdown for a candidate;
7. run a minimal training/evaluation command.

If these are achieved with clean module boundaries, the rebuild is on the right path.

---

## 7. BACE WNode Prefix Optimization Extension

The BACE paper path now has an additive, versioned optimization route:

1. audit frozen rank and coverage funnels;
2. precompute a calibration-only WNode action matrix;
3. compare frozen selector variants with grouped calibration CV;
4. conditionally expand candidates only when the calibration limitation gate
   identifies a candidate-limited pool;
5. freeze one rank-preserving Top20 sequence;
6. run one new test evaluation and a non-promoting paper artifact audit.

This extension reuses the production evaluator and does not alter the AIDS or
Mutagenicity roadmaps.
# BACE Connected Candidate-Aware v4 (2026-08-10)

- [x] Preserve legacy matrix admission as an explicit default policy.
- [x] Add chemistry-only `connected_feasible_v4` candidate admission.
- [x] Add 151-to-matrix attrition and cross-dataset threshold protocol audits.
- [x] Add a versioned full connected calibration matrix wrapper and union report.
- [ ] Freeze a method-independent pooled calibration threshold contract.
- [ ] Run the preregistered connected-aware generation rounds only when the
  calibration union gate reports candidate limitation.
- [x] Add an opt-in connected-deletion prompt and source-side chemistry gate.
- [x] Add fixed Round-1 generation, merge, and calibration-matrix wrappers.
- [x] Add a complete native-rank GCF attrition audit that preserves Top20.
- [x] Add the calibration-only hard-parent Round-2 cohort and fixed regimes.
- [x] Add method-balanced pooled calibration Q30/Q50 threshold freezing.
- [x] Add pre-test selection/protocol gates and a one-shot Ours/GCF test job.
- [ ] Freeze Ours/GCF calibration selections before exactly one v4 test run.

# Storage-Safe Two-Lane Recovery (2026-08-17)

- [x] Add persistent-scratch and SQLite WAL preflight checks.
- [x] Add fail-closed projected-capacity monitoring for COMRECGC full walks.
- [x] Unify AIDS validator/recovery frozen graph and alias closure.
- [x] Load BACE GCF thresholds exclusively from a frozen shared manifest.
- [x] Add resumable, deterministic GlobalGCE root/epoch checkpoints without
  changing the official mining or training objective.
- [x] Add current-queue GPU accounting and a static two-lane plan validator.
- [x] Make checkpoint, integrity, chemistry, gate, and freeze wrappers CPU-only.
- [ ] Run HPC scratch/checkpoint/plan gates from the committed recovery worktree.
- [ ] Submit fresh MUT retry8 and BACE retry chains with one GPU per lane.
- [ ] Complete the downstream connected four-method artifact audits.

# Three-Line Recovery v7 (2026-08-18)

- [x] Persist complete AIDS original-hash, alias, transition, frontier, and
  recourse closure requirements across payload reload.
- [x] Add a fail-closed COMRECGC resume-or-finalize decision and two-slice BACE
  continuation wrapper.
- [x] Classify BACE GlobalGCE native candidates as full counterfactual graphs
  and remove the deletion-fragment matrix adaptation.
- [x] Spill low-support gSpan reports to resumable scratch SQLite and preserve
  official stable support top-k semantics.
- [x] Include the live MUT and BACE allocations in a two-GPU project planner.
- [ ] Revalidate and recover the completed AIDS walk on CPU only.
- [ ] Recompute GlobalGCE calibration matrices for min-freq 18/7/4 and stream
  min-freq 2 without a GPU.
- [ ] Complete GlobalGCE and COMRECGC BACE artifacts, then run common4 audit.

# AutoDL Four-Lane Three-Line Recovery (2026-08-21)

- [x] Add a fixed-run, data-driven four-lane AutoDL process orchestrator.
- [x] Keep persistent run state, logs, process provenance, heartbeats, locks,
  and scientific success/failure sentinels outside the disposable NVMe root.
- [x] Reserve the fast root for lane-local caches and the BACE active state.
- [x] Enforce one independent GPU, PID file, writer lock, cache, input, output,
  and command per lane; never represent an AutoDL PID as a Slurm job id.
- [x] Require `DISALLOW_GENERATION=1` for every preserved MUT/AIDS stage.
- [x] Gate BACE common4 on both BACE COMRECGC final and GlobalGCE WNode
  scientific success sentinels.
- [x] Provide one persistent `start/status/resume/stop` interface and reject
  an unconfigured command, writable input snapshot, unknown dependency,
  second writer, or untracked nonempty first-start output root.
- [x] Fill the exact recovery/downstream commands and add the production stage
  runner plus paired Slurm wrapper.
- [x] Add immutable primary/static/Step0 input gates, SHA-bound atomic stage
  sentinels, and fail-closed partial-output handling.
- [x] Add mandatory MUT/AIDS preserved-lineage smoke gates and a BACE
  formal-configuration SIGKILL/fast-loss profile gate bound to repair content.
- [x] Persist BACE trace chunks and latest-two checkpoint mirrors, and require
  the BACE artifact gate before common4 publication.
- [x] Bind every reusable substage and top-level sentinel to current input,
  command, environment, code/config closure, vendor commit, marker, and output
  SHA evidence; reject marker-only crash windows and stale outputs.
- [x] Bind local children to kernel start time, command digest, and process
  group; never signal a stale or reused PID, and scrub inherited credentials.
- [x] Support fail-closed incremental `start/resume --lane` activation while
  keeping omitted lanes `NOT_STARTED` and preserving no-flag four-lane launch.
- [x] Bind worker identity and persisted run/lane state to exact process,
  spec-byte digest, schema, run, lane, and normalized roots.
- [x] Require persisted stage and producer-lane success in addition to
  scientific dependency proofs, with sentinel-first success publication.
- [x] Provide Python-3.10/glibc-compatible exact pidfd signalling, cooperative
  stop markers when the kernel lacks pidfd support, and fail-closed manual
  handling instead of persisted-orphan `killpg` targeting.
- [x] Make completed-walk freeze recovery mirror the live global
  first-recorded predecessor index, with exact replay for every repeated event
  and explicit alias/convergence/conflict audit counters.
- [x] Treat frozen global-hash graph parent metadata as audit-only and bind
  recovered lineage ownership to a parent-consistent selected-event chain.
- [x] Strictly remap a uniquely determined recorded NLC representative index,
  separate selected-transition and unique-candidate recovery counts, and emit
  a checksum-bound fresh-root adoption manifest.
- [ ] Pass the AutoDL integration smokes, then start the four formal lanes.

# BACE and TasteMolNet Frozen-GNN Route (2026-08-22)

- [x] Audit the BACE frozen split and classify legacy oracle provenance.
- [x] Replace BBBP with TasteMolNet in the active dataset contract while
  preserving historical artifacts.
- [x] Add the generic molecular GNN registry, checkpoint bundle, calibrated
  batched oracle API, and BACE/Taste RF guard.
- [x] Add shared binary/multiclass strict-flip, CFDrop, margin, and destination
  distribution semantics.
- [x] Prepare the fixed-commit TasteMolNet upstream data with conflict,
  standardization, license, and scaffold-leakage audits.
- [x] Add the bounded AutoDL GPU inventory/lock/experiment registry and BACE
  gated state machine; keep `RUN_TASTEMOLNET=0` by default.
- [x] Move the AutoDL control plane outside fast code worktrees, freeze the
  persistent control root and `smiles_pip118` interpreter into detached specs,
  and add predecessor-bound B4 temperature/B5 oracle-smoke launchers.
- [x] Make GNN training hold test fully unopened, freezing only path/SHA
  provenance and an explicit `NOT_EVALUATED` status until final evaluation.
- [ ] Execute and pass B4 on validation and B5 on the 16-parent correctly
  predicted source cohort from calibration after B3 passes.
- [ ] Pass CPU/PyG tests and BACE GINE smoke, then launch the single seed-7
  BACE full classifier on an idle AutoDL GPU.
- [x] Add an honest B6 calibrated-GNN scoring diagnostic that leaves the PPO
  stage BLOCKED, plus executable B7--B14 blocker contracts; never label that
  diagnostic PPO or use it to release B7.
- [x] Add fail-closed initializer provenance, a fresh raw-base LoRA and bounded
  train-only oracle-neutral SFT path for BACE; reject unknown/RF adapters.
- [x] Inject one cached/batched frozen GINE reward adapter into the existing
  stable decoded-chemistry PPO loop without adding a second optimizer stack.
- [x] Add a real fresh-root B6-v2 five-update gate and a conservative 300-step
  B7 contract with checkpoints 50--300; retain old B6 blocker evidence.
- [x] Add explicit B6--B14 split-access and dependency-release contracts that
  cannot turn READY evidence into a scientific PASS.
- [ ] Run and pass the real 7B LoRA B6-v2 on AutoDL, then release B7.
- [x] Decouple the non-formal adapter canary from stochastic generated-deletion
  yield by adding a same-adapter, eight-parent train-only connected-deletion
  GNN preflight; keep formal B6 dependent on PPO-generated deletion evidence.
- [x] Implement the provenance-clean B8/B9 fixed train-parent shards, B10
  deterministic merge, batched-GINE/all-match/MolCLR-WNode B11 and B13
  verification, calibration-only B12 freeze, and manifest-only B14 gate.
- [x] Add B6-released B7-parallel calibration caches, fixed shard manifests,
  output preflight, and a foreground command/output contract for the AutoDL
  four-GPU controller.
- [ ] Execute B6-v2 and B7, then let the controller advance the implemented
  B8--B14 route without using RF-contaminated artifacts or pre-freeze test data.
- [ ] Obtain explicit TasteMolNet data-license approval before committing data
  or enabling any heavy TasteMolNet experiment.
- [x] Add a manifest-driven, persistent four-GPU AutoDL recovery controller
  that reuses `exp_run`, UUID locks, atomic gates, and append-only registry
  semantics; include deterministic train/calibration sharding and gated
  four-shard B13 held-out evaluation.
- [x] Bind existing Commit-A writers by exact launch-spec provenance, publish
  the user-facing JSONL/Markdown registry mirrors, and keep execution clones
  free of Python bytecode writes.
- [x] Allow the exact scheduler-owned `TOKENIZERS_PARALLELISM` environment key
  through `exp_run` without weakening credential-like environment rejection.
- [x] Integrate the Frozen-GNN downstream foreground contract into the
  controller with passing-attempt shard tokens, dependency-produced parent
  manifests, explicit B11/B13 shard-to-merge joins, and a post-B12-only test
  boundary.
- [x] Add a fresh B11--B14 continuation builder that exact-adopts B6--B10,
  flattens eight historical B8/B9 shard runs into single-instance evidence
  tasks, and substitutes the passing MolCLR repair without mutating v2.
- [x] Hold the predecessor controller lock for the full continuation lifetime
  and support an opt-in terminal heartbeat/poll mode without dummy work.
- [ ] Fill the persistent controller manifest with Commit A MUT/AIDS and Commit
  B BACE foreground argv/evidence contracts, validate it, and launch on AutoDL.
# 2026-08-22 — Four-by-four main experiment continuation

- Keep TasteMolNet heavy stages behind the new offline license gate. Continue
  BACE and adoptable AIDS/Mutagenicity work while that independent gate is
  blocked; do not synthesize a Taste row in the paper matrix.
- A later explicit approval can release the existing three-class pipeline
  without changing the dataset, label semantics, or frozen evaluation schema.
## 2026-08-22 — Four-method matrix continuation

- [x] Add a fresh-root, PASS-last continuation from immutable AIDS and
  Mutagenicity COMRECGC recovery payloads into standardized Figure 3/Figure 4/
  Table 2 artifacts.
- [ ] Register the resulting cells only after their project-full gate and freeze
  manifests pass the common 4×4 provenance contract.
- [ ] Schedule the two dataset continuations independently so a semantic failure
  in one dataset releases its GPU without blocking the other dataset.
- [x] Add isolated persistent controller/status/launcher entrypoints for the
  four-method matrix while retaining the audited UUID-lock scheduler engine.
- [x] Add a fresh, exact-path core task-fragment builder for the Taste license
  audit, four blocked Taste cells, and recovered AIDS/Mutagenicity COMRECGC
  standardization without generation reruns.
- [x] Preserve BACE continuation predecessor-lock metadata when composing the
  wider four-by-four controller manifest.

## BACE native baseline Frozen-GINE continuation (2026-08-22)

- [x] Add one calibrated-GINE adapter for official GCFExplainer and ComRecGC
  one-hot graph edit runtimes.
- [x] Freeze train-only GINE-clean native candidate universes without RF
  ranking or calibration/test access.
- [x] Add four-shard native full-graph GINE/WNode verification, calibration-only
  selector freeze, held-out evaluation, and PASS-last final freeze.
- [x] Publish controller-readable resource/marker contracts and AutoDL commands.
- [x] Add a one-way native-to-generic fragment adapter with passing-attempt
  dependency tokens, non-primary runner datasets, baseline-specific
  test-after-selector gates, and a bounded CPU GlobalGCE native-action
  preflight followed by a static no-resource training block.
- [x] Implement and validate pinned-official-parity GlobalGCE attachment-aware
  LHS→RHS application plus exact frozen-GINE forward scoring.
- [ ] Design and scientifically approve a differentiable bridge from the
  official continuous RHS decoder to the frozen RDKit/categorical GINE input;
  until then keep BACE GlobalGCE full training fail-closed as
  `BLOCKED_GLOBALGCE_FROZEN_GINE_DIFFERENTIABLE_RULE_TRAINING_UNAVAILABLE`.

# Four-method × four-dataset paper matrix (2026-08-22)

- [x] Add an exact 16-cell registry with a closed state enum.
- [x] Add bounded, read-only multi-root artifact inventory and explicit
  candidate-root support.
- [x] Keep CLEAR distinct from ComRecGC and prevent render-only legacy CSVs from
  becoming paper PASS evidence.
- [x] Cross-check top-level continuation gates against nested standardized
  freeze manifests without inferring dataset or method from directory names.
- [x] Gate adoption on dataset/test/oracle/distance/threshold provenance and
  cross-method identity parity within each dataset.
- [x] Emit the unified WNode/strict-flip/K=1..20/Table2-K10 evaluation and
  standardized-export contract without synthesizing missing metrics.
- [x] Emit per-dataset evaluator-ready threshold contracts only from explicit
  calibration-frozen provenance; omit numerics when the contract is missing or
  test-derived.
- [x] Keep TasteMolNet blocked unless an explicit exact-data reuse basis passes.
- [ ] Populate the registry on AutoDL from persistent artifacts and exact
  expectations, then let the continuation controller schedule only missing or
  stale cells.
- [x] Add strict, fresh-root adoption for the exact frozen Mutagenicity Ours
  final result, including checksum closure, independent frozen-test replay
  audit, RF/MolCLR identity, and selector-before-test provenance.
- [x] Inventory the remaining AIDS/Mutagenicity legacy raw cells without
  rerunning generation; keep missing calibration/evaluation evidence
  `INCOMPLETE` and missing native GlobalGCE attachment semantics
  `BLOCKED_CODE`.
- [x] Freeze the exact Mutagenicity GCF Top20 exporter output into a fresh
  checksum-closed candidate package without rerunning generation; keep AIDS
  baseline roots absent from the AutoDL payload explicitly `MISSING`.
- [x] Register persistent Mut GCF freeze -> calibration selector freeze ->
  held-out evaluation tasks and extend the production no-test-before-freeze
  stage gate for that AM route.
- [x] Freeze and hash-check the shared AIDS/Mut 601-point threshold contract
  before either ComRecGC held-out continuation, with production controller
  enforcement of the selector-before-test dependency.
- [x] Track the matched 601-point Mutagenicity threshold expectation; mark the
  audited historical Ours bundle `STALE_METRIC` until its frozen pair matrix is
  deterministically re-exported under that common protocol.
- [x] Deterministically re-export the frozen Mutagenicity Ours 217-by-20 pair
  bundle on the matched 601-point protocol with no generation, selector,
  oracle, or MolCLR rerun and a fresh PASS-last standardized freeze.
- [x] Exclude CLEAR from every adoption and inventory path and provide distinct
  static blocked tasks for every known missing A/M terminal cell.
# 2026-08-22 — Deferred main-result ablation hooks

- [x] Keep a single registered GNN backbone axis (`gine`, `gin`, `gcn`,
  `gatv2`) and a dormant AutoDL task-plan builder.
- [x] Add a frozen-rule stability comparator covering exact, chemical,
  scaffold, coverage, and multiclass destination overlap.
- [ ] Run backbone/selector/reward/candidate-pool ablations only after the
  primary four-by-four matrix is scientifically unblocked and frozen.
- [x] Add a presentation-only Figure 3/Figure 4/Table 2 exporter that requires
  16/16 registry PASS plus per-cell hash/provenance closure, rejects CLEAR,
  preserves raw thresholds and Taste destinations, and emits no plausible
  numeric output for a partial matrix.
- [x] Add a generic CPU controller fragment whose final export task depends on
  16 distinct cell terminal PASS tasks and one post-cell matrix audit.
- [ ] Run the final exporter only after Taste licensing and every code gate
  release the full 16-cell matrix.
# 2026-08-22 — Final matrix dependency closure

- [x] Bind the post-cell registry audit to sixteen distinct successful attempt
  roots using controller dependency tokens.
- [x] Generate the strict final-export dependency contract from the same cell
  mapping, preventing a reporting task from silently changing cell identity.
