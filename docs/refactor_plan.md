# Refactor Plan

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
