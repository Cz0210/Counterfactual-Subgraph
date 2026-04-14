# Counterfactual Subgraph Generation v3

This repository implements a counterfactual subgraph generation framework for molecular graphs represented as SMILES.

The goal is to train a language model that takes one parent molecule SMILES as input and outputs one connected fragment SMILES such that removing this fragment from the parent molecule is likely to flip the molecule label.

This repository is rebuilt from scratch with the following priorities:

1. Preserve the **counterfactual objective** rather than ordinary rationale extraction or concept extraction.
2. Build a **modular and maintainable codebase** suitable for iterative research.
3. Support **SFT + RL** training, validation, checkpoint selection, inference, and evaluation.
4. Provide **strong diagnostics** for degeneration, reward collapse, and invalid fragment generation.
5. Keep the project easy to operate in **VS Code + Codex + HPC** workflows.

---

## 1. Project Objective

Given a molecule SMILES `x` and a binary label `y`, train an LLM to generate a fragment `g` such that:

- `g` is a valid SMILES;
- `g` is a connected substructure of `x`;
- deleting `g` from `x` produces a residual molecule `x \ g`;
- the predicted label of `x \ g` is likely to flip relative to `y`.

The project therefore studies **counterfactual fragment generation**, not standard explanation extraction.

---

## 2. Core Principles

### 2.1 What this project is

This project is about learning a generator that produces **counterfactual fragments**.

A successful fragment should satisfy both:

- **structural correctness**: it must be chemically and graph-theoretically reasonable;
- **counterfactual effectiveness**: deleting it should strongly affect the label.

### 2.2 What this project is not

This project is **not**:

- ordinary subgraph extraction;
- concept-subgraph extraction;
- rationale extraction whose purpose is only to preserve the original prediction;
- a task where the fragment alone should predict the original label.

If any old code path optimizes toward these non-counterfactual objectives, it should be treated as outdated behavior and rewritten or isolated.

---

## 3. Planned Repository Layout

```text
.
├── AGENTS.md
├── README.md
├── docs/
│   ├── cf_subgraph_v3_spec.md
│   ├── refactor_plan.md
│   └── decisions.md
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── reward/
│   └── eval/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── scripts/
│   ├── prepare_data.py
│   ├── run_sft.py
│   ├── run_rl.py
│   ├── run_eval.py
│   ├── run_infer.py
│   ├── train_sft.py
│   ├── train_rl.py
│   ├── eval_model.py
│   ├── infer_single.py
│   └── slurm/
├── src/
│   ├── data/
│   ├── models/
│   ├── rewards/
│   ├── train/
│   ├── eval/
│   ├── chem/
│   └── utils/
├── tests/
└── outputs/
```

This layout is only the initial target; implementation may proceed incrementally.

---

## 4. Planned Methodology

The overall pipeline is divided into three stages.

### Stage A: Format and topology-oriented SFT

The first stage teaches the model to output clean, valid, connected fragment-like SMILES under a controlled prompt format.

Typical goals:

- output only SMILES and no extra text;
- reduce instruction leakage;
- improve basic topology awareness;
- reduce parse failures.

### Stage B: Weakly supervised fragment SFT

The second stage uses weakly constructed fragment targets to further align the model with valid parent-substructure generation.

Typical goals:

- make the fragment a valid substructure of the parent molecule;
- maintain connectedness;
- teach the model the expected output distribution before RL.

### Stage C: RL for counterfactual optimization

The third stage performs reinforcement learning so that generation is optimized toward the true project objective: **counterfactual label flipping after deletion**.

Typical goals:

- maximize flip-related reward;
- maintain validity and substructure constraints;
- reduce policy collapse;
- control KL drift;
- keep generation diverse and chemically plausible.

---

## 5. Initial Engineering Priorities

When implementing the project from scratch, prioritize the following order:

1. establish clean data schemas and config files;
2. build deterministic utilities for SMILES parsing and fragment checking;
3. implement the inference contract for one parent SMILES → one fragment SMILES;
4. implement reward computation as independent, testable modules;
5. implement SFT entrypoints;
6. implement RL entrypoints;
7. implement evaluation and logging;
8. implement collapse diagnostics and best-checkpoint selection.

---

## 6. Expected Data Format

The minimal training/evaluation JSONL format is:

```json
{"id": 2, "smiles": "CC1(Cl)C(=O)NC(=O)NC1O", "label": 1}
```

Additional derived files may include prompts, weak targets, reward annotations, and evaluation outputs.

---

## 7. Immediate Next Steps

1. Finalize `docs/cf_subgraph_v3_spec.md` as the algorithmic source of truth.
2. Use `AGENTS.md` to instruct Codex how to behave in this repository.
3. Create `configs/`, `src/`, `scripts/`, and `tests/` step by step.
4. Implement a minimal chemistry utility layer first.
5. Then add SFT, RL, evaluation, and inference progressively.

---

## 8. Development Notes

- Prefer incremental refactoring over monolithic scripts.
- Preserve command-line usability for HPC training.
- Log enough information to diagnose collapse, especially repeated-token degeneration such as long sequences of `N`.
- Keep reward semantics explicit and testable.

---

## 9. Runtime Configuration

The repository now includes config-driven runtime entrypoints for local development
and Slurm-based HPC usage.

Primary config files:

- `configs/base.yaml`: shared defaults
- `configs/local.yaml`: local-machine overrides
- `configs/hpc.yaml`: HPC overrides for single-node single-GPU runs
- `configs/sft.yaml`: SFT-stage overrides
- `configs/rl.yaml`: RL-stage overrides
- `configs/eval.yaml`: evaluation-stage overrides

Config merge order for the new run scripts is:

1. `configs/base.yaml`
2. `configs/{local|hpc}.yaml`
3. one stage config such as `configs/sft.yaml`
4. any extra `--config ...` files
5. any CLI overrides such as `--model-path`, `--output-root`, or `--set section.key=value`

All paths in config files should stay relative to the repository root. The scripts
resolve them at runtime and save the resolved absolute paths into
`resolved_config.json` inside each run directory.

---

## 10. Local Development

Local runtime preparation examples:

```bash
python3 scripts/run_sft.py \
  --environment local \
  --run-name dev_sft \
  --model-path models/my_local_model \
  --tokenizer-path models/my_local_model \
  --print-config

python3 scripts/run_infer.py \
  --environment local \
  --model-path models/my_local_model \
  --tokenizer-path models/my_local_model \
  --smiles "CCO" \
  --label 1 \
  --print-config
```

The current runtime layer already supports a minimal single-example inference loop
in `scripts/run_infer.py`: it produces one heuristic fragment candidate, runs
chemistry checks, and prints a structured JSON result. Full SFT, RL, and standalone
evaluation execution logic are still incremental and not yet implemented in these
run scripts.

---

## 11. Local Model And Tokenizer Paths

Model and tokenizer loading is designed for local filesystem paths.

- Use `--model-path` to point to a local checkpoint directory.
- Use `--tokenizer-path` to point to a local tokenizer directory. If omitted in a
  future direct loader call, the model path may also be used as the tokenizer path.
- Set `--load-model` if you want the run script to attempt an actual local load.

The implementation intentionally keeps `local_files_only=true` by default in the
runtime config so that local development and HPC jobs do not silently depend on
network downloads.

---

## 12. HPC Usage

The HPC layer currently targets:

- local development on one machine
- Slurm jobs on one node with one GPU

It does not implement distributed training yet.

Example Slurm entrypoints:

- `scripts/slurm/train_sft.slurm`
- `scripts/slurm/train_rl.slurm`
- `scripts/slurm/eval.slurm`

Example submission:

```bash
sbatch scripts/slurm/train_sft.slurm
sbatch scripts/slurm/train_rl.slurm
sbatch scripts/slurm/eval.slurm
```

Each Slurm script derives the repository root relative to its own location and does
not hardcode machine-specific absolute paths. Override local artifact locations with
environment variables such as `MODEL_PATH`, `TOKENIZER_PATH`, and `CHECKPOINT_PATH`
before submission if needed.

---

## 13. Suggested First Commands

Once the repository skeleton is created, a reasonable next sequence is:

```bash
mkdir -p docs configs/data configs/model configs/train configs/reward configs/eval
mkdir -p data/raw data/processed data/splits
mkdir -p scripts src/data src/models src/rewards src/train src/eval src/chem src/utils tests outputs
```

After that, start by implementing:

- `src/chem/smiles_utils.py`
- `src/rewards/counterfactual_reward.py`
- `scripts/infer_single.py`

These three pieces define the earliest stable backbone of the project.
