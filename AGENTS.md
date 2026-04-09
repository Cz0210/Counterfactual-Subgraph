# AGENTS.md

## 1. Repository Role

This repository implements a molecular counterfactual subgraph generation system based on SMILES.

The core task is:

> Given a parent molecule SMILES and its label, generate one connected fragment SMILES such that removing this fragment from the parent is likely to flip the label.

This repository must be developed with this counterfactual objective as the highest priority.

---

## 2. Source of Truth

Before making any code change, always read the following files in order:

1. `README.md`
2. `docs/cf_subgraph_v3_spec.md`
3. `docs/refactor_plan.md`
4. `docs/decisions.md`

If implementation and documentation conflict, prefer `docs/cf_subgraph_v3_spec.md` and document the discrepancy.

---

## 3. Task Definition

The project is **counterfactual fragment generation**, not concept extraction.

The generated fragment must ideally satisfy all of the following:

1. valid SMILES syntax;
2. chemically parseable;
3. connected fragment;
4. valid substructure of the parent molecule;
5. deleting it is likely to flip the molecule label;
6. output should contain SMILES only and no explanation text.

Any implementation that instead optimizes for “fragment alone predicts original label” should be treated as misaligned with the current project objective.

---

## 4. Required Development Style

When modifying this repository:

- prefer modular code over monolithic files;
- keep training, evaluation, reward, chemistry utilities, and data logic separated;
- favor explicit dataclasses, typed configs, and small interfaces;
- preserve command-line usability;
- write code that can run in HPC environments;
- add logging that helps diagnose instability and collapse.

---

## 5. Required Project Structure

The intended structure is:

```text
src/
  data/
  models/
  rewards/
  train/
  eval/
  chem/
  utils/
scripts/
configs/
tests/
docs/
```

As code is built from scratch, keep file responsibilities clear.

Examples:

- `src/chem/`: RDKit-related parsing, substructure matching, deletion logic, connectivity checks.
- `src/rewards/`: reward terms and reward aggregation.
- `src/train/`: train loops, callbacks, checkpoint logic.
- `src/eval/`: evaluation metrics and reporting.
- `src/data/`: dataset reading, prompt construction, collators.
- `scripts/`: thin CLI entrypoints only.

---

## 6. Hard Constraints

Do not do the following without explicit justification and documentation:

1. Do not redefine the task as concept-subgraph extraction.
2. Do not optimize only for fragment validity while ignoring counterfactual effect.
3. Do not silently change JSONL schemas.
4. Do not bury reward logic inside a giant training script.
5. Do not remove logging needed for diagnosis.
6. Do not produce unstructured one-file pipelines if modular alternatives are possible.
7. Do not hide breaking behavior changes.

---

## 7. Implementation Priorities

When starting from an empty repository, use this order:

1. establish data schema and configuration schema;
2. implement chemistry utilities;
3. implement reward modules;
4. implement single-example inference pipeline;
5. implement SFT pipeline;
6. implement RL pipeline;
7. implement validation and checkpoint selection;
8. implement tests and reporting.

---

## 8. RL-Specific Guidance

Because this project previously suffered from collapse and unstable RL behavior, all RL implementations should expose diagnostics for:

- reward mean and per-term breakdown;
- valid/parseable/substructure rates;
- output length statistics;
- repeated-token collapse;
- KL behavior;
- representative generated examples;
- validation-time flip-related metrics.

Repeated outputs such as long sequences of a single token, especially repeated `N`, must be treated as severe degeneration and surfaced explicitly in logs.

---

## 9. Evaluation Requirements

Any implementation should eventually support reporting at least the following metrics:

- format success rate;
- parseable rate;
- valid chemistry rate;
- connected fragment rate;
- parent-substructure match rate;
- counterfactual flip rate;
- reward statistics;
- collapse diagnostics;
- qualitative example generations.

---

## 10. Documentation Policy

For every substantial code change:

- update `docs/decisions.md` with date, motivation, and impact;
- update `docs/refactor_plan.md` if the roadmap changes;
- keep examples and CLI usage in sync.

---

## 11. Completion Criteria

A task is not complete unless all applicable items are satisfied:

1. code is added or updated;
2. file responsibilities remain coherent;
3. relevant docs are updated;
4. command-line usage remains clear;
5. tests are added or adjusted where possible;
6. breaking changes are explicitly recorded.

---

## 12. How to Collaborate in This Repository

When asked to help with this project:

1. first summarize the current architecture;
2. identify risks and ambiguities;
3. propose an incremental plan;
4. only then implement changes;
5. keep changes scoped and reviewable.

If the task is ambiguous, do not invent a new research objective. Preserve the counterfactual v3 objective unless the user explicitly revises it.
