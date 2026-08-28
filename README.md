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

Given a molecule SMILES `x` and a class label `y`, train an LLM to generate a fragment `g` such that:

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

## Evaluation Protocol

The project currently uses native-action Close Counterfactual Rule Coverage (CCRCov) as the main cross-baseline evaluation metric. SuppCov is temporarily reserved for subgraph-specific auxiliary analysis. For details, see:

- `docs/EVALUATION_PROTOCOL.md`
- `docs/BASELINE_ADAPTATION.md`
- `docs/PROJECT_CONVENTIONS.md`
- `docs/BASELINE_GLOBALGCE.md`
- `docs/baselines/GCFEXPLAINER_OFFICIAL.md`
- `docs/CCRCOV_DISTANCE_REPRODUCTION.md`

---

## Experiment Tracking

Slurm jobs should be submitted through `scripts/exp_sbatch.py` or `scripts/exp_sbatch.sh` so that job id, command, git commit, environment snapshot, and expected output paths are automatically recorded.

See:

- `docs/EXPERIMENT_TRACKING.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/AUTODL_BACE_FROZEN_GNN_DOWNSTREAM.md` (foreground B7 prep and B8--B14)
- `docs/AUTODL_AM_LEGACY_STANDARDIZATION.md` (strict frozen-result adoption and
  deterministic matched-protocol re-export plus fail-closed
  AIDS/Mutagenicity legacy inventory)
- `docs/AUTODL_AIDS_COMRECGC_EXACT_SHORTCUT.md` (release-gated fresh adoption
  of a promoted read-only pair store, with closed Cartesian chunks only as a
  storage-audited fallback, exact adaptive DBSCAN proof, and bounded
  one-cluster replay; never a continuation inside the old repair-v4 root)
- `docs/AUTODL_AIDS_COMRECGC_DISCONNECTED_RECOVERY.md` (typed, CPU-only
  recovery from the c766 disconnected-anchor evidence into an exact
  multi-component partition and streaming downstream; release pins remain
  mandatory and the failed source is never treated as an ordinary PASS)
- `docs/AUTODL_TASTEMOLNET_T6_OURS_PPO.md` (release-disabled implementation
  contract for the real bounded three-class TasteMolNet Ours PPO smoke; it
  cannot run until a reviewed controller receipt and strict terminal consumer
  are integrated)
- `docs/AUTODL_MANAGED_EXECUTION_V2.md` (UUID attempt/checkpoint identities,
  launcher+worker lineage, quarantine-without-signals, SEALED worker evidence,
  independent verification, and atomic directory publication without mutable
  file links)
- `docs/AUTODL_TASTEMOLNET_T7_GCF_SMOKE.md` (release-disabled bounded native
  full-graph GCFExplainer smoke using the same calibrated three-class GINE;
  binary BACE adapters, NeuroSED claims, RF, test payloads, and molecule-bearing
  output are excluded)
- `docs/AUTODL_TASTEMOLNET_T8_GLOBALGCE_SMOKE.md` (release-disabled contract
  for the real bounded native GlobalGCE two-target smoke on physical GPU 2;
  the paired Slurm entrypoint is an intentional static refusal)
- `docs/TASTE_GCF_NEUROSED_PROTOCOL.md` (Taste-specific, train-fit/
  validation-select NeuroSED auxiliary distance model, official GREED/GCF
  runner semantics, no calibration/test access, and managed-v2 publication)

---

## Dataset Contract

Final experiments use the canonical AIDS/HIV dataset: `data/raw/AIDS/HIV.csv` with `smiles` and `HIV_active` columns. Internal names `hiv`, `hiv_quick`, and `aids` may appear in scripts, but final results must trace back to this canonical CSV. `ogbg_molhiv` is engineering validation only.

See `docs/DATASET_CONTRACT.md`.

The active paper experiment matrix is AIDS, Mutagenicity, BACE, and
TasteMolNet. AIDS and Mutagenicity retain their historical RF-backed contracts.
BACE and TasteMolNet use independent task-specific frozen GNN classifiers; RF
oracles are forbidden for those two active routes. TasteMolNet is a genuine
three-class Bitter/Sweet/Tasteless task with Sweet as the source class and an
untargeted strict flip to either other class. Historical BBBP artifacts are
preserved but BBBP is no longer in the active matrix. See
`docs/BBBP_TO_TASTEMOLNET_MIGRATION.md`.

The active 4 × 4 paper matrix has a fail-closed, read-only AutoDL registry at
`scripts/autodl/audit_four_methods_four_datasets.py`. It inventories multiple
output roots and writes the exact 16-cell status, oracle registry, frozen
evaluation contract, artifact inventory, stale-artifact list, and adoption
report without recomputing scientific results. See
`docs/AUTODL_FOUR_BY_FOUR_MATRIX_AUDIT.md`.

The exact checksum-pinned legacy v4 AIDS/Mutagenicity tables may be adopted for
only `Ours`, `GCFExplainer`, and `GlobalGCE` through
`scripts/autodl/adopt_user_approved_frozen_v4.py`. This is an explicit
`USER_APPROVED_FROZEN_V4` registry exception: numeric strings are copied from
the five approved CSV/JSON source files without scientific recomputation, while
missing legacy raw/provenance identities remain visibly unavailable. CLEAR is
excluded and is never relabeled as ComRecGC. See
`docs/AUTODL_USER_APPROVED_FROZEN_V4_ADOPTION.md`.

Final Figure 3, Figure 4, and Table 2 aggregation is separately gated by
`scripts/autodl/export_four_by_four_main_results.py`. It produces no numerical
or graphical final output until the registry and all standardized artifact
closures prove 16/16 eligible cells; incomplete campaigns receive a
non-numeric staging audit only. See
`docs/AUTODL_FOUR_BY_FOUR_MAIN_RESULTS_EXPORT.md`.

The partial three-dataset release has a separate persistent CPU-only
supervisor. It hash-binds all twelve standardized cell roots to their external
owner manifest/task/output contracts, remains non-numeric while fewer than
twelve cells pass, and publishes only runtime staging artifacts; it never
modifies a scientific controller or `paper/`. See
`docs/AUTODL_THREE_DATASET_RELEASE_SUPERVISOR.md`.

Failed four-by-four terminal closures can be retried without reopening the
main controller or any PASS/FAILED root by the bounded, source-audited repair
manifest builder at
`scripts/autodl/build_four_by_four_repair_manifest.py`. It shares project-wide
GPU UUID locks with the still-running main controller, caps its CPU task
concurrency at two, and deliberately inherits no old BACE continuation guard.
See `docs/AUTODL_FOUR_BY_FOUR_REPAIR_CONTINUATION.md`.

The current persistent controllers can be inspected through the loopback-only,
GET-only Chinese AutoDL dashboard at
`scripts/autodl/serve_four_by_four_dashboard.py`. It discovers physical
controller directories dynamically, shares one GPU/UUID-lock probe across the
page, and is intended to be accessed through an SSH tunnel. See
`docs/AUTODL_FOUR_BY_FOUR_DASHBOARD.md`.

Optimized BACE ComRecGC 50k and two-task `shared_lowmem` scheduling have
separate immutable evidence gates. Exact 500/1000 replay closure is mandatory
for the former; measured same-GPU >=20% co-location speedup with result/health/
VRAM parity is mandatory for the latter. See
`docs/AUTODL_ACCELERATION_RELEASE_GATES.md`.

Fresh BACE GCF quick50/quick100/formal-M500 replays can be queued behind the
four protected AutoDL UUID locks through a separate immutable sidecar; it does
not append to active controller manifests or duplicate the already-running
ComRecGC M=500 pair. See `docs/AUTODL_BACE_EQUIVALENCE_SIDECAR.md`.

The TasteMolNet baseline routes share one explicit three-class GINE and
an untargeted Sweet-to-non-Sweet strict-flip contract. The historical
licence-review block remains immutable evidence but is superseded for private
research by policy v2; it is not rewritten or called a licence PASS. The fresh
main controller binds the policy/data/cache receipt, reserves physical GPU 1
for the formal GINE, exposes a separate classifier-independent GPU-2 READY
lane, keeps backbone ablation disabled, and never registers the GINE as a main
method cell. Execution requires fresh output, training-state, and controller
roots plus a 20-GiB planning reservation that leaves at least 100 GiB free.
The GINE controller durably gates worker startup,
adopts the same live PID generation after controller loss, permits only one
process-loss retry against the same checkpoint root, and revalidates the full
output/state/policy closure while holding their physical authorities until
controller `PASS` is published last.  Final bundle publication is deterministic
and uses atomic directory no-replace. Upstream licence terms remain exactly
`NOT_EXPLICITLY_STATED`; project authorization is recorded separately and no
route may emit `LICENSE_PASS`. See
`docs/TASTEMOLNET_DATA_USAGE_POLICY.md` and
`docs/AUTODL_TASTEMOLNET_MAIN_TABLE_V1_HANDOFF.md` and
`docs/AUTODL_TASTEMOLNET_MULTICLASS_BASELINE_ADAPTERS.md`. The T5 clean-policy
initializer is a separate, release-disabled-by-default zero-step LoRA route
with one combined descriptor-held source/adapter consumer authority. A
controller declaration is not GPU-lock ownership, so release remains disabled
until final T3/T4/source pins and a physical execution receipt are reviewed; see
`docs/AUTODL_TASTEMOLNET_CLEAN_POLICY_INITIALIZER.md`.

For the current TasteMolNet four-method campaign, T5 may instead adopt the
unchanged generic `ChemLLM-7B-Chat` base without creating a LoRA or opening any
Taste split. The managed-v2 worker records a complete per-file source
inventory and stops at `SEALED`; a separate verifier rehashes the entire
external model tree, validates the Hugging Face config/tokenizer and exact
safetensors index-to-shard closure, and atomically publishes the small receipt.
Its semantic state is `ADOPTED_CLEAN_GENERIC_BASE`, `optimizer_steps=0`, loaded
Taste splits are `[]`, and it is not a matrix method cell. No model weight is
copied into the receipt. See
`docs/AUTODL_TASTEMOLNET_T5_CLEAN_BASE_ADOPTION_V2.md`.

After a formal Taste GINE bundle closes, T3 adopts its existing
validation-fitted temperature without refitting or copying the checkpoint.
T4 then uses only the authenticated calibration graph cache for a bounded
sixteen-parent multiclass oracle smoke on physical GPU 1, with exactly four
real connected deletions per parent and both non-Sweet strict-flip destinations
required. The T4 evidence is aggregate-only and never writes a split prediction
CSV, SMILES, molecule IDs, or test payloads; the supplemental policy remains
no-redistribution and AutoDL-only. T3/T4 outputs are exact direct fresh children
under the Taste GINE seed-7 artifact root, are published through retained
root-to-leaf descriptors, and reclose every input before creating the PASS
marker as the final commit. Their only T2 authority is the descriptor-held
fresh five-file adoption root plus reviewed gate, receipt, and embedded-source
SHA-256 pins. The receipt-only holder checks the canonical hash DAG, physical
file bindings, fixed source IDs, and the formal 19-file GINE inventory without
reopening the historical controller, training-state, or execution roots. T3
records the full binding, T4 must match it exactly, T5 freezes and reopens it,
and T6 records the same binding plus all three pins. The same supplemental
policy now also types a
future T6 Ours smoke as train-only,
using only the frozen prepared train CSV with a frozen-GINE reward and no
RF/no-validation/no-calibration/no-test; it does not by itself implement or
launch T6. T7 similarly remains release-disabled until a typed T2 adoption,
retained T3/T4 authorities, their common checkpoint, and controller/GPU-1
receipts are pinned. Its smoke reuses the official native full-graph VRRW loop
but separates `1-p(Sweet)` importance from the actual three-class
`argmax != Sweet` candidate condition and uses the independently verified
Taste-specific NeuroSED through the official normalized threshold-coverage
function. Global selection and full/paper readiness remain explicitly
unevaluated. The reusable
`TasteFrozenGINENativeAdapter` preserves original three-class order and native
graph identity for later method smokes. Its stable source-graph entrypoints
override the shared binary lineage codec with identity labels `0/1/2`, matching
PyG `y` and per-record `source_label`, and expose only the two untargeted
destination classes. T7 itself keeps the receipt-only T2
five-file authority and exact T3/T4 binding/checkpoint evidence open through a
final checkout/blob/controller/GPU revalidation. Its sixteen official VRRW
steps are physically interrupted 8+8: a private UUIDv4/generation/inode/SHA-
bound `checkpoints/<uuid>/` checkpoint
captures the complete walk, bridge, scorer, and RNG state, all in-memory
progress is reset, and continuation must restore the exact saved cursor rather
than restart from the seed. T7 is a managed-execution-v2 worker: it writes only
raw evidence, worker-exit evidence, and SEALED, then stops. A separate
verifier cross-binds NeuroSED/GINE/checkpoint evidence and owns atomic
no-replace publication; T7 contains no hardlink terminal publisher and cannot
write PASS, a final gate, or an adoption receipt. See
`docs/TASTE_GCF_NEUROSED_PROTOCOL.md` and
`docs/AUTODL_TASTEMOLNET_T7_GCF_SMOKE.md`.

The GlobalGCE frozen-GINE bridge has an explicit multiclass target view:
official internal classes `0/1` map to the reviewed frozen source/destination
classes while the calibrated three-class logits remain intact. T8 now
implements separate Sweet-to-Bitter and Sweet-to-Tasteless native GlobalGCE
branches on that one GINE, deliberately checkpoint-stops and resumes both,
then merges and canonical-deduplicates native LHS-to-RHS actions before an
independent original-order three-class strict-flip check. Its only dataset
payload is the descriptor-held prepared train CSV; the branch-local holdout is
derived from those train rows and no validation, calibration, or test split is
opened. The terminal root is aggregate-only and has exactly `state.json`,
`manifest.json`, `gate.json`, `input_hashes.json`, `output_hashes.json`, and
`PASS`.

This is a stage-frozen, release-disabled implementation, not permission to
run science. The AutoDL wrapper refuses before path or GPU inspection and the
paired Slurm script always exits before Python. A later reviewed release must
bind the exact T2 receipt-only authority, matching held T3/T4 checkpoint
binding, managed physical-GPU2 child authority, immutable implementation and
official source, fresh state/output roots, and the strict public T8 consumer.
Environment variables cannot bypass the checked-in release bits.

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

AutoDL's completed TasteMolNet GINE result is adopted through the independent
v2 verifier documented in
`docs/AUTODL_TASTEMOLNET_T2_ADOPTION_V2.md`. The verifier preserves the old
identity-drift controller failure and never retrains the classifier.

TasteMolNet T3 performs a new validation-only scalar-temperature fit through a
managed-v2 worker and independent verifier, as documented in
`docs/AUTODL_TASTEMOLNET_T3_CALIBRATION_V2.md`.

TasteMolNet T4 now has a managed-v2 GPU-2 successor that consumes only the
published T3 checkpoint and authenticated calibration graph cache, repeats the
bounded three-class smoke in an independent verifier, and publishes aggregate
evidence atomically. See
`docs/AUTODL_TASTEMOLNET_T4_ORACLE_SMOKE_V2.md`.
