# Decisions Log

This file records major design decisions for the counterfactual subgraph v3 project.

It should be updated whenever a meaningful implementation, algorithmic, or interface decision is made.

---

## Template

```md
## [YYYY-MM-DD] Decision title

### Background
Why was this decision needed?

### Decision
What was decided?

### Alternatives considered
What other options were considered?

### Consequences
What changes because of this decision?

### Status
Proposed / Accepted / Deprecated / Superseded
```

---

## [2026-04-09] Rebuild repository from scratch with documentation-first workflow

### Background
The previous project evolved through multiple experimental iterations. It likely accumulated script coupling, outdated assumptions, and mixed objectives inherited from earlier versions. To prevent repeated confusion during reconstruction, the repository needs a stable written definition before code implementation begins.

### Decision
Rebuild the repository from an empty root using a documentation-first workflow. The initial authoritative files are:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

These files define the objective, engineering rules, roadmap, and future design log.

### Alternatives considered
1. Start coding immediately and add documentation later.
2. Recover the old repository structure first and refactor afterward.
3. Keep all design notes only in chat history.

### Consequences
- The repository gets a clear source of truth from the start.
- Codex can be guided by repository-local instructions instead of relying on conversational memory.
- Early progress is slower in appearance but more stable in direction.

### Status
Accepted

---

## [2026-04-15] Implement first PPO training path with residual-based reward wrapper

### Background
The repository already had a trained SFT adapter, a lightweight AIDS oracle,
and a chemistry layer for capped fragment validation and deletion. What was
still missing was a real PPO training path that could optimize ChemLLM outputs
toward the deletion-based counterfactual objective on HPC hardware.

### Decision
Add a concrete PPO training entrypoint in `scripts/train_ppo.py` together with a
unified reward wrapper in `src/rewards/reward_wrapper.py`.

The reward wrapper uses a three-stage early-stop flow:

- parseability and connectedness checks;
- parent-subgraph validation for capped fragments;
- residual-molecule scoring after deleting the fragment from the parent.

The semantic reward term is computed on the residual graph rather than on the
fragment alone, because the v3 objective is label flipping after deletion.

The PPO script loads:

- ChemLLM-7B-Chat in 4-bit mode;
- the SFT LoRA checkpoint as the initial policy;
- a frozen LoRA-backed reference model for KL control;
- prompts from either the raw HIV CSV or a JSONL prompt file.

The training loop logs reward statistics, structural pass rates, simple
collapse diagnostics, and representative generations into the run directory.

### Alternatives considered
1. Keep RL as a runtime-preparation placeholder only.
2. Score the fragment alone with the oracle instead of the residual molecule.
3. Hardcode one dataset schema and refuse either CSV or JSONL prompts.

### Consequences
- The repository now has a runnable PPO-stage backbone aligned with the
  counterfactual v3 objective.
- KL control remains anchored to the SFT adapter rather than drifting from the
  base model directly.
- PPO runs surface chemistry and collapse signals explicitly instead of hiding
  them inside trainer internals.

### Status
Accepted

---

## [2026-04-15] Switch PPO policy loading back to native causal LM for experimental TRL

### Background
After the first PPO entrypoint landed, newer `trl.experimental.ppo` builds
showed constructor-time incompatibilities with explicit value-head wrappers,
including failures around missing wrapper attributes such as
`base_model_prefix`. This indicated that the trainer now expects native causal
LM policies and prefers to manage policy/value wrapping internally.

### Decision
Update `scripts/train_ppo.py` so that PPO loads native
`AutoModelForCausalLM`-style PEFT models for both the trainable policy and the
frozen reference policy, without constructing
`AutoModelForCausalLMWithValueHead`.

The PPO trainer initialization is now version-adaptive in three ways:

- `PPOConfig` kwargs are filtered against the runtime signature;
- trainer kwargs map across `args/config`, `model/policy`, and
  `ref_model/ref_policy` variants;
- external `value_model` is omitted or forced to `None` so TRL can own value
  wrapping internally.

The script also provides a lightweight reward adapter that exposes
`ChemRLRewarder` as either `reward_model` or `reward_funcs` when those newer
experimental hooks are present.

### Alternatives considered
1. Keep the explicit value-head wrapper and patch missing attributes one by one.
2. Pin the repository to one older TRL version instead of adapting the code.
3. Move the whole PPO path back to a handwritten optimizer loop.

### Consequences
- The PPO script is better aligned with current experimental TRL architecture.
- Fewer wrapper-specific attribute mismatches should appear when the trainer is
  upgraded.
- The script still keeps the repository's deletion-based counterfactual reward
  logic outside trainer internals.

### Status
Accepted

---

## [2026-04-15] Use explicit value and reward models for experimental PPOTrainer

### Background
Follow-up PPO integration work revealed another compatibility shift in newer
`trl.experimental.ppo.PPOTrainer` builds: the trainer now expects a real
transformers-style `value_model` object and also routes internal scoring
through a PyTorch `reward_model(input_ids=..., attention_mask=..., ...)`
interface.

### Decision
Keep the policy and reference networks as native causal LMs, but explicitly add:

- a 4-bit `AutoModelForSequenceClassification` value model with `num_labels=1`;
- a torch `reward_model` wrapper that decodes generated sequences back to text,
  reconstructs the parent / fragment pair, and calls `ChemRLRewarder`.

The trainer initialization in `scripts/train_ppo.py` now wires:

- `model`: native causal LM policy with the SFT LoRA adapter;
- `ref_model`: frozen native causal LM reference policy;
- `value_model`: explicit scalar sequence-classification model;
- `reward_model`: deletion-based chemistry reward wrapper.

### Alternatives considered
1. Continue passing `None` for the value model and rely on implicit trainer behavior.
2. Keep reward scoring entirely outside the trainer and ignore new internal hooks.
3. Reintroduce custom non-transformers wrapper classes around the value head.

### Consequences
- PPO initialization is better aligned with stricter experimental TRL releases.
- Internal trainer scoring can now call a PyTorch-compatible reward interface
  without losing the repository's residual-graph counterfactual objective.
- The value network contract is explicit instead of version-dependent.

### Status
Accepted

---

## [2026-04-09] Treat counterfactual fragment generation as the sole primary objective

### Background
Earlier project stages and surrounding discussions may involve concept extraction, class-indicative subgraphs, or rationale-like objectives. Those formulations are related but not identical to the present v3 goal.

### Decision
Define the project strictly as counterfactual fragment generation: the generated fragment should be useful because deleting it is likely to flip the label.

### Alternatives considered
1. Optimize for fragment-only label predictiveness.
2. Optimize for class-shared concept extraction.
3. Optimize for rationale sufficiency rather than deletion-based effect.

### Consequences
- Reward design must include deletion-based semantics.
- Evaluation must include flip-related metrics.
- Old code paths aligned to non-counterfactual targets should be treated as legacy behavior.

### Status
Accepted

---

## [2026-04-09] Use modular repository structure rather than monolithic training script

### Background
The project involves chemistry checks, prompt generation, SFT, RL, evaluation, and logging. A single large script would make the system difficult to debug and easy to break.

### Decision
Adopt a modular structure with separate folders for data, models, rewards, training, evaluation, chemistry utilities, and general utilities.

### Alternatives considered
1. Keep everything in one file for convenience.
2. Split only by training stage.
3. Organize by experiments rather than functionality.

### Consequences
- Initial setup requires more files.
- The system becomes easier to test, replace, and extend.
- Reward and chemistry behavior can be validated independently.

### Status
Accepted

---

## [2026-04-09] Prioritize chemistry and reward layers before large-scale training code

### Background
The most brittle parts of this project are likely to be chemistry correctness and reward semantics. If these layers are unstable, training results will be misleading.

### Decision
Implement chemistry utilities and reward computation before building full SFT/RL training pipelines.

### Alternatives considered
1. Start with full training script and fill utility functions later.
2. Start with model wrapper only.
3. Start with evaluation only.

### Consequences
- Early iteration focuses on correctness rather than speed.
- The train loop can remain thinner and easier to debug.
- Reward failures and chemistry mismatches can be unit-tested.

### Status
Accepted

---

## [2026-04-09] Bootstrap the repository with typed module interfaces before any training implementation

### Background
After the documentation-first reset, the repository still had no source tree, no config folders, and no stable interface boundaries. If training code were introduced at that point, chemistry logic, reward semantics, and CLI behavior would likely become coupled again.

### Decision
Create the repository skeleton first and define minimum typed interfaces for:

- `src/data/`
- `src/chem/`
- `src/rewards/`
- `src/models/`
- `src/train/`
- `src/eval/`
- `src/utils/`

Also add thin placeholder CLI files under `scripts/` and minimal smoke tests for prompt and reward contracts.

The bootstrap intentionally implements only safe low-level helpers such as JSONL IO, prompt construction, reward-term aggregation, and text-level collapse diagnostics. RDKit-backed chemistry logic and all training loops remain deferred.

### Alternatives considered
1. Start with one large training script and refactor later.
2. Implement RDKit parsing and training code immediately without freezing interfaces first.
3. Leave the directory layout undocumented and decide file boundaries ad hoc.

### Consequences
- Future chemistry, reward, inference, and training work now has a stable import surface.
- The repository stays aligned to the deletion-based counterfactual objective instead of drifting toward concept extraction.
- Scripts remain thin by design, because domain logic now has clear module ownership.
- The next implementation phase can focus on chemistry correctness rather than file organization.

### Status
Accepted

---

## [2026-04-09] Implement chemistry utilities with RDKit-first behavior and explicit failure types

### Background
The v3 objective depends on structural correctness before any reward or training logic can be trusted. The repository therefore needs a chemistry layer that can safely parse SMILES, check connectedness and parent-substructure relations, and attempt fragment deletion without forcing train code to interpret raw RDKit exceptions.

### Decision
Implement the first chemistry utility layer in:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

The layer is RDKit-first when RDKit is available, but it must also degrade safely when RDKit is missing by returning structured failure types instead of crashing. The shared result dataclasses now carry normalized failure categories, and `validate_fragment_candidate(...)` aggregates these into one interface for downstream reward and evaluation code.

### Alternatives considered
1. Raise raw exceptions from RDKit and let train or eval scripts handle them.
2. Hard-require RDKit at import time and fail the whole repository if it is absent.
3. Delay chemistry implementation until reward or inference code exists.

### Consequences
- Reward and evaluation modules can consume chemistry results without duplicating error handling.
- The repository remains usable in environments where RDKit has not been installed yet.
- Deletion behavior is deterministic at the interface level, with the first matched connected fragment removed and failures surfaced explicitly.

### Status
Accepted

---

## [2026-04-10] Add a config-driven local and HPC runtime layer without hardcoded paths

### Background
The rebuilt repository now has modular source folders and chemistry utilities, but it still needs a stable way to run from a local workstation and from an HPC cluster. Without a shared runtime layer, path handling, log directories, and environment-specific behavior would drift into ad hoc script logic.

### Decision
Add a runtime adaptation layer based on repository-relative config files and thin CLI entrypoints.

This layer includes:

- base, local, hpc, sft, rl, and eval config files under `configs/`
- environment detection and config merging in `src/utils/env.py`
- repository-relative path resolution in `src/utils/paths.py`
- file-backed logging helpers in `src/utils/logging_utils.py`
- run entrypoints in `scripts/run_sft.py`, `scripts/run_rl.py`, `scripts/run_eval.py`, and `scripts/run_infer.py`
- Slurm templates for single-node single-GPU execution under `scripts/slurm/`

Model and tokenizer handling must support local filesystem paths and avoid silent remote downloads by default.

### Alternatives considered
1. Hardcode local and HPC paths directly inside the stage scripts.
2. Depend on an external YAML package before adding any runtime config support.
3. Jump directly to distributed training setup before the single-node path is stable.

### Consequences
- Local development and HPC submission now share one config and path resolution story.
- The codebase remains portable because configs stay relative and resolved at runtime.
- The repository can prepare deterministic run manifests before full SFT, RL, and evaluation logic is implemented.
- Distributed training remains intentionally out of scope for this phase.

### Status
Accepted

---

## [2026-04-10] Implement a minimal single-sample inference loop with heuristic fragment generation

### Background
The repository already has chemistry utilities and runtime adaptation, but it still needs a runnable end-to-end path from one parent SMILES to one fragment candidate. This is necessary to validate the IO contract before wiring any full SFT or RL training logic.

### Decision
Implement a minimal inference closure centered on `scripts/run_infer.py` and `src/eval/inference.py`.

This path:

- accepts one parent SMILES from CLI or config
- produces one heuristic fragment candidate without using a trained policy
- runs chemistry utilities for parseability, connectedness, and parent-substructure checks
- prints a structured JSON result
- stays independent of the full training stack

The heuristic prefers small connected parent substructures when chemistry backends are available, and falls back to the parent SMILES when no smaller valid candidate can be established.

### Alternatives considered
1. Keep `run_infer.py` as config-only runtime preparation.
2. Wait for a trained model before implementing any inference path.
3. Print only free-form text instead of a machine-readable result.

### Consequences
- The project now has a minimal runnable contract for one-sample inference.
- Chemistry utilities can be exercised from a real CLI path without invoking training code.
- The current fragment proposal is intentionally heuristic and not yet counterfactual-optimal, but it keeps the repository moving toward the v3 objective with a reviewable baseline.

### Status
Accepted

---

## [2026-04-14] Add a lightweight RandomForest Oracle and residual-based PPO reward wrapper

### Background
The repository has completed the SFT stage and is moving toward PPO. At that point the project needs a reward path that is both chemically constrained and fast enough to run inside RL rollouts. The existing codebase already has RDKit-backed capped-subgraph validation and deletion, but it still lacked:

- a lightweight local Oracle for fast activity scoring;
- a PPO-facing reward wrapper that combines validity, subgraph, and counterfactual terms;
- a clear decision on whether the semantic reward should score the fragment itself or the residual molecule after deletion.

### Decision
Add:

- `scripts/train_aids_oracle.py` to train a RandomForest classifier on Morgan fingerprints from `data/aids_dataset.csv`;
- `src/rewards/chem_rules.py` as a small structural reward engine for validity and parent-subgraph checks;
- `src/rewards/reward_calculator.py` as the PPO-facing reward wrapper that loads the Oracle bundle and computes `R_valid + R_subgraph + R_counterfactual`.

The counterfactual term is explicitly defined on the residual molecule `x \ g` after deleting the generated fragment, not on the fragment alone. Dummy atoms (`*`) are cleaned through RDKit graph editing before Morgan fingerprint extraction.

### Alternatives considered
1. Score the generated fragment by itself and treat that as the counterfactual term.
2. Put Oracle logic directly inside the PPO training script.
3. Save a custom Python class inside the pickle bundle instead of a plain dictionary.

### Consequences
- PPO reward computation stays aligned with the v3 deletion-based counterfactual objective.
- Oracle scoring remains fast enough for rollout-time use because it relies on RandomForest plus Morgan fingerprints.
- Structural chemistry checks stay modular and reusable instead of being buried in the train loop.
- The saved Oracle bundle is easier to reload across scripts because it stores plain metadata alongside the fitted sklearn model.

### Status
Accepted

---

## [2026-04-13] Add base-model metric tooling and presentation visualization for the SFT stage

### Background
The SFT stage now has stable headline numbers for validity, capping behavior, and token accuracy, but the project still lacked two practical tools: a deterministic way to compute baseline metrics from JSONL inference logs, and a reusable plotting pipeline for lab-meeting summaries.

### Decision
Add a small evaluation utility layer for base-model JSONL logs and extend the SFT visualization module with presentation-ready outputs.

This change introduces:

- `src/eval/base_metrics.py` for RDKit-backed capping/validity statistics
- `src/eval/base_inference.py` for deterministic base-model batch inference over `sft_val.jsonl`
- `scripts/eval_base_metrics.py` as a thin CLI entrypoint for base-model log evaluation
- `scripts/run_infer_base.py` as a thin CLI entrypoint that saves base-model predictions to JSONL
- an upgraded `src/eval/sft_visualization.py` that renders:
  - base-vs-SFT comparison bars
  - a dual-axis simulated training dynamics figure
  - a high-resolution RDKit rendering of a capped fragment
- the updated `scripts/visualize_sft_summary.py` CLI for parameterized figure generation

### Alternatives considered
1. Compute baseline metrics manually in notebooks.
2. Keep visualization code in a single ad hoc script under `scripts/`.
3. Hardcode baseline numbers directly into plotting code without a reusable evaluation helper.

### Consequences
- The repository now has a reproducible path from base-model inference logs to group-meeting figures.
- Evaluation logic remains modular under `src/eval/`, while CLI scripts stay thin and HPC-friendly.
- Presentation plots can be regenerated with different baseline numbers without editing source code.
- The training dynamics figure now uses discrete epoch-level line charts instead of dense smooth curves, which makes the plot easier to explain during presentations.

### Status
Accepted

---

## [2026-04-11] Add balanced capped-fragment SFT data preparation for the HIV dataset

### Background
The rebuilt repository already has RDKit-backed capped-subgraph utilities and thin runtime entrypoints, but it still lacked a concrete path for constructing supervised fine-tuning data from the local HIV CSV. Because the HIV benchmark is heavily label-imbalanced, a naive sample would underexpose positive molecules during SFT.

### Decision
Add a new balanced SFT data-preparation path centered on `scripts/prepare_sft_data.py` and `src/data/sft_preparation.py`.

This path:

- loads `data/raw/AIDS/HIV.csv` with pandas
- filters out invalid parent SMILES using the existing chemistry layer
- keeps all valid positive molecules and fills the remaining target size with sampled negatives
- constructs capped fragment targets by cutting one or two acyclic single bonds with RDKit dummy-atom capping
- validates generated fragments with the shared capped-subgraph checks
- writes minimal `instruction` / `output` JSONL files for ChemLLM SFT

### Alternatives considered
1. Reuse the placeholder `scripts/prepare_data.py` without adding reusable source modules.
2. Sample the HIV dataset according to its natural class distribution.
3. Generate weak targets without validating capped-subgraph correctness against the parent molecule.

### Consequences
- The repository now has an end-to-end path for building balanced SFT supervision data aligned with the capped-fragment objective.
- Positive molecules are intentionally overrepresented relative to the raw dataset so ChemLLM sees enough active-molecule structure during SFT.
- Fragment generation can still fail for some molecules, so the script explicitly reports success rate and observed label ratios after construction.

### Status
Accepted

---

## [2026-04-10] Treat dummy atoms as attachment-point caps in the chemistry layer

### Background
The project now needs chemically valid graph cutting. Plain atom deletion breaks
valence and makes the fragment contract ambiguous whenever the generated
subgraph has open attachment points to the rest of the parent molecule.

### Decision
Adopt dummy atoms (`*`) as the canonical capping representation inside
`src/chem/`.

This means:

- capped fragment SMILES such as `*c1ccccc1` are parsed and sanitized through a
  dummy-aware RDKit path;
- dummy atoms are treated as attachment-point queries when checking whether a
  fragment is a valid parent subgraph;
- capped deletion uses RDKit core replacement semantics so the remainder graph
  is also capped with dummy atoms instead of leaving broken chemistry.

### Alternatives considered
1. Keep using uncapped raw atom deletion and repair broken valence later.
2. Represent cut points with text markers outside SMILES instead of dummy atoms.
3. Treat dummy atoms as ordinary fragment atoms that should also be deleted from
   the parent match.

### Consequences
- The chemistry layer now distinguishes between real fragment atoms and dummy
  attachment points.
- Validation and deletion stay aligned with the v3 counterfactual objective,
  because both fragment extraction and residual construction use the same capped
  semantics.
- Future model outputs can use capped fragment SMILES directly without adding an
  extra post-processing representation.

### Status
Accepted

---

## [2026-04-10] Bootstrap local-only ChemLLM inference with dataset-backed sampling

### Background
The chemistry layer can now validate capped fragment SMILES, but the repository
still needs a real local-model inference path before SFT or RL code is added.
At the same time, the user needs a safe way to confirm that the local AIDS/HIV
CSV file and local ChemLLM checkpoint are usable without any accidental network
requests.

### Decision
Add a local-only ChemLLM inference bootstrap consisting of:

- `scripts/test_assets.py` for local dataset and model asset validation;
- a ChemLLM-specific prompt builder with hard-coded capped-fragment few-shot
  examples;
- a lightweight Hugging Face `ChemLLMGenerator` that always loads with
  `local_files_only=True`;
- dataset-backed `run_infer.py` behavior that samples one real molecule from the
  local AIDS/HIV CSV file when no explicit SMILES is provided.

If the local transformers stack is unavailable at runtime, the CLI is allowed to
fall back to the existing heuristic inference path, but it must surface the
model-side failure explicitly in the structured result.

### Alternatives considered
1. Wait for training code before integrating any real LLM backend.
2. Hardcode one demo SMILES instead of sampling a real local dataset row.
3. Allow `from_pretrained` to use default remote resolution behavior.

### Consequences
- The project now has a true local model inference integration point without
  depending on vLLM or distributed serving.
- Asset validation becomes safer because the repository refuses to reach out to
  Hugging Face when local files are incomplete.
- The inference CLI stays usable in lightweight dev environments because it can
  degrade to heuristic generation while still reporting why the model path did
  not run.

### Status
Accepted
