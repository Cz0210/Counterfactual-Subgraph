# Mutagenicity Fresh SFT and PPO

## Scope

This route is an independent Mutagenicity source-1-to-target-0 experiment:

```text
Pure ChemLLM-7B-Chat
-> randomly initialized Mutagenicity LoRA
-> strict-first Mutagenicity SFT
-> validation task-level checkpoint selection
-> fresh stable PPO with an audited flip-dominant reward
```

It does not replace or modify the existing AIDS-to-Mutagenicity transfer
ablation under `outputs/hpc/mutagenicity/sft_continued_v1` and
`outputs/hpc/mutagenicity/ppo_stable_v1`.

Train is the only parameter-training split. Validation is used for checkpoint
selection. Calibration is reserved for later distance/selector calibration,
and test is not read by any script in this route.

## Target Selection Audit

The v1 call chain is:

1. `src/data/mutagenicity_sft_ppo.py` loads teacher-correct source parents.
2. `src/data/sft_v3_builder.py::enumerate_reference_candidates_for_parent`
   enumerates and filters structural candidates.
3. `CounterfactualTeacherScorer` supplies original/residual predictions.
4. `src/data/sft_v3_builder.py::_candidate_ranking_key` chooses the v1 target.

The v1 ranking key already places `oracle_ok`, `cf_flip`, and `cf_drop` ahead
of atom-ratio and strategy priorities. Therefore the source code does not
support the hypothesis that a non-strict `bond_k2` candidate can outrank an
available strict candidate merely because of strategy priority. The v1
artifact does not retain the full proposal inventory, so
`audit_mutagenicity_sft_target_selection.py` deterministically replays the
same proposal/filter path and fixed RF teacher before making a data-level
claim.

The generic teacher's historical `cf_flip` is `pred_after != original_label`.
For the fixed teacher-correct label-1 parent cohort this is equivalent to 1->0,
but v2 deliberately ignores that convenience and requires:

```text
oracle_ok and pred_before == 1 and pred_after == 0
```

Code evidence at implementation time:

- v1 Mutagenicity adapts into the shared selector at
  `src/data/mutagenicity_sft_ppo.py:424`;
- v1 enumeration and selection are
  `src/data/sft_v3_builder.py:343-455`;
- its exact legacy key is `src/data/sft_v3_builder.py:1011-1034`;
- the generic historical weak expression is
  `src/rewards/counterfactual_oracle.py:342`;
- the explicit v2 hard partition is
  `src/data/mutagenicity_sft_v2.py:183`;
- fresh-base rejection, one-time wrapping, and runtime adapter audit are
  `src/train/mutagenicity_fresh_sft.py:71`, `:105`, and `:146`;
- task metric aggregation/ranking are
  `src/eval/mutagenicity_generator.py:59` and `:241`;
- the opt-in reward config/profile are
  `src/rewards/reward_wrapper_stable.py:37` and `:343`.

## Strict-First v2

`src/data/mutagenicity_sft_v2.py` implements a hard two-level policy:

1. If any strict candidate exists, select only from strict candidates.
2. Otherwise, the fallback ablation may select a legal positive-CFDrop
   candidate.
3. Other candidates cannot become SFT targets.

Within one level, ties are resolved by decreasing CFDrop, increasing atom
ratio, increasing split-local completion frequency, stable strategy order,
and canonical fragment SMILES. Completion frequencies are computed separately
for train and validation.

The independent output root is:

```text
outputs/hpc/mutagenicity/sft_ppo_data_v2
```

It contains strict-only, strict-first-fallback, optional strict multitarget,
all-parent PPO prompt, candidate inventory, parent candidate count, diversity,
summary, and leakage artifacts. PPO prompt views retain all 1448 train and 260
validation source parents, including parents without a strict SFT target.

The user-observed v1 selected counts (1317/250) and strict counts (1118/191)
remain hypotheses for replay validation. The v2 builder records actual counts
in `dataset_summary.json`; no local code-only check should claim those HPC
counts as newly verified.

## Fresh LoRA Provenance

`scripts/train_mutagenicity_sft_fresh.py` reuses the existing:

- 4-bit ChemLLM loader;
- completion-only tokenizer semantics;
- prompt masking and EOS handling;
- train/validation isolation validation;
- coverage-aware collator;
- Hugging Face Trainer setup.

The fresh-only sequence is:

1. load pure `pretrained_models/ChemLLM-7B-Chat`;
2. reject any existing PEFT state or LoRA parameters;
3. create a new `LoraConfig`;
4. call `get_peft_model` exactly once;
5. verify one configured and active `default` adapter;
6. verify all non-LoRA base parameters are frozen;
7. record `source_adapter_checkpoint=null` and
   `aids_adapter_weights_loaded=false`.

The LoRA architecture matches the prior AIDS configuration (`r=8`,
`alpha=16`, dropout `0.05`, targets `wqkv,wo,w1,w2,w3`) but no adapter weights
are read. If the base tokenizer is unavailable, an old checkpoint may supply
tokenizer files only. The audit records tokenizer reuse separately and never
describes it as adapter-weight reuse. The post-run audit hashes every periodic
checkpoint and the final root adapter against the forbidden AIDS adapter as a
secondary artifact-level check.

Default full SFT settings are 300 max steps, save/eval every 50 steps,
early-stopping patience 2, learning rate `2e-4`, seed 7, maximum sequence
length 1024, and effective batch size 16. Token eval loss remains an auxiliary
checkpoint result.

## Task-Level Generator Evaluation

`scripts/evaluate_mutagenicity_generator.py` evaluates every model with the
same complete 260-parent teacher-correct validation cohort. It supports pure
ChemLLM, SFT adapters, PPO adapters, `N=1,4,8`, fixed seeds, projection, and
the unchanged Mutagenicity RF teacher.

Hit@1 checkpoint ranking is lexicographic:

1. strict CF flip rate, descending;
2. mean CFDrop, descending;
3. final substructure rate, descending;
4. parse rate, descending;
5. atom-ratio target deviation, ascending;
6. duplicate rate, ascending.

The evaluator stores both validation cohort and decoding configuration hashes.
It also reports top-fragment frequencies, entropy, parent-conditioned
uniqueness, and optional SFT/PPO parent difficulty categories. When
`--difficulty-models SFT_NAME,PPO_NAME` is supplied, it writes
`parent_difficulty.csv`, `hard_parent_summary.json`, and
`strategy_failure_summary.csv`; this remains a read-only analysis and does
not enable hard-parent oversampling.

The default `--cohort-split val` run is the only checkpoint-selection-eligible
mode. A separate `--cohort-split train --expected-parent-count 1448` run can
produce train hard-parent diagnostics, but deliberately does not write
`best_task_checkpoint.json`.

## Reward Audit and Flip-Dominant Profile

Run the read-only reward audit against the completed transfer PPO before fresh
PPO. It reports p10/p50 strict rewards, p90 non-flip reward, component scales,
clipping, projection groups, and an observed reward margin.

The generated `recommended_reward_config.json` scales auxiliary components
from their observed p90 magnitudes, derives the strict bonus needed for the
requested margin, and expands the upper clip only as required by the observed
strict distribution. This is a proposed training configuration, not evidence
of improved validation behavior.

The shared stable wrapper defaults to `reward_profile=legacy`, which bypasses
all new reweighting. For that profile the historical final-fragment correction
and teacher-confidence order is unchanged. `mutagenicity_flip_dominant`
retains the confidence-gate audit fields and then runs as the final reward
producer after final-fragment atom-ratio correction. This prevents a stale
legacy counterfactual component from being subtracted from the newly composed
reward. The profile:

- recomputes strict flip as `pred_before==1 and pred_after==0`;
- uses final-fragment chemistry and atom ratio;
- gives an independent strict-flip bonus;
- caps all positive non-flip auxiliary terms;
- keeps negative syntax/validity evidence;
- logs pre-cap, post-cap, unclipped, and clipped reward components.

## Fresh PPO Plan

Fresh PPO is a thin provenance/configuration layer over
`scripts/train_mutagenicity_ppo_stable.py`; it does not reimplement PPO.

| Mode | Parents | Rollout batch | Updates | Eval/save |
| --- | ---: | ---: | ---: | ---: |
| smoke | 5 | 1 | 5 | 1 |
| medium | 256 | 16 | 16 | 4 |
| full | 1448 | 16 | 91 | 10 |

All modes use deterministic shuffling without replacement. The full plan is
`ceil(1448/16)=91` updates and exactly one parent epoch. The unchanged loop
retains parent projection, projected CF reward, substructure distance,
strict-flip teacher semantics, final-fragment atom ratio, frozen reference,
adaptive KL, gradient clipping, isolated validation RNG, final validation,
and full parent coverage checks.

## HPC Runbook

Submit from the project root after creating `logs`.

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs

# 1. Replay/audit v1 target selection.
sbatch scripts/slurm/audit_mutagenicity_sft_target_selection.sh

# 2. Build independent strict-first v2 data.
sbatch scripts/slurm/build_mutagenicity_sft_ppo_v2.sh

# 3. Train and audit a tiny Fresh-v1 LoRA initialization route.
sbatch scripts/slurm/train_mutagenicity_sft_fresh_smoke.sh

# 4. Train full fresh SFT variants (start with strict-v2).
sbatch scripts/slurm/train_mutagenicity_sft_fresh_strict_v2_full.sh
# Ablations:
# sbatch scripts/slurm/train_mutagenicity_sft_fresh_v1_full.sh
# sbatch scripts/slurm/train_mutagenicity_sft_fresh_fallback_v2_full.sh

# 5. Evaluate checkpoints. MODELS is semicolon-separated NAME=PATH.
MODELS='Pure-ChemLLM=PURE_BASE;Fresh-strict-50=outputs/hpc/mutagenicity/sft_fresh_strict_v2/checkpoint-50' \
  sbatch scripts/slurm/evaluate_mutagenicity_generator.sh

# 6. Audit the old transfer PPO reward and generate an explicit profile.
sbatch scripts/slurm/audit_mutagenicity_ppo_reward_components.sh

# 7. Point the stable Fresh-SFT link at the task-best checkpoint explicitly,
# then run fresh PPO in increasing cost order.
sbatch scripts/slurm/train_mutagenicity_ppo_fresh_smoke.sh
sbatch scripts/slurm/train_mutagenicity_ppo_fresh_medium.sh
sbatch scripts/slurm/train_mutagenicity_ppo_fresh_full.sh
```

Do not create the `sft_fresh_strict_v2_best` link from token loss alone. Read
`best_task_checkpoint.json` from the complete validation generator evaluation
first.
