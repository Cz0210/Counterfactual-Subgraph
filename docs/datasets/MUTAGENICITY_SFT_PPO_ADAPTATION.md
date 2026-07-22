# Mutagenicity SFT/PPO Adaptation

## 1. Scope

This adapter builds the Mutagenicity data artifacts used by the existing route:

```text
ChemLLM SFT -> stable PPO -> candidate pool -> WNode-aware selector -> strict-flip CCRCov
```

It does not change ChemLLM training, PPO optimization, projection, selector,
WNode, or CCRCov code. The experiment direction is fixed to mutagenic label 1
parents and non-mutagenic label 0 counterfactuals.

## 2. AIDS SFT v3 audit

### Build entry and source files

The actual AIDS/HIV SFT v3 build entry is
`scripts/build_sft_v3_from_hiv.py`. Its implementation is
`src/data/sft_v3_builder.py`, and the HPC wrapper is
`scripts/slurm/build_sft_v3_from_hiv.sh`.

The historical SFT v3 train and validation artifacts are JSONL, not CSV. Each
row produced by `SFTV3Example.to_json()` contains at least:

```text
id, graph_id, smiles, parent_smiles, label,
prompt, completion, response, instruction, output,
reference_fragment, task_type, meta
```

`scripts/train_sft.py` normalizes `prompt/completion`,
`instruction/output`, or the legacy `prompt/response` aliases. The standard
SFT run uses `--max-steps 500`; the resulting adapter is consumed from a path
such as `.../checkpoint-500` by stable PPO.

### Real target-fragment source

The AIDS SFT v3 completion is a weak parent-derived target, not an external
annotation and not a fragment copied from a PPO candidate pool. For every
parent, `select_reference_candidate_for_parent()` performs the following:

1. Enumerate `build_parent_projection_candidates()` plus Murcko-like candidates.
2. Normalize candidates with `normalize_core_fragment(..., keep_largest_component=True)`.
3. Require an exact parent match with `match_core_fragment_to_parent()`.
4. Reject the full parent, out-of-range atom ratios, and fragments outside the
   configured atom-count range.
5. Require successful deletion and a non-empty residual molecule.
6. Optionally rank the filtered candidates with the deletion-based RF teacher;
   otherwise use the existing deterministic strategy/size ranking.

The default retained range is atom ratio 0.10 to 0.55, 3 to 30 fragment atoms,
and at most 160 enumerated candidates per parent. `raw_fragment` may contain
dummy atoms as attachment-point caps for audit. The actual `core_fragment`,
`output`, and `completion` are dummy-free core SMILES.

The canonical SFT prompt is built by `build_counterfactual_prompt()` with the
parent SMILES and no original-label line by default. The completion is the
dummy-free core fragment, prefixed by a newline for `SFTTrainer` separation.

### Parent sampling and splits

The AIDS builder identifies a parent by its raw source-row-derived sample ID.
It normalizes SMILES but does not perform a separate canonical-SMILES dedup
pass. It samples by scaffold and atom-count strata, then uses a label-aware
scaffold holdout for SFT train/validation.

Oracle use during SFT v3 is candidate ranking, not a mandatory parent filter.
The historical label-1 PPO CSV is a minimal `smiles,label` file. Stable PPO
loads it through `load_ppo_prompt_records()`, resolves `parent_smiles` or
`smiles`, resolves `label`, and constructs a canonical label-aware prompt when
the CSV has no prompt column. A separate teacher-confidence filter exists but
is not intrinsic to the original minimal label-1 CSV.

Stable PPO uses `DATASET_PATH` for the train prompt CSV and optionally
`VAL_DATASET_PATH` for validation. It expects a usable SMILES column and binary
label column; current label-1 jobs use `parent_smiles`/`smiles` and `label`.

## 3. Mutagenicity input sufficiency

The fixed teacher-consistent files contain complete parent-molecule SMILES,
stable molecule IDs, scaffold assignments, ground-truth labels, and RF teacher
predictions/probabilities. They therefore provide the same structural input
needed by the AIDS parent-derived target constructor. No extra fragment labels
are required.

The adapter calls the existing AIDS selector directly through a compatibility
`HIVParentRecord`; this is an interface reuse only and does not rename the
Mutagenicity dataset. If the existing selector cannot produce a valid target
for a parent, that parent is omitted from SFT and recorded with a drop reason.
It remains in the one-row-per-parent PPO prompt set. No pseudo-fragment fallback
is generated.

Default construction inputs are:

```text
outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/
  train_source_label1_teacher_correct.csv
  val_source_label1_teacher_correct.csv
```

These views are reproducibly built by:

```text
scripts/data/build_mutagenicity_teacher_consistent_views.py
```

For each split, the builder uses
`outputs/hpc/datasets/final/mutagenicity_v1_processed/<split>.csv` as the
authoritative metadata table and strictly joins
`outputs/hpc/oracle/final/mutagenicity_rf_v1/predictions_<split>.csv` by unique
`molecule_id`. It verifies equal row and ID sets, exact SMILES and label
agreement, and the processed `split` value. The processed `semantic_label` and
`scaffold_smiles` are preserved rather than reconstructed from prediction
files. Both SFT/PPO Slurm wrappers run this builder before consuming the views.

The builder writes three views per split:

```text
<split>_source_label1_all.csv
<split>_source_label1_teacher_correct.csv
<split>_target_label0_teacher_correct.csv
```

along with `teacher_consistent_summary.json` and
`teacher_consistent_report.md`. The source teacher-correct view requires
`label=1`, `teacher_pred=1`, and `teacher_correct=true`.

The calibration and test teacher-consistent files are read only as exclusion
manifests. They never enter candidate generation, SFT, or PPO output.

The RF teacher used only for candidate ranking is:

```text
outputs/hpc/oracle/final/mutagenicity_rf_v1/mutagenicity_rf_model.pkl
```

## 4. Output schema

Every SFT and PPO row retains:

```text
molecule_id, parent_smiles, smiles, label, source_label, target_label,
semantic_label, split, scaffold_smiles, teacher_pred,
teacher_prob_0, teacher_prob_1, teacher_correct
```

SFT CSV/JSONL rows additionally contain:

```text
prompt, completion, instruction, output, response, reference_fragment,
raw_fragment, core_fragment, candidate_strategy, atom_ratio,
residual_smiles, cf_drop, cf_flip, oracle_ok
```

The compatibility mapping is:

```text
prompt      -> existing SFT prompt
completion  -> newline + core_fragment
instruction -> prompt alias
output      -> core_fragment alias
response    -> core_fragment legacy alias
```

PPO CSVs include `prompt` in addition to the common fields. The existing stable
PPO loader can consume them directly using `parent_smiles`, `label`, and
`prompt`. `molecule_id` remains in each raw row so later trainer coverage
instrumentation can report `num_unique_parents_seen`; the current trainer does
not yet emit those dataset-coverage counters, and this adapter does not modify
PPO core code.

## 5. Leakage and quality gates

Before target construction, the builder requires the full source counts:

```text
train=1448, val=260, calibration=235, test=217
```

Every source row must be label 1, teacher prediction 1, teacher-correct, a
single sanitized non-dummy molecule, and carry the expected split. The builder
rejects overlap in molecule ID, canonical SMILES, or scaffold between train and
validation and between either training split and calibration/test.

Smoke sampling happens only after those full-source checks. It uses the same
scaffold+atom-size stratified sampler as the AIDS builder, capped at 100 train
and 40 validation parents.

## 6. Outputs

Full output root:

```text
outputs/hpc/mutagenicity/sft_ppo_data_v1
```

Smoke output root:

```text
outputs/hpc/mutagenicity/sft_ppo_data_v1_smoke
```

In addition to the requested CSVs, the builder writes
`mutagenicity_sft_train.jsonl` and `mutagenicity_sft_val.jsonl` so the current
JSONL-based `scripts/train_sft.py` can consume the result directly.

## 7. Commands

Submit from the project root so Slurm can open `logs/%j.out`:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/build_mutagenicity_sft_ppo_data_smoke.sh
```

Full build:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/build_mutagenicity_sft_ppo_data_full.sh
```

The builder itself can be inspected with:

```bash
python scripts/data/build_mutagenicity_sft_ppo_data.py --help
```
