# CLEAR Baseline Workflow

This document describes how to run the official CLEAR / GraphCFE baseline from this repository without modifying the official source under `baselines/clear_official/src/`.

## 1. Local Development Workflow

Use VSCode + Codex locally to edit only project-owned wrapper scripts, patch files, and documentation. Do not commit direct dirty changes under the CLEAR official submodule; project-owned compatibility changes should live under `patches/clear_official/`.

Typical local Git workflow:

```bash
git status
git add patches/clear_official scripts/baselines/clear scripts/hpc_pull_clear.sh docs/clear_baseline_workflow.md .gitignore docs/decisions.md
git commit -m "Add CLEAR baseline HPC wrappers"
git push
```

## 2. HPC Sync Workflow

After pushing locally, log into the HPC, enter the project root, and run:

```bash
bash scripts/hpc_pull_clear.sh
```

The script performs:

```bash
git pull --recurse-submodules
git submodule sync --recursive
git submodule update --init --recursive
```

It also creates CLEAR runtime directories and reports which dataset files are missing.
It applies project-owned CLEAR compatibility patches from
`patches/clear_official/` so that the official checkout saves the CFE generator
checkpoints needed by the test stage.

## 3. Dataset Location

CLEAR official code expects paths relative to `baselines/clear_official/src/`, such as `../dataset` and `../models_save`. Therefore dataset files should be copied to:

```text
baselines/clear_official/dataset/
```

Required files:

```text
AIDS main experiment:
  source CSV: data/raw/AIDS/HIV.csv
  SMILES_COLUMN=smiles
  LABEL_COLUMN=HIV_active
  TARGET_LABEL=1
  prepared CLEAR files:
    baselines/clear_official/dataset/aids_full.pickle
    baselines/clear_official/dataset/aids_datasplit.pickle

community:
  baselines/clear_official/dataset/community_3.pickle
  baselines/clear_official/dataset/community_datasplit.pickle

ogbg_molhiv:
  baselines/clear_official/dataset/ogbg_molhiv_full.pickle
  baselines/clear_official/dataset/ogbg_molhiv_datasplit.pickle

imdb_m:
  baselines/clear_official/dataset/imdb_m.pickle
  baselines/clear_official/dataset/imdb_m_datasplit.pickle
  baselines/clear_official/dataset/IMDBMULTI.mat
```

`ogbg_molhiv` was used as a CLEAR engineering validation dataset. It is not the
AIDS/HIV main result. The AIDS main baseline should use CLEAR dataset name
`aids`, prepared from `data/raw/AIDS/HIV.csv` with `SMILES_COLUMN=smiles` and
`LABEL_COLUMN=HIV_active`. This follows the repository-wide AIDS/HIV dataset
contract in `docs/DATASET_CONTRACT.md`.

Prepare the AIDS pickles on HPC with:

```bash
sbatch scripts/slurm/prepare_clear_aids_dataset.sh
```

This writes:

```text
baselines/clear_official/dataset/aids_full.pickle
baselines/clear_official/dataset/aids_datasplit.pickle
outputs/hpc/baselines/clear/aids/dataset/clear_aids_dataset_summary.json
```

The split is a deterministic stratified CLEAR-internal train/val/test split
with default ratios `0.8/0.1/0.1`, seed `0`, and `10` split repetitions so it
matches CLEAR's official `exp_num=10` loader expectation. The AIDS preparation
keeps molecules with at most `100` atoms by default (`MAX_NUM_NODES=100`) to
reduce sample filtering caused by CLEAR's dense graph padding. This split is
used for CLEAR's own prediction/CFE training. It is not the same thing as the
historical `hiv_quick` full label-1 evaluation pool.

Do not place `ogbg_molhiv` CLEAR metrics in the final AIDS/HIV baseline table.
Those runs are engineering validation only.

### AIDS Class Imbalance

The canonical AIDS/HIV source is highly imbalanced:

```text
data/raw/AIDS/HIV.csv
HIV_active=0: 39684
HIV_active=1: 1443
```

The repository currently does not contain a ready-to-use balanced parent
molecule classification dataset for CLEAR graph prediction. Existing
`balanced`, `label01`, `sft_v3`, PPO prompt, candidate pool, selector, and
`hiv_quick` artifacts are not CLEAR graphPred training datasets. They must not
be used as a substitute for `aids_full.pickle` / `aids_datasplit.pickle`.

For `dataset=aids`, the project patch
`patches/clear_official/004_aids_weighted_graphpred.patch` keeps the same
prepared AIDS max100 x10 dataset and fixes class imbalance inside CLEAR
graphPred training by using class-weighted `CrossEntropyLoss`. The class weight
is computed from the current training split:

```text
weight_c = total_train / (num_classes * count_c)
```

The patched pred run logs:

```text
[CLEAR_AIDS_CLASS_BALANCE]
train_label_counts=...
class_weights=...
```

It also computes prediction metrics over the full validation/test split rather
than averaging batch-level AUC/F1 values. AIDS graphPred checkpoint selection
uses validation F1 by default instead of only validation loss, and the logs
include `y_true_counts`, `y_pred_counts`, `positive_pred_rate`,
`balanced_accuracy`, `f1`, and `roc_auc`.

## 4. Why Data and Checkpoints Are Not Committed

Datasets, checkpoints, model weights, generated logs, and experiment outputs can be large and machine-specific. They should not be committed to ordinary Git history.

Ignored runtime paths include:

```text
baselines/clear_official/dataset/
baselines/clear_official/models_save/
baselines/clear_official/logs/
logs/baselines/clear/
outputs/
```

If model weights must be versioned through Git, use Git LFS instead of normal Git blobs.

## 5. Running CLEAR on HPC

The Slurm entrypoint is:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch <dataset> <stage>
```

On the current HPC, the CLEAR wrapper defaults to the shared conda environment
`smiles_pip118`. Override it only when needed:

```bash
CLEAR_CONDA_ENV=smiles_pip118 sbatch scripts/baselines/clear/slurm_clear.sbatch community pred
```

CLEAR jobs should be submitted to the A800 GPU queue through this wrapper. The
Slurm script requests one A800 GPU, 7 CPU cores, and 32G memory:

```text
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
```

Supported datasets:

```text
community
ogbg_molhiv
aids
imdb_m
```

Supported stages:

```text
pred
train
test
export_test
export_test_small
baseline_random
baseline_IST
baseline_RM
all
```

The wrapper always runs official CLEAR from:

```text
baselines/clear_official/src/
```

This preserves CLEAR's official relative path assumptions.

## 6. Checkpoint Behavior

The `pred` stage trains and saves the graph prediction model under:

```text
baselines/clear_official/models_save/prediction/weights_graphPred__<dataset>.pt
```

The `train` stage runs CLEAR and, through the project patch
`patches/clear_official/001_save_cfe_checkpoints.patch`, saves CFE generator
checkpoints under:

```text
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp0_epoch900.pt
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp1_epoch900.pt
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp2_epoch900.pt
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp0_epoch<epochs>.pt
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp1_epoch<epochs>.pt
baselines/clear_official/models_save/weights_graphCFE_CLEAR_<dataset>_exp2_epoch<epochs>.pt
```

The official CLEAR test path loads epoch 900 by default. The wrapper checks for
the three expected epoch-900 files before running `test`. If an exp-specific
epoch-900 file is absent but another CFE checkpoint exists for that experiment,
the wrapper creates a symlink from the highest available epoch to the epoch-900
filename. If no CFE checkpoint exists, `test` fails early with a clear
`[CLEAR_CKPT_ERROR]` message. In that case, rerun `train`.

The patch is idempotent: `scripts/baselines/clear/apply_clear_patches.sh`
checks for the marker `CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT` and skips if the
patch is already present.

## 7. Per-Instance Export

`pred`, `train`, and `test` preserve the official CLEAR workflow and aggregate
metrics. The project-owned `export_test` stage is added for unified
CCRCov/action-rule evaluation. It keeps the official `test` aggregate metrics,
then exports per-instance counterfactual graphs for the full test split.

The export patch marker is:

```text
CLEAR_WRAPPER_EXPORT_TEST_COUNTERFACTUALS
```

Run:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch aids export_test
```

Default output path:

```text
outputs/hpc/baselines/clear/aids/test_exports/
```

Per experiment, the export writes:

```text
clear_aids_exp0_test_counterfactuals.pkl
clear_aids_exp0_test_counterfactuals.jsonl
clear_aids_exp1_test_counterfactuals.pkl
clear_aids_exp1_test_counterfactuals.jsonl
clear_aids_exp2_test_counterfactuals.pkl
clear_aids_exp2_test_counterfactuals.jsonl
```

The pickle files contain full arrays: original features/adjacency, reconstructed
counterfactual features, thresholded counterfactual adjacency, and
counterfactual adjacency probabilities. The JSONL files contain lightweight
metadata such as labels, target labels, original/CF prediction probabilities,
predicted labels, original node count, batch-level CLEAR metrics, and
`source=CLEAR`.

For small debug export:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch aids export_test_small
```

or cap records:

```bash
CLEAR_EXPORT_MAX_ITEMS=20 sbatch scripts/baselines/clear/slurm_clear.sbatch aids export_test
```

Exported files under `outputs/hpc/baselines/clear/` are runtime artifacts and
must not be committed. They are intended as the next input for building a CLEAR
candidate/action pool and then running unified SuppCov, CCRCov, CFDrop,
FlipRate, Cost, StructRed, and CovRed evaluation.

## 8. Candidate/Action Pool Conversion

After `export_test` finishes, convert CLEAR original/counterfactual graph pairs
into the project-owned unified candidate/action pool:

```bash
python scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py \
  --export-dir outputs/hpc/baselines/clear/aids/test_exports \
  --dataset aids \
  --out-jsonl outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl \
  --out-summary outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs_summary.json \
  --include-full-graphs
```

The Slurm wrapper is:

```bash
sbatch scripts/slurm/convert_clear_exports_to_candidate_pool.sh
```

For AIDS with full graph arrays preserved:

```bash
DATASET=aids \
EXPORT_DIR=outputs/hpc/baselines/clear/aids/test_exports \
OUT_JSONL=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl \
OUT_SUMMARY=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs_summary.json \
INCLUDE_FULL_GRAPHS=1 \
sbatch scripts/slurm/convert_clear_exports_to_candidate_pool.sh
```

For a smoke conversion:

```bash
MAX_RECORDS=100 sbatch scripts/slurm/convert_clear_exports_to_candidate_pool.sh
```

The conversion keeps CLEAR official diagnostic fields, including
`official_flip`, `official_target_success`, `official_original_correct`, and
the official prediction probabilities. CLEAR official flip rate may be zero;
that is not used as the final paper-facing result. Final `FlipRate`, `CFDrop`,
and `CCRCov` must be recomputed later by the unified frozen teacher/oracle.

The candidate pool stores action-level graph differences:

- edge additions and deletions from `original_adj` to `cf_adj`;
- continuous node-feature changes from `original_x` to `cf_x`;
- aggregate action costs such as edge cost, feature L1/L2 cost, and total cost.

By default, full graph arrays are not written to JSONL to keep the pool small.
Use `--include-full-graphs` only for debugging. Files under
`outputs/hpc/baselines/clear/` remain runtime artifacts and must not be
committed.

## 9. Unified Evaluation

The final CLEAR baseline metrics must be recomputed with an explicitly recorded
teacher path. CLEAR official `validity` and `official_flip` are retained only as
diagnostics. In particular, an `official_flip_rate` of zero is allowed and
should not be used to filter candidates before unified evaluation.

For AIDS/HIV, the historical RF oracle at
`outputs/hpc/oracle/aids_rf_model.pkl` is a SMILES/Morgan-fingerprint oracle.
It cannot directly consume CLEAR's continuous reconstructed graph tensors.
Final strict-flip CCRCov therefore requires a full-graph candidate pool and a
graph-to-teacher adapter or another explicitly documented unified teacher path.
The current graph-teacher path for CLEAR AIDS is:

```text
TEACHER_KIND=clear_graphpred
TEACHER_PATH=baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt
```

The evaluator recomputes `strict_flip_eval`, `cf_drop_eval`, and label
probabilities from `original_adj`, `cf_adj`, `original_x`, and `cf_x`.
`official_flip` remains a diagnostic comparison field and is not the final
FlipRate.

Evaluation entrypoint:

```bash
python scripts/baselines/clear/evaluate_clear_candidate_pool.py \
  --candidate-pool outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl \
  --dataset aids \
  --teacher-kind clear_graphpred \
  --teacher-path baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt \
  --out-dir outputs/hpc/baselines/clear/aids/eval \
  --cf-mode strict_flip \
  --batch-size 128 \
  --top-k 1,5,10,20 \
  --distance-method action
```

Slurm entrypoint:

```bash
DATASET=aids \
CANDIDATE_POOL=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl \
TEACHER_KIND=clear_graphpred \
TEACHER_PATH=baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt \
OUT_DIR=outputs/hpc/baselines/clear/aids/eval_clear_graphpred \
sbatch scripts/slurm/evaluate_clear_candidate_pool.sh
```

Smoke diagnostics:

```bash
DATASET=aids \
MAX_CANDIDATES=100 \
TEACHER_KIND=clear_graphpred \
TEACHER_PATH=baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt \
sbatch scripts/slurm/evaluate_clear_candidate_pool.sh
```

The default non-graph candidate pool does not include full graph arrays in JSONL.
For `TEACHER_KIND=clear_graphpred`, use the `.with_graphs.jsonl` pool produced
with `INCLUDE_FULL_GRAPHS=1`. The evaluator refuses to report final strict
`FlipRate`, `CFDrop`, or `CCRCov` from CLEAR official prediction fields. For
cost-only pipeline smoke checks, use:

```bash
ALLOW_ACTION_ONLY=1 MAX_CANDIDATES=100 sbatch scripts/slurm/evaluate_clear_candidate_pool.sh
```

Action-only output is diagnostic only and must not replace final
native-action CCRCov results.

## 9.1 CLEAR-RF Unified Fair-Table Evaluation

`TEACHER_KIND=clear_graphpred` with action distance is a CLEAR native diagnostic.
It is useful for checking whether CLEAR changed its own graph classifier, but it
is not the final fair main-table result when Ours and GT-FullGraph are evaluated
with the shared RF oracle.

For the final fair table, use `CLEAR-RF-FullGraph`:

```text
parent set: outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
SMILES_COL=smiles
LABEL_COL=label
TARGET_LABEL=1
teacher: outputs/hpc/oracle/aids_rf_model.pkl
distance: GREED-GED primary, MolCLR embedding supplementary
CF_MODE=strict_flip
```

First audit whether CLEAR full graph arrays can be conservatively converted to
RF-readable SMILES:

```bash
sbatch scripts/slurm/clear_aids_audit_rf_unified.sh
```

Then convert valid CLEAR counterfactual full graphs:

```bash
sbatch scripts/slurm/clear_aids_convert_rf_fullgraph_candidates.sh
```

The conversion is intentionally conservative. CLEAR `cf_x` is a continuous
decoder output, not a reliable atom vocabulary. The adapter never treats raw
float feature values as atom labels. It reuses the AIDS feature schema from
`prepare_clear_aids_dataset.py`: current AIDS nodes are descriptor vectors with
`original_x[:, 2] = atomic_num / 100`. If a future export contains true
one-hot or soft categorical atom rows, the adapter can decode them with row-wise
argmax using the fixed AIDS atom vocabulary (`C`, `N`, `O`, `F`, `S`, `Cl`).
For the current CLEAR AIDS pool, atom identity is recovered from the descriptor
slot and `cf_adj` provides the counterfactual topology.

`cf_adj` is symmetrized, its diagonal is removed, and edges above
`ADJ_THRESHOLD=0.5` are considered as single bonds. During RDKit conversion the
adapter adds candidate bonds in descending `cf_adj` score and skips edges that
would exceed simple single-bond valence limits. The conversion summary records
`original_x_onehot_like_rate`, `cf_x_onehot_like_rate`,
`cf_x_continuous_rate`, argmax distributions, decode-mode counts,
`cf_adj_min/max/mean`, skipped-valence information, and the node-mask source.

This writes:

```text
outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv
outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.jsonl
outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates_summary.json
```

The conversion must pass a quality gate before downstream fair evaluation:

```text
MIN_VALID_CANDIDATES=20
MIN_VALID_RATE=0.001
ADJ_THRESHOLD=0.5
```

If the gate fails, the script prints
`[CLEAR_RF_CONVERT_FAILED_QUALITY_GATE]` and exits non-zero. A completed job
with only a few valid SMILES must not be treated as a successful fair-table
candidate set.

The converter also validates the written CSV by re-reading `candidate_smiles`
with RDKit. To run only this quick validation on an existing CSV:

```bash
VALIDATE_ONLY=1 sbatch scripts/slurm/clear_aids_convert_rf_fullgraph_candidates.sh
```

The validation reports whether the `candidate_smiles` column exists, RDKit
valid count/rate, and min/max atom count. The CSV check is part of the quality
gate, so a file with fewer than `MIN_VALID_CANDIDATES` RDKit-valid candidates
fails even if the conversion process itself did not crash.

Run GREED-GED final-table evaluation:

```bash
sbatch scripts/slurm/clear_aids_eval_rf_greed_full.sh
```

Run MolCLR embedding evaluation after the embedding directory includes CLEAR
candidate SMILES. The existing precompute script accepts
`CLEAR_FULLGRAPH_CANDIDATES_PATH`:

```bash
CLEAR_FULLGRAPH_CANDIDATES_PATH=outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv \
sbatch scripts/slurm/molclr_hiv_precompute_embeddings_full.sh

sbatch scripts/slurm/clear_aids_eval_rf_molclr_full.sh
```

For the MolCLR-Node-FGW auxiliary line, use fullgraph-only mode so CLEAR is
scored as a candidate set without re-running ours in the same job:

```bash
RUN_OURS=0 RUN_FULLGRAPH=1 RUN_GT_FULLGRAPH=0 \
CLEAR_FULLGRAPH_CANDIDATES_PATH=outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv \
FULLGRAPH_METHOD_NAME=CLEAR-RF-FullGraph \
OUTPUT_DIR=outputs/hpc/eval/ccrcov_molclr_node_fgw_clear_smoke \
sbatch scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

If the audit reports `rf_oracle_usable=false`, do not report CLEAR as an
RF-unified final-table method. Keep the clear_graphpred/action-distance result
as diagnostic only.

## 10. Examples

Community:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch community pred
sbatch scripts/baselines/clear/slurm_clear.sbatch community train
sbatch scripts/baselines/clear/slurm_clear.sbatch community test
sbatch scripts/baselines/clear/slurm_clear.sbatch community all
```

OGBG-MolHIV:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv pred
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv train
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv test
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv export_test
sbatch scripts/slurm/convert_clear_exports_to_candidate_pool.sh
sbatch scripts/slurm/evaluate_clear_candidate_pool.sh
```

AIDS/HIV main CLEAR run:

```bash
sbatch scripts/slurm/prepare_clear_aids_dataset.sh
sbatch scripts/baselines/clear/slurm_clear.sbatch aids pred
sbatch scripts/baselines/clear/slurm_clear.sbatch aids train
sbatch scripts/baselines/clear/slurm_clear.sbatch aids test
sbatch scripts/baselines/clear/slurm_clear.sbatch aids export_test
DATASET=aids \
EXPORT_DIR=outputs/hpc/baselines/clear/aids/test_exports \
OUT_JSONL=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl \
OUT_SUMMARY=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs_summary.json \
INCLUDE_FULL_GRAPHS=1 \
sbatch scripts/slurm/convert_clear_exports_to_candidate_pool.sh
```

Dependency-submitted AIDS/HIV run:

```bash
jid_prep=$(sbatch --parsable scripts/slurm/prepare_clear_aids_dataset.sh)
jid_pred=$(sbatch --parsable --dependency=afterok:${jid_prep} scripts/baselines/clear/slurm_clear.sbatch aids pred)
jid_train=$(sbatch --parsable --dependency=afterok:${jid_pred} scripts/baselines/clear/slurm_clear.sbatch aids train)
jid_test=$(sbatch --parsable --dependency=afterok:${jid_train} scripts/baselines/clear/slurm_clear.sbatch aids test)
jid_export=$(sbatch --parsable --dependency=afterok:${jid_test} scripts/baselines/clear/slurm_clear.sbatch aids export_test)
jid_convert=$(DATASET=aids EXPORT_DIR=outputs/hpc/baselines/clear/aids/test_exports OUT_JSONL=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl OUT_SUMMARY=outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs_summary.json INCLUDE_FULL_GRAPHS=1 sbatch --parsable --dependency=afterok:${jid_export} scripts/slurm/convert_clear_exports_to_candidate_pool.sh)
echo "prep=${jid_prep} pred=${jid_pred} train=${jid_train} test=${jid_test} export=${jid_export} convert=${jid_convert}"
```

Dependency-submitted OGBG-MolHIV run:

```bash
jid_pred=$(sbatch --parsable scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv pred)
jid_train=$(sbatch --parsable --dependency=afterok:${jid_pred} scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv train)
jid_test=$(sbatch --parsable --dependency=afterok:${jid_train} scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv test)
echo "pred=${jid_pred} train=${jid_train} test=${jid_test}"
```

IMDB-M:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch imdb_m pred
sbatch scripts/baselines/clear/slurm_clear.sbatch imdb_m train
sbatch scripts/baselines/clear/slurm_clear.sbatch imdb_m test
```

Baselines:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch community baseline_random
sbatch scripts/baselines/clear/slurm_clear.sbatch community baseline_IST
sbatch scripts/baselines/clear/slurm_clear.sbatch community baseline_RM
```

## 11. Direct Wrapper Use

For lightweight interactive debugging on a compute node, the non-Slurm wrapper is:

```bash
bash scripts/baselines/clear/run_clear.sh community pred
```

Main experiments should be submitted with `sbatch`.
