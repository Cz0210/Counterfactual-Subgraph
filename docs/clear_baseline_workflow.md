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
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv export_test
```

Default output path:

```text
outputs/hpc/baselines/clear/ogbg_molhiv/test_exports/
```

Per experiment, the export writes:

```text
clear_ogbg_molhiv_exp0_test_counterfactuals.pkl
clear_ogbg_molhiv_exp0_test_counterfactuals.jsonl
clear_ogbg_molhiv_exp1_test_counterfactuals.pkl
clear_ogbg_molhiv_exp1_test_counterfactuals.jsonl
clear_ogbg_molhiv_exp2_test_counterfactuals.pkl
clear_ogbg_molhiv_exp2_test_counterfactuals.jsonl
```

The pickle files contain full arrays: original features/adjacency, reconstructed
counterfactual features, thresholded counterfactual adjacency, and
counterfactual adjacency probabilities. The JSONL files contain lightweight
metadata such as labels, target labels, original/CF prediction probabilities,
predicted labels, original node count, batch-level CLEAR metrics, and
`source=CLEAR`.

For small debug export:

```bash
sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv export_test_small
```

or cap records:

```bash
CLEAR_EXPORT_MAX_ITEMS=20 sbatch scripts/baselines/clear/slurm_clear.sbatch ogbg_molhiv export_test
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
  --export-dir outputs/hpc/baselines/clear/ogbg_molhiv/test_exports \
  --dataset ogbg_molhiv \
  --out-jsonl outputs/hpc/baselines/clear/ogbg_molhiv/candidate_pool/clear_ogbg_molhiv_candidate_pool.jsonl \
  --out-summary outputs/hpc/baselines/clear/ogbg_molhiv/candidate_pool/clear_ogbg_molhiv_candidate_pool_summary.json
```

The Slurm wrapper is:

```bash
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

## 9. Examples

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

## 10. Direct Wrapper Use

For lightweight interactive debugging on a compute node, the non-Slurm wrapper is:

```bash
bash scripts/baselines/clear/run_clear.sh community pred
```

Main experiments should be submitted with `sbatch`.
