# CLEAR Baseline Workflow

This document describes how to run the official CLEAR / GraphCFE baseline from this repository without modifying the official source under `baselines/clear_official/src/`.

## 1. Local Development Workflow

Use VSCode + Codex locally to edit only project-owned wrapper scripts and documentation. Do not edit CLEAR official source files unless there is a separately documented reason.

Typical local Git workflow:

```bash
git status
git add scripts/baselines/clear scripts/hpc_pull_clear.sh docs/clear_baseline_workflow.md .gitignore docs/decisions.md
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

## 6. Examples

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

## 7. Direct Wrapper Use

For lightweight interactive debugging on a compute node, the non-Slurm wrapper is:

```bash
bash scripts/baselines/clear/run_clear.sh community pred
```

Main experiments should be submitted with `sbatch`.
