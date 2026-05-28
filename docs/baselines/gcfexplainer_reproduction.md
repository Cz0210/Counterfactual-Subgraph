# Official GCFExplainer AIDS Reproduction

This document records the isolated scaffold for running the official
GCFExplainer AIDS sanity check on HPC.

## Scope

The official GCFExplainer code under `baselines/gcfexplainer_official/` is
treated as a frozen third-party baseline. The Slurm wrappers in this repository
only activate a separate conda environment, run the upstream scripts, copy their
outputs, and collect the printed summary metrics.

This AIDS reproduction is a sanity check for the official graph baseline. It is
not a fair comparison against the current HIV/SMILES counterfactual fragment
generation method, because the task representation, data contract, model family,
and evaluation protocol are different.

## Local Preparation

From the repository root, check that the official files are present:

```bash
test -f baselines/gcfexplainer_official/vrrw.py
test -f baselines/gcfexplainer_official/summary.py
test -f baselines/gcfexplainer_official/environment.yml
```

The wrappers fail fast if either `vrrw.py` or `summary.py` is missing.

If `baselines/gcfexplainer_official/` is not already tracked by git, make sure it
is either committed or otherwise copied to the HPC checkout before running the
Slurm jobs.

## HPC Environment

Create an environment separate from the main `smiles_pip118` project
environment. The default wrapper environment name is `gcfexplainer_py38`:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
conda env create -n gcfexplainer_py38 -f baselines/gcfexplainer_official/environment.yml
```

If the upstream YAML creates an environment named `gcf` instead, either rename
it or submit with `CONDA_ENV=gcf`.

Quick check:

```bash
conda activate gcfexplainer_py38
python --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Bashrc And Conda Activation

Do not initialize the GCFExplainer Slurm wrappers with `source ~/.bashrc`.
On this HPC, `~/.bashrc` can transitively source `/etc/bashrc`, where
`BASHRCSOURCED` may be referenced before it is defined. If `set -u` is already
enabled, Bash exits with:

```text
/etc/bashrc: line 12: BASHRCSOURCED: unbound variable
```

The GCFExplainer wrappers therefore source conda directly:

```bash
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-gcfexplainer_py38}
set -u
```

If a new wrapper must source a shell initialization file, temporarily disable
unbound-variable checking with `set +u` before the source step and re-enable it
after conda activation.

## Train Official AIDS GNN

The official `vrrw.py` path expects the AIDS GNN checkpoint at:

```text
baselines/gcfexplainer_official/data/aids/gnn/model_best.pth
```

Train it first if the checkpoint is missing:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/gcfexplainer/train_aids_gnn.sh
```

Optional overrides:

```bash
GNN_EPOCHS=1000 CUDA_ID=0 CONDA_ENV=gcfexplainer_py38 \
  sbatch scripts/slurm/gcfexplainer/train_aids_gnn.sh
```

The training wrapper writes Slurm logs to `logs/%j.out` and `logs/%j.err`, and
mirrors stdout/stderr to:

```text
outputs/hpc/gcfexplainer/official_aids_gnn_train/${SLURM_JOB_ID}/gnn_train.log
```

At the end it checks for `data/aids/gnn/model_best.pth` and prints the last 80
lines of the official `data/aids/gnn/log.txt`.

## Slurm Commands

Run the full AIDS reproduction in one job:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/gcfexplainer/reproduce_aids_all.sh
```

Or run the two stages separately:

```bash
mkdir -p logs
sbatch scripts/slurm/gcfexplainer/reproduce_aids_vrrw.sh
sbatch scripts/slurm/gcfexplainer/reproduce_aids_summary.sh
```

Override the conda environment if needed:

```bash
CONDA_ENV=gcf sbatch scripts/slurm/gcfexplainer/reproduce_aids_all.sh
```

All outputs are written under:

```text
outputs/hpc/gcfexplainer/official_aids/${SLURM_JOB_ID}/
```

Important files:

```text
vrrw.log
summary.log
official_aids_summary.json
official_aids_summary.csv
results/aids/
```

## Result Checks

Replace `${JOB_ID}` with the Slurm job id:

```bash
OUT=outputs/hpc/gcfexplainer/official_aids/${JOB_ID}
tail -n 80 "${OUT}/vrrw.log"
tail -n 120 "${OUT}/summary.log"
cat "${OUT}/official_aids_summary.json"
column -s, -t < "${OUT}/official_aids_summary.csv"
find "${OUT}/results/aids" -maxdepth 3 -type f | sort
```

If the parser cannot recognize the upstream `summary.py` text format, it still
writes `official_aids_summary.json` and `official_aids_summary.csv` with
`parse_ok=false`, `raw_summary_log_path`, and `error_message` so the failure is
visible rather than silently swallowed.
