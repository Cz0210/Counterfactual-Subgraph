#!/bin/bash
#SBATCH --job-name=comrecgc_gate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET="${DATASET:-}"
MODE="${MODE:-smoke}"
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
python scripts/baselines/comrecgc/gate_run.py --dataset "$DATASET" --mode "$MODE" --base-root "$BASE_ROOT"
