#!/bin/bash
#SBATCH --job-name=mut_sft_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT or SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD"

mkdir -p "$PROJECT_ROOT/logs"
echo "hostname=$(hostname)"
echo "pwd=$PWD"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"

python scripts/audit_mutagenicity_sft_target_selection.py \
  --config configs/hpc.yaml \
  --data-root outputs/hpc/mutagenicity/final/sft_ppo_data_v1 \
  --parent-root outputs/hpc/datasets/mutagenicity_v1_teacher_consistent \
  --teacher-path outputs/hpc/oracle/final/mutagenicity_rf_v1/mutagenicity_rf_model.pkl \
  --output-dir outputs/hpc/mutagenicity/audits/sft_target_selection_v1
