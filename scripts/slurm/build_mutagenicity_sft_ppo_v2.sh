#!/bin/bash
#SBATCH --job-name=mut_sft_v2
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=08:00:00
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

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_ppo_data_v2}"
mkdir -p "$PROJECT_ROOT/logs"
echo "hostname=$(hostname)"
echo "pwd=$PWD"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python scripts/build_mutagenicity_sft_ppo_v2.py \
  --config configs/hpc.yaml \
  --teacher-consistent-root outputs/hpc/datasets/mutagenicity_v1_teacher_consistent \
  --teacher-path outputs/hpc/oracle/final/mutagenicity_rf_v1/mutagenicity_rf_model.pkl \
  --output-dir "$OUTPUT_DIR"

FINAL_LINK="$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v2"
mkdir -p "$(dirname "$FINAL_LINK")"
if [[ -e "$FINAL_LINK" || -L "$FINAL_LINK" ]]; then
  echo "[ERROR] Refusing to replace existing final link: $FINAL_LINK" >&2
  exit 2
fi
ln -s "$OUTPUT_DIR" "$FINAL_LINK"
