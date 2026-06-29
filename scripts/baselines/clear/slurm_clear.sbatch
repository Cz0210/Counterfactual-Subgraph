#!/bin/bash
#SBATCH --job-name=clear
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baselines/clear/slurm_%x_%j.out
#SBATCH --error=logs/baselines/clear/slurm_%x_%j.err

set -euo pipefail

DATASET="${1:-}"
STAGE="${2:-}"

if [ -z "${DATASET}" ] || [ -z "${STAGE}" ]; then
  echo "Usage: sbatch scripts/baselines/clear/slurm_clear.sbatch <dataset> <stage>" >&2
  exit 2
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/baselines/clear/run_clear.sh" ]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${SCRIPT_DIR}/../../.."
fi

mkdir -p logs/baselines/clear

echo "===== CLEAR SLURM ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "dataset=${DATASET}"
echo "stage=${STAGE}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "======================="

set +u
source ~/.bashrc
conda activate clear
set -u

bash scripts/baselines/clear/run_clear.sh "${DATASET}" "${STAGE}"
