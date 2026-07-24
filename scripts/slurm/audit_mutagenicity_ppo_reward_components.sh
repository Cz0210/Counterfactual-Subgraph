#!/bin/bash
# Read-only reward component audit for the completed transfer PPO run.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=mut_reward_audit
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"; fi
[[ -n "$PROJECT_ROOT" ]] || { echo "[ERROR] Could not determine PROJECT_ROOT" >&2; exit 2; }
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/hpc/mutagenicity/ppo_stable_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/hpc/mutagenicity/audits/ppo_reward_components_v1}"
MARGIN="${STRICT_FLIP_REWARD_MARGIN:-0.5}"
mkdir -p "$PROJECT_ROOT/logs"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STRICT_FLIP_REWARD_MARGIN=$MARGIN"
echo "git_commit=$(git rev-parse HEAD || true)"

python scripts/audit_mutagenicity_ppo_reward_components.py \
  --config configs/hpc.yaml \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --strict-flip-reward-margin "$MARGIN"
