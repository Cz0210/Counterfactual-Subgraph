#!/bin/bash
# Static provenance/coverage audit for one completed Fresh PPO run.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=mut_fresh_ppo_audit
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

RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed Fresh PPO run}"
EXPECTED_MODE="${EXPECTED_MODE:?Set EXPECTED_MODE to smoke, medium, or full}"
mkdir -p "$PROJECT_ROOT/logs"
echo "RUN_DIR=$RUN_DIR EXPECTED_MODE=$EXPECTED_MODE"
echo "git_commit=$(git rev-parse HEAD || true)"

python scripts/audit_mutagenicity_ppo_fresh.py \
  --config configs/hpc.yaml \
  --run-dir "$RUN_DIR" \
  --expected-mode "$EXPECTED_MODE"
