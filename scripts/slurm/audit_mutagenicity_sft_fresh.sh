#!/bin/bash
# Static provenance audit for one completed Fresh Mutagenicity SFT run.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=mut_fresh_sft_audit
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

OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a completed Fresh SFT run}"
FORBIDDEN_ADAPTER="${FORBIDDEN_ADAPTER:-$PROJECT_ROOT/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500}"
mkdir -p "$PROJECT_ROOT/logs"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "FORBIDDEN_ADAPTER=$FORBIDDEN_ADAPTER"
echo "git_commit=$(git rev-parse HEAD || true)"

python scripts/audit_mutagenicity_sft_fresh.py \
  --config configs/hpc.yaml \
  --output-root "$OUTPUT_ROOT" \
  --forbidden-adapter-checkpoint "$FORBIDDEN_ADAPTER"
