#!/bin/bash
#SBATCH --job-name=bbbp_random_cf
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
: "${VARIANT:?VARIANT is required}"
: "${REFERENCE_CANDIDATE_JSONL:?REFERENCE_CANDIDATE_JSONL is required}"
: "${OUTPUT_JSONL:?OUTPUT_JSONL is required}"
: "${SUMMARY_JSON:?SUMMARY_JSON is required}"
args=(
  scripts/generate_bbbp_candidate_controls.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --parent-csv "${PARENT_CSV:-data/processed/BBBP/train.csv}"
  --variant "$VARIANT"
  --candidates-per-parent "${CANDIDATES_PER_PARENT:-8}"
  --reference-candidate-jsonl "$REFERENCE_CANDIDATE_JSONL"
  --seed "${SEED:-13}"
  --max-attempts "${MAX_ATTEMPTS:-200}"
  --output-jsonl "$OUTPUT_JSONL"
  --summary-json "$SUMMARY_JSON"
)
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then args+=(--validate-only); fi
if [ "${DRY_RUN:-0}" = "1" ]; then args+=(--dry-run); fi
python "${args[@]}"
