#!/bin/bash
#SBATCH --job-name=bace_ours_audit_v2
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs
SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_wnode_prefix_v2}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_ours_wnode_prefix_v2}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$SELECTOR_ROOT/frozen_selection.json" "$OUTPUT_DIR/run_manifest.json"; do test -s "$path"; done
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo "[BACE_OURS_AUDIT_V2_VALIDATE_OK]"; exit 0; fi
python scripts/audit_bace_ours_wnode_prefix_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --mode final --root "$OUTPUT_DIR" --selector-root "$SELECTOR_ROOT" \
  --output-json "$OUTPUT_DIR/bace_ours_wnode_prefix_v2_audit.json"
echo "[BACE_OURS_AUDIT_V2_SUCCESS]"
