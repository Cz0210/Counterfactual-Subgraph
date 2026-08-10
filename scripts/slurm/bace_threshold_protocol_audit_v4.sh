#!/bin/bash
#SBATCH --job-name=bace_threshold_protocol_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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

MUT_THRESHOLD=${MUT_THRESHOLD:-$ARTIFACT_ROOT/outputs/hpc/mutagenicity/selectors/wnode_prefix_full_p235_c683_k20_v1/thresholds.json}
BACE_OLD_THRESHOLD=${BACE_OLD_THRESHOLD:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
BACE_CONNECTED_THRESHOLD=${BACE_CONNECTED_THRESHOLD:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_threshold_protocol_v4}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$MUT_THRESHOLD" "$BACE_OLD_THRESHOLD" "$BACE_CONNECTED_THRESHOLD"; do
  test -s "$path" || { echo "[BACE_THRESHOLD_PROTOCOL_V4_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
args=(
  python scripts/audit_common_threshold_protocol.py
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --mut-threshold "$MUT_THRESHOLD"
  --bace-old-threshold "$BACE_OLD_THRESHOLD"
  --bace-connected-threshold "$BACE_CONNECTED_THRESHOLD"
  --output-dir "$OUTPUT_DIR"
)

echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_THRESHOLD_PROTOCOL_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_THRESHOLD_PROTOCOL_V4_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
echo "[BACE_THRESHOLD_PROTOCOL_V4_SUCCESS]"
