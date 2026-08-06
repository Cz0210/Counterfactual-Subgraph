#!/bin/bash
#SBATCH --job-name=comrecgc_mut_trace
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
export PYTHONHASHSEED=0

REFERENCE="${REFERENCE:-outputs/hpc/baselines/comrecgc/mutagenicity/smoke_comrecgc_smoke_budget_retry_20260806_v4/generation/counterfactuals.pt}"
REFERENCE_SHA256="${REFERENCE_SHA256:-060879cbaf69b1e3279301350f587cab809d48991559a80ff5227c46466df8d0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity/recovery_trace_v1/generation}"
TRACE_DIR="${TRACE_DIR:-$OUTPUT_DIR/trace}"
DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/gnn/model_best.pth}"
DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
[[ "$(sha256sum "$REFERENCE" | awk '{print $1}')" == "$REFERENCE_SHA256" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] trace reference SHA256 mismatch" >&2; exit 2;
}
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete=$OUTPUT_DIR" >&2; exit 2; }
for input in "$DATASET_DIR" "$GNN_CHECKPOINT" "$DISTANCE_CHECKPOINT"; do
  [[ -e "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_trace_replay parents=64 steps=100 heads=2 cap=200 sample=128 seed=0"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/run_generation.py \
  --route project \
  --dataset mutagenicity \
  --mode smoke \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --dataset-dir "$DATASET_DIR" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --distance-checkpoint "$DISTANCE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --parent-limit 64 \
  --device cuda:0 \
  --batch-size 128 \
  --trace-output-dir "$TRACE_DIR" \
  --parity-reference "$REFERENCE"
test -s "$OUTPUT_DIR/trace_parity.json"
test -s "$TRACE_DIR/candidate_action_lineage.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_MUT_TRACE_PARITY_SUCCESS]"
