#!/bin/bash
#SBATCH --job-name=comrecgc_mut_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
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

BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/mutagenicity/full_chemrepair_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/generation}"
TRACE_DIR="${TRACE_DIR:-$OUTPUT_DIR/trace}"
DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/gnn/model_best.pth}"
DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete=$OUTPUT_DIR" >&2; exit 2; }
for input in "$DATASET_DIR" "$GNN_CHECKPOINT" "$DISTANCE_CHECKPOINT"; do
  [[ -e "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_full parents=1448 steps=50000 heads=5 k=100000 sample_size=10000 seed=0 trace_only=true transition_cache=exact_action_replay_with_bounded_expanded_lru_v1 expanded_capacity=5"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/run_generation.py \
  --route project \
  --dataset mutagenicity \
  --mode full \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --dataset-dir "$DATASET_DIR" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --distance-checkpoint "$DISTANCE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --parent-limit 1448 \
  --device cuda:0 \
  --batch-size 128 \
  --trace-output-dir "$TRACE_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$TRACE_DIR/candidate_action_lineage.json"
echo "[COMRECGC_MUT_FULL_GENERATION_SUCCESS]"
