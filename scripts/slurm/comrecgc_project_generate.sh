#!/bin/bash
#SBATCH --job-name=comrecgc_generate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
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
mkdir -p logs

DATASET="${DATASET:-}"
MODE="${MODE:-smoke}"
RESUME="${RESUME:-false}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] DATASET must be aids or mutagenicity" >&2; exit 2;
}
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] MODE must be smoke or full" >&2; exit 2;
}
[[ "$RESUME" == "false" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] project generation has no proven cross-job RNG/state resume; use a fresh versioned output" >&2; exit 2;
}
if [[ "$MODE" == "smoke" ]]; then
  EXPECTED_PARENT_LIMIT=64
  TRANSITION_STATE_POLICY="pinned_upstream_in_memory_transitions_v1"
else
  [[ "$DATASET" == "aids" ]] && EXPECTED_PARENT_LIMIT=1283 || EXPECTED_PARENT_LIMIT=1448
  TRANSITION_STATE_POLICY="authoritative_backing_live_graph_resolution_v2"
fi
PARENT_LIMIT="${PARENT_LIMIT:-$EXPECTED_PARENT_LIMIT}"
[[ "$PARENT_LIMIT" == "$EXPECTED_PARENT_LIMIT" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] parent_limit=$PARENT_LIMIT expected=$EXPECTED_PARENT_LIMIT" >&2; exit 2;
}
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/generation}"
TRACE_DIR="${TRACE_DIR:-$OUTPUT_DIR/trace}"
GRAPH_STATE_DIR="${GRAPH_STATE_DIR:-$OUTPUT_DIR/graph_state}"
if [[ "$RESUME" != "true" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[COMRECGC_CONFIG_ERROR] non-empty output with RESUME=false: $OUTPUT_DIR" >&2; exit 2
fi
if [[ "$DATASET" == "aids" ]]; then
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}"
  SOURCE_CSV="${SOURCE_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  GNN_CHECKPOINT="${GNN_CHECKPOINT:-outputs/hpc/gcfexplainer_hiv_csv/gnn/model_best.pth}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}"
  SOURCE_ARGS=(--source-csv "$SOURCE_CSV")
else
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
  GNN_CHECKPOINT="${GNN_CHECKPOINT:-outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/gnn/model_best.pth}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
  SOURCE_ARGS=()
fi
for input in "$DATASET_DIR" "$GNN_CHECKPOINT" "$DISTANCE_CHECKPOINT"; do
  [[ -e "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing input=$input" >&2; exit 2; }
done
echo "[COMRECGC_STAGE_CONFIG] stage=project_generation dataset=$DATASET mode=$MODE parent_limit=$PARENT_LIMIT transition_state_policy=$TRANSITION_STATE_POLICY output_dir=$OUTPUT_DIR"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/run_generation.py \
  --route project --dataset "$DATASET" --mode "$MODE" \
  --project-root "$PROJECT_ROOT" --upstream-root external/COMRECGC \
  --dataset-dir "$DATASET_DIR" "${SOURCE_ARGS[@]}" \
  --gnn-checkpoint "$GNN_CHECKPOINT" --distance-checkpoint "$DISTANCE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" --parent-limit "$PARENT_LIMIT" --device cuda:0 \
  --trace-output-dir "$TRACE_DIR" --graph-state-dir "$GRAPH_STATE_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$TRACE_DIR/candidate_action_lineage.json"
if [[ "$MODE" == "full" ]]; then
  test -s "$OUTPUT_DIR/graph_state_audit.json"
  test -s "$GRAPH_STATE_DIR/authoritative_graph_store.sqlite3"
fi
echo "[COMRECGC_PROJECT_GENERATION_SUCCESS]"
