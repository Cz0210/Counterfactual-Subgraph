#!/bin/bash
#SBATCH --job-name=comrecgc_bace_generate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
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
export PYTHONHASHSEED=0
mkdir -p logs

BASE_ROOT=${BASE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/baselines/comrecgc/bace/project_full_connected_v1}
DATASET_DIR=${DATASET_DIR:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset}
GNN_CHECKPOINT=${GNN_CHECKPOINT:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/gnn/model_best.pth}
DISTANCE_CHECKPOINT=${DISTANCE_CHECKPOINT:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE_ROOT/generation}
TRACE_DIR=${TRACE_DIR:-$OUTPUT_DIR/trace}
GRAPH_STATE_DIR=${GRAPH_STATE_DIR:-$OUTPUT_DIR/graph_state}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$OUTPUT_DIR/generation_checkpoints}
CHECKPOINT_MIRROR_ROOT=${CHECKPOINT_MIRROR_ROOT:?CHECKPOINT_MIRROR_ROOT must be an independent persistent path}
COMRECGC_EXPECTED_COMMIT=${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}
COMRECGC_ROOT=${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
RESUME=${RESUME:-0}
BACE_PREPROCESS_ENGINE=${BACE_PREPROCESS_ENGINE:-legacy_sequential_rdkit_v1}
BACE_PREPROCESS_WORKERS=${BACE_PREPROCESS_WORKERS:-0}
BACE_PREPROCESS_MAX_INFLIGHT=${BACE_PREPROCESS_MAX_INFLIGHT:-64}
BACE_SOURCE_CACHE_CAPACITY=${BACE_SOURCE_CACHE_CAPACITY:-0}
BACE_CANDIDATE_CACHE_CAPACITY=${BACE_CANDIDATE_CACHE_CAPACITY:-0}

for path in "$DATASET_DIR/dataset_summary.json" "$GNN_CHECKPOINT" "$DISTANCE_CHECKPOINT"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done
python scripts/verify_comrecgc_checkout.py \
  --config configs/hpc.yaml \
  --root "$COMRECGC_ROOT" \
  --expected-commit "$COMRECGC_EXPECTED_COMMIT" \
  --validate-imports
args=(
  python scripts/baselines/comrecgc/run_generation.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --route project
  --dataset bace
  --mode full
  --project-root "$PROJECT_ROOT"
  --upstream-root "$COMRECGC_ROOT"
  --dataset-dir "$DATASET_DIR"
  --gnn-checkpoint "$GNN_CHECKPOINT"
  --distance-checkpoint "$DISTANCE_CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --parent-limit 360
  --device cuda:0
  --batch-size 128
  --bace-preprocess-engine "$BACE_PREPROCESS_ENGINE"
  --bace-preprocess-workers "$BACE_PREPROCESS_WORKERS"
  --bace-preprocess-max-inflight "$BACE_PREPROCESS_MAX_INFLIGHT"
  --bace-source-cache-capacity "$BACE_SOURCE_CACHE_CAPACITY"
  --bace-candidate-cache-capacity "$BACE_CANDIDATE_CACHE_CAPACITY"
  --trace-output-dir "$TRACE_DIR"
  --graph-state-dir "$GRAPH_STATE_DIR"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --checkpoint-mirror-root "$CHECKPOINT_MIRROR_ROOT"
  --checkpoint-interval-steps 500
  --checkpoint-keep-last 2
  --progress-interval-steps 25
)
if [[ "$RESUME" == 1 ]]; then args+=(--resume); fi
[[ "$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT_ROOT")" != "$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT_MIRROR_ROOT")" ]] || { echo 'checkpoint mirror must differ from fast checkpoint root' >&2; exit 2; }
echo "hostname=$(hostname)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "git_commit=$(git rev-parse HEAD)"
echo 'scientific_parameters=seed0,steps50000,heads5,k100000,sample_size10000,teleport0.1,theta0.1'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  echo '[COMRECGC_BACE_GENERATE_VALIDATE_OK]'
  exit 0
fi
if [[ "$RESUME" == 1 ]]; then
  test -s "$CHECKPOINT_ROOT/LATEST" || { echo "missing resume checkpoint: $CHECKPOINT_ROOT/LATEST" >&2; exit 2; }
else
  test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
fi
"${args[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$TRACE_DIR/candidate_action_lineage.json"
find "$GRAPH_STATE_DIR" -maxdepth 1 -type f -name 'authoritative_graph_store*.sqlite3' -size +0c -print -quit | grep -q .
echo '[COMRECGC_BACE_GENERATE_SUCCESS]'
