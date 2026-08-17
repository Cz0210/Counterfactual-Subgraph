#!/bin/bash
#SBATCH --job-name=comrecgc_bace_retry3_storage_v6
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
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export PYTHONHASHSEED=0; mkdir -p logs
BASE_ROOT=${BASE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/baselines/comrecgc/bace/project_full_retry3_storage_v6}
PERSISTENT_SCRATCH_ROOT=${PERSISTENT_SCRATCH_ROOT:-/share/project/p20526/u20526/counterfactual-subgraph}
SCRATCH_STATE_ROOT=${SCRATCH_STATE_ROOT:-$PERSISTENT_SCRATCH_ROOT/comrecgc_bace_retry3_v6}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_STATE_ROOT/generation_fresh}
TRACE_DIR=${TRACE_DIR:-$OUTPUT_DIR/trace}; GRAPH_STATE_DIR=${GRAPH_STATE_DIR:-$OUTPUT_DIR/graph_state}
DATASET_DIR=${DATASET_DIR:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset}
GNN_CHECKPOINT=${GNN_CHECKPOINT:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/gnn/model_best.pth}
DISTANCE_CHECKPOINT=${DISTANCE_CHECKPOINT:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt}
COMRECGC_EXPECTED_COMMIT=${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}
COMRECGC_ROOT=${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$DATASET_DIR/dataset_summary.json" "$GNN_CHECKPOINT" "$DISTANCE_CHECKPOINT"; do test -s "$path" || { echo "missing input: $path" >&2; exit 2; }; done
python scripts/verify_comrecgc_checkout.py --config configs/hpc.yaml --root "$COMRECGC_ROOT" --expected-commit "$COMRECGC_EXPECTED_COMMIT" --validate-imports
args=(python scripts/baselines/comrecgc/run_generation.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --route project --dataset bace --mode full --project-root "$PROJECT_ROOT" --upstream-root "$COMRECGC_ROOT"
  --dataset-dir "$DATASET_DIR" --gnn-checkpoint "$GNN_CHECKPOINT" --distance-checkpoint "$DISTANCE_CHECKPOINT"
  --output-dir "$OUTPUT_DIR" --parent-limit 360 --device cuda:0 --batch-size 128
  --trace-output-dir "$TRACE_DIR" --graph-state-dir "$GRAPH_STATE_DIR"
  --storage-guard-root "$OUTPUT_DIR" --storage-check-every-steps 500
  --storage-min-free-gib 50 --storage-min-free-ratio 0.02 --storage-min-free-inodes 100000)
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
python -c 'import torch; print("cuda_available="+str(torch.cuda.is_available()), "device_count="+str(torch.cuda.device_count())); assert torch.cuda.device_count()==1'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_STORAGE_V6_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "scratch output collision: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$BASE_ROOT" "$(dirname "$OUTPUT_DIR")"
if [[ -L "$BASE_ROOT/generation" ]]; then
  test "$(readlink -f "$BASE_ROOT/generation")" = "$(readlink -f "$OUTPUT_DIR")" || { echo 'generation symlink mismatch' >&2; exit 2; }
elif [[ -e "$BASE_ROOT/generation" ]]; then echo "generation path collision" >&2; exit 2
else ln -s "$OUTPUT_DIR" "$BASE_ROOT/generation"; fi
"${args[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$GRAPH_STATE_DIR/authoritative_graph_store.sqlite3"
echo '[COMRECGC_BACE_STORAGE_V6_SUCCESS]'
