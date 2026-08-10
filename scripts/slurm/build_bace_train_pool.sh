#!/bin/bash
#SBATCH --job-name=bace_globalgce_pool
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
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

TRAIN_CSV=${TRAIN_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
NATIVE_TRAIN_CSV=${NATIVE_TRAIN_CSV:-$ARTIFACT_ROOT/data/processed/BACE/train.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$ARTIFACT_ROOT/baselines/globalgce_official}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/baselines/globalgce/bace/train_pool_connected_v1}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$TRAIN_CSV" "$NATIVE_TRAIN_CSV" "$TEACHER_PATH"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done
test -d "$OFFICIAL_ROOT" || { echo "missing official root: $OFFICIAL_ROOT" >&2; exit 2; }

args=(
  python scripts/baselines/globalgce/build_bace_train_pool.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --train-csv "$TRAIN_CSV"
  --native-train-csv "$NATIVE_TRAIN_CSV"
  --teacher-path "$TEACHER_PATH"
  --official-root "$OFFICIAL_ROOT"
  --output-dir "$OUTPUT_DIR"
  --expected-parent-count 360
  --seed 13
  --epochs 100
  --top-k-native 20
  --learning-rate 0.1
  --dropout 0.5
  --device cuda
  --generation-chunk-size 32
  --generation-num-workers 0
  --memory-log-every-chunks 1
  --resume
)

echo "hostname=$(hostname)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  echo '[BACE_GLOBALGCE_POOL_VALIDATE_OK]'
  exit 0
fi
"${args[@]}"
python scripts/baselines/globalgce/audit_bace_train_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --run-dir "$OUTPUT_DIR" \
  --train-csv "$TRAIN_CSV" \
  --expected-parent-count 360
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$OUTPUT_DIR/train_pool_audit.json"
echo '[BACE_GLOBALGCE_POOL_SUCCESS]'
