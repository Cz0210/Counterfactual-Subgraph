#!/bin/bash
#SBATCH --job-name=bace_ours_matrix_v2
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

CANDIDATE_POOL=${CANDIDATE_POOL:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours/candidate_pool.jsonl}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_wnode_prefix_v2/calibration_matrix}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/bace_ours_wnode_prefix_v2.sqlite3}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
EXPECTED_POOL_ROWS=${EXPECTED_POOL_ROWS:-1440}
EXPECTED_SOURCE_PARENT_COUNT=${EXPECTED_SOURCE_PARENT_COUNT:-360}
EXPECTED_SOURCE_ELIGIBLE_ROWS=${EXPECTED_SOURCE_ELIGIBLE_ROWS:-657}
EXPECTED_UNIQUE_CANDIDATES=${EXPECTED_UNIQUE_CANDIDATES:-154}

for path in "$CANDIDATE_POOL" "$CALIBRATION_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT"; do
  test -s "$path" || { echo "[BACE_V2_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
[[ "$CALIBRATION_CSV" != *test* ]] || { echo "[BACE_V2_LEAKAGE_ERROR] test path" >&2; exit 2; }

args=(
  python scripts/precompute_wnode_action_matrix.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset BACE
  --split calibration
  --parent-csv "$CALIBRATION_CSV"
  --candidate-pool "$CANDIDATE_POOL"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --wnode-cache-db "$WNODE_CACHE_DB"
  --output-dir "$OUTPUT_DIR"
  --expected-parent-count 60
  --expected-pool-rows "$EXPECTED_POOL_ROWS"
  --expected-source-parent-count "$EXPECTED_SOURCE_PARENT_COUNT"
  --expected-source-eligible-rows "$EXPECTED_SOURCE_ELIGIBLE_ROWS"
  --expected-unique-candidates "$EXPECTED_UNIQUE_CANDIDATES"
  --cf-mode strict_flip
  --device cuda
  --resume
)

echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_OURS_MATRIX_V2_VALIDATE_OK]"
  exit 0
fi
"${args[@]}"
test -s "$OUTPUT_DIR/pair_matrix.jsonl"
python scripts/audit_wnode_action_matrix.py \
  --config configs/hpc.yaml \
  --run-dir "$OUTPUT_DIR" \
  --expected-parent-count 60 \
  --require-strict-flip-pair
echo "[BACE_OURS_MATRIX_V2_SUCCESS]"
