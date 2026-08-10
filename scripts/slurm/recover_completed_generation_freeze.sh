#!/bin/bash
#SBATCH --job-name=comrecgc_freeze_recover
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

ACTION="${ACTION:-validate}"
DATASET="${DATASET:?DATASET is required}"
SOURCE_GENERATION_DIR="${SOURCE_GENERATION_DIR:?SOURCE_GENERATION_DIR is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}"
EXPECTED_STEPS="${EXPECTED_STEPS:-50000}"
EXPECTED_PROJECT_COMMIT="${EXPECTED_PROJECT_COMMIT:-}"
SOURCE_CSV="${SOURCE_CSV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 2
[[ "$ACTION" == "validate" || "$ACTION" == "recover" ]] || exit 2

args=(
  --source-generation-dir "$SOURCE_GENERATION_DIR"
  --dataset "$DATASET"
  --dataset-dir "$DATASET_DIR"
  --audit-output "$AUDIT_OUTPUT"
  --expected-steps "$EXPECTED_STEPS"
)
[[ -z "$SOURCE_CSV" ]] || args+=(--source-csv "$SOURCE_CSV")
[[ -z "$EXPECTED_PROJECT_COMMIT" ]] || args+=(--expected-project-commit "$EXPECTED_PROJECT_COMMIT")
if [[ "$ACTION" == "validate" ]]; then
  args+=(--validate-only)
else
  [[ -n "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR is required for recovery" >&2; exit 2; }
  args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[COMRECGC_STAGE_CONFIG] action=$ACTION dataset=$DATASET source=$SOURCE_GENERATION_DIR output=${OUTPUT_DIR:-none}"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
python -V
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
if [[ "$ACTION" == "recover" && "$DATASET" == "aids" ]]; then
  env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python scripts/baselines/comrecgc/recover_completed_generation_freeze.py "${args[@]}"
else
  python scripts/baselines/comrecgc/recover_completed_generation_freeze.py "${args[@]}"
fi

test -s "$AUDIT_OUTPUT"
if [[ "$ACTION" == "recover" ]]; then
  test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
  test -s "$OUTPUT_DIR/frozen_payload_closure_audit.json"
fi
