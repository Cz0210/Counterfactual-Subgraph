#!/bin/bash
#SBATCH --job-name=comrecgc_native
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

NATIVE_DATASET="${NATIVE_DATASET:-aids}"
NATIVE_PARENT_LIMIT="${NATIVE_PARENT_LIMIT:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/native_smoke/$NATIVE_DATASET}"
TRUSTED_DATASET_ARGS=()
[[ "$NATIVE_DATASET" == "aids" || "$NATIVE_DATASET" == "mutagenicity" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] invalid native dataset=$NATIVE_DATASET" >&2; exit 2;
}
[[ "$NATIVE_PARENT_LIMIT" == "64" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] native smoke parent_limit=$NATIVE_PARENT_LIMIT expected=64" >&2; exit 2;
}
[[ -d external/COMRECGC/.git ]] || { echo "[COMRECGC_CONFIG_ERROR] pinned upstream missing" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete" >&2; exit 2; }
if [[ "$NATIVE_DATASET" == "aids" ]]; then
  CACHE_TRUST_BEFORE="${CACHE_TRUST_BEFORE:-${OUTPUT_DIR}.cache_trust_before.json}"
  TRUSTED_DATASET_PAYLOAD="${TRUSTED_DATASET_PAYLOAD:-${OUTPUT_DIR}.trusted_dataset.pt}"
  python scripts/baselines/comrecgc/audit_trusted_aids_cache.py \
    --upstream-root external/COMRECGC \
    --output "$CACHE_TRUST_BEFORE"
  CACHE_SHA256_BEFORE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_sha256"])' "$CACHE_TRUST_BEFORE")"
  env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python scripts/baselines/comrecgc/materialize_trusted_aids_cache.py \
    --cache-trust-json "$CACHE_TRUST_BEFORE" \
    --output "$TRUSTED_DATASET_PAYLOAD"
  TRUSTED_DATASET_ARGS=(
    --trusted-dataset-payload "$TRUSTED_DATASET_PAYLOAD"
    --expected-cache-inventory-sha256 "$CACHE_SHA256_BEFORE"
  )
fi
echo "[COMRECGC_STAGE_CONFIG] stage=native_smoke dataset=$NATIVE_DATASET parent_limit=$NATIVE_PARENT_LIMIT output_dir=$OUTPUT_DIR"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -m py_compile scripts/baselines/comrecgc/run_generation.py
python scripts/baselines/comrecgc/run_generation.py \
  --route native \
  --dataset "$NATIVE_DATASET" \
  --mode smoke \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --output-dir "$OUTPUT_DIR" \
  --parent-limit "$NATIVE_PARENT_LIMIT" \
  "${TRUSTED_DATASET_ARGS[@]}" \
  --device cuda:0
test -s "$OUTPUT_DIR/counterfactuals.pt"
test -s "$OUTPUT_DIR/native_common_recourse.json"
test -s "$OUTPUT_DIR/native_representative_counterfactuals.pt"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_NATIVE_SMOKE_SUCCESS]"
