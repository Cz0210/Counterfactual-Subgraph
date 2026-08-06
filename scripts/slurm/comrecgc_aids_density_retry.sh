#!/bin/bash
#SBATCH --job-name=comrecgc_aids_density
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-outputs/hpc/baselines/comrecgc/native_smoke/aids/comrecgc_native_common_64p_20260806_v6/counterfactuals.pt}"
EXPECTED_SHA256="${EXPECTED_SHA256:-096ddd0f4ac31126a0665a11effb7362c2137229ff6b53b50e16c081ef6c274a}"
EXISTING_AUDIT="${EXISTING_AUDIT:-outputs/hpc/baselines/comrecgc/aids_native_dbscan_audit_v1/audit.json}"
PREREGISTRATION="${PREREGISTRATION:-outputs/hpc/baselines/comrecgc/preregistration/aids_native_parent_density_retry_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/aids_native_parent_density_retry_v1}"
[[ -s "$EXISTING_AUDIT" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing audit=$EXISTING_AUDIT" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/manifest.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output exists=$OUTPUT_DIR" >&2; exit 2; }
if [[ ! -e "$PREREGISTRATION" ]]; then
  python scripts/baselines/comrecgc/preregister_recovery.py aids-density \
    --existing-audit-path "$EXISTING_AUDIT" \
    --output-path "$PREREGISTRATION"
fi
test -s "$PREREGISTRATION"
echo "[COMRECGC_RECOVERY_CONFIG] stage=aids_density_retry candidate_regeneration=false parent_universe=all_native_reject"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD) prereg_sha=$(sha256sum "$PREREGISTRATION" | awk '{print $1}')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/audit_aids_native_dbscan.py \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --counterfactuals-path "$SOURCE_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-sha256 "$EXPECTED_SHA256" \
  --expected-candidates 31 \
  --full-reject-parent-universe \
  --preregistration-path "$PREREGISTRATION" \
  --device cuda:0 \
  --batch-size 128
test -s "$OUTPUT_DIR/audit.json"
echo "[COMRECGC_AIDS_DENSITY_RETRY_SUCCESS]"
