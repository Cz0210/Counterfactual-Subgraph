#!/bin/bash
#SBATCH --job-name=comrecgc_mut_adopt
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

SOURCE_FAILED_GENERATION_DIR="${SOURCE_FAILED_GENERATION_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity/recovery_smoke_comrecgc_recovery_20260806_mut_retry1/generation}"
REFERENCE="${REFERENCE:-outputs/hpc/baselines/comrecgc/mutagenicity/smoke_comrecgc_smoke_budget_retry_20260806_v4/generation/counterfactuals.pt}"
REFERENCE_SHA256="${REFERENCE_SHA256:-060879cbaf69b1e3279301350f587cab809d48991559a80ff5227c46466df8d0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity/recovery_adopted_trace_v1/generation}"
[[ -s "$SOURCE_FAILED_GENERATION_DIR/counterfactuals.pt" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] missing source trace artifact" >&2; exit 2;
}
[[ -s "$SOURCE_FAILED_GENERATION_DIR/trace/_TRACE_COMPLETE.json" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] missing source trace marker" >&2; exit 2;
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] adoption output already exists=$OUTPUT_DIR" >&2; exit 2;
}
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_trace_adopt algorithm_rerun=false candidates=164"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
python -m py_compile scripts/baselines/comrecgc/recover_mutagenicity_trace.py
python scripts/baselines/comrecgc/recover_mutagenicity_trace.py \
  --source-failed-generation-dir "$SOURCE_FAILED_GENERATION_DIR" \
  --reference-counterfactuals-path "$REFERENCE" \
  --expected-reference-sha256 "$REFERENCE_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --expected-candidate-count 164
test -s "$OUTPUT_DIR/recovery_manifest.json"
test -s "$OUTPUT_DIR/trace_parity.json"
test -s "$OUTPUT_DIR/trace/candidate_action_lineage.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_MUT_TRACE_ADOPT_SUCCESS]"
