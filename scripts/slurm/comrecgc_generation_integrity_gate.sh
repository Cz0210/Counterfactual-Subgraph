#!/bin/bash
#SBATCH --job-name=comrecgc_gen_integrity
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
mkdir -p logs

GENERATION_DIR="${GENERATION_DIR:?GENERATION_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
EXPECTED_STEPS="${EXPECTED_STEPS:-50000}"

echo "[COMRECGC_STAGE_CONFIG] stage=generation_integrity_gate generation_dir=$GENERATION_DIR output_dir=$OUTPUT_DIR expected_steps=$EXPECTED_STEPS"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
python scripts/baselines/comrecgc/audit_generation_integrity.py \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --expected-steps "$EXPECTED_STEPS"
test -s "$OUTPUT_DIR/generation_integrity_gate.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_GENERATION_INTEGRITY_GATE_SUCCESS]"
