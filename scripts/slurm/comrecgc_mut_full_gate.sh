#!/bin/bash
#SBATCH --job-name=comrecgc_mut_full_gate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
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
[[ -n "${BASE_ROOT:-}" ]] || { echo "[COMRECGC_CONFIG_ERROR] full gate requires explicit BASE_ROOT" >&2; exit 2; }
INPUT_DIR="${INPUT_DIR:-$BASE_ROOT/unified_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/full_gate}"
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_full_gate input=$INPUT_DIR output=$OUTPUT_DIR"
python scripts/baselines/comrecgc/gate_recovery.py \
  --stage mut-full --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_MUT_FULL_GATE_PASS]"
