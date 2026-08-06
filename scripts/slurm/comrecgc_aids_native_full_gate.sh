#!/bin/bash
#SBATCH --job-name=comrecgc_aids_gate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
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
INPUT_DIR="${INPUT_DIR:-outputs/hpc/baselines/comrecgc/native_full/aids/native_full_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_DIR/gate}"
EXPECTED_PROJECT_COMMIT="${EXPECTED_PROJECT_COMMIT:-}"
[[ -n "$EXPECTED_PROJECT_COMMIT" ]] || { echo "[COMRECGC_CONFIG_ERROR] expected project commit required" >&2; exit 2; }
echo "[COMRECGC_RECOVERY_CONFIG] stage=aids_native_full_gate input=$INPUT_DIR"
python scripts/baselines/comrecgc/gate_recovery.py \
  --stage aids-native-full \
  --expected-project-commit "$EXPECTED_PROJECT_COMMIT" \
  --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_AIDS_NATIVE_FULL_GATE_PASS]"
