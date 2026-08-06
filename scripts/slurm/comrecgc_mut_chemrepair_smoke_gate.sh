#!/bin/bash
#SBATCH --job-name=comrecgc_mut_gate
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
INPUT_DIR="${INPUT_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity_chemistry_audit_v1}"
EVAL_DIR="${EVAL_DIR:-$INPUT_DIR/unified_eval_smoke_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_DIR/gate}"
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_chemrepair_smoke_gate input=$INPUT_DIR eval=$EVAL_DIR"
python scripts/baselines/comrecgc/gate_recovery.py \
  --stage mut-chemistry-smoke --input-dir "$INPUT_DIR" --eval-dir "$EVAL_DIR" --output-dir "$OUTPUT_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_MUT_CHEMREPAIR_SMOKE_GATE_PASS]"
