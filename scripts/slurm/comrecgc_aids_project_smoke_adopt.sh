#!/bin/bash
#SBATCH --job-name=comrecgc_aids_project_smoke
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

SOURCE_ROOT="${SOURCE_ROOT:-outputs/hpc/baselines/comrecgc/aids/smoke_comrecgc_smoke_budget_retry_20260806_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
MOLCLR_CHECKPOINT="${MOLCLR_CHECKPOINT:-pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}"
[[ -n "$OUTPUT_DIR" ]] || { echo "[COMRECGC_CONFIG_ERROR] explicit OUTPUT_DIR required" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "[COMRECGC_CONFIG_ERROR] output exists=$OUTPUT_DIR" >&2; exit 2; }
echo "[COMRECGC_STAGE_CONFIG] stage=aids_project_smoke_adopt source=$SOURCE_ROOT output=$OUTPUT_DIR algorithm_rerun=false"
python scripts/baselines/comrecgc/adopt_aids_project_smoke.py \
  --project-root "$PROJECT_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_AIDS_PROJECT_SMOKE_ADOPT_PASS]"
