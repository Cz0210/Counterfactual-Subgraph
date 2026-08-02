#!/bin/bash
#SBATCH --job-name=mut_gcf_gnn
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$PROJECT_ROOT" ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROJECT_ROOT is required." >&2; exit 2; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE="${PROFILE:-smoke}"
if [[ "$PROFILE" == "smoke" ]]; then
  EPOCHS="${EPOCHS:-5}"
  TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
  VAL_LIMIT="${VAL_LIMIT:-128}"
elif [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" == "true" ]]; then
  EPOCHS="${EPOCHS:-1000}"
  TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
  VAL_LIMIT="${VAL_LIMIT:-0}"
else
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] invalid/unauthorized PROFILE=$PROFILE." >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
GNN_DIR="${GNN_DIR:-$RUN_ROOT/gnn}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DROPOUT="${DROPOUT:-0.0}"
SEED="${SEED:-13}"
RESUME="${RESUME:-true}"

[[ "$SEED" -eq 13 ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] seed must be 13." >&2; exit 2; }
test -s "$DATASET_DIR/_PHASE_A_COMPLETE.json"
if [[ -e "$GNN_DIR/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized GNN output cannot be overwritten." >&2
  exit 2
fi
if [[ -s "$GNN_DIR/_RUN_COMPLETE.json" ]]; then
  if [[ "$RESUME" == "true" ]]; then
    echo "[MUTAGENICITY_GCFEXPLAINER_GNN_REUSED] $GNN_DIR"
    exit 0
  fi
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] completed GNN output cannot be overwritten." >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "DATASET_DIR=$DATASET_DIR"
echo "GNN_DIR=$GNN_DIR"
echo "OFFICIAL_ROOT=$OFFICIAL_ROOT"
echo "EPOCHS=$EPOCHS"
echo "TRAIN_LIMIT=$TRAIN_LIMIT"
echo "VAL_LIMIT=$VAL_LIMIT"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

RESUME_ARG="--no-resume"
[[ "$RESUME" == "true" ]] && RESUME_ARG="--resume"
python scripts/baselines/gcfexplainer/train_mutagenicity_gnn.py \
  --config configs/hpc.yaml \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$OFFICIAL_ROOT" \
  --output-dir "$GNN_DIR" \
  --profile "$PROFILE" \
  --epochs "$EPOCHS" \
  --train-limit "$TRAIN_LIMIT" \
  --val-limit "$VAL_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --device cuda:0 \
  "$RESUME_ARG" \
  --forbid-calibration-test

test -s "$GNN_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_GNN_WRAPPER_OK]"
