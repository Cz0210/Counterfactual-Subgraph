#!/bin/bash
#SBATCH --job-name=mut_gcf_vrrw
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
export PYTHONHASHSEED="${PYTHONHASHSEED:-13}"

PROFILE="${PROFILE:-smoke}"
if [[ "$PROFILE" == "smoke" ]]; then
  PARENT_LIMIT="${PARENT_LIMIT:-64}"
  MAX_STEPS="${MAX_STEPS:-500}"
elif [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" == "true" ]]; then
  PARENT_LIMIT="${PARENT_LIMIT:-1448}"
  MAX_STEPS="${MAX_STEPS:-50000}"
else
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] invalid/unauthorized PROFILE=$PROFILE." >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
GNN_DIR="${GNN_DIR:-$RUN_ROOT/gnn}"
VRRW_DIR="${VRRW_DIR:-$RUN_ROOT/vrrw}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-$GNN_DIR/model_best.pth}"
NEUROSED_CHECKPOINT="${NEUROSED_CHECKPOINT:-$OFFICIAL_ROOT/data/mutagenicity/neurosed/best_model.pt}"
ALPHA="${ALPHA:-1.0}"
THETA="${THETA:-0.05}"
TELEPORT="${TELEPORT:-0.1}"
CANDIDATE_CAPACITY="${CANDIDATE_CAPACITY:-100000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
SEED="${SEED:-13}"
RESUME="${RESUME:-true}"

[[ "$ALPHA" == "1.0" && "$SEED" -eq 13 ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] alpha=1.0 and seed=13 are fixed." >&2; exit 2; }
if [[ "$PROFILE" == "smoke" ]]; then
  [[ "$PARENT_LIMIT" -eq 64 && ( "$MAX_STEPS" -eq 500 || "$MAX_STEPS" -eq 1000 ) ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] smoke requires 64 parents and M=500 or 1000." >&2; exit 2; }
else
  [[ "$PARENT_LIMIT" -eq 1448 && "$MAX_STEPS" -eq 50000 ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] full requires 1448 parents and M=50000." >&2; exit 2; }
fi
test -s "$GNN_CHECKPOINT"
test -s "$NEUROSED_CHECKPOINT"
if [[ "${GNN_CHECKPOINT,,}" == *aids* || "${GNN_CHECKPOINT,,}" == *hiv* ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] AIDS/HIV checkpoint is forbidden." >&2
  exit 2
fi
if [[ -e "$VRRW_DIR/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized VRRW output cannot be overwritten." >&2
  exit 2
fi
if [[ -s "$VRRW_DIR/_RUN_COMPLETE.json" ]]; then
  if [[ "$RESUME" == "true" ]]; then
    echo "[MUTAGENICITY_GCFEXPLAINER_VRRW_REUSED] $VRRW_DIR"
    exit 0
  fi
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] completed VRRW output cannot be overwritten." >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "PARENT_LIMIT=$PARENT_LIMIT"
echo "MAX_STEPS=$MAX_STEPS"
echo "ALPHA=$ALPHA"
echo "THETA=$THETA"
echo "GNN_CHECKPOINT=$GNN_CHECKPOINT"
echo "NEUROSED_CHECKPOINT=$NEUROSED_CHECKPOINT"
echo "VRRW_DIR=$VRRW_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

RESUME_ARG="--no-resume"
[[ "$RESUME" == "true" ]] && RESUME_ARG="--resume"
python scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py \
  --config configs/hpc.yaml \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$OFFICIAL_ROOT" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --neurosed-checkpoint "$NEUROSED_CHECKPOINT" \
  --output-dir "$VRRW_DIR" \
  --profile "$PROFILE" \
  --parent-limit "$PARENT_LIMIT" \
  --max-steps "$MAX_STEPS" \
  --alpha "$ALPHA" \
  --theta "$THETA" \
  --teleport "$TELEPORT" \
  --candidate-capacity "$CANDIDATE_CAPACITY" \
  --no-sample \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --device1 cuda:0 \
  --device2 cuda:0 \
  "$RESUME_ARG" \
  --forbid-calibration-test

test -s "$VRRW_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_VRRW_WRAPPER_OK]"
