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
SEED="${SEED:-13}"

config_error() {
  local field="$1"
  local actual="$2"
  local expected="$3"
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" >&2
  echo "profile=$PROFILE" >&2
  echo "field=$field" >&2
  echo "actual=$actual" >&2
  echo "expected=$expected" >&2
  exit 2
}

if [[ "$PROFILE" == "smoke" ]]; then
  EPOCHS="${EPOCHS:-5}"
  TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
  VAL_LIMIT="${VAL_LIMIT:-128}"
elif [[ "$PROFILE" == "full" ]]; then
  EPOCHS="${EPOCHS:-1000}"
  TRAIN_LIMIT="${TRAIN_LIMIT:-2885}"
  VAL_LIMIT="${VAL_LIMIT:-355}"
  [[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -eq 1000 ]] || config_error "epochs" "$EPOCHS" "1000"
  [[ "$TRAIN_LIMIT" =~ ^[0-9]+$ && "$TRAIN_LIMIT" -eq 2885 ]] || config_error "train_limit" "$TRAIN_LIMIT" "2885"
  [[ "$VAL_LIMIT" =~ ^[0-9]+$ && "$VAL_LIMIT" -eq 355 ]] || config_error "val_limit" "$VAL_LIMIT" "355"
  [[ "$SEED" =~ ^[0-9]+$ && "$SEED" -eq 13 ]] || config_error "seed" "$SEED" "13"
else
  config_error "profile" "$PROFILE" "smoke_or_full"
fi
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
if [[ "$PROFILE" == "full" ]]; then
  GNN_DIR="${GNN_DIR:-}"
  [[ -n "$GNN_DIR" ]] || config_error "gnn_dir" "" "explicit_nonempty_path"
  if [[ "$GNN_DIR" =~ (^|/)smoke([^/]*)($|/) ]]; then
    config_error "gnn_dir" "$GNN_DIR" "explicit_non_smoke_output_path"
  fi
else
  GNN_DIR="${GNN_DIR:-$RUN_ROOT/gnn}"
fi
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DROPOUT="${DROPOUT:-0.0}"
RESUME="${RESUME:-true}"

[[ "$SEED" =~ ^[0-9]+$ && "$SEED" -eq 13 ]] || config_error "seed" "$SEED" "13"

echo "[MUTAGENICITY_GCFEXPLAINER_GNN_CONFIG]"
echo "profile=$PROFILE"
echo "epochs=$EPOCHS"
echo "train_limit=$TRAIN_LIMIT"
echo "val_limit=$VAL_LIMIT"
echo "seed=$SEED"
echo "calibration_loaded=false"
echo "test_loaded=false"

test -s "$DATASET_DIR/_PHASE_A_COMPLETE.json"
if [[ -e "$GNN_DIR/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized GNN output cannot be overwritten." >&2
  exit 2
fi
if [[ -s "$GNN_DIR/_RUN_COMPLETE.json" ]]; then
  if [[ "$PROFILE" == "full" ]]; then
    [[ -s "$GNN_DIR/run_manifest.json" ]] || config_error "existing_run_manifest" "missing" "full_profile_manifest"
    EXISTING_PROFILE="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("profile", ""))' "$GNN_DIR/run_manifest.json")" \
      || config_error "existing_run_manifest" "unreadable" "full_profile_manifest"
    [[ "$EXISTING_PROFILE" == "full" ]] || config_error "existing_profile" "$EXISTING_PROFILE" "full"
  fi
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
