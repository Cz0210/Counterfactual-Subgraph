#!/bin/bash
#SBATCH --job-name=mut_gcf_summary
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
  EXPECTED_PARENT_LIMIT=64
elif [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" == "true" ]]; then
  EXPECTED_PARENT_LIMIT=1448
else
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] invalid/unauthorized PROFILE=$PROFILE." >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
GNN_DIR="${GNN_DIR:-$RUN_ROOT/gnn}"
VRRW_DIR="${VRRW_DIR:-$RUN_ROOT/vrrw}"
SUMMARY_DIR="${SUMMARY_DIR:-$RUN_ROOT/native_summary}"
EXPORT_DIR="${EXPORT_DIR:-}"
if [[ -z "$EXPORT_DIR" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] EXPORT_DIR must be provided explicitly." >&2
  exit 2
fi
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-$GNN_DIR/model_best.pth}"
NEUROSED_CHECKPOINT="${NEUROSED_CHECKPOINT:-$OFFICIAL_ROOT/data/mutagenicity/neurosed/best_model.pt}"
TEACHER_PATH="${TEACHER_PATH:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
SUMMARY_THETA="${SUMMARY_THETA:-0.1}"
MINIMUM_NATIVE_EXPORT="${MINIMUM_NATIVE_EXPORT:-100}"
TOP_K="${TOP_K:-20}"
RESUME="${RESUME:-true}"
if [[ "$PROFILE" == "smoke" ]]; then
  EXPORT_SUCCESS_MARKER="$EXPORT_DIR/_SMOKE_AUDIT_COMPLETE.json"
else
  EXPORT_SUCCESS_MARKER="$EXPORT_DIR/_RUN_COMPLETE.json"
fi

test -s "$DATASET_DIR/_PHASE_A_COMPLETE.json"
test -s "$GNN_DIR/_RUN_COMPLETE.json"
test -s "$VRRW_DIR/_RUN_COMPLETE.json"
test -s "$GNN_CHECKPOINT"
test -s "$NEUROSED_CHECKPOINT"
test -s "$TEACHER_PATH"
if [[ "${GNN_CHECKPOINT,,}" == *aids* || "${GNN_CHECKPOINT,,}" == *hiv* ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] AIDS/HIV checkpoint is forbidden." >&2
  exit 2
fi
for directory in "$SUMMARY_DIR" "$EXPORT_DIR"; do
  if [[ -e "$directory/_FINALIZED.json" ]]; then
    echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized output cannot be overwritten: $directory" >&2
    exit 2
  fi
done

mkdir -p "$PROJECT_ROOT/logs"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "EXPECTED_PARENT_LIMIT=$EXPECTED_PARENT_LIMIT"
echo "SUMMARY_DIR=$SUMMARY_DIR"
echo "EXPORT_DIR=$EXPORT_DIR"
echo "GNN_CHECKPOINT=$GNN_CHECKPOINT"
echo "NEUROSED_CHECKPOINT=$NEUROSED_CHECKPOINT"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

if [[ ! -s "$SUMMARY_DIR/_RUN_COMPLETE.json" ]]; then
  if [[ -d "$SUMMARY_DIR" && -n "$(find "$SUMMARY_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] incomplete native summary requires a fresh output directory." >&2
    exit 2
  fi
  python scripts/baselines/gcfexplainer/run_mutagenicity_summary.py \
    --config configs/hpc.yaml \
    --dataset-dir "$DATASET_DIR" \
    --official-root "$OFFICIAL_ROOT" \
    --vrrw-dir "$VRRW_DIR" \
    --gnn-checkpoint "$GNN_CHECKPOINT" \
    --neurosed-checkpoint "$NEUROSED_CHECKPOINT" \
    --output-dir "$SUMMARY_DIR" \
    --profile "$PROFILE" \
    --theta "$SUMMARY_THETA" \
    --minimum-native-export "$MINIMUM_NATIVE_EXPORT" \
    --device cuda:0 \
    --forbid-calibration-test
elif [[ "$RESUME" != "true" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] completed summary exists and RESUME is not true." >&2
  exit 2
else
  echo "[MUTAGENICITY_GCFEXPLAINER_SUMMARY_REUSED] $SUMMARY_DIR"
fi

if [[ ! -s "$EXPORT_SUCCESS_MARKER" ]]; then
  if [[ -d "$EXPORT_DIR" && -n "$(find "$EXPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] incomplete candidate export requires a fresh output directory." >&2
    exit 2
  fi
  python scripts/baselines/gcfexplainer/export_mutagenicity_fullgraph_candidates.py \
    --config configs/hpc.yaml \
    --dataset-dir "$DATASET_DIR" \
    --summary-dir "$SUMMARY_DIR" \
    --teacher-path "$TEACHER_PATH" \
    --output-dir "$EXPORT_DIR" \
    --profile "$PROFILE" \
    --parent-limit "$EXPECTED_PARENT_LIMIT" \
    --top-k "$TOP_K" \
    --forbid-calibration-test
elif [[ "$RESUME" != "true" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] completed export exists and RESUME is not true." >&2
  exit 2
else
  echo "[MUTAGENICITY_GCFEXPLAINER_EXPORT_REUSED] $EXPORT_DIR"
fi

python scripts/baselines/gcfexplainer/audit_mutagenicity_run.py \
  --config configs/hpc.yaml \
  --dataset-dir "$DATASET_DIR" \
  --gnn-dir "$GNN_DIR" \
  --vrrw-dir "$VRRW_DIR" \
  --summary-dir "$SUMMARY_DIR" \
  --export-dir "$EXPORT_DIR" \
  --profile "$PROFILE" \
  --require-complete \
  --forbid-calibration-test

test -s "$EXPORT_DIR/audit.json"
test -s "$EXPORT_SUCCESS_MARKER"
echo "[MUTAGENICITY_GCFEXPLAINER_SUMMARY_WRAPPER_OK]"
