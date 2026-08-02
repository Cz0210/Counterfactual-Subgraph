#!/bin/bash
#SBATCH --job-name=mut_gcf_data
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROJECT_ROOT is required." >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE="${PROFILE:-smoke}"
if [[ "$PROFILE" != "smoke" && "$PROFILE" != "full" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROFILE must be smoke or full." >&2
  exit 2
fi
if [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" != "true" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] full requires ALLOW_FULL=true." >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/outputs/hpc/datasets/mutagenicity_v1_teacher_consistent}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
RESUME="${RESUME:-true}"

if [[ -e "$DATASET_DIR/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized dataset cannot be overwritten." >&2
  exit 2
fi
if [[ -s "$DATASET_DIR/_PHASE_A_COMPLETE.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_DATASET_REUSED] $DATASET_DIR"
  exit 0
fi
if [[ -d "$DATASET_DIR" && -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] incomplete dataset output is non-empty." >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_DIR=$DATASET_DIR"
echo "RESUME=$RESUME"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

python scripts/baselines/gcfexplainer/prepare_mutagenicity_dataset.py \
  --config configs/hpc.yaml \
  --train-source-csv "$DATA_ROOT/train_source_label1_teacher_correct.csv" \
  --train-target-csv "$DATA_ROOT/train_target_label0_teacher_correct.csv" \
  --val-source-csv "$DATA_ROOT/val_source_label1_teacher_correct.csv" \
  --val-target-csv "$DATA_ROOT/val_target_label0_teacher_correct.csv" \
  --output-dir "$DATASET_DIR" \
  --forbid-calibration-test

test -s "$DATASET_DIR/_PHASE_A_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_DATASET_WRAPPER_OK]"
