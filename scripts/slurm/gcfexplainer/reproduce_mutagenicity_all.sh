#!/bin/bash
#SBATCH --job-name=mut_gcf_all
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

PROFILE="${PROFILE:-smoke}"
RESUME="${RESUME:-true}"
if [[ "$PROFILE" == "smoke" ]]; then
  VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"
  VRRW_M="${VRRW_M:-500}"
elif [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" == "true" ]]; then
  VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"
  VRRW_M="${VRRW_M:-50000}"
elif [[ "$PROFILE" == "full" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] full requires ALLOW_FULL=true." >&2
  exit 2
else
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROFILE must be smoke or full." >&2
  exit 2
fi
VRRW_ALPHA="${VRRW_ALPHA:-1.0}"
VRRW_THETA="${VRRW_THETA:-0.05}"
VRRW_SEED="${VRRW_SEED:-13}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
EXPORT_DIR="${EXPORT_DIR:-}"
if [[ -z "$EXPORT_DIR" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] EXPORT_DIR must be provided explicitly." >&2
  exit 2
fi
if [[ -e "$RUN_ROOT/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized run cannot be overwritten." >&2
  exit 2
fi

export PROJECT_ROOT PROFILE RUN_ROOT
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "RUN_ROOT=$RUN_ROOT"
echo "EXPORT_DIR=$EXPORT_DIR"
echo "VRRW_PARENT_LIMIT=$VRRW_PARENT_LIMIT"
echo "VRRW_M=$VRRW_M"
echo "VRRW_ALPHA=$VRRW_ALPHA"
echo "VRRW_THETA=$VRRW_THETA"
echo "VRRW_SEED=$VRRW_SEED"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

env PROFILE="$PROFILE" RUN_ROOT="$RUN_ROOT" RESUME="$RESUME" \
  bash scripts/slurm/gcfexplainer/prepare_mutagenicity_dataset.sh
env PROFILE="$PROFILE" RUN_ROOT="$RUN_ROOT" RESUME="$RESUME" \
  bash scripts/slurm/gcfexplainer/train_mutagenicity_gnn.sh
env \
  PROFILE="$PROFILE" \
  RUN_ROOT="$RUN_ROOT" \
  RESUME="$RESUME" \
  VRRW_PARENT_LIMIT="$VRRW_PARENT_LIMIT" \
  VRRW_M="$VRRW_M" \
  VRRW_ALPHA="$VRRW_ALPHA" \
  VRRW_THETA="$VRRW_THETA" \
  VRRW_SEED="$VRRW_SEED" \
  bash scripts/slurm/gcfexplainer/reproduce_mutagenicity_vrrw.sh
env \
  PROFILE="$PROFILE" \
  RUN_ROOT="$RUN_ROOT" \
  RESUME="$RESUME" \
  EXPORT_DIR="$EXPORT_DIR" \
  PARENT_LIMIT="$VRRW_PARENT_LIMIT" \
  bash scripts/slurm/gcfexplainer/reproduce_mutagenicity_summary.sh

test -s "$EXPORT_DIR/audit.json"
if [[ "$PROFILE" == "smoke" ]]; then
  test -s "$EXPORT_DIR/_SMOKE_AUDIT_COMPLETE.json"
  echo "[MUTAGENICITY_GCFEXPLAINER_SMOKE_OK]"
else
  test -s "$EXPORT_DIR/_RUN_COMPLETE.json"
  echo "[MUTAGENICITY_GCFEXPLAINER_FULL_OK]"
fi
