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
if [[ "$PROFILE" == "full" && "${ALLOW_FULL:-false}" != "true" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] full requires ALLOW_FULL=true." >&2
  exit 2
fi
if [[ "$PROFILE" != "smoke" && "$PROFILE" != "full" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROFILE must be smoke or full." >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
if [[ -e "$RUN_ROOT/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized run cannot be overwritten." >&2
  exit 2
fi

export PROJECT_ROOT PROFILE RUN_ROOT
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PROFILE=$PROFILE"
echo "RUN_ROOT=$RUN_ROOT"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

bash scripts/slurm/gcfexplainer/prepare_mutagenicity_dataset.sh
bash scripts/slurm/gcfexplainer/train_mutagenicity_gnn.sh
bash scripts/slurm/gcfexplainer/reproduce_mutagenicity_vrrw.sh
bash scripts/slurm/gcfexplainer/reproduce_mutagenicity_summary.sh

test -s "$RUN_ROOT/export/audit.json"
test -s "$RUN_ROOT/export/_RUN_COMPLETE.json"
if [[ "$PROFILE" == "smoke" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_SMOKE_OK]"
else
  echo "[MUTAGENICITY_GCFEXPLAINER_FULL_OK]"
fi
