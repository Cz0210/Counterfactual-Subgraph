#!/bin/bash
#SBATCH -J plot_pareto_traj
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

mkdir -p logs

MANIFEST=${MANIFEST:-outputs/hpc/comparison/hiv_quick/pareto_two_trajectories_manifest.csv}
OUT_DIR=${OUT_DIR:-outputs/hpc/comparison/hiv_quick/pareto_two_trajectories}
K=${K:-20}
TITLE_PREFIX=${TITLE_PREFIX:-HIV label=1}

echo "===== PARETO TWO TRAJECTORIES PLOT ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as exc:
    print("matplotlib import failed:", repr(exc))
PY
echo "MANIFEST=${MANIFEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "K=${K}"
echo "TITLE_PREFIX=${TITLE_PREFIX}"
echo "=================================================="

if [ ! -f "${MANIFEST}" ]; then
  echo "[ERROR] missing manifest: ${MANIFEST}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
python scripts/analysis/plot_pareto_two_trajectories.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --k "${K}" \
  --title-prefix "${TITLE_PREFIX}"

echo "===== PARETO TWO TRAJECTORIES OUTPUTS ====="
ls -lh "${OUT_DIR}"
