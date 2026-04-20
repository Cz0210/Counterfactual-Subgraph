#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

cd "${PROJECT_DIR}"

export PYTHONPATH=$PWD
echo "repo pwd: $(pwd)"
echo "PYTHONPATH(after export): ${PYTHONPATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  exit 1
fi

echo "===== RUNNING PPO TRAINING ====="
python scripts/train_ppo.py \
  --config configs/hpc.yaml \
  --teacher-path "${TEACHER_PATH}" \
  --require-teacher-sem \
  --teacher-sem-scale 1.0 \
  --teacher-sem-missing-penalty -5.0 \
  --teacher-cf-flip-bonus 1.0 \
  "$@"
echo "===== DONE ====="
