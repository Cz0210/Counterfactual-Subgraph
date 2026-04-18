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

cd /share/home/u20526/czx/counterfactual-subgraph

export PYTHONPATH=$PWD
echo "repo pwd: $(pwd)"
echo "PYTHONPATH(after export): ${PYTHONPATH}"
echo "TEACHER_PATH(env, optional): ${TEACHER_PATH:-<use train_ppo.py default>}"

echo "===== RUNNING PPO TRAINING ====="
python scripts/train_ppo.py --config configs/hpc.yaml "$@"
echo "===== DONE ====="
