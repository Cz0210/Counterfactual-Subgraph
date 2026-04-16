#!/bin/bash
# HPC smoke test template for diagnosing which ChemLLM cache module is loaded.
# If your cluster uses different queue parameters, adjust the SBATCH resource
# lines below before submission.
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=debug_check_chemllm_runtime_path

source ~/.bashrc
conda activate smiles_pip118

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV}"
which python
python -V
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "PYTHONPATH(before cd)=${PYTHONPATH}"
echo "====================="

# TODO(HPC): confirm this path matches your actual repository checkout.
cd /share/home/u20526/czx/counterfactual-subgraph

export PYTHONPATH=$PWD

echo "===== REPO CHECK ====="
echo "pwd after cd: $(pwd)"
git rev-parse --show-toplevel
git rev-parse HEAD
echo "PYTHONPATH(after cd)=${PYTHONPATH}"
echo "======================"

echo "===== RUNTIME PYTHON CHECK ====="
python - <<'PY'
import os
import sys

print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
PY
echo "==============================="

echo "===== RUNNING DEBUG PPO SMOKE TEST ====="
# Keep this smoke test intentionally small so the first failure still lands in
# the initialization / first-generation area instead of burning a long queue slot.
python scripts/train_rl.py \
  --config configs/hpc.yaml \
  --max-steps 2 \
  --logging-steps 1 \
  --max-prompt-examples 8 \
  --output-dir outputs/hpc/rl_checkpoints/debug_runtime_path \
  "$@"
echo "===== DONE ====="
