#!/bin/bash
# Evaluate core-only SFT fragment outputs.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=eval_sft_v3_core

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "python path: $(which python)"
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

export PYTHONPATH=$PWD

python scripts/eval_sft_fragment_quality.py \
  --config configs/hpc.yaml \
  --eval-file data/sft_v3_core_val.jsonl \
  --lora-root ckpt/sft_v3_core_lora \
  --output-dir outputs/sft_v3_core_eval
