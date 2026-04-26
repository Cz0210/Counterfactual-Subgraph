#!/bin/bash
# Rebuild the v3_core SFT dataset with core_no_dummy responses.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=prep_sft_v3_core

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

python scripts/prepare_sft_data.py \
  --config configs/hpc.yaml \
  --target-format core \
  --train-output data/sft_v3_core_train.jsonl \
  --val-output data/sft_v3_core_val.jsonl \
  --audit-output outputs/sft_v3_core_audit.json \
  --target-examples 5000 \
  --train-size 4500 \
  --val-size 500 \
  --seed 7 \
  --min-real-atoms 4 \
  --max-cut-attempts 24
