#!/bin/bash
# 200-step decoded-chem PPO diagnose run in core_output_mode.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ppo_v3_core_diag

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

python scripts/train_rl.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --output-dir outputs/ppo_v3_core_diagnose \
  --candidate-pool-path outputs/ppo_v3_core_diagnose/candidate_pool.jsonl \
  --sft-lora-path ckpt/sft_v3_core_lora \
  --teacher-path outputs/hpc/oracle/aids_rf_model.pkl \
  --ppo-loop decoded_chem \
  --require-chemistry-reward-path \
  --decoded-chem-smoke-test \
  --diagnose-reward-flow \
  --require-teacher-sem \
  --core-output-mode \
  --dummy-output-penalty -0.25 \
  --enable-parent-projection \
  --projection-min-score 0.35 \
  --projection-max-candidates 128 \
  --projection-min-atoms 3 \
  --projection-max-atom-ratio 0.70 \
  --projection-penalty 0.5 \
  --no-projection-enable-khop3 \
  --projection-mcs-timeout 1 \
  --enable-minimal-syntax-repair \
  --enable-component-salvage \
  --min-fragment-atoms 3 \
  --tiny-fragment-hard-fail-penalty -6.0 \
  --max-steps 200 \
  --logging-steps 5 \
  --save-steps 100
