#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.err


#SBATCH --job-name=chem_sft

source ~/.bashrc
# 注意：如果你创建了新的环境请修改这里，这里暂用你旧脚本的环境名或我们之前定的 smiles_hpc
conda activate smiles_pip118
echo "===== ENV CHECK ====="
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

# 确保在项目根目录下执行
cd /share/home/u20526/czx/counterfactual-subgraph

export PYTHONPATH=$PWD

python scripts/visualize_sft_summary.py \
  --base-validity 70.0 \
  --base-capping 0.0 \
  --final-accuracy 87.8 \
  --training-epochs 1.78 \
  --output-dir outputs/hpc/figures/sft_summary
