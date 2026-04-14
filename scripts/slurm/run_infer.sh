#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

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

echo "===== RUNNING INFERENCE ====="
# 执行闭环推理，强制不回退，并指定 hpc 配置
python scripts/run_infer.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
echo "===== DONE ====="
