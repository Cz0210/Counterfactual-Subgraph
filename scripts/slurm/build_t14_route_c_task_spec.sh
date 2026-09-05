#!/usr/bin/env bash
#SBATCH --job-name=t14-route-c-spec
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "fresh_retry_memory_gate=384GiB_start_3_samples,96GiB_runtime_3_samples"
echo "fresh_retry_checkpoints=50,100,250,500;2500..20000;fallback=22500,25000"
echo "engineering_retry=second_and_final,fresh_step_zero,no_failed_checkpoint_reuse"
python scripts/autodl/build_t14_route_c_task_spec.py --config configs/hpc.yaml "$@"
