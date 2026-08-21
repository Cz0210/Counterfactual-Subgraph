#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=autodl-three-lines-status
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# This paired wrapper exists for repository CLI parity only.  It is deliberately
# read-only and never starts or resumes AutoDL work.  AutoDL PIDs are not Slurm
# job IDs; operational control belongs to scripts/autodl/three_lines.sh.
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
PY

python scripts/autodl/run_three_lines.py status \
  --spec ops/specs/autodl_three_lines_20260821.yaml \
  --config configs/hpc.yaml
