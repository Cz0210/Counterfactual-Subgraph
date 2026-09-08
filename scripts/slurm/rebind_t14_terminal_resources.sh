#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# AutoDL-only control tool. This paired compatibility wrapper must NOT be
# submitted to HPC: it does no HPC science and cannot control AutoDL paths.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
set -u
echo "Python: $(command -v python)"
python --version
echo "AutoDL one-shot control only; refusing execution on the HPC login/compute host."
exit 2
# Exact CLI documentation (run on AutoDL, not sbatch):
# python -I -B scripts/autodl/rebind_t14_terminal_resources.py --config configs/hpc.yaml --action ... --output-root ...
