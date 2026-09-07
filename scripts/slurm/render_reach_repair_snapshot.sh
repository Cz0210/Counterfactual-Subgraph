#!/usr/bin/env bash
#SBATCH --job-name=reach-result-snapshot
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# CPU-only paper rendering: deliberately no GPU allocation or science inference.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "Python: $(command -v python)"
python --version
echo "CPU-only plot rendering; CUDA inference not requested"
python scripts/paper/render_reach_repair_snapshot.py --config configs/hpc.yaml "$@"
