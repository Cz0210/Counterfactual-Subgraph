#!/bin/bash
# import_cm_crem.py also accepts --resume-staging for verified extraction recovery.
# User-authorized CPU/record-only exception to the default A800 template.
#SBATCH --partition=intel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "${CM_EXECUTION_ROOT:?immutable CM execution root required}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
echo "CM record-only import: $(command -v python)"
python -V
python -I -B scripts/import_cm_crem.py --config configs/hpc.yaml "$@"
