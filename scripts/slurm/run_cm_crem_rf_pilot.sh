#!/bin/bash
# User-authorized RF/CM CPU adapter, no A800/GPU request.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "${CM_RF_CODE:?immutable root required}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "RF pilot $SLURM_JOB_ID on $(hostname) Python=$(command -v python)"
python -V
# Uses only the actual frozen RF: no heuristic-fallback CLI exists.
python -I -B scripts/run_cm_crem_rf_pilot.py --config configs/hpc.yaml --spec "${CM_RF_SPEC:?sealed spec required}" --action run
