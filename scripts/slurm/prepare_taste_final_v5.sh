#!/bin/bash
# V5 user authorization: CPU-only metadata adoption, no model/heuristic generation.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${TASTE_V5_CODE:?immutable worktree}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "V5 contract adoption python=$(command -v python), CUDA hidden"
python -V
python -I -B scripts/prepare_taste_final_v5.py --config configs/hpc.yaml \
 --ours-root "${TASTE_V5_OURS_SOURCE:?accepted Ours}" --yaml "${TASTE_V5_YAML:?authorized YAML}" \
 --out-dir "${TASTE_V5_OUTPUT:?fresh output}"
