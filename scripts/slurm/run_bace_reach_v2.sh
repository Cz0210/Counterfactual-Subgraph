#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Site-specific AutoDL GPU runs use the existing owner action, not this HPC wrapper.
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
# Explicit CPU stages may run here. GPU tasks require a bound existing owner FD;
# a scheduler allocation alone must not be misrepresented as that owner receipt.
# Post-search actions: train-gate; freeze-final (requires old-test pair and
# raw-distance-source descriptors); final-test requires a post-freeze raw-test
# index descriptor, the same output root and real CPU resource-config.
# cpu-closeout is the narrow persistent CPU dependency chain for these stages;
# its raw-test-index-descriptor is published only after the new final freeze.
# export-calibration-raw is a no-inference/no-OT CPU migration of sealed current
# parent records plus the accepted old index, for downstream backbone reuse.
python scripts/run_bace_reach_v2.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
