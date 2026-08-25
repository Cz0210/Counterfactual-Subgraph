#!/usr/bin/env bash
# Static CLI-parity wrapper only. The recovery is authorized for AutoDL CPU,
# never for HPC/Slurm execution.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
echo "[AIDS_EXACT_RECOVERY_AUTODL_ONLY] refusing HPC execution" >&2
echo "Use scripts/autodl/launch_aids_comrecgc_exact_recovery_v1.sh on AutoDL." >&2
exit 78

# CLI parity reference (intentionally unreachable):
# python scripts/autodl/run_aids_comrecgc_exact_recovery_stage.py \
#   --config configs/hpc.yaml subset \
#   --controller-manifest /absolute/controller_manifest.json \
#   --adoption-gate /absolute/01_failed_selection_adoption.json \
#   --output-dir /absolute/science/subset_preflight
# This protocol audit is neither inference nor evaluation and exposes no
# heuristic-fallback option.
