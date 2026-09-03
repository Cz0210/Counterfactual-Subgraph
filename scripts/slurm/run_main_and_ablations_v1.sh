#!/usr/bin/env bash
#SBATCH --job-name=main-ablation-sidecar
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
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
# The repaired AutoDL sidecar accepts launches only through five immutable
# component task specs and confirms the real science owner after dispatch.
# A Slurm allocation cannot provide those AutoDL /proc, GPU-lock, or authority
# bindings, so this paired workflow entrypoint remains an intentional refusal.
echo "AutoDL task-spec recovery scheduler only; Slurm launch is disabled" >&2
exit 64
# MUT_CONTINUATION_TASK_SPEC=/runtime/control/specs/mut.json T14_RESUME_TASK_SPEC=/runtime/control/specs/t14.json python scripts/autodl/run_main_and_ablations_v1.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --state-root /runtime/control/main-and-ablations-v1 --matrix-authority /runtime/control/fast16_matrix_authority/state.json
