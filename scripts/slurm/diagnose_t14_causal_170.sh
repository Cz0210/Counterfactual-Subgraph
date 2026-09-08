#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Paired CLI wrapper only: actual deployment is AutoDL CPU inspection then
# GPU2 via its existing UUID lock. This template is not GPU2 authorization.
# follower-preflight is a CPU-only full raw-array save/reopen test; new replay
# requires an independent ADDITIONAL_FOLLOWER_CAPTURE_170 campaign receipt.
# compare-followers audits saved335 vectors/actions and never grants promotion.
# compare-replay is CPU-only postprocessing (optional bounded 60s terminal
# checks); production uses the existing AutoDL170 campaign, not this GPU job.
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
# No heuristic fallback is available in this narrowly frozen replay entry.
python -I -B scripts/autodl/diagnose_t14_causal_170.py --config configs/hpc.yaml "$@"
