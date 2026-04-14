#!/bin/bash
# NOTE: Slurm opens --output/--error before the shell body runs.
# Make sure /share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs exists before sbatch.
#SBATCH --job-name=prep_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.err

set -euo pipefail

source /share/home/u20526/anaconda3/bin/activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
python scripts/prepare_sft_data.py
