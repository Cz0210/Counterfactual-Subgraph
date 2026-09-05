#!/usr/bin/env bash
# Thin login-side submission only; all science runs inside intel Slurm jobs.
set -euo pipefail
cd "${GNN_EXECUTION_WORKTREE:?set immutable worktree}"
export PYTHONPATH=$PWD
exec "${GNN_PYTHON:-/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python}" scripts/hpc/gnn/submit_bace_gnn_seed7.py --config configs/hpc.yaml "$@"
