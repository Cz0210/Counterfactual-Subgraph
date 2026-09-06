#!/usr/bin/env bash
#SBATCH --job-name=t14-formal-binding
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicit task exception: bounded configuration preparation is CPU-only.
# Do not submit science or take a GPU while the AutoDL canary is running.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
echo "task=config-only; gpu_requested=false; no_model_loading=true"
: "${T14_ROUTE_C_TASK_SPEC:?required}"
: "${T14_ROUTE_C_CONTINUATION_SPEC:?required}"
: "${T14_FORMAL_AUTHORIZATION:?required}"
: "${T14_FORMAL_BINDING_ROOT:?required}"
python scripts/autodl/prepare_t14_formal_binding.py --config configs/hpc.yaml \
  --task-spec "$T14_ROUTE_C_TASK_SPEC" --continuation-spec "$T14_ROUTE_C_CONTINUATION_SPEC" \
  --authorization "$T14_FORMAL_AUTHORIZATION" --output-root "$T14_FORMAL_BINDING_ROOT"
