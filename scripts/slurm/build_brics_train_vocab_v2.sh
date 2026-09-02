#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=brics-train-v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${BACE_TRAIN_CSV:?set exact full BACE train.csv}"
: "${BACE_TRAIN_CSV_SHA256:?set exact train.csv SHA256}"
: "${BACE_PROPOSAL_COHORT_MANIFEST:?set exact frozen 386-parent manifest}"
: "${BACE_PROPOSAL_COHORT_SHA256:?set exact cohort manifest SHA256}"
: "${BACE_LLM_REFERENCE_CONTRACT:?set exact BACE/Ours LLM reference v2}"
: "${BACE_LLM_REFERENCE_CONTRACT_SHA256:?set exact reference contract file SHA256}"
: "${BRICS_ATTEMPTS_PER_PARENT:?set exact attempts_per_parent from the reference contract}"
: "${BRICS_OUTPUT_ROOT:?set a fresh output root}"

ionice -c 2 -n 7 nice -n 10 python \
  scripts/ablations/llm/build_brics_train_vocab_v2.py \
  --config configs/hpc.yaml \
  --train-csv "$BACE_TRAIN_CSV" \
  --train-csv-sha256 "$BACE_TRAIN_CSV_SHA256" \
  --expected-train-rows "${BACE_EXPECTED_TRAIN_ROWS:-959}" \
  --proposal-cohort-manifest "$BACE_PROPOSAL_COHORT_MANIFEST" \
  --proposal-cohort-sha256 "$BACE_PROPOSAL_COHORT_SHA256" \
  --reference-contract "$BACE_LLM_REFERENCE_CONTRACT" \
  --reference-contract-sha256 "$BACE_LLM_REFERENCE_CONTRACT_SHA256" \
  --expected-proposal-parents "${BACE_EXPECTED_PROPOSAL_PARENTS:-386}" \
  --attempts-per-parent "$BRICS_ATTEMPTS_PER_PARENT" \
  --workers "${BRICS_WORKERS:-1}" \
  --output-root "$BRICS_OUTPUT_ROOT"
