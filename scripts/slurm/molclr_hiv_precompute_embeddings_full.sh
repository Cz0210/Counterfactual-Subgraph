#!/bin/bash
#SBATCH -J molclr_pre_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

if [ -z "${MOLCLR_ROOT:-}" ] || [ -z "${MOLCLR_CKPT:-}" ]; then
  echo "[ERROR] MOLCLR_ROOT and MOLCLR_CKPT are required."
  exit 2
fi

python scripts/precompute_molclr_embeddings_for_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}" \
  --ours-selected-path "${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH:-outputs/hpc/selectors/gt_fullgraph_tanimoto_baseline_label1/beta_20p0_gamma_5p0}" \
  --molclr-root "${MOLCLR_ROOT}" \
  --molclr-checkpoint "${MOLCLR_CKPT}" \
  --output-dir "${OUTPUT_DIR:-outputs/hpc/molclr_ccrcov_embeddings}" \
  --label "${LABEL:-1}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --device "${DEVICE:-cuda}" \
  --invalid-policy skip

echo "===== MOLCLR CCRCov PRECOMPUTE FULL DONE ====="
