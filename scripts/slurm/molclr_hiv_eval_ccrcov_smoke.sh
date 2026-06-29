#!/bin/bash
#SBATCH -J molclr_eval_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

if [ -z "${TEACHER_PATH:-}" ]; then
  echo "[ERROR] TEACHER_PATH is required."
  exit 2
fi

echo "===== MOLCLR CCRCov SMOKE ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version

python scripts/evaluate_ccrcov_with_molclr.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}" \
  --ours-selected-path "${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH:-outputs/hpc/selectors/gt_fullgraph_tanimoto_baseline_label1/beta_20p0_gamma_5p0}" \
  --teacher-path "${TEACHER_PATH}" \
  --embedding-dir "${EMBEDDING_DIR:-outputs/hpc/molclr_ccrcov_embeddings_smoke}" \
  --label "${LABEL:-1}" \
  --embedding-thresholds "${EMBEDDING_THRESHOLDS:-0.02,0.05,0.10,0.15,0.20,0.25,0.30}" \
  --output-root "${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_molclr_hiv_smoke}" \
  --max-parents "${MAX_PARENTS:-25}" \
  --max-candidates "${MAX_CANDIDATES:-20}" \
  --partial-every 500

echo "===== MOLCLR CCRCov SMOKE DONE ====="
