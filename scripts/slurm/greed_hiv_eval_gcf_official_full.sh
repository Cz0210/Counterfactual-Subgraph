#!/bin/bash
#SBATCH -J greed_gcf_off
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

DATASET_CSV=${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
GCF_OFFICIAL_CANDIDATES_PATH=${GCF_OFFICIAL_CANDIDATES_PATH:-outputs/hpc/gcfexplainer_official/graph_to_smiles/gcf_graph_smiles_candidates.csv}
GREED_CHECKPOINT=${GREED_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_greed_gcf_official_full}
CF_MODE=${CF_MODE:-strict_flip}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

echo "===== GREED CCRCov WITH OFFICIAL GCFEXPLAINER CANDIDATES ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
echo "GCF_OFFICIAL_CANDIDATES_PATH=${GCF_OFFICIAL_CANDIDATES_PATH}"
echo "CF_MODE=${CF_MODE}"
echo "MIN_CF_DROP=${MIN_CF_DROP}"

if [ ! -f "${GCF_OFFICIAL_CANDIDATES_PATH}" ]; then
  echo "[ERROR] GCF official SMILES candidates not found: ${GCF_OFFICIAL_CANDIDATES_PATH}"
  echo "Run scripts/slurm/gcf_official_graph_to_smiles_rf_eval.sh first."
  exit 2
fi

python scripts/evaluate_ccrcov_with_greed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GCF_OFFICIAL_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --greed-checkpoint "${GREED_CHECKPOINT}" \
  --label "${LABEL:-1}" \
  --thresholds "${THRESHOLDS:-0.05,0.10,0.20}" \
  --cf-mode "${CF_MODE}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --output-root "${OUTPUT_ROOT}" \
  --partial-every "${PARTIAL_EVERY:-5000}"

echo "===== GREED GCF OFFICIAL CCRCov DONE ====="
cat "${OUTPUT_ROOT}/combined/combined_threshold_summary.csv"

