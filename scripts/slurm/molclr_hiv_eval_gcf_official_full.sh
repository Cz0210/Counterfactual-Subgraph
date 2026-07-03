#!/bin/bash
#SBATCH -J molclr_gcf_off
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
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-${PROJECT_ROOT}/pretrained_models/MolCLR}
MOLCLR_CKPT=${MOLCLR_CKPT:-${MOLCLR_ROOT}/ckpt/pretrained_gin/checkpoints/model.pth}
EMBEDDING_DIR=${EMBEDDING_DIR:-outputs/hpc/molclr_ccrcov_embeddings_gcf_official}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_molclr_gcf_official_full}
CF_MODE=${CF_MODE:-strict_flip}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

echo "===== MOLCLR CCRCov WITH OFFICIAL GCFEXPLAINER CANDIDATES ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
echo "GCF_OFFICIAL_CANDIDATES_PATH=${GCF_OFFICIAL_CANDIDATES_PATH}"
echo "MOLCLR_ROOT=${MOLCLR_ROOT}"
echo "MOLCLR_CKPT=${MOLCLR_CKPT}"
echo "EMBEDDING_DIR=${EMBEDDING_DIR}"
echo "CF_MODE=${CF_MODE}"
echo "MIN_CF_DROP=${MIN_CF_DROP}"

if [ ! -f "${GCF_OFFICIAL_CANDIDATES_PATH}" ]; then
  echo "[ERROR] GCF official SMILES candidates not found: ${GCF_OFFICIAL_CANDIDATES_PATH}"
  echo "Run scripts/slurm/gcf_official_graph_to_smiles_rf_eval.sh first."
  exit 2
fi

LIMIT_ARGS=()
if [ -n "${MAX_PARENTS:-}" ]; then
  LIMIT_ARGS+=(--max-parents "${MAX_PARENTS}")
fi
if [ -n "${MAX_CANDIDATES:-}" ]; then
  LIMIT_ARGS+=(--max-candidates "${MAX_CANDIDATES}")
fi

python scripts/precompute_molclr_embeddings_for_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GCF_OFFICIAL_CANDIDATES_PATH}" \
  --molclr-root "${MOLCLR_ROOT}" \
  --molclr-checkpoint "${MOLCLR_CKPT}" \
  --output-dir "${EMBEDDING_DIR}" \
  --label "${LABEL:-1}" \
  "${LIMIT_ARGS[@]}"

python scripts/evaluate_ccrcov_with_molclr.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GCF_OFFICIAL_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --label "${LABEL:-1}" \
  --embedding-thresholds "${EMBEDDING_THRESHOLDS:-0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02}" \
  --cf-mode "${CF_MODE}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --output-root "${OUTPUT_ROOT}" \
  --partial-every "${PARTIAL_EVERY:-5000}"

echo "===== MOLCLR GCF OFFICIAL CCRCov DONE ====="
cat "${OUTPUT_ROOT}/combined/combined_threshold_summary.csv"
