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

MOLCLR_ROOT=${MOLCLR_ROOT:-${PROJECT_ROOT}/pretrained_models/MolCLR}
MOLCLR_CKPT=${MOLCLR_CKPT:-}
MOLCLR_CANDIDATES=()
GT_FULLGRAPH_DEFAULT="${PROJECT_ROOT}/outputs/hpc/baselines/gt_fullgraph/label1_opposite_fullgraph_candidates_max2000_seed0.csv"
if [ -z "${GT_FULLGRAPH_CANDIDATES_PATH:-}" ] && [ -n "${GCF_CANDIDATES_PATH:-}" ]; then
  GT_FULLGRAPH_CANDIDATES_PATH="${GCF_CANDIDATES_PATH}"
fi
GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH:-${GT_FULLGRAPH_DEFAULT}}
CLEAR_FULLGRAPH_CANDIDATES_PATH=${CLEAR_FULLGRAPH_CANDIDATES_PATH:-}

resolve_molclr_checkpoint() {
  if [ -n "${MOLCLR_CKPT:-}" ]; then
    return
  fi
  local default_ckpt="${MOLCLR_ROOT}/ckpt/pretrained_gin/checkpoints/model.pth"
  if [ -f "${default_ckpt}" ]; then
    MOLCLR_CKPT="${default_ckpt}"
    return
  fi
  local search_root
  for search_root in "${MOLCLR_ROOT}/ckpt/pretrained_gin" "${MOLCLR_ROOT}/ckpt/pretrained_gcn"; do
    if [ -d "${search_root}" ]; then
      while IFS= read -r candidate; do
        MOLCLR_CANDIDATES+=("${candidate}")
      done < <(find "${search_root}" -type f \( -iname "*.pth" -o -iname "*.pt" -o -iname "*.ckpt" \) | sort)
    fi
  done
  local candidate
  for candidate in "${MOLCLR_CANDIDATES[@]}"; do
    if [[ "${candidate}" == *pretrained_gin* ]]; then
      MOLCLR_CKPT="${candidate}"
      return
    fi
  done
  if [ "${#MOLCLR_CANDIDATES[@]}" -gt 0 ]; then
    MOLCLR_CKPT="${MOLCLR_CANDIDATES[0]}"
  fi
}

resolve_molclr_checkpoint

echo "[MOLCLR_CONFIG]"
echo "MOLCLR_ROOT=${MOLCLR_ROOT}"
echo "MOLCLR_CKPT=${MOLCLR_CKPT:-}"
echo "[GT_FULLGRAPH_CONFIG]"
echo "GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH}"
echo "[CLEAR_RF_FULLGRAPH_CONFIG]"
echo "CLEAR_FULLGRAPH_CANDIDATES_PATH=${CLEAR_FULLGRAPH_CANDIDATES_PATH:-}"

if [ ! -d "${MOLCLR_ROOT}" ] || [ -z "${MOLCLR_CKPT:-}" ] || [ ! -f "${MOLCLR_CKPT}" ]; then
  echo "[MOLCLR_CANDIDATES]"
  if [ "${#MOLCLR_CANDIDATES[@]}" -gt 0 ]; then
    printf '%s\n' "${MOLCLR_CANDIDATES[@]}"
  else
    echo "(none found under ${MOLCLR_ROOT}/ckpt/pretrained_gin or ${MOLCLR_ROOT}/ckpt/pretrained_gcn)"
  fi
  echo "[ERROR] MOLCLR_ROOT and MOLCLR_CKPT are required. Please submit with:"
  echo "MOLCLR_ROOT=/path/to/MolCLR MOLCLR_CKPT=/path/to/model.pth sbatch scripts/slurm/molclr_hiv_precompute_embeddings_full.sh"
  exit 2
fi

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset-csv "${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  --ours-selected-path "${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}"
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}"
  --molclr-root "${MOLCLR_ROOT}"
  --molclr-checkpoint "${MOLCLR_CKPT}"
  --output-dir "${OUTPUT_DIR:-outputs/hpc/molclr_ccrcov_embeddings}"
  --label "${LABEL:-1}"
  --batch-size "${BATCH_SIZE:-64}"
  --device "${DEVICE:-cuda}"
  --invalid-policy skip
)
if [ -n "${CLEAR_FULLGRAPH_CANDIDATES_PATH:-}" ]; then
  args+=(--clear-fullgraph-candidates-path "${CLEAR_FULLGRAPH_CANDIDATES_PATH}")
fi

python scripts/precompute_molclr_embeddings_for_ccrcov.py "${args[@]}"

echo "===== MOLCLR CCRCov PRECOMPUTE FULL DONE ====="
