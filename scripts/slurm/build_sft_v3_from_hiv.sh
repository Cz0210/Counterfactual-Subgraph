#!/bin/bash
# Build a higher-quality SFT v3 dataset directly from raw HIV.csv.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=build_sft_v3

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
INPUT_CSV=${INPUT_CSV:-${PROJECT_DIR}/data/raw/AIDS/HIV.csv}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${PROJECT_DIR}/data/sft_v3_hiv_train.jsonl}
VAL_OUTPUT=${VAL_OUTPUT:-${PROJECT_DIR}/data/sft_v3_hiv_val.jsonl}
RUN_NAME=${RUN_NAME:-build_sft_v3_$(date +%Y%m%d_%H%M%S)}
POSITIVE_LABEL=${POSITIVE_LABEL:-1}
NEG_POS_RATIO=${NEG_POS_RATIO:-2.0}
VAL_RATIO=${VAL_RATIO:-0.1}
MAX_PARENTS=${MAX_PARENTS:-}
MIN_ATOM_RATIO=${MIN_ATOM_RATIO:-0.10}
MAX_ATOM_RATIO=${MAX_ATOM_RATIO:-0.55}
MIN_FRAG_ATOMS=${MIN_FRAG_ATOMS:-3}
MAX_FRAG_ATOMS=${MAX_FRAG_ATOMS:-30}
ORACLE_PATH=${ORACLE_PATH:-}
USE_ORACLE_RANKING=${USE_ORACLE_RANKING:-true}
SEED=${SEED:-7}
WARN_LOG=${WARN_LOG:-${PROJECT_DIR}/logs/${RUN_NAME}.warn.log}

cd "${PROJECT_DIR}"
mkdir -p logs

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "python path: $(which python)"
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "INPUT_CSV=${INPUT_CSV}"
echo "TRAIN_OUTPUT=${TRAIN_OUTPUT}"
echo "VAL_OUTPUT=${VAL_OUTPUT}"
echo "RUN_NAME=${RUN_NAME}"
echo "POSITIVE_LABEL=${POSITIVE_LABEL}"
echo "NEG_POS_RATIO=${NEG_POS_RATIO}"
echo "VAL_RATIO=${VAL_RATIO}"
echo "MAX_PARENTS=${MAX_PARENTS}"
echo "MIN_ATOM_RATIO=${MIN_ATOM_RATIO}"
echo "MAX_ATOM_RATIO=${MAX_ATOM_RATIO}"
echo "MIN_FRAG_ATOMS=${MIN_FRAG_ATOMS}"
echo "MAX_FRAG_ATOMS=${MAX_FRAG_ATOMS}"
echo "ORACLE_PATH=${ORACLE_PATH}"
echo "USE_ORACLE_RANKING=${USE_ORACLE_RANKING}"
echo "SEED=${SEED}"
echo "WARN_LOG=${WARN_LOG}"
echo "====================="

export PYTHONPATH=$PWD

CMD=(
  python -u scripts/build_sft_v3_from_hiv.py
  --config configs/hpc.yaml
  --input-csv "${INPUT_CSV}"
  --train-output "${TRAIN_OUTPUT}"
  --val-output "${VAL_OUTPUT}"
  --positive-label "${POSITIVE_LABEL}"
  --neg-pos-ratio "${NEG_POS_RATIO}"
  --val-ratio "${VAL_RATIO}"
  --min-atom-ratio "${MIN_ATOM_RATIO}"
  --max-atom-ratio "${MAX_ATOM_RATIO}"
  --min-frag-atoms "${MIN_FRAG_ATOMS}"
  --max-frag-atoms "${MAX_FRAG_ATOMS}"
  --seed "${SEED}"
)

if [ -n "${MAX_PARENTS}" ]; then
  CMD+=(--max-parents "${MAX_PARENTS}")
fi

if [ -n "${ORACLE_PATH}" ]; then
  CMD+=(--oracle-path "${ORACLE_PATH}")
fi

case "$(echo "${USE_ORACLE_RANKING}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y)
    CMD+=(--use-oracle-ranking)
    ;;
  *)
    CMD+=(--no-use-oracle-ranking)
    ;;
esac

"${CMD[@]}" "$@" 2> >(tee -a "${WARN_LOG}" >&2)
