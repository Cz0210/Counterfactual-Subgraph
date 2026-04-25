#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-}
GEN_TOP_P=${GEN_TOP_P:-}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-}
ENABLE_PARENT_AWARE_REPAIR=${ENABLE_PARENT_AWARE_REPAIR:-}
REPAIR_MIN_SIMILARITY=${REPAIR_MIN_SIMILARITY:-}
REPAIR_MAX_CANDIDATES=${REPAIR_MAX_CANDIDATES:-}
ENABLE_PARENT_PROJECTION=${ENABLE_PARENT_PROJECTION:-}
PROJECTION_MIN_SCORE=${PROJECTION_MIN_SCORE:-}
PROJECTION_MAX_CANDIDATES=${PROJECTION_MAX_CANDIDATES:-}
PROJECTION_MIN_ATOMS=${PROJECTION_MIN_ATOMS:-}
PROJECTION_MAX_ATOM_RATIO=${PROJECTION_MAX_ATOM_RATIO:-}
PROJECTION_PENALTY=${PROJECTION_PENALTY:-}
PROJECTION_ENABLE_KHOP3=${PROJECTION_ENABLE_KHOP3:-}
PROJECTION_MCS_TIMEOUT=${PROJECTION_MCS_TIMEOUT:-}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

cd "${PROJECT_DIR}"

export PYTHONPATH=$PWD
echo "repo pwd: $(pwd)"
echo "PYTHONPATH(after export): ${PYTHONPATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-<unset>}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE:-<unset>}"
echo "GEN_TOP_P=${GEN_TOP_P:-<unset>}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-<unset>}"
echo "ENABLE_PARENT_AWARE_REPAIR=${ENABLE_PARENT_AWARE_REPAIR:-<unset>}"
echo "REPAIR_MIN_SIMILARITY=${REPAIR_MIN_SIMILARITY:-<unset>}"
echo "REPAIR_MAX_CANDIDATES=${REPAIR_MAX_CANDIDATES:-<unset>}"
echo "ENABLE_PARENT_PROJECTION=${ENABLE_PARENT_PROJECTION:-<unset>}"
echo "PROJECTION_MIN_SCORE=${PROJECTION_MIN_SCORE:-<unset>}"
echo "PROJECTION_MAX_CANDIDATES=${PROJECTION_MAX_CANDIDATES:-<unset>}"
echo "PROJECTION_MIN_ATOMS=${PROJECTION_MIN_ATOMS:-<unset>}"
echo "PROJECTION_MAX_ATOM_RATIO=${PROJECTION_MAX_ATOM_RATIO:-<unset>}"
echo "PROJECTION_PENALTY=${PROJECTION_PENALTY:-<unset>}"
echo "PROJECTION_ENABLE_KHOP3=${PROJECTION_ENABLE_KHOP3:-<unset>}"
echo "PROJECTION_MCS_TIMEOUT=${PROJECTION_MCS_TIMEOUT:-<unset>}"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  exit 1
fi

echo "===== RUNNING PPO TRAINING ====="
cmd=(
  python
  scripts/train_ppo.py
  --config
  configs/hpc.yaml
  --teacher-path
  "${TEACHER_PATH}"
  --require-teacher-sem
  --teacher-sem-scale
  1.0
  --teacher-sem-missing-penalty
  -5.0
  --teacher-cf-flip-bonus
  1.0
)

if [ -n "${GEN_MAX_NEW_TOKENS}" ]; then
  cmd+=(--gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}")
fi
if [ -n "${GEN_TEMPERATURE}" ]; then
  cmd+=(--gen-temperature "${GEN_TEMPERATURE}")
fi
if [ -n "${GEN_TOP_P}" ]; then
  cmd+=(--gen-top-p "${GEN_TOP_P}")
fi
if [ -n "${GEN_DO_SAMPLE}" ]; then
  case "${GEN_DO_SAMPLE,,}" in
    1|true|yes|on)
      cmd+=(--gen-do-sample)
      ;;
    0|false|no|off)
      cmd+=(--no-gen-do-sample)
      ;;
    *)
      echo "[ERROR] GEN_DO_SAMPLE must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${ENABLE_PARENT_AWARE_REPAIR}" ]; then
  case "${ENABLE_PARENT_AWARE_REPAIR,,}" in
    1|true|yes|on)
      cmd+=(--enable-parent-aware-repair)
      ;;
    0|false|no|off)
      cmd+=(--no-enable-parent-aware-repair)
      ;;
    *)
      echo "[ERROR] ENABLE_PARENT_AWARE_REPAIR must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${REPAIR_MIN_SIMILARITY}" ]; then
  cmd+=(--repair-min-similarity "${REPAIR_MIN_SIMILARITY}")
fi
if [ -n "${REPAIR_MAX_CANDIDATES}" ]; then
  cmd+=(--repair-max-candidates "${REPAIR_MAX_CANDIDATES}")
fi
if [ -n "${ENABLE_PARENT_PROJECTION}" ]; then
  case "${ENABLE_PARENT_PROJECTION,,}" in
    1|true|yes|on)
      cmd+=(--enable-parent-projection)
      ;;
    0|false|no|off)
      cmd+=(--no-enable-parent-projection)
      ;;
    *)
      echo "[ERROR] ENABLE_PARENT_PROJECTION must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${PROJECTION_MIN_SCORE}" ]; then
  cmd+=(--projection-min-score "${PROJECTION_MIN_SCORE}")
fi
if [ -n "${PROJECTION_MAX_CANDIDATES}" ]; then
  cmd+=(--projection-max-candidates "${PROJECTION_MAX_CANDIDATES}")
fi
if [ -n "${PROJECTION_MIN_ATOMS}" ]; then
  cmd+=(--projection-min-atoms "${PROJECTION_MIN_ATOMS}")
fi
if [ -n "${PROJECTION_MAX_ATOM_RATIO}" ]; then
  cmd+=(--projection-max-atom-ratio "${PROJECTION_MAX_ATOM_RATIO}")
fi
if [ -n "${PROJECTION_PENALTY}" ]; then
  cmd+=(--projection-penalty "${PROJECTION_PENALTY}")
fi
if [ -n "${PROJECTION_ENABLE_KHOP3}" ]; then
  case "${PROJECTION_ENABLE_KHOP3,,}" in
    1|true|yes|on)
      cmd+=(--projection-enable-khop3)
      ;;
    0|false|no|off)
      cmd+=(--no-projection-enable-khop3)
      ;;
    *)
      echo "[ERROR] PROJECTION_ENABLE_KHOP3 must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${PROJECTION_MCS_TIMEOUT}" ]; then
  cmd+=(--projection-mcs-timeout "${PROJECTION_MCS_TIMEOUT}")
fi
cmd+=("$@")

"${cmd[@]}"
echo "===== DONE ====="
