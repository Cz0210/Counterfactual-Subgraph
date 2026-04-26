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
RUN_NAME=${RUN_NAME:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
if [ -n "${RUN_NAME}" ] && [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}
fi
MAX_STEPS=${MAX_STEPS:-}
LOGGING_STEPS=${LOGGING_STEPS:-}
SFT_LORA_PATH=${SFT_LORA_PATH:-}
PPO_LOOP=${PPO_LOOP:-}
DIAGNOSE_REWARD_FLOW=${DIAGNOSE_REWARD_FLOW:-}
REQUIRE_CHEMISTRY_REWARD_PATH=${REQUIRE_CHEMISTRY_REWARD_PATH:-}
DECODED_CHEM_SMOKE_TEST=${DECODED_CHEM_SMOKE_TEST:-}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-}
GEN_TOP_P=${GEN_TOP_P:-}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-}
FULL_PARENT_PENALTY=${FULL_PARENT_PENALTY:-}
EMPTY_RESIDUAL_PENALTY=${EMPTY_RESIDUAL_PENALTY:-}
REWARD_MAX_FRAGMENT_CHARS=${REWARD_MAX_FRAGMENT_CHARS:-}
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
ENABLE_MINIMAL_SYNTAX_REPAIR=${ENABLE_MINIMAL_SYNTAX_REPAIR:-}
REPAIR_MAX_EDITS=${REPAIR_MAX_EDITS:-}
REPAIR_MIN_ATOMS=${REPAIR_MIN_ATOMS:-}
REPAIR_ALLOW_PARENTHESES_FIX=${REPAIR_ALLOW_PARENTHESES_FIX:-}
REPAIR_ALLOW_RING_FIX=${REPAIR_ALLOW_RING_FIX:-}
REPAIR_ALLOW_TAIL_TRIM=${REPAIR_ALLOW_TAIL_TRIM:-}
REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE=${REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE:-}
REPAIR_PREFER_PREFIX_SALVAGE=${REPAIR_PREFER_PREFIX_SALVAGE:-}
REPAIR_MAX_SUFFIX_TRIM=${REPAIR_MAX_SUFFIX_TRIM:-}
REPAIR_MAX_ADDED_CLOSURES=${REPAIR_MAX_ADDED_CLOSURES:-}
ENABLE_COMPONENT_SALVAGE=${ENABLE_COMPONENT_SALVAGE:-}
COMPONENT_SALVAGE_METHOD=${COMPONENT_SALVAGE_METHOD:-}
COMPONENT_SALVAGE_MIN_ATOMS=${COMPONENT_SALVAGE_MIN_ATOMS:-}
MULTI_DUMMY_HARD_FAIL_THRESHOLD=${MULTI_DUMMY_HARD_FAIL_THRESHOLD:-}
ENABLE_LIGHT_DUMMY_SALVAGE=${ENABLE_LIGHT_DUMMY_SALVAGE:-}
NEAR_PARENT_HARD_RATIO=${NEAR_PARENT_HARD_RATIO:-}
MIN_RESIDUAL_ATOMS=${MIN_RESIDUAL_ATOMS:-}
MIN_RESIDUAL_RATIO=${MIN_RESIDUAL_RATIO:-}
MIN_FRAGMENT_ATOMS=${MIN_FRAGMENT_ATOMS:-}
TINY_FRAGMENT_HARD_FAIL_PENALTY=${TINY_FRAGMENT_HARD_FAIL_PENALTY:-}

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
echo "RUN_NAME=${RUN_NAME:-<unset>}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<unset>}"
echo "MAX_STEPS=${MAX_STEPS:-<unset>}"
echo "LOGGING_STEPS=${LOGGING_STEPS:-<unset>}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH:-<unset>}"
echo "PPO_LOOP=${PPO_LOOP:-<unset>}"
echo "DIAGNOSE_REWARD_FLOW=${DIAGNOSE_REWARD_FLOW:-<unset>}"
echo "REQUIRE_CHEMISTRY_REWARD_PATH=${REQUIRE_CHEMISTRY_REWARD_PATH:-<unset>}"
echo "DECODED_CHEM_SMOKE_TEST=${DECODED_CHEM_SMOKE_TEST:-<unset>}"
echo "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-<unset>}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE:-<unset>}"
echo "GEN_TOP_P=${GEN_TOP_P:-<unset>}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-<unset>}"
echo "FULL_PARENT_PENALTY=${FULL_PARENT_PENALTY:-<unset>}"
echo "EMPTY_RESIDUAL_PENALTY=${EMPTY_RESIDUAL_PENALTY:-<unset>}"
echo "REWARD_MAX_FRAGMENT_CHARS=${REWARD_MAX_FRAGMENT_CHARS:-<unset>}"
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
echo "ENABLE_MINIMAL_SYNTAX_REPAIR=${ENABLE_MINIMAL_SYNTAX_REPAIR:-<unset>}"
echo "REPAIR_MAX_EDITS=${REPAIR_MAX_EDITS:-<unset>}"
echo "REPAIR_MIN_ATOMS=${REPAIR_MIN_ATOMS:-<unset>}"
echo "REPAIR_ALLOW_PARENTHESES_FIX=${REPAIR_ALLOW_PARENTHESES_FIX:-<unset>}"
echo "REPAIR_ALLOW_RING_FIX=${REPAIR_ALLOW_RING_FIX:-<unset>}"
echo "REPAIR_ALLOW_TAIL_TRIM=${REPAIR_ALLOW_TAIL_TRIM:-<unset>}"
echo "REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE=${REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE:-<unset>}"
echo "REPAIR_PREFER_PREFIX_SALVAGE=${REPAIR_PREFER_PREFIX_SALVAGE:-<unset>}"
echo "REPAIR_MAX_SUFFIX_TRIM=${REPAIR_MAX_SUFFIX_TRIM:-<unset>}"
echo "REPAIR_MAX_ADDED_CLOSURES=${REPAIR_MAX_ADDED_CLOSURES:-<unset>}"
echo "ENABLE_COMPONENT_SALVAGE=${ENABLE_COMPONENT_SALVAGE:-<unset>}"
echo "COMPONENT_SALVAGE_METHOD=${COMPONENT_SALVAGE_METHOD:-<unset>}"
echo "COMPONENT_SALVAGE_MIN_ATOMS=${COMPONENT_SALVAGE_MIN_ATOMS:-<unset>}"
echo "MULTI_DUMMY_HARD_FAIL_THRESHOLD=${MULTI_DUMMY_HARD_FAIL_THRESHOLD:-<unset>}"
echo "ENABLE_LIGHT_DUMMY_SALVAGE=${ENABLE_LIGHT_DUMMY_SALVAGE:-<unset>}"
echo "NEAR_PARENT_HARD_RATIO=${NEAR_PARENT_HARD_RATIO:-<unset>}"
echo "MIN_RESIDUAL_ATOMS=${MIN_RESIDUAL_ATOMS:-<unset>}"
echo "MIN_RESIDUAL_RATIO=${MIN_RESIDUAL_RATIO:-<unset>}"
echo "MIN_FRAGMENT_ATOMS=${MIN_FRAGMENT_ATOMS:-<unset>}"
echo "TINY_FRAGMENT_HARD_FAIL_PENALTY=${TINY_FRAGMENT_HARD_FAIL_PENALTY:-<unset>}"
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

if [ -n "${OUTPUT_DIR}" ]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${MAX_STEPS}" ]; then
  cmd+=(--max-steps "${MAX_STEPS}")
fi
if [ -n "${LOGGING_STEPS}" ]; then
  cmd+=(--logging-steps "${LOGGING_STEPS}")
fi
if [ -n "${SFT_LORA_PATH}" ]; then
  cmd+=(--sft-lora-path "${SFT_LORA_PATH}")
fi
if [ -n "${PPO_LOOP}" ]; then
  cmd+=(--ppo-loop "${PPO_LOOP}")
fi
append_bool_flag() {
  local value="$1"
  local true_flag="$2"
  local false_flag="$3"
  local name="$4"
  if [ -n "${value}" ]; then
    case "${value,,}" in
      1|true|yes|on)
        if [ -n "${true_flag}" ]; then
          cmd+=("${true_flag}")
        fi
        ;;
      0|false|no|off)
        if [ -n "${false_flag}" ]; then
          cmd+=("${false_flag}")
        fi
        ;;
      *)
        echo "[ERROR] ${name} must be one of: true/false/1/0/yes/no/on/off"
        exit 1
        ;;
    esac
  fi
}
append_bool_flag "${DIAGNOSE_REWARD_FLOW}" "--diagnose-reward-flow" "" "DIAGNOSE_REWARD_FLOW"
append_bool_flag "${REQUIRE_CHEMISTRY_REWARD_PATH}" "--require-chemistry-reward-path" "" "REQUIRE_CHEMISTRY_REWARD_PATH"
append_bool_flag "${DECODED_CHEM_SMOKE_TEST}" "--decoded-chem-smoke-test" "" "DECODED_CHEM_SMOKE_TEST"
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
if [ -n "${FULL_PARENT_PENALTY}" ]; then
  cmd+=(--full-parent-penalty "${FULL_PARENT_PENALTY}")
fi
if [ -n "${EMPTY_RESIDUAL_PENALTY}" ]; then
  cmd+=(--empty-residual-penalty "${EMPTY_RESIDUAL_PENALTY}")
fi
if [ -n "${REWARD_MAX_FRAGMENT_CHARS}" ]; then
  cmd+=(--reward-max-fragment-chars "${REWARD_MAX_FRAGMENT_CHARS}")
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
append_bool_flag "${ENABLE_MINIMAL_SYNTAX_REPAIR}" "--enable-minimal-syntax-repair" "--no-enable-minimal-syntax-repair" "ENABLE_MINIMAL_SYNTAX_REPAIR"
if [ -n "${REPAIR_MAX_EDITS}" ]; then
  cmd+=(--repair-max-edits "${REPAIR_MAX_EDITS}")
fi
if [ -n "${REPAIR_MIN_ATOMS}" ]; then
  cmd+=(--repair-min-atoms "${REPAIR_MIN_ATOMS}")
fi
append_bool_flag "${REPAIR_ALLOW_PARENTHESES_FIX}" "--repair-allow-parentheses-fix" "--no-repair-allow-parentheses-fix" "REPAIR_ALLOW_PARENTHESES_FIX"
append_bool_flag "${REPAIR_ALLOW_RING_FIX}" "--repair-allow-ring-fix" "--no-repair-allow-ring-fix" "REPAIR_ALLOW_RING_FIX"
append_bool_flag "${REPAIR_ALLOW_TAIL_TRIM}" "--repair-allow-tail-trim" "--no-repair-allow-tail-trim" "REPAIR_ALLOW_TAIL_TRIM"
append_bool_flag "${REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE}" "--repair-allow-balanced-prefix-salvage" "--no-repair-allow-balanced-prefix-salvage" "REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE"
append_bool_flag "${REPAIR_PREFER_PREFIX_SALVAGE}" "--repair-prefer-prefix-salvage" "--no-repair-prefer-prefix-salvage" "REPAIR_PREFER_PREFIX_SALVAGE"
if [ -n "${REPAIR_MAX_SUFFIX_TRIM}" ]; then
  cmd+=(--repair-max-suffix-trim "${REPAIR_MAX_SUFFIX_TRIM}")
fi
if [ -n "${REPAIR_MAX_ADDED_CLOSURES}" ]; then
  cmd+=(--repair-max-added-closures "${REPAIR_MAX_ADDED_CLOSURES}")
fi
append_bool_flag "${ENABLE_COMPONENT_SALVAGE}" "--enable-component-salvage" "--no-enable-component-salvage" "ENABLE_COMPONENT_SALVAGE"
if [ -n "${COMPONENT_SALVAGE_METHOD}" ]; then
  cmd+=(--component-salvage-method "${COMPONENT_SALVAGE_METHOD}")
fi
if [ -n "${COMPONENT_SALVAGE_MIN_ATOMS}" ]; then
  cmd+=(--component-salvage-min-atoms "${COMPONENT_SALVAGE_MIN_ATOMS}")
fi
if [ -n "${MULTI_DUMMY_HARD_FAIL_THRESHOLD}" ]; then
  cmd+=(--multi-dummy-hard-fail-threshold "${MULTI_DUMMY_HARD_FAIL_THRESHOLD}")
fi
append_bool_flag "${ENABLE_LIGHT_DUMMY_SALVAGE}" "--enable-light-dummy-salvage" "--no-enable-light-dummy-salvage" "ENABLE_LIGHT_DUMMY_SALVAGE"
if [ -n "${NEAR_PARENT_HARD_RATIO}" ]; then
  cmd+=(--near-parent-hard-ratio "${NEAR_PARENT_HARD_RATIO}")
fi
if [ -n "${MIN_RESIDUAL_ATOMS}" ]; then
  cmd+=(--min-residual-atoms "${MIN_RESIDUAL_ATOMS}")
fi
if [ -n "${MIN_RESIDUAL_RATIO}" ]; then
  cmd+=(--min-residual-ratio "${MIN_RESIDUAL_RATIO}")
fi
if [ -n "${MIN_FRAGMENT_ATOMS}" ]; then
  cmd+=(--min-fragment-atoms "${MIN_FRAGMENT_ATOMS}")
fi
if [ -n "${TINY_FRAGMENT_HARD_FAIL_PENALTY}" ]; then
  cmd+=(--tiny-fragment-hard-fail-penalty "${TINY_FRAGMENT_HARD_FAIL_PENALTY}")
fi
cmd+=("$@")

"${cmd[@]}"
echo "===== DONE ====="
