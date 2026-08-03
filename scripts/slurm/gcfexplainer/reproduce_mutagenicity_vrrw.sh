#!/bin/bash
#SBATCH --job-name=mut_gcf_vrrw
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$PROJECT_ROOT" ]] || { echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] PROJECT_ROOT is required." >&2; exit 2; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-13}"

PROFILE="${PROFILE:-smoke}"

config_error() {
  local field="$1"
  local actual="$2"
  local expected="$3"
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" >&2
  echo "profile=$PROFILE" >&2
  echo "field=$field" >&2
  echo "actual=$actual" >&2
  echo "expected=$expected" >&2
  exit 2
}

if [[ "$PROFILE" == "smoke" ]]; then
  VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"
  VRRW_M="${VRRW_M:-500}"
elif [[ "$PROFILE" == "full" ]]; then
  VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"
  VRRW_M="${VRRW_M:-50000}"
else
  config_error "profile" "$PROFILE" "smoke_or_full"
fi
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/${PROFILE}_v1}"
DATASET_DIR="${DATASET_DIR:-$RUN_ROOT/dataset}"
GNN_DIR="${GNN_DIR:-$RUN_ROOT/gnn}"
if [[ "$PROFILE" == "full" ]]; then
  VRRW_DIR="${VRRW_DIR:-}"
  [[ -n "$VRRW_DIR" ]] || config_error "vrrw_dir" "" "explicit_nonempty_path"
  if [[ "$VRRW_DIR" =~ (^|/)smoke([^/]*)($|/) ]]; then
    config_error "vrrw_dir" "$VRRW_DIR" "explicit_non_smoke_output_path"
  fi
else
  VRRW_DIR="${VRRW_DIR:-$RUN_ROOT/vrrw}"
fi
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-$GNN_DIR/model_best.pth}"
NEUROSED_CHECKPOINT="${NEUROSED_CHECKPOINT:-$OFFICIAL_ROOT/data/mutagenicity/neurosed/best_model.pt}"
VRRW_ALPHA="${VRRW_ALPHA:-1.0}"
VRRW_THETA="${VRRW_THETA:-0.05}"
TELEPORT="${TELEPORT:-0.1}"
CANDIDATE_CAPACITY="${CANDIDATE_CAPACITY:-100000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
VRRW_SEED="${VRRW_SEED:-13}"
RESUME="${RESUME:-true}"
EXPECTED_FULL_GNN_CHECKPOINT="$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/gnn/model_best.pth"
EXPECTED_FULL_GNN_SHA256="22045e5a6a833d6ed980cef9834859859136a1e2f644d19d78bd63345585f239"

if [[ "$PROFILE" == "full" ]]; then
  [[ "$VRRW_PARENT_LIMIT" =~ ^[0-9]+$ && "$VRRW_PARENT_LIMIT" -eq 1448 ]] || config_error "parent_limit" "$VRRW_PARENT_LIMIT" "1448"
  [[ "$VRRW_M" =~ ^[0-9]+$ && "$VRRW_M" -eq 50000 ]] || config_error "M" "$VRRW_M" "50000"
  [[ "$VRRW_ALPHA" == "1.0" ]] || config_error "alpha" "$VRRW_ALPHA" "1.0"
  [[ "$VRRW_THETA" == "0.05" ]] || config_error "theta" "$VRRW_THETA" "0.05"
  [[ "$VRRW_SEED" =~ ^[0-9]+$ && "$VRRW_SEED" -eq 13 ]] || config_error "seed" "$VRRW_SEED" "13"
  [[ "$GNN_CHECKPOINT" == "$EXPECTED_FULL_GNN_CHECKPOINT" ]] || config_error "gnn_checkpoint" "$GNN_CHECKPOINT" "$EXPECTED_FULL_GNN_CHECKPOINT"
fi

GNN_CHECKPOINT_LOWER="$(printf '%s' "$GNN_CHECKPOINT" | tr '[:upper:]' '[:lower:]')"
NEUROSED_CHECKPOINT_LOWER="$(printf '%s' "$NEUROSED_CHECKPOINT" | tr '[:upper:]' '[:lower:]')"
if [[ "$GNN_CHECKPOINT_LOWER" == *aids* || "$GNN_CHECKPOINT_LOWER" == *hiv* ]]; then
  config_error "gnn_checkpoint" "$GNN_CHECKPOINT" "mutagenicity_checkpoint"
fi
if [[ "$PROFILE" == "full" && "$GNN_CHECKPOINT" =~ (^|/)smoke([^/]*)($|/) ]]; then
  config_error "gnn_checkpoint" "$GNN_CHECKPOINT" "full_v2_checkpoint"
fi
if [[ "$NEUROSED_CHECKPOINT_LOWER" == *aids* || "$NEUROSED_CHECKPOINT_LOWER" == *hiv* ]]; then
  config_error "neurosed_checkpoint" "$NEUROSED_CHECKPOINT" "mutagenicity_checkpoint"
fi

[[ -s "$DATASET_DIR/_PHASE_A_COMPLETE.json" ]] || config_error "dataset_complete_marker" "$DATASET_DIR/_PHASE_A_COMPLETE.json" "existing_nonempty_file"
[[ -s "$GNN_CHECKPOINT" ]] || config_error "gnn_checkpoint" "$GNN_CHECKPOINT" "existing_nonempty_file"
[[ -s "$NEUROSED_CHECKPOINT" ]] || config_error "neurosed_checkpoint" "$NEUROSED_CHECKPOINT" "existing_nonempty_file"

GNN_CHECKPOINT_SHA256="$(sha256sum "$GNN_CHECKPOINT" | awk '{print $1}')"
NEUROSED_CHECKPOINT_SHA256="$(sha256sum "$NEUROSED_CHECKPOINT" | awk '{print $1}')"
if [[ "$PROFILE" == "full" && "$GNN_CHECKPOINT_SHA256" != "$EXPECTED_FULL_GNN_SHA256" ]]; then
  config_error "gnn_checkpoint_sha256" "$GNN_CHECKPOINT_SHA256" "$EXPECTED_FULL_GNN_SHA256"
fi

if [[ -e "$VRRW_DIR/_FINALIZED.json" ]]; then
  echo "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR] finalized VRRW output cannot be overwritten." >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"
echo "[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]"
echo "profile=$PROFILE"
echo "parent_limit=$VRRW_PARENT_LIMIT"
echo "M=$VRRW_M"
echo "alpha=$VRRW_ALPHA"
echo "theta=$VRRW_THETA"
echo "seed=$VRRW_SEED"
echo "dataset_dir=$DATASET_DIR"
echo "gnn_checkpoint=$GNN_CHECKPOINT"
echo "gnn_checkpoint_sha256=$GNN_CHECKPOINT_SHA256"
echo "neurosed_checkpoint=$NEUROSED_CHECKPOINT"
echo "neurosed_checkpoint_sha256=$NEUROSED_CHECKPOINT_SHA256"
echo "generation_source_rows=1448"
echo "parent_id_hash=run_manifest.generation_source_cohort_hash"
echo "official_compatibility_patch=vrrw_alpha_endpoint_none_safe_v1"
echo "calibration_loaded=false"
echo "test_loaded=false"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "GNN_CHECKPOINT=$GNN_CHECKPOINT"
echo "NEUROSED_CHECKPOINT=$NEUROSED_CHECKPOINT"
echo "VRRW_DIR=$VRRW_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"torch_version_cuda={torch.__version__}:{torch.cuda.is_available()}")'

RESUME_ARG="--no-resume"
[[ "$RESUME" == "true" ]] && RESUME_ARG="--resume"
python scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py \
  --config configs/hpc.yaml \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$OFFICIAL_ROOT" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --neurosed-checkpoint "$NEUROSED_CHECKPOINT" \
  --output-dir "$VRRW_DIR" \
  --profile "$PROFILE" \
  --parent-limit "$VRRW_PARENT_LIMIT" \
  --m "$VRRW_M" \
  --alpha "$VRRW_ALPHA" \
  --theta "$VRRW_THETA" \
  --teleport "$TELEPORT" \
  --candidate-capacity "$CANDIDATE_CAPACITY" \
  --no-sample \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$VRRW_SEED" \
  --device1 cuda:0 \
  --device2 cuda:0 \
  "$RESUME_ARG" \
  --forbid-calibration-test

test -s "$VRRW_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_VRRW_WRAPPER_OK]"
