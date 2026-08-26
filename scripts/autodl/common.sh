#!/usr/bin/env bash
# Shared, side-effect-light defaults for frozen-GNN AutoDL launchers.

set -euo pipefail

AUTODL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$AUTODL_SCRIPT_DIR" rev-parse --show-toplevel)"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# AutoDL execution clones are immutable evidence.  Never let imports create
# __pycache__ inside the code worktree.
export PYTHONDONTWRITEBYTECODE=1

export AUTODL_PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
if [[ "$AUTODL_PYTHON" != /* || ! -x "$AUTODL_PYTHON" ]]; then
  echo "AUTODL_PYTHON must be an absolute executable path: $AUTODL_PYTHON" >&2
  return 64 2>/dev/null || exit 64
fi

export AUTODL_MAX_GPUS="${AUTODL_MAX_GPUS:-2}"
export AUTODL_MIN_FREE_MEMORY_MB="${AUTODL_MIN_FREE_MEMORY_MB:-16000}"
export AUTODL_IDLE_UTIL_THRESHOLD="${AUTODL_IDLE_UTIL_THRESHOLD:-10}"
export AUTODL_IDLE_STABLE_SECONDS="${AUTODL_IDLE_STABLE_SECONDS:-60}"
export RUN_TASTEMOLNET="${RUN_TASTEMOLNET:-0}"
export PRIMARY_GNN_BACKBONE="${PRIMARY_GNN_BACKBONE:-gine}"
export PRIMARY_SEED="${PRIMARY_SEED:-7}"

TASTEMOLNET_FIXED_UPSTREAM_COMMIT="16af8ead8a17b6bd3941d9eb5879c5be75c14114"
if [[ -n "${TASTEMOLNET_UPSTREAM_COMMIT:-}" && "${TASTEMOLNET_UPSTREAM_COMMIT:-}" != "$TASTEMOLNET_FIXED_UPSTREAM_COMMIT" ]]; then
  echo "TASTEMOLNET_UPSTREAM_COMMIT conflicts with the frozen source commit" >&2
  return 64 2>/dev/null || exit 64
fi
export TASTEMOLNET_UPSTREAM_COMMIT="$TASTEMOLNET_FIXED_UPSTREAM_COMMIT"
unset TASTEMOLNET_FIXED_UPSTREAM_COMMIT

if [[ -z "${AUTODL_DATA_ROOT:-}" ]]; then
  if [[ -d /autodl-fs/data && -w /autodl-fs/data ]]; then
    AUTODL_DATA_ROOT=/autodl-fs/data
  elif [[ -d /root/autodl-fs && -w /root/autodl-fs ]]; then
    AUTODL_DATA_ROOT=/root/autodl-fs
  else
    echo "No persistent AutoDL data root found; set absolute AUTODL_DATA_ROOT" >&2
    return 64 2>/dev/null || exit 64
  fi
fi
export AUTODL_DATA_ROOT
export AUTODL_RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-$AUTODL_DATA_ROOT/counterfactual-subgraph-runtime}"
export AUTODL_ARTIFACT_ROOT="${AUTODL_ARTIFACT_ROOT:-$AUTODL_RUNTIME_ROOT/outputs}"
export AUTODL_CONTROL_ROOT="${AUTODL_CONTROL_ROOT:-$AUTODL_RUNTIME_ROOT/control}"
for autodl_absolute_path in \
  "$AUTODL_DATA_ROOT" \
  "$AUTODL_RUNTIME_ROOT" \
  "$AUTODL_ARTIFACT_ROOT" \
  "$AUTODL_CONTROL_ROOT"; do
  if [[ "$autodl_absolute_path" != /* ]]; then
    echo "AutoDL runtime paths must be absolute: $autodl_absolute_path" >&2
    return 64 2>/dev/null || exit 64
  fi
done
unset autodl_absolute_path

AUTODL_STEP0_ROOT_DEFAULT="$AUTODL_DATA_ROOT/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project"
if [[ -z "${BACE_SPLIT_ROOT:-}" ]]; then
  if [[ -d "$AUTODL_STEP0_ROOT_DEFAULT/data/processed/BACE" ]]; then
    BACE_SPLIT_ROOT="$AUTODL_STEP0_ROOT_DEFAULT/data/processed/BACE"
  elif [[ -d "$PROJECT_ROOT/data/processed/BACE" ]]; then
    BACE_SPLIT_ROOT="$PROJECT_ROOT/data/processed/BACE"
  else
    BACE_SPLIT_ROOT="$AUTODL_STEP0_ROOT_DEFAULT/data/processed/BACE"
  fi
fi
export BACE_SPLIT_ROOT
export TASTEMOLNET_SPLIT_ROOT="${TASTEMOLNET_SPLIT_ROOT:-$AUTODL_RUNTIME_ROOT/data/tastemolnet/prepared/$TASTEMOLNET_UPSTREAM_COMMIT/splits}"
export TASTEMOLNET_PREPARED_ROOT="${TASTEMOLNET_PREPARED_ROOT:-$AUTODL_RUNTIME_ROOT/data/tastemolnet/prepared/$TASTEMOLNET_UPSTREAM_COMMIT}"
export TASTEMOLNET_GRAPH_CACHE_ROOT="${TASTEMOLNET_GRAPH_CACHE_ROOT:-$AUTODL_RUNTIME_ROOT/cache/tastemolnet/$TASTEMOLNET_UPSTREAM_COMMIT/molecular_graph_v1}"
export TASTEMOLNET_LICENSE_MARKER="${TASTEMOLNET_LICENSE_MARKER:-$AUTODL_RUNTIME_ROOT/data/tastemolnet/prepared/$TASTEMOLNET_UPSTREAM_COMMIT/LICENSE_REVIEW_REQUIRED}"
for autodl_taste_path in "$TASTEMOLNET_PREPARED_ROOT" "$TASTEMOLNET_SPLIT_ROOT" "$TASTEMOLNET_GRAPH_CACHE_ROOT" "$TASTEMOLNET_LICENSE_MARKER"; do
  if [[ "$autodl_taste_path" != /* ]]; then
    echo "TasteMolNet runtime paths must be absolute: $autodl_taste_path" >&2
    return 64 2>/dev/null || exit 64
  fi
done
unset autodl_taste_path

autodl_require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "AUTODL_REQUIRED_FILE_MISSING: $path" >&2
    return 2
  fi
}

autodl_require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "AUTODL_REQUIRED_DIRECTORY_MISSING: $path" >&2
    return 2
  fi
}

autodl_find_split_manifest() {
  local split_root="$1"
  local candidate
  for candidate in \
    "$split_root/split_manifest.json" \
    "$split_root/splits/split_manifest.json" \
    "$split_root/manifest.json"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 0
}

autodl_select_one_gpu() {
  local output rc
  set +e
  output="$(
    "$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
      --project-root "$PROJECT_ROOT" \
      --data-root "$AUTODL_DATA_ROOT" \
      --max-gpus 1 \
      --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
      --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
      --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" \
      --format lines \
      --require-idle
  )"
  rc=$?
  set -e
  if [[ $rc -eq 3 ]]; then
    echo "WAITING_FOR_IDLE_GPU" >&2
    return 75
  fi
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi
  printf '%s\n' "$(printf '%s\n' "$output" | head -n 1)"
}

autodl_new_output_dir() {
  local dataset="$1" backbone="$2" profile="$3"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  printf '%s\n' "$AUTODL_ARTIFACT_ROOT/gnn_oracles/$dataset/$backbone/seed${PRIMARY_SEED}/${profile}-${stamp}"
}

autodl_passed_stage_output() {
  local stage="$1"
  shift
  "$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/autodl/exp_run.py" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$AUTODL_DATA_ROOT" \
    stage-output \
    --stage "$stage" \
    "$@"
}
