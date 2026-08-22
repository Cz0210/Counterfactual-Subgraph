#!/usr/bin/env bash
# Foreground AutoDL bridge for the already-generated Mutagenicity GCF pool.
# It delegates only calibration/held-out evaluation to the frozen project
# implementation; candidate generation is deliberately unreachable here.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

: "${ACTION:?Set ACTION to calibration or heldout}"
: "${FROZEN_ROOT:?Set FROZEN_ROOT to the adopted read-only GCF candidate package}"
: "${FULLGRAPH_CANDIDATES_PATH:?Set FULLGRAPH_CANDIDATES_PATH explicitly}"
: "${FROZEN_MANIFEST:?Set FROZEN_MANIFEST explicitly}"
: "${TEACHER_PATH:?Set TEACHER_PATH to the frozen Mutagenicity RF}"
: "${MOLCLR_ROOT:?Set MOLCLR_ROOT explicitly}"
: "${MOLCLR_CKPT:?Set MOLCLR_CKPT explicitly}"
: "${THRESHOLDS_JSON:?Set THRESHOLDS_JSON to the frozen calibration thresholds}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh persistent attempt root}"
: "${WNODE_CACHE_DB:?Set WNODE_CACHE_DB outside the immutable worktree}"
: "${NODE_EMB_CACHE_DIR:?Set NODE_EMB_CACHE_DIR outside the immutable worktree}"

case "$OUTPUT_ROOT" in
  /autodl-fs/data/*) ;;
  *)
    echo "OUTPUT_ROOT must be under /autodl-fs/data: $OUTPUT_ROOT" >&2
    exit 2
    ;;
esac
export RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-$(dirname "$OUTPUT_ROOT")/logs}"

case "$ACTION" in
  calibration)
    : "${CALIBRATION_CSV:?Set CALIBRATION_CSV explicitly}"
    if [[ -e "$OUTPUT_ROOT" ]]; then
      echo "Calibration OUTPUT_ROOT must be fresh: $OUTPUT_ROOT" >&2
      exit 2
    fi
    DATASET_CSV="$CALIBRATION_CSV" \
    OUTPUT_DIR="$OUTPUT_ROOT" \
    RESUME=false \
    bash scripts/slurm/gcfexplainer/build_mutagenicity_wnode_calibration_matrix.sh
    echo "[MUT_GCF_LEGACY_CALIBRATION_PASS]"
    ;;
  heldout)
    : "${HELDOUT_CSV:?Set HELDOUT_CSV explicitly}"
    : "${CALIBRATION_RUN_DIR:?Set CALIBRATION_RUN_DIR to the passing dependency}"
    : "${OURS_SCHEMA_ROOT:?Set OURS_SCHEMA_ROOT to the strictly adopted Ours raw source}"
    if [[ -e "$OUTPUT_ROOT" ]]; then
      echo "Held-out OUTPUT_ROOT must be fresh: $OUTPUT_ROOT" >&2
      exit 2
    fi
    DATASET_CSV="$HELDOUT_CSV" \
    OUTPUT_DIR="$OUTPUT_ROOT/final" \
    TEST_MATRIX_DIR="$OUTPUT_ROOT/matrix" \
    RESUME=false \
    bash scripts/slurm/gcfexplainer/evaluate_mutagenicity_wnode_frozen_test.sh
    test -s "$OUTPUT_ROOT/final/final_artifact_audit.json"
    echo "[MUT_GCF_LEGACY_HELDOUT_PASS]"
    ;;
  *)
    echo "Unsupported ACTION=$ACTION" >&2
    exit 2
    ;;
esac
