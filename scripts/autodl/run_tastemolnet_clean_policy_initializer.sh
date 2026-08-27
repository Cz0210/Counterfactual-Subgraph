#!/usr/bin/env bash
# Foreground T5 payload.  A persistent controller must own GPU locks,
# scheduling, retries, PID state, and heartbeats.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

required=(
  TASTEMOLNET_T5_RELEASE_AUTHORITY
  TASTEMOLNET_T5_RELEASE_AUTHORITY_SHA256
  TASTEMOLNET_POLICY_RECEIPT
  TASTEMOLNET_CHEMLLM_BASE
  OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required; T5 remains release-disabled" >&2
    exit 78
  fi
  if [[ "${!name}" != /* && "$name" != TASTEMOLNET_T5_RELEASE_AUTHORITY_SHA256 ]]; then
    echo "$name must be absolute" >&2
    exit 64
  fi
done

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "2" ]]; then
  echo "T5 requires the persistent controller to bind CUDA_VISIBLE_DEVICES=2" >&2
  exit 78
fi

expected_parent="$AUTODL_RUNTIME_ROOT/outputs/autodl/tastemolnet/clean-policy-initializer"
if [[ "$(dirname "$OUTPUT_ROOT")" != "$expected_parent" ]]; then
  echo "OUTPUT_ROOT must be one direct child of $expected_parent" >&2
  exit 64
fi
if [[ ! "$(basename "$OUTPUT_ROOT")" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "OUTPUT_ROOT leaf must be an exact UTC timestamp" >&2
  exit 64
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT must be absent and fresh: $OUTPUT_ROOT" >&2
  exit 73
fi

mkdir -p "$expected_parent"
chmod 700 "$expected_parent"
export PYTHONDONTWRITEBYTECODE=1

exec "$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/build_tastemolnet_clean_policy_initializer.py" \
  build \
  --config "$PROJECT_ROOT/configs/autodl/tastemolnet_clean_policy_initializer_v1.yaml" \
  --release-authority "$TASTEMOLNET_T5_RELEASE_AUTHORITY" \
  --expected-release-authority-sha256 "$TASTEMOLNET_T5_RELEASE_AUTHORITY_SHA256" \
  --policy "$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml" \
  --policy-receipt "$TASTEMOLNET_POLICY_RECEIPT" \
  --model-path "$TASTEMOLNET_CHEMLLM_BASE" \
  --output-root "$OUTPUT_ROOT" \
  --seed "${TASTEMOLNET_T5_SEED:-7}"
