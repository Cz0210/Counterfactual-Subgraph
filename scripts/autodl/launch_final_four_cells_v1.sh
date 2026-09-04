#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_RUNTIME_ROOT:=/autodl-fs/data/counterfactual-subgraph-runtime}"
: "${MATRIX_AUTHORITY_ROOT:=$AUTODL_RUNTIME_ROOT/control/fast16_matrix_authority/state.json}"
: "${AUTODL_PYTHON:=/root/miniconda3/envs/smiles_pip118/bin/python}"
: "${FINAL_FOUR_REPO_ROOT:=$PWD}"
: "${FINAL_FOUR_STATE_ROOT:=$AUTODL_RUNTIME_ROOT/control/final-four-cells-v1}"
: "${SCHEDULER_POLL_SECONDS:=60}"
: "${RUN_LLM_ABLATION:=0}"
: "${RUN_GNN_ABLATION:=0}"

if [[ "$RUN_LLM_ABLATION" != 0 || "$RUN_GNN_ABLATION" != 0 ]]; then
  echo "final-four observer forbids ablation science" >&2
  exit 2
fi
if [[ ! -x "$AUTODL_PYTHON" || ! -f "$MATRIX_AUTHORITY_ROOT" ]]; then
  echo "python or matrix authority is missing" >&2
  exit 2
fi
mkdir -p "$FINAL_FOUR_STATE_ROOT"
chmod 700 "$FINAL_FOUR_STATE_ROOT"

args=(
  "$AUTODL_PYTHON" -I -B
  "$FINAL_FOUR_REPO_ROOT/scripts/autodl/run_final_four_cells_v1.py"
  --config "$FINAL_FOUR_REPO_ROOT/configs/hpc.yaml"
  --set inference.fallback_to_heuristic=false
  --state-root "$FINAL_FOUR_STATE_ROOT"
  --matrix-authority "$MATRIX_AUTHORITY_ROOT"
  --poll-seconds "$SCHEDULER_POLL_SECONDS"
)
if [[ -n "${FINAL_FOUR_TASK_SPECS:-}" ]]; then
  IFS=: read -r -a specs <<<"$FINAL_FOUR_TASK_SPECS"
  for spec in "${specs[@]}"; do
    args+=(--task-spec "$spec")
  done
fi
if [[ -n "${HPC_T8_POINTER_MIRROR:-}" ]]; then
  args+=(--hpc-t8-pointer "$HPC_T8_POINTER_MIRROR")
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
log="$FINAL_FOUR_STATE_ROOT/observer.log"
if command -v tmux >/dev/null 2>&1; then
  session="final-four-cells-v1"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "tmux session already exists: $session" >&2
    exit 75
  }
  quoted=$(printf '%q ' "${args[@]}")
  tmux new-session -d -s "$session" "cd $(printf '%q' "$FINAL_FOUR_REPO_ROOT") && exec $quoted >>$(printf '%q' "$log") 2>&1"
  echo "tmux_session=$session"
else
  cd "$FINAL_FOUR_REPO_ROOT"
  nohup "${args[@]}" >>"$log" 2>&1 </dev/null &
  echo "observer_pid=$!"
fi
echo "state_root=$FINAL_FOUR_STATE_ROOT"
echo "status_command=$AUTODL_PYTHON -I -B $FINAL_FOUR_REPO_ROOT/scripts/autodl/status_final_four_cells_v1.py --state-root $FINAL_FOUR_STATE_ROOT"
