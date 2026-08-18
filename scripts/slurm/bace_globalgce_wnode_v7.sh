#!/bin/bash
#SBATCH --job-name=bace_globalgce_wnode_v7
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_globalgce_v7}
CANDIDATE_PATH=$SELECTOR_ROOT/selected_top20_for_eval.csv
SELECTION_MANIFEST=$SELECTOR_ROOT/frozen_selection.json
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/thresholds.json}
REFERENCE_ROOT=${REFERENCE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/ours}
COMMON_ROOT=${COMMON_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6}
WORK_DIR=${WORK_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/work/bace_globalgce_v7}
PERSISTENT_SCRATCH_ROOT=${PERSISTENT_SCRATCH_ROOT:-/share/project/p20526/u20526/counterfactual-subgraph}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$PERSISTENT_SCRATCH_ROOT/globalgce_bace_v7/test_wnode.sqlite3}
for path in "$CANDIDATE_PATH" "$SELECTION_MANIFEST" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON" "$REFERENCE_ROOT/final_artifact_audit.json"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
args=(python scripts/evaluate_bace_method.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --method globalgce --candidate-path "$CANDIDATE_PATH" --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --test-csv "$TEST_CSV" --thresholds-json "$THRESHOLDS_JSON" --work-dir "$WORK_DIR"
  --output-dir "$COMMON_ROOT/globalgce" --expected-test-parents 116 --device cuda
  --selection-manifest "$SELECTION_MANIFEST" --test-evaluation-count 1
  --reference-artifact-root "$REFERENCE_ROOT" --action-semantics-version connected_sanitized_residual_v1
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
  --wnode-cache-db "$WNODE_CACHE_DB" --resume)
echo "hostname=$(hostname) commit=$(git rev-parse HEAD)"; python -c 'import torch; assert torch.cuda.device_count()==1; print("cuda_devices=1")'
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then echo '[BACE_GLOBALGCE_WNODE_V7_VALIDATE_OK]'; exit 0; fi
test ! -e "$COMMON_ROOT/globalgce" || { echo "output collision" >&2; exit 2; }
mkdir -p "$(dirname "$WNODE_CACHE_DB")"
"${args[@]}"
