#!/bin/bash
#SBATCH --job-name=bace_globalgce_artifact_audit_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
ROOT=${ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/globalgce}
for path in final_artifact_audit.json summary.json run_manifest.json figure3_coverage_vs_k.csv figure4_coverage_vs_threshold.csv table2_globalgce_k10.csv; do test -s "$ROOT/$path" || { echo "missing $ROOT/$path" >&2; exit 2; }; done
python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); audit=json.load(open(root/'final_artifact_audit.json')); manifest=json.load(open(root/'run_manifest.json'))
assert audit.get('passed') is True
assert manifest['selection_performed_in_eval'] is False
assert manifest['threshold_fitted_on_test'] is False
assert manifest['test_used_for_selection'] is False
assert manifest['strict_flip'] is True
print('[BACE_GLOBALGCE_ARTIFACT_AUDIT_V7_OK]')
PY
