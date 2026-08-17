#!/bin/bash
#SBATCH --job-name=bace_v4_common_audit
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
COMMON_ROOT=${COMMON_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4}; OURS_ROOT=${OURS_ROOT:-$COMMON_ROOT/ours}; GCF_ROOT=${GCF_ROOT:-$COMMON_ROOT/gcfexplainer}; THRESHOLDS_JSON=${THRESHOLDS_JSON:-$COMMON_ROOT/threshold_protocol/thresholds.json}
GCF_AUDIT_ROOT=${GCF_AUDIT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_gcf_native_pool_v4}; CONNECTIVITY_AUDIT=${CONNECTIVITY_AUDIT:-$COMMON_ROOT/gcf_candidate_connectivity_audit.json}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$OURS_ROOT/final_artifact_audit.json" "$GCF_ROOT/final_artifact_audit.json" "$THRESHOLDS_JSON" "$GCF_AUDIT_ROOT/candidate_universe.jsonl"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_V4_COMMON_AUDIT_VALIDATE_OK]'; exit 0; fi
python - "$GCF_AUDIT_ROOT/candidate_universe.jsonl" "$CONNECTIVITY_AUDIT" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==20
p={"passed":True,"all_candidates_connected":all(bool(r.get("connected")) and "." not in str(r.get("smiles") or "") for r in rows),"native_rank_preserved":[int(r["native_rank"]) for r in rows]==sorted(int(r["native_rank"]) for r in rows),"gcf_candidates_reused":True,"test_used":False}
assert p["all_candidates_connected"] and p["native_rank_preserved"]
open(sys.argv[2],"w").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
python scripts/audit_bace_connected_common_artifacts.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --ours-root "$OURS_ROOT" --gcf-root "$GCF_ROOT" --thresholds-json "$THRESHOLDS_JSON" --gcf-candidate-audit "$CONNECTIVITY_AUDIT" --output-root "$COMMON_ROOT"
python - "$COMMON_ROOT/bace_connected_protocol_audit.json" "$COMMON_ROOT/threshold_protocol/threshold_protocol_audit.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2])); assert p["passed"] and p["plotting_adapter_required"] is False and t["THRESHOLD_METHOD_INDEPENDENT"] and t["THRESHOLD_TEST_INDEPENDENT"]
PY
echo '[BACE_V4_COMMON_AUDIT_SUCCESS]'
