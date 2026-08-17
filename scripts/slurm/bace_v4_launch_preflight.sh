#!/bin/bash
#SBATCH --job-name=bace_v4_launch_preflight
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_v4_infra_recovery_20260811/preflight}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs
echo "hostname=$(hostname)"
date --iso-8601=seconds
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo 'nvidia_smi=unavailable_cpu_preflight'
fi
echo "python=$(which python)"
python --version
conda env list
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
test -s configs/hpc.yaml
test -s "$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl"
test -d "$ARTIFACT_ROOT/data/processed/BACE"
test -d "$ARTIFACT_ROOT/pretrained_models/MolCLR"
temporary=$(mktemp "$PROJECT_ROOT/.bace-v4-preflight.XXXXXX")
printf 'bace-v4-preflight\n' > "$temporary"
grep -qx 'bace-v4-preflight' "$temporary"
rm -f "$temporary"
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  echo '[BACE_V4_LAUNCH_PREFLIGHT_VALIDATE_OK]'
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
python - "$OUTPUT_DIR/preflight.json" <<'PY'
import json,os,socket,subprocess,sys
payload={
  "passed": True,
  "hostname": socket.gethostname(),
  "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
  "project_root": os.getcwd(),
}
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
echo '[BACE_V4_LAUNCH_PREFLIGHT_SUCCESS]'
