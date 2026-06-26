#!/bin/bash
#SBATCH -J diag_gcf_path
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

mkdir -p logs

echo "===== GCF CANDIDATE PATH DIAGNOSTIC ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "PYTHONPATH=${PYTHONPATH}"
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import rdkit
    print("rdkit:", rdkit.__version__)
except Exception as exc:
    print("rdkit diagnostics failed:", repr(exc))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "==================================================="

MAX_SAMPLE_ROWS=${MAX_SAMPLE_ROWS:-20}
OUT_DIR=${OUT_DIR:-outputs/hpc/diagnostics/gcf_candidate_path_search}

python scripts/diagnose_gcf_candidates_path.py \
  --config configs/hpc.yaml \
  --project-root /share/home/u20526/czx/counterfactual-subgraph \
  --search-roots outputs/hpc logs scripts src \
  --out-dir "${OUT_DIR}" \
  --max-sample-rows "${MAX_SAMPLE_ROWS}"

echo "===== REPORT ====="
cat "${OUT_DIR}/gcf_candidate_path_report.md"
