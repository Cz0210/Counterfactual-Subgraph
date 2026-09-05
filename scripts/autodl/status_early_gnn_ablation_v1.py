#!/usr/bin/env python3
"""Display the new CPU admission and optional, separate AutoDL fallback gate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.ablations.gnn.early_policy import hpc_cpu_allowed,gpu_allowed

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',required=True)
    a=p.parse_args()
    e=json.loads(Path(a.evidence).read_text())
    print(json.dumps({'hpc_cpu':hpc_cpu_allowed(main_cells=e.get('main_cells',0),bace_reference_pass=e.get('bace_reference_pass',False),active_jobs=e.get('active_hpc_gnn_jobs',2)),
                     'autodl_gpu':gpu_allowed(e,family='gnn')},indent=2))
