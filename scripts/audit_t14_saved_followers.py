#!/usr/bin/env python3
"""CPU-only bounded postprocessing of saved T14 records; never a science replay."""
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.t14_saved_follower_audit import audit
from src.utils.main_ready_task_specs import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True);p.add_argument('--set', action='append',default=[])
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu'],required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('CPU_ONLY_JOB_REQUIRED')
    if a.set!=['inference.fallback_to_heuristic=false'] or not Path(a.config).is_file():
        raise ValueError('EXPLICIT_CONFIG_AND_NO_HEURISTIC_REQUIRED')
    a.output_root.mkdir(parents=True,exist_ok=False)
    allocation=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
    if any('gpu' in token.lower() for token in allocation.split() if token.startswith(('AllocTRES=','ReqTRES=','TresPerNode='))):
        raise ValueError('SLURM_ALLOCATION_CONTAINS_GPU')
    result=audit(a.input_root)
    result.update(slurm_job_id=os.environ['SLURM_JOB_ID'],slurm_allocation=allocation,
                  input_root=str(a.input_root),cuda_visible_devices='')
    atomic_json(a.output_root/'follower_evidence_audit.json',result)
    print(json.dumps({k:result[k] for k in ['state','slurm_job_id','new_transitions','first_action_difference','first_blocker']}))
    return 0
if __name__=='__main__': raise SystemExit(main())
