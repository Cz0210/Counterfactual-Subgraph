#!/usr/bin/env python3
"""Seal a bounded afterok correction chain; no duplicate campaign or GPU job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.temperature_repair import context, require
from src.eval.bace_frozen_gnn_contracts import atomic_json,sha256_file


def submit(output_root,worktree,expected_commit):
    root,contract,*_=context(output_root)
    tree=Path(worktree).resolve(strict=True)
    require(not (root/'submission.json').exists(),'EXISTING_CHAIN_DO_NOT_DUPLICATE')
    require(subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()==expected_commit
        and contract['driver_commit']==expected_commit,'CORRECTION_COMMIT_MISMATCH')
    require(not subprocess.check_output(['git','-C',str(tree),'status','--porcelain'],text=True).strip(),
        'DIRTY_CORRECTION_WORKTREE')
    partition=subprocess.check_output(['scontrol','show','partition','intel'],text=True)
    require('State=UP' in partition,'CPU_PARTITION_UNAVAILABLE')
    import shutil
    free=shutil.disk_usage(root).free
    require(free-max(2*1024**3,free//5)>1024**3,'CORRECTION_STORAGE_SHORTFALL')
    (root/'logs').mkdir(exist_ok=True)
    receipt=dict(state='SUBMITTING',root=str(root),worktree=str(tree),driver_commit=expected_commit,
        repair_contract_sha256=sha256_file(root/'repair_contract.json'),jobs={},commands={},
        gpu_requested=False,max_concurrent_heavy_jobs=1,main_matrix_write=False,
        storage_available_bytes=free,storage_budget_bytes=1024**3,partition_evidence=partition)
    previous=None
    for action in ('fit','reconcile-calibration','freeze','reconcile-test','finish','verify-package'):
        heavy=action in {'reconcile-calibration','reconcile-test','finish','verify-package'}
        command=['sbatch','--parsable','--partition=intel',f'--cpus-per-task={8 if heavy else 2}',
            f'--mem={"32G" if heavy else "8G"}',f'--time={"12:00:00" if heavy else "01:00:00"}',
            f'--job-name=gnn-temp-{action}',f'--chdir={tree}',
            f'--output={root}/logs/%j.out',f'--error={root}/logs/%j.err',
            f'--export=ALL,GNN_EXECUTION_WORKTREE={tree},CUDA_VISIBLE_DEVICES=']
        if previous:command+=['--dependency=afterok:'+previous]
        command += [str(tree/'scripts/slurm/repair_bace_seed7_temperature_contract.sh'),
            '--action',action,'--output-root',str(root)]
        receipt['commands'][action]=command
        atomic_json(root/'submission.json',receipt)
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        require(job.isdigit(),'AMBIGUOUS_SUBMISSION_INSPECT_DO_NOT_RETRY')
        receipt['jobs'][action]=job
        atomic_json(root/'submission.json',receipt)
        previous=job
    receipt['state']='SUBMITTED'
    atomic_json(root/'submission.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--worktree',required=True)
    p.add_argument('--expected-commit',required=True)
    a=p.parse_args()
    print(json.dumps(submit(a.output_root,a.worktree,a.expected_commit),indent=2))
