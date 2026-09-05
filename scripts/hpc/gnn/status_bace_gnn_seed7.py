#!/usr/bin/env python3
"""One read-only snapshot of a GNN campaign and its Slurm dependency jobs."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign-root',required=True)
    args=p.parse_args()
    root=Path(args.campaign_root)
    receipt=json.loads((root/'submission.json').read_text())
    continuation_path=root/'resume_submission.json'
    if continuation_path.is_file():
        continuation=json.loads(continuation_path.read_text())
        if continuation['source_bundle_manifest_sha256'] != receipt['bundle_manifest_sha256']:
            raise ValueError('resume receipt input binding differs from original campaign')
        receipt['historical_jobs']=dict(receipt['jobs'])
        receipt['jobs'].update(continuation['jobs'])
        receipt['resume_driver_commit']=continuation['resume_driver_commit']
        receipt['resume_receipt']=str(continuation_path)
    states={}
    for name,path in receipt['attempt_roots'].items():
        states[name]={}
        for file in ('cpu_progress.json','benchmark.json','training_terminal.json','auto_terminal.json'):
            target=Path(path)/file
            if target.is_file():
                states[name][file]=json.loads(target.read_text())
    query=subprocess.run(['sacct','-j',','.join(receipt['jobs'].values()),'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS','-P'],text=True,capture_output=True)
    print(json.dumps({'submission':receipt,'backbones':states,'slurm':query.stdout,
                      'LLM_GPU_start_allowed':False,'LLM_gate':'verify final GNN seed7 audit then live main GPU gate'},indent=2))

if __name__=='__main__':main()
