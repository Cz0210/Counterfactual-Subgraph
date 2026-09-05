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
    publication_path=root/'publication_submission.json'
    if publication_path.is_file():
        publication=json.loads(publication_path.read_text())
        if publication['source_bundle_manifest_sha256'] != receipt['bundle_manifest_sha256']:
            raise ValueError('publication receipt input binding differs from original campaign')
        receipt['pre_publication_jobs']=dict(receipt['jobs'])
        receipt['jobs'].update(publication['jobs'])
        receipt['publication_driver_commit']=publication['publication_driver_commit']
        receipt['publication_receipt']=str(publication_path)
    sharded_path=root/'sharded_submission.json'
    sharded_state={}
    if sharded_path.is_file():
        continuation=json.loads(sharded_path.read_text())
        if continuation['source_bundle_manifest_sha256'] != receipt['bundle_manifest_sha256']:
            raise ValueError('exact-shard input binding differs from original campaign')
        receipt['pre_sharding_jobs']=dict(receipt['jobs'])
        receipt['jobs']=dict(continuation['jobs'])
        receipt['exact_sharded_continuation']=continuation
        evaluation=Path(continuation['evaluation_root'])
        for split in ('calibration','test'):
            plan_path=evaluation/f'{split}_partition.json'
            if plan_path.is_file():
                plan=json.loads(plan_path.read_text())
                total=sum(len(t['parent_ids']) for t in plan['shards'])
                completed=0
                for terminal in (evaluation/'shards'/split).glob('*/terminal.json'):
                    value=json.loads(terminal.read_text())
                    if value.get('state')=='PASS':
                        completed+=len(value['task']['parent_ids'])
                sharded_state[split]={'completed_parent_units':completed,'total_parent_units':total}
        for field,path in (('independent_core_audit',Path(continuation['publication_root'])/'independent_core_audit.json'),
                           ('result_package',Path(continuation['publication_root'])/'result_package.json')):
            if path.is_file():
                sharded_state[field]=json.loads(path.read_text())
    states={}
    for name,path in receipt['attempt_roots'].items():
        states[name]={}
        for file in ('cpu_progress.json','benchmark.json','training_terminal.json','auto_terminal.json'):
            target=Path(path)/file
            if target.is_file():
                states[name][file]=json.loads(target.read_text())
    query=subprocess.run(['sacct','-j',','.join(sorted(set(receipt['jobs'].values()))),'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS','-P'],text=True,capture_output=True)
    print(json.dumps({'submission':receipt,'backbones':states,'sharded_evaluation':sharded_state,'slurm':query.stdout,
                      'LLM_GPU_start_allowed':False,'LLM_gate':'verify final GNN seed7 audit then live main GPU gate'},indent=2))

if __name__=='__main__':main()
