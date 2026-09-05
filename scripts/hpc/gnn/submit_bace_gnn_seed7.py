#!/usr/bin/env python3
"""Submit one bounded CPU dependency chain; no main-table controller access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.hpc_bundle import atomic_json, verify_bundle


def submit(worktree, bundle, output, expected_commit):
    worktree=Path(worktree).resolve(strict=True)
    bundle=Path(bundle).resolve(strict=True)
    output=Path(output).resolve()
    scope=Path('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn')
    if not output.is_relative_to(scope) or output == scope or output.exists():
        raise ValueError('fresh GNN-scoped campaign root required')
    def git(*args):
        return subprocess.check_output(['git','-C',str(worktree),*args],text=True).strip()
    if git('rev-parse','HEAD') != expected_commit or git('status','--porcelain'):
        raise ValueError('immutable execution worktree mismatch')
    manifest=verify_bundle(bundle)
    if manifest['execution_commit'] != expected_commit:
        raise ValueError('input bundle must pin this exact execution commit')
    active=subprocess.check_output(['squeue','-u',os.environ.get('USER','u20526'),'-h','-o','%j'],text=True)
    if any(name.startswith('gnn-bace-') for name in active.splitlines()):
        raise ValueError('an earlier GNN campaign is already queued; inspect its receipts')
    free=shutil.disk_usage(scope.parent).free
    reserve=max(2*1024**3,int(free*.2))
    projected=2*1024**3
    if projected>free-reserve:
        raise ValueError('HPC_GNN_STORAGE_SHORTFALL')
    output.mkdir(parents=True)
    (output/'logs').mkdir()
    attempt_roots={name:str(output/name/str(uuid.uuid4())) for name in ('gatedgcn_plus','gin','gcn','gatv2')}
    model_roots={'gine':str(bundle/manifest['gine_reference_root']),**{k:str(Path(v)/'classifier') for k,v in attempt_roots.items()}}
    atomic_json(output/'model_roots.json',model_roots)
    receipt={'schema_version':'bace_gnn_seed7_slurm_v1','execution_commit':expected_commit,
             'worktree':str(worktree),'bundle':str(bundle),'bundle_manifest_sha256':manifest['manifest_sha256'],
             'attempt_roots':attempt_roots,'jobs':{},'commands':{},'max_concurrent_training_jobs':2,
             'gpu_requested':False,'main_matrix_write_allowed':False,'LLM_GPU_BLOCKED_WAITING_GNN_CORE':True,
             'storage':{'free_bytes':free,'reserve_bytes':reserve,'projected_persistent_bytes':projected},
             'evaluation_root':str(output/'evaluation'),'package_root':str(output/'package')}
    atomic_json(output/'submission.json',receipt)
    def sbatch(key, script, args, dependency=None):
        command=['sbatch','--parsable',f'--job-name=gnn-bace-{key}',f'--chdir={worktree}',
                 f'--output={output}/logs/%j.out',f'--error={output}/logs/%j.err',
                 f'--export=ALL,GNN_EXECUTION_WORKTREE={worktree},GNN_EXECUTION_COMMIT={expected_commit},GNN_INPUT_BUNDLE={bundle},GNN_MODEL_ROOTS_JSON={output}/model_roots.json,GNN_EVALUATION_ROOT={output}/evaluation,GNN_PACKAGE_ROOT={output}/package,GNN_ENVIRONMENT_MANIFEST={output}/environment.json,CUDA_VISIBLE_DEVICES=']
        if dependency:
            command.append('--dependency='+dependency)
        command += [str(worktree/script),*map(str,args)]
        receipt['commands'][key]=command
        atomic_json(output/'submission.json',receipt)
        result=subprocess.check_output(command,text=True).strip()
        job=result.split(';')[0]
        if not job.isdigit():
            raise ValueError('ambiguous sbatch response: '+result)
        receipt['jobs'][key]=job
        atomic_json(output/'submission.json',receipt)
        return job
    pre=sbatch('preflight','scripts/slurm/preflight_bace_gnn_cpu.sh',
               ['--bundle-root',bundle,'--output',output/'environment.json'])
    for name,previous in (('gatedgcn_plus',None),('gin',None),('gcn','gatedgcn_plus'),('gatv2','gin')):
        dependency='afterok:'+pre if previous is None else 'afterany:'+receipt['jobs'][previous]
        sbatch(name,'scripts/slurm/run_bace_gnn_cpu.sh',
               ['--bundle-root',bundle,'--backbone',name,'--phase','auto','--output-root',attempt_roots[name],
                '--config',bundle/manifest['backbone_configs'][name],'--cpu-threads','8'],dependency)
    dependency='afterok:'+':'.join(receipt['jobs'][k] for k in attempt_roots)
    evaluation=sbatch('evaluate','scripts/slurm/evaluate_bace_gnn_seed7.sh',
        ['--bundle-root',bundle,'--model-roots-json',output/'model_roots.json','--output-root',output/'evaluation'],dependency)
    sbatch('package','scripts/slurm/package_bace_gnn_seed7.sh',
        ['--evaluation-root',output/'evaluation','--output-root',output/'package',
         '--environment-manifest',output/'environment.json','--execution-commit',expected_commit],'afterok:'+evaluation)
    receipt['state']='SUBMITTED'
    atomic_json(output/'submission.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/hpc.yaml')
    for key in ('worktree','bundle','output','expected-commit'):
        p.add_argument('--'+key,required=True)
    a=vars(p.parse_args());a.pop('config')
    print(json.dumps(submit(**a),indent=2))
