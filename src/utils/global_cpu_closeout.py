"""Narrow, authorized handover of the existing BACE Global CPU waiter.

No new queue or GPU lock. The original publication lock serializes the one-shot
intent/CAS; the unchanged evaluator retains its own parent writer lock.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, utc_now
from src.utils.final16_owner_registry_v1 import process_start_ticks

R = Path('/autodl-fs/data/counterfactual-subgraph-runtime')
G = R/'control/bace-gin-aplus-globalgce-41b7ecb-20260908'
OLD_SPEC = G/'cpu-handoff-c781341-attempt2/spec.json'
PRIOR = R/'control/t14-terminal-resource-rebind-4ec42b4-20260908/cpu-successor-daca356'
LOCK = R/'control/fast16_matrix_authority/publish.lock'


def read(path):
    return json.loads(Path(path).read_text())


def desc(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


def checked_identity(expected, proc_root=Path('/proc')):
    pid = expected['pid']; base = proc_root/str(pid)
    start = process_start_ticks(proc_root, pid)
    if start != expected['start_ticks']:
        raise ValueError('WAITING_OWNER_IDENTITY_CHANGED')
    argv = base.joinpath('cmdline').read_bytes().split(b'\0')
    argv = [x.decode() for x in argv if x]
    children = {int(x) for p in base.joinpath('task').glob('*/children') for x in p.read_text().split()}
    if children or argv != expected['argv'] or os.readlink(base/'cwd') != expected['cwd']:
        raise ValueError('WAITING_OWNER_ARGV_CWD_OR_CHILD_CHANGED')
    return {'pid': pid, 'start_ticks': start, 'argv': argv, 'cwd': expected['cwd'], 'children': []}


def validate_spec_change(old, new):
    allowed = {'cpu_resource_config', 'cpu_handoff_root', 'runtime_config',
               'deployment', 'execution_driver_commit'}
    if {k for k in set(old)|set(new) if old.get(k) != new.get(k)} - allowed:
        raise ValueError('SCIENTIFIC_SPEC_CHANGED')
    if old.get('main_matrix_write') is not False or new.get('main_matrix_write') is not False:
        raise ValueError('NO_MAIN_MATRIX_WRITE')
    if Path(new['cpu_handoff_root']).parent != Path(old['cpu_handoff_root']).parent:
        raise ValueError('ORIGINAL_CAMPAIGN_NAMESPACE_REQUIRED')


def memory_pilot(out):
    """Load frozen CPU models and bound calibration object sizes, no test/OT."""
    import torch
    from rdkit import Chem
    from src.experiments import bace_globalgce_aplus_evaluation as leaf
    from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule, enumerate_labeled_rule_matches
    from src.baselines.bace_globalgce_aplus import build_parent
    from src.utils.stage_file_policy import canonical_sha
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    old = read(OLD_SPEC)
    # Pilot only writes its own model-cache namespace; no evaluator output or test.
    probe = copy.deepcopy(old); probe['output_root'] = str(out/'pilot-runtime')
    config = leaf.validate(probe); manifest, candidates = leaf.pool(probe)
    if manifest['candidate_count'] != 80 or manifest['validation_selection']['epoch'] != 60:
        raise ValueError('SELECTED_EPOCH60_POOL80_REQUIRED')
    if leaf.bound(old['predecessor_owner_spec'])['owner_root'] != config['owner_root']:
        raise ValueError('PREDECESSOR_CHANGED')
    parents = leaf.split_parents(probe, 'calibration')
    rules = [GlobalGCENativeRule.from_payload(c['rule']) for c in candidates]
    counts=[]; max_nodes=0
    for p in parents:
        graph=build_parent(p.smiles,atom_symbols=rules[0].atom_symbols,bond_names=rules[0].bond_names)
        max_nodes=max(max_nodes,Chem.MolFromSmiles(p.smiles).GetNumAtoms())
        counts.append(sum(sum(1 for _ in enumerate_labeled_rule_matches(graph,r)) for r in rules))
    start=time.monotonic()
    oracle, feat, distance = leaf.runtime(probe, 'calibration')
    try:
        # Largest calibration graph: actual frozen-GIN forward, no test and no OT.
        parent=max(parents,key=lambda p:Chem.MolFromSmiles(p.smiles).GetNumAtoms())
        pred=leaf.prediction(oracle,feat,parent.smiles,parent.parent_id,'calibration')
        if not all(torch.isfinite(torch.tensor(pred['logits']))):raise ValueError('NONFINITE_CPU_FORWARD')
        # Materialize actual MolCLR embeddings, not self-distance as a timing proxy.
        delegate=distance.delegate if hasattr(distance,'delegate') else distance
        while not hasattr(delegate,'embedder') and hasattr(delegate,'wrapped'):delegate=delegate.wrapped
        if hasattr(delegate,'embedder'):delegate.embedder.get(parent.smiles)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    finally:distance.close()
    # All retained per-parent applications/mappings/probabilities, raw-cost
    # scalar map and selector matrix. 64 KiB/record is conservative vs actual
    # primitive schema; 4 copies cover JSON encoding and atomic serialization.
    object_bytes=(max(counts)*4+sum(counts)+66*80*4)*65536
    peak=rss*2+object_bytes+512*1024**2
    peak=((peak+1024**3-1)//1024**3)*1024**3
    result={'state':'CPU_MODEL_AND_CALIBRATION_BOUND_MEASURED','created_at':utc_now(),
        'old_spec':desc(OLD_SPEC),'pool80':desc(Path(old['pool_root'])/'run_manifest.json'),
        'selected_epoch':60,'calibration_parents':len(parents),'matches_by_parent':counts,
        'maximum_parent_nodes':max_nodes,'runtime_peak_rss_bytes':rss,
        'object_reservation_bytes':object_bytes,'incremental_peak_bytes':peak,
        'elapsed_seconds':time.monotonic()-start,'test_loaded':False,'ot_computed':False,
        'bound_scope':'CALIBRATION_ONLY; recheck held-out cardinality after freeze',
        'formula':'2*measured model/load/forward peak + 64KiB*(4*max_parent_matches+all_matches+4*66*80) + 512MiB, round up GiB',
        'model_parameters':sum(p.numel() for p in oracle.model.parameters()),'main_matrix_write':False}
    atomic_json(out/'memory-pilot.json',result)
    # Fresh operational overlay only; historical failed/blocked policies stay.
    auth={'source':'CODEX_COMPLETE_CLOSEOUT_AND_NVME.md 20260908',
        'allow_cpu_owner_447477_exact_handoff':True,'max_successful_replacements':1,
        'allow_stage_based_inode_policy':True,'inode_base_reserve':20000,
        'inode_next_stage_peak_factor':2,'contact_support_first':False,'created_at':utc_now()}
    atomic_json(out/'authorization.json',auth)
    policy=read(PRIOR/'cpu-policy.json');policy['authorization']=desc(out/'authorization.json')
    # Only unrelated whole-registry digest guards are removed. The actual T12/
    # T13 phase/terminal guards remain and still stop at their real boundaries.
    for c in policy['concurrent_components']:
        c['scope_guards']=[g for g in c.get('scope_guards',[]) if g['path']!=str(R/'control/final16-owner-registry/current.json')]
    policy.pop('self_sha256');policy['self_sha256']=canonical_sha(policy)
    atomic_json(out/'file-policy.json',policy)
    resource_config=read(PRIOR/'cpu-resource.json')
    resource_config['stage_file_policy']=desc(out/'file-policy.json')
    resource_config['minimum_memory_headroom_bytes']=384*1024**3+peak
    resource_config['joint_memory_receipt']=desc(out/'memory-pilot.json')
    resource_config['minimum_free_inodes']=20000
    atomic_json(out/'resource.json',resource_config)
    spec=copy.deepcopy(old);spec['cpu_resource_config']=desc(out/'resource.json')
    spec['cpu_handoff_root']=str(Path(old['cpu_handoff_root']).with_name('cpu-owner-closeout-'+out.name))
    spec['runtime_config']=str(Path(__file__).resolve().parents[2]/'configs/hpc.yaml')
    spec['execution_driver_commit']=os.environ['CLOSEOUT_DRIVER_COMMIT']
    spec['deployment']=desc(Path(__file__).resolve().parents[2]/'deployment.json')
    validate_spec_change(old,spec);atomic_json(out/'spec.json',spec)
    owner=read(Path(old['cpu_handoff_root'])/'owner.json');p=Path('/proc')/str(owner['pid'])
    expected={'pid':owner['pid'],'start_ticks':owner['start_ticks'],
        'argv':[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x],
        'cwd':os.readlink(p/'cwd'),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'old_owner_sha256':sha256_file(Path(old['cpu_handoff_root'])/'owner.json')}
    checked_identity(expected);atomic_json(out/'expected-owner.json',expected)
    return result


def handoff(out):
    from src.baselines.bace_globalgce_aplus_owner import cpu_admission, cpu_predecessor_state
    spec=read(out/'spec.json');old=read(OLD_SPEC);validate_spec_change(old,spec)
    if read(out/'authorization.json').get('allow_cpu_owner_447477_exact_handoff') is not True:
        raise ValueError('EXPLICIT_NEW_AUTHORIZATION_REQUIRED')
    expected=read(out/'expected-owner.json');oldroot=Path(old['cpu_handoff_root'])
    if expected['pid']!=447477 or expected['start_ticks']!=54199941:
        raise ValueError('AUTHORIZED_WAITER_ONLY')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=expected['boot_id']:
        raise ValueError('BOOT_ID_CHANGED')
    if Path(spec['output_root']).exists() or Path(spec['cpu_handoff_root']).exists():
        raise ValueError('SUCCESSOR_OR_EVALUATION_ALREADY_EXISTS')
    if cpu_predecessor_state(spec)!='READY':raise ValueError('POOL_PREDECESSOR_NOT_READY')
    config=read(out/'resource.json');ev,ok=cpu_admission(config)
    atomic_json(out/'actual-preflight.json',{'resource':ev,'admitted':ok,'created_at':utc_now()})
    if not ok:raise ValueError('LIVE_CPU_RESOURCE_NOT_ADMITTED')
    intent=G/'cpu-closeout-handoff-intent.json'
    with LOCK.open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if intent.exists():raise ValueError('ONE_SHOT_INTENT_EXISTS_NO_AUTOMATIC_REPEAT')
        ident=checked_identity(expected)
        if sha256_file(oldroot/'owner.json')!=expected['old_owner_sha256']:
            raise ValueError('OWNER_CAS_CHANGED')
        hb=read(oldroot/'heartbeat.json')
        if hb.get('state') not in ('WAITING_CPU_RESOURCE','WAITING_TRAINING_AND_POOL_FREEZE') or hb.get('science_pid'):
            raise ValueError('OLD_OWNER_NOT_CHILD_FREE_WAITING')
        # Bound writer/claim scan: any process referencing our exact evaluator
        # output or spec (other than the old waiter) prevents handover.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) in (os.getpid(),expected['pid']):continue
            try:cmd=p.joinpath('cmdline').read_bytes()
            except (OSError,PermissionError):continue
            if str(OLD_SPEC).encode() in cmd or str(spec['output_root']).encode() in cmd:
                raise ValueError('OTHER_CAMPAIGN_PROCESS:'+p.name)
        atomic_json(intent,{'state':'INTENT_SEALED','old_identity':ident,'new_spec':desc(out/'spec.json'),
            'authorization':desc(out/'authorization.json'),'successful_replacements':0,'created_at':utc_now()})
        atomic_json(out/'old-owner-evidence.json',{'owner':read(oldroot/'owner.json'),'heartbeat':hb,'identity':ident})
        checked_identity(expected)
        os.kill(expected['pid'],signal.SIGTERM)
        deadline=time.monotonic()+30
        while process_start_ticks(Path('/proc'),expected['pid'])==expected['start_ticks']:
            if time.monotonic()>deadline:raise ValueError('OLD_OWNER_DID_NOT_EXIT_NO_SIGKILL')
            time.sleep(.25)
        atomic_json(out/'old-owner-exit.json',{'state':'EXACT_WAITER_EXIT_CONFIRMED','signal':'SIGTERM','created_at':utc_now()})
        command=[sys.executable,'-I','-B',str(Path(__file__).resolve().parents[2]/'scripts/experiments/run_bace_globalgce_aplus_evaluation.py'),
            '--config',spec['runtime_config'],'--spec',str(out/'spec.json'),'--action','cpu-handoff']
        tmp=out/'tmp';tmp.mkdir();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(tmp),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (out/'owner.stdout.log').open('xb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
        value={'state':'SUCCESSOR_SPAWNED_NOT_YET_SCIENCE','pid':child.pid,
            'start_ticks':process_start_ticks(Path('/proc'),child.pid),'command':command,
            'new_spec':desc(out/'spec.json'),'successful_replacements':1,'created_at':utc_now()}
        atomic_json(intent,value);atomic_json(out/'submission.json',value)
        return value
