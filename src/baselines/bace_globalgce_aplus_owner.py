"""Dataset-specific stage wiring into the existing AutoDL owner and leases.

No reservation, registry, lock, or scheduler is defined here. The same bound
repair root gets one fresh formal campaign and may resume its own checkpoint.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.ablations.llm.existing_gpu_owner import ResourceSampler, run_owned_child, read_small, memory_headroom
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, stable_sha256, utc_now
from src.utils.stage_file_policy import config_file_admission
from src.utils.final16_owner_registry_v1 import process_start_ticks


def predecessor_ready(descriptor):
    terminal_path=Path(descriptor['prepared_root'])/'terminal.json'
    if not terminal_path.is_file(): return False
    value,_=read_small(terminal_path)
    if (value.get('state')!='INPUTS_BOUND'
        or value.get('config_sha256')!=descriptor['training_config_sha256']
        or not value.get('identity_fixtures')
        or value.get('positive_control',{}).get('state')!='PASS'
        or value.get('test_loaded') is not False):
        raise ValueError('APLUS_REAL_INPUT_IDENTITY_POSITIVE_GATE_REQUIRED')
    return True


def training_command(spec, config, stage, *, resume):
    action = 'train-canary' if stage == 'canary' else 'train'
    root = config['gpu_canary_root'] if stage == 'canary' else config['formal_output_root']
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve().parents[2] / 'scripts/run_bace_globalgce_aplus.py'),
        '--config', spec['runtime_config'], '--repair-config', spec['training_contract'],
        '--rematerialization-root', config['rematerialization_root'], '--device', 'cuda:0',
        '--action', action, '--output-root', root]
    if resume: command.append('--resume')
    return command


def cpu_admission(config):
    disk = os.statvfs(config['persistent_root'])
    evidence = config_file_admission(config, disk.f_favail, stage_id='llm_cpu_evaluation')
    headroom = memory_headroom(Path(config['proc_root']), Path(config['cgroup_memory_root']))
    evidence.update(headroom_bytes=headroom, free_bytes=disk.f_bavail*disk.f_frsize)
    return evidence, bool(evidence['admitted'] and not evidence['pause_requested'] and
        headroom >= config['minimum_memory_headroom_bytes'] and
        disk.f_bavail*disk.f_frsize >= config['minimum_persistent_free_bytes'])


def run_owner(spec_path):
    spec, _ = read_small(spec_path)
    if spec.get('schema') != 'bace_globalgce_aplus_existing_owner_v1':
        raise ValueError('EXPLICIT_GLOBALGCE_APLUS_STAGE_SPEC_REQUIRED')
    for key in ('training_contract','resource_config','owner_root','runtime_config'):
        if not Path(spec[key]).is_absolute(): raise ValueError('OWNER_ABSOLUTE_PATH_REQUIRED:'+key)
    if sha256_file(spec['training_contract']) != spec['training_contract_file_sha256']:
        raise ValueError('TRAINING_CONTRACT_CHANGED')
    config, _ = read_small(spec['training_contract'])
    if config.get('owner_root') != spec['owner_root']:
        raise ValueError('ONE_CANONICAL_OWNER_ROOT_PER_REPAIR_CONTRACT')
    resource, _ = read_small(spec['resource_config'])
    if spec['gpu_index'] != 0 or spec['gpu_borrow_or_colocation'] is not False:
        raise ValueError('GPU0_EXCLUSIVE_ONLY')
    root = Path(spec['owner_root']); root.mkdir(parents=True, exist_ok=False)
    identity = {'pid':os.getpid(), 'start_ticks':process_start_ticks(Path('/proc'),os.getpid()),
        'owner_spec':str(spec_path), 'owner_spec_sha256':sha256_file(spec_path), 'created_at':utc_now()}
    atomic_json(root/'owner.json', identity)
    def heartbeat(state, **extra):
        atomic_json(root/'heartbeat.json', {**identity, 'state':state, 'updated_at':utc_now(), **extra})
    try:
        for stage in ('canary','formal'):
            output = Path(config['gpu_canary_root'] if stage=='canary' else config['formal_output_root'])
            expected = 'CANARY_COMPLETE' if stage=='canary' else 'REPAIR_TRAINING_COMPLETE'
            attempt=0
            while True:
                if (output/'terminal.json').exists():
                    terminal,_=read_small(output/'terminal.json')
                    if terminal.get('state') != expected: raise ValueError('SCIENCE_TERMINAL_NOT_COMPLETE:'+stage)
                    break
                while not predecessor_ready(spec['prepared_inputs']):
                    heartbeat('WAITING_GIN_INPUT_AND_IDENTITY_GATE', next_stage=stage,
                        resource_lock_acquired=False, science_started=False)
                    time.sleep(60)
                resume=(output/'latest.pt').is_file()
                if output.exists() and not resume:
                    raise ValueError('PARTIAL_STAGE_WITHOUT_LEGAL_CHECKPOINT:'+str(output))
                attempt+=1
                sampler=ResourceSampler(resource,0,spec['gpu_uuid'],task_family='globalgce_aplus',
                    reach_contract={'path':spec['training_contract'],'sha256':spec['training_contract_file_sha256']})
                child_owner=root/f'{stage}-lease-{attempt:04d}'
                heartbeat('WAITING_EXISTING_EXCLUSIVE_GPU_LEASE', next_stage=stage, child_owner=str(child_owner))
                code=run_owned_child(command=training_command(spec,config,stage,resume=resume),
                    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    sampler=sampler,output_root=child_owner,lock_root=resource['gpu_lock_root'],
                    run_id='bace-aplus-v2-'+root.name+'-'+stage, interval=60,max_wait_seconds=86400)
                if code==75:
                    heartbeat('PAUSED_AT_COMMITTED_OPTIMIZER_BOUNDARY', next_stage=stage, child_owner=str(child_owner))
                    time.sleep(60); continue
                if code!=0: raise RuntimeError(f'{stage} child failed code={code}; no automatic fresh retry')
        terminal,_=read_small(Path(config['formal_output_root'])/'terminal.json')
        if not terminal.get('performance_target_met'):
            heartbeat('RESEARCH_TARGET_UNMET', formal_terminal=terminal)
            atomic_json(root/'terminal.json', {'state':'RESEARCH_TARGET_UNMET','created_at':utc_now()})
            return 0
        # GPU has already exited and released the same owner locks. Each CPU
        # command is an existing evaluator stage, with fresh output paths.
        for stage in spec['cpu_successors']:
            if stage.get('opens_test') and not Path(stage['requires_selector_freeze']).is_file():
                raise ValueError('TEST_REQUIRES_FRESH_SELECTOR_FREEZE')
            while True:
                evidence, admitted=cpu_admission(resource)
                heartbeat('CPU_STAGE_ADMISSION', next_stage=stage['name'],resource=evidence)
                if admitted: break
                time.sleep(60)
            output=root/(stage['name']+'.log')
            with output.open('xb') as stream:
                child=subprocess.Popen(stage['command'],stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),close_fds=True)
                while child.poll() is None:
                    heartbeat('CPU_SCIENCE_RUNNING',next_stage=stage['name'],science_pid=child.pid)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:pass
            if child.returncode: raise RuntimeError(f"CPU stage {stage['name']} failed code={child.returncode}")
        # The original authority's version-supersession interface is a distinct
        # independent audit/publishing stage. Never claim it ran from exit0 here.
        final_receipt=spec.get('final_evaluation_receipt')
        evaluated=False
        if final_receipt and Path(final_receipt).is_file():
            final,_=read_small(final_receipt)
            evaluated=(final.get('state')=='APLUS_GLOBALGCE_EVALUATION_COMPLETE'
                and final.get('training_contract_sha256')==stable_sha256(config)
                and final.get('main_matrix_write') is False)
        final_state=('APLUS_INDEPENDENT_CPU_EVALUATION_COMPLETE' if evaluated else
            'POOL_FROZEN_EVALUATION_SUCCESSOR_PENDING' if spec['cpu_successors'] else
            'BLOCKED_CPU_SUCCESSORS_NOT_BOUND')
        atomic_json(root/'terminal.json', {'state':final_state,'created_at':utc_now(),'main_matrix_written':False})
        heartbeat(final_state)
        return 0 if spec['cpu_successors'] else 2
    except BaseException as error:
        atomic_json(root/'terminal.json', {'state':'FAILED_ENGINEERING','error':repr(error),'created_at':utc_now()})
        raise


def cpu_predecessor_state(evaluation_spec):
    """One-time successor of this exact completed GPU owner, not a new queue."""
    from src.experiments.bace_globalgce_aplus_evaluation import bound, validate, pool
    config=validate(evaluation_spec)
    predecessor=bound(evaluation_spec['predecessor_owner_spec'])
    if predecessor['owner_root']!=config['owner_root'] or predecessor['training_contract']!=evaluation_spec['training_contract']['path']:
        raise ValueError('CPU_PREDECESSOR_OWNER_CHANGED')
    root=Path(predecessor['owner_root']);terminal=root/'terminal.json'
    if not terminal.exists():return 'WAITING_TRAINING_AND_POOL_FREEZE'
    state=read_small(terminal)[0].get('state')
    if state not in ('POOL_FROZEN_EVALUATION_SUCCESSOR_PENDING','APLUS_INDEPENDENT_CPU_EVALUATION_COMPLETE'):
        raise ValueError('CPU_PREDECESSOR_NOT_SCIENTIFICALLY_COMPLETE:'+str(state))
    manifest,_=pool(evaluation_spec)
    if manifest['state']!='TRAIN_POOL_FROZEN':raise ValueError('FROZEN_TRAIN_POOL_REQUIRED')
    return 'READY'


def prepare_cpu_freeze_repair(spec,spec_path,receipt_path):
    """Resume only the failed exporter, under the original publication lock."""
    import fcntl
    from src.experiments.bace_globalgce_aplus_evaluation import bound
    receipt=read_small(receipt_path)[0]
    if (receipt.get('reason')!='MISSING_CALIBRATION_SUMMARY_STRICT_FLIP_COUNT'
        or receipt.get('source_spec_sha256')!=sha256_file(spec_path)
        or receipt.get('repair_attempt') not in (1,2)
        or receipt.get('resume_action')!='freeze'
        or receipt.get('science_records_changed') is not False):
        raise ValueError('NARROW_FREEZE_REPAIR_RECEIPT_REQUIRED')
    old=bound(receipt['failed_owner']);terminal=bound(receipt['failed_terminal'])
    if terminal.get('state')!='BLOCKED' or 'freeze failed code=1' not in terminal.get('error',''):
        raise ValueError('FAILED_FREEZE_ONLY')
    completed=bound(receipt['calibration_terminal'])
    if completed.get('state')!='EVALUATION_COMPLETE' or completed.get('spec_sha256')!=stable_sha256(spec):
        raise ValueError('COMPLETED_CALIBRATION_REQUIRED')
    root=Path(receipt['fresh_owner_root'])
    if root.parent!=Path(spec['output_root']).parent or root==Path(spec['cpu_handoff_root']):
        raise ValueError('FRESH_SAME_CAMPAIGN_OWNER_REQUIRED')
    lock=Path('/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/publish.lock')
    with lock.open('r+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if process_start_ticks(Path('/proc'),old['pid'])==old['start_ticks']:
            raise ValueError('PREVIOUS_OWNER_STILL_ALIVE')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if str(spec_path).encode() in argv:
                raise ValueError('DUPLICATE_EVALUATION_PROCESS:'+proc.name)
            for fd in proc.joinpath('fd').glob('*'):
                try:target=os.readlink(fd).removesuffix(' (deleted)')
                except OSError:continue
                if target.startswith(str(spec['output_root'])+'/'):
                    raise ValueError('EVALUATION_OPEN_FILE:'+proc.name)
        root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'repair_claim.json',{'receipt_path':str(receipt_path),
            'receipt_sha256':sha256_file(receipt_path),'pid':os.getpid(),
            'start_ticks':process_start_ticks(Path('/proc'),os.getpid()),'created_at':utc_now()})
        atomic_json(Path(spec['output_root'])/'freeze_repair_execution.json',receipt)
    return root


def run_cpu_handoff(evaluation_spec_path,*,repair_receipt=None):
    """Fresh immutable four-stage CPU handoff after the existing owner retires.

    Uses the same resource admission and output contracts. No GPU lock is
    acquired; no live GPU owner's spec/code is edited or dynamically reloaded.
    """
    from src.experiments.bace_globalgce_aplus_evaluation import bound,validate
    spec=read_small(evaluation_spec_path)[0];validate(spec)
    root=Path(spec['cpu_handoff_root'])
    if not root.is_absolute() or root.parent!=Path(spec['output_root']).parent:
        raise ValueError('CANONICAL_CPU_HANDOFF_PATH_REQUIRED')
    if repair_receipt:
        root=prepare_cpu_freeze_repair(spec,evaluation_spec_path,repair_receipt)
    else:root.mkdir(parents=True,exist_ok=False)
    identity={'pid':os.getpid(),'start_ticks':process_start_ticks(Path('/proc'),os.getpid()),
        'spec_path':str(evaluation_spec_path),'spec_sha256':sha256_file(evaluation_spec_path),'created_at':utc_now(),
        'cpu_only':True,'gpu_lease_acquired':False}
    atomic_json(root/'owner.json',identity)
    def heartbeat(state,**extra):atomic_json(root/'heartbeat.json',{**identity,'state':state,'updated_at':utc_now(),**extra})
    try:
        while cpu_predecessor_state(spec)!='READY':
            heartbeat('WAITING_TRAINING_AND_POOL_FREEZE',science_started=False);time.sleep(60)
        actions=('freeze','test','aggregate') if repair_receipt else ('calibration','freeze','test','aggregate')
        for action in actions:
            attempt=0
            while True:
                resource=bound(spec['cpu_resource_config']);evidence,ready=cpu_admission(resource)
                if not ready:
                    heartbeat('WAITING_CPU_RESOURCE',next_stage=action,resource=evidence);time.sleep(60);continue
                attempt+=1
                command=[sys.executable,'-I','-B',str(Path(__file__).resolve().parents[2]/'scripts/experiments/run_bace_globalgce_aplus_evaluation.py'),
                    '--config',spec['runtime_config'],'--spec',str(evaluation_spec_path),'--action',action]
                with (root/f'{action}-{attempt:04d}.log').open('xb') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,close_fds=True,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                    while child.poll() is None:
                        heartbeat('CPU_SCIENCE_RUNNING',next_stage=action,science_pid=child.pid)
                        try:child.wait(timeout=60)
                        except subprocess.TimeoutExpired:pass
                if child.returncode==75:
                    heartbeat('PAUSED_AT_PARENT_CHECKPOINT',next_stage=action);time.sleep(60);continue
                if child.returncode:raise RuntimeError(f'{action} failed code={child.returncode}; saved parents retained; no automatic engineering retry')
                break
        final=read_small(Path(spec['output_root'])/'final_audit.json')[0]
        if final.get('state')!='APLUS_GLOBALGCE_EVALUATION_COMPLETE' or final.get('spec_sha256')!=stable_sha256(spec):
            raise ValueError('ACTUAL_FINAL_AUDIT_REQUIRED')
        atomic_json(root/'terminal.json',{'state':'APLUS_GLOBALGCE_EVALUATION_COMPLETE','final_audit':str(Path(spec['output_root'])/'final_audit.json'),
            'main_matrix_write':False,'created_at':utc_now()});heartbeat('APLUS_GLOBALGCE_EVALUATION_COMPLETE')
        return 0
    except BaseException as error:
        atomic_json(root/'terminal.json',{'state':'BLOCKED','error':repr(error),'created_at':utc_now()});raise
