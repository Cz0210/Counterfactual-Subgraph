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
    contract, _ = read_small(descriptor['search_contract'])
    if sha256_file(descriptor['search_contract']) != descriptor['search_contract_sha256']:
        raise ValueError('OURS_TRAIN_PREDECESSOR_CONTRACT_CHANGED')
    path = Path(descriptor['candidate_freeze'])
    if not path.exists():
        return False
    value, _ = read_small(path)
    body = {k:v for k,v in value.items() if k != 'self_sha256'}
    if (stable_sha256(body) != value.get('self_sha256') or
        value.get('state') != 'TRAIN_ONLY_POOL_FROZEN' or
        value.get('search_contract_sha256') != contract.get('self_sha256') or
        value.get('proposal_source') != 'OURS_MAIN_PPO_66' or value.get('test_opened') is not False):
        raise ValueError('OURS_TRAIN_PREDECESSOR_NOT_VALID_FROZEN_POOL')
    # This is a stage receipt, not a GPU lease. ResourceSampler subsequently
    # proves the child has exited and both original exclusive locks are free.
    return True


def training_command(spec, config, stage, *, resume):
    action = 'train-canary' if stage == 'canary' else 'train'
    root = config['gpu_canary_root'] if stage == 'canary' else config['formal_output_root']
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve().parents[2] / 'scripts/run_bace_globalgce_chemaligned.py'),
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
    if spec.get('schema') != 'bace_chemaligned_existing_owner_stages_v1':
        raise ValueError('EXPLICIT_CHEMALIGNED_STAGE_SPEC_REQUIRED')
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
                while not predecessor_ready(spec['ours_train_predecessor']):
                    heartbeat('WAITING_OURS_TRAIN_GPU_PREDECESSOR', next_stage=stage,
                        resource_lock_acquired=False, science_started=False)
                    time.sleep(60)
                resume=(output/'latest.pt').is_file()
                if output.exists() and not resume:
                    raise ValueError('PARTIAL_STAGE_WITHOUT_LEGAL_CHECKPOINT:'+str(output))
                attempt+=1
                sampler=ResourceSampler(resource,0,spec['gpu_uuid'],task_family='globalgce_chemaligned',
                    reach_contract={'path':spec['training_contract'],'sha256':spec['training_contract_file_sha256']})
                child_owner=root/f'{stage}-lease-{attempt:04d}'
                heartbeat('WAITING_EXISTING_EXCLUSIVE_GPU_LEASE', next_stage=stage, child_owner=str(child_owner))
                code=run_owned_child(command=training_command(spec,config,stage,resume=resume),
                    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    sampler=sampler,output_root=child_owner,lock_root=resource['gpu_lock_root'],
                    run_id='bace-chemaligned-v2-'+root.name+'-'+stage, interval=60,max_wait_seconds=86400)
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
        final_state='CPU_EVALUATION_COMPLETE_PUBLICATION_PENDING' if spec['cpu_successors'] else 'BLOCKED_CPU_SUCCESSORS_NOT_BOUND'
        atomic_json(root/'terminal.json', {'state':final_state,'created_at':utc_now(),'main_matrix_written':False})
        heartbeat(final_state)
        return 0 if spec['cpu_successors'] else 2
    except BaseException as error:
        atomic_json(root/'terminal.json', {'state':'FAILED_ENGINEERING','error':repr(error),'created_at':utc_now()})
        raise
