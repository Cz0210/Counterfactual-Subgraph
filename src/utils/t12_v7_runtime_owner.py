"""Narrow runtime FD/provider binding for the existing reviewed T12 stage.

The reviewed four scientific files, checkpoint and diagnostic budget are not
changed. This owner creates its identity only after acquiring the existing
lease, then seals a child spec; no pretend future PID or descriptor is stored.
"""
import copy
import fcntl
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.utils.main_ready_task_specs import atomic_json, load_spec, owner_command_sha256, stable_sha256
from src.utils.final16_owner_registry_v1 import (process_start_ticks, validate_owner_registry,
    build_owner_registry, atomic_write_owner_registry)
from src.utils.t13_v7_binding import CUTOFF, DEADLINE, GIB


def read(path):
    return json.loads(Path(path).read_text())


def provider(root):
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    from src.utils.autodl_runtime import query_gpu_inventory
    root=Path(root); policy=read(root/'resource_policy.json'); runtime=read(root/'runtime_identity.json')
    gpu=next(g for g in query_gpu_inventory() if g.uuid==runtime['gpu_uuid'])
    free=os.statvfs(root); headroom=memory_headroom(Path('/proc'),Path('/sys/fs/cgroup/memory'))
    # The original T12 descriptor's 64-GiB minimum is retained as its admitted
    # stage increment, not represented as a measured full-trajectory peak.
    b=policy['original_t12_minimum_increment_bytes']; r=policy['t13_maximum_remaining_increment_bytes']; s=64*GIB
    need=b+r+s
    blockers=[]
    if headroom<need:blockers.append('JOINT_RAM_ADMISSION')
    if any(p.pid!=runtime.get('child_pid') for p in gpu.processes):blockers.append('UNEXPECTED_GPU_PROCESS')
    required_slots=8192+policy['dynamic_buffer']+policy['t12_new_entries']+policy['t13_pending_entries']
    if free.f_favail<required_slots:blockers.append('JOINT_FILE_SLOTS')
    if free.f_bavail*free.f_frsize<policy['persistent_available_minimum_bytes']:blockers.append('PERSISTENT_BYTES')
    return dict(actual_resources_resampled=True, measured_at_unix_seconds=time.time(),
        stage_id='reference_reload_501_510',gpu_uuid=runtime['gpu_uuid'],allowed=not blockers,
        blockers=blockers,B_bytes=b,R_bytes=r,S_bytes=s,required_headroom_bytes=need,
        observed_headroom_bytes=headroom,available_file_slots=free.f_favail,required_file_slots=required_slots,
        stage_budget_basis=policy['basis'],policy_sha256=stable_sha256(policy))


def registry_update(path, spec, *, finish=False):
    path=Path(path);before=path.read_bytes();reg=validate_owner_registry(json.loads(before),check_processes=False)
    lock=Path(reg['matrix_authority_root'])/'publish.lock'
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.read_bytes()!=before:raise ValueError('T12_REGISTRY_CAS_CHANGED')
        targets=[x for x in reg['tasks'] if x['gpu']==3 and x['method']=='GCFExplainer']
        if len(targets)!=1:raise ValueError('T12_CANONICAL_TARGET_NOT_UNIQUE')
        target=targets[0]
        for row in reg['tasks']:
            pid=row.get('owner_pid')
            if row['gpu']==3 and pid and pid!=os.getpid() and process_start_ticks('/proc',pid)==row.get('owner_start_ticks'):
                raise ValueError('T12_LIVE_CANONICAL_OWNER')
        if finish and target.get('owner_pid')!=os.getpid():raise ValueError('T12_FINISH_OWNER_CHANGED')
        target.update(owner_state='BLOCKED' if finish else 'RUNNING',owner_pid=None if finish else os.getpid(),
            owner_start_ticks=None if finish else process_start_ticks('/proc',os.getpid()),
            heartbeat=spec['expected_heartbeat_path'],output_root=spec['output_root'],
            stage='V7_STAGE_EXITED_CHECK_TERMINAL' if finish else 'V7_RESTORE500_THEN501_510',
            task_spec_sha=spec['spec_sha256'],execution_commit=spec['execution_commit'])
        leases=[x for x in reg['gpu_leases'] if x['task_id']==target['task_id']]
        if len(leases)!=1 or leases[0]['lease_path']!=spec['gpu_request']['lease_path']:
            raise ValueError('T12_ORIGINAL_LEASE_PATH_MISMATCH')
        leases[0]['state']='RELEASED' if finish else 'HELD'
        updated=build_owner_registry(registry_id=reg['registry_id'],matrix_authority_root=reg['matrix_authority_root'],
            tasks=reg['tasks'],publishers=reg['publishers'],gpu_leases=reg['gpu_leases'],check_processes=False)
        atomic_write_owner_registry(path,updated)


def owner(*, template, root, registry, code_root, observer_receipt=None):
    root=Path(root); code_root=Path(code_root)
    if datetime.now(timezone.utc)>=datetime.fromisoformat(CUTOFF):raise ValueError('V7_CUTOFF')
    template=Path(template)
    old=load_spec(template)
    if observer_receipt is not None:
        from src.utils.t12_real_regression_v9 import verify
        receipt=Path(observer_receipt)
        verified=verify(receipt.parent,publish=False)
        if read(receipt)!=verified:raise ValueError('T12_REAL_OBSERVER_RECEIPT_CHANGED')
        if receipt.name!='real-adapter-regression.json' or verified['model_input_hashes']!=old['input_hashes']:
            raise ValueError('T12_REAL_OBSERVER_INPUT_BINDING_CHANGED')
    old_stage=Path(old['output_root'])/'shadow-ledger/reference_reload_501_510/attempt.json'
    if old_stage.exists():raise ValueError('PREVIOUS_STAGE_BUDGET_REQUIRES_RECONCILIATION:'+str(old_stage))
    root.mkdir(parents=True,exist_ok=False)
    state=dict(owner_pid=os.getpid(),owner_start_ticks=process_start_ticks('/proc',os.getpid()),
        gpu_uuid=old['gpu_request']['uuid'],child_pid=None,phase='BEFORE_EXISTING_LEASE')
    atomic_json(root/'runtime_identity.json',state)
    minimum=int(old['memory_request']['minimum_parent_headroom_bytes'])
    policy=dict(schema='T12_V7_EXISTING_STAGE_JOINT_ADMISSION',
        original_t12_minimum_increment_bytes=minimum,t13_maximum_remaining_increment_bytes=32*GIB,
        safety_margin_bytes=64*GIB,external_future_reserved_bytes=0,
        t12_new_entries=256,t13_pending_entries=256,dynamic_buffer=256,
        persistent_available_minimum_bytes=512*GIB,deadline=DEADLINE,science_cutoff=CUTOFF,
        basis='Retain original T12 stage memory request; compact append journals and ten-step checkpoint/raw ledger, not per-candidate files; reserve T13 full32GiB concurrently.',
        full_epoch_peak_measured=False,source_template=str(template))
    atomic_json(root/'resource_policy.json',policy)
    evidence=provider(root);atomic_json(root/'admission.json',evidence)
    if not evidence['allowed']:raise ValueError('T12_ADMISSION:'+','.join(evidence['blockers']))
    lease=Path(old['gpu_request']['lease_path'])
    if lease.is_symlink() or not lease.is_file():raise ValueError('EXISTING_T12_LEASE_REQUIRED')
    claimed=False;stop=threading.Event();thread=None
    with lease.open('a+b') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=copy.deepcopy(old);spec['output_root']=str(root/'science')
        # Distinguish the current owner wrapper from the separately reviewed
        # scientific source tree. Neither commit is relabelled as the other.
        spec['repo_root']=str(code_root)
        spec['execution_commit']=subprocess.check_output(['git','-C',str(code_root),'rev-parse','HEAD'],text=True).strip()
        spec['entrypoint']=str(code_root/'scripts/run_t12_v7_child.py')
        spec['config_path']=str(code_root/'configs/hpc.yaml')
        from src.utils.main_ready_task_specs import file_sha256
        spec['config_sha256']=file_sha256(Path(spec['config_path']))
        spec['science_contract']['v7_wrapper_binding']=dict(template=str(template),
            template_sha256=file_sha256(template),source_base_binding=str(template.parent/'reviewed_source_base.json'),
            source_base_binding_sha256=file_sha256(template.parent/'reviewed_source_base.json'))
        spec['expected_heartbeat_path']=str(root/'heartbeat.json');spec['expected_pid_file']=str(root/'runtime_identity.json')
        if 'expected_terminal_path' in spec:spec['expected_terminal_path']=str(root/'terminal.json')
        binding=spec['science_contract']['shadow_binding']
        if observer_receipt is not None:
            binding['observer_regression_receipt']=str(receipt)
            binding['observer_regression_receipt_sha256']=file_sha256(receipt)
        binding['owner_identity']=dict(pid=os.getpid(),start_ticks=state['owner_start_ticks'])
        binding['resource_provider_command']=[spec['python'],'-I','-B',str(code_root/'scripts/run_t12_v7_owner.py'),
            '--config',str(code_root/'configs/hpc.yaml'),'--action','provider','--root',str(root)]
        spec['science_contract']['disposable_index_root']=str(root/'science/disposable-history-index')
        spec_path=root/'runtime-task-spec.json'
        spec['arguments']=[str(spec_path) if x==str(template) else spec['config_path'] if x==old['config_path'] else x for x in spec['arguments']]
        spec['expected_owner_command_sha256']=owner_command_sha256(spec)
        spec.pop('spec_sha256',None);spec['spec_sha256']=stable_sha256(spec)
        atomic_json(spec_path,spec);load_spec(spec_path)
        try:
            registry_update(registry,spec);claimed=True
            state['phase']='RESTORE500_CHILD_STARTING'
            atomic_json(root/'runtime_identity.json',state)
            environment=dict(os.environ,**spec['required_environment'])
            environment.update(CUDA_VISIBLE_DEVICES=state['gpu_uuid'],T12_OWNER_HELD_GPU_FD=str(held.fileno()))
            from src.utils.main_ready_task_specs import command_from_spec
            with (root/'science.log').open('xb') as log:
                child=subprocess.Popen(command_from_spec(spec),cwd=spec['repo_root'],env=environment,
                    pass_fds=(held.fileno(),),stdout=log,stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid,child_start_ticks=process_start_ticks('/proc',child.pid),phase='CHILD_COMPLETE_RESTORE_PENDING')
                atomic_json(root/'runtime_identity.json',state)
                def heartbeat():
                    while not stop.is_set():
                        atomic_json(root/'heartbeat.json',dict(**state,observed_at=datetime.now(timezone.utc).isoformat(),
                            output_root=spec['output_root'],task_id=spec['task_id'],gpu_lock_held=True))
                        stop.wait(60)
                thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
                rc=child.wait()
            state.update(phase='CHILD_EXITED',returncode=rc)
            atomic_json(root/'terminal.json',dict(**state,science_pass_claimed=False,
                next_stage='REQUIRED_RAW_PARITY_THEN_EXISTING_ACTIVATION' if rc==0 else 'READ_FIRST_RUNTIME_FAILURE',
                diagnostic_checkpoint_promotable=False))
            return rc
        finally:
            stop.set()
            if thread:thread.join(timeout=2)
            if claimed:registry_update(registry,spec,finish=True)
