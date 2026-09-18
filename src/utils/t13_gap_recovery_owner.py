"""T13 GPU1 compact recovery adapter for the existing owned-child mechanism.

No new lock or registry. Keep the canonical T13 reservation, reject any live
predecessor, and sample actual cgroup/NVML/filesystems before each boundary.
"""
from datetime import datetime, timezone
import json, os, time
from pathlib import Path

GIB=1024**3
SCHEMA='T13_GAP_FIRST_COMPACT_GPU_PROBE_20260914'
FORMAL_SCHEMA='T13_GAP_FIRST_SAME_RUN_CONTINUATION_20260914'

def stage_memory_policy(spec):
    """V5 replaces the unidentified old 384GiB constant, not real reservations."""
    if 'stage_resource_policy' not in spec:
        if spec['process_peak_bound_bytes']!=16*GIB or spec['other_remaining_reserve_bytes']!=384*GIB:
            raise ValueError('T13_BOUNDED_PROBE_MEMORY_CONTRACT')
        return None
    from src.utils.t13_performance_dispatch import bound_json
    p=bound_json(spec['stage_resource_policy'])
    if (spec['schema']!=FORMAL_SCHEMA or p.get('schema')!='T13_V5_STAGE_INCREMENT_V1'
            or p.get('formal_quota')!='1/1' or p.get('gpu_uuid')!=spec['gpu_uuid']
            or p.get('safety_margin_bytes')!=64*GIB or p.get('process_peak_bound_bytes')!=32*GIB
            or p.get('maximum_retained_train_batches')!=5):
        raise ValueError('T13_V5_POLICY_SCOPE')
    memory=bound_json(p['adopted_probe_memory'])['samples']
    peak=max(int(r['VmHWM_bytes']) for r in memory)
    if peak!=p['observed_probe_host_peak_bytes'] or 5*peak>p['process_peak_bound_bytes']:
        raise ValueError('T13_V5_BOUNDED_FIVE_BATCH_ENVELOPE_UNSUPPORTED')
    if p.get('retired_default_reason')!='UNIDENTIFIED_FIXED_HEADROOM_NOT_EXTERNAL_TASK_INCREMENT':
        raise ValueError('T13_384_RETIREMENT_REASON_REQUIRED')
    for row in p['concurrent_future_increments']:
        if type(row.get('additional_bytes')) is not int or row['additional_bytes']<0 or not row.get('evidence'):
            raise ValueError('UNKNOWN_CONCURRENT_INCREMENT:'+str(row.get('task_id')))
    return p

def decision(spec, evidence):
    blockers=list(evidence.get('source_blockers', []))
    if spec.get('schema') not in {SCHEMA,FORMAL_SCHEMA} or spec.get('gpu_index')!=1:
        blockers.append('WRONG_T13_GPU1_SCOPE')
    if not evidence.get('memory_safe'): blockers.append('T13_INCREMENTAL_MEMORY_RESERVE')
    if not evidence.get('storage_safe'): blockers.append('T13_COMPACT_STORAGE_BOUND')
    if not evidence.get('physical_gpu_safe'): blockers.append('T13_GPU1_ACTUAL_OCCUPANCY')
    return dict(allowed=not blockers,blockers=blockers,science_started=False)

def resource_wait_seconds(spec, now=None):
    """Finite existing-owner wait, never beyond the sealed dispatch cutoff."""
    requested=spec.get('resource_wait_seconds',0)
    if type(requested) is not int or not 0<=requested<=86400:
        raise ValueError('T13_FINITE_RESOURCE_WAIT_REQUIRED')
    if not requested:return 0
    now=now or datetime.now(timezone.utc)
    cutoff=datetime.fromisoformat(spec['science_dispatch_cutoff_utc'])
    remaining=int((cutoff-now).total_seconds())
    if remaining<=0:raise ValueError('T13_DISPATCH_CUTOFF_REACHED')
    return min(requested,remaining)

class RecoverySampler:
    task_family='t13_performance_diagnostic'
    def __init__(self,spec):
        if spec['schema'] not in {SCHEMA,FORMAL_SCHEMA} or spec['gpu_index']!=1 or spec['formal_quota_used']!='1/1':
            raise ValueError('T13_RECOVERY_SCOPE')
        self.stage_policy=stage_memory_policy(spec)
        self.t13_dispatch=spec;self.config={'proc_root':'/proc'};self.uuid=spec['gpu_uuid'];self.index=1
        self.idle_since=None;self.admitted_idle_seconds=None
    def bind_t13_held_lease(self,fd,run_id):
        if run_id!=self.t13_dispatch['task_id']:raise ValueError('T13_TASK_BINDING')
        if self.t13_dispatch.get('v7_registry_binding_root'):
            from src.utils.t13_v7_binding import registry_claim
            registry_claim(self.t13_dispatch['v7_registry_binding_root'],held_fd=fd)
    def sample(self,*,child_pid=None,child_start_ticks=None):
        from src.utils.autodl_runtime import query_gpu_inventory
        from src.utils.final16_owner_registry_v1 import process_start_ticks,validate_owner_registry
        spec=self.t13_dispatch;block=[]
        if spec.get('science_dispatch_cutoff_utc') and not child_pid:
            if datetime.now(timezone.utc)>=datetime.fromisoformat(spec['science_dispatch_cutoff_utc']):
                block.append('T13_DISPATCH_CUTOFF_REACHED')
        reg=validate_owner_registry(json.loads(Path(spec['registry']).read_text()),check_processes=False)
        matches=[r for r in reg['tasks'] if r['task_id']==spec['canonical_task_id']]
        if len(matches)!=1 or matches[0]['gpu']!=1 or matches[0]['method']!='GlobalGCE':
            raise ValueError('CANONICAL_T13_RESERVATION_CHANGED')
        for r in reg['tasks']:
            if r['gpu']!=1:continue
            pid=r.get('owner_pid')
            if pid and pid!=os.getpid() and process_start_ticks('/proc',pid)==r.get('owner_start_ticks'):
                block.append('LIVE_T13_PREDECESSOR:'+str(pid))
        leases=[r for r in reg['gpu_leases'] if r['gpu']==1 and r['state']!='RELEASED']
        if any(r['task_id']!=spec['canonical_task_id'] for r in leases) or not leases:
            block.append('CANONICAL_T13_LEASE_SCOPE')
        inventory=query_gpu_inventory();gpu=next(g for g in inventory if g.uuid==self.uuid and g.index==1)
        physical=all(p.pid==child_pid for p in gpu.processes)
        if not child_pid:physical=physical and gpu.memory_free_mb>=70000
        cg=Path('/sys/fs/cgroup/memory');usage=int((cg/'memory.usage_in_bytes').read_text());limit=int((cg/'memory.limit_in_bytes').read_text())
        rss=0
        if child_pid:
            if process_start_ticks('/proc',child_pid)!=child_start_ticks:raise ValueError('CHILD_IDENTITY_CHANGED')
            lines=Path(f'/proc/{child_pid}/status').read_text().splitlines()
            rss=next(int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:'))
        policy=self.stage_policy
        if policy:
            from src.utils.t13_performance_dispatch import bound_json
            # Actual window evidence is separate from occupancy and cannot be a
            # renewed timestamp on the old snapshot. Missing confirmation blocks.
            window=bound_json(policy['resource_window'])
            now=datetime.now(timezone.utc)
            if (window.get('state')!='CONFIRMED_EXCLUSIVE_WINDOW'
                    or window.get('gpu_uuid')!=self.uuid or not window.get('authority_evidence')
                    or now>=datetime.fromisoformat(window['ends_at'])
                    or window.get('future_reservations_complete') is not True):
                block.append('LAWFUL_GPU_AND_FUTURE_RESOURCE_WINDOW_UNCONFIRMED')
            bound=policy['process_peak_bound_bytes']
            other=sum(r['additional_bytes'] for r in policy['concurrent_future_increments'])
            safety=policy['safety_margin_bytes']
            need=max(0,bound-rss)+other+safety
            from src.ablations.llm.existing_gpu_owner import memory_headroom
            effective=memory_headroom(Path('/proc'),cg)
        else:
            bound=spec['process_peak_bound_bytes'];other=spec['other_remaining_reserve_bytes'];safety=0
            need=other+max(0,bound-rss);effective=limit-usage
        memory=effective>=need and rss<=bound
        p=os.statvfs(spec['persistent_root']);n=os.statvfs(spec['nvme_root'])
        storage=(p.f_favail-spec['persistent_uncreated_peak']-spec['other_uncreated_peak']>=8192+spec['dynamic_buffer']
            and p.f_bavail*p.f_frsize>=100*GIB and n.f_bavail*n.f_frsize>=2*GIB+spec['nvme_uncreated_peak'])
        e=dict(task_family=self.task_family,observed_at=datetime.now(timezone.utc).isoformat(),source_blockers=block,
            memory_safe=memory,storage_safe=storage,physical_gpu_safe=physical,
            memory_headroom_bytes=effective,required_headroom_bytes=need,cgroup_usage_bytes=usage,
            stage_process_bound_bytes=bound,other_future_increment_bytes=other,safety_margin_bytes=safety,
            stage_policy_sha256=spec.get('stage_resource_policy',{}).get('sha256'),
            child_rss_bytes=rss,gpu_index=1,gpu_uuid=self.uuid,target_gpu_uuid=self.uuid,logical_device='cuda:0',
            actual_gpu_observation=gpu.as_json(),gpu_idle_seconds=0,
            persistent_free_entries=p.f_favail,nvme_free_bytes=n.f_bavail*n.f_frsize,
            plan_sha256=spec['plan_sha256'],task_id=spec['task_id'],registry=spec['registry'])
        e['t13_admission']=decision(spec,e);e['pause_requested']=not e['t13_admission']['allowed']
        return e

def run(spec_path):
    from src.ablations.llm.existing_gpu_owner import run_owned_child
    from src.utils.autodl_runtime import sanitized_environment
    from src.eval.bace_frozen_gnn_contracts import sha256_file
    spec=json.loads(Path(spec_path).read_text())
    if sha256_file(spec['plan_path'])!=spec['plan_sha256']:raise ValueError('T13_PLAN_CHANGED')
    env=dict(sanitized_environment());backend=json.loads(Path(spec['backend_receipt']).read_text())
    for k,v in backend['thread_environment'].items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',TMPDIR=spec['nvme_root'])
    return run_owned_child(command=spec['science_command_without_owner_fds'],environment=env,
        sampler=RecoverySampler(spec),output_root=spec['owner_root'],lock_root=spec['lock_root'],
        run_id=spec['task_id'],interval=60,max_wait_seconds=resource_wait_seconds(spec))
