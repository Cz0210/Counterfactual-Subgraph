"""One-shot T14 terminal CAS and Global/AIDS bindings; no new waiting service."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, utc_now
from src.utils.final16_owner_registry_v1 import validate_owner_registry, atomic_write_owner_registry, process_start_ticks
from src.utils.stage_file_policy import canonical_sha, load_stage_policy, config_file_admission
from src.utils.terminal_resource_dependency import verify_terminal_dependency

R = Path('/autodl-fs/data/counterfactual-subgraph-runtime')
REG = R/'control/final16-owner-registry/current.json'
G = R/'control/bace-gin-aplus-globalgce-41b7ecb-20260908'
A = R/'outputs/autodl/recovery/aids-rfaligned-v2-20260907/resource-925ed27f'
TID = 't14-route-c-59f101cd-f30b-458d-aa8c-2eb93ae82609'
T = R/'control/t14_route_c/owners/route-c-59f101cd-f30b-458d-aa8c-2eb93ae82609'


def read(path):
    return json.loads(Path(path).read_text())


def descriptor(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


def locked(path, stack):
    stream = stack.enter_context(open(path, 'r+'))
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def assert_no_writer(roots):
    writers = []
    for directory in Path('/proc').glob('[0-9]*'):
        for fd in (directory/'fd').glob('*'):
            try:
                target = os.readlink(fd)
                if not any(target == str(root) or target.startswith(str(root)+'/') for root in roots):
                    continue
                info = dict(line.split(':', 1) for line in (directory/'fdinfo'/fd.name).read_text().splitlines())
                if int(info['flags'].strip(), 8) & os.O_ACCMODE:
                    writers.append({'pid': int(directory.name), 'fd': fd.name, 'path': target})
            except (OSError, ValueError):
                continue
    if writers:
        raise ValueError('ACTIVE_WRITER:'+json.dumps(writers))
    return writers


def seal_policy(path, policy):
    policy.pop('self_sha256', None)
    policy['self_sha256'] = canonical_sha(policy)
    atomic_json(path, policy)
    load_stage_policy(descriptor(path), R)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'authorization.json', {
        'source': 'CODEX_COMPLETE_RECOVERY.md/user explicit execution authorization 20260908',
        'allow_t14_terminal_reconciliation': True, 'allow_global_waiting_owner_handover': True,
        'allow_aids_dynamic_resource_rebind': True, 'preserve_aids_child_and_pairs': True,
        'global_resume_same_epoch60_campaign': True, 'no_new_fresh_campaign': True,
        'allow_stage_based_inode_policy': True, 'inode_base_reserve':20000,
        'inode_next_stage_peak_factor':2, 'contact_support_first':False,
        'created_at':utc_now()})
    original = read(REG); expected = sha256_file(REG)
    validate_owner_registry(original, check_processes=False)
    row = next(r for r in original['tasks'] if r['task_id'] == TID)
    terminal = read(T/'terminal.json')
    if terminal.get('status') != 'FAILED' or terminal.get('owner_pid') != row['owner_pid']:
        raise ValueError('T14_TERMINAL_OWNER_BINDING_FAILED')
    if row['owner_pid'] != 326543 or row['owner_start_ticks'] != 39926973:
        raise ValueError('T14_OWNER_CHANGED_REAUDIT')
    if any(Path('/proc',str(pid)).exists() for pid in (326543,423689,423777)):
        raise ValueError('OLD_T14_PROCESS_PRESENT_REAUDIT_IDENTITY')
    assert_no_writer([T, Path(row['output_root'])])
    leases = [r for r in original['gpu_leases'] if r['task_id'] == TID]
    if len(leases) != 1: raise ValueError('T14_LEASE_AMBIGUOUS')
    publication_lock = Path(original['matrix_authority_root'])/'publish.lock'
    with ExitStack() as stack:
        locked(publication_lock, stack)
        lease = locked(Path(leases[0]['lease_path']), stack)
        if sha256_file(REG) != expected: raise ValueError('REGISTRY_CAS_CHANGED')
        atomic_json(out/'registry_before.json', original)
        updated = copy.deepcopy(original)
        target = next(r for r in updated['tasks'] if r['task_id'] == TID)
        target.update(owner_state='BLOCKED', stage='FAILED_PARITY_WAITING_COMPONENT_EVIDENCE',
                      owner_pid=None, owner_start_ticks=None)
        next(r for r in updated['gpu_leases'] if r['task_id'] == TID)['state'] = 'RELEASED'
        updated['updated_at'] = utc_now(); updated.pop('self_sha256')
        updated['self_sha256'] = canonical_sha(updated)
        validate_owner_registry(updated, check_processes=False)
        info = os.fstat(lease.fileno())
        receipt = {'schema':'t14_terminal_resource_release_v1','task_id':TID,
            'scientific_status':'FAILED','future_stage':'WAITING_PARITY',
            'science_dependency_for_global_aids':False,'physical_lease_released':True,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'retired_processes':[{'pid':326543,'start_ticks':39926973},
                {'pid':423689,'start_ticks':49996103},{'pid':423777,'start_ticks':50002141}],
            'old_science_pid_absent':423689, 'writer_set':[],
            'terminal_path':str(T/'terminal.json'),'terminal_sha256':sha256_file(T/'terminal.json'),
            'lease_path':leases[0]['lease_path'],'lease_device_inode':[info.st_dev,info.st_ino],
            'registry_before_sha256':expected,'registry_after_self_sha256':updated['self_sha256'],
            'authorization':descriptor(out/'authorization.json'),'created_at':utc_now()}
        atomic_json(out/'t14_terminal_release.json',receipt)
        atomic_write_owner_registry(REG, updated)
    verify_terminal_dependency(descriptor(out/'t14_terminal_release.json'), read(REG), Path('/proc'))
    atomic_json(out/'future_resources.json', {'t14_diagnostic':{'state':'PENDING_EVIDENCE','max_transitions':64},
        't14_formal':{'state':'WAITING_PARITY','peak':None},
        't12_shadows':{'state':'PENDING_EVIDENCE','max_transitions':540},
        't13_performance':{'state':'PENDING_EVIDENCE','max_train_batches':2,'max_validation_batches':1},
        'unknown_peaks_are_not_zero':True,'must_rebind_before_activation':True})
    gp, ap = read(G/'policy.json'), read(A/'policy.json')
    live = [r for r in gp['concurrent_components'] if r['component_id'] in {'t13_next_checkpoint','t12_next_checkpoint'}]
    # Registry-CAS scope changes force an explicit next-stage re-admission.
    for component in live:
        component.setdefault('scope_guards', []).append({'path':str(REG),
            'allowed_values':{'self_sha256':[updated['self_sha256']]}})
    aids_stages = [d for d in ap['stages'].values() if d.get('state') == 'BOUNDED']
    aids_components = aids_stages[0]['components']
    if len(aids_components) != 1 or aids_components[0]['peak_new_files'] != 4096:
        raise ValueError('AIDS_EXISTING_BOUND_CHANGED')
    global_component = gp['stages']['llm_gpu_generation']['components'][0]
    for policy, name, concurrent in ((gp,'global-policy.json',live+aids_components),
                                     (ap,'aids-policy.json',live+[global_component])):
        policy['authorization'] = descriptor(out/'authorization.json')
        policy['concurrent_components'] = copy.deepcopy(concurrent)
        policy['terminal_resource_release'] = descriptor(out/'t14_terminal_release.json')
        policy['future_resource_requirements'] = descriptor(out/'future_resources.json')
        seal_policy(out/name, policy)
    gc = read(G/'resource_config.json'); ac = read(A/'resource_config.json')
    atomic_json(out/'global-resource-before.json',gc); atomic_json(out/'aids-resource-before.json',ac)
    gc['stage_file_policy'] = descriptor(out/'global-policy.json')
    gc['terminal_resource_dependencies'] = [descriptor(out/'t14_terminal_release.json')]
    ac['stage_file_policy'] = descriptor(out/'aids-policy.json')
    atomic_json(out/'global-resource.json',gc); atomic_json(out/'aids-resource.json',ac)
    atomic_json(out/'prepared.json', {'state':'PREPARED','existing_publication_lock':str(publication_lock),
        'aids_previous_resource_file_sha256':sha256_file(A/'resource_config.json'), 'created_at':utc_now()})
    print(json.dumps(read(out/'prepared.json')))


def activate_aids(out):
    prepared = read(out/'prepared.json')
    if process_start_ticks(Path('/proc'),451323) != 54975873:
        raise ValueError('AIDS_CHILD_IDENTITY_CHANGED')
    config = read(out/'aids-resource.json')
    policy = load_stage_policy(config['stage_file_policy'],R)
    admissions = {stage:config_file_admission(config,os.statvfs(R).f_favail,stage_id=stage,policy=policy)
                  for stage,d in policy['stages'].items() if d['state']=='BOUNDED'}
    if not all(v['admitted'] for v in admissions.values()): raise ValueError(json.dumps(admissions))
    with ExitStack() as stack:
        locked(Path(prepared['existing_publication_lock']),stack)
        if sha256_file(A/'resource_config.json') != prepared['aids_previous_resource_file_sha256']:
            raise ValueError('AIDS_DYNAMIC_CONFIG_CAS_CHANGED')
        atomic_json(A/'resource_config.json',config)
    atomic_json(out/'aids_dynamic_rebind.json',{'state':'DYNAMIC_POINTER_REBOUND','child_pid':451323,
        'child_start_ticks':54975873,'old_config':descriptor(out/'aids-resource-before.json'),
        'new_config':descriptor(A/'resource_config.json'),'inode_admissions':admissions,
        'memory_contract_unchanged':True,'science_restarted':False,'created_at':utc_now()})
    print(json.dumps(read(out/'aids_dynamic_rebind.json')))


def handover_global(out):
    from src.ablations.llm.existing_gpu_owner import ResourceSampler
    from src.ablations.gnn.early_policy import gpu_allowed
    spec = read(G/'owner_spec.json'); resource = read(out/'global-resource.json')
    sampler = ResourceSampler(resource,0,spec['gpu_uuid'],task_family='globalgce_aplus',
        reach_contract={'path':spec['training_contract'],'sha256':spec['training_contract_file_sha256']})
    evidence = sampler.sample(); decision = gpu_allowed({**evidence,'gnn_core_seed7_audit':'PASS'},family='globalgce_aplus')
    atomic_json(out/'global_activation_preflight.json',{'evidence':evidence,'decision':decision})
    if not decision['allowed']: raise ValueError(json.dumps(decision))
    owner = G/'owner'; old = read(owner/'owner.json')
    if old['pid'] != 445964 or old['start_ticks'] != 53981973 or process_start_ticks(Path('/proc'),445964) != 53981973:
        raise ValueError('GLOBAL_WAITING_OWNER_IDENTITY_CHANGED')
    cmd = Path('/proc/445964/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(G/'run_bound_owner.py') not in cmd or str(G/'owner_spec.json') not in cmd:
        raise ValueError('GLOBAL_WAITING_OWNER_COMMAND_CHANGED')
    children = [p for p in Path('/proc/445964/task').glob('*/children') if p.read_text().strip()]
    if children: raise ValueError('GLOBAL_WAITING_OWNER_HAS_CHILD')
    hb = read(owner/'heartbeat.json')
    if hb['state'] != 'WAITING_EXISTING_EXCLUSIVE_GPU_LEASE': raise ValueError('GLOBAL_OWNER_NOT_WAITING')
    assert_no_writer([Path(read(spec['training_contract'])['formal_output_root'])])
    ledger = read(G/'formal_campaign_ledger.json')
    if ledger['fresh_campaigns_used'] != 1 or ledger['max_fresh_campaigns'] != 1:
        raise ValueError('GLOBAL_FORMAL_QUOTA_CHANGED')
    atomic_json(out/'global_owner_before.json',old); atomic_json(out/'global_heartbeat_before.json',hb)
    old_sha = sha256_file(owner/'owner.json')
    with ExitStack() as stack:
        locked(Path(resource['gpu_lock_root'])/('gpu-'+spec['gpu_uuid']+'.lock'),stack)
        os.kill(445964,signal.SIGTERM)
    deadline=time.monotonic()+75
    boundary_signal_sent=False
    while process_start_ticks(Path('/proc'),445964)==53981973 and time.monotonic()<deadline:
        # The old wrapper treats inner lease return75 as retry. Only its
        # explicitly child-free outer sleep has restored the default handler.
        current=read(owner/'heartbeat.json')
        if (not boundary_signal_sent and current.get('state')=='PAUSED_AT_COMMITTED_OPTIMIZER_BOUNDARY'
                and not any(p.read_text().strip() for p in Path('/proc/445964/task').glob('*/children'))):
            os.kill(445964,signal.SIGTERM)
            boundary_signal_sent=True
        time.sleep(1)
    if process_start_ticks(Path('/proc'),445964)==53981973: raise ValueError('GLOBAL_WAITING_OWNER_DID_NOT_EXIT_NO_FORCE')
    atomic_json(out/'global_owner_exit.json',{'state':'OLD_WAITING_OWNER_EXIT_CONFIRMED','pid':445964,
        'start_ticks':53981973,'signal':'SIGTERM','outer_wait_boundary_sigterm':boundary_signal_sent,
        'science_child_signaled':False,'created_at':utc_now()})
    spec['resource_config']=str(out/'global-resource.json')
    spec['science_entrypoint']=str(R/'code/globalgce-aplus-41b7ecb-overlay-20260908/scripts/run_bace_globalgce_aplus.py')
    spec['science_entrypoint_sha256']=sha256_file(spec['science_entrypoint'])
    spec['owner_driver_path']=str(Path(__file__).resolve().parents[2]/'src/baselines/bace_globalgce_aplus_owner.py')
    spec['owner_driver_sha256']=sha256_file(spec['owner_driver_path'])
    spec['waiting_owner_handover']={'old_owner_sha256':old_sha,'old_pid':445964,'old_start_ticks':53981973,
        'receipt_path':str(out/'global_owner_exit.json'),'receipt_sha256':sha256_file(out/'global_owner_exit.json'),
        'existing_publication_lock':read(out/'prepared.json')['existing_publication_lock']}
    atomic_json(out/'global_owner_spec.json',spec)
    # Existing dataset owner; this one-shot process exits after handing it off.
    command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--action','run-global-owner','--output-root',str(out)]
    with (out/'global_owner.log').open('xb') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic_json(out/'global_owner_submission.json',{'pid':child.pid,'command':command,'fresh_training_started':False,
        'resume_same_campaign':True,'created_at':utc_now()})
    print(json.dumps(read(out/'global_owner_submission.json')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Paired CLI compatibility; resource contracts remain explicitly bound')
    parser.add_argument('--action',choices=['prepare','activate-aids','handover-global','run-global-owner'],required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    if not args.output_root.is_absolute() or not str(args.output_root).startswith(str(R/'control')+'/'):
        raise ValueError('PROJECT_CONTROL_FRESH_ROOT_REQUIRED')
    if args.action=='prepare': prepare(args.output_root)
    elif args.action=='activate-aids': activate_aids(args.output_root)
    elif args.action=='handover-global': handover_global(args.output_root)
    else:
        from src.baselines.bace_globalgce_aplus_owner import run_owner
        sys.exit(run_owner(args.output_root/'global_owner_spec.json'))
