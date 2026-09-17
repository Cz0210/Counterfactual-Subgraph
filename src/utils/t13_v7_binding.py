"""V7 binding of the existing T13 same-run owner; no new scientific authority.

Preparation is static. Only the original owner and its inherited descriptors
admit a child. Final evaluation has a separate provider and cannot borrow the
training file/RAM envelope. Old specs and the formal-start ledger are read-only.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import fcntl
import json
import os
import subprocess

from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils.t13_performance_dispatch import bound_json

GIB = 1024**3
DEADLINE = '2026-09-24T15:59:59+00:00'
CUTOFF = '2026-09-24T13:59:59+00:00'
SCHEMA = 'P0_P1_PAPER_CLOSEOUT_20260917_V7'


def descriptor(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha256_file(path)}


def seal(path, value):
    if Path(path).exists():
        raise ValueError('FRESH_V7_OVERLAY_REQUIRED:' + str(path))
    atomic_json(Path(path), value)
    return descriptor(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(*, old_plan_root, old_publisher_root, root, execution_root, authorization_file):
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    execution_root = Path(execution_root).resolve()
    commit = subprocess.check_output(['git', '-C', str(execution_root), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(execution_root), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise ValueError('IMMUTABLE_EXECUTION_CHECKOUT_DIRTY')
    plan = read(Path(old_plan_root)/'plan.json')
    dispatch = read(Path(old_plan_root)/'dispatch.json')
    ledger = read(plan['formal_ledger'])
    if ledger['attempt_id'] != plan['original_formal_attempt_id'] or ledger['max_full_starts'] != 1:
        raise ValueError('ORIGINAL_FORMAL_1_OF_1_CHANGED')
    auth = seal(root/'authorization_overlay.json', dict(schema=SCHEMA,
        source=descriptor(authorization_file), deadline=DEADLINE, science_dispatch_cutoff=CUTOFF,
        external_future_reserved_ram_bytes=0,
        external_reservation_basis='EXPLICIT_USER_V7_REPLACEMENT_NOT_A_KERNEL_MEASUREMENT',
        gpu_indices=[0,1,2,3], hpc_gpu_allowed=False, hpc_max_cpu_jobs=2,
        actual_usage_always_counted=True, scientific_budgets_reset=False))
    window = seal(root/'resource_window.json', dict(state='CONFIRMED_EXCLUSIVE_WINDOW',
        gpu_uuid=plan['gpu_uuid'], ends_at=DEADLINE, future_reservations_complete=True,
        authority_evidence=auth, source='EXPLICIT_USER_V7_RESOURCE_ALLOCATION',
        unexpected_live_processes_must_not_be_interrupted=True))
    probe_memory = Path(plan['gpu_probe_receipt']).parent/'memory_boundaries.json'
    peak = max(int(x['VmHWM_bytes']) for x in read(probe_memory)['samples'])
    policy = seal(root/'training_resource_policy.json', dict(schema='T13_V5_STAGE_INCREMENT_V1',
        formal_quota='1/1', gpu_uuid=plan['gpu_uuid'], safety_margin_bytes=64*GIB,
        process_peak_bound_bytes=32*GIB, maximum_retained_train_batches=5,
        adopted_probe_memory=descriptor(probe_memory), observed_probe_host_peak_bytes=peak,
        process_bound_scope='PROBE_DERIVED_BOUNDED_LOADER_NOT_FULL_EPOCH_PEAK_MEASUREMENT',
        retired_default_reason='UNIDENTIFIED_FIXED_HEADROOM_NOT_EXTERNAL_TASK_INCREMENT',
        concurrent_future_increments=[], external_future_reservation_authorization=auth,
        resource_window=window, own_other_loads_must_be_serial=True))
    # This provider is executable now but must not invent a bound for unbounded
    # new node-embedding files. It is intentionally independent of training.
    post_policy = seal(root/'postprocess_resource_policy.json', dict(schema='T13_V7_POSTPROCESS_RESOURCE_V1',
        safety_margin_bytes=64*GIB, phase='T13_FINAL_EVALUATION',
        pair_chunk_entries=2*(468+468), cache_metadata_temp_entries='UNKNOWN',
        process_peak_bound_bytes='UNKNOWN', external_future_reserved_ram_bytes=0,
        basis='evaluate_split_resumable: JSONL + manifest per parent; MolCLR node cache NPZ per distinct graph',
        cannot_reuse_training_256_entries=True, authorization=auth))
    old_specs = Path(old_publisher_root)
    old_t13 = read(old_specs/'t13_from_hpc_task_spec.json')
    inputs = old_t13['input_paths']
    old_theta = read(inputs['threshold_contract'])
    values = sorted(set([float(x) for x in old_theta.get('thresholds', [])] + [0.0,0.1,0.2]))
    from src.baselines.tastemolnet_globalgce_full import FINAL_EVAL_V6, stable_json_sha256, load_threshold_contract
    theta = dict(dataset='TasteMolNet', thresholds=values, theta_star=0.1,
        cost_cap=0.03416003659645076, final_eval_protocol=FINAL_EVAL_V6,
        primary_report_k=20, k_mode='AT_MOST_K_NO_PADDING',
        threshold_config_hash=stable_json_sha256(values), threshold_source='V6/V7 explicit user protocol',
        threshold_source_split='frozen_protocol', test_used_for_selection=False,
        prior_threshold_source=descriptor(inputs['threshold_contract']))
    seal(root/'final_threshold_v6.json', theta)
    load_threshold_contract(root/'final_threshold_v6.json')
    output = root/'science'
    publisher_root = root/'publisher-spec'
    publisher_root.mkdir()
    manifest = read(old_specs/'spec_set_manifest.json')
    from src.utils.t8_hpc_t13_successor_v1 import canonical_sha256, validate_spec_set
    specs = {}
    for role, name in manifest['specs'].items():
        value = copy.deepcopy(read(old_specs/name))
        value['execution_commit'] = commit
        if 'repo_root' in value: value['repo_root'] = str(execution_root)
        if role == 't13':
            value['output_root'] = str(output)
            value['attempt_id'] = plan['original_formal_attempt_id']
            value['task_id'] = dispatch['canonical_task_id']
            value['owner_entrypoint'] = str(execution_root/'scripts/autodl/run_t13_from_hpc_owner_v1.py')
            command = value['command']
            replacements = {str(old_specs):str(publisher_root), old_t13['repo_root']:str(execution_root),
                            old_t13['output_root']:str(output)}
            for old,new in replacements.items(): command = [x.replace(old,new) for x in command]
            value['command'] = command
        if role == 'publisher': value['expected_terminal_root'] = str(output)
        value.pop('task_spec_sha256', None)
        value['task_spec_sha256'] = canonical_sha256(value)
        seal(publisher_root/name, value)
        specs[role] = value
    manifest.update(spec_root=str(publisher_root), execution_commit=commit,
        task_spec_sha256s={r:s['task_spec_sha256'] for r,s in specs.items()})
    manifest.pop('spec_set_sha256',None)
    manifest['spec_set_sha256'] = canonical_sha256(manifest)
    seal(publisher_root/'spec_set_manifest.json',manifest)
    validate_spec_set(publisher_root, check_files=False)
    authority = {k:inputs[k] for k in ('train_csv','calibration_csv','test_csv','gnn_checkpoint','official_root','molclr_root','molclr_checkpoint')}
    authority.update(t8_pass_root=None, hpc_import_root=old_t13['required_import_root'],
                     threshold_contract=str(root/'final_threshold_v6.json'))
    python = dispatch['science_command_without_owner_fds'][0]
    final = seal(root/'final_evaluation_binding.json',dict(protocol=FINAL_EVAL_V6,
        training_plan_original_attempt_id=plan['original_formal_attempt_id'], formal_quota_used='1/1',
        output_root=str(output), input_authority=authority,
        wnode_cache_db=inputs['wnode_cache_db'], node_embedding_cache_dir=inputs['node_embedding_cache_dir'],
        publisher_spec_root=str(publisher_root), resource_policy_sha256=post_policy['sha256'],
        resource_provider_command=[python,'-I','-B',str(execution_root/'scripts/run_t13_v7_binding.py'),
            '--config',str(execution_root/'configs/hpc.yaml'),'--action','post-provider','--root',str(root)],
        execution_chain=['continue_branches','run_final_successor','run_t13_full','verify_t13_output','publish_verified_t13_locator']))
    plan.update(deadline_utc=DEADLINE,science_dispatch_cutoff_utc=CUTOFF,
        stage_resource_policy=policy,final_evaluation_binding=final,output_root=str(output),
        immutable_execution_commit=commit,other_remaining_reserve_bytes=0,
        authorization_overlay=auth,old_plan=descriptor(Path(old_plan_root)/'plan.json'))
    plan_binding = seal(root/'plan.json',plan)
    dispatch.update(plan_path=plan_binding['path'],plan_sha256=plan_binding['sha256'],
        stage_resource_policy=policy,other_remaining_reserve_bytes=0,process_peak_bound_bytes=32*GIB,
        task_id='t13-v7-same-run-'+root.name,owner_root=str(root/'owner'),
        v7_registry_binding_root=str(root),
        nvme_root=str(Path(old_plan_root)),science_dispatch_cutoff_utc=CUTOFF,
        science_command_without_owner_fds=[python,'-I','-B',str(execution_root/'scripts/run_t13_compact_recovery.py'),
            '--config',str(execution_root/'configs/hpc.yaml'),'--plan',str(root/'plan.json'),'--action','continue'])
    seal(root/'dispatch.json',dispatch)
    seal(root/'old_to_same_run_mapping.json',dict(original_formal_root=ledger['output_root'],
        original_checkpoint=plan['source_checkpoint'],prepared_checkpoint=plan['working_checkpoint'],
        same_formal_attempt_id=ledger['attempt_id'],new_output_root=str(output),
        original_publisher_root=str(old_specs),new_publisher_root=str(publisher_root),
        max_full_starts=1,new_fresh_start=False))
    return dict(state='STATIC_BINDING_COMPLETE_NOT_SCIENCE',root=str(root),commit=commit)


def post_provider(root):
    root=Path(root);p=read(root/'postprocess_resource_policy.json')
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    free=os.statvfs(root)
    return dict(stage='T13_FINAL_EVALUATION',policy_sha256=sha256_file(root/'postprocess_resource_policy.json'),
        observed_at=datetime.now(timezone.utc).isoformat(),allowed=False,
        blockers=['POSTPROCESS_NEW_NODE_EMBEDDING_CACHE_PEAK_UNQUANTIFIED','POSTPROCESS_RAM_BOUND_UNQUANTIFIED'],
        pair_chunk_entries=p['pair_chunk_entries'],cache_metadata_temp_entries=p['cache_metadata_temp_entries'],
        current_available_entries=free.f_favail,current_available_bytes=free.f_bavail*free.f_frsize,
        memory_headroom_bytes=memory_headroom(Path('/proc'),Path('/sys/fs/cgroup/memory')))


def registry_claim(root, *, finish=False, held_fd=None):
    """Narrow CAS under the existing publication lock, preserving all other rows."""
    from src.utils.final16_owner_registry_v1 import (validate_owner_registry, build_owner_registry,
        atomic_write_owner_registry, process_start_ticks)
    root=Path(root);spec=read(root/'dispatch.json');path=Path(spec['registry'])
    if held_fd is not None:
        os.fstat(held_fd)
    before=path.read_bytes();reg=validate_owner_registry(json.loads(before),check_processes=False)
    lock=Path(reg['matrix_authority_root'])/'publish.lock'
    with lock.open('a+') as h:
        fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.read_bytes()!=before:raise ValueError('REGISTRY_CAS_CHANGED')
        target=next(r for r in reg['tasks'] if r['task_id']==spec['canonical_task_id'])
        for row in reg['tasks']:
            if row['gpu']!=1 or not row['owner_pid']:continue
            if process_start_ticks('/proc',row['owner_pid'])==row['owner_start_ticks'] and row['owner_pid']!=os.getpid():
                raise ValueError('LIVE_GPU1_OWNER:'+str(row['owner_pid']))
        if finish and target['owner_pid']!=os.getpid():raise ValueError('OWNER_CAS_IDENTITY_CHANGED')
        ticks=process_start_ticks('/proc',os.getpid())
        target.update(owner_state='BLOCKED' if finish else 'RUNNING',owner_pid=None if finish else os.getpid(),
            owner_start_ticks=None if finish else ticks,heartbeat=str(root/'registry_heartbeat.json'),
            output_root=read(root/'plan.json')['output_root'],execution_commit=read(root/'plan.json')['immutable_execution_commit'],
            task_spec_sha=sha256_file(root/'dispatch.json'),
            stage='V7_OWNER_EXITED_CHECK_TERMINAL' if finish else 'V7_SAME_RUN_EPOCH29_RECOVERY')
        for lease in reg['gpu_leases']:
            if lease['task_id']==spec['canonical_task_id']:
                lease['state']='RELEASED' if finish else 'HELD' if held_fd is not None else 'PREDEPLOYED'
        atomic_json(root/'registry_heartbeat.json',dict(pid=os.getpid(),start_ticks=ticks,
            state=target['stage'],observed_at=datetime.now(timezone.utc).isoformat()))
        updated=build_owner_registry(registry_id=reg['registry_id'],matrix_authority_root=reg['matrix_authority_root'],
            tasks=reg['tasks'],publishers=reg['publishers'],gpu_leases=reg['gpu_leases'],check_processes=False)
        atomic_write_owner_registry(path,updated)


def owner(root):
    root=Path(root)
    plan=read(root/'plan.json');bound_json(plan['authorization_overlay']);bound_json(plan['final_evaluation_binding'])
    if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['science_dispatch_cutoff_utc']):
        raise ValueError('V7_NEW_SCIENCE_CUTOFF_REACHED')
    if (root/'owner').exists():raise ValueError('EXISTING_OWNER_ATTEMPT_REQUIRES_TERMINAL_REVIEW')
    registry_claim(root)
    try:
        from src.utils.t13_gap_recovery_owner import run
        return run(root/'dispatch.json')
    finally:registry_claim(root,finish=True)
