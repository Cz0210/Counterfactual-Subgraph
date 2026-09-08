"""Existing Global CPU resource binding and narrowly authorized waiter handoff.

Legacy prepare/activate-aids remain unchanged. The 20260908 closeout actions
require their separate authorization and preserve the frozen pool/science.
"""
from __future__ import annotations
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
if (_root/'source.zip').is_file(): sys.path.insert(0, str(_root/'source.zip'))
import argparse
import copy
import fcntl
import json
import os
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, utc_now
from src.utils.stage_file_policy import canonical_sha, load_stage_policy, config_file_admission
from src.utils.final16_owner_registry_v1 import process_start_ticks
from src.utils.terminal_resource_dependency import verify_terminal_dependency
from src.utils.global_cpu_resource_successor import (serial_chain_peak, assert_export_only,
    prepare_cpu_spec, assert_dynamic_config_only, joint_memory_assessment)

R = Path('/autodl-fs/data/counterfactual-subgraph-runtime')
G = R/'control/bace-gin-aplus-globalgce-41b7ecb-20260908'
H = G/'cpu-handoff-c781341-attempt2'
A = R/'outputs/autodl/recovery/aids-rfaligned-v2-20260907/resource-925ed27f'
PRIOR = R/'control/t14-terminal-resource-rebind-4ec42b4-20260908'
REG = R/'control/final16-owner-registry/current.json'


def read(path): return json.loads(Path(path).read_text())


def descriptor(path): return {'path': str(path), 'sha256': sha256_file(path)}


def seal(path, policy):
    policy.pop('self_sha256', None)
    policy['self_sha256'] = canonical_sha(policy)
    atomic_json(path, policy)
    return load_stage_policy(descriptor(path), R)


def identity(pid):
    proc = Path('/proc')/str(pid)
    ticks = process_start_ticks(Path('/proc'), pid)
    return {'pid': pid, 'start_ticks': ticks,
        'command': (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode() if ticks else None,
        'children': [int(x) for p in (proc/'task').glob('*/children') for x in p.read_text().split()]}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cpu_spec = read(H/'spec.json')
    if Path(cpu_spec['output_root']).exists():
        raise ValueError('CPU_EVALUATION_ALREADY_STARTED_PRESERVE_OWNER_AND_SPEC')
    gpu_spec = read(PRIOR/'global_owner_spec.json')
    assert_export_only(gpu_spec)
    gpu_resource = read(PRIOR/'global-resource.json')
    for item in gpu_resource['terminal_resource_dependencies']:
        verify_terminal_dependency(item, read(REG), Path('/proc'))
    gpu_policy = load_stage_policy(gpu_resource['stage_file_policy'], R)
    old_cpu_resource = read(cpu_spec['cpu_resource_config']['path'])
    old_cpu_policy = load_stage_policy(old_cpu_resource['stage_file_policy'], R)
    old_aids = read(A/'resource_config.json')
    aids_policy = load_stage_policy(old_aids['stage_file_policy'], R)
    cpu_identity = identity(447477)
    if cpu_identity['start_ticks'] != 54199941 or cpu_identity['children']:
        raise ValueError('PROTECTED_CPU_WAITING_OWNER_IDENTITY_CHANGED')
    cpu_owner = read(Path(cpu_spec['cpu_handoff_root'])/'owner.json')
    heartbeat = read(Path(cpu_spec['cpu_handoff_root'])/'heartbeat.json')
    if heartbeat.get('state') != 'WAITING_TRAINING_AND_POOL_FREEZE':
        raise ValueError('PROTECTED_CPU_OWNER_NOT_WAITING')
    atomic_json(out/'protected_cpu_owner.json', {'process': cpu_identity, 'owner': cpu_owner,
        'heartbeat': heartbeat, 'signaled': False, 'replaced': False})
    atomic_json(out/'authorization.json', {'source': 'CODEX_COMPLETE_RECOVERY.md 20260908 section4.1',
        'allow_stage_based_inode_policy': True, 'inode_base_reserve': 20000,
        'inode_next_stage_peak_factor': 2, 'contact_support_first': False,
        'allow_aids_dynamic_resource_rebind': True, 'cpu_owner_447477_must_not_stop': True,
        'created_at': utc_now()})
    gpu_component = gpu_policy['stages']['llm_gpu_generation']['components'][0]
    export_component = gpu_policy['stages']['llm_cpu_evaluation']['components'][0]
    evaluation_component = old_cpu_policy['stages']['llm_cpu_evaluation']['components'][0]
    peak = serial_chain_peak(gpu_component['peak_new_files'], export_component['peak_new_files'],
                             evaluation_component['peak_new_files'])
    if peak != 136: raise ValueError('GLOBAL_CHAIN_STAGE_BOUNDS_CHANGED_REVIEW')
    # Current owned source is immutable. Bind the statements proving export
    # child completion -> terminal publication -> independent CPU READY.
    owner_source = Path(gpu_spec['owner_driver_path'])
    evaluation_owner_source = Path(cpu_spec['raw_kernel_source_root'])/'src/baselines/bace_globalgce_aplus_owner.py'
    owned = owner_source.read_text()
    evaluating = evaluation_owner_source.read_text()
    if not (owned.index('if child.returncode:') < owned.index("final_state=(")
            < owned.index("atomic_json(root/'terminal.json', {'state':final_state")):
        raise ValueError('EXPORT_TERMINAL_ORDERING_NOT_PROVEN')
    if not ("if not terminal.exists():return 'WAITING_TRAINING_AND_POOL_FREEZE'" in evaluating
            and "while cpu_predecessor_state(spec)!='READY':" in evaluating):
        raise ValueError('CPU_PREDECESSOR_SERIAL_BOUNDARY_CHANGED')
    boundary = 'gpu_or_export_exit_then_frozen_pool_terminal_then_independent_cpu_parent'
    proof = {'state': 'CODE_BOUND_FILE_PEAK', 'scientific_changes': False,
        'created_at': utc_now(), 'components': {'globalgce_serial_gpu_export_cpu': {
            'peak_new_files': peak, 'safe_boundary': boundary,
            'source_references': [descriptor(owner_source), descriptor(evaluation_owner_source),
                descriptor(PRIOR/'global_owner_spec.json'), descriptor(H/'spec.json')],
            'derivation': {'gpu_peak': gpu_component['peak_new_files'],
                'export_peak': export_component['peak_new_files'],
                'independent_cpu_peak': evaluation_component['peak_new_files'],
                'waiting_owner_atomic_record_margin': 8,
                'serial_formula': 'max(gpu+8, export+8, independent_cpu)',
                'export_child_exit_precedes_owner_terminal': True,
                'independent_cpu_requires_frozen_pool_and_owner_terminal': True,
                'bounded_waiter_writes': 'owner.json once; one heartbeat plus atomic tmp; no per-loop file/log',
                'future_t14_or_other_stage_requires_scope_rebind': True},
            'upstream_file_peak_evidence': [gpu_component['evidence'], export_component['evidence'],
                                           evaluation_component['evidence']]}}}
    atomic_json(out/'global_serial_chain_peak.json', proof)
    live = copy.deepcopy(gpu_policy['concurrent_components'])
    cpu_policy = copy.deepcopy(old_cpu_policy)
    cpu_policy['concurrent_components'] = live
    cpu_policy['authorization'] = descriptor(out/'authorization.json')
    cpu_policy['terminal_resource_release'] = descriptor(PRIOR/'t14_terminal_release.json')
    cpu_policy['future_resource_requirements'] = descriptor(PRIOR/'future_resources.json')
    seal(out/'cpu-policy.json', cpu_policy)
    fresh_cpu = copy.deepcopy(old_cpu_resource)
    fresh_cpu['stage_file_policy'] = descriptor(out/'cpu-policy.json')
    fresh_cpu['terminal_resource_dependencies'] = gpu_resource['terminal_resource_dependencies']
    atomic_json(out/'cpu-resource.json', fresh_cpu)
    proposed_spec = prepare_cpu_spec(cpu_spec, resource_descriptor=descriptor(out/'cpu-resource.json'))
    atomic_json(out/'cpu-spec-proposed.json', proposed_spec)
    atomic_json(out/'cpu-spec-before.json', cpu_spec)
    aids_common = [copy.deepcopy(x) for x in live if x['component_id'] in {'t13_next_checkpoint','t12_next_checkpoint'}]
    if len(aids_common) != 2: raise ValueError('LIVE_T13_T12_SCOPE_REQUIRED')
    aids_common.append({'component_id': 'globalgce_serial_gpu_export_cpu', 'peak_new_files': peak,
        'safe_boundary': boundary, 'already_existing_files_counted': False,
        'bound_kind': 'SERIAL_SOURCE_AND_EXISTING_STAGE_BOUNDS_WITH_WAITING_MARGIN',
        'evidence': descriptor(out/'global_serial_chain_peak.json')})
    aids_policy['concurrent_components'] = aids_common
    aids_policy['authorization'] = descriptor(out/'authorization.json')
    seal(out/'aids-policy.json', aids_policy)
    fresh_aids = copy.deepcopy(old_aids)
    fresh_aids['stage_file_policy'] = descriptor(out/'aids-policy.json')
    assert_dynamic_config_only(old_aids, fresh_aids)
    atomic_json(out/'aids-resource-before.json', old_aids)
    atomic_json(out/'aids-resource.json', fresh_aids)
    stat = os.statvfs(R)
    admissions = {'cpu': config_file_admission(fresh_cpu, stat.f_favail, stage_id='llm_cpu_evaluation')}
    admissions.update({stage: config_file_admission(fresh_aids, stat.f_favail, stage_id=stage)
        for stage, detail in aids_policy['stages'].items() if detail['state'] == 'BOUNDED'})
    atomic_json(out/'prepared.json', {'state': 'CPU_SPEC_SEALED_NOT_ACTIVATED_AIDS_READY_FOR_DYNAMIC_CAS',
        'cpu_blocker': 'PROTECTED_OWNER_HAS_NO_RELOAD_AND_IMMUTABLE_RESOURCE_SHA; explicit owner-boundary authorization required',
        'cpu_owner_preserved': 447477, 'cpu_spec_written_only': str(out/'cpu-spec-proposed.json'),
        'existing_publication_lock': str(Path(read(REG)['matrix_authority_root'])/'publish.lock'),
        'aids_previous_resource_sha256': sha256_file(A/'resource_config.json'),
        'admissions': admissions, 'created_at': utc_now()})
    print(json.dumps(read(out/'prepared.json')))


def activate_aids(out):
    prepared = read(out/'prepared.json')
    if process_start_ticks(Path('/proc'), 451323) != 54975873:
        raise ValueError('AIDS_EXISTING_CHILD_IDENTITY_CHANGED')
    new = read(out/'aids-resource.json')
    verify_terminal_dependency(descriptor(PRIOR/'t14_terminal_release.json'), read(REG), Path('/proc'))
    assert_dynamic_config_only(read(out/'aids-resource-before.json'), new)
    policy = load_stage_policy(new['stage_file_policy'], R)
    admissions = {key: config_file_admission(new, os.statvfs(R).f_favail, stage_id=key, policy=policy)
        for key, detail in policy['stages'].items() if detail['state'] == 'BOUNDED'}
    if not all(item['admitted'] and not item['pause_requested'] for item in admissions.values()):
        raise ValueError('AIDS_DYNAMIC_ADMISSION_BLOCKED:'+json.dumps(admissions))
    with open(prepared['existing_publication_lock'], 'r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256_file(A/'resource_config.json') != prepared['aids_previous_resource_sha256']:
            raise ValueError('AIDS_CONFIG_CAS_CHANGED')
        atomic_json(A/'resource_config.json', new)
    atomic_json(out/'aids_dynamic_rebind.json', {'state': 'DYNAMIC_CONFIG_REBOUND_NOT_YET_CHILD_ACK',
        'child_pid': 451323, 'start_ticks': 54975873, 'child_restarted': False,
        'new_config': descriptor(A/'resource_config.json'), 'admissions': admissions,
        'memory_and_byte_contract_unchanged': True, 'global_cpu_owner_signaled': False,
        'created_at': utc_now()})
    print(json.dumps(read(out/'aids_dynamic_rebind.json')))


def audit_memory(out):
    config = read(out/'cpu-resource.json')
    concurrent = read(A/'resource_config.json')['other_tasks_headroom_reserve_bytes']
    cg = Path(config['cgroup_memory_root'])
    limit = int((cg/'memory.limit_in_bytes').read_text())
    usage = int((cg/'memory.usage_in_bytes').read_text())
    audit = joint_memory_assessment(legacy_floor=config['minimum_memory_headroom_bytes'],
        concurrent_reserve=concurrent, headroom=limit-usage)
    audit.update(created_at=utc_now(), cgroup_limit_bytes=limit, cgroup_usage_bytes=usage,
        cpu_spec_activation=False, source_cpu_resource=descriptor(out/'cpu-resource.json'),
        previous_legacy_only_preflight=str(out/'cpu_full_resource_preflight.json'),
        legacy_return_does_not_establish_joint_admission=True,
        next_action='Bind actual evaluator incremental peak and concurrency, or wait for a proven serial AIDS boundary; keep protected owner unchanged')
    atomic_json(out/'cpu_joint_memory_audit.json', audit)
    print(json.dumps(audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--action', required=True, choices=['prepare', 'activate-aids', 'audit-memory', 'status', 'memory-pilot', 'handoff'])
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    if not Path(args.config).is_file(): parser.error('existing config required')
    if not args.output_root.is_absolute() or args.output_root.parent != PRIOR:
        parser.error('fresh subdirectory of existing terminal rebind required')
    if args.action == 'prepare': prepare(args.output_root)
    elif args.action == 'activate-aids': activate_aids(args.output_root)
    elif args.action == 'audit-memory': audit_memory(args.output_root)
    elif args.action in ('memory-pilot', 'handoff'):
        from src.utils import global_cpu_closeout
        print(json.dumps(getattr(global_cpu_closeout, args.action.replace('-', '_'))(args.output_root)))
    else:
        for name in ('prepared.json', 'aids_dynamic_rebind.json', 'aids_child_ack.json', 'cpu_joint_memory_audit.json'):
            path = args.output_root/name
            if path.exists(): print(name, json.dumps(read(path)))


if __name__ == '__main__': main()
