"""One retry2 formal execution overlay; immutable canaries stay evidence only."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.baselines import tastemolnet_t14_route_c_fresh as route

SCHEMA = 't14_retry2_formal_execution_binding_v1'
IDENTITY_FIELDS = {'execution_commit', 'science_wrapper', 'science_wrapper_sha256',
                   'owner_entrypoint', 'owner_entrypoint_sha256', 'spec_sha256',
                   'formal_runtime_binding'}
# This overlay changes only validators and dispatch. Everything evaluated by
# the active canary must have the same semantic AST in the formal driver.
SOURCE_FILES = (
    'src/baselines/tastemolnet_t14_route_c_fresh.py',
    'src/baselines/tastemolnet_comrecgc_full.py',
    'scripts/run_tastemolnet_comrecgc_full.py',
    'src/utils/tastemolnet_t9_managed_v2.py',
    'src/oracles/gnn_oracle.py',
)


def read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024**2:
        raise route.T14RouteCFreshError(f'bounded physical T14 evidence required: {path}')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise route.T14RouteCFreshError('T14 evidence must be an object')
    return value


def bind(path: Path) -> dict[str, str]:
    read(path)  # manifests only, no scientific payloads
    return {'path': str(path), 'sha256': route.file_sha256(path)}


def check(binding: Mapping[str, str]) -> dict[str, Any]:
    path = Path(binding['path'])
    value = read(path)
    if bind(path) != dict(binding):
        raise route.T14RouteCFreshError(f'T14 binding changed: {path}')
    return value


def _semantic_ast(path: Path) -> str:
    tree = ast.parse(path.read_text())
    if path.name == 'tastemolnet_t14_route_c_fresh.py':
        tree.body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                                  and n.name == 'validate_spec')]
    return route.stable_sha256(ast.dump(tree, include_attributes=False))


def source_inventory(canary_root: Path, formal_root: Path) -> list[dict[str, Any]]:
    paths = set(SOURCE_FILES)
    for root in (canary_root, formal_root):
        paths.update(str(p.relative_to(root)) for p in (root/'src/baselines/comrecgc').rglob('*.py'))
    rows = []
    for name in sorted(paths):
        old, new = canary_root/name, formal_root/name
        if not old.is_file() or not new.is_file():
            raise route.T14RouteCFreshError(f'T14 canary/formal source absent: {name}')
        old_ast, new_ast = _semantic_ast(old), _semantic_ast(new)
        if old_ast != new_ast:
            raise route.T14RouteCFreshError(f'T14 affected science requires new parity: {name}')
        rows.append({'relative_path': name, 'canary_ast': old_ast, 'formal_ast': new_ast,
                     'canary_file_sha256': route.file_sha256(old),
                     'formal_file_sha256': route.file_sha256(new)})
    return rows


def validate_runtime_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the original authorized science spec after validating the overlay."""
    binding = spec['formal_runtime_binding']
    receipt = check(binding)
    if (receipt.get('schema_version') != SCHEMA or receipt.get('retry_index') != 2
            or receipt.get('retry3_created') is not False
            or receipt.get('diagnostic_checkpoint_promotion_allowed') is not False
            or receipt.get('receipt_sha256') != route.stable_sha256(
                {k:v for k,v in receipt.items() if k != 'receipt_sha256'})):
        raise route.T14RouteCFreshError('T14 formal binding schema/authorization changed')
    source = check(receipt['source_master'])
    if 'formal_runtime_binding' in source:
        raise route.T14RouteCFreshError('T14 nested formal overlays forbidden')
    route.validate_spec(source, check_files=False)
    if (source.get('fresh_retry', {}).get('retry_index') != 2
            or spec.get('execution_commit') != receipt['formal_execution_commit']
            or {k:v for k,v in spec.items() if k not in IDENTITY_FIELDS}
               != {k:v for k,v in source.items() if k not in IDENTITY_FIELDS}):
        raise route.T14RouteCFreshError('T14 formal identity/scientific contract changed')
    authorization = check(receipt['authorization'])
    expected = {'authorized_by':'user_project_owner', 'allow_t14_formal_version_binding_repair':True,
                'allow_retry3':False, 'reference_rerun_allowed':False, 'parity_required':True}
    if (any(authorization.get(k) != v for k,v in expected.items())
            or authorization.get('source_master') != receipt['source_master']
            or authorization.get('formal_execution_commit') != receipt['formal_execution_commit']):
        raise route.T14RouteCFreshError('T14 formal authorization is not exact')
    for field in ('science_wrapper', 'owner_entrypoint'):
        if spec[field] != str(Path(receipt['formal_worktree']) / receipt['runtime_relative_paths'][field]):
            raise route.T14RouteCFreshError('T14 formal executable path changed')
        if route.file_sha256(Path(spec[field])) != spec[field+'_sha256']:
            raise route.T14RouteCFreshError('T14 formal executable bytes changed')
    for row in receipt['science_source_inventory']:
        # The code is immutable; small source files, never a checkpoint or OT.
        if route.file_sha256(Path(receipt['formal_worktree'])/row['relative_path']) != row['formal_file_sha256']:
            raise route.T14RouteCFreshError('T14 formal science bytes changed')
    check(receipt['source_authorization'])
    check(receipt['source_cadence'])
    check(receipt['bootstrap_binding'])
    return source


def prepare(master_path: Path, continuation_path: Path, authorization_path: Path,
            formal_root: Path, *, driver_root: Path) -> dict[str, Any]:
    from scripts.autodl import run_t14_route_c_owner as owner
    master = route.load_spec(master_path)
    root = Path(master['owner_root'])
    if formal_root != root/'formal_execution_binding' or formal_root.exists():
        raise route.T14RouteCFreshError('T14 unique fresh formal binding root required')
    if Path(master['output_root']).exists() or any(Path(master['promotion_root']).glob('*.json')):
        raise route.T14RouteCFreshError('T14 formal branch already has scientific progress')
    bootstrap_path = root/'stage_replacement_retry2/bootstrap_rebind/receipt.json'
    bootstrap = read(bootstrap_path)
    child = route.load_spec(Path(bootstrap['children']['LOW_MEMORY_CONTINUOUS_510']['spec_path']))
    canary_repo = Path(child['owner_entrypoint']).parents[2]
    commit = owner._replacement_commit(driver_root)
    auth = read(authorization_path)
    expected = {'authorized_by':'user_project_owner', 'allow_t14_formal_version_binding_repair':True,
                'allow_retry3':False, 'reference_rerun_allowed':False, 'parity_required':True,
                'source_master':bind(master_path), 'formal_execution_commit':commit}
    if any(auth.get(k) != v for k,v in expected.items()):
        raise route.T14RouteCFreshError('T14 formal explicit user authorization mismatch')
    inventory = source_inventory(canary_repo, driver_root)
    old_auth_path = Path(master['fresh_retry']['authorization_receipt'])
    old_cadence_path = Path(master['fresh_retry']['formal_cadence_contract'])
    old_auth, old_cadence = read(old_auth_path), read(old_cadence_path)
    bindings = [
        {'stage_role':'formal_science_identity', 'field':'master.execution_commit',
         'old_commit':master['execution_commit'], 'required_commit':commit,
         'reason':'run real storage-v2 implementation already under canary, not old storage-v1'},
        {'stage_role':'retry2_authorization_execution_mapping',
         'field':'fresh_retry.authorization_receipt.corrected_execution_commit',
         'old_commit':old_auth['corrected_execution_commit'], 'required_commit':commit,
         'reason':'new user-authorized execution overlay preserves old retry2 authorization and retirement chain'},
        {'stage_role':'unchanged_cadence_new_source_binding',
         'field':'fresh_retry.formal_cadence_contract.execution_commit',
         'old_commit':old_cadence['execution_commit'], 'required_commit':commit,
         'reason':'same early/main/fallback/convergence cadence; source identity recorded independently'},
    ]
    relpaths = {'science_wrapper':'scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh',
                'owner_entrypoint':'scripts/autodl/run_t14_route_c_owner.py'}
    receipt = dict(schema_version=SCHEMA, status='SEALED_WAITING_EXISTING_CANARY', retry_index=2,
        retry3_created=False, diagnostic_checkpoint_promotion_allowed=False,
        source_master=bind(master_path), source_continuation=bind(continuation_path),
        source_authorization=bind(old_auth_path), source_cadence=bind(old_cadence_path),
        source_scientific_config_sha256=route.stable_sha256(route._retry_scientific_config(master)),
        authorization=bind(authorization_path), bootstrap_binding=bind(bootstrap_path),
        canary_execution_commit=child['execution_commit'], canary_worktree=str(canary_repo),
        formal_execution_commit=commit, formal_worktree=str(driver_root), runtime_relative_paths=relpaths,
        science_source_inventory=inventory, version_pin_binding=bindings,
        cadences=old_cadence['cadences'], source_checkpoint_schema='comrecgc_generation_checkpoint_pending_v1',
        source_graph_codec=route.GRAPH_STORE_SCHEMA,
        promotable_branch_policy='ORIGINAL_UNSTARTED_PROMOTABLE_ROOT_FRESH_0_TO_500_THEN_PROMOTE',
        original_reference_rerun=False, existing_canaries_rerun=False,
        expected_parity_receipt=str(root/'stage_replacement_retry2/canary_verification.json'),
        created_at=owner._utc_now())
    receipt['receipt_sha256'] = route.stable_sha256(receipt)
    formal_root.mkdir(parents=True, exist_ok=False)
    receipt_path = formal_root/'binding.json'
    route.atomic_json(receipt_path, receipt)
    new = dict(master, execution_commit=commit, formal_runtime_binding=bind(receipt_path))
    for field, relative in relpaths.items():
        new[field] = str(driver_root/relative)
        new[field+'_sha256'] = route.file_sha256(driver_root/relative)
    new['spec_sha256'] = route.stable_sha256({k:v for k,v in new.items() if k != 'spec_sha256'})
    spec_path = formal_root/'formal_task_spec.json'
    route.write_spec(spec_path, new)
    continuation = read(continuation_path)
    continuation.update(descriptor_path=str(formal_root/'continuation_spec.json'),
        route_c_spec=str(spec_path), route_c_spec_sha256=route.file_sha256(spec_path),
        route_c_execution_commit=commit, formal_runtime_binding=bind(receipt_path))
    # Postprocess science remains the already sealed old immutable code; only
    # this descriptor-aware existing continuation driver changes.
    continuation['continuation_entrypoint'] = str(driver_root/'scripts/autodl/run_t14_route_c_continuation.py')
    continuation['continuation_entrypoint_sha256'] = route.file_sha256(Path(continuation['continuation_entrypoint']))
    continuation['spec_sha256'] = route.stable_sha256({k:v for k,v in continuation.items() if k != 'spec_sha256'})
    from src.baselines.tastemolnet_t14_route_c_continuation import validate_continuation_spec
    validate_continuation_spec(continuation)
    route.atomic_json(formal_root/'continuation_spec.json', continuation)
    command = [master['python'], '-I', '-B', new['owner_entrypoint'], '--config',
        str(driver_root/'configs/hpc.yaml'), '--task-spec', str(spec_path),
        '--continuation-spec', continuation['descriptor_path'], '--formal-binding', str(receipt_path)]
    route.atomic_json(formal_root/'dispatch.json', dict(status='SEALED_WAITING_EXISTING_CANARY',
        command=command, owner_lock=str(root/'owner.lock'), no_second_active_owner=True,
        resource_contract=master['memory'], binding=bind(receipt_path)))
    return receipt


def ready_plan(master: Mapping[str, Any], binding_path: Path, *, proc_root=Path('/proc')) -> dict[str, Any]:
    """Fail before writing owner state unless the old canary naturally exited."""
    from scripts.autodl import run_t14_route_c_owner as owner
    validate_runtime_spec(master)
    receipt = check(master['formal_runtime_binding'])
    if master['formal_runtime_binding'] != bind(binding_path):
        raise route.T14RouteCFreshError('T14 command binding mismatch')
    root = Path(master['owner_root'])
    old_owner = read(root/'owner.json')
    if old_owner.get('task_spec_sha256') == master['spec_sha256']:
        # Actual formal checkpoint resume is allowed; never repeat canaries.
        pass
    else:
        original = check(receipt['source_master'])
        if (old_owner.get('task_spec') != receipt['source_master']['path']
                or old_owner.get('task_spec_sha256') != original['spec_sha256']):
            raise route.T14RouteCFreshError('T14 prior owner task binding changed')
        if (proc_root/str(old_owner['owner_pid'])).exists():
            raise route.T14RouteCFreshError('T14 prior canary owner remains live; wait for natural exit')
    gate = read(Path(receipt['expected_parity_receipt']))
    bootstrap = check(receipt['bootstrap_binding'])
    required_names = {'reference_vs_lowmemory_1_500', 'continuous_vs_reload_1_500',
                      'continuous_vs_reload_501_510'}
    if (gate.get('status') != 'CANARY_PARITY_PASS_FORMAL_EXECUTION_REBIND_REQUIRED'
            or gate.get('driver_commit') != receipt['canary_execution_commit']
            or gate.get('bootstrap_rebind') != bootstrap
            or set(gate.get('receipts',{})) != required_names
            or any(x.get('status') != 'PASS' for x in gate['receipts'].values())
            or gate.get('full_started') is not False):
        raise route.T14RouteCFreshError('T14 complete original canary parity required')
    children = dict(bootstrap['children'], REFERENCE_500=bootstrap['preserved_reference'])
    for row in children.values():
        child = route.load_spec(Path(row['spec_path']))
        if child['spec_sha256'] != row['spec_sha256'] or child['output_root'] != row['output_root']:
            raise route.T14RouteCFreshError('T14 canary child spec changed')
    tokens = {row['spec_path'] for row in children.values()} | {master['output_root']}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = set((directory/'cmdline').read_bytes().decode().split('\0'))
        except (OSError, UnicodeError):
            continue
        if tokens & arguments:
            raise route.T14RouteCFreshError('T14 exact predecessor/formal scientific writer remains')
    return {'children':children, 'formal_binding':receipt}
