"""One same-UUID T14 bootstrap correction, before any science root exists."""
from pathlib import Path
import os

from scripts.autodl import run_t14_route_c_owner as owner
from src.baselines.tastemolnet_t14_route_c_fresh import stable_sha256,write_spec

SCHEMA='t14_retry2_same_stage_bootstrap_rebind_v1'
ERROR='CUDA_VISIBLE_DEVICES differs from the selected physical GPU'
IDENTITY_FIELDS={'execution_commit','science_wrapper','science_wrapper_sha256',
                 'owner_entrypoint','owner_entrypoint_sha256','spec_sha256'}


def _binding(path):return {'path':str(path),'sha256':owner._small_sha(path)}


def _original(master,master_path,replacement_path,plan):
    receipt=owner._json_object(replacement_path)
    old=owner.load_spec(Path(receipt['children']['LOW_MEMORY_CONTINUOUS_510']['spec_path']))
    old_repo=Path(old['owner_entrypoint']).parents[2]
    return owner.load_failed_stage_replacement(master,master_path,plan,replacement_path,driver_root=old_repo)


def _new_child(old,commit):
    fresh=dict(old,execution_commit=commit,
        science_wrapper=str(owner.REPO_ROOT/'scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh'),
        owner_entrypoint=str(owner.REPO_ROOT/'scripts/autodl/run_t14_route_c_owner.py'))
    for field in ('science_wrapper','owner_entrypoint'):
        fresh[field+'_sha256']=owner._small_sha(Path(fresh[field]))
    fresh['spec_sha256']=stable_sha256({k:v for k,v in fresh.items() if k!='spec_sha256'})
    assert {k:v for k,v in fresh.items() if k not in IDENTITY_FIELDS}=={k:v for k,v in old.items() if k not in IDENTITY_FIELDS}
    return fresh


def prepare_bootstrap_rebind(master,master_path,replacement_path,authorization_path,output,*,proc_root=Path('/proc')):
    root=Path(master['owner_root'])
    if output!=root/'stage_replacement_retry2/bootstrap_rebind/receipt.json' or output.parent.exists():
        raise owner.T14RouteCFreshError('T14 one bootstrap rebind already used or wrong location')
    plan=owner._load_or_create_plan(master,master_path=master_path,owner_root=root)
    original=_original(master,master_path,replacement_path,plan)
    auth=owner._json_object(authorization_path)
    expected=dict(schema_version='t14_same_stage_bootstrap_user_authorization_v1',authorized_by='user_project_owner',
        retry_index=2,stage_replacement_count=1,bootstrap_correction_index=1,max_bootstrap_corrections=1,
        same_stage_uuid_required=True,require_science_output_absent=True,checkpoint_resume=False,
        retry3_allowed=False,reference_rerun_allowed=False,formal_execution_rebind_required=True)
    if any(auth.get(k)!=v for k,v in expected.items()) or auth.get('original_replacement')!=_binding(replacement_path):
        raise owner.T14RouteCFreshError('T14 exact bootstrap-only authorization required')
    commit=owner._replacement_commit()
    terminal_path=root/'terminal.json';terminal=owner._json_object(terminal_path)
    if (auth.get('driver_commit')!=commit or auth.get('failed_terminal')!=_binding(terminal_path)
            or terminal.get('status')!='FAILED'
            or terminal.get('error')!='Route C science phase failed: lowmemory-continuous-510, exit=1'):
        raise owner.T14RouteCFreshError('T14 bootstrap failure/driver binding changed')
    if (proc_root/str(terminal['owner_pid'])).exists():
        raise owner.T14RouteCFreshError('T14 bootstrap prior owner still exists')
    stderr=root/'logs/lowmemory-continuous-510.err'
    with stderr.open('rb') as stream:
        stream.seek(max(0,stderr.stat().st_size-8192));tail=stream.read().decode('utf8','replace')
    if ERROR not in tail or 'require_gpu_runtime' not in tail:
        raise owner.T14RouteCFreshError('T14 exact GPU-mask bootstrap failure absent')
    old_specs={role:owner.load_spec(Path(row['spec_path'])) for role,row in original['children'].items()
               if role!='REFERENCE_500'}
    for spec in old_specs.values():
        if Path(spec['output_root']).exists() or Path(spec['output_root']).is_symlink():
            raise owner.T14RouteCFreshError('T14 bootstrap cannot relabel an existing science root')
    tokens={str(master_path),*(row['spec_path'] for role,row in original['children'].items() if role!='REFERENCE_500')}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:argv=set((directory/'cmdline').read_bytes().decode().split('\0'))
        except (OSError,UnicodeError):continue
        if tokens & argv:raise owner.T14RouteCFreshError('T14 bootstrap exact writer remains')
    output.parent.mkdir(parents=True,exist_ok=False)
    children={}
    for role,old in old_specs.items():
        fresh=_new_child(old,commit)
        path=output.parent/(role.lower()+'.json')
        write_spec(path,fresh)
        children[role]={'spec_path':str(path),'spec_sha256':fresh['spec_sha256'],'output_root':fresh['output_root']}
    receipt=dict(schema_version=SCHEMA,retry_index=2,stage_replacement_count=1,bootstrap_correction_index=1,
        max_bootstrap_corrections=1,original_replacement=_binding(replacement_path),
        authorization=_binding(authorization_path),old_driver_commit=original['failed_stage_replacement']['driver_commit'],
        driver_commit=commit,source_science_commit=master['execution_commit'],children=children,
        original_children={role:original['children'][role] for role in old_specs},
        preserved_reference=original['children']['REFERENCE_500'],failure_terminal=terminal,
        failure_terminal_sha256=owner._small_sha(terminal_path),bootstrap_error=ERROR,
        failure_stderr_tail=tail,science_roots_absent_at_binding=True,checkpoint_resume=False,
        same_uuids_and_outputs=True,reference_rerun=False,retry3_created=False,formal_execution_rebind_required=True,
        created_at=owner._utc_now())
    receipt['receipt_sha256']=stable_sha256(receipt)
    owner.atomic_json(output,receipt)
    return receipt


def load_bootstrap_rebind(master,master_path,plan,replacement_path,path):
    original=_original(master,master_path,replacement_path,plan)
    receipt=owner._json_object(path)
    if (path!=Path(master['owner_root'])/'stage_replacement_retry2/bootstrap_rebind/receipt.json'
            or receipt.get('receipt_sha256')!=stable_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'})
            or receipt.get('schema_version')!=SCHEMA or receipt.get('driver_commit')!=owner._replacement_commit()
            or receipt.get('original_replacement')!=_binding(replacement_path)
            or receipt.get('source_science_commit')!=master['execution_commit']
            or receipt.get('old_driver_commit')!=original['failed_stage_replacement']['driver_commit']
            or receipt.get('preserved_reference')!=original['children']['REFERENCE_500']
            or receipt.get('same_uuids_and_outputs') is not True or receipt.get('checkpoint_resume') is not False):
        raise owner.T14RouteCFreshError('T14 bootstrap rebind provenance changed')
    auth=receipt['authorization']
    if _binding(Path(auth['path']))!=auth:raise owner.T14RouteCFreshError('T14 bootstrap authorization bytes changed')
    children=dict(original['children'])
    for role in ('LOW_MEMORY_CONTINUOUS_510','LOW_MEMORY_RELOAD_510'):
        if receipt['original_children'][role]!=original['children'][role]:
            raise owner.T14RouteCFreshError('T14 original child binding changed')
        old=owner.load_spec(Path(original['children'][role]['spec_path']))
        row=receipt['children'][role];fresh=owner.load_spec(Path(row['spec_path']))
        if fresh!=_new_child(old,receipt['driver_commit']) or row['spec_sha256']!=fresh['spec_sha256'] or row['output_root']!=fresh['output_root']:
            raise owner.T14RouteCFreshError('T14 bootstrap changed fields beyond actual driver identity')
        children[role]=row
    return dict(original,children=children,bootstrap_rebind=receipt)
