"""T14 binding is configuration/dispatch, not a third canary or science PASS."""
import ast
import copy
import json
import subprocess
from pathlib import Path

import pytest

from src.utils import t14_formal_binding as binding
from scripts.autodl import run_t14_route_c_owner as owner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_canonical_scientific_ast_allows_validator_only(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    for root in (old, new):
        for name in binding.SOURCE_FILES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('def science():\n return 7\n\ndef validate_spec(x):\n return x\n')
    (new/binding.SOURCE_FILES[0]).write_text('def science():\n return 7\n\ndef validate_spec(x):\n return dict(x)\n')
    assert len(binding.source_inventory(old,new)) == len(binding.SOURCE_FILES)
    (new/binding.SOURCE_FILES[0]).write_text('def science():\n return 8\n')
    with pytest.raises(binding.route.T14RouteCFreshError,match='new parity'):
        binding.source_inventory(old,new)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root=tmp_path/'owner';root.mkdir()
    source=dict(owner_root=str(root),output_root=str(tmp_path/'formal-unused'),
        execution_commit='a'*40,fresh_retry={'retry_index':2},spec_sha256='oldsha',
        science_environment={'seed':'7'},m_configured_max=20000)
    source_path=root/'master.json';write(source_path,source)
    old_auth=root/'oldauth.json';write(old_auth,{'corrected_execution_commit':'a'*40})
    old_cadence=root/'oldcadence.json';write(old_cadence,{'execution_commit':'a'*40})
    bootstrap_path=root/'bootstrap.json'
    bootstrap={'children':{},'preserved_reference':{}}
    for role in ('REFERENCE_500','LOW_MEMORY_CONTINUOUS_510','LOW_MEMORY_RELOAD_510'):
        child=dict(spec_sha256=role,output_root=str(root/role))
        path=root/(role+'.json');write(path,child)
        row={'spec_path':str(path),**child}
        if role=='REFERENCE_500':bootstrap['preserved_reference']=row
        else:bootstrap['children'][role]=row
    write(bootstrap_path,bootstrap)
    authorization=root/'auth.json'
    write(authorization,dict(authorized_by='user_project_owner',
        allow_t14_formal_version_binding_repair=True,allow_retry3=False,
        reference_rerun_allowed=False,parity_required=True,source_master=binding.bind(source_path),
        formal_execution_commit='b'*40))
    receipt=dict(schema_version=binding.SCHEMA,retry_index=2,retry3_created=False,
        diagnostic_checkpoint_promotion_allowed=False,source_master=binding.bind(source_path),
        source_authorization=binding.bind(old_auth),source_cadence=binding.bind(old_cadence),
        bootstrap_binding=binding.bind(bootstrap_path),authorization=binding.bind(authorization),
        formal_execution_commit='b'*40,canary_execution_commit='c'*40,
        formal_worktree=str(tmp_path),runtime_relative_paths={'science_wrapper':'wrapper.sh','owner_entrypoint':'owner.py'},
        science_source_inventory=[],expected_parity_receipt=str(root/'parity.json'))
    for name in ('wrapper.sh','owner.py'):(tmp_path/name).write_text('same code')
    receipt['receipt_sha256']=binding.route.stable_sha256(receipt)
    path=root/'binding.json';write(path,receipt)
    spec=dict(source,execution_commit='b'*40,formal_runtime_binding=binding.bind(path),spec_sha256='newsha',
        science_wrapper=str(tmp_path/'wrapper.sh'),owner_entrypoint=str(tmp_path/'owner.py'))
    for name in ('science_wrapper','owner_entrypoint'):spec[name+'_sha256']=binding.route.file_sha256(Path(spec[name]))
    monkeypatch.setattr(binding.route,'validate_spec',lambda value,**kw:value)
    monkeypatch.setattr(binding.route,'load_spec',lambda path,**kw:binding.read(path))
    write(root/'owner.json',dict(owner_pid=321,task_spec=str(source_path),task_spec_sha256='oldsha'))
    proc=tmp_path/'proc';proc.mkdir()
    return root,spec,path,bootstrap,proc


def test_runtime_overlay_preserves_old_contract_and_refuses_drift(runtime):
    root,spec,path,bootstrap,proc=runtime
    original={p:p.read_bytes() for p in root.glob('*.json')}
    assert binding.validate_runtime_spec(spec)['execution_commit']=='a'*40
    assert original=={p:p.read_bytes() for p in root.glob('*.json')}
    drift=copy.deepcopy(spec);drift['science_environment']['seed']='17'
    with pytest.raises(binding.route.T14RouteCFreshError,match='scientific contract'):
        binding.validate_runtime_spec(drift)


def test_runtime_overlay_does_not_accept_arbitrary_commit(runtime):
    _,spec,_,_,_=runtime;spec['execution_commit']='c'*40
    with pytest.raises(binding.route.T14RouteCFreshError,match='identity'):
        binding.validate_runtime_spec(spec)


def write_gate(root,bootstrap):
    gate=dict(status='CANARY_PARITY_PASS_FORMAL_EXECUTION_REBIND_REQUIRED',driver_commit='c'*40,
        bootstrap_rebind=bootstrap,full_started=False,receipts={n:{'status':'PASS'} for n in
            ('reference_vs_lowmemory_1_500','continuous_vs_reload_1_500','continuous_vs_reload_501_510')})
    write(root/'parity.json',gate)
    return gate


def test_active_owner_cannot_be_replaced(runtime):
    root,spec,path,bootstrap,proc=runtime
    (proc/'321').mkdir();write_gate(root,bootstrap)
    with pytest.raises(binding.route.T14RouteCFreshError,match='remains live'):
        binding.ready_plan(spec,path,proc_root=proc)


@pytest.mark.parametrize('failure',['missing','failed_component','wrong_driver','promotion'])
def test_incomplete_canary_never_dispatches(runtime,failure):
    root,spec,path,bootstrap,proc=runtime
    gate=write_gate(root,bootstrap)
    if failure=='missing':del gate['receipts']['continuous_vs_reload_501_510']
    if failure=='failed_component':gate['receipts']['reference_vs_lowmemory_1_500']['status']='FAIL'
    if failure=='wrong_driver':gate['driver_commit']='z'*40
    if failure=='promotion':gate['full_started']=True
    write(root/'parity.json',gate)
    with pytest.raises(binding.route.T14RouteCFreshError,match='complete original'):
        binding.ready_plan(spec,path,proc_root=proc)


def test_completed_canary_reuses_exact_roots_and_rejects_orphan(runtime):
    root,spec,path,bootstrap,proc=runtime;write_gate(root,bootstrap)
    plan=binding.ready_plan(spec,path,proc_root=proc)
    assert plan['children']['REFERENCE_500']==bootstrap['preserved_reference']
    writer=proc/'999';writer.mkdir()
    (writer/'cmdline').write_bytes(bootstrap['children']['LOW_MEMORY_CONTINUOUS_510']['spec_path'].encode()+b'\0')
    with pytest.raises(binding.route.T14RouteCFreshError,match='writer remains'):
        binding.ready_plan(spec,path,proc_root=proc)


def test_existing_formal_implementation_shared_not_reimplemented():
    source=Path(owner.__file__).read_text();tree=ast.parse(source)
    funcs={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
    body=ast.get_source_segment(source,funcs['_continue_formal'])
    assert 'continuous_vs_promotable_1_500' in body
    assert '(*EARLY_CHECKPOINT_STEPS, 500)' in body
    assert 'publish_generation_handoff' in body and 'launch_continuation_owner' in body
    main=ast.get_source_segment(source,funcs['main'])
    assert main.count('return _continue_formal(')==2
    assert main.index('ready_plan(master')<main.index('owner_root.mkdir')
    assert 'T14 formal binding cannot launch or prepare canaries' in main


def test_formal_identity_pin_fields_are_individually_declared():
    source=Path(binding.__file__).read_text()
    assert "'field':'master.execution_commit'" in source
    assert "'field':'fresh_retry.authorization_receipt.corrected_execution_commit'" in source
    assert "'field':'fresh_retry.formal_cadence_contract.execution_commit'" in source
    assert 'ORIGINAL_UNSTARTED_PROMOTABLE_ROOT_FRESH_0_TO_500_THEN_PROMOTE' in source


def test_extracted_formal_loop_exact_ast_matches_original_driver():
    repo=Path(owner.__file__).resolve().parents[2]
    old=subprocess.check_output(['git','-C',str(repo),'show',
        '56fcd3c7fa64abec204a237621d1f71bfc75c1e8:scripts/autodl/run_t14_route_c_owner.py'],text=True)
    lines=old.splitlines()
    start=next(i for i,line in enumerate(lines) if 'for early_checkpoint in (*EARLY_CHECKPOINT_STEPS, 500):' in line)
    end=next(i for i,line in enumerate(lines[start:],start) if 'launch_continuation_owner(args.continuation_spec)' in line)
    original='\n'.join(line[8:] for line in lines[start:end+1])
    original=original.replace('args.task_spec','task_spec_path').replace('args.continuation_spec','continuation_spec_path')
    expected=ast.parse(original).body
    tree=ast.parse(Path(owner.__file__).read_text())
    actual=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_continue_formal').body[:-1]
    assert ast.dump(ast.Module(body=expected,type_ignores=[]),include_attributes=False)==ast.dump(
        ast.Module(body=actual,type_ignores=[]),include_attributes=False)
