"""Bounded owner-binding tests; no Torch payload, GPU or live task operations."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.autodl import run_t14_route_c_owner as owner


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'owner';root.mkdir()
    proc=tmp_path/'proc';proc.mkdir()
    master_path=root/'master.json'
    master=dict(owner_root=str(root),output_root=str(tmp_path/'formal-unused'),
        fresh_retry={'retry_index':2},execution_commit='a'*40,spec_sha256='1'*64,
        storage_mode='lowmemory',canary_role='PROMOTABLE_LOW_MEMORY',
        science_environment={'RUN_GNN_ABLATION':'0'},memory={'launch_headroom_bytes':384*1024**3},
        route_c_state={'exact':True},m_configured_max=20000,m_fallback_max=25000,
        forbidden_legacy_root=str(tmp_path/'forbidden-42gb'),production_checkpoint_steps=[50,100,250,500])
    write(master_path,master)
    plan={'children':{},'master_spec':str(master_path),'master_spec_sha256':master['spec_sha256']}
    for role,mode in [('REFERENCE_500','reference'),('LOW_MEMORY_CONTINUOUS_510','lowmemory'),('LOW_MEMORY_RELOAD_510','lowmemory')]:
        childroot=root/'canaries'/role.lower()/str(uuid4())
        child=dict(master,owner_root=str(childroot),output_root=str(childroot/'science'),
                   canary_role=role,storage_mode=mode,spec_sha256=role)
        path=childroot/'route_c_spec.json';write(path,child)
        plan['children'][role]={'spec_path':str(path),'spec_sha256':role,'output_root':child['output_root']}
    write(root/'owner_plan.json',plan)
    reference=Path(plan['children']['REFERENCE_500']['output_root']);reference.mkdir()
    (reference/'route_c_step_states.jsonl').write_text('{}\n'*500)
    failed=Path(plan['children']['LOW_MEMORY_CONTINUOUS_510']['output_root']);failed.mkdir()
    (failed/'checkpoints').mkdir()
    write(failed/'progress.json',{'completed_step':0})
    terminal={'status':'FAILED','owner_pid':333,'error':'Route C science phase failed: lowmemory-continuous-510, exit=1'}
    write(root/'terminal.json',terminal)
    auth=dict(schema_version='t14_retry2_failed_stage_replacement_authorization_v1',authorized_by='user_project_owner',
        same_retry_index=2,max_stage_replacements=1,stage='LOW_MEMORY_CONTINUOUS_510',
        allow_storage_only_driver_repair=True,reference_rerun_allowed=False,
        old_failed_root_mutation_allowed=False,retry3_allowed=False,formal_execution_rebind_required=True,
        master_spec={'path':str(master_path),'sha256':owner._small_sha(master_path)},
        failed_terminal={'path':str(root/'terminal.json'),'sha256':owner._small_sha(root/'terminal.json')},driver_commit='b'*40)
    authpath=tmp_path/'authorization.json';write(authpath,auth)
    monkeypatch.setattr(owner,'_replacement_commit',lambda:'b'*40)
    monkeypatch.setattr(owner,'load_spec',lambda path:json.loads(Path(path).read_text()))
    monkeypatch.setattr(owner,'_load_or_create_plan',lambda *args,**kwargs:plan)
    monkeypatch.setattr(owner,'validate_checkpoint_boundary',lambda *args,**kwargs:{'boundary':{'checkpoint_digest':'c'*64}})
    def child_spec(runtime_master,*,owner_root,role,storage_mode):
        childroot=owner_root/'canaries'/role.lower()/str(uuid4())
        child=dict(runtime_master,owner_root=str(childroot),output_root=str(childroot/'science'),
                   canary_role=role,storage_mode=storage_mode,spec_sha256=str(uuid4()))
        path=childroot/'route_c_spec.json';write(path,child)
        return child,path
    monkeypatch.setattr(owner,'_child_spec',child_spec)
    return dict(root=root,master=master,master_path=master_path,auth=authpath,plan=plan,failed=failed,
                proc=proc,receipt=root/'stage_replacement_retry2/receipt.json',reference=reference)


def prepare(data):
    return owner.prepare_failed_stage_replacement(data['master'],master_path=data['master_path'],
        authorization_path=data['auth'],output=data['receipt'],proc_root=data['proc'])


def test_replacement_is_once_and_preserves_reference_failed_plan(setup):
    before={p:p.read_bytes() for p in [setup['master_path'],setup['root']/'owner_plan.json',
        setup['root']/'terminal.json',setup['failed']/'progress.json',setup['reference']/'route_c_step_states.jsonl']}
    receipt=prepare(setup)
    assert receipt['retry_index']==2 and not receipt['reference_rerun'] and not receipt['retry3_created']
    assert receipt['preserved_reference']==setup['plan']['children']['REFERENCE_500']
    assert set(receipt['children'])=={'LOW_MEMORY_CONTINUOUS_510','LOW_MEMORY_RELOAD_510'}
    assert all(not Path(v['output_root']).exists() for v in receipt['children'].values())
    assert all(p.read_bytes()==data for p,data in before.items())
    with pytest.raises(owner.T14RouteCFreshError,match='already used'):prepare(setup)


@pytest.mark.parametrize('change',['progress','checkpoint','live_owner','retry3','formal_exists'])
def test_replacement_rejects_nonempty_or_unauthorized_state(setup,change):
    if change=='progress':write(setup['failed']/'progress.json',{'completed_step':1})
    elif change=='checkpoint':(setup['failed']/'checkpoints/pending.json').write_text('{}')
    elif change=='live_owner':(setup['proc']/'333').mkdir()
    elif change=='retry3':
        auth=json.loads(setup['auth'].read_text());auth['same_retry_index']=3;write(setup['auth'],auth)
    else:Path(setup['master']['output_root']).mkdir()
    with pytest.raises(owner.T14RouteCFreshError):prepare(setup)
    assert not setup['receipt'].exists()


def test_replacement_load_rejects_changed_resource_contract(setup):
    receipt=prepare(setup)
    child=Path(receipt['children']['LOW_MEMORY_CONTINUOUS_510']['spec_path'])
    row=json.loads(child.read_text());row['memory']={'launch_headroom_bytes':1};write(child,row)
    with pytest.raises(owner.T14RouteCFreshError,match='resource contract changed'):
        owner.load_failed_stage_replacement(setup['master'],setup['master_path'],setup['plan'],setup['receipt'])


def test_missing_reference_cannot_trigger_science_regeneration():
    source=Path(owner.__file__).read_text()
    gate=source.index("if args.failed_stage_replacement and not reference_root.is_dir():")
    assert gate < source.index('label="reference-500",',gate)
    assert 'rerun forbidden' in source[gate:gate+220]


def test_owner_runs_only_replacement_canaries_then_holds_before_old_master(setup,monkeypatch):
    prepare(setup)
    continuation=setup['root']/'continuation.json';write(continuation,{})
    config=setup['root']/'config.yaml';config.write_text('test: true\n')
    monkeypatch.setattr(owner,'load_continuation_spec',lambda path:dict(route_c_spec=str(setup['master_path']),
        generation_root=setup['master']['output_root'],generation_owner_root=str(setup['root']),
        matrix={'authority_state_path':'/unused/matrix','authority_lock_path':'/unused/lock'}))
    monkeypatch.setattr(owner,'audit_route_c_matrix_cell_absent',lambda **kwargs:None)
    monkeypatch.setattr(owner,'_start_ticks',lambda pid:1)
    sealed=set();calls=[]
    def run(spec,path,**kwargs):
        calls.append((kwargs['label'],kwargs['resume'],kwargs['stop_step']))
        root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=True)
        if kwargs['stop_step']==510:sealed.add(str(root))
    monkeypatch.setattr(owner,'_run_science',run)
    monkeypatch.setattr(owner,'_replay_boundary_valid',lambda spec:str(spec['output_root']) in sealed)
    monkeypatch.setattr(owner,'_promote_checkpoint_fresh_process',lambda *args,**kwargs:None)
    monkeypatch.setattr(owner,'compare_step_ledgers',lambda *args,**kwargs:{'status':'PASS'})
    code=owner.main(['--config',str(config),'--task-spec',str(setup['master_path']),
        '--continuation-spec',str(continuation),'--failed-stage-replacement',str(setup['receipt'])])
    assert code==75
    assert calls==[('lowmemory-continuous-510',False,510),('lowmemory-reload-250',False,250),('lowmemory-reload-510',True,510)]
    result=json.loads((setup['root']/'stage_replacement_retry2/canary_verification.json').read_text())
    assert result['status']=='CANARY_PARITY_PASS_FORMAL_EXECUTION_REBIND_REQUIRED'
    assert not result['full_started'] and not result['retry3_created']
    assert not Path(setup['master']['output_root']).exists()
    assert len(list((setup['root']/'terminal_history').glob('*.json')))==1
