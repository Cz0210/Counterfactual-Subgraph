import json
from types import SimpleNamespace
import pytest,torch
from dataclasses import replace
from src.baselines.bace_globalgce_aplus import build_parent,SCHEMA
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule
from src.experiments import bace_globalgce_aplus_evaluation as e
ATOMS=('C','O');BONDS=('no_edge','single','double','triple')
def rule(smiles='CC'):
    p=build_parent(smiles,atom_symbols=ATOMS,bond_names=BONDS)
    r=GlobalGCENativeRule('test-rule',0,p.feature,p.adjacency,p.edge_attr,p.feature,p.adjacency,p.edge_attr,ATOMS,BONDS)
    return {'candidate_id':r.rule_id,'rule':r.to_payload(),'molecular_adapter':SCHEMA}
class Distance:
    def __init__(self,ok=True):self.ok=ok;self.calls=[]
    def distance(self,p,c):self.calls.append((p,c));return {'ok':self.ok,'distance':.02,'error':None,'cache_hit':False}
def fixture(monkeypatch):
    oracle=SimpleNamespace(backbone='gin',checkpoint_id='fixed-gin')
    parent=SimpleNamespace(parent_id='p',label=1,smiles='CCC')
    def pred(_o,_f,smiles,pid,split):
        label=int(pid=='p');return {'predicted_label':label,'probabilities':[.1,.9] if label else [.8,.2],
            'logits':[0.,1.] if label else [1.,0.]}
    monkeypatch.setattr(e,'prediction',pred)
    return parent,oracle
def test_true_native_match_minimum_and_mapping(monkeypatch):
    p,o=fixture(monkeypatch);d=Distance()
    pairs,apps=e.evaluate_parent(p,[rule()],o,None,d,'calibration')
    assert len(pairs)==1 and len(apps)==4
    assert pairs[0]['pair_strict_flip'] and pairs[0]['wnode_distance']==.02
    assert pairs[0]['selected_match_index']==0
    assert all(a['native_mapping'] and a['residual_smiles']=='CCC' for a in apps)
def test_failed_distance_raises_not_zero(monkeypatch):
    p,o=fixture(monkeypatch)
    with pytest.raises(ValueError,match='RAW_DISTANCE_FAILURE_NOT_ZERO'):
        e.evaluate_parent(p,[rule()],o,None,Distance(False),'calibration')
def test_disconnected_is_rejection_not_padding_or_connection(monkeypatch):
    p,o=fixture(monkeypatch);p.smiles='CC';c=rule()
    r=GlobalGCENativeRule.from_payload(c['rule']);z=r.rhs_edge_attr.clone();z[0]=torch.tensor([1.,0.,0.,0.])
    c['rule']=replace(r,rhs_edge_attr=z,rhs_adjacency=torch.zeros_like(r.rhs_adjacency)).to_payload()
    d=Distance();pairs,apps=e.evaluate_parent(p,[c],o,None,d,'calibration')
    assert not pairs[0]['applicable'] and not d.calls
    assert all('DISCONNECTED' in a['failure_reason'] for a in apps)
def test_no_lhs_match_record_remains_unavailable(monkeypatch):
    p,o=fixture(monkeypatch);pairs,apps=e.evaluate_parent(p,[rule('CO')],o,None,Distance(),'calibration')
    assert apps==[] and pairs[0]['failure_reason']=='lhs_unmatched' and pairs[0]['wnode_distance'] is None
def test_test_runtime_stops_before_raw_source(monkeypatch):
    monkeypatch.setattr(e,'validate',lambda _: {})
    monkeypatch.setattr(e,'verified_freeze',lambda _: (_ for _ in ()).throw(ValueError('OWN_FREEZE')))
    monkeypatch.setattr(e,'bound',lambda _:pytest.fail('opened source before own freeze'))
    with pytest.raises(ValueError,match='OWN_FREEZE'):e.runtime({},'test')
def test_global_selector_requires_actual_original_proof(tmp_path):
    from src.experiments.bace_gin_native_baselines import select_order
    with pytest.raises(ValueError,match='CONTEXT_REQUIRED'):
        select_order(tmp_path,{'method':'globalgce','test_loaded':False})

def test_split_bytes_are_bound_not_just_count(tmp_path,monkeypatch):
    p=tmp_path/'cal.csv';p.write_text('changed')
    spec={'split_bindings':{'calibration':{'path':str(p),'sha256':'a'*64}}}
    monkeypatch.setattr(e,'bound',lambda _: {'splits':{'calibration':'data/cal.csv'},'files':{'data/cal.csv':{'sha256':'a'*64}}})
    spec['bundle_manifest']={}
    with pytest.raises(ValueError,match='FROZEN_SPLIT_CONTENT_CHANGED'):e.split_parents(spec,'calibration')

def test_test_split_hash_waits_for_own_freeze(monkeypatch):
    monkeypatch.setattr(e,'verified_freeze',lambda _: (_ for _ in ()).throw(ValueError('OWN_FREEZE')))
    monkeypatch.setattr(e,'bound',lambda _:pytest.fail('manifest read too early'))
    with pytest.raises(ValueError,match='OWN_FREEZE'):e.split_parents({},'test')

def test_partial_matrix_resume_checks_existing_contents(tmp_path):
    p=tmp_path/'rows.jsonl';v=[{'parent_id':'p'}]
    e.sealed_matrix_file(p,v,jsonl=True);e.sealed_matrix_file(p,v,jsonl=True)
    with pytest.raises(ValueError,match='SEALED_CALIBRATION_MATRIX_CHANGED'):
        e.sealed_matrix_file(p,[{'parent_id':'other'}],jsonl=True)

def test_freeze_order_tamper_rejected_before_test_sources(tmp_path,monkeypatch):
    f={'state':'FROZEN','ordered_rule_ids':['a','b']}
    f['self_sha256']=e.stable_sha256(f);f['ordered_rule_ids'].reverse()
    (tmp_path/'selection_freeze.json').write_text(json.dumps(f))
    monkeypatch.setattr(e,'pool',lambda _:pytest.fail('opened pool after invalid freeze'))
    with pytest.raises(ValueError,match='FREEZE_SEAL_CHANGED'):e.verified_freeze({'output_root':str(tmp_path)})

def test_cpu_handoff_waits_for_exact_owner_terminal(tmp_path,monkeypatch):
    from src.baselines.bace_globalgce_aplus_owner import cpu_predecessor_state
    config={'owner_root':str(tmp_path),'training_summary':'summary'}
    monkeypatch.setattr(e,'validate',lambda _:config)
    monkeypatch.setattr(e,'bound',lambda _:{'owner_root':str(tmp_path),'training_contract':'/contract'})
    monkeypatch.setattr(e,'pool',lambda _:pytest.fail('pool before producer terminal'))
    spec={'predecessor_owner_spec':{},'training_contract':{'path':'/contract'}}
    assert cpu_predecessor_state(spec)=='WAITING_TRAINING_AND_POOL_FREEZE'
    (tmp_path/'terminal.json').write_text(json.dumps({'state':'FAILED_ENGINEERING'}))
    with pytest.raises(ValueError,match='NOT_SCIENTIFICALLY_COMPLETE'):cpu_predecessor_state(spec)

def test_global_at_most_budget_preserves_real_prefix(tmp_path,monkeypatch):
    from src.experiments.bace_gin_native_baselines import select_order
    from src.eval import mutagenicity_wnode_selector as s
    calls=[]
    def run(**kw):
        calls.append(kw);out=kw['output_dir'];out.mkdir()
        (out/'variants/A1').mkdir(parents=True)
        (out/'calibration_decision.json').write_text(json.dumps({'selected_variant':'A1','decision_rule':'fixed'}))
        (out/'variants/A1/selected_top20.json').write_text(json.dumps({'candidates':[{'candidate_id':f'r{i}'} for i in range(kw['top_k'])]}))
    monkeypatch.setattr(s,'run_mutagenicity_wnode_selector',run)
    monkeypatch.setattr(s,'threshold_bundle_from_dict',lambda v:v)
    cfg={'top_k':20,'table_k':10,'seed':13,'local_swap_passes':2,'parent_limit':0,'candidate_limit':0,'forbid_test':True,'prefix_weights':[1.]*10+[.5]*10}
    result=select_order(tmp_path,{'method':'globalgce','native_attachment_contract':SCHEMA,
        'original_global_selector_verified':True,'test_loaded':False,'available_rule_count':15,
        'rule_budget_semantics':'AT_MOST_K','original_selector_config':cfg,'output_root':str(tmp_path/'selector'),
        'thresholds':{},'threshold_provenance':{}})
    assert result['effective_top_k']==15 and len(result['ordered_rule_ids'])==15
    assert calls[0]['top_k']==15 and calls[0]['prefix_weights']==[1.]*10+[.5]*5

def test_deferred_test_raw_binding_waits_and_rejects_later_change(tmp_path,monkeypatch):
    p=tmp_path/'old_test.json';p.write_text(json.dumps({'split':'test'}))
    (tmp_path/'selection_freeze.json').write_text('frozen')
    spec={'output_root':str(tmp_path),'raw_cost_indexes':{'test':{'path':str(p),
        'binding_policy':'FIRST_READ_AND_SEAL_AFTER_OWN_SELECTOR_FREEZE'}}}
    monkeypatch.setattr(e,'verified_freeze',lambda _: (_ for _ in ()).throw(ValueError('OWN_FREEZE')))
    with pytest.raises(ValueError,match='OWN_FREEZE'):e.raw_index(spec,'test')
    monkeypatch.setattr(e,'verified_freeze',lambda _: {'state':'FROZEN'})
    index,receipt=e.raw_index(spec,'test')
    assert receipt['source_file_hash_previously_bound'] is False
    assert not (tmp_path/'test_raw_adoption.json').exists() # only runtime after kernel validation seals
    e.atomic_json(tmp_path/'test_raw_adoption.json',receipt)
    assert e.raw_index(spec,'test')[1]==receipt
    p.write_text(json.dumps({'split':'test','changed':True}))
    with pytest.raises(ValueError,match='SEALED_TEST_RAW_ADOPTION_CHANGED'):e.raw_index(spec,'test')
