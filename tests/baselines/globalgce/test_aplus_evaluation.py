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
