from copy import deepcopy
import pytest
from src.experiments.bace_gin_aplus_funnel import summarize_scope,summarize_many,first_failure

def row(pid,cid,**kw):
    return dict(parent_id=pid,candidate_id=cid,split='calibration',pred_before=1,pred_after=0,
        applicable=True,num_matches=2,num_valid_residuals=1,num_strict_flip_matches=1,
        pair_strict_flip=True,wnode_distance=.02,**kw)

def run(units,**kw):
    return summarize_scope(units,scope='fixture',split='calibration',order=None,theta=.01,cap=.03,
        expected_parents=len(units),expected_candidates=2,**kw)

def test_complete_layer_counts_and_threshold_decomposition():
    units=[{'parent_id':'p','pair_rows':[row('p','a'),row('p','b')]}]
    units[0]['pair_rows'][1]['wnode_distance']=.005
    m,f,p=run(units)
    assert m['pair_count']==2 and m['source_parents']==1
    assert m['num_matches']==4 and m['num_valid_residuals']==2 and m['strict_flip_matches']==2
    assert m['finite_pairs']==2 and m['theta_pairs']==1 and m['cap_pairs']==2
    assert m['theta_parents']==1 and p[0]['first_failure']=='COVERED'
    assert sum(x['pair_count'] for x in f)==2 and sum(x['parent_count'] for x in f)==1

@pytest.mark.parametrize('args,expected',[
    ((0,0,0,0,[],.01),'SOURCE_NOT_1'),((1,0,0,0,[],.01),'NO_MATCH'),
    ((1,2,0,0,[],.01),'NO_VALID_RESIDUAL'),((1,2,1,0,[],.01),'NO_STRICT_FLIP'),
    ((1,2,1,1,[],.01),'NO_FINITE_DISTANCE'),((1,2,1,1,[.02],.01),'THETA_EXCEEDED'),
    ((1,2,1,1,[.005],.01),'COVERED'),((1,None,0,0,[],.01),'UNKNOWN')])
def test_first_failure_is_ordered(args,expected):assert first_failure(*args)==expected

def test_unknown_count_not_filled_as_zero():
    rows=[row('p','a'),row('p','b')];rows[0].pop('num_matches');rows[0]['wnode_distance']=None
    m,f,p=run([{'parent_id':'p','pair_rows':rows}])
    assert m['num_matches'] is None and m['num_matches_known_sum']==2 and m['num_matches_unknown_count']==1
    assert m['finite_pairs'] is None and m['finite_pairs_known_sum']==1
    assert p[0]['first_failure']=='UNKNOWN'

def test_candidate_subset_is_exact_scope_not_full_pool():
    rows=[row('p','a'),row('p','b')];rows[0]['wnode_distance']=.001
    m,_,p=summarize_scope([{'parent_id':'p','pair_rows':rows}],scope='control_K10',split='calibration',
        order=['b'],theta=.01,cap=.03,expected_parents=1,expected_candidates=1)
    assert m['pair_count']==1 and m['theta_pairs']==0 and p[0]['first_failure']=='THETA_EXCEEDED'

def test_missing_duplicate_and_drift_rejected():
    units=[{'parent_id':'p','pair_rows':[row('p','a'),row('p','a')]}]
    with pytest.raises(ValueError,match='DUPLICATE_SAVED_PAIR'):run(units)
    units=[{'parent_id':'p','pair_rows':[row('p','a'),row('p','b')]},
           {'parent_id':'q','pair_rows':[row('q','a'),row('q','c')]}]
    with pytest.raises(ValueError,match='POOL_CHANGED'):run(units)

def test_multiple_controls_stream_parent_records_once():
    calls=[]
    def units():
        for p in ('p','q'):
            calls.append(p);yield {'parent_id':p,'pair_rows':[row(p,'a'),row(p,'b')]}
    m,f,p=summarize_many(units(),[('a',['a'],1),('all',['a','b'],2)],split='calibration',
        theta=.01,cap=.03,expected_parents=2)
    assert calls==['p','q'] and [x['pair_count'] for x in m]==[2,4]
    assert [x['num_matches'] for x in m]==[4,8] and len(p)==4

def test_test_files_not_iterated_without_actual_freeze(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.experiments import bace_gin_aplus_funnel as f,bace_gin_reach_v2 as d,bace_gin_reach_test_raw as raw,bace_gin_fixed_pool as fixed
    from src.eval import mutagenicity_wnode_selector as selector
    spec={'output_root':str(tmp_path),'base_counts':{'calibration':1,'test':1},'thresholds':{}}
    (tmp_path/'adopted2607/test').mkdir(parents=True);(tmp_path/'adopted2607/test/terminal.json').touch()
    def verified(p):
        if p.name=='contract.json':return {'spec_sha256':f.stable_sha256(spec),'adopted_candidate_count':2}
        if 'calibration' in str(p):return {'state':'PARENT_EVALUATION_COMPLETE','spec_sha256':f.stable_sha256(spec),'parent_count':1,'candidate_count':2}
        if p.name=='selection_freeze.json':return {'invalid':'fixture'}
        pytest.fail('test terminal touched before actual freeze validator')
    monkeypatch.setattr(d,'verified',verified)
    monkeypatch.setattr(d,'_iter_units',lambda *a:pytest.fail('parent file iterated before freeze'))
    monkeypatch.setattr(raw,'validate_aplus_freeze',lambda *a,**k:(_ for _ in ()).throw(ValueError('ACTUAL_FREEZE')))
    monkeypatch.setattr(fixed,'bound_json',lambda _: {})
    monkeypatch.setattr(selector,'threshold_bundle_from_dict',lambda _:SimpleNamespace(theta_star=.01,cost_cap=.03))
    with pytest.raises(ValueError,match='ACTUAL_FREEZE'):f.export(spec,tmp_path/'fresh')
