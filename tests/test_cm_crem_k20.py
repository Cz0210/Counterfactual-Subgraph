import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock


def test_completion_receipt_resume_does_not_read_source_or_compute(tmp_path):
    from src.baselines.cm_crem_k20_experiment import K20Experiment
    x=K20Experiment.__new__(K20Experiment);x.root=tmp_path
    (tmp_path/'calibration').mkdir();(tmp_path/'calibration/complete.json').write_text('{}')
    x.get=Mock(return_value={'parents':66});x.oldread=Mock(side_effect=AssertionError('must not read'))
    assert x.matrix_stage('calibration')=={'parents':66}
    x.oldread.assert_not_called()


def test_complete_5474_pool_reuses_saved_records_only():
    from src.baselines.cm_crem_k20_experiment import K20Experiment
    from src.baselines.cm_crem_runtime import digest
    x=K20Experiment.__new__(K20Experiment)
    class Root:
        def __truediv__(self,name):return SimpleNamespace(exists=lambda:False)
    x.root=Root();x.old=SimpleNamespace(sha='a'*64);x.spec={'resolved_oracle':{'model_sha256':'b'*64}}
    ids=[str(i) for i in range(5474)];ids.sort(key=lambda c:(digest([x.old.sha,7,c]),c))
    name='filter_units/'+digest('p')[:20]+'.json';x.pins={name:'c'*64}
    records={'attribution.json':{'parents':[{'parent_id':'p'}]},
      name:{'status':'FILTER_COMPLETE','source_prediction':{'predicted_label':1},'accepted':[
          {'candidate_id':cid,'predicted_label':0,'oracle_weight_sha256':'b'*64} for cid in ids]},
      'pool_freeze.json':{'candidate_ids':ids[:2000]}}
    x.oldread=lambda name:records[name];x.put=lambda name,data:data
    result=x.prepare();assert len(result['ids'])==5474 and result['generation_calls']==0
import pytest
from src.baselines.cm_crem_k20 import objective,optimize,freeze_k20
from src.baselines.cm_crem_selection import evaluate_frozen_test


def test_unknown_not_zero():
    with pytest.raises(ValueError):optimize(np.array([[np.nan]]),['a'],.1,1,[0,.1,1])


def test_zero_coverage_still_fills_and_is_deterministic():
    d=np.arange(90,dtype=float).reshape(3,30)+2
    ids=[f'{i:03d}' for i in range(30)]
    s,r=optimize(d,ids,.1,20,[0,.1,1,20])
    assert len(s)==20 and s==optimize(d,ids,.1,20,[0,.1,1,20])[0]
    assert r['accepted_swap_count']<=2 and r['objective'][0]==0


def test_incumbent_never_worse_and_small_pool_at_most():
    d=np.random.default_rng(7).random((8,24));ids=list(map(str,range(24)))
    old=ids[:20];s,r=optimize(d,ids,.2,1,[0,.2,.5,1],old)
    assert r['objective']>=objective(d[:,:20].min(axis=1),.2,1,np.array([0,.2,.5,1]))
    small,_=optimize(d[:,:3],ids[:3],.2,1,[0,.2,.5,1]);assert len(small)==3


def test_variant_bound_freeze_and_test_twenty_only():
    d=np.array([[.2,.1],[np.inf,np.inf]])
    f,r=freeze_k20(d,[['OK','OK'],['BEFORE_NOT_SOURCE']*2],['p','q'],['a','b'],
        np.array([True,False]),.15,1,[0,.15,1],'a'*64,'b'*64)
    assert f.method_id=='CM-Global-K20-v2'
    order=[['a','b'].index(x) for x in f.selected_candidate_ids]
    out=evaluate_frozen_test(f,d[:,order],pair_status=np.array([['OK','OK'],['BEFORE_NOT_SOURCE']*2])[:,order],
        parent_ids=['t','u'],candidate_ids=f.selected_candidate_ids,source_mask=np.array([True,False]),contract_sha256='a'*64)
    assert out.prefix_metrics()[-1]['effective_k']==2
    assert out.prefix_metrics()[-1]['coverage']==.5
