import numpy as np
import pytest
from src.baselines.cm_crem_postfilter import decode_status, check_cohort, commit_npz
from src.baselines.cm_crem_k20 import freeze_k20
from src.baselines.cm_crem_selection import evaluate_frozen_test


def test_unknown_is_not_zero_or_infinity():
    for val in (0.0,np.inf,np.nan):
        with pytest.raises(ValueError):decode_status(np.array([[val]]),np.array([[0]]))
    assert decode_status(np.array([[np.inf]]),np.array([[2]]))[0,0]=='BEFORE_NOT_SOURCE'


def test_empty_pool_is_valid_zero_not_crash():
    x=np.empty((2,0));s=np.empty((2,0),dtype=str)
    f,_=freeze_k20(x,s,['a','b'],[],np.array([True,False]),.1,.2,[.1,.2],'a'*64,'b'*64)
    e=evaluate_frozen_test(f,x,pair_status=s,parent_ids=['c','d'],candidate_ids=[],source_mask=np.array([True,False]),contract_sha256='a'*64)
    assert e.prefix_metrics()[-1]['coverage']==0
    assert e.prefix_metrics()[-1]['conditional_median_cost'] is None


def test_three_class_fixed_base_and_at_most_k():
    x=np.array([[.1,.2],[np.inf,np.inf],[.3,.05]])
    s=np.array([['OK','OK'],['BEFORE_NOT_SOURCE']*2,['OK','OK']])
    f,_=freeze_k20(x,s,['a','b','c'],['u','v'],np.array([True,False,True]),.15,.4,[.15,.4],'a'*64,'b'*64)
    cols=[['u','v'].index(i) for i in f.selected_candidate_ids]
    e=evaluate_frozen_test(f,x[:,cols],pair_status=s[:,cols],parent_ids=['x','y','z'],candidate_ids=f.selected_candidate_ids,source_mask=np.array([True,False,True]),contract_sha256='a'*64)
    assert e.prefix_metrics()[-1]['covered_count']==2
    assert e.prefix_metrics()[-1]['base_parent_count']==3
    assert e.prefix_metrics()[-1]['effective_k']==2


def test_bound_cohort_no_predicate_invented():
    rows=[dict(id='b',s='C',label='1'),dict(id='a',s='O',label='0')]
    b=dict(id_field='id',smiles_field='s',count=2)
    assert [p['parent_id'] for p in check_cohort(rows,b,'test')]==['b','a']
    with pytest.raises(ValueError):check_cohort(rows,{**b,'count':1},'test')


def test_npz_resume_not_overwrite(tmp_path):
    path=tmp_path/'b.npz';commit_npz(path,values=np.array([1.]))
    commit_npz(path,values=np.array([1.]))
    with pytest.raises(ValueError):commit_npz(path,values=np.array([2.]))
