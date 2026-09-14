import importlib.util
from pathlib import Path
import pytest
from src.baselines.cm_crem_descriptive import selection_ids,validate_scope,EXPERIMENT

def case():
    g=[f'p{i}' for i in range(1097)];e=g+[f'n{i}' for i in range(186)];c=selection_ids(g)
    scope=dict(experiment_id=EXPERIMENT,heldout=False,selection_ids=c,evaluation_ids=e,
        generation_count=1097,selection_count=220,evaluation_count=1283,
        selection_evaluation_overlap=220,generation_evaluation_overlap=1097,
        generated_again=False,filter_repeated=False,original_deadline_utc='2026-09-16T16:27:44Z')
    return scope,{'generation_ids':g},c,e

def test_hash_subset_is_fixed_not_outcome_or_input_order():
    s,p,c,e=case(); assert selection_ids(list(reversed(p['generation_ids'])))==c
    assert len(c)==220 and len(set(c))==220
    assert validate_scope(s,p,c,e)['main_scope_compatible'] is False

@pytest.mark.parametrize('field,value',[('heldout',True),('selection_evaluation_overlap',0),('generated_again',True),('original_deadline_utc','2030-01-01T00:00:00Z')])
def test_wrong_claim_rejected(field,value):
    s,p,c,e=case();s[field]=value
    with pytest.raises(ValueError):validate_scope(s,p,c,e)

def test_wrong_order_or_dropped_nonsource_rejected():
    s,p,c,e=case()
    with pytest.raises(ValueError):validate_scope(s,p,c[::-1],e)
    with pytest.raises(ValueError):validate_scope(s,p,c,e[:1097])

def test_aids_existing_relay_scope_and_deadline():
    path=Path(__file__).parents[1]/'scripts/run_cm_crem_relay.py'
    module_spec=importlib.util.spec_from_file_location('descriptive_relay_test',path)
    r=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(r)
    identity=dict(start_time_utc='2026-09-09T16:27:44Z',planning_deadline_utc='2026-09-16T16:27:44Z')
    plan=dict(schema='cm_postfilter_existing_relay_delivery_v1',deadline_utc=identity['planning_deadline_utc'],start_time_utc=identity['start_time_utc'],datasets=[dict(dataset='aids',oracle='rf',scope=EXPERIMENT,heldout=False,package_job_id='1234',hpc_root='/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2/aids-gap-first-20260914/fixture',local_root='/Volumes/DireRaven/counterfactual-hpc-offload/cm-aids-k20-closeout-20260914/fixture')])
    assert r.validate_postfilter_delivery(plan,identity)==plan
    plan['datasets'][0]['heldout']=True
    with pytest.raises(ValueError):r.validate_postfilter_delivery(plan,identity)
