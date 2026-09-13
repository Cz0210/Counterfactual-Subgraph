import copy,importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_cm_crem_relay.py'
spec=importlib.util.spec_from_file_location('cm_postfilter_existing_relay',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def plan():
 return dict(schema='cm_postfilter_existing_relay_delivery_v1',deadline_utc='2026-09-16T16:27:44Z',start_time_utc='2026-09-09T16:27:44Z',datasets=[dict(dataset='tastemolnet',oracle='gine',package_job_id='2688519',hpc_root='/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2/postfilter-20260914/tastemolnet-production',local_root='/Volumes/DireRaven/counterfactual-hpc-offload/cm-crem-global-v2/tastemolnet-postfilter-20260914')])

def test_same_original_deadline_and_actual_successor():
 p=plan();assert m.validate_postfilter_delivery(p,dict(planning_deadline_utc=p['deadline_utc'],start_time_utc=p['start_time_utc'])) is p

@pytest.mark.parametrize('field,value',[('deadline_utc','2026-09-18T16:27:44Z'),('start_time_utc','2026-09-14T00:00:00Z')])
def test_no_deadline_reset(field,value):
 p=plan();old=dict(planning_deadline_utc=p['deadline_utc'],start_time_utc=p['start_time_utc']);p[field]=value
 with pytest.raises(ValueError,match='deadline'):m.validate_postfilter_delivery(p,old)

def test_no_remote_escape_or_duplicate_dataset():
 p=plan();old=dict(planning_deadline_utc=p['deadline_utc'],start_time_utc=p['start_time_utc']);p['datasets'][0]['hpc_root']='/tmp/escape'
 with pytest.raises(ValueError,match='outside'):m.validate_postfilter_delivery(p,old)
 p=plan();p['datasets']*=2
 with pytest.raises(ValueError,match='unique'):m.validate_postfilter_delivery(p,old)
