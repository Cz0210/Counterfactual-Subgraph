import json,hashlib
import pytest
from src.utils.t13_gap_recovery_owner import stage_memory_policy,FORMAL_SCHEMA,GIB

def save(tmp,name,value):
    p=tmp/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def spec(tmp):
    p=dict(schema='T13_V5_STAGE_INCREMENT_V1',formal_quota='1/1',gpu_uuid='test-uuid',safety_margin_bytes=64*GIB,
        process_peak_bound_bytes=32*GIB,maximum_retained_train_batches=5,
        adopted_probe_memory=save(tmp,'probe.json',dict(samples=[dict(VmHWM_bytes=5310369792)])),
        observed_probe_host_peak_bytes=5310369792,
        retired_default_reason='UNIDENTIFIED_FIXED_HEADROOM_NOT_EXTERNAL_TASK_INCREMENT',concurrent_future_increments=[])
    return p,dict(schema=FORMAL_SCHEMA,gpu_uuid='test-uuid',stage_resource_policy=save(tmp,'policy.json',p))

def test_observed_envelope_not_400_minus_384(tmp_path):
    p,s=spec(tmp_path);q=stage_memory_policy(s)
    assert q['process_peak_bound_bytes']==32*GIB and q['safety_margin_bytes']==64*GIB

def test_unknown_not_zero(tmp_path):
    p,s=spec(tmp_path);p['concurrent_future_increments']=[dict(task_id='other',additional_bytes=None,evidence='unknown')]
    s['stage_resource_policy']=save(tmp_path,'policy2.json',p)
    with pytest.raises(ValueError,match='UNKNOWN_CONCURRENT'):stage_memory_policy(s)

def test_old_mode_keeps_original_gate():
    with pytest.raises(ValueError):stage_memory_policy(dict(process_peak_bound_bytes=16*GIB,other_remaining_reserve_bytes=0))
