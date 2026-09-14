import json
import pytest
from src.baselines.t13_compact_continuation import validate_recovery

def test_formal_only_adopts_original_checkpoint_not_probe_updates(tmp_path):
    p=tmp_path/'probe.json';p.write_text(json.dumps(dict(state='REAL_GPU_TWO_TRAIN_ONE_VALIDATION_RELOAD_PASS',formal_updates=0)))
    plan=dict(gpu_probe_receipt=str(p),formal_quota_used='1/1',target_order=[0,2])
    ck=dict(next_epoch=30,model_state={},optimizer_state={})
    validate_recovery(plan,ck)
    for bad in [dict(ck,next_epoch=31),dict(next_epoch=30)]:
        with pytest.raises(ValueError):validate_recovery(plan,bad)
    p.write_text(json.dumps(dict(state='REAL_GPU_TWO_TRAIN_ONE_VALIDATION_RELOAD_PASS',formal_updates=2)))
    with pytest.raises(ValueError):validate_recovery(plan,ck)

def test_no_new_optimizer_loop_or_validation_skip():
    import inspect
    from src.baselines.t13_compact_continuation import continue_branches
    source=inspect.getsource(continue_branches)
    assert 'return original(**kw)' in source
    assert 'for target in [0,2]' in source
    assert 'optimizer.step(' not in source
    assert 'EXISTING_TASTE_FULL_MERGE_CALIBRATION_TEST' in source
