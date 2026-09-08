import copy
from pathlib import Path
import pytest
from src.utils.global_cpu_closeout import validate_spec_change, checked_identity


def test_resource_owner_location_only_science_unchanged():
    old={'main_matrix_write':False,'cpu_handoff_root':'/campaign/old','pool':'sealed80','selected_epoch':60}
    new={**old,'cpu_handoff_root':'/campaign/new','cpu_resource_config':{'new':True}}
    validate_spec_change(old,new)
    assert old['cpu_handoff_root']=='/campaign/old'
    for k,v in [('pool','other'),('selected_epoch',100),('main_matrix_write',True)]:
        with pytest.raises(ValueError):validate_spec_change(old,{**new,k:v})


def test_no_cross_namespace_handoff():
    old={'main_matrix_write':False,'cpu_handoff_root':'/campaign/old'}
    with pytest.raises(ValueError):validate_spec_change(old,{**old,'cpu_handoff_root':'/other/new'})


def test_pid_reuse_refused(tmp_path):
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):
        checked_identity({'pid':447477,'start_ticks':54199941},tmp_path)


def test_source_has_precise_one_shot_boundary_and_no_killall():
    text=(Path(__file__).resolve().parents[1]/'src/utils/global_cpu_closeout.py').read_text()
    for token in ['ONE_SHOT_INTENT_EXISTS','OLD_OWNER_NOT_CHILD_FREE_WAITING','OWNER_CAS_CHANGED',
                  'OTHER_CAMPAIGN_PROCESS','LIVE_CPU_RESOURCE_NOT_ADMITTED','signal.SIGTERM']:
        assert token in text
    assert 'SIGKILL' not in text.replace('OLD_OWNER_DID_NOT_EXIT_NO_SIGKILL','')
    assert 'main_matrix_write' in text
    assert 'CUDA_VISIBLE_DEVICES=\'\'' in text


def test_pilot_does_not_open_test_or_compute_ot():
    text=(Path(__file__).resolve().parents[1]/'src/utils/global_cpu_closeout.py').read_text()
    assert "'test_loaded':False,'ot_computed':False" in text
    assert "leaf.split_parents(probe, 'calibration')" in text
    assert 'selected_epoch' in text
    assert '384*1024**3+peak' in text
