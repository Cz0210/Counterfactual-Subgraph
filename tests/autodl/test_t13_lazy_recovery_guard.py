import json

import pytest

from src.utils.t13_lazy_recovery_guard import T13LazyRecoveryGuard, GIB


def guard(tmp_path):
    obj=object.__new__(T13LazyRecoveryGuard)
    obj.owner=tmp_path/'owner'; obj.owner.mkdir()
    obj.canary=obj.owner/'lazy-canary'; obj.canary.mkdir()
    obj.path=tmp_path/'control'/'repair'/'authorization.json'
    obj.path.parent.mkdir(parents=True)
    obj.runtime=tmp_path
    obj.spec={'attempt_id':'fresh','task_id':'t13','output_root':str(tmp_path/'science'),
              'task_spec_sha256':'spec'}
    obj.authorization_sha256='authorization'; obj.peak=2*GIB; obj.samples=3
    obj.min_headroom=400*GIB; obj.baseline={'failcnt':9}
    return obj


def receipt(obj, **changes):
    row=dict(state='T13_INDEXED_DATA_AND_SHORT_TRAINING_CANARY_PASS',
             targets_order=[0,2],seed=7,configured_epochs=100,
             test_loaded=False,calibration_loaded=False,mining_recomputed=False,
             independent_reload_pass=True,index_contract_pass=True,
             mask_rng_batch_parity=True,training_step_parity=True,reload_parity=True)
    row.update(changes)
    (obj.canary/'canary.json').write_text(json.dumps(row))
    (obj.canary/'memory_samples.json').write_text(json.dumps({'samples':[{'VmHWM_bytes':3*GIB}]}))


def test_complete_canary_claims_only_one_full_start(tmp_path,monkeypatch):
    obj=guard(tmp_path); receipt(obj)
    monkeypatch.setattr('src.utils.t13_lazy_recovery_guard.resources',lambda _:dict(headroom_bytes=450*GIB,failcnt=9))
    result=obj.accept_canary_and_claim_full()
    assert result['full_trajectory_parity_claimed'] is False
    assert result['process_tree_peak_bytes']==3*GIB
    assert (obj.path.parent/'full_start.json').is_file()
    with pytest.raises(Exception): obj.accept_canary_and_claim_full()


@pytest.mark.parametrize('key',['reload_parity','mask_rng_batch_parity','training_step_parity','index_contract_pass'])
def test_missing_gate_cannot_consume_formal_attempt(tmp_path,monkeypatch,key):
    obj=guard(tmp_path); receipt(obj,**{key:False})
    with pytest.raises(ValueError,match='CANARY_GATE_MISSING'): obj.accept_canary_and_claim_full()
    assert not (obj.path.parent/'full_start.json').exists()


def test_memory_gate_is_not_relaxed(tmp_path,monkeypatch):
    obj=guard(tmp_path); receipt(obj)
    monkeypatch.setattr('src.utils.t13_lazy_recovery_guard.resources',lambda _:dict(headroom_bytes=383*GIB,failcnt=9))
    with pytest.raises(ValueError,match='FULL_PEAK_ADMISSION_FAILED'): obj.accept_canary_and_claim_full()
    assert not (obj.path.parent/'full_start.json').exists()


def test_child_uses_original_train_and_import_no_test(tmp_path):
    obj=guard(tmp_path)
    obj.spec.update(python='/python',repo_root='/code',required_import_root='/sealed-import',
                    input_paths={'official_root':'/official','train_csv':'/train.csv','gnn_checkpoint':'/gine'})
    command=obj.command()
    assert '/sealed-import/adoption_proof.json' in command
    assert command[-4:]==['--device','cuda:0','--targets','0,2']
    assert '--test-csv' not in command


@pytest.mark.parametrize('sample',[
    {'VmHWM_bytes':97*GIB},
    {'memory.limit_in_bytes':480*GIB,'memory.usage_in_bytes':300*GIB},
    {'memory.failcnt':10},
])
def test_transient_boundary_pressure_blocks_full(tmp_path,sample):
    obj=guard(tmp_path); receipt(obj)
    (obj.canary/'memory_samples.json').write_text(json.dumps({'samples':[sample]}))
    with pytest.raises(ValueError,match='TRANSIENT'): obj.accept_canary_and_claim_full()
    assert not (obj.path.parent/'full_start.json').exists()
