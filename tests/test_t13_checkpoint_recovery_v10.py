import copy
import hashlib
from pathlib import Path
import pytest
from src.baselines import globalgce_resumable as native
from src.baselines.t13_checkpoint_recovery_v10 import stable_checkpoint_bytes,checkpoint_io_scope,validate_state


def test_content_binding(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'checkpoint')
    sha=hashlib.sha256(b'checkpoint').hexdigest()
    assert stable_checkpoint_bytes(p,expected_sha256=sha)[0]==b'checkpoint'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='CONTENT_BINDING'):stable_checkpoint_bytes(p,expected_sha256=sha)


def test_only_named_ctime_order_is_relaxed(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_bytes(b'checkpoint')
    old=native._named_checkpoint_stat
    def lag(path):return dict(old(path),ctime_ns=1)
    monkeypatch.setattr(native,'_named_checkpoint_stat',lag)
    assert stable_checkpoint_bytes(p)[0]==b'checkpoint'
    with pytest.raises(RuntimeError,match='ctime moved backwards'):native._open_regular_file_evidence(p)


def test_atomic_path_switch_rejected(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_bytes(b'checkpoint')
    old=native._named_checkpoint_stat
    def switch(path):
        q=tmp_path/'new';q.write_bytes(b'checkpoint');q.replace(p)
        return old(path)
    monkeypatch.setattr(native,'_named_checkpoint_stat',switch)
    with pytest.raises(ValueError,match='PATH_VERSION_CHANGED'):stable_checkpoint_bytes(p)


def test_mutation_rejected(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_bytes(b'checkpoint')
    old=native._regular_fd_evidence;calls=[]
    def mutate(fd):
        calls.append(1)
        if len(calls)==2:p.write_bytes(b'changed')
        return old(fd)
    monkeypatch.setattr(native,'_regular_fd_evidence',mutate)
    with pytest.raises(ValueError,match='READ_MUTATION'):stable_checkpoint_bytes(p)


def test_scope_restored(tmp_path):
    old=native._open_regular_file_evidence
    with checkpoint_io_scope(tmp_path):assert native._open_regular_file_evidence is not old
    assert native._open_regular_file_evidence is old


def state_and_binding():
    state=dict(checkpoint_schema_version='globalgce_epoch_checkpoint_v2',model_state={},optimizer_state={'state':{0:{'step':77}}},scheduler_state={'last_epoch':77},python_rng_state=(1,),numpy_rng_state={'x':1},torch_rng_state=[1],cuda_rng_state=[[1]],sampler_state={'next_epoch':77,'next_batch':0},augmented_dataset_identity={'identity_sha256':'compact'},resume_identity={'source_label':1,'target_label':0,'training_config':{'epochs':100}},resume_identity_sha256='resume',next_epoch=77,best_loss=1.,best_state_seen=True)
    binding=dict(original_formal_attempt_id='7b647a43-1919-4d8e-a05a-0f6071255f2e',formal_quota='1/1',target=0,completed_epoch=76,next_epoch=77,resume_identity_sha256='resume',compact_identity_sha256='compact',producer_phase='AFTER_OPTIMIZER_SCHEDULER_AND_DUE_VALIDATION',historical_callback_rewritten=False,producer_evidence=[{'historical_callback_epoch':75}])
    return state,binding


def test_callback_native_difference_requires_explicit_phase():
    state,binding=state_and_binding();validate_state(state,binding,tensor_finite=lambda x:True)
    binding['producer_phase']='HEARTBEAT_ONLY'
    with pytest.raises(ValueError,match='PRODUCER_PHASE'):validate_state(state,binding,tensor_finite=lambda x:True)


@pytest.mark.parametrize('key',['model_state','optimizer_state','scheduler_state','sampler_state','cuda_rng_state'])
def test_missing_state_rejected(key):
    state,binding=state_and_binding();del state[key]
    with pytest.raises(ValueError,match='INCOMPLETE'):validate_state(state,binding,tensor_finite=lambda x:True)
