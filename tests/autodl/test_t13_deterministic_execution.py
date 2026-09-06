import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.utils import t13_deterministic_execution as d
from src.utils.t13_lazy_recovery_guard import T13LazyRecoveryGuard, GIB


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return d.file_sha(path)


def evidence(tmp_path):
    root=tmp_path/'owner/lazy-canary/matched-deterministic'
    put(root.parent/'canary.json',dict(state='T13_COMPONENT_DIAGNOSTIC_FAILED',failure={'eager_self_repeatable':False}))
    report=dict(state='T13_MATCHED_DETERMINISTIC_DIAGNOSTIC_PASS',targets_order=[0,2],seed=7,
        configured_epochs=100,diagnostic_profile='deterministic',test_loaded=False,calibration_loaded=False,
        mining_recomputed=False,full_successor_started=False,full_trajectory_parity_claimed=False,
        train_only=True,independent_reload_pass=True,index_contract_pass=True,mask_rng_batch_parity=True,
        training_step_parity=True,reload_parity=True,targets={})
    mem={'samples':[{'VmHWM_bytes':23*GIB,'memory.limit_in_bytes':480*GIB,
        'memory.usage_in_bytes':60*GIB,'memory.failcnt':4306}]}
    report['memory_samples_sha256']=put(root/'memory_samples.json',mem)
    for t in [0,2]:
        branch=root/f'target_{t}'
        comp={'exact':True,'tolerance_used':False}
        names=['before','batches','rules','losses','gradients','rng_after_rules','rng_after_backward','after_update']
        diag=dict(state='T13_COMPONENT_DIAGNOSTIC_PASS',numeric_contract=d.BACKEND,
            eager_repetitions_completed=3,eager_self_repeatable=True,tolerance_used=False,warn_only_accepted=False,
            first_difference=None,execution_error=None,
            initial_state_bindings=[dict(arm=arm,state_sha256='initial') for arm in ['eager_0','eager_1','eager_2','lazy']],
            comparisons=[dict(kind=k,epoch=e,exact=True,components={n:comp for n in names})
                for k,e in [('EAGER_SELF_CONTROL',0),('EAGER_SELF_CONTROL',0),('EAGER_LAZY',0),('EAGER_LAZY',1)]])
        sha=put(branch/'training_canary/component_diagnostics.json',diag)
        reload=dict(state='PASS',checkpoint_sha256=f'checkpoint-{t}')
        put(branch/'training_canary/checkpoint_reload.json',reload)
        put(branch/'independent_reload/verification.json',reload)
        put(branch/'training_canary/memory_boundaries.json',mem)
        report['targets'][str(t)]=dict(component_diagnostics_sha256=sha,numeric_contract=d.BACKEND,
            optimizer_updates_per_arm=2,eager_repetitions=3,eager_lazy_batch_exact=True,forward_loss_exact=True,
            model_optimizer_scheduler_rng_exact=True,checkpoint_reload_exact=True,checkpoint_sha256=f'checkpoint-{t}',
            component_evidence_sha256='existing-receipt',dataset_identity=dict(index_sha256='index',masks_sha256='mask',
                sample_count=10,materialization_rng_unchanged=True,all_masks_reconstructed_exactly=True,
                sampler={'num_workers':0}))
    put(root/'canary.json',report)
    return root


def test_complete_deterministic_evidence_reused_without_checkpoint_load(tmp_path):
    root=evidence(tmp_path); before=(root.parent/'canary.json').read_bytes()
    result=d.inspect_evidence(root)
    assert result['backend']['cudnn_allow_tf32'] is True
    assert result['process_peak_bytes']==23*GIB
    assert set(result['targets'])=={'0','2'}
    assert before==(root.parent/'canary.json').read_bytes()
    assert not list(root.rglob('*.pt')) # No checkpoint reload or rehash during adoption.


@pytest.mark.parametrize('change',['short_controls','missing_gradient','tf32','bad_reload','scope'])
def test_no_pass_string_only_adoption(tmp_path,change):
    root=evidence(tmp_path); report=d.read(root/'canary.json')
    diagpath=root/'target_0/training_canary/component_diagnostics.json';diag=d.read(diagpath)
    if change=='short_controls':diag['comparisons'].pop(0)
    if change=='missing_gradient':diag['comparisons'][2]['components'].pop('gradients')
    if change=='tf32':diag['numeric_contract']['cudnn_allow_tf32']=False
    if change=='bad_reload':put(root/'target_0/independent_reload/verification.json',{'state':'PASS','checkpoint_sha256':'wrong'})
    if change=='scope':report['test_loaded']=True
    report['targets']['0']['component_diagnostics_sha256']=put(diagpath,diag)
    put(root/'canary.json',report)
    with pytest.raises(ValueError):d.inspect_evidence(root)


class TorchMock:
    __version__='2.7.1+cu118'
    version=NS(cuda='11.8')
    cuda=NS(is_initialized=lambda:False)
    def __init__(self):
        self.backends=NS(cudnn=NS(deterministic=False,benchmark=True,allow_tf32=False),cuda=NS(matmul=NS(allow_tf32=True)))
        self.enabled=False;self.warn=True
    def use_deterministic_algorithms(self,enabled,*,warn_only):self.enabled=enabled;self.warn=warn_only
    def are_deterministic_algorithms_enabled(self):return self.enabled
    def is_deterministic_algorithms_warn_only_enabled(self):return self.warn


def test_actual_worker_backend_readback_tf32_true(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch=TorchMock();assert d.apply_backend(torch,d.BACKEND)==d.BACKEND
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_backend_requires_environment_before_cuda(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(ValueError,match='BEFORE_CHILD'):d.apply_backend(TorchMock(),d.BACKEND)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch=TorchMock();torch.cuda=NS(is_initialized=lambda:True)
    with pytest.raises(ValueError,match='PRECEDE_CUDA'):d.apply_backend(torch,d.BACKEND)


def test_native_entry_does_not_change_environment(monkeypatch):
    monkeypatch.delenv('T13_DETERMINISTIC_EXECUTION_CONTRACT',raising=False)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','unchanged')
    assert d.activate_from_environment('/unused') is None
    assert d.os.environ['CUBLAS_WORKSPACE_CONFIG']=='unchanged'


def test_admission_keeps_original_ledger_and_t13_inode_contract(tmp_path,monkeypatch):
    obj=object.__new__(T13LazyRecoveryGuard)
    obj.path=tmp_path/'original/authorization.json';obj.path.parent.mkdir();obj.owner=tmp_path/'fresh-owner';obj.owner.mkdir()
    obj.spec={'task_spec_sha256':'spec','attempt_id':'id','task_id':'t13','output_root':str(tmp_path/'science')}
    obj.authorization_sha256='auth';obj.runtime=tmp_path;obj.baseline={'failcnt':11}
    contract={'original_authorization_path':str(obj.path),'original_authorization_sha256':'auth',
        'formal_quota_ledger':str(obj.path.parent/'full_start.json'),'process_peak_bytes':23*GIB,
        'min_headroom_bytes':420*GIB,'canary_failcnt_increment':0}
    monkeypatch.setattr(d,'validate_contract',lambda *a:contract)
    # 94k is below Mut's guard, but above T13's explicit inclusive 8192 floor.
    now=dict(headroom_bytes=450*GIB,failcnt=11,free_bytes=200*GIB,free_inodes=94644)
    monkeypatch.setattr('src.utils.t13_lazy_recovery_guard.resources',lambda _:now)
    result=obj.accept_deterministic_contract_and_claim_full('/contract','sha')
    assert result['canary_rerun'] is False and result['compact_reserved_inodes']==4096
    assert (obj.path.parent/'full_start.json').exists()
    assert not (obj.owner/'full_start.json').exists()
    with pytest.raises(ValueError,match='ALREADY_CONSUMED'):
        obj.accept_deterministic_contract_and_claim_full('/contract','sha')
