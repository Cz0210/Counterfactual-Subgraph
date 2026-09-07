import copy
import json
from pathlib import Path
import pytest
import torch
from src.baselines import bace_globalgce_aplus_training as training
from src.ablations.llm import existing_gpu_owner as owner
from src.ablations.gnn.early_policy import gpu_allowed
from src.eval.bace_frozen_gnn_contracts import sha256_file

def descriptor(tmp_path):
    config={'repair_kind':'GIN_ALIGNED_EPOCH35_WARMSTART','formal_fresh_campaigns_max':1,
        'training_contract':training.TRAINING_CONTRACT,'total_epochs_max':100}
    path=tmp_path/'contract.json';path.write_text(json.dumps(config))
    return config,{'path':str(path),'sha256':sha256_file(path)}

def test_new_family_requires_real_bound_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'validate_resource_config',lambda _:None)
    config,binding=descriptor(tmp_path)
    sampler=owner.ResourceSampler({},0,'GPU-fixture',task_family='globalgce_aplus',reach_contract=binding)
    assert sampler.task_family=='globalgce_aplus'
    with pytest.raises(ValueError,match='EXACT_TRAIN_SCOPE'):
        owner.ResourceSampler({},1,'GPU-fixture',task_family='globalgce_aplus',reach_contract=binding)
    config['training_contract']=dict(training.TRAINING_CONTRACT,test_used=True)
    Path(binding['path']).write_text(json.dumps(config));binding['sha256']=sha256_file(binding['path'])
    with pytest.raises(ValueError,match='EXACT_TRAIN_SCOPE'):
        owner.ResourceSampler({},0,'GPU-fixture',task_family='globalgce_aplus',reach_contract=binding)

def test_actual_gpu_and_main_reservations_still_gate():
    good={k:True for k in ('owners_healthy','registry_healthy','memory_safe','storage_safe','checkpoint_resume_pass')}
    good.update(globalgce_aplus_contract_verified=True,gpu_index=0,actual_gpu_observation={'process_count':0},
        main_ready_waiting_gpu=False,gpu_main_reservation=False,active_early_ablation_gpus=0,gpu_idle_seconds=0)
    assert gpu_allowed(good,family='globalgce_aplus')['allowed']
    for change in ({'gpu_main_reservation':True},{'main_ready_waiting_gpu':True},
                   {'actual_gpu_observation':{'process_count':1}},{'globalgce_aplus_contract_verified':False}):
        assert not gpu_allowed(dict(good,**change),family='globalgce_aplus')['allowed']
    assert not gpu_allowed(good,family='llm')['allowed']

def test_warmstart_loads_epoch35_weights_not_old_optimizer(tmp_path,monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self,*a,**kw):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.]));self.fsg=type('F',(),{})()
        def create_decoders(self): pass
    monkeypatch.setattr(training,'validate_official_globalgce_root',lambda _:{'runtime_source_authority':{},'official_commit':'pinned'})
    monkeypatch.setattr(training,'_import_official_modules',lambda *a,**kw:{'GlobalGCE':Tiny})
    path=tmp_path/'epoch35.pt'
    torch.save({'epoch_completed':35,'run_kind':'FORMAL_REPAIR_FINETUNE','config_sha256':'prior',
        'model':{'weight':torch.tensor([35.])},'optimizer':{'old':'must not adopt'}},path)
    config={'official_root':str(tmp_path),'warmstart_checkpoint':str(path),'warmstart_source_config_sha256':'prior'}
    rules={'feat':torch.zeros(1,2,3),'edge_attr':torch.zeros(1,1,4)}
    model,_=training.load_generator(config,{'frequent_subgraphs_path':'fixed'},rules,None,'cpu')
    assert model.weight.item()==35
    assert training.TRAINING_CONTRACT['epochs']==100
    assert 'new_GIN_optimizer' in training.TRAINING_CONTRACT['initialization']
    state=torch.load(path,weights_only=False);assert state['epoch_completed']==35 and state['optimizer']=={'old':'must not adopt'}
    bad=dict(config,warmstart_source_config_sha256='wrong')
    with pytest.raises(ValueError,match='SOURCE_CONTRACT_CHANGED'):
        training.load_generator(bad,{'frequent_subgraphs_path':'fixed'},rules,None,'cpu')

def test_real_gpu_canary_not_replaced_by_cpu_or_pass_marker(tmp_path):
    config={'gpu_canary_root':str(tmp_path)}
    for name,value in {'terminal.json':{'state':'CANARY_COMPLETE','optimizer_updates':2},
        'execution_receipt.json':{'config_sha256':'sealed','device':'cpu'},
        'reload_receipt.json':{'state':'PASS','fresh_generator_loaded':True,'engineering_next_optimizer_update':3},
        'identity_oracle_canary.json':{'state':'PASS','frozen_GIN_gradient':False,'target_flip_claimed':False}}.items():
        (tmp_path/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='real GPU'):
        training.require_gpu_canary(config,'sealed')

