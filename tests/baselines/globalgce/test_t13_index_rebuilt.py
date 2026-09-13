import copy
import pytest
from src.baselines.t13_index_rebuilt import INVARIANTS, LAYOUT_FIELDS, validate_rebuilt_identity, adapted_checkpoint
from src.eval.bace_frozen_gnn_contracts import stable_sha256

def identity():
    d={k:'fixed' for k in INVARIANTS}
    d.update({k:'old' for k in LAYOUT_FIELDS})
    d.update(full_augmented_tensor_materialization=False,all_masks_reconstructed_exactly=True,sampler={'batch_size':500})
    return seal(d)

def seal(d):
    d['identity_sha256']=stable_sha256({k:v for k,v in d.items() if k!='identity_sha256'});return d

def test_layout_change_is_disclosed_not_old_parity():
    old=identity();new=copy.deepcopy(old);new['masks_sha256']='new';seal(new)
    r=validate_rebuilt_identity(old,new)
    assert r['changed_layout_fields']==['masks_sha256'] and not r['old_trajectory_parity_claimed']
    assert old['masks_sha256']=='old'

@pytest.mark.parametrize('field',INVARIANTS)
def test_scientific_invariants_reject(field):
    old=identity();new=copy.deepcopy(old);new[field]='changed';seal(new)
    with pytest.raises(ValueError,match='NONLAYOUT'):validate_rebuilt_identity(old,new)

def test_working_copy_preserves_state_and_owed_validation():
    old=identity();new=copy.deepcopy(old);new['masks_sha256']='new';seal(new)
    c=dict(model_state={'w':[1,2]},optimizer_state={'state':{'x':[3]}},scheduler_state={'last_epoch':30},
      python_rng_state=[1],numpy_rng_state=[2],torch_rng_state=[3],cuda_rng_state=[4],
      resume_identity={'dataset':'TasteMolNet','source_label':1,'target_label':0},resume_identity_sha256='source',
      sampler_state=dict(old['sampler'],next_epoch=30),augmented_dataset_identity=old,
      next_epoch=30,best_loss=1.0,config={'epochs':100})
    before=copy.deepcopy(c);out,r=adapted_checkpoint(c,new,original_identity=old,formal_ledger={'max_full_starts':1,'attempt_id':'original'})
    assert c==before and r['validation_epoch30_required'] and not r['persistent_committed']
    for key in c:
      if key not in ('augmented_dataset_identity','sampler_state'):assert out[key]==c[key]
    out['model_state']['w'][0]=999
    assert c['model_state']['w'][0]==1
