import copy
import math
import pytest
from src.experiments.bace_migration_acceptance import close, inspect_parent


def fixture():
    match=dict(parent_id='p',candidate_id='c',oracle_checkpoint_hash='w',delete_valid=True,
        residual_connected=True,pred_before=1,pred_after=0,teacher_strict_flip=True,cf_flip=True,
        distance_ok=True,wnode_distance=.2,cf_drop=.4,match_atom_indices=[1],residual_smiles='CC')
    pair=dict(match,applicable=True,pair_strict_flip=True,best_match_atom_indices=[1])
    return dict(parent_id='p',pair_rows=[pair],match_rows=[match])


def test_actual_match_pair_reduction():
    assert inspect_parent(fixture(),['c'],'w')==(1,{'c':.2})


def test_reject_wrong_oracle_and_fake_zero():
    with pytest.raises(ValueError,match='MODEL_BINDING'):inspect_parent(fixture(),['c'],'other')
    r=fixture();r['pair_rows'][0]['wnode_distance']=0
    with pytest.raises(ValueError,match='BEST_MATCH'):inspect_parent(r,['c'],'w')


def test_reject_old_flip_and_missing_distance():
    r=fixture();r['match_rows'][0]['pred_before']=0
    with pytest.raises(ValueError,match='STRICT_FLIP'):inspect_parent(r,['c'],'w')
    r=fixture();r['match_rows'][0]['distance_ok']=False
    with pytest.raises(ValueError,match='DISTANCE_GAP'):inspect_parent(r,['c'],'w')


def test_null_is_not_zero():
    close('',None,'N/A')
    with pytest.raises(ValueError):close('0',None,'N/A')
    with pytest.raises(ValueError):close('',0,'zero')


def test_candidate_order_must_match():
    with pytest.raises(ValueError,match='POOL_ORDER'):inspect_parent(fixture(),['different'],'w')
