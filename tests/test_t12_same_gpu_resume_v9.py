import copy
import pytest
from src.utils.t12_same_gpu_resume_v9 import validate_same_gpu_resume_identity as validate


def identity():
    return dict(identity_template=dict(gpu_uuid='GPU-original',execution_commit='a'*40,
        execution_tree='b'*40,runtime_identity_sha256='c'*64,model_sha='d'*64),
        runtime=dict(execution_commit='a'*40,execution_tree='b'*40,
            gpu=dict(gpu_uuid='GPU-original',physical_index=3,visible_selector='3',name='A800'),
            deterministic=True),transition_contract_sha256='e'*64,cohort=['parent1'])


def test_same_gpu_all_scientific_fields_exact():
    old=identity();new=copy.deepcopy(old)
    new['runtime']['gpu']['visible_selector']='GPU-original'
    receipt=validate(current=new,authority=old)
    assert receipt['status']=='SAME_GPU_IDENTITY_VERIFIED_NOT_ALGORITHM_PARITY'
    assert not receipt['scientific_equivalence_claimed_before_parity']
    assert old==identity()


@pytest.mark.parametrize('field',['gpu','model','cohort','runtime','commit'])
def test_nontransport_changes_rejected(field):
    old=identity();new=copy.deepcopy(old)
    if field=='gpu':new['identity_template']['gpu_uuid']='GPU-other'
    if field=='model':new['identity_template']['model_sha']='f'*64
    if field=='cohort':new['cohort']=['parent2']
    if field=='runtime':new['runtime']['deterministic']=False
    if field=='commit':new['identity_template']['execution_commit']='f'*40
    with pytest.raises(ValueError):validate(current=new,authority=old)


def test_cross_commit_still_requires_complete_reviewed_receipt():
    from src.utils.tastemolnet_t12_accelerated_from250 import T12AcceleratedError
    old=identity();new=copy.deepcopy(old)
    new['identity_template']['execution_commit']='f'*40
    new['runtime']['execution_commit']='f'*40
    with pytest.raises(T12AcceleratedError,match='receipt'):
        validate(current=new,authority=old)
