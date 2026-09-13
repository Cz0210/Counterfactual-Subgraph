import pytest
from src.baselines.cm_crem_release import _audit_gate, ReleaseRejected, _allowed

def test_postfilter_requires_real_stage_inventory():
    with pytest.raises(ReleaseRejected,match='missing actual stage'):
        _audit_gate(lambda n:{},lambda n:'x',{'pool_binding.json'})

def test_train_fixture_cannot_publish_as_dataset_result():
    names={'spec.json','pool_binding.json','selection_report.json','selection_freeze.json',
           'calibration_prepared.json','test_prepared.json','test_evaluation.json',
           'audit/final_audit.json','results/export_manifest.json'}
    with pytest.raises(ReleaseRejected,match='Train fixture'):
        _audit_gate(lambda n: {'fixture':True,'batch_size':32} if n=='spec.json' else {},lambda n:'x',names)

def test_compact_matrices_allowed_but_models_excluded():
    assert _allowed('calibration/block-0000.npz')
    assert _allowed('pool_binding.json')
    assert not _allowed('models/weights.pt')
    assert not _allowed('encoding/pool-0000.npz')
