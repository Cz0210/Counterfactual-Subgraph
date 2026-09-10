import copy
import numpy as np
import pytest
from src.baselines.cm_crem_k20 import freeze_k20
from src.baselines.cm_crem_runtime import digest
from src.baselines.cm_crem_release import _audit_gate, RESULT_FILES, ReleaseRejected


def fixture_records():
    spec={'experiment_id':'CM-Global-K20-v2','generation_enabled':False,
          'pool_count':5474,'test_used_for_selection':False,'source_science_sha':'a'*64}
    sha=digest(spec)
    f,_=freeze_k20(np.array([[.1,.2]]),[['OK','OK']],['p'],['a','b'],np.array([True]),.15,1,[0,.15,1],sha,'b'*64)
    records={name:{'scope_sha256':sha} for name in ['pool.json','calibration_comparison.json','calibration/complete.json','test/complete.json','test_evaluation.json']}
    records['spec.json']=spec
    records['freeze.json']={'scope_sha256':sha,'freeze':f.to_dict(),'test_loaded_by_this_run':False}
    records['calibration_comparison.json'].update(calibration_only=True,test_used_for_choice=False)
    records['calibration/complete.json'].update(parents=66)
    records['test/complete.json'].update(parents=141,candidate_count=2,test_reads_after_freeze=True)
    records['audit/k20_audit.json']={'scope_sha256':sha,'status':'K20_RECORDS_AND_INDEPENDENT_PROTOTYPE_CHECKS_PASS',
        'main_matrix_written':False,'independent_calibration_replay':True,'original_generation_reused':True,
        'new_generation_calls':0,'checked_prototypes':['a','b'],'independent_new_ot_checks':[['p','a'],['p','b']],
        'selection_freeze_sha':f.freeze_sha256,'test_count':141}
    records['results/export_manifest.json']={'fixture':False,'dataset':'bace','oracle':'gine','source_files':{n:'c'*64 for n in RESULT_FILES}}
    return records


def gate(records):
    return _audit_gate(records.__getitem__,lambda n:'c'*64,set(records)|{'results/source_csv/'+n for n in RESULT_FILES})


def test_v2_distinct_gate_without_v1_alias_or_model():
    result=gate(fixture_records())
    assert result['variant']=='CM-Global-K20-v2'
    assert result['independent_spotcheck']['path']=='audit/k20_audit.json'


@pytest.mark.parametrize('name,key,value',[
    ('audit/k20_audit.json','independent_calibration_replay',False),
    ('audit/k20_audit.json','checked_prototypes',[]),
    ('audit/k20_audit.json','selection_freeze_sha','d'*64),
    ('test/complete.json','parents',140),
    ('test/complete.json','test_reads_after_freeze',False),
    ('calibration_comparison.json','test_used_for_choice',True),
    ('results/export_manifest.json','oracle','gin'),
    ('results/export_manifest.json','source_files',{}),
])
def test_v2_missing_or_conflicting_evidence_rejected(name,key,value):
    records=fixture_records();records[name][key]=value
    with pytest.raises((ReleaseRejected,ValueError)):gate(records)
