from pathlib import Path
import pytest
from src.experiments.bace_globalgce_aplus_evaluation import calibration_summary


def pair(parent,candidate,flip):
    return dict(parent_id=parent,candidate_id=candidate,split='calibration',pair_strict_flip=flip)


def test_actual_nonzero_and_zero_counts():
    candidates=[{'candidate_id':'r'}]
    assert calibration_summary([pair('p','r',True)],candidates)['strict_flip_pair_count']==1
    assert calibration_summary([pair('p','r',False)],candidates)['strict_flip_pair_count']==0


def test_export_passes_real_unchanged_matrix_validator(tmp_path):
    from src.experiments.bace_globalgce_aplus_evaluation import sealed_matrix_file
    from src.eval.mutagenicity_wnode_selector import load_calibration_matrix
    tmp_path=tmp_path/'calibration_matrix'
    tmp_path.mkdir()
    candidates=[{'candidate_id':'r'}]
    pairs=[{**pair('p','r',True),'applicable':True,'wnode_distance':.01,'cf_drop':.2}]
    sealed_matrix_file(tmp_path/'pair_matrix.jsonl',pairs,jsonl=True)
    sealed_matrix_file(tmp_path/'selected_candidate_universe.jsonl',candidates,jsonl=True)
    sealed_matrix_file(tmp_path/'summary.json',calibration_summary(pairs,candidates))
    sealed_matrix_file(tmp_path/'run_manifest.json',{'inputs':{'cohort_name':'calibration'},'split':'calibration','test_loaded':False})
    assert load_calibration_matrix(tmp_path).full_strict_flip_pair_count==1


def test_full_cartesian_and_bool_required():
    c=[{'candidate_id':'a'},{'candidate_id':'b'}]
    for pairs in ([pair('p','a',True)], [pair('p','a',True),pair('p','a',True)],
                  [pair('p','a',1),pair('p','b',False)]):
        with pytest.raises(ValueError):calibration_summary(pairs,c)


def test_old_failed_matrix_preserved_and_calibration_not_reexecuted():
    root=Path(__file__).resolve().parents[1]
    leaf=(root/'src/experiments/bace_globalgce_aplus_evaluation.py').read_text()
    owner=(root/'src/baselines/bace_globalgce_aplus_owner.py').read_text()
    assert "matrix=root/'calibration_matrix_summary_v2'" in leaf
    assert "root/'selector_summary_v2'" in leaf
    assert "actions=('freeze','test','aggregate') if repair_receipt" in owner
    assert 'PREVIOUS_OWNER_STILL_ALIVE' in owner and 'EVALUATION_OPEN_FILE' in owner
    assert 'source_spec_sha256' in owner and 'COMPLETED_CALIBRATION_REQUIRED' in owner
