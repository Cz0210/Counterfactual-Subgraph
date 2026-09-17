from types import SimpleNamespace
import json
import numpy as np
import pytest
from src.eval.selector_controlled_v7 import Objective, load_matrix, prefix_rows

def obj(d):
    d=np.array(d,dtype=float);n=d.shape[1]
    return Objective(d,SimpleNamespace(structural_similarity=np.eye(n),normalized_sizes=np.ones(n)))

def test_positive_common_weights_do_not_change_append_argmax():
    o=obj([[.03,.05,.2],[.2,.03,.02],[.07,.2,.05]])
    values=[o.value([0,j],multi=True,prefix=False) for j in (1,2)]
    weighted=[.3*o.value([0],multi=True,prefix=False)+.7*x for x in values]
    assert np.argmax(values)==np.argmax(weighted)

def test_unknown_is_not_infinity(tmp_path):
    p=tmp_path/'pairs.jsonl'
    p.write_text(json.dumps(dict(parent_id='p',candidate_id='c',pair_strict_flip=False,failure_reason='ERROR'))+'\n')
    with pytest.raises(ValueError,match='UNCLASSIFIED'):load_matrix(p,['c'])

def test_legacy_bace_label_needs_actual_zero_strict_count(tmp_path):
    p=tmp_path/'pairs.jsonl'
    row=dict(parent_id='p',candidate_id='c',pair_strict_flip=False,
             failure_reason='no_valid_strict_flip_with_finite_wnode',num_strict_flip_matches=1)
    p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='UNCLASSIFIED'):load_matrix(p,['c'])
    row['num_strict_flip_matches']=0;p.write_text(json.dumps(row)+'\n')
    assert np.isinf(load_matrix(p,['c'])[1][0,0])

def test_same_members_terminal_invariant_and_cost_separate():
    d=np.array([[.12,.2],[np.inf,.04]])
    a=prefix_rows(d,[0,1],.03)[-1];b=prefix_rows(d,[1,0],.03)[-1]
    assert a==b and a['covered']==1 and a['capped_mean']==.03
    assert a['conditional_median']==.08

def test_zero_gain_still_fills_distinct_and_stable_id_tie():
    o=obj([[np.inf]*4]);seq=o.greedy(['d','c','b','a'],multi=False)
    assert seq==[3,2,1,0]

def test_refine_never_changes_fixed_members_or_decreases_objective():
    o=obj([[.01,np.inf,.2],[.15,.02,np.inf]])
    seq,stats=o.refine([2,1,0],multi=True,prefix=True,replacement=False,max_proposals=40)
    assert set(seq)=={0,1,2}
    assert o.value(seq,multi=True,prefix=True)>=o.value([2,1,0],multi=True,prefix=True)
    assert stats['proposals']<=40

def test_known_nonsource_completes_base_without_inventing_unknown(tmp_path):
    from src.eval.selector_controlled_v7 import complete_non_source_rows,base_parent_ids
    parents=tmp_path/'parents.csv';parents.write_text('molecule_id,label\na,1\nb,1\n')
    pred=tmp_path/'pred.csv';pred.write_text('parent_id,checkpoint_id,backbone,temperature,source_label,predicted_label\nb,sha,gine,1.5,1,0\n')
    spec=dict(calibration_parent_csv=str(parents),calibration_label_filter=1,
              calibration_predictions_csv=str(pred),oracle_sha256='sha',temperature=1.5)
    base,d,proof=complete_non_source_rows(spec,['a'],np.array([[.2]]))
    assert base==['a','b'] and np.isinf(d[1,0]) and proof[0]['parent_id']=='b'
    pred.write_text(pred.read_text().replace('1.5,1,0','1.5,1,1'))
    with pytest.raises(ValueError,match='REMAINS_UNKNOWN'):
        complete_non_source_rows(spec,['a'],np.array([[.2]]))

def test_resume_test_does_not_rerun_calibration(tmp_path,monkeypatch):
    import src.eval.selector_controlled_v7 as s
    root=tmp_path/'p0';root.mkdir()
    spec={'output_root':str(tmp_path)};path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    (root/'input_binding.json').write_text(json.dumps(dict(spec_sha=s.digest(spec),candidate_ids=['c'])))
    (root/'ALL_CALIBRATION_FROZEN.json').write_text(json.dumps(dict(variants=['S0'])))
    f=dict(ordered_candidate_ids=['c']);f['freeze_sha256']=s.digest(f)
    (root/'S0_freeze.json').write_text(json.dumps(f))
    monkeypatch.setattr(s,'consume_saved_test',lambda *a: {'reused':True})
    monkeypatch.setattr(s.Objective,'refine',lambda *a,**k:pytest.fail('must not rerun selector'))
    assert s.resume_test(path)=={'reused':True}
