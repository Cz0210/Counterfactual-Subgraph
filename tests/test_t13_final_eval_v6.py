import json
import pytest
from src.baselines import tastemolnet_globalgce_full as f


def contract(tmp_path, **updates):
    grid=[0.0,0.1,0.2]
    data=dict(dataset='TasteMolNet',thresholds=grid,theta_star=.1,cost_cap=.03416003659645076,
        final_eval_protocol=f.FINAL_EVAL_V6,primary_report_k=20,k_mode='AT_MOST_K_NO_PADDING',
        threshold_config_hash=f.stable_json_sha256(grid),threshold_source='V6 explicit user protocol',
        threshold_source_split='frozen_protocol',test_used_for_selection=False)
    data.update(updates);path=tmp_path/'threshold.json';path.write_text(json.dumps(data))
    return f.load_threshold_contract(path)


def rows(n):
    return [dict(split='test',rf_oracle_used=False,parent_id=p,candidate_id=str(i),
        pair_strict_flip=True,wnode_distance=d,pred_before=1,destination_label=2,applicable=True,cf_drop=.2)
        for p,d in [('a',.12),('b',.05)] for i in range(n)]


@pytest.mark.parametrize('n',[1,7,19,20])
def test_actual_v6_at_most_raw_coverage(tmp_path,n):
    t=contract(tmp_path);r=f.compute_standardized_metrics(rows(n),list(map(str,range(n))),t,parent_ids=['a','b'])
    assert len(r['prefix'])==20 and r['table2'][0]['k']==20
    assert r['table2'][0]['coverage']==.5  # capped .12 must not become covered
    assert r['table2'][0]['cost']==t.cost_cap
    assert all(x['coverage']==.5 for x in r['prefix'])


def test_zero_requires_full_science_and_explicit_base(tmp_path):
    t=contract(tmp_path)
    with pytest.raises(f.TasteGlobalGCEFullError):f.compute_standardized_metrics([],[],t,parent_ids=['a'])
    r=f.compute_standardized_metrics([],[],t,parent_ids=['a'],complete_scientific_run=True)
    assert r['table2'][0]['coverage']==0 and r['table2'][0]['cost']==t.cost_cap
    assert r['table2'][0]['conditional_median_cost']=='N/A'
    assert len(r['parent_best'])==20


def test_unknown_not_zero_and_base_preserved(tmp_path):
    t=contract(tmp_path);r=rows(1);r[0]['status']='UNKNOWN'
    with pytest.raises(f.TasteGlobalGCEFullError):f.compute_standardized_metrics(r,['0'],t)
    with pytest.raises(f.TasteGlobalGCEFullError):f.compute_standardized_metrics(rows(1),['0'],t,parent_ids=['a','b','c'])


def test_legacy_and_v6_validation_are_distinct(tmp_path):
    with pytest.raises(f.TasteGlobalGCEFullError):contract(tmp_path,final_eval_protocol='LEGACY_T13')
    with pytest.raises(f.TasteGlobalGCEFullError):contract(tmp_path,primary_report_k=10)
    with pytest.raises(f.TasteGlobalGCEFullError):contract(tmp_path,cost_cap=0)
    with pytest.raises(f.TasteGlobalGCEFullError):contract(tmp_path,theta_star=-.1)
    t=contract(tmp_path,final_eval_protocol='LEGACY_T13',cost_cap=.2)
    assert t.minimum_final_rules==10 and t.primary_report_k==10
    assert 'final_eval_protocol' not in t.to_dict()


def test_empty_selector_is_explicit_not_padding():
    rules,receipt=f.select_rules_on_calibration([],[],theta_star=.1,minimum_final_rules=0)
    assert rules==[] and receipt['ordered_rule_ids']==[]
    with pytest.raises(f.TasteGlobalGCEFullError):f.select_rules_on_calibration([],[],theta_star=.1)
