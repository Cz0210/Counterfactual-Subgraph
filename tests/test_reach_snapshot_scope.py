from scripts.paper.render_reach_repair_snapshot import corrected_llm_rows
import pytest


def fixture():
    old = [dict(variant='L0', eligible_rules='15', source='l0 accepted')]
    source = dict(audit=dict(state='PASS', test_selection=False, valid_unique_rule_count=19),
                  metrics={'cohort_size': 141, 'CCRCov@10': .1, 'CCRCov@20': .2,
                           'conditional_median_WNode': 999},
                  candidate_metrics={'proposal_attempts': 3088}, root='bound/source',
                  table2=[dict(k='10', effective_k='10', fixed_capped_mean_cost='.03',
                               conditional_median_cost='.01', strict_flip_parent_count='20')])
    return old, {name: source for name in ['CHEMLLM_7B_OFF_THE_SHELF',
        'CHEMLLM_7B_PPO_LORA_MAIN', 'CHEMLLM_2B_OFF_THE_SHELF']}


def test_k10_cost_not_ambiguous_k20_auxiliary():
    old, new = fixture()
    rows = corrected_llm_rows(old, new)
    assert rows[1]['conditional_median_WNode'] == '.01'
    assert rows[1]['new_reach_v2_state'].startswith('PENDING')
    assert old[0] == dict(variant='L0', eligible_rules='15', source='l0 accepted')


def test_no_unaccepted_or_test_selected_correction():
    old, new = fixture()
    new['CHEMLLM_7B_OFF_THE_SHELF']['audit']['test_selection'] = True
    with pytest.raises(ValueError):
        corrected_llm_rows(old, new)


def test_no_unscoped_auxiliary_table():
    old, new = fixture()
    new['CHEMLLM_7B_OFF_THE_SHELF']['table2'][0]['k'] = '20'
    with pytest.raises(ValueError):
        corrected_llm_rows(old, new)
