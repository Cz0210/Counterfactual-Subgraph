import pytest
from src.eval.selector_selected_test_v7 import validate_new_block

def test_failed_strict_distance_not_zero_or_inf():
    block={'pairs':[{'parent_id':'p','candidate_id':'c'}], 'matches':[{'cf_flip':True,'wnode_distance':None}]}
    with pytest.raises(ValueError,match='NUMERICAL_FAILURE'):validate_new_block(block,{('p','c')})

def test_only_selected_missing_set_allowed():
    b={'pairs':[{'parent_id':'p','candidate_id':'c'}],'matches':[]}
    validate_new_block(b,{('p','c')})
    with pytest.raises(ValueError,match='SET_CHANGED'):validate_new_block(b,{('p','other')})

def test_unknown_failure_cannot_be_infinity():
    b={'pairs':[{'parent_id':'p','candidate_id':'c','pair_strict_flip':False,'failure_reason':'ERROR'}],'matches':[]}
    with pytest.raises(ValueError,match='UNKNOWN_PAIR'):validate_new_block(b,{('p','c')})
