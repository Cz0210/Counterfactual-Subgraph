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

def test_native_mut_action_key_requires_real_oracle_identity():
    from src.eval.node_wasserstein_distance import node_wasserstein_action_key
    kw=dict(canonical_parent_smiles='CCC',canonical_residual_smiles='CC',checkpoint_identity='molclr',
        feature_cost='cosine',node_mass='uniform',size_penalty_beta=0.,distance_namespace='molclr_node_wasserstein_v1')
    context=dict(candidate_id='c',match_atom_indices=[0],action_semantics_version='hard_delete_all_matches_v1',
        match_selection_policy='min_wnode_then_cfdrop_then_match_index_v1',teacher_sha256='rf',
        distance_implementation_version='molclr_node_wasserstein_exact_emd2_v1')
    assert node_wasserstein_action_key(**kw,action_context=context)
    del context['teacher_sha256']
    with pytest.raises(ValueError):node_wasserstein_action_key(**kw,action_context=context)
