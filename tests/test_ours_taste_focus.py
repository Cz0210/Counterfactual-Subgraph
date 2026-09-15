import json
import numpy as np
import pytest
from src.eval.ours_taste_focus_matrix import compact_pairs
from src.eval.ours_taste_focus_selector import maximum_cover, ceilings, prefix, Selector


def test_negative_objective_dual_and_actual_incumbent():
    a=np.array([[1,0,0],[1,1,0],[0,0,1],[0,1,1]],bool)
    r=maximum_cover(a,k=1,seconds=2)
    assert r['lower']==r['upper']==2 and len(r['selected'])==1
    assert r['negative_objective_dual_bound']==-2


def test_unknown_not_infinity_and_source_upper():
    d=np.array([[.1,np.nan],[np.inf,np.inf],[np.nan,np.nan]])
    r=ceilings(d,np.array([True,True,False]),theta=.2,k=1,seconds=2)
    assert r['theta']['pool_lower']==1 and r['theta']['pool_upper']==1
    assert r['finite_recourse']['upper']==1


def test_prefix_at_most_k_and_cost_not_conditional():
    d=np.array([[.1,np.inf],[np.inf,.8],[np.inf,np.inf]])
    r=prefix(d,[0,1],.2,.5)
    assert r[19]['effective_k']==2 and r[19]['covered_count']==1
    assert r[19]['finite_count']==2
    assert r[19]['capped_mean']==pytest.approx(1.1/3)
    assert r[19]['conditional_median']==pytest.approx(.45)


def test_selector_actual_replacement_preserves_budget():
    # 21 rules, old 20 misses the only covering rule. A real member must change.
    d=np.full((2,21),np.inf);d[0,20]=.01;d[1,0]=.02
    s=Selector(d,[str(i) for i in range(21)],.1,.5,[.1,.3],np.ones(21))
    selected,trace=s.optimize(list(range(20)),[],seconds=3,proposals=100,accepts=5)
    assert len(set(selected))==20 and 20 in selected
    assert s.key(selected)<s.key(list(range(20)))
    assert trace['claim_global_optimal'] is False


def test_incomplete_matrix_cannot_freeze():
    with pytest.raises(AssertionError,match='COMPLETE'):
        Selector(np.array([[np.nan]]),['a'],.1,.5,[.1],[1])


@pytest.mark.parametrize('destination',[0,2])
def test_multiclass_or_and_pair_identity(tmp_path,destination):
    identity={key:'bound' for key in ('oracle_checkpoint_hash','temperature_calibration_hash',
        'feature_schema_hash','molclr_checkpoint_hash','distance_namespace',
        'action_semantics_version','match_selection_policy')}
    row={**identity,'split':'calibration','source_label':1,'cf_mode':'strict_flip',
         'classifier_family':'gine','rf_oracle_used':False,'parent_id':'p','candidate_id':'c',
         'pred_before':1,'pred_after':destination,'destination_label':destination,'applicable':True,
         'num_valid_residuals':1,'num_matches':1,'pair_strict_flip':True,'wnode_distance':.1,'residual_smiles':'C'}
    path=tmp_path/'raw.jsonl';path.write_text(json.dumps(row)+'\n')
    d,pred,ev=compact_pairs(path,['p'],['c'],'calibration',identity)
    assert d[0,0]==.1 and pred[0]==1 and ev['unknown_pairs']==0
    row['pred_before']=0;path.write_text(json.dumps(row)+'\n')
    with pytest.raises(AssertionError): compact_pairs(path,['p'],['c'],'calibration',identity)
