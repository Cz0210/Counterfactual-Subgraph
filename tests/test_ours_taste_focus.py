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


def test_train_scaffold_margin_cohort_is_stable_and_test_free():
    from src.eval.ours_taste_development import stratified_development
    rows=[{'parent_id':str(i),'smiles':'C'*(i+1),'pred_before':1,'p_before':[.1,.8,.1]} for i in range(8)]
    a=stratified_development(rows,limit=4)
    b=stratified_development(list(reversed(rows)),limit=4)
    assert a==b and len({r['parent_id'] for r in a})==4
    assert all(r['margin_band'] in range(4) for r in a)


def test_actual_registry_schema_and_gpu_uuid_binding():
    from src.eval.ours_taste_development import validate_gpu_reservation
    r={'gpu_leases':[{'gpu':1,'state':'RELEASED','lease_path':'existing'}]}
    validate_gpu_reservation(r,1,'GPU-fixed','GPU-fixed\n')
    with pytest.raises(AssertionError,match='UUID_CHANGED'):validate_gpu_reservation(r,1,'GPU-fixed','GPU-wrong')
    r['gpu_leases'][0]['state']='PREDEPLOYED'
    with pytest.raises(AssertionError,match='RESERVED'):validate_gpu_reservation(r,1,'GPU-fixed','GPU-fixed')


def test_compact_node_cache_reload_exact_no_per_graph_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.eval.ours_taste_development import compact_embedder_class
    cls=compact_embedder_class()
    ckpt=tmp_path/'weights';ckpt.write_bytes(b'fixed fixture')
    loaded=SimpleNamespace(model=SimpleNamespace(num_layer=5,emb_dim=3),checkpoint_path=ckpt)
    expected=np.arange(6,dtype=np.float32).reshape(2,3)
    monkeypatch.setattr(cls,'_compute_node_embeddings',lambda self,s:expected.copy())
    args={'compact_db':tmp_path/'nodes.sqlite','molclr_root':tmp_path,'molclr_ckpt':ckpt,
          'node_emb_cache_dir':tmp_path/'old_nodes','loaded_model':loaded}
    cache=cls(**args); x=cache.get('CC');cache.commit();cache.close()
    monkeypatch.setattr(cls,'_compute_node_embeddings',lambda *_:pytest.fail('RECOMPUTED'))
    cache=cls(**args);y=cache.get('CC');cache.close()
    assert np.array_equal(x.H,y.H) and np.array_equal(y.H,expected)
    assert not list((tmp_path/'old_nodes').glob('*.npz'))
