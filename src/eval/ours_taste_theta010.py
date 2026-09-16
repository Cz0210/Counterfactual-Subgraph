"""Taste-only, user-authorized absolute-theta protocol overlay.

No generation, oracle training, main-authority mutation or generic scheduling.
Selection is a separate calibration-only stage. Saved raw distances stay raw.
"""
from datetime import datetime, timezone, timedelta
import csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from .ours_taste_focus_matrix import read_json, dump_json
from .ours_taste_focus_selector import ceilings, Selector, prefix, write_csv


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def raw_valid_distances(raw,predictions,valid=None):
    raw=np.asarray(raw,dtype=float)
    if np.isnan(raw).any(): raise ValueError('UNKNOWN_OR_CENSORED_NOT_A_COMPLETE_RESULT')
    if (raw<0).any(): raise ValueError('NEGATIVE_DISTANCE')
    keep=np.asarray(predictions)==1
    if raw.ndim==2:keep=keep[:,None]
    if valid is not None:keep=keep&np.asarray(valid,dtype=bool)
    return np.where(keep,raw,np.inf)


def metrics(raw,predictions,order,theta,cap,valid=None):
    return prefix(raw_valid_distances(raw,predictions,valid),order,theta,cap)


def prepare(prior,config,root):
    cfg=yaml.safe_load(config.read_text())
    assert cfg['campaign']=='OURS_TASTE_K20_THETA010_V3'
    for v in (cfg['protocol_change']['theta_star'],cfg['selector']['theta_star'],cfg['reporting']['primary_theta']):assert v==.1
    assert cfg['contracts']['k_max']==20 and cfg['reporting']['primary_k']==20
    assert cfg['protocol_change']['cost_cap_change_allowed'] is False
    base=read_json(prior/'compact/contract.json');model=read_json(root/'inputs/model_card.json')
    temp=read_json(root/'inputs/temperature_scaling.json');labels=read_json(root/'inputs/label_map.json')
    assert model['checkpoint_id']==base['checkpoint_id']==temp['source_model_sha256']
    assert hashlib.sha256((root/'inputs/temperature_scaling.json').read_bytes()).hexdigest()==base['temperature_calibration_hash']
    assert labels=={'0':'Bitter','1':'Sweet','2':'Tasteless'} and model['backbone']=='gine'
    p0=read_json(prior/'compact/candidate_pool.json');p1=read_json(prior/'search-round1/candidate_pool_P1.json')
    old=read_json(prior/'search-round1/selection_freeze_manifest.json')
    cal=np.load(prior/'search-round1/calibration_P1.npz',allow_pickle=False)
    test=np.load(prior/'search-round1/test_selected_union.npz',allow_pickle=False)
    assert len(cal['parents'])==cfg['contracts']['expected_calibration_base_count']
    assert len(test['parents'])==cfg['contracts']['expected_test_base_count']
    assert cal['candidates'].tolist()==[p['candidate_id'] for p in p1]
    resolved={'experiment_id':cfg['campaign'],'prior_root':str(prior),'old_contract':base,
      'k_max':20,'primary_report_k':20,'primary_theta':.1,'selector_primary_theta':.1,
      'new_search_target_theta':.1,'primary_export_theta':.1,'theta_old':base['theta_star'],
      'cost_cap':base['cost_cap'],'oracle_id':base['checkpoint_id'],'temperature':temp['temperature'],
      'class_mapping':labels,'source_label':1,'destinations':[0,2],
      'pool_p0_hash':digest(p0),'pool_p1_hash':digest(p1),'p0_count':len(p0),'p1_count':len(p1),
      'auxiliary_theta_grid':base['theta_grid'],'auxiliary_weights':'UNCHANGED_UNIFORM_AS_PRIOR_D',
      'selector_sha256':hashlib.sha256(Path(__file__).with_name('ours_taste_focus_selector.py').read_bytes()).hexdigest(),
      'selector_budget':{'seconds':300,'proposals':10000,'accepts':50},
      'orders':{'R0':old['ordered_ids']['A'],'R1':old['ordered_ids']['D']},
      'calibration_ids':cal['parents'].tolist(),'test_ids':test['parents'].tolist(),
      'calibration_ids_sha':digest(cal['parents'].tolist()),'test_ids_sha':digest(test['parents'].tolist()),
      'source_calibration':int((cal['predictions']==1).sum()),'source_test':int((test['predictions']==1).sum()),
      'absolute_deadline':cfg['absolute_deadline'],'search_cutoff':(datetime.fromisoformat(cfg['absolute_deadline'])-timedelta(hours=3)).isoformat(),
      'scope':'POST_HOC_PROTOCOL_REVISION_NOT_MODEL_IMPROVEMENT','main_matrix_write':False}
    assert not (root/'resolved_contract.json').exists()
    dump_json(root/'resolved_contract.json',resolved)
    dump_json(root/'protocol_change_receipt.json',{'theta_old':base['theta_star'],'theta_new':.1,
      'absolute_uncapped_wnode':True,'cost_cap_unchanged':base['cost_cap'],'model_unchanged':True,
      'user_selected_not_refitted_theta':True,'test_previously_observed':True,'main_matrix_write':False})
    dump_json(root/'cache_adoption.json',{'old_evaluator':'evaluate_parent in tastemolnet_ours_full.py',
      'raw_distance_role':'wnode_distance before any prefix/cap reduction','theta_or_cap_censoring_in_producer':False,
      'calibration_unknown_pairs':int(np.isnan(cal['distances']).sum()),
      'finite_calibration_pairs_above_cap':int((np.isfinite(cal['distances'])&(cal['distances']>base['cost_cap'])).sum()),
      'finite_calibration_pairs_above_new_theta':int((np.isfinite(cal['distances'])&(cal['distances']>.1)).sum()),
      'source_acceptance_reused':str(prior/'release/final_audit.json')})
    rows=[]
    for variant,order in resolved['orders'].items():
        ix=[test['candidates'].tolist().index(i) for i in order]
        a=metrics(test['distances'],test['predictions'],ix,base['theta_star'],base['cost_cap'])
        b=metrics(test['distances'],test['predictions'],ix,.1,base['cost_cap'])
        for old_row,new_row in zip(a,b):
            for key in ('reach','capped_mean','conditional_median'):assert old_row[key]==new_row[key]
            assert old_row['coverage']<=new_row['coverage']
        rows.extend({'variant':variant,'theta':theta,**row} for theta,series in [(base['theta_star'],a),(.1,b)] for row in series)
    write_csv(root/'R0_R1_threshold_reduction.csv',rows)
    return resolved


def select(root):
    from rdkit import Chem
    from .tastemolnet_ours_full import select_on_calibration
    c=read_json(root/'resolved_contract.json');prior=Path(c['prior_root'])
    assert not (root/'R2_selection_freeze.json').exists()
    p1=read_json(prior/'search-round1/candidate_pool_P1.json');ids=[r['candidate_id'] for r in p1]
    z=np.load(prior/'search-round1/calibration_P1.npz',allow_pickle=False)
    d=raw_valid_distances(z['distances'],z['predictions']);theta=c['selector_primary_theta'];assert theta==.1
    bounds=ceilings(d,z['predictions']==1,theta,k=20,seconds=120)
    dump_json(root/'calibration_bounds.json',bounds)
    # Preserve the original D's C-initialization and implementation, changing its
    # primary theta at both initialization and Selector constructors only.
    records=[{'candidate_id':cid,'parent_id':pid,'split':'calibration','pair_strict_flip':bool(np.isfinite(d[i,j])),
              'wnode_distance':float(d[i,j])} for i,pid in enumerate(z['parents']) for j,cid in enumerate(ids)]
    initial,_=select_on_calibration(p1,records,theta_star=theta);del records
    s=Selector(d,ids,theta,c['cost_cap'],c['auxiliary_theta_grid'],[Chem.MolFromSmiles(r['canonical_fragment']).GetNumHeavyAtoms() for r in p1])
    chosen,trace=s.optimize([ids.index(r['candidate_id']) for r in initial],[x['selected'] for x in bounds.values()],**c['selector_budget'])
    order=[ids[i] for i in chosen]
    freeze={'state':'CALIBRATION_FROZEN_BEFORE_R2_TEST','variant':'R2','candidate_ids':order,
      'pool_hash':c['pool_p1_hash'],'order_sha':digest(order),'oracle_id':c['oracle_id'],
      'primary_theta':s.theta,'original_cost_cap':c['cost_cap'],'selector_trace':trace,
      'test_used_for_selection':False,'created_at':datetime.now(timezone.utc).isoformat()}
    dump_json(root/'R2_selection_freeze.json',freeze)
    rows=metrics(d,z['predictions'],chosen,theta,c['cost_cap']);write_csv(root/'R2_calibration_prefix.csv',rows)
    dump_json(root/'runtime_theta_receipt.json',{'selector_actual_theta':s.theta,'evaluator_primary_theta':c['primary_theta'],
        'search_target_theta':c['new_search_target_theta'],'export_primary_theta':c['primary_export_theta'],
        'auxiliary_grid_unchanged':c['auxiliary_theta_grid'],'cost_cap':c['cost_cap']})
    # This ledger adopts observed absence of any V2 campaign/owner; historical
    # v1 consumption is explicitly separate, never reset by renaming.
    budget={'prior_v1_search_queries':1679,'v2_search_queries_used':0,'v2_lm_outputs_used':0,
      'v2_search_wnode_used':0,'v2_new_unique_used':0,'max_residual_oracle':8192,'max_search_wnode':4096,
      'max_lm_outputs':1024,'max_new_unique':512,'provenance':'20260916 scoped registry/process/root inventory; V1 only, no V2 run found'}
    dump_json(root/'budget_ledger.json',budget)
    eligible=bounds['theta']['pool_lower']<c['source_calibration'] and datetime.now(timezone.utc)<datetime.fromisoformat(c['search_cutoff'])
    dump_json(root/'R3_branch_decision.json',{'state':'ELIGIBLE_PREDECLARED_AFTER_R2_DELIVERY' if eligible else 'NOT_ADMITTED',
       'decision_before_R2_test':True,'uses_test_results':False,'pool_cov_cal':bounds['theta']['pool_lower'],
       'selected_cov_cal':rows[-1]['covered_count'],'source_cal':c['source_calibration'],
       'next_requirement':'theta-aware bounded search implementation and actual resource admission',
       'budget_ledger':str(root/'budget_ledger.json'),'search_cutoff':c['search_cutoff'],
       'created_at':datetime.now(timezone.utc).isoformat()})
    return freeze


def evaluate_selected(spec,out,compact,contract,scorer,provider,identity,pause):
    from .ours_taste_search_chain import evaluate_delta
    from .tastemolnet_ours_full import load_prepared_split
    root=Path(spec['protocol_root']);c=read_json(root/'resolved_contract.json');f=read_json(root/'R2_selection_freeze.json')
    assert c['primary_theta']==f['primary_theta']==.1 and not f['test_used_for_selection']
    assert (root/'R3_branch_decision.json').is_file(),'R3_DECISION_MUST_PRECEDE_NEW_TEST'
    prior=Path(spec['prior_remote_root']);pool=read_json(prior/'search-round1/candidate_pool_P1.json')
    assert digest(pool)==c['pool_p1_hash'];by={r['candidate_id']:r for r in pool}
    old=np.load(prior/'search-round1/test_selected_union.npz',allow_pickle=False)
    orders={**c['orders'],'R2':f['candidate_ids']};union=sorted({i for s in orders.values() for i in s})
    missing=[cid for cid in union if cid not in old['candidates']]
    parents=load_prepared_split(Path(contract['test_path']),expected_split='test',expected_sha256=contract['declared_test_sha256'])
    assert [p.parent_id for p in parents]==c['test_ids']==old['parents'].tolist()
    if missing:
        d,pred=evaluate_delta([{'parent_id':p.parent_id,'smiles':p.smiles} for p in parents],
                [by[i] for i in missing],'test',scorer,provider,identity,out,pause,datetime.fromisoformat(c['absolute_deadline']),spec)
        assert np.array_equal(pred,old['predictions'])
    combined=np.full((len(parents),len(union)),np.nan)
    for j,cid in enumerate(union):
        combined[:,j]=old['distances'][:,old['candidates'].tolist().index(cid)] if cid in old['candidates'] else d[:,missing.index(cid)]
    np.savez_compressed(out/'test_selected_union.npz',distances=combined,parents=old['parents'],predictions=old['predictions'],candidates=np.asarray(union))
    rows=[{'variant':v,'theta':theta,**r} for v,order in orders.items() for theta in (c['theta_old'],c['primary_theta'])
            for r in metrics(combined,old['predictions'],[union.index(i) for i in order],theta,c['cost_cap'])]
    write_csv(out/'test_prefix.csv',rows)
    dump_json(out/'runtime_theta_receipt.json',{'actual_evaluation_theta':c['primary_theta'],'actual_cost_cap':c['cost_cap'],
                 'coverage_uses_uncapped_raw':True,'source_predicate':1,'reused_rules':len(union)-len(missing),'new_rules':len(missing)})
    dump_json(out/'terminal.json',{'state':'R2_TEST_COMPLETE_AWAITING_EXPORT_AUDIT','new_pairs':len(missing)*len(parents),
         'new_rules':missing,'rule_union':len(union),'theta':c['primary_theta'],'cost_cap':c['cost_cap'],
         'distance_statistics':provider.stats_dict(),'completed_at':datetime.now(timezone.utc).isoformat()})
