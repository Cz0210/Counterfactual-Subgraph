"""Independent saved-row audit and numeric export for the theta0.1 overlay."""
import csv,gzip,json,math,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .ours_taste_focus_matrix import read_json,dump_json
from .ours_taste_focus_selector import write_csv
from .ours_taste_theta010 import metrics,digest


def export(root,include_r3=False):
    from rdkit import Chem
    from src.chem.hard_deletion import enumerate_connected_hard_deletions
    from scripts.finalize_ours_taste_focus import verify_probability_semantics
    c=read_json(root/'resolved_contract.json');prior=Path(c['prior_root']);run=root/'R2-test'
    if (root/'R2-test-attempt2/terminal.json').exists():run=root/'R2-test-attempt2'
    f=read_json(root/'R2_selection_freeze.json');terminal=read_json(run/'terminal.json')
    assert terminal['state']=='R2_TEST_COMPLETE_AWAITING_EXPORT_AUDIT'
    assert datetime.fromisoformat(f['created_at'])<datetime.fromisoformat(terminal['completed_at'])
    assert read_json(root/'R3_branch_decision.json')['decision_before_R2_test']
    assert read_json(run/'runtime_theta_receipt.json')['actual_evaluation_theta']==.1
    pool=read_json(prior/'search-round1/candidate_pool_P1.json');poolby={r['candidate_id']:r for r in pool}
    assert digest(pool)==c['pool_p1_hash']
    orders={**c['orders'],'R2':f['candidate_ids']}
    resultroot=root
    r3=root/'R3-search'
    if include_r3:
        t3=read_json(r3/'terminal.json');assert t3['state']=='R3_COMPLETE_AWAITING_SAVED_RAW_AUDIT'
        f3=read_json(r3/'selection_freeze.json');assert not f3['test_used'] and f3['theta']==.1
        assert datetime.fromisoformat(f3['created_at'])<datetime.fromisoformat(t3['completed_at'])
        pool=read_json(r3/'candidate_pool_P2.json');assert digest(pool)==f3['pool_hash']
        poolby={r['candidate_id']:r for r in pool};orders['R3']=f3['candidate_ids']
        terminal={**terminal,'new_rules':terminal['new_rules']+t3['new_test_rules'],'new_pairs':terminal['new_pairs']+t3['new_test_pairs']}
        resultroot=root/'release-R3';resultroot.mkdir(exist_ok=True)
        dump_json(resultroot/'resolved_contract.json',c)
    z=np.load((r3 if include_r3 else run)/'test_selected_union.npz',allow_pickle=False)
    old=np.load(prior/'search-round1/test_selected_union.npz',allow_pickle=False)
    d=z['distances'];ids=z['candidates'].tolist();pids=z['parents'].tolist();pred=z['predictions']
    assert pids==c['test_ids']==old['parents'].tolist() and np.array_equal(pred,old['predictions'])
    assert not np.isnan(d).any() and len(pids)==468
    for j,cid in enumerate(ids):
        if cid in old['candidates']:assert np.array_equal(d[:,j],old['distances'][:,old['candidates'].tolist().index(cid)])
    testcsv=prior/'inputs/test.csv'
    assert hashlib.sha256(testcsv.read_bytes()).hexdigest()==c['old_contract']['declared_test_sha256']
    smiles={r['molecule_id']:r['model_smiles'] for r in csv.DictReader(testcsv.open())}
    rawfiles=[root/'inputs/old_test_pair_details.jsonl',prior/'search-round1/test_new_pairs.jsonl.gz']
    if (run/'test_new_pairs.jsonl.gz').is_file():rawfiles.append(run/'test_new_pairs.jsonl.gz')
    if include_r3 and (r3/'test_new_pairs.jsonl.gz').is_file():rawfiles.append(r3/'test_new_pairs.jsonl.gz')
    rows={};funnel={'pair_records':0,'applicable':0,'legal_residuals':0,'strict_flip':0}
    pi={p:i for i,p in enumerate(pids)};ci={p:i for i,p in enumerate(ids)}
    for path in rawfiles:
        op=gzip.open if path.suffix=='.gz' else open
        with op(path,'rt') as stream:
            for line in stream:
                r=json.loads(line);key=(r['parent_id'],r['candidate_id'])
                if key[0] not in pi or key[1] not in ci:continue
                assert key not in rows,'DUPLICATE_PAIR_RECORD'
                assert r['split']=='test' and r['source_label']==1
                for rk,ck in [('oracle_checkpoint_hash','checkpoint_id'),('temperature_calibration_hash','temperature_calibration_hash'),('feature_schema_hash','feature_schema_hash'),('molclr_checkpoint_hash','molclr_checkpoint_sha256')]:
                    assert r[rk]==c['old_contract'][ck]
                assert r['parent_smiles']==smiles[key[0]]
                assert r['canonical_fragment']==poolby[key[1]]['canonical_fragment']
                assert r['pred_before']==pred[pi[key[0]]]
                verify_probability_semantics(r)
                expected=float(r['wnode_distance']) if r['pair_strict_flip'] else np.inf
                assert expected==d[pi[key[0]],ci[key[1]]]
                if r['pair_strict_flip']:
                    assert math.isfinite(expected) and expected>=0 and r['pred_after'] in (0,2)
                    if key[1] in terminal['new_rules']:
                        results=enumerate_connected_hard_deletions(r['parent_smiles'],r['canonical_fragment'])
                        assert any(o.valid and o.residual_smiles==r['residual_smiles'] and list(o.match_atom_indices)==r['best_match_atom_indices'] for o in results)
                else:assert r['wnode_distance'] is None
                rows[key]=r
                funnel['pair_records']+=1;funnel['applicable']+=bool(r['applicable'])
                funnel['legal_residuals']+=r['num_valid_residuals'];funnel['strict_flip']+=bool(r['pair_strict_flip'])
    assert len(rows)==len(pids)*len(ids),'INCOMPLETE_RAW_SELECTED_UNION'
    dest=resultroot/'source_csv';dest.mkdir(exist_ok=True)
    allmetrics=[];parentrows=[];ecdf=[];best20={};beyond={}
    for variant,order in orders.items():
        ix=[ids.index(cid) for cid in order]
        fractions={}
        for k in range(1,21):
            vals=d[:,ix[:k]].min(axis=1)
            finite=[]
            for i,pid in enumerate(pids):
                winner=None
                if np.isfinite(vals[i]):
                    candidates=[rows[(pid,cid)] for cid in order[:k] if rows[(pid,cid)]['pair_strict_flip']]
                    winner=min(candidates,key=lambda r:(r['wnode_distance'],-r['cf_drop'],r['candidate_id'],r['destination_label']))
                    mol=Chem.MolFromSmiles(smiles[pid])
                    frac=sum(mol.GetAtomWithIdx(int(j)).GetAtomicNum()>1 for j in winner['best_match_atom_indices'])/mol.GetNumHeavyAtoms()
                    finite.append(frac)
                if k in (10,20):
                    parentrows.append({'variant':variant,'k':k,'parent_id':pid,'pred_before':int(pred[i]),'source':bool(pred[i]==1),
                        'state':'FINITE_VALID_STRICT_FLIP' if winner else 'PROVEN_NO_VALID_RECOURSE',
                        'raw_best_distance':float(vals[i]) if winner else 'INF','capped_distance':min(float(vals[i]),c['cost_cap']),
                        'covered_theta010':bool(vals[i]<=.1),'covered_old_theta':bool(vals[i]<=c['theta_old']),
                        'best_candidate_id':winner['candidate_id'] if winner else 'N/A',
                        'deleted_heavy_atom_fraction':frac if winner else 'N/A'})
            fractions[k]=float(np.mean(finite)) if finite else 'N/A'
            if k in (10,20):
                points=np.unique(np.r_[np.linspace(0,.2,401),vals[np.isfinite(vals)&(vals<=.2)],.1,c['theta_old']])
                ecdf.extend({'variant':variant,'k':k,'theta':float(t),'num_covered':int((vals<=t).sum()),'N_base':len(pids),'coverage_fraction':float((vals<=t).mean())} for t in points)
                beyond[f'{variant}_K{k}']=int((np.isfinite(vals)&(vals>.2)).sum())
                if k==20:best20[variant]=vals
        for theta in (c['theta_old'],.1):
            for r in metrics(d,pred,ix,theta,c['cost_cap']):
                allmetrics.append({'variant':variant,'oracle_id':c['oracle_id'],'pool_sha':f3['pool_hash'] if variant=='R3' else c['pool_p0_hash'] if variant=='R0' else c['pool_p1_hash'],
                    'order_sha':digest(order),'split':'test','N_base':r['parent_count'],'N_source':int((pred==1).sum()),
                    'k_requested':r['k'],'k_effective':r['effective_k'],'theta':theta,'num_covered':r['covered_count'],
                    'coverage_fraction':r['coverage'],'finite_recourse_count':r['finite_count'],'finite_recourse_fraction':r['reach'],
                    'original_cost_cap':c['cost_cap'],'fixed_capped_mean_cost':r['capped_mean'],
                    'conditional_median_all_finite':r['conditional_median'],
                    'average_deleted_heavy_atom_fraction':fractions[r['k']]})
    write_csv(dest/'all_prefix_metrics.csv',allmetrics)
    write_csv(dest/'figure3_taste_ours_theta010.csv',[r for r in allmetrics if r['theta']==.1])
    for k in (10,20):write_csv(dest/f'figure4_taste_ours_k{k}.csv',[r for r in ecdf if r['k']==k])
    write_csv(dest/'table2_taste_ours_k20_theta010.csv',[r for r in allmetrics if r['theta']==.1 and r['k_requested']==20])
    write_csv(dest/'table2_taste_ours_k20_oldtheta.csv',[r for r in allmetrics if r['theta']==c['theta_old'] and r['k_requested']==20])
    write_csv(dest/'parent_best_distances_uncapped.csv',parentrows)
    effects=[]
    effect_specs=[('R0','R0',c['theta_old'],.1,'THRESHOLD_ONLY'),('R1','R1',c['theta_old'],.1,'THRESHOLD_ONLY'),('R1','R2',.1,.1,'SELECTION_ONLY')]
    if include_r3:effect_specs.append(('R2','R3',.1,.1,'POOL_EXPANSION_SAME_SELECTOR'))
    for before,after,t0,t1,kind in effect_specs:
        a=next(r for r in allmetrics if r['variant']==before and r['theta']==t0 and r['k_requested']==20)
        b=next(r for r in allmetrics if r['variant']==after and r['theta']==t1 and r['k_requested']==20)
        effects.append({'effect':kind,'before':before,'after':after,'theta_before':t0,'theta_after':t1,
            'covered_delta':b['num_covered']-a['num_covered'],'coverage_delta':b['coverage_fraction']-a['coverage_fraction'],
            'finite_delta':b['finite_recourse_count']-a['finite_recourse_count'],'capped_cost_delta':b['fixed_capped_mean_cost']-a['fixed_capped_mean_cost']})
        if kind=='THRESHOLD_ONLY':
            for key in ('finite_recourse_count','fixed_capped_mean_cost','conditional_median_all_finite'):assert a[key]==b[key]
    write_csv(dest/'threshold_vs_reselection_effect.csv',effects)
    bounds=read_json((r3 if include_r3 else root)/'calibration_bounds.json')
    write_csv(dest/'calibration_pool_and_k20_bounds.csv',[{'scope':'P2_calibration_only' if include_r3 else 'P1_calibration_only','target':k,'N_base':len(c['calibration_ids']),
      'N_source':c['source_calibration'],'pool_lower':v['pool_lower'],'pool_upper':v['pool_upper'],'K20_lower':v['lower'],'K20_upper':v['upper'],
      'solver_optimal':v['optimal'],'all_deletion_space':'UNKNOWN'} for k,v in bounds.items()])
    rng=np.random.default_rng(7);boot={v:[] for v in best20 if v!='R0'}
    for _ in range(1000):
        ix=rng.integers(0,len(pids),len(pids));a=best20['R0'][ix]
        for v in boot:
            b=best20[v][ix];boot[v].append([float((b<=.1).mean()-(a<=.1).mean()),float(np.minimum(b,c['cost_cap']).mean()-np.minimum(a,c['cost_cap']).mean())])
    dump_json(resultroot/'paired_parent_intervals.json',{'unit':'fixed_base_parent','replicates':1000,'seed':7,'not_training_seed_uncertainty':True,
      'versus_R0':{v:np.quantile(x,[.025,.975],axis=0).tolist() for v,x in boot.items()}})
    dump_json(resultroot/'final_audit.json',{'state':'SAVED_RAW_GRAPH_ORACLE_AND_NUMERIC_AUDIT_PASS','scope':'Independent saved-row audit; not repeated oracle inference',
      'oracle_recomputed_for_audit':False,'old_results_unchanged':True,'main_matrix_write':False,'N_base':len(pids),'N_source':int((pred==1).sum()),
      'theta_actual':.1,'cap_actual':c['cost_cap'],'funnel':funnel,'new_pairs':terminal['new_pairs'],'rule_union':len(ids),
      'finite_above_figure4_endpoint':beyond,'post_hoc':True,'baseline_ranking_claim':False,'completed_at':datetime.now(timezone.utc).isoformat(),
      'source_csv_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.glob('*.csv'))}})
    return read_json(resultroot/'final_audit.json')
