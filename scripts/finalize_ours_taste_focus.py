#!/usr/bin/env python3
"""One finite CPU finalizer: wait for this Taste stage, audit raw, export plots.

No science dispatch, no restart, no GPU claim, no main-matrix write.
"""
import argparse,csv,gzip,hashlib,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.eval.ours_taste_focus_matrix import read_json,dump_json
from src.eval.ours_taste_focus_selector import prefix,write_csv


def verify_probability_semantics(row):
    before=np.asarray(row['p_before'],dtype=float)
    assert before.shape==(3,) and np.isfinite(before).all() and (before>=0).all()
    assert abs(before.sum()-1)<1e-5 and int(before.argmax())==row['pred_before']
    if row['pair_strict_flip']:
        after=np.asarray(row['p_after'],dtype=float)
        assert after.shape==(3,) and np.isfinite(after).all() and (after>=0).all()
        assert abs(after.sum()-1)<1e-5 and int(after.argmax())==row['pred_after'] in (0,2)
        assert row['pred_before']==1 and row['destination_label']==row['pred_after']
        assert abs(float(before[1]-after[1])-row['cf_drop'])<1e-12
    else:
        # Existing evaluate_parent encodes absent after-probabilities as [],
        # while after-label is null. Do not rewrite sealed producer records.
        assert row['p_after']==[] and row['pred_after'] is None


def finalize(source,compact,output,test_csv=None):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    assert not output.exists();output.mkdir(parents=True)
    terminal=read_json(source/'terminal.json')
    assert terminal['state']=='FOURWAY_SCIENCE_COMPLETE_AWAITING_INDEPENDENT_EXPORT_AUDIT'
    contract=read_json(compact/'contract.json');freeze=read_json(source/'selection_freeze_manifest.json')
    test_csv=test_csv or Path(contract['test_path'])
    assert hashlib.sha256(test_csv.read_bytes()).hexdigest()==contract['declared_test_sha256'], 'TEST_INPUT_CHANGED'
    smiles={r['molecule_id']:r['model_smiles'] for r in csv.DictReader(test_csv.open())}
    pool=read_json(source/'candidate_pool_P1.json');pool_by_id={r['candidate_id']:r for r in pool}
    assert not freeze['test_used_for_selection'] and freeze['recommended_on_calibration'] in ('B','C','D')
    z=np.load(source/'test_selected_union.npz',allow_pickle=False)
    d=z['distances'];pids=z['parents'].tolist();ids=z['candidates'].tolist()
    assert len(set(ids))==len(ids)<=80 and not np.isnan(d).any()
    old=np.load(compact/'test.npz',allow_pickle=False)
    assert pids==old['parents'].tolist() and np.array_equal(z['predictions'],old['predictions'])
    for j,cid in enumerate(ids):
        if cid in old['candidates']: assert np.array_equal(d[:,j],old['distances'][:,old['candidates'].tolist().index(cid)])
    new=[cid for cid in ids if cid not in old['candidates']]
    reconstructed=np.full((len(pids),len(new)),np.nan);pi={x:i for i,x in enumerate(pids)};ci={x:i for i,x in enumerate(new)}
    raw=source/'test_new_pairs.jsonl.gz';funnel={'pairs':0,'matches':0,'legal_residuals':0,'finite_pairs':0}
    if new:
        with gzip.open(raw,'rt') as f:
            for line in f:
                row=json.loads(line);i,j=pi[row['parent_id']],ci[row['candidate_id']]
                assert np.isnan(reconstructed[i,j]) and row['split']=='test'
                assert row['oracle_checkpoint_hash']==contract['checkpoint_id']
                assert row['temperature_calibration_hash']==contract['temperature_calibration_hash']
                assert row['feature_schema_hash']==contract['feature_schema_hash']
                assert row['molclr_checkpoint_hash']==contract['molclr_checkpoint_sha256']
                assert row['source_label']==1 and row['rf_oracle_used'] is False
                assert row['parent_smiles']==smiles[row['parent_id']]
                assert row['canonical_fragment']==pool_by_id[row['candidate_id']]['canonical_fragment']
                assert row['action_semantics_version']=='connected_sanitized_residual_v1'
                assert row['match_selection_policy']=='existential_min_wnode_among_valid_connected_strict_flips_v1'
                assert row['distance_namespace']=='tastemolnet_ours_full_wnode_v1'
                verify_probability_semantics(row)
                funnel['pairs']+=1;funnel['matches']+=row['num_matches'];funnel['legal_residuals']+=row['num_valid_residuals']
                if row['pair_strict_flip']:
                    from src.chem.hard_deletion import enumerate_connected_hard_deletions
                    assert row['pred_before']==1 and row['pred_after'] in (0,2) and row['residual_smiles']
                    assert row['num_valid_residuals']>0 and row['best_match_atom_indices']
                    mol=Chem.MolFromSmiles(row['residual_smiles']);assert mol is not None and len(Chem.GetMolFrags(mol))==1
                    outcomes=enumerate_connected_hard_deletions(smiles[row['parent_id']],pool_by_id[row['candidate_id']]['canonical_fragment'])
                    assert any(o.valid and o.residual_smiles==row['residual_smiles'] and list(o.match_atom_indices)==row['best_match_atom_indices'] for o in outcomes), 'RESIDUAL_NOT_BOUND_TO_ORIGINAL_SINGLE_DELETE'
                    value=float(row['wnode_distance']);assert math.isfinite(value) and value>=0
                    funnel['finite_pairs']+=1
                else:
                    assert row['wnode_distance'] is None
                    value=np.inf
                reconstructed[i,j]=value
        assert not np.isnan(reconstructed).any()
        for j,cid in enumerate(new):assert np.array_equal(reconstructed[:,j],d[:,ids.index(cid)])
        assert (source/'selection_freeze_manifest.json').stat().st_mtime_ns <= raw.stat().st_mtime_ns
    theta,cap=contract['theta_star'],contract['cost_cap']
    best={};metrics=[];parent_best=[];ecdf=[]
    for variant,order in freeze['ordered_ids'].items():
        assert len(order)==len(set(order))<=20
        indices=[ids.index(x) for x in order]
        rows=prefix(d,indices,theta,cap)
        assert all(rows[i]['covered_count']<=rows[i+1]['covered_count'] and rows[i]['capped_mean']>=rows[i+1]['capped_mean']-1e-15 for i in range(19))
        metrics.extend({'variant':variant,**r} for r in rows)
        for k in (10,20):
            values=np.min(d[:,indices[:k]],axis=1)
            if k==20:best[variant]=values
            parent_best.extend({'variant':variant,'k':k,'parent_id':pid,'best_distance':float(v) if np.isfinite(v) else 'N/A',
                       'covered':bool(v<=theta),'finite_recourse':bool(np.isfinite(v)),'capped_distance':min(float(v),cap)} for pid,v in zip(pids,values))
            points=np.unique(np.r_[0,values[np.isfinite(values)],contract['theta_grid']])
            ecdf.extend({'variant':variant,'k':k,'threshold':float(x),'coverage':float((values<=x).mean())} for x in points)
    write_csv(output/'figure3_taste_ours_v0_v1.csv',metrics)
    write_csv(output/'parent_best_distances.csv',parent_best)
    write_csv(output/'figure4_taste_ours_v0_v1_k20.csv',[r for r in ecdf if r['k']==20])
    write_csv(output/'figure4_taste_ours_v0_v1_k10.csv',[r for r in ecdf if r['k']==10])
    write_csv(output/'table2_taste_ours_v0_v1_k20.csv',[r for r in metrics if r['k']==20])
    # Common scaffold resamples for all paired variants, not training-seed CIs.
    groups={}
    for i,pid in enumerate(pids):
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles[pid]),includeChirality=True)
        groups.setdefault(scaffold,[]).append(i)
    gs=list(groups.values());rng=np.random.default_rng(7);deltas={v:[] for v in best if v!='A'}
    for _ in range(1000):
        ix=np.concatenate([gs[j] for j in rng.integers(0,len(gs),len(gs))])
        a=best['A'][ix]
        for v,values in best.items():
            if v!='A':deltas[v].append([float((values[ix]<=theta).mean()-(a<=theta).mean()),
                                      float(np.minimum(values[ix],cap).mean()-np.minimum(a,cap).mean())])
    dump_json(output/'paired_intervals.json',{'replicates':1000,'seed':7,'unit':'Murcko_scaffold',
                 'not_training_seed_uncertainty':True,'versus_A':{v:np.quantile(values,[.025,.975],axis=0).tolist() for v,values in deltas.items()}})
    write_csv(output/'candidate_sources.csv',[{'candidate_id':r['candidate_id'],'fragment':r['canonical_fragment'],
               'sources':json.dumps(r.get('source_labels',[])+r.get('source_modes',[])),
               'train_parent_ids':json.dumps(r.get('source_parent_ids',[]))} for r in pool])
    dump_json(output/'baseline_contract.json',contract)
    dump_json(output/'selection_freeze_manifest.json',freeze)
    dump_json(output/'coverage_ceiling_report.json',{
        'scope':'Exact only within measured P0/P1 matrices; all connected deletion space UNKNOWN',
        'full_deletion_space':'UNKNOWN','source_calibration':int((np.load(source/'calibration_P1.npz')['predictions']==1).sum()),
        'calibration_denominator':len(np.load(source/'calibration_P1.npz')['parents']),
        'P1_calibration':read_json(source/'calibration_P1_bounds.json'),
        'P0_train':read_json(source/'train_P0_bounds.json'),'train_change':read_json(source/'train_reach_change.json')})
    dump_json(output/'candidate_funnel.json',{'pool':read_json(source/'pool_P1_freeze.json'),'new_test_pairs':funnel,
                                          'saved_lm_funnel':contract['train_generation_funnel']})
    dump_json(output/'budget_ledger.json',read_json(source/'budget_ledger.json'))
    render(output,metrics,ecdf,cap)
    dump_json(output/'final_audit.json',{'state':'RAW_GRAPH_ORACLE_EVIDENCE_AND_NUMERIC_REDUCTION_PASS',
       'audit_scope':'Independent saved-row/graph/identity/matrix/metric audit; not fresh oracle inference',
       'oracle_recomputed':False,'main_matrix_write':False,'source_root':str(source),'old_result_unchanged':True,
       'test_parents':len(pids),'source_predicate_count':int((z['predictions']==1).sum()),
       'rule_union':len(ids),'raw_new_pairs_checked':len(pids)*len(new),'recommended':freeze['recommended_on_calibration'],
       'created_at':datetime.now(timezone.utc).isoformat(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}})


def render(output,metrics,ecdf,cap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'pdf.fonttype':42})
    colors={'A':'black','B':'green','C':'#b000b0','D':'#cc2020'}
    fig,ax=plt.subplots(2,1,figsize=(6.8,4.8),constrained_layout=True)
    for v in colors:
        rows=[r for r in metrics if r['variant']==v]
        for a,key,factor in [(ax[0],'coverage',100),(ax[1],'capped_mean',1)]:
            a.plot([r['k'] for r in rows],[factor*r[key] for r in rows],c=colors[v],lw=1,label=v)
    ax[0].set_title('TasteMolNet - fixed GINE, post-hoc A/B/C/D');ax[0].set_ylabel('Coverage (%)');ax[0].legend(ncol=4)
    ax[1].set_ylabel('Fixed-capped mean cost');ax[1].set_xlabel('Size (K)')
    for a in ax:a.set_xlim(1,20);a.grid(alpha=.3)
    for ext in ('png','pdf'):fig.savefig(output/f'figure3_taste_ours_v0_v1.{ext}',dpi=220)
    plt.close(fig)
    for k in (10,20):
        fig,ax=plt.subplots(figsize=(6.8,2.8),constrained_layout=True)
        for v in colors:
            rows=[r for r in ecdf if r['variant']==v and r['k']==k]
            ax.step([r['threshold'] for r in rows],[100*r['coverage'] for r in rows],where='post',c=colors[v],lw=1,label=v)
        ax.set_xlim(0,cap);ax.set_xlabel('WNode distance threshold');ax.set_ylabel('Coverage (%)')
        ax.set_title(f'TasteMolNet - fixed GINE K{k}');ax.legend(ncol=4);ax.grid(alpha=.3)
        for ext in ('png','pdf'):fig.savefig(output/f'figure4_taste_ours_v0_v1_k{k}.{ext}',dpi=220)
        plt.close(fig)
    rows=[r for r in metrics if r['k']==20]
    fig,ax=plt.subplots(figsize=(8,2.2));ax.axis('off')
    table=ax.table(cellText=[[r['variant'],f"{r['covered_count']}/{r['parent_count']}",f"{100*r['coverage']:.4f}",
                         f"{r['finite_count']}/{r['parent_count']}",f"{r['capped_mean']:.9f}"] for r in rows],
                   colLabels=['Variant','Covered K20','Coverage (%)','Finite recourse','Capped mean'],loc='center')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.4)
    ax.set_title('TasteMolNet / frozen GINE / K20 / fixed 468 parents',fontsize=10)
    fig.savefig(output/'table2_taste_ours_v0_v1_k20.pdf',bbox_inches='tight');plt.close(fig)
    lines=[r'\begin{tabular}{lrrrr}',r'Variant & Covered & Coverage (\%) & Finite & Capped mean \\',r'\hline']
    lines.extend(f"{r['variant']} & {r['covered_count']}/{r['parent_count']} & {100*r['coverage']:.4f} & {r['finite_count']}/{r['parent_count']} & {r['capped_mean']:.9f} \\\\" for r in rows)
    lines.append(r'\end{tabular}')
    (output/'table2_taste_ours_v0_v1_k20.tex').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--compact',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--wait-until',required=True)
    p.add_argument('--test-csv',type=Path,help='Relocated byte-identical test CSV for offline audit')
    a=p.parse_args();assert a.config.resolve()==(ROOT/'configs/hpc.yaml').resolve()
    end=datetime.fromisoformat(a.wait_until)
    while not (a.source/'terminal.json').exists():
        if (a.source/'failure.json').exists():raise RuntimeError('SCIENCE_FAILED_PRESERVE_OUTPUT')
        if datetime.now(timezone.utc)>=end:raise TimeoutError('ORIGINAL_CAMPAIGN_END')
        time.sleep(60)
    finalize(a.source,a.compact,a.output,a.test_csv)
    print(json.dumps({'state':'EXPORTED','output':str(a.output)}))

if __name__=='__main__':main()
