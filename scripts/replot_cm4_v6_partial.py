#!/usr/bin/env python3
"""Offline accepted-record V6 partial release. No selection, model, OT or dispatch.

Other CM datasets are explicitly OLD_ORDER_THETA010_DIAGNOSTIC, not claimed
to have completed V6 reselection. Only accepted Taste R3/CM are in the main plot.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def reduce_columns(distances, source, candidates, cap, theta=.1):
    d=np.asarray(distances,dtype=float);source=np.asarray(source,dtype=bool)
    if d.ndim!=2 or d.shape[1]!=len(candidates) or len(candidates)>20 or not len(candidates):raise ValueError('Selected matrix shape')
    if np.isnan(d).any() or (d<0).any() or np.isfinite(d[~source]).any():raise ValueError('Unknown/non-source raw')
    previous=np.full(len(d),np.inf);rows=[]
    for k in range(1,21):
        best=d[:,:min(k,d.shape[1])].min(axis=1)
        rows.append(dict(k=k,effective_k=min(k,d.shape[1]),true_prefix_sha=sha(candidates[:k]),
            covered=int((best<=theta).sum()),finite=int(np.isfinite(best).sum()),
            new_covered=int(((best<=theta)&(previous>theta)).sum()),
            changed_best_distance=int((best!=previous).sum()),capped_mean=float(np.minimum(best,cap).mean())))
        previous=best
    return rows


def assemble(base,out):
    v5=base/'taste-k20-theta010-final-v5/run-20260916T070500Z';c=read(v5/'taste_eval_contract.json');cm=v5/'cm'
    audit=read(cm/'audit/final_audit.json');freeze=read(cm/'selection_freeze.json');chosen=freeze['selected_candidate_ids'];ids=c['test_ids']
    if audit['status']!='TASTE_V5_CM_ACCEPTED' or c['primary_theta']!=.1 or len(ids)!=468:raise ValueError('Source acceptance')
    matrix=np.full((len(ids),len(chosen)),np.nan);states=np.zeros(matrix.shape,dtype=np.uint8);mask=[];blocks=[]
    for b,start in enumerate(range(0,len(ids),8)):
        p=cm/f'test/block-{b:04d}';r=read(p.with_suffix('.json'));stop=min(start+8,len(ids))
        if r['parent_ids']!=ids[start:stop] or r['candidate_ids']!=chosen or r['freeze_sha256']!=freeze['freeze_sha256']:raise ValueError('Raw identity')
        raw=p.with_suffix('.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['npz_sha256']:raise ValueError('Changed compact raw block')
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:matrix[start:stop]=z['values'];states[start:stop]=z['states']
        blocks.append(r);mask.extend(r['source_mask'])
    mask=np.array(mask,dtype=bool)
    if not np.all(states[mask]==1) or not np.all(states[~mask]==2):raise ValueError('Incomplete semantic states')
    prefixes=reduce_columns(matrix,mask,chosen,c['cost_cap'])
    old=read(base/'cm-crem-global-v2/tastemolnet-postfilter-20260914/verified/test_prepared.json')
    new=set(chosen)-set(old['candidate_ids']);first=matrix[mask,0]
    schedule=[]
    for b in blocks:
        for pid,is_source in zip(b['parent_ids'],b['source_mask']):
            for cid in b['candidate_ids']:
                if is_source and cid in new and len(schedule)<16:schedule.append([pid,cid])
    sat=dict(status='RAW_PREFIX_REDUCTION_CONFIRMED_NOT_NEW_GRAPH_REAUDIT',source_audit=audit,
        first_prototype=chosen[0],first_source_count=int(mask.sum()),min=float(first.min()),p50=float(np.quantile(first,.5)),
        p95=float(np.quantile(first,.95)),max=float(first.max()),above_theta=int((first>.1).sum()),missing=int(np.isnan(first).sum()),
        zero_distance_count=int((first==0).sum()),logical_slots=int(matrix.size),new_prototypes=len(new),old_prototypes=len(chosen)-len(new),
        new_distances=sum(b['computed'] for b in blocks),old_slots=sum(b['reused'] for b in blocks),
        new_nonsource_slots=int((~mask).sum())*len(new),adopted_independent_checks=16,
        first_column_in_recorded_deterministic_audit_schedule=any(x[1]==chosen[0] for x in schedule),
        schedule_reconstruction_is_not_new_numeric_recomputation=True,
        remaining_evidence='REMOTE_SAVED_PREDICTION_AND_FULL_GRAPH_ENCODING_IDENTITY_REVIEW',new_ot_calls=0)
    write(out/'source_csv/cm_taste_saturation_audit.csv',prefixes)
    write(out/'source_csv/cm_taste_first_prototype_raw.csv',[dict(parent_id=pid,source=bool(mask[i]),candidate_id=chosen[0],
        raw_distance=float(matrix[i,0]) if mask[i] else 'inf',state='OK' if mask[i] else 'BEFORE_NOT_SOURCE') for i,pid in enumerate(ids)])
    dump(out/'audit/cm_taste_saturation.json',sat)
    dump(out/'audit/adopted_16_pair_schedule.json',schedule)
    # Adopt already accepted R3 and V5 exports, reconcile against actual raw.
    f2=csvread(v5/'source_csv/figure3_taste_theta010.csv');f3=csvread(v5/'source_csv/figure4_taste_k20_raw_ecdf.csv')
    table=csvread(v5/'source_csv/table2_taste_k20_theta010.csv')
    for r in f2:
        if r['method']=='CM-CReM-Global':
            a=prefixes[int(r['k'])-1]
            if int(r['covered_count'])!=a['covered'] or abs(float(r['fixed_capped_mean'])-a['capped_mean'])>1e-15:raise ValueError('Published/raw prefix conflict')
    for r in f3:
        if r['method']=='CM-CReM-Global' and int(r['covered_count'])!=int((matrix.min(axis=1)<=float(r['theta'])).sum()):raise ValueError('ECDF/raw conflict')
    write(out/'source_csv/figure2_coverage_cost_vs_k.csv',f2)
    write(out/'source_csv/figure3_coverage_vs_theta.csv',f3)
    write(out/'source_csv/table2_k20_theta010_versioned.csv',table)
    write(out/'source_csv/parent_best_distances_selected.csv',csvread(cm/'source_csv/parent_best_distances.csv'))
    paths={
        'AIDS':base/'cm-aids-k20-closeout-20260914/run-20260914T151000Z/cm_aids/verified',
        'Mutagenicity':base/'cm-crem-global-v2/mutagenicity-postfilter-20260914/verified',
        'BACE':base/'cm-crem-global-v2/bace-k20-20260910/import',
        'TasteMolNet':cm}
    scopes=[];diagnostics=[]
    for dataset,p in paths.items():
        ev=read(p/'test_evaluation.json');selection=ev['selection'];best=np.array(ev['best_distances_uncapped'],dtype=float)[19]
        cap=selection['cap'];finite=best[np.isfinite(best)];description=dataset=='AIDS'
        scope=read(p/'scope_contract.json') if description else {}
        scopes.append(dict(dataset=dataset,oracle='RF' if dataset in ('AIDS','Mutagenicity') else 'ORIGINAL_GINE',
            mask_source='RF_FEATURE_OCCLUSION' if dataset in ('AIDS','Mutagenicity') else 'FROZEN_GINE_GRAD_CAM',
            operation='FULL_GRAPH_GLOBAL_PROTOTYPE',pool_count=len(selection['pool_candidate_ids']),
            selection_count=len(selection['calibration_parent_ids']),evaluation_count=len(ev['parent_ids']),
            source_count=sum(ev['source_mask']),selection_evaluation_overlap=len(set(selection['calibration_parent_ids'])&set(ev['parent_ids'])),
            scope='SOURCE-DESCRIPTIVE_NON_HELDOUT' if description else 'ORIGINAL_HELDOUT_BASE',
            k=20,theta=.1,cost_cap=cap,original_selection_theta=selection['theta'],
            status='ACCEPTED_V5_ADOPTED' if dataset=='TasteMolNet' else 'PENDING_V6_CALIBRATION_RESELECTION',
            freeze_sha=selection['freeze_sha256'],source_root=str(p)))
        diagnostics.append(dict(dataset=dataset,scope=scopes[-1]['scope'],version='ACCEPTED_V5' if dataset=='TasteMolNet' else 'OLD_ORDER_THETA010_DIAGNOSTIC_NOT_V6_FINAL',
            k=20,theta=.1,covered=int((best<=.1).sum()),base=len(best),source=sum(ev['source_mask']),finite=len(finite),
            capped_mean=float(np.minimum(best,cap).mean()),conditional_median=float(np.median(finite)) if len(finite) else 'N/A',cost_cap=cap))
    write(out/'source_csv/cm_four_dataset_scope.csv',scopes)
    write(out/'source_csv/cm_old_order_theta010_diagnostic.csv',diagnostics)
    dump(out/'audit/release.json',dict(status='PARTIAL_ACCEPTED_TASTE_ONLY',accepted_taste_methods=2,total_taste_methods=5,
        other_datasets_final_v6=False,new_graph_audit_complete=False,original_authority_unchanged=True,
        previous_figure3='Figure2',previous_figure4='Figure3',source_root=str(v5)))
    print(json.dumps(dict(saturation=sat,diagnostics=diagnostics),indent=2))


def plot(out):
    rows=csvread(out/'source_csv/figure2_coverage_cost_vs_k.csv');ecdf=csvread(out/'source_csv/figure3_coverage_vs_theta.csv')
    table=csvread(out/'source_csv/table2_k20_theta010_versioned.csv');figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42})
    style={'Ours':('black','s'),'CM-CReM-Global':('#008000','*')}
    fig,axes=plt.subplots(1,2,figsize=(8.8,2.6));fig.subplots_adjust(bottom=.27,wspace=.30,top=.85)
    for method,(color,marker) in style.items():
        group=[r for r in rows if r['method']==method];k=np.array([int(r['k']) for r in group])
        for ax,key,mul in [(axes[0],'coverage',100),(axes[1],'fixed_capped_mean',1)]:
            y=np.array([float(r[key])*mul for r in group]);ax.plot(k,y,color=color,lw=1,label=method)
            ix=np.isin(k,[1,5,10,15,20]);ax.plot(k[ix],y[ix],ls='none',marker=marker,ms=4,color=color)
            ax.set_xlim(1,20);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Number of prototypes / rules (K)');ax.grid(alpha=.35)
    axes[0].set_ylim(0,70);axes[0].set_yticks([0,20,40,60,70]);axes[0].set_ylabel('Coverage (%)')
    ceiling=100*285/468;axes[0].axhline(ceiling,color='.6',ls=':',lw=.7);axes[0].text(1.3,64,'Source ceiling: 285/468',fontsize=8)
    axes[1].set_ylim(0,.035);axes[1].set_yticks([0,.01,.02,.03,.035]);axes[1].set_ylabel('Capped mean cost')
    fig.suptitle('TasteMolNet · theta = 0.1 · PARTIAL 2/5',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    for ext in ('pdf','png'):fig.savefig(figdir/f'figure2_coverage_cost_vs_k.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,2.6));fig.subplots_adjust(bottom=.25,top=.85)
    for method,(color,marker) in style.items():
        group=[r for r in ecdf if r['method']==method];x=np.array([float(r['theta']) for r in group]);y=np.array([100*float(r['coverage']) for r in group])
        ax.step(x,y,where='post',color=color,lw=1,label=method)
    ax.axvline(.1,color='.5',ls='--',lw=.7);ax.axhline(ceiling,color='.6',ls=':',lw=.7)
    ax.set(xlim=(0,.2),ylim=(0,70),xlabel='Raw WNode threshold',ylabel='Coverage (%)',title='TasteMolNet · K = 20 · exact ECDF · PARTIAL 2/5')
    ax.set_xticks([0,.05,.1,.15,.2]);ax.set_yticks([0,20,40,60,70]);ax.grid(alpha=.35)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    for ext in ('pdf','png'):fig.savefig(figdir/f'figure3_coverage_vs_theta.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    accepted=[r for r in table if r['status']=='ACCEPTED'];tex=['\\begin{tabular}{lrrrr}','\\toprule','Method & Covered & Coverage (\\%) & Capped mean & Conditional median \\\\','\\midrule']
    for r in accepted:tex.append(f"{r['method']} & {r['covered_count']}/468 & {100*float(r['coverage']):.2f} & {float(r['fixed_capped_mean']):.8f} & {float(r['conditional_median']):.8f} \\\\")
    tex.extend(['\\bottomrule','\\end{tabular}']);(figdir/'table2_k20_theta010.tex').write_text('\n'.join(tex)+'\n')
    fig,ax=plt.subplots(figsize=(8.8,1.75));ax.axis('off')
    cells=[[r['method'],r['covered_count']+'/468',f"{100*float(r['coverage']):.2f}",f"{float(r['fixed_capped_mean']):.8f}",f"{float(r['conditional_median']):.8f}"] for r in accepted]
    t=ax.table(cellText=cells,colLabels=['Method','Covered','Cov. (%)','Capped mean','Cond. median'],loc='center',cellLoc='center',edges='horizontal')
    t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.7)
    ax.set_title('TasteMolNet · K20 / theta 0.1 · PARTIAL: accepted rows only',fontsize=10)
    fig.text(.5,.02,'GCFExplainer, GlobalGCE and ComRecGC remain PENDING; no zero placeholders.',ha='center',fontsize=8)
    for ext in ('pdf','png'):fig.savefig(figdir/f'table2_k20_theta010.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--archive-base',type=Path);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--replot-only',action='store_true')
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    if not a.replot_only:assemble(a.archive_base,a.out_dir)
    plot(a.out_dir)
if __name__=='__main__':main()
