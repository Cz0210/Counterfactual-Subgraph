"""Offline CM4 V6 publication: adopted receipts plus transferred compact raw.

No selection, oracle, OT, registry mutation or interpolation. AIDS is explicitly
descriptive, and other methods without a matching V6 record remain unplotted.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

DATASETS=('AIDS','Mutagenicity','BACE','TasteMolNet')
CM='CM-CReM-Global'

def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)

def verify_raw(root):
    """One transfer check per compact block, and actual column-wise prefix audit."""
    f=read(root/'selection_freeze.json');audit=read(root/'audit/final_audit.json')
    if audit['status']!='CM4_V6_ACCEPTED' or audit['contract_sha256']!=f['contract_sha256']:raise ValueError('Not accepted')
    if f['theta']!=.1 or len(f['selected_candidate_ids'])>20:raise ValueError('V6 contract')
    blocks=[];ids=[];mask=[]
    for p in sorted((root/'test').glob('block-*.json')):
        r=read(p);npz=p.with_suffix('.npz')
        if hashlib.sha256(npz.read_bytes()).hexdigest()!=r['npz_sha256']:raise ValueError('Transfer raw checksum')
        if r['candidate_ids']!=f['selected_candidate_ids'] or r['freeze_sha256']!=f['freeze_sha256']:raise ValueError('Column identity')
        with np.load(npz,allow_pickle=False) as z:
            d=z['values'].copy();s=z['states'];m=np.asarray(r['source_mask'],dtype=bool)
            if np.isnan(d).any() or not np.all(s[m]==1) or not np.all(s[~m]==2):raise ValueError('Unknown/error raw')
            if np.isfinite(d[~m]).any() or not np.isfinite(d[m]).all():raise ValueError('Raw semantic mask')
        blocks.append(d);ids.extend(r['parent_ids']);mask.extend(r['source_mask'])
    d=np.concatenate(blocks);prefix=rows(root/'source_csv/prefix_metrics.csv')
    if len(ids)!=len(set(ids)) or len(prefix)!=20:raise ValueError('Incomplete/duplicate cohort')
    for k,r in enumerate(prefix,1):
        best=d[:,:min(k,d.shape[1])].min(axis=1)
        if int(r['covered_count'])!=int((best<=.1).sum()) or abs(float(r['cost'])-np.minimum(best,f['cap']).mean())>1e-15:raise ValueError('Prefix/raw conflict')
    return prefix,ids,mask,d,f,audit

def assemble(source,hpc,out):
    out.mkdir(parents=True,exist_ok=True)
    taste_rows=rows(source/'source_csv/figure2_coverage_cost_vs_k.csv')
    f2=[dict(dataset='TasteMolNet',**r) for r in taste_rows]
    taste_ecdf=rows(source/'source_csv/figure3_coverage_vs_theta.csv')
    f3=[dict(dataset='TasteMolNet',method=r['method'],k=r['k'],theta=r['theta'],covered_count=r['covered_count'],coverage=r['coverage'],N_base=r['N_base']) for r in taste_ecdf]
    table=[dict(dataset='TasteMolNet',**r) for r in rows(source/'source_csv/table2_k20_theta010_versioned.csv') if r['status']=='ACCEPTED']
    scope=rows(source/'source_csv/cm_four_dataset_scope.csv');proofs={};all_parent=[]
    for row in scope:
        row.update(oracle_sha256='',temperature='',version='',contract_sha256='',calibration_id_digest='',evaluation_id_digest='')
    taste_parent=rows(source/'source_csv/parent_best_distances_selected.csv')
    for r in taste_parent:all_parent.append(dict(dataset='TasteMolNet',method=CM,**r))
    for dataset in DATASETS[:-1]:
        root=hpc/dataset;p,ids,mask,d,f,audit=verify_raw(root);proofs[dataset]=audit
        for r in p:
            f2.append(dict(dataset=dataset,method=CM,k=r['k'],effective_k=r['effective_k'],N_base=r['base_parent_count'],N_source=r['source_parent_count'],theta=r['theta'],cap=r['cap'],covered_count=r['covered_count'],coverage=r['coverage'],finite_count=r['finite_recourse_count'],fixed_capped_mean=r['fixed_capped_mean_cost'],conditional_median=r['conditional_median_cost']))
        best=d.min(axis=1)
        for t in sorted({0.,.1,.2,*best[np.isfinite(best)].tolist()}):
            f3.append(dict(dataset=dataset,method=CM,k=20,theta=t,covered_count=int((best<=t).sum()),coverage=float((best<=t).mean()),N_base=len(best)))
        sc=next(r for r in scope if r['dataset']==dataset)
        c=read(root/'contract.json')
        id_digest=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
        sc.update(status='CM4_V6_ACCEPTED',freeze_sha=f['freeze_sha256'],source_root=str(root),original_selection_theta=.1,
            oracle_sha256=c['oracle_sha256'],temperature=c['temperature'],version='CM4_V6',contract_sha256=f['contract_sha256'],
            calibration_id_digest=id_digest(f['calibration_parent_ids']),evaluation_id_digest=id_digest(ids))
        r=f2[-1];table.append(dict(**r,status='ACCEPTED',scope=sc['scope']))
        for r in rows(root/'source_csv/parent_best_distances.csv'):all_parent.append(dict(dataset=dataset,method=CM,**r))
    first=read(hpc/'taste-flat-audit/first_prototype_audit.json')
    if first['status']!='FIRST_PROTOTYPE_GRAPH_AND_8_EXACT_PAIRS_PASS' or first['new_ot_calls']!=8:raise ValueError('C1 independent proof absent')
    sat=read(source/'audit/cm_taste_saturation.json')
    if sat['first_prototype']!=first['source_pairs'][0]['candidate_id'] or sat['max']!=first['first_max']:raise ValueError('Saturation identity conflict')
    taste_contract=read(Path(read(source/'audit/release.json')['source_root'])/'taste_eval_contract.json')
    next(r for r in scope if r['dataset']=='TasteMolNet').update(oracle_sha256=taste_contract['oracle_sha256'],
        temperature=taste_contract['temperature'],version='CM_V5_ADOPTED_PLUS_V6_C1_AUDIT',contract_sha256=sat['source_audit']['contract_sha256'],
        calibration_id_digest=id_digest(taste_contract['calibration_ids']),evaluation_id_digest=id_digest(taste_contract['test_ids']))
    for r in table:
        a=next(v for v in f2 if v['dataset']==r['dataset'] and v['method']==r['method'] and int(v['k'])==20)
        b=next(v for v in f3 if v['dataset']==r['dataset'] and v['method']==r['method'] and float(v['theta'])==.1)
        if int(a['covered_count'])!=int(r['covered_count']) or int(b['covered_count'])!=int(r['covered_count']):raise ValueError('Cross-figure theta/K20 inconsistency')
    # Ours selected-distance CSV is separate; do not fabricate an Ours row in CM raw.
    write(out/'source_csv/figure2_coverage_cost_vs_k.csv',f2)
    write(out/'source_csv/figure3_coverage_vs_theta.csv',f3)
    write(out/'source_csv/table2_k20_theta010_versioned.csv',table)
    write(out/'source_csv/cm_four_dataset_scope.csv',scope)
    write(out/'source_csv/parent_best_distances_selected.csv',all_parent)
    write(out/'source_csv/cm_taste_saturation_audit.csv',rows(source/'source_csv/cm_taste_saturation_audit.csv'))
    dump(out/'audit/cm_taste_first_prototype_audit.json',first)
    dump(out/'audit/cm_taste_saturation.json',dict(**sat,independent_v6_status=first['status']))
    dump(out/'audit/final_audit.json',dict(status='CM_FOUR_DATASET_V6_ACCEPTED_TASTE_COMPARISON_PARTIAL',cm_accepted_datasets=4,
        taste_methods_accepted=2,taste_methods_total=5,AIDS_heldout=False,original_matrix_written=False,
        original_main_completed=13,original_main_total=16,extended_registered=16,extended_total=20,
        v6_numeric_rows=5,v6_complete_cm_cells=4,not_all_twenty_cells_same_scope=True,
        source_release=str(source),hpc_import=str(hpc),cm_receipts=proofs,
        first_prototype_independent_pairs=8,original_temperature_and_generation_repeated=False))

def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42})
    f2=rows(out/'source_csv/figure2_coverage_cost_vs_k.csv');f3=rows(out/'source_csv/figure3_coverage_vs_theta.csv');table=rows(out/'source_csv/table2_k20_theta010_versioned.csv')
    target=out/'figures';target.mkdir(exist_ok=True);styles={CM:('#008000','*'),'Ours':('black','s')}
    def save(fig,name):
        for ext in ('pdf','png'):fig.savefig(target/(name+'.'+ext),dpi=320,bbox_inches='tight')
        plt.close(fig)
    ceilings={d:max(float(r['N_source'])/float(r['N_base']) for r in f2 if r['dataset']==d) for d in DATASETS}
    ymax={d:min(105.,5*np.ceil((ceilings[d]*100+5)/5)) for d in DATASETS}
    fig,axes=plt.subplots(2,4,figsize=(13.2,4.5));fig.subplots_adjust(wspace=.33,hspace=.24,bottom=.17,top=.88)
    for j,dataset in enumerate(DATASETS):
        for method,(color,marker) in styles.items():
            g=[r for r in f2 if r['dataset']==dataset and r['method']==method]
            if not g:continue
            k=np.array([int(r['k']) for r in g]);ix=np.isin(k,[1,5,10,15,20])
            for ax,key,mul in [(axes[0,j],'coverage',100),(axes[1,j],'fixed_capped_mean',1)]:
                y=np.array([float(r[key])*mul for r in g]);ax.plot(k,y,color=color,lw=1,label=method);ax.plot(k[ix],y[ix],ls='none',marker=marker,ms=4,color=color)
        cap=max(float(r['cap']) for r in f2 if r['dataset']==dataset)
        axes[0,j].set_ylim(0,ymax[dataset]);axes[0,j].set_yticks(np.linspace(0,ymax[dataset],5));axes[0,j].axhline(100*ceilings[dataset],c='.6',ls=':',lw=.7)
        axes[0,j].set_title(dataset+('\nSOURCE-DESCRIPTIVE' if dataset=='AIDS' else ''))
        axes[1,j].set_ylim(0,cap*1.08);axes[1,j].set_yticks(np.linspace(0,cap*1.08,5));axes[1,j].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[1,j].set_xlabel('K')
        for ax in axes[:,j]:ax.set_xlim(1,20);ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.3)
    axes[0,0].set_ylabel('Coverage (%)');axes[1,0].set_ylabel('Capped mean cost')
    fig.legend(*axes[0,3].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Figure 2 | theta = 0.1 | CM four datasets; cross-method comparison PARTIAL',fontsize=10)
    save(fig,'figure2_coverage_cost_vs_k')
    fig,axes=plt.subplots(1,4,figsize=(13.2,2.7));fig.subplots_adjust(wspace=.30,bottom=.27,top=.79)
    for ax,dataset in zip(axes,DATASETS):
        for method,(color,_) in styles.items():
            g=[r for r in f3 if r['dataset']==dataset and r['method']==method]
            if g:ax.step([float(r['theta']) for r in g],[100*float(r['coverage']) for r in g],where='post',c=color,lw=1,label=method)
        ax.axvline(.1,c='.5',ls='--',lw=.7);ax.axhline(100*ceilings[dataset],c='.6',ls=':',lw=.7)
        ax.set(xlim=(0,.2),ylim=(0,ymax[dataset]),xlabel='Raw WNode threshold',title=dataset+('\nSOURCE-DESCRIPTIVE' if dataset=='AIDS' else ''))
        ax.set_xticks([0,.05,.1,.15,.2]);ax.set_yticks(np.linspace(0,ymax[dataset],5));ax.grid(alpha=.3)
    axes[0].set_ylabel('Coverage (%)');fig.legend(*axes[3].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Figure 3 | K20 | Exact ECDF; dotted line = source ceiling | comparison PARTIAL',fontsize=10)
    save(fig,'figure3_coverage_vs_theta')
    cells=[];tex=['\\begin{tabular}{lllrrr}','\\toprule','Dataset & Method & Covered & Cov. (\\%) & Capped mean & Cond. median \\\\','\\midrule']
    for dataset in DATASETS:
        for r in table:
            if r['dataset']!=dataset:continue
            cells.append([dataset+(' (descriptive)' if dataset=='AIDS' else ''),r['method'],r['covered_count']+'/'+r['N_base'],f"{100*float(r['coverage']):.2f}",f"{float(r['fixed_capped_mean']):.8f}",f"{float(r['conditional_median']):.8f}"])
            tex.append(' & '.join(cells[-1])+r' \\')
    tex.extend(['\\bottomrule','\\end{tabular}']);(target/'table2_k20_theta010.tex').write_text('\n'.join(tex)+'\n')
    fig,ax=plt.subplots(figsize=(13.2,2.9));ax.axis('off');t=ax.table(cellText=cells,colLabels=['Dataset','Method','Covered','Cov. (%)','Capped mean','Cond. median'],loc='center',cellLoc='center',edges='horizontal')
    t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.5);ax.set_title('Table 2 | K20 / theta 0.1 | Accepted versioned rows only; comparison PARTIAL',fontsize=10)
    fig.text(.5,.025,'AIDS is SOURCE-DESCRIPTIVE, not held-out. Missing Taste baselines are PENDING, not zero.',ha='center',fontsize=8)
    save(fig,'table2_k20_theta010')

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-release',type=Path);p.add_argument('--hpc-import',type=Path);p.add_argument('--out-dir',required=True,type=Path);p.add_argument('--replot-only',action='store_true');a=p.parse_args()
    if not a.replot_only:assemble(a.source_release,a.hpc_import,a.out_dir)
    plot(a.out_dir)

if __name__=='__main__':main()
