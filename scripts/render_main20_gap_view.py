#!/usr/bin/env python3
"""Offline, versioned saved-distance view; not a scientific publisher."""
import argparse,csv,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.main20_saved_view import read_csv,write_csv,reduce_distances,exact_curve,close

DATASETS=['AIDS','Mutagenicity','BACE','TasteMolNet']
METHODS=['GlobalGCE','COMRECGC','GCFExplainer','Ours','CM-CReM']

def adapt(rows,d,m,theta,cap,parent_rows,source):
    prefix=[];curves=[]
    for k in range(1,21):
        saved=next(r for r in rows if int(r['k'])==k)
        parents=[r for r in parent_rows if int(r['k'])==k]
        metric,vals=reduce_distances(parents,theta,cap)
        if not close(metric['coverage'],float(saved['coverage'])):
            raise ValueError(f'{d}/{m}/{k}: saved coverage/raw conflict')
        costkey='conditional_median_cost' if d in ['AIDS','Mutagenicity'] else 'fixed_capped_mean_cost'
        prefix.append(dict(dataset=d,method=m,k=k,cost=metric[costkey],cost_definition=costkey,
            **metric,theta_star=theta,cap=cap,effective_k=saved.get('effective_k',saved.get('k_effective',k)),
            source_root=str(source),stage='VERSIONED_SAVED_RAW_REDUCED',plot_allowed=True,
            scientific_recomputation=False,full_cross_method_reaudit=False))
        if k in [10,20]:curves.extend(dict(dataset=d,method=m,k=k,theta_star=theta,**x) for x in exact_curve(vals,theta))
    return prefix,curves

def build(a):
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);(out/'source_csv').mkdir()
    rows=read_csv(Path(a.prior_view)/'source_csv/figure3.csv')
    curves=[]
    for k in [10,20]:curves+=read_csv(Path(a.prior_view)/f'source_csv/figure4_k{k}_exact.csv')
    for r in rows:
        r['method']=r['method'].replace('ComRecGC','COMRECGC')
        r['k']=int(r['k']);r['plot_allowed']=str(r['plot_allowed']).lower()=='true'
        for key in ['coverage','cost','theta_star']:
            try:r[key]=float(r[key])
            except (TypeError,ValueError):pass
    for r in curves:
        r['method']=r['method'].replace('ComRecGC','COMRECGC');r['k']=int(r['k'])
        r['distance']=float(r['distance']);r['coverage']=float(r['coverage'])
    sources=[];corrections=[]
    def replace(d,m,p,c):
        nonlocal rows,curves
        rows=[r for r in rows if (r['dataset'],r['method'])!=(d,m)]+p
        curves=[r for r in curves if (r['dataset'],r['method'])!=(d,m)]+c
    # Reconstruct available original numeric rows from sealed parent minima,
    # not an old presentation CSV. This does not repeat oracle or WNode work.
    for d,m in [('BACE','ComRecGC'),('BACE','GCFExplainer'),('BACE','Ours'),('TasteMolNet','Ours')]:
        root=Path(a.original_cells)/d/m
        summary=json.loads((root/'summary.json').read_text())
        p,c=adapt(read_csv(root/'prefix_metrics.csv'),d,m.replace('ComRecGC','COMRECGC'),
            float(summary['theta_star']),float(summary['cost_cap']),read_csv(root/'parent_best_distances.csv'),root)
        for new in p:
            old=next((r for r in rows if (r['dataset'],r['method'],r['k'])==(d,new['method'],new['k'])),None)
            if old and isinstance(old['cost'],(float,int)) and not close(old['cost'],new['cost']):
                corrections.append(dict(dataset=d,method=new['method'],k=new['k'],
                    presentation_cost=old['cost'],raw_reduced_cost=new['cost'],source_root=str(root)))
        replace(d,m.replace('ComRecGC','COMRECGC'),p,c)
    for d,root in [('BACE',a.cm_bace),('Mutagenicity',a.cm_mut),('TasteMolNet',a.cm_taste)]:
        root=Path(root);pr=read_csv(root/'prefix_metrics.csv');parent=read_csv(root/'parent_best_distances.csv')
        table=read_csv(root/'table2_k20.csv')[0]
        parent=[dict(r,best_distance=r['best_distance_uncapped'],strict_recourse_available=r['finite_strict_flip']) for r in parent]
        p,c=adapt(pr,d,'CM-CReM',float(table['theta']),float(table['cap']),parent,root)
        replace(d,'CM-CReM',p,c);sources.append(dict(dataset=d,path=str(root),generation_rerun=False,ot_rerun=False))
    recovery=Path(a.mut_recovery)
    for m in ['GCFExplainer','GlobalGCE']:
        root=recovery/m;pr=read_csv(root/'prefix_metrics.csv');pa=read_csv(root/'parent_best_distances.csv')
        p,c=adapt(pr,'Mutagenicity',m,float(pr[0]['theta_star']),float(pr[0]['cost_cap']),pa,root);replace('Mutagenicity',m,p,c)
    theta_mut=next(r['theta_star'] for r in rows if r['dataset']=='Mutagenicity' and r['method']=='CM-CReM')
    conflicts=[]
    for r in rows:
        if r['dataset']=='Mutagenicity' and not close(r['theta_star'],theta_mut):
            r['plot_allowed']=False;r['primary_contract_state']='PENDING_ORIGINAL_THETA_REDUCTION'
            if int(r['k'])==20:conflicts.append(dict(method=r['method'],saved_theta=r['theta_star'],required_theta=theta_mut))
    allowed={(r['dataset'],r['method']) for r in rows if r['plot_allowed']}
    curves=[r for r in curves if (r['dataset'],r['method']) in allowed]
    write_csv(out/'source_csv/figure3_versioned.csv',rows)
    for k in [10,20]:write_csv(out/f'source_csv/figure4_k{k}_exact.csv',[r for r in curves if r['k']==k])
    table=[]
    for d in DATASETS:
        for m in METHODS:
            found=next((dict(r) for r in rows if r['dataset']==d and r['method']==m and r['k']==20),None)
            row=found or dict(dataset=d,method=m,k=20,coverage='PENDING',cost='PENDING',stage='PENDING')
            if not row.get('plot_allowed',False):
                row.update(saved_version_coverage=row.get('coverage'),saved_version_cost=row.get('cost'),
                    saved_version_finite_recourse=row.get('finite_recourse_count'),coverage='PENDING',cost='PENDING',finite_recourse_count='PENDING')
            for key in ['N','finite_recourse_count']:
                if row.get(key)=='N/A':row[key]='PENDING'
            table.append(row)
    write_csv(out/'source_csv/table2_k20.csv',table)
    counts=dict(registered=16,versioned_k20_numeric_complete=10,combined_release_ready=5,total=20,
        count_basis='13 original authority +3 CM imports; numeric10 already includes new CM results; release5 retained, not inferred from numeric equality',
        new_scientific_authority=False,cm_median_theta_filtered=False,cm_sources=sources,
        derived_export_corrections=corrections,
        unmatched_mut_theta=conflicts,publication='PARTIAL_VERSIONED_NUMERICAL_VIEW_NOT_FINAL_COMBINED_AUDIT')
    for d in ['mutagenicity','tastemolnet']:
        receipt=json.loads((Path(a.release_receipts)/(d+'.json')).read_text())
        if receipt['status']!='CM_RESULT_IMPORT_VERIFIED':raise ValueError('CM actual import absent')
    (out/'audit.json').write_text(json.dumps(counts,indent=2))
    render(out,rows,curves,table)
    return counts

def render(out,rows,curves,table):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    colors=dict(zip(METHODS,['#cc3333','#26833b','#111111','#b42eb4','#257aa8']))
    marks=dict(zip(METHODS,['x','*','s','^','o']))
    def finish(fig,name):
        fig.tight_layout(rect=(0,.13,1,.92));fig.legend([plt.Line2D([],[],color=colors[m],marker=marks[m],lw=1) for m in METHODS],METHODS,
            loc='upper center',ncol=5,frameon=False,fontsize=9)
        fig.text(.5,.025,'PARTIAL versioned RF/GINE numerical view; unresolved contracts omitted, not zero. Not final combined scientific acceptance.',ha='center',fontsize=7)
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=220)
        plt.close(fig)
    fig,ax=plt.subplots(2,4,figsize=(13.2,4.3));nodes=[1,3,6,9,12,15,18,20]
    for i,d in enumerate(DATASETS):
        ax[0,i].set_title(d,fontsize=11)
        for m in METHODS:
            rs=[r for r in rows if r['dataset']==d and r['method']==m and r['plot_allowed']]
            for j,key in enumerate(['coverage','cost']):
                val=[r for r in rs if isinstance(r[key],(int,float)) and math.isfinite(r[key]) and r['k'] in nodes]
                if len(val)<2:continue
                val=sorted(val,key=lambda r:r['k']);x=np.array([r['k'] for r in val]);y=np.array([r[key] for r in val])*(100 if j==0 else 1)
                xx=np.linspace(x[0],x[-1],240);ax[j,i].plot(xx,PchipInterpolator(x,y)(xx),color=colors[m],lw=1)
                mask=(x>1)&(x<20);ax[j,i].plot(x[mask],y[mask],linestyle='none',marker=marks[m],color=colors[m],ms=4)
        for j in range(2):
            ax[j,i].set_xlim(1,20);ax[j,i].set_xticks([1,5,10,15,20]);ax[j,i].set_ylim(bottom=0);ax[j,i].grid(alpha=.3)
            hi=ax[j,i].get_ylim()[1];ax[j,i].set_yticks(np.linspace(0,hi,5))
        ax[1,i].set_xlabel('Size (K)');ax[0,i].set_ylabel('Coverage (%)' if i==0 else '')
        ax[1,i].set_ylabel('Conditional median' if d in ['AIDS','Mutagenicity'] else 'Capped mean')
    finish(fig,'figure3_k20_partial')
    for k in [10,20]:
        fig,axs=plt.subplots(1,4,figsize=(13.2,2.8))
        for ax,d in zip(axs,DATASETS):
            ax.set_title(d,fontsize=11)
            drawn=False
            for m in METHODS:
                rs=sorted([r for r in curves if r['dataset']==d and r['method']==m and r['k']==k],key=lambda r:r['distance'])
                if not rs:continue
                drawn=True
                ax.step([r['distance'] for r in rs],[100*r['coverage'] for r in rs],where='post',color=colors[m],lw=1)
            if not drawn:
                ax.text(.5,.5,'PENDING raw K20 distances' if k==20 else 'PENDING exact distances',
                    transform=ax.transAxes,ha='center',va='center',fontsize=9)
                ax.set_xlim(0,1);ax.set_ylim(0,100)
            ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.grid(alpha=.3)
            hi=min(100,max(5,math.ceil(ax.get_ylim()[1]/5)*5))
            ax.set_ylim(0,hi);ax.set_yticks(np.linspace(0,hi,5));ax.set_xlabel('Distance threshold')
        axs[0].set_ylabel('Coverage (%)');finish(fig,f'figure4_k{k}_partial')
    fig,ax=plt.subplots(figsize=(11.7,6.2));ax.axis('off')
    text=[]
    for r in table:
        f=lambda v:f'{float(v):.6f}' if isinstance(v,(int,float)) else str(v)
        text.append([r['dataset'],r['method'],f(r['coverage']),f(r['cost']),str(r.get('finite_recourse_count','PENDING')),str(r.get('N','PENDING'))])
    t=ax.table(cellText=text,colLabels=['Dataset','Method','Cov@20','Main cost@20','Finite recourse','N'],loc='center',cellLoc='center')
    t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.28)
    ax.set_title('Table 2 @ K20 — PARTIAL versioned numerical view',fontsize=12)
    fig.text(.5,.01,'AIDS/Mut: conditional median without theta filtering. BACE/Taste: fixed capped mean. PENDING is not zero.',ha='center',fontsize=8)
    fig.savefig(out/'table2_k20.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['prior-view','original-cells','cm-bace','cm-mut','cm-taste','mut-recovery','release-receipts','out-dir']:p.add_argument('--'+n,required=True)
    print(json.dumps(build(p.parse_args()),indent=2))
