#!/usr/bin/env python3
"""Offline V5 paper export. Accepted CSV/raw only; no model, selector or OT."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def assemble(root,ours,cm):
    c=json.loads((root/'taste_eval_contract.json').read_text());n=len(c['test_ids']);cap=c['cost_cap'];theta=c['primary_theta']
    if n!=468 or theta!=.1 or c['primary_report_k']!=20:raise ValueError('Not V5')
    prefix=[];best={};method_meta={}
    source=read(ours/'source_csv/figure3_taste_ours_theta010.csv')
    rows=[r for r in source if r['variant']=='R3' and float(r['theta'])==theta]
    if len(rows)!=20 or [int(r['k_requested']) for r in rows]!=list(range(1,21)):raise ValueError('Accepted R3 prefixes incomplete')
    for r in rows:
        if r['oracle_id']!=c['oracle_sha256'] or float(r['original_cost_cap'])!=cap or int(r['N_base'])!=n:raise ValueError('Ours scope conflict')
        prefix.append(dict(method='Ours',k=int(r['k_requested']),effective_k=int(r['k_effective']),N_base=n,N_source=int(r['N_source']),
            theta=theta,cap=cap,covered_count=int(r['num_covered']),coverage=float(r['coverage_fraction']),
            finite_count=int(r['finite_recourse_count']),fixed_capped_mean=float(r['fixed_capped_mean_cost']),
            conditional_median=float(r['conditional_median_all_finite']) if r['conditional_median_all_finite']!='N/A' else 'N/A'))
    raw=[r for r in read(ours/'source_csv/parent_best_distances_uncapped.csv') if r['variant']=='R3' and int(r['k'])==20]
    if [r['parent_id'] for r in raw]!=c['test_ids']:raise ValueError('Ours ordered test IDs differ')
    if any(r['state'] not in ['FINITE_VALID_STRICT_FLIP','PROVEN_NO_VALID_RECOURSE'] for r in raw):raise ValueError('Unresolved Ours raw')
    best['Ours']=np.array([float(r['raw_best_distance']) for r in raw]);method_meta['Ours']='R3 (R2 ordered-sequence alias); connected deletion'
    if cm is not None and (cm/'audit/final_audit.json').exists():
        audit=json.loads((cm/'audit/final_audit.json').read_text())
        if audit['status']!='TASTE_V5_CM_ACCEPTED':raise ValueError('CM not accepted')
        rawcm=json.loads((cm/'test_evaluation.json').read_text())
        if rawcm['parent_ids']!=c['test_ids'] or sum(rawcm['source_mask'])!=c['source_test']:raise ValueError('CM cohort conflict')
        if rawcm['selection']['theta']!=theta or rawcm['selection']['cap']!=cap:raise ValueError('CM theta/cap conflict')
        for r in read(cm/'source_csv/prefix_metrics.csv'):
            prefix.append(dict(method='CM-CReM-Global',k=int(r['k']),effective_k=int(r['effective_k']),N_base=int(r['base_parent_count']),
                N_source=int(r['source_parent_count']),theta=float(r['theta']),cap=float(r['cap']),covered_count=int(r['covered_count']),
                coverage=float(r['coverage']),finite_count=int(r['finite_recourse_count']),fixed_capped_mean=float(r['fixed_capped_mean_cost']),
                conditional_median=float(r['conditional_median_cost']) if r['conditional_median_cost'] else 'N/A'))
        best['CM-CReM-Global']=np.array([float(x) for x in rawcm['best_distances_uncapped'][19]])
        method_meta['CM-CReM-Global']='CM original full-graph prototypes; calibration reselected at theta0.1'
    ecdf=[];table=[];checks=[]
    for method,b in best.items():
        if np.isnan(b).any():raise ValueError('UNKNOWN cannot be rendered as failure')
        group=[r for r in prefix if r['method']==method];tail=group[-1];finite=b[np.isfinite(b)]
        if (tail['covered_count']!=int((b<=theta).sum()) or tail['finite_count']!=len(finite)
                or not math.isclose(tail['fixed_capped_mean'],float(np.minimum(b,cap).mean()),abs_tol=1e-16,rel_tol=1e-14)
                or any(y['covered_count']<x['covered_count'] or y['fixed_capped_mean']>x['fixed_capped_mean'] for x,y in zip(group,group[1:]))):
            raise ValueError('Prefix/Table2/raw consistency failed: '+method)
        for x in sorted(set([0.,theta,.2,c['theta_old']]+finite[finite<=.2].tolist())):
            count=int((b<=x).sum());ecdf.append(dict(method=method,k=20,theta=x,covered_count=count,N_base=n,coverage=count/n))
        table.append({**tail,'status':'ACCEPTED','scope':method_meta[method]})
        checks.append(dict(method=method,K20_count=tail['covered_count'],ecdf_theta010_count=int((b<=theta).sum()),table2_count=tail['covered_count']))
    for method in ['GCFExplainer','GlobalGCE','ComRecGC']:
        table.append(dict(method=method,k=20,effective_k='PENDING',N_base=n,N_source=c['source_test'],theta=theta,cap=cap,
            covered_count='PENDING',coverage='PENDING',finite_count='PENDING',fixed_capped_mean='PENDING',conditional_median='PENDING',
            status='PENDING',scope='No accepted V5 complete experiment'))
    write(root/'source_csv/figure3_taste_theta010.csv',prefix)
    write(root/'source_csv/figure4_taste_k20_raw_ecdf.csv',ecdf)
    write(root/'source_csv/table2_taste_k20_theta010.csv',table)
    dump(root/'audits/cross_figure_consistency.json',dict(status='PASS_FOR_ACCEPTED_ROWS_ONLY',completed_rows=len(best),total_rows=5,
        checks=checks,missing_rows_not_zero=True,source_ours=str(ours),source_cm=str(cm),raw_threshold_before_capping=True))

def plot(root):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42})
    source=root/'source_csv';out=root/'figures';out.mkdir(exist_ok=True)
    rows=read(source/'figure3_taste_theta010.csv');curves=read(source/'figure4_taste_k20_raw_ecdf.csv');table=read(source/'table2_taste_k20_theta010.csv')
    methods=list(dict.fromkeys(r['method'] for r in rows));style={'Ours':('black','s'),'CM-CReM-Global':('#008000','*')}
    fig,axs=plt.subplots(1,2,figsize=(8.6,2.85));fig.subplots_adjust(bottom=.28,top=.86,wspace=.30)
    for method in methods:
        r=[x for x in rows if x['method']==method];k=[int(x['k']) for x in r];color,marker=style[method]
        for ax,key,scale in [(axs[0],'coverage',100),(axs[1],'fixed_capped_mean',1)]:
            y=[float(x[key])*scale for x in r]
            ax.plot(k,y,color=color,lw=1.2,label=method)
            ix=[i for i,v in enumerate(k) if v%3==0 and v!=20]
            ax.plot(np.array(k)[ix],np.array(y)[ix],ls='none',marker=marker,ms=4,color=color)
            ax.set_xlim(1,20);ax.set_xticks([1,3,6,9,12,15,18,20]);ax.set_xlabel('Size (k)');ax.grid(alpha=.45,lw=.6)
    ymax=max(10,math.ceil(max(float(x['coverage'])*100 for x in rows)/10)*10)
    axs[0].set_ylim(0,ymax);axs[0].set_yticks(np.linspace(0,ymax,5));axs[0].set_ylabel('Coverage (%)')
    axs[1].set_ylim(0,.035);axs[1].set_yticks([0,.007,.014,.021,.028,.035]);axs[1].set_ylabel('Capped mean cost')
    fig.suptitle(f'TasteMolNet — theta = 0.1 — PARTIAL {len(methods)}/5',y=.99)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    for ext in ['png','pdf']:fig.savefig(out/f'figure3_taste_theta010.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,2.85));fig.subplots_adjust(bottom=.27,top=.86)
    for method in methods:
        r=[x for x in curves if x['method']==method];color,marker=style[method]
        x=np.array([float(a['theta']) for a in r]);y=np.array([float(a['coverage'])*100 for a in r])
        ax.step(x,y,where='post',color=color,lw=1.2,label=method)
        mx=np.array([.025,.05,.075,.1,.125,.15,.175]);mi=np.searchsorted(x,mx,side='right')-1
        ax.plot(mx,y[mi],ls='none',marker=marker,ms=4,color=color)
    ax.axvline(.1,color='.5',lw=.8,ls='--');ax.set_xlim(0,.2)
    ymax=max(10,math.ceil(max(float(x['coverage'])*100 for x in curves)/10)*10)
    ax.set_ylim(0,ymax);ax.set_yticks(np.linspace(0,ymax,5));ax.set_xticks([0,.05,.1,.15,.2])
    ax.grid(alpha=.45,lw=.6);ax.set_xlabel('Distance threshold (theta)');ax.set_ylabel('Coverage (%)')
    ax.set_title(f'TasteMolNet — K = 20 — exact ECDF — PARTIAL {len(methods)}/5')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    for ext in ['png','pdf']:fig.savefig(out/f'figure4_taste_k20.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.4,2.55));ax.axis('off')
    cells=[]
    for r in table:
        ok=r['status']=='ACCEPTED'
        cells.append([r['method'],r['effective_k'],f"{r['covered_count']}/468" if ok else 'PENDING',
            f"{float(r['coverage'])*100:.2f}" if ok else 'PENDING',r['finite_count'],
            f"{float(r['fixed_capped_mean']):.8f}" if ok else 'PENDING',
            f"{float(r['conditional_median']):.8f}" if ok and r['conditional_median']!='N/A' else r['conditional_median']])
    t=ax.table(cellText=cells,colLabels=['Method','Eff. K','Covered','Cov. (%)','Finite','Capped mean','Cond. median'],
        cellLoc='center',loc='center',colWidths=[.23,.07,.14,.12,.10,.17,.17])
    t.auto_set_font_size(False);t.set_fontsize(8.5);t.scale(1,1.6)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('.75');cell.set_linewidth(.5)
        if i==0:cell.set_facecolor('#eeeeee')
    ax.set_title(f'TasteMolNet — K20, theta 0.1 — fixed base 468 (source 285) — PARTIAL {len(methods)}/5',pad=12)
    fig.text(.5,.02,'Cap = 0.03416003659645076; conditional median over all finite valid recourses. Native operations differ.',ha='center',fontsize=8)
    for ext in ['png','pdf']:fig.savefig(out/f'table2_taste_k20_theta010.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--ours-release');p.add_argument('--cm-root');p.add_argument('--replot-only',action='store_true')
    p.add_argument('--config',help='Site config for Slurm interface compatibility; offline CSV only')
    a=p.parse_args();root=Path(a.root)
    if not a.replot_only:assemble(root,Path(a.ours_release),Path(a.cm_root) if a.cm_root else None)
    plot(root)
if __name__=='__main__':main()
