#!/usr/bin/env python3
"""Offline raw-CSV renderer: no interpolation, model or selector access."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True,type=Path);p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--table-dir',required=True,type=Path)
    p.add_argument('--f3-xlim',nargs=2,type=float,default=[1,20])
    p.add_argument('--f3-coverage-ylim',nargs=2,type=float,default=[0,70])
    p.add_argument('--f3-cost-ylim',nargs=2,type=float,default=[.028,.035])
    p.add_argument('--f4-xlim',nargs=2,type=float,default=[0,.2]);p.add_argument('--f4-ylim',nargs=2,type=float,default=[0,70])
    p.add_argument('--dpi',type=int,default=240)
    a=p.parse_args();a.out_dir.mkdir(exist_ok=True,parents=True);a.table_dir.mkdir(exist_ok=True,parents=True)
    def read(name):return list(csv.DictReader((a.source_dir/name).open()))
    rows=read('figure3_taste_ours_theta010.csv');old=read('table2_taste_ours_k20_oldtheta.csv')
    cap=float(rows[0]['original_cost_cap']);source=100*int(rows[0]['N_source'])/int(rows[0]['N_base'])
    c=json.loads((a.source_dir.parent/'resolved_contract.json').read_text());thetaold=c['theta_old']
    colors={'R0':'black','R1':'#188125','R2':'#c31c27'};markers={'R0':'s','R1':'^','R2':'o','R3':'D'}
    if any(r['variant']=='R3' for r in rows):colors['R3']='#9b239e'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42})
    def save(fig,name,folder):
        for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=a.dpi,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.2,2.7),constrained_layout=True)
    for v,col in colors.items():
        r=[r for r in rows if r['variant']==v];x=[int(r['k_requested']) for r in r]
        for ax,key,scale in [(axs[0],'coverage_fraction',100),(axs[1],'fixed_capped_mean_cost',1)]:
            ax.plot(x,[float(row[key])*scale for row in r],color=col,lw=1,marker=markers[v],markevery=[2,5,8,11,14,17],ms=3.5,label=v)
    axs[0].axhline(source,color='.5',ls=':',lw=.8,label='Source ceiling')
    axs[0].set_ylabel('Coverage (%)');axs[0].set_ylim(*a.f3_coverage_ylim)
    axs[1].set_ylabel('Fixed-capped mean cost');axs[1].set_ylim(*a.f3_cost_ylim)
    for ax in axs:ax.set_xlim(*a.f3_xlim);ax.set_xlabel('Size (K)');ax.grid(alpha=.3);ax.legend(fontsize=8,ncol=2)
    fig.suptitle(r'TasteMolNet / frozen GINE / fixed 468 / $\theta=0.1$')
    save(fig,'figure3_taste_ours_theta010',a.out_dir)
    for k in (20,10):
        erows=read(f'figure4_taste_ours_k{k}.csv')
        fig,ax=plt.subplots(figsize=(7.5,2.9),constrained_layout=True)
        for v,col in colors.items():
            rr=[r for r in erows if r['variant']==v];x=np.array([float(r['theta']) for r in rr]);y=np.array([100*float(r['coverage_fraction']) for r in rr])
            ax.step(x,y,where='post',color=col,lw=1,label=v)
            mx=np.arange(.02,.2,.03);my=y[np.maximum(0,np.searchsorted(x,mx,side='right')-1)]
            ax.plot(mx,my,linestyle='none',marker=markers[v],ms=3.5,color=col)
        ax.axvline(.1,color='.35',ls='--',lw=.8,label=r'Primary $\theta=0.1$')
        ax.axvline(thetaold,color='.6',ls=':',lw=.8,label=f'Old θ={thetaold:.5f}')
        ax.axhline(source,color='.5',ls='-.',lw=.7,label='Source ceiling')
        ax.set_xlim(*a.f4_xlim);ax.set_ylim(*a.f4_ylim);ax.set_yticks(np.linspace(*a.f4_ylim,8));ax.grid(alpha=.3)
        ax.set_xlabel('Uncapped WNode distance threshold');ax.set_ylabel('Coverage (%)')
        ax.set_title(f'TasteMolNet / frozen GINE / fixed 468 / K{k} / exact ECDF')
        ax.legend(ncol=3,fontsize=7,loc='upper left')
        save(fig,f'figure4_taste_ours_k{k}',a.out_dir)
    tr=read('table2_taste_ours_k20_theta010.csv');fig,ax=plt.subplots(figsize=(10,2.3));ax.axis('off')
    table=ax.table(cellText=[[r['variant'],f"{r['num_covered']}/{r['N_base']}",f"{100*float(r['coverage_fraction']):.3f}%",
        r['finite_recourse_count'],f"{float(r['fixed_capped_mean_cost']):.9f}",f"{float(r['conditional_median_all_finite']):.9f}"] for r in tr],
        colLabels=['Variant','Covered','Coverage','Finite','Capped mean','Finite median'],loc='center')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6)
    ax.set_title('TasteMolNet / frozen GINE / K20 / θ=0.1 / fixed 468',fontsize=10)
    fig.text(.5,.09,f'Original cap={cap:.17g}; finite median is not θ-filtered. Post-hoc protocol revision.',ha='center',fontsize=8)
    save(fig,'table2_taste_ours_k20_theta010',a.table_dir)
    lines=[r'\begin{tabular}{lrrrrr}',r'Variant & Covered & Coverage (\%) & Finite & Capped mean & Finite median \\',r'\hline']
    lines.extend(f"{r['variant']} & {r['num_covered']}/{r['N_base']} & {100*float(r['coverage_fraction']):.3f} & {r['finite_recourse_count']} & {float(r['fixed_capped_mean_cost']):.9f} & {float(r['conditional_median_all_finite']):.9f} \\\\" for r in tr)
    lines.append(r'\end{tabular}')
    (a.table_dir/'table2_taste_ours_k20_theta010.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
