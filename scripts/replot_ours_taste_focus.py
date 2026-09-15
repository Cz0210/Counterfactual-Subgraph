#!/usr/bin/env python3
"""Offline exact-source diagnostic figures, clearly separating calibration/test."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact-root',type=Path,required=True)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    contract=json.loads((a.compact_root/'contract.json').read_text())
    old=list(csv.DictReader((a.analysis_root/'old_test_prefix_replayed.csv').open()))
    cal=list(csv.DictReader((a.analysis_root/'calibration_prefix.csv').open()))
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(8.4,4.7),constrained_layout=True)
    for col,groups,title in [(0,[('Published A',old,'black','-')],'TasteMolNet - published test'),
                              (1,[(v,[r for r in cal if r['variant']==v],c,ls) for v,c,ls in [('A','black','-'),('B','green','--')]],'TasteMolNet - calibration only')]:
        for label,rows,c,ls in groups:
            k=[int(r['k']) for r in rows]
            axs[0,col].plot(k,[100*float(r['coverage']) for r in rows],color=c,ls=ls,lw=1,label=label)
            axs[1,col].plot(k,[float(r['capped_mean']) for r in rows],color=c,ls=ls,lw=1)
        axs[0,col].set_title(title);axs[0,col].set_ylim(0,12);axs[0,col].set_yticks([0,3,6,9,12])
        axs[1,col].set_ylim(0.025,.035);axs[1,col].set_yticks([.025,.030,.035]);axs[1,col].set_xlabel('Size (K)')
        for ax in axs[:,col]:ax.set_xlim(1,20);ax.set_xticks([1,3,6,9,12,15,18,20]);ax.grid(alpha=.35)
        axs[0,col].legend(loc='lower right')
    axs[0,0].set_ylabel('Coverage (%)');axs[1,0].set_ylabel('Fixed-capped mean cost')
    fig.suptitle('P0 development diagnostic - new-pool / new-test results PENDING',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(a.out_dir/f'figure3_ours_taste_diagnostic.{ext}',dpi=200)
    plt.close(fig)
    test=np.load(a.compact_root/'test.npz',allow_pickle=False)['distances']
    fig,axs=plt.subplots(1,2,figsize=(8.4,2.7),constrained_layout=True);ecdf=[]
    for ax,k in zip(axs,[10,20]):
        best=np.min(test[:,:k],axis=1);finite=best[np.isfinite(best)]
        x=np.unique(np.r_[0.,finite,contract['theta_star'],contract['cost_cap']]); y=np.array([(best<=v).mean()*100 for v in x])
        ax.step(x,y,where='post',c='black',lw=1)
        ax.axvline(contract['theta_star'],ls=':',c='gray',lw=.8)
        ax.set_title(f'TasteMolNet - published test K{k}')
        ax.set_xlim(0,contract['cost_cap']);ax.set_ylim(0,40);ax.set_yticks([0,10,20,30,40])
        ax.set_xlabel('WNode distance threshold');ax.grid(alpha=.35)
        ecdf.extend({'scope':'PUBLISHED_TEST_A','k':k,'threshold':v,'coverage':w/100} for v,w in zip(x,y))
    axs[0].set_ylabel('Coverage (%)')
    for ext in ('png','pdf'):fig.savefig(a.out_dir/f'figure4_ours_taste_old_K10_K20.{ext}',dpi=200)
    plt.close(fig)
    with (a.out_dir/'figure4_exact_source.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ecdf[0]));w.writeheader();w.writerows(ecdf)
    table=[{'scope':'PUBLISHED_TEST_A',**old[-1]},*({'scope':f'CALIBRATION_{r["variant"]}',**{k:v for k,v in r.items() if k!='variant'}} for r in cal if r['k']=='20')]
    with (a.out_dir/'table2_k20_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,ax=plt.subplots(figsize=(8.4,1.9));ax.axis('off')
    cell=[[r['scope'],r['covered_count']+'/'+r['parent_count'],r['finite_count']+'/'+r['parent_count'],f'{float(r["capped_mean"]):.8f}'] for r in table]
    tab=ax.table(cellText=cell,colLabels=['Scope','Coverage@20','Finite recourse@20','Capped cost@20'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.5)
    ax.set_title('Distinct evaluation cohorts - not interchangeable paper rows',fontsize=10,pad=3)
    fig.savefig(a.out_dir/'table2_k20_diagnostic.pdf',bbox_inches='tight');plt.close(fig)
    print(a.out_dir)

if __name__=='__main__':main()
