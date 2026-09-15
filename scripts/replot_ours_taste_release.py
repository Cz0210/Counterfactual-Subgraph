#!/usr/bin/env python3
"""Offline plotting of frozen Ours–Taste CSVs; no model/data/registry access."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--formats',nargs='+',choices=['png','pdf'],default=['png','pdf'])
    p.add_argument('--f3-xlim',nargs=2,type=float,default=[1,20])
    p.add_argument('--f3-coverage-ylim',nargs=2,type=float,default=[0,12])
    p.add_argument('--f3-cost-ylim',nargs=2,type=float,default=[.028,.035])
    p.add_argument('--f4-xlim',nargs=2,type=float,default=[0,.03416003659645076])
    p.add_argument('--f4-ylim',nargs=2,type=float,default=[0,35])
    p.add_argument('--width-in',type=float,default=6.8)
    p.add_argument('--figure3-height-in',type=float,default=4.8)
    p.add_argument('--figure4-height-in',type=float,default=2.8)
    p.add_argument('--dpi',type=int,default=220)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((a.source_dir/'figure3_taste_ours_v0_v1.csv').open()))
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'pdf.fonttype':42})
    colors={'A':'black','B':'green','C':'#b000b0','D':'#cc2020'}
    labels={'A':'A: old pool / original selector','B':'B: old pool / new selector',
            'C':'C: expanded pool / original selector','D':'D: expanded pool / new selector'}
    def save(fig,name):
        for ext in a.formats:fig.savefig(a.out_dir/f'{name}.{ext}',dpi=a.dpi)
        plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(a.width_in,a.figure3_height_in),constrained_layout=True)
    for v,color in colors.items():
        r=[x for x in rows if x['variant']==v]
        for ax,key,scale in [(axs[0],'coverage',100),(axs[1],'capped_mean',1)]:
            ax.plot([int(x['k']) for x in r],[scale*float(x[key]) for x in r],color=color,lw=1,label=labels[v])
    axs[0].set_title('TasteMolNet / frozen GINE / fixed 468 parents')
    axs[0].set_ylabel('Coverage (%)');axs[0].set_ylim(*a.f3_coverage_ylim)
    axs[0].legend(fontsize=7,ncol=2)
    axs[1].set_ylabel('Fixed-capped mean cost');axs[1].set_xlabel('Size (K)');axs[1].set_ylim(*a.f3_cost_ylim)
    for ax in axs:ax.set_xlim(*a.f3_xlim);ax.grid(alpha=.3)
    save(fig,'figure3_taste_ours_v0_v1')
    for k in (10,20):
        ecdf=list(csv.DictReader((a.source_dir/f'figure4_taste_ours_v0_v1_k{k}.csv').open()))
        fig,ax=plt.subplots(figsize=(a.width_in,a.figure4_height_in),constrained_layout=True)
        for v,color in colors.items():
            r=[x for x in ecdf if x['variant']==v]
            ax.step([float(x['threshold']) for x in r],[100*float(x['coverage']) for x in r],
                    where='post',color=color,lw=1,label=v)
        ax.set_xlim(*a.f4_xlim);ax.set_ylim(*a.f4_ylim);ax.grid(alpha=.3);ax.legend(ncol=4)
        ax.set_xlabel('WNode distance threshold');ax.set_ylabel('Coverage (%)')
        ax.set_title(f'TasteMolNet / frozen GINE / K{k}')
        save(fig,f'figure4_taste_ours_v0_v1_k{k}')
    table_rows=[r for r in rows if int(r['k'])==20]
    fig,ax=plt.subplots(figsize=(8,2.2));ax.axis('off')
    table=ax.table(cellText=[[r['variant'],f"{r['covered_count']}/{r['parent_count']}",
                      f"{100*float(r['coverage']):.4f}",f"{r['finite_count']}/{r['parent_count']}",
                      f"{float(r['capped_mean']):.9f}"] for r in table_rows],
                   colLabels=['Variant','Covered K20','Coverage (%)','Finite recourse','Capped mean'],loc='center')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.4)
    ax.set_title('TasteMolNet / frozen GINE / K20',fontsize=10)
    save(fig,'table2_taste_ours_v0_v1_k20')


if __name__=='__main__':main()
