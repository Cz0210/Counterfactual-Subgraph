#!/usr/bin/env python3
"""Offline plots from this experiment's exact source CSVs, not log summaries."""
import argparse
import csv
from pathlib import Path

def render(source: Path, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=('Ours','GlobalGCE','GCFExplainer','ComRecGC')
    colors=dict(zip(methods,('#d64b40','#6c4b9c','#2878b5','#3a923a')))
    def read(name):
        with (source/name).open(newline='') as f: return list(csv.DictReader(f))
    table=read('bace_gin_fixed141_table2.csv')
    ready={r['method'] for r in table if r.get('state')=='EVALUATED'}
    if not ready: raise ValueError('No evaluated methods: do not invent a plot')
    subtitle=('PARTIAL ' if len(ready)<4 else '')+'BACE · frozen GIN · fixed 141 base parents'
    output.mkdir(parents=True,exist_ok=True)
    for name,filename,xlabel,xkey in (
        ('figure3_bace_gin_fixed141','bace_gin_fixed141_figure3.csv','Rule budget K (at most)','K_requested'),
        ('figure4_bace_gin_fixed141','bace_gin_fixed141_figure4_exact_ecdf.csv','WNode threshold (exact ECDF, K=20)','threshold')):
        rows=read(filename)
        if 'figure4' in name: rows=[r for r in rows if int(r['K_requested'])==20]
        fig,ax=plt.subplots(figsize=(7.6,4.8),constrained_layout=True)
        peak=0.
        for method in methods:
            selected=sorted((r for r in rows if r['method']==method),key=lambda r:float(r[xkey]))
            if not selected: continue
            xs=[float(r[xkey]) for r in selected]; ys=[100*float(r['coverage']) for r in selected]
            peak=max(peak,max(ys))
            if 'figure4' in name: ax.step(xs,ys,where='post',label=method,color=colors[method])
            else: ax.plot(xs,ys,label=method,color=colors[method],marker='o',markersize=3)
        ax.set(xlabel=xlabel,ylabel='Coverage (%)',title=subtitle,ylim=(0,max(5,peak*1.08)))
        ax.grid(alpha=.2); ax.legend()
        if len(ready)<4:
            fig.text(.02,.005,'Not plotted: '+', '.join(m for m in methods if m not in ready)+' (PENDING / UNDER_REPAIR)',fontsize=8)
        for ext in ('png','pdf'): fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    display=[]
    for r in table:
        def value(key): return 'N/A' if r.get(key,'') in ('','None') else f'{float(r[key]):.6g}'
        display.append([r['method'],r.get('state','PENDING'),r.get('denominator','—'),
            r.get('K_effective','—'),value('coverage'),value('fixed_capped_mean'),value('conditional_median')])
    fig,ax=plt.subplots(figsize=(11,2.8),constrained_layout=True); ax.axis('off'); ax.set_title(subtitle+' · K=10')
    t=ax.table(cellText=display,colLabels=['Method','State','N','K effective','Coverage','Capped mean','Conditional median'],loc='center')
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1,1.5)
    fig.savefig(output/'table2_bace_gin_fixed141.pdf',bbox_inches='tight'); plt.close(fig)
    tex=['% POST_HOC_FIXED_POOL_CROSS_CLASSIFIER_COMPARISON; no end-to-end retraining claim',
         r'\begin{tabular}{lrrrr}',r'Method & N & Coverage@10 & Capped mean & Conditional median \\']
    for row in display: tex.append(' & '.join([row[0],row[2],row[4],row[5],row[6]])+r' \\')
    tex.append(r'\end{tabular}')
    (output/'table2_bace_gin_fixed141.tex').write_text('\n'.join(tex)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source-csv',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); render(args.source_csv.resolve(strict=True),args.output.resolve())
