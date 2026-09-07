#!/usr/bin/env python3
"""Versioned three-control figures; explicit provisional scientific scope."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.bace_reach_result_display import checked_control_rows
from src.eval.ecdf_display import staircase, simplify


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--thresholds', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    if not args.config.is_file() or args.output_root.exists():
        p.error('Existing config and fresh output root required')
    audit = json.loads((args.source_root/'final_audit.json').read_text())
    if audit['state'] != 'EXECUTION_VALID' or audit['test_selected_variant'] or audit['test_campaigns'] != 1:
        raise ValueError('REQUIRE_COMPLETED_ONE_PREDECLARED_TEST')
    threshold = json.loads(args.thresholds.read_text())
    theta, cap = threshold['theta_star'], threshold['cost_cap']
    names = ['old_pool_old_selector', 'new_pool_old_selector', 'new_pool_reach_first']
    if set(audit['three_controls_evaluated']) != set(names):
        raise ValueError('THREE_CONTROLS_REQUIRED_NO_TEST_SELECTION')
    records = {name: checked_control_rows(json.loads((args.source_root/name/'explanation_metrics.json').read_text()),
        parent_count=audit['test_parent_count'], theta=theta, cap=cap) for name in names}
    args.output_root.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    colors = ['#777777', '#247b75', '#a12132']
    labels = ['Old pool / old selector', 'New pool / old selector', 'Reach-v2 (predeclared)']
    prefix, exact_rows, display_rows, display_audits, table = [], [], [], [], []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, label, color in zip(names, labels, colors):
        rows = records[name]
        ax.plot(range(1, 21), [r['covered_theta']/r['N'] for r in rows], label=label, color=color)
        prefix.extend(dict(control=name, **{k:v for k,v in r.items() if k != 'ecdf'}) for r in rows)
        table.extend(dict(control=name, **{k:v for k,v in rows[i].items() if k != 'ecdf'}) for i in (9,19))
    ax.set(xlabel='Nested rule budget K', ylabel='Coverage at original theta*', ylim=(0,1), xticks=range(1,21,2))
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('BACE Reach-v2: one descriptive test (previously seen benchmark)')
    fig.text(.5,.015,'Numerical export verified; independent scientific acceptance / main publication pending.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ('pdf','png'): fig.savefig(args.output_root/f'figure3_reach_v2_provisional.{ext}',dpi=180)
    plt.close(fig)
    for mode in ('exact','display'):
        fig, axes = plt.subplots(1,2,figsize=(11.8,4.8))
        for k, ax in zip((10,20),axes):
            for name,label,color in zip(names,labels,colors):
                points = staircase(records[name][k-1]['ecdf'], keys=(theta,cap))
                shown, proof = simplify(points, keys=(theta,cap))
                chosen = points if mode=='exact' else shown
                ax.plot([p.x for p in chosen],[p.y for p in chosen],label=label,color=color)
                if mode=='exact':
                    exact_rows.extend(dict(control=name,k=k,threshold=p.x,coverage=p.y) for p in points)
                    display_rows.extend(dict(control=name,k=k,threshold=p.x,coverage=p.y) for p in shown)
                    display_audits.append(dict(control=name,k=k,**proof))
            ax.set(title=f'K={k}',xlabel='Uncapped strict-flip WNode distance',ylim=(0,1))
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
            for x in (theta,cap): ax.axvline(x,ls=':',lw=.7,color='#555555')
        axes[0].set_ylabel('Coverage / unchanged 141-parent cohort')
        axes[1].legend(fontsize=8)
        fig.suptitle(f'BACE Reach-v2 {mode} ECDF — provisional science publication')
        fig.text(.5,.015,'All three predeclared controls shown; display error ≤0.5 pp. Statistics use exact saved distances.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.045,1,.95))
        for ext in ('pdf','png'):fig.savefig(args.output_root/f'figure4_reach_v2_{mode}_provisional.{ext}',dpi=180)
        plt.close(fig)
    for filename, rows in [('figure3_source.csv',prefix),('figure4_exact.csv',exact_rows),
                           ('figure4_display.csv',display_rows),('table2_k10_k20.csv',table)]:
        write_csv(args.output_root/filename,rows)
    fig, ax = plt.subplots(figsize=(12,4.2));ax.axis('off')
    body = [[labels[names.index(r['control'])],r['k'],r['N'],r['finite_reach'],
             f"{100*r['covered_theta']/r['N']:.2f}",f"{100*r['covered_cap']/r['N']:.2f}",
             'N/A' if r['conditional_median'] is None else f"{r['conditional_median']:.6f}",
             f"{r['capped_mean']:.6f}"] for r in table]
    tab=ax.table(cellText=body,colLabels=['Frozen control','K','N','Reach count','Cov theta* %',
        'Cov cap %','Conditional median','Capped mean'],cellLoc='center',loc='center',
        colWidths=[.25,.045,.045,.095,.13,.12,.16,.14])
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.65)
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#e7ebef');cell.set_text_props(fontweight='bold')
    fig.suptitle('BACE Reach-v2 Table2: all predeclared controls, unchanged costs')
    fig.text(.5,.025,'Descriptive test after adaptive development; numerical export verified.\n'
        'Independent science acceptance and main-cell supersession pending; not a final four-dataset table.',
        ha='center',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.18,left=.02,right=.98)
    for ext in ('pdf','png'):fig.savefig(args.output_root/f'table2_reach_v2_provisional.{ext}',dpi=180)
    plt.close(fig)
    (args.output_root/'display_audit.json').write_text(json.dumps(dict(
        state='NUMERICAL_EXPORT_VERIFIED_NOT_INDEPENDENT_SCIENCE_ACCEPTANCE',
        source_root=str(args.source_root),final_binding_sha256=audit['final_binding_sha256'],
        thresholds=threshold,controls=names,main_matrix_written=False,
        scientific_publication_pending=True,display_metrics_used=False,display_audits=display_audits),indent=2)+'\n')


if __name__ == '__main__': main()
